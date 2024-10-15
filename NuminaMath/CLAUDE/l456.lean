import Mathlib

namespace NUMINAMATH_CALUDE_book_arrangement_proof_l456_45650

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def arrange_books (total : ℕ) (math_copies : ℕ) (physics_copies : ℕ) : ℕ :=
  factorial total / (factorial math_copies * factorial physics_copies)

theorem book_arrangement_proof :
  arrange_books 7 3 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l456_45650


namespace NUMINAMATH_CALUDE_water_fountain_length_l456_45615

/-- Given the conditions for building water fountains, prove the length of the fountain built by 20 men in 7 days -/
theorem water_fountain_length 
  (men1 : ℕ) (days1 : ℕ) (men2 : ℕ) (days2 : ℕ) (length2 : ℝ)
  (h1 : men1 = 20)
  (h2 : days1 = 7)
  (h3 : men2 = 35)
  (h4 : days2 = 3)
  (h5 : length2 = 42)
  (h_prop : ∀ (m d : ℕ) (l : ℝ), (m * d : ℝ) / (men2 * days2 : ℝ) = l / length2) :
  let length1 := (men1 * days1 : ℝ) * length2 / (men2 * days2 : ℝ)
  length1 = 56 := by
  sorry

end NUMINAMATH_CALUDE_water_fountain_length_l456_45615


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l456_45648

/-- Given points A, B, C, D, and E on a line in that order, with specified distances between them,
    this theorem states that the minimum sum of squared distances from these points to any point P
    on the same line is 66. -/
theorem min_sum_squared_distances (A B C D E P : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_AB : B - A = 1)
  (h_BC : C - B = 2)
  (h_CD : D - C = 3)
  (h_DE : E - D = 4)
  (h_P : A ≤ P ∧ P ≤ E) :
  66 ≤ (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l456_45648


namespace NUMINAMATH_CALUDE_solve_system_l456_45641

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l456_45641


namespace NUMINAMATH_CALUDE_fraction_denominator_problem_l456_45633

theorem fraction_denominator_problem (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_problem_l456_45633


namespace NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l456_45635

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Calculates the new ratio after adding new boarders -/
def new_ratio (initial : Ratio) (new_boarders : ℕ) : Ratio :=
  { boarders := initial.boarders + new_boarders,
    day_students := initial.day_students }

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.boarders r.day_students
  { boarders := r.boarders / gcd,
    day_students := r.day_students / gcd }

theorem new_ratio_is_one_to_two :
  let initial_ratio : Ratio := { boarders := 330, day_students := 792 }
  let new_boarders : ℕ := 66
  let final_ratio := simplify_ratio (new_ratio initial_ratio new_boarders)
  final_ratio.boarders = 1 ∧ final_ratio.day_students = 2 := by
  sorry


end NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l456_45635


namespace NUMINAMATH_CALUDE_mountain_climb_speeds_l456_45680

theorem mountain_climb_speeds (V₁ V₂ V k m n : ℝ) 
  (hpos : V₁ > 0 ∧ V₂ > 0 ∧ V > 0 ∧ k > 0 ∧ m > 0 ∧ n > 0)
  (hV₂ : V₂ = k * V₁)
  (hVm : V = m * V₁)
  (hVn : V = n * V₂) : 
  m = 2 * k / (1 + k) ∧ m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mountain_climb_speeds_l456_45680


namespace NUMINAMATH_CALUDE_leo_statement_true_only_on_tuesday_l456_45692

-- Define the days of the week
inductive Day : Type
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

-- Define Leo's lying pattern
def lies_on_day (d : Day) : Prop :=
  match d with
  | Day.monday => True
  | Day.tuesday => True
  | Day.wednesday => True
  | _ => False

-- Define the 'yesterday' and 'tomorrow' functions
def yesterday (d : Day) : Day :=
  match d with
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday
  | Day.sunday => Day.saturday

def tomorrow (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

-- Define Leo's statement
def leo_statement (d : Day) : Prop :=
  lies_on_day (yesterday d) ∧ lies_on_day (tomorrow d)

-- Theorem: Leo's statement is true only on Tuesday
theorem leo_statement_true_only_on_tuesday :
  ∀ (d : Day), leo_statement d ↔ d = Day.tuesday :=
by sorry

end NUMINAMATH_CALUDE_leo_statement_true_only_on_tuesday_l456_45692


namespace NUMINAMATH_CALUDE_girls_in_math_class_l456_45628

theorem girls_in_math_class
  (boy_girl_ratio : ℚ)
  (math_science_ratio : ℚ)
  (science_lit_ratio : ℚ)
  (total_students : ℕ)
  (h1 : boy_girl_ratio = 5 / 8)
  (h2 : math_science_ratio = 7 / 4)
  (h3 : science_lit_ratio = 3 / 5)
  (h4 : total_students = 720) :
  ∃ (girls_math : ℕ), girls_math = 176 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_math_class_l456_45628


namespace NUMINAMATH_CALUDE_max_value_implies_a_l456_45660

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 4, f a x = 3) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l456_45660


namespace NUMINAMATH_CALUDE_trivia_game_score_l456_45606

/-- Calculates the final score in a trivia game given the specified conditions -/
def calculateFinalScore (firstHalfCorrect secondHalfCorrect : ℕ) 
  (firstHalfOddPoints firstHalfEvenPoints : ℕ)
  (secondHalfOddPoints secondHalfEvenPoints : ℕ)
  (bonusPoints : ℕ) : ℕ :=
  let firstHalfOdd := firstHalfCorrect / 2 + firstHalfCorrect % 2
  let firstHalfEven := firstHalfCorrect / 2
  let secondHalfOdd := secondHalfCorrect / 2 + secondHalfCorrect % 2
  let secondHalfEven := secondHalfCorrect / 2
  let firstHalfMultiplesOf3 := (firstHalfCorrect + 2) / 3
  let secondHalfMultiplesOf3 := (secondHalfCorrect + 1) / 3
  (firstHalfOdd * firstHalfOddPoints + firstHalfEven * firstHalfEvenPoints +
   secondHalfOdd * secondHalfOddPoints + secondHalfEven * secondHalfEvenPoints +
   (firstHalfMultiplesOf3 + secondHalfMultiplesOf3) * bonusPoints)

theorem trivia_game_score :
  calculateFinalScore 10 12 2 4 3 5 5 = 113 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_score_l456_45606


namespace NUMINAMATH_CALUDE_binomial_13_8_l456_45677

theorem binomial_13_8 (h1 : Nat.choose 14 7 = 3432) 
                      (h2 : Nat.choose 14 8 = 3003) 
                      (h3 : Nat.choose 12 7 = 792) : 
  Nat.choose 13 8 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_8_l456_45677


namespace NUMINAMATH_CALUDE_difference_of_squares_l456_45666

theorem difference_of_squares : 72^2 - 54^2 = 2268 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l456_45666


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_four_l456_45697

theorem factorial_fraction_equals_four :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_four_l456_45697


namespace NUMINAMATH_CALUDE_multiples_difference_cubed_zero_l456_45638

theorem multiples_difference_cubed_zero : 
  let a := (Finset.filter (fun x => x % 12 = 0 ∧ x > 0) (Finset.range 60)).card
  let b := (Finset.filter (fun x => x % 4 = 0 ∧ x % 3 = 0 ∧ x > 0) (Finset.range 60)).card
  (a - b)^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_multiples_difference_cubed_zero_l456_45638


namespace NUMINAMATH_CALUDE_coconut_cost_is_fifty_cents_l456_45640

/-- Represents the cost per coconut on Rohan's farm -/
def coconut_cost (farm_size : ℕ) (trees_per_sqm : ℕ) (coconuts_per_tree : ℕ) 
  (harvest_interval : ℕ) (months : ℕ) (total_earnings : ℚ) : ℚ :=
  let total_trees := farm_size * trees_per_sqm
  let total_coconuts := total_trees * coconuts_per_tree
  let harvests := months / harvest_interval
  let total_harvested := total_coconuts * harvests
  total_earnings / total_harvested

/-- Proves that the cost per coconut on Rohan's farm is $0.50 -/
theorem coconut_cost_is_fifty_cents :
  coconut_cost 20 2 6 3 6 240 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_coconut_cost_is_fifty_cents_l456_45640


namespace NUMINAMATH_CALUDE_probability_four_different_socks_l456_45657

/-- The number of pairs of socks in the bag -/
def num_pairs : ℕ := 5

/-- The number of socks drawn in each sample -/
def sample_size : ℕ := 4

/-- The probability of drawing 4 different socks in the first draw -/
def p1 : ℚ := 8 / 21

/-- The probability of drawing exactly one pair and two different socks in the first draw -/
def p2 : ℚ := 4 / 7

/-- The probability of drawing 2 different socks in the next draw, given that we already have 3 different socks and one pair discarded -/
def p3 : ℚ := 4 / 15

/-- The theorem stating the probability of ending up with 4 socks of different colors -/
theorem probability_four_different_socks : 
  p1 + p2 * p3 = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_four_different_socks_l456_45657


namespace NUMINAMATH_CALUDE_marinara_stains_l456_45658

theorem marinara_stains (grass_time : ℕ) (marinara_time : ℕ) (grass_count : ℕ) (total_time : ℕ) :
  grass_time = 4 →
  marinara_time = 7 →
  grass_count = 3 →
  total_time = 19 →
  (total_time - grass_time * grass_count) / marinara_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_marinara_stains_l456_45658


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l456_45694

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 → x = 5 * y → |x - y| = 60 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l456_45694


namespace NUMINAMATH_CALUDE_calculate_expression_l456_45613

theorem calculate_expression : 3000 * (3000^2999) * 2 = 2 * 3000^3000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l456_45613


namespace NUMINAMATH_CALUDE_sally_quarters_now_l456_45685

/-- The number of quarters Sally had initially -/
def initial_quarters : ℕ := 760

/-- The number of quarters Sally spent -/
def spent_quarters : ℕ := 418

/-- Theorem: Sally has 342 quarters now -/
theorem sally_quarters_now : initial_quarters - spent_quarters = 342 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_now_l456_45685


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l456_45629

/-- The cost of stationery items -/
structure StationeryCost where
  pencil : ℝ  -- Cost of one pencil
  pen : ℝ     -- Cost of one pen
  eraser : ℝ  -- Cost of one eraser

/-- Given conditions on stationery costs -/
def stationery_conditions (c : StationeryCost) : Prop :=
  4 * c.pencil + 3 * c.pen + c.eraser = 5.40 ∧
  2 * c.pencil + 2 * c.pen + 2 * c.eraser = 4.60

/-- Theorem stating the cost of 1 pencil, 2 pens, and 3 erasers -/
theorem stationery_cost_theorem (c : StationeryCost) 
  (h : stationery_conditions c) : 
  c.pencil + 2 * c.pen + 3 * c.eraser = 4.60 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l456_45629


namespace NUMINAMATH_CALUDE_shortest_path_length_on_tetrahedron_l456_45686

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ

/-- A path on the surface of a regular tetrahedron -/
structure SurfacePath (t : RegularTetrahedron) where
  length : ℝ
  start_vertex : Fin 4
  end_midpoint : Fin 6

/-- The shortest path on the surface of a regular tetrahedron -/
def shortest_path (t : RegularTetrahedron) : SurfacePath t :=
  sorry

theorem shortest_path_length_on_tetrahedron :
  let t : RegularTetrahedron := ⟨2⟩
  (shortest_path t).length = 3 := by sorry

end NUMINAMATH_CALUDE_shortest_path_length_on_tetrahedron_l456_45686


namespace NUMINAMATH_CALUDE_water_fraction_after_three_replacements_l456_45646

/-- Represents the fraction of water remaining in a radiator after repeated partial replacements with antifreeze. -/
def waterFractionAfterReplacements (initialVolume : ℚ) (replacementVolume : ℚ) (numReplacements : ℕ) : ℚ :=
  ((initialVolume - replacementVolume) / initialVolume) ^ numReplacements

/-- Theorem stating that after three replacements in a 20-quart radiator, 
    the fraction of water remaining is 27/64. -/
theorem water_fraction_after_three_replacements :
  waterFractionAfterReplacements 20 5 3 = 27 / 64 := by
  sorry

#eval waterFractionAfterReplacements 20 5 3

end NUMINAMATH_CALUDE_water_fraction_after_three_replacements_l456_45646


namespace NUMINAMATH_CALUDE_smallest_area_squared_l456_45678

/-- A regular hexagon ABCDEF with side length 10 inscribed in a circle ω -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 10)

/-- Points X, Y, Z on minor arcs AB, CD, EF respectively -/
structure TriangleXYZ (h : RegularHexagon) :=
  (X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (Z : ℝ × ℝ)
  (X_on_AB : True)  -- Placeholder for the condition that X is on minor arc AB
  (Y_on_CD : True)  -- Placeholder for the condition that Y is on minor arc CD
  (Z_on_EF : True)  -- Placeholder for the condition that Z is on minor arc EF

/-- The area of triangle XYZ -/
def triangle_area (h : RegularHexagon) (t : TriangleXYZ h) : ℝ :=
  sorry  -- Definition of triangle area

/-- The theorem stating the smallest possible area squared -/
theorem smallest_area_squared (h : RegularHexagon) :
  ∃ (t : TriangleXYZ h), ∀ (t' : TriangleXYZ h), (triangle_area h t)^2 ≤ (triangle_area h t')^2 ∧ (triangle_area h t)^2 = 7500 :=
sorry

end NUMINAMATH_CALUDE_smallest_area_squared_l456_45678


namespace NUMINAMATH_CALUDE_horner_method_v3_equals_20_l456_45610

def f (x : ℝ) : ℝ := 2*x^5 + 3*x^3 - 2*x^2 + x - 1

def horner_v3 (a : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x
  let v2 := (v1 + 3) * x - 2
  (v2 * x + 1) * x - 1

theorem horner_method_v3_equals_20 :
  horner_v3 f 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_equals_20_l456_45610


namespace NUMINAMATH_CALUDE_f_minimum_value_l456_45656

def f (x : ℝ) : ℝ := |x - 1| + |x - 2| - |x - 3|

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ -1) ∧ (∃ x : ℝ, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l456_45656


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l456_45675

/-- Represents the profit maximization problem for a product -/
structure ProfitProblem where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialQuantity : ℝ
  priceElasticity : ℝ

/-- Calculates the profit for a given price increase -/
def profit (problem : ProfitProblem) (priceIncrease : ℝ) : ℝ :=
  let newPrice := problem.initialSellingPrice + priceIncrease
  let newQuantity := problem.initialQuantity - problem.priceElasticity * priceIncrease
  (newPrice - problem.initialPurchasePrice) * newQuantity

/-- Theorem stating that the profit-maximizing price is 95 yuan -/
theorem profit_maximizing_price (problem : ProfitProblem) 
  (h1 : problem.initialPurchasePrice = 80)
  (h2 : problem.initialSellingPrice = 90)
  (h3 : problem.initialQuantity = 400)
  (h4 : problem.priceElasticity = 20) :
  ∃ (maxProfit : ℝ), ∀ (price : ℝ), 
    profit problem (price - problem.initialSellingPrice) ≤ maxProfit ∧
    profit problem (95 - problem.initialSellingPrice) = maxProfit :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l456_45675


namespace NUMINAMATH_CALUDE_first_grade_enrollment_l456_45626

theorem first_grade_enrollment :
  ∃ (n : ℕ),
    200 ≤ n ∧ n ≤ 300 ∧
    ∃ (r : ℕ), n = 25 * r + 10 ∧
    ∃ (l : ℕ), n = 30 * l - 15 ∧
    n = 285 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_enrollment_l456_45626


namespace NUMINAMATH_CALUDE_base_nine_proof_l456_45645

theorem base_nine_proof (b : ℕ) : 
  (∃ (n : ℕ), n = 144 ∧ 
    n = (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)) →
  b = 9 :=
by sorry

end NUMINAMATH_CALUDE_base_nine_proof_l456_45645


namespace NUMINAMATH_CALUDE_sister_glue_sticks_l456_45665

theorem sister_glue_sticks (total : ℕ) (emily : ℕ) (sister : ℕ) : 
  total = 13 → emily = 6 → sister = total - emily → sister = 7 := by
  sorry

end NUMINAMATH_CALUDE_sister_glue_sticks_l456_45665


namespace NUMINAMATH_CALUDE_largest_selected_is_57_l456_45688

/-- Represents the systematic sampling of students. -/
structure StudentSampling where
  total_students : Nat
  first_selected : Nat
  second_selected : Nat

/-- Calculates the sample interval based on the first two selected numbers. -/
def sample_interval (s : StudentSampling) : Nat :=
  s.second_selected - s.first_selected

/-- Calculates the number of selected students. -/
def num_selected (s : StudentSampling) : Nat :=
  s.total_students / sample_interval s

/-- Calculates the largest selected number. -/
def largest_selected (s : StudentSampling) : Nat :=
  s.first_selected + (sample_interval s) * (num_selected s - 1)

/-- Theorem stating that the largest selected number is 57 for the given conditions. -/
theorem largest_selected_is_57 (s : StudentSampling) 
    (h1 : s.total_students = 60)
    (h2 : s.first_selected = 3)
    (h3 : s.second_selected = 9) : 
  largest_selected s = 57 := by
  sorry

#eval largest_selected { total_students := 60, first_selected := 3, second_selected := 9 }

end NUMINAMATH_CALUDE_largest_selected_is_57_l456_45688


namespace NUMINAMATH_CALUDE_total_match_sequences_l456_45605

/-- Represents the number of players in each team -/
def n : ℕ := 7

/-- Calculates the number of possible match sequences for one team winning -/
def sequences_for_one_team_winning : ℕ := Nat.choose (2 * n - 1) (n - 1)

/-- Theorem stating the total number of possible match sequences -/
theorem total_match_sequences : 2 * sequences_for_one_team_winning = 3432 := by
  sorry

end NUMINAMATH_CALUDE_total_match_sequences_l456_45605


namespace NUMINAMATH_CALUDE_replacement_stove_cost_l456_45624

/-- The cost of a replacement stove and wall repair, given specific conditions. -/
theorem replacement_stove_cost (stove_cost wall_cost : ℚ) : 
  wall_cost = (1 : ℚ) / 6 * stove_cost →
  stove_cost + wall_cost = 1400 →
  stove_cost = 1200 := by
sorry

end NUMINAMATH_CALUDE_replacement_stove_cost_l456_45624


namespace NUMINAMATH_CALUDE_total_pamphlets_printed_prove_total_pamphlets_l456_45661

/-- Calculates the total number of pamphlets printed by Mike and Leo -/
theorem total_pamphlets_printed (mike_initial_speed : ℕ) (mike_initial_hours : ℕ) 
  (mike_additional_hours : ℕ) (leo_speed_multiplier : ℕ) : ℕ :=
  let mike_initial_pamphlets := mike_initial_speed * mike_initial_hours
  let mike_reduced_speed := mike_initial_speed / 3
  let mike_additional_pamphlets := mike_reduced_speed * mike_additional_hours
  let leo_hours := mike_initial_hours / 3
  let leo_speed := mike_initial_speed * leo_speed_multiplier
  let leo_pamphlets := leo_speed * leo_hours
  mike_initial_pamphlets + mike_additional_pamphlets + leo_pamphlets

/-- Proves that Mike and Leo print 9400 pamphlets in total -/
theorem prove_total_pamphlets : total_pamphlets_printed 600 9 2 2 = 9400 := by
  sorry

end NUMINAMATH_CALUDE_total_pamphlets_printed_prove_total_pamphlets_l456_45661


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l456_45631

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 12 ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧
  (∀ (m : ℕ), m > 12 → ∃ (l : ℕ), l > 0 ∧ ¬(m ∣ (l * (l + 1) * (l + 2) * (l + 3)))) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l456_45631


namespace NUMINAMATH_CALUDE_square_perimeter_transformation_l456_45603

-- Define a square type
structure Square where
  perimeter : ℝ

-- Define the transformation function
def transform (s : Square) : Square :=
  { perimeter := 12 * s.perimeter }

-- Theorem statement
theorem square_perimeter_transformation (s : Square) :
  (transform s).perimeter = 12 * s.perimeter := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_transformation_l456_45603


namespace NUMINAMATH_CALUDE_product_mod_six_l456_45651

theorem product_mod_six : 2017 * 2018 * 2019 * 2020 ≡ 0 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_product_mod_six_l456_45651


namespace NUMINAMATH_CALUDE_product_of_number_and_sum_of_digits_l456_45682

theorem product_of_number_and_sum_of_digits : 
  let n : ℕ := 26
  let tens : ℕ := n / 10
  let units : ℕ := n % 10
  units = tens + 4 →
  n * (tens + units) = 208 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_number_and_sum_of_digits_l456_45682


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l456_45622

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l456_45622


namespace NUMINAMATH_CALUDE_square_side_estimate_l456_45612

theorem square_side_estimate (A : ℝ) (h : A = 30) :
  ∃ s : ℝ, s^2 = A ∧ 5 < s ∧ s < 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_estimate_l456_45612


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l456_45655

/-- Simplification of a complex expression involving square roots and exponents -/
theorem simplify_sqrt_expression :
  let x := Real.sqrt 3
  (x - 1) ^ (1 - Real.sqrt 2) / (x + 1) ^ (1 + Real.sqrt 2) = 4 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l456_45655


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l456_45616

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l456_45616


namespace NUMINAMATH_CALUDE_modulo_23_equivalence_l456_45674

theorem modulo_23_equivalence (n : ℤ) : 0 ≤ n ∧ n < 23 ∧ -207 ≡ n [ZMOD 23] → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_23_equivalence_l456_45674


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l456_45630

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 → x ≠ -4 →
  (6 * x + 3) / (x^2 - 8 * x - 48) = (75 / 16) / (x - 12) + (21 / 16) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l456_45630


namespace NUMINAMATH_CALUDE_base7_addition_l456_45669

-- Define a function to convert base 7 numbers to natural numbers
def base7ToNat (a b c : Nat) : Nat :=
  a * 7^2 + b * 7 + c

-- Define the two numbers in base 7
def num1 : Nat := base7ToNat 0 2 5
def num2 : Nat := base7ToNat 2 4 6

-- Define the result in base 7
def result : Nat := base7ToNat 3 1 3

-- Theorem statement
theorem base7_addition :
  num1 + num2 = result := by
  sorry

end NUMINAMATH_CALUDE_base7_addition_l456_45669


namespace NUMINAMATH_CALUDE_pretzels_john_ate_l456_45627

/-- Given a bowl of pretzels and information about how many pretzels three people ate,
    prove how many pretzels John ate. -/
theorem pretzels_john_ate (total : ℕ) (john alan marcus : ℕ) 
    (h1 : total = 95)
    (h2 : alan = john - 9)
    (h3 : marcus = john + 12)
    (h4 : marcus = 40) :
    john = 28 := by sorry

end NUMINAMATH_CALUDE_pretzels_john_ate_l456_45627


namespace NUMINAMATH_CALUDE_first_group_work_days_l456_45634

/-- Represents the daily work units done by a person -/
@[ext] structure WorkUnit where
  value : ℚ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Calculates the total work done by a group in a given number of days -/
def totalWork (g : WorkGroup) (manUnit boyUnit : WorkUnit) (days : ℚ) : ℚ :=
  (g.men : ℚ) * manUnit.value * days + (g.boys : ℚ) * boyUnit.value * days

theorem first_group_work_days : 
  let manUnit : WorkUnit := ⟨2⟩
  let boyUnit : WorkUnit := ⟨1⟩
  let firstGroup : WorkGroup := ⟨12, 16⟩
  let secondGroup : WorkGroup := ⟨13, 24⟩
  let secondGroupDays : ℚ := 4
  totalWork firstGroup manUnit boyUnit 5 = totalWork secondGroup manUnit boyUnit secondGroupDays := by
  sorry

end NUMINAMATH_CALUDE_first_group_work_days_l456_45634


namespace NUMINAMATH_CALUDE_no_45_degree_rectangle_with_odd_intersections_l456_45695

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℝ
  y : ℝ

/-- Represents a rectangle on a grid --/
structure GridRectangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint
  D : GridPoint

/-- Checks if a point is on a grid line --/
def isOnGridLine (p : GridPoint) : Prop :=
  ∃ n : ℤ, p.x = n ∨ p.y = n

/-- Checks if a line segment intersects the grid at a 45° angle --/
def intersectsAt45Degrees (p1 p2 : GridPoint) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ p2.x - p1.x = k ∧ p2.y - p1.y = k

/-- Counts the number of grid lines intersected by a line segment --/
noncomputable def gridLinesIntersected (p1 p2 : GridPoint) : ℕ :=
  sorry

/-- Main theorem: No rectangle exists with the given properties --/
theorem no_45_degree_rectangle_with_odd_intersections :
  ¬ ∃ (rect : GridRectangle),
    (¬ isOnGridLine rect.A) ∧ (¬ isOnGridLine rect.B) ∧ 
    (¬ isOnGridLine rect.C) ∧ (¬ isOnGridLine rect.D) ∧
    (intersectsAt45Degrees rect.A rect.B) ∧ 
    (intersectsAt45Degrees rect.B rect.C) ∧
    (intersectsAt45Degrees rect.C rect.D) ∧ 
    (intersectsAt45Degrees rect.D rect.A) ∧
    (Odd (gridLinesIntersected rect.A rect.B)) ∧
    (Odd (gridLinesIntersected rect.B rect.C)) ∧
    (Odd (gridLinesIntersected rect.C rect.D)) ∧
    (Odd (gridLinesIntersected rect.D rect.A)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_45_degree_rectangle_with_odd_intersections_l456_45695


namespace NUMINAMATH_CALUDE_stating_modified_mindmaster_secret_codes_l456_45608

/-- The number of different colors available in the modified Mindmaster game -/
def num_colors : ℕ := 6

/-- The number of slots to be filled in the modified Mindmaster game -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the modified Mindmaster game -/
def num_secret_codes : ℕ := num_colors ^ num_slots

/-- 
Theorem stating that the number of possible secret codes in the modified Mindmaster game is 7776,
given 6 colors, 5 slots, allowing color repetition, and no empty slots.
-/
theorem modified_mindmaster_secret_codes : num_secret_codes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_stating_modified_mindmaster_secret_codes_l456_45608


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l456_45667

theorem sufficient_not_necessary_negation 
  (p q : Prop) 
  (h1 : ¬p → q)  -- ¬p is sufficient for q
  (h2 : ¬(q → ¬p)) -- ¬p is not necessary for q
  : (¬q → p) ∧ ¬(p → ¬q) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l456_45667


namespace NUMINAMATH_CALUDE_translation_result_l456_45684

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally and vertically -/
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_result :
  let p := Point2D.mk (-3) 2
  let p_translated := translate (translate p 2 0) 0 (-4)
  p_translated = Point2D.mk (-1) (-2) := by
  sorry


end NUMINAMATH_CALUDE_translation_result_l456_45684


namespace NUMINAMATH_CALUDE_f_range_l456_45617

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 0 }

-- Define the range
def range : Set ℝ := { y | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem f_range : range = { y | -6 ≤ y ∧ y ≤ -2 } := by sorry

end NUMINAMATH_CALUDE_f_range_l456_45617


namespace NUMINAMATH_CALUDE_sin_inequality_l456_45696

theorem sin_inequality (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 6)
  f x ≤ |f (π / 6)| := by
sorry

end NUMINAMATH_CALUDE_sin_inequality_l456_45696


namespace NUMINAMATH_CALUDE_bread_cost_is_1_1_l456_45620

/-- The cost of each bread given the conditions of the problem -/
def bread_cost (total_breads : ℕ) (num_people : ℕ) (compensation : ℚ) : ℚ :=
  (compensation * 2 * num_people) / total_breads

/-- Theorem stating that the cost of each bread is 1.1 yuan -/
theorem bread_cost_is_1_1 :
  bread_cost 12 3 (22/10) = 11/10 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_is_1_1_l456_45620


namespace NUMINAMATH_CALUDE_galaxy_gym_member_ratio_l456_45614

theorem galaxy_gym_member_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℝ),
    f_avg = 35 →
    m_avg = 45 →
    total_avg = 40 →
    (f_avg * f + m_avg * m) / (f + m) = total_avg →
    f = m :=
by
  sorry

end NUMINAMATH_CALUDE_galaxy_gym_member_ratio_l456_45614


namespace NUMINAMATH_CALUDE_max_sum_of_xy_l456_45636

theorem max_sum_of_xy (x y : ℕ+) : 
  (x * y : ℕ) - (x + y : ℕ) = Nat.gcd x y + Nat.lcm x y → 
  (∃ (c : ℕ), ∀ (a b : ℕ+), 
    (a * b : ℕ) - (a + b : ℕ) = Nat.gcd a b + Nat.lcm a b → 
    (a + b : ℕ) ≤ c) ∧ 
  (x + y : ℕ) ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_xy_l456_45636


namespace NUMINAMATH_CALUDE_tangent_points_constant_sum_l456_45676

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- Checks if a line through two points is tangent to the parabola -/
def isTangent (p1 p2 : Point) : Prop :=
  p2 ∈ Parabola ∧ (∃ k : ℝ, p1.y - p2.y = k * (p1.x - p2.x) ∧ k = p2.x / 2)

theorem tangent_points_constant_sum (a : ℝ) :
  ∀ A B : Point,
  isTangent (Point.mk a (-2)) A ∧
  isTangent (Point.mk a (-2)) B ∧
  A ≠ B →
  A.x * B.x + A.y * B.y = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_constant_sum_l456_45676


namespace NUMINAMATH_CALUDE_percent_of_y_l456_45662

theorem percent_of_y (y : ℝ) (h : y > 0) : ((4 * y) / 20 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l456_45662


namespace NUMINAMATH_CALUDE_scientific_notation_of_19_4_billion_l456_45642

theorem scientific_notation_of_19_4_billion :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 19.4 * (10 ^ 9) = a * (10 ^ n) ∧ a = 1.94 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_19_4_billion_l456_45642


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l456_45664

/-- The distance between two people walking in opposite directions for a given time -/
def distance_apart (maya_speed : ℚ) (lucas_speed : ℚ) (time : ℚ) : ℚ :=
  maya_speed * time + lucas_speed * time

/-- Theorem stating the distance apart after 2 hours -/
theorem distance_after_two_hours :
  let maya_speed : ℚ := 1 / 20 -- miles per minute
  let lucas_speed : ℚ := 3 / 40 -- miles per minute
  let time : ℚ := 120 -- 2 hours in minutes
  distance_apart maya_speed lucas_speed time = 15 := by
  sorry

#eval distance_apart (1/20) (3/40) 120

end NUMINAMATH_CALUDE_distance_after_two_hours_l456_45664


namespace NUMINAMATH_CALUDE_min_sequence_length_l456_45623

def S : Finset Nat := {1, 2, 3, 4}

def isValidPermutation (perm : List Nat) : Prop :=
  perm.length = 4 ∧ perm.toFinset = S ∧ perm.getLast? ≠ some 1

def containsAllValidPermutations (seq : List Nat) : Prop :=
  ∀ perm : List Nat, isValidPermutation perm →
    ∃ i₁ i₂ i₃ i₄ : Nat,
      i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧
      i₄ ≤ seq.length ∧
      seq.get? i₁ = some (perm.get! 0) ∧
      seq.get? i₂ = some (perm.get! 1) ∧
      seq.get? i₃ = some (perm.get! 2) ∧
      seq.get? i₄ = some (perm.get! 3)

theorem min_sequence_length :
  ∃ seq : List Nat, seq.length = 11 ∧ containsAllValidPermutations seq ∧
  ∀ seq' : List Nat, seq'.length < 11 → ¬containsAllValidPermutations seq' :=
sorry

end NUMINAMATH_CALUDE_min_sequence_length_l456_45623


namespace NUMINAMATH_CALUDE_jerry_makes_two_trips_l456_45663

def jerry_trips (carry_capacity : ℕ) (total_trays : ℕ) : ℕ :=
  (total_trays + carry_capacity - 1) / carry_capacity

theorem jerry_makes_two_trips (carry_capacity : ℕ) (total_trays : ℕ) :
  carry_capacity = 8 → total_trays = 16 → jerry_trips carry_capacity total_trays = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_makes_two_trips_l456_45663


namespace NUMINAMATH_CALUDE_A_intersect_B_l456_45659

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem A_intersect_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l456_45659


namespace NUMINAMATH_CALUDE_quiz_passing_requirement_l456_45687

theorem quiz_passing_requirement (total_questions : ℕ) 
  (chemistry_questions biology_questions physics_questions : ℕ)
  (chemistry_correct_percent biology_correct_percent physics_correct_percent : ℚ)
  (passing_grade : ℚ) :
  total_questions = 100 →
  chemistry_questions = 20 →
  biology_questions = 40 →
  physics_questions = 40 →
  chemistry_correct_percent = 80 / 100 →
  biology_correct_percent = 50 / 100 →
  physics_correct_percent = 55 / 100 →
  passing_grade = 65 / 100 →
  (passing_grade * total_questions : ℚ).ceil - 
  (chemistry_correct_percent * chemistry_questions +
   biology_correct_percent * biology_questions +
   physics_correct_percent * physics_questions : ℚ).floor = 7 := by
  sorry

end NUMINAMATH_CALUDE_quiz_passing_requirement_l456_45687


namespace NUMINAMATH_CALUDE_projectile_max_height_l456_45600

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 161

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height : 
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l456_45600


namespace NUMINAMATH_CALUDE_N_subset_M_l456_45672

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 4}

theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l456_45672


namespace NUMINAMATH_CALUDE_money_distribution_l456_45673

theorem money_distribution (a b c d e : ℕ) : 
  a + b + c + d + e = 1000 →
  a + c = 300 →
  b + c = 200 →
  d + e = 350 →
  a + d = 250 →
  b + e = 150 →
  a + b + c = 400 →
  (a = 200 ∧ b = 100 ∧ c = 100 ∧ d = 50 ∧ e = 300) :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l456_45673


namespace NUMINAMATH_CALUDE_inscribed_squares_circles_area_difference_l456_45611

/-- The difference between the sum of areas of squares and circles in an infinite inscribed sequence -/
theorem inscribed_squares_circles_area_difference :
  let square_areas : ℕ → ℝ := λ n => (1 / 2 : ℝ) ^ n
  let circle_areas : ℕ → ℝ := λ n => π / 4 * (1 / 2 : ℝ) ^ n
  (∑' n, square_areas n) - (∑' n, circle_areas n) = 2 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_circles_area_difference_l456_45611


namespace NUMINAMATH_CALUDE_abs_ratio_equal_sqrt_seven_thirds_l456_45618

theorem abs_ratio_equal_sqrt_seven_thirds (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 5*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/3) := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_equal_sqrt_seven_thirds_l456_45618


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l456_45643

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  (a + b + c = 180) →
  (b = 6/5 * a) →
  (c = 7/5 * a) →
  a = 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l456_45643


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l456_45698

theorem rectangle_area_diagonal_relation :
  ∀ (length width diagonal : ℝ),
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 5 / 2 →
  diagonal^2 = length^2 + width^2 →
  diagonal = 13 →
  ∃ (k : ℝ), length * width = k * diagonal^2 ∧ k = 10 / 29 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l456_45698


namespace NUMINAMATH_CALUDE_pool_capacity_l456_45653

theorem pool_capacity (current_water : ℝ) (h1 : current_water > 0) 
  (h2 : current_water + 300 = 0.8 * 1875) 
  (h3 : current_water + 300 = 1.25 * current_water) : 
  1875 = 1875 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l456_45653


namespace NUMINAMATH_CALUDE_total_cost_is_54_44_l456_45644

-- Define the quantities and prices
def book_quantity : ℕ := 1
def book_price : ℚ := 16
def binder_quantity : ℕ := 3
def binder_price : ℚ := 2
def notebook_quantity : ℕ := 6
def notebook_price : ℚ := 1
def pen_quantity : ℕ := 4
def pen_price : ℚ := 1/2
def calculator_quantity : ℕ := 2
def calculator_price : ℚ := 12

-- Define discount and tax rates
def discount_rate : ℚ := 1/10
def tax_rate : ℚ := 7/100

-- Define the total cost function
def total_cost : ℚ :=
  let book_cost := book_quantity * book_price
  let binder_cost := binder_quantity * binder_price
  let notebook_cost := notebook_quantity * notebook_price
  let pen_cost := pen_quantity * pen_price
  let calculator_cost := calculator_quantity * calculator_price
  
  let discounted_book_cost := book_cost * (1 - discount_rate)
  let discounted_binder_cost := binder_cost * (1 - discount_rate)
  
  let subtotal := discounted_book_cost + discounted_binder_cost + notebook_cost + pen_cost + calculator_cost
  let tax := (notebook_cost + pen_cost + calculator_cost) * tax_rate
  
  subtotal + tax

-- Theorem statement
theorem total_cost_is_54_44 : total_cost = 5444 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_54_44_l456_45644


namespace NUMINAMATH_CALUDE_total_remaining_is_13589_08_l456_45619

/-- Represents the daily sales and ingredient cost data for Du Chin's meat pie business --/
structure DailyData where
  pies_sold : ℕ
  sales : ℚ
  ingredient_cost : ℚ
  remaining : ℚ

/-- Calculates the daily data for Du Chin's meat pie business over a week --/
def calculate_week_data : List DailyData :=
  let monday_data : DailyData := {
    pies_sold := 200,
    sales := 4000,
    ingredient_cost := 2400,
    remaining := 1600
  }
  let tuesday_data : DailyData := {
    pies_sold := 220,
    sales := 4400,
    ingredient_cost := 2640,
    remaining := 1760
  }
  let wednesday_data : DailyData := {
    pies_sold := 209,
    sales := 4180,
    ingredient_cost := 2376,
    remaining := 1804
  }
  let thursday_data : DailyData := {
    pies_sold := 209,
    sales := 4180,
    ingredient_cost := 2376,
    remaining := 1804
  }
  let friday_data : DailyData := {
    pies_sold := 240,
    sales := 4800,
    ingredient_cost := 2494.80,
    remaining := 2305.20
  }
  let saturday_data : DailyData := {
    pies_sold := 221,
    sales := 4420,
    ingredient_cost := 2370.06,
    remaining := 2049.94
  }
  let sunday_data : DailyData := {
    pies_sold := 232,
    sales := 4640,
    ingredient_cost := 2370.06,
    remaining := 2269.94
  }
  [monday_data, tuesday_data, wednesday_data, thursday_data, friday_data, saturday_data, sunday_data]

/-- Calculates the total remaining money for the week --/
def total_remaining (week_data : List DailyData) : ℚ :=
  week_data.foldl (fun acc day => acc + day.remaining) 0

/-- Theorem stating that the total remaining money for the week is $13589.08 --/
theorem total_remaining_is_13589_08 :
  total_remaining (calculate_week_data) = 13589.08 := by
  sorry


end NUMINAMATH_CALUDE_total_remaining_is_13589_08_l456_45619


namespace NUMINAMATH_CALUDE_power_fraction_equality_l456_45639

theorem power_fraction_equality : (27 ^ 20) / (81 ^ 10) = 3 ^ 20 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l456_45639


namespace NUMINAMATH_CALUDE_earl_initial_ascent_l456_45681

def building_height : ℕ := 20

def initial_floor : ℕ := 1

theorem earl_initial_ascent (x : ℕ) : 
  x + 5 = building_height - 9 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_earl_initial_ascent_l456_45681


namespace NUMINAMATH_CALUDE_time_for_accidents_l456_45607

-- Define the frequency of car collisions and big crashes
def collision_frequency : ℕ := 10  -- seconds
def crash_frequency : ℕ := 20  -- seconds

-- Define the total number of accidents
def total_accidents : ℕ := 36

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Theorem to prove
theorem time_for_accidents : 
  ∃ (minutes : ℕ), 
    (seconds_per_minute / collision_frequency + seconds_per_minute / crash_frequency) * minutes = total_accidents ∧
    minutes = 4 :=
by sorry

end NUMINAMATH_CALUDE_time_for_accidents_l456_45607


namespace NUMINAMATH_CALUDE_parallelogram_properties_independence_l456_45621

/-- A parallelogram with potentially equal sides and/or right angles -/
structure Parallelogram where
  has_equal_sides : Bool
  has_right_angles : Bool

/-- Theorem: There exist parallelograms with equal sides but not right angles, 
    and parallelograms with right angles but not equal sides -/
theorem parallelogram_properties_independence :
  ∃ (p q : Parallelogram), 
    (p.has_equal_sides ∧ ¬p.has_right_angles) ∧
    (q.has_right_angles ∧ ¬q.has_equal_sides) :=
by
  sorry


end NUMINAMATH_CALUDE_parallelogram_properties_independence_l456_45621


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l456_45691

theorem not_p_or_q_false_implies_p_or_q (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l456_45691


namespace NUMINAMATH_CALUDE_rhombus_diagonals_property_inequality_or_equality_l456_45652

-- Definition for rhombus properties
def diagonals_perpendicular (r : Type) : Prop := sorry
def diagonals_bisect (r : Type) : Prop := sorry

-- Theorem for the first compound proposition
theorem rhombus_diagonals_property :
  ∀ (r : Type), diagonals_perpendicular r ∧ diagonals_bisect r :=
sorry

-- Theorem for the second compound proposition
theorem inequality_or_equality : 2 < 3 ∨ 2 = 3 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_property_inequality_or_equality_l456_45652


namespace NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_purchase_l456_45699

theorem additional_money_needed (initial_amount : ℝ) 
  (additional_fraction : ℝ) (discount_percentage : ℝ) : ℝ :=
  let total_before_discount := initial_amount * (1 + additional_fraction)
  let discounted_amount := total_before_discount * (1 - discount_percentage / 100)
  discounted_amount - initial_amount

theorem mrs_smith_purchase : 
  additional_money_needed 500 (2/5) 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_purchase_l456_45699


namespace NUMINAMATH_CALUDE_basketball_cards_per_box_l456_45604

theorem basketball_cards_per_box : 
  ∀ (basketball_cards_per_box : ℕ),
    (4 * basketball_cards_per_box + 5 * 8 = 58 + 22) → 
    basketball_cards_per_box = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_cards_per_box_l456_45604


namespace NUMINAMATH_CALUDE_number_wall_value_l456_45689

/-- Represents a simplified Number Wall with four bottom values and a top value --/
structure NumberWall where
  bottom_left : ℕ
  bottom_mid_left : ℕ
  bottom_mid_right : ℕ
  bottom_right : ℕ
  top : ℕ

/-- The Number Wall is valid if it follows the construction rules --/
def is_valid_number_wall (w : NumberWall) : Prop :=
  ∃ (mid_left mid_right : ℕ),
    w.bottom_left + w.bottom_mid_left = mid_left ∧
    w.bottom_mid_left + w.bottom_mid_right = mid_right ∧
    w.bottom_mid_right + w.bottom_right = w.top - mid_left ∧
    mid_left + mid_right = w.top

theorem number_wall_value (w : NumberWall) 
    (h : is_valid_number_wall w)
    (h1 : w.bottom_mid_left = 6)
    (h2 : w.bottom_mid_right = 10)
    (h3 : w.bottom_right = 9)
    (h4 : w.top = 64) :
  w.bottom_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_value_l456_45689


namespace NUMINAMATH_CALUDE_complex_circle_range_l456_45668

theorem complex_circle_range (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  (Complex.abs (z - Complex.mk 3 4) = 1) →
  (16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36) :=
by sorry

end NUMINAMATH_CALUDE_complex_circle_range_l456_45668


namespace NUMINAMATH_CALUDE_octagonal_pyramid_volume_l456_45679

/-- The volume of a regular octagonal pyramid with given dimensions -/
theorem octagonal_pyramid_volume :
  ∀ (base_side_length equilateral_face_side_length : ℝ),
    base_side_length = 5 →
    equilateral_face_side_length = 10 →
    ∃ (volume : ℝ),
      volume = (250 * Real.sqrt 3 * (1 + Real.sqrt 2)) / 3 ∧
      volume = (1 / 3) * (2 * (1 + Real.sqrt 2) * base_side_length^2) * 
               ((equilateral_face_side_length * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_octagonal_pyramid_volume_l456_45679


namespace NUMINAMATH_CALUDE_original_girls_count_l456_45647

/-- Represents the number of boys and girls in a school club. -/
structure ClubMembers where
  boys : ℕ
  girls : ℕ

/-- Defines the conditions of the club membership problem. -/
def ClubProblem (initial : ClubMembers) : Prop :=
  -- Initially, there was one boy for every girl
  initial.boys = initial.girls ∧
  -- After 25 girls leave, there are three boys for each remaining girl
  3 * (initial.girls - 25) = initial.boys ∧
  -- After that, 60 boys leave, and then there are six girls for each remaining boy
  6 * (initial.boys - 60) = initial.girls - 25

/-- Theorem stating that given the conditions, the original number of girls is 67. -/
theorem original_girls_count (initial : ClubMembers) :
  ClubProblem initial → initial.girls = 67 := by
  sorry


end NUMINAMATH_CALUDE_original_girls_count_l456_45647


namespace NUMINAMATH_CALUDE_binomial_square_expansion_l456_45625

theorem binomial_square_expansion (x : ℝ) : (1 - x)^2 = 1 - 2*x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_expansion_l456_45625


namespace NUMINAMATH_CALUDE_characterize_f_l456_45632

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ n, f n ≠ 1 ∧ f n + f (n + 1) = f (n + 2) + f (n + 3) - 168

theorem characterize_f (f : ℕ → ℕ) (h : is_valid_f f) :
  ∃ (c d a : ℕ), (∀ n, f (2 * n) = c + n * d) ∧
                 (∀ n, f (2 * n + 1) = (168 - d) * n + a - c) ∧
                 c > 1 ∧
                 a > c + 1 :=
sorry

end NUMINAMATH_CALUDE_characterize_f_l456_45632


namespace NUMINAMATH_CALUDE_red_pens_count_l456_45637

/-- The number of red pens in Maria's desk drawer. -/
def red_pens : ℕ := sorry

/-- The number of black pens in Maria's desk drawer. -/
def black_pens : ℕ := red_pens + 10

/-- The number of blue pens in Maria's desk drawer. -/
def blue_pens : ℕ := red_pens + 7

/-- The total number of pens in Maria's desk drawer. -/
def total_pens : ℕ := 41

theorem red_pens_count : red_pens = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_pens_count_l456_45637


namespace NUMINAMATH_CALUDE_sqrt_23_minus_1_lt_4_l456_45601

theorem sqrt_23_minus_1_lt_4 : Real.sqrt 23 - 1 < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_23_minus_1_lt_4_l456_45601


namespace NUMINAMATH_CALUDE_pencil_count_l456_45671

/-- The number of pencils Mitchell has -/
def mitchell_pencils : ℕ := 30

/-- The number of pencils Antonio has -/
def antonio_pencils : ℕ := mitchell_pencils - (mitchell_pencils * 20 / 100)

/-- The number of pencils Elizabeth has -/
def elizabeth_pencils : ℕ := 2 * antonio_pencils

/-- The total number of pencils Mitchell, Antonio, and Elizabeth have together -/
def total_pencils : ℕ := mitchell_pencils + antonio_pencils + elizabeth_pencils

theorem pencil_count : total_pencils = 102 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l456_45671


namespace NUMINAMATH_CALUDE_solution_set_l456_45654

theorem solution_set (x : ℝ) : (x^2 - 3*x > 8 ∧ |x| > 2) ↔ x < -2 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l456_45654


namespace NUMINAMATH_CALUDE_bird_nest_difference_l456_45649

theorem bird_nest_difference :
  let num_birds : ℕ := 6
  let num_nests : ℕ := 3
  num_birds - num_nests = 3 := by sorry

end NUMINAMATH_CALUDE_bird_nest_difference_l456_45649


namespace NUMINAMATH_CALUDE_half_sum_squares_even_odd_l456_45683

theorem half_sum_squares_even_odd (a b : ℤ) :
  (∃ x y : ℤ, (4 * a^2 + 4 * b^2) / 2 = x^2 + y^2) ∨
  (∃ x y : ℤ, ((2 * a + 1)^2 + (2 * b + 1)^2) / 2 = x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_half_sum_squares_even_odd_l456_45683


namespace NUMINAMATH_CALUDE_incorrect_bracket_expansion_l456_45693

theorem incorrect_bracket_expansion : ∀ x : ℝ, 3 * x^2 - 3 * (x + 6) ≠ 3 * x^2 - 3 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_bracket_expansion_l456_45693


namespace NUMINAMATH_CALUDE_subjective_not_set_l456_45670

-- Define what it means for a collection to have objective membership criteria
def has_objective_criteria (C : Type → Prop) : Prop :=
  ∀ (x : Type), (C x ∨ ¬C x) ∧ (∃ (f : Type → Bool), ∀ y, C y ↔ f y = true)

-- Define a set as a collection with objective membership criteria
def is_set (S : Type → Prop) : Prop := has_objective_criteria S

-- Define a collection with subjective criteria (e.g., "good friends")
def subjective_collection (x : Type) : Prop := sorry

-- Theorem: A collection with subjective criteria cannot form a set
theorem subjective_not_set : ¬(is_set subjective_collection) :=
sorry

end NUMINAMATH_CALUDE_subjective_not_set_l456_45670


namespace NUMINAMATH_CALUDE_apples_bought_correct_l456_45609

/-- Represents the number of apples Mary bought -/
def apples_bought : ℕ := 6

/-- Represents the number of apples Mary ate -/
def apples_eaten : ℕ := 2

/-- Represents the number of trees planted per apple eaten -/
def trees_per_apple : ℕ := 2

/-- Theorem stating that the number of apples Mary bought is correct -/
theorem apples_bought_correct : 
  apples_bought = apples_eaten + apples_eaten * trees_per_apple :=
by sorry

end NUMINAMATH_CALUDE_apples_bought_correct_l456_45609


namespace NUMINAMATH_CALUDE_starting_number_sequence_l456_45602

theorem starting_number_sequence (n : ℕ) : 
  (n ≤ 79) →                          -- Last number is less than or equal to 79
  (n % 11 = 0) →                      -- Last number is divisible by 11
  (∃ (m : ℕ), n = m * 11) →           -- n is a multiple of 11
  (∃ (k : ℕ), n = 11 * 7 - k * 11) →  -- n is the 7th number in the sequence
  (11 : ℕ) = n - 6 * 11               -- Starting number is 11
  := by sorry

end NUMINAMATH_CALUDE_starting_number_sequence_l456_45602


namespace NUMINAMATH_CALUDE_roosevelt_bonus_points_l456_45690

/-- Represents the points scored by Roosevelt High School in each game and the bonus points received --/
structure RooseveltPoints where
  first_game : ℕ
  second_game : ℕ
  third_game : ℕ
  bonus : ℕ

/-- Represents the total points scored by Greendale High School --/
def greendale_points : ℕ := 130

/-- Calculates the total points scored by Roosevelt High School before bonus --/
def roosevelt_total (p : RooseveltPoints) : ℕ :=
  p.first_game + p.second_game + p.third_game

/-- Theorem stating the bonus points received by Roosevelt High School --/
theorem roosevelt_bonus_points :
  ∀ p : RooseveltPoints,
  p.first_game = 30 →
  p.second_game = p.first_game / 2 →
  p.third_game = p.second_game * 3 →
  greendale_points = roosevelt_total p + p.bonus →
  p.bonus = 40 := by
  sorry

end NUMINAMATH_CALUDE_roosevelt_bonus_points_l456_45690
