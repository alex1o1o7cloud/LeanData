import Mathlib

namespace total_shaded_area_l1009_100963

/-- Calculates the total shaded area of a floor tiled with patterned square tiles. -/
theorem total_shaded_area (floor_length floor_width tile_size circle_radius : ℝ) : 
  floor_length = 8 ∧ 
  floor_width = 10 ∧ 
  tile_size = 2 ∧ 
  circle_radius = 1 →
  (floor_length * floor_width / (tile_size * tile_size)) * (tile_size * tile_size - π * circle_radius ^ 2) = 80 - 20 * π :=
by sorry

end total_shaded_area_l1009_100963


namespace biographies_shelved_l1009_100977

def total_books : ℕ := 46
def top_section_books : ℕ := 24
def western_novels : ℕ := 5

def bottom_section_books : ℕ := total_books - top_section_books

def mystery_books : ℕ := bottom_section_books / 2

theorem biographies_shelved :
  total_books - top_section_books - mystery_books - western_novels = 6 :=
by sorry

end biographies_shelved_l1009_100977


namespace train_speed_l1009_100996

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1000) (h2 : time = 200) :
  length / time = 5 := by
  sorry

end train_speed_l1009_100996


namespace parallel_vectors_m_value_l1009_100924

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (m, 4) (3, -2) → m = -6 := by
  sorry

end parallel_vectors_m_value_l1009_100924


namespace triangle_inequality_with_square_roots_l1009_100956

/-- Given a triangle with sides a, b, and c, the sum of the square roots of the semiperimeter minus each side is less than or equal to the sum of the square roots of the sides. Equality holds if and only if the triangle is equilateral. -/
theorem triangle_inequality_with_square_roots (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) ≤ 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ∧
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) = 
   Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_with_square_roots_l1009_100956


namespace calculate_product_l1009_100967

theorem calculate_product : 
  (0.125 : ℝ)^3 * (-8 : ℝ)^3 = -1 :=
by
  sorry

end calculate_product_l1009_100967


namespace solve_for_x_l1009_100997

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 := by
  sorry

end solve_for_x_l1009_100997


namespace direct_proportion_function_l1009_100929

/-- A function that is directly proportional to 2x+3 and passes through the point (1, -5) -/
def f (x : ℝ) : ℝ := -2 * x - 3

theorem direct_proportion_function :
  (∃ k : ℝ, ∀ x, f x = k * (2 * x + 3)) ∧
  f 1 = -5 ∧
  (∀ x, f x = -2 * x - 3) ∧
  (f (5/2) = 2) := by
  sorry

#check direct_proportion_function

end direct_proportion_function_l1009_100929


namespace line_perp_parallel_implies_planes_perp_l1009_100906

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relationship between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perpendicular_planes α β :=
sorry

end line_perp_parallel_implies_planes_perp_l1009_100906


namespace polynomial_identity_l1009_100940

variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

def polynomial_equation : Prop :=
  ∀ x : ℝ, (1 + x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5

theorem polynomial_identity (h : polynomial_equation a₀ a₁ a₂ a₃ a₄ a₅) :
  a₀ = 1 ∧ (a₀ / 1 + a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 21 / 2) :=
by sorry

end polynomial_identity_l1009_100940


namespace diagonals_25_sided_polygon_l1009_100921

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 25 sides is 275 -/
theorem diagonals_25_sided_polygon :
  num_diagonals 25 = 275 := by sorry

end diagonals_25_sided_polygon_l1009_100921


namespace F_sum_positive_l1009_100947

/-- The function f(x) = ax^2 + bx + 1 -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function F(x) defined piecewise based on f(x) -/
noncomputable def F (a b x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

/-- Theorem stating that F(m) + F(n) > 0 under given conditions -/
theorem F_sum_positive (a b m n : ℝ) : 
  f a b (-1) = 0 → 
  (∀ x, f a b x ≥ 0) → 
  m * n < 0 → 
  m + n > 0 → 
  a > 0 → 
  (∀ x, f a b x = f a b (-x)) → 
  F a b m + F a b n > 0 := by
  sorry

end F_sum_positive_l1009_100947


namespace function_behavior_l1009_100959

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
    (h_odd : is_odd f)
    (h_periodic : ∀ x, f x = f (x - 2))
    (h_decreasing : is_decreasing_on f 1 2) :
  is_decreasing_on f (-3) (-2) ∧ is_increasing_on f 0 1 := by
  sorry

end function_behavior_l1009_100959


namespace contrapositive_square_equality_l1009_100923

theorem contrapositive_square_equality (a b : ℝ) : a^2 ≠ b^2 → a ≠ b := by
  sorry

end contrapositive_square_equality_l1009_100923


namespace specific_pentagon_area_l1009_100955

/-- Pentagon formed by cutting a right-angled triangular corner from a rectangle -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  parallel_sides : Bool
  triangle_leg1 : ℝ
  triangle_leg2 : ℝ

/-- The area of the pentagon -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∃ (p : Pentagon),
    p.side1 = 9 ∧ p.side2 = 16 ∧ p.side3 = 30 ∧ p.side4 = 40 ∧ p.side5 = 41 ∧
    p.parallel_sides = true ∧
    p.triangle_leg1 = 9 ∧ p.triangle_leg2 = 40 ∧
    pentagon_area p = 1020 := by
  sorry

end specific_pentagon_area_l1009_100955


namespace ceiling_sum_sqrt_l1009_100934

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l1009_100934


namespace intersection_P_Q_l1009_100903

def P : Set ℤ := {x | |x - 1| < 2}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {0, 1, 2} := by sorry

end intersection_P_Q_l1009_100903


namespace tens_digit_of_8_pow_2015_l1009_100935

/-- The function that returns the last two digits of 8^n -/
def lastTwoDigits (n : ℕ) : ℕ := (8^n) % 100

/-- The cycle length of the last two digits of 8^n -/
def cycleLengthOfLastTwoDigits : ℕ := 20

theorem tens_digit_of_8_pow_2015 :
  ∃ (f : ℕ → ℕ),
    (∀ n, f n = lastTwoDigits n) ∧
    (∀ n, f (n + cycleLengthOfLastTwoDigits) = f n) ∧
    (f 15 = 32) →
    (8^2015 / 10) % 10 = 3 := by
  sorry

end tens_digit_of_8_pow_2015_l1009_100935


namespace abs_2x_minus_7_not_positive_l1009_100978

theorem abs_2x_minus_7_not_positive (x : ℚ) : |2*x - 7| ≤ 0 ↔ x = 7/2 := by
  sorry

end abs_2x_minus_7_not_positive_l1009_100978


namespace zoo_incident_final_counts_l1009_100958

def wombat_count : ℕ := 9
def rhea_count : ℕ := 3
def porcupine_count : ℕ := 2

def carson_claw_per_wombat : ℕ := 4
def ava_claw_per_rhea : ℕ := 1
def liam_quill_per_porcupine : ℕ := 6

def carson_reduction_percent : ℚ := 25 / 100
def ava_reduction_percent : ℚ := 25 / 100
def liam_reduction_percent : ℚ := 50 / 100

def carson_initial_claws : ℕ := wombat_count * carson_claw_per_wombat
def ava_initial_claws : ℕ := rhea_count * ava_claw_per_rhea
def liam_initial_quills : ℕ := porcupine_count * liam_quill_per_porcupine

theorem zoo_incident_final_counts :
  (carson_initial_claws - Int.floor (↑carson_initial_claws * carson_reduction_percent) = 27) ∧
  (ava_initial_claws - Int.floor (↑ava_initial_claws * ava_reduction_percent) = 3) ∧
  (liam_initial_quills - Int.floor (↑liam_initial_quills * liam_reduction_percent) = 6) :=
by sorry

end zoo_incident_final_counts_l1009_100958


namespace root_value_theorem_l1009_100962

theorem root_value_theorem (a : ℝ) : a^2 + 3*a + 2 = 0 → a^2 + 3*a = -2 := by
  sorry

end root_value_theorem_l1009_100962


namespace vector_magnitude_problem_l1009_100965

theorem vector_magnitude_problem (m : ℝ) (a : ℝ × ℝ) :
  m > 0 → a = (m, 4) → ‖a‖ = 5 → m = 3 := by sorry

end vector_magnitude_problem_l1009_100965


namespace frisbee_cost_l1009_100972

/-- The cost of a frisbee given initial money, kite cost, and remaining money --/
theorem frisbee_cost (initial_money kite_cost remaining_money : ℕ) : 
  initial_money = 78 → 
  kite_cost = 8 → 
  remaining_money = 61 → 
  initial_money - kite_cost - remaining_money = 9 := by
sorry

end frisbee_cost_l1009_100972


namespace deepak_age_l1009_100907

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
  sorry

end deepak_age_l1009_100907


namespace sales_tax_satisfies_conditions_l1009_100945

/-- The sales tax percentage that satisfies the given conditions -/
def sales_tax_percentage : ℝ :=
  -- Define the sales tax percentage
  -- We don't know its exact value yet
  sorry

/-- The cost of the lunch before tax and tip -/
def lunch_cost : ℝ := 100

/-- The tip percentage -/
def tip_percentage : ℝ := 0.06

/-- The total amount paid -/
def total_paid : ℝ := 110

/-- Theorem stating that the sales tax percentage satisfies the given conditions -/
theorem sales_tax_satisfies_conditions :
  lunch_cost + sales_tax_percentage + 
  tip_percentage * (lunch_cost + sales_tax_percentage) = total_paid :=
by
  sorry

end sales_tax_satisfies_conditions_l1009_100945


namespace arithmetic_sequence_problem_l1009_100992

theorem arithmetic_sequence_problem :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence property
  (a 4 + a 5 + a 6 + a 7 = 56) →    -- given condition
  (a 4 * a 7 = 187) →               -- given condition
  ((a 1 = 5 ∧ d = 2) ∨ (a 1 = 23 ∧ d = -2)) := by
sorry

end arithmetic_sequence_problem_l1009_100992


namespace base_conversion_l1009_100982

/-- Given that the base 6 number 123₆ is equal to the base b number 203ᵦ,
    prove that the positive value of b is 2√6. -/
theorem base_conversion (b : ℝ) (h : b > 0) : 
  (1 * 6^2 + 2 * 6 + 3 : ℝ) = 2 * b^2 + 3 → b = 2 * Real.sqrt 6 := by
  sorry

end base_conversion_l1009_100982


namespace arrangements_count_l1009_100928

/-- The number of possible arrangements for 5 male students and 3 female students
    standing in a row, where the female students must stand together. -/
def num_arrangements : ℕ :=
  let num_male_students : ℕ := 5
  let num_female_students : ℕ := 3
  num_male_students.factorial * (num_male_students + 1) * num_female_students.factorial

theorem arrangements_count : num_arrangements = 720 := by
  sorry

end arrangements_count_l1009_100928


namespace conference_games_l1009_100999

/-- The number of games in a complete season for a sports conference --/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

/-- Theorem stating the number of games in the specific conference setup --/
theorem conference_games : total_games 16 8 3 2 = 296 := by
  sorry

end conference_games_l1009_100999


namespace probability_at_least_one_female_l1009_100961

def total_students : ℕ := 10
def male_students : ℕ := 6
def female_students : ℕ := 4
def selected_students : ℕ := 3

theorem probability_at_least_one_female :
  let total_combinations := Nat.choose total_students selected_students
  let all_male_combinations := Nat.choose male_students selected_students
  (1 : ℚ) - (all_male_combinations : ℚ) / (total_combinations : ℚ) = 5 / 6 := by
  sorry

end probability_at_least_one_female_l1009_100961


namespace chestnut_picking_l1009_100994

/-- The amount of chestnuts picked by Mary, Peter, and Lucy -/
theorem chestnut_picking (mary peter lucy : ℝ) 
  (h1 : mary = 2 * peter)  -- Mary picked twice as much as Peter
  (h2 : lucy = peter + 2)  -- Lucy picked 2 kg more than Peter
  (h3 : mary = 12)         -- Mary picked 12 kg
  : mary + peter + lucy = 26 := by
  sorry

#check chestnut_picking

end chestnut_picking_l1009_100994


namespace medium_boxes_count_l1009_100995

def tape_large : ℕ := 4
def tape_medium : ℕ := 2
def tape_small : ℕ := 1
def tape_label : ℕ := 1
def large_boxes : ℕ := 2
def small_boxes : ℕ := 5
def total_tape : ℕ := 44

theorem medium_boxes_count : 
  ∃ (medium_boxes : ℕ), 
    large_boxes * (tape_large + tape_label) + 
    medium_boxes * (tape_medium + tape_label) + 
    small_boxes * (tape_small + tape_label) = total_tape ∧ 
    medium_boxes = 8 := by
sorry

end medium_boxes_count_l1009_100995


namespace probability_yellow_ball_l1009_100983

def total_balls : ℕ := 5
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3

theorem probability_yellow_ball :
  (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by sorry

end probability_yellow_ball_l1009_100983


namespace luke_fillets_l1009_100908

/-- Calculates the total number of fish fillets Luke has after fishing for a given number of days -/
def total_fillets (fish_per_day : ℕ) (days : ℕ) (fillets_per_fish : ℕ) : ℕ :=
  fish_per_day * days * fillets_per_fish

/-- Proves that Luke has 120 fillets after fishing for 30 days, catching 2 fish per day, with 2 fillets per fish -/
theorem luke_fillets :
  total_fillets 2 30 2 = 120 := by
  sorry

end luke_fillets_l1009_100908


namespace fraction_division_multiplication_l1009_100989

theorem fraction_division_multiplication :
  (3 : ℚ) / 7 / 4 * 2 = 3 / 14 := by
  sorry

end fraction_division_multiplication_l1009_100989


namespace university_packaging_cost_l1009_100914

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the pricing scheme for boxes -/
structure BoxPricing where
  initialPrice : ℝ
  initialQuantity : ℕ
  additionalPrice : ℝ

/-- Calculates the minimum cost for packaging a given number of items -/
def minimumPackagingCost (boxDim : BoxDimensions) (pricing : BoxPricing) (itemCount : ℕ) : ℝ :=
  let initialCost := pricing.initialPrice * pricing.initialQuantity
  let additionalBoxes := max (itemCount - pricing.initialQuantity) 0
  let additionalCost := pricing.additionalPrice * additionalBoxes
  initialCost + additionalCost

/-- Theorem stating the minimum packaging cost for the university's collection -/
theorem university_packaging_cost :
  let boxDim : BoxDimensions := { length := 18, width := 22, height := 15 }
  let pricing : BoxPricing := { initialPrice := 0.60, initialQuantity := 100, additionalPrice := 0.55 }
  let itemCount : ℕ := 127
  minimumPackagingCost boxDim pricing itemCount = 74.85 := by
  sorry


end university_packaging_cost_l1009_100914


namespace g_at_3_l1009_100954

def g (x : ℝ) : ℝ := 5 * x^3 - 3 * x^2 + 7 * x - 2

theorem g_at_3 : g 3 = 127 := by
  sorry

end g_at_3_l1009_100954


namespace battle_station_staffing_l1009_100951

/-- The number of ways to assign n distinct objects to k distinct positions,
    where each position must be filled by exactly one object. -/
def permutations (n k : ℕ) : ℕ := Nat.descFactorial n k

/-- The number of job openings -/
def num_jobs : ℕ := 6

/-- The number of suitable candidates -/
def num_candidates : ℕ := 15

theorem battle_station_staffing :
  permutations num_candidates num_jobs = 3276000 := by sorry

end battle_station_staffing_l1009_100951


namespace least_addend_for_divisibility_least_addend_for_929_div_30_l1009_100966

theorem least_addend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! x : ℕ, x < d ∧ (n + x) % d = 0 :=
by sorry

theorem least_addend_for_929_div_30 :
  (∃! x : ℕ, x < 30 ∧ (929 + x) % 30 = 0) ∧
  (∀ y : ℕ, y < 30 ∧ (929 + y) % 30 = 0 → y = 1) :=
by sorry

end least_addend_for_divisibility_least_addend_for_929_div_30_l1009_100966


namespace pams_age_l1009_100933

/-- Proves that Pam's current age is 5 years, given the conditions of the problem -/
theorem pams_age (pam_age rena_age : ℕ) 
  (h1 : pam_age = rena_age / 2)
  (h2 : rena_age + 10 = (pam_age + 10) + 5) : 
  pam_age = 5 := by sorry

end pams_age_l1009_100933


namespace tom_game_sale_amount_l1009_100950

/-- Calculates the amount received from selling a portion of an asset that has increased in value -/
def sellPartOfAppreciatedAsset (initialValue : ℝ) (appreciationFactor : ℝ) (portionSold : ℝ) : ℝ :=
  initialValue * appreciationFactor * portionSold

/-- Proves that Tom sold his games for $240 -/
theorem tom_game_sale_amount : 
  sellPartOfAppreciatedAsset 200 3 0.4 = 240 := by
  sorry

end tom_game_sale_amount_l1009_100950


namespace solve_for_y_l1009_100913

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 7 = y + 2) (h2 : x = -5) : y = 45 := by
  sorry

end solve_for_y_l1009_100913


namespace savings_calculation_l1009_100901

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  p1.income = 3000 ∧
  4 * p1.income = 5 * p2.income ∧
  2 * p1.expenditure = 3 * p2.expenditure ∧
  p1.income - p1.expenditure = p2.income - p2.expenditure

/-- The theorem to prove -/
theorem savings_calculation (p1 p2 : Person) :
  financialProblem p1 p2 → p1.income - p1.expenditure = 1200 := by
  sorry

#check savings_calculation

end savings_calculation_l1009_100901


namespace expression_simplification_l1009_100944

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 1) :
  (x^2 - 1) / x / (1 + 1/x) = Real.sqrt 5 := by
  sorry

end expression_simplification_l1009_100944


namespace sqrt_equation_solution_l1009_100975

theorem sqrt_equation_solution (x : ℝ) : 
  (Real.sqrt (4 - 5*x + x^2) = 9) ↔ (x = (5 + Real.sqrt 333) / 2 ∨ x = (5 - Real.sqrt 333) / 2) := by
  sorry

end sqrt_equation_solution_l1009_100975


namespace eighteenth_decimal_is_nine_l1009_100957

/-- Represents the decimal expansion of a fraction -/
def DecimalExpansion := ℕ → Fin 10

/-- The decimal expansion of 10/11 -/
def decimal_expansion_10_11 : DecimalExpansion :=
  fun n => if n % 2 = 0 then 0 else 9

theorem eighteenth_decimal_is_nine
  (h : ∀ n : ℕ, decimal_expansion_10_11 (20 - n) = 9 → decimal_expansion_10_11 n = 9) :
  decimal_expansion_10_11 18 = 9 :=
by
  sorry


end eighteenth_decimal_is_nine_l1009_100957


namespace max_xy_given_sum_l1009_100946

theorem max_xy_given_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  x * y ≤ 81 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 18 ∧ x * y = 81 :=
by sorry

end max_xy_given_sum_l1009_100946


namespace quadratic_roots_l1009_100912

theorem quadratic_roots (p q : ℚ) : 
  (∃ f : ℚ → ℚ, (∀ x, f x = x^2 + p*x + q) ∧ f p = 0 ∧ f q = 0) ↔ 
  ((p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2)) :=
by sorry

end quadratic_roots_l1009_100912


namespace marcella_shoes_l1009_100952

/-- Given a number of initial shoe pairs and lost individual shoes, 
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) : ℕ :=
  initial_pairs - lost_shoes

/-- Theorem stating that with 24 initial pairs and 9 lost shoes, 
    the maximum number of complete pairs remaining is 15. -/
theorem marcella_shoes : max_remaining_pairs 24 9 = 15 := by
  sorry

end marcella_shoes_l1009_100952


namespace mod_29_graph_intercepts_sum_l1009_100998

theorem mod_29_graph_intercepts_sum : ∃ (x₀ y₀ : ℕ), 
  x₀ < 29 ∧ y₀ < 29 ∧
  (∀ x : ℤ, (4 * x) % 29 = (5 * 0 - 1) % 29 ↔ x % 29 = x₀) ∧
  (∀ y : ℤ, (4 * 0) % 29 = (5 * y - 1) % 29 ↔ y % 29 = y₀) ∧
  x₀ + y₀ = 30 :=
by sorry

end mod_29_graph_intercepts_sum_l1009_100998


namespace same_color_probability_is_two_twentyfifths_l1009_100976

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (purple : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (silver : ℕ)
  (total : ℕ)
  (h_total : purple + green + blue + silver = total)

/-- The probability of getting the same color on all three dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.purple : ℚ) ^ 3 / d.total ^ 3 +
  (d.green : ℚ) ^ 3 / d.total ^ 3 +
  (d.blue : ℚ) ^ 3 / d.total ^ 3 +
  (d.silver : ℚ) ^ 3 / d.total ^ 3

/-- The specific die configuration in the problem -/
def problem_die : ColoredDie :=
  { purple := 6
  , green := 8
  , blue := 10
  , silver := 6
  , total := 30
  , h_total := by simp }

/-- Theorem stating the probability of getting the same color on all three dice -/
theorem same_color_probability_is_two_twentyfifths :
  same_color_probability problem_die = 2 / 25 := by
  sorry

end same_color_probability_is_two_twentyfifths_l1009_100976


namespace zeros_properties_l1009_100918

noncomputable def f (θ : ℝ) : ℝ := Real.sin (4 * θ) + Real.sin (3 * θ)

theorem zeros_properties (θ₁ θ₂ θ₃ : ℝ) 
  (h1 : 0 < θ₁ ∧ θ₁ < π) 
  (h2 : 0 < θ₂ ∧ θ₂ < π) 
  (h3 : 0 < θ₃ ∧ θ₃ < π) 
  (h4 : θ₁ ≠ θ₂ ∧ θ₁ ≠ θ₃ ∧ θ₂ ≠ θ₃) 
  (h5 : f θ₁ = 0) 
  (h6 : f θ₂ = 0) 
  (h7 : f θ₃ = 0) : 
  (θ₁ + θ₂ + θ₃ = 12 * π / 7) ∧ 
  (Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ = 1 / 8) ∧ 
  (Real.cos θ₁ + Real.cos θ₂ + Real.cos θ₃ = -1 / 2) :=
by sorry

end zeros_properties_l1009_100918


namespace function_symmetry_and_value_l1009_100960

/-- Given a function f(x) = 2cos(ωx + φ) + m with ω > 0, 
    if f(π/4 - t) = f(t) for all real t and f(π/8) = -1, 
    then m = -3 or m = 1 -/
theorem function_symmetry_and_value (ω φ m : ℝ) (h_ω : ω > 0) 
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.cos (ω * x + φ) + m) 
  (h_sym : ∀ t, f (π/4 - t) = f t) (h_val : f (π/8) = -1) : 
  m = -3 ∨ m = 1 := by
  sorry

end function_symmetry_and_value_l1009_100960


namespace integer_sum_difference_product_square_difference_l1009_100931

theorem integer_sum_difference_product_square_difference 
  (a b : ℕ+) 
  (sum_eq : a + b = 40)
  (diff_eq : a - b = 8) : 
  a * b = 384 ∧ a^2 - b^2 = 320 := by
sorry

end integer_sum_difference_product_square_difference_l1009_100931


namespace game_boxes_needed_l1009_100917

theorem game_boxes_needed (initial_games : ℕ) (sold_games : ℕ) (games_per_box : ℕ) : 
  initial_games = 76 → sold_games = 46 → games_per_box = 5 → 
  (initial_games - sold_games) / games_per_box = 6 := by
  sorry

end game_boxes_needed_l1009_100917


namespace ball_bounce_distance_l1009_100986

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := Finset.sum (Finset.range (bounces + 1)) (fun i => initialHeight * reboundFactor^i)
  let ascendDistances := Finset.sum (Finset.range bounces) (fun i => initialHeight * reboundFactor^(i+1))
  descendDistances + ascendDistances

/-- Theorem: The total distance traveled by a ball dropped from 25 meters,
    rebounding to 2/3 of its previous height for four bounces, is 1900/27 meters -/
theorem ball_bounce_distance :
  totalDistance 25 (2/3) 4 = 1900/27 := by
  sorry

end ball_bounce_distance_l1009_100986


namespace min_value_sum_l1009_100949

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + 2*a*b + 4*b*c + 2*c*a = 16) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    x^2 + 2*x*y + 4*y*z + 2*z*x = 16 → x + y + z ≥ m :=
by sorry

end min_value_sum_l1009_100949


namespace supplement_of_complement_of_35_degrees_l1009_100941

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125°. -/
theorem supplement_of_complement_of_35_degrees :
  let original_angle : ℝ := 35
  let complement : ℝ := 90 - original_angle
  let supplement : ℝ := 180 - complement
  supplement = 125 := by
  sorry

end supplement_of_complement_of_35_degrees_l1009_100941


namespace problem_1_problem_2_problem_3_l1009_100990

-- Problem 1
theorem problem_1 (a b c : ℚ) : (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = 3/2 * b * c := by sorry

-- Problem 2
theorem problem_2 (m n : ℚ) : (-3*m - 2*n) * (3*m + 2*n) = -9*m^2 - 12*m*n - 4*n^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℚ) (h : y ≠ 0) : ((x - 2*y)^2 - (x - 2*y)*(x + 2*y)) / (2*y) = -2*x + 4*y := by sorry

end problem_1_problem_2_problem_3_l1009_100990


namespace equation_system_solution_l1009_100916

/-- A system of 1000 equations where each x_i^2 = a * x_{i+1} + 1, with x_1000 wrapping back to x_1 -/
def EquationSystem (a : ℝ) (x : Fin 1000 → ℝ) : Prop :=
  ∀ i : Fin 1000, x i ^ 2 = a * x (i.succ) + 1

/-- The solutions to the equation system -/
def Solutions (a : ℝ) : Set ℝ :=
  {x | x = (a + Real.sqrt (a^2 + 4)) / 2 ∨ x = (a - Real.sqrt (a^2 + 4)) / 2}

theorem equation_system_solution (a : ℝ) (ha : |a| > 1) :
  ∀ x : Fin 1000 → ℝ, EquationSystem a x ↔ (∀ i, x i ∈ Solutions a) := by
  sorry

end equation_system_solution_l1009_100916


namespace subtracted_number_l1009_100968

theorem subtracted_number (x : ℤ) (y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  sorry

end subtracted_number_l1009_100968


namespace parallel_condition_l1009_100981

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_condition 
  (α β : Plane) 
  (m : Line) 
  (h_distinct : α ≠ β) 
  (h_m_in_α : line_in_plane m α) : 
  (line_parallel_plane m β → plane_parallel α β) ∧ 
  ¬(plane_parallel α β → line_parallel_plane m β) :=
sorry

end parallel_condition_l1009_100981


namespace apple_problem_l1009_100964

/-- Represents the types of Red Fuji apples --/
inductive AppleType
  | A
  | B

/-- Represents the purchase and selling prices for each apple type --/
def price (t : AppleType) : ℕ × ℕ :=
  match t with
  | AppleType.A => (28, 42)
  | AppleType.B => (22, 34)

/-- Represents the total number of apples purchased in the first batch --/
def totalApples : ℕ := 30

/-- Represents the total cost of the first batch of apples --/
def totalCost : ℕ := 720

/-- Represents the maximum number of apples to be purchased in the second batch --/
def maxApples : ℕ := 80

/-- Represents the maximum cost allowed for the second batch --/
def maxCost : ℕ := 2000

/-- Represents the initial daily sales of type B apples at original price --/
def initialSales : ℕ := 4

/-- Represents the increase in daily sales for every 1 yuan price reduction --/
def salesIncrease : ℕ := 2

/-- Represents the target daily profit for type B apples --/
def targetProfit : ℕ := 90

theorem apple_problem :
  ∃ (x y : ℕ),
    x + y = totalApples ∧
    x * (price AppleType.A).1 + y * (price AppleType.B).1 = totalCost ∧
    ∃ (m : ℕ),
      m ≤ maxApples ∧
      m * (price AppleType.A).1 + (maxApples - m) * (price AppleType.B).1 ≤ maxCost ∧
      ∀ (k : ℕ),
        k ≤ maxApples →
        k * (price AppleType.A).1 + (maxApples - k) * (price AppleType.B).1 ≤ maxCost →
        m * ((price AppleType.A).2 - (price AppleType.A).1) +
          (maxApples - m) * ((price AppleType.B).2 - (price AppleType.B).1) ≥
        k * ((price AppleType.A).2 - (price AppleType.A).1) +
          (maxApples - k) * ((price AppleType.B).2 - (price AppleType.B).1) ∧
    ∃ (a : ℕ),
      (initialSales + salesIncrease * a) * ((price AppleType.B).2 - a - (price AppleType.B).1) = targetProfit :=
by
  sorry


end apple_problem_l1009_100964


namespace jack_emails_l1009_100969

theorem jack_emails (morning_emails : ℕ) (difference : ℕ) (afternoon_emails : ℕ) : 
  morning_emails = 6 → 
  difference = 4 → 
  morning_emails = afternoon_emails + difference → 
  afternoon_emails = 2 := by
sorry

end jack_emails_l1009_100969


namespace fraction_simplest_form_l1009_100919

theorem fraction_simplest_form (n : ℤ) : Int.gcd (39*n + 4) (26*n + 3) = 1 := by
  sorry

end fraction_simplest_form_l1009_100919


namespace hyperbola_asymptotes_l1009_100915

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x ∨ y = -(Real.sqrt 5 / 2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(√5/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
by sorry

end hyperbola_asymptotes_l1009_100915


namespace min_value_arithmetic_sequence_l1009_100930

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  (a 4 * a 14 = 8) →
  (∀ x y : ℝ, 2 * a 7 + a 11 ≥ x + y → x * y ≤ 16) :=
by sorry

end min_value_arithmetic_sequence_l1009_100930


namespace mans_to_sons_age_ratio_l1009_100971

/-- Given a man who is 28 years older than his son, and the son's present age is 26,
    prove that the ratio of the man's age to his son's age in two years is 2:1. -/
theorem mans_to_sons_age_ratio (son_age : ℕ) (man_age : ℕ) : 
  son_age = 26 → 
  man_age = son_age + 28 → 
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end mans_to_sons_age_ratio_l1009_100971


namespace repeating_decimal_sum_l1009_100953

theorem repeating_decimal_sum : 
  ∃ (a b c : ℚ), 
    (∀ n : ℕ, a = (234 : ℚ) / 10^3 + (234 : ℚ) / (999 * 10^n)) ∧
    (∀ n : ℕ, b = (567 : ℚ) / 10^3 + (567 : ℚ) / (999 * 10^n)) ∧
    (∀ n : ℕ, c = (891 : ℚ) / 10^3 + (891 : ℚ) / (999 * 10^n)) ∧
    a - b + c = 186 / 333 :=
by sorry

end repeating_decimal_sum_l1009_100953


namespace odd_function_domain_symmetry_l1009_100905

/-- A function f is odd if its domain is symmetric about the origin -/
def is_odd_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x ∈ domain, -x ∈ domain

/-- The domain of the function -/
def function_domain (t : ℝ) : Set ℝ := Set.Ioo t (2*t + 3)

/-- Theorem: If f is an odd function with domain (t, 2t+3), then t = -1 -/
theorem odd_function_domain_symmetry (f : ℝ → ℝ) (t : ℝ) 
  (h : is_odd_function f (function_domain t)) : 
  t = -1 := by
  sorry

end odd_function_domain_symmetry_l1009_100905


namespace no_more_birds_can_join_l1009_100939

/-- Represents the weight capacity of the fence in pounds -/
def fence_capacity : ℝ := 20

/-- Represents the weight of the first bird in pounds -/
def bird1_weight : ℝ := 2.5

/-- Represents the weight of the second bird in pounds -/
def bird2_weight : ℝ := 3.5

/-- Represents the number of additional birds that joined -/
def additional_birds : ℕ := 4

/-- Represents the weight of each additional bird in pounds -/
def additional_bird_weight : ℝ := 2.8

/-- Represents the weight of each new bird that might join in pounds -/
def new_bird_weight : ℝ := 3

/-- Calculates the total weight of birds currently on the fence -/
def current_weight : ℝ := bird1_weight + bird2_weight + additional_birds * additional_bird_weight

/-- Theorem stating that no more 3 lb birds can join the fence without exceeding its capacity -/
theorem no_more_birds_can_join : 
  ∀ n : ℕ, current_weight + n * new_bird_weight ≤ fence_capacity → n = 0 := by
  sorry

end no_more_birds_can_join_l1009_100939


namespace sin_cube_identity_l1009_100932

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = -(1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end sin_cube_identity_l1009_100932


namespace line_mb_equals_nine_l1009_100974

/-- Given a line with equation y = mx + b that intersects the y-axis at y = -3
    and rises 3 units for every 1 unit to the right, prove that mb = 9. -/
theorem line_mb_equals_nine (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  b = -3 →                      -- y-intercept
  (∀ x : ℝ, m * (x + 1) + b = m * x + b + 3) →  -- Slope condition
  m * b = 9 := by
sorry

end line_mb_equals_nine_l1009_100974


namespace probability_of_even_sum_l1009_100948

def M : Finset ℕ := {1, 2, 3}
def N : Finset ℕ := {4, 5, 6}

def is_sum_even (x : ℕ) (y : ℕ) : Bool := Even (x + y)

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (M.product N).filter (fun (x, y) => is_sum_even x y)

theorem probability_of_even_sum :
  (favorable_outcomes.card : ℚ) / ((M.card * N.card) : ℚ) = 4 / 9 := by
  sorry

end probability_of_even_sum_l1009_100948


namespace mans_speed_against_current_l1009_100942

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: Given the conditions, prove that the man's speed against the current is 10 km/hr -/
theorem mans_speed_against_current :
  let speed_with_current : ℝ := 15
  let current_speed : ℝ := 2.5
  speed_against_current speed_with_current current_speed = 10 := by
  sorry

end mans_speed_against_current_l1009_100942


namespace tammy_mountain_climb_l1009_100910

/-- Proves that given the conditions of Tammy's mountain climb, her speed on the second day was 4 km/h -/
theorem tammy_mountain_climb (total_time : ℝ) (total_distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ)
  (h1 : total_time = 14)
  (h2 : speed_increase = 0.5)
  (h3 : time_decrease = 2)
  (h4 : total_distance = 52) :
  ∃ (speed1 : ℝ) (time1 : ℝ),
    speed1 > 0 ∧
    time1 > 0 ∧
    time1 + (time1 - time_decrease) = total_time ∧
    speed1 * time1 + (speed1 + speed_increase) * (time1 - time_decrease) = total_distance ∧
    speed1 + speed_increase = 4 := by
  sorry

end tammy_mountain_climb_l1009_100910


namespace no_valid_N_l1009_100936

theorem no_valid_N : ¬ ∃ (N P R : ℕ), 
  N < 40 ∧ 
  P + R = N ∧ 
  (71 * P + 56 * R : ℚ) / N = 66 ∧
  (76 * P : ℚ) / P = 75 ∧
  (61 * R : ℚ) / R = 59 ∧
  P = 2 * R :=
by sorry

end no_valid_N_l1009_100936


namespace sqrt_144_div_6_l1009_100909

theorem sqrt_144_div_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end sqrt_144_div_6_l1009_100909


namespace eighth_fibonacci_is_21_l1009_100927

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem eighth_fibonacci_is_21 : fibonacci 7 = 21 := by
  sorry

end eighth_fibonacci_is_21_l1009_100927


namespace problem_solution_l1009_100987

theorem problem_solution :
  (∃ (x y : ℝ),
    (x = (3 * Real.sqrt 18 + (1/5) * Real.sqrt 50 - 4 * Real.sqrt (1/2)) / Real.sqrt 32 ∧ x = 2) ∧
    (y = (1 + Real.sqrt 2 + Real.sqrt 3) * (1 + Real.sqrt 2 - Real.sqrt 3) ∧ y = 2 * Real.sqrt 2)) :=
by sorry

end problem_solution_l1009_100987


namespace pure_imaginary_square_root_l1009_100920

theorem pure_imaginary_square_root (a : ℝ) : 
  (∃ (b : ℝ), (1 + a * Complex.I)^2 = b * Complex.I) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_square_root_l1009_100920


namespace sum_reciprocal_lower_bound_l1009_100979

theorem sum_reciprocal_lower_bound (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (h_sum : a₁ + a₂ + a₃ + a₄ = 1) : 
  1/a₁ + 1/a₂ + 1/a₃ + 1/a₄ ≥ 16 := by
  sorry

end sum_reciprocal_lower_bound_l1009_100979


namespace shopkeeper_decks_l1009_100937

-- Define the number of red cards the shopkeeper has
def total_red_cards : ℕ := 208

-- Define the number of cards in a standard deck
def cards_per_deck : ℕ := 52

-- Theorem stating the number of decks the shopkeeper has
theorem shopkeeper_decks : 
  total_red_cards / cards_per_deck = 4 := by
  sorry

end shopkeeper_decks_l1009_100937


namespace degree_to_seconds_one_point_four_five_deg_to_seconds_l1009_100925

theorem degree_to_seconds (deg : Real) (min_per_deg : Nat) (sec_per_min : Nat) 
  (h1 : min_per_deg = 60) (h2 : sec_per_min = 60) :
  deg * (min_per_deg * sec_per_min) = deg * 3600 := by
  sorry

theorem one_point_four_five_deg_to_seconds :
  1.45 * 3600 = 5220 := by
  sorry

end degree_to_seconds_one_point_four_five_deg_to_seconds_l1009_100925


namespace factory_production_l1009_100980

/-- Calculates the total number of products produced by a factory in 5 days -/
def total_products_in_five_days (refrigerators_per_hour : ℕ) (coolers_difference : ℕ) (hours_per_day : ℕ) : ℕ :=
  let coolers_per_hour := refrigerators_per_hour + coolers_difference
  let products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_hours := 5 * hours_per_day
  products_per_hour * total_hours

/-- Theorem stating that the factory produces 11250 products in 5 days -/
theorem factory_production :
  total_products_in_five_days 90 70 9 = 11250 :=
by
  sorry

end factory_production_l1009_100980


namespace leading_digit_logarithm_l1009_100900

-- Define a function to get the leading digit of a real number
noncomputable def leadingDigit (x : ℝ) : ℕ := sorry

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := sorry

-- State the theorem
theorem leading_digit_logarithm (M : ℝ) (a : ℕ) :
  (leadingDigit (6 * 47 * log10 M) = a) →
  ((leadingDigit (log10 (1000 / M)) = 3 - a) ∨
   (leadingDigit (log10 (1000 / M)) = 2 - a)) :=
by sorry

end leading_digit_logarithm_l1009_100900


namespace ratio_transformation_l1009_100993

theorem ratio_transformation (y : ℚ) : 
  (2 + y : ℚ) / (3 + y) = 4 / 5 → (2 + y = 4 ∧ 3 + y = 5) := by
  sorry

#check ratio_transformation

end ratio_transformation_l1009_100993


namespace melissa_games_played_l1009_100902

def total_points : ℕ := 81
def points_per_game : ℕ := 27

theorem melissa_games_played :
  total_points / points_per_game = 3 := by sorry

end melissa_games_played_l1009_100902


namespace cake_baking_fraction_l1009_100970

theorem cake_baking_fraction (total_cakes : ℕ) 
  (h1 : total_cakes = 60) 
  (initially_baked : ℕ) 
  (h2 : initially_baked = total_cakes / 2) 
  (first_day_baked : ℕ) 
  (h3 : first_day_baked = (total_cakes - initially_baked) / 2) 
  (second_day_baked : ℕ) 
  (h4 : total_cakes - initially_baked - first_day_baked - second_day_baked = 10) :
  (second_day_baked : ℚ) / (total_cakes - initially_baked - first_day_baked) = 1 / 3 := by
sorry

end cake_baking_fraction_l1009_100970


namespace real_roots_of_polynomial_l1009_100904

theorem real_roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end real_roots_of_polynomial_l1009_100904


namespace square_minus_four_times_product_specific_calculation_l1009_100926

theorem square_minus_four_times_product (a b : ℕ) : 
  (a + b) ^ 2 - 4 * a * b = (a - b) ^ 2 :=
by sorry

theorem specific_calculation : 
  (476 + 424) ^ 2 - 4 * 476 * 424 = 5776 :=
by sorry

end square_minus_four_times_product_specific_calculation_l1009_100926


namespace binomial_self_one_binomial_600_600_l1009_100991

theorem binomial_self_one (n : ℕ) : Nat.choose n n = 1 := by sorry

theorem binomial_600_600 : Nat.choose 600 600 = 1 := by sorry

end binomial_self_one_binomial_600_600_l1009_100991


namespace g_range_l1009_100973

noncomputable def g (x : ℝ) : ℝ := Real.arctan (x^3) + Real.arctan ((1 - x^3) / (1 + x^3))

theorem g_range :
  Set.range g = Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) :=
sorry

end g_range_l1009_100973


namespace largest_c_for_negative_five_in_range_l1009_100988

theorem largest_c_for_negative_five_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, x^2 + 4*x + c = -5) ↔ c ≤ -1 :=
by sorry

end largest_c_for_negative_five_in_range_l1009_100988


namespace complex_exponential_sum_l1009_100922

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (2 / 5 : ℂ) + (4 / 9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (2 / 5 : ℂ) - (4 / 9 : ℂ) * Complex.I := by
  sorry

end complex_exponential_sum_l1009_100922


namespace increasing_interval_of_f_l1009_100911

-- Define the function
def f (x : ℝ) : ℝ := -(x - 2) * x

-- State the theorem
theorem increasing_interval_of_f :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x < f y :=
by sorry

end increasing_interval_of_f_l1009_100911


namespace general_term_formula_l1009_100984

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n - 2 * a (n + 1) + a (n + 2) = 0

/-- Theorem stating the general term formula for the sequence -/
theorem general_term_formula (a : ℕ+ → ℝ) 
    (h_recurrence : RecurrenceSequence a) 
    (h_initial1 : a 1 = 2) 
    (h_initial2 : a 2 = 4) : 
    ∀ n : ℕ+, a n = 2 * n := by
  sorry

end general_term_formula_l1009_100984


namespace ratio_equation_solution_l1009_100943

theorem ratio_equation_solution (a b c : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 15 - 4 * a - c)
  (h3 : c = a + 2) :
  a = 13 / 9 := by
sorry

end ratio_equation_solution_l1009_100943


namespace two_props_true_l1009_100938

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 5 → x^2 - 8*x + 15 = 0

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x^2 - 8*x + 15 = 0 → x = 5

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x ≠ 5 → x^2 - 8*x + 15 ≠ 0

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x^2 - 8*x + 15 ≠ 0 → x ≠ 5

-- Theorem stating that exactly two propositions (including the original) are true
theorem two_props_true :
  (∀ x, original_prop x) ∧
  (¬ ∀ x, converse_prop x) ∧
  (¬ ∀ x, inverse_prop x) ∧
  (∀ x, contrapositive_prop x) :=
sorry

end two_props_true_l1009_100938


namespace power_of_power_l1009_100985

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l1009_100985
