import Mathlib

namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_seven_fortieths_l3187_318726

theorem ten_thousandths_place_of_seven_fortieths (n : ℕ) : 
  (7 : ℚ) / 40 * 10000 - ((7 : ℚ) / 40 * 10000).floor = (0 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_seven_fortieths_l3187_318726


namespace NUMINAMATH_CALUDE_no_three_fractions_product_one_l3187_318725

theorem no_three_fractions_product_one :
  ¬ ∃ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 100 ∧
    (a : ℚ) / (101 - a) * (b : ℚ) / (101 - b) * (c : ℚ) / (101 - c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_three_fractions_product_one_l3187_318725


namespace NUMINAMATH_CALUDE_travis_cereal_cost_l3187_318753

/-- The amount Travis spends on cereal in a year -/
def cereal_cost (boxes_per_week : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (boxes_per_week * weeks_per_year : ℚ) * cost_per_box

/-- Theorem: Travis spends $312.00 on cereal in a year -/
theorem travis_cereal_cost :
  cereal_cost 2 3 52 = 312 := by
  sorry

#eval cereal_cost 2 3 52

end NUMINAMATH_CALUDE_travis_cereal_cost_l3187_318753


namespace NUMINAMATH_CALUDE_expression_equality_l3187_318704

theorem expression_equality (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 2*x + y/2 ≠ 0) : 
  (2*x + y/2)⁻¹ * ((2*x)⁻¹ + (y/2)⁻¹) = (x*y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3187_318704


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_herons_l3187_318706

theorem geese_percentage_among_non_herons :
  let total_birds : ℝ := 100
  let geese_percentage : ℝ := 30
  let swans_percentage : ℝ := 25
  let herons_percentage : ℝ := 20
  let ducks_percentage : ℝ := 25
  let non_heron_percentage : ℝ := total_birds - herons_percentage
  geese_percentage / non_heron_percentage * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_herons_l3187_318706


namespace NUMINAMATH_CALUDE_pickle_theorem_l3187_318741

def pickle_problem (sammy_slices tammy_slices ron_slices : ℕ) : Prop :=
  tammy_slices = 2 * sammy_slices →
  sammy_slices = 15 →
  ron_slices = 24 →
  (tammy_slices - ron_slices : ℚ) / tammy_slices * 100 = 20

theorem pickle_theorem : pickle_problem 15 30 24 := by sorry

end NUMINAMATH_CALUDE_pickle_theorem_l3187_318741


namespace NUMINAMATH_CALUDE_leigh_has_16_shells_l3187_318745

-- Define the number of seashells each person has
def mimi_shells : ℕ := 24  -- 2 dozen seashells
def kyle_shells : ℕ := 2 * mimi_shells
def leigh_shells : ℕ := kyle_shells / 3

-- Theorem to prove
theorem leigh_has_16_shells : leigh_shells = 16 := by
  sorry

end NUMINAMATH_CALUDE_leigh_has_16_shells_l3187_318745


namespace NUMINAMATH_CALUDE_exam_average_problem_l3187_318763

theorem exam_average_problem (n : ℕ) : 
  (15 : ℝ) * 75 + (10 : ℝ) * 90 = (n : ℝ) * 81 → n = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_average_problem_l3187_318763


namespace NUMINAMATH_CALUDE_number_problem_l3187_318749

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 10 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3187_318749


namespace NUMINAMATH_CALUDE_odd_function_sum_l3187_318734

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 8) :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3187_318734


namespace NUMINAMATH_CALUDE_mary_final_weight_l3187_318789

def weight_change (initial_weight : ℕ) : ℕ :=
  let first_loss := 12
  let second_gain := 2 * first_loss
  let third_loss := 3 * first_loss
  let final_gain := 6
  initial_weight - first_loss + second_gain - third_loss + final_gain

theorem mary_final_weight (initial_weight : ℕ) (h : initial_weight = 99) :
  weight_change initial_weight = 81 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_weight_l3187_318789


namespace NUMINAMATH_CALUDE_existence_of_unsolvable_degree_l3187_318727

-- Define a polynomial equation of degree n
def PolynomialEquation (n : ℕ) := ℕ → ℝ → Prop

-- Define a solution expressed in terms of radicals
def RadicalSolution (n : ℕ) := ℕ → ℝ → Prop

-- Axiom: Quadratic equations have solutions in terms of radicals
axiom quadratic_solvable : ∀ (eq : PolynomialEquation 2), ∃ (sol : RadicalSolution 2), sol 2 = eq 2

-- Axiom: Cubic equations have solutions in terms of radicals
axiom cubic_solvable : ∀ (eq : PolynomialEquation 3), ∃ (sol : RadicalSolution 3), sol 3 = eq 3

-- Axiom: Quartic equations have solutions in terms of radicals
axiom quartic_solvable : ∀ (eq : PolynomialEquation 4), ∃ (sol : RadicalSolution 4), sol 4 = eq 4

-- Theorem: There exists a degree n such that not all polynomial equations of degree ≥ n are solvable by radicals
theorem existence_of_unsolvable_degree :
  ∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → ¬(∀ (eq : PolynomialEquation m), ∃ (sol : RadicalSolution m), sol m = eq m) :=
sorry

end NUMINAMATH_CALUDE_existence_of_unsolvable_degree_l3187_318727


namespace NUMINAMATH_CALUDE_car_speed_proof_l3187_318781

/-- Proves that a car traveling at speed v km/h takes 2 seconds longer to travel 1 kilometer
    than it would at 225 km/h if and only if v = 200 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v * 3600 = 1 / 225 * 3600 + 2) ↔ v = 200 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3187_318781


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3187_318752

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 8 * x + 3 > 0 ↔ x < -1/3 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3187_318752


namespace NUMINAMATH_CALUDE_john_gave_twenty_l3187_318767

/-- The amount of money John gave to the store for buying Slurpees -/
def money_given (cost_per_slurpee : ℕ) (num_slurpees : ℕ) (change_received : ℕ) : ℕ :=
  cost_per_slurpee * num_slurpees + change_received

/-- Proof that John gave $20 to the store -/
theorem john_gave_twenty :
  money_given 2 6 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_gave_twenty_l3187_318767


namespace NUMINAMATH_CALUDE_participant_selection_count_l3187_318784

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_participants : ℕ := 4

def select_participants (boys girls participants : ℕ) : ℕ :=
  (Nat.choose boys 3 * Nat.choose girls 1) +
  (Nat.choose boys 2 * Nat.choose girls 2) +
  (Nat.choose boys 1 * Nat.choose girls 3)

theorem participant_selection_count :
  select_participants num_boys num_girls num_participants = 34 := by
  sorry

end NUMINAMATH_CALUDE_participant_selection_count_l3187_318784


namespace NUMINAMATH_CALUDE_stratified_sampling_young_teachers_l3187_318742

theorem stratified_sampling_young_teachers 
  (total_teachers : ℕ) 
  (young_teachers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_teachers = 200)
  (h2 : young_teachers = 100)
  (h3 : sample_size = 40) :
  (young_teachers : ℚ) / (total_teachers : ℚ) * (sample_size : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_young_teachers_l3187_318742


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_x_plus_x_cubed_l3187_318777

theorem integral_sqrt_plus_x_plus_x_cubed (f : ℝ → ℝ) :
  (∫ x in (0)..(1), (Real.sqrt (1 - x^2) + x + x^3)) = (π + 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_x_plus_x_cubed_l3187_318777


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l3187_318731

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_base_for_perfect_square :
  ∀ b : ℕ, b > 6 →
    (∀ k : ℕ, k > 6 ∧ k < b → ¬ is_perfect_square (4 * k + 5)) →
    is_perfect_square (4 * b + 5) →
    b = 11 := by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l3187_318731


namespace NUMINAMATH_CALUDE_correct_answer_is_nothing_l3187_318729

/-- Represents the possible answers to the Question of Questions -/
inductive Answer
| Something
| Nothing

/-- Represents a priest -/
structure Priest where
  knowsAnswer : Bool
  alwaysLies : Bool

/-- The response given by a priest -/
def priestResponse : Answer := Answer.Something

/-- Theorem: If a priest who knows the correct answer responds with "Something exists,"
    then the correct answer is "Nothing exists" -/
theorem correct_answer_is_nothing 
  (priest : Priest) 
  (h1 : priest.knowsAnswer = true) 
  (h2 : priest.alwaysLies = true) 
  (h3 : priestResponse = Answer.Something) : 
  Answer.Nothing = Answer.Nothing := by sorry


end NUMINAMATH_CALUDE_correct_answer_is_nothing_l3187_318729


namespace NUMINAMATH_CALUDE_square_difference_formula_l3187_318740

theorem square_difference_formula (a b : ℚ) 
  (sum_eq : a + b = 11/17) 
  (diff_eq : a - b = 1/143) : 
  a^2 - b^2 = 11/2431 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l3187_318740


namespace NUMINAMATH_CALUDE_tom_catches_jerry_l3187_318797

/-- Represents the figure-eight track -/
structure Track :=
  (small_loop : ℝ)
  (large_loop : ℝ)
  (h_large_double_small : large_loop = 2 * small_loop)

/-- Represents the runners -/
structure Runner :=
  (speed : ℝ)

theorem tom_catches_jerry (track : Track) (tom jerry : Runner) 
  (h1 : tom.speed = track.small_loop / 10)
  (h2 : jerry.speed = track.small_loop / 20)
  (h3 : tom.speed = 2 * jerry.speed) :
  (2 * track.large_loop) / (tom.speed - jerry.speed) = 40 := by
  sorry

#check tom_catches_jerry

end NUMINAMATH_CALUDE_tom_catches_jerry_l3187_318797


namespace NUMINAMATH_CALUDE_division_problem_l3187_318779

theorem division_problem (smaller larger : ℕ) : 
  larger - smaller = 1395 →
  larger = 1656 →
  larger % smaller = 15 →
  larger / smaller = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3187_318779


namespace NUMINAMATH_CALUDE_f_minimum_and_a_range_l3187_318787

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_minimum_and_a_range :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -1 / Real.exp 1) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 1 → f x ≥ a * x - 1) ↔ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_and_a_range_l3187_318787


namespace NUMINAMATH_CALUDE_card_number_solution_l3187_318770

theorem card_number_solution : ∃ (L O M N S V : ℕ), 
  (L < 10) ∧ (O < 10) ∧ (M < 10) ∧ (N < 10) ∧ (S < 10) ∧ (V < 10) ∧
  (L ≠ O) ∧ (L ≠ M) ∧ (L ≠ N) ∧ (L ≠ S) ∧ (L ≠ V) ∧
  (O ≠ M) ∧ (O ≠ N) ∧ (O ≠ S) ∧ (O ≠ V) ∧
  (M ≠ N) ∧ (M ≠ S) ∧ (M ≠ V) ∧
  (N ≠ S) ∧ (N ≠ V) ∧
  (S ≠ V) ∧
  (0 < O) ∧ (O < M) ∧ (O < S) ∧
  (L + O * S + O * M + N * M * S + O * M = 10 * M * S + V * M * S) :=
by sorry


end NUMINAMATH_CALUDE_card_number_solution_l3187_318770


namespace NUMINAMATH_CALUDE_triangle_angle_ranges_l3187_318711

def triangle_angles (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

theorem triangle_angle_ranges (α β γ : Real) (h : triangle_angles α β γ) :
  60 ≤ max α (max β γ) ∧ max α (max β γ) < 180 ∧
  0 < min α (min β γ) ∧ min α (min β γ) ≤ 60 ∧
  0 < (max (min α β) (min (max α β) γ)) ∧ (max (min α β) (min (max α β) γ)) < 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ranges_l3187_318711


namespace NUMINAMATH_CALUDE_divisibility_by_11_l3187_318713

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def seven_digit_number (m : ℕ) : ℕ :=
  856 * 10000 + m * 1000 + 248

theorem divisibility_by_11 (m : ℕ) : 
  is_divisible_by_11 (seven_digit_number m) ↔ m = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_11_l3187_318713


namespace NUMINAMATH_CALUDE_min_value_problem_l3187_318744

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3187_318744


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3187_318738

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpLine : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpPlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : perpLine a b) 
  (h2 : perpPlane a α) : 
  subset b α ∨ parallel b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3187_318738


namespace NUMINAMATH_CALUDE_book_reading_percentage_l3187_318733

theorem book_reading_percentage (total_pages : ℕ) (remaining_pages : ℕ) : 
  total_pages = 400 → remaining_pages = 320 → 
  (((total_pages - remaining_pages) : ℚ) / total_pages) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_book_reading_percentage_l3187_318733


namespace NUMINAMATH_CALUDE_cola_sales_count_l3187_318760

/-- Represents the number of bottles sold for each drink type -/
structure DrinkSales where
  cola : ℕ
  juice : ℕ
  water : ℕ

/-- Calculates the total earnings from drink sales -/
def totalEarnings (sales : DrinkSales) : ℚ :=
  3 * sales.cola + 1.5 * sales.juice + 1 * sales.water

/-- Theorem stating that the number of cola bottles sold is 15 -/
theorem cola_sales_count : ∃ (sales : DrinkSales), 
  sales.juice = 12 ∧ 
  sales.water = 25 ∧ 
  totalEarnings sales = 88 ∧ 
  sales.cola = 15 := by
  sorry

end NUMINAMATH_CALUDE_cola_sales_count_l3187_318760


namespace NUMINAMATH_CALUDE_one_non_negative_root_l3187_318705

theorem one_non_negative_root (a : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ (x = a + Real.sqrt (a^2 - 4*a + 3) ∨ x = a - Real.sqrt (a^2 - 4*a + 3)) ∧
   ¬∃ y : ℝ, y ≠ x ∧ y ≥ 0 ∧ (y = a + Real.sqrt (a^2 - 4*a + 3) ∨ y = a - Real.sqrt (a^2 - 4*a + 3))) ↔ 
  ((3/4 ≤ a ∧ a < 1) ∨ (a > 3) ∨ (0 < a ∧ a < 3/4)) :=
by sorry

end NUMINAMATH_CALUDE_one_non_negative_root_l3187_318705


namespace NUMINAMATH_CALUDE_customer_satisfaction_probability_l3187_318796

/-- The probability that a dissatisfied customer leaves an angry review -/
def prob_angry_given_dissatisfied : ℝ := 0.8

/-- The probability that a satisfied customer leaves a positive review -/
def prob_positive_given_satisfied : ℝ := 0.15

/-- The number of angry reviews received -/
def angry_reviews : ℕ := 60

/-- The number of positive reviews received -/
def positive_reviews : ℕ := 20

/-- The probability that a customer is satisfied with the service -/
def prob_satisfied : ℝ := 0.64

theorem customer_satisfaction_probability :
  prob_satisfied = 0.64 :=
sorry

end NUMINAMATH_CALUDE_customer_satisfaction_probability_l3187_318796


namespace NUMINAMATH_CALUDE_oranges_per_box_l3187_318728

/-- Given 24 oranges distributed equally among 3 boxes, prove that each box contains 8 oranges. -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : total_oranges = 24) 
  (h2 : num_boxes = 3) 
  (h3 : oranges_per_box * num_boxes = total_oranges) : 
  oranges_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3187_318728


namespace NUMINAMATH_CALUDE_absent_children_l3187_318776

theorem absent_children (total_children : ℕ) (bananas : ℕ) (absent : ℕ) : 
  total_children = 840 →
  bananas = 840 * 2 →
  bananas = (840 - absent) * 4 →
  absent = 420 := by
sorry

end NUMINAMATH_CALUDE_absent_children_l3187_318776


namespace NUMINAMATH_CALUDE_exists_intersecting_line_no_circle_through_origin_l3187_318730

-- Define the set of circles C_k
def C_k (k : ℕ+) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (k - 1))^2 + (p.2 - 3*k)^2 = 2*k^4}

-- Statement 1: There exists a fixed line that intersects all circles
theorem exists_intersecting_line :
  ∃ (m b : ℝ), ∀ (k : ℕ+), ∃ (x y : ℝ), (y = m*x + b) ∧ (x, y) ∈ C_k k :=
sorry

-- Statement 2: No circle passes through the origin
theorem no_circle_through_origin :
  ∀ (k : ℕ+), (0, 0) ∉ C_k k :=
sorry

end NUMINAMATH_CALUDE_exists_intersecting_line_no_circle_through_origin_l3187_318730


namespace NUMINAMATH_CALUDE_triangle_law_of_sines_l3187_318739

theorem triangle_law_of_sines (A B : ℝ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : 0 ≤ A) (h4 : A < π) (h5 : 0 ≤ B) (h6 : B < π) :
  a = 3 → b = 5 → Real.sin A = 1/3 → Real.sin B = 5/9 := by sorry

end NUMINAMATH_CALUDE_triangle_law_of_sines_l3187_318739


namespace NUMINAMATH_CALUDE_discount_ratio_l3187_318761

/-- Calculates the total discount for a given number of gallons -/
def calculateDiscount (gallons : ℕ) : ℚ :=
  let firstTier := min gallons 10
  let secondTier := min (gallons - 10) 10
  let thirdTier := max (gallons - 20) 0
  (firstTier : ℚ) * (5 / 100) + (secondTier : ℚ) * (10 / 100) + (thirdTier : ℚ) * (15 / 100)

/-- The discount ratio theorem -/
theorem discount_ratio :
  let kimDiscount := calculateDiscount 20
  let isabellaDiscount := calculateDiscount 25
  let elijahDiscount := calculateDiscount 30
  (isabellaDiscount : ℚ) / kimDiscount = 3 / 2 ∧
  (elijahDiscount : ℚ) / kimDiscount = 4 / 2 :=
by sorry

end NUMINAMATH_CALUDE_discount_ratio_l3187_318761


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3187_318702

/-- An arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + 2 * (n - 1)

/-- The general formula for the n-th term of the arithmetic sequence -/
theorem arithmetic_sequence_formula (n : ℕ) :
  arithmetic_sequence n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3187_318702


namespace NUMINAMATH_CALUDE_range_of_a_l3187_318724

-- Define the complex number z
def z (a : ℝ) : ℂ := 1 + a * Complex.I

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (Complex.abs (z a) ≤ 2) ↔ (a ≥ -Real.sqrt 3 ∧ a ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3187_318724


namespace NUMINAMATH_CALUDE_exists_valid_strategy_l3187_318775

/-- Represents a strategy for distributing balls in boxes -/
structure Strategy where
  distribute : Fin 2018 → ℕ

/-- Represents the game setup and rules -/
structure Game where
  boxes : Fin 2018
  pairs : Fin 4032
  pairAssignment : Fin 4032 → Fin 2018 × Fin 2018

/-- Predicate to check if a strategy results in distinct ball counts -/
def isValidStrategy (g : Game) (s : Strategy) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → s.distribute i ≠ s.distribute j

/-- Theorem stating the existence of a valid strategy -/
theorem exists_valid_strategy (g : Game) : ∃ s : Strategy, isValidStrategy g s := by
  sorry


end NUMINAMATH_CALUDE_exists_valid_strategy_l3187_318775


namespace NUMINAMATH_CALUDE_train_speed_l3187_318723

/-- Given a train crossing a bridge, calculate its speed in km/hr -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : bridge_length = 225)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3187_318723


namespace NUMINAMATH_CALUDE_right_triangle_from_equations_l3187_318793

theorem right_triangle_from_equations (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  x^2 + 2*a*x + b^2 = 0 →
  x^2 + 2*c*x - b^2 = 0 →
  a^2 = b^2 + c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_equations_l3187_318793


namespace NUMINAMATH_CALUDE_book_cost_solution_l3187_318783

def book_cost_problem (x : ℕ) : Prop :=
  x > 0 ∧ 10 * x ≤ 1100 ∧ 11 * x > 1200

theorem book_cost_solution : ∃ (x : ℕ), book_cost_problem x ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_solution_l3187_318783


namespace NUMINAMATH_CALUDE_clearance_sale_gain_percentage_shopkeeper_clearance_sale_gain_l3187_318764

/-- Calculates the gain percentage during a clearance sale -/
theorem clearance_sale_gain_percentage 
  (original_price : ℝ) 
  (original_gain_percent : ℝ) 
  (discount_percent : ℝ) : ℝ :=
  let cost_price := original_price / (1 + original_gain_percent / 100)
  let discounted_price := original_price * (1 - discount_percent / 100)
  let new_gain := discounted_price - cost_price
  let new_gain_percent := (new_gain / cost_price) * 100
  new_gain_percent

/-- The gain percentage during the clearance sale is approximately 21.5% -/
theorem shopkeeper_clearance_sale_gain :
  ∃ ε > 0, abs (clearance_sale_gain_percentage 30 35 10 - 21.5) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_clearance_sale_gain_percentage_shopkeeper_clearance_sale_gain_l3187_318764


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3187_318786

/-- Given that θ is in the fourth quadrant and sin(θ + π/4) = 5/13, 
    prove that tan(θ - π/4) = -12/5 -/
theorem tan_theta_minus_pi_fourth (θ : Real) 
  (h1 : π < θ ∧ θ < 2*π) -- θ is in the fourth quadrant
  (h2 : Real.sin (θ + π/4) = 5/13) : 
  Real.tan (θ - π/4) = -12/5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3187_318786


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_jumbo_l3187_318748

def jumbo_capacity : ℕ := 1200
def regular_capacity : ℕ := 75
def mini_capacity : ℕ := 50

theorem min_bottles_to_fill_jumbo :
  (jumbo_capacity / regular_capacity = 16 ∧ jumbo_capacity % regular_capacity = 0) ∧
  (jumbo_capacity / mini_capacity = 24 ∧ jumbo_capacity % mini_capacity = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_jumbo_l3187_318748


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_div_11_l3187_318708

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_number_with_digit_sum_div_11 (N : ℕ) : 
  ∃ k ∈ Finset.range 39, 11 ∣ sum_of_digits (N + k) := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_div_11_l3187_318708


namespace NUMINAMATH_CALUDE_farmer_earnings_l3187_318799

/-- Calculates the total earnings from selling potatoes and carrots -/
def total_earnings (potato_count : ℕ) (potato_bundle_size : ℕ) (potato_bundle_price : ℚ)
                   (carrot_count : ℕ) (carrot_bundle_size : ℕ) (carrot_bundle_price : ℚ) : ℚ :=
  let potato_bundles := potato_count / potato_bundle_size
  let carrot_bundles := carrot_count / carrot_bundle_size
  potato_bundles * potato_bundle_price + carrot_bundles * carrot_bundle_price

/-- The farmer's earnings from selling all harvested crops -/
theorem farmer_earnings : 
  total_earnings 250 25 1.9 320 20 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_farmer_earnings_l3187_318799


namespace NUMINAMATH_CALUDE_divisibility_problem_l3187_318707

theorem divisibility_problem (a b c : ℕ+) 
  (h1 : a ∣ b^4) 
  (h2 : b ∣ c^4) 
  (h3 : c ∣ a^4) : 
  (a * b * c) ∣ (a + b + c)^21 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3187_318707


namespace NUMINAMATH_CALUDE_max_second_term_is_9_l3187_318782

/-- An arithmetic sequence of three positive integers with sum 27 -/
structure ArithSeq27 where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_27 : a + (a + d) + (a + 2*d) = 27

/-- The second term of an arithmetic sequence -/
def second_term (seq : ArithSeq27) : ℕ := seq.a + seq.d

/-- Theorem: The maximum value of the second term in any ArithSeq27 is 9 -/
theorem max_second_term_is_9 : 
  ∀ seq : ArithSeq27, second_term seq ≤ 9 ∧ ∃ seq : ArithSeq27, second_term seq = 9 := by
  sorry

#check max_second_term_is_9

end NUMINAMATH_CALUDE_max_second_term_is_9_l3187_318782


namespace NUMINAMATH_CALUDE_spade_equation_solution_l3187_318756

-- Define the spade operation
def spade (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

-- Theorem statement
theorem spade_equation_solution :
  ∃! A : ℝ, spade A 5 = 59 ∧ A = 9.5 := by sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l3187_318756


namespace NUMINAMATH_CALUDE_minimum_laptops_l3187_318712

theorem minimum_laptops (n p : ℕ) (h1 : n > 3) (h2 : p > 0) : 
  (p / n + (n - 3) * (p / n + 15) - p = 105) → n ≥ 10 :=
by
  sorry

#check minimum_laptops

end NUMINAMATH_CALUDE_minimum_laptops_l3187_318712


namespace NUMINAMATH_CALUDE_f_negative_a_l3187_318732

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 10

theorem f_negative_a (a M : ℝ) (h : f a = M) : f (-a) = 2 * a^2 - M := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l3187_318732


namespace NUMINAMATH_CALUDE_money_value_difference_l3187_318718

theorem money_value_difference (exchange_rate : ℝ) (marco_dollars : ℝ) (juliette_euros : ℝ) :
  exchange_rate = 1.5 →
  marco_dollars = 600 →
  juliette_euros = 350 →
  let juliette_dollars := juliette_euros * exchange_rate
  (marco_dollars - juliette_dollars) / marco_dollars * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_money_value_difference_l3187_318718


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3187_318743

theorem abs_inequality_solution_set (x : ℝ) :
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3187_318743


namespace NUMINAMATH_CALUDE_initial_girls_count_l3187_318737

theorem initial_girls_count (p : ℕ) : 
  p > 0 →  -- Ensure p is positive
  (p : ℚ) / 2 - 3 = ((p : ℚ) * 2) / 5 → 
  (p : ℚ) / 2 = 15 :=
by
  sorry

#check initial_girls_count

end NUMINAMATH_CALUDE_initial_girls_count_l3187_318737


namespace NUMINAMATH_CALUDE_gcd_105_90_l3187_318720

theorem gcd_105_90 : Nat.gcd 105 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_90_l3187_318720


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3187_318755

theorem complex_absolute_value (α β : ℂ) : 
  (∃ (a b : ℝ), α = Complex.mk a b ∧ β = Complex.mk a (-b)) → -- α and β are conjugate
  (∃ (r : ℝ), α / β^2 = r) → -- α/β² is real
  Complex.abs (α - β) = 2 * Real.sqrt 5 →
  Complex.abs α = 2 * Real.sqrt 15 / 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3187_318755


namespace NUMINAMATH_CALUDE_minimum_value_of_f_l3187_318780

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + a else x^2 - a*x

theorem minimum_value_of_f (a : ℝ) :
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_l3187_318780


namespace NUMINAMATH_CALUDE_hamburger_cost_calculation_l3187_318750

/-- Represents the cost calculation for hamburgers with higher quality meat -/
theorem hamburger_cost_calculation 
  (original_meat_pounds : ℝ) 
  (original_cost_per_pound : ℝ) 
  (original_hamburger_count : ℝ) 
  (new_hamburger_count : ℝ) 
  (cost_increase_percentage : ℝ) :
  original_meat_pounds = 5 →
  original_cost_per_pound = 4 →
  original_hamburger_count = 10 →
  new_hamburger_count = 30 →
  cost_increase_percentage = 0.25 →
  (original_meat_pounds / original_hamburger_count) * new_hamburger_count * 
  (original_cost_per_pound * (1 + cost_increase_percentage)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_cost_calculation_l3187_318750


namespace NUMINAMATH_CALUDE_cosine_sum_special_case_l3187_318759

theorem cosine_sum_special_case (α β : Real) 
  (h1 : α - β = π/3)
  (h2 : Real.tan α - Real.tan β = 3 * Real.sqrt 3) :
  Real.cos (α + β) = -1/6 := by sorry

end NUMINAMATH_CALUDE_cosine_sum_special_case_l3187_318759


namespace NUMINAMATH_CALUDE_equal_difference_implies_square_equal_difference_equal_difference_and_equal_square_difference_implies_constant_l3187_318794

/-- A sequence is an "equal difference" sequence if the difference between consecutive terms is constant. -/
def IsEqualDifference (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is an "equal square difference" sequence if the difference between consecutive squared terms is constant. -/
def IsEqualSquareDifference (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ n : ℕ, (a (n + 1))^2 - (a n)^2 = p

/-- If a sequence is an "equal difference" sequence, then its square is also an "equal difference" sequence. -/
theorem equal_difference_implies_square_equal_difference (a : ℕ → ℝ) :
    IsEqualDifference a → IsEqualDifference (fun n ↦ (a n)^2) := by sorry

/-- If a sequence is both an "equal difference" sequence and an "equal square difference" sequence,
    then it is a constant sequence. -/
theorem equal_difference_and_equal_square_difference_implies_constant (a : ℕ → ℝ) :
    IsEqualDifference a → IsEqualSquareDifference a → ∃ c : ℝ, ∀ n : ℕ, a n = c := by sorry

end NUMINAMATH_CALUDE_equal_difference_implies_square_equal_difference_equal_difference_and_equal_square_difference_implies_constant_l3187_318794


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_2_subset_condition_l3187_318751

-- Define sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_2 :
  (M ∩ N 2 = {3}) ∧ (M ∪ N 2 = M) := by sorry

-- Theorem for part (2)
theorem subset_condition :
  ∀ a : ℝ, (M ⊇ N a) ↔ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_2_subset_condition_l3187_318751


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l3187_318795

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := sorry

/-- Check if a point lies on the parabola -/
def containsPoint (p : Parabola) (x y : ℚ) : Prop := sorry

/-- Check if the parabola has a vertical axis of symmetry -/
def hasVerticalAxisOfSymmetry (p : Parabola) : Prop := sorry

theorem parabola_coefficient_sum 
  (p : Parabola) 
  (h1 : vertex p = (5, 3))
  (h2 : hasVerticalAxisOfSymmetry p)
  (h3 : containsPoint p 2 0) :
  p.a + p.b + p.c = -7/3 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l3187_318795


namespace NUMINAMATH_CALUDE_max_gcd_lcm_value_l3187_318736

theorem max_gcd_lcm_value (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) : 
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a' b' c' : ℕ), Nat.gcd (Nat.lcm a' b') c' = 10 ∧ 
    Nat.gcd (Nat.lcm a' b') c' * Nat.lcm (Nat.gcd a' b') c' = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_value_l3187_318736


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l3187_318754

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l3187_318754


namespace NUMINAMATH_CALUDE_prob_sum_8_twice_eq_l3187_318766

/-- The number of sides on each die -/
def num_sides : ℕ := 7

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The probability of rolling a sum of 8 with two dice -/
def prob_sum_8 : ℚ := 7 / 49

/-- The probability of rolling a sum of 8 twice in a row with two dice -/
def prob_sum_8_twice : ℚ := (prob_sum_8) * (prob_sum_8)

/-- Theorem: The probability of rolling a sum of 8 twice in a row
    with two 7-sided dice (numbered 1 to 7) is equal to 49/2401 -/
theorem prob_sum_8_twice_eq : prob_sum_8_twice = 49 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_8_twice_eq_l3187_318766


namespace NUMINAMATH_CALUDE_six_seat_colorings_eq_66_l3187_318771

/-- Represents the number of ways to paint n seats in a circular arrangement
    with the first seat fixed as red, using three colors (red, blue, green)
    such that adjacent seats have different colors. -/
def S : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 2
| 3 => 2
| (n + 2) => S (n + 1) + 2 * S n

/-- The number of ways to paint six seats in a circular arrangement
    using three colors (red, blue, green) such that adjacent seats
    have different colors. -/
def six_seat_colorings : ℕ := 3 * S 6

theorem six_seat_colorings_eq_66 : six_seat_colorings = 66 := by
  sorry

end NUMINAMATH_CALUDE_six_seat_colorings_eq_66_l3187_318771


namespace NUMINAMATH_CALUDE_candidate_x_wins_by_16_percent_l3187_318722

/-- Represents the election scenario with given conditions -/
structure ElectionScenario where
  repubRatio : ℚ
  demRatio : ℚ
  repubVoteX : ℚ
  demVoteX : ℚ
  (ratio_positive : repubRatio > 0 ∧ demRatio > 0)
  (vote_percentages : repubVoteX ≥ 0 ∧ repubVoteX ≤ 1 ∧ demVoteX ≥ 0 ∧ demVoteX ≤ 1)

/-- Calculates the percentage by which candidate X is expected to win -/
def winPercentage (e : ElectionScenario) : ℚ :=
  let totalVoters := e.repubRatio + e.demRatio
  let votesForX := e.repubRatio * e.repubVoteX + e.demRatio * e.demVoteX
  let votesForY := totalVoters - votesForX
  (votesForX - votesForY) / totalVoters * 100

/-- Theorem stating that under the given conditions, candidate X wins by 16% -/
theorem candidate_x_wins_by_16_percent :
  ∀ e : ElectionScenario,
    e.repubRatio = 3 ∧
    e.demRatio = 2 ∧
    e.repubVoteX = 4/5 ∧
    e.demVoteX = 1/4 →
    winPercentage e = 16 := by
  sorry


end NUMINAMATH_CALUDE_candidate_x_wins_by_16_percent_l3187_318722


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l3187_318772

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ),
    (∀ x ≤ 0, f_inv (f x) = x) ∧
    (∀ y ≥ 2, f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l3187_318772


namespace NUMINAMATH_CALUDE_racing_cars_lcm_l3187_318778

theorem racing_cars_lcm (lap_time_A lap_time_B : ℕ) 
  (h1 : lap_time_A = 28) 
  (h2 : lap_time_B = 24) : 
  Nat.lcm lap_time_A lap_time_B = 168 := by
  sorry

end NUMINAMATH_CALUDE_racing_cars_lcm_l3187_318778


namespace NUMINAMATH_CALUDE_candidate_marks_l3187_318773

theorem candidate_marks (max_marks : ℝ) (pass_percentage : ℝ) (fail_margin : ℕ) 
  (h1 : max_marks = 152.38)
  (h2 : pass_percentage = 0.42)
  (h3 : fail_margin = 22) : 
  ∃ (secured_marks : ℕ), secured_marks = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_marks_l3187_318773


namespace NUMINAMATH_CALUDE_partnership_profit_l3187_318769

/-- Calculates the total profit of a partnership business given the investments and one partner's share of the profit -/
theorem partnership_profit 
  (investment_A investment_B investment_C : ℕ) 
  (profit_share_A : ℕ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 4260) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 14200 := by
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_partnership_profit_l3187_318769


namespace NUMINAMATH_CALUDE_hyperbola_sum_theorem_l3187_318709

def F₁ : ℝ × ℝ := (2, -1)
def F₂ : ℝ × ℝ := (2, 3)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  abs (dist P F₁ - dist P F₂) = 2

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

theorem hyperbola_sum_theorem (h k a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y, hyperbola_equation x y h k a b ↔ is_on_hyperbola (x, y)) :
  h + k + a + b = 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_theorem_l3187_318709


namespace NUMINAMATH_CALUDE_share_of_a_l3187_318788

theorem share_of_a (total : ℚ) (a b c : ℚ) : 
  total = 200 →
  total = a + b + c →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 60 := by
  sorry

end NUMINAMATH_CALUDE_share_of_a_l3187_318788


namespace NUMINAMATH_CALUDE_least_reciprocal_iterations_l3187_318700

def reciprocal (x : ℚ) : ℚ := 1 / x

def reciprocalIteration (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => reciprocal (reciprocalIteration n x)

theorem least_reciprocal_iterations (n : ℕ) :
  (n > 0 ∧ reciprocalIteration n 50 = 50) → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_least_reciprocal_iterations_l3187_318700


namespace NUMINAMATH_CALUDE_pi_is_monomial_l3187_318765

-- Define what a polynomial is
def is_polynomial (f : ℝ → ℝ) : Prop := sorry

-- Define what a monomial is
def is_monomial (f : ℝ → ℝ) : Prop := sorry

-- Define what a binomial is
def is_binomial (f : ℝ → ℝ) : Prop := sorry

-- Define the degree of a polynomial
def polynomial_degree (f : ℝ → ℝ) : ℕ := sorry

-- Theorem: π is a monomial
theorem pi_is_monomial : is_monomial (λ _ => Real.pi) := by sorry

end NUMINAMATH_CALUDE_pi_is_monomial_l3187_318765


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3187_318768

theorem pipe_fill_time (fill_rate_B fill_rate_both : ℝ) 
  (hB : fill_rate_B = 1 / 15)
  (hBoth : fill_rate_both = 1 / 6)
  (hSum : fill_rate_B + (1 / fill_time_A) = fill_rate_both) :
  fill_time_A = 10 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3187_318768


namespace NUMINAMATH_CALUDE_sum_a_d_g_equals_six_l3187_318721

-- Define the variables
variable (a b c d e f g : ℤ)

-- State the theorem
theorem sum_a_d_g_equals_six 
  (eq1 : a + b + e = 7)
  (eq2 : b + c + f = 10)
  (eq3 : c + d + g = 6)
  (eq4 : e + f + g = 9) :
  a + d + g = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_d_g_equals_six_l3187_318721


namespace NUMINAMATH_CALUDE_line_intercept_sum_l3187_318714

/-- Given a line 3x + 5y + d = 0, if the sum of its x-intercept and y-intercept is 15,
    then d = -225/8 -/
theorem line_intercept_sum (d : ℚ) : 
  (∃ x y : ℚ, 3 * x + 5 * y + d = 0 ∧ x + y = 15) → d = -225/8 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l3187_318714


namespace NUMINAMATH_CALUDE_solution_set_equality_l3187_318701

open Set

def S : Set ℝ := {x | |x + 1| + |x - 4| ≥ 7}

theorem solution_set_equality : S = Iic (-2) ∪ Ici 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3187_318701


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3187_318757

/-- An isosceles triangle with perimeter 16 and base 4 has legs of length 6 -/
theorem isosceles_triangle_leg_length :
  ∀ (leg_length : ℝ),
  leg_length > 0 →
  leg_length + leg_length + 4 = 16 →
  leg_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3187_318757


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3187_318719

theorem discount_percentage_proof (coat_price pants_price : ℝ)
  (coat_discount pants_discount : ℝ) :
  coat_price = 100 →
  pants_price = 50 →
  coat_discount = 0.3 →
  pants_discount = 0.4 →
  let total_original := coat_price + pants_price
  let total_savings := coat_price * coat_discount + pants_price * pants_discount
  let savings_percentage := total_savings / total_original * 100
  savings_percentage = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3187_318719


namespace NUMINAMATH_CALUDE_orchids_in_vase_orchids_count_is_two_l3187_318774

theorem orchids_in_vase (initial_roses : ℕ) (initial_orchids : ℕ) 
  (current_roses : ℕ) (rose_orchid_difference : ℕ) : ℕ :=
  let current_orchids := current_roses - rose_orchid_difference
  current_orchids

#check orchids_in_vase 5 3 12 10

theorem orchids_count_is_two :
  orchids_in_vase 5 3 12 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_orchids_in_vase_orchids_count_is_two_l3187_318774


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3187_318716

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 1 / 3 →    -- Given ratio of sides
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- Segments r and s form the hypotenuse
  r / s = 1 / 9 :=   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3187_318716


namespace NUMINAMATH_CALUDE_wang_parts_processed_l3187_318798

/-- Represents the number of parts processed by a worker in a given time -/
def parts_processed (rate : ℕ) (time : ℕ) : ℕ := rate * time

/-- Represents Xiao Wang's work cycle -/
def wang_cycle (total_time : ℕ) : ℕ :=
  parts_processed 15 (2 * (total_time / 3))

/-- Represents Xiao Li's work -/
def li_work (total_time : ℕ) : ℕ :=
  parts_processed 12 total_time

theorem wang_parts_processed :
  ∃ (t : ℕ), t > 0 ∧ wang_cycle t = li_work t ∧ wang_cycle t = 60 :=
sorry

end NUMINAMATH_CALUDE_wang_parts_processed_l3187_318798


namespace NUMINAMATH_CALUDE_fraction_power_product_l3187_318746

theorem fraction_power_product : (8 / 9 : ℚ)^3 * (1 / 3 : ℚ)^3 = 512 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3187_318746


namespace NUMINAMATH_CALUDE_bond_interest_rate_l3187_318762

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := 0.04

/-- The initial investment amount in yuan -/
def initial_investment : ℝ := 1000

/-- The amount spent after the first maturity in yuan -/
def spent_amount : ℝ := 440

/-- The final amount received after the second maturity in yuan -/
def final_amount : ℝ := 624

/-- Theorem stating that the annual interest rate is 4% given the problem conditions -/
theorem bond_interest_rate :
  (initial_investment * (1 + annual_interest_rate) - spent_amount) * (1 + annual_interest_rate) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_bond_interest_rate_l3187_318762


namespace NUMINAMATH_CALUDE_number_square_equation_l3187_318717

theorem number_square_equation : ∃ x : ℝ, x^2 + 145 = (x - 19)^2 ∧ x = 108/19 := by
  sorry

end NUMINAMATH_CALUDE_number_square_equation_l3187_318717


namespace NUMINAMATH_CALUDE_six_lines_intersections_l3187_318710

/-- The maximum number of intersection points between n straight lines -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The maximum number of intersection points between 6 straight lines is 15 -/
theorem six_lines_intersections :
  max_intersections 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_six_lines_intersections_l3187_318710


namespace NUMINAMATH_CALUDE_part_one_part_two_l3187_318703

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := -x^2 + 4*a*x - 3*a^2

-- Define the set q
def q : Set ℝ := {x | -x^2 + 11*x - 18 ≥ 0}

-- Part 1
theorem part_one : 
  {x : ℝ | f x 1 > 0} ∩ q = Set.Icc 2 3 := by sorry

-- Part 2
theorem part_two : 
  {a : ℝ | a > 0 ∧ ∀ x, f x a > 0 → x ∈ Set.Ioo 2 9} = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3187_318703


namespace NUMINAMATH_CALUDE_cauchy_schwarz_iag_equivalence_l3187_318735

theorem cauchy_schwarz_iag_equivalence :
  (∀ (a b c d : ℝ), (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2)) ↔
  (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) ≤ (x + y) / 2) :=
by sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_iag_equivalence_l3187_318735


namespace NUMINAMATH_CALUDE_tv_sale_increase_l3187_318790

theorem tv_sale_increase (original_price original_quantity : ℝ) 
  (h_price_reduction : ℝ) (h_net_effect : ℝ) :
  h_price_reduction = 0.2 →
  h_net_effect = 0.44000000000000014 →
  ∃ (new_quantity : ℝ),
    (1 - h_price_reduction) * original_price * new_quantity = 
      (1 + h_net_effect) * original_price * original_quantity ∧
    (new_quantity / original_quantity - 1) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_tv_sale_increase_l3187_318790


namespace NUMINAMATH_CALUDE_stream_speed_l3187_318785

theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 39 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3187_318785


namespace NUMINAMATH_CALUDE_school_classrooms_l3187_318758

/-- Given a school with a total number of students and a fixed number of students per classroom,
    calculate the number of classrooms. -/
def number_of_classrooms (total_students : ℕ) (students_per_classroom : ℕ) : ℕ :=
  total_students / students_per_classroom

/-- Theorem stating that in a school with 120 students and 5 students per classroom,
    there are 24 classrooms. -/
theorem school_classrooms :
  number_of_classrooms 120 5 = 24 := by
  sorry

#eval number_of_classrooms 120 5

end NUMINAMATH_CALUDE_school_classrooms_l3187_318758


namespace NUMINAMATH_CALUDE_min_value_of_f_l3187_318715

/-- The function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -44) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3187_318715


namespace NUMINAMATH_CALUDE_orange_bags_weight_l3187_318747

/-- If 12 bags of oranges weigh 24 pounds, then 8 bags of oranges weigh 16 pounds. -/
theorem orange_bags_weight (weight_12_bags : ℝ) (h : weight_12_bags = 24) : 
  (8 / 12) * weight_12_bags = 16 := by
  sorry

end NUMINAMATH_CALUDE_orange_bags_weight_l3187_318747


namespace NUMINAMATH_CALUDE_log_always_defined_range_log_sometimes_undefined_range_l3187_318791

-- Define the function f(m, x)
def f (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + m + 3

-- Theorem 1: Range of m for which the logarithm is always defined
theorem log_always_defined_range (m : ℝ) :
  (∀ x : ℝ, f m x > 0) ↔ m ∈ Set.Ici 0 ∩ Set.Iio 1 :=
sorry

-- Theorem 2: Range of m for which the logarithm is undefined for some x
theorem log_sometimes_undefined_range (m : ℝ) :
  (∃ x : ℝ, f m x ≤ 0) ↔ m ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_log_always_defined_range_log_sometimes_undefined_range_l3187_318791


namespace NUMINAMATH_CALUDE_inequality_holds_l3187_318792

theorem inequality_holds (f : ℝ → ℝ) (a b x : ℝ) 
  (h_f : ∀ x, f x = 4 * x - 1)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_x : |x - 2*b| < b)
  (h_ab : a ≤ 4*b) : 
  (x + a)^2 + |f x - 3*b| < a^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_l3187_318792
