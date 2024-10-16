import Mathlib

namespace NUMINAMATH_CALUDE_trevor_taxi_cost_l4157_415736

/-- Calculates the total cost of Trevor's taxi ride downtown -/
def total_taxi_cost (uber_cost lyft_cost taxi_cost detour_rate tip_rate : ℚ) : ℚ :=
  let detour_cost := taxi_cost * detour_rate
  let tip := taxi_cost * tip_rate
  taxi_cost + detour_cost + tip

/-- Proves that the total cost of Trevor's taxi ride downtown is $20.25 -/
theorem trevor_taxi_cost :
  let uber_cost : ℚ := 22
  let lyft_cost : ℚ := uber_cost - 3
  let taxi_cost : ℚ := lyft_cost - 4
  let detour_rate : ℚ := 15 / 100
  let tip_rate : ℚ := 20 / 100
  total_taxi_cost uber_cost lyft_cost taxi_cost detour_rate tip_rate = 8100 / 400 := by
  sorry

#eval total_taxi_cost 22 19 15 (15/100) (20/100)

end NUMINAMATH_CALUDE_trevor_taxi_cost_l4157_415736


namespace NUMINAMATH_CALUDE_binomial_12_choose_5_l4157_415793

theorem binomial_12_choose_5 : Nat.choose 12 5 = 792 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_5_l4157_415793


namespace NUMINAMATH_CALUDE_max_checkers_8x8_l4157_415712

/-- Represents a chess board -/
structure Board :=
  (size : Nat)

/-- Represents a configuration of checkers on a board -/
structure CheckerConfiguration :=
  (board : Board)
  (numCheckers : Nat)

/-- Predicate to check if a configuration is valid (all checkers under attack) -/
def isValidConfiguration (config : CheckerConfiguration) : Prop := sorry

/-- The maximum number of checkers that can be placed on a board -/
def maxCheckers (b : Board) : Nat := sorry

/-- Theorem stating the maximum number of checkers on an 8x8 board -/
theorem max_checkers_8x8 :
  ∃ (config : CheckerConfiguration),
    config.board.size = 8 ∧
    isValidConfiguration config ∧
    config.numCheckers = maxCheckers config.board ∧
    config.numCheckers = 32 :=
  sorry

end NUMINAMATH_CALUDE_max_checkers_8x8_l4157_415712


namespace NUMINAMATH_CALUDE_f_extrema_l4157_415795

def f (x : ℝ) := x^2 - 2*x

theorem f_extrema :
  ∀ x ∈ Set.Icc (-1 : ℝ) 5,
    -1 ≤ f x ∧ f x ≤ 15 ∧
    (∃ x₁ ∈ Set.Icc (-1 : ℝ) 5, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-1 : ℝ) 5, f x₂ = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l4157_415795


namespace NUMINAMATH_CALUDE_union_of_sets_l4157_415741

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 1, a}
def B (a : ℝ) : Set ℝ := {0, 3, 3*a}

-- Theorem statement
theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {0, 3}) : 
  A a ∪ B a = {0, 1, 3, 9} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l4157_415741


namespace NUMINAMATH_CALUDE_unanswered_completion_count_l4157_415737

/-- A structure representing a multiple choice test -/
structure MultipleChoiceTest where
  total_questions : Nat
  choices_per_question : Nat
  single_answer_questions : Nat
  multi_select_questions : Nat
  correct_choices_per_multi : Nat

/-- The number of ways to complete the test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : Nat :=
  1

/-- Theorem stating that there is only one way to complete the test with all questions unanswered -/
theorem unanswered_completion_count (test : MultipleChoiceTest)
  (h1 : test.total_questions = 10)
  (h2 : test.choices_per_question = 8)
  (h3 : test.single_answer_questions = 6)
  (h4 : test.multi_select_questions = 4)
  (h5 : test.correct_choices_per_multi = 2)
  (h6 : test.total_questions = test.single_answer_questions + test.multi_select_questions) :
  ways_to_complete_unanswered test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_completion_count_l4157_415737


namespace NUMINAMATH_CALUDE_negation_existential_statement_l4157_415780

theorem negation_existential_statement :
  ¬(∃ (x : ℝ), x^2 - x + 2 > 0) ≠ (∀ (x : ℝ), x^2 - x + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existential_statement_l4157_415780


namespace NUMINAMATH_CALUDE_zero_in_interval_one_two_l4157_415777

noncomputable def f (x : ℝ) := Real.exp x + 2 * x - 6

theorem zero_in_interval_one_two :
  ∃ z ∈ Set.Ioo 1 2, f z = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_one_two_l4157_415777


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l4157_415716

/-- Number of partitions of n into at most k parts -/
def num_partitions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to put 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : num_partitions 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l4157_415716


namespace NUMINAMATH_CALUDE_building_height_calculation_l4157_415778

/-- Given a building and a pole, calculate the height of the building using similar triangles. -/
theorem building_height_calculation (building_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ)
  (h_building_shadow : building_shadow = 20)
  (h_pole_height : pole_height = 2)
  (h_pole_shadow : pole_shadow = 3) :
  (pole_height / pole_shadow) * building_shadow = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_building_height_calculation_l4157_415778


namespace NUMINAMATH_CALUDE_tangent_perpendicular_line_l4157_415714

-- Define the curve
def C : ℝ → ℝ := fun x ↦ x^2

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at P
def tangent_slope : ℝ := 2

-- Define the perpendicular line
def perpendicular_line (a : ℝ) : ℝ → ℝ := fun x ↦ -a * x - 1

-- State the theorem
theorem tangent_perpendicular_line :
  ∀ a : ℝ, (C P.1 = P.2) →
  (tangent_slope * (-1/a) = -1) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_line_l4157_415714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4157_415704

theorem arithmetic_sequence_sum : 
  let a₁ : ℤ := 1
  let aₙ : ℤ := 1996
  let n : ℕ := 96
  let s := n * (a₁ + aₙ) / 2
  s = 95856 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4157_415704


namespace NUMINAMATH_CALUDE_point_D_coordinates_l4157_415783

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, 5)

theorem point_D_coordinates :
  let AD : ℝ × ℝ := (3 * (B.1 - A.1), 3 * (B.2 - A.2))
  let D : ℝ × ℝ := (A.1 + AD.1, A.2 + AD.2)
  D = (-7, 9) := by sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l4157_415783


namespace NUMINAMATH_CALUDE_factorization_equivalence_l4157_415784

theorem factorization_equivalence (x y : ℝ) : -(2*x - y) * (2*x + y) = -4*x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equivalence_l4157_415784


namespace NUMINAMATH_CALUDE_first_transfer_amount_l4157_415760

/-- Proves that the amount of the first bank transfer is approximately $91.18 given the initial and final balances and service charge. -/
theorem first_transfer_amount (initial_balance : ℝ) (final_balance : ℝ) (service_charge_rate : ℝ) :
  initial_balance = 400 →
  final_balance = 307 →
  service_charge_rate = 0.02 →
  ∃ (transfer_amount : ℝ), 
    initial_balance - (transfer_amount * (1 + service_charge_rate)) = final_balance ∧
    (transfer_amount ≥ 91.17 ∧ transfer_amount ≤ 91.19) :=
by sorry

end NUMINAMATH_CALUDE_first_transfer_amount_l4157_415760


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4157_415774

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -15) →
  (a 2 * a 18 = 16) →
  a 3 * a 10 * a 17 = -64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4157_415774


namespace NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l4157_415700

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_equal_absolute_log 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hf : f a = f b) : 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l4157_415700


namespace NUMINAMATH_CALUDE_inequality_solution_l4157_415771

theorem inequality_solution (a : ℝ) :
  (a = 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x ≠ 1/2) ∧
  (a < 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x > a ∨ x < 1 - a) ∧
  (a > 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x > a ∨ x < 1 - a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4157_415771


namespace NUMINAMATH_CALUDE_floor_equation_solution_l4157_415794

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l4157_415794


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l4157_415701

/-- Two circles are internally tangent if the distance between their centers
    plus the radius of the smaller circle equals the radius of the larger circle. -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d + min r₁ r₂ = max r₁ r₂

/-- The theorem states that two circles with radii 3 and 7, whose centers are 4 units apart,
    are internally tangent. -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 7
  let d : ℝ := 4
  internally_tangent r₁ r₂ d :=
by
  sorry


end NUMINAMATH_CALUDE_circles_internally_tangent_l4157_415701


namespace NUMINAMATH_CALUDE_m_range_l4157_415799

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem m_range (m : ℝ) (h1 : ¬(p m)) (h2 : p m ∨ q m) : 1 < m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l4157_415799


namespace NUMINAMATH_CALUDE_smallest_positive_e_l4157_415767

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure IntPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : IntPolynomial) (x : ℚ) : Prop :=
  p.a * x^4 + p.b * x^3 + p.c * x^2 + p.d * x + p.e = 0

/-- The main theorem stating the smallest possible value of e -/
theorem smallest_positive_e (p : IntPolynomial) : 
  p.e > 0 → 
  isRoot p (-2) → 
  isRoot p 5 → 
  isRoot p 9 → 
  isRoot p (-1/3) → 
  p.e ≥ 90 ∧ ∃ q : IntPolynomial, q.e = 90 ∧ 
    isRoot q (-2) ∧ isRoot q 5 ∧ isRoot q 9 ∧ isRoot q (-1/3) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_e_l4157_415767


namespace NUMINAMATH_CALUDE_cube_sum_problem_l4157_415779

theorem cube_sum_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (sum_prod_eq : x * y + y * z + z * x = -6)
  (prod_eq : x * y * z = -6) :
  x^3 + y^3 + z^3 = 25 := by sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l4157_415779


namespace NUMINAMATH_CALUDE_finite_solutions_of_equation_l4157_415755

theorem finite_solutions_of_equation : 
  Finite {xyz : ℕ × ℕ × ℕ | (1 : ℚ) / xyz.1 + (1 : ℚ) / xyz.2.1 + (1 : ℚ) / xyz.2.2 = (1 : ℚ) / 1983} :=
by sorry

end NUMINAMATH_CALUDE_finite_solutions_of_equation_l4157_415755


namespace NUMINAMATH_CALUDE_orthocenter_ratio_zero_l4157_415751

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define the orthocenter
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the altitude
def altitude (t : Triangle) (side : ℝ) : ℝ := sorry

-- Define the ratio
def ratio (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem orthocenter_ratio_zero (t : Triangle) 
  (h1 : t.a = 11) (h2 : t.b = 13) (h3 : t.c = 16) : 
  ratio t = 0 := by sorry

end NUMINAMATH_CALUDE_orthocenter_ratio_zero_l4157_415751


namespace NUMINAMATH_CALUDE_train_length_calculation_l4157_415702

theorem train_length_calculation (passing_time man_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  passing_time = 8 →
  man_time = 8 →
  platform_length = 273 →
  platform_time = 20 →
  ∃ (train_length : ℝ), train_length = 182 ∧
    train_length / man_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l4157_415702


namespace NUMINAMATH_CALUDE_bill_apples_left_l4157_415724

/-- The number of apples Bill has left after distributing to teachers and baking pies -/
def apples_left (initial_apples : ℕ) (num_children : ℕ) (apples_per_teacher : ℕ) 
  (num_teachers_per_child : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_children * apples_per_teacher * num_teachers_per_child) - (num_pies * apples_per_pie)

/-- Theorem stating that Bill has 18 apples left -/
theorem bill_apples_left : 
  apples_left 50 2 3 2 2 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bill_apples_left_l4157_415724


namespace NUMINAMATH_CALUDE_percentage_sum_l4157_415729

theorem percentage_sum : 
  (20 / 100 * 30) + (15 / 100 * 50) + (25 / 100 * 120) + (-10 / 100 * 45) = 39 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l4157_415729


namespace NUMINAMATH_CALUDE_prob_two_gold_given_at_least_one_gold_l4157_415735

/-- The probability of selecting two gold medals given that at least one gold medal is selected -/
theorem prob_two_gold_given_at_least_one_gold 
  (total_medals : ℕ) 
  (gold_medals : ℕ) 
  (silver_medals : ℕ) 
  (bronze_medals : ℕ) 
  (h1 : total_medals = gold_medals + silver_medals + bronze_medals)
  (h2 : total_medals = 10)
  (h3 : gold_medals = 5)
  (h4 : silver_medals = 3)
  (h5 : bronze_medals = 2) :
  (Nat.choose gold_medals 2 : ℚ) / (Nat.choose total_medals 2 - Nat.choose (silver_medals + bronze_medals) 2) = 2/7 :=
sorry

end NUMINAMATH_CALUDE_prob_two_gold_given_at_least_one_gold_l4157_415735


namespace NUMINAMATH_CALUDE_kitten_weight_l4157_415759

theorem kitten_weight (k d1 d2 : ℝ) 
  (total_weight : k + d1 + d2 = 30)
  (larger_dog_relation : k + d2 = 3 * d1)
  (smaller_dog_relation : k + d1 = d2 + 10) :
  k = 25 / 2 := by
sorry

end NUMINAMATH_CALUDE_kitten_weight_l4157_415759


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l4157_415708

def numbers : List ℝ := [15, 23, 37, 45]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 30 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l4157_415708


namespace NUMINAMATH_CALUDE_impossible_table_l4157_415740

/-- Represents a 7x7 table of natural numbers -/
def Table := Fin 7 → Fin 7 → ℕ

/-- Checks if the sum of numbers in a 2x2 square starting at (i, j) is odd -/
def is_2x2_sum_odd (t : Table) (i j : Fin 7) : Prop :=
  Odd (t i j + t i (j+1) + t (i+1) j + t (i+1) (j+1))

/-- Checks if the sum of numbers in a 3x3 square starting at (i, j) is odd -/
def is_3x3_sum_odd (t : Table) (i j : Fin 7) : Prop :=
  Odd (t i j + t i (j+1) + t i (j+2) +
       t (i+1) j + t (i+1) (j+1) + t (i+1) (j+2) +
       t (i+2) j + t (i+2) (j+1) + t (i+2) (j+2))

/-- The main theorem stating that it's impossible to construct a table satisfying the conditions -/
theorem impossible_table : ¬ ∃ (t : Table), 
  (∀ (i j : Fin 7), i < 6 ∧ j < 6 → is_2x2_sum_odd t i j) ∧ 
  (∀ (i j : Fin 7), i < 5 ∧ j < 5 → is_3x3_sum_odd t i j) :=
sorry

end NUMINAMATH_CALUDE_impossible_table_l4157_415740


namespace NUMINAMATH_CALUDE_coin_coverage_probability_l4157_415742

/-- The probability of a coin covering part of the black region on a square -/
theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) : 
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  (78 + 5 * Real.pi + 12 * Real.sqrt 2) / 64 = 
    (4 * (triangle_leg^2 / 2 + Real.pi + 2 * triangle_leg) + 
     2 * diamond_side^2 + Real.pi + 4 * diamond_side) / 
    ((square_side - coin_diameter)^2) := by
  sorry

end NUMINAMATH_CALUDE_coin_coverage_probability_l4157_415742


namespace NUMINAMATH_CALUDE_divisor_function_ratio_l4157_415738

/-- τ(n) denotes the number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := sorry

theorem divisor_function_ratio (n : ℕ+) (h : τ (n^2) / τ n = 3) : 
  τ (n^7) / τ n = 29 := by sorry

end NUMINAMATH_CALUDE_divisor_function_ratio_l4157_415738


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l4157_415785

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l4157_415785


namespace NUMINAMATH_CALUDE_equal_sums_iff_odd_l4157_415773

def is_valid_seating (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∀ (boy : ℕ) (girl1 : ℕ) (girl2 : ℕ),
    boy ≤ n ∧ n < girl1 ∧ girl1 ≤ 2*n ∧ n < girl2 ∧ girl2 ≤ 2*n →
    boy + girl1 + girl2 = 4*n + (3*n + 3)/2

theorem equal_sums_iff_odd (n : ℕ) :
  is_valid_seating n ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_equal_sums_iff_odd_l4157_415773


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l4157_415749

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_two : 2 * lg 5 + lg 4 = 2 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l4157_415749


namespace NUMINAMATH_CALUDE_seashell_count_l4157_415753

theorem seashell_count (sally_shells tom_shells jessica_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : tom_shells = 7)
  (h3 : jessica_shells = 5) :
  sally_shells + tom_shells + jessica_shells = 21 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l4157_415753


namespace NUMINAMATH_CALUDE_expression_always_zero_l4157_415727

theorem expression_always_zero (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ((x / |y| - |x| / y) * (y / |z| - |y| / z) * (z / |x| - |z| / x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_zero_l4157_415727


namespace NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_l4157_415706

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 2|

-- Part 1
theorem solution_set_for_negative_one (x : ℝ) :
  (f (-1) x ≥ 6) ↔ (x ≤ -1 ∨ x ≥ 3) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 3*a^2 - |2 - x|) → (-1 ≤ a ∧ a ≤ 4/3) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_l4157_415706


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_of_2047_4_plus_1_l4157_415732

theorem least_odd_prime_factor_of_2047_4_plus_1 (p : Nat) : 
  p = 41 ↔ 
    Prime p ∧ 
    Odd p ∧ 
    p ∣ (2047^4 + 1) ∧ 
    ∀ q : Nat, Prime q → Odd q → q ∣ (2047^4 + 1) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_of_2047_4_plus_1_l4157_415732


namespace NUMINAMATH_CALUDE_entree_cost_l4157_415775

theorem entree_cost (total : ℝ) (difference : ℝ) (entree : ℝ) (dessert : ℝ)
  (h1 : total = 23)
  (h2 : difference = 5)
  (h3 : entree = dessert + difference)
  (h4 : total = entree + dessert) :
  entree = 14 := by
sorry

end NUMINAMATH_CALUDE_entree_cost_l4157_415775


namespace NUMINAMATH_CALUDE_inequality_proof_l4157_415731

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 + x)^n + (1 - x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4157_415731


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4157_415754

theorem sufficient_not_necessary : 
  (∃ a : ℝ, a = 1 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4157_415754


namespace NUMINAMATH_CALUDE_gcd_2023_2052_l4157_415713

theorem gcd_2023_2052 : Nat.gcd 2023 2052 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2023_2052_l4157_415713


namespace NUMINAMATH_CALUDE_relationship_between_3a_3b_4a_l4157_415747

theorem relationship_between_3a_3b_4a (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_3a_3b_4a_l4157_415747


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l4157_415757

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 10 ∧ (1056 + x) % 26 = 0 ∧ ∀ (y : ℕ), y < x → (1056 + y) % 26 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l4157_415757


namespace NUMINAMATH_CALUDE_product_closure_l4157_415717

def A : Set ℤ := {n | ∃ t s : ℤ, n = t^2 + s^2}

theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l4157_415717


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4157_415711

theorem trigonometric_equation_solution (x : ℝ) :
  (∃ (k : ℤ), x = 2 * π * k / 3) ∨ 
  (∃ (n : ℤ), x = π * (4 * n + 1) / 6) ↔ 
  (Real.cos (3 * x / 2) ≠ 0 ∧ 
   Real.sin ((3 * x - 7 * π) / 2) * Real.cos ((π - 3 * x) / 2) = 
   Real.arccos (3 * x / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4157_415711


namespace NUMINAMATH_CALUDE_wallpapering_solution_l4157_415781

/-- Represents the number of days needed to complete the wallpapering job -/
structure WallpaperingJob where
  worker1 : ℝ  -- Days needed for worker 1 to complete the job alone
  worker2 : ℝ  -- Days needed for worker 2 to complete the job alone

/-- The wallpapering job satisfies the given conditions -/
def satisfies_conditions (job : WallpaperingJob) : Prop :=
  -- Worker 1 needs 3 days more than Worker 2
  job.worker1 = job.worker2 + 3 ∧
  -- The combined work of both workers in 7 days equals the whole job
  (7 / job.worker1) + (5.5 / job.worker2) = 1

/-- The theorem stating the solution to the wallpapering problem -/
theorem wallpapering_solution :
  ∃ (job : WallpaperingJob), satisfies_conditions job ∧ job.worker1 = 14 ∧ job.worker2 = 11 := by
  sorry


end NUMINAMATH_CALUDE_wallpapering_solution_l4157_415781


namespace NUMINAMATH_CALUDE_second_class_size_l4157_415748

theorem second_class_size (students1 : ℕ) (avg1 : ℚ) (avg2 : ℚ) (avg_total : ℚ) :
  students1 = 25 →
  avg1 = 40 →
  avg2 = 60 →
  avg_total = 50.90909090909091 →
  ∃ students2 : ℕ, 
    students2 = 30 ∧
    (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℚ) = avg_total :=
by sorry

end NUMINAMATH_CALUDE_second_class_size_l4157_415748


namespace NUMINAMATH_CALUDE_probability_theorem_l4157_415798

def harmonic_number (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℚ))

def probability_all_own_hats (n : ℕ) : ℚ :=
  (Finset.prod (Finset.range n) (λ i => harmonic_number (i + 1))) / (n.factorial : ℚ)

theorem probability_theorem (n : ℕ) :
  probability_all_own_hats n =
    (Finset.prod (Finset.range n) (λ i => harmonic_number (i + 1))) / (n.factorial : ℚ) :=
by sorry

#eval probability_all_own_hats 10

end NUMINAMATH_CALUDE_probability_theorem_l4157_415798


namespace NUMINAMATH_CALUDE_factorial_plus_twelve_square_l4157_415743

theorem factorial_plus_twelve_square (m n : ℕ) : m.factorial + 12 = n^2 ↔ m = 4 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_twelve_square_l4157_415743


namespace NUMINAMATH_CALUDE_calculate_rates_l4157_415707

/-- Represents the rates and quantities in the problem -/
structure Rates where
  d : ℕ  -- number of dishes washed
  b : ℕ  -- number of bananas cooked
  p : ℕ  -- number of pears picked
  tp : ℕ -- time spent picking pears (in hours)
  tb : ℕ -- time spent cooking bananas (in hours)
  tw : ℕ -- time spent washing dishes (in hours)
  rp : ℚ -- rate of picking pears (pears per hour)
  rb : ℚ -- rate of cooking bananas (bananas per hour)
  rw : ℚ -- rate of washing dishes (dishes per hour)

/-- The main theorem stating the conditions and the result to be proved -/
theorem calculate_rates (r : Rates) 
    (h1 : r.d = r.b + 10)
    (h2 : r.b = 3 * r.p)
    (h3 : r.p = 50)
    (h4 : r.tp = 4)
    (h5 : r.tb = 2)
    (h6 : r.tw = 5)
    (h7 : r.rp = r.p / r.tp)
    (h8 : r.rb = r.b / r.tb)
    (h9 : r.rw = r.d / r.tw) :
    r.rp = 25/2 ∧ r.rb = 75 ∧ r.rw = 32 := by
  sorry


end NUMINAMATH_CALUDE_calculate_rates_l4157_415707


namespace NUMINAMATH_CALUDE_campground_distance_l4157_415797

/-- Calculates the total distance traveled given multiple segments of driving at different speeds. -/
def total_distance (segments : List (ℝ × ℝ)) : ℝ :=
  segments.map (fun (speed, time) => speed * time) |>.sum

/-- The driving segments for Sue's family vacation. -/
def vacation_segments : List (ℝ × ℝ) :=
  [(50, 3), (60, 2), (55, 1), (65, 2)]

/-- Theorem stating that the total distance to the campground is 455 miles. -/
theorem campground_distance :
  total_distance vacation_segments = 455 := by
  sorry

#eval total_distance vacation_segments

end NUMINAMATH_CALUDE_campground_distance_l4157_415797


namespace NUMINAMATH_CALUDE_eccentricity_properties_l4157_415733

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the line y = x + 4
def line (x : ℝ) : ℝ := x + 4

-- Define the eccentricity function
noncomputable def eccentricity (x₀ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (x₀, line x₀)
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let a := (PA + PB) / 2
  let c := 2  -- half the distance between foci
  c / a

-- Theorem statement
theorem eccentricity_properties :
  (∀ ε > 0, ∃ x₀ : ℝ, eccentricity x₀ < ε) ∧
  (∃ M : ℝ, ∀ x₀ : ℝ, eccentricity x₀ ≤ M) :=
sorry

end NUMINAMATH_CALUDE_eccentricity_properties_l4157_415733


namespace NUMINAMATH_CALUDE_no_square_with_two_or_three_ones_l4157_415789

/-- Represents a number in base-10 using only 0 and 1 digits -/
def IsBaseOneZero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- Counts the number of ones in the base-10 representation of a number -/
def CountOnes (n : ℕ) : ℕ :=
  (n.digits 10).filter (· = 1) |>.length

/-- Main theorem: No square number exists with only 0 and 1 digits and exactly 2 or 3 ones -/
theorem no_square_with_two_or_three_ones :
  ¬ ∃ n : ℕ, IsBaseOneZero (n^2) ∧ (CountOnes (n^2) = 2 ∨ CountOnes (n^2) = 3) :=
sorry

end NUMINAMATH_CALUDE_no_square_with_two_or_three_ones_l4157_415789


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l4157_415720

/-- Proves that 850 billion yuan is equal to 8.5 × 10^11 yuan -/
theorem billion_to_scientific_notation :
  let billion : ℝ := 10^9
  850 * billion = 8.5 * 10^11 := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l4157_415720


namespace NUMINAMATH_CALUDE_overtime_calculation_l4157_415770

/-- Calculates the number of overtime hours worked given the total gross pay, regular hourly rate, overtime hourly rate, and regular hours limit. -/
def overtime_hours (gross_pay : ℚ) (regular_rate : ℚ) (overtime_rate : ℚ) (regular_hours_limit : ℕ) : ℕ :=
  sorry

/-- The number of overtime hours worked is 10 given the specified conditions. -/
theorem overtime_calculation :
  let gross_pay : ℚ := 622
  let regular_rate : ℚ := 11.25
  let overtime_rate : ℚ := 16
  let regular_hours_limit : ℕ := 40
  overtime_hours gross_pay regular_rate overtime_rate regular_hours_limit = 10 := by
  sorry

end NUMINAMATH_CALUDE_overtime_calculation_l4157_415770


namespace NUMINAMATH_CALUDE_octagon_area_in_square_l4157_415744

/-- The area of a regular octagon inscribed in a square -/
theorem octagon_area_in_square (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) :
  let octagon_area := s^2 - 8
  octagon_area = 16 + 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_octagon_area_in_square_l4157_415744


namespace NUMINAMATH_CALUDE_square_expression_l4157_415718

theorem square_expression (x y : ℝ) (square : ℝ) :
  4 * x^2 * square = 81 * x^3 * y → square = (81/4) * x * y := by
  sorry

end NUMINAMATH_CALUDE_square_expression_l4157_415718


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l4157_415752

theorem cube_sum_theorem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l4157_415752


namespace NUMINAMATH_CALUDE_colonization_combinations_l4157_415703

/-- Represents the number of Earth-like planets -/
def earth_like_planets : Nat := 7

/-- Represents the number of Mars-like planets -/
def mars_like_planets : Nat := 8

/-- Represents the colonization units required for an Earth-like planet -/
def earth_like_units : Nat := 3

/-- Represents the colonization units required for a Mars-like planet -/
def mars_like_units : Nat := 1

/-- Represents the total available colonization units -/
def total_units : Nat := 21

/-- Calculates the number of different combinations of planets that can be occupied -/
def count_combinations : Nat := sorry

theorem colonization_combinations : count_combinations = 981 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_l4157_415703


namespace NUMINAMATH_CALUDE_cos_fourteen_pi_thirds_l4157_415725

theorem cos_fourteen_pi_thirds : Real.cos (14 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_fourteen_pi_thirds_l4157_415725


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4157_415722

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4157_415722


namespace NUMINAMATH_CALUDE_cone_base_circumference_l4157_415746

/-- The circumference of the base of a cone formed from a 180° sector of a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = π → 2 * π * r * (θ / (2 * π)) = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l4157_415746


namespace NUMINAMATH_CALUDE_cut_pyramid_volume_ratio_l4157_415728

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  lateralAreaRatio : ℚ  -- Ratio of lateral surface areas (small pyramid : frustum)
  volumeRatio : ℚ       -- Ratio of volumes (small pyramid : frustum)

/-- Theorem: If the lateral area ratio is 9:16, then the volume ratio is 27:98 -/
theorem cut_pyramid_volume_ratio (p : CutPyramid) 
  (h : p.lateralAreaRatio = 9 / 16) : p.volumeRatio = 27 / 98 := by
  sorry

end NUMINAMATH_CALUDE_cut_pyramid_volume_ratio_l4157_415728


namespace NUMINAMATH_CALUDE_intersection_and_complement_eq_union_l4157_415726

/-- Given the universal set ℝ, prove that the intersection of M and the complement of N in ℝ
    is the union of {x | x < -2} and {x | x ≥ 3} -/
theorem intersection_and_complement_eq_union (M N : Set ℝ) : 
  M = {x : ℝ | x^2 > 4} →
  N = {x : ℝ | (x - 3) / (x + 1) < 0} →
  M ∩ (Set.univ \ N) = {x : ℝ | x < -2} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_eq_union_l4157_415726


namespace NUMINAMATH_CALUDE_monkeys_required_for_new_bananas_l4157_415739

/-- Represents the number of monkeys eating bananas -/
def num_monkeys : ℕ := 5

/-- Represents the number of bananas eaten in the initial scenario -/
def initial_bananas : ℕ := 5

/-- Represents the time taken to eat the initial number of bananas -/
def initial_time : ℕ := 5

/-- Represents the number of bananas to be eaten in the new scenario -/
def new_bananas : ℕ := 15

/-- Theorem stating that the number of monkeys required to eat the new number of bananas
    is equal to the initial number of monkeys -/
theorem monkeys_required_for_new_bananas :
  (num_monkeys : ℕ) = (num_monkeys : ℕ) := by sorry

end NUMINAMATH_CALUDE_monkeys_required_for_new_bananas_l4157_415739


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l4157_415705

/-- The number of students who know the secret on day n -/
def students_knowing_secret (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day of the week given a number of days since Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem secret_spread_theorem : 
  ∃ n : ℕ, students_knowing_secret n = 2186 ∧ day_of_week n = "Sunday" :=
by sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l4157_415705


namespace NUMINAMATH_CALUDE_fold_cut_unfold_result_l4157_415709

/-- Represents a square sheet of paper with two sides --/
structure Sheet :=
  (side_length : ℝ)
  (white_side : Bool)
  (gray_side : Bool)

/-- Represents a fold on the sheet --/
inductive Fold
  | Vertical
  | Horizontal

/-- Represents a cut on the folded sheet --/
structure Cut :=
  (size : ℝ)

/-- The result of unfolding the sheet after folding and cutting --/
structure UnfoldedResult :=
  (num_cutouts : ℕ)
  (symmetric : Bool)

/-- Function to fold the sheet --/
def fold_sheet (s : Sheet) (f : Fold) : Sheet :=
  sorry

/-- Function to cut the folded sheet --/
def cut_sheet (s : Sheet) (c : Cut) : Sheet :=
  sorry

/-- Function to unfold the sheet --/
def unfold_sheet (s : Sheet) : UnfoldedResult :=
  sorry

/-- Theorem stating the result of folding twice, cutting, and unfolding --/
theorem fold_cut_unfold_result (s : Sheet) (f1 f2 : Fold) (c : Cut) :
  let folded := fold_sheet (fold_sheet s f1) f2
  let cut := cut_sheet folded c
  let result := unfold_sheet cut
  result.num_cutouts = 4 ∧ result.symmetric = true :=
sorry

end NUMINAMATH_CALUDE_fold_cut_unfold_result_l4157_415709


namespace NUMINAMATH_CALUDE_equation_solution_l4157_415762

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.4 * x = 2.1 * x - 30) ∧ (x = -35 / 1.3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4157_415762


namespace NUMINAMATH_CALUDE_value_is_appropriate_for_project_assessment_other_terms_not_appropriate_l4157_415761

-- Define the possible options
inductive ProjectAssessmentTerm
  | Price
  | Value
  | Cost
  | Expense

-- Define a function that determines if a term is appropriate for project assessment
def isAppropriateForProjectAssessment (term : ProjectAssessmentTerm) : Prop :=
  match term with
  | ProjectAssessmentTerm.Value => True
  | _ => False

-- Theorem stating that "Value" is the appropriate term
theorem value_is_appropriate_for_project_assessment :
  isAppropriateForProjectAssessment ProjectAssessmentTerm.Value :=
by sorry

-- Theorem stating that other terms are not appropriate
theorem other_terms_not_appropriate (term : ProjectAssessmentTerm) :
  term ≠ ProjectAssessmentTerm.Value →
  ¬(isAppropriateForProjectAssessment term) :=
by sorry

end NUMINAMATH_CALUDE_value_is_appropriate_for_project_assessment_other_terms_not_appropriate_l4157_415761


namespace NUMINAMATH_CALUDE_triangle_side_length_l4157_415758

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2 / 3 →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4157_415758


namespace NUMINAMATH_CALUDE_alexs_score_l4157_415756

theorem alexs_score (total_students : ℕ) (initial_students : ℕ) (initial_avg : ℕ) (final_avg : ℕ) :
  total_students = 20 →
  initial_students = 19 →
  initial_avg = 76 →
  final_avg = 78 →
  (initial_students * initial_avg + (total_students - initial_students) * x) / total_students = final_avg →
  x = 116 :=
by sorry

end NUMINAMATH_CALUDE_alexs_score_l4157_415756


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l4157_415769

/-- A point in the second quadrant with given absolute values for its coordinates -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  abs_x : |x| = 2
  abs_y : |y| = 3

/-- The symmetric point with respect to the origin -/
def symmetric_point (p : SecondQuadrantPoint) : ℝ × ℝ := (-p.x, -p.y)

/-- Theorem stating that the symmetric point has coordinates (2, -3) -/
theorem symmetric_point_coordinates (p : SecondQuadrantPoint) : 
  symmetric_point p = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l4157_415769


namespace NUMINAMATH_CALUDE_production_problem_l4157_415710

def initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) : ℕ :=
  ((n + 1) * new_average - today_production) / n

theorem production_problem :
  let n : ℕ := 3
  let today_production : ℕ := 90
  let new_average : ℕ := 75
  initial_average_production n today_production new_average = 70 := by
  sorry

end NUMINAMATH_CALUDE_production_problem_l4157_415710


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l4157_415723

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 14*x^2 + 49*x - 24 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 14*s^2 + 49*s - 24) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 123 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l4157_415723


namespace NUMINAMATH_CALUDE_evaluate_64_to_5_6th_power_l4157_415745

theorem evaluate_64_to_5_6th_power : (64 : ℝ) ^ (5/6) = 32 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_64_to_5_6th_power_l4157_415745


namespace NUMINAMATH_CALUDE_potion_combinations_eq_thirteen_l4157_415787

/-- The number of ways to combine roots and minerals for a potion. -/
def potionCombinations : ℕ :=
  let totalRoots : ℕ := 3
  let totalMinerals : ℕ := 5
  let incompatibleCombinations : ℕ := 2
  totalRoots * totalMinerals - incompatibleCombinations

/-- Theorem stating that the number of potion combinations is 13. -/
theorem potion_combinations_eq_thirteen : potionCombinations = 13 := by
  sorry

end NUMINAMATH_CALUDE_potion_combinations_eq_thirteen_l4157_415787


namespace NUMINAMATH_CALUDE_evaluate_expression_l4157_415765

theorem evaluate_expression : (18 ^ 36) / (54 ^ 18) = 6 ^ 18 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4157_415765


namespace NUMINAMATH_CALUDE_theater_seats_l4157_415792

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given properties has 770 seats -/
theorem theater_seats :
  ∀ t : Theater,
    t.first_row_seats = 14 →
    t.seat_increase = 2 →
    t.last_row_seats = 56 →
    total_seats t = 770 := by
  sorry

#eval total_seats { first_row_seats := 14, seat_increase := 2, last_row_seats := 56 }

end NUMINAMATH_CALUDE_theater_seats_l4157_415792


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l4157_415796

theorem correct_quotient_proof (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 42 →  -- Dividing by 12 (incorrect divisor) yields 42
  D / 21 = 24  -- The correct quotient when dividing by 21 is 24
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l4157_415796


namespace NUMINAMATH_CALUDE_power_sum_of_i_l4157_415790

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^11 + i^111 = -2 * i :=
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l4157_415790


namespace NUMINAMATH_CALUDE_intersection_of_sets_l4157_415719

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l4157_415719


namespace NUMINAMATH_CALUDE_actual_average_height_l4157_415766

/-- Proves that the actual average height of students is 174.62 cm given the initial conditions --/
theorem actual_average_height (n : ℕ) (initial_avg : ℝ) 
  (h1_recorded h1_actual h2_recorded h2_actual h3_recorded h3_actual : ℝ) :
  n = 50 ∧ 
  initial_avg = 175 ∧
  h1_recorded = 151 ∧ h1_actual = 136 ∧
  h2_recorded = 162 ∧ h2_actual = 174 ∧
  h3_recorded = 185 ∧ h3_actual = 169 →
  (n : ℝ) * initial_avg - (h1_recorded - h1_actual + h2_recorded - h2_actual + h3_recorded - h3_actual) = n * 174.62 :=
by sorry

end NUMINAMATH_CALUDE_actual_average_height_l4157_415766


namespace NUMINAMATH_CALUDE_rays_dog_walks_63_blocks_l4157_415768

/-- Represents the distance of a single walk in blocks -/
structure Walk where
  to_destination : ℕ
  to_second_place : ℕ
  back_home : ℕ

/-- Calculates the total distance of a walk -/
def Walk.total (w : Walk) : ℕ := w.to_destination + w.to_second_place + w.back_home

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  morning : Walk
  afternoon : Walk
  evening : Walk

/-- Calculates the total distance of all walks in a day -/
def DailyWalk.total_distance (d : DailyWalk) : ℕ :=
  d.morning.total + d.afternoon.total + d.evening.total

/-- Ray's actual daily walk routine -/
def rays_routine : DailyWalk := {
  morning := { to_destination := 4, to_second_place := 7, back_home := 11 }
  afternoon := { to_destination := 3, to_second_place := 5, back_home := 8 }
  evening := { to_destination := 6, to_second_place := 9, back_home := 10 }
}

/-- Theorem stating that Ray's dog walks 63 blocks each day -/
theorem rays_dog_walks_63_blocks : DailyWalk.total_distance rays_routine = 63 := by
  sorry

end NUMINAMATH_CALUDE_rays_dog_walks_63_blocks_l4157_415768


namespace NUMINAMATH_CALUDE_andy_late_demerits_l4157_415786

/-- The maximum number of demerits Andy can get before being fired -/
def max_demerits : ℕ := 50

/-- The number of times Andy showed up late -/
def late_instances : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def joke_demerits : ℕ := 15

/-- The number of additional demerits Andy can get this month before being fired -/
def remaining_demerits : ℕ := 23

/-- The number of demerits Andy gets per instance of being late -/
def demerits_per_late_instance : ℕ := 2

theorem andy_late_demerits :
  late_instances * demerits_per_late_instance + joke_demerits = max_demerits - remaining_demerits :=
sorry

end NUMINAMATH_CALUDE_andy_late_demerits_l4157_415786


namespace NUMINAMATH_CALUDE_unique_prime_pair_sum_73_l4157_415788

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_pair_sum_73 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 73 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_sum_73_l4157_415788


namespace NUMINAMATH_CALUDE_charcoal_for_900ml_l4157_415721

/-- Given a ratio of charcoal to water and a volume of water, calculate the amount of charcoal needed. -/
def charcoal_needed (charcoal_ratio : ℚ) (water_volume : ℚ) : ℚ :=
  water_volume / (30 / charcoal_ratio)

/-- Theorem: The amount of charcoal needed for 900 ml of water is 60 grams, given the ratio of 2 grams of charcoal per 30 ml of water. -/
theorem charcoal_for_900ml :
  charcoal_needed 2 900 = 60 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_for_900ml_l4157_415721


namespace NUMINAMATH_CALUDE_total_skateboarding_distance_l4157_415715

/-- The distance John skateboarded to the park -/
def distance_to_park : ℝ := 16

/-- Theorem: John's total skateboarding distance is 32 miles -/
theorem total_skateboarding_distance :
  2 * distance_to_park = 32 :=
sorry

end NUMINAMATH_CALUDE_total_skateboarding_distance_l4157_415715


namespace NUMINAMATH_CALUDE_james_pizza_slices_l4157_415782

theorem james_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) (james_fraction : ℚ) : 
  num_pizzas = 2 → 
  slices_per_pizza = 6 → 
  james_fraction = 2/3 →
  (↑num_pizzas * ↑slices_per_pizza : ℚ) * james_fraction = 8 := by
  sorry

end NUMINAMATH_CALUDE_james_pizza_slices_l4157_415782


namespace NUMINAMATH_CALUDE_faye_pencils_l4157_415750

/-- The number of pencils Faye has in total -/
def total_pencils (pencils_per_row : ℕ) (num_rows : ℕ) : ℕ :=
  pencils_per_row * num_rows

/-- Theorem: Faye has 32 pencils in total -/
theorem faye_pencils : total_pencils 8 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l4157_415750


namespace NUMINAMATH_CALUDE_no_reverse_equal_base6_l4157_415730

/-- Function to reverse the digits of a natural number in base 10 --/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Function to convert a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Theorem stating that no natural number greater than 5 has its reversed decimal representation equal to its base 6 representation --/
theorem no_reverse_equal_base6 :
  ∀ n : ℕ, n > 5 → reverseDigits n ≠ toBase6 n :=
sorry

end NUMINAMATH_CALUDE_no_reverse_equal_base6_l4157_415730


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l4157_415764

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Proof that the discriminant of 5x² + 3x - 8 is 169 -/
theorem discriminant_of_5x2_plus_3x_minus_8 :
  discriminant 5 3 (-8) = 169 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l4157_415764


namespace NUMINAMATH_CALUDE_parallelogram_xy_sum_l4157_415776

/-- A parallelogram with sides a, b, c, d where opposite sides are equal -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  opposite_sides_equal : a = c ∧ b = d

/-- The specific parallelogram from the problem -/
def problem_parallelogram (x y : ℝ) : Parallelogram where
  a := 6 * y - 2
  b := 12
  c := 3 * x + 4
  d := 9
  opposite_sides_equal := by sorry

theorem parallelogram_xy_sum (x y : ℝ) :
  (problem_parallelogram x y).a = (problem_parallelogram x y).c ∧
  (problem_parallelogram x y).b = (problem_parallelogram x y).d →
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_xy_sum_l4157_415776


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4157_415772

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4157_415772


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l4157_415734

theorem complex_number_coordinates : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i^3) / (2 + i)
  Complex.re z = 0 ∧ Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l4157_415734


namespace NUMINAMATH_CALUDE_brothers_age_fraction_l4157_415791

theorem brothers_age_fraction :
  let younger_age : ℕ := 27
  let total_age : ℕ := 46
  let older_age : ℕ := total_age - younger_age
  ∃ f : ℚ, younger_age = f * older_age + 10 ∧ f = 17 / 19 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_fraction_l4157_415791


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l4157_415763

/-- Given two rectangles of equal area, where one rectangle has dimensions 5 inches by 24 inches,
    and the other rectangle has a length of 3 inches, prove that the width of the second rectangle
    is 40 inches. -/
theorem equal_area_rectangles_width (area : ℝ) (width : ℝ) :
  area = 5 * 24 →  -- Carol's rectangle area
  area = 3 * width →  -- Jordan's rectangle area
  width = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l4157_415763
