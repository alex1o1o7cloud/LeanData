import Mathlib

namespace NUMINAMATH_CALUDE_exists_point_product_nonnegative_l1483_148363

theorem exists_point_product_nonnegative 
  (f : ℝ → ℝ) 
  (hf : ContDiff ℝ 3 f) : 
  ∃ a : ℝ, f a * (deriv f a) * (deriv^[2] f a) * (deriv^[3] f a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_point_product_nonnegative_l1483_148363


namespace NUMINAMATH_CALUDE_best_fit_line_slope_for_given_data_l1483_148335

/-- Data point representing height and weight measurements -/
structure DataPoint where
  height : ℝ
  weight : ℝ

/-- Calculate the slope of the best-fit line for given data points -/
def bestFitLineSlope (data : List DataPoint) : ℝ :=
  sorry

/-- Theorem stating that the slope of the best-fit line for the given data is 0.525 -/
theorem best_fit_line_slope_for_given_data :
  let data := [
    DataPoint.mk 150 50,
    DataPoint.mk 160 55,
    DataPoint.mk 170 60.5
  ]
  bestFitLineSlope data = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_best_fit_line_slope_for_given_data_l1483_148335


namespace NUMINAMATH_CALUDE_max_voters_with_95_percent_support_l1483_148325

/-- Represents the election scenario with an initial poll and subsequent groups -/
structure ElectionPoll where
  initial_voters : ℕ
  initial_support : ℕ
  group_size : ℕ
  group_support : ℕ

/-- Calculates the total number of voters and supporters for a given number of additional groups -/
def totalVoters (poll : ElectionPoll) (additional_groups : ℕ) : ℕ × ℕ :=
  (poll.initial_voters + poll.group_size * additional_groups,
   poll.initial_support + poll.group_support * additional_groups)

/-- Checks if the support percentage is at least 95% -/
def isSupportAboveThreshold (total : ℕ) (support : ℕ) : Prop :=
  (support : ℚ) / (total : ℚ) ≥ 95 / 100

/-- Theorem stating the maximum number of voters while maintaining 95% support -/
theorem max_voters_with_95_percent_support :
  ∃ (poll : ElectionPoll) (max_groups : ℕ),
    poll.initial_voters = 100 ∧
    poll.initial_support = 98 ∧
    poll.group_size = 10 ∧
    poll.group_support = 9 ∧
    (let (total, support) := totalVoters poll max_groups
     isSupportAboveThreshold total support) ∧
    (∀ g > max_groups,
      let (total, support) := totalVoters poll g
      ¬(isSupportAboveThreshold total support)) ∧
    poll.initial_voters + poll.group_size * max_groups = 160 :=
  sorry

end NUMINAMATH_CALUDE_max_voters_with_95_percent_support_l1483_148325


namespace NUMINAMATH_CALUDE_min_weighings_three_l1483_148372

/-- Represents the outcome of a weighing --/
inductive WeighingOutcome
  | Equal : WeighingOutcome
  | LeftHeavier : WeighingOutcome
  | RightHeavier : WeighingOutcome

/-- Represents a coin --/
inductive Coin
  | Real : Coin
  | Fake : Coin

/-- Represents a weighing strategy --/
def WeighingStrategy := List (List Coin × List Coin)

/-- The total number of coins --/
def totalCoins : Nat := 2023

/-- The number of fake coins --/
def fakeCoins : Nat := 2

/-- The number of real coins --/
def realCoins : Nat := totalCoins - fakeCoins

/-- A function that determines the outcome of a weighing --/
def weighOutcome (left right : List Coin) : WeighingOutcome := sorry

/-- A function that determines if a strategy is valid --/
def isValidStrategy (strategy : WeighingStrategy) : Prop := sorry

/-- A function that determines if a strategy solves the problem --/
def solvesProblem (strategy : WeighingStrategy) : Prop := sorry

/-- The main theorem stating that the minimum number of weighings is 3 --/
theorem min_weighings_three :
  ∃ (strategy : WeighingStrategy),
    strategy.length = 3 ∧
    isValidStrategy strategy ∧
    solvesProblem strategy ∧
    ∀ (other : WeighingStrategy),
      isValidStrategy other →
      solvesProblem other →
      other.length ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_weighings_three_l1483_148372


namespace NUMINAMATH_CALUDE_binomial_coefficient_minus_two_divisible_by_prime_l1483_148397

theorem binomial_coefficient_minus_two_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (2 * p).factorial / (p.factorial * p.factorial) - 2 = k * p :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_minus_two_divisible_by_prime_l1483_148397


namespace NUMINAMATH_CALUDE_triangle_congruence_criteria_triangle_congruence_criteria_2_l1483_148337

-- Define the structure for a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define side lengths
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define angle measure
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_congruence_criteria (ABC A'B'C' : Triangle) :
  (side_length ABC.A ABC.B = side_length A'B'C'.A A'B'C'.B ∧
   side_length ABC.B ABC.C = side_length A'B'C'.B A'B'C'.C ∧
   side_length ABC.A ABC.C = side_length A'B'C'.A A'B'C'.C) →
  congruent ABC A'B'C' :=
sorry

theorem triangle_congruence_criteria_2 (ABC A'B'C' : Triangle) :
  (side_length ABC.A ABC.B = side_length A'B'C'.A A'B'C'.B ∧
   angle_measure ABC.A ABC.B ABC.C = angle_measure A'B'C'.A A'B'C'.B A'B'C'.C ∧
   angle_measure ABC.B ABC.C ABC.A = angle_measure A'B'C'.B A'B'C'.C A'B'C'.A) →
  congruent ABC A'B'C' :=
sorry

end NUMINAMATH_CALUDE_triangle_congruence_criteria_triangle_congruence_criteria_2_l1483_148337


namespace NUMINAMATH_CALUDE_simplify_expression_l1483_148303

theorem simplify_expression (m n : ℤ) (h : m * n = m + 3) :
  2 * m * n + 3 * m - 5 * m * n - 10 = -19 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1483_148303


namespace NUMINAMATH_CALUDE_male_25_plus_percentage_proof_l1483_148308

/-- The percentage of male students in a graduating class -/
def male_percentage : ℝ := 0.4

/-- The percentage of female students who are 25 years old or older -/
def female_25_plus_percentage : ℝ := 0.4

/-- The probability of randomly selecting a student less than 25 years old -/
def under_25_probability : ℝ := 0.56

/-- The percentage of male students who are 25 years old or older -/
def male_25_plus_percentage : ℝ := 0.5

theorem male_25_plus_percentage_proof :
  male_25_plus_percentage = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_male_25_plus_percentage_proof_l1483_148308


namespace NUMINAMATH_CALUDE_second_container_capacity_l1483_148346

/-- Represents a container with dimensions and sand capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  sandCapacity : ℝ

/-- Theorem stating the sand capacity of the second container -/
theorem second_container_capacity 
  (c1 : Container) 
  (c2 : Container) 
  (h1 : c1.height = 3)
  (h2 : c1.width = 4)
  (h3 : c1.length = 6)
  (h4 : c1.sandCapacity = 72)
  (h5 : c2.height = 3 * c1.height)
  (h6 : c2.width = 2 * c1.width)
  (h7 : c2.length = c1.length) :
  c2.sandCapacity = 432 := by
  sorry


end NUMINAMATH_CALUDE_second_container_capacity_l1483_148346


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1483_148304

theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |a * b| = Real.sqrt 867.75 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1483_148304


namespace NUMINAMATH_CALUDE_initial_bottles_count_l1483_148380

/-- The number of bottles Jason buys -/
def jason_bottles : ℕ := 5

/-- The number of bottles Harry buys -/
def harry_bottles : ℕ := 6

/-- The number of bottles left on the shelf after purchases -/
def remaining_bottles : ℕ := 24

/-- The initial number of bottles on the shelf -/
def initial_bottles : ℕ := jason_bottles + harry_bottles + remaining_bottles

theorem initial_bottles_count : initial_bottles = 35 := by
  sorry

end NUMINAMATH_CALUDE_initial_bottles_count_l1483_148380


namespace NUMINAMATH_CALUDE_solution_set_is_real_solution_set_is_empty_solution_set_has_element_l1483_148398

-- Define the quadratic expression
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Define the solution set for the inequality
def solution_set (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Theorem 1: The solution set is ℝ iff a ∈ (-∞, 0]
theorem solution_set_is_real : ∀ a : ℝ, solution_set a = Set.univ ↔ a ≤ 0 := by sorry

-- Theorem 2: The solution set is ∅ iff a ∈ [3, +∞)
theorem solution_set_is_empty : ∀ a : ℝ, solution_set a = ∅ ↔ a ≥ 3 := by sorry

-- Theorem 3: There is at least one real solution iff a ∈ (-∞, 3)
theorem solution_set_has_element : ∀ a : ℝ, (∃ x : ℝ, x ∈ solution_set a) ↔ a < 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_real_solution_set_is_empty_solution_set_has_element_l1483_148398


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l1483_148344

theorem no_solution_to_inequality_system : 
  ¬ ∃ x : ℝ, (x - 3 ≥ 0) ∧ (2*x - 5 < 1) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l1483_148344


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l1483_148378

theorem triangle_two_solutions (a b : ℝ) (A : ℝ) :
  a = 6 →
  b = 6 * Real.sqrt 3 →
  A = π / 6 →
  ∃! (c₁ c₂ B₁ B₂ C₁ C₂ : ℝ),
    (c₁ = 12 ∧ B₁ = π / 3 ∧ C₁ = π / 2) ∧
    (c₂ = 6 ∧ B₂ = 2 * π / 3 ∧ C₂ = π / 6) ∧
    (∀ c B C : ℝ,
      (c = c₁ ∧ B = B₁ ∧ C = C₁) ∨
      (c = c₂ ∧ B = B₂ ∧ C = C₂)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l1483_148378


namespace NUMINAMATH_CALUDE_exists_efficient_coin_ordering_strategy_l1483_148392

/-- A strategy for ordering coins by weight using a balance scale. -/
structure CoinOrderingStrategy where
  /-- The number of coins to be ordered -/
  num_coins : Nat
  /-- The expected number of weighings required by the strategy -/
  expected_weighings : ℝ

/-- A weighing action compares two coins and determines which is heavier -/
def weighing_action (coin1 coin2 : Nat) : Bool := sorry

/-- Theorem stating that there exists a strategy for ordering 4 coins with expected weighings < 4.8 -/
theorem exists_efficient_coin_ordering_strategy :
  ∃ (strategy : CoinOrderingStrategy),
    strategy.num_coins = 4 ∧
    strategy.expected_weighings < 4.8 := by sorry

end NUMINAMATH_CALUDE_exists_efficient_coin_ordering_strategy_l1483_148392


namespace NUMINAMATH_CALUDE_sarahs_sweaters_sarahs_sweaters_proof_l1483_148362

theorem sarahs_sweaters (machine_capacity : ℕ) (num_shirts : ℕ) (num_loads : ℕ) : ℕ :=
  let total_pieces := machine_capacity * num_loads
  let num_sweaters := total_pieces - num_shirts
  num_sweaters

theorem sarahs_sweaters_proof 
  (h1 : sarahs_sweaters 5 43 9 = 2) : sarahs_sweaters 5 43 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_sweaters_sarahs_sweaters_proof_l1483_148362


namespace NUMINAMATH_CALUDE_l_shaped_room_flooring_cost_l1483_148310

/-- Represents the dimensions of a rectangular room section -/
structure RoomSection where
  length : ℝ
  width : ℝ

/-- Calculates the total cost of replacing flooring in an L-shaped room -/
def total_flooring_cost (section1 section2 : RoomSection) (removal_cost per_sqft_cost : ℝ) : ℝ :=
  let total_area := section1.length * section1.width + section2.length * section2.width
  removal_cost + total_area * per_sqft_cost

/-- Theorem: The total cost to replace the floor in the given L-shaped room is $150 -/
theorem l_shaped_room_flooring_cost :
  let section1 : RoomSection := ⟨8, 7⟩
  let section2 : RoomSection := ⟨6, 4⟩
  let removal_cost : ℝ := 50
  let per_sqft_cost : ℝ := 1.25
  total_flooring_cost section1 section2 removal_cost per_sqft_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_room_flooring_cost_l1483_148310


namespace NUMINAMATH_CALUDE_initial_average_calculation_l1483_148364

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) :
  n = 10 →
  correct_avg = 16 →
  error = 10 →
  (n * correct_avg - error) / n = 15 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l1483_148364


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1483_148375

/-- The function f(x) = x^2 + 2ax - a + 2 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - a + 2

/-- Statement 1: For any x ∈ ℝ, f(x) ≥ 0 if and only if a ∈ [-2,1] --/
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a ∈ Set.Icc (-2) 1 := by sorry

/-- Statement 2: For any x ∈ [-1,1], f(x) ≥ 0 if and only if a ∈ [-3,1] --/
theorem problem_2 (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 0) ↔ a ∈ Set.Icc (-3) 1 := by sorry

/-- Statement 3: For any a ∈ [-1,1], x^2 + 2ax - a + 2 > 0 if and only if x ≠ -1 --/
theorem problem_3 (x : ℝ) :
  (∀ a ∈ Set.Icc (-1) 1, f a x > 0) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1483_148375


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l1483_148331

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (ordered : x ≤ y ∧ y ≤ z)
  (max_twice_min : z ≤ 2 * x) :
  x * y * z ≥ 1 / 32 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 1 ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 2 * a ∧ a * b * c = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l1483_148331


namespace NUMINAMATH_CALUDE_kolya_optimal_strategy_l1483_148370

/-- Represents the three methods Kolya can choose from -/
inductive Method
  | largest_smallest
  | two_middle
  | choice_with_payment

/-- Represents a division of nuts -/
structure NutDivision where
  a₁ : ℕ
  a₂ : ℕ
  b₁ : ℕ
  b₂ : ℕ

/-- Calculates the number of nuts Kolya gets for a given method and division -/
def nuts_for_kolya (m : Method) (d : NutDivision) : ℕ :=
  match m with
  | Method.largest_smallest => max d.a₁ d.b₁ + min d.a₂ d.b₂
  | Method.two_middle => d.a₁ + d.a₂ + d.b₁ + d.b₂ - (max d.a₁ (max d.a₂ (max d.b₁ d.b₂))) - (min d.a₁ (min d.a₂ (min d.b₁ d.b₂)))
  | Method.choice_with_payment => max (max d.a₁ d.b₁ + min d.a₂ d.b₂) (d.a₁ + d.a₂ + d.b₁ + d.b₂ - (max d.a₁ (max d.a₂ (max d.b₁ d.b₂))) - (min d.a₁ (min d.a₂ (min d.b₁ d.b₂)))) - 1

/-- Theorem stating the existence of most and least advantageous methods for Kolya -/
theorem kolya_optimal_strategy (n : ℕ) (h : n ≥ 2) :
  ∃ (best worst : Method) (d : NutDivision),
    (d.a₁ + d.a₂ + d.b₁ + d.b₂ = 2*n + 1) ∧
    (d.a₁ ≥ 1 ∧ d.a₂ ≥ 1 ∧ d.b₁ ≥ 1 ∧ d.b₂ ≥ 1) ∧
    (∀ m : Method, nuts_for_kolya best d ≥ nuts_for_kolya m d) ∧
    (∀ m : Method, nuts_for_kolya worst d ≤ nuts_for_kolya m d) :=
  sorry

end NUMINAMATH_CALUDE_kolya_optimal_strategy_l1483_148370


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l1483_148360

/-- The locus of points (x, y) satisfying both equations 2ux - 3y - 2u = 0 and x - 3uy + 2 = 0,
    where u is a real parameter, is a circle. -/
theorem intersection_locus_is_circle :
  ∀ (x y u : ℝ), (2 * u * x - 3 * y - 2 * u = 0) ∧ (x - 3 * u * y + 2 = 0) →
  ∃ (c : ℝ × ℝ) (r : ℝ), (x - c.1)^2 + (y - c.2)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l1483_148360


namespace NUMINAMATH_CALUDE_negation_of_existence_cube_lt_pow_three_negation_l1483_148333

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem cube_lt_pow_three_negation :
  (¬ ∃ x : ℕ, x^3 < 3^x) ↔ (∀ x : ℕ, x^3 ≥ 3^x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cube_lt_pow_three_negation_l1483_148333


namespace NUMINAMATH_CALUDE_complex_square_l1483_148385

theorem complex_square : (1 - Complex.I) ^ 2 = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1483_148385


namespace NUMINAMATH_CALUDE_product_equation_minimum_sum_l1483_148329

theorem product_equation_minimum_sum (x y z a : ℤ) : 
  (x - 10) * (y - a) * (z - 2) = 1000 →
  x + y + z ≥ 7 →
  (∀ x' y' z' : ℤ, (x' - 10) * (y' - a) * (z' - 2) = 1000 → x' + y' + z' ≥ x + y + z) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equation_minimum_sum_l1483_148329


namespace NUMINAMATH_CALUDE_problem_solution_l1483_148391

theorem problem_solution (a b : ℝ) (h_distinct : a ≠ b) (h_sum_squares : a^2 + b^2 = 5) :
  (ab = 2 → a + b = 3 ∨ a + b = -3) ∧
  (a^2 - 2*a = b^2 - 2*b → a + b = 2 ∧ a^2 - 2*a = (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1483_148391


namespace NUMINAMATH_CALUDE_rhombus_square_diagonals_l1483_148381

-- Define a rhombus
structure Rhombus :=
  (sides_equal : ∀ s1 s2 : ℝ, s1 = s2)
  (diagonals_perpendicular : Bool)

-- Define a square as a special case of rhombus
structure Square extends Rhombus :=
  (all_angles_right : Bool)

-- Theorem statement
theorem rhombus_square_diagonals :
  ∃ (r : Rhombus), ¬(∀ d1 d2 : ℝ, d1 = d2) ∧
  ∀ (s : Square), ∀ d1 d2 : ℝ, d1 = d2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_square_diagonals_l1483_148381


namespace NUMINAMATH_CALUDE_log_equation_solution_l1483_148367

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 4 → x = 2401 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1483_148367


namespace NUMINAMATH_CALUDE_unique_solution_sum_l1483_148356

def star_operation (m n : ℕ) : ℕ := m^n + m*n

theorem unique_solution_sum (m n : ℕ) 
  (hm : m ≥ 2) 
  (hn : n ≥ 2) 
  (h_star : star_operation m n = 64) : 
  m + n = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_sum_l1483_148356


namespace NUMINAMATH_CALUDE_sum_is_six_digit_multiple_of_four_l1483_148317

def sum_of_numbers (A B : Nat) : Nat :=
  98765 + A * 1000 + 532 + B * 100 + 41 + 1021

theorem sum_is_six_digit_multiple_of_four (A B : Nat) 
  (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : 1 ≤ B ∧ B ≤ 9) : 
  ∃ (n : Nat), sum_of_numbers A B = n ∧ 
  100000 ≤ n ∧ n < 1000000 ∧ 
  n % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_is_six_digit_multiple_of_four_l1483_148317


namespace NUMINAMATH_CALUDE_least_largest_factor_l1483_148345

theorem least_largest_factor (a b c d e : ℕ+) : 
  a * b * c * d * e = 55 * 60 * 65 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (∀ x y z w v : ℕ+, 
    x * y * z * w * v = 55 * 60 * 65 ∧ 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ 
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ 
    z ≠ w ∧ z ≠ v ∧ 
    w ≠ v →
    max a (max b (max c (max d e))) ≤ max x (max y (max z (max w v)))) →
  max a (max b (max c (max d e))) = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_largest_factor_l1483_148345


namespace NUMINAMATH_CALUDE_steves_speed_back_l1483_148340

/-- Proves that Steve's speed on the way back from work is 14 km/h given the conditions --/
theorem steves_speed_back (distance : ℝ) (total_time : ℝ) (speed_ratio : ℝ) : 
  distance = 28 → 
  total_time = 6 → 
  speed_ratio = 2 → 
  (distance / (distance / (2 * (distance / (total_time - distance / (2 * (distance / total_time)))))) = 14) := by
  sorry

end NUMINAMATH_CALUDE_steves_speed_back_l1483_148340


namespace NUMINAMATH_CALUDE_inverse_f_zero_solution_l1483_148324

noncomputable section

variables (a b c : ℝ)
variable (f : ℝ → ℝ)

-- Define the function f
def f_def : f = λ x => 1 / (a * x^2 + b * x + c) := by sorry

-- Conditions: a, b, and c are nonzero
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom c_nonzero : c ≠ 0

-- Theorem: The only solution to f^(-1)(x) = 0 is x = 1/c
theorem inverse_f_zero_solution :
  ∀ x : ℝ, (Function.invFun f) x = 0 ↔ x = 1 / c := by sorry

end

end NUMINAMATH_CALUDE_inverse_f_zero_solution_l1483_148324


namespace NUMINAMATH_CALUDE_groups_with_pair_fraction_l1483_148369

-- Define the number of people
def n : ℕ := 6

-- Define the size of each group
def k : ℕ := 3

-- Define the function to calculate combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem groups_with_pair_fraction :
  C (n - 2) (k - 2) / C n k = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_groups_with_pair_fraction_l1483_148369


namespace NUMINAMATH_CALUDE_least_positive_integer_mod_l1483_148326

theorem least_positive_integer_mod (n : ℕ) : 
  ∃ x : ℕ, x > 0 ∧ (x + 7237 : ℤ) ≡ 5017 [ZMOD 12] ∧ 
  ∀ y : ℕ, y > 0 ∧ (y + 7237 : ℤ) ≡ 5017 [ZMOD 12] → x ≤ y :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_mod_l1483_148326


namespace NUMINAMATH_CALUDE_school_cafeteria_discussion_l1483_148388

theorem school_cafeteria_discussion (students_like : ℕ) (students_dislike : ℕ) : 
  students_like = 383 → students_dislike = 431 → students_like + students_dislike = 814 :=
by sorry

end NUMINAMATH_CALUDE_school_cafeteria_discussion_l1483_148388


namespace NUMINAMATH_CALUDE_paula_candy_distribution_l1483_148366

theorem paula_candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) :
  initial_candies = 20 →
  additional_candies = 4 →
  num_friends = 6 →
  (initial_candies + additional_candies) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_paula_candy_distribution_l1483_148366


namespace NUMINAMATH_CALUDE_luna_budget_theorem_l1483_148339

/-- Luna's monthly budget problem --/
theorem luna_budget_theorem 
  (house_rental : ℝ) 
  (food : ℝ) 
  (phone : ℝ) 
  (h1 : food = 0.6 * house_rental) 
  (h2 : house_rental + food = 240) 
  (h3 : house_rental + food + phone = 249) :
  phone = 0.1 * food := by
  sorry

end NUMINAMATH_CALUDE_luna_budget_theorem_l1483_148339


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l1483_148394

theorem fractional_equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + 5) / (x - 3) = 2 - m / (3 - x)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l1483_148394


namespace NUMINAMATH_CALUDE_no_divisible_by_three_for_all_x_l1483_148311

theorem no_divisible_by_three_for_all_x : ¬∃ (p q : ℤ), ∀ (x : ℤ), 3 ∣ (x^2 + p*x + q) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_three_for_all_x_l1483_148311


namespace NUMINAMATH_CALUDE_white_washing_cost_l1483_148342

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem white_washing_cost
  (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ)
  (window_width window_height : ℝ)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_width : door_width = 6)
  (h_door_height : door_height = 3)
  (h_window_width : window_width = 4)
  (h_window_height : window_height = 3)
  (h_num_windows : num_windows = 3)
  (h_cost_per_sqft : cost_per_sqft = 6) :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_width * door_height
  let window_area := window_width * window_height
  let total_opening_area := door_area + num_windows * window_area
  let paintable_area := total_wall_area - total_opening_area
  let total_cost := paintable_area * cost_per_sqft
  total_cost = 5436 := by sorry


end NUMINAMATH_CALUDE_white_washing_cost_l1483_148342


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_at_least_four_divisors_l1483_148314

/-- Two natural numbers are consecutive primes if they are both prime and there is no prime between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬ Prime k

/-- The number of positive divisors of a natural number n. -/
def numPositiveDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range n.succ)).card

theorem sum_of_consecutive_odd_primes_has_at_least_four_divisors
  (p q : ℕ) (h : ConsecutivePrimes p q) :
  4 ≤ numPositiveDivisors (p + q) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_at_least_four_divisors_l1483_148314


namespace NUMINAMATH_CALUDE_cubic_root_approximation_bound_l1483_148368

theorem cubic_root_approximation_bound :
  ∃ (c : ℝ), c > 0 ∧ ∀ (m n : ℤ), n ≥ 1 →
    |2^(1/3 : ℝ) - (m : ℝ) / (n : ℝ)| > c / (n : ℝ)^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_approximation_bound_l1483_148368


namespace NUMINAMATH_CALUDE_athlete_arrangements_l1483_148300

/-- The number of athletes and tracks -/
def n : ℕ := 6

/-- Function to calculate the number of arrangements where A, B, and C are not adjacent -/
def arrangements_not_adjacent : ℕ := sorry

/-- Function to calculate the number of arrangements where there is one person between A and B -/
def arrangements_one_between : ℕ := sorry

/-- Function to calculate the number of arrangements where A is not on first or second track, and B is on fifth or sixth track -/
def arrangements_restricted : ℕ := sorry

/-- Theorem stating the correct number of arrangements for each scenario -/
theorem athlete_arrangements :
  arrangements_not_adjacent = 144 ∧
  arrangements_one_between = 192 ∧
  arrangements_restricted = 144 :=
sorry

end NUMINAMATH_CALUDE_athlete_arrangements_l1483_148300


namespace NUMINAMATH_CALUDE_total_students_is_150_l1483_148307

/-- Proves that the total number of students is 150 given the conditions -/
theorem total_students_is_150 
  (total : ℕ) 
  (boys : ℕ) 
  (girls : ℕ) 
  (h1 : total = boys + girls) 
  (h2 : boys = 60 → girls = (60 * total) / 100) : 
  total = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_150_l1483_148307


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1483_148334

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecularWeight (carbon_atoms : ℕ) (hydrogen_atoms : ℕ) (oxygen_atoms : ℕ) 
                    (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + (hydrogen_atoms : ℝ) * hydrogen_weight + (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 3 Carbon, 6 Hydrogen, and 1 Oxygen is approximately 58.078 g/mol -/
theorem compound_molecular_weight :
  let carbon_atoms : ℕ := 3
  let hydrogen_atoms : ℕ := 6
  let oxygen_atoms : ℕ := 1
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  ∃ ε > 0, |molecularWeight carbon_atoms hydrogen_atoms oxygen_atoms 
                            carbon_weight hydrogen_weight oxygen_weight - 58.078| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1483_148334


namespace NUMINAMATH_CALUDE_fib_like_seq_a7_l1483_148319

/-- An increasing sequence of positive integers satisfying the Fibonacci-like recurrence -/
def FibLikeSeq (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a (n + 1) + a n)

theorem fib_like_seq_a7 (a : ℕ → ℕ) (h : FibLikeSeq a) (h6 : a 6 = 50) : 
  a 7 = 83 := by
sorry

end NUMINAMATH_CALUDE_fib_like_seq_a7_l1483_148319


namespace NUMINAMATH_CALUDE_total_accidents_l1483_148393

/-- Represents the accident rate for a highway -/
structure AccidentRate where
  accidents : ℕ
  vehicles : ℕ

/-- Calculates the number of accidents for a given traffic volume -/
def calculateAccidents (rate : AccidentRate) (traffic : ℕ) : ℕ :=
  (rate.accidents * traffic + rate.vehicles - 1) / rate.vehicles

theorem total_accidents (highwayA_rate : AccidentRate) (highwayB_rate : AccidentRate) (highwayC_rate : AccidentRate)
  (highwayA_traffic : ℕ) (highwayB_traffic : ℕ) (highwayC_traffic : ℕ) :
  highwayA_rate = ⟨200, 100000000⟩ →
  highwayB_rate = ⟨150, 50000000⟩ →
  highwayC_rate = ⟨100, 150000000⟩ →
  highwayA_traffic = 2000000000 →
  highwayB_traffic = 1500000000 →
  highwayC_traffic = 2500000000 →
  calculateAccidents highwayA_rate highwayA_traffic +
  calculateAccidents highwayB_rate highwayB_traffic +
  calculateAccidents highwayC_rate highwayC_traffic = 10168 := by
  sorry

end NUMINAMATH_CALUDE_total_accidents_l1483_148393


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1483_148396

/-- The distance between the foci of a hyperbola given by the equation 9x^2 - 27x - 16y^2 - 32y = 72 -/
theorem hyperbola_foci_distance : 
  let equation := fun (x y : ℝ) => 9 * x^2 - 27 * x - 16 * y^2 - 32 * y - 72
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
        ((x - 3/2)^2 / a^2) - ((y + 1)^2 / b^2) = 1 ∧
        c^2 = a^2 + b^2) ∧
    2 * c = Real.sqrt 41775 / 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1483_148396


namespace NUMINAMATH_CALUDE_jenny_cat_expenditure_first_year_l1483_148371

/-- Calculates Jenny's expenditure on a cat for the first year -/
def jennys_cat_expenditure (adoption_fee : ℕ) (vet_costs : ℕ) (monthly_food_cost : ℕ) (jenny_toy_costs : ℕ) : ℕ :=
  let shared_costs := adoption_fee + vet_costs
  let jenny_shared_costs := shared_costs / 2
  let annual_food_cost := monthly_food_cost * 12
  let jenny_food_cost := annual_food_cost / 2
  jenny_shared_costs + jenny_food_cost + jenny_toy_costs

/-- Theorem stating Jenny's total expenditure on the cat in the first year -/
theorem jenny_cat_expenditure_first_year : 
  jennys_cat_expenditure 50 500 25 200 = 625 := by
  sorry

end NUMINAMATH_CALUDE_jenny_cat_expenditure_first_year_l1483_148371


namespace NUMINAMATH_CALUDE_green_yarn_length_l1483_148357

theorem green_yarn_length :
  ∀ (green_length red_length : ℕ),
  red_length = 3 * green_length + 8 →
  green_length + red_length = 632 →
  green_length = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_green_yarn_length_l1483_148357


namespace NUMINAMATH_CALUDE_system_solution_existence_l1483_148316

theorem system_solution_existence (a : ℝ) : 
  (∃ x y : ℝ, y = (x + |x|) / x ∧ (x - a)^2 = y + a) ↔ 
  (a > -1 ∧ a ≤ 0) ∨ (a > 0 ∧ a < 1) ∨ (a ≥ 1 ∧ a ≤ 2) ∨ (a > 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1483_148316


namespace NUMINAMATH_CALUDE_number_problem_l1483_148389

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 12 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1483_148389


namespace NUMINAMATH_CALUDE_representation_of_231_l1483_148330

theorem representation_of_231 : 
  ∃ (list : List ℕ), (list.sum = 231) ∧ (list.prod = 231) := by
  sorry

end NUMINAMATH_CALUDE_representation_of_231_l1483_148330


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l1483_148322

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l1483_148322


namespace NUMINAMATH_CALUDE_cos_210_degrees_l1483_148321

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l1483_148321


namespace NUMINAMATH_CALUDE_power_of_power_three_cubed_fourth_l1483_148320

theorem power_of_power_three_cubed_fourth : (3^3)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_cubed_fourth_l1483_148320


namespace NUMINAMATH_CALUDE_warehouse_bins_count_l1483_148355

/-- Calculates the total number of bins in a warehouse given specific conditions. -/
def totalBins (totalCapacity : ℕ) (twentyTonBins : ℕ) (twentyTonCapacity : ℕ) (fifteenTonCapacity : ℕ) : ℕ :=
  twentyTonBins + (totalCapacity - twentyTonBins * twentyTonCapacity) / fifteenTonCapacity

/-- Theorem stating that under given conditions, the total number of bins is 30. -/
theorem warehouse_bins_count :
  totalBins 510 12 20 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_bins_count_l1483_148355


namespace NUMINAMATH_CALUDE_range_of_m_l1483_148361

def P (x : ℝ) : Prop := x^2 - 4*x - 12 ≤ 0

def Q (x m : ℝ) : Prop := |x - m| ≤ m^2

theorem range_of_m : 
  ∀ m : ℝ, (∀ x : ℝ, P x → Q x m) ∧ 
            (∃ x : ℝ, Q x m ∧ ¬P x) ∧ 
            (∃ x : ℝ, P x) 
  ↔ m ≤ -3 ∨ m > 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1483_148361


namespace NUMINAMATH_CALUDE_number_problem_l1483_148315

theorem number_problem (x : ℝ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1483_148315


namespace NUMINAMATH_CALUDE_abs_not_always_positive_l1483_148332

theorem abs_not_always_positive : ¬ (∀ x : ℝ, |x| > 0) := by
  sorry

end NUMINAMATH_CALUDE_abs_not_always_positive_l1483_148332


namespace NUMINAMATH_CALUDE_caesars_rental_cost_is_800_l1483_148305

/-- Caesar's room rental cost -/
def caesars_rental_cost : ℝ := sorry

/-- Caesar's per-person meal cost -/
def caesars_meal_cost : ℝ := 30

/-- Venus Hall's room rental cost -/
def venus_rental_cost : ℝ := 500

/-- Venus Hall's per-person meal cost -/
def venus_meal_cost : ℝ := 35

/-- Number of guests at which the costs are equal -/
def equal_cost_guests : ℕ := 60

theorem caesars_rental_cost_is_800 :
  caesars_rental_cost = 800 :=
by
  have h : caesars_rental_cost + caesars_meal_cost * equal_cost_guests =
           venus_rental_cost + venus_meal_cost * equal_cost_guests :=
    sorry
  sorry

end NUMINAMATH_CALUDE_caesars_rental_cost_is_800_l1483_148305


namespace NUMINAMATH_CALUDE_george_money_left_l1483_148341

def monthly_income : ℕ := 240

def donation : ℕ := monthly_income / 2

def remaining_after_donation : ℕ := monthly_income - donation

def groceries_cost : ℕ := 20

def amount_left : ℕ := remaining_after_donation - groceries_cost

theorem george_money_left : amount_left = 100 := by
  sorry

end NUMINAMATH_CALUDE_george_money_left_l1483_148341


namespace NUMINAMATH_CALUDE_petr_speed_l1483_148365

theorem petr_speed (total_distance : ℝ) (ivan_speed : ℝ) (remaining_distance : ℝ) (time : ℝ) :
  total_distance = 153 →
  ivan_speed = 46 →
  remaining_distance = 24 →
  time = 1.5 →
  ∃ petr_speed : ℝ,
    petr_speed = 40 ∧
    total_distance - remaining_distance = (ivan_speed + petr_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_petr_speed_l1483_148365


namespace NUMINAMATH_CALUDE_complex_simplification_l1483_148343

/-- The imaginary unit -/
axiom I : ℂ

/-- The property of the imaginary unit -/
axiom I_squared : I^2 = -1

/-- Theorem stating the equality of the complex expressions -/
theorem complex_simplification : 7 * (4 - 2*I) - 2*I * (7 - 3*I) = 22 - 28*I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1483_148343


namespace NUMINAMATH_CALUDE_triangle_problem_l1483_148306

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + b = 5, c = √7, and 4sin²((A + B)/2) - cos(2C) = 7/2,
    then the measure of angle C is π/3 and the area of triangle ABC is 3√3/2 -/
theorem triangle_problem (a b c A B C : Real) : 
  a + b = 5 →
  c = Real.sqrt 7 →
  4 * Real.sin (A + B) ^ 2 / 4 - Real.cos (2 * C) = 7 / 2 →
  C = π / 3 ∧ 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1483_148306


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1483_148353

/-- Given two rotations about the same point Q:
    1. A 735-degree clockwise rotation of point P to point R
    2. A y-degree counterclockwise rotation of point P to the same point R
    where y < 360, prove that y = 345 degrees. -/
theorem rotation_equivalence (y : ℝ) (h1 : y < 360) : 
  (735 % 360 : ℝ) + y = 360 → y = 345 := by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1483_148353


namespace NUMINAMATH_CALUDE_geometric_figure_x_length_l1483_148313

theorem geometric_figure_x_length 
  (total_area : ℝ)
  (square1_side : ℝ → ℝ)
  (square2_side : ℝ → ℝ)
  (triangle_leg1 : ℝ → ℝ)
  (triangle_leg2 : ℝ → ℝ)
  (h1 : total_area = 1000)
  (h2 : ∀ x, square1_side x = 3 * x)
  (h3 : ∀ x, square2_side x = 4 * x)
  (h4 : ∀ x, triangle_leg1 x = 3 * x)
  (h5 : ∀ x, triangle_leg2 x = 4 * x)
  (h6 : ∀ x, (square1_side x)^2 + (square2_side x)^2 + 1/2 * (triangle_leg1 x) * (triangle_leg2 x) = total_area) :
  ∃ x : ℝ, x = 10 * Real.sqrt 31 / 31 := by
  sorry

end NUMINAMATH_CALUDE_geometric_figure_x_length_l1483_148313


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1483_148359

/-- A point M with coordinates (a-2, a+1) lies on the x-axis if and only if its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (a + 1 = 0 ∧ (a - 2, a + 1) = (-3, 0)) ↔ (a - 2, a + 1) = (-3, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1483_148359


namespace NUMINAMATH_CALUDE_phillips_jars_l1483_148350

-- Define the given quantities
def cucumbers : ℕ := 10
def initial_vinegar : ℕ := 100
def pickles_per_cucumber : ℕ := 6
def pickles_per_jar : ℕ := 12
def vinegar_per_jar : ℕ := 10
def remaining_vinegar : ℕ := 60

-- Define the function to calculate the number of jars
def number_of_jars : ℕ :=
  min
    (cucumbers * pickles_per_cucumber / pickles_per_jar)
    ((initial_vinegar - remaining_vinegar) / vinegar_per_jar)

-- Theorem statement
theorem phillips_jars :
  number_of_jars = 4 :=
sorry

end NUMINAMATH_CALUDE_phillips_jars_l1483_148350


namespace NUMINAMATH_CALUDE_fraction_simplification_l1483_148327

theorem fraction_simplification :
  (3 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (3 * Real.sqrt 2) / 38 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1483_148327


namespace NUMINAMATH_CALUDE_triangle_problem_l1483_148336

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  2 * a * Real.sin A = (2 * b + c) * Real.sin B + (2 * c + b) * Real.sin C →
  a = 7 →
  a * (15 * Real.sqrt 3 / 14) / 2 = b * c * Real.sin A →
  (A = 2 * π / 3 ∧ ((b = 3 ∧ c = 5) ∨ (b = 5 ∧ c = 3))) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l1483_148336


namespace NUMINAMATH_CALUDE_common_divisors_9240_10010_l1483_148338

theorem common_divisors_9240_10010 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 10010) (Finset.range 10011)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10010_l1483_148338


namespace NUMINAMATH_CALUDE_perpendicular_feet_circle_area_l1483_148323

/-- Given two points in the plane, calculate the area of the circle described by the perpendicular
    feet and find its floor. -/
theorem perpendicular_feet_circle_area (B C : ℝ × ℝ) (h_B : B = (20, 14)) (h_C : C = (18, 0)) :
  let midpoint := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let radius := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / 2
  let area := π * radius^2
  area = 50 * π ∧ Int.floor area = 157 := by sorry

end NUMINAMATH_CALUDE_perpendicular_feet_circle_area_l1483_148323


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_expression_l1483_148373

theorem simplify_and_evaluate (a b : ℝ) :
  (a - b)^2 - 2*a*(a + b) + (a + 2*b)*(a - 2*b) = -4*a*b - 3*b^2 :=
by sorry

theorem evaluate_expression :
  let a : ℝ := -1
  let b : ℝ := 4
  (a - b)^2 - 2*a*(a + b) + (a + 2*b)*(a - 2*b) = -32 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_expression_l1483_148373


namespace NUMINAMATH_CALUDE_students_with_one_fruit_l1483_148301

theorem students_with_one_fruit (total_apples : Nat) (total_bananas : Nat) (both_fruits : Nat) 
  (h1 : total_apples = 12)
  (h2 : total_bananas = 8)
  (h3 : both_fruits = 5) :
  (total_apples - both_fruits) + (total_bananas - both_fruits) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_with_one_fruit_l1483_148301


namespace NUMINAMATH_CALUDE_pens_distribution_ways_l1483_148387

/-- The number of ways to distribute n identical objects among k recipients,
    where each recipient must receive at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The number of ways to distribute 9 pens among 3 friends,
    where each friend must receive at least one pen. -/
def distribute_pens : ℕ := distribute 9 3

theorem pens_distribution_ways : distribute_pens = 28 := by sorry

end NUMINAMATH_CALUDE_pens_distribution_ways_l1483_148387


namespace NUMINAMATH_CALUDE_no_prime_solution_l1483_148377

theorem no_prime_solution : ¬∃ p : ℕ, Nat.Prime p ∧ p^3 + 6*p^2 + 4*p + 28 = 6*p^2 + 17*p + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1483_148377


namespace NUMINAMATH_CALUDE_robotics_club_neither_l1483_148349

/-- The number of students in the robotics club who take neither computer science nor electronics -/
theorem robotics_club_neither (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ) 
  (h_total : total = 60)
  (h_cs : cs = 42)
  (h_elec : elec = 35)
  (h_both : both = 25) :
  total - (cs + elec - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_neither_l1483_148349


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l1483_148302

/-- Q(N, k) represents the probability that no blue ball is adjacent to the red ball -/
def Q (N k : ℕ) : ℚ := (N + 1 : ℚ) / (N + k + 1 : ℚ)

/-- The smallest sum of N and k satisfying the conditions -/
def smallest_sum : ℕ := 4

theorem smallest_sum_proof :
  ∀ N k : ℕ,
    (N + k) % 4 = 0 →
    Q N k < 7/9 →
    N + k ≥ smallest_sum :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l1483_148302


namespace NUMINAMATH_CALUDE_solution_concentration_l1483_148386

/-- Theorem: Concentration of solution to be added to achieve target concentration --/
theorem solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (drain_volume : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 50)
  (h2 : initial_concentration = 0.6)
  (h3 : drain_volume = 35)
  (h4 : final_concentration = 0.46)
  : ∃ (x : ℝ), 
    (initial_volume - drain_volume) * initial_concentration + drain_volume * x = 
    initial_volume * final_concentration ∧ 
    x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_solution_concentration_l1483_148386


namespace NUMINAMATH_CALUDE_linear_regression_equation_l1483_148318

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are positively correlated --/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Calculates the sample mean of a variable --/
def sample_mean (x : ℝ → ℝ) : ℝ := sorry

/-- Checks if a point lies on the regression line --/
def point_on_line (model : LinearRegression) (x y : ℝ) : Prop :=
  y = model.slope * x + model.intercept

theorem linear_regression_equation 
  (x y : ℝ → ℝ) 
  (h_corr : positively_correlated x y)
  (h_mean_x : sample_mean x = 2)
  (h_mean_y : sample_mean y = 3)
  : ∃ (model : LinearRegression), 
    model.slope = 2 ∧ 
    model.intercept = -1 ∧ 
    point_on_line model 2 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_equation_l1483_148318


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l1483_148379

theorem factorization_of_polynomial (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 - x + 3) ∧
  (∀ a b c d e f : ℤ, (-8*x^2 + x - 3) = a*x^2 + b*x + c ∧
                       (8*x^2 - x + 3) = d*x^2 + e*x + f ∧
                       a < d) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l1483_148379


namespace NUMINAMATH_CALUDE_train_length_l1483_148395

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60.994720422366214 →
  man_speed = 5 →
  passing_time = 6 →
  ∃ (train_length : ℝ), 
    109.98 < train_length ∧ train_length < 110 :=
by sorry


end NUMINAMATH_CALUDE_train_length_l1483_148395


namespace NUMINAMATH_CALUDE_club_president_secretary_choices_l1483_148390

/-- A club with boys and girls -/
structure Club where
  total : ℕ
  boys : ℕ
  girls : ℕ

/-- The number of ways to choose a president (boy) and secretary (girl) from a club -/
def choosePresidentAndSecretary (c : Club) : ℕ :=
  c.boys * c.girls

/-- Theorem stating that for a club with 30 members (18 boys and 12 girls),
    the number of ways to choose a president and secretary is 216 -/
theorem club_president_secretary_choices :
  let c : Club := { total := 30, boys := 18, girls := 12 }
  choosePresidentAndSecretary c = 216 := by
  sorry

end NUMINAMATH_CALUDE_club_president_secretary_choices_l1483_148390


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1483_148384

/-- Given a triangle ABC with side lengths a and b, and angles A, B, and C,
    prove that C > B > A when a = 1, b = √3, A = 30°, and B is acute. -/
theorem triangle_angle_inequality (a b : ℝ) (A B C : ℝ) : 
  a = 1 → 
  b = Real.sqrt 3 → 
  A = π / 6 → 
  0 < B ∧ B < π / 2 → 
  a * Real.sin B = b * Real.sin A →
  A + B + C = π →
  C > B ∧ B > A := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1483_148384


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1483_148382

theorem arithmetic_operations :
  ((-3) + (-1) = -4) ∧
  (0 - 11 = -11) ∧
  (97 - (-3) = 100) ∧
  ((-7) * 5 = -35) ∧
  ((-8) / (-1/4) = 32) ∧
  ((-2/3)^3 = -8/27) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1483_148382


namespace NUMINAMATH_CALUDE_leo_marbles_count_l1483_148309

/-- The number of marbles in each pack -/
def marbles_per_pack : ℕ := 10

/-- The fraction of packs given to Manny -/
def manny_fraction : ℚ := 1/4

/-- The fraction of packs given to Neil -/
def neil_fraction : ℚ := 1/8

/-- The number of packs Leo kept for himself -/
def leo_packs : ℕ := 25

/-- The total number of packs Leo had initially -/
def total_packs : ℕ := 40

/-- The total number of marbles Leo had initially -/
def total_marbles : ℕ := total_packs * marbles_per_pack

theorem leo_marbles_count :
  manny_fraction * total_packs + neil_fraction * total_packs + leo_packs = total_packs ∧
  total_marbles = 400 :=
sorry

end NUMINAMATH_CALUDE_leo_marbles_count_l1483_148309


namespace NUMINAMATH_CALUDE_fibonacci_fraction_bound_l1483_148312

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_fraction_bound (a b n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : n ≥ 2) :
  ((fib n / fib (n - 1) < a / b ∧ a / b < fib (n + 1) / fib n) ∨
   (fib (n + 1) / fib n < a / b ∧ a / b < fib n / fib (n - 1))) →
  b ≥ fib (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_fraction_bound_l1483_148312


namespace NUMINAMATH_CALUDE_front_view_of_stack_map_l1483_148399

/-- Represents a stack map with four columns --/
structure StackMap :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Represents the front view of a stack map --/
def FrontView := List ℕ

/-- Computes the front view of a stack map --/
def computeFrontView (sm : StackMap) : FrontView :=
  [sm.A, sm.B, sm.C, sm.D]

/-- Theorem: The front view of the given stack map is [3, 5, 2, 4] --/
theorem front_view_of_stack_map :
  let sm : StackMap := { A := 3, B := 5, C := 2, D := 4 }
  computeFrontView sm = [3, 5, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_front_view_of_stack_map_l1483_148399


namespace NUMINAMATH_CALUDE_popcorn_distribution_l1483_148358

/-- Given the conditions of the popcorn problem, prove that each of Jared's friends can eat 60 pieces of popcorn. -/
theorem popcorn_distribution (pieces_per_serving : ℕ) (jared_pieces : ℕ) (num_friends : ℕ) (total_servings : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_pieces = 90)
  (h3 : num_friends = 3)
  (h4 : total_servings = 9) :
  (total_servings * pieces_per_serving - jared_pieces) / num_friends = 60 :=
by sorry

end NUMINAMATH_CALUDE_popcorn_distribution_l1483_148358


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1483_148383

/-- A quadratic function with axis of symmetry at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) :
  (∀ x, f b c (2 - x) = f b c (2 + x)) →  -- axis of symmetry at x = 2
  (∀ x₁ x₂, x₁ < x₂ → f b c x₁ > f b c x₂ → f b c x₂ > f b c (2*x₂ - x₁)) →  -- opens upwards
  f b c 2 < f b c 1 ∧ f b c 1 < f b c 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1483_148383


namespace NUMINAMATH_CALUDE_odometer_difference_l1483_148328

theorem odometer_difference (initial_reading final_reading : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : final_reading = 584.3) :
  final_reading - initial_reading = 372 := by
  sorry

end NUMINAMATH_CALUDE_odometer_difference_l1483_148328


namespace NUMINAMATH_CALUDE_min_value_3m_plus_n_l1483_148374

/-- Given a triangle ABC with point G satisfying the centroid condition,
    and points M on AB and N on AC with specific vector relationships,
    prove that the minimum value of 3m + n is 4/3 + 2√3/3 -/
theorem min_value_3m_plus_n (A B C G M N : ℝ × ℝ) (m n : ℝ) :
  (G.1 - A.1 + G.1 - B.1 + G.1 - C.1 = 0 ∧
   G.2 - A.2 + G.2 - B.2 + G.2 - C.2 = 0) →
  (∃ t : ℝ, M = (1 - t) • A + t • B ∧
            N = (1 - t) • A + t • C) →
  (M.1 - A.1 = m * (B.1 - A.1) ∧
   M.2 - A.2 = m * (B.2 - A.2)) →
  (N.1 - A.1 = n * (C.1 - A.1) ∧
   N.2 - A.2 = n * (C.2 - A.2)) →
  m > 0 →
  n > 0 →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 3 * m + n ≤ 3 * m' + n') →
  3 * m + n = 4/3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3m_plus_n_l1483_148374


namespace NUMINAMATH_CALUDE_remaining_eggs_l1483_148352

theorem remaining_eggs (initial_eggs : ℕ) (morning_eaten : ℕ) (afternoon_eaten : ℕ) :
  initial_eggs = 20 → morning_eaten = 4 → afternoon_eaten = 3 →
  initial_eggs - (morning_eaten + afternoon_eaten) = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_eggs_l1483_148352


namespace NUMINAMATH_CALUDE_average_licks_to_center_l1483_148351

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks
def num_people : ℕ := 5

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  (total_licks : ℚ) / num_people = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_licks_to_center_l1483_148351


namespace NUMINAMATH_CALUDE_no_valid_n_l1483_148348

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l1483_148348


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1483_148376

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1483_148376


namespace NUMINAMATH_CALUDE_tan_product_l1483_148347

theorem tan_product (α β : Real) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) : 
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_l1483_148347


namespace NUMINAMATH_CALUDE_deposit_exceeds_target_on_saturday_l1483_148354

def initial_deposit : ℕ := 2
def multiplication_factor : ℕ := 3
def target_amount : ℕ := 500 * 100  -- Convert $500 to cents

def deposit_on_day (n : ℕ) : ℕ :=
  initial_deposit * multiplication_factor ^ n

def total_deposit (n : ℕ) : ℕ :=
  (List.range (n + 1)).map deposit_on_day |>.sum

def days_of_week := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

theorem deposit_exceeds_target_on_saturday :
  (total_deposit 5 ≤ target_amount) ∧ 
  (total_deposit 6 > target_amount) ∧
  (days_of_week[(6 : ℕ) % 7] = "Saturday") := by
  sorry

end NUMINAMATH_CALUDE_deposit_exceeds_target_on_saturday_l1483_148354
