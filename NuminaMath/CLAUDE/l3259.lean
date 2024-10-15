import Mathlib

namespace NUMINAMATH_CALUDE_min_segment_length_l3259_325963

/-- A cube with edge length 1 -/
structure Cube :=
  (edge_length : ℝ)
  (edge_length_eq : edge_length = 1)

/-- A point on the diagonal A₁D of the cube -/
structure PointM (cube : Cube) :=
  (coords : ℝ × ℝ × ℝ)

/-- A point on the edge CD₁ of the cube -/
structure PointN (cube : Cube) :=
  (coords : ℝ × ℝ × ℝ)

/-- The condition that MN is parallel to A₁ACC₁ -/
def is_parallel_to_diagonal_face (cube : Cube) (m : PointM cube) (n : PointN cube) : Prop :=
  sorry

/-- The length of segment MN -/
def segment_length (cube : Cube) (m : PointM cube) (n : PointN cube) : ℝ :=
  sorry

/-- The main theorem -/
theorem min_segment_length (cube : Cube) :
  ∃ (m : PointM cube) (n : PointN cube),
    is_parallel_to_diagonal_face cube m n ∧
    ∀ (m' : PointM cube) (n' : PointN cube),
      is_parallel_to_diagonal_face cube m' n' →
      segment_length cube m n ≤ segment_length cube m' n' ∧
      segment_length cube m n = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_segment_length_l3259_325963


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3259_325961

theorem solve_linear_equation :
  ∃ x : ℝ, 3 * x - 7 = 2 * x + 5 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3259_325961


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3259_325954

/-- Given two functions f and g, prove that |k| ≤ 2 is a sufficient but not necessary condition
for f(x) ≥ g(x) to hold for all x ∈ ℝ. -/
theorem sufficient_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + 3 ≥ k*x - 1) ↔ -6 ≤ k ∧ k ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3259_325954


namespace NUMINAMATH_CALUDE_smallest_number_problem_l3259_325965

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  b = 29 →
  max a (max b c) = b + 8 →
  min a (min b c) = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l3259_325965


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3259_325988

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (containedIn : Line → Plane → Prop)
variable (parallelTo : Line → Plane → Prop)
variable (planeparallel : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (m : Line) (α β : Plane) :
  containedIn m α → planeparallel α β → parallelTo m β := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3259_325988


namespace NUMINAMATH_CALUDE_exam_time_proof_l3259_325918

/-- Proves that the examination time is 3 hours given the specified conditions -/
theorem exam_time_proof (total_questions : ℕ) (type_a_problems : ℕ) (type_a_time : ℚ) :
  total_questions = 200 →
  type_a_problems = 15 →
  type_a_time = 25116279069767444 / 1000000000000000 →
  ∃ (type_b_time : ℚ),
    type_b_time > 0 ∧
    type_a_time = 2 * type_b_time * type_a_problems ∧
    (type_b_time * (total_questions - type_a_problems) + type_a_time) / 60 = 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_proof_l3259_325918


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l3259_325934

def f (x : ℝ) : ℝ := x^2 - x

theorem at_least_one_non_negative (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  f m ≥ 0 ∨ f n ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l3259_325934


namespace NUMINAMATH_CALUDE_pollution_data_median_mode_l3259_325922

def pollution_data : List ℕ := [31, 35, 31, 34, 30, 32, 31]

def median (l : List ℕ) : ℕ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem pollution_data_median_mode : 
  median pollution_data = 31 ∧ mode pollution_data = 31 := by sorry

end NUMINAMATH_CALUDE_pollution_data_median_mode_l3259_325922


namespace NUMINAMATH_CALUDE_blue_balls_count_l3259_325985

theorem blue_balls_count (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 35 → 
  removed = 5 → 
  prob = 5 / 21 → 
  (∃ initial : ℕ, 
    initial ≤ total ∧ 
    (initial - removed : ℚ) / (total - removed : ℚ) = prob ∧ 
    initial = 12) :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l3259_325985


namespace NUMINAMATH_CALUDE_round_robin_28_games_8_teams_l3259_325949

/-- The number of games in a single round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A single round-robin tournament with 28 games requires 8 teams -/
theorem round_robin_28_games_8_teams :
  ∃ (n : ℕ), n > 0 ∧ num_games n = 28 ∧ n = 8 := by sorry

end NUMINAMATH_CALUDE_round_robin_28_games_8_teams_l3259_325949


namespace NUMINAMATH_CALUDE_max_sum_of_arithmetic_progression_l3259_325935

def arithmetic_progression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem max_sum_of_arithmetic_progression (a : ℕ → ℕ) (d : ℕ) :
  arithmetic_progression a d →
  (∀ n, a n > 0) →
  a 3 = 13 →
  (∀ n, a (n + 1) > a n) →
  (a (a 1) + a (a 2) + a (a 3) + a (a 4) + a (a 5) ≤ 365) ∧
  (∃ a d, arithmetic_progression a d ∧
          (∀ n, a n > 0) ∧
          a 3 = 13 ∧
          (∀ n, a (n + 1) > a n) ∧
          a (a 1) + a (a 2) + a (a 3) + a (a 4) + a (a 5) = 365) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_arithmetic_progression_l3259_325935


namespace NUMINAMATH_CALUDE_dianes_honey_harvest_l3259_325946

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest (last_year_harvest : ℕ) (increase : ℕ) : 
  last_year_harvest = 2479 → increase = 6085 → last_year_harvest + increase = 8564 := by
  sorry

end NUMINAMATH_CALUDE_dianes_honey_harvest_l3259_325946


namespace NUMINAMATH_CALUDE_total_jokes_after_four_weeks_l3259_325978

def total_jokes (initial_jessy initial_alan initial_tom initial_emily : ℕ)
                (rate_jessy rate_alan rate_tom rate_emily : ℕ)
                (weeks : ℕ) : ℕ :=
  let jessy := initial_jessy * (rate_jessy ^ weeks - 1) / (rate_jessy - 1)
  let alan := initial_alan * (rate_alan ^ weeks - 1) / (rate_alan - 1)
  let tom := initial_tom * (rate_tom ^ weeks - 1) / (rate_tom - 1)
  let emily := initial_emily * (rate_emily ^ weeks - 1) / (rate_emily - 1)
  jessy + alan + tom + emily

theorem total_jokes_after_four_weeks :
  total_jokes 11 7 5 3 3 2 4 4 4 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_total_jokes_after_four_weeks_l3259_325978


namespace NUMINAMATH_CALUDE_fund_raising_exceeded_goal_l3259_325957

def fund_raising (ken_amount : ℝ) : Prop :=
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  let total_collected := ken_amount + mary_amount + scott_amount
  let goal := 4000
  ken_amount = 600 → total_collected - goal = 600

theorem fund_raising_exceeded_goal : fund_raising 600 := by
  sorry

end NUMINAMATH_CALUDE_fund_raising_exceeded_goal_l3259_325957


namespace NUMINAMATH_CALUDE_simplify_expression_l3259_325968

theorem simplify_expression (b : ℝ) (h : b ≠ 1) :
  1 - (1 / (1 + b / (1 - b))) = b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3259_325968


namespace NUMINAMATH_CALUDE_code_transformation_correct_l3259_325956

def initial_code : List (Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  [(4, 0, 2, 2), (0, 7, 1, 0), (4, 1, 9, 9)]

def complement_to_nine (n : Fin 10) : Fin 10 :=
  9 - n

def apply_rule (segment : Fin 10 × Fin 10 × Fin 10 × Fin 10) : Fin 10 × Fin 10 × Fin 10 × Fin 10 :=
  let (a, b, c, d) := segment
  (a, complement_to_nine b, c, complement_to_nine d)

def new_code : List (Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  initial_code.map apply_rule

theorem code_transformation_correct :
  new_code = [(4, 9, 2, 7), (0, 2, 1, 9), (4, 8, 9, 0)] :=
by sorry

end NUMINAMATH_CALUDE_code_transformation_correct_l3259_325956


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3259_325983

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3259_325983


namespace NUMINAMATH_CALUDE_unique_intersection_l3259_325908

/-- The value of k for which the graphs of y = kx^2 - 5x + 4 and y = 2x - 6 intersect at exactly one point -/
def intersection_k : ℚ := 49/40

/-- First equation: y = kx^2 - 5x + 4 -/
def equation1 (k : ℚ) (x : ℚ) : ℚ := k * x^2 - 5*x + 4

/-- Second equation: y = 2x - 6 -/
def equation2 (x : ℚ) : ℚ := 2*x - 6

/-- Theorem stating that the graphs intersect at exactly one point if and only if k = 49/40 -/
theorem unique_intersection :
  ∀ k : ℚ, (∃! x : ℚ, equation1 k x = equation2 x) ↔ k = intersection_k :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l3259_325908


namespace NUMINAMATH_CALUDE_correct_calculation_l3259_325920

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y + 2 * y * x^2 = 5 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3259_325920


namespace NUMINAMATH_CALUDE_mortgage_more_beneficial_l3259_325989

/-- Represents the annual dividend rate of the preferred shares -/
def dividend_rate : ℝ := 0.17

/-- Represents the annual interest rate of the mortgage loan -/
def mortgage_rate : ℝ := 0.125

/-- Theorem stating that the net return from keeping shares and taking a mortgage is positive -/
theorem mortgage_more_beneficial : dividend_rate - mortgage_rate > 0 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_more_beneficial_l3259_325989


namespace NUMINAMATH_CALUDE_power_of_product_l3259_325998

theorem power_of_product (a b : ℝ) : (a * b^3)^3 = a^3 * b^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3259_325998


namespace NUMINAMATH_CALUDE_no_divisors_between_2_and_100_l3259_325927

theorem no_divisors_between_2_and_100 (n : ℕ+) 
  (h : ∀ k ∈ Finset.range 99, (Finset.sum (Finset.range n) (fun i => (i + 1) ^ (k + 1))) % n = 0) :
  ∀ d ∈ Finset.range 99, d > 1 → ¬(d ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_divisors_between_2_and_100_l3259_325927


namespace NUMINAMATH_CALUDE_train_crossing_time_l3259_325990

/-- Proves that given two trains of equal length, where one train takes 15 seconds to cross a
    telegraph post, and they cross each other traveling in opposite directions in 7.5 seconds,
    the other train will take 5 seconds to cross the telegraph post. -/
theorem train_crossing_time
  (train_length : ℝ)
  (second_train_time : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : second_train_time = 15)
  (h3 : crossing_time = 7.5) :
  train_length / (train_length / second_train_time + train_length / crossing_time - train_length / second_train_time) = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3259_325990


namespace NUMINAMATH_CALUDE_compare_expressions_l3259_325909

theorem compare_expressions (x y : ℝ) (h1 : x * y > 0) (h2 : x ≠ y) :
  x^4 + 6*x^2*y^2 + y^4 > 4*x*y*(x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l3259_325909


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3259_325987

theorem unique_integer_solution : ∃! (x y : ℤ), x^4 + y^2 - 4*y + 4 = 4 := by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3259_325987


namespace NUMINAMATH_CALUDE_bus_children_count_l3259_325936

theorem bus_children_count (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  initial = 64 → joined = 14 → total = initial + joined → total = 78 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_count_l3259_325936


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l3259_325958

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

-- Define the property of having two zeros
def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem two_zeros_iff_a_positive :
  ∀ a : ℝ, has_two_zeros (f a) ↔ a > 0 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l3259_325958


namespace NUMINAMATH_CALUDE_minimally_intersecting_triples_count_l3259_325903

def Universe : Finset Nat := Finset.range 8

structure MinimallyIntersectingTriple (A B C : Finset Nat) : Prop where
  subset_universe : A ⊆ Universe ∧ B ⊆ Universe ∧ C ⊆ Universe
  intersection_size : (A ∩ B).card = 1 ∧ (B ∩ C).card = 1 ∧ (C ∩ A).card = 1
  empty_triple_intersection : (A ∩ B ∩ C).card = 0

def M : Nat := (Finset.powerset Universe).card

theorem minimally_intersecting_triples_count : M % 1000 = 344 := by
  sorry

end NUMINAMATH_CALUDE_minimally_intersecting_triples_count_l3259_325903


namespace NUMINAMATH_CALUDE_new_mean_after_adding_specific_problem_l3259_325947

theorem new_mean_after_adding (n : ℕ) (original_mean add_value : ℝ) :
  n > 0 →
  let new_mean := (n * original_mean + n * add_value) / n
  new_mean = original_mean + add_value :=
by sorry

theorem specific_problem :
  let n : ℕ := 15
  let original_mean : ℝ := 40
  let add_value : ℝ := 13
  (n * original_mean + n * add_value) / n = 53 :=
by sorry

end NUMINAMATH_CALUDE_new_mean_after_adding_specific_problem_l3259_325947


namespace NUMINAMATH_CALUDE_remainder_sum_factorials_l3259_325951

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_sum_factorials (n : ℕ) (h : n ≥ 50) :
  sum_factorials n % 24 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 24 :=
sorry

end NUMINAMATH_CALUDE_remainder_sum_factorials_l3259_325951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3259_325945

/-- Given an arithmetic sequence with first term 3² and last term 3⁴, 
    the middle term y is equal to 45. -/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (a 0 = 3^2) → 
    (a 2 = 3^4) → 
    (∀ n : ℕ, n < 2 → a (n + 1) - a n = a 1 - a 0) → 
    a 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3259_325945


namespace NUMINAMATH_CALUDE_reciprocal_power_l3259_325941

theorem reciprocal_power (a : ℝ) (h : a⁻¹ = -1) : a^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_l3259_325941


namespace NUMINAMATH_CALUDE_A_initial_investment_l3259_325910

/-- Represents the initial investment of A in dollars -/
def A_investment : ℝ := sorry

/-- Represents B's investment in dollars -/
def B_investment : ℝ := 9000

/-- Represents the number of months A invested -/
def A_months : ℕ := 12

/-- Represents the number of months B invested -/
def B_months : ℕ := 7

/-- Represents A's share in the profit ratio -/
def A_ratio : ℕ := 2

/-- Represents B's share in the profit ratio -/
def B_ratio : ℕ := 3

theorem A_initial_investment :
  A_investment * A_months * B_ratio = B_investment * B_months * A_ratio :=
sorry

end NUMINAMATH_CALUDE_A_initial_investment_l3259_325910


namespace NUMINAMATH_CALUDE_bargain_bin_books_theorem_l3259_325975

/-- Calculates the number of books in the bargain bin after two weeks of sales and additions. -/
def books_after_two_weeks (initial : ℕ) (sold_week1 sold_week2 added_week1 added_week2 : ℕ) : ℕ :=
  initial - sold_week1 + added_week1 - sold_week2 + added_week2

/-- Theorem stating that given the initial number of books and the changes during two weeks,
    the final number of books in the bargain bin is 391. -/
theorem bargain_bin_books_theorem :
  books_after_two_weeks 500 115 289 65 230 = 391 := by
  sorry

#eval books_after_two_weeks 500 115 289 65 230

end NUMINAMATH_CALUDE_bargain_bin_books_theorem_l3259_325975


namespace NUMINAMATH_CALUDE_order_of_constants_l3259_325924

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := Real.exp (1/2)
noncomputable def c : ℝ := Real.log 2 / Real.log 10

-- Theorem statement
theorem order_of_constants : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_order_of_constants_l3259_325924


namespace NUMINAMATH_CALUDE_a_is_four_l3259_325944

def rounds_to_9430 (a b : ℕ) : Prop :=
  9000 + 100 * a + 30 + b ≥ 9425 ∧ 9000 + 100 * a + 30 + b < 9435

theorem a_is_four (a b : ℕ) (h : rounds_to_9430 a b) : a = 4 :=
sorry

end NUMINAMATH_CALUDE_a_is_four_l3259_325944


namespace NUMINAMATH_CALUDE_simplify_fraction_l3259_325917

theorem simplify_fraction : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3259_325917


namespace NUMINAMATH_CALUDE_modular_arithmetic_proof_l3259_325960

theorem modular_arithmetic_proof :
  ∃ (a b : ℤ), (a * 7 ≡ 1 [ZMOD 63]) ∧ 
               (b * 13 ≡ 1 [ZMOD 63]) ∧ 
               ((3 * a + 5 * b) % 63 = 13) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_proof_l3259_325960


namespace NUMINAMATH_CALUDE_triangle_property_l3259_325970

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove A = 2π/3
  A = 2 * π / 3 ∧
  -- Prove maximum perimeter is 3 + 2√3
  (b + c ≤ 2 * Real.sqrt 3 ∧ a + b + c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3259_325970


namespace NUMINAMATH_CALUDE_total_money_l3259_325992

-- Define Tim's and Alice's money as fractions of a dollar
def tim_money : ℚ := 5/8
def alice_money : ℚ := 2/5

-- Theorem statement
theorem total_money :
  tim_money + alice_money = 1.025 := by sorry

end NUMINAMATH_CALUDE_total_money_l3259_325992


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_l3259_325906

def adult_price : ℝ := 10

def grandparent_discount : ℝ := 0.2
def child_discount : ℝ := 0.6

def grandparent_price : ℝ := adult_price * (1 - grandparent_discount)
def child_price : ℝ := adult_price * (1 - child_discount)

def total_cost : ℝ := 2 * grandparent_price + adult_price + child_price

theorem zoo_ticket_cost : total_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_l3259_325906


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3259_325929

theorem smallest_n_satisfying_conditions (n : ℕ) : 
  n > 10 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 2 → 
  n ≥ 27 ∧ 
  (∀ m : ℕ, m > 10 ∧ m % 4 = 3 ∧ m % 5 = 2 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3259_325929


namespace NUMINAMATH_CALUDE_all_PQ_pass_through_common_point_l3259_325969

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
structure Setup where
  S : Circle
  A : ℝ × ℝ
  B : ℝ × ℝ
  L : Line
  c : ℝ

-- Define the condition for X and Y
def satisfiesCondition (setup : Setup) (X Y : ℝ × ℝ) : Prop :=
  X ≠ Y ∧ 
  (X.1 - setup.A.1) * (Y.1 - setup.A.1) + (X.2 - setup.A.2) * (Y.2 - setup.A.2) = setup.c

-- Define the intersection points P and Q
def getIntersectionP (setup : Setup) (X : ℝ × ℝ) : ℝ × ℝ := sorry
def getIntersectionQ (setup : Setup) (Y : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the line PQ
def linePQ (P Q : ℝ × ℝ) : Line := ⟨P, Q⟩

-- Theorem statement
theorem all_PQ_pass_through_common_point (setup : Setup) :
  ∃ (commonPoint : ℝ × ℝ), ∀ (X Y : ℝ × ℝ),
    satisfiesCondition setup X Y →
    let P := getIntersectionP setup X
    let Q := getIntersectionQ setup Y
    let PQ := linePQ P Q
    -- The common point lies on line PQ
    (commonPoint.1 - PQ.point1.1) * (PQ.point2.2 - PQ.point1.2) = 
    (commonPoint.2 - PQ.point1.2) * (PQ.point2.1 - PQ.point1.1) :=
sorry

end NUMINAMATH_CALUDE_all_PQ_pass_through_common_point_l3259_325969


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l3259_325976

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) :
  (1/a + 1/b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    Real.sqrt 3 = Real.sqrt (3^a₀ * 3^b₀) ∧ 1/a₀ + 1/b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l3259_325976


namespace NUMINAMATH_CALUDE_inequality_propositions_l3259_325962

theorem inequality_propositions :
  ∃ (correct : Finset (Fin 4)), correct.card = 2 ∧
  (∀ i, i ∈ correct ↔
    (i = 0 ∧ (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b)) ∨
    (i = 1 ∧ (∀ a b c d : ℝ, a > b → c > d → a + c > b + d)) ∨
    (i = 2 ∧ (∀ a b c d : ℝ, a > b → c > d → a * c > b * d)) ∨
    (i = 3 ∧ (∀ a b : ℝ, a > b → 1 / a > 1 / b))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_propositions_l3259_325962


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3259_325928

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x > -3, 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3259_325928


namespace NUMINAMATH_CALUDE_min_value_of_P_sum_l3259_325952

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_of_P_sum (x y : ℝ) (h : x + y = 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (a b : ℝ), a + b = 0 → P a + P b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_P_sum_l3259_325952


namespace NUMINAMATH_CALUDE_correct_equation_representation_l3259_325955

/-- Represents a rectangular field with width and length in steps -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- The area of a rectangular field in square steps -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem stating that the equation x(x+12) = 864 correctly represents the problem -/
theorem correct_equation_representation (x : ℝ) :
  let field := RectangularField.mk x (x + 12)
  area field = 864 → x * (x + 12) = 864 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_representation_l3259_325955


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3259_325994

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with foci F₁ and F₂, and real axis length 2a -/
structure Hyperbola where
  F₁ : Point
  F₂ : Point
  a : ℝ

/-- Theorem: Perimeter of triangle ABF₂ in a hyperbola -/
theorem hyperbola_triangle_perimeter 
  (h : Hyperbola) 
  (A B : Point) 
  (m : ℝ) 
  (h_line : A.x = B.x ∧ A.x = h.F₁.x) -- A, B, and F₁ are collinear
  (h_on_hyperbola : 
    |A.x - h.F₂.x| + |A.y - h.F₂.y| - |A.x - h.F₁.x| - |A.y - h.F₁.y| = 2 * h.a ∧
    |B.x - h.F₂.x| + |B.y - h.F₂.y| - |B.x - h.F₁.x| - |B.y - h.F₁.y| = 2 * h.a)
  (h_AB : |A.x - B.x| + |A.y - B.y| = m) :
  |A.x - h.F₂.x| + |A.y - h.F₂.y| + |B.x - h.F₂.x| + |B.y - h.F₂.y| + m = 4 * h.a + 2 * m := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3259_325994


namespace NUMINAMATH_CALUDE_max_sum_after_pyramid_addition_l3259_325981

/-- Represents a polyhedron with faces, edges, and vertices -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Represents the result of adding a pyramid to a face of a polyhedron -/
structure PyramidAddition where
  newFaces : ℕ
  newEdges : ℕ
  newVertices : ℕ

/-- Calculates the sum of faces, edges, and vertices after adding a pyramid -/
def sumAfterPyramidAddition (p : Polyhedron) (pa : PyramidAddition) : ℕ :=
  (p.faces - 1 + pa.newFaces) + (p.edges + pa.newEdges) + (p.vertices + pa.newVertices)

/-- The pentagonal prism -/
def pentagonalPrism : Polyhedron :=
  { faces := 7, edges := 15, vertices := 10 }

/-- Adding a pyramid to a pentagonal face -/
def pentagonalFaceAddition : PyramidAddition :=
  { newFaces := 5, newEdges := 5, newVertices := 1 }

/-- Adding a pyramid to a quadrilateral face -/
def quadrilateralFaceAddition : PyramidAddition :=
  { newFaces := 4, newEdges := 4, newVertices := 1 }

/-- Theorem: The maximum sum of faces, edges, and vertices after adding a pyramid is 42 -/
theorem max_sum_after_pyramid_addition :
  (max 
    (sumAfterPyramidAddition pentagonalPrism pentagonalFaceAddition)
    (sumAfterPyramidAddition pentagonalPrism quadrilateralFaceAddition)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_after_pyramid_addition_l3259_325981


namespace NUMINAMATH_CALUDE_smallest_number_l3259_325913

theorem smallest_number (a b c d : ℝ) (ha : a = 0) (hb : b = -1/2) (hc : c = -1) (hd : d = Real.sqrt 2) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l3259_325913


namespace NUMINAMATH_CALUDE_parallelogram_point_C_l3259_325942

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x) ∧ (B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x) ∧ (D.y - A.y = C.y - B.y)

def InFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

theorem parallelogram_point_C : 
  ∀ (A B C D : Point),
    Parallelogram A B C D →
    InFirstQuadrant A →
    InFirstQuadrant B →
    InFirstQuadrant C →
    InFirstQuadrant D →
    A.x = 2 ∧ A.y = 3 →
    B.x = 7 ∧ B.y = 3 →
    D.x = 3 ∧ D.y = 7 →
    C.x = 8 ∧ C.y = 7 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_point_C_l3259_325942


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l3259_325907

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l3259_325907


namespace NUMINAMATH_CALUDE_library_wall_leftover_space_l3259_325967

theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (min_spacing : ℝ)
  (h_wall : wall_length = 15)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  (h_spacing : min_spacing = 0.5)
  : ∃ (n : ℕ), 
    n * (desk_length + bookcase_length + min_spacing) ≤ wall_length ∧
    (n + 1) * (desk_length + bookcase_length + min_spacing) > wall_length ∧
    wall_length - n * (desk_length + bookcase_length + min_spacing) = 3 :=
by sorry

end NUMINAMATH_CALUDE_library_wall_leftover_space_l3259_325967


namespace NUMINAMATH_CALUDE_units_digit_of_4_pow_3_pow_5_l3259_325972

theorem units_digit_of_4_pow_3_pow_5 : (4^(3^5)) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_4_pow_3_pow_5_l3259_325972


namespace NUMINAMATH_CALUDE_area_ratio_is_9_32_l3259_325977

-- Define the triangle XYZ
structure Triangle :=
  (XY YZ XZ : ℝ)

-- Define the points M, N, O
structure Points (t : Triangle) :=
  (p q r : ℝ)
  (p_pos : p > 0)
  (q_pos : q > 0)
  (r_pos : r > 0)
  (sum_eq : p + q + r = 3/4)
  (sum_sq_eq : p^2 + q^2 + r^2 = 1/2)

-- Define the function to calculate the ratio of areas
def areaRatio (t : Triangle) (pts : Points t) : ℝ :=
  -- The actual calculation of the ratio would go here
  sorry

-- State the theorem
theorem area_ratio_is_9_32 (t : Triangle) (pts : Points t) 
  (h1 : t.XY = 12) (h2 : t.YZ = 16) (h3 : t.XZ = 20) : 
  areaRatio t pts = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_9_32_l3259_325977


namespace NUMINAMATH_CALUDE_triangle_construction_from_feet_l3259_325932

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The foot of an altitude in a triangle -/
def altitude_foot (T : Triangle) (v : Point) : Point :=
  sorry

/-- The foot of an angle bisector in a triangle -/
def angle_bisector_foot (T : Triangle) (v : Point) : Point :=
  sorry

/-- Theorem: A unique triangle exists given the feet of two altitudes and one angle bisector -/
theorem triangle_construction_from_feet 
  (A₁ B₁ B₂ : Point) : 
  ∃! (T : Triangle), 
    altitude_foot T T.A = A₁ ∧ 
    altitude_foot T T.B = B₁ ∧ 
    angle_bisector_foot T T.B = B₂ :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_from_feet_l3259_325932


namespace NUMINAMATH_CALUDE_equation_solution_l3259_325948

theorem equation_solution :
  ∀ x : ℚ, (Real.sqrt (4 * x + 9) / Real.sqrt (8 * x + 9) = Real.sqrt 3 / 2) → x = 9/8 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3259_325948


namespace NUMINAMATH_CALUDE_brad_lemonade_profit_l3259_325986

/-- Calculates the net profit from a lemonade stand given the specified conditions. -/
def lemonade_stand_profit (
  glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ) : ℚ :=
  let total_glasses := glasses_per_gallon * gallons_made
  let glasses_sold := total_glasses - (glasses_drunk + glasses_unsold)
  let total_cost := cost_per_gallon * gallons_made
  let total_revenue := price_per_glass * glasses_sold
  total_revenue - total_cost

/-- Theorem stating that Brad's net profit is $14.00 given the specified conditions. -/
theorem brad_lemonade_profit :
  lemonade_stand_profit 16 3.5 2 1 5 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_brad_lemonade_profit_l3259_325986


namespace NUMINAMATH_CALUDE_batsman_highest_score_l3259_325940

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : overall_average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (highest_score + lowest_score : ℚ) = 
      total_innings * overall_average - (total_innings - 2) * average_excluding_extremes ∧
    highest_score = 202 :=
by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l3259_325940


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3259_325991

/-- Given two natural numbers m and n, where mn = 34^8 and m has a units digit of 4,
    prove that the units digit of n is 4. -/
theorem units_digit_of_n (m n : ℕ) 
  (h1 : m * n = 34^8)
  (h2 : m % 10 = 4) : 
  n % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3259_325991


namespace NUMINAMATH_CALUDE_middle_number_bounds_l3259_325979

theorem middle_number_bounds (a b c : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : a + b + c = 10) (h4 : a - c = 3) : 7/3 < b ∧ b < 13/3 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_bounds_l3259_325979


namespace NUMINAMATH_CALUDE_arithmetic_comparisons_l3259_325926

theorem arithmetic_comparisons :
  (80 / 4 > 80 / 5) ∧
  (16 * 21 > 14 * 22) ∧
  (32 * 25 = 16 * 50) ∧
  (320 / 8 < 320 / 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_comparisons_l3259_325926


namespace NUMINAMATH_CALUDE_next_simultaneous_visit_l3259_325995

def visit_interval_1 : ℕ := 6
def visit_interval_2 : ℕ := 8
def visit_interval_3 : ℕ := 9

theorem next_simultaneous_visit :
  Nat.lcm (Nat.lcm visit_interval_1 visit_interval_2) visit_interval_3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_visit_l3259_325995


namespace NUMINAMATH_CALUDE_colinear_vector_problem_l3259_325904

/-- Given vector a and b in ℝ², prove that if a = (1, -2), b is colinear with a, and |b| = 4|a|, then b = (4, -8) or b = (-4, 8) -/
theorem colinear_vector_problem (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  (∃ (k : ℝ), b = k • a) → 
  Real.sqrt ((b.1)^2 + (b.2)^2) = 4 * Real.sqrt ((a.1)^2 + (a.2)^2) → 
  b = (4, -8) ∨ b = (-4, 8) := by
sorry

end NUMINAMATH_CALUDE_colinear_vector_problem_l3259_325904


namespace NUMINAMATH_CALUDE_max_product_853_l3259_325930

def digits : List Nat := [3, 5, 6, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit_number (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  (three_digit_number a b c) * (two_digit_number d e)

theorem max_product_853 :
  ∀ a b c d e,
    is_valid_combination a b c d e →
    product 8 5 3 9 6 ≥ product a b c d e :=
by sorry

end NUMINAMATH_CALUDE_max_product_853_l3259_325930


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3259_325911

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) :
  ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3259_325911


namespace NUMINAMATH_CALUDE_function_shift_l3259_325997

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_shift (x : ℝ) : f (x + 1) = x^2 - 2*x - 3 → f x = x^2 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l3259_325997


namespace NUMINAMATH_CALUDE_elliptic_curve_solutions_l3259_325905

theorem elliptic_curve_solutions (p : ℕ) (hp : Nat.Prime p) :
  (∃ (S : Finset (Fin p × Fin p)),
    S.card = p ∧
    ∀ (x y : Fin p), (x, y) ∈ S ↔ y^2 ≡ x^3 + 4*x [ZMOD p]) ↔
  p = 2 ∨ p ≡ 3 [MOD 4] :=
sorry

end NUMINAMATH_CALUDE_elliptic_curve_solutions_l3259_325905


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l3259_325973

theorem unknown_blanket_rate (price1 price2 avg_price : ℕ) 
  (count1 count2 count_unknown : ℕ) (total_count : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 150 →
  count1 = 2 →
  count2 = 5 →
  count_unknown = 2 →
  total_count = count1 + count2 + count_unknown →
  ∃ (unknown_price : ℕ), 
    (count1 * price1 + count2 * price2 + count_unknown * unknown_price) / total_count = avg_price ∧
    unknown_price = 200 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l3259_325973


namespace NUMINAMATH_CALUDE_tan_double_special_angle_l3259_325950

/-- An angle with vertex at the origin, initial side on positive x-axis, and terminal side on y = 2x -/
structure SpecialAngle where
  θ : Real
  terminal_side : (x : Real) → y = 2 * x

theorem tan_double_special_angle (α : SpecialAngle) : Real.tan (2 * α.θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_special_angle_l3259_325950


namespace NUMINAMATH_CALUDE_tangent_line_to_unit_circle_l3259_325919

/-- The equation of the tangent line to the unit circle at point (a, b) -/
theorem tangent_line_to_unit_circle (a b : ℝ) (h : a^2 + b^2 = 1) :
  ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = (a*x + b*y - 1)^2 / (a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_unit_circle_l3259_325919


namespace NUMINAMATH_CALUDE_inequality_proof_l3259_325974

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (|x^2 + y^2|) / (x + y) < (|x^2 - y^2|) / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3259_325974


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l3259_325982

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is five-digit -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_sum_20 : 
  ∀ n : ℕ, is_five_digit n → sum_of_digits n = 20 → n ≤ 99200 := by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l3259_325982


namespace NUMINAMATH_CALUDE_function_value_at_three_l3259_325900

/-- Given a function f(x) = ax^4 + b cos(x) - x where f(-3) = 7, prove that f(3) = 1 -/
theorem function_value_at_three (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^4 + b * Real.cos x - x)
  (h2 : f (-3) = 7) : 
  f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3259_325900


namespace NUMINAMATH_CALUDE_exists_appropriate_ratio_in_small_interval_l3259_325915

-- Define a type for the cutting ratio
def CuttingRatio := {a : ℝ // 0 < a ∧ a < 1}

-- Define a predicate for appropriate cutting ratios
def isAppropriate (a : CuttingRatio) : Prop :=
  ∃ (n : ℕ), ∀ (w : ℝ), w > 0 → ∃ (w1 w2 : ℝ), w1 = w2 ∧ w1 + w2 = w ∧
  ∃ (cuts : List ℝ), cuts.length ≤ n ∧ 
    (∀ c ∈ cuts, c = a.val * w ∨ c = (1 - a.val) * w)

-- State the theorem
theorem exists_appropriate_ratio_in_small_interval :
  ∀ x : ℝ, 0 < x → x < 0.999 →
  ∃ a : CuttingRatio, x < a.val ∧ a.val < x + 0.001 ∧ isAppropriate a :=
sorry

end NUMINAMATH_CALUDE_exists_appropriate_ratio_in_small_interval_l3259_325915


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3259_325938

theorem consecutive_integers_product (n : ℤ) : 
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3259_325938


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3259_325966

theorem polynomial_evaluation (x : ℝ) (h : x = 4) : x^4 + x^3 + x^2 + x + 1 = 341 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3259_325966


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3259_325923

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3259_325923


namespace NUMINAMATH_CALUDE_tan_sum_15_30_l3259_325933

theorem tan_sum_15_30 : 
  ∀ (tan : Real → Real),
  (∀ α β, tan (α + β) = (tan α + tan β) / (1 - tan α * tan β)) →
  tan (45 * π / 180) = 1 →
  tan (15 * π / 180) + tan (30 * π / 180) + tan (15 * π / 180) * tan (30 * π / 180) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_15_30_l3259_325933


namespace NUMINAMATH_CALUDE_angle_measure_l3259_325971

theorem angle_measure (α : Real) : 
  (90 - α) + (90 - (180 - α)) = 90 → α = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3259_325971


namespace NUMINAMATH_CALUDE_sum_of_series_in_base7_l3259_325996

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Converts a number from base 7 to base 10 -/
def fromBase7 (n : ℕ) : ℕ :=
  sorry

/-- Computes the sum of an arithmetic series -/
def arithmeticSeriesSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_of_series_in_base7 :
  let last_term := fromBase7 33
  let sum := arithmeticSeriesSum last_term
  toBase7 sum = 606 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_in_base7_l3259_325996


namespace NUMINAMATH_CALUDE_veggies_per_day_l3259_325914

/-- The number of servings of veggies eaten in a week -/
def weekly_servings : ℕ := 21

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of servings of veggies eaten per day -/
def daily_servings : ℕ := weekly_servings / days_in_week

theorem veggies_per_day : daily_servings = 3 := by
  sorry

end NUMINAMATH_CALUDE_veggies_per_day_l3259_325914


namespace NUMINAMATH_CALUDE_triangle_theorem_l3259_325953

noncomputable section

theorem triangle_theorem (a b c A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin A = 4 * b * Real.sin B →
  a * c = Real.sqrt 5 * (a^2 - b^2 - c^2) →
  Real.cos A = -Real.sqrt 5 / 5 ∧
  Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l3259_325953


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l3259_325916

/-- The smallest positive angle θ, in degrees, that satisfies the given equation is 50°. -/
theorem smallest_angle_theorem : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ 
  Real.cos (θ * π / 180) = Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - 
                           Real.sin (20 * π / 180) - Real.cos (10 * π / 180) ∧
  θ = 50 ∧ 
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → 
    Real.cos (φ * π / 180) ≠ Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - 
                             Real.sin (20 * π / 180) - Real.cos (10 * π / 180) :=
by sorry


end NUMINAMATH_CALUDE_smallest_angle_theorem_l3259_325916


namespace NUMINAMATH_CALUDE_square_area_ratio_l3259_325984

/-- The ratio of the area of a square with side length x to the area of a square with side length 3x is 1/9 -/
theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3259_325984


namespace NUMINAMATH_CALUDE_triangle_angles_from_height_intersections_l3259_325993

/-- Given an acute-angled triangle ABC with circumscribed circle,
    let p, q, r be positive real numbers representing the ratio of arc lengths
    formed by the intersections of the extended heights with the circle.
    This theorem states the relationship between these ratios and the angles of the triangle. -/
theorem triangle_angles_from_height_intersections
  (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  let α := Real.pi / 2 * ((q + r) / (p + q + r))
  let β := Real.pi / 2 * (q / (p + q + r))
  let γ := Real.pi / 2 * (r / (p + q + r))
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_from_height_intersections_l3259_325993


namespace NUMINAMATH_CALUDE_circle_angle_theorem_l3259_325999

-- Define the circle and angles
def Circle (F : Point) : Prop := sorry

def angle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem circle_angle_theorem (F A B C D E : Point) :
  Circle F →
  angle B F C = 2 * angle A F B →
  angle C F D = 3 * angle A F B →
  angle D F E = 4 * angle A F B →
  angle E F A = 5 * angle A F B →
  angle B F C = 48 := by
  sorry

end NUMINAMATH_CALUDE_circle_angle_theorem_l3259_325999


namespace NUMINAMATH_CALUDE_train_speed_l3259_325931

/-- The speed of a train crossing a platform -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) 
  (h1 : train_length = 120)
  (h2 : platform_length = 130.02)
  (h3 : crossing_time = 15) : 
  ∃ (speed : Real), abs (speed - 60.0048) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3259_325931


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3259_325937

theorem arithmetic_calculations :
  (12 - (-18) + (-7) - 20 = 3) ∧ (-4 / (1/2) * 8 = -64) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3259_325937


namespace NUMINAMATH_CALUDE_power_fraction_multiply_simplify_fraction_two_thirds_cubed_times_half_l3259_325964

theorem power_fraction_multiply (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) = (a ^ 3 * c) / (b ^ 3 * d) :=
by sorry

theorem simplify_fraction_two_thirds_cubed_times_half :
  (2 / 3 : ℚ) ^ 3 * (1 / 2) = 4 / 27 :=
by sorry

end NUMINAMATH_CALUDE_power_fraction_multiply_simplify_fraction_two_thirds_cubed_times_half_l3259_325964


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_420_l3259_325921

theorem distinct_prime_factors_of_420 : Nat.card (Nat.factors 420).toFinset = 4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_420_l3259_325921


namespace NUMINAMATH_CALUDE_coefficient_of_a_is_one_l3259_325939

-- Define a monomial type
def Monomial := ℚ → ℕ → ℚ

-- Define the coefficient of a monomial
def coefficient (m : Monomial) : ℚ := m 1 0

-- Define the monomial 'a'
def a : Monomial := fun c n => if n = 1 then 1 else 0

-- Theorem statement
theorem coefficient_of_a_is_one : coefficient a = 1 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_a_is_one_l3259_325939


namespace NUMINAMATH_CALUDE_f_properties_l3259_325959

noncomputable def f (x : ℝ) := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) - 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi/3) (Real.pi/12), ∀ y ∈ Set.Icc (-Real.pi/3) (Real.pi/12), x ≤ y → f y ≤ f x) ∧
  (∀ x ∈ Set.Icc (Real.pi/12) (Real.pi/3), ∀ y ∈ Set.Icc (Real.pi/12) (Real.pi/3), x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3259_325959


namespace NUMINAMATH_CALUDE_original_paint_intensity_l3259_325925

theorem original_paint_intensity 
  (f : ℝ) 
  (h1 : f = 2/3)
  (h2 : (1 - f) * I + f * 0.3 = 0.4) : 
  I = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l3259_325925


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3259_325912

variable {a b c x1 x2 : ℝ}

theorem quadratic_roots_transformation (ha : a ≠ 0)
  (hroots : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ k, k * ((x1 + 2*x2) - x)* ((x2 + 2*x1) - x) = a^2 * x^2 + 3*a*b * x + 2*b^2 + a*c :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3259_325912


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l3259_325901

/-- A line in 3D space -/
structure Line3D where
  -- Define the line structure (omitted for brevity)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (omitted for brevity)

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def lineParallelToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def linePerpendicularToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to another line -/
def linePerpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to a plane and another line is parallel to the plane,
    then the two lines are perpendicular to each other -/
theorem perpendicular_parallel_implies_perpendicular
  (a b : Line3D) (α : Plane3D)
  (h1 : linePerpendicularToPlane a α)
  (h2 : lineParallelToPlane b α) :
  linePerpendicular a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l3259_325901


namespace NUMINAMATH_CALUDE_cubic_extrema_opposite_signs_l3259_325980

/-- A cubic function with coefficients p and q -/
def cubic_function (p q : ℝ) (x : ℝ) : ℝ := x^3 + p*x + q

/-- The derivative of the cubic function -/
def cubic_derivative (p : ℝ) (x : ℝ) : ℝ := 3*x^2 + p

/-- Condition for opposite signs of extremum points -/
def opposite_signs_condition (p q : ℝ) : Prop := 
  (q/2)^2 + (p/3)^3 < 0 ∧ p < 0

theorem cubic_extrema_opposite_signs 
  (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    cubic_derivative p x₁ = 0 ∧ 
    cubic_derivative p x₂ = 0 ∧ 
    cubic_function p q x₁ * cubic_function p q x₂ < 0) ↔ 
  opposite_signs_condition p q :=
sorry

end NUMINAMATH_CALUDE_cubic_extrema_opposite_signs_l3259_325980


namespace NUMINAMATH_CALUDE_oats_per_meal_is_four_l3259_325902

/-- The amount of oats each horse eats per meal, in pounds -/
def oats_per_meal (num_horses : ℕ) (grain_per_horse : ℕ) (total_food : ℕ) (num_days : ℕ) : ℚ :=
  let total_food_per_day := total_food / num_days
  let grain_per_day := num_horses * grain_per_horse
  let oats_per_day := total_food_per_day - grain_per_day
  oats_per_day / (2 * num_horses)

theorem oats_per_meal_is_four :
  oats_per_meal 4 3 132 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oats_per_meal_is_four_l3259_325902


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3259_325943

theorem distinct_prime_factors_count : 
  let n : ℕ := 101 * 103 * 107 * 109
  ∀ (is_prime_101 : Nat.Prime 101) 
    (is_prime_103 : Nat.Prime 103) 
    (is_prime_107 : Nat.Prime 107) 
    (is_prime_109 : Nat.Prime 109),
  Finset.card (Nat.factors n).toFinset = 4 := by
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3259_325943
