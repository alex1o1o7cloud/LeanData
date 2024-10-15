import Mathlib

namespace NUMINAMATH_CALUDE_sphere_volume_in_specific_cone_l1459_145967

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle perpendicular to the base. -/
  vertex_angle : ℝ

/-- Calculate the volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed sphere in the specific cone. -/
theorem sphere_volume_in_specific_cone :
  let cone : ConeWithSphere := { base_diameter := 24, vertex_angle := 90 }
  sphere_volume cone = 288 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_in_specific_cone_l1459_145967


namespace NUMINAMATH_CALUDE_dividend_proof_l1459_145901

theorem dividend_proof : 
  let dividend : ℕ := 11889708
  let divisor : ℕ := 12
  let quotient : ℕ := 990809
  dividend = divisor * quotient := by sorry

end NUMINAMATH_CALUDE_dividend_proof_l1459_145901


namespace NUMINAMATH_CALUDE_rabbit_carrot_problem_l1459_145909

theorem rabbit_carrot_problem (rabbit_holes fox_holes : ℕ) : 
  rabbit_holes * 3 = fox_holes * 5 →
  fox_holes = rabbit_holes - 6 →
  rabbit_holes * 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrot_problem_l1459_145909


namespace NUMINAMATH_CALUDE_number_of_jeans_to_wash_l1459_145904

/-- The number of shirts Alex has to wash -/
def shirts : ℕ := 18

/-- The number of pants Alex has to wash -/
def pants : ℕ := 12

/-- The number of sweaters Alex has to wash -/
def sweaters : ℕ := 17

/-- The maximum number of items the washing machine can wash per cycle -/
def items_per_cycle : ℕ := 15

/-- The time in minutes each washing cycle takes -/
def minutes_per_cycle : ℕ := 45

/-- The total time in hours it takes to wash all clothes -/
def total_wash_time : ℕ := 3

/-- Theorem stating the number of jeans Alex has to wash -/
theorem number_of_jeans_to_wash : 
  ∃ (jeans : ℕ), 
    (shirts + pants + sweaters + jeans) = 
      (total_wash_time * 60 / minutes_per_cycle) * items_per_cycle ∧
    jeans = 13 := by
  sorry

end NUMINAMATH_CALUDE_number_of_jeans_to_wash_l1459_145904


namespace NUMINAMATH_CALUDE_georges_trivia_score_l1459_145992

/-- George's trivia game score calculation -/
theorem georges_trivia_score :
  ∀ (first_half_correct second_half_correct points_per_question : ℕ),
    first_half_correct = 6 →
    second_half_correct = 4 →
    points_per_question = 3 →
    (first_half_correct + second_half_correct) * points_per_question = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_georges_trivia_score_l1459_145992


namespace NUMINAMATH_CALUDE_square_difference_63_57_l1459_145903

theorem square_difference_63_57 : 63^2 - 57^2 = 720 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_square_difference_63_57_l1459_145903


namespace NUMINAMATH_CALUDE_baseball_groups_l1459_145937

/-- The number of groups formed from baseball players -/
def number_of_groups (new_players returning_players players_per_group : ℕ) : ℕ :=
  (new_players + returning_players) / players_per_group

/-- Theorem: The number of groups formed is 9 -/
theorem baseball_groups :
  number_of_groups 48 6 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_baseball_groups_l1459_145937


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1459_145935

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 5) :
  (9 / a + 16 / b + 25 / c) ≥ 30 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 5 ∧ 9 / a₀ + 16 / b₀ + 25 / c₀ = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1459_145935


namespace NUMINAMATH_CALUDE_highest_throw_is_37_l1459_145991

def highest_throw (christine_first : ℕ) (janice_first_diff : ℕ) 
                  (christine_second_diff : ℕ) (christine_third_diff : ℕ) 
                  (janice_third_diff : ℕ) : ℕ :=
  let christine_first := christine_first
  let janice_first := christine_first - janice_first_diff
  let christine_second := christine_first + christine_second_diff
  let janice_second := janice_first * 2
  let christine_third := christine_second + christine_third_diff
  let janice_third := christine_first + janice_third_diff
  max christine_first (max christine_second (max christine_third 
    (max janice_first (max janice_second janice_third))))

theorem highest_throw_is_37 : 
  highest_throw 20 4 10 4 17 = 37 := by
  sorry

end NUMINAMATH_CALUDE_highest_throw_is_37_l1459_145991


namespace NUMINAMATH_CALUDE_min_max_problem_l1459_145974

theorem min_max_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (xy = 15 → x + y ≥ 2 * Real.sqrt 15 ∧ (x + y = 2 * Real.sqrt 15 ↔ x = Real.sqrt 15 ∧ y = Real.sqrt 15)) ∧
  (x + y = 15 → x * y ≤ 225 / 4 ∧ (x * y = 225 / 4 ↔ x = 15 / 2 ∧ y = 15 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_min_max_problem_l1459_145974


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1459_145963

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  is_in_second_quadrant (-7 : ℝ) (3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1459_145963


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1459_145956

/-- The first term of the geometric series -/
def a₁ : ℚ := 7/8

/-- The second term of the geometric series -/
def a₂ : ℚ := -21/32

/-- The third term of the geometric series -/
def a₃ : ℚ := 63/128

/-- The common ratio of the geometric series -/
def r : ℚ := -3/4

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1459_145956


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1459_145976

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 + a * Complex.I) / (1 - Complex.I) = -2 - Complex.I →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1459_145976


namespace NUMINAMATH_CALUDE_income_growth_equation_l1459_145980

theorem income_growth_equation (x : ℝ) : 
  let initial_income : ℝ := 12000
  let final_income : ℝ := 14520
  initial_income * (1 + x)^2 = final_income := by
  sorry

end NUMINAMATH_CALUDE_income_growth_equation_l1459_145980


namespace NUMINAMATH_CALUDE_reciprocal_statements_l1459_145977

def reciprocal (n : ℕ+) : ℚ := 1 / n.val

theorem reciprocal_statements : 
  (¬(reciprocal 4 + reciprocal 8 = reciprocal 12)) ∧
  (¬(reciprocal 9 - reciprocal 3 = reciprocal 6)) ∧
  (reciprocal 3 * reciprocal 9 = reciprocal 27) ∧
  ((reciprocal 15) / (reciprocal 3) = reciprocal 5) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_statements_l1459_145977


namespace NUMINAMATH_CALUDE_simplify_expression_l1459_145921

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 64) - Real.sqrt (9 + 1/4))^2 = 69/4 - 2 * Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1459_145921


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l1459_145987

theorem point_coordinates_sum (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  let slope : ℝ := (3 - 0) / (x - 0)
  slope = 3/4 → x + 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l1459_145987


namespace NUMINAMATH_CALUDE_power_of_two_with_nines_l1459_145989

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, ∃ m : ℕ,
    (2^n % 10^k = m) ∧ 
    (∃ count : ℕ, count ≥ k/2 ∧ 
      (∀ i : ℕ, i < k → 
        ((m / 10^i) % 10 = 9 → count > 0) ∧
        ((m / 10^i) % 10 ≠ 9 → count = count))) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_with_nines_l1459_145989


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l1459_145979

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x² = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_is_quadratic_l1459_145979


namespace NUMINAMATH_CALUDE_valid_last_score_l1459_145999

def scores : List Nat := [65, 72, 75, 79, 82, 86, 90, 98]

def isIntegerAverage (sublist : List Nat) : Prop :=
  (sublist.sum * 100) % sublist.length = 0

def isValidLastScore (last : Nat) : Prop :=
  ∀ n : Nat, n ≥ 1 → n < 8 → 
    isIntegerAverage (scores.take n ++ [last])

theorem valid_last_score : 
  isValidLastScore 79 := by sorry

end NUMINAMATH_CALUDE_valid_last_score_l1459_145999


namespace NUMINAMATH_CALUDE_soda_cost_theorem_l1459_145944

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℝ := 2

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℝ := 3

/-- The cost of Bob's fruit drink in dollars -/
def fruit_drink_cost : ℝ := 2

/-- The cost of Andy's soda in dollars -/
def soda_cost : ℝ := 4

/-- Andy's total spending in dollars -/
def andy_spending : ℝ := 2 * hamburger_cost + soda_cost

/-- Bob's total spending in dollars -/
def bob_spending : ℝ := 2 * sandwich_cost + fruit_drink_cost

theorem soda_cost_theorem : 
  andy_spending = bob_spending → soda_cost = 4 := by sorry

end NUMINAMATH_CALUDE_soda_cost_theorem_l1459_145944


namespace NUMINAMATH_CALUDE_prove_correct_statements_l1459_145950

-- Define the types of relationships
inductive Relationship
| Functional
| Correlation

-- Define the properties of relationships
def isDeterministic (r : Relationship) : Prop :=
  match r with
  | Relationship.Functional => True
  | Relationship.Correlation => False

-- Define regression analysis
def regressionAnalysis (r : Relationship) : Prop :=
  match r with
  | Relationship.Functional => False
  | Relationship.Correlation => True

-- Define the set of correct statements
def correctStatements : Set Nat :=
  {1, 2, 4}

-- Theorem to prove
theorem prove_correct_statements :
  (isDeterministic Relationship.Functional) ∧
  (¬isDeterministic Relationship.Correlation) ∧
  (regressionAnalysis Relationship.Correlation) →
  correctStatements = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_prove_correct_statements_l1459_145950


namespace NUMINAMATH_CALUDE_employee_count_l1459_145924

theorem employee_count (avg_salary : ℝ) (salary_increase : ℝ) (manager_salary : ℝ)
  (h1 : avg_salary = 1500)
  (h2 : salary_increase = 600)
  (h3 : manager_salary = 14100) :
  ∃ n : ℕ, 
    (n : ℝ) * avg_salary + manager_salary = ((n : ℝ) + 1) * (avg_salary + salary_increase) ∧
    n = 20 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l1459_145924


namespace NUMINAMATH_CALUDE_product_of_third_and_fourth_primes_above_20_l1459_145913

def third_prime_above_20 : ℕ := 31

def fourth_prime_above_20 : ℕ := 37

theorem product_of_third_and_fourth_primes_above_20 :
  third_prime_above_20 * fourth_prime_above_20 = 1147 := by
  sorry

end NUMINAMATH_CALUDE_product_of_third_and_fourth_primes_above_20_l1459_145913


namespace NUMINAMATH_CALUDE_find_n_l1459_145971

theorem find_n : ∃ n : ℕ, (2^3 * 8^3 = 2^(2*n)) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1459_145971


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_169_l1459_145946

theorem greatest_prime_factor_of_169 : ∃ p : ℕ, p.Prime ∧ p ∣ 169 ∧ ∀ q : ℕ, q.Prime → q ∣ 169 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_169_l1459_145946


namespace NUMINAMATH_CALUDE_abc_remainder_l1459_145930

theorem abc_remainder (a b c : ℕ) : 
  a < 9 → b < 9 → c < 9 →
  (a + 3*b + 2*c) % 9 = 3 →
  (2*a + 2*b + 3*c) % 9 = 6 →
  (3*a + b + 2*c) % 9 = 1 →
  (a*b*c) % 9 = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_l1459_145930


namespace NUMINAMATH_CALUDE_first_group_size_l1459_145962

/-- Given a work that takes 25 days for some men to complete and 21 days for 50 men to complete,
    prove that the number of men in the first group is 42. -/
theorem first_group_size (days_first : ℕ) (days_second : ℕ) (men_second : ℕ) :
  days_first = 25 →
  days_second = 21 →
  men_second = 50 →
  (men_second * days_second : ℕ) = days_first * (42 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l1459_145962


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1459_145984

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1459_145984


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l1459_145960

/-- Calculates the total loan amount given the down payment, monthly payment, and loan duration in years. -/
def totalLoanAmount (downPayment : ℕ) (monthlyPayment : ℕ) (years : ℕ) : ℕ :=
  downPayment + monthlyPayment * (years * 12)

/-- Theorem stating that a loan with a $10,000 down payment and $600 monthly payments for 5 years totals $46,000. -/
theorem loan_amount_calculation :
  totalLoanAmount 10000 600 5 = 46000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l1459_145960


namespace NUMINAMATH_CALUDE_percentage_difference_l1459_145982

theorem percentage_difference (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwz : w = 1.152 * z) : 
  w = 0.8 * x := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1459_145982


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l1459_145908

/-- Theorem: Distance between a point on a parabola and its focus
For a parabola defined by y^2 = 3x, if a point M on the parabola is at a distance
of 1 from the y-axis, then the distance between point M and the focus of the
parabola is 7/4. -/
theorem parabola_point_focus_distance
  (M : ℝ × ℝ) -- Point M on the parabola
  (h_on_parabola : M.2^2 = 3 * M.1) -- M is on the parabola y^2 = 3x
  (h_distance_from_y_axis : M.1 = 1) -- M is at distance 1 from y-axis
  : ∃ F : ℝ × ℝ, -- There exists a focus F
    (F.1 = 3/4 ∧ F.2 = 0) ∧ -- The focus is at (3/4, 0)
    Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 7/4 -- Distance between M and F is 7/4
  := by sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l1459_145908


namespace NUMINAMATH_CALUDE_logarithm_proportionality_l1459_145918

theorem logarithm_proportionality (P K a b : ℝ) 
  (hP : P > 0) (hK : K > 0) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  (Real.log P / Real.log a) / (Real.log P / Real.log b) = 
  (Real.log K / Real.log a) / (Real.log K / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_proportionality_l1459_145918


namespace NUMINAMATH_CALUDE_restaurant_customer_prediction_l1459_145986

theorem restaurant_customer_prediction 
  (breakfast_customers : ℕ) 
  (lunch_customers : ℕ) 
  (dinner_customers : ℕ) 
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87) :
  2 * (breakfast_customers + lunch_customers + dinner_customers) = 574 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_customer_prediction_l1459_145986


namespace NUMINAMATH_CALUDE_inequality_solution_l1459_145994

-- Define the given inequality and its solution set
def given_inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3*x + 2 > 0
def solution_set (b : ℝ) (x : ℝ) : Prop := x < 1 ∨ x > b

-- Define the values to be proven
def a_value : ℝ := 1
def b_value : ℝ := 2

-- Define the new inequality
def new_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 - (2*m + 1)*x + 2 < 0

-- Define the solution sets for different m values
def solution_set_m_zero (x : ℝ) : Prop := x > 2
def solution_set_m_gt_half (m : ℝ) (x : ℝ) : Prop := 1/m < x ∧ x < 2
def solution_set_m_half : Set ℝ := ∅
def solution_set_m_between_zero_half (m : ℝ) (x : ℝ) : Prop := 2 < x ∧ x < 1/m
def solution_set_m_neg (m : ℝ) (x : ℝ) : Prop := x < 1/m ∨ x > 2

-- State the theorem
theorem inequality_solution :
  (∀ x, given_inequality a_value x ↔ solution_set b_value x) ∧
  (∀ m x, new_inequality m x ↔
    (m = 0 ∧ solution_set_m_zero x) ∨
    (m > 1/2 ∧ solution_set_m_gt_half m x) ∨
    (m = 1/2 ∧ x ∈ solution_set_m_half) ∨
    (0 < m ∧ m < 1/2 ∧ solution_set_m_between_zero_half m x) ∨
    (m < 0 ∧ solution_set_m_neg m x)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1459_145994


namespace NUMINAMATH_CALUDE_ball_difference_l1459_145931

theorem ball_difference (blue red : ℕ) 
  (h1 : red - 152 = (blue + 152) + 346) : 
  red - blue = 650 := by
sorry

end NUMINAMATH_CALUDE_ball_difference_l1459_145931


namespace NUMINAMATH_CALUDE_composition_equality_condition_l1459_145917

theorem composition_equality_condition (m n p q : ℝ) :
  let f : ℝ → ℝ := λ x ↦ m * x + n
  let g : ℝ → ℝ := λ x ↦ p * x + q
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) :=
by sorry

end NUMINAMATH_CALUDE_composition_equality_condition_l1459_145917


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l1459_145972

theorem shopkeeper_profit_percentage 
  (cost_price selling_price_profit selling_price_loss : ℕ)
  (h1 : selling_price_loss = 540)
  (h2 : selling_price_profit = 900)
  (h3 : cost_price = 720)
  (h4 : selling_price_loss = (75 * cost_price) / 100) :
  (selling_price_profit - cost_price) * 100 / cost_price = 25 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l1459_145972


namespace NUMINAMATH_CALUDE_product_of_sums_l1459_145947

theorem product_of_sums (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 35)
  (h_sum_prod : a * b + b * c + c * a = 320)
  (h_prod : a * b * c = 600) :
  (a + b) * (b + c) * (c + a) = 10600 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l1459_145947


namespace NUMINAMATH_CALUDE_triangle_problem_l1459_145912

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : a - c = (Real.sqrt 6 / 6) * b) 
  (h2 : Real.sin B = Real.sqrt 6 * Real.sin C) : 
  Real.cos A = Real.sqrt 6 / 4 ∧ 
  Real.sin (2 * A + π / 6) = (3 * Real.sqrt 5 - 1) / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1459_145912


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l1459_145902

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l1459_145902


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l1459_145983

/-- Given a line with equation 3x + 4y + 5 = 0, prove its slope is -3/4 and y-intercept is -5/4 -/
theorem line_slope_and_intercept :
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y + 5 = 0}
  ∃ m b : ℝ, m = -3/4 ∧ b = -5/4 ∧ ∀ x y : ℝ, (x, y) ∈ line ↔ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l1459_145983


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l1459_145948

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

/-- Theorem stating that the number of lattice points on the given line segment is 6 -/
theorem lattice_points_on_segment : latticePointCount 5 26 40 146 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l1459_145948


namespace NUMINAMATH_CALUDE_max_visitable_halls_is_91_l1459_145936

/-- Represents a triangular castle divided into smaller triangular halls. -/
structure TriangularCastle where
  total_halls : ℕ
  side_length : ℝ
  hall_side_length : ℝ

/-- Represents a path through the castle halls. -/
def VisitPath (castle : TriangularCastle) := List ℕ

/-- Checks if a path is valid (no repeated visits). -/
def is_valid_path (castle : TriangularCastle) (path : VisitPath castle) : Prop :=
  path.length ≤ castle.total_halls ∧ path.Nodup

/-- The maximum number of halls that can be visited. -/
def max_visitable_halls (castle : TriangularCastle) : ℕ :=
  91

/-- Theorem stating that the maximum number of visitable halls is 91. -/
theorem max_visitable_halls_is_91 (castle : TriangularCastle) 
  (h1 : castle.total_halls = 100)
  (h2 : castle.side_length = 100)
  (h3 : castle.hall_side_length = 10) :
  ∀ (path : VisitPath castle), is_valid_path castle path → path.length ≤ max_visitable_halls castle :=
by sorry

end NUMINAMATH_CALUDE_max_visitable_halls_is_91_l1459_145936


namespace NUMINAMATH_CALUDE_five_volunteers_four_events_l1459_145923

/-- The number of ways to allocate volunteers to events --/
def allocationSchemes (volunteers : ℕ) (events : ℕ) : ℕ :=
  (volunteers.choose 2) * (events.factorial)

/-- Theorem stating the number of allocation schemes for 5 volunteers and 4 events --/
theorem five_volunteers_four_events :
  allocationSchemes 5 4 = 240 := by
  sorry

#eval allocationSchemes 5 4

end NUMINAMATH_CALUDE_five_volunteers_four_events_l1459_145923


namespace NUMINAMATH_CALUDE_roots_product_l1459_145998

def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

theorem roots_product (d e f : ℝ) :
  (∀ x, Q d e f x = 0 ↔ x = Real.cos (2*π/9) ∨ x = Real.cos (4*π/9) ∨ x = Real.cos (8*π/9)) →
  d * e * f = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_roots_product_l1459_145998


namespace NUMINAMATH_CALUDE_inequality_proof_l1459_145958

theorem inequality_proof (x y : ℝ) (hx : x < 0) (hy : y < 0) :
  x^4 / y^4 + y^4 / x^4 - x^2 / y^2 - y^2 / x^2 + x / y + y / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1459_145958


namespace NUMINAMATH_CALUDE_kelly_egg_income_l1459_145954

/-- Calculates the money made from selling eggs over a given period. -/
def money_from_eggs (num_chickens : ℕ) (eggs_per_day : ℕ) (price_per_dozen : ℕ) (num_weeks : ℕ) : ℕ :=
  let eggs_per_week := num_chickens * eggs_per_day * 7
  let total_eggs := eggs_per_week * num_weeks
  let dozens := total_eggs / 12
  dozens * price_per_dozen

/-- Proves that Kelly makes $280 in 4 weeks from selling eggs. -/
theorem kelly_egg_income : money_from_eggs 8 3 5 4 = 280 := by
  sorry

end NUMINAMATH_CALUDE_kelly_egg_income_l1459_145954


namespace NUMINAMATH_CALUDE_sprinkler_system_water_usage_l1459_145942

theorem sprinkler_system_water_usage 
  (morning_usage : ℝ) 
  (evening_usage : ℝ) 
  (total_water : ℝ) 
  (h1 : morning_usage = 4)
  (h2 : evening_usage = 6)
  (h3 : total_water = 50) :
  (total_water / (morning_usage + evening_usage) = 5) :=
by sorry

end NUMINAMATH_CALUDE_sprinkler_system_water_usage_l1459_145942


namespace NUMINAMATH_CALUDE_squarefree_term_existence_l1459_145985

/-- A positive integer is squarefree if it's not divisible by any square number greater than 1 -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = 1

/-- An arithmetic sequence of positive integers -/
def IsArithmeticSeq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem squarefree_term_existence :
  ∃ C : ℝ, C > 0 ∧
    ∀ a : ℕ → ℕ, IsArithmeticSeq a →
      IsSquarefree (Nat.gcd (a 1) (a 2)) →
        ∃ m : ℕ, m > 0 ∧ m ≤ ⌊C * (a 2)^2⌋ ∧ IsSquarefree (a m) :=
sorry

end NUMINAMATH_CALUDE_squarefree_term_existence_l1459_145985


namespace NUMINAMATH_CALUDE_bottles_drank_l1459_145993

def initial_bottles : ℕ := 17
def remaining_bottles : ℕ := 14

theorem bottles_drank : initial_bottles - remaining_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottles_drank_l1459_145993


namespace NUMINAMATH_CALUDE_rogers_coin_donation_l1459_145997

theorem rogers_coin_donation (pennies nickels dimes coins_left : ℕ) :
  pennies = 42 →
  nickels = 36 →
  dimes = 15 →
  coins_left = 27 →
  pennies + nickels + dimes - coins_left = 66 := by
  sorry

end NUMINAMATH_CALUDE_rogers_coin_donation_l1459_145997


namespace NUMINAMATH_CALUDE_prism_volume_theorem_l1459_145934

def prism_volume (AC_1 PQ phi : ℝ) (sin_phi cos_phi : ℝ) : Prop :=
  AC_1 = 3 ∧ 
  PQ = Real.sqrt 3 ∧ 
  phi = 30 * Real.pi / 180 ∧ 
  sin_phi = 1 / 2 ∧ 
  cos_phi = Real.sqrt 3 / 2 ∧ 
  ∃ (DL PK OK CL AL AC CC_1 : ℝ),
    DL = PK ∧
    DL = 1 / 2 * PQ * sin_phi ∧
    OK = 1 / 2 * PQ * cos_phi ∧
    CL / AL = (AC_1 / 2 - OK) / (AC_1 / 2 + OK) ∧
    AC = CL + AL ∧
    DL ^ 2 = CL * AL ∧
    CC_1 ^ 2 = AC_1 ^ 2 - AC ^ 2 ∧
    AC * DL * CC_1 = Real.sqrt 6 / 2

theorem prism_volume_theorem (AC_1 PQ phi sin_phi cos_phi : ℝ) :
  prism_volume AC_1 PQ phi sin_phi cos_phi → 
  ∃ (V : ℝ), V = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_theorem_l1459_145934


namespace NUMINAMATH_CALUDE_triangle_area_rational_l1459_145996

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a triangle with three IntPoints
structure IntTriangle where
  p1 : IntPoint
  p2 : IntPoint
  p3 : IntPoint

-- Function to calculate the area of a triangle given its vertices
def triangleArea (t : IntTriangle) : ℚ :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Theorem stating that the area of the triangle is rational
theorem triangle_area_rational (t : IntTriangle) 
  (h1 : t.p1.x - t.p1.y = 1)
  (h2 : t.p2.x - t.p2.y = 1)
  (h3 : t.p3.x - t.p3.y = 1) :
  ∃ (q : ℚ), triangleArea t = q :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_area_rational_l1459_145996


namespace NUMINAMATH_CALUDE_distance_of_problem_lines_l1459_145938

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  direction : ℝ × ℝ

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  sorry

/-- The specific parallel lines from the problem -/
def problem_lines : ParallelLines :=
  { point1 := (3, -4)
  , point2 := (-1, 1)
  , direction := (2, -5) }

theorem distance_of_problem_lines :
  distance_between_parallel_lines problem_lines = (150 * Real.sqrt 2) / 29 :=
sorry

end NUMINAMATH_CALUDE_distance_of_problem_lines_l1459_145938


namespace NUMINAMATH_CALUDE_bisection_interval_valid_l1459_145900

-- Define the function f(x) = x^3 + 5
def f (x : ℝ) : ℝ := x^3 + 5

-- Theorem statement
theorem bisection_interval_valid :
  ∃ (a b : ℝ), a = -2 ∧ b = 1 ∧ f a * f b < 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_interval_valid_l1459_145900


namespace NUMINAMATH_CALUDE_octal_arithmetic_l1459_145925

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def OctalNumber := List Nat

/-- Addition of two octal numbers -/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction of two octal numbers -/
def octalSub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Convert a natural number to its octal representation -/
def toOctal (n : Nat) : OctalNumber :=
  sorry

/-- Convert an octal number to its decimal representation -/
def fromOctal (o : OctalNumber) : Nat :=
  sorry

theorem octal_arithmetic :
  let a := [2, 5, 6]  -- 652₈
  let b := [7, 4, 1]  -- 147₈
  let c := [3, 5]     -- 53₈
  let result := [0, 5] -- 50₈
  octalSub (octalAdd a b) c = result := by
  sorry

end NUMINAMATH_CALUDE_octal_arithmetic_l1459_145925


namespace NUMINAMATH_CALUDE_tan_five_pi_quarters_l1459_145966

theorem tan_five_pi_quarters : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_quarters_l1459_145966


namespace NUMINAMATH_CALUDE_department_store_problem_l1459_145940

/-- The cost price per item in yuan -/
def cost_price : ℝ := 120

/-- The relationship between selling price and daily sales volume -/
def price_volume_relation (x y : ℝ) : Prop := x + y = 200

/-- The daily profit function -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * (200 - x)

theorem department_store_problem :
  (∃ a : ℝ, price_volume_relation 180 a ∧ a = 20) ∧
  (∃ x : ℝ, daily_profit x = 1600 ∧ x = 160) ∧
  (∀ m n : ℝ, m ≠ n → daily_profit (200 - m) = daily_profit (200 - n) → m + n = 80) :=
sorry

end NUMINAMATH_CALUDE_department_store_problem_l1459_145940


namespace NUMINAMATH_CALUDE_f_monotone_increasing_a_value_for_odd_function_l1459_145933

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_monotone_increasing (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

theorem a_value_for_odd_function :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_a_value_for_odd_function_l1459_145933


namespace NUMINAMATH_CALUDE_audrey_balls_l1459_145955

theorem audrey_balls (jake_balls : ℕ) (difference : ℕ) : 
  jake_balls = 7 → difference = 34 → jake_balls + difference = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_audrey_balls_l1459_145955


namespace NUMINAMATH_CALUDE_max_parts_five_lines_max_parts_recurrence_l1459_145916

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_parts m + (m + 1)

/-- Theorem stating the maximum number of parts for 5 lines -/
theorem max_parts_five_lines :
  max_parts 5 = 16 :=
by
  -- The proof goes here
  sorry

/-- Lemma for one line -/
lemma one_line_two_parts :
  max_parts 1 = 2 :=
by
  -- The proof goes here
  sorry

/-- Lemma for two lines -/
lemma two_lines_four_parts :
  max_parts 2 = 4 :=
by
  -- The proof goes here
  sorry

/-- Theorem proving the recurrence relation -/
theorem max_parts_recurrence (n : ℕ) :
  max_parts (n + 1) = max_parts n + (n + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_max_parts_five_lines_max_parts_recurrence_l1459_145916


namespace NUMINAMATH_CALUDE_complex_number_range_l1459_145990

theorem complex_number_range (z₁ z₂ : ℂ) (a : ℝ) : 
  z₁ = ((-1 + 3*I) * (1 - I) - (1 + 3*I)) / I →
  z₂ = z₁ + a * I →
  Complex.abs z₂ ≤ 2 →
  a ∈ Set.Icc (1 - Real.sqrt 3) (1 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_range_l1459_145990


namespace NUMINAMATH_CALUDE_correct_calculation_l1459_145905

theorem correct_calculation (x : ℝ) : 2 * x^2 - x^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1459_145905


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1459_145964

theorem painted_cube_theorem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1459_145964


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_value_l1459_145975

/-- The y-intercept of the common external tangent to two circles --/
def common_external_tangent_y_intercept : ℝ := sorry

/-- First circle center --/
def center1 : ℝ × ℝ := (1, 3)

/-- Second circle center --/
def center2 : ℝ × ℝ := (13, 6)

/-- First circle radius --/
def radius1 : ℝ := 3

/-- Second circle radius --/
def radius2 : ℝ := 6

theorem common_external_tangent_y_intercept_value :
  ∃ (m : ℝ), m > 0 ∧ 
  ∀ (x y : ℝ), y = m * x + common_external_tangent_y_intercept →
  ((x - center1.1)^2 + (y - center1.2)^2 = radius1^2 ∨
   (x - center2.1)^2 + (y - center2.2)^2 = radius2^2) →
  ∀ (x' y' : ℝ), (x' - center1.1)^2 + (y' - center1.2)^2 < radius1^2 →
                 (x' - center2.1)^2 + (y' - center2.2)^2 < radius2^2 →
                 y' ≠ m * x' + common_external_tangent_y_intercept := by
  sorry

#check common_external_tangent_y_intercept_value

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_value_l1459_145975


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l1459_145978

theorem exponential_function_sum_of_extrema (a : ℝ) (h : a > 0) : 
  (∀ x ∈ Set.Icc 0 1, ∃ y, a^x = y) → 
  (a^0 + a^1 = 4/3) → 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l1459_145978


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l1459_145968

theorem largest_integer_for_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by
  sorry

theorem six_satisfies_inequality :
  (6 : ℤ)^2 - 11*6 + 28 < 0 :=
by
  sorry

theorem seven_does_not_satisfy_inequality :
  (7 : ℤ)^2 - 11*7 + 28 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l1459_145968


namespace NUMINAMATH_CALUDE_total_path_is_2125_feet_l1459_145965

/-- Represents the scale of the plan in feet per inch -/
def scale : ℝ := 500

/-- Represents the initial path length on the plan in inches -/
def initial_path : ℝ := 3

/-- Represents the path extension on the plan in inches -/
def path_extension : ℝ := 1.25

/-- Calculates the total path length in feet -/
def total_path_length : ℝ := (initial_path + path_extension) * scale

/-- Theorem stating that the total path length is 2125 feet -/
theorem total_path_is_2125_feet : total_path_length = 2125 := by sorry

end NUMINAMATH_CALUDE_total_path_is_2125_feet_l1459_145965


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1459_145957

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) :
  ArithmeticSequence a →
  a 7 + a 9 = 16 →
  a 4 = 1 →
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1459_145957


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1459_145914

theorem quadratic_coefficient (b m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/5 = (x + m)^2 + 1/20) → 
  b = -2 * Real.sqrt (3/20) := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1459_145914


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l1459_145988

/-- A quadratic function y = x^2 - 4x + n, where n is a constant -/
def quadratic (n : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + n

theorem quadratic_point_relation (n : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  quadratic n x₁ = y₁ →
  quadratic n x₂ = y₂ →
  y₁ > y₂ →
  |x₁ - 2| > |x₂ - 2| :=
sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l1459_145988


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l1459_145995

/-- Proves that the wrongly written height of a boy is 176 cm given the following conditions:
  * There are 35 boys in a class
  * The initially calculated average height was 182 cm
  * One boy's height was recorded incorrectly
  * The boy's actual height is 106 cm
  * The correct average height is 180 cm
-/
theorem wrong_height_calculation (n : ℕ) (initial_avg correct_avg actual_height : ℝ) :
  n = 35 →
  initial_avg = 182 →
  correct_avg = 180 →
  actual_height = 106 →
  (n : ℝ) * initial_avg - (n : ℝ) * correct_avg + actual_height = 176 := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l1459_145995


namespace NUMINAMATH_CALUDE_transform_standard_deviation_l1459_145927

def standardDeviation (sample : Fin 10 → ℝ) : ℝ := sorry

theorem transform_standard_deviation 
  (x : Fin 10 → ℝ) 
  (h : standardDeviation x = 8) : 
  standardDeviation (fun i => 2 * x i - 1) = 16 := by sorry

end NUMINAMATH_CALUDE_transform_standard_deviation_l1459_145927


namespace NUMINAMATH_CALUDE_number_of_installments_l1459_145919

def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def monthly_installment : ℕ := 300
def cash_saving : ℕ := 4000

theorem number_of_installments : 
  (cash_price + cash_saving - deposit) / monthly_installment = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_installments_l1459_145919


namespace NUMINAMATH_CALUDE_sum_division_l1459_145928

/-- The problem of dividing a sum among four people with specific ratios -/
theorem sum_division (w x y z : ℝ) (total : ℝ) : 
  w > 0 ∧ 
  x = 0.8 * w ∧ 
  y = 0.65 * w ∧ 
  z = 0.45 * w ∧
  y = 78 →
  total = w + x + y + z ∧ total = 348 := by
  sorry

end NUMINAMATH_CALUDE_sum_division_l1459_145928


namespace NUMINAMATH_CALUDE_lcm_20_45_75_l1459_145920

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_20_45_75_l1459_145920


namespace NUMINAMATH_CALUDE_candles_used_l1459_145906

/-- Given a candle that lasts 8 nights when burned for 1 hour per night,
    calculate the number of candles used when burned for 2 hours per night for 24 nights -/
theorem candles_used
  (nights_per_candle : ℕ)
  (hours_per_night : ℕ)
  (total_nights : ℕ)
  (h1 : nights_per_candle = 8)
  (h2 : hours_per_night = 2)
  (h3 : total_nights = 24) :
  (total_nights * hours_per_night) / (nights_per_candle * 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candles_used_l1459_145906


namespace NUMINAMATH_CALUDE_julia_running_time_difference_l1459_145945

/-- Julia's running times with different shoes -/
theorem julia_running_time_difference (x : ℝ) : 
  let old_pace : ℝ := 10  -- minutes per mile in old shoes
  let new_pace : ℝ := 13  -- minutes per mile in new shoes
  let miles_for_known_difference : ℝ := 5
  let known_time_difference : ℝ := 15  -- minutes difference for 5 miles
  -- Prove that the time difference for x miles is 3x minutes
  (new_pace - old_pace) * x = 3 * x ∧
  -- Also prove that this is consistent with the given information for 5 miles
  (new_pace - old_pace) * miles_for_known_difference = known_time_difference
  := by sorry

end NUMINAMATH_CALUDE_julia_running_time_difference_l1459_145945


namespace NUMINAMATH_CALUDE_no_real_solutions_exponential_equation_l1459_145951

theorem no_real_solutions_exponential_equation :
  ∀ x : ℝ, (2 : ℝ)^(5*x+2) * (4 : ℝ)^(2*x+4) ≠ (8 : ℝ)^(3*x+7) := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_exponential_equation_l1459_145951


namespace NUMINAMATH_CALUDE_january_salary_l1459_145915

/-- The average salary calculation problem -/
theorem january_salary 
  (avg_jan_to_apr : ℝ) 
  (avg_feb_to_may : ℝ) 
  (may_salary : ℝ) 
  (h1 : avg_jan_to_apr = 8000)
  (h2 : avg_feb_to_may = 8500)
  (h3 : may_salary = 6500) :
  let jan_salary := 4 * avg_jan_to_apr - (4 * avg_feb_to_may - may_salary)
  jan_salary = 4500 := by
sorry


end NUMINAMATH_CALUDE_january_salary_l1459_145915


namespace NUMINAMATH_CALUDE_older_brother_height_l1459_145907

theorem older_brother_height
  (younger_height : ℝ)
  (your_height : ℝ)
  (older_height : ℝ)
  (h1 : younger_height = 1.1)
  (h2 : your_height = younger_height + 0.2)
  (h3 : older_height = your_height + 0.1) :
  older_height = 1.4 := by
sorry

end NUMINAMATH_CALUDE_older_brother_height_l1459_145907


namespace NUMINAMATH_CALUDE_interior_edges_sum_is_seven_l1459_145941

/-- A rectangular picture frame with specific properties -/
structure PictureFrame where
  /-- Width of the wood pieces used in the frame -/
  woodWidth : ℝ
  /-- Length of one outer edge of the frame -/
  outerEdgeLength : ℝ
  /-- Exposed area of the frame (excluding the picture) -/
  exposedArea : ℝ

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
def interiorEdgesSum (frame : PictureFrame) : ℝ :=
  sorry

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 7 inches -/
theorem interior_edges_sum_is_seven 
  (frame : PictureFrame)
  (h1 : frame.woodWidth = 2)
  (h2 : frame.outerEdgeLength = 6)
  (h3 : frame.exposedArea = 30) :
  interiorEdgesSum frame = 7 :=
sorry

end NUMINAMATH_CALUDE_interior_edges_sum_is_seven_l1459_145941


namespace NUMINAMATH_CALUDE_decimal_127_to_octal_has_three_consecutive_digits_l1459_145929

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

def is_consecutive_digits (digits : List ℕ) : Bool :=
  match digits with
  | [] => true
  | [_] => true
  | x :: y :: rest => (y = x + 1) && is_consecutive_digits (y :: rest)

theorem decimal_127_to_octal_has_three_consecutive_digits :
  let octal_digits := decimal_to_octal 127
  octal_digits.length = 3 ∧ is_consecutive_digits octal_digits = true :=
sorry

end NUMINAMATH_CALUDE_decimal_127_to_octal_has_three_consecutive_digits_l1459_145929


namespace NUMINAMATH_CALUDE_paint_coverage_per_quart_l1459_145911

/-- Represents the cost of paint per quart in dollars -/
def paint_cost_per_quart : ℝ := 3.20

/-- Represents the total cost to paint the cube in dollars -/
def total_paint_cost : ℝ := 192

/-- Represents the length of one edge of the cube in feet -/
def cube_edge_length : ℝ := 10

/-- Theorem stating the coverage of one quart of paint in square feet -/
theorem paint_coverage_per_quart : 
  (6 * cube_edge_length^2) / (total_paint_cost / paint_cost_per_quart) = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_coverage_per_quart_l1459_145911


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1459_145926

def is_valid_solution (x : ℝ) (a b c d e f : ℝ) : Prop :=
  x ≥ 1 ∧
  a = x ∧ b = x ∧ c = 1/x ∧ d = x ∧ e = x ∧ f = 1/x ∧
  a = max (1/b) (1/c) ∧
  b = max (1/c) (1/d) ∧
  c = max (1/d) (1/e) ∧
  d = max (1/e) (1/f) ∧
  e = max (1/f) (1/a) ∧
  f = max (1/a) (1/b)

theorem solution_satisfies_system :
  ∀ x : ℝ, x > 0 → is_valid_solution x x x (1/x) x x (1/x) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1459_145926


namespace NUMINAMATH_CALUDE_parallel_line_equation_l1459_145961

/-- The curve function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x - 4

theorem parallel_line_equation :
  let P : ℝ × ℝ := (-1, 2)
  let M : ℝ × ℝ := (1, 1)
  let m : ℝ := f' M.1  -- Slope of the tangent line at M
  let line (x y : ℝ) := 2 * x - y + 4 = 0
  (∀ x y, line x y ↔ y - P.2 = m * (x - P.1)) ∧  -- Point-slope form
  (f M.1 = M.2) ∧  -- M is on the curve
  (f' M.1 = m)  -- Slope at M equals the derivative
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l1459_145961


namespace NUMINAMATH_CALUDE_part1_part2_part3_l1459_145973

/-- Definition of X(n) function -/
def is_X_n_function (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℝ), ∀ (x : ℝ), f (2 * n - x) = f x

/-- Part 1: Prove that |x| and x^2 - x are X(n) functions -/
theorem part1 :
  (is_X_n_function (fun x => |x|)) ∧
  (is_X_n_function (fun x => x^2 - x)) :=
sorry

/-- Part 2: Prove k = -1 for the given parabola conditions -/
theorem part2 (k : ℝ) :
  (∀ x, (x^2 + k - 4) * (x^2 + k - 4) ≤ 0 → 
   ((0 - x)^2 + (k - 4))^2 = 3 * (x^2 + k - 4)^2) →
  k = -1 :=
sorry

/-- Part 3: Prove t = -2 or t = 0 for the given quadratic function conditions -/
theorem part3 (a b t : ℝ) :
  (∀ x, (a*x^2 + b*x - 4) = (a*(2-x)^2 + b*(2-x) - 4)) →
  (a*(-1)^2 + b*(-1) - 4 = 2) →
  (∀ x ∈ Set.Icc t (t+4), a*x^2 + b*x - 4 ≥ -6) →
  (∃ x ∈ Set.Icc t (t+4), a*x^2 + b*x - 4 = 12) →
  (t = -2 ∨ t = 0) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l1459_145973


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1459_145943

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoy plane in 3D space -/
def xoy_plane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xoy plane -/
def symmetric_wrt_xoy (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetric_point_xoy_plane :
  let M : Point3D := ⟨2, 5, 8⟩
  let N : Point3D := ⟨2, 5, -8⟩
  symmetric_wrt_xoy M N := by sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1459_145943


namespace NUMINAMATH_CALUDE_equation_solution_l1459_145952

theorem equation_solution (a : ℚ) : 
  (∀ x : ℚ, (2*a*x + 3) / (a - x) = 3/4 ↔ x = 1) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1459_145952


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1459_145932

theorem floor_ceiling_sum : ⌊(-3.75 : ℝ)⌋ + ⌈(34.25 : ℝ)⌉ = 31 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1459_145932


namespace NUMINAMATH_CALUDE_median_in_70_79_interval_l1459_145949

/-- Represents a score interval with its lower bound and number of students -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (count : ℕ)

/-- The distribution of scores for 100 students -/
def score_distribution : List ScoreInterval :=
  [⟨90, 22⟩, ⟨80, 18⟩, ⟨70, 20⟩, ⟨60, 15⟩, ⟨50, 25⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- The position of the median in the sorted list of scores -/
def median_position : ℕ := total_students / 2

/-- Finds the interval containing the median score -/
def find_median_interval (distribution : List ScoreInterval) (total : ℕ) (median_pos : ℕ) : ScoreInterval :=
  sorry

/-- Theorem stating that the interval 70-79 contains the median score -/
theorem median_in_70_79_interval :
  find_median_interval score_distribution total_students median_position = ⟨70, 20⟩ :=
sorry

end NUMINAMATH_CALUDE_median_in_70_79_interval_l1459_145949


namespace NUMINAMATH_CALUDE_balloon_count_l1459_145959

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem balloon_count : total_balloons = 81 := by sorry

end NUMINAMATH_CALUDE_balloon_count_l1459_145959


namespace NUMINAMATH_CALUDE_train_length_l1459_145953

/-- The length of a train given its crossing times over a platform and a signal pole -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 200)
  (h2 : platform_time = 30)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ), 
    train_length + platform_length = (train_length / pole_time) * platform_time ∧ 
    train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1459_145953


namespace NUMINAMATH_CALUDE_decimal_arithmetic_proof_l1459_145981

theorem decimal_arithmetic_proof : (5.92 + 2.4) - 3.32 = 5.00 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_proof_l1459_145981


namespace NUMINAMATH_CALUDE_square_of_binomial_l1459_145939

theorem square_of_binomial (k : ℚ) : 
  (∃ t u : ℚ, ∀ x, k * x^2 + 28 * x + 9 = (t * x + u)^2) → k = 196 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1459_145939


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1459_145922

theorem quadratic_roots_relation (p q : ℝ) : 
  (∀ x, x^2 - p^2*x + p*q = 0 ↔ ∃ y, y^2 + p*y + q = 0 ∧ x = y + 1) →
  ((p = -1 ∧ q = -1) ∨ (p = 2 ∧ q = -1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1459_145922


namespace NUMINAMATH_CALUDE_gain_percentage_l1459_145910

/-- Given an article sold for $110 with a gain of $10, prove that the gain percentage is 10%. -/
theorem gain_percentage (selling_price : ℝ) (gain : ℝ) (h1 : selling_price = 110) (h2 : gain = 10) :
  (gain / (selling_price - gain)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_gain_percentage_l1459_145910


namespace NUMINAMATH_CALUDE_alien_trees_conversion_l1459_145970

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (hundreds tens units : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + units * 7^0

/-- The problem statement --/
theorem alien_trees_conversion :
  base7ToBase10 2 5 3 = 136 := by
  sorry

end NUMINAMATH_CALUDE_alien_trees_conversion_l1459_145970


namespace NUMINAMATH_CALUDE_rotation_composition_implies_triangle_angles_l1459_145969

/-- Represents a rotation in 2D space -/
structure Rotation2D where
  angle : ℝ
  center : ℝ × ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Composition of rotations -/
def compose_rotations (r1 r2 r3 : Rotation2D) : Rotation2D :=
  sorry

/-- Check if a rotation is the identity transformation -/
def is_identity (r : Rotation2D) : Prop :=
  sorry

/-- Get the angle at a vertex of a triangle -/
def angle_at_vertex (t : Triangle) (v : ℝ × ℝ) : ℝ :=
  sorry

theorem rotation_composition_implies_triangle_angles 
  (α β γ : ℝ) (t : Triangle) (r_A r_B r_C : Rotation2D) :
  0 < α ∧ α < π →
  0 < β ∧ β < π →
  0 < γ ∧ γ < π →
  α + β + γ = π →
  r_A.angle = 2 * α →
  r_B.angle = 2 * β →
  r_C.angle = 2 * γ →
  r_A.center = t.A →
  r_B.center = t.B →
  r_C.center = t.C →
  is_identity (compose_rotations r_C r_B r_A) →
  angle_at_vertex t t.A = α ∧
  angle_at_vertex t t.B = β ∧
  angle_at_vertex t t.C = γ :=
by sorry

end NUMINAMATH_CALUDE_rotation_composition_implies_triangle_angles_l1459_145969
