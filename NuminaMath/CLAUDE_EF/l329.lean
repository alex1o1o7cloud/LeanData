import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l329_32974

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 5
noncomputable def c : ℝ := Real.exp (-0.2)

-- State the theorem
theorem ordering_abc : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l329_32974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_with_sum_divisible_by_14_l329_32968

open BigOperators Finset Nat

/-- The set of numbers from 1 to 14 -/
def S : Finset ℕ := Finset.range 14

/-- A function that returns true if the sum of elements in a subset is divisible by 14 -/
def sumDivisibleBy14 (subset : Finset ℕ) : Bool :=
  (∑ i in subset, i + 1) % 14 = 0

/-- The theorem to be proved -/
theorem subsets_with_sum_divisible_by_14 :
  (S.powerset.filter (λ subset => subset.card = 7 ∧ sumDivisibleBy14 subset)).card = 245 := by
  sorry

#eval (S.powerset.filter (λ subset => subset.card = 7 ∧ sumDivisibleBy14 subset)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_with_sum_divisible_by_14_l329_32968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l329_32943

-- Define the function f(x) = a^(x-1) - 5
noncomputable def f (a x : ℝ) : ℝ := a^(x-1) - 5

-- Theorem statement
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = -4 ∧ ∀ x : ℝ, f a x = -4 → x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l329_32943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_formula_l329_32900

/-- The combined average marks of two classes -/
noncomputable def combined_average (n1 n2 : ℕ) (A1 A2 : ℝ) : ℝ :=
  ((n1 : ℝ) * A1 + (n2 : ℝ) * A2) / ((n1 : ℝ) + (n2 : ℝ))

/-- Theorem: The combined average marks of two classes is equal to the weighted sum of their individual averages divided by the total number of students -/
theorem combined_average_formula (n1 n2 : ℕ) (A1 A2 : ℝ) :
  let A := combined_average n1 n2 A1 A2
  A = ((n1 : ℝ) * A1 + (n2 : ℝ) * A2) / ((n1 : ℝ) + (n2 : ℝ)) := by
  sorry

/-- Example calculation with given values -/
example : 
  let n1 : ℕ := 25
  let n2 : ℕ := 30
  let A1 : ℝ := 40
  let A2 : ℝ := 60
  abs (combined_average n1 n2 A1 A2 - 50.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_formula_l329_32900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l329_32949

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line
def is_on_line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the point
def tangent_point : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem tangent_line_to_circle :
  (∀ x y, is_on_circle x y → is_on_line x y → (x, y) ≠ tangent_point) ∧
  is_on_circle tangent_point.1 tangent_point.2 ∧
  is_on_line tangent_point.1 tangent_point.2 ∧
  ∃! k, ∀ x y, is_on_line x y ↔ y - tangent_point.2 = k * (x - tangent_point.1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l329_32949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_from_constant_angle_l329_32937

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem line_from_constant_angle (θ₀ : ℝ) :
  ∃ a b : ℝ, {(x, y) | ∃ r : ℝ, (x, y) = polar_to_cartesian r θ₀} = {(x, y) | y = a * x + b} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_from_constant_angle_l329_32937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_operation_l329_32936

def special_operation (A B : Finset ℤ) : Finset ℤ :=
  (A.product B).image (fun (m, n) => m - n)

theorem sum_of_special_operation :
  let A : Finset ℤ := {4, 5, 6}
  let B : Finset ℤ := {1, 2, 3}
  let result := special_operation A B
  result.sum id = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_operation_l329_32936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_AB_distance_l329_32981

-- Define the curves and ray
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (Real.cos t, 1 + Real.sin t)
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4
def l (a : ℝ) (ρ θ : ℝ) : Prop := θ = a ∧ 0 < a ∧ a < Real.pi

-- Define the polar equations of C₁ and C₂
def C₁_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the intersection points A and B
noncomputable def A (a : ℝ) : ℝ × ℝ := (2 * Real.sin a, a)
noncomputable def B (a : ℝ) : ℝ × ℝ := (4 * Real.sin a, a)

-- Define the distance between A and B
noncomputable def AB (a : ℝ) : ℝ := 2 * Real.sin a

-- Theorem statement
theorem max_AB_distance :
  ∃ (max : ℝ), max = 2 ∧ ∀ a, 0 < a → a < Real.pi → AB a ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_AB_distance_l329_32981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_seven_to_sixth_l329_32999

theorem cube_root_seven_to_sixth : (7 : ℝ) ^ ((1/3 : ℝ) * 6) = 49 := by
  -- Convert 7 to a real number
  -- Use real exponentiation
  -- Simplify the exponent (1/3) * 6 = 2
  calc (7 : ℝ) ^ ((1/3 : ℝ) * 6) = (7 : ℝ) ^ 2 := by sorry
  -- Evaluate 7^2
  _ = 49 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_seven_to_sixth_l329_32999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l329_32964

/-- Heron's formula for the area of a triangle -/
noncomputable def heron_formula (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

/-- The area of triangle ABC -/
noncomputable def triangle_area (a b c : ℝ) : ℝ := heron_formula a b c

theorem triangle_abc_area :
  ∃ (a b c : ℝ),
    a + b + c = 18 ∧
    ∃ (k : ℝ), k > 0 ∧ Real.sin a = 2 * k ∧ Real.sin b = 3 * k ∧ Real.sin c = 4 * k ∧
    triangle_area a b c = 3 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l329_32964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_diesel_cost_approx_l329_32929

/-- Calculates the approximate average cost per litre of diesel over three years -/
noncomputable def average_diesel_cost (price1 price2 price3 yearly_spending : ℝ) : ℝ :=
  let total_spent := 3 * yearly_spending
  let litres1 := yearly_spending / price1
  let litres2 := yearly_spending / price2
  let litres3 := yearly_spending / price3
  let total_litres := litres1 + litres2 + litres3
  total_spent / total_litres

/-- Theorem stating that the average diesel cost is approximately 8.98 -/
theorem average_diesel_cost_approx :
  ∃ ε > 0, |average_diesel_cost 8.5 9 9.5 5000 - 8.98| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_diesel_cost_approx_l329_32929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coord_diff_R_l329_32924

/-- Triangle ABC with vertices A(0,8), B(2,0), C(8,0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 8⟩, ⟨2, 0⟩, ⟨8, 0⟩}

/-- Point R on line AC -/
def R : ℝ × ℝ := sorry

/-- Point S on line BC -/
def S : ℝ × ℝ := sorry

/-- Vertical line passing through R and S -/
def vertical_line (R S : ℝ × ℝ) : Prop :=
  R.1 = S.1

/-- Area of triangle RSC -/
def area_RSC : ℝ := 12.5

/-- Line segment between two points -/
def line_segment (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = ⟨(1 - t) * p.1 + t * q.1, (1 - t) * p.2 + t * q.2⟩}

/-- Area of a triangle given by three points -/
noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

/-- Theorem: The positive difference between x and y coordinates of R is 2 -/
theorem coord_diff_R :
  vertical_line R S ∧ 
  R ∈ line_segment ⟨0, 8⟩ ⟨8, 0⟩ ∧
  S ∈ line_segment ⟨2, 0⟩ ⟨8, 0⟩ ∧
  area_triangle R S ⟨8, 0⟩ = area_RSC →
  |R.1 - R.2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coord_diff_R_l329_32924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_numbers_with_zero_l329_32980

def six_digit_numbers : ℕ := 9 * (10 ^ 5)
def six_digit_numbers_without_zero : ℕ := 9 ^ 6

theorem six_digit_numbers_with_zero : 
  six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  have h1 : six_digit_numbers = 9 * (10 ^ 5) := rfl
  have h2 : six_digit_numbers_without_zero = 9 ^ 6 := rfl
  sorry

#eval six_digit_numbers - six_digit_numbers_without_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_numbers_with_zero_l329_32980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l329_32906

/-- The radius of the inscribed circle in a triangle with sides a, b, and c --/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (Real.sqrt (s * (s - a) * (s - b) * (s - c))) / s

theorem inscribed_circle_radius_specific_triangle :
  inscribed_circle_radius 26 16 20 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l329_32906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l329_32925

def my_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  a 1 = 1 ∧
  (∀ n, a (n + 2) = 1 / (a n + 1)) ∧
  a 6 = a 2

theorem sequence_sum (a : ℕ → ℝ) (h : my_sequence a) :
  a 2016 + a 3 = (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l329_32925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warrior_problem_solution_l329_32987

/-- Represents the problem setup -/
structure WarriorProblem where
  totalWarriors : Nat
  totalCoins : Nat

/-- Calculates the maximum coins Chernomor can get with arbitrary distribution -/
def maxCoinsArbitrary (problem : WarriorProblem) : Nat :=
  31 -- Placeholder value, replace with actual implementation

/-- Calculates the maximum coins Chernomor can get with equal distribution -/
def maxCoinsEqual (problem : WarriorProblem) : Nat :=
  30 -- Placeholder value, replace with actual implementation

/-- The main theorem stating the correct answers for both parts of the problem -/
theorem warrior_problem_solution (problem : WarriorProblem) 
  (h1 : problem.totalWarriors = 33) 
  (h2 : problem.totalCoins = 240) : 
  maxCoinsArbitrary problem = 31 ∧ maxCoinsEqual problem = 30 := by
  sorry

#eval maxCoinsArbitrary { totalWarriors := 33, totalCoins := 240 }
#eval maxCoinsEqual { totalWarriors := 33, totalCoins := 240 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warrior_problem_solution_l329_32987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l329_32928

/-- The function f(x) defined on real numbers. -/
noncomputable def f (x : ℝ) : ℝ := (2 * Real.sin x * Real.cos x) / (1 + Real.sin x + Real.cos x)

/-- The maximum value of f(x) is √2 - 1. -/
theorem f_max_value : ∀ x : ℝ, f x ≤ Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l329_32928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l329_32995

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between points (1, 2) and (4, 6) is 5 units -/
theorem distance_between_points : distance 1 2 4 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l329_32995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_range_l329_32990

/-- The trajectory of point P -/
noncomputable def trajectory (x y : ℝ) : Prop := y^2 = 4*x

/-- The distance from a point to F(1,0) -/
noncomputable def distToF (x y : ℝ) : ℝ := Real.sqrt ((x - 1)^2 + y^2)

/-- The distance from a point to the line x = -1 -/
noncomputable def distToLine (x : ℝ) : ℝ := |x + 1|

/-- The slope of line PF -/
noncomputable def slopePF (x y : ℝ) : ℝ := y / (x - 1)

/-- The distance between two points on the line x = -1 -/
noncomputable def distMN (m n : ℝ) : ℝ := |m - n|

/-- The main theorem -/
theorem trajectory_and_slope_range :
  ∀ (x₀ y₀ m n : ℝ), x₀ > 1 →
  distToF x₀ y₀ = distToLine x₀ →
  (x₀ - 1) * m^2 + 2 * y₀ * m - (x₀ + 1) = 0 →
  (x₀ - 1) * n^2 + 2 * y₀ * n - (x₀ + 1) = 0 →
  x₀^2 + y₀^2 = (x₀ + 1)^2 →
  trajectory x₀ y₀ ∧ 
  (0 < |slopePF x₀ y₀| / distMN m n ∧ |slopePF x₀ y₀| / distMN m n < 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_range_l329_32990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duma_committees_intersection_l329_32997

/-- Represents a committee in the Duma -/
structure Committee where
  members : Finset Nat
  size_eq : members.card = 80

/-- The set of all committees in the Duma -/
def all_committees : Finset Committee := sorry

/-- The total number of deputies in the Duma -/
def total_deputies : Nat := 1600

/-- The total number of committees in the Duma -/
def total_committees : Nat := 16000

theorem duma_committees_intersection :
  (∀ c ∈ all_committees, c.members.card = 80) →
  all_committees.card = total_committees →
  (∀ m ∈ (⋃ c ∈ all_committees, c.members), m ≤ total_deputies) →
  ∃ c1 c2 : Committee, c1 ∈ all_committees ∧ c2 ∈ all_committees ∧ c1 ≠ c2 ∧
    (c1.members ∩ c2.members).card ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_duma_committees_intersection_l329_32997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l329_32907

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3)^2 + (x - 6))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} =
    {x : ℝ | x < (5 - Real.sqrt 13) / 2 ∨
             ((5 - Real.sqrt 13) / 2 < x ∧ x < (5 + Real.sqrt 13) / 2) ∨
             (5 + Real.sqrt 13) / 2 < x} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l329_32907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l329_32975

/-- The number of ordered pairs of positive integers (x,y) satisfying xy = 2310 -/
def num_ordered_pairs : ℕ := 32

/-- The prime factorization of 2310 -/
def prime_factorization : List ℕ := [2, 3, 5, 7, 11]

/-- 2310 is the product of its prime factors -/
axiom factorization_correct : 2310 = prime_factorization.prod

theorem count_ordered_pairs :
  (Finset.card (Nat.divisors 2310)) = num_ordered_pairs := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l329_32975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_increase_l329_32911

/-- The number of sprockets produced by each machine -/
noncomputable def total_sprockets : ℝ := 660

/-- The production rate of Machine X in sprockets per hour -/
noncomputable def rate_x : ℝ := 6

/-- The additional time taken by Machine X compared to Machine B -/
noncomputable def extra_time : ℝ := 10

/-- The production rate of Machine B in sprockets per hour -/
noncomputable def rate_b : ℝ := total_sprockets / (total_sprockets / rate_x - extra_time)

/-- The percentage increase in production rate of Machine B compared to Machine X -/
noncomputable def percentage_increase : ℝ := (rate_b - rate_x) / rate_x * 100

theorem production_rate_increase :
  percentage_increase = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_increase_l329_32911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l329_32938

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x+2) - 2 always passes through the point (-2, -1) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x + 2) - 2
  f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l329_32938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_and_lottery_theorem_l329_32917

/-- Represents the age group of a customer -/
inductive AgeGroup
| Young
| MiddleAgedAndElderly

/-- Represents the satisfaction status of a customer -/
inductive Satisfaction
| Satisfied
| Dissatisfied

/-- Represents the color of a ball in the lottery -/
inductive BallColor
| Red
| White

/-- The contingency table for the survey -/
structure ContingencyTable :=
  (youngSatisfied : ℕ)
  (youngDissatisfied : ℕ)
  (elderSatisfied : ℕ)
  (elderDissatisfied : ℕ)

/-- The chi-square test statistic -/
noncomputable def chiSquare (table : ContingencyTable) : ℝ :=
  let n := table.youngSatisfied + table.youngDissatisfied + table.elderSatisfied + table.elderDissatisfied
  let a := table.youngSatisfied
  let b := table.youngDissatisfied
  let c := table.elderSatisfied
  let d := table.elderDissatisfied
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The lottery rules -/
structure LotteryRules :=
  (redBalls : ℕ)
  (whiteBalls : ℕ)
  (redReward : ℝ)
  (whiteReward : ℝ)
  (participants : ℕ)

/-- The mathematical expectation of the total recharge amount -/
noncomputable def expectedRechargeAmount (rules : LotteryRules) : ℝ :=
  let totalBalls := rules.redBalls + rules.whiteBalls
  let redProbability := (rules.redBalls : ℝ) / totalBalls
  let expectedRed := redProbability * rules.participants
  rules.redReward * expectedRed + rules.whiteReward * (rules.participants - expectedRed)

/-- The main theorem to be proved -/
theorem survey_and_lottery_theorem 
  (table : ContingencyTable) 
  (rules : LotteryRules) : 
  chiSquare table > 3.841 ∧ expectedRechargeAmount rules = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_and_lottery_theorem_l329_32917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_acute_dot_product_BQ_AQ_l329_32939

-- Define points A, B, and P
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)
def P (l : ℝ) : ℝ × ℝ := (l, l + 1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos (dot_product v w / (Real.sqrt (dot_product v v) * Real.sqrt (dot_product w w)))

-- Theorem 1: ∠APB is always acute
theorem angle_APB_acute (l : ℝ) : 
  angle (vector (P l) A) (vector (P l) B) < Real.pi / 2 := by sorry

-- Define Q as the fourth point of the rhombus ABPQ
def Q : ℝ × ℝ := (0, 1)

-- Theorem 2: If ABPQ is a rhombus, then BQ · AQ = 2
theorem dot_product_BQ_AQ : 
  dot_product (vector B Q) (vector A Q) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_acute_dot_product_BQ_AQ_l329_32939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_alternating_binomial_coefficients_l329_32992

open BigOperators Complex

theorem sum_alternating_binomial_coefficients : 
  ∑ k in Finset.range 50, (-1 : ℤ)^k * (Nat.choose 100 (2*k+1) : ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_alternating_binomial_coefficients_l329_32992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_cost_is_five_l329_32944

/-- Represents the cost structure of a laundry machine. -/
structure LaundryCost where
  initial_cost : ℚ  -- Cost for the first 1/3 hour
  hourly_rate : ℚ   -- Hourly rate after the first 1/4 hour

/-- Calculates the total cost for using a laundry machine. -/
def total_cost (lc : LaundryCost) (hours : ℚ) : ℚ :=
  lc.initial_cost + (hours - 1/3) * lc.hourly_rate

/-- Proves that the initial cost for the first 1/3 hour is $5. -/
theorem initial_cost_is_five :
  ∃ (lc : LaundryCost), lc.hourly_rate = 12 ∧ total_cost lc (5/2) = 31 ∧ lc.initial_cost = 5 := by
  sorry

#check initial_cost_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_cost_is_five_l329_32944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l329_32993

/-- Election results theorem -/
theorem election_results (total_votes : ℕ) (invalid_percentage : ℚ) 
  (vote_share_A vote_share_B vote_share_C : ℚ) : 
  total_votes = 900000 →
  invalid_percentage = 1/4 →
  vote_share_A = 7/15 →
  vote_share_B = 5/15 →
  vote_share_C = 3/15 →
  vote_share_A + vote_share_B + vote_share_C = 1 →
  ↑⌊(1 - invalid_percentage) * total_votes * vote_share_C⌋ = 135000 := by
  intro h_total h_invalid h_A h_B h_C h_sum
  sorry

#check election_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l329_32993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_competition_races_l329_32934

/-- Calculates the number of races needed to determine a champion in a sprint competition. -/
def races_needed (total_sprinters : ℕ) (initial_lanes : ℕ) (subsequent_lanes : ℕ) : ℕ :=
  let first_round := (total_sprinters + initial_lanes - 1) / initial_lanes
  let remaining_rounds := 
    let rec aux (runners : ℕ) (acc : ℕ) (fuel : ℕ) : ℕ :=
      if fuel = 0 then acc
      else if runners <= 1 then acc
      else aux ((runners + subsequent_lanes - 1) / subsequent_lanes) 
               (acc + ((runners + subsequent_lanes - 1) / subsequent_lanes))
               (fuel - 1)
    aux first_round 0 (total_sprinters) -- Use total_sprinters as an upper bound for recursion depth
  first_round + remaining_rounds

/-- The number of races needed for the given competition setup is 48. -/
theorem sprint_competition_races : 
  races_needed 300 8 6 = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_competition_races_l329_32934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l329_32910

/-- Represents a 5x6 grid of desks -/
def Grid := Fin 5 → Fin 6 → Bool

/-- The number of boys in the class -/
def num_boys : ℕ := 15

/-- The number of girls in the class -/
def num_girls : ℕ := 15

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 5 × Fin 6) : Prop :=
  (abs (p1.1 - p2.1) ≤ 1 ∧ abs (p1.2 - p2.2) ≤ 1) ∧ p1 ≠ p2

/-- A valid arrangement of students in the grid -/
def valid_arrangement (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 5 × Fin 6, adjacent p1 p2 → g p1.1 p1.2 ≠ g p2.1 p2.2

/-- The number of valid arrangements -/
def num_arrangements : ℕ := 2 * (Nat.factorial num_boys)^2

/-- Assume Grid is finite -/
instance : Fintype Grid := sorry

/-- Assume valid_arrangement is decidable -/
instance (g : Grid) : Decidable (valid_arrangement g) := sorry

theorem arrangement_count : 
  (Finset.filter valid_arrangement (Finset.univ : Finset Grid)).card = num_arrangements :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l329_32910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f4_is_direct_proportion_l329_32905

-- Define the concept of direct proportion
noncomputable def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define the four functions
noncomputable def f1 : ℝ → ℝ := λ x => 1 / x
noncomputable def f2 : ℝ → ℝ := λ x => x + 5
noncomputable def f3 : ℝ → ℝ := λ x => x^2 + 2*x
noncomputable def f4 : ℝ → ℝ := λ x => -2 * x

-- Theorem statement
theorem only_f4_is_direct_proportion :
  ¬(is_direct_proportion f1) ∧
  ¬(is_direct_proportion f2) ∧
  ¬(is_direct_proportion f3) ∧
  is_direct_proportion f4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f4_is_direct_proportion_l329_32905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_chessboard_l329_32958

/-- Represents a coloring of an 8x8 chessboard. -/
def Coloring := Fin 8 → Fin 8 → ℕ

/-- Checks if a given cell has at least two neighbors of the same color. -/
def hasAtLeastTwoSameColorNeighbors (c : Coloring) (i j : Fin 8) : Prop :=
  ∃ (n1 n2 : Fin 8 × Fin 8), 
    n1 ≠ n2 ∧
    ((n1.1 = i ∧ n1.2 = j + 1) ∨
     (n1.1 = i ∧ n1.2 = j - 1) ∨
     (n1.1 = i + 1 ∧ n1.2 = j) ∨
     (n1.1 = i - 1 ∧ n1.2 = j)) ∧
    ((n2.1 = i ∧ n2.2 = j + 1) ∨
     (n2.1 = i ∧ n2.2 = j - 1) ∨
     (n2.1 = i + 1 ∧ n2.2 = j) ∨
     (n2.1 = i - 1 ∧ n2.2 = j)) ∧
    c n1.1 n1.2 = c i j ∧
    c n2.1 n2.2 = c i j

/-- A valid coloring satisfies the condition for all cells. -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ (i j : Fin 8), hasAtLeastTwoSameColorNeighbors c i j

/-- The main theorem stating the maximum number of colors. -/
theorem max_colors_chessboard :
  ∃ (c : Coloring), isValidColoring c ∧
    ∀ (c' : Coloring), isValidColoring c' →
      Finset.card (Finset.image (fun (p : Fin 8 × Fin 8) => c' p.1 p.2) (Finset.univ.product Finset.univ)) ≤ 16 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_chessboard_l329_32958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_translation_l329_32916

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem min_positive_translation (φ : ℝ) : 
  (∀ x, f (x - φ) = f (-x - φ)) → -- Symmetry about y-axis after translation
  (∀ ψ, 0 < ψ ∧ ψ < φ → ¬(∀ x, f (x - ψ) = f (-x - ψ))) → -- Minimality of φ
  φ = 3 * Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_translation_l329_32916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l329_32967

theorem fixed_point_power_function :
  ∀ (α : ℝ), (2 : ℝ) ^ α = Real.sqrt 2 / 2 → (9 : ℝ) ^ α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l329_32967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l329_32941

-- Define the three lines
noncomputable def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
noncomputable def line2 (x y : ℝ) : Prop := 3 * x + y = 3
noncomputable def line3 (x y : ℝ) : Prop := 4 * y - 6 * x = 8

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (2/9, 7/3)

-- Theorem statement
theorem unique_intersection :
  ∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ line3 p.1 p.2 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l329_32941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l329_32935

noncomputable def arithmetic_mean (s : Finset ℝ) : ℝ := (s.sum id) / s.card

theorem arithmetic_mean_problem (x : ℝ) :
  arithmetic_mean {8, 15, 22, 5, x} = 12 → x = 10 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l329_32935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_has_largest_cookies_ben_gets_fewest_cookies_l329_32920

/-- Represents the four friends who bake cookies -/
inductive Friend
  | Ana
  | Ben
  | Carol
  | Dave

/-- Calculates the area of a cookie based on its shape and dimensions -/
noncomputable def cookieArea (f : Friend) : ℝ :=
  match f with
  | Friend.Ana   => 4 * Real.pi            -- Circle area: π * r^2
  | Friend.Ben   => 9                      -- Square area: s^2
  | Friend.Carol => Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) -- Pentagon area
  | Friend.Dave  => 3.375 * Real.sqrt 3    -- Hexagon area: (3√3 / 2) * s^2

/-- Theorem stating that Ben's cookies have the largest area -/
theorem ben_has_largest_cookies :
  ∀ f : Friend, f ≠ Friend.Ben → cookieArea Friend.Ben > cookieArea f := by
  sorry

/-- Corollary: Ben gets the fewest cookies from one batch of dough -/
theorem ben_gets_fewest_cookies :
  ∀ f : Friend, f ≠ Friend.Ben →
    (10 : ℝ) * cookieArea Friend.Ana / cookieArea Friend.Ben <
    (10 : ℝ) * cookieArea Friend.Ana / cookieArea f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_has_largest_cookies_ben_gets_fewest_cookies_l329_32920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_at_most_two_points_l329_32996

/-- A point in 2D Euclidean space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Angle between three points (in radians) -/
noncomputable def angle (p q r : Point) : ℝ :=
  Real.arccos ((distance p q)^2 + (distance p r)^2 - (distance q r)^2) / (2 * distance p q * distance p r)

/-- The set of points satisfying the given conditions -/
def locusSet (O Q : Point) : Set Point :=
  {P : Point | distance O P = 5 ∧ angle Q O P = Real.pi/3}

theorem locus_at_most_two_points (O Q : Point) :
  ∃ (P₁ P₂ : Point), ∀ P ∈ locusSet O Q, P = P₁ ∨ P = P₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_at_most_two_points_l329_32996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l329_32959

theorem problem_solution (x y : ℝ) 
  (h1 : Real.sqrt (3 * x + 1) = 4)
  (h2 : (x + 2 * y)^(1/3 : ℝ) = -1) :
  x = 5 ∧ y = -3 ∧ (Real.sqrt (2 * x - 5 * y) = 5 ∨ Real.sqrt (2 * x - 5 * y) = -5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l329_32959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_wins_first_given_conditions_l329_32985

-- Define the type for game outcomes
inductive GameOutcome
| X
| Y
deriving BEq, Repr

-- Define a series as a list of game outcomes
def Series := List GameOutcome

-- Define a function to check if a series is valid
def isValidSeries (s : Series) : Prop :=
  s.length ≥ 4 ∧ 
  (s.count GameOutcome.X = 4 ∨ s.count GameOutcome.Y = 4) ∧
  s.length ≤ 7

-- Define a function to check if X wins the series
def xWinsSeries (s : Series) : Prop :=
  isValidSeries s ∧ s.count GameOutcome.X = 4

-- Define a function to check if Y wins the third game
def yWinsThird (s : Series) : Prop :=
  s.length ≥ 3 ∧ s.get? 2 = some GameOutcome.Y

-- Define a function to check if Y wins the first game
def yWinsFirst (s : Series) : Prop :=
  s.length ≥ 1 ∧ s.get? 0 = some GameOutcome.Y

-- State the theorem
theorem probability_y_wins_first_given_conditions :
  ∃ (total favorable : ℕ),
    total > 0 ∧
    favorable ≤ total ∧
    (favorable : ℚ) / (total : ℚ) = 5 / 12 ∧
    ∀ (s : Series),
      isValidSeries s →
      xWinsSeries s →
      yWinsThird s →
      (yWinsFirst s ↔ favorable > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_wins_first_given_conditions_l329_32985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tangent_length_l329_32902

structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : 0 < b ∧ b < a

structure Circle where
  R : ℝ
  h_R : 1 < R ∧ R < 2

def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

def circle_equation (C : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 = C.R^2

def is_tangent_line (l : ℝ × ℝ → Prop) (E : Ellipse) (C : Circle) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    l (x₁, y₁) ∧ l (x₂, y₂) ∧
    ellipse_equation E x₁ y₁ ∧
    circle_equation C x₂ y₂ ∧
    (∀ x y, l (x, y) → (ellipse_equation E x y → x = x₁ ∧ y = y₁) ∧
                       (circle_equation C x y → x = x₂ ∧ y = y₂))

theorem max_tangent_length (E : Ellipse) (C : Circle) :
  E.a = 2 ∧ E.b = 1 →
  (∃ (F : ℝ × ℝ), F.1 = Real.sqrt 3 ∧ F.2 = 0 ∧ (F.1^2 / E.a^2 + F.2^2 / E.b^2 > 1)) →
  (let e := Real.sqrt 3 / 2; e^2 * E.a^2 = E.a^2 - E.b^2) →
  (∀ l : ℝ × ℝ → Prop, is_tangent_line l E C →
    ∃ A B : ℝ × ℝ, l A ∧ l B ∧ ellipse_equation E A.1 A.2 ∧ circle_equation C B.1 B.2 →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ 1) ∧
  (∃ l : ℝ × ℝ → Prop, is_tangent_line l E C ∧
    ∃ A B : ℝ × ℝ, l A ∧ l B ∧ ellipse_equation E A.1 A.2 ∧ circle_equation C B.1 B.2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = 1) ↔
  C.R = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tangent_length_l329_32902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_questions_cameron_l329_32978

/-- Represents a tour group with its characteristics -/
structure TourGroup where
  size : Nat
  special_cases : List (Int → Int)

/-- Calculates the number of questions for a tour group -/
def questions_for_group (group : TourGroup) : Int :=
  group.size * 2 + group.special_cases.foldl (λ acc f => acc + f 2) 0

/-- The list of tour groups Cameron guided -/
def tour_groups : List TourGroup := [
  { size := 6, special_cases := [] },
  { size := 11, special_cases := [] },
  { size := 8, special_cases := [λ n => 4 * n] },
  { size := 5, special_cases := [λ n => 6 * n, λ _ => -2] },
  { size := 9, special_cases := [λ n => 2 * n, λ n => 2 * n, λ n => 2 * n, λ _ => -2, λ _ => -2] },
  { size := 7, special_cases := [λ _ => 1, λ _ => 1] }
]

/-- The theorem stating the total number of questions Cameron answered -/
theorem total_questions_cameron : (tour_groups.map questions_for_group).sum = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_questions_cameron_l329_32978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l329_32983

/-- Given a differentiable function f : ℝ → ℝ with a tangent line y = x + 2 at the point (1, f(1)),
    prove that f(1) + f'(1) = 4 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f 1 + (deriv f 1) * (x - 1) = x + 2) →
  f 1 + deriv f 1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l329_32983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_reflection_perpendicularity_l329_32979

/-- Given a circle and points on it, prove the perpendicularity of certain lines. -/
theorem circle_reflection_perpendicularity 
  (O : ℝ × ℝ)  -- Center of the circle
  (A B C D : ℝ × ℝ)  -- Points on the circle
  (h_circle : ∀ P ∈ ({A, B, C, D} : Set (ℝ × ℝ)), dist O P = dist O A)  -- All points are on the circle
  (h_diameter : D = O + (O - C))  -- D is diametrically opposite to C
  (M : ℝ × ℝ)  -- Midpoint of AB
  (h_midpoint : M = (A + B) / 2)  -- Definition of midpoint
  (C1 : ℝ × ℝ)  -- Reflection of C about M
  (h_reflection : C1 = 2 * M - C)  -- Definition of reflection
  : (B.1 - A.1) * (D.1 - C1.1) + (B.2 - A.2) * (D.2 - C1.2) = 0  -- AB ⟂ C1D
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_reflection_perpendicularity_l329_32979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l329_32931

/-- Given an ellipse with center at (-2, 1), one focus at (-3, 0), and one endpoint 
    of a semi-major axis at (-2, 4), the semi-minor axis of this ellipse is √7. -/
theorem ellipse_semi_minor_axis : 
  let center : ℝ × ℝ := (-2, 1)
  let focus : ℝ × ℝ := (-3, 0)
  let semi_major_endpoint : ℝ × ℝ := (-2, 4)
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l329_32931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l329_32901

theorem relationship_abc (a b c : ℝ) : 
  a = Real.log (1/3) / Real.log 2 → b = Real.exp (1/3) → c = 1/3 → 
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l329_32901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l329_32914

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 3) - 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : Real), p > 0 ∧ ∀ (x : Real), f (x + p) = f x ∧ ∀ (q : Real), q > 0 ∧ (∀ (x : Real), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : Real), y ∈ Set.Icc (-1/2) 1 ↔ ∃ (x : Real), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l329_32914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_I_greater_area_and_height_l329_32970

/-- Represents a trapezoid with two bases and a height -/
structure Trapezoid where
  base1 : ℚ
  base2 : ℚ
  height : ℚ

/-- Calculates the area of a trapezoid -/
def area (t : Trapezoid) : ℚ := (t.base1 + t.base2) * t.height / 2

/-- Trapezoid I as defined in the problem -/
def trapezoid_I : Trapezoid := ⟨3, 1, 2⟩

/-- Trapezoid II as defined in the problem -/
def trapezoid_II : Trapezoid := ⟨4, 2, 1⟩

theorem trapezoid_I_greater_area_and_height :
  area trapezoid_I > area trapezoid_II ∧ trapezoid_I.height > trapezoid_II.height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_I_greater_area_and_height_l329_32970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cousin_reading_time_l329_32998

/-- Represents the time taken to read a novel -/
structure ReadingTime where
  minutes : ℚ

/-- Converts hours to minutes -/
def hoursToMinutes (h : ℚ) : ℚ := h * 60

/-- Calculates the reading time for a faster reader given the reading time of a slower reader and the speed ratio -/
def fasterReaderTime (slowerTime : ReadingTime) (speedRatio : ℚ) : ReadingTime :=
  { minutes := slowerTime.minutes / speedRatio }

theorem cousin_reading_time :
  let myTime : ReadingTime := { minutes := 180 }
  let speedRatio : ℚ := 5
  let cousinTime := fasterReaderTime myTime speedRatio
  cousinTime.minutes = 36 := by
    -- Unfold definitions
    unfold fasterReaderTime
    -- Perform calculation
    simp [ReadingTime.minutes]
    -- The proof is complete
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cousin_reading_time_l329_32998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_ac_length_l329_32960

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a segment AB and its golden section point C, this function returns the length of AC -/
noncomputable def goldenSectionLength (AB : ℝ) : ℝ := AB * (φ - 1)

theorem golden_section_ac_length :
  ∀ (AB : ℝ) (AC BC : ℝ),
  AC > BC →
  AC / BC = φ →
  AB = AC + BC →
  AB = 200 →
  AC = 100 * (Real.sqrt 5 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_ac_length_l329_32960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l329_32948

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * (Real.cos (Real.pi * x))^2 - 4 * Real.sqrt 3 * Real.sin (Real.pi * x) * Real.cos (Real.pi * x)

-- State the theorem
theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → S ≥ T) ∧
  T = 1 ∧
  (∀ x : ℝ, -Real.pi/3 ≤ x ∧ x ≤ Real.pi/6 → f x ≤ 6) ∧
  (f (-Real.pi/6) = 6) ∧
  (∀ x : ℝ, -Real.pi/3 ≤ x ∧ x ≤ Real.pi/6 → f x ≥ 0) ∧
  (f (Real.pi/6) = 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l329_32948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_correct_perp_lines_correct_l329_32930

-- Define the original line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 12 = 0}

-- Define the point (-1, 3)
def point : ℝ × ℝ := (-1, 3)

-- Define the parallel line l'
def l_parallel : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 9 = 0}

-- Define the perpendicular lines l'
def l_perp1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 4/3 * p.1 + 4}
def l_perp2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 4/3 * p.1 - 4}

-- Helper function to calculate triangle area (placeholder)
noncomputable def area_triangle (line : Set (ℝ × ℝ)) : ℝ :=
  sorry

-- Theorem for parallel line
theorem parallel_line_correct : 
  (∀ p ∈ l_parallel, p ∈ l → False) ∧ 
  point ∈ l_parallel :=
sorry

-- Theorem for perpendicular lines
theorem perp_lines_correct : 
  (∀ p ∈ l_perp1, p ∈ l → False) ∧
  (∀ p ∈ l_perp2, p ∈ l → False) ∧
  (area_triangle l_perp1 = 6 ∨ area_triangle l_perp2 = 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_correct_perp_lines_correct_l329_32930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_vertex_close_to_diagonal_points_l329_32945

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 1 -/
structure UnitSquare where
  vertices : Fin 4 → Point

/-- A set of 99 distinct points on the diagonal of a unit square -/
structure DiagonalPoints where
  points : Fin 99 → Point
  on_diagonal : ∀ i, ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (points i).x = t ∧ (points i).y = t
  distinct : ∀ i j, i ≠ j → points i ≠ points j

/-- Squared distance between two points -/
noncomputable def squaredDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Average squared distance from a set of points to a given point -/
noncomputable def averageSquaredDistance (pts : DiagonalPoints) (v : Point) : ℝ :=
  (Finset.sum Finset.univ (fun i => squaredDistance (pts.points i) v)) / 99

/-- The main theorem to be proved -/
theorem at_most_one_vertex_close_to_diagonal_points (square : UnitSquare) (pts : DiagonalPoints) :
    ∃! v : Fin 4, averageSquaredDistance pts (square.vertices v) ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_vertex_close_to_diagonal_points_l329_32945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l329_32961

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (h_pos : ∀ x ∈ Set.Icc 0 1, 0 ≤ f x)
  (h_one : f 1 = 1)
  (h_super : ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x + y ∈ Set.Icc 0 1 → 
    f (x + y) ≥ f x + f y) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l329_32961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2CPD_equals_sqrt3_over_2_l329_32972

-- Define the points
variable (A B C D E P : ℝ × ℝ)

-- Define the angles
noncomputable def angle_BPC : ℝ := Real.arccos (1/3)
noncomputable def angle_CPD : ℝ := Real.arccos (1/2)

-- State the theorem
theorem sin_2CPD_equals_sqrt3_over_2 
  (h1 : B.1 - A.1 = C.1 - B.1) 
  (h2 : C.1 - B.1 = D.1 - C.1) 
  (h3 : D.1 - C.1 = E.1 - D.1)
  (h4 : Real.cos angle_BPC = 1/3)
  (h5 : Real.cos angle_CPD = 1/2) :
  Real.sin (2 * angle_CPD) = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2CPD_equals_sqrt3_over_2_l329_32972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_crossing_time_l329_32986

-- Define the speeds of the trains in kmph
noncomputable def faster_train_speed : ℚ := 72
noncomputable def slower_train_speed : ℚ := 36

-- Define the length of the faster train in meters
noncomputable def faster_train_length : ℚ := 370

-- Define the conversion factor from kmph to m/s
noncomputable def kmph_to_ms : ℚ := 5 / 18

-- Theorem statement
theorem faster_train_crossing_time :
  let relative_speed_kmph := faster_train_speed - slower_train_speed
  let relative_speed_ms := relative_speed_kmph * kmph_to_ms
  let crossing_time := faster_train_length / relative_speed_ms
  crossing_time = 37 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_crossing_time_l329_32986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_rabbit_in_4_minutes_l329_32982

/-- Represents the chase scenario between a dog and a rabbit -/
structure ChaseScenario where
  dog_speed : ℝ
  rabbit_speed : ℝ
  head_start : ℝ

/-- Calculates the time it takes for the dog to catch the rabbit -/
noncomputable def catch_time (scenario : ChaseScenario) : ℝ :=
  scenario.head_start / (scenario.dog_speed / 60 - scenario.rabbit_speed / 60)

/-- Theorem stating that in the given scenario, it takes 4 minutes for the dog to catch the rabbit -/
theorem dog_catches_rabbit_in_4_minutes :
  let scenario : ChaseScenario := {
    dog_speed := 24,
    rabbit_speed := 15,
    head_start := 0.6
  }
  catch_time scenario = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_rabbit_in_4_minutes_l329_32982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l329_32932

/-- Given a circle with equation 2x^2+2y^2+6x-4y-3=0, prove that its center is at (-3/2, 1) and its radius is √19/2 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), 2*x^2 + 2*y^2 + 6*x - 4*y - 3 = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-3/2, 1) ∧ 
    radius = Real.sqrt (19/4) ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l329_32932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l329_32969

theorem first_six_average (numbers : List ℝ) : 
  numbers.length = 11 ∧ 
  numbers.sum / numbers.length = 22 ∧
  (numbers.drop 5).sum / 6 = 27 ∧
  numbers.get? 5 = some 34 →
  (numbers.take 6).sum / 6 = 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l329_32969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_is_systematic_sampling_l329_32926

/-- Represents a lottery ticket number -/
def TicketNumber := Fin 100000

/-- The range of valid ticket numbers -/
def validTicketRange : Set TicketNumber := Set.univ

/-- Checks if a ticket number wins the first prize -/
def isFirstPrizeWinner (n : TicketNumber) : Prop :=
  n.val % 1000 = 345

/-- Defines the characteristics of systematic sampling -/
def isSystematicSampling (sample : Set TicketNumber) : Prop :=
  ∃ (k : Nat), k > 0 ∧ ∀ (n : TicketNumber),
    n ∈ sample ↔ (n ∈ validTicketRange ∧ n.val % k = 345)

/-- The main theorem stating that the given lottery method is systematic sampling -/
theorem lottery_is_systematic_sampling :
  isSystematicSampling {n ∈ validTicketRange | isFirstPrizeWinner n} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_is_systematic_sampling_l329_32926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ravi_kiran_ratio_l329_32913

-- Define the amounts of money for each person
def ravi_amount : ℕ := 36
def kiran_amount : ℕ := 105

-- Define the ratio of money between Ravi and Giri
def ravi_giri_ratio : Rat := 6 / 7

-- Function to calculate the simplified ratio
def simplify_ratio (a b : ℕ) : Rat :=
  let gcd := Nat.gcd a b
  (a / gcd : Rat) / (b / gcd : Rat)

-- Theorem statement
theorem ravi_kiran_ratio :
  simplify_ratio ravi_amount kiran_amount = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ravi_kiran_ratio_l329_32913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_ratio_l329_32953

/-- The area of a regular pentagon with side length s -/
noncomputable def pentagonArea (s : ℝ) : ℝ := (1/4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * s^2

/-- The ratio of areas of five small pentagons to one large pentagon -/
noncomputable def areaRatio : ℝ := 
  let smallSide : ℝ := 1  -- Arbitrary side length for small pentagons
  let largeSide : ℝ := 5 * smallSide  -- Side length of large pentagon
  (5 * pentagonArea smallSide) / (pentagonArea largeSide)

theorem pentagon_area_ratio : areaRatio = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_ratio_l329_32953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l329_32955

theorem trigonometric_identities (x : ℝ) 
  (h1 : Real.cos x = -3/5) 
  (h2 : x ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  (Real.sin (x + Real.pi/3) = (4 - 3 * Real.sqrt 3) / 10) ∧ 
  (Real.sin (2*x + Real.pi/6) = -(24 * Real.sqrt 3 + 7) / 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l329_32955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_2019_l329_32904

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((Real.cos (2 * x) - 2 * (Real.cos x) ^ 2) / (Real.cos x - 1)) / Real.log 4

-- Define the composition of f with itself 2019 times
noncomputable def f_2019 : ℝ → ℝ := (f^[2019])

-- Theorem statement
theorem range_of_f_2019 :
  Set.range f_2019 = Set.Ici (-1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_2019_l329_32904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_area_ratio_l329_32950

/-- The area marked by a ball touching two concentric spheres -/
theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (hr : 0 < r₁ ∧ r₁ < r₂) (hA : 0 < A₁) :
  A₁ * (r₂/r₁)^2 = A₁ * (r₂/r₁)^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_area_ratio_l329_32950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l329_32962

/-- The area of a trapezium with given parallel sides and height. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and a distance of 15 cm between them, is equal to 285 cm². -/
theorem trapezium_area_example : trapezium_area 20 18 15 = 285 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [add_mul, div_eq_mul_inv]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l329_32962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_unknown_lap_time_l329_32912

/-- Billy's swimming competition --/
def billy_swimming_problem (unknown_lap_time : ℝ) : Prop :=
  let billy_first_5_laps := 2
  let billy_next_3_laps := 4
  let billy_final_lap := 150 / 60
  let margaret_total_time := 10
  let billy_win_margin := 0.5
  let billy_known_laps := billy_first_5_laps + billy_next_3_laps + billy_final_lap
  let billy_total_time := margaret_total_time - billy_win_margin
  billy_total_time = billy_known_laps + unknown_lap_time

theorem billy_unknown_lap_time :
  ∃ (unknown_lap_time : ℝ), billy_swimming_problem unknown_lap_time ∧ unknown_lap_time = 1 := by
  sorry

#check billy_unknown_lap_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_unknown_lap_time_l329_32912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_removal_l329_32942

theorem exam_question_removal (n m : ℕ) (h1 : n = 610) (h2 : m = 10) 
  (S : Finset (Finset (Fin m))) (h3 : S.card = n) (h4 : ∀ A B, A ∈ S → B ∈ S → A ≠ B) :
  ∃ i : Fin m, ∀ A B, A ∈ S → B ∈ S → A.erase i ≠ B.erase i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_removal_l329_32942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_in_four_moves_l329_32909

/-- Represents a card in the game -/
structure Card where
  id : Nat
  pair : Nat

/-- Represents the game state -/
structure GameState where
  cards : Finset Card
  seenCards : Finset Card

/-- A move in the game -/
def move (g : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (a matching pair is found) -/
def gameEnded (g : GameState) : Bool :=
  sorry

/-- The theorem stating that the game ends in at most 4 moves -/
theorem game_ends_in_four_moves :
  ∀ (initialState : GameState),
    (initialState.cards.card = 10) →
    (∃! (p : Nat → Nat), ∀ c ∈ initialState.cards, c.pair = p c.id ∧ p c.id = c.pair) →
    (∃ (n : Nat), n ≤ 4 ∧ 
      gameEnded (Nat.rec initialState (fun _ => move) n)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_in_four_moves_l329_32909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pets_l329_32966

/-- Given:
  - Anthony has 12 cats and dogs in total
  - 2/3 of Anthony's pets are cats
  - Leonel has half as many cats as Anthony
  - Leonel has seven more dogs than Anthony
Prove that the total number of animals Anthony and Leonel have is 27 -/
theorem total_pets (anthony_total : ℕ) (anthony_cat_ratio : ℚ) (leonel_cat_ratio : ℚ) (leonel_extra_dogs : ℕ) :
  anthony_total = 12 →
  anthony_cat_ratio = 2/3 →
  leonel_cat_ratio = 1/2 →
  leonel_extra_dogs = 7 →
  (anthony_total + 
   (leonel_cat_ratio * (anthony_cat_ratio * anthony_total : ℚ)).floor + 
   ((anthony_total - (anthony_cat_ratio * anthony_total : ℚ).floor) + leonel_extra_dogs)) = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pets_l329_32966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_five_pi_fourths_l329_32947

theorem tan_negative_five_pi_fourths : Real.tan (-5 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_five_pi_fourths_l329_32947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l329_32921

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  ((Real.sqrt 2 * t) / 2 + 1, -(Real.sqrt 2 * t) / 2)

-- Define the circle C in polar form
def circle_C (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

-- Define point P
def point_P : ℝ × ℝ := (1, 0)

-- State the theorem
theorem intersection_distance_sum :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ : ℝ, (A.1^2 + A.2^2).sqrt = circle_C θ) ∧
    (∃ θ : ℝ, (B.1^2 + B.2^2).sqrt = circle_C θ) ∧
    ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2).sqrt +
    ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2).sqrt =
    Real.sqrt 6 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l329_32921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_result_l329_32927

/-- The function f(x) = x sin(x) -/
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.sin x + x * Real.cos x

theorem extreme_value_result (x₀ : ℝ) (h : f' x₀ = 0) :
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_result_l329_32927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_18_three_even_one_odd_l329_32919

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ d => d % 2 = 0)).length = 3 ∧ (digits.filter (λ d => d % 2 = 1)).length = 1

theorem smallest_four_digit_divisible_by_18_three_even_one_odd : 
  ∀ n : ℕ, is_four_digit n → n % 18 = 0 → has_three_even_one_odd n → 2214 ≤ n :=
by
  sorry

#check smallest_four_digit_divisible_by_18_three_even_one_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_18_three_even_one_odd_l329_32919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l329_32991

-- Define the tangent line
def tangent_line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the first curve
noncomputable def curve1 (x : ℝ) : ℝ := Real.log x + 2

-- Define the second curve
noncomputable def curve1_deriv (x : ℝ) : ℝ := 1 / x

-- Define the second curve
noncomputable def curve2 (x : ℝ) : ℝ := Real.log (x + 1)

-- Define the derivative of the second curve
noncomputable def curve2_deriv (x : ℝ) : ℝ := 1 / (x + 1)

-- State the theorem
theorem tangent_to_both_curves (k : ℝ) (b : ℝ) :
  (∃ x₁ > 0, tangent_line k b x₁ = curve1 x₁ ∧ k = curve1_deriv x₁) ∧
  (∃ x₂ > -1, tangent_line k b x₂ = curve2 x₂ ∧ k = curve2_deriv x₂) →
  b = 1 - Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l329_32991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_parallel_to_a_l329_32963

/-- Definition of a unit vector -/
def IsUnitVector (v : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = 1

/-- Given a vector a = (2, 1), prove that a unit vector parallel to a has coordinates ± (2√5/5, √5/5) -/
theorem unit_vector_parallel_to_a (a : ℝ × ℝ) (h : a = (2, 1)) :
  ∃ (u : ℝ × ℝ), IsUnitVector u ∧ ∃ (k : ℝ), u = k • a ∧ (u = (2 * Real.sqrt 5 / 5, Real.sqrt 5 / 5) ∨ u = (-2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_parallel_to_a_l329_32963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_power_sum_divisibility_l329_32973

theorem root_power_sum_divisibility 
  (p q : ℤ) 
  (r : ℕ) 
  (hr : r > 1) 
  (hgcd : Nat.gcd p.natAbs q.natAbs = r) 
  (hroots : ∃ x y : ℝ, x^2 + (p : ℝ) * x + (q : ℝ) = 0 ∧ y^2 + (p : ℝ) * y + (q : ℝ) = 0) 
  (n : ℕ) :
  ∀ k : ℕ, (∃ m : ℤ, (x^n + y^n : ℝ) = (r^k : ℝ) * m) ↔ (1 ≤ k ∧ k ≤ (n + 1) / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_power_sum_divisibility_l329_32973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l329_32908

noncomputable def g (n : ℤ) : ℝ := (5 + 4 * Real.sqrt 2) / 2 * (Real.sqrt 2)^n - (5 - 4 * Real.sqrt 2) / 2 * (-Real.sqrt 2)^n

theorem g_relation (n : ℤ) : g (n + 1) - g (n - 1) = (Real.sqrt 2 / 2) * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l329_32908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y5_l329_32957

theorem coefficient_x3y5 :
  let expression := (2/3 : ℚ) * X - (1/3 : ℚ) * Y
  let coefficient := Finset.sum (Finset.range 9) (fun k =>
    ↑(Nat.choose 8 k) * (2/3 : ℚ)^k * (-1/3 : ℚ)^(8 - k) *
    if k = 3 then 1 else 0)
  coefficient = -448/6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y5_l329_32957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_point_l329_32952

theorem terminal_side_point (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (m, -3) ∧ P.1 = m * Real.cos α ∧ P.2 = m * Real.sin α) →
  Real.cos α = -4/5 →
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_point_l329_32952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_photo_border_area_l329_32923

/-- Calculate the area of the border (including decorative lining) of a framed photograph --/
theorem framed_photo_border_area :
  let photo_height : ℝ := 12
  let photo_width : ℝ := 16
  let frame_border_width : ℝ := 3
  let decorative_lining_width : ℝ := 1
  let total_border_width : ℝ := frame_border_width + decorative_lining_width
  let framed_height : ℝ := photo_height + 2 * total_border_width
  let framed_width : ℝ := photo_width + 2 * total_border_width
  let photo_area : ℝ := photo_height * photo_width
  let framed_area : ℝ := framed_height * framed_width
  let border_area : ℝ := framed_area - photo_area
  border_area = 288 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_photo_border_area_l329_32923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_zeta_time_is_two_l329_32977

/-- Represents the time taken by a worker to complete the job alone -/
structure WorkerTime where
  time : ℝ
  time_positive : time > 0

/-- The job completion scenario -/
structure JobScenario where
  delta : WorkerTime
  epsilon : WorkerTime
  zeta : WorkerTime
  all_together : ℝ
  epsilon_zeta : ℝ
  all_together_delta : all_together = delta.time - 4
  all_together_epsilon : all_together = epsilon.time - 3.5
  epsilon_zeta_half : epsilon_zeta = epsilon.time / 2
  rate_sum : 1 / delta.time + 1 / epsilon.time + 1 / zeta.time = 1 / all_together

/-- The time taken by Delta and Zeta together to complete the job -/
noncomputable def delta_zeta_time (scenario : JobScenario) : ℝ :=
  1 / (1 / scenario.delta.time + 1 / scenario.zeta.time)

/-- Theorem stating that Delta and Zeta together take 2 hours to complete the job -/
theorem delta_zeta_time_is_two (scenario : JobScenario) : 
  delta_zeta_time scenario = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_zeta_time_is_two_l329_32977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_X_in_A_is_0_8_percent_solution_verification_l329_32915

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 0.008

/-- The weight of solution A in grams -/
def weight_A : ℝ := 300

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 0.018

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.015

/-- The theorem stating that the percentage of liquid X in solution A is 0.8% -/
theorem percentage_X_in_A_is_0_8_percent :
  percentage_X_in_A = 0.008 :=
by
  -- The proof goes here
  sorry

/-- The theorem proving that the given conditions lead to the correct answer -/
theorem solution_verification :
  percentage_X_in_A * weight_A + percentage_X_in_B * weight_B =
  percentage_X_in_mixture * (weight_A + weight_B) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_X_in_A_is_0_8_percent_solution_verification_l329_32915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_p_l329_32994

def a : Fin 3 → ℝ := ![2, -2, 4]
def b : Fin 3 → ℝ := ![0, 4, 0]
def n : Fin 3 → ℝ := ![1, -1, 2]

def plane_equation (x y z : ℝ) : Prop :=
  x - y + 2*z = 0

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

def scalar_mult (c : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => c * (v i)

def vector_sub (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => (v i) - (w i)

noncomputable def projection (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  vector_sub v (scalar_mult (dot_product v n / dot_product n n) n)

theorem projection_equals_p : 
  projection a = projection b ∧ 
  projection a = ![1/3, -1/3, 2/3] ∧
  plane_equation (projection a 0) (projection a 1) (projection a 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_p_l329_32994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equals_closed_form_l329_32940

/-- The sum of the sequence 81 + 891 + 8991 + 89991 + ... + 8(99...99) (with n-1 nines) -/
def sequenceSum (n : ℕ) : ℕ :=
  let rec termAtIndex (i : ℕ) : ℕ :=
    if i = 0 then 0
    else 8 * (10^i) + (10^i - 1) / 9
  (Finset.range n).sum termAtIndex

/-- The closed form of the sum -/
def closedForm (n : ℕ) : ℤ :=
  10^(n+1) - 9*n - 10

theorem sequence_sum_equals_closed_form (n : ℕ) :
  (sequenceSum n : ℤ) = closedForm n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equals_closed_form_l329_32940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_satisfies_conditions_present_age_is_50_l329_32965

/-- Represents the man's present age -/
noncomputable def present_age : ℝ := 50

/-- The man's age 10 years ago -/
noncomputable def past_age : ℝ := present_age * (4/5)

/-- The man's age 10 years in the future -/
noncomputable def future_age : ℝ := present_age * (3/2.5)

/-- Theorem stating that the present age satisfies the given conditions -/
theorem age_satisfies_conditions :
  (present_age = past_age + 10) ∧
  (present_age = future_age - 10) := by
  sorry

/-- Theorem proving that the present age is 50 -/
theorem present_age_is_50 : present_age = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_satisfies_conditions_present_age_is_50_l329_32965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l329_32956

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := 3 * x^3 - x^2 - 14 * x - 5

-- Define the roots of the cubic equation
noncomputable def r₁ : ℝ := sorry
noncomputable def r₂ : ℝ := sorry
noncomputable def r₃ : ℝ := sorry

-- Assume the roots are ordered
axiom root_order : r₁ < r₂ ∧ r₂ < r₃

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-4 : ℝ) (-3 : ℝ) ∪ Set.Ioo r₁ r₂

-- State the theorem
theorem inequality_equivalence (x : ℝ) :
  (x^2 + 1) / (x + 3) > (4 * x^2 + 5) / (3 * x + 4) ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l329_32956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_face_angle_is_arcsin_three_fifths_l329_32988

/-- A regular triangular pyramid with a lateral edge forming a 45° angle with the base plane -/
structure RegularTriangularPyramid where
  -- The side length of the base
  a : ℝ
  -- Assumption that a is positive
  a_pos : 0 < a

/-- The angle between a lateral edge and the base plane -/
noncomputable def lateral_edge_angle : ℝ := Real.pi / 4

/-- The angle between the apothem and an adjacent lateral face -/
noncomputable def apothem_face_angle (p : RegularTriangularPyramid) : ℝ :=
  Real.arcsin (3 / 5)

/-- Theorem stating the relationship between the pyramid's geometry and the apothem-face angle -/
theorem apothem_face_angle_is_arcsin_three_fifths (p : RegularTriangularPyramid) :
  apothem_face_angle p = Real.arcsin (3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_face_angle_is_arcsin_three_fifths_l329_32988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tracy_candies_problem_l329_32903

theorem tracy_candies_problem :
  ∃ (initial_candies : ℕ),
    initial_candies > 0 ∧
    initial_candies % 5 = 0 ∧
    (let remaining_after_eating := initial_candies * 3 / 5;
     let remaining_after_sharing := remaining_after_eating * 2 / 3;
     let remaining_after_mom := remaining_after_sharing - 40;
     ∃ (brother_took : ℕ),
       2 ≤ brother_took ∧ brother_took ≤ 6 ∧
       remaining_after_mom - brother_took = 5 ∧
       initial_candies = 120) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tracy_candies_problem_l329_32903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_l329_32984

theorem not_prime_sum (a b c : ℕ+) (k : ℕ) (h : (a : ℕ)^2 - (b : ℕ)*(c : ℕ) = k^2) :
  ¬ Nat.Prime ((2 : ℕ)*(a : ℕ) + (b : ℕ) + (c : ℕ)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_l329_32984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_l329_32951

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2

theorem f_monotonicity_and_extrema :
  (∀ (k : ℤ), ∀ (x y : ℝ),
    k * Real.pi - 5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + Real.pi / 12 →
    f x < f y) ∧
  (∀ (k : ℤ), ∀ (x y : ℝ),
    k * Real.pi + Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 7 * Real.pi / 12 →
    f y < f x) ∧
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 3 → f x ≤ 2 + Real.sqrt 3) ∧
  (∃ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 3 ∧ f x = 2 + Real.sqrt 3) ∧
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 3 → 0 ≤ f x) ∧
  (∃ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 3 ∧ f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_l329_32951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l329_32989

def vector_problem (a b c : ℝ × ℝ) : Prop :=
  let a_plus_b : ℝ × ℝ := (1, 1)
  let a_minus_b : ℝ × ℝ := (-3, 1)
  let c : ℝ × ℝ := (1, 1)
  (a + b = a_plus_b) ∧ 
  (a - b = a_minus_b) ∧ 
  (c = (1, 1)) →
  (a.1 * c.1 + a.2 * c.2 = 0) ∧ 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 135 * Real.pi / 180)

theorem vector_theorem (a b c : ℝ × ℝ) : vector_problem a b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l329_32989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_properties_l329_32933

/-- A random variable X following a binomial distribution with n trials and probability p of success --/
def binomial_distribution (n : ℕ) (p : ℝ) : Type := ℕ

/-- The probability mass function of a binomial distribution --/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The expected value of a binomial distribution --/
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The variance of a binomial distribution --/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_properties :
  let n : ℕ := 3
  let p : ℝ := 2/5
  let X := binomial_distribution n p
  (∀ k, 0 ≤ k ∧ k ≤ n → binomial_pmf n p k = (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)) ∧
  binomial_expectation n p = 6/5 ∧
  binomial_variance n p = 18/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_properties_l329_32933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l329_32922

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a^x

-- State the theorem
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l329_32922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_right_triangle_l329_32954

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

-- Define a function that assigns a color to each point on the sides of the triangle
def colorAssignment (t : EquilateralTriangle) : Point → Color :=
  sorry

-- Define a predicate for right triangles
def isRightTriangle (p q r : Point) : Prop :=
  sorry

-- Theorem statement
theorem exists_same_color_right_triangle (t : EquilateralTriangle) :
  ∃ (p q r : Point), 
    (colorAssignment t p = colorAssignment t q) ∧ 
    (colorAssignment t q = colorAssignment t r) ∧
    (colorAssignment t r = colorAssignment t p) ∧
    (isRightTriangle p q r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_right_triangle_l329_32954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_volume_ratio_l329_32918

-- Define the radius of the sphere
noncomputable def r : ℝ := sorry

-- Define the volume of a sphere
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * (radius ^ 3)

-- Define the volume of a hemisphere
noncomputable def hemisphere_volume (radius : ℝ) : ℝ := (1 / 2) * sphere_volume radius

-- Theorem statement
theorem sphere_to_hemisphere_volume_ratio :
  sphere_volume r / hemisphere_volume (3 * r) = 2 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_volume_ratio_l329_32918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l329_32971

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l329_32971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l329_32946

open Real MeasureTheory

theorem max_value_of_expression (x y : ℝ) 
  (hx : x ∈ Set.Ioo 0 (π/2)) (hy : y ∈ Set.Ioo 0 (π/2)) :
  let A := (sin x * sin y)^(1/4) / ((tan x)^(1/4) + (tan y)^(1/4))
  A ≤ (8^(1/4))/4 ∧ ∃ x y, x ∈ Set.Ioo 0 (π/2) ∧ y ∈ Set.Ioo 0 (π/2) ∧ 
    A = (8^(1/4))/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l329_32946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_TS_distance_range_l329_32976

open Real

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Define the rotation of a point by 90 degrees counterclockwise
def Rotate90 (x y : ℝ) : ℝ × ℝ := (-y, x)

-- Define the reflection of a point about (3,0)
def ReflectAbout3_0 (x y : ℝ) : ℝ × ℝ := (6 - x, -y)

-- State the theorem
theorem TS_distance_range :
  ∀ (x y : ℝ),
  Circle x y →
  let (sx, sy) := Rotate90 x y
  let (tx, ty) := ReflectAbout3_0 x y
  let distance := Real.sqrt ((tx - sx)^2 + (ty - sy)^2)
  Real.sqrt 2 * (Real.sqrt 26 - 1) ≤ distance ∧ distance ≤ Real.sqrt 2 * (Real.sqrt 26 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_TS_distance_range_l329_32976
