import Mathlib

namespace NUMINAMATH_CALUDE_a_earnings_l1510_151012

-- Define the work rates and total wages
def a_rate : ℚ := 1 / 10
def b_rate : ℚ := 1 / 15
def total_wages : ℚ := 3400

-- Define A's share of the work when working together
def a_share : ℚ := a_rate / (a_rate + b_rate)

-- Theorem stating A's earnings
theorem a_earnings : a_share * total_wages = 2040 := by
  sorry

end NUMINAMATH_CALUDE_a_earnings_l1510_151012


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l1510_151018

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ -1 ∧ (x^3 - x^2) / (x^2 + 2*x + 1) + 2*x = -4 :=
by
  use 4/3
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l1510_151018


namespace NUMINAMATH_CALUDE_not_identity_element_l1510_151032

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b + 1

-- Theorem stating that 1 is not an identity element for * in S
theorem not_identity_element :
  ¬ (∀ a ∈ S, (star 1 a = a ∧ star a 1 = a)) :=
by sorry

end NUMINAMATH_CALUDE_not_identity_element_l1510_151032


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1510_151057

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 17) → 
  (∃ m : ℤ, N = 13 * m + 4) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1510_151057


namespace NUMINAMATH_CALUDE_tan_value_for_given_point_l1510_151006

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) (h : ∃ (r : Real), r * (Real.cos θ) = -Real.sqrt 3 / 2 ∧ r * (Real.sin θ) = 1 / 2) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_point_l1510_151006


namespace NUMINAMATH_CALUDE_appetizer_cost_per_person_l1510_151080

def potato_chip_cost : ℝ := 1.00
def creme_fraiche_cost : ℝ := 5.00
def caviar_cost : ℝ := 73.00
def num_people : ℕ := 3
def num_potato_chip_bags : ℕ := 3

theorem appetizer_cost_per_person :
  (num_potato_chip_bags * potato_chip_cost + creme_fraiche_cost + caviar_cost) / num_people = 27.00 := by
  sorry

end NUMINAMATH_CALUDE_appetizer_cost_per_person_l1510_151080


namespace NUMINAMATH_CALUDE_sum_perfect_square_values_l1510_151017

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_perfect_square_values :
  ∀ K : ℕ, K > 0 → K < 150 →
    (∃ N : ℕ, N < 150 ∧ sum_first_n K = N * N) ↔ K = 8 ∨ K = 49 ∨ K = 59 := by
  sorry

end NUMINAMATH_CALUDE_sum_perfect_square_values_l1510_151017


namespace NUMINAMATH_CALUDE_cubic_integer_roots_l1510_151061

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of distinct integer roots of a cubic polynomial -/
def num_distinct_integer_roots (p : CubicPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of distinct integer roots -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  num_distinct_integer_roots p ∈ ({0, 1, 2, 3} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_l1510_151061


namespace NUMINAMATH_CALUDE_area_ratio_concentric_spheres_specific_sphere_areas_l1510_151023

/-- Given two concentric spheres with radii R₁ and R₂, if a region on the smaller sphere
    has an area A₁, then the corresponding region on the larger sphere has an area A₂. -/
theorem area_ratio_concentric_spheres (R₁ R₂ A₁ A₂ : ℝ) 
    (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : A₁ > 0) :
  R₁ = 4 → R₂ = 6 → A₁ = 37 → A₂ = (R₂ / R₁)^2 * A₁ → A₂ = 83.25 := by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_sphere_areas :
  let R₁ : ℝ := 4
  let R₂ : ℝ := 6
  let A₁ : ℝ := 37
  let A₂ : ℝ := (R₂ / R₁)^2 * A₁
  A₂ = 83.25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_concentric_spheres_specific_sphere_areas_l1510_151023


namespace NUMINAMATH_CALUDE_lake_superior_weighted_average_l1510_151034

/-- Represents the data for fish caught in a lake -/
structure LakeFishData where
  species : List String
  counts : List Nat
  weights : List Float

/-- Calculates the weighted average weight of fish in a lake -/
def weightedAverageWeight (data : LakeFishData) : Float :=
  let totalWeight := (List.zip data.counts data.weights).map (fun (c, w) => c.toFloat * w) |> List.sum
  let totalCount := data.counts.sum
  totalWeight / totalCount.toFloat

/-- The fish data for Lake Superior -/
def lakeSuperiorData : LakeFishData :=
  { species := ["Perch", "Northern Pike", "Whitefish"]
  , counts := [17, 15, 8]
  , weights := [2.5, 4.0, 3.5] }

/-- Theorem stating that the weighted average weight of fish in Lake Superior is 3.2625kg -/
theorem lake_superior_weighted_average :
  weightedAverageWeight lakeSuperiorData = 3.2625 := by
  sorry

end NUMINAMATH_CALUDE_lake_superior_weighted_average_l1510_151034


namespace NUMINAMATH_CALUDE_race_length_race_length_is_165_l1510_151039

theorem race_length : ℝ → Prop :=
  fun x =>
    ∀ (speed_a speed_b speed_c : ℝ),
      speed_a > 0 ∧ speed_b > 0 ∧ speed_c > 0 →
      x > 35 →
      speed_b * x = speed_a * (x - 15) →
      speed_c * x = speed_a * (x - 35) →
      speed_c * (x - 15) = speed_b * (x - 22) →
      x = 165

theorem race_length_is_165 : race_length 165 := by
  sorry

end NUMINAMATH_CALUDE_race_length_race_length_is_165_l1510_151039


namespace NUMINAMATH_CALUDE_probability_test_l1510_151020

def probability_at_least_3_of_4 (p : ℝ) : ℝ :=
  (4 : ℝ) * p^3 * (1 - p) + p^4

theorem probability_test (p : ℝ) (hp : p = 4/5) :
  probability_at_least_3_of_4 p = 512/625 := by
  sorry

end NUMINAMATH_CALUDE_probability_test_l1510_151020


namespace NUMINAMATH_CALUDE_solution_set_R_solution_set_m_lower_bound_l1510_151014

-- Define the inequality
def inequality (x m : ℝ) : Prop := x^2 - 2*(m+1)*x + 4*m ≥ 0

-- Statement 1
theorem solution_set_R (m : ℝ) : 
  (∀ x, inequality x m) ↔ m = 1 := by sorry

-- Statement 2
theorem solution_set (m : ℝ) :
  (m = 1 ∧ ∀ x, inequality x m) ∨
  (m > 1 ∧ ∀ x, inequality x m ↔ (x ≤ 2 ∨ x ≥ 2*m)) ∨
  (m < 1 ∧ ∀ x, inequality x m ↔ (x ≤ 2*m ∨ x ≥ 2)) := by sorry

-- Statement 3
theorem m_lower_bound :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → inequality x m) → m ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_R_solution_set_m_lower_bound_l1510_151014


namespace NUMINAMATH_CALUDE_binomial_simplification_l1510_151092

/-- Given two binomials M and N in terms of x, prove that if 2(M) - 3(N) = 4x - 6 - 9x - 15,
    then N = 3x + 5 and the simplified expression P = -5x - 21 -/
theorem binomial_simplification (x : ℝ) (M N : ℝ → ℝ) :
  (∀ x, 2 * M x - 3 * N x = 4 * x - 6 - 9 * x - 15) →
  (∀ x, N x = 3 * x + 5) ∧
  (∀ x, 2 * M x - 3 * N x = -5 * x - 21) :=
by sorry

end NUMINAMATH_CALUDE_binomial_simplification_l1510_151092


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1510_151008

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 48) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1510_151008


namespace NUMINAMATH_CALUDE_inequalities_proof_l1510_151010

theorem inequalities_proof (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (a^3 * b < a * b^3) ∧ (a / b + b / a < -2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1510_151010


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1510_151029

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 48 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1510_151029


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1510_151024

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (7 * n + 21 = 2821) → (n + 6 = 406) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1510_151024


namespace NUMINAMATH_CALUDE_probability_ratio_l1510_151035

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing four slips with the same number -/
def p : ℚ := (distinct_numbers * (slips_per_number.choose drawn_slips)) / (total_slips.choose drawn_slips)

/-- The probability of drawing two pairs of different numbers -/
def q : ℚ := (distinct_numbers.choose 2 * (slips_per_number.choose 2) * (slips_per_number.choose 2)) / (total_slips.choose drawn_slips)

theorem probability_ratio :
  q / p = 90 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l1510_151035


namespace NUMINAMATH_CALUDE_p_plus_q_equals_27_over_2_l1510_151036

theorem p_plus_q_equals_27_over_2 (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 135 = 0)
  (hq : 12*q^3 - 90*q^2 - 450*q + 4950 = 0) :
  p + q = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_27_over_2_l1510_151036


namespace NUMINAMATH_CALUDE_number_of_observations_l1510_151054

theorem number_of_observations (original_mean new_mean : ℝ) (correction : ℝ) :
  original_mean = 36 →
  correction = 1 →
  new_mean = 36.02 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * new_mean = (n : ℝ) * original_mean + correction :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l1510_151054


namespace NUMINAMATH_CALUDE_temperature_is_dependent_l1510_151073

/-- Represents the variables in the solar water heater scenario -/
inductive SolarHeaterVariable
  | IntensityOfSunlight
  | TemperatureOfWater
  | DurationOfExposure
  | CapacityOfHeater

/-- Represents the relationship between two variables -/
structure Relationship where
  independent : SolarHeaterVariable
  dependent : SolarHeaterVariable

/-- Defines the relationship in the solar water heater scenario -/
def solarHeaterRelationship : Relationship :=
  { independent := SolarHeaterVariable.DurationOfExposure,
    dependent := SolarHeaterVariable.TemperatureOfWater }

/-- Theorem: The temperature of water is the dependent variable in the solar water heater scenario -/
theorem temperature_is_dependent :
  solarHeaterRelationship.dependent = SolarHeaterVariable.TemperatureOfWater :=
by sorry

end NUMINAMATH_CALUDE_temperature_is_dependent_l1510_151073


namespace NUMINAMATH_CALUDE_log_equation_solution_l1510_151000

theorem log_equation_solution :
  ∀ x : ℝ, (Real.log 729 / Real.log (3 * x) = x) →
    (x = 3 ∧ ¬ ∃ n : ℕ, x = n^2 ∧ ¬ ∃ m : ℕ, x = m^3 ∧ ∃ k : ℤ, x = k) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1510_151000


namespace NUMINAMATH_CALUDE_ball_picking_problem_l1510_151041

/-- Represents a bag with black and white balls -/
structure BallBag where
  total_balls : ℕ
  white_balls : ℕ
  black_balls : ℕ
  h_total : total_balls = white_balls + black_balls

/-- The probability of picking a white ball -/
def prob_white (bag : BallBag) : ℚ :=
  bag.white_balls / bag.total_balls

theorem ball_picking_problem (bag : BallBag) 
  (h_total : bag.total_balls = 4)
  (h_prob : prob_white bag = 1/2) :
  (bag.white_balls = 2) ∧ 
  (1/3 : ℚ) = (bag.white_balls * (bag.white_balls - 1) + bag.black_balls * (bag.black_balls - 1)) / 
               (bag.total_balls * (bag.total_balls - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ball_picking_problem_l1510_151041


namespace NUMINAMATH_CALUDE_root_cube_relation_l1510_151048

/-- The polynomial f(x) = x^3 + 2x^2 + 3x + 4 -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- The polynomial h(x) = x^3 + bx^2 + cx + d -/
def h (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- Theorem stating the relationship between f and h, and the values of b, c, and d -/
theorem root_cube_relation (b c d : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧
    (∀ x : ℝ, h x b c d = 0 ↔ ∃ r : ℝ, f r = 0 ∧ x = r^3)) →
  b = 6 ∧ c = -8 ∧ d = 16 := by
sorry

end NUMINAMATH_CALUDE_root_cube_relation_l1510_151048


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l1510_151003

theorem min_a_for_quadratic_inequality :
  (∀ x : ℝ, 0 < x → x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l1510_151003


namespace NUMINAMATH_CALUDE_selene_purchase_cost_l1510_151068

/-- Calculates the total cost of items after applying a discount -/
def total_cost_after_discount (camera_price : ℚ) (frame_price : ℚ) (camera_count : ℕ) (frame_count : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_before_discount := camera_price * camera_count + frame_price * frame_count
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

/-- Proves that Selene pays $551 for her purchase -/
theorem selene_purchase_cost :
  let camera_price : ℚ := 110
  let frame_price : ℚ := 120
  let camera_count : ℕ := 2
  let frame_count : ℕ := 3
  let discount_rate : ℚ := 5 / 100
  total_cost_after_discount camera_price frame_price camera_count frame_count discount_rate = 551 := by
  sorry


end NUMINAMATH_CALUDE_selene_purchase_cost_l1510_151068


namespace NUMINAMATH_CALUDE_cauchy_mean_value_theorem_sine_cosine_l1510_151019

open Real

theorem cauchy_mean_value_theorem_sine_cosine :
  ∃ c : ℝ, 0 < c ∧ c < π / 2 ∧
    (cos c) / (-sin c) = (sin (π / 2) - sin 0) / (cos (π / 2) - cos 0) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_mean_value_theorem_sine_cosine_l1510_151019


namespace NUMINAMATH_CALUDE_first_term_is_four_l1510_151045

/-- Geometric sequence with common ratio -2 and sum of first 5 terms equal to 44 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * (-2)) ∧ 
  (a 1 + a 2 + a 3 + a 4 + a 5 = 44)

/-- The first term of the geometric sequence is 4 -/
theorem first_term_is_four (a : ℕ → ℝ) (h : geometric_sequence a) : a 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_four_l1510_151045


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1510_151071

theorem linear_equation_condition (a : ℝ) : 
  (|a| - 1 = 1 ∧ a - 2 ≠ 0) ↔ a = -2 := by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1510_151071


namespace NUMINAMATH_CALUDE_apples_handed_out_is_19_l1510_151037

/-- Calculates the number of apples handed out to students in a cafeteria. -/
def apples_handed_out (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_pies * apples_per_pie)

/-- Proves that the number of apples handed out to students is 19. -/
theorem apples_handed_out_is_19 :
  apples_handed_out 75 7 8 = 19 := by
  sorry

#eval apples_handed_out 75 7 8

end NUMINAMATH_CALUDE_apples_handed_out_is_19_l1510_151037


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l1510_151083

/-- Calculates the final alcohol percentage of a solution after adding more alcohol and water. -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_alcohol_percentage = 0.05)
  (h3 : added_alcohol = 3.5)
  (h4 : added_water = 6.5) :
  let initial_alcohol := initial_volume * initial_alcohol_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_alcohol_percentage := final_alcohol / final_volume
  final_alcohol_percentage = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l1510_151083


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l1510_151089

/-- Represents a pyramid with a square base and specified face areas -/
structure Pyramid where
  base_area : ℝ
  face_area1 : ℝ
  face_area2 : ℝ

/-- Calculates the volume of a pyramid given its properties -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  let base_side := Real.sqrt p.base_area
  let height1 := 2 * p.face_area1 / base_side
  let height2 := 2 * p.face_area2 / base_side
  let a := (height1^2 - height2^2 + base_side^2) / (2 * base_side)
  let h := Real.sqrt (height1^2 - (base_side - a)^2)
  (1/3) * p.base_area * h

/-- The theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  let p := Pyramid.mk 256 128 112
  ∃ ε > 0, |pyramid_volume p - 1230.83| < ε :=
sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l1510_151089


namespace NUMINAMATH_CALUDE_least_period_is_30_l1510_151053

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

/-- The main theorem -/
theorem least_period_is_30 :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → IsLeastPeriod f 30 :=
sorry

end NUMINAMATH_CALUDE_least_period_is_30_l1510_151053


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1510_151051

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - x₁ - k^2 = 0 ∧ x₂^2 - x₂ - k^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1510_151051


namespace NUMINAMATH_CALUDE_package_cost_proof_l1510_151064

/-- The cost of a 12-roll package of paper towels -/
def package_cost : ℝ := 9

/-- The cost of one roll sold individually -/
def individual_roll_cost : ℝ := 1

/-- The number of rolls in a package -/
def rolls_per_package : ℕ := 12

/-- The percent of savings per roll for the package -/
def savings_percent : ℝ := 25

theorem package_cost_proof : 
  package_cost = rolls_per_package * (individual_roll_cost * (1 - savings_percent / 100)) :=
by sorry

end NUMINAMATH_CALUDE_package_cost_proof_l1510_151064


namespace NUMINAMATH_CALUDE_eggs_per_crate_l1510_151043

theorem eggs_per_crate (initial_crates : ℕ) (given_away : ℕ) (additional_crates : ℕ) (final_count : ℕ) :
  initial_crates = 6 →
  given_away = 2 →
  additional_crates = 5 →
  final_count = 270 →
  ∃ (eggs_per_crate : ℕ), eggs_per_crate = 30 ∧
    final_count = (initial_crates - given_away + additional_crates) * eggs_per_crate :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_crate_l1510_151043


namespace NUMINAMATH_CALUDE_existence_of_sequence_with_divisors_l1510_151077

theorem existence_of_sequence_with_divisors :
  ∃ f : ℕ → ℕ, ∀ k : ℕ, ∃ d : Finset ℕ,
    d.card ≥ k ∧
    ∀ x ∈ d, x > 0 ∧ (f k)^2 + f k + 2023 ≡ 0 [MOD x] :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequence_with_divisors_l1510_151077


namespace NUMINAMATH_CALUDE_probability_of_selecting_letter_l1510_151059

theorem probability_of_selecting_letter (total_letters : ℕ) (unique_letters : ℕ) 
  (h1 : total_letters = 26) (h2 : unique_letters = 8) : 
  (unique_letters : ℚ) / total_letters = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_letter_l1510_151059


namespace NUMINAMATH_CALUDE_max_min_difference_z_l1510_151001

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 3) 
  (sum_squares_condition : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w, w = x ∨ w = y ∨ w = z → z_min ≤ w ∧ w ≤ z_max) ∧ 
    z_max - z_min = 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l1510_151001


namespace NUMINAMATH_CALUDE_f_max_value_l1510_151016

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := S n / ((n + 32) * S (n + 1))

theorem f_max_value :
  (∀ n : ℕ, n > 0 → f n ≤ 1/50) ∧ (∃ n : ℕ, n > 0 ∧ f n = 1/50) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1510_151016


namespace NUMINAMATH_CALUDE_angle_relations_l1510_151082

theorem angle_relations (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan (α/2) = 1/3) (h4 : Real.cos (α - β) = -4/5) :
  Real.sin α = 3/5 ∧ 2*α + β = π := by
  sorry

end NUMINAMATH_CALUDE_angle_relations_l1510_151082


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1510_151007

theorem marble_selection_ways (total_marbles : ℕ) (selected_marbles : ℕ) (blue_marble : ℕ) :
  total_marbles = 10 →
  selected_marbles = 4 →
  blue_marble = 1 →
  (total_marbles.choose (selected_marbles - blue_marble)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1510_151007


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1510_151078

theorem polynomial_simplification (p : ℝ) :
  (2 * p^4 + 5 * p^3 - 3 * p + 4) + (-p^4 + 2 * p^3 - 7 * p^2 + 4 * p - 2) =
  p^4 + 7 * p^3 - 7 * p^2 + p + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1510_151078


namespace NUMINAMATH_CALUDE_shopping_trip_expenses_l1510_151076

theorem shopping_trip_expenses (total : ℝ) (food_percent : ℝ) :
  (total > 0) →
  (food_percent ≥ 0) →
  (food_percent ≤ 100) →
  (0.5 * total + food_percent / 100 * total + 0.4 * total = total) →
  (0.04 * 0.5 * total + 0.08 * 0.4 * total = 0.052 * total) →
  food_percent = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_expenses_l1510_151076


namespace NUMINAMATH_CALUDE_octal_7624_is_decimal_3988_l1510_151094

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem octal_7624_is_decimal_3988 : octal_to_decimal 7624 = 3988 := by
  sorry

end NUMINAMATH_CALUDE_octal_7624_is_decimal_3988_l1510_151094


namespace NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l1510_151093

/-- The probability of a specific lamp arrangement and state --/
def specific_arrangement_probability (total_lamps : ℕ) (purple_lamps : ℕ) (green_lamps : ℕ) (lamps_on : ℕ) : ℚ :=
  let total_arrangements := Nat.choose total_lamps purple_lamps * Nat.choose total_lamps lamps_on
  let specific_arrangements := Nat.choose (total_lamps - 2) (purple_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_on - 1)
  specific_arrangements / total_arrangements

/-- The main theorem statement --/
theorem specific_lamp_arrangement_probability :
  specific_arrangement_probability 8 4 4 4 = 4 / 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l1510_151093


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomials_l1510_151098

theorem roots_of_cubic_polynomials (a b : ℝ) (r s : ℝ) :
  (∃ t, r + s + t = 0 ∧ r * s + r * t + s * t = a) →
  (∃ t', r + 4 + s - 3 + t' = 0 ∧ (r + 4) * (s - 3) + (r + 4) * t' + (s - 3) * t' = a) →
  b = -330 ∨ b = 90 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomials_l1510_151098


namespace NUMINAMATH_CALUDE_proportion_with_reciprocals_l1510_151015

theorem proportion_with_reciprocals (a b c d : ℝ) : 
  a / b = c / d →  -- proportion
  b * c = 1 →      -- inner terms are reciprocals
  a = 0.2 →        -- one outer term is 0.2
  d = 5 :=         -- prove the other outer term is 5
by sorry

end NUMINAMATH_CALUDE_proportion_with_reciprocals_l1510_151015


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1510_151022

/-- Given a hyperbola E with equation x²/a² - y²/b² = 1 (where a > 0 and b > 0),
    if one of its asymptotes has a slope of 30°, then its eccentricity is 2√3/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.tan (π / 6)) →
  Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1510_151022


namespace NUMINAMATH_CALUDE_remainder_of_482157_div_6_l1510_151040

theorem remainder_of_482157_div_6 : 482157 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_482157_div_6_l1510_151040


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l1510_151002

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  ∃ x : ℝ, (x^2 + a*x + b = 0) ∨ (x^2 + c*x + d = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l1510_151002


namespace NUMINAMATH_CALUDE_last_three_positions_l1510_151063

/-- Represents the position of a person in the line after a certain number of rounds -/
def position (round : ℕ) : ℕ :=
  match round with
  | 0 => 3
  | n + 1 =>
    let prev := position n
    if prev % 2 = 1 then (3 * prev - 1) / 2 else (3 * prev - 2) / 2

/-- The theorem stating the initial positions of the last three people remaining -/
theorem last_three_positions (initial_count : ℕ) (h : initial_count = 2009) :
  ∃ (rounds : ℕ), position rounds = 1600 ∧ 
    (∀ k, k > rounds → position k < 1600) ∧
    (∀ n, n ≤ initial_count → n ≠ 1 → n ≠ 2 → n ≠ 1600 → 
      ∃ m, m ≤ rounds ∧ (3 * (position m)) % n = 0) :=
sorry

end NUMINAMATH_CALUDE_last_three_positions_l1510_151063


namespace NUMINAMATH_CALUDE_abs_one_plus_i_over_i_l1510_151067

def i : ℂ := Complex.I

theorem abs_one_plus_i_over_i : Complex.abs ((1 + i) / i) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_abs_one_plus_i_over_i_l1510_151067


namespace NUMINAMATH_CALUDE_more_stable_performance_l1510_151084

/-- Represents a student's performance in throwing solid balls -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if a student's performance is more stable than another's -/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students with equal average scores, the one with smaller variance has more stable performance -/
theorem more_stable_performance (student_A student_B : StudentPerformance)
  (h_equal_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 0.1)
  (h_B_variance : student_B.variance = 0.02) :
  more_stable student_B student_A :=
by sorry

end NUMINAMATH_CALUDE_more_stable_performance_l1510_151084


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l1510_151049

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l1510_151049


namespace NUMINAMATH_CALUDE_domain_shift_l1510_151044

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc (-1) 1

-- Define the domain of f(x + 1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) 0

-- Theorem statement
theorem domain_shift :
  (∀ x ∈ domain_f, f x ≠ 0) →
  (∀ y ∈ domain_f_shifted, f (y + 1) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l1510_151044


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1510_151091

theorem quadratic_equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = -3 ∧ 
  (∀ x : ℝ, x^2 + 2*x + 1 = 4 ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1510_151091


namespace NUMINAMATH_CALUDE_string_length_for_circular_token_l1510_151088

theorem string_length_for_circular_token : 
  let area : ℝ := 616
  let pi_approx : ℝ := 22 / 7
  let extra_length : ℝ := 5
  let radius : ℝ := Real.sqrt (area * 7 / 22)
  let circumference : ℝ := 2 * pi_approx * radius
  circumference + extra_length = 93 := by sorry

end NUMINAMATH_CALUDE_string_length_for_circular_token_l1510_151088


namespace NUMINAMATH_CALUDE_billy_youtube_suggestions_l1510_151005

/-- The number of suggestion sets Billy watches before finding a video he likes -/
def num_sets : ℕ := 5

/-- The number of videos Billy watches from the final set -/
def videos_from_final_set : ℕ := 5

/-- The total number of videos Billy watches -/
def total_videos : ℕ := 65

/-- The number of suggestions generated each time -/
def suggestions_per_set : ℕ := 15

theorem billy_youtube_suggestions :
  (num_sets - 1) * suggestions_per_set + videos_from_final_set = total_videos :=
by sorry

end NUMINAMATH_CALUDE_billy_youtube_suggestions_l1510_151005


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l1510_151046

/-- The probability of selecting at least one female student when choosing 2 people
    from a group of 3 male and 2 female students is 0.7 -/
theorem probability_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (selected : ℕ) (h1 : total_students = male_students + female_students)
  (h2 : total_students = 5) (h3 : male_students = 3) (h4 : female_students = 2) (h5 : selected = 2) :
  (Nat.choose total_students selected - Nat.choose male_students selected : ℚ) /
  Nat.choose total_students selected = 7/10 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l1510_151046


namespace NUMINAMATH_CALUDE_boy_scouts_permission_slips_l1510_151042

theorem boy_scouts_permission_slips 
  (total_with_slips : ℝ) 
  (boy_scouts_percentage : ℝ) 
  (girl_scouts_with_slips : ℝ) :
  total_with_slips = 0.60 →
  boy_scouts_percentage = 0.45 →
  girl_scouts_with_slips = 0.6818 →
  (boy_scouts_percentage * (total_with_slips - (1 - boy_scouts_percentage) * girl_scouts_with_slips)) / 
  (boy_scouts_percentage * (1 - (1 - boy_scouts_percentage) * girl_scouts_with_slips)) = 0.50 :=
by sorry

end NUMINAMATH_CALUDE_boy_scouts_permission_slips_l1510_151042


namespace NUMINAMATH_CALUDE_apples_bought_l1510_151072

theorem apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) : 
  initial = 17 → used = 2 → final = 38 → final - (initial - used) = 23 := by
  sorry

end NUMINAMATH_CALUDE_apples_bought_l1510_151072


namespace NUMINAMATH_CALUDE_exam_students_count_l1510_151026

theorem exam_students_count : 
  ∀ (total : ℕ) (first_div second_div just_passed : ℝ),
    first_div = 0.25 * total →
    second_div = 0.54 * total →
    just_passed = total - first_div - second_div →
    just_passed = 63 →
    total = 300 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l1510_151026


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1510_151079

theorem regular_polygon_sides (exterior_angle : ℝ) : 
  exterior_angle = 30 → (360 / exterior_angle : ℝ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1510_151079


namespace NUMINAMATH_CALUDE_vectors_theorem_l1510_151030

/-- Two non-collinear vectors in a plane -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  e₁ : V
  e₂ : V
  noncollinear : ¬ ∃ (r : ℝ), e₁ = r • e₂

/-- Definition of vectors AB, CB, and CD -/
def vectors_relation (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (ncv : NonCollinearVectors V) (k : ℝ) : Prop :=
  ∃ (A B C D : V),
    B - A = ncv.e₁ - k • ncv.e₂ ∧
    B - C = 2 • ncv.e₁ + ncv.e₂ ∧
    D - C = 3 • ncv.e₁ - ncv.e₂

/-- Collinearity of points A, B, and D -/
def collinear (V : Type*) [AddCommGroup V] [Module ℝ V] (A B D : V) : Prop :=
  ∃ (t : ℝ), D - A = t • (B - A)

/-- The main theorem -/
theorem vectors_theorem (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (ncv : NonCollinearVectors V) :
  ∀ k, vectors_relation V ncv k → 
  (∃ A B D, collinear V A B D) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_vectors_theorem_l1510_151030


namespace NUMINAMATH_CALUDE_sum_of_squares_impossible_l1510_151074

theorem sum_of_squares_impossible (n : ℤ) :
  (n % 4 = 3 → ¬∃ (a b : ℤ), n = a^2 + b^2) ∧
  (n % 8 = 7 → ¬∃ (a b c : ℤ), n = a^2 + b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_impossible_l1510_151074


namespace NUMINAMATH_CALUDE_eighth_root_of_549755289601_l1510_151021

theorem eighth_root_of_549755289601 :
  let n : ℕ := 549755289601
  (n = 1 * 100^8 + 8 * 100^7 + 28 * 100^6 + 56 * 100^5 + 70 * 100^4 + 
       56 * 100^3 + 28 * 100^2 + 8 * 100 + 1) →
  (n : ℝ)^(1/8 : ℝ) = 101 := by
sorry

end NUMINAMATH_CALUDE_eighth_root_of_549755289601_l1510_151021


namespace NUMINAMATH_CALUDE_children_count_l1510_151038

def number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) : ℕ :=
  total_crayons / crayons_per_child

theorem children_count : number_of_children 6 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_children_count_l1510_151038


namespace NUMINAMATH_CALUDE_total_peanuts_l1510_151097

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35

theorem total_peanuts : jose_peanuts + kenya_peanuts + malachi_peanuts = 386 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l1510_151097


namespace NUMINAMATH_CALUDE_min_y_value_l1510_151096

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 36*y) :
  ∃ (y_min : ℝ), y_min = 18 - Real.sqrt 388 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 16*x' + 36*y' → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_value_l1510_151096


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1510_151033

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k ∈ Finset.range n, ¬ Nat.Prime (n! - k) :=
by sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1510_151033


namespace NUMINAMATH_CALUDE_sophomore_freshman_difference_l1510_151090

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  freshman : ℕ
  sophomore : ℕ
  junior : ℕ

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  freshman : ℕ
  sophomore : ℕ
  junior : ℕ

/-- Calculates the stratified sample distribution based on the grade distribution and total sample size -/
def stratifiedSample (grades : GradeDistribution) (totalSample : ℕ) : SampleDistribution :=
  let total := grades.freshman + grades.sophomore + grades.junior
  let freshmanSample := (grades.freshman * totalSample) / total
  let sophomoreSample := (grades.sophomore * totalSample) / total
  let juniorSample := totalSample - freshmanSample - sophomoreSample
  { freshman := freshmanSample
  , sophomore := sophomoreSample
  , junior := juniorSample }

/-- The main theorem to be proved -/
theorem sophomore_freshman_difference
  (grades : GradeDistribution)
  (h1 : grades.freshman = 1000)
  (h2 : grades.sophomore = 1050)
  (h3 : grades.junior = 1200)
  (totalSample : ℕ)
  (h4 : totalSample = 65) :
  let sample := stratifiedSample grades totalSample
  sample.sophomore = sample.freshman + 1 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_freshman_difference_l1510_151090


namespace NUMINAMATH_CALUDE_total_theme_parks_eq_395_l1510_151056

/-- The number of theme parks in four towns -/
def total_theme_parks (jamestown venice marina_del_ray newport_beach : ℕ) : ℕ :=
  jamestown + venice + marina_del_ray + newport_beach

/-- Theorem: The total number of theme parks in four towns is 395 -/
theorem total_theme_parks_eq_395 :
  ∃ (jamestown venice marina_del_ray newport_beach : ℕ),
    jamestown = 35 ∧
    venice = jamestown + 40 ∧
    marina_del_ray = jamestown + 60 ∧
    newport_beach = 2 * marina_del_ray ∧
    total_theme_parks jamestown venice marina_del_ray newport_beach = 395 :=
by sorry

end NUMINAMATH_CALUDE_total_theme_parks_eq_395_l1510_151056


namespace NUMINAMATH_CALUDE_production_exceeds_target_in_2022_l1510_151069

def initial_production : ℕ := 20000
def annual_increase_rate : ℝ := 0.2
def target_production : ℕ := 60000
def start_year : ℕ := 2015

theorem production_exceeds_target_in_2022 :
  let production_after_n_years (n : ℕ) := initial_production * (1 + annual_increase_rate) ^ n
  ∀ y : ℕ, y < 2022 - start_year → production_after_n_years y ≤ target_production ∧
  production_after_n_years (2022 - start_year) > target_production :=
by sorry

end NUMINAMATH_CALUDE_production_exceeds_target_in_2022_l1510_151069


namespace NUMINAMATH_CALUDE_embankment_construction_time_l1510_151052

/-- Given that 60 workers take 3 days to build half of an embankment,
    prove that 45 workers would take 8 days to build the entire embankment,
    assuming all workers work at the same rate. -/
theorem embankment_construction_time
  (workers_60 : ℕ) (days_60 : ℕ) (half_embankment : ℚ)
  (workers_45 : ℕ) (days_45 : ℕ) (full_embankment : ℚ)
  (h1 : workers_60 = 60)
  (h2 : days_60 = 3)
  (h3 : half_embankment = 1/2)
  (h4 : workers_45 = 45)
  (h5 : days_45 = 8)
  (h6 : full_embankment = 1)
  (h7 : ∀ w d, w * d * half_embankment = workers_60 * days_60 * half_embankment →
               w * d * full_embankment = workers_45 * days_45 * full_embankment) :
  workers_45 * days_45 * full_embankment = workers_60 * days_60 * full_embankment :=
by sorry

end NUMINAMATH_CALUDE_embankment_construction_time_l1510_151052


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_perfect_square_coefficient_l1510_151062

/-- A quadratic expression is a perfect square if and only if its discriminant is zero -/
theorem quadratic_is_perfect_square (a b c : ℝ) :
  (∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2) ↔ b^2 = 4 * a * c := by sorry

/-- The main theorem: If 6x^2 + cx + 16 is a perfect square, then c = 8√6 -/
theorem perfect_square_coefficient (c : ℝ) :
  (∃ p q : ℝ, ∀ x, 6 * x^2 + c * x + 16 = (p * x + q)^2) → c = 8 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_perfect_square_coefficient_l1510_151062


namespace NUMINAMATH_CALUDE_circle_center_distance_l1510_151099

theorem circle_center_distance (x y : ℝ) : 
  (x^2 + y^2 = 6*x + 8*y + 9) → 
  Real.sqrt ((11 - x)^2 + (5 - y)^2) = Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_circle_center_distance_l1510_151099


namespace NUMINAMATH_CALUDE_sandras_puppies_l1510_151060

theorem sandras_puppies (total_portions : ℕ) (num_days : ℕ) (feedings_per_day : ℕ) :
  total_portions = 105 →
  num_days = 5 →
  feedings_per_day = 3 →
  (total_portions / num_days) / feedings_per_day = 7 :=
by sorry

end NUMINAMATH_CALUDE_sandras_puppies_l1510_151060


namespace NUMINAMATH_CALUDE_john_gathered_20_l1510_151085

/-- Given the total number of milk bottles and the number Marcus gathered,
    calculate the number of milk bottles John gathered. -/
def john_bottles (total : ℕ) (marcus : ℕ) : ℕ :=
  total - marcus

/-- Theorem stating that given 45 total milk bottles and 25 gathered by Marcus,
    John gathered 20 milk bottles. -/
theorem john_gathered_20 :
  john_bottles 45 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_gathered_20_l1510_151085


namespace NUMINAMATH_CALUDE_meeting_point_symmetry_l1510_151047

theorem meeting_point_symmetry 
  (d : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / (a + b) * (d - a / 2) - b / (a + b) * d = -2) →
  (a / (a + b) * (d - b / 2) - a / (a + b) * d = -2) :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_symmetry_l1510_151047


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l1510_151070

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y < 3 → f a x > f a y) → 
  (a ≥ 0 ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l1510_151070


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_a_geq_2_sufficient_not_necessary_l1510_151055

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | -1-a ≤ x ∧ x ≤ 1-a}

-- Theorem 1: If A ∩ B = {x | 1/2 ≤ x < 1}, then a = -3/2
theorem intersection_implies_a_value (a : ℝ) : 
  A ∩ B a = {x | 1/2 ≤ x ∧ x < 1} → a = -3/2 := by sorry

-- Theorem 2: a ≥ 2 is a sufficient but not necessary condition for A ∩ B = ∅
theorem a_geq_2_sufficient_not_necessary (a : ℝ) :
  (a ≥ 2 → A ∩ B a = ∅) ∧ ¬(A ∩ B a = ∅ → a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_a_geq_2_sufficient_not_necessary_l1510_151055


namespace NUMINAMATH_CALUDE_airplane_purchase_exceeds_budget_l1510_151095

/-- Proves that the total cost of purchasing the airplane exceeds $5.00 USD -/
theorem airplane_purchase_exceeds_budget : 
  let initial_budget : ℝ := 5.00
  let airplane_cost_eur : ℝ := 3.80
  let exchange_rate : ℝ := 0.82
  let sales_tax_rate : ℝ := 0.075
  let credit_card_surcharge_rate : ℝ := 0.035
  let processing_fee_usd : ℝ := 0.25
  
  let airplane_cost_usd : ℝ := airplane_cost_eur / exchange_rate
  let sales_tax : ℝ := airplane_cost_usd * sales_tax_rate
  let credit_card_surcharge : ℝ := airplane_cost_usd * credit_card_surcharge_rate
  let total_cost : ℝ := airplane_cost_usd + sales_tax + credit_card_surcharge + processing_fee_usd
  
  total_cost > initial_budget := by
  sorry

#check airplane_purchase_exceeds_budget

end NUMINAMATH_CALUDE_airplane_purchase_exceeds_budget_l1510_151095


namespace NUMINAMATH_CALUDE_sum_of_roots_l1510_151075

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 - 4*a + 12 = 0)
  (hb : 3*b^3 + 9*b^2 - 11*b - 3 = 0) : 
  a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1510_151075


namespace NUMINAMATH_CALUDE_distance_AB_is_336_l1510_151086

/-- The distance between two points A and B, given the conditions of the problem. -/
def distance_AB : ℝ :=
  let t_total := 3.5  -- Total time in hours
  let t_car3 := 3     -- Time for Car 3 to reach A
  let d_car1_left := 84  -- Distance left for Car 1 at 10:30 AM
  let d_car2_fraction := 3/8  -- Fraction of total distance Car 2 has traveled when Car 1 and 3 meet
  336

/-- The theorem stating that the distance between A and B is 336 km. -/
theorem distance_AB_is_336 :
  let d := distance_AB
  let v1 := d / 3.5 - 24  -- Speed of Car 1
  let v2 := d / 3.5       -- Speed of Car 2
  let v3 := d / 6         -- Speed of Car 3
  (v1 + v3 = 8/3 * v2) ∧  -- Condition when Car 1 and 3 meet
  (v3 * 3 = d / 2) ∧      -- Car 3 reaches A at 10:00 AM
  (v2 * 3.5 = d) ∧        -- Car 2 reaches A at 10:30 AM
  (d - v1 * 3.5 = 84) →   -- Car 1 is 84 km from B at 10:30 AM
  d = 336 := by
  sorry


end NUMINAMATH_CALUDE_distance_AB_is_336_l1510_151086


namespace NUMINAMATH_CALUDE_sum_of_distances_less_than_diagonal_l1510_151013

-- Define the quadrilateral ABCD and point P
variable (A B C D P : ℝ × ℝ)

-- Define the conditions
variable (h1 : IsConvex A B C D)
variable (h2 : dist A B = dist C D)
variable (h3 : IsInside P A B C D)
variable (h4 : angle P B A + angle P C D = π)

-- State the theorem
theorem sum_of_distances_less_than_diagonal :
  dist P B + dist P C < dist A D :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_less_than_diagonal_l1510_151013


namespace NUMINAMATH_CALUDE_problem_solution_l1510_151031

theorem problem_solution : (2200 - 2023)^2 / 196 = 144 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1510_151031


namespace NUMINAMATH_CALUDE_tan_order_l1510_151025

open Real

noncomputable def f (x : ℝ) := tan (x + π/4)

theorem tan_order : f 0 > f (-1) ∧ f (-1) > f 1 := by sorry

end NUMINAMATH_CALUDE_tan_order_l1510_151025


namespace NUMINAMATH_CALUDE_wall_volume_calculation_l1510_151028

/-- Proves that the volume of a wall is 345 cubic meters given specific brick dimensions and quantity --/
theorem wall_volume_calculation (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
  (brick_count : ℕ) (h1 : brick_length = 20) (h2 : brick_width = 10) (h3 : brick_height = 7.5) 
  (h4 : brick_count = 23000) : 
  (brick_length * brick_width * brick_height * brick_count) / 1000000 = 345 := by
  sorry

end NUMINAMATH_CALUDE_wall_volume_calculation_l1510_151028


namespace NUMINAMATH_CALUDE_min_fraction_value_l1510_151011

/-- A function that checks if a natural number contains the digit string "11235" -/
def contains_11235 (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem min_fraction_value (N k : ℕ) (h1 : N > 0) (h2 : k > 0) (h3 : contains_11235 N) (h4 : 10^k > N) :
  (∀ N' k' : ℕ, N' > 0 → k' > 0 → contains_11235 N' → 10^k' > N' →
    (10^k' - 1) / Nat.gcd (10^k' - 1) N' ≥ 89) ∧
  (∃ N' k' : ℕ, N' > 0 ∧ k' > 0 ∧ contains_11235 N' ∧ 10^k' > N' ∧
    (10^k' - 1) / Nat.gcd (10^k' - 1) N' = 89) :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_value_l1510_151011


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1510_151027

theorem inequality_not_always_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a * c < 0) :
  ∃ a b c, a < b ∧ b < c ∧ a * c < 0 ∧ c^2 / a ≥ b^2 / a :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1510_151027


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l1510_151050

theorem triangle_side_b_value (A B C : ℝ) (a b c : ℝ) :
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 2 →
  (a / Real.sin A = b / Real.sin B) →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l1510_151050


namespace NUMINAMATH_CALUDE_teacher_number_game_l1510_151081

theorem teacher_number_game (x : ℝ) : 
  let max_result := 2 * (3 * (x + 1))
  let lisa_result := 2 * ((max_result / 2) - 1)
  lisa_result = 2 * x + 2 := by sorry

end NUMINAMATH_CALUDE_teacher_number_game_l1510_151081


namespace NUMINAMATH_CALUDE_triangle_area_l1510_151004

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_area (abc : Triangle) (h1 : abc.c = 3) 
  (h2 : abc.a / Real.cos abc.A = abc.b / Real.cos abc.B)
  (h3 : Real.cos abc.C = 1/4) : 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = (3 * Real.sqrt 15) / 4 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l1510_151004


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l1510_151009

theorem sum_x_y_equals_negative_one (x y : ℝ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : 5 * x + 3 * y = 1) : 
  x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l1510_151009


namespace NUMINAMATH_CALUDE_problem_solution_l1510_151066

theorem problem_solution (P Q : ℚ) : 
  (4 / 7 : ℚ) = P / 63 ∧ (4 / 7 : ℚ) = 98 / (Q - 14) → P + Q = 221.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1510_151066


namespace NUMINAMATH_CALUDE_six_paths_from_M_to_N_l1510_151065

/- Define a directed graph with vertices and edges -/
def Graph : Type := List (Char × Char)

/- Define the graph structure for our problem -/
def problemGraph : Graph := [
  ('M', 'A'), ('M', 'B'),
  ('A', 'C'), ('A', 'D'),
  ('B', 'C'), ('B', 'N'),
  ('C', 'N'), ('D', 'N')
]

/- Function to count paths between two vertices -/
def countPaths (g : Graph) (start finish : Char) : Nat :=
  sorry

/- Theorem stating that there are 6 paths from M to N -/
theorem six_paths_from_M_to_N :
  countPaths problemGraph 'M' 'N' = 6 :=
sorry

end NUMINAMATH_CALUDE_six_paths_from_M_to_N_l1510_151065


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l1510_151087

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3360

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 330

theorem vegetable_ghee_weight : 
  weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
  weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume = total_weight := by
sorry

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l1510_151087


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1510_151058

/-- The polynomial P(x) = 3(x^8 - 2x^5 + 4x^3 - 7) - 5(2x^4 - 3x^2 + 8) + 6(x^6 - 3) -/
def P (x : ℝ) : ℝ := 3 * (x^8 - 2*x^5 + 4*x^3 - 7) - 5 * (2*x^4 - 3*x^2 + 8) + 6 * (x^6 - 3)

/-- The sum of the coefficients of P(x) is equal to P(1) -/
theorem sum_of_coefficients : P 1 = -59 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1510_151058
