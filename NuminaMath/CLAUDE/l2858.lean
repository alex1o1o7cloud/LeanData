import Mathlib

namespace NUMINAMATH_CALUDE_P_symmetric_l2858_285879

-- Define the polynomial sequence P_m
def P : ℕ → ℝ → ℝ → ℝ → ℝ
  | 0, x, y, z => 1
  | m + 1, x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

-- State the theorem
theorem P_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m x z y ∧ P m x y z = P m y x z := by
  sorry

end NUMINAMATH_CALUDE_P_symmetric_l2858_285879


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2858_285880

theorem complete_square_quadratic (x : ℝ) : ∃ (a b : ℝ), (x^2 + 10*x - 1 = 0) ↔ ((x + a)^2 = b) ∧ b = 26 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2858_285880


namespace NUMINAMATH_CALUDE_decreasing_prop_function_k_range_l2858_285877

/-- A proportional function y = (k-3)x where y decreases as x increases -/
def decreasing_prop_function (k : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → y x₁ > y x₂ ∧ y = λ t => (k - 3) * x t

/-- Theorem: If y decreases as x increases in the function y = (k-3)x, then k < 3 -/
theorem decreasing_prop_function_k_range (k : ℝ) (x y : ℝ → ℝ) :
  decreasing_prop_function k x y → k < 3 := by
  sorry


end NUMINAMATH_CALUDE_decreasing_prop_function_k_range_l2858_285877


namespace NUMINAMATH_CALUDE_inequality_proof_l2858_285837

theorem inequality_proof (a b c : ℝ) :
  a^4 + b^4 + c^4 ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 ∧
  a^2*b^2 + b^2*c^2 + c^2*a^2 ≥ a*b*c*(a + b + c) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2858_285837


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2858_285831

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2858_285831


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l2858_285830

/-- The maximum area of an equilateral triangle inscribed in a 12x13 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), A = 205 * Real.sqrt 3 - 468 ∧
  ∀ (triangle_area : ℝ),
    (∃ (x y : ℝ),
      0 ≤ x ∧ x ≤ 12 ∧
      0 ≤ y ∧ y ≤ 13 ∧
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l2858_285830


namespace NUMINAMATH_CALUDE_newsstand_profit_optimization_l2858_285887

/-- Newsstand profit optimization problem -/
theorem newsstand_profit_optimization :
  let buying_price : ℚ := 60 / 100
  let selling_price : ℚ := 1
  let return_price : ℚ := 10 / 100
  let high_demand_days : ℕ := 20
  let low_demand_days : ℕ := 10
  let high_demand_sales : ℕ := 400
  let low_demand_sales : ℕ := 250
  let total_days : ℕ := high_demand_days + low_demand_days
  let profit_per_sold (x : ℕ) : ℚ := 
    (high_demand_days * (min x high_demand_sales) + 
     low_demand_days * (min x low_demand_sales)) * (selling_price - buying_price)
  let loss_per_unsold (x : ℕ) : ℚ := 
    (high_demand_days * (x - min x high_demand_sales) + 
     low_demand_days * (x - min x low_demand_sales)) * (buying_price - return_price)
  let total_profit (x : ℕ) : ℚ := profit_per_sold x - loss_per_unsold x
  ∀ x : ℕ, total_profit x ≤ total_profit high_demand_sales ∧ 
           total_profit high_demand_sales = 2450 / 100 := by
  sorry

end NUMINAMATH_CALUDE_newsstand_profit_optimization_l2858_285887


namespace NUMINAMATH_CALUDE_car_average_speed_l2858_285882

/-- Proves that the average speed of a car traveling 140 km in the first hour
    and 40 km in the second hour is 90 km/h. -/
theorem car_average_speed : 
  let speed1 : ℝ := 140 -- Speed in km/h for the first hour
  let speed2 : ℝ := 40  -- Speed in km/h for the second hour
  let time1 : ℝ := 1    -- Time in hours for the first hour
  let time2 : ℝ := 1    -- Time in hours for the second hour
  let total_distance : ℝ := speed1 * time1 + speed2 * time2
  let total_time : ℝ := time1 + time2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 90 := by
sorry

end NUMINAMATH_CALUDE_car_average_speed_l2858_285882


namespace NUMINAMATH_CALUDE_train_speed_l2858_285852

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 360) (h2 : bridge_length = 140) (h3 : time = 40) :
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2858_285852


namespace NUMINAMATH_CALUDE_power_calculation_l2858_285862

theorem power_calculation : 9^6 * 3^9 / 27^5 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2858_285862


namespace NUMINAMATH_CALUDE_suv_highway_mileage_l2858_285826

/-- Given an SUV with specified city mileage and maximum distance on a fixed amount of gasoline,
    calculate its highway mileage. -/
theorem suv_highway_mileage 
  (city_mpg : ℝ) 
  (max_distance : ℝ) 
  (gas_amount : ℝ) 
  (h_city_mpg : city_mpg = 7.6)
  (h_max_distance : max_distance = 292.8)
  (h_gas_amount : gas_amount = 24) :
  max_distance / gas_amount = 12.2 := by
  sorry

end NUMINAMATH_CALUDE_suv_highway_mileage_l2858_285826


namespace NUMINAMATH_CALUDE_first_supply_cost_l2858_285885

theorem first_supply_cost (total_budget : ℕ) (remaining_budget : ℕ) (second_supply_cost : ℕ) :
  total_budget = 56 →
  remaining_budget = 19 →
  second_supply_cost = 24 →
  total_budget - remaining_budget - second_supply_cost = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_first_supply_cost_l2858_285885


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2858_285800

theorem pure_imaginary_condition (θ : ℝ) : 
  let z : ℂ := (Complex.exp (Complex.I * -θ)) * (1 + Complex.I)
  θ = 3 * Real.pi / 4 → Complex.re z = 0 ∧ Complex.im z ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2858_285800


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l2858_285891

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90) 
  (h2 : male_count = 8) 
  (h3 : male_average = 84) 
  (h4 : female_average = 92) : 
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧ 
    female_count = 24 :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l2858_285891


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l2858_285838

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (X : BinomialDistribution) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_two_successes :
  let X : BinomialDistribution := { n := 6, p := 1/3, h_p := by norm_num }
  binomialPMF X 2 = 80/243 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l2858_285838


namespace NUMINAMATH_CALUDE_ellipse_condition_l2858_285881

/-- Represents the equation (x^2)/(6-k) + (y^2)/(k-4) = 1 --/
def is_ellipse (k : ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, x^2 / (6-k) + y^2 / (k-4) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (k : ℝ) :
  (is_ellipse k → 4 < k ∧ k < 6) ∧
  (∃ k : ℝ, 4 < k ∧ k < 6 ∧ ¬is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2858_285881


namespace NUMINAMATH_CALUDE_work_scaling_l2858_285813

theorem work_scaling (people : ℕ) (work : ℕ) (days : ℕ) :
  (people = 3 ∧ work = 3 ∧ days = 3) →
  ∃ (scaled_people : ℕ), 
    scaled_people * work * days = 9 * people * work * days ∧
    scaled_people = 9 :=
by sorry

end NUMINAMATH_CALUDE_work_scaling_l2858_285813


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2858_285824

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * 2 + 1 * a + (Complex.I : ℂ) * (2 * a) + b = (Complex.I : ℂ) * 2 → 
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2858_285824


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_line_through_intersection_with_equal_intercepts_l2858_285859

-- Define the lines
def line1 (x y : ℝ) := x + 2*y - 5 = 0
def line2 (x y : ℝ) := 3*x - y - 1 = 0
def line3 (x y : ℝ) := 5*x - y + 100 = 0
def line4 (x y : ℝ) := 2*x + y - 8 = 0
def line5 (x y : ℝ) := x - 2*y + 1 = 0

-- Define the result lines
def result_line1 (x y : ℝ) := 5*x - y - 3 = 0
def result_line2a (x y : ℝ) := 2*x - 3*y = 0
def result_line2b (x y : ℝ) := x + y - 5 = 0

-- Theorem for the first part
theorem line_through_intersection_and_parallel :
  ∃ (x₀ y₀ : ℝ), line1 x₀ y₀ ∧ line2 x₀ y₀ →
  ∀ (x y : ℝ), (y - y₀ = (5 : ℝ) * (x - x₀)) ↔ result_line1 x y :=
sorry

-- Theorem for the second part
theorem line_through_intersection_with_equal_intercepts :
  ∃ (x₀ y₀ : ℝ), line4 x₀ y₀ ∧ line5 x₀ y₀ →
  ∀ (x y : ℝ), (∃ (a : ℝ), x = a ∧ y = a) →
  (result_line2a x y ∨ result_line2b x y) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_line_through_intersection_with_equal_intercepts_l2858_285859


namespace NUMINAMATH_CALUDE_unique_similar_triangles_l2858_285851

theorem unique_similar_triangles :
  ∀ (a b c a' b' c' : ℕ),
    a = 8 →
    a < a' →
    a < b →
    b < c →
    (b = b' ∧ c = c') ∨ (a = b' ∧ b = c') →
    (a' * b = a * b') ∧ (a' * c = a * c') →
    (a = 8 ∧ b = 12 ∧ c = 18 ∧ a' = 12 ∧ b' = 12 ∧ c' = 18) ∨
    (a = 8 ∧ b = 12 ∧ c = 18 ∧ a' = 12 ∧ b' = 18 ∧ c' = 27) :=
by sorry

end NUMINAMATH_CALUDE_unique_similar_triangles_l2858_285851


namespace NUMINAMATH_CALUDE_overtime_pay_rate_l2858_285855

/-- Calculate overtime pay rate given regular work conditions and total earnings --/
theorem overtime_pay_rate 
  (regular_days_per_week : ℕ)
  (regular_hours_per_day : ℕ)
  (regular_pay_rate : ℚ)
  (total_weeks : ℕ)
  (total_earnings : ℚ)
  (total_hours_worked : ℕ)
  (h1 : regular_days_per_week = 6)
  (h2 : regular_hours_per_day = 10)
  (h3 : regular_pay_rate = 21/10)
  (h4 : total_weeks = 4)
  (h5 : total_earnings = 525)
  (h6 : total_hours_worked = 245) :
  let regular_hours := regular_days_per_week * regular_hours_per_day * total_weeks
  let overtime_hours := total_hours_worked - regular_hours
  let regular_earnings := regular_hours * regular_pay_rate
  let overtime_earnings := total_earnings - regular_earnings
  overtime_earnings / overtime_hours = 21/5 := by
  sorry

#eval (21 : ℚ) / 5  -- This should output 4.2

end NUMINAMATH_CALUDE_overtime_pay_rate_l2858_285855


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l2858_285847

theorem coefficient_x_cubed_in_binomial_expansion :
  let n : ℕ := 10
  let k : ℕ := 3
  let a : ℤ := 1
  let b : ℤ := -2
  (Nat.choose n k) * b^k * a^(n-k) = -960 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l2858_285847


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2858_285815

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3 →
  x ≥ 40.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2858_285815


namespace NUMINAMATH_CALUDE_danny_collection_difference_l2858_285816

/-- The number of wrappers Danny has in his collection -/
def wrappers : ℕ := 67

/-- The number of soda cans Danny has in his collection -/
def soda_cans : ℕ := 22

/-- The difference between the number of wrappers and soda cans in Danny's collection -/
def wrapper_soda_difference : ℕ := wrappers - soda_cans

theorem danny_collection_difference :
  wrapper_soda_difference = 45 := by sorry

end NUMINAMATH_CALUDE_danny_collection_difference_l2858_285816


namespace NUMINAMATH_CALUDE_cow_ratio_proof_l2858_285898

theorem cow_ratio_proof (initial_cows : ℕ) (added_cows : ℕ) (remaining_cows : ℕ) : 
  initial_cows = 51 → 
  added_cows = 5 → 
  remaining_cows = 42 → 
  (initial_cows + added_cows - remaining_cows) / (initial_cows + added_cows) = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_proof_l2858_285898


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2858_285899

theorem complex_modulus_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = 4 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2858_285899


namespace NUMINAMATH_CALUDE_larger_number_proof_l2858_285811

theorem larger_number_proof (S L : ℤ) : 
  L = 4 * (S + 30) → L - S = 480 → L = 600 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2858_285811


namespace NUMINAMATH_CALUDE_trig_identity_l2858_285858

theorem trig_identity (x y : ℝ) :
  Real.sin (2 * x - y) * Real.cos (3 * y) + Real.cos (2 * x - y) * Real.sin (3 * y) = Real.sin (2 * x + 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2858_285858


namespace NUMINAMATH_CALUDE_quarterback_passes_l2858_285814

theorem quarterback_passes (total : ℕ) (left : ℕ) (right : ℕ) (center : ℕ) : 
  total = 50 ∧ 
  right = 2 * left ∧ 
  center = left + 2 ∧ 
  total = left + right + center → 
  left = 12 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l2858_285814


namespace NUMINAMATH_CALUDE_length_NM_is_3_l2858_285868

-- Define the points and segments
variable (A B M N : ℝ × ℝ)
variable (AB AM NM : ℝ)

-- State the given conditions
axiom length_AB : AB = 12
axiom M_midpoint_AB : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
axiom N_midpoint_AM : N = ((A.1 + M.1) / 2, (A.2 + M.2) / 2)

-- Define the theorem
theorem length_NM_is_3 : NM = 3 :=
sorry

end NUMINAMATH_CALUDE_length_NM_is_3_l2858_285868


namespace NUMINAMATH_CALUDE_amp_pamp_theorem_l2858_285872

-- Define the & operation
def amp (x : ℝ) : ℝ := 7 - x

-- Define the & prefix operation
def pamp (x : ℝ) : ℝ := x - 7

-- Theorem statement
theorem amp_pamp_theorem : pamp (amp 12) = -12 := by
  sorry

end NUMINAMATH_CALUDE_amp_pamp_theorem_l2858_285872


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l2858_285883

theorem trader_gain_percentage (num_sold : ℕ) (num_gain : ℕ) (h1 : num_sold = 88) (h2 : num_gain = 22) :
  (num_gain : ℚ) / num_sold * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l2858_285883


namespace NUMINAMATH_CALUDE_board_number_problem_l2858_285833

theorem board_number_problem (x : ℕ) : 
  x > 0 ∧ x < 2022 ∧ 
  (∀ n : ℕ, n ≤ 10 → (2022 + x) % (2^n) = 0) →
  x = 998 := by
sorry

end NUMINAMATH_CALUDE_board_number_problem_l2858_285833


namespace NUMINAMATH_CALUDE_prime_triple_problem_l2858_285829

theorem prime_triple_problem (p q r : ℕ) : 
  Prime p → Prime q → Prime r →
  5 ≤ p → p < q → q < r →
  2 * p^2 - r^2 ≥ 49 →
  2 * q^2 - r^2 ≤ 193 →
  p = 17 ∧ q = 19 ∧ r = 23 :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_problem_l2858_285829


namespace NUMINAMATH_CALUDE_at_least_one_perpendicular_l2858_285893

structure GeometricSpace where
  Plane : Type
  Line : Type
  perpendicular_planes : Plane → Plane → Prop
  line_in_plane : Line → Plane → Prop
  perpendicular_lines : Line → Line → Prop
  line_perpendicular_to_plane : Line → Plane → Prop

variable (S : GeometricSpace)

theorem at_least_one_perpendicular
  (α β : S.Plane) (a b : S.Line)
  (h1 : S.perpendicular_planes α β)
  (h2 : S.line_in_plane a α)
  (h3 : S.line_in_plane b β)
  (h4 : S.perpendicular_lines a b) :
  S.line_perpendicular_to_plane a β ∨ S.line_perpendicular_to_plane b α :=
sorry

end NUMINAMATH_CALUDE_at_least_one_perpendicular_l2858_285893


namespace NUMINAMATH_CALUDE_servings_per_jar_l2858_285896

/-- The number of servings of peanut butter consumed per day -/
def servings_per_day : ℕ := 2

/-- The number of days the peanut butter should last -/
def days : ℕ := 30

/-- The number of jars needed to last for the given number of days -/
def jars : ℕ := 4

/-- Theorem stating that each jar contains 15 servings of peanut butter -/
theorem servings_per_jar : 
  (servings_per_day * days) / jars = 15 := by sorry

end NUMINAMATH_CALUDE_servings_per_jar_l2858_285896


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l2858_285840

/-- Proves that Vishal invested 10% more than Trishul -/
theorem vishal_investment_percentage (raghu_investment trishul_investment vishal_investment : ℝ) :
  raghu_investment = 2100 →
  trishul_investment = 0.9 * raghu_investment →
  vishal_investment + trishul_investment + raghu_investment = 6069 →
  (vishal_investment - trishul_investment) / trishul_investment = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l2858_285840


namespace NUMINAMATH_CALUDE_shuttlecock_weight_probability_l2858_285866

/-- The probability that a shuttlecock's weight is less than 4.8 g -/
def prob_less_4_8 : ℝ := 0.3

/-- The probability that a shuttlecock's weight is not greater than 4.85 g -/
def prob_not_greater_4_85 : ℝ := 0.32

/-- The probability that a shuttlecock's weight is within the range [4.8, 4.85] g -/
def prob_range_4_8_to_4_85 : ℝ := prob_not_greater_4_85 - prob_less_4_8

theorem shuttlecock_weight_probability :
  prob_range_4_8_to_4_85 = 0.02 := by sorry

end NUMINAMATH_CALUDE_shuttlecock_weight_probability_l2858_285866


namespace NUMINAMATH_CALUDE_min_voters_for_tall_victory_l2858_285821

/-- Represents the voting structure and outcome of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : ℕ
  num_districts : ℕ
  num_sections_per_district : ℕ
  num_voters_per_section : ℕ
  winner_name : String

/-- Calculates the minimum number of voters required for the winner to secure victory -/
def min_voters_for_victory (contest : GiraffeContest) : ℕ :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win_per_district := contest.num_sections_per_district / 2 + 1
  let voters_to_win_per_section := contest.num_voters_per_section / 2 + 1
  districts_to_win * sections_to_win_per_district * voters_to_win_per_section

/-- The main theorem stating the minimum number of voters required for Tall to win -/
theorem min_voters_for_tall_victory (contest : GiraffeContest)
  (h1 : contest.total_voters = 105)
  (h2 : contest.num_districts = 5)
  (h3 : contest.num_sections_per_district = 7)
  (h4 : contest.num_voters_per_section = 3)
  (h5 : contest.winner_name = "Tall")
  : min_voters_for_victory contest = 24 := by
  sorry

#eval min_voters_for_victory {
  total_voters := 105,
  num_districts := 5,
  num_sections_per_district := 7,
  num_voters_per_section := 3,
  winner_name := "Tall"
}

end NUMINAMATH_CALUDE_min_voters_for_tall_victory_l2858_285821


namespace NUMINAMATH_CALUDE_cos_4theta_l2858_285820

theorem cos_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.cos (4 * θ) = -287 / 625 := by
  sorry

end NUMINAMATH_CALUDE_cos_4theta_l2858_285820


namespace NUMINAMATH_CALUDE_approximate_fish_population_l2858_285860

/-- Represents the fish population in a pond with tagging and recapture. -/
structure FishPopulation where
  total : ℕ  -- Total number of fish in the pond
  tagged : ℕ  -- Number of fish tagged in the first catch
  recaptured : ℕ  -- Number of fish recaptured in the second catch
  tagged_recaptured : ℕ  -- Number of tagged fish in the second catch

/-- The conditions of the problem -/
def pond_conditions : FishPopulation := {
  total := 0,  -- Unknown, to be determined
  tagged := 50,
  recaptured := 50,
  tagged_recaptured := 5
}

/-- Theorem stating the approximate number of fish in the pond -/
theorem approximate_fish_population (p : FishPopulation) 
  (h1 : p.tagged = pond_conditions.tagged)
  (h2 : p.recaptured = pond_conditions.recaptured)
  (h3 : p.tagged_recaptured = pond_conditions.tagged_recaptured)
  (h4 : p.tagged_recaptured / p.recaptured = p.tagged / p.total) :
  p.total = 500 := by
  sorry


end NUMINAMATH_CALUDE_approximate_fish_population_l2858_285860


namespace NUMINAMATH_CALUDE_function_extremum_l2858_285892

/-- The function f(x) = (x-2)e^x has a minimum value of -e and no maximum value -/
theorem function_extremum :
  let f : ℝ → ℝ := λ x => (x - 2) * Real.exp x
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x) ∧
  f (Real.log 1) = -Real.exp 1 ∧
  ¬∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max :=
by sorry

end NUMINAMATH_CALUDE_function_extremum_l2858_285892


namespace NUMINAMATH_CALUDE_max_poles_with_distinct_distances_l2858_285878

/-- 
Given a natural number k, this theorem states that the maximum number of poles
that can be painted in k colors, such that all distances between pairs of 
same-colored poles (with no other same-colored pole between them) are different,
is 3k - 1.
-/
theorem max_poles_with_distinct_distances (k : ℕ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_max_poles_with_distinct_distances_l2858_285878


namespace NUMINAMATH_CALUDE_seventh_grade_class_size_l2858_285828

theorem seventh_grade_class_size :
  let excellent_chinese : ℕ := 15
  let excellent_math : ℕ := 18
  let excellent_both : ℕ := 8
  let not_excellent : ℕ := 20
  excellent_chinese + excellent_math - excellent_both + not_excellent = 45 := by
  sorry

end NUMINAMATH_CALUDE_seventh_grade_class_size_l2858_285828


namespace NUMINAMATH_CALUDE_base_2_representation_of_125_l2858_285870

theorem base_2_representation_of_125 :
  ∃ (a b c d e f g : Nat),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    125 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_125_l2858_285870


namespace NUMINAMATH_CALUDE_final_concentration_calculation_l2858_285846

/-- Calculates the final concentration of a hydrochloric acid solution after draining and adding a new solution. -/
theorem final_concentration_calculation
  (initial_amount : ℝ)
  (initial_concentration : ℝ)
  (drained_amount : ℝ)
  (added_concentration : ℝ)
  (h1 : initial_amount = 300)
  (h2 : initial_concentration = 0.20)
  (h3 : drained_amount = 25)
  (h4 : added_concentration = 0.80) :
  let initial_acid := initial_amount * initial_concentration
  let removed_acid := drained_amount * initial_concentration
  let added_acid := drained_amount * added_concentration
  let final_acid := initial_acid - removed_acid + added_acid
  let final_amount := initial_amount
  final_acid / final_amount = 0.25 := by sorry

end NUMINAMATH_CALUDE_final_concentration_calculation_l2858_285846


namespace NUMINAMATH_CALUDE_room_width_calculation_l2858_285812

theorem room_width_calculation (length : ℝ) (partial_area : ℝ) (additional_area : ℝ) : 
  length = 11 → 
  partial_area = 16 → 
  additional_area = 149 → 
  (partial_area + additional_area) / length = 15 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2858_285812


namespace NUMINAMATH_CALUDE_school_committee_formation_l2858_285807

def total_people : ℕ := 14
def students : ℕ := 11
def teachers : ℕ := 3
def committee_size : ℕ := 8

theorem school_committee_formation :
  (Nat.choose total_people committee_size) - (Nat.choose students committee_size) = 2838 :=
by sorry

end NUMINAMATH_CALUDE_school_committee_formation_l2858_285807


namespace NUMINAMATH_CALUDE_min_chord_length_l2858_285802

/-- The minimum length of a chord passing through (1,1) in the circle (x-2)^2 + (y-3)^2 = 9 is 4 -/
theorem min_chord_length (x y : ℝ) : 
  let circle := fun (x y : ℝ) => (x - 2)^2 + (y - 3)^2 = 9
  let point := (1, 1)
  let chord_length := fun (a b c d : ℝ) => Real.sqrt ((a - c)^2 + (b - d)^2)
  ∃ (a b c d : ℝ), 
    circle a b ∧ circle c d ∧ 
    (1 - a) * (d - b) = (1 - c) * (b - 1) ∧ 
    (∀ (e f g h : ℝ), circle e f ∧ circle g h ∧ 
      (1 - e) * (h - f) = (1 - g) * (f - 1) → 
      chord_length a b c d ≤ chord_length e f g h) ∧
    chord_length a b c d = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_chord_length_l2858_285802


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2858_285834

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 * k + 1 ∧ p.2^2 = 4 * p.1) → 
  k = -1 ∨ k = 0 ∨ k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2858_285834


namespace NUMINAMATH_CALUDE_board_numbers_l2858_285876

theorem board_numbers (n : ℕ) (h : n = 2014) : 
  ∀ (S : Finset ℤ), 
    S.card = n → 
    (∀ (a b c : ℤ), a ∈ S → b ∈ S → c ∈ S → (a + b + c) / 3 ∈ S) → 
    ∃ (x : ℤ), ∀ (y : ℤ), y ∈ S → y = x :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_l2858_285876


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2858_285836

theorem digit_sum_problem (a b c d : ℕ) 
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h1 : a + c = 10)
  (h2 : b + c + 1 = 10)
  (h3 : a + d + 1 = 10) :
  a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2858_285836


namespace NUMINAMATH_CALUDE_equation_is_parabola_l2858_285884

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation |y-3| = √((x+1)² + (y-1)²) -/
def conicEquation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 1)^2 + (p.y - 1)^2)

/-- Defines a parabola as a set of points satisfying a quadratic equation in x or y -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    (∀ p ∈ S, p.x = a * p.y^2 + b * p.y + c ∨ p.y = a * p.x^2 + b * p.x + c) ∧
    (∀ p, p ∈ S ↔ conicEquation p)

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  ∃ S : Set Point2D, isParabola S :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l2858_285884


namespace NUMINAMATH_CALUDE_william_has_more_money_l2858_285823

def oliver_initial_usd : ℝ := 10 * 20 + 3 * 5
def oliver_uk_pounds : ℝ := 200
def oliver_japan_yen : ℝ := 7000
def pound_to_usd : ℝ := 1.38
def yen_to_usd : ℝ := 0.0091
def oliver_expense_usd : ℝ := 75
def oliver_expense_pounds : ℝ := 55
def oliver_expense_yen : ℝ := 3000

def william_initial_usd : ℝ := 15 * 10 + 4 * 5
def william_europe_euro : ℝ := 250
def william_canada_cad : ℝ := 350
def euro_to_usd : ℝ := 1.18
def cad_to_usd : ℝ := 0.78
def william_expense_usd : ℝ := 20
def william_expense_euro : ℝ := 105
def william_expense_cad : ℝ := 150

theorem william_has_more_money :
  let oliver_remaining := (oliver_initial_usd - oliver_expense_usd) +
                          (oliver_uk_pounds * pound_to_usd - oliver_expense_pounds * pound_to_usd) +
                          (oliver_japan_yen * yen_to_usd - oliver_expense_yen * yen_to_usd)
  let william_remaining := (william_initial_usd - william_expense_usd) +
                           (william_europe_euro * euro_to_usd - william_expense_euro * euro_to_usd) +
                           (william_canada_cad * cad_to_usd - william_expense_cad * cad_to_usd)
  william_remaining - oliver_remaining = 100.6 := by sorry

end NUMINAMATH_CALUDE_william_has_more_money_l2858_285823


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2858_285874

theorem sin_2alpha_value (α : Real) 
  (h : Real.sin (π - α) = -2 * Real.sin (π / 2 + α)) : 
  Real.sin (2 * α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2858_285874


namespace NUMINAMATH_CALUDE_prob_one_to_three_l2858_285806

/-- A random variable with normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  pos : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def prob (X : NormalRV) (a b : ℝ) : ℝ := sorry

/-- The given property of normal distributions -/
axiom normal_prob_property (X : NormalRV) : 
  prob X (X.μ - X.σ) (X.μ + X.σ) = 0.6826

/-- The specific normal distribution N(1, 4) -/
def X : NormalRV := { μ := 1, σ := 2, pos := by norm_num }

/-- The theorem to prove -/
theorem prob_one_to_three : prob X 1 3 = 0.3413 := by sorry

end NUMINAMATH_CALUDE_prob_one_to_three_l2858_285806


namespace NUMINAMATH_CALUDE_convex_polygon_triangulation_l2858_285888

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : ℕ

/-- A triangulation of a polygon -/
structure Triangulation (V : ConvexPolygon) where
  triangles : List (Fin V.vertices × Fin V.vertices × Fin V.vertices)

/-- The number of triangles a vertex is part of in a triangulation -/
def vertexTriangleCount (V : ConvexPolygon) (t : Triangulation V) (v : Fin V.vertices) : ℕ :=
  sorry

/-- Theorem stating the triangulation properties for convex polygons -/
theorem convex_polygon_triangulation (V : ConvexPolygon) :
  (∃ (t : Triangulation V), V.vertices % 3 = 0 →
    ∀ (v : Fin V.vertices), Odd (vertexTriangleCount V t v)) ∧
  (∃ (t : Triangulation V), V.vertices % 3 ≠ 0 →
    ∃ (v1 v2 : Fin V.vertices),
      Even (vertexTriangleCount V t v1) ∧
      Even (vertexTriangleCount V t v2) ∧
      ∀ (v : Fin V.vertices), v ≠ v1 → v ≠ v2 → Odd (vertexTriangleCount V t v)) :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_triangulation_l2858_285888


namespace NUMINAMATH_CALUDE_boat_speed_calculation_l2858_285867

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 15

/-- The speed of the stream -/
def stream_speed : ℝ := 3

/-- The time taken to travel downstream -/
def downstream_time : ℝ := 1

/-- The time taken to travel upstream -/
def upstream_time : ℝ := 1.5

/-- The distance traveled (same for both directions) -/
def distance : ℝ := boat_speed + stream_speed

theorem boat_speed_calculation :
  (distance = (boat_speed + stream_speed) * downstream_time) ∧
  (distance = (boat_speed - stream_speed) * upstream_time) →
  boat_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_calculation_l2858_285867


namespace NUMINAMATH_CALUDE_room_width_is_seven_l2858_285848

/-- The width of a room with specific dimensions and features. -/
def room_width : ℝ :=
  let room_length : ℝ := 10
  let room_height : ℝ := 5
  let door_width : ℝ := 1
  let door_height : ℝ := 3
  let num_doors : ℕ := 2
  let large_window_width : ℝ := 2
  let large_window_height : ℝ := 1.5
  let num_large_windows : ℕ := 1
  let small_window_width : ℝ := 1
  let small_window_height : ℝ := 1.5
  let num_small_windows : ℕ := 2
  let paint_cost_per_sqm : ℝ := 3
  let total_paint_cost : ℝ := 474

  7 -- The actual width value

/-- Theorem stating that the room width is 7 meters. -/
theorem room_width_is_seven : room_width = 7 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_seven_l2858_285848


namespace NUMINAMATH_CALUDE_solution_of_equation_l2858_285849

theorem solution_of_equation : 
  ∃ x : ℝ, 7 * (2 * x - 3) + 4 = -3 * (2 - 5 * x) ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2858_285849


namespace NUMINAMATH_CALUDE_seventh_root_unity_sum_l2858_285871

theorem seventh_root_unity_sum (q : ℂ) (h : q^7 = 1) :
  q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6) = 
    if q = 1 then 3/2 else -2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_sum_l2858_285871


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2858_285819

/-- The value of 'a' for which the circle (x - a)² + (y - 1)² = 16 is tangent to the line 3x + 4y - 5 = 0 -/
theorem circle_tangent_to_line (a : ℝ) (h : a > 0) :
  (∃! p : ℝ × ℝ, (p.1 - a)^2 + (p.2 - 1)^2 = 16 ∧ 3*p.1 + 4*p.2 - 5 = 0) →
  a = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2858_285819


namespace NUMINAMATH_CALUDE_soda_water_ratio_l2858_285822

theorem soda_water_ratio (water soda k : ℕ) : 
  water + soda = 54 →
  soda = k * water - 6 →
  k > 0 →
  soda * 5 = water * 4 := by
sorry

end NUMINAMATH_CALUDE_soda_water_ratio_l2858_285822


namespace NUMINAMATH_CALUDE_volleyball_committee_combinations_l2858_285825

/-- The number of teams in the volleyball league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members sent by the host team to the committee -/
def host_committee_size : ℕ := 4

/-- The number of members sent by each non-host team to the committee -/
def non_host_committee_size : ℕ := 3

/-- The total size of the tournament committee -/
def total_committee_size : ℕ := 16

/-- The number of different committees that can be formed -/
def num_committees : ℕ := 3442073600

theorem volleyball_committee_combinations :
  num_committees = num_teams * (Nat.choose team_size host_committee_size) * 
    (Nat.choose team_size non_host_committee_size)^(num_teams - 1) :=
by sorry

end NUMINAMATH_CALUDE_volleyball_committee_combinations_l2858_285825


namespace NUMINAMATH_CALUDE_lemonade_sales_ratio_l2858_285839

theorem lemonade_sales_ratio :
  ∀ (katya_sales ricky_sales tina_sales : ℕ),
    katya_sales = 8 →
    ricky_sales = 9 →
    tina_sales = katya_sales + 26 →
    ∃ (m : ℕ), tina_sales = m * (katya_sales + ricky_sales) →
    (tina_sales : ℚ) / (katya_sales + ricky_sales : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sales_ratio_l2858_285839


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2858_285886

theorem cookie_jar_problem :
  ∃ C : ℕ, (C - 1 = (C + 5) / 2) ∧ (C = 7) :=
by sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2858_285886


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2858_285897

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2858_285897


namespace NUMINAMATH_CALUDE_lanas_concert_expense_l2858_285895

def ticket_price : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lanas_concert_expense :
  (tickets_for_friends + extra_tickets) * ticket_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_lanas_concert_expense_l2858_285895


namespace NUMINAMATH_CALUDE_polyline_intersection_theorem_l2858_285843

/-- A polyline is represented as a list of points -/
def Polyline := List (ℝ × ℝ)

/-- Calculate the length of a polyline -/
def polylineLength (p : Polyline) : ℝ :=
  sorry

/-- Check if a polyline is inside a unit square -/
def insideUnitSquare (p : Polyline) : Prop :=
  sorry

/-- Check if a polyline intersects itself -/
def selfIntersecting (p : Polyline) : Prop :=
  sorry

/-- Count the number of intersections between a line and a polyline -/
def intersectionCount (line : ℝ × ℝ → Prop) (p : Polyline) : ℕ :=
  sorry

/-- The main theorem -/
theorem polyline_intersection_theorem (p : Polyline) 
  (h1 : insideUnitSquare p)
  (h2 : ¬selfIntersecting p)
  (h3 : polylineLength p > 1000) :
  ∃ (line : ℝ × ℝ → Prop), 
    (∀ x y, line (x, y) ↔ (x = 0 ∨ x = 1 ∨ y = 0 ∨ y = 1)) ∧
    intersectionCount line p ≥ 501 :=
  sorry

end NUMINAMATH_CALUDE_polyline_intersection_theorem_l2858_285843


namespace NUMINAMATH_CALUDE_sum_of_u_and_v_l2858_285844

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 5 * u - 3 * v = 26)
  (eq2 : 3 * u + 5 * v = -19) : 
  u + v = -101 / 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_u_and_v_l2858_285844


namespace NUMINAMATH_CALUDE_smallest_base_for_120_l2858_285842

theorem smallest_base_for_120 : ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 120 ∧ 120 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 120 → 120 ≥ x^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_120_l2858_285842


namespace NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l2858_285869

theorem walk_a_thon_miles_difference 
  (last_year_rate : ℝ) 
  (this_year_rate : ℝ) 
  (last_year_amount : ℝ) : 
  last_year_rate = 4 →
  this_year_rate = 2.75 →
  last_year_amount = 44 →
  (last_year_amount / this_year_rate) - (last_year_amount / last_year_rate) = 5 :=
by sorry

end NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l2858_285869


namespace NUMINAMATH_CALUDE_ages_of_linda_and_jane_l2858_285889

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem ages_of_linda_and_jane (jane_age linda_age kevin_age : ℕ) :
  linda_age = 2 * jane_age + 3 →
  is_prime (linda_age - jane_age) →
  linda_age + jane_age + 10 = kevin_age + 5 →
  kevin_age = 4 * jane_age →
  jane_age = 8 ∧ linda_age = 19 :=
by sorry

end NUMINAMATH_CALUDE_ages_of_linda_and_jane_l2858_285889


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2858_285873

theorem geometric_sequence_product (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ = 8/3 ∧ a₅ = 27/2 ∧ 
  (∃ q : ℝ, q ≠ 0 ∧ a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q) →
  |a₂ * a₃ * a₄| = 216 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2858_285873


namespace NUMINAMATH_CALUDE_smallest_odd_k_for_cubic_irreducibility_l2858_285818

/-- A cubic polynomial with integer coefficients -/
def CubicPolynomial := ℤ → ℤ

/-- Checks if a number is prime -/
def isPrime (n : ℤ) : Prop := sorry

/-- Checks if a polynomial is irreducible over ℤ -/
def isIrreducible (f : CubicPolynomial) : Prop := sorry

/-- The main theorem -/
theorem smallest_odd_k_for_cubic_irreducibility : 
  ∃ (k : ℕ), k % 2 = 1 ∧
  (∀ (j : ℕ), j % 2 = 1 → j < k →
    ∃ (f : CubicPolynomial),
      (∃ (S : Finset ℤ), S.card = j ∧ ∀ n ∈ S, isPrime (|f n|)) ∧
      ¬isIrreducible f) ∧
  (∀ (f : CubicPolynomial),
    (∃ (S : Finset ℤ), S.card = k ∧ ∀ n ∈ S, isPrime (|f n|)) →
    isIrreducible f) ∧
  k = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_k_for_cubic_irreducibility_l2858_285818


namespace NUMINAMATH_CALUDE_problem_solution_l2858_285864

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∀ x, x ∈ A ∩ B (-4) ↔ 1/2 ≤ x ∧ x < 2) ∧
  (∀ x, x ∈ A ∪ B (-4) ↔ -2 < x ∧ x ≤ 3) ∧
  (∀ a, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2858_285864


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l2858_285853

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (I)
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for part (II)
theorem intersection_complement_A_B : (Set.compl A) ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l2858_285853


namespace NUMINAMATH_CALUDE_probability_is_31_over_473_l2858_285805

/-- Represents a standard deck of cards --/
def StandardDeck : ℕ := 52

/-- Number of cards per rank in a standard deck --/
def CardsPerRank : ℕ := 4

/-- Number of pairs removed (two pairs of Aces and two pairs of Kings) --/
def PairsRemoved : ℕ := 2

/-- Number of ranks affected by pair removal --/
def RanksAffected : ℕ := 2

/-- Number of unaffected ranks (from Two to Queen) --/
def UnaffectedRanks : ℕ := 11

/-- Calculates the probability of selecting a pair from the modified deck --/
def probability_of_pair (deck : ℕ) (cards_per_rank : ℕ) (pairs_removed : ℕ) (ranks_affected : ℕ) (unaffected_ranks : ℕ) : ℚ :=
  let remaining_cards := deck - 2 * pairs_removed * cards_per_rank
  let total_combinations := remaining_cards.choose 2
  let affected_pairs := ranks_affected
  let unaffected_pairs := unaffected_ranks * (cards_per_rank.choose 2)
  let favorable_outcomes := affected_pairs + unaffected_pairs
  ↑favorable_outcomes / ↑total_combinations

theorem probability_is_31_over_473 :
  probability_of_pair StandardDeck CardsPerRank PairsRemoved RanksAffected UnaffectedRanks = 31 / 473 :=
sorry

end NUMINAMATH_CALUDE_probability_is_31_over_473_l2858_285805


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_of_squares_l2858_285827

theorem consecutive_even_numbers_sum_of_squares (n : ℤ) : 
  (∀ k : ℕ, k < 6 → 2 ∣ (n + 2 * k)) → 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) →
  (n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_of_squares_l2858_285827


namespace NUMINAMATH_CALUDE_school_band_seats_l2858_285841

/-- Represents the number of seats needed for the school band --/
def total_seats (flute trumpet trombone drummer clarinet french_horn : ℕ) : ℕ :=
  flute + trumpet + trombone + drummer + clarinet + french_horn

/-- Theorem stating the total number of seats needed for the school band --/
theorem school_band_seats :
  ∃ (flute trumpet trombone drummer clarinet french_horn : ℕ),
    flute = 5 ∧
    trumpet = 3 * flute ∧
    trombone = trumpet - 8 ∧
    drummer = trombone + 11 ∧
    clarinet = 2 * flute ∧
    french_horn = trombone + 3 ∧
    total_seats flute trumpet trombone drummer clarinet french_horn = 65 := by
  sorry

end NUMINAMATH_CALUDE_school_band_seats_l2858_285841


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2858_285804

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {3, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2858_285804


namespace NUMINAMATH_CALUDE_wheel_center_travel_distance_l2858_285809

/-- The distance traveled by the center of a wheel rolling one complete revolution -/
theorem wheel_center_travel_distance (r : ℝ) (h : r = 1) :
  let circumference := 2 * Real.pi * r
  circumference = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_wheel_center_travel_distance_l2858_285809


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l2858_285863

theorem cube_root_125_times_fourth_root_256_times_sqrt_16 : 
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l2858_285863


namespace NUMINAMATH_CALUDE_men_added_to_group_l2858_285817

theorem men_added_to_group (original_days : ℕ) (new_days : ℕ) (new_men : ℕ) 
  (h1 : original_days = 24)
  (h2 : new_days = 16)
  (h3 : new_men = 12)
  (h4 : ∃ (original_men : ℕ), original_men * original_days = new_men * new_days) :
  new_men - (new_men * new_days / original_days) = 4 := by
  sorry

end NUMINAMATH_CALUDE_men_added_to_group_l2858_285817


namespace NUMINAMATH_CALUDE_chicken_admission_problem_l2858_285845

theorem chicken_admission_problem :
  let n : ℕ := 4  -- Total number of chickens
  let k : ℕ := 2  -- Number of chickens to be admitted to evening department
  Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_admission_problem_l2858_285845


namespace NUMINAMATH_CALUDE_set_M_characterization_inequality_holds_complement_M_l2858_285875

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | f x > 2}

-- Theorem 1
theorem set_M_characterization : M = {x : ℝ | x < (1/2) ∨ x > (5/2)} := by sorry

-- Theorem 2
theorem inequality_holds_complement_M (a b x : ℝ) (ha : a ≠ 0) (hx : (1/2) ≤ x ∧ x ≤ (5/2)) :
  |a + b| + |a - b| ≥ |a| * (f x) := by sorry

end NUMINAMATH_CALUDE_set_M_characterization_inequality_holds_complement_M_l2858_285875


namespace NUMINAMATH_CALUDE_alarm_set_time_l2858_285810

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Converts time to total minutes -/
def Time.toMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

/-- The rate at which the faster watch gains time (in minutes per hour) -/
def gainRate : ℕ := 2

/-- The time shown on the faster watch when the alarm rings -/
def fasterWatchTime : Time := ⟨4, 12, by norm_num⟩

/-- The correct time when the alarm rings -/
def correctTime : Time := ⟨4, 0, by norm_num⟩

/-- Calculates the number of hours passed based on the time difference and gain rate -/
def hoursPassed (timeDiff : ℕ) (rate : ℕ) : ℚ := timeDiff / rate

theorem alarm_set_time :
  let timeDiff := fasterWatchTime.toMinutes - correctTime.toMinutes
  let hours := hoursPassed timeDiff gainRate
  (correctTime.hours - hours.floor : ℤ) = 22 := by sorry

end NUMINAMATH_CALUDE_alarm_set_time_l2858_285810


namespace NUMINAMATH_CALUDE_no_positive_solution_l2858_285890

theorem no_positive_solution :
  ¬ ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a * d^2 + b * d - c = 0) ∧
    (Real.sqrt a * d + Real.sqrt b * Real.sqrt d - Real.sqrt c = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solution_l2858_285890


namespace NUMINAMATH_CALUDE_max_intersection_area_point_l2858_285832

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Calculates the ratio of two line segments -/
def ratio (P Q R : Point) : ℝ :=
  sorry

/-- Calculates the area of a triangle -/
def triangleArea (P Q R : Point) : ℝ :=
  sorry

/-- Calculates the area of the intersection of two triangles -/
def intersectionArea (P Q R S T U : Point) : ℝ :=
  sorry

/-- Theorem: The point M on BC that maximizes the intersection area satisfies BM/MC = AK/KD -/
theorem max_intersection_area_point (ABCD : Trapezoid) (K : Point) :
  ∃ (M : Point),
    (∀ (M' : Point), intersectionArea ABCD.A ABCD.B ABCD.C ABCD.D K M ≥ 
                      intersectionArea ABCD.A ABCD.B ABCD.C ABCD.D K M') ↔
    (ratio ABCD.B M ABCD.C = ratio ABCD.A K ABCD.D) :=
  sorry

end NUMINAMATH_CALUDE_max_intersection_area_point_l2858_285832


namespace NUMINAMATH_CALUDE_boat_speed_proof_l2858_285850

/-- The speed of the stream in km/h -/
def stream_speed : ℝ := 8

/-- The distance covered downstream in km -/
def downstream_distance : ℝ := 64

/-- The distance covered upstream in km -/
def upstream_distance : ℝ := 32

/-- The speed of the boat in still water in km/h -/
def boat_speed : ℝ := 24

theorem boat_speed_proof :
  (downstream_distance / (boat_speed + stream_speed) = 
   upstream_distance / (boat_speed - stream_speed)) ∧
  (boat_speed > stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l2858_285850


namespace NUMINAMATH_CALUDE_set_operations_l2858_285856

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∪ (B ∩ C) = A) ∧
  (A ∩ (A \ (B ∩ C)) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2858_285856


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l2858_285808

theorem right_triangle_area_perimeter_relation : 
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c^2 = a^2 + b^2 ∧
    (a * b : ℚ) / 2 = 4 * (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l2858_285808


namespace NUMINAMATH_CALUDE_probability_in_word_l2858_285865

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The word we're analyzing -/
def word : String := "MATHEMATICS"

/-- The set of unique letters in the word -/
def unique_letters : Finset Char := word.toList.toFinset

/-- The probability of selecting a letter from the alphabet that appears in the word -/
theorem probability_in_word : 
  (unique_letters.card : ℚ) / alphabet_size = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_in_word_l2858_285865


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2858_285854

theorem pie_eating_contest (adam bill sierra : ℕ) : 
  adam = bill + 3 →
  sierra = 2 * bill →
  sierra = 12 →
  adam + bill + sierra = 27 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2858_285854


namespace NUMINAMATH_CALUDE_rulers_remaining_l2858_285894

theorem rulers_remaining (initial_rulers : ℕ) (rulers_taken : ℕ) : 
  initial_rulers = 46 → rulers_taken = 25 → initial_rulers - rulers_taken = 21 :=
by sorry

end NUMINAMATH_CALUDE_rulers_remaining_l2858_285894


namespace NUMINAMATH_CALUDE_incorrect_expression_l2858_285861

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : 
  (x - 2 * y) / y ≠ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l2858_285861


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2858_285801

theorem necessary_but_not_sufficient (x : ℝ) :
  (∀ x, (abs x = -x → x^2 ≥ -x)) ∧
  (∃ x, x^2 ≥ -x ∧ abs x ≠ -x) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2858_285801


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2858_285803

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2858_285803


namespace NUMINAMATH_CALUDE_excavation_volume_scientific_notation_l2858_285835

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem excavation_volume_scientific_notation :
  toScientificNotation 632000 = ScientificNotation.mk 6.32 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_excavation_volume_scientific_notation_l2858_285835


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_22_l2858_285857

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  angle_between_diagonals : ℝ

/-- Calculate the perimeter of a trapezoid with the given properties -/
def perimeter (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with the given properties has a perimeter of 22 -/
theorem trapezoid_perimeter_is_22 (t : Trapezoid) 
    (h1 : t.base1 = 3)
    (h2 : t.base2 = 5)
    (h3 : t.diagonal = 8)
    (h4 : t.angle_between_diagonals = 60) :
    perimeter t = 22 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_22_l2858_285857
