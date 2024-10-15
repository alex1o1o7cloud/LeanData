import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l20_2075

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l20_2075


namespace NUMINAMATH_CALUDE_count_permutations_2007_l20_2023

/-- The number of permutations of integers 1 to n with exactly one descent -/
def permutations_with_one_descent (n : ℕ) : ℕ :=
  2^n - (n + 1)

/-- The theorem to be proved -/
theorem count_permutations_2007 :
  permutations_with_one_descent 2007 = 2^3 * (2^2004 - 251) := by
  sorry

end NUMINAMATH_CALUDE_count_permutations_2007_l20_2023


namespace NUMINAMATH_CALUDE_square_a_minus_2b_l20_2067

theorem square_a_minus_2b (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2*b)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_a_minus_2b_l20_2067


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l20_2050

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (3 * X ^ 2 - 20 * X + 68 : Polynomial ℚ) = (X - 4) * q + 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l20_2050


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l20_2009

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 27 = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l20_2009


namespace NUMINAMATH_CALUDE_lawn_maintenance_time_l20_2085

theorem lawn_maintenance_time (mow_time fertilize_time total_time : ℕ) : 
  mow_time = 40 →
  fertilize_time = 2 * mow_time →
  total_time = mow_time + fertilize_time →
  total_time = 120 := by
sorry

end NUMINAMATH_CALUDE_lawn_maintenance_time_l20_2085


namespace NUMINAMATH_CALUDE_function_relation_implies_a_half_l20_2053

/-- Given two functions f and g defined on ℝ satisfying certain conditions, prove that a = 1/2 -/
theorem function_relation_implies_a_half :
  ∀ (f g : ℝ → ℝ) (a : ℝ),
    (∀ x, f x = a^x * g x) →
    (a > 0) →
    (a ≠ 1) →
    (∀ x, g x ≠ 0 → f x * (deriv g x) > (deriv f x) * g x) →
    (f 1 / g 1 + f (-1) / g (-1) = 5/2) →
    a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_relation_implies_a_half_l20_2053


namespace NUMINAMATH_CALUDE_product_loss_percentage_l20_2057

/-- Proves the percentage loss of a product given specific selling prices and gain percentages --/
theorem product_loss_percentage 
  (cp : ℝ) -- Cost price
  (sp_gain : ℝ) -- Selling price with gain
  (sp_loss : ℝ) -- Selling price with loss
  (gain_percent : ℝ) -- Gain percentage
  (h1 : sp_gain = cp * (1 + gain_percent / 100)) -- Condition for selling price with gain
  (h2 : sp_gain = 168) -- Given selling price with gain
  (h3 : gain_percent = 20) -- Given gain percentage
  (h4 : sp_loss = 119) -- Given selling price with loss
  : (cp - sp_loss) / cp * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_loss_percentage_l20_2057


namespace NUMINAMATH_CALUDE_students_liking_sports_l20_2008

theorem students_liking_sports (B C : Finset Nat) 
  (hB : B.card = 10)
  (hC : C.card = 8)
  (hBC : (B ∩ C).card = 4) :
  (B ∪ C).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l20_2008


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l20_2024

theorem fitness_center_membership_ratio 
  (f m : ℕ) -- f: number of female members, m: number of male members
  (hf : f > 0) -- ensure f is positive
  (hm : m > 0) -- ensure m is positive
  (h_avg : (45 * f + 20 * m) / (f + m) = 25) : -- condition for overall average age
  f / m = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l20_2024


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l20_2042

theorem solve_exponential_equation :
  ∃ x : ℝ, 2^(2*x - 1) = (1/4 : ℝ) ∧ x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l20_2042


namespace NUMINAMATH_CALUDE_problem_1_l20_2055

theorem problem_1 (m n : ℝ) (h1 : m = 2) (h2 : n = 1) : 
  (2*m^2 - 3*m*n + 8) - (5*m*n - 4*m^2 + 8) = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_l20_2055


namespace NUMINAMATH_CALUDE_min_value_theorem_l20_2096

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_bisect : ∀ (x y : ℝ), 2*x + y - 2 = 0 → x^2 + y^2 - 2*a*x - 4*b*y + 1 = 0 → 
    ∃ (x' y' : ℝ), x'^2 + y'^2 - 2*a*x' - 4*b*y' + 1 = 0 ∧ 2*x' + y' - 2 ≠ 0) : 
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 2/a' + 1/(2*b') ≥ 9/2) ∧ 
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 2/a' + 1/(2*b') = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l20_2096


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_y_l20_2047

theorem max_value_of_x_plus_y : 
  ∃ (M : ℝ), M = 4 ∧ 
  ∀ (x y : ℝ), x^2 + y + 3*x - 3 = 0 → x + y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_y_l20_2047


namespace NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_5_l20_2052

def is_abundant (n : ℕ) : Prop :=
  (Finset.sum (Finset.range n) (λ i => if n % (i + 1) = 0 then i + 1 else 0)) > n

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_abundant_not_multiple_of_5 : 
  (∀ m : ℕ, m < 12 → (¬is_abundant m ∨ is_multiple_of_5 m)) ∧
  is_abundant 12 ∧ 
  ¬is_multiple_of_5 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_5_l20_2052


namespace NUMINAMATH_CALUDE_sector_arc_length_l20_2049

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  r = 3 → θ_deg = 150 → l = (5 * π) / 2 → 
  l = r * ((θ_deg * π) / 180) :=
sorry

end NUMINAMATH_CALUDE_sector_arc_length_l20_2049


namespace NUMINAMATH_CALUDE_seed_germination_experiment_l20_2022

theorem seed_germination_experiment (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot2 : ℚ) (total_germination_rate : ℚ)
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot2 = 35 / 100)
  (h4 : total_germination_rate = 32 / 100)
  (h5 : (seeds_plot1 + seeds_plot2) * total_germination_rate =
        seeds_plot1 * (germination_rate_plot1 : ℚ) + seeds_plot2 * germination_rate_plot2) :
  germination_rate_plot1 = 30 / 100 := by
  sorry

#check seed_germination_experiment

end NUMINAMATH_CALUDE_seed_germination_experiment_l20_2022


namespace NUMINAMATH_CALUDE_h_of_three_equals_five_l20_2086

-- Define the function h
def h (x : ℝ) : ℝ := 2*(x-2) + 3

-- State the theorem
theorem h_of_three_equals_five : h 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_h_of_three_equals_five_l20_2086


namespace NUMINAMATH_CALUDE_electron_transfer_for_N2_production_l20_2018

-- Define the chemical elements and compounds
def Zn : Type := Unit
def H : Type := Unit
def N : Type := Unit
def O : Type := Unit
def HNO3 : Type := Unit
def NH4NO3 : Type := Unit
def H2O : Type := Unit
def ZnNO3_2 : Type := Unit

-- Define the reaction
def reaction : Type := Unit

-- Define Avogadro's constant
def Na : ℕ := sorry

-- Define the electron transfer function
def electron_transfer (r : reaction) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem electron_transfer_for_N2_production (r : reaction) :
  electron_transfer r 1 = 5 * Na := by sorry

end NUMINAMATH_CALUDE_electron_transfer_for_N2_production_l20_2018


namespace NUMINAMATH_CALUDE_no_common_root_l20_2014

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x : ℝ, (x^2 + b*x + c = 0) ∧ (x^2 + a*x + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_l20_2014


namespace NUMINAMATH_CALUDE_clock_strike_problem_l20_2039

/-- Represents a clock that strikes at regular intervals. -/
structure Clock where
  interval : ℕ

/-- Calculates the time of the last strike given two clocks and total strikes. -/
def lastStrikeTime (clock1 clock2 : Clock) (totalStrikes : ℕ) : ℕ :=
  sorry

/-- Calculates the time between first and last strikes. -/
def timeBetweenStrikes (clock1 clock2 : Clock) (totalStrikes : ℕ) : ℕ :=
  sorry

theorem clock_strike_problem :
  let clock1 : Clock := { interval := 2 }
  let clock2 : Clock := { interval := 3 }
  let totalStrikes : ℕ := 13
  timeBetweenStrikes clock1 clock2 totalStrikes = 18 :=
by sorry

end NUMINAMATH_CALUDE_clock_strike_problem_l20_2039


namespace NUMINAMATH_CALUDE_initial_sweets_count_proof_initial_sweets_count_l20_2087

theorem initial_sweets_count : ℕ → Prop :=
  fun x => 
    (x / 2 + 4 + 7 = x) → 
    x = 22

-- The proof is omitted
theorem proof_initial_sweets_count : initial_sweets_count 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_sweets_count_proof_initial_sweets_count_l20_2087


namespace NUMINAMATH_CALUDE_largest_logarithm_l20_2073

theorem largest_logarithm (h : 0 < Real.log 2 ∧ Real.log 2 < 1) :
  2 * Real.log 2 > Real.log 2 ∧ 
  Real.log 2 > (Real.log 2)^2 ∧ 
  (Real.log 2)^2 > Real.log (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_logarithm_l20_2073


namespace NUMINAMATH_CALUDE_slope_y_intercept_ratio_l20_2046

/-- A line in the coordinate plane with slope m, y-intercept b, and x-intercept 2 -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  x_intercept_eq_two : m * 2 + b = 0  -- condition for x-intercept = 2

/-- The slope is some fraction of the y-intercept -/
def slope_fraction (k : Line) (c : ℝ) : Prop :=
  k.m = c * k.b

theorem slope_y_intercept_ratio (k : Line) :
  ∃ c : ℝ, slope_fraction k c ∧ c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_slope_y_intercept_ratio_l20_2046


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_tangent_points_coordinates_l20_2011

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

-- Theorem to prove the y-coordinates
theorem y_coordinates (x : ℝ) :
  (x = 1 ∨ x = -1) → (f x = 0 ∨ f x = -4) :=
by sorry

-- Main theorem combining the above results
theorem tangent_points_coordinates :
  ∃ (x y : ℝ), (f' x = 4 ∧ f x = y) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_tangent_points_coordinates_l20_2011


namespace NUMINAMATH_CALUDE_remainder_3005_div_99_l20_2021

theorem remainder_3005_div_99 : 3005 % 99 = 35 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3005_div_99_l20_2021


namespace NUMINAMATH_CALUDE_margarets_mean_score_l20_2000

def scores : List ℝ := [86, 88, 91, 93, 95, 97, 99, 100]

theorem margarets_mean_score 
  (h1 : scores.length = 8)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    margaret_scores.length = 4 ∧ 
    cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    cyprian_scores.sum / cyprian_scores.length = 92) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 4 ∧ 
    margaret_scores.sum / margaret_scores.length = 95.25 := by
  sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l20_2000


namespace NUMINAMATH_CALUDE_erased_number_proof_l20_2036

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n ≥ 3 →
  x ≥ 3 →
  x ≤ n →
  (n * (n + 1) / 2 - 3 - x) / (n - 2 : ℚ) = 151 / 3 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l20_2036


namespace NUMINAMATH_CALUDE_money_distribution_l20_2056

/-- Given three people A, B, and C with a total amount of money,
    prove that B and C together have a specific amount. -/
theorem money_distribution (total A_C C B_C : ℕ) : 
  total = 1000 →
  A_C = 700 →
  C = 300 →
  B_C = total - (A_C - C) →
  B_C = 600 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l20_2056


namespace NUMINAMATH_CALUDE_four_points_plane_count_l20_2032

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to count the number of planes determined by four points
def countPlanesFromFourPoints (A B C D : Point3D) : Nat :=
  sorry

-- Theorem statement
theorem four_points_plane_count (A B C D : Point3D) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) : 
  countPlanesFromFourPoints A B C D = 1 ∨ countPlanesFromFourPoints A B C D = 4 :=
sorry

end NUMINAMATH_CALUDE_four_points_plane_count_l20_2032


namespace NUMINAMATH_CALUDE_complement_of_union_l20_2066

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {4, 5, 6}

theorem complement_of_union : 
  U \ (A ∪ B) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l20_2066


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l20_2099

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l20_2099


namespace NUMINAMATH_CALUDE_jackie_sleep_time_l20_2084

theorem jackie_sleep_time (total_hours work_hours exercise_hours free_hours : ℕ) 
  (h1 : total_hours = 24)
  (h2 : work_hours = 8)
  (h3 : exercise_hours = 3)
  (h4 : free_hours = 5) :
  total_hours - (work_hours + exercise_hours + free_hours) = 8 := by
  sorry

end NUMINAMATH_CALUDE_jackie_sleep_time_l20_2084


namespace NUMINAMATH_CALUDE_equation_describes_cone_l20_2059

/-- Spherical coordinates -/
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in spherical coordinates -/
def IsCone (c : ℝ) (f : SphericalCoordinates → Prop) : Prop :=
  ∀ p : SphericalCoordinates, f p ↔ p.ρ = c * Real.sin p.φ

/-- The main theorem: the equation ρ = c * sin φ describes a cone -/
theorem equation_describes_cone (c : ℝ) (hc : c > 0) :
  IsCone c (fun p => p.ρ = c * Real.sin p.φ) :=
sorry

end NUMINAMATH_CALUDE_equation_describes_cone_l20_2059


namespace NUMINAMATH_CALUDE_upstream_distance_l20_2012

theorem upstream_distance
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h1 : boat_speed = 20)
  (h2 : downstream_distance = 96)
  (h3 : downstream_time = 3)
  (h4 : upstream_time = 11)
  : ∃ (upstream_distance : ℝ), upstream_distance = 88 :=
by
  sorry

#check upstream_distance

end NUMINAMATH_CALUDE_upstream_distance_l20_2012


namespace NUMINAMATH_CALUDE_sixth_term_is_32_l20_2025

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  (a 2 + a 3) / (a 1 + a 2) = 2 ∧ a 4 = 8

/-- Theorem stating that for a geometric sequence satisfying the given conditions, the 6th term is 32 -/
theorem sixth_term_is_32 (a : ℕ → ℝ) 
    (h_geo : is_geometric_sequence a) 
    (h_cond : sequence_conditions a) : 
  a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_32_l20_2025


namespace NUMINAMATH_CALUDE_factor_81_minus_4y4_l20_2072

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_4y4_l20_2072


namespace NUMINAMATH_CALUDE_interest_rate_problem_l20_2088

/-- Calculates the amount after simple interest is applied -/
def amountAfterSimpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem interest_rate_problem (originalRate : ℝ) :
  let principal : ℝ := 1000
  let time : ℝ := 5
  let increasedRate : ℝ := originalRate + 0.05
  amountAfterSimpleInterest principal increasedRate time = 1750 →
  amountAfterSimpleInterest principal originalRate time = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l20_2088


namespace NUMINAMATH_CALUDE_sarah_investment_l20_2005

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem sarah_investment :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 3046.28| < ε :=
sorry

end NUMINAMATH_CALUDE_sarah_investment_l20_2005


namespace NUMINAMATH_CALUDE_complete_square_plus_integer_l20_2069

theorem complete_square_plus_integer :
  ∃ (k : ℤ) (b : ℝ), ∀ (x : ℝ), x^2 + 14*x + 60 = (x + b)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_complete_square_plus_integer_l20_2069


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l20_2061

theorem reciprocal_comparison : 
  (let numbers := [-1/2, -3, 1/3, 3, 3/2]
   ∀ x ∈ numbers, x < 1/x ↔ (x = -3 ∨ x = 1/3)) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l20_2061


namespace NUMINAMATH_CALUDE_recurrence_relation_solution_l20_2030

def a (n : ℕ) : ℤ := 2 * 4^n - 2*n + 2
def b (n : ℕ) : ℤ := 2 * 4^n + 2*n - 2

theorem recurrence_relation_solution :
  (∀ n : ℕ, a (n + 1) = 3 * a n + b n - 4) ∧
  (∀ n : ℕ, b (n + 1) = 2 * a n + 2 * b n + 2) ∧
  a 0 = 4 ∧
  b 0 = 0 := by sorry

end NUMINAMATH_CALUDE_recurrence_relation_solution_l20_2030


namespace NUMINAMATH_CALUDE_correct_result_l20_2037

theorem correct_result (x : ℝ) : (-1.25 * x) - 0.25 = 1.25 * x → -1.25 * x = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l20_2037


namespace NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l20_2074

theorem x_zero_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0) ∧
  ¬(∀ x y : ℝ, x = 0 → x^2 + y^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l20_2074


namespace NUMINAMATH_CALUDE_max_value_expression_l20_2091

/-- For positive real numbers a and b, and angle θ where 0 ≤ θ ≤ π/2,
    the maximum value of 2(a - x)(x + cos(θ)√(x^2 + b^2)) is a^2 + cos^2(θ)b^2 -/
theorem max_value_expression (a b : ℝ) (θ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hθ : 0 ≤ θ ∧ θ ≤ π/2) :
  (⨆ x, 2 * (a - x) * (x + Real.cos θ * Real.sqrt (x^2 + b^2))) = a^2 + Real.cos θ^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l20_2091


namespace NUMINAMATH_CALUDE_vertex_of_parabola_l20_2095

/-- The function f(x) = 3(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola defined by f -/
def vertex : ℝ × ℝ := (1, 2)

theorem vertex_of_parabola :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_parabola_l20_2095


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l20_2026

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l20_2026


namespace NUMINAMATH_CALUDE_parabola_focus_value_hyperbola_standard_equation_l20_2064

-- Problem 1
theorem parabola_focus_value (p : ℝ) (h1 : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x - y - 4 = 0 ∧ x = p ∧ y = 0) →
  p = 2 := by sorry

-- Problem 2
theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = 3 / 4) ∧ 
  (a^2 / (a^2 + b^2).sqrt = 16 / 5) →
  ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_value_hyperbola_standard_equation_l20_2064


namespace NUMINAMATH_CALUDE_triangle_side_length_l20_2060

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 2 * π / 3) -- 120° in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) -- positive side lengths
  (h6 : a / (Real.sin A) = b / (Real.sin B)) -- sine rule
  (h7 : b / (Real.sin B) = c / (Real.sin C)) -- sine rule
  : a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l20_2060


namespace NUMINAMATH_CALUDE_train_passing_time_l20_2062

/-- Calculates the time for a train to pass a person moving in the opposite direction -/
theorem train_passing_time 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (person_speed : ℝ) 
  (h1 : train_length = 110) 
  (h2 : train_speed = 65) 
  (h3 : person_speed = 7) : 
  (train_length / ((train_speed + person_speed) * (5/18))) = 5.5 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l20_2062


namespace NUMINAMATH_CALUDE_distance_from_pole_to_line_l20_2077

/-- Given a line with polar equation ρ sin(θ + π/4) = 1, 
    the distance from the pole to this line is 1. -/
theorem distance_from_pole_to_line (ρ θ : ℝ) : 
  ρ * Real.sin (θ + π/4) = 1 → 
  (∃ d : ℝ, d = 1 ∧ d = abs (2) / Real.sqrt (2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_pole_to_line_l20_2077


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l20_2003

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 2/y ≥ 1/a + 2/b) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l20_2003


namespace NUMINAMATH_CALUDE_min_swaps_for_geese_order_l20_2097

def initial_order : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
def final_order : List ℕ := List.range 20 |>.map (· + 1)

def count_inversions (l : List ℕ) : ℕ :=
  l.foldr (fun x acc => acc + (l.filter (· < x) |>.filter (fun y => l.indexOf y > l.indexOf x) |>.length)) 0

def min_swaps_to_sort (l : List ℕ) : ℕ := count_inversions l

theorem min_swaps_for_geese_order :
  min_swaps_to_sort initial_order = 55 :=
sorry

end NUMINAMATH_CALUDE_min_swaps_for_geese_order_l20_2097


namespace NUMINAMATH_CALUDE_two_thirds_in_M_l20_2019

open Set

-- Define the sets A and B as open intervals
def A : Set ℝ := Ioo (-4) 1
def B : Set ℝ := Ioo (-2) 5

-- Define M as the intersection of A and B
def M : Set ℝ := A ∩ B

-- Theorem statement
theorem two_thirds_in_M : (2/3 : ℝ) ∈ M := by sorry

end NUMINAMATH_CALUDE_two_thirds_in_M_l20_2019


namespace NUMINAMATH_CALUDE_vector_problem_l20_2090

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := (5, 6)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c (y : ℝ) : ℝ × ℝ := (2, y)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Perpendicular vectors have zero dot product -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

/-- Parallel vectors are scalar multiples of each other -/
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), w = (k * v.1, k * v.2)

/-- Main theorem -/
theorem vector_problem :
  ∃ (x y : ℝ),
    perpendicular a (b x) ∧
    parallel a (c y) ∧
    x = -18/5 ∧
    y = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l20_2090


namespace NUMINAMATH_CALUDE_kids_went_home_l20_2038

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℕ) : 
  initial_kids = 22.0 → remaining_kids = 8 → initial_kids - remaining_kids = 14 := by
  sorry

end NUMINAMATH_CALUDE_kids_went_home_l20_2038


namespace NUMINAMATH_CALUDE_consecutive_eight_product_divisible_by_ten_l20_2083

theorem consecutive_eight_product_divisible_by_ten (n : ℕ+) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) := by
  sorry

#check consecutive_eight_product_divisible_by_ten

end NUMINAMATH_CALUDE_consecutive_eight_product_divisible_by_ten_l20_2083


namespace NUMINAMATH_CALUDE_range_of_m_l20_2070

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (x - a) / 3 < 0 ∧ 2 * (x - 5) < 3 * x - 8

-- Define the solution set
def solution_set (a : ℝ) : Set ℤ :=
  {x : ℤ | inequality_system x a}

-- State the theorem
theorem range_of_m (a : ℝ) (m : ℝ) :
  (∀ x : ℤ, x ∈ solution_set a ↔ (x = -1 ∨ x = 0)) →
  (10 * a = 2 * m + 5) →
  -2.5 < m ∧ m ≤ 2.5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l20_2070


namespace NUMINAMATH_CALUDE_perpendicular_slope_l20_2031

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := -a / b
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = b / a :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l20_2031


namespace NUMINAMATH_CALUDE_simplify_radical_product_l20_2007

theorem simplify_radical_product (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l20_2007


namespace NUMINAMATH_CALUDE_converse_of_negative_square_positive_l20_2044

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_negative_square_positive_l20_2044


namespace NUMINAMATH_CALUDE_eggs_for_cake_l20_2045

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of eggs Megan bought -/
def bought : ℕ := dozen

/-- The number of eggs Megan's neighbor gave her -/
def given : ℕ := dozen

/-- The number of eggs Megan used for an omelet -/
def omelet : ℕ := 2

/-- The number of eggs Megan plans to use for her next meals -/
def meal_plan : ℕ := 3 * 3

theorem eggs_for_cake :
  ∃ (cake : ℕ),
    bought + given - omelet - (bought + given - omelet) / 2 - meal_plan = cake ∧
    cake = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_for_cake_l20_2045


namespace NUMINAMATH_CALUDE_factorization_equality_l20_2035

theorem factorization_equality (x y : ℝ) :
  3 * y * (y^2 - 4) + 5 * x * (y^2 - 4) = (3*y + 5*x) * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l20_2035


namespace NUMINAMATH_CALUDE_percentage_less_than_l20_2043

theorem percentage_less_than (w x y z P : ℝ) : 
  w = x * (1 - P / 100) →
  x = y * 0.6 →
  z = y * 0.54 →
  z = w * 1.5 →
  P = 40 :=
by sorry

end NUMINAMATH_CALUDE_percentage_less_than_l20_2043


namespace NUMINAMATH_CALUDE_find_numbers_l20_2065

theorem find_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 12) : x = 26 ∧ y = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_numbers_l20_2065


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l20_2028

def total_highlighters : ℕ := 15
def pink_highlighters : ℕ := 3
def yellow_highlighters : ℕ := 7

theorem blue_highlighters_count :
  total_highlighters - (pink_highlighters + yellow_highlighters) = 5 :=
by sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l20_2028


namespace NUMINAMATH_CALUDE_quadratic_equation_property_l20_2016

/-- A quadratic equation with two equal real roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  condition : a - b + c = 0
  equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x)

/-- Theorem stating that for a quadratic equation with two equal real roots and a - b + c = 0, we have 2a - b = 0 -/
theorem quadratic_equation_property (eq : QuadraticEquation) : 2 * eq.a - eq.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_property_l20_2016


namespace NUMINAMATH_CALUDE_sophies_money_correct_l20_2076

/-- The amount of money Sophie's aunt gave her --/
def sophies_money : ℝ := 260

/-- The cost of one shirt --/
def shirt_cost : ℝ := 18.50

/-- The number of shirts Sophie bought --/
def num_shirts : ℕ := 2

/-- The cost of the trousers --/
def trouser_cost : ℝ := 63

/-- The cost of one additional article of clothing --/
def additional_item_cost : ℝ := 40

/-- The number of additional articles of clothing Sophie plans to buy --/
def num_additional_items : ℕ := 4

/-- Theorem stating that the amount of money Sophie's aunt gave her is correct --/
theorem sophies_money_correct : 
  sophies_money = 
    shirt_cost * num_shirts + 
    trouser_cost + 
    additional_item_cost * num_additional_items := by
  sorry

end NUMINAMATH_CALUDE_sophies_money_correct_l20_2076


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l20_2089

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 * x + 9) = 13 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l20_2089


namespace NUMINAMATH_CALUDE_impossible_constant_average_l20_2078

theorem impossible_constant_average (n : ℕ) (initial_total_age : ℕ) : 
  initial_total_age = n * 19 →
  ¬ ∃ (new_total_age : ℕ), new_total_age = initial_total_age + 1 ∧ 
    new_total_age / (n + 1) = 19 :=
by sorry

end NUMINAMATH_CALUDE_impossible_constant_average_l20_2078


namespace NUMINAMATH_CALUDE_three_hour_therapy_charge_l20_2002

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hoursForStandardSession : ℕ
  totalChargeForStandardSession : ℕ

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating that given the conditions, the total charge for 3 hours of therapy is $188. -/
theorem three_hour_therapy_charge 
  (pricing : TherapyPricing) 
  (h1 : pricing.firstHourCharge = pricing.additionalHourCharge + 20)
  (h2 : pricing.hoursForStandardSession = 5)
  (h3 : pricing.totalChargeForStandardSession = 300)
  (h4 : totalCharge pricing pricing.hoursForStandardSession = pricing.totalChargeForStandardSession) :
  totalCharge pricing 3 = 188 :=
by
  sorry


end NUMINAMATH_CALUDE_three_hour_therapy_charge_l20_2002


namespace NUMINAMATH_CALUDE_kolya_walking_speed_l20_2013

/-- Represents the scenario of Kolya's journey to the store -/
structure JourneyScenario where
  total_distance : ℝ
  initial_speed : ℝ
  doubled_speed : ℝ
  store_closing_time : ℝ

/-- Calculates Kolya's walking speed given a JourneyScenario -/
def calculate_walking_speed (scenario : JourneyScenario) : ℝ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating that Kolya's walking speed is 20/3 km/h -/
theorem kolya_walking_speed (scenario : JourneyScenario) 
  (h1 : scenario.initial_speed = 10)
  (h2 : scenario.doubled_speed = 2 * scenario.initial_speed)
  (h3 : scenario.store_closing_time = scenario.total_distance / scenario.initial_speed)
  (h4 : scenario.total_distance > 0) :
  calculate_walking_speed scenario = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_kolya_walking_speed_l20_2013


namespace NUMINAMATH_CALUDE_sqrt_a_minus_2_real_l20_2017

theorem sqrt_a_minus_2_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_2_real_l20_2017


namespace NUMINAMATH_CALUDE_tenth_fibonacci_is_89_l20_2027

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem tenth_fibonacci_is_89 : fibonacci 9 = 89 := by
  sorry

end NUMINAMATH_CALUDE_tenth_fibonacci_is_89_l20_2027


namespace NUMINAMATH_CALUDE_factorial_equality_l20_2051

theorem factorial_equality (N : ℕ) (h : N > 0) :
  (7 : ℕ).factorial * (11 : ℕ).factorial = 18 * N.factorial → N = 11 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l20_2051


namespace NUMINAMATH_CALUDE_unique_number_five_times_less_than_digit_sum_l20_2080

def sum_of_digits (x : ℝ) : ℕ :=
  sorry

theorem unique_number_five_times_less_than_digit_sum :
  ∃! x : ℝ, x ≠ 0 ∧ x = (sum_of_digits x : ℝ) / 5 ∧ x = 1.8 :=
sorry

end NUMINAMATH_CALUDE_unique_number_five_times_less_than_digit_sum_l20_2080


namespace NUMINAMATH_CALUDE_selection_for_38_classes_6_routes_l20_2092

/-- The number of ways for a given number of classes to each choose one of a given number of routes. -/
def number_of_selections (num_classes : ℕ) (num_routes : ℕ) : ℕ := num_routes ^ num_classes

/-- Theorem stating that the number of ways for 38 classes to each choose one of 6 routes is 6^38. -/
theorem selection_for_38_classes_6_routes : number_of_selections 38 6 = 6^38 := by
  sorry

#eval number_of_selections 38 6

end NUMINAMATH_CALUDE_selection_for_38_classes_6_routes_l20_2092


namespace NUMINAMATH_CALUDE_last_ball_is_green_l20_2004

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the state of the box with balls -/
structure BoxState where
  red : Nat
  blue : Nat
  green : Nat

/-- Represents an exchange operation -/
inductive Exchange
  | RedBlueToGreen
  | RedGreenToBlue
  | BlueGreenToRed

/-- Applies an exchange operation to a box state -/
def applyExchange (state : BoxState) (ex : Exchange) : BoxState :=
  match ex with
  | Exchange.RedBlueToGreen => 
      { red := state.red - 1, blue := state.blue - 1, green := state.green + 1 }
  | Exchange.RedGreenToBlue => 
      { red := state.red - 1, blue := state.blue + 1, green := state.green - 1 }
  | Exchange.BlueGreenToRed => 
      { red := state.red + 1, blue := state.blue - 1, green := state.green - 1 }

/-- Checks if the box state has only one ball left -/
def isLastBall (state : BoxState) : Bool :=
  state.red + state.blue + state.green = 1

/-- Gets the color of the last ball -/
def getLastBallColor (state : BoxState) : Option Color :=
  if state.red = 1 then some Color.Red
  else if state.blue = 1 then some Color.Blue
  else if state.green = 1 then some Color.Green
  else none

/-- The main theorem to prove -/
theorem last_ball_is_green (exchanges : List Exchange) :
  let initialState : BoxState := { red := 10, blue := 11, green := 12 }
  let finalState := exchanges.foldl applyExchange initialState
  isLastBall finalState → getLastBallColor finalState = some Color.Green :=
by sorry

end NUMINAMATH_CALUDE_last_ball_is_green_l20_2004


namespace NUMINAMATH_CALUDE_book_selection_theorem_l20_2079

theorem book_selection_theorem (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  (Nat.choose (n - 1) (m - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l20_2079


namespace NUMINAMATH_CALUDE_constant_ratio_sum_l20_2020

theorem constant_ratio_sum (x₁ x₂ x₃ x₄ : ℝ) (k : ℝ) 
  (h_not_all_equal : ¬(x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄))
  (h_ratio_12_34 : (x₁ + x₂) / (x₃ + x₄) = k)
  (h_ratio_13_24 : (x₁ + x₃) / (x₂ + x₄) = k)
  (h_ratio_14_23 : (x₁ + x₄) / (x₂ + x₃) = k)
  (h_ratio_34_12 : (x₃ + x₄) / (x₁ + x₂) = k)
  (h_ratio_24_13 : (x₂ + x₄) / (x₁ + x₃) = k)
  (h_ratio_23_14 : (x₂ + x₃) / (x₁ + x₄) = k) :
  k = -1 := by sorry

end NUMINAMATH_CALUDE_constant_ratio_sum_l20_2020


namespace NUMINAMATH_CALUDE_children_per_seat_l20_2015

theorem children_per_seat (total_children : ℕ) (total_seats : ℕ) (h1 : total_children = 58) (h2 : total_seats = 29) :
  total_children / total_seats = 2 := by
  sorry

end NUMINAMATH_CALUDE_children_per_seat_l20_2015


namespace NUMINAMATH_CALUDE_wilson_number_l20_2029

theorem wilson_number (N : ℚ) : N - (1/3) * N = 16/3 → N = 8 := by
  sorry

end NUMINAMATH_CALUDE_wilson_number_l20_2029


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l20_2001

def is_valid_number (n : ℕ) : Prop :=
  n % 5 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4

theorem least_number_with_remainder_four :
  is_valid_number 184 ∧ ∀ m : ℕ, m < 184 → ¬ is_valid_number m :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l20_2001


namespace NUMINAMATH_CALUDE_nelly_painting_payment_l20_2058

/-- The amount Nelly paid for a painting at an auction, given Joe's bid and the condition of her payment. -/
theorem nelly_painting_payment (joe_bid : ℕ) (h : joe_bid = 160000) : 
  3 * joe_bid + 2000 = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_painting_payment_l20_2058


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_one_l20_2034

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_implies_a_leq_neg_one :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y ≤ 2 → f a x > f a y) → a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_one_l20_2034


namespace NUMINAMATH_CALUDE_height_percentage_difference_l20_2071

theorem height_percentage_difference (height_A height_B : ℝ) :
  height_B = height_A * (1 + 0.42857142857142854) →
  (height_B - height_A) / height_B * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l20_2071


namespace NUMINAMATH_CALUDE_seating_arrangements_l20_2048

/-- The number of seats in the front row -/
def front_seats : ℕ := 11

/-- The number of seats in the back row -/
def back_seats : ℕ := 12

/-- The total number of seats -/
def total_seats : ℕ := front_seats + back_seats

/-- The number of restricted seats in the front row -/
def restricted_seats : ℕ := 3

/-- The number of people to be seated -/
def people : ℕ := 2

/-- The number of arrangements without restrictions -/
def arrangements_without_restrictions : ℕ := total_seats * (total_seats - 2)

/-- The number of arrangements with one person in restricted seats -/
def arrangements_with_one_restricted : ℕ := restricted_seats * (total_seats - 3)

/-- The number of arrangements with both people in restricted seats -/
def arrangements_both_restricted : ℕ := restricted_seats * (restricted_seats - 1)

theorem seating_arrangements :
  arrangements_without_restrictions - 2 * arrangements_with_one_restricted + arrangements_both_restricted = 346 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l20_2048


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l20_2094

theorem quadratic_equation_roots : ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + 2*x - 3 = 0) ∧ (y^2 + 2*y - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l20_2094


namespace NUMINAMATH_CALUDE_power_of_power_l20_2098

theorem power_of_power (a : ℝ) : (a ^ 3) ^ 4 = a ^ 12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l20_2098


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l20_2081

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l20_2081


namespace NUMINAMATH_CALUDE_kocourkov_coins_l20_2006

theorem kocourkov_coins (a b : ℕ+) : 
  (∀ n > 53, ∃ x y : ℕ, n = a.val * x + b.val * y) ∧ 
  (¬ ∃ x y : ℕ, 53 = a.val * x + b.val * y) → 
  ((a.val = 2 ∧ b.val = 55) ∨ (a.val = 3 ∧ b.val = 28) ∨ 
   (a.val = 55 ∧ b.val = 2) ∨ (a.val = 28 ∧ b.val = 3)) :=
by sorry

end NUMINAMATH_CALUDE_kocourkov_coins_l20_2006


namespace NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l20_2041

theorem smallest_denominator_between_fractions :
  ∃ (p q : ℕ), 
    q = 4027 ∧ 
    (1 : ℚ) / 2014 < (p : ℚ) / q ∧ 
    (p : ℚ) / q < (1 : ℚ) / 2013 ∧
    (∀ (p' q' : ℕ), 
      (1 : ℚ) / 2014 < (p' : ℚ) / q' ∧ 
      (p' : ℚ) / q' < (1 : ℚ) / 2013 → 
      q ≤ q') :=
by sorry

end NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l20_2041


namespace NUMINAMATH_CALUDE_digit_product_le_unique_solution_l20_2033

-- Define p(n) as the product of digits of n
def digit_product (n : ℕ) : ℕ := sorry

-- Theorem 1: For any natural number n, p(n) ≤ n
theorem digit_product_le (n : ℕ) : digit_product n ≤ n := by sorry

-- Theorem 2: 45 is the only natural number satisfying 10p(n) = n^2 + 4n - 2005
theorem unique_solution :
  ∀ n : ℕ, 10 * (digit_product n) = n^2 + 4*n - 2005 ↔ n = 45 := by sorry

end NUMINAMATH_CALUDE_digit_product_le_unique_solution_l20_2033


namespace NUMINAMATH_CALUDE_factorization1_factorization2_l20_2040

-- Define the expressions
def expr1 (x y : ℝ) : ℝ := 4 - 12 * (x - y) + 9 * (x - y)^2

def expr2 (a x : ℝ) : ℝ := 2 * a * (x^2 + 1)^2 - 8 * a * x^2

-- State the theorems
theorem factorization1 (x y : ℝ) : expr1 x y = (2 - 3*x + 3*y)^2 := by sorry

theorem factorization2 (a x : ℝ) : expr2 a x = 2 * a * (x - 1)^2 * (x + 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization1_factorization2_l20_2040


namespace NUMINAMATH_CALUDE_third_arrangement_is_goat_monkey_donkey_l20_2068

-- Define the animals
inductive Animal : Type
| Monkey : Animal
| Donkey : Animal
| Goat : Animal

-- Define a seating arrangement as a triple of animals
def Arrangement := (Animal × Animal × Animal)

-- Define the property of an animal being in a specific position
def isInPosition (a : Animal) (pos : Nat) (arr : Arrangement) : Prop :=
  match pos, arr with
  | 0, (x, _, _) => x = a
  | 1, (_, x, _) => x = a
  | 2, (_, _, x) => x = a
  | _, _ => False

-- Define the property that each animal has been in each position
def eachAnimalInEachPosition (arr1 arr2 arr3 : Arrangement) : Prop :=
  ∀ (a : Animal) (p : Nat), p < 3 → 
    isInPosition a p arr1 ∨ isInPosition a p arr2 ∨ isInPosition a p arr3

-- Main theorem
theorem third_arrangement_is_goat_monkey_donkey 
  (arr1 arr2 arr3 : Arrangement)
  (h1 : isInPosition Animal.Monkey 2 arr1)
  (h2 : isInPosition Animal.Donkey 1 arr2)
  (h3 : eachAnimalInEachPosition arr1 arr2 arr3) :
  arr3 = (Animal.Goat, Animal.Monkey, Animal.Donkey) :=
sorry

end NUMINAMATH_CALUDE_third_arrangement_is_goat_monkey_donkey_l20_2068


namespace NUMINAMATH_CALUDE_cheese_needed_for_event_l20_2063

def meat_for_10_sandwiches : ℝ := 4
def number_of_sandwiches_planned : ℕ := 30
def initial_sandwich_count : ℕ := 10

theorem cheese_needed_for_event :
  let meat_per_sandwich : ℝ := meat_for_10_sandwiches / initial_sandwich_count
  let cheese_per_sandwich : ℝ := meat_per_sandwich / 2
  cheese_per_sandwich * number_of_sandwiches_planned = 6 := by
sorry

end NUMINAMATH_CALUDE_cheese_needed_for_event_l20_2063


namespace NUMINAMATH_CALUDE_unique_solution_m_n_l20_2082

theorem unique_solution_m_n : ∃! (m n : ℕ+), (m + n : ℕ)^(m : ℕ) = n^(m : ℕ) + 1413 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_m_n_l20_2082


namespace NUMINAMATH_CALUDE_max_value_M_l20_2054

/-- The maximum value of M = 11xy + 3x + 2012yz, where x, y, z are non-negative integers and x + y + z = 1000 -/
theorem max_value_M : 
  ∃ (x y z : ℕ), 
    x + y + z = 1000 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 1000 → 
      11 * x * y + 3 * x + 2012 * y * z ≥ 11 * a * b + 3 * a + 2012 * b * c ∧
      11 * x * y + 3 * x + 2012 * y * z = 503000000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_M_l20_2054


namespace NUMINAMATH_CALUDE_high_school_science_club_payment_l20_2010

theorem high_school_science_club_payment (B C : Nat) : 
  B < 10 → C < 10 → 
  (100 * B + 40 + C) % 15 = 0 → 
  (100 * B + 40 + C) % 5 = 0 → 
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_high_school_science_club_payment_l20_2010


namespace NUMINAMATH_CALUDE_salt_dilution_l20_2093

theorem salt_dilution (initial_seawater : ℝ) (initial_salt_percentage : ℝ) 
  (final_salt_percentage : ℝ) (added_freshwater : ℝ) :
  initial_seawater = 40 →
  initial_salt_percentage = 0.05 →
  final_salt_percentage = 0.02 →
  added_freshwater = 60 →
  (initial_seawater * initial_salt_percentage) / (initial_seawater + added_freshwater) = final_salt_percentage :=
by
  sorry

#check salt_dilution

end NUMINAMATH_CALUDE_salt_dilution_l20_2093
