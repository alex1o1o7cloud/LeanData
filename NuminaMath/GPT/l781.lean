import Mathlib
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Gcd
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.PrimeFactors
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Optimization
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Compositions
import Mathlib.Data.Finset
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.LinearAlgebra.FiniteDimensional
import Mathlib.Probability.Basic
import Mathlib.Tactic
import ProbabilityTheory
import Real

namespace find_phi_l781_781195

theorem find_phi (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0) 
  (hφ_range : -π / 2 < φ ∧ φ < π / 2) 
  (h_sym_dist : ∃ d, d = π / 6 ∧ d = π / (2ω)) 
  (h_sym_pt : ∃ x, x = 5π / 18 ∧ 2 * sin (ω * x + φ) = 0) 
  : φ = π / 3 := 
sorry

end find_phi_l781_781195


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781508

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781508


namespace ball_distribution_l781_781239

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781239


namespace range_of_m_l781_781330

variable (M : Set ℝ) (N : Set ℝ) (m : ℝ)

def M_def := {x : ℝ | x ≤ m}
def N_def := {y : ℝ | y = (λ x : ℝ, (x - 1)^2 - 1) x ∧ x ∈ ℝ}

theorem range_of_m (hM : M = M_def m) (hN : N = {y : ℝ | y ≥ -1}) (h_empty : M ∩ N = ∅) : m < -1 :=
sorry

end range_of_m_l781_781330


namespace volunteer_count_change_l781_781093

theorem volunteer_count_change :
  let x := 1
  let fall_increase := 1.09
  let winter_increase := 1.15
  let spring_decrease := 0.81
  let summer_increase := 1.12
  let summer_end_decrease := 0.95
  let final_ratio := x * fall_increase * winter_increase * spring_decrease * summer_increase * summer_end_decrease
  (final_ratio - x) / x * 100 = 19.13 :=
by
  sorry

end volunteer_count_change_l781_781093


namespace forest_problem_l781_781497

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781497


namespace forest_problem_l781_781498

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781498


namespace num_ways_distribute_balls_l781_781256

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781256


namespace solutions_p_eq_x3_plus_y3_l781_781128
open Nat

theorem solutions_p_eq_x3_plus_y3 (p : ℕ) [hp : Fact (Nat.prime p)] (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  p^n = x^3 + y^3 →
  ((p = 2 ∧ ∃ k : ℕ, x = 2^k ∧ y = 2^k ∧ n = 3*k + 1) ∨
   (p = 3 ∧ ∃ k : ℕ, (x = 3^k ∧ y = 2*3^k ∧ n = 3*k + 2) ∨ (x = 2*3^k ∧ y = 3^k ∧ n = 3*k + 2))) :=
begin
  intro h,
  sorry
end

end solutions_p_eq_x3_plus_y3_l781_781128


namespace beetle_ate_maggots_twice_l781_781115

theorem beetle_ate_maggots_twice
  (initial_maggots : ℕ)
  (maggots_ate_first_time : ℕ)
  (total_maggots : ℕ)
  (attempted_maggots_second_time : ℕ)
  (remaining_maggots_after_first : ℕ)
  (remaining_maggots_after_second : ℕ) :
  initial_maggots = 10 →
  maggots_ate_first_time = 1 →
  total_maggots = 20 →
  attempted_maggots_second_time = 10 →
  remaining_maggots_after_first = total_maggots - maggots_ate_first_time →
  remaining_maggots_after_second = remaining_maggots_after_first - attempted_maggots_second_time →
  remaining_maggots_after_second = 9 :=
by
  intros h1 h2 h3 h4 h5 h6,
  sorry

end beetle_ate_maggots_twice_l781_781115


namespace minimum_value_of_expression_l781_781719

variable {f : ℝ → ℝ} 
variable (a b : ℝ)

noncomputable def problemConditions :=
  -- Condition 1: f is an odd function
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  -- Condition 2: f is monotonically increasing
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  -- Condition 3: f is defined on ℝ (implicit by the type f : ℝ → ℝ)
  -- Condition 4: a and b are positive real numbers
  (0 < a ∧ 0 < b) ∧ 
  -- Condition 5: f(2a) + f(b - 4) = 0
  (f (2 * a) + f (b - 4) = 0)

theorem minimum_value_of_expression (h : problemConditions f a b) :
  ∃ (v : ℝ), (∀ x y : ℝ, ((problemConditions f x y) → (v ≤ (1 / (x + 1) + 2 / y))) ∧ v = 4 / 3) :=
sorry

end minimum_value_of_expression_l781_781719


namespace estimated_total_volume_correct_l781_781530

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781530


namespace find_daily_wage_of_c_l781_781436

def dailyWagesInRatio (a b c : ℕ) : Prop :=
  4 * a = 3 * b ∧ 5 * a = 3 * c

def totalEarnings (a b c : ℕ) (total : ℕ) : Prop :=
  6 * a + 9 * b + 4 * c = total

theorem find_daily_wage_of_c (a b c : ℕ) (total : ℕ) 
  (h1 : dailyWagesInRatio a b c) 
  (h2 : totalEarnings a b c total) 
  (h3 : total = 1406) : 
  c = 95 :=
by
  -- We assume the conditions and solve the required proof.
  sorry

end find_daily_wage_of_c_l781_781436


namespace share_of_B_is_7_over_6_l781_781884

theorem share_of_B_is_7_over_6 :
  ∃ (a d : ℚ), (5 * a = 5) ∧ (a - 2 * d) + (a - d) = (a) + (a + d) + (a + 2 * d) ∧ (a - d = 7 / 6) :=
begin
  sorry
end

end share_of_B_is_7_over_6_l781_781884


namespace tangent_line_y_intercept_l781_781063

theorem tangent_line_y_intercept :
  let center1 := (3: ℝ, 0: ℝ)
  let radius1 := 3
  let center2 := (7: ℝ, 0: ℝ)
  let radius2 := 2
  ∃ (m: ℝ) (b: ℝ), 
    (∀ (x: ℝ), (x - center1.fst) ^ 2 + (m * x + b - center1.snd) ^ 2 = radius1 ^ 2) ∧
    (∀ (x: ℝ), (x - center2.fst) ^ 2 + (m * x + b - center2.snd) ^ 2 = radius2 ^ 2) ∧
    b = 2 * Real.sqrt 17 :=
begin
  sorry
end

end tangent_line_y_intercept_l781_781063


namespace find_y_l781_781153

-- Define the constants and conditions 
def x : ℝ := 12
def lhs (y : ℝ) : ℝ := (17.28 / x) / (y * 0.2)

-- Define the problem statement
theorem find_y : ∃ y : ℝ, lhs y = 2 ∧ y = 3.6 :=
by
  -- Provide the proof steps to prove the theorem (to be filled)
  sorry

end find_y_l781_781153


namespace sector_area_l781_781420

theorem sector_area (theta : ℝ) (d : ℝ) (r : ℝ := d / 2) (circle_area : ℝ := π * r^2) 
    (sector_area : ℝ := (theta / 360) * circle_area) : 
  theta = 120 → d = 6 → sector_area = 3 * π :=
by
  intro htheta hd
  sorry

end sector_area_l781_781420


namespace correct_calculation_is_A_l781_781932

theorem correct_calculation_is_A :
  (-(((-27 : ℝ) ^ (1/3))) + (-real.sqrt(3))^2 = 6) ∧
  ((-2 : ℝ)^3 = -8) ∧
  (|2 - real.sqrt(3)| = 2 - real.sqrt(3)) ∧
  (real.sqrt(8) - real.sqrt(2) = real.sqrt(2)) → 
  true := sorry

end correct_calculation_is_A_l781_781932


namespace parabola_line_unique_eq_l781_781728

noncomputable def parabola_line_equation : Prop :=
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 4 * A.1) ∧ (B.2^2 = 4 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) ∧ ((A.2 + B.2) / 2 = 2) ∧
    ∀ x y, (y - 2 = 1 * (x - 2)) → (x - y = 0)

theorem parabola_line_unique_eq : parabola_line_equation :=
  sorry

end parabola_line_unique_eq_l781_781728


namespace ab_eq_e_l781_781193

theorem ab_eq_e (f : ℝ → ℝ) (a b : ℝ)
  (h₀ : f = λ x, |Real.log x - 1 / 2|)
  (h₁ : a ≠ b)
  (h₂ : f a = f b) : a * b = Real.exp 1 :=
by
  sorry

end ab_eq_e_l781_781193


namespace floor_sqrt_50_l781_781603

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781603


namespace area_of_right_triangle_l781_781011

theorem area_of_right_triangle (A B C : ℝ) (hA : A = 64) (hB : B = 36) (hC : C = 100) : 
  (1 / 2) * (Real.sqrt A) * (Real.sqrt B) = 24 :=
by
  sorry

end area_of_right_triangle_l781_781011


namespace integral_f_eq_7_over_6_l781_781198

noncomputable def f (x : ℝ) : ℝ := -x^2 + x + 1

theorem integral_f_eq_7_over_6 : ∫ x in 0..1, f x = 7 / 6 := by
  sorry

end integral_f_eq_7_over_6_l781_781198


namespace train_platform_length_l781_781044

theorem train_platform_length (time_platform : ℝ) (time_man : ℝ) (speed_km_per_hr : ℝ) :
  time_platform = 34 ∧ time_man = 20 ∧ speed_km_per_hr = 54 →
  let speed_m_per_s := speed_km_per_hr * (5/18)
  let length_train := speed_m_per_s * time_man
  let time_to_cover_platform := time_platform - time_man
  let length_platform := speed_m_per_s * time_to_cover_platform
  length_platform = 210 := 
by {
  sorry
}

end train_platform_length_l781_781044


namespace math_problem_statement_l781_781898

def floor (x : ℝ) : ℤ := int.floor x
def frac (x : ℝ) : ℝ := x - floor x

-- Definitions
def relatively_prime (p q : ℕ) : Prop := nat.gcd p q = 1
def product_of_real_numbers (P : set ℝ) : ℝ := P.prod id

-- Given problem static data
def a (p q : ℕ) (hpq: relatively_prime p q) : ℚ := ⟨p, q, hpq⟩

-- The main theorem we should prove
theorem math_problem_statement (p q : ℕ) (hpq: relatively_prime p q) (x_set : set ℝ)
  (h1 : ∀ x ∈ x_set, floor x + frac x ^ 2 = (a p q hpq : ℝ) * x ^ 3) 
  (h2 : product_of_real_numbers x_set = 1728) :
  p + q = 145 := by
  sorry

end math_problem_statement_l781_781898


namespace forest_trees_properties_l781_781515

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781515


namespace rational_sine_cosine_l781_781366

theorem rational_sine_cosine {α : ℝ} {a b c d : ℤ} 
  (h1 : sin α = a / c) 
  (h2 : cos α = b / d) 
  (hac : Int.gcd a c = 1) 
  (hbd : Int.gcd b d = 1) 
  (pythagorean : a^2 / c^2 + b^2 / d^2 = 1) :
  (c = d) ∧ (∃ k : ℤ, c = 4 * k + 1) ∧ (Int.gcd a b = 1) ∧ ((a % 2 = 0 ∧ (4 ∣ a)) ∨ (b % 2 = 0 ∧ (4 ∣ b))) ∧ (a % 2 ≠ b % 2) :=
sorry

end rational_sine_cosine_l781_781366


namespace a_range_of_proposition_l781_781176

theorem a_range_of_proposition (a : ℝ) : (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + 5 <= a * x) ↔ a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end a_range_of_proposition_l781_781176


namespace minimum_value_is_correct_l781_781324

open Real

noncomputable def minimum_value (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : 1 / x^2 + 1 / y + 1 / z = 6) : ℝ :=
  x^3 * y^2 * z^2

theorem minimum_value_is_correct :
  ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → (1 / x^2 + 1 / y + 1 / z = 6) → minimum_value x y z rfl rfl rfl rfl = 1 / (8 * sqrt 2) :=
by
  intros x y z h₁ h₂ h₃ h₄
  sorry

end minimum_value_is_correct_l781_781324


namespace sqrt_floor_eq_seven_l781_781626

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781626


namespace floor_sqrt_50_l781_781615

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781615


namespace smallest_positive_period_of_f_max_min_values_of_f_in_interval_l781_781209

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_positive_period_of_f :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi) :=
by sorry

theorem max_min_values_of_f_in_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ f x ≥ -1 / 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_in_interval_l781_781209


namespace impossible_to_visit_all_l781_781048

structure Cube :=
  (dimensions : ℕ × ℕ × ℕ)
  (unit_cubes : Finset (ℕ × ℕ × ℕ))

def is_adjacent (a b : (ℕ × ℕ × ℕ)) : Prop :=
  -- Two cubes are adjacent if they share a face
  (abs (a.1 - b.1) + abs (a.2 - b.2) + abs (a.3 - b.3)) = 1

def valid_move (a b c : (ℕ × ℕ × ℕ)) : Prop :=
  -- Valid move if a and b are adjacent and b and c are not in the same direction
  is_adjacent a b ∧ is_adjacent b c ∧ ¬(a.1 = b.1 ∧ b.1 = c.1) ∧ ¬(a.2 = b.2 ∧ b.2 = c.2) ∧ ¬(a.3 = b.3 ∧ b.3 = c.3)

theorem impossible_to_visit_all :
  ∀ (moves : List (ℕ × ℕ × ℕ)),
  moves.length = 27 →
  (∀ (a b c : (ℕ × ℕ × ℕ)), (a ∈ moves ∧ b ∈ moves ∧ c ∈ moves) → valid_move a b c) →
  ¬(Finset.univ ⊆ (Finset.mk moves sorry)) :=
by
  -- The proof is omitted
  sorry

end impossible_to_visit_all_l781_781048


namespace floor_sqrt_50_l781_781636

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781636


namespace remainder_eq_six_l781_781030

theorem remainder_eq_six
  (Dividend : ℕ) (Divisor : ℕ) (Quotient : ℕ) (Remainder : ℕ)
  (h1 : Dividend = 139)
  (h2 : Divisor = 19)
  (h3 : Quotient = 7)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Remainder = 6 :=
by
  sorry

end remainder_eq_six_l781_781030


namespace binom_n_n_sub_2_l781_781921

theorem binom_n_n_sub_2 (n : ℕ) (h : n > 0) : (Nat.choose n (n - 2)) = (n * (n - 1)) / 2 := by
  sorry

end binom_n_n_sub_2_l781_781921


namespace initial_premium_correct_l781_781844

-- Definitions for the conditions
def initial_premium (P : ℝ) : ℝ := P
def after_accident (P : ℝ) : ℝ := 1.10 * P
def after_tickets (P : ℝ) (num_tickets : ℕ) : ℝ := 1.10 * P + 5 * num_tickets
def new_premium (P : ℝ) : ℝ := after_tickets P 3

-- The proof problem
theorem initial_premium_correct (P : ℝ) : initial_premium P = 50 :=
by
  have h1: new_premium P = 70 := sorry  -- given condition
  have h2: after_tickets P 3 = 1.10 * P + 15 := by rfl
  have : 1.10 * P + 15 = 70 := by sorry  -- from h1
  have : 1.10 * P = 55 := by linarith
  show P = 50 from by linarith ⟩

end initial_premium_correct_l781_781844


namespace abc_plus_reciprocal_l781_781942

-- Define the real numbers a, b, c and their conditions
variables (a b c : ℝ)

-- Declare the hypotheses
hypothesis h1 : a + 1/b = 9
hypothesis h2 : b + 1/c = 10
hypothesis h3 : c + 1/a = 11

-- The goal is to prove abc + 1/(abc) = 960
theorem abc_plus_reciprocal {a b c : ℝ} (h1 : a + 1/b = 9) (h2 : b + 1/c = 10) (h3 : c + 1/a = 11) : 
  a * b * c + 1 / (a * b * c) = 960 :=
sorry

end abc_plus_reciprocal_l781_781942


namespace triangle_probability_l781_781998

theorem triangle_probability
  : ∀ (sticks : List ℕ), 
    sticks = [1, 4, 6, 8, 9, 10, 12, 15] →
    (∃ (selected : List ℕ), (selected.length = 4 ∧
    ∃ (triangle : List ℕ), (triangle.length = 3 ∧ 
                             (triangle.sum > 20 ∧
                              (∀ a b c, a ≤ b ∧ b ≤ c ∧ a + b > c))) / 7)
      sorry

end triangle_probability_l781_781998


namespace alice_age_30_l781_781911

variable (A T : ℕ)

def tom_younger_alice (A T : ℕ) := T = A - 15
def ten_years_ago (A T : ℕ) := A - 10 = 4 * (T - 10)

theorem alice_age_30 (A T : ℕ) (h1 : tom_younger_alice A T) (h2 : ten_years_ago A T) : A = 30 := 
by sorry

end alice_age_30_l781_781911


namespace cayuga_to_khaki_campbell_ratio_l781_781403

-- Definitions based on conditions
def ducks_total : ℕ := 90
def muscovy : ℕ := 39
def cayuga : ℕ := muscovy - 4
def khaki_campbell : ℕ := ducks_total - muscovy - cayuga

-- Lean theorem statement
theorem cayuga_to_khaki_campbell_ratio : cayuga.to_nat / khaki_campbell.to_nat = 35 / 16 :=
by
  sorry

end cayuga_to_khaki_campbell_ratio_l781_781403


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781504

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781504


namespace tangent_circles_bc_length_l781_781140

/-- 
If two circles with centers at points A and B are externally tangent, 
with radii 6 and 4 respectively, and a line externally tangent to 
both circles intersects ray AB at point C, then the length BC is 20.
-/
theorem tangent_circles_bc_length
  (A B C : Type)
  [Point A] [Point B] [Point C]
  (R1 R2 : ℝ)
  (hA : radius A R1)
  (hB : radius B R2)
  (hTangent: externally_tangent_circles A B)
  (hR1 : R1 = 6)
  (hR2 : R2 = 4)
  (hIntersect : (line_tangent_to_circles A B ∩ (ray A B)) = {C}) :
  length B C = 20 := 
by
  sorry

end tangent_circles_bc_length_l781_781140


namespace sum_of_coefficients_l781_781743

theorem sum_of_coefficients :
  ∀ (a_0 a_1 ... a_2009 : ℝ), (1 - 2 * x) ^ 2009 = a_0 + a_1 * x + ... + a_2009 * x ^ 2009 →
  a_0 + a_1 + ... + a_2009 = -1 :=
begin
  sorry
end

end sum_of_coefficients_l781_781743


namespace speed_conversion_l781_781062

theorem speed_conversion (speed_kmh : ℝ) (conversion_factor : ℝ) :
  speed_kmh = 1.1 ∧ conversion_factor = 1000 / 3600 → speed_kmh * conversion_factor = 0.31 := 
by
  intros h
  cases h with h_speed h_conversion
  rw [h_speed, h_conversion]
  norm_num
  sorry

end speed_conversion_l781_781062


namespace focus_of_ellipse_l781_781114

theorem focus_of_ellipse (center : ℤ × ℤ) (major_ep1 major_ep2 minor_ep1 minor_ep2 : ℤ × ℤ) :
  center = (5, -2) ∧
  major_ep1 = (0, -2) ∧ major_ep2 = (10, -2) ∧
  minor_ep1 = (5, 0) ∧ minor_ep2 = (5, -4) →
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 - b^2) in
  (5 + c, -2) = (5 + Real.sqrt 21, -2) :=
by
  intros h
  sorry

end focus_of_ellipse_l781_781114


namespace measure_six_pints_l781_781959

-- Define initial conditions
def total_wine := 12
def eight_pint_capacity := 8
def five_pint_capacity := 5

-- Define the Lean statement to prove the given problem
theorem measure_six_pints :
  ∃ (x y : ℕ), x ≤ eight_pint_capacity ∧ y ≤ five_pint_capacity ∧ 
               (x = 6) ∧ 
               lets_initialize x y :=
sorry

end measure_six_pints_l781_781959


namespace find_original_number_of_men_l781_781439

theorem find_original_number_of_men (x : ℕ) (h1 : x * 12 = (x - 6) * 14) : x = 42 :=
  sorry

end find_original_number_of_men_l781_781439


namespace balls_in_boxes_l781_781259

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781259


namespace analytical_expression_of_f_minimum_value_of_g_l781_781729

noncomputable def f (m x : ℝ) : ℝ := (m^2 - 3*m + 3) * x^(m^2 - 3/2*m + 1/2)
noncomputable def g (a x : ℝ) : ℝ := x + a * (f 2 x)^(1/3)

theorem analytical_expression_of_f :
  ∀ x > 0, f 2 x = x^(3/2) :=
sorry

theorem minimum_value_of_g :
  ∃ a : ℝ, a = -1 ∧ ∀ x ∈ set.Icc 1 9, g a x ≥ 0 ∧ (∃ u ∈ set.Icc 1 9, g a u = 0) :=
sorry

end analytical_expression_of_f_minimum_value_of_g_l781_781729


namespace field_trip_buses_needed_l781_781867

theorem field_trip_buses_needed
    (fifth_graders : ℕ) (sixth_graders : ℕ) (seventh_graders : ℕ)
    (teachers_per_grade : ℕ) (parents_per_grade : ℕ)
    (grades : ℕ) (seats_per_bus : ℕ)
    (H_fg : fifth_graders = 109)
    (H_sg : sixth_graders = 115)
    (H_sg2 : seventh_graders = 118)
    (H_tpg : teachers_per_grade = 4)
    (H_ppg : parents_per_grade = 2)
    (H_gr : grades = 3)
    (H_spb : seats_per_bus = 72) :
    let students := fifth_graders + sixth_graders + seventh_graders,
        adults := grades * (teachers_per_grade + parents_per_grade),
        total_people := students + adults in
    total_people / seats_per_bus = 5 := by
    sorry

end field_trip_buses_needed_l781_781867


namespace angle_between_OB_OC_tan_alpha_AC_perp_BC_l781_781175

-- Question 1: Prove the angle between OB and OC is π/6
theorem angle_between_OB_OC (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (2 + Real.cos α)^2 + (Real.sin α)^2 = 7) :
  Real.angle ⟨0, 2⟩ ⟨Real.cos α, Real.sin α⟩ = π/6 :=
sorry

-- Question 2: Prove that tan(α) = - (4 + sqrt(7)) / 3
theorem tan_alpha_AC_perp_BC (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (Real.cos α - 2, Real.sin α).dot (Real.cos α, Real.sin α - 2) = 0) :
  Real.tan α = -(4 + Real.sqrt 7) / 3 :=
sorry

end angle_between_OB_OC_tan_alpha_AC_perp_BC_l781_781175


namespace max_num_pairwise_relatively_prime_subsets_l781_781874

theorem max_num_pairwise_relatively_prime_subsets (S : Finset ℕ) (hS : S.card = 2001) :
  ∃ n, n = 2^2000 + 1 ∧ ∀ A B : Finset ℕ, A ∈ S.powerset → B ∈ S.powerset → A ≠ B → Nat.Coprime (A.sum) (B.sum) → n = 2^2000 + 1 :=
sorry

end max_num_pairwise_relatively_prime_subsets_l781_781874


namespace real_solutions_conditions_l781_781864

open Classical

theorem real_solutions_conditions (x y z p q : ℝ) 
  (h1 : sqrt x + sqrt y = z) 
  (h2 : 2 * x + 2 * y + p = 0) 
  (h3 : z^4 + p * z^2 + q = 0) : 
  p ≤ 0 ∧ q ≥ 0 ∧ p^2 - 4 * q ≥ 0 := sorry

end real_solutions_conditions_l781_781864


namespace michael_work_days_l781_781337

variables {M A E : ℝ}

def combined_work_rate (M A : ℝ) := M + A = 1 / 20
def emma_work_rate (A : ℝ) := E = 2 * A

theorem michael_work_days :
  combined_work_rate M A →
  emma_work_rate A →
  (15 * (M + 3 * A) + 3 * 8 * A + 2 * A = 1) →
  (M = 51 / 1120) →
  1 / M = 22 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end michael_work_days_l781_781337


namespace shaded_region_area_l781_781089

theorem shaded_region_area (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (cos_alpha : Real.cos α = 4 / 5) :
  let side_length : ℝ := 1 in
  let area_common := 1 / 2 in
  area_common = 1 / 2 := 
sorry

end shaded_region_area_l781_781089


namespace floor_sqrt_fifty_l781_781622

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781622


namespace problem1_problem2_l781_781051

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ 0) : 
  (a - b^2 / a) / ((a^2 + 2 * a * b + b^2) / a) = (a - b) / (a + b) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (6 - 2 * x ≥ 4) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (x ≤ 1) :=
by
  sorry

end problem1_problem2_l781_781051


namespace quadratic_real_solutions_l781_781774

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end quadratic_real_solutions_l781_781774


namespace quadrilateral_is_cyclic_l781_781789

variables {A B C D P Q : Point}
variables [convex_quadrilateral A B C D]
variables (h1 : perpendicular (line_through D C) (line_through P A))
variables (h2 : perpendicular (line_through C D) (line_through Q B))
variables (h3 : AQ = QC)
variables (h4 : BP = PD)

theorem quadrilateral_is_cyclic : cyclic_quadrilateral A B C D :=
by
  sorry

end quadrilateral_is_cyclic_l781_781789


namespace sin_sum_pi_over_4_l781_781269

theorem sin_sum_pi_over_4 (α : ℝ) 
  (h : (cos (2 * α) / sin (α - π/4)) = - (sqrt 2) / 2 ) :
  sin (α + π / 4) = sqrt 2 / 4 :=
sorry

end sin_sum_pi_over_4_l781_781269


namespace locus_of_tangent_points_l781_781971

noncomputable section
open Real Topology

variable {A B C D : ℝ×ℝ}

/-- Given a cyclic quadrilateral ABCD with AB not parallel to CD,
    the locus of points P for which we can find circles through
    AB and CD touching at P is the circle centered at the intersection
    of AB and CD with radius equal to the tangent length from
    the intersection to the circumcircle of ABCD --/
theorem locus_of_tangent_points (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_not_parallel : ¬are_parallel (line_through_points A B) (line_through_points C D)) :
  ∃ X R, (intersection_point_of_lines (line_through_points A B) (line_through_points C D) X) ∧
    locus P (circle_centered_radius X R) :=
sorry

end locus_of_tangent_points_l781_781971


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781542

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781542


namespace cone_surface_area_l781_781689

-- Define the radius of the base
def r : ℕ := 3

-- Define the height of the cone
def h : ℕ := 4 

-- Calculate the slant height using the Pythagorean theorem
def slant_height : ℕ := Math.sqrt (r * r + h * h)

-- Define the base's circumference
def base_circumference : ℝ := 2 * Math.pi * r

-- Define the area of the base
def base_area : ℝ := Math.pi * r * r

-- Define the lateral surface area
def lateral_surface_area : ℝ := (1 / 2) * base_circumference * slant_height

-- Define the total surface area
def total_surface_area : ℝ := lateral_surface_area + base_area

-- Statement to prove
theorem cone_surface_area : total_surface_area = 24 * π := by
  sorry

end cone_surface_area_l781_781689


namespace probability_of_selecting_one_painted_face_and_one_unpainted_face_l781_781582

noncomputable def probability_of_specific_selection :
  ℕ → ℕ → ℕ → ℚ
| total_cubes, painted_face_cubes, unpainted_face_cubes =>
  let total_pairs := (total_cubes * (total_cubes - 1)) / 2
  let success_pairs := painted_face_cubes * unpainted_face_cubes
  success_pairs / total_pairs

theorem probability_of_selecting_one_painted_face_and_one_unpainted_face :
  probability_of_specific_selection 36 13 17 = 221 / 630 :=
by
  sorry

end probability_of_selecting_one_painted_face_and_one_unpainted_face_l781_781582


namespace max_percent_error_l781_781305

-- Define radius and error
def actual_radius : ℝ := 15
def error_percent : ℝ := 0.1

-- Define actual circumference and range of computed circumferences
def actual_circumference := 2 * Real.pi * actual_radius
def min_circumference := 2 * Real.pi * (actual_radius * (1 - error_percent))
def max_circumference := 2 * Real.pi * (actual_radius * (1 + error_percent))

-- Theorem to prove the maximum possible percent error in the computed circumference
theorem max_percent_error (r : ℝ) (h1 : r = actual_radius) (h2 : error_percent = 0.1) :
  ∃ e : ℝ, e = 10 :=
by 
  let computed_circumferences := (min_circumference, max_circumference)
  sorry

end max_percent_error_l781_781305


namespace share_of_B_is_correct_l781_781938

def investment_A (x : ℝ) := 2 * x
def investment_B (x : ℝ) := (2/3) * x
def investment_C (x : ℝ) := x
def total_investment (x : ℝ) := investment_A x + investment_B x + investment_C x

def profit_B_share (total_profit : ℝ) (x : ℝ) := (investment_B x / total_investment x) * total_profit

theorem share_of_B_is_correct : 
  ∀ (total_profit : ℝ), total_profit = 8800 → 
  ∀ (x : ℝ), profit_B_share total_profit x = 2514.29 :=
by
  intros total_profit h_profit x
  rw [h_profit]
  sorry

end share_of_B_is_correct_l781_781938


namespace number_of_sets_satisfying_condition_l781_781899

theorem number_of_sets_satisfying_condition : 
  set.count (λ B : set ℕ, {1, 3} ∪ B = {1, 3, 5}) = 4 :=
sorry

end number_of_sets_satisfying_condition_l781_781899


namespace find_a_l781_781965

theorem find_a (a : ℝ) (h : average [3, 6, 3, 5, a, 3] = median [3, 6, 3, 5, a, 3]) : a = -2 := sorry

end find_a_l781_781965


namespace count_arithmetic_sequences_with_odd_diff_l781_781132

/-- Prove that the number of sets of three distinct digits 
from the set {0, 1, 2, ..., 9} such that the digits in each 
set are in an arithmetic sequence with an odd common difference 
is 12. -/
theorem count_arithmetic_sequences_with_odd_diff : 
  let S := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let is_arithmetic_sequence (s : Finset ℤ) :=
    ∃ (a d : ℤ), d % 2 = 1 ∧ (s = {a, a + d, a + 2 * d}) ∧ s ⊆ S
  (Finset.card (S.powerset.filter is_arithmetic_sequence)) = 12 :=
by
  sorry

end count_arithmetic_sequences_with_odd_diff_l781_781132


namespace range_m_domain_R_range_m_increasing_interval_l781_781160

noncomputable def f (x m : ℝ) : ℝ := log 0.5 (x^2 - m * x - m)

theorem range_m_domain_R :
  (∀ x, x^2 - m * x - m > 0) ↔ m ∈ set.Ioo (-4 : ℤ) 0 :=
by sorry

theorem range_m_increasing_interval :
  (∀ x ∈ set.Icc (-2 : ℝ) (-1/2), deriv (λ x, log 0.5 (x^2 - m * x - m)) x > 0) ↔ m ∈ set.Icc (-1 : ℝ) (1/2) :=
by sorry

end range_m_domain_R_range_m_increasing_interval_l781_781160


namespace num_ways_distribute_balls_l781_781255

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781255


namespace weight_of_second_square_l781_781488

-- Define the conditions
variables (s₁ s₂ : ℝ) (m₁ : ℝ)
-- First square's side length is 4 inches and weight is 16 ounces
def side_length_first_square := (4 : ℝ)
def weight_first_square := (16 : ℝ)
-- Second square's side length is 6 inches
def side_length_second_square := (6 : ℝ)

-- Define the proportionality relationship
def area (s : ℝ) := s^2
def weight (s : ℝ) (m : ℝ) := (area s) * m / (area side_length_first_square)

-- The theorem to prove
theorem weight_of_second_square:
  weight side_length_second_square weight_first_square = 36 :=
by
  sorry

end weight_of_second_square_l781_781488


namespace find_m_l781_781441

theorem find_m (m : ℕ) (h : 8 ^ 36 * 6 ^ 21 = 3 * 24 ^ m) : m = 43 :=
sorry

end find_m_l781_781441


namespace smallest_reducible_fraction_l781_781673

theorem smallest_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 8)) ∧ n = 144 := by
  sorry

end smallest_reducible_fraction_l781_781673


namespace distance_traveled_l781_781391

theorem distance_traveled 
(variable P_b P_f : ℕ) 
(variable D R_b R_f : ℕ)
(h1 : P_b = 9)
(h2 : P_f = 7)
(h3 : R_f = R_b + 10)
(h4 : D = R_b * P_b)
(h5 : D = R_f * P_f)
: D = 315 := by
  sorry

end distance_traveled_l781_781391


namespace no_Rin_is_Bin_l781_781584

variables (Bin Fin Rin : Type) 
variable (is_Fin : Bin → Fin → Prop)
variable (is_not_Fin : Rin → Fin → Prop)

-- Premises
def premise1 (b : Bin) (f : Fin) : Prop :=
  is_Fin b f

def premise2 (r : Rin) (n : Fin) : Prop :=
  is_not_Fin r n

-- Conclusion
def conclusion (r : Rin) (b : Bin) (f : Fin) : Prop :=
  ¬is_Fin b f

theorem no_Rin_is_Bin (Bin Fin Rin : Type) 
  (is_Fin : Bin → Fin → Prop)
  (is_not_Fin : Rin → Fin → Prop)
  (H1 : ∀ (b : Bin) (f : Fin), is_Fin b f)
  (H2 : ∃ (r : Rin) (n : Fin), is_not_Fin r n) : 
  ∀ (r : Rin) (b : Bin), ¬ (is_Fin b f) :=
sorry

end no_Rin_is_Bin_l781_781584


namespace closed_path_in_grid_l781_781284

theorem closed_path_in_grid :
  ∃ (path : (ℕ × ℕ) → (ℕ × ℕ)),
    (∀ p p' : (ℕ × ℕ), path p = path p' → p = p') ∧
    (∀ (x y : ℕ), x < 2018 ∧ y < 2018 →
      (∃ d : ℕ,
        path (x, y) = if d = 0 then (x + 1, y + 1) else (x + 1, y - 1) ∨
        path (x, y) = if d = 1 then (x - 1, y - 1) else (x - 1, y + 1) ∨
        path (x, y) = if d = 2 then (x + 1, y - 1) else (x - 1, y + 1) ∨
        path (x, y) = if d = 3 then (x - 1, y + 1) else (x + 1, y - 1)) ∧
    ∀ x, x ≤ 2018 * 2018 →
      ((path (x % 2018, x / 2018)) = (0, 0) ↔ x = 0)) :=
sorry

end closed_path_in_grid_l781_781284


namespace maximum_minimum_F_on_interval_l781_781161

def F (x: ℝ) := (1/3) * x^3 + x^2 - 8 * x

theorem maximum_minimum_F_on_interval :
  ∃ (max min : ℝ), max = F 3 ∧ min = F 2 ∧
  (∀ x ∈ set.Icc 1 3, F x ≤ max ∧ F x ≥ min) :=
by
  use F 3
  use F 2
  dsimp [F]
  split
  . rw [mul_assoc, ←mul_add, add_comm]
  . rw [mul_assoc, ←mul_add]
  split
  . exact sorry
  . exact sorry

end maximum_minimum_F_on_interval_l781_781161


namespace ball_distribution_l781_781240

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781240


namespace floor_sqrt_50_l781_781633

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781633


namespace euler_formula_complex_multiplication_l781_781138

theorem euler_formula_complex_multiplication :
  let z := Complex.exp (Real.pi / 2 * Complex.I)
  in z * (1 + 2 * Complex.I) = -2 + Complex.I :=
by
  let z := Complex.exp (Real.pi / 2 * Complex.I)
  show z * (1 + 2 * Complex.I) = -2 + Complex.I
  sorry

end euler_formula_complex_multiplication_l781_781138


namespace part1_a_eq_e_part2_solution_set_l781_781174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * a * x^2 - log x - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := exp x - a * x

theorem part1_a_eq_e (a : ℝ) (h : ∃ x, f a x = g a x) : a = Real.exp := sorry

noncomputable def h (x : ℝ) : ℝ := Real.exp x - (Real.exp * x^2) / (1 + log x)

theorem part2_solution_set : {x : ℝ | h x < 0} = Ioo 0 1 := sorry

end part1_a_eq_e_part2_solution_set_l781_781174


namespace total_earnings_l781_781576

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end total_earnings_l781_781576


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781506

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781506


namespace polygon_sides_arithmetic_progression_l781_781390

theorem polygon_sides_arithmetic_progression 
  (n : ℕ) 
  (h1 : ∀ n, ∃ a_1, ∃ a_n, ∀ i, a_n = 172 ∧ (a_i = a_1 + (i - 1) * 4) ∧ (i ≤ n))
  (h2 : ∀ S, S = 180 * (n - 2)) 
  (h3 : ∀ S, S = n * ((172 - 4 * (n - 1) + 172) / 2)) 
  : n = 12 := 
by 
  sorry

end polygon_sides_arithmetic_progression_l781_781390


namespace floor_sqrt_50_l781_781601

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781601


namespace ab_leq_one_l781_781887

theorem ab_leq_one (a b : ℝ) (h : (a + b) * (a + b + a + b) = 9) : a * b ≤ 1 := 
  sorry

end ab_leq_one_l781_781887


namespace tree_age_when_planted_l781_781013

theorem tree_age_when_planted :
  ∀ (initial_height growth_rate age_when_measured measured_height age_when_planted : ℕ),
  initial_height = 5 →
  growth_rate = 3 →
  age_when_measured = 7 →
  measured_height = 23 →
  age_when_measured - (measured_height - initial_height) / growth_rate = age_when_planted →
  age_when_planted = 1 :=
by
  intros initial_height growth_rate age_when_measured measured_height age_when_planted h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end tree_age_when_planted_l781_781013


namespace bisect_triangle_l781_781642

theorem bisect_triangle {A B C : Type} [EuclideanStructure A] [Triangle A B C] (AC BC AB : ℝ) :
  AC = 4 ∧ BC = 3 ∧ AB = 5 →
  ∃ x y : ℝ, x = 3 - sqrt 6 / 2 ∧ y = 3 + sqrt 6 / 2 ∧
    bisects_perimeter_and_area (triangle A B C AC BC AB) x y :=
by
  sorry

#print bisect_triangle

end bisect_triangle_l781_781642


namespace total_rainfall_january_l781_781997

theorem total_rainfall_january (R1 R2 T : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 21) : T = 35 :=
by 
  let R1 := 14
  let R2 := 21
  let T := R1 + R2
  sorry

end total_rainfall_january_l781_781997


namespace transform_negation_l781_781103

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l781_781103


namespace has_exactly_two_solutions_iff_l781_781664

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l781_781664


namespace floor_sqrt_50_l781_781638

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781638


namespace pet_store_regular_discount_l781_781081

theorem pet_store_regular_discount :
  ∀ (MSRP final_price : ℝ) (additional_discount_percentage : ℝ),
  MSRP = 45 → final_price = 25.2 → additional_discount_percentage = 20 →
  ∃ x : ℝ, final_price = MSRP * ((100 - x) / 100) * ((80) / 100) ∧ x = 30 :=
by
  intros MSRP final_price additional_discount_percentage MSRP_eq final_price_eq additional_discount_percentage_eq
  use 30
  rw [MSRP_eq, final_price_eq, additional_discount_percentage_eq]
  have eq1 : final_price = 45 * ((100 - 30) / 100) * (80 / 100) := by norm_num
  exact ⟨eq1, rfl⟩
  sorry

end pet_store_regular_discount_l781_781081


namespace vertical_angles_equal_l781_781974

theorem vertical_angles_equal (l1 l2 : ℝ → Prop) (h : ∃ p, l1 p ∧ l2 p) :
  ∀ θ1 θ2, (θ1 = θ2 ∨ θ1 = π - θ2) → θ1 = θ2 := 
sorry

end vertical_angles_equal_l781_781974


namespace boxes_left_l781_781301

-- Define the initial number of boxes
def initial_boxes : ℕ := 10

-- Define the number of boxes sold
def boxes_sold : ℕ := 5

-- Define a theorem stating that the number of boxes left is 5
theorem boxes_left : initial_boxes - boxes_sold = 5 :=
by
  sorry

end boxes_left_l781_781301


namespace find_n_l781_781178

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - imaginary_unit) = 1 + n * imaginary_unit) : n = 1 :=
sorry

end find_n_l781_781178


namespace balls_in_boxes_l781_781219

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781219


namespace problem_statement_l781_781813

theorem problem_statement : 
  let M := finset.card 
    (finset.filter 
      (λ n : ℕ, n ≤ 1500 ∧ (n.bits.len % 2 = 0) ∧ 
        (finset.card (finset.filter (λ b : Bool, b) n.bits.to_finset) > 
         finset.card (finset.filter (λ b : Bool, ¬b) n.bits.to_finset)))
      (finset.range 1501)) in
  M % 500 = 357 := 
by {
  sorry
}

end problem_statement_l781_781813


namespace part_a_part_b_l781_781811

-- Definition of the sequence
noncomputable def seq : ℕ → ℕ
| 1     := x1
| 2     := x2
| (n+1) := seq n * seq (n-1) + 1

-- Part (a)
theorem part_a (x1 x2 : ℕ) (h_rel_prime : Nat.gcd x1 x2 = 1) :
  ∀ i > 1, ∃ j > i, (seq x1 x2 i)^i ∣ (seq x1 x2 j)^j := 
sorry

-- Part (b)
theorem part_b (x1 x2 : ℕ) (h_rel_prime : Nat.gcd x1 x2 = 1) :
  ¬(∃ j > 1, x1 ∣ (seq x1 x2 j)^j) := 
sorry

end part_a_part_b_l781_781811


namespace john_receives_amount_l781_781442

theorem john_receives_amount (total_amount : ℕ) (part_john part_jose part_binoy : ℕ)
  (h1 : total_amount = 4320) (h2 : part_john = 2) (h3 : part_jose = 4) (h4 : part_binoy = 6) :
  let total_parts := part_john + part_jose + part_binoy in
  let value_per_part := total_amount / total_parts in
  let amount_john := value_per_part * part_john in
  amount_john = 720 :=
by
  intros
  sorry

end john_receives_amount_l781_781442


namespace largest_number_with_sum_19_l781_781028

theorem largest_number_with_sum_19 : ∃ (n : ℕ), 
  (∀ (d : ℕ), d ∈ digits 10 n → d ≠ 0) ∧ 
  (∀ (d1 d2 : ℕ), d1 ∈ digits 10 n → d2 ∈ digits 10 n → d1 ≠ d2) ∧
  (list.sum (digits 10 n) = 19) ∧ 
  (∀ (m : ℕ), 
    (∀ (d : ℕ), d ∈ digits 10 m → d ≠ 0) → 
    (∀ (d1 d2 : ℕ), d1 ∈ digits 10 m → d2 ∈ digits 10 m → d1 ≠ d2) →
    (list.sum (digits 10 m) = 19) → 
    (m ≤ n)) ∧ 
  n = 982 := 
sorry

end largest_number_with_sum_19_l781_781028


namespace estimated_total_volume_correct_l781_781523

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781523


namespace magazines_per_bookshelf_l781_781116

noncomputable def total_books : ℕ := 23
noncomputable def total_books_and_magazines : ℕ := 2436
noncomputable def total_bookshelves : ℕ := 29

theorem magazines_per_bookshelf : (total_books_and_magazines - total_books) / total_bookshelves = 83 :=
by
  sorry

end magazines_per_bookshelf_l781_781116


namespace exact_two_solutions_l781_781653

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l781_781653


namespace number_of_ways_to_put_balls_in_boxes_l781_781248

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781248


namespace forest_trees_properties_l781_781512

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781512


namespace balls_in_boxes_l781_781216

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781216


namespace triangles_reflection_not_perpendicular_l781_781092

noncomputable section

variables {a b c d : ℝ}

def in_first_quadrant (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0

def reflected_point (x y : ℝ) : ℝ × ℝ :=
  (y, x)

def slope (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1)

theorem triangles_reflection_not_perpendicular 
  (hPa : in_first_quadrant a b) (hQa : in_first_quadrant c d) (hRa : in_first_quadrant a b)
  (hne : a ≠ b ∧ c ≠ d):
  ¬ slope (a, b) (c, d) * slope (b, a) (d, c) = -1 :=
sorry

end triangles_reflection_not_perpendicular_l781_781092


namespace sum_of_all_angles_l781_781690

-- Define the angles involved
variables {α β γ δ ε ζ η : ℝ}

-- Define the sums of the angles within the triangle and quadrilateral
def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180
def sum_of_quadrilateral_angles (δ ε ζ η : ℝ) : Prop := δ + ε + ζ + η = 360

-- Theorem statement
theorem sum_of_all_angles
  (h_triangle : sum_of_triangle_angles α β γ)
  (h_quadrilateral : sum_of_quadrilateral_angles δ ε ζ η) :
  α + β + γ + δ + ε + ζ + η = 540 :=
begin
  sorry -- Proof omitted
end

end sum_of_all_angles_l781_781690


namespace slope_yintercept_sum_l781_781355

variables (x1 x2 : ℝ) (h : x2 > x1)

def point_c := (x1, 10)
def point_d := (x2, 15)

theorem slope_yintercept_sum : 
  (5 / (x2 - x1)) * (1 - x1) + 10 = do
  let slope := (15 - 10) / (x2 - x1)
  let y_intercept := (- (5 / (x2 - x1)) * x1 + 10)
  slope + y_intercept := 
sorry

end slope_yintercept_sum_l781_781355


namespace max_ab_min_4a2_b2_l781_781706

-- Define the positive numbers a and b and the condition 2a + b = 1
variables {a b : ℝ}
variable (h : a > 0 ∧ b > 0 ∧ 2 * a + b = 1)

-- Prove that the maximum value of ab is 1/8
theorem max_ab (h : a > 0 ∧ b > 0 ∧ 2 * a + b = 1) :
  ∃ a b, h → ab ≤ 1 / 8 :=
begin
  sorry  -- Actual proof goes here
end

-- Prove that the minimum value of 4a^2 + b^2 is 1/2
theorem min_4a2_b2 (h : a > 0 ∧ b > 0 ∧ 2 * a + b = 1) :
  ∃ a b, h → 4 * a^2 + b^2 ≥ 1 / 2 :=
begin
  sorry  -- Actual proof goes here
end

end max_ab_min_4a2_b2_l781_781706


namespace monomial_properties_l781_781035

def monomial_coeff (x y : ℝ) : ℝ := - (3 * x^2 * y) / 5

def monomial_degree (x y : ℝ) : ℕ := 2 + 1

theorem monomial_properties :
  monomial_coeff x y = -3/5 ∧ monomial_degree x y = 3 :=
by
  sorry

end monomial_properties_l781_781035


namespace average_score_decrease_l781_781597

theorem average_score_decrease :
  ∃ new_avg : ℝ, 
    let original_avg := 82.5,
        student_A_score := 86,
        new_student_A_score := 74,
        total_students := 8,
        total_score := total_students * original_avg in
    total_score - student_A_score + new_student_A_score = total_students * new_avg ∧
    original_avg - new_avg = 1.5 :=
by
  sorry

end average_score_decrease_l781_781597


namespace work_done_in_a_day_l781_781061

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end work_done_in_a_day_l781_781061


namespace polynomial_rad_property_l781_781676

-- Define rad function
def rad (n : ℕ) : ℕ :=
  if n = 0 ∨ n = 1 then 1
  else let primes := (List.range (n + 1)).filter (λ p, Nat.prime p ∧ n % p = 0) in primes.foldr (λ p acc, p * acc) 1

-- Define the main property to be proved in Lean
theorem polynomial_rad_property (f : ℕ → ℕ) (h : ∀ n : ℕ, 0 ≤ f n) :
  (∀ n : ℕ, rad (f n) ∣ rad (f (n ^ rad n))) →
  ∃ A B : ℕ, ∀ n : ℕ, f n = A * n ^ B := sorry

end polynomial_rad_property_l781_781676


namespace rectangle_area_transformation_l781_781376

theorem rectangle_area_transformation
  (A : ℝ)
  (hA : A = 432) :
  (0.9 * 1.1 * A).round = 428 :=
by
  sorry

end rectangle_area_transformation_l781_781376


namespace certain_event_l781_781034

theorem certain_event (a : ℝ) : a^2 ≥ 0 := 
sorry

end certain_event_l781_781034


namespace sum_c_n_l781_781180

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c_n (n : ℕ) : ℕ := a_n n + b_n n

theorem sum_c_n (n : ℕ) : 
  (∑ i in Finset.range n, c_n (i + 1)) = n^2 + (3^n - 1)/2 :=
by
  sorry

end sum_c_n_l781_781180


namespace forest_problem_l781_781495

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781495


namespace forest_trees_properties_l781_781520

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781520


namespace tan_neg_two_simplifies_l781_781761

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l781_781761


namespace part_a_part_b_l781_781371

/-- Conditions -/
variables {A B C D E F X Y K: ℝ} (a: ℝ) 

/-- Definitions -/
def square_drawn_exterior : Prop := (A + X + C + Y = B)
def equilateral_AB_AC : Prop := (abs(AB - AC) = 0)
def midpoint_D : Prop := (D = (B + C) / 2)
def feet_perpendicular (K: ℝ) : Prop := 
  (E = (perp_distance K BY) ∧ F = (perp_distance K CX))

/-- Part a -/
theorem part_a (h1: square_drawn_exterior) (h2: equilateral_AB_AC) 
  (h3: midpoint_D) (h4: feet_perpendicular K) : DE = DF := sorry

/-- Part b -/
theorem part_b (h1: square_drawn_exterior) (h2: equilateral_AB_AC) 
  (h3: midpoint_D) (h4: feet_perpendicular K): 
  ∀ (x: ℝ) , (mid_coord x) = (x / 2, (a / 2) - (x / 2)) := sorry

/-- Helper definition for midpoint coordinates -/
def mid_coord (x: ℝ): ℝ × ℝ :=
  (x / 2, (a - x) / 2)

end part_a_part_b_l781_781371


namespace marbles_problem_l781_781104

theorem marbles_problem (a : ℚ) (h1: 34 * a = 156) : a = 78 / 17 := 
by
  sorry

end marbles_problem_l781_781104


namespace total_earnings_l781_781578

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end total_earnings_l781_781578


namespace num_functions_with_range_R_l781_781406

noncomputable def f1 (x : ℝ) : ℝ := 3 - x
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x^2 + 1)
noncomputable def f3 (x : ℝ) : ℝ := x^2 + 2 * x - 10
noncomputable def f4 (x : ℝ) : ℝ := if x ≤ 0 then -x else -1 / x

theorem num_functions_with_range_R : 
  (∃ x : (ℝ → ℝ), x = f1 ∧ ∀ y : ℝ, ∃ x0 : ℝ, y = f1 x0) ∧
  (∃ x : (ℝ → ℝ), x = f4 ∧ ∀ y : ℝ, ∃ x0 : ℝ, y = f4 x0) ∧
  (¬(∃ x : (ℝ → ℝ), x = f2 ∧ ∀ y : ℝ, ∃ x0 : ℝ, y = f2 x0)) ∧
  (¬(∃ x : (ℝ → ℝ), x = f3 ∧ ∀ y : ℝ, ∃ x0 : ℝ, y = f3 x0)) ∧
  (∀ f : (ℝ → ℝ), f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) →
  2 := sorry

end num_functions_with_range_R_l781_781406


namespace exact_two_solutions_l781_781651

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l781_781651


namespace field_trip_buses_l781_781868

-- Definitions of conditions
def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def seats_per_bus : ℕ := 72

-- Total calculations
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_people : ℕ := total_students + total_chaperones
def buses_needed : ℕ := (total_people + seats_per_bus - 1) / seats_per_bus

theorem field_trip_buses : buses_needed = 6 := by
  unfold buses_needed
  unfold total_people total_students total_chaperones chaperones_per_grade
  norm_num
  sorry

end field_trip_buses_l781_781868


namespace problem_prob_div_4_l781_781851

noncomputable def probability_divisible_by_4
  (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 1004)
  (h2 : 1 ≤ b ∧ b ≤ 1004)
  (h3 : 1 ≤ c ∧ c ≤ 1004) : ℚ :=
  1/4

theorem problem_prob_div_4 :
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 1004) →
                 (1 ≤ b ∧ b ≤ 1004) →
                 (1 ≤ c ∧ c ≤ 1004) →
                 (probability_divisible_by_4 a b c (and.intro by assumption) (and.intro by assumption) (and.intro by assumption)) = 1/4 :=
begin
  sorry
end

end problem_prob_div_4_l781_781851


namespace tan_value_sin_cos_ratio_sin_squared_expression_l781_781682

theorem tan_value (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  Real.tan α = -1 / 3 :=
sorry

theorem sin_cos_ratio (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2 :=
sorry

theorem sin_squared_expression (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5 :=
sorry

end tan_value_sin_cos_ratio_sin_squared_expression_l781_781682


namespace ranking_Y_X_W_l781_781434

variables {W X Y : ℕ}

-- William's statement: "I didn't get the highest score in our class"
def william_not_highest := X > W

-- Yvonne's statement: "I didn't get the lowest score"
def yvonne_not_lowest := Y > X

-- Question: What is the ranking of the three pupils from highest to lowest?
theorem ranking_Y_X_W (h1 : william_not_highest) (h2 : yvonne_not_lowest) : Y > X ∧ X > W :=
by {
  have h3 : X > W := h1,
  have h4 : Y > X := h2,
  split;
  assumption
}

end ranking_Y_X_W_l781_781434


namespace david_marks_in_english_l781_781991

theorem david_marks_in_english : 
  ∀ (E : ℕ), 
  let math_marks := 85 
  let physics_marks := 82 
  let chemistry_marks := 87 
  let biology_marks := 85 
  let avg_marks := 85 
  let total_subjects := 5 
  let total_marks := avg_marks * total_subjects 
  let total_known_subject_marks := math_marks + physics_marks + chemistry_marks + biology_marks 
  total_marks = total_known_subject_marks + E → 
  E = 86 :=
by 
  intros
  sorry

end david_marks_in_english_l781_781991


namespace angle_subtraction_correct_polynomial_simplification_correct_l781_781575

noncomputable def angleSubtraction : Prop :=
  let a1 := 34 * 60 + 26 -- Convert 34°26' to total minutes
  let a2 := 25 * 60 + 33 -- Convert 25°33' to total minutes
  let diff := a1 - a2 -- Subtract in minutes
  let degrees := diff / 60 -- Convert back to degrees
  let minutes := diff % 60 -- Remainder in minutes
  degrees = 8 ∧ minutes = 53 -- Expected result in degrees and minutes

noncomputable def polynomialSimplification (m : Int) : Prop :=
  let expr := 5 * m^2 - (m^2 - 6 * m) - 2 * (-m + 3 * m^2)
  expr = -2 * m^2 + 8 * m -- Simplified form

-- Statements needing proof
theorem angle_subtraction_correct : angleSubtraction := by
  sorry

theorem polynomial_simplification_correct (m : Int) : polynomialSimplification m := by
  sorry

end angle_subtraction_correct_polynomial_simplification_correct_l781_781575


namespace curve_rectangular_eq_line_polar_eq_l781_781792

theorem curve_rectangular_eq (theta rho: ℝ) (h: rho = 6 * cos theta) : 
  ∃ (x y: ℝ), (x - 3)^2 + y^2 = 9 := 
sorry

theorem line_polar_eq (t: ℝ) (alpha: ℝ) (hα: alpha = (Real.pi) / 4) : 
  ∃ (x y: ℝ), y = -1 + t * sin alpha ∧ x = 2 + t * cos alpha ∧ (x - y - 3 = 0) := 
sorry

end curve_rectangular_eq_line_polar_eq_l781_781792


namespace fraction_uncovered_l781_781042

def area_rug (length width : ℕ) : ℕ := length * width
def area_square (side : ℕ) : ℕ := side * side

theorem fraction_uncovered 
  (rug_length rug_width floor_area : ℕ)
  (h_rug_length : rug_length = 2)
  (h_rug_width : rug_width = 7)
  (h_floor_area : floor_area = 64)
  : (floor_area - area_rug rug_length rug_width) / floor_area = 25 / 32 := 
sorry

end fraction_uncovered_l781_781042


namespace complex_distance_proof_l781_781788

def complex_point := Complex.div (2 : ℂ) (1 - Complex.i) = 1 + Complex.i

def point_line_distance := 
  ∀ (x y : ℝ), (x, y) = (1, 1) →
  (∀ c : ℝ, c = (1 / Real.sqrt 2) →
  1 / Real.sqrt 2 = Real.sqrt 2 / 2)

theorem complex_distance_proof : complex_point ∧ point_line_distance := sorry

end complex_distance_proof_l781_781788


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781505

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781505


namespace jake_delay_l781_781111

-- Define the conditions as in a)
def floors_jake_descends : ℕ := 8
def steps_per_floor : ℕ := 30
def steps_per_second_jake : ℕ := 3
def elevator_time_seconds : ℕ := 60 -- 1 minute = 60 seconds

-- Define the statement based on c)
theorem jake_delay (floors : ℕ) (steps_floor : ℕ) (steps_second : ℕ) (elevator_time : ℕ) :
  (floors = floors_jake_descends) →
  (steps_floor = steps_per_floor) →
  (steps_second = steps_per_second_jake) →
  (elevator_time = elevator_time_seconds) →
  (floors * steps_floor / steps_second - elevator_time = 20) :=
by
  intros
  sorry

end jake_delay_l781_781111


namespace hyperbola_asymptotes_angle_l781_781880

theorem hyperbola_asymptotes_angle : 
  let hyperbola_eq : ∀ x y, (x^2 / 3) - y^2 = 1, 
      asymptote_eqn : ∀ x y, y = (sqrt 3 / 3) * x ∨ y = -((sqrt 3 / 3) * x) in
  ∠ (asymptote_eqn 1) (asymptote_eqn 2) = π / 3 :=
by
  sorry

end hyperbola_asymptotes_angle_l781_781880


namespace value_of_f_log2_9_l781_781722

def f : ℝ → ℝ
| x := if x ≤ 0 then 2^x else f (x - 1) - 1

theorem value_of_f_log2_9 : f (Real.log 9 / Real.log 2) = -(55 / 16) :=
by
  sorry

end value_of_f_log2_9_l781_781722


namespace DE_plus_FG_equals_19_div_6_l781_781016

theorem DE_plus_FG_equals_19_div_6
    (AB AC : ℝ)
    (BC : ℝ)
    (h_isosceles : AB = 2 ∧ AC = 2 ∧ BC = 1.5)
    (D E G F : ℝ)
    (h_parallel_DE_BC : D = E)
    (h_parallel_FG_BC : F = G)
    (h_same_perimeter : 2 + D = 2 + F ∧ 2 + F = 5.5 - F) :
    D + F = 19 / 6 := by
  sorry

end DE_plus_FG_equals_19_div_6_l781_781016


namespace trapezoid_DC_length_l781_781141

theorem trapezoid_DC_length 
  (ABCD : Trapezoid)
  (AB DC : ℝ)
  (AB_parallel_DC : ABCD.AB ∥ ABCD.DC)
  (AB_eq_7 : AB = 7)
  (BC_eq_4sqrt3 : ABCD.BC = 4 * real.sqrt 3)
  (angle_BCD_eq_60 : ABCD.angle B C D = 60)
  (angle_CDA_eq_45 : ABCD.angle C D A = 45) :
  ABCD.DC = 17 := 
sorry

end trapezoid_DC_length_l781_781141


namespace inequality_must_hold_l781_781764

variable (a b c : ℝ)

theorem inequality_must_hold (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := 
sorry

end inequality_must_hold_l781_781764


namespace parabola_sum_l781_781476

def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - b * x + c

def f (a b c : ℝ) (x : ℝ) : ℝ := a * (x + 7) ^ 2 - b * (x + 7) + c

def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 3) ^ 2 - b * (x - 3) + c

def fg (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum (a b c x : ℝ) : fg a b c x = 2 * a * x ^ 2 + (8 * a - 2 * b) * x + (58 * a - 4 * b + 2 * c) := by
  sorry

end parabola_sum_l781_781476


namespace floor_sqrt_50_l781_781599

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781599


namespace largest_number_among_list_l781_781933

theorem largest_number_among_list :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  sorry

end largest_number_among_list_l781_781933


namespace forest_problem_l781_781500

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781500


namespace money_together_l781_781562

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l781_781562


namespace num_valid_colorings_is_25_l781_781058

-- Define a 3x3 grid
def grid := Fin 3 × Fin 3

-- Define the colors
inductive Color
| green
| red
| blue
deriving DecidableEq

-- Define a function to count the number of valid colorings
def valid_colorings (coloring : grid → Color) : Prop :=
  ∀ i j : Fin 3,
  (
    (coloring (i, j) = Color.green → 
     (i < 2 → coloring (i.succ, j) ≠ Color.red) ∧ 
     (j < 2 → coloring (i, j.succ) ≠ Color.red)) ∧
    (coloring (i, j) = Color.blue → 
     (i > 0 → coloring (i.pred, j) ≠ Color.red ∧ 
      coloring (i.pred, j) ≠ Color.green))
  )

def count_valid_colorings : Nat :=
  -- Here, "sorry" is used because we are not providing the actual proof steps.
  sorry

theorem num_valid_colorings_is_25 : count_valid_colorings = 25 := by
  sorry

end num_valid_colorings_is_25_l781_781058


namespace logarithm_problem_l781_781272

variable {x k : ℝ}

theorem logarithm_problem (h₁ : log 10 x * log k 10 = 4) (h₂ : k^2 = 100) : x = 10000 := by
  sorry

end logarithm_problem_l781_781272


namespace cost_price_per_meter_l781_781483

variable (total_meters : ℕ)
variable (total_selling_price : ℝ)
variable (loss_per_meter : ℝ)
variable (correct_cost_price_per_meter : ℝ)

theorem cost_price_per_meter 
  (h1 : total_meters = 450)
  (h2 : total_selling_price = 35000)
  (h3 : loss_per_meter = 25)
  (h4 : correct_cost_price_per_meter = 102.78) :
  let selling_price_per_meter := total_selling_price / total_meters in
  let cost_price_per_meter := selling_price_per_meter + loss_per_meter in
  cost_price_per_meter = correct_cost_price_per_meter :=
by
  sorry

end cost_price_per_meter_l781_781483


namespace has_exactly_two_solutions_iff_l781_781662

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l781_781662


namespace min_magnitude_difference_l781_781207

def vector_a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)
def vector_b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
noncomputable def vector_difference (t : ℝ) : ℝ × ℝ × ℝ := 
  let (ax, ay, az) := vector_a t
  let (bx, by, bz) := vector_b t
  (bx - ax, by - ay, bz - az)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := v
  real.sqrt (x^2 + y^2 + z^2)

theorem min_magnitude_difference : ∃ t : ℝ, magnitude (vector_difference t) = real.sqrt 2 :=
sorry

end min_magnitude_difference_l781_781207


namespace angle_between_vectors_l781_781905

open Real InnerProductSpace EuclideanSpace

variables (a b c : EuclideanSpace ℝ (Fin 3))
variables (θ : ℝ)

def norm_eq_three (v : EuclideanSpace ℝ (Fin 3)) := ‖v‖ = 3
def norm_eq_six (v : EuclideanSpace ℝ (Fin 3)) := ‖v‖ = 6

theorem angle_between_vectors {a b c : EuclideanSpace ℝ (Fin 3)} 
  (h1 : norm_eq_three a)
  (h2 : norm_eq_three b)
  (h3 : norm_eq_six c)
  (h4 : a × (a × c) + 2 • b = 0) :
  θ = 35.264 ∨ θ = 144.736 :=
sorry

end angle_between_vectors_l781_781905


namespace johns_overall_profit_l781_781303

def cost_price_grinder : ℕ := 15000
def cost_price_mobile : ℕ := 8000
def loss_percent_grinder : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10

noncomputable def loss_amount_grinder : ℝ := loss_percent_grinder * cost_price_grinder
noncomputable def selling_price_grinder : ℝ := cost_price_grinder - loss_amount_grinder

noncomputable def profit_amount_mobile : ℝ := profit_percent_mobile * cost_price_mobile
noncomputable def selling_price_mobile : ℝ := cost_price_mobile + profit_amount_mobile

noncomputable def total_cost_price : ℝ := cost_price_grinder + cost_price_mobile
noncomputable def total_selling_price : ℝ := selling_price_grinder + selling_price_mobile
noncomputable def overall_profit : ℝ := total_selling_price - total_cost_price

theorem johns_overall_profit :
  overall_profit = 50 := 
by
  sorry

end johns_overall_profit_l781_781303


namespace problem_l781_781711

/-- Define the greatest integer function -/
def greatest_int (t : ℝ) : ℤ := ⌊t⌋

/-- Define f1, g, and f2 functions -/
def f1 (x : ℝ) : ℤ := greatest_int (4 * x)
def g (x : ℝ) : ℝ := 4 * x - greatest_int (4 * x)
def f2 (x : ℝ) : ℤ := f1 (g x)

/-- Proof Statement -/
theorem problem (x : ℝ) : 
  (f1 x = 1 ∧ f2 x = 3) ↔ (7 / 16 ≤ x ∧ x < 1 / 2) :=
by
  sorry

end problem_l781_781711


namespace max_value_f_l781_781192

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * sin x * cos x - sin x ^ 2 + 1 / 2

theorem max_value_f (a : ℝ) (h_symm : f a (π / 6) = f a 0) : ∃ x : ℝ, f (sqrt 3) x = 1 :=
by
  sorry

end max_value_f_l781_781192


namespace geometric_progression_common_ratio_l781_781379

/--
If \( a_1, a_2, a_3 \) are terms of an arithmetic progression with common difference \( d \neq 0 \),
and the products \( a_1 a_2, a_2 a_3, a_3 a_1 \) form a geometric progression,
then the common ratio of this geometric progression is \(-2\).
-/
theorem geometric_progression_common_ratio (a₁ a₂ a₃ d : ℝ) (h₀ : d ≠ 0) (h₁ : a₂ = a₁ + d)
  (h₂ : a₃ = a₁ + 2 * d) (h₃ : (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) :
  (a₂ * a₃) / (a₁ * a₂) = -2 :=
by
  sorry

end geometric_progression_common_ratio_l781_781379


namespace floor_sqrt_fifty_l781_781623

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781623


namespace find_certain_number_l781_781927

theorem find_certain_number (d q r : ℕ) (HD : d = 37) (HQ : q = 23) (HR : r = 16) :
    ∃ n : ℕ, n = d * q + r ∧ n = 867 := by
  sorry

end find_certain_number_l781_781927


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781539

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781539


namespace jake_delay_l781_781110

-- Define the conditions as in a)
def floors_jake_descends : ℕ := 8
def steps_per_floor : ℕ := 30
def steps_per_second_jake : ℕ := 3
def elevator_time_seconds : ℕ := 60 -- 1 minute = 60 seconds

-- Define the statement based on c)
theorem jake_delay (floors : ℕ) (steps_floor : ℕ) (steps_second : ℕ) (elevator_time : ℕ) :
  (floors = floors_jake_descends) →
  (steps_floor = steps_per_floor) →
  (steps_second = steps_per_second_jake) →
  (elevator_time = elevator_time_seconds) →
  (floors * steps_floor / steps_second - elevator_time = 20) :=
by
  intros
  sorry

end jake_delay_l781_781110


namespace point_quadrant_l781_781681

def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0

def quadrant (x y : ℝ) : String :=
  if is_positive x ∧ is_positive y then "First quadrant"
  else if is_negative x ∧ is_positive y then "Second quadrant"
  else if is_negative x ∧ is_negative y then "Third quadrant"
  else if is_positive x ∧ is_negative y then "Fourth quadrant"
  else "On an axis"

theorem point_quadrant (s1 : ℝ) (c2 : ℝ) (h_s1 : is_positive s1) (h_c2 : is_negative c2) :
  quadrant c2 s1 = "Second quadrant" :=
sorry

end point_quadrant_l781_781681


namespace total_amount_is_200_l781_781569

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l781_781569


namespace smallest_positive_integer_solution_l781_781996

theorem smallest_positive_integer_solution 
  (y : ℤ)
  (h₁ : y + 3721 ≡ 803 [MOD 17]) :
  y = 14 :=
by
  sorry

end smallest_positive_integer_solution_l781_781996


namespace perfect_square_add_3_perfect_square_add_1_l781_781644

theorem perfect_square_add_3 (n : ℤ) (k : ℤ) (h : 2^n + 3 = k^2) : n = 0 :=
by sorry

theorem perfect_square_add_1 (n : ℤ) (k : ℤ) (h : 2^n + 1 = k^2) : n = 3 :=
by sorry

end perfect_square_add_3_perfect_square_add_1_l781_781644


namespace train_crosses_platform_time_l781_781468

theorem train_crosses_platform_time (speed_kmph : ℕ) (time_pole_seconds : ℕ) (platform_length_meters : ℕ) :
  let speed_mps := (speed_kmph * 1000) / 3600,
      length_train := time_pole_seconds * speed_mps,
      total_distance := length_train + platform_length_meters,
      time_platform_seconds := total_distance / speed_mps
  in
  (speed_kmph = 90) ∧ (time_pole_seconds = 8) ∧ (platform_length_meters = 650) →
  time_platform_seconds = 34 :=
by
  intros hyp
  apply and.elim hyp
  intro h1
  apply and.elim hyp.right
  intro h2
  intro h3
  sorry

end train_crosses_platform_time_l781_781468


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781546

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781546


namespace ratio_first_part_l781_781462

theorem ratio_first_part (second_part : ℝ) (perc : ℝ) (ratio : perc = 120) (sp : second_part = 5) : (first_part : ℝ) := 
by
  have hp : perc / 100 = 1.2 := by sorry
  let first_part := perc * second_part / 100
  rw hp
  rw sp
  norm_num
  exact 6

end ratio_first_part_l781_781462


namespace more_females_with_spots_than_males_with_horns_l781_781947

theorem more_females_with_spots_than_males_with_horns
  (total_cows : ℕ)
  (M F : ℕ)
  (half_females_spotted : ℕ)
  (half_males_horned : ℕ)
  (h_total : total_cows = 300)
  (h_females_twice_males : F = 2 * M)
  (h_sum_cows : F + M = total_cows)
  (h_half_females_spotted : half_females_spotted = F / 2)
  (h_half_males_horned : half_males_horned = M / 2) :
  half_females_spotted - half_males_horned = 50 :=
begin
  sorry
end

end more_females_with_spots_than_males_with_horns_l781_781947


namespace inequality_proof_l781_781164

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  (1 / (a^2 + 1)) + (1 / (b^2 + 1)) + (1 / (c^2 + 1)) ≤ 9 / 4 :=
by
  sorry

end inequality_proof_l781_781164


namespace minimum_value_of_expression_l781_781188

theorem minimum_value_of_expression {k x1 x2 : ℝ} 
  (h1 : x1 + x2 = -2 * k)
  (h2 : x1 * x2 = k^2 + k + 3) : 
  (x1 - 1)^2 + (x2 - 1)^2 ≥ 8 :=
sorry

end minimum_value_of_expression_l781_781188


namespace inequality_and_equality_equality_condition_l781_781741

theorem inequality_and_equality (a b : ℕ) (ha : a > 1) (hb : b > 2) : a^b + 1 ≥ b * (a + 1) :=
by sorry

theorem equality_condition (a b : ℕ) : a = 2 ∧ b = 3 → a^b + 1 = b * (a + 1) :=
by
  intro h
  cases h
  sorry

end inequality_and_equality_equality_condition_l781_781741


namespace distinct_solution_condition_l781_781659

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l781_781659


namespace marcel_paid_total_amount_l781_781335

variable (pen_cost : ℝ) (briefcase_cost : ℝ) (notebook_cost : ℝ) (calculator_cost : ℝ) (total_cost_before_tax : ℝ) (tax : ℝ) (total_cost_including_tax : ℝ)

def marcel_shopping_problem (pen_cost briefcase_cost notebook_cost calculator_cost total_cost_before_tax tax total_cost_including_tax : ℝ) : Prop :=
  pen_cost = 4 ∧
  briefcase_cost = 5 * pen_cost ∧
  notebook_cost = 2 * pen_cost ∧
  calculator_cost = 3 * notebook_cost ∧
  total_cost_before_tax = pen_cost + briefcase_cost + notebook_cost + calculator_cost ∧
  tax = 0.10 * total_cost_before_tax ∧
  total_cost_including_tax = total_cost_before_tax + tax ∧
  total_cost_including_tax = 61.60

theorem marcel_paid_total_amount :
  marcel_shopping_problem 4 (5 * 4) (2 * 4) (3 * (2 * 4)) (4 + (5 * 4) + (2 * 4) + (3 * (2 * 4))) (0.10 * (4 + (5 * 4) + (2 * 4) + (3 * (2 * 4)))) (61.60) :=
by {
  unfold marcel_shopping_problem,
  repeat {split},
  repeat {norm_num},
}

end marcel_paid_total_amount_l781_781335


namespace sides_of_triangle_inequality_l781_781001

theorem sides_of_triangle_inequality (a b c : ℝ) (h : a + b > c) : a + b > c := 
by 
  exact h

end sides_of_triangle_inequality_l781_781001


namespace roots_of_equation_l781_781645

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l781_781645


namespace found_bottle_caps_is_correct_l781_781125

def initial_bottle_caps : ℕ := 6
def total_bottle_caps : ℕ := 28

theorem found_bottle_caps_is_correct : total_bottle_caps - initial_bottle_caps = 22 := by
  sorry

end found_bottle_caps_is_correct_l781_781125


namespace ball_distribution_l781_781237

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781237


namespace find_x_l781_781026

theorem find_x (x : ℝ) (h : (0.4 + x) / 2 = 0.2025) : x = 0.005 :=
by
  sorry

end find_x_l781_781026


namespace balls_in_boxes_l781_781222

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781222


namespace repeat_2013_divisible_by_9_l781_781039

theorem repeat_2013_divisible_by_9 :
  (∀ n : ℕ, (let repeated_sum := n * (2 + 0 + 1 + 3) in repeated_sum % 9 = 0) ↔ (∃ k : ℕ, 201320132013 = k * 9)) :=
by sorry

end repeat_2013_divisible_by_9_l781_781039


namespace num_valid_pairs_l781_781826

/-- 
Let S(n) denote the sum of the digits of a natural number n.
Define the predicate to check if the pair (m, n) satisfies the given conditions.
-/
def S (n : ℕ) : ℕ := (toString n).foldl (fun acc ch => acc + ch.toNat - '0'.toNat) 0

def valid_pair (m n : ℕ) : Prop :=
  m < 100 ∧ n < 100 ∧ m > n ∧ m + S n = n + 2 * S m

/-- 
Theorem: There are exactly 99 pairs (m, n) that satisfy the given conditions.
-/
theorem num_valid_pairs : ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 99 ∧
  ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
sorry

end num_valid_pairs_l781_781826


namespace parallel_line_perpendicular_plane_perpendicular_lines_l781_781182

-- Definitions
variable {Point Line Plane : Type}

-- Conditions
variable (a b : Line) (α : Plane)
variable [HasPerpendicular Line Plane] [HasParallel Line Plane]

-- Problem Statement
theorem parallel_line_perpendicular_plane_perpendicular_lines :
  (a ∥ α) → (b ⟂ α) → (a ⟂ b) :=
by
  sorry

end parallel_line_perpendicular_plane_perpendicular_lines_l781_781182


namespace find_cost_of_books_l781_781265

theorem find_cost_of_books
  (C_L C_G1 C_G2 : ℝ)
  (h1 : C_L + C_G1 + C_G2 = 1080)
  (h2 : 0.9 * C_L = 1.15 * C_G1 + 1.25 * C_G2)
  (h3 : C_G1 + C_G2 = 1080 - C_L) :
  C_L = 784 :=
sorry

end find_cost_of_books_l781_781265


namespace has_exactly_two_solutions_iff_l781_781660

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l781_781660


namespace floor_sqrt_50_l781_781618

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781618


namespace evaluate_nested_f_l781_781159

def f (x : ℝ) : ℝ :=
if x > 0 then x + 1 else
if x = 0 then 2 else 0

theorem evaluate_nested_f :
  f (f (f (-1))) = 3 :=
by
  sorry

end evaluate_nested_f_l781_781159


namespace total_money_together_l781_781566

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l781_781566


namespace all_round_trips_miss_capital_same_cost_l781_781407

open Set

variable {City : Type} [Inhabited City]
variable {f : City → City → ℝ}
variable (capital : City)
variable (round_trip_cost : List City → ℝ)

-- The conditions
axiom flight_cost_symmetric (A B : City) : f A B = f B A
axiom equal_round_trip_cost (R1 R2 : List City) :
  (∀ (city : City), city ∈ R1 ↔ city ∈ R2) → 
  round_trip_cost R1 = round_trip_cost R2

noncomputable def constant_trip_cost := 
  ∀ (cities1 cities2 : List City),
     (∀ (city : City), city ∈ cities1 ↔ city ∈ cities2) →
     ¬(capital ∈ cities1 ∨ capital ∈ cities2) →
     round_trip_cost cities1 = round_trip_cost cities2

-- Goal to prove
theorem all_round_trips_miss_capital_same_cost : constant_trip_cost capital round_trip_cost := 
  sorry

end all_round_trips_miss_capital_same_cost_l781_781407


namespace f_four_times_even_l781_781827

variable (f : ℝ → ℝ) (x : ℝ)

-- Definition stating f is an odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem f_four_times_even (h : is_odd f) : is_even (f (f (f (f x)))) :=
by sorry

end f_four_times_even_l781_781827


namespace sequence_a_b_10_l781_781347

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end sequence_a_b_10_l781_781347


namespace forest_trees_properties_l781_781514

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781514


namespace floor_sqrt_50_l781_781605

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781605


namespace range_of_a_l781_781708

open Real

def prop_p (a x : ℝ) : Prop := x ^ 2 - (2 * a + 4) * x + a ^ 2 + 4 * a < 0
def prop_q (x : ℝ) : Prop := (x - 2) * (x - 3) < 0 

theorem range_of_a :
  (∀ a, (¬(prop_p a x) → ¬(prop_q x)) → ∀ x, 2 > a ∧ 3 ≤ a + 4) →
  (-1 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l781_781708


namespace odell_kershaw_meetings_l781_781348

noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

noncomputable def angular_speed (speed circumference : ℝ) : ℝ := (speed / circumference) * 2 * Real.pi

noncomputable def meeting_times (time : ℝ) (angular_speed1 angular_speed2 : ℝ) : ℕ := 
  ⌊time / ((2 * Real.pi) / (angular_speed1 + angular_speed2))⌋

theorem odell_kershaw_meetings 
  (time : ℝ) (speed1 radius1 speed2 radius2 : ℝ) 
  (h_time : time = 35) 
  (h_speed1 : speed1 = 260) 
  (h_radius1 : radius1 = 55)
  (h_speed2 : speed2 = 310) 
  (h_radius2 : radius2 = 65) : 
  meeting_times time (angular_speed speed1 (circumference radius1)) (angular_speed speed2 (circumference radius2)) = 52 := 
sorry

end odell_kershaw_meetings_l781_781348


namespace determine_quadratic_function_l781_781995

def quadratic_function_satisfies_condition (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (f : ℝ → ℝ) (f := λ x, x^2 + a * x + b),
    f(f(x) + x) / f(x) = x^2 + 2023 * x + 1776

theorem determine_quadratic_function : 
  quadratic_function_satisfies_condition 2021 (-246) :=
sorry

end determine_quadratic_function_l781_781995


namespace sum_of_coefficients_l781_781745

theorem sum_of_coefficients :
  let a : ℕ → ℝ := λ k, (PolynomialExpansion (1 - 2 * x) ^ 2009) k in -- Using a hypothetical PolynomialExpansion function to represent coefficients
  (a 0 + a 1 + ... + a 2009) = -1 :=
by 
  sorry

end sum_of_coefficients_l781_781745


namespace limit_r_as_m_approaches_zero_l781_781812

-- Defining the function L(m) which gives the x-coordinate of the left endpoint of the intersection
def L (m : ℝ) : ℝ := -2 - real.sqrt (m + 7)

-- Defining the function r(m) as described in the problem
def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Stating the limit equality that we need to prove
theorem limit_r_as_m_approaches_zero :
  filter.tendsto (λ m, r m) (nhds 0) (nhds (1 / real.sqrt 7)) :=
sorry

end limit_r_as_m_approaches_zero_l781_781812


namespace parallel_line_proof_l781_781785

noncomputable def triangle_angle_conditions (A B C D : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  let α := some_angle_in_triangle ABC
  let β := 2 * α
  let γ := 180 - 3 * α
  ∃ (triangle ABC : triangle), ∃ (is_right_triangle : triangle.angle B ≠ 90),
  ∃ (interior_bisector B : line), ∃ (perp_bisector AC : line),
  ∃ (D : point), 
  is_interior_bisector B intersects perp_bisector AC at D ∧ 
  angle_of_triangle_at_A = α ∧
  angle_of_triangle_at_B = β ∧
  angle_of_triangle_at_C = γ

theorem parallel_line_proof (A B C D : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :
  triangle_angle_conditions A B C D →
  line_parallel AB CD :=
by
  sorry

end parallel_line_proof_l781_781785


namespace total_attendance_l781_781908

variable (S C : ℕ)
variable (total_sales : ℕ = 2280)
variable (couple_tickets_sold : ℕ = 16)
variable (single_ticket_price : ℕ = 20)
variable (couple_ticket_price : ℕ = 35)

theorem total_attendance :
    couple_tickets_sold = 16 →
    single_ticket_price * S + couple_ticket_price * couple_tickets_sold = total_sales →
    S = (total_sales - couple_ticket_price * couple_tickets_sold) / single_ticket_price →
    S + 2 * couple_tickets_sold = 118 :=
by
  intros h1 h2 h3
  rw [h1] at h2
  rw [h1] at h3
  have S_def : S = 86 := h3
  rw [S_def] at *
  sorry

end total_attendance_l781_781908


namespace dealer_overall_profit_percentage_l781_781952

-- Definitions of conditions
def purchase_price_A := 15 * 25
def purchase_price_B := 20 * 35
def discount_A := 0.05 * purchase_price_A
def discount_B := 0.10 * purchase_price_B
def total_cost_price := purchase_price_A + purchase_price_B
def total_discount := discount_A + discount_B
def total_cost_after_discount := total_cost_price - total_discount
def selling_price_A := 12 * 33
def selling_price_B := 18 * 45
def total_selling_price := selling_price_A + selling_price_B
def profit_or_loss := total_selling_price - total_cost_after_discount
def profit_or_loss_percentage := (profit_or_loss / total_cost_after_discount) * 100

-- The theorem to prove
theorem dealer_overall_profit_percentage : profit_or_loss_percentage ≈ 22.28 :=
by sorry


end dealer_overall_profit_percentage_l781_781952


namespace forest_trees_properties_l781_781511

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781511


namespace balls_in_boxes_l781_781230

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781230


namespace column_product_neg_one_l781_781832

open Finset

theorem column_product_neg_one {n : ℕ} (h : n = 1000) 
   (x y : Fin n → ℝ) 
   (h_distinct : (x.to_list ++ y.to_list).nodup) 
   (h_row_prod : ∀ i, (univ.prod (λ j, x i + y j)) = 1) :
   ∀ j, (univ.prod (λ i, x i + y j)) = -1 :=
by 
  sorry

end column_product_neg_one_l781_781832


namespace non_negative_number_is_nine_values_of_a_and_m_l781_781052

theorem non_negative_number_is_nine (a : ℝ) (h1 : 2 * a - 1 = sqrt 9) (h2 : a - 5 = sqrt 9) : 
  (2 * a - 1) ^ 2 = 9 :=
by
  sorry

theorem values_of_a_and_m (a m : ℝ) (h1 : a - 1 = sqrt m) (h2 : 5 - 2 * a = sqrt m) : 
  (a = 2 → m = 1) ∧ (a = 4 → m = 9) :=
by
  sorry

end non_negative_number_is_nine_values_of_a_and_m_l781_781052


namespace prove_statements_true_l781_781433

def problem_conditions : Prop :=
  (∃ (n : ℤ), 24 = 4 * n) ∧        -- Condition A: 4 is a factor of 24
  (∃ (m : ℤ), 90 = 30 * m) ∧        -- Condition C: 30 is a divisor of 90
  (∃ (p : ℤ), 200 = 10 * p)         -- Condition E: 10 is a factor of 200

def problem_a_true : Prop :=
  ∃ (n : ℤ), 24 = 4 * n

def problem_e_true : Prop :=
  ∃ (p : ℤ), 200 = 10 * p

def problem_a_and_e_true : Prop :=
  problem_a_true ∧ problem_e_true

theorem prove_statements_true : problem_conditions → problem_a_and_e_true :=
by
  intro h
  cases h with hA hRest
  cases hRest with hC hE
  exact and.intro hA hE

end prove_statements_true_l781_781433


namespace distinct_solution_condition_l781_781656

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l781_781656


namespace sequence_21st_term_l781_781793

theorem sequence_21st_term :
  ∀ n : ℕ, (a : ℕ → ℝ) (h1 : ∀ n, a n = (n + 3) / 2), a 21 = 12 :=
by
  intros
  simp only [h1]
  norm_num
  sorry

end sequence_21st_term_l781_781793


namespace find_digits_l781_781587

-- Define real repeating decimal as a rational number
def repeating_decimal_ab (a b : ℕ) : ℚ := (10 * a + b) / 99
def repeating_decimal_abc (a b c : ℕ) : ℚ := (100 * a + 10 * b + c) / 999

-- Define the condition that their sum is 12/13 and that a, b, and c are distinct
noncomputable def condition (a b c : ℕ) : Prop :=
  (repeating_decimal_ab a b + repeating_decimal_abc a b c = 12 / 13) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Define the theorem that proves the digits
theorem find_digits : ∃ a b c : ℕ, condition a b c ∧ a = 4 ∧ b = 6 ∧ c = 3 :=
by
  existsi (4 : ℕ)
  existsi (6 : ℕ)
  existsi (3 : ℕ)
  have h1 : 4 ≠ 6 ∧ 6 ≠ 3 ∧ 4 ≠ 3 := by simp
  have h2 : repeating_decimal_ab 4 6 + repeating_decimal_abc 4 6 3 = 12 / 13 := by sorry
  exact ⟨⟨h2, h1⟩, rfl, rfl, rfl⟩

end find_digits_l781_781587


namespace inequality_f_x_l781_781191

def f (x : ℝ) : ℝ :=
  if x < 2 then x else x - 1/x

theorem inequality_f_x (x : ℝ) : f x < 8/3 ↔ x < 3 :=
sorry

end inequality_f_x_l781_781191


namespace polynomial_min_value_l781_781897

noncomputable def poly (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

theorem polynomial_min_value : 
  ∃ x y : ℝ, poly x y = -18 :=
by
  sorry

end polynomial_min_value_l781_781897


namespace ball_distribution_l781_781241

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781241


namespace random_point_probability_parallelogram_l781_781848

noncomputable def parallelogram_vertices : set (ℝ × ℝ) :=
  {(4, 4), (-2, -2), (-8, -2), (-2, 4)}

noncomputable def parallelogram_EFGH : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a b, (p = a • (4, 4) + b • (-2, -2)) ∧
    (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)}

noncomputable def line_y_neg1 : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -1}

noncomputable def region_between_xaxis_and_y_neg1 : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.2 ∧ p.2 ≤ -1}

theorem random_point_probability_parallelogram : 
  ∀ (E F G H : ℝ × ℝ), 
    parallelogram_EFGH = parallelogram_vertices →
    (∃ P : set (ℝ × ℝ), P ∈ (region_between_xaxis_and_y_neg1 ∩ parallelogram_EFGH) →
    (measure_theory.measure P / measure_theory.measure parallelogram_EFGH) = 1 / 6) :=
sorry

end random_point_probability_parallelogram_l781_781848


namespace lcm_fractions_correct_l781_781423

noncomputable def lcm_fractions : Rat → Rat → Rat → Rat :=
  λ a b c, (1 : Rat) / Rat.lcm a⁻¹ b⁻¹ c⁻¹

theorem lcm_fractions_correct (x : ℤ) (hx : x ≠ 0) :
  lcm_fractions (1 / x) (1 / (3 * x)) (1 / (4 * x)) = 1 / (12 * x) :=
by
  have h1 : 1 / x = (x)⁻¹ := by norm_num -- Reciprocal transformation
  have h3 : 1 / (3 * x) = (3 * x)⁻¹ := by norm_num -- Reciprocal transformation
  have h4 : 1 / (4 * x) = (4 * x)⁻¹ := by norm_num -- Reciprocal transformation
  rw [lcm_fractions, h1, h3, h4]
  have := Rat.lcm (x)⁻¹ (3 * x)⁻¹ (4 * x)⁻¹
  simp only [Rat.lcm_computes_lcm]
  -- More steps would follow to show this equals 12x⁻¹
  sorry

end lcm_fractions_correct_l781_781423


namespace decreasing_interval_l781_781588

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 2

theorem decreasing_interval : ∀ x : ℝ, (-2 < x ∧ x < 0) → (deriv f x < 0) := 
by
  sorry

end decreasing_interval_l781_781588


namespace binom_n_n_minus_2_l781_781924

noncomputable def factorial : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def binom (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem binom_n_n_minus_2 (n : ℕ) (h : 0 < n) : 
  binom n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binom_n_n_minus_2_l781_781924


namespace rosie_purchase_price_of_art_piece_l781_781012

-- Define the conditions as hypotheses
variables (P : ℝ)
variables (future_value increase : ℝ)

-- Given conditions
def conditions := future_value = 3 * P ∧ increase = 8000 ∧ increase = future_value - P

-- The statement to be proved
theorem rosie_purchase_price_of_art_piece (h : conditions P future_value increase) : P = 4000 :=
sorry

end rosie_purchase_price_of_art_piece_l781_781012


namespace maximum_intersection_points_l781_781675

-- Define the problem conditions and the theorem
theorem maximum_intersection_points (n : ℕ) (h₁ : n > 2) 
  (C₁ C₂ C₃ : set (ℝ × ℝ)) (h₂ : is_convex_ngon C₁ n) 
  (h₃ : is_convex_ngon C₂ n) 
  (h₄ : is_convex_ngon C₃ n) 
  (h₅ : (C₁ ∩ C₂).finite) 
  (h₆ : (C₂ ∩ C₃).finite) 
  (h₇ : (C₁ ∩ C₃).finite) :
  (C₁ ∩ C₂ ∩ C₃).card ≤ 3 * n / 2 := 
sorry

end maximum_intersection_points_l781_781675


namespace four_P_plus_five_square_of_nat_l781_781843

theorem four_P_plus_five_square_of_nat 
  (a b : ℕ)
  (P : ℕ)
  (hP : P = (Nat.lcm a b) / (a + 1) + (Nat.lcm a b) / (b + 1))
  (h_prime : Nat.Prime P) : 
  ∃ n : ℕ, 4 * P + 5 = (2 * n + 1) ^ 2 :=
by
  sorry

end four_P_plus_five_square_of_nat_l781_781843


namespace square_lines_b_product_l781_781388

theorem square_lines_b_product : 
  (∃ (b : ℝ), let side_length := 3 in 
    (x = 2 ∧ x = b ∧ abs (b - 2) = side_length) → (b = -1 ∨ b = 5)) → 
  (-1 * 5 = -5) :=
by sorry

end square_lines_b_product_l781_781388


namespace wheel_revolutions_l781_781131

-- Define the diameter of the wheel
def diameter (d : ℝ) := (d = 8)

-- Define the distance in feet to be traveled
def distance_in_feet (D : ℝ) := (D = 2640)

-- Define the number of revolutions
noncomputable def number_of_revolutions (N : ℝ) := 
  (N = 330 / real.pi)

-- The main theorem statement
theorem wheel_revolutions (d D N : ℝ) (h1 : diameter d) (h2 : distance_in_feet D) :
  number_of_revolutions N := 
sorry

end wheel_revolutions_l781_781131


namespace area_of_rectangle_with_diagonal_length_l781_781963

variable (x : ℝ)

def rectangle_area_given_diagonal_length (x : ℝ) : Prop :=
  ∃ (w l : ℝ), l = 3 * w ∧ w^2 + l^2 = x^2 ∧ (w * l = (3 / 10) * x^2)

theorem area_of_rectangle_with_diagonal_length (x : ℝ) :
  rectangle_area_given_diagonal_length x :=
sorry

end area_of_rectangle_with_diagonal_length_l781_781963


namespace inscribed_circle_radius_1304_l781_781397

noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let term1 := (1 / a) + (1 / b) + (1 / c)
  let term2 := 2 * Real.sqrt((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))
  1 / (term1 + term2)

theorem inscribed_circle_radius_1304 : 
  inscribed_circle_radius 5 10 15 ≈ 1.304 :=
by
  sorry

end inscribed_circle_radius_1304_l781_781397


namespace find_p_plus_q_l781_781416

noncomputable def sides := (15 : ℝ, 36 : ℝ, 39 : ℝ)
noncomputable def semi_perimeter := (sides.1 + sides.2 + sides.3) / 2
noncomputable def area_triangle := 
  real.sqrt (semi_perimeter * (semi_perimeter - sides.1) * (semi_perimeter - sides.2) * (semi_perimeter - sides.3))

noncomputable def ω_1 := 36
noncomputable def ω_2 := 18

def condition_1 (α β : ℝ) : ω_1 * α - ω_1^2 * β = 0 := 
by
  sorry

def condition_2 (α β : ℝ) : 
  area_triangle / 2 = α * ω_2 - β * ω_2^2 := 
by
  sorry

noncomputable def β := (135 : ℝ) / 324
noncomputable def rational_β := 
  let (p, q) := (5, 12) in 
  (p : ℚ) / (q : ℚ)

theorem find_p_plus_q : β = rational_β → 5 + 12 = 17 := 
by
  sorry

end find_p_plus_q_l781_781416


namespace pierre_ate_100_grams_l781_781457

-- Define the conditions:
def cake_weight : Real := 400
def parts : Int := 8
def nathalie_fraction : Real := 1/8
def nathalie : Real := (nathalie_fraction * cake_weight)
def double (x : Real) := 2 * x

-- Statement to prove Pierre's consumption:
theorem pierre_ate_100_grams : double nathalie = 100 :=
by
  -- The detailed proof will be filled in here.
  sorry

end pierre_ate_100_grams_l781_781457


namespace probability_of_multiplying_to_120_l781_781778

def set_of_numbers := {2, 5, 10, 15, 24, 30, 60}

def multiply_to_multiple_of_120 (a b : ℕ) : Prop :=
  (a * b) % 120 = 0

noncomputable def favorable_pairs : ℕ :=
  {n | n ∈ set_of_numbers}.to_finset.filter (λ n, ∃ m, m ∈ set_of_numbers ∧ multiply_to_multiple_of_120 n m ∧ n ≠ m).card / 2 -- each pair counted twice

noncomputable def total_pairs : ℕ :=
  (set_of_numbers.to_finset.card * (set_of_numbers.to_finset.card - 1)) / 2

theorem probability_of_multiplying_to_120 : (favorable_pairs : ℚ) / total_pairs = 4 / 21 :=
by
  sorry

end probability_of_multiplying_to_120_l781_781778


namespace range_2y_plus_3x_l781_781809

-- Define the points and sides of the triangle
variable (A B C P Q : Point ℝ)
variable (AB BC CA : ℝ)
variable (AP AQ PQ x y : ℝ)

-- Given conditions
axiom h1 : BC = 5
axiom h2 : CA = 3
axiom h3 : AB = 4
axiom h4 : ∃ (P Q: Point ℝ), P ∈ Segment A B ∧ Q ∈ Segment A C
axiom h5 : Area (triangle A P Q) = (1/2) * Area (triangle A B C)
axiom h6 : x = perp_dist (midpoint P Q) (line A B)
axiom h7 : y = perp_dist (midpoint P Q) (line A C)

-- The expression of 2y + 3x
def expr_2y_plus_3x := 2 * y + 3 * x

-- The correct range of the values of 2y + 3x
theorem range_2y_plus_3x : 6 ≤ expr_2y_plus_3x ∧ expr_2y_plus_3x ≤ 6.5 := sorry

end range_2y_plus_3x_l781_781809


namespace total_money_together_l781_781567

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l781_781567


namespace sum_of_coefficients_l781_781744

theorem sum_of_coefficients :
  let a : ℕ → ℝ := λ k, (PolynomialExpansion (1 - 2 * x) ^ 2009) k in -- Using a hypothetical PolynomialExpansion function to represent coefficients
  (a 0 + a 1 + ... + a 2009) = -1 :=
by 
  sorry

end sum_of_coefficients_l781_781744


namespace train_crossing_time_l781_781968

theorem train_crossing_time
  (train_length : ℕ) (train_speed : ℝ) (person_speed : ℝ) (relative_speed_factor : ℝ) :
    train_length = 300 →
    train_speed = 80 →
    person_speed = 16 →
    relative_speed_factor = (5 / 18) →
    ((↑train_length:ℝ) / ((train_speed - person_speed) * relative_speed_factor) ≈ 16.87) := 
begin
  intros h1 h2 h3 h4,
  sorry
end

end train_crossing_time_l781_781968


namespace floor_sqrt_50_l781_781606

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781606


namespace forest_trees_properties_l781_781519

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781519


namespace more_females_with_spots_than_males_with_horns_l781_781948

theorem more_females_with_spots_than_males_with_horns
  (total_cows : ℕ)
  (M F : ℕ)
  (half_females_spotted : ℕ)
  (half_males_horned : ℕ)
  (h_total : total_cows = 300)
  (h_females_twice_males : F = 2 * M)
  (h_sum_cows : F + M = total_cows)
  (h_half_females_spotted : half_females_spotted = F / 2)
  (h_half_males_horned : half_males_horned = M / 2) :
  half_females_spotted - half_males_horned = 50 :=
begin
  sorry
end

end more_females_with_spots_than_males_with_horns_l781_781948


namespace function_passes_through_point_l781_781385

theorem function_passes_through_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : ∃ x y, x = 3 ∧ y = 1 ∧ y = a^(x - 3) :=
by 
  use (3, 1)
  sorry

end function_passes_through_point_l781_781385


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781535

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781535


namespace potato_bag_weight_l781_781041

theorem potato_bag_weight (w : ℕ) (h₁ : w = 36) : w = 36 :=
by
  sorry

end potato_bag_weight_l781_781041


namespace find_a_l781_781210

axiom nat_star (n: ℕ) : ℕ := n + 1

theorem find_a (n : ℕ) (x : ℝ) (h1: n > 0) (h2: 0 < x) : x + (n^n / x^n) >= n + 1 -> a = n^n :=
by sorry

end find_a_l781_781210


namespace integral_fx_equals_two_l781_781270

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then x^3 + Real.sin x else
  if 1 < x ∧ x ≤ 2 then 2 else 0

theorem integral_fx_equals_two :
  ∫ x in -1..2, f x = 2 :=
by sorry

end integral_fx_equals_two_l781_781270


namespace sum_of_possible_k_values_l781_781289

theorem sum_of_possible_k_values :
  ∑ (k : ℕ) in {k | ∃ j : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = 1 / 2}, k = 13 :=
by sorry

end sum_of_possible_k_values_l781_781289


namespace balls_in_boxes_l781_781233

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781233


namespace find_cherry_pirozhok_optimally_l781_781076

def arrangement : Type := List String

def isCherry (p : String) : Prop := p = "cherry"
def isNotCherry (p : String) : Prop := p ≠ "cherry"

noncomputable def findCherryPirozhok (arr : arrangement) : Nat :=
  if h : ¬arr.contains "cherry" then 7
  else if arr.headI = "cherry" then 0
  else if arr.tailI.headI = "cherry" then 1
  else 2

theorem find_cherry_pirozhok_optimally (arr : arrangement) 
  (h1 : arr.length = 7) 
  (h2 : arr.count "rice" = 3) 
  (h3 : arr.count "cabbage" = 3) 
  (h4 : arr.count "cherry" = 1) : 
  findCherryPirozhok arr ≤ 2 :=
sorry

end find_cherry_pirozhok_optimally_l781_781076


namespace ellipse_equation_hyperbola_equation_l781_781152

-- Definitions based on the conditions:
def isEllipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ∃ f : ℝ, f = sqrt 2 ∧ ∃ d : ℝ, d = 2 * sqrt 2 ∧
  (abs f < a) ∧ (d = a^2 / f)

def ellipseStandardEquation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def isHyperbola (x y : ℝ) : Prop :=
  ∃ a b : ℝ, ∃ λ : ℝ, λ ≠ 0 ∧
  (y = 2 * x ∨ y = -2 * x) ∧
  (x,y) = (sqrt 2, 2) ∧
  λ = x^2 - 2 ∧
  x^2 - y^2 / 4 = λ

def hyperbolaStandardEquation (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

-- Theorems (statements) to be proved:
theorem ellipse_equation (x y : ℝ) :
  (isEllipse 2 sqrt 2) → ellipseStandardEquation x y 2 sqrt 2 := sorry

theorem hyperbola_equation (x y : ℝ) :
  isHyperbola x y → hyperbolaStandardEquation x y := sorry

end ellipse_equation_hyperbola_equation_l781_781152


namespace find_k_l781_781896

theorem find_k 
    (x y k : ℝ)
    (h1 : 1.5 * x + y = 20)
    (h2 : -4 * x + y = k)
    (hx : x = -6) :
    k = 53 :=
by
  sorry

end find_k_l781_781896


namespace bucket_problem_l781_781909

theorem bucket_problem 
  (C : ℝ) -- original capacity of the bucket
  (N : ℕ) -- number of buckets required to fill the tank with the original bucket size
  (h : N * C = 25 * (2/5) * C) : 
  N = 10 :=
by
  sorry

end bucket_problem_l781_781909


namespace monthly_salary_l781_781361

variable {S : ℝ}

-- Conditions based on the problem description
def spends_on_food (S : ℝ) : ℝ := 0.40 * S
def spends_on_house_rent (S : ℝ) : ℝ := 0.20 * S
def spends_on_entertainment (S : ℝ) : ℝ := 0.10 * S
def spends_on_conveyance (S : ℝ) : ℝ := 0.10 * S
def savings (S : ℝ) : ℝ := 0.20 * S

-- Given savings
def savings_amount : ℝ := 2500

-- The proof statement for the monthly salary
theorem monthly_salary (h : savings S = savings_amount) : S = 12500 := by
  sorry

end monthly_salary_l781_781361


namespace expectation_bound_inequality_l781_781316

noncomputable theory

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define the non-negative random variable ξ
variable (ξ : Ω → ℝ) (hxi_nonneg : ∀ ω, 0 ≤ ξ ω)
-- Define ε and a with the given constraints
variable (ε : ℝ) (hε_pos : 0 < ε) (a : ℝ) (ha_nonneg : 0 ≤ a)
-- Define the probability condition
variable (hp : Probability.measure (Event.set (λ ω, ξ ω > ε)) ≤ a)

-- Define the expectation conditions
variable (h_exp_defined : ∀ ⦃f : Ω → ℝ⦄, Integrable f → E f = ∫⁻ x, f x ∂(Probability.measure))

-- Define the required inequality to be proven
theorem expectation_bound_inequality :
  E ξ ≤
  ε / (1 - ( (a * E (λ ω, (ξ ω)^2)) ^ (1/2) / E (λ ω, ξ ω))) :=
sorry

end expectation_bound_inequality_l781_781316


namespace floor_sqrt_50_l781_781598

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781598


namespace num_ways_distribute_balls_l781_781254

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781254


namespace minimize_max_value_l781_781383

theorem minimize_max_value :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ (3 * Real.pi) / 2 → 
  ∃ A B : ℝ, (∀ x, |cos(x)^2 + 2 * sin(x) * cos(x) - sin(x)^2 + A * x + B| ≤ sqrt(2) + (|B|)) → 
  A = 0 ∧ B = 0) :=
sorry

end minimize_max_value_l781_781383


namespace allowance_spent_l781_781360

variable (A x y : ℝ)
variable (h1 : x = 0.20 * (A - y))
variable (h2 : y = 0.05 * (A - x))

theorem allowance_spent : (x + y) / A = 23 / 100 :=
by 
  sorry

end allowance_spent_l781_781360


namespace platform_length_l781_781451

theorem platform_length
  (L_train : ℕ) (T_platform : ℕ) (T_pole : ℕ) (P : ℕ)
  (h1 : L_train = 300)
  (h2 : T_platform = 39)
  (h3 : T_pole = 10)
  (h4 : L_train / T_pole * T_platform = L_train + P) :
  P = 870 := 
sorry

end platform_length_l781_781451


namespace geometric_sequence_not_sufficient_nor_necessary_l781_781168

theorem geometric_sequence_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) → 
  (¬ (q > 1 → ∀ n : ℕ, a n < a (n + 1))) ∧ (¬ (∀ n : ℕ, a n < a (n + 1) → q > 1)) :=
by
  sorry

end geometric_sequence_not_sufficient_nor_necessary_l781_781168


namespace derivative_condition_l781_781973

noncomputable def fA (x : ℝ) : ℝ := Real.exp x
noncomputable def fB (x : ℝ) : ℝ := -x^2
noncomputable def fC (x : ℝ) : ℝ := 1 / x
noncomputable def fD (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem derivative_condition :
  (∀ x, (0 < Real.exp x)) ∧ 
  (∀ x, (¬ ∀ x, -2 * x < 0)) ∧ 
  (∀ x, -1 / x^2 < 0) ∧
  (∀ x, (¬ ∀ x, 1 / (x * Real.log 2) < 0)) 
→  (y = fC) := 
by
  sorry

end derivative_condition_l781_781973


namespace cube_division_l781_781069

/-- A cube of edge 5 cm is cut into N smaller cubes, and not all cubes are the same size. 
Each edge of the smaller cubes is a whole number of centimeters. 
Prove that the total number of smaller cubes N is 118. -/
theorem cube_division : 
  ∃ N, 
    (∀ (e: ℕ), e ∈ {1, 5} → (∃ k: ℕ, k * e^3 ≤ 5^3) ∧ 
    (k ≠ 0 ∨ e ≠ 5)) ∧
    (∀ (cubes : list ℕ), ∃ r: ℕ, ∃ unit_cubes: ℕ, 
      r = (5^3 - 8) ∧ 
      cubes = r :: unit_cubes :: []) ∧
    N = 118 :=
by {
sorry
}

end cube_division_l781_781069


namespace part1_part2_l781_781190

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part1 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a ≥ 0)) ↔ (0 < a ∧ a ≤ 2) := sorry

theorem part2 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (x - 1) * f x a ≥ 0) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_part2_l781_781190


namespace sum_of_T_12_digits_eq_three_l781_781985

noncomputable def T_12 : ℕ :=
  Nat.find (λ t, (finset.range 13).filter (λ k, k ≠ 0 ∧ t % k = 0).card ≥ 6) -- find smallest t

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum -- sum of digits base 10

theorem sum_of_T_12_digits_eq_three : sum_of_digits T_12 = 3 := by
  sorry

end sum_of_T_12_digits_eq_three_l781_781985


namespace fewest_posts_required_for_garden_l781_781482

def rectangular_garden (length width area rockwall_side : ℕ) : Prop :=
  length * width = area ∧
  rockwall_side * width = area ∧
  rockwall_side = 70 ∧
  length = 30 

def posts_per_side (side_length post_interval : ℕ) : ℕ :=
  (side_length / post_interval) + 1

theorem fewest_posts_required_for_garden :
  ∃ p : ℕ, p = 14 ∧ ∀ length width,
  rectangular_garden length width 2100 70 →
  p = posts_per_side 70 10 + (posts_per_side 30 10 - 1) * 2 :=
begin
  sorry
end

end fewest_posts_required_for_garden_l781_781482


namespace jackie_break_duration_l781_781299

noncomputable def push_ups_no_breaks : ℕ := 30

noncomputable def push_ups_with_breaks : ℕ := 22

noncomputable def total_breaks : ℕ := 2

theorem jackie_break_duration :
  (5 * 6 - push_ups_with_breaks) * (10 / 5) / total_breaks = 8 := by
-- Given that
-- 1) Jackie does 5 push-ups in 10 seconds
-- 2) Jackie takes 2 breaks in one minute and performs 22 push-ups
-- We need to prove the duration of each break
sorry

end jackie_break_duration_l781_781299


namespace strawberries_per_person_l781_781308

noncomputable def total_strawberries (baskets : ℕ) (strawberries_per_basket : ℕ) : ℕ :=
  baskets * strawberries_per_basket

noncomputable def kimberly_strawberries (brother_strawberries : ℕ) : ℕ :=
  8 * brother_strawberries

noncomputable def parents_strawberries (kimberly_strawberries : ℕ) : ℕ :=
  kimberly_strawberries - 93

noncomputable def total_family_strawberries (kimberly : ℕ) (brother : ℕ) (parents : ℕ) : ℕ :=
  kimberly + brother + parents

noncomputable def equal_division (total_strawberries : ℕ) (people : ℕ) : ℕ :=
  total_strawberries / people

theorem strawberries_per_person :
  let brother_baskets := 3
  let strawberries_per_basket := 15
  let brother_strawberries := total_strawberries brother_baskets strawberries_per_basket
  let kimberly_straw := kimberly_strawberries brother_strawberries
  let parents_straw := parents_strawberries kimberly_straw
  let total := total_family_strawberries kimberly_straw brother_strawberries parents_straw
  equal_division total 4 = 168 :=
by
  simp [total_strawberries, kimberly_strawberries, parents_strawberries, total_family_strawberries, equal_division]
  sorry

end strawberries_per_person_l781_781308


namespace log_3125_between_l781_781004

theorem log_3125_between :
  let c := 4
  let d := 6
  let log_base_5 (n : ℕ) := Real.log n / Real.log 5
  in log_base_5 5^4 < log_base_5 5^5 ∧ log_base_5 5^5 < log_base_5 5^6 →
     c + d = 10 := 
by
  intros
  sorry

end log_3125_between_l781_781004


namespace iterative_mean_difference_l781_781977

def iterative_mean (xs : List ℝ) : ℝ :=
  xs.tail.foldl (λ avg x => (avg + x) / 2) xs.head!

theorem iterative_mean_difference :
  let nums := [1, 3, 5, 7, 9].map (λ x => x : ℝ)
  let permutations := nums.permutations
  (permutations.map iterative_mean).maximum - 
  (permutations.map iterative_mean).minimum = 4.25 :=
sorry

end iterative_mean_difference_l781_781977


namespace speed_ratio_liu_guan_l781_781845

variable (distance : ℝ) -- Distance between long pavilion and short pavilion
variable (t_guan : ℝ := 3) -- Time in hours for Guan Yu to reach short pavilion
variable (t_liu_start : ℝ := 3) -- Time in hours after which Liu Bei starts
variable (meet_time : ℝ := 2) -- Time in hours for Liu Bei to meet Guan Yu

-- Liu Bei's speed
def speed_liu (distance : ℝ) (meet_time : ℝ) : ℝ := distance / (2 * meet_time)

-- Guan Yu's speed
def speed_guan (distance : ℝ) (t_guan : ℝ) : ℝ := distance / t_guan

-- The ratio of Liu Bei's speed to Guan Yu's speed
def speed_ratio (distance : ℝ) (t_guan : ℝ) (meet_time : ℝ) : ℝ := 
  speed_liu distance meet_time / speed_guan distance t_guan

theorem speed_ratio_liu_guan : 
  speed_ratio distance t_guan meet_time = 5 / 6 := 
by sorry

end speed_ratio_liu_guan_l781_781845


namespace complex_number_quadrant_l781_781787

-- Definitions and setup
def complex_plane_quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "Origin or On axis"

-- The main theorem that needs to be proven
theorem complex_number_quadrant :
  let z : ℂ := (-2 + I) * I in
  complex_plane_quadrant z = "Third quadrant" :=
by
  sorry

end complex_number_quadrant_l781_781787


namespace theo_reduces_friendships_l781_781109

noncomputable def event (G : SimpleGraph (Fin 2019)) (A B C : Fin 2019) : SimpleGraph (Fin 2019) :=
  if G.Adj A B ∧ G.Adj A C ∧ ¬G.Adj B C
  then G.deleteEdge A B |>.deleteEdge A C |>.addEdge B C
  else G

theorem theo_reduces_friendships :
  ∃ F : List (Fin 2019 × Fin 2019) → SimpleGraph (Fin 2019), ∀ (G : SimpleGraph (Fin 2019)),
    (∀ v, v < 1009 → G.degree v = 1010) →
    (∀ v, 1009 ≤ v → G.degree v = 1009) →
    (∀ u v, G.Adj u v → G.Adj v u) →
    (∀ v, (F G).degree v ≤ 1) :=
sorry

end theo_reduces_friendships_l781_781109


namespace no_duplicate_among_expressions_l781_781077

theorem no_duplicate_among_expressions
  (N a1 a2 b1 b2 c1 c2 d1 d2 : ℕ)
  (ha : a1 = x^2)
  (hb : b1 = y^3)
  (hc : c1 = z^5)
  (hd : d1 = w^7)
  (ha2 : a2 = m^2)
  (hb2 : b2 = n^3)
  (hc2 : c2 = p^5)
  (hd2 : d2 = q^7)
  (h1 : N = a1 - a2)
  (h2 : N = b1 - b2)
  (h3 : N = c1 - c2)
  (h4 : N = d1 - d2) :
  ¬ (a1 = b1 ∨ a1 = c1 ∨ a1 = d1 ∨ b1 = c1 ∨ b1 = d1 ∨ c1 = d1) :=
by
  -- Begin proof here
  sorry

end no_duplicate_among_expressions_l781_781077


namespace eccentricity_of_ellipse_l781_781975

noncomputable def problem_conditions (a b c e : ℝ) :=
  a > b ∧ b > 0 ∧
  c = real.sqrt (a^2 - b^2) ∧
  (∃ (x y: ℝ), x^2 / a^2 + y^2 / b^2 = 1) ∧
  ∃ n, isosceles_obtuse_triangle (c, 0) (14 * a^2 / (9 * c), 0) n

theorem eccentricity_of_ellipse (a b c e : ℝ) (h : problem_conditions a b c e) :
  e = real.sqrt (1 - (b^2 / a^2)) :=
begin
  sorry
end

end eccentricity_of_ellipse_l781_781975


namespace base6_multiplication_l781_781119

-- Definitions of the base-six numbers
def base6_132 := [1, 3, 2] -- List representing 132_6
def base6_14 := [1, 4] -- List representing 14_6

-- Function to convert a base-6 list to a base-10 number
def base6_to_base10 (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc x, acc * 6 + x) 0

-- The conversion of our specified numbers to base-10
def base10_132 := base6_to_base10 base6_132
def base10_14 := base6_to_base10 base6_14

-- The product of the conversions
def base10_product := base10_132 * base10_14

-- Function to convert a base-10 number to a base-6 list
def base10_to_base6 (n : ℕ) : List ℕ :=
  let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else loop (n / 6) ((n % 6) :: acc)
  loop n []

-- The conversion of the product back to base-6
def base6_product := base10_to_base6 base10_product

-- The expected base-6 product
def expected_base6_product := [1, 3, 3, 2]

-- The formal theorem statement
theorem base6_multiplication :
  base6_product = expected_base6_product := by
  sorry

end base6_multiplication_l781_781119


namespace distributive_property_example_l781_781380

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = (3/4) * (-36) + (7/12) * (-36) - (5/9) * (-36) :=
by
  sorry

end distributive_property_example_l781_781380


namespace negation_of_neither_even_l781_781450

variable (a b : Nat)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

theorem negation_of_neither_even 
  (H : ¬ (¬ is_even a ∧ ¬ is_even b)) : is_even a ∨ is_even b :=
sorry

end negation_of_neither_even_l781_781450


namespace trigonometric_expression_evaluation_l781_781749

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l781_781749


namespace gcd_256_180_720_l781_781422

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end gcd_256_180_720_l781_781422


namespace wrapping_paper_area_l781_781083

theorem wrapping_paper_area :
  ∀ (w h d : ℕ), 
  w = 10 → 
  h = 5 → 
  d = 2 → 
  let side_len := w / 2 + h + d in
  side_len * side_len = 144 :=
by
  intros w h d hw hh hd
  let side_len := w / 2 + h + d
  have hw' : w = 10 := hw
  have hh' : h = 5 := hh
  have hd' : d = 2 := hd
  rw [hw', hh', hd']
  let side_len := 10 / 2 + 5 + 2
  have side_len_eq : side_len = 12 := by norm_num
  rw [side_len_eq]
  norm_num
  sorry

end wrapping_paper_area_l781_781083


namespace task_completed_ahead_of_schedule_l781_781935

def total_components : ℕ := 15000
def planned_days : ℕ := 30
def additional_components_per_day : ℕ := 250

theorem task_completed_ahead_of_schedule 
    (total_components = 15000)
    (planned_days = 30)
    (additional_components_per_day = 250) : 
    planned_days - total_components / (total_components / planned_days + additional_components_per_day) = 10 :=
by
  sorry

end task_completed_ahead_of_schedule_l781_781935


namespace incident_ray_line_equation_l781_781714

noncomputable def point (x y : ℝ) := (x, y)

def line := {l : ℝ × ℝ × ℝ // l.1 ≠ 0 ∨ l.2 ≠ 0}

def passes_through (p : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : Prop :=
l.1 * p.1 + l.2 * p.2 + l.3 = 0

def reflection_point_of_line (p : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : ℝ × ℝ :=
let (a, b, c) := l in
let (x, y) := p in
((x * (a^2 - b^2) - 2 * b * (a * y + b * x + c)) / (a^2 + b^2),
 (y * (b^2 - a^2) - 2 * a * (a * x + b * y + c)) / (a^2 + b^2))

def collinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
(p₂.2 - p₁.2) * (p₃.1 - p₁.1) = (p₃.2 - p₁.2) * (p₂.1 - p₁.1)

theorem incident_ray_line_equation :
  let M := point (-1) 0 in
  let N := point 0 1 in
  let l := (1, -1, -1) in
  let N' := reflection_point_of_line N l in
  collinear M N' (0, -((1 / 3))) ∧
  passes_through (1/3, 0) (1, 3, 1) ∧
  collinear (1/3, 0) (0, -((1 / 3))) (1, -((1 / 3))) :=
sorry

end incident_ray_line_equation_l781_781714


namespace floor_sqrt_50_l781_781608

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781608


namespace euler_totient_divisibility_l781_781315

open Nat

theorem euler_totient_divisibility (n : ℕ) (hn : n ≥ 2) : n ∣ euler_totient (2^n - 1) :=
  sorry

end euler_totient_divisibility_l781_781315


namespace range_of_m_l781_781199

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x

theorem range_of_m (m : ℝ) : (∃ t₁ t₂ t₃ : ℝ, tangent_to_curve f t₁ 1 ∧ tangent_to_curve f t₂ 1 ∧ tangent_to_curve f t₃ 1) → 
  (-3 < m ∧ m < -2) :=
sorry

end range_of_m_l781_781199


namespace forest_problem_l781_781496

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781496


namespace ratio_of_areas_theorem_l781_781375

noncomputable def ratio_of_areas (R : ℝ) (α : ℝ) : ℝ :=
  R^2 * sin α / (R^2 * (2 * α - sin α))

theorem ratio_of_areas_theorem (R α : ℝ) (hR : 0 < R) (hα : 0 < α) :
  ratio_of_areas R α = sin α / (2 * α - sin α) :=
by
  -- Math proof goes here
  sorry

end ratio_of_areas_theorem_l781_781375


namespace number_of_special_divisors_l781_781818

theorem number_of_special_divisors (n : ℕ) (h : n = 2^35 * 3^17) :
  let divisors_n_squared := (70 + 1) * (34 + 1)
  let factors_less_than_n := (divisors_n_squared - 1) / 2
  let divisors_n := (35 + 1) * (17 + 1)
  let divisors_less_than_n := divisors_n - 1
  factors_less_than_n - divisors_less_than_n = 594 :=
by {
  dsimp at *,
  sorry
}

end number_of_special_divisors_l781_781818


namespace angle_DBE_zero_l781_781288

theorem angle_DBE_zero
  (A B C D E : Type*)
  (hABC : angle A B C = 90)
  (hABD : AB = BD)
  (hEonAC : E lies_on AC) :
  angle D B E = 0 :=
sorry

end angle_DBE_zero_l781_781288


namespace go_game_prob_l781_781782

theorem go_game_prob :
  ∀ (pA pB : ℝ),
    (pA = 0.6) →
    (pB = 0.4) →
    ((pA ^ 2) + (pB ^ 2) = 0.52) :=
by
  intros pA pB hA hB
  rw [hA, hB]
  sorry

end go_game_prob_l781_781782


namespace forest_problem_l781_781494

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781494


namespace range_of_x_l781_781323

theorem range_of_x (x y : ℝ) (h : x - 6 * real.sqrt y - 4 * real.sqrt (x - y) + 12 = 0) :
  14 - 2 * real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * real.sqrt 13 :=
sorry

end range_of_x_l781_781323


namespace production_today_l781_781677

theorem production_today (n : ℕ) (P T : ℕ) 
  (h1 : n = 4) 
  (h2 : (P + T) / (n + 1) = 58) 
  (h3 : P = n * 50) : 
  T = 90 := 
by
  sorry

end production_today_l781_781677


namespace equal_area_triangles_l781_781264

theorem equal_area_triangles
  (A B C D E F G : Type)
  [linear_ordered_field A]
  (S₁ S₂ S₃ S₄ S₅ : A) 
  (area_ABC CD AC BE AB DF AD AG AE : A) 
  (h₁ : S₁ = S₂ ∧ S₂ = S₃ ∧ S₃ = S₄ ∧ S₄ = S₅) 
  (h₂ : S₁ = (1 / 5) * area_ABC) 
  (h_cd : CD = (1 / 5) * AC)
  (h_be : BE = (1 / 4) * AB)
  (h_df : DF = (1 / 3) * AD)
  (h_ag : AG = (1 / 2) * AE) :
  True := 
by
  sorry

end equal_area_triangles_l781_781264


namespace problem_statement_l781_781583

-- We need to define M and sum of digits function then state the theorem.
noncomputable def M := ∑ k in (finset.range 200).map (λ x, x + 1), (10^(k + 1) - 2)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem problem_statement : sum_of_digits M = 207 :=
sorry

end problem_statement_l781_781583


namespace exact_two_solutions_l781_781652

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l781_781652


namespace hyperbola_eccentricity_l781_781287

-- Define the given hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - (y^2 / 6) = 1

-- Define the value of a and c based on the given condition
def a : ℝ := Real.sqrt 3
def c : ℝ := 3

-- Define the eccentricity based on the equation e = c / a
def eccentricity : ℝ := c / a 

-- Statement of the theorem to prove the eccentricity equals sqrt(3)
theorem hyperbola_eccentricity : eccentricity = Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_l781_781287


namespace rhombus_perimeter_l781_781914

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) 
  (h3 : ∀ {x y : ℝ}, (x = d1 / 2 ∧ y = d2 / 2) → (x^2 + y^2 = (d1 / 2)^2 + (d2 / 2)^2)) : 
  4 * (Real.sqrt ((d1/2)^2 + (d2/2)^2)) = 156 :=
by 
  rw [h1, h2]
  simp
  sorry

end rhombus_perimeter_l781_781914


namespace inverseP_l781_781201

-- Mathematical definitions
def isOdd (a : ℕ) : Prop := a % 2 = 1
def isPrime (a : ℕ) : Prop := Nat.Prime a

-- Given proposition P (hypothesis)
def P (a : ℕ) : Prop := isOdd a → isPrime a

-- Inverse proposition: if a is prime, then a is odd
theorem inverseP (a : ℕ) (h : isPrime a) : isOdd a :=
sorry

end inverseP_l781_781201


namespace negation_p_equiv_l781_781356

noncomputable def negation_of_proposition_p : Prop :=
∀ m : ℝ, ¬ ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem negation_p_equiv (p : Prop) (h : p = ∃ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0) :
  ¬ p ↔ negation_of_proposition_p :=
by {
  sorry
}

end negation_p_equiv_l781_781356


namespace triangles_fit_in_pan_l781_781333

theorem triangles_fit_in_pan (pan_length pan_width triangle_base triangle_height : ℝ)
  (h1 : pan_length = 15) (h2 : pan_width = 24) (h3 : triangle_base = 3) (h4 : triangle_height = 4) :
  (pan_length * pan_width) / (1/2 * triangle_base * triangle_height) = 60 :=
by
  sorry

end triangles_fit_in_pan_l781_781333


namespace combined_area_percentage_l781_781770

theorem combined_area_percentage (D_S : ℝ) (D_R : ℝ) (D_T : ℝ) (A_S A_R A_T : ℝ)
  (h1 : D_R = 0.20 * D_S)
  (h2 : D_T = 0.40 * D_R)
  (h3 : A_R = Real.pi * (D_R / 2) ^ 2)
  (h4 : A_T = Real.pi * (D_T / 2) ^ 2)
  (h5 : A_S = Real.pi * (D_S / 2) ^ 2) :
  ((A_R + A_T) / A_S) * 100 = 4.64 := by
  sorry

end combined_area_percentage_l781_781770


namespace balls_in_boxes_l781_781258

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781258


namespace balls_in_boxes_l781_781263

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781263


namespace wine_count_l781_781558

theorem wine_count (S B total W : ℕ) (hS : S = 22) (hB : B = 17) (htotal : S - B + W = total) (htotal_val : total = 31) : W = 26 :=
by
  sorry

end wine_count_l781_781558


namespace total_students_in_school_l781_781087

noncomputable def small_school_students (boys girls : ℕ) (total_students : ℕ) : Prop :=
boys = 42 ∧ 
(girls : ℕ) = boys / 7 ∧
total_students = boys + girls

theorem total_students_in_school : small_school_students 42 6 48 :=
by
  sorry

end total_students_in_school_l781_781087


namespace money_together_l781_781561

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l781_781561


namespace probability_FQG_is_acute_l781_781353

noncomputable def F : ℝ × ℝ := (-2, 3)
noncomputable def G : ℝ × ℝ := (5, -2)
noncomputable def H : ℝ × ℝ := (7, 3)

def is_acute (a b c : ℝ × ℝ) : Prop :=
  let θ := Real.atan2 (c.2 - b.2) (c.1 - b.1) - Real.atan2 (a.2 - b.2) (a.1 - b.1)
  θ < Real.pi / 2 ∧ θ > -Real.pi / 2

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def semicircle_area (M : ℝ × ℝ) (r : ℝ) : ℝ :=
  0.5 * π * r^2

theorem probability_FQG_is_acute : 
  let M := ((F.1 + G.1) / 2, (F.2 + G.2) / 2)
  let FG := Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2)
  let radius := FG / 2
  let semicircle := semicircle_area M radius
  let triangle := triangle_area F G H
  (semicircle / triangle) = 37 * π / 86 := by sorry

end probability_FQG_is_acute_l781_781353


namespace total_money_together_l781_781565

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l781_781565


namespace percentage_sqrt_condition_l781_781000

theorem percentage_sqrt_condition :
  ∃ P : ℝ, P = 25 ∧ sqrt ((P / 100) * 5) * sqrt 5 = 0.25000000000000006 :=
by
  use 25
  split
  norm_num
  have sqrt_eq : sqrt ((25 / 100) * 5) = sqrt (25 / 20) := by simp
  rw [sqrt_eq, sqrt_div, sqrt_div]
  norm_num
  field_simp
  sorry

end percentage_sqrt_condition_l781_781000


namespace find_c_l781_781134

open Real

theorem find_c (c : ℝ) (h : ∀ x, (x ∈ Set.Iio 2 ∨ x ∈ Set.Ioi 7) → -x^2 + c * x - 9 < -4) : 
  c = 9 :=
sorry

end find_c_l781_781134


namespace brown_stripes_l781_781418

theorem brown_stripes (B G Bl : ℕ) (h1 : G = 3 * B) (h2 : Bl = 5 * G) (h3 : Bl = 60) : B = 4 :=
by {
  sorry
}

end brown_stripes_l781_781418


namespace exists_point_X_on_line_l781_781693

noncomputable def line := ℝ → ℝ -- Assuming a line l can be represented as a function from ℝ to ℝ

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (P Q : Point) : ℝ :=
  ( (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 ) ^ 0.5

variables (l : line) (A B : Point) (a : ℝ)

-- Reflect a point A over a line l
def reflect_over_line (l : line) (A : Point) : Point :=
sorry -- Implementation of reflection 

-- Define the existence of point X
theorem exists_point_X_on_line : 
  ∃ (X : Point), distance A X + distance X B = a ∧ ∃ t : ℝ, l t = X.y :=
sorry

end exists_point_X_on_line_l781_781693


namespace work_done_in_a_day_l781_781060

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end work_done_in_a_day_l781_781060


namespace john_bought_six_bagels_l781_781350

theorem john_bought_six_bagels (b m : ℕ) (expenditure_in_dollars_whole : (90 * b + 60 * m) % 100 = 0) (total_items : b + m = 7) : 
b = 6 :=
by
  -- The proof goes here. For now, we skip it with sorry.
  sorry

end john_bought_six_bagels_l781_781350


namespace complete_residue_system_infinitely_many_positive_integers_l781_781086

def is_complete_residue_system (n m : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → i ≠ j → (i^n % m ≠ j^n % m)

theorem complete_residue_system_infinitely_many_positive_integers (m : ℕ) (h_pos : 0 < m) :
  ∃ᶠ n in at_top, is_complete_residue_system n m :=
sorry

end complete_residue_system_infinitely_many_positive_integers_l781_781086


namespace pierre_ate_100_grams_l781_781456

-- Define the conditions:
def cake_weight : Real := 400
def parts : Int := 8
def nathalie_fraction : Real := 1/8
def nathalie : Real := (nathalie_fraction * cake_weight)
def double (x : Real) := 2 * x

-- Statement to prove Pierre's consumption:
theorem pierre_ate_100_grams : double nathalie = 100 :=
by
  -- The detailed proof will be filled in here.
  sorry

end pierre_ate_100_grams_l781_781456


namespace range_of_m_l781_781709

open Real

-- Given propositions

variables (m : ℝ) 

-- Proposition p: m + 2 < 0
def p := m + 2 < 0

-- Proposition q: The equation x^2 + mx + 1 = 0 has no real roots
def q := ∀ x : ℝ, ¬(x^2 + m * x + 1 = 0)

-- Given ¬p is false
axiom not_not_p : ¬(¬p)

-- Given p ∧ q is false
axiom not_p_and_q : ¬(p ∧ q)

-- The range of real numbers for m
theorem range_of_m : m < -2 :=
by
  sorry

end range_of_m_l781_781709


namespace oldest_person_Jane_babysat_age_l781_781801

def Jane_current_age : ℕ := 32
def Jane_stop_babysitting_age : ℕ := 22 -- 32 - 10
def max_child_age_when_Jane_babysat : ℕ := Jane_stop_babysitting_age / 2  -- 22 / 2
def years_since_Jane_stopped : ℕ := Jane_current_age - Jane_stop_babysitting_age -- 32 - 22

theorem oldest_person_Jane_babysat_age :
  max_child_age_when_Jane_babysat + years_since_Jane_stopped = 21 :=
by
  sorry

end oldest_person_Jane_babysat_age_l781_781801


namespace problem_statement_l781_781758

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l781_781758


namespace boundary_count_sum_eq_twice_edges_l781_781474

theorem boundary_count_sum_eq_twice_edges (g s : ℕ) (n : Fin g → ℕ) :
  (Finset.univ.sum (λ i : Fin g, n i) = 2 * g) := 
sorry

end boundary_count_sum_eq_twice_edges_l781_781474


namespace path_count_l781_781950

theorem path_count (f : ℕ → (ℤ × ℤ)) :
  (∀ n, (f (n + 1)).1 = (f n).1 + 1 ∨ (f (n + 1)).2 = (f n).2 + 1) ∧
  f 0 = (-6, -6) ∧ f 24 = (6, 6) ∧
  (∀ n, ¬(-3 ≤ (f n).1 ∧ (f n).1 ≤ 3 ∧ -3 ≤ (f n).2 ∧ (f n).2 ≤ 3)) →
  ∃ N, N = 2243554 :=
by {
  sorry
}

end path_count_l781_781950


namespace volume_of_cone_from_half_sector_l781_781070

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * π * r * r * h

theorem volume_of_cone_from_half_sector :
  let sector_radius : ℝ := 6
  let base_radius : ℝ := 3
  let height : ℝ := 3 * real.sqrt 3
  cone_volume base_radius height = 9 * π * real.sqrt 3 :=
by {
  let sector_radius := 6 : ℝ,
  let base_radius := 3 : ℝ,
  let height := 3 * real.sqrt 3 : ℝ,
  calc
  cone_volume base_radius height
      = (1 / 3) * π * base_radius^2 * height : rfl
  ... = (1 / 3) * π * 3^2 * (3 * real.sqrt 3) : by rw [base_radius, height]
  ... = 9 * π * real.sqrt 3 : by norm_num
}

end volume_of_cone_from_half_sector_l781_781070


namespace find_parallelogram_height_l781_781144

def parallelogram_height (base area : ℕ) : ℕ := area / base

theorem find_parallelogram_height :
  parallelogram_height 32 448 = 14 :=
by {
  sorry
}

end find_parallelogram_height_l781_781144


namespace measure_4_liters_l781_781936

theorem measure_4_liters (faucet : Type) (three_liter_cont : Type) (five_liter_cont : Type)
  [has_capacity three_liter_cont 3] [has_capacity five_liter_cont 5] : 
  ∃ (five_liter_full: Galen five_liter_cont).measure 4 :=
begin
  sorry
end

class has_capacity (c : Type) (n : ℕ) := 
(capacity : nat)

instance three_capacity : has_capacity three_liter_cont 3 := {capacity := 3}
instance five_capacity : has_capacity five_liter_cont 5 := {capacity := 5}

structure Galen (c : Type) :=
(measure : ℕ)

end measure_4_liters_l781_781936


namespace solve_for_n_l781_781428

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by
  sorry

end solve_for_n_l781_781428


namespace concurrency_lines_l781_781171

/-- Given an acute triangle ABC with AC > BC and the circumcenter O.
The altitude of triangle ABC from C intersects AB at D and the circumcircle at E.
A line through O which is parallel to AB intersects AC at F.
Show that the line CO, the line through F perpendicular to AC, and the line through E parallel to DO are concurrent. -/
theorem concurrency_lines
  (A B C O D E F : Point)
  (h₁ : acute_triangle A B C)
  (h₂ : AC > BC)
  (h₃ : is_circumcenter O A B C)
  (h₄ : is_altitude C D A B)
  (h₅ : line_intersects_circumcircle C E A B)
  (h₆ : parallel (line_through O) (line_through A B))
  (h₇ : line_intersects AC F (line_through O))
  (h₈ : parallel (line_through E) (line_through D O))
  (h₉ : ⟂ (line_through F) AC) :
  are_concurrent [line_through C O, ⟂ (line_through F) AC, parallel (line_through E) (line_through D O)] :=
sorry

end concurrency_lines_l781_781171


namespace visited_attractions_l781_781980

-- Defining the predicates for each visit location
variable (E : Prop) -- Visited Eiffel Tower
variable (A : Prop) -- Visited Arc de Triomphe
variable (M : Prop) -- Visited Montparnasse
variable (P : Prop) -- Visited Hall for ball games

-- Each child lied exactly once

-- First nephew's statements
variable (first_nephew_statement : E ∧ ¬M ∧ A)
-- Second nephew's statements
variable (second_nephew_statement : E ∧ ¬A ∧ M ∧ ¬P)
-- Third nephew's statements
variable (third_nephew_statement : ¬E ∧ A)

-- Prove what was actually visited given the constraints
theorem visited_attractions (h1 : E ∨ ¬E)
                           (h2 : A ∨ ¬A)
                           (h3 : M ∨ ¬M)
                           (h4 : P ∨ ¬P)
                           (first_lied_once : ∃ (E1 E2 E3 : Prop), first_nephew_statement = E1 ∧ ¬E2 ∧ E3 ∧ num_false ([E1, ¬E2, E3]) = 1)
                           (second_lied_once : ∃ (E1 E2 E3 E4 : Prop), second_nephew_statement = E1 ∧ E2 ∧ E3 ∧ E4 ∧ num_false ([E1, E2, E3, E4]) = 1)
                           (third_lied_once : ∃ (E1 E2 : Prop), third_nephew_statement = E1 ∧ E2 ∧ num_false ([E1, E2]) = 1) :
                           E ∧ A ∧ M ∧ ¬P :=
by sorry

end visited_attractions_l781_781980


namespace doughnut_savings_l781_781779

theorem doughnut_savings :
  let cost_one_dozen : ℕ := 8
      cost_two_dozens : ℕ := 14
      cost_six_one_dozen := 6 * cost_one_dozen
      cost_three_two_dozens := 3 * cost_two_dozens
      savings := cost_six_one_dozen - cost_three_two_dozens
  in savings = 6 :=
by
  let cost_one_dozen := 8
  let cost_two_dozens := 14
  let cost_six_one_dozen := 6 * cost_one_dozen
  let cost_three_two_dozens := 3 * cost_two_dozens
  let savings := cost_six_one_dozen - cost_three_two_dozens
  have : savings = 6 := by sorry
  exact this

end doughnut_savings_l781_781779


namespace least_positive_integer_condition_l781_781029

theorem least_positive_integer_condition :
  ∃ (n : ℕ), n > 0 ∧ (n % 2 = 1) ∧ (n % 5 = 4) ∧ (n % 7 = 6) ∧ n = 69 :=
by
  sorry

end least_positive_integer_condition_l781_781029


namespace sum_of_roots_of_qubic_polynomial_l781_781464

noncomputable def Q (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_qubic_polynomial (a b c d : ℝ) 
  (h₁ : ∀ x : ℝ, Q a b c d (x^4 + x) ≥ Q a b c d (x^3 + 1))
  (h₂ : Q a b c d 1 = 0) : 
  -b / a = 3 / 2 :=
sorry

end sum_of_roots_of_qubic_polynomial_l781_781464


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781532

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781532


namespace distance_between_x_intercepts_l781_781471

theorem distance_between_x_intercepts (p : Point ℝ) (h1 : p = (8, 20)) :
  ∃ l1 l2 : Line ℝ, 
    l1.slope = 4 ∧ l2.slope = -2 ∧
    l1.contains p ∧ l2.contains p ∧
    distance (x_intercept l1) (x_intercept l2) = 15 :=
sorry

end distance_between_x_intercepts_l781_781471


namespace min_value_six_x_plus_one_over_x6_l781_781130

theorem min_value_six_x_plus_one_over_x6 (x : ℝ) (hx : 0 < x) : 
  (6 * x + 1 / x^6) ≥ 7 :=
begin
  sorry
end

end min_value_six_x_plus_one_over_x6_l781_781130


namespace quadratic_real_solutions_l781_781773

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end quadratic_real_solutions_l781_781773


namespace equilateral_triangle_third_vertex_y_coordinate_l781_781976

theorem equilateral_triangle_third_vertex_y_coordinate 
  (A B : ℝ × ℝ) (hx : A.1 = 2) (hy : A.2 = 7) (hx' : B.1 = 10) (hy' : B.2 = 7) :
  ∃ C : ℝ × ℝ, C.2 = 7 + 4 * Real.sqrt 3 ∧ 0 < C.1 ∧ 0 < C.2 :=
by
  let h_dist : dist A B = 8 := sorry -- distance between two vertices
  let h_altitude : altitude = (8 * Real.sqrt 3) / 2 := sorry -- altitude of equilateral triangle
  use (some_x_value, 7 + 4 * Real.sqrt 3)
  sorry -- proof of the theorem that the y-coordinate of the third vertex is 7 + 4sqrt(3)

end equilateral_triangle_third_vertex_y_coordinate_l781_781976


namespace num_ways_distribute_balls_l781_781252

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781252


namespace chocolate_and_gum_l781_781311

/--
Kolya says that two chocolate bars are more expensive than five gum sticks, 
while Sasha claims that three chocolate bars are more expensive than eight gum sticks. 
When this was checked, only one of them was right. Is it true that seven chocolate bars 
are more expensive than nineteen gum sticks?
-/
theorem chocolate_and_gum (c g : ℝ) (hk : 2 * c > 5 * g) (hs : 3 * c > 8 * g) (only_one_correct : ¬((2 * c > 5 * g) ∧ (3 * c > 8 * g)) ∧ (2 * c > 5 * g ∨ 3 * c > 8 * g)) : 7 * c < 19 * g :=
by
  sorry

end chocolate_and_gum_l781_781311


namespace car_value_decrease_l781_781807

theorem car_value_decrease (original_price : ℝ) (decrease_percent : ℝ) (current_value : ℝ) :
  original_price = 4000 → decrease_percent = 0.30 → current_value = original_price * (1 - decrease_percent) → current_value = 2800 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end car_value_decrease_l781_781807


namespace garden_yield_l781_781341

theorem garden_yield
  (steps_length : ℕ)
  (steps_width : ℕ)
  (step_to_feet : ℕ → ℝ)
  (yield_per_sqft : ℝ)
  (h1 : steps_length = 18)
  (h2 : steps_width = 25)
  (h3 : ∀ n : ℕ, step_to_feet n = n * 2.5)
  (h4 : yield_per_sqft = 2 / 3)
  : (step_to_feet steps_length * step_to_feet steps_width) * yield_per_sqft = 1875 :=
by
  sorry

end garden_yield_l781_781341


namespace solution_set_f_f_x_leq_3_l781_781721

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^(-x) - 1 else -x^2 + x

theorem solution_set_f_f_x_leq_3 :
  { x : ℝ | f (f x) ≤ 3 } = { x : ℝ | x ≤ 2 } :=
sorry

end solution_set_f_f_x_leq_3_l781_781721


namespace total_expenditure_is_3500_l781_781473

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thurs : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300
def cost_earphone : ℕ := 620
def cost_pen : ℕ := 30
def cost_notebook : ℕ := 50

def expenditure_fri : ℕ := cost_earphone + cost_pen + cost_notebook
def total_expenditure : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thurs + expenditure_fri + expenditure_sat + expenditure_sun

theorem total_expenditure_is_3500 : total_expenditure = 3500 := by
  sorry

end total_expenditure_is_3500_l781_781473


namespace has_exactly_two_solutions_iff_l781_781661

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l781_781661


namespace quadrilateral_inscribed_l781_781853

variables {O : Type} [metric_space O] [signal_point O]
variables {A B C D X Y E F G : O}
variables {q : ℝ}

axiom h1 : ℝ
axiom AB : dist A B = 5
axiom BC : dist B C = 4
axiom CD : dist C D = 9
axiom DA : dist D A = 7
axiom h2 : ∃ q : ℝ, dist B D = q
axiom DX_BD_1_3 : dist D X = q / 3
axiom BY_BD_5_18 : dist B Y = (5 / 18) * q
axiom E_def : X ∉ A ∩ Y ∧ Y = A ∧ ∃ E : O, line A X ∩ line D A = E
axiom F_def : X ∉ A ∩ C ∧ C = A ∧ ∃ F : O, line C X ∩ line E A = F
axiom G_def : ∃ G, G ≠ C ∧ G ∈ circle O ∧ line C X ∩ circle O = G

theorem quadrilateral_inscribed {AB CD BD_q q DX_BD_1_3 BY_BD_5_18 E_def F_def G_def} :
  XF * XG = (13 / 27) * q^2 :=
sorry

end quadrilateral_inscribed_l781_781853


namespace exist_point_satisfying_condition_l781_781825

-- Define a point in 2D space.
structure Point :=
  (x y : ℝ)

-- Define a convex polygon by its vertices.
structure ConvexPolygon :=
  (vertices : list Point)
  (is_convex : ∀ (A B C : Point) (hA : A ∈ vertices) (hB : B ∈ vertices) (hC : C ∈ vertices), 
                segment_intersects_boundary : ∃ (P : Point), ∃ (Q : Point), (P ≠ Q ∧ P ∈ boundary ∧ Q ∈ boundary ∧ B ∈ segment P Q))

-- Define the segment between two points.
def segment (P Q : Point) : set Point :=
  {R : Point | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ R.x = (1 - t) * P.x + t * Q.x ∧ R.y = (1 - t) * P.y + t * Q.y}

-- Second intersection of the line segment with the boundary of the polygon.
noncomputable def second_intersection (A X : Point) (poly : ConvexPolygon) : Point :=
  ∃! (B : Point), B ∈ poly.boundary ∧ B ≠ A ∧ (∃ t, 0 < t ∧ X = (1 - t) • A + t • B)

-- Define the condition on the lengths of segments.
noncomputable def length (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

-- The theorem statement.
theorem exist_point_satisfying_condition 
  (P : ConvexPolygon)
  (hP : P.is_convex) :
  ∃ (X : Point), 
  ∀ (i : Point) (h : i ∈ P.vertices), 
    let Bi := second_intersection i X in
    length X i / length X Bi ≤ 2 :=
sorry

end exist_point_satisfying_condition_l781_781825


namespace kittens_per_bunny_l781_781840

-- Conditions
def total_initial_bunnies : ℕ := 30
def fraction_given_to_friend : ℚ := 2 / 5
def total_bunnies_after_birth : ℕ := 54

-- Determine the number of kittens each bunny gave birth to
theorem kittens_per_bunny (initial_bunnies given_fraction total_bunnies_after : ℕ) 
  (h1 : initial_bunnies = total_initial_bunnies)
  (h2 : given_fraction = fraction_given_to_friend)
  (h3 : total_bunnies_after = total_bunnies_after_birth) :
  (total_bunnies_after - (total_initial_bunnies - (total_initial_bunnies * fraction_given_to_friend))) / 
    (total_initial_bunnies * (1 - fraction_given_to_friend)) = 2 :=
by
  sorry

end kittens_per_bunny_l781_781840


namespace number_2013_repeated_three_times_divisible_by_9_l781_781038

theorem number_2013_repeated_three_times_divisible_by_9 :
  ∃ n : ℕ, (n > 0) ∧ (6 * n) ≡ 0 [MOD 9] :=
begin
  use 3,     -- The answer \( n = 3 \)
  split,
  { exact nat.succ_pos' 2 },  -- Prove \( n > 0 \)
  { show 6 * 3 ≡ 0 [MOD 9],
    norm_num }
end

end number_2013_repeated_three_times_divisible_by_9_l781_781038


namespace interior_alternate_angles_implies_parallel_l781_781393

-- Definitions from the conditions
def interior_alternate_angles (l1 l2 : ℝ → ℝ → Prop) (t : ℝ → ℝ → Prop) : Prop :=
  ∃ α β : ℝ → ℝ, 
    t α β ∧ 
    ∀ x y : ℝ, (l1 x y → l2 x y) ∧ (interior_alternate_angles α β = interior_alternate_angles α β)

def are_parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ t : ℝ → ℝ → Prop, interior_alternate_angles l1 l2 t → l1 = l2

-- The main statement
theorem interior_alternate_angles_implies_parallel (l1 l2 : ℝ → ℝ → Prop) (t : ℝ → ℝ → Prop) :
  interior_alternate_angles l1 l2 t → are_parallel l1 l2 :=
by
  sorry

end interior_alternate_angles_implies_parallel_l781_781393


namespace bc_eq_fc_l781_781850

variable {α : Type} [EuclideanGeometry α]

-- Definitions of points and lines
variables (A B C D E F : α)

-- Midpoint property
def is_midpoint (E A D : α) : Prop := dist E A = dist E D

-- Intersection property
def intersects (B D E C F : α) : Prop := Line_through B D ≠ Line_through E C ∧ F ∈ Line_through B D ∧ F ∈ Line_through E C

-- Perpendicularity property
def is_perpendicular (A F D : α) : Prop := ∠ A F D = 90

-- The conjecture we need to prove
def BC_eq_FC (A B C D E F : α) : Prop :=
is_midpoint E A D →
intersects B D E C F →
is_perpendicular A F D →
dist B C = dist F C

-- The theorem
theorem bc_eq_fc (A B C D E F : α) :
  BC_eq_FC A B C D E F :=
by {
  sorry
}

end bc_eq_fc_l781_781850


namespace probability_more_grandsons_or_granddaughters_l781_781338

noncomputable def binomial_probability (n r : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n r : ℚ) * (p^r) * ((1 - p)^(n - r))

theorem probability_more_grandsons_or_granddaughters
  (n : ℕ) (p_male : ℚ) (p_female : ℚ) (P : ℚ)
  (h_n : n = 12)
  (h_male : p_male = 0.6)
  (h_female : p_female = 0.4)
  (h_P : P = 1 - binomial_probability 12 6 0.6) :
  P = 0.823 :=
by
  rw [h_n, h_male, h_female, h_P]
  sorry

end probability_more_grandsons_or_granddaughters_l781_781338


namespace megatek_manufacturing_percentage_l781_781877

theorem megatek_manufacturing_percentage (angle_manufacturing : ℝ) (full_circle : ℝ) 
  (h1 : angle_manufacturing = 162) (h2 : full_circle = 360) :
  (angle_manufacturing / full_circle) * 100 = 45 :=
by
  sorry

end megatek_manufacturing_percentage_l781_781877


namespace total_students_count_l781_781767

theorem total_students_count (n1 n2 n: ℕ) (avg1 avg2 avg_tot: ℝ)
  (h1: n1 = 15) (h2: avg1 = 70) (h3: n2 = 10) (h4: avg2 = 90) (h5: avg_tot = 78)
  (h6: (n1 * avg1 + n2 * avg2) / (n1 + n2) = avg_tot) :
  n = 25 :=
by
  sorry

end total_students_count_l781_781767


namespace four_digit_numbers_count_unique_four_digit_numbers_count_l781_781919

theorem four_digit_numbers_count :
  let digits := {0, 1, 2, 3, 4, 5, 6}
  let four_digit_candidates := digits.erase 0
  let full_set := digits ∪ {10} -- using 10 as dummy to represent no limitation for other places
  let four_digit := λ n : ℕ, n ∈ four_digit_candidates × (full_set*3)
  let odd_digit := [1, 3, 5]
  let even_digit := digits.erase 1 ∪ digits.erase 3 ∪ digits.erase 5
  (∑ n in four_digit, 1) = 2058 ∧
  (∑ n in (odd_digit × (four_digit_candidates × (full_set*2))), 1) = 882 ∧
  (∑ n in (even_digit × (four_digit_candidates × (full_set*2))), 1) = 1176 :=
sorry

theorem unique_four_digit_numbers_count :
  let digits := {0, 1, 2, 3, 4, 5, 6}
  let four_digit_candidates := digits.erase 0
  let permutations := list.permutations four_digit_candidates.to_list
  let unique_digits := arbitrary { val : List Nat // list.pairwise (≠) val }
  let odd_digit := [1, 3, 5]
  let even_digit := digits.erase 1 ∪ digits.erase 3 ∪ digits.erase 5
  (\sum n in permutations, 1 ) = 720 ∧
  (\sum n in (odd_digit × (unique_digits.erase 0).erase_duplicates.permutations.last 3), 1) = 300 ∧
  (\sum n in (even_digit × (unique_digits.erase 0).erase_duplicates.permutations.last 3), 1) = 420 :=
sorry

end four_digit_numbers_count_unique_four_digit_numbers_count_l781_781919


namespace find_n_l781_781429

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by {
  sorry,
}

end find_n_l781_781429


namespace surface_area_of_sphere_l781_781084

theorem surface_area_of_sphere (l w h : ℝ) (s t : ℝ) :
  l = 3 ∧ w = 2 ∧ h = 1 ∧ (s = (l^2 + w^2 + h^2).sqrt / 2) → t = 4 * Real.pi * s^2 → t = 14 * Real.pi :=
by
  intros
  sorry

end surface_area_of_sphere_l781_781084


namespace nearest_integer_log_sum_l781_781424

noncomputable def nearest_log_sum : ℤ :=
Int.nearest_to (Real.log 2050 / Real.log 2)

theorem nearest_integer_log_sum :
  (∑ n in Finset.range 2049, Real.log (n+2) / Real.log 2 - Real.log (n+1) / Real.log 2) = Real.log 2050 / Real.log 2
  ∧ nearest_log_sum = 11 :=
sorry

end nearest_integer_log_sum_l781_781424


namespace find_number_of_equation_l781_781055

theorem find_number_of_equation (x : ℝ) (h : (sqrt 97 + sqrt x) / sqrt 54 = 4.340259786868312) :
  x = 485.7460897270573 :=
sorry

end find_number_of_equation_l781_781055


namespace max_value_of_expression_l781_781670

noncomputable def max_value (x : ℝ) : ℝ :=
  x * (1 + x) * (3 - x)

theorem max_value_of_expression :
  ∃ x : ℝ, 0 < x ∧ max_value x = (70 + 26 * Real.sqrt 13) / 27 :=
sorry

end max_value_of_expression_l781_781670


namespace imo_1989_q6_l781_781831

-- Define the odd integer m greater than 2
def isOdd (m : ℕ) := ∃ k : ℤ, m = 2 * k + 1

-- Define the condition for divisibility
def smallest_n (m : ℕ) (k : ℕ) (p : ℕ) : ℕ :=
  if k ≤ 1989 then 2 ^ (1989 - k) else 1

theorem imo_1989_q6 
  (m : ℕ) (h_m_gt2 : m > 2) (h_m_odd : isOdd m) (k : ℕ) (p : ℕ) (h_m_form : m = 2^k * p - 1) (h_p_odd : isOdd p) (h_k_gt1 : k > 1) :
  ∃ n : ℕ, (2^1989 ∣ m^n - 1) ∧ n = smallest_n m k p :=
by
  sorry

end imo_1989_q6_l781_781831


namespace min_value_frac_inv_sum_l781_781823

theorem min_value_frac_inv_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + 3 * y = 1) : 
  ∃ (minimum_value : ℝ), minimum_value = 4 + 2 * Real.sqrt 3 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 1 → (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3) := 
sorry

end min_value_frac_inv_sum_l781_781823


namespace length_RS_approx_866_l781_781292

theorem length_RS_approx_866
  (θ : ℝ)
  (FD DR FR FS : ℝ)
  (hFD : FD = 3)
  (hDR : DR = 8)
  (hFR : FR = 6)
  (hFS : FS = 9)
  (hθ : ∠ RFS = ∠ FDR = θ)
  : RS ≈ 8.66 :=
by
  sorry

end length_RS_approx_866_l781_781292


namespace field_length_l781_781386

theorem field_length 
  (w l : ℝ)
  (pond_area : ℝ := 25)
  (h1 : l = 2 * w)
  (h2 : pond_area = 25)
  (h3 : pond_area = (1 / 8) * (l * w)) :
  l = 20 :=
by
  sorry

end field_length_l781_781386


namespace even_quadruple_composition_l781_781830

variable {α : Type*} [AddGroup α]

-- Definition of an odd function
def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

theorem even_quadruple_composition {f : α → α} 
  (hf_odd : is_odd_function f) : 
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
by
  sorry

end even_quadruple_composition_l781_781830


namespace balls_in_boxes_l781_781260

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781260


namespace proof_questions_l781_781277

noncomputable def question1 (a b c : ℝ) (A : ℝ) : Prop :=
  b^2 + c^2 - a^2 = b * c → A = Real.pi / 3

noncomputable def question2 (a b : ℝ) (B : ℝ) : Prop :=
  let f (x : ℝ) := Real.sin x + 2 * Real.cos (x / 2)^2 in
  a = 2 → f B = Real.sqrt 2 + 1 → B = Real.pi / 4 → b = 2 * Real.sqrt 6 / 3

theorem proof_questions (a b c A B : ℝ) :
  (b^2 + c^2 - a^2 = b * c ∧ a = 2 ∧ (Real.sin B + 2 * (Real.cos (B / 2))^2 = Real.sqrt 2 + 1)) →
  (question1 a b c A ∧ question2 a b B)
:= by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  split
  { -- Proof of question1
    intro hyp,
    rw hyp,
    exact Real.pi_div_three_eq }
  {
   -- Proof of question2
   sorry -- skipping the proof
  }

end proof_questions_l781_781277


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781540

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781540


namespace constant_term_in_expansion_is_15_l781_781329

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then log x else x + a^3

theorem constant_term_in_expansion_is_15 (a : ℝ) (h : f (f 1 a) = 1) : 
  a = 1 → 
  let expansion := (4^x - 2^(-x))^6
  let constant_term := binomial_constant_term expansion 6 4 -- Pseudocode representing general term extraction
  constant_term = 15 :=
by
  sorry

end constant_term_in_expansion_is_15_l781_781329


namespace problem_statement_l781_781746

theorem problem_statement (x : ℝ) (h : 2 * x^2 + 1 = 17) : 4 * x^2 + 1 = 33 :=
by sorry

end problem_statement_l781_781746


namespace number_of_rational_roots_l781_781479

-- Define the polynomial with integer coefficients

theorem number_of_rational_roots (b₃ b₂ b₁ : ℤ) : 
  let p := 16 * x ^ 4 + b₃ * x ^ 3 + b₂ * x ^ 2 + b₁ * x + 24 in
  ∃ (s : Set ℚ), 
    {x ∈ s | p.eval x = 0} ∧ s.card = 28 :=
sorry

end number_of_rational_roots_l781_781479


namespace min_balls_to_ensure_17_of_one_color_l781_781066

theorem min_balls_to_ensure_17_of_one_color :
  ∀ (orange purple brown gray silver golden : ℕ),
  orange = 26 →
  purple = 21 →
  brown = 20 →
  gray = 15 →
  silver = 12 →
  golden = 10 →
  (orange + purple + brown + gray + silver + golden) = 104 →
  ∃ n, n = 86 ∧ ∀ draws,
  draws ≥ 86 → 
  (draws - 16 - 16 - 16 - 15 - 12 - 10 ≥ 1) :=
by
  intro orange purple brown gray silver golden 
  intros horange hpurple hbrown hgray hsilver hgolden htotal
  existsi 86
  split
  · refl
  sorry

end min_balls_to_ensure_17_of_one_color_l781_781066


namespace max_path_length_l781_781954

-- Define the rectangular prism dimensions 
def length : ℝ := 2
def width : ℝ := 1
def height : ℝ := 3

-- Define the distances
def edge_distance1 : ℝ := length
def edge_distance2 : ℝ := width
def edge_distance3 : ℝ := height
def face_diagonal1 : ℝ := Real.sqrt (length^2 + width^2)
def face_diagonal2 : ℝ := Real.sqrt (width^2 + height^2)
def face_diagonal3 : ℝ := Real.sqrt (length^2 + height^2)
def space_diagonal : ℝ := Real.sqrt (length^2 + width^2 + height^2)

-- Statement of the theorem to prove the maximum path length
theorem max_path_length : 
    2 * space_diagonal + 2 * face_diagonal3 + 2 * face_diagonal2 = 2 * Real.sqrt(14) + 2 * Real.sqrt(13) + 2 * Real.sqrt(10) := 
by 
    sorry

end max_path_length_l781_781954


namespace prism_diagonal_correct_l781_781918

open Real

noncomputable def prism_diagonal_1 := 2 * sqrt 6
noncomputable def prism_diagonal_2 := sqrt 66

theorem prism_diagonal_correct (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  (prism_diagonal_1 = 2 * sqrt 6 ∧ prism_diagonal_2 = sqrt 66) :=
by
  sorry

end prism_diagonal_correct_l781_781918


namespace total_earnings_l781_781577

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end total_earnings_l781_781577


namespace inequality_proof_l781_781816

noncomputable def a : ℝ := 0.9 ^ 2
noncomputable def b : ℝ := 2 ^ 0.9
noncomputable def c : ℝ := Real.log 0.9 / Real.log 2

theorem inequality_proof : b > a ∧ a > c :=
by
  sorry

end inequality_proof_l781_781816


namespace balls_in_boxes_l781_781220

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781220


namespace find_value_of_x_l781_781730

-- Define the given sequence according to the problem
def sequence : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 5
| 5 := 8
| 7 := 21
| 8 := 34
| 9 := 55
| n := sequence (n - 1) + sequence (n - 2)

-- State the theorem to prove the value of x
theorem find_value_of_x : sequence 6 = 13 :=
by
  sorry

end find_value_of_x_l781_781730


namespace pierre_ate_grams_l781_781454

variable (cake_total_weight : ℕ) (parts : ℕ) (nathalie_fraction : ℚ) (pierre_multiple : ℚ)

-- Definitions based on the problem conditions
def part_weight := cake_total_weight / parts

def nathalie_eats := nathalie_fraction * cake_total_weight

def pierre_eats := pierre_multiple * nathalie_eats

-- The actual proof statement goal
theorem pierre_ate_grams (h1 : cake_total_weight = 400)
                         (h2 : parts = 8)
                         (h3 : nathalie_fraction = 1 / 8)
                         (h4 : pierre_multiple = 2)
                         : pierre_eats cake_total_weight parts nathalie_fraction pierre_multiple = 100 := by
  sorry

end pierre_ate_grams_l781_781454


namespace BothNormal_l781_781107

variable (Normal : Type) (Person : Type) (MrA MrsA : Person)
variables (isNormal : Person → Prop)

-- Conditions given in the problem
axiom MrA_statement : ∀ p : Person, p = MrsA → isNormal MrA → isNormal MrsA
axiom MrsA_statement : ∀ p : Person, p = MrA → isNormal MrsA → isNormal MrA

-- Question (translated to proof problem): 
-- prove that Mr. A and Mrs. A are both normal persons
theorem BothNormal : isNormal MrA ∧ isNormal MrsA := 
  by 
    sorry -- proof is omitted

end BothNormal_l781_781107


namespace base_19_solution_l781_781296

noncomputable def find_base (B : ℕ) : Prop :=
  7 * B^2 + 9 * B + 2 = 3 * (2 * B^2 + 9 * B + 7)

theorem base_19_solution : ∃ B : ℕ, B ≥ 10 ∧ find_base B ∧ B = 19 :=
by {
  use 19,
  split,
  { exact Nat.le_of_eq rfl },
  split,
  { unfold find_base, linarith },
  { exact Eq.refl _ },
}

end base_19_solution_l781_781296


namespace sixth_bin_cans_l781_781556

-- The number of cans in the first five bins
def bins : ℕ → ℕ 
| 1 := 2
| 2 := 4
| 3 := 7
| 4 := 11
| 5 := 16
| (n + 1) := bins n + (n + 1)

-- Theorem to prove the number of cans in the sixth bin is 22
theorem sixth_bin_cans : bins 6 = 22 := 
sorry

end sixth_bin_cans_l781_781556


namespace time_to_cross_tree_l781_781949

noncomputable def train_length := 1200  -- in meters
noncomputable def platform_length := 1000  -- in meters
noncomputable def time_to_pass_platform := 220  -- in seconds
noncomputable def total_distance := train_length + platform_length  -- total distance covered when passing the platform

theorem time_to_cross_tree :
  let speed := total_distance / time_to_pass_platform in
  let time_to_pass_tree := train_length / speed in
  time_to_pass_tree = 120 :=
sorry

end time_to_cross_tree_l781_781949


namespace train_length_proof_l781_781022

noncomputable def length_of_each_train (L : ℝ) : Prop :=
  let relative_speed_kmph := 46 - 36 in
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600 in
  let time_sec := 27 in
  let total_length := relative_speed_mps * time_sec in
  total_length = 2 * L ∧ L = 37.5 

theorem train_length_proof : ∃ L : ℝ, length_of_each_train L :=
by
  use 37.5
  unfold length_of_each_train
  sorry

end train_length_proof_l781_781022


namespace inequality_solution_l781_781398

theorem inequality_solution (x : ℝ) : (x - 1) / 3 > 2 → x > 7 :=
by
  intros h
  sorry

end inequality_solution_l781_781398


namespace infinite_odd_integers_sum_of_squares_l781_781858

theorem infinite_odd_integers_sum_of_squares :
  ∃ (f : ℕ → ℕ), (∀ i, ∃ a b : ℕ, coprime a b ∧ n = 2i + 1 → (f i)^5 + 2 * (f i) + 1 = a^2 + b^2) ∧
  (∀ i, ∃ j > i, f j = 2i + 1) :=
sorry

end infinite_odd_integers_sum_of_squares_l781_781858


namespace number_2013_repeated_three_times_divisible_by_9_l781_781037

theorem number_2013_repeated_three_times_divisible_by_9 :
  ∃ n : ℕ, (n > 0) ∧ (6 * n) ≡ 0 [MOD 9] :=
begin
  use 3,     -- The answer \( n = 3 \)
  split,
  { exact nat.succ_pos' 2 },  -- Prove \( n > 0 \)
  { show 6 * 3 ≡ 0 [MOD 9],
    norm_num }
end

end number_2013_repeated_three_times_divisible_by_9_l781_781037


namespace centroid_flies_eq_centroid_triangle_l781_781010

-- Define the triangle vertices and centroid
structure Triangle (α : Type) :=
  (A B C : α)

def centroid {α : Type} [Add α] [Mul α] [HasSmul ℕ α] (T : Triangle α) : α :=
  (T.A + T.B + T.C) / 3

-- Define the condition that the flies' triangle centroid remains stationary
structure FlyTriangle (α : Type) :=
  (D E F : α)
  (stationary_centroid : α)

def verifyStationaryCentroid (T : Triangle ℝ) (F : FlyTriangle ℝ) : Bool :=
  centroid F = F.stationary_centroid

-- Given the conditions, we need to prove that the stationary point is the centroid of the original triangle
theorem centroid_flies_eq_centroid_triangle (T : Triangle ℝ) (F : FlyTriangle ℝ) (h1 : verifyStationaryCentroid T F = true)
  (h2 : ∃ D, ∃ E, ∃ F, F.D ∈ [T.A, T.B, T.C] ∧ F.E ∈ [T.A, T.B, T.C] ∧ F.F ∈ [T.A, T.B, T.C] ∧ 
                 ∃ D', ∃ E', ∃ F', [D, E, F] = [D', E', F'] ∧
                 (∀ Q ∈ {T.A, T.B, T.C}, ∃ t : ℝ, ∃ s : ℝ, ∃ r : ℝ, Q = t • T.A + s • T.B + r • T.C ∧ t + s + r = 1)) :
  F.stationary_centroid = centroid T :=
by
  sorry

end centroid_flies_eq_centroid_triangle_l781_781010


namespace sum_is_maximized_at_9_or_10_l781_781318

def a (n : ℕ) : ℝ := - (n:ℝ)^2 + 9 * n + 10

def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

theorem sum_is_maximized_at_9_or_10 (n : ℕ) :
  (S n ≤ S (n + 1)) → (n = 9 ∨ n = 10) :=
by
  sorry

end sum_is_maximized_at_9_or_10_l781_781318


namespace divide_equally_l781_781307

-- Define the input values based on the conditions.
def brother_strawberries := 3 * 15
def kimberly_strawberries := 8 * brother_strawberries
def parents_strawberries := kimberly_strawberries - 93
def total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
def family_members := 4

-- Define the theorem to prove the question.
theorem divide_equally : 
    (total_strawberries / family_members) = 168 :=
by
    -- (proof goes here)
    sorry

end divide_equally_l781_781307


namespace max_cross_sectional_area_l781_781463

noncomputable def cone_base_radius := 2
noncomputable def cone_height := 6

-- Define a function to calculate the cross-sectional area S
def cross_sectional_area (x : ℝ) : ℝ :=
  π * ((cone_base_radius - ((cone_base_radius / cone_height) * x)) ^ 2)

theorem max_cross_sectional_area : ∃ (S_max : ℝ), S_max = 6 ∧ 
  S_max = max (cross_sectional_area x) :=
sorry

end max_cross_sectional_area_l781_781463


namespace range_of_f_on_interval_l781_781589

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem range_of_f_on_interval :
  Set.Icc (-1 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = y} :=
by
  sorry

end range_of_f_on_interval_l781_781589


namespace floor_sqrt_50_l781_781610

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781610


namespace floor_sqrt_50_l781_781611

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781611


namespace whisker_ratio_l781_781158

theorem whisker_ratio 
  (p : ℕ) (c : ℕ) (h1 : p = 14) (h2 : c = 22) (s := c + 6) :
  s / p = 2 := 
by
  sorry

end whisker_ratio_l781_781158


namespace number_of_ways_to_put_balls_in_boxes_l781_781246

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781246


namespace transform_negation_l781_781099

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l781_781099


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781533

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781533


namespace distinct_permutations_BOOKS_l781_781212

theorem distinct_permutations_BOOKS :
  let n := 5
  let k := 2
  (Fact n) / (Fact k) = 60 :=
  by 
  let n := 5
  let k := 2
  sorry

end distinct_permutations_BOOKS_l781_781212


namespace fx_decreasing_range_of_a_l781_781384

/-- Part 1: Prove that f(x) is a decreasing function given the conditions -/
theorem fx_decreasing (f : ℝ → ℝ) (h1 : ∀ m n : ℝ, f(m + n) = f(m) * f(n))
   (h2 : ∀ x : ℝ, x > 0 → 0 < f(x) ∧ f(x) < 1) : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) > f(x2) :=
by
  sorry

/-- Part 2: Prove that if A ∩ B = ∅, then the range of a is -1 ≤ a ≤ 1 -/
theorem range_of_a (f : ℝ → ℝ) 
    (h1 : ∀ m n : ℝ, f(m + n) = f(m) * f(n)) 
    (h2 : ∀ x : ℝ, x > 0 → 0 < f(x) ∧ f(x) < 1) 
    (A : set (ℝ × ℝ)) (B : set (ℝ × ℝ)) (a : ℝ)
    (hA : ∀ (x y : ℝ), (x, y) ∈ A ↔ f(x^2) * f(y^2) > f(1)) 
    (hB : ∀ (x y : ℝ) a, (x, y) ∈ B ↔ f(a * x - y + real.sqrt 2) = 1)
    (h_inter : A ∩ B = ∅) : -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end fx_decreasing_range_of_a_l781_781384


namespace ratio_to_percentage_l781_781838

theorem ratio_to_percentage (x y : ℚ) (h : (2/3 * x) / (4/5 * y) = 5 / 6) : (5 / 6 : ℚ) * 100 = 83.33 :=
by
  sorry

end ratio_to_percentage_l781_781838


namespace tickets_needed_l781_781913

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end tickets_needed_l781_781913


namespace field_trip_buses_l781_781869

-- Definitions of conditions
def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def seats_per_bus : ℕ := 72

-- Total calculations
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_people : ℕ := total_students + total_chaperones
def buses_needed : ℕ := (total_people + seats_per_bus - 1) / seats_per_bus

theorem field_trip_buses : buses_needed = 6 := by
  unfold buses_needed
  unfold total_people total_students total_chaperones chaperones_per_grade
  norm_num
  sorry

end field_trip_buses_l781_781869


namespace forest_trees_properties_l781_781516

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781516


namespace tickets_needed_l781_781912

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end tickets_needed_l781_781912


namespace angle_between_tangents_l781_781018

theorem angle_between_tangents (R1 R2 : ℝ) (k : ℝ) (h_ratio : R1 = 2 * k ∧ R2 = 3 * k)
  (h_touching : (∃ O1 O2 : ℝ, (R2 - R1 = k))) : 
  ∃ θ : ℝ, θ = 90 := sorry

end angle_between_tangents_l781_781018


namespace cash_price_eq_8000_l781_781856

noncomputable def cash_price (d m s : ℕ) : ℕ :=
  d + 30 * m - s

theorem cash_price_eq_8000 :
  cash_price 3000 300 4000 = 8000 :=
by
  -- Proof omitted.
  sorry

end cash_price_eq_8000_l781_781856


namespace halfway_fraction_l781_781024

theorem halfway_fraction : 
  ∃ (x : ℚ), x = 1/2 * ((2/3) + (4/5)) ∧ x = 11/15 :=
by
  sorry

end halfway_fraction_l781_781024


namespace yearly_savings_l781_781340

-- Definition of the conditions
def monthly_salary (S : ℕ) : Prop :=
  let remaining_salary := 0.40 * S
  let clothes_transport := 0.20 * S
  clothes_transport = 4038

-- The main theorem to be proven
theorem yearly_savings (S : ℕ) (hS : monthly_salary S) : 
  let monthly_savings := 0.20 * S
  let yearly_savings := monthly_savings * 12
  yearly_savings = 48456 := 
by
  -- Skipping the proof for now
  sorry

end yearly_savings_l781_781340


namespace no_integer_pairs_satisfy_equation_l781_781147

def equation_satisfaction (m n : ℤ) : Prop :=
  m^3 + 3 * m^2 + 2 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ (m n : ℤ), equation_satisfaction m n :=
by
  sorry

end no_integer_pairs_satisfy_equation_l781_781147


namespace desired_salt_percentage_is_ten_percent_l781_781071

-- Define the initial conditions
def initial_pure_water_volume : ℝ := 100
def saline_solution_percentage : ℝ := 0.25
def added_saline_volume : ℝ := 66.67
def total_volume : ℝ := initial_pure_water_volume + added_saline_volume
def added_salt : ℝ := saline_solution_percentage * added_saline_volume
def desired_salt_percentage (P : ℝ) : Prop := added_salt = P * total_volume

-- State the theorem and its result
theorem desired_salt_percentage_is_ten_percent (P : ℝ) (h : desired_salt_percentage P) : P = 0.1 :=
sorry

end desired_salt_percentage_is_ten_percent_l781_781071


namespace sqrt_floor_eq_seven_l781_781631

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781631


namespace range_of_a_l781_781276

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → x^2 + 2 * a * x + 1 ≥ 0) ↔ a ≥ -1 := 
by
  sorry

end range_of_a_l781_781276


namespace floor_sqrt_50_l781_781637

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781637


namespace gcd_84_108_132_156_l781_781917

theorem gcd_84_108_132_156 : Nat.gcd (Nat.gcd 84 108) (Nat.gcd 132 156) = 12 := 
by
  sorry

end gcd_84_108_132_156_l781_781917


namespace jeremy_coins_l781_781802

theorem jeremy_coins (x y z: ℕ) (h1: x + y + z = 15) (h2: 14 + y + 4z = 21) : z = 1 :=
sorry

end jeremy_coins_l781_781802


namespace money_together_l781_781563

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l781_781563


namespace construct_triangle_exists_l781_781988

variable {Point : Type} [AddGroup Point]
variable (O P Q R X Y Z : Point)
variable [Nonempty (Triangle Point)]

noncomputable def construct_triangle (X Y Z : Point) : Triangle Point :=
  sorry

theorem construct_triangle_exists :
  ∃ ABC : Triangle Point, 
    let O := incenter ABC
    let P := escribed_center ABC A
    let Q := escribed_center ABC B
    let R := escribed_center ABC C
    let X := midpoint O P
    let Y := midpoint O Q
    let Z := midpoint O R
    true :=
begin
  use construct_triangle X Y Z,
  sorry
end

end construct_triangle_exists_l781_781988


namespace integral_inequality_l781_781674

variable {f : ℝ → ℝ}

axiom positive : ∀ x ∈ Icc (0:ℝ) 1, 0 ≤ f x
axiom monotone_decreasing : ∀ x y ∈ Icc (0:ℝ) 1, x ≤ y → f y ≤ f x

theorem integral_inequality :
  (∫ x in (0:ℝ)..1, f x) * (∫ x in (0:ℝ)..1, x * (f x)^2) ≤ 
  (∫ x in (0:ℝ)..1, (f x)^2) * (∫ x in (0:ℝ)..1, x * f x) :=
sorry

end integral_inequality_l781_781674


namespace find_varphi_l781_781197

noncomputable def distance_between_adj_symmetric_axes: ℝ := π / 6
noncomputable def symmetry_point : ℝ × ℝ := (5 * π / 18, 0)
noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * sin (ω * x + ϕ)

theorem find_varphi 
  (ω > 0) 
  (ϕ > -π/2) 
  (ϕ < π/2) 
  (period: distance_between_adj_symmetric_axes * 2 = 2 * π / ω) 
  (symmetry: f symmetry_point.fst ω ϕ = 0) : ϕ = π / 3 := 
sorry

end find_varphi_l781_781197


namespace find_n_l781_781430

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by {
  sorry,
}

end find_n_l781_781430


namespace boat_licenses_count_l781_781486

theorem boat_licenses_count : 
  let letter_choices := 2 in 
  let digit_choices := 10 in
  let total_choices := letter_choices * (digit_choices ^ 5) in
  total_choices = 200000 := 
  by {
    sorry
  }

example : boat_licenses_count := by
  sorry

end boat_licenses_count_l781_781486


namespace complex_number_identity_l781_781683

theorem complex_number_identity (a b : ℝ) (i : ℂ) (h : (a + i) * (1 + i) = b * i) : a + b * i = 1 + 2 * i := 
by
  sorry

end complex_number_identity_l781_781683


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781541

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781541


namespace f_when_positive_l781_781712

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * (x + 2) else - (x * x) + 2 * x

theorem f_when_positive (x : ℝ) (h : x > 0) : f x = -x^2 + 2*x := by
  unfold f
  simp
  sorry

end f_when_positive_l781_781712


namespace balls_in_boxes_l781_781234

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781234


namespace sum_of_solutions_eq_zero_l781_781326

noncomputable def f : ℝ → ℝ := λ x, if x < 0 then 2 * x + 6 else x ^ 2 - 2

theorem sum_of_solutions_eq_zero :
  (finset.sum (finset.filter (λ x, f x = 2) (finset.univ : finset ℝ))) = 0 :=
sorry

end sum_of_solutions_eq_zero_l781_781326


namespace length_real_axis_l781_781200

def hyperbola1 (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1
def hyperbola2 (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def same_eccentricity (a c : ℝ) : Prop := c / a = real.sqrt 5 / 2
def orthogonal (OM MF2 : ℝ) : Prop := OM * MF2 = 0
def area_triangle (OM MF2 a b : ℝ) : Prop := 0.5 * a * b = 16
def pythagorean (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem length_real_axis (a b c : ℝ) :
  (∀ x y, hyperbola1 x y) →
  (∀ x y, hyperbola2 x y a b) →
  0 < a ∧ 0 < b →
  same_eccentricity a c →
  orthogonal (real.sqrt (c^2 - b^2)) b →
  area_triangle (real.sqrt (c^2 - b^2)) b a b →
  pythagorean a b c →
  2 * a = 16 :=
by
  intros
  sorry

end length_real_axis_l781_781200


namespace calculate_sale_prices_l781_781090

def sale_price_per_kg (cost_price: ℝ) (profit_percent: ℝ) : ℝ :=
  cost_price + (profit_percent / 100) * cost_price

theorem calculate_sale_prices :
  sale_price_per_kg 25 45 = 36.25 ∧
  sale_price_per_kg 30 35 = 40.5 ∧
  sale_price_per_kg 50 25 = 62.5 ∧
  sale_price_per_kg 70 20 = 84 :=
by 
  simp [sale_price_per_kg];
  split; norm_num
  split; norm_num
  split; norm_num
  norm_num

end calculate_sale_prices_l781_781090


namespace probability_of_two_kings_or_at_least_one_ace_l781_781271

def modified_deck := 54
def num_aces := 5
def num_kings := 5
def num_jokers := 2

-- Define a function to calculate the probability
noncomputable def probability_two_kings_or_at_least_one_ace : ℚ :=
  (10 / 1431) + (255 / 1431)

-- The main statement
theorem probability_of_two_kings_or_at_least_one_ace :
  ∃ (p : ℚ), p = probability_two_kings_or_at_least_one_ace ∧ p = 265 / 1431 :=
by
  use probability_two_kings_or_at_least_one_ace
  split
  · rfl
  · sorry

end probability_of_two_kings_or_at_least_one_ace_l781_781271


namespace exact_two_solutions_l781_781654

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l781_781654


namespace f_of_f_of_f_of_3_l781_781126

def f (x : ℕ) : ℕ := 
  if x > 9 then x - 1 
  else x ^ 3

theorem f_of_f_of_f_of_3 : f (f (f 3)) = 25 :=
by sorry

end f_of_f_of_f_of_3_l781_781126


namespace garlic_cloves_left_l781_781336

theorem garlic_cloves_left :
  let kitchen_initial := 750
  let pantry_initial := 450
  let basement_initial := 300
  let kitchen_used := 500
  let pantry_used := 230
  let basement_used := 170
  let kitchen_left := kitchen_initial - kitchen_used
  let pantry_left := pantry_initial - pantry_used
  let basement_left := basement_initial - basement_used
  let total_left := kitchen_left + pantry_left + basement_left
  total_left = 600 :=
by
  -- Definitions
  let kitchen_initial := 750
  let pantry_initial := 450
  let basement_initial := 300
  let kitchen_used := 500
  let pantry_used := 230
  let basement_used := 170
  let kitchen_left := kitchen_initial - kitchen_used
  let pantry_left := pantry_initial - pantry_used
  let basement_left := basement_initial - basement_used
  let total_left := kitchen_left + pantry_left + basement_left
  -- Prove the total number of remaining cloves
  show total_left = 600
  from sorry

end garlic_cloves_left_l781_781336


namespace smallest_root_of_equation_l781_781987

theorem smallest_root_of_equation :
  let a := (x : ℝ) - 4 / 5
  let b := (x : ℝ) - 2 / 5
  let c := (x : ℝ) - 1 / 2
  (a^2 + a * b + c^2 = 0) → (x = 4 / 5 ∨ x = 14 / 15) ∧ (min (4 / 5) (14 / 15) = 14 / 15) :=
by
  sorry

end smallest_root_of_equation_l781_781987


namespace sin_cos_sum_l781_781573

theorem sin_cos_sum {α β : ℝ} (hα : α = 27) (hβ : β = 18) :
  sin (α * (Real.pi / 180)) * cos (β * (Real.pi / 180)) + 
  cos (α * (Real.pi / 180)) * sin (β * (Real.pi / 180)) =
  (Real.sqrt 2) / 2 :=
by
  sorry

end sin_cos_sum_l781_781573


namespace KolyaWinsInDominoGame_l781_781310

def Kolya_has_winning_strategy : Prop :=
  ∀ (board : ℕ × ℕ → bool) (moveCount : ℕ),
    (∀ col row, col < 8 → row < 8 → board (col, row) = false) →
    (∀ i, i < moveCount → board (i % 8, i / 8) = true) →
    moveCount < 64 →
    (∃ (playerMove : ℕ × ℕ → bool),
      playerMove = board) 

theorem KolyaWinsInDominoGame : Kolya_has_winning_strategy :=
  sorry

end KolyaWinsInDominoGame_l781_781310


namespace num_ways_distribute_balls_l781_781250

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781250


namespace complex_solutions_count_l781_781738

noncomputable def count_solutions : ℕ :=
  2 * (number_of_solutions (λ y, |(cos y + i * sin y) - (y^2 - 1) / (y^2 + 1) - i * (2 * y) / (y^2 + 1)| < 20))

theorem complex_solutions_count (k : ℕ) :
  count_solutions = 2 * k :=
sorry

end complex_solutions_count_l781_781738


namespace alice_bakes_pie_time_l781_781551

variables (A : ℝ)

-- Conditions
def bob_bakes_pies_in_60_minutes : ℝ := 60 / 6
def alice_bakes_pies_in_60_minutes : ℝ := bob_bakes_pies_in_60_minutes + 2

-- Proof statement
theorem alice_bakes_pie_time :
  A = 60 / alice_bakes_pies_in_60_minutes :=
sorry

end alice_bakes_pie_time_l781_781551


namespace translate_point_up_l781_781413

theorem translate_point_up {a b : ℤ} :
  (a + b = 5) ∧ (a - b = 1) → (a = 3 ∧ b = 2) :=
by
  intro h
  cases h with hab hmb
  sorry

end translate_point_up_l781_781413


namespace range_of_tangent_transform_l781_781672

noncomputable def f (x : ℝ) : ℝ := Real.tan (π / 2 - x)

theorem range_of_tangent_transform {x : ℝ} (h1 : x ∈ Set.Icc (-π / 4) (π / 4)) (h2 : x ≠ 0) :
  Set.Icc (-π / 4) (π / 4) \ {0} ⊆ {y : ℝ | y ≤ -1 ∨ y ≥ 1} :=
by
  sorry

end range_of_tangent_transform_l781_781672


namespace kerosene_cost_l781_781280

theorem kerosene_cost (R E K : ℕ) (h1 : E = R) (h2 : K = 6 * E) (h3 : R = 24) : 2 * K = 288 :=
by
  sorry

end kerosene_cost_l781_781280


namespace forest_trees_properties_l781_781517

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781517


namespace photo_album_pages_l781_781553

noncomputable def P1 := 0
noncomputable def P2 := 10
noncomputable def remaining_pages := 20

theorem photo_album_pages (photos total_pages photos_per_page_set1 photos_per_page_set2 photos_per_page_remaining : ℕ) 
  (h1 : photos = 100)
  (h2 : total_pages = 30)
  (h3 : photos_per_page_set1 = 3)
  (h4 : photos_per_page_set2 = 4)
  (h5 : photos_per_page_remaining = 3) : 
  P1 = 0 ∧ P2 = 10 ∧ remaining_pages = 20 :=
by
  sorry

end photo_album_pages_l781_781553


namespace number_of_maple_trees_planted_l781_781008

def before := 53
def after := 64
def planted := after - before

theorem number_of_maple_trees_planted : planted = 11 := by
  sorry

end number_of_maple_trees_planted_l781_781008


namespace shortest_path_distance_l781_781283

/-
Define the coordinates according to the conditions in the problem.
-/
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (0.5, Math.sqrt 3 / 2, 0)
def D : ℝ × ℝ × ℝ := (0, 0, Math.sqrt 2)

/-
Define the midpoints M and N based on their respective edge midpoints.
-/
def M : ℝ × ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
def N : ℝ × ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2, (C.3 + D.3) / 2)

/- 
A function to calculate the Euclidean distance between two 3D points 
-/
def distance (P Q : ℝ × ℝ × ℝ) : ℝ := 
  Math.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

/-
The statement of the theorem proving the shortest path distance between midpoints M and N.
-/
theorem shortest_path_distance : distance M N = Math.sqrt 3 / 2 :=
by
  sorry

end shortest_path_distance_l781_781283


namespace problem_solution_l781_781765

theorem problem_solution
  (a d : ℝ)
  (h : (∀ x : ℝ, (x - 3) * (x + a) = x^2 + d * x - 18)) :
  d = 3 := 
sorry

end problem_solution_l781_781765


namespace smallest_even_x_l781_781425

theorem smallest_even_x (x : ℤ) (h1 : x < 3 * x - 10) (h2 : ∃ k : ℤ, x = 2 * k) : x = 6 :=
by {
  sorry
}

end smallest_even_x_l781_781425


namespace series_sum_l781_781121

theorem series_sum : (Finset.sum (Finset.range (102 + 1)) (λ n, if n % 2 = 0 then n else -n)) = 52 := by
  sorry

end series_sum_l781_781121


namespace ratio_proof_l781_781202

-- Definitions of the quadratic polynomials
def f1 (x : ℝ) (a : ℝ) := x^2 - a * x + 2
def f2 (x : ℝ) (b : ℝ) := x^2 + 3 * x + b
def f3 (x : ℝ) (a : ℝ) (b : ℝ) := 3 * x^2 + (3 - 2 * a) * x + 4 + b
def f4 (x : ℝ) (a : ℝ) (b : ℝ) := 3 * x^2 + (6 - a) * x + 2 + 2 * b

-- Definitions of the differences of roots
def A (a : ℝ) := Real.sqrt (a^2 - 8)
def B (b : ℝ) := Real.sqrt (9 - 4 * b)
def C (a : ℝ) (b : ℝ) := (1/3) * Real.sqrt (4 * a^2 - 12 * a - 39 - 12 * b)
def D (a : ℝ) (b : ℝ) := (1/3) * Real.sqrt (a^2 - 12 * a + 12 - 24 * b)

-- The condition |A| ≠ |B|
def condition (a : ℝ) (b : ℝ) := A a ≠ B b

-- The goal to be proved
theorem ratio_proof (a b : ℝ) (h : condition a b) :
  (C a b)^2 - (D a b)^2 = (1 / 3) * ((A a)^2 - (B b)^2) := by
  sorry

end ratio_proof_l781_781202


namespace integral_repr_factorial_integral_new_var_subst_integral_limit_log_gamma_second_repr_gamma_recursive_beta_gamma_relation_l781_781294

section IntegrationProofs

variables (α : ℝ) (n : ℕ) (a s : ℝ)

-- Problem 1
theorem integral_repr_factorial (hα : α > 0) (hn : n ≥ 0) :
  ∫ x in 0..1, x^(α-1) * (1-x)^n = n! / (α * (α + 1) * ... * (α + n)) :=
sorry

-- Problem 2
theorem integral_new_var_subst (a : ℝ) (hn : n ≥ 0) (ha : a > 0) :
  ∫ x in 0..1, ((1-x^a)/a)^n = n! / ( (1+a) * (1+2*a) * ... * (1+an) ) :=
sorry

-- Problem 3
theorem integral_limit_log (hn : n ≥ 0) :
  ∫ x in 0..1, (-log x)^n = n! :=
sorry

-- Problem 4
theorem gamma_second_repr (hs : s > 0) :
  Γ(s) = ∫ x in 0..∞, x^(s-1) * exp(-x) :=
sorry

-- Problem 5
theorem gamma_recursive (hs : s > 0) :
  Γ(s + 1) = s * Γ(s) :=
sorry

-- Problem 6
theorem beta_gamma_relation (hα : α > 0) (hβ : β > 0) :
  B(α, β) = Γ(α) * Γ(β) / Γ(α + β) :=
sorry

end IntegrationProofs

end integral_repr_factorial_integral_new_var_subst_integral_limit_log_gamma_second_repr_gamma_recursive_beta_gamma_relation_l781_781294


namespace final_coordinate_S_l781_781351

variable (S : ℝ × ℝ)
variable (S' : ℝ × ℝ)
variable (S'' : ℝ × ℝ)

-- Initial condition for point S
def S := (5 : ℝ, 0 : ℝ)

-- Reflect S across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Apply the reflection across x-axis to get S'
def S' := reflect_x S

-- Reflect a point across the line y = -x
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Apply the reflection across y = -x to get S''
def S'' := reflect_y_eq_neg_x S'

theorem final_coordinate_S'' :
  S'' = (0, -5) := by
  sorry

end final_coordinate_S_l781_781351


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781536

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781536


namespace reverse_order_possible_l781_781795

def valid_operation (a b c : ℝ) : ℝ × ℝ × ℝ := (c, a - (1/10), b + (1/10))

theorem reverse_order_possible :
  ∃ operations : list (ℝ × ℝ × ℝ → ℝ × ℝ × ℝ), 
  let initial_state := list.range 31.map (λ n, n + 1 : ℝ) in
  let final_state := initial_state.reverse in
  apply_operations operations initial_state = final_state :=
begin
  sorry  -- Proof will go here
end

-- Helper function to apply a list of operations
def apply_operations : list (ℝ × ℝ × ℝ → ℝ × ℝ × ℝ) → list ℝ → list ℝ
| []     state := state
| (op::ops) state := 
    let (a, b, c) := (state.nth 0, state.nth 1, state.nth 2) in
    match (a, b, c) with
    | (some a, some b, some c) := apply_operations ops ((op a b c).to_list)
    | _ := state -- Invalid state or insufficient elements; unchanged
    end

end reverse_order_possible_l781_781795


namespace vasya_made_a_mistake_l781_781572

theorem vasya_made_a_mistake :
  ∀ x : ℝ, x^4 - 3*x^3 - 2*x^2 - 4*x + 1 = 0 → ¬ x < 0 :=
by sorry

end vasya_made_a_mistake_l781_781572


namespace distinct_solution_condition_l781_781655

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l781_781655


namespace first_dog_bones_l781_781961

theorem first_dog_bones (x : ℕ) 
  (h1 : ∀ x, (x + (x - 1) + 2*(x - 1) + 1 + 2 = 12)) : 
  x = 3 :=
by {
  intro x,
  have h_bones : x + (x - 1) + 2 * (x - 1) + 1 + 2 = 12 := sorry,
  calc
    x = 3 : sorry
}  

end first_dog_bones_l781_781961


namespace initial_rope_length_is_50_l781_781981

noncomputable def rope_initial_length (R : ℝ) : Prop :=
  let art_rope := (1/5) * R
  let remaining_after_art := R - art_rope
  let given_to_friend := (1/2) * remaining_after_art
  let remaining_after_giving := remaining_after_art - given_to_friend
  let sections := 10
  let section_length := 2
  remaining_after_giving = sections * section_length

theorem initial_rope_length_is_50 : rope_initial_length 50 :=
by
  let R := 50
  let art_rope := (1/5) * R
  let remaining_after_art := R - art_rope
  let given_to_friend := (1/2) * remaining_after_art
  let remaining_after_giving := remaining_after_art - given_to_friend
  let sections := 10
  let section_length := 2
  have h : remaining_after_giving = sections * section_length := by
    calc
      remaining_after_giving
          = remaining_after_art - given_to_friend : by rfl
      ... = (4/5 * R) - (2/5 * R) : by rw [(1/2) * (4/5 * R), <-mul_div_assoc, <-div_eq_mul_one_div]; ring
      ... = 2/5 * R : by ring
      ... = 20 : by simp [R]
  rw [h] -- shows that this fulfills the condition
  sorry

end initial_rope_length_is_50_l781_781981


namespace balls_in_boxes_l781_781228

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781228


namespace balls_in_boxes_l781_781215

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781215


namespace parabola_equation_and_area_min_l781_781694

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt (((p1.1 - p2.1) ^ 2) + ((p1.2 - p2.2) ^ 2))

theorem parabola_equation_and_area_min :
    (∃ (p : ℝ) (M : ℝ × ℝ), p > 0 ∧ M = (3, real.sqrt (6)) ∧ distance M (3 + (p / 2), 0) = 4 ∧
     ∀ (x y : ℝ), (y^2 = 2 * p * x) ↔ (y^2 = 4 * x)) ∧
    (∀ (A B : ℝ × ℝ) (l : ℝ → ℝ)
        (O : ℝ × ℝ), O = (4, 0) →
        l 0 = 4 ∧ 
        ((A.1^2 * B.1^2 = 4 * A.2 * l(O.2)) ∧ 
         (l(A.2) = B.2) ∧ 
         A ≠ B) → 
      ∃ min_area, (min_area = 16)) :=
by
  sorry

end parabola_equation_and_area_min_l781_781694


namespace range_of_a_l781_781183

theorem range_of_a (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, 0 ≤ x → (f x = if x > 2 then (1/2)^x + 1 else (5/16)*x^2))
  (h3 : ∀ x, [f x]^2 + (a * f x) + b = 0)
  (h4 : (∃! r : {x // x ≥ 0}, f r = 5/4 ∧ f r = 0)) : 
  (a ∈ set.Ioo (-5/2) (-9/4) ∪ set.Ioc (-9/4) (-1)) :=
sorry

end range_of_a_l781_781183


namespace correct_result_l781_781910

theorem correct_result (y : ℤ) (h : 4 * y + 7 = 39) : (y + 7) * 4 = 60 :=
by
  -- Define the number y and the incorrect method yielding 39
  have hy : y = 8 := by
    linarith [h]
  -- Substituting y = 8 in the correct method
  rw [hy, add_comm]
  -- Simplifying the final expression
  norm_num
  done

-- Shell for the proof
sorry

end correct_result_l781_781910


namespace balls_in_boxes_l781_781217

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781217


namespace max_angle_between_skew_lines_l781_781881

-- Define the angle between the oblique line and the plane
def angle_between_line_and_plane (l : Line) (α : Plane) : ℝ := 30

-- Define that the maximum angle formed between the oblique line and any line in the plane α that does not pass through the foot of the perpendicular is 90 degrees
theorem max_angle_between_skew_lines 
(oblique_line : Line) 
(plane_α : Plane)
(h_angle : angle_between_line_and_plane oblique_line plane_α = 30) :
  ∃ l ∈ (lines_in_plane_not_through_perpendicular oblique_line plane_α), angle_between_lines oblique_line l = 90 := 
sorry

end max_angle_between_skew_lines_l781_781881


namespace floor_sqrt_50_l781_781607

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781607


namespace sum_of_coefficients_l781_781742

theorem sum_of_coefficients :
  ∀ (a_0 a_1 ... a_2009 : ℝ), (1 - 2 * x) ^ 2009 = a_0 + a_1 * x + ... + a_2009 * x ^ 2009 →
  a_0 + a_1 + ... + a_2009 = -1 :=
begin
  sorry
end

end sum_of_coefficients_l781_781742


namespace vector_parallel_to_sum_l781_781990

structure Vector2D where
  x : ℝ
  y : ℝ

def sequence (n : ℕ) : Vector2D := sorry -- assuming definition of sequence

def vector_sum (n : ℕ) (f : ℕ → Vector2D) : Vector2D :=
  (List.range (n+1)).map f |>.foldl (λ a b => Vector2D.mk (a.x + b.x) (a.y + b.y)) ⟨0, 0⟩

def is_arithmetic_sequence (f : ℕ → Vector2D) (d : Vector2D) : Prop :=
  ∀ n ≥ 1, f n = Vector2D.mk (f 0).x (f 0).y + (n * d)

theorem vector_parallel_to_sum :
  ∀ (d : Vector2D) (f : ℕ → Vector2D),
    is_arithmetic_sequence f d →
    (vector_sum 21 f) = 21 • f 10 →
    (vector_sum 21 f).x * (f 10).y = (vector_sum 21 f).y * (f 10).x :=
by sorry

end vector_parallel_to_sum_l781_781990


namespace tan_neg_two_simplifies_l781_781759

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l781_781759


namespace distance_at_40_kmph_l781_781437

theorem distance_at_40_kmph (x : ℝ) (h1 : x / 40 + (250 - x) / 60 = 5) : x = 100 := 
by
  sorry

end distance_at_40_kmph_l781_781437


namespace coeffs_of_quadratic_eq_l781_781097

theorem coeffs_of_quadratic_eq :
  ∃ (a b c : ℤ), (3 : ℤ) * x^2 + (4 : ℤ) * x - (2 : ℤ) = 0 :=
begin
  sorry
end

end coeffs_of_quadratic_eq_l781_781097


namespace ball_distribution_l781_781236

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781236


namespace find_units_min_selling_price_l781_781412

-- Definitions for the given conditions
def total_units : ℕ := 160
def cost_A : ℕ := 150
def cost_B : ℕ := 350
def total_cost : ℕ := 36000
def min_profit : ℕ := 11000

-- Part 1: Proving number of units purchased
theorem find_units :
  ∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x :=
by
  sorry

-- Part 2: Finding the minimum selling price per unit of model A for the profit condition
theorem min_selling_price (t : ℕ) :
  (∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x) →
  100 * (t - cost_A) + 60 * 2 * (t - cost_A) ≥ min_profit →
  t ≥ 200 :=
by
  sorry

end find_units_min_selling_price_l781_781412


namespace hyperbola_eccentricity_l781_781903

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h_right_focus : ∀ x y, x = c ∧ y = 0)
    (h_circle : ∀ x y, (x - c)^2 + y^2 = 4 * a^2)
    (h_tangent : ∀ x y, x = c ∧ y = 0 → (x^2 + y^2 = a^2 + b^2))
    : ∃ e : ℝ, e = sqrt 5 := by sorry

end hyperbola_eccentricity_l781_781903


namespace calc_expression_l781_781574

theorem calc_expression : real.sqrt 9 + 2⁻¹ + (-1 : ℝ) ^ 2023 = 5 / 2 := by
  sorry

end calc_expression_l781_781574


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781509

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781509


namespace find_vector_BC_l781_781054

variables (A B C D E O : Type) [AddCommGroup A] [VectorSpace ℝ A]
variables (a b : A)
variables [finite_dimensional ℝ A]

-- Let's consider the definitions from the problem as conditions
variable (AD_median : is_median A B C D)
variable (BE_median : is_median B A C E)
variable (vec_AD : (D : A) - (A : A) = a)
variable (vec_BE : (E : A) - (B : A) = b)

-- Definition for vector BC
def vec_BC := 4 / 3 • b + 2 / 3 • a

-- The theorem to prove the statement
theorem find_vector_BC 
  (AD_median : is_median A B C D)
  (BE_median : is_median B A C E)
  (vec_AD : (D : A) - (A : A) = a)
  (vec_BE : (E : A) - (B : A) = b):
  (C : A) - (B : A) = vec_BC a b := by
   sorry

end find_vector_BC_l781_781054


namespace roots_of_equation_l781_781646

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l781_781646


namespace forest_problem_l781_781493

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781493


namespace rope_length_after_sixth_cut_l781_781964

theorem rope_length_after_sixth_cut : 
  let initial_length := 1
  let cut_function := (λ x : ℝ, x / 2)
  let final_length := cut_function (cut_function (cut_function (cut_function (cut_function (cut_function initial_length)))))
  final_length = (1 / 2) ^ 6 :=
by
  sorry

end rope_length_after_sixth_cut_l781_781964


namespace estimated_total_volume_correct_l781_781521

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781521


namespace not_super_lucky_years_l781_781094

def sum_of_month_and_day (m d : ℕ) : ℕ := m + d
def product_of_month_and_day (m d : ℕ) : ℕ := m * d
def sum_of_last_two_digits (y : ℕ) : ℕ :=
  let d1 := y / 10 % 10
  let d2 := y % 10
  d1 + d2

def is_super_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), sum_of_month_and_day m d = 24 ∧
               product_of_month_and_day m d = 2 * sum_of_last_two_digits y

theorem not_super_lucky_years :
  ¬ is_super_lucky_year 2070 ∧
  ¬ is_super_lucky_year 2081 ∧
  ¬ is_super_lucky_year 2092 :=
by {
  sorry
}

end not_super_lucky_years_l781_781094


namespace balls_in_boxes_ways_l781_781214

theorem balls_in_boxes_ways : ∀ (balls boxes : ℕ), balls = 7 → boxes = 3 → 
  (∃ (f : ℕ → ℕ), 
    (∀ x, x < boxes → f x > 0) ∧ 
    (∑ x in finset.range boxes, f x = balls)
  ) → 15 :=
by
  intros balls boxes hballs hboxes hexists
  have h_comb : ∑ x in finset.range boxes, f x = 7 := sorry -- use of the hballs and hboxes
  have non_empty_boxes : ∀ x, x < boxes → f x > 0 := sorry -- given each box must contain at least one ball
  exact 15

end balls_in_boxes_ways_l781_781214


namespace tan_neg_two_simplifies_l781_781762

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l781_781762


namespace black_region_probability_l781_781088

theorem black_region_probability
  (square_side : ℝ) (white_circle_radius : ℝ) (black_square_side : ℝ)
  (coin_diameter : ℝ) (black_squares_distance_to_center : ℝ)
  (black_square_positions : fin 4 → ℝ × ℝ) :
  square_side = 10 →
  white_circle_radius = 3 →
  black_square_side = 1 →
  coin_diameter = 2 →
  (∀ i : fin 4, (black_square_positions i).1 = if i = 0 then -5 else if i = 1 then 5 else if i = 2 then -5 else 5) →
  (∀ i : fin 4, (black_square_positions i).2 = if i = 0 then -5 else if i = 1 then 5 else if i = 2 then 5 else -5) →
  black_squares_distance_to_center = 5 →
  let probability := (36/64 : ℝ) in
  probability = 9 / 16 :=
sorry

end black_region_probability_l781_781088


namespace divide_equally_l781_781306

-- Define the input values based on the conditions.
def brother_strawberries := 3 * 15
def kimberly_strawberries := 8 * brother_strawberries
def parents_strawberries := kimberly_strawberries - 93
def total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
def family_members := 4

-- Define the theorem to prove the question.
theorem divide_equally : 
    (total_strawberries / family_members) = 168 :=
by
    -- (proof goes here)
    sorry

end divide_equally_l781_781306


namespace distinct_triangles_with_positive_integer_areas_l781_781124

def number_of_distinct_triangles : ℕ :=
  3540

theorem distinct_triangles_with_positive_integer_areas :
  (∃ P Q : ℕ × ℕ, P ≠ Q ∧ 
    (17 * P.1 + P.2 = 2030) ∧ 
    (17 * Q.1 + Q.2 = 2030) ∧ 
    (1 / 2) * |P.1 * Q.2 - Q.1 * P.2| ∈ ℕ) → 
  number_of_distinct_triangles = 3540 :=
by
  sorry

end distinct_triangles_with_positive_integer_areas_l781_781124


namespace how_many_large_glasses_l781_781352

theorem how_many_large_glasses (cost_small cost_large : ℕ) 
                               (total_money money_left change : ℕ) 
                               (num_small : ℕ) : 
  cost_small = 3 -> 
  cost_large = 5 -> 
  total_money = 50 -> 
  money_left = 26 ->
  change = 1 ->
  num_small = 8 ->
  (money_left - change) / cost_large = 5 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end how_many_large_glasses_l781_781352


namespace general_formula_a_n_bound_on_T_n_l781_781057
open Nat

-- Define sequence a_n and condition
def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^n

def sum_S (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2 * sequence_a n - 2

-- 1. Prove the general formula for the sequence a_n
theorem general_formula_a_n (n : ℕ) (h_pos : 0 < n) :
  sequence_a n = 2^n :=
sorry

-- Define sequence b_n and sum T_n
def sequence_b (n : ℕ) : ℝ :=
  sorry -- Need more context on b_n value

def sum_T (n : ℕ) : ℝ :=
  if n = 0 then 0 else ∑ i in range n, sequence_b i

-- 2. Prove that for any positive integer n, it always holds that T_n < 2
theorem bound_on_T_n (n : ℕ) (h_pos : 0 < n) :
  sum_T n < 2 :=
sorry

end general_formula_a_n_bound_on_T_n_l781_781057


namespace polar_to_rectangular_l781_781989

theorem polar_to_rectangular (r θ : ℝ) (x y : ℝ) 
  (hr : r = 10) 
  (hθ : θ = (3 * Real.pi) / 4) 
  (hx : x = r * Real.cos θ) 
  (hy : y = r * Real.sin θ) 
  :
  x = -5 * Real.sqrt 2 ∧ y = 5 * Real.sqrt 2 := 
by
  -- We assume that the problem is properly stated
  -- Proof omitted here
  sorry

end polar_to_rectangular_l781_781989


namespace estimated_total_volume_correct_l781_781529

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781529


namespace compute_expression_l781_781984

theorem compute_expression :
    ( (2 / 3) * Real.sqrt 15 - Real.sqrt 20 ) / ( (1 / 3) * Real.sqrt 5 ) = 2 * Real.sqrt 3 - 6 :=
by
  sorry

end compute_expression_l781_781984


namespace kopecks_problem_l781_781358

theorem kopecks_problem (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b :=
sorry

end kopecks_problem_l781_781358


namespace complement_intersection_l781_781734

open Finset

-- Definitions of sets
def I : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {1, 3, 5}
def B : Finset ℕ := {2, 3, 6}
def C (S : Finset ℕ) : Finset ℕ := I \ S

theorem complement_intersection :
  (C A ∩ B) = {2, 6} := by
  sorry

end complement_intersection_l781_781734


namespace last_score_entered_is_75_l781_781842

theorem last_score_entered_is_75 (scores : List ℕ) (h : scores = [62, 75, 83, 90]) :
  ∃ last_score, last_score ∈ scores ∧ 
    (∀ (num list : List ℕ), list ≠ [] → list.length ≤ scores.length → 
    ¬ list.sum % list.length ≠ 0) → 
  last_score = 75 :=
by
  sorry

end last_score_entered_is_75_l781_781842


namespace question_1_question_2_question_3_l781_781799

-- Define the sequence a_n and related conditions
def a : ℕ → ℕ := sorry
def d1 : ℕ := sorry
def d2 : ℕ := sorry

-- Condition definitions as previously identified in the problem
def sequence_conditions : Prop :=
  (a 1 = 1) ∧  
  (a 2 = 2) ∧ 
  (∀ n, odd n → a n = 1 + (n - 1)/2 * d1) ∧
  (∀ n, even n → a n = 2 + (n/2 - 1) * d2)

-- Question 1
theorem question_1 : sequence_conditions ∧ (S 5 = 16) ∧ (a 4 = a 5) → a 10 = 14 := 
sorry

-- Question 2
theorem question_2 : sequence_conditions ∧ (S 15 = 15 * a 8) ∧ (∀ n, a n < a (n + 1)) → (∀ n, a n = n) := 
sorry

-- Question 3
theorem question_3 : sequence_conditions ∧ (d1 = 3 * d2) ∧ (d1 ≠ 0) ∧ (∃ m n, m ≠ n ∧ a m = a n) → 
  (∀ n, (odd n → a n = 3/2 * n - 1/2) ∧ (even n → a n = n/2 + 1)) := 
sorry

end question_1_question_2_question_3_l781_781799


namespace polynomial_ratio_le_l781_781332

noncomputable theory
open_locale classical real

variables {R : Type*} [linear_ordered_field R]

def polynomial (n : ℕ) := R → R

variables (n : ℕ)
variables (a : ℝ) (c : ℝ)

def f (x : ℝ) := ∑ i in finset.range (n + 1), (λ i : ℕ, a) * (x ^ i)
def g (x : ℝ) := ∑ i in finset.range (n + 2), (λ i : ℕ, c) * (x ^ i)

theorem polynomial_ratio_le (r : ℝ) (a n c : ℝ)
  (hf : f n a = ∑ i in finset.range (n + 1), (λ i : ℕ, a) * (r ^ i))
  (hg : g n c = ∑ i in finset.range (n + 2), (λ i : ℕ, c) * (r ^ i))
  (ha : a = finset.sup' (finset.range (n + 1)) (λ i, |a|))
  (hc : c = finset.sup' (finset.range (n + 2)) (λ i, |c|)) :
  a / c ≤ n + 1 :=
sorry

end polynomial_ratio_le_l781_781332


namespace statement_skew_lines_planes_infinitely_many_pairs_l781_781317

/-
Theorem statement:
Given lines a and b are skew and form an angle of 30 degrees between them,
prove that there are infinitely many pairs of planes alpha and beta such that:
- a is contained in alpha
- b is contained in beta
- alpha is perpendicular to beta
-/
theorem skew_lines_planes_infinitely_many_pairs
  (a b : Line) (alpha beta : Plane)
  (h_skew : skew a b) (h_angle : angle_between a b = 30) :
  ∃ (pairs : Set (Plane × Plane)), (∞ ∈ pairs) ∧ 
  (∀ (p : pairs), 
    (a ⊆ p.1) ∧ 
    (b ⊆ p.2) ∧ 
    (p.1 ⊥ p.2)) := by
  sorry

end statement_skew_lines_planes_infinitely_many_pairs_l781_781317


namespace part1_part2_l781_781177

namespace MathProblems

-- Definitions of sets A and B
def A := λ x : ℝ, -3 ≤ x ∧ x < 4
def B (m : ℝ) := λ x : ℝ, 2 * m - 1 ≤ x ∧ x ≤ m + 1

-- Part (1): Prove that if B ⊆ A, then m ∈ [-1, +∞)
theorem part1 (m : ℝ) (h1 : ∀ x, B m x → A x) : m ∈ Ici (-1) := sorry

-- Part (2): Prove that if ∃ x ∈ A such that x ∈ B, then m ∈ [-4, 2]
theorem part2 (m : ℝ) (h2 : ∃ x, A x ∧ B m x) : m ∈ Icc (-4) 2 := sorry

end MathProblems

end part1_part2_l781_781177


namespace find_some_number_l781_781768

-- The conditions of the problem
variables (x y : ℝ)
axiom cond1 : 2 * x + y = 7
axiom cond2 : x + 2 * y = 5

-- The "some number" we want to prove exists
def some_number := 3

-- Statement of the problem: the value of 2xy / some_number should equal 2
theorem find_some_number (x y : ℝ) (cond1 : 2 * x + y = 7) (cond2 : x + 2 * y = 5) :
  2 * x * y / some_number = 2 :=
sorry

end find_some_number_l781_781768


namespace base_six_product_correct_l781_781117

namespace BaseSixProduct

-- Definitions of the numbers in base six
def num1_base6 : ℕ := 1 * 6^2 + 3 * 6^1 + 2 * 6^0
def num2_base6 : ℕ := 1 * 6^1 + 4 * 6^0

-- Their product in base ten
def product_base10 : ℕ := num1_base6 * num2_base6

-- Convert the base ten product back to base six
def product_base6 : ℕ := 2 * 6^3 + 3 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Theorem statement
theorem base_six_product_correct : product_base10 = 560 ∧ product_base6 = 2332 := by
  sorry

end BaseSixProduct

end base_six_product_correct_l781_781117


namespace trigonometric_expression_evaluation_l781_781752

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l781_781752


namespace total_shoes_l781_781955

variables (people : ℕ) (shoes_per_person : ℕ)

-- There are 10 people
axiom h1 : people = 10
-- Each person has 2 shoes
axiom h2 : shoes_per_person = 2

-- The total number of shoes kept outside the library is 10 * 2 = 20
theorem total_shoes (people shoes_per_person : ℕ) (h1 : people = 10) (h2 : shoes_per_person = 2) : people * shoes_per_person = 20 :=
by sorry

end total_shoes_l781_781955


namespace f_four_times_even_l781_781828

variable (f : ℝ → ℝ) (x : ℝ)

-- Definition stating f is an odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem f_four_times_even (h : is_odd f) : is_even (f (f (f (f x)))) :=
by sorry

end f_four_times_even_l781_781828


namespace rectangle_perimeter_l781_781485

theorem rectangle_perimeter (s : ℕ) (h : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end rectangle_perimeter_l781_781485


namespace find_m_l781_781392

-- The coefficient m^2 - m - 1 must be positive
def coeff_positive (m : ℝ) : Prop := m^2 - m - 1 > 0

-- The exponent m^2 - 2m - 3 must be negative
def exponent_negative (m : ℝ) : Prop := m^2 - 2m - 3 < 0

-- Combining the conditions
def function_decreasing (m : ℝ) : Prop := coeff_positive m ∧ exponent_negative m

-- Prove the only integer value for m that satisfies the conditions is 2
theorem find_m (m : ℝ) (hm_int : m = 2) : function_decreasing m :=
by
  sorry

end find_m_l781_781392


namespace minimum_daily_production_to_avoid_losses_l781_781399

theorem minimum_daily_production_to_avoid_losses (x : ℕ) :
  (∀ x, (10 * x) ≥ (5 * x + 4000)) → (x ≥ 800) :=
sorry

end minimum_daily_production_to_avoid_losses_l781_781399


namespace num_ways_distribute_balls_l781_781251

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781251


namespace distinct_points_count_l781_781129

-- Definitions based on conditions
def eq1 (x y : ℝ) : Prop := (x + y = 7) ∨ (2 * x - 3 * y = -7)
def eq2 (x y : ℝ) : Prop := (x - y = 3) ∨ (3 * x + 2 * y = 18)

-- The statement combining conditions and requiring the proof of 3 distinct solutions
theorem distinct_points_count : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (eq1 p1.1 p1.2 ∧ eq2 p1.1 p1.2) ∧ 
    (eq1 p2.1 p2.2 ∧ eq2 p2.1 p2.2) ∧ 
    (eq1 p3.1 p3.2 ∧ eq2 p3.1 p3.2) ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
sorry

end distinct_points_count_l781_781129


namespace chessboard_flour_distribution_l781_781349

theorem chessboard_flour_distribution 
  (n k : ℕ)
  (flour_distribution : Fin n → Fin n → ℝ) 
  (h_dist_sum_rows : ∀ i : Fin n, ∑ j : Fin n, flour_distribution i j = 1) 
  (h_dist_sum_cols : ∀ j : Fin n, ∑ i : Fin n, flour_distribution i j = 1) 
  (h_pos : ∀ (i j : Fin n), 0 ≤ flour_distribution i j) :
  ∃ (c : Fin k → ℝ) 
    (remove_flour : Fin k → Fin n → Fin n → Prop),
    (∀ (i : Fin k), 
      ∑ (x : Fin n), ∑ (y : Fin n), if remove_flour i x y then flour_distribution x y else 0 = c i
    ) ∧
    (∀ (i : Fin k), 
      ∀ (x y : Fin n),
        remove_flour i x y → flour_distribution x y = 0) ∧
    (∀ (i : Fin n), 
      ∑ (x : Fin k), remove_flour x i i) :=
  sorry

end chessboard_flour_distribution_l781_781349


namespace estimated_total_volume_correct_l781_781525

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781525


namespace sum_primes_less_than_20_greater_than_3_is_72_l781_781032

noncomputable def is_prime (n : ℕ) : Prop :=
nat.prime n

def primes_less_than_20 : List ℕ :=
List.filter (λ n, is_prime n) (List.range 20)

def primes_less_than_20_greater_than_3 : List ℕ :=
List.filter (λ n, 3 < n) primes_less_than_20

def sum_primes_less_than_20_greater_than_3 : ℕ :=
List.sum primes_less_than_20_greater_than_3

theorem sum_primes_less_than_20_greater_than_3_is_72 : sum_primes_less_than_20_greater_than_3 = 72 :=
by
  sorry

end sum_primes_less_than_20_greater_than_3_is_72_l781_781032


namespace coeff_a2_l781_781162

-- Definition of the polynomial f(x)
def f (x : ℕ) : ℕ := (1 + x) + (1 + x)^2 + (1 + x)^3 + ... + (1 + x)^10

-- Stating the main proposition that a₂ = 165
theorem coeff_a2 {x : ℕ} :
  let f (x : ℕ) := (1 + x) + (1 + x)^2 + (1 + x)^3 + ... + (1 + x)^10 in
  let a₂ := coeff x^2 in
  a₂ = 165 :=
by sorry

end coeff_a2_l781_781162


namespace number_of_liberal_arts_in_sample_l781_781956

def total_students : ℕ := 1000
def liberal_arts_students : ℕ := 200
def sample_size : ℕ := 100

theorem number_of_liberal_arts_in_sample 
  (total_students = 1000) 
  (liberal_arts_students = 200) 
  (sample_size = 100) 
  (stratified_sampling : ℕ → Prop) 
  : stratified_sampling 20 :=
sorry


end number_of_liberal_arts_in_sample_l781_781956


namespace more_pie_eaten_l781_781108

theorem more_pie_eaten (e f : ℝ) (h1 : e = 0.67) (h2 : f = 0.33) : e - f = 0.34 :=
by sorry

end more_pie_eaten_l781_781108


namespace forest_trees_properties_l781_781518

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781518


namespace coin_toss_probability_l781_781978

-- Defining the problem conditions
def unfair_coin_probability : ℝ := 3 / 4
def num_tosses : ℕ := 60
def successful_heads : ℕ := 45

-- Lean proof statement
theorem coin_toss_probability :
  Pr { outcomes : ℕ // outcomes > successful_heads | binomial num_tosses unfair_coin_probability } = 0.9524 :=
sorry

end coin_toss_probability_l781_781978


namespace probability_of_red_first_then_green_is_correct_l781_781278

noncomputable def prob_red_first_green_second : ℚ :=
  let total_pieces := 32
  let red_pieces := 16
  let green_pieces := 16
  let prob_red_first := (red_pieces : ℚ) / (total_pieces : ℚ)
  let prob_green_second := (green_pieces : ℚ) / (total_pieces - 1 : ℚ)
  prob_red_first * prob_green_second

theorem probability_of_red_first_then_green_is_correct :
  prob_red_first_green_second = 8 / 31 :=
by
  unfold prob_red_first_green_second
  norm_num
  sorry

end probability_of_red_first_then_green_is_correct_l781_781278


namespace balls_in_boxes_l781_781231

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781231


namespace find_value_l781_781046

theorem find_value : 3 + 2 * (8 - 3) = 13 := by
  sorry

end find_value_l781_781046


namespace solve_for_x_l781_781447

theorem solve_for_x : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by
  use -5
  sorry

end solve_for_x_l781_781447


namespace find_matrix_l781_781669

-- Define the vectors i, j, k
def i : Matrix (Fin 3) (Fin 1) ℤ := ![![3], ![4], ![-9]]
def j : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![6], ![-3]]
def k : Matrix (Fin 3) (Fin 1) ℤ := ![![8], ![-2], ![5]]

-- Define the matrix N
def N : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, 8], ![4, 6, -2], ![-9, -3, 5]]

-- Define the standard basis vectors
def e1 : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![0], ![0]]
def e2 : Matrix (Fin 3) (Fin 1) ℤ := ![![0], ![1], ![0]]
def e3 : Matrix (Fin 3) (Fin 1) ℤ := ![![0], ![0], ![1]]

-- The problem statement: prove that given the conditions, N is the specified matrix
theorem find_matrix (N : Matrix (Fin 3) (Fin 3) ℤ) 
  (h1 : N.mulVec e1 = i) 
  (h2 : N.mulVec e2 = j) 
  (h3 : N.mulVec e3 = k) : 
  N = ![![3, 1, 8], ![4, 6, -2], ![-9, -3, 5]] :=
sorry -- proof to be provided

end find_matrix_l781_781669


namespace min_value_proof_l781_781820

noncomputable def min_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 3 * y = 1) : ℝ :=
  if hx : x = 0 ∨ y = 0 then 
    0 -- this case will not occur due to the h₁ and h₂ constraints
  else
    let a := (1 / x) + (1 / y)
    in 5 + 3 * Real.sqrt 3

theorem min_value_proof (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 3 * y = 1) :
  min_value x y h₁ h₂ h₃ = 5 + 3 * Real.sqrt 3 :=
by
  sorry

end min_value_proof_l781_781820


namespace balls_in_boxes_l781_781262

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781262


namespace laser_beam_total_distance_l781_781470

theorem laser_beam_total_distance :
  let A := (3, 5)
  let D := (7, 5)
  let D'' := (-7, -5)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  distance A D'' = 10 * Real.sqrt 2 :=
by
  -- definitions and conditions are captured
  sorry -- the proof goes here, no proof is required as per instructions

end laser_beam_total_distance_l781_781470


namespace binom_n_n_sub_2_l781_781922

theorem binom_n_n_sub_2 (n : ℕ) (h : n > 0) : (Nat.choose n (n - 2)) = (n * (n - 1)) / 2 := by
  sorry

end binom_n_n_sub_2_l781_781922


namespace find_complex_solution_l781_781167

noncomputable def z (z : ℂ) : Prop :=
  let z1 := z / (1 + z^2)
  let z2 := z^2 / (1 + z)
  z1.im = 0 ∧ z2.im = 0

theorem find_complex_solution (z : ℂ) (h : z z) : z = -1/2 + complex.I * real.sqrt(3)/2 ∨ z = -1/2 - complex.I * real.sqrt(3)/2 :=
sorry

end find_complex_solution_l781_781167


namespace number_of_ways_to_put_balls_in_boxes_l781_781244

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781244


namespace problem_proof_l781_781203

open Set

noncomputable def A : Set ℝ := {x | abs (4 * x - 1) < 9}
noncomputable def B : Set ℝ := {x | x / (x + 3) ≥ 0}
noncomputable def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 5 / 2}
noncomputable def correct_answer : Set ℝ := Iio (-3) ∪ Ici (5 / 2)

theorem problem_proof : (compl A) ∩ B = correct_answer := 
  by
    sorry

end problem_proof_l781_781203


namespace intersection_points_at_three_l781_781133

noncomputable def curve_intersections (a : ℝ) : ℂ :=
  (x - 1)^2 + y^2 = a^2 ∧ y = x^2 - a

theorem intersection_points_at_three (a : ℝ) :
  (∃ p : set (ℝ × ℝ), curve_intersections p a ∧ set.card p = 3) ↔
  (a = (3 + real.sqrt 5) / 2 ∨ a = (3 - real.sqrt 5) / 2) :=
by sorry

end intersection_points_at_three_l781_781133


namespace fib_50_mod_5_l781_781876

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end fib_50_mod_5_l781_781876


namespace prime_factorial_division_l781_781319

theorem prime_factorial_division (p k n : ℕ) (hp : Prime p) (h : p^k ∣ n!) : (p!)^k ∣ n! :=
sorry

end prime_factorial_division_l781_781319


namespace estimated_total_volume_correct_l781_781526

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781526


namespace john_total_revenue_2_months_l781_781304

noncomputable def calculate_friday_revenue : ℕ := 
  let first_12_hours := 4000 * 12
  let remaining_14_hours := 4400 * 14
  first_12_hours + remaining_14_hours

noncomputable def calculate_saturday_revenue : ℕ :=
  let first_12_hours := 5000 * 12
  let remaining_14_hours := 6000 * 14
  first_12_hours + remaining_14_hours

noncomputable def calculate_sunday_revenue : ℕ :=
  let day_initial_rate := 5000
  let first_10_hours := let initial_revenue := 4250 in initial_revenue * 10
  let fluctuating_revenues := 
      let rate := 4250 in
      (2 * (rate + (rate * 5 / 100))) + 
      (4 * (rate + (rate * 30 / 100))) + 
      (2 * (rate - (rate * 10 / 100))) + 
      (1 * (rate + (rate * 20 / 100))) + 
      (7 * (rate - (rate * 25 / 100)))
  first_10_hours + fluctuating_revenues

noncomputable def calculate_weekend_revenue : ℕ :=
  calculate_friday_revenue + calculate_saturday_revenue + calculate_sunday_revenue

noncomputable def calculate_total_revenue (num_weeks : ℕ) : ℕ :=
  num_weeks * calculate_weekend_revenue 

theorem john_total_revenue_2_months : calculate_total_revenue 8 = 2849500 := 
  by sorry

end john_total_revenue_2_months_l781_781304


namespace incorrect_calculation_l781_781431

theorem incorrect_calculation :
  (4 + (-2) = 2) ∧
  (-2 - (-1.5) = -0.5) ∧
  (-(-4) + 4 = 8) ∧
  (| -6 | + | 2 | ≠ 4) := by
  sorry

end incorrect_calculation_l781_781431


namespace total_amount_is_200_l781_781571

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l781_781571


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781531

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781531


namespace floor_sqrt_50_l781_781613

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781613


namespace area_ratio_l781_781937

noncomputable def pentagon_area (R s : ℝ) := (5 / 2) * R * s * Real.sin (Real.pi * 2 / 5)
noncomputable def triangle_area (s : ℝ) := (s^2) / 4

theorem area_ratio (R s : ℝ) (h : R = s / (2 * Real.sin (Real.pi / 5))) :
  (pentagon_area R s) / (triangle_area s) = 5 * (Real.sin ((2 * Real.pi) / 5) / Real.sin (Real.pi / 5)) :=
by
  sorry

end area_ratio_l781_781937


namespace sum_of_consecutive_pages_l781_781453

theorem sum_of_consecutive_pages (n : ℕ) 
  (h : n * (n + 1) = 20412) : n + (n + 1) + (n + 2) = 429 := by
  sorry

end sum_of_consecutive_pages_l781_781453


namespace intersection_point_on_ellipse_min_distance_CD_l781_781703

noncomputable def ellipse_params : ℝ × ℝ :=
  (4, 1) -- values for a^2 and b^2

def standard_equation (a : ℝ) (b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_on_ellipse (x y : ℝ) : Prop :=
  let (a, b) := ellipse_params in
  standard_equation a b x y

theorem intersection_point_on_ellipse (a : ℝ) :
  a > 1 → is_on_ellipse (3 * a / 5) (4 / 5) :=
by
  intro h
  let (a_sq, b_sq) := ellipse_params
  have ha : a_sq = 4 := rfl
  have hb : b_sq = 1 := rfl
  rw [ha, hb]
  sorry

theorem min_distance_CD (a : ℝ) :
  a > 1 → P = (-Q) → minimum_distance 6 :=
by
  intro h1 h2
  sorry

end intersection_point_on_ellipse_min_distance_CD_l781_781703


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781502

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781502


namespace arrow_in_48th_position_l781_781771

def arrow_sequence := ["→", "↔", "↓", "→", "↕"]

theorem arrow_in_48th_position :
  arrow_sequence[48 % arrow_sequence.length] = "↓" :=
by
  sorry

end arrow_in_48th_position_l781_781771


namespace abc_arithmetic_sequence_l781_781267

-- Definitions and Conditions
def a := log 2 3
def b := log 2 6
def c := log 2 12

-- Statement of the proof problem
theorem abc_arithmetic_sequence : b - a = 1 ∧ c - b = 1 :=
sorry

end abc_arithmetic_sequence_l781_781267


namespace freeze_time_l781_781800

theorem freeze_time :
  ∀ (minutes_per_smoothie total_minutes num_smoothies freeze_time: ℕ),
    minutes_per_smoothie = 3 →
    total_minutes = 55 →
    num_smoothies = 5 →
    freeze_time = total_minutes - (num_smoothies * minutes_per_smoothie) →
    freeze_time = 40 :=
by
  intros minutes_per_smoothie total_minutes num_smoothies freeze_time
  intros H1 H2 H3 H4
  subst H1
  subst H2
  subst H3
  subst H4
  sorry

end freeze_time_l781_781800


namespace floor_sqrt_50_l781_781639

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781639


namespace X_between_D_and_M_angle_MAX_lt_angle_DAX_if_acute_l781_781490

variables {A B C M X D : Type*} [point_space A] [left_of A M B] [right_of A M C]

theorem X_between_D_and_M 
  (h_ABC : is_triangle ABC)
  (h_AB_AC : AB ≠ AC)
  (h_AM_median : is_median A M)
  (h_AX_angle_bisector : is_angle_bisector A X)
  (h_AD_altitude : is_altitude A D)
  : lies_between X D M :=
sorry

theorem angle_MAX_lt_angle_DAX_if_acute
  (h_ABC : is_triangle ABC)
  (h_ABC_acute : is_acute_angle_triangle ABC)
  (h_AB_AC : AB ≠ AC)
  (h_AM_median : is_median A M)
  (h_AX_angle_bisector : is_angle_bisector A X)
  (h_AD_altitude : is_altitude A D)
  : angle_at X M A < angle_at D A X :=
sorry

end X_between_D_and_M_angle_MAX_lt_angle_DAX_if_acute_l781_781490


namespace correct_statements_l781_781700

namespace MathProof

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 8 ∧ a 2 = 1 ∧ (∀ n, (even n → a (n + 2) = -a n) ∧ (odd n → a (n + 2) = a n - 2))

def sum_seq (a : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  (∀ n, T (n + 1) = T n + a (n + 1)) ∧ T 0 = 0

theorem correct_statements (a : ℕ → ℤ) (T : ℕ → ℤ) 
  (h_seq : sequence a) (h_sum : sum_seq a T) :
  (∀ n, even n → a n = (-1) ^ (n / 2 - 1)) ∧ T 99 = -2049 :=
by 
  sorry

end MathProof

end correct_statements_l781_781700


namespace compare_apothems_l781_781082

-- Definitions of rectangle and hexagon satisfying given conditions
def rectangle_apothem (a b : ℝ) (h1 : a * b = 2 * (a + b)) (h2 : b = 2 * a) : ℝ :=
  a / 2

def hexagon_apothem (s : ℝ) (h3 : 3 * Real.sqrt 3 / 2 * s^2 = 6 * s) : ℝ :=
  Real.sqrt 3 / 2 * s

-- Given conditions specific to the problem
lemma rectangle_has_sides (x : ℝ) (h : x ≠ 0) : rectangle_apothem x (2 * x) (by linarith) (by linarith) = 3/2 :=
  by sorry

lemma hexagon_side_length : ∃ s > 0, hexagon_apothem s (by linarith) = 2 * Real.sqrt 3 :=
  by sorry

-- Main theorem comparing apothems of the rectangle and hexagon
theorem compare_apothems (x : ℝ) (h : x ≠ 0) :
  rectangle_apothem x (2 * x) (by linarith) (by linarith) = 
  (Real.sqrt 3 / 4) * (some (hexagon_side_length).val) :=
  by sorry

end compare_apothems_l781_781082


namespace suzy_current_age_l781_781786

-- Definitions based on conditions
def current_age_mary : Nat := 8
def future_age_mary (n : Nat) : Nat := current_age_mary + n
def future_age_suzy (S : Nat) (n : Nat) : Nat := S + n

-- The proof problem statement
theorem suzy_current_age (S : Nat) (h : future_age_suzy S 4 = 2 * future_age_mary 4) : S = 20 :=
by
  -- actual proof is omitted
  sorry

#eval suzy_current_age 20 -- This should return true because 20 is the correct answer

end suzy_current_age_l781_781786


namespace triangle_solutions_l781_781170

theorem triangle_solutions :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a = 7.012 ∧
  c - b = 1.753 ∧
  B = 38 + 12/60 + 48/3600 ∧
  A = 81 + 47/60 + 12.5/3600 ∧
  C = 60 ∧
  b = 4.3825 ∧
  c = 6.1355 :=
sorry -- Proof goes here

end triangle_solutions_l781_781170


namespace analytical_expression_of_f_l781_781685

theorem analytical_expression_of_f (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f(x + 1) = x^2 - x + 1) : 
  ∀ x : ℝ, f(x) = x^2 - 3x + 3 := 
by
  sorry

end analytical_expression_of_f_l781_781685


namespace cube_surface_area_l781_781005

theorem cube_surface_area (V : ℕ) (hV : V = 1728) : 
  ∃ A : ℕ, A = 864 :=
by
  let s := Fin.pow 3 (nat_root.real_cbrt_eq_nat_cbrt 1728) in
  let A := 6 * s^2 in
  use A
  sorry

end cube_surface_area_l781_781005


namespace find_p_l781_781769

theorem find_p (p : ℕ) (h : 81^6 = 3^p) : p = 24 :=
sorry

end find_p_l781_781769


namespace problem_statement_l781_781756

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l781_781756


namespace derivative_graph_transformation_l781_781667

theorem derivative_graph_transformation {f : ℝ → ℝ} :
  (∀ x, f x = sin (2 * x)) →
  (∀ x, (deriv f) x = 2 * sin (2 * (x + (π / 4)))) :=
by
  intro hf
  have h_derivative : ∀ x, deriv f x = 2 * cos (2 * x) :=
    by sorry -- This step would be solved by applying the chain rule.
  rw [←hf]
  rw [trig_identity] -- This step would utilize the trigonometric identity transformation.
  sorry -- Additional transformations showing the graph transformation.


end derivative_graph_transformation_l781_781667


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781543

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781543


namespace customer_paid_l781_781901

theorem customer_paid (cost_price : ℕ) (markup_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 6672 → markup_percent = 25 → selling_price = cost_price + (markup_percent * cost_price / 100) → selling_price = 8340 :=
by
  intros h_cost_price h_markup_percent h_selling_price
  rw [h_cost_price, h_markup_percent] at h_selling_price
  exact h_selling_price

end customer_paid_l781_781901


namespace part_a_l781_781017

theorem part_a (students : Fin 64 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (A B : Fin 64), (students A).1 ≥ (students B).1 ∧ (students A).2.1 ≥ (students B).2.1 ∧ (students A).2.2 ≥ (students B).2.2 :=
sorry

end part_a_l781_781017


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781547

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781547


namespace total_sales_correct_l781_781857

open Int Real

noncomputable def max_sales : ℤ := 24

noncomputable def seth_sales : ℤ := 3 * max_sales + 6

noncomputable def combined_sales : ℤ := seth_sales + max_sales

noncomputable def emma_sales : ℤ := (2/3 : ℝ) * combined_sales - 4

noncomputable def total_sales : ℝ := (seth_sales : ℝ) + (max_sales : ℝ) + emma_sales

theorem total_sales_correct (h_max_sales : max_sales = 24)
                            (h_seth_sales : seth_sales = 3 * max_sales + 6)
                            (h_combined_sales : combined_sales = seth_sales + max_sales)
                            (h_emma_sales : emma_sales = (2/3 : ℝ) * combined_sales - 4) :
    total_sales = 166 :=
by
  -- Proof required, but replaced with 'sorry' for now
  sorry

end total_sales_correct_l781_781857


namespace remaining_student_number_l781_781279

theorem remaining_student_number (s1 s2 s3 : ℕ) (h1 : s1 = 5) (h2 : s2 = 29) (h3 : s3 = 41) (N : ℕ) (hN : N = 48) :
  ∃ s4, s4 < N ∧ s4 ≠ s1 ∧ s4 ≠ s2 ∧ s4 ≠ s3 ∧ (s4 = 17) :=
by
  sorry

end remaining_student_number_l781_781279


namespace taeyeon_height_proof_l781_781875

noncomputable def seonghee_height : ℝ := 134.5
noncomputable def taeyeon_height : ℝ := seonghee_height * 1.06

theorem taeyeon_height_proof : taeyeon_height = 142.57 := 
by
  sorry

end taeyeon_height_proof_l781_781875


namespace balls_in_boxes_l781_781257

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781257


namespace number_of_ways_to_put_balls_in_boxes_l781_781243

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781243


namespace balls_in_boxes_l781_781221

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781221


namespace part_a_part_b_l781_781808

variable (a : ℝ) (f : ℝ → ℝ)
  (h_a : 1 < a)
  (h_continuous : ContinuousOn f (set.Ici 0))
  (h_limit : ∃ L ∈ ℝ, tendsto (λ x, x * f x) at_top (𝓝 L))

theorem part_a :
  ∫ x in 1..∞, (f x/x) = tendsto (λ t, t * ∫ x in (1 : ℝ)..(a), f (x^t)) at_top (𝓝 (∫ x in 1..∞, f x / x)) :=
sorry

theorem part_b :
  tendsto (λ t, t * ∫ x in 1..a, (1 / (1 + x^t))) at_top (𝓝 (Real.log 2)) :=
sorry

end part_a_part_b_l781_781808


namespace exact_two_solutions_l781_781650

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l781_781650


namespace left_handed_rock_music_lovers_count_l781_781007

def num_left_handed_rock_music_lovers (total_members left_handed rock_music right_handed_dislike_rock : ℕ) : ℕ :=
  total_members - (total_members - rock_music) - right_handed_dislike_rock

theorem left_handed_rock_music_lovers_count :
  num_left_handed_rock_music_lovers 25 10 18 4 = 7 :=
by
  let x := 7
  have h1 : 10 - x = 3,
  have h2 : 18 - x = 11,
  have h3 : 4 = 4,
  have h4 : x + (10 - x) + (18 - x) + 4 = 25,
  simp [h4],
  exact rfl

end left_handed_rock_music_lovers_count_l781_781007


namespace balls_in_boxes_l781_781218

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l781_781218


namespace div_by_64_l781_781852

theorem div_by_64 (n : ℕ) (h : n ≥ 1) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end div_by_64_l781_781852


namespace forest_trees_properties_l781_781513

section ForestTrees

variables (x y : Fin 10 → ℝ)
variables (X_total : ℝ)
variables given_sums : 
  (∑ i, x i = 0.6) ∧ 
  (∑ i, y i = 3.9) ∧ 
  (∑ i, (x i)^2 = 0.038) ∧ 
  (∑ i, (y i)^2 = 1.6158) ∧ 
  (∑ i, (x i) * (y i) = 0.2474)
variables X_total_val : X_total = 186

def average (values : Fin 10 → ℝ) : ℝ :=
  (∑ i, values i) / 10

def correlation_coefficient (x y : Fin 10 → ℝ) : ℝ :=
  (∑ i, (x i - average x) * (y i - average y)) / 
  (Real.sqrt ((∑ i, (x i - average x)^2) * (∑ i, (y i - average y)^2)))

def total_volume_estimate (x_total avg_x avg_y : ℝ) : ℝ :=
  (avg_y / avg_x) * x_total

theorem forest_trees_properties :
  average x = 0.06 ∧ 
  average y = 0.39 ∧ 
  correlation_coefficient x y = 0.97 ∧ 
  total_volume_estimate X_total (average x) (average y) = 1209 :=
by 
  have avg_x : average x = 0.06 := sorry -- Proof not provided
  have avg_y : average y = 0.39 := sorry -- Proof not provided
  have corr_coeff : correlation_coefficient x y = 0.97 := sorry -- Proof not provided
  have total_vol : total_volume_estimate X_total (average x) (average y) = 1209 := sorry -- Proof not provided
  exact ⟨avg_x, avg_y, corr_coeff, total_vol⟩ 

end ForestTrees

end forest_trees_properties_l781_781513


namespace gcd_256_180_720_l781_781421

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end gcd_256_180_720_l781_781421


namespace statements_correctness_l781_781592

theorem statements_correctness (a b x y : ℝ) (b > 0) (b ≠ 1) : 
  (sin (a + b) = sin a + sin b) = false ∧
  (real.exp (a + b) = real.exp a + real.exp b) = false ∧
  (cos (a - b) = cos a * cos b + sin a * sin b) = true ∧
  (tan (a - b) = (tan a - tan b) / (1 + tan a * tan b)) = true ∧
  (real.log (b * x * y) = real.log (b * x) + real.log (b * y)) = true := sorry

end statements_correctness_l781_781592


namespace problem_statement_l781_781755

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l781_781755


namespace sum_of_coefficients_is_225_l781_781790

theorem sum_of_coefficients_is_225 :
  let C4 := 1
  let C41 := 4
  let C42 := 6
  let C43 := 4
  (C4 + C41 + C42 + C43)^2 = 225 :=
by
  sorry

end sum_of_coefficients_is_225_l781_781790


namespace wise_men_game_optimal_difference_l781_781915

theorem wise_men_game_optimal_difference :
  let numbers := Finset.range 1025
  ∃ (a b : ℕ), a ∈ numbers ∧ b ∈ numbers ∧
    ∀ S : Finset ℕ, S.card = 2 → 
    (∃ (strategy1 strategy2 : ℕ → Finset ℕ → Finset ℕ), (strategy1 512 (numbers \ strategy2 256 (numbers \ strategy1 128 (numbers \ strategy2 64 (numbers \ strategy1 32 (numbers \ strategy2 16 (numbers \ strategy1 8 (numbers \ strategy2 4 (numbers \ strategy1 2 (numbers \ strategy2 1 numbers)))))))))) = S /\ 
    ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ (x - y = 32 ∨ y - x = 32)) :=
begin 
  sorry
end

end wise_men_game_optimal_difference_l781_781915


namespace cone_to_cylinder_volume_ratio_l781_781465

-- Define the height and radius of the cylinder
def cylinder_height : ℝ := 15
def cylinder_radius : ℝ := 5

-- Define the height and radius of the cone
def cone_height : ℝ := cylinder_height / 3
def cone_radius : ℝ := cylinder_radius

-- Define volumes of the cylinder and the cone
def volume_cylinder : ℝ := Real.pi * cylinder_radius^2 * cylinder_height
def volume_cone : ℝ := 1 / 3 * Real.pi * cone_radius^2 * cone_height

-- Define the expected ratio
def expected_ratio : ℝ := 1 / 9

-- The statement to prove
theorem cone_to_cylinder_volume_ratio :
  (volume_cone / volume_cylinder) = expected_ratio :=
by
  sorry

end cone_to_cylinder_volume_ratio_l781_781465


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781534

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781534


namespace find_k_vector_perpendicular_l781_781208

theorem find_k_vector_perpendicular :
  let a := (3, 4 : ℝ)
  let b := (-1, 5 : ℝ)
  let c := (2, -3 : ℝ)
  ∃ k : ℝ, (k * a.1 + 2 * b.1) * c.1 + (k * a.2 + 2 * b.2) * c.2 = 0 → k = -17 / 3 :=
begin
  sorry
end

end find_k_vector_perpendicular_l781_781208


namespace parallel_if_interior_alternate_angles_equal_l781_781395

theorem parallel_if_interior_alternate_angles_equal
  (L₁ L₂ : set (set ℝ)) 
  (T : set (set ℝ)) 
  (interior_alternate_angles : ∀ (α β : ℝ), (α ∈ T → β ∈ T →(α, β) ∈ (((L₁ ∪ L₂)ᶜ) ∩ T)))
  (equal_angles : ∀ (α β : ℝ), (α ∈ T → β ∈ T → α = β)) 
  : is_parallel L₁ L₂ :=
by
  sorry

end parallel_if_interior_alternate_angles_equal_l781_781395


namespace simplify_polynomial_l781_781859

def poly1 (x : ℝ) : ℝ := 5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7
def poly2 (x : ℝ) : ℝ := 7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3
def expected (x : ℝ) : ℝ := 12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3

theorem simplify_polynomial (x : ℝ) : poly1 x + poly2 x = expected x :=
  by sorry

end simplify_polynomial_l781_781859


namespace total_amount_is_200_l781_781570

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l781_781570


namespace log_continuous_l781_781367

variable (b : ℝ) (h₁ : b > 0) (h₂ : b ≠ 1)

theorem log_continuous (a : ℝ) (h₃ : a > 0) :
  tendsto (λ x : ℝ, log x / log b) (𝓝 a) (𝓝 (log a / log b)) :=
sorry

end log_continuous_l781_781367


namespace probability_perfect_square_l781_781374

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def successful_outcomes : Finset ℕ := {1, 4}

def total_possible_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_perfect_square :
  (successful_outcomes.card : ℚ) / (total_possible_outcomes.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_perfect_square_l781_781374


namespace quadratic_vertex_form_a_value_l781_781892

theorem quadratic_vertex_form_a_value : 
  (∀ (a : ℝ), ( ∀ (y x : ℝ), y = a * x^2 + x - 3 → y = a * (x + 2)^2 - 3) → a = 1/3) 
:= 
by 
  have h : a = 1/3 := sorry
  exact h

end quadratic_vertex_form_a_value_l781_781892


namespace functions_unique_l781_781643

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem functions_unique (f g: ℝ → ℝ) :
  (∀ x : ℝ, x < 0 → (f (g x) = x / (x * f x - 2)) ∧ (g (f x) = x / (x * g x - 2))) →
  (∀ x : ℝ, 0 < x → (f x = 3 / x ∧ g x = 3 / x)) :=
by
  sorry

end functions_unique_l781_781643


namespace calculate_expression_l781_781123

noncomputable def f (x : ℝ) : ℝ :=
  (x^3 + 5 * x^2 + 6 * x) / (x^3 - x^2 - 2 * x)

def num_holes (f : ℝ → ℝ) : ℕ := 1 -- hole at x = -2
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2 -- vertical asymptotes at x = 0 and x = 1
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 0 -- no horizontal asymptote
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 1 -- oblique asymptote at y = x + 4

theorem calculate_expression : num_holes f + 2 * num_vertical_asymptotes f + 3 * num_horizontal_asymptotes f + 4 * num_oblique_asymptotes f = 9 :=
by
  -- Provide the proof here
  sorry

end calculate_expression_l781_781123


namespace nearly_tricky_7_tiny_count_l781_781478

-- Define a tricky polynomial
def is_tricky (P : Polynomial ℤ) : Prop :=
  Polynomial.eval 4 P = 0

-- Define a k-tiny polynomial
def is_k_tiny (k : ℤ) (P : Polynomial ℤ) : Prop :=
  P.degree ≤ 7 ∧ ∀ i, abs (Polynomial.coeff P i) ≤ k

-- Define a 1-tiny polynomial
def is_1_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 1 P

-- Define a nearly tricky polynomial as the sum of a tricky polynomial and a 1-tiny polynomial
def is_nearly_tricky (P : Polynomial ℤ) : Prop :=
  ∃ Q T : Polynomial ℤ, is_tricky Q ∧ is_1_tiny T ∧ P = Q + T

-- Define a 7-tiny polynomial
def is_7_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 7 P

-- Count the number of nearly tricky 7-tiny polynomials
def count_nearly_tricky_7_tiny : ℕ :=
  -- Simplification: hypothetical function counting the number of polynomials
  sorry

-- The main theorem statement
theorem nearly_tricky_7_tiny_count :
  count_nearly_tricky_7_tiny = 64912347 :=
sorry

end nearly_tricky_7_tiny_count_l781_781478


namespace find_varphi_l781_781196

noncomputable def distance_between_adj_symmetric_axes: ℝ := π / 6
noncomputable def symmetry_point : ℝ × ℝ := (5 * π / 18, 0)
noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * sin (ω * x + ϕ)

theorem find_varphi 
  (ω > 0) 
  (ϕ > -π/2) 
  (ϕ < π/2) 
  (period: distance_between_adj_symmetric_axes * 2 = 2 * π / ω) 
  (symmetry: f symmetry_point.fst ω ϕ = 0) : ϕ = π / 3 := 
sorry

end find_varphi_l781_781196


namespace find_x_square_l781_781150

theorem find_x_square (x : ℝ) (h_pos : x > 0) (h_condition : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end find_x_square_l781_781150


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781544

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781544


namespace factory_growth_rate_equation_l781_781461

theorem factory_growth_rate_equation (x : ℝ) :
  let july_prod := 50 in
  let august_prod := 50 * (1 + x) in
  let september_prod := 50 * (1 + x)^2 in
  let total_prod := 196 in
  (july_prod + august_prod + september_prod = total_prod) :=
by
  sorry

end factory_growth_rate_equation_l781_781461


namespace order_of_6_proof_l781_781870

noncomputable def f (x : ℕ) := x^2 % 13

def order_of_6 : ℕ := 36

theorem order_of_6_proof : (∃ n, n > 0 ∧ f^[n] 6 = 6 ∧ (∀ m < n, m > 0 → f^[m] 6 ≠ 6)) ∧ (order_of_6 = 36) :=
begin
  use 36,
  split,
  { split,
    { norm_num, },
    { split,
      { norm_num, },
      { intros m hm1 hm2,
        sorry, } } },
  norm_num,
end

end order_of_6_proof_l781_781870


namespace integer_solutions_l781_781127

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n, 1 < n → n < p → p % n ≠ 0

def abs_val (n : ℤ) : ℕ := if n < 0 then -n else n.toNat

def satisfies_conditions (n : ℤ) : Prop :=
  is_prime (abs_val (n^3 - 4 * n^2 + 3 * n - 35)) ∧ is_prime (abs_val (n^2 + 4 * n + 8))

theorem integer_solutions :
  {n : ℤ | satisfies_conditions n} = {-3, -1, 5} :=
sorry

end integer_solutions_l781_781127


namespace estimated_total_volume_correct_l781_781527

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781527


namespace find_translation_vector_l781_781015

noncomputable def translation_vector (f g : ℝ → ℝ) (a : ℝ × ℝ) : Prop :=
∀ x : ℝ, g x = f (x + a.1) + a.2

theorem find_translation_vector :
  translation_vector (λ x, 4 * x + 3) (λ x, 4 * x + 16) (0, 13) :=
by
  intros x
  simp
  sorry

end find_translation_vector_l781_781015


namespace probability_of_negative_l781_781972

def set_of_numbers : Set ℤ := {-2, 1, 4, -3, 0}
def negative_numbers : Set ℤ := {-2, -3}
def total_numbers : ℕ := 5
def total_negative_numbers : ℕ := 2

theorem probability_of_negative :
  (total_negative_numbers : ℚ) / (total_numbers : ℚ) = 2 / 5 := 
by 
  sorry

end probability_of_negative_l781_781972


namespace car_value_decrease_l781_781806

theorem car_value_decrease (original_price : ℝ) (decrease_percent : ℝ) (current_value : ℝ) :
  original_price = 4000 → decrease_percent = 0.30 → current_value = original_price * (1 - decrease_percent) → current_value = 2800 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end car_value_decrease_l781_781806


namespace range_of_a_is_correct_l781_781686

noncomputable def nth_degree_zero_point_function (n : ℕ) (f g : ℝ → ℝ) : Prop :=
  ∃ α β, f α = 0 ∧ g β = 0 ∧ |α - β| < n

def f (x : ℝ) : ℝ := 3^(2 - x) - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * exp x

theorem range_of_a_is_correct :
  nth_degree_zero_point_function 1 f (g a) →
  a ∈ set.Ioc (1 / exp 1) (4 / exp 2) :=
sorry

end range_of_a_is_correct_l781_781686


namespace rachel_removed_bottle_caps_l781_781854

def original_bottle_caps : ℕ := 87
def remaining_bottle_caps : ℕ := 40

theorem rachel_removed_bottle_caps :
  original_bottle_caps - remaining_bottle_caps = 47 := by
  sorry

end rachel_removed_bottle_caps_l781_781854


namespace max_value_of_quadratic_l781_781766

theorem max_value_of_quadratic :
  ∀ x : ℝ, let y := -x^2 + 2 * x + 8 in y ≤ 9 :=
by
  sorry

end max_value_of_quadratic_l781_781766


namespace sin_cos_pi_over_12_l781_781640

theorem sin_cos_pi_over_12 :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
sorry

end sin_cos_pi_over_12_l781_781640


namespace round_robin_chess_tournament_l781_781370

open Nat

noncomputable theory
def teams := {A, B, C, D, E, F}

def first_day_matches := {pair B D}
def second_day_matches := {pair C E}
def third_day_matches := {pair D F}
def fourth_day_matches := {pair B C}

theorem round_robin_chess_tournament (teams : Finset String) (first_day second_day third_day fourth_day : Finset (String × String)) :
  third_day = insert (A, C) (third_day_spec) ∧
  fifth_day = insert (A, B) (fifth_day_spec) :=
by
  let third_day_spec := {pair B E, pair A C}
  let fifth_day_spec := {pair D C, pair E F}
  sorry

end round_robin_chess_tournament_l781_781370


namespace initial_position_of_Xiaoming_l781_781435

theorem initial_position_of_Xiaoming :
  ∃ x : ℕ, x - 1 = 6 ∧ x = 7 :=
by
  existsi 7
  split
  { simp }
  { simp }
  done

end initial_position_of_Xiaoming_l781_781435


namespace quadratic_properties_l781_781186

theorem quadratic_properties
  (a b c : ℝ)
  (h1: a ≠ 0)
  (h2: f x = a * x^2 + b * x + c)
  (y_neg2 : f (-2) = -11)
  (y_neg1 : f (-1) = 9)
  (y_0 : f (0) = 21)
  (y_3 : f (3) = 9) :
  (∀ x : ℝ, f x < f 0) ∧
  (∀ x : ℝ, (f x = 0 → 3 < x ∧ x < 4)) ∧
  (∀ x : ℝ, (f x > 21 ↔ 0 < x ∧ x < 2)) :=
by {
  sorry
}

end quadratic_properties_l781_781186


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781510

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781510


namespace solve_for_n_l781_781427

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by
  sorry

end solve_for_n_l781_781427


namespace inequality_for_a_l781_781810

noncomputable def f (x : ℝ) : ℝ :=
  2^x + (Real.log x) / (Real.log 2)

theorem inequality_for_a (n : ℕ) (a : ℝ) (h₁ : 2 < n) (h₂ : 0 < a) (h₃ : 2^a + Real.log a / Real.log 2 = n^2) :
  2 * Real.log n / Real.log 2 > a ∧ a > 2 * Real.log n / Real.log 2 - 1 / n :=
by
  sorry

end inequality_for_a_l781_781810


namespace yuvraj_cuts_pieces_l781_781362

theorem yuvraj_cuts_pieces 
    (Rohan_cuts : Finset ℚ := {k / 9 | k : ℕ, k < 9}.to_finset)
    (Jai_cuts : Finset ℚ := {m / 8 | m : ℕ, m < 8}.to_finset)
    (total_cuts : Finset ℚ := Rohan_cuts ∪ Jai_cuts) :
    total_cuts.card + 1 = 16 := 
sorry

end yuvraj_cuts_pieces_l781_781362


namespace probability_three_digit_l781_781960

theorem probability_three_digit (S : set ℕ) (hS : S = {n | 50 ≤ n ∧ n ≤ 1050}) :
  (∃ A : set ℕ, A = {n | 100 ≤ n ∧ n ≤ 999} ∧ (A ⊆ S) ∧ (finset.card A = 900) ∧ (finset.card S = 1001)) →
  (probability (λ n, n ∈ A) S = 900 / 1001) :=
begin
  sorry
end

end probability_three_digit_l781_781960


namespace separation_of_N_l781_781364

noncomputable def A : set ℕ := {n | ∃ (m ∈ A), ∃ (h : ℕ), n = 2^h + 2}

noncomputable def B : set ℕ := {n | ∃ (m ∈ B), ∃ (h : ℕ), n = 2^h + 2}

theorem separation_of_N (h : 1987 ∈ B ∧ 1988 ∈ A ∧ 1989 ∈ B) : 
  ∃ (A B : set ℕ), 
    (∀ n ∈ A, n ≠ 1987) ∧ 
    ∀ (n m ∈ A), n ≠ m → 
      n + m ≠ 2^(nat.log2 (n + m)) + 2 ∧ 
    (∀ n ∈ B, n ≠ 1987) :=
sorry

end separation_of_N_l781_781364


namespace surface_area_of_sphere_l781_781716

noncomputable def sphere_surface_area (a h v : ℝ) (h₀ : v = (1/3) * a^2 * h) (h₁ : h = 3) (h₂ : v = 6) : ℝ :=
4 * real.pi * (real.sqrt (3 / real.pi))^2

theorem surface_area_of_sphere :
  ∀ (a h v : ℝ), v = (1/3) * a^2 * h → h = 3 → v = 6 → sphere_surface_area a h v = 16 * real.pi :=
by
  intros a h v h₀ h₁ h₂
  rw [sphere_surface_area, h₀, h₁, h₂]
  sorry

end surface_area_of_sphere_l781_781716


namespace thermometer_reading_l781_781885

theorem thermometer_reading (reading : ℝ)
  (h1 : 89.5 ≤ reading)
  (h2 : reading ≤ 90.0)
  (closest_to : reading ≈ 89.8) : reading = 89.8 := 
by
  sorry

end thermometer_reading_l781_781885


namespace number_of_leap_years_ending_in_double_zeroes_l781_781475

def leap_years_between_2050_and_5050 (y : ℕ) : Prop :=
  (y % 100 = 0) ∧ (y % 1200 = 300 ∨ y % 1200 = 600) ∧ (2050 < y) ∧ (y < 5050)

theorem number_of_leap_years_ending_in_double_zeroes :
  ∃ (n : ℕ), n = 4 ∧ {y : ℕ | leap_years_between_2050_and_5050 y}.to_finset.card = n := 
sorry

end number_of_leap_years_ending_in_double_zeroes_l781_781475


namespace neg_irrational_less_than_neg3_l781_781036

theorem neg_irrational_less_than_neg3 : 
  let x : ℝ := -3 * Real.sqrt 2
  in x < -3 ∧ Irrational x := 
by
  sorry

end neg_irrational_less_than_neg3_l781_781036


namespace rationalize_denominator_correct_minimum_A_B_C_D_value_l781_781359

noncomputable def rationalized_expression : Rat :=
  (25.sqrt * 2.sqrt + 5.sqrt * 10.sqrt) / 20

theorem rationalize_denominator_correct :
  5.sqrt * 2.sqrt / 4 + 1.sqrt / 4 = rationalized_expression :=
by sorry

theorem minimum_A_B_C_D_value :
  let A := 5
  let B := 2
  let C := 1
  let D := 4
  A + B + C + D = 12 :=
by sorry

end rationalize_denominator_correct_minimum_A_B_C_D_value_l781_781359


namespace balls_in_boxes_l781_781235

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781235


namespace machines_used_l781_781438

variable (R S : ℕ)

/-- 
  A company has two types of machines, type R and type S. 
  Operating at a constant rate, a machine of type R does a certain job in 36 hours, 
  and a machine of type S does the job in 9 hours. 
  If the company used the same number of each type of machine to do the job in 12 hours, 
  then the company used 15 machines of type R.
-/
theorem machines_used (hR : ∀ ⦃n⦄, n * (1 / 36) + n * (1 / 9) = (1 / 12)) :
  R = 15 := 
by 
  sorry

end machines_used_l781_781438


namespace find_y_given_conditions_l781_781904

theorem find_y_given_conditions (k : ℝ) (h1 : ∀ (x y : ℝ), xy = k) (h2 : ∀ (x y : ℝ), x + y = 30) (h3 : ∀ (x y : ℝ), x - y = 10) :
    ∀ x y, x = 8 → y = 25 :=
by
  sorry

end find_y_given_conditions_l781_781904


namespace power_equivalence_l781_781266

theorem power_equivalence (x : ℝ) (h : 2^(3 * x) = 128) : 2^(3 * x - 2) = 32 := 
by 
  sorry

end power_equivalence_l781_781266


namespace floor_sqrt_50_l781_781614

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781614


namespace total_earnings_l781_781579

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end total_earnings_l781_781579


namespace safe_flight_probability_l781_781059

-- Lean 4 statement to define the problem
theorem safe_flight_probability :
  (let cube_edge_length := 5
       safe_flight_edge_length := 4
       volume_original_cube := cube_edge_length^3
       volume_safe_flight_cube := safe_flight_edge_length^3
       probability_safe_flight := (volume_safe_flight_cube : ℚ) / volume_original_cube
   in probability_safe_flight = 64 / 125) := 
by
  -- Proof step is skipped
  sorry

end safe_flight_probability_l781_781059


namespace arithmetic_seq_a5_a6_l781_781702

theorem arithmetic_seq_a5_a6 :
  (∃ a : ℕ → ℝ, let S := (λ n, ∑ i in Finset.range (n + 1), a i) in
    S 10 = ∫ x in 0..1, (sqrt (1 - x^2) + 2 * x - (π / 4))) →
  ∃ a_5 a_6, a_5 + a_6 = 1 / 5 :=
begin
  sorry
end

end arithmetic_seq_a5_a6_l781_781702


namespace bottles_drunk_l781_781841

theorem bottles_drunk (initial_bottles remaining_bottles : ℕ)
  (h₀ : initial_bottles = 17) (h₁ : remaining_bottles = 14) :
  initial_bottles - remaining_bottles = 3 :=
sorry

end bottles_drunk_l781_781841


namespace closest_point_on_line_l781_781671

theorem closest_point_on_line (x y : ℝ) (h : y = (x + 3) / 2) : 
  ∃ (p : ℝ × ℝ), p = (7, 5) ∧ (λ y, y = (x + 3) / 2) p :=
sorry

end closest_point_on_line_l781_781671


namespace cost_function_correct_l781_781106
noncomputable def quadratic_cost_function : ℝ → ℝ :=
  λ x, (1 / 10) * (x - 15) ^ 2 + 17.5

theorem cost_function_correct :
  (∀ x, 10 ≤ x ∧ x ≤ 25 → quadratic_cost_function 10 = 20 ∧ quadratic_cost_function 15 = 17.5) ∧
  (let profit := λ x, 1.6 * x - quadratic_cost_function x in 
   ∀ x, 10 ≤ x ∧ x ≤ 25 → 
   profit 23 = 12.9 ∧ 
   (∀ y, 10 ≤ y ∧ y ≤ 25 → profit 23 ≥ profit y)) :=
by
  -- Definition of the quadratic cost function
  dsimp [quadratic_cost_function]
  -- Verification of conditions
  split
  { intros x hx,
    split
    { -- Cost at x = 10
      calc
        quadratic_cost_function 10
        = (1 / 10) * (10 - 15) ^ 2 + 17.5 : by simp [quadratic_cost_function]
        = (1 / 10) * 25 + 17.5 : by norm_num
        = 2.5 + 17.5 : by norm_num
        = 20 : by norm_num
    },
    { -- Cost at x = 15
      calc
        quadratic_cost_function 15
        = (1 / 10) * (15 - 15) ^ 2 + 17.5 : by simp [quadratic_cost_function]
        = 17.5 : by simp
    }
  },
  { -- Maximum profit
    let profit := λ x, 1.6 * x - quadratic_cost_function x,
    intros x hx,
    calc
      profit 23
      = 1.6 * 23 - quadratic_cost_function 23 : by simp [profit]
      = 36.8 - ((1 / 10) * (23 - 15) ^ 2 + 17.5) : by norm_num
      = 36.8 - (6.4 + 17.5) : by norm_num
      = 12.9 : by norm_num,
    intros y hy,
    have : profit y = - (1 / 10) * (y - 23) ^ 2 + 12.9,
    { calc
        profit y
        = 1.6 * y - quadratic_cost_function y : by simp [profit]
        = 1.6 * y - ((1 / 10) * (y - 15) ^ 2 + 17.5) : by simp [quadratic_cost_function]
        = 1.6 * y - (1 / 10) * (y - 15) ^ 2 - 17.5 : by linarith
        = - (1 / 10) * (y - 23) ^ 2 + 12.9 : by ring },
    linarith [sub_nonneg_of_le (mul_self_nonneg (y - 23))],
  }

end cost_function_correct_l781_781106


namespace numblarian_words_count_l781_781846

theorem numblarian_words_count :
  let num_letters := 7 in
  let max_word_length := 4 in
  let count_words_of_length (n : Nat) := num_letters ^ n in
  (count_words_of_length 1) + (count_words_of_length 2) + (count_words_of_length 3) + (count_words_of_length 4) = 2800 :=
sorry

end numblarian_words_count_l781_781846


namespace possible_integer_roots_l781_781480

theorem possible_integer_roots (b₁ b₂ : ℤ) :
  ∃ x ∈ { -13, -1, 1, 13 }, x ∈ ℤ ∧ (x^3 + b₂ * x^2 + b₁ * x - 13 = 0) := sorry

end possible_integer_roots_l781_781480


namespace ball_distribution_l781_781238

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781238


namespace find_last_number_2_l781_781882

theorem find_last_number_2 (A B C D : ℤ) 
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) : 
  D = 2 := 
sorry

end find_last_number_2_l781_781882


namespace line_equation_satisfies_m_plus_b_l781_781072

theorem line_equation_satisfies_m_plus_b :
  ∃ (m b : ℝ), m = 8 ∧ (∃ (x y : ℝ), x = -2 ∧ y = 4 ∧ y = m * x + b) ∧ (m + b = 28) := 
by
  let m := 8
  have pass_through_point : ∃ (b : ℝ), 4 = m * (-2) + b
  { use 20
    simp [m], }
  rcases pass_through_point with ⟨b, hb⟩
  use [m, b]
  constructor
  . rfl
  constructor
  . use [-2, 4]
    simp [hb]
  . simp [hb]
  sorry

end line_equation_satisfies_m_plus_b_l781_781072


namespace order_of_6_with_respect_to_f_is_undefined_l781_781872

noncomputable def f (x : ℕ) : ℕ := x ^ 2 % 13

def order_of_6_undefined : Prop :=
  ∀ m : ℕ, m > 0 → f^[m] 6 ≠ 6

theorem order_of_6_with_respect_to_f_is_undefined : order_of_6_undefined :=
by
  sorry

end order_of_6_with_respect_to_f_is_undefined_l781_781872


namespace part1_part2_l781_781725

--  Definition for the function f(x)
def f (x a : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Part (1) proof statement: range of f(x) with a=1 on [0,2] is [0,9]
theorem part1 (f : ℝ → ℝ) (a : ℝ) (h : a = 1) :
  set.range (λ x, f x 1) = set.Icc 0 9 := sorry

-- Part (2) proof statement: if f(x) has a minimum value of 3 on [0,2], then a = 1 - √2 or a = 5 + √10
theorem part2 (f : ℝ → ℝ) (a : ℝ) (h : ∃ x ∈ set.Icc 0 2, f x a = 3) :
  a = 1 - real.sqrt 2 ∨ a = 5 + real.sqrt 10 := sorry

end part1_part2_l781_781725


namespace ellen_chairs_example_l781_781999

def ellen_bought_chairs (cost_per_chair total_spent : ℕ) : Prop :=
  ∃ (num_chairs : ℕ), total_spent = num_chairs * cost_per_chair

theorem ellen_chairs_example : ellen_bought_chairs 15 180 :=
by
  use 12
  simp
  exact Nat.mul_succ_pred_eq_self 12


end ellen_chairs_example_l781_781999


namespace distance_from_point_to_line_l781_781666

noncomputable def distance_point_to_line : ℝ :=
  let point := (2:ℝ, 3, 1)
  let line_point := (4:ℝ, 8, 5)
  let direction := (2:ℝ, 3, -3)
  let closest_point := line_point + (-7 / 22) • direction in
  real.sqrt ((closest_point.1 - point.1)^2 + (closest_point.2 - point.2)^2 + (closest_point.3 - point.3)^2)

theorem distance_from_point_to_line :
  distance_point_to_line = real.sqrt (20702) / 22 :=
by
  sorry

end distance_from_point_to_line_l781_781666


namespace remainder_98_pow_50_mod_100_l781_781031

/-- 
Theorem: The remainder when \(98^{50}\) is divided by 100 is 24.
-/
theorem remainder_98_pow_50_mod_100 : (98^50 % 100) = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l781_781031


namespace radius_of_sphere_l781_781970

theorem radius_of_sphere (r1 r2 BC HB CH radius : ℝ) 
  (h_r1 : r1 = 24) 
  (h_r2 : r2 = 4) 
  (h_BC : BC = r1 + r2) 
  (h_HB : HB = r1 - r2) 
  (h_CH : CH = real.sqrt (BC^2 - HB^2)) 
  (h_radius : radius = CH / 2) : 
  radius = real.sqrt 96 := 
by
  have h1 : BC = 28 := by 
    rw [h_BC, h_r1, h_r2]
  have h2 : HB = 20 := by 
    rw [h_HB, h_r1, h_r2]
  have h3 : CH = real.sqrt 384 := by 
    rw [h_CH, h1, h2]
    exact real.sqrt 384 
  have h4 : radius = real.sqrt 96 := by 
    rw [h_radius, h3]
    exact real.sqrt 96
  exact h4

end radius_of_sphere_l781_781970


namespace binom_n_n_minus_2_l781_781923

noncomputable def factorial : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def binom (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem binom_n_n_minus_2 (n : ℕ) (h : 0 < n) : 
  binom n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binom_n_n_minus_2_l781_781923


namespace first_year_with_property_l781_781796

def is_valid_permutation (y : ℕ) (p : ℕ) : Prop :=
  ∃ (digits_y : List ℕ) (digits_p : List ℕ), 
    digits_y.perm digits_p ∧ 
    p.digits = digits_p ∧ 
    y.digits = digits_y ∧ 
    (p ≥ y ∨ p < 1000 ∧ p.digits.head ≠ 0)

def satisfies_property (y : ℕ) : Prop :=
  y > 2009 ∧ 
  ∀ p : ℕ, is_valid_permutation y p

theorem first_year_with_property : 
  ∃ y : ℕ, satisfies_property y ∧ y = 2022 :=
by 
  sorry

end first_year_with_property_l781_781796


namespace min_hexagon_area_triangle_l781_781313

-- Define triangle ABC with given sides
structure Triangle :=
  (A B C : ℝ)
  (AB BC CA : ℝ)
  (sides_condition : AB = 26 ∧ BC = 51 ∧ CA = 73)

-- Define the point O inside the triangle and the lines passing through it
def point_and_parallel_lines (T : Triangle) : Prop :=
  let O := (T.A + T.B + T.C) / 3 in  -- O is the centroid for simplicity
  ∃ (l1 l2 l3 : ℝ), l1 = T.AB ∧ l2 = T.BC ∧ l3 = T.CA

-- Define the function to compute the area using Heron's formula
noncomputable def area_triangle (T : Triangle) : ℝ :=
  let s := (T.AB + T.BC + T.CA) / 2 in
  Real.sqrt (s * (s - T.AB) * (s - T.BC) * (s - T.CA))

-- Define the minimum area of the hexagon
noncomputable def min_hexagon_area (T : Triangle) : ℝ :=
  (2 / 3) * area_triangle T

-- The statement to prove the minimum hexagon area
theorem min_hexagon_area_triangle (T : Triangle) (h : point_and_parallel_lines T) :
  min_hexagon_area T = 280 :=
  sorry

end min_hexagon_area_triangle_l781_781313


namespace AD_bisects_angle_EDF_l781_781847

/-!
# Triangle and Angle Bisection

Given triangle ABC with altitude AD and point P on AD. Points E and F
are intersections of BP and CP with CA and AB respectively. Prove that AD
bisects ∠EDF.
-/

variables {A B C D E F P : Type}
variables [point A] [point B] [point C] [point D] [point E] [point F] [point P]

-- Definitions of points on lines and meeting points
variable (h₀ : altitude AD A B C)
variable (h₁ : P ∈ AD)
variable (h₂ : E ∈ line_segment B P)
variable (h₃ : F ∈ line_segment C P)
variable (h₄ : line_segment B P ∩ line_segment C F = {E})
variable (h₅ : line_segment C P ∩ line_segment A B = {F})

theorem AD_bisects_angle_EDF :
  ∃ A B C D E F P, P ∈ AD ∧ line_segment B P ∩ line_segment CA = {E} ∧ line_segment C P ∩ line_segment AB = {F} ∧ bisects AD ∠EDF :=
sorry

end AD_bisects_angle_EDF_l781_781847


namespace compute_value_l781_781325

noncomputable def p_and_q_roots : Type := {p q : ℝ // 3*p^2 - 5*p - 8 = 0 ∧ 3*q^2 - 5*q - 8 = 0}

theorem compute_value (t : p_and_q_roots) : 
  let p := t.1;
      q := t.2;
      sum_pq := (p + q);
      prod_pq := (p * q) in
  3*p^2 - 5*p - 8 = 0 ∧ 3*q^2 - 5*q - 8 = 0 ∧
  sum_pq = 5/3 ∧ prod_pq = -8/3 → 
  (9*p^3 - 9*q^3) / (p - q) = 49 :=
by
  intros p q sum_pq prod_pq;
  intro h;
  use sorry

end compute_value_l781_781325


namespace length_of_real_axis_of_hyperbola_l781_781895

theorem length_of_real_axis_of_hyperbola :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 -> ∃ a : ℝ, 2 * a = 4 :=
by
intro x y h
sorry

end length_of_real_axis_of_hyperbola_l781_781895


namespace car_value_reduction_l781_781805

/-- Jocelyn bought a car 3 years ago at $4000. 
If the car's value has reduced by 30%, calculate the current value of the car. 
Prove that it is equal to $2800. -/
theorem car_value_reduction (initial_value : ℝ) (reduction_percentage : ℝ) (current_value : ℝ) 
  (h_initial : initial_value = 4000)
  (h_reduction : reduction_percentage = 30)
  (h_current : current_value = initial_value - (reduction_percentage / 100) * initial_value) :
  current_value = 2800 :=
by
  -- Formal proof goes here
  sorry

end car_value_reduction_l781_781805


namespace flowchart_output_is_minus_nine_l781_781889

-- Given initial state and conditions
def initialState : ℤ := 0

-- Hypothetical function representing the sequence of operations in the flowchart
-- (hiding the exact operations since they are speculative)
noncomputable def flowchartOperations (S : ℤ) : ℤ := S - 9  -- Assuming this operation represents the described flowchart

-- The proof problem
theorem flowchart_output_is_minus_nine : flowchartOperations initialState = -9 :=
by
  sorry

end flowchart_output_is_minus_nine_l781_781889


namespace sqrt_floor_eq_seven_l781_781630

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781630


namespace percent_singles_l781_781136

theorem percent_singles (total_hits home_runs triples doubles singles : ℕ) 
  (h_hits : total_hits = 50) 
  (h_home_runs : home_runs = 2)
  (h_triples : triples = 3)
  (h_doubles : doubles = 8)
  (h_singles : singles = total_hits - (home_runs + triples + doubles)) :
  (singles * 100) / total_hits = 74 :=
by
  have non_single_hits := home_runs + triples + doubles
  have singles := total_hits - non_single_hits
  calc
    (singles * 100) / total_hits = (37 * 100) / 50 : by sorry
    ... = 74 : by sorry

end percent_singles_l781_781136


namespace repeat_2013_divisible_by_9_l781_781040

theorem repeat_2013_divisible_by_9 :
  (∀ n : ℕ, (let repeated_sum := n * (2 + 0 + 1 + 3) in repeated_sum % 9 = 0) ↔ (∃ k : ℕ, 201320132013 = k * 9)) :=
by sorry

end repeat_2013_divisible_by_9_l781_781040


namespace solve_for_x_l781_781861

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_l781_781861


namespace polynomial_bound_l781_781331

noncomputable def P (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (h : ∀ x : ℝ, |x| < 1 → |P a b c d x| ≤ 1) :
  |a| + |b| + |c| + |d| ≤ 7 :=
sorry

end polynomial_bound_l781_781331


namespace total_lives_l781_781944

theorem total_lives :
  ∀ (num_friends num_new_players lives_per_friend lives_per_new_player : ℕ),
  num_friends = 2 →
  lives_per_friend = 6 →
  num_new_players = 2 →
  lives_per_new_player = 6 →
  (num_friends * lives_per_friend + num_new_players * lives_per_new_player) = 24 :=
by
  intros num_friends num_new_players lives_per_friend lives_per_new_player
  intro h1 h2 h3 h4
  sorry

end total_lives_l781_781944


namespace avg_disk_space_per_minute_l781_781466

noncomputable def avg_mb_per_min (days : ℕ) (total_disk_space_mb : ℕ) : ℕ :=
  let minutes_per_day := 24 * 60
  let total_minutes := days * minutes_per_day
  let avg_mb := total_disk_space_mb / total_minutes
  Nat.round avg_mb

theorem avg_disk_space_per_minute :
  avg_mb_per_min 15 30000 = 1 :=
by
  sorry

end avg_disk_space_per_minute_l781_781466


namespace fewer_free_throws_l781_781883

noncomputable def Deshawn_free_throws : ℕ := 12
noncomputable def Kayla_free_throws : ℕ := Deshawn_free_throws + (Deshawn_free_throws / 2)
noncomputable def Annieka_free_throws : ℕ := 14

theorem fewer_free_throws :
  Annieka_free_throws = Kayla_free_throws - 4 :=
by
  sorry

end fewer_free_throws_l781_781883


namespace modulus_of_Z_l781_781166

open Complex

noncomputable def Z : ℂ := complex.of_real 2 + I

theorem modulus_of_Z : (∃ Z : ℂ, Z * I = 2 + I) → |Z| = sqrt 5 := by
  sorry

end modulus_of_Z_l781_781166


namespace hundredth_number_is_201_l781_781803

-- Mathematical definition of the sequence
def counting_sequence (n : ℕ) : ℕ :=
  3 + (n - 1) * 2

-- Statement to prove
theorem hundredth_number_is_201 : counting_sequence 100 = 201 :=
by
  sorry

end hundredth_number_is_201_l781_781803


namespace circumcircle_tangent_to_omega_l781_781581

open EuclideanGeometry

/-- Given two circles ω and Ω that intersect at points A and B.
    M is the midpoint of the arc AB on ω that lies inside Ω.
    A chord MP of circle ω intersects circle Ω at Q, where Q is inside ω.
    Let ℓ_P be the tangent to ω at P, and ℓ_Q be the tangent to Ω at Q.
    Prove that the circumcircle of the triangle formed by ℓ_P, ℓ_Q, and AB is tangent to Ω. -/
theorem circumcircle_tangent_to_omega
  {ω Ω : Circle}
  {A B M P Q : Point}
  {ℓ_P ℓ_Q : Line}
  (h_intersect : ω ∩ Ω = {A, B})
  (hM : M = midpointof_arc AB ω)
  (h_MP : ∈ M chord P ω)
  (hQ : ∈ Q chord Ω Q)
  (h₂ : tangent ℓ_P ω P)
  (h₃ : tangent ℓ_Q Ω Q) :
  tangent (circumcircle (triangle_of_lines ℓ_P ℓ_Q (line_of_points A B))) Ω := sorry

end circumcircle_tangent_to_omega_l781_781581


namespace transform_negation_l781_781098

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l781_781098


namespace balls_in_boxes_l781_781225

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781225


namespace like_terms_sum_l781_781184

theorem like_terms_sum (n m : ℕ) 
  (h1 : n + 1 = 3) 
  (h2 : m - 1 = 3) : 
  m + n = 6 := 
  sorry

end like_terms_sum_l781_781184


namespace fraction_division_l781_781925

theorem fraction_division :
  (3 / 7) / (2 / 5) = (15 / 14) :=
by
  sorry

end fraction_division_l781_781925


namespace soccer_school_admission_prob_l781_781489

theorem soccer_school_admission_prob :
  let p_ac := 0.5 in  -- Probability of passing an assistant coach interview
  let p_hc := 0.3 in  -- Probability of passing the head coach's final review
  let p_both_ac := p_ac * p_ac in  -- Probability of passing both assistant coach interviews
  let p_one_ac_then_hc := 2 * (p_ac * (1 - p_ac) * p_hc) in  -- Probability of passing one assistant coach and head coach
  p_both_ac + p_one_ac_then_hc = 0.4 :=

by
  simp [p_ac, p_hc, p_both_ac, p_one_ac_then_hc],
  sorry

end soccer_school_admission_prob_l781_781489


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781503

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781503


namespace floor_sqrt_50_l781_781617

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781617


namespace platform_length_l781_781967

-- Definitions from the problem conditions
def l_train : ℝ := 300  -- Length of the train in meters
def v_train : ℝ := 100 * (1000 / 3600)  -- Speed of the train in meters per second (converting from 100 km/h to m/s)
def t : ℝ := 30  -- Time in seconds to cross the platform

-- The distance covered by the train in the given time
def total_distance : ℝ := v_train * t

-- The length of the platform is the total distance covered by the train minus the length of the train
def length_of_platform : ℝ := total_distance - l_train

-- The theorem to be proved
theorem platform_length : length_of_platform = 533.4 := by
  calc 
    length_of_platform = total_distance - l_train     : by rfl
                    ... = v_train * t - l_train       : by rfl
                    ... = (100 * (1000 / 3600)) * 30 - 300 : by rfl
                    ... = 833.333... - 300            : by rfl
                    ... = 533.333...                  : by rfl
                    ... = 533.4                       : by sorry  -- Approximation step

end platform_length_l781_781967


namespace floor_sqrt_50_l781_781602

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781602


namespace beginning_of_winter_function_range_l781_781879

noncomputable def f (x : ℝ) : ℝ := sorry

-- definition: odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- condition: inequality for positive real numbers
def inequality_condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → 
  (f x1) ^ (2023 / 2025) - (f x2) ^ (2023 / 2025) < 0 → 
  x1 ^ (8 / 11) - x2 ^ (8 / 11) < 0

-- specific values
def f_neg4 : ℝ := f (-4)

-- definition: beginning of winter function
noncomputable def g (x : ℝ) : ℝ :=
  (f (x - 2) / (x + 2)) ^ 1925

-- proof statement
theorem beginning_of_winter_function_range (h_odd : odd_function f)
  (h_inequality : inequality_condition f)
  (h_f_neg4 : f_neg4 = 0):
  {x : ℝ | g x ≥ 0} = set.interval 2 6 := 
sorry

end beginning_of_winter_function_range_l781_781879


namespace distance_between_intersections_l781_781555

theorem distance_between_intersections :
  let ellipse_eq : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, x^2 / 16 + y^2 / 4 = 1
  let parabola_eq : (ℝ × ℝ) → Prop := λ ⟨x, y⟩, x = 2 * (5 + 2 * sqrt 3) * y^2 + (5 + 2 * sqrt 3) / 2
  ∀ (y₁ y₂: ℝ), 
  (∃ x₁, ellipse_eq (x₁, y₁) ∧ parabola_eq (x₁, y₁)) ∧
  (∃ x₂, ellipse_eq (x₂, y₂) ∧ parabola_eq (x₂, y₂)) →
  real.dist y₁ y₂ = 2 * |y₁ - y₂| := 
begin
  sorry
end

end distance_between_intersections_l781_781555


namespace hyperbola_eccentricity_asymptotes_l781_781886

theorem hyperbola_eccentricity_asymptotes :
  (let a := 2, 
       b := 2 * Real.sqrt 3, 
       c := Real.sqrt (a ^ 2 + b ^ 2)) 
    in c / a = 2 ∧ (∀ x : ℝ, y = x * (Real.sqrt 3) ∨ y = -x * (Real.sqrt 3)) :=
by 
  let a := 2
  let b := 2 * Real.sqrt 3 
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  have h1 : c / a = 2 := sorry
  have h2 : ∀ x : ℝ, y = x * Real.sqrt 3 ∨ y = -x * Real.sqrt 3 := sorry
  exact ⟨h1, h2⟩

end hyperbola_eccentricity_asymptotes_l781_781886


namespace dice_product_divisible_by_8_l781_781916

noncomputable def probability_divisible_by_8 : ℚ := 15 / 16

theorem dice_product_divisible_by_8 :
  ∀ (dices : Fin 8 → Fin 6), 
    (∃ (p : ℕ), (∀ i, dices i.1.succ ∈ {1, 3, 5}) → ¬ (8 ∣ p) → p = dices.prod (λ j, (j + 1 : ℕ))) → 
      (1 - probability_divisible_by_8 = 1 / 16) :=
by
  sorry

end dice_product_divisible_by_8_l781_781916


namespace balls_in_boxes_l781_781227

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781227


namespace incenter_symmetry_l781_781409

-- Define the Triangle and its properties
structure Triangle :=
  (A B C: Point)
  (angle_BAC angle_ABC angle_ACB: ℝ)

-- Define the Incenter properties structure
structure Incenter (T : Triangle) :=
  (I: Point)
  (angle_BIX angle_CIX : ℝ) -- Angles as described in the solution steps

-- Condition: Three given lines intersecting at one point forming triangle
theorem incenter_symmetry (T : Triangle) 
  (I: Incenter T)
  (X Y: Point)
  (H1: ∠ BIX = 120)
  (H2: ∠ CIX = 120)
  : collinear X Y I :=
by
  sorry

end incenter_symmetry_l781_781409


namespace proof_problem_l781_781894

noncomputable def collinear_centers (A B C M N P I O H : Point) :=
  incircle_triangle_meet ABC M N P →   -- condition: incircle meets sides at M, N, P
  orthocenter M N P H →                -- initial statement of the orthocenter of triangle MNP
  incenter ABC I →                     -- initial statement of the incenter of triangle ABC
  circumcenter ABC O →                 -- initial statement of the circumcenter of triangle ABC
  collinear I O H                      -- statement that I, O, and H are collinear

-- Establish the proof problem
theorem proof_problem (A B C M N P : Point) 
    (h : incircle_triangle_meet ABC M N P) 
    (i : incenter ABC I)
    (j : circumcenter ABC O) 
    (k : orthocenter M N P H) : 
    collinear I O H :=
sorry

end proof_problem_l781_781894


namespace champion_is_d_l781_781293

-- Define the players
inductive Player
| A | B | C | D

open Player

-- Define the predictions of each player
def prediction (p : Player) (champion : Player) : Prop :=
  match p with
  | A => champion ≠ B
  | B => champion = C ∨ champion = D
  | C => champion ≠ A ∧ champion ≠ D
  | D => true  -- D did not make a prediction

-- Define the main statement
theorem champion_is_d (champion : Player) :
  (∃ p, ¬prediction p champion) →
  (¬prediction A champion ∨ ¬prediction B champion ∨ ¬prediction C champion ∨ ¬prediction D champion) →
  (exactly_one (λ p, ¬prediction p champion)) →
  champion = D :=
sorry

end champion_is_d_l781_781293


namespace floor_sqrt_50_l781_781612

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781612


namespace problem_statement_l781_781754

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l781_781754


namespace tangent_lines_l781_781187

noncomputable def curve : ℝ → ℝ := λ x, (1/3 : ℝ) * x^3 + (4/3 : ℝ)

noncomputable def derivative_curve : ℝ → ℝ := λ x, x^2

theorem tangent_lines (x : ℝ) (y : ℝ):
  (y = curve x) ∧ ((x = 2 ∧ y = 4) ∨ (2, 4) ∈ set_of (λ p : ℝ × ℝ, p.2 = curve p.1)) →
  ((4 * x - y - 4 = 0) ∨ (x - y + 2 = 0)) :=
sorry

end tangent_lines_l781_781187


namespace sufficient_but_not_necessary_condition_l781_781327

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → a^2 > 1) ∧ (¬ (a^2 > 1 → a > 1)) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l781_781327


namespace LM_eq_LN_l781_781050

open_locale classical
noncomputable theory

variables {A B C L M N : Point}
variables {b c : Line}
variables {h₀ : IsAngleBisector A L B C}
variables {h₁ : Parallel B c}
variables {h₂ : EquidistantFromVertex A b c}
variables {h₃ : Midpoint (SegmentIntersection L M) B A}
variables {h₄ : Midpoint (SegmentIntersection L N) C A}

-- Theorem statement
theorem LM_eq_LN
  (angle_bisector : is_angle_bisector A L B C)
  (lines_parallel : are_parallel b c)
  (eq_dist_from_vertex : equidistant_from_vertex A b c)
  (midpoint_AB : midpoint (?) (segment_intersection L M) AB)
  (midpoint_AC : midpoint (?) (segment_intersection L N) AC)
  : distance LM = distance LN :=
sorry -- Proof goes here, omitted as per instructions

end LM_eq_LN_l781_781050


namespace cube_volume_l781_781002

theorem cube_volume (A : ℝ) (h : A = 24) : 
  ∃ V : ℝ, V = 8 :=
by
  sorry

end cube_volume_l781_781002


namespace number_of_ways_to_put_balls_in_boxes_l781_781245

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781245


namespace min_distance_between_intersections_l781_781387

theorem min_distance_between_intersections :
  ∃ m : ℝ, let A := (x1 : ℝ) × ℝ, B := (x2 : ℝ) × ℝ in
  A.2 = 2 * A.1 + 3 ∧ B.2 = B.1 + Real.log B.1 ∧ B.2 = m ∧ A.2 = m →
  ∀ x2 > 0, 
    let x1 := 1 / 2 * (x2 + Real.log x2) - 3 / 2 in
    |x1 - x2| ≥ 0 ∧ 
    (∀ x > 0, (1 / 2 * (x - Real.log x) + 3 / 2) ≥ 2) ∧
    | (1 / 2 * (1 - Real.log 1) + 3 / 2) - 1 | = 2 :=
sorry

end min_distance_between_intersections_l781_781387


namespace max_total_balls_l781_781986

theorem max_total_balls
  (r₁ : ℕ := 89)
  (t₁ : ℕ := 90)
  (r₂ : ℕ := 8)
  (t₂ : ℕ := 9)
  (y : ℕ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (h₃ : 92 ≤ (r₁ + r₂ * y) * 100 / (t₁ + t₂ * y))
  : y ≤ 22 → 90 + 9 * y = 288 :=
by sorry

end max_total_balls_l781_781986


namespace balls_in_boxes_l781_781232

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781232


namespace limit_x_to_0_x_pow_x_eq_one_limit_x_to_0_inv_x_pow_sin_x_eq_one_limit_x_to_1_x_pow_1_div_x_minus_1_eq_e_l781_781145

-- Problem statement 1: Prove that lim (x -> 0) x^x = 1
theorem limit_x_to_0_x_pow_x_eq_one : tendsto (λ x : ℝ, x^x) (𝓝 0) (𝓝 1) :=
by sorry

-- Problem statement 2: Prove that lim (x -> 0) (1/x)^sin(x) = 1
theorem limit_x_to_0_inv_x_pow_sin_x_eq_one : tendsto (λ x : ℝ, (1/x) ^ (Real.sin x)) (𝓝 0) (𝓝 1) :=
by sorry

-- Problem statement 3: Prove that lim (x -> 1) x^(1/(x-1)) = e
theorem limit_x_to_1_x_pow_1_div_x_minus_1_eq_e : tendsto (λ x : ℝ, x ^ ((1 : ℝ) / (x - 1))) (𝓝 1) (𝓝 Real.exp 1) :=
by sorry

end limit_x_to_0_x_pow_x_eq_one_limit_x_to_0_inv_x_pow_sin_x_eq_one_limit_x_to_1_x_pow_1_div_x_minus_1_eq_e_l781_781145


namespace circumcircle_fixed_point_exists_l781_781417

-- Define the points and conditions
variables (A B P : Type) [uniform_space A] [uniform_space B] [metric_space P]
variables (line1 line2 : set P) (t : ℝ)

-- Assume points A and B move uniformly along their respective lines
variable (moves_uniformly : ∀ t : ℝ, A ∈ line1 ∧ B ∈ line2)

-- Assume A and B do not pass through P simultaneously
variable (not_simultaneous : ¬ ∃ t : ℝ, A = P ∧ B = P)

-- Required proof: There exists a fixed point O (distinct from P) that lies on the circumcircle of ΔAPB for any t
theorem circumcircle_fixed_point_exists (A B P : Type) 
  [uniform_space A] [uniform_space B] [metric_space P] 
  (line1 line2 : set P) 
  (moves_uniformly : ∀ t : ℝ, A ∈ line1 ∧ B ∈ line2)
  (not_simultaneous : ¬ ∃ t : ℝ, A = P ∧ B = P) :
  ∃ (O : P), (O ≠ P) ∧ (∀ t, ∃ circ : P, circ ∈ (circle_through_points A B P)) :=
sorry

end circumcircle_fixed_point_exists_l781_781417


namespace floor_sqrt_fifty_l781_781621

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781621


namespace sqrt_floor_eq_seven_l781_781627

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781627


namespace necklace_possible_l781_781419

-- Define the data types and the hypothesis
variables (R B Y G : ℕ) -- Number of Red, Blue, Yellow, and Green beads
variable (H : R + B + Y + G = 100) -- Total number of beads
variable (H_R : R ≤ 50) -- No more than 50 red beads
variable (H_B : B ≤ 50) -- No more than 50 blue beads
variable (H_Y : Y ≤ 50) -- No more than 50 yellow beads
variable (H_G : G ≤ 50) -- No more than 50 green beads

-- Define the proposition to be proved
theorem necklace_possible (R B Y G : ℕ) (H : R + B + Y + G = 100)
  (H_R : R ≤ 50) (H_B : B ≤ 50) (H_Y : Y ≤ 50) (H_G : G ≤ 50) :
  ∃ (arrangement : list ℕ), (∀ (i : ℕ), i < 99 → arrangement.nth i ≠ arrangement.nth (i + 1)) :=
sorry

end necklace_possible_l781_781419


namespace intersection_M_N_l781_781731

open Set Real

def M : Set ℝ := { x : ℝ | x ^ 2 - 3 * x - 4 ≤ 0 }
def N : Set ℝ := { x : ℝ | ln x ≥ 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 1 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end intersection_M_N_l781_781731


namespace find_digit_l781_781678

theorem find_digit {x y z : ℕ} (h : 100 * x + 10 * y + z - (x + y + z) = 261) : y = 7 :=
by
  have := h
  calc
    100 * x + 10 * y + z - (x + y + z) = 99 * x + 9 * y               : by ring
                                ...  = 261                           : by assumption
                                ...  ÷ 9  = 29          : by linarith
                                ...  = 11 * x + y     : by assumption
  sorry

end find_digit_l781_781678


namespace cos_five_pi_over_three_constant_term_binomial_flower_arrangement_adj_bc_even_function_real_number_a_l781_781056

-- Problem 1:
theorem cos_five_pi_over_three : Real.cos (5 * Real.pi / 3) = 1 / 2 := 
by
sor..

-- Problem 2:
theorem constant_term_binomial : (Mathlib.Real.Coeff.mul_left (Mathlib.Real.of_rat.coeff (some_expr involving C6r and 2 ∘ r)) where some_expr... =
60 := 
by 
sor.

-- Problem 3:
theorem flower_arrangement_adj_bc : (let arrangements := {set of all valid arrangements with b and c}; arrangements.card = 12 :=
by 
sor.

-- Problem 4:
theorem even_function_real_number_a :
(∀ x : Real, x = Real.neg x → (lg (10 ^ x + 1) + a * x) = (lg (10 ^ Real.abs x + 1) + a * Real.neg x)) →a = -(1/2) :=
by
sor.

end cos_five_pi_over_three_constant_term_binomial_flower_arrangement_adj_bc_even_function_real_number_a_l781_781056


namespace total_time_correct_l781_781298

-- Definitions according to the conditions
def woody_writing_time : ℝ := 18                -- Woody's writing time in months
def ivanka_writing_time : ℝ := woody_writing_time + 3  -- Ivanka's writing time in months
def alice_writing_time : ℝ := woody_writing_time / 2   -- Alice's writing time in months
def tom_writing_time : ℝ := alice_writing_time * 2     -- Tom's writing time in months

-- Editing times as 25% of writing times
def ivanka_editing_time : ℝ := 0.25 * ivanka_writing_time
def woody_editing_time : ℝ := 0.25 * woody_writing_time
def alice_editing_time : ℝ := 0.25 * alice_writing_time
def tom_editing_time : ℝ := 0.25 * tom_writing_time

-- Revising times as 15% of writing times
def ivanka_revising_time : ℝ := 0.15 * ivanka_writing_time
def woody_revising_time : ℝ := 0.15 * woody_writing_time
def alice_revising_time : ℝ := 0.15 * alice_writing_time
def tom_revising_time : ℝ := 0.15 * tom_writing_time

-- Total times combining writing, editing, and revising times
def ivanka_total_time : ℝ := ivanka_writing_time + ivanka_editing_time + ivanka_revising_time
def woody_total_time : ℝ := woody_writing_time + woody_editing_time + woody_revising_time
def alice_total_time : ℝ := alice_writing_time + alice_editing_time + alice_revising_time
def tom_total_time : ℝ := tom_writing_time + tom_editing_time + tom_revising_time

-- Total sum of all times
def total_time : ℝ := ivanka_total_time + woody_total_time + alice_total_time + tom_total_time

-- The theorem to prove
theorem total_time_correct : total_time = 92.4 :=
by
  sorry

end total_time_correct_l781_781298


namespace germination_relative_frequency_l781_781928

theorem germination_relative_frequency {n m : ℕ} (h₁ : n = 1000) (h₂ : m = 1000 - 90) : 
  (m : ℝ) / (n : ℝ) = 0.91 := by
  sorry

end germination_relative_frequency_l781_781928


namespace seating_arrangements_l781_781006

theorem seating_arrangements :
  ∃ (n : ℕ), let num_seats := 22 
             let num_candidates := 4
             let min_empty_seats := 5
  in  n = 840 := 
by
  sorry

end seating_arrangements_l781_781006


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781501

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781501


namespace quadrilateral_is_square_l781_781595

noncomputable def is_square (ABCD : quadrilateral) : Prop := sorry

noncomputable def extends_to_circle (ABCD : quadrilateral) (points : list point) : Prop := sorry

theorem quadrilateral_is_square {ABCD : quadrilateral} (h1 : convex ABCD) 
(h2 : ∀ (sides : list segment), extends_sides ABCD sides) 
(h3 : ∃ points : list point, extends_to_circle ABCD points) :
is_square ABCD :=
sorry

end quadrilateral_is_square_l781_781595


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781550

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781550


namespace Jake_later_than_Austin_l781_781112

theorem Jake_later_than_Austin 
    (floors : ℕ) 
    (steps_per_floor : ℕ) 
    (elevator_time_sec : ℕ)
    (steps_per_sec : ℕ) 
    (jh : floors = 9) 
    (spf : steps_per_floor = 30) 
    (et : elevator_time_sec = 60) 
    (steps_sec : steps_per_sec = 3) 
    : 90 - 60 = 30 := 
by
  have total_steps := 9 * 30
  have time_jake := total_steps / 3
  have Jake_additional_time := time_jake - 60
  rw [eq_one_of_eq_succ_eq_succ jh, eq_one_of_eq_succ_eq_succ spf, eq_one_of_eq_succ_eq_succ et, eq_one_of_eq_succ_eq_succ steps_sec] at *
  norm_num at *
  exact Jake_additional_time

end Jake_later_than_Austin_l781_781112


namespace floor_sqrt_fifty_l781_781624

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781624


namespace Jake_later_than_Austin_l781_781113

theorem Jake_later_than_Austin 
    (floors : ℕ) 
    (steps_per_floor : ℕ) 
    (elevator_time_sec : ℕ)
    (steps_per_sec : ℕ) 
    (jh : floors = 9) 
    (spf : steps_per_floor = 30) 
    (et : elevator_time_sec = 60) 
    (steps_sec : steps_per_sec = 3) 
    : 90 - 60 = 30 := 
by
  have total_steps := 9 * 30
  have time_jake := total_steps / 3
  have Jake_additional_time := time_jake - 60
  rw [eq_one_of_eq_succ_eq_succ jh, eq_one_of_eq_succ_eq_succ spf, eq_one_of_eq_succ_eq_succ et, eq_one_of_eq_succ_eq_succ steps_sec] at *
  norm_num at *
  exact Jake_additional_time

end Jake_later_than_Austin_l781_781113


namespace forest_problem_l781_781492

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781492


namespace max_sum_marks_l781_781641

theorem max_sum_marks (a b c : ℕ) (h1 : a + b + c = 2019) (h2 : a ≤ c + 2) : 
  2 * a + b ≤ 2021 :=
by {
  -- We'll skip the proof but formulate the statement following conditions strictly.
  sorry
}

end max_sum_marks_l781_781641


namespace sqrt_floor_eq_seven_l781_781628

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781628


namespace proof_A_l781_781707

def prop_p : Prop := ∀ x : ℝ, log 2 (x^2 + 4) ≥ 2
def prop_q : Prop := ∀ x : ℝ, order_dual (x ^ (1 / 2))

theorem proof_A (hp : prop_p) (hq : ¬ prop_q) : prop_p ∨ ¬ prop_q :=
by
  exact or.inl hp

end proof_A_l781_781707


namespace matches_between_withdrawn_players_l781_781282

theorem matches_between_withdrawn_players (n r : ℕ) (h : 50 = (n - 3).choose 2 + (6 - r) + r) : r = 1 :=
sorry

end matches_between_withdrawn_players_l781_781282


namespace distribute_tasks_l781_781929

theorem distribute_tasks (tasks boys : ℕ) (H_tasks : tasks = 6) (H_boys : boys = 3) : 
  (∑ s in (Finset.powerset (Finset.univ : Finset (Fin tasks))), ite (0 < s.card ∧ s.card < boys) (λ _, 0)) 
  = 540 :=
by
  simp only [H_tasks, H_boys]
  sorry

end distribute_tasks_l781_781929


namespace problem_statement_l781_781757

variable (θ : ℝ)

-- Define given condition
def tan_theta : Prop := Real.tan θ = -2

-- Define the expression to be evaluated
def expression : ℝ := (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ)

-- Theorem statement
theorem problem_statement : tan_theta θ → expression θ = 2 / 5 :=
by
  sorry

end problem_statement_l781_781757


namespace total_number_of_paths_from_A_to_F_is_5_l781_781791

structure Grid where
  nodes : List String
  edges : List (String × String)

def exampleGrid : Grid :=
  { nodes := ["A", "B", "C", "D", "E", "F"],
    edges := [("A", "D"), ("A", "E"), ("A", "B"), ("B", "E"), ("B", "C"), ("B", "F"), ("D", "E"), ("E", "F"), ("C", "F")] }

-- Define a function that counts the number of distinct paths from a start node to an end node in the grid.
def countPaths (g : Grid) (start end : String) : Nat := sorry

-- Assertion statement for the proof
theorem total_number_of_paths_from_A_to_F_is_5 :
  countPaths exampleGrid "A" "F" = 5 := sorry

end total_number_of_paths_from_A_to_F_is_5_l781_781791


namespace conclusion_ratio_ratio_l781_781580

/-
Proof Problem:
Given:
1. The introduction is 450 words.
2. The essay consists of four body sections, each 800 words.
3. The total length of the essay is 5000 words.
Prove that the length of the conclusion is 1350 words and the ratio of the length of the conclusion to the length of the introduction is 3.
-/

theorem conclusion_ratio (intro_len body_len total_len : ℕ) (num_body_sections : ℕ) 
  (h_intro : intro_len = 450) 
  (h_body : body_len = 800) 
  (h_total : total_len = 5000) 
  (h_num_body_sections : num_body_sections = 4): 
  (450 + 4 * 800 + l = 5000) -> l = 1350 := 
by
  have h_total_body_len : 4 * 800 = 3200 := by norm_num,
  have h_sum_intro_body : 450 + 3200 = 3650 := by norm_num,
  have h_l_eq : l = 5000 - 3650 := by rw h_total; norm_num,
  have h_l : l = 1350 := by rw h_l_eq; norm_num,
  exact h_l

theorem ratio (intro_len l : ℕ)
  (h_intro : intro_len = 450)
  (h_l : l = 1350) :
  l / intro_len = 3 :=
by
  rw [h_intro, h_l]
  norm_num

#eval conclusion_ratio 450 800 5000 4 rfl rfl rfl rfl
#eval ratio 450 1350 rfl rfl

end conclusion_ratio_ratio_l781_781580


namespace subsets_excluding_pair_l781_781213

open Finset

theorem subsets_excluding_pair (excluded_pair : Finset ℕ) (h : excluded_pair.card = 2) (h_excluded_pair : excluded_pair ⊆ {1, 23, 45, 67}): 
  (((univ : Finset {x : Finset ℕ // x.card = 2}).image subtype.val).filter (λ s, ¬ excluded_pair ⊆ s)).card = 5 :=
by
  sorry

end subsets_excluding_pair_l781_781213


namespace special_collection_books_l781_781079

def books_in_special_collection (initial_books loaned_books returned_percentage: ℝ) : ℝ :=
  let loaned_books_rounded := loaned_books.ceil
  let returned_books := loaned_books_rounded * returned_percentage
  let not_returned_books := loaned_books_rounded - returned_books
  initial_books - not_returned_books

theorem special_collection_books (initial_books loaned_books: ℝ) (returned_percentage: ℝ) 
  (h_initial_books: initial_books = 75) 
  (h_loaned_books: loaned_books = 29.999999999999996) 
  (h_returned_percentage: returned_percentage = 0.70) : 
  books_in_special_collection initial_books loaned_books returned_percentage = 66 :=
by
  sorry

end special_collection_books_l781_781079


namespace erdos_ginzburg_ziv_l781_781211

theorem erdos_ginzburg_ziv {n : ℕ} (h : Nat.Prime n) (a : Fin (2 * n - 1) → ℤ) :
  ∃ S : Finset (Fin (2 * n - 1)), S.card = n ∧ (∑ i in S, a i) % n = 0 :=
sorry

end erdos_ginzburg_ziv_l781_781211


namespace ellipse_equation_max_area_triangle_l781_781705

-- Definitions based on the conditions
def ellipse_eq (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a c : ℝ) : Prop := c / a = sqrt 3 / 3
def slope_of_line : ℝ := sqrt 3 / 3
def distance (F M : ℝ × ℝ) : ℝ := (F.fst - M.fst)^2 + (F.snd - M.snd)^2
def line_passing_through_left_focus (F : ℝ × ℝ) (k : ℝ) (M : ℝ × ℝ) : Prop :=
  M.snd = k * (M.fst + F.fst)

-- Problem 1: Prove the equation of the ellipse
theorem ellipse_equation (a b : ℝ) (a_gt_b_gt_0 : a > b ∧ b > 0)
  (eccen : eccentricity a (sqrt(a^2 - b^2)))
  (M_eq : ∃ x y, line_passing_through_left_focus (0, -sqrt(a^2 - b^2)) slope_of_line (x, y) ∧
                ellipse_eq a b x y)
  : ellipse_eq 3 2 x y :=
sorry

-- Problem 2: Prove the maximum area of the triangle AOB
theorem max_area_triangle (a b : ℝ) (a_gt_b_gt_0 : a > b ∧ b > 0)
  (eccen : eccentricity a (sqrt(a^2 - b^2)))
  (M_eq : ∃ x y, line_passing_through_left_focus (0, -sqrt(a^2 - b^2)) slope_of_line (x, y) ∧
                ellipse_eq a b x y)
  (symmetry_AB : ∀ A B : ℝ × ℝ, A.snd = B.snd ∧ A.fst = -B.fst)
  : ∃ A B : ℝ × ℝ, ∀ y0 x0 : ℝ, ellipse_eq 3 2 A.fst A.snd ∧ ellipse_eq 3 2 B.fst B.snd ∧
                   A.snd = y0 ∧ B.snd = -y0 ∧ A.fst = x0 ∧ B.fst = x0 ∧
                   (x0 * y0 ≤ sqrt(6) / 2) :=
sorry

end ellipse_equation_max_area_triangle_l781_781705


namespace plywood_long_side_length_l781_781962

theorem plywood_long_side_length (L : ℕ) (h1 : 2 * (L + 5) = 22) : L = 6 :=
by
  sorry

end plywood_long_side_length_l781_781962


namespace pierre_ate_grams_l781_781455

variable (cake_total_weight : ℕ) (parts : ℕ) (nathalie_fraction : ℚ) (pierre_multiple : ℚ)

-- Definitions based on the problem conditions
def part_weight := cake_total_weight / parts

def nathalie_eats := nathalie_fraction * cake_total_weight

def pierre_eats := pierre_multiple * nathalie_eats

-- The actual proof statement goal
theorem pierre_ate_grams (h1 : cake_total_weight = 400)
                         (h2 : parts = 8)
                         (h3 : nathalie_fraction = 1 / 8)
                         (h4 : pierre_multiple = 2)
                         : pierre_eats cake_total_weight parts nathalie_fraction pierre_multiple = 100 := by
  sorry

end pierre_ate_grams_l781_781455


namespace proof_DH_eq_DK_proof_triangle_DKH_sim_ABK_l781_781695

variable {A B C D H K : Type} 

-- Given that ABCD is a parallelogram and A is an acute angle
parameter (parallelogram_ABCD : Parallelogram A B C D)
parameter (angle_A_acute : AcuteAngle A)

-- Points H and K on rays AB and CB respectively
parameter (H_on_AB : OnRay H A B)
parameter (K_on_CB : OnRay K C B)

-- Given CH = BC and AK = AB
parameter (CH_eq_BC : Distance C H = Distance B C)
parameter (AK_eq_AB : Distance A K = Distance A B)

-- Prove part (a) DH = DK
theorem proof_DH_eq_DK : Distance D H = Distance D K := 
  sorry

-- Prove part (b) ∆DKH ∼ ∆ABK
theorem proof_triangle_DKH_sim_ABK : Similar (Triangle D K H) (Triangle A B K) := 
  sorry

end proof_DH_eq_DK_proof_triangle_DKH_sim_ABK_l781_781695


namespace roots_of_equation_l781_781649

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l781_781649


namespace sample_variance_is_2_l781_781699

noncomputable def sample_variance (s : list ℤ) : ℚ :=
match s with
| [a, b, c, d, e] => 
    let mean := (a + b + c + d + e) / 5 in
    let square_diffs := [(a - mean)^2, (b - mean)^2, (c - mean)^2, (d - mean)^2, (e - mean)^2] in
    (square_diffs.sum) / 5
| _ => 0 -- This case won't happen given our specific case

theorem sample_variance_is_2 (x : ℤ) (hx : (1 + 4 + 2 + 5 + x) / 5 = 3) : 
  sample_variance [1, 4, 2, 5, x] = 2 := 
by
  sorry

end sample_variance_is_2_l781_781699


namespace inverse_function_value_l781_781275

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ y : ℝ, f (3^y) = y) : f 3 = 1 :=
sorry

end inverse_function_value_l781_781275


namespace prob_top_three_cards_all_hearts_l781_781966

-- Define the total numbers of cards and suits
def total_cards := 52
def hearts_count := 13

-- Define the probability calculation as per the problem statement
def prob_top_three_hearts : ℚ :=
  (13 * 12 * 11 : ℚ) / (52 * 51 * 50 : ℚ)

-- The theorem states that the probability of the top three cards being all hearts is 11/850
theorem prob_top_three_cards_all_hearts : prob_top_three_hearts = 11 / 850 := by
  -- The proof details are not required, just stating the structure
  sorry

end prob_top_three_cards_all_hearts_l781_781966


namespace min_value_expr_l781_781320

open Real

theorem min_value_expr(p q r : ℝ)(hp : 0 < p)(hq : 0 < q)(hr : 0 < r) :
  (5 * r / (3 * p + q) + 5 * p / (q + 3 * r) + 4 * q / (2 * p + 2 * r)) ≥ 5 / 2 :=
sorry

end min_value_expr_l781_781320


namespace circle_equation_l781_781688

/-- Given a circle passing through points P(4, -2) and Q(-1, 3), and with the length of the segment 
intercepted by the circle on the y-axis as 4, prove that the standard equation of the circle
is (x-1)^2 + y^2 = 13 or (x-5)^2 + (y-4)^2 = 37 -/
theorem circle_equation {P Q : ℝ × ℝ} {a b k : ℝ} :
  P = (4, -2) ∧ Q = (-1, 3) ∧ k = 4 →
  (∃ (r : ℝ), (∀ y : ℝ, (b - y)^2 = r^2) ∧
    ((a - 1)^2 + b^2 = 13 ∨ (a - 5)^2 + (b - 4)^2 = 37)
  ) :=
by
  sorry

end circle_equation_l781_781688


namespace probability_of_odd_face_l781_781953

theorem probability_of_odd_face (h : Fin 12) :
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let chosen_face := faces[h]
  let dots_remaining (n : Nat) := if n % 2 = 0 then n - 1 else n - 1
  let odd_faces := [2, 4, 6, 8, 10, 12]
  (∑ n in odd_faces, 1 / 12) = 1 / 2 := sorry

end probability_of_odd_face_l781_781953


namespace tan_neg_two_simplifies_l781_781763

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l781_781763


namespace min_value_proof_l781_781821

noncomputable def min_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 3 * y = 1) : ℝ :=
  if hx : x = 0 ∨ y = 0 then 
    0 -- this case will not occur due to the h₁ and h₂ constraints
  else
    let a := (1 / x) + (1 / y)
    in 5 + 3 * Real.sqrt 3

theorem min_value_proof (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 3 * y = 1) :
  min_value x y h₁ h₂ h₃ = 5 + 3 * Real.sqrt 3 :=
by
  sorry

end min_value_proof_l781_781821


namespace quadratic_real_solutions_l781_781775

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end quadratic_real_solutions_l781_781775


namespace log_product_simplification_l781_781369

-- Define functions to represent logarithms with different bases
def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

-- Simplify the given logarithm product
theorem log_product_simplification :
    (log_base 8 9) * (log_base 27 32) = 10 / 9 :=
by
  -- This is where the proof would go
  sorry

end log_product_simplification_l781_781369


namespace same_type_polynomial_l781_781934

-- Definitions
def polynomial (p : ℕ → ℕ) : Prop :=
  (p 2 = 1) ∧ (p 1 = 1) ∧ (p 0 = 0)

-- Conditions
def condition : ∀ (p : ℕ → ℕ), polynomial p → p = λ x, if x = 2 then 1 else if x = 1 then 1 else 0 := sorry

-- Statement
theorem same_type_polynomial : polynomial (λ x, if x = 2 then 1 else if x = 1 then 1 else 0) :=
by
  apply condition
  sorry

end same_type_polynomial_l781_781934


namespace gcd_of_36_between_70_and_85_is_81_l781_781893

theorem gcd_of_36_between_70_and_85_is_81 {n : ℕ} (h1 : n ≥ 70) (h2 : n ≤ 85) (h3 : Nat.gcd 36 n = 9) : n = 81 :=
by
  -- proof
  sorry

end gcd_of_36_between_70_and_85_is_81_l781_781893


namespace possible_values_f3_l781_781817

theorem possible_values_f3 (f : ℤ → ℤ) (h : ∀ m n : ℤ, f (m + n) + f (mn + 1) = f m * f n + 2) : 
  (∃ s : ℤ, ∃ n : ℤ, n = 2 ∧ s = -6 ∧ n * s = -12) :=
begin
  sorry
end

end possible_values_f3_l781_781817


namespace floor_sqrt_50_l781_781634

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781634


namespace trig_identity_simplify_trig_identity_evaluate_l781_781449

-- Problem 1
theorem trig_identity_simplify (θ : Real) :
  (cos (θ + π) * sin (θ + 3 * π) ^ 2) / (tan (θ + 4 * π) * tan (θ + π) * cos (-π - θ) ^ 3) = 1 := by
  sorry

-- Problem 2
theorem trig_identity_evaluate :
  (sqrt (1 - 2 * sin 10 * cos 10)) / (cos 10 - sqrt (1 - cos 170 ^ 2)) = 1 := by
  sorry

end trig_identity_simplify_trig_identity_evaluate_l781_781449


namespace no_colorful_number_is_1995_colorful_number_l781_781023

def is_colorful (n : ℕ) : Prop :=
  ∀ m : ℕ, mn_digits (m * n)

def is_n_colorful (n k : ℕ) : Prop :=
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ k ∧ mn_digits (m * n)
  
-- Part (a)
theorem no_colorful_number : ¬ ∃ n : ℕ, is_colorful n := 
by
  sorry

-- Part (b)
theorem is_1995_colorful_number : ∃ n : ℕ, is_n_colorful n 1995 :=
by
  sorry

end no_colorful_number_is_1995_colorful_number_l781_781023


namespace solve_for_x_l781_781862

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end solve_for_x_l781_781862


namespace new_person_weight_l781_781377

theorem new_person_weight :
  (8 * 2.5 + 75 = 95) :=
by sorry

end new_person_weight_l781_781377


namespace trigonometric_expression_evaluation_l781_781750

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l781_781750


namespace find_function_l781_781692

theorem find_function (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(2 + x) = 2 - f(x)) (h2 : ∀ x : ℝ, f(x + 3) ≥ f(x)) :
  ∀ x : ℝ, f(x) = 1 :=
  by
    sorry

end find_function_l781_781692


namespace area_difference_l781_781481

theorem area_difference (d : ℝ) (r : ℝ) (ratio : ℝ) (h1 : d = 10) (h2 : ratio = 2) (h3 : r = 5) :
  (π * r^2 - ((d^2 / (ratio^2 + 1)).sqrt * (2 * d^2 / (ratio^2 + 1)).sqrt)) = 38.5 :=
by
  sorry

end area_difference_l781_781481


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781549

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781549


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781537

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781537


namespace Yoongi_stack_taller_than_Taehyung_l781_781982

theorem Yoongi_stack_taller_than_Taehyung :
  let height_A := 3
  let height_B := 3.5
  let count_A := 16
  let count_B := 14
  let total_height_A := height_A * count_A
  let total_height_B := height_B * count_B
  total_height_B > total_height_A ∧ (total_height_B - total_height_A = 1) :=
by
  sorry

end Yoongi_stack_taller_than_Taehyung_l781_781982


namespace remainder_of_98_mul_102_mod_9_l781_781926

theorem remainder_of_98_mul_102_mod_9 : (98 * 102) % 9 = 6 := 
by 
  -- Introducing the variables and arithmetic
  let x := 98 * 102 
  have h1 : x = 9996 := 
    by norm_num
  have h2 : x % 9 = 6 := 
    by norm_num
  -- Result
  exact h2

end remainder_of_98_mul_102_mod_9_l781_781926


namespace solid_of_revolution_volume_l781_781402

noncomputable def volume_of_solid_of_revolution : ℝ :=
π * ∫ x in 1 .. 4, (4 / x)^2

theorem solid_of_revolution_volume :
  volume_of_solid_of_revolution = 12 * π :=
by
  -- proof omitted
  sorry

end solid_of_revolution_volume_l781_781402


namespace balls_in_boxes_l781_781261

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l781_781261


namespace total_amount_is_200_l781_781568

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l781_781568


namespace has_exactly_two_solutions_iff_l781_781663

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end has_exactly_two_solutions_iff_l781_781663


namespace point_on_inverse_proportion_graph_l781_781718

theorem point_on_inverse_proportion_graph 
  (x y k : ℝ)
  (h1 : y = k / x)
  (h2 : x = 2)
  (h3 : y = 3)
  (hx : (1, 6) ∈ set_of (λ p : ℝ × ℝ, p.snd = k / p.fst)) : True :=
by
  sorry

end point_on_inverse_proportion_graph_l781_781718


namespace transform_negation_l781_781101

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l781_781101


namespace modulus_of_complex_number_l781_781139

-- Define the conditions
def a : ℝ := 2/3
def b : ℝ := 3

-- Define the complex number
def z : ℂ := complex.mk a b

-- Define the expected value
def expected_value : ℝ := real.sqrt 85 / 3

-- Statement of the problem
theorem modulus_of_complex_number :
  complex.abs z = expected_value :=
by sorry

end modulus_of_complex_number_l781_781139


namespace beautiful_numbers_count_l781_781920

def is_beautiful (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → (n * 10 + k) % 11 ≠ 0

def in_range (n : ℕ) : Prop :=
  3100 ≤ n ∧ n ≤ 3600

def beautiful_numbers_in_range : ℕ :=
  (List.range' 3100 (3601-3100)).filter (λ n, is_beautiful n).length

theorem beautiful_numbers_count :
  beautiful_numbers_in_range = 46 :=
sorry

end beautiful_numbers_count_l781_781920


namespace P_is_sufficient_but_not_necessary_for_Q_l781_781680

def P (x : ℝ) : Prop := (2 * x - 3)^2 < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_is_sufficient_but_not_necessary_for_Q : 
  (∀ x, P x → Q x) ∧ (∃ x, Q x ∧ ¬ P x) :=
by
  sorry

end P_is_sufficient_but_not_necessary_for_Q_l781_781680


namespace darry_total_steps_l781_781585

def largest_ladder_steps : ℕ := 20
def largest_ladder_times : ℕ := 12

def medium_ladder_steps : ℕ := 15
def medium_ladder_times : ℕ := 8

def smaller_ladder_steps : ℕ := 10
def smaller_ladder_times : ℕ := 10

def smallest_ladder_steps : ℕ := 5
def smallest_ladder_times : ℕ := 15

theorem darry_total_steps :
  (largest_ladder_steps * largest_ladder_times)
  + (medium_ladder_steps * medium_ladder_times)
  + (smaller_ladder_steps * smaller_ladder_times)
  + (smallest_ladder_steps * smallest_ladder_times)
  = 535 := by
  sorry

end darry_total_steps_l781_781585


namespace compare_a_b_2022_l781_781314

-- Define the sequence a
def a_seq : ℕ → ℝ
| 0 => 1
| (n+1) => 2 / (2 + a_seq n)

-- Define the sequence b
def b_seq : ℕ → ℝ
| 0 => 1
| (n+1) => 3 / (3 + b_seq n)

-- State the Lean theorem
theorem compare_a_b_2022 : a_seq 2022 < b_seq 2022 :=
by
  sorry

end compare_a_b_2022_l781_781314


namespace strawberries_per_person_l781_781309

noncomputable def total_strawberries (baskets : ℕ) (strawberries_per_basket : ℕ) : ℕ :=
  baskets * strawberries_per_basket

noncomputable def kimberly_strawberries (brother_strawberries : ℕ) : ℕ :=
  8 * brother_strawberries

noncomputable def parents_strawberries (kimberly_strawberries : ℕ) : ℕ :=
  kimberly_strawberries - 93

noncomputable def total_family_strawberries (kimberly : ℕ) (brother : ℕ) (parents : ℕ) : ℕ :=
  kimberly + brother + parents

noncomputable def equal_division (total_strawberries : ℕ) (people : ℕ) : ℕ :=
  total_strawberries / people

theorem strawberries_per_person :
  let brother_baskets := 3
  let strawberries_per_basket := 15
  let brother_strawberries := total_strawberries brother_baskets strawberries_per_basket
  let kimberly_straw := kimberly_strawberries brother_strawberries
  let parents_straw := parents_strawberries kimberly_straw
  let total := total_family_strawberries kimberly_straw brother_strawberries parents_straw
  equal_division total 4 = 168 :=
by
  simp [total_strawberries, kimberly_strawberries, parents_strawberries, total_family_strawberries, equal_division]
  sorry

end strawberries_per_person_l781_781309


namespace probability_m_n_units_digit_one_l781_781078

-- Define the sets of m and n
def m_set : Set ℕ := {7, 11, 13, 17}
def n_set : Set ℕ := {n | 2000 ≤ n ∧ n ≤ 2022 ∧ (n % 2 = 0)}

-- Define the units digit function
def units_digit (x : ℕ) : ℕ := x % 10

-- Define the event that m^n has a units digit of 1
def event (m n : ℕ) : Prop := units_digit (m ^ n) = 1

-- Calculate the probability of the event
def probability_event : ℚ := 5 / 8

-- The main statement to prove
theorem probability_m_n_units_digit_one :
  (∃ (m ∈ m_set) (n ∈ n_set), event m n) ∧
  (fraction_of_possible_values m_set n_set event = probability_event) :=
sorry

end probability_m_n_units_digit_one_l781_781078


namespace min_val_n_minus_m_l781_781736

def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)
def g (x : ℝ) : ℝ := (1 / 4 : ℝ) + Real.log (x / 2)

theorem min_val_n_minus_m (m n : ℝ) 
  (h : f m = g n) : n - m = (1 / 2 : ℝ) + Real.log 2 := by
  sorry

end min_val_n_minus_m_l781_781736


namespace incorrect_propositions_l781_781189

def proposition_1 (a b : Line) : Prop :=
  skew_lines a b → ∃! l : Line, (l ⊥ a) ∧ (l ⊥ b)

def proposition_2 (a b : Line) : Prop :=
  (¬ ∃ p : Point, on_line p a ∧ on_line p b) → skew_lines a b

def proposition_3 (a b : Line) : Prop :=
  skew_lines a b → ∀ p : Point, ∃ l : Line, (l ∩ a) ∧ (l ∩ b) ∧ (on_line p l)

def proposition_4 (a b : Line) : Prop :=
  in_plane a → ¬ in_plane b → skew_lines a b

def proposition_5 (a b : Line) : Prop :=
  (∃ p p1 : Plane, ¬ (p = p1) ∧ in_plane a p ∧ in_plane b p1) → skew_lines a b

def proposition_6 (a b c : Line) : Prop :=
  (a ∩ b) ∧ (b ∩ c) → (a ∩ c)

def proposition_7 (a b c : Line) : Prop :=
  skew_lines a b → skew_lines b c → skew_lines a c

theorem incorrect_propositions :
  ¬ proposition_1 a b ∧ ¬ proposition_2 a b ∧ ¬ proposition_3 a b ∧ 
  ¬ proposition_4 a b ∧ ¬ proposition_5 a b ∧ ¬ proposition_6 a b c ∧ 
  ¬ proposition_7 a b c :=
sorry

end incorrect_propositions_l781_781189


namespace special_pairs_even_l781_781169

variables {Π : Type} [nonempty Π]

-- Definition of a polygonal chain
structure PolygonalChain where
  vertices : List Π
  non_self_intersecting : Π → Π → Prop
  no_three_collinear : Π → Π → Π → Prop

-- Definition for special pairs of non-adjacent segments
def is_special (p : PolygonalChain) (i j : ℕ) : Prop :=
  i < p.vertices.length ∧ j < p.vertices.length ∧
  abs (i - j) > 1 ∧ (extension_of (ith_segment p i)).intersects (ith_segment p j)

-- Main theorem statement
theorem special_pairs_even (p : PolygonalChain) : 
  ∃ n : ℕ, even n ∧ n = special_pair_count p :=
sorry

end special_pairs_even_l781_781169


namespace a_pow_10_plus_b_pow_10_l781_781344

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end a_pow_10_plus_b_pow_10_l781_781344


namespace plastering_cost_l781_781487

-- Definitions based on given conditions
def length := 25 -- meters
def width := 12 -- meters
def depth := 6 -- meters
def cost_per_sqm_paise := 75 -- paise per square meter
def cost_per_sqm_rupees := (cost_per_sqm_paise : ℝ) / 100 -- converting paise to rupees

-- Surface area calculations
def surface_area_long_walls := 2 * (length * depth) -- area of long walls
def surface_area_wide_walls := 2 * (width * depth) -- area of wide walls
def surface_area_bottom := length * width -- area of the bottom
def total_surface_area := surface_area_long_walls + surface_area_wide_walls + surface_area_bottom

-- Cost calculation
def total_cost := total_surface_area * cost_per_sqm_rupees

-- The statement we want to prove
theorem plastering_cost : total_cost = 558 := by
  sorry

end plastering_cost_l781_781487


namespace wheel_revolutions_l781_781900

theorem wheel_revolutions (d : ℝ) (distance : ℝ) (miles_to_feet : ℝ) (pi : ℝ) :
  d = 6 →
  distance = 1 →
  miles_to_feet = 5280 →
  pi * (real.circle_area (d / 2)) = 6 * pi →
  (distance * miles_to_feet) / (d * pi) = 880 / pi :=
by
  intros hd hdistance hmiles hpi
  rw [hd, hdistance, hmiles, hpi]
  sorry

end wheel_revolutions_l781_781900


namespace expected_allergies_correct_expected_both_correct_l781_781020

noncomputable def p_allergies : ℚ := 2 / 7
noncomputable def sample_size : ℕ := 350
noncomputable def expected_allergies : ℚ := (2 / 7) * 350

noncomputable def p_left_handed : ℚ := 3 / 10
noncomputable def expected_both : ℚ := (3 / 10) * (2 / 7) * 350

theorem expected_allergies_correct : expected_allergies = 100 := by
  sorry

theorem expected_both_correct : expected_both = 30 := by
  sorry

end expected_allergies_correct_expected_both_correct_l781_781020


namespace part1_geo_seq_cosB_part2_arith_seq_perimeter_l781_781781

-- Part (1)
theorem part1_geo_seq_cosB 
  (a b c : ℝ)
  (A B C : ℝ)
  (cosB : ℝ)
  (hb : a * c = b^2)
  (hcosB: cosB = 12 / 13)
  (hB_pos : 0 < B)
  (hB_lt_pi : B < Real.pi) :
  ((Real.cos A) / (Real.sin A) + (Real.cos C) / (Real.sin C)) = 13/5 := 
sorry

-- Part (2)
theorem part2_arith_seq_perimeter 
  (α : ℝ)
  (B C : ℝ)
  (b : ℝ := 2)
  (h_sum_angles : α + B + C = Real.pi)
  (hB_arith_seq : B = (Real.pi / 3)) :
  (∀ l : ℝ, l = 4 * Real.sin(α + Real.pi / 6) + 2 → l ≤ 6) := 
sorry


end part1_geo_seq_cosB_part2_arith_seq_perimeter_l781_781781


namespace maria_total_distance_l781_781593

-- Definitions
def total_distance (D : ℝ) : Prop :=
  let d1 := D/2   -- Distance traveled before first stop
  let r1 := D - d1 -- Distance remaining after first stop
  let d2 := r1/4  -- Distance traveled before second stop
  let r2 := r1 - d2 -- Distance remaining after second stop
  let d3 := r2/3  -- Distance traveled before third stop
  let r3 := r2 - d3 -- Distance remaining after third stop
  r3 = 270 -- Remaining distance after third stop equals 270 miles

-- Theorem statement
theorem maria_total_distance : ∃ D : ℝ, total_distance D ∧ D = 1080 :=
sorry

end maria_total_distance_l781_781593


namespace equal_chords_l781_781983

noncomputable theory
open_locale real

-- Definition of circles and tangents
variables {O O' : Type*} [metric_space O] [metric_space O']
          {circleO circleO' : set O} {A B A' B' T S T' S' : O} 
          [is_circle circleO] [is_circle circleO']
          (tangent₁ : is_tangent O S')
          (tangent₂ : is_tangent O T')
          (tangent₃ : is_tangent O' S)
          (tangent₄ : is_tangent O' T')

-- Conditions
-- Circle \( \odot O \) and circle \( \odot O' \) are externally tangent.
-- The tangents \( O T' \) and \( O S' \) of \( \odot O' \) intersect \( \odot O \) at points \( A \) and \( B \) respectively.
-- The tangents \( O T \) and \( O S \) of \( \odot O \) intersect \( \odot O' \) at points \( A' \) and \( B' \) respectively.

-- Lean 4 statement equivalent proof problem
theorem equal_chords (h1 : circleO ≠ ∅) (h2 : circleO' ≠ ∅) (hc : ∀ x ∈ circleO, x ∉ circleO') 
  (tang_O_T' : A ∈ circleO) (tang_O_S' : B ∈ circleO)
  (tang_O'_T : A' ∈ circleO') (tang_O'_S : B' ∈ circleO') :
  dist A B = dist A' B' :=
by sorry

end equal_chords_l781_781983


namespace estimated_total_volume_correct_l781_781522

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781522


namespace trapezoid_shorter_base_l781_781784

theorem trapezoid_shorter_base (x y m : ℝ)
  (h1 : m = 4)
  (h2 : x = 100)
  (h3 : (x - y) / 2 = m) : y = 92 :=
begin
  sorry
end

end trapezoid_shorter_base_l781_781784


namespace solve_for_x_l781_781863

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end solve_for_x_l781_781863


namespace roots_of_equation_l781_781647

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l781_781647


namespace base_six_product_correct_l781_781118

namespace BaseSixProduct

-- Definitions of the numbers in base six
def num1_base6 : ℕ := 1 * 6^2 + 3 * 6^1 + 2 * 6^0
def num2_base6 : ℕ := 1 * 6^1 + 4 * 6^0

-- Their product in base ten
def product_base10 : ℕ := num1_base6 * num2_base6

-- Convert the base ten product back to base six
def product_base6 : ℕ := 2 * 6^3 + 3 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Theorem statement
theorem base_six_product_correct : product_base10 = 560 ∧ product_base6 = 2332 := by
  sorry

end BaseSixProduct

end base_six_product_correct_l781_781118


namespace sandy_savings_l781_781312

-- Definition and conditions
def last_year_savings (S : ℝ) : ℝ := 0.06 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_savings (S : ℝ) : ℝ := 1.8333333333333333 * last_year_savings S

-- The percentage P of this year's salary that Sandy saved
def this_year_savings_perc (S : ℝ) (P : ℝ) : Prop :=
  P * this_year_salary S = this_year_savings S

-- The proof statement: Sandy saved 10% of her salary this year
theorem sandy_savings (S : ℝ) (P : ℝ) (h: this_year_savings_perc S P) : P = 0.10 :=
  sorry

end sandy_savings_l781_781312


namespace find_phi_l781_781194

theorem find_phi (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0) 
  (hφ_range : -π / 2 < φ ∧ φ < π / 2) 
  (h_sym_dist : ∃ d, d = π / 6 ∧ d = π / (2ω)) 
  (h_sym_pt : ∃ x, x = 5π / 18 ∧ 2 * sin (ω * x + φ) = 0) 
  : φ = π / 3 := 
sorry

end find_phi_l781_781194


namespace probability_minor_arc_l781_781354

theorem probability_minor_arc (circumference : ℝ) (h_circ : circumference = 3) (A B : ℝ) (hA : 0 ≤ A ∧ A < circumference) (hB : 0 ≤ B ∧ B < circumference) :
  let arc_length := if (B ≥ A) then B - A else circumference + B - A in
  let minor_arc_length := min arc_length (circumference - arc_length) in
  probability (minor_arc_length < 1) = 1/3 :=
sorry

end probability_minor_arc_l781_781354


namespace estimated_total_volume_correct_l781_781524

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781524


namespace floor_sqrt_50_l781_781609

theorem floor_sqrt_50 : ⌊real.sqrt 50⌋ = 7 :=
by
  have h1: 7 < real.sqrt 50 := sorry
  have h2: real.sqrt 50 < 8 := sorry
  have h3: 7 ≤ ⌊real.sqrt 50⌋ := sorry
  have h4: ⌊real.sqrt 50⌋ < 8 := sorry
  exact sorry

end floor_sqrt_50_l781_781609


namespace floor_sqrt_50_l781_781616

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781616


namespace part1_l781_781710

def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - a^2 - 2*a < 0}
def setB (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x - 2*a ∧ x ≤ 2}

theorem part1 (a : ℝ) (h : a = 3) : setA 3 ∪ setB 3 = Set.Ioo (-6) 5 :=
by
  sorry

end part1_l781_781710


namespace B_needs_days_l781_781459

theorem B_needs_days (A_rate B_rate Combined_rate : ℝ) (x : ℝ) (W : ℝ) (h1: A_rate = W / 140)
(h2: B_rate = W / (3 * x)) (h3 : Combined_rate = 60 * W) (h4 : Combined_rate = A_rate + B_rate) :
 x = 140 / 25197 :=
by
  sorry

end B_needs_days_l781_781459


namespace number_of_ways_to_put_balls_in_boxes_l781_781249

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781249


namespace find_a_l781_781720

noncomputable def f (x a : ℝ) : ℝ := -x * Real.log x + a * x
noncomputable def g (x a : ℝ) : ℝ := abs (Real.exp x - a) + (a * a) / 2

theorem find_a :
  (∀ x ∈ Set.Ioo 0 Real.exp 1, (f x 2 ≤ f x y) ↔ (a ≥ 2)) ∧
  (∀ x ∈ Set.Icc 0 (Real.log 3), 
    (g 0 a = -1 + a + (a * a) / 2) ∧ (g (Real.log 3) a = 3 - a + (a * a) / 2) ∧
    (abs ((g (Real.log 3) a) - (g 0 a)) = 3 / 2)
  ) → 
  a = 5 / 2 :=
by
  ext
  sorry

end find_a_l781_781720


namespace process_stops_l781_781096

def grid (n : ℕ) := matrix (fin n) (fin n) ℕ

def operation (G : grid 2021) (dec : ℕ → ℕ) (inc : ℕ → ℕ) : Prop :=
  ∀ i j, G i j ≥ 5 → G (i, j) = dec (G (i, j)) 
             ∧ (∀ nb, nb ∈ neighbors (i, j) → G nb = inc (G nb))

theorem process_stops (G : grid 2021) (S : ℕ) :
  (∀ i j, 0 ≤ G i j ∧ G i j ≤ S) →
  (∀ G', operation G (λ x, x - 4) (λ x, x + 1)) →
  ∃ t, ∀ G'', ¬ operation G'' (λ x, x - 4) (λ x, x + 1) :=
sorry

end process_stops_l781_781096


namespace nested_sum_value_l781_781590

theorem nested_sum_value :
  1005 + (1 / 3) * (1004 + (1 / 3) * (1003 + ... + (1 / 3) * (3 + (1 / 3) * 2) ... )) 
  = 1508 - (7 / (2 * 3^1002)) := by
  sorry

end nested_sum_value_l781_781590


namespace exists_invisible_square_l781_781477

open Int

def isInvisible (p q : ℤ) : Prop := (gcd p q) > 1

theorem exists_invisible_square (n : ℕ) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n → j < n → isInvisible (a + i) (b + j) :=
by
  sorry

end exists_invisible_square_l781_781477


namespace largest_c_inequality_l781_781993

theorem largest_c_inequality (x : Fin 51 → ℝ) (h_sum_zero : ∑ i, x i = 0) :
  (∑ i, (x i)^2) ≥ (1/51) * ( (1 / 51) * ∑ i, |x i| )^2 :=
by
  -- placeholder for proof steps
  sorry

end largest_c_inequality_l781_781993


namespace equation_of_curve_C_equation_of_line_MN_l781_781204

-- Define the points F1, F2, N
def F1 : (ℝ × ℝ) := (-2, 0)
def F2 : (ℝ × ℝ) := (2, 0)
def N : (ℝ × ℝ) := (-4, 0)

-- Define the distance function
def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Provide the necessary properties for the moving point M on curve C
def point_on_curve (M : ℝ × ℝ) : Prop :=
  dist M F1 + dist M F2 = 8

-- Condition of the area ratios for triangles MNF2 and PNF2
def area_ratio_condition (M P : ℝ × ℝ) : Prop :=
  let area_triangle (A B C : (ℝ × ℝ)) : ℝ :=
    0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  area_triangle M N F2 / area_triangle P N F2 = 3 / 2

-- Statement for the equation of curve C
theorem equation_of_curve_C (M : ℝ × ℝ) (hM : point_on_curve M) :
    (M.1^2 / 16) + (M.2^2 / 12) = 1 :=
  sorry

-- Statement for the equation of line MN
theorem equation_of_line_MN (M P: ℝ × ℝ)
  (hM: point_on_curve M)
  (hP: point_on_curve P)
  (h_area_ratio: area_ratio_condition M P) :
    ∃ k : ℝ, k ≠ 0 ∧ (P.2 = k * (P.1 + 4)) ∧ (k = sqrt (21) / 6 ∨ k = -sqrt (21) / 6) :=
  sorry

end equation_of_curve_C_equation_of_line_MN_l781_781204


namespace range_of_a_for_monotonic_function_l781_781772

theorem range_of_a_for_monotonic_function (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 0 ≤ (1 / x) + a) → a ≥ -1 / 2 := 
by
  sorry

end range_of_a_for_monotonic_function_l781_781772


namespace find_fraction_of_number_l781_781946

theorem find_fraction_of_number (N : ℚ) (h : (3/10 : ℚ) * N - 8 = 12) :
  (1/5 : ℚ) * N = 40 / 3 :=
by
  sorry

end find_fraction_of_number_l781_781946


namespace necessary_but_not_sufficient_condition_l781_781163

variable (f : ℝ → ℝ) (m : ℝ)
def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def poly_monotonically_increasing_condition (m : ℝ) : Prop :=
  is_monotonically_increasing (λ x, x^3 - 2*x^2 - m*x + 1)

def condition_m (m : ℝ) : Prop :=
  m > 4 / 3

theorem necessary_but_not_sufficient_condition :
  ∀ m : ℝ, poly_monotonically_increasing_condition m → condition_m m → False :=
  sorry

end necessary_but_not_sufficient_condition_l781_781163


namespace compute_b_from_roots_l781_781179

-- Definition: Given conditions
variable (a b : ℚ)
variable (h_root1 : 2 + Real.sqrt 3) -- root \(2 + \sqrt{3}\)
variable (h_root2 : 2 - Real.sqrt 3) -- corresponding root's conjugate
variable (h_poly : Polynomial ℚ) -- polynomial with rational coefficients

-- Statement: Proving that \( b = -79 \)
theorem compute_b_from_roots (h_poly_def : h_poly = Polynomial.mk [0, b, a, 1]) 
                              (h_rational_a_b : ∀ (x : ℚ), (x ∈ [a, b])) 
                              (h_root1_of_poly : Polynomial.eval (2 + Real.sqrt 3) h_poly = 0) 
                              (h_root2_of_poly : Polynomial.eval (2 - Real.sqrt 3) h_poly = 0) : 
                              b = -79 := 
sorry

end compute_b_from_roots_l781_781179


namespace trig_identity_l781_781400

theorem trig_identity :
  (Real.cos (80 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) + 
   Real.sin (80 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  sorry

end trig_identity_l781_781400


namespace part1_part2_l781_781732

-- Definitions
def A := {x : ℝ | 3 < 3^x ∧ 3^x < 9}
def B := {x : ℝ | 1 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 4}
def C (a : ℝ) := set.Ioo a (a + 3)

-- Part 1: Prove that (C_{R}A) ∩ B = (2,16)
theorem part1 (a : ℝ) : set.Ici (2 : ℝ) ∩ B = set.Ioo 2 16 := 
sorry

-- Part 2: Prove that -1 ≤ a ≤ 1 if A ⊆ C
theorem part2 (a : ℝ) : A ⊆ C a → (-1 : ℝ) ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2_l781_781732


namespace trigonometric_expression_evaluation_l781_781751

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l781_781751


namespace impossible_to_determine_order_with_9_questions_l781_781405

theorem impossible_to_determine_order_with_9_questions : 
  (∀ (weights : List ℝ), weights.length = 5 → (∃ (ask : (List ℝ) → Bool), (∀ i j k : ℕ, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ i < weights.length ∧ j < weights.length ∧ k < weights.length →
   let weights_sorted := weights.iota.sort
   ask ([weights.nth i, weights.nth j, weights.nth k].map (λ x, x < x)))
   ∧ (List.length (tactic.perm.weights searchable -> 5 != 9 → false)))) :=
begin
  sorry
end

end impossible_to_determine_order_with_9_questions_l781_781405


namespace transformation_property_l781_781726

noncomputable def f (x : ℝ) : ℝ :=
  if h1 : -3 ≤ x ∧ x < 0 then 1 - x
  else if h2 : x = 0 ∨ (0 < x ∧ x ≤ 2 ∧ (x - 1 : ℝ) ^ 2 + (-2 : ℝ) ^ 2 ≤ 2 ^ 2) then
    - (1 - (x - 1 : ℝ)^2)^0.5 + (-2)
  else if h3 : 2 < x ∧ x ≤ 3 then 2 * x - 4
  else 0  -- not part of the graph defined

def g (x : ℝ) : ℝ := f (x - 1)

theorem transformation_property : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → g(x) = 2 - x) ∧ 
  (∀ x, 1 ≤ x ∧ x ≤ 3 → (x - 2) ^ 2 + (-2) ^ 2 ≤ 2 ^ 2 → g(x) = - ((x - 2) ^ 2 - 1)^0.5 + (-2)) ∧
  (∀ x, 3 ≤ x ∧ x ≤ 4 → g(x) = 2 * x - 6) :=
by
  sorry

end transformation_property_l781_781726


namespace floor_sqrt_50_l781_781635

theorem floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
by
  sorry

end floor_sqrt_50_l781_781635


namespace coeff_x3_expansion_l781_781665

/-- The coefficient of the x^3 term in the expansion of (1 - x)^6 * (2 - x) is -35. -/
theorem coeff_x3_expansion : 
  (polynomial.expand (λ x, (1 - x)^6 * (2 - x))).coeff 3 = -35 :=
by 
  sorry

end coeff_x3_expansion_l781_781665


namespace fg_jh_cd_eq_one_add_sqrt_five_l781_781291

variable (A B C D E F G H J : ℝ)
variable (isRegularPentagon : ∀ A B C D E : ℝ, true)
variable (AG_eq_1 : G = 1)

theorem fg_jh_cd_eq_one_add_sqrt_five (h_reg_pent : isRegularPentagon A B C D E) (h_ag : AG_eq_1) :
  F + G + C = 1 + Real.sqrt 5 :=
by
  sorry

end fg_jh_cd_eq_one_add_sqrt_five_l781_781291


namespace ernest_used_parts_l781_781137

noncomputable def wire_length : ℝ := 50
noncomputable def num_parts : ℝ := 5
noncomputable def unused_length : ℝ := 20

theorem ernest_used_parts :
  let part_length := wire_length / num_parts in
  let unused_parts := unused_length / part_length in
  (num_parts - unused_parts) = 3 :=
by
  sorry

end ernest_used_parts_l781_781137


namespace last_one_present_is_arn_l781_781372

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 contains 7

def should_eliminate (n : ℕ) : Prop :=
  is_prime n ∨ contains_digit_seven n

def last_student_standing : String :=
  let rec eliminate (students : List String) (count : ℕ) : String :=
    match students with
    | [last] => last
    | _ =>
      let new_students := if should_eliminate count then students.tail else students.rotate_left 1
      eliminate new_students (count + 1)
  eliminate ["Bob", "Cyd", "Dan", "Arn"] 1

theorem last_one_present_is_arn :
  last_student_standing = "Arn" :=
sorry

end last_one_present_is_arn_l781_781372


namespace estimated_total_volume_correct_l781_781528

noncomputable def estimate_total_volume (x̄ ȳ total_x_section_area : ℝ) : ℝ :=
  ȳ / x̄ * total_x_section_area

theorem estimated_total_volume_correct :
  let x̄ := 0.06
  let ȳ := 0.39
  let total_x_section_area := 186
  estimate_total_volume x̄ ȳ total_x_section_area = 1209 :=
by
  unfold estimate_total_volume
  sorry

end estimated_total_volume_correct_l781_781528


namespace floor_sqrt_fifty_l781_781619

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781619


namespace sqrt_floor_eq_seven_l781_781632

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781632


namespace bound_on_k_l781_781122

open Nat

theorem bound_on_k (n k : ℕ) (a : Fin k → ℕ) (h1 : 1 ≤ k)
  (h2 : ∀ i j, i < j → a i < a j) 
  (h3 : ∀ i, 1 ≤ a i ∧ a i ≤ n) 
  (h4 : ∀ i j, 1 ≤ i ∧ i ≤ k → 1 ≤ j ∧ j ≤ k → lcm (a i) (a j) ≤ n) : 
  k ≤ 2 * (⌊sqrt n⌋) := 
sorry

end bound_on_k_l781_781122


namespace num_ways_distribute_balls_l781_781253

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l781_781253


namespace correct_perspective_property_l781_781432

-- Define the four perspective drawing conditions
def perspective_equilateral_triangle_obtuse_scalene : Prop :=
  -- Perspective drawing of an equilateral triangle
  ∀ (T : Type) [triangle T], equilateral T → obtuse_scalene (perspective T)

def perspective_parallelogram_parallelogram : Prop :=
  -- Perspective drawing of a parallelogram
  ∀ (P : Type) [parallelogram P], parallelogram (perspective P)

def perspective_rectangle_parallelogram : Prop :=
  -- Perspective drawing of a rectangle
  ∀ (R : Type) [rectangle R], parallelogram (perspective R)

def perspective_circle_ellipse : Prop :=
  -- Perspective drawing of a circle
  ∀ (C : Type) [circle C], ellipse (perspective C)

-- The goal is to prove that the correct statement about perspective drawing is that
-- a parallelogram remains a parallelogram.
theorem correct_perspective_property :
  perspective_equilateral_triangle_obtuse_scalene →
  perspective_parallelogram_parallelogram →
  perspective_rectangle_parallelogram →
  perspective_circle_ellipse →
  perspective_parallelogram_parallelogram :=
by
  intros h1 h2 h3 h4
  exact h2

end correct_perspective_property_l781_781432


namespace integers_difference_l781_781274

theorem integers_difference (x y : ℤ) (h1 : x = -5) (h2 : y = -4) (h_sum : x + y = -9) (h_diff : y - x = 1) : y - x = 1 := 
by 
  rw [h1, h2]
  exact h_diff.headioon_sym@@3614ta<|vq_8153|>Combining this with our previously identified conditions and the goal of proving the difference between the two integers is \(1\), here is the Lean statement:


end integers_difference_l781_781274


namespace bicycle_selection_ways_l781_781135

theorem bicycle_selection_ways :
  ∃ (ways : ℕ), 
    ways = 15 * 14 * 13 ∧
    (∀ (children : ℕ) (brands : ℕ), children = 3 → brands = 15 → ways = ∏ i in (finset.range children).map (λ n, brands - n), id) :=
begin
  use 2730,
  split,
  { norm_num, },
  { intros children brands hchildren hbrands,
    rw [hchildren, hbrands],
    norm_num }
end

end bicycle_selection_ways_l781_781135


namespace minimum_value_l781_781321

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 128) : 
  ∃ (m : ℝ), (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a * b * c = 128 → (a^2 + 8 * a * b + 4 * b^2 + 8 * c^2) ≥ m) 
  ∧ m = 384 :=
sorry


end minimum_value_l781_781321


namespace isosceles_triangle_angle_l781_781415

theorem isosceles_triangle_angle
  (A B C : ℝ)
  (h1 : A = C)
  (h2 : B = 2 * A - 40)
  (h3 : A + B + C = 180) :
  B = 70 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_angle_l781_781415


namespace smallest_possible_final_number_l781_781446

def initial_squares (n : ℕ) : list ℕ := (list.range (n + 1)).map (λ i, i^2)

def final_number_after_operations (numbers : list ℕ) : ℕ := sorry

theorem smallest_possible_final_number :
  final_number_after_operations (initial_squares 101) = 1 :=
sorry

end smallest_possible_final_number_l781_781446


namespace linear_regression_equation_l781_781297

theorem linear_regression_equation
  (neg_correlated : ∀ x y : ℝ, x * y ≤ 0)
  (mean_x : ℝ := 3)
  (mean_y : ℝ := 2.7)
  (candidate_B : ∀ x : ℝ, y x := -0.2 * x + 3.3)
  (candidate_D : ∀ x : ℝ, y x := -2 * x + 8.6) :
  candidate_B mean_x = mean_y ∧ ∀ x, x * (candidate_B x) ≤ 0 :=
by
  sorry

end linear_regression_equation_l781_781297


namespace increasing_sequence_of_missing_values_l781_781154

-- Definition of a_n as the closest integer to sqrt(n)
def closest_integer_to_sqrt (n : ℕ) : ℕ :=
if √n - Int.floor √n < 0.5 then Int.floor √n else Int.ceil √n

-- Definition of b_n = n + a_n
def b_n (n : ℕ) : ℕ :=
n + closest_integer_to_sqrt n

-- The proof problem statement
theorem increasing_sequence_of_missing_values (n : ℕ) (c : ℕ → ℕ): 
  (∀ k, c k = k^2) → (∀ m, c m = m^2) := 
by
  sorry

end increasing_sequence_of_missing_values_l781_781154


namespace tower_remainder_l781_781065

theorem tower_remainder :
  let T := 1458,
  T % 1000 = 458
:= by
  sorry

end tower_remainder_l781_781065


namespace interior_alternate_angles_implies_parallel_l781_781394

-- Definitions from the conditions
def interior_alternate_angles (l1 l2 : ℝ → ℝ → Prop) (t : ℝ → ℝ → Prop) : Prop :=
  ∃ α β : ℝ → ℝ, 
    t α β ∧ 
    ∀ x y : ℝ, (l1 x y → l2 x y) ∧ (interior_alternate_angles α β = interior_alternate_angles α β)

def are_parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ t : ℝ → ℝ → Prop, interior_alternate_angles l1 l2 t → l1 = l2

-- The main statement
theorem interior_alternate_angles_implies_parallel (l1 l2 : ℝ → ℝ → Prop) (t : ℝ → ℝ → Prop) :
  interior_alternate_angles l1 l2 t → are_parallel l1 l2 :=
by
  sorry

end interior_alternate_angles_implies_parallel_l781_781394


namespace nate_search_time_l781_781343

def sectionG_rows : ℕ := 15
def sectionG_cars_per_row : ℕ := 10
def sectionH_rows : ℕ := 20
def sectionH_cars_per_row : ℕ := 9
def cars_per_minute : ℕ := 11

theorem nate_search_time :
  (sectionG_rows * sectionG_cars_per_row + sectionH_rows * sectionH_cars_per_row) / cars_per_minute = 30 :=
  by
    sorry

end nate_search_time_l781_781343


namespace tv_cost_l781_781334

theorem tv_cost (savings : ℕ) (fraction_spent_on_furniture : ℚ) (amount_spent_on_furniture : ℚ) (remaining_savings : ℚ) :
  savings = 1000 →
  fraction_spent_on_furniture = 3/5 →
  amount_spent_on_furniture = fraction_spent_on_furniture * savings →
  remaining_savings = savings - amount_spent_on_furniture →
  remaining_savings = 400 :=
by
  sorry

end tv_cost_l781_781334


namespace a_pow_10_plus_b_pow_10_l781_781345

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end a_pow_10_plus_b_pow_10_l781_781345


namespace distinct_solution_condition_l781_781657

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l781_781657


namespace balls_in_boxes_l781_781224

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781224


namespace line_parallel_not_coincident_l781_781378

theorem line_parallel_not_coincident (a : ℝ) :
  (a = 3) ↔ (∀ x y, (a * x + 2 * y + 3 * a = 0) ∧ (3 * x + (a - 1) * y + 7 - a = 0) → 
              (∃ k : Real, a / 3 = k ∧ k ≠ 3 * a / (7 - a))) :=
by
  sorry

end line_parallel_not_coincident_l781_781378


namespace projection_c_on_d_l781_781206

open Real

def vector (α : Type*) := α × α

noncomputable def length (v : vector ℝ) : ℝ := 
  sqrt (v.fst * v.fst + v.snd * v.snd)

noncomputable def dot_product (u v : vector ℝ) : ℝ := 
  u.fst * v.fst + u.snd * v.snd

noncomputable def angle (u v : vector ℝ) : ℝ :=
  arccos (dot_product u v / (length u * length v))

noncomputable def projection (u v : vector ℝ) : ℝ :=
  dot_product u v / length v

def a : vector ℝ := (1, 0)
def b_length : ℝ := 2
def b_angle_with_a : ℝ := π / 3
def c := (a.fst + 2 * cos (π / 3), a.snd + 2 * sin (π / 3))
def d := (a.fst - 2 * cos (π / 3), a.snd - 2 * sin (π / 3))

theorem projection_c_on_d : projection c d = -sqrt 3 :=
by sorry

end projection_c_on_d_l781_781206


namespace sin_52_pi_over_3_l781_781448

theorem sin_52_pi_over_3 : Real.sin (52 * Real.pi / 3) = -√3 / 2 := by
  sorry

end sin_52_pi_over_3_l781_781448


namespace cross_section_area_l781_781951

-- Definitions of the conditions
def base_side := 8
def pyramid_height := 12
def middle_line := base_side / 2
def cross_sectional_height := (3/4) * pyramid_height

-- Theorem statement
theorem cross_section_area :
  let MN := middle_line in
  let PH := cross_sectional_height in
  (1/2) * MN * PH = 18 :=
by
  let MN := middle_line
  let PH := cross_sectional_height
  sorry

end cross_section_area_l781_781951


namespace forest_problem_l781_781491

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781491


namespace area_ratio_PQR_area_ratio_DEF_l781_781943

variables {ABC PQR DEF : Type*}
variables {S : ABC → ℝ} {AF FB BD DC CE EA : ℝ}

noncomputable def ratio_PQR (S_ABC : ℝ) (λ1 λ2 λ3 : ℝ) : ℝ :=
(S_ABC * (1 - λ1 * λ2 * λ3)^2) / 
((1 + λ3 + λ1 * λ3) * (1 + λ1 + λ1 * λ2) * (1 + λ2 + λ2 * λ3))

noncomputable def ratio_DEF (S_ABC : ℝ) (λ1 λ2 λ3 : ℝ) : ℝ :=
(S_ABC * (1 + λ1 * λ2 * λ3)) / 
((1 + λ1) * (1 + λ2) * (1 + λ3))

theorem area_ratio_PQR {ABC PQR : Type*}
  (S_ABC : ℝ) (S_PQR : ℝ) (λ1 λ2 λ3 : ℝ) 
  (h1 : AF / FB = λ1) (h2 : BD / DC = λ2) (h3 : CE / EA = λ3) :
  S_PQR / S_ABC = (1 - λ1 * λ2 * λ3)^2 / 
    ((1 + λ3 + λ1 * λ3) * (1 + λ1 + λ1 * λ2) * (1 + λ2 + λ2 * λ3)) :=
sorry

theorem area_ratio_DEF {ABC DEF : Type*}
  (S_ABC : ℝ) (S_DEF : ℝ) (λ1 λ2 λ3 : ℝ) 
  (h1 : AF / FB = λ1) (h2 : BD / DC = λ2) (h3 : CE / EA = λ3) :
  S_DEF / S_ABC = (1 + λ1 * λ2 * λ3) / 
    ((1 + λ1) * (1 + λ2) * (1 + λ3)) :=
sorry

end area_ratio_PQR_area_ratio_DEF_l781_781943


namespace brooks_catch_carter_after_Andrews_l781_781411

variable {d : ℝ} (v_A v_B v_C : ℝ)

-- Initial conditions
def initial_distance_AB := d
def initial_distance_BC := 2 * d

-- Time conditions
def time_to_catch_up_AB := 7
def time_to_catch_up_AC := 12

-- Speed relations based on given times
def speed_relation_AB := v_A - v_B = d / 7
def speed_relation_AC := v_A - v_C = d / 4

-- Relative speed difference between Brooks and Carter
def relative_speed_BC := v_B - v_C = (d / 7 - d / 4)

-- Time for Brooks to catch up with Carter using relative speed
def time_to_catch_up_BC := 2 * d / (v_B - v_C)

-- Converting time to 'minutes after Andrews catches up with Carter'
def solution := time_to_catch_up_BC - 12

-- Target proof statement
theorem brooks_catch_carter_after_Andrews :
  initial_distance_AB = d →
  initial_distance_BC = 2 * d →
  time_to_catch_up_AB = 7 →
  time_to_catch_up_AC = 12 →
  speed_relation_AB →
  speed_relation_AC →
  relative_speed_BC →
  solution = 6 + 2 / 3 :=
begin
  intros,
  sorry
end

end brooks_catch_carter_after_Andrews_l781_781411


namespace championship_winner_l781_781902

def Class (n : Nat) : Prop := True

def A1 : Prop := Class 902 -- A's statement "Class 902 wins the championship"
def A2 : Prop := Class 904 -- A's statement "Class 904 gets 3rd"
def B1 : Prop := Class 901 -- B's statement "Class 901 gets 4th"
def B2 : Prop := Class 903 -- B's statement "Class 903 gets 2nd"
def C1 : Prop := Class 903 -- C's statement "Class 903 gets 3rd"
def C2 : Prop := Class 904 -- C's statement "Class 904 wins the championship"

theorem championship_winner :
  (xor A1 A2) ∧
  (xor B1 B2) ∧
  (xor C1 C2) →
  Class 902 := by
  sorry

end championship_winner_l781_781902


namespace solve_for_x_l781_781860

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_l781_781860


namespace train_travel_time_l781_781091

theorem train_travel_time (d v : ℝ) (hv : 0 < v) : 
  let t1 := d / (2 * v),
      t2 := d / (8 * v) in
  t1 + 15 + t2 = 40 := 
by 
  sorry

end train_travel_time_l781_781091


namespace probability_of_authentic_and_defective_l781_781157

theorem probability_of_authentic_and_defective :
  (let total_items := 5 in
   let authentic_items := 4 in
   let defective_items := 1 in
   let total_outcomes := Nat.choose total_items 2 in
   let favorable_outcomes := (Nat.choose authentic_items 1) * (Nat.choose defective_items 1) in
   favorable_outcomes / total_outcomes = 2 / 5) :=
by
  let total_items := 5
  let authentic_items := 4
  let defective_items := 1
  let total_outcomes := Nat.choose total_items 2
  let favorable_outcomes := (Nat.choose authentic_items 1) * (Nat.choose defective_items 1)
  have h_total_outcomes : total_outcomes = 10 := by sorry
  have h_favorable_outcomes : favorable_outcomes = 4 := by sorry
  have h_probability := (favorable_outcomes / total_outcomes : ℚ)
  rw [h_total_outcomes, h_favorable_outcomes]
  norm_num
  exact h_probability

-- This theorem states that given 5 items where 4 are authentic and 1 is defective,
-- the probability of selecting one authentic and one defective item is 2/5.

end probability_of_authentic_and_defective_l781_781157


namespace less_than_51_percent_vertex_cover_probability_l781_781783

-- Define the graph and conditions
variable (G : Type) [Graph G]

-- Define a function for vertex cover probability within the graph
noncomputable def vertex_cover_probability (G : Type) [Graph G] (n : ℕ) : ℝ := sorry

-- Given a graph G with 1996 vertices and no isolated vertices
axiom graph_with_1996_vertices_no_isolated (G : Type) [Graph G] : 
  (vertex_set G).card = 1996 ∧ (∀ v ∈ vertex_set G, (degree G v) ≠ 0)

-- The main theorem to be proven
theorem less_than_51_percent_vertex_cover_probability 
  (G : Type) [Graph G] 
  (h : graph_with_1996_vertices_no_isolated G) : 
  vertex_cover_probability G 1996 < 0.51 :=
sorry

end less_than_51_percent_vertex_cover_probability_l781_781783


namespace necessity_not_sufficiency_l781_781691

section

variables {F M : Type} {l : Type} (d : ℝ)

def is_focus (F : Point) (M : Point) (l : Line) := -- F is a fixed point

def is_directrix (l : Line) (M : Point) := -- l is a fixed line

def is_moving_point (M : Point) := -- M is a moving point

def distance_to_line (M : Point) (l : Line) := d -- Distance from M to l is d

def parabola_condition (M : Point) (F : Point) (l : Line) := -- The condition |MF| = d

theorem necessity_not_sufficiency 
  (F : Point) (l : Line) (M : Point) 
  (d : ℝ) 
  (h1 : is_focus F M l)
  (h2 : is_directrix l M)
  (h3 : is_moving_point M) 
  (h4 : distance_to_line M l = d)
  (h5 : parabola_condition M F l) : 
  necessary_but_not_sufficient_condition (|MF| = d) (trajectory_is_parabola M F l) :=
sorry
end

end necessity_not_sufficiency_l781_781691


namespace equal_roots_implies_specific_m_l781_781591

theorem equal_roots_implies_specific_m (m : ℝ) :
  (∀ x : ℝ, (x^2 * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x^2 / m) →
  (∃ x_1 x_2 : ℝ, x_1 = x_2 ∧ (x_1^2 * (x_1 - 2) - (m + 2)) / ((x_1 - 2) * (m - 2)) = x_1^2 / m) →
  (m = -1 + real.sqrt 3 ∨ m = -1 - real.sqrt 3) :=
by
  sorry

end equal_roots_implies_specific_m_l781_781591


namespace small_samovar_cools_faster_l781_781021

theorem small_samovar_cools_faster
    (n : ℝ) (V_S V_L A_S A_L : ℝ)
    (hV : V_L = n^3 * V_S)
    (hA : A_L = n^2 * A_S)
    (hQ : ∀ (Q : ℝ), Q ∝ A) :
  ∃ k : ℝ, (k * A_L / V_L = k * A_S / V_S / n) → ∀ (t : ℝ), t = "small" := sorry

end small_samovar_cools_faster_l781_781021


namespace quadratic_real_solutions_l781_781776

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end quadratic_real_solutions_l781_781776


namespace triangle_incircle_ineq_l781_781414

/-- Given a triangle ABC inscribed in a circle C, and A', B', C' are points where the angle bisectors of ∠A, ∠B, ∠C meet the circircle again respectively, 
    let I be the incenter of triangle ABC. Prove that 
    (IA' / IA) + (IB' / IB) + (IC' / IC) ≥ 3 and IA' + IB' + IC' ≥ IA + IB + IC. -/
theorem triangle_incircle_ineq (ABC C : Type) (A B C A' B' C' I : ABC)
    (h0 : inscribed_triangle ABC C)
    (h1 : angle_bisectors_meet A B C A' B' C' C)
    (h2 : incenter I ABC) :
  (IA' / IA) + (IB' / IB) + (IC' / IC) ≥ 3 ∧ IA' + IB' + IC' ≥ IA + IB + IC := 
begin
  sorry -- Proof goes here
end

end triangle_incircle_ineq_l781_781414


namespace correct_system_of_equations_l781_781410

-- Define the conditions
def speed_uphill : ℝ := 3
def speed_flat : ℝ := 4
def speed_downhill : ℝ := 5
def time_A_to_B_hours : ℝ := 54 / 60
def time_B_to_A_hours : ℝ := 42 / 60

-- Define the lengths of the sections
variable (x y : ℝ)

-- Define the problem in terms of the given conditions
theorem correct_system_of_equations :
  (x / speed_uphill + y / speed_flat = time_A_to_B_hours) ∧
  (x / speed_downhill + y / speed_flat = time_B_to_A_hours) :=
sorry

end correct_system_of_equations_l781_781410


namespace magnitude_of_b_l781_781737

def vector_a : ℝ × ℝ × ℝ := (-1, 2, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (3, x, 1)

theorem magnitude_of_b (x : ℝ) (h : vector_a.1 * vector_b x.1 + vector_a.2 * vector_b x.2 + vector_a.3 * vector_b x.3 = 0) : 
  ∥vector_b 1∥ = real.sqrt 11 :=
  sorry

end magnitude_of_b_l781_781737


namespace Austin_work_hours_on_Wednesdays_l781_781559

variable {W : ℕ}

theorem Austin_work_hours_on_Wednesdays
  (h1 : 5 * 2 + 5 * W + 5 * 3 = 25 + 5 * W)
  (h2 : 6 * (25 + 5 * W) = 180)
  : W = 1 := by
  sorry

end Austin_work_hours_on_Wednesdays_l781_781559


namespace project_B_days_l781_781458

theorem project_B_days (B : ℕ) : 
  (1 / 20 + 1 / B) * 10 + (1 / B) * 5 = 1 -> B = 30 :=
by
  sorry

end project_B_days_l781_781458


namespace even_quadruple_composition_l781_781829

variable {α : Type*} [AddGroup α]

-- Definition of an odd function
def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

theorem even_quadruple_composition {f : α → α} 
  (hf_odd : is_odd_function f) : 
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
by
  sorry

end even_quadruple_composition_l781_781829


namespace adam_simon_distance_l781_781095

theorem adam_simon_distance :
  ∀ (x : ℝ), (∀ x, sqrt ((12 * x)^2 + (6 * x)^2) = 90 ↔ x = 3 * sqrt 5) :=
begin
  assume x,
  sorry,
end

end adam_simon_distance_l781_781095


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781548

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781548


namespace balls_in_boxes_l781_781226

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781226


namespace trajectory_equation_l781_781205

open Real

-- Define points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the moving point P
def P (x y : ℝ) : Prop := 
  (4 * Real.sqrt ((x + 2) ^ 2 + y ^ 2) + 4 * (x - 2) = 0) → 
  (y ^ 2 = -8 * x)

-- The theorem stating the desired proof problem
theorem trajectory_equation (x y : ℝ) : P x y :=
sorry

end trajectory_equation_l781_781205


namespace angle_in_polygon_l781_781941

variables (n : ℕ) (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)

def is_regular_polygon (A : Fin n → ℝ × ℝ) : Prop := 
  ∃ r : ℝ, ∀ i : Fin n, ∃ θ : ℝ, A i = (r * Real.cos θ, r * Real.sin θ) ∧ 
  ∀ j : Fin n, ∃ ψ : ℝ, A j = (r * Real.cos (θ + 2 * j * π / n), r * Real.sin (θ + 2 * j * π / n))

def is_interior : Prop := 
  ∃ (θ₁ θ₂ : ℝ), θ₁ < θ₂ ∧ 0 ≤ θ₁ ∧ θ₂ ≤ 2 * π ∧ ∀ p ∈ A, p ∈ interval θ₁ θ₂

theorem angle_in_polygon
  (h1 : is_regular_polygon n A)
  (h2 : is_interior O) :
  ∃ (i j : Fin n), i < j ∧ π * (1 - 1 / n) ≤ ∠(A i, O, A j) ∧ ∠(A i, O, A j) ≤ π :=
sorry

end angle_in_polygon_l781_781941


namespace cone_curved_surface_area_l781_781443

theorem cone_curved_surface_area (r l : ℝ) (π : ℝ) : 
  r = 10 ∧ l = 21 ∧ π = real.pi → (π * r * l = 210 * real.pi) :=
by
  assume h : r = 10 ∧ l = 21 ∧ π = real.pi
  sorry

end cone_curved_surface_area_l781_781443


namespace min_value_frac_inv_sum_l781_781822

theorem min_value_frac_inv_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + 3 * y = 1) : 
  ∃ (minimum_value : ℝ), minimum_value = 4 + 2 * Real.sqrt 3 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 1 → (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3) := 
sorry

end min_value_frac_inv_sum_l781_781822


namespace max_intersection_points_of_perpendiculars_l781_781173

/-- Given five points in a plane and that any pair of points forms a line that is
neither parallel, perpendicular, nor coincident. Perpendiculars are drawn from each point
to the lines formed by connecting the remaining four points. Prove that the maximum number
of intersection points of these perpendiculars, excluding the given five points, is 310. -/
theorem max_intersection_points_of_perpendiculars
  (P : Fin 5 → Point)
  (h : ∀ i j, i ≠ j → ¬(P i = P j))
  (non_parallel_non_perpendicular_non_coincident: ∀ i j k, i ≠ j → j ≠ k → k ≠ i → 
  ¬((line_through (P i) (P j)).parallel (line_through (P j) (P k)) 
  ∨ (line_through (P i) (P j)).perpendicular (line_through (P j) (P k)) 
  ∨ (line_through (P i) (P j)).coincident (line_through (P j) (P k)))) :
  ∃ (n : ℕ), n = 310 :=
begin
  sorry
end

end max_intersection_points_of_perpendiculars_l781_781173


namespace base6_multiplication_l781_781120

-- Definitions of the base-six numbers
def base6_132 := [1, 3, 2] -- List representing 132_6
def base6_14 := [1, 4] -- List representing 14_6

-- Function to convert a base-6 list to a base-10 number
def base6_to_base10 (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc x, acc * 6 + x) 0

-- The conversion of our specified numbers to base-10
def base10_132 := base6_to_base10 base6_132
def base10_14 := base6_to_base10 base6_14

-- The product of the conversions
def base10_product := base10_132 * base10_14

-- Function to convert a base-10 number to a base-6 list
def base10_to_base6 (n : ℕ) : List ℕ :=
  let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else loop (n / 6) ((n % 6) :: acc)
  loop n []

-- The conversion of the product back to base-6
def base6_product := base10_to_base6 base10_product

-- The expected base-6 product
def expected_base6_product := [1, 3, 3, 2]

-- The formal theorem statement
theorem base6_multiplication :
  base6_product = expected_base6_product := by
  sorry

end base6_multiplication_l781_781120


namespace sqrt_floor_eq_seven_l781_781629

theorem sqrt_floor_eq_seven :
  ∀ (x : ℝ), (49 < x ∧ x < 64) ∧ sqrt 49 = 7 ∧ sqrt 64 = 8 → floor (sqrt 50) = 7 :=
by
  intro x
  sorry

end sqrt_floor_eq_seven_l781_781629


namespace find_b_l781_781373

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  2 * x / (x^2 + b * x + 1)

noncomputable def f_inverse (y : ℝ) : ℝ :=
  (1 - y) / y

theorem find_b (b : ℝ) (h : ∀ x, f_inverse (f x b) = x) : b = 4 :=
sorry

end find_b_l781_781373


namespace decision_on_manufacturers_l781_781992

def manufacturer_a_data : List ℕ := [3, 4, 5, 5, 5, 7, 9, 10, 12, 13, 15]
def manufacturer_b_data : List ℕ := [3, 3, 4, 5, 5, 6, 8, 8, 8, 10, 11]
def manufacturer_c_data : List ℕ := [3, 3, 4, 4, 4, 8, 9, 10, 11, 12, 13]

-- Definitions for mean, median, and mode
def mean (l : List ℕ) : ℚ :=
    (l.foldl (λ acc x => acc + x) 0 : ℚ) / l.length

def median (l : List ℕ) : ℚ :=
  let sorted := l.sorted in
    if sorted.length % 2 = 0 then
      ((sorted.get! (sorted.length/2 - 1) + sorted.get! (sorted.length/2)) : ℚ) / 2
    else
      sorted.get! (sorted.length/2)

def mode (l : List ℕ) : ℕ :=
  let m := l.foldl (λ acc x => acc.insert x (acc.findD x 0 + 1)) Std.HashMap.empty
  m.toList.foldl (λ acc x => if x.2 > acc.2 then x else acc) (0,0) |>.fst  


-- Lean theorem statement
theorem decision_on_manufacturers:
(mean manufacturer_a_data = 8 ∧ median manufacturer_a_data = 7 ∧ mode manufacturer_a_data = 5) →
(mean manufacturer_b_data = 6.45 ∧ median manufacturer_b_data = 6 ∧ mode manufacturer_b_data = 8) →
(mean manufacturer_c_data = 7.36 ∧ median manufacturer_c_data = 8 ∧ mode manufacturer_c_data = 4) →
"Manufacturer A used the mean, Manufacturer B used the mode, and Manufacturer C used the median." ∧ 
"The customers should choose Manufacturer A's product because its average lifespan is higher than that of the other two manufacturers."
:= by
  sorry

end decision_on_manufacturers_l781_781992


namespace speed_in_still_water_l781_781958

-- Definitions for the conditions
def upstream_speed : ℕ := 30
def downstream_speed : ℕ := 60

-- Prove that the speed of the man in still water is 45 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end speed_in_still_water_l781_781958


namespace juice_dispenser_capacity_l781_781891

theorem juice_dispenser_capacity (x : ℕ) (h : 0.48 * x = 60) : x = 125 :=
by
  sorry

end juice_dispenser_capacity_l781_781891


namespace find_different_weighted_coins_l781_781045

-- Define the conditions and the theorem
def num_coins : Nat := 128
def weight_types : Nat := 2
def coins_of_each_weight : Nat := 64

theorem find_different_weighted_coins (weighings_at_most : Nat := 7) :
  ∃ (w1 w2 : Nat) (coins : Fin num_coins → Nat), w1 ≠ w2 ∧ 
  (∃ (pair : Fin num_coins × Fin num_coins), pair.fst ≠ pair.snd ∧ coins pair.fst ≠ coins pair.snd) :=
sorry

end find_different_weighted_coins_l781_781045


namespace probability_ge_first_second_l781_781467

noncomputable def probability_ge_rolls : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_ge_first_second :
  probability_ge_rolls = 9 / 16 :=
by
  sorry

end probability_ge_first_second_l781_781467


namespace field_trip_buses_needed_l781_781866

theorem field_trip_buses_needed
    (fifth_graders : ℕ) (sixth_graders : ℕ) (seventh_graders : ℕ)
    (teachers_per_grade : ℕ) (parents_per_grade : ℕ)
    (grades : ℕ) (seats_per_bus : ℕ)
    (H_fg : fifth_graders = 109)
    (H_sg : sixth_graders = 115)
    (H_sg2 : seventh_graders = 118)
    (H_tpg : teachers_per_grade = 4)
    (H_ppg : parents_per_grade = 2)
    (H_gr : grades = 3)
    (H_spb : seats_per_bus = 72) :
    let students := fifth_graders + sixth_graders + seventh_graders,
        adults := grades * (teachers_per_grade + parents_per_grade),
        total_people := students + adults in
    total_people / seats_per_bus = 5 := by
    sorry

end field_trip_buses_needed_l781_781866


namespace passwordLockProbability_l781_781080

theorem passwordLockProbability
    (digits : Fin 10 → Fin 10 → Fin 10 → Fin 10 → Bool) 
    (firstTwoKnown : (a b : Fin 10) → ∃ (c d : Fin 10), digits a b c d = true)
    (forgotLastTwo : ∀ (c d : Fin 10), digits c d = true) :
    (1 : ℝ) / 100 = (1 : ℝ) / 10 / 10 :=
by
    sorry

end passwordLockProbability_l781_781080


namespace problem_equiv_solution_set_l781_781033

noncomputable def same_solution_set (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = 0 ↔ g x = 0

theorem problem_equiv_solution_set :
  same_solution_set (λ x, sin x + cos x) (λ x, cos (2 * x) / (sin x - cos x)) :=
by
  sorry

end problem_equiv_solution_set_l781_781033


namespace estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781507

noncomputable def totalRootAreaSum : ℝ := 0.6
noncomputable def totalVolumeSum : ℝ := 3.9
noncomputable def squaredRootAreaSum : ℝ := 0.038
noncomputable def squaredVolumeSum : ℝ := 1.6158
noncomputable def crossProductSum : ℝ := 0.2474
noncomputable def totalForestRootArea : ℝ := 186

noncomputable def averageRootArea : ℝ := totalRootAreaSum / 10
noncomputable def averageVolume : ℝ := totalVolumeSum / 10

noncomputable def sampleCorrelationCoefficient : ℝ :=
  crossProductSum / (Math.sqrt (squaredRootAreaSum * squaredVolumeSum))

noncomputable def estimatedTotalVolume : ℝ :=
  (averageVolume / averageRootArea) * totalForestRootArea

theorem estimate_average_root_area : averageRootArea = 0.06 := by
  sorry

theorem estimate_average_volume : averageVolume = 0.39 := by
  sorry

theorem calculate_sample_correlation : sampleCorrelationCoefficient ≈ 0.97 := by
  sorry

theorem estimate_total_volume : estimatedTotalVolume = 1209 := by
  sorry

end estimate_average_root_area_estimate_average_volume_calculate_sample_correlation_estimate_total_volume_l781_781507


namespace transform_negation_l781_781102

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l781_781102


namespace max_students_with_bonus_l781_781342

/-- Define the problem conditions -/
def class_size : ℕ := 81
def median_bonus_points_condition (scores : list ℕ) : Prop :=
  list.length scores = class_size ∧
  ∃ median, median = list.nth_le scores (list.length scores / 2) (by linarith) ∧
  ∀ i, (i > list.length scores / 2) → (list.nth_le scores i (by linarith) > median)

/-- State the theorem that needs to be proven -/
theorem max_students_with_bonus (scores : list ℕ) (h : median_bonus_points_condition scores) :
  ∃ n, n = 40 := 
sorry

end max_students_with_bonus_l781_781342


namespace john_climbs_9_flights_l781_781302

variable (fl : Real := 10)  -- Each flight of stairs is 10 feet
variable (step_height_inches : Real := 18)  -- Each step is 18 inches
variable (steps : Nat := 60)  -- John climbs 60 steps

theorem john_climbs_9_flights :
  (steps * (step_height_inches / 12) / fl = 9) :=
by
  sorry

end john_climbs_9_flights_l781_781302


namespace solve_xy_l781_781142

theorem solve_xy (x y : ℝ) :
  (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1 / 3 → 
  x = 34 / 3 ∧ y = 35 / 3 :=
by
  intro h
  sorry

end solve_xy_l781_781142


namespace coefficient_of_x_l781_781290

theorem coefficient_of_x :
  let p := x^2 + 3 * x + 2
  in let expansion := p ^ 5
  in ∃ k : ℕ, k = 240 ∧ has_term_with_coefficient expansion (λ term, term = x) k :=
by
  sorry

end coefficient_of_x_l781_781290


namespace total_cats_l781_781339

theorem total_cats (a b c d : ℝ) (ht : a = 15.5) (hs : b = 11.6) (hg : c = 24.2) (hr : d = 18.3) :
  a + b + c + d = 69.6 :=
by
  sorry

end total_cats_l781_781339


namespace boyA_not_adjacent_arrangements_count_l781_781009

/-- 
Three boys (B1, B2, B_A) and three girls (G1, G2, G3) are to be lined up in a row. Prove that 
the number of different arrangements where boy B_A is not adjacent to the other two boys 
(B1 and B2) is 288.
-/
theorem boyA_not_adjacent_arrangements_count : 
  ∃ (A B1 B2 G1 G2 G3 : Type), 
  (¬ adjacent (A, B1) ∧ ¬ adjacent (A, B2)) → 
  arrangement_count (A, B1, B2, G1, G2, G3) = 288
  :=
sorry

end boyA_not_adjacent_arrangements_count_l781_781009


namespace part1_part2_part3_l781_781704

section
  variable {x y b k k₁ k₂ k₃ k₄ : ℝ}

  -- Given definitions from conditions:
  def ellipse (x y : ℝ) := (x^2 / 6) + (y^2 / 3) = 1
  def P := (-2, 1)
  def Q := (2, -1)
  def C (a b : ℝ) := a * x + b * y = 1
  def line_l (k b : ℝ) (x y : ℝ) := y = k * x + b

  -- Proof statements (missing proofs):
  theorem part1 : k = 1 → ∃ b, C (-2/3 : ℝ) (1/3 : ℝ) :=
  sorry

  theorem part2 : k = 2 → ∀ b, -5 < b ∧ b < 5 → 
                      ( (20/9 : ℝ) < sqrt(5) * |2 * sqrt(54 - 2 * b^2) / 9| ∧ 
                       |2 * sqrt(54 - 2 * b^2) / 9| <= (10 * sqrt 6 / 3 : ℝ) ) := 
  sorry

  theorem part3 ( b ≠ 0 ) ( midpoint_cond: ∀ x₁ y₁ x₂ y₂, 
                          (x₁ + x₂) / 2 + (y₁ + y₂) / 2 = 0 ) :
      let k₁ k₂ : ℚ := (1/2 : ℚ) 
      let k₃ k₄ : ℚ := (1/2 : ℚ) 
      k₁ * k₂ = (1/2 : ℚ) ∧ (k₁ ^ 2 + k₂ ^ 2 > 2 * k₃ * k₄) :=
  sorry
end

end part1_part2_part3_l781_781704


namespace floor_sqrt_fifty_l781_781625

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781625


namespace find_stream_speed_l781_781452

-- Define the problem based on the provided conditions
theorem find_stream_speed (b s : ℝ) (h1 : b + s = 250 / 7) (h2 : b - s = 150 / 21) : s = 14.28 :=
by
  sorry

end find_stream_speed_l781_781452


namespace roots_of_equation_l781_781648

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l781_781648


namespace floor_sqrt_50_l781_781604

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781604


namespace transform_negation_l781_781100

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l781_781100


namespace forest_problem_l781_781499

noncomputable def avg_root_cross_sectional_area (x : Fin 10 → ℝ) : ℝ :=
  (∑ i, x i) / 10

noncomputable def avg_volume (y : Fin 10 → ℝ) : ℝ :=
  (∑ i, y i) / 10

noncomputable def sample_corr_coeff (x y : Fin 10 → ℝ) (x_bar y_bar : ℝ) : ℝ :=
  let numerator := ∑ i, (x i - x_bar) * (y i - y_bar)
  let denominator_x := ∑ i, (x i - x_bar) ^ 2
  let denominator_y := ∑ i, (y i - y_bar) ^ 2
  numerator / (Real.sqrt (denominator_x * denominator_y))

noncomputable def estimate_total_volume (x_bar y_bar : ℝ) (total_x : ℝ) : ℝ :=
  (y_bar / x_bar) * total_x

theorem forest_problem
  (x y : Fin 10 → ℝ)
  (hx_sum : ∑ i, x i = 0.6)
  (hy_sum : ∑ i, y i = 3.9)
  (hx_sum_sq : ∑ i, (x i) ^ 2 = 0.038)
  (hy_sum_sq : ∑ i, (y i) ^ 2 = 1.6158)
  (hxy_sum : ∑ i, x i * y i = 0.2474)
  (total_root_area : ℝ := 186) :
  let x_bar := avg_root_cross_sectional_area x
  let y_bar := avg_volume y
  x_bar = 0.06 ∧
  y_bar = 0.39 ∧
  sample_corr_coeff x y x_bar y_bar = 0.97 ∧
  estimate_total_volume x_bar y_bar total_root_area = 1209 :=
by
  sorry

end forest_problem_l781_781499


namespace cos_sin_eq_range_l781_781155

theorem cos_sin_eq_range (x k : ℝ) : 
  (∃ x : ℝ, cos (2*x) - 2 * sqrt 3 * sin x * cos x = k + 1) ↔ k ∈ Icc (-3 : ℝ) 1 := 
sorry

end cos_sin_eq_range_l781_781155


namespace angle_sine_condition_l781_781780

theorem angle_sine_condition (A B C : ℝ) (h₀ : A + B + C = 180) (h₁ : A > 30) :
  (A > 30 → sin (A * (Real.pi / 180)) > 1 / 2) ∧ (¬ (sin (A * (Real.pi / 180)) > 1 / 2) → A ≤ 30) :=
sorry

end angle_sine_condition_l781_781780


namespace translate_sin_right_up_l781_781382

theorem translate_sin_right_up (x : ℝ) :
  (sin (x - π / 2) + 1) = 1 - cos x :=
by
  sorry

end translate_sin_right_up_l781_781382


namespace average_speed_is_correct_l781_781460

variable (speed1 speed2 speed3 speed4 : ℝ)
variable (distance1 distance2 distance3 distance4 : ℝ)
variable (time1 time2 time3 time4 : ℝ)

def total_distance : ℝ :=
  distance1 + distance2 + distance3 + distance4

def total_time : ℝ :=
  time1 + time2 + time3 + time4

def average_speed : ℝ :=
  total_distance / total_time

axiom h1 : speed1 = 30
axiom h2 : distance1 = 15
axiom h3 : time1 = distance1 / speed1

axiom h4 : speed2 = 60
axiom h5 : distance2 = 35
axiom h6 : time2 = distance2 / speed2

axiom h7 : speed3 = (40 + 80) / 2
axiom h8 : distance3 = 10
axiom h9 : time3 = distance3 / speed3

axiom h10 : speed4 = 36
axiom h11 : distance4 = speed4 * 0.5  -- 30 minutes is 0.5 hours
axiom h12 : time4 = 0.5

theorem average_speed_is_correct :
  average_speed = 44.57 := by
  sorry

end average_speed_is_correct_l781_781460


namespace greatest_number_of_elements_l781_781085

-- Definitions of the set and conditions
def is_valid_set (T : Set ℕ) : Prop :=
  ∃ (M : ℕ) (m : ℕ), 
  (∀ y ∈ T, ∃ m : ℕ, (M - y) % m = 0) ∧ -- Arithmetic mean condition
  2 ∈ T ∧ -- Condition: 2 is in the set
  (∀ y ∈ T, y ≠ M) ∧ -- Condition: All elements are distinct positive integers.
  (∃ x ∈ T, x = 1001) -- Condition: 1001 is the largest element

-- The proof problem
theorem greatest_number_of_elements (T : Set ℕ) (hT : is_valid_set T) : 
  ∃ n : ℕ, n = 28 ∧ card(T) = n :=
sorry

end greatest_number_of_elements_l781_781085


namespace area_transformation_l781_781814

theorem area_transformation (A : Matrix (Fin 2) (Fin 2) ℝ) (area_T : ℝ) (det_A : ℝ) :
  A = ![![3, 1], ![5, 4]] →
  area_T = 6 →
  det_A = Matrix.det A →
  area_T * det_A = 42 :=
by
  intros hA h_area h_det
  rw [hA, h_area, h_det]
  simp [Matrix.det, Finset.sum_univ_succ, Matrix.det_fin_two]
  linarith

end area_transformation_l781_781814


namespace triangle_inequality_problem_l781_781047

noncomputable def RightAngleTriangle (a b c : ℝ) :=
  a^2 + b^2 = c^2

noncomputable def CircleTangent_r (r a b c : ℝ) :=
  r = a * b / (a + c)

noncomputable def CircleTangent_t (t a b c : ℝ) :=
  t = a * b / (b + c)

theorem triangle_inequality_problem (a b c r t : ℝ) 
  (h_right : RightAngleTriangle a b c)
  (h_r : CircleTangent_r r a b c)
  (h_t : CircleTangent_t t a b c) :
  1 / r + 1 / t ≥ (Real.sqrt 2 + 1) * (1 / a + 1 / b) :=
sorry

end triangle_inequality_problem_l781_781047


namespace initial_oranges_in_box_l781_781408

theorem initial_oranges_in_box (X : ℝ) :
  (∃ (initial_oranges : ℝ), initial_oranges + 35.0 = 90) → X = 55.0 :=
by
  intro h
  cases h with initial_oranges h_cond
  sorry

end initial_oranges_in_box_l781_781408


namespace rational_sum_of_squares_l781_781365

/-- 
There exist infinitely many coprime integer solutions to the equation a^2 + b^2 = c^2.
-/
theorem rational_sum_of_squares :
  ∃ f : ℕ → ℚ × ℚ, (∀ n, (f n).1^2 + (f n).2^2 = 169) ∧ function.injective f := 
sorry

end rational_sum_of_squares_l781_781365


namespace circumference_of_tangent_circle_l781_781273

theorem circumference_of_tangent_circle (r : ℝ) (c : ℝ) (h1 : r * (real.pi / 2) = 15) :
  c = 60 - 30 * real.sqrt 2 :=
by
  let r := 30 / real.pi
  sorry

end circumference_of_tangent_circle_l781_781273


namespace complex_division_l781_781713

-- Define the complex numbers in Lean
def i : ℂ := Complex.I

-- Claim to be proved
theorem complex_division :
  (1 + i) / (3 - i) = (1 + 2 * i) / 5 :=
by
  sorry

end complex_division_l781_781713


namespace locus_of_orthocenters_is_ellipse_l781_781172

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

noncomputable def chord (a b x1 x2 : ℝ) : set (ℝ × ℝ) :=
  let y0 := sqrt (b^2 - ((x1^2 + x2^2) / (2 * a^2)))
  { p | p.2 = y0 ∧ (p.1 = x1 ∨ p.1 = x2)}

noncomputable def geometric_locus_of_orthocenters (a b x1 x2 : ℝ) : set (ℝ × ℝ) :=
  let y0 := sqrt (b^2 - ((x1^2 + x2^2) / (2 * a^2)))
  { p | exists mid_orthocenter, mid_orthocenter = (0.5 * (x1 + x2), y0) ∧
            exists orthocenter, orthocenter.1 = mid_orthocenter.1 ∧ orthocenter ∈ ellipse (a * 0.5) (b * 0.5)}

theorem locus_of_orthocenters_is_ellipse (a b x1 x2 : ℝ) :
  ∃ E : set (ℝ × ℝ), E = ellipse (a * 0.5) (b * 0.5) ∧ E = geometric_locus_of_orthocenters a b x1 x2 :=
sorry

end locus_of_orthocenters_is_ellipse_l781_781172


namespace locus_of_Q_is_AX_l781_781105

-- Define the fixed and moving points as hypotheses
variables {A B C D P Q H X : Point}
variables {locus : Set Point}

-- Given conditions
axiom is_triangle : Triangle A B C
axiom D_on_BC : D ∈ segment B C
axiom P_on_AD : P ∈ line_through A D
axiom PH_perp_BC : ∃ H, PH ⊥ BC ∧ H ∈ line_segment B C
axiom Q_on_PH : Q ∈ line_segment P H
axiom feet_perpendiculars_collinear : 
  ∃ M N, foot_of_perpendicular Q A B M ∧ foot_of_perpendicular Q A C N ∧ collinear P M N

-- The target statement to prove in Lean 4
theorem locus_of_Q_is_AX : locus = { Q | ∃ t, Q = A + t * (X - A) } :=
sorry

end locus_of_Q_is_AX_l781_781105


namespace list_price_of_article_l781_781389

theorem list_price_of_article (P : ℝ) (h : 0.882 * P = 57.33) : P = 65 :=
by
  sorry

end list_price_of_article_l781_781389


namespace total_weight_cashew_nuts_and_peanuts_l781_781075

theorem total_weight_cashew_nuts_and_peanuts (weight_cashew_nuts weight_peanuts : ℕ) (h1 : weight_cashew_nuts = 3) (h2 : weight_peanuts = 2) : 
  weight_cashew_nuts + weight_peanuts = 5 := 
by
  sorry

end total_weight_cashew_nuts_and_peanuts_l781_781075


namespace max_sin_x_value_l781_781824

theorem max_sin_x_value (x y z : ℝ) (h1 : Real.sin x = Real.cos y) (h2 : Real.sin y = Real.cos z) (h3 : Real.sin z = Real.cos x) : Real.sin x ≤ Real.sqrt 2 / 2 :=
by
  sorry

end max_sin_x_value_l781_781824


namespace area_of_quadrilateral_l781_781969

/-- A triangle divided into three smaller triangles and a quadrilateral by two lines from vertices
to their opposite sides with the following areas: 3, 7, 7 respectively. Prove that the area of the quadrilateral is 18. -/
theorem area_of_quadrilateral
  (area_triangle_efa : ℝ)
  (area_triangle_fab : ℝ)
  (area_triangle_fbd : ℝ)
  (total_area : ℝ) :
  area_triangle_efa = 3 →
  area_triangle_fab = 7 →
  area_triangle_fbd = 7 →
  total_area = area_triangle_efa + area_triangle_fab + area_triangle_fbd + 18 :=
begin
  intros h1 h2 h3,
  -- proof will go here
  sorry
end

end area_of_quadrilateral_l781_781969


namespace exists_coloring_for_n_gon_l781_781067

theorem exists_coloring_for_n_gon (n : ℕ) (h : n % 2 = 1) :
  ∃ (coloring : (fin n) → fin n → fin n), 
    ∀ (c1 c2 c3 : fin n), c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 → 
    ∃ (i j k : fin n), 
      (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
      (coloring i j = c1 ∧ coloring j k = c2 ∧ coloring k i = c3) := 
sorry

end exists_coloring_for_n_gon_l781_781067


namespace vector_projection_l781_781931

def vector3 : ℝ × ℝ := (3, -2)
def vector2 : ℝ × ℝ := (2, 5)

-- The direction vector from vector3 to vector2
def direction : ℝ × ℝ := (vector2.1 - vector3.1, vector2.2 - vector3.2)

-- The orthogonality condition 
def orthogonal (q : ℝ × ℝ) : Prop :=
  q.1 * direction.1 + q.2 * direction.2 = 0

-- The parameterized line through vector3 in the direction of direction
def line_parameterized (t : ℝ) : ℝ × ℝ :=
  (vector3.1 + t * direction.1, vector3.2 + t * direction.2)

-- Existence of t such that orthogonal line_parameterized t
noncomputable def t : ℝ := 17 / 50
def q : ℝ × ℝ := line_parameterized t

theorem vector_projection :
  ∃ (t : ℝ), orthogonal (line_parameterized t) ∧ q = (133 / 50, 69 / 50) :=
by {
  use t,
  split,
  sorry, -- Proof of orthogonality
  sorry  -- Proof of q equality
}

end vector_projection_l781_781931


namespace f_analytic_expression_f_monotonic_increasing_interval_find_a_of_max_value_l781_781181

/-- Given conditions -/
variable (x : ℝ)
variable (a : ℝ)
variable (h_nonzero : a ≠ 0)
variable (OA : ℝ × ℝ := (a * cos x ^ 2, 1))
variable (OB : ℝ × ℝ := (2, sqrt 3 * a * sin (2 * x - a)))

/-- Definition of function f -/
noncomputable def f (x : ℝ) : ℝ := 
  (OA.1 * OB.1) + (OA.2 * OB.2)

/-- Theorem Statements -/
theorem f_analytic_expression : 
  f x = 2 * a * sin (2 * x + π / 6) :=
begin
  sorry
end

theorem f_monotonic_increasing_interval (k : ℤ) (h_pos : a > 0) : 
  ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 
    (Real.derivative f) x > 0 :=
begin
  sorry
end

theorem find_a_of_max_value (hx : x ∈ Set.Icc 0 (π / 2)) (h_max : f x = 5) :
  a = 5 / 2 :=
begin
  sorry
end

end f_analytic_expression_f_monotonic_increasing_interval_find_a_of_max_value_l781_781181


namespace solve_system_I_solve_system_II_l781_781865

theorem solve_system_I (x y : ℝ) (h1 : y = x + 3) (h2 : x - 2 * y + 12 = 0) : x = 6 ∧ y = 9 :=
by
  sorry

theorem solve_system_II (x y : ℝ) (h1 : 4 * (x - y - 1) = 3 * (1 - y) - 2) (h2 : x / 2 + y / 3 = 2) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_system_I_solve_system_II_l781_781865


namespace car_value_reduction_l781_781804

/-- Jocelyn bought a car 3 years ago at $4000. 
If the car's value has reduced by 30%, calculate the current value of the car. 
Prove that it is equal to $2800. -/
theorem car_value_reduction (initial_value : ℝ) (reduction_percentage : ℝ) (current_value : ℝ) 
  (h_initial : initial_value = 4000)
  (h_reduction : reduction_percentage = 30)
  (h_current : current_value = initial_value - (reduction_percentage / 100) * initial_value) :
  current_value = 2800 :=
by
  -- Formal proof goes here
  sorry

end car_value_reduction_l781_781804


namespace triangle_AC_length_l781_781797

open Real

theorem triangle_AC_length (A B C : Point) 
  (hA : ∠A = π / 3) 
  (hAB : dist A B = 2) 
  (areaABC : area (triangle A B C) = sqrt 3 / 2) : 
  dist A C = 1 := 
by
  sorry

end triangle_AC_length_l781_781797


namespace smallest_positive_integer_divisible_by_8_11_15_l781_781151

-- Define what it means for a number to be divisible by another
def divisible_by (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

-- Define a function to find the least common multiple of three numbers
noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Statement of the theorem
theorem smallest_positive_integer_divisible_by_8_11_15 : 
  ∀ n : ℕ, (n > 0) ∧ divisible_by n 8 ∧ divisible_by n 11 ∧ divisible_by n 15 ↔ n = 1320 :=
sorry -- Proof is omitted

end smallest_positive_integer_divisible_by_8_11_15_l781_781151


namespace tetrahedron_isogonal_iff_medians_circumcenter_coincide_l781_781357

variable (A B C D O : Point)
variable [MetricSpace Point]
variable [HilbertSpace Point]

def tetrahedron_faces_congruent (A B C D : Point) : Prop :=
  let △ABC := triangle A B C
  let △ABD := triangle A B D
  let △ACD := triangle A C D
  let △BCD := triangle B C D
  congruent △ABC △ABD ∧ congruent △ABC △ACD ∧ congruent △ABC △BCD

def medians_intersect_circumcenter (A B C D O : Point) : Prop :=
  let circumcenter := circumcenter_tetrahedron A B C D
  let medians := medians_intersection_tetrahedron A B C D
  O = circumcenter ∧ O = medians

theorem tetrahedron_isogonal_iff_medians_circumcenter_coincide
  (A B C D O : Point) [MetricSpace Point] [HilbertSpace Point] :
  tetrahedron_faces_congruent A B C D ↔ medians_intersect_circumcenter A B C D O :=
sorry

end tetrahedron_isogonal_iff_medians_circumcenter_coincide_l781_781357


namespace circumradius_geq_three_times_inradius_l781_781979

def Tetrahedron (A1 A2 A3 A4 : Point) := Σ R r, (Circumradius A1 A2 A3 A4 = R) ∧ (Inradius A1 A2 A3 A4 = r)

theorem circumradius_geq_three_times_inradius
  {A1 A2 A3 A4 : Point}
  (R r : ℝ)
  (hR : Circumradius A1 A2 A3 A4 = R)
  (hr : Inradius A1 A2 A3 A4 = r) :
  R ≥ 3 * r := by
  sorry

end circumradius_geq_three_times_inradius_l781_781979


namespace sum_of_digits_base2_222_l781_781426

theorem sum_of_digits_base2_222 : ∑ d in Nat.digits 2 222, d = 6 := by
  sorry

end sum_of_digits_base2_222_l781_781426


namespace order_of_6_proof_l781_781871

noncomputable def f (x : ℕ) := x^2 % 13

def order_of_6 : ℕ := 36

theorem order_of_6_proof : (∃ n, n > 0 ∧ f^[n] 6 = 6 ∧ (∀ m < n, m > 0 → f^[m] 6 ≠ 6)) ∧ (order_of_6 = 36) :=
begin
  use 36,
  split,
  { split,
    { norm_num, },
    { split,
      { norm_num, },
      { intros m hm1 hm2,
        sorry, } } },
  norm_num,
end

end order_of_6_proof_l781_781871


namespace subset_A_B_l781_781837

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem subset_A_B : A ⊆ B := sorry

end subset_A_B_l781_781837


namespace fraction_without_cable_or_vcr_l781_781295

theorem fraction_without_cable_or_vcr (T : ℕ) (h1 : ℚ) (h2 : ℚ) (h3 : ℚ) 
  (h1 : h1 = 1 / 5 * T) 
  (h2 : h2 = 1 / 10 * T) 
  (h3 : h3 = 1 / 3 * (1 / 5 * T)) 
: (T - (1 / 5 * T + 1 / 10 * T - 1 / 3 * (1 / 5 * T))) / T = 23 / 30 := 
by 
  sorry

end fraction_without_cable_or_vcr_l781_781295


namespace seating_arrangements_l781_781594

/-
Given:
1. There are 8 students.
2. Four different classes: (1), (2), (3), and (4).
3. Each class has 2 students.
4. There are 2 cars, Car A and Car B, each with a capacity for 4 students.
5. The two students from Class (1) (twin sisters) must ride in the same car.

Prove:
The total number of ways to seat the students such that exactly 2 students from the same class are in Car A is 24.
-/

theorem seating_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 24 :=
sorry

end seating_arrangements_l781_781594


namespace length_of_AB_l781_781073

noncomputable def parabola_intersection (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
|x1 - x2|

theorem length_of_AB : 
  ∀ (x1 x2 y1 y2 : ℝ),
    (x1 + x2 = 6) →
    (A = (x1, y1)) →
    (B = (x2, y2)) →
    (y1^2 = 4 * x1) →
    (y2^2 = 4 * x2) →
    parabola_intersection x1 x2 y1 y2 = 8 :=
by
  sorry

end length_of_AB_l781_781073


namespace range_of_a_l781_781723

-- Define the function f(x) and its condition
def f (x a : ℝ) : ℝ := x^2 + (a + 2) * x + (a - 1)

-- Given condition: f(-1, a) = -2
def condition (a : ℝ) : Prop := f (-1) a = -2

-- Requirement for the domain of g(x) = ln(f(x) + 3) being ℝ
def domain_requirement (a : ℝ) : Prop := ∀ x : ℝ, f x a + 3 > 0

-- Main theorem to prove the range of a
theorem range_of_a : {a : ℝ // condition a ∧ domain_requirement a} = {a : ℝ // -2 < a ∧ a < 2} :=
by sorry

end range_of_a_l781_781723


namespace lockers_number_l781_781557

theorem lockers_number (total_cost : ℝ) (cost_per_digit : ℝ) (total_lockers : ℕ) 
  (locker_numbered_from_one : ∀ n : ℕ, n >= 1) :
  total_cost = 248.43 → cost_per_digit = 0.03 → total_lockers = 2347 :=
by
  intros h_total_cost h_cost_per_digit
  sorry

end lockers_number_l781_781557


namespace order_of_6_with_respect_to_f_is_undefined_l781_781873

noncomputable def f (x : ℕ) : ℕ := x ^ 2 % 13

def order_of_6_undefined : Prop :=
  ∀ m : ℕ, m > 0 → f^[m] 6 ≠ 6

theorem order_of_6_with_respect_to_f_is_undefined : order_of_6_undefined :=
by
  sorry

end order_of_6_with_respect_to_f_is_undefined_l781_781873


namespace sequence_problems_l781_781148

def first_sequence : ℕ → ℕ
| 0 := 14
| 1 := 17
| 2 := 20
| 3 := 23
| 4 := 26
| (n + 5) := (first_sequence n) + 3

def second_sequence : ℕ → ℕ
| 0 := 2
| 1 := 4
| 2 := 8
| 3 := 16
| 4 := 32
| (n + 5) := (second_sequence n) * 2

def third_sequence : ℕ → ℕ
| 0 := 2
| 1 := 3
| 2 := 5
| 3 := 8
| 4 := 13
| 5 := 21
| (n + 6) := (third_sequence (n + 4)) + (third_sequence (n + 5))

def fourth_sequence : ℕ → ℕ
| 1 := 1
| 2 := 4
| 3 := 9
| 4 := 16
| 5 := 25
| 7 := 49
| (n + 6) := (n + 6) * (n + 6)

theorem sequence_problems :
  (first_sequence 5 = 29 ∧ first_sequence 6 = 32) ∧
  (second_sequence 5 = 64 ∧ second_sequence 6 = 128) ∧
  (third_sequence 6 = 34) ∧
  (fourth_sequence 6 = 36) :=
by
  sorry

end sequence_problems_l781_781148


namespace locus_of_C_l781_781697

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)

theorem locus_of_C :
  ∀ (C : ℝ × ℝ), (C.2 = (b / a) * C.1 ∧ (a * b / Real.sqrt (a ^ 2 + b ^ 2) ≤ C.1) ∧ (C.1 ≤ a)) :=
sorry

end locus_of_C_l781_781697


namespace average_scissors_correct_l781_781794

-- Definitions for the initial number of scissors in each drawer
def initial_scissors_first_drawer : ℕ := 39
def initial_scissors_second_drawer : ℕ := 27
def initial_scissors_third_drawer : ℕ := 45

-- Definitions for the new scissors added by Dan
def added_scissors_first_drawer : ℕ := 13
def added_scissors_second_drawer : ℕ := 7
def added_scissors_third_drawer : ℕ := 10

-- Calculate the final number of scissors after Dan's addition
def final_scissors_first_drawer : ℕ := initial_scissors_first_drawer + added_scissors_first_drawer
def final_scissors_second_drawer : ℕ := initial_scissors_second_drawer + added_scissors_second_drawer
def final_scissors_third_drawer : ℕ := initial_scissors_third_drawer + added_scissors_third_drawer

-- Statement to prove the average number of scissors in all three drawers
theorem average_scissors_correct :
  (final_scissors_first_drawer + final_scissors_second_drawer + final_scissors_third_drawer) / 3 = 47 := by
  sorry

end average_scissors_correct_l781_781794


namespace ratio_blue_red_l781_781019

def diameter_small : ℝ := 2
def diameter_large : ℝ := 4
def side_square : ℝ := 1

def radius_small : ℝ := diameter_small / 2
def radius_large : ℝ := diameter_large / 2

def area_circle (r : ℝ) : ℝ := π * r^2
def area_square (s : ℝ) : ℝ := s^2

def area_small := area_circle radius_small
def area_large := area_circle radius_large
def area_square_inside := area_square side_square

def area_red := area_small - area_square_inside
def area_blue := area_large - area_small

def ratio_area_blue_red := area_blue / area_red

theorem ratio_blue_red : ratio_area_blue_red = 3 * π / (π - 1) := by
  -- the proof goes here
  sorry

end ratio_blue_red_l781_781019


namespace degree_measure_of_angle_5_l781_781839

noncomputable def angle1_measure (x : ℝ) : Prop :=
  x = 36

noncomputable def measure_of_angle1 (a2 : ℝ) :=
  ∃ x : ℝ, x = a2 / 4

theorem degree_measure_of_angle_5 (p q : Prop) (a2 a5 : ℝ) (h_parallel : p = q) (h_angle1 : measure_of_angle1 a2) : angle1_measure a5 :=
by
  rcases h_angle1 with ⟨x, hx⟩
  have h₁ : 4 * x + x = 180 := sorry -- since angles on a straight line sum to 180 degrees
  have h₂ : 5 * x = 180 := by linarith
  have h₃ : x = 36 := by linarith
  exact h₃

end degree_measure_of_angle_5_l781_781839


namespace balls_in_boxes_l781_781229

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l781_781229


namespace smallest_value_of_abs_z_l781_781322

noncomputable def smallest_possible_value_of_z (z : ℂ) : ℝ :=
  if |z - 9| + |z - 6 * complex.i| = 15 then 3.6 else 0 -- default to 0 if condition isn't met

theorem smallest_value_of_abs_z (z : ℂ) (h : |z - 9| + |z - 6 * complex.i| = 15) :
  smallest_possible_value_of_z z = 3.6 := 
by
  -- Proof goes here
  sorry

end smallest_value_of_abs_z_l781_781322


namespace derivative_of_f_l781_781143

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem derivative_of_f (x : ℝ) : derivative f x = x * Real.cos x :=
by
  sorry

end derivative_of_f_l781_781143


namespace number_of_valid_sets_l781_781735

open Set

noncomputable def universal_set : Set ℤ := {x | -4 ≤ x ∧ x ≤ 4}
def core_set : Set ℤ := {-1, 1, 3}

def valid_sets (P : Set ℤ) : Prop := compl P ⊆ core_set

theorem number_of_valid_sets : 
    (∃! n : ℕ, n = 8 ∧ 
    ∀ P : Set ℤ, (compl P ⊆ core_set) ↔ (compl P).finite ∧ 
    compl P = {x | ((x ∉ universal_set) ∨ (x ∈ core_set)) ∧ P ⊆ universal_set }) :=
sorry

end number_of_valid_sets_l781_781735


namespace correct_choice_l781_781444

-- Definitions based on conditions
def options := ["both of them", "either of them", "none of them", "neither of them"]
def statement := "Actually, I didn’t like"

-- Theorem statement in Lean
theorem correct_choice : (statement ++ " either of them") ∈ [statement ++ " " ++ option | option ∈ options] :=
by sorry

end correct_choice_l781_781444


namespace sequence_a_b_10_l781_781346

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end sequence_a_b_10_l781_781346


namespace tan_neg_two_simplifies_l781_781760

theorem tan_neg_two_simplifies :
  ∀ θ : Real, tan θ = -2 → (sin θ * (1 + sin (2 * θ))) / (sin θ + cos θ) = 2 / 5 := by
  intro θ h
  sorry

end tan_neg_two_simplifies_l781_781760


namespace find_triangle_C_coordinates_find_triangle_area_l781_781185

noncomputable def triangle_C_coordinates (A B : (ℝ × ℝ)) (median_eq altitude_eq : (ℝ × ℝ × ℝ)) : Prop :=
  ∃ C : ℝ × ℝ, C = (3, 1) ∧
    let A := (1,2)
    let B := (3, 4)
    let median_eq := (2, 1, -7)
    let altitude_eq := (2, -1, -2)
    true

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : Prop :=
  ∃ S : ℝ, S = 3 ∧
    let A := (1,2)
    let B := (3, 4)
    let C := (3, 1)
    true

theorem find_triangle_C_coordinates : triangle_C_coordinates (1,2) (3,4) (2, 1, -7) (2, -1, -2) :=
by { sorry }

theorem find_triangle_area : triangle_area (1,2) (3,4) (3,1) :=
by { sorry }

end find_triangle_C_coordinates_find_triangle_area_l781_781185


namespace reduced_price_per_kg_of_oil_l781_781043

theorem reduced_price_per_kg_of_oil
  (P : ℝ)
  (h : (1000 / (0.75 * P) - 1000 / P = 5)) :
  0.75 * (1000 / 15) = 50 := 
sorry

end reduced_price_per_kg_of_oil_l781_781043


namespace max_value_of_a_l781_781727

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  exp (2 * x) - exp (-2 * x) - 4 * x - a * exp x + a * exp (-x) + 2 * a * x

theorem max_value_of_a : ∀ x > 0, f a x ≥ 0 ↔ a ≤ 8 :=
begin
  sorry
end

end max_value_of_a_l781_781727


namespace combined_age_in_ten_years_l781_781286

theorem combined_age_in_ten_years (B A: ℕ) (hA : A = 20) (h1: A + 10 = 2 * (B + 10)): 
  (A + 10) + (B + 10) = 45 := 
by
  sorry

end combined_age_in_ten_years_l781_781286


namespace ball_distribution_l781_781242

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l781_781242


namespace fourth_smallest_9867_l781_781930

def nums : List ℕ := [9867, 8976, 9876, 9687, 9689]

theorem fourth_smallest_9867 : (nums.sorted.get? 3).getD 0 = 9867 := by
  sorry

end fourth_smallest_9867_l781_781930


namespace hunter_ants_l781_781363

variable (spiders : ℕ) (ladybugs_before : ℕ) (ladybugs_flew : ℕ) (total_insects : ℕ)

theorem hunter_ants (h1 : spiders = 3)
                    (h2 : ladybugs_before = 8)
                    (h3 : ladybugs_flew = 2)
                    (h4 : total_insects = 21) :
  ∃ ants : ℕ, ants = total_insects - (spiders + (ladybugs_before - ladybugs_flew)) ∧ ants = 12 :=
by
  sorry

end hunter_ants_l781_781363


namespace tangent_line_at_pi_one_l781_781888

noncomputable def function (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1
noncomputable def tangent_line (x : ℝ) (y : ℝ) : ℝ := x * Real.exp Real.pi + y - 1 - Real.pi * Real.exp Real.pi

theorem tangent_line_at_pi_one :
  tangent_line x y = 0 ↔ y = function x → x = Real.pi ∧ y = 1 :=
by
  sorry

end tangent_line_at_pi_one_l781_781888


namespace exists_zero_f_l781_781668

def f (x : ℝ) : ℝ := 2^x + x

theorem exists_zero_f : ∃ x ∈ Ioo (-1 : ℝ) (0 : ℝ), f x = 0 := by
  sorry

end exists_zero_f_l781_781668


namespace average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781545

noncomputable def data :=
{ x : list ℝ // x.length = 10 ∧ (x.map (λ i, i * i)).sum = 0.038 ∧ x.sum = 0.6 },
{ y : list ℝ // y.length = 10 ∧ (y.map (λ i, i * i)).sum = 1.6158 ∧ y.sum = 3.9 },
{ xy_prod : list ℝ // (x.zip_with (*) y).sum = 0.2474 }

theorem average_root_cross_sectional_area_and_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hlen : list.length x = 10 ∧ list.length y = 10) :
  x.sum / 10 = 0.06 ∧ y.sum / 10 = 0.39 :=
by sorry

theorem sample_correlation_coefficient
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (hx2 : (x.map (λ i, i * i)).sum = 0.038)
  (hy2 : (y.map (λ i, i * i)).sum = 1.6158)
  (hxy : (x.zip_with (*) y).sum = 0.2474) : 
  ((0.2474 - (0.6 * 3.9 / 10)) / (sqrt ((0.038 - (0.6^2 / 10)) * (1.6158 - (3.9^2 / 10))))) ≈ 0.97 :=
by sorry

theorem estimate_total_volume
  (x y : list ℝ)
  (hx : x.sum = 0.6) (hy : y.sum = 3.9)
  (total_root_area : ℝ := 186) :
  (0.39 / 0.06) * total_root_area = 1209 :=
by sorry

end average_root_cross_sectional_area_and_volume_sample_correlation_coefficient_estimate_total_volume_l781_781545


namespace triangle_area_correct_l781_781025

open Real

def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)

theorem triangle_area_correct :
  triangle_area (4, 6) (-4, 6) (0, 2) = 16 :=
by
  sorry

end triangle_area_correct_l781_781025


namespace reynald_soccer_balls_l781_781855

theorem reynald_soccer_balls (total_balls basketballs_more soccer tennis baseball more_baseballs volleyballs : ℕ) 
(h_total_balls: total_balls = 145) 
(h_basketballs_more: basketballs_more = 5)
(h_tennis: tennis = 2 * soccer)
(h_more_baseballs: more_baseballs = 10)
(h_volleyballs: volleyballs = 30) 
(sum_eq: soccer + (soccer + basketballs_more) + tennis + (soccer + more_baseballs) + volleyballs = total_balls) : soccer = 20 := 
by
  sorry

end reynald_soccer_balls_l781_781855


namespace floor_sqrt_fifty_l781_781620

theorem floor_sqrt_fifty : int.floor (real.sqrt 50) = 7 := sorry

end floor_sqrt_fifty_l781_781620


namespace triangle_inequality_11_side_l781_781906

def valid_triangle_count : ℕ :=
  let pairs : Finset (ℕ × ℕ) := (Finset.Icc 1 10).product (Finset.Icc 1 10)
  pairs.filter (λ (ab : ℕ × ℕ), ab.1 + ab.2 > 11 ∧ ab.1 ≤ ab.2).card

theorem triangle_inequality_11_side :
  valid_triangle_count = 36 :=
by
  sorry

end triangle_inequality_11_side_l781_781906


namespace equivalent_discount_l781_781957

variable (P d1 d2 d : ℝ)

-- Given conditions:
def original_price : ℝ := 50
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.10
def equivalent_single_discount_rate : ℝ := 0.325

-- Final conclusion:
theorem equivalent_discount :
  let final_price_after_first_discount := (original_price * (1 - first_discount_rate))
  let final_price_after_second_discount := (final_price_after_first_discount * (1 - second_discount_rate))
  final_price_after_second_discount = (original_price * (1 - equivalent_single_discount_rate)) :=
by
  sorry

end equivalent_discount_l781_781957


namespace proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781538

-- Define the sample data as lists
def x : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

-- Given conditions
axiom sum_x_squared : ∑ x_i in x, x_i^2 = 0.038
axiom sum_y_squared : ∑ y_i in y, y_i^2 = 1.6158
axiom sum_x_y : ∑ i in List.range 10, x[i] * y[i] = 0.2474

-- Additional conditions
def X : ℝ := 186
noncomputable def sqrt_1_896 : ℝ := 1.377

-- Average root cross-sectional area
def avg_x : ℝ := 0.06
-- Average volume
def avg_y : ℝ := 0.39
-- Sample correlation coefficient
def r : ℝ := 0.97
-- Estimated total volume
def Y : ℝ := 1209

-- Tasks to prove
theorem proof_avg_x : (∑ x_i in x, x_i) / 10 = avg_x := sorry
theorem proof_avg_y : (∑ y_i in y, y_i) / 10 = avg_y := sorry
theorem proof_r : (0.2474 - 10 * avg_x * avg_y) / (sqrt ((10 * 0.06^2 - (∑ x_i in x, x_i)^2 / 10) * (10 * 0.39^2 - (∑ y_i in y, y_i)^2 / 10))) = r := sorry
theorem proof_Y : (avg_y / avg_x) * X = Y := sorry

end proof_avg_x_proof_avg_y_proof_r_proof_Y_l781_781538


namespace max_distance_MN_l781_781733

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x)
noncomputable def MN (a : ℝ) : ℝ := |f a - g a|

-- Prove that the maximum value of |MN| is 3
theorem max_distance_MN (a : ℝ) : MN(a) ≤ 3 := sorry

end max_distance_MN_l781_781733


namespace solomon_took_pieces_l781_781907

theorem solomon_took_pieces (initial_pieces talitha_pieces remaining_pieces : ℕ) 
  (h_init : initial_pieces = 349) 
  (h_talitha : talitha_pieces = 108) 
  (h_remaining : remaining_pieces = 88) : 
  initial_pieces - talitha_pieces - remaining_pieces = 153 :=
by
  rw [h_init, h_talitha, h_remaining]
  norm_num
  sorry

end solomon_took_pieces_l781_781907


namespace translated_function_correct_l781_781014

def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 2

def translate_left (g : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := g(x + a)

def translate_up (g : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := g x + b

theorem translated_function_correct :
  ∀ x : ℝ, translate_up (translate_left f 1) 1 x = (x - 1) ^ 2 + 3 := 
by
  sorry

end translated_function_correct_l781_781014


namespace intersect_circle_line_find_m_l781_781165

open Real

def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

theorem intersect_circle_line (m : ℝ) :
  let r := sqrt 5
  let d := abs m / sqrt (m^2 + 1)
  d <= r :=
sorry

theorem find_m (m : ℝ) :
  let r := sqrt 5
  let d := abs m / sqrt (m^2 + 1)
  |AB| = sqrt 17 →
  (|AB| / 2) = sqrt (r^2 - d^2) → 
  m^2 = 3 ∨ m = sqrt 3 ∨ m = -sqrt 3 :=
sorry

end intersect_circle_line_find_m_l781_781165


namespace hexagon_side_sum_squares_l781_781940
noncomputable theory

open_locale classical

-- Define the problem setup in Lean
variables {O A B C A' B' C' : Type*} [metric_space O] [metric_space A] [metric_space B] [metric_space C] [metric_space A'] [metric_space B'] [metric_space C']

-- Assume reflections
variables (reflection_A : A' = sym_point O A)
variables (reflection_B : B' = sym_point O B)
variables (reflection_C : C' = sym_point O C)

-- Assume distances from O to sides
variables (DO EO FO : ℝ) -- distances from O to sides AB, BC, AC

-- Define the goal
theorem hexagon_side_sum_squares 
  (h₁ : dist O A = DO)
  (h₂ : dist O B = EO)
  (h₃ : dist O C = FO) :
  (dist A B')^2 + (dist B' C)^2 + (dist C A')^2 + (dist A' B)^2 + (dist B C')^2 + (dist C' A)^2 = 
  8 * (DO^2 + EO^2 + FO^2) := by sorry

end hexagon_side_sum_squares_l781_781940


namespace difference_of_final_shares_l781_781554

theorem difference_of_final_shares (Vasim_share_before_tax : ℝ) (Faruk_ratio Vasim_ratio Ranjith_ratio : ℝ)
  (Faruk_tax Vasim_tax Ranjith_tax : ℝ) (val : ℝ) :
  Vasim_share_before_tax = 1500 →
  Faruk_ratio = 3 →
  Vasim_ratio = 5 →
  Ranjith_ratio = 8 →
  Faruk_tax = 0.10 →
  Vasim_tax = 0.15 →
  Ranjith_tax = 0.12 →
  val = Vasim_share_before_tax / Vasim_ratio →
  let Faruk_share_before_tax := Faruk_ratio * val,
      Ranjith_share_before_tax := Ranjith_ratio * val,
      Faruk_final_share := Faruk_share_before_tax * (1 - Faruk_tax),
      Ranjith_final_share := Ranjith_share_before_tax * (1 - Ranjith_tax) in
      Ranjith_final_share - Faruk_final_share = 1302 := by
  sorry

end difference_of_final_shares_l781_781554


namespace total_money_together_l781_781564

-- Define the conditions
def Sam_has := 75

def Billy_has (Sam_has : Nat) := 2 * Sam_has - 25

-- Define the total money calculation
def total_money (Sam_has : Nat) (Billy_has : Nat) := Sam_has + Billy_has Sam_has

-- Define the theorem to prove the equivalent problem
theorem total_money_together : total_money Sam_has (Billy_has Sam_has) = 200 :=
by
  sorry

end total_money_together_l781_781564


namespace average_first_21_multiples_of_6_l781_781027

-- Define the arithmetic sequence and its conditions.
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

-- Define the problem statement.
theorem average_first_21_multiples_of_6 :
  let a1 := 6
  let d := 6
  let n := 21
  let an := arithmetic_sequence a1 d n
  (a1 + an) / 2 = 66 := by
  sorry

end average_first_21_multiples_of_6_l781_781027


namespace dodecagon_area_ratio_l781_781285

-- Assume the existence of a regular dodecagon and its vertices
variable (A B C D E F G H I J K L : Point)

-- Define P as the midpoint of segment BC
def P : Point := midpoint B C

-- Define Q as the midpoint of segment HI
def Q : Point := midpoint H I

-- Define the areas of the polygons in question
def area_ABDP : ℝ := area (polygon.mk [A, B, D, P])
def area_FGHJP : ℝ := area (polygon.mk [F, G, H, J, P])

-- Define the ratio of the two areas
def ratio : ℝ := area_ABDP / area_FGHJP

-- The theorem statement
theorem dodecagon_area_ratio :
  ratio = 3 / 5 := by
  sorry

end dodecagon_area_ratio_l781_781285


namespace complex_div_eq_i_l781_781687

noncomputable def i := Complex.I

theorem complex_div_eq_i : (1 + i) / (1 - i) = i := 
by
  sorry

end complex_div_eq_i_l781_781687


namespace tangents_perpendicular_l781_781833

theorem tangents_perpendicular
  (A B C H F : Type*)
  [geometry A B C H F]
  (is_orthocenter : orthocenter A B C H)
  (is_intersection : intersects (circle (diameter A H)) (circle (diameter B C)) F) :
  perpendicular (tangent (circle (diameter A H)) F) (tangent (circle (diameter B C)) F) :=
begin
  sorry
end

end tangents_perpendicular_l781_781833


namespace trigonometric_expression_evaluation_l781_781753

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l781_781753


namespace money_together_l781_781560

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l781_781560


namespace count_three_digit_congruent_to_5_mod_7_l781_781739

theorem count_three_digit_congruent_to_5_mod_7 : 
  (100 ≤ 7 * k + 5 ∧ 7 * k + 5 ≤ 999) → ∃ n : ℕ, n = 129 := sorry

end count_three_digit_congruent_to_5_mod_7_l781_781739


namespace number_of_ways_to_put_balls_in_boxes_l781_781247

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l781_781247


namespace gluten_free_fraction_l781_781281

-- Define given conditions
variables (v g t : ℕ)
variable (h_v : v = 6)
variable (h_fraction : v = t / 3)
variable (h_g : g = 4)

-- Define what we want to prove
theorem gluten_free_fraction :
  (v - g) = 6 → v = t / 3 → g = 4 → (v - g).toReal / t.toReal = 1 / 9 :=
by
  -- Skip the proof
  intros
  sorry

end gluten_free_fraction_l781_781281


namespace num_of_dogs_l781_781404

theorem num_of_dogs (num_puppies : ℕ) (dog_food_per_meal : ℕ) (dog_meals_per_day : ℕ) (total_food : ℕ)
  (h1 : num_puppies = 4)
  (h2 : dog_food_per_meal = 4)
  (h3 : dog_meals_per_day = 3)
  (h4 : total_food = 108)
  : ∃ (D : ℕ), num_puppies * (dog_food_per_meal / 2) * (dog_meals_per_day * 3) + D * (dog_food_per_meal * dog_meals_per_day) = total_food ∧ D = 3 :=
by
  sorry

end num_of_dogs_l781_781404


namespace solve_equation_l781_781149

theorem solve_equation :
  ∀ (x m n : ℕ), 
    0 < x → 0 < m → 0 < n → 
    x^m = 2^(2 * n + 1) + 2^n + 1 →
    (x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4) :=
by
  sorry

end solve_equation_l781_781149


namespace range_of_m_l781_781836

open Set Real

theorem range_of_m (M N : Set ℝ) (m : ℝ) :
    (M = {x | x ≤ m}) →
    (N = {y | ∃ x : ℝ, y = 2^(-x)}) →
    (M ∩ N ≠ ∅) → m > 0 := by
  intros hM hN hMN
  sorry

end range_of_m_l781_781836


namespace find_m_l781_781717

theorem find_m (m : ℝ) (x : ℝ) (hx : x ≠ 0) (h : ∃ (f : ℝ → ℝ), f = λ x, (m^2 - 3) * x^(2 * m)) : m = 2 ∨ m = -2 :=
by
  sorry

end find_m_l781_781717


namespace range_floor_f_l781_781747

def f (x : ℝ) : ℝ := (2^x) / (1 + 2^x) - 1 / 2

theorem range_floor_f : set.range (λ x : ℝ, Int.floor (f x)) = {0, -1} :=
by
  sorry

end range_floor_f_l781_781747


namespace inequality_am_gm_l781_781445

variable {u v : ℝ}

theorem inequality_am_gm (hu : 0 < u) (hv : 0 < v) : u ^ 3 + v ^ 3 ≥ u ^ 2 * v + v ^ 2 * u := by
  sorry

end inequality_am_gm_l781_781445


namespace balls_in_boxes_l781_781223

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l781_781223


namespace sqrt_difference_square_l781_781684

theorem sqrt_difference_square (a b : ℝ) (h₁ : a = Real.sqrt 3 + Real.sqrt 2) (h₂ : b = Real.sqrt 3 - Real.sqrt 2) : a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end sqrt_difference_square_l781_781684


namespace equilateral_triangle_perimeter_l781_781074

theorem equilateral_triangle_perimeter :
  ∃ (L₁ L₂ L₃ : ℝ → ℝ) (A B C : (ℝ × ℝ)), 
    (L₁ = λ x, \frac{\sqrt{3}}{4} * x) ∧
    (∃ m : ℝ, L₂ = λ x, m * x ∧ L₂ 0 = 0) ∧
    (L₃ = λ _, 2) ∧
    (A = (2, L₁ 2)) ∧ (B = (2, L₂ 2)) ∧
    (C = ((L₂ 2 - L₁ 2) / (L₁ 1 - L₂ 1), 0)) ∧
    (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧
    (∀ P Q R : (ℝ × ℝ), {P, Q, R} = {A, B, C} → 
      dist P Q = dist Q R ∧ dist Q R = dist R P) ∧
    3 * (dist A B) = \frac{51 * \sqrt{3}}{6} := 
begin
  sorry,
end

end equilateral_triangle_perimeter_l781_781074


namespace right_triangle_third_side_square_l781_781698

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) 
  (h₁ : a = 3) (h₂ : b = 4) (h₃ : a^2 + b^2 = c^2) :
  c^2 = 25 ∨ a^2 + c^2 = b^2 ∨ a^2 + b^2 = 7 :=
by
  sorry

end right_triangle_third_side_square_l781_781698


namespace triangle_largest_angle_l781_781715

noncomputable def largest_angle (a b c : ℝ) : ℝ :=
  real.acos ( (a^2 + b^2 - c^2) / (2 * a * b) )

theorem triangle_largest_angle : 
  ∃ (a b c : ℝ), 
    (9 * a = 12 * b ∧ 12 * b = 18 * c) →
    largest_angle (4 : ℝ) (3 : ℝ) (2 : ℝ) = real.acos (-1/4) := 
begin
  sorry
end

end triangle_largest_angle_l781_781715


namespace concylic_points_IFEH_l781_781701

open EuclideanGeometry

/-- Given an acute-angled triangle ABC with side lengths AB and AC as diameters. Let point I lie on the circumcircle of △ABC. 
Draw a line through I parallel to AC intersecting the semicircle at point E. Also, draw a line through I parallel to AB intersecting the semicircle at point F. 
Prove that points I, E, F, and H are concyclic. -/
theorem concylic_points_IFEH
  {A B C I E F H : Point}
  (h_triangle : triangle A B C)
  (h_right_angle_A : right_angle A)
  (h_diameters : is_diameter A B ∧ is_diameter A C)
  (h_I_on_circumcircle : on_circumcircle I A B C)
  (h_parallel_IE_AC : parallel (line_through I E) (line_through A C))
  (h_parallel_IF_AB : parallel (line_through I F) (line_through A B))
  (h_point_H : Point H) :
  concyclic_points I E F H :=
by sorry

end concylic_points_IFEH_l781_781701


namespace find_angle_B_find_c_l781_781798

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
(h1 : ∀ a b c A C, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
     ∠A + ∠B + ∠C = π ∧ 
     sin A * b = sin B * a ∧ 
     sin B * c = sin C * b ∧ 
     cos C = (a^2 + b^2 - c^2) / (2 * a * b) ∧ 
     cos A = (b^2 + c^2 - a^2) / (2 * b * c))
(h2 : a / (b * c) + c / (a * b) - b / (a * c) = 1 / (a * cos C + c * cos A)) :
B = π / 3 := by sorry

theorem find_c 
(a c : ℝ) (h1 : ∀ b R, (R : ℝ) = sqrt 3 ∧ 2 * R * sin (π / 3) = b ∧ b = 3)
(h2 : sqrt 3 * a * c = 6)
(h3 : a^2 + c^2 = 15)
(h4 : c > a) :
c = 2 * sqrt 3 := by sorry

end find_angle_B_find_c_l781_781798


namespace distance_to_big_rock_l781_781440

theorem distance_to_big_rock :
  ∃ (D : ℝ), 
  (∀ (speed_still_water speed_current total_time : ℝ),
    speed_still_water = 7 ∧
    speed_current = 1 ∧
    total_time = 1 →
    (D / (speed_still_water - speed_current) + D / (speed_still_water + speed_current) = total_time) →
    D = 24 / 7) :=
by
  use 24 / 7
  intros speed_still_water speed_current total_time hs hsum
  rw [hs.1, hs.2.1] at hsum
  sorry -- Proof is omitted as requested

end distance_to_big_rock_l781_781440


namespace square_area_parabola_inscribed_l781_781484

theorem square_area_parabola_inscribed (s : ℝ) (x y : ℝ) :
  (y = x^2 - 6 * x + 8) ∧
  (s = -2 + 2 * Real.sqrt 5) ∧
  (x = 3 - s / 2 ∨ x = 3 + s / 2) →
  s ^ 2 = 24 - 8 * Real.sqrt 5 :=
by
  sorry

end square_area_parabola_inscribed_l781_781484


namespace distinct_solution_condition_l781_781658

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l781_781658


namespace range_of_x_l781_781268

theorem range_of_x (x : ℝ) (h : ∃ y : ℝ, y = (x - 3) ∧ y > 0) : x > 3 :=
sorry

end range_of_x_l781_781268


namespace stops_away_pinedale_mall_from_yahya_house_l781_781878

-- Definitions based on problem conditions
def bus_speed_kmh : ℕ := 60
def stop_interval_minutes : ℕ := 5
def distance_to_mall_km : ℕ := 40

-- Definition of how many stops away is Pinedale mall from Yahya's house
def stops_to_mall : ℕ := distance_to_mall_km / (bus_speed_kmh / 60 * stop_interval_minutes)

-- Lean statement to prove the given conditions lead to the correct number of stops
theorem stops_away_pinedale_mall_from_yahya_house :
  stops_to_mall = 8 :=
by 
  -- This is a placeholder for the proof. 
  -- Actual proof steps would convert units and calculate as described in the problem.
  sorry

end stops_away_pinedale_mall_from_yahya_house_l781_781878


namespace cost_of_replaced_tomatoes_l781_781740

def original_order : ℝ := 25
def delivery_tip : ℝ := 8
def new_total : ℝ := 35
def original_tomatoes : ℝ := 0.99
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00

def increase_in_lettuce := new_lettuce - original_lettuce
def increase_in_celery := new_celery - original_celery
def total_increase_except_tomatoes := increase_in_lettuce + increase_in_celery
def original_total_with_delivery := original_order + delivery_tip
def total_increase := new_total - original_total_with_delivery
def increase_due_to_tomatoes := total_increase - total_increase_except_tomatoes
def replaced_tomatoes := original_tomatoes + increase_due_to_tomatoes

theorem cost_of_replaced_tomatoes : replaced_tomatoes = 2.20 := by
  sorry

end cost_of_replaced_tomatoes_l781_781740


namespace sum_of_roots_is_zero_polynomial_p_value_l781_781890

-- Define the polynomial based on the conditions
def polynomial (p q : ℝ) := (x : ℝ) → x^4 + p * x^2 + q * x - 144

-- Define roots in arithmetic progression
def roots_in_arithmetic_progression (a d : ℝ) := [a, a + d, a + 2*d, a + 3*d]

-- Define the Vieta's formula condition for the sum of roots being zero
theorem sum_of_roots_is_zero (a d : ℝ) : (4 * a + 6 * d = 0) → (d = - (2/3) * a) := 
begin
  intro h,
  linarith,
end

-- Given a polynomial with roots in arithmetic progression, prove p = -40
theorem polynomial_p_value (a : ℝ) (h : a ≠ 0) :
  let d := - (2/3) * a in
  let roots := roots_in_arithmetic_progression a d in
  (polynomial (-40) (144 / ((a * (a / 3) * (-(a / 3)) * (-a)))) roots) = polynomial (-40) := 
begin
  sorry  -- Proof to be filled in
end

end sum_of_roots_is_zero_polynomial_p_value_l781_781890


namespace range_of_x_l781_781724

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x ≤ 1 then 3 * (1 - 2^x) / (2^x + 1)
else - 1 / 4 * (x^3 + 3 * x)

theorem range_of_x (x : ℝ) (m : ℝ) (h_m : -3 ≤ m ∧ m ≤ 2) (h_cond : ∀ m ∈ Icc (-3 : ℝ) 2, f (m * x - 1) + f x > 0) :
  x ∈ Ioo (-1 / 2 : ℝ) (1 / 3 : ℝ) :=
sorry

end range_of_x_l781_781724


namespace floor_sqrt_50_l781_781600

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l781_781600


namespace Jerry_needs_72_dollars_l781_781300

def action_figures_current : ℕ := 7
def action_figures_total : ℕ := 16
def cost_per_figure : ℕ := 8
def money_needed : ℕ := 72

theorem Jerry_needs_72_dollars : 
  (action_figures_total - action_figures_current) * cost_per_figure = money_needed :=
by
  sorry

end Jerry_needs_72_dollars_l781_781300


namespace num_complex_solutions_eq_two_l781_781146

noncomputable def complex_solutions_count (z : ℂ) : Prop :=
  (z^4 - 1) / (z^3 + 2 * z^2 - z - 2) = 0

theorem num_complex_solutions_eq_two : 
  {z : ℂ | complex_solutions_count z}.to_finset.card = 2 := 
sorry

end num_complex_solutions_eq_two_l781_781146


namespace rowing_distance_downstream_l781_781472

theorem rowing_distance_downstream :
  let speed_boat := 6 -- km/hr
  let speed_current := 3 -- km/hr
  let time_downstream := 31.99744020478362 -- seconds
  let effective_speed_downstream := (speed_boat + speed_current) * 1000 / 3600 -- m/s conversion
  let distance_covered := effective_speed_downstream * time_downstream -- m
  distance_covered ≈ 79.9936 :=
by
  -- We use sorry to skip the proof.
  sorry

end rowing_distance_downstream_l781_781472


namespace unattainable_y_l781_781156

def function_y (x : ℝ) := (2 - x) / (3 * x + 4)

theorem unattainable_y : ∀ (y : ℝ), ∀ (x : ℝ), (y = function_y x ∧ x ≠ -4/3) → y ≠ -1/3 :=
begin
  sorry
end

end unattainable_y_l781_781156


namespace circle_chords_intersection_l781_781064

theorem circle_chords_intersection (O : Point) (A B C D P : Point) 
  (r AB CD : ℝ) (M N : Point)
  (hO : dist O A = r)
  (hA : dist A B = AB)
  (hC : dist C D = CD)
  (hP : lie_on_line_segment P A B ∧ lie_on_line_segment P C D)
  (hM : midpoint M A B)
  (hN : midpoint N C D)
  (hMN : dist M N = 10) :
  ∃ m n : ℕ, m.coprime n ∧ (O.dist P)^2 = (↑m / ↑n : ℝ) := 
sorry

end circle_chords_intersection_l781_781064


namespace extremum_of_f_exists_x_for_inequality_inequality_with_lambdas_l781_781834

-- Problem (1)
theorem extremum_of_f {a : ℝ} {λ : ℝ} (hλ : 0 < λ ∧ λ < 1) :
  ∃ x, f x = (1 - λ) * exp a := 
sorry

-- Problem (2)
theorem exists_x_for_inequality {a : ℝ} (ha : 0 < a) :
  ∃ x, x = log (a + 1) ∧ abs ((exp x - 1) / x - 1) < a :=
sorry
  
-- Problem (3)
theorem inequality_with_lambdas {a₁ a₂ : ℝ} {λ₁ λ₂ : ℝ} (ha₁ : 0 < a₁) (ha₂ : 0 < a₂) (hλ : λ₁ + λ₂ = 1 ∧ 0 < λ₁ ∧ 0 < λ₂) :
  a₁^λ₁ * a₂^λ₂ ≤ λ₁ * a₁ + λ₂ * a₂ :=
sorry

end extremum_of_f_exists_x_for_inequality_inequality_with_lambdas_l781_781834


namespace seating_arrangements_l781_781552

inductive Person
| Alice | Bob | Carla | Derek | Eric | Frank

open Person

def sits_next_to (p1 p2 : Person) (seating : list Person) : Prop :=
∃ i, seating.nth i = some p1 ∧ seating.nth (i+1) = some p2 ∨ seating.nth (i-1) = some p2

def valid_seating (seating : list Person) : Prop :=
¬ sits_next_to Alice Bob seating ∧
¬ sits_next_to Alice Carla seating ∧
¬ sits_next_to Derek Eric seating ∧
¬ sits_next_to Derek Frank seating ∧
¬ sits_next_to Frank Bob seating

def count_valid_seatings : ℕ :=
(list.permutations [Alice, Bob, Carla, Derek, Eric, Frank]).filter valid_seating |>.length

theorem seating_arrangements :
  count_valid_seatings = 72 :=
by
  sorry

end seating_arrangements_l781_781552


namespace area_of_triangle_COD_l781_781849

-- Definitions of points and triangle
structure Point := (x : ℝ) (y : ℝ)

def point_O := Point.mk 0 0
def point_D := Point.mk 15 0
def point_C (p : ℝ) (h : 0 ≤ p ∧ p ≤ 15) := Point.mk 0 p

-- Definition of the area of triangle COD
def area_triangle_COD (p : ℝ) (h : 0 ≤ p ∧ p ≤ 15) : ℝ := (15 * p) / 2

-- The theorem stating the problem
theorem area_of_triangle_COD (p : ℝ) (h : 0 ≤ p ∧ p ≤ 15) :
  area_triangle_COD p h = (15 * p) / 2 := by
  sorry

end area_of_triangle_COD_l781_781849


namespace area_of_spherical_n_gon_l781_781068

theorem area_of_spherical_n_gon 
  (n : ℕ) 
  (R : ℝ) 
  (sigma : ℝ) 
  (h1 : 2 ≤ n) 
  (h2 : R > 0)
  (h3 : sigma > 0) : 
  let area := R^2 * (sigma - (n-2) * Real.pi) in 
  area = R^2 * (sigma - (n-2) * Real.pi) := 
sorry

end area_of_spherical_n_gon_l781_781068


namespace candle_burn_time_l781_781469

theorem candle_burn_time :
  let H := 119 -- initial height
  let burn_time (k : ℕ) := 10 * k -- time to burn the k-th cm
  let T := ∑ k in Finset.range (H + 1), burn_time k
  let half_time := T / 2
  let k := Nat.floor ((-1 + Real.sqrt (1 + 4 * (half_time / 5))) / 2)
  10 * (H - k) = 350 :=
by
  let H := 119
  let burn_time (k : ℕ) := 10 * k
  let T := ∑ k in Finset.range (H + 1), burn_time k
  let half_time := T / 2
  let k := Nat.floor ((-1 + Real.sqrt (1 + 4 * (half_time / 5))) / 2)
  show 10 * (H - k) = 350
  sorry

end candle_burn_time_l781_781469


namespace angle_QRP_eq_60_l781_781835

theorem angle_QRP_eq_60
  (Θ : circle)
  (Δ_DEF : triangle)
  (Δ_PQR : triangle)
  (P : Δ_DEF.vertex EF)
  (Q : Δ_DEF.vertex DE)
  (R : Δ_DEF.vertex DF)
  (incircle_DEF : Θ.is_incircle Δ_DEF)
  (circumcircle_PQR : Θ.is_circumcircle Δ_PQR)
  (angle_D : angle Δ_DEF.D = 50)
  (angle_E : angle Δ_DEF.E = 70)
  (angle_F : angle Δ_DEF.F = 60) :
  angle Δ_PQR.QRP = 60 :=
begin
  sorry
end

end angle_QRP_eq_60_l781_781835


namespace acute_or_right_triangle_sum_distances_obtuse_triangle_sum_distances_l781_781368

variables {α β γ : ℝ} {R r : ℝ}
variables {OA2 OB2 OC2 : ℝ} (O : Point) (A B C : Point) [Triangle O A B C]

-- Acute or right triangle case
theorem acute_or_right_triangle_sum_distances (hαβγ : α + β + γ = π)
  (h_dist : OA2 = R * real.cos α ∧ OB2 = R * real.cos β ∧ OC2 = R * real.cos γ)
  (h_trig_id : real.cos α + real.cos β + real.cos γ = 1 + r / R) :
  OA2 + OB2 + OC2 = R + r :=
by
  sorry

-- Obtuse triangle case
theorem obtuse_triangle_sum_distances (hαβγ : α + β + γ = π)
  (h_OB2_OC2_neg : OA2 = R * real.cos α ∧ OB2 = R * real.cos β ∧ OC2 = -R * real.cos γ) 
  (h_trig_id_obtuse : real.cos α + real.cos β - real.cos γ = 1 + r / R) :
  OA2 + OB2 - OC2 = R + r :=
by
  sorry

end acute_or_right_triangle_sum_distances_obtuse_triangle_sum_distances_l781_781368


namespace max_divisors_c_pow_m_l781_781003

theorem max_divisors_c_pow_m (c m : ℕ) (hc : 1 ≤ c ∧ c ≤ 20) (hm : 1 ≤ m ∧ m ≤ 20) : 
  ∃ cm : ℕ, (cm = c^m) ∧ 
    ∀ c' m' : ℕ, (1 ≤ c' ∧ c' ≤ 20) → (1 ≤ m' ∧ m' ≤ 20) → 
    (∃ cm' : ℕ, (cm' = c'^m') → (∀ d : ℕ, (d ∣ cm') → ((nat.divisors_count d) ≤ (nat.divisors_count (c^m)))) :=
begin
  existsi (18^20),
  split,
  { refl },
  { intros c' m' hc' hm',
    existsi (c'^m'),
    intros hcm',
    subst hcm',
    sorry  -- The proof would go here
  }
end

end max_divisors_c_pow_m_l781_781003


namespace problem_a_problem_b_l781_781049

-- Definition of M(i)
def M (i : ℕ) : ℕ :=
  if (Nat.popcount i) % 2 = 0 then 0 else 1

-- Proof Problem (a)
theorem problem_a :
  (∑ i in Finset.range 1000, if M i = M (i + 1) then 1 else 0) ≥ 320 := sorry

-- Proof Problem (b)
theorem problem_b :
  (∑ i in Finset.range 1000000, if M i = M (i + 7) then 1 else 0) ≥ 450000 := sorry

end problem_a_problem_b_l781_781049


namespace equal_angles_in_triangle_l781_781053

open EuclideanGeometry

variables {A B C I M E F N D : Point}

-- Conditions
def is_incenter (I : Point) (ABC : Triangle) : Prop := sorry
def is_midpoint (M : Point) (P Q : Point) : Prop := sorry
def is_midpoint_arc (F : Point) (BC : Line) (circum_A_B_C : Circle) : Prop := sorry
def intersects (MN BC : Line) : Point := sorry

-- Theorem statement
theorem equal_angles_in_triangle
    (hI : is_incenter I (Triangle.mk A B C))
    (hM : is_midpoint M (Segment.mk B I))
    (hE : is_midpoint E (Segment.mk B C))
    (hF : is_midpoint_arc F (Segment.mk B C) (circumcircle_of_triangle A B C))
    (hN : is_midpoint N (Segment.mk E F))
    (hD : D = intersects (Line.mk M N) (Line.mk B C)) :
  ∠ADM = ∠BDM := sorry

end equal_angles_in_triangle_l781_781053


namespace solutions_of_equation_depending_on_a_l781_781994

theorem solutions_of_equation_depending_on_a (a : ℝ) :
  (a < 2.5 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 2) ∧
  (a = 2.5 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 3) ∧
  (2.5 < a ∧ a < 3 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 4) ∧
  (a = 3 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 3) ∧
  (3 < a ∧ a < 4.5 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 4) ∧
  (a = 4.5 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 3) ∧
  (a > 4.5 → ∃! x : ℝ, a(x + |x| - 2) = x^2 + 4x - 5 ∧ count_solutions (a(x + |x| - 2) = x^2 + 4x - 5) = 2) := 
sorry

end solutions_of_equation_depending_on_a_l781_781994


namespace planting_cost_l781_781381

-- Define the costs of the individual items
def cost_of_flowers : ℝ := 9
def cost_of_clay_pot : ℝ := cost_of_flowers + 20
def cost_of_soil : ℝ := cost_of_flowers - 2
def cost_of_fertilizer : ℝ := cost_of_flowers + (0.5 * cost_of_flowers)
def cost_of_tools : ℝ := cost_of_clay_pot - (0.25 * cost_of_clay_pot)

-- Define the total cost
def total_cost : ℝ :=
  cost_of_flowers + cost_of_clay_pot + cost_of_soil + cost_of_fertilizer + cost_of_tools

-- The statement to prove
theorem planting_cost : total_cost = 80.25 :=
by
  sorry

end planting_cost_l781_781381


namespace apple_slices_per_group_l781_781596

-- defining the conditions
variables (a g : ℕ)

-- 1. Equal number of apple slices and grapes in groups
def equal_group (a g : ℕ) : Prop := a = g

-- 2. Grapes packed in groups of 9
def grapes_groups_of_9 (g : ℕ) : Prop := ∃ k : ℕ, g = 9 * k

-- 3. Smallest number of grapes is 18
def smallest_grapes (g : ℕ) : Prop := g = 18

-- theorem stating that the number of apple slices per group is 9
theorem apple_slices_per_group : equal_group a g ∧ grapes_groups_of_9 g ∧ smallest_grapes g → a = 9 := by
  sorry

end apple_slices_per_group_l781_781596


namespace shared_divisors_count_l781_781586

-- Definitions and conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8820

-- Proof problem statement
theorem shared_divisors_count (a b : ℕ) (ha : a = num1) (hb : b = num2) :
  (Nat.divisors (Nat.gcd a b)).card = 24 := by
  subst_vars
  sorry

end shared_divisors_count_l781_781586


namespace evening_temperature_l781_781777

-- Define the given conditions
def t_noon : ℤ := 1
def d : ℤ := 3

-- The main theorem stating that the evening temperature is -2℃
theorem evening_temperature : t_noon - d = -2 := by
  sorry

end evening_temperature_l781_781777


namespace terminal_side_quadrant_l781_781748

theorem terminal_side_quadrant (α : ℝ) (h : α = -5) : (4 : ℕ) := by
  sorry

end terminal_side_quadrant_l781_781748


namespace infinite_product_of_functions_infinite_l781_781819

theorem infinite_product_of_functions_infinite {f g : ℝ → ℝ} (x₀ : ℝ) (M : ℝ)
  (hM : 0 < M)
  (h₁ : ∀ x, |g x| ≥ M)
  (h₂ : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| < δ → |f x| > 1 / ε) :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| < δ → |(f x) * (g x)| > 1 / ε :=
by
  sorry

end infinite_product_of_functions_infinite_l781_781819


namespace ab_lt_ba_necessary_not_sufficient_l781_781328

-- Definitions of necessary conditions
def ln_div_x (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def derivative_ln_div_x (x : ℝ) : ℝ := 
  ((1 : ℝ) - Real.log x) / (x ^ 2)

-- The main proof problem
theorem ab_lt_ba_necessary_not_sufficient 
  (a b : ℝ) 
  (h0 : a > 0) 
  (h1 : b > 0) 
  (h2 : a > b) 
  (h3 : b > Real.exp 1) : 
  ∃ a b : ℝ, a > b ∧ b > Real.exp 1 ∧ a^b < b^a :=
sorry

end ab_lt_ba_necessary_not_sufficient_l781_781328


namespace evaluate_expression_l781_781945

theorem evaluate_expression : 
    8 * 7 / 8 * 7 = 49 := 
by sorry

end evaluate_expression_l781_781945


namespace log_base_10_bounds_l781_781401

theorem log_base_10_bounds (c d : ℕ) (h₁ : 10^5 ≤ 157489) (h₂ : 157489 < 10^6) 
  (h₃ : c = 5) (h₄ : d = 6) : c + d = 11 :=
by 
  have h₅ : 5 < Real.log10 157489 := sorry
  have h₆ : Real.log10 157489 < 6 := sorry
  sorry

end log_base_10_bounds_l781_781401


namespace probability_of_prime_sum_when_two_dice_are_tossed_l781_781939

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- List of prime numbers that can be obtained as a sum of the numbers on two dice
def prime_sums := [2, 3, 5, 7, 11]

-- Total possible outcomes when two dice are tossed
def total_outcomes := (finset.range 7).erase 0.product ((finset.range 7).erase 0)

-- Compute the probability
def prime_sum_probability := (15 : ℚ) / 36

theorem probability_of_prime_sum_when_two_dice_are_tossed :
  (∑ p in finset.filter (λ n, n ∈ prime_sums) (total_outcomes.image (λ p, p.1 + p.2)) = 15) →
  prime_sum_probability = 5 / 12 := sorry

end probability_of_prime_sum_when_two_dice_are_tossed_l781_781939


namespace not_possible_placement_l781_781696

theorem not_possible_placement :
  ∀ (V : Finset ℤ), V.card = 45 → ∀ (D : Finset ℤ), D = Finset.range 10 →
  ¬ (∃ (f : ℤ → ℤ), (∀ a ∈ D, f a ∈ V) ∧
  (∀ a b ∈ D, a ≠ b → ∃ (v₁ v₂ ∈ V), (v₁ ≠ v₂ ∧ f a = v₁ ∧ f b = v₂))) := by
  sorry

end not_possible_placement_l781_781696


namespace lattice_points_line_l781_781815

def T := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20}

theorem lattice_points_line (m : ℚ) (h : ∃ p ∈ T, list.count (λ (p : ℕ × ℕ), p.2 ≤ m * p.1) T = 200) : 
  let interval_width := 1 / 50 in 
  m = interval_width := 
sorry

end lattice_points_line_l781_781815


namespace find_value_of_a_l781_781679

theorem find_value_of_a (a : ℝ) (n : ℕ) (a0 a1 a2 : ℝ) 
  (h1 : a1 = 4) (h2 : a2 = 7) 
  (h3 : (∀ x : ℝ, (ax + 1)^n = ∑ i in finset.range (n + 1), (a_i x^i))) :
  a = 1/2 :=
by 
  sorry

end find_value_of_a_l781_781679


namespace parallel_if_interior_alternate_angles_equal_l781_781396

theorem parallel_if_interior_alternate_angles_equal
  (L₁ L₂ : set (set ℝ)) 
  (T : set (set ℝ)) 
  (interior_alternate_angles : ∀ (α β : ℝ), (α ∈ T → β ∈ T →(α, β) ∈ (((L₁ ∪ L₂)ᶜ) ∩ T)))
  (equal_angles : ∀ (α β : ℝ), (α ∈ T → β ∈ T → α = β)) 
  : is_parallel L₁ L₂ :=
by
  sorry

end parallel_if_interior_alternate_angles_equal_l781_781396
