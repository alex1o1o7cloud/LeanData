import Mathlib
import Mathlib.Algebra.ArithmeticSeq
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Pi
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.LCM
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Convolutions.Basic
import Mathlib.Analysis.SpecialFunctions.AMGM
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Factorization
import Mathlib.Data.Probability
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.ProbabilityTheory.Distributions
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace f_at_2009_l43_43500

noncomputable def f (a b α β x : ℝ) := a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem f_at_2009 (a b α β : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : f a b α β 2008 = -1) : 
  f a b α β 2009 = 1 := 
by
  sorry

end f_at_2009_l43_43500


namespace small_cubes_with_two_faces_painted_l43_43838

theorem small_cubes_with_two_faces_painted :
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  12 * (n - 2) = 36 :=
by
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  exact sorry

end small_cubes_with_two_faces_painted_l43_43838


namespace gcd_765432_654321_eq_3_l43_43221

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43221


namespace shaded_region_perimeter_l43_43797

theorem shaded_region_perimeter (C : ℝ) (hC : C = 72) :
  let r := C / (2 * π) in
  let angle := 60 in -- in degrees
  let arc_length := (angle / 360) * C in
  let perimeter := 3 * arc_length in
  perimeter = 36 := 
by
  have r := C / (2 * π),
  have angle := 60, -- in degrees
  have arc_length := (angle / 360) * C,
  have perimeter := 3 * arc_length,
  sorry

end shaded_region_perimeter_l43_43797


namespace sin_add_pi_over_2_l43_43392

theorem sin_add_pi_over_2 (θ : ℝ) (h : Real.cos θ = -3 / 5) : Real.sin (θ + π / 2) = -3 / 5 :=
sorry

end sin_add_pi_over_2_l43_43392


namespace min_distance_circle_to_line_l43_43420

-- Problem Definitions
def line (x y : ℝ) : Prop := x - y + 4 = 0

def circle (θ : ℝ) : (ℝ × ℝ) :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Distance from a point to a line
def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 4| / Real.sqrt (1^2 + 1^2)

-- The Main Statement to Prove
theorem min_distance_circle_to_line :
  ∃ θ : ℝ, distance_to_line (circle θ) = 2 * Real.sqrt 2 - 2 :=
sorry

end min_distance_circle_to_line_l43_43420


namespace alejandro_rearrangement_l43_43465

noncomputable def rearrange_alejandro : Nat :=
  let X_choices := 2
  let total_letters := 8
  let repeating_letter_factorial := 2
  X_choices * Nat.factorial total_letters / Nat.factorial repeating_letter_factorial

theorem alejandro_rearrangement : rearrange_alejandro = 40320 := by
  sorry

end alejandro_rearrangement_l43_43465


namespace h_at_zero_h_expression_and_min_l43_43409

def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 1
  else if 2 < x ∧ x ≤ 3 then 1/2 * x^2 - 1
  else 0

def h (a : ℝ) : ℝ :=
  let g := λ x => f x - a * x
  let max_g := max (g 1) (max (g 2) (g 3))
  let min_g := min (g 1) (min (g 2) (g 3))
  max_g - min_g

theorem h_at_zero : h 0 = 5 / 2 :=
  sorry

theorem h_expression_and_min :
  ∀ a : ℝ, 
  (h a = if a ≤ 0 then 5/2 - 2 * a
         else if 0 < a ∧ a ≤ 5/4 then 5/2 - a
         else if 5/4 < a ∧ a ≤ 2 then a
         else if 2 < a ∧ a ≤ 3 then 1/2 * a^2 - a + 2
         else 2 * a - 5/2) ∧
  ((∃ a : ℝ, a = 5/4) ∧ (h a = 5/4)) :=
  sorry

end h_at_zero_h_expression_and_min_l43_43409


namespace computation_problem_points_l43_43645

def num_problems : ℕ := 30
def points_per_word_problem : ℕ := 5
def total_points : ℕ := 110
def num_computation_problems : ℕ := 20

def points_per_computation_problem : ℕ := 3

theorem computation_problem_points :
  ∃ x : ℕ, (num_computation_problems * x + (num_problems - num_computation_problems) * points_per_word_problem = total_points) ∧ x = points_per_computation_problem :=
by
  use points_per_computation_problem
  simp
  sorry

end computation_problem_points_l43_43645


namespace sequence_sum_eq_zero_l43_43954

variable {α : Type*} [AddCommGroup α] [Fintype α]

theorem sequence_sum_eq_zero (n : ℕ) (a : Fin n → α) (h : a 0 = a (Fin.last n)) :
  (Finset.sum (Finset.range (n - 1)) (λ i, a (Fin.castSucc i.succ) - a i.castSucc)) = 0 :=
by
  sorry

end sequence_sum_eq_zero_l43_43954


namespace slope_of_equidistant_line_l43_43936

def point := ℝ × ℝ

def slope (p1 p2 : point) : ℝ :=
(p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_equidistant_line :
  let P : point := (0, 2)
  let Q : point := (12, 8)
  let M : point := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let slope_PQ := slope P Q
  let slope_perpendicular := -1 / slope_PQ
  let point_through := (4, 4)
  slope point_through (point_through.1 + 1, point_through.2 + slope_perpendicular) = -2 :=
by
  sorry

end slope_of_equidistant_line_l43_43936


namespace num_double_yolk_eggs_l43_43625

noncomputable def double_yolk_eggs (total_eggs total_yolks : ℕ) (double_yolk_contrib : ℕ) : ℕ :=
(total_yolks - total_eggs + double_yolk_contrib) / double_yolk_contrib

theorem num_double_yolk_eggs (total_eggs total_yolks double_yolk_contrib expected : ℕ)
    (h1 : total_eggs = 12)
    (h2 : total_yolks = 17)
    (h3 : double_yolk_contrib = 2)
    (h4 : expected = 5) :
  double_yolk_eggs total_eggs total_yolks double_yolk_contrib = expected :=
by
  rw [h1, h2, h3, h4]
  dsimp [double_yolk_eggs]
  norm_num
  sorry

end num_double_yolk_eggs_l43_43625


namespace average_and_variance_after_increase_l43_43545

variables {n : ℕ} {x : ℕ → ℝ}

def average (x : ℕ → ℝ) (n : ℕ) : ℝ := (finset.sum (finset.range n) x) / n
def variance (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.sum (finset.range n) (λ i, (x i - average x n) ^ 2)) / n

theorem average_and_variance_after_increase
  (x : ℕ → ℝ)
  (n : ℕ)
  (h_avg : average x n = 4.8)
  (h_var : variance x n = 3.6) :
  average (λ i, 60 + x i) n = 64.8 ∧ variance (λ i, 60 + x i) n = 3.6 :=
by
  -- Proof goes here
  sorry

end average_and_variance_after_increase_l43_43545


namespace flag_movement_distance_l43_43987

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l43_43987


namespace A_inter_B_eq_l43_43066

def A := {x : ℤ | 1 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 3}
def B := {x : ℤ | 5 ≤ x ∧ x < 9}

theorem A_inter_B_eq : A ∩ B = {5, 6, 7} :=
by sorry

end A_inter_B_eq_l43_43066


namespace intersection_A_B_l43_43740

-- Defining set A condition
def A : Set ℝ := {x | x - 1 < 2}

-- Defining set B condition
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- The goal to prove
theorem intersection_A_B : {x | x > 0 ∧ x < 3} = (A ∩ { x | 0 < x ∧ x < 8 }) :=
by
  sorry

end intersection_A_B_l43_43740


namespace last_digit_of_S_l43_43326

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_S : last_digit (54 ^ 2020 + 28 ^ 2022) = 0 :=
by 
  -- The Lean proof steps would go here
  sorry

end last_digit_of_S_l43_43326


namespace find_rate_percent_l43_43922

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem find_rate_percent (SI P T : ℝ) (h1 : SI = 200) (h2 : P = 1600) (h3 : T = 4) :
  ∃ R, simple_interest P R T = SI ∧ R = 3.125 :=
by
  use 3.125
  rw [simple_interest, h2, h3]
  norm_num
  split
  { norm_num }
  { refl }
  sorry

end find_rate_percent_l43_43922


namespace remainder_1234_mul_2047_mod_600_l43_43231

theorem remainder_1234_mul_2047_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end remainder_1234_mul_2047_mod_600_l43_43231


namespace prob_white_first_yellow_second_l43_43906

-- Defining the number of yellow and white balls
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

-- Defining the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the events A and B
def event_A : Prop := true -- event A: drawing a white ball first
def event_B : Prop := true -- event B: drawing a yellow ball second

-- Conditional probability P(B|A)
def prob_B_given_A : ℚ := 6 / (total_balls - 1)

-- Main theorem stating the proof problem
theorem prob_white_first_yellow_second : prob_B_given_A = 2 / 3 :=
by
  sorry

end prob_white_first_yellow_second_l43_43906


namespace solve_equation_l43_43130

theorem solve_equation : ∃ x : ℝ, 4^x - 2^x - 6 = 0 ∧ x = Real.log 3 / Real.log 2 :=
by
  exists Real.log 3 / Real.log 2
  split
  · sorry
  · exact rfl

end solve_equation_l43_43130


namespace gcd_765432_654321_l43_43179

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43179


namespace rhombus_side_length_15_l43_43709

variable {p : ℝ} (h_p : p = 60)
variable {n : ℕ} (h_n : n = 4)

noncomputable def side_length_of_rhombus (p : ℝ) (n : ℕ) : ℝ :=
p / n

theorem rhombus_side_length_15 (h_p : p = 60) (h_n : n = 4) :
  side_length_of_rhombus p n = 15 :=
by
  sorry

end rhombus_side_length_15_l43_43709


namespace log_base_three_l43_43227

open Real

noncomputable def log_exp : ℝ :=
  log 81 / log 3 + log 27^(1/3) / log 3 + log 4^(1/2) / log 3

theorem log_base_three (a b c : ℝ) (ha : a = 81) (hb : b = 27) (hc : c = 4) :
  log_base 3 (a^(1 : ℝ) * b^(1/3 : ℝ) * c^(1/2 : ℝ)) = 5 + log_base 3 2 := by
  -- Given the assumptions:
  -- ha : a = 81
  -- hb : b = 27
  -- hc : c = 4
  have h1 : log_base 3 a = 4 := by rw [ha, log_base_pow, log_base_self]; norm_num
  have h2 : log_base 3 b^(1/3 : ℝ) = log_base 3 3 := by rw [hb, log_base_pow, mul_inv_cancel, log_base_self] ; norm_num
  have h3 : log_base 3 c^(1/2 : ℝ) = log 2 / log 3 := by rw [hc, log_base_pow, log_base_self]; norm_num
  rw [log_base_mul, log_base_mul, h1, h2, h3]; ring
  sorry

end log_base_three_l43_43227


namespace spaceship_journey_time_l43_43642

theorem spaceship_journey_time
  (initial_travel_1 : ℕ)
  (first_break : ℕ)
  (initial_travel_2 : ℕ)
  (second_break : ℕ)
  (travel_per_segment : ℕ)
  (break_per_segment : ℕ)
  (total_break_time : ℕ)
  (remaining_break_time : ℕ)
  (num_segments : ℕ)
  (total_travel_time : ℕ)
  (total_time : ℕ) :
  initial_travel_1 = 10 →
  first_break = 3 →
  initial_travel_2 = 10 →
  second_break = 1 →
  travel_per_segment = 11 →
  break_per_segment = 1 →
  total_break_time = 8 →
  remaining_break_time = total_break_time - (first_break + second_break) →
  num_segments = remaining_break_time / break_per_segment →
  total_travel_time = initial_travel_1 + initial_travel_2 + (num_segments * travel_per_segment) →
  total_time = total_travel_time + total_break_time →
  total_time = 72 :=
by
  intros
  sorry

end spaceship_journey_time_l43_43642


namespace large_root_change_l43_43548

theorem large_root_change (p q p' q' : ℝ) (h∆p : |p' - p| ≤ 0.001) (h∆q : |q' - q| ≤ 0.001) :
  ∃ r r', r ≠ r' ∧ (|r' - r| > 1000) :=
sorrry

end large_root_change_l43_43548


namespace hypotenuse_of_30_60_90_triangle_l43_43839

theorem hypotenuse_of_30_60_90_triangle (leg : ℝ) (hypotenuse : ℝ) (h_leg : leg = 15) (h_angle : ∠ABC = 30) : hypotenuse = 30 := by
  -- One leg of the right triangle is given as 15
  have h_leg : leg = 15 := h_leg
  -- Hypotenuse is twice the length of the leg opposite the 30-degree angle
  have h_hyp : hypotenuse = 2 * leg := by sorry
  -- Substitute the given leg value to find the hypotenuse
  rw h_leg at h_hyp
  exact h_hyp

end hypotenuse_of_30_60_90_triangle_l43_43839


namespace gcd_of_765432_and_654321_l43_43198

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43198


namespace ratio_of_areas_l43_43279

theorem ratio_of_areas (r : ℝ) (w_smaller : ℝ) (h_smaller : ℝ) (h_semi : ℝ) :
  (5 / 4) * 40 = r + 40 →
  h_semi = 20 →
  w_smaller = 5 →
  h_smaller = 20 →
  2 * w_smaller * h_smaller / ((1 / 2) * π * h_semi^2) = 1 / π :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_l43_43279


namespace sqrt_18_eq_ab2_l43_43372

def a := Real.sqrt 2
def b := Real.sqrt 3

theorem sqrt_18_eq_ab2 : Real.sqrt 18 = a * b ^ 2 :=
by
  sorry

end sqrt_18_eq_ab2_l43_43372


namespace probability_negative_product_l43_43992

open_locale classical -- Use classical logic

theorem probability_negative_product :
  let S := {-3, -1, 2, 6, 5, -4}
  let total_ways := nat.choose (fintype.card S) 2
  let favorable_ways := 3 * 3 -- 3 ways to choose a negative from 3, 3 ways to choose a positive from 3
  (favorable_ways : ℝ) / total_ways = (3 : ℝ) / 5 :=
by
  sorry

end probability_negative_product_l43_43992


namespace initial_amounts_A_is_48_l43_43657

noncomputable def initial_amounts : Type :=
  { a : ℤ // ∃ b c : ℤ, 
    let A' := 16 * a - 16 * b - 16 * c,
        B' := -8 * a + 24 * b - 8 * c,
        C' := -4 * a - 4 * b + 28 * c in
    (A' = 32) ∧ (B' = 32) ∧ (C' = 32) }

theorem initial_amounts_A_is_48 : initial_amounts :=
⟨48, by
  use 32
  use 20
  have A_final : 16 * 48 - 16 * 32 - 16 * 20 = 32, by norm_num
  have B_final : -8 * 48 + 24 * 32 - 8 * 20 = 32, by norm_num
  have C_final : -4 * 48 - 4 * 32 + 28 * 20 = 32, by norm_num
  exact ⟨A_final, B_final, C_final⟩⟩

end initial_amounts_A_is_48_l43_43657


namespace cookies_left_l43_43031

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end cookies_left_l43_43031


namespace proof_problem_l43_43739

noncomputable def proposition_p : Prop := ∃ x : ℝ, x - 2 > 0
noncomputable def proposition_q : Prop := ∀ x : ℝ, real.sqrt x < x

theorem proof_problem : proposition_p ∧ ¬ proposition_q := sorry

end proof_problem_l43_43739


namespace proj_3a_minus_b_onto_b_l43_43397

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Conditions
-- 1. |b| = 1
axiom norm_b_one : ‖b‖ = 1

-- 2. a • b = -2
axiom dot_a_b_neg_two : inner a b = -2

-- Prove the projection of (3a - b) onto b is -7b
theorem proj_3a_minus_b_onto_b : (orthogonalProjection b (3 • a - b)) = -7 • b := by
  sorry

end proj_3a_minus_b_onto_b_l43_43397


namespace gcd_765432_654321_l43_43209

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43209


namespace sum_on_simple_interest_is_1750_l43_43250

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem sum_on_simple_interest_is_1750 :
  let P_ci := 4000
  let r_ci := 0.10
  let t_ci := 2
  let r_si := 0.08
  let t_si := 3
  let CI := compound_interest P_ci r_ci t_ci
  let SI := CI / 2
  let P_si := SI / (r_si * t_si)
  P_si = 1750 :=
by
  sorry

end sum_on_simple_interest_is_1750_l43_43250


namespace triangle_area_heron_l43_43605

theorem triangle_area_heron :
  let a := 36 : ℝ
  let b := 34 : ℝ
  let c := 20 : ℝ
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c)) ≈ 333.73 := by
    let a := 36 : ℝ
    let b := 34 : ℝ
    let c := 20 : ℝ
    let s := (a + b + c) / 2
    have semi_perimeter : s = 45 := by norm_num
    have herons_area : sqrt (s * (s - a) * (s - b) * (s - c)) = sqrt (45 * 9 * 11 * 25) := by norm_num
    have approximate_area : sqrt (45 * 9 * 11 * 25) ≈ 333.73 := by norm_num
    exact approximate_area

end triangle_area_heron_l43_43605


namespace dividend_calculation_l43_43161

noncomputable def dividend_per_share (investment price_per_share total_income : ℝ) : ℝ :=
  let shares := (investment / price_per_share).floor in
  total_income / shares

theorem dividend_calculation :
  dividend_per_share 3200 85 250 ≈ 6.757 := 
sorry

end dividend_calculation_l43_43161


namespace depth_of_river_l43_43640

variable (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ)

def flow_rate_mpm := flow_rate_kmph * 1000 / 60

theorem depth_of_river 
  (h1 : width = 45) 
  (h2 : flow_rate_kmph = 7) 
  (h3 : volume_per_minute = 10500) :
  volume_per_minute / (flow_rate_mpm flow_rate_kmph * width) = 2 :=
by 
  sorry

end depth_of_river_l43_43640


namespace exists_pos_real_x_y_l43_43083

noncomputable def pos_real := {x : ℝ // 0 < x}

open Function

theorem exists_pos_real_x_y (f : pos_real → pos_real) : 
  ∃ x y : pos_real, f (x + y) < f x + y * f (f x) :=
by 
  sorry

end exists_pos_real_x_y_l43_43083


namespace ap_sub_aq_l43_43133

variable {n : ℕ} (hn : n > 0)

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) (hn : n > 0) : ℕ :=
S n - S (n - 1)

theorem ap_sub_aq (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : p - q = 5) :
  a p hp - a q hq = 20 :=
sorry

end ap_sub_aq_l43_43133


namespace min_value_of_y_abs_plus_dist_on_parabola_l43_43737

-- We define some structures and notation to help state our theorem

structure Point where
  x : ℝ
  y : ℝ

def parabola (P : Point) : Prop :=
  P.x^2 = -4 * P.y

def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

def |P : Point, Q : Point| := distance P Q

noncomputable def y_abs (P : Point) : ℝ := |P.y|

noncomputable def min_value_y_abs_plus_dist (Q P : Point) : ℝ :=
  y_abs P + (|P Q|)

theorem min_value_of_y_abs_plus_dist_on_parabola :
  let Q : Point := {x := -2 * real.sqrt 2, y := 0}
  ∃ P : Point, parabola P →
  (min_value_y_abs_plus_dist Q P = 2) :=
by sorry

end min_value_of_y_abs_plus_dist_on_parabola_l43_43737


namespace base6_120_eq_base2_110000_l43_43994

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => let digit := n % 10
         let rest := n / 10
         digit * 6 ^ Nat.clog10 (n + 1) + base6_to_base10 rest

def base10_to_base2 (n : Nat) : List Bool :=
  if n = 0 then []
  else (n % 2 = 1 :: base10_to_base2 (n / 2)).reverse

def base6_to_base2 (n : Nat) : List Bool :=
  base10_to_base2 (base6_to_base10 n)

theorem base6_120_eq_base2_110000 : base6_to_base2 120 = [true, true, false, false, false, false] :=
by
  sorry

end base6_120_eq_base2_110000_l43_43994


namespace remainder_when_abc_divided_by_7_is_0_l43_43768

theorem remainder_when_abc_divided_by_7_is_0
  (a b c : ℕ)
  (h1 : a < 7)
  (h2 : b < 7)
  (h3 : c < 7)
  (h4 : a + 2 * b + 3 * c ≡ 0 [MOD 7])
  (h5 : 2 * a + 3 * b + c ≡ 2 [MOD 7])
  (h6 : 3 * a + b + 2 * c ≡ 4 [MOD 7]) :
  (a * b * c) % 7 = 0 :=
begin
  sorry
end

end remainder_when_abc_divided_by_7_is_0_l43_43768


namespace max_area_triangle_l43_43115

def line (k : ℝ) : set (ℝ × ℝ) := { p | k * p.1 - p.2 + 2 = 0 }
def circle : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 - 4 * p.1 - 12 = 0 }
def intersection_points (k : ℝ) := (line k) ∩ circle

theorem max_area_triangle (k : ℝ) :
  let QRC := intersection_points k
  -- this is a placeholder definition, you need to define area correctly
  -- suppose area is a function that computes the area of triangle given three points
  -- assuming here Q and R are points from QRC and C is the center of the circle.
  area := sorry in
  -- max_area is the maximum area of the triangle formed by Q, R, and center C
  ∃ max_area : ℝ, (forall Q R C, Q ∈ QRC → R ∈ QRC → C = (2,0) → area Q R C ≤ max_area) ∧
  max_area = 8 := sorry

end max_area_triangle_l43_43115


namespace boat_license_combinations_l43_43627

/--
Given a boat license system where each license consists of one of four letters (A, M, S, or T)
followed by any six digits, prove that the total number of different boat license number combinations
is 4,000,000.
-/
theorem boat_license_combinations :
  let letters := 4              -- 4 choices (A, M, S, T)
  let digits := 10 ^ 6          -- 10 possibilities for each of the 6 digits
  letters * digits = 4000000 := 
by
  let letters := 4
  let digits := 10 ^ 6
  show letters * digits = 4000000 from sorry

end boat_license_combinations_l43_43627


namespace selected_ids_l43_43635

def is_valid_id (id : ℕ) : Prop :=
  id < 60 

def selection_rule (id : ℕ) : Bool :=
  (id % 6 == 3)

def sample_ids (n : ℕ) : List ℕ :=
  List.range n |>.filter is_valid_id |>.filter selection_rule

theorem selected_ids (n : ℕ) (hn : n = 10) :
  sample_ids 60 = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57] :=
sorry

end selected_ids_l43_43635


namespace cos_identity_l43_43371

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, (Real.sqrt 3) / 2)
  let b := (Real.sin (x - Real.pi / 3), 1)
  a.1 * b.1 + a.2 * b.2

theorem cos_identity (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3))
  (hf : f x0 = 4 / 5) :
  Real.cos (2 * x0 - Real.pi / 12) = -7 * Real.sqrt 2 / 10 :=
sorry

end cos_identity_l43_43371


namespace find_root_product_l43_43826

theorem find_root_product :
  (∃ r s t : ℝ, (∀ x : ℝ, (x - r) * (x - s) * (x - t) = x^3 - 15 * x^2 + 26 * x - 8) ∧
  (1 + r) * (1 + s) * (1 + t) = 50) :=
sorry

end find_root_product_l43_43826


namespace summer_sales_is_2_million_l43_43961

def spring_sales : ℝ := 4.8
def autumn_sales : ℝ := 7
def winter_sales : ℝ := 2.2
def spring_percentage : ℝ := 0.3

theorem summer_sales_is_2_million :
  ∃ (total_sales : ℝ), total_sales = (spring_sales / spring_percentage) ∧
  ∃ summer_sales : ℝ, total_sales = spring_sales + summer_sales + autumn_sales + winter_sales ∧
  summer_sales = 2 :=
by
  sorry

end summer_sales_is_2_million_l43_43961


namespace arrangements_count_l43_43854

-- Define the conditions of the problem
def peopleOrderingsNotAdjacent (A B : ℕ) (numRows : ℕ) (numCols : ℕ) (positions : List ℕ) : Prop :=
  (numRows = 2) ∧ (numCols = 3) ∧
  (length positions = 6) ∧
  ∀ p ∈ positions, p ≠ A → p ≠ B →
    ¬adjacentHorizontallyOrVertically A B positions

-- Auxiliary definition for adjacency
def adjacentHorizontallyOrVertically (A B : ℕ) (positions : List ℕ) : Prop :=
  (A + 1 = B ∨ A - 1 = B ∨ A + 3 = B ∨ A - 3 = B)

-- The statement to prove in Lean 4
theorem arrangements_count (numRows : ℕ) (numCols : ℕ) (positions : List ℕ) :
  (peopleOrderingsNotAdjacent 1 2 numRows numCols positions) → 
  (numArrangements 1 2 numRows numCols = 384) := 
sorry

end arrangements_count_l43_43854


namespace find_bc_and_area_of_triangle_l43_43714

variable (a b c A : ℝ)

axiom condition1 : a = 2 * Real.sqrt 3
axiom condition2 : b + c = 4
axiom condition3 : Real.cos A = -1 / 2

theorem find_bc_and_area_of_triangle (h1 : a = 2 * Real.sqrt 3) (h2 : b + c = 4) (h3 : Real.cos A = -1 / 2) :
  ∃ (bc : ℝ) (area : ℝ), bc = 4 ∧ area = Real.sqrt 3 :=
by {
  -- We assume bc and area are given such that the statements below hold:
  let bc := 4,
  let area := Real.sqrt 3,
  use [bc, area],
  split,
  { -- Proof showing bc = 4
    sorry
  },
  { -- Proof showing area = sqrt(3)
    sorry
  }
}

end find_bc_and_area_of_triangle_l43_43714


namespace find_k_l43_43957

-- Definitions for the points and the equation of the line
def point1 : ℝ × ℝ := (4, -5)
def point2 (k : ℝ) : ℝ × ℝ := (k, 23)
def line_eq (x y : ℝ) := 3 * x - 4 * y = 12

-- The condition that the line through point1 and point2 is parallel to line_eq.
def parallel_condition (k : ℝ) : Prop :=
  let slope_p1_p2 := (23 + 5) / (k - 4) in
  slope_p1_p2 = 3 / 4

-- The proof goal
theorem find_k (k : ℝ) (h : parallel_condition k) : k = 124 / 3 := 
by
  sorry

end find_k_l43_43957


namespace product_decimal_places_product_rounded_quotient_value_remainder_value_l43_43876

-- Define the base numbers
def num1 := 2.96
def num2 := 4.39
def num3 := 9.97
def num4 := 3.21

-- Define their products
def product := num1 * num2
def quotient := num3 / num4
def remainder := num3 % num4

-- Statements to prove
theorem product_decimal_places : (product * 10000) % 1 = 0 :=
sorry

theorem product_rounded : Real.round (product * 100) / 100 = 12.99 :=
sorry

theorem quotient_value : quotient.toInt = 3 :=
sorry

theorem remainder_value : remainder = 0.34 :=
sorry

end product_decimal_places_product_rounded_quotient_value_remainder_value_l43_43876


namespace distinct_real_roots_of_quadratic_l43_43451

theorem distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, x^2 - 4*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂)) ↔ m < 2 := by
sorry

end distinct_real_roots_of_quadratic_l43_43451


namespace isosceles_triangle_base_length_and_range_l43_43734

theorem isosceles_triangle_base_length_and_range (x y : ℝ) (h_perimeter : 2 * x + y = 20) :
  y = -2 * x + 20 ∧ 5 < x ∧ x < 10 :=
begin
  sorry
end

end isosceles_triangle_base_length_and_range_l43_43734


namespace num_white_squares_in_24th_row_l43_43615

-- Define the function that calculates the total number of squares in the nth row
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

-- Define the function that calculates the number of white squares in the nth row
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2

-- Problem statement for the Lean 4 theorem
theorem num_white_squares_in_24th_row : white_squares 24 = 23 :=
by {
  -- Lean proof generation will be placed here
  sorry
}

end num_white_squares_in_24th_row_l43_43615


namespace sequence_general_term_l43_43879

theorem sequence_general_term (a : ℕ → ℤ) (h₀ : a 0 = 1) (hstep : ∀ n, a (n + 1) = if a n = 1 then 0 else 1) :
  ∀ n, a n = (1 + (-1)^(n + 1)) / 2 :=
sorry

end sequence_general_term_l43_43879


namespace travel_west_l43_43009

-- Define the condition
def travel_east (d: ℝ) : ℝ := d

-- Define the distance for east
def east_distance := (travel_east 3 = 3)

-- The theorem to prove that traveling west for 2km should be -2km
theorem travel_west (d: ℝ) (h: east_distance) : travel_east (-d) = -d := 
by
  sorry

-- Applying this theorem to the specific case of 2km travel
example (h: east_distance): travel_east (-2) = -2 :=
by 
  apply travel_west 2 h

end travel_west_l43_43009


namespace line_circle_tangent_or_disjoint_l43_43065

theorem line_circle_tangent_or_disjoint (m : ℝ) (h : m > 0) :
  let d := (1 + m) / 2 in
  let r := Real.sqrt m in
  (d - r) >= 0 ↔ (d = r ∨ d > r) := 
sorry

end line_circle_tangent_or_disjoint_l43_43065


namespace ensure_bomb_l43_43238

-- Define the deck and the problem setting
def rank := {n : ℕ // n ≥ 1 ∧ n ≤ 13}
def deck := fin 52

-- Problem statement in Lean 4
theorem ensure_bomb (drawn_cards : finset deck) (h1 : drawn_cards.card = 40) :
  ∃ r : rank, 4 ≤ (drawn_cards.filter (λ (c : deck), c.1 % 13 = r.1)).card := by
  sorry

end ensure_bomb_l43_43238


namespace symmetry_of_complex_numbers_l43_43830

theorem symmetry_of_complex_numbers (z1 z2 : ℂ) (hz1 : z1 = 1 + ⟨0,1⟩i)
  (h_symm : ∃ x : ℂ, z1 = x + ⟨0,1⟩i ∧ z2 = x - ⟨0,1⟩i) :
  z1 * z2 = 2 := by
sorry

end symmetry_of_complex_numbers_l43_43830


namespace gcd_765432_654321_l43_43212

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43212


namespace gcd_765432_654321_eq_3_l43_43217

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43217


namespace total_flag_distance_moved_l43_43983

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l43_43983


namespace beautiful_infinite_l43_43289

def is_beautiful (n : ℕ) : Prop :=
  ∀ (b : ℕ), 4 ≤ b ∧ b ≤ 10000 → ∃ (k : ℕ), 2023 = nat.digits b n

theorem beautiful_infinite : ¬(∃ (s : set ℕ), s = { n | is_beautiful n } ∧ s.finite) :=
by
  intro h
  sorry 

end beautiful_infinite_l43_43289


namespace sum_sin_squared_l43_43504

theorem sum_sin_squared {T : Set ℝ} (hT : T = { x | 0 < x ∧ x < π ∧ (∃ p q r : ℝ, {sin x, cos x, tan x} = {p, q, r} ∧ is_isosceles {p, q, r}) }) :
  ∑ x in T, sin x ^ 2 = 1 / 2 :=
by
  sorry

def is_isosceles {α : Type*} [OrderedCommSemiring α] (s : Set α) : Prop :=
  ∃ a b, s = {a, a, b} ∨ s = {a, b, b}

end sum_sin_squared_l43_43504


namespace find_value_of_number_l43_43946

theorem find_value_of_number (n : ℝ) (v : ℝ) (h : n = -72.0) (h_val : v = 0.833 * n) : v = -59.976 :=
by
  rw [h, h_val]
  sorry

end find_value_of_number_l43_43946


namespace rotten_oranges_percentage_l43_43294

variable (total_oranges total_bananas rotten_bananas_percentage fruits_good_percentage : ℝ)

-- Given conditions
def total_fruits := total_oranges + total_bananas
def fruits_good := 0.898 * total_fruits
def rotten_bananas := 0.03 * total_bananas
def good_oranges := fruits_good - total_bananas + rotten_bananas
def rotten_oranges := total_oranges - good_oranges

-- Theorem to prove the percentage of rotten oranges
theorem rotten_oranges_percentage (total_oranges total_bananas : ℝ) (fruits_good_percentage : ℝ) 
  (rotten_bananas_percentage : ℝ) : 
  (rotten_oranges / total_oranges) * 100 = 15 := by
  -- substituting the conditions with specific values as per problem statement
  let total_oranges := 600
  let total_bananas := 400
  let fruits_good_percentage := 0.898
  let rotten_bananas_percentage := 0.03

  -- proofs are skipped, thus using sorry
  sorry

end rotten_oranges_percentage_l43_43294


namespace sum_of_areas_of_triangles_l43_43890

theorem sum_of_areas_of_triangles (m n p : ℤ) :
  let vertices := {v : ℝ × ℝ × ℝ | v.1 ∈ {0, 2} ∧ v.2 ∈ {0, 2} ∧ v.3 ∈ {0, 2}},
      triangles := {t : tuple ℝ 9 | 
                     ∃ v1 v2 v3 ∈ vertices, 
                     t = (v1.1, v1.2, v1.3, v2.1, v2.2, v2.3, v3.1, v3.2, v3.3)},
      triangle_area := λ t : tuple ℝ 9, 
        let (x1, y1, z1, x2, y2, z2, x3, y3, z3) := t in
        1 / 2 * real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2) *
                 real.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2 + (z3 - z1) ^ 2) *
                 real.sqrt ((x3 - x2) ^ 2 + (y3 - y2) ^ 2 + (z3 - z2) ^ 2)
  in (∑ t in triangles, triangle_area t) = 48 + real.sqrt 4608 + real.sqrt 3072 :=
sorry

end sum_of_areas_of_triangles_l43_43890


namespace roots_rational_l43_43087

/-- Prove that the roots of the equation x^2 + px + q = 0 are always rational,
given the rational numbers p and q, and a rational n where p = n + q / n. -/
theorem roots_rational
  (n p q : ℚ)
  (hp : p = n + q / n)
  : ∃ x y : ℚ, x^2 + p * x + q = 0 ∧ y^2 + p * y + q = 0 ∧ x ≠ y :=
sorry

end roots_rational_l43_43087


namespace gcd_of_765432_and_654321_l43_43201

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43201


namespace number_of_students_l43_43013

-- Defining the parameters and conditions
def passing_score : ℕ := 65
def average_score_whole_class : ℕ := 66
def average_score_passed : ℕ := 71
def average_score_failed : ℕ := 56
def increased_score : ℕ := 5
def post_increase_average_passed : ℕ := 75
def post_increase_average_failed : ℕ := 59
def num_students_lb : ℕ := 15 
def num_students_ub : ℕ := 30

-- Lean statement to prove the number of students in the class
theorem number_of_students (x y n : ℕ) 
  (h1 : average_score_passed * x + average_score_failed * y = average_score_whole_class * (x + y))
  (h2 : (average_score_whole_class + increased_score) * (x + y) = post_increase_average_passed * (x + n) + post_increase_average_failed * (y - n))
  (h3 : num_students_lb < x + y ∧ x + y < num_students_ub)
  (h4 : x = 2 * y)
  (h5 : y = 4 * n) : x + y = 24 :=
sorry

end number_of_students_l43_43013


namespace find_least_n_l43_43823

def sequence (n : ℕ) : ℕ :=
  if n = 5 then 5
  else if n > 5 then 50 * sequence (n - 1) + 5 * n
  else 0 -- This is just a fallback, won't be used

theorem find_least_n :
  ∃ (n : ℕ), n > 5 ∧ sequence n % 55 = 0 ∧ (∀ m, m > 5 → m < n → sequence m % 55 ≠ 0) :=
begin
  use 7,
  split,
  { -- 7 > 5
    norm_num,
  },
  split,
  { -- sequence 7 is divisible by 55
    change sequence 7 % 55 = 0,
    have h_seq_6 : sequence 6 = 280,
    { -- Calculating sequence 6
      norm_num,
      unfold sequence,
      rw if_pos rfl,
      unfold sequence at h_seq_6,
      rw if_neg (show 5 ≠ 6 by norm_num),
      rw if_pos rfl,
      norm_num,
    },
    have h_seq_7 : sequence 7 = 14035,
    { -- Calculating sequence 7
      unfold sequence,
      rw if_neg (show 7 ≠ 5 by norm_num),
      rw if_pos (show 7 > 5 by norm_num),
      rw h_seq_6,
      norm_num,
    },
    rw h_seq_7,
    norm_num,
  },
  { -- For all m > 5, m < 7, sequence m is not divisible by 55
    intros m hm1 hm2,
    cases hm2,
    exact if_neg (by norm_num : 5 < 6) (show sequence 6 % 55 = 0, by norm_num),
  }
end

end find_least_n_l43_43823


namespace even_number_of_rooks_on_black_squares_l43_43516

def is_black_square (i j : ℕ) : Prop := (i + j) % 2 = 0

def non_attacking_rooks (positions : Fin 8 → (ℕ × ℕ)) : Prop :=
  ∀ i j : Fin 8, i ≠ j → (positions i).fst ≠ (positions j).fst ∧ (positions i).snd ≠ (positions j).snd

theorem even_number_of_rooks_on_black_squares (positions : Fin 8 → (ℕ × ℕ))
  (h : non_attacking_rooks positions) : ∃ n, is_even (Finset.count is_black_square (Finset.univ.image (λ i, (positions i).fst + (positions i).snd))) :=
sorry

end even_number_of_rooks_on_black_squares_l43_43516


namespace triangle_area_multiplier_l43_43774

theorem triangle_area_multiplier (a b : ℝ) (θ : ℝ) :
  let A := (1/2) * a * b * sin θ
  let a' := 1.5 * a
  let b' := 1.5 * b
  let θ' := θ + 15 * (π / 180)
  let A' := (1/2) * a' * b' * sin θ'
  0 < a → 0 < b → 0 < sin θ →
  A' / A ≈ 1.2 := 
by
  sorry

end triangle_area_multiplier_l43_43774


namespace transformed_sum_cannot_be_175_l43_43332

theorem transformed_sum_cannot_be_175 (nums : Fin 13 → ℤ) (h_sum : (∑ i, nums i) = 125) 
  (transform : Fin 13 → ℤ → ℤ) (h_transform : ∀ i, transform i (nums i) = (nums i / 3) ∨ transform i (nums i) = (nums i * 5)) :
  (∑ i, transform i (nums i)) ≠ 175 := 
sorry

end transformed_sum_cannot_be_175_l43_43332


namespace train_length_l43_43649

theorem train_length (cross_time : ℕ) (speed_kmh : ℕ) 
                     (conversion_factor : 1 * 1000 / 3600 = (1 / 3.6)) 
                     (speed_ms := speed_kmh * (1 / 3.6)) :
  cross_time = 10 → speed_kmh = 144 → speed_ms = 40 → (speed_ms * cross_time = 400) :=
by
  sorry

end train_length_l43_43649


namespace find_value_at_frac_one_third_l43_43399

theorem find_value_at_frac_one_third
  (f : ℝ → ℝ) 
  (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 1 / 4) :
  f (1 / 3) = 9 := 
  sorry

end find_value_at_frac_one_third_l43_43399


namespace perpendicular_lines_slope_l43_43761

theorem perpendicular_lines_slope (a : ℝ) :
  (∀ x1 y1 x2 y2: ℝ, y1 = a * x1 - 2 ∧ y2 = x2 + 1 → (a * 1) = -1) → a = -1 :=
by
  sorry

end perpendicular_lines_slope_l43_43761


namespace smallest_positive_period_of_f_l43_43882

def f (x : ℝ) : ℝ := |sin (2 * x) + cos (2 * x)|

theorem smallest_positive_period_of_f : ∃ p > 0, (∀ x, f (x + p) = f x) ∧ 
                                        (∀ q, q > 0 → (∀ x, f (x + q) = f x) → q ≥ p) ∧ 
                                        p = π :=
sorry

end smallest_positive_period_of_f_l43_43882


namespace vector_addition_l43_43432

-- Definitions for the vectors
def a : ℝ × ℝ := (5, 2)
def b : ℝ × ℝ := (1, 6)

-- Proof statement (Note: "theorem" is used here instead of "def" because we are stating something to be proven)
theorem vector_addition : a + b = (6, 8) := by
  sorry

end vector_addition_l43_43432


namespace vector_combination_l43_43473

noncomputable def vector_length (v : V) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry
noncomputable def angle_between (v w : V) : ℝ := sorry

variables (OA OB OC : V)

axiom condition_1 : vector_length OA = 1
axiom condition_2 : vector_length OB = 1
axiom condition_3 : vector_length OC = sqrt 3
axiom condition_4 : tan (angle_between OA OC) = 3
axiom condition_5 : angle_between OB OC = 60

theorem vector_combination :
  ∃ (m n : ℝ), OC = m • OA + n • OB ∧ m = (3 + 3*sqrt 3) / 8 ∧ n = (1 + 3*sqrt 3) / 8 := sorry

end vector_combination_l43_43473


namespace math_problem_l43_43820

theorem math_problem (f_star f_ast : ℕ → ℕ → ℕ) (h₁ : f_star 20 5 = 15) (h₂ : f_ast 15 5 = 75) :
  (f_star 8 4) / (f_ast 10 2) = (1:ℚ) / 5 := by
  sorry

end math_problem_l43_43820


namespace circumcircle_radius_of_triangle_l43_43784

noncomputable def circumcircle_radius (a : ℝ) (A : ℝ) : ℝ :=
  a / (2 * Real.sin A)

theorem circumcircle_radius_of_triangle (a A : ℝ) (h₁ : a = 2) (h₂ : A = Real.pi * 2 / 3) :
  circumcircle_radius a A = 2 * Real.sqrt 3 / 3 :=
by 
  rw [circumcircle_radius, h₁, h₂, Real.sin_pi_mul_two_thirds]
  simp
  sorry

end circumcircle_radius_of_triangle_l43_43784


namespace gcd_765432_654321_l43_43170

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43170


namespace original_function_l43_43002

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l43_43002


namespace gcd_765432_654321_l43_43163

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43163


namespace find_complex_number_l43_43393

theorem find_complex_number (Z : ℂ) (hz : (Z.conj / (1 + I)) = (2 + I)) :
  Z = 1 - 3 * I :=
by {
  sorry
}

end find_complex_number_l43_43393


namespace remaining_volume_of_cube_l43_43863

theorem remaining_volume_of_cube (s : ℝ) (r : ℝ) (h : ℝ) (hs : s = 6) (hr : r = 3) (hh : h = 6) : 
  s^3 - (π * r^2 * h) = 216 - 54 * π :=
by
  rw [hs, hr, hh]
  norm_num
  rw [pow_succ, pow_succ]
  norm_num
  ring

end remaining_volume_of_cube_l43_43863


namespace coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l43_43736

-- Definition of the conditions
def condition_1 (Px: ℝ) (Py: ℝ) : Prop := Px = 0

def condition_2 (Px: ℝ) (Py: ℝ) : Prop := Py = Px + 3

def condition_3 (Px: ℝ) (Py: ℝ) : Prop := 
  abs Py = 2 ∧ Px > 0 ∧ Py < 0

-- Proof problem for condition 1
theorem coordinate_P_condition_1 : ∃ (Px Py: ℝ), condition_1 Px Py ∧ Px = 0 ∧ Py = -7 := 
  sorry

-- Proof problem for condition 2
theorem coordinate_P_condition_2 : ∃ (Px Py: ℝ), condition_2 Px Py ∧ Px = 10 ∧ Py = 13 :=
  sorry

-- Proof problem for condition 3
theorem coordinate_P_condition_3 : ∃ (Px Py: ℝ), condition_3 Px Py ∧ Px = 5/2 ∧ Py = -2 :=
  sorry

end coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l43_43736


namespace maximize_profit_l43_43619

noncomputable def y_expression (x : ℝ) : ℝ :=
  -10 * x + 600

def daily_sales_profit (x : ℝ) : ℝ :=
  (x - 30) * (y_expression x)

theorem maximize_profit :
  ∃ x : ℝ, 30 ≤ x ∧ x < 60 ∧ ∀ x' : ℝ, 30 ≤ x' ∧ x' < 60 → daily_sales_profit x' ≤ daily_sales_profit 45 ∧ daily_sales_profit 45 = 2250 :=
by
  sorry

end maximize_profit_l43_43619


namespace largest_possible_n_l43_43304

theorem largest_possible_n :
  ∃ (n : ℕ), 
    (∀ d ∈ (List.range (n + 1)).filter (λ x, n % x = 0), d ≠ 1 ∧ d ≠ n → (d = 101 ∨ d = n / 101)) ∧
    (List.range (n + 1)).filter (λ x, n % x = 0) = [1] ++ [101] ++ [n/101] ++ [n] ∧
    n = 101^3 :=
by
  sorry

end largest_possible_n_l43_43304


namespace valid_grid_iff_divisible_by_9_l43_43255

-- Definitions for the letters used in the grid
inductive Letter
| I
| M
| O

-- Function that captures the condition that each row and column must contain exactly one-third of each letter
def valid_row_col (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ row, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ col, grid row col ∈ [Letter.I, Letter.M, Letter.O])) ∧
  ∀ col, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ row, grid row col ∈ [Letter.I, Letter.M, Letter.O]))

-- Function that captures the condition that each diagonal must contain exactly one-third of each letter when the length is a multiple of 3
def valid_diagonals (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ k, (3 ∣ k → (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = k / 3 ∧ count_M = k / 3 ∧ count_O = k / 3 ∧
    ((∀ (i j : ℕ), (i + j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]) ∨
     (∀ (i j : ℕ), (i - j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]))))

-- The main theorem stating that if we can fill the grid according to the rules, then n must be a multiple of 9
theorem valid_grid_iff_divisible_by_9 (n : ℕ) :
  (∃ grid : ℕ → ℕ → Letter, valid_row_col n grid ∧ valid_diagonals n grid) ↔ 9 ∣ n :=
by
  sorry

end valid_grid_iff_divisible_by_9_l43_43255


namespace sum_of_arcs_approaches_semi_circumference_l43_43106

-- Define the original circle's diameter D as a positive real number
variable (D : ℝ) (hD : 0 < D)

-- Define a function to sum the arcs of the smaller semicircles as n approaches infinity
def arc_sum_as_n_large (n : ℕ) : ℝ :=
  3 * (↑n * (π * D / (6 * ↑n)))

theorem sum_of_arcs_approaches_semi_circumference (n : ℕ) (hn : 0 < n) :
  filter.tendsto (arc_sum_as_n_large D) filter.at_top (nhds (π * D / 2)) :=
begin
  -- The proof will show that the sum of the lengths of the arcs
  -- approaches the semi-circumference as n becomes very large.
  sorry
end

end sum_of_arcs_approaches_semi_circumference_l43_43106


namespace flag_movement_distance_l43_43986

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l43_43986


namespace percentage_chromium_new_alloy_approx_l43_43466

-- Definitions from the problem conditions
def weight_alloy1 : ℝ := 20
def weight_alloy2 : ℝ := 35
def percent_chromium_alloy1 : ℝ := 0.12
def percent_chromium_alloy2 : ℝ := 0.08

-- The percentage of chromium in the new alloy formed
theorem percentage_chromium_new_alloy_approx : 
  ((percent_chromium_alloy1 * weight_alloy1 + percent_chromium_alloy2 * weight_alloy2) / 
  (weight_alloy1 + weight_alloy2)) * 100 ≈ 9.45 := 
by
  sorry

end percentage_chromium_new_alloy_approx_l43_43466


namespace village_population_after_events_l43_43948

theorem village_population_after_events (initial_population : ℕ)
  (death_rate : ℝ) (fear_leave_rate : ℝ) :
  initial_population = 4400 ∧ death_rate = 0.05 ∧ fear_leave_rate = 0.15 →
  let died_by_bombardment := (death_rate * initial_population).to_nat in
  let remaining_after_bombardment := initial_population - died_by_bombardment in
  let left_due_to_fear := (fear_leave_rate * remaining_after_bombardment).to_nat in
  remaining_after_bombardment - left_due_to_fear = 3553 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2, h3]
  let died_by_bombardment := (0.05 * 4400).to_nat
  have h_died_by_bombardment : died_by_bombardment = 220 := by norm_num
  rw [h_died_by_bombardment]
  let remaining_after_bombardment := 4400 - 220
  have h_remaining_after_bombardment : remaining_after_bombardment = 4180 := by norm_num
  rw [h_remaining_after_bombardment]
  let left_due_to_fear := (0.15 * 4180).to_nat
  have h_left_due_to_fear : left_due_to_fear = 627 := by norm_num
  rw [h_left_due_to_fear]
  norm_num
  sorry

end village_population_after_events_l43_43948


namespace solve_for_t_l43_43363

theorem solve_for_t (t : ℝ) : abs (1 + 2 * t * complex.i) = 5 ↔ t = real.sqrt 6 ∨ t = -real.sqrt 6 := 
sorry

end solve_for_t_l43_43363


namespace seedlings_planted_by_father_l43_43849

theorem seedlings_planted_by_father (remi_day1_seedlings : ℕ) (total_seedlings : ℕ) :
  remi_day1_seedlings = 200 →
  total_seedlings = 1200 →
  let remi_day2_seedlings := 2 * remi_day1_seedlings in
  total_seedlings = remi_day1_seedlings + remi_day2_seedlings + 600 :=
begin
  assume h1 h2,
  sorry,
end

end seedlings_planted_by_father_l43_43849


namespace remaining_pipes_l43_43907

def triangular_pyramid_pipes (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem remaining_pipes : ∃ (n : ℕ), 200 - triangular_pyramid_pipes n = 10 :=
by
  use 19
  have Sn_19 : triangular_pyramid_pipes 19 = 190 := by
    simp [triangular_pyramid_pipes]
  rw Sn_19
  norm_num

end remaining_pipes_l43_43907


namespace seats_to_remove_l43_43272

theorem seats_to_remove (rows seats people : ℕ)
  (h1 : rows = 8)
  (h2 : seats = 240)
  (h3 : people = 150)
  (h4 : seats_needed := (8 * ((people + 7) // 8)))
  (h5 : seats_to_remove := seats - seats_needed) :
  seats_to_remove = 88 :=
by
  sorry

end seats_to_remove_l43_43272


namespace lattice_points_bound_l43_43046

def maximumLatticePointsOnCircle (r : ℝ) : ℕ := sorry

theorem lattice_points_bound (r : ℝ) (h : ℕ) :
  h = maximumLatticePointsOnCircle r →
  h < 6 * (3 * real.pi * r^2)^(1/3) :=
sorry

end lattice_points_bound_l43_43046


namespace rate_of_milk_per_litre_l43_43612

theorem rate_of_milk_per_litre (
    let volume_milk : ℝ := 60,
    let volume_water : ℝ := 15,
    let total_volume : ℝ := volume_milk + volume_water,
    let mixture_value_per_litre : ℝ := 32 / 3,
    let total_value_mixture : ℝ := total_volume * mixture_value_per_litre,
    let total_value_milk : ℝ := volume_milk * cost_per_litre
) : ∃ (rate_of_milk_per_litre : ℝ), 
    total_value_milk = total_value_mixture ∧ 
    rate_of_milk_per_litre = 800 / 60 :=
by
  sorry

end rate_of_milk_per_litre_l43_43612


namespace sausage_cut_length_l43_43666

-- Define the length of the sausage
def sausage_length : ℝ := 8

-- Define the number of pieces
def num_pieces : ℝ := 12

-- Define the expected length of each piece should be derived from these conditions
def length_of_each_piece (s : ℝ) (n : ℝ) : ℝ := s / n

-- The main statement we want to prove
theorem sausage_cut_length :
  length_of_each_piece sausage_length num_pieces = 0.6667 := 
  sorry

end sausage_cut_length_l43_43666


namespace sequence_uniquely_determined_l43_43084

theorem sequence_uniquely_determined (a : ℕ → ℝ) (p q : ℝ) (a0 a1 : ℝ)
  (h : ∀ n, a (n + 2) = p * a (n + 1) + q * a n)
  (h0 : a 0 = a0)
  (h1 : a 1 = a1) :
  ∀ n, ∃! a_n, a n = a_n :=
sorry

end sequence_uniquely_determined_l43_43084


namespace problem_solution_l43_43411

def f : ℕ → ℕ
| 1 := 4
| 2 := 3
| 3 := 2
| 4 := 5
| 5 := 1
| _ := 0  -- assuming f is only defined for the domain {1, 2, 3, 4, 5}.

-- Assume that f has an inverse function.
noncomputable def f_inv : ℕ → ℕ
| 4 := 1
| 3 := 2
| 2 := 3
| 5 := 4
| 1 := 5
| _ := 0  -- similarly, restrict the range for practical purposes.

theorem problem_solution :
  f_inv (f_inv (f_inv 3)) = 2 :=
by {
  have h1 : f_inv (3) = 2, from rfl,
  have h2 : f_inv (2) = 3, from rfl,
  have h3 : f_inv (3) = 2, from rfl,
  rw [h1, h2, h3],
  exact rfl,
}

end problem_solution_l43_43411


namespace marjorie_cakes_l43_43072

variable (n : ℤ)

def cakes_made_day_1 := n
def cakes_made_day_2 := 2 * n
def cakes_made_day_3 := 3 * n
def cakes_made_day_4 := (3 * n) + (3 * n / 4)
def cakes_made_day_5 := (3.75 * n) + (3.75 * n / 4)
def cakes_made_day_6 := (4.6875 * n) + (4.6875 * n / 4)

theorem marjorie_cakes (h : cakes_made_day_6 n = 450) : n = 76 :=
by {
  -- Solution goes here
  sorry
}

end marjorie_cakes_l43_43072


namespace tensor_op_correct_l43_43997

-- Define the operation ⊗
def tensor_op (x y : ℝ) : ℝ := x^2 + y

-- Goal: Prove h ⊗ (h ⊗ h) = 2h^2 + h for some h in ℝ
theorem tensor_op_correct (h : ℝ) : tensor_op h (tensor_op h h) = 2 * h^2 + h :=
by
  sorry

end tensor_op_correct_l43_43997


namespace part_a_arrangement_part_b_i_n_cannot_be_11_part_b_ii_n_is_10_part_c_n_is_13_l43_43944

-- Proof statement for part (a)
theorem part_a_arrangement : ∃ (l : list ℕ), l = [1, 3, 5, 2, 4] ∧ 
  (∀ (i : ℕ), i < l.length - 1 → |l.nth_le i _ - l.nth_le (i+1) _| = 2) :=
begin
  sorry
end

-- Proof statement for part (b)(i)
theorem part_b_i_n_cannot_be_11 (N : ℕ) (h_arr : ∀ (a : ℕ), a ∈ [1,2,...,20] → |a - b| ≥ N) : 
  N ≤ 10 :=
begin
  sorry
end

-- Proof statement for part (b)(ii)
theorem part_b_ii_n_is_10 : ∃ (l : list ℕ), l = [10, 20, 9, 19, 8, 18, 7, 17, 6, 16, 5, 15, 4, 14, 3, 13, 2, 12, 1, 11] ∧ 
  (∀ (i : ℕ), i < l.length - 1 → |l.nth_le i _ - l.nth_le (i+1) _| <= 10) :=
begin
  sorry
end

-- Proof statement for part (c)
theorem part_c_n_is_13 : ∃ (l : list ℕ), l = [14, 27, 13, 26, 12, 25, 11, 24, 10, 23, 9, 22, 8, 21, 7, 20, 6, 19, 5, 18, 4, 17, 3, 16, 2, 15, 1] ∧ 
  (∀ i, i < l.length - 1 → |l.nth_le i _ - l.nth_le (i+1) _| = 13) :=
begin
  sorry
end

end part_a_arrangement_part_b_i_n_cannot_be_11_part_b_ii_n_is_10_part_c_n_is_13_l43_43944


namespace largest_in_set_l43_43380

theorem largest_in_set (a : ℝ) (h : a = 3) :
  let s := {-3 * a, 2 * a, 18 / a, a^2, 1}
  in ∀ s', s' ∈ s → s' ≤ a^2 :=
begin
  sorry
end

end largest_in_set_l43_43380


namespace keith_placed_scissors_l43_43913

theorem keith_placed_scissors :
  ∀ (initial final placed : ℕ), initial = 54 → final = 76 → placed = final - initial → placed = 22 :=
by
  intros initial final placed h_initial h_final h_placed
  rw [h_initial, h_final, h_placed]
  simp
  sorry

end keith_placed_scissors_l43_43913


namespace find_b2_l43_43880

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 25) (h10 : b 10 = 105) 
  (h : ∀ n, n ≥ 4 → b n = (b 1 + b 2 + (∑ i in finset.range (n - 2), b (i + 3) )) / (n - 1)) : 
  b 2 = 176 :=
sorry

end find_b2_l43_43880


namespace friends_gcd_l43_43945

theorem friends_gcd {a b : ℤ} (h : ∃ n : ℤ, a * b = n * n) : 
  ∃ m : ℤ, a * Int.gcd a b = m * m :=
sorry

end friends_gcd_l43_43945


namespace more_apples_obtained_l43_43291

-- The given conditions
variables (P : ℝ) (total_spent original_price reduced_price : ℝ)
variable (price_reduction : ℝ -> ℝ)
variable (calculate_dozens : ℝ -> ℝ -> ℝ)

-- The equivalences derived from problem
def reduction_equation : Prop := reduced_price = price_reduction P
def original_price_equation : Prop := original_price = Rs 2 / price_reduction P
def dozen_difference_equation : Prop := (total_spent / reduced_price) - (total_spent / original_price) = 4.5

-- The total spent and the effective difference in apples
noncomputable def total_apples_difference : ℝ := 4.5 * 12

-- Theorem to prove the number of more apples obtained after price reduction
theorem more_apples_obtained (h₁ : reduced_price = Rs 2)
                             (h₂ : total_spent = Rs 30)
                             (h₃ : price_reduction P = 0.70)
                             (h₄ : total_apples_difference = 54) :
  (4.5 * 12) = 54 :=
begin
  sorry
end

end more_apples_obtained_l43_43291


namespace area_increases_l43_43801

structure Polygon (n : ℕ) :=
(vertices : Fin n → ℝ × ℝ)
(sides_non_parallel : ∀ i j : Fin n, i ≠ j → vector_angle (vertices i) (vertices j) ≠ 0)

def is_non_adjacent (n : ℕ) (A B : Fin n) : Prop :=
  ∃ k : ℕ, 2 < (n - k) ∧ (A.val + k) % n = B.val

def is_non_convex (p : Polygon n) : Prop :=
  ¬ ∃ line : ℝ → ℝ, ∀ (i : Fin n), (line (p.vertices i).1 ≤ (p.vertices i).2)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def reflect (p : Polygon n) (A B : Fin n) (line : List (Fin n)) : Polygon n :=
  sorry

theorem area_increases (p : Polygon n) (A B : Fin n) (non_adjacent: is_non_adjacent n A B)
  (non_convex : is_non_convex p) : 
  ∃ q : Polygon n, (reflect p A B (List.range n)) ≠ p ∧ is_non_convex q :=
sorry

end area_increases_l43_43801


namespace adjusted_target_heart_rate_l43_43284

theorem adjusted_target_heart_rate (age : ℕ) (recovery_heart_rate : ℕ) (max_heart_rate : ℕ) 
(target_heart_rate : ℕ) (adjusted_target : ℕ) : 
  age = 30 -> 
  recovery_heart_rate = 120 -> 
  max_heart_rate = 220 - age -> 
  target_heart_rate = 0.7 * max_heart_rate -> 
  adjusted_target = target_heart_rate + 0.1 * target_heart_rate -> 
  146 = Int.to_nat (Int.ofReal (Real.round adjusted_target)) :=
sorry

end adjusted_target_heart_rate_l43_43284


namespace PQRS_perimeter_correct_l43_43021

noncomputable def PQRS_perimeter (PQ QR RS : ℝ) (PQ_pos : 0 < PQ) (QR_pos : 0 < QR) (RS_pos : 0 < RS) : ℝ :=
  PQ + QR + RS + Real.sqrt (PQ^2 + QR^2 + RS^2 + (PQ^2 + QR^2))

theorem PQRS_perimeter_correct :
  ∀ (PQ QR RS : ℝ),
    PQ = 24 →
    QR = 28 →
    RS = 16 →
    PQRS_perimeter PQ QR RS 0 0 0 = 68 + Real.sqrt 1616 := by
  sorry

end PQRS_perimeter_correct_l43_43021


namespace triangle_area_of_tangent_line_l43_43416

theorem triangle_area_of_tangent_line
  (f : ℝ → ℝ)
  (P : ℝ × ℝ)
  (hf : f = λ x, x^3 - 2 * x^2 + x + 6)
  (hP : P = (-1, 2)) :
  let f' := λ x, 3 * x^2 - 4 * x + 1 in
  let slope := f' (-1) in
  let tangent_line := λ x, slope * (x + 1) + 2 in
  let x_intercept := -5 / 4 in
  let y_intercept := 10 in
  let base := abs x_intercept in
  let height := y_intercept in
  (1 / 2) * base * height = 25 / 4 :=
by
  sorry

end triangle_area_of_tangent_line_l43_43416


namespace final_amount_l43_43026

-- Definitions for the initial amount, price per pound, and quantity purchased.
def initial_amount : ℕ := 20
def price_per_pound : ℕ := 2
def quantity_purchased : ℕ := 3

-- Formalizing the statement
theorem final_amount (A P Q : ℕ) (hA : A = initial_amount) (hP : P = price_per_pound) (hQ : Q = quantity_purchased) :
  A - P * Q = 14 :=
by
  sorry

end final_amount_l43_43026


namespace alice_distance_from_start_l43_43303

noncomputable def alice_displacement : ℝ :=
let east_meters : ℝ := 15,
    east_meters_in_feet : ℝ := east_meters * 3.28,
    north_feet : ℝ := 50,
    west_feet : ℝ := 15 * 3.28 + 10
in
real.sqrt ((east_meters_in_feet - west_feet) ^ 2 + north_feet ^ 2)

theorem alice_distance_from_start : alice_displacement = 51 :=
by
  sorry

end alice_distance_from_start_l43_43303


namespace middle_number_is_10_l43_43147

theorem middle_number_is_10 (x y z : ℤ) (hx : x < y) (hy : y < z) 
    (h1 : x + y = 18) (h2 : x + z = 25) (h3 : y + z = 27) : y = 10 :=
by 
  sorry

end middle_number_is_10_l43_43147


namespace pentagon_perimeter_ratio_l43_43123

-- Define the given parameters and conditions

def equilateral_triangle (a : ℝ) : Prop := ∀ b, b = 3 * a

def congruent_isosceles_triangles (perimeter : ℝ) : Prop := ∀ b, b = perimeter

def sum_perimeters (a : ℝ) (perimeter1 perimeter2 perimeter3 perimeter4 : ℝ) : ℝ :=
  perimeter1 + perimeter2 + perimeter3 + perimeter4

-- Define the problem statement
theorem pentagon_perimeter_ratio (a : ℝ) (PQRST : ℝ) (PQR : ℝ) (PTU : ℝ) (SUT : ℝ) (RSU : ℝ) :
  (equilateral_triangle a) →
  (congruent_isosceles_triangles (3 * a) PTU) →
  (congruent_isosceles_triangles (3 * a) SUT) →
  (congruent_isosceles_triangles (3 * a) RSU) →
  (sum_perimeters a PQR PTU SUT RSU = PQRST) →
  (PQRST / PQR = (5 : ℝ) / (3 : ℝ)) :=
by
  sorry

end pentagon_perimeter_ratio_l43_43123


namespace number_of_connections_l43_43146

theorem number_of_connections (n m : ℕ) (h1 : n = 30) (h2 : m = 4) :
    (n * m) / 2 = 60 := by
  -- Since each switch is connected to 4 others,
  -- and each connection is counted twice, 
  -- the number of unique connections is 60.
  sorry

end number_of_connections_l43_43146


namespace ellipse_eccentricity_l43_43407

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c^2 = a^2 - b^2)
  (h4 : let F := (-c, 0) in let A := (F.1 / 2, (sqrt 3 / 2) * F.1) in (A.1^2 / a^2) + (A.2^2 / b^2) = 1)
  : ∃ e : ℝ, e = sqrt 3 - 1 :=
by
  sorry

end ellipse_eccentricity_l43_43407


namespace alpha_beta_sum_l43_43909

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, x ≠ 54 → x ≠ -60 → (x - α) / (x + β) = (x^2 - 72 * x + 945) / (x^2 + 45 * x - 3240)) :
  α + β = 81 :=
sorry

end alpha_beta_sum_l43_43909


namespace gcd_765432_654321_eq_3_l43_43222

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43222


namespace count_SSn_eq_n_l43_43438

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def S (n : ℕ) : ℕ :=
  let a := 10^(floor (Real.log n / Real.log 10))
  let b := n / a
  let c := n - a * b
  b + 10 * c

theorem count_SSn_eq_n : (Finset.range 2012).filter (λ n, S (S n) = n).card = 108 := by sorry

end count_SSn_eq_n_l43_43438


namespace gcd_765432_654321_l43_43177

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43177


namespace min_value_frac_eq_nine_halves_l43_43421

theorem min_value_frac_eq_nine_halves {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 2) :
  ∃ (x y : ℝ), 2 / x + 1 / y = 9 / 2 := by
  sorry

end min_value_frac_eq_nine_halves_l43_43421


namespace cube_volume_increase_l43_43927

theorem cube_volume_increase (s : ℝ) (h : s > 0) :
  let new_edge_length := 1.6 * s
  let original_volume := s^3
  let new_volume := new_edge_length^3
  (new_volume - original_volume) / original_volume * 100 = 309.6 :=
by
  let new_edge_length := 1.6 * s
  let original_volume := s^3
  let new_volume := new_edge_length^3
  hope_addition data_type
  sorry

end cube_volume_increase_l43_43927


namespace shaded_region_area_l43_43052

theorem shaded_region_area :
  ∀ (RECT : Type) (T E : RECT) (ET ER : ℝ), 
  (ET = 2) → 
  (ER = 6) →
  let RT := Real.sqrt(ET^2 + ER^2)
  let radius := RT
  let circle_area := π * radius^2
  let rectangle_area := ER * ET
  let shaded_area := (1/4) * circle_area - rectangle_area
  shaded_area = 10 * π - 12 :=
by 
  intros RECT T E ET ER hET hER RT radius circle_area rectangle_area shaded_area
  sorry

end shaded_region_area_l43_43052


namespace find_function_form_l43_43375

theorem find_function_form (f : ℝ → ℝ) :
  (∀ x : ℝ, f (real.sqrt x + 1) = x + 2 * real.sqrt x) → 
  (∀ x : ℝ, x ≥ 1 → f x = x^2 - 1) :=
by
  intro h x hx
  sorry

end find_function_form_l43_43375


namespace tan_phi_computation_l43_43460

def tan_phi (phi : Real) : Real :=
  let sq6 := Real.sqrt 6
  let expr := (5 - 2 * sq6) / (5 + 2 * sq6)
  Real.sqrt expr

theorem tan_phi_computation (beta phi : Real) (h1 : Real.sin beta = 1 / Real.sqrt 3)
  (h2 : phi = beta / 2) :
  Real.tan phi = tan_phi phi := by
  sorry

end tan_phi_computation_l43_43460


namespace normal_distribution_probability_l43_43387

noncomputable def normal_distribution_X : MeasureTheory.Measure ℝ :=
  MeasureTheory.Measure.gaussian 1 σ^2

def P_X_le_0 : ℝ :=
  MeasureTheory.Measure.measure_of normal_distribution_X (λ x, x ≤ 0)

def P_X_ge_1 : ℝ :=
  MeasureTheory.Measure.measure_of normal_distribution_X (λ x, x ≥ 1)

def P_1_le_X_le_2 : ℝ :=
  MeasureTheory.Measure.measure_of normal_distribution_X (λ x, 1 ≤ x ∧ x ≤ 2)

theorem normal_distribution_probability (σ : ℝ) (h : P_X_le_0 = 0.1) : 
  P_1_le_X_le_2 = 0.4 := by
  sorry

end normal_distribution_probability_l43_43387


namespace value_of_x_l43_43390

-- Define the propositions
def p (x : ℝ) : Prop := x^2 - 3 * x - 4 ≠ 0
def q (x : ℝ) : Prop := x ∈ Set ℕ \ {0}

-- The final statement, we need to prove that x = 4 given the conditions
theorem value_of_x (h1 : ¬ (p x ∧ q x)) (h2 : ¬ (¬ q x)) : x = 4 :=
sorry

end value_of_x_l43_43390


namespace volume_increase_by_eight_l43_43127

theorem volume_increase_by_eight (r : ℝ) (V : ℝ) (hV : V = (4 / 3) * π * r^3) :
  let V_new := (4 / 3) * π * (2 * r)^3 in
  V_new = 8 * V :=
by
  sorry

end volume_increase_by_eight_l43_43127


namespace alpha_perpendicular_beta_l43_43361

variable {m n : Type}
variables {α β : set m} [vector_space m n] [affine_space m n]

-- Definitions from conditions
def line_parallel (l1 l2 : set m) := l1 ∥ l2
def line_perpendicular (l1 l2 : set m) := l1 ⟂ l2
def line_in_plane (l : set m) (p : set m) := l ⊆ p
def plane_perpendicular (p1 p2 : set m) := p1 ⟂ p2

-- Given conditions
variable (h1 : line_parallel m n)
variable (h2 : line_in_plane m α)
variable (h3 : line_perpendicular n β)

-- Statement to be proved
theorem alpha_perpendicular_beta :
  plane_perpendicular α β :=
sorry

end alpha_perpendicular_beta_l43_43361


namespace problem1_problem2_l43_43686

-- Definition for sine and cosine values
def sin_30_eq_cos_60 : Real := sin (Real.pi / 6) + cos (Real.pi / 3)

-- Theorem for the first problem
theorem problem1 : sin_30_eq_cos_60 = 1 := by
  sorry

-- Definitions for variables and equation
def quadratic_eqn (x : Real) := x^2 - 4 * x - 12

-- Theorem for the second problem
theorem problem2 : ∃ x : Real, quadratic_eqn x = 0 ∧ (x = 6 ∨ x = -2) := by
  sorry

end problem1_problem2_l43_43686


namespace expected_value_is_4_5_l43_43591

-- Define the probabilities for each face of the biased die
def p1 : ℝ := 1 / 10
def p2 : ℝ := 1 / 10
def p3 : ℝ := 1 / 10
def p4 : ℝ := 1 / 10
def p5 : ℝ := 1 / 10
def p6 : ℝ := 1 / 2

-- Define faces of the die
def faces : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the EV calculation
noncomputable def expected_value : ℝ := 
  p1 * faces.head! + 
  p2 * faces.nth! 1 + 
  p3 * faces.nth! 2 + 
  p4 * faces.nth! 3 + 
  p5 * faces.nth! 4 + 
  p6 * faces.nth! 5

theorem expected_value_is_4_5 : expected_value = 4.5 := by
  sorry

end expected_value_is_4_5_l43_43591


namespace radius_of_inscribed_circle_l43_43018

theorem radius_of_inscribed_circle (r1 r2 : ℝ) (AC BC AB : ℝ) 
  (h1 : AC = 2 * r1)
  (h2 : BC = 2 * r2)
  (h3 : AB = 2 * Real.sqrt (r1^2 + r2^2)) : 
  (r1 + r2 - Real.sqrt (r1^2 + r2^2)) = ((2 * r1 + 2 * r2 - 2 * Real.sqrt (r1^2 + r2^2)) / 2) := 
by
  sorry

end radius_of_inscribed_circle_l43_43018


namespace number_of_ordered_pairs_l43_43498

theorem number_of_ordered_pairs : 
  ∃ (a b : ℤ), (zeta : ℂ) (zeta_nonreal_root : ∀ x : ℂ, ((x^4 = 1) ∧ ¬(x.re = 0 ∧ x.im = 0)) → (|a * zeta + b| = 1)) 
  → (multiset.card (multiset.filter (λ ab : ℤ × ℤ, |(ab.1 : ℂ) * zeta + ab.2| = 1) (multiset.finset_ℤ × multiset.finset_ℤ)) = 4) :=
sorry

end number_of_ordered_pairs_l43_43498


namespace train_speed_proof_l43_43951

noncomputable def train_speed (L : ℕ) (t : ℝ) (v_m : ℝ) : ℝ :=
  let v_m_m_s := v_m * (1000 / 3600)
  let v_rel := L / t
  v_rel + v_m_m_s

theorem train_speed_proof
  (L : ℕ)
  (t : ℝ)
  (v_m : ℝ)
  (hL : L = 900)
  (ht : t = 53.99568034557235)
  (hv_m : v_m = 3)
  : train_speed L t v_m = 63.0036 :=
  by sorry

end train_speed_proof_l43_43951


namespace solve_system_of_equations_l43_43538

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2^(x + 2*y) + 2^x = 3 * 2^y) ∧ (2^(2*x + y) + 2 * 2^y = 4 * 2^x) ∧ (x = 1 / 2) ∧ (y = 1 / 2) := 
by
  let x := (1:ℝ) / 2
  let y := (1:ℝ) / 2
  have h1 : 2^(x + 2*y) + 2^x = 3 * 2^y := sorry
  have h2 : 2^(2*x + y) + 2 * 2^y = 4 * 2^x := sorry
  exact ⟨x, y, h1, h2, rfl, rfl⟩

end solve_system_of_equations_l43_43538


namespace jenny_reading_time_l43_43485

theorem jenny_reading_time 
  (days : ℕ)
  (words_first_book : ℕ)
  (words_second_book : ℕ)
  (words_third_book : ℕ)
  (reading_speed : ℕ) : 
  days = 10 →
  words_first_book = 200 →
  words_second_book = 400 →
  words_third_book = 300 →
  reading_speed = 100 →
  (words_first_book + words_second_book + words_third_book) / reading_speed / days * 60 = 54 :=
by
  intros hdays hwords1 hwords2 hwords3 hspeed
  rw [hdays, hwords1, hwords2, hwords3, hspeed]
  norm_num
  sorry

end jenny_reading_time_l43_43485


namespace rachel_milk_amount_l43_43700

theorem rachel_milk_amount : 
  let don_milk := (3 : ℚ) / 7
  let rachel_fraction := 4 / 5
  let rachel_milk := rachel_fraction * don_milk
  rachel_milk = 12 / 35 :=
by sorry

end rachel_milk_amount_l43_43700


namespace rhombus_area_250_l43_43864

variable (d1 d2 : ℝ)
variable (area : ℝ)
variable (d1_val d2_val : ℝ)
variable (area_val : ℝ)

-- Conditions
def rhombus.diagonal1 := d1 = d1_val
def rhombus.diagonal2 := d2 = d2_val
def rhombus.area_formula := area = (d1 * d2) / 2

-- Problem Statement
theorem rhombus_area_250 (hd1 : rhombus.diagonal1 20) (hd2 : rhombus.diagonal2 25) : rhombus.area_formula 250 := by
  sorry

end rhombus_area_250_l43_43864


namespace stratified_random_sampling_sum_square_l43_43401

def sample_mean (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1 / n) * (Finset.sum (Finset.range n) (λ i, x i))

def sample_variance (x : ℕ → ℝ) (n : ℕ) (mean : ℝ) : ℝ :=
  (1 / n) * (Finset.sum (Finset.range n) (λ i, (x i - mean) ^ 2))

theorem stratified_random_sampling_sum_square 
  (n : ℕ) (x : ℕ → ℝ) (ω : ℝ) (mean : ℝ) (variance : ℝ) 
  (h_mean : mean = sample_mean x n)
  (h_variance : variance = sample_variance x n mean):
  (Finset.sum (Finset.range n) (λ i, (x i - ω) ^ 2)) = 
    n * variance + n * (mean - ω) ^ 2 := by
  sorry

end stratified_random_sampling_sum_square_l43_43401


namespace solve_x_l43_43773

theorem solve_x : ∃ x : ℝ, 2^(Real.log 5 / Real.log 2) = 3 * x + 4 ∧ x = 1 / 3 :=
by
  use 1 / 3
  sorry

end solve_x_l43_43773


namespace isosceles_triangle_of_perpendiculars_intersect_at_one_point_l43_43085

theorem isosceles_triangle_of_perpendiculars_intersect_at_one_point
  (ABC : Type)
  (a b c : ℝ)
  (BC : | BC - a | ≤ 0)
  (CA : | CA - b | ≤ 0)
  (AB : | AB - c | ≤ 0)
  (BD DC b1 b2 c1 c2 : ℝ)
  (h1 : BD = (a * c) / (b + c))
  (h2 : DC = (a * b) / (a + b))
  (h3 : CE = (b ^ 2) / (a + c))
  (h4 : EA = (b * c) / (b + c))
  (h5 : AF = (c ^ 2) / (a + b))
  (h6 : FB = (a * c) / (a + c))
  (perpendiculars_intersect : 
    (a * c / (b + c)) ^ 2 + (a * b / (a + c)) ^ 2 + (b * c / (a + b)) ^ 2 = 
    (a * b / (b + c)) ^ 2 + (b * c / (a + c)) ^ 2 + (a * c / (a + b)) ^ 2) :
  (b = c) ∨ (a = b) ∨ (a = c) := by
  sorry

end isosceles_triangle_of_perpendiculars_intersect_at_one_point_l43_43085


namespace people_present_l43_43914

-- Define the number of parents, pupils, and teachers as constants
def p := 73
def s := 724
def t := 744

-- The theorem to prove the total number of people present
theorem people_present : p + s + t = 1541 := 
by
  -- Proof is inserted here
  sorry

end people_present_l43_43914


namespace fraction_value_l43_43237

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end fraction_value_l43_43237


namespace correct_statement_l43_43665

def correct_input_format_1 (s : String) : Prop :=
  s = "INPUT a, b, c"

def correct_input_format_2 (s : String) : Prop :=
  s = "INPUT x="

def correct_output_format_1 (s : String) : Prop :=
  s = "PRINT A="

def correct_output_format_2 (s : String) : Prop :=
  s = "PRINT 3*2"

theorem correct_statement : (correct_input_format_1 "INPUT a; b; c" = false) ∧
                            (correct_input_format_2 "INPUT x=3" = false) ∧
                            (correct_output_format_1 "PRINT“A=4”" = false) ∧
                            (correct_output_format_2 "PRINT 3*2" = true) :=
by sorry

end correct_statement_l43_43665


namespace club_committee_selections_l43_43952

theorem club_committee_selections : (Nat.choose 18 3) = 816 := by
  sorry

end club_committee_selections_l43_43952


namespace gcd_765432_654321_l43_43186

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43186


namespace number_property_l43_43874

theorem number_property : ∀ n : ℕ, (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ n = 1 ∨ n = 4 :=
by sorry

end number_property_l43_43874


namespace gcd_of_765432_and_654321_l43_43206

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43206


namespace sum_fraction_2023_l43_43530

theorem sum_fraction_2023 : 
  (\sum n in Finset.range 2023, 1 / (n + 1) / (n + 2)) = 2023 / 2024 :=
by
  sorry

end sum_fraction_2023_l43_43530


namespace goldfish_equal_number_after_n_months_l43_43682

theorem goldfish_equal_number_after_n_months :
  ∃ (n : ℕ), 2 * 4^n = 162 * 3^n ∧ n = 6 :=
by
  sorry

end goldfish_equal_number_after_n_months_l43_43682


namespace right_triangle_trigonometry_l43_43344

theorem right_triangle_trigonometry (AB BC : ℝ)
  (hAB : AB = 15) (hBC : BC = 20) :
  let AC := real.sqrt (AB^2 + BC^2) in
  let cos_B := AB / AC in
  let sin_A := AB / AC in
  cos_B = 3 / 5 ∧ sin_A = 3 / 5 :=
by
  sorry

end right_triangle_trigonometry_l43_43344


namespace cylinder_volume_ratio_l43_43996

theorem cylinder_volume_ratio
  (h : ℝ)     -- height of cylinder B (radius of cylinder A)
  (r : ℝ)     -- radius of cylinder B (height of cylinder A)
  (VA : ℝ)    -- volume of cylinder A
  (VB : ℝ)    -- volume of cylinder B
  (cond1 : r = h / 3)
  (cond2 : VB = 3 * VA)
  (cond3 : VB = N * Real.pi * h^3) :
  N = 1 / 3 := 
sorry

end cylinder_volume_ratio_l43_43996


namespace M_squared_is_odd_l43_43499

theorem M_squared_is_odd (a b : ℤ) (h1 : a = b + 1) (c : ℤ) (h2 : c = a * b) (M : ℤ) (h3 : M^2 = a^2 + b^2 + c^2) : M^2 % 2 = 1 := 
by
  sorry

end M_squared_is_odd_l43_43499


namespace minimize_S_n_l43_43732

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

axiom arithmetic_sequence : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
axiom sum_first_n_terms : ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * d)
axiom condition1 : a 0 + a 4 = -14
axiom condition2 : S 9 = -27

theorem minimize_S_n : ∃ n, ∀ m, S n ≤ S m := sorry

end minimize_S_n_l43_43732


namespace sum_binomials_l43_43926

theorem sum_binomials (m n : ℕ) (h : m ≤ n) :
  ∑ r in Finset.range (n + 1), if m ≤ r then 
    (-1)^r * (Nat.choose n r) * (Nat.choose r m) else 0 
  = if m = n then (-1)^m else 0 := by
  sorry

end sum_binomials_l43_43926


namespace percentage_of_chemical_a_l43_43855

-- Define the percentage of chemical a in solutions x and y and the mixture
variables (P : ℝ) (a_x b_x : ℝ) (a_y b_y : ℝ)

-- Define the conditions
def conditions : Prop :=
  a_x = 0.10 ∧ 
  b_x = 0.90 ∧ 
  b_y = 0.80 ∧ 
  0.80 * a_x + 0.20 * P = 0.12

-- The proof goal
theorem percentage_of_chemical_a (h : conditions) : P = 0.20 :=
sorry

end percentage_of_chemical_a_l43_43855


namespace expand_product_l43_43340

theorem expand_product (x : ℝ) : (x + 4) * (x - 7) = x^2 - 3x - 28 :=
by
  sorry

end expand_product_l43_43340


namespace intersection_M_N_l43_43758

-- Definitions of sets M and N
def M : Set ℝ := { y : ℝ | ∃ x : ℝ, y = Real.cos x }
def N : Set ℤ := { x : ℤ | (2 - x) / (1 + x) ≥ 0 }

-- The theorem we want to prove
theorem intersection_M_N : (M ∩ (N : Set ℝ)) = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_M_N_l43_43758


namespace sum_of_triangle_areas_in_cube_l43_43899

theorem sum_of_triangle_areas_in_cube :
  let m : ℤ := 48,
      n : ℤ := 4608,
      p : ℤ := 3072
  in m + n + p = 7728 :=
by
  sorry

end sum_of_triangle_areas_in_cube_l43_43899


namespace circle_equation_l43_43962

theorem circle_equation 
  (a b r : ℝ)
  (h1 : a + b = 0)
  (h2 : (2 - a) ^ 2 + (1 - b) ^ 2 = r ^ 2)
  (h3 : r ^ 2 - ((abs (a - b + 1)) / Real.sqrt 2) ^ 2 = 1 / 2) :
  (x y : ℝ) → (x - 1) ^ 2 + (y + 1) ^ 2 = 5 :=
by {
  assume x y,
  sorry
}

end circle_equation_l43_43962


namespace smallest_number_l43_43923

theorem smallest_number (n : ℕ) : 
  (∃ n, ∀ m : ℕ, (m + 2) % 12 = 0 ∧ (m + 2) % 30 = 0 ∧ (m + 2) % 48 = 0 ∧ (m + 2) % 74 = 0 ∧ (m + 2) % 100 = 0 → n = m → m = 44398) :=
begin
  sorry -- the proof is omitted 
end

end smallest_number_l43_43923


namespace tan_alpha_eq_2_sqrt_2_l43_43743

-- Define the variables and hypotheses
variable (α : ℝ)
hypothesis h1 : Real.cos (Real.pi / 2 + α) = (2 * Real.sqrt 2) / 3
hypothesis h2 : α > Real.pi / 2 ∧ α < 3 * Real.pi / 2

-- The target statement or theorem to be proved
theorem tan_alpha_eq_2_sqrt_2 (α : ℝ) (h1 : Real.cos (Real.pi / 2 + α) = (2 * Real.sqrt 2) / 3) (h2 : α > Real.pi / 2 ∧ α < 3 * Real.pi / 2) :
  Real.tan α = 2 * Real.sqrt 2 :=
by
  sorry  -- Proof to be filled in

end tan_alpha_eq_2_sqrt_2_l43_43743


namespace intersection_of_A_and_B_l43_43424

open Set

def A : Set ℕ := {0, 1, 3}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {3} :=
sorry

end intersection_of_A_and_B_l43_43424


namespace find_x_for_equation_l43_43569

theorem find_x_for_equation 
  (x : ℝ)
  (h : (32 : ℝ)^(x-2) / (8 : ℝ)^(x-2) = (512 : ℝ)^(3 * x)) : 
  x = -4/25 :=
by
  sorry

end find_x_for_equation_l43_43569


namespace f_0_ne_0_l43_43829

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_not_identically_zero : ¬ ∀ x, f(x) = 0

axiom f_pi_over_2_zero : f(Real.pi / 2) = 0

axiom f_functional_eq (x y : ℝ) : f(x) + f(y) = 2 * f((x + y) / 2) * f((x - y) / 2)

theorem f_0_ne_0 : f(0) ≠ 0 :=
sorry

end f_0_ne_0_l43_43829


namespace angle_between_skew_lines_l43_43477

noncomputable def mid_point (x y : ℝ^3) : ℝ^3 := (x + y) / 2

theorem angle_between_skew_lines (A B C D P Q R : ℝ^3):
  P = mid_point A B ∧
  Q = mid_point B C ∧
  R = mid_point C D ∧
  dist P Q = 2 ∧
  dist P R = 3 →
  angle_between (line_through A C) (line_through B D) = π / 2 :=
by
  sorry

end angle_between_skew_lines_l43_43477


namespace angle_RPQ_measure_l43_43025

def angle_measure_degrees_EQ (angle_deg : ℚ) : Prop :=
  P_on_RS : ℝ → Prop := sorry -- placeholder
  QP_bisects_angle_SQR : ℝ → Prop := sorry -- placeholder
  RQ_eq_RP : ℝ → Prop := sorry -- placeholder
  RSQ_eq_3y : ℝ → ℝ → Prop := sorry -- placeholder
  RPQ_eq_4y : ℝ → ℝ → Prop := sorry -- placeholder

theorem angle_RPQ_measure : 
  (∀ P : ℝ, P_on_RS P) →
  (∀ Q : ℝ, ∀ R : ℝ, ∀ S : ℝ, QP_bisects_angle_SQR S) →
  (∀ R P : ℝ, RQ_eq_RP R P) →
  (∀ R S Q : ℝ, y : ℝ, RSQ_eq_3y R S Q y) →
  (∀ R P Q : ℝ, y : ℝ, RPQ_eq_4y R P Q y) →
  angle_measure_degrees_EQ (720 / 7) :=
by
  apply sorry

end angle_RPQ_measure_l43_43025


namespace first_year_after_2020_with_digit_sum_18_l43_43592

theorem first_year_after_2020_with_digit_sum_18 : 
  ∃ (y : ℕ), y > 2020 ∧ (∃ a b c : ℕ, (2 + a + b + c = 18 ∧ y = 2000 + 100 * a + 10 * b + c)) ∧ y = 2799 := 
sorry

end first_year_after_2020_with_digit_sum_18_l43_43592


namespace cost_of_one_stamp_l43_43620

-- Define the cost of one stamp
def stamp_cost (x : ℝ) : Prop :=
  (3 * x = 1.02)

-- The theorem to prove that the cost of one stamp is $0.34
theorem cost_of_one_stamp : ∃ x : ℝ, stamp_cost x ∧ x = 0.34 :=
by
  exists 0.34
  unfold stamp_cost
  simp
  norm_num
  sorry

end cost_of_one_stamp_l43_43620


namespace lattice_points_on_segment_l43_43327

theorem lattice_points_on_segment : 
  let x1 := 5 
  let y1 := 23 
  let x2 := 47 
  let y2 := 297 
  ∃ n, n = 3 ∧ ∀ p : ℕ × ℕ, (p = (x1, y1) ∨ p = (x2, y2) ∨ ∃ t : ℕ, p = (x1 + t * (x2 - x1) / 2, y1 + t * (y2 - y1) / 2)) := 
sorry

end lattice_points_on_segment_l43_43327


namespace gcd_765432_654321_eq_3_l43_43223

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43223


namespace range_of_a_l43_43374

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1/real.exp 1) 1, ∃ x2 ∈ set.Icc 0 1, real.log x1 - x1 + 1 + a = x2^2 * real.exp x2) ->
  1/real.exp 1 < a ∧ a ≤ real.exp 1 :=
by 
  sorry

end range_of_a_l43_43374


namespace triangle_isosceles_l43_43010

theorem triangle_isosceles (A B C : ℝ) (h1 : sin A = sin C) (h2 : 0 < A + C ∧ A + C < π) : 
  A = C ∨ A + C = π - A ∧ A + C = π - C := sorry

end triangle_isosceles_l43_43010


namespace angle_between_vectors_l43_43762

/-- Angle between two vectors given specific conditions -/
theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h₀ : a = (2, 0)) 
  (h₁ : (b.1^2 + b.2^2) = 1) 
  (h₂ : ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 7) : 
  real.angle a b = real.pi / 3 := 
sorry

end angle_between_vectors_l43_43762


namespace probability_sum_seven_l43_43244

theorem probability_sum_seven :
  let diceA := {1, 2, 3, 4, 5, 6} in
  let diceB := {1, 2, 3, 4, 5, 6, 0} in -- Die B's 7 counts as 0
  let outcomes := (diceA × diceB).to_finset in
  let favorable_outcomes := ((1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)).to_finset in
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1/7 :=
by
  sorry

end probability_sum_seven_l43_43244


namespace isosceles_triangle_side_lengths_l43_43098

theorem isosceles_triangle_side_lengths (s t : ℝ) (P : EuclideanGeometry.Point 2) (A B C : EuclideanGeometry.Point 2)
  (h_isosceles : AB = AC) (h_BC : BC = t)
  (h_AP : EuclideanGeometry.Point.dist A P = 2)
  (h_BP : EuclideanGeometry.Point.dist B P = 2 * Real.sqrt 2)
  (h_CP : EuclideanGeometry.Point.dist C P = 3) :
  s = 2 * Real.sqrt 3 ∧ t = 6 :=
sorry

end isosceles_triangle_side_lengths_l43_43098


namespace percent_nurses_with_neither_l43_43644

-- Define the number of nurses in each category
def total_nurses : ℕ := 150
def nurses_with_hbp : ℕ := 90
def nurses_with_ht : ℕ := 50
def nurses_with_both : ℕ := 30

-- Define a predicate that checks the conditions of the problem
theorem percent_nurses_with_neither :
  ((total_nurses - (nurses_with_hbp + nurses_with_ht - nurses_with_both)) * 100 : ℚ) / total_nurses = 2667 / 100 :=
by sorry

end percent_nurses_with_neither_l43_43644


namespace find_C_D_l43_43343

theorem find_C_D (x C D : ℚ) 
  (h : 7 * x - 5 ≠ 0) -- Added condition to avoid zero denominator
  (hx : x^2 - 8 * x - 48 = (x - 12) * (x + 4))
  (h_eq : 7 * x - 5 = C * (x + 4) + D * (x - 12))
  (h_c : C = 79 / 16)
  (h_d : D = 33 / 16)
: 7 * x - 5 = 79 / 16 * (x + 4) + 33 / 16 * (x - 12) :=
by sorry

end find_C_D_l43_43343


namespace pyramid_surface_area_1056_l43_43053

noncomputable def total_surface_area_of_pyramid
  (A B C D : Point)
  (edges : List ℕ)
  (all_edges_spec : ∀ e ∈ edges, e = 16 ∨ e = 34)
  (no_equilateral_face : ∀ t : Triangle, isFace t DABC → ¬ (isEquilateral t)) :
  ℕ :=
  4 * (1 / 2 * 16 * 33)

theorem pyramid_surface_area_1056
  {A B C D: Point}
  (h1: ∀ e ∈ [distance A B, distance B C, distance C A, distance A D, distance B D, distance C D], e = 16 ∨ e = 34)
  (h2: ∀ t : Triangle, isFace t [A, B, C, D] → ¬ (isEquilateral t)) :
  total_surface_area_of_pyramid A B C D [distance A B, distance B C, distance C A, distance A D, distance B D, distance C D] h1 h2 = 1056 :=
by
  sorry

end pyramid_surface_area_1056_l43_43053


namespace triangle_problem_l43_43783

noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def B : ℝ := 45
noncomputable def S : ℝ := 3 + Real.sqrt 3

noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 6
noncomputable def C : ℝ := 75

theorem triangle_problem
  (a_val : a = 2 * Real.sqrt 3)
  (B_val : B = 45)
  (S_val : S = 3 + Real.sqrt 3) :
  c = Real.sqrt 2 + Real.sqrt 6 ∧ C = 75 :=
by
  sorry

end triangle_problem_l43_43783


namespace walmart_total_sales_l43_43156

-- Define the constants for the prices
def thermometer_price : ℕ := 2
def hot_water_bottle_price : ℕ := 6

-- Define the quantities and relationships
def hot_water_bottles_sold : ℕ := 60
def thermometer_ratio : ℕ := 7
def thermometers_sold : ℕ := thermometer_ratio * hot_water_bottles_sold

-- Define the total sales for thermometers and hot-water bottles
def thermometer_sales : ℕ := thermometers_sold * thermometer_price
def hot_water_bottle_sales : ℕ := hot_water_bottles_sold * hot_water_bottle_price

-- Define the total sales amount
def total_sales : ℕ := thermometer_sales + hot_water_bottle_sales

-- Theorem statement
theorem walmart_total_sales : total_sales = 1200 := by
  sorry

end walmart_total_sales_l43_43156


namespace calendar_cost_l43_43637

def total_items := 500
def calendars := 300
def date_books := 200
def cost_date_books := 0.50
def total_cost := 300

theorem calendar_cost :
  let cost_calendars := total_cost - (date_books * cost_date_books)
  \let cost_per_calendar := cost_calendars / calendars in
  cost_per_calendar = 2 / 3 :=
sorry

end calendar_cost_l43_43637


namespace cookies_left_l43_43032

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end cookies_left_l43_43032


namespace fraction_exponent_multiplication_l43_43160

theorem fraction_exponent_multiplication :
  ( (8/9 : ℚ)^2 * (1/3 : ℚ)^2 = (64/729 : ℚ) ) :=
by
  -- here we would write out the detailed proof
  sorry

end fraction_exponent_multiplication_l43_43160


namespace equilateral_triangle_in_ellipse_l43_43306

-- Given
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def altitude_on_y_axis (v : ℝ × ℝ := (0, 1)) : Prop := 
  v.1 = 0 ∧ v.2 = 1

-- The problem statement translated into a Lean proof goal
theorem equilateral_triangle_in_ellipse :
  ∃ (m n : ℕ), 
    (∀ (x y : ℝ), ellipse x y) →
    altitude_on_y_axis (0,1) →
    m.gcd n = 1 ∧ m + n = 937 :=
sorry

end equilateral_triangle_in_ellipse_l43_43306


namespace range_of_m_l43_43955

noncomputable def f (x : ℝ) : ℝ :=
if x' : 0 ≤ x ∧ x < 1 then (1 / 2) - 2 * x ^ 2
else if x' : 1 ≤ x ∧ x < 2 then -2 ^ (1 - abs ((3 / 2) - x))
else sorry -- To model the behavior of f outside [0, 2)

noncomputable def g (x m : ℝ) : ℝ := Real.log x - m

theorem range_of_m :
  (∀ x ∈ Ico (-4 : ℝ) (-2), ∃ x2 ∈ Icc (Real.exp (-1)) (Real.exp 2), f x - g x2 m ≥ 0)
  ↔ 7 ≤ m := 
sorry

end range_of_m_l43_43955


namespace minimum_value_f_l43_43751

noncomputable def f (x : ℕ) (hx : x > 0) : ℝ := (x^2 + 33 : ℝ) / x

theorem minimum_value_f : ∃ x ∈ {x : ℕ | x > 0}, f x (by exact x_pos_proof) = 23 / 2 := 
by
  use 6
  split
  -- 6 is indeed a positive natural number
  norm_num
  -- showing the function evaluated at 6 is 23/2
  unfold f
sorry

end minimum_value_f_l43_43751


namespace solution_l43_43313

noncomputable def problem : Prop :=
  let a := (Real.sqrt 5 + 2) ^ 2
  let b := (-1 / 2)⁻¹
  let c := Real.sqrt 49
  a + b - c = 4 * Real.sqrt 5

theorem solution : problem :=
  by
  apply problem
  sorry

end solution_l43_43313


namespace sum_of_areas_of_triangles_l43_43889

theorem sum_of_areas_of_triangles (m n p : ℤ) :
  let vertices := {v : ℝ × ℝ × ℝ | v.1 ∈ {0, 2} ∧ v.2 ∈ {0, 2} ∧ v.3 ∈ {0, 2}},
      triangles := {t : tuple ℝ 9 | 
                     ∃ v1 v2 v3 ∈ vertices, 
                     t = (v1.1, v1.2, v1.3, v2.1, v2.2, v2.3, v3.1, v3.2, v3.3)},
      triangle_area := λ t : tuple ℝ 9, 
        let (x1, y1, z1, x2, y2, z2, x3, y3, z3) := t in
        1 / 2 * real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2) *
                 real.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2 + (z3 - z1) ^ 2) *
                 real.sqrt ((x3 - x2) ^ 2 + (y3 - y2) ^ 2 + (z3 - z2) ^ 2)
  in (∑ t in triangles, triangle_area t) = 48 + real.sqrt 4608 + real.sqrt 3072 :=
sorry

end sum_of_areas_of_triangles_l43_43889


namespace cycle_selling_price_l43_43953

noncomputable def selling_price (cost_price : ℝ) (gain_percent : ℝ) : ℝ :=
  let gain_amount := (gain_percent / 100) * cost_price
  cost_price + gain_amount

theorem cycle_selling_price :
  selling_price 450 15.56 = 520.02 :=
by
  sorry

end cycle_selling_price_l43_43953


namespace sum_of_cubes_consecutive_integers_l43_43125

theorem sum_of_cubes_consecutive_integers (x : ℕ) (h1 : 0 < x) (h2 : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 :=
by
  -- proof will go here
  sorry

end sum_of_cubes_consecutive_integers_l43_43125


namespace no_real_roots_of_quadratic_l43_43931

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem no_real_roots_of_quadratic (h : quadratic_discriminant 1 (-1) 1 < 0) :
  ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l43_43931


namespace beprisque_count_l43_43960

def is_beprisque (n : ℕ) : Prop :=
  ∃ p k, nat.prime p ∧ k * k = n + 1 ∧ k > 1 ∧ nat.prime (p + 2)

theorem beprisque_count : ∀ n, 10 ≤ n ∧ n < 100 → is_beprisque n ↔ n = 10 :=
begin
  -- Proof skipped for conciseness
  sorry,
end

end beprisque_count_l43_43960


namespace midpoints_on_radical_axis_chords_on_line_equal_l43_43246

namespace CircleProofs

variables 
  {C1 C2 : Type} [metric_space C1] [metric_space C2]
  {O1 O2 : Type} [inhabited O1] [inhabited O2] -- Centers of circles
  (midpoints_ext_tangents : set (C1 × C2)) -- Midpoints of external tangents
  (radical_axis : set (C1 × C2))  -- Radical axis of two circles
  (tangent_points_ext : set (C1 × C2)) -- Points of tangency of external tangents
  (intersect_points : set (C1 × C2)) -- Intersected points where chords meet

open set

-- Part (a)
theorem midpoints_on_radical_axis (A : C1) (B : C2) :
  midpoints_ext_tangents ⊆ radical_axis := 
sorry

-- Part (b)
theorem chords_on_line_equal (A B : C1) (C D : C2) (A1 C1 : C1) (O : O1) (condition : O A1 * O A = O C1 * O C) :
  A A1 = C C1 := 
sorry

end CircleProofs

end midpoints_on_radical_axis_chords_on_line_equal_l43_43246


namespace range_of_k_l43_43566

noncomputable def f (k x : ℝ) : ℝ := (x-4) * exp(x) - (k / 20) * x^5 + (1 / 6) * k * x^4

-- Definition of the first derivative
noncomputable def f' (k x : ℝ) : ℝ := (x-3) * exp(x) - (k / 4) * x^4 + (2 / 3) * k * x^3

-- Definition of the second derivative
noncomputable def f'' (k x : ℝ) : ℝ := (x-2) * (exp(x) - k * x^2)

theorem range_of_k (k : ℝ) :
  (∀ x, f'' k x ≠ 0 ∧ (x ≠ 2 → f'' k x ≠ 0)) →
  (∃ x, f'' k 2 = 0 ∧ (∀ ε > 0, f'' k (2 - ε) * f'' k (2 + ε) < 0)) →
  k ∈ set.Iic (exp 2 / 4) :=
sorry

end range_of_k_l43_43566


namespace real_root_quadratic_complex_eq_l43_43422

open Complex

theorem real_root_quadratic_complex_eq (a : ℝ) :
  ∀ x : ℝ, a * (1 + I) * x^2 + (1 + a^2 * I) * x + (a^2 + I) = 0 →
  a = -1 :=
by
  intros x h
  -- We need to prove this, but we're skipping the proof for now.
  sorry

end real_root_quadratic_complex_eq_l43_43422


namespace guessing_game_prizes_l43_43281

theorem guessing_game_prizes : 
  let digits := [1, 2, 2, 3, 3, 3, 4]
  ∃ (D E F : ℕ), 
    D < 10000 ∧ E < 10000 ∧ F < 10000 ∧ 
    Multiset.card (Multiset.ofList digits) = 7 ∧ -- digits consist of exactly these elements
    Multiset.card (Multiset.ofList [D, E, F].bind (λ n, Multiset.ofList (Nat.digits 10 n))) = 7 →
  (∃! (guesses : List (ℕ × ℕ × ℕ)), guesses.length = 2520) :=
by
  sorry

end guessing_game_prizes_l43_43281


namespace exists_three_numbers_with_sum_l43_43564

theorem exists_three_numbers_with_sum (A B C : set ℕ) (hA : A.card = 672) (hB : B.card = 672) (hC : C.card = 672)
  (h_disjoint : disjoint A (B ∪ C)) (h_union : A ∪ B ∪ C = {n | 1 ≤ n ∧ n ≤ 2016}) :
  ∃ a ∈ A, ∃ b ∈ B, ∃ c ∈ C, a + b = c ∨ a + c = b ∨ b + c = a := 
sorry

end exists_three_numbers_with_sum_l43_43564


namespace constant_term_binomial_expansion_l43_43099

theorem constant_term_binomial_expansion (n : ℕ) (x : ℝ) 
  (h_coeff_sum : (sqrt x + 2 / x)^n = 729) : 
  constant_term_of_expansion(sqrt x + 2 / x)^n = 60 :=
sorry

end constant_term_binomial_expansion_l43_43099


namespace area_shaded_region_l43_43555

theorem area_shaded_region (r R : ℝ) (h1 : 0 < r) (h2 : r < R)
  (h3 : 60 = 2 * sqrt (R^2 - r^2)) :
  π * (R^2 - r^2) = 900 * π :=
by
  sorry

end area_shaded_region_l43_43555


namespace exists_m_leq_O_k_squared_l43_43045

/-- Conditions -/
def f (n k : ℕ) : ℕ :=
  Nat.find (λ m, ((kn)! / (n!)^m ∈ ℤ) ∧ ((kn)! / (n!)^(m + 1) ∉ ℤ))

def m (k : ℕ) : ℕ :=
  Nat.find (λ n, ∀ k, n ≥ m(k) → f(n, k) = k)

/-- Existence of m(k) such that m(k) ≤ O(k^2) -/
theorem exists_m_leq_O_k_squared (k : ℕ) : ∃ m : ℕ, (∀ n, n ≥ m → f(n, k) = k) ∧ m ≤ C * k^2 :=
sorry

end exists_m_leq_O_k_squared_l43_43045


namespace scientific_notation_of_203000_l43_43794

-- Define the number
def n : ℝ := 203000

-- Define the representation of the number in scientific notation
def scientific_notation (a b : ℝ) : Prop := n = a * 10^b ∧ 1 ≤ a ∧ a < 10

-- The theorem to state 
theorem scientific_notation_of_203000 : ∃ a b : ℝ, scientific_notation a b ∧ a = 2.03 ∧ b = 5 :=
by
  use 2.03
  use 5
  sorry

end scientific_notation_of_203000_l43_43794


namespace num_perfect_square_cube_factors_of_1800_l43_43764

theorem num_perfect_square_cube_factors_of_1800 :
  let n := 1800 in
  let factors := (2^3) * (3^2) * (5^2) in
  let is_perfect_square (x : ℕ) := ∀ p e, prime p → x = p^e → e % 2 = 0 in
  let is_perfect_cube (x : ℕ) := ∀ p e, prime p → x = p^e → e % 3 = 0 in
  let is_perfect_square_cube (x : ℕ) := is_perfect_square x ∧ is_perfect_cube x in
  ∃! x, x ∣ factors ∧ is_perfect_square_cube x
  :=
by {
  sorry
}

end num_perfect_square_cube_factors_of_1800_l43_43764


namespace direct_proportion_function_l43_43450

theorem direct_proportion_function (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = k * x) (h2 : f 3 = 6) : ∀ x, f x = 2 * x := by
  sorry

end direct_proportion_function_l43_43450


namespace general_term_formula_max_sum_first_six_terms_l43_43023

noncomputable def a (n : ℕ) : ℤ := 26 - 4 * n

def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * -4)) / 2

theorem general_term_formula :
  ∀ n, a n = 26 - 4 * n :=
by
  intros n
  -- proof steps are skipped
  sorry

theorem max_sum_first_six_terms :
  S 4 = S 8 → S 6 = 72 :=
by 
  intros h
  -- proof steps are skipped
  sorry

end general_term_formula_max_sum_first_six_terms_l43_43023


namespace bernardo_silvia_f2_l43_43056

noncomputable def a_n (n : ℕ) : ℕ := 1000 + n^3
noncomputable def b_n (n : ℕ) : ℕ := (1000 + n^3) / 100

theorem bernardo_silvia_f2 : f(2) = 13 :=
by
  -- Define the sequence a_n for k=2
  let a_n := λ (n : ℕ), 1000 + n^3

  -- Silvia's operation: keep the first 2 digits
  let b_n := λ (n : ℕ), (a_n n) / 100

  -- Prove that the smallest missing integer for k=2 is 13
  sorry

end bernardo_silvia_f2_l43_43056


namespace opposite_of_point_one_l43_43121

theorem opposite_of_point_one : ∃ x : ℝ, 0.1 + x = 0 ∧ x = -0.1 :=
by
  sorry

end opposite_of_point_one_l43_43121


namespace option_d_correct_l43_43928

theorem option_d_correct (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b :=
by
  sorry

end option_d_correct_l43_43928


namespace angle_between_vectors_l43_43763

noncomputable def vec (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def θ (a c : ℝ × ℝ) : ℝ :=
  let dot_prod := dot_product a c
  let mag_prod := magnitude a * magnitude c
  Real.arccos (dot_prod / mag_prod)

def problem : Prop :=
  let a := vec 1 2
  let b := vec (-2) (-4)
  ∀ (c : ℝ × ℝ), magnitude c = Real.sqrt 5
    → dot_product (vec ((1 - 2)) ((2 - 4))) c = 5 / 2
    → θ a c = (2 * Real.pi / 3)

theorem angle_between_vectors : problem :=
  sorry

end angle_between_vectors_l43_43763


namespace jeremy_age_l43_43977

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end jeremy_age_l43_43977


namespace perimeter_of_playground_l43_43290

theorem perimeter_of_playground 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 900) 
  (h2 : x * y = 216) : 
  2 * (x + y) = 72 := 
by 
  sorry

end perimeter_of_playground_l43_43290


namespace calculate_result_l43_43998

def binary_op (x y : ℝ) : ℝ := x^2 + y^2

theorem calculate_result (h : ℝ) : binary_op (binary_op h h) (binary_op h h) = 8 * h^4 :=
by
  sorry

end calculate_result_l43_43998


namespace OAC_angle_OAC_area_l43_43308

-- Define the conditions
variables (a b : ℝ)
noncomputable def y1 (x : ℝ) := -x^2 + a * x
noncomputable def y2 (x : ℝ) := x^2 + b * x

-- Define the intersection points and midpoints
def O := (0 : ℝ, 0 : ℝ)
def A := (a : ℝ, 0 : ℝ)
def B := (-b : ℝ, 0 : ℝ)

-- Condition that B is the midpoint of OA
axiom midpoint_condition : B = (a / 2, 0)

-- Intersection points of y1 and y2
noncomputable def C := (3 * a / 4 : ℝ, y1 (3 * a / 4))

-- Perpendicular condition
axiom perpendicular_condition : let AC := (A.1 - C.1, A.2 - C.2) in
                                let OC := (C.1 - O.1, C.2 - O.2) in
                                (AC.1 * OC.1 + AC.2 * OC.2) = 0

-- Problem 1: Proving angle OAC
theorem OAC_angle : angle O A C = 60 := sorry

-- Problem 2: Proving the area of triangle OAC
theorem OAC_area : abs ((a * (3 * a / 4) / 2) / 2) = (2 * sqrt 3) / 3 := sorry

end OAC_angle_OAC_area_l43_43308


namespace tangent_product_sqrt_seven_tangent_squares_sum_l43_43609

-- Part 1
theorem tangent_product_sqrt_seven 
  (θ1 θ2 θ3 : ℝ)
  (h1: θ1 = π / 7)
  (h2: θ2 = 2 * π / 7)
  (h3: θ3 = 3 * π / 7) : 
  (Real.tan θ1) * (Real.tan θ2) * (Real.tan θ3) = Real.sqrt 7 := 
sorry

-- Part 2
theorem tangent_squares_sum 
  (y1 y2 y3 : ℝ)
  (h : y1^3 - 21*y1^2 + 35*y1 - 7 = 0)
  (y1 = (Real.tan (π / 7))^2)
  (y2 = (Real.tan (2 * π / 7))^2)
  (y3 = (Real.tan (3 * π / 7))^2) : 
  y1 + y2 + y3 = 21 := 
sorry

end tangent_product_sqrt_seven_tangent_squares_sum_l43_43609


namespace leif_fruit_weight_difference_l43_43489

theorem leif_fruit_weight_difference :
  let apples_ounces := 27.5
  let grams_per_ounce := 28.35
  let apples_grams := apples_ounces * grams_per_ounce
  let dozens_oranges := 5.5
  let oranges_per_dozen := 12
  let total_oranges := dozens_oranges * oranges_per_dozen
  let weight_per_orange := 45
  let oranges_grams := total_oranges * weight_per_orange
  let weight_difference := oranges_grams - apples_grams
  weight_difference = 2190.375 := by
{
  sorry
}

end leif_fruit_weight_difference_l43_43489


namespace lcm_of_1_to_9_l43_43482

theorem lcm_of_1_to_9 :
  let n := 2520 in
  ∀ k, (k ∈ {i | i > 0 ∧ i < 10}) → n % k = 0 :=
by
  sorry

end lcm_of_1_to_9_l43_43482


namespace angle_A_value_l43_43480

theorem angle_A_value 
  (a b c A B C : ℝ)
  (h1 : (b - a) * (Real.sin B + Real.sin A) = c * (Real.sqrt 3 * Real.sin B - Real.sin C)) 
  (h2 : a = side_len_opposite_to_angle A)
  (h3 : b = side_len_opposite_to_angle B)
  (h4 : c = side_len_opposite_to_angle C) :
  A = Real.pi / 6 :=
sorry

end angle_A_value_l43_43480


namespace range_of_a_l43_43757

theorem range_of_a (a : ℝ) :
  let P := λ x : ℝ, x^2 + (2 * a^2 + 2) * x - a^2 + 4 * a - 7
  let Q := λ x : ℝ, x^2 + (a^2 + 4 * a - 5) * x - a^2 + 4 * a - 7
  ∀ x : ℝ, 
    (P x / Q x < 0) →
    let length_sum := (a^2 - 4 * a + 7)
    (length_sum ≥ 4) → (a ≤ 1 ∨ a ≥ 3) := sorry

end range_of_a_l43_43757


namespace tetrahedron_triangle_sides_l43_43063

-- Define the regular tetrahedron structure and its planes
structure Tetrahedron (V : Type) [MetricSpace V] :=
(A B C D : V)
(is_regular : Metric.btw V [A, B, C, D] = all sides equal) -- This is a placeholder for regularity condition.

-- Define points M and N in respective planes
def point_in_plane (p : V) (plane : set V) : Prop := p ∈ plane

-- Define the hypothesis and proof goal
theorem tetrahedron_triangle_sides (V : Type) [MetricSpace V] 
    (T : Tetrahedron V) (M N : V) 
    (hM : point_in_plane M {T.A, T.B, T.C}) 
    (hN : point_in_plane N {T.A, T.D, T.C}) :
    ∃ t : Triangle V, t.a = Metric.dist M N ∧ t.b = Metric.dist N T.B ∧ t.c = Metric.dist M T.D  :=
by
  sorry

end tetrahedron_triangle_sides_l43_43063


namespace scientific_notation_of_5_35_million_l43_43659

theorem scientific_notation_of_5_35_million : 
  (5.35 : ℝ) * 10^6 = 5.35 * 10^6 := 
by
  sorry

end scientific_notation_of_5_35_million_l43_43659


namespace find_a_for_three_roots_l43_43708

noncomputable def has_three_roots (a : ℝ) : Prop :=
  let f := λ x => x^2 - 2 * a * x + 1
  let discriminant := λ a c => a^2 - 4 * c
  (f = λ x => a ∧ (discriminant (2 * a) (a - 1) ≥ 0) ∨
   f = λ x => -a ∧ (discriminant (2 * a) (1 + a) ≥ 0)) ∧
  two_unique_real_roots (f = λ x => a) ∧
  one_unique_real_root (f = λ x => -a)

theorem find_a_for_three_roots (a : ℝ) :
  a = (1 + Real.sqrt 5) / 2 ↔ (a ≥ 0 ∧ has_three_roots a) :=
by sorry

end find_a_for_three_roots_l43_43708


namespace select_more_stable_athlete_l43_43311

-- Define the problem conditions
def athlete_average_score : ℝ := 9
def athlete_A_variance : ℝ := 1.2
def athlete_B_variance : ℝ := 2.4

-- Define what it means to have more stable performance
def more_stable (variance_A variance_B : ℝ) : Prop := variance_A < variance_B

-- The theorem to prove
theorem select_more_stable_athlete :
  more_stable athlete_A_variance athlete_B_variance →
  "A" = "A" :=
by
  sorry

end select_more_stable_athlete_l43_43311


namespace discount_percentage_inconsistent_l43_43287

theorem discount_percentage_inconsistent
  (purchase_price : ℝ) (marked_price_each : ℝ) (num_articles : ℝ)
  (purchase_price = 50) (marked_price_each = 22.5) (num_articles = 2) :
  ¬(∃ d : ℝ, d > 0 ∧ purchase_price = (num_articles * marked_price_each) * (1 - d / 100)) :=
by
  sorry

end discount_percentage_inconsistent_l43_43287


namespace johns_cookies_left_l43_43033

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end johns_cookies_left_l43_43033


namespace cevian_concurrency_l43_43760

theorem cevian_concurrency 
  (A B C P Q R X Y Z : Type)
  [triangle A B C]
  [trisect A P Q]
  [trisect B Q R]
  [trisect C R P]
  [bisector AX A B C]
  [bisector BY B A C]
  [bisector CZ C A B]
  [intersection X AX QR]
  [intersection Y BY RP]
  [intersection Z CZ PQ] :
  concurrent PX QY RZ :=
sorry

end cevian_concurrency_l43_43760


namespace cindy_correct_answer_l43_43314

theorem cindy_correct_answer (x : ℝ) (h : (x - 4) / 7 = 43) : (x - 7) / 4 = 74.5 :=
by
  have hx : x = 305 := by
    linarith
  rw hx
  linarith

end cindy_correct_answer_l43_43314


namespace range_of_a_l43_43453

-- Definition of the problem statement
def set_of_reals (a : ℝ) : Set ℝ := {2 * a, a^2 - a}

-- Condition: The set must have exactly 4 subsets, i.e., 2^2 = 4 implies two distinct elements
def has_four_subsets (a : ℝ) : Prop := (set_of_reals a).card = 2

-- The range of values for "a" such that the set_of_reals(a) has exactly 4 subsets
theorem range_of_a : {a : ℝ | has_four_subsets a} = {a : ℝ | a ≠ 0 ∧ a ≠ 3} :=
by
  sorry

end range_of_a_l43_43453


namespace most_frequent_data_is_mode_l43_43549

def most_frequent_data_name (dataset : Type) : String := "Mode"

theorem most_frequent_data_is_mode (dataset : Type) :
  most_frequent_data_name dataset = "Mode" :=
by
  sorry

end most_frequent_data_is_mode_l43_43549


namespace tetrahedron_plane_square_intersection_l43_43843

-- Define the regular tetrahedron and its properties
structure Tetrahedron :=
  (A B C D : Point)
  (edge_length : ℝ)
  (regular : (dist A B = edge_length) ∧ (dist A C = edge_length) ∧ (dist A D = edge_length) ∧
             (dist B C = edge_length) ∧ (dist B D = edge_length) ∧ (dist C D = edge_length))

-- Definition of midpoint of a segment
def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

-- Define the problem statement
theorem tetrahedron_plane_square_intersection (T : Tetrahedron) :
  ∃ plane : Plane, ∃ M N P Q : Point,
    (M = midpoint T.A T.B) ∧ (N = midpoint T.A T.D) ∧
    (P = midpoint T.B T.C) ∧ (Q = midpoint T.C T.D) ∧
    quadrilateral_is_square M N P Q :=
sorry

end tetrahedron_plane_square_intersection_l43_43843


namespace minimum_S_transform_780_dice_sum_l43_43925

theorem minimum_S_transform_780_dice_sum (n : ℕ) (h : 12 * n ≥ 780) : 
  ∃ S, (S = 13 * n - 780) ∧ S = 65 := 
by {
  use 65,
  split,
  { sorry },   -- Proof of S = 13 * n - 780
  { sorry }    -- Proof of S = 65 given n ≥ 65
}

end minimum_S_transform_780_dice_sum_l43_43925


namespace right_triangle_hypotenuse_log8_log2_l43_43111

theorem right_triangle_hypotenuse_log8_log2 :
  let h := Real.sqrt ((Real.log 125 / Real.log 8)^2 + (Real.log 49 / Real.log 2)^2)
  in 8^h = 35 :=
by
  sorry

end right_triangle_hypotenuse_log8_log2_l43_43111


namespace rth_term_of_arithmetic_progression_l43_43359

noncomputable def Sn (n : ℕ) : ℕ := 2 * n + 3 * n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) : 
  (Sn r - Sn (r - 1)) = 3 * r^2 + 5 * r - 2 :=
by sorry

end rth_term_of_arithmetic_progression_l43_43359


namespace std_dev_correct_l43_43382

axiom x1 : ℝ := 4
axiom x2 : ℝ := 5
axiom x3 : ℝ := 6

def mean (a b c : ℝ) : ℝ := (a + b + c) / 3

-- Sample variance for n=3
def variance (a b c : ℝ) : ℝ :=
  let μ := mean a b c
  in (1/3) * ((a - μ)^2 + (b - μ)^2 + (c - μ)^2)

def std_deviation (a b c : ℝ) : ℝ := Real.sqrt (variance a b c)

-- Main theorem to prove
theorem std_dev_correct :
  std_deviation x1 x2 x3 = Real.sqrt 6 / 3 :=
sorry

end std_dev_correct_l43_43382


namespace quotient_2032_div_12_in_base_4_l43_43342

def base_4_to_base_10 (n : ℕ) (digits : List ℕ) : ℕ :=
  digits.reverse.enum_from(0) |>.map (λ ⟨i, d⟩ => d * (4 ^ i)) |>.sum

def base_10_to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else List.unfold (λ n => if n = 0 then none else some (n % 4, n / 4)) n

theorem quotient_2032_div_12_in_base_4 :
  let n1 := base_4_to_base_10 4 [2, 3, 0, 2]
  let n2 := base_4_to_base_10 4 [2, 1]
  let quotient := n1 / n2
  let remainder := n1 % n2
  base_10_to_base_4 quotient = [3, 1, 1] ∧ base_10_to_base_4 remainder = [0, 1] :=
by
  let n1 := base_4_to_base_10 4 [2, 3, 0, 2]
  let n2 := base_4_to_base_10 4 [2, 1]
  let quotient := n1 / n2
  let remainder := n1 % n2
  have hq : base_10_to_base_4 quotient = [3, 1, 1] := by sorry
  have hr : base_10_to_base_4 remainder = [0, 1] := by sorry
  exact ⟨hq, hr⟩

end quotient_2032_div_12_in_base_4_l43_43342


namespace bicycle_has_four_wheels_l43_43408

-- Define the universe and properties of cars
axiom Car : Type
axiom Bicycle : Car
axiom has_four_wheels : Car → Prop
axiom all_cars_have_four_wheels : ∀ c : Car, has_four_wheels c

-- Define the theorem
theorem bicycle_has_four_wheels : has_four_wheels Bicycle :=
by
  sorry

end bicycle_has_four_wheels_l43_43408


namespace increase_semi_major_axis_l43_43565

noncomputable def delta_a (a1 a2 : ℝ) : ℝ :=
  a2 - a1

theorem increase_semi_major_axis
  (a1 a2 : ℝ)
  (initial_perm : π * (3 * (a1 + 5) - real.sqrt ((3 * a1 + 5) * (a1 + 15))) = 30)
  (new_perm : π * (3 * (a2 + 5) - real.sqrt ((3 * a2 + 5) * (a2 + 15))) = 40) :
  delta_a a1 a2 = a2 - a1 := by
  sorry

end increase_semi_major_axis_l43_43565


namespace mary_pays_51_dollars_l43_43511

def apple_cost := 1
def orange_cost := 2
def banana_cost := 3
def peach_cost := 4
def grape_cost := 5

def discount_per_5_fruits := 1
def discount_peaches_grapes := 3

def apples_bought := 5
def oranges_bought := 3
def bananas_bought := 2
def peaches_bought := 6
def grapes_bought := 4

theorem mary_pays_51_dollars :
  let total_fruit_cost := (apples_bought * apple_cost) + (oranges_bought * orange_cost) + 
                          (bananas_bought * banana_cost) + (peaches_bought * peach_cost) + 
                          (grapes_bought * grape_cost) in
  let total_fruits := apples_bought + oranges_bought + bananas_bought + peaches_bought + grapes_bought in
  let discount_5_fruits := (total_fruits / 5) * discount_per_5_fruits in
  let discount_peaches_grapes_sets := (peaches_bought / 3) * (grapes_bought / 2) * discount_peaches_grapes in
  let total_discount := discount_5_fruits + discount_peaches_grapes_sets in
  let final_cost := total_fruit_cost - total_discount in
  final_cost = 51 :=
by
  sorry

end mary_pays_51_dollars_l43_43511


namespace trains_crossing_time_l43_43807

-- Definitions for the conditions
def length_train (train_id : Nat) : ℝ := 100  -- meters
def speed_train (train_id : Nat) : ℝ := 80 * (1000 / 3600) -- 80 km/h converted to m/s

-- Combined length of both trains
def combined_length : ℝ := length_train 1 + length_train 2

-- Relative speed when two trains are moving towards each other
def relative_speed : ℝ := speed_train 1 + speed_train 2

-- Time taken for the trains to cross each other completely
def crossing_time : ℝ := combined_length / relative_speed

-- The statement to prove
theorem trains_crossing_time : crossing_time = 2.25 := 
by 
  -- The proof is deliberately omitted
  sorry

end trains_crossing_time_l43_43807


namespace flag_movement_distance_l43_43988

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l43_43988


namespace max_sum_partition_l43_43886

theorem max_sum_partition (S : ℝ) (hS : S = 171 / 10) 
  (n : ℕ) (a : ℕ → ℝ) (ha_bounds : ∀ i, 0 ≤ a i ∧ a i ≤ 1) 
  (ha_sum : (finset.range n).sum a ≤ S): 
  ∃ (A B : finset ℕ), A ∪ B = finset.range n ∧
  A ∩ B = ∅ ∧
  (A.sum a ≤ 9 ∧ B.sum a ≤ 9) :=
sorry

end max_sum_partition_l43_43886


namespace star_2_3_l43_43999

-- Definitions and conditions
def star (x y k : ℝ) : ℝ := x^y * k

variable (k : ℝ) (h_k : 0 < k)
variable (x y : ℝ) (h_xy : 0 < x) (h_y : 0 < y)

axiom cond1 : (star x y k)^y = x * (star y y k)
axiom cond2 : (star x 1 k) * k = star x 1 k
axiom cond3 : star 1 1 k = k

-- The target statement to prove
theorem star_2_3 : star 2 3 k = 8 * k :=
sorry

end star_2_3_l43_43999


namespace max_q_le_max_p_l43_43047

theorem max_q_le_max_p (a : ℕ → ℂ) (c : ℕ → ℝ) (h_convex : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → 2 * c k ≤ c (k - 1) + c (k + 1)) 
  (h_c_seq : 1 = c 0 ∧ ∀ k, 0 ≤ k ∧ k ≤ n → c k ≤ c (k - 1)) :
  ∀ z, |z| ≤ 1 → (c 0 * a 0 + c 1 * a 1 * z + c 2 * a 2 * z^2 + ⋯ + c n * a n * z^n) ≤ max {z : z ∈ ℂ ∣ |z| ≤ 1} (a 0 + a 1 * z + a 2 * z^2 + ⋯ + a n * z^n) := 
sorry

end max_q_le_max_p_l43_43047


namespace max_distance_OA_OB_l43_43476

def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, Real.sqrt 3 + 2 * Real.sin θ)

def polar_line_l1 (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

def polar_line_l2 (α : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ ∈ ℝ ∧ θ = α + Real.pi / 3

theorem max_distance_OA_OB {α : ℝ} (hα : 0 < α ∧ α < Real.pi / 2) :
  ∃ A B : ℝ × ℝ, |OA| + |OB| = 4 * Real.sqrt 3 :=
sorry

end max_distance_OA_OB_l43_43476


namespace quadratic_roots_l43_43550

theorem quadratic_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + k*x1 + (k - 1) = 0) ∧ (x2^2 + k*x2 + (k - 1) = 0) :=
by
  sorry

end quadratic_roots_l43_43550


namespace relationship_and_range_max_profit_find_a_l43_43848

noncomputable def functional_relationship (x : ℝ) : ℝ :=
if 40 ≤ x ∧ x ≤ 50 then 5
else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x
else 0  -- default case to handle x out of range, though ideally this should not occur in the context.

theorem relationship_and_range : 
  ∀ (x : ℝ), (40 ≤ x ∧ x ≤ 100) →
    (functional_relationship x = 
    (if 40 ≤ x ∧ x ≤ 50 then 5 else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x else 0)) :=
sorry

noncomputable def monthly_profit (x : ℝ) : ℝ :=
(x - 40) * functional_relationship x

theorem max_profit : 
  (∀ x, 40 ≤ x ∧ x ≤ 100 → monthly_profit x ≤ 90) ∧
  (monthly_profit 70 = 90) :=
sorry

noncomputable def donation_profit (x a : ℝ) : ℝ :=
(x - 40 - a) * (10 - 0.1 * x)

theorem find_a (a : ℝ) : 
  (∀ x, x ≤ 70 → donation_profit x a ≤ 78) ∧
  (donation_profit 70 a = 78) → 
  a = 4 :=
sorry

end relationship_and_range_max_profit_find_a_l43_43848


namespace gcd_765432_654321_l43_43182

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43182


namespace chord_length_of_circle_l43_43103

theorem chord_length_of_circle (x y : ℝ) :
  (x^2 + y^2 - 4 * x - 4 * y - 1 = 0) ∧ (y = x + 2) → 
  2 * Real.sqrt 7 = 2 * Real.sqrt 7 :=
by sorry

end chord_length_of_circle_l43_43103


namespace closest_ratio_l43_43004

theorem closest_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : (x + y) / 2 = 3 * Real.sqrt (x * y)) :
  abs (x / y - 34) < abs (x / y - n) :=
by sorry

end closest_ratio_l43_43004


namespace remaining_speed_l43_43604
open Real

theorem remaining_speed
  (D T : ℝ) (h1 : 40 * (T / 3) = (2 / 3) * D)
  (h2 : (T / 3) * 3 = T) :
  (D / 3) / ((2 * ((2 / 3) * D) / (40) / (3)) * 2 / 3) = 10 :=
by
  sorry

end remaining_speed_l43_43604


namespace first_tray_holds_260_cups_l43_43079

variable (x : ℕ)

def first_tray_holds_x_cups (tray1 : ℕ) := tray1 = x
def second_tray_holds_x_minus_20_cups (tray2 : ℕ) := tray2 = x - 20
def total_cups_in_both_trays (tray1 tray2: ℕ) := tray1 + tray2 = 500

theorem first_tray_holds_260_cups (tray1 tray2 : ℕ) :
  first_tray_holds_x_cups x tray1 →
  second_tray_holds_x_minus_20_cups x tray2 →
  total_cups_in_both_trays tray1 tray2 →
  x = 260 := by
  sorry

end first_tray_holds_260_cups_l43_43079


namespace gcd_765432_654321_l43_43213

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43213


namespace find_vector_a_l43_43759

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def vec_a_solutions : Point3D × Point3D :=
  let a1 := ⟨1 / real.sqrt 22, 7 / real.sqrt 22, 4 / real.sqrt 22⟩
  let a2 := ⟨-1 / real.sqrt 22, -7 / real.sqrt 22, -4 / real.sqrt 22⟩
  (a1, a2)

-- Lean statement for the proof problem
theorem find_vector_a : 
  ∃ (a : Point3D),
    magnitude a = real.sqrt 3 ∧
    dot_product a (vector_sub ⟨-2, 1, 6⟩ ⟨0, 2, 3⟩) = 0 ∧
    dot_product a (vector_sub ⟨1, -1, 5⟩ ⟨0, 2, 3⟩) = 0 ∧
    (a = (vec_a_solutions).1 ∨ a = (vec_a_solutions).2) :=
by
  sorry

end find_vector_a_l43_43759


namespace find_x1_value_l43_43741

theorem find_x1_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
  (h_eq : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
  x1 = 2 / 3 := 
sorry

end find_x1_value_l43_43741


namespace right_triangle_hypotenuse_and_perimeter_l43_43594

theorem right_triangle_hypotenuse_and_perimeter (a b : ℕ) (ha : a = 60) (hb : b = 80) : 
  let c := Math.sqrt (a^2 + b^2) in
  let P := a + b + c in
  c = 100 ∧ P = 240 :=
by
  have h_c : c = Math.sqrt (60^2 + 80^2) := by rw [ha, hb]
  have h_c_val : c = 100 := by rw [h_c]; norm_num
  have h_p_val : P = 60 + 80 + 100 := by rw [ha, hb, h_c_val]
  have h_p : P = 240 := by rw [h_p_val]; norm_num
  exact ⟨h_c_val, h_p⟩

end right_triangle_hypotenuse_and_perimeter_l43_43594


namespace conditional_probability_B_given_A_l43_43077

variables (Ω : Type) [Fintype Ω] [UniformProbability Ω]

noncomputable def zongzi : Multiset ℕ := {1, 1, 2, 2, 2}.to_multiset

noncomputable def event_A : Event Ω := { z | (z ∈ {1, 1, 2, 2, 2}.to_multiset).card = 2 ∧ ∃ (a : ℕ), (z.filter (λ x, x = a)).card = 2 }

noncomputable def event_B : Event Ω := { z | (z ∈ {2, 2, 2}.to_multiset).card = 2 }

noncomputable def conditional_probability := (event_B ∩ event_A).card.toRat / event_A.card.toRat

theorem conditional_probability_B_given_A :
  conditional_probability = 3 / 4 := sorry

end conditional_probability_B_given_A_l43_43077


namespace heap_holds_20_sheets_l43_43680

theorem heap_holds_20_sheets :
  ∀ (num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets : ℕ),
    num_bundles = 3 →
    num_bunches = 2 →
    num_heaps = 5 →
    sheets_per_bundle = 2 →
    sheets_per_bunch = 4 →
    total_sheets = 114 →
    (total_sheets - (num_bundles * sheets_per_bundle + num_bunches * sheets_per_bunch)) / num_heaps = 20 := 
by
  intros num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end heap_holds_20_sheets_l43_43680


namespace one_percent_as_decimal_l43_43248

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := 
by 
  sorry

end one_percent_as_decimal_l43_43248


namespace isosceles_triangle_sides_correct_l43_43684

noncomputable def isosceles_triangle_sides (r ρ : ℝ) (h : r ≥ 2 * ρ) : ℝ × ℝ :=
  let m₁ := r + ρ + real.sqrt (r * (r - 2 * ρ))
  let m₂ := r + ρ - real.sqrt (r * (r - 2 * ρ))
  ((2 * real.sqrt (ρ * (2 * r - ρ - 2 * real.sqrt (r * (r - 2 * ρ))))),
   (2 * real.sqrt (ρ * (2 * r - ρ + 2 * real.sqrt (r * (r - 2 * ρ))))))

theorem isosceles_triangle_sides_correct (r ρ : ℝ) (h : r ≥ 2 * ρ) :
  isosceles_triangle_sides r ρ h = 
  (2 * real.sqrt (ρ * (2 * r - ρ - 2 * real.sqrt (r * (r - 2 * ρ)))),
   2 * real.sqrt (ρ * (2 * r - ρ + 2 * real.sqrt (r * (r - 2 * ρ))))) :=
sorry

end isosceles_triangle_sides_correct_l43_43684


namespace number_of_parrots_in_each_cage_l43_43631

theorem number_of_parrots_in_each_cage (num_cages : ℕ) (total_birds : ℕ) (parrots_per_cage parakeets_per_cage : ℕ)
    (h1 : num_cages = 9)
    (h2 : parrots_per_cage = parakeets_per_cage)
    (h3 : total_birds = 36)
    (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) :
  parrots_per_cage = 2 :=
by
  sorry

end number_of_parrots_in_each_cage_l43_43631


namespace base_of_square_eq_l43_43261

theorem base_of_square_eq (b : ℕ) (h : b > 6) : 
  (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 → b = 7 :=
by
  sorry

end base_of_square_eq_l43_43261


namespace average_of_integers_l43_43576

theorem average_of_integers (A B C D : ℤ) (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D = 90) (h5 : 5 ≤ A) (h6 : A ≠ B ∧ B ≠ C ∧ C ≠ D) :
  (A + B + C + D) / 4 = 27 :=
by
  sorry

end average_of_integers_l43_43576


namespace probability_at_least_one_girls_pair_l43_43282

-- Define necessary conditions and parameters
def total_people : ℕ := 18
def boys : ℕ := 9
def girls : ℕ := 9
def pairs : ℕ := 9

-- Calculate factorial of a number
noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Total number of ways to pair 18 people in 9 pairs
noncomputable def N_total : ℝ :=
  (factorial total_people : ℝ) / ((factorial pairs : ℝ) * (2^(pairs : ℝ)) * (factorial girls : ℝ))

-- Number of ways to pair 9 boys and 9 girls with no girls-only pair
noncomputable def N_no_girls_pair : ℝ :=
  (factorial boys : ℝ) * (factorial girls : ℝ)

-- Probability of no girls-only pair
noncomputable def P_no_girls_pair : ℝ :=
  N_no_girls_pair / N_total

-- Probability of at least one girls-only pair
noncomputable def P_girls_pair : ℝ :=
  1 - P_no_girls_pair

-- Lean statement to prove
theorem probability_at_least_one_girls_pair : P_girls_pair ≈ 0.99 := sorry

end probability_at_least_one_girls_pair_l43_43282


namespace collinear_midpoints_and_center_l43_43088

-- Define basic structures and points
variable {Point : Type} [preorder Point]
variable {Line : Type} [preorder Line]
variable {Circle : Type} [preorder Circle]

-- Definitions for geometric configurations
def Quadrilateral := (a b c d : Point)
def CircumscribedCircle (quad : Quadrilateral) (circle : Circle) := 
    -- Placeholder definitions for tangency conditions
    sorry

def ExtensionIntersect (a b c d : Point) (o : Point) := 
    -- Placeholder for checking intersection of extensions
    sorry

def TangentCircle (side1 side2 : Line) (circle : Circle) (tangentPoint : Point) (extension1 extension2 : Line) := 
    -- Placeholder definitions for tangency circle with tangents
    sorry

def Collinear (p1 p2 p3 : Point) := 
    -- Placeholder for checking collinearity
    sorry

def Midpoint (p1 p2 : Point) : Point :=
    -- Placeholder for computing midpoint
    sorry

-- Main theorem statement
theorem collinear_midpoints_and_center
    (a b c d o k l : Point)
    (omega omega1 omega2 : Circle)
    (quad : Quadrilateral a b c d)
    (circ_omega : CircumscribedCircle quad omega)
    (intersect: ExtensionIntersect a b c d o)
    (tan_omega1 : TangentCircle b c omega1 k (.extension a b) (extension c d))
    (tan_omega2 : TangentCircle a d omega2 l (extension a b) (extension c d))
    (collinear_okl : Collinear o k l) : 
    Collinear (Midpoint b c) (Midpoint a d) (Center omega) :=
sorry

end collinear_midpoints_and_center_l43_43088


namespace monthly_fixed_cost_is_correct_l43_43622

-- Definitions based on the conditions in the problem
def production_cost_per_component : ℕ := 80
def shipping_cost_per_component : ℕ := 5
def components_per_month : ℕ := 150
def minimum_price_per_component : ℕ := 195

-- Monthly fixed cost definition based on the provided solution
def monthly_fixed_cost := components_per_month * (minimum_price_per_component - (production_cost_per_component + shipping_cost_per_component))

-- Theorem stating that the calculated fixed cost is correct.
theorem monthly_fixed_cost_is_correct : monthly_fixed_cost = 16500 :=
by
  unfold monthly_fixed_cost
  norm_num
  sorry

end monthly_fixed_cost_is_correct_l43_43622


namespace irrational_count_is_three_l43_43930

-- Define the given numbers
def number_list : List ℝ := [
  rat_of_pnat (pnat.mk 1000000001 1000000),  -- as a proxy for 0.010010001...
  0,
  Real.sqrt 5,
  3.14,
  -5 / 11,
  Real.pi,
  Real.cbrt 8  -- as a proxy for 3 √ 8
]

-- Define a predicate for irrational numbers
def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Count the number of irrational numbers in the list
def count_irrationals (l : List ℝ) : ℕ := l.countp is_irrational

-- Theorem stating the number of irrational numbers is 3
theorem irrational_count_is_three :
  count_irrationals number_list = 3 :=
sorry

end irrational_count_is_three_l43_43930


namespace parabola_hyperbola_distance_l43_43551

def parabola_focus : ℝ × ℝ := (0, 1)

def hyperbola_asymptotes (x y : ℝ) : Prop :=
  (y = (sqrt 3 / 3) * x) ∨ (y = -(sqrt 3 / 3) * x)

theorem parabola_hyperbola_distance :
  let d := abs (1) / sqrt (1 + (sqrt 3 / 3)^2)
  d = sqrt 3 / 2 :=
by
  sorry

end parabola_hyperbola_distance_l43_43551


namespace parallel_case_perpendicular_case_l43_43433

variables (m : ℝ)
def a := (2, -1)
def b := (-1, m)
def c := (-1, 2)
def sum_ab := (1, m - 1)

-- Parallel case (dot product is zero)
theorem parallel_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = -1 :=
by
  sorry

-- Perpendicular case (dot product is zero)
theorem perpendicular_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = 3 / 2 :=
by
  sorry

end parallel_case_perpendicular_case_l43_43433


namespace gcd_proof_l43_43193

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43193


namespace maximum_value_of_expression_is_4_l43_43772

noncomputable def maximimum_integer_value (x : ℝ) : ℝ :=
    (5 * x^2 + 10 * x + 12) / (5 * x^2 + 10 * x + 2)

theorem maximum_value_of_expression_is_4 :
    ∃ x : ℝ, ∀ y : ℝ, maximimum_integer_value y ≤ 4 ∧ maximimum_integer_value x = 4 := 
by 
  -- Proof omitted for now
  sorry

end maximum_value_of_expression_is_4_l43_43772


namespace stadium_surface_area_correct_l43_43563

noncomputable def stadium_length_yards : ℝ := 62
noncomputable def stadium_width_yards : ℝ := 48
noncomputable def stadium_height_yards : ℝ := 30

noncomputable def stadium_length_feet : ℝ := stadium_length_yards * 3
noncomputable def stadium_width_feet : ℝ := stadium_width_yards * 3
noncomputable def stadium_height_feet : ℝ := stadium_height_yards * 3

def total_surface_area_stadium (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

theorem stadium_surface_area_correct :
  total_surface_area_stadium stadium_length_feet stadium_width_feet stadium_height_feet = 110968 := by
  sorry

end stadium_surface_area_correct_l43_43563


namespace arithmetic_sequence_sum_l43_43461

variable (a_n : ℕ → ℕ)

theorem arithmetic_sequence_sum (h1: a_n 1 + a_n 2 = 5) (h2 : a_n 3 + a_n 4 = 7) (arith : ∀ n, a_n (n + 1) - a_n n = a_n 2 - a_n 1) :
  a_n 5 + a_n 6 = 9 := 
sorry

end arithmetic_sequence_sum_l43_43461


namespace correct_survey_sequence_l43_43626

-- Definition of conditions as predicates
def collect_data (steps : List ℕ) : Prop := steps.head? = some 2
def organize_data (steps : List ℕ) : Prop := steps.nth? 1 = some 4
def draw_pie_chart (steps : List ℕ) : Prop := steps.nth? 2 = some 1
def analyze_statistics (steps : List ℕ) : Prop := steps.nth? 3 = some 3

-- The main theorem to be proved
theorem correct_survey_sequence (steps : List ℕ) :
  collect_data steps ∧ organize_data steps ∧ draw_pie_chart steps ∧ analyze_statistics steps ↔ 
  steps = [2, 4, 1, 3] :=
sorry

end correct_survey_sequence_l43_43626


namespace cube_triangle_area_sum_solution_l43_43903

def cube_vertex_triangle_area_sum (m n p : ℤ) : Prop :=
  m + n + p = 121 ∧
  (∀ (a : ℕ) (b : ℕ) (c : ℕ), a * b * c = 8) -- Ensures the vertices are for a 2*2*2 cube

theorem cube_triangle_area_sum_solution :
  cube_vertex_triangle_area_sum 48 64 9 :=
by
  unfold cube_vertex_triangle_area_sum
  split
  · exact rfl -- m + n + p = 121
  · intros a b c h
    sorry -- Conditions ensuring these m, n, p were calculated from a 2x2x2 cube

end cube_triangle_area_sum_solution_l43_43903


namespace two_AP_eq_BP_l43_43463

variables {A B C D P : Type} 

-- Condition 1: ABCD is a convex quadrilateral with diagonals AC and BD intersecting at P
def is_convex_quadrilateral (A B C D P : Type) : Prop := 
  -- We would need a specific definition or axioms to establish convexity, intersections, etc. 
  sorry

-- Condition 2: angle DAC = 90 degrees
def angle_DAC_90 (A B C D P : Type) [has_angle A D C (90 : ℝ) (100 : ℝ)] : Prop := -- Assuming some angle structure exists
  sorry

-- Condition 3: 2 * angle ADB = angle ACB
def angle_relation_ADB_ACB (A B C D P : Type) [has_angle A D B (x : ℝ)] [has_angle A C B (2 * x : ℝ)] : Prop :=
  sorry

-- Condition 4: angle DBC + 2 * angle ADC = 180 degrees
def angle_relation_DBC_ADC (A B C D P : Type) [has_angle D B C (y : ℝ)] [has_angle A D C (z : ℝ)] : Prop :=
  y + 2 * z = 180

-- Prove that 2 * AP = BP given the conditions
theorem two_AP_eq_BP (A B C D P : Type) 
  [is_convex_quadrilateral A B C D P] 
  [angle_DAC_90 A B C D P] 
  [angle_relation_ADB_ACB A B C D P] 
  [angle_relation_DBC_ADC A B C D P] : 
  ∃ (a p: ℝ), 2 * a * p = a * p :=
  
sorry

end two_AP_eq_BP_l43_43463


namespace gcd_765432_654321_l43_43188

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43188


namespace entry_15_is_22_l43_43716

def r_7 (n : ℕ) : ℕ := n % 7

def satisfies_condition (m : ℕ) : Prop := 
  r_7 (3 * m) ≤ 3

noncomputable def nth_entry (n : ℕ) : ℕ :=
  (List.filter satisfies_condition (List.range 1000000)).nth n |>.getD 0

theorem entry_15_is_22 :
  nth_entry 14 = 22 :=
by
  sorry

end entry_15_is_22_l43_43716


namespace exists_line_segment_l43_43149

-- Definitions from the conditions
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

def intersects (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, onCircle c1 p ∧ onCircle c2 p

-- Setting up main problem statement
theorem exists_line_segment 
  (c1 c2 : Circle) 
  (h_intersection : intersects c1 c2) 
  (L : ℝ) :
  ∃ NP : (ℝ × ℝ) × (ℝ × ℝ), 
    (NP.1 ≠ NP.2) ∧
    ∃ M : ℝ × ℝ,
      onCircle c1 M ∧ onCircle c2 M ∧
      (NP.1 = M ∨ NP.2 = M) ∧
      ∃ AO : ℝ × ℝ,
        (∃ k : ℝ, k * (AO.1 - M.1) = NP.1 - NP.2) ∧
        (∃ seg_points : ℝ × ℝ × ℝ × ℝ, 
          onCircle c1 seg_points.1 ∧ 
          onCircle c1 seg_points.2 ∧
          onCircle c2 seg_points.1 ∧
          onCircle c2 seg_points.2 ∧
          ((seg_points.1 - seg_points.2) = L ∨ (seg_points.2 - seg_points.1) = L)) :=
sorry

end exists_line_segment_l43_43149


namespace frog_climbs_out_l43_43280

theorem frog_climbs_out (d climb slip : ℕ) (h : d = 20) (h_climb : climb = 3) (h_slip : slip = 2) :
  ∃ n : ℕ, n = 20 ∧ d ≤ n * (climb - slip) + climb :=
sorry

end frog_climbs_out_l43_43280


namespace avg_next_six_consecutive_l43_43887

theorem avg_next_six_consecutive (x y : ℤ) (h : y = 6 * x + 15) :
    (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5)) / 6 = 6 * x + 17.5 :=
by
  sorry

end avg_next_six_consecutive_l43_43887


namespace coeff_m5n4_in_m_plus_n_power_9_l43_43596

theorem coeff_m5n4_in_m_plus_n_power_9 :
  (∃ c : ℕ, (m + n)^9 = c * m^5 * n^4 + ...) ∧ c = 126 :=
by
  sorry

end coeff_m5n4_in_m_plus_n_power_9_l43_43596


namespace proof_f_x_plus_2_minus_f_x_l43_43442

def f (x : ℝ) : ℝ := 9 ^ x

theorem proof_f_x_plus_2_minus_f_x (x : ℝ) : f(x + 2) - f(x) = 80 * f(x) :=
by
  sorry

end proof_f_x_plus_2_minus_f_x_l43_43442


namespace diagonal_le_two_l43_43384

-- Define a structure for a point in 2D space.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the distance function between two points.
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a convex hexagon and the condition that its sides are at most 1.
structure ConvexHexagon :=
  (A B C D E F : Point)
  (convex : ∀ (P1 P2 P3 : Point), P1 ≠ P2 → P2 ≠ P3 → P3 ≠ P1 → 
    distance P1 P2 + distance P2 P3 + distance P3 P1 ≤ 
    distance P1 P3 + distance P3 P2 + distance P2 P1)
  (sides_le_one : ∀ {P : Point}, 
    (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F) → 
    (distance A B ≤ 1 ∧ distance B C ≤ 1 ∧ distance C D ≤ 1 ∧ 
     distance D E ≤ 1 ∧ distance E F ≤ 1 ∧ distance F A ≤ 1))

-- Prove that at least one of the diagonals is ≤ 2.
theorem diagonal_le_two (hex : ConvexHexagon) :
    distance hex.A hex.D ≤ 2 ∨ distance hex.B hex.E ≤ 2 ∨ distance hex.C hex.F ≤ 2 :=
  sorry

end diagonal_le_two_l43_43384


namespace nate_age_is_14_l43_43333

def nate_current_age (N : ℕ) : Prop :=
  ∃ E : ℕ, E = N / 2 ∧ N - E = 7

theorem nate_age_is_14 : nate_current_age 14 :=
by {
  sorry
}

end nate_age_is_14_l43_43333


namespace sum_S2014_l43_43885

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → 1 / (a n * a (n + 1)) + 1 / (a n * a (n + 2)) + 1 / (a (n + 1) * a (n + 2)) = 1)
  ∧ (a 1 + a 3 = 6)
  ∧ (∀ n : ℕ, n ≥ 1 → a n < a (n + 1))
  ∧ (∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n)

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1)

theorem sum_S2014 (a : ℕ → ℝ) (ha : sequence a) :
  ⌊Sn a 2014⌋ = 5368 := sorry

end sum_S2014_l43_43885


namespace determine_r_l43_43699

theorem determine_r (r : ℝ) (h : 16 = 2^(3*r + 4)) : r = 0 :=
sorry

end determine_r_l43_43699


namespace alice_bob_game_l43_43662

def game_outcome (n : ℕ) : string :=
if ∃ k : ℕ, n = 3*k + 2 then "Bob wins"
else if n >= 3 then "Alice wins"
else "Invalid n"

theorem alice_bob_game (n : ℕ) :
  n >= 3 →
  (∃ k : ℕ, n = 3*k + 2 → game_outcome n = "Bob wins") ∧
  (¬ (∃ k : ℕ, n = 3*k + 2) → game_outcome n = "Alice wins") := by
  sorry

end alice_bob_game_l43_43662


namespace spa_polish_total_digits_l43_43956

theorem spa_polish_total_digits (girls : ℕ) (digits_per_girl : ℕ) (total_digits : ℕ)
  (h1 : girls = 5) (h2 : digits_per_girl = 20) : total_digits = 100 :=
by
  sorry

end spa_polish_total_digits_l43_43956


namespace gcd_lcm_sum_l43_43233

theorem gcd_lcm_sum (a b : ℕ) (h₁ : a = 120) (h₂ : b = 3507) :
  Nat.gcd a b + Nat.lcm a b = 140283 := by 
  sorry

end gcd_lcm_sum_l43_43233


namespace sum_of_areas_of_triangles_in_cube_l43_43893

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l43_43893


namespace t_shirts_in_two_hours_l43_43676

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end t_shirts_in_two_hours_l43_43676


namespace cone_section_height_ratio_l43_43102

def cone_height_ratio (h : ℝ): Prop :=
∃ (s_h_upper s_h_lower : ℝ), 
  s_h_upper / s_h_lower = 1 / (Real.sqrt 2 - 1) ∧ 
  s_h_upper + s_h_lower = h ∧ 
  (s_h_upper / h) ^ 2 = 1 / 2

theorem cone_section_height_ratio
  (h : ℝ) 
  (h_positive : h > 0)
  (A_base : ℝ)
  (A_half_section : ℝ) :
  A_half_section = 1 / 2 * A_base → cone_height_ratio h :=
by
  intros h_positive A_base A_half_section h_pos half_A_base_eq
  sorry

end cone_section_height_ratio_l43_43102


namespace slope_angle_60_l43_43881

noncomputable def slope_of_line (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem slope_angle_60 (P Q : ℝ × ℝ) (hP : P = (0, 0)) (hQ : Q = (1, Real.sqrt 3)) :
  let k := slope_of_line P Q in
  k = Real.sqrt 3 → ∃ θ : ℝ, θ = 60 ∧ Real.tan θ = k :=
by
  intro k hk
  use 60
  constructor
  · rfl
  · sorry

end slope_angle_60_l43_43881


namespace maximize_fraction_l43_43798

theorem maximize_fraction (a b c : ℕ) (h_diff1 : a ≠ b) (h_diff2 : a ≠ c) (h_diff3 : b ≠ c) 
  (h_a : a = 2 ∨ a = 3 ∨ a = 6) 
  (h_b : b = 2 ∨ b = 3 ∨ b = 6) 
  (h_c : c = 2 ∨ c = 3 ∨ c = 6) 
  (h_max : (a : ℚ) / (b : ℚ / c) = 9) : 
  b = 2 :=
sorry

end maximize_fraction_l43_43798


namespace geometric_k_value_sum_reciprocal_less_two_sum_of_elements_in_A_l43_43725

-- Question 1
theorem geometric_k_value (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (k : ℤ) (h1 : ∀ n, a_n n = 2^n + 3^n) :
  (b_n = λ n, a_n (n + 1) + k * a_n n) →
  (∀ n, (b_n (n + 1))^2 = b_n n * b_n (n + 2)) →
  (k = -2 ∨ k = -3) :=
sorry

-- Question 2
theorem sum_reciprocal_less_two (a_n : ℕ → ℕ) (C_n : ℕ → ℤ) (S_n : ℕ → ℤ) 
  (h1 : ∀ n, a_n n = 2^n + 3^n) 
  (h2 : ∀ n, C_n n = log 3 (a_n n - 2^n))
  (h3 : ∀ n, S_n n = ∑ i in range n, C_n i) :
  (∑ i in range n, 1 / S_n i) < 2 :=
sorry

-- Question 3
theorem sum_of_elements_in_A (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (A : set ℕ) (k : ℤ) 
  (h1 : ∀ n, a_n n = 2^n + 3^n) 
  (h2 : ∀ n, b_n n = a_n (n + 1) + k * a_n n) 
  (hk : k = -2)
  (hA : A = {n ∈ ℕ | (2 * n - 1) / b_n n > 1 / 9}) :
  (H : ∑ n in A, n = 6) :=
sorry

end geometric_k_value_sum_reciprocal_less_two_sum_of_elements_in_A_l43_43725


namespace fred_onions_l43_43089

theorem fred_onions (sara_onions sally_onions total_onions : ℕ)
    (h_sara : sara_onions = 4)
    (h_sally : sally_onions = 5)
    (h_total : total_onions = 18) :
    ∃ fred_onions : ℕ, fred_onions = total_onions - (sara_onions + sally_onions) ∧ fred_onions = 9 :=
by
  have h_sum := h_sara.symm ▸ h_sally.symm ▸ (sara_onions + sally_onions)
  have h_fred := h_total.symm ▸ h_sum ▸ 9
  use 9
  finish

end fred_onions_l43_43089


namespace trapezoid_area_equal_l43_43017

namespace Geometry

-- Define the areas of the outer and inner equilateral triangles.
def outer_triangle_area : ℝ := 25
def inner_triangle_area : ℝ := 4

-- The number of congruent trapezoids formed between the triangles.
def number_of_trapezoids : ℕ := 4

-- Prove that the area of one trapezoid is 5.25 square units.
theorem trapezoid_area_equal :
  (outer_triangle_area - inner_triangle_area) / number_of_trapezoids = 5.25 := by
  sorry

end Geometry

end trapezoid_area_equal_l43_43017


namespace concurrency_of_perpendiculars_l43_43048

variable {A B C M_A M_B M_C M_A' M_B' M_C' P_A P_B P_C : Point}

-- Definitions of points and properties as conditions state
def is_midpoint (M : Point) (A B : Point) : Prop :=
  dist M A = dist M B

def is_arc_midpoint (M' : Point) (A B : Point) (circumcircle : Circle) : Prop :=
  -- Assuming a function minor_arc that gives the minor arc length
  minor_arc circumcircle A B / 2 = dist M' (circumcircle.center)

-- Given an acute triangle ABC
def acute_triangle (A B C : Point) : Prop :=
  ∠ A B C < 90 ∧ ∠ B C A < 90 ∧ ∠ C A B < 90

-- The main theorem statement
theorem concurrency_of_perpendiculars
  (acute_triangle A B C)
  (is_midpoint M_A B C)
  (is_midpoint M_B C A)
  (is_midpoint M_C A B)
  (is_arc_midpoint M_A' B C (circumcircle A B C))
  (is_arc_midpoint M_B' C A (circumcircle A B C))
  (is_arc_midpoint M_C' A B (circumcircle A B C))
  (P_A : Line (M_B ⟶ M_C) ∩ perpendicular M_B' ⟶ M_C' through A)
  (P_B : Line (M_C ⟶ M_A) ∩ perpendicular M_C' ⟶ M_A' through B)
  (P_C : Line (M_A ⟶ M_B) ∩ perpendicular M_A' ⟶ M_B' through C) :
  concurrent (M_A ⟶ P_A) (M_B ⟶ P_B) (M_C ⟶ P_C) :=
sorry

end concurrency_of_perpendiculars_l43_43048


namespace expression_in_terms_of_x_difference_between_x_l43_43447

variable (E x : ℝ)

theorem expression_in_terms_of_x (h1 : E / (2 * x + 15) = 3) : E = 6 * x + 45 :=
by 
  sorry

variable (x1 x2 : ℝ)

theorem difference_between_x (h1 : E / (2 * x1 + 15) = 3) (h2: E / (2 * x2 + 15) = 3) (h3 : x2 - x1 = 12) : True :=
by 
  sorry

end expression_in_terms_of_x_difference_between_x_l43_43447


namespace find_a_value_l43_43394

def f (a x : ℝ) : ℝ :=
if x ≥ 0 then 4^x else 2^(a - x)

theorem find_a_value (a : ℝ) (h : a ≠ 1) : f a (1 - a) = f a (a - 1) ↔ a = 1 / 2 :=
by
  sorry

end find_a_value_l43_43394


namespace problem_proof_l43_43494

variable (n : ℕ) (a : Fin n → ℝ) 

noncomputable def t (k : ℕ) : ℝ := ∑ i, (a i) ^ k

theorem problem_proof 
  (h_pos : ∀ i, 0 < a i ) : 
  (t n a 5) ^ 2 * (t n a 1) ^ 6 / 15 - (t n a 4) ^ 4 * (t n a 2) ^ 2 * (t n a 1) ^ 2 / 6 + (t n a 2) ^ 3 * (t n a 4) ^ 5 / 10 ≥ 0 := 
  sorry

end problem_proof_l43_43494


namespace slices_with_both_cheese_and_onions_l43_43949

theorem slices_with_both_cheese_and_onions :
  ∀ (total_slices cheese_slices onion_slices : ℕ), 
  total_slices = 16 → cheese_slices = 9 → onion_slices = 13 → 
  ∃ n : ℕ, (cheese_slices - n) + (onion_slices - n) + n = total_slices ∧ n = 6 := 
by
  intros total_slices cheese_slices onion_slices htotal hcheese honion
  use 6
  split
  {
    calc
      (cheese_slices - 6) + (onion_slices - 6) + 6
          = (9 - 6) + (13 - 6) + 6 : by rw [hcheese, honion]
      ... = 3 + 7 + 6 : by norm_num
      ... = 16 : by norm_num,
  }
  {
    refl
  }

end slices_with_both_cheese_and_onions_l43_43949


namespace total_cost_nancy_spends_l43_43265

def price_crystal_beads : ℝ := 12
def price_metal_beads : ℝ := 15
def sets_crystal_beads : ℕ := 3
def sets_metal_beads : ℕ := 4
def discount_crystal : ℝ := 0.10
def tax_metal : ℝ := 0.05

theorem total_cost_nancy_spends :
  sets_crystal_beads * price_crystal_beads * (1 - discount_crystal) + 
  sets_metal_beads * price_metal_beads * (1 + tax_metal) = 95.40 := 
  by sorry

end total_cost_nancy_spends_l43_43265


namespace twelve_women_reseated_l43_43585

/-- Define the sequence of number of ways the women can be reseated --/
def reseat_ways : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 3
| n := reseat_ways (n - 1) + reseat_ways (n - 2) + reseat_ways (n - 3)

/-- The number of ways twelve women can be reseated --/
theorem twelve_women_reseated : reseat_ways 12 = 927 := 
sorry

end twelve_women_reseated_l43_43585


namespace decomposition_l43_43243

open Real

def vector := ℝ × ℝ × ℝ

def x : vector := (-1, 7, -4)
def p : vector := (-1, 2, 1)
def q : vector := (2, 0, 3)
def r : vector := (1, 1, -1)

theorem decomposition :
  x = (2 : ℝ) • p + (-1 : ℝ) • q + (3 : ℝ) • r :=
sorry

end decomposition_l43_43243


namespace problem_1_simplified_problem_2_simplified_l43_43685

noncomputable def problem_1 : ℝ :=
  2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32

theorem problem_1_simplified : problem_1 = 3 * Real.sqrt 2 :=
  sorry

noncomputable def problem_2 : ℝ :=
  (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2

theorem problem_2_simplified : problem_2 = -7 + 2 * Real.sqrt 5 :=
  sorry

end problem_1_simplified_problem_2_simplified_l43_43685


namespace _l43_43526

variable (EFGH : Type) [rhombus : Rhombus EFGH]
variables (a b c d : EFGH)
variables (len_diagonal_EG len_perimeter : ℝ)
variable (d1 d2 : ℝ)

-- Assuming EFGH is a rhombus and specified lengths of sides and diagonal
def conditions
  [H1 : rhombus EFGH]
  (H2 : len_perimeter = 40)
  (H3 : len_diagonal_EG = 16):
  Prop :=
  by
    -- Extract side length from perimeter
    let side_length := len_perimeter / 4;
    
    -- Calculate half of the diagonal given
    let half_diagonal_EG := len_diagonal_EG / 2;
    
    -- Use Pythagorean theorem to find the other half diagonal
    let half_diagonal_FH := 
      sqrt (side_length ^ 2 - half_diagonal_EG ^ 2);
    
    -- Full length of the other diagonal
    let len_diagonal_FH := 2 * half_diagonal_FH;
    let area := len_diagonal_EG * len_diagonal_FH / 2;
    exact area = 96

end _l43_43526


namespace time_2556_hours_from_now_main_l43_43134

theorem time_2556_hours_from_now (h : ℕ) (mod_res : h % 12 = 0) :
  (3 + h) % 12 = 3 :=
by {
  sorry
}

-- Constants
def current_time : ℕ := 3
def hours_passed : ℕ := 2556
-- Proof input
def modular_result : hours_passed % 12 = 0 := by {
 sorry -- In the real proof, we should show that 2556 is divisible by 12
}

-- Main theorem instance
theorem main : (current_time + hours_passed) % 12 = 3 := 
  time_2556_hours_from_now hours_passed modular_result

end time_2556_hours_from_now_main_l43_43134


namespace parallel_lines_perpendicular_lines_l43_43429

-- Definitions of the lines
def l1 (a x y : ℝ) := x + a * y - 2 * a - 2 = 0
def l2 (a x y : ℝ) := a * x + y - 1 - a = 0

-- Statement for parallel lines
theorem parallel_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = 0 ∨ x = 1) → a = 1 :=
by 
  -- proof outline
  sorry

-- Statement for perpendicular lines
theorem perpendicular_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = y) → a = 0 :=
by 
  -- proof outline
  sorry

end parallel_lines_perpendicular_lines_l43_43429


namespace problem_D_correct_l43_43599

theorem problem_D_correct (a b : ℝ) (ha : 0 < a) (hb : b < 0) :
  (a / b + b / a = -(((-a) / b) + ((-b) / a)) ∧
   -( ((-a) / b) + ((-b) / a)) <= -2 * real.sqrt (((-a) / b) * ((-b) / a))) :=
by
  sorry

end problem_D_correct_l43_43599


namespace gcd_765432_654321_l43_43164

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43164


namespace gcd_765432_654321_l43_43215

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43215


namespace Anh_trip_time_l43_43517

noncomputable def total_travel_time (d_interstate d_mountain_pass speed_ratio time_mountain_pass : ℝ) :=
  let v := d_mountain_pass / (time_mountain_pass / 60) in
  let v_interstate := speed_ratio * v in
  let time_interstate := d_interstate / v_interstate in
  time_mountain_pass + time_interstate * 60

theorem Anh_trip_time
  (d_interstate : ℝ := 75)
  (d_mountain_pass : ℝ := 15)
  (speed_ratio : ℝ := 4)
  (time_mountain_pass : ℝ := 45):
  total_travel_time d_interstate d_mountain_pass speed_ratio time_mountain_pass = 101.25 :=
by 
  -- Proof omitted
  sorry

end Anh_trip_time_l43_43517


namespace inequality_holds_for_all_x_iff_a_in_interval_l43_43560

theorem inequality_holds_for_all_x_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by sorry

end inequality_holds_for_all_x_iff_a_in_interval_l43_43560


namespace circle_center_and_radius_l43_43590

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * y - 1 = 0 ↔ (x, y) = (0, 2) ∧ 5 = (0 - x)^2 + (2 - y)^2 :=
by sorry

end circle_center_and_radius_l43_43590


namespace trip_time_at_50mph_l43_43573

def distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem trip_time_at_50mph :
  let initial_speed := 80
  let initial_time := 5.5
  let new_speed := 50
  -- Calculate distance at initial speed and time
  let dist := distance initial_speed initial_time
  -- Calculate travel time at new_speed
  travel_time dist new_speed = 8.80 :=
by
  sorry

end trip_time_at_50mph_l43_43573


namespace total_annual_gain_l43_43299

-- Definitions based on given conditions
variable (A B C : Type) [Field ℝ]

-- Assume initial investments and time factors
variable (x : ℝ) (A_share : ℝ := 5000) -- A's share is Rs. 5000

-- Total annual gain to be proven
theorem total_annual_gain (x : ℝ) (A_share B_share C_share Total_Profit : ℝ) :
  A_share = 5000 → 
  B_share = (2 * x) * (6 / 12) → 
  C_share = (3 * x) * (4 / 12) → 
  (A_share / (x * 12)) * Total_Profit = 5000 → -- A's determined share from profit
  Total_Profit = 15000 := 
by 
  sorry

end total_annual_gain_l43_43299


namespace abs_abc_eq_one_l43_43059

variable (a b c : ℝ)

-- Conditions
axiom distinct_nonzero : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)
axiom condition : a^2 + 1/(b^2) = b^2 + 1/(c^2) ∧ b^2 + 1/(c^2) = c^2 + 1/(a^2)

theorem abs_abc_eq_one : |a * b * c| = 1 :=
by
  sorry

end abs_abc_eq_one_l43_43059


namespace prove_k_values_l43_43347

noncomputable def k_values : Set ℝ :=
  {k : ℝ | ∥(k • (⟨3, -2⟩ : ℝ × ℝ) - ⟨5, 8⟩)∥ = 3 * Real.sqrt 13}

theorem prove_k_values :
  k_values = {18 / 13, -20 / 13} := by
  sorry

end prove_k_values_l43_43347


namespace domain_of_log_function_l43_43107

noncomputable def f (x : ℝ) : ℝ := log 3 (x - 1)

theorem domain_of_log_function :
  ∀ x : ℝ, (x > 1) ↔ x ∈ set.Ioi 1 :=
by
  sorry

end domain_of_log_function_l43_43107


namespace length_of_train_l43_43296

noncomputable def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def total_distance (speed_m_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_s * time_s

noncomputable def train_length (total_distance : ℝ) (bridge_length : ℝ) : ℝ :=
  total_distance - bridge_length

theorem length_of_train
  (speed_kmh : ℝ)
  (time_s : ℝ)
  (bridge_length : ℝ)
  (speed_in_kmh : speed_kmh = 45)
  (time_in_seconds : time_s = 30)
  (length_of_bridge : bridge_length = 220.03) :
  train_length (total_distance (speed_kmh_to_ms speed_kmh) time_s) bridge_length = 154.97 :=
by
  sorry

end length_of_train_l43_43296


namespace yoongi_correct_calculation_l43_43933

theorem yoongi_correct_calculation (x : ℕ) (h : x + 9 = 30) : x - 7 = 14 :=
sorry

end yoongi_correct_calculation_l43_43933


namespace cross_product_correct_l43_43348

def v : ℝ × ℝ × ℝ := (-3, 4, 5)
def w : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1,
 a.2.2 * b.1 - a.1 * b.2.2,
 a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct : cross_product v w = (21, 22, -5) :=
by
  sorry

end cross_product_correct_l43_43348


namespace angle_DSO_is_67_5_l43_43479

-- Original conditions
variables (DOG : Type) [triangle DOG]
variables (D G O : DOG)
variables (DG GO : Segment DOG) -- DG and GO are segments of the triangle
variables (angle_DGO angle_DOG : ℝ) (OS : Ray DOG DOG)
variables (bisect_OS_DOG : isBisection OS angle_DOG)
variables (angle_DSO : ℝ)

-- given conditions
axiom angle_DGO_eq_angle_DOG : angle_DGO = angle_DOG
axiom angle_DOG_30 : angle_DOG = 30

-- conclusion to prove
noncomputable def angle_DSO_value : Prop :=
  angle_DSO = 67.5

theorem angle_DSO_is_67_5 :
  angle_DGO = angle_DOG → angle_DOG = 30 → isBisection OS angle_DOG → angle_DSO = 67.5 :=
by
  sorry

end angle_DSO_is_67_5_l43_43479


namespace limit_value_l43_43412

noncomputable def f (x : ℝ) := 2 * Real.log(3 * x) + 8 * x + 1

theorem limit_value :
  tendsto (λ Δx, (f (1 - 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-20)) :=
sorry

end limit_value_l43_43412


namespace functional_identity_l43_43349

theorem functional_identity (f : ℝ → ℝ) :
  (∀ (x y : ℝ), f(x - f(x - y)) + x = f(x + y)) →
  (∀ (x : ℝ), f(x) = x) :=
begin
  intros,
  sorry
end

end functional_identity_l43_43349


namespace force_of_water_pressure_on_dam_l43_43653

-- Defining the given conditions
def upper_base := 60
def lower_base := 20
def height := 10
variable (ρ : ℝ) (g : ℝ)

-- Statement of the theorem
theorem force_of_water_pressure_on_dam :
  ρ * g * (height^2 / 3) = ρ * g * (10^2 / 3) :=
by
  sorry

end force_of_water_pressure_on_dam_l43_43653


namespace hyperbolas_same_asymptotes_l43_43993

-- Define the given hyperbolas
def hyperbola1 (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
def hyperbola2 (x y M : ℝ) : Prop := (y^2 / 25) - (x^2 / M) = 1

-- The main theorem statement
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, hyperbola1 x y → hyperbola2 x y M) ↔ M = 225/16 :=
by
  sorry

end hyperbolas_same_asymptotes_l43_43993


namespace sum_of_powers_l43_43232

theorem sum_of_powers : (-1: ℤ) ^ 2006 - (-1) ^ 2007 + 1 ^ 2008 + 1 ^ 2009 - 1 ^ 2010 = 3 := by
  sorry

end sum_of_powers_l43_43232


namespace expression_of_f_monotonicity_of_f_inequality_solution_l43_43398

-- Definitions of the problem
def f (x : ℝ) : ℝ := x / (1 + x^2)

-- Problem (1)
theorem expression_of_f :
  (∀ x, f x = (∃ a b, f x = (ax + b) / (1 + x^2) ∧ ∀ y, f y = -f (-y)) ∧ f (1/2) = 2/5) → 
  f x = x / (1 + x^2) := sorry

-- Problem (2)
theorem monotonicity_of_f :
  (∀ x, f x = x / (1 + x^2)) → (∀ x1 x2, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2) := sorry

-- Problem (3)
theorem inequality_solution :
  (∀ x, f x = x / (1 + x^2)) → (∀ t, f (t-1) + f t < 0 ↔ 0 < t ∧ t < 1/2) := sorry

end expression_of_f_monotonicity_of_f_inequality_solution_l43_43398


namespace gcd_765432_654321_l43_43185

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43185


namespace star_perimeter_difference_zero_l43_43049

-- Define the properties of an equiangular convex hexagon with a perimeter of 1
structure EquiangularHexagon :=
  (sides : Fin 6 → ℝ)
  (perimeter_eq_one : (Finset.univ.sum sides) = 1)
  (equiangular : ∀ i, angle_eq (sides i) 120)

-- Define the star polygon formed by extending the hexagon's sides
structure StarPolygon :=
  (equilateral_triangles : Fin 6 → ℝ)
  (side_lengths_eq : ∀ i, star_side_length (equilateral_triangles i) (sides i))

-- The theorem statement
theorem star_perimeter_difference_zero (hex : EquiangularHexagon) : 
  ∀ star : StarPolygon, (star.max_perimeter hex - star.min_perimeter hex) = 0 :=
sorry

end star_perimeter_difference_zero_l43_43049


namespace max_area_triangle_QRC_l43_43113

noncomputable def maxAreaOfTriangle
  (k : ℝ)
  (Q R : ℝ × ℝ)
  (l : ∀ x y : ℝ, k * x - y + 2 = 0)
  (C : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 12 = 0)
  : ℝ :=
  -- Maximum area is 8
  8

theorem max_area_triangle_QRC (k : ℝ) :
  ∃ Q R : ℝ × ℝ,
    let l x y := k * x - y + 2 = 0
    let C := x^2 + y^2 - 4 * x - 12 = 0
    maxAreaOfTriangle k Q R l C = 8 :=
sorry

end max_area_triangle_QRC_l43_43113


namespace medicine_liquid_poured_l43_43277

theorem medicine_liquid_poured (x : ℝ) (h : 63 * (1 - x / 63) * (1 - x / 63) = 28) : x = 18 :=
by
  sorry

end medicine_liquid_poured_l43_43277


namespace probability_of_satisfaction_l43_43672

-- Definitions for the conditions given in the problem
def dissatisfied_customers_leave_negative_review_probability : ℝ := 0.8
def satisfied_customers_leave_positive_review_probability : ℝ := 0.15
def negative_reviews : ℕ := 60
def positive_reviews : ℕ := 20
def expected_satisfaction_probability : ℝ := 0.64

-- The problem to prove
theorem probability_of_satisfaction :
  ∃ p : ℝ, (dissatisfied_customers_leave_negative_review_probability * (1 - p) = negative_reviews / (negative_reviews + positive_reviews)) ∧
           (satisfied_customers_leave_positive_review_probability * p = positive_reviews / (negative_reviews + positive_reviews)) ∧
           p = expected_satisfaction_probability := 
by
  sorry

end probability_of_satisfaction_l43_43672


namespace min_holiday_days_l43_43029

theorem min_holiday_days 
  (rained_days : ℕ) 
  (sunny_mornings : ℕ)
  (sunny_afternoons : ℕ) 
  (condition1 : rained_days = 7) 
  (condition2 : sunny_mornings = 5) 
  (condition3 : sunny_afternoons = 6) :
  ∃ (days : ℕ), days = 9 :=
by
  -- The specific steps of the proof are omitted as per the instructions
  sorry

end min_holiday_days_l43_43029


namespace max_sum_no_change_forshilling_l43_43225

def shilling := ℝ
def pence := ℝ

noncomputable def is_valid_combination (coins : list ℝ) : Prop :=
  (∀ combination : list ℝ, (combination.sum = 10) → (combination ⊆ coins) → false) 
  ∧ (∀ coin, coin ∈ coins → (coin = 5 ∨ coin = 2.5 ∨ coin = 2 ∨ coin = 1 ∨ coin = 0.5 ∨ coin = 0.25))

theorem max_sum_no_change_forshilling :
  ∃ s : shilling, ∃ p : pence, 15.75 ≤ s + p / 12 ∧ is_valid_combination [5, 2.5, 2, 1, 0.5, 0.25] := 
begin
  use 15,
  use 9 / 12,
  split,
  { norm_num },
  { split,
    { intros combination hcomb hsub,
      sorry },
    { intros coin hcoin,
      repeat {cases hcoin; norm_num}} }
end

end max_sum_no_change_forshilling_l43_43225


namespace average_speed_l43_43872

def s (t : ℝ) : ℝ := 3 + t^2

theorem average_speed {t1 t2 : ℝ} (h1 : t1 = 2) (h2: t2 = 2.1) :
  (s t2 - s t1) / (t2 - t1) = 4.1 :=
by
  sorry

end average_speed_l43_43872


namespace coordinates_Q_at_pi_by_4_range_of_g_l43_43022

noncomputable def P : ℝ × ℝ := (1/2, (real.sqrt 3) / 2)
noncomputable def Q (x : ℝ) : ℝ × ℝ :=
  (real.cos ((real.pi / 3) + x) * P.fst - real.sin ((real.pi / 3) + x) * P.snd,
   real.sin ((real.pi / 3) + x) * P.fst + real.cos ((real.pi / 3) + x) * P.snd)

def f (x : ℝ) : ℝ :=
  P.fst * (Q x).fst + P.snd * (Q x).snd

def g (x : ℝ) : ℝ :=
  f(x) * f(x + (real.pi / 3))

theorem coordinates_Q_at_pi_by_4 :
  Q (real.pi / 4) = ((real.sqrt 2 - real.sqrt 6) / 4, (real.sqrt 6 + real.sqrt 2) / 4) :=
sorry

theorem range_of_g :
  set.Icc (-1/4 : ℝ) (3/4) ⊆ set.range g :=
sorry

end coordinates_Q_at_pi_by_4_range_of_g_l43_43022


namespace minimum_value_fraction_l43_43870

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

/-- Given that the function f(x) = log_a(4x-3) + 1 (where a > 0 and a ≠ 1) has a fixed point A(m, n), 
if for any positive numbers x and y, mx + ny = 3, 
then the minimum value of 1/(x+1) + 1/y is 1. -/
theorem minimum_value_fraction (a x y : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (hx : x + y = 3) : 
  (1 / (x + 1) + 1 / y) = 1 := 
sorry

end minimum_value_fraction_l43_43870


namespace faster_pipe_rate_l43_43520

-- Set up our variables and the condition
variable (F S : ℝ)
variable (n : ℕ)

-- Given conditions
axiom S_rate : S = 1 / 180
axiom combined_rate : F + S = 1 / 36
axiom faster_rate : F = n * S

-- Theorem to prove
theorem faster_pipe_rate : n = 4 := by
  sorry

end faster_pipe_rate_l43_43520


namespace least_pieces_to_take_away_l43_43696

-- Define the conditions
def candies : ℕ := 25
def sisters : ℕ := 4

-- The mathematical problem translated into Lean 4: prove that the least number of pieces
-- Daniel should take away so that he could distribute the candy equally among his sisters is 1.
theorem least_pieces_to_take_away (c : ℕ) (s : ℕ) (sisters_division : s > 0) : c = 25 → s = 4 → (c % s) = 1 → ∃ k : ℕ, c - k = 24 ∧ k = 1 :=
by
  -- Define constants for candy and sisters
  assume hc : c = 25,
  assume hs : s = 4,

  -- Calculate remainder
  have rem : c % s = 1,
  from hc ▸ hs ▸ Nat.mod_eq_of_lt (by norm_num : 25 < 4 * 6),

  -- Remove k candies
  use 1,
  split,
  {
    -- This shows that after taking away 1 candy, 24 remain
    calc
      c - 1 = 25 - 1 : by rw hc
      ... = 24 : by norm_num,
  },
  {
    -- 1 candy was taken away
    norm_num,
  }

end least_pieces_to_take_away_l43_43696


namespace num_male_students_selected_l43_43080

def total_students := 220
def male_students := 60
def selected_female_students := 32

def selected_male_students (total_students male_students selected_female_students : Nat) : Nat :=
  (selected_female_students * male_students) / (total_students - male_students)

theorem num_male_students_selected : selected_male_students total_students male_students selected_female_students = 12 := by
  unfold selected_male_students
  sorry

end num_male_students_selected_l43_43080


namespace find_f_zero_l43_43559

theorem find_f_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y - x * y) 
  (h1 : f 1 = 1) : 
  f 0 = 0 := 
sorry

end find_f_zero_l43_43559


namespace det_B_l43_43497

open Matrix

-- Define matrix B
def B (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 2], ![-3, y]]

-- Define the condition B + 2 * B⁻¹ = 0
def condition (x y : ℝ) : Prop :=
  let Binv := (1 / (x * y + 6)) • ![![y, -2], ![3, x]]
  B x y + 2 • Binv = 0

-- Prove that if the condition holds, then det B = 2
theorem det_B (x y : ℝ) (h : condition x y) : det (B x y) = 2 :=
  sorry

end det_B_l43_43497


namespace min_value_d_l43_43841

theorem min_value_d (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (unique_solution : ∃! x y : ℤ, 2 * x + y = 2007 ∧ y = (abs (x - a) + abs (x - b) + abs (x - c) + abs (x - d))) :
  d = 504 :=
sorry

end min_value_d_l43_43841


namespace area_shaded_region_l43_43554

theorem area_shaded_region (r R : ℝ) (h1 : 0 < r) (h2 : r < R)
  (h3 : 60 = 2 * sqrt (R^2 - r^2)) :
  π * (R^2 - r^2) = 900 * π :=
by
  sorry

end area_shaded_region_l43_43554


namespace keith_initial_cards_l43_43037

theorem keith_initial_cards (new_cards : ℕ) (cards_after_incident : ℕ) (total_cards : ℕ) :
  new_cards = 8 →
  cards_after_incident = 46 →
  total_cards = 2 * cards_after_incident →
  (total_cards - new_cards) = 84 :=
by
  intros
  sorry

end keith_initial_cards_l43_43037


namespace keith_cards_initial_count_l43_43040

theorem keith_cards_initial_count :
  ∃ (x : ℕ), let final_count := 46 in
  let cards_after_dog := 2 * final_count in
  let total_after_buying := cards_after_dog - 8 in
  (x = total_after_buying) ∧ (total_after_buying = 84) :=
begin
  sorry
end

end keith_cards_initial_count_l43_43040


namespace find_m_and_other_root_l43_43748

theorem find_m_and_other_root (m a : ℝ) 
  (h1: (1:ℝ) is a root of (λ x, x^2 - 4 * x + m + 1))
  (h2: a + 1 = 4): 
  m = 2 ∧ a = 3 :=
by
  sorry

end find_m_and_other_root_l43_43748


namespace rhombus_area_correct_l43_43528

def is_rhombus (E F G H : Type) [metric_space E]
  (EF : ℝ) (FG : ℝ) (GH : ℝ) (HE : ℝ) : Prop :=
metric.dist E F = EF ∧ metric.dist F G = FG ∧
metric.dist G H = GH ∧ metric.dist H E = HE ∧
EF = FG ∧ FG = GH ∧ GH = HE

def diagonal_length (E I G : Type) [metric_space E]
  (EG : ℝ) (EI IG : ℝ) : Prop :=
metric.dist E G = EG ∧ metric.dist E I = EI ∧
metric.dist I G = IG ∧ EI + IG = EG

def rhombus_area (EG FH : ℝ) : ℝ :=
(EG * FH) / 2

theorem rhombus_area_correct (E F G H I : Type) [metric_space E]
  (EFmet : ℝ) (EG FH : ℝ)
  (h1 : is_rhombus E F G H EFmet EFmet EFmet EFmet)
  (h2 : diagonal_length E I G EG (EG / 2) (EG / 2))
  (HF : metric.dist F I = √(EFmet^2 - (EG / 2)^2))
  (FH := 2 * HF) :
  rhombus_area EG FH = 96 :=
by
  sorry

end rhombus_area_correct_l43_43528


namespace find_number_l43_43628

-- Define given numbers
def a : ℕ := 555
def b : ℕ := 445

-- Define given conditions
def sum : ℕ := a + b
def difference : ℕ := a - b
def quotient : ℕ := 2 * difference
def remainder : ℕ := 30

-- Define the number we're looking for
def number := sum * quotient + remainder

-- The theorem to prove
theorem find_number : number = 220030 := by
  -- Use the let expressions to simplify the calculation for clarity
  let sum := a + b
  let difference := a - b
  let quotient := 2 * difference
  let number := sum * quotient + remainder
  show number = 220030
  -- Placeholder for proof
  sorry

end find_number_l43_43628


namespace root_in_interval_l43_43875

theorem root_in_interval (a b c : ℝ)
  (h_poly : ∃ x1 x2 x3 : ℝ, ∀ x : ℝ, (x - x1) * (x - x2) * (x - x3) = x^3 + a * x^2 + b * x + c)
  (h_sum : 2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ x : ℝ, x ∈ set.Icc (0 : ℝ) (2 : ℝ) ∧ (x^3 + a * x^2 + b * x + c = 0) :=
sorry

end root_in_interval_l43_43875


namespace exists_congruent_triangle_with_colored_sides_l43_43315

noncomputable def triangle_congruent_with_colored_sides (T : Triangle) (color : Point → Fin 1992) : Prop :=
  ∃ T' : Triangle, 
    T' ≅ T ∧ 
    ∀ (side : Side T'), ∃ (p : Point), p ∈ side ∧ color p = color (point_on_other_side side)

theorem exists_congruent_triangle_with_colored_sides (T : Triangle) (color : Point → Fin 1992) 
  (hused : ∀ k : Fin 1992, ∃ p : Point, color p = k) : triangle_congruent_with_colored_sides T color :=
sorry

end exists_congruent_triangle_with_colored_sides_l43_43315


namespace sin_225_l43_43688

-- Definitions of the unit circle and the point Q at 225 degrees
def unit_circle : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def Q : ℝ × ℝ := (-real.sqrt 2 / 2, -real.sqrt 2 / 2)

-- The problem statement (proof omitted)
theorem sin_225 : Q ∈ unit_circle ∧ Q = ⟨-real.sqrt 2 / 2, -real.sqrt 2 / 2⟩ → real.sin (225 * (real.pi / 180)) = -real.sqrt 2 / 2 :=
by
  intro h
  sorry

end sin_225_l43_43688


namespace inverse_function_passes_through_P_l43_43775

-- Define the function f(x) = a^(x + 2) for any positive a ≠ 1
def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

-- Assume a is a positive number and a ≠ 1
variables (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1)

-- Define the point P that the inverse function of f passes through
def point_P (a : ℝ) : ℝ × ℝ := (1, -2)

-- The statement that we need to prove
theorem inverse_function_passes_through_P (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) :
  ∃ x, f a x = 1 →
  (∃ y, (1, -2) = (y, x)) :=
by {
  sorry
}

end inverse_function_passes_through_P_l43_43775


namespace B_listing_method_l43_43423

-- Definitions for given conditions
def A : Set ℤ := {-2, -1, 1, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t*t}

-- The mathematically equivalent proof problem
theorem B_listing_method :
  B = {4, 1, 9, 16} := 
by {
  sorry
}

end B_listing_method_l43_43423


namespace speed_of_second_bus_thm_l43_43153

noncomputable def speed_of_second_bus (v : ℕ) : Prop :=
  let speed_bus_1 := 55
  let time : ℕ := 4
  let distance := 460
  (speed_bus_1 * time) + (v * time) = distance

theorem speed_of_second_bus_thm : speed_of_second_bus 60 :=
by
  simp [speed_of_second_bus]
  -- The actual proof steps would follow here
  sorry

end speed_of_second_bus_thm_l43_43153


namespace max_sum_of_factors_l43_43803

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 3003) : A + B + C ≤ 117 :=
sorry

end max_sum_of_factors_l43_43803


namespace calculate_PR_length_l43_43865

noncomputable def side_length : ℝ := 2
noncomputable def num_triangles : ℕ := 6

noncomputable def PR : ℝ := 2 * Real.sqrt 13

theorem calculate_PR_length : 
  let equilateral_triangle_area (side : ℝ) := (Real.sqrt 3) / 4 * side^2 in
  let parallelogram_area (num_triangles : ℕ) (triangle_area : ℝ) := num_triangles * triangle_area in
  let parallelogram_diagonal (side_length : ℝ) := Real.sqrt (4 * side_length ^ 2 + 3 * side_length ^ 2) in
  parallelogram_diagonal side_length = PR := 
by
  sorry

end calculate_PR_length_l43_43865


namespace lecture_minutes_per_disc_l43_43293

theorem lecture_minutes_per_disc 
  (total_minutes : ℕ)
  (max_disc_capacity : ℕ)
  (total_minutes_eq : total_minutes = 480)
  (max_disc_capacity_eq : max_disc_capacity = 70) :
  let number_of_discs := (total_minutes + max_disc_capacity - 1) / max_disc_capacity in
  total_minutes / number_of_discs = 68 :=
by
  sorry

end lecture_minutes_per_disc_l43_43293


namespace linear_regression_equation_demand_prediction_l43_43105

def data_x : List ℝ := [12, 11, 10, 9, 8]
def data_y : List ℝ := [5, 6, 8, 10, 11]

noncomputable def mean_x : ℝ := (12 + 11 + 10 + 9 + 8) / 5
noncomputable def mean_y : ℝ := (5 + 6 + 8 + 10 + 11) / 5

noncomputable def numerator : ℝ := 
  (12 - mean_x) * (5 - mean_y) + 
  (11 - mean_x) * (6 - mean_y) +
  (10 - mean_x) * (8 - mean_y) +
  (9 - mean_x) * (10 - mean_y) +
  (8 - mean_x) * (11 - mean_y)

noncomputable def denominator : ℝ := 
  (12 - mean_x)^2 + 
  (11 - mean_x)^2 +
  (10 - mean_x)^2 +
  (9 - mean_x)^2 +
  (8 - mean_x)^2

noncomputable def slope_b : ℝ := numerator / denominator
noncomputable def intercept_a : ℝ := mean_y - slope_b * mean_x

theorem linear_regression_equation :
  (slope_b = -1.6) ∧ (intercept_a = 24) :=
by
  sorry

noncomputable def predicted_y (x : ℝ) : ℝ :=
  slope_b * x + intercept_a

theorem demand_prediction :
  predicted_y 6 = 14.4 ∧ (predicted_y 6 < 15) :=
by
  sorry

end linear_regression_equation_demand_prediction_l43_43105


namespace visitors_previous_day_l43_43301

theorem visitors_previous_day (total_visitors : ℕ) (current_day_visitors : ℕ) (h1 : total_visitors = 406) (h2 : current_day_visitors = 132) : 
  total_visitors - current_day_visitors = 274 :=
by
  rw [h1, h2]
  exact eq.refl _

end visitors_previous_day_l43_43301


namespace average_marks_l43_43015

theorem average_marks (n : ℕ) (m : ℕ) (a b c d : ℕ) 
  (h1 : n = 27) 
  (h2 : m = 95) 
  (h3 : a = 5) 
  (h4 : b = 3) 
  (h5 : c = 45) 
  (h6 : d = n - (a + b)) 
  (h7 : d = 19)
  (h8 : ((a * b) * m + d * c) / n = 49.26): 
  ((5 * 95) + (3 * 0) + (19 * 45)) / 27 = 49.26 := by
  sorry

end average_marks_l43_43015


namespace scientific_notation_l43_43518

theorem scientific_notation (n : ℤ) (e : ℤ) (h : 28000000 = n * 10^e) : n = 28 ∧ e = 6 :=
sorry

end scientific_notation_l43_43518


namespace quadratic_two_distinct_roots_example_l43_43706

theorem quadratic_two_distinct_roots_example :
  ∃ (m : ℝ), (x^2 - x + m = 0 ∧ 1 - 4 * m > 0) :=
begin
  use 0,
  split,
  { sorry },  -- this would be the proof that the quadratic equation with m = 0 is correctly set.
  { linarith }
end

end quadratic_two_distinct_roots_example_l43_43706


namespace stock_percentage_change_l43_43544

theorem stock_percentage_change (x : ℝ) :
  let day1 := 0.75 * x
  let day2 := 1.40 * day1
  let percentage_change := (day2 - x) / x * 100
  percentage_change = 5 :=
by
  let day1 := 0.75 * x
  let day2 := 1.40 * day1
  let percentage_change := (day2 - x) / x * 100
  have h1 : day1 = 0.75 * x := by rfl
  have h2 : day2 = 1.40 * day1 := by rfl
  have h3 : percentage_change = ((1.40 * 0.75 * x - x) / x) * 100 := by rfl
  rw [h1, h2, h3]
  sorry

end stock_percentage_change_l43_43544


namespace f_at_count_l43_43362

def f (a b c : ℕ) : ℕ := (a * b * c) / (Nat.gcd (Nat.gcd a b) c * Nat.lcm (Nat.lcm a b) c)

def is_f_at (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≤ 60 ∧ y ≤ 60 ∧ z ≤ 60 ∧ f x y z = n

theorem f_at_count : ∃ (n : ℕ), n = 70 ∧ ∀ k, is_f_at k → k ≤ 70 := 
sorry

end f_at_count_l43_43362


namespace tangent_line_at_x_eq_1_range_m_two_zeros_l43_43413

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2 + 2 * x

noncomputable def g (x m : ℝ) : ℝ := 2 * Real.log x - x^2 + m

theorem tangent_line_at_x_eq_1
  (x : ℝ) (hx : x = 1) (a : ℝ) (ha : a = 2) :
  ∃ k b : ℝ, k = 2 ∧ b = -1 ∧ (∀ y : ℝ, has_tangent f (λ x, k * x + b) y) :=
sorry

theorem range_m_two_zeros
  (m : ℝ) :
  ∃ a b : ℝ, (1 < m ∧ m ≤ 2 + 1 / Real.exp 2) :=
sorry

end tangent_line_at_x_eq_1_range_m_two_zeros_l43_43413


namespace cube_hexagonal_cross_section_area_l43_43695

theorem cube_hexagonal_cross_section_area (a : ℝ) (h : 0 < a) : 
  ∃ (A : ℝ), A = (3 * real.sqrt 3 / 4) * a^2 :=
by
  sorry

end cube_hexagonal_cross_section_area_l43_43695


namespace range_of_m_l43_43781

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 :=
by
  intro h
  sorry

end range_of_m_l43_43781


namespace squares_similar_l43_43239

def similar (s1 s2 : Type) [HasShape s1] [HasShape s2] : Prop :=
  ∀ (a b : s1) (c d : s2), similar_shapes a b c d

structure Square (side_length : ℝ) :=
  (is_square : is_shape_square side_length)

theorem squares_similar (s1 s2 : Square) : similar s1 s2 := by
  sorry

end squares_similar_l43_43239


namespace area_of_TURS_eq_area_of_PQRS_l43_43608

-- Definition of the rectangle PQRS
structure Rectangle where
  length : ℕ
  width : ℕ
  area : ℕ

-- Definition of the trapezoid TURS
structure Trapezoid where
  base1 : ℕ
  base2 : ℕ
  height : ℕ
  area : ℕ

-- Condition: PQRS is a rectangle whose area is 20 square units
def PQRS : Rectangle := { length := 5, width := 4, area := 20 }

-- Question: Prove the area of TURS equals area of PQRS
theorem area_of_TURS_eq_area_of_PQRS (TURS_area : ℕ) : TURS_area = PQRS.area :=
  sorry

end area_of_TURS_eq_area_of_PQRS_l43_43608


namespace odd_number_probability_limit_l43_43911

axiom pn_seq : ℕ → ℚ
axiom p1_initial : pn_seq 1 = 1 / 2
axiom recursive_relation : ∀ (n : ℕ), pn_seq (n + 1) = (1 / 4) * pn_seq n + (1 / 4)

theorem odd_number_probability_limit : 
  filter.tendsto pn_seq filter.at_top (nhds (1 / 3)) :=
sorry

end odd_number_probability_limit_l43_43911


namespace trajectory_of_point_l43_43117

theorem trajectory_of_point (x y z : ℝ) (h : y = 3) : 
  ∃ (plane : set (ℝ × ℝ × ℝ)), plane = {p : ℝ × ℝ × ℝ | p.snd.snd = 3} := sorry

end trajectory_of_point_l43_43117


namespace correct_calculation_among_options_l43_43663

theorem correct_calculation_among_options :
  (∀ (x : ℝ), x^3 - x^2 ≠ x) ∧
  (∀ (a : ℝ), a ≠ 0 → a^{10} / a^9 = a) ∧
  (∀ (p q : ℝ), (-3 * p * q)^2 ≠ 6 * p * q) ∧
  (∀ (x : ℝ), x^3 * x^2 ≠ x^6) :=
begin
  sorry
end

end correct_calculation_among_options_l43_43663


namespace yellow_jelly_bean_probability_l43_43283

theorem yellow_jelly_bean_probability :
  let p_red := 0.15
  let p_orange := 0.35
  let p_green := 0.25
  let p_yellow := 1 - (p_red + p_orange + p_green)
  p_yellow = 0.25 := by
    let p_red := 0.15
    let p_orange := 0.35
    let p_green := 0.25
    let p_yellow := 1 - (p_red + p_orange + p_green)
    show p_yellow = 0.25
    sorry

end yellow_jelly_bean_probability_l43_43283


namespace average_students_present_l43_43016

-- Define the total number of students
def total_students : ℝ := 50

-- Define the absent rates for each day
def absent_rate_mon : ℝ := 0.10
def absent_rate_tue : ℝ := 0.12
def absent_rate_wed : ℝ := 0.15
def absent_rate_thu : ℝ := 0.08
def absent_rate_fri : ℝ := 0.05

-- Define the number of students present each day
def present_mon := (1 - absent_rate_mon) * total_students
def present_tue := (1 - absent_rate_tue) * total_students
def present_wed := (1 - absent_rate_wed) * total_students
def present_thu := (1 - absent_rate_thu) * total_students
def present_fri := (1 - absent_rate_fri) * total_students

-- Define the statement to prove
theorem average_students_present : 
  (present_mon + present_tue + present_wed + present_thu + present_fri) / 5 = 45 :=
by 
  -- The proof would go here
  sorry

end average_students_present_l43_43016


namespace find_range_of_f_minus_x_l43_43693

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then -x
  else x - 1

theorem find_range_of_f_minus_x : 
  set.range (λ x : ℝ, f x - x) = set.Icc (-4 : ℝ) 2 := sorry

end find_range_of_f_minus_x_l43_43693


namespace expectation_of_non_moving_surnames_l43_43525

noncomputable def expected_non_moving_surnames (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (1 : ℝ) / (k + 1)

theorem expectation_of_non_moving_surnames (n : ℕ) :
  ∑ k in finset.range n, (1 : ℝ) / (k + 1) =
  ∑ k in finset.range n, (1 : ℝ) / (k + 1) :=
by
  sorry

end expectation_of_non_moving_surnames_l43_43525


namespace inequality_solution_set_l43_43884

noncomputable def solution_set : set ℝ :=
  {x | (4 - x^2 ≥ 0 ∧ x > 0) ∨ (4 - x^2 ≥ 1 ∧ x < 0)}

theorem inequality_solution_set :
  {x : ℝ | sqrt (4 - x^2) + abs x / x ≥ 0} = set.union (set.Ioo (-(real.sqrt 3)) 0) (set.Icc 0 2) :=
by
  sorry

end inequality_solution_set_l43_43884


namespace domain_of_sqrt_expression_l43_43867

noncomputable def domain_of_function : Set ℝ := {x | 0 ≤ x}

theorem domain_of_sqrt_expression :
  (∀ x : ℝ, (1 - (1 / 2)^x ≥ 0) ↔ x ∈ domain_of_function) :=
by
  sorry

end domain_of_sqrt_expression_l43_43867


namespace a1_not_in_neg2_1_l43_43044

theorem a1_not_in_neg2_1 (a : ℕ → ℝ)
  (h : ∀ n, a (n + 1) = real.sqrt (a n ^ 2 + a n - 1)) :
  a 1 ≤ -2 ∨ a 1 ≥ 1 := 
sorry

end a1_not_in_neg2_1_l43_43044


namespace polynomial_problem_l43_43495

noncomputable def F (x : ℝ) : ℝ := sorry

theorem polynomial_problem
  (F : ℝ → ℝ)
  (h1 : F 4 = 22)
  (h2 : ∀ x : ℝ, (F (2 * x) / F (x + 2) = 4 - (16 * x + 8) / (x^2 + x + 1))) :
  F 8 = 1078 / 9 := sorry

end polynomial_problem_l43_43495


namespace present_ratio_students_teachers_l43_43568

/-- The present ratio of students to teachers is 50 to 1 given the conditions. -/
theorem present_ratio_students_teachers (S : ℕ) (T : ℕ) (hT : T = 3)
    (h_ratio : (S + 50) / 8 = 25) : S / T = 50 :=
by
  -- Define T (number of teachers)
  rewrite [hT]
  -- Simplify the problem using given conditions
  have h_students : S + 50 = 25 * 8 :=
    by linarith
  -- Solve for S
  have hS : S = 200 - 50 := 
    by linarith
  -- Substitute S back to find the ratio
  have hR : S = 150 := by linarith
  -- Show the final ratio
  have h_final : 150 / 3 = 50 := 
    by norm_num
  exact h_final

end present_ratio_students_teachers_l43_43568


namespace number_of_valid_grids_l43_43690

-- Define the concept of a grid and the necessary properties
structure Grid (n : ℕ) :=
  (cells: Fin (n * n) → ℕ)
  (unique: Function.Injective cells)
  (ordered_rows: ∀ i j : Fin n, i < j → cells ⟨i * n + j, sorry⟩ > cells ⟨i * n + j - 1, sorry⟩)
  (ordered_columns: ∀ i j : Fin n, i < j → cells ⟨j * n + i, sorry⟩ > cells ⟨(j - 1) * n + i, sorry⟩)

-- Define the 4x4 grid
def grid_4x4 := Grid 4

-- Statement of the problem: prove there are 2 valid grid_4x4 configurations
theorem number_of_valid_grids : ∃ g : grid_4x4, (∃ g1 g2 : grid_4x4, (g1 ≠ g2) ∧ (∀ g3 : grid_4x4, g3 = g1 ∨ g3 = g2)) :=
  sorry

end number_of_valid_grids_l43_43690


namespace probability_of_drawing_red_second_draw_l43_43905

theorem probability_of_drawing_red_second_draw :
  (∀ (balls : list ℕ),
    length balls = 5 →
    count (λ x, x = 0) balls = 3 →
    count (λ x, x = 1) balls = 2 →
    noncomputable_prob (draw_with_replacement balls 2) (λ draws, draws.nth 1 = some 0) = 3/5) :=
begin
  sorry
end

end probability_of_drawing_red_second_draw_l43_43905


namespace polynomial_evaluation_l43_43379

theorem polynomial_evaluation (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2005 = 2006 :=
sorry

end polynomial_evaluation_l43_43379


namespace multiplier_for_obsolete_books_l43_43335

theorem multiplier_for_obsolete_books 
  (x : ℕ) 
  (total_books_removed number_of_damaged_books : ℕ) 
  (h1 : total_books_removed = 69) 
  (h2 : number_of_damaged_books = 11) 
  (h3 : number_of_damaged_books + (x * number_of_damaged_books - 8) = total_books_removed) 
  : x = 6 := 
by 
  sorry

end multiplier_for_obsolete_books_l43_43335


namespace actual_distance_between_cities_l43_43866

/-- Problem Statement:
   Prove that the actual distance between the two city centers is 2400 km given that 
   the distance between them on the map is 120 cm and the scale is 1 cm : 20 km. -/

theorem actual_distance_between_cities
  (map_distance : ℝ)
  (scale_factor : ℝ)
  (actual_distance : ℝ)
  (h1 : map_distance = 120)
  (h2 : scale_factor = 20) :
  actual_distance = map_distance * scale_factor :=
by
  have h_actual_distance : actual_distance = 120 * 20 := by sorry -- Skip the proof
  exact h_actual_distance

end actual_distance_between_cities_l43_43866


namespace find_x_l43_43567

theorem find_x (x : ℝ) : let P := (2, 0)
                          let Q := (11,-3)
                          let R := (x, 3)
                          let anglePQR_is_right : true := (∠PQR = 90)
                          in x = 13 := sorry

end find_x_l43_43567


namespace min_product_sum_l43_43842

theorem min_product_sum (a : Fin 7 → ℕ) (b : Fin 7 → ℕ) 
  (h2 : ∀ i, 2 ≤ a i) 
  (h3 : ∀ i, a i ≤ 166) 
  (h4 : ∀ i, a i ^ b i % 167 = a (i + 1) % 7 + 1 ^ 2 % 167) : 
  b 0 * b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * (b 0 + b 1 + b 2 + b 3 + b 4 + b 5 + b 6) = 675 := sorry

end min_product_sum_l43_43842


namespace find_f_expression_l43_43395

theorem find_f_expression (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = x^2 + 2 * x + 1 :=
by
  sorry

end find_f_expression_l43_43395


namespace fraction_value_l43_43236

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end fraction_value_l43_43236


namespace acute_triangle_problem_l43_43813

theorem acute_triangle_problem
  (ABC : Triangle)
  (acute : is_acute ABC)
  (hAB : ABC.AB = 15)
  (hBC : ABC.BC = 8)
  (D : Point)
  (hD : D ∈ line_segment ABC.A ABC.B)
  (hBD : distance D ABC.B = 8)
  (hE : ∀ E, E ∈ line_segment ABC.A ABC.C →
          (∠ D E B = ∠ B E C)) :
  ⌊(AE)^2⌋ = AE^2 := sorry

end acute_triangle_problem_l43_43813


namespace greatest_integer_l43_43506

noncomputable def y : ℝ := (∑ n in Finset.range 50, Real.cos (2 * (n + 1) * Real.pi / 180)) / 
                          (∑ n in Finset.range 50, Real.sin (2 * (n + 1) * Real.pi / 180))

theorem greatest_integer (hy : y = (∑ n in Finset.range 50, Real.cos (2 * (n + 1) * Real.pi / 180)) / 
                                    (∑ n in Finset.range 50, Real.sin (2 * (n + 1) * Real.pi / 180))) : 
    floor (100 * y) = 74 :=
by
  sorry

end greatest_integer_l43_43506


namespace gen_term_of_a_n_sum_of_first_n_b_n_l43_43745

-- We define the sequence $\{a_n\}$ and $\{b_n\}$ given the conditions and state the problem
def a_n (n : ℕ) := 3 * n - 1
def b_n : ℕ → ℚ 
| 0       := 1 -- 0-indexed version for convenience
| 1       := 1/3
| (n + 1) := (n * b_n n - (3 * n - 1) * b_n (n + 1)) / (3 * (n - 1))

-- Assume the relationship holds for $a_n$ and $b_n$
axiom a_n_b_n_condition : ∀ n : ℕ, a_n n * b_n (n + 1) + b_n (n + 1) = n * b_n n

-- Theorem for the general term of $\{a_n\}$
theorem gen_term_of_a_n : ∀ n : ℕ, a_n n = 3 * n - 1 := sorry

-- Theorem for the sum of the first $n$ terms of $\{b_n\}$
theorem sum_of_first_n_b_n (n : ℕ) : (Σ i in finset.range n, b_n i) = (3/2) - (1/2 * 3 ^ (n - 1)) := sorry

end gen_term_of_a_n_sum_of_first_n_b_n_l43_43745


namespace find_x_l43_43468

theorem find_x (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ π)
  (h_ortho : (2 * real.cos x + 1) * real.cos x + (-2 * real.cos (2 * x) - 2) * 1 = 0) :
  x = π / 2 ∨ x = π / 3 :=
sorry

end find_x_l43_43468


namespace gcd_765432_654321_l43_43211

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43211


namespace max_distinct_numbers_is_4_l43_43515

noncomputable def max_distinct_numbers (board : List ℕ) : ℕ :=
  if h₀ : board.length = 10 ∧ ∀ n ∈ board, n^2 ∣ (board.sum - n) then
    List.length (List.nub board) else
    0

theorem max_distinct_numbers_is_4 (board : List ℕ) :
  board.length = 10 ∧ (∀ n ∈ board, n^2 ∣ board.sum - n) →
  List.length (List.nub board) ≤ 4 :=
sorry

end max_distinct_numbers_is_4_l43_43515


namespace probability_all_same_color_l43_43263

def total_marbles := 15
def red_marbles := 4
def white_marbles := 5
def blue_marbles := 6

def prob_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
def prob_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
def prob_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))

def prob_all_same_color := prob_all_red + prob_all_white + prob_all_blue

theorem probability_all_same_color :
  prob_all_same_color = (34/455) :=
by sorry

end probability_all_same_color_l43_43263


namespace task_completed_earlier_by_108_minutes_l43_43963

-- Define the initial conditions
def initial_time_hours : ℝ := 9
def parts_total : ℕ := 72
def initial_rate : ℝ := parts_total / initial_time_hours
def time_saved_per_swap : ℝ := 1
def parts_per_hour_swap_effect : ℝ := 1

-- Define the swapping effects
def swapped_time_hours_A_B := initial_time_hours - time_saved_per_swap
def swapped_rate_A_B := parts_total / swapped_time_hours_A_B

def swapped_time_hours_C_D := initial_time_hours - time_saved_per_swap
def swapped_rate_C_D := parts_total / swapped_time_hours_C_D

def combined_rate_improvement := initial_rate + parts_per_hour_swap_effect + parts_per_hour_swap_effect
def new_time_per_part_min := 60 / combined_rate_improvement
def initial_time_per_part_min := 60 / initial_rate

-- Calculate the total time saved when both swaps occur simultaneously
def total_time_saved_min := parts_total * (initial_time_per_part_min - new_time_per_part_min)

-- The theorem to prove
theorem task_completed_earlier_by_108_minutes :
  total_time_saved_min = 108 :=
sorry

end task_completed_earlier_by_108_minutes_l43_43963


namespace point_C_coordinates_l43_43082

theorem point_C_coordinates :
  ∃ C : ℝ × ℝ,
    let A := (-3, -2) in
    let B := (5, 10) in
    ((dist C A)^2 = 4 * (dist C B)^2) ∧
    C = (11/3, 8) := 
by {
  sorry
}

end point_C_coordinates_l43_43082


namespace max_area_ABP_BCP_l43_43634

theorem max_area_ABP_BCP (s : ℝ) (AP BP : ℝ) (hAP : AP = 15) (hBP : BP = 8) :
  ∃ P, AP = 15 ∧ BP = 8 ∧ 
  (∃ (ABC : triangle), ABC.equilateral ∧ 
  max_area (triangle_area ABC P) = 23 + 23 / 4 * sqrt 177) := sorry

end max_area_ABP_BCP_l43_43634


namespace problem_statement_l43_43941

theorem problem_statement :
  let pct := 208 / 100
  let initial_value := 1265
  let step1 := pct * initial_value
  let step2 := step1 ^ 2
  let answer := step2 / 12
  answer = 576857.87 := 
by 
  sorry

end problem_statement_l43_43941


namespace gcd_765432_654321_l43_43174

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43174


namespace keith_cards_initial_count_l43_43039

theorem keith_cards_initial_count :
  ∃ (x : ℕ), let final_count := 46 in
  let cards_after_dog := 2 * final_count in
  let total_after_buying := cards_after_dog - 8 in
  (x = total_after_buying) ∧ (total_after_buying = 84) :=
begin
  sorry
end

end keith_cards_initial_count_l43_43039


namespace hyperbola_tangent_to_circle_l43_43354

theorem hyperbola_tangent_to_circle :
  ∃ (a b c : ℝ), (a = 8 ∧ b = -1 ∧ c = 1) ∧ ∀ x y: ℝ,
  ((x - 3)^2 + y^2 = 1 → (x^2 - 8 * y^2 = 1) ∧ (x + 2 * sqrt 2 * y) * (x - 2 * sqrt 2 * y) = 0 →
  (1^2 - 8 * (1/2)^2 = -1)) :=
by
  sorry

end hyperbola_tangent_to_circle_l43_43354


namespace find_a_l43_43410

variable (f : ℝ → ℝ) (a : ℝ)

def f_def (x : ℝ) : Prop := f x = real.sqrt (x + 9)
def f0_def : Prop := f (0) = real.sqrt (9)
def f_comp_def : Prop := f (f (0)) = 4 * a

theorem find_a : f_def f ∧ f0_def f ∧ f_comp_def f a → a = real.sqrt(3) / 2 := by
  sorry

end find_a_l43_43410


namespace max_area_triangle_l43_43116

def line (k : ℝ) : set (ℝ × ℝ) := { p | k * p.1 - p.2 + 2 = 0 }
def circle : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 - 4 * p.1 - 12 = 0 }
def intersection_points (k : ℝ) := (line k) ∩ circle

theorem max_area_triangle (k : ℝ) :
  let QRC := intersection_points k
  -- this is a placeholder definition, you need to define area correctly
  -- suppose area is a function that computes the area of triangle given three points
  -- assuming here Q and R are points from QRC and C is the center of the circle.
  area := sorry in
  -- max_area is the maximum area of the triangle formed by Q, R, and center C
  ∃ max_area : ℝ, (forall Q R C, Q ∈ QRC → R ∈ QRC → C = (2,0) → area Q R C ≤ max_area) ∧
  max_area = 8 := sorry

end max_area_triangle_l43_43116


namespace statement_A_is_correct_statement_B_is_correct_statement_D_is_correct_l43_43417

-- Define the function f(x)
def f (x : ℝ) : ℝ := exp x - (1 / 2) * x^2

-- Statement A: The tangent line to f(x) at x = 0 is x - y + 1 = 0
theorem statement_A_is_correct : (f 0 = 1) ∧ (f' 0 = 1) :=
by
  sorry

-- Statement B: 3/2 < f(ln 2) < 2
theorem statement_B_is_correct : (3 / 2 < f (log 2) ∧ f (log 2) < 2) :=
by
  sorry

-- Statement D: f(x) has a unique zero
theorem statement_D_is_correct : ∃! x : ℝ, f x = 0 :=
by
  sorry

end statement_A_is_correct_statement_B_is_correct_statement_D_is_correct_l43_43417


namespace factorization_problem1_factorization_problem2_l43_43704

-- Mathematical statements
theorem factorization_problem1 (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 := by
  sorry

theorem factorization_problem2 (a : ℝ) : 18 * a^2 - 50 = 2 * (3 * a + 5) * (3 * a - 5) := by
  sorry

end factorization_problem1_factorization_problem2_l43_43704


namespace quadratic_function_properties_l43_43871

theorem quadratic_function_properties :
  ∀ (a b c m : ℝ),
  a ≠ 0 →
  (∀ x, x ∈ {-2, -1, 0, 1, 2} → 
       if x = -2 then m
       else if x = -1 then 1
       else if x = 0 then -1
       else if x = 1 then 1
       else if x = 2 then 7
       else 0) = (λ x, a * x^2 + b * x + c) →
  (∀ x, if x = 0 then true else (a * -x^2 + b * -x + c) = (a * x^2 + b * x + c)) →
  (
    -- Symmetry Axis and Vertex Coordinates
    (∃ x, vertex (λ x, a * x^2 + b * x + c) = (0, -1)) ∧
    -- Value of m
    m = 7 ∧
    -- Opening Direction
    a > 0
  ) :=
by
  intros a b c m h_cond1 h_table h_symmetry
  -- Symmetry Axis and Vertex Coordinates
  split
  sorry -- proof will be inserted here
  -- Value of m
  split
  exact rfl
  -- Opening Direction
  exact h_cond1

end quadratic_function_properties_l43_43871


namespace PE_bisects_CD_given_conditions_l43_43057

variables {A B C D E P : Type*}

noncomputable def cyclic_quadrilateral (A B C D : Type*) : Prop := sorry

noncomputable def AD_squared_plus_BC_squared_eq_AB_squared (A B C D : Type*) : Prop := sorry

noncomputable def angles_equality_condition (A B C D P : Type*) : Prop := sorry

noncomputable def line_PE_bisects_CD (P E C D : Type*) : Prop := sorry

theorem PE_bisects_CD_given_conditions
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : AD_squared_plus_BC_squared_eq_AB_squared A B C D)
  (h3 : angles_equality_condition A B C D P) :
  line_PE_bisects_CD P E C D :=
sorry

end PE_bisects_CD_given_conditions_l43_43057


namespace minimum_bags_needed_l43_43600

theorem minimum_bags_needed (n : ℕ) (h : n = 127) : 
  ∃ k, (k = 7) ∧ (∀ m ≤ n, ∃ S : set ℕ, 
    (S ⊆ {x | (∃ y : ℤ, 0 ≤ y ∧ y < n ∧ x = 2 ^ y)} ∧ m = S.sum)) :=
by 
  sorry

end minimum_bags_needed_l43_43600


namespace seven_digit_number_exists_no_eight_digit_number_exists_l43_43245

-- Define the properties for part (a)
def is_seven_digit_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000

def all_digits_different (n : ℕ) : Prop :=
  let digits := (n.toString.toList.map (λ ch, nat.ofChar ch)).filter (λ x, x ≠ 0)
  digits.nodup

def divisible_by_all_digits (n : ℕ) : Prop :=
  let digits := (n.toString.toList.map (λ ch, nat.ofChar ch)).filter (λ x, x ≠ 0)
  ∀ d ∈ digits, d ∣ n

-- Statement for part (a)
theorem seven_digit_number_exists : ∃ (n : ℕ), is_seven_digit_number n ∧ all_digits_different n ∧ divisible_by_all_digits n :=
  ∃ n, n = 7639128 ∧ is_seven_digit_number n ∧ all_digits_different n ∧ divisible_by_all_digits n

-- Define the properties for part (b)
def is_eight_digit_number (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000

-- Statement for part (b)
theorem no_eight_digit_number_exists : ¬∃ (n : ℕ), is_eight_digit_number n ∧ all_digits_different n ∧ divisible_by_all_digits n :=
  sorry

end seven_digit_number_exists_no_eight_digit_number_exists_l43_43245


namespace identify_1000g_weight_l43_43141

-- Define the problem and conditions
def weight (i : ℕ) : ℕ
| 0 := 1000
| 1 := 1001
| 2 := 1002
| 3 := 1004
| 4 := 1007
| _ := 0 -- extra cases should not matter in this context

-- Assume the existence of the weighings and their process
theorem identify_1000g_weight :
  ∃ (weigh : ℕ → ℕ → ℕ), -- weigh function that combines two weights
  ∀ (combine_weights : (list ℕ) → list ℕ),
    combine_weights = [weight 0, weight 1, weight 2, weight 3, weight 4] →
    (∃ (A B C D E : ℕ),
      (combine_weights = [A, B, C, D, E] ∧
       A ∈ [1000, 1001, 1002, 1004, 1007] ∧
       B ∈ [1000, 1001, 1002, 1004, 1007] ∧
       C ∈ [1000, 1001, 1002, 1004, 1007] ∧
       D ∈ [1000, 1001, 1002, 1004, 1007] ∧
       E ∈ [1000, 1001, 1002, 1004, 1007] ∧
       A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
       B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E ∧
       (∃ (i j : ℕ), (weigh i j = A + B ∨ weigh i j = C + D ∨ weigh i j = E + weight i ∨ weigh i j = weight i + weight j) ∧
        ((A = 1000 ∨ B = 1000 ∨ C = 1000 ∨ D = 1000 ∨ E = 1000) = true)))
     ) :=
sorry

end identify_1000g_weight_l43_43141


namespace total_value_of_coins_l43_43271

theorem total_value_of_coins (n_dimes n_nickels : ℕ) (value_nickel value_dime : ℕ) (H1 : n_dimes + n_nickels = 70) (H2 : n_nickels = 29) (H3 : value_nickel = 5) (H4 : value_dime = 10) : (n_nickels * value_nickel + n_dimes * value_dime) / 100 = 5.55 :=
by
  sorry

end total_value_of_coins_l43_43271


namespace area_of_enclosed_region_is_1257_l43_43094

/-- Represents a square with given side length and coordinate setup --/
structure Square (side: ℝ) :=
  (A B C D: ℝ × ℝ)
  (A_eq: A = (0, 0))
  (B_eq: B = (side, 0))
  (C_eq: C = (side, side))
  (D_eq: D = (0, side))

/-- Set T is the set of all line segments of length 4 with endpoints on adjacent sides --/
def T (sq: Square 4) : set (ℝ × ℝ) :=
  {p | ∃ x y: ℝ, (x, 0) = p ∨ (0, y) = p ∨ (x, 4) = p ∨ (4, y) = p ∧ x^2 + y^2 = 16}

/-- The midpoints of line segments in set T form an ellipse enclosing a specified region 
  whose area to the nearest hundredth is m --/
def enclosed_midpoints_area_100m : ℝ := 100 * Real.pi * 2 * 2

/-- Main proposition: proving 100m = 1257 --/
theorem area_of_enclosed_region_is_1257 (sq: Square 4) (set_T : T sq) : enclosed_midpoints_area_100m = 1257 :=
by
  sorry

end area_of_enclosed_region_is_1257_l43_43094


namespace triangle_circumradius_l43_43274

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) (h4 : a^2 + b^2 = c^2) :
  ∃ R, R = 17 / 2 :=
by
  use 17 / 2
  sorry

end triangle_circumradius_l43_43274


namespace find_m_l43_43496

def divisors (n : ℕ) : set ℕ := { d | d > 0 ∧ d ∣ n }

def S := divisors (15 ^ 7)

def good_probability (a1 a2 a3 : ℕ) : Prop :=
  a1 ∈ S ∧ a2 ∈ S ∧ a3 ∈ S ∧ (a1 ∣ a2) ∧ (a2 ∣ a3)

theorem find_m (m n : ℕ) (relprime : Nat.coprime m n) (prob_eq : (∃ h : good_probability, true) / (S.card * S.card * S.card) = (m / n)) : m = 225 :=
sorry

end find_m_l43_43496


namespace exists_circle_with_half_points_inside_l43_43076

theorem exists_circle_with_half_points_inside (n : ℕ) (points : Finset (ℝ × ℝ)) 
  (h_card : points.card = 2 * n + 3)
  (h_no_three_collinear : ∀ (A B C : (ℝ × ℝ)), A ≠ B → B ≠ C → A ≠ C → 
    ¬Collinear ℝ ({A, B, C} : Set (ℝ × ℝ)))
  (h_no_four_concyclic : ∀ (A B C D : (ℝ × ℝ)), A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D → 
    ¬Concyclic ℝ ({A, B, C, D} : Set (ℝ × ℝ))) : 
  ∃ (A B C : (ℝ × ℝ)) (disk : Set (ℝ × ℝ)), 
    A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
    Set.Bounded disk ∧ 
    (∀ x, x ∈ disk ↔ (dist x A + dist x B + dist x C <= fixed_radius)) ∧
    (points.filter (λ p, p ∈ disk)).card = n :=
sorry

end exists_circle_with_half_points_inside_l43_43076


namespace false_proposition_D_l43_43974

/-
Conditions:
- A: The perpendicular segment is the shortest.
- B: Corresponding angles are equal.
- C: Vertical angles are equal.
- D: Adjacent supplementary angles are always complementary.
-/

theorem false_proposition_D (A B C D : Prop) :
  (A = "The perpendicular segment is the shortest.") ∧
  (B = "Corresponding angles are equal.") ∧
  (C = "Vertical angles are equal.") ∧
  (D = "Adjacent supplementary angles are always complementary.") →
  ¬D :=
by
  intro h
  sorry

end false_proposition_D_l43_43974


namespace max_distance_P_to_C2_area_of_triangle_ABC1_l43_43796

noncomputable def C1 : ℝ × ℝ → Prop :=
λ p, ∃ α : ℝ, p.1 = -2 + Real.cos α ∧ p.2 = -1 + Real.sin α

def C2 (p : ℝ × ℝ) : Prop :=
p.1 = 3

noncomputable def C3 (p : ℝ × ℝ) : Prop :=
p.2 = p.1

theorem max_distance_P_to_C2 : 
  ∀ P : ℝ × ℝ, C1 P → ∃ d, d = 6 :=
begin
  sorry
end

theorem area_of_triangle_ABC1 :
  ∀ A B : ℝ × ℝ, 
  C1 A ∧ C1 B ∧ C3 A ∧ C3 B →
  ∃ S, S = 1 / 2 :=
begin
  sorry
end

end max_distance_P_to_C2_area_of_triangle_ABC1_l43_43796


namespace area_of_triangle_CFG_l43_43847

-- Define the rectangle and its properties
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ

def CDEF : Rectangle := { length := 2 * (real.sqrt 24), width := real.sqrt 24, area := 48 }

-- Define a point as a structure
structure Point where
  x : ℝ
  y : ℝ

-- Define function that gives midpoint of a segment
def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

-- Define points C, D, G where G is the midpoint of CD
def C : Point := { x := 0, y := 0 }
def D : Point := { x := 0, y := CDEF.length }
def G : Point := midpoint C D

-- Define point F, for right triangle CFG
def F : Point := { x := CDEF.width, y := 0 }

-- Define function to compute distance between two points
def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2)

-- Compute the area of triangle CFG
def triangle_area (A B C : Point) : ℝ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem area_of_triangle_CFG : triangle_area C F G = 12 := by 
  sorry

end area_of_triangle_CFG_l43_43847


namespace gcd_765432_654321_eq_3_l43_43216

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43216


namespace number_of_dogs_on_boat_l43_43618

theorem number_of_dogs_on_boat 
  (initial_sheep : ℕ) (initial_cows : ℕ) (initial_dogs : ℕ)
  (drowned_sheep : ℕ) (drowned_cows : ℕ)
  (made_it_to_shore : ℕ)
  (H1 : initial_sheep = 20)
  (H2 : initial_cows = 10)
  (H3 : drowned_sheep = 3)
  (H4 : drowned_cows = 2 * drowned_sheep)
  (H5 : made_it_to_shore = 35)
  : initial_dogs = 14 := 
sorry

end number_of_dogs_on_boat_l43_43618


namespace parabola_distance_to_focus_l43_43552

theorem parabola_distance_to_focus :
  ∀ y : ℝ, y^2 = 4 * 2 → abs (2 - (-1)) = 3 :=
by
  intro y h
  have hx : abs (2 - (-1)) = abs (3) := by rfl
  have hy2 : abs (2 - (-1)) = 3 := by simp
  rw [hx, hy2]
  sorry

end parabola_distance_to_focus_l43_43552


namespace find_other_number_l43_43253

theorem find_other_number (A B : ℕ) (hcf : ℕ) (lcm : ℕ) 
  (H1 : hcf = 12) 
  (H2 : lcm = 312) 
  (H3 : A = 24) 
  (H4 : hcf * lcm = A * B) : 
  B = 156 :=
by sorry

end find_other_number_l43_43253


namespace max_area_triangle_QRC_l43_43114

noncomputable def maxAreaOfTriangle
  (k : ℝ)
  (Q R : ℝ × ℝ)
  (l : ∀ x y : ℝ, k * x - y + 2 = 0)
  (C : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 12 = 0)
  : ℝ :=
  -- Maximum area is 8
  8

theorem max_area_triangle_QRC (k : ℝ) :
  ∃ Q R : ℝ × ℝ,
    let l x y := k * x - y + 2 = 0
    let C := x^2 + y^2 - 4 * x - 12 = 0
    maxAreaOfTriangle k Q R l C = 8 :=
sorry

end max_area_triangle_QRC_l43_43114


namespace xyz_value_l43_43532

theorem xyz_value (x y z : ℝ)
  (h1 : 2^x = 16^(y + 3))
  (h2 : 27^y = 3^(z - 2))
  (h3 : 256^z = 4^(x + 4)) :
  x * y * z = 24.5 := 
sorry

end xyz_value_l43_43532


namespace arithmetic_sequence_sum_l43_43069

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) 
  (h2 : S m = 0) 
  (h3 : S (m + 1) = 3) : 
  m = 5 :=
by sorry

end arithmetic_sequence_sum_l43_43069


namespace tan_sum_eq_one_l43_43767

theorem tan_sum_eq_one (a b : ℝ) (h1 : Real.tan a = 1 / 2) (h2 : Real.tan b = 1 / 3) :
    Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_eq_one_l43_43767


namespace distance_AB_coordinates_B_triangle_ABC_is_right_l43_43531

-- Definition for distance between two points in a Cartesian coordinate plane
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

-- Question 1: Prove the distance between A(0,5) and B(-3,6) is sqrt(10)
theorem distance_AB : distance 0 5 (-3) 6 = Real.sqrt 10 := by sorry

-- Question 2: Given A(-5, -1/2), B(-5, y), AB = 10, the coordinates of B are (-5, 9.5) or (-5, -9.5)
theorem coordinates_B (y : ℝ) (h : distance (-5) (-1/2) (-5) y = 10) : 
  y = 9.5 ∨ y = -9.5 := by sorry

-- Question 3: Given points A(0,6), B(4,0), C(-9,0), prove triangle ABC is right-angled
def right_angle_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  distance x1 y1 x2 y2 ^ 2 + distance x1 y1 x3 y3 ^ 2 = distance x2 y2 x3 y3 ^ 2

theorem triangle_ABC_is_right : right_angle_triangle 0 6 4 0 (-9) 0 := by sorry

end distance_AB_coordinates_B_triangle_ABC_is_right_l43_43531


namespace company_supervisors_l43_43276

theorem company_supervisors (workers : ℕ) (team_leads_per_worker : ℕ) (supervisors_per_team_lead : ℕ) :
  workers = 390 → team_leads_per_worker = 10 → supervisors_per_team_lead = 3 → 
  workers / team_leads_per_worker / supervisors_per_team_lead = 13 :=
by
  intros h_workers h_team_leads_per_worker h_supervisors_per_team_lead
  rw [h_workers, h_team_leads_per_worker, h_supervisors_per_team_lead]
  norm_num
  sorry

end company_supervisors_l43_43276


namespace irreducible_fraction_iff_not_congruent_mod_5_l43_43712

theorem irreducible_fraction_iff_not_congruent_mod_5 (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := 
by 
  sorry

end irreducible_fraction_iff_not_congruent_mod_5_l43_43712


namespace odd_function_has_a_equal_2_l43_43006

def f (a : ℝ) (x : ℝ) : ℝ := (a * 3^x + 4 - a) / (4 * (3^x - 1))

theorem odd_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) + f a x = 0) ↔ a = 2 :=
by
  sorry

end odd_function_has_a_equal_2_l43_43006


namespace exponent_subtraction_l43_43769

variable {a : ℝ} {m n : ℕ}

theorem exponent_subtraction (hm : a ^ m = 12) (hn : a ^ n = 3) : a ^ (m - n) = 4 :=
by
  sorry

end exponent_subtraction_l43_43769


namespace negation_of_at_most_four_l43_43118

theorem negation_of_at_most_four (n : ℕ) : ¬(n ≤ 4) → n ≥ 5 := 
by
  sorry

end negation_of_at_most_four_l43_43118


namespace area_of_shaded_region_l43_43556

-- Definitions of conditions
def center (O : Type) := O
def radius_large_circle (R : ℝ) := R
def radius_small_circle (r : ℝ) := r
def length_chord_CD (CD : ℝ) := CD = 60
def chord_tangent_to_smaller_circle (r : ℝ) (R : ℝ) := r^2 = R^2 - 900

-- Theorem for the area of the shaded region
theorem area_of_shaded_region 
(O : Type) 
(R r : ℝ) 
(CD : ℝ)
(h1 : length_chord_CD CD)
(h2 : chord_tangent_to_smaller_circle r R) : 
  π * (R^2 - r^2) = 900 * π := by
  sorry

end area_of_shaded_region_l43_43556


namespace solve_for_x_l43_43857

theorem solve_for_x : ∃ x : ℝ, 3^(3 * x + 2) = 1 / 9 ∧ x = -4 / 3 := by
  exists -4 / 3
  split
  · norm_num
  · rfl

end solve_for_x_l43_43857


namespace angle_between_vectors_is_2pi_over_3_l43_43431

open EuclideanGeometry

variables (a b : V) (h₁ : ‖a‖ = 2) (h₂ : ‖b‖ = 2) (h₃ : b ⬝ (2 • a + b) = 0)

theorem angle_between_vectors_is_2pi_over_3 : real.angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_is_2pi_over_3_l43_43431


namespace middle_three_digit_multiple_is_560_l43_43646

def divisible_by (n m : Nat) : Prop := m % n = 0

def is_middle_number (x : List Nat) (y : Nat) : Prop := 
  y = x[(x.length) / 2]

theorem middle_three_digit_multiple_is_560 :
  ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧ divisible_by 4 n ∧ divisible_by 5 n ∧ divisible_by 7 n ∧ is_middle_number (List.filter (λ n, 100 ≤ n ∧ n < 1000 ∧ divisible_by 4 n ∧ divisible_by 5 n ∧ divisible_by 7 n) (List.range 1000)) 560 :=
by
  sorry

end middle_three_digit_multiple_is_560_l43_43646


namespace inequality_solution_l43_43537

open Real

theorem inequality_solution (a x : ℝ) :
  (a = 0 ∧ x > 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a = 1 ∧ ∀ x, ¬ (a * x^2 - (2 * a + 2) * x + 4 > 0)) ∨
  (a < 0 ∧ (x < 2/a ∨ x > 2) ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (0 < a ∧ a < 1 ∧ 2 < x ∧ x < 2/a ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a > 1 ∧ 2/a < x ∧ x < 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) := 
sorry

end inequality_solution_l43_43537


namespace solve_real_number_pairs_l43_43346

theorem solve_real_number_pairs (x y : ℝ) :
  (x^2 + y^2 - 48 * x - 29 * y + 714 = 0 ∧ 2 * x * y - 29 * x - 48 * y + 756 = 0) ↔
  (x = 31.5 ∧ y = 10.5) ∨ (x = 20 ∧ y = 22) ∨ (x = 28 ∧ y = 7) ∨ (x = 16.5 ∧ y = 18.5) :=
by
  sorry

end solve_real_number_pairs_l43_43346


namespace gcd_of_765432_and_654321_l43_43200

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43200


namespace main_theorem_l43_43791

-- Definitions of the triangle and configurations
variable {α : Type*} [EuclideanSpace α] -- considering geometry in a Euclidean space

noncomputable def condition_A := ∀ {A B C D : α} (s b : ℝ), 
 (triangle.is_isosceles A B C b) ∧ 
 (excircle.is_tangent_to_side A B C D) ∧
 (incenter_circle_of_triangle_is_internally_tangent A B C) ∧
 (C = (0 : α)) ∧ 
 (B = (s : α))

-- Defining the minimum perimeter function for the isosceles triangle ABC
noncomputable def min_perimeter_iso_triangle := 
 ∀ {A B C : α} (s b : ℝ), condition_A → 
 (triangle.perimeter A B C s b) = 34

-- The main proof statement, skipping the proof
theorem main_theorem : min_perimeter_iso_triangle := 
by sorry

end main_theorem_l43_43791


namespace angle_between_a_b_l43_43430

noncomputable def a : Type := Vector ℝ
noncomputable def b : Type := Vector ℝ
noncomputable def dot_product (u v: Vector ℝ) := vector_dot u v

variable (mam: Set ℝ) 
variables (a b : Vector ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : dot_product (a + b) (a - 2*b) = -7)

theorem angle_between_a_b (ha : |a| = 1) (hb : |b| = 2) (hc : dot_product (a + b) (a - 2*b) = -7) : 
  angle_between a b = π / 2 := 
begin
  sorry
end

end angle_between_a_b_l43_43430


namespace vertex_angle_of_obtuse_isosceles_triangle_l43_43020

noncomputable def isosceles_obtuse_triangle (a b h : ℝ) (φ : ℝ) : Prop :=
  a^2 = 2 * b * h ∧
  b = 2 * a * Real.cos ((180 - φ) / 2) ∧
  h = a * Real.sin ((180 - φ) / 2) ∧
  90 < φ ∧ φ < 180

theorem vertex_angle_of_obtuse_isosceles_triangle (a b h : ℝ) (φ : ℝ) :
  isosceles_obtuse_triangle a b h φ → φ = 150 :=
by
  sorry

end vertex_angle_of_obtuse_isosceles_triangle_l43_43020


namespace exists_perpendicular_base_on_side_l43_43724

-- Define a polygon and the perpendiculars dropped from each vertex.
variables (V : Type) [LinearOrderedField V] -- V represents the type of vertices (assuming a Euclidean plane)
variables (n : ℕ) -- number of vertices in the polygon
variables (polygon : Finₓ n → V × V) -- the polygon represented as a function from finite set to vertices

-- Assume the polygon is non-degenerate and simple
variables [Nonempty (Finₓ n)] [SimplePolygon polygon]

-- Define the perpendicular dropping operation
def perpendicular (A B C : V × V) : Prop :=
  ∃ D : V × V, IsPerpendicular (C - A) (D - A) ∧ Distance B D = Distance B (ProjectPointOnLineSegment (B, A) C)

-- Define the main theorem to be proven
theorem exists_perpendicular_base_on_side :
  ∃ (v : Finₓ n), ∃ (side : Finₓ n × Finₓ n), side.1 ≠ v ∧ side.2 ≠ v ∧ perpendicular (polygon side.1) (polygon side.2) (polygon v) :=
sorry

end exists_perpendicular_base_on_side_l43_43724


namespace coefficient_sum_l43_43312

theorem coefficient_sum :
  (∑ k in finset.range (2009 - 5 + 1), nat.choose (k + 5) 5) = nat.choose 2010 6 := 
sorry

end coefficient_sum_l43_43312


namespace sum_of_triangle_areas_in_cube_l43_43896

theorem sum_of_triangle_areas_in_cube :
  let m : ℤ := 48,
      n : ℤ := 4608,
      p : ℤ := 3072
  in m + n + p = 7728 :=
by
  sorry

end sum_of_triangle_areas_in_cube_l43_43896


namespace eval_polys_sum_at_2_l43_43490

def poly1 (x : ℤ) : ℤ := x^2 + x + 1
def poly2 (x : ℤ) : ℤ := x^4 - x^3 - 1

theorem eval_polys_sum_at_2 : poly1 2 + poly2 2 = 14 :=
by {
  -- Poly1 evaluation
  have h1 : poly1 2 = 2^2 + 2 + 1,
  -- Simplification steps, if needed
  calc
    2^2 + 2 + 1 = 4 + 2 + 1 := by norm_num
           ... = 7         := by norm_num,

  -- Poly2 evaluation
  have h2 : poly2 2 = 2^4 - 2^3 - 1,
  -- Simplification steps, if needed
  calc
    2^4 - 2^3 - 1 = 16 - 8 - 1 := by norm_num
           ... = 7            := by norm_num,

  -- sum evaluations
  show poly1 2 + poly2 2 = 14,
  calc
    7 + 7 = 14 := by norm_num
}

end eval_polys_sum_at_2_l43_43490


namespace max_value_of_f_l43_43322

noncomputable def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

theorem max_value_of_f : ∃ x : ℝ, f x = 12.5 :=
by
  sorry

end max_value_of_f_l43_43322


namespace profit_percentage_correct_l43_43995

variables (P : ℝ) (profit_percentage_before_decrease : ℝ)

-- Conditions
def manufacturing_cost_now : ℝ := 50
def manufacturing_cost_before : ℝ := 65
def profit_now : ℝ := 0.5 * P

-- We know that the profit is the selling price minus the manufacturing cost
def profit_equation : P - manufacturing_cost_now = profit_now := by sorry

-- Solving for P
def selling_price := P := by sorry

-- Profit before the decrease
def profit_before_decrease := P - manufacturing_cost_before

-- Profit percentage before the decrease
def profit_percentage_equation : profit_percentage_before_decrease = (profit_before_decrease / P) * 100 := by sorry

-- The theorem to prove
theorem profit_percentage_correct : profit_percentage_before_decrease = 35 :=
by sorry

end profit_percentage_correct_l43_43995


namespace hockey_league_games_l43_43137

theorem hockey_league_games (num_teams games_per_pairing : ℕ) (h1 : num_teams = 16) (h2 : games_per_pairing = 10) : 
  let total_games := (num_teams * (num_teams - 1) / 2) * games_per_pairing in
  total_games = 1200 := 
by
  have h_num_teams : num_teams = 16 := h1
  have h_games_per_pairing : games_per_pairing = 10 := h2
  let total_games := (num_teams * (num_teams - 1) / 2) * games_per_pairing
  sorry

end hockey_league_games_l43_43137


namespace area_of_shaded_region_l43_43557

-- Definitions of conditions
def center (O : Type) := O
def radius_large_circle (R : ℝ) := R
def radius_small_circle (r : ℝ) := r
def length_chord_CD (CD : ℝ) := CD = 60
def chord_tangent_to_smaller_circle (r : ℝ) (R : ℝ) := r^2 = R^2 - 900

-- Theorem for the area of the shaded region
theorem area_of_shaded_region 
(O : Type) 
(R r : ℝ) 
(CD : ℝ)
(h1 : length_chord_CD CD)
(h2 : chord_tangent_to_smaller_circle r R) : 
  π * (R^2 - r^2) = 900 * π := by
  sorry

end area_of_shaded_region_l43_43557


namespace gcd_765432_654321_eq_3_l43_43218

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43218


namespace minimum_translation_symmetry_l43_43779

def translate_and_symmetric (f : ℝ → ℝ) (m : ℝ) : (ℝ → ℝ) :=
λ x, f (x + m)

theorem minimum_translation_symmetry :
  ∀ m : ℝ, m > 0 → symmetric_about_y (translate_and_symmetric (λ x, Real.sin (3 * x + Real.pi / 6)) m) ↔ m = Real.pi / 9 :=
sorry

end minimum_translation_symmetry_l43_43779


namespace simone_fraction_per_day_l43_43534

theorem simone_fraction_per_day 
  (x : ℚ) -- Define the fraction of an apple Simone ate each day as x.
  (h1 : 16 * x + 15 * (1/3) = 13) -- Condition: Simone and Lauri together ate 13 apples.
  : x = 1/2 := 
 by 
  sorry

end simone_fraction_per_day_l43_43534


namespace equation_of_line_l43_43448

theorem equation_of_line (x y : ℝ) 
  (l1 : 4 * x + y + 6 = 0) 
  (l2 : 3 * x - 5 * y - 6 = 0) 
  (midpoint_origin : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * x₁ + y₁ + 6 = 0) ∧ 
    (3 * x₂ - 5 * y₂ - 6 = 0) ∧ 
    (x₁ + x₂ = 0) ∧ 
    (y₁ + y₂ = 0)) : 
  7 * x + 4 * y = 0 :=
sorry

end equation_of_line_l43_43448


namespace daniel_drives_60_miles_l43_43338

-- Define the main problem terms
def distance (D : ℝ) : Prop :=
  ∀ (x : ℝ),
  x > 0 →
  let T_sunday := D / x in
  let T_monday := (32 / (2 * x)) + ((D - 32) / (x / 2)) in
  T_monday = T_sunday * 1.20 →
  D = 60

-- Theorem statement
theorem daniel_drives_60_miles : ∃ D, distance D :=
  by sorry

end daniel_drives_60_miles_l43_43338


namespace annual_growth_rate_proof_l43_43269

-- Lean 4 statement for the given problem
theorem annual_growth_rate_proof (profit_2021 : ℝ) (profit_2023 : ℝ) (r : ℝ)
  (h1 : profit_2021 = 3000)
  (h2 : profit_2023 = 4320)
  (h3 : profit_2023 = profit_2021 * (1 + r) ^ 2) :
  r = 0.2 :=
by sorry

end annual_growth_rate_proof_l43_43269


namespace cost_per_adult_is_3_l43_43982

-- Define the number of people in the group
def total_people : ℕ := 12

-- Define the number of kids in the group
def kids : ℕ := 7

-- Define the total cost for the group
def total_cost : ℕ := 15

-- Define the number of adults, which is the total number of people minus the number of kids
def adults : ℕ := total_people - kids

-- Define the cost per adult meal, which is the total cost divided by the number of adults
noncomputable def cost_per_adult : ℕ := total_cost / adults

-- The theorem stating the cost per adult meal is $3
theorem cost_per_adult_is_3 : cost_per_adult = 3 :=
by
  -- The proof is skipped
  sorry

end cost_per_adult_is_3_l43_43982


namespace number_decomposition_l43_43262

theorem number_decomposition (n : ℕ) : n = 6058 → (n / 1000 = 6) ∧ ((n % 100) / 10 = 5) ∧ (n % 10 = 8) :=
by
  -- Actual proof will go here
  sorry

end number_decomposition_l43_43262


namespace maximum_items_6_yuan_l43_43142

theorem maximum_items_6_yuan :
  ∃ (x : ℕ), (∀ (x' : ℕ), (∃ (y z : ℕ), 6 * x' + 4 * y + 2 * z = 60 ∧ x' + y + z = 16) →
    x' ≤ 7) → x = 7 :=
by
  sorry

end maximum_items_6_yuan_l43_43142


namespace no_positive_integer_pair_small_exists_positive_integer_pair_medium_no_positive_integer_pair_large_l43_43124

variables (k : ℕ) (A : set ℕ)

-- Assume conditions
axiom pos_k : 0 < k
axiom A_subset : A ⊆ {x : ℕ | 1 ≤ x ∧ x ≤ 3 * k}
axiom A_distinct : ∀ a b c ∈ A, (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 2 * b ≠ a + c

def small (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ k
def medium (x : ℕ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k
def large (x : ℕ) : Prop := 2 * k + 1 ≤ x ∧ x ≤ 3 * k

-- Part (a): Small Numbers
theorem no_positive_integer_pair_small :
  ¬∃ (x d : ℕ), x ≠ x + d ∧ (x % (3 * k) ∈ A) ∧ ((x + d) % (3 * k) ∈ A) ∧ ((x + 2 * d) % (3 * k) ∈ A) ∧ small (x % (3 * k)) ∧ small ((x + d) % (3 * k)) :=
sorry

-- Part (b): Medium Numbers
theorem exists_positive_integer_pair_medium :
  ∃ (x d : ℕ), x ≠ x + d ∧ (x % (3 * k) ∈ A) ∧ ((x + d) % (3 * k) ∈ A) ∧ ((x + 2 * d) % (3 * k) ∈ A) ∧ medium (x % (3 * k)) ∧ medium ((x + d) % (3 * k)) :=
sorry

-- Part (c): Large Numbers
theorem no_positive_integer_pair_large :
  ¬∃ (x d : ℕ), x ≠ x + d ∧ (x % (3 * k) ∈ A) ∧ ((x + d) % (3 * k) ∈ A) ∧ ((x + 2 * d) % (3 * k) ∈ A) ∧ large (x % (3 * k)) ∧ large ((x + d) % (3 * k)) :=
sorry

end no_positive_integer_pair_small_exists_positive_integer_pair_medium_no_positive_integer_pair_large_l43_43124


namespace rational_area_of_shifted_integer_triangle_l43_43651

theorem rational_area_of_shifted_integer_triangle 
  (x1 x2 x3 y1 y2 y3 : ℤ) : 
  ∃ r : ℚ, 
    let x := (x1 : ℚ) + 0.5,
        y := (y1 : ℚ) + 0.5,
        x' := (x2 : ℚ) + 0.5,
        y' := (y2 : ℚ) + 0.5,
        x'' := (x3 : ℚ) + 0.5,
        y'' := (y3 : ℚ) + 0.5,
        area := 0.5 * abs ((x * (y' - y'') + x' * (y'' - y) + x'' * (y - y')))
    in area = r :=
  sorry

end rational_area_of_shifted_integer_triangle_l43_43651


namespace solve_log_equation_l43_43260

theorem solve_log_equation : ∃ x : ℝ, log 2 (9^(x-1) - 5) = log 2 (3^(x-1) - 2) + 2 ∧ x = 2 :=
by
  sorry

end solve_log_equation_l43_43260


namespace average_speed_for_both_trips_l43_43630

noncomputable def average_speed (distance_1 distance_2 time_1 time_2 : ℝ) : ℝ :=
  (distance_1 + distance_2) / (time_1 + time_2)

theorem average_speed_for_both_trips :
  let trip_distance : ℝ := 120
  let plain_speed : ℝ := 30
  let increase_percentage : ℝ := 0.30
  let decrease_percentage : ℝ := 0.15
  let increased_speed : ℝ := plain_speed * (1 + increase_percentage)
  let final_speed : ℝ := increased_speed * (1 - decrease_percentage)
  let time_plain : ℝ := trip_distance / plain_speed
  let time_uphill : ℝ := trip_distance / final_speed
  let total_distance : ℝ := 2 * trip_distance
  let total_time : ℝ := time_plain + time_uphill
  average_speed total_distance total_distance time_plain time_uphill ≈ 31.5 := 
by
  sorry

end average_speed_for_both_trips_l43_43630


namespace total_stones_is_60_l43_43851

-- Definitions of the number of stones in each pile based on conditions
variables {x : ℕ}

def num_stones_third_pile := x
def num_stones_fifth_pile := 6 * num_stones_third_pile
def num_stones_second_pile := 2 * (num_stones_third_pile + num_stones_fifth_pile)
def num_stones_fourth_pile := num_stones_second_pile / 2
def num_stones_first_pile := num_stones_fifth_pile / 3

-- Condition about the first pile being 10 stones fewer than the fourth pile
lemma condition_first_pile (x : ℕ) : num_stones_first_pile = num_stones_fourth_pile - 10 := 
by sorry

-- Statement of the problem in Lean:
theorem total_stones_is_60 : 
  ∀ x : ℕ, 
  num_stones_first_pile = num_stones_fourth_pile - 10 →
  let total_stones := num_stones_third_pile + num_stones_fifth_pile + num_stones_second_pile + num_stones_fourth_pile + num_stones_first_pile
  in 
  total_stones = 60 :=
by sorry

end total_stones_is_60_l43_43851


namespace part_I_part_II_part_III_l43_43024

noncomputable def f (m n : ℕ) : ℝ := sorry 

theorem part_I (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (m % 2 = 0 ∧ n % 2 = 0 → f m n = 0) ∧ (m % 2 = 1 ∧ n % 2 = 1 → f m n = 0.5) :=
begin
  sorry
end

theorem part_II (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  f m n ≤ 0.5 * (max m n) :=
begin
  sorry
end

theorem part_III : ¬ ∃ c : ℝ, ∀ (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0), f m n < c :=
begin
  sorry
end

end part_I_part_II_part_III_l43_43024


namespace weight_of_new_person_l43_43862

-- Definitions for the conditions given.

-- Average weight increase
def avg_weight_increase : ℝ := 2.5

-- Number of persons
def num_persons : ℕ := 8

-- Weight of the person being replaced
def weight_replaced : ℝ := 65

-- Total weight increase
def total_weight_increase : ℝ := num_persons * avg_weight_increase

-- Statement to prove the weight of the new person
theorem weight_of_new_person : 
  ∃ (W_new : ℝ), W_new = weight_replaced + total_weight_increase :=
sorry

end weight_of_new_person_l43_43862


namespace standard_equation_of_ellipse_maximizing_AB_length_l43_43403

-- Conditions
variables {F1 F2 P : ℝ × ℝ}
variables {a b m : ℝ}
variables (x y : ℝ)

-- Definitions based on conditions
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def P : ℝ × ℝ := (1, (Real.sqrt 2) / 2)

-- Proof goals
theorem standard_equation_of_ellipse :
  (F1 = (-1,0)) ∧ (F2 = (1,0)) ∧ (P = (1, (Real.sqrt 2) / 2)) → 
  (∃ a b : ℝ, (a = Real.sqrt 2) ∧ (b = 1) ∧ ∀ (x y : ℝ), 
    (x^2 / 2 + y^2 = 1)) :=
sorry

theorem maximizing_AB_length :
  (F1 = (-1,0)) ∧ (F2 = (1,0)) ∧ (P = (1, (Real.sqrt 2) / 2)) ∧ 
  (∀ (x : ℝ), (F : ℝ → ℝ) (F x = x + m)) →
  (∃ m : ℝ, (m = 0) ∧ (∀ (x : ℝ), F x = x)) :=
sorry

end standard_equation_of_ellipse_maximizing_AB_length_l43_43403


namespace _l43_43527

variable (EFGH : Type) [rhombus : Rhombus EFGH]
variables (a b c d : EFGH)
variables (len_diagonal_EG len_perimeter : ℝ)
variable (d1 d2 : ℝ)

-- Assuming EFGH is a rhombus and specified lengths of sides and diagonal
def conditions
  [H1 : rhombus EFGH]
  (H2 : len_perimeter = 40)
  (H3 : len_diagonal_EG = 16):
  Prop :=
  by
    -- Extract side length from perimeter
    let side_length := len_perimeter / 4;
    
    -- Calculate half of the diagonal given
    let half_diagonal_EG := len_diagonal_EG / 2;
    
    -- Use Pythagorean theorem to find the other half diagonal
    let half_diagonal_FH := 
      sqrt (side_length ^ 2 - half_diagonal_EG ^ 2);
    
    -- Full length of the other diagonal
    let len_diagonal_FH := 2 * half_diagonal_FH;
    let area := len_diagonal_EG * len_diagonal_FH / 2;
    exact area = 96

end _l43_43527


namespace product_mn_l43_43458

-- Define the problem conditions
namespace CircleProblem

variables {R : Real} (O P A B C D : R^2)
variable (r : Real := 7) -- radius of the circle
variable (hAB : AB = 8) -- AB is bisected at P meaning AP = PB = 4
variable (hPB : PB = 4) -- PB = 4
variable (m n : Int)

theorem product_mn (h : ∃ (m n : Int), m / n = (sin (2 * asin (4/7) * sqrt (33)/7))) :
  m * n = 12936 :=
by
  sorry

end CircleProblem

end product_mn_l43_43458


namespace fk_even_iff_l43_43062

def validColoringScheme (n k : ℕ) : Prop :=
  ∀ (coloring : Fin n → Bool),
    (∀ (i : Fin n), ∃ j : Fin k, coloring ((i.val + j) % n) = tt) → 
    -- tt represents red and ff represents blue
    True

theorem fk_even_iff (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  (∃ f : validColoringScheme n k, True) → (f_k n).even ↔ (k.even ∧ (n % (k + 1) = 0)) :=
sorry

end fk_even_iff_l43_43062


namespace num_elements_in_M_l43_43357

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def nat_oplus (m n : ℕ) : ℕ := 
  if is_even m ∧ is_even n ∨ is_odd m ∧ is_odd n then m + n else m * n

def PosNat := { n : ℕ // 0 < n }
def M : Set (PosNat × PosNat) := 
  { p | nat_oplus p.1 p.2 = 12 ∧ 0 < p.1 ∧ 0 < p.2 }

theorem num_elements_in_M : M.toFinset.card = 15 := 
by sorry

end num_elements_in_M_l43_43357


namespace exp_function_properties_l43_43376

theorem exp_function_properties
  (a : ℝ) (m n : ℝ) 
  (h_a_pos : 0 < a) 
  (h_a_ne_one : a ≠ 1) 
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) :
  (∀ (x : ℝ), f x = a^x) → 
  (f (m + n) = f m * f n) ∧ 
  (f (m + n) / 2 ≤ (f m + f n) / 2) := by
  sorry

end exp_function_properties_l43_43376


namespace gcd_of_765432_and_654321_l43_43204

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43204


namespace determine_b_l43_43358

noncomputable def find_b (a b c d : ℂ) : Prop :=
  b = 59

theorem determine_b (a b c d : ℂ) (z w : ℂ)
  (h1 : x^4 + a*x^3 + b*x^2 + c*x + d = 0)
  (h2 : z*w = 15 + ⟨1⟩i)
  (h3 : conjugate z + conjugate w = 2 + 5*⟨1⟩i)
  (h4 : i^2 = -⟨1⟩) : find_b a b c d :=
by
  sorry

end determine_b_l43_43358


namespace common_element_in_F_l43_43492

noncomputable def K : Set (Fin 5) := {0, 1, 2, 3, 4}
noncomputable def F : Set (Set (Fin 5)) := {A ∈ Powerset K |  ∃ B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ ∀ x ∈ A ∪ B ∪ C, x ∈ A ∪ B ∪ C}

axiom H : ∀ A ∈ F, ∀ B ∈ F, ∀ C ∈ F, (A ≠ B ∧ A ≠ C ∧ B ≠ C) → (A ∩ B ∩ C ≠ ∅)

theorem common_element_in_F : ∃ x ∈ K, ∀ A ∈ F, x ∈ A :=
sorry

end common_element_in_F_l43_43492


namespace sum_of_vectors_is_zero_sum_of_squared_distances_is_constant_l43_43058

variables {n : ℕ} (r : ℝ)
noncomputable def vertices (i : fin n) : ℂ :=
  r * complex.exp (2 * real.pi * complex.I * (i : ℤ) / n)

theorem sum_of_vectors_is_zero (r : ℝ) (n : ℕ) :
  ∑ i : fin n, vertices r i = 0 :=
sorry

theorem sum_of_squared_distances_is_constant (r : ℝ) (n : ℕ) (P : ℂ) (hP : complex.abs P = r) :
  ∑ i : fin n, complex.abs (P - vertices r i) ^ 2 = 2 * n * r^2 :=
sorry

end sum_of_vectors_is_zero_sum_of_squared_distances_is_constant_l43_43058


namespace five_times_seven_divided_by_ten_l43_43235

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end five_times_seven_divided_by_ten_l43_43235


namespace range_of_m_l43_43735

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry

end range_of_m_l43_43735


namespace original_function_l43_43003

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l43_43003


namespace annulus_area_l43_43383

-- Definitions based on the given conditions
def radius : ℝ := 3
def segment_length : ℝ := 6

-- The annulus area problem in Lean 4 statement form
theorem annulus_area : 
  let r := radius,
      R := r * Real.sqrt 2 in
  ∀ (r : ℝ) (R : ℝ), 
    r = 3 ∧ R = 3 * Real.sqrt 2 → 
    π * R^2 - π * r^2 = 9 * π :=
by
  intros r R h
  rcases h with ⟨hr, hR⟩
  rw [hr, hR]
  sorry

end annulus_area_l43_43383


namespace johns_cookies_left_l43_43034

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end johns_cookies_left_l43_43034


namespace num_permutations_l43_43815

theorem num_permutations (a : Fin 14 → ℕ) (h1 : ∀ i, 1 ≤ a i ∧ a i ≤ 14) 
(h2 : ∀ i j, i ≠ j → a i ≠ a j)
(h3 : a 0 > a 1 > a 2 > a 3 > a 4 > a 5 > a 6)
(h4 : a 6 < a 7 < a 8 < a 9 < a 10 < a 11 < a 12 < a 13) :
  ∃ (n : ℕ), n = 1716 := 
sorry

end num_permutations_l43_43815


namespace solution_set_of_inequality_l43_43129

theorem solution_set_of_inequality (a : ℝ) (ha : a > 0) :
  { x : ℝ | (a * x^2 - (a + 2) * x + 2) ≥ 0 } =
  if a = 2 then
    set.univ
  else if 0 < a ∧ a < 2 then
    { x : ℝ | x ≤ 1 } ∪ { x : ℝ | x ≥ 2 / a }
  else if a > 2 then
    { x : ℝ | x ≤ 2 / a } ∪ { x : ℝ | x ≥ 1 }
  else
    ∅ :=
begin
  sorry
end

end solution_set_of_inequality_l43_43129


namespace regression_properties_l43_43143

noncomputable section

-- Define data sets
def data1 : List (ℝ × ℝ) := [(11, 1), (11.3, 2), (11.8, 3), (12.5, 4), (13.4, 5)]
def data2 : List (ℝ × ℝ) := [(11, 5), (11.3, 4), (11.8, 3), (12.5, 2), (13.4, 1)]

-- Define linear regression equations
structure Line := 
  (b : ℝ) -- slope
  (a : ℝ) -- intercept

-- Define correlation coefficients
structure Correlation := 
  (r : ℝ)

axiom reg_eqn1 : Line
axiom reg_eqn2 : Line

axiom corr1 : Correlation
axiom corr2 : Correlation

-- The proof problem
theorem regression_properties :
  let l1 := reg_eqn1 in
  let l2 := reg_eqn2 in
  let r1 := corr1.r in
  let r2 := corr2.r in
  
  -- Intersection point
  (12 * l1.b + l1.a = 3) ∧ (12 * l2.b + l2.a = 3) ∧
  
  -- Correlation product
  (r1 * r2 < 0) ∧
  
  -- Slope sum
  (l1.b + l2.b = 0) ∧
  
  -- Intercept sum
  (l1.a + l2.a = 6) :=
sorry

end regression_properties_l43_43143


namespace indicator_is_lambda_system_l43_43051

-- Definitions of conditions
variables {Ω : Type*} {𝔽 : set (set Ω)}

def measurable_func_system (𝒽 : set (Ω → ℝ)) : Prop :=
  (∀ (f g : Ω → ℝ), f ∈ 𝒽 → g ∈ 𝒽 → (λ x, f x + g x) ∈ 𝒽 ∧ ∀ c : ℝ, (λ x, c * f x) ∈ 𝒽) ∧
  (∀ (h : ℕ → Ω → ℝ), (∀ n, h n ∈ 𝒽) → ∀ x, (∀ n, h n x ≤ h (n+1) x) → (λ x, ⨆ n, h n x) ∈ 𝒽) ∧
  ((λ x : Ω, 1) ∈ 𝒽)

def indicator_in_system (𝒾 𝒽 : set (Ω → ℝ)) : set (set Ω) :=
  {A ∈ 𝔽 | (λ x, if x ∈ A then 1 else 0) ∈ 𝒽}

-- Statements
def is_lambda_system (𝒾 : set (set Ω)) : Prop :=
  (Ω ∈ 𝒾) ∧
  (∀ A B ∈ 𝒾, A ⊆ B → (B \ A) ∈ 𝒾) ∧
  (∀ (A : ℕ → set Ω), (∀ n, A n ∈ 𝒾) → (∃ A' ∈ 𝒾, (λ x, ∃ n, x ∈ A n) → x ∈ A') ∧ A' ⊆ (λ x, ∃ n, x ∈ A n))

-- Main theorem statement
theorem indicator_is_lambda_system (𝒽 : set (Ω → ℝ)) (h_prop : measurable_func_system 𝒽) :
  is_lambda_system (indicator_in_system 𝔽 𝒽) :=
  sorry

end indicator_is_lambda_system_l43_43051


namespace average_length_of_rods_l43_43606

theorem average_length_of_rods :
  let rods_20cm := 23
  let rods_21cm := 64
  let rods_22cm := 32
  let total_length := 460 + 1344 + 704
  let total_rods := rods_20cm + rods_21cm + rods_22cm
  let average_length := total_length / total_rods
  average_length ≈ 21.08 := 
by
  sorry

end average_length_of_rods_l43_43606


namespace math_problem_solution_l43_43579

-- Define the purchase price of garbage bins
variables (x y a m n : ℝ)

-- Conditions as given in the problem statement
def conditions : Prop := 
  (2 * x + y = 280) ∧ 
  (3 * x + 2 * y = 460) ∧
  (100 * a + 80 * (100 - a) ≤ 9000) ∧ 
  (a ≥ 0.8 * (100 - a)) ∧ 
  ∀ a : ℕ, (a ∈ [45, 50])

-- Define the proof goals
def proof_goals : Prop :=
  (x = 100) ∧ 
  (y = 80) ∧ 
  (6 = (50 - 45 + 1)) ∧ 
  (m - n = 20)

-- Formal Lean statement
theorem math_problem_solution : conditions → proof_goals :=
by
  sorry

end math_problem_solution_l43_43579


namespace Julia_played_with_kids_on_Monday_l43_43810

theorem Julia_played_with_kids_on_Monday (kids_tuesday : ℕ) (more_kids_monday : ℕ) :
  kids_tuesday = 14 → more_kids_monday = 8 → (kids_tuesday + more_kids_monday = 22) :=
by
  sorry

end Julia_played_with_kids_on_Monday_l43_43810


namespace count_squares_in_specificGrid_l43_43319

-- Define the structure of the grid
structure Grid :=
  (vertical_lines : ℕ)  -- Number of vertical lines
  (horizontal_lines : ℕ)  -- Number of horizontal lines
  (column_widths : List ℕ)  -- Widths of columns
  (row_heights : List ℕ)  -- Heights of rows

-- Define the specific grid described in the problem
def specificGrid : Grid :=
  { vertical_lines := 5,
    horizontal_lines := 6,
    column_widths := [1, 2, 1, 1],
    row_heights := [2, 1, 1, 1] }

-- The main theorem to prove the number of squares in the specific grid
theorem count_squares_in_specificGrid : Σ (n : ℕ), n = 23 :=
  sorry  -- Proof is not provided

end count_squares_in_specificGrid_l43_43319


namespace inequality_condition_l43_43771

theorem inequality_condition (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 :=
sorry

end inequality_condition_l43_43771


namespace total_dishes_l43_43969

variable (D : ℕ) (B : ℕ) (S : ℕ) (L : ℕ)

-- Given conditions
axiom beans_lentils : ∃ x, x = 2
axiom beans_seitan : ∃ y, y = 2
axiom total_lentils : ∃ z, z = 4
axiom lentils_only : ∃ w, w = z - x
axiom half_remaining_beans : ∃ u, u = (D - x - y - w) / 2
axiom triple_beans_seitan : ∃ v, v = 3 * S
axiom zero_seitan_beans : S = 0 ∧ B = 0

-- Proof goal
theorem total_dishes : D = 6 :=
sory

end total_dishes_l43_43969


namespace number_is_12_l43_43254

theorem number_is_12 (x : ℝ) (h : 4 * x - 3 = 9 * (x - 7)) : x = 12 :=
by
  sorry

end number_is_12_l43_43254


namespace correct_random_error_causes_l43_43598

-- Definitions based on conditions
def is_random_error_cause (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

-- Theorem: Valid causes of random errors are options (1), (2), and (3)
theorem correct_random_error_causes :
  (is_random_error_cause 1) ∧ (is_random_error_cause 2) ∧ (is_random_error_cause 3) :=
by
  sorry

end correct_random_error_causes_l43_43598


namespace triangle_inequality_l43_43090

theorem triangle_inequality (a b c : ℝ) (h : a^2 = b^2 + c^2) : 
  (b - c)^2 * (a^2 + 4 * b * c)^2 ≤ 2 * a^6 :=
by
  sorry

end triangle_inequality_l43_43090


namespace sum_fib_series_eq_2_l43_43042

noncomputable def fib (n : ℕ) : ℕ :=
  if n = 1 ∨ n = 2 then 1
  else fib (n - 1) + fib (n - 2)

theorem sum_fib_series_eq_2 : 
  (∑' n : ℕ, (fib (n + 1) : ℝ) / 2^(n + 1)) = 2 :=
  sorry

end sum_fib_series_eq_2_l43_43042


namespace inequality_solution_l43_43091

theorem inequality_solution (a x : ℝ) : 
  (ax^2 + (2 - a) * x - 2 < 0) → 
  ((a = 0) → x < 1) ∧ 
  ((a > 0) → (-2/a < x ∧ x < 1)) ∧ 
  ((a < 0) → 
    ((-2 < a ∧ a < 0) → (x < 1 ∨ x > -2/a)) ∧
    (a = -2 → (x ≠ 1)) ∧
    (a < -2 → (x < -2/a ∨ x > 1)))
:=
sorry

end inequality_solution_l43_43091


namespace merchant_cost_price_l43_43285

theorem merchant_cost_price (x : ℝ) (h₁ : x + (x^2 / 100) = 39) : x = 30 :=
sorry

end merchant_cost_price_l43_43285


namespace range_of_f_ge_1_l43_43068

def f (x : ℝ) : ℝ :=
  if x < 1 then (x + 1) ^ 2
  else 4 - Real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  { x : ℝ | f x ≥ 1 } = { x : ℝ | x ≤ -2 } ∪ { x : ℝ | 0 ≤ x ∧ x ≤ 10 } :=
by
  sorry

end range_of_f_ge_1_l43_43068


namespace alfreds_gain_percent_is_correct_l43_43603

theorem alfreds_gain_percent_is_correct :
  ∀ (purchase_price repair_costs selling_price : ℕ),
  purchase_price = 4700 →
  repair_costs = 1000 →
  selling_price = 5800 →
  (float (selling_price - (purchase_price + repair_costs)) / float (purchase_price + repair_costs)) * 100 = 1.75 :=
by
  intros purchase_price repair_costs selling_price h1 h2 h3
  sorry

end alfreds_gain_percent_is_correct_l43_43603


namespace gcd_proof_l43_43189

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43189


namespace acute_triangle_cos_condition_obtuse_triangle_cos_condition_l43_43844

theorem acute_triangle_cos_condition (A B C : ℝ) (h : A + B + C = π) :
  (cos A)^2 + (cos B)^2 + (cos C)^2 < 1 ↔ A < π / 2 ∧ B < π / 2 ∧ C < π / 2 :=
sorry

theorem obtuse_triangle_cos_condition (A B C : ℝ) (h : A + B + C = π) :
  (cos A)^2 + (cos B)^2 + (cos C)^2 > 1 ↔ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) :=
sorry

end acute_triangle_cos_condition_obtuse_triangle_cos_condition_l43_43844


namespace find_k_l43_43132

noncomputable theory

section
variables (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (k : ℕ)

-- Conditions
-- The sum of the first 9 terms of the arithmetic sequence \( \{a_n\} \) is equal to the sum of the first 4 terms
def sum_terms_eq : Prop := (S 9) = (S 4)

-- \( a_1 \neq 0 \)
def a1_nonzero : Prop := a 1 ≠ 0

-- \( S_{k+3} = 0 \)
def Sk_plus_3_zero : Prop := S (k + 3) = 0

-- Arithmetic sequence sum definition
def arithmetic_sum (n : ℕ) : ℤ := n * a 1 + (n * (n - 1) / 2) * d

-- Proof statement
theorem find_k : 
(sum_terms_eq a S) →
(a1_nonzero a) →
(Sk_plus_3_zero S k) →
k = 4 :=
by
  sorry
end

end find_k_l43_43132


namespace complex_number_quadrant_l43_43831

-- Lean 4 statement for the equivalent proof problem
theorem complex_number_quadrant
  (z : ℂ)
  (h : z * (1 + complex.i) = 2 * complex.i + 1) :
  0 < z.re ∧ 0 < z.im :=
begin
  sorry
end

end complex_number_quadrant_l43_43831


namespace original_function_l43_43001

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l43_43001


namespace words_per_page_l43_43266

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 270 [MOD 221]) (h2 : p ≤ 120) : p = 107 :=
sorry

end words_per_page_l43_43266


namespace no_condition_problems_l43_43664

def problem1_requires_condition := false
def problem2_requires_condition := false
def problem3_requires_condition := true
def problem4_requires_condition := true

theorem no_condition_problems :
  (¬ problem1_requires_condition) ∧ (¬ problem2_requires_condition) ∧ 
  problem3_requires_condition ∧ problem4_requires_condition :=
by
  split
  . exact true.intro
  split
  . exact true.intro
  split
  . exact true.intro
  . exact true.intro

end no_condition_problems_l43_43664


namespace intersection_of_sets_l43_43425

def setA : Set ℝ := {x | x^2 - 1 ≥ 0}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets : (setA ∩ setB) = {x | 1 ≤ x ∧ x < 4} := 
by 
  sorry

end intersection_of_sets_l43_43425


namespace first_magnificent_monday_after_school_session_l43_43462

theorem first_magnificent_monday_after_school_session
  (start_day : ℕ) (month_days : ℕ) (is_monday : ℕ → Prop) :
  start_day = 2 → month_days = 31 →
  (∀ d, is_monday d ↔ d = 2 ∨ d = 9 ∨ d = 16 ∨ d = 23 ∨ d = 30) →
  (∀ m, ∃! fifth_monday, (∑ i in range 1..31, if is_monday i then 1 else 0) > 4 → is_monday fifth_monday) →
  ∃! magnificent_monday, magnificent_monday = 30 :=
by
  sorry

end first_magnificent_monday_after_school_session_l43_43462


namespace area_ratio_of_triangles_l43_43582

noncomputable def Area (A B C : Point) : ℝ := sorry

theorem area_ratio_of_triangles
  (ω1 ω2 : Circle)
  (r1 r2 : ℝ)
  (a b c : Line)
  (A1 B1 C1 A2 B2 C2 : Point)
  (tangent_a_ω1 : tangent a ω1)
  (tangent_b_ω1 : tangent b ω1)
  (tangent_c_ω1 : tangent c ω1)
  (tangent_a_ω2 : tangent a ω2)
  (tangent_b_ω2 : tangent b ω2)
  (tangent_c_ω2 : tangent c ω2)
  (A1_on_a : touches A1 ω1 a)
  (B1_on_b : touches B1 ω1 b)
  (C1_on_c : touches C1 ω1 c)
  (A2_on_a : touches A2 ω2 a)
  (B2_on_b : touches B2 ω2 b)
  (C2_on_c : touches C2 ω2 c)
  (ω1_center : ω1.center O1)
  (ω2_center : ω2.center O2)
  (ω1_radius : ω1.radius = r1)
  (ω2_radius : ω2.radius = r2) :
  Area A1 B1 C1 / Area A2 B2 C2 = r1 / r2 := sorry

end area_ratio_of_triangles_l43_43582


namespace gcd_proof_l43_43192

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43192


namespace relationship_among_abc_l43_43878

noncomputable section

open Real

def a : ℝ := 0.5 ^ 6
def b : ℝ := log 5 0.6
def c : ℝ := 6 ^ 0.5

theorem relationship_among_abc : b < a ∧ a < c :=
by
  sorry

end relationship_among_abc_l43_43878


namespace range_of_a_increasing_function_l43_43754

noncomputable def f (x a : ℝ) := x^3 + a * x + 1 / x

noncomputable def f' (x a : ℝ) := 3 * x^2 - 1 / x^2 + a

theorem range_of_a_increasing_function (a : ℝ) :
  (∀ x : ℝ, x > 1/2 → f' x a ≥ 0) ↔ a ≥ 13 / 4 := 
sorry

end range_of_a_increasing_function_l43_43754


namespace chromatic_number_plane_bounds_l43_43104

-- Definition of chromatic number \(\chi\) of the plane
def chromatic_number_plane := 
  ∀ (χ : ℕ), (∀ coloring : ℝ² → ℕ, 
               (∀ x y : ℝ², dist x y = 1 → coloring x ≠ coloring y) ↔ χ <= ∥chromatic_number_plane∥)

theorem chromatic_number_plane_bounds :
  (4 ≤ chromatic_number_plane ∧ chromatic_number_plane ≤ 7) :=
sorry

end chromatic_number_plane_bounds_l43_43104


namespace least_number_130_divisors_remainder_l43_43226

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem least_number_130_divisors_remainder :
  let n := 130
  let d6 := 6
  let d7 := 7
  let d9 := 9
  let d18 := 18
  let lcm_d6_d7_d9_d18 := lcm d7 d9 * d6 / Nat.gcd (lcm d7 d9) d6 / Nat.gcd (lcm d7 d9 * d6 / Nat.gcd (lcm d7 d9) d6) d18
  lcm_d6_d7_d9_d18 = 126 ∧ ∃ r, (r = n - lcm_d6_d7_d9_d18) ∧ (r = 4)
  :=
by
  let n := 130
  let d6 := 6
  let d7 := 7
  let d9 := 9
  let d18 := 18
  let lcm_d6_d7_d9 := lcm (lcm d7 d9) d18
  have lcm_d6_d7_d9_d18 : ℕ := lcm (lcm_d6_d7_d9 / Nat.gcd lcm_d6_d7_d9 d6) d6
  
  have h1 := Nat.lcm_comm d7 d9
  have h2 := Nat.lcm_comm 7 18
  have h3 := Nat.gcd_comm (lcm d7 d9) d6
  
  show lcm_d6_d7_d9_d18 = 126 from sorry
  
  existsi r := n - lcm_d6_d7_d9_d18
  show r = 4 from sorry

end least_number_130_divisors_remainder_l43_43226


namespace sum_of_areas_of_triangles_l43_43888

theorem sum_of_areas_of_triangles (m n p : ℤ) :
  let vertices := {v : ℝ × ℝ × ℝ | v.1 ∈ {0, 2} ∧ v.2 ∈ {0, 2} ∧ v.3 ∈ {0, 2}},
      triangles := {t : tuple ℝ 9 | 
                     ∃ v1 v2 v3 ∈ vertices, 
                     t = (v1.1, v1.2, v1.3, v2.1, v2.2, v2.3, v3.1, v3.2, v3.3)},
      triangle_area := λ t : tuple ℝ 9, 
        let (x1, y1, z1, x2, y2, z2, x3, y3, z3) := t in
        1 / 2 * real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2) *
                 real.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2 + (z3 - z1) ^ 2) *
                 real.sqrt ((x3 - x2) ^ 2 + (y3 - y2) ^ 2 + (z3 - z2) ^ 2)
  in (∑ t in triangles, triangle_area t) = 48 + real.sqrt 4608 + real.sqrt 3072 :=
sorry

end sum_of_areas_of_triangles_l43_43888


namespace alcohol_to_water_ratio_l43_43587

variables {V r s : ℝ}
hypothesis (h1 : 2 * V > 0) (h2 : 3 * V > 0) (h3 : r > 0) (h4 : s > 0)

theorem alcohol_to_water_ratio (h1 : 2 * V > 0) (h2 : 3 * V > 0) (h3 : r > 0) (h4 : s > 0) :
  (let alcohol := (2 * r * V / (r + 3)) + (3 * s * V / (s + 2)),
       water := (6 * V / (r + 3)) + (6 * V / (s + 2))
   in alcohol / water) = ((2 * r * (r + 3)) + (3 * s * (s + 2))) / (6 * (r + s + 5)) :=
by sorry

end alcohol_to_water_ratio_l43_43587


namespace events_are_mutually_exclusive_but_not_opposite_l43_43846

-- Definitions based on the conditions:
structure BallBoxConfig where
  ball1 : Fin 4 → ℕ     -- Function representing the placement of ball number 1 into one of the 4 boxes
  h_distinct : ∀ i j, i ≠ j → ball1 i ≠ ball1 j

def event_A (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 1
def event_B (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 2

-- The proof problem:
theorem events_are_mutually_exclusive_but_not_opposite (cfg : BallBoxConfig) :
  (event_A cfg ∨ event_B cfg) ∧ ¬ (event_A cfg ∧ event_B cfg) :=
sorry

end events_are_mutually_exclusive_but_not_opposite_l43_43846


namespace find_x0_l43_43406

noncomputable def slopes_product_eq_three (x : ℝ) : Prop :=
  let y1 := 2 - 1 / x
  let y2 := x^3 - x^2 + 2 * x
  let dy1_dx := 1 / (x^2)
  let dy2_dx := 3 * x^2 - 2 * x + 2
  dy1_dx * dy2_dx = 3

theorem find_x0 : ∃ (x0 : ℝ), slopes_product_eq_three x0 ∧ x0 = 1 :=
by {
  use 1,
  sorry
}

end find_x0_l43_43406


namespace prime_k_form_l43_43729

theorem prime_k_form {p k : ℕ} (hp : Nat.Prime p) :
  (∀ n : ℕ+, 
     p^(Int.ofNat (Nat.floor ((k-1 : ℕ) * (n : ℕ) / (p-1))) + 1)
     + ((kn)! / n)! 
    ) -> (∃ α : ℕ, k = p^α) := 
sorry

end prime_k_form_l43_43729


namespace min_value_expression_l43_43738

open Classical

theorem min_value_expression (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 25 ∧ ∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y = 1 → (4*x/(x - 1) + 9*y/(y - 1)) ≥ m :=
by 
  sorry

end min_value_expression_l43_43738


namespace find_m_l43_43782

theorem find_m (m : ℝ) : (mx - y - 1 = 0 → x - 2y + 3 = 0) → m = 1/2 :=
  sorry

end find_m_l43_43782


namespace initial_total_money_l43_43365

theorem initial_total_money (a b d : ℕ) (c : ℕ := 24) :
  let b' := 2 * b,
      c' := 2 * c,
      d' := 2 * d,
      a' := a - (b + c + d),

      a'' := 2 * a',
      c'' := 2 * c',
      d'' := 2 * d',
      b'' := b' - (a' + c' + d'),

      a''' := 2 * a'',
      b''' := 2 * b'',
      d''' := 2 * d'',
      c''' := c'' - (a'' + b'' + d''),

      a_iv := 2 * a''',
      b_iv := 2 * b''',
      c_iv := 2 * c''',
      d_iv := d''' - (a''' + b''' + c''')
  in d_iv = d → a + b + c + d = 96 :=
sorry

end initial_total_money_l43_43365


namespace range_of_a_l43_43415

noncomputable def f (x a b : ℝ) : ℝ := real.exp (2 * x) - a * x^2 + b * x - 1

theorem range_of_a (a b : ℝ) (h₀ : f 1 a b = 0) (h₁ : ∃ x₁ x₂ ∈ Ioo 0 1, (∀ x ∈ Ioo 0 1, deriv (deriv (f x a b)) x = 0 → x = x₁ ∨ x = x₂)) :
  a ∈ Ioo (real.exp 2 - 3) (real.exp 2 + 1) :=
sorry

end range_of_a_l43_43415


namespace kids_meals_sold_l43_43310

/--
Given the ratio of kids meals sold to adult meals sold is 10:7,
and there are 49 adult meals sold,
prove that the number of kids meals sold is 70.
-/
theorem kids_meals_sold (adult_meals : ℕ) (ratio_kids : ℕ) (ratio_adults : ℕ) (num_adult_meals : ℕ) :
  ratio_kids = 10 → ratio_adults = 7 → num_adult_meals = 49 → adult_meals = num_adult_meals → 
  ∃ kids_meals : ℕ, kids_meals = (ratio_kids * (num_adult_meals / ratio_adults)) := 
by
  intros h1 h2 h3 h4
  use 70
  rw [h1, h2, h3]
  simp
  sorry

end kids_meals_sold_l43_43310


namespace integral_value_l43_43337

noncomputable def integral_expr := ∫ x in -1..1, (sqrt (1 - x^2) + x)

theorem integral_value : integral_expr = Real.pi / 2 := by
  sorry

end integral_value_l43_43337


namespace area_of_enclosed_region_is_1257_l43_43095

/-- Represents a square with given side length and coordinate setup --/
structure Square (side: ℝ) :=
  (A B C D: ℝ × ℝ)
  (A_eq: A = (0, 0))
  (B_eq: B = (side, 0))
  (C_eq: C = (side, side))
  (D_eq: D = (0, side))

/-- Set T is the set of all line segments of length 4 with endpoints on adjacent sides --/
def T (sq: Square 4) : set (ℝ × ℝ) :=
  {p | ∃ x y: ℝ, (x, 0) = p ∨ (0, y) = p ∨ (x, 4) = p ∨ (4, y) = p ∧ x^2 + y^2 = 16}

/-- The midpoints of line segments in set T form an ellipse enclosing a specified region 
  whose area to the nearest hundredth is m --/
def enclosed_midpoints_area_100m : ℝ := 100 * Real.pi * 2 * 2

/-- Main proposition: proving 100m = 1257 --/
theorem area_of_enclosed_region_is_1257 (sq: Square 4) (set_T : T sq) : enclosed_midpoints_area_100m = 1257 :=
by
  sorry

end area_of_enclosed_region_is_1257_l43_43095


namespace domain_of_function_l43_43553

theorem domain_of_function :
  {x : ℝ | 0 < x ∧ x ≤ 1000} = {x : ℝ | 0 < x ∧ 3 - real.log10 x ≥ 0} :=
by
  sorry

end domain_of_function_l43_43553


namespace central_angle_of_regular_polygon_l43_43776

theorem central_angle_of_regular_polygon (n : ℕ) (h : 360 ∣ 360 - 36 * n) :
  n = 10 :=
by
  sorry

end central_angle_of_regular_polygon_l43_43776


namespace daisies_in_garden_l43_43139

noncomputable def number_of_daisies (r t : ℕ) (F : ℕ) (hF : 0.25 * F = r) : ℕ :=
  F - r - t

theorem daisies_in_garden (r t : ℕ) (H1 : r = 25) (H2 : t = 40) (F : ℕ) (H3 : 0.25 * F = r) :
  number_of_daisies r t F H3 = 35 :=
by
  unfold number_of_daisies
  sorry

end daisies_in_garden_l43_43139


namespace part_one_l43_43481

theorem part_one (a b c A B C : ℝ) (h1 : a = Real.sqrt 2 * c) (h2 : sin A = tan B) (h3 : a^2 + c^2 - b^2 = 2 * b * c) : 
  C = Real.pi / 4 :=
sorry

end part_one_l43_43481


namespace trajectory_equation_of_M_l43_43795

theorem trajectory_equation_of_M (θ : ℝ) (ρ ρ₁ ρ₂ : ℝ)
  (h1 : ρ₁ = 2)
  (h2 : ρ₂ * cos θ = 4)
  (h3 : ρ = (ρ₁ + ρ₂) / 2) :
  ρ = 1 + 2 / (cos θ) := by
  sorry

end trajectory_equation_of_M_l43_43795


namespace isosceles_triangle_of_perpendiculars_intersect_at_one_point_l43_43086

theorem isosceles_triangle_of_perpendiculars_intersect_at_one_point
  (ABC : Type)
  (a b c : ℝ)
  (BC : | BC - a | ≤ 0)
  (CA : | CA - b | ≤ 0)
  (AB : | AB - c | ≤ 0)
  (BD DC b1 b2 c1 c2 : ℝ)
  (h1 : BD = (a * c) / (b + c))
  (h2 : DC = (a * b) / (a + b))
  (h3 : CE = (b ^ 2) / (a + c))
  (h4 : EA = (b * c) / (b + c))
  (h5 : AF = (c ^ 2) / (a + b))
  (h6 : FB = (a * c) / (a + c))
  (perpendiculars_intersect : 
    (a * c / (b + c)) ^ 2 + (a * b / (a + c)) ^ 2 + (b * c / (a + b)) ^ 2 = 
    (a * b / (b + c)) ^ 2 + (b * c / (a + c)) ^ 2 + (a * c / (a + b)) ^ 2) :
  (b = c) ∨ (a = b) ∨ (a = c) := by
  sorry

end isosceles_triangle_of_perpendiculars_intersect_at_one_point_l43_43086


namespace find_f_2005_l43_43385

open Int

noncomputable def f : ℕ → ℤ := sorry

axiom condition1 : ∀ n : ℕ, n > 0 → f(n + 2) = 2 * f(n + 1) - f(n)
axiom condition2 : f 1 = 2
axiom condition3 : f 3 = 6

theorem find_f_2005 : f 2005 = 4010 := sorry

end find_f_2005_l43_43385


namespace outfit_choices_l43_43437

theorem outfit_choices:
  let shirts := 8
  let pants := 8
  let hats := 8
  -- Each has 8 different colors
  -- No repetition of color within type of clothing
  -- Refuse to wear same color shirt and pants
  (shirts * pants * hats) - (shirts * hats) = 448 := 
sorry

end outfit_choices_l43_43437


namespace cube_triangle_area_sum_solution_l43_43900

def cube_vertex_triangle_area_sum (m n p : ℤ) : Prop :=
  m + n + p = 121 ∧
  (∀ (a : ℕ) (b : ℕ) (c : ℕ), a * b * c = 8) -- Ensures the vertices are for a 2*2*2 cube

theorem cube_triangle_area_sum_solution :
  cube_vertex_triangle_area_sum 48 64 9 :=
by
  unfold cube_vertex_triangle_area_sum
  split
  · exact rfl -- m + n + p = 121
  · intros a b c h
    sorry -- Conditions ensuring these m, n, p were calculated from a 2x2x2 cube

end cube_triangle_area_sum_solution_l43_43900


namespace last_years_harvest_l43_43329

theorem last_years_harvest (this_year_harvest : ℕ) (increase : ℕ) (last_year_harvest : ℕ) :
  this_year_harvest = 8564 → increase = 6085 → last_year_harvest = 8564 - 6085 → last_year_harvest = 2479 :=
by
  intros
  rw [h, h_1] at h_2
  exact h_2

end last_years_harvest_l43_43329


namespace leftover_value_in_dollars_l43_43292

-- Definitions based on conditions
def roll_of_quarters : ℕ := 25
def roll_of_dimes : ℕ := 40
def john_quarters : ℕ := 47
def john_dimes : ℕ := 71
def mark_quarters : ℕ := 78
def mark_dimes : ℕ := 132

-- Main theorem statement
theorem leftover_value_in_dollars :
  let total_quarters := john_quarters + mark_quarters
  let total_dimes := john_dimes + mark_dimes
  let leftover_quarters := total_quarters % roll_of_quarters
  let leftover_dimes := total_dimes % roll_of_dimes
  (leftover_quarters * 0.25) + (leftover_dimes * 0.10) = 2.30 :=
sorry

end leftover_value_in_dollars_l43_43292


namespace initial_alcohol_solution_percentage_l43_43950

noncomputable def initial_percentage_of_alcohol (P : ℝ) :=
  let initial_volume := 6 -- initial volume of solution in liters
  let added_alcohol := 1.2 -- added volume of pure alcohol in liters
  let final_volume := initial_volume + added_alcohol -- final volume in liters
  let final_percentage := 0.5 -- final percentage of alcohol
  ∃ P, (initial_volume * (P / 100) + added_alcohol) / final_volume = final_percentage

theorem initial_alcohol_solution_percentage : initial_percentage_of_alcohol 40 :=
by 
  -- Prove that initial percentage P is 40
  have hs : initial_percentage_of_alcohol 40 := by sorry
  exact hs

end initial_alcohol_solution_percentage_l43_43950


namespace original_price_l43_43629

theorem original_price (P : ℝ) (S : ℝ) (h1 : S = 1.3 * P) (h2 : S = P + 650) : P = 2166.67 :=
by
  sorry

end original_price_l43_43629


namespace number_of_paintings_of_red_cubes_l43_43616

theorem number_of_paintings_of_red_cubes : 
  let n := 4 in
  let k := 16 in
  let total_cubes := n * n * n in
  let red_cubes := k in
  let rows := n in
  let cols := n in
  -- Define that each row and each column in a 4x4 grid contains exactly one red cube
  let condition := ∀ (grid: fin n → fin n → bool),
    (∀ r, (∃! c, grid r c) ∧ ∀ c, (∃! r, grid r c)) in
  -- The number of valid placements of red cubes satisfying the condition
  let valid_configurations := 24 * 24 in
  red_cubes = 16 →
  total_cubes = 64 →
  valid_configurations = 576 :=
by intros; sorry

end number_of_paintings_of_red_cubes_l43_43616


namespace four_real_solutions_l43_43352

-- Definitions used in the problem
def P (x : ℝ) : Prop := (6 * x) / (x^2 + 2 * x + 5) + (4 * x) / (x^2 - 4 * x + 5) = -2 / 3

-- Statement of the problem
theorem four_real_solutions : ∃ (x1 x2 x3 x4 : ℝ), P x1 ∧ P x2 ∧ P x3 ∧ P x4 ∧ 
  ∀ x, P x → (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :=
sorry

end four_real_solutions_l43_43352


namespace twenty_ninth_number_base_five_l43_43787

def base_five (n : ℕ) : string :=
  if n = 0 then "0"
  else 
    let rec digits (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc
      else digits (n / 5) ((n % 5)::acc)
    in 
    (digits n []).foldl (λ acc d, acc ++ to_string d) ""

theorem twenty_ninth_number_base_five : base_five 29 = "104" :=
  sorry

end twenty_ninth_number_base_five_l43_43787


namespace polynomial_at_most_one_positive_root_l43_43990

theorem polynomial_at_most_one_positive_root :
  ∀ (P : ℝ[X]),
  P = (λ x, x^2022 - 2*x^2021 - 3*x^2020 - ∑ k in (finset.range 2020), (k + 4) * x^(2020 - k)) →
  ∃! x > 0, P.eval x = 0 :=
begin
  sorry
end

end polynomial_at_most_one_positive_root_l43_43990


namespace river_current_speed_correct_l43_43636

noncomputable def speed_of_river_current (p : ℝ) (distance_AB : ℝ) (meeting_time : ℝ) : ℝ :=
  let r := 5 in
  let kayak_speed := 1.5 * r in
  let power_boat_down_time := distance_AB / (p + r) in
  let kayak_distance := meeting_time * kayak_speed in
  let power_boat_up_distance := (p - r) * (meeting_time - power_boat_down_time) in
  if (distance_AB + power_boat_up_distance = kayak_distance) then r else 0

theorem river_current_speed_correct :
  speed_of_river_current 10 20 6 = 5 :=
begin
  sorry
end

end river_current_speed_correct_l43_43636


namespace eight_digit_decreasing_mod_1000_l43_43050

theorem eight_digit_decreasing_mod_1000 :
  let M := (Finset.range 10).raising_subsequences 8
  in M.card % 1000 = 310 :=
by
  -- Placeholder for proof
  sorry

end eight_digit_decreasing_mod_1000_l43_43050


namespace area_triangle_ABC_l43_43078

variables (R x : ℝ)
-- conditions
def parallel_lines : Prop := ∃ l₁ l₂ : set (ℝ × ℝ), l₁ ∥ l₂ ∧ ( ∃ A : ℝ × ℝ, A ∈ l₁ ∧ dist A (0,0) = R ) ∧ 
  ∃ B C : ℝ × ℝ, B ∈ l₂ ∧ C ∈ l₂ ∧ dist (B,C) = x

theorem area_triangle_ABC (R x : ℝ) (h : parallel_lines R x) : 
  ∃ S : ℝ, S = x * (sqrt(2 * R * x - x^2)) :=
sorry

end area_triangle_ABC_l43_43078


namespace remainder_of_expansion_l43_43230

theorem remainder_of_expansion (x : ℤ) : ((x + 1) ^ 2012) % (x^2 - x + 1) = 1 := 
  sorry

end remainder_of_expansion_l43_43230


namespace ap_contains_sixth_power_l43_43670

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end ap_contains_sixth_power_l43_43670


namespace gcd_proof_l43_43195

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43195


namespace complex_numbers_count_l43_43005

theorem complex_numbers_count (z : ℂ) (h1 : z^24 = 1) (h2 : ∃ r : ℝ, z^6 = r) : ℕ :=
  sorry -- Proof goes here

end complex_numbers_count_l43_43005


namespace sin_C_value_circumscribed_circle_circumference_l43_43805

variables {A B C R c : ℝ}
variables [Fact (sin A = 4 / 5)] [Fact (cos B = 5 / 13)] [Fact (c = 56)]
-- Note: Fact is used to create these conditions as assumptions.

theorem sin_C_value (h1 : sin A = 4 / 5) (h2 : cos B = 5 / 13) : 
  let sin_B := real.sqrt (1 - cos B ^ 2) in
  let cos_A := real.sqrt (1 - sin A ^ 2) in 
  sin (A + B) = sin A * cos B + cos A * sin B := sorry

theorem circumscribed_circle_circumference (h1 : sin A = 4 / 5) (h2 : cos B = 5 / 13) (h3 : c = 56)
  (h4 : sin (A + B) = sin A * cos B + cos A * sin B) :
  2 * R * real.pi = 65 * real.pi := sorry

end sin_C_value_circumscribed_circle_circumference_l43_43805


namespace baby_whales_on_second_trip_l43_43808

def iwishmael_whales_problem : Prop :=
  let male1 := 28
  let female1 := 2 * male1
  let male3 := male1 / 2
  let female3 := female1
  let total_whales := 178
  let total_without_babies := (male1 + female1) + (male3 + female3)
  total_whales - total_without_babies = 24

theorem baby_whales_on_second_trip : iwishmael_whales_problem :=
  by
  sorry

end baby_whales_on_second_trip_l43_43808


namespace find_value_of_x_l43_43752

def f (x : ℝ) : ℝ := 5 * x^3 + 7

theorem find_value_of_x (x : ℝ) (h : f (3) = x) : x = 142 := 
by 
  have f3 : ℝ := f(3)
  sorry

end find_value_of_x_l43_43752


namespace ap_contains_sixth_power_l43_43669

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end ap_contains_sixth_power_l43_43669


namespace length_of_RZ_l43_43474

-- Definitions based on conditions
def AC_parallel_WZ : Prop := -- placeholder definition that states AC is parallel to WZ
sorry 

def AW_eq_60 : Prop := AW = 60

def CR_eq_15 : Prop := CR = 15

def RW_eq_45 : Prop := RW = 45

-- The theorem that we need to prove
theorem length_of_RZ (AC_WZ_parallel : AC_parallel_WZ) (AW_60 : AW_eq_60) (CR_15 : CR_eq_15) (RW_45 : RW_eq_45) :
  RZ = 45 := 
sorry

end length_of_RZ_l43_43474


namespace num_ways_to_arrange_cousins_l43_43512

-- Define the problem
theorem num_ways_to_arrange_cousins :
  ∃ (n : ℕ), n = 76 ∧ ∀ (cousins : Fin 5 → ℕ), ∀ (rooms : Fin 4 → ℕ), 
  (sum (λ r : Fin 4, rooms r) = 5) ∧ 
  (room_arrangement: ∃ (b : Fin 5 → Fin 4), true) → 
  n = 76 := 
sorry

end num_ways_to_arrange_cousins_l43_43512


namespace line_through_point_equal_distances_l43_43300

-- Definitions for circles, points, and lines will be necessary
noncomputable theory

structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def intersects (P : Point) (c1 c2 : Circle) : Prop :=
  ∃ (A₁ A₂ : Point), distance P A₁ = distance P A₂ ∧
                      distance c1.center A₁ = c1.radius ∧
                      distance c2.center A₂ = c2.radius

theorem line_through_point_equal_distances :
  ∀ (O₁ O₂ P : Point) (r₁ r₂ : ℝ),
  let c1 := Circle.mk O₁ r₁ in
  let c2 := Circle.mk O₂ r₂ in
  intersects P c1 c2 :=
sorry

end line_through_point_equal_distances_l43_43300


namespace gcd_765432_654321_l43_43172

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43172


namespace find_a_range_l43_43386

-- Define the function f(x) as given in the problem
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 1 then a * x^2 - x - 1/4 else Real.log a x - 1

-- Non-strictly decreasing function condition
def is_decreasing (a : ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 a - f x2 a) / (x1 - x2) < 0

-- Main theorem statement translating the math problem
theorem find_a_range (a : ℝ) : (0 < a ∧ a < 1) ∧ (∀ x, x ≤ 1 → x < 1/(2*a)) ∧ (a = 1/4)
  → (1/4 ≤ a ∧ a ≤ 1/2) :=
by
  intro h
  sorry

end find_a_range_l43_43386


namespace elvis_editing_time_is_correct_l43_43701

def elvis_editing_time : ℕ :=
let number_of_songs := 10
let studio_time_hours := 5
let recording_time_per_song := 12
let writing_time_per_song := 15
let total_writing_time := number_of_songs * writing_time_per_song
let total_recording_time := number_of_songs * recording_time_per_song
let total_studio_time := studio_time_hours * 60
let total_time_spent := total_writing_time + total_recording_time
total_studio_time - total_time_spent

theorem elvis_editing_time_is_correct : elvis_editing_time = 30 := by
  simp [elvis_editing_time]
  sorry

end elvis_editing_time_is_correct_l43_43701


namespace perfect_matching_pins_through_polygons_l43_43508

theorem perfect_matching_pins_through_polygons (n : ℕ)
  (h_n : n = 2019) 
  (areas : fin (n^2) → ℝ)
  (h_areas : ∀ i, areas i = 1) 
  (intersects : fin (n^2) → fin (n^2) → Prop)
  (h_intersects : ∀ i, ∃ j, intersects i j):
  ∃ f : fin (n^2) → fin (n^2), function.bijective f :=
by
  sorry

end perfect_matching_pins_through_polygons_l43_43508


namespace gcd_765432_654321_l43_43183

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43183


namespace hours_per_day_is_8_l43_43150

-- Define the conditions
def hire_two_bodyguards (day_count : ℕ) (total_payment : ℕ) (hourly_rate : ℕ) (daily_hours : ℕ) : Prop :=
  2 * hourly_rate * day_count * daily_hours = total_payment

-- Define the correct answer
theorem hours_per_day_is_8 :
  hire_two_bodyguards 7 2240 20 8 :=
by
  -- Here, you would provide the step-by-step justification, but we use sorry since no proof is required.
  sorry

end hours_per_day_is_8_l43_43150


namespace james_burns_300_calories_per_hour_walking_l43_43027

-- Definitions based on given conditions
def burns_calories_walking_per_hour (C : ℕ) := C
def burns_calories_dancing_per_hour (C : ℕ) := 2 * C

def hours_dancing_per_day := 2 * 0.5
def days_dancing_per_week := 4
def total_hours_dancing_per_week := hours_dancing_per_day * days_dancing_per_week

def calories_burned_per_week_dancing (C : ℕ) := 
  burns_calories_dancing_per_hour C * total_hours_dancing_per_week

-- Given conditions
def given_calories_burned_per_week_dancing := 2400

-- The proof statement to prove
theorem james_burns_300_calories_per_hour_walking : 
  ∀ (C : ℕ), C = 300 ↔ calories_burned_per_week_dancing C = given_calories_burned_per_week_dancing :=
by
  sorry

end james_burns_300_calories_per_hour_walking_l43_43027


namespace blue_polygons_exceed_red_polygons_by_1770_l43_43075

theorem blue_polygons_exceed_red_polygons_by_1770:
  ∃ red_points blue_point,
  set.card red_points = 60 ∧ 
  blue_point ∉ red_points ∧
  ∀ polygons, polygons ⊆ red_points ∨ polygons ⊆ red_points ∪ {blue_point} →
  (count_blue_polygons red_points blue_point) - (count_red_polygons red_points) = 1770 :=
by
  sorry

end blue_polygons_exceed_red_polygons_by_1770_l43_43075


namespace a2020_less_than_inverse_2020_l43_43493

theorem a2020_less_than_inverse_2020 (a_0 : ℝ) (h_a0 : a_0 > 0) :
  (∀ n : ℕ, 1 ≤ n → 2020 ≥ n → ∃ a_n : ℝ, 
    (a_n = (a_n - 1 / (has_sqrt.sqrt (1 + 2020 * (a_n - 1) ^ 2))))) →

  a_2020 < 1 / 2020 :=
by
  sorry

end a2020_less_than_inverse_2020_l43_43493


namespace sum_S_15_l43_43756

def a_n (n : ℕ) : ℤ := (-1)^(n-1) * (n-1)

def S_n (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a_n (i + 1)

theorem sum_S_15 : S_n 15 = 7 :=
by
  sorry

end sum_S_15_l43_43756


namespace max_value_expression_l43_43443

noncomputable theory

def abs (x : ℚ) : ℚ := if x < 0 then -x else x

def sign (x : ℚ) : ℚ := if x < 0 then -1 else 1

theorem max_value_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ M : ℚ, M = 1 ∧ ∀ x y : ℚ, x ≠ 0 → y ≠ 0 →
    (x / abs x + abs y / y - x * y / abs (x * y)) ≤ M :=
by
  sorry

end max_value_expression_l43_43443


namespace simplify_expression_l43_43535

theorem simplify_expression (a b : ℕ) (h : a / b = 1 / 3) : 
    1 - (a - b) / (a - 2 * b) / ((a ^ 2 - b ^ 2) / (a ^ 2 - 4 * a * b + 4 * b ^ 2)) = 3 / 4 := 
by sorry

end simplify_expression_l43_43535


namespace geometric_series_nonneg_l43_43524

theorem geometric_series_nonneg (x : ℝ) (hx : x ≥ -1) : 1 + x + x^2 + x^3 + ... + x^2011 ≥ 0 := 
sorry

end geometric_series_nonneg_l43_43524


namespace intersection_point_x_coordinate_l43_43691

noncomputable def hyperbola (x y b : ℝ) := x^2 - (y^2 / b^2) = 1

noncomputable def c := 1 + Real.sqrt 3

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_point_x_coordinate
  (x y b : ℝ)
  (h_hyperbola : hyperbola x y b)
  (h_distance_foci : distance (2 * c, 0) (0, 0) = 2 * c)
  (h_circle_center : distance (x, y) (0, 0) = c)
  (h_p_distance : distance (x, y) (2 * c, 0) = c + 2) :
  x = (Real.sqrt 3 + 1) / 2 :=
sorry

end intersection_point_x_coordinate_l43_43691


namespace ice_cream_cost_calculation_l43_43509

theorem ice_cream_cost_calculation:
  (∀ (q1 q2 b hc pkg_s pkg_r hc_pkg : ℕ) 
      (cost_s cost_r cost_h total_cost: ℤ),
  4 * b + 2 * hc = 4 ∧ 
  q1 = 1 ∧ 
  q2 = 1 ∧ 
  pkg_s = 2 ∧ 
  cost_s = 3 ∧ 
  pkg_r = 2 ∧ 
  cost_r = 5 ∧ 
  hc_pkg = 4 ∧ 
  cost_h = 4 ∧ 
  b = 4 ∧ 
  hc = 2 ∧ 
  total_cost = (cost_s * 2) + (cost_r * 2) + cost_h := 20 :=
  q1 * q2 * b * hc =
  total_cost = 20

end ice_cream_cost_calculation_l43_43509


namespace gcd_765432_654321_eq_3_l43_43219

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43219


namespace midpoints_area_l43_43092

def square_area : ℝ := 16
def quarter_circle_area : ℝ := (Real.pi * (2^2)) / 4

theorem midpoints_area (side_length : ℝ) (segment_length : ℝ) : 
  side_length = 4 → 
  segment_length = 4 → 
  100 * (square_area - 4 * quarter_circle_area) = 972 := 
by
  intros h1 h2
  let square_area := side_length ^ 2
  let quarter_circle_area := (Real.pi * (segment_length ^ 2) / 4)
  have h3 : square_area = 16 := by sorry
  have h4 : quarter_circle_area = (\(∏ ^ 2) / 4) := by sorry
  rw [h1, h3, h4]
  rw [← sub_eq_add_neg, mul_sub, mul_one, sub_eq_add_neg]
  norm_num
  sorry

end midpoints_area_l43_43092


namespace seedlings_planted_by_father_l43_43850

theorem seedlings_planted_by_father (remi_day1_seedlings : ℕ) (total_seedlings : ℕ) :
  remi_day1_seedlings = 200 →
  total_seedlings = 1200 →
  let remi_day2_seedlings := 2 * remi_day1_seedlings in
  total_seedlings = remi_day1_seedlings + remi_day2_seedlings + 600 :=
begin
  assume h1 h2,
  sorry,
end

end seedlings_planted_by_father_l43_43850


namespace triangle_area_correct_l43_43861

-- Define the necessary objects and proofs
noncomputable def triangle_area_tangent_curve : ℝ :=
  let x0 := 3
  let y0 := x0^3
  let dydx := 3 * x0^2
  let tangent_line := λ x : ℝ, dydx * (x - x0) + y0
  let x_intercept := x0 - y0 / dydx
  let y_intercept := tangent_line 0
  let area := 1 / 2 * x_intercept * -y_intercept
  area

theorem triangle_area_correct : triangle_area_tangent_curve = 54 := by
  sorry

end triangle_area_correct_l43_43861


namespace example_numbers_exist_l43_43959

-- Define the operation that Maria performs
def operation (n : ℕ) : ℕ :=
  if ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ d1 = n / 10 ∧ d2 = n % 10 then
    3 * n
  else if ∃ d : ℕ, d = n / 10 ∧ d = n % 10 then
    (n / 10)
  else
    n

def Maria_numbers := { n : ℕ | 10 ≤ n ∧ n < 100 ∧ operation (operation (operation n)) = n }

theorem example_numbers_exist : ∃ n : ℕ, n ∈ Maria_numbers :=
by
  use 25
  dsimp [Maria_numbers]
  simp
  sorry

end example_numbers_exist_l43_43959


namespace gcd_765432_654321_l43_43180

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43180


namespace travel_company_promotion_l43_43297

theorem travel_company_promotion :
  ∃ (x : ℕ), 13 + 4 * x = x + 100 ∧ x = 29 :=
by
  existsi 29,
  split,
  { norm_num, },
  { norm_num, }

end travel_company_promotion_l43_43297


namespace problem_l43_43753

noncomputable def f (x : ℝ) := sin x * cos x - sqrt 3 * (cos x)^2 + (sqrt 3) / 2

theorem problem (k : ℤ) :
  (∀ x, f (x + π) = f x) ∧
  (∀ t, f t = sin (2 * t - π / 3)) ∧
  (∀ x, (2 * x - π / 3) = k * π → (x = k * π / 2 + π / 6)) ∧
  (∀ x, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) → ∀ t ∈ Icc (k * π - π / 12) (k * π + 5 * π / 12), deriv f t > 0) ∧
  (∀ x ∈ Icc 0 (π / 2), - (sqrt 3) / 2 ≤ f x ∧ f x ≤ 1) ∧
  (∀ x ∈ Icc 0 (π / 2), f x = 1 → x = π / 3) ∧
  (∀ x ∈ Icc 0 (π / 2), f x = - (sqrt 3) / 2 → x = 0) := 
sorry

end problem_l43_43753


namespace sum_of_x_coordinates_l43_43074

theorem sum_of_x_coordinates : 
  ∑ x in finset.filter (λ x, (3 * x + 4) % 13 = (8 * x + 9) % 13) (finset.range 13), x = 6 := 
by
  sorry

end sum_of_x_coordinates_l43_43074


namespace parkway_elementary_students_l43_43799

/-- The number of students in the fifth grade at Parkway Elementary School. -/
theorem parkway_elementary_students :
  ∀ (boys total_students_play_soccer : ℕ) 
    (percent_boys_play_soccer : ℚ)
    (girls_not_play_soccer : ℕ),
    boys = 312 →
    total_students_play_soccer = 250 →
    percent_boys_play_soccer = 82 / 100 →
    girls_not_play_soccer = 63 →
    let boys_play_soccer := percent_boys_play_soccer * total_students_play_soccer in
    let boys_not_play_soccer := boys - boys_play_soccer in
    let total_not_play_soccer := boys_not_play_soccer + girls_not_play_soccer in
    let total_students := total_students_play_soccer + total_not_play_soccer in
    total_students = 420 :=
by
  intros boys total_students_play_soccer percent_boys_play_soccer girls_not_play_soccer
  intros Hboys Htotal_soccer Hpercent Hgirls_not_play
  let boys_play_soccer := percent_boys_play_soccer * total_students_play_soccer
  have boys_play_soccer_def : boys_play_soccer = 205 := by sorry
  let boys_not_play_soccer := boys - boys_play_soccer
  have boys_not_play_soccer_def : boys_not_play_soccer = 107 := by sorry
  let total_not_play_soccer := boys_not_play_soccer + girls_not_play_soccer
  have total_not_play_soccer_def : total_not_play_soccer = 170 := by sorry
  let total_students := total_students_play_soccer + total_not_play_soccer
  have total_students_def : total_students = 420 := by sorry
  exact total_students_def

end parkway_elementary_students_l43_43799


namespace smallest_m_satisfying_condition_l43_43381

noncomputable def a : ℕ → ℕ
| 0       := 3
| (n+1) := 3^(a n)

noncomputable def b : ℕ → ℕ
| 0       := 100
| (n+1) := 100^(b n)

theorem smallest_m_satisfying_condition : ∃ m : ℕ, b m > a 100 ∧ ∀ k : ℕ, k < m → b k ≤ a 100 :=
begin
  use 99,
  sorry
end

end smallest_m_satisfying_condition_l43_43381


namespace sum_abc_eq_neg_ten_thirds_l43_43822

variable (a b c d y : ℝ)

-- Define the conditions
def condition_1 : Prop := a + 2 = y
def condition_2 : Prop := b + 3 = y
def condition_3 : Prop := c + 4 = y
def condition_4 : Prop := d + 5 = y
def condition_5 : Prop := a + b + c + d + 6 = y

-- State the theorem
theorem sum_abc_eq_neg_ten_thirds
    (h1 : condition_1 a y)
    (h2 : condition_2 b y)
    (h3 : condition_3 c y)
    (h4 : condition_4 d y)
    (h5 : condition_5 a b c d y) :
    a + b + c + d = -10 / 3 :=
sorry

end sum_abc_eq_neg_ten_thirds_l43_43822


namespace orthogonal_vectors_l43_43705

/-- Define the two vectors. -/
def v1 : ℝ^4 :=
  ⟨2, -4, 3, 1⟩

def v2 (z : ℝ) : ℝ^4 :=
  ⟨-3, z, 4, -2⟩

/-- Define the dot product of two vectors. -/
def dot_product (u v : ℝ^4) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 + u.4 * v.4

/-- Prove that z = 1 for the vectors to be orthogonal. -/
theorem orthogonal_vectors :
  (dot_product v1 (v2 1)) = 0 :=
by
  sorry

end orthogonal_vectors_l43_43705


namespace proj_onto_w_equals_correct_l43_43356

open Real

noncomputable def proj (w v : ℝ × ℝ) : ℝ × ℝ :=
  let dot (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
  let scalar_mul c (a : ℝ × ℝ) := (c * a.1, c * a.2)
  let w_dot_w := dot w w
  if w_dot_w = 0 then (0, 0) else scalar_mul (dot v w / w_dot_w) w

theorem proj_onto_w_equals_correct (v w : ℝ × ℝ)
  (hv : v = (2, 3))
  (hw : w = (-4, 1)) :
  proj w v = (20 / 17, -5 / 17) :=
by
  -- The proof would go here. We add sorry to skip it.
  sorry

end proj_onto_w_equals_correct_l43_43356


namespace arctan_equivalence_l43_43687

def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)
def arctan_deg (x : ℝ) : ℝ := Real.arctan x * 180 / Real.pi

theorem arctan_equivalence :
  arctan_deg (tan_deg 75 - 3 * tan_deg 30 + tan_deg 45) = 15 :=
by
  have h1 : tan_deg 30 = 1 / Real.sqrt 3 := by sorry
  have h2 : tan_deg 45 = 1 := by sorry
  have h3 : tan_deg 75 = (tan_deg 45 + tan_deg 30) / (1 - tan_deg 45 * tan_deg 30) := by sorry
  sorry

end arctan_equivalence_l43_43687


namespace letter_addition_problem_l43_43804

theorem letter_addition_problem (S I X : ℕ) (E L V N : ℕ) 
  (hS : S = 8) 
  (hX_odd : X % 2 = 1)
  (h_diff_digits : ∀ (a b c d e f : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a)
  (h_sum : 2 * S * 100 + 2 * I * 10 + 2 * X = E * 10000 + L * 1000 + E * 100 + V * 10 + E + N) :
  I = 3 :=
by
  sorry

end letter_addition_problem_l43_43804


namespace housewife_more_oil_l43_43639

variables (x : ℝ)

def f (x : ℝ) := 0.7 * x = 30

def y1 (x : ℝ) := 900 / x

def y2 : ℝ := 900 / 30

theorem housewife_more_oil (h : f x) : y2 - y1 x = 9 := 
sorry

end housewife_more_oil_l43_43639


namespace determine_k_for_one_real_solution_l43_43718

theorem determine_k_for_one_real_solution (k : ℝ):
  (∃ x : ℝ, 9 * x^2 + k * x + 49 = 0 ∧ (∀ y : ℝ, 9 * y^2 + k * y + 49 = 0 → y = x)) → k = 42 :=
sorry

end determine_k_for_one_real_solution_l43_43718


namespace friends_total_games_l43_43036

theorem friends_total_games (new_games : ℕ) (old_games : ℕ) (h1 : new_games = 88) (h2 : old_games = 53) :
  new_games + old_games = 141 :=
by
  rw [h1, h2]
  norm_num
  sorry

end friends_total_games_l43_43036


namespace gcd_765432_654321_l43_43169

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43169


namespace complex_number_magnitude_l43_43727

theorem complex_number_magnitude (z : ℂ) (h : (1 + complex.i) / z = 1 - complex.i) : complex.abs z = 1 :=
sorry

end complex_number_magnitude_l43_43727


namespace village_population_l43_43614

-- Defining the variables and the condition
variable (P : ℝ) (h : 0.9 * P = 36000)

-- Statement of the theorem to prove
theorem village_population : P = 40000 :=
by sorry

end village_population_l43_43614


namespace triangle_possible_combinations_and_area_l43_43746

theorem triangle_possible_combinations_and_area (a b c A B C: ℝ)
    (h1 : a = sqrt 3)
    (h2 : b = 2)
    (h3 : (sin B + sin C) / sin A = (a + c) / (b - c))
    (h4 : cos^2 ((B - C) / 2) - sin B * sin C = 1 / 4)
    (angle_sum : A + B + C = π)
    (angle_positive_A : 0 < A)
    (angle_positive_B : 0 < B)
    (angle_positive_C : 0 < C) :
    a = sqrt 3 ∧ b = 2 ∧ ((sin B + sin C) / sin A = (a + c) / (b - c)) → 
        (∃ S : ℝ, S = (3 * (sqrt 7 - sqrt 3)) / 8) ∧
    a = sqrt 3 ∧ b = 2 ∧ (cos^2 ((B - C) / 2) - sin B * sin C = 1 / 4) → 
        (∃ S : ℝ, S = sqrt 3 / 2) :=
    sorry

end triangle_possible_combinations_and_area_l43_43746


namespace distance_after_2_hours_l43_43484

def jay_speed : ℝ := 1 / 20 -- miles per minute
def paul_speed : ℝ := 3 / 40 -- miles per minute
def initial_distance : ℝ := 2 -- miles
def time_in_minutes : ℝ := 120 -- minutes

theorem distance_after_2_hours : 
  let jay_distance := jay_speed * time_in_minutes in
  let paul_distance := paul_speed * time_in_minutes in
  initial_distance + jay_distance + paul_distance = 17 :=
by
  sorry

end distance_after_2_hours_l43_43484


namespace pentagon_diagonal_relationship_l43_43788

variable (a b c : ℝ)

-- The problem conditions can be formalized here
axiom regular_pentagon (r : ℝ) : 2 * r * (1 - cos (144 * real.pi / 180)) = b^2 ∧ 2 * r * (1 - cos (216 * real.pi / 180)) = c^2

theorem pentagon_diagonal_relationship (r : ℝ) (condition : 2 * r * (1 - cos (144 * real.pi / 180)) = b^2 ∧ 2 * r * (1 - cos (216 * real.pi / 180)) = c^2) : 
  c > b :=
by 
  sorry

end pentagon_diagonal_relationship_l43_43788


namespace gcd_20244_46656_l43_43350

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end gcd_20244_46656_l43_43350


namespace number_of_true_statements_l43_43976

def statement1 : Prop :=
  ∀ x : ℝ, 4 * sin (2 * (x + π / 3)) = 4 * sin (2 * x + π / 3)

def statement2 : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, 4 * cos (2 * (π / 6 - x) + (k * π + π / 6)) = -4 * cos (2 * x + (k * π + π / 6))

def statement3 : Prop :=
  ∀ x : ℝ, 4 * tan x / (1 - tan ^ 2 x) = 2 * tan (2 * x)

def statement4 : Prop :=
  sqrt (1 + sin 2) - sqrt (1 - sin 2) = 2 * sin 1

theorem number_of_true_statements : 
  (¬ statement1) ∧ statement2 ∧ statement3 ∧ ¬ statement4 →
  (2 : ℕ) = 2 :=
by
  intro h
  sorry

end number_of_true_statements_l43_43976


namespace expand_polynomial_l43_43339

variable {R : Type*} [CommRing R]

-- Define the polynomial expression
def polynomial_expansion (x : R) : R := (7 * x^2 + 5 * x + 8) * 3 * x

-- The theorem to expand the expression
theorem expand_polynomial (x : R) :
  polynomial_expansion x = 21 * x^3 + 15 * x^2 + 24 * x :=
by {
  sorry
}

end expand_polynomial_l43_43339


namespace movie_length_after_cuts_l43_43915

theorem movie_length_after_cuts:
  ∀ (OrigLen Cut1 Cut2 Cut3 : ℝ), 
    OrigLen = 97 → 
    Cut1 = 4.5 → 
    Cut2 = 2.75 → 
    Cut3 = 6.25 → 
    (OrigLen - (Cut1 + Cut2 + Cut3) = 83.5) :=
by
  intros OrigLen Cut1 Cut2 Cut3 hOrigLen hCut1 hCut2 hCut3
  rw [hOrigLen, hCut1, hCut2, hCut3]
  norm_num
  sorry

end movie_length_after_cuts_l43_43915


namespace bottles_produced_l43_43249

def machine_rate (total_machines : ℕ) (total_bottles_per_minute : ℕ) : ℕ :=
  total_bottles_per_minute / total_machines

def total_bottles (total_machines : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  total_machines * bottles_per_minute * minutes

theorem bottles_produced (machines1 machines2 minutes : ℕ) (bottles1 : ℕ) :
  machine_rate machines1 bottles1 = bottles1 / machines1 →
  total_bottles machines2 (bottles1 / machines1) minutes = 2160 :=
by
  intros machine_rate_eq
  sorry

end bottles_produced_l43_43249


namespace gcd_765432_654321_l43_43178

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43178


namespace alice_sugar_fill_l43_43302

theorem alice_sugar_fill (sugar_needed : ℚ) (cup_capacity : ℚ) (num_fills : ℕ) :
  sugar_needed = 3 + 3 / 4 ∧ cup_capacity = 1 / 3 ∧ num_fills = 12 → 
  (sugar_needed / cup_capacity).ceil = num_fills :=
begin
  intros h,
  cases h with h_sugar h_rest,
  cases h_rest with h_capacity h_fills,
  rw [h_sugar, h_capacity],
  have : 3 + 3 / 4 = 15 / 4 := by norm_num,
  have : 1 / 3 = (1 : ℚ) / 3 := rfl,
  rw this,
  norm_cast,
  have : (15 / 4 : ℚ) / (1 / 3 : ℚ) = 45 / 4 := by field_simp,
  rw this,
  norm_cast,
  exact eq.symm (show (45 / 4).ceil = 12, by norm_num),
end

end alice_sugar_fill_l43_43302


namespace four_racers_meet_l43_43785

/-- In a circular auto race, four racers participate. Their cars start simultaneously from 
the same point and move at constant speeds, and for any three cars, there is a moment 
when they meet. Prove that after the start of the race, there will be a moment when all 
four cars meet. (Assume the race continues indefinitely in time.) -/
theorem four_racers_meet (V1 V2 V3 V4 : ℝ) (L : ℝ) (t : ℝ) 
  (h1 : 0 ≤ t) 
  (h2 : V1 ≤ V2 ∧ V2 ≤ V3 ∧ V3 ≤ V4)
  (h3 : ∀ t1 t2 t3, ∃ t, t1 * V1 = t ∧ t2 * V2 = t ∧ t3 * V3 = t) :
  ∃ t, t > 0 ∧ ∃ t', V1 * t' % L = 0 ∧ V2 * t' % L = 0 ∧ V3 * t' % L = 0 ∧ V4 * t' % L = 0 :=
sorry

end four_racers_meet_l43_43785


namespace kyler_wins_three_l43_43081

noncomputable def kyler_wins (peter_wins peter_losses : ℕ) (emma_wins emma_losses : ℕ) (kyler_games kyler_losses : ℕ) :=
  let kyler_wins := kyler_games - kyler_losses in
  kyler_wins

theorem kyler_wins_three :
  ∀ (peter_wins peter_losses emma_wins emma_losses kyler_games kyler_losses : ℕ), 
  peter_wins = 5 → 
  peter_losses = 3 → 
  emma_wins = 4 → 
  emma_losses = 4 → 
  kyler_games = 5 → 
  kyler_losses = 2 → 
  kyler_wins peter_wins peter_losses emma_wins emma_losses kyler_games kyler_losses = 3 :=
by
  intros peter_wins peter_losses emma_wins emma_losses kyler_games kyler_losses hpw hpl hew hel hkg hkl
  simp [kyler_wins]
  rw [hpw, hpl, hew, hel, hkg, hkl]
  simp [kyler_wins]
  exact rfl

end kyler_wins_three_l43_43081


namespace PedoeInequalityHolds_l43_43821

noncomputable def PedoeInequality 
  (a b c a1 b1 c1 : ℝ) (Δ Δ1 : ℝ) :
  Prop :=
  a^2 * (b1^2 + c1^2 - a1^2) + 
  b^2 * (c1^2 + a1^2 - b1^2) + 
  c^2 * (a1^2 + b1^2 - c1^2) >= 16 * Δ * Δ1 

axiom areas_triangle 
  (a b c : ℝ) : ℝ 

axiom areas_triangle1 
  (a1 b1 c1 : ℝ) : ℝ 

theorem PedoeInequalityHolds 
  (a b c a1 b1 c1 : ℝ) 
  (Δ := areas_triangle a b c) 
  (Δ1 := areas_triangle1 a1 b1 c1) :
  PedoeInequality a b c a1 b1 c1 Δ Δ1 :=
sorry

end PedoeInequalityHolds_l43_43821


namespace sum_of_triangle_areas_in_cube_l43_43897

theorem sum_of_triangle_areas_in_cube :
  let m : ℤ := 48,
      n : ℤ := 4608,
      p : ℤ := 3072
  in m + n + p = 7728 :=
by
  sorry

end sum_of_triangle_areas_in_cube_l43_43897


namespace largest_fraction_is_D_l43_43929

-- Define the fractions as Lean variables
def A : ℚ := 2 / 6
def B : ℚ := 3 / 8
def C : ℚ := 4 / 12
def D : ℚ := 7 / 16
def E : ℚ := 9 / 24

-- Define a theorem to prove the largest fraction is D
theorem largest_fraction_is_D : max (max (max A B) (max C D)) E = D :=
by
  sorry

end largest_fraction_is_D_l43_43929


namespace baker_initial_cakes_l43_43989

theorem baker_initial_cakes (sold : ℕ) (left : ℕ) (initial : ℕ) 
  (h_sold : sold = 41) (h_left : left = 13) : 
  sold + left = initial → initial = 54 :=
by
  intros
  exact sorry

end baker_initial_cakes_l43_43989


namespace John_and_Rose_work_together_l43_43487

theorem John_and_Rose_work_together (John_work_days : ℕ) (Rose_work_days : ℕ) (combined_work_days: ℕ) 
  (hJohn : John_work_days = 10) (hRose : Rose_work_days = 40) :
  combined_work_days = 8 :=
by 
  sorry

end John_and_Rose_work_together_l43_43487


namespace sum_of_triangle_areas_in_cube_l43_43898

theorem sum_of_triangle_areas_in_cube :
  let m : ℤ := 48,
      n : ℤ := 4608,
      p : ℤ := 3072
  in m + n + p = 7728 :=
by
  sorry

end sum_of_triangle_areas_in_cube_l43_43898


namespace restore_original_message_l43_43958

open function

namespace cryptography

/-- Helper function to reorder intercepted segments according to transmission rules -/
def reorder (seg : String) : String :=
    let indices := [3, 7, 11, 2, 6, 10, 1, 5, 9, 0, 4, 8]
    String.mk (indices.map (seg.get! ∘ Nat.toFin.∘ (· % seg.length)))

/-- The original message restoration problem -/
theorem restore_original_message (intercepted_segs : List String) :
  intercepted_segs = [
    "СО-ГЖТПНБЛЖО",
    "РСТКДКСПХЕУБ",
    "-Е-ПФПУБ-ЮОБ",
    "СП-ЕОКЖУУЛЖЛ",
    "СМЦХБЭКГОЩПЫ",
    "УЛКЛ-ИКНТЛЖГ"
  ] → 
  (exists (original : String), 
    original = "СОВРЕМЕННАЯ КРИПТОГРАФИЯ ЭТО-НАУКА-О СЕКРЕТНОСТИ ШИФРОВАЛЬНЫХ СИСТЕМ-СВЯЗИ") :=
  sorry

end cryptography

end restore_original_message_l43_43958


namespace multiplicative_inverse_of_17_mod_43_l43_43316

theorem multiplicative_inverse_of_17_mod_43 : 
  ∃ b : ℤ, 0 ≤ b ∧ b < 43 ∧ 17 * b ≡ 1 [MOD 43] :=
begin
  use 6,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  { norm_num, }
end

end multiplicative_inverse_of_17_mod_43_l43_43316


namespace sine_ratio_product_eq_one_l43_43503

variables (A B C D Z : Type)
variables (α1 α2 β1 β2 γ1 γ2 δ1 δ2 : ℝ)
variables [convex_quadrilateral A B C D]
variables [inside_point Z A B C D]

-- Assumptions about the angles based on the problem conditions
def angle_ZAD := α1
def angle_ZAB := α2
def angle_ZBA := β1
def angle_ZBC := β2
def angle_ZCB := γ1
def angle_ZCD := γ2
def angle_ZDC := δ1
def angle_ZDA := δ2

theorem sine_ratio_product_eq_one :
  (sin α1) / (sin α2) * (sin β1) / (sin β2) * (sin γ1) / (sin γ2) * (sin δ1) / (sin δ2) = 1 :=
sorry

end sine_ratio_product_eq_one_l43_43503


namespace hall_ratio_l43_43135

open Real

theorem hall_ratio (w l : ℝ) (h_area : w * l = 288) (h_diff : l - w = 12) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l43_43135


namespace stratified_sampling_pines_l43_43623

def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

theorem stratified_sampling_pines :
  sample_size * pine_saplings / total_saplings = 20 := by
  sorry

end stratified_sampling_pines_l43_43623


namespace x_coordinate_D_l43_43519

noncomputable def find_x_coordinate_D (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ := 
  let l := -a * b
  let x := l / c
  x

theorem x_coordinate_D (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (D_on_parabola : d^2 = (a + b) * (d) + l)
  (lines_intersect_y_axis : ∃ l : ℝ, (a^2 = (b + a) * a + l) ∧ (b^2 = (b + a) * b + l) ∧ (c^2 = (d + c) * c + l)) :
  d = (a * b) / c :=
by sorry

end x_coordinate_D_l43_43519


namespace arrangement_count_l43_43307

def subjects : Type := { s : String // s ∈ ["Chinese", "Mathematics", "Physics", "History", "Foreign Language"] }

def valid_arrangements (arrangement : Fin 5 → subjects) : Prop :=
  ∃ i j : Fin 5, i < j ∧ arrangement i = ⟨"Mathematics", sorry⟩ ∧ arrangement j = ⟨"History", sorry⟩

theorem arrangement_count : (∃ arrangement : Fin 5 → subjects, valid_arrangements arrangement) =
  (24 : ℕ) := sorry

end arrangement_count_l43_43307


namespace train_passes_jogger_time_l43_43934

theorem train_passes_jogger_time (speed_jogger_kmph : ℝ) 
                                (speed_train_kmph : ℝ) 
                                (distance_ahead_m : ℝ) 
                                (length_train_m : ℝ) : 
  speed_jogger_kmph = 9 → 
  speed_train_kmph = 45 →
  distance_ahead_m = 250 →
  length_train_m = 120 →
  (distance_ahead_m + length_train_m) / (speed_train_kmph - speed_jogger_kmph) * (1000 / 3600) = 37 :=
by
  intros h1 h2 h3 h4
  sorry

end train_passes_jogger_time_l43_43934


namespace train_length_l43_43920

-- Definitions based on conditions
def speed_kmph : ℕ := 108  -- Speed of each train in km/hr
def time_sec : ℕ := 4      -- Time to cross each other in seconds
def speed_mps : ℕ := (speed_kmph * 1000) / 3600  -- Conversion to m/s
def relative_speed_mps : ℕ := 2 * speed_mps  -- Relative speed since they are moving in opposite directions

-- The theorem statement
theorem train_length (L : ℕ) (speed_kmph = 108) (time_sec = 4) : 
  2 * L = (2 * (speed_kmph * 1000) / 3600) * time_sec → L = 120 :=
by
  sorry

end train_length_l43_43920


namespace solve_for_X_l43_43439

variable (X Y : ℝ)

def diamond (X Y : ℝ) := 4 * X + 3 * Y + 7

theorem solve_for_X (h : diamond X 5 = 75) : X = 53 / 4 :=
by
  sorry

end solve_for_X_l43_43439


namespace total_flag_distance_moved_l43_43985

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l43_43985


namespace gcd_proof_l43_43197

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43197


namespace total_books_after_loss_l43_43533

-- Define variables for the problem
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

-- Prove the final number of books together
theorem total_books_after_loss : (sandy_books + tim_books - benny_lost_books) = 19 := by
  sorry

end total_books_after_loss_l43_43533


namespace gcd_765432_654321_l43_43162

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43162


namespace total_area_of_combined_shape_l43_43638

theorem total_area_of_combined_shape
  (length_rectangle : ℝ) (width_rectangle : ℝ) (side_square : ℝ)
  (h_length : length_rectangle = 0.45)
  (h_width : width_rectangle = 0.25)
  (h_side : side_square = 0.15) :
  (length_rectangle * width_rectangle + side_square * side_square) = 0.135 := 
by 
  sorry

end total_area_of_combined_shape_l43_43638


namespace no_extreme_values_l43_43119

-- Given function f(x) = x^3 + 3x^2 + 3x - a
def f (x a : ℝ) : ℝ := x^3 + 3*x^2 + 3*x - a

-- Prove that the function has 0 extreme values
theorem no_extreme_values (a : ℝ) : 
  ∀ x, (f x a)' = 3*(x + 1)^2 → ∀ x, ¬∃ y, (f y a)' = 0 :=
by sorry

end no_extreme_values_l43_43119


namespace f_sum_identity_l43_43828

def f (n : ℕ) : ℝ := 
  let root := n ^ (1/4 : ℝ)
  if root - ⌊root⌋ < 0.5 then ⌊root⌋ else ⌊root⌋ + 1

theorem f_sum_identity : ∑ k in finset.range 2018.succ, 1/(f k) = 2823 / 7 := 
by 
  sorry

end f_sum_identity_l43_43828


namespace min_m_n_sum_l43_43543

theorem min_m_n_sum (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 108 * m = n^3) : m + n = 8 :=
sorry

end min_m_n_sum_l43_43543


namespace arithmetic_sequence_sum_l43_43320

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (S : ℕ → ℤ),
  -- Condition: The sum of the first 11 terms is 88
  (S 11 = 88) →
  -- Definition of the sum of the first n terms of an arithmetic sequence
  (∀ n, S n = n * (a 1 + a n) / 2) →
  -- We want to prove that
  (a 3 + a 6 + a 9 = 24) :=
begin
  intros a S hS11 hSum,
  sorry
end

end arithmetic_sequence_sum_l43_43320


namespace sum_a1_a4_l43_43749

variables (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := n^2 + n + 1

-- Define the individual terms of the sequence
def term_seq (n : ℕ) : ℕ :=
if n = 1 then sum_seq 1 else sum_seq n - sum_seq (n - 1)

-- Prove that the sum of the first and fourth terms equals 11
theorem sum_a1_a4 : 
  (term_seq 1) + (term_seq 4) = 11 :=
by
  -- to be completed with proof steps
  sorry

end sum_a1_a4_l43_43749


namespace complete_remaining_parts_l43_43355

-- Define the main conditions and the proof goal in Lean 4
theorem complete_remaining_parts :
  ∀ (total_parts processed_parts workers days_off remaining_parts_per_day),
  total_parts = 735 →
  processed_parts = 135 →
  workers = 5 →
  days_off = 1 →
  remaining_parts_per_day = total_parts - processed_parts →
  (workers * 2 - days_off) * 15 = processed_parts →
  remaining_parts_per_day / (workers * 15) = 8 :=
by
  -- Starting the proof
  intros total_parts processed_parts workers days_off remaining_parts_per_day
  intros h_total_parts h_processed_parts h_workers h_days_off h_remaining_parts_per_day h_productivity
  -- Replace given variables with their values
  sorry

end complete_remaining_parts_l43_43355


namespace circle_covers_points_l43_43726

theorem circle_covers_points 
  (n : ℕ)
  (points : Fin n → ℝ × ℝ)
  (h1 : ∀ i j : Fin n, i ≠ j → dist (points i) (points j) < 1)
  (h2 : ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → is_acute_triangle (points i) (points j) (points k)) : 
  ∃ (c : ℝ × ℝ) (r : ℝ), r ≤ 0.5 ∧ ∀ i : Fin n, dist c (points i) ≤ r :=
sorry

def is_acute_triangle (a b c : ℝ × ℝ) : Prop := 
  ∀ angle in [angle_between a b c, angle_between b c a, angle_between c a b], angle < π / 2

def angle_between (a b c : ℝ × ℝ) : ℝ := 
  let u := (b.1 - a.1, b.2 - a.2)
  let v := (c.1 - a.1, c.2 - a.2)
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_u := Real.sqrt (u.1 * u.1 + u.2 * u.2)
  let norm_v := Real.sqrt (v.1 * v.1 + v.2 * v.2)
  Real.acos (dot / (norm_u * norm_v))

end circle_covers_points_l43_43726


namespace product_of_possible_values_of_b_l43_43324

def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def f_inv (b x : ℝ) : ℝ := (5 * b + 8) / 6

theorem product_of_possible_values_of_b :
  (∀ (b : ℝ), f b 3 = f_inv b (b + 2)) → (∏' (b : ℝ), b == 2) :=
by
  sorry

end product_of_possible_values_of_b_l43_43324


namespace find_x_coordinate_of_y3_l43_43802

theorem find_x_coordinate_of_y3 (a b : ℝ) (h1 : (↑-2, -3) : ℝ × ℝ)
    (h2 : ∃ x : ℝ, (x, 0) = (4, 0)) :
    (∃ x : ℝ, (x, 3) : ℝ × ℝ) → ∃ x : ℝ, x = 10 := by
  sorry

end find_x_coordinate_of_y3_l43_43802


namespace students_taking_only_science_l43_43981

open Finset

variables 
  (students : Finset ℕ) -- Represents the set of all students
  (science technology : Finset ℕ) -- Represents the sets of students taking science and technology classes respectively

-- The given conditions
variables 
  (h_total : students.card = 150)
  (h_science : science.card = 110)
  (h_technology : technology.card = 97)
  (h_union : (science ∪ technology).card = students.card)

-- The theorem to prove
theorem students_taking_only_science : (science \ technology).card = 53 :=
sorry

end students_taking_only_science_l43_43981


namespace math_problem_l43_43683

-- Definitions based on the given conditions
def a : ℝ := Real.sqrt 2
def b : ℝ := Real.sqrt 3
def c : ℝ := 1 / Real.sqrt 6

-- The statement to be proved
theorem math_problem : a * b / c = 6 :=
by
  sorry

end math_problem_l43_43683


namespace gcd_of_765432_and_654321_l43_43203

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43203


namespace number_of_nonempty_subsets_l43_43449

theorem number_of_nonempty_subsets (A : Set ℕ) (hA : A = {1, 2, 3}) : 
  (∑ B in 𝒫 A, ¬ B = ∅ ∧ B ⊆ A) = 7 := 
sorry

end number_of_nonempty_subsets_l43_43449


namespace find_certain_number_l43_43445

theorem find_certain_number (h1 : 2994 / 14.5 = 171) (h2 : ∃ x : ℝ, x / 1.45 = 17.1) : ∃ x : ℝ, x = 24.795 :=
by
  sorry

end find_certain_number_l43_43445


namespace net_change_in_price_l43_43008

theorem net_change_in_price (P : ℝ) : 
  let initial_price := P
  let price_after_discount := initial_price - 0.20 * initial_price
  let price_after_promotion := price_after_discount + 0.55 * price_after_discount
  let final_price := price_after_promotion + 0.12 * price_after_promotion
  let net_change := final_price - initial_price
  let percentage_change := (net_change / initial_price) * 100
  in percentage_change = 38.88 := by sorry

end net_change_in_price_l43_43008


namespace complex_magnitude_min_value_l43_43378

-- Definition of complex numbers and the magnitude function
open Complex

theorem complex_magnitude_min_value
  (u v : ℂ) 
  (h1 : abs (u + v) = 2) 
  (h2 : abs (u^2 + v^2) = 11) :
  abs (u^3 + v^3) ≥ 14.5 :=
by sorry

end complex_magnitude_min_value_l43_43378


namespace wallet_more_expensive_l43_43834

-- Define variables for the cost of wallet, shirt, and food
variables (W S F : ℕ)

-- Define the conditions as hypotheses
hypothesis (h1 : S = W / 3)
hypothesis (h2 : W > F)
hypothesis (h3 : F = 30)
hypothesis (h4 : S + W + F = 150)

-- The theorem to prove the difference in the cost of wallet and food
theorem wallet_more_expensive : W - F = 60 :=
sorry

end wallet_more_expensive_l43_43834


namespace find_alpha_l43_43742

theorem find_alpha 
  (α : ℝ) 
  (hα : α ∈ set.Ioo (Real.pi / 2) Real.pi) 
  (h_max : ∃ x : ℝ, (Real.sin α)^(x^2 - 2*x + 3) = 1 / 4) : 
  α = 5 * Real.pi / 6 :=
sorry

end find_alpha_l43_43742


namespace value_of_b_pos_sum_for_all_x_l43_43455

noncomputable def f (b : ℝ) (x : ℝ) := 3 * x^2 - 2 * x + b
noncomputable def g (b : ℝ) (x : ℝ) := x^2 + b * x - 1
noncomputable def sum_f_g (b : ℝ) (x : ℝ) := f b x + g b x

theorem value_of_b (b : ℝ) (h : ∀ x : ℝ, (sum_f_g b x = 4 * x^2 + (b - 2) * x + (b - 1))) :
  b = 2 := 
sorry

theorem pos_sum_for_all_x :
  ∀ x : ℝ, 4 * x^2 + 1 > 0 := 
sorry

end value_of_b_pos_sum_for_all_x_l43_43455


namespace arithmetic_sequence_general_formula_sum_of_bn_first_n_terms_l43_43402

theorem arithmetic_sequence_general_formula 
  (a : ℕ → ℕ) (a₁ d : ℕ) 
  (h1 : ∑ i in range 5, (a₁ + i * d) = 55) 
  (h2 : a₁ + 5 * d + a₁ + 6 * d = 36) : 
  ∀ n, a n = 2 * n + 5 :=
sorry

theorem sum_of_bn_first_n_terms 
  (b : ℕ → ℝ) (a : ℕ → ℕ) 
  (a₁ d : ℕ) 
  (h1 : ∑ i in range 5, (a₁ + i * d) = 55) 
  (h2 : a₁ + 5 * d + a₁ + 6 * d = 36) 
  (h3 : ∀ n, a n = 2 * n + 5) 
  (h4 : ∀ n, b n = 1 / ((a n - 6) * (a n - 4))) :
  ∀ n, (∑ i in range n, b i) = n / (2 * n + 1) := 
sorry

end arithmetic_sequence_general_formula_sum_of_bn_first_n_terms_l43_43402


namespace value_of_b_l43_43404

theorem value_of_b {b : ℝ} 
  (h: (realPart (\frac{2 - complex.I * b}{1 + 2 * complex.I}) = -(imaginaryPart (\frac{2 - complex.I * b}{1 + 2 * complex.I})))) : 
  b = -(2 / 3) :=
sorry

end value_of_b_l43_43404


namespace find_opposite_pair_l43_43972

def P1 := -(-1)
def P2 := (-1)^2
def P3 := | -1 |
def P4 := -(1^2)

def Opposite (a b : Int): Prop := a = -b

theorem find_opposite_pair: 
  (¬ Opposite P1 1) ∧ 
  (¬ Opposite P2 1) ∧
  (¬ Opposite P3 1) ∧
  (Opposite P4 1) := by
  sorry

end find_opposite_pair_l43_43972


namespace altitudes_through_O_l43_43938

theorem altitudes_through_O {A B C O A₁ B₁ C₁ : Type*}
  (circumcenter_ABC : is_circumcenter O ABC)
  (symmetric_A₁ : symmetric A₁ O BC)
  (symmetric_B₁ : symmetric B₁ O CA)
  (symmetric_C₁ : symmetric C₁ O AB) :
  are_altitudes_through (triangle A₁ B₁ C₁) O := by
  sorry

end altitudes_through_O_l43_43938


namespace fixed_point_l43_43558

def f (a x : ℝ) := log a (4 * x - 3)

theorem fixed_point (a : ℝ) (h_a : 1 < a) : f a 1 = 0 :=
by
  unfold f
  sorry

end fixed_point_l43_43558


namespace angela_is_taller_l43_43671

theorem angela_is_taller (Amy_height : ℕ) (h_Amy : Amy_height = 150)
  (h_Helen : ∀ Helen_height : ℕ, Helen_height = Amy_height + 3) 
  (Angela_height : ℕ) (h_Angela : Angela_height = 157) :
  ∃ (diff : ℕ), diff = Angela_height - (Amy_height + 3) ∧ diff = 4 :=
by
  have Helen_height := Amy_height + 3
  have h_Helen_height : Helen_height = 150 + 3 := by rw [h_Amy]; rfl
  have h_diff : 157 - 153 = 4 := rfl
  use 4
  rw [h_diff]
  exact ⟨rfl, rfl⟩

end angela_is_taller_l43_43671


namespace trapezoid_area_l43_43840

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem trapezoid_area :
  let A := (0, 0) 
  let B := (1, 2)
  let C := (3, 2)
  let D := (4, 0)
  ∃ (area : ℝ), 
  area = 6 ∧ 
  distance B C = 2 ∧ 
  distance A D = 4 ∧ 
  distance A B = 2 := 
begin
  intro A,
  intro B,
  intro C,
  intro D,
  existsi (6 : ℝ),
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end trapezoid_area_l43_43840


namespace student_total_cost_l43_43723

-- Definitions of cost constants
def pants_cost : ℝ := 20
def shirt_cost : ℝ := 2 * pants_cost
def tie_cost : ℝ := shirt_cost / 5
def socks_cost : ℝ := 3
def jacket_cost : ℝ := 3 * shirt_cost
def shoes_cost : ℝ := 40

-- Definition of total cost of one full uniform without discount
def uniform_cost : ℝ := pants_cost + shirt_cost + tie_cost + socks_cost + jacket_cost + shoes_cost

-- Number of uniforms to purchase
def num_uniforms : ℕ := 5

-- Discount definition (10% for 3-5 uniforms)
def discount_rate : ℝ := 0.1

-- Total cost of uniforms with discount
def discounted_uniform_cost : ℝ := uniform_cost * (1 - discount_rate)
def total_discounted_uniform_cost : ℝ := discounted_uniform_cost * num_uniforms

-- Additional fee and tax rates
def uniform_fee : ℝ := 15
def sales_tax_rate : ℝ := 0.06

-- Calculation of final cost with fee and tax
def subtotal : ℝ := total_discounted_uniform_cost + uniform_fee
def tax : ℝ := subtotal * sales_tax_rate
def final_cost : ℝ := subtotal + tax

-- The theorem that encapsulates the correct answer
theorem student_total_cost : final_cost = 1117.77 := by
  -- Define all the intermediate values
  let _pants_cost : ℝ := 20
  let _shirt_cost := 2 * _pants_cost
  let _tie_cost := _shirt_cost / 5
  let _socks_cost := 3
  let _jacket_cost := 3 * _shirt_cost
  let _shoes_cost := 40

  let _uniform_cost := _pants_cost + _shirt_cost + _tie_cost + _socks_cost + _jacket_cost + _shoes_cost

  let _num_uniforms := 5
  let _discount_rate := 0.1

  let _discounted_uniform_cost := _uniform_cost * (1 - _discount_rate)
  let _total_discounted_uniform_cost := _discounted_uniform_cost * _num_uniforms

  let _uniform_fee := 15
  let _sales_tax_rate := 0.06

  let _subtotal := _total_discounted_uniform_cost + _uniform_fee
  let _tax := _subtotal * _sales_tax_rate
  let _final_cost := _subtotal + _tax
  
  show _final_cost = 1117.77 from
    sorry -- Proof steps go here

end student_total_cost_l43_43723


namespace nellie_legos_l43_43073

theorem nellie_legos :
  let initial_legos := 380
  let lost_legos := 0.15 * initial_legos
  let remaining_after_loss := initial_legos - lost_legos
  let legos_given_to_sister := floor (1/8 * remaining_after_loss)
  let final_legos := remaining_after_loss - legos_given_to_sister
  final_legos = 283 :=
by
  sorry

end nellie_legos_l43_43073


namespace inradii_comparison_l43_43845

theorem inradii_comparison {ABC A'B'C' : Triangle} (h_inABC_inA'B'C' : inside ABC A'B'C') :
  inradius ABC < inradius A'B'C' :=
sorry

end inradii_comparison_l43_43845


namespace even_perfect_square_factors_l43_43436

theorem even_perfect_square_factors :
  let factors := 2^6 * 5^4 * 7^3
  ∃ (count : ℕ), count = (3 * 3 * 2) ∧
  ∀ (a b c : ℕ), (0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ c ∧ c ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 
  a % 2 = 0 ∧ 2 ≤ a ∧ c % 2 = 0 ∧ b % 2 = 0) → 
  a * b * c < count :=
by
  sorry

end even_perfect_square_factors_l43_43436


namespace goods_train_speed_l43_43601

theorem goods_train_speed:
  let speed_mans_train := 100   -- in km/h
  let length_goods_train := 280 -- in meters
  let passing_time := 9         -- in seconds
  ∃ speed_goods_train: ℝ, 
  (speed_mans_train + speed_goods_train) * (5 / 18) * passing_time = length_goods_train ↔ speed_goods_train = 12 :=
by
  sorry

end goods_train_speed_l43_43601


namespace prove_weight_of_a_l43_43251

noncomputable def weight_proof (A B C D : ℝ) : Prop :=
  (A + B + C) / 3 = 60 ∧
  50 ≤ A ∧ A ≤ 80 ∧
  50 ≤ B ∧ B ≤ 80 ∧
  50 ≤ C ∧ C ≤ 80 ∧
  60 ≤ D ∧ D ≤ 90 ∧
  (A + B + C + D) / 4 = 65 ∧
  70 ≤ D + 3 ∧ D + 3 ≤ 100 ∧
  (B + C + D + (D + 3)) / 4 = 64 → 
  A = 87

-- Adding a theorem statement to make it clear we need to prove this.
theorem prove_weight_of_a (A B C D : ℝ) : weight_proof A B C D :=
sorry

end prove_weight_of_a_l43_43251


namespace Dan_tshirts_total_l43_43674

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end Dan_tshirts_total_l43_43674


namespace solve_trig_eq_l43_43536

theorem solve_trig_eq 
  (x : ℝ) (k : ℤ) 
  (h₀ : sin (3 * x) = 3 * sin x - 4 * (sin x)^3)
  (h₁ : cos (2 * x) = 1 - 2 * (sin x)^2) 
  : (|sin x| - sin (3 * x)) / (cos x * cos (2 * x)) = 2 * sqrt 3 
  -> x = 2 * π / 3 + 2 * k * π 
  \/ x = -2 * π / 3 + 2 * k * π 
  \/ x = -π / 6 + 2 * k * π := sorry

end solve_trig_eq_l43_43536


namespace find_frac_sin_cos_l43_43744

theorem find_frac_sin_cos (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin (3 * Real.pi / 2 + α)) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 :=
by
  sorry

end find_frac_sin_cos_l43_43744


namespace prime_squared_plus_seventeen_mod_12_l43_43935

theorem prime_squared_plus_seventeen_mod_12 (p : ℕ) (prime_p : p.prime) (p_gt_3 : p > 3) : 
  (p ^ 2 + 17) % 12 = 6 :=
by sorry

end prime_squared_plus_seventeen_mod_12_l43_43935


namespace digging_foundation_l43_43483

def men1 := 15
def days1 := 4
def men2 := 25
def total_work : ℕ := men1 * days1

theorem digging_foundation :
  total_work = 60 →
  (∃ d : ℝ, men2 * d = total_work ∧ d = 2.4) :=
by
  intro h
  use 60 / (men2 : ℝ)
  split
  · rw [h, mul_div_cancel']
    norm_num
    exact ne_of_gt (by norm_num : (25 : ℝ) > 0)
  · norm_num
    exact h
#align digging_foundation

end digging_foundation_l43_43483


namespace house_construction_days_l43_43288

theorem house_construction_days
  (D : ℕ) -- number of planned days to build the house
  (Hwork_done : 1000 + 200 * (D - 10) = 100 * (D + 90)) : 
  D = 110 :=
sorry

end house_construction_days_l43_43288


namespace gcd_765432_654321_l43_43187

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43187


namespace customers_at_each_table_l43_43654

theorem customers_at_each_table (initial_customers left_customers tables : ℕ) (h1 : initial_customers = 44)
                                    (h2 : left_customers = 12) (h3 : tables = 4) :
  (initial_customers - left_customers) / tables = 8 :=
by
  rw [h1, h2, h3]
  sorry

end customers_at_each_table_l43_43654


namespace gcd_765432_654321_eq_3_l43_43224

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43224


namespace bun_distribution_invariance_l43_43908

-- Define the conditions and the question as a theorem statement.
theorem bun_distribution_invariance :
  ∀ (initial_distribution : Fin 25 → ℕ),
  (∃ i, initial_distribution i ≠ initial_distribution 0) →
  ¬ (∃ final_distribution : Fin 25 → ℕ,
    (∀ j, final_distribution j = 2) ∧
    (∀ k, initial_distribution k = 2) :=
begin
  intro initial_distribution,
  intro not_all_equal_initial,
  intro h,
  sorry
end

end bun_distribution_invariance_l43_43908


namespace polar_line_intersection_l43_43469

open Real

theorem polar_line_intersection
  (M : ℝ × ℝ)
  (eq_line1 : ∃ ρ θ, θ = 2 * π / 3 ∧ ρ * sin (θ + π / 6) = 1 ∧ M = (ρ * cos θ, ρ * sin θ)) 
  (x_eq : M.1 = -1) (y_eq : M.2 = sqrt 3)
  (circle_eq : ∀ t, let x := -1 + 1 / 2 * t in
                       let y := sqrt 3 + sqrt 3 / 2 * t in
                       x^2 + y^2 = 7)
  : (∃ (A B : ℝ × ℝ), ((A.1 = -1 + 1 / 2 * (-3) ∧ A.2 = sqrt 3 + sqrt 3 / 2 * (-3)) ∧ 
                        (B.1 = -1 + 1 / 2 * 1 ∧ B.2 = sqrt 3 + sqrt 3 / 2 * 1)) ∧ 
                        (1 / dist M A + 1 / dist M B = 4 / 3))
:=
sorry

end polar_line_intersection_l43_43469


namespace pairs_of_shoes_l43_43041

theorem pairs_of_shoes (n : ℕ) (h1 : 2 * n = 18) (h2 : 1 / (2 * n - 1) = 0.058823529411764705) : n = 9 := 
sorry

end pairs_of_shoes_l43_43041


namespace arithmetic_sequence_k_l43_43789

theorem arithmetic_sequence_k (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (ha : ∀ n, S (n + 1) = S n + a (n + 1))
  (hS3_S8 : S 3 = S 8) 
  (hS7_Sk : ∃ k, S 7 = S k)
  : ∃ k, k = 4 :=
by
  sorry

end arithmetic_sequence_k_l43_43789


namespace better_sequence_is_BAB_l43_43581

def loss_prob_andrei : ℝ := 0.4
def loss_prob_boris : ℝ := 0.3

def win_prob_andrei : ℝ := 1 - loss_prob_andrei
def win_prob_boris : ℝ := 1 - loss_prob_boris

def prob_qualify_ABA : ℝ :=
  win_prob_andrei * loss_prob_boris * win_prob_andrei +
  win_prob_andrei * win_prob_boris +
  loss_prob_andrei * win_prob_boris * win_prob_andrei

def prob_qualify_BAB : ℝ :=
  win_prob_boris * loss_prob_andrei * win_prob_boris +
  win_prob_boris * win_prob_andrei +
  loss_prob_boris * win_prob_andrei * win_prob_boris

theorem better_sequence_is_BAB : prob_qualify_BAB = 0.742 ∧ prob_qualify_BAB > prob_qualify_ABA :=
by 
  sorry

end better_sequence_is_BAB_l43_43581


namespace probability_hitting_exactly_2_times_probability_hitting_3_times_in_a_row_l43_43268

def archer_prob_hit_each_shot : ℝ := 2 / 3
def archer_prob_miss_each_shot : ℝ := 1 - archer_prob_hit_each_shot
def total_shots : ℕ := 5

theorem probability_hitting_exactly_2_times :
    let X := binomial_prob total_shots archer_prob_hit_each_shot 2
    X = 40 / 243 := by
  sorry

theorem probability_hitting_3_times_in_a_row :
    let A := (archer_prob_hit_each_shot ^ 3) * (archer_prob_miss_each_shot ^ 2)
    let P := 3 * A
    P = 8 / 81 := by
  sorry

end probability_hitting_exactly_2_times_probability_hitting_3_times_in_a_row_l43_43268


namespace smallest_B_l43_43136

theorem smallest_B :
  ∃ B : ℤ, ∀ n : ℤ, 4 ≤ n ∧ n < B ^ (1/3 : ℝ) ↔ (4 ≤ n ∧ n < 17) ∧ B = 4097 := by
  sorry

end smallest_B_l43_43136


namespace parabola_tangent_line_l43_43328

theorem parabola_tangent_line (a b : ℝ) :
  (∀ x : ℝ, ax^2 + bx + 2 = 2x + 3 → 0) → a = -((b - 2)^2) / 4 :=
by sorry

end parabola_tangent_line_l43_43328


namespace correct_meiosis_sequence_l43_43330

-- Define the events as types
inductive Event : Type
| Replication : Event
| Synapsis : Event
| Separation : Event
| Division : Event

-- Define options as lists of events
def option_A := [Event.Replication, Event.Synapsis, Event.Separation, Event.Division]
def option_B := [Event.Synapsis, Event.Replication, Event.Separation, Event.Division]
def option_C := [Event.Synapsis, Event.Replication, Event.Division, Event.Separation]
def option_D := [Event.Replication, Event.Separation, Event.Synapsis, Event.Division]

-- Define the theorem to be proved
theorem correct_meiosis_sequence : option_A = [Event.Replication, Event.Synapsis, Event.Separation, Event.Division] :=
by
  sorry

end correct_meiosis_sequence_l43_43330


namespace inequalities_hold_if_ac_lt_0_l43_43441

theorem inequalities_hold_if_ac_lt_0 
  {a c : ℝ} (h : a * c < 0) : 
  (∃ n, n = 3 ∧ 
         (∃ h1 : a / c < 0, h1) ∧ 
         (∃ h4 : c^3 * a < 0, h4) ∧ 
         (∃ h5 : c * a^3 < 0, h5)) := 
sorry

end inequalities_hold_if_ac_lt_0_l43_43441


namespace part1_real_point_exists_part2_find_k_part3_find_t_l43_43836

definition is_real_point (f : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ a, P = (a, a + 2) ∧ f a = a + 2

theorem part1_real_point_exists :
  ∃ P, is_real_point (λ x, (1 / 3) * x + 4) P ∧ P = (3, 5) :=
by
  sorry

theorem part2_find_k (k : ℝ) :
  (∃ x1 x2, is_real_point (λ x, x ^ 2 + 3 * x + 2 - k) (x1, x1 + 2) ∧ is_real_point (λ x, x ^ 2 + 3 * x + 2 - k) (x2, x2 + 2) ∧ dist (x1, x1 + 2) (x2, x2 + 2) = 2 * real.sqrt 2) →
  k = 0 :=
by
  sorry

theorem part3_find_t (t : ℝ) :
  (∃ a, is_real_point (λ x, (1 / 8) * x ^ 2 + (m - t + 1) * x + 2 * n + 2 * t - 2) (a, a + 2) ∧ (-2 ≤ m) ∧ (m ≤ 3) ∧ (n = t + 4)) →
  t = -1 :=
by
  sorry

end part1_real_point_exists_part2_find_k_part3_find_t_l43_43836


namespace fraction_always_defined_l43_43971

theorem fraction_always_defined (x : ℝ) : (x ^ 2 + 2) ≠ 0 :=
by {
  -- We can note that for all x in real numbers, x^2 + 2 is always positive.
  intro h,
  have : x^2 + 2 > 0 := by {
    apply add_pos_of_nonneg_of_pos (pow_two_nonneg x) zero_lt_two,
  },
  linarith,
}

end fraction_always_defined_l43_43971


namespace probability_Alex_Zhu_same_section_l43_43575

theorem probability_Alex_Zhu_same_section :
  (100.choose 60) > 0 → 
  (98.choose 58) > 0 → 
  (58.choose 18) > 0 → 
  (40.choose 20) > 0 → 
  (60.choose 20) > 0 → 
  (3 * (58.choose 18) / (60.choose 20) = (19 / 165)) := 
by
  intros h1 h2 h3 h4 h5
  sorry

end probability_Alex_Zhu_same_section_l43_43575


namespace final_salt_concentration_is_25_l43_43611

-- Define the initial conditions
def original_solution_weight : ℝ := 100
def original_salt_concentration : ℝ := 0.10
def added_salt_weight : ℝ := 20

-- Define the amount of salt in the original solution
def original_salt_weight := original_solution_weight * original_salt_concentration

-- Define the total amount of salt after adding pure salt
def total_salt_weight := original_salt_weight + added_salt_weight

-- Define the total weight of the new solution
def new_solution_weight := original_solution_weight + added_salt_weight

-- Define the final salt concentration
noncomputable def final_salt_concentration := (total_salt_weight / new_solution_weight) * 100

-- Prove the final salt concentration equals 25%
theorem final_salt_concentration_is_25 : final_salt_concentration = 25 :=
by
  sorry

end final_salt_concentration_is_25_l43_43611


namespace rhombus_area_correct_l43_43529

def is_rhombus (E F G H : Type) [metric_space E]
  (EF : ℝ) (FG : ℝ) (GH : ℝ) (HE : ℝ) : Prop :=
metric.dist E F = EF ∧ metric.dist F G = FG ∧
metric.dist G H = GH ∧ metric.dist H E = HE ∧
EF = FG ∧ FG = GH ∧ GH = HE

def diagonal_length (E I G : Type) [metric_space E]
  (EG : ℝ) (EI IG : ℝ) : Prop :=
metric.dist E G = EG ∧ metric.dist E I = EI ∧
metric.dist I G = IG ∧ EI + IG = EG

def rhombus_area (EG FH : ℝ) : ℝ :=
(EG * FH) / 2

theorem rhombus_area_correct (E F G H I : Type) [metric_space E]
  (EFmet : ℝ) (EG FH : ℝ)
  (h1 : is_rhombus E F G H EFmet EFmet EFmet EFmet)
  (h2 : diagonal_length E I G EG (EG / 2) (EG / 2))
  (HF : metric.dist F I = √(EFmet^2 - (EG / 2)^2))
  (FH := 2 * HF) :
  rhombus_area EG FH = 96 :=
by
  sorry

end rhombus_area_correct_l43_43529


namespace angle_PQR_degree_l43_43818

noncomputable def P : ℝ × ℝ × ℝ := (-3, 1, 5)
noncomputable def Q : ℝ × ℝ × ℝ := (-4, 0, 2)
noncomputable def R : ℝ × ℝ × ℝ := (-5, 1, 3)

def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

noncomputable def angle_PQR : ℝ :=
  Real.arccos ((distance P Q ^ 2 + distance Q R ^ 2 - distance P R ^ 2) / (2 * distance P Q * distance Q R))

theorem angle_PQR_degree :
  angle_PQR * (180 / Real.pi) ≈ 69.74 := sorry

end angle_PQR_degree_l43_43818


namespace trig_identity_simplify_l43_43852

theorem trig_identity_simplify :
  (sin 10 * (π / 180) + sin 20 * (π / 180) * cos 30 * (π / 180)) /
  (cos 10 * (π / 180) - sin 20 * (π / 180) * sin 30 * (π / 180)) = 
  sqrt 3 / 3 :=
by 
  sorry

end trig_identity_simplify_l43_43852


namespace cookies_per_person_l43_43488

theorem cookies_per_person
  (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ)
  (hA : A = 50) (hB : B = 10) (hC : C = 8) (hD : D = 16) :
  (A - (B + C)) / D = 2 :=
by
  rw [hA, hB, hC, hD]
  sorry

end cookies_per_person_l43_43488


namespace find_a_l43_43755

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3*x^2 + a*x + 2

theorem find_a :
  ∃ a : ℝ, (f a 0 = 2) ∧ 
           (let slope := 3*(0:ℝ)^2 - 6*(0:ℝ) + a in
            let tangent_line := λ (x : ℝ), slope*x + 2 in
            tangent_line (-2) = 0) ∧ 
           a = 1 :=
by 
  sorry

end find_a_l43_43755


namespace arithmetic_sequence_general_term_geometric_sequence_sum_l43_43571

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (n : ℤ) (h10 : a 10 = 30) (h20 : a 20 = 50) :
  a n = 2 * n + 10 :=
by
  sorry

theorem geometric_sequence_sum (a b : ℕ → ℕ) (T : ℕ → ℤ) (n : ℕ) 
  (h_a : ∀ n, a n = 2 * n + 10) 
  (h_b : ∀ n, b n = 2^(a n - 10)) 
  (h_T : T n = (4 * (4^n - 1)) / 3) :
  T n = ∑ i in range n, b i :=
by
  sorry

end arithmetic_sequence_general_term_geometric_sequence_sum_l43_43571


namespace seashells_given_joan_to_mike_l43_43809

-- Declaring the context for the problem: Joan's seashells
def initial_seashells := 79
def remaining_seashells := 16

-- Proving how many seashells Joan gave to Mike
theorem seashells_given_joan_to_mike : (initial_seashells - remaining_seashells) = 63 :=
by
  -- This proof needs to be completed
  sorry

end seashells_given_joan_to_mike_l43_43809


namespace stickers_distribution_l43_43435

noncomputable def numWaysToDistributeStickers (stickers sheets : ℕ) : ℕ :=
  Nat.choose (stickers - 1) (sheets - 1)

theorem stickers_distribution :
  numWaysToDistributeStickers 10 5 = 126 :=
by
  -- We use the stars and bars theorem here
  have h1 : numWaysToDistributeStickers 10 5 = Nat.choose (10 - 1) (5 - 1),
  { simp [numWaysToDistributeStickers] },
  rw h1,
  -- Calculation of binomial coefficients through natural number arithmetic
  have h2 : Nat.choose 9 4 = 126,
  { calc
      Nat.choose 9 4 = 9 * 8 * 7 * 6 / (4 * 3 * 2 * 1) : by sorry
      ... = 126 : by sorry 
  },
  exact h2

end stickers_distribution_l43_43435


namespace solve_inequality_l43_43607

def smallest_int_not_less_than (x : ℝ) : ℤ := ⌈x⌉
def largest_int_not_greater_than (x : ℝ) : ℤ := ⌊x⌋

theorem solve_inequality (x : ℝ) :
  ((smallest_int_not_less_than x)^2 + 4 (largest_int_not_greater_than x + 1) + 4 = 0) →
  -3 < x ∧ x < -2 :=
by sorry

end solve_inequality_l43_43607


namespace gcd_765432_654321_l43_43207

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43207


namespace min_cost_227_students_l43_43964

noncomputable def cost_notebook (n : ℕ) : ℝ :=
  if n % 12 = 0 then
    if n / 12 > 10 then 2.70 * (n / 12) else 3.00 * (n / 12)
  else
    if n > 10 * 12 then 2.70 * (n / 12) + 0.30 * (n % 12) else 3.00 * (n / 12) + 0.30 * (n % 12)

def min_cost (students : ℕ) : ℝ :=
  cost_notebook students

theorem min_cost_227_students : min_cost 227 = 51.3 :=
  sorry

end min_cost_227_students_l43_43964


namespace circle_equation_l43_43353

theorem circle_equation
  (a : ℝ) (r : ℝ)
  (h1 : ∃ a : ℝ, y = 2 * a)
  (h2 : (0 - a)^2 + (-2 - 2 * a)^2 = r^2)
  (h3 : ∃ a : ℝ, ∃ r : ℝ, abs(a - 2 * a - 2) / sqrt(2) = r) :
  ∃ a : ℝ, ∃ r : ℝ,
  a = -2 / 3 ∧ r = 2 * sqrt 2 / 3 ∧
  ((x + 2/3)^2 + (y + 4/3)^2 = 8 / 9) :=
begin
  sorry
end

end circle_equation_l43_43353


namespace total_camels_l43_43910

theorem total_camels (x y : ℕ) (humps_eq : x + 2 * y = 23) (legs_eq : 4 * (x + y) = 60) : x + y = 15 :=
by
  sorry

end total_camels_l43_43910


namespace length_of_pond_l43_43562

theorem length_of_pond (W L A_pond: ℝ) (h1: L = 48) (h2: L = 2 * W) (h3: A_pond = (1/18) * L * W) : 
sqrt A_pond = 8 := by
  sorry

end length_of_pond_l43_43562


namespace points_equidistant_from_circle_and_tangents_l43_43273

noncomputable def circle_radius := 4
noncomputable def tangent_distance := 6

theorem points_equidistant_from_circle_and_tangents :
  ∃! (P : ℝ × ℝ), dist P (0, 0) = circle_radius ∧
                 dist P (0, tangent_distance) = tangent_distance - circle_radius ∧
                 dist P (0, -tangent_distance) = tangent_distance - circle_radius :=
by {
  sorry
}

end points_equidistant_from_circle_and_tangents_l43_43273


namespace pyramid_sphere_surface_area_ratio_l43_43475

theorem pyramid_sphere_surface_area_ratio
  (A B C D P : Point)
  (hABC : ABCD_is_square A B C D)
  (hPA_perp_ABCD : PA_perp_to_plane_ ABCD P A)
  (hPA : distance P A = 6)
  (hAB : distance A B = 8) :
  ratio_of_sphere_surface_areas P A B C D = 41 / 4 :=
by
  sorry

end pyramid_sphere_surface_area_ratio_l43_43475


namespace positive_difference_mean_median_l43_43574

noncomputable def vertical_drops : List ℚ := [250, 210, 180, 330, 290, 160]

def mean (l : List ℚ) : ℚ := l.sum / l.length

def median (l : List ℚ) : ℚ := 
  let sorted := l.qsort (· < ·)
  if sorted.length % 2 = 0 then
    let mid1 := sorted.get! (sorted.length / 2 - 1)
    let mid2 := sorted.get! (sorted.length / 2)
    (mid1 + mid2) / 2
  else
    sorted.get! (sorted.length / 2)

def positive_difference (a b : ℚ) : ℚ := abs (a - b)

theorem positive_difference_mean_median :
  positive_difference (mean vertical_drops) (median vertical_drops) = 20 / 3 :=
by
  sorry

end positive_difference_mean_median_l43_43574


namespace squares_similar_l43_43240

def similar (s1 s2 : Type) [HasShape s1] [HasShape s2] : Prop :=
  ∀ (a b : s1) (c d : s2), similar_shapes a b c d

structure Square (side_length : ℝ) :=
  (is_square : is_shape_square side_length)

theorem squares_similar (s1 s2 : Square) : similar s1 s2 := by
  sorry

end squares_similar_l43_43240


namespace arithmetic_progression_contains_sixth_power_l43_43667

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end arithmetic_progression_contains_sixth_power_l43_43667


namespace hydrochloric_acid_reacts_with_sodium_bicarbonate_l43_43351

def hydrochloric_acid : Type := ℝ
def sodium_bicarbonate : Type := ℝ
def sodium_chloride : Type := ℝ

constant one_mole_HCl : hydrochloric_acid := 1
constant one_mole_NaHCO3 : sodium_bicarbonate := 1

theorem hydrochloric_acid_reacts_with_sodium_bicarbonate :
  ∀ (HCl : hydrochloric_acid) (NaHCO3 : sodium_bicarbonate), 
  HCl = 1 → NaHCO3 = 1 → sodium_chloride = 1 :=
by intros hcl nahco3 hcl_eq nahco3_eq
   sorry

end hydrochloric_acid_reacts_with_sodium_bicarbonate_l43_43351


namespace max_value_x_minus_2y_l43_43452

open Real

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 
  x - 2*y ≤ 10 :=
sorry

end max_value_x_minus_2y_l43_43452


namespace total_cost_cardshop_l43_43546

theorem total_cost_cardshop : 
  let price_A := 1.25
  let price_B := 1.50
  let price_C := 2.25
  let price_D := 2.50
  let discount_10_percent := 0.10
  let discount_15_percent := 0.15
  let sales_tax_rate := 0.06
  let qty_A := 6
  let qty_B := 4
  let qty_C := 10
  let qty_D := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  total_cost = 60.82
:= 
by
  have price_A : ℝ := 1.25
  have price_B : ℝ := 1.50
  have price_C : ℝ := 2.25
  have price_D : ℝ := 2.50
  have discount_10_percent : ℝ := 0.10
  have discount_15_percent : ℝ := 0.15
  have sales_tax_rate : ℝ := 0.06
  have qty_A : ℕ := 6
  have qty_B : ℕ := 4
  have qty_C : ℕ := 10
  have qty_D : ℕ := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  sorry

end total_cost_cardshop_l43_43546


namespace complex_conjugate_of_square_l43_43405

-- Define the complex number z
def z : ℂ := 2 + Complex.i

-- Define the main theorem stating the required equality
theorem complex_conjugate_of_square :
  Complex.conj (z ^ 2) = 3 + 4 * Complex.i := by
sorry

end complex_conjugate_of_square_l43_43405


namespace find_m_l43_43426

def vector (α : Type*) := prod α α

def a := (1, 3) : vector ℝ
def b (m : ℝ) := (m, 2 * m - 1) : vector ℝ

def collinear (u v : vector ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem find_m (m : ℝ) : collinear a (b m) → m = -1 :=
by sorry

end find_m_l43_43426


namespace sum_of_distances_l43_43043

theorem sum_of_distances 
  (A B C D : Type*) 
  [InnerProductSpace ℝ A] 
  [InnerProductSpace ℝ B] 
  [InnerProductSpace ℝ C] 
  [InnerProductSpace ℝ D] 
  (AB BC AC : ℝ)
  (V : ℝ)
  (I_A I_B I_C I_D : A → B → C → D → Type*)
  (DA DB DC : ℝ) 
  (concurrent : ∀ (AI_A BI_B CI_C DI_D : A → B → C → D → Prop), AI_A ∧ BI_B ∧ CI_C ∧ DI_D)
  (volume_eq : V = (15 * Real.sqrt 39) / 2)
  (side_ab : AB = 6)
  (side_bc : BC = 8)
  (side_ac : AC = 10)
  (sum_distances_eq : DA + DB + DC = 47 / 2) :
  let k := 1 / 2 in
  DA = 15 * k ∧ DB = 12 * k ∧ DC = 20 * k ∧ 47 = 49 :=
by
  sorry

end sum_of_distances_l43_43043


namespace book_distribution_l43_43012

theorem book_distribution (x y z : ℕ) 
  (h1 : x + y + z = 187)
  (h2 : 2.75 * x + 1.5 * y + (1 / 3) * z = 189)
  (h3 : 2.75 * x >= 1.5 * y)
  (h4 : 2.75 * x >= (1 / 3) * z)
  (h5 : 1.5 * y >= (1 / 3) * z) :
  (x = 36) ∧ (y = 34) ∧ (z = 117) :=
sorry

end book_distribution_l43_43012


namespace value_of_w_l43_43456

-- Define the positivity of w
def positive_integer (w : ℕ) := w > 0

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Define the function which encapsulates the problem
def problem_condition (w : ℕ) := sum_of_digits (10^w - 74)

-- The main proof problem
theorem value_of_w (w : ℕ) (h : positive_integer w) : problem_condition w = 17 :=
by
  sorry

end value_of_w_l43_43456


namespace solve_for_x_l43_43856

theorem solve_for_x : ∃ x : ℝ, (x - 3)^3 = 27 ∧ x = 6 :=
by {
  use 6,
  split,
  {
    -- part (x - 3)^3 = 27
    calc
      (6 - 3)^3 = 3^3 : by sorry -- (Resolve with definition and calculation)
      ... = 27         : by sorry -- (Resolve with definition)
  },
  {
    -- part x = 6
    exact rfl,
  }
}

end solve_for_x_l43_43856


namespace construct_inaccessible_angle_bisector_l43_43694

-- Definitions for problem context
structure Point :=
  (x y : ℝ)

structure Line :=
  (p1 p2 : Point)

structure Angle := 
  (vertex : Point)
  (ray1 ray2 : Line)

-- Predicate to determine if a line bisects an angle
def IsAngleBisector (L : Line) (A : Angle) : Prop := sorry

-- The inaccessible vertex angle we are considering
-- Let's assume the vertex is defined but we cannot access it physically in constructions
noncomputable def inaccessible_angle : Angle := sorry

-- Statement to prove: Construct a line that bisects the inaccessible angle
theorem construct_inaccessible_angle_bisector :
  ∃ L : Line, IsAngleBisector L inaccessible_angle :=
sorry

end construct_inaccessible_angle_bisector_l43_43694


namespace complex_number_problem_l43_43060

variables {a b c x y z : ℂ}

theorem complex_number_problem (h1 : a = (b + c) / (x - 2))
    (h2 : b = (c + a) / (y - 2))
    (h3 : c = (a + b) / (z - 2))
    (h4 : x * y + y * z + z * x = 67)
    (h5 : x + y + z = 2010) :
    x * y * z = -5892 :=
sorry

end complex_number_problem_l43_43060


namespace cube_root_fraction_l43_43341

theorem cube_root_fraction :
  (∛(4 / 18) : ℝ) = (∛(2 : ℝ) / (3 * ∛(3 : ℝ)) : ℝ) :=
by sorry

end cube_root_fraction_l43_43341


namespace workers_complete_time_l43_43932

theorem workers_complete_time 
  (time_A time_B time_C : ℕ) 
  (hA : time_A = 10)
  (hB : time_B = 12) 
  (hC : time_C = 15) : 
  let rate_A := (1: ℚ) / time_A
  let rate_B := (1: ℚ) / time_B
  let rate_C := (1: ℚ) / time_C
  let total_rate := rate_A + rate_B + rate_C
  1 / total_rate = 4 := 
by
  sorry

end workers_complete_time_l43_43932


namespace find_non_negative_integers_l43_43707

def has_exactly_two_distinct_solutions (a : ℕ) (m : ℕ) : Prop :=
  ∃ (x₁ x₂ : ℕ), (x₁ < m) ∧ (x₂ < m) ∧ (x₁ ≠ x₂) ∧ (x₁^2 + a) % m = 0 ∧ (x₂^2 + a) % m = 0

theorem find_non_negative_integers (a : ℕ) (m : ℕ := 2007) : 
  a < m ∧ has_exactly_two_distinct_solutions a m ↔ a = 446 ∨ a = 1115 ∨ a = 1784 :=
sorry

end find_non_negative_integers_l43_43707


namespace problem_statement_l43_43011

-- Definition of the problem
noncomputable section

-- Given that sides a, b, c are opposite to angles A, B, C in triangle ABC respectively.
-- Given the equation: 2 * a * sin A = (2 * b + c) * sin B + (2 * c + b) * sin C, 

-- Prove that A = 2 * pi / 3, and sin B + sin C attains a maximum value of 1 based on this information.
theorem problem_statement (a b c : ℝ) (A B C : ℝ) (h1 : 2 * a * Real.sin A = (2 * b + c) * Real.sin B + (2 * c + b) * Real.sin C) :
  A = 2 * Real.pi / 3 ∧ ∃ B C, (Real.sin B + Real.sin C = 1) := 
begin
  sorry
end

end problem_statement_l43_43011


namespace congruent_triangles_l43_43730

noncomputable theory

variables (A B C D P Q : Type*)

-- Conditions
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup P] [AddGroup Q]
variables [AffineSpace A B] [AffineSpace A C] [AffineSpace A D] [AffineSpace A P] [AffineSpace A Q]
variables [AffineSpace B C] [AffineSpace B D] [AffineSpace B P] [AffineSpace B Q]
variables [AffineSpace C D] [AffineSpace C P] [AffineSpace C Q]

-- Midpoints of sides
variables (A_m B_m C_m D_m : AffineSpace A)

-- Midpoints of diagonals
variables (P_m Q_m : AffineSpace A)

-- Define midpoints
axiom A_mid : A_m = midpoint B C
axiom B_mid : B_m = midpoint A D
axiom C_mid : C_m = midpoint D A
axiom D_mid : D_m = midpoint B C
axiom P_mid : P_m = midpoint A C
axiom Q_mid : Q_m = midpoint B D

theorem congruent_triangles : triangle B C P ≅ triangle A D Q :=
by sorry

end congruent_triangles_l43_43730


namespace trapezoid_side_BC_l43_43478

theorem trapezoid_side_BC
  (area : ℝ) (h_area : area = 200)
  (altitude : ℝ) (h_altitude : altitude = 10)
  (AB : ℝ) (h_AB : AB = 15)
  (CD : ℝ) (h_CD : CD = 25)
  (angle_AD_CD : ℝ) (h_angle_AD_CD : angle_AD_CD = real.pi / 4)
  (AD_perpendicular : ∀ (x : ℝ), x = 10 → real.cos angle_AD_CD = 0) :
  let BC := (200 - (25 * real.sqrt 5 + 25 * real.sqrt 21)) / 10 in
  BC = (200 - (25 * real.sqrt 5 + 25 * real.sqrt 21)) / 10 :=
by
  sorry

end trapezoid_side_BC_l43_43478


namespace superhero_speed_l43_43967

def convert_speed (speed_mph : ℕ) (mile_to_km : ℚ) : ℚ :=
  let speed_kmh := (speed_mph : ℚ) * (1 / mile_to_km)
  speed_kmh / 60

theorem superhero_speed :
  convert_speed 36000 (6 / 10) = 1000 :=
by sorry

end superhero_speed_l43_43967


namespace union_complement_eq_l43_43832

open Set

variable (I A B : Set ℤ)
variable (I_def : I = {-3, -2, -1, 0, 1, 2})
variable (A_def : A = {-1, 1, 2})
variable (B_def : B = {-2, -1, 0})

theorem union_complement_eq :
  A ∪ (I \ B) = {-3, -1, 1, 2} :=
by 
  rw [I_def, A_def, B_def]
  sorry

end union_complement_eq_l43_43832


namespace gcd_765432_654321_l43_43214

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43214


namespace game_divisibility_ensure_l43_43109

-- Defining the relationship of divisibility by 9 sums
def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

-- Defining the game logic problem
structure game (k : ℕ) :=
  (odd_pos_pick : list ℕ)
  (even_pos_pick : list ℕ)
  (condition_length : odd_pos_pick.length = k ∧ even_pos_pick.length = k)
  (total_digits : odd_pos_pick.length + even_pos_pick.length = 2*k)
  (digit_range : ∀ d ∈ odd_pos_pick ++ even_pos_pick, d ∈ finset.range 10)

-- Main conditions
def game_conditions : Prop :=
  ∀ (k : ℕ),
    (k = 10 → ¬(∃ odd even : list ℕ,
      game.mk odd even ⟨rfl, rfl⟩ (by simp) sorry ∧
      is_divisible_by_9 (odd.sum + even.sum))) ∧
    (k = 15 → ∃ odd even : list ℕ,
      game.mk odd even ⟨rfl, rfl⟩ (by simp) sorry ∧
      is_divisible_by_9 (odd.sum + even.sum))

theorem game_divisibility_ensure : game_conditions := sorry

end game_divisibility_ensure_l43_43109


namespace particle_sphere_intersect_l43_43464

noncomputable def lineParam (t : ℝ) : EuclideanSpace ℝ (Fin 3) :=
  ![1 + t, 1 + 2 * t, 1 + 3 * t]

def unit_sphere (x : ℝ) (y : ℝ) (z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 1

theorem particle_sphere_intersect (a b : ℕ) : 
  a = 10 → b = 2 → 
  unit_sphere (lineParam t).1 (lineParam t).2 (lineParam t).3 → 
  a + b = 12 :=
begin
  intros h1 h2 h3,
  simp [h1, h2],
  exact rfl,
end

end particle_sphere_intersect_l43_43464


namespace find_n_l43_43428

theorem find_n (n : ℕ) (k : ℕ) (x : ℝ) (h1 : k = 1) (h2 : x = 180 - 360 / n) (h3 : 1.5 * x = 180 - 360 / (n + 1)) :
    n = 3 :=
by
  -- proof steps will be provided here
  sorry

end find_n_l43_43428


namespace prism_faces_prism_faces_count_l43_43144

theorem prism_faces (V E F : ℕ) (n : ℕ)
  (hV : V = 2 * n)
  (hE : E = 3 * n)
  (hSum : V + E = 40) :
  F = n + 2 := by
  -- Proof omitted
  sorry

theorem prism_faces_count : 
  ∃ F : ℕ, F = 10 :=
  let n := 8 in
  have hV : 2 * n = 16 := rfl
  have hE : 3 * n = 24 := rfl
  have hSum : 16 + 24 = 40 := rfl
  have hF := prism_faces 16 24 (n + 2) 8 hV hE hSum
  exists.intro (n + 2) hF

end prism_faces_prism_faces_count_l43_43144


namespace problem_proof_l43_43373

theorem problem_proof (a b : ℝ) (h : {a, b / a, 1} = {a^2, a + b, 0}) : a ≠ 0 → a ^ 2023 + b ^ 2023 = -1 :=
by
  sorry

end problem_proof_l43_43373


namespace prod_estimate_l43_43702

theorem prod_estimate : 
  (∏ n in Finset.range ∞, n ^ (n ^ (-5 / 4))) ≈ 9000000 :=
by
sorry

end prod_estimate_l43_43702


namespace gcd_of_765432_and_654321_l43_43205

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43205


namespace compare_a_x_l43_43827

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem compare_a_x (x a b : ℝ) (h1 : a = log_base 5 (3^x + 4^x))
                    (h2 : b = log_base 4 (5^x - 3^x)) (h3 : a ≥ b) : x ≤ a :=
by
  sorry

end compare_a_x_l43_43827


namespace probability_divisor_of_60_l43_43229

theorem probability_divisor_of_60 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ a b c : ℕ, n = 2 ^ a * 3 ^ b * 5 ^ c ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) → 
  ∃ p : ℚ, p = 1 / 5 :=
by
  sorry

end probability_divisor_of_60_l43_43229


namespace gcd_proof_l43_43194

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43194


namespace mean_score_of_seniors_l43_43586

theorem mean_score_of_seniors (num_students : ℕ) (mean_score : ℚ) 
  (ratio_non_seniors_seniors : ℚ) (ratio_mean_seniors_non_seniors : ℚ) (total_score_seniors : ℚ) :
  num_students = 200 →
  mean_score = 80 →
  ratio_non_seniors_seniors = 1.25 →
  ratio_mean_seniors_non_seniors = 1.2 →
  total_score_seniors = 7200 →
  let num_seniors := (num_students : ℚ) / (1 + ratio_non_seniors_seniors)
  let mean_score_seniors := total_score_seniors / num_seniors
  mean_score_seniors = 80.9 :=
by 
  sorry

end mean_score_of_seniors_l43_43586


namespace gcd_765432_654321_eq_3_l43_43220

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l43_43220


namespace certain_number_is_50_l43_43613

theorem certain_number_is_50 (x : ℝ) (h : 0.6 * x = 0.42 * 30 + 17.4) : x = 50 :=
by
  sorry

end certain_number_is_50_l43_43613


namespace depth_of_canal_l43_43252

/-- The cross-section of a canal is a trapezium with a top width of 12 meters, 
a bottom width of 8 meters, and an area of 840 square meters. 
Prove that the depth of the canal is 84 meters.
-/
theorem depth_of_canal (top_width bottom_width area : ℝ) (h : ℝ) :
  top_width = 12 → bottom_width = 8 → area = 840 → 1 / 2 * (top_width + bottom_width) * h = area → h = 84 :=
by
  intros ht hb ha h_area
  sorry

end depth_of_canal_l43_43252


namespace jeremy_age_l43_43980

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end jeremy_age_l43_43980


namespace points_do_not_exist_l43_43673

/-- 
  If \( A, B, C, D \) are four points in space and 
  \( AB = 8 \) cm, 
  \( CD = 8 \) cm, 
  \( AC = 10 \) cm, 
  \( BD = 10 \) cm, 
  \( AD = 13 \) cm, 
  \( BC = 13 \) cm, 
  then such points \( A, B, C, D \) cannot exist.
-/
theorem points_do_not_exist 
  (A B C D : Type)
  (AB CD AC BD AD BC : ℝ) 
  (h1 : AB = 8) 
  (h2 : CD = 8) 
  (h3 : AC = 10)
  (h4 : BD = 10)
  (h5 : AD = 13)
  (h6 : BC = 13) : 
  false :=
sorry

end points_do_not_exist_l43_43673


namespace gcd_765432_654321_l43_43175

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43175


namespace length_P3P7_l43_43904

theorem length_P3P7 :
  ∃ P_3 P_7, (∃ x, 2 * cos (x + π / 4) * cos (x - π / 4) = 1 / 2 ∧ x > 0 ∧ x = P_3) ∧
  (∃ y, 2 * cos (y + π / 4) * cos (y - π / 4) = 1 / 2 ∧ y > 0 ∧ y = P_7 ∧ y = P_3 + 2 * π) ∧
  |P_3 - P_7| = 2 * π :=
sorry

end length_P3P7_l43_43904


namespace tom_speed_from_A_to_B_l43_43151

theorem tom_speed_from_A_to_B (D S : ℝ) (h1 : 2 * D = S * (3 * D / 36 - D / 20))
  (h2 : S * (3 * D / 36 - D / 20) = 3 * D / 36 ∨ 3 * D / 36 = S * (3 * D / 36 - D / 20))
  (h3 : D > 0) : S = 60 :=
by { sorry }

end tom_speed_from_A_to_B_l43_43151


namespace polynomial_roots_problem_l43_43148

theorem polynomial_roots_problem (a b c d e : ℝ) (h1 : a ≠ 0) 
    (h2 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
    (h3 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
    (h4 : a + b + c + d + e = 0) :
    (b + c + d) / a = -7 := 
sorry

end polynomial_roots_problem_l43_43148


namespace max_a3_a18_l43_43790

open_locale big_operators

noncomputable def arithmetic_sequence (a d : ℕ → ℕ) : Prop :=
  ∃ (a1 : ℕ) (d : ℕ), ∀ n, a n = a1 + (n - 1) * d

theorem max_a3_a18 
  (a : ℕ → ℕ) 
  (h_seq : arithmetic_sequence a) 
  (h_positive : ∀ n, 0 < a n)
  (h_sum20 : ∑ i in (finset.range 20), a (i + 1) = 100) :
  ∃ m, a 3 * a 18 = 25 :=
by
  sorry

end max_a3_a18_l43_43790


namespace medium_birdhouse_price_l43_43513

/-- Define the constants given in the problem --/
def price_of_large_birdhouse : ℕ := 22
def price_of_small_birdhouse : ℕ := 7
def number_of_large_birdhouses_sold : ℕ := 2
def number_of_medium_birdhouses_sold : ℕ := 2
def number_of_small_birdhouses_sold : ℕ := 3
def total_amount_made : ℕ := 97

/-- Define the variable for the medium birdhouse price --/
variable (M : ℕ)

/-- Define the proof statement --/
theorem medium_birdhouse_price :
  2 * price_of_large_birdhouse + 2 * M + 3 * price_of_small_birdhouse = total_amount_made → 
  M = 16 :=
sorry

end medium_birdhouse_price_l43_43513


namespace simplify_identity_l43_43853

theorem simplify_identity :
  ∀ θ : ℝ, θ = 160 → (1 / (Real.sqrt (1 + Real.tan (θ : ℝ) ^ 2))) = -Real.cos (θ : ℝ) :=
by
  intro θ h
  rw [h]
  sorry  

end simplify_identity_l43_43853


namespace find_n_coins_l43_43259

def num_coins : ℕ := 5

theorem find_n_coins (n : ℕ) (h : (n^2 + n + 2) = 2^n) : n = num_coins :=
by {
  -- Proof to be filled in
  sorry
}

end find_n_coins_l43_43259


namespace games_within_division_l43_43617

/-- 
Given a baseball league with two four-team divisions,
where each team plays N games against other teams in its division,
and M games against teams in the other division.
Given that N > 2M and M > 6, and each team plays a total of 92 games in a season,
prove that each team plays 60 games within its own division.
-/
theorem games_within_division (N M : ℕ) (hN : N > 2 * M) (hM : M > 6) (h_total : 3 * N + 4 * M = 92) :
  3 * N = 60 :=
by
  -- The proof is omitted.
  sorry

end games_within_division_l43_43617


namespace pipe_filling_time_l43_43633

/-- 
A problem involving two pipes filling and emptying a tank. 
Time taken for the first pipe to fill the tank is proven to be 16.8 minutes.
-/
theorem pipe_filling_time :
  ∃ T : ℝ, (∀ T, let r1 := 1 / T
                let r2 := 1 / 24
                let time_both_pipes_open := 36
                let time_first_pipe_only := 6
                (r1 - r2) * time_both_pipes_open + r1 * time_first_pipe_only = 1) ∧
           T = 16.8 :=
by
  sorry

end pipe_filling_time_l43_43633


namespace graduation_students_sum_l43_43583

theorem graduation_students_sum :
  (∑ x in {x | ∃ (y : ℕ), x * y = 360 ∧ 18 ≤ x ∧ 12 ≤ y}) = 24 :=
by
  sorry

end graduation_students_sum_l43_43583


namespace gcd_765432_654321_l43_43208

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43208


namespace translate_line_up_l43_43584

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := 2 * x - 4

-- Define the new line equation after translating upwards by 5 units
def new_line (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement to prove the translation result
theorem translate_line_up (x : ℝ) : original_line x + 5 = new_line x :=
by
  -- This would normally be where the proof goes, but we'll insert a placeholder
  sorry

end translate_line_up_l43_43584


namespace intersection_M_N_l43_43817

open Set

variable (x y : ℝ)

theorem intersection_M_N :
  let M := {x | x < 1}
  let N := {y | ∃ x, x < 1 ∧ y = 1 - 2 * x}
  M ∩ N = ∅ := sorry

end intersection_M_N_l43_43817


namespace taylor_series_expansion_tan_l43_43711

noncomputable def taylor_series_tan (z : ℂ) : ℂ :=
  z + (2 / 3!) * z^3 + (16 / 5!) * z^5

theorem taylor_series_expansion_tan :
  ∀ (z : ℂ), |z| < π / 2 → (∑' n, (1 / n!) * (fderiv ℂ^[n] (λ z, tan z) 0) • z^n) = taylor_series_tan z := by
  sorry

end taylor_series_expansion_tan_l43_43711


namespace distance_between_edges_BD_and_CE_l43_43472

variable (a : ℝ) (ABCDEFGH : Set (ℝ × ℝ × ℝ))  -- Coordinates for the vertices of the cube

noncomputable def distance_between_edges_of_cube (BD CE' : Set (ℝ × ℝ × ℝ)) : ℝ :=
  -- Dummy definition for distance between two sets of points
  sorry 

theorem distance_between_edges_BD_and_CE'_is_a_div_sqrt3
  (a_pos : 0 < a)
  (BD CE' : Set (ℝ × ℝ × ℝ))
  (H1 : cube_with_edge_length ABCDEFGH a)
  (H2 : is_segment BD)
  (H3 : is_segment CE')
  (H4 : BD = {B, D})
  (H5 : CE' = {C, E'})
  (H6 : are_opposite_edges BD CE') :
  distance_between_edges_of_cube BD CE' = a / Real.sqrt 3 := sorry

end distance_between_edges_BD_and_CE_l43_43472


namespace pump_out_time_correct_l43_43264

noncomputable def pump_out_time (length width water_depth : ℝ) (pump_rate num_pumps cubic_ft_gallons : ℝ) : ℕ :=
  let volume_cubic_ft := water_depth * length * width
  let volume_gallons := volume_cubic_ft * cubic_ft_gallons
  let total_pumping_rate := pump_rate * num_pumps
  let time_minutes := volume_gallons / total_pumping_rate
  ⌈time_minutes⌉

theorem pump_out_time_correct :
  pump_out_time 15 40 (21 / 12) 10 2 7.5 = 394 := 
by
  sorry

end pump_out_time_correct_l43_43264


namespace domain_of_g_l43_43096

def f (x : ℝ) : Prop := x ∈ Set.Icc (-12.0) 6.0

def g (x : ℝ) : Prop := f (3 * x)

theorem domain_of_g : Set.Icc (-4.0) 2.0 = {x : ℝ | g x} := 
by 
    sorry

end domain_of_g_l43_43096


namespace maximum_c_value_l43_43943

noncomputable def problem_statement (a b c : ℝ) : Prop :=
  2^a + 2^b = 2^(a + b) ∧ 2^a + 2^b + 2^c = 2^(a + b + c)

theorem maximum_c_value (a b c : ℝ) (h : problem_statement a b c) : c ≤ 2 - real.log (3) / real.log (2) := sorry

end maximum_c_value_l43_43943


namespace petya_cannot_win_l43_43014

def player : Type := Fin 10

def points (p : player) : ℕ := sorry -- Points scored by player p

def disqualified : player := sorry -- The disqualified player

def total_games := (10 * 9) / 2

theorem petya_cannot_win
  (h1 : ∀ p1 p2 : player, p1 ≠ p2 → points p1 ≠ points p2) -- Every player has unique points
  (h2 : ∀ p : player, points p ≤ 9) -- Every player played 9 games
  (h3 : ∀ p : player, if p = disqualified then points p = 9 else points p < 9) -- Points of disqualified player
  (h4 : ∀ p : player, p ≠ disqualified → points p ≤ 4) -- Points of remaining players are less than or equal to 4
  : ∀ p : player, p ≠ disqualified → ¬(points p > 4) := by -- No player can score more than 4 after disqualification
begin
  sorry
end

end petya_cannot_win_l43_43014


namespace sum_of_areas_of_triangles_in_cube_l43_43892

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l43_43892


namespace sqrt_combination_l43_43305

theorem sqrt_combination :
  (∀ x, (x = √6) ∨ (x = √12) ∨ (x = √15) ∨ (x = √18) ∨ (x = √24) → 
  (√24 = 2 * √6)) := 
by
  sorry

end sqrt_combination_l43_43305


namespace gcd_765432_654321_l43_43210

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l43_43210


namespace average_marks_l43_43323

theorem average_marks (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) 
                      (chemistry_marks : ℕ) (biology_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 →
  chemistry_marks = 87 → biology_marks = 81 → 
  (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / 5 = 85 :=
by
  intros
  sorry

end average_marks_l43_43323


namespace molecular_weight_of_one_mole_l43_43228

-- Conditions
def molecular_weight_6_moles : ℤ := 1404
def num_moles : ℤ := 6

-- Theorem
theorem molecular_weight_of_one_mole : (molecular_weight_6_moles / num_moles) = 234 := by
  sorry

end molecular_weight_of_one_mole_l43_43228


namespace quadrilateral_abcd_l43_43467

theorem quadrilateral_abcd 
  (BC CD AD : ℝ) (ang_A ang_B : ℝ) 
  (p q : ℕ)
  (side_sum : p + q = 197) :
  BC = 10 → CD = 15 → AD = 12 → ang_A = 75 → ang_B = 75 → 
  (∃ (AB : ℝ), AB = p + real.sqrt q) :=
by
  intros hBC hCD hAD hA hB
  use (p + real.sqrt q)
  sorry

end quadrilateral_abcd_l43_43467


namespace incircle_hexagon_area_ratio_l43_43318

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def radius_incircle (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 2

noncomputable def area_incircle (r : ℝ) : ℝ :=
  Real.pi * r^2

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let A_hexagon := area_hexagon s
  let r := radius_incircle s
  let A_incircle := area_incircle r
  A_incircle / A_hexagon

theorem incircle_hexagon_area_ratio (s : ℝ) (h : s = 1) :
  area_ratio s = (Real.pi * Real.sqrt 3) / 6 :=
by
  sorry

end incircle_hexagon_area_ratio_l43_43318


namespace correlation_coefficient_determination_l43_43597

variable (height_variation : ℝ)
variable (random_error : ℝ)

-- Conditions given
axiom height_variation_accounting : height_variation = 0.76
axiom random_error_accounting : random_error = 0.24

-- Prove the correlation coefficient R^2 is 0.76
theorem correlation_coefficient_determination : 
  (height_variation + random_error = 1) → 
  height_variation = 0.76 := 
by
  intros h
  rw [height_variation_accounting, random_error_accounting]
  sorry

end correlation_coefficient_determination_l43_43597


namespace inequality_a_b_c_l43_43054

noncomputable def a := Real.logBase (1/3) 6
noncomputable def b := (1/4)^0.8
noncomputable def c := Real.log Real.pi

theorem inequality_a_b_c : c > b ∧ b > a :=
by
  sorry

end inequality_a_b_c_l43_43054


namespace sum_of_possible_k_squared_l43_43257

theorem sum_of_possible_k_squared :
  ∃ (k_vals : Finset ℚ), 
    ∀ (AB AC : ℚ) (AD : ℕ), 
    (AB = 4) → (AC = 9) → 
    (0 < AD) → 
    (AD ∈ (6 : ℚ) • (Finset.range 6).image (λ m, m) → 
    (k_vals = Finset.of_list [11 / 36, 5 / 9, 3 / 4, 8 / 9, 35 / 36]) →
    k_vals.sum = 125 / 36 := 
sorry

end sum_of_possible_k_squared_l43_43257


namespace sum_converges_series_sum_l43_43061

open Set
open Filter

noncomputable def f0 : ℝ → ℝ := sorry -- define the continuous function on [0, 1]

def fn (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then f0 x else ∫ t in 0..x, fn (n - 1) t

theorem sum_converges (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  ∃ l : ℝ, has_sum (λ n, fn n x) l :=
sorry

theorem series_sum (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  ∑' n, fn n x = ∫ t in 0..x, f0 t * Real.exp (x - t) :=
sorry

end sum_converges_series_sum_l43_43061


namespace probability_product_zero_is_10_over_21_l43_43641

-- Define the set and its elements
def my_set := {-3, -1, 0, 0, 2, 5, 7}

-- Define the total number of ways to choose two different elements from the set
def total_ways : ℕ := Nat.choose 7 2

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := 2 * 5

-- Define the probability of the product being zero
def probability_zero_product := favorable_outcomes / total_ways

-- State the theorem
theorem probability_product_zero_is_10_over_21 : probability_zero_product = 10 / 21 := sorry

end probability_product_zero_is_10_over_21_l43_43641


namespace mooncake_fruit_probability_l43_43681

theorem mooncake_fruit_probability : 
  let fruits := 5
  let meats := 4
  let fruit_combinations := Nat.choose fruits 2
  let meat_combinations := Nat.choose meats 2
  let total_combinations := fruit_combinations + meat_combinations
  in fruit_combinations / total_combinations = 5 / 8 :=
by
  sorry

end mooncake_fruit_probability_l43_43681


namespace gcd_of_765432_and_654321_l43_43199

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43199


namespace determine_p_l43_43325

noncomputable def roots (p : ℝ) : ℝ × ℝ :=
  let discr := p ^ 2 - 48
  ((-p + Real.sqrt discr) / 2, (-p - Real.sqrt discr) / 2)

theorem determine_p (p : ℝ) :
  let (x1, x2) := roots p
  (x1 - x2 = 1) → (p = 7 ∨ p = -7) :=
by
  intros
  sorry

end determine_p_l43_43325


namespace count_valid_combinations_l43_43800

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def evaluate_expression (a b c d : ℕ) (op1 op2 op3 : ℕ → ℕ → ℕ) : ℕ :=
  op3 (op2 (op1 a b) c) d

def operator_combinations : List (ℕ → ℕ → ℕ) := [Nat.add, Nat.sub, Nat.mul]

def valid_prime_combinations : List (ℕ → ℕ → ℕ → ℕ → ℕ → ℕ) :=
  [(λ a b c d op1 op2 op3 => evaluate_expression a b c d op1 op2 op3),
   (λ a b c d op1 op2 op3 => evaluate_expression a b c d op1 op3 op2),
   (λ a b c d op1 op2 op3 => evaluate_expression a b c d op2 op1 op3),
   (λ a b c d op1 op2 op3 => evaluate_expression a b c d op2 op3 op1),
   (λ a b c d op1 op2 op3 => evaluate_expression a b c d op3 op1 op2),
   (λ a b c d op1 op2 op3 => evaluate_expression a b c d op3 op2 op1)]

theorem count_valid_combinations : (valid_prime_combinations.count 
  (λ f, is_prime (f 2 3 5 7 _ _ _)) = 8) :=
by
  sorry

end count_valid_combinations_l43_43800


namespace associate_professor_pencils_l43_43309

theorem associate_professor_pencils
  (A B P : ℕ)
  (h1 : A + B = 7)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 11) :
  P = 2 :=
by {
  -- Variables declarations and assumptions
  -- Combine and manipulate equations to prove P = 2
  sorry
}

end associate_professor_pencils_l43_43309


namespace greatest_common_ratio_l43_43577

theorem greatest_common_ratio {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : (b = (a + c) / 2 → b^2 = a * c) ∨ (c = (a + b) / 2 ∧ b = -a / 2)) :
  ∃ r : ℝ, r = -2 :=
by
  sorry

end greatest_common_ratio_l43_43577


namespace minimum_value_am_hm_l43_43825

theorem minimum_value_am_hm (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hpqr : p + q + r = 3) :
  \(\frac{1}{p + 3q} + \frac{1}{q + 3r} + \frac{1}{r + 3p} \geq \frac{3}{4}\) :=
by
  sorry

end minimum_value_am_hm_l43_43825


namespace smallest_n_exists_l43_43019

-- Definition of the problem
def exam_problem (f : ℕ → vector (fin 4) 5) : Prop :=
  ∃ n: ℕ, (∀ (s : finset (fin 2000)), s.card = n → 
  ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (finset.card (finset.filter (λ i, f a.nth i = f b.nth i ∧ f a.nth i = f c.nth i ∧ f a.nth i = f d.nth i) (finset.range 5))) ≤ 3)

theorem smallest_n_exists :
  ∃ n, exam_problem (λ i, vector.of_fn (λ j, (i / (4 ^ j)) % 4 + 1)) ∧ n = 25 :=
begin
  sorry
end

end smallest_n_exists_l43_43019


namespace total_cans_collected_l43_43521

theorem total_cans_collected 
  (bags_saturday : ℕ) 
  (bags_sunday : ℕ) 
  (cans_per_bag : ℕ) 
  (h1 : bags_saturday = 6) 
  (h2 : bags_sunday = 3) 
  (h3 : cans_per_bag = 8) : 
  bags_saturday + bags_sunday * cans_per_bag = 72 := 
by 
  simp [h1, h2, h3]; -- Simplify using the given conditions
  sorry -- Placeholder for the computation proof

end total_cans_collected_l43_43521


namespace k_range_correct_l43_43523

noncomputable def k_range (k : ℝ) : Prop :=
  (∀ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
  (∀ x : ℝ, k * x ^ 2 + k * x + 1 > 0) ∧
  ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∨
   (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0)) ∧
  ¬ ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
    (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0))

theorem k_range_correct (k : ℝ) : k_range k ↔ (-3 < k ∧ k < 0) ∨ (3 ≤ k ∧ k < 4) :=
sorry

end k_range_correct_l43_43523


namespace cost_of_3000_pencils_l43_43267

-- Define the cost per box and the number of pencils per box
def cost_per_box : ℝ := 36
def pencils_per_box : ℕ := 120

-- Define the number of pencils to buy
def pencils_to_buy : ℕ := 3000

-- Define the total cost to prove
def total_cost_to_prove : ℝ := 900

-- The theorem to prove
theorem cost_of_3000_pencils : 
  (cost_per_box / pencils_per_box) * pencils_to_buy = total_cost_to_prove :=
by
  sorry

end cost_of_3000_pencils_l43_43267


namespace fill_pool_in_approx_3_69_hours_l43_43632

theorem fill_pool_in_approx_3_69_hours :
  let rateA := 1 / 8
  let rateB := 1 / 12
  let rateC := 1 / 16
  let combined_rate := rateA + rateB + rateC
  let time_to_fill := 1 / combined_rate
  abs (time_to_fill - (48 / 13)) < 0.01 :=
by
  sorry

end fill_pool_in_approx_3_69_hours_l43_43632


namespace sum_abs_ineq_l43_43391

open Real

theorem sum_abs_ineq (n : ℕ) (x : ℕ → ℝ) : 
    ∑ h in Finset.range n, ∑ j in Finset.range n, abs (x h + x j) ≥ n * ∑ i in Finset.range n, abs (x i) := 
sorry

end sum_abs_ineq_l43_43391


namespace maximize_profit_l43_43270

open Real

noncomputable def y (x : ℝ) : ℝ := -2 * x + 100

noncomputable def w (x : ℝ) : ℝ := -2 * x^2 + 160 * x - 2760

theorem maximize_profit :
  (20 ≤ 36) →
  (36 ≤ 36) →
  (∀ x, 20 ≤ x → x ≤ 36 → w(x) ≤ w(36)) ∧
  (w 36 = 408) :=
by
  intros
  split
  · intro x hx1 hx2
    sorry
  · sorry

end maximize_profit_l43_43270


namespace coefficient_x2_expansion_l43_43547

open Nat

theorem coefficient_x2_expansion (x : ℕ) : 
  (∃ n : ℕ, n = 7) → 
  (∃ k : ℤ, k = -21 ∧ 
  (x - 1)^7 = (x^2 * k + (polynomial.term x 0))) := 
begin
  intros h1,
  cases h1 with n hn,
  use [-21],
  split,
  { refl },
  { sorry }
end

end coefficient_x2_expansion_l43_43547


namespace frustum_properties_l43_43298

noncomputable def pyramid_volume (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * altitude

noncomputable def smaller_pyramid_volume (original_volume : ℝ) (ratio : ℝ) : ℝ :=
  original_volume * (ratio ^ 3)

noncomputable def frustum_volume (total_volume : ℝ) (smaller_volume : ℝ) : ℝ :=
  total_volume - smaller_volume

noncomputable def frustum_surface_area (base_edge : ℝ) (small_base_edge : ℝ) (slant_height : ℝ) : ℝ :=
  (small_base_edge ^ 2) + (base_edge ^ 2) + (1 / 2) * (small_base_edge ^ 2 + base_edge ^ 2) * slant_height

theorem frustum_properties :
  let base_edge := 20
  let altitude := 15
  let cut_height := 9
  let ratio := (cut_height : ℝ) / altitude
  let original_volume := pyramid_volume base_edge altitude
  let smaller_volume := smaller_pyramid_volume original_volume ratio
  let frustum_vol := frustum_volume original_volume smaller_volume
  let small_base_edge := ratio * base_edge
  let slant_height := Real.sqrt ((small_base_edge - base_edge) ^ 2 + (altitude - cut_height) ^ 2)
  let frustum_surf_area := frustum_surface_area base_edge small_base_edge slant_height
  frustum_vol = 1568 ∧ frustum_surf_area ≈ 1113.8 :=
by
  sorry

end frustum_properties_l43_43298


namespace jim_saves_money_by_buying_gallon_l43_43486

theorem jim_saves_money_by_buying_gallon :
  let gallon_price := 8
  let bottle_price := 3
  let ounces_per_gallon := 128
  let ounces_per_bottle := 16
  (ounces_per_gallon / ounces_per_bottle) * bottle_price - gallon_price = 16 :=
by
  sorry

end jim_saves_money_by_buying_gallon_l43_43486


namespace distinct_sums_is_98_l43_43388

def arithmetic_sequence_distinct_sums (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :=
  (∀ n : ℕ, S n = (n * (2 * a_n 0 + (n - 1) * d)) / 2) ∧
  S 5 = 0 ∧
  d ≠ 0 →
  (∃ distinct_count : ℕ, distinct_count = 98 ∧
   ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 ∧ S i = S j → i = j)

theorem distinct_sums_is_98 (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence_distinct_sums a_n S d) :
  ∃ distinct_count : ℕ, distinct_count = 98 :=
sorry

end distinct_sums_is_98_l43_43388


namespace sum_a_b_c_d_eq_nine_l43_43360

theorem sum_a_b_c_d_eq_nine
  (a b c d : ℤ)
  (h : (Polynomial.X ^ 2 + (Polynomial.C a) * Polynomial.X + Polynomial.C b) *
       (Polynomial.X ^ 2 + (Polynomial.C c) * Polynomial.X + Polynomial.C d) =
       Polynomial.X ^ 4 + 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 11 * Polynomial.X + 6) :
  a + b + c + d = 9 :=
by
  sorry

end sum_a_b_c_d_eq_nine_l43_43360


namespace solve_system_of_equations_l43_43539

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2^(x + 2*y) + 2^x = 3 * 2^y) ∧ (2^(2*x + y) + 2 * 2^y = 4 * 2^x) ∧ (x = 1 / 2) ∧ (y = 1 / 2) := 
by
  let x := (1:ℝ) / 2
  let y := (1:ℝ) / 2
  have h1 : 2^(x + 2*y) + 2^x = 3 * 2^y := sorry
  have h2 : 2^(2*x + y) + 2 * 2^y = 4 * 2^x := sorry
  exact ⟨x, y, h1, h2, rfl, rfl⟩

end solve_system_of_equations_l43_43539


namespace math_problem_l43_43710

theorem math_problem (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
(h : 4 * real.root 4 (real.root 3 5 - real.root 3 3) = real.root 4 p + real.root 4 q - real.root 4 r) : 
(p + q + r = 63) :=
sorry

end math_problem_l43_43710


namespace determine_quarters_given_l43_43507

def total_initial_coins (dimes quarters nickels : ℕ) : ℕ :=
  dimes + quarters + nickels

def updated_dimes (original_dimes added_dimes : ℕ) : ℕ :=
  original_dimes + added_dimes

def updated_nickels (original_nickels factor : ℕ) : ℕ :=
  original_nickels + original_nickels * factor

def total_coins_after_addition (dimes quarters nickels : ℕ) (added_dimes added_quarters added_nickels_factor : ℕ) : ℕ :=
  updated_dimes dimes added_dimes +
  (quarters + added_quarters) +
  updated_nickels nickels added_nickels_factor

def quarters_given_by_mother (total_coins initial_dimes initial_quarters initial_nickels added_dimes added_nickels_factor : ℕ) : ℕ :=
  total_coins - total_initial_coins initial_dimes initial_quarters initial_nickels - added_dimes - initial_nickels * added_nickels_factor

theorem determine_quarters_given :
  quarters_given_by_mother 35 2 6 5 2 2 = 10 :=
by
  sorry

end determine_quarters_given_l43_43507


namespace estate_value_l43_43514

-- Ms. K's estate division and the conditions
variables (E : ℝ) (x : ℝ) (y : ℝ)
variable h_gardener : y = 600   -- The gardener received $600
variable h_ratio : 5 * x + 3 * x + 2 * x = (3 / 5) * E  -- The daughters and son received 3/5 of the estate in the ratio 5:3:2
variable h_husband : 3 * 2 * x  -- The husband received three times as much as the son
variable h_estate : E = 6 * x + 5 * x + 3 * x + 2 * x + y  -- Sum of all shares

-- Prove the entire estate was $15000
theorem estate_value : E = 15000 := by
  sorry

end estate_value_l43_43514


namespace log_sum_l43_43377

variable (m a b : ℝ)
variable (m_pos : 0 < m)
variable (m_ne_one : m ≠ 1)
variable (h1 : m^2 = a)
variable (h2 : m^3 = b)

theorem log_sum (m_pos : 0 < m) (m_ne_one : m ≠ 1) (h1 : m^2 = a) (h2 : m^3 = b) :
  2 * Real.log (a) / Real.log (m) + Real.log (b) / Real.log (m) = 7 := 
sorry

end log_sum_l43_43377


namespace simple_interest_correct_l43_43966

def principal : ℝ := 8032.5
def rate : ℝ := 10
def time : ℝ := 5

def simple_interest (P R T : ℝ) := P * R * T / 100

theorem simple_interest_correct :
  simple_interest principal rate time = 4016.25 :=
by
  -- Proof goes here
  sorry

end simple_interest_correct_l43_43966


namespace factor_expression_eq_l43_43991

-- Define the given expression
def given_expression (x : ℝ) : ℝ :=
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6)

-- Define the correct factored form
def factored_expression (x : ℝ) : ℝ :=
  5 * x * (3 * x^2 + 17)

-- The theorem stating the equality of the given expression and its factored form
theorem factor_expression_eq (x : ℝ) : given_expression x = factored_expression x :=
  by
  sorry

end factor_expression_eq_l43_43991


namespace ratio_of_discounted_bricks_l43_43917

theorem ratio_of_discounted_bricks (total_bricks discounted_price full_price total_spending: ℝ) 
  (h1 : total_bricks = 1000) 
  (h2 : discounted_price = 0.25) 
  (h3 : full_price = 0.50) 
  (h4 : total_spending = 375) : 
  ∃ D : ℝ, (D / total_bricks = 1 / 2) ∧ (0.25 * D + 0.50 * (total_bricks - D) = total_spending) := 
  sorry

end ratio_of_discounted_bricks_l43_43917


namespace find_g_zero_l43_43064

noncomputable def g (x : ℝ) : ℝ := sorry  -- fourth-degree polynomial

-- Conditions
axiom cond1 : |g 1| = 16
axiom cond2 : |g 3| = 16
axiom cond3 : |g 4| = 16
axiom cond4 : |g 5| = 16
axiom cond5 : |g 6| = 16
axiom cond6 : |g 7| = 16

-- statement to prove
theorem find_g_zero : |g 0| = 54 := 
by sorry

end find_g_zero_l43_43064


namespace find_ratio_of_time_l43_43833

variable (r : ℝ)
variable (route1_total_time route2_total_time : ℝ)
variable (uphill_time flat_path_time : ℝ)
variable (stage3_time2 stage3_time1 : ℝ)

def ratio_of_time (r : ℝ) := r = 17 / 4

def first_route_time (r : ℝ) :=
  let uphill_time := 6
  let path_time := 6 * r
  let stage1_2 := uphill_time + path_time
  let stage3_time1 := stage1_2 / 3
  uphill_time + path_time + stage3_time1

def second_route_time :=
  let flat_path_time := 14
  let stage3_time2 := 2 * flat_path_time
  flat_path_time + stage3_time2

axiom route_times_relation (r : ℝ) :
  first_route_time r + 18 = second_route_time

theorem find_ratio_of_time :
  ratio_of_time r := by
  sorry

end find_ratio_of_time_l43_43833


namespace two_squares_similar_l43_43241

-- Definition of a shape and its properties
inductive Shape
| parallelogram : Shape
| square : Shape
| rectangle : Shape
| rhombus : Shape

-- Predicate definition for similarity
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.square, Shape.square => true
  | _, _ => false

-- Theorem stating that two squares are always similar
theorem two_squares_similar : ∀ (s1 s2 : Shape), s1 = Shape.square ∧ s2 = Shape.square → similar s1 s2 :=
by
  intros s1 s2 h
  cases h
  simp [similar]

-- Sorry placeholder to skip actual proof
sorry

end two_squares_similar_l43_43241


namespace triangle_inequality_condition_l43_43921

theorem triangle_inequality_condition (u v w a b c : ℝ) 
  (h_poly : Polynomial.root (Polynomial.C (w) + Polynomial.C (-v) * Polynomial.X + Polynomial.C (-u) * Polynomial.X^2 + Polynomial.X^3) a b c) 
  (h_vieta1 : a + b + c = u)
  (h_vieta2 : a * b + b * c + c * a = v)
  (h_vieta3 : a * b * c = w) : 
  u * v > 2 * w :=
sorry

end triangle_inequality_condition_l43_43921


namespace joe_total_time_l43_43028

variable (r_w t_w : ℝ) 
variable (t_total : ℝ)

-- Given conditions:
def joe_problem_conditions : Prop :=
  (r_w > 0) ∧ 
  (t_w = 9) ∧
  (3 * r_w * (3)) / 2 = r_w * 9 / 2 + 1 / 2

-- The statement to prove:
theorem joe_total_time (h : joe_problem_conditions r_w t_w) : t_total = 13 :=
by { sorry }

end joe_total_time_l43_43028


namespace circle_radii_sum_tangent_conditions_l43_43275

theorem circle_radii_sum_tangent_conditions :
  let D_center := (s: ℝ) * (1, 1) in  -- Center of the circle
  let tangent_eq := ∀ s: ℝ, (s - 4) * (s - 4) + s * s = (s + 2) * (s + 2) in
  ∀ (s: ℝ), tangent_eq s → s = 6 + 2 * Real.sqrt 6 ∨ s = 6 - 2 * Real.sqrt 6 →
  ∃! sum, sum = 12 :=
by
  sorry 

end circle_radii_sum_tangent_conditions_l43_43275


namespace football_game_spectators_l43_43459

-- Define the conditions and the proof goals
theorem football_game_spectators 
  (A C : ℕ) 
  (h_condition_1 : 2 * A + 2 * C + 40 = 310) 
  (h_condition_2 : C = A / 2) : 
  A = 90 ∧ C = 45 ∧ (A + C + 20) = 155 := 
by 
  sorry

end football_game_spectators_l43_43459


namespace integer_solutions_eq_3_l43_43778

theorem integer_solutions_eq_3 {a : ℝ} :
  (∃ S : finset ℤ, S.card = 3 ∧ ∀ x ∈ S, ||(x - 2 : ℝ) - 1| = a) → a = 1 :=
sorry

end integer_solutions_eq_3_l43_43778


namespace cricket_team_members_l43_43278

theorem cricket_team_members (n : ℕ) (captain_age wicket_keeper_age average_whole_age average_remaining_age : ℕ) :
  captain_age = 24 →
  wicket_keeper_age = 31 →
  average_whole_age = 23 →
  average_remaining_age = 22 →
  n * average_whole_age - captain_age - wicket_keeper_age = (n - 2) * average_remaining_age →
  n = 11 :=
by
  intros h_cap_age h_wk_age h_avg_whole h_avg_remain h_eq
  sorry

end cricket_team_members_l43_43278


namespace perimeter_eq_120_plus_2_sqrt_1298_l43_43793

noncomputable def total_perimeter_of_two_quadrilaterals (AB BC CD : ℝ) (AC : ℝ := Real.sqrt (AB ^ 2 + BC ^ 2)) (AD : ℝ := Real.sqrt (AC ^ 2 + CD ^ 2)) : ℝ :=
2 * (AB + BC + CD + AD)

theorem perimeter_eq_120_plus_2_sqrt_1298 (hAB : AB = 15) (hBC : BC = 28) (hCD : CD = 17) :
  total_perimeter_of_two_quadrilaterals 15 28 17 = 120 + 2 * Real.sqrt 1298 :=
by
  sorry

end perimeter_eq_120_plus_2_sqrt_1298_l43_43793


namespace triangles_from_chord_intersections_l43_43859

/-- Given ten points on a circle, calculate the number of triangles with all three vertices in the interior of the circle, formed by intersections of chords, with the condition that no three chords intersect in a point inside the circle. --/
theorem triangles_from_chord_intersections :
  let n := 10 in
  let number_of_intersections := Nat.choose n 4 in
  let number_of_triangles := number_of_intersections in
  number_of_triangles = 210 := by
  sorry

end triangles_from_chord_intersections_l43_43859


namespace quasi_pythagorean_prime_divisor_l43_43652

def is_quasi_pythagorean_triple (a b c : ℕ) : Prop :=
  c^2 = a^2 + a * b + b^2

theorem quasi_pythagorean_prime_divisor (a b c : ℕ) (hc : is_quasi_pythagorean_triple a b c) :
  ∃ p : ℕ, p > 5 ∧ nat.prime p ∧ p ∣ c :=
sorry

end quasi_pythagorean_prime_divisor_l43_43652


namespace quadratic_transformation_l43_43661

theorem quadratic_transformation :
  ∀ x : ℝ, (x^2 - 6 * x - 5 = 0) → ((x - 3)^2 = 14) :=
by
  intros x h
  sorry

end quadratic_transformation_l43_43661


namespace bus_capacity_total_kids_l43_43578

-- Definitions based on conditions
def total_rows : ℕ := 25
def lower_deck_rows : ℕ := 15
def upper_deck_rows : ℕ := 10
def lower_deck_capacity_per_row : ℕ := 5
def upper_deck_capacity_per_row : ℕ := 3
def staff_members : ℕ := 4

-- Theorem statement
theorem bus_capacity_total_kids : 
  (lower_deck_rows * lower_deck_capacity_per_row) + 
  (upper_deck_rows * upper_deck_capacity_per_row) - staff_members = 101 := 
by
  sorry

end bus_capacity_total_kids_l43_43578


namespace simplest_square_root_l43_43975

open Real

theorem simplest_square_root :
  (sqrt 4 = 2) → 
  (sqrt 12 = 2 * sqrt 3) → 
  (sqrt (1 / 2) = sqrt 2 / 2) → 
  (∀ x, x ∈ {sqrt 4, sqrt 12, sqrt (1 / 2), sqrt 5} → (sqrt 5 ≤ x)) :=
by
  intros h1 h2 h3 x hx
  cases hx
  . simp [hx]
  . simp [hx] at h1
    rw [h1]
    linarith
  . simp [hx] at h2
    rw [h2]
    apply sqrt_le_sqrt
    norm_num
  . simp [hx] at h3
    rw [h3]
    apply sqrt_le_sqrt
    norm_num
  sorry

end simplest_square_root_l43_43975


namespace arc_length_of_circle_l43_43112

theorem arc_length_of_circle (r θ : ℝ) (h_r : r = 2) (h_θ : θ = 120) : 
  (θ / 180 * r * Real.pi) = (4 / 3) * Real.pi := by
  sorry

end arc_length_of_circle_l43_43112


namespace problem1_l43_43942

variable {x : ℝ} {b c : ℝ}

theorem problem1 (hb : b = 9) (hc : c = -11) :
  b + c = -2 := 
by
  simp [hb, hc]
  sorry

end problem1_l43_43942


namespace quadratic_no_solution_l43_43454

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_no_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  0 < a ∧ discriminant a b c ≤ 0 :=
by
  sorry

end quadratic_no_solution_l43_43454


namespace total_cost_l43_43510

-- Define the given conditions
def total_tickets : Nat := 10
def discounted_tickets : Nat := 4
def full_price : ℝ := 2.00
def discounted_price : ℝ := 1.60

-- Calculation of the total cost Martin spent
theorem total_cost : (discounted_tickets * discounted_price) + ((total_tickets - discounted_tickets) * full_price) = 18.40 := by
  sorry

end total_cost_l43_43510


namespace total_flag_distance_moved_l43_43984

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l43_43984


namespace gcd_divisors_remainders_l43_43593

theorem gcd_divisors_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end gcd_divisors_remainders_l43_43593


namespace sqrt_12_minus_neg_half_inv_abs_sqrt3_plus3_plus_2023_minus_pi_pow_0_eq_sqrt3_simplify_and_substitute_expression_l43_43258

namespace MathProof

-- Problem 1
theorem sqrt_12_minus_neg_half_inv_abs_sqrt3_plus3_plus_2023_minus_pi_pow_0_eq_sqrt3 :
  (Real.sqrt 12 - (-1/2)^(-1) - Real.abs (Real.sqrt 3 + 3) + (2023 - Real.pi) ^ 0 = Real.sqrt 3) := 
  sorry

-- Problem 2
variable (x : ℝ) 
theorem simplify_and_substitute_expression (h : 0 < x ∧ x ≤ 3 ∧ x ≠ 1 ∧ x ≠ 0) : 
  (1 < x ∧ x < 3 → ((3 * x - 8) / (x - 1) - (x + 1) / x / ((x^2 - 1) / (x^2 - 3 * x)) = (2*x - 5) / (x - 1)) ∧ (x = 2 → (2*x - 5) / (x - 1) = -1)) :=
  sorry

end MathProof

end sqrt_12_minus_neg_half_inv_abs_sqrt3_plus3_plus_2023_minus_pi_pow_0_eq_sqrt3_simplify_and_substitute_expression_l43_43258


namespace general_integral_of_ODE_l43_43868

noncomputable def general_solution (x y : ℝ) (m C : ℝ) : Prop :=
  (x^2 * y - x - m) / (x^2 * y - x + m) = C * Real.exp (2 * m / x)

theorem general_integral_of_ODE (m : ℝ) (y : ℝ → ℝ) (C : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∀ (y' : ℝ → ℝ) (x : ℝ), deriv y x = m^2 / x^4 - (y x)^2) ∧ 
  (y 1 = 1 / x + m / x^2) ∧ 
  (y 2 = 1 / x - m / x^2) →
  general_solution x (y x) m C :=
by 
  sorry

end general_integral_of_ODE_l43_43868


namespace equipment_B_production_l43_43647

theorem equipment_B_production
  (total_production : ℕ)
  (sample_size : ℕ)
  (A_sample_production : ℕ)
  (B_sample_production : ℕ)
  (A_total_production : ℕ)
  (B_total_production : ℕ)
  (total_condition : total_production = 4800)
  (sample_condition : sample_size = 80)
  (A_sample_condition : A_sample_production = 50)
  (B_sample_condition : B_sample_production = 30)
  (ratio_condition : (A_sample_production / B_sample_production) = (5 / 3))
  (production_condition : A_total_production + B_total_production = total_production) :
  B_total_production = 1800 := 
sorry

end equipment_B_production_l43_43647


namespace water_height_in_tank_l43_43656

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

def height_of_water (r h : ℝ) : ℝ :=
  let V_tank := cone_volume r h
  let V_water := 0.20 * V_tank
  let x := (V_water / V_tank)^ (1/3 : ℝ)
  h * x

theorem water_height_in_tank:
  let r := 20
  let h := 100
  let V_tank := cone_volume r h
  let V_water := 0.20 * V_tank
  let x := (V_water / V_tank)^ (1/3 : ℝ)
  let h_water := height_of_water r h
  h_water = 50* (real.cbrt(2/5)) ∧ 50 + 2 = 52 :=
by
  sorry

end water_height_in_tank_l43_43656


namespace probability_of_heads_9_and_die_6_l43_43158

noncomputable def prob_heads_9_and_die_6 : ℚ :=
  let total_coin_outcomes := (2:ℚ)^12
  let ways_to_get_9_heads := nat.choose 12 9
  let prob_9_heads := ways_to_get_9_heads / total_coin_outcomes
  let prob_die_6 := 1 / 6
  prob_9_heads * prob_die_6

theorem probability_of_heads_9_and_die_6 :
  prob_heads_9_and_die_6 = 55 / 6144 := 
by
  sorry

end probability_of_heads_9_and_die_6_l43_43158


namespace irregular_lines_n_eq_4_irregular_lines_n_eq_5_l43_43624

def is_irregular (soldiers : List ℕ) : Prop :=
  ∀ i, i < soldiers.length - 2 → (soldiers[i] < soldiers[i+1] ∧ soldiers[i+1] > soldiers[i+2]) ∨ (soldiers[i] > soldiers[i+1] ∧ soldiers[i+1] < soldiers[i+2])

def count_irregular_lines (n : ℕ) : ℕ :=
  (Finset.univ.filter (λ l, is_irregular (l.val))).card

theorem irregular_lines_n_eq_4 : count_irregular_lines 4 = 10 := 
  by sorry

theorem irregular_lines_n_eq_5 : count_irregular_lines 5 = 32 := 
  by sorry

end irregular_lines_n_eq_4_irregular_lines_n_eq_5_l43_43624


namespace arithmetic_progression_contains_sixth_power_l43_43668

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end arithmetic_progression_contains_sixth_power_l43_43668


namespace parabola_translation_correct_l43_43919

noncomputable def translate_parabola (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) : Prop :=
  let x' := x - 1
  let y' := y + 3
  y' = -2 * x'^2 - 1

theorem parabola_translation_correct (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) :
  translate_parabola x y h :=
sorry

end parabola_translation_correct_l43_43919


namespace wendi_chickens_problem_l43_43589

theorem wendi_chickens_problem :
  ∃ (r : ℝ), 
    let initial_chickens := 4 in
    let increased_chickens := initial_chickens * r in
    let eaten_chickens := increased_chickens - 1 in
    let found_chickens := 10 - 4 in
    let final_chickens := eaten_chickens + found_chickens in
    final_chickens = 13 ∧ r = 2 :=
begin
  sorry
end

end wendi_chickens_problem_l43_43589


namespace smallest_prime_less_cost_l43_43860

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

def option1_cost (n : ℕ) : ℕ :=
  digit_sum n

def option2_cost (n : ℕ) : ℕ :=
  binary_sum n

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def smallest_prime_transmission_cost : ℕ :=
  Nat.find_greatest (λ n, n < 5000 ∧ is_prime n ∧ option2_cost n < option1_cost n) 487

theorem smallest_prime_less_cost :
  smallest_prime_transmission_cost = 487 :=
by
  sorry

end smallest_prime_less_cost_l43_43860


namespace vector_sum_magnitude_l43_43396

noncomputable def vector_norm (a : ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 ^ 2 + a.2 ^ 2)

theorem vector_sum_magnitude
  (a b : ℝ × ℝ)
  (ha : vector_norm a = 1)
  (hb : vector_norm b = real.sqrt 2)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0) :
  vector_norm (a.1 + b.1, a.2 + b.2) = real.sqrt 3 :=
by
  sorry

end vector_sum_magnitude_l43_43396


namespace abe_bob_matching_probability_l43_43658

-- Definitions of the conditions
def AbeJellyBeans := {green := 1, red := 2, total := 3}
def BobJellyBeans := {green := 2, yellow := 1, red := 1, total := 4}

-- Statement of the problem
theorem abe_bob_matching_probability :
  let p_green_abe := (1 / AbeJellyBeans.total : ℚ)
  let p_green_bob := (2 / BobJellyBeans.total : ℚ)
  let p_both_green := p_green_abe * p_green_bob

  let p_red_abe := (2 / AbeJellyBeans.total : ℚ)
  let p_red_bob := (1 / BobJellyBeans.total : ℚ)
  let p_both_red := p_red_abe * p_red_bob

  p_both_green + p_both_red = 1 / 3 :=
begin
  let p_green_abe := (1 / AbeJellyBeans.total : ℚ),
  let p_green_bob := (2 / BobJellyBeans.total : ℚ),
  let p_both_green := p_green_abe * p_green_bob,
  
  let p_red_abe := (2 / AbeJellyBeans.total : ℚ),
  let p_red_bob := (1 / BobJellyBeans.total : ℚ),
  let p_both_red := p_red_abe * p_red_bob,

  have h1 : p_both_green = 1 / 6, by norm_num,
  have h2 : p_both_red = 1 / 6, by norm_num,
  rw [h1, h2],
  norm_num
end

end abe_bob_matching_probability_l43_43658


namespace angles_wrt_orthocenter_and_altitude_feet_l43_43816

variable (A B C H H_A H_B H_C : Point)
variable (angle_A angle_B angle_C : Real)

-- Assume A, B, C form an acute-angled triangle ABC.
-- H is the orthocenter of triangle ABC.
-- H_A, H_B, and H_C are the feet of the altitudes from A, B, and C, respectively.
-- The angles of triangle ABC are respectively angle_A, angle_B, and angle_C.

axiom acute_tri (ABC : Triangle) : AcuteTriangle ABC
axiom orthocenter (ABC H : Triangle) : H = orthocenter_of ABC
axiom feet_of_altitudes (ABC H_A H_B H_C : Triangle) : 
  H_A = foot_of_altitude_from A ∧ 
  H_B = foot_of_altitude_from B ∧ 
  H_C = foot_of_altitude_from C

theorem angles_wrt_orthocenter_and_altitude_feet 
  (h_tri_acute : acute_tri (Triangle.mk A B C))
  (h_orthocenter : orthocenter (Triangle.mk A B C) H)
  (h_feet_of_altitudes : feet_of_altitudes (Triangle.mk A B C) H_A H_B H_C)
  : 
  ∠A H_B H_C = angle_C ∧ 
  ∠A H_C H_B = angle_B ∧ 
  ∠B H_C H_A = angle_C ∧ 
  ∠B H_A H_C = angle_A ∧ 
  ∠C H_A H_B = angle_A ∧ 
  ∠C H_B H_A = angle_B ∧ 
  ∠H_A H_B H_C = 180 - 2 * angle_B ∧ 
  ∠H_B H_C H_A = 180 - 2 * angle_C ∧ 
  ∠H_C H_A H_B = 180 - 2 * angle_A := by sorry

end angles_wrt_orthocenter_and_altitude_feet_l43_43816


namespace t_shirts_in_two_hours_l43_43677

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end t_shirts_in_two_hours_l43_43677


namespace ellipse_concyclic_points_l43_43733

open Real EuclideanGeometry

-- Definitions and conditions based on the problem statement
variables {a b : ℝ} (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
variables {P Q F1 F2 M1 M2 : EuclideanSpace ℝ 2}

-- Assume P is a point on the ellipse C, F1 and F2 are foci, and l1 and l2 are directrices
-- Foci and Directrices are related and used in the proof
variables (P_ellipse : ellipse a b P) (line_parallel : ∀ (x : ℝ), M1 y = M2 y) 
(line_intersections : ∃ (Q : EuclideanSpace ℝ 2), line M1 F1 ∩ line M2 F2 = Q)
variables (para_F1F2 : M1 = M2 ∧ M1 y = P y) 

theorem ellipse_concyclic_points :
  ∃ (Q : EuclideanSpace ℝ 2), is_concyclic P F1 Q F2 :=
sorry

end ellipse_concyclic_points_l43_43733


namespace average_minutes_run_l43_43679

theorem average_minutes_run (t : ℕ) (t_pos : 0 < t) 
  (average_first_graders : ℕ := 8) 
  (average_second_graders : ℕ := 12) 
  (average_third_graders : ℕ := 16)
  (num_first_graders : ℕ := 9 * t)
  (num_second_graders : ℕ := 3 * t)
  (num_third_graders : ℕ := t) :
  (8 * 9 * t + 12 * 3 * t + 16 * t) / (9 * t + 3 * t + t) = 10 := 
by
  sorry

end average_minutes_run_l43_43679


namespace smallest_number_among_options_l43_43973

noncomputable def binary_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 111111 => 63
  | _ => 0

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 210 => 2 * 6^2 + 1 * 6
  | _ => 0

noncomputable def base_nine_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 85 => 8 * 9 + 5
  | _ => 0

theorem smallest_number_among_options :
  min 75 (min (binary_to_decimal 111111) (min (base_six_to_decimal 210) (base_nine_to_decimal 85))) = binary_to_decimal 111111 :=
by 
  sorry

end smallest_number_among_options_l43_43973


namespace coloring_symmetry_ways_l43_43678

def is_rotational_symmetry (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ x y, grid x y ↔ grid (2 - x) (2 - y)

theorem coloring_symmetry_ways : 
  (∃ (grid : ℕ → ℕ → Prop), 
    (is_rotational_symmetry grid) ∧ 
    (∀ x y, (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) → grid x y ↔ ((x = 1 ∧ y = 1) ∨ 
    ((x, y) ∈ {(0, 0), (0, 2), (2, 0), (2, 2)} → grid x y \/ grid (2 - x) (2 - y))) ∧
    (finset.univ.filter (grid 1 1 ∨ grid 0 0 ∨ grid 0 2 ∨ grid 2 0 ∨ grid 2 2)).card = 3) ↔
    4) :=
sorry

end coloring_symmetry_ways_l43_43678


namespace relative_positions_of_lines_and_circle_l43_43159

-- Defining the concepts of intersection (I) and tangency (T)
def intersect (P Q : Point) : Prop := ∃ l₁ l₂ : Line, P ∈ l₁ ∧ P ∈ l₂ ∧ Q ∈ l₁ ∧ Q ∈ l₂
def tangent (l : Line) (C : Circle) : Prop := ∃ P : Point, P ∈ l ∧ P ∈ C ∧ ∀ Q : Point, Q ≠ P → Q ∉ l ∨ Q ∉ C

-- Defining the possible cases of relative positions
def relativePositions : List Nat :=
  [4, 3, 3, 2, 2, 1, 0]

-- The main statement: Proving the set of possible relative positions is exactly 7
theorem relative_positions_of_lines_and_circle : relativePositions.length = 7 := by
  -- The proof will be inserted here
  sorry

end relative_positions_of_lines_and_circle_l43_43159


namespace gcd_proof_l43_43190

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43190


namespace height_relationship_l43_43588

theorem height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
sorry

end height_relationship_l43_43588


namespace truncated_cone_circumscribed_sphere_l43_43968

noncomputable def truncated_cone_vs_sphere_volume_ratio
  (a : ℝ) (r : ℝ) (x : ℝ) (h_upper : r^2 = (sqrt a) * x^2) : ℝ :=
  (a + sqrt a + 1) / (2 * sqrt a)

theorem truncated_cone_circumscribed_sphere (a: ℝ) (r: ℝ) (x: ℝ)
  (h_upper : r^2 = (sqrt a) * x^2):
  truncated_cone_vs_sphere_volume_ratio a r x h_upper =
  (a + sqrt a + 1) / (2 * sqrt a) :=
sorry

end truncated_cone_circumscribed_sphere_l43_43968


namespace arithmetic_sequences_count_l43_43321

theorem arithmetic_sequences_count :
    ∃ n : ℕ, ∃ a1 d : ℕ, (n ≥ 3 ∧ n (a1 + ((n - 1) * d) / 2) = 97^2) ∧
    (n = 97^2 ∧ a1 = 1 ∧ d = 0 ∨
    n = 97 ∧ (a1 + 48 * d = 97 ∧ (
        (d = 0 ∧ a1 = 97) ∨
        (d = 1 ∧ a1 = 49) ∨
        (d = 2 ∧ a1 = 1)))) :=
begin
    sorry
end

end arithmetic_sequences_count_l43_43321


namespace maria_reaches_one_l43_43541

theorem maria_reaches_one :
  ∃ n : ℕ, n = 5 ∧ (Nat.iterate (λ x => x / 3) n 200 = 1) :=
by
  use 5
  unfold Nat.iterate
  ext
  
  -- First application: ⌊200 / 3⌋ = 66
  have h1 : 200 / 3 = 66 := sorry,
  rw h1,
  
  -- Second application: ⌊66 / 3⌋ = 22
  have h2 : 66 / 3 = 22 := sorry,
  rw [h2],
  
  -- Third application: ⌊22 / 3⌋ = 7
  have h3 : 22 / 3 = 7 := sorry,
  rw [h3],
  
  -- Fourth application: ⌊7 / 3⌋ = 2
  have h4 : 7 / 3 = 2 := sorry,
  rw [h4],
  
  -- Fifth application: ⌊2 / 3⌋ = 1
  have h5 : 2 / 3 = 1 := sorry,
  rw [h5],
  
  -- At this point, Maria's number is 1
  refl

end maria_reaches_one_l43_43541


namespace number_of_valid_x_l43_43120

def star (a b : ℤ) : ℤ := (a * a) / b

theorem number_of_valid_x : 
  (finset.univ.filter (λ x : ℤ, x > 0 ∧ star 15 x = 225 / x)).card = 9 :=
by 
  sorry

end number_of_valid_x_l43_43120


namespace exists_positive_integer_x_for_triangle_l43_43650

theorem exists_positive_integer_x_for_triangle {x : ℕ} : (8 : ℝ) + 12 > (x : ℝ)^3 + 1 ∧ (8 : ℝ) + (x : ℝ)^3 + 1 > 12 ∧ (12 : ℝ) + (x : ℝ)^3 + 1 > 8 →
  x ∈ {2, 3} :=
by {
  intro h,
  sorry
}

end exists_positive_integer_x_for_triangle_l43_43650


namespace two_squares_similar_l43_43242

-- Definition of a shape and its properties
inductive Shape
| parallelogram : Shape
| square : Shape
| rectangle : Shape
| rhombus : Shape

-- Predicate definition for similarity
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.square, Shape.square => true
  | _, _ => false

-- Theorem stating that two squares are always similar
theorem two_squares_similar : ∀ (s1 s2 : Shape), s1 = Shape.square ∧ s2 = Shape.square → similar s1 s2 :=
by
  intros s1 s2 h
  cases h
  simp [similar]

-- Sorry placeholder to skip actual proof
sorry

end two_squares_similar_l43_43242


namespace find_f_inv_128_l43_43446

open Function

theorem find_f_inv_128 (f : ℕ → ℕ) 
  (h₀ : f 5 = 2) 
  (h₁ : ∀ x, f (2 * x) = 2 * f x) : 
  f⁻¹' {128} = {320} :=
by
  sorry

end find_f_inv_128_l43_43446


namespace sum_of_possible_x_values_l43_43918

theorem sum_of_possible_x_values (x y : ℕ) (h1 : x * y = 300) (h2 : x ≥ 18) (h3 : y ≥ 12) :
  (∑ (x' : ℕ) in (Finset.filter (λ x', 300 % x' = 0 ∧ x' ≥ 18 ∧ (300 / x') ≥ 12) (Finset.range 301)), x') = 45 :=
by
  sorry

end sum_of_possible_x_values_l43_43918


namespace prove_seq_properties_l43_43389

theorem prove_seq_properties (a b : ℕ → ℕ) (S T : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_sum : ∀ n, 2 * S n = a n ^ 2 + n)
  (h_b : ∀ n, b n = a (n + 1) * 2 ^ n)
  : (∀ n, a n = n) ∧ (∀ n, T n = n * 2 ^ (n + 1)) :=
sorry

end prove_seq_properties_l43_43389


namespace sum_of_areas_of_triangles_l43_43891

theorem sum_of_areas_of_triangles (m n p : ℤ) :
  let vertices := {v : ℝ × ℝ × ℝ | v.1 ∈ {0, 2} ∧ v.2 ∈ {0, 2} ∧ v.3 ∈ {0, 2}},
      triangles := {t : tuple ℝ 9 | 
                     ∃ v1 v2 v3 ∈ vertices, 
                     t = (v1.1, v1.2, v1.3, v2.1, v2.2, v2.3, v3.1, v3.2, v3.3)},
      triangle_area := λ t : tuple ℝ 9, 
        let (x1, y1, z1, x2, y2, z2, x3, y3, z3) := t in
        1 / 2 * real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2) *
                 real.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2 + (z3 - z1) ^ 2) *
                 real.sqrt ((x3 - x2) ^ 2 + (y3 - y2) ^ 2 + (z3 - z2) ^ 2)
  in (∑ t in triangles, triangle_area t) = 48 + real.sqrt 4608 + real.sqrt 3072 :=
sorry

end sum_of_areas_of_triangles_l43_43891


namespace range_floor_plus_one_l43_43869

noncomputable def floor_plus_one (x : ℝ) : ℝ := real.floor x + 1

theorem range_floor_plus_one : set.range (λ x, floor_plus_one x) = {0, 1, 2, 3} :=
begin
  sorry
end

end range_floor_plus_one_l43_43869


namespace new_volume_of_cylinder_l43_43126

theorem new_volume_of_cylinder
  (r h : ℝ) -- original radius and height
  (V : ℝ) -- original volume
  (h_volume : V = π * r^2 * h) -- volume formula for the original cylinder
  (new_radius : ℝ := 3 * r) -- new radius is three times the original radius
  (new_volume : ℝ) -- new volume to be determined
  (h_original_volume : V = 10) -- original volume equals 10 cubic feet
  : new_volume = 9 * V := -- new volume should be 9 times the original volume
by
  sorry

end new_volume_of_cylinder_l43_43126


namespace graph_symmetry_l43_43717

variable {α β : Type*}
variable {f : α → β}

theorem graph_symmetry (f : ℝ → ℝ) :
  (∀ (x : ℝ), f (-(x - 1)) = f (x - 1)) → 
  (∀ (x : ℝ), f (x - 1) = f (-x + 1)) :=
by
  intros
  simp
  sorry

end graph_symmetry_l43_43717


namespace circle_through_fixed_points_locus_of_intersection_l43_43750

-- Define the ellipse E and points P and Q
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def point_on_ellipse_B (x0 y0 : ℝ) : Prop :=
  ellipse x0 y0

def point_on_ellipse_C (x0 y0 : ℝ) : Prop :=
  ellipse (-x0) (-y0)

-- Define the fixed points M and N
def fixed_point_M : (ℝ × ℝ) := (-1, 0)
def fixed_point_N : (ℝ × ℝ) := (1, 0)

-- Define points P and Q in terms of B and C
def y1 (x0 y0 : ℝ) : ℝ :=
  2 * y0 / (2 - x0)

def y2 (x0 y0 : ℝ) : ℝ :=
  -2 * y0 / (2 + x0)

def point_P (x0 y0 : ℝ) : (ℝ × ℝ) :=
  (0, y1 x0 y0)

def point_Q (x0 y0 : ℝ) : (ℝ × ℝ) :=
  (0, y2 x0 y0)

-- Proof Problem 1: The circle Γ with diameter PQ passes through fixed points
theorem circle_through_fixed_points (x0 y0 : ℝ) (h : ellipse x0 y0) : 
  let P := point_P x0 y0,
      Q := point_Q x0 y0 in
  ∃ Γ : (ℝ × ℝ) → Prop, Γ (0, y1 x0 y0) ∧ Γ (0, y2 x0 y0) ∧ (Γ fixed_point_M ∨ Γ fixed_point_N) :=
begin
  sorry
end

-- Proof Problem 2: Equation of the locus of the intersection T
theorem locus_of_intersection (xT yT : ℝ) (h : xT^2 - yT^2 = 1 ∧ xT ≠ 1 ∧ xT ≠ -1) : 
  ellipse ((xT + 1) / (2 - (yT / xT))) yT ∧ ellipse ((xT - 1) / (2 + (yT / xT))) yT :=
begin
  sorry
end

end circle_through_fixed_points_locus_of_intersection_l43_43750


namespace problem_I_problem_II_l43_43055

noncomputable def f (x a : ℝ) := ((4 * x + a) * Real.log x) / (3 * x + 1)

theorem problem_I (a : ℝ) (h : ∀ (a : ℝ), 
  let f (x : ℝ) := ((4 * x + a) * Real.log x) / (3 * x + 1) 
  in (3 + a) * (Real.log 1 + 1) = 4):
a = 0 := 
sorry

theorem problem_II (m : ℝ) :
(∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → 
   let f (x : ℝ) := (4 * Real.log x) / (3 * x + 1) 
   in f x ≤ m * x) → m ≥ 4 / (3 * Real.exp 1 + 1) := 
sorry

end problem_I_problem_II_l43_43055


namespace part_i_part_ii_l43_43491

noncomputable section

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) (a : ℂ)
def A_star : Matrix (Fin n) (Fin n) ℂ := conjTranspose A

-- Part (i)
theorem part_i (h1 : A - A_star A = 2 * a • (1 : Matrix (Fin n) (Fin n) ℂ)) : 
  |det A| ≥ |a|^n := 
by 
  sorry

-- Part (ii)
theorem part_ii (h1 : A - A_star A = 2 * a • (1 : Matrix (Fin n) (Fin n) ℂ)) 
                (h2 : |det A| = |a|^n) : 
  A = a • (1 : Matrix (Fin n) (Fin n) ℂ) := 
by 
  sorry

end part_i_part_ii_l43_43491


namespace gcd_765432_654321_l43_43165

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43165


namespace problem_stated_l43_43940

-- Definitions of constants based on conditions
def a : ℕ := 5
def b : ℕ := 4
def c : ℕ := 3
def d : ℕ := 400
def x : ℕ := 401

-- Mathematical theorem stating the question == answer given conditions
theorem problem_stated : a * x + b * x + c * x + d = 5212 := 
by 
  sorry

end problem_stated_l43_43940


namespace feifei_sheep_count_l43_43812

noncomputable def sheep_number (x y : ℕ) : Prop :=
  (y = 3 * x + 15) ∧ (x = y - y / 3)

theorem feifei_sheep_count :
  ∃ x y : ℕ, sheep_number x y ∧ x = 5 :=
sorry

end feifei_sheep_count_l43_43812


namespace coloring_triangle_l43_43655

theorem coloring_triangle (total_circles blue_circles green_circles red_circles : ℕ) :
  total_circles = 6 →
  blue_circles = 4 →
  green_circles = 1 →
  red_circles = 1 →
  (nat.choose total_circles 2 * 2 = 30) :=
by
  intros h_total h_blue h_green h_red
  rw [h_total, h_blue, h_green, h_red]
  sorry

end coloring_triangle_l43_43655


namespace jeremy_age_l43_43978

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end jeremy_age_l43_43978


namespace maximum_pairs_l43_43366

theorem maximum_pairs (k : ℕ) (a b : fin k → ℕ) :
  (∀ i, 1 ≤ a i ∧ a i ≤ 4019) →
  (∀ i, 1 ≤ b i ∧ b i ≤ 4019) →
  (∀ i, a i < b i) →
  (∀ i j, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ a i ≠ b j ∧ a j ≠ b i) →
  (∀ i j, i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i, a i + b i ≤ 4019) →
  k ≤ 1607 :=
by sorry

end maximum_pairs_l43_43366


namespace infinite_solutions_of_system_l43_43540

theorem infinite_solutions_of_system :
  ∃x y : ℝ, (3 * x - 4 * y = 10 ∧ 6 * x - 8 * y = 20) :=
by
  sorry

end infinite_solutions_of_system_l43_43540


namespace jeremy_age_l43_43979

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end jeremy_age_l43_43979


namespace problem_statement_l43_43067

/-- 
Let f be a differentiable function on ℝ with the following properties:
1. f'(x) exists for all x ∈ ℝ.
2. f(x) = 4 * x^2 - f(-x) for all x ∈ ℝ.
3. For x ∈ (-∞, 0), f'(x) + 1/2 < 4 * x.
4. f(m + 1) ≤ f(-m) + 4 * m + 2.

Prove that m ≥ -1/2.
-/
theorem problem_statement (f : ℝ → ℝ) (h_diff : ∀ x, DifferentiableAt ℝ f x)
  (h_eq : ∀ x, f x = 4 * x^2 - f (-x))
  (h_deriv_cond : ∀ x, x < 0 → f' x + 1 / 2 < 4 * x)
  (h_ineq : ∀ m, f (m + 1) ≤ f (-m) + 4 * m + 2)
  : ∀ m, m ≥ -1 / 2 := 
sorry

end problem_statement_l43_43067


namespace prove_A_eq_lg_b_l43_43256

-- Define logarithms and square root functions based on the given problem
noncomputable def lg (x : ℝ) : ℝ := real.log x / real.log 10
noncomputable def log_2 (x : ℝ) : ℝ := real.log x / real.log 2

theorem prove_A_eq_lg_b (b : ℝ) (hb : b > 1) : 
  let A := (lg b * 2 ^ (log_2 (lg b)))^(1/2) * (lg (b^2))^(-1/2) / 
                   (real.sqrt ((lg b)^2 + 1) / (2 * lg b) + 1 - 10 ^ (0.5 * lg (lg (b^(1/2))))) in
  A = lg b := 
by 
  sorry

end prove_A_eq_lg_b_l43_43256


namespace midpoints_area_l43_43093

def square_area : ℝ := 16
def quarter_circle_area : ℝ := (Real.pi * (2^2)) / 4

theorem midpoints_area (side_length : ℝ) (segment_length : ℝ) : 
  side_length = 4 → 
  segment_length = 4 → 
  100 * (square_area - 4 * quarter_circle_area) = 972 := 
by
  intros h1 h2
  let square_area := side_length ^ 2
  let quarter_circle_area := (Real.pi * (segment_length ^ 2) / 4)
  have h3 : square_area = 16 := by sorry
  have h4 : quarter_circle_area = (\(∏ ^ 2) / 4) := by sorry
  rw [h1, h3, h4]
  rw [← sub_eq_add_neg, mul_sub, mul_one, sub_eq_add_neg]
  norm_num
  sorry

end midpoints_area_l43_43093


namespace gcd_765432_654321_l43_43171

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43171


namespace female_students_count_l43_43786

variable (F M : ℕ)

def numberOfMaleStudents (F : ℕ) : ℕ := 3 * F

def totalStudents (F M : ℕ) : Prop := F + M = 52

theorem female_students_count :
  totalStudents F (numberOfMaleStudents F) → F = 13 :=
by
  intro h
  sorry

end female_students_count_l43_43786


namespace triangle_area_eq_64_l43_43697

noncomputable def Triangle_area (y1 y2 y3 : ℝ → ℝ) (y4 : ℝ) : ℝ :=
  let O := (0, 0)
  let A := (y4, y4)
  let B := (-y4, y4)
  let base := y4 - (-y4)
  let height := y4
  (1 / 2) * base * height

theorem triangle_area_eq_64 :
  Triangle_area (λ x, x) (λ x, -x) (λ x, 8) 8 = 64 :=
by
  sorry

end triangle_area_eq_64_l43_43697


namespace number_of_N_l43_43719

/-- Problem Statement:
    Prove that there are exactly 105 integers N in the range 1 to 2000 such that
    gcd(N^2 + 9, N + 5) > 1.
-/
theorem number_of_N (h : ∃ N : ℕ, 1 ≤ N ∧ N ≤ 2000 ∧ Nat.gcd (N^2 + 9) (N + 5) > 1) :
    (finset.filter (λ N, 1 ≤ N ∧ N ≤ 2000 ∧ Nat.gcd (N^2 + 9) (N + 5) > 1) (finset.range 2001)).card = 105 := 
sorry

end number_of_N_l43_43719


namespace sum_valid_n_eq_28_l43_43924

open Nat

theorem sum_valid_n_eq_28 :
  (∑ n in {n | choose 28 14 + choose 28 n = choose 29 15}, n) = 28 :=
by
  sorry

end sum_valid_n_eq_28_l43_43924


namespace base_k_for_repeating_series_equals_fraction_l43_43721

-- Define the fraction 5/29
def fraction := 5 / 29

-- Define the repeating series in base k
def repeating_series (k : ℕ) : ℚ :=
  (1 / k) / (1 - 1 / k^2) + (3 / k^2) / (1 - 1 / k^2)

-- State the problem
theorem base_k_for_repeating_series_equals_fraction (k : ℕ) (hk1 : 0 < k) (hk2 : k ≠ 1):
  repeating_series k = fraction ↔ k = 8 := sorry

end base_k_for_repeating_series_equals_fraction_l43_43721


namespace sum_of_distances_leq_3r_l43_43814

-- Definitions and conditions from the problem
variables (T : Type) [Plane T] (△: Triangle T) (H : Point T)
variable [is_Acute_Triangle △]
variable (r : ℝ) -- inradius of the triangle
variables (x y z : ℝ) -- distances from the orthocenter to the sides of the triangle

-- Given conditions
axiom orthocenter_distances : (H = orthocenter △) ∧ distance_from_orthocenter_to_sides △ H = (x, y, z)
axiom inradius_definition : inradius △ = r
axiom inradius_inequality : r >= 0

-- Proof statement (the theorem to prove)
theorem sum_of_distances_leq_3r : x + y + z ≤ 3 * r :=
sorry

end sum_of_distances_leq_3r_l43_43814


namespace minimum_perimeter_of_rectangle_l43_43698

variables (a b c d : ℝ)

theorem minimum_perimeter_of_rectangle :
  (∃ (H H₁ : ℝ) (a b c d : ℝ), 
  (a * b = 1) ∧ 
  (c * d = 1.5) ∧ 
  (2 * (a + b) = H) ∧ 
  (2 * (c + d) = (1 / 2) * H) ∧ 
  (H = 4 * Real.sqrt 6)) :=
begin
  -- existence proof required
  sorry 
end

end minimum_perimeter_of_rectangle_l43_43698


namespace divisor_unique_l43_43007

theorem divisor_unique {b : ℕ} (h1 : 826 % b = 7) (h2 : 4373 % b = 8) : b = 9 :=
sorry

end divisor_unique_l43_43007


namespace cube_triangle_area_sum_solution_l43_43902

def cube_vertex_triangle_area_sum (m n p : ℤ) : Prop :=
  m + n + p = 121 ∧
  (∀ (a : ℕ) (b : ℕ) (c : ℕ), a * b * c = 8) -- Ensures the vertices are for a 2*2*2 cube

theorem cube_triangle_area_sum_solution :
  cube_vertex_triangle_area_sum 48 64 9 :=
by
  unfold cube_vertex_triangle_area_sum
  split
  · exact rfl -- m + n + p = 121
  · intros a b c h
    sorry -- Conditions ensuring these m, n, p were calculated from a 2x2x2 cube

end cube_triangle_area_sum_solution_l43_43902


namespace compute_27_inv_cubed_l43_43689

theorem compute_27_inv_cubed : 27^(-1 / 3 : ℝ) = 1 / 3 := 
sorry

end compute_27_inv_cubed_l43_43689


namespace train_passes_tree_in_time_l43_43648

-- Define the conditions
def train_length : ℝ := 750
def train_speed_kmh : ℝ := 85
def conversion_factor : ℝ := 5 / 18
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor

-- Define the question and proof goal
def time_to_pass_tree (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_passes_tree_in_time : 
    time_to_pass_tree train_length train_speed_ms ≈ 31.77 := 
by
  sorry

end train_passes_tree_in_time_l43_43648


namespace stratified_sampling_correct_l43_43610

-- Define the total number of students and the ratio of students in grades 10, 11, and 12
def total_students : ℕ := 4000
def ratio_grade10 : ℕ := 32
def ratio_grade11 : ℕ := 33
def ratio_grade12 : ℕ := 35

-- The total sample size
def sample_size : ℕ := 200

-- Define the expected numbers of students drawn from each grade in the sample
def sample_grade10 : ℕ := 64
def sample_grade11 : ℕ := 66
def sample_grade12 : ℕ := 70

-- The theorem to be proved
theorem stratified_sampling_correct :
  (sample_grade10 + sample_grade11 + sample_grade12 = sample_size) ∧
  (sample_grade10 = (ratio_grade10 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade11 = (ratio_grade11 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade12 = (ratio_grade12 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) :=
by
  sorry

end stratified_sampling_correct_l43_43610


namespace overall_average_of_25_results_l43_43138

theorem overall_average_of_25_results (first_12_avg last_12_avg thirteenth_result : ℝ) 
  (h1 : first_12_avg = 14) (h2 : last_12_avg = 17) (h3 : thirteenth_result = 78) :
  (12 * first_12_avg + thirteenth_result + 12 * last_12_avg) / 25 = 18 :=
by
  sorry

end overall_average_of_25_results_l43_43138


namespace eval_expr_at_3_l43_43336

theorem eval_expr_at_3 : (3^2 - 5 * 3 + 6) / (3 - 2) = 0 := by
  sorry

end eval_expr_at_3_l43_43336


namespace vector_relationship_l43_43457

variables {A B C D E : Type}
variables [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
variables (A B C D E : A)

-- Define the median condition
def is_median (A B C D : A) : Prop :=
  D = (B + C) / 2

-- Define the midpoint condition
def is_midpoint (D E : A) : Prop :=
  E = D / 2

-- Main theorem to prove the vector relationship
theorem vector_relationship (A B C D E : A) (h1 : is_median A B C D) (h2 : is_midpoint D E) :
  E - B = 3/4 * (A - B) - 1/4 * (A - C) :=
sorry

end vector_relationship_l43_43457


namespace probability_sum_is_3_l43_43100

theorem probability_sum_is_3:
  let balls : list ℕ := [1, 1, 1, 2, 2, 3] in
  ∑ ball1 in balls, ∑ ball2 in balls, if ball1 + ball2 = 3 then 1 else 0 = 12 →
  6 * 6 = 36 →
  (12 : ℚ) / 36 = (1 / 3 : ℚ) :=
by sorry

end probability_sum_is_3_l43_43100


namespace geom_series_first_term_l43_43572

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 10) 
  (h2 : a + a * r = 7) : 
  a = 10 * (1 - real.sqrt (3 / 10)) ∨ a = 10 * (1 + real.sqrt (3 / 10)) :=
  sorry

end geom_series_first_term_l43_43572


namespace problem_solution_l43_43824

def g (x : ℝ) := -3 * Real.sin (2 * Real.pi * x)

theorem problem_solution :
  let solutions := {x : ℝ | g (g (g x)) = g x ∧ -3 ≤ x ∧ x ≤ 3} in
  solutions.count = 48 :=
by
  sorry

end problem_solution_l43_43824


namespace cyclic_points_l43_43097

variables {A B C M B1 C1 H I : Point}
variables [Triangle ABC]
variables [PointOnSide M BC]
variables [IsEqualDistance MB MB1]
variables [IsEqualDistance MC MC1]
variables [Orthocenter H ABC]
variables [InCenter I MB1C1]

theorem cyclic_points : 
  PointsLieOnCircle {A, B1, H, I, C1} :=
by
  sorry

end cyclic_points_l43_43097


namespace evaluated_result_l43_43703

noncomputable def evaluate_expression (y : ℝ) (hy : y ≠ 0) : ℝ :=
  (18 * y^3) * (4 * y^2) * (1 / (2 * y)^3)

theorem evaluated_result (y : ℝ) (hy : y ≠ 0) : evaluate_expression y hy = 9 * y^2 :=
by
  sorry

end evaluated_result_l43_43703


namespace candy_bar_calories_unit_l43_43140

-- Definitions based on conditions
def calories_unit := "calories per candy bar"

-- There are 4 units of calories in a candy bar
def units_per_candy_bar : ℕ := 4

-- There are 2016 calories in 42 candy bars
def total_calories : ℕ := 2016
def number_of_candy_bars : ℕ := 42

-- The statement to prove
theorem candy_bar_calories_unit : (total_calories / number_of_candy_bars = 48) → calories_unit = "calories per candy bar" :=
by
  sorry

end candy_bar_calories_unit_l43_43140


namespace distance_between_A_and_B_is_45_kilometers_l43_43522

variable (speedA speedB : ℝ)
variable (distanceAB : ℝ)

noncomputable def problem_conditions := 
  speedA = 1.2 * speedB ∧
  ∃ (distanceMalfunction : ℝ), distanceMalfunction = 5 ∧
  ∃ (timeFixingMalfunction : ℝ), timeFixingMalfunction = (distanceAB / 6) / speedB ∧
  ∃ (increasedSpeedB : ℝ), increasedSpeedB = 1.6 * speedB ∧
  ∃ (timeA timeB timeB_new : ℝ),
    timeA = (distanceAB / speedA) ∧
    timeB = (distanceMalfunction / speedB) + timeFixingMalfunction + (distanceAB - distanceMalfunction) / increasedSpeedB ∧
    timeA = timeB

theorem distance_between_A_and_B_is_45_kilometers
  (speedA speedB distanceAB : ℝ) 
  (cond : problem_conditions speedA speedB distanceAB) :
  distanceAB = 45 :=
sorry

end distance_between_A_and_B_is_45_kilometers_l43_43522


namespace magnitude_2a_plus_b_l43_43434

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (1, 0)

def scale_vector (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

theorem magnitude_2a_plus_b : (real.sqrt ((-3)^2 + 2^2)) = real.sqrt 13 :=
by
  let a : ℝ × ℝ := vector_a
  let b : ℝ × ℝ := vector_b
  let vector_2a := scale_vector 2 a
  let result_vector := add_vectors vector_2a b
  have h : result_vector = (-3, 2) := sorry
  rw [h]
  sorry

end magnitude_2a_plus_b_l43_43434


namespace polynomial_continuous_l43_43602

theorem polynomial_continuous (P : ℝ → ℝ) [is_polynomial P] : continuous P := 
sorry

end polynomial_continuous_l43_43602


namespace parallel_vectors_dot_product_l43_43370

noncomputable def a : ℝ × ℝ := (-1, 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, m - 4)
noncomputable def c (m : ℝ) : ℝ × ℝ := (2 * m, 3)

theorem parallel_vectors_dot_product (m : ℝ) 
  (h_parallel : -1 * (m - 4) - 3 * m = 0) : 
  let b := b m in
  let c := c m in
  b.1 * c.1 + b.2 * c.2 = -7 :=
by sorry

end parallel_vectors_dot_product_l43_43370


namespace circumcircle_tangent_condition_l43_43806

namespace Geometry

open EuclideanGeometry

/-- 
In triangle ABC, points M and N are the midpoints of sides AC and BC respectively.
Prove that the circumcircle of △ CMN is tangent to side AB if and only if AB = (AC + BC) / sqrt(2). 
-/
theorem circumcircle_tangent_condition (A B C M N : Point) (hM : midpoint A C M) (hN : midpoint B C N) :
  (tangent (circumcircle (triangle.mk C M N)) (line_through A B)) ↔ (dist A B = (dist A C + dist B C) / real.sqrt 2) :=
sorry

end Geometry

end circumcircle_tangent_condition_l43_43806


namespace first_airplane_completes_first_l43_43152

-- Define points A, B, C, D
def A := ℝ
def B := ℝ
def C := ℝ
def D := ℝ

-- Define distances between points
variable (d : ℝ → ℝ → ℝ)
-- Assuming all distances are positive
axiom d_pos (x y : ℝ) : d x y > 0

-- Define the routes
def route1 := [A, B, D, C, A, D, B, C, A]
def route2 := [A, B, C, D, A, B, C, D, A, B, C, D, A]

-- Calculate total distance for a given route
def total_distance (route : List ℝ) : ℝ :=
  route.zip (route.tail ++ [route.head]).map (λ (x, y) => d x y).sum

-- Total distance for the first airplane
def S1 := total_distance d route1

-- Total distance for the second airplane
def S2 := total_distance d route2

-- Proof statement: The first airplane will complete the flight first
theorem first_airplane_completes_first (h : ∀ {x y z : ℝ}, d x y + d y z > d x z):
  S1 < S2 := by
  sorry

end first_airplane_completes_first_l43_43152


namespace intersection_curve_polar_radius_of_intersection_l43_43470

def line_l1 (t k : ℝ) : ℝ × ℝ := (2 + t, k * t)
def line_l2 (m k : ℝ) : ℝ × ℝ := (-2 + m, m / k)
def curve_C (x y : ℝ) : Prop := y ≠ 0 ∧ x^2 - y^2 = 4
def line_l3 (θ : ℝ) : ℝ×ℝ := (sqrt 2 - y, sqrt 2 - x)

theorem intersection_curve
  (k : ℝ) (t m x y : ℝ) 
  (h_l1 : line_l1 t k = (x, y)) 
  (h_l2 : line_l2 m k = (x, y)) :
  curve_C x y :=
sorry
  
theorem polar_radius_of_intersection 
  (θ : ℝ) (x y ρ : ℝ) 
  (h_l3 : line_l3 θ = (x, y)) 
  (h_curve : curve_C x y) :
  ρ^2 = 5 :=
sorry

end intersection_curve_polar_radius_of_intersection_l43_43470


namespace roses_in_vase_l43_43912

theorem roses_in_vase (initial_roses added_roses : ℕ) (h₀ : initial_roses = 10) (h₁ : added_roses = 8) : initial_roses + added_roses = 18 :=
by
  sorry

end roses_in_vase_l43_43912


namespace ab_divisibility_l43_43364

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end ab_divisibility_l43_43364


namespace gcd_765432_654321_l43_43166

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43166


namespace Mrs_Hilt_remaining_money_l43_43835

noncomputable def initial_money := 10
noncomputable def cost_truck := 3
noncomputable def cost_pencil_case := 2
noncomputable def cost_notebook := 1.50
noncomputable def cost_candy_bar := 0.75
noncomputable def discount_rate := 0.05

noncomputable def total_cost_before_discount := cost_truck + cost_pencil_case + cost_notebook + cost_candy_bar
noncomputable def discount := discount_rate * total_cost_before_discount
noncomputable def rounded_discount := Float.round (discount : Float)

noncomputable def remaining_money := initial_money - (total_cost_before_discount - rounded_discount)

theorem Mrs_Hilt_remaining_money : remaining_money = 3.11 := by
  sorry

end Mrs_Hilt_remaining_money_l43_43835


namespace boat_distance_against_stream_l43_43792

-- Define the speed of the boat in still water
def speed_boat_still : ℝ := 8

-- Define the distance covered by the boat along the stream in one hour
def distance_along_stream : ℝ := 11

-- Define the time duration for the journey
def time_duration : ℝ := 1

-- Define the speed of the stream
def speed_stream : ℝ := distance_along_stream - speed_boat_still

-- Define the speed of the boat against the stream
def speed_against_stream : ℝ := speed_boat_still - speed_stream

-- Define the distance covered by the boat against the stream in one hour
def distance_against_stream (t : ℝ) : ℝ := speed_against_stream * t

-- The main theorem: The boat travels 5 km against the stream in one hour
theorem boat_distance_against_stream : distance_against_stream time_duration = 5 := by
  sorry

end boat_distance_against_stream_l43_43792


namespace sum_is_square_l43_43502

noncomputable theory

def A (n : ℕ) : ℕ := 4 * (10^(2 * n) - 1) / 9
def B (n : ℕ) : ℕ := 8 * (10^n - 1) / 9

theorem sum_is_square (n : ℕ) :
  A n + 2 * B n + 4 = (2 * (10^n + 2) / 3)^2 := sorry

end sum_is_square_l43_43502


namespace isosceles_trapezoid_from_pentagon_l43_43345

theorem isosceles_trapezoid_from_pentagon :
  ∀ (ABCD : Type) [is_trapezoid ABCD] (AD BC AC : ℝ),
    isosceles_trapezoid ABCD AD BC AC → 
    AD > BC →
    (is_isosceles_triangle (△ ABC) AC) ∧ (is_isosceles_triangle (△ ADC) AC) → 
  ∃ (P : Type) [regular_pentagon P] (Diagonal : ℝ),
    is_pentagon_trapezoid P ABCD Diagonal :=
by
  intros ABCD is_trapezoid_inst AD BC AC isosceles_trap AD_gt_BC iso_triangles
  sorry

end isosceles_trapezoid_from_pentagon_l43_43345


namespace boat_length_in_steps_l43_43334

theorem boat_length_in_steps (L E S : ℝ) 
  (h1 : 250 * E = L + 250 * S) 
  (h2 : 50 * E = L - 50 * S) :
  L = 83 * E :=
by sorry

end boat_length_in_steps_l43_43334


namespace line_passes_through_fixed_point_l43_43837

theorem line_passes_through_fixed_point (k : ℝ) :
  ∀ x y : ℝ, 
  (2k - 1) * x - (k + 3) * y - (k - 11) = 0 ↔ (x, y) = (2, 3) :=
by
  sorry

end line_passes_through_fixed_point_l43_43837


namespace maintenance_cost_relation_maximize_average_profit_l43_43542

def maintenance_cost (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1400 * n - 1000

theorem maintenance_cost_relation :
  maintenance_cost 2 = 1800 ∧ maintenance_cost 5 = 6000 ∧
  (∀ n, n ≥ 2 → maintenance_cost n = 1400 * n - 1000) :=
by
  sorry

noncomputable def average_profit (n : ℕ) : ℝ :=
  if n < 2 then 0 else 60000 - (1 / n) * (137600 + 1400 * ((n - 1) * (n + 2) / 2) - 1000 * (n - 1))

theorem maximize_average_profit (n : ℕ) :
  n = 14 ↔ (average_profit n = 40700) :=
by
  sorry

end maintenance_cost_relation_maximize_average_profit_l43_43542


namespace trihedral_angle_cross_section_acute_l43_43970

theorem trihedral_angle_cross_section_acute {O A B C : Point} 
  (h1 : ∠AOX = 90) (h2 : ∠BOY = 90) (h3 : ∠COZ = 90) 
  (h_cross_section : ¬ O ∈ {A, B, C}) :
  is_acute_triangle (triangle.mk A B C) :=
sorry

end trihedral_angle_cross_section_acute_l43_43970


namespace gcd_765432_654321_l43_43173

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43173


namespace cube_triangle_area_sum_solution_l43_43901

def cube_vertex_triangle_area_sum (m n p : ℤ) : Prop :=
  m + n + p = 121 ∧
  (∀ (a : ℕ) (b : ℕ) (c : ℕ), a * b * c = 8) -- Ensures the vertices are for a 2*2*2 cube

theorem cube_triangle_area_sum_solution :
  cube_vertex_triangle_area_sum 48 64 9 :=
by
  unfold cube_vertex_triangle_area_sum
  split
  · exact rfl -- m + n + p = 121
  · intros a b c h
    sorry -- Conditions ensuring these m, n, p were calculated from a 2x2x2 cube

end cube_triangle_area_sum_solution_l43_43901


namespace geometry_problem_l43_43858

-- Definitions for geometrical entities
variable {Point : Type} -- type representing points

variable (Line : Type) -- type representing lines
variable (Plane : Type) -- type representing planes

-- Parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop) 
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Given entities
variables (m : Line) (n : Line) (α : Plane) (β : Plane)

-- Given conditions
axiom condition1 : perpendicular α β
axiom condition2 : perpendicular_line_plane m β
axiom condition3 : ¬ contained_in m α

-- Statement of the problem in Lean 4
theorem geometry_problem : parallel m α :=
by
  -- proof will involve using the axioms and definitions
  sorry

end geometry_problem_l43_43858


namespace gcd_765432_654321_l43_43168

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43168


namespace no_intersection_range_chord_length_m_zero_l43_43471

-- Define the parametric equations and the polar equation of curve C as given conditions
def line_parametric (m : ℝ) : ℝ → ℝ × ℝ :=
  λ t, (1/2 * t, m + (ℝ.sqrt 3)/2 * t)

def C_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*θ.cos - 4 = 0

-- Define the Cartesian form of curve C for later use
def C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 5

-- State the first proof problem regarding the range of m for no intersections
theorem no_intersection_range (m : ℝ) :
  (m < -ℝ.sqrt 3 - 2 * ℝ.sqrt 5 ∨ m > ℝ.sqrt 3 + 2 * ℝ.sqrt 5) ↔
  ∀ t : ℝ, ¬ C_cartesian (1/2 * t) (m + (ℝ.sqrt 3)/2 * t) :=
sorry

-- State the second proof problem about the length of the chord when m = 0
theorem chord_length_m_zero :
  ∀ ρ : ℝ, C_polar ρ (ℝ.pi/3) ↔
  ∀ t : ℝ, let p := line_parametric 0 t in
    let distance := (p.1 - ρ.1)^2 + (p.2 - ρ.2)^2 in
    distance = 17 :=
sorry

end no_intersection_range_chord_length_m_zero_l43_43471


namespace percentage_increase_is_25_l43_43122

-- Variables
variable (P : ℝ)

-- Conditions
def original_price := 160
def coupon_discount_factor := 0.75
def final_price := 150
def increased_price := original_price + (original_price * P / 100)
def discounted_price := coupon_discount_factor * increased_price

-- Theorem
theorem percentage_increase_is_25 (h : discounted_price = final_price) : P = 25 := by
  sorry

end percentage_increase_is_25_l43_43122


namespace gcd_765432_654321_l43_43176

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l43_43176


namespace gcd_proof_l43_43191

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43191


namespace find_k_if_parallel_l43_43070

def vector := ℝ × ℝ

def a : vector := (2, 1)
def b : vector := (1, 2)
def v1 : vector := (5, 4)
def v2 (k : ℝ) : vector := (1 + k, 1 / 2 + 2 * k)

theorem find_k_if_parallel (k : ℝ) : 
  (2 * (fst a) + fst b = 1 + k) ∧ (2 * (snd a) + snd b = 1 / 2 + 2 * k) → k = 1 / 4 :=
by
  intros h
  sorry

end find_k_if_parallel_l43_43070


namespace min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l43_43765

noncomputable def min_trials_sum_of_15 : ℕ :=
  15

noncomputable def min_trials_sum_at_least_15 : ℕ :=
  8

theorem min_number_of_trials_sum_15 (x : ℕ) :
  (∀ (x : ℕ), (103/108 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_of_15) := sorry

theorem min_number_of_trials_sum_at_least_15 (x : ℕ) :
  (∀ (x : ℕ), (49/54 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_at_least_15) := sorry

end min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l43_43765


namespace Dan_tshirts_total_l43_43675

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end Dan_tshirts_total_l43_43675


namespace problem_l43_43444

theorem problem (x : ℝ) (h : x * Real.log 2 / Real.log 3 = 1) : 2^x + 2^(-x) = 10 / 3 :=
sorry

end problem_l43_43444


namespace keith_initial_cards_l43_43038

theorem keith_initial_cards (new_cards : ℕ) (cards_after_incident : ℕ) (total_cards : ℕ) :
  new_cards = 8 →
  cards_after_incident = 46 →
  total_cards = 2 * cards_after_incident →
  (total_cards - new_cards) = 84 :=
by
  intros
  sorry

end keith_initial_cards_l43_43038


namespace range_of_g_l43_43713

-- Define the function g(t)
def g (t : ℝ) : ℝ := (t^2 + 1/2 * t) / (t^2 + 1)

-- Define the range predicate to state that a value y is in the range of g
def in_range (y : ℝ) : Prop :=
  ∃ t : ℝ, y = g t

-- Define the closed interval [-1/4, 1/4]
def closed_interval (a b : ℝ) (y : ℝ) : Prop :=
  a ≤ y ∧ y ≤ b

-- The main theorem to be proven
theorem range_of_g : ∀ y : ℝ, closed_interval (-1/4) (1/4) y ↔ in_range y :=
by
  sorry

end range_of_g_l43_43713


namespace chinese_exam_paper_probability_l43_43580

theorem chinese_exam_paper_probability (chinese tibetan english : ℕ) 
  (h_chinese : chinese = 2) 
  (h_tibetan : tibetan = 3) 
  (h_english : english = 1) 
  (total_papers : chinese + tibetan + english = 6) :
  (2 : ℚ) / 6 = 1 / 3 :=
by {
  rw [h_chinese, h_tibetan, h_english, total_papers],
  norm_num,
  sorry
}

end chinese_exam_paper_probability_l43_43580


namespace cannot_hold_statement_l43_43643

-- Definitions and Conditions
def correct_rate (f : ℕ → ℚ) (n : ℕ) := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} 

-- Problem Statement
theorem cannot_hold_statement (f : ℕ → ℚ) :
  correct_rate f 8 → correct_rate f 9 → correct_rate f 10 →
  ¬(f 8 < f 9 ∧ f 9 = f 10) :=
by
  sorry

end cannot_hold_statement_l43_43643


namespace largest_b_no_lattice_points_l43_43692

theorem largest_b_no_lattice_points (
  (m : ℝ) (b : ℝ) 
  (h1 : ∀ (x : ℕ), 0 < x ∧ x ≤ 150 → ¬∃ (y : ℕ), y = m * x + 3)
  (h2 : 0.5 < m ∧ m < b)
) : b = 76 / 151 :=
sorry

end largest_b_no_lattice_points_l43_43692


namespace solution_set_l43_43883

-- Define the conditions as hypotheses
def system_solution : Prop :=
  ∃ x y : ℝ, (x + y = 1) ∧ (x^2 - y^2 = 9) ∧ (x = 5 ∧ y = -4)

-- Prove that the solution set for the given system of equations is {(5, -4)}
theorem solution_set : system_solution :=
begin
  sorry
end

end solution_set_l43_43883


namespace power_function_is_x_cubed_l43_43780

/-- Define the power function and its property -/
def power_function (a : ℕ) (x : ℝ) : ℝ := x ^ a

/-- The given condition that the function passes through the point (3, 27) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 3 = 27

/-- Prove that the power function is x^3 -/
theorem power_function_is_x_cubed (f : ℝ → ℝ)
  (h : passes_through_point f) : 
  f = fun x => x ^ 3 := 
by
  sorry -- proof to be filled in

end power_function_is_x_cubed_l43_43780


namespace find_n_l43_43715

noncomputable def sum_sqrt (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (∑ k in Finset.range n, real.sqrt (((2 * k.val + 1 : ℤ)^2) + (a k)^2))

theorem find_n (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∑ k in Finset.range n, a k = 17)
  (h2 : ∀ k, 0 < a k) :
  ∃ (n : ℕ), n = 12 ∧ (∃ m : ℤ, sum_sqrt n a = m) :=
sorry

end find_n_l43_43715


namespace time_to_groom_chihuahua_l43_43811

variables (time_rottweiler time_border_collie time_chihuahua total_time : ℕ)
variables (num_rottweilers num_border_collies : ℕ)

-- Conditions defined as hypotheses
def conditions := 
  (time_rottweiler = 20) ∧ 
  (time_border_collie = 10) ∧ 
  (num_rottweilers = 6) ∧ 
  (num_border_collies = 9) ∧ 
  (total_time = 255) ∧
  (total_time = num_rottweilers * time_rottweiler + num_border_collies * time_border_collie + time_chihuahua)

-- Statement to prove
theorem time_to_groom_chihuahua (h : conditions): time_chihuahua = 45 :=
by sorry

end time_to_groom_chihuahua_l43_43811


namespace gcd_765432_654321_l43_43181

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43181


namespace margie_can_drive_150_miles_l43_43071

theorem margie_can_drive_150_miles
  (cost_per_gallon : ℕ)
  (miles_per_gallon : ℕ)
  (total_money : ℕ)
  (number_of_gallons : ℕ := total_money / cost_per_gallon)
  (total_miles : ℕ := number_of_gallons * miles_per_gallon) :
  cost_per_gallon = 5 → 
  miles_per_gallon = 25 → 
  total_money = 30 → 
  total_miles = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [Nat.div_self, Nat.mul_comm, Nat.div_eq_of_eq_mul_left (by norm_num) rfl, add_mul]
  norm_num
  sorry

end margie_can_drive_150_miles_l43_43071


namespace max_x_value_l43_43155

variables {x y : ℝ}
variables (data : list (ℝ × ℝ))
variables (linear_relation : ℝ → ℝ → Prop)

def max_y : ℝ := 10

-- Given conditions
axiom linear_data :
  (data = [(16, 11), (14, 9), (12, 8), (8, 5)]) ∧
  (∀ (p : ℝ × ℝ), p ∈ data → linear_relation p.1 p.2)

-- Prove the maximum value of x for which y ≤ max_y
theorem max_x_value (h : ∀ (x y : ℝ), linear_relation x y → y = 11 - (16 - x) / 3):
  ∀ (x : ℝ), (∃ y : ℝ, linear_relation x y) → y ≤ max_y → x ≤ 15 :=
sorry

end max_x_value_l43_43155


namespace ice_cream_volume_is_correct_l43_43110

noncomputable def total_ice_cream_volume (r h : ℝ) : ℝ :=
  let V_cone := (1 / 3) * π * (r ^ 2) * h
  let V_hemisphere := (2 / 3) * π * (r ^ 3)
  V_cone + V_hemisphere

theorem ice_cream_volume_is_correct : total_ice_cream_volume 3 10 = 48 * π := 
by
  -- proof skipped
  sorry

end ice_cream_volume_is_correct_l43_43110


namespace trains_crossing_time_l43_43937

def length_first_train : ℕ := 200
def length_second_train : ℕ := 800
def speed_first_train_kmh : ℕ := 60
def speed_second_train_kmh : ℕ := 40
def kmh_to_mps (v_kmh : ℕ) : ℚ := (5 / 18) * v_kmh

theorem trains_crossing_time :
  let relative_speed_mps := kmh_to_mps (speed_first_train_kmh + speed_second_train_kmh)
      combined_length := length_first_train + length_second_train in
  combined_length / relative_speed_mps = 36 :=
by
  sorry

end trains_crossing_time_l43_43937


namespace gcd_proof_l43_43196

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l43_43196


namespace collinear_points_minimize_distance_l43_43819

-- Definitions and assumptions
variables {a b : ℝ}  -- Let a and b be non-collinear non-zero real vectors
variable (t : ℝ)  -- real number t
def OA := a
def OB := t * b
def OC := (1 / 3) * (a + b)

-- Question 1: Proof statement
theorem collinear_points (a b : ℝ) (h : a ≠ b) : t = 1 / 2 ↔ collinear OA OB OC :=
sorry

-- Additional conditions for question 2
variables {x : ℝ}
def a_norm := 1
def b_norm := 1
def angle_ab := 2 * Math.pi / 3

-- Question 2: Proof statement
theorem minimize_distance (x : ℝ) : 
  x = -1/2 ↔ 
  (|a - x * b| = (Real.sqrt 3 / 2)) :=
sorry

end collinear_points_minimize_distance_l43_43819


namespace remainder_of_n_squared_plus_4n_plus_5_l43_43770

theorem remainder_of_n_squared_plus_4n_plus_5 {n : ℤ} (h : n % 50 = 1) : (n^2 + 4*n + 5) % 50 = 10 :=
by
  sorry

end remainder_of_n_squared_plus_4n_plus_5_l43_43770


namespace distance_vertex_cone_to_vertex_cube_l43_43295

-- Definitions of the given conditions
def edge_length_cube : ℝ := 3
def diameter_cone : ℝ := 8
def height_cone : ℝ := 24
def interior_diagonal_cube : ℝ := (3 * real.sqrt 3)

-- Given that one interior diagonal of the cube coincides with the axis of the cone
theorem distance_vertex_cone_to_vertex_cube :
  let distance : ℝ := 6 * real.sqrt 6 - real.sqrt 3 in
  distance_vertex_cone_to_vertex_cube == distance := sorry

end distance_vertex_cone_to_vertex_cube_l43_43295


namespace proofProblem_l43_43108

noncomputable def f (x: ℝ) := 2^(2*x) - 2^(x + 1) + 2

def domainM := {x : ℝ | x ≤ 1}

def statement1 := domainM = set.Icc 0 1
def statement2 := domainM = set.Iio 1
def statement3 := set.Icc 0 1 ⊆ domainM
def statement4 := domainM ⊆ set.Iic 1
def statement5 := 1 ∈ domainM
def statement6 := -1 ∈ domainM

theorem proofProblem : 
  (f '' domainM = set.Icc 1 2) →
  (3 + 4 + 5 + 6 - (¬statement1 → 1) - (¬statement2 → 1) = 4) :=
by intro h; sorry

end proofProblem_l43_43108


namespace range_of_a_l43_43440

theorem range_of_a (a : ℝ) : (∃ (x : ℝ), x^2 ≤ a) → a ∈ set.Ici (0 : ℝ) :=
by
  sorry

end range_of_a_l43_43440


namespace analytic_expression_of_f_maximum_value_m_l43_43747

-- Definition of the linear function and its properties
def linear_function (f : ℝ → ℝ) := ∃ k b : ℝ, (k ≠ 0) ∧ (∀ x : ℝ, f(x) = k * x + b) ∧ (∀ x : ℝ, f(k * 1 + b) = -1) ∧ (∀ p : ℝ × ℝ, f(p.1) = p.2 ↔ p.2 = p.1)

-- Sequence definition and recurrence relation
def sequence_a (a : ℕ → ℝ) := a 1 = 1 ∧ (∀ n : ℕ, n ≥ 2 → (a (n + 1)) / (a n) - (a n) / (a (n - 1)) = 1)

-- Series definition
def series_S (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = ∑ k in Finset.range (n + 1), a k / (k + 2)!

-- Theorem statements
theorem analytic_expression_of_f :
  ∀ f : ℝ → ℝ, linear_function f → (∃ k b : ℝ, (f = fun x => k * x + b) ∧ (k = 1 ∧ b = -1)) :=
by sorry

theorem maximum_value_m :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ), sequence_a a → series_S a S → (∀ n : ℕ, S n > 0) :=
by sorry

end analytic_expression_of_f_maximum_value_m_l43_43747


namespace polynomial_remainder_l43_43595

-- Define the divisor polynomial
def divisor : Polynomial ℤ := Polynomial.Coeff (λ n, if n = 2 then 1 else if n = 1 then 5 else if n = 0 then -3 else 0)

-- Define the dividend polynomial
def dividend : Polynomial ℤ := Polynomial.Coeff (λ n, if n = 4 then 3 else if n = 3 then 8 else if n = 2 then -29 else if n = 1 then -17 else if n = 0 then 34 else 0)

theorem polynomial_remainder :
  let r := Polynomial.rem dividend divisor in
  r = Polynomial.Coeff (λ n, if n = 1 then 79 else if n = 0 then -11 else 0) :=
by
  sorry

end polynomial_remainder_l43_43595


namespace part1_zeros_m0_part2_m_range_l43_43418

def f (x : ℝ) (m : ℝ) : ℝ :=
  x^2 - (2 * m + 1) * x + m * (m + 1)

theorem part1_zeros_m0 :
  {x : ℝ | f x 0 = 0} = {0, 1} :=
by {
  sorry
}

theorem part2_m_range :
  {m : ℝ | ∃ x ∈ (1 : ℝ) .. 3, f x m = 0 ∧ (∀ y ∈ (1 : ℝ) .. 3, (y ≠ x → f y m ≠ 0))} =
  set.Ioo 0 1 ∪ set.Ico 2 3 :=
by {
  sorry
}

end part1_zeros_m0_part2_m_range_l43_43418


namespace fuel_a_added_l43_43247

theorem fuel_a_added (capacity : ℝ) (ethanolA : ℝ) (ethanolB : ℝ) (total_ethanol : ℝ) (x : ℝ) : 
  capacity = 200 ∧ ethanolA = 0.12 ∧ ethanolB = 0.16 ∧ total_ethanol = 28 →
  0.12 * x + 0.16 * (200 - x) = 28 → x = 100 :=
sorry

end fuel_a_added_l43_43247


namespace entire_set_not_necessarily_divisible_l43_43728

-- Define the concept of divisibility of points.
def divisible (points : Finset (Finset Point) → Prop :=
  ∃ (Δ : Triangle), 
    (∀ redPoint ∈ points.red, redPoint ∈ Δ.interior) ∧ 
    (∀ greenPoint ∈ points.green, greenPoint ∉ Δ.interior)

-- Define the original problem statement in Lean 4.
theorem entire_set_not_necessarily_divisible (A : Finset Point) :
  (∀ (s : Finset Point), s.card = 1000 → divisible s) → ¬divisible A :=
by 
  sorry

end entire_set_not_necessarily_divisible_l43_43728


namespace shenzhen_vaccination_count_l43_43101

theorem shenzhen_vaccination_count :
  2410000 = 2.41 * 10^6 :=
  sorry

end shenzhen_vaccination_count_l43_43101


namespace minimum_value_of_quadratic_l43_43419

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = -p / 2 ∧ (∀ y : ℝ, (y - x) ^ 2 + 2*q ≥ (x ^ 2 + p * x + 2*q)) :=
by
  sorry

end minimum_value_of_quadratic_l43_43419


namespace stamp_blocks_inequalities_l43_43766

noncomputable def b (n : ℕ) : ℕ := sorry

theorem stamp_blocks_inequalities (n : ℕ) (m : ℕ) (hn : 0 < n) :
  ∃ c d : ℝ, c = 2 / 7 ∧ d = (4 * m^2 + 4 * m + 40) / 5 ∧
    (1 / 7 : ℝ) * n^2 - c * n ≤ b n ∧ 
    b n ≤ (1 / 5 : ℝ) * n^2 + d * n := 
  sorry

end stamp_blocks_inequalities_l43_43766


namespace sum_of_ages_l43_43035

-- Define the ages as four distinct single-digit positive integers
variables {a b c d : ℕ}

-- Conditions that two pairs multiply to specific products
def conditions : Prop :=
  a * b = 28 ∧ c * d = 45 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  1 ≤ a ∧ a < 10 ∧ 1 ≤ b ∧ b < 10 ∧ 1 ≤ c ∧ c < 10 ∧ 1 ≤ d ∧ d < 10

-- Prove that the sum of these four ages is 25
theorem sum_of_ages (h : conditions) : a + b + c + d = 25 :=
by
  sorry

end sum_of_ages_l43_43035


namespace length_AD_l43_43369

theorem length_AD {O A B C D : Point} (circumcenter : isCircumcenter O A B C)
  (midpoint : isMidpoint D B C) (dot_product : vectorDot (vector O A) (vector A D) = 4)
  (len_BC : dist B C = 2 * Real.sqrt 6) :
  dist A D = Real.sqrt 2 := by
  sorry

end length_AD_l43_43369


namespace sum_of_areas_of_triangles_in_cube_l43_43894

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l43_43894


namespace length_of_largest_side_l43_43731

-- Definition of a triangle with sides that are consecutive natural numbers
def is_consecutive_natural_sides (a b c : ℕ) : Prop :=
  (a + 1 = b ∧ b + 1 = c) ∨ (b + 1 = a ∧ a + 1 = c) ∨ (c + 1 = a ∧ a + 1 = b)

-- Given conditions
variable (a b c : ℕ)
variable (ha : a = 3)
variable (hconsec : is_consecutive_natural_sides a b c)
variable (max_angle_is_double_smallest : 1 < b ∧ 1 < c → ∀ (A B C : ℝ), A = 2 * B → (a = fin.atan A + fin.atan C ∧ max B C = A * 2))

-- Question to prove
theorem length_of_largest_side (hconsec: is_consecutive_natural_sides a b c) (ha : a = 3) (max_angle_is_double_smallest: 1 < b ∧ 1 < c → ∀ (A B C : ℝ), A = 2 * B → (a = fin.atan A + fin.atan C ∧ max B C = A * 2)): (∃ d : ℕ, d = 6) :=
  sorry

end length_of_largest_side_l43_43731


namespace find_x_l43_43621

-- Define the conditions
def is_purely_imaginary (z : Complex) : Prop :=
  z.re = 0

-- Define the problem
theorem find_x (x : ℝ) (z : Complex) (h1 : z = Complex.ofReal (x^2 - 1) + Complex.I * (x + 1)) (h2 : is_purely_imaginary z) : x = 1 :=
sorry

end find_x_l43_43621


namespace conjugate_of_z_l43_43501

noncomputable def z : ℂ := (2 + complex.i) / complex.i

theorem conjugate_of_z : complex.conj z = 1 + 2 * complex.i := by
  sorry

end conjugate_of_z_l43_43501


namespace correct_fill_l43_43939

/- Define the conditions and the statement in Lean 4 -/
def sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

/- The condition is that the phrase without the number has 21 characters -/
def initial_length : ℕ := 21

/- Define the term "тридцать две" as the correct number to fill the blank -/
def correct_number := "тридцать две"

/- The target phrase with the correct number filled in -/
def target_sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ " ++ correct_number ++ " БУКВЫ"

/- Prove that the correct number fills the blank correctly -/
theorem correct_fill :
  (String.length target_sentence = 38) :=
by
  /- Convert everything to string length and verify -/
  sorry

end correct_fill_l43_43939


namespace sum_of_areas_of_triangles_in_cube_l43_43895

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l43_43895


namespace sum_of_zeroes_transformed_polynomial_l43_43505

-- Given conditions:
variables {S : ℝ} {f : ℝ → ℝ}
-- f(x) is a polynomial of degree 2016 with 2016 zeroes and the sum of zeroes is S.

-- The proof problem:
theorem sum_of_zeroes_transformed_polynomial :
  (sum_of_zeroes f 2016 S) →
  sum_of_zeroes (λ x, f (2 * x - 3)) 2016 ((1/2) * S + 3024) :=
sorry

-- Definitions to support theorem (depending on the actual implementation of these functions)
def sum_of_zeroes (f : ℝ → ℝ) (n : ℕ) (sum : ℝ) : Prop :=
  ∃ r : fin n → ℝ, (∀ i, f (r i) = 0) ∧ (finset.univ.sum r = sum)

end sum_of_zeroes_transformed_polynomial_l43_43505


namespace gcd_of_765432_and_654321_l43_43202

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l43_43202


namespace number_of_storks_joined_l43_43145

theorem number_of_storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (total_birds_and_storks : ℕ) 
    (h1 : initial_birds = 3) (h2 : initial_storks = 4) (h3 : total_birds_and_storks = 13) : 
    (total_birds_and_storks - (initial_birds + initial_storks)) = 6 := 
by
  sorry

end number_of_storks_joined_l43_43145


namespace negation_of_existence_l43_43873

theorem negation_of_existence : 
  (¬ ∃ x_0 : ℝ, (x_0 + 1 < 0) ∨ (x_0^2 - x_0 > 0)) ↔ ∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0) := 
by
  sorry

end negation_of_existence_l43_43873


namespace distance_point_P_l43_43722

/--
Four two-inch squares are placed with their bases on a line.
The second square from the left is lifted out, rotated 45 degrees, then centered 
and lowered back until it touches its adjacent squares on both sides.
Point P is defined as the top vertex of this rotated square.
We are to prove that the distance from point P to the line is 1 + sqrt(2) inches.
-/
theorem distance_point_P (a : ℝ) : 
  (∀ s, s = 2) →
  (P.is_Rotated s 45 P.height (1 + sqrt 2)) := 
by sorry

end distance_point_P_l43_43722


namespace additional_track_length_l43_43286

theorem additional_track_length (elevation_gain : ℝ) (orig_grade new_grade : ℝ) (Δ_track : ℝ) :
  elevation_gain = 800 ∧ orig_grade = 0.04 ∧ new_grade = 0.015 ∧ Δ_track = ((elevation_gain / new_grade) - (elevation_gain / orig_grade)) ->
  Δ_track = 33333 :=
by sorry

end additional_track_length_l43_43286


namespace rectangle_problem_l43_43570

-- Definition of the vertices conditions
def vertex_condition (y : ℝ) : Prop :=
  ∃ (x1 x2 x3 x4 y1 y2 : ℝ), 
  x1 = -2 ∧ x2 = 6 ∧ x3 = -2 ∧ x4 = 6 ∧ y1 = y ∧ y2 = 2

-- Definition of the area condition
def area_condition (y : ℝ) : Prop :=
  8 * (y - 2) = 64

-- Definition of the perimeter condition
def perim_condition (y : ℝ) : Prop :=
  2 * (8 + (y - 2)) = 32

-- The overarching condition combining all
def combined_condition (y : ℝ) : Prop :=
  vertex_condition(y) ∧ area_condition(y) ∧ perim_condition(y) ∧ y > 0

-- The final statement to prove
theorem rectangle_problem (y : ℝ) (h : combined_condition y) : y = 10 :=
  sorry

end rectangle_problem_l43_43570


namespace only_n_for_polynomial_quotient_l43_43317

noncomputable def P_n (n : ℕ) (x y z : ℚ) : ℚ :=
  (x - y)^(2 * n) * (y - z)^(2 * n) + 
  (y - z)^(2 * n) * (z - x)^(2 * n) + 
  (z - x)^(2 * n) * (x - y)^(2 * n)

noncomputable def Q_n (n : ℕ) (x y z : ℚ) : ℚ :=
  ((x - y)^(2 * n) + (y - z)^(2 * n) + (z - x)^(2 * n))^(2 * n)

theorem only_n_for_polynomial_quotient (n : ℕ) (h : n > 0) : 
  (∃ (f : ℚ → ℚ → ℚ → ℚ) (Hf : is_polynomial f), 
    ∀ x y z : ℚ, Q_n n x y z = f x y z * P_n n x y z) ↔ n = 1 :=
by
  sorry

end only_n_for_polynomial_quotient_l43_43317


namespace quadratic_inequality_l43_43877

variable {α : Type*} [LinearOrder α]

noncomputable def quadratic_function (f : α → α) : Prop :=
  ∃ (a b c : α), a > 0 ∧ ∀ x, f(x) = a * x^2 + b * x + c

theorem quadratic_inequality (f : α → α) (h1 : quadratic_function f) (h2 : ∀ x, f(x) = f(4 - x))
  (a : α) : f(2 - a^2) < f(1 + a - a^2) → a < 1 :=
sorry

end quadratic_inequality_l43_43877


namespace is_diff_solution_eq_2x4_diff_solution_eq_4xm_diff_solution_4x_ab_plus_a_3_diff_solution_4x_mn_plus_m_diff_solution_neg2x_mn_plus_n_final_proof_l43_43157

-- 1. Prove that 2x = 4 is a "difference solution equation"
theorem is_diff_solution_eq_2x4 (x: ℝ) : (∃ x, x = (4/2)) ↔ (x = 4-2) := sorry

-- 2. Prove that if 4x = m is a "difference solution equation", then m = 16/3
theorem diff_solution_eq_4xm (x m: ℝ) (h: x = m - 4): (x = m / 4) → m = 16 / 3 :=
sorry

-- 3. Prove that if 4x = ab + a is a "difference solution equation", then 3(ab + a) = 16
theorem diff_solution_4x_ab_plus_a_3 (x a b: ℝ) (h: x = ab + a - 4): 
  (x = (ab + a) / 4) → 3 * (ab + a) = 16 := 
sorry

-- 4a. Prove that if 4x = mn + m and x = (mn + m) / 4, then 3(mn + m) = 16
theorem diff_solution_4x_mn_plus_m (x m n: ℝ) (h1: x = mn + m - 4) :
  (x = (mn + m) / 4) → 3 * (mn + m) = 16 :=
sorry

-- 4b. Prove that if -2x = mn + n and x = -(mn + n) / 2, then 9(mn + n)^2 = 16
theorem diff_solution_neg2x_mn_plus_n (x m n: ℝ) (h2: x = mn + n + 2) :
  (x = -(mn + n) / 2) → 9 * (mn + n) ^ 2 = 16 :=
sorry

-- 4c. Prove that given the above, 3(mn + m) - 9(mn + n)^2 = 0
theorem final_proof (m n: ℝ) (h1: 3 * (mn + m) = 16) (h2: 9 * (mn + n) ^ 2 = 16):
  3*(mn + m) - 9*(mn + n) ^ 2 = 0 :=
by {
  rw [h1, h2],
  exact sub_self 16,
}

end is_diff_solution_eq_2x4_diff_solution_eq_4xm_diff_solution_4x_ab_plus_a_3_diff_solution_4x_mn_plus_m_diff_solution_neg2x_mn_plus_n_final_proof_l43_43157


namespace charlie_share_l43_43947

theorem charlie_share (A B C : ℕ) 
  (h1 : (A - 10) * 18 = (B - 20) * 11)
  (h2 : (A - 10) * 24 = (C - 15) * 11)
  (h3 : A + B + C = 1105) : 
  C = 495 := 
by
  sorry

end charlie_share_l43_43947


namespace different_colors_at_minus_plus_1990_l43_43331

open Int

/-
Given: 
1. A function f: ℤ → Fin 100 that maps each integer to one of the 100 colors.
2. All 100 colors are used by f.
3. For any two intervals [a, b] and [c, d] of the same length, if f(a) = f(c) and f(b) = f(d), 
   then f(a + x) = f(c + x) for all 0 ≤ x ≤ b - a.
Goal: 
   Prove that f(-1990) ≠ f(1990).
-/

noncomputable def f : ℤ → Fin 100 := sorry -- Function mapping integers to one of 100 colors

axiom color_all_used : ∀ n, ∃ x, f x = n -- All 100 colors are used

axiom color_same_condition : ∀ (a b c d : ℤ), 
  b - a = d - c → (f a = f c → f b = f d → ∀ x, 0 ≤ x ∧ x ≤ b - a → f (a + x) = f (c + x))

theorem different_colors_at_minus_plus_1990 : f (-1990) ≠ f (1990) :=
sorry

end different_colors_at_minus_plus_1990_l43_43331


namespace investment_value_change_l43_43367

theorem investment_value_change (k m : ℝ) : 
  let increaseFactor := 1 + k / 100
  let decreaseFactor := 1 - m / 100 
  let overallFactor := increaseFactor * decreaseFactor 
  let changeFactor := overallFactor - 1
  let percentageChange := changeFactor * 100 
  percentageChange = k - m - (k * m) / 100 := 
by 
  sorry

end investment_value_change_l43_43367


namespace gcd_765432_654321_l43_43167

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l43_43167


namespace john_apartment_number_l43_43030

variable (k d m : ℕ)

theorem john_apartment_number (h1 : k = m) (h2 : d + m = 239) (h3 : 10 * (k - 1) + 1 ≤ d) (h4 : d ≤ 10 * k) : d = 217 := 
by 
  sorry

end john_apartment_number_l43_43030


namespace complex_number_real_implies_m_is_5_l43_43777

theorem complex_number_real_implies_m_is_5 (m : ℝ) (h : m^2 - 2 * m - 15 = 0) : m = 5 :=
  sorry

end complex_number_real_implies_m_is_5_l43_43777


namespace ratio_of_max_min_distance_l43_43400

variables (a b c : ℝ^2) 
variables (M m : ℝ)

-- Define the conditions as hypotheses
axiom h1 : ‖a‖ = 2
axiom h2 : ‖b‖ = 2
axiom h3 : inner a b = 2
axiom h4 : inner c (a + 2 • b - 2 • c) = 2

-- Define the expressions for the maximum and minimum distances
def a_minus_c := λ (c : ℝ^2), ‖a - c‖

theorem ratio_of_max_min_distance : 
  (M = Sup { d | ∃ c, d = a_minus_c c }) →
  (m = Inf { d | ∃ c, d = a_minus_c c }) →
  M / m = (5 + Real.sqrt 21) / 2 :=
by
  intro hM hM
  sorry

end ratio_of_max_min_distance_l43_43400


namespace gcd_765432_654321_l43_43184

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l43_43184


namespace find_b_l43_43368

-- Define the given vectors and conditions
def a : ℝ × ℝ := (1, -2)

-- |b| = 2 * sqrt 5
def magnitude_b (b : ℝ × ℝ) : Prop :=
  ∥b∥ = 2 * Real.sqrt 5

-- a is parallel to b
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Final proposition to be proved
theorem find_b (b : ℝ × ℝ) : magnitude_b b ∧ parallel a b → b = (2, -4) ∨ b = (-2, 4) :=
by
  sorry

end find_b_l43_43368


namespace bugs_meet_again_l43_43154

def radius_large_circle : ℝ := 7
def radius_small_circle : ℝ := 3
def speed_large_circle : ℝ := 4 * Real.pi
def speed_small_circle : ℝ := 3 * Real.pi

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def time_to_complete_circle (circumference speed : ℝ) : ℝ := circumference / speed

def lcm (a b : ℝ) : ℝ := sorry  -- Assume we have a function to compute LCM for reals

theorem bugs_meet_again : 
  lcm (time_to_complete_circle (circumference radius_large_circle) speed_large_circle) 
      (time_to_complete_circle (circumference radius_small_circle) speed_small_circle) = 14 := 
sorry

end bugs_meet_again_l43_43154


namespace original_function_l43_43000

def f (x : ℝ) : ℝ := sin (x - π / 4)

theorem original_function :
  ∃ g : ℝ → ℝ, (∀ x, f (2 * (x + π / 3)) = sin (x - π / 4)) ∧ g x = sin (x / 2 + π / 12) :=
sorry

end original_function_l43_43000


namespace five_times_seven_divided_by_ten_l43_43234

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end five_times_seven_divided_by_ten_l43_43234


namespace line_parallel_to_plane_sufficient_not_necessary_l43_43131

-- Define the conditions of the problem and required statement

-- Definitions:
-- Line l does not intersect plane α
def LineDoesNotIntersectPlane (l : Line) (α : Plane) : Prop := 
  ∀ p : Point, ¬ (p ∈ l ∧ p ∈ α)

-- Line l is coplanar with infinite number of lines within plane α
def LineIsCoplanarWithInfiniteLines (l : Line) (α : Plane) : Prop := 
  ∃ (f : Nat → Line), (∀ n : Nat, ∀ p : Point, ¬ (p ∈ l ∧ p ∈ (f n))) ∧ (∀ n : Nat, ∀ m : Nat, m ≠ n → ∀ p : Point, (p ∈ (f n) ↔ p ∈ α))

-- The statement we want to prove: Line l parallel to Plane α
def LineParallelToPlane (l : Line) (α : Plane) : Prop :=
  LineDoesNotIntersectPlane l α

-- Proof statement:
theorem line_parallel_to_plane_sufficient_not_necessary (l : Line) (α : Plane) :
  (LineParallelToPlane l α → LineIsCoplanarWithInfiniteLines l α) ∧ 
  (¬ (LineIsCoplanarWithInfiniteLines l α → LineParallelToPlane l α)) :=
sorry

end line_parallel_to_plane_sufficient_not_necessary_l43_43131


namespace amplitude_period_phase_l43_43414

noncomputable def f (x : ℝ) := 3 * Real.sin (3 * x + Real.pi / 3)

theorem amplitude_period_phase :
  (∃ A T φ : ℝ, A = 3 ∧ T = 2 * Real.pi / 3 ∧ φ = Real.pi / 3) :=
by
  use 3, 2 * Real.pi / 3, Real.pi / 3
  exact ⟨rfl, rfl, rfl⟩

end amplitude_period_phase_l43_43414


namespace find_subtracted_value_l43_43965

theorem find_subtracted_value (x y : ℕ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 :=
by
  sorry

end find_subtracted_value_l43_43965


namespace count_total_coins_l43_43916

theorem count_total_coins (quarters nickels : Nat) (h₁ : quarters = 4) (h₂ : nickels = 8) : quarters + nickels = 12 :=
by sorry

end count_total_coins_l43_43916


namespace total_brothers_age_correct_l43_43128

noncomputable def total_age_of_brothers (older_young_ratio : ℕ → ℕ) (age_diff : ℕ) : ℕ :=
let (r1, r2) := (3, 2) in
if ratio : older_young_ratio r1 r2 = true ∧ age_diff = 24 then
  let x := 24 in (3 * x + 2 * x)
else 0

theorem total_brothers_age_correct :
  total_age_of_brothers (λ r1 r2, r1 * 2 = r2 * 3) 24 = 120 :=
by sorry

end total_brothers_age_correct_l43_43128


namespace storm_deposit_eq_120_billion_gallons_l43_43660

theorem storm_deposit_eq_120_billion_gallons :
  ∀ (initial_content : ℝ) (full_percentage_pre_storm : ℝ) (full_percentage_post_storm : ℝ) (reservoir_capacity : ℝ),
  initial_content = 220 * 10^9 → 
  full_percentage_pre_storm = 0.55 →
  full_percentage_post_storm = 0.85 →
  reservoir_capacity = initial_content / full_percentage_pre_storm →
  (full_percentage_post_storm * reservoir_capacity - initial_content) = 120 * 10^9 :=
by
  intro initial_content full_percentage_pre_storm full_percentage_post_storm reservoir_capacity
  intros h_initial_content h_pre_storm h_post_storm h_capacity
  sorry

end storm_deposit_eq_120_billion_gallons_l43_43660


namespace trajectory_of_P_is_line_l43_43427

noncomputable def P_trajectory_is_line (a m : ℝ) (P : ℝ × ℝ) : Prop :=
  let A := (-a, 0)
  let B := (a, 0)
  let PA := (P.1 + a) ^ 2 + P.2 ^ 2
  let PB := (P.1 - a) ^ 2 + P.2 ^ 2
  PA - PB = m → P.1 = m / (4 * a)

theorem trajectory_of_P_is_line (a m : ℝ) (h : a ≠ 0) :
  ∀ (P : ℝ × ℝ), (P_trajectory_is_line a m P) := sorry

end trajectory_of_P_is_line_l43_43427


namespace total_divisors_f_1005_l43_43720

def f (n : ℕ) : ℕ := 2^(n+1)

theorem total_divisors_f_1005 :
  let div_count (n : ℕ) := n + 1 in
  div_count (f 1005) = 1007 := 
by
  intro div_count
  have h1 : f 1005 = 2^1006, sorry
  have h2 : ∀ (n : ℕ), div_count (2^n) = n + 1, sorry
  rw h1
  exact h2 1006
  --  sorry skips the formal part of the proof

end total_divisors_f_1005_l43_43720


namespace cylinder_lateral_surface_area_l43_43561

theorem cylinder_lateral_surface_area 
  (diameter height : ℝ) 
  (h1 : diameter = 2) 
  (h2 : height = 2) : 
  2 * Real.pi * (diameter / 2) * height = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l43_43561
