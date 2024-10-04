import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time.Calendar
import Mathlib.SetTheory.Cardinal
import Mathlib.Tactic
import Real

namespace sum_max_min_l146_146433

theorem sum_max_min (α : ℚ) (a b : ℝ) (hα : true) (ha : 0 < a) (hb : a < b)  
    (h_max_f : ∀ (x : ℝ), a ≤ x ∧ x ≤ b → x^α + 1 ≤ 6)
    (h_min_f : ∀ (x : ℝ), a ≤ x ∧ x ≤ b → 3 ≤ x^α + 1) :
    ∃ (sum_1 : ℝ), sum_1 = 9 ∨ sum_1 = -5 :=
begin
  sorry
end

end sum_max_min_l146_146433


namespace exponent_power_identity_l146_146336

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146336


namespace average_age_of_11_students_l146_146480

theorem average_age_of_11_students 
  (age_avg_15 : ℕ) (age_avg_3 : ℕ) (age_15th : ℕ)
  (h1 : age_avg_15 = 15)
  (h2 : age_avg_3 = 14)
  (h3 : age_15th = 7) : 
  (15 * 15 - 14 * 3 - 7) / 11 = 16 :=
by
  have total_age_15 := 225
  have total_age_3 := 42
  have total_age_11 := total_age_15 - total_age_3 - age_15th
  calc
    (15 * 15 - 14 * 3 - 7) / 11 = 176 / 11 : by norm_num
    ... = 16 : by norm_num

end average_age_of_11_students_l146_146480


namespace power_of_three_l146_146372

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146372


namespace cube_root_of_a_plus_b_l146_146127

theorem cube_root_of_a_plus_b
  (a b : ℝ)
  (h1 : a = (2 * b - 1) ^ 2)
  (h2 : a = (b + 4) ^ 2)
  (h3 : 2 * b - 1 + (b + 4) = 0) :
  real.cbrt (a + b) = 2 :=
sorry

end cube_root_of_a_plus_b_l146_146127


namespace number_of_days_with_equal_mondays_and_wednesdays_l146_146124

theorem number_of_days_with_equal_mondays_and_wednesdays :
  ∃ d: ℕ, d = 2 ∧
    ∀ (start: ℕ), start < 7 →
    (let days := (List.range 31) in
     let weekdays := days.map (λ n, (start + n) % 7) in
     let mondays := weekdays.countp (λ d, d = 0) in
     let wednesdays := weekdays.countp (λ d, d = 2) in
     mondays = wednesdays → (start = 3 ∨ start = 4)) := sorry

end number_of_days_with_equal_mondays_and_wednesdays_l146_146124


namespace find_matrix_B_l146_146743

theorem find_matrix_B (A B : Matrix (Fin 2) (Fin 2) ℤ) :
  A = ![![1, 0], ![1, 2]] →
  B = ![[-4, 3], [4, -2]] →
  A ⬝ B = ![[-4, 3], [4, -1]] :=
by 
  intros hA hB
  rw [hA, hB]
  -- Here, we would perform the matrix multiplication to verify the result.
  sorry

end find_matrix_B_l146_146743


namespace cartesian_eq_of_curve_C_tan_alpha_value_l146_146024

-- Step (I): The Cartesian equation of curve \(C\) is \( y^2 = 4x \).
theorem cartesian_eq_of_curve_C (ρ θ : ℝ) (h : ρ * sin θ ^ 2 = 4 * cos θ) : 
  ∃ x y : ℝ, y^2 = 4 * x := 
sorry

-- Step (II): The value of \(\tan \alpha\) is \( \sqrt{3} \) or \(\tan \alpha = -\sqrt{3}\).
theorem tan_alpha_value (α : ℝ) (t : ℝ) 
  (hx : 0 < α ∧ α < π)
  (hC : ∃ y x, y^2 = 4 * x) 
  (hPA_PB : (2 + t * cos α - 2)^2 + (1 + t * sin α - 1)^2 = 28 / 3) : 
  tan α = sqrt 3 ∨ tan α = -sqrt 3 := 
sorry

end cartesian_eq_of_curve_C_tan_alpha_value_l146_146024


namespace ab_squared_value_l146_146614

noncomputable def regular_tetrahedron (α : plane) (A B C D : point) : Prop :=
all_vertices_on_one_side_of_plane A B C D α ∧ 
projections_form_square A B C D α ∧ 
dist_to_plane A α = 17 ∧
dist_to_plane B α = 21

theorem ab_squared_value (α : plane) (A B C D : point) :
  regular_tetrahedron α A B C D →
  distance_squared A B = 32 :=
sorry

end ab_squared_value_l146_146614


namespace number_of_outfits_l146_146921

theorem number_of_outfits : (5 * 4 * 6 * 3) = 360 := by
  sorry

end number_of_outfits_l146_146921


namespace cot_add_tan_eq_csc_l146_146014

theorem cot_add_tan_eq_csc (deg20 deg10 : ℝ) (h1 : deg20 = 20) (h2 : deg10 = 10) : 
    Real.cot (Real.pi * deg20 / 180) + Real.tan (Real.pi * deg10 / 180) = Real.csc (Real.pi * deg20 / 180) := 
sorry

end cot_add_tan_eq_csc_l146_146014


namespace coins_ordering_expected_weighings_lt_4_8_l146_146520

-- Definitions based on given conditions
def coins : Type := fin 4
def weigh (a b : coins) : Prop := true -- Abstract representation of weighing action

-- Problem statement in Lean
theorem coins_ordering_expected_weighings_lt_4_8 :
  ∃ order : list coins, expected_weighings order < 4.8 :=
sorry

end coins_ordering_expected_weighings_lt_4_8_l146_146520


namespace proof_problem_l146_146510

noncomputable def aₙ (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d
noncomputable def Sₙ (a₁ d : ℝ) (n : ℕ) := n * a₁ + (n * (n - 1) / 2) * d

def given_conditions (Sₙ : ℕ → ℝ) : Prop :=
  Sₙ 10 = 0 ∧ Sₙ 15 = 25

theorem proof_problem (a₁ d : ℝ) (Sₙ : ℕ → ℝ)
  (h₁ : Sₙ 10 = 0) (h₂ : Sₙ 15 = 25) :
  (aₙ a₁ d 5 = -1/3) ∧
  (∀ n, Sₙ n = (1 / 3) * (n ^ 2 - 10 * n) → n = 5) ∧
  (∀ n, n * Sₙ n = (n ^ 3 / 3) - (10 * n ^ 2 / 3) → min (n * Sₙ n) = -49) ∧
  (¬ ∃ n, (Sₙ n / n) > 0) :=
sorry

end proof_problem_l146_146510


namespace fourth_training_session_end_time_l146_146625

theorem fourth_training_session_end_time :
  (let start_time := ⟨8, 0⟩ : Time -- 8:00 AM
   let session_duration := 40 -- minutes
   let break_duration := 15 -- minutes
   let num_sessions := 4
   let total_training_duration := num_sessions * session_duration
   let num_breaks := num_sessions - 1
   let total_break_duration := num_breaks * break_duration
   let total_duration := total_training_duration + total_break_duration
   let hours := total_duration / 60
   let minutes := total_duration % 60
   let end_time := Time.mk (start_time.hour + hours) (start_time.minute + minutes)
   end_time = ⟨11, 25⟩
  ) := sorry

end fourth_training_session_end_time_l146_146625


namespace john_total_cost_l146_146856

theorem john_total_cost :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 4
  let base_video_card_cost := 300
  let upgraded_video_card_cost := 2.5 * base_video_card_cost
  let video_card_discount := 0.12 * upgraded_video_card_cost
  let upgraded_video_card_final_cost := upgraded_video_card_cost - video_card_discount
  let foreign_monitor_cost_local := 200
  let exchange_rate := 1.25
  let foreign_monitor_cost_usd := foreign_monitor_cost_local / exchange_rate
  let peripherals_sales_tax := 0.05 * peripherals_cost
  let subtotal := computer_cost + peripherals_cost + upgraded_video_card_final_cost + peripherals_sales_tax
  let store_loyalty_discount := 0.07 * (computer_cost + peripherals_cost + upgraded_video_card_final_cost)
  let final_cost := subtotal - store_loyalty_discount + foreign_monitor_cost_usd
  final_cost = 2536.30 := sorry

end john_total_cost_l146_146856


namespace exponent_power_identity_l146_146344

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146344


namespace variance_of_sample_data_l146_146382

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def variance (s : List ℝ) : ℝ :=
  let μ := mean s
  (List.map (λ x, (x - μ) ^ 2) s).sum / s.length

theorem variance_of_sample_data :
  mean [8, 12, 10, 11, 9] = 10 →
  variance [8, 12, 10, 11, 9] = 2 := by
  intros h_mean
  sorry

end variance_of_sample_data_l146_146382


namespace power_addition_l146_146297

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146297


namespace range_of_a_l146_146891

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 4 else 2^x

theorem range_of_a (a : ℝ) :
  (∀ a, f (f a) > f (f a + 1)) → a ∈ Ioo (-(5:ℝ) / 2) (-2) ∪ { -2 } :=
by sorry

end range_of_a_l146_146891


namespace cube_root_of_zero_l146_146927

theorem cube_root_of_zero : (∃ x : ℝ, x ^ 3 = 0) → x = 0 :=
by
  intro h
  cases h with x hx
  exact hx.symm
  sorry

end cube_root_of_zero_l146_146927


namespace angle_tangent_proof_l146_146448

theorem angle_tangent_proof (O1 O2 : Point) (circle1 circle2 : Circle) (P A B: Point)
(h1 : O1 ∈ circle1.center) (h2 : O2 ∈ circle2.center) (h3 : P ∈ circle1 ∧ P ∈ circle2)
(h4 : tangent A O1 P) (h5 : tangent B O2 P) :
2 * ∠ O1 P O2 = ∠ O1 P A + ∠ O2 P B := sorry

end angle_tangent_proof_l146_146448


namespace binom_9_5_l146_146676

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146676


namespace binom_9_5_eq_126_l146_146685

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146685


namespace solve_x2_y2_eq_3z2_in_integers_l146_146015

theorem solve_x2_y2_eq_3z2_in_integers (x y z : ℤ) : x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_x2_y2_eq_3z2_in_integers_l146_146015


namespace average_gas_mileage_round_trip_l146_146588

def distance_motorcycle : ℕ := 150
def mileage_motorcycle : ℕ := 50
def distance_truck : ℕ := 180
def mileage_truck : ℕ := 15

def total_distance : ℕ :=
  distance_motorcycle + distance_truck

def gasoline_motorcycle : ℕ :=
  distance_motorcycle / mileage_motorcycle

def gasoline_truck : ℕ :=
  distance_truck / mileage_truck

def total_gasoline : ℕ :=
  gasoline_motorcycle + gasoline_truck

def avg_mileage : ℕ :=
  total_distance / total_gasoline

theorem average_gas_mileage_round_trip :
  avg_mileage = 22 :=
by
  unfold avg_mileage
  unfold total_distance
  unfold total_gasoline
  unfold gasoline_motorcycle gasoline_truck
  unfold distance_motorcycle mileage_motorcycle distance_truck mileage_truck
  norm_num
  sorry -- Proof can be completed with tangible number calculations using norm_num


end average_gas_mileage_round_trip_l146_146588


namespace total_bird_count_correct_l146_146833

-- Define initial counts
def initial_sparrows : ℕ := 89
def initial_pigeons : ℕ := 68
def initial_finches : ℕ := 74

-- Define additional birds
def additional_sparrows : ℕ := 42
def additional_pigeons : ℕ := 51
def additional_finches : ℕ := 27

-- Define total counts
def initial_total : ℕ := 231
def final_total : ℕ := 312

theorem total_bird_count_correct :
  initial_sparrows + initial_pigeons + initial_finches = initial_total ∧
  (initial_sparrows + additional_sparrows) + 
  (initial_pigeons + additional_pigeons) + 
  (initial_finches + additional_finches) = final_total := by
    sorry

end total_bird_count_correct_l146_146833


namespace power_addition_l146_146293

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146293


namespace transformed_sample_variance_l146_146821

theorem transformed_sample_variance (a : ℕ → ℝ) (n : ℕ) (h : variance a = 3) :
  variance (λ i, 3 * a i + 1) = 27 :=
by
  -- proof steps would go here
  sorry

end transformed_sample_variance_l146_146821


namespace conference_handshakes_l146_146566

theorem conference_handshakes (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  sorry

end conference_handshakes_l146_146566


namespace exponent_power_identity_l146_146335

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146335


namespace unique_double_digit_in_range_l146_146847

theorem unique_double_digit_in_range (a b : ℕ) (h₁ : a = 10) (h₂ : b = 40) : 
  ∃! n : ℕ, (10 ≤ n ∧ n ≤ 40) ∧ (n % 10 = n / 10) ∧ (n % 10 = 3) :=
by {
  sorry
}

end unique_double_digit_in_range_l146_146847


namespace equal_mondays_wednesdays_l146_146122

theorem equal_mondays_wednesdays (days_in_month : ℕ) (days_in_week : ℕ) 
  (starts_on_possible_days : Finset ℕ) : 
  days_in_month = 31 ∧ days_in_week = 7 ∧ 
  starts_on_possible_days = {0, 1, 2, 3, 4, 5, 6} → 
  (Finset.filter (λ d, (d + days_in_month - 1) % days_in_week = d + 2 % days_in_week)
    starts_on_possible_days).card = 3 := 
by 
  sorry

end equal_mondays_wednesdays_l146_146122


namespace line_through_center_l146_146769

-- Define the parametric equations of the curve and the conditions
def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

lemma parametric_bounds (θ : ℝ) : 0 ≤ θ ∧ θ ≤ 2 * Real.pi :=
sorry

-- Define the direction vector
def direction_vector : ℝ × ℝ := (1, 1)

-- Define the center of the circle derived from the parametric equations
def center_of_curve : ℝ × ℝ := (1, 1)

-- Define the target equation of the line
def line_equation (x y : ℝ) : Prop := y = x

-- The theorem to be proved
theorem line_through_center :
  ∀ x y : ℝ, (let (cx, cy) := center_of_curve in (cy - 1) / (cx - 1) = 1) → line_equation x y :=
sorry

end line_through_center_l146_146769


namespace log_base_12_of_5_eq_l146_146231

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

variables (a b : ℝ)
axiom lg_two_eq_a : Real.log 2 = a
axiom ten_pow_b_eq_three : 10^b = 3

theorem log_base_12_of_5_eq : log_base 12 5 = (1 - a) / (2 * a + b) :=
by sorry

end log_base_12_of_5_eq_l146_146231


namespace sweets_original_count_l146_146572

theorem sweets_original_count (S : ℕ) (children : ℕ) (sweets_per_child : ℕ) (third_remaining : ℕ) :
  children = 48 → sweets_per_child = 4 → third_remaining = 1 / 3 * S → S = 288 := by
  intros h1 h2 h3
  have h4 : 48 * 4 = 192 := rfl
  have h5 : 2 / 3 * S = 192 := by
    rw [h3, h4]
  have h6 : S = 192 * 3 / 2 := by
    rw [h5]
  have h7 : 192 * 3 / 2 = 288 := rfl
  rw [h7]
  exact rfl

end sweets_original_count_l146_146572


namespace binom_9_5_eq_126_l146_146659

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146659


namespace sum_digits_greatest_prime_divisor_16386_l146_146491

theorem sum_digits_greatest_prime_divisor_16386 :
  ∃ p : ℕ, prime p ∧ p ∣ 16386 ∧ (digits 10 p).sum = 5 := 
by
  sorry

end sum_digits_greatest_prime_divisor_16386_l146_146491


namespace power_of_three_l146_146277

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146277


namespace parallelogram_perimeter_l146_146956

def perimeter_of_parallelogram (a b : ℝ) : ℝ :=
  2 * (a + b)

theorem parallelogram_perimeter
  (side1 side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_side2 : side2 = 12) :
  perimeter_of_parallelogram side1 side2 = 60 := 
by
  sorry

end parallelogram_perimeter_l146_146956


namespace find_b4_b_inv4_l146_146023

variable {b : ℝ}

theorem find_b4_b_inv4 (h : 5 = b + b⁻¹) : b^4 + b⁻⁴ = 527 := by
  sorry

end find_b4_b_inv4_l146_146023


namespace external_common_tangent_length_l146_146262

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) (f : ℝ) : Prop := 
  y = k * (x - f)

noncomputable def distance (A B : (ℝ × ℝ)) : ℝ := 
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def tangent_length (AF BF : ℝ) : ℝ := 
  2 * real.sqrt (AF * BF)

theorem external_common_tangent_length {x1 x2 y1 y2 k : ℝ} (hf : k > 0)
  (h1 : parabola x1 y1) (h2 : parabola x2 y2)
  (h3 : line_through_focus k x1 y1 1) (h4 : line_through_focus k x2 y2 1)
  (h_AB : distance (x1, y1) (x2, y2) = 5) :
  tangent_length (real.sqrt ((x1 + 1)^2 + y1^2)) (real.sqrt ((x2 + 1)^2 + y2^2)) = 2 * real.sqrt 5 := 
  sorry

end external_common_tangent_length_l146_146262


namespace colten_chickens_l146_146002

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l146_146002


namespace largest_multiple_of_18_with_digits_9_or_0_l146_146037

theorem largest_multiple_of_18_with_digits_9_or_0 :
  ∃ (n : ℕ), (n = 9990) ∧ (n % 18 = 0) ∧ (∀ d ∈ (n.digits 10), d = 9 ∨ d = 0) ∧ (n / 18 = 555) :=
by
  sorry

end largest_multiple_of_18_with_digits_9_or_0_l146_146037


namespace largest_multiple_of_15_under_500_l146_146999

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146999


namespace find_range_of_x_l146_146764

open Real

theorem find_range_of_x (a b x : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h : a^(log b (x - 3)) < 1) : 3 < x ∧ x < 4 :=
  sorry

end find_range_of_x_l146_146764


namespace power_calculation_l146_146305

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146305


namespace functional_eq_app_only_solutions_l146_146182

noncomputable def f : Real → Real := sorry

theorem functional_eq_app (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2 :=
sorry

theorem only_solutions (f : ℝ → ℝ) (hf : ∀ n : ℕ, ∀ x : Fin n → ℝ, (∀ i, 0 ≤ x i) → f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2) :
  f = (fun x => 0) ∨ f = (fun x => x) :=
sorry

end functional_eq_app_only_solutions_l146_146182


namespace min_balls_to_guarantee_15_single_color_l146_146106

theorem min_balls_to_guarantee_15_single_color
    (r g y b p o : ℕ)
    (hr : r = 24)
    (hg : g = 23)
    (hy : y = 22)
    (hb : b = 15)
    (hp : p = 10)
    (ho : o = 6) :
    ∃ n, n = 73 ∧ 
      (∀ (red green yellow blue pink orange : ℕ),
        red ≤ r ∧ green ≤ g ∧ yellow ≤ y ∧ blue ≤ b ∧ pink ≤ p ∧ orange ≤ o →
         (red < 15) ∧ (green < 15) ∧ (yellow < 15) ∧ (blue < 15) ∧ (pink < 15) ∧ (orange < 15) ∧
         red + green + yellow + blue + pink + orange = 72) →
          n = min (red + 1) (green + 1) (yellow + 1) (blue + 1) (pink + 1) (orange + 1) :=
begin
  sorry
end

end min_balls_to_guarantee_15_single_color_l146_146106


namespace betty_garden_total_plants_l146_146630

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l146_146630


namespace smallest_positive_debt_l146_146963

/-- Given that pigs are worth \$300 and goats are worth \$210, prove that the smallest positive
debt that can be resolved using these values is \$30. -/
theorem smallest_positive_debt :
  let pig_value := 300
  let goat_value := 210
  ∃ (d : ℕ), d > 0 ∧ (∃ (a b : ℤ), d = a * pig_value + b * goat_value) ∧ d = Nat.gcd pig_value goat_value :=
by
  let pig_value := 300
  let goat_value := 210
  use Nat.gcd pig_value goat_value
  sorry

end smallest_positive_debt_l146_146963


namespace power_of_3_l146_146356

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146356


namespace find_y_l146_146969

variable (h : ℕ) -- integral number of hours

-- Distance between A and B
def distance_AB : ℕ := 60

-- Speed and distance walked by woman starting at A
def speed_A : ℕ := 3
def distance_A (h : ℕ) : ℕ := speed_A * h

-- Speed and distance walked by woman starting at B
def speed_B_1st_hour : ℕ := 2
def distance_B (h : ℕ) : ℕ := (h * (h + 3)) / 2

-- Meeting point equation
def meeting_point_eqn (h : ℕ) : Prop := (distance_A h) + (distance_B h) = distance_AB

-- Requirement: y miles nearer to A whereas y = distance_AB - 2 * distance_B (since B meets closer to A by y miles)
def y_nearer_A (h : ℕ) : ℕ := distance_AB - 2 * (distance_A h)

-- Prove y = 6 for the specific value of h
theorem find_y : ∃ (h : ℕ), meeting_point_eqn h ∧ y_nearer_A h = 6 := by
  sorry

end find_y_l146_146969


namespace total_buttons_needed_l146_146005

def shirts_sewn_on_monday := 4
def shirts_sewn_on_tuesday := 3
def shirts_sewn_on_wednesday := 2
def buttons_per_shirt := 5

theorem total_buttons_needed : 
  (shirts_sewn_on_monday + shirts_sewn_on_tuesday + shirts_sewn_on_wednesday) * buttons_per_shirt = 45 :=
by 
  sorry

end total_buttons_needed_l146_146005


namespace trigonometric_simplification_l146_146176

theorem trigonometric_simplification :
  (sin 10 + sin 20 + sin 30 + sin 40 + sin 50 + sin 60 + sin 70 + sin 80) / (cos 5 * cos 10 * cos 20) = 4 * real.sqrt 2 :=
sorry

end trigonometric_simplification_l146_146176


namespace combined_weight_proof_l146_146543

-- Definitions of atomic weights
def weight_C : ℝ := 12.01
def weight_H : ℝ := 1.01
def weight_O : ℝ := 16.00
def weight_S : ℝ := 32.07

-- Definitions of molar masses of compounds
def molar_mass_C6H8O7 : ℝ := (6 * weight_C) + (8 * weight_H) + (7 * weight_O)
def molar_mass_H2SO4 : ℝ := (2 * weight_H) + weight_S + (4 * weight_O)

-- Definitions of number of moles
def moles_C6H8O7 : ℝ := 8
def moles_H2SO4 : ℝ := 4

-- Combined weight
def combined_weight : ℝ := (moles_C6H8O7 * molar_mass_C6H8O7) + (moles_H2SO4 * molar_mass_H2SO4)

theorem combined_weight_proof : combined_weight = 1929.48 :=
by
  -- calculations as explained in the problem
  let wC6H8O7 := moles_C6H8O7 * molar_mass_C6H8O7
  let wH2SO4 := moles_H2SO4 * molar_mass_H2SO4
  have h1 : wC6H8O7 = 8 * 192.14 := by sorry
  have h2 : wH2SO4 = 4 * 98.09 := by sorry
  have h3 : combined_weight = wC6H8O7 + wH2SO4 := by simp [combined_weight, wC6H8O7, wH2SO4]
  rw [h3, h1, h2]
  simp
  sorry -- finish the proof as necessary

end combined_weight_proof_l146_146543


namespace csc_315_l146_146177

theorem csc_315 (deg_to_rad : ℝ → ℝ) (sin : ℝ → ℝ) (csc : ℝ := λ θ, 1 / sin θ) :
  (∀ θ, sin (2 * π - θ) = -sin θ) →
  sin (π / 4) = 1 / real.sqrt 2 →
  csc (7 * π / 4) = -real.sqrt 2 :=
by
  intro angle_identity
  intro sin_45
  sorry

end csc_315_l146_146177


namespace power_calculation_l146_146303

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146303


namespace determinant_sin_eq_zero_l146_146653

theorem determinant_sin_eq_zero (a b : ℝ) : 
  matrix.det ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by
  sorry

end determinant_sin_eq_zero_l146_146653


namespace larry_spent_on_lunch_l146_146863

noncomputable def starting_amount : ℕ := 22
noncomputable def ending_amount : ℕ := 15
noncomputable def amount_given_to_brother : ℕ := 2

theorem larry_spent_on_lunch : 
  (starting_amount - (ending_amount + amount_given_to_brother)) = 5 :=
by
  -- The conditions and the proof structure would be elaborated here
  sorry

end larry_spent_on_lunch_l146_146863


namespace angle_between_medians_of_triangle_is_right_l146_146499

theorem angle_between_medians_of_triangle_is_right
  (T : Type)
  [EuclideanSpace T] 
  (A B C A₁ G : T)
  (h₀ : between A₁ B C)
  (h₁ : mid_point A A₁ B C)
  (h₂ : is_centroid G A B C)
  (h₃ : norm (A - A₁) = (3/2) * norm (B - C))
  : (@angle_between_medians_of_triangle_is_right A B C (2 : ℝ/ 90)) := 
  sorry

end angle_between_medians_of_triangle_is_right_l146_146499


namespace find_f_zero_l146_146936

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x * y) = f x + f y

theorem find_f_zero : f 0 = 0 := 
by {
  have h : f (0 * 0) = f 0 + f 0 := functional_eq 0 0,
  simp at h,
  exact add_self_eq_zero.mp h,
}

end find_f_zero_l146_146936


namespace tenth_pirate_receives_exactly_1296_coins_l146_146115

noncomputable def pirate_coins (n : ℕ) : ℕ :=
  if n = 0 then 0
  else Nat.factorial 9 / 11^9 * 11^(10 - n)

theorem tenth_pirate_receives_exactly_1296_coins :
  pirate_coins 10 = 1296 :=
sorry

end tenth_pirate_receives_exactly_1296_coins_l146_146115


namespace total_oranges_l146_146961

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end total_oranges_l146_146961


namespace minimum_value_of_expression_l146_146872

theorem minimum_value_of_expression (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ (x y z : ℝ), (x^2 + (y - 1)^2 + z^2) = 18 / 7 ∧ y = -2 / 7 :=
sorry

end minimum_value_of_expression_l146_146872


namespace binomial_coefficient_9_5_l146_146667

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146667


namespace find_m_l146_146489

noncomputable def m_value (m : ℝ) := 
  ((m ^ 2) - m - 1, (m ^ 2) - 2 * m - 1)

theorem find_m (m : ℝ) (h1 : (m ^ 2) - m - 1 = 1) (h2 : (m ^ 2) - 2 * m - 1 < 0) : 
  m = 2 :=
by sorry

end find_m_l146_146489


namespace smallest_number_in_set_l146_146523

theorem smallest_number_in_set :
  let S := {3.4, (7 : ℝ) / 2, 1.7, (27 : ℝ) / 10, 2.9}
  in ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 1.7 :=
begin
  let S : set ℝ := {3.4, 3.5, 1.7, 2.7, 2.9},
  have h : 1.7 ∈ S, { sorry },
  use 1.7,
  split,
  { exact h },
  { intro y,
    split,
    { intro hy,
      interval_cases y; linarith },
    { refl }
  }
end

end smallest_number_in_set_l146_146523


namespace range_of_f_l146_146046

noncomputable def f (x : ℝ) : ℝ := (x^2 - x) / (x^2 - x + 1)

theorem range_of_f :
  set_of (λ y, ∃ x : ℝ, y = f x) = set.Icc (-⅓) 1 :=
sorry

end range_of_f_l146_146046


namespace compare_arrangements_l146_146458

-- Definitions based on the problem conditions
def rope_length := 8
def side_length := 16
def radius1 := rope_length
def radius2 := 4

-- Areas for different arrangements
def area_arrangement_I := (1 / 2) * Real.pi * radius1^2
def area_main_part_arrangement_II := (1 / 2) * Real.pi * radius1^2
def area_additional_part_arrangement_II := (1 / 4) * Real.pi * radius2^2
def total_area_arrangement_II := area_main_part_arrangement_II + area_additional_part_arrangement_II

-- Difference in roaming areas
def area_difference := total_area_arrangement_II - area_arrangement_I

-- The statement to prove
theorem compare_arrangements : total_area_arrangement_II > area_arrangement_I ∧ area_difference = 4 * Real.pi := by
  sorry

end compare_arrangements_l146_146458


namespace tom_roses_per_day_l146_146067

-- Define variables and conditions
def total_roses := 168
def days_in_week := 7
def dozen := 12

-- Theorem to prove
theorem tom_roses_per_day : (total_roses / dozen) / days_in_week = 2 :=
by
  -- The actual proof would go here, using the sorry placeholder
  sorry

end tom_roses_per_day_l146_146067


namespace evaluate_f_of_a_plus_one_l146_146258

theorem evaluate_f_of_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x, x^2 + 1 
  in f(a + 1) = a^2 + 2*a + 2 :=
by
  let f : ℝ → ℝ := λ x, x^2 + 1
  sorry

end evaluate_f_of_a_plus_one_l146_146258


namespace a_n_general_term_b_n_general_term_b_seq_satisfies_condition_l146_146243

open_locale big_operators

def a_seq (n : ℕ) : ℕ := 2 * n + 2

def b_seq : ℕ → ℕ
| 0     := 0    -- since Lean sequence are usually 0-based.
| 1     := 6
| (n+2) := 2 * (n+2)

theorem a_n_general_term : ∀ n : ℕ, a_seq n = 2 * n + 2 :=
begin
  sorry
end

theorem b_n_general_term : ∀ n : ℕ,
  b_seq n = if n = 1 then 6 else 2 * n :=
begin
  sorry
end

theorem b_seq_satisfies_condition :
  ∀ n : ℕ, (finset.sum (finset.range (n+1)) (λ k, b_seq (k+1) / (k+1))) = a_seq (n+1) :=
begin
  sorry
end

end a_n_general_term_b_n_general_term_b_seq_satisfies_condition_l146_146243


namespace no_real_root_l146_146918

-- Define the equation as a condition
def equation (x : ℝ) : Prop := sqrt (x + 9) - sqrt (x - 2) + 2 = 0

-- State the theorem we want to prove
theorem no_real_root : ¬∃ x : ℝ, equation x :=
by {
    -- We omit the actual proof, as requested
    sorry
}

end no_real_root_l146_146918


namespace magic_square_base_l146_146168

theorem magic_square_base : ∃ b : ℕ, b ≠ 0 ∧ (∀ (x y z : ℤ), x = 1 + (2 * b + 1) + 2 → y = (1 * b + 4) + 1 + 4 → x = y) := 
begin
  use 5,
  split,
  { -- b ≠ 0
    exact nat.succ_ne_zero 4 },
  { -- Sums are equal when b = 5
    intros x y z h₁ h₂,
    have h₃ : x = 3 + 2 * 5 + 1 := by rw h₁,
    have h₄ : y = 1 * 5 + 4 + 1 + 4 := by rw h₂,
    rw [h₃, h₄],
    norm_num,
  }
end

end magic_square_base_l146_146168


namespace power_of_three_l146_146359

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146359


namespace projection_of_a_onto_b_l146_146889

-- Define vectors a and b in a real vector space
variables (a b : ℝ × ℝ)
-- Define magnitudes of the vectors
def mag_a := 2
def mag_b := 2
-- Define the angle between vectors a and b
def angle := real.pi / 3 -- 60 degrees in radians

-- Definition to calculate dot product given the angle and magnitudes
def dot_product (a b : ℝ × ℝ) (mag_a mag_b : ℝ) (angle : ℝ) :=
  mag_a * mag_b * real.cos angle

-- Given conditions
axiom mag_a_eq_2 : ∥a∥ = mag_a
axiom mag_b_eq_2 : ∥b∥ = mag_b
axiom angle_eq_60 : ∡ a b = angle

-- The specific theorem to prove
theorem projection_of_a_onto_b : (dot_product a b 2 (real.pi / 3) / (∥b∥ ^ 2)) • b = (1 / 2) • b :=
by
  sorry  -- Proof will be written here

end projection_of_a_onto_b_l146_146889


namespace correct_statements_l146_146151

variable {a x : ℝ} {λ k : ℝ}
variable {am bm : ℝ}
variable {p q : Prop}

-- Definitions for statement ①
def prop_p (a x : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def prop_q (x : ℝ) : Prop := (x - 1) * (x - 3) < 0

-- Definitions for statement ②
def prop_p2 : Prop := (-2, λ) ≃ (1, 2) → λ = -4
def prop_q2 : Prop := ∀ k : ℝ, ∃ x y : ℝ, y = k * x ∧ (x^2 + y^2 - 2 * y = 0)

-- Definitions for statement ③
def exists_x : Prop := ∃ x : ℝ, x ^ 2 + x + 1 < 0

-- Definitions for statement ④
def prop_impl (x v : ℝ) : Prop := x = v → cos x = cos v

-- Definitions for statement ⑤
def prop_am_bm (am bm : ℝ) (m : ℝ) : Prop := am * m^2 < bm * m^2 → am < bm

-- Definitions for statement ⑥
def necessary_and_sufficient (x y : ℝ) : Prop :=
  x = y ↔ x * y ≥ (x + y) ^ 2 / 4

-- Definitions for statement ⑦
def prop_exists_x : Prop := ∃ x : ℝ, x ^ 2 + x + 1 < 0

-- Definitions for statement ⑧
def prop_imp (a b : ℝ) : Prop := a > b → 2 ^ a > 2 ^ b - 1

-- Main theorem stating the correct statements among those given
theorem correct_statements :
  let p1 := prop_p a x → prop_q x → (prop_q x → prop_p a x) ∨ ¬ prop_q x → [-1 ≤ a ∧ a ≤ 5]
  let p2 := prop_p2 ∨ prop_q2
  let p3 := ¬(∀ x : ℝ, x ^ 2 + x + 1 < 0)
  let p4 := prop_impl x v ∨ ¬prop_impl x v
  let p5 := (a < b ∧ am * m^2 < bm * m^2)
  let p6 := necessary_and_sufficient x y
  let p7 := ( ∃ x : ℝ, x ^ 2 + x + 1 < 0) → ( ∀ x : ℝ, x ^ 2 + x + 1 ≥ 0)
  let p8 := prop_imp a b → ¬ (a ≤ b ∧ 2 ^ a ≤ 2 ^ b - 1)
  p1 ∧ p2 ∧ ¬p3 ∧ p4 ∧ ¬p5 ∧ p6 ∧ p7 ∧ p8 := sorry

end correct_statements_l146_146151


namespace find_S10_l146_146754

-- Definitions
variables (a_n : ℕ → ℝ) (d : ℝ)
noncomputable def arithmetic_sequence := ∀ n, a_n (n + 1) = a_n n + d

noncomputable def geometric_mean_condition := a_n 3 * a_n 7 = (a_n 4) ^ 2

noncomputable def sum_of_first_eight_terms := 8 * a_n 1 + 28 * d = 32

-- Theorem statement: Prove S_10 = 60
theorem find_S10 :
  arithmetic_sequence a_n d →
  geometric_mean_condition a_n →
  sum_of_first_eight_terms a_n d →
  ∑ i in (finset.range 10), a_n i = 60 :=
by
  sorry

end find_S10_l146_146754


namespace inverse_of_5_mod_35_l146_146722

theorem inverse_of_5_mod_35 : (5 * 28) % 35 = 1 :=
by
  sorry

end inverse_of_5_mod_35_l146_146722


namespace proof_problem_l146_146104

theorem proof_problem (s t: ℤ) (h : 514 - s = 600 - t) : s < t ∧ t - s = 86 :=
by
  sorry

end proof_problem_l146_146104


namespace find_pairs_l146_146416

def sequence_a : Nat → Int
| 0 => 0
| 1 => 0
| n+2 => 2 * sequence_a (n+1) - sequence_a n + 2

def sequence_b : Nat → Int
| 0 => 8
| 1 => 8
| n+2 => 2 * sequence_b (n+1) - sequence_b n

theorem find_pairs :
  (sequence_a 1992 = 31872 ∧ sequence_b 1992 = 31880) ∨
  (sequence_a 1992 = -31872 ∧ sequence_b 1992 = -31864) :=
sorry

end find_pairs_l146_146416


namespace sum_of_coefficients_eq_48_l146_146731

noncomputable def sum_of_coefficients : ℕ :=
  let p := -3 * (X^8 - X^5 + 2 * X^3 - 6) + 5 * (X^4 + 3 * X^2) - 4 * (X^6 - 5)
  polynomial.eval 1 p

theorem sum_of_coefficients_eq_48 : sum_of_coefficients = 48 :=
  sorry

end sum_of_coefficients_eq_48_l146_146731


namespace find_ω_φ_find_monotonically_increasing_interval_find_range_on_interval_l146_146252

noncomputable def ω : ℝ := 2
noncomputable def φ : ℝ := -π / 3

def f (x : ℝ) : ℝ := cos (ω * x + φ)

theorem find_ω_φ :
  (∃ ω φ : ℝ, ω > 0 ∧ (-π / 2 < φ ∧ φ < 0) ∧ 2 * ω = 2 * π ∧ f (π / 4) = sqrt 3 / 2)
  ↔ (ω = 2 ∧ φ = -π / 3) := 
sorry

theorem find_monotonically_increasing_interval :
  (∀ k : ℤ, [k * π - π / 3, k * π + π / 6] ⊆ {x : ℝ | ∀ y : ℝ, y ∈ [x, x + π / 2] → f y > f x})
  := 
sorry

theorem find_range_on_interval : 
  (∀ x ∈ Icc 0 (π / 2), 
    -1 / 2 ≤ f x ∧ f x ≤ 1) :=
sorry

end find_ω_φ_find_monotonically_increasing_interval_find_range_on_interval_l146_146252


namespace projection_of_a_onto_b_l146_146887

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (h1 : angle a b = 60 * (Real.pi / 180)) (h2 : ‖a‖ = 2) (h3 : ‖b‖ = 2)

theorem projection_of_a_onto_b :
  orthogonal_projection (ℝ ∙ b) a = (1 / 2 : ℝ) • b :=
sorry

end projection_of_a_onto_b_l146_146887


namespace cos120_sin_neg45_equals_l146_146516

noncomputable def cos120_plus_sin_neg45 : ℝ :=
  Real.cos (120 * Real.pi / 180) + Real.sin (-45 * Real.pi / 180)

theorem cos120_sin_neg45_equals : cos120_plus_sin_neg45 = - (1 + Real.sqrt 2) / 2 :=
by
  sorry

end cos120_sin_neg45_equals_l146_146516


namespace probability_hat_to_glasses_is_1_6_l146_146400

-- Given statements
def num_people_wearing_hats : ℕ := 60
def num_people_wearing_glasses : ℕ := 40
def prob_glasses_to_hat : ℚ := 1 / 4

-- Defining the problem
def num_people_wearing_both : ℕ := nat.ceil (prob_glasses_to_hat * num_people_wearing_glasses)
def prob_hat_to_glasses : ℚ := num_people_wearing_both / num_people_wearing_hats

-- Statement to prove
theorem probability_hat_to_glasses_is_1_6 : prob_hat_to_glasses = 1 / 6 := by
  sorry

end probability_hat_to_glasses_is_1_6_l146_146400


namespace english_and_japanese_teachers_selection_l146_146623

theorem english_and_japanese_teachers_selection :
  let A := 3 -- English teachers only
  let B := 2 -- Japanese teachers only
  let C := 4 -- Both English and Japanese teachers
  ∑ (x : ℕ) in {0, 1, 2}, 
    nat.choose B x * nat.choose C (3 - x) * nat.choose (A + C - (3 - x)) 3 = 420 := by
  sorry

end english_and_japanese_teachers_selection_l146_146623


namespace sum_all_cells_is_576_diff_sums_black_white_is_0_sum_black_cells_is_288_l146_146579

-- Definitions based on problem conditions
def cell_value (row col : ℕ) : ℕ := row + col

def sum_all_cells : ℕ := (Σ i in Finset.range 8, Σ j in Finset.range 8, cell_value i.succ j.succ)

def sum_black_cells : ℕ := 
  (Σ i in Finset.range 8, Σ j in Finset.range 8, if ((i + j) % 2 = 0) then cell_value i.succ j.succ else 0)

def sum_white_cells : ℕ := 
  (Σ i in Finset.range 8, Σ j in Finset.range 8, if ((i + j) % 2 = 1) then cell_value i.succ j.succ else 0)

-- Statements to be proven
theorem sum_all_cells_is_576 : sum_all_cells = 576 := 
sorry

theorem diff_sums_black_white_is_0 : (sum_black_cells - sum_white_cells) = 0 := 
sorry

theorem sum_black_cells_is_288 : sum_black_cells = 288 := 
sorry

end sum_all_cells_is_576_diff_sums_black_white_is_0_sum_black_cells_is_288_l146_146579


namespace projection_of_a_on_b_l146_146799

noncomputable def vector_a : ℝ × ℝ × ℝ := (-1, 2, 3)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, 1, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def projection (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_on_b : 
  projection vector_a vector_b = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end projection_of_a_on_b_l146_146799


namespace new_person_weight_l146_146093

theorem new_person_weight (W : ℝ) (N : ℝ)
  (h1 : ∀ avg_increase : ℝ, avg_increase = 2.5 → N = 55) 
  (h2 : ∀ original_weight : ℝ, original_weight = 35) 
  : N = 55 := 
by 
  sorry

end new_person_weight_l146_146093


namespace power_calculation_l146_146315

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146315


namespace sum_of_possible_values_l146_146174

def satisfies_conditions (N : ℕ) : Prop :=
  N % 13 = 6 ∧ N % 9 = 3 ∧ N < 150

theorem sum_of_possible_values :
  (∑ N in (Finset.filter satisfies_conditions (Finset.range 150)), id N) = 252 :=
sorry

end sum_of_possible_values_l146_146174


namespace fourth_root_105413504_l146_146166

theorem fourth_root_105413504 : Real.root 4 105413504 = 101 :=
by
  sorry

end fourth_root_105413504_l146_146166


namespace largest_multiple_of_15_under_500_l146_146994

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146994


namespace wally_buys_bears_l146_146074

theorem wally_buys_bears :
  ∃ n : ℕ, let a1 := 4.0 in let d := -0.5 in let S := 354.0 in
            S = (n / 2) * (2 * a1 + (n - 1) * d) ∧ n = 48 :=
by
  sorry

end wally_buys_bears_l146_146074


namespace find_L_l146_146146

noncomputable def L := (14 * Real.sqrt 14) / 3

theorem find_L :
  (∃ (r : ℝ), 4 * Real.pi * r^2 = 28 ∧ ((4 / 3) * Real.pi * r^3 = (L * Real.sqrt 2) / (Real.sqrt Real.pi))) :=
begin
  use (Real.sqrt 7 / Real.sqrt Real.pi),
  split,
  { -- Proof that 4πr^2 = 28
    calc
      4 * Real.pi * (Real.sqrt 7 / Real.sqrt Real.pi)^2
          = 4 * Real.pi * (7 / Real.pi) : by rw [Real.sqrt_sq, Real.mul_div_cancel' 7 (Real.sqrt_ne_zero'.mpr Real.pi_pos)]
      ... = 4 * 7 : by rw [Real.pi_div_cancel, Real.mul_one]
      ... = 28 : by norm_num },
  { -- Proof that (4/3)πr^3 = (L * √2) / √π
    calc
      ((4 / 3) * Real.pi * (Real.sqrt 7 / Real.sqrt Real.pi)^3)
          = ((4 / 3) * Real.pi * ((7 * Real.sqrt 7) / (Real.sqrt Real.pi * Real.sqrt Real.pi)))
              : by rw [Real.sqrt_mul, Real.sqrt_sq, Real.sqrt_mul]
      ... = ((4 / 3) * Real.pi * (7 * Real.sqrt 7) / Real.pi^((3/2)))
              : by rw [Real.sqrt_mul, Real.sqrt_sq, Real.pi_pow_three_halves]
      ... = ((4 / 3) * 7 * (Real.sqrt 7) / Real.pi^(3/2)) : by ...
      ... = (L * Real.sqrt 2) / (Real.sqrt Real.pi) : by sorry }
end

end find_L_l146_146146


namespace smallest_of_five_consecutive_sum_100_l146_146522

theorem smallest_of_five_consecutive_sum_100 :
  ∃ n : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) ∧ n = 18 :=
begin
  sorry
end

end smallest_of_five_consecutive_sum_100_l146_146522


namespace filling_time_l146_146071

-- Define rates as constants
def pipeA_rate := 1 / 10
def pipeB_rate := 1 / 12
def pipeC_rate := -1 / 20

-- Define the combined rate of all three pipes
def combined_rate := pipeA_rate + pipeB_rate + pipeC_rate

-- Define the time required to fill the cistern when all pipes are opened
def time_to_fill := 1 / combined_rate

-- The theorem to prove that the time to fill the cistern is 7.5 hours
theorem filling_time : time_to_fill = 7.5 := by
  unfold time_to_fill combined_rate pipeA_rate pipeB_rate pipeC_rate
  norm_num
  sorry

end filling_time_l146_146071


namespace range_of_x_for_odd_monotonic_function_l146_146771

theorem range_of_x_for_odd_monotonic_function 
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_increasing_on_R : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  ∀ x : ℝ, (0 < x) → ( (|f (Real.log x) - f (Real.log (1 / x))| / 2) < f 1 ) → (Real.exp (-1) < x ∧ x < Real.exp 1) := 
by
  sorry

end range_of_x_for_odd_monotonic_function_l146_146771


namespace crescent_shapes_area_equals_square_area_l146_146271

theorem crescent_shapes_area_equals_square_area (s : ℝ) : 
  let semicircle_area (d : ℝ) := (π * (d / 2) ^ 2) / 2
  let total_semicircle_area := 4 * semicircle_area s
  let circumscribed_circle_area := π * (s * s) / 2
  let crescent_shapes_area := total_semicircle_area - circumscribed_circle_area
  in crescent_shapes_area = s^2 := 
sorry

end crescent_shapes_area_equals_square_area_l146_146271


namespace equal_mondays_wednesdays_l146_146121

theorem equal_mondays_wednesdays (days_in_month : ℕ) (days_in_week : ℕ) 
  (starts_on_possible_days : Finset ℕ) : 
  days_in_month = 31 ∧ days_in_week = 7 ∧ 
  starts_on_possible_days = {0, 1, 2, 3, 4, 5, 6} → 
  (Finset.filter (λ d, (d + days_in_month - 1) % days_in_week = d + 2 % days_in_week)
    starts_on_possible_days).card = 3 := 
by 
  sorry

end equal_mondays_wednesdays_l146_146121


namespace incorrect_statements_l146_146548

-- Define points and lines for the statements
structure Point where
  x : ℝ
  y : ℝ

def line_through_points (M N : Point) : Prop :=
  ∃ x y : ℝ, (x ≠ N.x → x ≠ M.x) ∧ (N.y - M.y ≠ 0 → (y - M.y) / (N.y - M.y) = (x - M.x) / (N.x - M.x))

def line_with_slope (P : Point) (α : ℝ) : Prop :=
  P = ⟨1, 0⟩ → (∀ x y : ℝ, y = (x - 1) * real.tan α)

def line_in_first_quadrant (m : ℝ) : Prop :=
  ∃ x y : ℝ, mx - (m - 1)y - 4 = 0 → x > 0 ∧ y > 0

def lines_with_equal_intercepts (a : ℝ) : Prop :=
  ∃ x y : ℝ, a ≠ 0 → (x + y = a → x = y)

-- The statements

def incorrect_statement_A : Prop := 
  ∃ M N : Point, M.x = N.x ∧ line_through_points M N

def incorrect_statement_B : Prop := 
  ∃ α : ℝ, α = real.pi / 2 ∧ line_with_slope ⟨1, 0⟩ α

def incorrect_statement_D : Prop := 
  ∀ a : ℝ, a ≠ 0 → ¬ lines_with_equal_intercepts a

theorem incorrect_statements :
  incorrect_statement_A ∧ incorrect_statement_B ∧ incorrect_statement_D :=
by
  -- Proof will be done here
  sorry

end incorrect_statements_l146_146548


namespace circle_points_divisible_by_10_l146_146904

theorem circle_points_divisible_by_10 (n : ℕ) (points : finset ℕ) (h_points : points.card = n)
  (h_dist1 : ∀ x ∈ points, ∃ y ∈ points, y = (x + 1) % 15)
  (h_dist2 : ∀ x ∈ points, ∃ z ∈ points, z = (x + 2) % 15) :
  10 ∣ n := 
sorry

end circle_points_divisible_by_10_l146_146904


namespace total_buttons_needed_l146_146006

def shirts_sewn_on_monday := 4
def shirts_sewn_on_tuesday := 3
def shirts_sewn_on_wednesday := 2
def buttons_per_shirt := 5

theorem total_buttons_needed : 
  (shirts_sewn_on_monday + shirts_sewn_on_tuesday + shirts_sewn_on_wednesday) * buttons_per_shirt = 45 :=
by 
  sorry

end total_buttons_needed_l146_146006


namespace prob_not_answered_after_three_rings_l146_146044

def prob_first_ring_answered := 0.1
def prob_second_ring_answered := 0.25
def prob_third_ring_answered := 0.45

theorem prob_not_answered_after_three_rings : 
  1 - prob_first_ring_answered - prob_second_ring_answered - prob_third_ring_answered = 0.2 :=
by
  sorry

end prob_not_answered_after_three_rings_l146_146044


namespace exponent_power_identity_l146_146337

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146337


namespace power_of_3_l146_146349

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146349


namespace cells_count_after_9_days_l146_146107

theorem cells_count_after_9_days :
  let a := 5
  let r := 3
  let n := 3
  a * r^(n-1) = 45 :=
by
  let a := 5
  let r := 3
  let n := 3
  sorry

end cells_count_after_9_days_l146_146107


namespace expected_height_L_trimino_rectangle_probability_height_12_L_trimino_rectangle_l146_146609

/-- 
 Given a tall rectangle with a width of 2, open at the top, 
 and k L-trimino tiles falling into it in random orientation,
 prove that the expected height of the resulting polygon is (7 * k + 1) / 4.
-/
theorem expected_height_L_trimino_rectangle (k : ℕ) : 
  let E_height := (7 * k + 1) / 4 
  in True := sorry

/-- 
 Given a tall rectangle with a width of 2, open at the top, 
 and 7 L-trimino tiles falling into it in random orientation,
 prove that the probability that the height of the polygon will be 12 is 0.133.
-/
theorem probability_height_12_L_trimino_rectangle : 
  let p_height_12 := 0.133 
  in True := sorry

end expected_height_L_trimino_rectangle_probability_height_12_L_trimino_rectangle_l146_146609


namespace find_f_at_2_l146_146805

noncomputable def f (x : ℝ) : ℝ :=
  (x + 2) * (x - 1) * (x - 3) * (x + 4) - x^2

theorem find_f_at_2 :
  (f(-2) = -4) ∧ (f(1) = -1) ∧ (f(3) = -9) ∧ (f(-4) = -16) ∧ (f 2 = -28) :=
by
  have h₁ : f (-2) = (0 : ℝ) := by sorry
  have h₂ : f 1 = (0 : ℝ) := by sorry
  have h₃ : f 3 = (0 : ℝ) := by sorry
  have h₄ : f (-4) = (0 : ℝ) := by sorry
  have h₅ : f 2 = (0 : ℝ) := by sorry
  exact ⟨h₁, h₂, h₃, h₄, h₅⟩

end find_f_at_2_l146_146805


namespace find_length_DB_l146_146399

-- Define the setup and necessary conditions
variables {A B C E D : Type} [metric_space A]
variables (right_triangle_ABC : right_triangle A B C)
variables (circle_on_AC : circle (diameter A C))
variables (E_on_hypotenuse : E ∈ hypotenuse A B)
variables (AE : ℝ) (BE : ℝ)
variables (AE_val : AE = 6) (BE_val : BE = 2)
variables (E_in_Circle : E ∈ circle_on_AC)
variables (D_on_CB : D ∈ leg C B)

-- Define the theorem that proves the sought result
theorem find_length_DB :
  BD = 2 :=
sorry

end find_length_DB_l146_146399


namespace power_of_three_l146_146281

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146281


namespace domain_of_f_l146_146973

noncomputable def f (x : ℝ) : ℝ := log 7 (log 5 (log 3 (log 2 x)))

theorem domain_of_f :
  ∀ x : ℝ, x > 2^243 → ∃ y, y = f(x) ∧ y > 0 :=
by
  intro x hx
  have hlog2x : log 2 x > 243 := sorry
  have hlog3log2x : log 3 (log 2 x) > 5 := sorry
  have hlog5log3log2x : log 5 (log 3 (log 2 x)) > 1 := sorry
  use f(x)
  split
  { refl }
  { sorry }

end domain_of_f_l146_146973


namespace valid_fahrenheit_temperatures_count_valid_fahrenheit_temperatures_l146_146970

theorem valid_fahrenheit_temperatures (F : ℤ) (h1 : 32 ≤ F) (h2 : F ≤ 2000) :
  (F = Int.round ((9 / 5 : ℚ) * (Int.round ((5 / 9 : ℚ) * (F - 32))) + 32) ↔
   F = Int.round (9 * (Int.round ((5 / 9 : ℚ) * (F - 32)) / 5 + 32))) :=
sorry

theorem count_valid_fahrenheit_temperatures :
  {F : ℤ | 32 ≤ F ∧ F ≤ 2000 ∧ F = Int.round ((9 / 5 : ℚ) * (Int.round ((5 / 9 : ℚ) * (F - 32))) + 32)}.card = 1095 :=
sorry

end valid_fahrenheit_temperatures_count_valid_fahrenheit_temperatures_l146_146970


namespace power_equality_l146_146328

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146328


namespace minimize_waiting_time_l146_146103

-- Given that there are 10 individuals with different time requirements t_i
variables {t : Fin 10 → ℕ}

-- Definition of the total waiting time W based on the order of filling
def waiting_time (order : Fin 10 → Fin 10) : ℕ :=
  ∑ i, (10 - i) * t (order i)

-- Statement to prove that ordering by increasing fill times minimizes waiting time
theorem minimize_waiting_time (h : ∀ i j : Fin 10, i < j ↔ t i < t j) :
  ∃ order : Fin 10 → Fin 10, 
    (∀ i, ∃ j, order j = i) ∧ (∑ i, (10 - i) * t (order i) = waiting_time (λ i, i)) :=
sorry

end minimize_waiting_time_l146_146103


namespace find_BG_l146_146069

-- Define given lengths and the required proof
def BC : ℝ := 5
def BF : ℝ := 12

theorem find_BG : BG = 13 := by
  -- Formal proof would go here
  sorry

end find_BG_l146_146069


namespace max_value_of_parabola_l146_146785

theorem max_value_of_parabola : ∀ x : ℝ, -2 * x^2 + 8 ≤ 8 :=
by
  intro x
  have : -2 * x^2 + 8 ≤ 8 := le_refl 8 -- The actual proof logic will be more involved.
  exact this

end max_value_of_parabola_l146_146785


namespace num_solutions_l146_146193

noncomputable def sign (a : ℝ) : ℝ :=
if a > 0 then 1 else if a < 0 then -1 else 0

def satisfies_equations (x y z : ℝ) : Prop :=
  x = 1000 - 1001 * sign (y + z + 1) ∧
  y = 1000 - 1001 * sign (x + z - 1) ∧
  z = 1000 - 1001 * sign (x + y + 2)

theorem num_solutions : 
  {triple : ℝ × ℝ × ℝ // satisfies_equations triple.1 triple.2 triple.3}.to_finset.card = 3 :=
sorry

end num_solutions_l146_146193


namespace mass_of_man_l146_146573

-- Definitions based on problem conditions
def boat_length : ℝ := 8
def boat_breadth : ℝ := 3
def sinking_height : ℝ := 0.01
def water_density : ℝ := 1000

-- Mass of the man to be proven
theorem mass_of_man : boat_density * (boat_length * boat_breadth * sinking_height) = 240 :=
by
  sorry

end mass_of_man_l146_146573


namespace tangent_and_normal_lines_l146_146096

noncomputable def x (t : ℝ) := 2 * Real.exp t
noncomputable def y (t : ℝ) := Real.exp (-t)

theorem tangent_and_normal_lines (t0 : ℝ) (x0 y0 : ℝ) (m_tangent m_normal : ℝ)
  (hx0 : x0 = x t0)
  (hy0 : y0 = y t0)
  (hm_tangent : m_tangent = -(1 / 2))
  (hm_normal : m_normal = 2) :
  (∀ x y : ℝ, y = m_tangent * x + 2) ∧ (∀ x y : ℝ, y = m_normal * x - 3) :=
by
  sorry

end tangent_and_normal_lines_l146_146096


namespace power_of_three_l146_146365

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146365


namespace smallest_m_inequality_l146_146561

theorem smallest_m_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l146_146561


namespace hyperbola_eccentricity_l146_146379

-- Define the hyperbola and the condition of the asymptote passing through (2,1)
def hyperbola (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧
               (a ≠ 0 ∧ b ≠ 0) ∧
               (x, y) = (2, 1)

-- Define the eccentricity of the hyperbola
def eccentricity (a b e : ℝ) : Prop :=
  a^2 + b^2 = (b * e)^2

theorem hyperbola_eccentricity (a b e : ℝ) 
  (hx : hyperbola a b)
  (ha : a = 2 * b)
  (ggt: (a^2 = 4 * b^2)) :
  eccentricity a b e → e = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_eccentricity_l146_146379


namespace find_y_satisfying_log_l146_146187

theorem find_y_satisfying_log (y : ℝ) : log 8 (y + 8) = 5 / 3 → y = 24 := 
by 
  intro h
  sorry

end find_y_satisfying_log_l146_146187


namespace power_addition_l146_146294

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146294


namespace ideal_gas_efficiency_l146_146485

open Real

/-- Define the conditions for the ideal gas process. -/
def pressure_temperature_relation (P T : ℝ) : Prop := P ^ 2 = T

def ideal_gas_law (P V n R T : ℝ) : Prop := P * V = n * R * T

def change_in_internal_energy (ΔU n R ΔT : ℝ) : Prop := ΔU = (3 / 2) * n * R * ΔT

/-- Define the efficiency calculation. -/
def efficiency (A ΔU : ℝ) : ℝ :=
  A / (ΔU + A)

/-- Given the conditions, we want to prove the efficiency is 0.25. -/
theorem ideal_gas_efficiency
  (P V n R T ΔU A : ℝ)
  (PT_relation : pressure_temperature_relation P T)
  (ig_law : ideal_gas_law P V n R T)
  (internal_energy : change_in_internal_energy ΔU n R (T - T))
  (work_done : A = (3 / 2) * P * V)
  (U_change : ΔU = (9 / 2) * P * V) :
  efficiency A ΔU = 0.25 :=
sorry

end ideal_gas_efficiency_l146_146485


namespace chinese_volleyball_team_winning_probability_l146_146393

noncomputable def probability_of_winning_best_of_five (p : ℝ) : ℝ :=
  let q := 1 - p in
  let P_3 := (5.choose 3) * (p ^ 3) * (q ^ 2) in
  let P_4 := (5.choose 4) * (p ^ 4) * q in
  let P_5 := p ^ 5 in
  P_3 + P_4 + P_5

theorem chinese_volleyball_team_winning_probability :
  probability_of_winning_best_of_five 0.4 = 0.32 :=
sorry

end chinese_volleyball_team_winning_probability_l146_146393


namespace hexagon_chord_length_l146_146117

theorem hexagon_chord_length {p q : ℕ} (h_rel_prime : Nat.gcd p q = 1)
    (h_hexagon : (∃ (A B C D E F : ℝ), 
               let sides := [A, B, C, D, E, F] in
               [A = 4, B = 4, C = 4, D = 7, E = 7, F = 7] ∧
               inscribed_in_circle sides ∧ 
               ∃ (chord_length : ℝ),
               divides_hexagon sides chord_length ∈ (p / q)))
               : p + q = 1017 := sorry

end hexagon_chord_length_l146_146117


namespace range_of_a_l146_146714

theorem range_of_a (a : ℝ) (h₀ : a > 0) : (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 :=
sorry

end range_of_a_l146_146714


namespace max_colouoring_sets_l146_146949

-- Define the conditions
def total_points : ℕ := 2012

-- Define the distinct colour counts condition
def distinct_colour_counts (n : ℕ) (counts : Fin n → ℕ) : Prop :=
  ∀ i j : Fin n, i ≠ j → counts i ≠ counts j

-- Define the valid colouring condition
def valid_colouring (n : ℕ) (counts : Fin n → ℕ) : Prop :=
  distinct_colour_counts n counts ∧ 
  (counts.toList.sum = total_points)

-- Define the multicoloured set condition
def multi_coloured_sets (n : ℕ) (counts : Fin n → ℕ) : ℕ :=
  counts.toList.prod

-- Define the maximum n theorem
theorem max_colouoring_sets : ∃ n counts, valid_colouring n counts ∧ n = 61 := 
by
  sorry

end max_colouoring_sets_l146_146949


namespace largest_value_l146_146085

noncomputable def expr1 : ℝ := 15847 + (1 / 3174)
noncomputable def expr2 : ℝ := 15847 - (1 / 3174)
noncomputable def expr3 : ℝ := 15847 * (1 / 3174)
noncomputable def expr4 : ℝ := 15847 / (1 / 3174)
noncomputable def expr5 : ℝ := 15847 ^ 1.03174

theorem largest_value :
  max (max (max (max expr1 expr2) expr3) expr4) expr5 = expr4 :=
  sorry

end largest_value_l146_146085


namespace range_of_a_when_min_f_ge_neg_a_l146_146245

noncomputable def f (a x : ℝ) := a * Real.log x + 2 * x

theorem range_of_a_when_min_f_ge_neg_a (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x > 0, f a x ≥ -a) :
  -2 ≤ a ∧ a < 0 :=
sorry

end range_of_a_when_min_f_ge_neg_a_l146_146245


namespace find_phi_l146_146937

noncomputable def phi_value : ℝ := π / 12

theorem find_phi (φ : ℝ) (h1 : -π / 2 < φ) (h2 : φ ≤ π / 2) :
  ∀ k : ℤ, φ = -2 * k * π + (π / 12) → φ = phi_value := 
by
  intro k
  intro h3
  simp [phi_value, h3]
  contradiction  -- Here we assume the solution leads to the contradiction as k should be 0 according to the solution. You could refine if more context is required.
  sorry

end find_phi_l146_146937


namespace tangent_line_equation_at_1_range_of_a_l146_146249

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_equation_at_1 (a : ℝ) (h : a = 1) : 
  let f_1 := f 1 1 in
  let f_prime (x : ℝ) := Real.log x + 1 / x in
  let f_prime_at_1 := f_prime 1 in
  f_1 = 0 ∧ f_prime_at_1 = 1 ∧ 
  (∀ x y : ℝ, (x, y) = (1, f_1) → (x - y - 1 = 0)) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 < x → f x a > 0) → a ≤ 2 := sorry

end tangent_line_equation_at_1_range_of_a_l146_146249


namespace angle_between_a_b_l146_146234

noncomputable def vector_a : ℝ × ℝ := (1, -Real.sqrt 3)

noncomputable def vector_b : ℝ × ℝ
  | (x, y) := sorry -- b is arbitrary unit vector

axiom b_magnitude : Real.sqrt (vector_b.1 ^ 2 + vector_b.2 ^ 2) = 1
axiom a_plus_2b_magnitude : Real.sqrt ((vector_a.1 + 2 * vector_b.1) ^ 2 + (vector_a.2 + 2 * vector_b.2) ^ 2) = 2

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.acos (dot_product v w / (magnitude v * magnitude w))

theorem angle_between_a_b : angle_between vector_a vector_b = 2 * Real.pi / 3 :=
  sorry

end angle_between_a_b_l146_146234


namespace simplify_complex_expression_l146_146462

theorem simplify_complex_expression : 
  ∀ (i : ℂ), (i ^ 2 = -1) → (3 * (4 - 2 * i) + 2 * i * (3 - i) = 14) :=
by
  intro i
  intro hi
  sorry

end simplify_complex_expression_l146_146462


namespace axis_of_symmetry_l146_146925

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x

-- The theorem to prove the axis of symmetry
theorem axis_of_symmetry :
  ∃ c : ℝ, (∀ x : ℝ, parabola x = (x - c) ^ 2 + (parabola c - c^2)) ∧ c = 1 :=
begin
  -- sorry is used to skip the proof
  sorry
end

end axis_of_symmetry_l146_146925


namespace angle_between_CE_and_BF_is_75_l146_146908

noncomputable def acute_angle_between_lines (A B C E F : Point) : ℝ :=
  if h : (AC = AB) ∧ (square_construction A B D E) ∧ (equilateral_triangle_construction C F A) ∧ same_side_line A B C then
    75
  else sorry

theorem angle_between_CE_and_BF_is_75 (A B C D E F : Point) :
  let h : (C = A - B)
  (AC = AB) →
  square_construction A B D E →
  equilateral_triangle_construction C F A →
  same_side_line A B C →
  acute_angle_between_lines A B C E F = 75 :=
by
  intros
  sorry

end angle_between_CE_and_BF_is_75_l146_146908


namespace probability_of_hitting_four_times_in_five_shots_l146_146045

-- Definitions based on problem conditions
variable (p : ℝ) -- p is the probability of hitting the target in a single shot
variable (P_two_hit : ℝ) -- probability of hitting the target at least once in two shots, given as 0.96

-- Auxiliary definitions derived from conditions
def q := 1 - p -- q is the probability of missing the target in a single shot
def P_hit_at_least_once_in_two_shots := 1 - (q * q)
def P_5_4 := (5.choose 4) * p^4 * q -- Binomial probability of 4 hits in 5 shots

-- Theorem statement: given conditions, prove the required probability
theorem probability_of_hitting_four_times_in_five_shots :
  P_two_hit = 0.96 → p = 0.8 → P_5_4 = 0.4096 := 
by
  sorry

end probability_of_hitting_four_times_in_five_shots_l146_146045


namespace integral_value_l146_146559

noncomputable def definite_integral := ∫ x in 0..(π/4), (sin x - cos x) / ((cos x + sin x)^5)

theorem integral_value : definite_integral = -3/16 := 
by
  sorry

end integral_value_l146_146559


namespace car_passing_problem_l146_146443

noncomputable def maxCarsPerHourDividedBy10 : ℕ :=
  let unit_length (n : ℕ) := 5 * (n + 1)
  let cars_passed_in_one_hour (n : ℕ) := 10000 * n / unit_length n
  Nat.div (2000) (10)

theorem car_passing_problem : maxCarsPerHourDividedBy10 = 200 :=
  by
  sorry

end car_passing_problem_l146_146443


namespace initial_nickels_l146_146009

/-- Sandy initially had 31 nickels before her dad borrowed 20 nickels. -/
theorem initial_nickels (initial_nickels : ℕ) (borrowed_nickels : ℕ) (remaining_nickels : ℕ) 
  (h1 : borrowed_nickels = 20) (h2 : remaining_nickels = 11) (h3 : remaining_nickels + borrowed_nickels = initial_nickels) : 
  initial_nickels = 31 :=
by
  unfold initial_nickels borrowed_nickels remaining_nickels at *
  rw [h1, h2] at h3
  linarith

end initial_nickels_l146_146009


namespace hexagon_area_l146_146079

noncomputable def radius_of_circle (A : ℝ) : ℝ :=
  real.sqrt (A / π)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (s ^ 2 * real.sqrt 3) / 4

noncomputable def area_of_regular_hexagon (r : ℝ) : ℝ :=
  6 * area_of_equilateral_triangle r

theorem hexagon_area {A : ℝ} (hA : A = 576 * π) :
  area_of_regular_hexagon (radius_of_circle A) = 864 * real.sqrt 3 :=
by
  sorry

end hexagon_area_l146_146079


namespace simplify_cube_root_l146_146916

theorem simplify_cube_root : ∀ (a b c : ℝ), a = 50 → b = 60 → c = 70 → 
  (∛(a^3 + b^3 + c^3)) = 10 * ∛684 := 
by
  intros a b c h1 h2 h3
  rw [h1, h2, h3]
  sorry

end simplify_cube_root_l146_146916


namespace exists_four_distinct_numbers_with_equal_sums_l146_146741

theorem exists_four_distinct_numbers_with_equal_sums (S : Finset ℕ) (hS : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 37) (h_card : S.card = 10) :
  ∃ a b c d ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by
  sorry

end exists_four_distinct_numbers_with_equal_sums_l146_146741


namespace power_of_3_l146_146357

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146357


namespace half_minus_quarter_of_number_is_two_l146_146102

/-
0.5 of a number is a certain amount more than 0.25 of the number. The number is 8. How much more is 0.5 of the number than 0.25 of the number?
-/

theorem half_minus_quarter_of_number_is_two (n : ℝ) (h : n = 8) : 0.5 * n - 0.25 * n = 2 :=
by
  rw [h]
  norm_num
  sorry

end half_minus_quarter_of_number_is_two_l146_146102


namespace solve_equation_l146_146552

theorem solve_equation :
  let lhs := ((4 - 3.5 * (15/7 - 6/5)) / 0.16)
  let rhs := ((23/7 - (3/14) / (1/6)) / (3467/84 - 2449/60))
  lhs / 1 = rhs :=
by
  sorry

end solve_equation_l146_146552


namespace find_a_l146_146217

noncomputable def z1 (a : ℝ) : ℂ := -4 * a + 1 + complex.i * (2 * a ^ 2 + 3 * a)
noncomputable def z2 (a : ℝ) : ℂ := 2 * a + complex.i * (a ^ 2 + a)

def condition (a : ℝ) := z1 a.re > z2 a.re ∧ (∃ x : ℝ, a = x)

theorem find_a (a : ℝ) (h : condition a) : a = 0 := 
by sorry

end find_a_l146_146217


namespace eta_same_distribution_as_xi_l146_146868

open MeasureTheory

-- Definitions of the sequences and stopping time
variables {Ω : Type*} {F : MeasurableSpace Ω} {P : MeasureTheory.ProbabilityMeasure Ω}
variables (ξ : ℕ → Ω → ℝ) (τ : Ω → ℕ)

-- Conditions on the sequences and stopping time
def i.i.d_sequence (ξ : ℕ → Ω → ℝ) : Prop := 
  ∀ (n : ℕ), MeasureTheory.IndepFun (λ ω, ξ n ω) (λ ω, ξ (n + 1) ω) P ∧
  ∀ n m, MeasureTheory.Probability (λ ω, ξ n ω = ξ m ω)

def filtration := λ n : ℕ, MeasureTheory.GenerateFrom (λ ω, ξ n ω)

def is_stopping_time (τ : Ω → ℕ) (ℱ : ℕ → MeasurableSpace Ω) : Prop :=
  ∀ t : ℕ, {ω | τ ω = t} ∈ ℱ t

-- The sequence η defined in terms of τ
def η (ξ : ℕ → Ω → ℝ) (τ : Ω → ℕ) : ℕ → Ω → ℝ := λ n ω, ξ (n + τ ω) ω

-- The theorem statement
theorem eta_same_distribution_as_xi (ξ : ℕ → Ω → ℝ) (τ : Ω → ℕ)
  (h_iid : i.i.d_sequence ξ) 
  (ℱ : ℕ → MeasurableSpace Ω) (h_fil : ∀ n, ℱ n = filtration ξ n)
  (h_stopping : is_stopping_time τ ℱ) :
  ∀ (B : Set ℝ), 
  MeasureTheory.Probability (λ ω, (η ξ τ 1 ω) ∈ B) = 
  MeasureTheory.Probability (λ ω, ξ 1 ω ∈ B) := 
sorry

end eta_same_distribution_as_xi_l146_146868


namespace nth_equation_l146_146899

-- Define the sequence of sums on the left-hand side.
def nth_sum (n : ℕ) : ℕ := 
  (Finset.range (2 * n - 1)).sum (λ k, n + k)

-- The theorem formalizing the pattern identified.
theorem nth_equation (n : ℕ) : nth_sum n = (2 * n - 1) ^ 2 := by
  sorry

end nth_equation_l146_146899


namespace power_equality_l146_146327

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146327


namespace y_intercept_of_line_l146_146057

theorem y_intercept_of_line (x y : ℝ) (h : x - y + 3 = 0) : y = 3 := 
by 
  have h_0 := h
  simp at h_0
  -- Substitute x = 0 into the equation
  specialize h_0 0
  -- Simplify equation
  simp at h_0
  exact h_0
  sorry

end y_intercept_of_line_l146_146057


namespace min_static_friction_coeff_l146_146139

theorem min_static_friction_coeff (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π / 2) : 
  (∃ μ : ℝ, μ ≥ 1 / (2 * sqrt 2)) :=
by
  sorry

end min_static_friction_coeff_l146_146139


namespace cookies_on_first_plate_l146_146054

theorem cookies_on_first_plate :
  ∃ a1 a2 a3 a4 a5 a6 : ℤ, 
  a2 = 7 ∧ 
  a3 = 10 ∧
  a4 = 14 ∧
  a5 = 19 ∧
  a6 = 25 ∧
  a2 = a1 + 2 ∧ 
  a3 = a2 + 3 ∧ 
  a4 = a3 + 4 ∧ 
  a5 = a4 + 5 ∧ 
  a6 = a5 + 6 ∧ 
  a1 = 5 :=
sorry

end cookies_on_first_plate_l146_146054


namespace right_triangle_of_three_colors_exists_l146_146718

-- Define the type for color
inductive Color
| color1
| color2
| color3

open Color

-- Define the type for integer coordinate points
structure Point :=
(x : ℤ)
(y : ℤ)
(color : Color)

-- Define the conditions
def all_points_colored : Prop :=
∀ (p : Point), p.color = color1 ∨ p.color = color2 ∨ p.color = color3

def all_colors_used : Prop :=
∃ (p1 p2 p3 : Point), p1.color = color1 ∧ p2.color = color2 ∧ p3.color = color3

-- Define the right_triangle_exist problem
def right_triangle_exists : Prop :=
∃ (p1 p2 p3 : Point), 
  p1.color ≠ p2.color ∧ p2.color ≠ p3.color ∧ p3.color ≠ p1.color ∧
  (p1.x = p2.x ∧ p2.y = p3.y ∧ p1.y = p3.y ∨
   p1.y = p2.y ∧ p2.x = p3.x ∧ p1.x = p3.x ∨
   (p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y) = (p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) ∧
   (p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y) = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y))

theorem right_triangle_of_three_colors_exists (h1 : all_points_colored) (h2 : all_colors_used) : right_triangle_exists := 
sorry

end right_triangle_of_three_colors_exists_l146_146718


namespace sum_greatest_odd_divisors_property_l146_146415

-- Defining the sum of the greatest odd divisors
def greatest_odd_divisor (n : ℕ) : ℕ :=
if n = 0 then 0 else
let rec helper x :=
  if x % 2 = 1 then x else helper (x / 2)
in helper n

def sum_greatest_odd_divisors (n : ℕ) : ℕ :=
(nat.range (2^n + 1)).sum greatest_odd_divisor

theorem sum_greatest_odd_divisors_property (n : ℕ) : 
  3 * sum_greatest_odd_divisors n = 4^n + 2 := 
sorry

end sum_greatest_odd_divisors_property_l146_146415


namespace triangle_number_arrangement_l146_146774

noncomputable def num_ways_to_write_1_to_9_in_triangle : Prop :=
  ∃ (s t : ℕ) (a b c : ℕ) (x y u v m n : ℕ),
    {a, b, c} = {3, 6, 9} ∨ {a, b, c} = {1, 4, 7} ∨ {a, b, c} = {2, 5, 8} ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    {x, y, u, v, m, n} = {1, 2, 4, 5, 6, 7, 8, 9} \ {a, b, c} ∧
    s = (a + x + y + b) ∧ s = (b + u + v + c) ∧ s = (c + m + n + a) ∧
    t = (a^2 + x^2 + y^2 + b^2) ∧ t = (b^2 + u^2 + v^2 + c^2) ∧ t = (c^2 + m^2 + n^2 + a^2) ∧
    s = ((a + b + c + 45) / 3) ∧ t = ((a^2 + b^2 + c^2 + 285) / 3) ∧
    (x^2 + y^2 = 58) ∧ (u^2 + v^2 = 97) ∧ (m^2 + n^2 = 37)

theorem triangle_number_arrangement : num_ways_to_write_1_to_9_in_triangle :=
  sorry

end triangle_number_arrangement_l146_146774


namespace triangle_inequality_l146_146099

variables {A B C A1 B1 C1 : Type} [EuclideanGeometry A] [EuclideanGeometry A1]
variables {AB AC BC A1B1 A1C1 B1C1 : ℝ}
variables {angleA angleA1 : ℝ}

theorem triangle_inequality 
  (h1 : AB = A1B1)
  (h2 : AC = A1C1)
  (h3 : angleA > angleA1) :
  BC > B1C1 :=
sorry

end triangle_inequality_l146_146099


namespace problem_solution_l146_146426

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x ^ Real.sqrt 3 + y ^ Real.sqrt 3 = 189 :=
sorry

end problem_solution_l146_146426


namespace binom_9_5_eq_126_l146_146681

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146681


namespace usb_drive_available_space_l146_146901

theorem usb_drive_available_space (C P : ℝ) (hC : C = 16) (hP : P = 50) : 
  (1 - P / 100) * C = 8 :=
by
  sorry

end usb_drive_available_space_l146_146901


namespace infinite_equidistant_points_l146_146013

noncomputable theory

variables {A B C D E F G H : Point}
variables {BC DH EF : Line}
variables (cube : Cube)
variables (body_diagonal : Line)

def skew_lines : Prop := 
  ∀ (l1 l2 : Line), (l1 ≠ l2) → ¬ ∃ P : Point, P ∈ l1 ∧ P ∈ l2

def equidistant_from_lines (P : Point) : Prop := 
  dist P BC = dist P DH ∧ dist P DH = dist P EF

def on_body_diagonal (P : Point) : Prop := 
  P ∈ body_diagonal

axiom cube_structure : 
  is_cube cube ∧ vertices cube = {A, B, C, D, E, F, G, H} ∧
  edges cube = {BC, DH, EF} ∧
  body_diagonal = Line_through A G ∧
  skew_lines BC DH ∧ skew_lines DH EF ∧ skew_lines BC EF

theorem infinite_equidistant_points : 
  ∃ (P : Point) (l : Line), 
  l = body_diagonal ∧ (∀ Q : Point, Q ∈ l → equidistant_from_lines Q) :=
sorry

end infinite_equidistant_points_l146_146013


namespace sum_of_rearranged_digits_ne_999_125_nines_l146_146392

theorem sum_of_rearranged_digits_ne_999_125_nines
  (a b : ℕ)
  (a_rearranged : ∃ permutation, b = permute_digits permutation a) :
  a + b ≠ (∑ i in finset.range 125, 9 * 10^i) :=
by
  sorry

end sum_of_rearranged_digits_ne_999_125_nines_l146_146392


namespace sqrt_a_plus_sqrt_b_eq_3_l146_146380

theorem sqrt_a_plus_sqrt_b_eq_3 (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) : Real.sqrt a + Real.sqrt b = 3 :=
sorry

end sqrt_a_plus_sqrt_b_eq_3_l146_146380


namespace solve_sum_squares_roots_l146_146563

theorem solve_sum_squares_roots :
  ∀ (x : ℂ), (x^2 + |x| + 1 = 0) → (x^2 + (-x)^2 = - (3 + sqrt 5)) := 
by
  intro x
  intro h
  sorry

end solve_sum_squares_roots_l146_146563


namespace increasing_intervals_area_triangle_l146_146777

-- Conditions
def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)
def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Question (1)
def isMonotonicallyIncreasing (f : ℝ → ℝ) (I : Set.Icc) := ∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- Question (2)
def areaOfTriangle (a b c : ℝ) (A : ℝ) := 1 / 2 * b * c * Real.sin A

-- Proofs (using sorry to skip the actual proof)
theorem increasing_intervals : 
  ∀ k : ℤ, isMonotonicallyIncreasing (f) (Set.Icc (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi)) :=
by sorry 

theorem area_triangle (A : ℝ) (a b c : ℝ) : 
  f A = 2 → a = Real.sqrt 7 → b = 2 * c → A = Real.pi / 3 → c = Real.sqrt 21 / 3 → 
  areaOfTriangle a b c A = 7 * Real.sqrt 3 / 6 :=
by sorry

end increasing_intervals_area_triangle_l146_146777


namespace find_angle_E_l146_146849

-- Defining the trapezoid properties and angles
variables (EF GH : ℝ) (E H G F : ℝ)
variables (trapezoid_EFGH : Prop) (parallel_EF_GH : Prop)
variables (angle_E_eq_3H : Prop) (angle_G_eq_4F : Prop)

-- Conditions
def trapezoid_EFGH : Prop := ∃ E F G H EF GH, EF ≠ GH
def parallel_EF_GH : Prop := EF ∥ GH
def angle_E_eq_3H : Prop := E = 3 * H
def angle_G_eq_4F : Prop := G = 4 * F

-- Theorem statement
theorem find_angle_E (H_value : ℝ) (H_property : H = 45) :
  E = 135 :=
  by
  -- Assume necessary properties from the problem statements
  assume trapezoid_EFGH
  assume parallel_EF_GH : EF ∥ GH
  assume angle_E_eq_3H : E = 3 * H
  have H_value : H = 45 := sorry
  have angle_E_value : E = 135 := sorry
  exact angle_E_value

end find_angle_E_l146_146849


namespace prime_factorization_correct_l146_146162

-- Define the large number N
def N : ℕ := 1007021035035021007001

-- Define the prime factorization form of 1001
def factorize_1001 : ℕ := 1001
def prime_factors_1001 : list ℕ := [7, 11, 13]

-- Define the exponentiation of the factorized form
def exponent : ℕ := 7
def prime_factorization_N : list (ℕ × ℕ) :=
  (prime_factors_1001.map (λ p => (p, exponent)))

-- The main statement asserting the prime factorization
theorem prime_factorization_correct :
  N = 7^7 * 11^7 * 13^7 := sorry

end prime_factorization_correct_l146_146162


namespace equal_areas_centroid_l146_146531

def centroid (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ × ℝ :=
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

theorem equal_areas_centroid (m n : ℝ) (h : (m, n) = centroid 1 7 2 (-3) 8 4) :
  10 * m + n = 118 / 3 :=
by
  rw h
  simp [centroid]
  sorry

end equal_areas_centroid_l146_146531


namespace power_calculation_l146_146311

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146311


namespace power_of_three_l146_146287

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146287


namespace height_of_pole_l146_146120

-- Definitions for the conditions
def ascends_first_minute := 2
def slips_second_minute := 1
def net_ascent_per_two_minutes := ascends_first_minute - slips_second_minute
def total_minutes := 17
def pairs_of_minutes := (total_minutes - 1) / 2  -- because the 17th minute is separate
def net_ascent_first_16_minutes := pairs_of_minutes * net_ascent_per_two_minutes

-- The final ascent in the 17th minute
def ascent_final_minute := 2

-- Total ascent
def total_ascent := net_ascent_first_16_minutes + ascent_final_minute

-- Statement to prove the height of the pole
theorem height_of_pole : total_ascent = 10 :=
by
  sorry

end height_of_pole_l146_146120


namespace determine_unique_quadratic_l146_146440

open Set

theorem determine_unique_quadratic (a b c d: ℝ) :
  (∀ x y z w, {x, y, z, w} ⊆ {a, b, c, d} → x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) →
  ∃! (p q r s: ℝ), p = a ∧ q = b ∧ r = c ∧ s = d ∧ {p, q, r, s} = {a, b, c, d} :=
by
  intros H
  hint sorry  -- Proof would be provided here

#check determine_unique_quadratic

end determine_unique_quadratic_l146_146440


namespace range_of_a_l146_146822

theorem range_of_a {a : ℝ} :
  (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 4) ↔ (-2*Real.sqrt 2 < a ∧ a < 2*Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end range_of_a_l146_146822


namespace proved_problem_l146_146638

noncomputable def problem_statement : Prop :=
  nat.choose 1011 7 % 1000 = 11

theorem proved_problem : problem_statement := 
  by
    sorry  -- This will hold the proof of the theorem

end proved_problem_l146_146638


namespace num_values_f_50_eq_12_l146_146204

-- Define the number of divisors of n
def num_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).length

-- Define f_1(n) as twice the number of positive integer divisors of n
def f_1 (n : ℕ) : ℕ :=
  2 * num_divisors n

-- Define f_j(n) for j >= 2 recursively
def f_j : ℕ → ℕ → ℕ
| 1, n := f_1 n
| (j + 1), n := f_1 (f_j j n)

-- Prove the number of values of n <= 50 such that f_50(n) = 12
theorem num_values_f_50_eq_12 : (Finset.range 50).filter (λ n, f_j 50 (n + 1) = 12).card = 10 := by
  sorry

end num_values_f_50_eq_12_l146_146204


namespace min_value_of_a_l146_146256

noncomputable def f (x a : ℝ) : ℝ :=
  Real.exp x * (x + (3 / x) - 3) - (a / x)

noncomputable def g (x : ℝ) : ℝ :=
  (x^2 - 3 * x + 3) * Real.exp x

theorem min_value_of_a (a : ℝ) :
  (∃ x > 0, f x a ≤ 0) → a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l146_146256


namespace binomial_coefficient_9_5_l146_146671

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146671


namespace toothpick_sequence_l146_146488

theorem toothpick_sequence (a d n : ℕ) (h1 : a = 6) (h2 : d = 4) (h3 : n = 150) : a + (n - 1) * d = 602 := by
  sorry

end toothpick_sequence_l146_146488


namespace power_calculation_l146_146316

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146316


namespace wilsons_theorem_l146_146883

theorem wilsons_theorem (p : ℕ) (hp1 : p > 1) : 
  prime p ↔ factorial (p - 1) ≡ -1 [MOD p] :=
by
  sorry

end wilsons_theorem_l146_146883


namespace power_addition_l146_146300

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146300


namespace power_of_three_l146_146364

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146364


namespace abs_log_eq_b_two_solns_b_zero_l146_146247

theorem abs_log_eq_b_two_solns_b_zero (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : ∃ x1 x2 : ℝ, ∀ x : ℝ, ((|log a (|x + b|)| = b) ↔ (x = x1 ∨ x = x2))) :
  b = 0 := sorry

end abs_log_eq_b_two_solns_b_zero_l146_146247


namespace largest_multiple_of_15_less_than_500_l146_146976

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146976


namespace length_of_XY_l146_146826

theorem length_of_XY (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (a : Angle X Y Z) (angle_X : a = π / 2)
  (tan_Z : Real) (tan_Z_value : tan_Z = 6)
  (hypotenuse : Real) (hypotenuse_value : hypotenuse = 50)
  (XY XZ YZ : Real)
  (triangle_cond : XY^2 + XZ^2 = YZ^2)
  (tan_cond : tan_Z = XY / XZ) :
  XY = 6 * Real.sqrt (2500 / 37) :=
by {
  sorry
}

end length_of_XY_l146_146826


namespace sin_value_of_given_conditions_l146_146772

noncomputable def sin_of_angle (m : ℝ) (h1 : tan α = 3/4) (h2 : (m, 9) ∈ TrigFunc.terminalSide α) : ℝ :=
  sin α

theorem sin_value_of_given_conditions (m : ℝ) (α : ℝ) (h_tan : tan α = 3 / 4) (h_point : (m = 12 ∧ 9) = (12, 9)) : 
  sin α = 3 / 5 :=
by
  sorry

end sin_value_of_given_conditions_l146_146772


namespace determinant_zero_l146_146650

open Matrix

variables {R : Type*} [Field R] {a b : R}

def M : Matrix (Fin 3) (Fin 3) R :=
  ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ]

theorem determinant_zero : det M = 0 :=
by
  sorry

end determinant_zero_l146_146650


namespace complement_union_l146_146893

def U : set ℕ := {1, 2, 3, 4, 5}
def A : set ℕ := {1, 2}
def B : set ℕ := {2, 4}

theorem complement_union :
  U \ (A ∪ B) = {3, 5} := by
  sorry

end complement_union_l146_146893


namespace sum_of_cubes_l146_146819

theorem sum_of_cubes (x y : ℂ) (h1 : x + y = 1) (h2 : x * y = 1) : x^3 + y^3 = -2 := 
by 
  sorry

end sum_of_cubes_l146_146819


namespace colten_chickens_l146_146000

variable (C Q S : ℕ)

-- Conditions
def condition1 : Prop := Q + S + C = 383
def condition2 : Prop := Q = 2 * S + 25
def condition3 : Prop := S = 3 * C - 4

-- Theorem to prove
theorem colten_chickens : condition1 C Q S ∧ condition2 C Q S ∧ condition3 C Q S → C = 37 := by
  sorry

end colten_chickens_l146_146000


namespace angle_between_a_b_l146_146235

noncomputable def vector_a : ℝ × ℝ := (1, -Real.sqrt 3)

noncomputable def vector_b : ℝ × ℝ
  | (x, y) := sorry -- b is arbitrary unit vector

axiom b_magnitude : Real.sqrt (vector_b.1 ^ 2 + vector_b.2 ^ 2) = 1
axiom a_plus_2b_magnitude : Real.sqrt ((vector_a.1 + 2 * vector_b.1) ^ 2 + (vector_a.2 + 2 * vector_b.2) ^ 2) = 2

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.acos (dot_product v w / (magnitude v * magnitude w))

theorem angle_between_a_b : angle_between vector_a vector_b = 2 * Real.pi / 3 :=
  sorry

end angle_between_a_b_l146_146235


namespace betty_garden_total_plants_l146_146632

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l146_146632


namespace james_milk_left_l146_146854

-- Definitions
def gallons_to_ounces (gallons : ℕ) : ℕ := gallons * 128
def ounces_consumed (a b c : ℕ) : ℕ := a + b + c
def milk_left (initial consumed : ℕ) : ℕ := initial - consumed
def ounces_to_liters (ounces : ℕ) : ℝ := ounces * 0.0295735

-- Variables
variable (initial_gallons : ℕ)
variable (james_drink : ℕ)
variable (sarah_drink : ℕ)
variable (mark_drink : ℕ)

-- Problem Statement
theorem james_milk_left :
  initial_gallons = 3 →
  james_drink = 13 →
  sarah_drink = 20 →
  mark_drink = 25 →
  let initial_ounces := gallons_to_ounces initial_gallons in
  let consumed_ounces := ounces_consumed james_drink sarah_drink mark_drink in
  let remaining_ounces := milk_left initial_ounces consumed_ounces in
  let remaining_liters := ounces_to_liters remaining_ounces in
  remaining_ounces = 326 ∧ Real.ApproxEq (remaining_liters:ℝ) 9.64.toReal (.001) :=
by
  sorry

end james_milk_left_l146_146854


namespace largest_multiple_of_15_less_than_500_l146_146978

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146978


namespace contradiction_method_assumption_l146_146958

-- Definitions for three consecutive positive integers
variables {a b c : ℕ}

-- Definitions for the proposition and its negation
def consecutive_integers (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1
def at_least_one_divisible_by_2 (a b c : ℕ) : Prop := a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0
def all_not_divisible_by_2 (a b c : ℕ) : Prop := a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem contradiction_method_assumption (a b c : ℕ) (h : consecutive_integers a b c) :
  (¬ at_least_one_divisible_by_2 a b c) ↔ all_not_divisible_by_2 a b c :=
by sorry

end contradiction_method_assumption_l146_146958


namespace area_ratio_triangles_l146_146131

theorem area_ratio_triangles (k l p : ℕ) (L W : ℝ) (hk : k > 0) (hl : l > 0) (hp : p > 0) (hL : L > 0) (hW : W > 0) :
  let area_C := (1 / 2) * (W / k) * (L / l) in
  let area_D := (1 / 2) * (W / p) * (L / p) in
  area_C / area_D = (p^2 : ℝ) / (k * l : ℝ) :=
by
  sorry

end area_ratio_triangles_l146_146131


namespace sum_square_fib_eq_mul_fib_l146_146475

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

theorem sum_square_fib_eq_mul_fib (n : ℕ) : (∑ k in Finset.range (n + 1), (fib k)^2) = fib n * fib (n + 1) := 
by 
  sorry

end sum_square_fib_eq_mul_fib_l146_146475


namespace area_of_annulus_correct_l146_146514

noncomputable def area_between_circles : ℝ :=
  let C := 0  -- center at some arbitrary point, say 0 for simplicity
  let A := complex.of_real 10  -- AC = 10, interpreted as (10, 0) in the complex plane for simplicity
  let D := complex.of_real (-10)  -- AD symmetric to A, so (-10, 0)
  let B := 6 * complex.I  -- radius of inner circle as derived in the solution
  let chord_length := 16
  let outer_radius := complex.abs A
  let inner_radius := complex.abs B
  π * (outer_radius ^ 2 - inner_radius ^ 2)

theorem area_of_annulus_correct :
  area_between_circles = 64 * π :=
sorry

end area_of_annulus_correct_l146_146514


namespace allocation_methods_l146_146061

theorem allocation_methods (doctors nurses schools : ℕ) (h_doctors : doctors = 3) (h_nurses : nurses = 6) (h_schools : schools = 3) (h_each_school_doctor : ∀ s, s < schools → 1 ≤ doctors) (h_each_school_nurses : ∀ s, s < schools → 2 ≤ nurses / schools) : 
  ((doctors.fact) * ((nat.choose nurses 2) * (nat.choose (nurses - 2) 2))) = 540 :=
by
  sorry

end allocation_methods_l146_146061


namespace power_of_three_l146_146288

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146288


namespace even_function_decreasing_on_positive_domain_l146_146022

variable {f : ℝ → ℝ}

theorem even_function_decreasing_on_positive_domain 
  (h_even : ∀ x, f(x) = f(-x)) 
  (h_increasing : ∀ x y, x ≤ y → f(x) ≤ f(y)) 
  (h_domain : ∀ x, x ≥ -1 → ∀ y, y ≥ x → f(x) ≤ f(y)) :
  f(-2) > f(3) :=
by
  sorry

end even_function_decreasing_on_positive_domain_l146_146022


namespace problem_statement_l146_146431

variable (x y z : ℝ)

noncomputable def max_value (x y z : ℝ) : ℝ := 2 * x * y * Real.sqrt 5 + 9 * y * z

theorem problem_statement (h : x^2 + y^2 + z^2 = 1) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  max_value x y z ≤ Real.sqrt 86 :=
begin
  sorry
end

end problem_statement_l146_146431


namespace power_equality_l146_146322

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146322


namespace inequality_proof_l146_146870

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (b : Fin n → ℝ) (hb : ∀ i, 0 < b i) :
  (Finset.univ.sum (fun i => (a i)^2 / b i)) ≥
  ((Finset.univ.sum fun i => a i)^2 / (Finset.univ.sum fun i => b i)) :=
sorry

end inequality_proof_l146_146870


namespace log_a_b_integer_probability_l146_146537

theorem log_a_b_integer_probability :
  let S := { k : ℕ | 1 ≤ k ∧ k ≤ 25 },
      valid_pairs := (∏ x in S, (S.count (λ y, x ≠ y ∧ x ∣ y))) in
  (Σ' x y in valid_pairs, x ≠ y).card / (S.card.choose 2) = 31 / 300 := sorry

end log_a_b_integer_probability_l146_146537


namespace phase_shift_of_sine_function_l146_146194

def phase_shift (A B C : ℝ) : ℝ :=
  -C / B

theorem phase_shift_of_sine_function :
  phase_shift 3 4 (π / 4) = - (π / 16) :=
by
  sorry

end phase_shift_of_sine_function_l146_146194


namespace compute_five_fold_application_l146_146885

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then -x^2 else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -16 :=
by
  sorry

end compute_five_fold_application_l146_146885


namespace sixty_eighth_term_of_arithmetic_sequence_l146_146035

theorem sixty_eighth_term_of_arithmetic_sequence 
  (a₁ : ℕ) (a₂₁ : ℕ) (d : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₂₁ = 63)
  (h₃ : a₂₁ = a₁ + 20 * d) :
  (3 + 67 * d) = 204 :=
by {
  rw [h₁, h₂] at h₃,
  sorry
}

end sixty_eighth_term_of_arithmetic_sequence_l146_146035


namespace magic_square_sum_d_l146_146165

theorem magic_square_sum_d :
  (∃ (b c d e f g h : ℕ),
    d > 0 ∧ b > 0 ∧ c > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧
    60 * b * c = P ∧
    d * e * f = P ∧
    g * h * 3 = P ∧
    60 * e * 3 = P ∧
    b * e * h = P ∧
    c * f * 3 = P ∧
    60 * e * g = P ∧
    c * e * 3 = P ∧
    let d_values := [180, 90, 30, 18] in
    d_values.sum = 318) :=
sorry

end magic_square_sum_d_l146_146165


namespace power_of_three_l146_146371

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146371


namespace inequality_solution_l146_146470

noncomputable def inequality_expression (x : ℝ) : ℝ :=
  ((x - 1) * (x - 3)^2) / ((x - 2) * (x + 1) * (x - 4))

theorem inequality_solution :
  {x : ℝ | inequality_expression x > 0} = {x : ℝ | x ∈ (-∞, -1) ∪ (1, 2) ∪ (3, 4) ∪ (4, ∞)} :=
by
  sorry

end inequality_solution_l146_146470


namespace CE_eq_DE_l146_146153

theorem CE_eq_DE 
  (A B C D E : Type)
  [EquilateralTriangle A B C]
  (h1 : extend B C D)
  (h2 : extend B A E)
  (h3 : AE = BD)
  (h4 : connects C E)
  (h5 : connects D E) : CE = DE :=
sorry

end CE_eq_DE_l146_146153


namespace available_space_on_usb_l146_146903

theorem available_space_on_usb (total_capacity : ℕ) (used_percentage : ℝ) (total_capacity = 16) (used_percentage = 0.5) : 
  (total_capacity * (1 - used_percentage) = 8) := sorry

end available_space_on_usb_l146_146903


namespace paper_unfolded_holes_symmetry_l146_146551

-- Define the initial dimensions of the paper
def paper_width : ℕ := 8
def paper_height : ℕ := 4

-- Define the new dimensions after each fold
def fold1_width : ℕ := paper_width / 2
def fold1_height : ℕ := paper_height -- 4

def fold2_width : ℕ := fold1_width -- 4
def fold2_height : ℕ := fold1_height / 2 -- 2

-- Define the location of the hole in the folded paper
def hole_x : ℚ := fold2_width / 4 -- 1 unit from the left in the folded paper
def hole_y : ℚ := 3 * fold2_height / 4 -- 1.5 units from the bottom in the folded paper

-- Theorem stating that the paper will have holes symmetrically distributed across diagonal, horizontal, and vertical axes after unfolding
theorem paper_unfolded_holes_symmetry :
  (multiple_holes_symmetrically_distributed paper_width paper_height hole_x hole_y) :=
sorry
  
end paper_unfolded_holes_symmetry_l146_146551


namespace total_oranges_l146_146960

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end total_oranges_l146_146960


namespace orcs_carry_swords_l146_146028

theorem orcs_carry_swords:
  (let total_swords := 1200 in
   let squads := 10 in
   let orcs_per_squad := 8 in
   let total_orcs := squads * orcs_per_squad in
   let swords_per_orc := total_swords / total_orcs in
   swords_per_orc = 15) :=
by
  sorry

end orcs_carry_swords_l146_146028


namespace max_lessons_l146_146496

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end max_lessons_l146_146496


namespace first_term_value_l146_146512

noncomputable def find_first_term (a r : ℝ) := a / (1 - r) = 27 ∧ a^2 / (1 - r^2) = 108

theorem first_term_value :
  ∃ (a r : ℝ), find_first_term a r ∧ a = 216 / 31 :=
by
  sorry

end first_term_value_l146_146512


namespace power_of_3_l146_146358

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146358


namespace isosceles_trapezoid_inscribed_circle_ratio_l146_146152

noncomputable def ratio_perimeter_inscribed_circle (x : ℝ) : ℝ := 
  (50 * x) / (10 * Real.pi * x)

theorem isosceles_trapezoid_inscribed_circle_ratio 
  (x : ℝ)
  (h1 : x > 0)
  (r : ℝ) 
  (OK OP : ℝ) 
  (h2 : OK = 3 * x) 
  (h3 : OP = 5 * x) : 
  ratio_perimeter_inscribed_circle x = 5 / Real.pi :=
by
  sorry

end isosceles_trapezoid_inscribed_circle_ratio_l146_146152


namespace power_of_3_l146_146354

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146354


namespace exponent_power_identity_l146_146340

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146340


namespace surface_area_of_3D_figure_l146_146055

theorem surface_area_of_3D_figure :
  ∀ (cubes : Finset (Fin 9)) (edge_length : ℕ), 
  (∀ c ∈ cubes, edge_length = 1) → 
  (surface_area cubes edge_length = 32) := 
by 
  sorry

end surface_area_of_3D_figure_l146_146055


namespace mooncake_packaging_problem_l146_146062

theorem mooncake_packaging_problem
  (x y : ℕ)
  (L : ℕ := 9)
  (S : ℕ := 4)
  (M : ℕ := 35)
  (h1 : L = 9)
  (h2 : S = 4)
  (h3 : M = 35) :
  9 * x + 4 * y = 35 ∧ x + y = 5 := 
by
  sorry

end mooncake_packaging_problem_l146_146062


namespace minimum_value_of_g_l146_146169

def g (x : ℝ) : ℝ := (3 * x^2 + 6 * x + 19) / (8 * (1 + x))

theorem minimum_value_of_g : ∀ x : ℝ, x ≥ 0 → g x ≥ real.sqrt 3 := 
begin
  unfold g,
  sorry  -- placeholder for the proof
end

end minimum_value_of_g_l146_146169


namespace norm_more_than_twice_lisa_l146_146568

variable {L M N X : ℕ}

theorem norm_more_than_twice_lisa (h1 : N = 110) 
    (h2 : L + M = M + N - 60)
    (h3 : N = 2 * L + X) : X = 110 - 2 * L := 
by
  -- Given N = 110 from h1
  -- Substitute N in h3 to get 110 = 2 * L + X
  have hN : 110 = 2 * L + X := by rw [←h3, h1]
  -- Solving for X gives us the required proof
  rw [h1, h3] at hN
  exact sorry

end norm_more_than_twice_lisa_l146_146568


namespace OH_squared_l146_146428

theorem OH_squared (O H : Type) (a b c R : ℝ) (hR : R = 8) (h_squared_sum : a^2 + b^2 + c^2 = 50) :
  let OH^2 := 9 * R^2 - (a^2 + b^2 + c^2)
  OH^2 = 526 := 
by
  sorry

end OH_squared_l146_146428


namespace right_triangle_isosceles_l146_146133

-- Define the conditions for a right-angled triangle inscribed in a circle
variables (a b : ℝ)

-- Conditions provided in the problem
def right_triangle_inscribed (a b : ℝ) : Prop :=
  ∃ h : a ≠ 0 ∧ b ≠ 0, 2 * (a^2 + b^2) = (a + 2*b)^2 + b^2 ∧ 2 * (a^2 + b^2) = (2 * a + b)^2 + a^2

-- The theorem to prove based on the conditions
theorem right_triangle_isosceles (a b : ℝ) (h : right_triangle_inscribed a b) : a = b :=
by 
  sorry

end right_triangle_isosceles_l146_146133


namespace trajectory_equation_l146_146547

theorem trajectory_equation (x y a : ℝ) (h : x^2 + y^2 = a^2) :
  (x - y)^2 + 2*x*y = a^2 :=
by
  sorry

end trajectory_equation_l146_146547


namespace binom_9_5_eq_126_l146_146697

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146697


namespace angles_arith_prog_triangle_l146_146476

noncomputable def a : ℕ := 8
noncomputable def b : ℕ := 37
noncomputable def c : ℕ := 0

theorem angles_arith_prog_triangle (y : ℝ) (h1 : y = 8 ∨ y * y = 37) :
  a + b + c = 45 := by
  -- skipping the detailed proof steps
  sorry

end angles_arith_prog_triangle_l146_146476


namespace compute_combination_l146_146689

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146689


namespace power_of_three_l146_146362

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146362


namespace konigsberg_eulerian_paths_l146_146036

-- Definition of the problem and Eulerian path conditions
def degree (G : Type) (v : G) := 3 -- initial degree of each vertex

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def eulerian_path_condition (V : ℕ) (deg : ℕ → ℕ) : Prop :=
  let odd_vertices := {v | is_odd (deg v)} in
  odd_vertices.card = 0 ∨ odd_vertices.card = 2

noncomputable def ways_to_create_eulerian_path (initial_vertices : ℕ) (deg : ℕ → ℕ) : ℕ := 
  13023 -- The solution derived combinatorially 

-- Main theorem statement
theorem konigsberg_eulerian_paths :
  ways_to_create_eulerian_path 4 degree = 13023 := 
  by sorry

end konigsberg_eulerian_paths_l146_146036


namespace power_calculation_l146_146310

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146310


namespace sum_of_series_l146_146011

theorem sum_of_series : 
  let individual := 7 in
  let cats := 7 * individual in
  let mice := 7 * cats in
  let barley := 7 * mice in
  let grain := 7 * barley in
  individual + cats + mice + barley + grain = 19607 :=
by
  let individual := 7
  let cats := 7 * individual
  let mice := 7 * cats
  let barley := 7 * mice
  let grain := 7 * barley
  have sum_eq : individual + cats + mice + barley + grain = 19607 := sorry
  exact sum_eq

end sum_of_series_l146_146011


namespace valid_permutations_count_l146_146273

/-- 
Given five elements consisting of the numbers 1, 2, 3, and the symbols "+" and "-", 
we want to count the number of permutations such that no two numbers are adjacent.
-/
def count_valid_permutations : Nat := 
  let number_permutations := Nat.factorial 3 -- 3! permutations of 1, 2, 3
  let symbol_insertions := Nat.factorial 2  -- 2! permutations of "+" and "-"
  number_permutations * symbol_insertions

theorem valid_permutations_count : count_valid_permutations = 12 := by
  sorry

end valid_permutations_count_l146_146273


namespace train_speed_l146_146140

/-- Proof that calculates the speed of a train given the times to pass a man and a platform,
and the length of the platform, and shows it equals 54.00432 km/hr. -/
theorem train_speed (L V : ℝ) 
  (platform_length : ℝ := 360.0288)
  (time_to_pass_man : ℝ := 20)
  (time_to_pass_platform : ℝ := 44)
  (equation1 : L = V * time_to_pass_man)
  (equation2 : L + platform_length = V * time_to_pass_platform) :
  V = 15.0012 → V * 3.6 = 54.00432 :=
by sorry

end train_speed_l146_146140


namespace nancy_pictures_left_l146_146564

-- Given conditions stated in the problem
def picturesZoo : Nat := 49
def picturesMuseum : Nat := 8
def picturesDeleted : Nat := 38

-- The statement of the problem, proving Nancy still has 19 pictures after deletions
theorem nancy_pictures_left : (picturesZoo + picturesMuseum) - picturesDeleted = 19 := by
  sorry

end nancy_pictures_left_l146_146564


namespace smallest_union_cardinality_l146_146010

variables (A B : Set ℕ)

theorem smallest_union_cardinality (hA : A.card = 30) (hB : B.card = 20) (h_inter : 10 ≤ (A ∩ B).card) :
  (A ∪ B).card = 40 :=
by
  sorry

end smallest_union_cardinality_l146_146010


namespace product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l146_146081

theorem product_two_smallest_one_digit_primes_and_largest_three_digit_prime :
  2 * 3 * 997 = 5982 :=
by
  sorry

end product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l146_146081


namespace filling_material_heavier_than_sand_l146_146529

noncomputable def percentage_increase (full_sandbag_weight : ℝ) (partial_fill_percent : ℝ) (full_material_weight : ℝ) : ℝ :=
  let sand_weight := (partial_fill_percent / 100) * full_sandbag_weight
  let material_weight := full_material_weight
  let weight_increase := material_weight - sand_weight
  (weight_increase / sand_weight) * 100

theorem filling_material_heavier_than_sand :
  let full_sandbag_weight := 250
  let partial_fill_percent := 80
  let full_material_weight := 280
  percentage_increase full_sandbag_weight partial_fill_percent full_material_weight = 40 :=
by
  sorry

end filling_material_heavier_than_sand_l146_146529


namespace range_of_a_l146_146790

theorem range_of_a (a x y : ℝ)
  (h1 : x + y = 3 * a + 4)
  (h2 : x - y = 7 * a - 4)
  (h3 : 3 * x - 2 * y < 11) : a < 1 :=
sorry

end range_of_a_l146_146790


namespace doubled_cost_percent_l146_146926

-- Definitions
variable (t b : ℝ)
def cost (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percent :
  cost t (2 * b) = 16 * cost t b :=
by
  -- To be proved
  sorry

end doubled_cost_percent_l146_146926


namespace part_a_part_b_l146_146932

theorem part_a (λ α x : ℝ) :
  (x^5 + 5 * λ * x^4 - x^3 + (λ * α - 4) * x^2 - (8 * λ + 3) * x + λ * α - 2 = 0) →
  (x = 2) → 
  α = -64 / 5 :=
by
  sorry

theorem part_b (λ α x : ℝ) :
  (x^5 + 5 * λ * x^4 - x^3 + (λ * α - 4) * x^2 - (8 * λ + 3) * x + λ * α - 2 = 0) →
  ((x = (-(1 / 2)) + (sqrt 3 / 2) * I) ∨ (x = (-(1 / 2)) - (sqrt 3 / 2) * I)) → 
  α = -3 :=
by
  sorry

end part_a_part_b_l146_146932


namespace sum_abc_l146_146934

variables {a b c : ℤ}

theorem sum_abc : 
  (∃ a b : ℤ, (x^2 + 7x - 18 = (x + a) * (x + b)) ∧ a * b = -18 ∧ a + b = 7) ∧ 
  (∃ b c : ℤ, (x^2 + 11x + 24 = (x + b) * (x + c)) ∧ b * c = 24 ∧ b + c = 11) → 
  a + b + c = 20 :=
sorry

end sum_abc_l146_146934


namespace greatest_b_for_no_real_roots_l146_146726

theorem greatest_b_for_no_real_roots :
  ∀ (b : ℤ), (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) ↔ b ≤ 6 := sorry

end greatest_b_for_no_real_roots_l146_146726


namespace translated_graph_equation_l146_146068

theorem translated_graph_equation :
  ∀ (x y : ℝ), 
    y = 2 * Real.cos ((x / 3) + (Real.pi / 6)) →
    y - 2 = 2 * Real.cos (((x + Real.pi / 4) / 3) + (Real.pi / 6) - Real.pi / 12) →
    y = 2 * Real.cos ((x / 3) + (Real.pi / 4)) - 2 :=
begin
  sorry
end

end translated_graph_equation_l146_146068


namespace minimum_value_vector_diff_norm_l146_146226

open Real
open ComplexConjugate

variables {a b c : ℝ × ℝ}

def vector_norm (v : ℝ × ℝ) : ℝ := sqrt(v.1 * v.1 + v.2 * v.2)
def angle_between (v w : ℝ × ℝ) : ℝ :=
  let dot_product   := v.1 * w.1 + v.2 * w.2
  let norms_product := vector_norm v * vector_norm w
  let cos_theta     := dot_product / norms_product
  acos cos_theta

theorem minimum_value_vector_diff_norm
  (h₀ : vector_norm a = 2)
  (h₁ : vector_norm (b - c) = 1)
  (h₂ : angle_between a b = π / 3) :
  ∃ x, x = vector_norm (a - c) ∧ x ≥ sqrt(3) - 1 := sorry

end minimum_value_vector_diff_norm_l146_146226


namespace solve_A_solve_area_l146_146230

noncomputable def angle_A (A : ℝ) : Prop :=
  2 * (Real.cos (A / 2))^2 + Real.cos A = 0

noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 → b + c = 4 → A = 2 * Real.pi / 3 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3

theorem solve_A (A : ℝ) : angle_A A → A = 2 * Real.pi / 3 :=
sorry

theorem solve_area (a b c A S : ℝ) : 
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  A = 2 * Real.pi / 3 →
  area_triangle a b c A →
  S = Real.sqrt 3 :=
sorry

end solve_A_solve_area_l146_146230


namespace number_of_solutions_is_nine_l146_146710

noncomputable def count_solutions : ℕ :=
  {z : ℂ | abs z < 15 ∧ exp(2 * z) = (z - 2) / (z + 2)}.count sorry

theorem number_of_solutions_is_nine :
  count_solutions = 9 :=
sorry

end number_of_solutions_is_nine_l146_146710


namespace exponent_power_identity_l146_146339

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146339


namespace exponent_power_identity_l146_146332

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146332


namespace find_value_of_s_l146_146797

variable {r s : ℝ}

theorem find_value_of_s (hr : r > 1) (hs : s > 1) (h1 : 1/r + 1/s = 1) (h2 : r * s = 9) :
  s = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_value_of_s_l146_146797


namespace cos_double_angle_FQG_l146_146446

theorem cos_double_angle_FQG (E F G H Q : Point) (α β γ δ: ℝ) 
    (tan_eqg : tan α = 2) (tan_fqh : tan β = 5/3) (d_eq : δ = 180 - 2 * γ) : 
    cos (2 * γ) = -36 / 85 := by
  sorry

end cos_double_angle_FQG_l146_146446


namespace power_calculation_l146_146308

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146308


namespace transformed_data_average_variance_l146_146753

theorem transformed_data_average_variance (n : ℕ) (data : Fin n → ℝ)
  (h_avg : (∑ i, data i) / n = 3)
  (h_var : (∑ i, (data i - 3)^2) / n = 3) :
  let transformed := λ i : Fin n, 3 * (data i) - 2 in
  ((∑ i, transformed i) / n = 7) ∧ ((∑ i, ((transformed i - 7)^2)) / n = 27) := by
  sorry

end transformed_data_average_variance_l146_146753


namespace binom_9_5_eq_126_l146_146699

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146699


namespace binom_9_5_l146_146675

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146675


namespace difference_p_q_l146_146376

theorem difference_p_q (p q : ℝ) (hp : 3 / p = 6) (hq : 3 / q = 18) : p - q = 1 / 3 :=
by
  have hp' : p = 1 / 2 := by
    calc
      p = 3 / 6 : by field_simp [hp]
      ... = 1 / 2 : by norm_num
  have hq' : q = 1 / 6 := by
    calc
      q = 3 / 18 : by field_simp [hq]
      ... = 1 / 6 : by norm_num
  rw [hp', hq']
  calc
    1 / 2 - 1 / 6 = (3 / 6) - (1 / 6) : by field_simp
    ... = (3 - 1) / 6 : by ring
    ... = 2 / 6 : by norm_num
    ... = 1 / 3 : by norm_num

end difference_p_q_l146_146376


namespace sum_of_perimeters_l146_146051

theorem sum_of_perimeters (x y : ℝ) 
  (h1 : x^2 + y^2 = 113) 
  (h2 : x^2 - y^2 = 47) : 
  3 * (4 * x) + 4 * y = 48 * real.sqrt 5 + 4 * real.sqrt 33 := 
sorry

end sum_of_perimeters_l146_146051


namespace wilson_theorem_l146_146881

theorem wilson_theorem (p : ℕ) (hp : p > 1) : 
  prime p ↔ fact (p - 1) ≡ -1 [MOD p] := 
sorry

end wilson_theorem_l146_146881


namespace number_of_cookies_paco_ate_l146_146445

-- Defining the initial conditions
def initial_cookies : ℕ := 28
def cookies_left : ℕ := 7

-- The theorem to prove the number of cookies Paco ate
theorem number_of_cookies_paco_ate : initial_cookies - cookies_left = 21 :=
by
  calc
  initial_cookies - cookies_left = 28 - 7 : by sorry
  ...  = 21 : by sorry

end number_of_cookies_paco_ate_l146_146445


namespace analytical_expression_minimum_value_range_of_function_l146_146260

noncomputable def sine_function (A : ℝ) (ω : ℝ) (φ : ℝ) : ℝ → ℝ :=
  λ x => A * Real.sin (ω * x + φ)

-- Conditions
def A := 5
def omega := 2
def phi := -Real.pi / 6
def function := sine_function A omega phi

theorem analytical_expression :
  function = λ x => 5 * Real.sin (2 * x - (Real.pi / 6)) :=
by
  sorry

theorem minimum_value (x : ℝ) :
  (∃ k : ℤ, x = k * Real.pi - Real.pi / 6) → function x = -5 :=
by
  sorry

theorem range_of_function (x : ℝ) :
  0 ≤ x → x ≤ Real.pi / 2 → -5 / 2 ≤ function x ∧ function x ≤ 5 :=
by
  sorry

end analytical_expression_minimum_value_range_of_function_l146_146260


namespace colten_chickens_l146_146004

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l146_146004


namespace quadratic_zeros_l146_146818

theorem quadratic_zeros (a b : ℝ) (h1 : (4 - 2 * a + b = 0)) (h2 : (9 + 3 * a + b = 0)) : a + b = -7 := 
by
  sorry

end quadratic_zeros_l146_146818


namespace prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l146_146964

noncomputable def probability_A_exactly_2_hits :=
  let p_A := 1/2
  let trials := 3
  (trials.choose 2) * (p_A ^ 2) * ((1 - p_A) ^ (trials - 2))

noncomputable def probability_B_at_least_2_hits :=
  let p_B := 2/3
  let trials := 3
  (trials.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ (trials - 2)) + (trials.choose 3) * (p_B ^ 3)

noncomputable def probability_B_exactly_2_more_hits_A :=
  let p_A := 1/2
  let p_B := 2/3
  let trials := 3
  let B_2_A_0 := (trials.choose 2) * (p_B ^ 2) * (1 - p_B) * (trials.choose 0) * (p_A ^ 0) * ((1 - p_A) ^ trials)
  let B_3_A_1 := (trials.choose 3) * (p_B ^ 3) * (trials.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (trials - 1))
  B_2_A_0 + B_3_A_1

theorem prove_A_exactly_2_hits : probability_A_exactly_2_hits = 3/8 := sorry
theorem prove_B_at_least_2_hits : probability_B_at_least_2_hits = 20/27 := sorry
theorem prove_B_exactly_2_more_hits_A : probability_B_exactly_2_more_hits_A = 1/6 := sorry

end prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l146_146964


namespace power_of_three_l146_146279

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146279


namespace binom_9_5_eq_126_l146_146662

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146662


namespace min_subset_contains_pythagorean_triple_l146_146435

open Finset

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem min_subset_contains_pythagorean_triple :
  ∀ (S : Finset ℕ) (n : ℕ), 
  S = (range 51).filter (λ x, x ≠ 0) → 
  n = 42 →
  (∀ T : Finset ℕ, T ⊆ S → card T = n → 
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_pythagorean_triple a b c) :=
by
  intros S n hS hn T hsubset hcard
  sorry

end min_subset_contains_pythagorean_triple_l146_146435


namespace option_A_option_B_option_C_option_D_all_options_are_correct_l146_146148

variable {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]

-- Condition Definitions
variables (a b : V)
variables (A B C : V)

-- Lean 4 statements asserting the correctness of each condition
theorem option_A : a + b = b + a :=
by sorry

theorem option_B : -(-a) = a :=
by sorry

noncomputable def vector_AB : V := A - B
noncomputable def vector_BC : V := B - C
noncomputable def vector_CA : V := C - A

theorem option_C : vector_AB + vector_BC + vector_CA = 0 :=
by sorry

theorem option_D : a + (-a) = 0 :=
by sorry

-- Summarizing that all options are correct
theorem all_options_are_correct : (a + b = b + a) ∧ (-(-a) = a) ∧ (vector_AB + vector_BC + vector_CA = 0) ∧ (a + (-a) = 0) :=
by sorry

end option_A_option_B_option_C_option_D_all_options_are_correct_l146_146148


namespace correct_log_conclusions_l146_146735

variables {x₁ x₂ : ℝ} (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (h_diff : x₁ ≠ x₂)
noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem correct_log_conclusions :
  ¬ (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ¬ ((f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by {
  sorry
}

end correct_log_conclusions_l146_146735


namespace functional_equation_solution_l146_146724

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y z : ℝ, f(x * y * z) = f(x) * f(y) * f(z) - 6 * x * y * z) ↔ f = (λ x, 2 * x) :=
by
  intro f
  split
  sorry

end functional_equation_solution_l146_146724


namespace max_n_l146_146840

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def b (n : ℕ) : ℕ := 2 ^ a n

theorem max_n (n : ℕ) (h1 : a 2 = 2) (h2 : ∀ n, b n = 2 ^ a n)
  (h3 : b 4 = 4 * b 2) : n ≤ 9 :=
by 
  sorry

end max_n_l146_146840


namespace log_a_b_integer_probability_l146_146534

theorem log_a_b_integer_probability :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 25}
  let pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ a ≠ b}
  let valid_pairs := {pair | pair ∈ pairs ∧ ∃ k : ℕ, b = k * a ∧ k ≥ 2}
  let total_pairs := finset.card pairs
  let valid_pairs_count := finset.card valid_pairs
  (valid_pairs_count / total_pairs : ℝ) = 31 / 300 :=
sorry

end log_a_b_integer_probability_l146_146534


namespace max_valid_n_eq_3210_l146_146199

-- Define the digit sum function S
def digit_sum (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- The condition S(3n) = 3S(n) and all digits of n are distinct
def valid_n (n : ℕ) : Prop :=
  digit_sum (3 * n) = 3 * digit_sum n ∧ (Nat.digits 10 n).Nodup

-- Prove that the maximum value of such n is 3210
theorem max_valid_n_eq_3210 : ∃ n : ℕ, valid_n n ∧ n = 3210 :=
by
  existsi 3210
  sorry

end max_valid_n_eq_3210_l146_146199


namespace MN_min_length_l146_146787

theorem MN_min_length (a : ℝ) (h : a > 0) : 
  @Real.abs (a^2 - Real.log a) = @Real.abs ((Real.sqrt 2) / 2)^2 - Real.log ((Real.sqrt 2) / 2) := 
begin
  -- Pending proof
  sorry
end

end MN_min_length_l146_146787


namespace largest_multiple_of_15_under_500_l146_146996

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146996


namespace binom_9_5_eq_126_l146_146679

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146679


namespace find_percentage_l146_146375

def certain_percentage (x p : ℕ) : Prop := (0.20 * x : ℝ) = (p / 100.0) * 1500 - 15

theorem find_percentage {x : ℕ} (h : x = 1050) : certain_percentage x 15 :=
by {
  simp [certain_percentage, h],
  sorry
}

end find_percentage_l146_146375


namespace inverse_function_l146_146544

def f (x : ℝ) : ℝ := 4 - 5 * x

def g (y : ℝ) : ℝ := (4 - y) / 5

theorem inverse_function : ∀ x, f (g x) = x := by
  intro x
  sorry

end inverse_function_l146_146544


namespace total_work_duration_l146_146092

-- Definitions from the problem conditions
def work_rate_p : ℝ := 1 / 80
def work_rate_q : ℝ := 1 / 48
def initial_days_p_worked_alone : ℝ := 8

-- Theorem statement to prove the total work duration
theorem total_work_duration : 
  let W := 1 in
  let work_done_by_p_alone := initial_days_p_worked_alone * work_rate_p in
  let remaining_work := W - work_done_by_p_alone in
  let combined_work_rate := work_rate_p + work_rate_q in
  initial_days_p_worked_alone + (remaining_work / combined_work_rate) = 35 :=
by
  let W := 1
  let work_done_by_p_alone := initial_days_p_worked_alone * work_rate_p
  let remaining_work := W - work_done_by_p_alone
  let combined_work_rate := work_rate_p + work_rate_q
  have total_time := initial_days_p_worked_alone + (remaining_work / combined_work_rate)
  show total_time = 35
  sorry

end total_work_duration_l146_146092


namespace common_tangent_exists_l146_146942

-- Definitions and theorems needed for the problem
variables {ABC : Triangle}

-- Define the orthocenter H
def orthoCenter (ABC : Triangle) : Point := sorry

-- Define the incenter and the incircle
def incircleCenter (ABC : Triangle) : Point := sorry
def incircle (ABC : Triangle) : Circle := sorry

-- Condition: Orthocenter lies on the incircle
axiom ortho_center_on_incircle (ABC : Triangle) : orthoCenter ABC ∈ incircle ABC

-- Define three circles with centers A, B, C passing through H
def circleWithCenter (P : Point) (H : Point) : Circle := sorry

-- Define common tangent to circles
def commonTangent (C1 C2 C3 : Circle) : Line := sorry

-- The theorem to be proven
theorem common_tangent_exists (ABC : Triangle) (H : Point) (hH : H = orthoCenter ABC)
  (hInCenter : orthoCenter ABC ∈ incircle ABC) :
  ∃ L : Line, ∀ (P ∈ {ABC.A, ABC.B, ABC.C}), isTangent L (circleWithCenter P H) :=
sorry

end common_tangent_exists_l146_146942


namespace binom_9_5_eq_126_l146_146696

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146696


namespace projection_correct_l146_146730

noncomputable def projection_onto_plane (v : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let dot := λ u v, u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  p = (v.1 - (dot v n / dot n n) * n.1, v.2 - (dot v n / dot n n) * n.2, v.3 - (dot v n / dot n n) * n.3)

theorem projection_correct :
  projection_onto_plane (2, 3, 1) (2, 1, -3) (10/7, 19/7, 13/7) :=
by
  have v := (2, 3, 1)
  have n := (2, 1, -3)
  have p := (10/7, 19/7, 13/7)
  unfold projection_onto_plane
  sorry

end projection_correct_l146_146730


namespace meaningful_fraction_l146_146505

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end meaningful_fraction_l146_146505


namespace sum_of_inscribed_angles_divided_circle_l146_146483

theorem sum_of_inscribed_angles_divided_circle 
  (O : Type) [DiscreteField O] 
  (C : O) (x y : ℝ)
  (div_12_arcs : ℕ)
  (arcs_eq_degrees_30 : div_12_arcs = 12)
  (x_spans_2_arcs : x = 2 * 30 / 2)
  (y_spans_4_arcs : y = 4 * 30 / 2) : 
  x + y = 90 :=
by
  have div_12_pos : 0 < div_12_arcs := by linarith
  have arcs_deg : 360 / div_12_arcs = 30 := by rw arcs_eq_degrees_30; norm_num
  rw [x_spans_2_arcs, y_spans_4_arcs];
  norm_num;
  sorry

end sum_of_inscribed_angles_divided_circle_l146_146483


namespace max_equal_segments_l146_146974

theorem max_equal_segments (n : ℕ) (a : Fin n → α) (h : ∀ i : Fin (n - 1), a i ≠ a ⟨i + 1, Nat.lt_of_lt_pred i.prop (Nat.pred_lt_pred 0 (ne_of_gt (Fin.is_lt i)))⟩) :
  ∀ k : Nat, n = 2 * k ∨ n = 2 * k + 1 → 
  ∃ m : ℕ, m = if n % 2 = 0 then n / 2 else n / 2 + 1 :=
by
  sorry

end max_equal_segments_l146_146974


namespace max_value_of_function_l146_146498

theorem max_value_of_function : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y == (2*x^2 - 2*x + 3) / (x^2 - x + 1)) → y ≤ 10/3) ∧
  (∃ x : ℝ, (2*x^2 - 2*x + 3) / (x^2 - x + 1) = 10/3) := 
sorry

end max_value_of_function_l146_146498


namespace sum_faces_of_cube_l146_146917

-- Conditions in Lean 4
variables (a b c d e f : ℕ)

-- Sum of vertex labels
def vertex_sum := a * b * c + a * e * c + a * b * f + a * e * f +
                  d * b * c + d * e * c + d * b * f + d * e * f

-- Theorem statement
theorem sum_faces_of_cube (h : vertex_sum a b c d e f = 1001) :
  (a + d) + (b + e) + (c + f) = 31 :=
sorry

end sum_faces_of_cube_l146_146917


namespace sqrt_combination_l146_146374

theorem sqrt_combination (t : ℝ) (h : 2 * real.sqrt 3 = real.sqrt (2 * t - 1)) : t = 2 :=
by
  sorry

end sqrt_combination_l146_146374


namespace increase_by_twenty_percent_l146_146567

theorem increase_by_twenty_percent (original : ℕ) (percentage : ℕ) (final : ℕ) 
  (h1 : original = 240) 
  (h2 : percentage = 20) 
  (h3 : final = original + (original * percentage) / 100) : 
  final = 288 := 
begin
  -- Proof skipped
  sorry
end

end increase_by_twenty_percent_l146_146567


namespace range_of_m_if_not_p_and_q_l146_146761

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_if_not_p_and_q_l146_146761


namespace afforested_area_fourth_year_l146_146575

def initial_area : ℕ := 10000
def growth_rate : ℝ := 0.2
def year_1_area : ℕ := initial_area
def year_2_area : ℕ := (initial_area:ℝ * (1 + growth_rate)).toInt
def year_3_area : ℕ := (year_2_area:ℝ * (1 + growth_rate)).toInt
def year_4_area : ℕ := (year_3_area:ℝ * (1 + growth_rate)).toInt

theorem afforested_area_fourth_year : year_4_area = 17280 := by
  sorry

end afforested_area_fourth_year_l146_146575


namespace number_of_girls_l146_146390

-- Definitions and conditions
variables (g b : ℕ) (k : ℕ)
hypothesis h1 : g = 4 * k
hypothesis h2 : b = 3 * k
hypothesis h3 : g + b + 1 = 43

-- Proof goal
theorem number_of_girls : g = 24 :=
by
  -- Placeholder for the proof
  sorry

end number_of_girls_l146_146390


namespace sum_of_possible_b_coefficients_l146_146128

theorem sum_of_possible_b_coefficients :
  ∀ (b : ℤ), (∃ p q : ℕ, p ≠ q ∧ p * q = 24 ∧ b = - (p + q)) →
  (Σ' (b : ℤ), ∃ p q : ℕ, p ≠ q ∧ p * q = 24 ∧ b = - (p + q)) = -60 :=
by
  sorry

end sum_of_possible_b_coefficients_l146_146128


namespace geometric_sequence_product_l146_146846

-- Defining a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given data
def a := fun n => (4 : ℝ) * (2 : ℝ)^(n-4)

-- Main proof problem
theorem geometric_sequence_product (a : ℕ → ℝ) (h : is_geometric_sequence a) (h₁ : a 4 = 4) :
  a 2 * a 6 = 16 :=
by
  sorry

end geometric_sequence_product_l146_146846


namespace power_of_3_l146_146347

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146347


namespace common_elements_count_287_l146_146422

def set_U (n : ℕ) : Set ℕ := {k | ∃ i, ↑k = 5 * i ∧ i ≤ n}
def set_V (n : ℕ) : Set ℕ := {k | ∃ i, ↑k = 7 * i ∧ i ≤ n}
def lcm (a b : ℕ) := (a * b) / Nat.gcd a b

theorem common_elements_count_287 :
  let U := (set_U 2010)
  let V := (set_V 2010)
  (U ∩ V).to_finset.card = 287 := by
  sorry

end common_elements_count_287_l146_146422


namespace a_n_general_formula_b_n_general_formula_sum_first_n_log_b_sum_first_n_c_l146_146224

noncomputable def a_seq (n : ℕ) : ℤ := 2 * n - 3

noncomputable def b_seq : ℕ → ℤ
| 1 := 1
| (n+1) := 2^(a_seq n + 3) * b_seq n

def c_seq (n : ℕ) : ℤ := a_seq n * (real.sqrt 2) ^ (a_seq n + 1 : ℤ)

theorem a_n_general_formula : ∀ n : ℕ, a_seq n = 2 * n - 3 := sorry

theorem b_n_general_formula : ∀ n : ℕ, b_seq n = 2^(n*(n-1)) := sorry

theorem sum_first_n_log_b : ∀ n : ℕ, (∑ i in finset.range n,  1 / real.log ((b_seq (i+1)).to_real) / real.log 2) = n / (n + 1 : ℝ) := sorry

theorem sum_first_n_c : ∀ n : ℕ, (∑ i in finset.range n, c_seq (i + 1)) = (2 * n - 5) * 2^n + 5 := sorry

end a_n_general_formula_b_n_general_formula_sum_first_n_log_b_sum_first_n_c_l146_146224


namespace arithmetic_sequence_sum_l146_146636

open Nat

theorem arithmetic_sequence_sum :
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  3 * S = 3774 := 
by
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end arithmetic_sequence_sum_l146_146636


namespace max_floor_l146_146201

theorem max_floor (x : ℝ) (h : ⌊(x + 4) / 10⌋ = 5) : ⌊(6 * x) / 5⌋ = 67 :=
  sorry

end max_floor_l146_146201


namespace power_of_three_l146_146285

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146285


namespace find_x_squared_add_y_squared_l146_146723

noncomputable def x_squared_add_y_squared (x y : ℝ) : ℝ :=
  x^2 + y^2

theorem find_x_squared_add_y_squared (x y : ℝ) 
  (h1 : x + y = 48)
  (h2 : x * y = 168) :
  x_squared_add_y_squared x y = 1968 :=
by
  sorry

end find_x_squared_add_y_squared_l146_146723


namespace total_pieces_after_20th_placement_l146_146550

/-- Xiao Wang places some equilateral triangle paper pieces on the table. The first time he places 1 piece; the second time he places three more pieces around the first triangle; the third time he places more pieces around the shape formed in the second placement, and so on. The requirement is: each piece placed in each subsequent placement must share at least one edge with a piece placed in the previous placement, and apart from sharing edges, there should be no other overlaps. We need to prove that after the 20th placement, the total number of equilateral triangle pieces used is 571. -/
theorem total_pieces_after_20th_placement : 
  let total_pieces := 1 + 3 * (Finset.range 19).sum (λ n, n + 1)
  in total_pieces = 571 :=
by
  let total_pieces := 1 + 3 * (Finset.range 19).sum (λ n, n + 1)
  have h : total_pieces = 571 := sorry
  exact h

end total_pieces_after_20th_placement_l146_146550


namespace candy_problem_l146_146634

def total_candy_eaten (morning: ℕ) (afternoon: ℕ) (evening: ℕ) : ℕ :=
  morning + afternoon + evening

theorem candy_problem : 
  ∀ (morning afternoon evening : ℕ), 
  morning = 26 → 
  afternoon = 3 * morning → 
  evening = afternoon / 2 → 
  total_candy_eaten morning afternoon evening = 143 := 
by 
  intros morning afternoon evening hmorning haft hng
  rw [hmorning, haft, hng]
  simp
  sorry

end candy_problem_l146_146634


namespace exponent_power_identity_l146_146334

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146334


namespace compute_combination_l146_146686

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146686


namespace ned_games_l146_146898

theorem ned_games (F: ℕ) (bought_from_friend garage_sale non_working good total_games: ℕ) 
  (h₁: bought_from_friend = F)
  (h₂: garage_sale = 27)
  (h₃: non_working = 74)
  (h₄: good = 3)
  (h₅: total_games = non_working + good)
  (h₆: total_games = bought_from_friend + garage_sale) :
  F = 50 :=
by
  sorry

end ned_games_l146_146898


namespace projection_of_a_onto_b_l146_146890

-- Define vectors a and b in a real vector space
variables (a b : ℝ × ℝ)
-- Define magnitudes of the vectors
def mag_a := 2
def mag_b := 2
-- Define the angle between vectors a and b
def angle := real.pi / 3 -- 60 degrees in radians

-- Definition to calculate dot product given the angle and magnitudes
def dot_product (a b : ℝ × ℝ) (mag_a mag_b : ℝ) (angle : ℝ) :=
  mag_a * mag_b * real.cos angle

-- Given conditions
axiom mag_a_eq_2 : ∥a∥ = mag_a
axiom mag_b_eq_2 : ∥b∥ = mag_b
axiom angle_eq_60 : ∡ a b = angle

-- The specific theorem to prove
theorem projection_of_a_onto_b : (dot_product a b 2 (real.pi / 3) / (∥b∥ ^ 2)) • b = (1 / 2) • b :=
by
  sorry  -- Proof will be written here

end projection_of_a_onto_b_l146_146890


namespace power_of_3_l146_146352

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146352


namespace age_of_other_man_replaced_l146_146481

variable (men_avg_age new_avg_age other_man other_woman combined_age swap_age_diff : ℝ)

-- Given conditions as definitions in Lean 4
def condition_1 (seven_men_avg : ℝ) : Prop :=
  new_avg_age = seven_men_avg + 3

def condition_2 : Prop :=
  other_man = 18

def condition_3 : Prop :=
  combined_age = 30.5 + 30.5

-- Proof Problem: Given conditions, prove the other man's age
theorem age_of_other_man_replaced
  (seven_men_avg : ℝ)
  (h1: condition_1 seven_men_avg)
  (h2: condition_2)
  (h3: condition_3)
  : swap_age_diff = other_man + 18 :=
begin
  -- We need to prove that the age difference due to swapping the men with women satisfies the equation,
  -- which leads to determining the unknown man's age
  have : 7 * seven_men_avg + combined_age = 7 * (seven_men_avg + 3) + swap_age_diff,
  { sorry },
  -- Simplifying the equation leads us directly to the age of the replaced man.
  rw [h1, h2, h3] at this,
  sorry
end

end age_of_other_man_replaced_l146_146481


namespace part_a_n_eq_2_part_a_n_eq_3_part_a_n_eq_4_part_b_upper_bound_l146_146200

-- Indicates that we are working with non-computational definitions
noncomputable theory

-- Define the conditions as Lean definitions
def isConditionValid (n : ℕ) (z : ℂ) : Prop :=
  (abs z = 1) ∧ ((z^n + conj z^n) = 2 * (re z)^n)

-- Define the set S(n)
def S (n : ℕ) : set ℂ := {z : ℂ | isConditionValid n z}

-- Prove parts (a) and (b)
theorem part_a_n_eq_2 : S 2 = {1, -1} := 
sorry

theorem part_a_n_eq_3 : S 3 = {1, -1} := 
sorry

theorem part_a_n_eq_4 : S 4 = {1, -1, (1 / real.sqrt 7) + I * (real.sqrt (6 / 7)), 
                                      (1 / real.sqrt 7) - I * (real.sqrt (6 / 7)), 
                                      -(1 / real.sqrt 7) + I * (real.sqrt (6 / 7)), 
                                      -(1 / real.sqrt 7) - I * (real.sqrt (6 / 7))} := 
sorry

theorem part_b_upper_bound (n : ℕ) (h : n > 5) : ∀s ∈ S n, s ≤ n :=
sorry

end part_a_n_eq_2_part_a_n_eq_3_part_a_n_eq_4_part_b_upper_bound_l146_146200


namespace pencils_left_l146_146857

/-- Junyoung has 11 dozen pencils, gives 4 dozen to Taesoo and 9 pencils to Jongsoo. 
    This proof confirms the number of pencils left with Junyoung is 75. --/
theorem pencils_left (initial_dozen_pencils : ℕ) (junyoung_gives_tae : ℕ) (junyoung_gives_jong : ℕ)
  (h1 : initial_dozen_pencils = 11)
  (h2 : junyoung_gives_tae = 4)
  (h3 : junyoung_gives_jong = 9) :
  (initial_dozen_pencils * 12 - (junyoung_gives_tae * 12 + junyoung_gives_jong) = 75) :=
by {
  -- Based on the conditions, we need to show the following result is true
  have total_pencils : initial_dozen_pencils * 12 = 132 := by sorry,
  have total_given_away : (junyoung_gives_tae * 12 + junyoung_gives_jong) = 57 := by sorry,
  show (132 - 57) = 75, from sorry,
}

end pencils_left_l146_146857


namespace parallel_lines_proof_l146_146095

noncomputable def is_circumscribed (ω : Circle) (ABC : Triangle) : Prop :=
  ∀(A B C : Point), Circle.contains ω A ∧ Circle.contains ω B ∧ Circle.contains ω C

noncomputable def midpoint (M : Point) (C K : Segment) : Prop :=
  ∃ (M : Point), Segment.contains C M ∧ Segment.contains K M ∧
    Segment.length (Segment.mk C M) = Segment.length (Segment.mk M K)

noncomputable def tangent_intersects (ω : Circle) (C : Point) (K : Point) : Prop :=
  Line.contains (Tangent ω C) K ∧ 
  ∀ (A B : Point), Line.contains (Line.mk A B) K ∧ Circle.contains ω C

noncomputable def second_intersection (L M : Point) (ω : Circle) : Prop :=
  ∃ L, Line.contains (Line.mk ω.center L) M ∧ Circle.contains ω L ∧ L ≠ M

noncomputable def parallel_lines (L1 L2 : Line) : Prop :=
  ∀ (P Q : Point), Line.contains L1 P → Line.contains L2 Q → ¬ are_intersecting L1 L2

theorem parallel_lines_proof  (A B C K M L N : Point) (ω : Circle) (ABC : Triangle) :
  is_circumscribed ω ABC →
  tangent_intersects ω C K →
  midpoint M (Segment.mk C K) →
  second_intersection L M ω →
  second_intersection N K ω →
  parallel_lines (Line.mk A N) (Line.mk C K) :=
sorry

end parallel_lines_proof_l146_146095


namespace equal_sums_arithmetic_sequences_l146_146425

-- Define the arithmetic sequences and their sums
def s₁ (n : ℕ) : ℕ := n * (5 * n + 13) / 2
def s₂ (n : ℕ) : ℕ := n * (3 * n + 37) / 2

-- State the theorem: for given n != 0, prove s₁ n = s₂ n implies n = 12
theorem equal_sums_arithmetic_sequences (n : ℕ) (h : n ≠ 0) : 
  s₁ n = s₂ n → n = 12 :=
by
  sorry

end equal_sums_arithmetic_sequences_l146_146425


namespace sign_of_f_l146_146876

variable {a b c R r : ℝ}
variable (A B C : ℝ) -- Angles of the triangle

-- assuming the triangle sides inequalities
axiom sides_inequality : a ≤ b ∧ b ≤ c
-- Circumradius of the triangle
axiom circumradius : R = a * b * c / (4 * (Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a))))
-- Inradius of the triangle
axiom inradius : r = (a + b + c) / 2
-- definition of f
def f := a + b - 2 * R - 2 * r

theorem sign_of_f (h₁: a + b > c) (h₂: b + c > a) (h₃: c + a > b) (h₄: a > 0) (h₅: b > 0) (h₆: c > 0): 
  (0 < C ∧ C < Real.pi / 2 → f > 0) ∧ 
  (C = Real.pi / 2 → f = 0) ∧ 
  (Real.pi / 2 < C ∧ C < Real.pi → f < 0) := by
  sorry

end sign_of_f_l146_146876


namespace log_a_b_integer_probability_l146_146536

theorem log_a_b_integer_probability :
  let S := { k : ℕ | 1 ≤ k ∧ k ≤ 25 },
      valid_pairs := (∏ x in S, (S.count (λ y, x ≠ y ∧ x ∣ y))) in
  (Σ' x y in valid_pairs, x ≠ y).card / (S.card.choose 2) = 31 / 300 := sorry

end log_a_b_integer_probability_l146_146536


namespace sum_first_n_terms_seq_cn_l146_146423

theorem sum_first_n_terms_seq_cn (n : ℕ) : 
  (∑ i in Finset.range n, ((i + 1) - 2) + Real.log 2 ((2 : ℝ) ^ i)) = n^2 - 2*n :=
by
  sorry

end sum_first_n_terms_seq_cn_l146_146423


namespace initial_number_of_students_l146_146031

theorem initial_number_of_students (W : ℝ) (n : ℕ) (new_student_weight avg_weight1 avg_weight2 : ℝ)
  (h1 : avg_weight1 = 15)
  (h2 : new_student_weight = 13)
  (h3 : avg_weight2 = 14.9)
  (h4 : W = n * avg_weight1)
  (h5 : W + new_student_weight = (n + 1) * avg_weight2) : n = 19 := 
by
  sorry

end initial_number_of_students_l146_146031


namespace largest_multiple_of_15_under_500_l146_146992

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146992


namespace largest_multiple_of_15_under_500_l146_146990

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146990


namespace solve_abs_eq_l146_146467

theorem solve_abs_eq (x : ℝ) : (|x + 4| = 3 - x) → (x = -1/2) := by
  intro h
  sorry

end solve_abs_eq_l146_146467


namespace find_a_l146_146804

noncomputable def f (x a : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem find_a (a : ℝ) (h : ∀ x ∈ Icc 0 (Real.pi / 2), f x a ≥ -4) :
    a = -4 :=
sorry

end find_a_l146_146804


namespace max_value_in_product_range_l146_146021

theorem max_value_in_product_range 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ set.Icc (-7 : ℝ) 4) 
  (hg : ∀ x, g x ∈ set.Icc (-3 : ℝ) 2) : 
  ∃ b, b = 21 ∧ ∀ y, y ∈ set.range (λ x, f x * g x) → y ≤ b :=
by
  sorry

end max_value_in_product_range_l146_146021


namespace power_addition_l146_146291

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146291


namespace minimize_relative_waiting_time_l146_146700

theorem minimize_relative_waiting_time 
  (a b c : ℕ) (h : a < b ∧ b < c) :
  let relative_waiting_time_sum (x y z : ℕ) : ℕ := 1 + 1 + 1  -- Based on default values
  in
  relative_waiting_time_sum a b c ≤ relative_waiting_time_sum b c a ∧
  relative_waiting_time_sum a b c ≤ relative_waiting_time_sum c a b ∧
  relative_waiting_time_sum a b c ≤ relative_waiting_time_sum a c b := sorry

end minimize_relative_waiting_time_l146_146700


namespace find_f_at_2_l146_146806

noncomputable def f (x : ℝ) : ℝ :=
  (x + 2) * (x - 1) * (x - 3) * (x + 4) - x^2

theorem find_f_at_2 :
  (f(-2) = -4) ∧ (f(1) = -1) ∧ (f(3) = -9) ∧ (f(-4) = -16) ∧ (f 2 = -28) :=
by
  have h₁ : f (-2) = (0 : ℝ) := by sorry
  have h₂ : f 1 = (0 : ℝ) := by sorry
  have h₃ : f 3 = (0 : ℝ) := by sorry
  have h₄ : f (-4) = (0 : ℝ) := by sorry
  have h₅ : f 2 = (0 : ℝ) := by sorry
  exact ⟨h₁, h₂, h₃, h₄, h₅⟩

end find_f_at_2_l146_146806


namespace determinant_identity_l146_146641

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l146_146641


namespace csc_315_eq_neg_sqrt_2_l146_146179

theorem csc_315_eq_neg_sqrt_2 :
  (∀ θ, (csc θ) = 1 / sin θ) →
  sin (360 - 45 : ℝ) = sin 315 →
  (∀ θ : ℝ, sin (360 - θ) = - (sin θ)) →
  sin 45 = 1 / Real.sqrt 2 →
  csc 315 = - Real.sqrt 2 :=
by
  intros h1 h2 h3 h4
  sorry

end csc_315_eq_neg_sqrt_2_l146_146179


namespace find_n_satisfying_conditions_l146_146191

theorem find_n_satisfying_conditions : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -3402 [MOD 10] ∧ n = 8 :=
by
  sorry

end find_n_satisfying_conditions_l146_146191


namespace doubling_time_l146_146042

theorem doubling_time 
  (P0 : ℝ)
  (hP0 : P0 = 1000)
  (r : ℝ)
  (t : ℝ)
  (h_growth : ∀ t : ℝ, P0 * real.exp (r * t) > 128000)
  (h_exp_growth : P0 * real.exp (r * 10) > 128000) :
  (∃ T : ℝ, T > 1.43 ∧ P0 * real.exp (r * T) = 2 * P0) :=
sorry

end doubling_time_l146_146042


namespace courtyard_width_l146_146065

theorem courtyard_width 
  (length_of_courtyard : ℝ) 
  (num_paving_stones : ℕ) 
  (length_of_stone width_of_stone : ℝ) 
  (total_area_stone : ℝ) 
  (W : ℝ) : 
  length_of_courtyard = 40 →
  num_paving_stones = 132 →
  length_of_stone = 2.5 →
  width_of_stone = 2 →
  total_area_stone = 660 →
  40 * W = 660 →
  W = 16.5 :=
by
  intros
  sorry

end courtyard_width_l146_146065


namespace circle_packing_line_equation_l146_146474

theorem circle_packing_line_equation
  (d : ℝ) (n1 n2 n3 : ℕ) (slope : ℝ)
  (l_intersects_tangencies : ℝ → ℝ → Prop)
  (l_divides_R : Prop)
  (gcd_condition : ℕ → ℕ → ℕ → ℕ)
  (a b c : ℕ)
  (a_pos : 0 < a) (b_neg : b < 0) (c_pos : 0 < c)
  (gcd_abc : gcd_condition a b c = 1)
  (correct_equation_format : Prop) :
  n1 = 4 ∧ n2 = 4 ∧ n3 = 2 →
  d = 2 →
  slope = 5 →
  l_divides_R →
  l_intersects_tangencies 1 1 →
  l_intersects_tangencies 4 6 → 
  correct_equation_format → 
  a^2 + b^2 + c^2 = 42 :=
by sorry

end circle_packing_line_equation_l146_146474


namespace max_elements_S_l146_146160

-- Define concept of short rational number
def isShort (x : ℚ) : Prop :=
  ∃ (a b : ℕ), (2^a * 5^b : ℚ) * x ∈ ℤ

-- Define condition for m-tastic numbers
def is_m_tastic (m t : ℕ) : Prop :=
  ∃ (c : ℕ), 1 ≤ c ∧ c ≤ 2017 ∧
  (isShort ((10^t - 1) / (c * m : ℚ)) ∧
  ∀ k, 1 ≤ k ∧ k < t → ¬ isShort ((10^k - 1) / (c * m : ℚ)))

-- Define set S(m)
def S (m : ℕ) : Set ℕ :=
  {t | is_m_tastic m t}

-- Define the theorem
theorem max_elements_S : ∃ m, #(S m) = 807 :=
sorry

end max_elements_S_l146_146160


namespace smaller_angle_between_east_and_northwest_l146_146585

theorem smaller_angle_between_east_and_northwest
  (rays : ℕ)
  (each_angle : ℕ)
  (direction : ℕ → ℝ)
  (h1 : rays = 10)
  (h2 : each_angle = 36)
  (h3 : direction 0 = 0) -- ray at due North
  (h4 : direction 3 = 90) -- ray at due East
  (h5 : direction 5 = 135) -- ray at due Northwest
  : direction 5 - direction 3 = each_angle :=
by
  -- to be proved
  sorry

end smaller_angle_between_east_and_northwest_l146_146585


namespace chef_can_determine_friendships_l146_146578

/--
Given the conditions:
1. The chef supervises ten kitchen apprentices.
2. Some apprentices are friends with each other.
3. The chef assigns one or several apprentices to duty each working day for 45 days.
4. Each apprentice on duty takes one pastry for each of their non-duty friends at the end of the day.
5. The chef is informed of the number of missing pastries at the end of the day.

Prove that the chef can figure out which apprentices are friends with each other.
-/
theorem chef_can_determine_friendships 
    (chef : Type) 
    (apprentices : Fin 10 → Type) 
    (are_friends : apprentices → apprentices → Prop) 
    (assign_to_duty : ℕ → Set (Fin 10)) 
    (missing_pastries : ℕ → ℕ) :
  ∃ (day : Fin 45 → Set (Fin 10)), 
  (∀ n : Fin 45, ∃ s : Set (Fin 10), assign_to_duty n = s) →
  (∀ i j : Fin 10, i ≠ j → are_friends i j = 
    (missing_pastries (index_day i j) = missing_pastries (index_day i) + missing_pastries (index_day j))) →
  (∀ i : Fin 10, ∃ fi : Fin 10, are_friends i fi) :=
by
  sorry

end chef_can_determine_friendships_l146_146578


namespace largest_multiple_of_15_less_than_500_l146_146985

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146985


namespace domain_log_function_l146_146931

/-- The quadratic expression x^2 - 2x + 3 is always positive. -/
lemma quadratic_positive (x : ℝ) : x^2 - 2*x + 3 > 0 :=
by
  sorry

/-- The domain of the function y = log(x^2 - 2x + 3) is all real numbers. -/
theorem domain_log_function : ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*x + 3) :=
by
  have h := quadratic_positive
  sorry

end domain_log_function_l146_146931


namespace largest_multiple_of_15_less_than_500_l146_146977

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146977


namespace center_of_symmetry_intervals_of_increase_l146_146782

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - cos x ^ 2 - 1/2

theorem center_of_symmetry : ∃ k : ℤ, ∀ x, f x = f (π - x) ↔ x = (k * π / 2 + π / 12) := sorry

theorem intervals_of_increase (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) : 
    (0 ≤ x ∧ x ≤ π / 3) ∨ (5 * π / 6 ≤ x ∧ x ≤ π) → 
    (∀ y ∈ Icc x (x + π), f y ≤ f (y + ε) ∨ f y < f (y - ε) for ε > 0) := sorry

end center_of_symmetry_intervals_of_increase_l146_146782


namespace find_ratio_l146_146373

-- Define the variables and conditions
variables {m n r t : ℚ}

-- Conditions
def cond1 := m / n = 5 / 2
def cond2 := r / t = 7 / 5

-- Theorem
theorem find_ratio (h1 : cond1) (h2 : cond2) : (4 * m * r - 5 * n * t) / (3 * n * t - 2 * m * r) = -9 / 4 :=
by
  sorry

end find_ratio_l146_146373


namespace square_position_1011th_l146_146034

theorem square_position_1011th :
  ∀ (initial : ℕ → string), 
  initial 0 = "ABCD" →
  initial 1 = "BADC" →
  initial 2 = "DCBA" →
  initial 3 = "CBAD" →
  (∀ n, initial (n + 4) = initial n) →
  initial 1011 = "DCBA" :=
by
  intros initial h0 h1 h2 h3 hp
  sorry

end square_position_1011th_l146_146034


namespace orcs_carry_swords_l146_146027

theorem orcs_carry_swords:
  (let total_swords := 1200 in
   let squads := 10 in
   let orcs_per_squad := 8 in
   let total_orcs := squads * orcs_per_squad in
   let swords_per_orc := total_swords / total_orcs in
   swords_per_orc = 15) :=
by
  sorry

end orcs_carry_swords_l146_146027


namespace compute_combination_l146_146690

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146690


namespace tv_horizontal_length_l146_146896

noncomputable def rectangleTvLengthRatio (l h : ℝ) : Prop :=
  l / h = 16 / 9

noncomputable def rectangleTvDiagonal (l h d : ℝ) : Prop :=
  l^2 + h^2 = d^2

theorem tv_horizontal_length
  (h : ℝ)
  (h_positive : h > 0)
  (d : ℝ)
  (h_ratio : rectangleTvLengthRatio l h)
  (h_diagonal : rectangleTvDiagonal l h d)
  (h_diagonal_value : d = 36) :
  l = 56.27 :=
by
  sorry

end tv_horizontal_length_l146_146896


namespace unique_function_theorem_l146_146912

noncomputable def unique_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ≠ 0 → f(x) = x * f(1/x)) ∧
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x ≠ -y → f(x) + f(y) = 1 + f(x + y))

theorem unique_function_theorem : ∃! (f : ℝ → ℝ), unique_function f ∧ (∀ x : ℝ, x ≠ 0 → f x = x + 1) :=
sorry

end unique_function_theorem_l146_146912


namespace billiard_reflection_l146_146097

theorem billiard_reflection 
  (table_width : ℕ) (table_length : ℕ)
  (start_angle : ℝ)
  (incidence_equals_reflection : Prop) :
  table_width = 26 ∧ table_length = 1965 ∧ start_angle = 45 → 
  ∃ n : ℕ, hit_coordinates (n*50990, n*50990) = (0, table_length) :=
by
  sorry

end billiard_reflection_l146_146097


namespace power_equality_l146_146319

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146319


namespace csc_315_l146_146178

theorem csc_315 (deg_to_rad : ℝ → ℝ) (sin : ℝ → ℝ) (csc : ℝ := λ θ, 1 / sin θ) :
  (∀ θ, sin (2 * π - θ) = -sin θ) →
  sin (π / 4) = 1 / real.sqrt 2 →
  csc (7 * π / 4) = -real.sqrt 2 :=
by
  intro angle_identity
  intro sin_45
  sorry

end csc_315_l146_146178


namespace width_to_length_ratio_l146_146396

variable (w : ℕ)

def length := 10
def perimeter := 36

theorem width_to_length_ratio
  (h_perimeter : 2 * w + 2 * length = perimeter) :
  w / length = 4 / 5 :=
by
  -- Skipping proof steps, putting sorry
  sorry

end width_to_length_ratio_l146_146396


namespace intersection_point_on_circumcircle_l146_146527

theorem intersection_point_on_circumcircle
  (A B C O M N A' B' : Point)
  (h₀ : IsTriangle A B C)
  (h₁ : AngleBisectorsMeetAt A B C O)
  (h₂ : LineThroughPerpendicularTo CO O M N)
  (hM : M ∈ LineSegment A C)
  (hN : N ∈ LineSegment B C)
  (hA' : A' ∈ Intersection (Line AO) (Circumcircle A B C))
  (hB' : B' ∈ Intersection (Line BO) (Circumcircle A B C))
  : Intersection (Line A' N) (Line B' M) ∈ Circumcircle A B C := sorry

end intersection_point_on_circumcircle_l146_146527


namespace temperature_problem_l146_146154

theorem temperature_problem (N : ℤ) (P : ℤ) (D : ℤ) (D_3_pm : ℤ) (P_3_pm : ℤ) :
  D = P + N →
  D_3_pm = D - 8 →
  P_3_pm = P + 9 →
  |D_3_pm - P_3_pm| = 1 →
  (N = 18 ∨ N = 16) →
  18 * 16 = 288 :=
by
  sorry

end temperature_problem_l146_146154


namespace part1_monotonically_decreasing_part2_range_of_a_l146_146251

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) := 2 * x * log x - 2 * a * x^2

-- Part (1) statement
theorem part1_monotonically_decreasing (a : ℝ) (h : a = 1 / 2) : 
  ∀ x, 0 < x → deriv (λ x, f x (1 / 2)) x ≤ 0 :=
sorry

-- Part (2) statement
theorem part2_range_of_a (h : ∀ x, 1 < x → f x a ≤ (\frac{\partial f x a}{\partial x}) / 2 - log x - 1) : 
  ∀ a, a ∈ Set.Ici (1 : ℝ) :=
sorry

end part1_monotonically_decreasing_part2_range_of_a_l146_146251


namespace profit_percentage_correct_l146_146595

def CP : ℝ := 340  -- Cost Price in Rs
def SP : ℝ := 374  -- Selling Price in Rs
def profit : ℝ := SP - CP  -- Definition of Profit
def profit_percentage : ℝ := (profit / CP) * 100  -- Definition of Profit Percentage

theorem profit_percentage_correct : profit_percentage = 10 :=
by
  sorry -- Proof is omitted

end profit_percentage_correct_l146_146595


namespace payment_plan_months_l146_146612

theorem payment_plan_months 
  (M T : ℝ) (r : ℝ) 
  (hM : M = 100)
  (hT : T = 1320)
  (hr : r = 0.10)
  : ∃ t : ℕ, t = 12 ∧ T = (M * t) + (M * t * r) :=
by
  sorry

end payment_plan_months_l146_146612


namespace find_ratio_l146_146268

-- Given conditions
variable (x y a b : ℝ)
variable (h1 : 2 * x - y = a)
variable (h2 : 4 * y - 8 * x = b)
variable (h3 : b ≠ 0)

theorem find_ratio (a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 4 := by
  sorry

end find_ratio_l146_146268


namespace power_calculation_l146_146314

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146314


namespace problem_statement_l146_146555

noncomputable def nonnegative_reals : Type := {x : ℝ // 0 ≤ x}

theorem problem_statement (x : nonnegative_reals) :
  x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) ≥ 15*x.1 ∧
  (x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) = 15*x.1 ↔ (x.1 = 0 ∨ x.1 = 1)) :=
by
  sorry

end problem_statement_l146_146555


namespace probability_both_tails_l146_146072

-- Definitions based on the conditions
def Coin := Prop  -- A coin has two sides
def heads : Coin := -- Γ represents heads
sorry
def tails : Coin := -- L represents tails
sorry

-- Event for both coins showing tails
def both_tails (c1 c2 : Coin) : Prop :=
  c1 = tails ∧ c2 = tails

-- The possible outcomes when two symmetric coins are flipped
def outcomes : List (Coin × Coin) :=
  [(heads, heads), (heads, tails), (tails, heads), (tails, tails)]

-- The number of favorable outcomes (both tails)
def favorable_outcomes : Nat :=
  (outcomes.filter (λ x => both_tails x.fst x.snd)).length

-- The total number of possible outcomes
def total_outcomes : Nat := outcomes.length

-- Calculate the probability of event D
def probability_D : ℚ :=
  favorable_outcomes.toRational / total_outcomes.toRational

-- The theorem statement
theorem probability_both_tails : probability_D = 0.25 :=
sorry

end probability_both_tails_l146_146072


namespace distributor_profit_percentage_l146_146113

theorem distributor_profit_percentage :
  ∀ (c p : ℝ) (r : ℝ), 
  (c = 18) ∧ (p = 27) ∧ (r = 0.20) →
  ((1 - r) * p - c) / c * 100 = 25 :=
by
  intros c p r h
  cases h with h1 h2
  cases h2 with h2 h3
  rw [h1, h2, h3]
  sorry

end distributor_profit_percentage_l146_146113


namespace order_of_divides_cardinality_l146_146565

theorem order_of_divides_cardinality (G : Type*) [group G] [fintype G] (g : G) :
  order_of g ∣ fintype.card G :=
sorry

end order_of_divides_cardinality_l146_146565


namespace find_sum_S_l146_146867

noncomputable def sequence_a : ℕ → ℤ
| 1 := -1
| (n+1) := (λ S_n S_nplus1, S_n * S_nplus1) (sum (sequence_a <$> (range n))) (sum (sequence_a <$> (range (n + 1))))

noncomputable def sum_S (n : ℕ) : ℤ :=
sum (sequence_a <$> (range n))

theorem find_sum_S (n : ℕ) : sum_S n = -(1 / n) := sorry

end find_sum_S_l146_146867


namespace find_FC_l146_146745

theorem find_FC 
  (DC CB AD: ℝ)
  (h1 : DC = 9)
  (h2 : CB = 6)
  (h3 : AB = (1 / 3) * AD)
  (h4 : ED = (2 / 3) * AD) :
  FC = 9 :=
sorry

end find_FC_l146_146745


namespace total_buttons_needed_l146_146008

def shirtsMonday : ℕ := 4
def shirtsTuesday : ℕ := 3
def shirtsWednesday : ℕ := 2
def buttonsPerShirt : ℕ := 5

theorem total_buttons_needed : (shirtsMonday + shirtsTuesday + shirtsWednesday) * buttonsPerShirt = 45 :=
by
  have shirtsTotal : ℕ := shirtsMonday + shirtsTuesday + shirtsWednesday
  have buttonsTotal : ℕ := shirtsTotal * buttonsPerShirt
  have h1 := rfl
  rw [← h1, add_assoc, ← add_assoc shirtsTuesday shirtsWednesday, add_comm shirtsThursday shirtsWednesday, add_assoc shirtsTuesday shirtsWednesday shirtsMonday] at h1
  rw [← h1, mul_add, ← add_mul] at h1
  sorry

end total_buttons_needed_l146_146008


namespace binom_9_5_eq_126_l146_146658

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146658


namespace sum_of_intercepts_of_line_l146_146511

theorem sum_of_intercepts_of_line (x y : ℝ) (hx : 2 * x - 3 * y + 6 = 0) :
  2 + (-3) = -1 :=
sorry

end sum_of_intercepts_of_line_l146_146511


namespace power_addition_l146_146298

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146298


namespace coefficient_of_x_squared_l146_146404

theorem coefficient_of_x_squared :
  (∑ r in finset.range 11, ((-1)^r * (nat.desc_factorial 10 r) / (r!)) * (1 / x)^(10 - r) * (x^(1/2))^r = 45) :=
begin
  sorry
end

end coefficient_of_x_squared_l146_146404


namespace grocer_banana_pounds_l146_146589

theorem grocer_banana_pounds :
  ∃ P : ℕ, 
  (P / 3) * 0.50 = P * 0.1667 ∧
  (P / 4) * 1.00 = P * 0.25 ∧
  (P * 0.25 - P * 0.1667) = 7.00 :=
sorry

end grocer_banana_pounds_l146_146589


namespace isosceles_triangle_one_obtuse_angle_l146_146617

-- Define the basic properties of our triangle and the obtuse angle
def isosceles_triangle_obtuse (α θ : ℝ) (h_iso : θ = 30 ∧ θ + θ + α = 180) (h_obtuse : α = 120) : Prop :=
  α > 90 ∧ α < 180 ∧ θ < 90 ∧ θ < 90

-- Define the final proposition to be proved
theorem isosceles_triangle_one_obtuse_angle (α θ : ℝ)
  (h_iso : θ = 30 ∧ θ + θ + α = 180)
  (h_obtuse : α = 120) :
  isosceles_triangle_obtuse α θ h_iso h_obtuse :=
begin
  sorry
end

end isosceles_triangle_one_obtuse_angle_l146_146617


namespace find_m_l146_146402

noncomputable def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * x

def parametric_line (m t : ℝ) : ℝ × ℝ :=
  (m + (√3 / 2) * t, (1 / 2) * t)

def intersection_params (m t : ℝ) : Prop :=
  let ⟨x, y⟩ := parametric_line m t in
  curve x y

theorem find_m (m : ℝ) :
  (∃ t1 t2 : ℝ, intersection_params m t1 ∧ intersection_params m t2 ∧ |m^2 - 2*m| = 1) ↔
    (m = 1 ∨ m = 1 + real.sqrt 2 ∨ m = 1 - real.sqrt 2) :=
by
  sorry

end find_m_l146_146402


namespace sequence_formula_l146_146265

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 33) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 :=
by
  sorry

end sequence_formula_l146_146265


namespace power_of_3_l146_146351

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146351


namespace min_elements_in_S_l146_146865

noncomputable def exists_function_with_property (S : Type) [fintype S]
  (f : ℕ → S) : Prop :=
∀ (x y : ℕ), nat.prime (abs (x - y)) → f x ≠ f y

theorem min_elements_in_S (S : Type) [fintype S]
  (h : ∃ f : ℕ → S, exists_function_with_property S f) : 
  fintype.card S ≥ 4 :=
sorry

end min_elements_in_S_l146_146865


namespace binom_two_formula_l146_146541

def binom (n k : ℕ) : ℕ :=
  n.choose k

-- Formalizing the conditions
variable (n : ℕ)
variable (h : n ≥ 2)

-- Stating the problem mathematically in Lean
theorem binom_two_formula :
  binom n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_formula_l146_146541


namespace power_equality_l146_146321

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146321


namespace centroid_path_circle_l146_146794

-- Given triangle ABC with fixed base AB and midpoint M of AB.
-- Vertex C moves on a circle centered at M with radius R.
-- We need to prove that the path of the centroid G of triangle ABC is a circle with radius (2/3)R centered at M.

theorem centroid_path_circle (A B C M G : ℝ × ℝ) (R : ℝ) (hABC : AB = A ∨ B ∨ C)
                               (hM : M = (A + B) / 2)
                               (hC : dist M C = R)
                               (hG : G = (A + B + C) / 3) : 
  ∃ r, r = (2/3) * R ∧ dist M G = r :=
by
  sorry

end centroid_path_circle_l146_146794


namespace remove_balls_eq_catalan_number_l146_146212

-- Define the number of ways to remove the balls satisfying the given conditions.
noncomputable def num_ways_to_remove_balls (n : ℕ) : ℕ := (nat.choose (2 * n - 2) (n - 1)) / n

-- State the theorem that we want to prove
theorem remove_balls_eq_catalan_number (n : ℕ) : 
  num_ways_to_remove_balls n = (nat.choose (2 * n - 2) (n - 1)) / n := 
by
  sorry

end remove_balls_eq_catalan_number_l146_146212


namespace solve_equation_l146_146018

theorem solve_equation :
  ∀ (x : ℚ), x ≠ 1 → (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by
  intro x hx
  split
  { intro h
    sorry }
  { intro h
    rw [h]
    norm_num }

end solve_equation_l146_146018


namespace exceedances_convergence_l146_146560

noncomputable def Z (n : ℕ) (S : ℕ) : ℤ :=
2 * (S : ℤ) - n

def prob_exceedances (n j : ℕ) : ℝ :=
sorry  -- Placeholder for the actual probability function

theorem exceedances_convergence (n : ℕ) [fact (0 < n)] :
sup (λ j, | √(π * n) * (prob_exceedances (2 * n) j) - exp (-(j^2) / (4 * n)) |) →ᶠ[Filter.atTop] 0 := sorry

end exceedances_convergence_l146_146560


namespace range_of_a_l146_146170

theorem range_of_a (a : ℝ) : (∃ x : ℝ, sin x ^ 2 - 2 * sin x - a = 0) ↔ a ∈ set.Icc (-1 : ℝ) 3 :=
sorry

end range_of_a_l146_146170


namespace range_of_a_l146_146780

noncomputable theory

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := (2 / (x + 1)) - (1 / 2) * Real.log x

def tangent_at_1 : Prop := ∀ y : ℝ, 1 + y - 2 = 0 → y = 1

def monotonicity_on (a b : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x ∈ Ioo a b → y ∈ Ioo a b → x < y → f x > f y

-- Main theorem
theorem range_of_a (p : ℝ) (h1 : 0 < p ∧ p < 1)
  (h2 : f p = 2)
  (condition : ∀ x ∈ Ioo p 1, ∀ t ∈ Icc (1 / 2) 2, f x ≥ t^3 - t^2 - 2 * a * t + 2 ∨ f x ≤ t^3 - t^2 - 2 * a * t + 2) :
  a ∈ Set.Iic (-1 / 8) ∪ Set.Ici (5 / 4) :=
sorry

end range_of_a_l146_146780


namespace power_equality_l146_146330

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146330


namespace constant_term_expansion_eq_20_l146_146484

theorem constant_term_expansion_eq_20 :
  let T (r : ℕ) := (binomial 5 r) * (x^2)^(5-r) * (- sqrt (2/x))^r in
  (∃ r : ℕ, x^(10 - 2 * r - r / 2) = 1) ∧ r = 4 →
  T 4 = 20 := by sorry

end constant_term_expansion_eq_20_l146_146484


namespace a_5_equals_54_l146_146052

-- Definitions for the conditions.
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := (list.range (n + 1)).map a |>.sum

def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * S a n

-- The statement of the theorem (proof is omitted by sorry).
theorem a_5_equals_54 : a 5 = 54 :=
  by sorry

end a_5_equals_54_l146_146052


namespace find_b_l146_146820

theorem find_b
  (b : ℝ)
  (hx : ∃ y : ℝ, 4 * 3 + 2 * y = b ∧ 3 * 3 + 4 * y = 3 * b) :
  b = -15 :=
sorry

end find_b_l146_146820


namespace smallest_n_divisible_by_2009_l146_146715

theorem smallest_n_divisible_by_2009 : ∃ n : ℕ, n > 1 ∧ (n^2 * (n - 1)) % 2009 = 0 ∧ (∀ m : ℕ, m > 1 → (m^2 * (m - 1)) % 2009 = 0 → m ≥ n) :=
by
  sorry

end smallest_n_divisible_by_2009_l146_146715


namespace no_rational_roots_l146_146184

theorem no_rational_roots : ¬ ∃ x : ℚ, 5 * x^3 - 4 * x^2 - 8 * x + 3 = 0 :=
by
  sorry

end no_rational_roots_l146_146184


namespace students_exceed_pets_l146_146717

-- Defining the conditions
def num_students_per_classroom := 25
def num_rabbits_per_classroom := 3
def num_guinea_pigs_per_classroom := 3
def num_classrooms := 5

-- Main theorem to prove
theorem students_exceed_pets:
  let total_students := num_students_per_classroom * num_classrooms
  let total_rabbits := num_rabbits_per_classroom * num_classrooms
  let total_guinea_pigs := num_guinea_pigs_per_classroom * num_classrooms
  let total_pets := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 :=
by 
  sorry

end students_exceed_pets_l146_146717


namespace exponent_power_identity_l146_146341

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146341


namespace hyperbola_asymptotes_iff_l146_146449

def hyperbola_asymptotes_orthogonal (a b c d e f : ℝ) : Prop :=
  a + c = 0

theorem hyperbola_asymptotes_iff (a b c d e f : ℝ) :
  (∃ x y : ℝ, a * x^2 + 2 * b * x * y + c * y^2 + d * x + e * y + f = 0) →
  hyperbola_asymptotes_orthogonal a b c d e f ↔ a + c = 0 :=
by sorry

end hyperbola_asymptotes_iff_l146_146449


namespace trigonometric_identity_l146_146389

theorem trigonometric_identity (a : ℝ) (h : a ≠ π / 4 ∧ a ≠ 3 * π / 4) :
  (sin a / sqrt (1 - sin a ^ 2) + sqrt (1 - cos a ^ 2) / cos a) = 0 :=
by sorry

end trigonometric_identity_l146_146389


namespace probability_of_black_ball_l146_146518

-- Definitions
def total_balls : ℕ := 100
def red_balls : ℕ := 45
def prob_white : ℝ := 0.23

-- Theorem statement
theorem probability_of_black_ball :
  ∃ black_balls : ℕ,
  let white_balls := total_balls * prob_white,
      black_balls := total_balls - red_balls - white_balls
  in black_balls / total_balls = 0.32 :=
sorry

end probability_of_black_ball_l146_146518


namespace distribute_books_l146_146521

theorem distribute_books :
  ∃ (ways : Nat), ways = 60 ∧ 
  (ways = 
    let books := {1, 2, 3, 4}
    let people := {A, B, C}
    -- Each person must get at least one book
    ∑ (f : books → people) in
      finset.univ.filter (λ f, ∀ p, ∃ b, f b = p), 1
  ) := sorry

end distribute_books_l146_146521


namespace total_balls_l146_146059

theorem total_balls (box1 box2 : ℕ) (h1 : box1 = 3) (h2 : box2 = 4) : box1 + box2 = 7 := by
  rw [h1, h2]
  rfl

end total_balls_l146_146059


namespace total_hamburgers_sold_is_63_l146_146601

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end total_hamburgers_sold_is_63_l146_146601


namespace simplify_sqrt_1001_l146_146463

theorem simplify_sqrt_1001 :
  ¬(∃ k, 1001 = k^2) → ∀ x, (1001 = 7 * 11 * 13 ∧ (x = 2 ∨ x = 3) → ¬(x^2 ∣ 1001)) →
  (√1001 = √1001) :=
by
  intro h1 h2
  exact (eq.refl (√1001))

end simplify_sqrt_1001_l146_146463


namespace solve_equation_l146_146016

theorem solve_equation (x : ℚ) (h : x ≠ 1) : (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by 
  sorry

end solve_equation_l146_146016


namespace minimum_length_of_EF_l146_146401

open Real

-- Define the problem conditions
def equilateral_triangle (A B C : Point) : Prop := 
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

def point_on_segment (P A B : Point) : Prop :=
  dist A P + dist P B = dist A B

def area_ratio_triangle (A E F : Point) (AE AF : ℝ): Prop :=
  1 / 2 * AE * AF * sin (angle A (E) (F)) = 1 / 2 * 2 * 2 * sin (angle A (B) (C)) / 3

-- Main statement of the proof
theorem minimum_length_of_EF {A B C E F : Point} 
  (h_triangle : equilateral_triangle A B C)
  (h_E_on_AB : point_on_segment E A B)
  (h_F_on_AC : point_on_segment F A C)
  (h_area_ratio : area_ratio_triangle A E F 2) :
  ∃ (EF : ℝ), EF = 2 * sqrt 3 / 3 :=
begin
  sorry -- proof to be completed
end

end minimum_length_of_EF_l146_146401


namespace problem_1_problem_2_l146_146406

variables (n α q : ℝ) (a b c r Δ : ℕ → ℝ)
variables (a1 a2 R1 R2 : ℝ)

theorem problem_1 :
  (∏ i in finset.range n, a i ^ q) + 
  (∏ i in finset.range n, b i ^ q) + 
  (∏ i in finset.range n, c i ^ q) 
  ≥ 2^(n*q) * 3^((4 - n * q) / 4) * (∏ i in finset.range n, (Δ i) ^ (q / 2)) ∧ 
  (∏ i in finset.range n, Δ i ^ (q / 2)) 
  ≥ 2^(n*q) * 3^((2 + n*q) / 2) * (∏ i in finset.range n, r i ^ q) :=
sorry

theorem problem_2 (ha1 : 1 ≤ a1) (ha2 : 1 ≤ a2) :
  a1^a1 * a2^a2 + b1^a1 * b2^a2 + c1^a1 * c2^a2 
  ≤ 9 * 2^(a1 + a2 - 2) * R1^a1 * R2^a2 :=
sorry

end problem_1_problem_2_l146_146406


namespace T_10_eq_13377_T_2013_units_digit_eq_3_l146_146539

noncomputable def T : ℕ → ℕ
| 1        := 1
| 2        := 5
| n+3      := T (n+2) + 4 * T (n+1) + 2 * T n

theorem T_10_eq_13377 :
  T 10 = 13377 :=
sorry

theorem T_2013_units_digit_eq_3 :
  (T 2013) % 10 = 3 :=
sorry

end T_10_eq_13377_T_2013_units_digit_eq_3_l146_146539


namespace solution_set_inequality_l146_146871

-- Define f and its properties
variable {f : ℝ → ℝ}
variable [Differentiable ℝ f]
variable (H1 : ∀ x : ℝ, f x + deriv f x > 0)
variable (H2 : f 1 = 1)

-- Statement: The solution set of the inequality f(x) > exp(1 - x) is (1, +∞)
theorem solution_set_inequality : {x : ℝ | f x > exp (1 - x)} = Ioi 1 :=
by
  sorry

end solution_set_inequality_l146_146871


namespace total_hamburgers_sold_is_63_l146_146600

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end total_hamburgers_sold_is_63_l146_146600


namespace pedestrians_travel_times_l146_146966

-- Definitions based on the given conditions
def time_meet : Real := 2
def time_diff : Real := 1.67
def distance_A_to_B : Real := 18
def speed_A := distance_A_to_B / (time_meet + time_diff)
def speed_B := distance_A_to_B / (time_meet)

-- The main statement to prove the distances and speeds
theorem pedestrians_travel_times {d_AB : Real} {v_A v_B : Real} 
  (h1 : d_AB = 18) 
  (h2 : v_A = 5) 
  (h3 : v_B = 4) :
  time_meet = 2 ∧ time_diff = 1.67 ∧ d_AB = 18 ∧ v_A = 5 ∧ v_B = 4 :=
by
  sorry

end pedestrians_travel_times_l146_146966


namespace power_addition_l146_146290

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146290


namespace equal_savings_l146_146508

theorem equal_savings (A B AE BE AS BS : ℕ) 
  (hA : A = 2000)
  (hA_B : 5 * B = 4 * A)
  (hAE_BE : 3 * BE = 2 * AE)
  (hSavings : AS = A - AE ∧ BS = B - BE ∧ AS = BS) :
  AS = 800 ∧ BS = 800 :=
by
  -- Placeholders for definitions and calculations
  sorry

end equal_savings_l146_146508


namespace maximum_number_of_divisors_l146_146884

noncomputable def maximum_integers_with_divisors (π : Perm (Fin 2012)) : Nat :=
  (Finset.filter (λ n, π (Fin.castLT n sorry).val ∣ π (Fin.succ (Fin.castLT n sorry)).val) 
                 (Finset.range 2011)).card

theorem maximum_number_of_divisors {π : Perm (Fin 2012)} : 
  maximum_integers_with_divisors π = 1006 :=
sorry

end maximum_number_of_divisors_l146_146884


namespace bisects_segment_by_circumcircle_l146_146910

theorem bisects_segment_by_circumcircle (A B C O O_b M : Type) (hABC : Triangle A B C) 
(hO : inscribed_circle_center A B C O) 
(hOb : exscribed_circle_center_touching_AC A B C O_b) 
(hM : angle_bisector_intersects_circumcircle A B C M) : (segment_length M O) = (segment_length M O_b) := 
sorry

end bisects_segment_by_circumcircle_l146_146910


namespace unit_vector_parallel_l146_146515

def vector_d := (12, 5 : ℝ × ℝ)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 / magnitude v, v.2 / magnitude v)

theorem unit_vector_parallel {v : ℝ × ℝ} (h : v = vector_d) :
  unit_vector v = (12 / 13, 5 / 13) ∨ unit_vector v = (- 12 / 13, - 5 / 13) :=
by {
  sorry,
}

end unit_vector_parallel_l146_146515


namespace monotonically_increasing_minimum_value_of_f_on_interval_l146_146253

-- Define the function f
def f (x a : ℝ) := (1/2)*x^2 - 2*a*Real.log x + (a-2)*x

-- Define condition that a is a real number
def a_real (a : ℝ) := True

-- Problem 1: Monotonically increasing function
theorem monotonically_increasing (a : ℝ) (h : a_real a) : 
  (∀ x ∈ Ioi 2, 0 ≤ (x - 2)*(x + a)) ↔ a ≥ -2 := sorry

-- Problem 2: Minimum value of f on [2, e]
def g (a : ℝ) : ℝ :=
  if a ≥ -2 then -2*a*Real.log 2 + 2*a - 2
  else if -Real.exp 1 < a ∧ a < -2 then -1/2*a^2 - 2*a*Real.log (-a) + 2*a
  else (1/2)*Real.exp 2 - 2*a + a*Real.exp 1

theorem minimum_value_of_f_on_interval [Real.ordered_ring ℝ] : ∀ a ∈ Icc (-Real.exp 1) (2 : ℝ), 
  f (argmin f [2,Real.exp 1]) = g a := sorry

end monotonically_increasing_minimum_value_of_f_on_interval_l146_146253


namespace find_angle_E_l146_146848

-- Defining the trapezoid properties and angles
variables (EF GH : ℝ) (E H G F : ℝ)
variables (trapezoid_EFGH : Prop) (parallel_EF_GH : Prop)
variables (angle_E_eq_3H : Prop) (angle_G_eq_4F : Prop)

-- Conditions
def trapezoid_EFGH : Prop := ∃ E F G H EF GH, EF ≠ GH
def parallel_EF_GH : Prop := EF ∥ GH
def angle_E_eq_3H : Prop := E = 3 * H
def angle_G_eq_4F : Prop := G = 4 * F

-- Theorem statement
theorem find_angle_E (H_value : ℝ) (H_property : H = 45) :
  E = 135 :=
  by
  -- Assume necessary properties from the problem statements
  assume trapezoid_EFGH
  assume parallel_EF_GH : EF ∥ GH
  assume angle_E_eq_3H : E = 3 * H
  have H_value : H = 45 := sorry
  have angle_E_value : E = 135 := sorry
  exact angle_E_value

end find_angle_E_l146_146848


namespace power_division_l146_146077

theorem power_division : (8^15 / 64^3) = 8^9 :=
by
  -- Given that 64 = 8^2
  have h1 : 64 = 8^2 := rfl
  -- Use this to show the final result
  sorry

end power_division_l146_146077


namespace wilson_theorem_l146_146880

theorem wilson_theorem (p : ℕ) (hp : p > 1) : 
  prime p ↔ fact (p - 1) ≡ -1 [MOD p] := 
sorry

end wilson_theorem_l146_146880


namespace specific_natural_numbers_expr_l146_146183

theorem specific_natural_numbers_expr (a b c : ℕ) 
  (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1) : 
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ (n = (a + b) / c + (b + c) / a + (c + a) / b) :=
by sorry

end specific_natural_numbers_expr_l146_146183


namespace apples_target_l146_146859

theorem apples_target (initial_each : ℕ) (additional_rate : ℚ) (remaining_each : ℕ) :
  initial_each = 400 ∧ additional_rate = 3 / 4 ∧ remaining_each = 600 →
  2 * (initial_each + additional_rate * initial_each + remaining_each) = 2600 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2, h3]
  calc
    2 * (initial_each + additional_rate * initial_each + remaining_each)
    = 2 * (400 + (3 / 4) * 400 + 600) : by simp [h1, h2, h3]
    ... = 2 * (400 + 300 + 600) : by norm_num
    ... = 2 * 1300 : by norm_num
    ... = 2600 : by norm_num

end apples_target_l146_146859


namespace conjugate_of_z_l146_146216

theorem conjugate_of_z (z : ℂ) (hz : z = -1 + 3 * Complex.i) : Complex.conj z = -1 - 3 * Complex.i :=
by 
  rw [hz]
  sorry

end conjugate_of_z_l146_146216


namespace wedge_volume_is_131_l146_146584

noncomputable theory

def calculate_cylinder_wedge_volume (h r : ℝ) (sector_angle : ℝ) : ℝ :=
  let full_volume := π * r^2 * h
  let wedge_volume := (sector_angle / 360) * full_volume
  wedge_volume

theorem wedge_volume_is_131 :
  calculate_cylinder_wedge_volume 10 5 60 = 131 :=
by
  sorry

end wedge_volume_is_131_l146_146584


namespace solve_for_y_l146_146811

theorem solve_for_y (y : ℝ) (h : 9^y = 3^16) : y = 8 := 
by 
  sorry

end solve_for_y_l146_146811


namespace find_number_l146_146810

theorem find_number (x : ℝ) (h1 : 0.35 * x = 0.2 * 700) (h2 : 0.2 * 700 = 140) (h3 : 0.35 * x = 140) : x = 400 :=
by sorry

end find_number_l146_146810


namespace boat_travel_time_l146_146105

theorem boat_travel_time 
  (area_lake : ℝ) 
  (speed_boat : ℝ) 
  (time_length : ℝ)
  (length_lake : ℝ)
  (width_lake : ℝ) 
  (h_len_eq_wid : length_lake = width_lake) 
  (h_area: area_lake = length_lake * width_lake) 
  (h_speed: speed_boat = 10) 
  (h_time_length: time_length = 2) 
  (h_area_val: area_lake = 100) : 
  (time_length * speed_boat ≃ len_lake) (time_to_travel_width : ℝ) : 
  time_to_travel_width = 60 := by
    sorry

end boat_travel_time_l146_146105


namespace either_ZL_or_BB_signed_l146_146129

noncomputable def prob_either_signed (total_singers chosen_singers : ℕ) (prob_equal: Prop) : ℚ :=
  if prob_equal then
    1 - (\binom (total_singers - 2) (chosen_singers - 2)) / (\binom total_singers chosen_singers)
  else 0

theorem either_ZL_or_BB_signed {total_singers chosen_singers : ℕ} (h1 : total_singers = 5) (h2 : chosen_singers = 3)
  (h3 : ∀ (k : ℕ), k ∈ (finset.range 1 (total_singers + 1)).to_list → (k, h1) → k ≤ chosen_singers)
  : prob_either_signed total_singers chosen_singers (h3 total_singers) = 9 / 10 := 
sorry

end either_ZL_or_BB_signed_l146_146129


namespace inverse_proposition_l146_146938

-- Define the variables m, n, and a^2
variables (m n : ℝ) (a : ℝ)

-- State the proof problem
theorem inverse_proposition
  (h1 : m > n)
: m * a^2 > n * a^2 :=
sorry

end inverse_proposition_l146_146938


namespace line_intersects_circle_l146_146043

-- Definitions of the line and circle
def line (m : ℝ) : (ℝ × ℝ) → Prop := λ p, m * p.1 - p.2 + 1 = 0
def circle : (ℝ × ℝ) → Prop := λ p, p.1^2 + (p.2 - 1)^2 = 5

-- Prove that the line intersects the circle for some m
theorem line_intersects_circle (m : ℝ) : ∃ p : ℝ × ℝ, line m p ∧ circle p := by
  sorry

end line_intersects_circle_l146_146043


namespace route_time_comparison_l146_146441

theorem route_time_comparison :
  let
    distance_A := 8
    speed_A := 40

    distance_B1 := 5
    speed_B1 := 45

    distance_construction := 1
    speed_construction := 15

    distance_scenic := 1
    speed_scenic := 30

    time_A := (distance_A : ℝ) / speed_A * 60

    time_B1 := (distance_B1 : ℝ) / speed_B1 * 60
    time_construction := (distance_construction : ℝ) / speed_construction * 60
    time_scenic := (distance_scenic : ℝ) / speed_scenic * 60

    time_B := time_B1 + time_construction + time_scenic

    delta_t := time_B - time_A

  in delta_t = 0.67 :=
by
  sorry

end route_time_comparison_l146_146441


namespace fifth_graders_more_than_eighth_graders_l146_146026

theorem fifth_graders_more_than_eighth_graders 
  (cost : ℕ) 
  (h_cost : cost > 0) 
  (h_div_234 : 234 % cost = 0) 
  (h_div_312 : 312 % cost = 0) 
  (h_40_fifth_graders : 40 > 0) : 
  (312 / cost) - (234 / cost) = 6 := 
by 
  sorry

end fifth_graders_more_than_eighth_graders_l146_146026


namespace find_number_l146_146952

theorem find_number (n : ℕ) (h₁ : ∀ x : ℕ, 21 + 7 * x = n ↔ 3 + x = 47):
  n = 329 :=
by
  -- Proof will go here
  sorry

end find_number_l146_146952


namespace dot_product_a_b_l146_146748

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem dot_product_a_b : (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end dot_product_a_b_l146_146748


namespace no_pos_int_sol_l146_146454

theorem no_pos_int_sol (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ¬ ∃ (k : ℕ), (15 * a + b) * (a + 15 * b) = 3^k := 
sorry

end no_pos_int_sol_l146_146454


namespace nine_sided_polygon_odd_span_diagonals_count_l146_146800

theorem nine_sided_polygon_odd_span_diagonals_count :
    let n := 9 in 
    let total_vertices := 9 in 
    let total_diagonals := (total_vertices * (total_vertices - 3)) / 2 in 
    ∃ d, d = (total_vertices * 4) / 2 ∧ d = 18 :=
by
  sorry

end nine_sided_polygon_odd_span_diagonals_count_l146_146800


namespace carl_weight_l146_146144

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end carl_weight_l146_146144


namespace number_of_cars_per_package_l146_146066

variable (x : ℕ)

-- Conditions
def total_cars (x : ℕ) := 10 * x
def cars_given_away (x : ℕ) := (2 * (1/5 : ℝ) * total_cars x).toNumber
def cars_left (x : ℕ) := total_cars x - cars_given_away x

-- Theorem to prove
theorem number_of_cars_per_package (h : cars_left x = 30) : x = 5 := by
  sorry

end number_of_cars_per_package_l146_146066


namespace percentage_of_other_investment_l146_146619

theorem percentage_of_other_investment (investment total_interest interest_5 interest_other percentage_other : ℝ) 
  (h1 : investment = 18000)
  (h2 : interest_5 = 6000 * 0.05)
  (h3 : total_interest = 660)
  (h4 : percentage_other / 100 * (investment - 6000) = 360) : 
  percentage_other = 3 :=
by
  sorry

end percentage_of_other_investment_l146_146619


namespace area_of_enclosed_figure_l146_146479

def curve_fn (x : ℝ) : ℝ := x ^ 2 + 1
def line_fn (x : ℝ) : ℝ := -x + 3

theorem area_of_enclosed_figure : 
  let area_curve := ∫ x in 0..1, curve_fn x,
      area_triangle := (1 / 2) * (2 * 2)
  in area_curve + area_triangle = 10 / 3 :=
by
  sorry

end area_of_enclosed_figure_l146_146479


namespace sum_p_s_at_12_l146_146421

-- Definitions based on the conditions described
def S := {s : Fin 6 → Fin 2 // True}
def p_s (s : S) : ℕ → ℕ := 
  fun (x : ℕ) => if x <= 5 then s.val ⟨x, by linarith⟩ else 0

-- Mathematically equivalent proof problem
theorem sum_p_s_at_12 : ∑ s in S, p_s s 12 = 32 := sorry

end sum_p_s_at_12_l146_146421


namespace find_values_l146_146075

theorem find_values (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : a = 2 * b + 5) (h3 : Nat.Prime (a + 7 * b)) : (a = 9 ∧ b = 2) ∨ (a = 17 ∧ b = 6) :=
sorry

end find_values_l146_146075


namespace system_of_equations_solution_l146_146919

theorem system_of_equations_solution (x y : ℝ) :
  x - y = 1 ∧ 2 * x + y = 5 -> x = 2 ∧ y = 1 :=
by
  intro h
  cases h with h1 h2
  sorry

end system_of_equations_solution_l146_146919


namespace lisa_likes_only_last_digit_zero_l146_146701

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def is_divisible_by_2 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8

def is_divisible_by_5_and_2 (n : ℕ) : Prop :=
  is_divisible_by_5 n ∧ is_divisible_by_2 n

theorem lisa_likes_only_last_digit_zero : ∀ n, is_divisible_by_5_and_2 n → n % 10 = 0 :=
by
  sorry

end lisa_likes_only_last_digit_zero_l146_146701


namespace thirty_th_number_in_sequence_l146_146855

theorem thirty_th_number_in_sequence : 
  ∀ (a d n : ℕ), a = 2 → d = 2 → n = 30 → a + (n - 1) * d = 60 :=
by intros a d n ha hd hn
   rw [ha, hd, hn]
   exact Nat.add_sub_assoc (by norm_num : 1 ≤ 30) _ _
   norm_num
   rfl

end thirty_th_number_in_sequence_l146_146855


namespace decreasing_interval_of_composite_function_l146_146242

theorem decreasing_interval_of_composite_function :
  (∀ x, has_inverse (λ y, y = 1 / 3^x) (f x)) →
  (∀ x, f (2 * x - x^2) = -log 3 (2 * x - x^2)) →
  ∃ I, I = set.Ioc 0 1 ∧ ∀ x ∈ I, ∃ y ∈ I, f (2 * x - x^2) ≥ f (2 * y - y^2) :=
sorry

end decreasing_interval_of_composite_function_l146_146242


namespace find_RS_l146_146825

-- Define the setup for the problem
def triangle_PQR (P Q R : Type) :=
  ∃ (PQ QR PR : ℝ), 
    (PQ = 4) ∧ 
    (QR = 6) ∧ 
    (PR = Real.sqrt (52 + 24 * Real.sqrt 3)) ∧ 
    (angle PQR = 150)

def perpendicular_to_PQ_at_P (P Q : Type) := 
  true

def perpendicular_to_QR_at_R (R Q : Type) := 
  true

-- Define the assertion for RS
def length_RS (RS : ℝ) : Prop :=
  RS = 24 / Real.sqrt (52 + 24 * Real.sqrt 3)

-- Statement to be proved
theorem find_RS 
  (P Q R S : Type)
  (h1 : triangle_PQR P Q R)
  (h2 : perpendicular_to_PQ_at_P P Q)
  (h3 : perpendicular_to_QR_at_R R Q) :
  ∃ (RS : ℝ), length_RS RS :=
by
  apply Exists.intro
  exact 24 / Real.sqrt (52 + 24 * Real.sqrt 3)
  sorry

end find_RS_l146_146825


namespace minimum_obtuse_triangles_l146_146716

def is_regular_polygon (P : Finset (Fin n → ℝ × ℝ)) : Prop := sorry -- Definition of a regular polygon

def is_obtuse_triangle (T : Finset (ℝ × ℝ)) : Prop := sorry -- Definition of an obtuse triangle

theorem minimum_obtuse_triangles (n : ℕ) (h₁ : n ≥ 5) (P : Finset (Fin n → ℝ × ℝ)) 
  (h₂ : is_regular_polygon P) : 
  ∃ m : ℕ, (∀ T : Finset (ℝ × ℝ), T ∈ (triangulation P) → is_obtuse_triangle T) → m = n :=
sorry

end minimum_obtuse_triangles_l146_146716


namespace probability_real_roots_probability_point_in_region_l146_146524

/-
Given:
1. There are four balls numbered 1, 2, 3, and 4.
2. A ball is drawn from the bag and numbered a.
3. Another ball is drawn from the remaining three and numbered b.

Prove that the probability of the equation x^2 + 2ax + b^2 = 0 having real roots is 1/2.
-/
theorem probability_real_roots :
  let balls := {1, 2, 3, 4}
  let outcomes := ({(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), 
                    (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)})
  let valid_outcomes := ({(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)})
  (valid_outcomes.card : ℕ) / (outcomes.card : ℕ) = 1 / 2 := sorry

/-
Given:
1. There are four balls numbered 1, 2, 3, and 4.
2. A ball is drawn from the bag and numbered m, then replaced in the bag.
3. Another ball is drawn and numbered n.
4. Define point P as (m, n).

Prove the probability that point P lies within the region defined by:
x - y ≥ 0 and x + y - 5 < 0 is 1/4.
-/
theorem probability_point_in_region :
  let balls := {1, 2, 3, 4}
  let outcomes := {(x, y) | x ∈ balls ∧ y ∈ balls}
  let region := {(x, y) | x - y ≥ 0 ∧ x + y < 5}
  (region.card : ℕ) / (outcomes.card : ℕ) = 1 / 4 := sorry

end probability_real_roots_probability_point_in_region_l146_146524


namespace t_minus_d_l146_146530

-- Define amounts paid by Tom, Dorothy, and Sammy
def tom_paid : ℕ := 140
def dorothy_paid : ℕ := 90
def sammy_paid : ℕ := 220

-- Define the total amount and required equal share
def total_paid : ℕ := tom_paid + dorothy_paid + sammy_paid
def equal_share : ℕ := total_paid / 3

-- Define the amounts t and d where Tom and Dorothy balance the costs by paying Sammy
def t : ℤ := equal_share - tom_paid -- Amount Tom gave to Sammy
def d : ℤ := equal_share - dorothy_paid -- Amount Dorothy gave to Sammy

-- Prove that t - d = -50
theorem t_minus_d : t - d = -50 := by
  sorry

end t_minus_d_l146_146530


namespace benny_turnips_l146_146438

-- Definitions and conditions
def melanie_turnips : ℕ := 139
def total_turnips : ℕ := 252

-- Question to prove
theorem benny_turnips : ∃ b : ℕ, b = total_turnips - melanie_turnips ∧ b = 113 :=
by {
    sorry
}

end benny_turnips_l146_146438


namespace a_can_complete_work_in_18_days_l146_146089

noncomputable def efficiency_a_b (a b : Type) (eff_a : ℕ) (eff_b : ℕ) :=
  eff_a = 2 * eff_b

noncomputable def work_done_by_cd (c d : Type) (time_c : ℕ) (time_d : ℕ) : ℚ :=
  1 / time_c + 1 / time_d

noncomputable def work_done_per_day (a b c d : Type) (work_ab work_cd : ℚ) :=
  work_ab = work_cd

noncomputable def time_to_complete_work (a : Type) (work_per_day_a : ℚ) : ℕ :=
  1 / work_per_day_a

theorem a_can_complete_work_in_18_days :
  ∀ (a b c d : Type) (time_c time_d : ℕ),
  (efficiency_a_b a b 2 1) →
  (work_done_per_day a b c d (3 * (1 / 36)) (work_done_by_cd c d time_c time_d)) →
  time_c = 20 → time_d = 30 →
  time_to_complete_work a (2 * (1 / 36)) = 18 :=
by
  intros a b c d time_c time_d h_eff h_work h_time_c h_time_d,
  sorry

end a_can_complete_work_in_18_days_l146_146089


namespace sum_of_8n_plus_4_consecutive_integers_not_perfect_square_l146_146012

theorem sum_of_8n_plus_4_consecutive_integers_not_perfect_square (n : ℕ) (hn : n > 0) :
  ∀ a : ℕ, let S := (8*n + 4)*a + (8*n + 3)*(8*n + 4)/2 in ¬(∃ k : ℕ, k^2 = S) := by
  sorry

end sum_of_8n_plus_4_consecutive_integers_not_perfect_square_l146_146012


namespace angle_between_vectors_l146_146237

open Real

def vector_angle_problem :
  ℝ × ℝ × (ℝ × ℝ) := 
  let a : ℝ × ℝ := (1, -sqrt 3)
  let b : ℝ × ℝ := (cos 𝜃, sin 𝜃)
  let mag_b := 1
  let mag_a_plus_2b := 2
  (1, -sqrt 3, (b.fst, b.snd))

theorem angle_between_vectors (a b : ℝ × ℝ) (ha : a = (1, -sqrt 3))
  (hb : ‖b‖ = 1) (hab_plus_2b : ‖(λ a b, (a.1 + 2 * b.1, a.2 + 2 * b.2)) a b‖ = 2) :
  let θ := 2 * π / 3 in
  θ = arccos ((a.1 * b.1 + a.2 * b.2) / (sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2)))) :=
by
  sorry

end angle_between_vectors_l146_146237


namespace power_calculation_l146_146304

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146304


namespace find_line_equation_l146_146725

theorem find_line_equation (a b : ℝ) :
  (2 * a + 3 * b = 0 ∧ a * b < 0) ↔ (3 * a - 2 * b = 0 ∨ a - b + 1 = 0) :=
by
  sorry

end find_line_equation_l146_146725


namespace correct_answers_for_f_l146_146241

def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem correct_answers_for_f (φ : ℝ) (h0 : 0 < φ ∧ φ < π) (h_sym : ∀ x, f x φ = f (4 * π / 3 - x) φ) :
  φ = 2 * π / 3 ∧ 
  (∀ x, 0 < x ∧ x < 5 * π / 12 → f x φ < f (x - ε) φ) ∧ 
  (∃ x, f'(x) = 2 * Real.cos (2 * x + φ) ∧ 2 * Real.cos (2 * x + φ) = -1 ∧ 
    (∀ x, f'(x) = -tan(1) * x + sqrt(3) / 2))
:= 
by sorry

end correct_answers_for_f_l146_146241


namespace votes_for_winning_candidate_l146_146094

-- Definitions
variables (V : ℕ) 
variables (winner_votes : ℕ) 
variables (loser_votes : ℕ)

-- Conditions / Assumptions
def condition1 := winner_votes = 62 * V / 100
def condition2 := winner_votes - loser_votes = 336
def condition3 := winner_votes + loser_votes = V

-- Statement to Prove
theorem votes_for_winning_candidate : 
  winner_votes = 868 :=
begin
  sorry
end

end votes_for_winning_candidate_l146_146094


namespace degrees_for_salaries_l146_146110

def transportation_percent : ℕ := 15
def research_development_percent : ℕ := 9
def utilities_percent : ℕ := 5
def equipment_percent : ℕ := 4
def supplies_percent : ℕ := 2
def total_percent : ℕ := 100
def total_degrees : ℕ := 360

theorem degrees_for_salaries :
  total_degrees * (total_percent - (transportation_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent)) / total_percent = 234 := 
by
  sorry

end degrees_for_salaries_l146_146110


namespace fractional_multiplication_l146_146971

theorem fractional_multiplication :
  (\left(\frac{8}{9}\right)^2 \cdot \left(\frac{1}{3}\right)^2 \cdot \left(\frac{1}{4}\right)^2 = \frac{4}{729}) :=
by sorry

end fractional_multiplication_l146_146971


namespace sequence_sum_zero_l146_146159

theorem sequence_sum_zero : 
  let sequence := list.range 2008 |>
                  list.map (λ n, if n % 4 == 0 then n + 1 else if n % 4 == 1 then -(n + 1) else if n % 4 == 2 then -(n + 1) else n + 1)
  in sequence.sum = 0 :=
by
  sorry

end sequence_sum_zero_l146_146159


namespace second_integer_is_64_l146_146053

theorem second_integer_is_64
  (n : ℤ)
  (h1 : (n - 2) + (n + 2) = 128) :
  n = 64 := 
  sorry

end second_integer_is_64_l146_146053


namespace fraction_to_decimal_l146_146721

theorem fraction_to_decimal : (53 : ℚ) / (4 * 5^7) = 1325 / 10^7 := sorry

end fraction_to_decimal_l146_146721


namespace directrix_of_parabola_l146_146189

theorem directrix_of_parabola (a b c : ℝ) (h_eqn : ∀ x, b = -4 * x^2 + c) : 
  b = 5 → c = 0 → (∃ y, y = 81 / 16) :=
by
  sorry

end directrix_of_parabola_l146_146189


namespace grass_coverage_day_l146_146577

theorem grass_coverage_day (coverage : ℕ → ℚ) : 
  (∀ n : ℕ, coverage (n + 1) = 2 * coverage n) → 
  coverage 24 = 1 → 
  coverage 21 = 1 / 8 := 
by
  sorry

end grass_coverage_day_l146_146577


namespace classify_numbers_l146_146101

open Real

theorem classify_numbers :
  let numbers := [2.5, -0.5252252225, -10^2, -5, 0, 1/3, abs (-4), -3.6, pi / 2] in
  (let neg := {-0.5252252225, -100, -5, -3.6} in
  (let non_neg_int := {0, 4} in
  (let fract := {2.5, 1/3, -3.6} in
  (let irr := {-0.5252252225, pi/2} in
  (∀ x ∈ neg, x ∈ numbers ∧ x < 0) ∧
  (∀ x ∈ non_neg_int, x ∈ numbers ∧ (∃ n : ℕ, x = n)) ∧
  (∀ x ∈ fract, x ∈ numbers ∧ ∃ a b : ℤ, x = a / b ∧ b ≠ 0) ∧
  (∀ x ∈ irr, x ∈ numbers ∧ ¬∃ a b : ℤ, x = a / b ∧ b ≠ 0)))))) :=
by
  sorry

end classify_numbers_l146_146101


namespace power_of_3_l146_146348

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146348


namespace age_ratio_l146_146047

theorem age_ratio (R D : ℕ) (h1 : D = 15) (h2 : R + 6 = 26) : R / D = 4 / 3 := by
  sorry

end age_ratio_l146_146047


namespace problem_solution_l146_146798

variables {R : Type*} [real R]
variables (a b : vector_space R) (k : R)

-- Conditions: a and b are not collinear, and c = k * a + b, d = a - b
noncomputable def not_collinear (a b : vector_space R) : Prop := ¬ collinear a b
def c (a b : vector_space R) (k : R) : vector_space R := k • a + b
def d (a b : vector_space R) : vector_space R := a - b
def parallel (x y : vector_space R) : Prop := ∃ λ : R, x = λ • y

theorem problem_solution (h1 : not_collinear a b)
                         (h2 : parallel (c a b k) (d a b)) :
  k = -1 ∧ (c a b k) = - (d a b) := 
sorry

end problem_solution_l146_146798


namespace cevian_concurrency_l146_146562

-- Definitions for the acute triangle and the angles
structure AcuteTriangle (α β γ : ℝ) :=
  (A B C : ℝ)
  (acute_α : α > 0 ∧ α < π / 2)
  (acute_β : β > 0 ∧ β < π / 2)
  (acute_γ : γ > 0 ∧ γ < π / 2)
  (triangle_sum : α + β + γ = π)

-- Definition for the concurrency of cevians
def cevians_concurrent (α β γ : ℝ) (t : AcuteTriangle α β γ) :=
  ∀ (A₁ B₁ C₁ : ℝ), sorry -- placeholder

-- The main theorem with the proof of concurrency
theorem cevian_concurrency (α β γ : ℝ) (t : AcuteTriangle α β γ) :
  ∃ (A₁ B₁ C₁ : ℝ), cevians_concurrent α β γ t :=
  sorry -- proof to be provided


end cevian_concurrency_l146_146562


namespace integral_cos_approximation_l146_146637

theorem integral_cos_approximation :
  |∫ x in 0..0.1, Real.cos (100 * x^2) - 0.09| ≤ 0.001 := 
sorry

end integral_cos_approximation_l146_146637


namespace equivar_proof_l146_146759

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end equivar_proof_l146_146759


namespace problem_result_l146_146712

theorem problem_result : 
  let a := (5 / 6) * 180
  let b := 0.70 * 250
  let c := 0.35 * 480
  (a - b) / c = -25 / 168 := 
by
  let a := (5 / 6) * 180
  let b := 0.70 * 250
  let c := 0.35 * 480
  have h1 : a = 150 := by norm_num
  have h2 : b = 175 := by norm_num
  have h3 : c = 168 := by norm_num
  rw [h1, h2, h3]
  norm_num
  sorry

end problem_result_l146_146712


namespace find_m_l146_146257

theorem find_m (m : ℝ) :
  let f (x : ℝ) := x^3 - 3 * x + m in
  let max_f := f (-1) in
  let min_f := f (-3) in
  (max_f + min_f = -1) → m = 7.5 :=
by
  intros f max_f min_f h
  sorry

end find_m_l146_146257


namespace wilsons_theorem_l146_146882

theorem wilsons_theorem (p : ℕ) (hp1 : p > 1) : 
  prime p ↔ factorial (p - 1) ≡ -1 [MOD p] :=
by
  sorry

end wilsons_theorem_l146_146882


namespace students_per_group_l146_146954

theorem students_per_group (total_students not_picked_groups groups : ℕ) (h₁ : total_students = 65) (h₂ : not_picked_groups = 17) (h₃ : groups = 8) :
  (total_students - not_picked_groups) / groups = 6 := by
  sorry

end students_per_group_l146_146954


namespace num_O_atoms_is_seven_l146_146112

def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def molecular_weight_O : ℝ := 16.00

def num_C_atoms : ℕ := 6
def num_H_atoms : ℕ := 8
def total_molecular_weight : ℝ := 192

def mass_of_C : ℝ := num_C_atoms * molecular_weight_C
def mass_of_H : ℝ := num_H_atoms * molecular_weight_H
def total_mass_C_H : ℝ := mass_of_C + mass_of_H
def mass_of_O : ℝ := total_molecular_weight - total_mass_C_H

def num_O_atoms : ℝ := mass_of_O / molecular_weight_O

theorem num_O_atoms_is_seven :
  num_O_atoms ≈ 7 := by sorry

end num_O_atoms_is_seven_l146_146112


namespace max_profit_at_9_l146_146581

-- Define the sales revenue function R(x)
def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    10.8 - x^2 / 30
  else if x > 10 then
    108 / x - 1000 / (3 * x^2)
  else
    0  -- defining R(x) to be 0 for invalid inputs

-- Define the profit function W(x)
def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    8.1 * x - x^3 / 30 - 10
  else if x > 10 then
    98 - 1000 / (3 * x) - 2.7 * x
  else
    0  -- defining W(x) to be 0 for invalid inputs

-- Prove that W(x) is maximized at x = 9
theorem max_profit_at_9 : W 9 = 38.6 :=
  sorry

end max_profit_at_9_l146_146581


namespace work_days_remaining_l146_146590

theorem work_days_remaining (n1 d1 n2 D : ℕ) (h1 : n1 = 15) (h2 : d1 = 8) (h3 : n2 = n1 - 3) (h4 : 15 * 8 = n2 * D) : D = 10 :=
by
  -- Using the given condition
  have h5 : 15 * 8 = 120 := rfl
  have h6 : n2 = 12 := by rw [h1, h3]; exact rfl
  have h7 : 12 * D = 120 := by
    rw [h4, ←h6] at h5
    exact h5.symm
  have h8 : D = 10 := by
    calc
      D = 120 / 12 : (nat.mul_right_inj (by norm_num)).mp (by rw [nat.mul_comm, h7])
      ... = 10 : nat.div_self (by norm_num)
  exact h8

-- Marking the end of the theorem with a placeholder
sorry

end work_days_remaining_l146_146590


namespace find_number_to_be_multiplied_l146_146088

def correct_multiplier := 43
def incorrect_multiplier := 34
def difference := 1224

theorem find_number_to_be_multiplied (x : ℕ) : correct_multiplier * x - incorrect_multiplier * x = difference → x = 136 :=
by
  sorry

end find_number_to_be_multiplied_l146_146088


namespace smallest_value_a1_l146_146877

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n-1) - 2 * n

theorem smallest_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n) (h2 : seq a) : 
  a 1 ≥ 13 / 18 :=
sorry

end smallest_value_a1_l146_146877


namespace number_of_boys_l146_146832

variables (total_girls total_teachers total_people : ℕ)
variables (total_girls_eq : total_girls = 315) (total_teachers_eq : total_teachers = 772) (total_people_eq : total_people = 1396)

theorem number_of_boys (total_boys : ℕ) : total_boys = total_people - total_girls - total_teachers :=
by sorry

end number_of_boys_l146_146832


namespace colten_chickens_l146_146001

variable (C Q S : ℕ)

-- Conditions
def condition1 : Prop := Q + S + C = 383
def condition2 : Prop := Q = 2 * S + 25
def condition3 : Prop := S = 3 * C - 4

-- Theorem to prove
theorem colten_chickens : condition1 C Q S ∧ condition2 C Q S ∧ condition3 C Q S → C = 37 := by
  sorry

end colten_chickens_l146_146001


namespace Lucia_birthday_2029_l146_146436

/-- Definitions -/
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Inhabited

open DayOfWeek

def isLeapYear (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def nextDay (day : DayOfWeek) (is_leap : Bool) : DayOfWeek :=
  match day, is_leap with
  | Sunday, false => Monday
  | Monday, false => Tuesday
  | Tuesday, false => Wednesday
  | Wednesday, false => Thursday
  | Thursday, false => Friday
  | Friday, false => Saturday
  | Saturday, false => Sunday
  | Sunday, true  => Tuesday
  | Monday, true  => Wednesday
  | Tuesday, true  => Thursday
  | Wednesday, true  => Friday
  | Thursday, true  => Saturday
  | Friday, true  => Sunday
  | Saturday, true  => Monday

def iterDays (start : DayOfWeek) (start_year : ℕ) (end_year : ℕ) : DayOfWeek :=
  let rec helper (day : DayOfWeek) (year : ℕ) :=
    if year = end_year then day
    else helper (nextDay day (isLeapYear year)) (year + 1)
  helper start start_year

/-- Problem Statement -/
theorem Lucia_birthday_2029 :
  iterDays Friday 2020 2029 = Tuesday :=
begin
  sorry
end

end Lucia_birthday_2029_l146_146436


namespace tangent_line_ln_l146_146190

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :=
  let y' := deriv f x₀ in
  (λ x, y₀ + y' * (x - x₀))

theorem tangent_line_ln :
  tangent_line_equation (λ x, 2 * Real.log (x + 1)) 0 0 = λ x, 2 * x := 
by
  sorry

end tangent_line_ln_l146_146190


namespace power_of_three_l146_146275

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146275


namespace stable_configuration_count_l146_146734

theorem stable_configuration_count : 
  let stable (n : ℕ) (a : ℕ → ℕ) := ∀ i : ℕ, a i * a (i + 1) * a (i + 2) = n
  let admissible (n : ℕ) := 3 ≤ n ∧ n ≤ 2020 ∧ ∃ (a : ℕ → ℕ), stable n a
  count_3_2020 (P : ℕ → Prop) := ∑ k in (Finset.range 2018).filter (λ n, P (n + 3)), 1
    = 7 + 673
in count_3_2020 admissible = 680 := sorry

end stable_configuration_count_l146_146734


namespace power_calculation_l146_146309

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146309


namespace volume_piece_containing_vertex_A_l146_146611

theorem volume_piece_containing_vertex_A : 
  let unit_cube := (1 : ℝ) in
  let base_area := (1 / 2) * (1 / 2) / 2 in
  let height := 1 in
  let volume_piece := (1 / 3) * base_area * height in
  volume_piece = (1 / 24) :=
by
  sorry

end volume_piece_containing_vertex_A_l146_146611


namespace isosceles_triangle_l146_146824

-- Define the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Assume the given condition
axiom triangle_condition : a * Real.cos B = b * Real.cos A

-- Define the statement to be proven
theorem isosceles_triangle (h : triangle_condition) : (A = B → a = b) :=
sorry

end isosceles_triangle_l146_146824


namespace original_number_sweets_correct_l146_146569

-- Definition of the problem conditions
def num_children : ℕ := 48
def sweets_per_child : ℕ := 4
def fraction_remaining : ℚ := 1 / 3

-- Definition of the total number of sweets taken by children
def total_sweets_taken : ℕ := num_children * sweets_per_child

-- Original number of sweets in the pack
def original_number_sweets (total_sweets_taken : ℕ) (fraction_remaining : ℚ) : ℕ :=
  (total_sweets_taken * (1 / (1 - fraction_remaining))).to_nat

-- Assert the result
theorem original_number_sweets_correct :
  original_number_sweets total_sweets_taken fraction_remaining = 288 :=
by sorry

end original_number_sweets_correct_l146_146569


namespace bacteria_doubling_time_l146_146482

theorem bacteria_doubling_time
  (initial_bacteria : ℕ) 
  (final_bacteria : ℕ) 
  (doubling_time : ℕ) 
  (doubling_factor : ℕ) :
  initial_bacteria = 200 →
  final_bacteria = 102,400 →
  doubling_time = 3 →
  doubling_factor = 2 →
  ∃ time_required : ℕ, time_required = 27 :=
by
  intros h_initial h_final h_doubling_time h_doubling_factor
  sorry

end bacteria_doubling_time_l146_146482


namespace median_inequality_l146_146911

variable {a b c m : ℝ}

theorem median_inequality (htriangle : a + b > c ∧ a + c > b ∧ b + c > a) (hmedian : m = (1/2) * √(2*a^2 + 2*b^2 - c^2)) : 
  m > (a + b - c) / 2 := 
sorry

end median_inequality_l146_146911


namespace common_point_of_circles_l146_146864

open EuclideanGeometry

theorem common_point_of_circles 
  (A B C P D E F : Point) 
  (omega : Circle)
  (h_circle : OnCircle P (Circumcircle A B C))
  (h_reflections : Reflections P D (Amidline A) E (Bmidline B) F (Cmidline C))
  (h_omega : OmegaCircumcircleomega omega (PerpendicularBisectorsTriangle A D B E C F)) :
  ∃ X : Point, OnCircle X (Circumcircle A D P) ∧ OnCircle X (Circumcircle B E P) ∧ OnCircle X (Circumcircle C F P) ∧ OnCircle X omega :=
by
  sorry

end common_point_of_circles_l146_146864


namespace lowest_sale_price_is_30_percent_l146_146605

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end lowest_sale_price_is_30_percent_l146_146605


namespace scientific_notation_of_great_wall_l146_146513

theorem scientific_notation_of_great_wall : 
  ∀ n : ℕ, (6700010 : ℝ) = 6.7 * 10^6 :=
by
  sorry

end scientific_notation_of_great_wall_l146_146513


namespace f_of_f_one_fourth_l146_146776

def f (x : ℝ) : ℝ :=
  if 0 < x then log x / log 2 else 5 ^ x

theorem f_of_f_one_fourth : f (f (1 / 4)) = 1 / 25 := by
  sorry

end f_of_f_one_fourth_l146_146776


namespace ceiling_neg_sqrt_36_l146_146719

theorem ceiling_neg_sqrt_36 : ⌈-real.sqrt 36⌉ = -6 := 
by
  sorry

end ceiling_neg_sqrt_36_l146_146719


namespace frustum_volume_fraction_l146_146137

theorem frustum_volume_fraction {V_original V_frustum : ℚ} 
(base_edge : ℚ) (height : ℚ) 
(h1 : base_edge = 24) (h2 : height = 18) 
(h3 : V_original = (1 / 3) * (base_edge ^ 2) * height)
(smaller_base_edge : ℚ) (smaller_height : ℚ) 
(h4 : smaller_height = (1 / 3) * height) (h5 : smaller_base_edge = base_edge / 3) 
(V_smaller : ℚ) (h6 : V_smaller = (1 / 3) * (smaller_base_edge ^ 2) * smaller_height)
(h7 : V_frustum = V_original - V_smaller) :
V_frustum / V_original = 13 / 27 :=
sorry

end frustum_volume_fraction_l146_146137


namespace power_addition_l146_146299

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146299


namespace james_total_toys_l146_146412

-- Definition for the number of toy cars
def numToyCars : ℕ := 20

-- Definition for the number of toy soldiers
def numToySoldiers : ℕ := 2 * numToyCars

-- The total number of toys is the sum of toy cars and toy soldiers
def totalToys : ℕ := numToyCars + numToySoldiers

-- Statement to prove: James buys a total of 60 toys
theorem james_total_toys : totalToys = 60 := by
  -- Insert proof here
  sorry

end james_total_toys_l146_146412


namespace students_in_all_three_activities_l146_146519

def students_in_club := 25
def students_in_chess := 15
def students_in_soccer := 18
def students_in_music := 10
def students_at_least_two_activities := 14
def students_at_least_one_activity := students_in_club

theorem students_in_all_three_activities : 
  ∃ c, ∀ (a b d : ℕ), (a + b + c + d = students_at_least_two_activities) → 
  (students_in_chess - (a + c + d) + students_in_soccer - (a + b + c) + students_in_music - (b + c + d) + a + b + d + c = students_at_least_one_activity) → 
  c = 4 := 
by {
  use 4,
  intros a b d h1 h2,
  sorry
}

end students_in_all_three_activities_l146_146519


namespace binom_9_5_eq_126_l146_146661

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146661


namespace total_earrings_after_one_year_l146_146626

theorem total_earrings_after_one_year :
  let bella_earrings := 10
  let monica_earrings := 10 / 0.25
  let rachel_earrings := monica_earrings / 2
  let initial_total := bella_earrings + monica_earrings + rachel_earrings
  let olivia_earrings_initial := initial_total + 5
  let olivia_earrings_after := olivia_earrings_initial * 1.2
  let total_earrings := bella_earrings + monica_earrings + rachel_earrings + olivia_earrings_after
  total_earrings = 160 :=
by
  sorry

end total_earrings_after_one_year_l146_146626


namespace carl_weight_l146_146143

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end carl_weight_l146_146143


namespace sweets_original_count_l146_146571

theorem sweets_original_count (S : ℕ) (children : ℕ) (sweets_per_child : ℕ) (third_remaining : ℕ) :
  children = 48 → sweets_per_child = 4 → third_remaining = 1 / 3 * S → S = 288 := by
  intros h1 h2 h3
  have h4 : 48 * 4 = 192 := rfl
  have h5 : 2 / 3 * S = 192 := by
    rw [h3, h4]
  have h6 : S = 192 * 3 / 2 := by
    rw [h5]
  have h7 : 192 * 3 / 2 = 288 := rfl
  rw [h7]
  exact rfl

end sweets_original_count_l146_146571


namespace binomial_coefficient_9_5_l146_146669

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146669


namespace max_permutations_l146_146207

open Finset

-- Definitions based on the conditions part
def is_permutation (n : ℕ) (p : List ℕ) : Prop :=
  p.sort (· ≤ ·) = List.range (n + 1)

def between_condition (n : ℕ) (S : Finset (List ℕ)) : Prop :=
  ∀ (p ∈ S) (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c → a ∈ p → b ∈ p → c ∈ p →
  ¬((a < b ∧ b < c) ∨ (c < b ∧ b < a))

-- Statement of the problem
theorem max_permutations (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ) (S : Finset (List ℕ)), (∀ p ∈ S, is_permutation n p) ∧ between_condition n S ∧ m = 2^(n - 1) :=
sorry

end max_permutations_l146_146207


namespace power_of_3_l146_146353

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146353


namespace csc_315_eq_neg_sqrt_2_l146_146180

theorem csc_315_eq_neg_sqrt_2 :
  (∀ θ, (csc θ) = 1 / sin θ) →
  sin (360 - 45 : ℝ) = sin 315 →
  (∀ θ : ℝ, sin (360 - θ) = - (sin θ)) →
  sin 45 = 1 / Real.sqrt 2 →
  csc 315 = - Real.sqrt 2 :=
by
  intros h1 h2 h3 h4
  sorry

end csc_315_eq_neg_sqrt_2_l146_146180


namespace sum_of_roots_is_18_l146_146892

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_roots_is_18 
  (h_symm: ∀ x: ℝ, f (3 + x) = f (3 - x)) 
  (h_roots: ∃ s: set ℝ, s.finite ∧ s.card = 6 ∧ ∀ x ∈ s, f x = 0) :
  (∑ x in h_roots.some, x) = 18 :=
sorry

end sum_of_roots_is_18_l146_146892


namespace condition_sufficiency_not_necessity_l146_146809

variable {x y : ℝ}

theorem condition_sufficiency_not_necessity (hx : x ≥ 0) (hy : y ≥ 0) :
  (xy > 0 → |x + y| = |x| + |y|) ∧ (|x + y| = |x| + |y| → xy ≥ 0) :=
sorry

end condition_sufficiency_not_necessity_l146_146809


namespace vertical_asymptote_c_values_l146_146210

theorem vertical_asymptote_c_values (c : ℝ) :
  (∃ x : ℝ, (x^2 - x - 6) = 0 ∧ (x^2 - 2*x + c) ≠ 0 ∧ ∀ y : ℝ, ((y ≠ x) → (x ≠ 3) ∧ (x ≠ -2)))
  → (c = -3 ∨ c = -8) :=
by sorry

end vertical_asymptote_c_values_l146_146210


namespace construct_quadrilateral_l146_146704

-- Define all the points T_A, T_B, T_C, T_D and the line BD
variables {A B C D T_A T_B T_C T_D : Type*}
variables [line BD : Type*]
variables (onLine : ∀ {P Q R : Type*}, P ∉ line Q R)

-- Assumptions for the perpendicular feet
axiom TA_perpendicular : ∀ {CD : Type*}, A ∉ line C D → T_A ∈ line C D 
axiom TB_perpendicular : ∀ {DA : Type*}, B ∉ line D A → T_B ∈ line D A 
axiom TC_perpendicular : ∀ {AB : Type*}, C ∉ line A B → T_C ∈ line A B 
axiom TD_perpendicular : ∀ {BC : Type*}, D ∉ line B C → T_D ∈ line B C 

-- The existence of quadrilateral ABCD with the given properties
theorem construct_quadrilateral :
  ∃ (A B C D : Type*), 
    (A ∉ line C D ∧ T_A ∈ line C D) ∧
    (B ∉ line D A ∧ T_B ∈ line D A) ∧
    (C ∉ line A B ∧ T_C ∈ line A B) ∧
    (D ∉ line B C ∧ T_D ∈ line B C) :=
sorry

end construct_quadrilateral_l146_146704


namespace largest_multiple_of_15_less_than_500_l146_146980

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146980


namespace binom_9_5_l146_146674

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146674


namespace power_of_three_l146_146360

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146360


namespace total_worth_of_stock_l146_146603

noncomputable def shop_equation (X : ℝ) : Prop :=
  0.04 * X - 0.02 * X = 400

theorem total_worth_of_stock :
  ∃ (X : ℝ), shop_equation X ∧ X = 20000 :=
by
  use 20000
  have h : shop_equation 20000 := by
    unfold shop_equation
    norm_num
  exact ⟨h, rfl⟩

end total_worth_of_stock_l146_146603


namespace certain_event_at_least_one_genuine_l146_146738

def products : Finset (Fin 12) := sorry
def genuine : Finset (Fin 12) := sorry
def defective : Finset (Fin 12) := sorry
noncomputable def draw3 : Finset (Finset (Fin 12)) := sorry

-- Condition: 12 identical products, 10 genuine, 2 defective
axiom products_condition_1 : products.card = 12
axiom products_condition_2 : genuine.card = 10
axiom products_condition_3 : defective.card = 2
axiom products_condition_4 : ∀ x ∈ genuine, x ∈ products
axiom products_condition_5 : ∀ x ∈ defective, x ∈ products
axiom products_condition_6 : genuine ∩ defective = ∅

-- The statement to be proved: when drawing 3 products randomly, it is certain that at least 1 is genuine.
theorem certain_event_at_least_one_genuine :
  ∀ s ∈ draw3, ∃ x ∈ s, x ∈ genuine :=
sorry

end certain_event_at_least_one_genuine_l146_146738


namespace smaller_circle_radius_l146_146111

theorem smaller_circle_radius (A_1 A_2 : ℝ) (r : ℝ) (h1 : A_1 + A_2 = 16 * real.pi)
    (h2 : 2 * A_1 = 16 * real.pi - A_1) : r = 4 * real.sqrt 3 / 3 :=
by
  let h1_eq : 3 * A_1 = 16 * real.pi := by linarith
  have A_1_val : A_1 = 16 * real.pi / 3 := by linarith
  let Area_eq := A_1_val
  let h1_radius_eq: π * r^2 = A_1 := by sorry
  sorry

end smaller_circle_radius_l146_146111


namespace power_addition_l146_146296

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146296


namespace max_value_of_t_l146_146879

variable (n r t : ℕ)
variable (A : Finset (Finset (Fin n)))
variable (h₁ : n ≤ 2 * r)
variable (h₂ : ∀ s ∈ A, Finset.card s = r)
variable (h₃ : Finset.card A = t)

theorem max_value_of_t : 
  (n < 2 * r → t ≤ Nat.choose n r) ∧ 
  (n = 2 * r → t ≤ Nat.choose n r / 2) :=
by
  sorry

end max_value_of_t_l146_146879


namespace quadratic_eq_roots_l146_146752

theorem quadratic_eq_roots (
  α β : ℝ
  (h_sum : α + β = 5) 
  (h_prod : α * β = 6)
) : 
  ∃ c : ℝ, polynomial.expr α β = polynomial(expr : t (expr α β)) x
begin
  sorry
end

end quadratic_eq_roots_l146_146752


namespace target_apples_l146_146860

-- Definitions based on the conditions
def Kaiden_first_round_apples : ℕ := 400
def Adriel_first_round_apples : ℕ := 400
def fraction_of_first_round : ℚ := 3 / 4
def additional_apples_needed_each : ℕ := 600

-- Prove that the total target number of apples picked is 2600
theorem target_apples : 
  let first_round_each := Kaiden_first_round_apples,
      second_round_each := fraction_of_first_round * first_round_each,
      total_first_second_round_each := first_round_each + second_round_each,
      additional_apples_each := additional_apples_needed_each
  in 
    2 * (total_first_second_round_each + additional_apples_each) = 2600 :=
by
  sorry

end target_apples_l146_146860


namespace intersection_in_fourth_quadrant_l146_146387

theorem intersection_in_fourth_quadrant (m : ℝ) :
  let x := (3 * m + 2) / 4
  let y := (-m - 2) / 8
  (x > 0) ∧ (y < 0) ↔ (m > -2 / 3) :=
by
  sorry

end intersection_in_fourth_quadrant_l146_146387


namespace determinant_zero_l146_146649

open Matrix

variables {R : Type*} [Field R] {a b : R}

def M : Matrix (Fin 3) (Fin 3) R :=
  ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ]

theorem determinant_zero : det M = 0 :=
by
  sorry

end determinant_zero_l146_146649


namespace largest_multiple_of_15_less_than_500_l146_146981

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146981


namespace length_of_XY_l146_146834

theorem length_of_XY (X Y Z : Type*) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (XZ : ℝ) (angleXPY : ℝ) (angleZ : ℝ)
  (h1 : XZ = 10 * √3) 
  (h2 : angleXPY = 90)
  (h3 : angleZ = 60) :
  XY = 30 := 
sorry

end length_of_XY_l146_146834


namespace power_of_3_l146_146346

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146346


namespace log_to_exp_l146_146181

theorem log_to_exp (y : ℝ) (h : Real.logBase 18 (6 * y) = 3) : y = 972 :=
by
  sorry

end log_to_exp_l146_146181


namespace value_of_f4_l146_146770

noncomputable def f : ℝ → ℝ := sorry

variables (x y : ℝ)

axiom domain_f : ∀ x, f x ∈ ℝ
axiom functional_eq1 : ∀ {x : ℝ}, x ≠ 0 → f x = x^3 * f (1 / x)
axiom functional_eq2 : ∀ {x y : ℝ}, f x + f y - 2 * x * y = f (x + y)

theorem value_of_f4 : f 4 = -20 :=
sorry

end value_of_f4_l146_146770


namespace num_values_fifty_eq_twelve_l146_146205

def numDivisors (n : ℕ) : ℕ :=
  nat.divisors n |>.length

def f1 (n : ℕ) : ℕ :=
  2 * numDivisors n

def f (j : ℕ) (n : ℕ) : ℕ :=
  nat.iterate f1 j n

theorem num_values_fifty_eq_twelve :
  (finset.range 51).filter (λ n, f 50 n = 12) .card = 10 :=
by
  sorry

end num_values_fifty_eq_twelve_l146_146205


namespace card_count_proof_l146_146532

noncomputable theory

variables (cards : Fin 12 → CardType) -- initialize the 12 cards
-- Define the possible card types
inductive CardType
| BW
| BB
| WW

-- Definition of the initial counts
def initial_black_up : ℕ := 9 

-- Definitions for black sides up after flips
def black_up_after_first_flip : ℕ := 4
def black_up_after_second_flip : ℕ := 6
def black_up_after_third_flip : ℕ := 5

-- Proving how many cards of each type there are
theorem card_count_proof (h0 : (Finset.range 12).sum (λ i, if cards i = CardType.BW then 1 else 0) + 
                                       2 * (Finset.range 12).sum (λ i, if cards i = CardType.BB then 1 else 0) = initial_black_up)
                         (h1 : (Finset.range 6).sum (λ i, if cards i = CardType.BW then 1 else 0) +
                               2 * (Finset.range 6).sum (λ i, if cards i = CardType.BB then 1 else 0) ≤ black_up_after_first_flip)
                         (h2 : (Finset.range 12).sum (λ i, if i ≥ 3 ∧ i < 9 then (if cards i = CardType.BW then 1 else 0) else 0) +
                               2 * (Finset.range 12).sum (λ i, if i ≥ 3 ∧ i < 9 then (if cards i = CardType.BB then 1 else 0) else 0) = black_up_after_second_flip)
                         (h3 : (Finset.range 12).sum (λ i, if (i < 3 ∨ i ≥ 9) then (if cards i = CardType.BW then 1 else 0) else 0) +
                               2 * (Finset.range 12).sum (λ i, if (i < 3 ∨ i ≥ 9) then (if cards i = CardType.BB then 1 else 0) else 0) = black_up_after_third_flip) :
  (Finset.range 12).sum (λ i, if cards i = CardType.BW then 1 else 0) = 9 ∧
  (Finset.range 12).sum (λ i, if cards i = CardType.WW then 1 else 0) = 3 ∧
  (Finset.range 12).sum (λ i, if cards i = CardType.BB then 1 else 0) = 0 := 
by 
  sorry

end card_count_proof_l146_146532


namespace grid_has_diff_colors_nodes_at_distance_5_l146_146147

noncomputable def node_color (grid : ℕ × ℕ → Color) : Prop :=
∀ (x1 y1 x2 y2 : ℕ), (x1 ≠ x2 ∨ y1 ≠ y2) → 
  (grid (x1, y1) ≠ grid (x2, y2)) ∧
  (dist (x1, y1) (x2, y2) = 5 → grid (x1, y1) ≠ grid (x2, y2))

theorem grid_has_diff_colors_nodes_at_distance_5 
  (grid : ℕ × ℕ → Color)
  (H1 : ∃ (x y : ℕ), grid (x, y) = Color.Blue)
  (H2 : ∃ (x y : ℕ), grid (x, y) = Color.Red)
  : ∃ (x1 y1 x2 y2 : ℕ), dist (x1, y1) (x2, y2) = 5 ∧ grid (x1, y1) ≠ grid (x2, y2) :=
sorry

end grid_has_diff_colors_nodes_at_distance_5_l146_146147


namespace hotel_charge_per_night_l146_146173

theorem hotel_charge_per_night (x : ℝ) (h1 : 6 * x + 4 * 2 = 80 - 63) : x = 1.50 :=
by
  linarith

end hotel_charge_per_night_l146_146173


namespace actual_size_of_plot_l146_146935

/-
Theorem: The actual size of the plot of land is 61440 acres.
Given:
- The plot of land is a rectangle.
- The map dimensions are 12 cm by 8 cm.
- 1 cm on the map equals 1 mile in reality.
- One square mile equals 640 acres.
-/

def map_length_cm := 12
def map_width_cm := 8
def cm_to_miles := 1 -- 1 cm equals 1 mile
def mile_to_acres := 640 -- 1 square mile is 640 acres

theorem actual_size_of_plot
  (length_cm : ℕ) (width_cm : ℕ) (cm_to_miles : ℕ → ℕ) (mile_to_acres : ℕ → ℕ) :
  length_cm = 12 → width_cm = 8 →
  (cm_to_miles 1 = 1) →
  (mile_to_acres 1 = 640) →
  (length_cm * width_cm * mile_to_acres (cm_to_miles 1 * cm_to_miles 1) = 61440) :=
by
  intros
  sorry

end actual_size_of_plot_l146_146935


namespace minute_hand_40_min_angle_l146_146924

noncomputable def minute_hand_rotation_angle (minutes : ℕ): ℝ :=
  if minutes = 60 then -2 * Real.pi 
  else (minutes / 60) * -2 * Real.pi

theorem minute_hand_40_min_angle :
  minute_hand_rotation_angle 40 = - (4 / 3) * Real.pi :=
by
  sorry

end minute_hand_40_min_angle_l146_146924


namespace population_reaches_max_capacity_l146_146836

theorem population_reaches_max_capacity :
  ∀ (initial_population max_capacity doubling_period eruption_period years : ℕ),
    initial_population = 400 →
    max_capacity = 15000 →
    doubling_period = 20 →
    eruption_period = 40 →
    years = 200 →
    ∃ n, (initial_population * 2 ^ (years / doubling_period) / 2 ^ (years / eruption_period) ≥ max_capacity) :=
by
  assume initial_population max_capacity doubling_period eruption_period years,
  assume h1 : initial_population = 400,
  assume h2 : max_capacity = 15000,
  assume h3 : doubling_period = 20,
  assume h4 : eruption_period = 40,
  assume h5 : years = 200,
  sorry

end population_reaches_max_capacity_l146_146836


namespace arithmetic_sequence_seventh_term_l146_146947

theorem arithmetic_sequence_seventh_term
  (a d : ℝ)
  (h_sum : 4 * a + 6 * d = 20)
  (h_fifth : a + 4 * d = 8) :
  a + 6 * d = 10.4 :=
by
  sorry -- proof to be provided

end arithmetic_sequence_seventh_term_l146_146947


namespace number_of_ordered_pairs_l146_146703

theorem number_of_ordered_pairs (x y : ℕ) (h : x * y = 540) : 
  ∃ p q r : ℕ, (p = 2 ∧ q = 3 ∧ r = 5) ∧ 
  (∀ a b c : ℕ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1) ∧ 
  (∏ k in finset.univ, ite (k = 0) ((p : ℕ) ^ (finset.card (finset.range (2+1)))) 1 *
     ite (k = 1) ((q : ℕ) ^ (finset.card (finset.range (3+1)))) 1 * 
     ite (k = 2) ((r : ℕ) ^ (finset.card (finset.range (1+1)))) 1 = 540) ∧ 
   (finset.card (divisors 540) = 24) :=
sorry

end number_of_ordered_pairs_l146_146703


namespace exists_permutation_ab_minus_cd_ge_two_l146_146737

theorem exists_permutation_ab_minus_cd_ge_two (p q r s : ℝ) 
  (h1 : p + q + r + s = 9) 
  (h2 : p^2 + q^2 + r^2 + s^2 = 21) :
  ∃ (a b c d : ℝ), (a, b, c, d) = (p, q, r, s) ∨ (a, b, c, d) = (p, q, s, r) ∨ 
  (a, b, c, d) = (p, r, q, s) ∨ (a, b, c, d) = (p, r, s, q) ∨ 
  (a, b, c, d) = (p, s, q, r) ∨ (a, b, c, d) = (p, s, r, q) ∨ 
  (a, b, c, d) = (q, p, r, s) ∨ (a, b, c, d) = (q, p, s, r) ∨ 
  (a, b, c, d) = (q, r, p, s) ∨ (a, b, c, d) = (q, r, s, p) ∨ 
  (a, b, c, d) = (q, s, p, r) ∨ (a, b, c, d) = (q, s, r, p) ∨ 
  (a, b, c, d) = (r, p, q, s) ∨ (a, b, c, d) = (r, p, s, q) ∨ 
  (a, b, c, d) = (r, q, p, s) ∨ (a, b, c, d) = (r, q, s, p) ∨ 
  (a, b, c, d) = (r, s, p, q) ∨ (a, b, c, d) = (r, s, q, p) ∨ 
  (a, b, c, d) = (s, p, q, r) ∨ (a, b, c, d) = (s, p, r, q) ∨ 
  (a, b, c, d) = (s, q, p, r) ∨ (a, b, c, d) = (s, q, r, p) ∨ 
  (a, b, c, d) = (s, r, p, q) ∨ (a, b, c, d) = (s, r, q, p) ∧ ab - cd ≥ 2 :=
sorry

end exists_permutation_ab_minus_cd_ge_two_l146_146737


namespace max_lessons_l146_146494

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end max_lessons_l146_146494


namespace largest_multiple_of_15_less_than_500_l146_146975

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146975


namespace new_max_speed_same_l146_146610

-- Define the given conditions as Lean definitions
variables {R : ℝ} (m : ℝ) (M : ℝ) (G : ℝ)

-- Given original maximum speed v
variables (v : ℝ)

-- Define linear mass density
def lambda := M / (2 * π * R)

-- Mass of the new ring
def M' := 2 * M

-- Initial potential energy for distance x >> R
def U_i (x : ℝ) := - (G * M * m) / x

-- Final potential energy when the particle reaches the original ring of radius R
def U_f := - (G * M * m) / R

-- Final potential energy when the particle reaches the new ring of radius 2R
def U_f' := - (G * (2 * M) * m) / (2 * R)

-- Kinetic energy and speed equation
def K := (1 / 2) * m * v^2 = (G * M * m) / R

-- New kinetic energy and new speed
def K' := (1 / 2) * m * v'^2 = (G * M * m) / R

-- Theorem stating the new maximum speed v' is equal to the original speed v
theorem new_max_speed_same : v' = v :=
sorry

end new_max_speed_same_l146_146610


namespace power_of_three_l146_146370

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146370


namespace binom_9_5_eq_126_l146_146680

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146680


namespace power_calculation_l146_146313

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146313


namespace find_f_2_l146_146808

variable (f : ℤ → ℤ)

-- Definitions of the conditions
def is_monic_quartic (f : ℤ → ℤ) : Prop :=
  ∃ a b c d, ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d

variable (hf_monic : is_monic_quartic f)
variable (hf_conditions : f (-2) = -4 ∧ f 1 = -1 ∧ f 3 = -9 ∧ f (-4) = -16)

-- The main statement to prove
theorem find_f_2 : f 2 = -28 := sorry

end find_f_2_l146_146808


namespace sum_first_1234_terms_l146_146945

def sequence : ℕ → ℕ 
| 0 := 1
| (n+1) := if n % 2 = 0 then 2 else 1

noncomputable def sum_sequence (n : ℕ) : ℕ :=
∑ i in range n, sequence i

theorem sum_first_1234_terms : sum_sequence 1234 = 2732 := 
by
  sorry

end sum_first_1234_terms_l146_146945


namespace tenth_day_of_month_is_monday_l146_146025

def total_run_minutes_in_month (hours : ℕ) : ℕ := hours * 60

def run_minutes_per_week (runs_per_week : ℕ) (minutes_per_run : ℕ) : ℕ := 
  runs_per_week * minutes_per_run

def weeks_in_month (total_minutes : ℕ) (minutes_per_week : ℕ) : ℕ := 
  total_minutes / minutes_per_week

def identify_day_of_week (first_day : ℕ) (target_day : ℕ) : ℕ := 
  (first_day + target_day - 1) % 7

theorem tenth_day_of_month_is_monday :
  let hours := 5
  let runs_per_week := 3
  let minutes_per_run := 20
  let first_day := 6 -- Assuming 0=Sunday, ..., 6=Saturday
  let target_day := 10
  total_run_minutes_in_month hours = 300 ∧
  run_minutes_per_week runs_per_week minutes_per_run = 60 ∧
  weeks_in_month 300 60 = 5 ∧
  identify_day_of_week first_day target_day = 1 := -- 1 represents Monday
sorry

end tenth_day_of_month_is_monday_l146_146025


namespace card_distribution_methods_l146_146906

theorem card_distribution_methods :
  let cards := [1, 2, 3, 4, 5, 6]
  let envelopes := {A, B, C}

  -- Total number of different methods (ways) to place 6 cards into 3 envelopes
  (∃ (f : fin 6 → fin 3), 
    ∀ (i j : fin 6), 
    (cards[i] = 1 ∨ cards[i] = 2) ∧ (cards[j] = 1 ∨ cards[j] = 2) → f i = f j) = 18 := sorry

end card_distribution_methods_l146_146906


namespace prove_bounds_l146_146466

variable (a b : ℝ)

-- Conditions
def condition1 : Prop := 6 * a - b = 45
def condition2 : Prop := 4 * a + b > 60

-- Proof problem statement
theorem prove_bounds (h1 : condition1 a b) (h2 : condition2 a b) : a > 10.5 ∧ b > 18 :=
sorry

end prove_bounds_l146_146466


namespace binom_9_5_l146_146672

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146672


namespace relationship_among_abc_l146_146750

noncomputable def a := 5^(Real.log 3.4 / Real.log 3)
noncomputable def b := 5^(Real.log 3.6 / Real.log 4)
noncomputable def c := (1 / 5)^((Real.log 0.3 / Real.log 3))

theorem relationship_among_abc : a > c ∧ c > b :=
by
  sorry

end relationship_among_abc_l146_146750


namespace range_of_m_l146_146383

theorem range_of_m (x m : ℝ) : 
  (∃ x : ℝ, (sin x)^2 + sin (2 * x) = m + 2 * (cos x)^2) ↔ 
  m ∈ Set.Icc ((-1 - Real.sqrt 13) / 2) ((-1 + Real.sqrt 13) / 2) :=
by sorry

end range_of_m_l146_146383


namespace length_percentage_increase_l146_146478

/--
Given that the area of a rectangle is 460 square meters and the breadth is 20 meters,
prove that the percentage increase in length compared to the breadth is 15%.
-/
theorem length_percentage_increase (A : ℝ) (b : ℝ) (l : ℝ) (hA : A = 460) (hb : b = 20) (hl : l = A / b) :
  ((l - b) / b) * 100 = 15 :=
by
  sorry

end length_percentage_increase_l146_146478


namespace max_lessons_l146_146493

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end max_lessons_l146_146493


namespace rate_of_simple_interest_l146_146556

variable (P A T SI : ℝ)
variable (R : ℝ)

-- Conditions
def principal := P = 750
def amount := A = 900
def time := T = 4

-- Simple Interest calculated as A - P
def simple_interest := SI = A - P

-- Given the formula SI = P * R * T / 100
def simple_interest_formula := SI = P * R * T / 100

theorem rate_of_simple_interest :
  principal P ∧ amount A ∧ time T ∧ simple_interest SI ∧ simple_interest_formula P A T SI →
  R = 5 :=
sorry

end rate_of_simple_interest_l146_146556


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l146_146841

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (d : ℕ) (h1 : d > 0) 
  (h2 : a 1 + a 4 + a 7 = 12) (h3 : a 1 * a 4 * a 7 = 28) :
  ∀ n, a n = n :=
sorry

theorem geometric_sequence_formula (b : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : b 1 = 16) (h2 : a 2 * b 2 = 4) :
  ∀ n, b n = 2^(n + 3) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = n) (h2 : ∀ n, b n = 2^(n + 3)) 
  (h3 : ∀ n, c n = a n * b n) :
  ∀ n, T n = 8 * (2^n * (n + 1) - 1) :=
sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l146_146841


namespace sin_x_expression_l146_146763

variable {a b x : ℝ}

theorem sin_x_expression (h1 : tan x = ab / (a^2 - b^2)) 
                         (h2 : a > b) (h3 : b > 0) 
                         (h4 : 0 < x) (h5 : x < π / 2) : 
                         sin x = ab / (sqrt (a^4 - a^2 * b^2 + b^4)) := 
  sorry

end sin_x_expression_l146_146763


namespace binomial_coefficient_9_5_l146_146668

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146668


namespace divide_polyhedron_into_prisms_l146_146538

noncomputable def triangular_pyramid_divide (A B C D K L M P Q F : Point) (AD BD CD AB AC: Line) 
  (plane1 plane2 : Plane) : Prop :=
  ∀ (h1 : K ∈ AD) (h2 : L ∈ BD) (h3 : M ∈ CD) (h4 : P ∈ AB) (h5 : Q ∈ AC)
    (h6 : F ∈ BC) (h7 : plane1 ∈ parallel ABC) (h8 : plane2 ∈ parallel BCD)
    (h9 : K ∈ plane1) (h10 : K ∈ plane2) (h11 : L ∈ plane1) (h12 : M ∈ plane1)
    (h13 : P ∈ plane2) (h14 : Q ∈ plane2),
  is_parallelogram K L F Q ∧ 
  is_prism K L M Q F and is_prism K P Q L B F

theorem divide_polyhedron_into_prisms (A B C D K L M P Q F : Point) (AD BD CD AB AC: Line) 
  (plane1 plane2 : Plane) 
  (h1 : K ∈ AD) (h2 : L ∈ BD) (h3 : M ∈ CD)
  (h4 : P ∈ AB) (h5 : Q ∈ AC) (h6 : F ∈ BC)
  (h7 : plane1 ∈ parallel ABC) (h8 : plane2 ∈ parallel BCD)
  (h9 : face_intersect plane1 plane2 = K) 
  (h10 : face_intersect plane1 BD ∈ M)
  (h11 : face_intersect plane2 AC ∈ P) :
  triangular_pyramid_divide A B C D K L M P Q F AD BD CD AB AC plane1 plane2 :=
begin
 sorry
end

end divide_polyhedron_into_prisms_l146_146538


namespace problem_f_2010_l146_146779

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1 / 4
axiom f_eq : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2010 : f 2010 = 1 / 2 :=
sorry

end problem_f_2010_l146_146779


namespace solve_for_y_l146_146468

theorem solve_for_y : ∀ y : ℤ, 7 + y = 3 ↔ y = -4 :=
by
  intro y
  constructor
  intro h
  rw [add_comm, add_eq_sub_iff, eq_comm] at h
  exact h.symm
  intro h
  rw [h]
  norm_num

end solve_for_y_l146_146468


namespace cost_of_one_shirt_l146_146813

theorem cost_of_one_shirt (J S K : ℕ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) (h3 : 3 * J + 2 * S + K = 90) : S = 15 :=
by
  sorry

end cost_of_one_shirt_l146_146813


namespace sum_seven_l146_146786

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (q : ℤ)

noncomputable def geometric_seq (a_n : ℕ → ℤ) (q : ℤ) (n : ℕ) : ℤ := a_n 1 * q^(n-1)

noncomputable def sum_of_geometric_seq (a_n : ℕ → ℤ) (q : ℤ) (n : ℕ) : ℤ := 
  a_n 1 * (1 - q^n) / (1 - q)

-- Conditions
axiom a1 : a_n 1 = 1
axiom a4 : a_n 4 = -8
axiom q_def : q = -2

-- Sum evaluation for n = 7
theorem sum_seven : S_n 7 = 128 / 3 := by
  -- Defining the geometric sequence with common ratio q = -2 and a_1 = 1
  let a_n := fun n : ℕ => geometric_seq a_n (-2) n
  -- Proving using the sum formula for the first 7 terms
  let S_n := fun n : ℕ => sum_of_geometric_seq a_n (-2) n
  have h : S_n 7 = 1 * (1 - (-2)^7) / (1 - (-2))
  rw [q_def, a1, a4]
  simp [geometric_seq, sum_of_geometric_seq]
  sorry

end sum_seven_l146_146786


namespace num_values_f_50_eq_12_l146_146203

-- Define the number of divisors of n
def num_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).length

-- Define f_1(n) as twice the number of positive integer divisors of n
def f_1 (n : ℕ) : ℕ :=
  2 * num_divisors n

-- Define f_j(n) for j >= 2 recursively
def f_j : ℕ → ℕ → ℕ
| 1, n := f_1 n
| (j + 1), n := f_1 (f_j j n)

-- Prove the number of values of n <= 50 such that f_50(n) = 12
theorem num_values_f_50_eq_12 : (Finset.range 50).filter (λ n, f_j 50 (n + 1) = 12).card = 10 := by
  sorry

end num_values_f_50_eq_12_l146_146203


namespace range_of_a_for_negative_root_l146_146209

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x + 1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_negative_root_l146_146209


namespace RobertAteNine_l146_146457

-- Define the number of chocolates Nickel ate
def chocolatesNickelAte : ℕ := 2

-- Define the additional chocolates Robert ate compared to Nickel
def additionalChocolates : ℕ := 7

-- Define the total chocolates Robert ate
def chocolatesRobertAte : ℕ := chocolatesNickelAte + additionalChocolates

-- State the theorem we want to prove
theorem RobertAteNine : chocolatesRobertAte = 9 := by
  -- Skip the proof
  sorry

end RobertAteNine_l146_146457


namespace binom_9_5_eq_126_l146_146695

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146695


namespace polynomial_expansion_correct_l146_146720

def polynomial1 (z : ℤ) : ℤ := 3 * z^3 + 4 * z^2 - 5
def polynomial2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2
def expandedPolynomial (z : ℤ) : ℤ := 12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10

theorem polynomial_expansion_correct (z : ℤ) :
  (polynomial1 z) * (polynomial2 z) = expandedPolynomial z :=
by sorry

end polynomial_expansion_correct_l146_146720


namespace required_percentage_is_correct_l146_146395

-- Defining the total votes and the percentage Geoff received
def total_votes : ℕ := 6000
def geoff_percentage : ℝ := 1 / 100 -- 1 percent

-- Calculating the votes Geoff received and the additional votes needed
def geoff_votes := geoff_percentage * total_votes
def additional_votes : ℝ := 3000

-- Total votes needed to win
def votes_needed_to_win := geoff_votes + additional_votes

-- Required percentage to win the election
def required_percentage_to_win := (votes_needed_to_win / total_votes) * 100

-- The theorem to prove
theorem required_percentage_is_correct : required_percentage_to_win = 51 := by
  sorry

end required_percentage_is_correct_l146_146395


namespace initial_distance_l146_146125

-- Definitions from the problem conditions
def speed_criminal := 8 / 60 -- in km per minute
def speed_policeman := 9 / 60 -- in km per minute
def time := 3 -- in minutes
def distance_after_time := 130 -- in km

-- The goal is to prove the initial distance between the policeman and the criminal
theorem initial_distance :
  let distance_criminal := speed_criminal * time;
  let distance_policeman := speed_policeman * time;
  let gap_reduced := distance_policeman - distance_criminal;
  initial_distance = distance_after_time + gap_reduced ->
  initial_distance = 130.05 :=
by
  -- sorry as the proof is not required
  sorry

end initial_distance_l146_146125


namespace angle_ECD_40_l146_146823

open Real EuclideanGeometry

-- Definitions of given conditions
variables (A B C D E : Point)
variables (h_1 : AC = BC)
variables (h_2 : m ∠ DCB = 50)
variables (h_3 : CD ∥ AB)
variables (h_4 : ∠ DE = 90)

-- Lean 4 statement of the problem
theorem angle_ECD_40 :
  m ∠ ECD = 40 := sorry

end angle_ECD_40_l146_146823


namespace sum_of_coefficients_l146_146233

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) : 
  (\prod^6 (1 - 2*x)) = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 → 
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 :=
by
  sorry

end sum_of_coefficients_l146_146233


namespace complex_symmetric_product_l146_146225

def z1 : ℂ := 3 - Complex.i

def symmetric_about_y_eq_x (z1 z2 : ℂ) : Prop :=
  z2 = Complex.conj z1

theorem complex_symmetric_product :
  ∃ z2 : ℂ, symmetric_about_y_eq_x z1 z2 ∧ z1 * z2 = 10 * Complex.i := by
  use -1 + 3 * Complex.i
  split
  {
    -- Proof that -1 + 3i is the symmetric point of 3 - i about the line y = x
    sorry
  }
  {
    -- Proof that (3 - i) * (-1 + 3i) = 10i
    sorry
  }

end complex_symmetric_product_l146_146225


namespace find_c_l146_146814

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end find_c_l146_146814


namespace value_of_expression_l146_146546

theorem value_of_expression : (3023 - 2990) ^ 2 / 121 = 9 := by
  sorry

end value_of_expression_l146_146546


namespace square_side_length_increase_l146_146497

variables {a x : ℝ}

theorem square_side_length_increase 
  (h : (a * (1 + x / 100) * 1.8)^2 = (1 + 159.20000000000002 / 100) * (a^2 + (a * (1 + x / 100))^2)) : 
  x = 100 :=
by sorry

end square_side_length_increase_l146_146497


namespace available_space_on_usb_l146_146902

theorem available_space_on_usb (total_capacity : ℕ) (used_percentage : ℝ) (total_capacity = 16) (used_percentage = 0.5) : 
  (total_capacity * (1 - used_percentage) = 8) := sorry

end available_space_on_usb_l146_146902


namespace highest_sales_to_lowest_diff_total_weekly_deviation_weekly_profit_l146_146586

noncomputable def daily_sales_differences : List ℤ :=
  [4, -3, -2, 7, -6, 18, -5]

def selling_price_per_box := 65
def number_of_workers := 3
def daily_expense_per_worker := 80
def packaging_fee_per_box := 5
def planned_sales_per_day := 10

theorem highest_sales_to_lowest_diff :
  (List.maximum daily_sales_differences - List.minimum daily_sales_differences) = 24 :=
sorry

theorem total_weekly_deviation :
  List.sum daily_sales_differences = 13 :=
sorry

theorem weekly_profit : 
  let total_boxes := 7 * planned_sales_per_day + List.sum daily_sales_differences in
  let revenue := total_boxes * (selling_price_per_box - packaging_fee_per_box) in
  let total_expenses := number_of_workers * daily_expense_per_worker * 7 in
  revenue - total_expenses = 3300 :=
sorry

end highest_sales_to_lowest_diff_total_weekly_deviation_weekly_profit_l146_146586


namespace largest_multiple_of_15_less_than_500_l146_146987

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146987


namespace isosceles_trapezoid_rotation_generates_cylinder_two_cones_l146_146459

-- Given condition: the trapezoid is isosceles with a longer base.
def is_isosceles_trapezoid (t : Type) : Prop := 
  ∃ a b c d : ℝ, t = (a, b, c, d) ∧ a = c ∧ b ≠ d

-- Define the rotation generating the desired solid.
def rotation_generates_solids (t : Type) : Prop :=
  ∀ (t : Type),
  is_isosceles_trapezoid t →
  (∃ cylinder : Type, ∃ cone1 cone2: Type, 
  (rotation_of t contains cylinder) ∧ 
  (rotation_of t contains cone1) ∧ 
  (rotation_of t contains cone2))

-- The main theorem stating the result of the problem.
theorem isosceles_trapezoid_rotation_generates_cylinder_two_cones (t : Type) :
  is_isosceles_trapezoid t →
  rotation_generates_solids t :=
by
  sorry

end isosceles_trapezoid_rotation_generates_cylinder_two_cones_l146_146459


namespace determine_m_ratio_l146_146142

def ratio_of_C_to_A_investment (x : ℕ) (m : ℕ) (total_gain : ℕ) (a_share : ℕ) : Prop :=
  total_gain = 18000 ∧ a_share = 6000 ∧
  (12 * x / (12 * x + 4 * m * x) = 1 / 3)

theorem determine_m_ratio (x : ℕ) (m : ℕ) (h : ratio_of_C_to_A_investment x m 18000 6000) :
  m = 6 :=
by
  sorry

end determine_m_ratio_l146_146142


namespace num_values_fifty_eq_twelve_l146_146206

def numDivisors (n : ℕ) : ℕ :=
  nat.divisors n |>.length

def f1 (n : ℕ) : ℕ :=
  2 * numDivisors n

def f (j : ℕ) (n : ℕ) : ℕ :=
  nat.iterate f1 j n

theorem num_values_fifty_eq_twelve :
  (finset.range 51).filter (λ n, f 50 n = 12) .card = 10 :=
by
  sorry

end num_values_fifty_eq_twelve_l146_146206


namespace election_votes_l146_146955

noncomputable def V : ℕ := 12800
noncomputable def winner_votes : ℕ := 5120
noncomputable def second_place_votes : ℕ := 3584
noncomputable def third_place_votes : ℕ := 2560
noncomputable def fourth_place_votes : ℕ := 1536

theorem election_votes :
  (0.40 * V = winner_votes) ∧
  (0.28 * V = second_place_votes) ∧
  (0.20 * V = third_place_votes) ∧
  (winner_votes - second_place_votes = 1536) ∧
  (winner_votes - third_place_votes = 3840) ∧
  (winner_votes - fourth_place_votes = 5632) ∧
  (V = winner_votes + second_place_votes + third_place_votes + fourth_place_votes) :=
by {
  sorry
}

end election_votes_l146_146955


namespace power_of_three_l146_146367

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146367


namespace possible_triangle_areas_l146_146733

noncomputable def valid_triangle_area (a b c h_a h_b h_c r : ℕ) : Prop :=
  let Δ := (1/2 : ℝ) * (a * h_a) in
  h_a + h_b + h_c < 20 ∧
  r = Δ / ((a + b + c) / 2) ∧
  Δ = (1 / 2 * b * h_b : ℝ) ∧
  Δ = (1 / 2 * c * h_c : ℝ)

theorem possible_triangle_areas (a b c h_a h_b h_c r : ℕ) (Δ : ℕ)
  (h_conditions : valid_triangle_area a b c h_a h_b h_c r) :
  Δ = 6 ∨ Δ = 12 :=
sorry

end possible_triangle_areas_l146_146733


namespace distance_from_line_parallelogram_l146_146100

variables {A B C D : Point} {g : Line}

-- Assuming A, B, C, D are points of a parallelogram and g is a line passing through A
def is_parallelogram (A B C D : Point) : Prop := 
  (A - B) + (C - D) = 0 ∧ (A - D) + (B - C) = 0

def distance (p : Point) (l : Line) : Real := sorry  -- Define the distance function as needed

theorem distance_from_line_parallelogram (h₁ : is_parallelogram A B C D) (h₂ : g.contains A) :
  distance C g = distance B g + distance D g ∨ distance C g = |distance B g - distance D g| :=
by
  sorry

end distance_from_line_parallelogram_l146_146100


namespace power_equality_l146_146323

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146323


namespace evaluate_cubic_diff_l146_146751

theorem evaluate_cubic_diff (x y : ℝ) (h1 : x + y = 12) (h2 : 2 * x + y = 16) : x^3 - y^3 = -448 := 
by
    sorry

end evaluate_cubic_diff_l146_146751


namespace collinearity_condition_l146_146709

open Real EuclideanSpace

noncomputable def collinear_points (a b c d : ℝ) : Prop :=
  let p1 := ⟨(1 : ℝ), (0 : ℝ), a⟩
  let p2 := ⟨b, (1 : ℝ), (0 : ℝ)⟩
  let p3 := ⟨(0 : ℝ), c, (1 : ℝ)⟩
  let p4 := ⟨3 * d, 3 * d, -d⟩
  collinear ![p1, p2, p3, p4]

theorem collinearity_condition (a b c : ℝ) :
  ∀ (d : ℝ), collinear_points a b c d ↔ d = 2 := sorry

end collinearity_condition_l146_146709


namespace power_of_three_l146_146369

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146369


namespace binomial_coefficient_9_5_l146_146670

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146670


namespace binom_9_5_l146_146673

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146673


namespace determinant_identity_l146_146643

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l146_146643


namespace mod_exp_value_l146_146020

theorem mod_exp_value (m : ℕ) (h1: 0 ≤ m) (h2: m < 9) (h3: 14^4 ≡ m [MOD 9]) : m = 5 :=
by
  sorry

end mod_exp_value_l146_146020


namespace minimum_value_of_exponential_expression_l146_146747

open Real

theorem minimum_value_of_exponential_expression 
  (a b : ℝ) 
  (h : log 2 a + log 2 b ≥ 1) : 
  3^a + 9^b ≥ 18 :=
sorry

end minimum_value_of_exponential_expression_l146_146747


namespace max_lessons_l146_146495

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end max_lessons_l146_146495


namespace power_addition_l146_146302

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146302


namespace number_of_days_with_equal_mondays_and_wednesdays_l146_146123

theorem number_of_days_with_equal_mondays_and_wednesdays :
  ∃ d: ℕ, d = 2 ∧
    ∀ (start: ℕ), start < 7 →
    (let days := (List.range 31) in
     let weekdays := days.map (λ n, (start + n) % 7) in
     let mondays := weekdays.countp (λ d, d = 0) in
     let wednesdays := weekdays.countp (λ d, d = 2) in
     mondays = wednesdays → (start = 3 ∨ start = 4)) := sorry

end number_of_days_with_equal_mondays_and_wednesdays_l146_146123


namespace defective_pens_count_l146_146391

theorem defective_pens_count (total_pens : ℕ) (prob_not_defective : ℚ) (D : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : prob_not_defective = 0.5357142857142857) : 
  D = 2 := 
by
  sorry

end defective_pens_count_l146_146391


namespace complement_U_P_l146_146218

theorem complement_U_P :
  let U := {y : ℝ | y ≠ 0 }
  let P := {y : ℝ | 0 < y ∧ y < 1/2}
  let complement_U_P := {y : ℝ | y ∈ U ∧ y ∉ P}
  (complement_U_P = {y : ℝ | y < 0} ∪ {y : ℝ | y > 1/2}) :=
by
  sorry

end complement_U_P_l146_146218


namespace binom_9_5_l146_146678

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146678


namespace find_angle_E_l146_146851

def trapezoid_angles (E H F G : ℝ) : Prop :=
  E + H = 180 ∧ E = 3 * H ∧ G = 4 * F

theorem find_angle_E (E H F G : ℝ) 
  (h1 : E + H = 180)
  (h2 : E = 3 * H)
  (h3 : G = 4 * F) : 
  E = 135 := by
    sorry

end find_angle_E_l146_146851


namespace repeating_decimal_as_fraction_l146_146049

theorem repeating_decimal_as_fraction :
  ∃ (m n : ℕ), nat.coprime m n ∧ 2.0151515 = (m : ℚ) / (n : ℚ) ∧ (m + n = 199) :=
by
  sorry

end repeating_decimal_as_fraction_l146_146049


namespace perp_value_of_a_parallel_value_and_distance_of_a_l146_146796

-- Definitions for the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ × ℝ × ℝ := (a, 3, 1)
def l₂ (a : ℝ) : ℝ × ℝ × ℝ := (1, a - 2, a)

-- Condition for being perpendicular
def perpendicular (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1;
  let (A2, B2, C2) := l2;
  (A1 * A2 + B1 * B2 = 0)

-- Condition for being parallel
def parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1;
  let (A2, B2, C2) := l2;
  (A1 * B2 - A2 * B1 = 0)

-- Distance between two parallel lines
def distance (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  let (A1, B1, C1) := l1;
  let (A2, B2, C2) := l2;
  Real.abs (C2 * A1 - C1 * A2) / Real.sqrt (A1^2 + B1^2)

-- Theorem for perpendicular lines
theorem perp_value_of_a (a : ℝ) (h : perpendicular (l₁ a) (l₂ a)) : 
  a = 3 / 2 :=
sorry

-- Theorem for parallel lines and their distance
theorem parallel_value_and_distance_of_a (a : ℝ) (h : parallel (l₁ a) (l₂ a)) 
  (ha : a = 3) : distance (l₁ a) (l₂ a) = 4 * Real.sqrt 2 / 3 := 
sorry

end perp_value_of_a_parallel_value_and_distance_of_a_l146_146796


namespace find_a_l146_146778

-- Define the function f and the conditions
def f (a x : ℝ) : ℝ := abs (Real.log x / Real.log a)

-- Define the theorem to prove the values of 'a'
theorem find_a (a m n : ℝ) (h_dom : 0 < a) (ha : a < 1) (h_range : ∀ x ∈ set.Icc m n, f a x ∈ set.Icc 0 1) (h_min_diff : n - m = 1/3):
  a = 2/3 ∨ a = 3/4 :=
sorry

end find_a_l146_146778


namespace distance_inequality_l146_146420

theorem distance_inequality 
  (A B C D : Point)
  (dist : Point → Point → ℝ)
  (h_dist_pos : ∀ P Q : Point, dist P Q ≥ 0)
  (AC BD AD BC AB CD : ℝ)
  (hAC : AC = dist A C)
  (hBD : BD = dist B D)
  (hAD : AD = dist A D)
  (hBC : BC = dist B C)
  (hAB : AB = dist A B)
  (hCD : CD = dist C D) :
  AC^2 + BD^2 + AD^2 + BC^2 ≥ AB^2 + CD^2 := 
by
  sorry

end distance_inequality_l146_146420


namespace power_of_three_l146_146284

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146284


namespace tetrahedron_inequality_condition_for_equality_l146_146874

variables {A B C D : Type} [EuclideanSpace ℝ] {a b c d : A B C D}
variables (AB BC CA AD BD CD : ℝ)
variables (orthocenter : IsOrthocenter a b c d)

def tetrahedron (ABCD : Type*) := 
  B ≠ C ∧ BD ⊥ DC ∧ orthocenter

theorem tetrahedron_inequality (h: tetrahedron ABCD) : 
  (AB + BC + CA)^2 ≤ 6 * (AD^2 + BD^2 + CD^2) :=
by
  sorry

theorem condition_for_equality (h: tetrahedron ABCD) : 
  (AB + BC + CA)^2 = 6 * (AD^2 + BD^2 + CD^2) ↔ is_equilateral a b c :=
by
  sorry

end tetrahedron_inequality_condition_for_equality_l146_146874


namespace combined_work_rate_of_A_and_B_l146_146141

noncomputable def work_rate (days: ℕ) : ℚ := 1 / days

theorem combined_work_rate_of_A_and_B (hB : work_rate 36 = 1 / 36) 
    (hA : work_rate 36 = work_rate 36) :
    1 / (work_rate 36 + work_rate 36) = 18 :=
by
  have hA_rate : work_rate 36 = 1 / 36 := by sorry
  have hB_rate : work_rate 36 = 1 / 36 := hB
  have combined_rate : work_rate 36 + work_rate 36 = 1 / 18 := by sorry
  show 1 / (1 / 18) = 18 from by sorry

end combined_work_rate_of_A_and_B_l146_146141


namespace solve_equation_l146_146017

theorem solve_equation (x : ℚ) (h : x ≠ 1) : (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by 
  sorry

end solve_equation_l146_146017


namespace correct_statements_about_algorithms_l146_146149

-- Definitions of the conditions
def statement1 : Prop := ∃ steps, (unclear steps ∧ allows_more_problems steps)
def statement2 : Prop := ∀ (algorithm : α → β), (correct algorithm → yields_definite_result algorithm)
def statement3 : Prop := ∃ (algorithm1 algorithm2 : α → β), (solves_same_type_problems algorithm1 algorithm2 ∧ algorithm1 ≠ algorithm2)
def statement4 : Prop := ∀ (algorithm : α → β), (correct algorithm → terminates_finitely algorithm)

-- The theorem we need to prove
theorem correct_statements_about_algorithms :
  ¬ statement1 ∧ statement2 ∧ statement3 ∧ statement4 :=
by
  sorry

end correct_statements_about_algorithms_l146_146149


namespace exists_polynomial_l146_146432

variable {α : Type}
variable [Ring α]

def satisfies_condition (a : ℕ → α) : Prop :=
  ∃ N, ∀ m ≥ N, ∑ n in Finset.range (m + 1), a n * (-1)^n * (Nat.choose m n : α) = 0

theorem exists_polynomial (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∃ P : ℕ → ℝ, ∀ n, a n = P n :=
sorry

end exists_polynomial_l146_146432


namespace find_M_floor_l146_146229

theorem find_M_floor : 
  (∑ n in finset.range 7, 1 / (((3 + n)! : ℝ) * ((16 - n)! : ℝ))) = (M / (2 * 17)!) → 
  ⌊M / 100⌋ = 145 :=
by 
  intro h
  sorry

end find_M_floor_l146_146229


namespace determinant_zero_l146_146652

open Matrix

variables {R : Type*} [Field R] {a b : R}

def M : Matrix (Fin 3) (Fin 3) R :=
  ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ]

theorem determinant_zero : det M = 0 :=
by
  sorry

end determinant_zero_l146_146652


namespace projection_of_a_onto_b_l146_146888

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (h1 : angle a b = 60 * (Real.pi / 180)) (h2 : ‖a‖ = 2) (h3 : ‖b‖ = 2)

theorem projection_of_a_onto_b :
  orthogonal_projection (ℝ ∙ b) a = (1 / 2 : ℝ) • b :=
sorry

end projection_of_a_onto_b_l146_146888


namespace find_a1_in_arithmetic_sequence_l146_146424

noncomputable def arithmetic_sequence_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem find_a1_in_arithmetic_sequence :
  ∀ (a₁ d : ℤ), d = -2 →
  (arithmetic_sequence_sum a₁ d 11 = arithmetic_sequence_sum a₁ d 10) →
  a₁ = 20 :=
by
  intro a₁ d hd hs
  sorry

end find_a1_in_arithmetic_sequence_l146_146424


namespace time_to_grate_cheese_for_one_omelet_l146_146633

theorem time_to_grate_cheese_for_one_omelet : 
  (total_time : ℕ) (time_chop_pepper : ℕ) (num_peppers : ℕ) (time_chop_onion : ℕ)
  (num_onions : ℕ) (time_cook_one_omelet : ℕ) (num_omelets : ℕ) 
  (time_to_grate_one_omelet : ℕ) 
  (H1 : total_time = 50)
  (H2 : time_chop_pepper = 3)
  (H3 : num_peppers = 4)
  (H4 : time_chop_onion = 4)
  (H5 : num_onions = 2)
  (H6 : time_cook_one_omelet = 5)
  (H7 : num_omelets = 5)
  (H8 : total_time = num_peppers * time_chop_pepper + num_onions * time_chop_onion + num_omelets * time_cook_one_omelet + num_omelets * time_to_grate_one_omelet)
  : time_to_grate_one_omelet = 5 :=
by
  sorry

end time_to_grate_cheese_for_one_omelet_l146_146633


namespace lowest_fraction_job_done_in_1_hour_l146_146557

theorem lowest_fraction_job_done_in_1_hour (hA : 4 > 0) (hB : 5 > 0) (hC : 8 > 0) :
  let rateA := (1 : ℚ) / 4
  let rateB := (1 : ℚ) / 5
  let rateC := (1 : ℚ) / 8
  let combineTwoSlowest := rateB + rateC
  combineTwoSlowest = 13 / 40 :=
by
  let rateA := (1 : ℚ) / 4
  let rateB := (1 : ℚ) / 5
  let rateC := (1 : ℚ) / 8
  let combineTwoSlowest := rateB + rateC
  have : combineTwoSlowest = (1 / 5) + (1 / 8) := rfl
  have : combineTwoSlowest = (8 / 40) + (5 / 40) := by rw [this, div_eq_mul_one_div, div_eq_mul_one_div]; ratea
  have : combineTwoSlowest = 13 / 40 := by norm_num [this]
  exact this

end lowest_fraction_job_done_in_1_hour_l146_146557


namespace inverse_function_l146_146727

theorem inverse_function (x y : ℝ) (h : x ≥ 0) : 
  (y = 2 * sqrt x) ↔ (x = (y / 2)^2) := by
  sorry

end inverse_function_l146_146727


namespace angle_BDC_of_isosceles_triangle_l146_146959

theorem angle_BDC_of_isosceles_triangle 
  (A B C D : Point) 
  (h_isosceles : dist A B = dist A C)
  (h_bisectors_meet : ∀ {P : Point}, 
    angle A B C = angle A C B ∧ bisects_exterior_angle_at P B C ∧ bisects_exterior_angle_at P C B) :
  angle BDC = (180 - angle BAC) / 2 :=
sorry

end angle_BDC_of_isosceles_triangle_l146_146959


namespace power_of_3_l146_146355

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146355


namespace work_done_is_2700_joules_l146_146583

noncomputable def work_done_during_isothermal_compression 
  (P0 : ℝ) (H : ℝ) (h : ℝ) (R : ℝ) : ℝ :=
  P0 * π * R^2 * H * Real.log (H / (H - h))

theorem work_done_is_2700_joules :
  let P0 := 103300 -- P0 in Pascals
  let H := 0.4
  let h := 0.35
  let R := 0.1 in
  work_done_during_isothermal_compression P0 H h R ≈ 2700 :=
by
  sorry

end work_done_is_2700_joules_l146_146583


namespace sequence_length_l146_146801

theorem sequence_length 
  (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) (n : ℕ) 
  (h₁ : a₁ = -4) 
  (h₂ : d = 3) 
  (h₃ : aₙ = 32) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  n = 13 := 
by 
  sorry

end sequence_length_l146_146801


namespace king_min_horizontal_vertical_moves_l146_146076

theorem king_min_horizontal_vertical_moves :
  ∃ path : list (ℕ × ℕ),
    is_closed_non_intersecting_path path (8, 8) ∧
    (∀ i, (path.nth i ≠ none) → (path.nth (i + 1) ≠ none) →
      king_move_valid (path.nth i).get_or_else (0, 0) (path.nth (i + 1)).get_or_else (0, 0)) →
    (∀ x ∈ path, 0 ≤ x.1 ∧ x.1 < 8 ∧ 0 ≤ x.2 ∧ x.2 < 8) →
    path.length = 64 →
    return_to_start path →
    count_horizontal_vertical_moves path ≥ 28 :=
  sorry

end king_min_horizontal_vertical_moves_l146_146076


namespace power_of_3_l146_146350

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146350


namespace exponent_power_identity_l146_146338

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146338


namespace num_nine_letter_great_words_l146_146707

/-- A type representing the letters. -/
inductive Letter
| A | B | C | D

/-- Predicate indicating that a sequence is a great word. -/
def is_great_word : List Letter → Prop
| []       => True
| [x]      => True
| (x::y::xs) => match x, y with
                | Letter.A, Letter.B => False
                | Letter.B, Letter.C => False
                | Letter.C, Letter.D => False
                | Letter.D, Letter.A => False
                | _, _               => is_great_word (y::xs)

/-- Count the number of nine-letter great words. -/
def count_great_words (n : Nat) : Nat :=
  if h : n = 9 then 4 * 3^(n-1) else 0

theorem num_nine_letter_great_words : count_great_words 9 = 26244 :=
by
  unfold count_great_words
  simp [Nat.pow, pow_succ]
  sorry

end num_nine_letter_great_words_l146_146707


namespace triangle_AC_l146_146838

noncomputable def right_triangle := 
  ∃ (A B C : ℝ) (BC : ℝ) (sinB : ℝ), 
    A = 90 ∧ 
    BC = 10 ∧ 
    sinB = 0.6 ∧ 
    BC * sinB = 6

theorem triangle_AC :
  right_triangle → ∃ (AC : ℝ), AC = 6 :=
by
  intro h
  unfold right_triangle at h
  rcases h with ⟨A, B, C, BC, sinB, hA, hBC, hsinB, hAC⟩
  use 6
  exact hAC

end triangle_AC_l146_146838


namespace sum_of_cubes_sum_of_cubes_100_l146_146702

theorem sum_of_cubes (n : ℕ) :
  (∑ k in Finset.range (n + 1), k^3) = (n * (n + 1) / 2)^2 :=
by sorry

theorem sum_of_cubes_100 :
  (∑ k in Finset.range 101, k^3) = 25502500 :=
by sorry

end sum_of_cubes_sum_of_cubes_100_l146_146702


namespace find_f_2_l146_146807

variable (f : ℤ → ℤ)

-- Definitions of the conditions
def is_monic_quartic (f : ℤ → ℤ) : Prop :=
  ∃ a b c d, ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d

variable (hf_monic : is_monic_quartic f)
variable (hf_conditions : f (-2) = -4 ∧ f 1 = -1 ∧ f 3 = -9 ∧ f (-4) = -16)

-- The main statement to prove
theorem find_f_2 : f 2 = -28 := sorry

end find_f_2_l146_146807


namespace conical_hat_surface_area_l146_146221

noncomputable def lateral_surface_area_of_cone (r l : ℝ) : ℝ :=
  π * r * l

theorem conical_hat_surface_area :
  ∀ (d l : ℝ), d = 4 → l = 6 → lateral_surface_area_of_cone (d / 2) l = 12 * π :=
by
  intros d l hd hl
  rw [hd, hl]
  unfold lateral_surface_area_of_cone
  simp
  sorry

end conical_hat_surface_area_l146_146221


namespace find_two_digit_number_l146_146946

-- Define the conditions for the digits.
def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem find_two_digit_number : 
  ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 
           tens_digit n + units_digit n = 10 ∧ 
           tens_digit n * 4 = units_digit n :=
by
  use 28
  have h1 : tens_digit 28 = 2 := by norm_num
  have h2 : units_digit 28 = 8 := by norm_num
  have h3 : 28 / 10 + 28 % 10 = 10 := by norm_num
  have h4 : 2 * 4 = 8 := by norm_num
  split
  swap, repeat { split },
  all_goals { try { norm_num }, try { exact h3 }, try { exact h4 } }
  sorry

end find_two_digit_number_l146_146946


namespace largest_multiple_of_15_less_than_500_l146_146979

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146979


namespace circle_radius_l146_146041

theorem circle_radius (x : ℝ) (h : x > 3) 
    (hx1 : (x - 2)^2 + 6^2 = (x - 3)^2 + 2^2) : 
    sqrt ((x - 2)^2 + 36) = 13 := by
  sorry

end circle_radius_l146_146041


namespace rectangle_area_l146_146597

theorem rectangle_area (x : ℝ) (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l^2 + w^2 = x^2) :
    l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l146_146597


namespace log_a_b_integer_probability_l146_146535

theorem log_a_b_integer_probability :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 25}
  let pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ a ≠ b}
  let valid_pairs := {pair | pair ∈ pairs ∧ ∃ k : ℕ, b = k * a ∧ k ≥ 2}
  let total_pairs := finset.card pairs
  let valid_pairs_count := finset.card valid_pairs
  (valid_pairs_count / total_pairs : ℝ) = 31 / 300 :=
sorry

end log_a_b_integer_probability_l146_146535


namespace determinant_sin_eq_zero_l146_146656

theorem determinant_sin_eq_zero (a b : ℝ) : 
  matrix.det ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by
  sorry

end determinant_sin_eq_zero_l146_146656


namespace binom_two_formula_l146_146540

def binom (n k : ℕ) : ℕ :=
  n.choose k

-- Formalizing the conditions
variable (n : ℕ)
variable (h : n ≥ 2)

-- Stating the problem mathematically in Lean
theorem binom_two_formula :
  binom n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_formula_l146_146540


namespace polynomial_degree_condition_l146_146447

noncomputable def poly_degrees (P : ℝ → ℝ) : Prop :=
  (∃ n : ℕ, P = (λ t, if n = 0 then 1/2 else (t^2 - 1/2) * (λ u, u^(2 * n))(t^2 - 1/2) + 1/2)) 
  ∨ P = (λ t, 1/2)

theorem polynomial_degree_condition (P : ℝ → ℝ) 
  (h : ∀ x : ℝ, P (Math.sin x) + P (Math.cos x) = 1) : poly_degrees P :=
sorry

end polynomial_degree_condition_l146_146447


namespace option_A_ne_two_option_B_eq_two_option_C_eq_two_option_D_eq_two_l146_146084

noncomputable def option_A : ℝ := Real.sqrt ((Real.pi - 4)^2)
noncomputable def option_B : ℝ := Real.root 2023 (2^2023)
noncomputable def option_C : ℝ := -Real.cbrt (-(2^3))
noncomputable def option_D : ℝ := Real.sqrt ((-2)^2)

theorem option_A_ne_two : option_A ≠ 2 := by sorry
theorem option_B_eq_two : option_B = 2 := by sorry
theorem option_C_eq_two : option_C = 2 := by sorry
theorem option_D_eq_two : option_D = 2 := by sorry

end option_A_ne_two_option_B_eq_two_option_C_eq_two_option_D_eq_two_l146_146084


namespace intersection_with_x_axis_l146_146192

theorem intersection_with_x_axis (t : ℝ) (x y : ℝ) 
  (h1 : x = -2 + 5 * t) 
  (h2 : y = 1 - 2 * t) 
  (h3 : y = 0) : x = 1 / 2 := 
by 
  sorry

end intersection_with_x_axis_l146_146192


namespace scientific_calculator_ratio_l146_146119

theorem scientific_calculator_ratio (total : ℕ) (basic_cost : ℕ) (change : ℕ) (sci_ratio : ℕ → ℕ) (graph_ratio : ℕ → ℕ) : 
  total = 100 →
  basic_cost = 8 →
  sci_ratio basic_cost = 8 * x →
  graph_ratio (sci_ratio basic_cost) = 3 * sci_ratio basic_cost →
  change = 28 →
  8 + (8 * x) + (24 * x) = 72 →
  x = 2 :=
by
  sorry

end scientific_calculator_ratio_l146_146119


namespace relation_f_a1_f_b2_l146_146933

noncomputable def f (a b x : ℝ) : ℝ := real.log_base a (abs (x - b))

theorem relation_f_a1_f_b2 (a b : ℝ) (h_even : ∀ x, f a b (-x) = f a b x) (h_mono : ∀ x y, x < y → x ∈ set.Ioo (-∞) 0 → y ∈ set.Ioo (-∞) 0 → f a b x < f a b y) : 
  f a b (a + 1) > f a b (b + 2) :=
by
  sorry

end relation_f_a1_f_b2_l146_146933


namespace steve_keeps_total_money_excluding_advance_l146_146471

-- Definitions of the conditions
def totalCopies : ℕ := 1000000
def advanceCopies : ℕ := 100000
def pricePerCopy : ℕ := 2
def agentCommissionRate : ℚ := 0.1

-- Question and final proof
theorem steve_keeps_total_money_excluding_advance :
  let totalEarnings := totalCopies * pricePerCopy
  let agentCommission := agentCommissionRate * totalEarnings
  let moneyKept := totalEarnings - agentCommission
  moneyKept = 1800000 := by
  -- Proof goes here, but we skip it for now
  sorry

end steve_keeps_total_money_excluding_advance_l146_146471


namespace exponent_power_identity_l146_146331

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146331


namespace remainder_of_polynomial_division_is_88_l146_146195

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem remainder_of_polynomial_division_is_88 :
  p 2 = 88 :=
by
  sorry

end remainder_of_polynomial_division_is_88_l146_146195


namespace largest_multiple_of_15_under_500_l146_146988

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146988


namespace quadratic_fraction_evaluation_l146_146418

-- Defining the quadratic function
def quadratic_function (c h d : ℝ) (x : ℝ) : ℝ :=
  c * (x - h) ^ 2 + d

theorem quadratic_fraction_evaluation :
  ∀ (c h d : ℝ), (h = 3 / 2) → (∀ (a b : ℝ), a ≠ b → quadratic_function c h d a = quadratic_function c h d b → quadratic_function c h d (a^2 - 6*b - 1) = quadratic_function c h d (b^2 + 8)) →
  (quadratic_function c h d (8) - quadratic_function c h d (2)) / (quadratic_function c h d (2) - quadratic_function c h d (1)) = 13 :=
begin
  intros,
  sorry
end

end quadratic_fraction_evaluation_l146_146418


namespace largest_multiple_of_15_under_500_l146_146989

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146989


namespace binom_9_5_eq_126_l146_146693

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146693


namespace inverse_function_correct_l146_146038

noncomputable def original_function (x : ℝ) : ℝ := (π / 2) + Real.arcsin (3 * x)
noncomputable def candidate_inverse (y : ℝ) : ℝ := - (1 / 3) * Real.cos y

theorem inverse_function_correct (x : ℝ) (hx : -1 / 3 ≤ x ∧ x ≤ 1 / 3) :
  (∃ y, original_function x = y ∧ 0 ≤ y ∧ y ≤ π) ↔ 
  (∀ y, 0 ≤ y ∧ y ≤ π → candidate_inverse y = x) := by 
  sorry

end inverse_function_correct_l146_146038


namespace total_plants_in_garden_l146_146627

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l146_146627


namespace sequence_formula_l146_146264

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 1 then -1 else 2 / (n * (n + 1))

def Sn (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq (i + 1)

theorem sequence_formula (a_n : ℕ → ℚ) :
  (∀ n, a_n n = seq n) → 
  (Sn 1 = -1) → 
  (∀ n, n ≥ 2 → (Sn n)^2 - (a_n n) * (Sn n) = 2 * (a_n n)) →
  ∀ n, a_n n = seq n :=
  by { intros h1 h2 h3, sorry }

end sequence_formula_l146_146264


namespace part_a_part_b_l146_146852

-- Define conditions
variables (n : ℕ) (P : fin n → (ℝ × ℝ))

-- Additional constraints
-- Assume the square has a side length of 1, so all points must be within [0, 1] × [0, 1]
def is_in_square (p : ℝ × ℝ) : Prop := 
  0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1

-- Each point in P satisfies the constraint of being inside the square
axiom points_in_square : ∀ i, is_in_square (P i)

-- Prove (a): There exists a triangle with area ≤ 1/(2(n + 1))
theorem part_a : 
  ∃ (A B C : ℝ × ℝ), 
  (A ∈ (P i | i < n) ∪ {(0, 0), (0, 1), (1, 0), (1, 1)}) ∧
  (B ∈ (P i | i < n) ∪ {(0, 0), (0, 1), (1, 0), (1, 1)}) ∧
  (C ∈ (P i | i < n) ∪ {(0, 0), (0, 1), (1, 0), (1, 1)}) ∧
  (let area := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area ≤ 1/(2*(n + 1))) :=
sorry

-- Prove (b): There exists a triangle with area ≤ 1/(n - 2)
theorem part_b : 
  ∃ (A B C : ℝ × ℝ), 
  (A ∈ (P i | i < n)) ∧
  (B ∈ (P i | i < n)) ∧
  (C ∈ (P i | i < n)) ∧
  (let area := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area ≤ 1/(n - 2)) :=
sorry

end part_a_part_b_l146_146852


namespace exponent_power_identity_l146_146342

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146342


namespace power_addition_l146_146292

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146292


namespace ion_electronic_structure_l146_146039

theorem ion_electronic_structure (R M Z n m X : ℤ) (h1 : R + X = M - n) (h2 : M - n = Z - m) (h3 : n > m) : M > Z ∧ Z > R := 
by 
  sorry

end ion_electronic_structure_l146_146039


namespace find_f2019_l146_146222

def func_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f(x + 2) + f(x - 2) = 2 * f(2)) ∧
  (∀ x : ℝ, f(x + 2) = f(2 - x)) ∧
  f(1) = 2

theorem find_f2019 (f : ℝ → ℝ) (h : func_conditions f) : f 2019 = 2 :=
sorry

end find_f2019_l146_146222


namespace power_calculation_l146_146306

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146306


namespace total_plants_in_garden_l146_146628

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l146_146628


namespace power_of_three_l146_146286

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146286


namespace OB_value_l146_146232

-- Definitions
variables (O F₁ F₂ A B : ℝ × ℝ)
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Hypotheses
variable (hO : O = (0, 0))
variable (hF₁ : F₁ = (-√(4 - 3), 0))
variable (hF₂ : F₂ = (√(4 - 3), 0))
variable (hA : A ∈ set_of (λ p : ℝ × ℝ, ellipse p.1 p.2))
variable (hAF₂_perp : ∀ (s : ℝ), A = (s * (F₂.1 - O.1) / √((F₂.1 - O.1)^2 + (F₂.2 - O.2)^2) + O.1, s * (F₂.2 - O.2) / √((F₂.1 - O.1)^2 + (F₂.2 - O.2)^2) + O.2) → s ≠ 0)
variable (hAF₁y : ∃ (t : ℝ), F₁.1 + t * (A.1 - F₁.1) = 0 ∧ B = (0, F₁.2 + t * (A.2 - F₁.2)))

-- Goal
theorem OB_value : |(O.2 - B.2)| = 3/4 :=
sorry

end OB_value_l146_146232


namespace O_is_circumcenter_l146_146907

-- Definitions
variables (P A B C O : Point)
variables (plane_ABC : Plane)
variables (PO : Line)
variables (PA PB PC : ℝ)

-- Conditions
def point_outside_plane : P ∉ plane_ABC := sorry
def po_perpendicular_to_plane : PO ⊥ plane_ABC := sorry
def po_foot_O : foot PO = O := sorry
def pa_pb_pc_eq : PA = PB ∧ PB = PC := sorry

-- Theorem Statement
theorem O_is_circumcenter :
  point_outside_plane ∧ po_perpendicular_to_plane ∧ po_foot_O ∧ pa_pb_pc_eq → circumcenter O A B C := sorry

end O_is_circumcenter_l146_146907


namespace cyclist_speed_ratio_l146_146533

-- conditions: 
variables (T₁ T₂ o₁ o₂ : ℝ)
axiom h1 : o₁ + T₁ = o₂ + T₂
axiom h2 : T₁ = 2 * o₂
axiom h3 : T₂ = 4 * o₁

-- Proof statement to show that the second cyclist rides 1.5 times faster:
theorem cyclist_speed_ratio : T₁ / T₂ = 1.5 :=
by
  sorry

end cyclist_speed_ratio_l146_146533


namespace seashell_count_l146_146460

theorem seashell_count :
  ∀ (initial: ℕ) (given: ℕ), initial = 35 → given = 18 → initial - given = 17 :=
by
  intros initial given initial_condition given_condition
  rw [initial_condition, given_condition]
  rfl

end seashell_count_l146_146460


namespace expression_evaluation_l146_146639

def log10 (n : ℝ) : ℝ := 
  Real.log n / Real.log 10

theorem expression_evaluation : 
  log10 2 + log10 5 + (1 / 2)^(-2) = 5 := by
sorry

end expression_evaluation_l146_146639


namespace compare_abc_l146_146749

noncomputable def π : ℝ := real.pi
noncomputable def a : ℝ := π ^ 0.3
noncomputable def b : ℝ := real.log 3 / real.log π
noncomputable def c : ℝ := real.log (real.sin (2 * π / 3)) / real.log 3

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l146_146749


namespace exists_large_element_l146_146430

open Classical

variable {n : ℕ}
variable (a : Fin n → ℝ)

noncomputable def sequence_Ak : ℕ → (Fin n → ℝ)
| 0     => a
| (k+1) => let x := sequence_Ak k in
             let (I, J) := argmin (λ (part : (List (Fin n)) × (List (Fin n))), abs ((part.fst.map x).sum - (part.snd.map x).sum)) (partitions (Fin n))
             fun i => if i ∈ I.locals then x i + 1 else x i - 1

theorem exists_large_element :
    ∃ k x, x ∈ array.to_list (sequence_Ak k) ∧ abs x ≥ n / 2 := by 
   sorry

end exists_large_element_l146_146430


namespace monotonicity_interval_l146_146385

theorem monotonicity_interval (m : ℝ) (hm : 0 < m) :
  (∀ x ∈ set.Ioo m (2 * m + 1), monotone_on (λ x, |real.log x / real.log 2|) (set.Ioo m (2 * m + 1)) = false) ↔ 0 < m ∧ m < 1 :=
by
  sorry

end monotonicity_interval_l146_146385


namespace royalties_ratio_decrease_l146_146554

noncomputable def percentageDecrease (initial_ratio new_ratio : ℚ) : ℚ :=
  ((initial_ratio - new_ratio) / initial_ratio) * 100

theorem royalties_ratio_decrease :
  let r1 := (8/20 : ℚ) in
  let r2 := (9/108 : ℚ) in
  percentageDecrease r1 r2 ≈ 79.175 := 
by
  sorry

end royalties_ratio_decrease_l146_146554


namespace largest_multiple_of_15_under_500_l146_146995

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146995


namespace angle_between_vectors_l146_146236

open Real

def vector_angle_problem :
  ℝ × ℝ × (ℝ × ℝ) := 
  let a : ℝ × ℝ := (1, -sqrt 3)
  let b : ℝ × ℝ := (cos 𝜃, sin 𝜃)
  let mag_b := 1
  let mag_a_plus_2b := 2
  (1, -sqrt 3, (b.fst, b.snd))

theorem angle_between_vectors (a b : ℝ × ℝ) (ha : a = (1, -sqrt 3))
  (hb : ‖b‖ = 1) (hab_plus_2b : ‖(λ a b, (a.1 + 2 * b.1, a.2 + 2 * b.2)) a b‖ = 2) :
  let θ := 2 * π / 3 in
  θ = arccos ((a.1 * b.1 + a.2 * b.2) / (sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2)))) :=
by
  sorry

end angle_between_vectors_l146_146236


namespace student_marks_l146_146608

variables (x : ℕ)

theorem student_marks :
  (∀ total_marks pass_percentage fail_by max_marks,
    pass_percentage = 33 / 100 ∧ fail_by = 40 ∧ max_marks = 500 →
    (let passing_marks := pass_percentage * max_marks in
    x + fail_by = passing_marks → x = 125)) :=
begin
  sorry
end

end student_marks_l146_146608


namespace power_equality_l146_146324

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146324


namespace jill_spent_10_percent_on_food_l146_146444

noncomputable def jill_food_spent_percentage (T : ℝ) : ℝ :=
  let clothes := 0.50 * T
  let food := T * f
  let others := 0.40 * T
  let tax_clothes := 0.04 * clothes
  let tax_others := 0.08 * others
  let total_tax := tax_clothes + tax_others
  let expected_tax := 0.052 * T
  if total_tax = expected_tax then 10 else 0  -- the correct percentage if taxes match, otherwise 0

theorem jill_spent_10_percent_on_food (T : ℝ) (hT : T > 0) :
  jill_food_spent_percentage T = 10 :=
  by
  sorry  -- the detailed proof is omitted

end jill_spent_10_percent_on_food_l146_146444


namespace complex_number_z_l146_146817

-- Define the problem
theorem complex_number_z (z : ℂ) (h : z / (1 - complex.I) = complex.I) : 
  z = 1 + complex.I := 
sorry

end complex_number_z_l146_146817


namespace min_reciprocal_sum_l146_146248

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log (x + 2) / Real.log 3 else Real.exp x - 1

theorem min_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = f (f (Real.log 2))) :
  1 / m + 2 / n = 3 + 2 * Real.sqrt 2 := 
sorry

end min_reciprocal_sum_l146_146248


namespace cylinder_volume_l146_146070

theorem cylinder_volume (V1 V2 : ℝ) (π : ℝ) (r1 r3 h2 h5 : ℝ)
  (h_radii_ratio : r3 = 3 * r1)
  (h_heights_ratio : h5 = 5 / 2 * h2)
  (h_first_volume : V1 = π * r1^2 * h2)
  (h_V1_value : V1 = 40) :
  V2 = 900 :=
by sorry

end cylinder_volume_l146_146070


namespace binomial_coefficient_9_5_l146_146665

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146665


namespace find_k_and_profit_l146_146593

-- Define constants and functions as per given conditions
def sales_volume (m : ℕ) : ℝ := 3 - 2 / (m + 1 : ℝ)

def fixed_investment : ℝ := 80

def additional_investment (x : ℝ) : ℝ := 160 * x

def selling_price (x : ℝ) : ℝ := 1.5 * (8 + 16 * x) / x

def profit (m : ℕ) : ℝ :=
  let x := sales_volume m
  x * selling_price x - (8 + 16 * x + m)

-- The proof statements
theorem find_k_and_profit :
  (sales_volume 0 = 1 → true) ∧
  (- (16 / (m + 1 : ℝ) + (m + 1 : ℝ)) + 29 = profit m) ∧
  (m = 3 → profit m = 21) :=
by
  sorry

end find_k_and_profit_l146_146593


namespace triangle_is_isosceles_right_find_k_l146_146793

-- Definitions of points A, B, and C
structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)
def A : Point3D := ⟨-1, -1, 2⟩
def B : Point3D := ⟨0, 1, 0⟩
def C : Point3D := ⟨-2, 3, 1⟩

-- Vector between two points
def vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Definitions of vectors a, b, and c
def a : Point3D := vector B A
def b : Point3D := vector B C
def c : Point3D := vector A C

-- Length of a vector
def vector_length (v : Point3D) : ℝ :=
  real.sqrt (v.x^2 + v.y^2 + v.z^2)

-- Definitions of vector lengths
def length_a := vector_length a
def length_b := vector_length b
def length_c := vector_length c

-- Proof that triangle ABC is an isosceles right triangle
theorem triangle_is_isosceles_right :
  length_a^2 + length_b^2 = length_c^2 :=
sorry

-- Proof of the value of k
theorem find_k (k : ℝ) :
  (-2 * vector B A + k * vector B C).x / c.x = 
  (-2 * vector B A + k * vector B C).y / c.y ∧
  (-2 * vector B A + k * vector B C).y / c.y = 
  (-2 * vector B A + k * vector B C).z / c.z →
  k = 2 :=
sorry

end triangle_is_isosceles_right_find_k_l146_146793


namespace vector_perpendicular_solve_x_l146_146213

theorem vector_perpendicular_solve_x
  (x : ℝ)
  (a : ℝ × ℝ := (4, 8))
  (b : ℝ × ℝ := (x, 4))
  (h : 4 * x + 8 * 4 = 0) :
  x = -8 :=
sorry

end vector_perpendicular_solve_x_l146_146213


namespace real_number_condition_pure_imaginary_condition_simplification_l146_146244

noncomputable def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * (Complex.i)

theorem real_number_condition (m : ℝ) : (z m).im = 0 ↔ (m = 1 ∨ m = 2) := by
  sorry

theorem pure_imaginary_condition (m : ℝ) : (z m).re = 0 ↔ (m = -1/2) := by
  sorry

theorem simplification (m : ℝ) (h : m = 0) : (z m * z m) / (z m + 5 + 2 * Complex.i) = (-32 / 25 : ℝ) + (-24 / 25 : ℝ) * (Complex.i) := by
  have : z 0 = -2 + 2 * Complex.i := by
    sorry
  rw [h]
  calc
    (z 0 * z 0 : ℂ) / (z 0 + 5 + 2 * Complex.i)
        = sorry := by sorry
    ... = (-32 / 25 : ℝ) + (-24 / 25 : ℝ) * (Complex.i) := by sorry

end real_number_condition_pure_imaginary_condition_simplification_l146_146244


namespace sum_of_squares_nonnegative_l146_146451

theorem sum_of_squares_nonnegative (x y z : ℝ) : x^2 + y^2 + z^2 - x * y - x * z - y * z ≥ 0 :=
  sorry

end sum_of_squares_nonnegative_l146_146451


namespace binom_9_5_eq_126_l146_146664

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146664


namespace determinant_sin_eq_zero_l146_146654

theorem determinant_sin_eq_zero (a b : ℝ) : 
  matrix.det ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by
  sorry

end determinant_sin_eq_zero_l146_146654


namespace log_expression_result_l146_146155

noncomputable def log := Real.logBase 10

theorem log_expression_result :
  (log 2)^2 + (log 5)^2 + log 4 * log 5 = 1 := 
by
  -- proof is omitted
  sorry

end log_expression_result_l146_146155


namespace equal_tangent_segments_and_value_l146_146768

variable (a b c : ℝ) (A B C D2 E2 O O2 : Point)
variable (CA AB BC AD2 AE2 : LineSegment)
variable [Triangle ABC]
variable [Circle O]
variable [Excircle O2]

-- Given conditions
def circle_tangency_external (O O2 : Circle) (C : Point) :=
  ∃ (tangent_point : Point), Circle.is_tangent O O2 tangent_point

def tangency_extension (O2 : Circle) (AC AB : LineSegment) (D2 E2 : Point) :=
  Circle.is_tangent O2 (LineSegment.extend AC) D2 ∧
  Circle.is_tangent O2 (LineSegment.extend AB) E2

-- The statement to be proven
theorem equal_tangent_segments_and_value {a b c : ℝ} 
  (circle_tangent : circle_tangency_external O O2 A)
  (tangent_at_extensions : tangency_extension O2 CA AB D2 E2)
  (side_lengths : CA = b ∧ AB = c ∧ BC = a)
  : AD2 = AE2 ∧ AE2 = (2 * b * c) / (b + c - a) :=
sorry

end equal_tangent_segments_and_value_l146_146768


namespace sum_b_first_n_terms_l146_146490

-- Define the sequence {a_n} with the given formula
def a (n : ℕ) : ℕ := 4 * n - 1

-- Define the sequence {b_n} using the sum of {a_n}
def b (n : ℕ) : ℕ := (finset.range n).sum (λ i, a (i + 1)) / n

-- Define the sum of the first n terms of the sequence {b_n}
noncomputable def sum_b (n : ℕ) : ℕ := (finset.range n).sum (λ i, b (i + 1))

-- Prove that the sum of the first n terms of the sequence {b_n} is n² + 2n
theorem sum_b_first_n_terms (n : ℕ) : sum_b n = n^2 + 2 * n := by
  sorry

end sum_b_first_n_terms_l146_146490


namespace min_shift_right_l146_146781

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 3) * real.sin x + 3 * real.cos x

theorem min_shift_right (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, f (x - m) = f (-x + m)) → m = (5 * real.pi) / 6 :=
by
  sorry

end min_shift_right_l146_146781


namespace product_divisible_by_3_probability_divisible_by_3_l146_146616

theorem product_divisible_by_3 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 
  3 ∣ n * (n + 1) := 
sorry

theorem probability_divisible_by_3 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 
  ∑ x in finset.range 99, if 3 ∣ x * (x + 1) then 1 else 0 = 99 :=
sorry

end product_divisible_by_3_probability_divisible_by_3_l146_146616


namespace mr_green_yield_l146_146897

noncomputable def steps_to_feet (steps : ℕ) : ℝ :=
  steps * 2.5

noncomputable def total_yield (steps_x : ℕ) (steps_y : ℕ) (yield_potato_per_sqft : ℝ) (yield_carrot_per_sqft : ℝ) : ℝ :=
  let width := steps_to_feet steps_x
  let height := steps_to_feet steps_y
  let area := width * height
  (area * yield_potato_per_sqft) + (area * yield_carrot_per_sqft)

theorem mr_green_yield :
  total_yield 20 25 0.5 0.25 = 2343.75 :=
by
  sorry

end mr_green_yield_l146_146897


namespace circle_point_l146_146208

variable {t : ℝ}

def x (t : ℝ) : ℝ := (1 - t^2) / (1 + t^2)
def y (t : ℝ) : ℝ := (2 * t) / (1 + t^2)

theorem circle_point : (x t)^2 + (y t)^2 = 1 := by
  sorry

end circle_point_l146_146208


namespace power_of_three_l146_146368

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146368


namespace find_angle_x_l146_146843

variable (k ell : Type) [ParallelLines k ell]
variable (angle1 angle2 angle3 : ℝ)
variable (x : ℝ)

axiom given_angles : angle1 = 40 ∧ angle2 = 50 ∧ angle3 = 90

theorem find_angle_x (parallel : ParallelLines k ell) : x = 90 := by
  have : angle1 + angle2 + angle3 = 180 := sorry
  have : angle1 + angle2 = 90 := by linarith [given_angles]
  have : x + 90 = 180 := sorry  -- External angle property
  exact_mod_cast sorry

end find_angle_x_l146_146843


namespace production_difference_l146_146905

variables (p h : ℕ)

def first_day_production := p * h

def second_day_production := (p + 5) * (h - 3)

-- Given condition
axiom p_eq_3h : p = 3 * h

theorem production_difference : first_day_production p h - second_day_production p h = 4 * h + 15 :=
by
  sorry

end production_difference_l146_146905


namespace min_sum_hex_in_consecutive_nums_l146_146948

-- Definitions
def consecutive_natural_numbers (s : Set ℕ) : Prop :=
  ∃ x, (s = {y | y ∈ Finset.range 11 ∧ ∃ n, y = x + n})

def sum_of_extremes (s : Set ℕ) (v : ℕ) : Prop :=
  v = Finset.min' s (by sorry) + Finset.max 's (by sorry)

def min_possible_hexagon_sum (s : Set ℕ) (v : ℕ) : Prop :=
  ∀ config, config ∈ possible_configurations s → get_hexagon_sum config ≥ v 

-- Proposition
theorem min_sum_hex_in_consecutive_nums :
  ∀ s : Set ℕ, consecutive_natural_numbers s → sum_of_extremes s 90 → min_possible_hexagon_sum s 90 := sorry

end min_sum_hex_in_consecutive_nums_l146_146948


namespace probability_exactly_half_red_balls_l146_146172

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_half_red_balls :
  binomial_probability 8 4 (1/2) = 35/128 :=
by
  sorry

end probability_exactly_half_red_balls_l146_146172


namespace greatest_whole_number_satisfying_inequality_l146_146713

theorem greatest_whole_number_satisfying_inequality :
  ∀ (x : ℤ), 3 * x + 2 < 5 - 2 * x → x <= 0 :=
by
  sorry

end greatest_whole_number_satisfying_inequality_l146_146713


namespace winning_team_possible_scores_count_l146_146967

-- Definitions of the conditions
def totalScoresSum : ℕ := 55
def minWinScore : ℕ := (List.range' 1 5).sum
def maxWinScore : ℕ := (List.range' 6 5).sum

-- Main property
theorem winning_team_possible_scores_count : 
  ∃ n : ℕ, n = 13 ∧ 
  ∀ k ∈ (List.range' minWinScore (maxWinScore - minWinScore + 1)), 
    k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 ∨ k = 21 ∨ 
    k = 22 ∨ k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 ∨ k = 27 :=
by {
  have totsum := totalScoresSum,
  have minScore := minWinScore,
  have maxScore := maxWinScore,
  use 13,
  split,
  exact rfl,
  intro k,
  intro h,
  simp at h,
  interval_cases k; simp
}

end winning_team_possible_scores_count_l146_146967


namespace tan_double_angle_l146_146378

theorem tan_double_angle (α : ℝ) (h : sin α + 2 * cos α = 0) : tan (2 * α) = -4 / 3 :=
by
  sorry

end tan_double_angle_l146_146378


namespace junior_score_is_92_l146_146829

theorem junior_score_is_92 (n : ℕ) 
  (h1 : 0.2 * n = 0.2 * (n : ℕ)) 
  (h2 : 0.8 * n = 0.8 * (n : ℕ)) 
  (h3 : total_class_score = 84 * n) 
  (h4 : total_senior_score = 82 * 0.8 * n) 
  (h5 : total_junior_score = total_class_score - total_senior_score) 
  (h6 : total_junior_score / (0.2 * n) = junior_score) : 
  junior_score = 92 := 
by sorry

end junior_score_is_92_l146_146829


namespace power_of_3_l146_146345

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l146_146345


namespace power_of_three_l146_146363

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146363


namespace converse_is_false_contrapositive_is_true_l146_146788

-- Definitions based on the conditions
def rectangle (q : Type) [quadrilateral q] : Prop :=
∃ (ab cd : ℝ), is_diagonal q ab ∧ is_diagonal q cd ∧ ab = cd

-- The proposition: "The diagonals of a rectangle are of equal length"
def prop (q : Type) [quadrilateral q] : Prop :=
∀ (q : rectangle q), q.diagonal_length = q.other_diagonal_length

-- Converse: If a quadrilateral has diagonals of equal length, then it is a rectangle.
def converse (q : Type) [quadrilateral q] : Prop :=
∀ (q : Type) [quadrilateral q], diagonal_length q = other_diagonal_length q → is_rectangle q

-- Contrapositive: If a quadrilateral does not have diagonals of equal length, then it is not a rectangle.
def contrapositive (q : Type) [quadrilateral q] : Prop :=
∀ (q : Type) [quadrilateral q], diagonal_length q ≠ other_diagonal_length q → ¬ is_rectangle q

-- The Lean statements to be proven:
theorem converse_is_false (q : Type) [quadrilateral q] : ¬ converse q :=
by sorry

theorem contrapositive_is_true (q : Type) [quadrilateral q] : contrapositive q :=
by sorry

end converse_is_false_contrapositive_is_true_l146_146788


namespace intersection_planes_l146_146501

variables {R : ℝ}
variables {a b c d : EuclideanSpace ℝ}
variables {x : EuclideanSpace ℝ}

-- Conditions on the vectors based on the circumscribed sphere
axiom a_squared : ∥a∥^2 = R^2
axiom b_squared : ∥b∥^2 = R^2
axiom c_squared : ∥c∥^2 = R^2
axiom d_squared : ∥d∥^2 = R^2

-- Equations of the planes
def plane_α (x : EuclideanSpace ℝ) : Prop := a.dot x = R^2
def plane_β (x : EuclideanSpace ℝ) : Prop := b.dot x = R^2
def plane_γ (x : EuclideanSpace ℝ) : Prop := c.dot x = R^2
def plane_δ (x : EuclideanSpace ℝ) : Prop := d.dot x = R^2

-- Theorem statement
theorem intersection_planes
  (h1 : (a.dot c - R^2) * (b.dot d - R^2) = (a.dot d - R^2) * (b.dot c - R^2)) :
  (c.dot a - R^2) * (d.dot b - R^2) = (c.dot b - R^2) * (d.dot a - R^2) :=
sorry

end intersection_planes_l146_146501


namespace constant_term_in_binomial_expansion_is_five_l146_146844

theorem constant_term_in_binomial_expansion_is_five : 
  let f := λ x : ℝ, (x^2 + 1/ x.sqrt) in
  let term := (f 5)^5 in
  ∃ C : ℝ, ∀ x : ℝ, C = 5 := 
by
  sorry

end constant_term_in_binomial_expansion_is_five_l146_146844


namespace teacher_estimate_difference_l146_146492

theorem teacher_estimate_difference :
  let expected_increase := 2152
  let actual_increase := 1264
  let difference := expected_increase - actual_increase
  difference = 888 :=
by
  let expected_increase := 2152
  let actual_increase := 1264
  let difference := expected_increase - actual_increase
  show difference = 888 from sorry

end teacher_estimate_difference_l146_146492


namespace find_a₁₀_l146_146835

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a₁ a₁₉ : ℤ)

noncomputable def arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a₁₀ (h : arithmetic_sequence a)
                 (ha₁ : a 1 = a₁)
                 (ha₁₉ : a 19 = a₁₉)
                 (h_sum : a₁ + a₁₉ = -18) :
  a 10 = -9 :=
by
  sorry 

end find_a₁₀_l146_146835


namespace problem_statement_l146_146250

noncomputable def f (x φ : ℝ) : ℝ :=
  2 * Real.sin ((x + φ) / 2) * Real.cos ((x + φ) / 2)

theorem problem_statement (φ : ℝ) (hφ1 : |φ| < Real.pi / 2) :
  (∀ x : ℝ, f x φ ≤ f (Real.pi / 6) φ) →
  ∀ x : ℝ, f x (Real.pi / 3) = f (Real.pi / 3 - x) :=
by
  sorry

end problem_statement_l146_146250


namespace extremum_at_x_and_value_of_a_f_diff_bound_f_diff_l146_146215

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 2) * Real.exp x

theorem extremum_at_x_and_value_of_a :
  (∀ x : ℝ, (f a)'(1) = 0 → x = 1) → a = 1 := sorry

theorem f_diff (a : ℝ) (h_a : a = 1) (x : ℝ) : (f a)'(x) = (x - 1) * Real.exp x :=
by {
  rw h_a,
  exact diff_as_f'(x),
  sorry
}

theorem bound_f_diff (h_a : a = 1) (x1 x2 : ℝ) (h_x1 : 0 ≤ x1 ∧ x1 ≤ 2) (h_x2 : 0 ≤ x2 ∧ x2 ≤ 2):
  (f a x1 - f a x2 ≤ by {
    rw h_a,
    exact 0 - (-Real.exp 1)
    sorry
  } := sorry

end extremum_at_x_and_value_of_a_f_diff_bound_f_diff_l146_146215


namespace yellow_flowers_count_l146_146915

theorem yellow_flowers_count (red_flowers : ℕ) (bouquets : ℕ) (same_count_each_bouquet : Prop) (h1 : red_flowers = 16) (h2 : bouquets = 8) (h3 : same_count_each_bouquet = true) :
  ∃ yellow_flowers : ℕ, yellow_flowers = 16 :=
by
  have yellow_flowers := (red_flowers / bouquets) * bouquets
  use yellow_flowers
  simp [h1, h2]
  sorry

end yellow_flowers_count_l146_146915


namespace power_calculation_l146_146312

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146312


namespace part_I_part_II_l146_146784

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.log x - b * x^2

theorem part_I (a b : ℝ) :
  (f 1 a b = -1 / 2) ∧
  ((∃ a3 : ℝ, f.a - 2 * b = 0) ->
  (a = 1) ∧ (b = 1 / 2)) := by
  sorry

theorem part_II :
  let a := 1
  let b := 1 / 2
  f x a b is increasing on \left( frac {1}{e},1\right)
  f x a b is decreasing on \left(1,e \right)
  f(1) = - 1 / 2
  maximum_value_of_f_on_interval \left[ \frac{1}{e},e\right] is - 1 / 2 := by
  sorry


end part_I_part_II_l146_146784


namespace power_of_three_l146_146276

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146276


namespace polynomial_divisible_by_x_plus_y_plus_z_l146_146452

open Polynomial

noncomputable def polynomial_divisibility_statement : Prop :=
  ∀ (R : Type) [CommRing R] (x y z : R), 
    ∃ Q : R[X][Y][Z], 
      (X^3 + Y^3 + Z^3 - 3 * X * Y * Z) = (X + Y + Z) * Q

theorem polynomial_divisible_by_x_plus_y_plus_z : polynomial_divisibility_statement :=
by
  sorry

end polynomial_divisible_by_x_plus_y_plus_z_l146_146452


namespace fixed_point_locus_l146_146269

theorem fixed_point_locus (A B C D : ℝ²) (hABC_collinear : collinear ({A, B, C} : set ℝ²)) (hD_not_collinear : ¬ collinear ({A, B, D} : set ℝ²)) :
  ∃ L : ℝ², ∃ l : Line ℝ², ∃ M : ℝ²,
  (∀ D', ¬ collinear ({A, B, D'} : set ℝ²) →
  (let P := intersection (parallel_through (line_through A D) C) (line_through B D) in
   let Q := intersection (parallel_through (line_through B D) C) (line_through A D) in
   let PQ := line_through P Q in
   let M := foot_of_perpendicular C PQ in
   M ∈ l))
  ∧ (∀ D', ¬ collinear ({A, B, D'} : set ℝ²) →
  (let P := intersection (parallel_through (line_through A D') C) (line_through B D') in
   let Q := intersection (parallel_through (line_through B D') C) (line_through A D') in
   let PQ := line_through P Q in
   let M := foot_of_perpendicular C PQ in
   M = L)) :=
sorry

end fixed_point_locus_l146_146269


namespace original_number_sweets_correct_l146_146570

-- Definition of the problem conditions
def num_children : ℕ := 48
def sweets_per_child : ℕ := 4
def fraction_remaining : ℚ := 1 / 3

-- Definition of the total number of sweets taken by children
def total_sweets_taken : ℕ := num_children * sweets_per_child

-- Original number of sweets in the pack
def original_number_sweets (total_sweets_taken : ℕ) (fraction_remaining : ℚ) : ℕ :=
  (total_sweets_taken * (1 / (1 - fraction_remaining))).to_nat

-- Assert the result
theorem original_number_sweets_correct :
  original_number_sweets total_sweets_taken fraction_remaining = 288 :=
by sorry

end original_number_sweets_correct_l146_146570


namespace range_of_a_l146_146886

def prop_p (a : ℝ) : Prop :=
  ∀ x, sqrt 2 < x ∧ x < 2 → x + 2 / x > a

theorem range_of_a :
  (¬ prop_p a → False) → a ≤ 2 * sqrt 2 :=
by
  intro h
  have p_true : prop_p a := by contrapose! h with hp; exact id hp
  sorry

end range_of_a_l146_146886


namespace three_half_planes_suffice_l146_146618

noncomputable def half_planes_cover (H1 H2 H3 H4 : set (ℝ × ℝ)) : Prop :=
  (H1 ∪ H2 ∪ H3 ∪ H4) = set.univ

theorem three_half_planes_suffice (H1 H2 H3 H4 : set (ℝ × ℝ)) 
  (cover: half_planes_cover H1 H2 H3 H4) : 
  ∃ (Hi Hj Hk : set (ℝ × ℝ)), {Hi, Hj, Hk} ⊆ {H1, H2, H3, H4} ∧ (Hi ∪ Hj ∪ Hk) = set.univ :=
sorry

end three_half_planes_suffice_l146_146618


namespace coefficient_of_x_squared_in_expansion_l146_146032

theorem coefficient_of_x_squared_in_expansion (a : ℝ) 
    (h : (∃ r : ℕ, r = 4 ∧ (choose 8 r) * (-1)^r * a^(8-r) * (70 : ℝ) = 70)) :
    a = 1 ∨ a = -1 :=
by sorry

end coefficient_of_x_squared_in_expansion_l146_146032


namespace John_age_is_39_l146_146414

-- Definitions for conditions
variable (John_age_now James_age : ℕ)

-- Condition 1: James' older brother is 16 years old
def James_brother_age : ℕ := 16

-- Condition 2: James' older brother is 4 years older than James
def age_difference_between_brothers := James_brother_age - James_age = 4

-- Condition 3: 3 years ago, John was twice as old as James will be in 6 years
def John_age_3_years_ago := John_age_now - 3
def James_age_in_6_years := James_age + 6
def John_double_James_6_years := John_age_3_years_ago = 2 * James_age_in_6_years

-- Proof statement
theorem John_age_is_39 :
  age_difference_between_brothers James_age →
  John_double_James_6_years John_age_now James_age →
  John_age_now = 39 :=
by
  intros
  sorry

end John_age_is_39_l146_146414


namespace cot_squared_inequality_tan_squared_inequality_l146_146408

variable {A B C n : ℝ}

-- Helper lemma for the sum of angles in a triangle
lemma angle_sum_pi (h : A + B + C = Real.pi) : A + B + C = Real.pi := h

-- Part 1
theorem cot_squared_inequality 
  (h : A + B + C = Real.pi) 
  (hn : n ∈ ℤ) : 
  (Real.cot (n * A))^2 + (Real.cot (n * B))^2 + (Real.cot (n * C))^2 ≥ 1 :=
by sorry

-- Part 2
theorem tan_squared_inequality 
  (h : A + B + C = Real.pi) 
  (hn : n ∈ ℤ) 
  (odd_n : Odd n) : 
  (Real.tan (n * A / 2))^2 + (Real.tan (n * B / 2))^2 + (Real.tan (n * C / 2))^2 ≥ 1 :=
by sorry

end cot_squared_inequality_tan_squared_inequality_l146_146408


namespace compute_combination_l146_146691

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146691


namespace bananas_more_than_apples_l146_146640

variable (x y : ℕ)

-- Define the condition based on the problem
def condition1 : Prop := 0.4 * x + 0.5 * y = 0.5 * x + 0.4 * y + 1

-- State the theorem to be proved
theorem bananas_more_than_apples (h : condition1 x y) : y = x + 10 :=
by
  sorry

end bananas_more_than_apples_l146_146640


namespace original_cost_of_car_l146_146913

theorem original_cost_of_car (C : ℝ)
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (h1 : repairs_cost = 14000)
  (h2 : selling_price = 72900)
  (h3 : profit_percent = 17.580645161290324)
  (h4 : profit_percent = ((selling_price - (C + repairs_cost)) / C) * 100) :
  C = 50075 := 
sorry

end original_cost_of_car_l146_146913


namespace union_of_A_B_l146_146792

open Set

variable U : Set ℝ
variable A B : Set ℝ
variable p q : ℝ
def complement (S : Set ℝ) := {x | x ∉ S}

theorem union_of_A_B :
  U = univ ∧
  A = {x | x^2 + p * x + 12 = 0} ∧
  B = {x | x^2 - 5 * x + q = 0} ∧
  (complement A) ∩ B = {2} ∧
  A ∩ (complement B) = {4} →
  A ∪ B = {2, 3, 6} := 
by 
  sorry

end union_of_A_B_l146_146792


namespace cosine_angle_BHD_l146_146397

variables {α : Type*} [linear_ordered_field α]

noncomputable def cos_BHD (BD BH DH : α) : α :=
  (1 - (BD^2 + BH^2 - DH^2) / (2 * BD * BH))

theorem cosine_angle_BHD :
  let D: α := 1
  let CD: α := 1
  let DG: α := 1 / 2
  let HG: α := ( √3 / 2 )
  ∃ (BD BH DH : ℝ), 
    ∀(conditions : BD ^ 2 = ( √6 / 2) ^ 2 + D ^ 2 ∧ 
                  BH ^ 2 = ( √3 / 2 ) ^ 2 + 1 ^ 2),
    cos_BHD BD BH DH = - √30 / 12 := 
    sorry

end cosine_angle_BHD_l146_146397


namespace ben_winning_strategy_l146_146950

-- Define the initial setting
def cards : ℕ := 2019

-- Define a card with two sides
inductive Side
| Number : Side
| Letter : Side
deriving DecidableEq

-- Define the condition for flipping a card and its neighbors.
def flip_neighbors (cards : List Side) (index : ℕ) : List Side :=
  cards.zipWith (λ i side, if i = index ∨ i = index + 1 ∨ (i ≠ 0 ∧ i = index - 1) then
                          match side with
                          | Side.Number => Side.Letter
                          | Side.Letter => Side.Number
                          else side)
               (List.range cards.length)

-- Define the game conditions
structure Game :=
  (initial : List Side) -- initial card configuration with sides
  (flip : List Side → ℕ → List Side) -- flipping mechanism
  (touches : ℕ) -- maximum touches allowed for Ben

-- Define the problem as a proposition
theorem ben_winning_strategy :
  ∃ (g : Game), g.touches = cards ∧ 
  ∀ (config : List Side), (∀ i, i < cards → (config.get i = Side.Number ∨ config.get i = Side.Letter)) →
  ∃ t, t ≤ g.touches ∧ all_numbers (g.flip config t) 
  :=
by
  sorry

-- Function to check if all cards show numbers
def all_numbers (cards : List Side) : Prop :=
  ∀ i, i < cards.length → cards.get i = Side.Number

end ben_winning_strategy_l146_146950


namespace probability_of_two_white_balls_l146_146063

noncomputable def probability_two_white_balls (n : ℕ) :=
  let v1 := (Nat.choose (2 * n) n) * (1/4) ^ n * (3/4) ^ n,
      v2 := (Nat.choose (2 * n) n) * (1/2) ^ n * (1/2) ^ n,
      v3 := (Nat.choose (2 * n) n) * (3/4) ^ n * (1/4) ^ n in
  v2 / (v1 + v2 + v3)

theorem probability_of_two_white_balls (n : ℕ) :
  probability_two_white_balls n = (4 ^ n) / (2 * (3 ^ n) + 4 ^ n) :=
sorry

end probability_of_two_white_balls_l146_146063


namespace value_of_a_l146_146744

theorem value_of_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
by
  sorry

end value_of_a_l146_146744


namespace determinant_simplifies_to_zero_l146_146647

theorem determinant_simplifies_to_zero (a b : ℝ) :
  matrix.det ![
    ![1, real.sin (a - b), real.sin a],
    ![real.sin (a - b), 1, real.sin b],
    ![real.sin a, real.sin b, 1]
  ] = 0 := 
by
  sorry

end determinant_simplifies_to_zero_l146_146647


namespace power_equality_l146_146326

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146326


namespace determinant_identity_l146_146644

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l146_146644


namespace min_value_PF1_PF2_value_k_square_l146_146756

variable {x y : ℝ}

def ellipse_condition (P : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  (P.1^2 / 2 + P.2^2 = 1) ∧ (Q.1^2 / 2 + Q.2^2 = 1)

def perpendicular_PF2_QF2 (P : ℝ × ℝ) (Q : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  let PF2 := (F2.1 - P.1, F2.2 - P.2)
  let QF2 := (F2.1 - Q.1, F2.2 - Q.2)
  PF2.1 * QF2.1 + PF2.2 * QF2.2 = 0

def min_value_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  ∥((F1.1 - P.1) + (F2.1 - P.1), (F1.2 - P.2) + (F2.2 - P.2))∥ = 2

def perpendicular_PQ_F1F2_sum (P Q F1 F2 : ℝ × ℝ) : Prop :=
  let PF1 := (F1.1 - P.1, F1.2 - P.2)
  let PF2 := (F2.1 - P.1, F2.2 - P.2)
  let QF1 := (F1.1 - Q.1, F1.2 - Q.2)
  let QF2 := (F2.1 - Q.1, F2.2 - Q.2)
  (PF1.1 + PF2.1) * (QF1.1 + QF2.1) + (PF1.2 + PF2.2) * (QF1.2 + QF2.2) = 0

theorem min_value_PF1_PF2 (P Q F1 F2 : ℝ × ℝ) (h_ellipse : ellipse_condition P Q) (h_perpendicular : perpendicular_PF2_QF2 P Q F1 F2) : 
  min_value_condition P F1 F2 :=
  sorry

theorem value_k_square (P Q F1 F2 : ℝ × ℝ) (k : ℝ) (h_ellipse : ellipse_condition P Q) (h_perpendicular_PF2_QF2 : perpendicular_PF2_QF2 P Q F1 F2) (h_perpendicular_PQ_F1F2 : perpendicular_PQ_F1F2_sum P Q F1 F2) :
  k^2 = ((-5 + 2 * real.sqrt 10) / 10) :=
  sorry

end min_value_PF1_PF2_value_k_square_l146_146756


namespace positive_real_numbers_exist_l146_146417

theorem positive_real_numbers_exist
  (a : Fin 3 → Fin 3 → ℝ)
  (h_pos_diag : ∀ i : Fin 3, a i i > 0)
  (h_neg_off_diag : ∀ i j : Fin 3, i ≠ j → a i j < 0) :
  ∃ (c : Fin 3 → ℝ), (∀ i, c i > 0) ∧
    (let
      R1 := a 0 0 * c 0 + a 0 1 * c 1 + a 0 2 * c 2,
      R2 := a 1 0 * c 0 + a 1 1 * c 1 + a 1 2 * c 2,
      R3 := a 2 0 * c 0 + a 2 1 * c 1 + a 2 2 * c 2
    in (R1 > 0 ∧ R2 > 0 ∧ R3 > 0) ∨ (R1 < 0 ∧ R2 < 0 ∧ R3 < 0) ∨ (R1 = 0 ∧ R2 = 0 ∧ R3 = 0)) :=
by
  sorry

end positive_real_numbers_exist_l146_146417


namespace distance_between_A_and_B_l146_146740

theorem distance_between_A_and_B
  (isosceles_triangle : ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], 
    ∃ (angle_opposite_base : ℝ) (lateral_side_length : ℝ), 
    angle_opposite_base = 45 ∧ lateral_side_length = 1) : 
  dist A B = 2 :=
sorry

end distance_between_A_and_B_l146_146740


namespace prime_large_factor_l146_146202

theorem prime_large_factor (p : ℕ) (hp : Nat.Prime p) (hp_ge_3 : p ≥ 3) (x : ℕ) (hx_large : ∃ N, x ≥ N) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ (p + 3) / 2 ∧ (∃ q : ℕ, Nat.Prime q ∧ q > p ∧ q ∣ (x + i)) := by
  sorry

end prime_large_factor_l146_146202


namespace impossible_prime_decomposition_l146_146487

noncomputable def num_consecutive_digits := 2021

def is_base_5 (n : ℕ) : Prop :=
  ∀ d ∈ digits 5 n, d < 5

def is_compatible_sequence (s : List ℕ) : Prop :=
  List.length s = num_consecutive_digits ∧
  ∀ i, i < List.length s → (
    (i % 2 = 0 → s[i] = s[i+1] + 1) ∧
    (i % 2 = 1 → s[i] = s[i-1] - 1))

def large_number_from_digits (s : List ℕ) : ℕ :=
  s.foldl (λ acc d, acc * 5 + d) 0

theorem impossible_prime_decomposition :
  ∀ s : List ℕ, is_compatible_sequence s →
  let x := large_number_from_digits s in
  ∀ u v : ℕ, Prime u → Prime v → u ≠ v →
  u * v = x → (v = u + 2) → False :=
begin
  intros s hs x u v hu hv hne huv hvu,
  -- proof goes here
  sorry
end

end impossible_prime_decomposition_l146_146487


namespace side_length_12th_square_l146_146928

def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 1
| 3     := 3
| 4     := 4
| 5     := 7
| (n+6) := a (n+4) + a (n+5)

theorem side_length_12th_square :
  a 11 = 123 :=
sorry

end side_length_12th_square_l146_146928


namespace roots_ratio_quadratic_eq_l146_146732

theorem roots_ratio_quadratic_eq {k r s : ℝ} 
(h_eq : ∃ a b : ℝ, a * r = b * s) 
(ratio_3_2 : ∃ t : ℝ, r = 3 * t ∧ s = 2 * t) 
(eqn : r + s = -10 ∧ r * s = k) : 
k = 24 := 
sorry

end roots_ratio_quadratic_eq_l146_146732


namespace proof_inequality_l146_146866

theorem proof_inequality (n : ℕ) (a b : ℝ) (c : ℝ) (h_n : 1 ≤ n) (h_a : 1 ≤ a) (h_b : 1 ≤ b) (h_c : 0 < c) : 
  ((ab + c)^n - c) / ((b + c)^n - c) ≤ a^n :=
sorry

end proof_inequality_l146_146866


namespace nonneg_int_solutions_eq_174_l146_146941

theorem nonneg_int_solutions_eq_174 :
  ∃ (x : Fin 10 → ℕ), (x 0 + 2 * x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 = 3) ∧
  (0 ≤ x 0) ∧ (0 ≤ x 1) ∧ (0 ≤ x 2) ∧ (0 ≤ x 3) ∧ (0 ≤ x 4) ∧ (0 ≤ x 5) ∧ (0 ≤ x 6) ∧ (0 ≤ x 7) ∧ (0 ≤ x 8) ∧ (0 ≤ x 9) :=
begin
  sorry
end

end nonneg_int_solutions_eq_174_l146_146941


namespace jack_keeps_deer_weight_l146_146853

-- Conditions
def hunting_trips_per_month : Int := 6
def hunting_season_months : Int := 3
def deer_per_trip : Int := 2
def weight_per_deer : Int := 600
def fraction_kept : Rational := 1 / 2

-- Derived values
noncomputable def hunting_trips_per_season : Int := hunting_trips_per_month * hunting_season_months
noncomputable def weight_per_trip : Int := deer_per_trip * weight_per_deer
noncomputable def total_weight_per_season : Int := weight_per_trip * hunting_trips_per_season
noncomputable def weight_kept_per_year : Int := total_weight_per_season * (fraction_kept.num.toInt / fraction_kept.denom.toInt) -- assuming Rational conversion to Int

-- Theorem statement
theorem jack_keeps_deer_weight : weight_kept_per_year = 10800 := by
  sorry

end jack_keeps_deer_weight_l146_146853


namespace power_of_three_l146_146282

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146282


namespace determinant_identity_l146_146642

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l146_146642


namespace probability_china_dream_l146_146953

theorem probability_china_dream (cards : fin 5 → string)
  (h1 : ∃ i j, (i ≠ j) ∧ (cards i = "中") ∧ (cards j = "中"))
  (h2 : ∃ i j, (i ≠ j) ∧ (cards i = "国") ∧ (cards j = "国"))
  (h3 : ∃ i, cards i = "梦") :
  let total_draws := 10 in
  let favorable_draws := 4 in
  let probability := (favorable_draws : ℚ) / total_draws in
  probability = 2 / 5 := by
    sorry

end probability_china_dream_l146_146953


namespace find_N_l146_146729

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := λ i j, 
  match i, j with
  | 0, 0 => 2
  | 0, 1 => -5
  | 1, 0 => 4
  | 1, 1 => -3
  | _, _ => 0

noncomputable def Resultant : Matrix (Fin 2) (Fin 2) ℚ := λ i j, 
  match i, j with
  | 0, 0 => -21
  | 0, 1 => -2
  | 1, 0 => 13
  | 1, 1 => 1
  | _, _ => 0

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := λ i j, 
  match i, j with
  | 0, 0 => 71 / 14
  | 0, 1 => -109 / 14
  | 1, 0 => -43 / 14
  | 1, 1 => 67 / 14
  | _, _ => 0

theorem find_N : N ⬝ A = Resultant :=
by
  sorry

end find_N_l146_146729


namespace find_a_2018_l146_146223

noncomputable def a : ℕ → ℕ
| n => if n > 0 then 2 * n else sorry

theorem find_a_2018 (a : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 ∧ n > 0 → a m + a n = a (m + n)) 
  (h1 : a 1 = 2) : a 2018 = 4036 := by
  sorry

end find_a_2018_l146_146223


namespace single_reduction_equivalent_l146_146558

theorem single_reduction_equivalent (P : ℝ) (P_pos : 0 < P) : 
  (P - (P - 0.30 * P)) / P = 0.70 := 
by
  -- Let's denote the original price by P, 
  -- apply first 25% and then 60% reduction 
  -- and show that it's equivalent to a single 70% reduction
  sorry

end single_reduction_equivalent_l146_146558


namespace car_original_cost_l146_146592

noncomputable def original_cost_price (S : ℝ) (G : ℝ) : ℝ :=
    S / G

theorem car_original_cost (S : ℝ) (G : ℝ) (C : ℝ) :
    S = 54000 → G = 1.20 * 0.88 →
    C = original_cost_price S G →
    C = 51136.36 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end car_original_cost_l146_146592


namespace total_pages_read_l146_146802

def pages_read_yesterday : ℕ := 21
def pages_read_today : ℕ := 17

theorem total_pages_read : pages_read_yesterday + pages_read_today = 38 :=
by
  sorry

end total_pages_read_l146_146802


namespace pure_imag_iff_ab_zero_l146_146869

theorem pure_imag_iff_ab_zero (a b : ℝ) (i : ℂ) (hia : i*i = -1) :
  (a * b = 0) ↔ (∀ (z : ℂ), z = a + b * i → z.im ≠ 0 → z.re = 0) :=
begin
  sorry,
end

end pure_imag_iff_ab_zero_l146_146869


namespace finite_solutions_l146_146742

variable (a b : ℕ) (h1 : a ≠ b)

theorem finite_solutions (a b : ℕ) (h1 : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
  (x * y + z * w = a) ∧ (x * z + y * w = b) →
  (x, y, z, w) ∈ S :=
sorry

end finite_solutions_l146_146742


namespace power_addition_l146_146295

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146295


namespace power_addition_l146_146289

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146289


namespace cryptarithm_no_solution_proof_l146_146469

def cryptarithm_no_solution : Prop :=
  ∀ (D O N K A L E V G R : ℕ),
    D ≠ O ∧ D ≠ N ∧ D ≠ K ∧ D ≠ A ∧ D ≠ L ∧ D ≠ E ∧ D ≠ V ∧ D ≠ G ∧ D ≠ R ∧
    O ≠ N ∧ O ≠ K ∧ O ≠ A ∧ O ≠ L ∧ O ≠ E ∧ O ≠ V ∧ O ≠ G ∧ O ≠ R ∧
    N ≠ K ∧ N ≠ A ∧ N ≠ L ∧ N ≠ E ∧ N ≠ V ∧ N ≠ G ∧ N ≠ R ∧
    K ≠ A ∧ K ≠ L ∧ K ≠ E ∧ K ≠ V ∧ K ≠ G ∧ K ≠ R ∧
    A ≠ L ∧ A ≠ E ∧ A ≠ V ∧ A ≠ G ∧ A ≠ R ∧
    L ≠ E ∧ L ≠ V ∧ L ≠ G ∧ L ≠ R ∧
    E ≠ V ∧ E ≠ G ∧ E ≠ R ∧
    V ≠ G ∧ V ≠ R ∧
    G ≠ R ∧
    (D * 100 + O * 10 + N) + (O * 100 + K * 10 + A) +
    (L * 1000 + E * 100 + N * 10 + A) + (V * 10000 + O * 1000 + L * 100 + G * 10 + A) =
    A * 100000 + N * 10000 + G * 1000 + A * 100 + R * 10 + A →
    false

theorem cryptarithm_no_solution_proof : cryptarithm_no_solution :=
by sorry

end cryptarithm_no_solution_proof_l146_146469


namespace area_of_equilateral_triangle_on_hyperbola_l146_146238

theorem area_of_equilateral_triangle_on_hyperbola :
  ∀ (A B C : ℝ × ℝ),
  (A = (-1, 0)) →
  (B.1 ^ 2 - B.2 ^ 2 = 1) →
  (C.1 ^ 2 - C.2 ^ 2 = 1) →
  (B.1 > 0 ∧ C.1 > 0) →
  ∃ (Δ : ℝ), 
    Δ = 3 * Real.sqrt 3 ∧ 
    Δ = abs (1 / 2 * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))))
  :=
by
  intros A B C hA hB hC hBC 
  sorry

end area_of_equilateral_triangle_on_hyperbola_l146_146238


namespace power_of_three_l146_146278

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146278


namespace number_of_tiles_l146_146132

theorem number_of_tiles (w l : ℕ) (h1 : 2 * w + 2 * l - 4 = (w * l - (2 * w + 2 * l - 4)))
  (h2 : w > 0) (h3 : l > 0) : w * l = 48 ∨ w * l = 60 :=
by
  sorry

end number_of_tiles_l146_146132


namespace compute_combination_l146_146692

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146692


namespace largest_B_for_48B56_divisibility_l146_146845

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- Problem conditions
def valid_digit (B : ℕ) : Prop := B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Main theorem statement
theorem largest_B_for_48B56_divisibility : 
  ∃ B : ℕ, valid_digit B ∧ is_divisible_by_4 (50 + B) ∧ 
  ∀ B', valid_digit B' ∧ is_divisible_by_4 (50 + B') → B' ≤ B :=
sorry

end largest_B_for_48B56_divisibility_l146_146845


namespace no_such_line_exists_l146_146873

noncomputable def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ^ 2 = 4 * p.fst}

noncomputable def focus : ℝ × ℝ := (1, 0)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.fst - Q.fst) ^ 2 + (P.snd - Q.snd) ^ 2)

theorem no_such_line_exists :
  ¬ ∃ m : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ parabola ∧ B ∈ parabola ∧
    (-1, 0).snd = m * (-1) + (A.snd - m * A.fst) ∧
    (-1, 0).snd = m * (-1) + (B.snd - m * B.fst) ∧
    distance focus (midpoint A B) = 2 := 
sorry

end no_such_line_exists_l146_146873


namespace Merrill_marbles_Vivian_marbles_l146_146439

variable (M E S V : ℕ)

-- Conditions
axiom Merrill_twice_Elliot : M = 2 * E
axiom Merrill_Elliot_five_fewer_Selma : M + E = S - 5
axiom Selma_fifty_marbles : S = 50
axiom Vivian_35_percent_more_Elliot : V = (135 * E) / 100 -- since Lean works better with integers, use 135/100 instead of 1.35
axiom Vivian_Elliot_difference_greater_five : V - E > 5

-- Questions
theorem Merrill_marbles (M E S : ℕ) (h1: M = 2 * E) (h2: M + E = S - 5) (h3: S = 50) : M = 30 := by
  sorry

theorem Vivian_marbles (V E : ℕ) (h1: V = (135 * E) / 100) (h2: V - E > 5) (h3: E = 15) : V = 21 := by
  sorry

end Merrill_marbles_Vivian_marbles_l146_146439


namespace least_possible_average_of_integers_l146_146951

theorem least_possible_average_of_integers :
  ∃ (a b c d : ℤ), a < b ∧ b < c ∧ c < d ∧ d = 90 ∧ a ≥ 21 ∧ (a + b + c + d) / 4 = 39 := by
sorry

end least_possible_average_of_integers_l146_146951


namespace triangles_not_necessarily_symmetric_l146_146962

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A1 : Point)
(A2 : Point)
(A3 : Point)

structure Ellipse :=
(a : ℝ) -- semi-major axis
(b : ℝ) -- semi-minor axis

def inscribed_in (T : Triangle) (E : Ellipse) : Prop :=
  -- Assuming the definition of the inscribed, can be encoded based on the ellipse equation: x^2/a^2 + y^2/b^2 <= 1 for each vertex.
  sorry

def symmetric_wrt_axis (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to an axis (to be defined)
  sorry

def symmetric_wrt_center (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to the center (to be defined)
  sorry

theorem triangles_not_necessarily_symmetric {E : Ellipse} {T₁ T₂ : Triangle}
  (h₁ : inscribed_in T₁ E) (h₂ : inscribed_in T₂ E) (heq : T₁ = T₂) :
  ¬ symmetric_wrt_axis T₁ T₂ ∧ ¬ symmetric_wrt_center T₁ T₂ :=
sorry

end triangles_not_necessarily_symmetric_l146_146962


namespace problem_correct_conclusions_l146_146789

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x_0 : ℝ, x_0^2 + x_0 + 1 < 0
def q : Prop := ∀ a b c : ℝ, b > c → a * b > a * c

-- The proof goals
theorem problem_correct_conclusions : 
  (¬ p ∨ q) ∧ (¬ p ∧ ¬ q) :=
by
  have hp : ¬ p,
  { sorry },
  have hq : ¬ q,
  { sorry },
  split,
  { left, exact hp },
  { exact ⟨hp, hq⟩ }

end problem_correct_conclusions_l146_146789


namespace m_range_l146_146254

def f (x k : ℝ) : ℝ := log (16 ^ x + k) / log 2 - 2 * x

lemma k_equals_1 (k : ℝ) : (∀ x : ℝ, f x k = f (-x) k) → k = 1 :=
by
  sorry

def f_with_k1 (x : ℝ) : ℝ := log (4 ^ x + 4 ^ (-x)) / log 2

theorem m_range (m : ℝ) : 
  (∀ x ∈ set.Icc (-1 : ℝ) (1 / 2 : ℝ), m - 1 ≤ f_with_k1 x ∧ f_with_k1 x ≤ 2 * m + (log 17 / log 2)) → 
  -1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end m_range_l146_146254


namespace relationship_x_a_b_l146_146473

theorem relationship_x_a_b (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) : 
  x^2 > a * b ∧ a * b > a^2 :=
by
  sorry

end relationship_x_a_b_l146_146473


namespace probability_two_red_balls_l146_146220

theorem probability_two_red_balls :
  let total_balls := 4
  let red_balls := 2
  let white_balls := 2
  let drawn_balls := 2
  let total_combinations := Nat.choose total_balls drawn_balls
  let favorable_combinations := Nat.choose red_balls drawn_balls
  let probability := favorable_combinations / total_combinations
  probability = 1 / 6 :=
by
  let total_balls := 4
  let red_balls := 2
  let white_balls := 2
  let drawn_balls := 2
  let total_combinations := Nat.choose total_balls drawn_balls
  let favorable_combinations := Nat.choose red_balls drawn_balls
  let probability := favorable_combinations / total_combinations
  have h1 : total_combinations = 6 := by sorry
  have h2 : favorable_combinations = 1 := by sorry
  have h3 : probability = favorable_combinations / total_combinations := by sorry
  rw [h2, h1, h3]
  norm_num
  exact rfl

end probability_two_red_balls_l146_146220


namespace total_buttons_needed_l146_146007

def shirtsMonday : ℕ := 4
def shirtsTuesday : ℕ := 3
def shirtsWednesday : ℕ := 2
def buttonsPerShirt : ℕ := 5

theorem total_buttons_needed : (shirtsMonday + shirtsTuesday + shirtsWednesday) * buttonsPerShirt = 45 :=
by
  have shirtsTotal : ℕ := shirtsMonday + shirtsTuesday + shirtsWednesday
  have buttonsTotal : ℕ := shirtsTotal * buttonsPerShirt
  have h1 := rfl
  rw [← h1, add_assoc, ← add_assoc shirtsTuesday shirtsWednesday, add_comm shirtsThursday shirtsWednesday, add_assoc shirtsTuesday shirtsWednesday shirtsMonday] at h1
  rw [← h1, mul_add, ← add_mul] at h1
  sorry

end total_buttons_needed_l146_146007


namespace largest_multiple_of_15_under_500_l146_146993

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146993


namespace jack_round_trip_speed_l146_146411

noncomputable def jack_average_speed (d1 d2 : ℕ) (t1 t2 : ℕ) : ℕ :=
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let total_time_hours := total_time / 60
  total_distance / total_time_hours

theorem jack_round_trip_speed : jack_average_speed 3 3 45 15 = 6 := by
  -- Import necessary library
  sorry

end jack_round_trip_speed_l146_146411


namespace jony_speed_l146_146091

theorem jony_speed :
  let start_block := 10
  let end_block := 90
  let turn_around_block := 70
  let block_length := 40 -- meters
  let start_time := 0 -- 07:00 in minutes from the start of his walk
  let end_time := 40 -- 07:40 in minutes from the start of his walk
  let total_blocks_walked := (end_block - start_block) + (end_block - turn_around_block)
  let total_distance := total_blocks_walked * block_length
  let total_time := end_time - start_time
  total_distance / total_time = 100 :=
by
  sorry

end jony_speed_l146_146091


namespace solution_set_l146_146708

noncomputable def f : ℝ → ℝ := sorry

axiom f_differentiable : ∀ x > 0, differentiable_at ℝ f x
axiom f_condition : ∀ x > 0, x * (f'' x) < f x
axiom f_at_1 : f 1 = 0

theorem solution_set : { x : ℝ | 0 < x ∧ f x < 0 } = set.Ioi 1 :=
sorry

end solution_set_l146_146708


namespace meaningful_fraction_l146_146504

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end meaningful_fraction_l146_146504


namespace n_prime_or_power_of_two_l146_146755

theorem n_prime_or_power_of_two
  (n : ℕ)
  (h1 : n > 6)
  (a : ℕ → ℕ)
  (h2 : ∀ i, 1 ≤ i → a i < n)
  (h3 : ∀ i, 1 ≤ i → Nat.coprime (a i) n)
  (h4 : ∀ i, 2 ≤ i → a i - a (i - 1) = (a 2 - a 1)) :
  Nat.Prime n ∨ ∃ k : ℕ, n = 2^k :=
sorry

end n_prime_or_power_of_two_l146_146755


namespace variance_of_data_set_l146_146517

def data_set : List ℝ := [80, 81, 82, 83]

noncomputable def mean (data : List ℝ) : ℝ :=
  (data.sum) / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (λ x => (x - m)^2)).sum / data.length

theorem variance_of_data_set : variance data_set = 1.25 := 
by 
  sorry

end variance_of_data_set_l146_146517


namespace max_k_value_l146_146377

theorem max_k_value (x y : ℝ) (k : ℝ) (hx : 0 < x) (hy : 0 < y)
(h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) := sorry

end max_k_value_l146_146377


namespace exactly_one_true_l146_146164

def proposition1 (b : ℝ) : Prop := (b^2 = 9) → (b = 3)
def proposition2 : Prop := ¬(∀ (T₁ T₂ : Triangle), congruent T₁ T₂ → equal_areas T₁ T₂)
def proposition3 (c : ℝ) : Prop := (c ≤ 1) → ∃ (x : ℝ), x^2 + 2 * x + c = 0
def proposition4 (A B : Set) : Prop := ((A ∪ B) = A) → (A ⊆ B)

theorem exactly_one_true :
  (∀ b, ¬proposition1 b) ∧
  ¬proposition2 ∧
  (∃ c, proposition3 c) ∧
  (∀ A B, ¬proposition4 A B) → 
  (∃! p, p = proposition3) := 
by
  sorry

end exactly_one_true_l146_146164


namespace max_min_sum_l146_146261

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - (2 * x^2) / (2^x + 1) + 3 * Real.sin x + 1

theorem max_min_sum {M N : ℝ} 
  (hM : ∀ x ∈ set.Icc (-(1/2) : ℝ) (1/2), f x ≤ M) 
  (hN : ∀ x ∈ set.Icc (-(1/2) : ℝ) (1/2), N ≤ f x) 
  (hM_attain : ∃ x ∈ set.Icc (-(1/2) : ℝ) (1/2), f x = M) 
  (hN_attain : ∃ x ∈ set.Icc (-(1/2) : ℝ) (1/2), f x = N) 
  : M + N = 2 :=
sorry

end max_min_sum_l146_146261


namespace unoccupied_volume_correct_l146_146895

-- Definitions based on conditions
def side_length : ℕ := 12
def container_volume : ℕ := side_length ^ 3
def water_fraction : ℚ := 1 / 3
def water_volume : ℕ := (water_fraction * container_volume).toNat
def ice_cube_side : ℕ := 1
def ice_cube_volume : ℕ := ice_cube_side ^ 3
def num_ice_cubes : ℕ := 15
def total_ice_volume : ℕ := num_ice_cubes * ice_cube_volume
def occupied_volume : ℕ := water_volume + total_ice_volume
def unoccupied_volume : ℕ := container_volume - occupied_volume

-- Theorem statement
theorem unoccupied_volume_correct : unoccupied_volume = 1137 :=
by sorry

end unoccupied_volume_correct_l146_146895


namespace power_of_three_l146_146361

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146361


namespace intersection_of_A_and_B_is_intervals_l146_146266

def setA (x : ℝ) : set ℝ := {y : ℝ | y = -x^2 - 2*x}
def setB : set ℝ := {y : ℝ | ∃ x : ℝ, y = x + 1}

theorem intersection_of_A_and_B_is_intervals (h : ∀ x : ℝ, ∃ y : ℝ, y = -x^2 - 2*x) (h' : ∀ x : ℝ, ∃ y : ℝ, y = x + 1) :
  (setA ∩ setB) = {y : ℝ | y ≤ 1} :=
sorry

end intersection_of_A_and_B_is_intervals_l146_146266


namespace sum_distances_eq_sum_radii_l146_146033

-- Definitions of the elements involved.
variables {Δ : Type} [triangle Δ] {A B C : Δ}
variables {O : Δ}
variables {R r d_a d_b d_c : ℝ}
variables (acute : acute_triangle A B C)
variables (circumcenter : center_circle O A B C R)
variables (dist_to_sides : distances_to_sides O A B C d_a d_b d_c)

-- The actual theorem to be proven.
theorem sum_distances_eq_sum_radii : d_a + d_b + d_c = R + r :=
sorry

end sum_distances_eq_sum_radii_l146_146033


namespace pints_for_five_cookies_l146_146528

theorem pints_for_five_cookies
  (gallons_to_bake_eighteen : ℕ)
  (quarts_per_gallon : ℕ)
  (pints_per_quart : ℕ)
  (cookies : ℕ)
  (milk_needed : ℚ) :
  gallons_to_bake_eighteen = 3 →
  quarts_per_gallon = 4 →
  pints_per_quart = 2 →
  cookies = 18 →
  milk_needed = (3 : ℚ) * (4 : ℚ) * (2 : ℚ) →
  (5 : ℚ) * (milk_needed / cookies) = (20 : ℚ) / 3 :=
by
  intros h_gallons h_quarts h_pints h_cookies h_milk
  simp [h_gallons, h_quarts, h_pints, h_cookies] at h_milk
  sorry

end pints_for_five_cookies_l146_146528


namespace power_equality_l146_146317

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146317


namespace find_y_for_given_slope_l146_146228

theorem find_y_for_given_slope (y : ℝ) :
  let P := (-3 : ℝ, 4 : ℝ),
      Q := (5 : ℝ, y : ℝ) in
  (Q.snd - P.snd) / (Q.fst - P.fst) = 1 / 4 → y = 6 :=
by
  sorry

end find_y_for_given_slope_l146_146228


namespace river_flow_rate_kmph_l146_146134

def depth : ℝ := 2
def width : ℝ := 45
def volume_per_minute : ℝ := 10500

def area : ℝ := depth * width
def flow_rate : ℝ := volume_per_minute / 60
def velocity_mps : ℝ := flow_rate / area
def velocity_kmph : ℝ := velocity_mps * (1000 / 3600)

theorem river_flow_rate_kmph :
  velocity_kmph = 0.54 :=
sorry

end river_flow_rate_kmph_l146_146134


namespace monthly_payment_l146_146706

noncomputable def house_price := 280
noncomputable def deposit := 40
noncomputable def mortgage_years := 10
noncomputable def months_per_year := 12

theorem monthly_payment (house_price deposit : ℕ) (mortgage_years months_per_year : ℕ) :
  (house_price - deposit) / mortgage_years / months_per_year = 2 :=
by
  sorry

end monthly_payment_l146_146706


namespace final_weight_proof_l146_146862

variable (initial_weight : ℕ) (jellybeans_weight1 : ℕ) (multiplier_brownies : ℕ) (jellybeans_weight2 : ℕ) (multiplier_gummy_worms : ℕ)

variable (final_weight : ℕ)

-- Conditions
axiom initial_weight_cond : initial_weight = 2
axiom multiplier_brownies_cond : multiplier_brownies = 3
axiom jellybeans_weight2_cond : jellybeans_weight2 = 2
axiom multiplier_gummy_worms_cond : multiplier_gummy_worms = 2

def box_with_jellybeans_weight1 := initial_weight
def box_with_brownies_weight := box_with_jellybeans_weight1 * multiplier_brownies
def box_with_jellybeans_weight2 := box_with_brownies_weight + jellybeans_weight2
def box_with_gummy_worms_weight := box_with_jellybeans_weight2 * multiplier_gummy_worms

axiom final_weight_cond : box_with_gummy_worms_weight = final_weight

theorem final_weight_proof : final_weight = 16 :=
by
  rw [initial_weight_cond, multiplier_brownies_cond, jellybeans_weight2_cond, multiplier_gummy_worms_cond]
  unfold box_with_jellybeans_weight1 box_with_brownies_weight box_with_jellybeans_weight2 box_with_gummy_worms_weight
  rw [Nat.mul_add, Nat.mul_add, Nat.mul_one, Nat.one_mul]
  exact final_weight_cond
  sorry

end final_weight_proof_l146_146862


namespace find_overlap_length_l146_146944

-- Definitions of the given conditions
def total_length_of_segments := 98 -- cm
def edge_to_edge_distance := 83 -- cm
def number_of_overlaps := 6

-- Theorem stating the value of x in centimeters
theorem find_overlap_length (x : ℝ) 
  (h1 : total_length_of_segments = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : number_of_overlaps = 6) 
  (h4 : total_length_of_segments = edge_to_edge_distance + number_of_overlaps * x) : 
  x = 2.5 :=
  sorry

end find_overlap_length_l146_146944


namespace pirates_coins_l146_146116
open Nat

noncomputable def minimal_coins_needed (n : Nat) : Nat :=
  if n = 15 then (14 * 13 * 11 * 7) else 0 -- as per given conditions, we define for 15 pirates only

theorem pirates_coins :
  ∃ x : Nat, x = minimal_coins_needed 15 ∧ x = 12012 :=
by
  use 12012
  split
  . refl
  . refl

end pirates_coins_l146_146116


namespace min_value_l146_146766

/-- Given x and y are positive real numbers such that x + 3y = 2,
    the minimum value of (2x + y) / (xy) is 1/2 * (7 + 2 * sqrt 6). -/
theorem min_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = 2) :
  ∃ c : ℝ, c = (1/2) * (7 + 2 * Real.sqrt 6) ∧ ∀ (x y : ℝ), (0 < x) → (0 < y) → (x + 3 * y = 2) → ((2 * x + y) / (x * y)) ≥ c :=
sorry

end min_value_l146_146766


namespace f_decreasing_on_interval_l146_146450

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

theorem f_decreasing_on_interval : ∀ x1 x2 : ℝ, 1 < x1 → x1 < x2 → (f(x1) > f(x2)) :=
by
  sorry

end f_decreasing_on_interval_l146_146450


namespace base8_subtraction_correct_l146_146156

theorem base8_subtraction_correct : nat.sub 0o46 0o27 = 0o17 := by
  sorry

end base8_subtraction_correct_l146_146156


namespace square_area_twice_triangle_perimeter_l146_146138

noncomputable def perimeter_of_triangle (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area_of_square (side_length : ℕ) : ℕ :=
  side_length * side_length

theorem square_area_twice_triangle_perimeter (a b c : ℕ) (h1 : perimeter_of_triangle a b c = 22) (h2 : a = 5) (h3 : b = 7) (h4 : c = 10) : area_of_square (side_length_of_square (2 * perimeter_of_triangle a b c)) = 121 :=
by
  sorry

end square_area_twice_triangle_perimeter_l146_146138


namespace probability_of_not_getting_software_contract_l146_146582

theorem probability_of_not_getting_software_contract :
  let P_H : ℝ := 3 / 4,
      P_at_least_one : ℝ := 5 / 6,
      P_both : ℝ := 0.31666666666666654 in
  let P_S := P_at_least_one - P_H + P_both in
  1 - P_S = 0.6 :=
by
  simp only [←sub_sub, sub_self, sub_zero]
  sorry

end probability_of_not_getting_software_contract_l146_146582


namespace shelley_weight_l146_146083

theorem shelley_weight (p s r : ℕ) (h1 : p + s = 151) (h2 : s + r = 132) (h3 : p + r = 115) : s = 84 := 
  sorry

end shelley_weight_l146_146083


namespace man_rate_in_still_water_correct_l146_146553

structure BoatSpeed :=
(speed_with_stream : ℝ)
(speed_against_stream : ℝ)

def man_rate_in_still_water (s : BoatSpeed) : ℝ :=
(s.speed_with_stream + s.speed_against_stream) / 2

theorem man_rate_in_still_water_correct :
  ∀ (s : BoatSpeed), 
  s.speed_with_stream = 12 ∧ s.speed_against_stream = 4 → 
  man_rate_in_still_water s = 8 :=
by
  intro s
  intro h
  cases h with h1 h2
  rw [h1, h2]
  simp
  sorry

end man_rate_in_still_water_correct_l146_146553


namespace smallest_pos_z_l146_146472

theorem smallest_pos_z (x z : ℝ) (k m : ℤ) (hx : sin x = 0) (hz : sin (x + z) = sqrt 2 / 2) : 
  z = π / 4 := 
by
  sorry

end smallest_pos_z_l146_146472


namespace part1_part2_l146_146773

def f (x : ℝ) := (log x)^2 + 3 * log x + 3 / x

theorem part1 (x : ℝ) (hx : x = e) :
  deriv f x = -2 / e^2 :=
by
  sorry -- Proof that the tangent at (e, f(e)) is parallel to the line 2x + e^2 y = 0

theorem part2 (x : ℝ) (hx : 0 < x) :
  (f x / x) > 3 / exp x :=
by
  sorry -- Proof that f(x)/x > 3/e^x for all x > 0

end part1_part2_l146_146773


namespace smallest_degree_polynomial_l146_146594

noncomputable def smallest_possible_degree : ℕ :=
  let roots := { n + real.sqrt (n + 1) | n : ℕ, 1 ≤ n ∧ n ≤ 500 }
  let perfect_squares := { k * k - 1 | k : ℕ, 2 ≤ k ∧ k * k ≤ 501 }
  let irrational_count := 500 - perfect_squares.card
  2 * irrational_count + perfect_squares.card

theorem smallest_degree_polynomial :
  smallest_possible_degree = 979 :=
sorry

end smallest_degree_polynomial_l146_146594


namespace speed_of_car_in_second_hour_l146_146509

noncomputable def speed_in_first_hour : ℝ := 90
noncomputable def average_speed : ℝ := 82.5
noncomputable def total_time : ℝ := 2

theorem speed_of_car_in_second_hour : 
  ∃ (speed_in_second_hour : ℝ), 
  (speed_in_first_hour + speed_in_second_hour) / total_time = average_speed ∧ 
  speed_in_first_hour = 90 ∧ 
  average_speed = 82.5 → 
  speed_in_second_hour = 75 :=
by 
  sorry

end speed_of_car_in_second_hour_l146_146509


namespace sqrt_expr_defined_iff_l146_146384

theorem sqrt_expr_defined_iff (x : ℝ) : ∃ y : ℝ, y = sqrt (x + 5) ↔ x ≥ -5 := by
  sorry

end sqrt_expr_defined_iff_l146_146384


namespace evaluate_expression_l146_146158

theorem evaluate_expression :
  4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 :=
by
  sorry

end evaluate_expression_l146_146158


namespace average_apples_sold_is_six_l146_146437

-- Let's define the conditions as constants
def sales_in_first_hour : ℝ := 10
def sales_in_second_hour : ℝ := 2

-- The total apples sold and the number of hours
def total_apples_sold : ℝ := sales_in_first_hour + sales_in_second_hour
def total_hours : ℝ := 2

-- The average apples sold per hour
def average_apples_per_hour : ℝ := total_apples_sold / total_hours

-- We are proving that the average number of kg of apples sold per hour is 6 kg/hour.
theorem average_apples_sold_is_six : average_apples_per_hour = 6 := by
  sorry

end average_apples_sold_is_six_l146_146437


namespace power_calculation_l146_146307

theorem power_calculation (y : ℤ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_calculation_l146_146307


namespace rate_in_still_water_l146_146591

-- Definitions of given conditions
def downstream_speed : ℝ := 26
def upstream_speed : ℝ := 12

-- The statement we need to prove
theorem rate_in_still_water : (downstream_speed + upstream_speed) / 2 = 19 := by
  sorry

end rate_in_still_water_l146_146591


namespace inequality_solution_set_l146_146050

theorem inequality_solution_set : 
  { x : ℝ | (x + 1) / (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry 

end inequality_solution_set_l146_146050


namespace additional_cars_needed_l146_146705

theorem additional_cars_needed (current_cars : ℕ) (rows_of_7 : ℕ) : current_cars = 39 → rows_of_7 = 7 → (∃ additional_cars : ℕ, additional_cars = 3 ∧ (current_cars + additional_cars) % rows_of_7 = 0) :=
by
  intro h1 h2
  use 3
  split
  . exact rfl
  . rw [h1, h2]
  . norm_num
  sorry

end additional_cars_needed_l146_146705


namespace power_equality_l146_146318

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146318


namespace constant_term_in_expansion_l146_146405

theorem constant_term_in_expansion :
  ∃ (c : ℤ), ∃ (T_r : ℕ → ℤ), T_r = (λ r, (binomial 6 r) * (-2)^r * 6 - 2 * r) ∧ T_r 3 = c ∧ c = -160 :=
begin
  sorry
end

end constant_term_in_expansion_l146_146405


namespace exist_teams_winning_one_match_each_l146_146060

def football_teams : Type := fin 20

def score_after_one_match (t : football_teams) : ℕ
def score_after_two_matches (t : football_teams) : ℕ

axiom different_scores_one_match : ∀ (t1 t2 : football_teams), t1 ≠ t2 → score_after_one_match t1 ≠ score_after_one_match t2
axiom same_scores_two_matches : ∀ (t : football_teams), score_after_two_matches t = 38
axiom total_points_two_matches : ∑ t in finset.univ, score_after_two_matches t = 760

theorem exist_teams_winning_one_match_each :
  ∃ (t1 t2 : football_teams), t1 ≠ t2 ∧ score_after_two_matches t1 < score_after_two_matches t2 :=
  sorry

end exist_teams_winning_one_match_each_l146_146060


namespace find_angle_E_l146_146850

def trapezoid_angles (E H F G : ℝ) : Prop :=
  E + H = 180 ∧ E = 3 * H ∧ G = 4 * F

theorem find_angle_E (E H F G : ℝ) 
  (h1 : E + H = 180)
  (h2 : E = 3 * H)
  (h3 : G = 4 * F) : 
  E = 135 := by
    sorry

end find_angle_E_l146_146850


namespace limit_of_cubic_division_l146_146728

noncomputable def limit_expr := (fun x : ℝ => (x^3 - 1) / (x - 1))
def limit_point := 1
def limit_result := 3

theorem limit_of_cubic_division :
  filter.tendsto limit_expr (nhds limit_point) (nhds limit_result) := 
sorry

end limit_of_cubic_division_l146_146728


namespace angle_sum_solution_l146_146082

theorem angle_sum_solution
  (x : ℝ)
  (h : 3 * x + 140 = 360) :
  x = 220 / 3 :=
by
  sorry

end angle_sum_solution_l146_146082


namespace circumference_of_tank_B_l146_146922

noncomputable def radius_of_tank (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def volume_of_tank (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem circumference_of_tank_B 
  (h_A : ℝ) (C_A : ℝ) (h_B : ℝ) (volume_ratio : ℝ)
  (hA_pos : 0 < h_A) (CA_pos : 0 < C_A) (hB_pos : 0 < h_B) (vr_pos : 0 < volume_ratio) :
  2 * Real.pi * (radius_of_tank (volume_of_tank (radius_of_tank C_A) h_A / (volume_ratio * Real.pi * h_B))) = 17.7245 :=
by 
  sorry

end circumference_of_tank_B_l146_146922


namespace most_accurate_method_l146_146615

def method_a := "Judging by three-dimensional bar charts"
def method_b := "Judging by two-dimensional bar charts"
def method_c := "Judging by contour bar charts"
def method_d := "None of the above"

theorem most_accurate_method :
  ∀ (method : String),
  (method = method_a ∨ method = method_b ∨ method = method_c) → method ≠ method_d := 
by
  intros method h
  intro h_eq
  cases h;
  { rw h_eq at h; contradiction }
  
  sorry

end most_accurate_method_l146_146615


namespace smallest_zucchinis_received_l146_146620

theorem smallest_zucchinis_received (n : ℕ) (A : ℕ) (V : ℕ) (hA : n = 2 * A) (hA_square : ∃ a : ℕ, A = a^2) (hV : n = 3 * V) (hV_cube : ∃ b : ℕ, V = b^3) : n = 648 :=
by
  use 648
  -- We use 648 as the smallest number that satisfies all the conditions.
  sorry

end smallest_zucchinis_received_l146_146620


namespace equal_number_of_coins_l146_146118

theorem equal_number_of_coins (x : ℕ) (hx : 1 * x + 5 * x + 10 * x + 25 * x + 100 * x = 305) : x = 2 :=
sorry

end equal_number_of_coins_l146_146118


namespace range_of_m_if_not_p_and_q_l146_146760

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_if_not_p_and_q_l146_146760


namespace complex_quadrant_l146_146746

open Complex

theorem complex_quadrant 
  (z : ℂ) 
  (h : (1 - I) ^ 2 / z = 1 + I) :
  z = -1 - I :=
by
  sorry

end complex_quadrant_l146_146746


namespace combined_teaching_experience_l146_146413

theorem combined_teaching_experience :
  let James_experience := 40 in
  let Sarah_experience := James_experience - 10 in
  let Robert_experience := 2 * Sarah_experience in
  let Emily_experience := (3 * Sarah_experience) - 5 in
  James_experience + Sarah_experience + Robert_experience + Emily_experience = 215 :=
by
  sorry

end combined_teaching_experience_l146_146413


namespace compute_expression_l146_146657

theorem compute_expression : 2 * ((3 + 7) ^ 2 + (3 ^ 2 + 7 ^ 2)) = 316 := 
by
  sorry

end compute_expression_l146_146657


namespace total_water_needed_l146_146442

def tanks := 6
def first_two_tanks := 10
def next_two_tanks := first_two_tanks - 2
def last_two_tanks := first_two_tanks + 3
def days_per_change := 5
def total_days := 25

theorem total_water_needed :
  let water_per_change := 2 * first_two_tanks + 2 * next_two_tanks + 2 * last_two_tanks,
      changes := total_days / days_per_change
  in water_per_change * changes = 310 :=
by sorry

end total_water_needed_l146_146442


namespace fraction_meaningful_range_l146_146507

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_range_l146_146507


namespace line_symmetric_fixed_point_l146_146816

noncomputable def is_symmetric (p1 p2 : Point) (c : Point) : Prop :=
p1.1 + p2.1 = 2 * c.1 ∧ p1.2 + p2.2 = 2 * c.2

theorem line_symmetric_fixed_point (k : ℝ) :
∀ l2 : ℝ → ℝ, (∀ x, l2 x = k * (x - 4)) → 
∃ p : Point, p = (0, 2) :=
begin
  sorry
end

end line_symmetric_fixed_point_l146_146816


namespace positive_integer_triplets_l146_146167

theorem positive_integer_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_lcm : a + b + c = Nat.lcm a (Nat.lcm b c)) :
  (∃ k, k ≥ 1 ∧ a = k ∧ b = 2 * k ∧ c = 3 * k) :=
sorry

end positive_integer_triplets_l146_146167


namespace avg_sales_is_approximately_110_l146_146161

-- Define the sales data for each month
def sales : List ℕ := [150, 100, 50, 130, 120, 100]

-- Define the total number of months
def num_months : ℕ := 6

-- Define the total sales
def total_sales : ℕ := sales.sum

-- Define the average sales per month
def avg_sales_per_month : ℚ := total_sales / num_months

-- The theorem stating the average sales per month is approximately 110 dollars
theorem avg_sales_is_approximately_110 : avg_sales_per_month ≈ 110 := by
  sorry

end avg_sales_is_approximately_110_l146_146161


namespace taller_tree_height_l146_146056

-- Definitions and Variables
variables (h : ℝ)

-- Conditions as Definitions
def top_difference_condition := (h - 20) / h = 5 / 7

-- Proof Statement
theorem taller_tree_height (h : ℝ) (H : top_difference_condition h) : h = 70 := 
by {
  sorry
}

end taller_tree_height_l146_146056


namespace problem_f_value_at_2_l146_146434

def f (x : ℝ) (a b : ℝ) := a * x + b

theorem problem_f_value_at_2 (a b : ℝ) 
  (h_deriv : ∀ x, deriv (λ x, f x a b) x = a) 
  (h_f1 : f 1 a b = 2) 
  (h_deriv_at_1 : deriv (f 1) = 2) : 
  f 2 a b = 4 := sorry

end problem_f_value_at_2_l146_146434


namespace power_of_three_l146_146283

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146283


namespace find_b6b8_l146_146839

-- Define sequences {a_n} (arithmetic sequence) and {b_n} (geometric sequence)
variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Given conditions
axiom h1 : ∀ n m : ℕ, a m = a n + (m - n) * (a (n + 1) - a n) -- Arithmetic sequence property
axiom h2 : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0
axiom h3 : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1 -- Geometric sequence property
axiom h4 : b 7 = a 7
axiom h5 : ∀ n : ℕ, b n > 0                 -- Assuming b_n has positive terms
axiom h6 : ∀ n : ℕ, a n > 0                 -- Positive terms in sequence a_n

-- Proof objective
theorem find_b6b8 : b 6 * b 8 = 16 :=
by sorry

end find_b6b8_l146_146839


namespace discount_limit_l146_146576

theorem discount_limit (purchase_price original_price profit_margin : ℝ) (h : purchase_price = 6 ∧ original_price = 9 ∧ profit_margin = 0.05) : 
  ∀ discount_rate: ℝ, 
    (original_price * (discount_rate / 10) - purchase_price) <purchase_price * profit_margin  → discount_rate ≤ 7  := 
by 
  intros
  rw [← h.1, ← h.2, ← h.3] at *
  sorry

end discount_limit_l146_146576


namespace orthocenters_form_rectangle_l146_146270

-- Definitions based on conditions
variables {circle1 circle2 : Circle}
variable {A B S T M N : Point}
variable {ST MN : Line}

-- Additional axioms and assumptions
axioms
  (circle1_circle2_intersect : circle1 ∩ circle2 = {A, B})
  (ST_tangent_to_circle1_at_S : tangent_to_circle ST circle1 S)
  (ST_tangent_to_circle2_at_T : tangent_to_circle ST circle2 T)
  (MN_tangent_to_circle1_at_M : tangent_to_circle MN circle1 M)
  (MN_tangent_to_circle2_at_N : tangent_to_circle MN circle2 N)

-- The statement to prove
theorem orthocenters_form_rectangle :
  let H_AMN := orthocenter (triangle A M N),
      H_AST := orthocenter (triangle A S T),
      H_BMN := orthocenter (triangle B M N),
      H_BST := orthocenter (triangle B S T) in
  is_rectangle H_AMN H_AST H_BMN H_BST := 
sorry

end orthocenters_form_rectangle_l146_146270


namespace general_formula_sum_of_first_n_terms_S_n_l146_146403

variable {ℕ : Type} [Pos := 0]

-- Conditions
def a : ℕ → ℕ
def a_2 : a 2 = 1 := by rfl
def a_5 : a 5 = 4 := by rfl

-- Question 1: General formula for the sequence {a_n}
theorem general_formula (n : ℕ) : a n = n - 1 := 
begin
  sorry
end

-- Define sequence b_n such that b_n = 2^(a_n)
def b_n (n : ℕ) : ℕ := 2 ^ (a n)

-- Question 2: Sum of the first n terms S_n of the sequence {b_n}
theorem sum_of_first_n_terms_S_n (n : ℕ) : 
  (finset.range n).sum b_n = 2 ^ n - 1 := 
begin
  sorry
end

end general_formula_sum_of_first_n_terms_S_n_l146_146403


namespace compute_combination_l146_146687

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146687


namespace shaded_area_correct_l146_146461

open Real

def diameter : ℝ := 3
def overlap : ℝ := 0.5
def total_length : ℝ := 12

-- Assuming the effective length covered by each semicircle
def effective_length := diameter - overlap

-- Assuming the number of semicircles (rounded up)
def num_semicircles := Total_length / effective_length |> ceil

-- Assuming the equivalent number of full circles
def equivalent_full_circles := num_semicircles / 2

-- Theorem stating the shaded area is 5.625π square inches.
theorem shaded_area_correct :
  let area := equivalent_full_circles * (π * (diameter / 2)^2)
  area = 5.625 * π := by
    sorry

end shaded_area_correct_l146_146461


namespace find_X_l146_146080

theorem find_X (X : ℝ) (h : 0.80 * X - 0.35 * 300 = 31) : X = 170 :=
by
  sorry

end find_X_l146_146080


namespace inequality_l146_146427

theorem inequality (a b c d e p q : ℝ) 
  (h0 : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (h1 : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * ((1 / a) + (1 / b) + (1 / c) + (1 / d) + (1 / e)) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
by
  sorry

end inequality_l146_146427


namespace find_valid_number_l146_146929

def is_valid_number (N : ℕ) : Prop :=
  ∃ a b : ℕ, a ∈ Finset.range 10 ∧ b = 9 ∧ N = 10*a + b

theorem find_valid_number (N : ℕ) (a b : ℕ) (h1 : N = 10 * a + b) (h2 : N - a * b = a + b) : 
  (N = 19 ∨ N = 29 ∨ N = 39 ∨ N = 49 ∨ N = 59 ∨ N = 69 ∨ N = 79 ∨ N = 89 ∨ N = 99) :=
sorry

end find_valid_number_l146_146929


namespace crossing_time_correct_l146_146073

def length_first_train : ℝ := 340
def length_second_train : ℝ := 250
def speed_first_train_kmh : ℝ := 80
def speed_second_train_kmh : ℝ := 55
def conversion_factor : ℝ := 5 / 18

def relative_speed_mps : ℝ := (speed_first_train_kmh + speed_second_train_kmh) * conversion_factor

def total_distance : ℝ := length_first_train + length_second_train

def crossing_time : ℝ := total_distance / relative_speed_mps

theorem crossing_time_correct : abs (crossing_time - 7.87) < 0.01 := 
sorry

end crossing_time_correct_l146_146073


namespace find_equations_of_ellipse_and_parabola_l146_146622

noncomputable def F1 := (0, -1) -- Assume foci, it is determined from the ellipse properties
noncomputable def F2 := (1, 0)
noncomputable def O := (0, 0)
noncomputable def A := (3 / 2, Real.sqrt 6)

theorem find_equations_of_ellipse_and_parabola :
  ∃ (a b : ℝ), 
    (a = 3 ∧ b = Real.sqrt 8 ∧ 
     ∀ x y : ℝ, ((x, y) = A) → 
     ((x^2 / a^2) + (y^2 / b^2) = 1) ∧ 
     (y^2 = 4 * (x - 1) + 4)) := 
sorry

end find_equations_of_ellipse_and_parabola_l146_146622


namespace hamburger_per_meatball_l146_146621

theorem hamburger_per_meatball (family_members : ℕ) (total_hamburger : ℕ) (antonio_meatballs : ℕ) 
    (hmembers : family_members = 8)
    (hhamburger : total_hamburger = 4)
    (hantonio : antonio_meatballs = 4) : 
    (total_hamburger : ℝ) / (family_members * antonio_meatballs) = 0.125 := 
by
  sorry

end hamburger_per_meatball_l146_146621


namespace determinant_zero_l146_146651

open Matrix

variables {R : Type*} [Field R] {a b : R}

def M : Matrix (Fin 3) (Fin 3) R :=
  ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ]

theorem determinant_zero : det M = 0 :=
by
  sorry

end determinant_zero_l146_146651


namespace intersecting_lines_infinite_l146_146842

-- Definitions of the cube vertices and midpoints
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Assume points in 3D space
def A_1 : Point3D := sorry
def B_1 : Point3D := sorry
def C_1 : Point3D := sorry
def D_1 : Point3D := sorry
def A : Point3D := sorry
def B : Point3D := sorry
def C : Point3D := sorry
def D : Point3D := sorry

-- Midpoints
def E : Point3D := midpoint A_1 A
def F : Point3D := midpoint C_1 C

-- Function to calculate the midpoint
def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

-- Define the spatial lines intersecting
def line (p1 p2 : Point3D) : Set Point3D := sorry

-- Define the lines A_1D_1, EF, and DC
def lineA1D1 : Set Point3D := line A_1 D_1
def lineEF : Set Point3D := line E F
def lineDC : Set Point3D := line D C

-- The theorem statement
theorem intersecting_lines_infinite :
  ∃ (lines : Set (Set Point3D)), lines.Countable ∧
    ∀ l ∈ lines, l ∩ lineA1D1 ≠ ∅ ∧ l ∩ lineEF ≠ ∅ ∧ l ∩ lineDC ≠ ∅ :=
begin
  sorry,
end

end intersecting_lines_infinite_l146_146842


namespace simplify_radical_product_l146_146542

theorem simplify_radical_product : 
  (32^(1/5)) * (8^(1/3)) * (4^(1/2)) = 8 := 
by
  sorry

end simplify_radical_product_l146_146542


namespace rectangle_area_l146_146130

theorem rectangle_area (w : ℝ) (h : ℝ) (d : ℝ) (h1 : h = 3 * w) (h2 : d = 16) (h3 : d^2 = w^2 + h^2) :
  w * h = 76.8 :=
by
  -- Definitions and conditions
  have eq1 : h^2 = (3 * w)^2, from by rw [h1, sqr],
  have eq2 : w^2 + (3 * w)^2 = 256, from by rw [← h3, h2, pow_two],
  have eq3 : w^2 + 9 * w^2 = 256, from by rw [← add_mul, eq1],
  -- Simplify to find w^2
  have eq4 : 10 * w^2 = 256, from eq3,
  have ww : w^2 = 256 / 10, from eq4,
  -- Calculate area
  have area : w * (3 * w) = 76.8, from
    by rw [h1, ← mul_assoc, mul_comm w _]; exact by
    calc w * 3 * w = 3 * w^2 : by rw mul_comm
         ... = 3 * (256 / 10) : by rw ww
         ... = 768 / 10 : by rw mul_div_assoc
         ... = 76.8 : by norm_num,
  exact area
  sorry

end rectangle_area_l146_146130


namespace negation_of_universal_proposition_l146_146500

def P (x : ℝ) : Prop := x^3 + 2 * x ≥ 0

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 0 ≤ x → P x) ↔ (∃ x : ℝ, 0 ≤ x ∧ ¬ P x) :=
by
  sorry

end negation_of_universal_proposition_l146_146500


namespace rice_mix_ratio_l146_146409

-- Definitions based on conditions
def price1 : ℝ := 3.10
def price2 : ℝ := 3.60
def mixture_price : ℝ := 3.25

-- Proportion to be proved
theorem rice_mix_ratio :
  ∃ (x y : ℝ), price1 * x + price2 * y = mixture_price * (x + y)
             ∧ y / x = 3 / 7 :=
begin
  sorry
end

end rice_mix_ratio_l146_146409


namespace infinite_rel_prime_pairs_poly_l146_146455

theorem infinite_rel_prime_pairs_poly :
  ∃ (m n : ℕ), nat.coprime m n ∧ ∃ a b c : ℤ, 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a + b + c = 0 ∧
    ab + bc + ca = -↑n ∧
    -a * b * c = ↑m * ↑n ∧
    (∀ (x y : ℕ), nat.coprime x y →
      (n = (x * x + x * y + y * y)^3 ∧ m = (x + y) * x * y))
    :=
begin
  sorry
end

end infinite_rel_prime_pairs_poly_l146_146455


namespace length_of_bridge_l146_146040

-- Define the conditions
def train_length : ℕ := 130 -- length of the train in meters
def train_speed : ℕ := 45  -- speed of the train in km/hr
def crossing_time : ℕ := 30  -- time to cross the bridge in seconds

-- Prove that the length of the bridge is 245 meters
theorem length_of_bridge : 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 245 := 
by
  sorry

end length_of_bridge_l146_146040


namespace triangle_tangent_value_l146_146407

noncomputable def triangle_tangent (XYZ : Type) [metric_space XYZ] 
  (X Y Z : XYZ) (angle_Y : angle X Y Z = π / 2) (YZ : dist Y Z = 4) (XZ : dist X Z = 5) : ℚ :=
4 / 3

theorem triangle_tangent_value (XYZ : Type) [metric_space XYZ]
  (X Y Z : XYZ) (angle_Y : angle X Y Z = π / 2) (YZ : dist Y Z = 4) (XZ : dist X Z = 5) :
  triangle_tangent XYZ X Y Z angle_Y YZ XZ = 4 / 3 := by
  sorry

end triangle_tangent_value_l146_146407


namespace game_fairness_l146_146109

theorem game_fairness (balls : Set ℕ) (total : ∑ b in balls, 1 = 10) :
  (count (λ x, x % 3 = 0) balls) / 10 ≠ (count (λ x, x % 5 = 0) balls) / 10 → 
  (count (λ x, x % 4 = 0) balls) / 10 = (count (λ x, x % 5 = 0) balls) / 10 :=
by
  sorry

end game_fairness_l146_146109


namespace solve_equation_l146_146186

theorem solve_equation :
  ∀ (x : ℝ), x ≥ 5 →
  (9 / (Real.sqrt (x - 5) - 10) +
   2 / (Real.sqrt (x - 5) - 5) +
   8 / (Real.sqrt (x - 5) + 5) +
   15 / (Real.sqrt (x - 5) + 10) = 0) ↔
  (x = 14 ∨ x = 1335 / 17) :=
by 
  intro x h
  sorry

end solve_equation_l146_146186


namespace cone_volume_ratio_l146_146795

theorem cone_volume_ratio (r l₁ l₂ h₁ h₂ : ℝ) (S_A S_B V_A V_B : ℝ)
  (h_r : r > 0)
  (h1 : S_A / S_B = 2 / 3)
  (h2 : l₁ = 2 * r)
  (h3 : l₁ / l₂ = 2 / 3)
  (h4 : h₁ = real.sqrt (l₁^2 - r^2))
  (h5 : h₂ = real.sqrt (l₂^2 - r^2))
  (h6 : V_A = (1 / 3) * real.pi * r^2 * h₁)
  (h7 : V_B = (1 / 3) * real.pi * r^2 * h₂) :
  V_A / V_B = real.sqrt 6 / 4 :=
sorry

end cone_volume_ratio_l146_146795


namespace B_and_C_mutually_exclusive_l146_146596

def Event : Type := Set (ℕ → Prop)

-- Definitions of events based on given enums
def A : Event := {e | e = λ _ => true}
def B : Event := {e | ∃ n, n > 5 → e n}
def C : Event := {e | ∃ n, 1 < n ∧ n < 6 → e n}
def D : Event := {e | ∃ n, 0 < n ∧ n < 6 → e n}

-- State the theorem that asserts B and C are mutually exclusive
theorem B_and_C_mutually_exclusive (A B C D : Event) : 
  Disjoint B C :=
by
  sorry

end B_and_C_mutually_exclusive_l146_146596


namespace smallest_difference_in_subsquare_l146_146828

theorem smallest_difference_in_subsquare :
  ∀ (grid : ℕ → ℕ → ℕ), 
    (∀ x y, 0 ≤ x ∧ x < 2018 ∧ 0 ≤ y ∧ y < 2018 → grid x y = 0 ∨ grid x y = 1) ∧
      (∃ x y, (∀ i j, 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10 → grid (x + i) (y + j) = 0)) ∧
      (∃ x y, (∀ i j, 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10 → grid (x + i) (y + j) = 1)) ->
    ∃ x y, abs ((∑ i in Finset.range 10, ∑ j in Finset.range 10, grid (x + i) (y + j)) - 50) ≤ 10 :=
begin
  sorry
end

end smallest_difference_in_subsquare_l146_146828


namespace hamburgers_sold_last_week_l146_146598

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end hamburgers_sold_last_week_l146_146598


namespace sum_of_ages_l146_146613

theorem sum_of_ages (a b c : ℕ) (h₁ : a = 20 + b + c) (h₂ : a^2 = 2050 + (b + c)^2) : a + b + c = 80 :=
sorry

end sum_of_ages_l146_146613


namespace probability_calculation_l146_146126

noncomputable def probability_two_of_four_approve : Prop :=
  let P_A := 0.6
  let P_D := 1 - P_A
  ∃ p : ℝ, p = (nat.choose 4 2) * (P_A ^ 2 * P_D ^ 2) ∧ p = 0.3456

theorem probability_calculation : probability_two_of_four_approve :=
by
  have exact_probability : (nat.choose 4 2) * (0.6 ^ 2 * 0.4 ^ 2) = 0.3456,
  {
    sorry,
  }
  use (nat.choose 4 2) * (0.6 ^ 2 * 0.4 ^ 2),
  split,
  { exact exact_probability },
  { exact exact_probability },

end probability_calculation_l146_146126


namespace trigonometric_identity_l146_146465

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : sin (α + π) = -sin α)
  (h₂ : cos (π + α) = -cos α)
  (h₃ : tan (-α - 2 * π) = tan (-α))
  (h₄ : cos (-α - π) = -cos α) :
  (sin (α + π))^2 * cos (π + α) /
  (tan (-α - 2 * π) * tan (π + α) * (cos (-α - π))^3) = -1 :=
  sorry

end trigonometric_identity_l146_146465


namespace sam_cleaner_meetings_two_times_l146_146914

open Nat

noncomputable def sam_and_cleaner_meetings (sam_rate cleaner_rate cleaner_stop_time bench_distance : ℕ) : ℕ :=
  let cycle_time := (bench_distance / cleaner_rate) + cleaner_stop_time
  let distance_covered_in_cycle_sam := sam_rate * cycle_time
  let distance_covered_in_cycle_cleaner := bench_distance
  let effective_distance_reduction := distance_covered_in_cycle_cleaner - distance_covered_in_cycle_sam
  let number_of_cycles_until_meeting := bench_distance / effective_distance_reduction
  number_of_cycles_until_meeting + 1

theorem sam_cleaner_meetings_two_times :
  sam_and_cleaner_meetings 3 9 40 300 = 2 :=
by sorry

end sam_cleaner_meetings_two_times_l146_146914


namespace sum_f_log_geometric_sequence_eq_2023_l146_146775

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 3 + 1

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

def given_sequence_conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1012 = 1 / 10

theorem sum_f_log_geometric_sequence_eq_2023
  (a : ℕ → ℝ) (h : given_sequence_conditions a) :
  ∑ k in Finset.range 2023, f (Real.log (a k)) = 2023 :=
by
  sorry

end sum_f_log_geometric_sequence_eq_2023_l146_146775


namespace inequality_proof_l146_146909

open Nat

-- Define the gcd and phi functions as they are used in the conditions
def my_gcd : ℕ → ℕ → ℕ := nat.gcd
def my_phi : ℕ → ℕ := nat.totient

-- Statement of the problem
theorem inequality_proof (m n : ℕ) :  
  my_phi (my_gcd (2^m + 1) (2^n + 1)) * 2 ^ my_gcd m n ≥ my_gcd (my_phi (2^m + 1)) (my_phi (2^n + 1)) * (my_gcd m n) :=
sorry

end inequality_proof_l146_146909


namespace max_white_out_of_place_cells_l146_146171

def is_neighbor {α : Type} [DecidableEq α] (board : α × α → bool) (cell₁ cell₂ : α × α) : Prop :=
  abs (cell₁.fst - cell₂.fst) ≤ 1 ∧ abs (cell₁.snd - cell₂.snd) ≤ 1 ∧ (cell₁ ≠ cell₂)

def is_out_of_place (board : (Fin 10) × (Fin 10) → bool) (cell : (Fin 10) × (Fin 10)) : Prop :=
  (∑ neighbors in ((λ c, is_neighbor board cell c) '' ((Fin 10) × (Fin 10)).univ), 
      ite (board cell ≠ board neighbors) 1 0) ≥ 7

theorem max_white_out_of_place_cells :
  ∃ (w_cells : Finset ((Fin 10) × (Fin 10))), 
    (∀ cell, cell ∈ w_cells → board cell) ∧ 
    (∀ cell, cell ∈ w_cells → is_out_of_place board cell) ∧ 
    w_cells.card ≤ 26 := 
sorry

end max_white_out_of_place_cells_l146_146171


namespace colten_chickens_l146_146003

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l146_146003


namespace shift_graph_l146_146064

def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)

def g (x : ℝ) : ℝ := 2 * sin (2 * x)

theorem shift_graph (x : ℝ) : f x = g (x + π / 6) :=
by
  sorry

end shift_graph_l146_146064


namespace power_equality_l146_146320

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146320


namespace calculate_998_pow_5_using_binomial_theorem_l146_146157

noncomputable def binomial_expansion (x : ℝ) (y : ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), (nat.choose n k) * (x^ (n - k)) * (y^ k)

theorem calculate_998_pow_5_using_binomial_theorem :
  abs ((binomial_expansion 10 (-0.02) 5) - 99004) < 1 :=
by
  -- the main binomial expansion based calculation
  let approximate_value := binomial_expansion 10 (-0.02) 5
  change abs (approximate_value - 99004) < 1
  -- skipping the proof steps, as it mainly involves computation and approximation
  sorry

end calculate_998_pow_5_using_binomial_theorem_l146_146157


namespace count_integers_ending_with_1_l146_146940

theorem count_integers_ending_with_1 (a b : ℕ) (h₁ : 208 ≤ a) (h₂ : b ≤ 2008) (h₃ : ∀ x, 211 ≤ x → x ≤ 2001 → (x % 10 = 1)) :
  |{x : ℕ | 211 ≤ x ∧ x ≤ 2001 ∧ x % 10 = 1}| = 180 :=
sorry

end count_integers_ending_with_1_l146_146940


namespace sales_decreased_by_1_percent_l146_146136

def sales_in_january (x : ℝ) := x
def sales_in_february (x : ℝ) := 1.1 * x
def sales_in_march (x : ℝ) := sales_in_february x * 0.9

theorem sales_decreased_by_1_percent (x : ℝ) :
    sales_in_march x = 0.99 * sales_in_january x :=
by unfold sales_in_january sales_in_february sales_in_march; sorry

end sales_decreased_by_1_percent_l146_146136


namespace typing_speed_ratio_l146_146549

theorem typing_speed_ratio (T t : ℝ) (h1 : T + t = 12) (h2 : T + 1.25 * t = 14) : t / T = 2 :=
by
  sorry

end typing_speed_ratio_l146_146549


namespace school_cases_of_water_l146_146108

theorem school_cases_of_water (bottles_per_case bottles_used_first_game bottles_left_after_second_game bottles_used_second_game : ℕ)
  (h1 : bottles_per_case = 20)
  (h2 : bottles_used_first_game = 70)
  (h3 : bottles_left_after_second_game = 20)
  (h4 : bottles_used_second_game = 110) :
  let total_bottles_used := bottles_used_first_game + bottles_used_second_game
  let total_bottles_initial := total_bottles_used + bottles_left_after_second_game
  let number_of_cases := total_bottles_initial / bottles_per_case
  number_of_cases = 10 :=
by
  -- The proof goes here
  sorry

end school_cases_of_water_l146_146108


namespace sample_standard_deviation_apple_weights_l146_146739

/-- Define the sample standard deviation function for a given list of weights -/
def sample_standard_deviation (weights : List ℝ) : ℝ :=
  let mean := (weights.sum / weights.length)
  let variance := (weights.map (λ w => (w - mean) ^ 2)).sum / (weights.length - 1)
  real.sqrt variance

/-- Given list of weights -/
def apple_weights : List ℝ := [125, 124, 121, 123, 127]

/-- The sample standard deviation of the given weights is 2 grams. -/
theorem sample_standard_deviation_apple_weights : sample_standard_deviation apple_weights = 2 :=
  sorry

end sample_standard_deviation_apple_weights_l146_146739


namespace binomial_coefficient_9_5_l146_146666

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l146_146666


namespace largest_multiple_of_15_less_than_500_l146_146986

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146986


namespace binom_9_5_eq_126_l146_146682

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146682


namespace binom_9_5_eq_126_l146_146684

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146684


namespace binom_9_5_eq_126_l146_146663

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146663


namespace polar_equation_of_curve_l146_146943

theorem polar_equation_of_curve :
  (∃ t : ℝ, x = 1 + (real.sqrt 3) * t ∧ y = (real.sqrt 3) - t) →
  (∃ ρ θ : ℝ, x = ρ * real.cos θ ∧ y = ρ * real.sin θ ∧
  ρ * (real.sin (θ + (real.pi / 6))) = 2) :=
sorry

end polar_equation_of_curve_l146_146943


namespace theta_in_third_quadrant_l146_146274

noncomputable def condition (θ : Real) : Prop :=
  (cos θ) / (sqrt (1 + (tan θ) ^ 2)) + (sin θ) / (sqrt (1 + (cot θ) ^ 2)) = -1

theorem theta_in_third_quadrant (θ : Real) (h : condition θ) : π / 2 < θ ∧ θ < π :=
sorry

end theta_in_third_quadrant_l146_146274


namespace power_equality_l146_146329

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146329


namespace lines_intersect_at_point_l146_146965

noncomputable def intersect_point : (ℝ × ℝ) :=
  (17 / 4, 9 / 4)

theorem lines_intersect_at_point :
  ∃ t u : ℝ,
    (∀ x y : ℝ,
      (x, y) = (2 + 3 * t, 3 - t) →
      (x, y) = (4 + u, 1 + 5 * u) →
      (x, y) = intersect_point) :=
begin
  sorry
end

end lines_intersect_at_point_l146_146965


namespace power_of_three_l146_146280

theorem power_of_three (y : ℝ) (hy : 3^y = 81) : 3^(y + 3) = 2187 := 
by {
  sorry,
}

end power_of_three_l146_146280


namespace max_integer_value_of_expression_l146_146815

theorem max_integer_value_of_expression (x : ℝ) :
  ∃ M : ℤ, M = 15 ∧ ∀ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ M :=
sorry

end max_integer_value_of_expression_l146_146815


namespace terms_before_4_l146_146711

theorem terms_before_4 (a n : ℕ) (d : ℤ) (h₀ : a = 92) (h₁ : d = -3) :
  ∀ n, (92 - (3 * n) = 4) → n = 30 :=
begin
  intros n h,
  rw [h₀, h₁] at h,
  sorry,
end

end terms_before_4_l146_146711


namespace arithmetic_sequence_general_term_T_n_formula_l146_146163

variable (a b : ℕ → ℕ)

-- Conditions
def condition1 (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := 2 * a 5 - S 4 = 2
def condition2 (a : ℕ → ℕ) : Prop := 3 * a 2 + a 6 = 32

-- General term
def generalTerm (n : ℕ) : ℕ := 3 * n - 1

-- Define the sum T
def T (n : ℕ) : ℕ := ∑ k in Finset.range n, (generalTerm k.succ / 2^k.succ)

-- Desired formula for T_n
def desiredT (n : ℕ) : ℕ := 5 - (3 * n + 5) / 2^n

theorem arithmetic_sequence_general_term :
  (∀ n : ℕ, a n = generalTerm n) →
  condition1 a (λ n, (Finset.range n).sum a) →
  condition2 a →
  (∀ n : ℕ, a n = generalTerm n) :=
by
  intros h1 h2 h3
  sorry

theorem T_n_formula :
  (∀ n : ℕ, a n = generalTerm n) →
  condition1 a (λ n, (Finset.range n).sum a) →
  condition2 a →
  (∀ n : ℕ, T n = desiredT n) :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_general_term_T_n_formula_l146_146163


namespace mean_age_correct_l146_146029

def children_ages : List ℕ := [6, 6, 9, 12]

def number_of_children : ℕ := 4

def sum_of_ages (ages : List ℕ) : ℕ := ages.sum

def mean_age (ages : List ℕ) (num_children : ℕ) : ℚ :=
  sum_of_ages ages / num_children

theorem mean_age_correct :
  mean_age children_ages number_of_children = 8.25 := by
  sorry

end mean_age_correct_l146_146029


namespace polynomial_eq_of_sum_eq_l146_146878

variable {k : Type*} [Field k] {n : ℕ}
variable (f : Polynomial k) (α : k)
variable (c d : Fin n → k)

-- Conditions: 
-- 1. f(x) is an irreducible polynomial of degree n over field k
-- 2. α is one of the roots of f(x)
-- 3. ∑_{m=0}^{n-1} c_m α^m = ∑_{m=0}^{n-1} d_m α^m

theorem polynomial_eq_of_sum_eq {k : Type*} [Field k] {n : ℕ}
    (f : Polynomial k) (α : k) (c d : Fin n → k)
    (hf_deg : degree f = n)
    (hf_irr : Irreducible f)
    (h_root : Polynomial.eval α f = 0)
    (h_eq : (Finset.range n).sum (λ (m : ℕ), c ⟨m, Fin.isLt m n⟩ * α^m) =
            (Finset.range n).sum (λ (m : ℕ), d ⟨m, Fin.isLt m n⟩ * α^m)) :
    ∀ m : Fin n, c m = d m := sorry

end polynomial_eq_of_sum_eq_l146_146878


namespace intersection_of_M_and_N_l146_146267

-- Define the sets M and N
def M : set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def N : set ℝ := { y | ∃ x, y = 2 * x ∧ -1 ≤ x ∧ x ≤ 1 }

-- Statement that intersection of sets M and N is { x | -1 ≤ x ≤ 1 }
theorem intersection_of_M_and_N : (M ∩ N) = { x | -1 ≤ x ∧ x ≤ 1 } :=
  sorry

end intersection_of_M_and_N_l146_146267


namespace find_original_price_l146_146410

def initial_price (P : ℝ) : Prop :=
  let first_discount := P * 0.76
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 1.10
  final_price = 532

theorem find_original_price : ∃ P : ℝ, initial_price P :=
sorry

end find_original_price_l146_146410


namespace compute_combination_l146_146688

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l146_146688


namespace part_a_part_b_l146_146098

-- Definitions of the points and lines as described in the problem
variable (A B P Q R T S K L : Type)
variable [IsLine l]
variable (a b l : Type) [IsLine a] [IsLine b]
variable (h_a_through_A : LineThrough a A)
variable (h_b_through_B : LineThrough b B)
variable (h_a_perpendicular : Perpendicular a l)
variable (h_b_perpendicular : Perpendicular b l)
variable (h_line_P_through : ¬ LCollinear P l)
variable (h_line_intersect_a : LineIntersects pq a Q)
variable (h_line_intersect_b : LineIntersects pq b R)
variable (h_line_A_perpendicular_BQ : Perpendicular A BQ)
variable (h_T_intersect_BQ_BR : IntersectPoints (LineThrough A ⟂ BQ) BQ = L ∧ IntersectPoints (LineThrough A ⟂ BQ) BR = T)
variable (h_line_B_perpendicular_AR : Perpendicular B AR)
variable (h_K_intersect_AQ_AR : IntersectPoints (LineThrough B ⟂ AR) AR = K ∧ IntersectPoints (LineThrough B ⟂ AR) AQ = S)

-- Proof to show P, T, S are collinear
theorem part_a : Collinear P T S := by
  sorry

-- Proof to show P, K, L are collinear
theorem part_b : Collinear P K L := by
  sorry

end part_a_part_b_l146_146098


namespace exponent_power_identity_l146_146333

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146333


namespace constant_function_l146_146219

-- Given conditions
variables {a : ℝ} (h : a > 0) {f : ℝ → ℝ}
  (hf : f a = 1)
  (hf_eq : ∀ x y : ℝ, x > 0 → y > 0 → f(x) * f(y) + f (a / x) * f (a / y) = 2 * f(x * y))

-- The goal is to prove that f is a constant function, specifically f(x) = 1 for all x > 0
theorem constant_function (x : ℝ) (hx : x > 0) : f x = 1 :=
sorry

end constant_function_l146_146219


namespace determinant_simplifies_to_zero_l146_146648

theorem determinant_simplifies_to_zero (a b : ℝ) :
  matrix.det ![
    ![1, real.sin (a - b), real.sin a],
    ![real.sin (a - b), 1, real.sin b],
    ![real.sin a, real.sin b, 1]
  ] = 0 := 
by
  sorry

end determinant_simplifies_to_zero_l146_146648


namespace number_of_valid_ns_l146_146135

/-- Define the main problem -/
def valid_n (n : ℕ) : Prop :=
  n ≥ 9 ∧ n ≤ 2017 ∧
  (∃ T : ℕ, 
    (T * ((n - 5).choose 4) / (n.choose 9)) * 
    (T * ((n - 5).choose 3) / (n.choose 8)) = 1)

theorem number_of_valid_ns : 
  {n : ℕ | valid_n n}.to_finset.card = 557 := 
sorry

end number_of_valid_ns_l146_146135


namespace population_change_l146_146502

-- Define the conditions
variable (Population : Type)
variable (SexRatio : Population → ℝ) -- Sex ratio is a real number function of the population
variable (BirthRate : Population → ℝ) -- Birth rate is a real number function of the population
variable (Density : Population → ℝ) -- Density is a real number function of the population
variable (disruption : Population → Population) -- Disruption by luring and killing male adults

-- Hypotheses based on the conditions
axiom h1 : ∀ p : Population, disruption p affects SexRatio p
axiom h2 : ∀ p : Population, (SexRatio p) affects (BirthRate p)
axiom h3 : ∀ p : Population, (BirthRate p) affects (Density p)
axiom h4 : ∀ p : Population, (disruption p) decreases (SexRatio p)
axiom h5 : ∀ p : Population, (decreased SexRatio p) decreases (BirthRate p)
axiom h6 : ∀ p : Population, (decreased BirthRate p) significantly reduces (Density p)

-- Theorem to prove
theorem population_change (p : Population) : (Density (disruption p)) significantly reduces :=
by sorry

end population_change_l146_146502


namespace probability_at_least_three_same_value_l146_146736

theorem probability_at_least_three_same_value :
  let p := 7 / 72
  in let fair_dice (n : ℕ) := ∀ i : fin n, 1 ≤ i.1 + 1 ∧ i.1 + 1 ≤ 6
  in fair_dice 4 → p = 7 / 72 :=
by
  sorry

end probability_at_least_three_same_value_l146_146736


namespace power_of_three_l146_146366

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l146_146366


namespace range_of_a_l146_146255

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l146_146255


namespace remainder_is_l146_146939

def prime1 : ℕ := 43
def prime2 : ℕ := 47
def n : ℕ := 2024
def S : ℕ := ∑ k in finset.range 65, nat.choose n k
def mod_value : ℕ := prime1 * prime2

theorem remainder_is :
  S % mod_value = 1089 := 
  sorry

end remainder_is_l146_146939


namespace hundredth_decimal_place_of_fraction_l146_146078

theorem hundredth_decimal_place_of_fraction : 
  let repeating_decimal := "09"
  ∀ (n : Nat), n = 100 → ((n % String.length repeating_decimal) = 0 → String.get repeating_decimal 1 = '9') :=
by sorry

end hundredth_decimal_place_of_fraction_l146_146078


namespace kite_area_proof_l146_146456

noncomputable def is_kite (AB AD BC CD : ℝ) (angleB : ℝ) := 
  AB = AD ∧ BC = CD ∧ angleB = 90

noncomputable def region_area (AB AD BC CD : ℝ) (angleB : ℝ) (R : ℝ) : Prop :=
  is_kite AB AD BC CD angleB → R = 6.25

theorem kite_area_proof :
  ∀ (AB AD BC CD : ℝ) (angleB : ℝ), 
    is_kite AB AD BC CD angleB → 
    region_area AB AD BC CD angleB 6.25 := 
begin
  sorry -- Proof omitted
end

end kite_area_proof_l146_146456


namespace steps_in_staircase_l146_146526

theorem steps_in_staircase :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n = 19 :=
by
  sorry

end steps_in_staircase_l146_146526


namespace apples_target_l146_146858

theorem apples_target (initial_each : ℕ) (additional_rate : ℚ) (remaining_each : ℕ) :
  initial_each = 400 ∧ additional_rate = 3 / 4 ∧ remaining_each = 600 →
  2 * (initial_each + additional_rate * initial_each + remaining_each) = 2600 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2, h3]
  calc
    2 * (initial_each + additional_rate * initial_each + remaining_each)
    = 2 * (400 + (3 / 4) * 400 + 600) : by simp [h1, h2, h3]
    ... = 2 * (400 + 300 + 600) : by norm_num
    ... = 2 * 1300 : by norm_num
    ... = 2600 : by norm_num

end apples_target_l146_146858


namespace roots_negative_iff_l146_146211

theorem roots_negative_iff (p : ℝ) :
  (∀ x, x^2 + 2*(p+1)*x + (9*p - 5) < 0) ↔ p ∈ Set.Icc (5/9) 1 ∪ Set.Icc 6 ∞ := by
  sorry

end roots_negative_iff_l146_146211


namespace trains_total_distance_l146_146968

theorem trains_total_distance (speed_A speed_B : ℝ) (time_A time_B : ℝ) (dist_A dist_B : ℝ):
  speed_A = 90 ∧ 
  speed_B = 120 ∧ 
  time_A = 1 ∧ 
  time_B = 5/6 ∧ 
  dist_A = speed_A * time_A ∧ 
  dist_B = speed_B * time_B ->
  (dist_A + dist_B) = 190 :=
by 
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  sorry

end trains_total_distance_l146_146968


namespace albert_complete_laps_l146_146145

theorem albert_complete_laps (D L : ℝ) (I : ℕ) (hD : D = 256.5) (hL : L = 9.7) (hI : I = 6) :
  ⌊(D - I * L) / L⌋ = 20 :=
by
  sorry

end albert_complete_laps_l146_146145


namespace power_equality_l146_146325

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l146_146325


namespace determinant_simplifies_to_zero_l146_146645

theorem determinant_simplifies_to_zero (a b : ℝ) :
  matrix.det ![
    ![1, real.sin (a - b), real.sin a],
    ![real.sin (a - b), 1, real.sin b],
    ![real.sin a, real.sin b, 1]
  ] = 0 := 
by
  sorry

end determinant_simplifies_to_zero_l146_146645


namespace fraction_meaningful_range_l146_146506

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_range_l146_146506


namespace midpoints_quadrilateral_is_parallelogram_l146_146087

/-- The quadrilateral obtained by connecting the midpoints of the sides of any quadrilateral in sequence is a parallelogram. -/
theorem midpoints_quadrilateral_is_parallelogram (Q : Type) [quadrilateral Q] :
  is_parallelogram (connect_midpoints Q) :=
sorry

end midpoints_quadrilateral_is_parallelogram_l146_146087


namespace price_of_second_variety_l146_146923

-- Define prices and conditions
def price_first : ℝ := 126
def price_third : ℝ := 175.5
def mixture_price : ℝ := 153
def total_weight : ℝ := 4

-- Define unknown price
variable (x : ℝ)

-- Definition of the weighted mixture price
theorem price_of_second_variety :
  (1 * price_first) + (1 * x) + (2 * price_third) = total_weight * mixture_price →
  x = 135 :=
by
  sorry

end price_of_second_variety_l146_146923


namespace part_a_part_b_l146_146875

open EuclideanGeometry

variables
  (I : Point)
  (ABC : Triangle)
  (non_equilateral : ¬(Equilateral ABC))
  (I_A : Point)
  (I_A' : Point)
  (isReflectionA : Reflection I_A (line (BC ABC)) I_A')
  (AI_A' : Line)
  (isReflectionLineA : Reflection AI_A' (AI ABC) (line I_A' I))
  (I_B I_B' : Point)
  (lineIB' : Line)
  (isReflectionB : Reflection I_B (line (BC ABC)) I_B')
  (isReflectionLineB : Reflection lineIB' (AI ABC) (line I_B' I))
  (P : Point)
  (intersectionA_B : Intersection (line I_A' I) (line I_B' I) P)
  (O : Point)
  (circumcenter : Circumcenter ABC O)
  (tangent : Tangent P (incircle ABC))
  (Xa Ya : Point)
  (onCircumcircleX : OnCircumcircle (circumcircle ABC) Xa)
  (onCircumcircleY : OnCircumcircle (circumcircle ABC) Ya)
  (tangent_pp : Tangent (incircle ABC) Xa Ya)

-- Part (a)
theorem part_a (lies_on_OI : LiesOn P (line O I)) : LiesOn P (line O I) := sorry

-- Part (b)
theorem part_b (angle_condition : Angle XIY = 120) : ∠X I Y = 120 := sorry

end part_a_part_b_l146_146875


namespace distance_focus_to_asymptote_l146_146188

-- The condition of the parabola
def parabola := ∀ x y : ℝ, y^2 = 4 * x

-- The condition of the hyperbola
def hyperbola := ∀ x y : ℝ, (y^2) / 3 - x^2 = 1

-- The statement to prove the distance from the focus of the parabola to the asymptote of the hyperbola
theorem distance_focus_to_asymptote
  (h_parabola : parabola)
  (h_hyperbola : hyperbola)
  : ∀ x y : ℝ, dist (1, 0) (0, sqrt 3) = sqrt 3 / 2 := by
  sorry

end distance_focus_to_asymptote_l146_146188


namespace revised_problem_l146_146803

theorem revised_problem (x : ℝ) (h : 3^(2*x) = 6 * 2^(2*x) - 5 * 6^x) : 3^x = 2^x :=
sorry

end revised_problem_l146_146803


namespace area_of_triangle_l146_146972

-- Define the base and height from the given conditions
def base (t : ℕ) : ℕ := 2 * t
def height (t : ℕ) : ℕ := 3 * t + 2

-- Define the formula for the area given base and height
def area (t : ℕ) : ℕ := (base t) * (height t) / 2

/-- Theorem: The area of a triangle with base 2t and height 3t + 2, when t = 6, is 120. -/
theorem area_of_triangle (t : ℕ) (h : t = 6) : area t = 120 :=
by
  rw [h]
  sorry

end area_of_triangle_l146_146972


namespace total_plants_in_garden_l146_146629

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l146_146629


namespace binom_9_5_l146_146677

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l146_146677


namespace exponent_power_identity_l146_146343

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l146_146343


namespace sum_of_numbers_divisible_by_2_3_or_5_l146_146545

-- Definitions of sets A, B, and C
def S := { n : ℕ | n < 30 }
def A := { n : ℕ | n ∈ S ∧ n % 2 = 0 }
def B := { n : ℕ | n ∈ S ∧ n % 3 = 0 }
def C := { n : ℕ | n ∈ S ∧ n % 5 = 0 }

-- The proof problem
theorem sum_of_numbers_divisible_by_2_3_or_5 : 
  (∑ n in (A ∪ B ∪ C), n) = 301 :=
by {
  sorry
}

end sum_of_numbers_divisible_by_2_3_or_5_l146_146545


namespace find_base_l146_146381
-- Import the necessary library

-- Define the conditions and the result
theorem find_base (x y b : ℕ) (h1 : x - y = 9) (h2 : x = 9) (h3 : b^x * 4^y = 19683) : b = 3 :=
by
  sorry

end find_base_l146_146381


namespace evaluate_expression_l146_146175

theorem evaluate_expression (b : ℕ) (h : b = 4) : b^3 * b^6 * 2 = 524288 := by
  have h1 : b^9 = 262144 := by sorry
  show b^3 * b^6 * 2 = 524288 from by sorry

end evaluate_expression_l146_146175


namespace probability_of_three_tails_in_a_row_l146_146198

def prob_tail_fair : ℚ := 1 / 2
def prob_tail_biased : ℚ := 1 / 3

def at_least_three_tails_prob :=
  have p1 : ℚ := prob_tail_fair * prob_tail_fair * prob_tail_biased,
  have p2 : ℚ := prob_tail_fair * prob_tail_biased * prob_tail_fair,
  have p3 : ℚ := prob_tail_biased * prob_tail_fair * prob_tail_fair,
  have p4 : ℚ := prob_tail_fair * prob_tail_fair * prob_tail_biased * prob_tail_fair,
  have p5 : ℚ := prob_tail_biased * prob_tail_fair * prob_tail_fair * prob_tail_fair,
  have p6 : ℚ := prob_tail_fair * prob_tail_fair * prob_tail_biased * prob_tail_fair * prob_tail_fair,
  p1 + p2 + p3 + p4 + p5 + p6

theorem probability_of_three_tails_in_a_row : at_least_three_tails_prob = 13 / 48 :=
sorry

end probability_of_three_tails_in_a_row_l146_146198


namespace angle_E_degree_l146_146477

-- Given conditions
variables {E F G H : ℝ} -- degrees of the angles in quadrilateral EFGH

-- Condition 1: The angles satisfy a specific ratio
axiom angle_ratio : E = 3 * F ∧ E = 2 * G ∧ E = 6 * H

-- Condition 2: The sum of the angles in the quadrilateral is 360 degrees
axiom angle_sum : E + (E / 3) + (E / 2) + (E / 6) = 360

-- Prove the degree measure of angle E is 180 degrees
theorem angle_E_degree : E = 180 :=
by
  sorry

end angle_E_degree_l146_146477


namespace isosceles_triangle_area_l146_146030

theorem isosceles_triangle_area (b s : ℝ) (h_perimeter : s + b = 20) (h_pythagorean : b^2 + 10^2 = s^2) : 
  let area := b * 10 in 
  area = 75 :=
by
  sorry

end isosceles_triangle_area_l146_146030


namespace eval_f_at_neg_pi_over_24_l146_146783

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) * Real.cos x

theorem eval_f_at_neg_pi_over_24 : f (-π / 24) = (2 + Real.sqrt 2) / 4 := by
  sorry

end eval_f_at_neg_pi_over_24_l146_146783


namespace ages_of_children_l146_146114

theorem ages_of_children : ∃ (a1 a2 a3 a4 : ℕ),
  a1 + a2 + a3 + a4 = 33 ∧
  (a1 - 3) + (a2 - 3) + (a3 - 3) + (a4 - 3) = 22 ∧
  (a1 - 7) + (a2 - 7) + (a3 - 7) + (a4 - 7) = 11 ∧
  (a1 - 13) + (a2 - 13) + (a3 - 13) + (a4 - 13) = 1 ∧
  a1 = 14 ∧ a2 = 11 ∧ a3 = 6 ∧ a4 = 2 :=
by
  sorry

end ages_of_children_l146_146114


namespace impossible_prime_decomposition_l146_146486

noncomputable def num_consecutive_digits := 2021

def is_base_5 (n : ℕ) : Prop :=
  ∀ d ∈ digits 5 n, d < 5

def is_compatible_sequence (s : List ℕ) : Prop :=
  List.length s = num_consecutive_digits ∧
  ∀ i, i < List.length s → (
    (i % 2 = 0 → s[i] = s[i+1] + 1) ∧
    (i % 2 = 1 → s[i] = s[i-1] - 1))

def large_number_from_digits (s : List ℕ) : ℕ :=
  s.foldl (λ acc d, acc * 5 + d) 0

theorem impossible_prime_decomposition :
  ∀ s : List ℕ, is_compatible_sequence s →
  let x := large_number_from_digits s in
  ∀ u v : ℕ, Prime u → Prime v → u ≠ v →
  u * v = x → (v = u + 2) → False :=
begin
  intros s hs x u v hu hv hne huv hvu,
  -- proof goes here
  sorry
end

end impossible_prime_decomposition_l146_146486


namespace lowest_possible_sale_price_is_30_percent_l146_146607

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end lowest_possible_sale_price_is_30_percent_l146_146607


namespace find_f_l146_146214

theorem find_f (f : ℝ → ℝ) :
  (∀ x, f(2 * x - 3) = x ^ 2 + x + 1) →
  (∀ x, f(x) = (1 / 4) * x ^ 2 + 2 * x + 19 / 4) :=
by
  intros h x
  sorry

end find_f_l146_146214


namespace midpoint_locus_of_parallel_planes_l146_146196

-- Define the planes as two parallel planes
structure Plane := 
  (normal : ℝ × ℝ × ℝ)
  (const : ℝ)

-- Parallel planes:
def parallel (α β : Plane) : Prop := α.normal = β.normal ∧ α.const ≠ β.const

-- Midpoint between two points
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

-- The main theorem
theorem midpoint_locus_of_parallel_planes (α β : Plane) (h : parallel α β): 
  ∃ γ : Plane, 
  parallel α γ ∧ 
  ∀ A B : ℝ × ℝ × ℝ, 
    A ∈ α.set → B ∈ β.set → 
    (midpoint A B ∈ γ.set) :=
sorry

end midpoint_locus_of_parallel_planes_l146_146196


namespace initial_labels_invariance_l146_146957

variable (n : ℕ) (labels : Fin (2 * n - 3) → ℝ)

-- Define what it means for a coloring to be Ptolemaic
def ptolemaic_coloring (n : ℕ) (labels : Fin (2 * n - 3) → ℝ) : Prop :=
  ∀ (A B C D : Fin n),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D →
    let ab := labels (⟨(2 * n - 3) - |j|⟩ : Fin (2 * n - 3)), -- use appropriate label indexing
        bc := labels (⟨(2 * n - 3) - |k|⟩ : Fin (2 * n - 3)),
        cd := labels (⟨(2 * n - 3) - |m|⟩ : Fin (2 * n - 3)),
        da := labels (⟨(2 * n - 3) - |n|⟩ : Fin (2 * n - 3)),
        ac := labels (⟨(2 * n - 3) - |p|⟩ : Fin (2 * n - 3)),
        bd := labels (⟨(2 * n - 3) - |q|⟩ : Fin (2 * n - 3))
    in (ab * cd + da * bc = ac * bd)

-- Define the move operation on the quadrilateral
def move_coloring (A B C D : Fin n) (x y z t w : ℝ) : ℝ :=
  (x * z + y * t) / w

theorem initial_labels_invariance
  (initial_labels : Fin (2 * n - 3) → ℝ)
  (moves : List (Fin n × Fin n × Fin n × Fin n))
  [∀ A B C D, ∀ labels (A B C D : Fin n), 
      A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D → 
      move_coloring labels (A B C D : Fin n)] :
  initial_labels = labels :=
by
  sorry

end initial_labels_invariance_l146_146957


namespace gear_B_turns_l146_146090

theorem gear_B_turns (teeth_A teeth_B turns_A: ℕ) (h₁: teeth_A = 6) (h₂: teeth_B = 8) (h₃: turns_A = 12) :
(turn_A * teeth_A) / teeth_B = 9 :=
by  sorry

end gear_B_turns_l146_146090


namespace target_apples_l146_146861

-- Definitions based on the conditions
def Kaiden_first_round_apples : ℕ := 400
def Adriel_first_round_apples : ℕ := 400
def fraction_of_first_round : ℚ := 3 / 4
def additional_apples_needed_each : ℕ := 600

-- Prove that the total target number of apples picked is 2600
theorem target_apples : 
  let first_round_each := Kaiden_first_round_apples,
      second_round_each := fraction_of_first_round * first_round_each,
      total_first_second_round_each := first_round_each + second_round_each,
      additional_apples_each := additional_apples_needed_each
  in 
    2 * (total_first_second_round_each + additional_apples_each) = 2600 :=
by
  sorry

end target_apples_l146_146861


namespace consecutive_even_integers_sum_l146_146503

theorem consecutive_even_integers_sum (n : ℕ) (h : n % 2 = 0) (h_pro : n * (n + 2) * (n + 4) = 3360) :
  n + (n + 2) + (n + 4) = 48 :=
by sorry

end consecutive_even_integers_sum_l146_146503


namespace range_of_m_if_p_and_q_is_false_l146_146762

open Real

variable (m : ℝ)

def p : Prop := ∃ m : ℝ, m + 1 ≤ 0

def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m_if_p_and_q_is_false :
  ¬(p ∧ q) → (m ≤ -2 ∨ m > -1) :=
sorry

end range_of_m_if_p_and_q_is_false_l146_146762


namespace minimum_value_C2_minus_D2_l146_146429

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 11))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 4)) + (Real.sqrt (z + 9))

theorem minimum_value_C2_minus_D2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (C x y z)^2 - (D x y z)^2 ≥ 36 := by
  sorry

end minimum_value_C2_minus_D2_l146_146429


namespace starling_nests_flying_condition_l146_146827

theorem starling_nests_flying_condition (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → A ≠ B)
  (h3 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → 
  (∃ d1 d2 : ℝ, d1 < d2 ∧ d1 < d2 → d1 > d2)) : n = 3 :=
by
  sorry

end starling_nests_flying_condition_l146_146827


namespace projection_of_b_in_direction_of_a_l146_146767

-- Given conditions
variables (a b : ℝ) 
|a| = 1
b = (real.sqrt 3 / 3, real.sqrt 3 / 3)
|a + 3 * b| = 2

-- The projection of b in the direction of a
#check
theorem projection_of_b_in_direction_of_a :
  projection a b = -1/2 :=
sorry

end projection_of_b_in_direction_of_a_l146_146767


namespace hamburgers_sold_last_week_l146_146599

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end hamburgers_sold_last_week_l146_146599


namespace find_sum_of_a_and_b_l146_146246

theorem find_sum_of_a_and_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (abs (x^2 - 2 * a * x + b) = 8) → (x = a ∨ x = a + 4 ∨ x = a - 4))
  (h2 : a^2 + (a - 4)^2 = (a + 4)^2) :
  a + b = 264 :=
by
  sorry

end find_sum_of_a_and_b_l146_146246


namespace problem_real_values_of_x_l146_146185

theorem problem_real_values_of_x (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) : 
  (2 / (x + 1) + 8 / (x + 3) < 3) ↔ (x ∈ Set.Ioo (-3 : ℝ) (-1) ∪ Set.Ioo (-1/3) (5)) :=
begin
  sorry
end

end problem_real_values_of_x_l146_146185


namespace binom_9_5_eq_126_l146_146660

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l146_146660


namespace part_I_part_II_l146_146894

-- Let the volume V of the tetrahedron ABCD be given
def V : ℝ := sorry

-- Areas of the faces opposite vertices A, B, C, D
def S_A : ℝ := sorry
def S_B : ℝ := sorry
def S_C : ℝ := sorry
def S_D : ℝ := sorry

-- Definitions of the edge lengths and angles
def a : ℝ := sorry -- BC
def a' : ℝ := sorry -- DA
def b : ℝ := sorry -- CA
def b' : ℝ := sorry -- DB
def c : ℝ := sorry -- AB
def c' : ℝ := sorry -- DC
def alpha : ℝ := sorry -- Angle between BC and DA
def beta : ℝ := sorry -- Angle between CA and DB
def gamma : ℝ := sorry -- Angle between AB and DC

theorem part_I : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 = 
  (1 / 4) * ((a * a' * Real.sin alpha)^2 + (b * b' * Real.sin beta)^2 + (c * c' * Real.sin gamma)^2) := 
  sorry

theorem part_II : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 ≥ 9 * (3 * V^4)^(1/3) :=
  sorry

end part_I_part_II_l146_146894


namespace largest_multiple_of_15_under_500_l146_146997

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146997


namespace sequence_of_perfect_squares_l146_146920

theorem sequence_of_perfect_squares (A B C D: ℕ)
(h1: 10 ≤ 10 * A + B) 
(h2 : 10 * A + B < 100) 
(h3 : (10 * A + B) % 3 = 0 ∨ (10 * A + B) % 3 = 1)
(hC : 1 ≤ C ∧ C ≤ 9)
(hD : 1 ≤ D ∧ D ≤ 9)
(hCD : (C + D) % 3 = 0)
(hAB_square : ∃ k₁ : ℕ, k₁^2 = 10 * A + B) 
(hACDB_square : ∃ k₂ : ℕ, k₂^2 = 1000 * A + 100 * C + 10 * D + B) 
(hACCDDB_square : ∃ k₃ : ℕ, k₃^2 = 100000 * A + 10000 * C + 1000 * C + 100 * D + 10 * D + B) :
∀ n: ℕ, ∃ k : ℕ, k^2 = (10^n * A + (10^(n/2) * C) + (10^(n/2) * D) + B) := 
by
  sorry

end sequence_of_perfect_squares_l146_146920


namespace binom_9_5_eq_126_l146_146683

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l146_146683


namespace equal_angles_1_equal_angles_2_l146_146580

open EuclideanGeometry AngleMeasure Circle

variables {A B C : Point} {O : Point} {r : ℝ}
variables {TA TC : Line}

-- Define a circle, a triangle inscribed in the circle, and a tangent line
def circle := Circle.mk O r
def triangle := Triangle.mk A B C
def tangent_at_A := TangentLine circle A

-- Hypotheses based on conditions
hypothesis (h_triangle : triangle.inscribed_in circle)
hypothesis (h_tangent_A : tangent_at_A = TA)

-- Define the pairs of angles to be proven equal
def angle_TAB := Angle.mk TA (Line.mk A B)
def angle_TA := Angle.mk TA (Line.mk A C)
def angle_ACB := Angle.mk (Line.mk A C) (Line.mk B C)
def angle_ABC := Angle.mk (Line.mk A B) (Line.mk B C)

-- Theorem statements based on the problem description
theorem equal_angles_1 (h : tangent_segt_thm triangle circle A TA (Line.mk A B)) : angle_TAB = angle_ACB := by
  sorry

theorem equal_angles_2 (h : tangent_segt_thm triangle circle A TA (Line.mk A C)) : angle_TA = angle_ABC := by
  sorry

end equal_angles_1_equal_angles_2_l146_146580


namespace largest_multiple_of_15_less_than_500_l146_146982

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146982


namespace max_number_of_triples_l146_146058

-- Define the number of points
def num_points : ℕ := 1955

-- Define the max number of triples given the condition
def max_triples_condition := ∀ (triples : list (finset ℕ)), 
  (∀ t1 t2 ∈ triples, t1 ≠ t2 → (t1 ∩ t2).card = 1) → triples.length ≤ 977

theorem max_number_of_triples (h : max_triples_condition) : ∃ (triples : list (finset ℕ)), 
  (∀ t1 t2 ∈ triples, t1 ≠ t2 → (t1 ∩ t2).card = 1) ∧ triples.length = 977 :=
sorry

end max_number_of_triples_l146_146058


namespace sequence_general_term_l146_146263

open Nat

/-- Define the sequence recursively -/
def a : ℕ → ℤ
| 0     => -1
| (n+1) => 3 * a n - 1

/-- The general term of the sequence is given by - (3^n - 1) / 2 -/
theorem sequence_general_term (n : ℕ) : a n = - (3^n - 1) / 2 := 
by
  sorry

end sequence_general_term_l146_146263


namespace probability_of_picking_letter_in_mathematics_l146_146812

theorem probability_of_picking_letter_in_mathematics (total_letters : ℕ) (unique_letters : ℕ) (word : list Char)
  (h_total_letters : total_letters = 26)
  (h_unique_letters : unique_letters = 8)
  (h_word : word = "MATHEMATICS".toList) :
  (↑unique_letters / ↑total_letters : ℚ) = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l146_146812


namespace triangle_perimeter_l146_146388

theorem triangle_perimeter (x : ℕ) :
  (x = 6 ∨ x = 3) →
  ∃ (a b c : ℕ), (a = x ∧ (b = x ∨ c = x)) ∧ 
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by
  intro h
  sorry

end triangle_perimeter_l146_146388


namespace binom_9_5_eq_126_l146_146694

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146694


namespace trirectangular_tetrahedron_surface_area_l146_146837

variables (a b c : ℝ)
variables {V A B C : Type*}
variables (S : A → B → C → ℝ)

-- Given conditions
axiom right_triangle: a^2 + b^2 = c^2
axiom trirectangular_tetrahedron: ∀ (V A B C : Type*), ∠VAB = 90 ∧ ∠VBC = 90 ∧ ∠VCA = 90

-- The proof goal
theorem trirectangular_tetrahedron_surface_area :
  S A B C ^ 2 = S V A B ^ 2 + S V B C ^ 2 + S V C A ^ 2 :=
sorry

end trirectangular_tetrahedron_surface_area_l146_146837


namespace no_mem_is_veen_l146_146791
-- Import necessary libraries

-- Define the types Mem, En, and Veen
variables (Mem En Veen : Type)

-- Define the hypotheses
variables 
  (mem_to_en : Mem → En)                 -- Hypothesis I: All Mems are Ens
  (en_to_not_veens : En → ¬Veen)         -- Hypothesis II: No Ens are Veens

-- Define the statement that needs to be proven
theorem no_mem_is_veen : ∀ m : Mem, ¬ (∃ v : Veen, true) :=
by
  intro m
  apply en_to_not_veens (mem_to_en m)
  apply exists_true_of_false
  exact en_to_not_veens (mem_to_en m)

end no_mem_is_veen_l146_146791


namespace not_vshaped_f1_vshaped_f2_vshaped_f3_range_l146_146386

noncomputable def isVShaped (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f (x1 + x2) ≤ f x1 + f x2

noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := Real.log (x^2 + 2)
noncomputable def f3 (x a : ℝ) : ℝ := Real.log (2^x + a)

theorem not_vshaped_f1 : ¬ isVShaped f1 :=
sorry

theorem vshaped_f2 : isVShaped f2 :=
sorry

theorem vshaped_f3_range (a : ℝ) : isVShaped (λ x, f3 x a) ↔ (a ≥ 1 ∨ a = 0) :=
sorry

end not_vshaped_f1_vshaped_f2_vshaped_f3_range_l146_146386


namespace largest_multiple_of_15_under_500_l146_146991

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146991


namespace sequence_behavior_l146_146419

noncomputable def sequence (a : ℝ) (x : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 0 < a ∧ a < 1 → x 0 = a ∧ x (n+1) = a ^ (x n))

theorem sequence_behavior (a : ℝ) (x : ℕ → ℝ) (h : sequence a x) 
  (ha : 0 < a ∧ a < 1) :
  (∀ n : ℕ, (n % 2 = 0 → x n < x (n + 1)) ∧ (n % 2 = 1 → x n > x (n + 1))) :=
sorry

end sequence_behavior_l146_146419


namespace product_of_symmetric_complex_numbers_l146_146239

open Complex

-- Define the complex numbers and their properties
variables {z1 z2 : ℂ}

-- The given conditions
def symmetric_about_real_axis (z1 z2 : ℂ) : Prop :=
  z2 = conj z1

axiom z1_neq_1_plus_i : z1 ≠ 1 + I

-- The theorem to prove
theorem product_of_symmetric_complex_numbers (h_symm : symmetric_about_real_axis z1 z2) : z1 * z2 = 2 :=
by 
  sorry

end product_of_symmetric_complex_numbers_l146_146239


namespace inequality_of_geometric_means_l146_146757

theorem inequality_of_geometric_means 
  (a : ℕ → ℝ) (n : ℕ) (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) (h_n : n > 1) :
  let g_n := (∏ i in finset.range n, a (i + 1)) ^ (1 / n) in
  let A : ℕ → ℝ := λ k, (finset.range k).sum (λ i, a (i + 1)) / k in
  let G_n := (∏ k in finset.range n, A (k + 1)) ^ (1 / n) in
  n * ((G_n / A n) ^ (1 / n)) + (g_n / G_n) ≤ n + 1 :=
by 
  sorry

end inequality_of_geometric_means_l146_146757


namespace right_square_pyramid_dihedral_angle_l146_146398

theorem right_square_pyramid_dihedral_angle (a b c : ℕ) (θ : ℝ)  :
  ∠AOB = 30 ∧
  (∃ θ, dihedral_angle OAB OBC = θ ∧ cos θ = a * real.sqrt b - c) ∧
  (¬ ∃ p : ℕ, prime p ∧ p^2 ∣ b) →
  a + b + c = 14 :=
by
  intros,
  sorry

end right_square_pyramid_dihedral_angle_l146_146398


namespace largest_multiple_of_15_less_than_500_l146_146984

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146984


namespace determinant_simplifies_to_zero_l146_146646

theorem determinant_simplifies_to_zero (a b : ℝ) :
  matrix.det ![
    ![1, real.sin (a - b), real.sin a],
    ![real.sin (a - b), 1, real.sin b],
    ![real.sin a, real.sin b, 1]
  ] = 0 := 
by
  sorry

end determinant_simplifies_to_zero_l146_146646


namespace number_of_pairs_l146_146272

def equation_satisfied (m n : ℕ) : Prop :=
  m < n ∧ (3 * m * n = 2008 * (m + n))

def is_positive_integer (x : ℕ) : Prop := x > 0

theorem number_of_pairs : { p : ℕ × ℕ // equation_satisfied p.1 p.2 ∧ is_positive_integer p.1 ∧ is_positive_integer p.2 }.card = 4 :=
by
  -- Proof steps go here.
  sorry

end number_of_pairs_l146_146272


namespace simplify_and_evaluate_expression_l146_146464

variable (x y : ℝ)

theorem simplify_and_evaluate_expression
  (hx : x = 2)
  (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 :=
by
  sorry

end simplify_and_evaluate_expression_l146_146464


namespace find_abc_l146_146197

noncomputable def a_b_c_exist : Prop :=
  ∃ (a b c : ℝ), 
    (a + b + c = 21/4) ∧ 
    (1/a + 1/b + 1/c = 21/4) ∧ 
    (a * b * c = 1) ∧ 
    (a < b) ∧ (b < c) ∧ 
    (a = 1/4) ∧ (b = 1) ∧ (c = 4)

theorem find_abc : a_b_c_exist :=
sorry

end find_abc_l146_146197


namespace lowest_sale_price_is_30_percent_l146_146604

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end lowest_sale_price_is_30_percent_l146_146604


namespace largest_multiple_of_15_under_500_l146_146998

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l146_146998


namespace largest_multiple_of_15_less_than_500_l146_146983

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l146_146983


namespace lowest_possible_sale_price_is_30_percent_l146_146606

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end lowest_possible_sale_price_is_30_percent_l146_146606


namespace betty_garden_total_plants_l146_146631

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l146_146631


namespace arithmetic_sequence_second_term_l146_146831

theorem arithmetic_sequence_second_term (a1 a5 : ℝ) (h1 : a1 = 2020) (h5 : a5 = 4040) : 
  ∃ d a2 : ℝ, a2 = a1 + d ∧ d = (a5 - a1) / 4 ∧ a2 = 2525 :=
by
  sorry

end arithmetic_sequence_second_term_l146_146831


namespace monotonically_increasing_a_eq_1_l146_146259

def f (x : ℝ) (a : ℝ) : ℝ := exp(x) - (1/2) * a * x^2 - x

def f_derivative (x : ℝ) (a : ℝ) : ℝ := exp(x) - a * x - 1

theorem monotonically_increasing_a_eq_1 (a : ℝ) :
  (∀ x : ℝ, f_derivative x a ≥ 0) → a = 1 :=
by {
  sorry
}

end monotonically_increasing_a_eq_1_l146_146259


namespace power_addition_l146_146301

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l146_146301


namespace solve_equation_l146_146019

theorem solve_equation :
  ∀ (x : ℚ), x ≠ 1 → (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by
  intro x hx
  split
  { intro h
    sorry }
  { intro h
    rw [h]
    norm_num }

end solve_equation_l146_146019


namespace car_distance_problem_l146_146574

-- A definition for the initial conditions.
def initial_conditions (D : ℝ) (S : ℝ) (T : ℝ) : Prop :=
  T = 6 ∧ S = 50 ∧ (3/2 * T = 9)

-- The statement corresponding to the given problem.
theorem car_distance_problem (D : ℝ) (S : ℝ) (T : ℝ) :
  initial_conditions D S T → D = 450 :=
by
  -- leave the proof as an exercise.
  sorry

end car_distance_problem_l146_146574


namespace area_between_chords_l146_146394

theorem area_between_chords (R : ℝ) :
  let sector1 := (R ^ 2 * π) / 6 -- Area of sector for 60 degrees
  let triangle1 := (R ^ 2 * Real.sqrt 3) / 4 -- Area of triangle under the 60 degree chord
  let S1 := sector1 - triangle1 -- Area of segment formed by 60 degree arc
  let sector2 := (R ^ 2 * π) / 3 -- Area of sector for 120 degrees
  let triangle2 := (R ^ 2 * Real.sqrt 3) / 4 -- Area of triangle under the 120 degree chord
  let S2 := sector2 - triangle2 -- Area of segment formed by 120 degree arc
  let area_between := R ^ 2 * (π + Real.sqrt 3) / 2 -- Expected area between the chords
  (R ^ 2 * π - (S1 + S2)) = area_between :=
begin
  sorry
end

end area_between_chords_l146_146394


namespace calculate_expression_l146_146635

theorem calculate_expression :
  (-0.125) ^ 2009 * (8 : ℝ) ^ 2009 = -1 :=
sorry

end calculate_expression_l146_146635


namespace partition_exists_l146_146830

-- Define the graph (using typeclass for undirected graph).
class Graph (V : Type) :=
  (E : V → V → Prop)
  (symm : ∀ {u v : V}, E u v → E v u)

variables [Fintype V] [DecidableEq V]

-- Given the condition that we have at most 2015 unsociable groups.
axiom unsociable_bound (G : Graph V) : ∃ S : Finset (Finset V), S.card ≤ 2015 ∧ ∀ s ∈ S, odd s.card ∧ s.card ≥ 3 ∧ 
  (∀ v ∈ s, ∀ u ∈ s, v ≠ u → G.E v u)

-- Given G has chromatic number 11
axiom chromatic_number_11 (G : Graph V) : chromatic_number G = 11

-- Define the chromatic number for the partition
def chromatic_number (G : Graph V) : ℕ := 
  Inf {k : ℕ | ∃ (f : V → Fin k), ∀ {v w : V}, G.E v w → f v ≠ f w}

-- Main statement
theorem partition_exists (G : Graph V) 
  (h_unsociable : ∃ S : Finset (Finset V), S.card ≤ 2015 ∧ ∀ s ∈ S, odd s.card ∧ s.card ≥ 3 ∧ (∀ v ∈ s, ∀ u ∈ s, v ≠ u → G.E v u))
  (h_chromatic : chromatic_number G = 11) : 
  ∃ (f : V → Fin 11), ∀ {v w : V}, G.E v w → f v ≠ f w := 
sorry

end partition_exists_l146_146830


namespace increasing_on_interval_l146_146150

theorem increasing_on_interval (x : ℝ) : 
  (∀ x > 0, (f : ℝ → ℝ), f = (λ x, -1/x) → strict_mono_on f (set.Ioi 0))
  ∧ 
  (∀ x > 0, (f : ℝ → ℝ), f ≠ (λ x, -1/x) →
    (f = (λ x, x^2 - 2*x + 3) ∨ f = (λ x, (1/2)^x) ∨ f = (λ x, |x-1|)) →
    ¬ strict_mono_on f (set.Ioi 0)) :=
by
    sorry

end increasing_on_interval_l146_146150


namespace usb_drive_available_space_l146_146900

theorem usb_drive_available_space (C P : ℝ) (hC : C = 16) (hP : P = 50) : 
  (1 - P / 100) * C = 8 :=
by
  sorry

end usb_drive_available_space_l146_146900


namespace binom_9_5_eq_126_l146_146698

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l146_146698


namespace determinant_sin_eq_zero_l146_146655

theorem determinant_sin_eq_zero (a b : ℝ) : 
  matrix.det ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by
  sorry

end determinant_sin_eq_zero_l146_146655


namespace no_integers_abc_for_polynomial_divisible_by_9_l146_146453

theorem no_integers_abc_for_polynomial_divisible_by_9 :
  ¬ ∃ (a b c : ℤ), ∀ x : ℤ, 9 ∣ (x + a) * (x + b) * (x + c) - x ^ 3 - 1 :=
by
  sorry

end no_integers_abc_for_polynomial_divisible_by_9_l146_146453


namespace locus_of_A_is_segment_length_l146_146602

theorem locus_of_A_is_segment_length {A B C O : Point}
  (hABC : right_triangle A B C)
  (hAngleA : ∠ BAC = 90°)
  (hB_slides : B ∈ side1 ∧ B ∈ side2)
  (hC_slides: C ∈ side1 ∧ C ∈ side2)
  : length_of_segment A O = BC - AB := 
sorry

end locus_of_A_is_segment_length_l146_146602


namespace equivar_proof_l146_146758

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end equivar_proof_l146_146758


namespace number_relationship_l146_146048

theorem number_relationship : 0.7^6 < log 7 6 ∧ log 7 6 < 6^0.7 :=
by 
  sorry

end number_relationship_l146_146048


namespace hyperbola_eccentricity_l146_146765

theorem hyperbola_eccentricity (O F1 F2 A P M E N : ℝ × ℝ) (a b t : ℝ)
  (h_hyperbola : ∀ x y, (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_O_origin : O = (0, 0))
  (h_F1_focus : F1 = (-c, 0))
  (h_F2_focus : F2 = (c, 0))
  (h_A_vertex : A = (-a, 0))
  (h_P_on_hyperbola : ∃ y, P = (x, y) ∧ P ∈ C)
  (h_PF1_perpendicular : ∃ y, P = (-c, y) ∧ P ∈ C)
  (h_M_on_line : ∃ k, M = (-c, k))
  (h_E_on_yaxis : E = (0, (t * a) / (a - c)))
  (h_N_on_yaxis : N = (0, t / 2))
  (h_OE_eq_2ON : |OE| = 2 * |ON|)
  : eccentricity = 2 :=
begin
  sorry,
end

end hyperbola_eccentricity_l146_146765


namespace sphere_circular_cross_section_l146_146086

theorem sphere_circular_cross_section 
  (cylinder_cross_section : ∀ (h: Type), (cross_section_through_axis h = "cylinder" ↔ shape_of_cross_section h = "rectangular"))
  (cone_cross_section : ∀ (h: Type), (cross_section_through_axis h = "cone" ↔ shape_of_cross_section h = "triangular"))
  (frustum_cross_section : ∀ (h: Type), (cross_section_through_axis h = "frustum" ↔ shape_of_cross_section h = "isosceles trapezoidal")) :
  (cross_section_through_axis sphere = "circular") :=
by
  sorry

end sphere_circular_cross_section_l146_146086


namespace solve_problem_l146_146587

noncomputable def f (z : ℂ) : ℂ := -complex.I * conj z

theorem solve_problem :
  ∃ (z1 z2 : ℂ), |z1| = 7 ∧ f z1 = z1 ∧ |z2| = 7 ∧ f z2 = z2 ∧ z1 = 7 * complex.I ∧ z2 = -7 * complex.I := 
sorry

end solve_problem_l146_146587


namespace domain_of_f_l146_146930

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) + Real.log (x + 1)

theorem domain_of_f :
  {x : ℝ | 0 ≤ 2 - x ∧ 0 < x + 1} = set.Ioc (-1 : ℝ) 2 :=
by
  sorry

end domain_of_f_l146_146930


namespace MonotonicallyIncreasingInterval_l146_146240

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * sin x - cos x

noncomputable def g (x : ℝ) : ℝ := - Real.sqrt 2 * cos (3 * x)

lemma SymmetricCenter (a x : ℝ) (h : f (π / 4) a = 0) : a = 1 :=
by
  sorry

theorem MonotonicallyIncreasingInterval (k : ℤ) : 
    ∃ a : ℝ, f (π/4) a = 0 ∧ 
    ∀ x, (f x a = Real.sqrt 2 * sin (x - π / 4)) ∧ 
    g(x) = -Real.sqrt 2 * cos (3 * x) ∧ 
    (∃ x1 x2 : ℝ, (g x).increasing_on (set.Icc (2 * k * π / 3) (2 * k * π / 3 + π / 3))) :=
by
  sorry

end MonotonicallyIncreasingInterval_l146_146240


namespace stuffed_animal_tickets_correct_l146_146624

-- Define the total tickets spent
def total_tickets : ℕ := 14

-- Define the tickets spent on the hat
def hat_tickets : ℕ := 2

-- Define the tickets spent on the yoyo
def yoyo_tickets : ℕ := 2

-- Define the tickets spent on the stuffed animal
def stuffed_animal_tickets : ℕ := total_tickets - (hat_tickets + yoyo_tickets)

-- The theorem we want to prove.
theorem stuffed_animal_tickets_correct :
  stuffed_animal_tickets = 10 :=
by
  sorry

end stuffed_animal_tickets_correct_l146_146624


namespace optimal_play_first_player_wins_l146_146525

/-- Represents the game scenario where:
  * A sheet of graph paper measures 30 × 45 squares
  * Two players play in alternating turns
  * The first player starts cutting from the edge of the sheet
  * Each subsequent cut must continue the previous line of cuts
  * The player making a cut that causes the sheet to separate into two pieces wins the game
  * We need to prove that with optimal play, the first player will win -/
theorem optimal_play_first_player_wins :
  ∀ (n m : ℕ), n = 30 → m = 45 → 
  (∀ (move : ℕ → (ℕ × ℕ)),
  (move 0).fst = 0 ∨ (move 0).snd = 0) →
  (∀ k, k > 0 → 
    (move k).fst = (move (k-1)).fst ∨ (move k).snd = (move (k-1)).snd) →
  (∃ k, (∃ a, a.1 = (k-1) ∧ a.2 = k) ∨ (∃ b, b.1 = k ∧ b.2 = (k+1))) →
  first_player_wins :=
sorry

end optimal_play_first_player_wins_l146_146525


namespace vector_magnitude_l146_146227

variables (α β : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom α_norm : ∥α∥ = 1
axiom β_norm : ∥β∥ = 2
axiom orthogonal_cond : InnerProductSpace.inner α (α - 2 • β) = 0

-- Theorem to be proved
theorem vector_magnitude : ∥2 • α + β∥ = Real.sqrt 10 :=
sorry

end vector_magnitude_l146_146227
