import Mathlib
import Mathlib.Algebra.ContinuedFractions
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Quadratics
import Mathlib.Analysis.Calculus.ContinuousOn
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Vector
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib/Data/Matrix/Basic

namespace positive_difference_of_sums_l820_820292

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820292


namespace slope_angle_is_60_degrees_l820_820611

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (3, Real.sqrt 3)

-- Define the slope of the line passing through A and B    
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Define the function to compute the angle in degrees given the slope
def angle_from_slope (k : ℝ) : ℝ := Real.atan k * 180 / Real.pi

-- The theorem stating that the angle corresponding to the slope of line AB is 60 degrees
theorem slope_angle_is_60_degrees : angle_from_slope (slope A B) = 60 := by
  sorry

end slope_angle_is_60_degrees_l820_820611


namespace positive_difference_even_odd_sum_l820_820274

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820274


namespace f_has_two_zeros_iff_l820_820713

-- Conditions
def f (x m : ℝ) : ℝ := (1/x) - (m/x^2) - (x/3)

-- Proof statement:
theorem f_has_two_zeros_iff (m : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ m = 0 ∧ f x₂ m = 0) ↔ (0 < m ∧ m < 2 / 3) :=
by
  sorry

end f_has_two_zeros_iff_l820_820713


namespace no_such_polyhedron_l820_820554

theorem no_such_polyhedron (n : ℕ) (S : Fin n → ℝ) (H : ∀ i j : Fin n, i ≠ j → S i ≥ 2 * S j) : False :=
by
  sorry

end no_such_polyhedron_l820_820554


namespace positive_difference_even_odd_sums_l820_820251

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820251


namespace positive_difference_l820_820230

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820230


namespace positive_difference_even_odd_sums_l820_820255

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820255


namespace exams_left_to_grade_l820_820794

theorem exams_left_to_grade (E : ℕ) (p_m p_t : ℝ) (pm_nonneg : 0 ≤ p_m) (pm_le_one : p_m ≤ 1)
  (pt_nonneg : 0 ≤ p_t) (pt_le_one : p_t ≤ 1) (E_eq : E = 120) (pm_eq : p_m = 0.60) (pt_eq : p_t = 0.75) : 
  let graded_mon := p_m * E in
  let remaining_tue := E - graded_mon in
  let graded_tue := p_t * remaining_tue in
  let graded_mon_int := graded_mon.to_nat in
  let remaining_tue_int := remaining_tue.to_nat in
  let graded_tue_int := graded_tue.to_nat in
  let remaining_wed := remaining_tue_int - graded_tue_int in
  remaining_wed = 12 :=
by {
  sorry
}

end exams_left_to_grade_l820_820794


namespace speed_of_student_B_l820_820479

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820479


namespace student_b_speed_l820_820446

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820446


namespace volume_of_cuboid_l820_820307

-- Defining the original dimensions in meters
def width_cm : ℕ := 80
def length_cm : ℕ := 75
def height_cm : ℕ := 120

-- Function to convert centimeters to meters
def cm_to_m (cm : ℕ) : ℝ := cm / 100

-- Definition for dimensions doubled in meters
def doubled_width_m : ℝ := cm_to_m (2 * width_cm)
def doubled_length_m : ℝ := cm_to_m (2 * length_cm)
def doubled_height_m : ℝ := cm_to_m (2 * height_cm)

-- Definition for volume calculation
def volume (l w h : ℝ) : ℝ := l * w * h

-- The statement to prove
theorem volume_of_cuboid :
  volume doubled_length_m doubled_width_m doubled_height_m = 5.76 :=
by
  -- Add the proof here
  sorry

end volume_of_cuboid_l820_820307


namespace student_b_speed_l820_820364

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820364


namespace brother_pays_correct_amount_l820_820788

-- Definition of constants and variables
def friend_per_day := 5
def cousin_per_day := 4
def total_amount_collected := 119
def days := 7
def brother_per_day := 8

-- Statement of the theorem to be proven
theorem brother_pays_correct_amount :
  friend_per_day * days + cousin_per_day * days + brother_per_day * days = total_amount_collected :=
by {
  sorry
}

end brother_pays_correct_amount_l820_820788


namespace proof_p_q_e_f_l820_820053

theorem proof_p_q_e_f :
  ∃ (x y : ℕ) (p q e f : ℕ), 
  5 * x^7 = 13 * y^11 ∧ prime p ∧ prime q ∧ (x = p^e * q^f) ∧ (p + q + e + f = 31) :=
by
  -- Given x and y are positive integers with the equation 5x^7 = 13y^11
  -- Find the minimum value of x in terms of its prime factors p and q such that x = p^e * q^f
  -- Prove that the sum of the prime factors and their exponents is 31
  sorry

end proof_p_q_e_f_l820_820053


namespace log_base_a_of_3a_l820_820601

noncomputable theory
open Real

theorem log_base_a_of_3a (a : ℝ) (ha_pos : 0 < a) (ha_eq : a^a = (9 * a)^(8 * a)) :
  log a (3 * a) = 9 / 16 :=
sorry

end log_base_a_of_3a_l820_820601


namespace student_B_speed_l820_820511

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820511


namespace probability_P_plus_S_is_two_less_than_multiple_of_7_l820_820154

def is_distinct (a b : ℕ) : Prop :=
  a ≠ b

def in_range (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100

def mod_condition (a b : ℕ) : Prop :=
  (a * b + a + b) % 7 = 5

noncomputable def probability_p_s (p q : ℕ) : ℚ :=
  p / q

theorem probability_P_plus_S_is_two_less_than_multiple_of_7 :
  probability_p_s (1295) (4950) = 259 / 990 := 
sorry

end probability_P_plus_S_is_two_less_than_multiple_of_7_l820_820154


namespace max_binomial_term_l820_820540

theorem max_binomial_term (k : ℕ) :
  ∃ k : ℕ, k = 165 ∧
  ∀ m : ℕ, (m ≠ 165 → 
  (nat.choose 214 m * (real.sqrt 11) ^ m) < 
  (nat.choose 214 165 * (real.sqrt 11) ^ 165)) := 
sorry

end max_binomial_term_l820_820540


namespace positive_difference_of_sums_l820_820187

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820187


namespace positive_difference_even_odd_sums_l820_820258

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820258


namespace two_digit_number_swapped_condition_l820_820519

theorem two_digit_number_swapped_condition :
  (∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧
    let a := n / 10 in let b := n % 10 in
    10b + a = (7 * (10 * a + b)) / 4) →
  (finset.card {n ∈ finset.range 100 | 
    let a := n / 10 in let b := n % 10 in
    10 ≤ n ∧ 10b + a = 7 * (10 * a + b) / 4} = 4) :=
begin
  sorry
end

end two_digit_number_swapped_condition_l820_820519


namespace set_diff_P_Q_l820_820765

def P : Set ℝ := { x | Real.log x / Real.log 2 < 1 }
def Q : Set ℝ := { x | abs (x - 2) < 1 }

theorem set_diff_P_Q :
  P - Q = { x | 0 < x ∧ x ≤ 1 } := 
by
  sorry

end set_diff_P_Q_l820_820765


namespace speed_of_student_B_l820_820492

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820492


namespace positive_difference_even_odd_sums_l820_820303

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820303


namespace positive_difference_even_odd_sum_l820_820277

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820277


namespace positive_difference_eq_250_l820_820169

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820169


namespace sum_f_x₁_f_x₂_lt_0_l820_820965

variable (f : ℝ → ℝ)
variable (x₁ x₂ : ℝ)

-- Condition: y = f(x + 2) is an odd function
def odd_function_on_shifted_domain : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

-- Condition: f(x) is monotonically increasing for x > 2
def monotonically_increasing_for_x_gt_2 : Prop :=
  ∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Condition: x₁ + x₂ < 4
def sum_lt_4 : Prop :=
  x₁ + x₂ < 4

-- Condition: (x₁-2)(x₂-2) < 0
def product_shift_lt_0 : Prop :=
  (x₁ - 2) * (x₂ - 2) < 0

-- Main theorem to prove f(x₁) + f(x₂) < 0
theorem sum_f_x₁_f_x₂_lt_0
  (h1 : odd_function_on_shifted_domain f)
  (h2 : monotonically_increasing_for_x_gt_2 f)
  (h3 : sum_lt_4 x₁ x₂)
  (h4 : product_shift_lt_0 x₁ x₂) :
  f x₁ + f x₂ < 0 := sorry

end sum_f_x₁_f_x₂_lt_0_l820_820965


namespace inequality_proof_equality_condition_l820_820777

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) := 
by
  sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c :=
by
  sorry

end inequality_proof_equality_condition_l820_820777


namespace positive_difference_of_sums_l820_820291

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820291


namespace speed_of_student_B_l820_820476

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820476


namespace joy_fourth_rod_count_l820_820035

theorem joy_fourth_rod_count :
  let rods := (List.range' 1 40)
  let chosen := [4, 9, 18]
  let remaining := List.filter (λ x, x ∉ chosen) rods
  let valid_rods := List.filter (λ x, 6 ≤ x ∧ x ≤ 30) remaining
  List.length valid_rods = 22 :=
by
  let rods := (List.range' 1 40)
  let chosen := [4, 9, 18]
  let remaining := List.filter (λ x, x ∉ chosen) rods
  let valid_rods := List.filter (λ x, 6 ≤ x ∧ x ≤ 30) remaining
  have h : List.length valid_rods = 22 := sorry
  exact h

end joy_fourth_rod_count_l820_820035


namespace pool_cleaning_l820_820757

theorem pool_cleaning (full_capacity_liters : ℕ) (percent_full : ℕ) (loss_per_jump_ml : ℕ) 
    (full_capacity : full_capacity_liters = 2000) (trigger_clean : percent_full = 80) 
    (loss_per_jump : loss_per_jump_ml = 400) : 
    let trigger_capacity_liters := (full_capacity_liters * percent_full) / 100
    let splash_out_capacity_liters := full_capacity_liters - trigger_capacity_liters
    let splash_out_capacity_ml := splash_out_capacity_liters * 1000
    (splash_out_capacity_ml / loss_per_jump_ml) = 1000 :=
by {
    sorry
}

end pool_cleaning_l820_820757


namespace speed_of_student_B_l820_820493

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820493


namespace determine_machines_in_first_group_l820_820828

noncomputable def machines_in_first_group (x r : ℝ) : Prop :=
  (x * r * 6 = 1) ∧ (12 * r * 4 = 1)

theorem determine_machines_in_first_group (x r : ℝ) (h : machines_in_first_group x r) :
  x = 8 :=
by
  sorry

end determine_machines_in_first_group_l820_820828


namespace true_proposition_among_options_is_D_l820_820532

theorem true_proposition_among_options_is_D 
  (h_a1: ∃ a b, |a| > b ∧ ¬(a > b))
  (h_b: ∀ a b, |a| = |b| → ¬(a = b))
  (h_c: ∀ x, (x ≠ 2 → x^2 - 5*x + 6 ≠ 0)) :
  ∀ θ1 θ2, θ1 = θ2 → 
  (∀ f : ℝ → ℝ, f θ1 = f θ2) :=
by
  sorry

end true_proposition_among_options_is_D_l820_820532


namespace ordered_pair_exists_l820_820116

theorem ordered_pair_exists (s l : ℤ) :
  (∃ t : ℤ, ∃ x y : ℤ, (x, y) = ( -9 + t * l, s + t * (-6)) ∧ y = ( 1/3 : ℚ ) * x + 3) →
  (s = 0 ∧ l = -18) :=
begin
  sorry
end

end ordered_pair_exists_l820_820116


namespace speed_of_student_B_l820_820502

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820502


namespace acute_angle_probability_is_half_l820_820810

noncomputable def probability_acute_angle (h_mins : ℕ) (m_mins : ℕ) : ℝ :=
  if m_mins < 15 ∨ m_mins > 45 then 1 / 2 else 0

theorem acute_angle_probability_is_half (h_mins : ℕ) (m_mins : ℕ) :
  let P := probability_acute_angle h_mins m_mins in
  0 ≤ m_mins ∧ m_mins < 60 →
  P = 1 / 2 :=
sorry

end acute_angle_probability_is_half_l820_820810


namespace sum_of_exponents_outside_radical_l820_820087

theorem sum_of_exponents_outside_radical (a b c : ℕ) :
  let radical := (∛(40 * a ^ 5 * b ^ 8 * c ^ 14)) in
  radical = 2 * a * b ^ 2 * c ^ 4 * ∛(5 * a ^ 2 * b ^ 2 * c ^ 2) →
  1 + 2 + 4 = 7 :=
by
  intros
  exact sorry

end sum_of_exponents_outside_radical_l820_820087


namespace magnitude_of_b_l820_820687

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820687


namespace find_speed_of_B_l820_820349

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820349


namespace positive_diff_even_odd_sums_l820_820180

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820180


namespace theorem_8_4_3_l820_820704

variables (G : Type) (A B : set G)

-- Define the conditions
def no_true_wave {G : Type} (G : G) (A B : set G) : Prop :=
sorry -- Define what it means for G to not have a true A -> B wave

-- Define the desired outcome
def set_of_disjoint_A_B_paths_exists {G : Type} (G : G) (A B : set G) : Prop :=
sorry -- Define what it means for G to have a collection of disjoint A - B paths

theorem theorem_8_4_3 (h : no_true_wave G A B) : set_of_disjoint_A_B_paths_exists G A B :=
sorry

end theorem_8_4_3_l820_820704


namespace find_speed_of_B_l820_820359

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820359


namespace magnitude_b_sqrt_3_l820_820662

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820662


namespace sum_coefficients_of_parabola_l820_820983

noncomputable def parabola_vertex_form (p q r : ℝ) : Prop :=
∀ x y, y = px^2 + qx + r → (∃ a, y = a(x + 3)^2 + 4)

theorem sum_coefficients_of_parabola (p q r : ℝ) 
  (h_vertex : ∀ x y, y = px^2 + qx + r → (∃ a, y = a(x + 3)^2 + 4))
  (h_passes_origin : (0, 1) ∈ (λ x, px^2 + qx + r)) :
  p + q + r = -4/3 := by
sorry

end sum_coefficients_of_parabola_l820_820983


namespace probability_FG_half_l820_820078

-- Definitions
def points_consecutively_on_line_segment (A E F G B : ℝ) (line_segment : set ℝ) : Prop :=
  A < E ∧ E < F ∧ F < G ∧ G < B ∧ (∀ x, x ∈ line_segment ↔ A ≤ x ∧ x ≤ B)

def probability_between_FG (A E F G B : ℝ) : ℝ :=
  let AE := E - A in
  let AB := B - A in
  let FG := G - F in
  FG / AB

-- The conditions for our problem
variables (A E F G B : ℝ)
variable (line_segment : set ℝ)
hypothesis points_on_line : points_consecutively_on_line_segment A E F G B line_segment
hypothesis four_AE : B - A = 4 * (E - A)
hypothesis eight_BF : B - A = 8 * (B - F)

-- The goal to prove
theorem probability_FG_half :
  probability_between_FG A E F G B = 1 / 2 :=
  sorry

end probability_FG_half_l820_820078


namespace distinct_factorizations_72_l820_820952

-- Define the function D that calculates the number of distinct factorizations.
noncomputable def D (n : Nat) : Nat := 
  -- Placeholder function to represent D, the actual implementation is skipped.
  sorry

-- Theorem stating the number of distinct factorizations of 72 considering the order of factors.
theorem distinct_factorizations_72 : D 72 = 119 :=
  sorry

end distinct_factorizations_72_l820_820952


namespace student_b_speed_l820_820441

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820441


namespace value_of_f_at_112_5_l820_820060

noncomputable def f : ℝ → ℝ := sorry

lemma f_even_func (x : ℝ) : f x = f (-x) := sorry
lemma f_func_eq (x : ℝ) : f x + f (x + 1) = 4 := sorry
lemma f_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x + 12 := sorry

theorem value_of_f_at_112_5 : f 112.5 = 2 := sorry

end value_of_f_at_112_5_l820_820060


namespace minimum_handshakes_l820_820331

theorem minimum_handshakes (n : ℕ) (h : n = 30) :
  ∃ m, (∀ p ∈ finset.range n, (∃ q ∈ finset.range n, p ≠ q ∧ ⟦ shake(p, q) ⟧)) → m = 45 := 
by
  sorry

end minimum_handshakes_l820_820331


namespace positive_difference_l820_820231

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820231


namespace student_B_speed_l820_820454

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820454


namespace sqrt_cos_reduction_l820_820325

theorem sqrt_cos_reduction : sqrt (2 - 2 * cos 4) = 2 * sin 2 :=
by
  sorry

end sqrt_cos_reduction_l820_820325


namespace positive_difference_even_odd_l820_820198

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820198


namespace positive_difference_of_sums_l820_820192

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820192


namespace min_YZ_distance_l820_820732

theorem min_YZ_distance
  (X Y Z P Q : Type)
  (h_angle_XYZ : angle X Y Z = 50)
  (h_XY : dist X Y = 13)
  (h_XZ : dist X Z = 7)
  (h_P_on_XY : P ∈ open_segment X Y)
  (h_Q_on_XZ : Q ∈ open_segment X Z)
  : min_dist_YZ_PQZ = 18.92 := sorry

end min_YZ_distance_l820_820732


namespace tom_seashells_now_l820_820855

def original_seashells : ℕ := 5
def given_seashells : ℕ := 2

theorem tom_seashells_now : original_seashells - given_seashells = 3 :=
by
  sorry

end tom_seashells_now_l820_820855


namespace count_valid_8_digit_numbers_in_base_4_l820_820698

theorem count_valid_8_digit_numbers_in_base_4 : 
  ∃ n : ℕ, n = 2187 ∧ 
  (∀ (num : Fin 4 → ℕ), 
    (∀ i, num i ∈ {1, 2, 3}) ∧ (∀ d, ((∑ i, num i) % 3 = 0) → num = d)) :=
sorry

end count_valid_8_digit_numbers_in_base_4_l820_820698


namespace slope_angle_is_60_degrees_l820_820610

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (3, Real.sqrt 3)

-- Define the slope of the line passing through A and B    
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Define the function to compute the angle in degrees given the slope
def angle_from_slope (k : ℝ) : ℝ := Real.atan k * 180 / Real.pi

-- The theorem stating that the angle corresponding to the slope of line AB is 60 degrees
theorem slope_angle_is_60_degrees : angle_from_slope (slope A B) = 60 := by
  sorry

end slope_angle_is_60_degrees_l820_820610


namespace acute_angle_probability_l820_820804

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end acute_angle_probability_l820_820804


namespace student_B_speed_l820_820510

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820510


namespace find_speed_of_B_l820_820408

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820408


namespace max_whole_number_n_l820_820866

theorem max_whole_number_n (n : ℕ) : (1/2 + n/9 < 1) → n ≤ 4 :=
by
  sorry

end max_whole_number_n_l820_820866


namespace positive_difference_even_odd_sums_l820_820296

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820296


namespace patricia_money_l820_820954

/-!
# Problem: How much money does Patricia have, given the conditions.

Carmen needs $7 more to have twice the amount of money that Jethro has.
Patricia has 3 times as much money as Jethro.
The sum of all their money is $113.
-/

variables (J P : ℕ)

theorem patricia_money (h1 : 2 * J - 7 >= 0) (h2 : 7 < 2 * J) (hJ : J + (2 * J - 7) + 3 * J = 113) : P = 60 :=
by
  have h3 : P = 3 * J := sorry
  have h4 : 6 * J - 7 = 113 := by
    linarith [hJ]
  have h5 : 6 * J = 120 := by
    linarith [h4]
  have hJ_val : J = 20 := by
    exact eq_of_mul_eq_mul_right zero_lt_six h5
  have hPatricia : P = 3 * 20 := by
    rw [h3, hJ_val]
  exact congr_arg nat.succ (eq_of_mul_eq_mul_right zero_lt_three hPatricia)

lemma zero_lt_six : 6 ≠ 0 := by decide
lemma zero_lt_three : 3 ≠ 0 := by decide


end patricia_money_l820_820954


namespace find_speed_of_B_l820_820407

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820407


namespace positive_diff_even_odd_sums_l820_820177

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820177


namespace student_B_speed_l820_820469

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820469


namespace sum_of_possible_values_d_l820_820897

theorem sum_of_possible_values_d :
  let range_8 := (512, 4095)
  let digits_in_base_16 := 3
  (∀ n, n ∈ Set.Icc range_8.1 range_8.2 → (Nat.digits 16 n).length = digits_in_base_16)
  → digits_in_base_16 = 3 :=
by
  sorry

end sum_of_possible_values_d_l820_820897


namespace find_speed_of_B_l820_820356

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820356


namespace terminating_decimal_fraction_count_l820_820590

theorem terminating_decimal_fraction_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 399 ∧ (∃ k : ℕ, n = 21 * k)}.to_finset.card = 19 := by
  sorry

end terminating_decimal_fraction_count_l820_820590


namespace categorize_numbers_l820_820981

noncomputable def is_positive (x : ℚ) : Prop := 0 < x
noncomputable def is_negative (x : ℚ) : Prop := x < 0
noncomputable def is_integer (x : ℚ) : Prop := x.den = 1
noncomputable def is_fraction (x : ℚ) : Prop := ∃ n d : ℤ, d ≠ 0 ∧ x = n /. d

-- List of numbers given as rational approximations
def numbers : List ℚ :=
  [7, -3.14, -|(-5 : ℚ)|, 1 / 8, 0, -(1 * (3 / 2) : ℚ), 4.6, -(-3 / 4), -2]

theorem categorize_numbers :
  {x ∈ numbers | is_positive x} = {7, 1 / 8, 4.6, 3 / 4} ∧
  {x ∈ numbers | is_negative x} = {-3.14, -5, -(1 * (3 / 2)), -2} ∧
  {x ∈ numbers | is_integer x} = {7, -5, 0, -2} ∧
  {x ∈ numbers | is_fraction x} = {-3.14, 1 / 8, -(1 * (3 / 2)), 4.6, 3 / 4} :=
by
  sorry

end categorize_numbers_l820_820981


namespace golden_ratio_problem_l820_820742

theorem golden_ratio_problem
  (m n : ℝ) (sin cos : ℝ → ℝ)
  (h1 : m = 2 * sin (Real.pi / 10))
  (h2 : m ^ 2 + n = 4)
  (sin63 : sin (7 * Real.pi / 18) ≠ 0) :
  (m + Real.sqrt n) / (sin (7 * Real.pi / 18)) = 2 * Real.sqrt 2 := by
  sorry

end golden_ratio_problem_l820_820742


namespace positive_difference_even_odd_l820_820207

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820207


namespace drink_price_half_promotion_l820_820901

theorem drink_price_half_promotion (P : ℝ) (h : P + (1/2) * P = 13.5) : P = 9 := 
by
  sorry

end drink_price_half_promotion_l820_820901


namespace find_functions_l820_820567

noncomputable def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (p q r s : ℝ), p > 0 → q > 0 → r > 0 → s > 0 →
  (p * q = r * s) →
  (f p ^ 2 + f q ^ 2) / (f (r ^ 2) + f (s ^ 2)) = 
  (p ^ 2 + q ^ 2) / (r ^ 2 + s ^ 2)

theorem find_functions :
  ∀ (f : ℝ → ℝ),
  (satisfies_condition f) → 
  (∀ x : ℝ, x > 0 → f x = x ∨ f x = 1 / x) :=
by
  sorry

end find_functions_l820_820567


namespace positive_difference_l820_820227

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820227


namespace positive_diff_even_odd_sums_l820_820184

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820184


namespace positive_difference_sums_even_odd_l820_820242

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820242


namespace shaded_region_perimeter_l820_820737

theorem shaded_region_perimeter (O P Q : Point) (h1 : O.center_circle) 
  (h2 : dist O P = 8) (h3 : dist O Q = 8) (h4 : arc_PQ_frac := 5 / 6) : 
  perimeter_shaded_region O P Q = 16 + (40 / 3) * π := 
by 
  sorry

end shaded_region_perimeter_l820_820737


namespace orthocenters_collinear_l820_820055

theorem orthocenters_collinear (A B C M N L H_1 H_2 : Type) [TypeClass] 
  (h1: angle A C B = 90)
  (h2: M ∈ segment A C)
  (h3: N ∈ segment B C)
  (h4: L ∈ intersect lines (extend A N) (extend B M))
  (h5: H1 = orthocenter (triangle A M L))
  (h6: H2 = orthocenter (triangle B N L)) :
  collinear {H1, H2, C} :=
sorry

end orthocenters_collinear_l820_820055


namespace incorrect_value_in_sequence_l820_820854

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def first_differences (seq : List ℝ) : List ℝ :=
  seq.zipWith (· - ·) (List.tail seq)

def second_differences (seq : List ℝ) : List ℝ :=
  first_differences (first_differences seq)

theorem incorrect_value_in_sequence (a b c h : ℝ) (values : List ℝ)
  (h_length : values.length = 8) :
  values = [3844, 3969, 4096, 4227, 4356, 4489, 4624, 4761] →
  let diffs := second_differences values in
  ¬ diffs.all (λ d, d = (diffs.head)) :=
sorry

end incorrect_value_in_sequence_l820_820854


namespace jinho_total_distance_l820_820075

theorem jinho_total_distance (bus_distance_km : ℝ) (bus_distance_m : ℝ) (walk_distance_m : ℝ) :
  bus_distance_km = 4 → bus_distance_m = 436 → walk_distance_m = 1999 → 
  (2 * (bus_distance_km + bus_distance_m / 1000 + walk_distance_m / 1000)) = 12.87 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jinho_total_distance_l820_820075


namespace wheel_distance_per_rotation_l820_820521

theorem wheel_distance_per_rotation :
  ∀ (rotations_per_minute: ℕ) (distance_per_hour: ℕ)
  (minutes_per_hour: ℕ) (total_rotations: ℕ)
  (distance_per_rotation: ℕ),
  
  rotations_per_minute = 10 ∧
  distance_per_hour = 120000 ∧
  minutes_per_hour = 60 ∧
  total_rotations = rotations_per_minute * minutes_per_hour ∧
  distance_per_rotation = distance_per_hour / total_rotations →
  
  distance_per_rotation = 200 := 
begin
  sorry
end

end wheel_distance_per_rotation_l820_820521


namespace blending_marker_drawings_correct_l820_820850

-- Define the conditions
def total_drawings : ℕ := 25
def colored_pencil_drawings : ℕ := 14
def charcoal_drawings : ℕ := 4

-- Define the target proof statement
def blending_marker_drawings : ℕ := total_drawings - (colored_pencil_drawings + charcoal_drawings)

-- Proof goal
theorem blending_marker_drawings_correct : blending_marker_drawings = 7 := by
  sorry

end blending_marker_drawings_correct_l820_820850


namespace magnitude_b_eq_sqrt3_l820_820676

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820676


namespace positive_diff_even_odd_sums_l820_820176

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820176


namespace student_b_speed_l820_820436

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820436


namespace smallest_k_for_g_l820_820996

theorem smallest_k_for_g (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x + k = -3) ↔ k ≤ -3/4 := sorry

end smallest_k_for_g_l820_820996


namespace student_B_speed_l820_820457

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820457


namespace units_sold_to_A_l820_820517

-- Definitions relating to the conditions
def total_units (u : ℕ): Prop := u = 20
def defective_units (d : ℕ): Prop := d = 5
def units_sold_to_B (b : ℕ): Prop := b = 5
def units_sold_to_C (c : ℕ): Prop := c = 7
def all_non_defective_units_sold : Prop := true

-- Proof statement
theorem units_sold_to_A (A : ℕ) (u d b c non_defective_units : ℕ) :
  total_units u → 
  defective_units d → 
  units_sold_to_B b → 
  units_sold_to_C c → 
  all_non_defective_units_sold → 
  non_defective_units = u - d → 
  A = non_defective_units - (b + c) → 
  A = 3 :=
by
  intros hu hd hb hc hs hnd ha
  rw [hu, hd, hb, hc, hnd]
  sorry

end units_sold_to_A_l820_820517


namespace positive_difference_of_sums_l820_820282

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820282


namespace bicycle_speed_B_l820_820380

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820380


namespace positive_difference_even_odd_l820_820206

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820206


namespace jimmys_total_lodging_expense_l820_820556

-- Conditions
def hostel_nightly_cost : ℝ := 15
def hostel_service_tax_rate : ℝ := 0.10
def hostel_days : ℕ := 3

def cabin_total_cost_per_night : ℝ := 45
def cabin_tax_rate : ℝ := 0.07
def cabin_occupants : ℕ := 3

def discount_rate : ℝ := 0.15
def rounding (val: ℝ) : ℝ := Real.floor (val * 100 + 0.5) / 100

-- Total cost calculation for the hostel
def hostel_total_cost : ℝ := 
  let cost_with_tax := hostel_nightly_cost * (1 + hostel_service_tax_rate)
  cost_with_tax * hostel_days

-- Total cost calculation for the cabin without discount
def cabin_share_per_night : ℝ := 
  let cost_with_tax := cabin_total_cost_per_night * (1 + cabin_tax_rate)
  cost_with_tax / cabin_occupants

-- Total cost calculation for the cabin with discount on the fifth day
def cabin_discounted_cost_day_five : ℝ := 
  let base_share := cabin_total_cost_per_night / cabin_occupants
  let discounted_share := base_share * (1 - discount_rate)
  let discounted_share_with_tax := discounted_share * (1 + cabin_tax_rate)
  discounted_share_with_tax

-- Total cost calculation for all stays
def total_lodging_expense : ℝ := 
  hostel_total_cost + cabin_share_per_night + cabin_discounted_cost_day_five

theorem jimmys_total_lodging_expense :
  rounding total_lodging_expense = 79.19 := by sorry

end jimmys_total_lodging_expense_l820_820556


namespace distinct_positive_xyz_l820_820091

theorem distinct_positive_xyz (a b : ℝ) :
  (∃ (x y z : ℝ), x + y + z = a ∧ x^2 + y^2 + z^2 = b^2 ∧ x * y = z^2 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x > 0 ∧ y > 0 ∧ z > 0) ↔
  (|b| < a ∧ a < √(3) * |b|) :=
by 
  sorry

end distinct_positive_xyz_l820_820091


namespace triangle_relation_l820_820731

theorem triangle_relation (A B C a b : ℝ) (h : 4 * A = B ∧ B = C) (hABC : A + B + C = 180) : 
  a^3 + b^3 = 3 * a * b^2 := 
by 
  sorry

end triangle_relation_l820_820731


namespace surface_area_of_pyramid_DABC_l820_820768

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

def is_isosceles_13_13_24 (a b c : ℝ) := (a = 13 ∧ b = 13 ∧ c = 24) ∨ (a = 13 ∧ c = 13 ∧ b = 24) ∨ (b = 13 ∧ a = 13 ∧ c = 24) ∨ (b = 13 ∧ c = 13 ∧ a = 24) ∨ (c = 13 ∧ a = 13 ∧ b = 24) ∨ (c = 13 ∧ b = 13 ∧ a = 24)

def is_isosceles_13_24_24 (a b c : ℝ) := (a = 13 ∧ b = 24 ∧ c = 24) ∨ (a = 24 ∧ b = 13 ∧ c = 24) ∨ (a = 24 ∧ b = 24 ∧ c = 13)

def surface_area_DABC (triangles : Fin 4 → (ℝ × ℝ × ℝ)) : ℝ :=
  ∑ i, triangle_area (triangles i).1 (triangles i).2 (triangles i).3

theorem surface_area_of_pyramid_DABC : 
  ∀ triangles : Fin 4 → (ℝ × ℝ × ℝ), 
  (∀ i, is_isosceles_13_13_24 (triangles i).1 (triangles i).2 (triangles i).3 ∨ is_isosceles_13_24_24 (triangles i).1 (triangles i).2 (triangles i).3) → 
  (¬ ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ triangles i = triangles j ∧ triangles j = triangles k) →
  surface_area_DABC triangles = 510 := 
by
  intros triangles h1 h2
  sorry

end surface_area_of_pyramid_DABC_l820_820768


namespace positive_difference_even_odd_sums_l820_820304

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820304


namespace measured_percentage_weight_loss_l820_820315

variable (W : ℝ) -- W is the starting weight.
variable (weight_loss_percent : ℝ := 0.12) -- 12% weight loss.
variable (clothes_weight_percent : ℝ := 0.03) -- 3% clothes weight addition.
variable (beverage_weight_percent : ℝ := 0.005) -- 0.5% beverage weight addition.

theorem measured_percentage_weight_loss : 
  (W - ((0.88 * W) + (clothes_weight_percent * W) + (beverage_weight_percent * W))) / W * 100 = 8.5 :=
by
  sorry

end measured_percentage_weight_loss_l820_820315


namespace samantha_sleep_hours_l820_820084

def time_in_hours (hours minutes : ℕ) : ℕ :=
  hours + (minutes / 60)

def hours_slept (bed_time wake_up_time : ℕ) : ℕ :=
  if bed_time < wake_up_time then wake_up_time - bed_time + 12 else 24 - bed_time + wake_up_time

theorem samantha_sleep_hours : hours_slept 7 11 = 16 := by
  sorry

end samantha_sleep_hours_l820_820084


namespace positive_diff_even_odd_sums_l820_820183

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820183


namespace positive_difference_even_odd_sums_l820_820260

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820260


namespace cost_equivalence_l820_820948

theorem cost_equivalence (b a p : ℕ) (h1 : 4 * b = 3 * a) (h2 : 9 * a = 6 * p) : 24 * b = 12 * p :=
  sorry

end cost_equivalence_l820_820948


namespace norm_b_eq_sqrt_3_l820_820674

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820674


namespace sum_of_coordinates_of_B_l820_820814

def point := (ℝ × ℝ)

noncomputable def point_A : point := (0, 0)

def line_y_equals_6 (B : point) : Prop := B.snd = 6

def slope_AB (A B : point) (m : ℝ) : Prop := (B.snd - A.snd) / (B.fst - A.fst) = m

theorem sum_of_coordinates_of_B (B : point) 
  (h1 : B.snd = 6)
  (h2 : slope_AB point_A B (3/5)) :
  B.fst + B.snd = 16 :=
sorry

end sum_of_coordinates_of_B_l820_820814


namespace find_b_in_geometric_sequence_l820_820019

theorem find_b_in_geometric_sequence (a_1 : ℤ) :
  ∀ (n : ℕ), ∃ (b : ℤ), (3^n - b = (a_1 * (3^n - 1)) / 2) :=
by
  sorry

example (a_1 : ℤ) :
  ∃ (b : ℤ), ∀ (n : ℕ), 3^n - b = (a_1 * (3^n - 1)) / 2 :=
by
  use 1
  sorry

end find_b_in_geometric_sequence_l820_820019


namespace pool_cleaning_l820_820756

theorem pool_cleaning (full_capacity_liters : ℕ) (percent_full : ℕ) (loss_per_jump_ml : ℕ) 
    (full_capacity : full_capacity_liters = 2000) (trigger_clean : percent_full = 80) 
    (loss_per_jump : loss_per_jump_ml = 400) : 
    let trigger_capacity_liters := (full_capacity_liters * percent_full) / 100
    let splash_out_capacity_liters := full_capacity_liters - trigger_capacity_liters
    let splash_out_capacity_ml := splash_out_capacity_liters * 1000
    (splash_out_capacity_ml / loss_per_jump_ml) = 1000 :=
by {
    sorry
}

end pool_cleaning_l820_820756


namespace arithmetic_sequence_formula_arithmetic_sequence_sum_arithmetic_sequence_sum_minimum_l820_820766

theorem arithmetic_sequence_formula (n : ℕ) (a S : ℕ → ℤ) (h₁ : a 1 = -7) (h₂ : S 3 = -15) :
  (a n = 2 * n - 9) := 
sorry

theorem arithmetic_sequence_sum (n : ℕ) (a S : ℕ → ℤ) (h₁ : a 1 = -7) (h₂ : S 3 = -15)
  (h₃ : ∀ n, a n = 2 * n - 9) :
  (S n = (n - 4) ^ 2 - 16) :=
sorry

theorem arithmetic_sequence_sum_minimum (n : ℕ) (a S : ℕ → ℤ) (h₁ : a 1 = -7) (h₂ : S 3 = -15)
  (h₃ : ∀ n, a n = 2 * n - 9) :
  (∀ n, S 4 = -16) :=
sorry

end arithmetic_sequence_formula_arithmetic_sequence_sum_arithmetic_sequence_sum_minimum_l820_820766


namespace positive_difference_sums_even_odd_l820_820233

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820233


namespace positive_difference_even_odd_l820_820202

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820202


namespace find_speed_of_B_l820_820417

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820417


namespace area_triangle_BQW_l820_820018

-- Definitions of the given conditions
def is_rectangle (ABCD : Type) : Prop :=
  ∃ (A B C D : Point), parallelogram ABCD ∧
  ∀ (A B C D : Point), right_angle∠(B D A) ∧ right_angle∠(C D B)

-- Given conditions
variable {A B C D Z W Q : Point}
variable {length_ab length_az length_wc area_trap height_AD : ℝ}

-- Assume given conditions
axiom ABCD_rectangle : is_rectangle ABCD
axiom AZ_10_units : length AZ = 10
axiom WC_10_units : length WC = 10
axiom AB_12_units : length AB = 12
axiom area_trap_200_units : area ZWCD = 200
axiom height_AD : height AD = (50 / 3) + 20

-- The statement to be proved
theorem area_triangle_BQW : area BQW = 40 :=
sorry

end area_triangle_BQW_l820_820018


namespace gcd_36_60_l820_820571

theorem gcd_36_60 : Int.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l820_820571


namespace TV_height_l820_820525

theorem TV_height (Area Width Height : ℝ) (h_area : Area = 21) (h_width : Width = 3) (h_area_def : Area = Width * Height) : Height = 7 := 
by
  sorry

end TV_height_l820_820525


namespace speed_of_student_B_l820_820485

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820485


namespace find_m_find_theta_find_k_l820_820642

-- Given vectors
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Perpendicular condition
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Parallel condition
def parallel (u v : ℝ × ℝ) : Prop := ∃ (k : ℝ), u = (k * v.1, k * v.2)

-- Problem statements
theorem find_m : ∀ (m : ℝ), perpendicular a (a.1 - b(m).1, a.2 - b(m).2) → m = -4 :=
by
  sorry

theorem find_theta : ∀ (m : ℝ), (m = -4) → ∀ (θ : ℝ), a ≠ (0, 0) ∧ b(m) ≠ (0, 0) ∧
  cos θ = (a.1 * b(m).1 + a.2 * b(m).2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b(m).1^2 + b(m).2^2)) 
  ∧ 0 ≤ θ ∧ θ ≤ real.pi → θ = real.pi / 4 :=
by
  sorry

theorem find_k : ∀ (m : ℝ), (m = -4) → ∀ (k : ℝ), parallel (k * a.1 + b(m).1, k * a.2 + b(m).2) (a.1 - b(m).1, a.2 - b(m).2) → k = -1 :=
by
  sorry

end find_m_find_theta_find_k_l820_820642


namespace student_B_speed_l820_820472

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820472


namespace crow_eating_nuts_l820_820317

theorem crow_eating_nuts (T N : ℕ) (h : N > 0) :
  (T / 5) = 8 → T / 4 = 10 :=
by
  intro h1
  have h2 : T / 5 * 5 = T, from nat.div_mul_cancel h1,
  rw ←h2,
  have h3 : 8 * 5 = T, from nat.mul_eq_of_eq_div h1,
  sorry

end crow_eating_nuts_l820_820317


namespace find_speed_of_B_l820_820360

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820360


namespace problem_statement_l820_820851

variable {Ball : Type}
variable (Box1 Box2 Box3 : Finset Ball)
variable (labels : Ball → ℕ)

-- Definitions for Balls in boxes according to the conditions
def box1 : Finset Ball := {b | b ∈ Box1 ∧ (labels b = 1 ∨ labels b = 2 ∨ labels b = 3)} 
def box2 : Finset Ball := {b | b ∈ Box2 ∧ (labels b = 1 ∨ labels b = 3)}
def box3 : Finset Ball := {b | b ∈ Box3 ∧ (labels b = 1 ∨ labels b = 2)}

-- Defining the events A_i and B_i for Box and label i where box addition occurs
def Ai (i : ℕ) : Prop := ∃ b ∈ box1, labels b = i
def Bi (i : ℕ) : Prop := 
  (∃ b ∈ Box1, labels b = 1 → ∃ b' ∈ Box2, labels b' = i) ∨
  (∃ b ∈ Box1, labels b = 2 → ∃ b' ∈ Box3, labels b' = i) ∨
  (∃ b ∈ Box1, labels b = 3 → ∃ b' ∈ Box3, labels b' = i)

-- Statements to verify:
def incorrect_statement : Prop := ¬(P(Bi 3) = 13/48)

theorem problem_statement : incorrect_statement := sorry

end problem_statement_l820_820851


namespace student_B_speed_l820_820503

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820503


namespace speed_of_student_B_l820_820475

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820475


namespace positive_difference_l820_820221

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820221


namespace largest_triangle_angle_l820_820136

-- Define the angles
def angle_sum := (105 : ℝ) -- Degrees
def delta_angle := (36 : ℝ) -- Degrees
def total_sum := (180 : ℝ) -- Degrees

-- Theorem statement
theorem largest_triangle_angle (a b c : ℝ) (h1 : a + b = angle_sum)
  (h2 : b = a + delta_angle) (h3 : a + b + c = total_sum) : c = 75 :=
sorry

end largest_triangle_angle_l820_820136


namespace gumballs_difference_l820_820883

theorem gumballs_difference :
  ∃ (x_min x_max : ℕ), 
    19 ≤ (16 + 12 + x_min) / 3 ∧ (16 + 12 + x_min) / 3 ≤ 25 ∧
    19 ≤ (16 + 12 + x_max) / 3 ∧ (16 + 12 + x_max) / 3 ≤ 25 ∧
    (x_max - x_min = 18) :=
by
  sorry

end gumballs_difference_l820_820883


namespace square_of_cube_of_smallest_prime_l820_820306

def smallest_prime : Nat := 2

theorem square_of_cube_of_smallest_prime :
  ((smallest_prime ^ 3) ^ 2) = 64 := by
  sorry

end square_of_cube_of_smallest_prime_l820_820306


namespace Sn_correct_l820_820743

noncomputable def P_n (n : ℕ) : ℝ × ℝ :=
  (n+1, 2 / (n+1))

noncomputable def b_n (n : ℕ) : ℝ :=
  let (x1, y1) := P_n n
  let (x2, y2) := P_n (n+1)
  let m := (y2 - y1) / (x2 - x1)
  let c := y1 - m * x1
  let b := y1 + m * (-x1)
  let a := x2
  1 / 2 * a * b

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b_n i

theorem Sn_correct (n : ℕ) : S_n n = 4 * n + n / (n + 1) := sorry

end Sn_correct_l820_820743


namespace grasshopper_min_flights_l820_820906

theorem grasshopper_min_flights (square : Fin 10 × Fin 10) 
    (start : square = (0, 0)) 
    (moves : ∀ (current : square), (current.fst + 1 < 10 → square) ∨ (current.snd + 1 < 10 → square)) 
    (flights : ∀ (current : square), (current.fst = 9 → square.fst = 0) ∨ (current.snd = 9 → square.snd = 0)) :
    ∃ n ≥ 9, ∀ cell ∈ square, grasshopper visits cell at least once :=
sorry

end grasshopper_min_flights_l820_820906


namespace compute_b_c_sum_l820_820047

def polynomial_decomposition (Q : ℝ[X]) (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ) : Prop :=
  ∀ x : ℝ, Q.eval x = (x^2 + (b1*x) + c1) * (x^2 + (b2*x) + c2) * (x^2 + (b3*x) + c3) * (x^2 + (b4*x) + c4)

theorem compute_b_c_sum (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : polynomial_decomposition (polynomial.mk [1, -1, 1, -1, 1, -1, 1, -1, 1]) b1 b2 b3 b4 c1 c2 c3 c4)
  (c1_eq : c1 = 1) (c2_eq : c2 = 1) (c3_eq : c3 = 1) (c4_eq : c4 = 1) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 := 
sorry

end compute_b_c_sum_l820_820047


namespace positive_difference_sums_even_odd_l820_820238

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820238


namespace probability_pair_tile_l820_820973

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end probability_pair_tile_l820_820973


namespace highest_power_of_two_dividing_sum_of_coeffs_l820_820126

-- Define the polynomial sequence according to the given conditions
def p : ℕ → polynomial ℚ 
| 1 := polynomial.X ^ 2 - 1
| 2 := 2 * polynomial.X ^ 3 - 2 * polynomial.X
| (n+1) := λ x, 2 * x * p n x - p (n-1) x

-- Define the statement about the highest power of 2 dividing the sum of the absolute values of the coefficients
theorem highest_power_of_two_dividing_sum_of_coeffs (n : ℕ) (k : ℕ) (m : ℕ) (h : n = 2^k * (2*m + 1)) :
  ∃ k : ℕ, ∃ (m : ℕ), n = 2^k * (2*m + 1) ∧
  highest_power_of_two_dividing (polynomial.sum_of_coeffs_abs (p n)) = 2^(k+1) :=
sorry

end highest_power_of_two_dividing_sum_of_coeffs_l820_820126


namespace line_intersects_segment_l820_820995

-- Definitions for the problem
def point (ℝ : Type) := (ℝ × ℝ)

def A : point ℝ := (3,2)
def B : point ℝ := (2,3)

def slope (p1 p2 : point ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Conditions
def line_segment_intersect (k : ℝ) (p1 p2 : point ℝ) : Prop :=
  let slope1 := slope (1, 0) p1
  let slope2 := slope (1, 0) p2
  k >= 1 ∧ k <= slope2

-- Proof problem
theorem line_intersects_segment (k : ℝ) : 
  line_segment_intersect k A B ↔ (1 ≤ k ∧ k ≤ 3) :=
sorry

end line_intersects_segment_l820_820995


namespace magnitude_b_eq_sqrt3_l820_820680

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820680


namespace rectangular_eq_circle_max_area_triangle_PAB_l820_820020

-- Definition for polar coordinate equation of circle
def polar_eq_circle_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * sqrt 2 * sin (θ - π / 4)

-- Parametric equation of line
def parametric_eq_line (x y t : ℝ) : Prop :=
  x = - sqrt 2 / 3 * t ∧ y = -1 + sqrt 2 / 4 * t

-- Given conditions about intersection and points
def line_intersects_circle (A B P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  parametric_eq_line A.1 A.2 (some tA) ∧ parametric_eq_line B.1 B.2 (some tB) ∧
  polar_eq_circle_C (sqrt (A.1 ^ 2 + A.2 ^ 2)) (atan2 A.2 A.1) ∧
  polar_eq_circle_C (sqrt (B.1 ^ 2 + B.2 ^ 2)) (atan2 B.2 B.1) ∧ 
  A ≠ B ∧ (sqrt (P.1 ^ 2 + P.2 ^ 2) = sqrt 2 ∧ polar_eq_circle_C (sqrt (P.1 ^ 2 + P.2 ^ 2)) (atan2 P.2 P.1)) ∧ P ≠ A ∧ P ≠ B

-- Proof problem 1: Rectangular coordinate equation of the circle
theorem rectangular_eq_circle (C : ℝ × ℝ) (ρ θ : ℝ) :
  (polar_eq_circle_C ρ θ) → ((ρ * cos θ + 1) ^ 2 + (ρ * sin θ - 1) ^ 2 = 2) :=
sorry

-- Proof problem 2: Maximum area of triangle PAB
theorem max_area_triangle_PAB (A B P : ℝ × ℝ) (C : ℝ × ℝ) :
  (line_intersects_circle A B P C) → max_area_of_triangle A B P = 1 + sqrt 2 :=
sorry

end rectangular_eq_circle_max_area_triangle_PAB_l820_820020


namespace polar_to_cartesian_l820_820637

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the goal as a theorem
theorem polar_to_cartesian : ∀ (x y : ℝ), 
  (∃ θ : ℝ, polar_eq (Real.sqrt (x^2 + y^2)) θ ∧ x = (Real.sqrt (x^2 + y^2)) * Real.cos θ 
  ∧ y = (Real.sqrt (x^2 + y^2)) * Real.sin θ) → (x-1)^2 + y^2 = 1 :=
by
  intro x y
  intro h
  sorry

end polar_to_cartesian_l820_820637


namespace find_y_value_l820_820584

theorem find_y_value (y : ℝ) (h : 1 / (3 + 1 / (3 + 1 / (3 - y))) = 0.30337078651685395) : y = 0.3 :=
sorry

end find_y_value_l820_820584


namespace find_T_n_l820_820924

def a_sequence (n : ℕ) : ℕ := 
if n = 1 then 2 else
if n = 2 then 3 else
a_sequence (n-1) + (a_sequence (n-2) - 1)

def S (n : ℕ) : ℕ := 
∑ i in finset.range (n+1), a_sequence i

lemma arithmetic_sequence (n: ℕ) (hn1: 2 ≤ n):
  S (n + 1) + S (n - 1) = 2 * S n + 1 := 
sorry

def b_sequence (n : ℕ) : ℕ := 2^n * a_sequence n

def T (n : ℕ) : ℕ := 
∑ i in finset.range (n+1), b_sequence i

theorem find_T_n (n : ℕ) : 
T n = (n + 2) * 2^(n + 1) - 4 :=
sorry

end find_T_n_l820_820924


namespace greatest_N_no_substring_multiple_of_9_l820_820323

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def no_substring_multiple_of_9 (N : ℕ) : Prop :=
  ∀ (m : ℕ), m ∈ N.digits 10.attach_substring → sum_of_digits m % 9 ≠ 0

theorem greatest_N_no_substring_multiple_of_9 : 
  ∃ N, N = 88888888 ∧ no_substring_multiple_of_9 N ∧
    ∀ M, no_substring_multiple_of_9 M → M ≤ 88888888 :=
  sorry

end greatest_N_no_substring_multiple_of_9_l820_820323


namespace magnitude_of_b_l820_820688

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820688


namespace symmetric_points_power_equal_one_l820_820710

-- Definition of symmetry condition about the x-axis
def symmetric_about_x_axis (A B : ℝ × ℝ): Prop := A.1 = -B.1 ∧ A.2 = -B.2

-- The proof problem statement
theorem symmetric_points_power_equal_one :
  ∀ (m n : ℝ), symmetric_about_x_axis (m, 5) (-6, -n) → (m + n) ^ 2012 = 1 :=
by {
  intro m n,
  intro h,
  have h1 : m = -6 := by { cases h, exact h_left },
  have h2 : n = 5 := by { cases h, exact h_right },
  rw [h1, h2],
  norm_num,
  sorry,
}

end symmetric_points_power_equal_one_l820_820710


namespace speed_of_student_B_l820_820483

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820483


namespace interval_of_increase_l820_820633

open Real

def f (x : ℝ) : ℝ := log (-x ^ 2 - 2 * x + 3)

theorem interval_of_increase : 
  {x : ℝ | -3 < x ∧ x < -1} ⊆ {x : ℝ | deriv f x > 0} :=
by sorry

end interval_of_increase_l820_820633


namespace rods_quadrilateral_count_l820_820034

theorem rods_quadrilateral_count :
  let rods := list.range' 1 40, -- list of rods from 1 to 40
  let used_rods := [4, 9, 18], -- rods already used
  let remaining_rods := rods.filter (λ d => d ∉ used_rods),
  let valid_rods := remaining_rods.filter (λ d => 6 ≤ d ∧ d ≤ 30) in
  valid_rods.length = 22 :=
by
  let rods := list.range' 1 40
  let used_rods := [4, 9, 18]
  let remaining_rods := rods.filter (λ d => d ∉ used_rods)
  let valid_rods := remaining_rods.filter (λ d => 6 ≤ d ∧ d ≤ 30)
  sorry

end rods_quadrilateral_count_l820_820034


namespace positive_difference_even_odd_sums_l820_820256

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820256


namespace systematic_sampling_example_l820_820820

theorem systematic_sampling_example : 
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, 5 ≤ i ∧ i ≤ 5 → a i = 5 + 10 * (i - 1)) ∧ 
  ∀ i : ℕ, 1 ≤ i ∧ i < 6 → a i - a (i - 1) = a (i + 1) - a i :=
sorry

end systematic_sampling_example_l820_820820


namespace radius_sum_greater_l820_820316

variables (A B C D : Type) [InnerProductSpace ℝ A]
variables (triangle : A → A → A → Prop)
variables (r r₁ r₂ p p₁ p₂ S S₁ S₂ : ℝ)

-- Conditions:
variable (H1 : triangle A B C)        -- ABC is a triangle
variable (H2 : triangle A B D)        -- ABD is a triangle
variable (H3 : triangle D B C)        -- DBC is a triangle
variable (H4 : S = p * r)             -- Area formula for triangle ABC
variable (H5 : S₁ = p₁ * r₁)          -- Area formula for triangle ABD
variable (H6 : S₂ = p₂ * r₂)          -- Area formula for triangle DBC
variable (H7 : S = S₁ + S₂)           -- Area relation
variable (H8 : p = p₁ + p₂)           -- Semiperimeter relation

-- Theorem:
theorem radius_sum_greater : r < r₁ + r₂ :=
by {
    -- No proof provided
    sorry
}

end radius_sum_greater_l820_820316


namespace norm_b_eq_sqrt_3_l820_820670

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820670


namespace abs_difference_of_angles_l820_820623

theorem abs_difference_of_angles (a b : ℝ) (h1 : cos (2 * a) = 2 / 3) :
  |a - b| = sqrt 5 / 5 := by
  sorry

end abs_difference_of_angles_l820_820623


namespace square_area_l820_820016

theorem square_area (x : ℝ) (A B C D E F : ℝ)
  (h1 : E = x / 3)
  (h2 : F = (2 * x) / 3)
  (h3 : abs (B - E) = 40)
  (h4 : abs (E - F) = 40)
  (h5 : abs (F - D) = 40) :
  x^2 = 2880 :=
by
  -- Main proof here
  sorry

end square_area_l820_820016


namespace speed_of_student_B_l820_820419

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820419


namespace christian_charge_per_yard_l820_820544

-- Definitions of given data
def total_cost : ℝ := 50.0
def saved_christian : ℝ := 5.0
def saved_sue : ℝ := 7.0
def additional_needed : ℝ := 6.0
def yards_mowed : ℕ := 4
def dogs_walked : ℕ := 6
def charge_per_dog : ℝ := 2.0

-- The amount needed overall minus what they still need to make
def total_after_chores : ℝ := total_cost - additional_needed
-- The amount saved initially by both
def initial_savings : ℝ := saved_christian + saved_sue
-- The amount earned from chores in total
def total_chore_earnings : ℝ := total_after_chores - initial_savings
-- The amount Sue earned from walking dogs
def sue_earnings : ℝ := dogs_walked * charge_per_dog
-- The amount Christian earned from mowing yards
def christian_earnings : ℝ := total_chore_earnings - sue_earnings
-- Calculating how much Christian charged per yard
def charge_per_yard : ℝ := christian_earnings / yards_mowed

theorem christian_charge_per_yard : charge_per_yard = 5.0 := by
  sorry

end christian_charge_per_yard_l820_820544


namespace ellipse_equation_range_of_k_l820_820938

-- Given conditions
def ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def vertex_A_in_ellipse (a b : ℝ) := (-2)^2 / b^2 = 1

def area_of_quadrilateral := 4 * sqrt 5

def point_P := (0, -3)

def intersection_points_same_sign (a b k : ℝ) := 
  let d := (-30 * k)^2 - 4 * 25 * (4 + 5 * k^2) in
  d > 0

-- The equation of the ellipse
theorem ellipse_equation : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ 
  vertex_A_in_ellipse a b ∧
  (1 / 2 * 2 * a * 2 * b = area_of_quadrilateral) →
  ellipse a b :=
sorry

-- Range of values for k
theorem range_of_k :
  ∀ k : ℝ, let M := 1 in
  intersection_points_same_sign sqrt 5 2 k →
  |k| > 1 →
  (|PM| + |PN| ≤ 15) ↔
  k ∈ [-3, -1) ∪ (1, 3] :=
sorry


end ellipse_equation_range_of_k_l820_820938


namespace no_sum_of_perimeters_exceeds_1993_l820_820520

-- Defining the conditions of the problem
variable (squares : List ℝ) -- list of side lengths of the smaller squares
variable (unit_square : set (ℝ × ℝ)) -- unit square

-- Main diagonal of the unit square from (0,0) to (1,1)
def main_diagonal (p : ℝ × ℝ) : Prop := p.1 = p.2

-- Predicate to check if a square intersects the main diagonal
def intersects_main_diagonal (a : ℝ) : Prop :=
  ∃ p ∈ unit_square, main_diagonal p ∧ p ∈ a -- assuming a represents the square region

-- Sum of perimeters of squares intersecting the diagonal
def sum_perimeters (squares : List ℝ) : ℝ :=
  4 * (squares.filter intersects_main_diagonal).sum

-- Proof statement, asserting the sum of the perimeters cannot exceed 1993
theorem no_sum_of_perimeters_exceeds_1993 :
  sum_perimeters squares ≤ 4 :=
sorry

end no_sum_of_perimeters_exceeds_1993_l820_820520


namespace pq_composite_l820_820038

-- Define the problem conditions
def has_positive_integer_roots (p q : ℤ) : Prop :=
∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = -p ∧ a * b = q + 1

-- Main theorem statement
theorem pq_composite (p q : ℤ) (h : has_positive_integer_roots p q) : ∃ (m n : ℕ), 1 < m ∧ 1 < n ∧ m * n = (p^2 + q^2) :=
begin
  sorry
end

end pq_composite_l820_820038


namespace positive_difference_even_odd_sum_l820_820275

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820275


namespace base_number_in_exponent_l820_820305

theorem base_number_in_exponent (x : ℝ) (k : ℕ) (h₁ : k = 8) (h₂ : 64^k > x^22) : 
  x = 2^(24/11) :=
sorry

end base_number_in_exponent_l820_820305


namespace maximize_sum_product_l820_820122

theorem maximize_sum_product (a b c d : ℕ) (h1 : {a, b, c, d} = {2, 3, 4, 5}) :
  (a + b) * (c + d) ≤ 49 :=
by
  -- We have the multiset equality h1, but let's derive a stronger set equality just for clarity
  have h2 : {a, b, c, d} = {2, 3, 4, 5},
  sorry

end maximize_sum_product_l820_820122


namespace probability_six_is_265_over_720_l820_820146

-- Define derangements
def derangements (n : ℕ) : ℕ :=
(n.factorial * ∑ k in Finset.range (n+1), (-1:ℝ) ^ k / k.factorial).toReal.toNat

-- Define the total number of permutations
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Define the probability that no one receives the correct letter
def probability_no_correct_letter (n : ℕ) : ℝ :=
derangements n / total_permutations n

-- Prove the probability in the case of n = 6
theorem probability_six_is_265_over_720 : probability_no_correct_letter 6 = 265 / 720 := by
  sorry

end probability_six_is_265_over_720_l820_820146


namespace probability_tile_in_PAIR_l820_820978

theorem probability_tile_in_PAIR :
  let tiles := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
  let pair_letters := ['P', 'A', 'I', 'R']
  let matching_counts := (sum ([1, 1, 2, 1] : List ℕ))
  matching_counts.fst = 5
  let total_tiles := 12
  (matching_counts.toRational / total_tiles.toRational) = (5 / 12) :=
by sorry

end probability_tile_in_PAIR_l820_820978


namespace positive_difference_even_odd_sum_l820_820270

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820270


namespace range_of_x_plus_y_l820_820613

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : log (x + y) = log x + log y) :
  4 ≤ x + y :=
by
  sorry

end range_of_x_plus_y_l820_820613


namespace fraction_students_above_eight_l820_820735

theorem fraction_students_above_eight (total_students S₈ : ℕ) (below_eight_percent : ℝ)
    (num_below_eight : total_students * below_eight_percent = 10) 
    (total_equals : total_students = 50) 
    (students_eight : S₈ = 24) :
    (total_students - (total_students * below_eight_percent + S₈)) / S₈ = 2 / 3 := 
by 
  -- Solution steps can go here 
  sorry

end fraction_students_above_eight_l820_820735


namespace circumcircle_radii_equal_l820_820151

noncomputable def circle (center : Point) (radius : ℝ) : Set Point := sorry

def is_tangent (P : Point) (C1 C2 : Set Point) : Prop := sorry
def is_reflection (P Q M : Point) : Prop := sorry
def is_circumcircle (P Q R : Point) (C : Set Point) : Prop := sorry

variables {O1 O2 A B M N C D E F : Point}
variables {c1 c2 : ℝ}

-- Conditions
axiom circles_intersect : M ∈ (circle O1 c1) ∧ M ∈ (circle O2 c2) ∧ N ∈ (circle O1 c1) ∧ N ∈ (circle O2 c2)
axiom common_tangent : is_tangent A (circle O1 c1) (circle O2 c2) ∧ is_tangent B (circle O1 c1) (circle O2 c2)
axiom reflections : is_reflection C A M ∧ is_reflection D B M
axiom circumcircle_DCM : is_circumcircle D C M (circle O1 c1) ∧ is_circumcircle D C M (circle O2 c2)
axiom distinct_from_M : E ≠ M ∧ F ≠ M

-- Theorem statement
theorem circumcircle_radii_equal (hE : E ∈ circle O1 c1) (hF : F ∈ circle O2 c2) :
  let MEF_circ := circle_center_radius E F M,
      NEF_circ := circle_center_radius E F N in
  circle_RADIUS MEF_circ = circle_RADIUS NEF_circ := sorry

end circumcircle_radii_equal_l820_820151


namespace find_y_for_line_slope_45_degrees_l820_820117

theorem find_y_for_line_slope_45_degrees :
  ∃ y, (∃ x₁ y₁ x₂ y₂, x₁ = 4 ∧ y₁ = y ∧ x₂ = 2 ∧ y₂ = -3 ∧ (y₂ - y₁) / (x₂ - x₁) = 1) → y = -1 :=
by
  sorry

end find_y_for_line_slope_45_degrees_l820_820117


namespace student_B_speed_l820_820515

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820515


namespace perimeter_of_ABCDEFG_l820_820149

theorem perimeter_of_ABCDEFG :
  ∀ (A B C D E F G : Point) (AB AC AD AE AG : ℝ),
  equilateral_triangle A B C → equilateral_triangle A D E → equilateral_triangle E F G →
  midpoint D A C → midpoint G A E → midpoint F E G → 
  distance A B = 6 → 
  perimeter A B C D E F G = 25.5 :=
by
  -- We skip the proof itself for now.
  sorry

end perimeter_of_ABCDEFG_l820_820149


namespace right_triangle_hypotenuse_l820_820013

theorem right_triangle_hypotenuse :
  ∃ b a : ℕ, a^2 + 1994^2 = b^2 ∧ b = 994010 :=
by
  sorry

end right_triangle_hypotenuse_l820_820013


namespace perpendicular_vector_lambda_l820_820691

theorem perpendicular_vector_lambda:
  let a := (1, 2)
  let b := (1, -1)
  let c := (4, 5)
  ∃ λ: ℝ, (a.1 * (b.1 + λ * c.1) + a.2 * (b.2 + λ * c.2) = 0) -> 
  λ = 1 / 14 := 
by
  sorry

end perpendicular_vector_lambda_l820_820691


namespace expansion_properties_l820_820627

-- Define the context of expansion and ratio conditions
noncomputable def ratio_condition : ℕ → Prop :=
  λ n, ((Binomial.binom n 4) * (-2)^4 / (Binomial.binom n 2 * (-2)^2) = 56 / 3)

-- Define the rational terms for expansion of given function when n = 10
def rational_terms_expansion (n : ℕ) : Prop :=
  n = 10 ∧
  (∃ (c₀ c₄ : ℝ), c₀ = 1 ∧ c₄ = 13440)

-- Define the value calculation for given n = 10
def value_calculation : Prop :=
  (10 + 9 * Binomial.binom 10 2 + 81 * Binomial.binom 10 3 + ... + 9^9 * Binomial.binom 10 10 = 
  (10^10 - 1) / 9)

-- The final theorem statement
theorem expansion_properties :
  ratio_condition 10 → rational_terms_expansion 10 ∧ value_calculation :=
by
  sorry

end expansion_properties_l820_820627


namespace day_crew_fraction_loaded_l820_820882

-- Definitions and terms used in the problem
variables (D W : ℕ) -- D: boxes per day worker, W: number of day workers

-- Assumptions based on the problem conditions
def night_boxes_per_worker := D / 4
def night_workers := (4 / 5 : ℤ) * W

-- Total boxes loaded by day crew and night crew
def day_total_boxes := D * W
def night_total_boxes := night_boxes_per_worker * night_workers

-- Total number of boxes loaded by both crews
def total_boxes := day_total_boxes + night_total_boxes

-- Fraction of the boxes loaded by the day crew
def fraction_day_loaded := day_total_boxes / total_boxes

theorem day_crew_fraction_loaded :
  fraction_day_loaded = 5 / 6 :=
sorry -- skipping the proof

end day_crew_fraction_loaded_l820_820882


namespace opposite_of_fraction_l820_820838

theorem opposite_of_fraction : - (11 / 2022 : ℚ) = -(11 / 2022) := 
by
  sorry

end opposite_of_fraction_l820_820838


namespace fraction_solution_l820_820895

-- Define the given fraction equation
def fraction_equation (x : ℝ) : Prop :=
  (3 / 4) * (1 / 2) * x * 5100 = 765.0000000000001

-- Prove the fraction equation solves to 0.4
theorem fraction_solution : ∃ x : ℝ, fraction_equation x ∧ x = 0.4 :=
by
  use 0.4
  have h : fraction_equation 0.4 := by
    unfold fraction_equation
    norm_num
  exact ⟨h, rfl⟩

end fraction_solution_l820_820895


namespace bicycle_speed_B_l820_820385

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820385


namespace small_gate_width_l820_820539

-- Bob's garden dimensions
def garden_length : ℝ := 225
def garden_width : ℝ := 125

-- Total fencing needed, including the gates
def total_fencing : ℝ := 687

-- Width of the large gate
def large_gate_width : ℝ := 10

-- Perimeter of the garden without gates
def garden_perimeter : ℝ := 2 * (garden_length + garden_width)

-- Width of the small gate
theorem small_gate_width :
  2 * (garden_length + garden_width) + small_gate + large_gate_width = total_fencing → small_gate = 3 :=
by
  sorry

end small_gate_width_l820_820539


namespace total_purchase_cost_l820_820524

variable (kg_nuts : ℝ) (kg_dried_fruits : ℝ)
variable (cost_per_kg_nuts : ℝ) (cost_per_kg_dried_fruits : ℝ)

-- Define the quantities
def cost_nuts := kg_nuts * cost_per_kg_nuts
def cost_dried_fruits := kg_dried_fruits * cost_per_kg_dried_fruits

-- The total cost can be expressed as follows
def total_cost := cost_nuts + cost_dried_fruits

theorem total_purchase_cost (h1 : kg_nuts = 3) (h2 : kg_dried_fruits = 2.5)
  (h3 : cost_per_kg_nuts = 12) (h4 : cost_per_kg_dried_fruits = 8) :
  total_cost kg_nuts kg_dried_fruits cost_per_kg_nuts cost_per_kg_dried_fruits = 56 := by
  sorry

end total_purchase_cost_l820_820524


namespace do_perp_ci_of_isosceles_triangle_l820_820604

noncomputable def Triangle := {X : Type u} [metric_space X] (a b c: X)

variables {X : Type*} [metric_space X] (A B C : X)

/-- Definition of given triangle ABC being isosceles, with AB = BC --/
def is_isosceles_triangle (A B C : X) : Prop := dist A B = dist B C

/-- Definition of the circumcenter O of triangle ABC --/
def circumcenter (A B C : X) : X := sorry -- Omitted actual definition

/-- Definition of the incenter I of triangle ABC --/
def incenter (A B C : X) : X := sorry -- Omitted actual definition   

/-- D lies on BC such that DI is parallel to AB --/
def D_on_BC_and_DI_parallel_AB (D I A B C : X) : Prop :=
  sorry -- Omitted actual definition
  -- For example, there should be definitions checking point D on BC and DI parallel to AB

/-- Lines DO and CI are perpendicular --/
def DO_perp_CI (D O C I : X) : Prop :=
  dist D O + dist O C + dist C I = π / 2 -- Assumes a metric space definition of perpendicularity

/-- Main Theorem Statement: Given isosceles Triangle ABC, circumcenter O,
    incenter I, and D on BC such that DI ∥ AB, lines DO and CI are ⊥ --/
theorem do_perp_ci_of_isosceles_triangle 
  (A B C D O I : X)
  (h_isosceles : is_isosceles_triangle A B C)
  (h_circumcenter : circumcenter A B C = O)
  (h_incenter : incenter A B C = I)
  (h_D_condition : D_on_BC_and_DI_parallel_AB D I A B C) 
  : DO_perp_CI D O C I := 
sorry

end do_perp_ci_of_isosceles_triangle_l820_820604


namespace no_such_polynomial_exists_l820_820968

theorem no_such_polynomial_exists :
  ∀ (P : ℤ → ℤ), (∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4) → false :=
by
  sorry

end no_such_polynomial_exists_l820_820968


namespace bicycle_speed_B_l820_820389

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820389


namespace tangent_parallel_to_x_axis_l820_820712

theorem tangent_parallel_to_x_axis (k : ℝ) (h_tangent: ∃ (x y : ℝ), (x = 1 ∧ y = k ∧ y = k * x + log x) ∧ (derivative (λ x : ℝ, k * x + log x) x = 0)) : k = -1 :=
sorry

end tangent_parallel_to_x_axis_l820_820712


namespace incircle_radius_incircle_radius_l820_820150

-- Conditions
def is_right_angle_triangle (D E F : Type) (has_angle_F: Prop) (angleD_45: Prop) (DF: ℝ) :=
  has_angle_F ∧ angleD_45 ∧ DF = 6

-- **Our target is to prove the inradius r of the triangle DEF**
theorem incircle_radius (D E F : Type) [right_angle_triangle : is_right_angle_triangle D E F] (DF : ℝ) :
  r = 6 - 3 * Real.sqrt 2 := by
  sorry

-- Instantiate the conditions as hypotheses
namespace TriangleDEF
variable {D E F : Type}
variable [has_angle_F : right_angle_triangle D E F]

theorem incircle_radius (r : ℝ) :
  is_right_angle_triangle D E F has_angle_F angleD_45 DF →
  r = 6 - 3 * Real.sqrt 2 := by
  sorry
end TriangleDEF

end incircle_radius_incircle_radius_l820_820150


namespace triangle_area_range_l820_820615

theorem triangle_area_range (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : x₁ < 1) (h₂ : 1 < x₂) (h₃ : x₁ * x₂ = 1) :
  0 < (2 / (x₁ + 1 / x₁)) ∧ (2 / (x₁ + 1 / x₁)) < 1 :=
by
  sorry

end triangle_area_range_l820_820615


namespace annalise_boxes_purchase_l820_820942

theorem annalise_boxes_purchase:
  ∀ (boxes packs tissues_per_pack boxes_cost tissues_cost total_spent : ℕ),
  packs = 20 →
  tissues_per_pack = 100 →
  tissues_cost = 5 →
  total_spent = 1000 →
  boxes_cost = tissues_cost * tissues_per_pack * packs →
  (total_spent = boxes_cost * boxes) →
  boxes = 10 :=
by
  intros boxes packs tissues_per_pack boxes_cost tissues_cost total_spent
  assume h_packs h_tissues_per_pack h_tissues_cost h_total_spent h_boxes_cost h_total_boxes_cost
  sorry

end annalise_boxes_purchase_l820_820942


namespace speed_of_student_B_l820_820429

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820429


namespace find_speed_of_B_l820_820353

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820353


namespace positive_difference_sums_even_odd_l820_820236

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820236


namespace positive_diff_even_odd_sums_l820_820181

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820181


namespace fixed_point_of_log_function_l820_820113

theorem fixed_point_of_log_function (a : ℝ) (h : 1 < a) : ∃ (P : ℝ × ℝ), P = (4, 2) ∧ (P.snd = log a (P.fst - 3) + 2) :=
begin
  use (4, 2),
  split,
  {
    refl,
  },
  {
    change 2 = log a (4 - 3) + 2,
    simp [log_base_one],
  },
end

end fixed_point_of_log_function_l820_820113


namespace find_line_equation_proj_origin_l820_820127

theorem find_line_equation_proj_origin (P : ℝ × ℝ) (hP : P = (-2, 1)) :
    ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 5 := 
by
  sorry

end find_line_equation_proj_origin_l820_820127


namespace norm_b_eq_sqrt_3_l820_820671

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820671


namespace charlyn_viewable_area_rounded_l820_820955

noncomputable def charlyn_viewable_area (side_length visibility_radius : ℝ) : ℝ :=
  let inside_area := side_length ^ 2
  let outside_area_rectangles := 4 * side_length * (visibility_radius - side_length / 2)
  let outside_area_circles := 4 * (π * (visibility_radius ^ 2)) / 4
  inside_area + outside_area_rectangles + outside_area_circles

theorem charlyn_viewable_area_rounded (side_length visibility_radius : ℝ) (h_side_length : side_length = 5) (h_visibility_radius : visibility_radius = 2) :
  Real.toInt (charlyn_viewable_area side_length visibility_radius) = 58 :=
by
  sorry

end charlyn_viewable_area_rounded_l820_820955


namespace probability_pair_tile_l820_820974

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end probability_pair_tile_l820_820974


namespace total_oranges_l820_820848

theorem total_oranges (a b c : ℕ) 
  (h₁ : a = 22) 
  (h₂ : b = a + 17) 
  (h₃ : c = b - 11) : 
  a + b + c = 89 := 
by
  sorry

end total_oranges_l820_820848


namespace clock_angle_acute_probability_l820_820807

-- Given the condition that a clock stops randomly at any moment,
-- and defining the probability of forming an acute angle between the hour and minute hands,
-- prove that this probability is 1/2.

theorem clock_angle_acute_probability : 
  (probability (\theta : ℝ, is_acute ⟨θ % 360, 0 ≤ θ % 360 < 360⟩) = 1/2) :=
-- Definitions and conditions.
sorry

end clock_angle_acute_probability_l820_820807


namespace probability_correct_l820_820738

noncomputable def probability_B1_eq_5_given_WB : ℚ :=
  let P_B1_eq_5 : ℚ := 1 / 8
  let P_WB : ℚ := 1 / 5
  let P_WB_given_B1_eq_5 : ℚ := 1 / 16 + 369 / 2048
  (P_B1_eq_5 * P_WB_given_B1_eq_5) / P_WB

theorem probability_correct :
  probability_B1_eq_5_given_WB = 115 / 1024 :=
by
  sorry

end probability_correct_l820_820738


namespace positive_difference_even_odd_sums_l820_820293

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820293


namespace length_XG_in_triangle_XYZ_l820_820752

noncomputable def median_length (a b c : ℝ) : ℝ := real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

noncomputable def centroid_length (a b c : ℝ) : ℝ := (2 / 3) * median_length a b c

theorem length_XG_in_triangle_XYZ :
  let XY := 30,
      XZ := 29,
      YZ := 31 in
  centroid_length YZ XZ XY = 16.7 :=
by
  sorry -- The actual proof would be here

end length_XG_in_triangle_XYZ_l820_820752


namespace analytical_expression_of_f_monotonic_intervals_l820_820696

-- Given conditions
def m (x : ℝ) : ℝ × ℝ := ⟨-2 * sin (π - x), cos x⟩
def n (x : ℝ) : ℝ × ℝ := ⟨sqrt 3 * cos x, 2 * sin (π / 2 - x)⟩
def f (x : ℝ) : ℝ := 1 - (m x).1 * (n x).1 - (m x).2 * (n x).2

-- Questions
theorem analytical_expression_of_f :
  f(x) = sqrt 3 * sin (2 * x) - cos(2 * x) :=
sorry

theorem monotonic_intervals (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) :
  (0 ≤ x ∧ x ≤ π / 3) ∨ (5 * π / 6 ≤ x ∧ x ≤ π) → ∀ x1 x2, x1 < x2 → f(x1) ≤ f(x2) :=
sorry

end analytical_expression_of_f_monotonic_intervals_l820_820696


namespace clock_angle_acute_probability_l820_820809

-- Given the condition that a clock stops randomly at any moment,
-- and defining the probability of forming an acute angle between the hour and minute hands,
-- prove that this probability is 1/2.

theorem clock_angle_acute_probability : 
  (probability (\theta : ℝ, is_acute ⟨θ % 360, 0 ≤ θ % 360 < 360⟩) = 1/2) :=
-- Definitions and conditions.
sorry

end clock_angle_acute_probability_l820_820809


namespace clock_angle_acute_probability_l820_820801

noncomputable def probability_acute_angle : ℚ := 1 / 2

theorem clock_angle_acute_probability :
  ∀ (hour minute : ℕ), (hour >= 0 ∧ hour < 12) →
  (minute >= 0 ∧ minute < 60) →
  (let angle := min (60 * hour - 11 * minute) (720 - (60 * hour - 11 * minute)) in angle < 90 ↔ probability_acute_angle = 1 / 2) :=
sorry

end clock_angle_acute_probability_l820_820801


namespace positive_difference_even_odd_sum_l820_820271

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820271


namespace smallest_n_divisible_by_100_and_has_100_divisors_exists_l820_820579

/-- Definition of having exactly 100 divisors -/
def has_exactly_100_divisors (n : ℕ) := (n.factors + 1).prod = 100

/-- Definition of divisibility by 100 -/
def divisible_by_100 (n : ℕ) := n % 100 = 0

/-- Main theorem -/
theorem smallest_n_divisible_by_100_and_has_100_divisors_exists :
  ∃ n : ℕ, divisible_by_100 n ∧ has_exactly_100_divisors n ∧ n = 162000 := 
sorry

end smallest_n_divisible_by_100_and_has_100_divisors_exists_l820_820579


namespace sqrt_sum_of_roots_l820_820822

theorem sqrt_sum_of_roots :
  (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30).sqrt
  = (Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3) :=
by
  sorry

end sqrt_sum_of_roots_l820_820822


namespace norm_b_eq_sqrt_3_l820_820673

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820673


namespace student_b_speed_l820_820372

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820372


namespace find_triples_l820_820985

theorem find_triples (a b c : ℝ) 
  (h1 : a = (b + c) ^ 2) 
  (h2 : b = (a + c) ^ 2) 
  (h3 : c = (a + b) ^ 2) : 
  (a = 0 ∧ b = 0 ∧ c = 0) 
  ∨ 
  (a = 1/4 ∧ b = 1/4 ∧ c = 1/4) :=
  sorry

end find_triples_l820_820985


namespace find_gross_income_deceased_member_l820_820102

-- Definitions of gross incomes of four members
variables (A B C D : ℝ)

-- Tax rates for each member
def taxRateA := 0.10
def taxRateB := 0.15
def taxRateC := 0.20
def taxRateD := 0.25

-- Net incomes after tax
def netA := A * (1 - taxRateA)
def netB := B * (1 - taxRateB)
def netC := C * (1 - taxRateC)
def netD := D * (1 - taxRateD)

-- Total net income of the family when all members were earning
noncomputable def totalIncome := 4 * 782

-- Total net income of the family after one member died
noncomputable def remainingIncome := 3 * 650

theorem find_gross_income_deceased_member :
  netA + netB + netC + netD = totalIncome →
  netA + netB + netC = remainingIncome →
  D = 1178 / 0.75 :=
begin
  intros,
  sorry
end

end find_gross_income_deceased_member_l820_820102


namespace positive_difference_even_odd_sums_l820_820295

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820295


namespace smallest_sum_of_pairwise_distinct_squares_l820_820778

theorem smallest_sum_of_pairwise_distinct_squares :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = z^2 ∧ c + a = y^2 ∧ a + b + c = 55 :=
sorry

end smallest_sum_of_pairwise_distinct_squares_l820_820778


namespace sequence_a_2024_l820_820844

theorem sequence_a_2024 (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 1 - 1 / a n) : a 2024 = 1 / 2 :=
by
  sorry

end sequence_a_2024_l820_820844


namespace norm_b_eq_sqrt_3_l820_820669

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820669


namespace solution_exists_l820_820827

noncomputable def solveSystem (x y z : ℝ) : Prop :=
  (7 / (2 * x - 3) - 2 / (10 * z - 3 * y) + 3 / (3 * y - 8 * z) = 8) ∧
  (2 / (2 * x - 3 * y) - 3 / (10 * z - 3 * y) + 1 / (3 * y - 8 * z) = 0) ∧
  (5 / (2 * x - 3 * y) - 4 / (10 * z - 3 * y) + 7 / (3 * y - 8 * z) = 8)

theorem solution_exists : ∃ (x y z : ℝ), solveSystem x y z ∧ x = 5 ∧ y = 3 ∧ z = 1 :=
by
  use 5
  use 3
  use 1
  split
  all_goals {
    -- First equation
    show 7 / (2 * 5 - 3) - 2 / (10 * 1 - 3 * 3) + 3 / (3 * 3 - 8 * 1) = 8,
    sorry,
    -- Second equation
    show 2 / (2 * 5 - 3 * 3) - 3 / (10 * 1 - 3 * 3) + 1 / (3 * 3 - 8 * 1) = 0,
    sorry,
    -- Third equation
    show 5 / (2 * 5 - 3 * 3) - 4 / (10 * 1 - 3 * 3) + 7 / (3 * 3 - 8 * 1) = 8,
    sorry
  },
  -- Showing x = 5, y = 3, z = 1
  split
  all_goals { refl }

end solution_exists_l820_820827


namespace inequality_solution_l820_820090

theorem inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 5) →
  ((x * x - 4 * x - 5) / (x * x + 3 * x + 2) < 0 ↔ (x ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∨ x ∈ Set.Ioo (-1:ℝ) (5:ℝ))) :=
by
  sorry

end inequality_solution_l820_820090


namespace train_pass_time_l820_820881

def train_length : ℝ := 360
def bridge_length : ℝ := 160
def speed_kph : ℝ := 45
def speed_mps : ℝ := speed_kph * (1000 / 3600)
def total_distance : ℝ := train_length + bridge_length

theorem train_pass_time :
  (total_distance / speed_mps) = 41.6 :=
by
  sorry

end train_pass_time_l820_820881


namespace find_triples_l820_820569

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x^2 + y^2 = 3 * 2016^z + 77) :
  (x = 4 ∧ y = 8 ∧ z = 0) ∨ (x = 8 ∧ y = 4 ∧ z = 0) ∨
  (x = 14 ∧ y = 77 ∧ z = 1) ∨ (x = 77 ∧ y = 14 ∧ z = 1) ∨
  (x = 35 ∧ y = 70 ∧ z = 1) ∨ (x = 70 ∧ y = 35 ∧ z = 1) :=
sorry

end find_triples_l820_820569


namespace prove_true_proposition_l820_820042

variables (a b c : ℝ^3)
variables (p q : Prop)

def p := (a.dot b = 0 ∧ b.dot c = 0) → (a.dot c = 0)
def q := (a.parallel b ∧ b.parallel c) → a.parallel c

theorem prove_true_proposition :
  (p → False) ∨ q :=
by
  -- Proof would go here
  sorry

end prove_true_proposition_l820_820042


namespace positive_difference_eq_250_l820_820172

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820172


namespace norm_b_eq_sqrt_3_l820_820672

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820672


namespace positive_difference_even_odd_sums_l820_820294

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820294


namespace sqrt8_same_type_as_sqrt2_l820_820871

theorem sqrt8_same_type_as_sqrt2 :
  (∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 8) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 4) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 6) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 10) :=
by
  sorry

end sqrt8_same_type_as_sqrt2_l820_820871


namespace positive_difference_sums_even_odd_l820_820240

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820240


namespace product_of_undefined_x_l820_820994

theorem product_of_undefined_x {x : ℝ} (h : x^2 + 3 * x - 4 = 0) : x = -4 ∨ x = 1 → (-4) * 1 = -4 :=
by
  intro h1
  cases h1 with h2 h3
  {
    rw h2
    ring
  }
  {
    rw h3
    ring
  }

end product_of_undefined_x_l820_820994


namespace number_of_different_arrangements_l820_820008

-- Define the total number of students
def total_students := 6

-- Define the condition that any two adjacent students must be of different genders
def alternating_genders (arrangement : list (string × string)) : Prop :=
  ∀ (i : ℕ), (i < arrangement.length - 1) → (arrangement[i].snd ≠ arrangement[i+1].snd)

-- Define the condition that A and B must be adjacent
def a_and_b_adjacent (arrangement : list (string × string)) : Prop :=
  ∃ (i : ℕ), (arrangement[i] = ("A", "male") ∧ arrangement[i+1] = ("B", "female"))
            ∨ (arrangement[i] = ("B", "female") ∧ arrangement[i+1] = ("A", "male"))

-- Define the condition that neither A nor B can be at the ends of the row
def a_and_b_not_at_ends (arrangement : list (string × string)) : Prop :=
  arrangement.head ≠ ("A", "male") ∧ arrangement.head ≠ ("B", "female") ∧
  arrangement.last ≠ ("A", "male") ∧ arrangement.last ≠ ("B", "female")

-- Combine all the conditions in one definition
def valid_arrangement (arrangement : list (string × string)) : Prop :=
  arrangement.length = total_students ∧ alternating_genders arrangement ∧
  a_and_b_adjacent arrangement ∧ a_and_b_not_at_ends arrangement

-- Theorem stating the number of valid arrangements is 16
theorem number_of_different_arrangements : 
  ∃ (arrangements : list (list (string × string))), 
    (∀ arr, arr ∈ arrangements → valid_arrangement arr) ∧ 
    arrangements.length = 16 :=
sorry

end number_of_different_arrangements_l820_820008


namespace seven_lines_plane_regions_l820_820740

theorem seven_lines_plane_regions
  (lines : Fin 7 → AffineSubspace ℝ ℝ^2) 
  (h_parallel : ∃ (i j : Fin 7), i ≠ j ∧ is_parallel (lines i) (lines j)) 
  (h_non_parallel_non_concurrent : ∀ (i j : Fin 7), i ≠ j → ¬(is_parallel (lines i) (lines j) ∨ ∃ (p : AffineSubspace ℝ ℝ^2), is_intersection (lines j) p ∧ is_intersection (lines i) p)) :
  count_regions lines = 22 :=
by
  sorry

end seven_lines_plane_regions_l820_820740


namespace larry_channels_l820_820761

theorem larry_channels (initial_channels : ℕ)
                       (removed_channels : ℕ)
                       (added_channels1 : ℕ)
                       (downgrade_channels : ℕ)
                       (sports_package : ℕ)
                       (movie_package : ℕ)
                       (duplicate_channels : ℕ)
                       (international_package : ℕ)
                       (overlap_percentage : ℕ)
                       (extreme_sports_package : ℕ)
                       (basic_sports_package : ℕ) :
  initial_channels = 150 →
  removed_channels = 20 →
  added_channels1 = 12 →
  downgrade_channels = 15 →
  sports_package = 8 →
  movie_package = 22 →
  duplicate_channels = 4 →
  international_package = 30 →
  overlap_percentage = 15 →
  extreme_sports_package = 10 →
  basic_sports_package = 5 →
  let total_after_first_change := initial_channels - removed_channels + added_channels1 in
  let total_after_downgrade := total_after_first_change - downgrade_channels in
  let total_after_sports := total_after_downgrade + sports_package in
  let total_after_movie := total_after_sports + movie_package - duplicate_channels in
  let overlap := (overlap_percentage * total_after_movie + 99) / 100 in  -- rounding to the nearest integer
  let net_international := international_package - overlap in
  let total_after_international := total_after_movie + net_international in
  let total_after_extreme_sports := total_after_international + extreme_sports_package - basic_sports_package in
  total_after_extreme_sports = 165 := sorry

end larry_channels_l820_820761


namespace surface_area_of_cube_l820_820745

-- Definition of the problem in Lean 4
theorem surface_area_of_cube (a : ℝ) (s : ℝ) (h : s * Real.sqrt 3 = a) : 6 * (s^2) = 2 * a^2 :=
by
  sorry

end surface_area_of_cube_l820_820745


namespace math_problem_l820_820543

noncomputable def problem_statement : ℝ :=
  4 * Real.sin (60 * Real.pi / 180) - |1 - Real.sqrt 3| + (1 / 3)⁻¹ + Real.cbrt (-27)

theorem math_problem :
  problem_statement = Real.sqrt 3 + 1 :=
by
  sorry

end math_problem_l820_820543


namespace positive_difference_even_odd_l820_820197

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820197


namespace speed_of_student_B_l820_820428

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820428


namespace tiles_painted_in_15_minutes_l820_820555

open Nat

theorem tiles_painted_in_15_minutes:
  let don_rate := 3
  let ken_rate := don_rate + 2
  let laura_rate := 2 * ken_rate
  let kim_rate := laura_rate - 3
  don_rate + ken_rate + laura_rate + kim_rate == 25 → 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = 375 :=
by
  intros
  sorry

end tiles_painted_in_15_minutes_l820_820555


namespace magnitude_of_b_l820_820650

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820650


namespace polynomial_has_three_real_roots_l820_820916

theorem polynomial_has_three_real_roots (a b c : ℝ) (h1 : b < 0) (h2 : a * b = 9 * c) :
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ 
    (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ 
    (x3^3 + a * x3^2 + b * x3 + c = 0) := sorry

end polynomial_has_three_real_roots_l820_820916


namespace positive_difference_l820_820225

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820225


namespace positive_difference_even_odd_sum_l820_820273

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820273


namespace modulus_of_complex_number_l820_820542

def i : ℂ := complex.I

def complex_number : ℂ := (3 - 4 * i) / i

theorem modulus_of_complex_number : complex.abs complex_number = 5 := 
by sorry

end modulus_of_complex_number_l820_820542


namespace water_collection_possible_l820_820733

theorem water_collection_possible (n : ℕ) : 
  (∃ k : ℕ, n = 2^k) ↔ (∃ (f : fin n → ℕ), 
    (∀ i j, f i = f j) ∧
    ∀ m, (m < n) → f m ≠ 0) :=
sorry

end water_collection_possible_l820_820733


namespace sam_walk_distance_l820_820885

-- Define the constants for the problem
def distanceApart : ℝ := 55
def fredSpeed : ℝ := 6
def samSpeed : ℝ := 5

-- Define the main theorem
theorem sam_walk_distance :
  let t := distanceApart / (fredSpeed + samSpeed) in
  samSpeed * t = 25 :=
by
  sorry

end sam_walk_distance_l820_820885


namespace speed_of_student_B_l820_820494

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820494


namespace salt_concentration_l820_820339

theorem salt_concentration (volume_water volume_solution concentration_solution : ℝ)
  (h1 : volume_water = 1)
  (h2 : volume_solution = 0.5)
  (h3 : concentration_solution = 0.45) :
  (volume_solution * concentration_solution) / (volume_water + volume_solution) = 0.15 :=
by
  sorry

end salt_concentration_l820_820339


namespace circles_intersect_l820_820624

theorem circles_intersect (m : ℝ) 
  (h₁ : ∃ x y, x^2 + y^2 = m) 
  (h₂ : ∃ x y, x^2 + y^2 + 6*x - 8*y + 21 = 0) : 
  9 < m ∧ m < 49 :=
by sorry

end circles_intersect_l820_820624


namespace speed_of_student_B_l820_820481

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820481


namespace difference_q_r_after_tax_is_approx_2695_53_l820_820535

namespace MoneyShare

variable (T : ℝ) -- Total amount of money

-- conditions
def ratio_p_before_tax : ℝ := (3 / 22) * T
def ratio_q_before_tax : ℝ := (7 / 22) * T
def ratio_r_before_tax : ℝ := (12 / 22) * T

def p_after_tax : ℝ := ratio_p_before_tax T * (1 - 0.10)
def q_after_tax : ℝ := ratio_q_before_tax T * (1 - 0.15)
def r_after_tax : ℝ := ratio_r_before_tax T * (1 - 0.20)

-- Given difference between p's and q's shares after tax
def difference_pq_after_tax (T : ℝ) : ℝ := (q_after_tax T) - (p_after_tax T)

axiom pq_difference_is_2400 : difference_pq_after_tax T = 2400

-- Prove the difference between q and r's shares after tax
def difference_qr_after_tax (T : ℝ) : ℝ := q_after_tax T - r_after_tax T

theorem difference_q_r_after_tax_is_approx_2695_53 : 
  abs (difference_qr_after_tax T - 2695.53) < 0.01 :=
sorry

end MoneyShare

end difference_q_r_after_tax_is_approx_2695_53_l820_820535


namespace number_of_arrangements_l820_820559

noncomputable def arrangements_nonadjacent_teachers (A : ℕ → ℕ → ℕ) : ℕ :=
  let students_arrangements := A 8 8
  let gaps_count := 9
  let teachers_arrangements := A gaps_count 2
  students_arrangements * teachers_arrangements

theorem number_of_arrangements (A : ℕ → ℕ → ℕ) :
  arrangements_nonadjacent_teachers A = A 8 8 * A 9 2 := 
  sorry

end number_of_arrangements_l820_820559


namespace cross_prod_correct_l820_820988

open Matrix

def vec1 : ℝ × ℝ × ℝ := (3, -1, 4)
def vec2 : ℝ × ℝ × ℝ := (-4, 6, 2)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
  a.2.2 * b.1 - a.1 * b.2.2,
  a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_prod_correct :
  cross_product vec1 vec2 = (-26, -22, 14) := by
  -- sorry is used to simplify the proof.
  sorry

end cross_prod_correct_l820_820988


namespace vandermonde_identity_combinatorial_identity_l820_820328

open Nat

-- Problem 1: Vandermonde Identity
theorem vandermonde_identity (m n k : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : m + n ≥ k) :
  (Finset.range (k + 1)).sum (λ i => Nat.choose m i * Nat.choose n (k - i)) = Nat.choose (m + n) k :=
sorry

-- Problem 2:
theorem combinatorial_identity (p q n : ℕ) (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < n) :
  (Finset.range (p + 1)).sum (λ k => Nat.choose p k * Nat.choose q k * Nat.choose (n + k) (p + q)) =
  Nat.choose n p * Nat.choose n q :=
sorry

end vandermonde_identity_combinatorial_identity_l820_820328


namespace dennis_shirt_cost_l820_820931

variable (initial_amount : ℕ := 50)
variable (change_received : ℕ := 23)
variable (shirt_cost : ℕ)

theorem dennis_shirt_cost : shirt_cost = 27 :=
by
    have h : shirt_cost = initial_amount - change_received,
    { refl },
    rw [h],
    norm_num,
    rw [Nat.sub_eq_iff_eq_add],
    norm_num,
    exact Eq.refl _

end dennis_shirt_cost_l820_820931


namespace positive_difference_sums_l820_820210

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820210


namespace positive_difference_l820_820223

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820223


namespace positive_difference_even_odd_sum_l820_820272

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820272


namespace student_B_speed_l820_820468

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820468


namespace length_of_BC_in_triangle_l820_820022

theorem length_of_BC_in_triangle
  (A B C D : Type)
  [triangle : Triangle A B C]
  [angle_bisector : AngleBisector C D]
  (BD : Real) (CD : Real) (AC : Real)
  (hBD : BD = 1) (hCD : CD = 2) (hAC : AC = 2 * Real.sqrt 3) :
  ∃ BC : Real, BC = Real.sqrt 3 :=
by
  sorry

end length_of_BC_in_triangle_l820_820022


namespace spherical_to_rectangular_coordinates_and_distance_l820_820961

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * sin phi * cos theta, rho * sin phi * sin theta, rho * cos phi)

noncomputable def distance_from_origin (x y z : ℝ) : ℝ :=
  real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)

theorem spherical_to_rectangular_coordinates_and_distance :
  (spherical_to_rectangular 5 (7 * real.pi / 4) (real.pi / 3) = (-5 * real.sqrt 6 / 4, -5 * real.sqrt 6 / 4, 5 / 2)) ∧
  (distance_from_origin (-5 * real.sqrt 6 / 4) (-5 * real.sqrt 6 / 4) (5 / 2) = 5 * real.sqrt 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_and_distance_l820_820961


namespace student_b_speed_l820_820434

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820434


namespace digit_sum_problem_l820_820729

def digit_sum (n : ℕ) : ℕ := n.digits.sum

theorem digit_sum_problem (n : ℕ) :
  (digit_sum n = 111) →
  (digit_sum (7002 * n) = 990) →
  (digit_sum (2003 * n) = 555) :=
by
  sorry

end digit_sum_problem_l820_820729


namespace right_triangular_pyramid_volume_l820_820923

-- Assuming the conditions given.
def volume_of_sphere_circumscribed_around_pyramid
  (a b c : ℝ) (ha : a = sqrt 3) (hb : b = 2) (hc : c = 3) : ℝ :=
  let d := sqrt (a^2 + b^2 + c^2) in
  let r := d / 2 in
  (4 / 3) * pi * r^3

theorem right_triangular_pyramid_volume
  (a b c : ℝ) (ha : a = sqrt 3) (hb : b = 2) (hc : c = 3) :
  volume_of_sphere_circumscribed_around_pyramid a b c ha hb hc = 32 / 3 * pi :=
by
  sorry

end right_triangular_pyramid_volume_l820_820923


namespace sqrt_same_type_as_sqrt_2_l820_820872

theorem sqrt_same_type_as_sqrt_2 (a b : ℝ) :
  ((sqrt a)^2 = 8) ↔ (sqrt 2) * (sqrt 2) = 2 * (sqrt 2) * (sqrt 2)  :=
sorry

end sqrt_same_type_as_sqrt_2_l820_820872


namespace triangle_inequality_sin_csc_l820_820024

theorem triangle_inequality_sin_csc (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
(h_sum : A + B + C = π) :
  (sin (A / 2))^2 + (sin (B / 2))^2 + (sin (C / 2))^2 ≤ (Real.sqrt 3 / 8) * (csc A + csc B + csc C) :=
by
  sorry -- Proof to be filled in later

end triangle_inequality_sin_csc_l820_820024


namespace polygon_sides_l820_820921

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end polygon_sides_l820_820921


namespace number_of_true_propositions_is_four_l820_820531

-- Define each of the propositions as Lean definitions.
def proposition1 : Prop :=
  ∀ (L₁ L₂ L₃ : Line), intersected_by L₃ L₁ L₂ → complementary_interior_angles L₃ L₁ L₂ → corresponding_angles_equal L₃ L₁ L₂

def proposition2 : Prop :=
  ∀ (P : Point) (L : Line), ∃! (L' : Line), passes_through P L' ∧ perpendicular L L'

def proposition3 : Prop :=
  ∀ (P : Point) (L : Line), ∃! (L' : Line), passes_through P L' ∧ parallel L L'

def proposition4 : Prop :=
  ∀ (P : Point) (L : Line), distance P L = length_of_perpendicular_segment P L

def proposition5 : Prop :=
  ∀ (fig1 fig2 : Figure), translated fig1 fig2 → (∀ (P Q : Point), corresponding_points P Q fig1 fig2 → parallel_lines_segment P Q fig1 fig2)

-- The main theorem to prove the number of true propositions.
theorem number_of_true_propositions_is_four :
  (proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 ∧ proposition5) → (count_true [proposition1, proposition2, proposition3, proposition4, proposition5] = 4) :=
by
  sorry

end number_of_true_propositions_is_four_l820_820531


namespace magnitude_of_b_l820_820654

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820654


namespace positive_difference_l820_820224

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820224


namespace johnny_third_job_hours_per_day_l820_820033

theorem johnny_third_job_hours_per_day :
  let daily_wage_job1 := 3 * 7 in
  let daily_wage_job2 := 2 * 10 in
  let total_wage_5_days := 445 in
  let total_wage_no_job3 := 5 * (daily_wage_job1 + daily_wage_job2) in
  let total_wage_job3 := total_wage_5_days - total_wage_no_job3 in
  let hourly_rate_job3 := 12 in
  let total_hours_job3 := total_wage_job3 / hourly_rate_job3 in
  let hours_per_day_job3 := total_hours_job3 / 5 in
  hours_per_day_job3 = 4 :=
by 
  simp only [daily_wage_job1, daily_wage_job2, total_wage_no_job3, total_wage_job3, total_hours_job3, hours_per_day_job3]
  show 20 / 5 = 4,
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end johnny_third_job_hours_per_day_l820_820033


namespace find_speed_B_l820_820393

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820393


namespace positive_difference_eq_250_l820_820167

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820167


namespace maximum_value_of_f_l820_820991

section
variables (a : ℝ) (f : ℝ → ℝ)

noncomputable def f := λ x : ℝ, x^2 - 2 * x + 3

theorem maximum_value_of_f :
  (0 ≤ a ∧ a < 1 → ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ f x = 3) ∧
  (1 ≤ a ∧ a ≤ 2 → ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ f x = 3) ∧
  (a > 2 → ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ f x = a^2 - 2 * a + 3) :=
by
  sorry
end

end maximum_value_of_f_l820_820991


namespace parallelogram_area_l820_820858

noncomputable def unit_circle : ℝ := 1
noncomputable def distance_to_tangency_vertex : ℝ := sqrt 3
noncomputable def area_of_parallelogram := 4 + 8 * sqrt 3 / 3

theorem parallelogram_area : 
  let r := unit_circle in
  let d := distance_to_tangency_vertex in
  ∃ (A B C D : ℝ × ℝ), 
    let parallelogram := ((A, B), (B, C), (C, D), (D, A)) in
    (∀ (circle : ℝ × ℝ), 
      (circle.1 = r ∧ circle.2 ∈ {A, B, C, D} ∧ 
      (abs (circle.1 - A.1) = r ∨ 
       abs (circle.2 - A.2) = r ∨ 
       abs (circle.1 - B.1) = r ∨ 
       abs (circle.2 - B.2) = r ∨ 
       abs (circle.1 - C.1) = r ∨ 
       abs (circle.2 - C.2) = r ∨ 
       abs (circle.1 - D.1) = r ∨ 
       abs (circle.2 - D.2) = r))) →
    parallelogram ≠ adj.length (d) →
    parallelogram ≠ r.height * 8.62 := sorry

end parallelogram_area_l820_820858


namespace g_neither_even_nor_odd_l820_820025

def g (x : ℝ) : ℝ := log (x^2 + x + sqrt (1 + (x + 1)^2))

theorem g_neither_even_nor_odd : ¬ (∀ x, g x = g (-x)) ∧ ¬ (∀ x, g x = - g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l820_820025


namespace positive_difference_of_sums_l820_820185

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820185


namespace building_height_l820_820902

theorem building_height (h : ℕ) (flagpole_height : ℕ) (flagpole_shadow : ℕ) (building_shadow : ℕ) :
  flagpole_height = 18 ∧ flagpole_shadow = 45 ∧ building_shadow = 60 → h = 24 :=
by
  intros
  sorry

end building_height_l820_820902


namespace solve_for_x_l820_820089

theorem solve_for_x : 
  let x := 50 / (8 - (3 / 7))
  in x = 350 / 53 :=
by
  sorry

end solve_for_x_l820_820089


namespace positive_difference_sums_l820_820211

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820211


namespace speed_of_student_B_l820_820491

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820491


namespace number_of_solutions_f_2018_eq_zero_l820_820964

def f (x : ℝ) : ℝ := |x - 1|

theorem number_of_solutions_f_2018_eq_zero :
  {x : ℝ | (Nat.iterate f 2018 x) = 0}.card = 2 ^ 2017 :=
by
  sorry

end number_of_solutions_f_2018_eq_zero_l820_820964


namespace magnitude_of_b_l820_820643

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820643


namespace smallest_b_for_factoring_l820_820580

theorem smallest_b_for_factoring (b : ℕ) (p q : ℕ) (h1 : p * q = 1800) (h2 : p + q = b) : b = 85 :=
by
  sorry

end smallest_b_for_factoring_l820_820580


namespace student_B_speed_l820_820448

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820448


namespace basketball_starting_lineups_l820_820545

theorem basketball_starting_lineups : 
  let total_players := 15
  let fixed_stars := 3
  let lineup_size := 5
  let remaining_players := total_players - fixed_stars
  let additional_players_needed := lineup_size - fixed_stars
  nat.choose remaining_players additional_players_needed = 66 :=
by
  let total_players := 15
  let fixed_stars := 3
  let lineup_size := 5
  let remaining_players := total_players - fixed_stars
  let additional_players_needed := lineup_size - fixed_stars
  exact nat.choose remaining_players additional_players_needed
  sorry

end basketball_starting_lineups_l820_820545


namespace closest_fraction_l820_820989

theorem closest_fraction (n : ℤ) : 
  let frac1 := 37 / 57 
  let closest := 15 / 23
  n = 15 ∧ abs (851 - 57 * n) = min (abs (851 - 57 * 14)) (abs (851 - 57 * 15)) :=
by
  let frac1 := (37 : ℚ) / 57
  let closest := (15 : ℚ) / 23
  have h : 37 * 23 = 851 := by norm_num
  have denom : 57 * 23 = 1311 := by norm_num
  let num := 851
  sorry

end closest_fraction_l820_820989


namespace identify_plan_l820_820755

def scientificPlan1990 (plan : String) (countries : List String) : Prop :=
  plan = "Human Genome Project" ∧
  countries = ["United States", "United Kingdom", "France", "Germany", "Japan", "China"]

theorem identify_plan (plan : String) (countries : List String) :
  scientificPlan1990 plan countries → plan = "Human Genome Project" :=
begin
  intro h,
  cases h,
  exact h_left,
end

end identify_plan_l820_820755


namespace student_B_speed_l820_820514

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820514


namespace cosine_summation_identity_l820_820086

theorem cosine_summation_identity (α φ : ℝ) (n : ℕ) :
  0 < n →
  (∑ k in Finset.range n, Real.cos (k * φ + α)) = 
  (Real.cos ((n - 1 : ℝ) / 2 * φ + α) * Real.sin (n / 2 * φ) / Real.sin (φ / 2)) :=
by
  sorry

end cosine_summation_identity_l820_820086


namespace clock_angle_acute_probability_l820_820806

-- Given the condition that a clock stops randomly at any moment,
-- and defining the probability of forming an acute angle between the hour and minute hands,
-- prove that this probability is 1/2.

theorem clock_angle_acute_probability : 
  (probability (\theta : ℝ, is_acute ⟨θ % 360, 0 ≤ θ % 360 < 360⟩) = 1/2) :=
-- Definitions and conditions.
sorry

end clock_angle_acute_probability_l820_820806


namespace number_to_match_l820_820857

def twenty_five_percent_less (x: ℕ) : ℕ := 3 * x / 4

def one_third_more (n: ℕ) : ℕ := 4 * n / 3

theorem number_to_match (n : ℕ) (x : ℕ) 
  (h1 : x = 80) 
  (h2 : one_third_more n = twenty_five_percent_less x) : n = 45 :=
by
  -- Proof is skipped as per the instruction
  sorry

end number_to_match_l820_820857


namespace find_speed_B_l820_820391

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820391


namespace compute_b_c_sum_l820_820046

def polynomial_decomposition (Q : ℝ[X]) (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ) : Prop :=
  ∀ x : ℝ, Q.eval x = (x^2 + (b1*x) + c1) * (x^2 + (b2*x) + c2) * (x^2 + (b3*x) + c3) * (x^2 + (b4*x) + c4)

theorem compute_b_c_sum (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : polynomial_decomposition (polynomial.mk [1, -1, 1, -1, 1, -1, 1, -1, 1]) b1 b2 b3 b4 c1 c2 c3 c4)
  (c1_eq : c1 = 1) (c2_eq : c2 = 1) (c3_eq : c3 = 1) (c4_eq : c4 = 1) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 := 
sorry

end compute_b_c_sum_l820_820046


namespace positive_difference_even_odd_sums_l820_820245

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820245


namespace find_S5_l820_820767

variable (a_1 d : ℝ)
def S (n : ℕ) : ℝ := n * a_1 + (n * (n - 1) / 2) * d

theorem find_S5 (h1 : S a_1 d 3 = -3) (h2 : S a_1 d 7 = 7) : S a_1 d 5 = 0 :=
by
  -- We use h1 and h2 to derive that a_1 = -2 and d = 1
  -- Therefore, given those values, S 5 = 0
  sorry

end find_S5_l820_820767


namespace Q_ratio_value_l820_820125

-- Definitions and conditions
def g (x : ℂ) := x ^ 2001 - 14 * x ^ 2000 + 1

def distinct_zeros (s : ℕ → ℂ) := ∀ i j, (i ≠ j) → (s i ≠ s j)

def Q (s: ℕ → ℂ) (z : ℂ) := 
  ∏ j in finset.range 2001, (z - (s j + 1 / (s j)))

-- Statement to be proved
theorem Q_ratio_value (s : ℕ → ℂ) 
  (hs : distinct_zeros s) 
  (g_roots : ∀ j ∈ finset.range 2001, g (s j) = 0) : 
  Q s 1 / Q s (-1) = 259 / 289 :=
by
  sorry

end Q_ratio_value_l820_820125


namespace super_soup_new_stores_2020_l820_820099

theorem super_soup_new_stores_2020 :
  ∀ (initial_stores_2018: ℕ) (stores_opened_2019: ℕ) 
    (stores_closed_2019: ℕ) (stores_closed_2020: ℕ) (total_stores_2020: ℕ),
  initial_stores_2018 = 23 →
  stores_opened_2019 = 5 →
  stores_closed_2019 = 2 →
  stores_closed_2020 = 6 →
  total_stores_2020  = 30 →
  (total_stores_2020 = initial_stores_2018 + stores_opened_2019 - stores_closed_2019 - stores_closed_2020 + ?m),
  sorry

end super_soup_new_stores_2020_l820_820099


namespace positive_difference_even_odd_l820_820200

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820200


namespace cosine_third_angle_of_triangle_l820_820751

theorem cosine_third_angle_of_triangle (X Y Z : ℝ)
  (sinX_eq : Real.sin X = 4/5)
  (cosY_eq : Real.cos Y = 12/13)
  (triangle_sum : X + Y + Z = Real.pi) :
  Real.cos Z = -16/65 :=
by
  -- proof will be filled in
  sorry

end cosine_third_angle_of_triangle_l820_820751


namespace solution_set_inequalities_l820_820132

theorem solution_set_inequalities (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 2 * x) / 3 > x - 1) → (x ≤ 1) :=
by
  intros h
  sorry

end solution_set_inequalities_l820_820132


namespace maximal_product_decomposition_2008_l820_820550

theorem maximal_product_decomposition_2008
  (a : ℕ) (b : ℕ)
  (ha : a = 668)
  (hb : b = 2) :
  ∃ (x : ℕ → ℕ), (x 3 = a) ∧ (x 2 = b) ∧ (∑ n in set.univ, n * x n = 2008) ∧ 
  ∏ n in set.univ, n ^ x n = 3^668 * 2^2 := 
sorry

end maximal_product_decomposition_2008_l820_820550


namespace student_B_speed_l820_820507

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820507


namespace GCD_40_48_l820_820160

theorem GCD_40_48 : Int.gcd 40 48 = 8 :=
by sorry

end GCD_40_48_l820_820160


namespace student_b_speed_l820_820371

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820371


namespace difference_max_min_changes_l820_820537

theorem difference_max_min_changes (initial_yes: ℝ) (initial_no: ℝ) (final_yes: ℝ) (final_no: ℝ) (x_difference: ℝ) :
    initial_yes = 50 ∧ initial_no = 50 ∧ final_yes = 70 ∧ final_no = 30 →
    x_difference = 60 :=
begin
    sorry
end

end difference_max_min_changes_l820_820537


namespace positive_difference_even_odd_sums_l820_820268

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820268


namespace time_train_passes_jogger_l820_820907

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

noncomputable def initial_lead_m : ℝ := 150
noncomputable def train_length_m : ℝ := 100

noncomputable def total_distance_to_cover_m : ℝ := initial_lead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_to_cover_m / relative_speed_mps

theorem time_train_passes_jogger : time_to_pass_jogger_s = 25 := by
  sorry

end time_train_passes_jogger_l820_820907


namespace mabel_transactions_eq_90_l820_820073

variable (transactions_mabel : ℝ)

def anthony_transactions := 1.10 * transactions_mabel
def cal_transactions := (2/3) * anthony_transactions
def jade_transactions := cal_transactions + 16

theorem mabel_transactions_eq_90
  (h : jade_transactions = 82) : transactions_mabel = 90 := 
by sorry

end mabel_transactions_eq_90_l820_820073


namespace find_speed_of_B_l820_820418

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820418


namespace list_price_of_article_l820_820119

theorem list_price_of_article 
(paid_price : ℝ) 
(first_discount second_discount : ℝ)
(list_price : ℝ)
(h_paid_price : paid_price = 59.22)
(h_first_discount : first_discount = 0.10)
(h_second_discount : second_discount = 0.06000000000000002)
(h_final_price : paid_price = (1 - first_discount) * (1 - second_discount) * list_price) :
  list_price = 70 := 
by
  sorry

end list_price_of_article_l820_820119


namespace pi_rational_approximation_inequality_l820_820816

-- Given conditions
variables (p q : ℕ) (h : 1 < q)

-- Mathematical goal
theorem pi_rational_approximation_inequality :
  ∀ p q : ℕ, 1 < q → abs (Real.pi - p / q) ≥ q ^ (-42) :=
by
  intros p q h
  sorry

end pi_rational_approximation_inequality_l820_820816


namespace positive_diff_even_odd_sums_l820_820173

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820173


namespace compute_sum_of_products_of_coefficients_l820_820050

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end compute_sum_of_products_of_coefficients_l820_820050


namespace odd_subsets_S4_correct_odd_even_subset_count_equal_sum_capacities_equal_l820_820041

def is_subset {α : Type} (X S : set α) : Prop := X ⊆ S

def capacity (X : set ℕ) : ℕ := X.sum id

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

def is_even_subset (X S : set ℕ) : Prop := is_subset X S ∧ is_even (capacity X)
def is_odd_subset (X S : set ℕ) : Prop := is_subset X S ∧ is_odd (capacity X)

-- Problem I: Verify that the given list matches the set of odd subsets of S_4.
theorem odd_subsets_S4_correct : 
  ∀ X, X ∈ [∅, {1}, {2, 3}, {2, 4}, {3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}, {1, 2, 3, 4}] ↔ is_odd_subset X {1, 2, 3, 4} :=
by sorry

-- Problem II: Prove the number of odd subsets of S_n equals the number of even subsets.
theorem odd_even_subset_count_equal (n : ℕ) : 
  card {X : set ℕ // is_odd_subset X (finset.range n).to_set} = card {X : set ℕ // is_even_subset X (finset.range n).to_set} :=
by sorry

-- Problem III: Prove that for n ≥ 3, the sum of capacities of all odd subsets of S_n is equal 
-- to the sum of capacities of all even subsets.
theorem sum_capacities_equal (n : ℕ) (h : n ≥ 3) :
  ∑ X in (finset.powerset (finset.range n)), if is_odd (capacity X.to_set) then capacity X.to_set else 0 =
  ∑ X in (finset.powerset (finset.range n)), if is_even (capacity X.to_set) then capacity X.to_set else 0 :=
by sorry

end odd_subsets_S4_correct_odd_even_subset_count_equal_sum_capacities_equal_l820_820041


namespace magnitude_b_eq_sqrt3_l820_820682

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820682


namespace multiple_inverses_exist_l820_820862

theorem multiple_inverses_exist (op : α → α → α) (inv1 inv2 : α → α)
  (h_noncomm : ∃ x y : α, op x y ≠ op y x) :
  (∀ x : α, op (inv1 x) x = 1) ∧ (∀ x : α, op (inv2 x) x = 1) :=
sorry

end multiple_inverses_exist_l820_820862


namespace magnitude_b_eq_sqrt3_l820_820675

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820675


namespace integer_roots_of_polynomial_l820_820841

noncomputable def polynomial_eq : Polynomial ℚ :=
  Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C a * Polynomial.X ^ 3 +
  Polynomial.C b * Polynomial.X ^ 1 +
  Polynomial.C c * Polynomial.X ^ 0

theorem integer_roots_of_polynomial (a b c : ℚ) (h_roots : (3 + Real.sqrt 5) ∈ polynomial_eq.roots ∧ (3 - Real.sqrt 5) ∈ polynomial_eq.roots) :
  ∃ m n : ℤ, polynomial_eq.eval (m : ℚ) = 0 ∧ polynomial_eq.eval (n : ℚ) = 0 ∧ m + n = -6 :=
by
  sorry

end integer_roots_of_polynomial_l820_820841


namespace polygon_sides_l820_820919

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end polygon_sides_l820_820919


namespace speed_of_student_B_l820_820427

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820427


namespace relationship_between_volumes_l820_820155

-- Definitions
variables (r h : ℝ) (π : ℝ := Real.pi) (A M : ℝ)
def volume_cone (A : ℝ) : Prop := A = (1 / 3) * π * r^2 * h
def volume_cylinder (M : ℝ) : Prop := M = 2 * π * r^2 * h

-- Theorem statement
theorem relationship_between_volumes (h : ℝ) (r : ℝ) (A M : ℝ) 
  (cone_volume : volume_cone A) (cylinder_volume : volume_cylinder M) : A = (1 / 3) * M :=
by
  -- Sorry indicates the proof is omitted
  sorry

end relationship_between_volumes_l820_820155


namespace no_solution_exists_l820_820984

open Int

theorem no_solution_exists (x y z : ℕ) (hx : x > 0) (hy : y > 0)
  (hz : z = Nat.gcd x y) : x + y^2 + z^3 ≠ x * y * z := 
sorry

end no_solution_exists_l820_820984


namespace ellipse_equation_params_l820_820939

-- Defining the parametric equations
def parametric_ellipse (t : ℝ) : ℝ × ℝ :=
  ( (2 * (Real.sin t - 1)) / (2 - Real.cos t), 
    (3 * (Real.cos t - 5)) / (2 - Real.cos t) )

-- Statement of the problem in Lean 4
theorem ellipse_equation_params :
  ∃ A B C D E F : ℤ, 
    (∀ t : ℝ, 
      let (x, y) := parametric_ellipse t in
      (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) 
    ∧ Int.gcd_A_B_C_D_E_F A B C D E F = 1
    ∧ (|A| + |B| + |C| + |D| + |E| + |F| = 1381) :=
sorry

end ellipse_equation_params_l820_820939


namespace base7_to_base10_proof_l820_820829

theorem base7_to_base10_proof (c d : ℕ) (h1 : 764 = 4 * 100 + c * 10 + d) : (c * d) / 20 = 6 / 5 :=
by
  sorry

end base7_to_base10_proof_l820_820829


namespace student_B_speed_l820_820451

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820451


namespace Q_at_1_eq_1_l820_820849

theorem Q_at_1_eq_1
  (Q : Polynomial ℚ)
  (h1 : degree Q = 4)
  (h2 : Q.coeff 4 = 1)
  (h3 : Q.is_root (Real.sqrt 3 + Real.sqrt 7)) :
  Q.eval 1 = 1 :=
sorry

end Q_at_1_eq_1_l820_820849


namespace least_period_fibonacci_l820_820097

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem least_period_fibonacci : ∃ t : ℕ, (∀ n : ℕ, n > 0 → fibonacci n = fibonacci (n + t)) ∧ t = 60 := 
by 
  sorry

end least_period_fibonacci_l820_820097


namespace positive_difference_sums_even_odd_l820_820237

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820237


namespace g_max_min_l820_820629

-- Definition of the function f
def f (x : ℝ) : ℝ := 2^x

-- Definition of the function g
def g (x : ℝ) : ℝ := f (2 * x) - f (x + 2)

-- Statement: Proving the maximum and minimum values of g(x) over the interval [0, 1]
theorem g_max_min:
(∀ x, 0 ≤ x ∧ x ≤ 1 → g x ≤ -3 ∧ g x ≥ -4) ∧
(∃ x, 0 ≤ x ∧ x ≤ 1 ∧ g x = -3) ∧
(∃ x, 0 ≤ x ∧ x ≤ 1 ∧ g x = -4) :=
by
  sorry

end g_max_min_l820_820629


namespace positive_difference_l820_820226

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820226


namespace digit_difference_is_7_l820_820005

def local_value (d : Nat) (place : Nat) : Nat :=
  d * (10^place)

def face_value (d : Nat) : Nat :=
  d

def difference (d : Nat) (place : Nat) : Nat :=
  local_value d place - face_value d

def numeral : Nat := 65793

theorem digit_difference_is_7 :
  ∃ d place, 0 ≤ d ∧ d < 10 ∧ difference d place = 693 ∧ d ∈ [6, 5, 7, 9, 3] ∧ numeral = 65793 ∧
  (local_value 6 4 = 60000 ∧ local_value 5 3 = 5000 ∧ local_value 7 2 = 700 ∧ local_value 9 1 = 90 ∧ local_value 3 0 = 3 ∧
   face_value 6 = 6 ∧ face_value 5 = 5 ∧ face_value 7 = 7 ∧ face_value 9 = 9 ∧ face_value 3 = 3) ∧ 
  d = 7 :=
sorry

end digit_difference_is_7_l820_820005


namespace find_b_when_a_is_1600_l820_820819

theorem find_b_when_a_is_1600 :
  ∀ (a b : ℝ), (a * b = 400) ∧ ((2 * a) * b = 600) → (1600 * b = 600) → b = 0.375 :=
by
  intro a b
  intro h
  sorry

end find_b_when_a_is_1600_l820_820819


namespace trailing_zeros_sum_15_factorial_l820_820312

theorem trailing_zeros_sum_15_factorial : 
  let k := 5
  let h := 3
  k + h = 8 := by
  sorry

end trailing_zeros_sum_15_factorial_l820_820312


namespace max_value_of_expr_l820_820782

theorem max_value_of_expr  
  (a b c : ℝ) 
  (h₀ : 0 ≤ a)
  (h₁ : 0 ≤ b)
  (h₂ : 0 ≤ c)
  (h₃ : a + 2 * b + 3 * c = 1) :
  a + b^3 + c^4 ≤ 0.125 := 
sorry

end max_value_of_expr_l820_820782


namespace area_of_rectangle_l820_820348

theorem area_of_rectangle (A G Y : ℝ) 
  (hG : G = 0.15 * A) 
  (hY : Y = 21) 
  (hG_plus_Y : G + Y = 0.5 * A) : 
  A = 60 := 
by 
  -- proof goes here
  sorry

end area_of_rectangle_l820_820348


namespace positive_difference_even_odd_sums_l820_820250

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820250


namespace parabola_transformation_l820_820719

theorem parabola_transformation :
  ∀ (x y : ℝ), (y = x^2) →
  let shiftedRight := (x - 3)
  let shiftedUp := (shiftedRight)^2 + 4
  y = shiftedUp :=
by
  intros x y,
  intro hyp,
  let shiftedRight := (x - 3),
  let shiftedUp := (shiftedRight)^2 + 4,
  have eq_shifted : y = shiftedRight^2 := by sorry,
  have eq_final : y = shiftedUp := by sorry,
  exact eq_final
  
end parabola_transformation_l820_820719


namespace sum_abs_roots_of_polynomial_l820_820581

noncomputable def polynomial_sum_abs_roots : ℝ :=
  ∑ r in (roots (X^4 - 6*X^3 + 13*X^2 - 12*X + 4)).to_finset, |(r:ℝ)|

theorem sum_abs_roots_of_polynomial : polynomial_sum_abs_roots = 6 :=
sorry

end sum_abs_roots_of_polynomial_l820_820581


namespace positive_difference_even_odd_sums_l820_820267

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820267


namespace count_valid_b1_l820_820770

def sequence (b : ℕ → ℕ) :=
  ∀ n, (even (b n) → b (n + 1) = b n / 2) ∧ (odd (b n) → b (n + 1) = 5 * b n + 1)

theorem count_valid_b1 : 
  let valid_b1 := {b | b ≤ 3000 ∧ b < (λ b, if even b then b / 2 else 5 * b + 1) (λ b, if even b then b / 2 else 5 * b + 1 b)} in
  valid_b1.card = 1500 := 
by sorry

end count_valid_b1_l820_820770


namespace student_B_speed_l820_820458

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820458


namespace find_angle_ADE_l820_820015

theorem find_angle_ADE 
  (A B C D E : Type)
  [triangle ABC]
  (h_isosceles : AB = BC)
  (h_angle_ABC : ∠ABC = 20)
  (h_angle_DAC : ∠DAC = 60)
  (h_angle_ECA : ∠ECA = 50) : 
  ∠ADE = 30 :=
sorry

end find_angle_ADE_l820_820015


namespace sqrt_expression_value_l820_820583

theorem sqrt_expression_value : 
    (Real.sqrt 1.5) / (Real.sqrt 0.81) + 
    (Real.sqrt 1.44) / (Real.sqrt 0.49) ≈ 3.07511334917 := 
by
    sorry

end sqrt_expression_value_l820_820583


namespace hyperbola_asymptote_slope_l820_820999

theorem hyperbola_asymptote_slope :
  ∀ {x y : ℝ}, (x^2 / 144 - y^2 / 81 = 1) → (∃ m : ℝ, ∀ x, y = m * x ∨ y = -m * x ∧ m = 3 / 4) :=
by
  sorry

end hyperbola_asymptote_slope_l820_820999


namespace bc_sum_eq_neg_one_l820_820043

variables {b1 b2 b3 b4 c1 c2 c3 c4 : ℝ}

/-- Given the equation for all real numbers x:
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4),
    prove that b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1. -/
theorem bc_sum_eq_neg_one :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4))
  → b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by {
  sorry,
}

end bc_sum_eq_neg_one_l820_820043


namespace positive_difference_of_sums_l820_820186

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820186


namespace acute_angle_probability_l820_820802

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end acute_angle_probability_l820_820802


namespace find_speed_B_l820_820399

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820399


namespace trajectory_of_square_is_line_l820_820747

open Complex

theorem trajectory_of_square_is_line (z : ℂ) (h : z.re = z.im) : ∃ c : ℝ, z^2 = Complex.I * (c : ℂ) :=
by
  sorry

end trajectory_of_square_is_line_l820_820747


namespace magnitude_of_b_l820_820652

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820652


namespace conj_in_fourth_quadrant_l820_820593

def Z : ℂ := 2 * I / (1 + I)
def Z_conj : ℂ := conj Z
def point_quadrant (z : ℂ) : ℤ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0 -- This could correspond to the axes or origin, which isn't considered in this problem.

theorem conj_in_fourth_quadrant : point_quadrant (Z_conj) = 4 := by
  sorry

end conj_in_fourth_quadrant_l820_820593


namespace dollars_saved_is_correct_l820_820846

noncomputable def blender_in_store_price : ℝ := 120
noncomputable def juicer_in_store_price : ℝ := 80
noncomputable def blender_tv_price : ℝ := 4 * 28 + 12
noncomputable def total_in_store_price_with_discount : ℝ := (blender_in_store_price + juicer_in_store_price) * 0.90
noncomputable def dollars_saved : ℝ := total_in_store_price_with_discount - blender_tv_price

theorem dollars_saved_is_correct :
  dollars_saved = 56 := by
  sorry

end dollars_saved_is_correct_l820_820846


namespace number_of_four_digit_numbers_number_of_four_digit_even_numbers_number_of_four_digit_numbers_divisible_by_5_l820_820635

-- Define the digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

-- Problem 1
theorem number_of_four_digit_numbers (no_repeat : digits.nodup) :
  (list.countP (λ n, n >= 1000 ∧ n < 10000) 
    (list.permutations digits).filter (λ l, l.length == 4)).length = 720 :=
sorry

-- Problem 2
theorem number_of_four_digit_even_numbers (no_repeat : digits.nodup ) :
  (list.countP (λ n, n >= 1000 ∧ n < 10000 ∧ n % 2 == 0) 
    (list.permutations digits).filter (λ l, l.length == 4)).length = 420 := 
sorry

-- Problem 3
theorem number_of_four_digit_numbers_divisible_by_5 (no_repeat : digits.nodup) :
  (list.countP (λ n, n >= 1000 ∧ n < 10000 ∧ n % 5 == 0 )
    (list.permutations digits).filter (λ l, l.length == 4)).length = 220 :=
sorry

end number_of_four_digit_numbers_number_of_four_digit_even_numbers_number_of_four_digit_numbers_divisible_by_5_l820_820635


namespace student_b_speed_l820_820373

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820373


namespace num_possible_values_m_plus_n_l820_820616

-- Define constants m and n
variables (m n : ℝ)

-- Define the monomials
def monomial1 := λ x y : ℝ, 4 * x ^ 2 * y
def monomial2 := λ x y : ℝ, m * x ^ 3 - n ^ 2 * y
def monomial3 := λ x y : ℝ, 8 * x ^ 3 * y

-- Define the condition that the sum is a monomial
def is_monomial_sum := ∀ x y : ℝ, ∃ a b c : ℝ, 
  (monomial1 x y) + (monomial2 x y) + (monomial3 x y) = a * (x ^ b) * (y ^ c)

-- Define the proof problem
theorem num_possible_values_m_plus_n : is_monomial_sum m n → ∃ k, m + n = k :=
by
  sorry -- Proof will be completed here

end num_possible_values_m_plus_n_l820_820616


namespace student_B_speed_l820_820471

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820471


namespace sphere_checkerboard_half_black_l820_820853

-- Define the problem statement in Lean
theorem sphere_checkerboard_half_black (S : Sphere) (P : Point)
  (H : P ∈ S.interior)
  (π1 π2 π3 : Plane)
  (H1 : π1 ⊥ π2)
  (H2 : π1 ⊥ π3)
  (H3 : π2 ⊥ π3)
  (H4 : P ∈ π1 ∧ P ∈ π2 ∧ P ∈ π3)
  (coloring : AlternatingColoring (S.intersection π1) (S.intersection π2) (S.intersection π3)) :
  (colored_surface_area S black) = (S.surface_area / 2) :=
sorry

end sphere_checkerboard_half_black_l820_820853


namespace expected_value_min_eq_sum_l820_820891

-- Definitions for the conditions in the problem
noncomputable def xi : ℕ → ℝ := sorry
noncomputable def eta : ℕ → ℝ := sorry
def independent : Prop := sorry -- Definition of independence
def expected_value_xi : ℝ := sorry -- Definition of expected value of xi
def expected_value_eta : ℝ := sorry -- Definition of expected value of eta

axiom xi_eta_independent : independent
axiom expected_value_xi_finite_or_eta_finite : expected_value_xi < ∞ ∨ expected_value_eta < ∞

theorem expected_value_min_eq_sum :
  (∀ xi eta, independent → (expected_value_xi < ∞ ∨ expected_value_eta < ∞) →
  ∑ n in (nat.filter (λ n, n ≥ 1)), (P xi n) * (P eta n) = E (xi ⊓ eta)) := by sorry

end expected_value_min_eq_sum_l820_820891


namespace shift_parabola_3_right_4_up_l820_820727

theorem shift_parabola_3_right_4_up (x : ℝ) : 
  let y := x^2 in
  (shifted_y : ℝ) = ((x - 3)^2 + 4) :=
begin
  sorry
end

end shift_parabola_3_right_4_up_l820_820727


namespace positive_diff_even_odd_sums_l820_820179

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820179


namespace angles_same_terminal_side_l820_820936

def angle_equiv (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angles_same_terminal_side : angle_equiv (-390 : ℝ) (330 : ℝ) :=
sorry

end angles_same_terminal_side_l820_820936


namespace problem_statement_l820_820874

theorem problem_statement :
  ∀ x : ℝ, ¬ (x^2 - x + 1 = 0) := 
by
  intro x
  have h : (x^2 - x + 1) = ( (x - 1/2)^2 + 3/4 )
  { ring }
  rw h
  apply ne_of_gt
  apply add_pos_of_pos_of_nonneg
  apply pow_two_nonneg
  norm_num


end problem_statement_l820_820874


namespace total_admission_methods_l820_820970

noncomputable def number_of_admission_methods : ℕ :=
  (Combinatorics.choose 4 1) * (Combinatorics.perm 5 3)

theorem total_admission_methods : number_of_admission_methods = 240 := by
  sorry

end total_admission_methods_l820_820970


namespace speed_of_student_B_l820_820482

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820482


namespace magnitude_of_b_l820_820653

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820653


namespace min_diff_of_composite_sum_99_l820_820333

def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ n = p * q

theorem min_diff_of_composite_sum_99 :
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = 99 ∧ abs (a - b) = 1 :=
by
  sorry

end min_diff_of_composite_sum_99_l820_820333


namespace candy_bar_cost_correct_l820_820926

noncomputable def candy_bar_cost : ℕ := 25 -- Correct answer from the solution

theorem candy_bar_cost_correct (C : ℤ) (H1 : 3 * C + 150 + 50 = 11 * 25)
  (H2 : ∃ C, C ≥ 0) : C = candy_bar_cost :=
by
  sorry

end candy_bar_cost_correct_l820_820926


namespace positive_difference_eq_250_l820_820165

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820165


namespace quadratic_passing_point_l820_820114

theorem quadratic_passing_point :
  ∃ (m : ℝ), (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = 8 → x = 0) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = -10 → x = -1) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = m → x = 5) →
  m = 638 := by
  sorry

end quadratic_passing_point_l820_820114


namespace percentage_of_eggs_eaten_l820_820903

-- Definitions from conditions
def total_eggs : Nat := 800
def dry_up_percentage : Float := 0.10
def hatched_frogs : Nat := 40
def hatch_rate : Rational := 1 / 4

-- Mathematical proof problem
theorem percentage_of_eggs_eaten : 
  let dry_up_eggs := dry_up_percentage * total_eggs
  let remaining_eggs := total_eggs - dry_up_eggs
  let hatched_eggs := hatched_frogs / hatch_rate
  let eaten_eggs := remaining_eggs - hatched_eggs
  let percentage_eaten := (eaten_eggs / total_eggs) * 100
  percentage_eaten = 70 := by sorry

end percentage_of_eggs_eaten_l820_820903


namespace num_valid_digit_arrangements_l820_820861

theorem num_valid_digit_arrangements : 
  let digits := [1, 2, 3, 4, 5, 6] in
  let all_permutations := (List.permutations digits).filter (fun p => p.length = 6 ∧ p.nodup) in
  let valid_permutations := all_permutations.filter (fun p => p.indexOf 1 < p.indexOf 2 ∧ p.indexOf 1 < p.indexOf 3) in
  valid_permutations.length = 240 :=
by
  -- Definition of digits
  let digits := [1, 2, 3, 4, 5, 6]
  -- Calculate all permutations of digits
  let all_permutations := (List.permutations digits).filter (fun p => p.length = 6 ∧ p.nodup)
  -- Filter for valid permutations where 1 is to the left of both 2 and 3
  let valid_permutations := all_permutations.filter (fun p => p.indexOf 1 < p.indexOf 2 ∧ p.indexOf 1 < p.indexOf 3)
  -- The total number of valid permutations should be 240
  have : valid_permutations.length = 240 := sorry
  -- Conclusion
  exact this

end num_valid_digit_arrangements_l820_820861


namespace math_proof_problem_l820_820953

noncomputable def problem_statement : ℝ :=
  let sin_45 := Real.sin (Real.pi / 4)
  let log_1265 := Real.log 1265 / Real.log 10
  let cos_60 := Real.cos (Real.pi / 3)
  (1.68 * sin_45 * log_1265^2 / 21) / (6 - cos_60 * 9)

theorem math_proof_problem : problem_statement ≈ 3.481 := by
  sorry

end math_proof_problem_l820_820953


namespace target_heart_rate_of_30_year_old_l820_820937

variable (age : ℕ) (T M : ℕ)

def maximum_heart_rate (age : ℕ) : ℕ :=
  210 - age

def target_heart_rate (M : ℕ) : ℕ :=
  (75 * M) / 100

theorem target_heart_rate_of_30_year_old :
  maximum_heart_rate 30 = 180 →
  target_heart_rate (maximum_heart_rate 30) = 135 :=
by
  intros h1
  sorry

end target_heart_rate_of_30_year_old_l820_820937


namespace probability_tile_in_PAIR_l820_820976

theorem probability_tile_in_PAIR :
  let tiles := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
  let pair_letters := ['P', 'A', 'I', 'R']
  let matching_counts := (sum ([1, 1, 2, 1] : List ℕ))
  matching_counts.fst = 5
  let total_tiles := 12
  (matching_counts.toRational / total_tiles.toRational) = (5 / 12) :=
by sorry

end probability_tile_in_PAIR_l820_820976


namespace smallest_value_l820_820815

theorem smallest_value (a b : ℕ) (ha : a < 6) (hb : b < 9) : 
  ∃ a b, a < 6 ∧ b < 9 ∧ 3 * a - 2 * a * b = -65 :=
by
  use 5, 8
  simp
  sorry

end smallest_value_l820_820815


namespace number_of_white_balls_l820_820336

section
variable (totalBalls greenBalls yellowBalls redBalls purpleBalls whiteBalls : ℕ)
variable (probNoRedNoPurple : ℝ)

axiom h1 : totalBalls = 100
axiom h2 : greenBalls = 30
axiom h3 : yellowBalls = 10
axiom h4 : redBalls = 7
axiom h5 : purpleBalls = 3
axiom h6 : probNoRedNoPurple = 0.9

theorem number_of_white_balls :
  whiteBalls = totalBalls - (greenBalls + yellowBalls + redBalls + purpleBalls) → whiteBalls = 50 := by
  intros
  rw [h1, h2, h3, h4, h5]
  -- Calculation would be shown here but proof is omitted
  exact sorry
end

end number_of_white_balls_l820_820336


namespace cost_equiv_banana_pear_l820_820946

-- Definitions based on conditions
def Banana : Type := ℝ
def Apple : Type := ℝ
def Pear : Type := ℝ

-- Given conditions
axiom cost_equiv_1 : 4 * (Banana : ℝ) = 3 * (Apple : ℝ)
axiom cost_equiv_2 : 9 * (Apple : ℝ) = 6 * (Pear : ℝ)

-- Theorem to prove
theorem cost_equiv_banana_pear : 24 * (Banana : ℝ) = 12 * (Pear : ℝ) :=
by
  sorry

end cost_equiv_banana_pear_l820_820946


namespace area_ratio_ABC_PQC_l820_820754

open EuclideanGeometry

-- Assuming the existence of points A, B, C in the plane
variables (A B C P R Q : Point)
(hABC : Triangle A B C)
(hP : SegRatio A P B 1 2) -- Point P is one-third along AB closer to A
(hR : SegRatio P R B 1 2) -- Point R is one-third along PB closer to P
(hAngles : ∠ P C B = ∠ R Q B) -- Angles PCB and RQB are congruent
(hQ : OnLine Q B C) -- Q lies on segment BC

-- Define areas of the triangles
noncomputable def area_ABC := TriangleArea A B C
noncomputable def area_PQC := TriangleArea P Q C

-- The ratio of the areas of ΔABC and ΔPQC is 9:2
theorem area_ratio_ABC_PQC (hQ : angle P C B = angle R Q B): area_ABC A B C / area_PQC P Q C = 9 / 2 :=
sorry

end area_ratio_ABC_PQC_l820_820754


namespace magnitude_of_b_l820_820647

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820647


namespace perpendicular_lines_slope_l820_820969

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, y = -3 * x + 5 → 4 * y - a * x = 16 → (sorry)) → a = 4 / 3 :=
begin
  sorry
end

end perpendicular_lines_slope_l820_820969


namespace make_all_coins_heads_l820_820142

-- Define an array of 100 coins, each coin can be heads (true) or tails (false)
def CArray := array bool 100

-- Define an operation which is allowed: flipping seven equally spaced coins
structure Operation :=
  (start : Fin 100)  -- starting position of the operation
  (interval : ℕ)    -- interval between flipped coins, with interval > 0

-- Define the resulting coins array after performing a list of operations
def operates (ops: list Operation) (coins: CArray) : CArray :=
  -- This is a pseudo code definition; the actual implementation would depend on details
  sorry

-- Define the initial state where not all coins show heads
def initialCoins : CArray :=
  -- This can be any initial combination; let's consider all showing tails (False)
  Array.mk (List.replicate 100 false)

-- Define the target state where all coins show heads
def targetCoins : CArray :=
  Array.mk (List.replicate 100 true)

-- The statement to prove: there exists a list of operations such that starting with
-- the initial state, the resulting state's coin will be all heads.
theorem make_all_coins_heads : ∃ ops : list Operation,
  operates ops initialCoins = targetCoins :=
by
  sorry

end make_all_coins_heads_l820_820142


namespace positive_difference_sums_l820_820219

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820219


namespace value_of_a4_seq_is_geometric_general_formula_l820_820783

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 / 2 ∧
  a 3 = 5 / 4 ∧
  ∀ n, n ≥ 2 → (4 * (∑ i in finset.range (n + 2 + 1), a i) + 5 * (∑ i in finset.range (n + 1), a i) = 8 * (∑ i in finset.range (n + 1 + 1), a i) + (∑ i in finset.range n, a (i - 1)))

theorem value_of_a4 :
  ∃ a : ℕ → ℝ, sequence a ∧ a 4 = 7 / 8 :=
by sorry

theorem seq_is_geometric :
  ∃ a : ℕ → ℝ, sequence a ∧ ∃ r : ℝ, r > 0 ∧ ∀ n ≥ 1, (a (n + 1) - (1 / 2) * a n) = r * (a n - (1 / 2) * a (n - 1)) :=
by sorry

theorem general_formula :
  ∃ a : ℕ → ℝ, sequence a ∧ ∀ n, n ≥ 1 → a n = (2 * n - 1) * (1 / 2)^(n - 1) :=
by sorry

end value_of_a4_seq_is_geometric_general_formula_l820_820783


namespace simple_interest_years_l820_820518

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  P * (R / 100) * T

theorem simple_interest_years (P T R : ℝ) (hP : P = 1000) (h : simple_interest P (R + 3) T = simple_interest P R T + 120) :
  T = 4 :=
by
  rw [simple_interest, simple_interest] at h
  simp at h
  have eq1 : 1000 * (R + 3) * T / 100 - 1000 * R * T / 100 = 120,
  { sorry }
  rw [div_eq_iff, mul_comm 100, mul_mul_mul_comm, mul_mul_mul_comm] at eq1,
  { sorry },
  { sorry },
  have eq2 : 10 * (R + 3) * T - 10 * R * T = 120,
  { sorry }
  rw [← sub_eq_iff_eq_add] at eq2,
  simp only [mul_add, mul_sub, add_sub_cancel'_right] at eq2,
  have h1 : (R * T + 3 * T - R * T = 12) -> (3 * T = 12),
  { sorry }
  rw [← sub_eq_iff_eq_add] at h1,
  simp at h1,
  have h2 : 3 * T = 12,
  { sorry }
  field_simp at eq2 h1,
  norm_num at h,
  exact eq1

end simple_interest_years_l820_820518


namespace number_of_girls_in_school_l820_820741

theorem number_of_girls_in_school:
  ∃ (B G : ℕ), B + G = 640 ∧ 12 * B + 11 * G = 11.75 * 640 ∧ G = 160 :=
by
  sorry -- Proof to be completed

end number_of_girls_in_school_l820_820741


namespace smallest_positive_a_l820_820830

open Function

section problem

variable {α β : Type} [Add α] [Add β] {f : α → β}

-- Given that f(x - 20) = f(x)
axiom periodic_shift (f : α → β) (c : α) : (∀ x, f (x - c) = f x)

-- Let c be 20
def c : ℕ := 20

-- We need to show that the smallest positive a such that f∘(λ x, x / 5) shifted a units right gives the same graph is 100.
theorem smallest_positive_a (h : periodic_shift f c) : 
    ∃ (a : ℕ), 0 < a ∧ a = 100 ∧ 
    (∀ x, f (x / 5 - a / 5) = f (x / 5)) :=
sorry

end problem

end smallest_positive_a_l820_820830


namespace positive_difference_even_odd_sums_l820_820247

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820247


namespace new_energy_vehicle_price_l820_820070

theorem new_energy_vehicle_price (x : ℝ) :
  (5000 / (x + 1)) = (5000 * (1 - 0.2)) / x :=
sorry

end new_energy_vehicle_price_l820_820070


namespace compute_sum_of_products_of_coefficients_l820_820051

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end compute_sum_of_products_of_coefficients_l820_820051


namespace positive_difference_sums_even_odd_l820_820234

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820234


namespace initial_garrison_men_l820_820905

theorem initial_garrison_men (M : ℕ) (h1 : ∀ P : ℕ, P = M * 31) 
  (h2 : ∀ P_left : ℕ, P_left = M * 4) 
  (h3 : ∀ P_left_remain : ℕ, P_left_remain = (M - 200) * 8) : 
  M = 400 :=
by
  -- Assuming the conditions given
  have P := h1 (M * 31)
  have P_left := h2 (M * 4)
  have P_left_remain := h3 ((M - 200) * 8)
  -- Equating the two expressions for P_left and solving for M
  have P_eq : M * 4 = (M - 200) * 8, by sorry
  exact eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_eq_of_steps_of_eq_of_some_steps_fn_steps_eq_of_steps_fn_steps_eq_steps_steps_fn_steps_of_first_steps_step_eq_first_steps_eq_fn_fn_steps_fn_step... P_eq
  -- Steps to solve for M omitted for brevity
  exact sorry

end initial_garrison_men_l820_820905


namespace smallest_class_number_l820_820343

theorem smallest_class_number (sum_classes : ℕ) (n_classes interval number_of_classes : ℕ) 
                              (h_sum : sum_classes = 87) (h_n_classes : n_classes = 30) 
                              (h_interval : interval = 5) (h_number_of_classes : number_of_classes = 6) : 
                              ∃ x, x + (interval + x) + (2 * interval + x) + (3 * interval + x) 
                              + (4 * interval + x) + (5 * interval + x) = sum_classes ∧ x = 2 :=
by {
  use 2,
  sorry
}

end smallest_class_number_l820_820343


namespace border_area_is_198_l820_820918

-- We define the dimensions of the picture and the border width
def picture_height : ℝ := 12
def picture_width : ℝ := 15
def border_width : ℝ := 3

-- We compute the entire framed height and width
def framed_height : ℝ := picture_height + 2 * border_width
def framed_width : ℝ := picture_width + 2 * border_width

-- We compute the area of the picture and framed area
def picture_area : ℝ := picture_height * picture_width
def framed_area : ℝ := framed_height * framed_width

-- We compute the area of the border
def border_area : ℝ := framed_area - picture_area

-- Now we pose the theorem to prove the area of the border is 198 square inches
theorem border_area_is_198 : border_area = 198 := by
  sorry

end border_area_is_198_l820_820918


namespace max_value_of_a_l820_820714

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) → a ≤ 5 :=
by sorry

end max_value_of_a_l820_820714


namespace speed_of_student_B_l820_820489

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820489


namespace dot_product_triangle_l820_820002

variable (A B C : Type)
variable [EuclideanSpace A]
variables {a b c : A}
variable (x y z : ℝ)

-- given conditions
axiom AB_len : dist A B = 3
axiom BC_len : dist B C = 4
axiom CA_len : dist C A = 5

-- proof
theorem dot_product_triangle :
  (A - B)•(B - C) + (B - C)•(C - A) + (C - A)•(A - B) = -25 :=
sorry

end dot_product_triangle_l820_820002


namespace increase_output_with_assistant_l820_820319

theorem increase_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  let original_output := B / H,
      new_output := 1.8 * B / (0.9 * H),
      increase := (new_output - original_output) / original_output * 100
  in increase = 100 :=
by
  sorry

end increase_output_with_assistant_l820_820319


namespace positive_difference_of_sums_l820_820283

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820283


namespace student_B_speed_l820_820465

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820465


namespace speed_of_student_B_l820_820487

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820487


namespace parabola_shift_l820_820722

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l820_820722


namespace positive_difference_even_odd_sums_l820_820297

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820297


namespace min_inequality_leq_abc_l820_820057

theorem min_inequality_leq_abc
  (a b c : ℝ)
  (h_a : a ≥ 1)
  (h_b : b ≥ 1)
  (h_c : c ≥ 1) :
  min ( (10 * a^2 - 5 * a + 1) / (b^2 - 5 * b + 10),
        (10 * b^2 - 5 * b + 1) / (c^2 - 5 * c + 10),
        (10 * c^2 - 5 * c + 1) / (a^2 - 5 * a + 10)
      ) ≤ a * b * c := 
sorry

end min_inequality_leq_abc_l820_820057


namespace find_g_l820_820982

noncomputable def g (x : ℝ) : ℝ := -4 * x^5 + 4 * x^3 - 5 * x^2 + 2 * x + 4

theorem find_g (x : ℝ) :
  4 * x^5 + 3 * x^3 - 2 * x + g(x) = 7 * x^3 - 5 * x^2 + 4 :=
by
  sorry

end find_g_l820_820982


namespace number_of_terms_ap_l820_820134

variables (a d n : ℤ) 

def sum_of_first_thirteen_terms := (13 / 2) * (2 * a + 12 * d)
def sum_of_last_thirteen_terms := (13 / 2) * (2 * a + (2 * n - 14) * d)

def sum_excluding_first_three := ((n - 3) / 2) * (2 * a + (n - 4) * d)
def sum_excluding_last_three := ((n - 3) / 2) * (2 * a + (n - 1) * d)

theorem number_of_terms_ap (h1 : sum_of_first_thirteen_terms a d = (1 / 2) * sum_of_last_thirteen_terms a d)
  (h2 : sum_excluding_first_three a d / sum_excluding_last_three a d = 5 / 4) : n = 22 :=
sorry

end number_of_terms_ap_l820_820134


namespace bc_sum_eq_neg_one_l820_820044

variables {b1 b2 b3 b4 c1 c2 c3 c4 : ℝ}

/-- Given the equation for all real numbers x:
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4),
    prove that b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1. -/
theorem bc_sum_eq_neg_one :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4))
  → b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by {
  sorry,
}

end bc_sum_eq_neg_one_l820_820044


namespace shift_parabola_3_right_4_up_l820_820726

theorem shift_parabola_3_right_4_up (x : ℝ) : 
  let y := x^2 in
  (shifted_y : ℝ) = ((x - 3)^2 + 4) :=
begin
  sorry
end

end shift_parabola_3_right_4_up_l820_820726


namespace rice_grain_sorting_l820_820101

theorem rice_grain_sorting (n : ℕ) (h : \(\frac{n}{235} \leq 0.03\)) : n \leq 7 :=
sorry

end rice_grain_sorting_l820_820101


namespace count_positive_values_x_l820_820699

theorem count_positive_values_x :
  let valid_x := {x : ℕ | 25 ≤ x ∧ x ≤ 33} in
  set.finite valid_x ∧ set.count valid_x = 9 :=
by
  sorry

end count_positive_values_x_l820_820699


namespace positive_diff_even_odd_sums_l820_820174

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820174


namespace speed_of_student_B_l820_820498

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820498


namespace find_speed_of_B_l820_820409

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820409


namespace miguel_paint_area_l820_820067

def wall_height := 10
def wall_length := 15
def window_side := 3

theorem miguel_paint_area :
  (wall_height * wall_length) - (window_side * window_side) = 141 := 
by
  sorry

end miguel_paint_area_l820_820067


namespace positive_difference_of_sums_l820_820195

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820195


namespace existence_of_solution_l820_820986

theorem existence_of_solution
  (a : ℝ) :
  (∃ (b x y : ℝ), y = (8 / ((x - b)^2 + 1)) ∧ (x^2 + y^2 + 2 * a * (a + y - x) = 49)) ↔ 
  (a ∈ Icc (-15) 7) :=
sorry

end existence_of_solution_l820_820986


namespace smallest_norm_z_plus_2i_l820_820780

theorem smallest_norm_z_plus_2i (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.i)|) :
  ∃ z, complex.norm(z + 2 * complex.i) = 7 / 2 :=
sorry

end smallest_norm_z_plus_2i_l820_820780


namespace student_b_speed_l820_820369

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820369


namespace find_m_l820_820641

noncomputable def vector_a : ℝ × ℝ := (1, -3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 2)
noncomputable def vector_sum (m : ℝ) : ℝ × ℝ := (1 + m, -1)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) : dot_product vector_a (vector_sum m) = 0 → m = -4 :=
by
  sorry

end find_m_l820_820641


namespace student_B_speed_l820_820452

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820452


namespace ratio_of_models_l820_820098

noncomputable def kindergarten_models := 2
noncomputable def total_cost := 570
noncomputable def cost_per_model := 100
noncomputable def discount := 0.05

theorem ratio_of_models (e_models k_models : ℕ) :
  total_cost = cost_per_model * (kindergarten_models + e_models)
  - discount * cost_per_model * (kindergarten_models + e_models) ∧
  kindergarten_models + e_models > 5 →
  e_models / kindergarten_models = 2 :=
by
  sorry

end ratio_of_models_l820_820098


namespace total_cats_l820_820912

def num_white_cats : Nat := 2
def num_black_cats : Nat := 10
def num_gray_cats : Nat := 3

theorem total_cats : (num_white_cats + num_black_cats + num_gray_cats) = 15 :=
by
  sorry

end total_cats_l820_820912


namespace simplify_and_rationalize_denom_l820_820088

theorem simplify_and_rationalize_denom :
  (1 / (2 - (1 / (√5 + 2)))) = (4 + √5) / 11 :=
by
  sorry

end simplify_and_rationalize_denom_l820_820088


namespace number_of_common_points_l820_820575

noncomputable def count_intersections : Nat :=
  let eq1 := λ x y => (x - 2 * y + 3) * (3 * x + 2 * y - 5) = 0
  let eq2 := λ x y => (x + y - 3) * (x^2 - 5 * y + 6) = 0
  if ∃ (p : ℕ) (s: p = 4), p else 0

theorem number_of_common_points : 
  let eq1 := λ x y => (x - 2 * y + 3) * (3 * x + 2 * y - 5) = 0
  let eq2 := λ x y => (x + y - 3) * (x^2 - 5 * y + 6) = 0
  count_intersections = 4 := by
    sorry

end number_of_common_points_l820_820575


namespace speed_of_student_B_l820_820490

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820490


namespace inscribed_quadrilateral_l820_820607

-- Define the problem conditions and the goal
theorem inscribed_quadrilateral 
  (A B C D N : Point)
  (h_cyclic : IsCyclicQuadrilateral A B C D)
  (h_equal_sides : dist A B = dist B C)
  (h_right_angle : angle D N B = π / 2)
  : dist A D + dist N C = dist D N :=
sorry

end inscribed_quadrilateral_l820_820607


namespace find_speed_of_B_l820_820355

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820355


namespace positive_difference_even_odd_sums_l820_820246

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820246


namespace impossible_exchanges_l820_820736

theorem impossible_exchanges (n : ℕ) (h1 : n = 1970)
    (h2 : ∀ i : ℕ, i < n → (∃ exchanges : ℕ, exchanges = 10)) :
    ¬∃ residents : Fin n → ℕ → ℕ, 
        (∀ i : Fin n, ∑ j in Finset.range 10, (residents i j = 10) ∨ (residents i j = 2 * 5)) := sorry

end impossible_exchanges_l820_820736


namespace closed_pipe_length_l820_820697

def speed_of_sound : ℝ := 333
def fundamental_frequency : ℝ := 440

theorem closed_pipe_length :
  ∃ l : ℝ, l = 0.189 ∧ fundamental_frequency = speed_of_sound / (4 * l) :=
by
  sorry

end closed_pipe_length_l820_820697


namespace speed_of_student_B_l820_820495

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820495


namespace find_speed_of_B_l820_820351

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820351


namespace a_times_b_correct_l820_820039

def A := {x | ∃ y, y = sqrt (2 * x - x^2)}
def B := {y | ∃ x, y = 2^x ∧ x > 0}
def A_times_B := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem a_times_b_correct {A B : set ℝ} (H_A : A = {x | ∃ y, y = sqrt (2 * x - x^2)})
                      (H_B : B = {y | ∃ x, y = 2^x ∧ x > 0}) :
  A_times_B = {x | (0 : ℝ) ≤ x ∧ x ≤ 1 ∨ x > 2} := 
by sorry

end a_times_b_correct_l820_820039


namespace scientific_notation_suzhou_blood_donors_l820_820100

theorem scientific_notation_suzhou_blood_donors : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 124000 = a * 10^n ∧ a = 1.24 ∧ n = 5 :=
by
  sorry

end scientific_notation_suzhou_blood_donors_l820_820100


namespace positive_difference_sums_l820_820217

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820217


namespace positive_difference_even_odd_sum_l820_820279

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820279


namespace exams_left_to_grade_l820_820793

theorem exams_left_to_grade (E : ℕ) (p_m p_t : ℝ) (pm_nonneg : 0 ≤ p_m) (pm_le_one : p_m ≤ 1)
  (pt_nonneg : 0 ≤ p_t) (pt_le_one : p_t ≤ 1) (E_eq : E = 120) (pm_eq : p_m = 0.60) (pt_eq : p_t = 0.75) : 
  let graded_mon := p_m * E in
  let remaining_tue := E - graded_mon in
  let graded_tue := p_t * remaining_tue in
  let graded_mon_int := graded_mon.to_nat in
  let remaining_tue_int := remaining_tue.to_nat in
  let graded_tue_int := graded_tue.to_nat in
  let remaining_wed := remaining_tue_int - graded_tue_int in
  remaining_wed = 12 :=
by {
  sorry
}

end exams_left_to_grade_l820_820793


namespace annual_growth_rate_l820_820715

-- Define the monthly growth rate as a given real number p
variables (p : ℝ)

-- Define the annual growth rate as the required proof
theorem annual_growth_rate (p : ℝ) : ((1 + p) ^ 12) - 1 = annual_growth_rate :=
sorry

end annual_growth_rate_l820_820715


namespace positive_difference_of_sums_l820_820193

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820193


namespace student_B_speed_l820_820459

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820459


namespace intersection_M_N_l820_820062

noncomputable def M : Set ℝ := { x | x^2 = x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = {1} := by
  sorry

end intersection_M_N_l820_820062


namespace sqrt_same_type_as_sqrt_2_l820_820873

theorem sqrt_same_type_as_sqrt_2 (a b : ℝ) :
  ((sqrt a)^2 = 8) ↔ (sqrt 2) * (sqrt 2) = 2 * (sqrt 2) * (sqrt 2)  :=
sorry

end sqrt_same_type_as_sqrt_2_l820_820873


namespace magnitude_of_b_l820_820655

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820655


namespace acute_angle_probability_l820_820803

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end acute_angle_probability_l820_820803


namespace positive_difference_sums_l820_820216

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820216


namespace two_visitors_in_365_days_l820_820096

def max_visits : ℕ := 5
def nora_visits : ℕ := 6
def olivia_visits : ℕ := 7
def total_days : ℕ := 365

theorem two_visitors_in_365_days
  (h_max : ∀ n : ℕ, n ≠ 0 → (n * max_visits < total_days → n * max_visits < total_days))
  (h_nora : ∀ n : ℕ, n ≠ 0 → (n * nora_visits < total_days → n * nora_visits < total_days))
  (h_olivia : ∀ n : ℕ, n ≠ 0 → (n * olivia_visits < total_days → n * olivia_visits < total_days))
  : ∑ n in { m | m < total_days ∧ (m % max_visits = 0 ∨ m % nora_visits = 0 ∨ m % olivia_visits = 0) ∧ ¬ (m % max_visits = 0 ∧ m % nora_visits = 0 ∧ m % olivia_visits = 0 ) }, 1 = 27 := 
begin
  -- Proof goes here
  sorry
end

end two_visitors_in_365_days_l820_820096


namespace find_speed_of_B_l820_820415

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820415


namespace expression_value_l820_820541

-- Step c: Definitions based on conditions
def base1 : ℤ := -2
def exponent1 : ℕ := 4^2
def base2 : ℕ := 1
def exponent2 : ℕ := 3^3

-- The Lean statement for the problem
theorem expression_value :
  base1 ^ exponent1 + base2 ^ exponent2 = 65537 := by
  sorry

end expression_value_l820_820541


namespace magnitude_of_difference_l820_820617

noncomputable def vector_magnitude_subtraction (a b : ℝ^3) : ℝ :=
  ∥a - b∥

theorem magnitude_of_difference
  (a b : ℝ^3)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hangle : real.angle.rad a b = real.angle.pi / 3):
  vector_magnitude_subtraction a b = real.sqrt 7 := by
  sorry

end magnitude_of_difference_l820_820617


namespace point_P_path_length_l820_820878

/-- A rectangle PQRS in the plane with points P Q R S, where PQ = RS = 2 and QR = SP = 6. 
    The rectangle is rotated 90 degrees twice: first about point R and then 
    about the new position of point S after the first rotation. 
    The goal is to prove that the length of the path P travels is (3 + sqrt 10) * pi. -/
theorem point_P_path_length :
  ∀ (P Q R S : ℝ × ℝ), 
    dist P Q = 2 ∧ dist Q R = 6 ∧ dist R S = 2 ∧ dist S P = 6 →
    ∃ path_length : ℝ, path_length = (3 + Real.sqrt 10) * Real.pi :=
by
  sorry

end point_P_path_length_l820_820878


namespace max_regions_by_five_lines_l820_820077

theorem max_regions_by_five_lines : 
  ∀ (R : ℕ → ℕ), R 1 = 2 → R 2 = 4 → (∀ n, R (n + 1) = R n + (n + 1)) → R 5 = 16 :=
by
  intros R hR1 hR2 hRec
  sorry

end max_regions_by_five_lines_l820_820077


namespace positive_difference_even_odd_sums_l820_820299

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820299


namespace magnitude_b_eq_sqrt3_l820_820681

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820681


namespace multiply_and_multiply_l820_820993

theorem multiply_and_multiply :
  let a := 3.6 * 0.25 in
  let b := a * 0.4 in
  b = 0.36 :=
by
  sorry

end multiply_and_multiply_l820_820993


namespace fiona_reaches_pad_13_l820_820145

theorem fiona_reaches_pad_13 :
  let p_n : Nat × Nat → ℚ := λ n s =>
    if n = 13 ∧ s ∉ {4, 9}
    then 1
    else if s ∈ {4, 9} then 0
    else 0.5 * (p_n (n + 1, n + 1) + p_n (n + 2, n + 2))
  in p_n (0, 0) = 27 / 1024 := by
  sorry

end fiona_reaches_pad_13_l820_820145


namespace positive_difference_eq_250_l820_820161

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820161


namespace student_B_speed_l820_820455

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820455


namespace speed_of_student_B_l820_820488

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820488


namespace sum_of_right_column_equals_ab_l820_820892

theorem sum_of_right_column_equals_ab
  (a b : ℤ)
  (h1 : ∃ (n : ℤ) (f : ℤ → ℤ), (f a = 1 ∧ (∀ m : ℤ, m ≠ 1 → f m = m / if m % 2 = 0 then 2 else 2)))
  (h2 : ∃ (g : ℤ → ℤ), (g a = b ∧ ∀ m : ℤ, g (m / if m % 2 = 0 then 2 else 2) = 2 * g m)) :
  let sum_odd (f g : ℤ → ℤ) : ℤ := ∑ i in (finset.range n).filter (λ x, (f a) % 2 = 1), g (f x)
  in sum_odd f g = a * b :=
by 

end sum_of_right_column_equals_ab_l820_820892


namespace parabola_intercepts_l820_820967

def parabola_eqn : ℝ → ℝ := λ y, -3 * y^2 + 2 * y + 3

theorem parabola_intercepts :
  ((∀ y, parabola_eqn 0 = 3) ∧
   ∃ y1 y2 : ℝ, parabola_eqn y1 = 0 ∧ parabola_eqn y2 = 0 ∧ y1 ≠ y2) :=
by
  use [(-(1 / 3)) * (1 + Real.sqrt 10), (-(1 / 3)) * (1 - Real.sqrt 10)]
  simp [parabola_eqn]
  sorry

end parabola_intercepts_l820_820967


namespace tyrah_pencils_problem_l820_820859

noncomputable def pencils_proof : Prop :=
  ∀ (S Tyrah Tim : ℕ),
  (Tyrah = 6 * S) ∧ (Tim = 8 * S) ∧ (Tim = 16) → (Tyrah = 12)

theorem tyrah_pencils_problem : pencils_proof :=
by
  intros S Tyrah Tim
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  rw [h2, h3] at h1
  rw h3 at h1
  sorry

end tyrah_pencils_problem_l820_820859


namespace expand_expression_l820_820561

theorem expand_expression (y : ℝ) : (7 * y + 12) * 3 * y = 21 * y ^ 2 + 36 * y := by
  sorry

end expand_expression_l820_820561


namespace find_omega_l820_820110

def min_period_sine_function (ω : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = f x

theorem find_omega (ω : ℝ) (hω : ω > 0) (h : min_period_sine_function ω (λ x, sin (ω * x - π / 3))) : 
  ω = 2 :=
sorry

end find_omega_l820_820110


namespace find_candy_bars_per_week_l820_820760

-- Define the conditions
variables (x : ℕ)

-- Condition: Kim's dad buys Kim x candy bars each week
def candies_bought := 16 * x

-- Condition: Kim eats one candy bar every 4 weeks
def candies_eaten := 16 / 4

-- Condition: After 16 weeks, Kim has saved 28 candy bars
def saved_candies := 28

-- The theorem we want to prove
theorem find_candy_bars_per_week : (16 * x - (16 / 4) = 28) → x = 2 := by
  -- We will skip the actual proof for now.
  sorry

end find_candy_bars_per_week_l820_820760


namespace positive_difference_l820_820232

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820232


namespace positive_difference_even_odd_sums_l820_820300

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820300


namespace zoe_must_give_amount_to_equalize_costs_l820_820534

-- Given conditions
variables (X Y : ℝ) (h : X > Y)

-- Definition of the amount Zoe must give Amy
def amount_zoe_must_give (X Y : ℝ) : ℝ :=
  (X - Y) / 2

-- Statement of the proof problem
theorem zoe_must_give_amount_to_equalize_costs (X Y : ℝ) (h : X > Y) :
  amount_zoe_must_give X Y = (X - Y) / 2 :=
begin
  -- The proof is omitted 
  sorry
end

end zoe_must_give_amount_to_equalize_costs_l820_820534


namespace sequence_bound_l820_820925

theorem sequence_bound (n : ℕ) (h_n_pos : 0 < n) : 
  let a : ℕ → ℝ := λ k, if k = 0 then 1 / 2 else a (k-1) + (1 / n) * (a (k-1))^2 in
  1 - (1 / n) < a n ∧ a n < 1 :=
by
  let a : ℕ → ℝ := λ k, if k = 0 then 1 / 2 else a (k-1) + (1 / n) * (a (k-1))^2
  sorry

end sequence_bound_l820_820925


namespace find_speed_of_B_l820_820405

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820405


namespace find_speed_of_B_l820_820412

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820412


namespace light_glow_interval_l820_820115

theorem light_glow_interval :
  let t_start : ℕ := 1 * 3600 + 57 * 60 + 58   -- time in seconds from start of the day
  let t_end : ℕ := 3 * 3600 + 20 * 60 + 47     -- time in seconds from start of the day
  let total_time := t_end - t_start            -- total time in seconds
  let glows : ℚ := 236.61904761904762          -- number of glows in the period
  total_time / glows ≈ 21 :=
by
  let t_start := 1 * 3600 + 57 * 60 + 58   -- 1:57:58 am in seconds
  let t_end := 3 * 3600 + 20 * 60 + 47     -- 3:20:47 am in seconds
  let total_time := (t_end : ℚ) - t_start
  let glows := 236.61904761904762
  show total_time / glows = 21
  sorry

end light_glow_interval_l820_820115


namespace correct_answer_is_B_l820_820527

-- Define what it means to be a quadratic equation in one variable
def is_quadratic_in_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x ^ 2 + b * x + c = 0)

-- Conditions:
def eqA (x : ℝ) : Prop := 2 * x + 1 = 0
def eqB (x : ℝ) : Prop := x ^ 2 + 1 = 0
def eqC (x y : ℝ) : Prop := y ^ 2 + x = 1
def eqD (x : ℝ) : Prop := 1 / x + x ^ 2 = 1

-- Theorem statement: Prove which equation is a quadratic equation in one variable
theorem correct_answer_is_B : is_quadratic_in_one_variable eqB :=
sorry  -- Proof is not required as per the instructions

end correct_answer_is_B_l820_820527


namespace num_possible_values_of_r_l820_820934

theorem num_possible_values_of_r :
  ∃ (r : ℝ → ℝ), (∀ i : (ℕ%), (∃ P_i : Point,
    (P_i ∈ ω i ∧ P_i ∈ ω (i + 1)) ∧
    is_further_from_hexagon_centre P_i ∧
    ∃ Q_i : Point, (Q_i ∈ ω i ∧ Q_i, P_i, Q (i + 1) are_collinear) ∧
    ∃ hexagon : Hexagon, (is_regular hexagon)),
  count_distinct_r_values r = 5 := sorry

end num_possible_values_of_r_l820_820934


namespace speed_of_student_B_l820_820431

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820431


namespace positive_difference_even_odd_l820_820203

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820203


namespace proposition_1_correct_proposition_2_incorrect_proposition_3_correct_proposition_4_incorrect_l820_820108

noncomputable def a_value (area : ℝ) (c : ℝ) (A : ℝ) : ℝ :=
  real.sqrt (c ^ 2 * (1 - real.cos A ^ 2))

theorem proposition_1_correct (area : ℝ) (c : ℝ) (A : ℝ) (a : ℝ) :
  area = real.sqrt 3 / 2 ∧ c = 2 ∧ A = real.pi / 3 → a_value area c A = real.sqrt 3 := 
sorry

noncomputable def common_difference (a1 : ℝ) (a3 : ℝ) (a4 : ℝ) : ℝ := 
  if a1 + 2 * (a3 - a1 / 2) = a3 then 0 else -(1 / 2)

theorem proposition_2_incorrect (a1 : ℝ) (a3 : ℝ) (a4 : ℝ) :
  a1 = 2 ∧ (2 + 2 * (-1/2))^2 = 4 + 6 * (-1/2) ∧ a1 ≠ 4 → common_difference 2 a3 a4 ≠ -(1/2) :=
sorry

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  a + b = 1 → min (2 / a + 3 / b) := 5 + 2 * real.sqrt 6

theorem proposition_3_correct (a : ℝ) (b : ℝ) :
  0 < a ∧ 0 < b ∧ a + b = 1 → min_value a b = 5 + 2 * real.sqrt 6 :=
sorry

def acute_triangle (a b c : ℝ) := 
  ∃A B C, A < B + C → (trig.sin A)^2 < (trig.sin B)^2 + (trig.sin C)^2

theorem proposition_4_incorrect (a b c : ℝ) :
  ∀A B C, (trig.sin A)^2 < (trig.sin B)^2 + (trig.sin C)^2 → ¬ acute_triangle a b c :=
sorry

end proposition_1_correct_proposition_2_incorrect_proposition_3_correct_proposition_4_incorrect_l820_820108


namespace jacket_restore_percentage_l820_820321

variable (P : ℝ) (r : ℝ := 0.25)
noncomputable def final_price := (P * (1 - r)) * (1 - r)

theorem jacket_restore_percentage (P : ℝ) (hP : 0 < P) :
  final_price P (r:= r)  = P * (1 - r) * (1 - r) →
  (P - final_price P (r:= r)) / final_price P (r:= r) ≈ 0.7778 :=
by
  sorry

end jacket_restore_percentage_l820_820321


namespace range_of_a_point_Q_symmetric_l820_820634

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sin x ^ 2

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc a (pi / 16), f' x ≥ 0) → (a ∈ set.Ico (-pi / 8) (pi / 16)) :=
by
  admit

theorem point_Q_symmetric (x1 y1 : ℝ) :
  (x1 ∈ set.Icc (-pi / 4) (pi / 4)) →
  is_symmetric_about (f(x1)) (x1, y1) →
  (x1, y1) = (pi / 8, 1 / 2) :=
by
  admit

end range_of_a_point_Q_symmetric_l820_820634


namespace product_ab_eq_13_l820_820106

noncomputable theory
open Complex

def u : ℂ := -2 + 3 * I
def v : ℂ := 1 + I

theorem product_ab_eq_13 (a b : ℂ)
  (h : ∀ z : ℂ, (z = u ∨ z = v) → a * z + b * conj z = 10) :
  a * b = 13 :=
sorry

end product_ab_eq_13_l820_820106


namespace student_B_speed_l820_820509

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820509


namespace pythagorean_triangle_exists_l820_820823

theorem pythagorean_triangle_exists (a : ℤ) (h : a ≥ 5) : 
  ∃ (b c : ℤ), c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 :=
by {
  sorry
}

end pythagorean_triangle_exists_l820_820823


namespace triangle_cosA_cosB_max_triangle_sinA_sinB_min_triangle_tanA_tanB_range_triangle_sin_squared_sum_const_l820_820023

theorem triangle_cosA_cosB_max {A B C : ℝ} (h1 : A + B + C = 180) (h2 : tan (C / 2) = sin (A + B)) : 
  (cos A * cos B) ≤ 1 / 2 :=
sorry

theorem triangle_sinA_sinB_min {A B C : ℝ} (h1 : A + B + C = 180) (h2 : tan (C / 2) = sin (A + B)) : 
  1 ≤ (sin A + sin B) :=
sorry

theorem triangle_tanA_tanB_range {A B C : ℝ} (h1 : A + B + C = 180) (h2 : tan (C / 2) = sin (A + B)) : 
  ∃ x, 2 ≤ x ∧ ∀ y, y = (tan A + tan B) → y ≥ 2 :=
sorry

theorem triangle_sin_squared_sum_const {A B C : ℝ} (h1 : A + B + C = 180) (h2 : tan (C / 2) = sin (A + B)) : 
  (sin A)^2 + (sin B)^2 + (sin C)^2 = 2 :=
sorry

end triangle_cosA_cosB_max_triangle_sinA_sinB_min_triangle_tanA_tanB_range_triangle_sin_squared_sum_const_l820_820023


namespace find_radius_second_circle_l820_820941

variables (a α R r : ℝ)
-- Conditions
def is_isosceles_triangle (a α : ℝ) : Prop := a > 0 ∧ 0 < α ∧ α < π / 2
def is_first_circle_inscribed_triangle (a α R : ℝ) : Prop := R = a / (4 * sin α * cos α)
def second_circle_tangent_conditions (R r α : ℝ) : Prop := r = (2 * R - r) * cos α

-- Final statement
theorem find_radius_second_circle
  (ha : is_isosceles_triangle a α)
  (hc : is_first_circle_inscribed_triangle a α R)
  (ht : second_circle_tangent_conditions R r α) :
  r = a / (2 * sin α * (1 + cos α)) :=
  sorry

end find_radius_second_circle_l820_820941


namespace cost_equivalence_l820_820949

theorem cost_equivalence (b a p : ℕ) (h1 : 4 * b = 3 * a) (h2 : 9 * a = 6 * p) : 24 * b = 12 * p :=
  sorry

end cost_equivalence_l820_820949


namespace positive_difference_even_odd_sums_l820_820298

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820298


namespace largest_inclination_angle_l820_820528

-- Define the slopes and inclination angles
def slope (m : ℝ) : ℝ := m

def inclination_angle (m : ℝ) : ℝ :=
  if h : m ≠ 0 then real.atan m * 180 / real.pi else if m > 0 then 0 else 180

-- Definitions of the lines
def lineA : ℝ := slope (-1)
def lineB : ℝ := slope (1)
def lineC : ℝ := slope (2)
def lineD : ℝ := real.pi / 2 -- vertical line

-- Define inclination angles for each line
def angleA : ℝ := 135
def angleB : ℝ := 45
def angleC : ℝ := if 60 < real.pi / 2 then (60 + 90)/2 else 0 -- between 60 and 90 degrees
def angleD : ℝ := 90

-- Prove that angleA is the largest among angleA, angleB, angleC, and angleD
theorem largest_inclination_angle :
  angleA > angleB ∧ angleA > angleC ∧ angleA > angleD := by
  sorry

end largest_inclination_angle_l820_820528


namespace parabola_transformation_l820_820716

theorem parabola_transformation :
  ∀ (x y : ℝ), (y = x^2) →
  let shiftedRight := (x - 3)
  let shiftedUp := (shiftedRight)^2 + 4
  y = shiftedUp :=
by
  intros x y,
  intro hyp,
  let shiftedRight := (x - 3),
  let shiftedUp := (shiftedRight)^2 + 4,
  have eq_shifted : y = shiftedRight^2 := by sorry,
  have eq_final : y = shiftedUp := by sorry,
  exact eq_final
  
end parabola_transformation_l820_820716


namespace most_axes_of_symmetry_circle_l820_820533

/-- Among the following symmetrical figures, prove that the circle has the most axes of symmetry:
1. An equilateral triangle has 3 axes of symmetry.
2. A rhombus has 2 axes of symmetry.
3. A square has 4 axes of symmetry.
4. A circle has an infinite number of axes of symmetry. -/
theorem most_axes_of_symmetry_circle : 
  ∀ (n_triangle n_rhombus n_square n_circle : ℕ), n_triangle = 3 → n_rhombus = 2 → n_square = 4 → n_circle = ∞ → 
  n_circle > n_triangle ∧ n_circle > n_rhombus ∧ n_circle > n_square := 
sorry

end most_axes_of_symmetry_circle_l820_820533


namespace positive_difference_even_odd_sums_l820_820249

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820249


namespace smallest_seven_star_number_gt_2000_l820_820587

def is_divisor (a b : ℕ) := b % a = 0

def seven_star_number (N : ℕ) : Prop :=
  (1, 2, 3, 4, 5, 6, 7, 8, 9).count (λ n, is_divisor n N) ≥ 7

theorem smallest_seven_star_number_gt_2000 : ∃ N, N > 2000 ∧ seven_star_number N ∧
  ∀ M, M > 2000 ∧ seven_star_number M → N ≤ M :=
exists.intro 2016 (by {
  have div_2016: seven_star_number 2016 := by sorry,
  have minimal_2016: ∀ M, M > 2000 ∧ seven_star_number M → 2016 ≤ M := by sorry,
  exact ⟨2016, (by linarith), div_2016, minimal_2016⟩
})

end smallest_seven_star_number_gt_2000_l820_820587


namespace shift_parabola_3_right_4_up_l820_820725

theorem shift_parabola_3_right_4_up (x : ℝ) : 
  let y := x^2 in
  (shifted_y : ℝ) = ((x - 3)^2 + 4) :=
begin
  sorry
end

end shift_parabola_3_right_4_up_l820_820725


namespace largest_k_consecutive_sum_l820_820990

theorem largest_k_consecutive_sum (k n : ℕ) :
  (3^14 = (∑ i in range k, (n + 1) + i)) ↔
  ((k * (2 * n + k + 1) = 2 * 3^14) ∧ 
  (k ∣ 2 * 3^14) ∧ 
  (k < nat.sqrt (2 * 3^14)) ∧ 
  (n ≥ 0) ∧
  (k = 729)) :=
  sorry

end largest_k_consecutive_sum_l820_820990


namespace base_identification_l820_820308

theorem base_identification :
  ∃ (b : ℕ) (A B : ℕ), A ≠ B ∧ b = 8 ∧ 
  b^3 ≤ 777 ∧ 777 < b^4 ∧ 
  let d0 := 777 / b^3 in
  let rem1 := 777 % b^3 in
  let d1 := rem1 / b^2 in
  let rem2 := rem1 % b^2 in
  let d2 := rem2 / b in
  let d3 := rem2 % b in
  (A = d0 ∧ B = d1 ∧ A = d2 ∧ B = d3) :=
by
  sorry

end base_identification_l820_820308


namespace quadratic_form_l820_820128

theorem quadratic_form {a b c : ℝ} : 
  (∀ x : ℝ, 15 * x^2 + 90 * x + 405 = a * (x + b)^2 + c) → 
  a = 15 ∧ b = 3 ∧ c = 270 → 
  a + b + c = 288 :=
by
  intro h
  intro habc
  cases habc with ha hb
  cases hb with hb hc
  rw [ha, hb, hc]
  norm_num

end quadratic_form_l820_820128


namespace find_y_condition_l820_820565

theorem find_y_condition (y : ℝ) (h : log 10 (5 * y) = 3) : y = 200 := 
sorry

end find_y_condition_l820_820565


namespace shaded_area_inequality_l820_820972

theorem shaded_area_inequality 
    (A : ℝ) -- All three triangles have the same total area, A.
    {a1 a2 a3 : ℝ} -- a1, a2, a3 are the shaded areas of Triangle I, II, and III respectively.
    (h1 : a1 = A / 6) 
    (h2 : a2 = A / 2) 
    (h3 : a3 = (2 * A) / 3) : 
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 :=
by
  -- Proof steps would go here, but they are not required as per the instructions
  sorry

end shaded_area_inequality_l820_820972


namespace c_n_eq_square_l820_820326

noncomputable def a_n (n : ℕ) : ℕ :=
  if √n * √n < n then √n + 1 else √n

noncomputable def b_n (n : ℕ) : ℕ :=
  n + a_n n

noncomputable def c_n (n : ℕ) : ℕ :=
  n * n

theorem c_n_eq_square (n : ℕ) (h_n_pos : 0 < n) :
  c_n n = n * n := by
  sorry

end c_n_eq_square_l820_820326


namespace clock_angle_acute_probability_l820_820800

noncomputable def probability_acute_angle : ℚ := 1 / 2

theorem clock_angle_acute_probability :
  ∀ (hour minute : ℕ), (hour >= 0 ∧ hour < 12) →
  (minute >= 0 ∧ minute < 60) →
  (let angle := min (60 * hour - 11 * minute) (720 - (60 * hour - 11 * minute)) in angle < 90 ↔ probability_acute_angle = 1 / 2) :=
sorry

end clock_angle_acute_probability_l820_820800


namespace student_b_speed_l820_820375

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820375


namespace area_of_polygon_l820_820959

-- Define the structure of the equilateral triangular grid
structure TriangularGrid :=
  (points_on_side : ℕ)
  (points : finset (ℕ × ℕ))
  (total_points : points.card = (points_on_side * (points_on_side + 1)) / 2)

-- Define the polygon with specific properties
structure Polygon (G : TriangularGrid) :=
  (vertices : finset (ℕ × ℕ))
  (closed : (∃ v, v ∈ vertices) → (∀ v, v ∈ vertices ↔ v ∈ G.points . subst G))
  (non_selfintersecting : true) -- Placeholder, needs formal definition
  (uses_all_points : vertices = G.points)

-- Given grid G and polygon S, prove the area of S
def polygon_area (G : TriangularGrid) (S : Polygon G) : ℝ :=
  52 * Real.sqrt 3

theorem area_of_polygon (G : TriangularGrid) (S : Polygon G) : 
  polygon_area G S = 52 * Real.sqrt 3 := sorry

end area_of_polygon_l820_820959


namespace student_b_speed_l820_820440

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820440


namespace bc_sum_eq_neg_one_l820_820045

variables {b1 b2 b3 b4 c1 c2 c3 c4 : ℝ}

/-- Given the equation for all real numbers x:
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4),
    prove that b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1. -/
theorem bc_sum_eq_neg_one :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4))
  → b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by {
  sorry,
}

end bc_sum_eq_neg_one_l820_820045


namespace bicycle_speed_B_l820_820382

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820382


namespace derivative_ln_sqrt_l820_820598

theorem derivative_ln_sqrt (x : ℝ) : 
  let y := ln (1 / (sqrt (1 + x^2))) 
  in deriv (fun x => ln (1 / (sqrt (1 + x^2)))) x = -x / (1 + x^2) :=
by
  let y := ln (1 / (sqrt (1 + x^2)))
  sorry

end derivative_ln_sqrt_l820_820598


namespace concentric_circles_common_tangents_l820_820153

theorem concentric_circles_common_tangents (r R : ℝ) (h : r < R) :
  ∀ t, t ∈ {1, 2, 3, 4} → false :=
by
  sorry

end concentric_circles_common_tangents_l820_820153


namespace magnitude_b_sqrt_3_l820_820661

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820661


namespace magnitude_of_b_l820_820657

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820657


namespace probability_tile_in_PAIR_l820_820977

theorem probability_tile_in_PAIR :
  let tiles := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
  let pair_letters := ['P', 'A', 'I', 'R']
  let matching_counts := (sum ([1, 1, 2, 1] : List ℕ))
  matching_counts.fst = 5
  let total_tiles := 12
  (matching_counts.toRational / total_tiles.toRational) = (5 / 12) :=
by sorry

end probability_tile_in_PAIR_l820_820977


namespace problem_statement_l820_820632

noncomputable theory

def f (A ω φ x : ℝ) := A * sin (ω * x + φ)
variable (A ω φ : ℝ)
variables (H_A : A > 0) (H_ω : ω > 0) (H_φ : 0 < φ ∧ φ < π / 2)

def is_symmetry_axes_distance_correct : Prop := (π / ω) = (π / 2)
def can_shift_cos_sin_graph : Prop := ∃ y, (λ x, cos (2 * x) - sqrt 3 * sin (2 * x)) = (λ x, y) ∧ (λ x, y) = (λ x, 2 * sin (2 * (x - π / 4) + φ))
def is_monotonically_increasing : Prop := ∀ x, -π / 12 < x ∧ x < π / 6 → deriv (f A ω φ) x > 0
def is_symmetric : Prop := ∀ x, f A ω φ (π / 3 + x) + f A ω φ (π / 3 - x) = 0

theorem problem_statement :
  (is_symmetry_axes_distance_correct A ω φ H_A H_ω H_φ) →
  (can_shift_cos_sin_graph φ) →
  (¬ is_monotonically_increasing A ω φ H_A H_ω H_φ) →
  (is_symmetric A ω φ):
  (¬ is_monotonically_increasing A ω φ H_A H_ω H_φ) :=
begin
  sorry
end

end problem_statement_l820_820632


namespace positive_difference_sums_l820_820213

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820213


namespace parabola_transformation_l820_820717

theorem parabola_transformation :
  ∀ (x y : ℝ), (y = x^2) →
  let shiftedRight := (x - 3)
  let shiftedUp := (shiftedRight)^2 + 4
  y = shiftedUp :=
by
  intros x y,
  intro hyp,
  let shiftedRight := (x - 3),
  let shiftedUp := (shiftedRight)^2 + 4,
  have eq_shifted : y = shiftedRight^2 := by sorry,
  have eq_final : y = shiftedUp := by sorry,
  exact eq_final
  
end parabola_transformation_l820_820717


namespace positive_difference_of_sums_l820_820196

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820196


namespace triangle_property_l820_820781

variables {A B C D E F M N O P Q : Type*}
variables [LinearOrder A] [LinearOrder B] [LinearOrder C]
variables [LinearOrder D] [LinearOrder E] [LinearOrder F]
variables [LinearOrder M] [LinearOrder N] [LinearOrder O]
variables [LinearOrder P] [LinearOrder Q]

def midpoint (x y z : Type*) [LinearOrder x] [LinearOrder y] [LinearOrder z] : Prop :=
  sorry

def angle_bisector (α β γ : Type*) [LinearOrder α] [LinearOrder β] [LinearOrder γ] : Prop :=
  sorry

def intersects (x y z : Type*) [LinearOrder x] [LinearOrder y] [LinearOrder z] : Prop :=
  sorry

def meets (x y z w : Type*) [LinearOrder x] [LinearOrder y] [LinearOrder z] [LinearOrder w] : Prop :=
  sorry

theorem triangle_property
  (hABC : true)
  (hD : midpoint A B D)
  (hE : midpoint B C E)
  (hF : midpoint C A F)
  (hM : angle_bisector B D C ∧ meets D B C M)
  (hN : angle_bisector A D C ∧ meets D A C N)
  (hO_int : intersects M N O ∧ intersects O C D)
  (hP : meets E O A C P)
  (hQ : meets F O B C Q)
  :
  CD = PQ :=
sorry

end triangle_property_l820_820781


namespace find_speed_B_l820_820404

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820404


namespace positive_difference_sums_even_odd_l820_820239

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820239


namespace Mr_T_Estate_Value_l820_820068

theorem Mr_T_Estate_Value (E : ℝ)
  (x s : ℝ)
  (w = 3 * s)
  (g = 600)
  (sum_wg : w + g = E / 4)
  (sum_ds : 8 * x + s = 3 / 4 * E)
  (estate_sum : w + g + 8 * x + s = E) :
  E = 2400 :=
by
  -- Proof is omitted
  sorry

end Mr_T_Estate_Value_l820_820068


namespace positive_difference_even_odd_l820_820199

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820199


namespace xy_square_sum_l820_820821

theorem xy_square_sum (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 132) : x^2 + y^2 = 1336 :=
by
  sorry

end xy_square_sum_l820_820821


namespace speed_of_student_B_l820_820477

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820477


namespace jogger_ahead_of_train_l820_820344

theorem jogger_ahead_of_train (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (time_to_pass : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 100) 
  (h4 : time_to_pass = 34) : 
  ∃ d : ℝ, d = 240 :=
by
  sorry

end jogger_ahead_of_train_l820_820344


namespace identity_proof_l820_820156

theorem identity_proof (A B C A1 B1 C1 : ℝ) :
  (A^2 + B^2 + C^2) * (A1^2 + B1^2 + C1^2) - (A * A1 + B * B1 + C * C1)^2 =
    (A * B1 + A1 * B)^2 + (A * C1 + A1 * C)^2 + (B * C1 + B1 * C)^2 :=
by
  sorry

end identity_proof_l820_820156


namespace player_win_condition_l820_820950

def even_part {n : ℕ} (k : ℕ) : Prop := ∃ i, n = 2^k * i ∧ i % 2 = 1

theorem player_win_condition (r g k l : ℕ) (i j : ℕ)
  (hr : even_part r k) (hg : even_part g l)
  (odd_i : i % 2 = 1) (odd_j : j % 2 = 1) 
  (eq_k_l : k = l ∨ k ≠ l) :
  (k = l → ∃ strategy, ∀ moves, strategy wins_second_player) ∧
  (k ≠ l → ∃ strategy, ∀ moves, strategy wins_first_player) :=
sorry

end player_win_condition_l820_820950


namespace magnitude_of_b_l820_820649

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820649


namespace smallest_unique_sum_l820_820137

-- Define the natural numbers a and b such that their sum is uniquely represented with letters.
def is_unique_representation (a b : ℕ) : Prop :=
  ∀ (A B V : char), (A ≠ B ∧ A ≠ V ∧ B ≠ V) →
  (a + b) = 10

-- The theorem to prove the smallest sum.
theorem smallest_unique_sum (a b : ℕ) (h : is_unique_representation a b) : (a + b) = 10 :=
sorry

end smallest_unique_sum_l820_820137


namespace total_spent_on_toys_l820_820069

theorem total_spent_on_toys : 
  let cost_yoyo := 24
  let cost_whistle := 14
  in cost_yoyo + cost_whistle = 38 := 
by
  let cost_yoyo := 24
  let cost_whistle := 14
  sorry

end total_spent_on_toys_l820_820069


namespace tea_consumption_l820_820121

variable (h t : ℝ)
variable (k : ℝ)

theorem tea_consumption :
  (t * h = k) →
  (8 * 3 = k) →
  (h = 5 → t = 4.8) ∧ (h = 10 → t = 2.4) :=
by
  intros h_t_eq_k wednesday_eq_k h_eq
  split
  · intro h_is_5
    rw [←h_t_eq_k, ←wednesday_eq_k]
    have h_k_24 : k = 24 := rfl
    rw [h_k_24] at h_t_eq_k
    have t_5 : t = 24 / 5 := by ring_nf
    exact (eq_div_iff (ne_of_gt (by norm_num : (5:ℝ) > 0))).mpr t_5
  · intro h_is_10
    rw [←h_t_eq_k, ←wednesday_eq_k]
    have h_k_24 : k = 24 := rfl
    rw [h_k_24] at h_t_eq_k
    have t_10 : t = 24 / 10 := by ring_nf
    exact (eq_div_iff (ne_of_gt (by norm_num : (10:ℝ) > 0))).mpr t_10

end tea_consumption_l820_820121


namespace ratio_cityX_to_cityZ_l820_820884

variable (Population : Type)
variable [Inhabited Population] [HasSmul ℕ Population] [MulAction ℕ Population]
variable (pZ pY pX : Population)

-- Conditions
axiom cityY_twice_cityZ : pY = 2 • pZ
axiom cityX_thrice_cityY : pX = 3 • pY

-- Theorem to prove
theorem ratio_cityX_to_cityZ (hY : pY = 2 • pZ) (hX : pX = 3 • pY) : 
  pX = 6 • pZ := by
  sorry

end ratio_cityX_to_cityZ_l820_820884


namespace math_problem_l820_820772

noncomputable def f : ℝ → ℝ
| 1 := 3
| 2 := 13
| 3 := 8
| 5 := 1
| 8 := 0
| 13 := 5
| _ := 0

noncomputable def g : ℝ → ℝ
| 1 := 2
| 2 := 8
| 3 := 13
| 5 := 3
| 8 := 5
| 13 := 1
| _ := 0

noncomputable def f_inv : ℝ → ℝ
| 3 := 1
| 13 := 2
| 8 := 3
| 1 := 5
| 0 := 8
| 5 := 13
| _ := 0

noncomputable def g_inv : ℝ → ℝ
| 2 := 1
| 8 := 2
| 13 := 3
| 3 := 5
| 5 := 8
| 1 := 13
| _ := 0

theorem math_problem : 
  g (f_inv ((g_inv 5 + f_inv 13) / f_inv 1)) = 13 :=
by {
  sorry -- Proof to be completed
}

end math_problem_l820_820772


namespace required_draws_for_pairs_all_colors_l820_820004

/-- 
  There are hats and gloves of 5 different colors: red (41 items), green (23 items), orange (11 items), blue (15 items), yellow (10 items).
  This function represents the worst-case number of draws required to ensure obtaining a pair (one hat and one glove) of each color.
-/
def worst_case_draws (red green orange blue yellow : ℕ) : ℕ :=
  (red + 1) + (green + 1) + (orange + 1) + (blue + 1) + (yellow + 1)

theorem required_draws_for_pairs_all_colors : worst_case_draws 41 23 11 15 10 = 105 :=
by
  calc
    worst_case_draws 41 23 11 15 10
        = (41 + 1) + (23 + 1) + (11 + 1) + (15 + 1) + (10 + 1) : by rfl
    ... = 42 + 24 + 12 + 16 + 11                           : by rfl
    ... = 105                                              : by simp


end required_draws_for_pairs_all_colors_l820_820004


namespace frog_final_position_probability_l820_820904

noncomputable def prob_frog_within_2_meters : ℝ :=
  let μ := measure_theory.measure.uniform_measure (sphere (0 : euclidean_space ℝ (fin 3)) 1) 
  ∫ x in ball (0 : euclidean_space ℝ (fin 3)) 2, 
    (∫ u in sphere (0 : euclidean_space ℝ (fin 3)) 1, 
      ∫ v in sphere (0 : euclidean_space ℝ (fin 3)) 2, 
        ∫ w in sphere (0 : euclidean_space ℝ (fin 3)) 3, 
          if dist ((u + v + w) : euclidean_space ℝ (fin 3)) x ≤ 2 
          then 1 
          else 0) 
  sorry

theorem frog_final_position_probability : 
  prob_frog_within_2_meters = 1/5 := 
sorry

end frog_final_position_probability_l820_820904


namespace area_triang_ABC_l820_820730

-- Define the basic structure and conditions of the problem
structure RightTriangle (A B C H M : Type) [MetricSpace A] :=
  (angle_right_C : angle C == pi / 2)
  (altitude_CH : H ∈ line_through C B ∧ H ∈ line_through C M)
  (median_CM : M ∈ line_through C A ∧ M ∈ midpoint A B)
  (trisect_angle_C : angle A C M == angle M C B ∧ angle M C H == pi / 6)

-- Define the given constants
constant (A B C H M : Type) [MetricSpace] [RightTriangle A B C H M]
constant (area_triang_CHM : ℝ)
constant (K : ℝ) : area_triang_CHM = K

-- The theorem to be proved
theorem area_triang_ABC : 4 * K = area (line_segment A C B) :=
by
  sorry

end area_triang_ABC_l820_820730


namespace compute_b_c_sum_l820_820048

def polynomial_decomposition (Q : ℝ[X]) (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ) : Prop :=
  ∀ x : ℝ, Q.eval x = (x^2 + (b1*x) + c1) * (x^2 + (b2*x) + c2) * (x^2 + (b3*x) + c3) * (x^2 + (b4*x) + c4)

theorem compute_b_c_sum (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : polynomial_decomposition (polynomial.mk [1, -1, 1, -1, 1, -1, 1, -1, 1]) b1 b2 b3 b4 c1 c2 c3 c4)
  (c1_eq : c1 = 1) (c2_eq : c2 = 1) (c3_eq : c3 = 1) (c4_eq : c4 = 1) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 := 
sorry

end compute_b_c_sum_l820_820048


namespace value_range_of_f_l820_820847

noncomputable def f : ℝ → ℝ :=
λ x, if (0 < x ∧ x ≤ 3) then 2*x - x^2
     else if (-2 ≤ x ∧ x ≤ 0) then x^2 + 6*x
     else 0

theorem value_range_of_f : set.range f = set.Icc (-8 : ℝ) 1 := sorry

end value_range_of_f_l820_820847


namespace student_b_speed_l820_820438

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820438


namespace norm_b_eq_sqrt_3_l820_820668

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820668


namespace find_speed_of_B_l820_820354

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820354


namespace total_purchase_cost_l820_820523

variable (kg_nuts : ℝ) (kg_dried_fruits : ℝ)
variable (cost_per_kg_nuts : ℝ) (cost_per_kg_dried_fruits : ℝ)

-- Define the quantities
def cost_nuts := kg_nuts * cost_per_kg_nuts
def cost_dried_fruits := kg_dried_fruits * cost_per_kg_dried_fruits

-- The total cost can be expressed as follows
def total_cost := cost_nuts + cost_dried_fruits

theorem total_purchase_cost (h1 : kg_nuts = 3) (h2 : kg_dried_fruits = 2.5)
  (h3 : cost_per_kg_nuts = 12) (h4 : cost_per_kg_dried_fruits = 8) :
  total_cost kg_nuts kg_dried_fruits cost_per_kg_nuts cost_per_kg_dried_fruits = 56 := by
  sorry

end total_purchase_cost_l820_820523


namespace find_speed_of_B_l820_820410

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820410


namespace perfect_square_transformation_l820_820313

theorem perfect_square_transformation (a : ℤ) :
  (∃ x y : ℤ, x^2 + a = y^2) ↔ 
  ∃ α β : ℤ, α * β = a ∧ (α % 2 = β % 2) ∧ 
  ∃ x y : ℤ, x = (β - α) / 2 ∧ y = (β + α) / 2 :=
by
  sorry

end perfect_square_transformation_l820_820313


namespace find_speed_of_B_l820_820414

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820414


namespace find_speed_of_B_l820_820416

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820416


namespace cousin_problems_eq_180_l820_820795

variables (p t : ℕ) (h_p : p ≥ 15) (h_pos_p : 0 < p) (h_pos_t : 0 < t)

def my_rate := p
def my_time := t
def my_problems := my_rate * my_time

def cousin_rate := 3 * p - 5
def cousin_time := (t + 3) / 2
def cousin_problems := cousin_rate * cousin_time

theorem cousin_problems_eq_180 : cousin_problems p t = 180 :=
by
  have h_cousin_problems_simp : cousin_problems p t = (3 * p - 5) * (t + 3) / 2 := rfl
  sorry

end cousin_problems_eq_180_l820_820795


namespace student_b_speed_l820_820365

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820365


namespace P_plus_Q_is_26_l820_820056

theorem P_plus_Q_is_26 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3))) : 
  P + Q = 26 :=
sorry

end P_plus_Q_is_26_l820_820056


namespace speed_of_student_B_l820_820496

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820496


namespace positive_difference_of_sums_l820_820188

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820188


namespace student_B_speed_l820_820513

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820513


namespace part1_part2_l820_820586

-- Part (1) Lean statement
theorem part1 (n : ℕ) (hn : n ≥ 2) : 
  (finset.prod (finset.range (n-1)) (λ k, abs (cos (k * real.pi / n))) = 
  (1/2)^n * (1 - (-1)^n)) :=
by
  sorry

-- Part (2) Lean statement
theorem part2 (n : ℕ) (hn : n ≥ 2) : 
  (finset.sum (finset.range (n-1)) (λ k, sin (k * real.pi / n)) = 
   n * (1/2)^(n-1)) :=
by
  sorry

end part1_part2_l820_820586


namespace probability_more_boys_or_girls_l820_820789

theorem probability_more_boys_or_girls (n : ℕ) (h_n : n = 12) :
  let p := 1 / 2 in
  (let total_outcomes := 2 ^ n,
       equal_boys_girls := Nat.choose n (n / 2)
   in 1 - (equal_boys_girls / total_outcomes)) = 793 / 1024 := by
  sorry

end probability_more_boys_or_girls_l820_820789


namespace expansion_coeff_sum_l820_820104

theorem expansion_coeff_sum :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ, 
    (2*x - 1)^10 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10)
  → (1 - 20 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 1 → a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 20) :=
by
  sorry

end expansion_coeff_sum_l820_820104


namespace student_b_speed_l820_820374

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820374


namespace positive_diff_even_odd_sums_l820_820175

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820175


namespace students_not_in_biology_l820_820707

theorem students_not_in_biology (S : ℕ) (f : ℚ) (hS : S = 840) (hf : f = 0.35) :
  S - (f * S) = 546 :=
by
  sorry

end students_not_in_biology_l820_820707


namespace order_of_numbers_l820_820553

variable (a b c : ℝ)
variable (h₁ : a = (1 / 2) ^ (1 / 3))
variable (h₂ : b = (1 / 2) ^ (2 / 3))
variable (h₃ : c = (1 / 5) ^ (2 / 3))

theorem order_of_numbers (a b c : ℝ) (h₁ : a = (1 / 2) ^ (1 / 3)) (h₂ : b = (1 / 2) ^ (2 / 3)) (h₃ : c = (1 / 5) ^ (2 / 3)) :
  c < b ∧ b < a := 
by
  sorry

end order_of_numbers_l820_820553


namespace tile_difference_l820_820001

theorem tile_difference
  (original_blue : ℕ)
  (original_green : ℕ)
  (first_green_border : ℕ)
  (second_blue_border : ℕ)
  (total_blue : ℕ := original_blue + second_blue_border)
  (total_green : ℕ := original_green + first_green_border)
  : total_blue - total_green = 16 :=
by
  -- Given conditions
  assume h₁ : original_blue = 20,
  assume h₂ : original_green = 10,
  assume h₃ : first_green_border = 18,
  assume h₄ : second_blue_border = 24,
  assume h₅ : total_blue = original_blue + second_blue_border,
  assume h₆ : total_green = original_green + first_green_border,
  -- Expected result
  show total_blue - total_green = 16 from sorry

end tile_difference_l820_820001


namespace triangle_inradius_l820_820123

theorem triangle_inradius (P A : ℝ) (hP : P = 28) (hA : A = 35) : 
  let s := P / 2 in
  let r := A / s in
  r = 2.5 :=
by
  have s_def : s = 28 / 2 := by rw [hP]; norm_num
  have r_def : r = 35 / 14 := by rw [s_def, hA]; norm_num
  exact r_def

end triangle_inradius_l820_820123


namespace positive_difference_of_sums_l820_820281

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820281


namespace positive_difference_even_odd_sums_l820_820261

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820261


namespace probability_math_majors_consecutive_is_one_over_42_l820_820979

-- Define the problem's entities and conditions
def people : ℕ := 11
def math_majors : ℕ := 5
def physics_majors : ℕ := 3
def chemistry_majors : ℕ := 3

-- Total number of ways to arrange the people around the table
def total_ways : ℕ := (people - 1)!

-- Number of ways to arrange blocks and math majors
def blocks : ℕ := (people - math_majors)
def ways_to_arrange_blocks : ℕ := blocks!
def ways_to_arrange_math_majors : ℕ := math_majors!

-- Number of favorable outcomes
def favorable_ways : ℕ := ways_to_arrange_blocks * ways_to_arrange_math_majors

-- Probability calculation
def probability_math_majors_consecutive : ℚ := favorable_ways / total_ways

-- The mathematical proof problem statement
theorem probability_math_majors_consecutive_is_one_over_42 :
  probability_math_majors_consecutive = 1 / 42 := sorry

end probability_math_majors_consecutive_is_one_over_42_l820_820979


namespace speed_of_student_B_l820_820497

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820497


namespace pump_leak_drain_time_l820_820917

theorem pump_leak_drain_time {P L : ℝ} (hP : P = 0.25) (hPL : P - L = 0.05) : (1 / L) = 5 :=
by sorry

end pump_leak_drain_time_l820_820917


namespace liquidX_percentage_l820_820064

-- Definitions of given conditions
def percentX (s : ℝ) (m : ℝ) : ℝ := (s / 100) * m

def massA : ℝ := 250
def massB : ℝ := 450
def massC : ℝ := 350

def percentXA : ℝ := 0.8
def percentXB : ℝ := 1.8
def percentXC : ℝ := 2.5

def total_mass : ℝ := massA + massB + massC

def massX : ℝ := percentX percentXA massA
                + percentX percentXB massB
                + percentX percentXC massC

def percentXInMixture : ℝ := (massX / total_mass) * 100

theorem liquidX_percentage :
  (Real.floor (percentXInMixture * 100 + 0.5) / 100) = 1.80 := by
  sorry

end liquidX_percentage_l820_820064


namespace number_of_permissible_sandwiches_l820_820832

theorem number_of_permissible_sandwiches (b m c : ℕ) (h : b = 5) (me : m = 7) (ch : c = 6) 
  (no_ham_cheddar : ∀ bread, ¬(bread = ham ∧ cheese = cheddar))
  (no_turkey_swiss : ∀ bread, ¬(bread = turkey ∧ cheese = swiss)) : 
  5 * 7 * 6 - (5 * 1 * 1) - (5 * 1 * 1) = 200 := 
by 
  sorry

end number_of_permissible_sandwiches_l820_820832


namespace speed_of_student_B_l820_820432

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820432


namespace perfect_square_divisors_product_factorials_l820_820703

theorem perfect_square_divisors_product_factorials :
  let product: ℕ := (nat.factorial 1) * (nat.factorial 3) * (nat.factorial 5) * (nat.factorial 7) * (nat.factorial 9) * (nat.factorial 11)
  (number_of_perfect_square_divisors product = 432) :=
by
  sorry

end perfect_square_divisors_product_factorials_l820_820703


namespace determine_y_l820_820551

variable (x y : ℝ)

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

theorem determine_y : star 4 y = 22 → y = 1 / 3 := by
  intro h
  have : star 4 y = 5 * 4 - 2 * y + 2 * 4 * y := rfl
  rw this at h
  linarith

end determine_y_l820_820551


namespace sqrt2_approximation_from_sqrt51_l820_820092

noncomputable def sqrt_approximation (a b : ℕ) : Prop :=
  abs (real.sqrt 2 - (a / b : ℝ)) < 1e-5

theorem sqrt2_approximation_from_sqrt51
    (h : abs (real.sqrt 51 - (7 + real.sqrt 2 / 10)) < 1e-6) :
    sqrt_approximation 99 70 :=
sorry

end sqrt2_approximation_from_sqrt51_l820_820092


namespace preceding_integer_binary_l820_820705

theorem preceding_integer_binary (M : ℕ) (h : M = 0b110101) : 
  (M - 1) = 0b110100 :=
by
  sorry

end preceding_integer_binary_l820_820705


namespace speed_of_student_B_l820_820422

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820422


namespace jamie_cherry_pies_l820_820028

theorem jamie_cherry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36) (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) : 
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := 
by {
  sorry
}

end jamie_cherry_pies_l820_820028


namespace inflate_balloon_with_P_Q_l820_820558

variables (p q r : ℝ)

-- Given conditions
def condition1 := p + q + r = 1 / 2
def condition2 := p + r = 1 / 3
def condition3 := q + r = 1 / 6

-- Final statement to prove
theorem inflate_balloon_with_P_Q:
  condition1 → condition2 → condition3 → (1 / (p + q) = 2) :=
by
  intros h1 h2 h3,
  sorry

end inflate_balloon_with_P_Q_l820_820558


namespace positive_difference_of_sums_l820_820290

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820290


namespace magnitude_of_b_l820_820686

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820686


namespace convert_1024_base10_to_base8_to_base2_l820_820960

theorem convert_1024_base10_to_base8_to_base2 :
  ∃ n : ℕ, (n = 1024 ∧ nat.digits 8 n = [2, 0, 0, 0] ∧ nat.digits 2 (nat.of_digits 2 (nat.digits 8 n)) = [0, 1, 0, 0, 0, 0, 0, 0, 0]) :=
by
  sorry

end convert_1024_base10_to_base8_to_base2_l820_820960


namespace student_b_speed_l820_820445

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820445


namespace sequence_value_6_l820_820706

theorem sequence_value_6 (n : ℕ) (h : n ∈ {1, 2, 3, 4, 5}) : n * 6 = 6 * n := by sorry

example : 6 * 6 = 36 := by
  have h := sequence_value_6 6 (by simp)
  simp [h]
  sorry

end sequence_value_6_l820_820706


namespace units_digit_sum_l820_820138

theorem units_digit_sum (n m k : ℕ) (h1 : n ≡ 20 [MOD 4]) (h2 : m ≡ 21 [MOD 4]) (h3 : k ≡ 20 [MOD 4]) :
  let u2 := if h1 ≡ 0 [MOD 4] then 6 else 0
  let u3 := if h2 ≡ 1 [MOD 4] then 3 else 0
  let u7 := if h3 ≡ 0 [MOD 4] then 1 else 0
  (u2 + u3 + u7) % 10 = 0 :=
by 
  have h1_def : 2 ≡ u2 % 10 := sorry
  have h2_def : 3 ≡ u3 % 10 := sorry
  have h3_def : 7 ≡ u7 % 10 := sorry
  sorry

end units_digit_sum_l820_820138


namespace angle_of_line_AB_is_60_degrees_l820_820608

theorem angle_of_line_AB_is_60_degrees :
  ∀ (A B : ℕ × ℚ), 
    A = (2, 0) →
    B = (3, Real.sqrt 3) →
    ∃ α : ℝ, α = 60 :=
by
  sorry

end angle_of_line_AB_is_60_degrees_l820_820608


namespace student_B_speed_l820_820508

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820508


namespace number_of_outfits_l820_820876

theorem number_of_outfits :
  let shirts := 5 in
  let pants := 3 in
  let ties := 2 in
  shirts * pants * ties = 30 :=
by
  simp [mul_add]
  sorry

end number_of_outfits_l820_820876


namespace student_b_speed_l820_820444

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820444


namespace three_reflections_transform_to_glide_reflection_l820_820817

noncomputable def composition_of_three_reflections_is_glide_reflection
  (l1 l2 l3 : ℝ → ℝ)
  (h1 : ∀ x, l1 x ≠ l2 x)
  (h2 : ∀ x, l2 x ≠ l3 x)
  (h3 : ∀ x, l1 x ≠ l3 x)
  (h4 : ∀ x, ¬ (l1 x = l2 x ∧ l1 x = l3 x)) : Prop :=
∃ l : ℝ → ℝ, ∃ t : ℝ → ℝ, ∀ x, glide_reflection (l x) (t x)

axiom glide_reflection : (l t : ℝ → ℝ) (x : ℝ) → Prop

theorem three_reflections_transform_to_glide_reflection
  (l1 l2 l3 : ℝ → ℝ)
  (h1 : ∀ x, l1 x ≠ l2 x)
  (h2 : ∀ x, l2 x ≠ l3 x)
  (h3 : ∀ x, l1 x ≠ l3 x)
  (h4 : ∀ x, ¬ (l1 x = l2 x ∧ l1 x = l3 x)) :
  composition_of_three_reflections_is_glide_reflection l1 l2 l3 h1 h2 h3 h4 :=
sorry

end three_reflections_transform_to_glide_reflection_l820_820817


namespace find_p_l820_820638

theorem find_p
  (A B C r s p q : ℝ)
  (h1 : A ≠ 0)
  (h2 : r + s = -B / A)
  (h3 : r * s = C / A)
  (h4 : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by {
  sorry
}

end find_p_l820_820638


namespace speed_of_student_B_l820_820501

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820501


namespace positive_diff_even_odd_sums_l820_820178

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820178


namespace line_through_points_l820_820111

def line_equation_general_form (p1 p2 : ℝ × ℝ) := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  3 * x + 8 * y - 15 = 0

theorem line_through_points (x1 y1 x2 y2 : ℝ)
  (h₀ : (x1, y1) = (-5, 0))
  (h₁ : (x2, y2) = (3, -3)) :
  line_equation_general_form (x1, y1) (x2, y2) = (3 * x + 8 * y - 15 = 0) :=
by
  sorry

end line_through_points_l820_820111


namespace magnitude_b_sqrt_3_l820_820665

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820665


namespace arithmetic_sequence_collinear_points_l820_820619

variable {V : Type*} [InnerProductSpace ℝ V]

theorem arithmetic_sequence_collinear_points 
  {A B P : V} 
  (a : ℕ → ℝ) 
  (h_collinear : ∃ λ : ℝ, P = (1 - λ) • A + λ • B)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d)
  (h_OE : P = a 1 • A + a 4015 • B) :
  a 2008 = 1 / 2 :=
by
  sorry

end arithmetic_sequence_collinear_points_l820_820619


namespace magnitude_of_b_l820_820644

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820644


namespace angle_terminal_side_equiv_l820_820562

def radian_equiv_angle (deg_angle rad_angle : ℝ) (k : ℤ) : Prop :=
  rad_angle = (deg_angle / 180 * Real.pi) + 2 * k * Real.pi

theorem angle_terminal_side_equiv (k : ℤ) :
  ∀ β, β = (150 / 180 * Real.pi) + 2 * k * Real.pi ↔ β = 5 * Real.pi / 6 + 2 * k * Real.pi :=
by 
  intros β
  split
  { intro h
    rw [h] 
    }
  { intro h
    rw [h] 
    }

end angle_terminal_side_equiv_l820_820562


namespace cross_ratio_invariance_l820_820329

noncomputable def conic_bundle (conics : set (set ℝ × ℝ) → Prop) : Prop :=
∀ C ∈ conics, ∃ (P Q : set ℝ × ℝ), ∀ p ∈ C, p = P ∨ p = Q

noncomputable def circle_bundle (circles : set (set ℝ × ℝ) -> Prop) : Prop :=
∀ C ∈ circles, ∃ O r, ∀ p ∈ C, (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2

noncomputable def polar (P : ℝ × ℝ) (Γ : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
{Q : ℝ × ℝ | ∀ O r, Γ = {p | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2} → 
(Q.1 - P.1)*(O.1 - P.1) + (Q.2 - P.2)*(O.2 - P.2) = 1}

theorem cross_ratio_invariance (P : ℝ × ℝ) (Γ1 Γ2 Γ3 Γ4 : set (ℝ × ℝ))
(O1 O2 O3 O4 : ℝ × ℝ)
(h1 : conic_bundle {Γ1, Γ2, Γ3, Γ4})
(h2 : circle_bundle {Γ1, Γ2, Γ3, Γ4})
(h3 : ∀ Γ ∈ {Γ1, Γ2, Γ3, Γ4}, ∃ O r, Γ = {p | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2})
(h4 : ¬P ∈ O1 ∧ ¬P ∈ O2 ∧ ¬P ∈ O3 ∧ ¬P ∈ O4) :
∃ b, ∀ (Γ : set (ℝ × ℝ)), 
(Γ = Γ1 ∨ Γ = Γ2 ∨ Γ = Γ3 ∨ Γ = Γ4) →
b = (polar P Γ1, polar P Γ2, polar P Γ3, polar P Γ4) :=
by sorry

end cross_ratio_invariance_l820_820329


namespace final_position_distance_refuel_needed_l820_820909

section

-- Define the travel segments as given in the problem
def travel_segments : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]

-- Define the fuel consumption rate and initial fuel amount
def fuel_rate : Int := 3
def initial_fuel : Int := 180

-- Proving final position relative to point A
theorem final_position_distance (travel_segments : List Int) : 
  travel_segments.sum = 39 :=
by {
  have h1 : travel_segments.sum = 15 - 2 + 5 - 1 + 10 - 3 - 2 + 12 + 4 - 5 + 6 := rfl,
  norm_num at h1,
  exact h1,
}

-- Proving whether refueling is needed and by how much
theorem refuel_needed (travel_segments : List Int) (fuel_rate initial_fuel : Int) : 
  fuel_rate * (travel_segments.map (Int.natAbs)).sum - initial_fuel = 15 :=
by {
  have h2 : fuel_rate * (travel_segments.map (Int.natAbs)).sum = 3 * (15 + 2 + 5 + 1 + 10 + 3 + 2 + 12 + 4 + 5 + 6) := rfl,
  norm_num at h2,
  exact rfl.sub h2,  sorry,
}
end

end final_position_distance_refuel_needed_l820_820909


namespace magnitude_b_sqrt_3_l820_820659

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820659


namespace cannot_exist_l820_820748

-- Definition for divisibility by 9 and 10
def divisible_by_10 (n : Nat) : Prop := 
  n % 10 = 0

def divisible_by_9 (n : Nat) : Prop := 
  (n.digits.map (fun d => d.val)).sum % 9 = 0

-- Definitions translated from the problem's conditions
def ДЕВЯНОСТО (D E V Я Н С Т O : ℕ) := 100000000 * D + 10000000 * E + 1000000 * V + 100000 * Я + 10000 * Н + 1000 * С + 100 * Т + 10 * O
def ДЕВЯТКА (D E V Я T К A : ℕ) := 100000 * D + 10000 * E + 1000 * V + 100 * Я + 10 * T + К
def СOTКА (С O T K A : ℕ) := 10000 * С + 1000 * O + 100 * T + 10 * K + A

-- Hypotheses based on problem's conditions
variables (D E V Я Н С Т O T К A : ℕ)

-- Given conditions
hypothesis H1 : divisible_by_10 (ДЕВЯНОСТО D E V Я Н С Т O) 
hypothesis H2 : divisible_by_9 (ДЕВЯНОСТО D E V Я Н С Т O)
hypothesis H3 : divisible_by_9 (ДЕВЯТКА D E V Я T К A)

-- The final proof/goal statement
theorem cannot_exist :
  ¬ (∃ (С O T K A : ℕ), divisible_by_9 (СOTКА С O T K A)) :=
sorry

end cannot_exist_l820_820748


namespace slope_angle_of_line_l820_820578

theorem slope_angle_of_line (a b c : ℝ) (h : a = 1 ∧ b = sqrt 3 ∧ c = -8) : 
  (x y : ℝ) ∈ {p | a * p.1 + b * p.2 + c = 0} → ∃ α : ℝ, 0 < α ∧ α < π ∧ α = 5 * π / 6 :=
begin
  intros p hp,
  sorry
end

end slope_angle_of_line_l820_820578


namespace student_b_speed_l820_820363

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820363


namespace avg_age_boys_class_l820_820009

-- Definitions based on conditions
def avg_age_students : ℝ := 15.8
def avg_age_girls : ℝ := 15.4
def ratio_boys_girls : ℝ := 1.0000000000000044

-- Using the given conditions to define the average age of boys
theorem avg_age_boys_class (B G : ℕ) (A_b : ℝ) 
  (h1 : avg_age_students = (B * A_b + G * avg_age_girls) / (B + G)) 
  (h2 : B = ratio_boys_girls * G) : 
  A_b = 16.2 :=
  sorry

end avg_age_boys_class_l820_820009


namespace smallest_integer_2023m_54321n_l820_820868

theorem smallest_integer_2023m_54321n : ∃ (m n : ℤ), 2023 * m + 54321 * n = 1 :=
sorry

end smallest_integer_2023m_54321n_l820_820868


namespace relay_race_arrangements_l820_820900

def students := {A, B, C, D, E, F}

def valid_first_legs := {A, B}

def valid_fourth_legs := {A, C}

def chosen_participants := 4

theorem relay_race_arrangements:
  let n := (4 : ℕ) in
  ∀ (students : set char) (valid_first_legs valid_fourth_legs : set char) (chosen_participants : ℕ),
  students = {'A', 'B', 'C', 'D', 'E', 'F'} →
  valid_first_legs = {'A', 'B'} →
  valid_fourth_legs = {'A', 'C'} →
  chosen_participants = 4 →
  -- the number of different arrangements
  (n = 24) := 
begin
  intros students valid_first_legs valid_fourth_legs chosen_participants,
  intro h1,
  intro h2,
  intro h3,
  intro h4,
  sorry
end

end relay_race_arrangements_l820_820900


namespace inclination_angle_l820_820744

theorem inclination_angle (x y : ℝ) (h : x + y - 3 = 0) : 
  ∃ θ : ℝ, θ = (3 * real.pi) / 4 :=
by
  sorry

end inclination_angle_l820_820744


namespace necessary_but_not_sufficient_l820_820597

noncomputable def has_zero (m : ℝ) : Prop :=
  ∃ x : ℝ, 2^x + m - 1 = 0

def decreasing_on (m : ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → y < +∞ → log m x > log m y

theorem necessary_but_not_sufficient (m : ℝ) :
  (decreasing_on m ↔ 0 < m ∧ m < 1) →
  (0 < m ∧ m < 1 → has_zero m) ∧
  (has_zero m → 0 < m ∧ m < 1) :=
by
  sorry

end necessary_but_not_sufficient_l820_820597


namespace perpendicular_vector_lambda_l820_820692

theorem perpendicular_vector_lambda:
  let a := (1, 2)
  let b := (1, -1)
  let c := (4, 5)
  ∃ λ: ℝ, (a.1 * (b.1 + λ * c.1) + a.2 * (b.2 + λ * c.2) = 0) -> 
  λ = 1 / 14 := 
by
  sorry

end perpendicular_vector_lambda_l820_820692


namespace find_speed_B_l820_820403

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820403


namespace solve_equation_l820_820332

theorem solve_equation {x : ℝ} (hx : x = 1) : 9 - 3 / x / 3 + 3 = 3 := by
  rw [hx] -- Substitute x = 1
  norm_num -- Simplify the numerical expression
  sorry -- to be proved

end solve_equation_l820_820332


namespace student_B_speed_l820_820473

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820473


namespace acute_angle_probability_is_half_l820_820813

noncomputable def probability_acute_angle (h_mins : ℕ) (m_mins : ℕ) : ℝ :=
  if m_mins < 15 ∨ m_mins > 45 then 1 / 2 else 0

theorem acute_angle_probability_is_half (h_mins : ℕ) (m_mins : ℕ) :
  let P := probability_acute_angle h_mins m_mins in
  0 ≤ m_mins ∧ m_mins < 60 →
  P = 1 / 2 :=
sorry

end acute_angle_probability_is_half_l820_820813


namespace carpet_area_in_yards_l820_820037

def main_length_feet : ℕ := 15
def main_width_feet : ℕ := 12
def extension_length_feet : ℕ := 6
def extension_width_feet : ℕ := 5
def feet_per_yard : ℕ := 3

def main_length_yards : ℕ := main_length_feet / feet_per_yard
def main_width_yards : ℕ := main_width_feet / feet_per_yard
def extension_length_yards : ℕ := extension_length_feet / feet_per_yard
def extension_width_yards : ℕ := extension_width_feet / feet_per_yard

def main_area_yards : ℕ := main_length_yards * main_width_yards
def extension_area_yards : ℕ := extension_length_yards * extension_width_yards

theorem carpet_area_in_yards : (main_area_yards : ℚ) + (extension_area_yards : ℚ) = 23.33 := 
by
  apply sorry

end carpet_area_in_yards_l820_820037


namespace remainder_b55_l820_820771

def concat_ints (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else (n.digits 10).foldl (λ a i, a * 10 + i) 0

lemma digits_concat : ∀ n, (concat_ints n).digits 10 = (list.range (n + 1)).bind (λ x, x.digits 10) :=
sorry

noncomputable def b : ℕ → ℕ
| 0     := 0
| (n+1) := let bn := b n in
           list.range (n + 1).foldl (λ acc i, acc * (10 ^ (i.digits 10).length) + i) 0

theorem remainder_b55 : b 55 % 55 = 0 :=
sorry

end remainder_b55_l820_820771


namespace min_area_quadrilateral_l820_820636

theorem min_area_quadrilateral:
  let Γ := {p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1},
      F := (2, 0 : ℝ),
      tangent_yaxis_intersection_x (x : ℝ) := (0, 2 * (if x > 0 then x else -x)),
      area_quadrilateral (A B P Q : ℝ × ℝ) := 
        let AB := (B.1 - A.1) * (B.2 + A.2) / 2,
            BP := (P.1 - B.1) * (P.2 + B.2) / 2,
            PQ := (Q.1 - P.1) * (Q.2 + P.2) / 2,
            QA := (A.1 - Q.1) * (A.2 + Q.2) / 2
        in AB + BP + PQ + QA
  in
      ∃ A B P Q : ℝ × ℝ, 
      (A ∈ Γ ∧ B ∈ Γ ∧ A.1 > 0 ∧ B.1 < 0)
      ∧ (P = tangent_yaxis_intersection_x A.2 ∧ Q = tangent_yaxis_intersection_x B.2)
      ∧ (area_quadrilateral A B P Q = 12) :=
by
  sorry

end min_area_quadrilateral_l820_820636


namespace difference_of_radii_l820_820131

theorem difference_of_radii (r R : ℝ) (h1 : (π * R^2) / (π * r^2) = 4) : R - r = r :=
by 
  have h2 : R^2 / r^2 = 4 := by 
    rw [←div_eq_iff_eq_mul_right (pi_ne_zero : π ≠ 0), div_eq_iff_eq_mul_right (pi_ne_zero : π ≠ 0)] at h1 
    exact h1
    
  have h3 : R / r = 2 := by
    exact (eq_of_sq_eq_sq h2).trans (or.inl (by ring)),
   
  have h4 : R = 2 * r := by
    rw [div_eq_iff_eq_mul_right (ne_of_gt (by norm_num1) : (r : ℝ) ≠ 0)] at h3 
    exact h3,

  rw [h4, sub_eq_add_neg, add_neg_eq_iff_eq_add, ← two_mul],
  ring,
  sorry

end difference_of_radii_l820_820131


namespace cost_of_shingles_for_sandy_l820_820085

theorem cost_of_shingles_for_sandy :
  let base := 3 -- feet
  let height := 5 -- feet
  let number_of_triangles := 8
  let area_of_one_triangle := (1 / 2 : ℝ) * base * height
  let total_area := number_of_triangles * area_of_one_triangle
  let section_area := 10 * 10 -- square feet
  let cost_per_section := 35 -- dollars
  ceiling_div total_area section_area * cost_per_section = 35 :=
by
  -- Definitions
  let base := 3
  let height := 5
  let number_of_triangles := 8
  let area_of_one_triangle := (1 / 2 : ℝ) * base * height
  let total_area := number_of_triangles * area_of_one_triangle
  let section_area := 10 * 10
  let cost_per_section := 35
  -- Calculation
  have h₁ : area_of_one_triangle = (7.5 : ℝ) := by sorry
  have h₂ : total_area = (8 * 7.5 : ℝ) := by sorry
  have h₃ : total_area = (60 : ℝ) := by sorry
  have h₄ : section_area = 100 := by sorry
  have h₅ : ceiling_div 60 section_area = 1 := by sorry
  show ceiling_div total_area section_area * cost_per_section = 35, by sorry

end cost_of_shingles_for_sandy_l820_820085


namespace tangency_G_to_Γ_l820_820762

variables {Γ : Type*} [circle Γ] {A B C D P Q E F G : point Γ}

-- conditions
hypothesis (trapezoid : A, B, C, D in distinct points and AB ∥ CD)
hypothesis (P_Q_on_AB : P and Q are points on segment AB with P ≠ Q and ordered as A, P, Q, B)
hypothesis (AP_eq_QB : dist A P = dist Q B)
hypothesis (sec_inter_C_P : E is the second intersection point of line CP with circle Γ)
hypothesis (sec_inter_C_Q : F is the second intersection point of line CQ with circle Γ)
hypothesis (intersection_AB_EF : intersection of lines AB and EF is G)

-- goal
theorem tangency_G_to_Γ : tangent (line D G) Γ :=
sorry

end tangency_G_to_Γ_l820_820762


namespace carrots_bad_count_l820_820564

def total_carrots : ℕ := 23 + 42 + 12 + 18
def percent_good : ℚ := 60 / 100
def good_carrots : ℕ := (percent_good * total_carrots).toNat
def bad_carrots : ℕ := total_carrots - good_carrots

theorem carrots_bad_count :
  bad_carrots = 38 :=
by
  have total_carrots_calc: total_carrots = 95 := by sorry
  have good_carrots_calc: good_carrots = 57 := by sorry
  have bad_carrots_calc: bad_carrots = 38 := by sorry
  exact bad_carrots_calc

end carrots_bad_count_l820_820564


namespace surjective_injective_eq_l820_820779

theorem surjective_injective_eq (f g : ℕ → ℕ) 
  (hf : Function.Surjective f) 
  (hg : Function.Injective g) 
  (h : ∀ n : ℕ, f n ≥ g n) : 
  ∀ n : ℕ, f n = g n := 
by
  sorry

end surjective_injective_eq_l820_820779


namespace student_B_speed_l820_820450

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820450


namespace simplify_expression_l820_820863

theorem simplify_expression : (2^3 + 2^3) / (2^(-3) + 2^(-3)) = 64 := by
  sorry

end simplify_expression_l820_820863


namespace find_speed_B_l820_820401

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820401


namespace coefficient_x3y4_l820_820159

theorem coefficient_x3y4 :
  ∀ (x y : ℝ), 
  (∃ c : ℝ, c = (coeff (x^3 * y^4) (((2/3)*x - (3/4)*y)^9)) ∧ c = 441 / 992) :=
by
  sorry

end coefficient_x3y4_l820_820159


namespace bicycle_speed_B_l820_820377

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820377


namespace positive_difference_even_odd_sums_l820_820248

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820248


namespace danny_total_bottle_caps_l820_820963

def danny_initial_bottle_caps : ℕ := 37
def danny_found_bottle_caps : ℕ := 18

theorem danny_total_bottle_caps : danny_initial_bottle_caps + danny_found_bottle_caps = 55 := by
  sorry

end danny_total_bottle_caps_l820_820963


namespace induced_subgraph_theorem_l820_820589

open Classical

noncomputable def ramsey_number (r : ℕ) : ℕ := sorry

theorem induced_subgraph_theorem (r : ℕ) : ∃ n : ℕ, ∀ G : Graph, G.connected → G.order ≥ n → (∃ (H : Graph), H = (K r) ∨ H = (K_1 r) ∨ H = (P r)) :=
by sorry

end induced_subgraph_theorem_l820_820589


namespace cannot_be_right_triangle_l820_820529

theorem cannot_be_right_triangle (a b c : Real) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 4) (h3 : c = Real.sqrt 5) : 
  (a^2 + b^2 ≠ c^2) :=
by
  have h_a_sq : a^2 = 3 := by sorry
  have h_b_sq : b^2 = 4 := by sorry
  have h_c_sq : c^2 = 5 := by sorry
  calc
    a^2 + b^2 ≠ c^2 : by
      rw [h_a_sq, h_b_sq, h_c_sq]
      norm_num
  sorry

end cannot_be_right_triangle_l820_820529


namespace magnitude_of_b_l820_820683

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820683


namespace product_greater_than_sum_l820_820058

theorem product_greater_than_sum {n : ℕ} (a : Fin n → ℝ) (h1 : ∀ i, a i > -1) 
  (h2 : ∀ i j, (a i = 0 → a j = 0) ∧ (a i > 0 → a j > 0) ∧ (a i < 0 → a j < 0)) :
  ((∏ i, (1 + a i)) > 1 + (∏ i, (a i))) :=
by
  sorry

end product_greater_than_sum_l820_820058


namespace correlative_relationship_l820_820530

theorem correlative_relationship :
  ∃ (d : Prop), 
  (d ↔ (∃ (X Y : Type) (n : X → ℕ) (m : Y → ℕ), 
        (n = (λ x, number_of_fatigued_drivers x)) ∧ 
        (m = (λ y, occurrence_of_traffic_accidents y)) ∧ 
        (∃ r, ∀ x y, r (n x) (m y)))) :=
by
  -- Here we assume the existence of X, Y, n, m and a relation r such that 
  -- there is a correlative relationship between the number of fatigued drivers 
  -- and occurrence of traffic accidents.
  sorry

-- Definitions to align with the conditions
def number_of_fatigued_drivers : ℕ := sorry
def occurrence_of_traffic_accidents : ℕ := sorry

end correlative_relationship_l820_820530


namespace positive_difference_even_odd_sums_l820_820266

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820266


namespace magnitude_b_eq_sqrt3_l820_820677

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820677


namespace ant_final_position_ant_farthest_distance_ant_total_sesame_seeds_l820_820536

def distances : List ℤ := [+5, -4, +11, -7, -10, +12, -9]

-- Question (1)
theorem ant_final_position :
  distances.sum = -2 := by
  sorry

-- Question (2)
theorem ant_farthest_distance :
  let positions := List.scanl (· + ·) 0 distances
  - positions.maximum == 12 := by
  sorry

-- Question (3)
noncomputable def total_seeds : ℤ :=
  distances.sum.map (Int.ofNat ∘ abs).sum * 2

theorem ant_total_sesame_seeds :
  total_seeds = 116 := by
  sorry

end ant_final_position_ant_farthest_distance_ant_total_sesame_seeds_l820_820536


namespace precise_location_of_jiuquan_city_l820_820311

theorem precise_location_of_jiuquan_city :
  ∃ (latitude : ℝ) (longitude : ℝ), latitude = 39.75 ∧ longitude = 98.52 :=
by 
  use 39.75
  use 98.52
  split
  { refl }
  { refl }

end precise_location_of_jiuquan_city_l820_820311


namespace magnitude_of_b_l820_820646

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820646


namespace jake_watch_hours_l820_820027

theorem jake_watch_hours (d : ℕ) (hday : d = 24) :
  let
    monday_hours := d / 2,
    tuesday_hours := 4,
    wednesday_hours := d / 4,
    thursday_hours := d / 3,
    friday_hours := 2 * (d / 4),
    total_before_sunday := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours,
    sunday_hours := total_before_sunday / 2,
    total_watched_by_sunday := total_before_sunday + sunday_hours,
    total_show_hours := 75
  in total_show_hours - total_watched_by_sunday = 12 :=
by
  -- Definitions
  let monday_hours := d / 2
  let tuesday_hours := 4
  let wednesday_hours := d / 4
  let thursday_hours := d / 3
  let friday_hours := 2 * (d / 4)
  let total_before_sunday := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let sunday_hours := total_before_sunday / 2
  let total_watched_by_sunday := total_before_sunday + sunday_hours
  let total_show_hours := 75

  -- Main Goal
  have h1 : d = 24 := by sorry
  have h2 : monday_hours = 12 := by sorry -- because d / 2 = 24 / 2
  have h3 : wednesday_hours = 6 := by sorry -- because d / 4 = 24 / 4
  have h4 : thursday_hours = 8 := by sorry -- because d / 3 = 24 / 3
  have h5 : friday_hours = 12 := by sorry -- because 2 * (24 / 4)
  have h_total_before_sunday : total_before_sunday = 42 := by sorry -- summing the hours
  
  refine Eq.trans _ (by sorry) -- main goal simplification
  sorry

end jake_watch_hours_l820_820027


namespace find_num_oranges_l820_820824

def num_oranges (O : ℝ) (x : ℕ) : Prop :=
  6 * 0.21 + O * (x : ℝ) = 1.77 ∧ 2 * 0.21 + 5 * O = 1.27
  ∧ 0.21 = 0.21

theorem find_num_oranges (O : ℝ) (x : ℕ) (h : num_oranges O x) : x = 3 :=
  sorry

end find_num_oranges_l820_820824


namespace magnitude_of_b_l820_820658

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820658


namespace triangle_angles_l820_820006

open EuclideanGeometry

variable {A B C K M L : Point}

/-- Given the following conditions:
1. Points \( K \) and \( M \) are on sides \( AB \) and \( AC \) respectively.
2. \( L \) is the intersection of lines \( MB \) and \( KC \).
3. Quadrilaterals \( AKLM \) and \( KBCM \) are cyclic.
4. Both circumcircles of \( AKLM \) and \( KBCM \) have the same size.

Prove the measures of the interior angles of \( \triangle ABC \):
- \( \angle BAC = 45^\circ \)
- The sum of \( \angle ABC \) and \( \angle ACB \) is \( 135^\circ \)
-/
theorem triangle_angles (h1 : K ∈ Line(A, B))
                     (h2 : M ∈ Line(A, C))
                     (h3 : Concurrent (Line(M, B)) (Line(K, C)) L)
                     (h4 : CyclicQuad A K L M)
                     (h5 : CyclicQuad K B C M)
                     (h6 : CircumcircleSize_eq A K L M K B C M ) :
  ∠BAC = 45 ∧ (∠ABC + ∠ACB = 135) := 
  sorry

end triangle_angles_l820_820006


namespace surface_area_after_removing_corner_cubes_l820_820557

theorem surface_area_after_removing_corner_cubes :
  let original_cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (original_cube_side ^ 2)
  let corner_cube_faces_area := 3 * (corner_cube_side ^ 2)
  let net_change_due_to_one_corner_cube := -corner_cube_faces_area + corner_cube_faces_area
  let total_corners := 8
  let total_net_change := total_corners * net_change_due_to_one_corner_cube
  let remaining_surface_area := original_surface_area + total_net_change
  remaining_surface_area = 96 :=
by
  let original_cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (original_cube_side ^ 2)
  let corner_cube_faces_area := 3 * (corner_cube_side ^ 2)
  let net_change_due_to_one_corner_cube := -corner_cube_faces_area + corner_cube_faces_area
  let total_corners := 8
  let total_net_change := total_corners * net_change_due_to_one_corner_cube
  let remaining_surface_area := original_surface_area + total_net_change
  have h : remaining_surface_area = 96 := by
    sorry
    exact h

end surface_area_after_removing_corner_cubes_l820_820557


namespace clock_angle_acute_probability_l820_820808

-- Given the condition that a clock stops randomly at any moment,
-- and defining the probability of forming an acute angle between the hour and minute hands,
-- prove that this probability is 1/2.

theorem clock_angle_acute_probability : 
  (probability (\theta : ℝ, is_acute ⟨θ % 360, 0 ≤ θ % 360 < 360⟩) = 1/2) :=
-- Definitions and conditions.
sorry

end clock_angle_acute_probability_l820_820808


namespace positive_difference_even_odd_sums_l820_820301

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820301


namespace student_B_speed_l820_820453

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820453


namespace find_speed_B_l820_820398

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820398


namespace machine_x_production_rate_l820_820888

theorem machine_x_production_rate :
  ∃ (Wx : ℝ),
    (Ty = 1080 / (1.20 * Wx)) ∧
    (Tx = Ty + 60) ∧
    (1080 = Wx * Tx) ∧
    Wx = 3 :=
begin
  sorry
end

end machine_x_production_rate_l820_820888


namespace sum_of_reciprocals_l820_820746

-- Define the arithmetic sequence and the sum of the first n terms
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Given Conditions
axiom h1 : a 2 = 4
axiom h2 : a 9 = (1 / 2) * a 12 + 6

-- Define the sum of the first n terms of the sequence
def S_n (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i

-- Prove that the sum of the first 10 terms of {1 / S_n} is 10/11
theorem sum_of_reciprocals : (∑ n in finset.range 10, 1 / S_n n) = 10 / 11 :=
sorry

end sum_of_reciprocals_l820_820746


namespace difference_max_min_changes_l820_820538

theorem difference_max_min_changes (initial_yes: ℝ) (initial_no: ℝ) (final_yes: ℝ) (final_no: ℝ) (x_difference: ℝ) :
    initial_yes = 50 ∧ initial_no = 50 ∧ final_yes = 70 ∧ final_no = 30 →
    x_difference = 60 :=
begin
    sorry
end

end difference_max_min_changes_l820_820538


namespace speed_of_student_B_l820_820478

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820478


namespace problem_solution_l820_820614

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def F (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ :=
(x - b) * f (x - b) + 1009

theorem problem_solution
  (a c : ℝ)
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_b : ∃ b : ℝ, 2 * b = a + c) :
  (∃ (b : ℝ), 2 * b = a + c) → 
  F f (classical.some h_b) a + F f (classical.some h_b) c = 2018 :=
by
  -- the proof would go here
  sorry

end problem_solution_l820_820614


namespace angle_B44_B45_B43_is_45_degree_l820_820708

noncomputable theory

structure Triangle (α : Type*) [linear_ordered_field α] :=
  (B1 B2 B3 : α × α)
  (angle_B2_90 : (angle (B1 - B2) (B3 - B2)) = 90)
  (isosceles_right : (angle (B1 - B2) (B2 - B3)) = 45 ∧ (angle (B2 - B3) (B3 - B1)) = 45)

def midpoint {α : Type*} [linear_ordered_field α] (A B : α × α) : α × α :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def B_seq {α : Type*} [linear_ordered_field α] (B : ℕ → α × α) : ℕ → α × α
| 0     := B 0
| 1     := B 1
| 2     := B 2
| (n+3) := midpoint (B_seq n) (B_seq (n+1))

theorem angle_B44_B45_B43_is_45_degree (α : Type*) [linear_ordered_field α] 
  (B : ℕ → α × α) (B_triangle : Triangle α) :
  (angle (B_seq B 43 - B_seq B 45) (B_seq B 45 - B_seq B 44)) = 45 :=
sorry

end angle_B44_B45_B43_is_45_degree_l820_820708


namespace positive_difference_sums_l820_820214

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820214


namespace cristina_gave_nicky_head_start_l820_820071

theorem cristina_gave_nicky_head_start 
  (cristina_speed : ℕ)
  (nicky_speed : ℕ)
  (time : ℕ) 
  (distance_nicky : ℕ) 
  (distance_cristina : ℕ)
  (head_start : ℕ) 
  (h1 : cristina_speed = 4) 
  (h2 : nicky_speed = 3) 
  (h3 : time = 36) 
  (h4 : distance_nicky = nicky_speed * time) 
  (h5 : distance_cristina = cristina_speed * time) 
  (h6 : head_start = distance_cristina - distance_nicky) : 
  head_start = 36 :=
by 
  rw [h1, h2, h3] at h4 h5
  simp at h4 h5
  rw [h4, h5]
  simp [h6]
  sorry

end cristina_gave_nicky_head_start_l820_820071


namespace fixed_point_line_l820_820309

theorem fixed_point_line (k : ℝ) :
  ∀ k : ℝ, ∃ x y : ℝ, (kx + y - 2 = 3k) ∧ (x = 3) ∧ (y = 2) :=
by
  sorry

end fixed_point_line_l820_820309


namespace study_both_cooking_and_yoga_l820_820010

open Set

theorem study_both_cooking_and_yoga:
  ∀ (total_cooking cooking_only cooking_and_weaving all_curriculums: ℕ), 
  total_cooking = 15 → 
  cooking_only = 2 → 
  cooking_and_weaving = 3 → 
  all_curriculums = 3 → 
  total_cooking - cooking_only - cooking_and_weaving + all_curriculums = 10 := 
by 
  intros total_cooking cooking_only cooking_and_weaving all_curriculums
  intros h_total h_only h_weaving h_all
  rw [h_total, h_only, h_weaving, h_all]
  norm_num

end study_both_cooking_and_yoga_l820_820010


namespace v2_p_minus_1_ge_n2_l820_820059

theorem v2_p_minus_1_ge_n2 (n : ℕ) (hn : 1 < n) (p : ℕ) (hp : p.prime) (hdvd : p ∣ (2^(2^n) + 1)) : 
  int.nat_abs (multiplicity 2 (p - 1)) ≥ n + 2 :=
sorry

end v2_p_minus_1_ge_n2_l820_820059


namespace group_age_analysis_l820_820957

theorem group_age_analysis (total_members : ℕ) (average_age : ℝ) (zero_age_members : ℕ) 
  (h1 : total_members = 50) (h2 : average_age = 5) (h3 : zero_age_members = 10) :
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  non_zero_members = 40 ∧ non_zero_average_age = 6.25 :=
by
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  have h_non_zero_members : non_zero_members = 40 := by sorry
  have h_non_zero_average_age : non_zero_average_age = 6.25 := by sorry
  exact ⟨h_non_zero_members, h_non_zero_average_age⟩

end group_age_analysis_l820_820957


namespace clock_angle_acute_probability_l820_820798

noncomputable def probability_acute_angle : ℚ := 1 / 2

theorem clock_angle_acute_probability :
  ∀ (hour minute : ℕ), (hour >= 0 ∧ hour < 12) →
  (minute >= 0 ∧ minute < 60) →
  (let angle := min (60 * hour - 11 * minute) (720 - (60 * hour - 11 * minute)) in angle < 90 ↔ probability_acute_angle = 1 / 2) :=
sorry

end clock_angle_acute_probability_l820_820798


namespace number_of_ways_courses_different_l820_820334

theorem number_of_ways_courses_different : let courses : Finset ℕ := {1, 2, 3, 4} in
  let choose_two_courses (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ t, t.card = 2) in
  ∃ C_A C_B : Finset (Finset ℕ), 
    C_A = choose_two_courses courses ∧ 
    C_B = choose_two_courses courses ∧
    (∑ (s_A : Finset ℕ) in C_A, (∑ (s_B : Finset ℕ) in C_B, if (s_A ∩ s_B).card < 2 then 1 else 0)) = 30 := sorry

end number_of_ways_courses_different_l820_820334


namespace conditional_probability_of_A_given_target_hit_l820_820014

theorem conditional_probability_of_A_given_target_hit :
  (3 / 5 : ℚ) * ( ( 4 / 5 + 1 / 5) ) = (15 / 23 : ℚ) :=
  sorry

end conditional_probability_of_A_given_target_hit_l820_820014


namespace student_b_speed_l820_820442

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820442


namespace quadrilateral_is_trapezoid_l820_820749

-- Definitions of the points and quadrilateral
variables {A B C D O : Type*}

-- Conditions
variables {h1 : ∃ (O : Type*), A ≠ B ∧ B ≠ O ∧ C ≠ D ∧ D ≠ A ∧ B ≠ D ∧ A ≠ C ∧ O ∈ Line.segment A B ∧ O ∈ Line.segment C D}
variables {h2 : Distance A B = Distance O D}
variables {h3 : Distance A D = Distance O C}
variables {h4 : ∠BAC = ∠BDA}

-- Goal
theorem quadrilateral_is_trapezoid :
  is_trapezoid A B C D := 
sorry

end quadrilateral_is_trapezoid_l820_820749


namespace positive_difference_even_odd_sum_l820_820269

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820269


namespace sum_of_possible_b_l820_820913

theorem sum_of_possible_b : 
  (∀ b : ℤ, 
    (∃ r s : ℤ, 
      r ≠ s ∧ 
      r < 0 ∧ 
      s < 0 ∧ 
      r * s = 45 ∧ 
      r + s = b) → 
    ∑ b in { b | ∃ r s : ℤ, 
                     r ≠ s ∧ 
                     r < 0 ∧ 
                     s < 0 ∧ 
                     r * s = 45 ∧ 
                     r + s = b}, 
      b) 
  = -78 :=
sorry

end sum_of_possible_b_l820_820913


namespace range_of_m_value_of_m_l820_820695

open Real
open Function

-- Definitions of vectors in the conditions
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (-1, 2)
def c.m (m : ℝ) : ℝ × ℝ := (2, m)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Condition 1: Prove range of m
theorem range_of_m (m : ℝ) : dot_product a (c.m m) < m^2 → m > 4 ∨ m < -2 :=
by sorry

-- Addition of two vectors
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Condition 2: Prove value of m
theorem value_of_m (m : ℝ) : vector_add a (c.m m) = (λ k, (k * b)) := 
by sorry


end range_of_m_value_of_m_l820_820695


namespace min_number_of_snakes_l820_820322

-- Definitions based on the conditions
def total_people : ℕ := 79
def only_dogs : ℕ := 15
def only_cats : ℕ := 10
def only_cats_and_dogs : ℕ := 5
def cats_dogs_and_snakes : ℕ := 3

theorem min_number_of_snakes : ∃ n : ℕ, n ≤ cats_dogs_and_snakes :=
by {
  -- Proof placeholder
  existsi cats_dogs_and_snakes,
  sorry
}

end min_number_of_snakes_l820_820322


namespace number_of_roots_in_right_half_plane_is_one_l820_820576

def Q5 (z : ℂ) : ℂ := z^5 + z^4 + 2*z^3 - 8*z - 1

theorem number_of_roots_in_right_half_plane_is_one :
  (∃ n, ∀ z, Q5 z = 0 ∧ z.re > 0 ↔ n = 1) := 
sorry

end number_of_roots_in_right_half_plane_is_one_l820_820576


namespace exists_x0_condition_l820_820094

theorem exists_x0_condition
  (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_diff : ∀ x ∈ Set.Ioo 0 1, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 1)
  (h_f1 : f 1 = 0) :
  ∃ x₀ ∈ Set.Ioo 0 1, |fderiv ℝ f x₀| ≥ 2018 * (f x₀) ^ 2018 := by
  sorry

end exists_x0_condition_l820_820094


namespace hundredth_digit_l820_820864

theorem hundredth_digit (n : ℕ) (h : n = 100) :
  let recurring_seq := 03
  let digit := recurring_seq[(n - 1) % recurring_seq.length]
  \(\frac{21}{700}\) = 0.\overline{03} → digit = '3' :=
by sorry

end hundredth_digit_l820_820864


namespace manuscript_fee_3800_l820_820835

theorem manuscript_fee_3800 (tax_fee manuscript_fee : ℕ) 
  (h1 : tax_fee = 420) 
  (h2 : (0 < manuscript_fee) ∧ 
        (manuscript_fee ≤ 4000) → 
        tax_fee = (14 * (manuscript_fee - 800)) / 100) 
  (h3 : (manuscript_fee > 4000) → 
        tax_fee = (11 * manuscript_fee) / 100) : manuscript_fee = 3800 :=
by
  sorry

end manuscript_fee_3800_l820_820835


namespace josh_remaining_marbles_l820_820759

theorem josh_remaining_marbles : 
  let initial_marbles := 19 
  let lost_marbles := 11
  initial_marbles - lost_marbles = 8 := by
  sorry

end josh_remaining_marbles_l820_820759


namespace find_lambda_l820_820693

open Real

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -1)
def vec_c : ℝ × ℝ := (4, 5)

-- Define the dot product function for two-dimensional vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Definition of perpendicular vectors: their dot product is zero
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_lambda (λ : ℝ) : 
  perpendicular vec_a (vec_b.1 + λ * vec_c.1, vec_b.2 + λ * vec_c.2) →
  λ = 1 / 14 :=
by
  sorry

end find_lambda_l820_820693


namespace sum_of_prime_factors_eq_28_l820_820582

-- Define 2310 as a constant
def n : ℕ := 2310

-- Define the prime factors of 2310
def prime_factors : List ℕ := [2, 3, 5, 7, 11]

-- The sum of the prime factors
def sum_prime_factors : ℕ := prime_factors.sum

-- State the theorem
theorem sum_of_prime_factors_eq_28 : sum_prime_factors = 28 :=
by 
  sorry

end sum_of_prime_factors_eq_28_l820_820582


namespace speed_of_student_B_l820_820421

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820421


namespace find_mappings_l820_820776

namespace Mappings

variables {I : Type} [linear_ordered_field I] (k : ℝ) (G : Set (I × I))

def valid_mapping (f : I × I → I) : Prop :=
  (∀ x y z : I, f (f (x, y), z) = f (x, f (y, z))) ∧
  (∀ x : I, f (x, 1) = x ∧ f (1, x) = x) ∧
  (∀ x y z : I, f (z * x, z * y) = z^k * f (x, y))

theorem find_mappings (k : ℝ) :
  (∀ f : I × I → I, valid_mapping f → 
  (f = (λ p : I × I, ite (p.1 ≤ p.2) p.1 p.2)) ∨ 
  (f = (λ p : I × I, p.1 * p.2)))
  := sorry

end Mappings

end find_mappings_l820_820776


namespace student_B_speed_l820_820467

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820467


namespace min_colors_icosahedron_l820_820606

-- Define the structure and properties of an icosahedron
structure icosahedron where
  vertices : ℕ := 12
  faces : ℕ := 20
  edges : ℕ := 30
  faces_per_vertex : ℕ := 5
  adjacent_faces (f1 f2 : ℕ) : Prop

-- Define the coloring problem
def proper_coloring (icosa : icosahedron) (colors : ℕ) : Prop :=
  ∀ (c : ℕ → ℕ), (∀ f1 f2, icosa.adjacent_faces f1 f2 → c f1 ≠ c f2)

-- Define the minimum number of colors required
def min_colors (icosa : icosahedron) : ℕ :=
  if proper_coloring icosa 3 then 3 else 4

-- The theorem to prove
theorem min_colors_icosahedron : min_colors (icosahedron.mk 12 20 30 5 sorry) = 3 :=
  by sorry

end min_colors_icosahedron_l820_820606


namespace compute_sum_of_products_of_coefficients_l820_820049

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end compute_sum_of_products_of_coefficients_l820_820049


namespace student_b_speed_l820_820366

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820366


namespace vector_combination_l820_820063

noncomputable def a : ℝ × ℝ × ℝ := ⟨1, 1, -1⟩
noncomputable def b : ℝ × ℝ × ℝ := ⟨2, 0, -1⟩
noncomputable def c : ℝ × ℝ × ℝ := ⟨3, 3, 0⟩
noncomputable def v : ℝ × ℝ × ℝ := ⟨1, 4, -2⟩

noncomputable def dot_product (x y : ℝ × ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3

def mutually_orthogonal :=
  dot_product a b = 0 ∧ dot_product a c = 0 ∧ dot_product b c = 0

theorem vector_combination (hp hq hr : ℝ)
  (h : mutually_orthogonal) :
  v = (hp • a) + (hq • b) + (hr • c) :=
  sorry

example : vector_combination 1 0 (5 / 6) (by sorry) = true := 
  sorry

end vector_combination_l820_820063


namespace positive_difference_even_odd_sum_l820_820280

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820280


namespace magnitude_b_sqrt_3_l820_820666

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820666


namespace positive_difference_of_sums_l820_820289

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820289


namespace intersection_count_l820_820549

noncomputable def intersection_points (A : ℝ) (hA : A > 0) : Prop :=
  let f (x : ℝ) : ℝ := A * x ^ 2
  let g (y : ℝ) : ℝ := y ^ 2 + 3 = y ^ 2 + 4 * y
  ∃ (p : ℝ × ℝ), f p.1 = p.2 ∧ g p.1 p.2 = 0

theorem intersection_count (A : ℝ) (hA : A > 0) :
  intersection_points A hA → ∃ n, n = 4 :=
sorry

end intersection_count_l820_820549


namespace magnitude_b_sqrt_3_l820_820664

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820664


namespace decagon_winning_strategy_dodecagon_winning_strategy_l820_820318

-- Definition to specify the regular decagon and dodecagon
structure RegularPolygon (n : ℕ) :=
(vertices : list ℕ)
(color : ℕ → ℕ)
(vertices_count : vertices.length = n)
(alternate_colors : ∀ i, color (vertices.nth i % n) ≠ color (vertices.nth (i + 1) % n))

-- Definition of the game state
structure GameState :=
(polygon : RegularPolygon)
(segments : list (ℕ × ℕ))
(segments_valid : ∀ s ∈ segments, segments_disjoint s)

-- Condition for a valid move
def valid_move (state : GameState) (s : ℕ × ℕ) : Prop :=
state.polygon.color s.fst = state.polygon.color s.snd ∧
s ∉ state.segments ∧
segments_disjoint s state.segments

-- Winning strategy theorem for part (a)
theorem decagon_winning_strategy :
 ∀ (state : GameState),
 state.polygon.vertices_count = 10 → 
 ( ∃ strat : (GameState → option (ℕ × ℕ)),
 ∀ state, valid_move state (strat state).get → strat (update_state state (strat state).get) = none) ↔ 
 ∃ second_player_strategy : (GameState → option (ℕ × ℕ)), 
  ∀ state, valid_move state (second_player_strategy state).get :=
sorry

-- Winning strategy theorem for part (b)
theorem dodecagon_winning_strategy :
 ∀ (state : GameState),
 state.polygon.vertices_count = 12 → 
 ( ∃ strat : (GameState → option (ℕ × ℕ)),
 ∀ state, valid_move state (strat state).get → strat (update_state state (strat state).get) = none) ↔ 
 ∃ first_player_strategy : (GameState → option (ℕ × ℕ)), 
  ∀ state, valid_move state (first_player_strategy state).get :=
sorry

end decagon_winning_strategy_dodecagon_winning_strategy_l820_820318


namespace total_purchase_cost_l820_820522

variable (kg_nuts : ℝ) (kg_dried_fruits : ℝ)
variable (cost_per_kg_nuts : ℝ) (cost_per_kg_dried_fruits : ℝ)

-- Define the quantities
def cost_nuts := kg_nuts * cost_per_kg_nuts
def cost_dried_fruits := kg_dried_fruits * cost_per_kg_dried_fruits

-- The total cost can be expressed as follows
def total_cost := cost_nuts + cost_dried_fruits

theorem total_purchase_cost (h1 : kg_nuts = 3) (h2 : kg_dried_fruits = 2.5)
  (h3 : cost_per_kg_nuts = 12) (h4 : cost_per_kg_dried_fruits = 8) :
  total_cost kg_nuts kg_dried_fruits cost_per_kg_nuts cost_per_kg_dried_fruits = 56 := by
  sorry

end total_purchase_cost_l820_820522


namespace magnitude_b_sqrt_3_l820_820663

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820663


namespace proof_ABD_l820_820052

variable (f : ℤ → ℤ)

axiom f_1 : f 1 = 1
axiom f_2 : f 2 = 0
axiom f_neg1 : f (-1) < 0
axiom f_func : ∀ x y : ℤ, f (x + y) = f x * f (1 - y) + f (1 - x) * f y

theorem proof_ABD : f 0 = 0 ∧ (∀ x : ℤ, f (-x) = -f x) ∧ ∑ i in (Finset.range 100).map (λ i, i + 1), f i = 0 := 
by
  sorry

end proof_ABD_l820_820052


namespace train_passing_time_l820_820880

-- conditions
def train_length := 490 -- in meters
def train_speed_kmh := 63 -- in kilometers per hour
def conversion_factor := 1000 / 3600 -- to convert km/hr to m/s

-- conversion
def train_speed_ms := train_speed_kmh * conversion_factor -- speed in meters per second

-- expected correct answer
def expected_time := 28 -- in seconds

-- Theorem statement
theorem train_passing_time :
  train_length / train_speed_ms = expected_time :=
by
  sorry

end train_passing_time_l820_820880


namespace volume_pyramid_PQRS_l820_820082

-- Let's define our variables and conditions
variables {P Q R S : Type} [point P] [point Q] [point R] [point S]
variable (distance : P → R → ℝ)
variable (perpendicular : P → Q → Prop)
variable (rectangle_base : Q → R → Type)

-- Given Conditions
def QR : ℝ := 10
def RS : ℝ := 5
def PQ : ℝ := 20

-- Distance from P to RS is the height of the pyramid
axiom PQ_perp_RS : perpendicular P RS
axiom PQ_perp_QR : perpendicular P QR
axiom P_to_RS : distance P RS = PQ

-- Prove the volume of pyramid PQRS
theorem volume_pyramid_PQRS :
  let base_area := QR * RS,
      volume := (1 / 3 : ℝ) * base_area * PQ
  in volume = 1000 / 3 := by
  sorry

end volume_pyramid_PQRS_l820_820082


namespace find_speed_B_l820_820397

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820397


namespace positive_difference_eq_250_l820_820171

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820171


namespace will_initially_bought_seven_boxes_l820_820314

theorem will_initially_bought_seven_boxes :
  let given_away_pieces := 3 * 4
  let total_initial_pieces := given_away_pieces + 16
  let initial_boxes := total_initial_pieces / 4
  initial_boxes = 7 := 
by
  sorry

end will_initially_bought_seven_boxes_l820_820314


namespace positive_difference_even_odd_sums_l820_820262

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820262


namespace find_symmetric_point_l820_820992

def Point := (ℝ × ℝ × ℝ)
def Plane := (ℝ × ℝ × ℝ) × ℝ

def symmetric_point (M : Point) (plane : Plane) : Point :=
  let (a, b, c) := plane.fst
  let d := plane.snd
  let t := ((a * M.1 + b * M.2 + c * M.3 + d) / (a^2 + b^2 + c^2))
  (M.1 - 2 * a * t, M.2 - 2 * b * t, M.3 - 2 * c * t)

theorem find_symmetric_point :
  let M := (0, -3, -2)
  let plane := ((2, 10, 10), -1)
  symmetric_point M plane = (1, 2, 3) :=
by
  sorry

end find_symmetric_point_l820_820992


namespace monotone_increasing_solve_inequality_l820_820774

section MathProblem

variable {f : ℝ → ℝ}

theorem monotone_increasing (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₂ : ∀ x : ℝ, 1 < x → 0 < f x) : 
∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := sorry

theorem solve_inequality (h₃ : f 2 = 1) (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₅ : ∀ x : ℝ, 1 < x → 0 < f x) :
∀ x : ℝ, 0 < x → f x + f (x - 3) ≤ 2 → 3 < x ∧ x ≤ 4 := sorry

end MathProblem

end monotone_increasing_solve_inequality_l820_820774


namespace positive_difference_sums_l820_820220

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820220


namespace positive_difference_sums_l820_820212

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820212


namespace positive_difference_sums_even_odd_l820_820235

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820235


namespace magnitude_of_b_l820_820651

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820651


namespace positive_difference_even_odd_sums_l820_820302

theorem positive_difference_even_odd_sums :
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := 2 * (25 * 26 / 2)
  let sum_odd_20 := 20 * 20
  have h_sum_even_25 : sum_even_25 = 650 := by
    sorry
  have h_sum_odd_20 : sum_odd_20 = 400 := by
    sorry
  have h_diff : sum_even_25 - sum_odd_20 = 250 := by
    rw [h_sum_even_25, h_sum_odd_20]
    sorry
  exact h_diff

end positive_difference_even_odd_sums_l820_820302


namespace angle_of_line_AB_is_60_degrees_l820_820609

theorem angle_of_line_AB_is_60_degrees :
  ∀ (A B : ℕ × ℚ), 
    A = (2, 0) →
    B = (3, Real.sqrt 3) →
    ∃ α : ℝ, α = 60 :=
by
  sorry

end angle_of_line_AB_is_60_degrees_l820_820609


namespace find_number_of_socks_l820_820786

theorem find_number_of_socks :
  ∃ L : ℕ, (L + 20 + (20 / 5) + (3 * L + 8) = 80) ∧ L = 12 :=
by
  use 12
  sorry

end find_number_of_socks_l820_820786


namespace find_x_l820_820869

theorem find_x (x : ℤ) (h : 2 * x = (26 - x) + 19) : x = 15 :=
by
  sorry

end find_x_l820_820869


namespace bicycle_speed_B_l820_820390

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820390


namespace student_b_speed_l820_820370

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820370


namespace union_A_B_l820_820612

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 > 1}

-- Prove the union of A and B is the expected result
theorem union_A_B : A ∪ B = {x | x ≤ 0 ∨ x > 1} :=
by
  sorry

end union_A_B_l820_820612


namespace sum_of_terms_l820_820890

-- Defining the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Given conditions
theorem sum_of_terms (a d : ℕ) (h : (a + 3 * d) + (a + 11 * d) = 20) :
  12 * (a + 11 * d) / 2 = 60 :=
by
  sorry

end sum_of_terms_l820_820890


namespace positive_difference_even_odd_sums_l820_820264

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820264


namespace find_speed_B_l820_820400

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820400


namespace speed_of_student_B_l820_820426

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820426


namespace positive_difference_even_odd_sums_l820_820252

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820252


namespace angle_B_of_triangle_l820_820594

theorem angle_B_of_triangle {A B C a b c : ℝ} (h1 : b^2 = a * c) (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) : 
  B = Real.pi / 3 :=
sorry

end angle_B_of_triangle_l820_820594


namespace positive_difference_eq_250_l820_820170

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820170


namespace smallest_five_digit_palindrome_divisible_by_5_correct_l820_820867

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def smallest_five_digit_palindrome_divisible_by_5 : ℕ :=
  50005

theorem smallest_five_digit_palindrome_divisible_by_5_correct :
  is_palindrome smallest_five_digit_palindrome_divisible_by_5 ∧
  is_five_digit smallest_five_digit_palindrome_divisible_by_5 ∧
  is_divisible_by_5 smallest_five_digit_palindrome_divisible_by_5 ∧
  (∀ n, is_palindrome n ∧ is_five_digit n ∧ is_divisible_by_5 n → smallest_five_digit_palindrome_divisible_by_5 ≤ n) :=
by
  -- Proof here
  sorry

end smallest_five_digit_palindrome_divisible_by_5_correct_l820_820867


namespace magnitude_of_b_l820_820645

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820645


namespace find_speed_of_B_l820_820362

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820362


namespace find_speed_of_B_l820_820413

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820413


namespace num_factors_x11_sub_x_l820_820563

example : polynomial ℤ :=
begin
  sorry
end

theorem num_factors_x11_sub_x :
  ∃ (f g h k : polynomial ℤ), (polynomial.X ^ 11 - polynomial.X) = f * g * h * k ∧
  f.monic ∧ g.monic ∧ h.monic ∧ k.monic ∧
  f.degree = 1 ∧ g.degree = 1 ∧ h.degree = 4 ∧ k.degree = 5 :=
by {
  sorry
}

# eval num_factors_x11_sub_x

end num_factors_x11_sub_x_l820_820563


namespace student_B_speed_l820_820456

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820456


namespace hyperbola_through_C_l820_820750

noncomputable def equation_of_hyperbola_passing_through_C : Prop :=
  let A := (-1/2, 1/4)
  let B := (2, 4)
  let C := (-1/2, 4)
  ∃ (k : ℝ), k = -2 ∧ (∀ x : ℝ, x ≠ 0 → x * (4) = k)

theorem hyperbola_through_C :
  equation_of_hyperbola_passing_through_C :=
by
  sorry

end hyperbola_through_C_l820_820750


namespace energy_savings_l820_820021

theorem energy_savings (x y : ℝ) 
  (h1 : x = y + 27) 
  (h2 : x + 2.1 * y = 405) :
  x = 149 ∧ y = 122 :=
by
  sorry

end energy_savings_l820_820021


namespace boys_together_girls_together_arrangement_boys_and_girls_alternate_arrangement_boy_A_girl_B_order_arrangement_l820_820144

-- Definitions for the number of students, boys, and girls
def num_students := 7
def boys := 4
def girls := 3

-- Definitions for specific scenario conditions
def all_boys_together_all_girls_together : Prop :=
  ∀ (total_arrangements : ℕ),
  total_arrangements = factorial 2 * factorial 4 * factorial 3

def boys_and_girls_alternate : Prop :=
  ∀ (total_arrangements : ℕ),
  total_arrangements = factorial 4 * factorial 3

def boy_A_girl_B_certain_order : Prop :=
  ∀ (total_arrangements : ℕ),
  total_arrangements = nat.choose 7 2 * factorial 5

-- Main theorem statements
theorem boys_together_girls_together_arrangement :
  all_boys_together_all_girls_together := by
  sorry

theorem boys_and_girls_alternate_arrangement :
  boys_and_girls_alternate := by
  sorry

theorem boy_A_girl_B_order_arrangement :
  boy_A_girl_B_certain_order := by
  sorry

end boys_together_girls_together_arrangement_boys_and_girls_alternate_arrangement_boy_A_girl_B_order_arrangement_l820_820144


namespace magnitude_b_eq_sqrt3_l820_820678

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820678


namespace count_two_digit_numbers_l820_820935

theorem count_two_digit_numbers : 
  (∑ i in finset.range 10, ∑ j in finset.range 10, if i < j then 1 else 0) = 36 :=
by
  sorry

end count_two_digit_numbers_l820_820935


namespace student_B_speed_l820_820512

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820512


namespace perpendicular_tangent_line_l820_820622

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2

theorem perpendicular_tangent_line :
  ∃ (m : ℝ), (x + 4 * y = 0) ∧ (∀ (x : ℝ), x > 0 → (Real.log x + 2 * x) ≥ 2 * Real.sqrt 2) →
  ((Real.log x + 2 * x = 2 * Real.sqrt 2) → (¬ 0 < -1 / 4 < - Real.sqrt 2 / 4)) :=
sorry

end perpendicular_tangent_line_l820_820622


namespace relationship_among_a_b_c_l820_820773

noncomputable def f : ℝ → ℝ := sorry -- We assume f is a differentiable function on ℝ

variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]

-- Conditions
variable (h : ∀ x : ℝ, f x + x * (deriv f x) < 0)
def a := 2 * (f 2)
def b := f 1
def c := -(f (-1))

-- Proof Statement
theorem relationship_among_a_b_c : c > b ∧ b > a :=
by
  -- The proof here would be filled in based on the correct answer derived
  sorry

end relationship_among_a_b_c_l820_820773


namespace equation_has_one_real_root_l820_820105

theorem equation_has_one_real_root :
  ∃! x : ℝ, x + real.sqrt (x - 2) = 4 := 
sorry

end equation_has_one_real_root_l820_820105


namespace johnny_hours_worked_l820_820032

-- Define Johnny's hourly wage and total earnings
def hourly_wage : ℝ := 2.35
def total_earnings : ℝ := 11.75

-- Define the number of hours worked
def hours_worked : ℝ := total_earnings / hourly_wage

-- Statement of the problem: Prove that hours_worked is 5
theorem johnny_hours_worked : hours_worked = 5 := by
  sorry

end johnny_hours_worked_l820_820032


namespace student_B_speed_l820_820474

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820474


namespace circle_line_intersection_l820_820626

-- Define the circle C and its standard form
def circle_C_std_form : Prop :=
  ∀ x y : ℝ, (x + 4)^2 + y^2 = 1 ↔ x^2 + y^2 + 8x + 15 = 0

-- Define the inequality condition for the distance from the point to the line
def point_line_distance_inequality (k : ℝ) : Prop :=
  (abs (-4 * k - 2)) / (Real.sqrt (k^2 + 1)) ≤ 2

-- Define the proof problem statement
theorem circle_line_intersection (k : ℝ) :
  (-4 / 3 ≤ k ∧ k ≤ 0) ↔
    (∃ x y : ℝ, (x + 4)^2 + y^2 = 1 ∧ y = k * x - 2 ∧
      (abs (-4 * k - 2)) / (Real.sqrt (k^2 + 1)) ≤ 2) :=
by
  sorry

end circle_line_intersection_l820_820626


namespace average_speed_of_train_l820_820930

theorem average_speed_of_train
  (distance1 : ℝ) (time1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 240) (h2 : time1 = 3) (h3 : stop_time = 0.5)
  (h4 : distance2 = 450) (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = 81.18 := 
sorry

end average_speed_of_train_l820_820930


namespace student_B_speed_l820_820449

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820449


namespace jose_profit_share_correct_l820_820148

def tom_investment := 30000
def jose_investment_usd := 1000
def jose_exchange_rate_initial := 45
def angela_investment_gbp := 800
def angela_exchange_rate_initial := 95
def rebecca_investment_euro := 600
def rebecca_exchange_rate_initial := 80
def total_profit := 144000
def jose_share_percentage := 20
def end_of_year_exchange_rate_usd := 47

-- Convert investments to Indian Rupees
def jose_investment_inr := jose_investment_usd * jose_exchange_rate_initial
def angela_investment_inr := angela_investment_gbp * angela_exchange_rate_initial
def rebecca_investment_inr := rebecca_investment_euro * rebecca_exchange_rate_initial

-- Total investment amount
def total_investment_inr := tom_investment + jose_investment_inr + angela_investment_inr + rebecca_investment_inr

-- Calculate Jose's profit share in Indian Rupees
def jose_share_profit_inr := (jose_share_percentage / 100) * total_profit

-- Convert Jose's share to US Dollars using end-of-year exchange rate
def jose_share_profit_usd := jose_share_profit_inr / end_of_year_exchange_rate_usd

theorem jose_profit_share_correct : jose_share_profit_usd = 612.77 := by
  sorry

end jose_profit_share_correct_l820_820148


namespace weight_of_daughter_l820_820120

def mother_daughter_grandchild_weight (M D C : ℝ) :=
  M + D + C = 130 ∧
  D + C = 60 ∧
  C = 1/5 * M

theorem weight_of_daughter (M D C : ℝ) 
  (h : mother_daughter_grandchild_weight M D C) : D = 46 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_of_daughter_l820_820120


namespace speed_of_student_B_l820_820423

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820423


namespace student_B_speed_l820_820505

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820505


namespace integral_evaluation_l820_820621

noncomputable def coefficient_of_term (m : ℤ) : Prop :=
∃ a b : ℤ, (a + b = 5) ∧ ((a = 2 ∧ b = 4) ∨ (a = 2 ∧ b = 4)) ∧ m = -5

theorem integral_evaluation (m : ℤ) (h : coefficient_of_term m) :
  ∫ x in 1..2, x^m + 1/x = Real.log 2 + 15/64 :=
by
  sorry

end integral_evaluation_l820_820621


namespace polygon_sides_l820_820922

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end polygon_sides_l820_820922


namespace extra_flour_l820_820065

-- Define the conditions
def recipe_flour : ℝ := 7.0
def mary_flour : ℝ := 9.0

-- Prove the number of extra cups of flour Mary puts in
theorem extra_flour : mary_flour - recipe_flour = 2 :=
by
  sorry

end extra_flour_l820_820065


namespace cost_equiv_banana_pear_l820_820947

-- Definitions based on conditions
def Banana : Type := ℝ
def Apple : Type := ℝ
def Pear : Type := ℝ

-- Given conditions
axiom cost_equiv_1 : 4 * (Banana : ℝ) = 3 * (Apple : ℝ)
axiom cost_equiv_2 : 9 * (Apple : ℝ) = 6 * (Pear : ℝ)

-- Theorem to prove
theorem cost_equiv_banana_pear : 24 * (Banana : ℝ) = 12 * (Pear : ℝ) :=
by
  sorry

end cost_equiv_banana_pear_l820_820947


namespace cubic_roots_l820_820158

theorem cubic_roots (a b c : ℝ) :
  (b = a^2 / 3) ∨ (c = ab / 3 - 2a^3 / 27) →
  (∀ y, (y = (a^3 - 27c) ^ (1/3) → is_root (λ x, x^3 + a * x^2 + b * x + c) y) ∨
    (y = -a / 3) ∨ (y = sqrt (a^2 / 3 - b) ∨ y = -sqrt (a^2 / 3 - b)) → is_root (λ x, x^3 + a * x^2 + b * x + c) y)
  :=
sorry

end cubic_roots_l820_820158


namespace bicycle_speed_B_l820_820386

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820386


namespace no_broken_line_exists_l820_820026

def region_graph := { vertices : Type, edges : set (vertices × vertices) }

noncomputable def eulerian_path_exists (G : region_graph) : Prop :=
  let degree (v: G.vertices) := (G.edges.filter (λ e, e.fst = v ∨ e.snd = v)).card in
  let odd_degrees := (G.vertices.filter (λ v, degree v % 2 = 1)).card in
  odd_degrees = 0 ∨ odd_degrees = 2

theorem no_broken_line_exists :
  let G : region_graph := {
    vertices := {0, 1, 2, 3, 4, 5},
    edges := {(0,1), (0,2), (0,3), (0,4), (0,5), (1,2), (2,3), (3,4), (4,5)}
  }
  let degree (v: G.vertices) := (G.edges.filter (λ e, e.fst = v ∨ e.snd = v)).card in
  degree 0 = 5 ∧ (degree 1 = 5 ∨ degree 1 = 9) ∧ (degree 2 = 5 ∨ degree 2 = 9) ∧
  (degree 3 = 5 ∨ degree 3 = 9) ∧ (degree 4 = 5 ∨ degree 4 = 9) ∧ (degree 5 = 5 ∨ degree 5 = 9) →
  ¬ eulerian_path_exists G :=
begin
  intro h,
  sorry,
end

end no_broken_line_exists_l820_820026


namespace intervals_of_increase_and_decrease_y_eq_x2_intervals_of_increase_y_eq_x3_intervals_of_increase_f_x_eq_x3_plus_2x_minus_5_intervals_of_increase_and_decrease_y_eq_ln_x2_plus_2x_plus_3_intervals_of_increase_and_decrease_y_eq_2x2_minus_ln_x_intervals_of_decrease_y_eq_1_over_x_minus_2_l820_820573

open Real

noncomputable def intervals_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y
noncomputable def intervals_of_decrease (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem intervals_of_increase_and_decrease_y_eq_x2 :
  (intervals_of_increase (λ x, x^2) 0 ∞) ∧ (intervals_of_decrease (λ x, x^2) -∞ 0) := sorry

theorem intervals_of_increase_y_eq_x3 :
  (intervals_of_increase (λ x, x^3) -∞ ∞) := sorry

theorem intervals_of_increase_f_x_eq_x3_plus_2x_minus_5 :
  (intervals_of_increase (λ x, x^3 + 2 * x - 5) -∞ ∞) := sorry

theorem intervals_of_increase_and_decrease_y_eq_ln_x2_plus_2x_plus_3 :
  (intervals_of_increase (λ x, ln(x^2 + 2*x + 3)) -1 ∞) ∧
  (intervals_of_decrease (λ x, ln(x^2 + 2*x + 3)) -∞ -1) := sorry

theorem intervals_of_increase_and_decrease_y_eq_2x2_minus_ln_x :
  (intervals_of_increase (λ x, 2 * x^2 - ln x) (1/2) ∞) ∧ 
  (intervals_of_decrease (λ x, 2 * x^2 - ln x) 0 (1/2)) := sorry

theorem intervals_of_decrease_y_eq_1_over_x_minus_2 :
  (intervals_of_decrease (λ x, (1 / (x - 2))) -∞ 2) ∧ 
  (intervals_of_decrease (λ x, (1 / (x - 2))) 2 ∞) := sorry


end intervals_of_increase_and_decrease_y_eq_x2_intervals_of_increase_y_eq_x3_intervals_of_increase_f_x_eq_x3_plus_2x_minus_5_intervals_of_increase_and_decrease_y_eq_ln_x2_plus_2x_plus_3_intervals_of_increase_and_decrease_y_eq_2x2_minus_ln_x_intervals_of_decrease_y_eq_1_over_x_minus_2_l820_820573


namespace find_speed_of_B_l820_820411

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820411


namespace winning_ticket_probability_l820_820011

open BigOperators

-- Calculate n choose k
def choose (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def probability_PowerBall := (1 : ℚ) / 30
def probability_LuckyBalls := (1 : ℚ) / choose 49 6

-- Theorem to prove the result
theorem winning_ticket_probability :
  probability_PowerBall * probability_LuckyBalls = (1 : ℚ) / 419514480 := by
  sorry

end winning_ticket_probability_l820_820011


namespace line_intersects_circle_l820_820836

theorem line_intersects_circle (m : ℝ) : 
  (∃ x y : ℝ, x - y + m = 0 ∧ x^2 + y^2 - 2x - 1 = 0) ↔ (-3 < m ∧ m < 1) := 
by
  sorry

end line_intersects_circle_l820_820836


namespace plan_Y_cheaper_l820_820526

theorem plan_Y_cheaper (y : ℤ) :
  (15 * (y : ℚ) > 2500 + 8 * (y : ℚ)) ↔ y > 358 :=
by
  sorry

end plan_Y_cheaper_l820_820526


namespace positive_difference_of_sums_l820_820194

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820194


namespace z_in_quadrant_IV_l820_820625

-- Define the complex number z
def z : ℂ := (2 + complex.i) / (1 + complex.i)

-- Prove that z lies in the fourth quadrant (real part positive and imaginary part negative)
theorem z_in_quadrant_IV : z.re > 0 ∧ z.im < 0 := by
  sorry

end z_in_quadrant_IV_l820_820625


namespace positive_difference_l820_820222

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820222


namespace find_speed_B_l820_820394

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820394


namespace parabola_shift_l820_820720

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l820_820720


namespace student_b_speed_l820_820443

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820443


namespace positive_difference_even_odd_sum_l820_820276

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820276


namespace positive_difference_sums_even_odd_l820_820244

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820244


namespace max_abs_diff_f_l820_820628

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f {k x1 x2 : ℝ} (hk : -3 ≤ k ∧ k ≤ -1) 
    (hx1 : k ≤ x1 ∧ x1 ≤ k + 2) (hx2 : k ≤ x2 ∧ x2 ≤ k + 2) : 
    |f x1 - f x2| ≤ 4 * Real.exp 1 := 
sorry

end max_abs_diff_f_l820_820628


namespace log3_9_is_2_two_pow_a_is_sqrt3_l820_820893

noncomputable def log3_9 : ℝ :=
  real.logb 3 9

noncomputable def a : ℝ :=
  real.logb 4 3

theorem log3_9_is_2 : log3_9 = 2 :=
sorry

theorem two_pow_a_is_sqrt3 (h : a = real.logb 4 3) : 2^a = real.sqrt 3 :=
sorry

end log3_9_is_2_two_pow_a_is_sqrt3_l820_820893


namespace speed_of_student_B_l820_820424

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820424


namespace magnitude_of_b_l820_820690

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820690


namespace problem_solution_l820_820769

theorem problem_solution :
  (∑ n in Finset.range (2014) + 2, (binomial n 2) / 3^n) /
  ((binomial 2016 3) / 3) = 1 / 54 := 
begin
  sorry
end

end problem_solution_l820_820769


namespace largest_number_is_34_l820_820135

theorem largest_number_is_34 (a b c : ℕ) (h1 : a + b + c = 82) (h2 : c - b = 8) (h3 : b - a = 4) : c = 34 := 
by 
  sorry

end largest_number_is_34_l820_820135


namespace find_p_l820_820887

theorem find_p (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) ∧ (m + p = (n + 15) / 5 - 2 / 5) → p = 3 := 
by 
  intro h,
  cases h with h1 h2,
  have : p = ((n + 15) / 5 - 2 / 5) - (n / 5 - 2 / 5),
  { rw [← h1, h2],
    ring },
  field_simp at this,
  linarith,

end find_p_l820_820887


namespace positive_difference_even_odd_l820_820205

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820205


namespace positive_difference_even_odd_sums_l820_820257

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820257


namespace jessica_carrots_l820_820030

theorem jessica_carrots
  (joan_carrots : ℕ)
  (total_carrots : ℕ)
  (jessica_carrots : ℕ) :
  joan_carrots = 29 →
  total_carrots = 40 →
  jessica_carrots = total_carrots - joan_carrots →
  jessica_carrots = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jessica_carrots_l820_820030


namespace positive_difference_even_odd_sums_l820_820263

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820263


namespace positive_difference_even_odd_l820_820204

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820204


namespace rectangle_ratio_l820_820585

theorem rectangle_ratio (a : ℝ) (y : ℝ) (x : ℝ) 
  (h1 : ∀ i, i < 5 → congruent_rectangle a y x)
  (h2 : area_outer_pentagon a x y = 5 * area_inner_pentagon a) :
  x / y = sqrt 5 - 1 :=
sorry

-- Assume dummy definitions for the sake of completeness
def congruent_rectangle (a y x : ℝ) : Prop := 
  -- dummy definition
  true

def area_outer_pentagon (a x y : ℝ) : ℝ := 
  -- dummy definition
  a * x * y 

def area_inner_pentagon (a : ℝ) : ℝ := 
  -- dummy definition
  a ^ 2

end rectangle_ratio_l820_820585


namespace area_intersections_equals_area_outside_l820_820007

theorem area_intersections_equals_area_outside 
  (R : ℝ) : 
  let πR2 := π * R^2,
      smaller_circle_area := π * (R / 2)^2 in
  let total_smaller_circles := 4 * smaller_circle_area,
      overlap_area := (π * (R / 2)^2) / 8 * 4 in
  let area_outside := πR2 - (total_smaller_circles - overlap_area) in
  overlap_area = area_outside :=
by
  sorry

end area_intersections_equals_area_outside_l820_820007


namespace student_B_speed_l820_820516

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820516


namespace price_reduction_final_percentage_l820_820927

theorem price_reduction_final_percentage :
  let p0 := 100.0 in
  let p1 := p0 * (1 - 0.08) in
  let p2 := p1 * (1 - 0.10) in
  let p3 := p2 * (1 - 0.05) in
  let p4 := p3 * (1 - 0.07) in
  p4 = 73.1538 :=
by
  sorry

end price_reduction_final_percentage_l820_820927


namespace parabola_transformation_l820_820718

theorem parabola_transformation :
  ∀ (x y : ℝ), (y = x^2) →
  let shiftedRight := (x - 3)
  let shiftedUp := (shiftedRight)^2 + 4
  y = shiftedUp :=
by
  intros x y,
  intro hyp,
  let shiftedRight := (x - 3),
  let shiftedUp := (shiftedRight)^2 + 4,
  have eq_shifted : y = shiftedRight^2 := by sorry,
  have eq_final : y = shiftedUp := by sorry,
  exact eq_final
  
end parabola_transformation_l820_820718


namespace positive_difference_of_sums_l820_820190

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820190


namespace opposite_of_fraction_l820_820839

theorem opposite_of_fraction : - (11 / 2022 : ℚ) = -(11 / 2022) := 
by
  sorry

end opposite_of_fraction_l820_820839


namespace positive_difference_sums_even_odd_l820_820243

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820243


namespace range_of_m_l820_820833

open Real

noncomputable def f : ℝ → ℝ :=
  λ x, 2 * sin (π / 4 + x) - sqrt 3 * cos (2 * x)

theorem range_of_m (t : ℝ) (m : ℝ) :
  (0 < t ∧ t < π) →
  (∀ x, (π / 4 ≤ x ∧ x ≤ π / 2) → abs (f x - m) < 3) →
  -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l820_820833


namespace find_a_and_vertices_find_y_range_find_a_range_l820_820602

noncomputable def quadratic_function (x a : ℝ) : ℝ :=
  x^2 - 6 * a * x + 9

theorem find_a_and_vertices (a : ℝ) :
  quadratic_function 2 a = 7 →
  a = 1 / 2 ∧
  (3 * a, quadratic_function (3 * a) a) = (3 / 2, 27 / 4) :=
sorry

theorem find_y_range (x a : ℝ) :
  a = 1 / 2 →
  -1 ≤ x ∧ x < 3 →
  27 / 4 ≤ quadratic_function x a ∧ quadratic_function x a ≤ 13 :=
sorry

theorem find_a_range (a : ℝ) (x1 x2 : ℝ) :
  (3 * a - 2 ≤ x1 ∧ x1 ≤ 5 ∧ 3 * a - 2 ≤ x2 ∧ x2 ≤ 5) →
  (x1 ≥ 3 ∧ x2 ≥ 3 → quadratic_function x1 a - quadratic_function x2 a ≤ 9 * a^2 + 20) →
  1 / 6 ≤ a ∧ a ≤ 1 :=
sorry

end find_a_and_vertices_find_y_range_find_a_range_l820_820602


namespace problem_part1_period_and_min_value_problem_part2_side_c_l820_820596

noncomputable def f (x : ℝ) : ℝ :=
  let a := (Real.sin (2 * x), 2 * Real.cos x)
  let b := (Real.sqrt 3, Real.cos x)
  a.1 * b.1 + a.2 * b.2 - 1

theorem problem_part1_period_and_min_value :
  (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ (∀ x : ℝ, f x ≥ -2) :=
by
  -- Proof that the smallest positive period is π and the minimum value is -2.
  sorry

theorem problem_part2_side_c {a b : ℝ} (h₁ : a = 2 * Real.sqrt 13) (h₂ : b = 8) :
  ∀ {c : ℝ}, f (real.pi/12) = Real.sqrt 3 → c = 2 ∨ c = 6 :=
by
  -- Proof that side c can only be 2 or 6 given the conditions.
  sorry

end problem_part1_period_and_min_value_problem_part2_side_c_l820_820596


namespace general_formula_S_T_formula_P_lt_T_l820_820784
-- Import the Mathlib library to bring in the necessary components
noncomputable theory

-- Define the sequence a_n and the sum of the first n terms S_n
def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else sorry  -- We leave the definition of a_n for n >= 2 as sorry 

def S (n : ℕ) : ℝ :=
  if n = 1 then 1 else sorry  -- We leave this as sorry, to be derived from a_n

-- Define the recurrence relation given in the conditions
example (n : ℕ) (h : 2 ≤ n) : a n + 2 * S n * S (n-1) = 0 :=
  sorry  -- This needs proof. We assume the recurrence relation holds.

-- Prove the general formula for S_n
theorem general_formula_S (n : ℕ) : S n = (1 - (-1)^n) / 2 :=
  sorry  -- Proof not provided, just the statement

-- Define the function f and sequence b_n
def f (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := f n + 1

-- Define the sequences P_n and T_n
def P (n : ℕ) : ℝ := 
  ∑ i in (Finset.range n).map (finset.embedding.to_embedding (λ i, i+1)), S i * S (i+1)

def T (n : ℕ) : ℕ := 
  ∑ i in (Finset.range n).map (finset.embedding.to_embedding (λ i, i+1)), b i * b (i+1)

-- Prove the formula for T_n
theorem T_formula (n : ℕ) : T n = 2^n - 1 :=
  sorry  -- Proof not provided, just the statement

-- Prove that P_n < T_n
theorem P_lt_T (n : ℕ) : P n < T n :=
  sorry  -- Proof not provided, just the statement

end general_formula_S_T_formula_P_lt_T_l820_820784


namespace student_B_speed_l820_820466

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820466


namespace minimize_quadratic_function_l820_820998

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 3

theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function(x) ≤ quadratic_function(y)) ∧ x = -4 :=
by
  sorry

end minimize_quadratic_function_l820_820998


namespace intersection_of_A_and_B_l820_820639

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_of_A_and_B_l820_820639


namespace positive_difference_eq_250_l820_820166

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820166


namespace sum_of_ages_of_cousins_l820_820758

noncomputable def is_valid_age_group (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (1 ≤ a) ∧ (a ≤ 9) ∧ (1 ≤ b) ∧ (b ≤ 9) ∧ (1 ≤ c) ∧ (c ≤ 9) ∧ (1 ≤ d) ∧ (d ≤ 9)

theorem sum_of_ages_of_cousins :
  ∃ (a b c d : ℕ), is_valid_age_group a b c d ∧ (a * b = 40) ∧ (c * d = 36) ∧ (a + b + c + d = 26) := 
sorry

end sum_of_ages_of_cousins_l820_820758


namespace positive_difference_even_odd_sum_l820_820278

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l820_820278


namespace semi_integer_tiling_l820_820157

theorem semi_integer_tiling (a b : ℝ) (H : ∀ (i j : ℕ), a ≥ i ∧ b ≥ j → 
  (∃ a' b', a' ∈ ℤ ∨ b' ∈ ℤ ∧ (i : ℝ) < a' ∧ a' ≤ (i + 1 : ℝ) ∧ (j : ℝ) < b' ∧ b' ≤ (j + 1 : ℝ))) : 
  a ∈ ℤ ∨ b ∈ ℤ :=
sorry

end semi_integer_tiling_l820_820157


namespace negation_of_proposition_l820_820837

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > b → a^2 > b^2) ↔ ∃ (a b : ℝ), a ≤ b ∧ a^2 ≤ b^2 :=
sorry

end negation_of_proposition_l820_820837


namespace find_a_l820_820000

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - (3 / 2) * x^2 + a * x + 4

-- The derivative of f with respect to x
def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 3 * x + a

-- Given conditions:
-- 1. The function f has a specific derivative as defined.
-- 2. The roots of f'(x) = 0 should be exactly -1 and 4 for the interval [-1, 4] to be monotonically decreasing.

theorem find_a (a : ℝ) (h : f' a (-1) = 0 ∧ f' a 4 = 0) : a = -4 :=
by
  -- Proof would go here
  sorry

end find_a_l820_820000


namespace magnitude_of_b_l820_820648

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define the conditions
axiom condition1 : ∥a - b∥ = real.sqrt 3
axiom condition2 : ∥a + b∥ = ∥2 • a - b∥

-- State the theorem with the correct answer
theorem magnitude_of_b : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820648


namespace positive_difference_l820_820228

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820228


namespace bicycle_speed_B_l820_820384

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820384


namespace positive_difference_of_sums_l820_820286

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820286


namespace parabola_shift_l820_820723

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l820_820723


namespace rotation_fixed_point_l820_820109

theorem rotation_fixed_point :
  ∃ (c : ℂ), ((λ z, ((-1 + complex.I * real.sqrt 3) * z + (-2 * real.sqrt 3 - 18 * complex.I)) / 2) c = c)
  ∧ c = real.sqrt 3 - 5 * complex.I :=
by
  sorry

end rotation_fixed_point_l820_820109


namespace bicycle_speed_B_l820_820381

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820381


namespace find_equation_of_tangent_and_perpendicular_line_l820_820570

noncomputable def equation_of_tangent_and_perpendicular_line : Prop :=
  ∃ a b : ℝ, 
  (b = a^3 + 3 * a^2 - 5) ∧ 
  (3 * a^2 + 6 * a = -3) ∧ 
  (y = -3 * (x + a) + b) ∧ 
  (∃ x y : ℝ, (2 * x - 6 * y + 1 = 0) ∧ (3 * x + y + 6 = 0))

theorem find_equation_of_tangent_and_perpendicular_line :
  equation_of_tangent_and_perpendicular_line :=
sorry

end find_equation_of_tangent_and_perpendicular_line_l820_820570


namespace initial_output_increase_l820_820840

noncomputable def initial_increase_percentage : ℝ :=
  let O := (1 : ℝ) in
  let new_output := O + P / 100 * O in
  let holiday_output := new_output * (1 + 30 / 100) in
  let restored_output := holiday_output * (1 - 30.07 / 100) in
  P

theorem initial_output_increase (P : ℝ) :
  (let O := (1 : ℝ) in
   let new_output := O + P / 100 * O in
   let holiday_output := new_output * (1 + 30 / 100) in
   let restored_output := holiday_output * (1 - 30.07 / 100) in
   restored_output = O) → 
   P = 10 :=
begin
  sorry
end

end initial_output_increase_l820_820840


namespace coffee_intake_on_saturday_l820_820347

theorem coffee_intake_on_saturday :
  ∀ (h g : ℕ) (k : ℕ),
    (8 * 3 = k) →
    (g * 4 = 2 * k) →
    g = 12 := by
  intros h g k H1 H2
  rw [mul_comm 8 3] at H1
  rw mul_comm at H2
  have H3: k = 24, from by linarith,
  sorry

end coffee_intake_on_saturday_l820_820347


namespace ellipse_equation_k1_k2_prod_line_equations_l820_820605

section EllipseProblem

variables {a b : ℝ} {k e : ℝ} 
variables (F₁ F₂ A B O : ℝ × ℝ)

-- Given conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (e a c : ℝ) : Prop := e = c / a
def is_line (l : ℝ × ℝ → Prop) (k m : ℝ) : Prop := ∀ p, l p → p.2 = k * p.1 + m
def perimeter_AF2B (A B F₂ : ℝ × ℝ) : ℝ → Prop := sorry
def triangle_area (A B O : ℝ × ℝ) (area : ℝ) : Prop := sorry
def passes_through (l : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop := l p

-- Specific problem conditions
def problem_conditions : Prop :=
  let a := 2 in let b := 1 in let e := sqrt 3 / 2 in
  eccentricity e a (e * a) ∧
  is_line (λ p, passes_through (λ p, ellipse p.1 p.2 a b) p) (1/2) 0 ∧
  perimeter_AF2B A B F₂ 8 ∧
  passes_through (λ p, is_line (λ p, passes_through (λ p, ellipse p.1 p.2 a b) p) (1/2) 0) F₁ ∧
  triangle_area A B O (sqrt 7 / 4)

-- Prove
theorem ellipse_equation : problem_conditions → ellipse 0 1 2 1 :=
sorry

theorem k1_k2_prod (k₁ k₂ : ℝ) : problem_conditions → k₁ * k₂ = 1/4 :=
sorry

theorem line_equations (m : ℝ) : problem_conditions → 
  (is_line (λ p, passes_through (λ p, ellipse p.1 p.2 2 1) p) (1/2) (1/2 * m) ∨
   is_line (λ p, passes_through (λ p, ellipse p.1 p.2 2 1) p) (1/2) (sqrt 7 / 2 * m)) :=
sorry

end EllipseProblem

end ellipse_equation_k1_k2_prod_line_equations_l820_820605


namespace number_of_beetles_in_sixth_jar_l820_820896

theorem number_of_beetles_in_sixth_jar :
  ∃ (x : ℕ), 
      (x + (x+1) + (x+2) + (x+3) + (x+4) + (x+5) + (x+6) + (x+7) + (x+8) + (x+9) = 150) ∧
      (2 * x ≥ x + 9) ∧
      (x + 5 = 16) :=
by {
  -- This is just the statement, the proof steps are ommited.
  -- You can fill in the proof here using Lean tactics as necessary.
  sorry
}

end number_of_beetles_in_sixth_jar_l820_820896


namespace sum_of_7a_and_3b_l820_820324

theorem sum_of_7a_and_3b (a b : ℤ) (h : a + b = 1998) : 7 * a + 3 * b ≠ 6799 :=
by sorry

end sum_of_7a_and_3b_l820_820324


namespace student_B_speed_l820_820464

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820464


namespace positive_difference_eq_250_l820_820168

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820168


namespace MaryIvanovna_count_incorrect_l820_820592

theorem MaryIvanovna_count_incorrect:
  ∀ (k p : ℕ),
  -- Morning count: 2k + 15 (museum and café) must be odd
  -- Evening count: 2p + 8 (park and theater) must be even
  (∃ k : ℕ, odd (2*k + 15)) ∧ (∃ p : ℕ, even (2*p + 8)) :=
by
  sorry

end MaryIvanovna_count_incorrect_l820_820592


namespace exterior_angle_decreases_l820_820958

theorem exterior_angle_decreases (n : ℕ) (hn : n ≥ 3) : 
  (360 / n) > (360 / (n + 1)) := 
begin
  have h1 : n > 0, by linarith,
  have h2 : n + 1 > n, by linarith,
  calc
  360 / n > 360 / (n + 1) : by {
    apply div_lt_div_of_lt,
    all_goals {linarith},
  }
end

end exterior_angle_decreases_l820_820958


namespace find_initial_volume_l820_820338

-- Definitions for the conditions of the problem
def initial_volume (S : Type) [Group S] [Add S] (V : ℝ) (initial_percent : ℝ → ℝ) := initial_percent V
def final_volume (S : Type) [Group S] [Add S] (V : ℝ) (alcohol_added : ℝ) := V + alcohol_added
def alcohol_amount_initial (S : Type) [Group S] [Add S] (V : ℝ) := 0.25 * V
def alcohol_amount_final (S : Type) [Group S] [Add S] (V : ℝ) (alcohol_added : ℝ) := 0.25 * V + alcohol_added
def final_percent_alcohol (S : Type) [Group S] [Add S] (final_alcohol_volume : ℝ) (total_final_volume : ℝ) := final_alcohol_volume / total_final_volume

-- Lean statement for the proof problem
theorem find_initial_volume (V : ℝ) (initial_percent : ℝ) (alcohol_added : ℝ)
  (h1 : initial_percent = 0.25) (h2 : alcohol_added = 3) 
  (h3 : alcohol_amount_final V alcohol_added = 0.50 * final_volume V alcohol_added) :
  V = 6 :=
by
  -- Using proofs from defined conditions here
  sorry

end find_initial_volume_l820_820338


namespace clock_angle_acute_probability_l820_820799

noncomputable def probability_acute_angle : ℚ := 1 / 2

theorem clock_angle_acute_probability :
  ∀ (hour minute : ℕ), (hour >= 0 ∧ hour < 12) →
  (minute >= 0 ∧ minute < 60) →
  (let angle := min (60 * hour - 11 * minute) (720 - (60 * hour - 11 * minute)) in angle < 90 ↔ probability_acute_angle = 1 / 2) :=
sorry

end clock_angle_acute_probability_l820_820799


namespace maximize_perimeter_l820_820962

inductive Square : Type
| a | b | c | d | e | f | g | j | k | l | m | n
  deriving DecidableEq, Repr

open Square

def is_connected (shape : set Square) : Prop := sorry  -- Define connectivity

noncomputable def perimeter_increase (sq1 sq2 : Square) : ℤ := sorry -- Define perimeter computation

def candidate_pairs : set (Square × Square) :=
  { (d, k), (e, k) }

theorem maximize_perimeter (s1 s2 : Square) (H : (s1, s2) ∈ candidate_pairs) :
  ∀ (s t : Square), is_connected (candidate_pairs \ {s, t}) → 
  perimeter_increase s1 s2 ≥ perimeter_increase s t :=
sorry -- proof to be filled in

end maximize_perimeter_l820_820962


namespace student_B_speed_l820_820460

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820460


namespace hexagonal_pyramid_vertex_angle_l820_820124

theorem hexagonal_pyramid_vertex_angle (O A B C D E F : ℝ) (α : ℝ) (l : ℝ) 
  (O1 : ℝ) (FOE : ℝ) 
  (angle_eq : ∠FOE = ∠OFO1)
  (perpendicular : O1 = orthogonal_projection (span ({A, B, C, D, E, F} : set ℝ)) O)
  (OF_length : ∥O - F∥ = l)
  (FOE_angle_eq : FOE = α) :
  α = 2 * arcsin ((sqrt 3 - 1) / 2) :=
  sorry

end hexagonal_pyramid_vertex_angle_l820_820124


namespace exams_left_wednesday_l820_820792

-- Defining constants and intermediate computations as the provided conditions.
constant total_exams : ℕ := 120
constant monday_percentage : ℚ := 0.60
constant tuesday_percentage : ℚ := 0.75

-- Calculations according to the problem.
def exams_graded_monday : ℕ := total_exams * monday_percentage
def remaining_after_monday : ℕ := total_exams - exams_graded_monday
def exams_graded_tuesday : ℕ := remaining_after_monday * tuesday_percentage
def remaining_after_tuesday : ℕ := remaining_after_monday - exams_graded_tuesday

theorem exams_left_wednesday : remaining_after_tuesday = 12 := by
  sorry

end exams_left_wednesday_l820_820792


namespace magnitude_b_eq_sqrt3_l820_820679

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥2 • a - b∥

-- The theorem to be proved
theorem magnitude_b_eq_sqrt3 (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_eq_sqrt3_l820_820679


namespace trey_uses_47_nails_l820_820856

variable (D : ℕ) -- total number of decorations
variable (nails thumbtacks sticky_strips : ℕ)

-- Conditions
def uses_nails := nails = (5 * D) / 8
def uses_thumbtacks := thumbtacks = (9 * D) / 80
def uses_sticky_strips := sticky_strips = 20
def total_decorations := (21 * D) / 80 = 20

-- Question: Prove that Trey uses 47 nails when the conditions hold
theorem trey_uses_47_nails (D : ℕ) (h1 : uses_nails D nails) (h2 : uses_thumbtacks D thumbtacks) (h3 : uses_sticky_strips sticky_strips) (h4 : total_decorations D) : nails = 47 :=  
by
  sorry

end trey_uses_47_nails_l820_820856


namespace proof_main_l820_820620

noncomputable def proof_problem : Prop :=
let c := sqrt 12 in
let ellipse := ∀ (x y : ℝ), x^2 / 24 + y^2 / 12 = 1
in let line := ∀ (x y : ℝ), y = x + 3
in  (∃ (A B : ℝ × ℝ), A ≠ B ∧ ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ (A.1 + B.1) / 2 = -2 ∧ (A.2 + B.2) / 2 = 1 ∧ dist (A.1, A.2) (B.1, B.2) = 4*sqrt 3) →
ellipse ∧ line

theorem proof_main : proof_problem :=
begin
  sorry
end

end proof_main_l820_820620


namespace Kim_sold_saplings_l820_820036

theorem Kim_sold_saplings :
  ∀ (total_pits : ℕ) (percentage_sprouted : ℝ) (saplings_left : ℕ),
    total_pits = 80 →
    percentage_sprouted = 0.25 →
    saplings_left = 14 →
    (total_pits * percentage_sprouted).to_nat - saplings_left = 6 :=
by
  intros total_pits percentage_sprouted saplings_left h_total h_percentage h_left
  sorry

end Kim_sold_saplings_l820_820036


namespace speed_of_student_B_l820_820500

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820500


namespace student_B_speed_l820_820461

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820461


namespace student_b_speed_l820_820433

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820433


namespace arth_seq_val_a7_l820_820588

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arth_seq_val_a7 {a : ℕ → ℝ} 
  (h_arith : arithmetic_sequence a)
  (h_positive : ∀ n : ℕ, 0 < a n)
  (h_eq : 2 * a 6 + 2 * a 8 = (a 7) ^ 2) :
  a 7 = 4 := 
by sorry

end arth_seq_val_a7_l820_820588


namespace prove_inequalities_l820_820640

variable {a b c R r_a r_b r_c : ℝ}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_circumradius (a b c R : ℝ) : Prop :=
  ∃ S : ℝ, S = a * b * c / (4 * R)

def has_exradii (a b c r_a r_b r_c : ℝ) : Prop :=
  ∃ S : ℝ, 
    r_a = 2 * S / (b + c - a) ∧
    r_b = 2 * S / (a + c - b) ∧
    r_c = 2 * S / (a + b - c)

theorem prove_inequalities
  (h_triangle : is_triangle a b c)
  (h_circumradius : has_circumradius a b c R)
  (h_exradii : has_exradii a b c r_a r_b r_c)
  (h_two_R_le_r_a : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := 
sorry

end prove_inequalities_l820_820640


namespace blocks_collection_time_l820_820066

theorem blocks_collection_time:
  ∀ (total_blocks dad_rate mia_rate brother_rate block_target : ℕ),
  total_blocks = 50 →
  dad_rate = 5 →
  mia_rate = -3 →
  brother_rate = -1 →
  block_target = 50 → 
  ((total_blocks - block_target) * 30 / (dad_rate + mia_rate + brother_rate + dad_rate / 5)) / 60 = 23 := 
by
  intros _ _ _ _ _ H1 H2 H3 H4 H5
  sorry

end blocks_collection_time_l820_820066


namespace sum_of_squares_l820_820845

variable {x y : ℝ}

theorem sum_of_squares (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 :=
sorry

end sum_of_squares_l820_820845


namespace valid_side_probability_l820_820753

section triangle_side_probability

variable (AB AC : ℝ) (segments : Set ℝ)

def valid_BC (BC : ℝ) : Prop :=
  (AB + AC > BC) ∧ (AC + BC > AB) ∧ (AB + BC > AC)

theorem valid_side_probability :
  AB = 8 → AC = 6 →
  segments = {3, 6, 9, 12, 15} →
  (segments.filter (λ BC => valid_BC AB AC BC)).card = 4 →
  (segments.filter (λ BC => valid_BC AB AC BC)).card.to_rat / segments.card.to_rat = (4 / 5 : ℚ) :=
by
  intros h1 h2 h3 h4
  rw [h3]
  have h_card_segments : (finset.to_set finset.univ : Set ℝ).card = 5 := sorry
  rw [h_card_segments]
  rw [h4]
  norm_num

end triangle_side_probability

end valid_side_probability_l820_820753


namespace stratified_sampling_type_d_units_l820_820342

theorem stratified_sampling_type_d_units (prod_A prod_B prod_C prod_D total_selected : ℕ):
  prod_A = 100 → 
  prod_B = 200 → 
  prod_C = 300 → 
  prod_D = 400 →
  total_selected = 50 →
  (prod_D * total_selected) / (prod_A + prod_B + prod_C + prod_D) = 20 := 
by 
  intros hA hB hC hD hT
  rw [hA, hB, hC, hD, hT]
  norm_num
  sorry

end stratified_sampling_type_d_units_l820_820342


namespace stool_height_is_correct_l820_820933

-- Definitions from conditions
def ceiling_height_meters := 3
def light_bulb_below_ceiling_cm := 15
def alice_height_meters := 1.6
def alice_reach_above_head_cm := 50

-- Derived definitions
def ceiling_height_cm := ceiling_height_meters * 100
def alice_height_cm := alice_height_meters * 100
def alice_total_reach_cm := alice_height_cm + alice_reach_above_head_cm
def light_bulb_height_cm := ceiling_height_cm - light_bulb_below_ceiling_cm

-- The statement to prove the stool height
def stool_height_cm := light_bulb_height_cm - alice_total_reach_cm

-- Theorem to assert the stool height is 75 cm
theorem stool_height_is_correct : stool_height_cm = 75 := by
  sorry

end stool_height_is_correct_l820_820933


namespace range_of_a_l820_820129

def quadratic_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f (2 - x)

def quadratic_conditions (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f(a) ≤ f(0) ∧ f(0) ≤ f(1)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_sym : quadratic_symmetric f)
  (h_cond : quadratic_conditions f a) :
  a ≤ 0 ∨ a ≥ 4 :=
sorry

end range_of_a_l820_820129


namespace rate_of_premium_is_correct_l820_820928

-- Define the conditions
def original_value : ℝ := 87500
def insured_extent : ℝ := 4 / 5
def premium : ℝ := 910

-- Calculate the insured value
def insured_value : ℝ := insured_extent * original_value

-- Prove the rate of premium
def rate_of_premium : ℝ := (premium / insured_value) * 100

theorem rate_of_premium_is_correct : rate_of_premium = 1.3 := 
by 
  unfold rate_of_premium insured_value insured_extent original_value premium
  have h1 : insured_value = 4 / 5 * 87500 := rfl
  have h2 : insured_value = 70000 := by norm_num
  have h3 : rate_of_premium = (910 / 70000) * 100 := by norm_num
  have h4 : rate_of_premium = 1.3 := by norm_num
  exact h4

-- Add sorry to the place where we halted calculations
-- To pass the theorem just for the skeleton

end rate_of_premium_is_correct_l820_820928


namespace inequality_property_l820_820595

variable {a b : ℝ} (h : a > b) (c : ℝ)

theorem inequality_property : a * |c| ≥ b * |c| :=
sorry

end inequality_property_l820_820595


namespace treasure_location_l820_820914

theorem treasure_location
  (A B C1 C2 C3 A1 B1 P1 P2 P3 : Point)
  (h_eq_C1A1 : dist C1 A1 = dist C1 A)
  (h_perp_C1A1 : ∠ (line C1 A) (line C1 A1) = 90)
  (h_eq_C1B1 : dist C1 B1 = dist C1 B)
  (h_perp_C1B1 : ∠ (line C1 B) (line C1 B1) = 90)
  (h_P1 : intersection (line A B1) (line B A1) = P1)
  (h_eq_C2A2 : dist C2 A2 = dist C2 A)
  (h_perp_C2A2 : ∠ (line C2 A) (line C2 A2) = 90)
  (h_eq_C2B2 : dist C2 B2 = dist C2 B)
  (h_perp_C2B2 : ∠ (line C2 B) (line C2 B2) = 90)
  (h_P2 : intersection (line A B2) (line B A2) = P2)
  (h_eq_C3A3 : dist C3 A3 = dist C3 A)
  (h_perp_C3A3 : ∠ (line C3 A) (line C3 A3) = 90)
  (h_eq_C3B3 : dist C3 B3 = dist C3 B)
  (h_perp_C3B3 : ∠ (line C3 B) (line C3 B3) = 90)
  (h_P3 : intersection (line A B3) (line B A3) = P3) :
  ∃ M, M = midpoint A B ∧ is_circumcenter (triangle P1 P2 P3) M :=
by
  sorry

end treasure_location_l820_820914


namespace find_n_l820_820572

theorem find_n (n : ℤ) (hn : -180 ≤ n ∧ n ≤ 180) (hsin : Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180)) :
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by
  sorry

end find_n_l820_820572


namespace inverse_matrix_l820_820574

open Matrix

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![4, -1], ![2, 3]]

-- Define the expected inverse matrix
def A_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![3/14, 1/14], ![-1/7, 2/7]]

-- Main theorem statement: A * A_inv = 1
theorem inverse_matrix (A_inv_exists: det A ≠ 0) : A * A_inv = 1 :=
by
  sorry

end inverse_matrix_l820_820574


namespace positive_difference_of_sums_l820_820191

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820191


namespace therapy_sessions_l820_820898

theorem therapy_sessions (F A n : ℕ) 
  (h1 : F = A + 25)
  (h2 : F + A = 115)
  (h3 : F + (n - 1) * A = 250) : 
  n = 5 := 
by sorry

end therapy_sessions_l820_820898


namespace bicycle_speed_B_l820_820379

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820379


namespace water_in_pool_after_35_days_l820_820899

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end water_in_pool_after_35_days_l820_820899


namespace acute_angle_probability_l820_820805

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end acute_angle_probability_l820_820805


namespace positive_difference_even_odd_sums_l820_820253

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820253


namespace draw_non_defective_A_draw_at_least_one_defective_l820_820141

theorem draw_non_defective_A (total_products defective_products non_defective_products drawn_products : ℕ)
  (hc1 : total_products = 10)
  (hc2 : defective_products = 2)
  (hc3 : non_defective_products = 8)
  (hc4 : drawn_products = 3) :
  ∃ (A : ℕ), A ∈ finset.range 36 :=
by
  sorry

theorem draw_at_least_one_defective (total_products defective_products non_defective_products drawn_products : ℕ)
  (hc1 : total_products = 10)
  (hc2 : defective_products = 2)
  (hc3 : non_defective_products = 8)
  (hc4 : drawn_products = 3) :
  ∃ (ways : ℕ), ways = 64 :=
by
  sorry

end draw_non_defective_A_draw_at_least_one_defective_l820_820141


namespace positive_difference_even_odd_l820_820208

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820208


namespace professors_chairs_l820_820072

theorem professors_chairs :
  ∃ positions : Finset ℕ, positions.card = 3 ∧
  (∀ p ∈ positions, 2 ≤ p ∧ p ≤ 8 ) ∧
  (∃ students : Finset ℕ, students.card = 6 ∧ (positions ∪ students = Finset.range 9 \ {0,8})
  ∧ (positions ∩ students = ∅))
  ∧ (positions \ {positions.min', positions.max'} ≠ ∅)
  ∧ (∃ γ : list (Finset ℕ), γ.length = 3 ∧ γ.nodup ∧ γ.perm positions) :=
sorry

end professors_chairs_l820_820072


namespace positive_difference_of_sums_l820_820288

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820288


namespace positive_difference_l820_820229

-- Definition of the sum of the first n positive even integers
def sum_first_n_even (n : ℕ) : ℕ := 2 * n * (n + 1) / 2

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Theorem statement: Proving the positive difference between the sums
theorem positive_difference (he : sum_first_n_even 25 = 650) (ho : sum_first_n_odd 20 = 400) :
  abs (sum_first_n_even 25 - sum_first_n_odd 20) = 250 :=
by
  sorry

end positive_difference_l820_820229


namespace ajay_saves_each_month_l820_820932

def monthly_income : ℝ := 90000
def spend_household : ℝ := 0.50 * monthly_income
def spend_clothes : ℝ := 0.25 * monthly_income
def spend_medicines : ℝ := 0.15 * monthly_income
def total_spent : ℝ := spend_household + spend_clothes + spend_medicines
def amount_saved : ℝ := monthly_income - total_spent

theorem ajay_saves_each_month : amount_saved = 9000 :=
by sorry

end ajay_saves_each_month_l820_820932


namespace speed_of_student_B_l820_820425

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820425


namespace valid_code_count_l820_820796

def digit_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def original_code : Fin 10 × Fin 10 × Fin 10 := (7, 3, 4)

def is_valid_code (code : Fin 10 × Fin 10 × Fin 10) : Prop :=
  code.1 ∈ digit_set ∧ code.2 ∈ digit_set ∧ code.3 ∈ digit_set ∧
  (code.1, code.2, code.3) ≠ original_code ∧
  (code.1, code.2, code.3) ≠ (4, 3, 7) ∧
  (code.1, code.2, code.3) ≠ original_code.swap 0 1 2 ∧ -- invalid transpositions
  (code.1, code.2, code.3) ≠ original_code.swap 0 2 1 ∧  
  (code.1, code.2, code.3) ≠ original_code.swap 1 0 2 ∧    
  (code.1, code.2, code.3) ≠ original_code.swap 1 2 0 ∧    
  (code.1, code.2, code.3) ≠ original_code.swap 2 0 1 ∧    
  (code.1, code.2, code.3) ≠ original_code.swap 2 1 0 ∧     
  ¬ ((code.1 = 7 ∧ code.2 = 3) ∨ (code.1 = 7 ∧ code.3 = 4) ∨ (code.2 = 3 ∧ code.3 = 4))

theorem valid_code_count : ∃ n : ℕ, n = 702 ∧
  n = (digit_set × digit_set × digit_set).card - 27 := sorry

end valid_code_count_l820_820796


namespace bicycle_speed_B_l820_820387

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820387


namespace positive_difference_of_sums_l820_820285

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820285


namespace pyarelal_loss_is_correct_l820_820944

variables (P_P : ℝ) -- Pyarelal's capital
noncomputable def P_A : ℝ := (1 / 9) * P_P -- Ashok's capital

-- Total loss
def total_loss : ℝ := 1000

-- Pyarelal's share of the loss calculation
def pyarelal_loss : ℝ := (9 / 10) * total_loss

theorem pyarelal_loss_is_correct (h : P_A = (1 / 9) * P_P) (hl : total_loss = 1000) : 
  pyarelal_loss = 900 :=
by
  sorry

end pyarelal_loss_is_correct_l820_820944


namespace student_b_speed_l820_820376

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820376


namespace non_attacking_rooks_l820_820797

theorem non_attacking_rooks (r : ℕ) (c : ℕ)
    (num_rooks : ℕ) (placed_rooks : Fin r.Times Fin c → Bool)
    (h_r : r = 10) (h_c : c = 10)
    (h_num : num_rooks = 41)
    (h_placed_rooks : ∑ i : Fin r, ∑ j : Fin c, if placed_rooks ⟨i, j⟩ then 1 else 0 = num_rooks) :
    ∃ rooks_set : Finset (Fin r × Fin c), rooks_set.card = 5 ∧ 
    (∀ (rook1 rook2 : Fin r × Fin c), rook1 ∈ rooks_set → rook2 ∈ rooks_set → rook1 ≠ rook2 → rook1.1 ≠ rook2.1 ∧ rook1.2 ≠ rook2.2) :=
begin
  -- This is where the proof would go. We are specifying the conditions and expected outcome here.
  sorry
end

end non_attacking_rooks_l820_820797


namespace student_B_speed_l820_820463

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820463


namespace magnitude_complex_addition_l820_820711

noncomputable def z : ℂ := (2 * complex.I) / (1 - complex.I)

theorem magnitude_complex_addition : complex.abs (conj z + 3 * complex.I) = real.sqrt 5 := by
  sorry

end magnitude_complex_addition_l820_820711


namespace student_B_speed_l820_820504

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820504


namespace find_speed_of_B_l820_820357

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820357


namespace exists_m_n_l820_820943

theorem exists_m_n (k : ℕ) (h : k = 2^10 * 3^5 * 5^4 * 7^3 * 11^2) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m - n = k ∧ 
  (∃ xs : fin 2012 → ℕ, ∀ i, ∃ a b, m - (xs i)^2 = a^2 ∧ n - (xs i)^2 = b^2) :=
sorry

end exists_m_n_l820_820943


namespace triangle_AB_length_l820_820003

noncomputable def length_AB (AM BN : ℝ) (h1 : AM = 15) (h2 : BN = 20) (perpendicular : ⟦AM⟧.perpendicular ⟦BN⟧) : ℝ :=
  let AG := (2 / 3) * AM
  let BG := (2 / 3) * BN
  let AB := Math.sqrt (AG ^ 2 + BG ^ 2)
  AB

theorem triangle_AB_length :
  ∀ (AM BN : ℝ), (AM = 15 → BN = 20 → (AM.perpendicular BN) → length_AB AM BN = 50 / 3) :=
by {
  intros AM BN hAM hBN hPerpendicular,
  rw [hAM, hBN],
  let AG := (2 / 3) * 15,
  have hAG : AG = 10 := by norm_num,
  let BG := (2 / 3) * 20,
  have hBG : BG = 40 / 3 := by norm_num,
  rw [←hAG, ←hBG],
  dsimp only [length_AB],
  norm_num,
  sorry
}

end triangle_AB_length_l820_820003


namespace magnitude_b_sqrt_3_l820_820660

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
-- hypotheses
def h1 : ∥a - b∥ = real.sqrt 3 := sorry
def h2 : ∥a + b∥ = ∥2 • a - b∥ := sorry

-- theorem
theorem magnitude_b_sqrt_3 (h1 : ∥a - b∥ = real.sqrt 3)
                           (h2 : ∥a + b∥ = ∥2 • a - b∥) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_b_sqrt_3_l820_820660


namespace sum_of_series_base_6_l820_820997

noncomputable def base_6_to_decimal (n : ℕ) : ℕ := 
  let digits := (n.toDigits 6).reverse;
  digits.enum.sum (λ ⟨i, d⟩, d * 6^i)

noncomputable def decimal_to_base_6 (n : ℕ) : ℕ := 
  (List.of_digits 6 (n.to_digits 6))

theorem sum_of_series_base_6 : 
  let a := base_6_to_decimal 2       -- First term in base 10
  let d := base_6_to_decimal 2       -- Common difference in base 10
  let l := base_6_to_decimal 100     -- Last term in base 10
  let n := ((l - a) / d) + 1         -- Number of terms
  let sum := (n * (a + l)) / 2       -- Sum in base 10
  decimal_to_base_6 sum = 1330 
:=
by {
  let a := base_6_to_decimal 2
  let d := base_6_to_decimal 2
  let l := base_6_to_decimal 100
  let n := ((l - a) / d) + 1
  let sum := (n * (a + l)) / 2
  show decimal_to_base_6 sum = 1330, from sorry
}

end sum_of_series_base_6_l820_820997


namespace f_decreasing_on_interval_l820_820834

def f (x : ℝ) : ℝ := x / Real.exp x

theorem f_decreasing_on_interval : MonotoneDecreasingOn f (Set.Ioi 1) :=
sorry

end f_decreasing_on_interval_l820_820834


namespace positive_difference_eq_250_l820_820163

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820163


namespace break_even_number_of_performances_l820_820093

theorem break_even_number_of_performances :
  ∃ x : ℕ, (16000 * x = 81000 + 7000 * x) ∧ x = 9 :=
by
  use 9
  split
  sorry

end break_even_number_of_performances_l820_820093


namespace positive_difference_of_sums_l820_820284

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820284


namespace volume_of_QEFGH_l820_820083

noncomputable def volume_of_pyramid (base_area height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem volume_of_QEFGH:
  ∀ (EF FG QE : ℝ),
  EF = 10 →
  FG = 5 →
  QE = 7 →
  volume_of_pyramid (EF * FG) QE = 350 / 3 :=
by
  intros EF FG QE hEF hFG hQE
  rw [hEF, hFG, hQE]
  unfold volume_of_pyramid
  norm_num
  sorry

end volume_of_QEFGH_l820_820083


namespace quadruplet_solution_l820_820966

noncomputable def solve_quadruplet (a b c d : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 1 ∧
  a^2012 + 2012 * b = 2012 * c + d^2012 ∧
  2012 * a + b^2012 = c^2012 + 2012 * d

theorem quadruplet_solution (a b c d : ℝ) :
  solve_quadruplet a b c d → ∃ t > 0, a = t ∧ b = 1 / t ∧ c = 1 / t ∧ d = t :=
begin
  rw solve_quadruplet,
  intros h,
  -- Proof would go here
  sorry
end

end quadruplet_solution_l820_820966


namespace overlapping_region_area_l820_820152

noncomputable def radius : ℝ := 15
noncomputable def central_angle_radians : ℝ := Real.pi / 2
noncomputable def area_of_sector : ℝ := (1 / 4) * Real.pi * (radius^2)
noncomputable def side_length_equilateral_triangle : ℝ := radius
noncomputable def area_of_equilateral_triangle : ℝ := (Real.sqrt 3 / 4) * (side_length_equilateral_triangle^2)
noncomputable def overlapping_area : ℝ := 2 * area_of_sector - area_of_equilateral_triangle

theorem overlapping_region_area :
  overlapping_area = 112.5 * Real.pi - 56.25 * Real.sqrt 3 :=
by
  sorry
 
end overlapping_region_area_l820_820152


namespace find_radius_l820_820915

noncomputable def radius_of_probability (π : ℝ) : ℝ := real.sqrt(1 / (3 * π))

theorem find_radius :
  radius_of_probability real.pi = 0.3 :=
sorry

end find_radius_l820_820915


namespace inequality_a4_b4_c4_l820_820080

theorem inequality_a4_b4_c4 (a b c : Real) : a^4 + b^4 + c^4 ≥ abc * (a + b + c) := 
by
  sorry

end inequality_a4_b4_c4_l820_820080


namespace largest_noncomposite_sum_l820_820865

theorem largest_noncomposite_sum:
  ¬(∃ n > 0, ∀ m > 0, n = 60 * m + b ∧ composite b) := by
  sorry

end largest_noncomposite_sum_l820_820865


namespace arcsin_half_eq_pi_six_l820_820548

theorem arcsin_half_eq_pi_six : real.arcsin (1 / 2) = real.pi / 6 :=
sorry

end arcsin_half_eq_pi_six_l820_820548


namespace isosceles_triangle_sum_of_m_equals_R_l820_820618

theorem isosceles_triangle_sum_of_m_equals_R (ABC : Type) [is_tria ABC] (AB AC : ℝ) (D : ℕ → ABC → ℝ) 
  (Q : ℕ) (R : ℝ) (h1 : AB = AC) (h2 : AB = sqrt 2) 
  (h3 : ∀ i : ℕ, i < Q → D i BC > 0) -- dummy condition for illustration
  (h4 : ∑ i in finset.range Q, (AD (D i) ^ 2 + (BD (D i) * DC (D i))) = R) :
  R = 2 * Q := 
sorry

end isosceles_triangle_sum_of_m_equals_R_l820_820618


namespace magnitude_of_b_l820_820689

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820689


namespace factorize_quartic_l820_820818

-- Specify that p and q are real numbers (ℝ)
variables {p q : ℝ}

-- Statement: For any real numbers p and q, the polynomial x^4 + p x^2 + q can always be factored into two quadratic polynomials.
theorem factorize_quartic (p q : ℝ) : 
  ∃ a b c d e f : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + p * x^2 + q :=
sorry

end factorize_quartic_l820_820818


namespace activity_order_l820_820133

open Rat

-- Define the fractions of students liking each activity
def soccer_fraction : ℚ := 13 / 40
def swimming_fraction : ℚ := 9 / 24
def baseball_fraction : ℚ := 11 / 30
def hiking_fraction : ℚ := 3 / 10

-- Statement of the problem
theorem activity_order :
  soccer_fraction = 39 / 120 ∧
  swimming_fraction = 45 / 120 ∧
  baseball_fraction = 44 / 120 ∧
  hiking_fraction = 36 / 120 ∧
  45 / 120 > 44 / 120 ∧
  44 / 120 > 39 / 120 ∧
  39 / 120 > 36 / 120 →
  ("Swimming, Baseball, Soccer, Hiking" = "Swimming, Baseball, Soccer, Hiking") :=
by
  sorry

end activity_order_l820_820133


namespace range_of_m_l820_820591

theorem range_of_m (m : ℝ) (x0 : ℝ)
  (h : (4^(-x0) - m * 2^(-x0 + 1)) = -(4^x0 - m * 2^(x0 + 1))) :
  m ≥ 1/2 :=
sorry

end range_of_m_l820_820591


namespace speed_of_student_B_l820_820486

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820486


namespace pqrs_product_l820_820074

noncomputable def product_of_area_and_perimeter :=
  let P := (1, 3)
  let Q := (4, 4)
  let R := (3, 1)
  let S := (0, 0)
  let side_length := Real.sqrt ((1 - 0)^2 * 4 + (3 - 0)^2 * 4)
  let area := side_length ^ 2
  let perimeter := 4 * side_length
  area * perimeter

theorem pqrs_product : product_of_area_and_perimeter = 208 * Real.sqrt 52 := 
  by 
    sorry

end pqrs_product_l820_820074


namespace student_b_speed_l820_820367

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820367


namespace sleep_hours_for_desired_average_l820_820031

theorem sleep_hours_for_desired_average 
  (s_1 s_2 : ℝ) (h_1 h_2 : ℝ) (k : ℝ) 
  (h_inverse_relation : ∀ s h, s * h = k)
  (h_s1 : s_1 = 75)
  (h_h1 : h_1 = 6)
  (h_average : (s_1 + s_2) / 2 = 85) : 
  h_2 = 450 / 95 := 
by 
  sorry

end sleep_hours_for_desired_average_l820_820031


namespace solve_for_x_l820_820825

theorem solve_for_x : ∃ x : ℝ, (1 / 3 + 1 / x = 7 / 12) ∧ (x = 4) :=
begin
  sorry
end

end solve_for_x_l820_820825


namespace probability_divisible_by_5_l820_820346

open Set

theorem probability_divisible_by_5 :
  let numbers := { n : ℕ | 100 ≤ n ∧ n ≤ 999 }
  let divisible_by_5 := { n ∈ numbers | n % 5 = 0 }
  (card divisible_by_5 : ℚ) / (card numbers) = 1 / 5 :=
by
  sorry

end probability_divisible_by_5_l820_820346


namespace nonneg_solutions_count_eq_one_l820_820701

theorem nonneg_solutions_count_eq_one : 
  (finset.filter (λ x, 0 ≤ x) (finset.filter (λ x, x^2 + 6*x = 18) finset.univ)).card = 1 :=
by
  sorry

end nonneg_solutions_count_eq_one_l820_820701


namespace men_in_club_l820_820340

-- Definitions
variables (M W : ℕ) -- Number of men and women

-- Conditions
def club_members := M + W = 30
def event_participation := W / 3 + M = 18

-- Goal
theorem men_in_club : club_members M W → event_participation M W → M = 12 :=
sorry

end men_in_club_l820_820340


namespace sum_units_digit_l820_820764

theorem sum_units_digit (n : ℤ) : 
  (∑ k in Finset.range 2001, n^k) % 10 = 1 := by
  sorry

end sum_units_digit_l820_820764


namespace job_completion_time_l820_820875

theorem job_completion_time
  (A_completion_time : ℝ)
  (D_completion_time : ℝ)
  (A_rate := 1 / A_completion_time)
  (D_rate := 1 / D_completion_time)
  (combined_rate := A_rate + D_rate)
  (combined_completion_time := 1 / combined_rate) :
  A_completion_time = 12 →
  D_completion_time = 6 →
  combined_completion_time = 4 :=
by {
  intros hA hD,
  have hA_rate : A_rate = 1 / 12 := by rwa hA,
  have hD_rate : D_rate = 1 / 6 := by rwa hD,
  have h_combined_rate : combined_rate = (1 / 12) + (1 / 6) := by rw [hA_rate, hD_rate],
  have h_combined_rate_simplified : combined_rate = 1 / 4, from calc
    (1 / 12) + (1 / 6) = (1 / 12) + (2 / 12) : by norm_num
    ... = 3 / 12 : by ring
    ... = 1 / 4 : by norm_num,
  rw h_combined_rate_simplified at *,
  norm_num,
  sorry
}

end job_completion_time_l820_820875


namespace polygon_sides_l820_820920

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end polygon_sides_l820_820920


namespace trig_identity_l820_820139

theorem trig_identity :
  (cos (real.pi / 9) * sin (real.pi / 9)) / (cos (5 * real.pi / 36) ^ 2 - sin (5 * real.pi / 36) ^ 2) = 1 / 2 := 
by
  sorry

end trig_identity_l820_820139


namespace train_crosses_pole_in_9_seconds_l820_820929

theorem train_crosses_pole_in_9_seconds
  (speed_kmh : ℝ) (train_length_m : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 58) 
  (h2 : train_length_m = 145) 
  (h3 : time_s = train_length_m / (speed_kmh * 1000 / 3600)) :
  time_s = 9 :=
by
  sorry

end train_crosses_pole_in_9_seconds_l820_820929


namespace positive_difference_sums_even_odd_l820_820241

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l820_820241


namespace positive_difference_eq_250_l820_820162

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820162


namespace bicycle_speed_B_l820_820388

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820388


namespace find_speed_of_B_l820_820361

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820361


namespace maxSumEqualsEighteen_l820_820107

-- Definitions representing the conditions
def uniqueDieOppositePairs : List (ℕ × ℕ) := [(1, 8), (2, 7), (3, 6), (4, 5)]

noncomputable def maximumSumAtVertex : ℕ :=
  if h : ∃ x y z, (x, y) ∈ uniqueDieOppositePairs ∧ (y, z) ∈ uniqueDieOppositePairs ∧ (z, x) ∈ uniqueDieOppositePairs 
     then ∑ p in [(8, 6, 4)], (p.1 + p.2 + p.3)
     else 18

-- The problem statement to prove
theorem maxSumEqualsEighteen (h : ∀ (x y), (x, y) ∈ uniqueDieOppositePairs → x + y = 9) : maximumSumAtVertex = 18 := 
by sorry

end maxSumEqualsEighteen_l820_820107


namespace positive_difference_even_odd_l820_820201

theorem positive_difference_even_odd :
  ((2 * (1 + 2 + ... + 25)) - (1 + 3 + ... + 39)) = 250 := 
by
  sorry

end positive_difference_even_odd_l820_820201


namespace determine_value_of_a_l820_820112

noncomputable def vertex_quadratic (a : ℝ) := 
 ∀ x : ℝ, (a * (x + 4)^2 : ℝ)

theorem determine_value_of_a
  (h1 : vertex_quadratic a = 0  ∧  -4=  x)
  (h2 : vertex_quadratic a = -32  ∧ 2= x):
  a = -8/9 :=
by
  sorry

end determine_value_of_a_l820_820112


namespace count_positive_values_x_l820_820700

theorem count_positive_values_x :
  let valid_x := {x : ℕ | 25 ≤ x ∧ x ≤ 33} in
  set.finite valid_x ∧ set.count valid_x = 9 :=
by
  sorry

end count_positive_values_x_l820_820700


namespace student_b_speed_l820_820439

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820439


namespace sum_of_first_n_odd_numbers_l820_820987

theorem sum_of_first_n_odd_numbers (n : ℕ) : ∑ k in range n, (2 * k + 1) = n ^ 2 := 
by
  sorry

end sum_of_first_n_odd_numbers_l820_820987


namespace shift_parabola_3_right_4_up_l820_820724

theorem shift_parabola_3_right_4_up (x : ℝ) : 
  let y := x^2 in
  (shifted_y : ℝ) = ((x - 3)^2 + 4) :=
begin
  sorry
end

end shift_parabola_3_right_4_up_l820_820724


namespace find_speed_B_l820_820395

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820395


namespace expansion_sum_of_coefficients_expansion_coefficient_of_x2_l820_820017

noncomputable def sum_of_coefficients : ℤ :=
  (x^2 - 3 / sqrt(x))^6.eval(1)

noncomputable def coefficient_of_x2 : ℤ := by
  let x_term_coefficient := 6.choose 4 * (-3)^4
  exact x_term_coefficient

theorem expansion_sum_of_coefficients : sum_of_coefficients = 64 := by
  sorry

theorem expansion_coefficient_of_x2 : coefficient_of_x2 = 1215 := by
  sorry

end expansion_sum_of_coefficients_expansion_coefficient_of_x2_l820_820017


namespace find_speed_B_l820_820402

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820402


namespace doubled_radius_and_arc_length_invariant_l820_820728

theorem doubled_radius_and_arc_length_invariant (r l : ℝ) : (l / r) = (2 * l / (2 * r)) :=
by
  sorry

end doubled_radius_and_arc_length_invariant_l820_820728


namespace speed_of_student_B_l820_820484

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820484


namespace bicycle_speed_B_l820_820383

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820383


namespace expected_rolls_in_year_l820_820951

noncomputable def expected_rolls_in_day (p_stop: ℚ) (p_again: ℚ) : ℚ :=
1 / p_stop

theorem expected_rolls_in_year :
  let p_stop := (7 : ℚ) / 8 in
  let p_again := (1 : ℚ) / 8 in
  let E := expected_rolls_in_day p_stop p_again in
  E * 365 = 417 ∨ 
  abs (E * 365 - 417.14) < 1 :=
by
  -- Define all parameters and values
  let p_stop := (7 : ℚ) / 8
  let p_again := (1 : ℚ) / 8
  have E : ℚ := expected_rolls_in_day p_stop p_again
  
  -- Confirm the expected value }
  sorry

end expected_rolls_in_year_l820_820951


namespace speed_of_student_B_l820_820499

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l820_820499


namespace problem_l820_820763

noncomputable def S (x : ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in (Finset.range k), x i ^ k

theorem problem (n : ℕ) (x : ℕ → ℝ)
  (h₁ : 1 < n)
  (h₂ : ∀ i : ℕ, i < n → (x i) ∈ ℝ)
  (h₃ : ∀ k : ℕ, 1 ≤ k → S x k = S x (n+1)) :
  ∀ i : ℕ, i < n → x i = 0 ∨ x i = 1 :=
sorry

end problem_l820_820763


namespace positive_difference_even_odd_sums_l820_820265

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820265


namespace positive_difference_even_odd_sums_l820_820254

theorem positive_difference_even_odd_sums :
  let sum_even := 2 * (List.range 25).sum in
  let sum_odd := 20^2 in
  sum_even - sum_odd = 250 :=
by
  let sum_even := 2 * (List.range 25).sum;
  let sum_odd := 20^2;
  sorry

end positive_difference_even_odd_sums_l820_820254


namespace norm_b_eq_sqrt_3_l820_820667

variables {V : Type*} [inner_product_space ℝ V] 
(variable a b : V)
variable (h1 : ∥a - b∥ = √3)
variable (h2 : ∥a + b∥ = ∥2 • a - b∥)

theorem norm_b_eq_sqrt_3 : ∥b∥ = √3 :=
sorry

end norm_b_eq_sqrt_3_l820_820667


namespace find_speed_B_l820_820396

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820396


namespace remainder_of_prime_when_divided_by_240_l820_820095

theorem remainder_of_prime_when_divided_by_240 (n : ℕ) (hn : n > 0) (hp : Nat.Prime (2^n + 1)) : (2^n + 1) % 240 = 17 := 
sorry

end remainder_of_prime_when_divided_by_240_l820_820095


namespace acute_angle_probability_is_half_l820_820812

noncomputable def probability_acute_angle (h_mins : ℕ) (m_mins : ℕ) : ℝ :=
  if m_mins < 15 ∨ m_mins > 45 then 1 / 2 else 0

theorem acute_angle_probability_is_half (h_mins : ℕ) (m_mins : ℕ) :
  let P := probability_acute_angle h_mins m_mins in
  0 ≤ m_mins ∧ m_mins < 60 →
  P = 1 / 2 :=
sorry

end acute_angle_probability_is_half_l820_820812


namespace find_speed_of_B_l820_820350

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820350


namespace geometric_series_ratio_l820_820785

open_locale big_operators

theorem geometric_series_ratio (a q : ℝ) (hq : q ≠ 1) :
  (a * (1 - q^6) / (1 - q)) / (a * (1 - q^3) / (1 - q)) = 1 / 2 →
  (a * (1 - q^9) / (1 - q)) / (a * (1 - q^3) / (1 - q)) = 3 / 4 :=
by sorry

end geometric_series_ratio_l820_820785


namespace student_B_speed_l820_820506

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l820_820506


namespace positive_diff_even_odd_sums_l820_820182

theorem positive_diff_even_odd_sums : 
  (∑ k in finset.range 25, 2 * (k + 1)) - (∑ k in finset.range 20, 2 * k + 1) = 250 := 
by
  sorry

end positive_diff_even_odd_sums_l820_820182


namespace price_of_each_orange_l820_820945

theorem price_of_each_orange 
  (x : ℕ)
  (a o : ℕ)
  (h1 : a + o = 20)
  (h2 : 40 * a + x * o = 1120)
  (h3 : (a + o - 10) * 52 = 1120 - 10 * x) :
  x = 60 :=
sorry

end price_of_each_orange_l820_820945


namespace exams_left_wednesday_l820_820791

-- Defining constants and intermediate computations as the provided conditions.
constant total_exams : ℕ := 120
constant monday_percentage : ℚ := 0.60
constant tuesday_percentage : ℚ := 0.75

-- Calculations according to the problem.
def exams_graded_monday : ℕ := total_exams * monday_percentage
def remaining_after_monday : ℕ := total_exams - exams_graded_monday
def exams_graded_tuesday : ℕ := remaining_after_monday * tuesday_percentage
def remaining_after_tuesday : ℕ := remaining_after_monday - exams_graded_tuesday

theorem exams_left_wednesday : remaining_after_tuesday = 12 := by
  sorry

end exams_left_wednesday_l820_820791


namespace student_b_speed_l820_820435

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820435


namespace tenth_ring_unit_squares_l820_820956

theorem tenth_ring_unit_squares (n : ℕ) (n = 10): 
  let side_length := 2 * n + 3
  let prev_side_length := 2 * (n - 1) + 3
  ring_squares := (side_length ^ 2 - prev_side_length ^ 2) / 2
  ring_squares = 88 :=
by
  sorry

end tenth_ring_unit_squares_l820_820956


namespace solve_congruence_l820_820826

theorem solve_congruence :
  ∃ n : ℤ, 19 * n ≡ 13 [ZMOD 47] ∧ n ≡ 25 [ZMOD 47] :=
by
  sorry

end solve_congruence_l820_820826


namespace expression_evaluation_l820_820560

-- Define the constants and expression
def exp: ℕ := (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^{16} + 3^{16}) * (2^{32} + 3^{32}) * (2^{64} + 3^{64}) * (2 + 1)

-- Define the target value
def target: ℕ := 3^{129} - 3 * 2^{128}

-- The theorem to prove
theorem expression_evaluation : exp = target := by
  sorry

end expression_evaluation_l820_820560


namespace positive_difference_sums_l820_820209

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820209


namespace sum_cot_eq_l820_820081

theorem sum_cot_eq (n : ℕ) (hn : n ≠ 0) (x : ℝ) (hx : ∀ k : ℕ, k ≤ n → x ≠ n * π / (2 * k)) :
  (∑ i in finset.range n, (1 / real.sin (2 ^ (i + 1) * x))) = real.cot x - real.cot (2 ^ n * x) :=
sorry

end sum_cot_eq_l820_820081


namespace meaning_of_x_l820_820337

theorem meaning_of_x (x : ℝ) (h : 52 + 52 * (1 + x) + 52 * (1 + x) ^ 2 = 196) :
  x = ((production growth rate in August and September) / 2) :=
sorry

end meaning_of_x_l820_820337


namespace positive_difference_even_odd_sums_l820_820259

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end positive_difference_even_odd_sums_l820_820259


namespace sum_eval_eq_106_l820_820980

-- Definitions based on given conditions
noncomputable def sum_expr (x : ℕ) : ℝ := 2 * cos x * cos 2 * (1 + sec (x-2) * sec (x+2))
noncomputable def eval_sum : ℝ := ∑ x in finset.range (50 - 3 + 1) + 3, sum_expr x

-- Theorem stating the given equality based on the problem and its solution
theorem sum_eval_eq_106 : eval_sum = 106 :=
sorry

end sum_eval_eq_106_l820_820980


namespace magnitude_of_b_l820_820684

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820684


namespace expression_satisfying_pair_l820_820140

theorem expression_satisfying_pair : 
  let x2 := 7 + 4 * Real.sqrt 3
  let y2 := 7 - 4 * Real.sqrt 3 in
  (∃ (x y : ℝ), x^2 = x2 ∧ y^2 = y2 ∧ (x^3 / y - y^3 / x = 112 * Real.sqrt 3)) := 
sorry

end expression_satisfying_pair_l820_820140


namespace digits_right_of_decimal_of_fraction_l820_820552

theorem digits_right_of_decimal_of_fraction :
  let frac := (2^7 : ℚ) / (14^3 * 125)
  (frac.toReal.digits_to_right_of_decimal) = 8 :=
sorry

end digits_right_of_decimal_of_fraction_l820_820552


namespace expected_value_xi_l820_820143

open Finset  
open Classical  

noncomputable def pmax : Finset (Finset ℕ) := 
  (powerset (range 5).erase 0).filter (λ s, card s = 3)

noncomputable def xi (s : Finset ℕ) : ℕ := 
  s.max' (by { apply pmax.card_pos.2, simp, simp })

noncomputable def px (k : ℕ) : ℚ := 
  (pmax.filter (λ s, xi s = k)).card / pmax.card

theorem expected_value_xi : 
  let E_xi := (3 * (px 3) + 4 * (px 4) + 5 * (px 5)) in 
  E_xi = 4.5 := by
  sorry

end expected_value_xi_l820_820143


namespace find_speed_of_B_l820_820358

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820358


namespace sqrt8_same_type_as_sqrt2_l820_820870

theorem sqrt8_same_type_as_sqrt2 :
  (∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 8) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 4) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 6) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 10) :=
by
  sorry

end sqrt8_same_type_as_sqrt2_l820_820870


namespace speed_of_student_B_l820_820430

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820430


namespace distribute_diamonds_among_two_safes_l820_820971

theorem distribute_diamonds_among_two_safes (N : ℕ) :
  ∀ banker : ℕ, banker < 777 → ∃ s1 s2 : ℕ, s1 ≠ s2 ∧ s1 + s2 = N := sorry

end distribute_diamonds_among_two_safes_l820_820971


namespace positive_difference_of_sums_l820_820189

def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_of_sums :
  let even_sum := sum_first_n_even 25 in
  let odd_sum := sum_first_n_odd 20 in
  even_sum - odd_sum = 250 :=
by
  let even_sum := sum_first_n_even 25
  let odd_sum := sum_first_n_odd 20
  have h1 : even_sum = 25 * 26 := 
    by sorry
  have h2 : odd_sum = 20 * 20 := 
    by sorry
  show even_sum - odd_sum = 250 from 
    by calc
      even_sum - odd_sum = (25 * 26) - (20 * 20) := by sorry
      _ = 650 - 400 := by sorry
      _ = 250 := by sorry

end positive_difference_of_sums_l820_820189


namespace student_b_speed_l820_820368

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l820_820368


namespace find_k_l820_820118

-- Define the lines l1 and l2
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the fact that the quadrilateral formed by l1, l2, and the positive halves of the axes
-- has a circumscribed circle.
def has_circumscribed_circle (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), line1 x1 y1 ∧ line2 k x2 y2 ∧
  x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0 ∧
  (x1 - x2 = 0 ∨ y1 - y2 = 0) ∧
  (x1 = 0 ∨ y1 = 0 ∨ x2 = 0 ∨ y2 = 0)

-- The statement we need to prove
theorem find_k : ∀ k : ℝ, has_circumscribed_circle k → k = 3 := by
  sorry

end find_k_l820_820118


namespace loan_period_l820_820345

theorem loan_period (principal : ℝ) (rate_A rate_C gain_B : ℝ) (n : ℕ) 
  (h1 : principal = 3150)
  (h2 : rate_A = 0.08)
  (h3 : rate_C = 0.125)
  (h4 : gain_B = 283.5) :
  (gain_B = (rate_C * principal - rate_A * principal) * n) → n = 2 := by
  sorry

end loan_period_l820_820345


namespace problem_1_problem_2_l820_820327

open Real

/-- Statement for the first proof problem -/
theorem problem_1 :
  (1 / (sqrt 2 - 1)) - ((3 / 5) ^ 0) + ((9 / 4) ^ (-0.5)) + root (4:ℝ) ((sqrt 2 - exp 1)^4) = exp 1 + 2 / 3 := 
sorry

/-- Statement for the second proof problem -/
theorem problem_2 :
  log 500 + log (8 / 5) - 0.5 * log 64 + 50 * (log 2 + log 5)^2 = 52 := 
sorry

end problem_1_problem_2_l820_820327


namespace student_b_speed_l820_820437

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l820_820437


namespace percent_do_not_like_basketball_l820_820734

variable (total_students : ℕ) (male_ratio female_ratio : ℕ) (male_like_ratio female_like_ratio : ℚ)

theorem percent_do_not_like_basketball
  (h1 : total_students = 1000)
  (h2 : male_ratio = 3)
  (h3 : female_ratio = 2)
  (h4 : male_like_ratio = 2 / 3)
  (h5 : female_like_ratio = 1 / 5) :
  let male_students := male_ratio * (total_students / (male_ratio + female_ratio)),
      female_students := female_ratio * (total_students / (male_ratio + female_ratio)),
      male_like := male_like_ratio * male_students,
      female_like := female_like_ratio * female_students,
      total_like := male_like + female_like,
      total_dislike := total_students - total_like,
      percent_dislike : ℚ := 100 * (total_dislike / total_students) in
  percent_dislike = 52 := 
by
  sorry

end percent_do_not_like_basketball_l820_820734


namespace manoj_lent_amount_lent_to_ramu_l820_820787

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem manoj_lent_amount_lent_to_ramu :
  let P_anwar := 3900.0
  let r_anwar := 0.06
  let t_anwar := 3
  let P_owes_anwar := compound_interest P_anwar r_anwar 1 t_anwar
  let P_ramu := sorry
  let r_ramu := 0.09
  let t_ramu := 4
  let gain := 824.85
  let amount_lent_by_ramu := P_ramu * (1 + (r_ramu / 2)) ^ (2 * t_ramu)
  P_owes_anwar + gain = amount_lent_by_ramu →
  P_ramu ≈ 3842.99 :=
by
  sorry

end manoj_lent_amount_lent_to_ramu_l820_820787


namespace magnitude_of_b_l820_820656

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Defining the conditions
def condition1 : Prop := ∥a - b∥ = real.sqrt 3
def condition2 : Prop := ∥a + b∥ = ∥(2 : ℝ)•a - b∥

-- Goal: Prove |b| = sqrt(3)
theorem magnitude_of_b (h1 : condition1 a b) (h2 : condition2 a b) : ∥b∥ = real.sqrt 3 :=
sorry

end magnitude_of_b_l820_820656


namespace strict_increasing_when_a_neg1_upper_bound_when_a_in_interval_l820_820631

noncomputable def f (x a : ℝ) : ℝ := (x + 1) / Real.exp x - a * Real.log x

-- First Proof Problem: Monotonicity of f(x) when a = -1
theorem strict_increasing_when_a_neg1 : 
  ∀ x : ℝ, 0 < x → (Real.log x + (x+1) / Real.exp x) = f x (-1) → 0 < x → 
  ∀ x y : ℝ, x < y → f x (-1) < f y (-1) :=
sorry

-- Second Proof Problem: Upper bound for f when a in [-1/e, 0)
theorem upper_bound_when_a_in_interval : 
  ∀ x a : ℝ, a ∈ Icc (-1 / Real.exp 1) 0 → 0 < x ∧ x ≤ 2 →
  f x a < (1 - a - a^2) / Real.exp (-a) :=
sorry

end strict_increasing_when_a_neg1_upper_bound_when_a_in_interval_l820_820631


namespace range_neg_square_l820_820843

theorem range_neg_square (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) : 
  -9 ≤ -x^2 ∧ -x^2 ≤ 0 :=
sorry

end range_neg_square_l820_820843


namespace probability_pair_tile_l820_820975

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end probability_pair_tile_l820_820975


namespace find_speed_B_l820_820392

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l820_820392


namespace find_original_number_l820_820889

/-- The difference between a number increased by 18.7% and the same number decreased by 32.5% is 45. -/
theorem find_original_number (w : ℝ) (h : 1.187 * w - 0.675 * w = 45) : w = 45 / 0.512 :=
by
  sorry

end find_original_number_l820_820889


namespace tangent_parallel_to_x_axis_tangent_parallel_to_bisector_first_quadrant_l820_820577

noncomputable def parabola (x : ℝ) : ℝ := 2 - x^2

theorem tangent_parallel_to_x_axis :
  ∃ x y, parabola x = y ∧ (∂ (λ x, parabola x) / ∂ x) x = 0 ∧ (x, y) = (0, 2) := by
  sorry

theorem tangent_parallel_to_bisector_first_quadrant :
  ∃ x y, parabola x = y ∧ (∂ (λ x, parabola x) / ∂ x) x = 1 ∧ (x, y) = (-1/2, 7/4) := by
  sorry

end tangent_parallel_to_x_axis_tangent_parallel_to_bisector_first_quadrant_l820_820577


namespace height_of_tree_l820_820910

-- Definitions based on conditions
def net_gain (hop: ℕ) (slip: ℕ) : ℕ := hop - slip

def total_distance (hours: ℕ) (net_gain: ℕ) (final_hop: ℕ) : ℕ :=
  hours * net_gain + final_hop

-- Conditions
def hop : ℕ := 3
def slip : ℕ := 2
def time : ℕ := 20

-- Deriving the net gain per hour
#eval net_gain hop slip  -- Evaluates to 1

-- Final height proof problem
theorem height_of_tree : total_distance 19 (net_gain hop slip) hop = 22 := by
  sorry  -- Proof to be filled in

end height_of_tree_l820_820910


namespace b_3_value_S_m_formula_l820_820061

-- Definition of the sequences a_n and b_n
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n
def b_m (m : ℕ) : ℕ := a_n (3 * m)

-- Given b_m = 3^(2m) for m in ℕ*
lemma b_m_formula (m : ℕ) (h : m > 0) : b_m m = 3 ^ (2 * m) :=
by sorry -- (This proof step will later ensure that b_m m is defined as required)

-- Prove b_3 = 729
theorem b_3_value : b_m 3 = 729 :=
by sorry

-- Sum of the first m terms of the sequence b_n
def S_m (m : ℕ) : ℕ := (Finset.range m).sum (λ i => if i = 0 then 0 else b_m (i + 1))

-- Prove S_m = (3/8)(9^m - 1)
theorem S_m_formula (m : ℕ) : S_m m = (3 / 8) * (9 ^ m - 1) :=
by sorry

end b_3_value_S_m_formula_l820_820061


namespace find_speed_of_B_l820_820406

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l820_820406


namespace find_x_value_l820_820739

noncomputable def x_value (x : ℝ) :=
  (let angle_sum := 3 * x + (3 * x - 40) + 90 in angle_sum = 180)

theorem find_x_value (x : ℝ) (h : x_value x) : x = 65 / 3 :=
  by {
    sorry
  }

end find_x_value_l820_820739


namespace find_all_functions_satisfying_func_eq_l820_820566

-- Given a function f : ℝ → ℝ that satisfies a certain functional equation:
def satisfies_func_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (floor x * y) = f x * floor (f y)

-- The main proposition to prove:
theorem find_all_functions_satisfying_func_eq :
  ∀ f : ℝ → ℝ, satisfies_func_eq f → (f = (λ x, 0) ∨ (∃ c : ℝ, 1 ≤ c ∧ c < 2 ∧ f = (λ _, c))) :=
by
  -- Proof goes here
  sorry

end find_all_functions_satisfying_func_eq_l820_820566


namespace zain_has_80_coins_at_most_20_dollars_l820_820877

-- Define the number of coins Emerie has
def emerie_coins : nat := 6 + 7 + 5 + 10 + 2

-- Define the number of each type of coin Zain has
def zain_quarters : nat := 6 + 10
def zain_dimes : nat := 7 + 10
def zain_nickels : nat := 5 + 10
def zain_pennies : nat := 10 + 10
def zain_half_dollars : nat := 2 + 10

-- Calculate the total number of coins Zain has
def zain_total_coins : nat := zain_quarters + zain_dimes + zain_nickels + zain_pennies + zain_half_dollars

-- Define the values of each type of coin
def value_quarter : ℝ := 0.25
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_half_dollar : ℝ := 0.50

-- Calculate the total value of Zain's coins
def zain_total_value : ℝ :=
  (zain_quarters * value_quarter) +
  (zain_dimes * value_dime) +
  (zain_nickels * value_nickel) +
  (zain_pennies * value_penny) +
  (zain_half_dollars * value_half_dollar)

-- The theorem that needs to be proved
theorem zain_has_80_coins_at_most_20_dollars :
  zain_total_coins = 80 ∧ zain_total_value ≤ 20 := by
  sorry

end zain_has_80_coins_at_most_20_dollars_l820_820877


namespace acute_angle_probability_is_half_l820_820811

noncomputable def probability_acute_angle (h_mins : ℕ) (m_mins : ℕ) : ℝ :=
  if m_mins < 15 ∨ m_mins > 45 then 1 / 2 else 0

theorem acute_angle_probability_is_half (h_mins : ℕ) (m_mins : ℕ) :
  let P := probability_acute_angle h_mins m_mins in
  0 ≤ m_mins ∧ m_mins < 60 →
  P = 1 / 2 :=
sorry

end acute_angle_probability_is_half_l820_820811


namespace range_shifted_function_l820_820130

-- Define the range of the function y = f(x) as [-2, 2]
def range_f (f : ℝ → ℝ) : set ℝ := {y | ∃ x, f(x) = y}

-- Define the range condition for y = f(x)
axiom range_f_condition (f : ℝ → ℝ) : range_f f = set.Icc (-2 : ℝ) 2

-- Prove that the range of y = f(x-2) is also [-2, 2]
theorem range_shifted_function (f : ℝ → ℝ) : range_f (λ x, f (x - 2)) = set.Icc (-2) 2 :=
by
  sorry

end range_shifted_function_l820_820130


namespace smallest_blocks_required_l820_820341

theorem smallest_blocks_required (L H : ℕ) (block_height block_long block_short : ℕ) 
  (vert_joins_staggered : Prop) (consistent_end_finish : Prop) : 
  L = 120 → H = 10 → block_height = 1 → block_long = 3 → block_short = 1 → 
  (vert_joins_staggered) → (consistent_end_finish) → 
  ∃ n, n = 415 :=
by
  sorry

end smallest_blocks_required_l820_820341


namespace proof_problem_l820_820894

-- Conditions
def total_participants : ℕ := 3000
def freq_60_70 := 2
def freq_70_80 := 3
def freq_80_90 := 6
def freq_90_100 := 7
def freq_100_110 := 2

-- Given relative frequencies
def rel_freq_60_70 : ℚ := 0.1
def rel_freq_70_80 : ℚ := 0.15
def rel_freq_80_90 : ℚ := 0.3
def rel_freq_90_100 : ℚ := 0.35
def rel_freq_100_110 : ℚ := 0.1

-- Derived value of t
def t : ℚ := 1 / 200

-- Number of participants scoring in the range [80, 90)
def num_participants_80_90 : ℚ := total_participants * (6 * t)

-- Cumulative relative frequencies
def cumulative_rel_freq_70 : ℚ := 0.1
def cumulative_rel_freq_80 : ℚ := 0.1 + 0.15
def cumulative_rel_freq_90 : ℚ := 0.1 + 0.15 + 0.3

-- Minimum score for selecting 1500 out of 3000 participants
def min_score_for_final : ℚ := 78.33

-- Probability distribution and expectation for randomly selecting 4 participants
def prob_dist (i : ℕ) : ℚ := (nat.choose 4 i) * (0.6^(4-i)) * (0.4^i)
def expected_value : ℚ := 4 * 0.4

theorem proof_problem :
(t = 1 / 200) ∧
(num_participants_80_90 = 90) ∧
(min_score_for_final = 78.33) ∧
(∀ i ∈ {0, 1, 2, 3, 4}, prob_dist i = (nat.choose 4 i) * (0.6^(4-i)) * (0.4^i)) ∧
(expected_value = 1.6) := 
by
  -- To be proven
  sorry

end proof_problem_l820_820894


namespace solve_sqrt_eq_l820_820568

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2)^x) + Real.sqrt ((3 - 2 * Real.sqrt 2)^x) = 5) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_sqrt_eq_l820_820568


namespace positive_difference_sums_l820_820218

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820218


namespace whisky_replacement_l820_820879

variable (x : ℝ) -- Original quantity of whisky in the jar
variable (y : ℝ) -- Quantity of whisky replaced

-- Condition: A jar full of whisky contains 40% alcohol
-- Condition: After replacement, the percentage of alcohol is 24%
theorem whisky_replacement (h : 0 < x) : 
  0.40 * x - 0.40 * y + 0.19 * y = 0.24 * x → y = (16 / 21) * x :=
by
  intro h_eq
  -- Sorry for the proof
  sorry

end whisky_replacement_l820_820879


namespace find_speed_of_B_l820_820352

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l820_820352


namespace total_winnings_l820_820852

theorem total_winnings (x : ℝ)
  (h1 : x / 4 = first_person_share)
  (h2 : x / 7 = second_person_share)
  (h3 : third_person_share = 17)
  (h4 : first_person_share + second_person_share + third_person_share = x) :
  x = 28 := 
by sorry

end total_winnings_l820_820852


namespace positive_difference_of_sums_l820_820287

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l820_820287


namespace num_modified_balanced_eq_sum_valid_combinations_l820_820940

def isModifiedBalanced (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  let x1 := n / 1000,
      x2 := (n % 1000) / 100,
      x3 := (n % 100) / 10,
      x4 := n % 10 in
  x1 + x2 + x3 = x4 

def countModifiedBalanced : ℕ :=
  Finset.card {n | isModifiedBalanced n}

theorem num_modified_balanced_eq_sum_valid_combinations :
  countModifiedBalanced = ∑ i in Finset.range 9, 
  (Finset.card { (x1, x2, x3) : Finset (ℕ × ℕ × ℕ) | 
     x1 + x2 + x3 = i + 1 ∧ x1 < 10 ∧ x2 < 10 ∧ x3 < 10 ∧ x1 ≠ 0 }) :=
by
sorry

end num_modified_balanced_eq_sum_valid_combinations_l820_820940


namespace expression_simplify_l820_820709

theorem expression_simplify
  (a b : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = - (1 / (a * b)) :=
by
  sorry

end expression_simplify_l820_820709


namespace polynomial_divisibility_l820_820079

def is_divisible_by (f g : Polynomial ℤ) : Prop :=
  ∃ h : Polynomial ℤ, f = g * h

theorem polynomial_divisibility 
  (m n : ℕ)
  (f := λ x : Polynomial ℤ, x^(3*m+2) + (-x^2 - 1)^(3*n+1) + 1)
  : is_divisible_by (f(X)) (X^2 + X + 1) := 
begin
  sorry
end

end polynomial_divisibility_l820_820079


namespace average_of_set_eq_one_l820_820103

theorem average_of_set_eq_one (s : Set ℤ) (h : s = {-5, -2, 0, 4, 8}) :
  (s.sum id / s.size : ℚ) = 1 :=
by
  -- Convert the set to a list to use list.sum and list.size
  let l := [-5, -2, 0, 4, 8]
  have hl : l.to_finset = s, from by
    rw [Finset.ext_iff]
    intro x
    simp [list.mem_to_finset]
    split
    exact λ h' => by
      cases h with
      | intro _ | intro _ | intro _ | intro _ | intro _ | intro _ => exact h
    exact λ h' => List.mem_cons_of_mem _ $
      match h' with
      | Or.inl rfl | Or.inr (Or.inl rfl) | Or.inr (Or.inr (Or.inl rfl)) => h'
  calc
    (s.sum id / s.size : ℚ) = (l.sum id / l.size : ℚ) := by
      { rw [← hl, ← Finset.sum_coe, ← Finset.size_coe, coe_to_finset] }
    ... = (5 / 5 : ℚ) := by norm_num
    ... = 1 := by norm_num

end average_of_set_eq_one_l820_820103


namespace value_of_3_prime_prime_l820_820320

theorem value_of_3_prime_prime (q : ℤ) (q_prime : ℤ) : (3' : ℤ) = 6 → (3'') = 15 :=
by
  sorry

end value_of_3_prime_prime_l820_820320


namespace find_k_l820_820908

noncomputable def point1 : ℝ × ℝ := (7, 9)
noncomputable def point2 (k : ℝ) : ℝ × ℝ := (-3, k)
noncomputable def point3 : ℝ × ℝ := (-11, 5)

theorem find_k (k : ℝ) (h : ∀ x y, y = point1 ∨ y = point2 k ∨ y = point3 → 
                        ∀ u v, v ≠ y → v = point1 ∨ v = point2 k ∨ v = point3 →
                        (v.1 - y.1) * (u.2 - y.2) = (v.2 - y.2) * (u.1 - y.1) →
                        u = v): 
    k = 61 / 9 :=
begin 
  sorry 
end

end find_k_l820_820908


namespace parabola_shift_l820_820721

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l820_820721


namespace sticker_sum_mod_problem_l820_820310

theorem sticker_sum_mod_problem :
  ∃ N < 100, (N % 6 = 5) ∧ (N % 8 = 6) ∧ (N = 47 ∨ N = 95) ∧ (47 + 95 = 142) :=
by
  sorry

end sticker_sum_mod_problem_l820_820310


namespace rhombus_diagonal_length_l820_820831

theorem rhombus_diagonal_length (AB AC: ℝ) 
  (side_length : AB = 60)
  (section_area : (80 * AC * (Real.sin (Real.pi / 3))) = 7200) :
  (AC^2 + (60^2 + 60^2 - 2*60*60*(Real.cos (Real.pi / 3))) = 60^2 + 60^2) :=
by 
  let BD : ℝ := Real.sqrt (3600)
  have conditions := Real.sqrt (3600) = 60
  rw conditions
  sorry

end rhombus_diagonal_length_l820_820831


namespace student_B_speed_l820_820462

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820462


namespace bicycle_speed_B_l820_820378

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l820_820378


namespace find_max_A_l820_820054

-- Sets and properties under consideration
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1995 }
def A (n : ℕ) : Set ℕ := { x ∈ M | ¬ (19 * x ∈ M ∧ 19 * x ∈ { n' | n' ∈ {k | (k : ℕ) ∈ M} }) }

-- Condition
def condition (A : Set ℕ) := ∀ x ∈ A, 19 * x ∉ A

-- Theorem to prove |A| ≤ 1890
theorem find_max_A (A : Set ℕ) (h : ∀ x ∈ A, 19 * x ∉ A) : |A| ≤ 1890 := by
  sorry

end find_max_A_l820_820054


namespace mrs_santiago_more_roses_l820_820790

theorem mrs_santiago_more_roses :
  58 - 24 = 34 :=
by 
  sorry

end mrs_santiago_more_roses_l820_820790


namespace magnitude_of_b_l820_820685

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_b
  {a b : V}
  (h1 : ∥a - b∥ = real.sqrt 3)
  (h2 : ∥a + b∥ = ∥2 • a - b∥) :
  ∥b∥ = real.sqrt 3 := 
sorry

end magnitude_of_b_l820_820685


namespace find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l820_820860

noncomputable def length_width_rectangle_area_30 : Prop :=
∃ (x y : ℝ), x * y = 30 ∧ 2 * (x + y) = 22 ∧ x = 6 ∧ y = 5

noncomputable def impossible_rectangle_area_32 : Prop :=
¬(∃ (x y : ℝ), x * y = 32 ∧ 2 * (x + y) = 22)

-- Proof statements (without proofs)
theorem find_rectangle_dimensions_area_30 : length_width_rectangle_area_30 :=
sorry

theorem no_rectangle_dimensions_area_32 : impossible_rectangle_area_32 :=
sorry

end find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l820_820860


namespace find_optimal_addition_l820_820147

theorem find_optimal_addition (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ 1000 + (m - 1000) * 0.618 = 2618) →
  (m = 2000 ∨ m = 2618) :=
sorry

end find_optimal_addition_l820_820147


namespace batsman_average_after_17th_inning_l820_820335

theorem batsman_average_after_17th_inning
  (A : ℝ)
  (h1 : A + 10 = (16 * A + 200) / 17)
  : (A = 30 ∧ (A + 10) = 40) :=
by
  sorry

end batsman_average_after_17th_inning_l820_820335


namespace student_B_speed_l820_820447

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l820_820447


namespace valid_pairs_count_l820_820702

def count_valid_pairs : ℕ :=
  let valid_digit_pairs := 
    { (d₁, d₂) ∈ Finset.range 10 ×ˢ Finset.range 10 | d₁ ≠ d₂ } 
  let single_digit_pairs := 
    { (d : ℕ) ∈ Finset.range 10 | d ≠ 0 } 
  
  let case1_count := single_digit_pairs.filter (λ n, n * 999 = n * (999 - n)).card
  let case2_count := valid_digit_pairs.card * 3 -- Adjust count based on specific two-digit condition calculations

  case1_count + case2_count

theorem valid_pairs_count : count_valid_pairs = 170 :=
by 
  sorry

end valid_pairs_count_l820_820702


namespace student_B_speed_l820_820470

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l820_820470


namespace percentage_exceeds_self_l820_820911

theorem percentage_exceeds_self (N : ℝ) (P : ℝ) (hN : N = 75) (h_condition : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end percentage_exceeds_self_l820_820911


namespace find_lambda_l820_820694

open Real

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -1)
def vec_c : ℝ × ℝ := (4, 5)

-- Define the dot product function for two-dimensional vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Definition of perpendicular vectors: their dot product is zero
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_lambda (λ : ℝ) : 
  perpendicular vec_a (vec_b.1 + λ * vec_c.1, vec_b.2 + λ * vec_c.2) →
  λ = 1 / 14 :=
by
  sorry

end find_lambda_l820_820694


namespace positive_difference_sums_l820_820215

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l820_820215


namespace speed_of_student_B_l820_820420

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l820_820420


namespace sum_first_12_terms_l820_820603

-- Defining the basic sequence recurrence relation
def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + (-1 : ℝ) ^ n * a n = 2 * (n : ℝ) - 1

-- Theorem statement: Sum of the first 12 terms of the given sequence is 78
theorem sum_first_12_terms (a : ℕ → ℝ) (h : seq a) : 
  (Finset.range 12).sum a = 78 := 
sorry

end sum_first_12_terms_l820_820603


namespace max_value_of_expression_l820_820775

noncomputable def max_value (x y z : ℝ) : ℝ := x + y^3 + z^4

theorem max_value_of_expression (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
  (h_eq : x^2 + y + z = 1) : 
  max_value x y z ≤ 1 :=
begin
  sorry
end

end max_value_of_expression_l820_820775


namespace positive_difference_eq_250_l820_820164

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l820_820164


namespace arccos_sin_three_l820_820547

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l820_820547


namespace ratio_part_whole_l820_820076

theorem ratio_part_whole (P N : ℕ) 
  (h1 : (1/4 : ℝ) * (1/3 : ℝ) * P = 15) 
  (h2 : 0.40 * N = 180) : 
  P / N = 2 / 5 :=
begin
  sorry
end

end ratio_part_whole_l820_820076


namespace jake_present_weight_l820_820886

theorem jake_present_weight (J S : ℕ) 
  (h1 : J - 32 = 2 * S) 
  (h2 : J + S = 212) : 
  J = 152 := 
by 
  sorry

end jake_present_weight_l820_820886


namespace part_one_part_one_equality_part_two_l820_820599

-- Given constants and their properties
variables (a b c d : ℝ)

-- Statement for the first problem
theorem part_one : a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d ≥ -2 :=
sorry

-- Statement for the equality condition in the first problem
theorem part_one_equality (h : |a| = 1 ∧ |b| = 1 ∧ |c| = 1 ∧ |d| = 1) : 
  a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d = -2 :=
sorry

-- Statement for the second problem (existence of Mk for k >= 4 and odd)
theorem part_two (k : ℕ) (hk1 : 4 ≤ k) (hk2 : k % 2 = 1) : ∃ Mk : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k * a * b * c * d ≥ Mk :=
sorry

end part_one_part_one_equality_part_two_l820_820599


namespace quiz_minimum_correct_l820_820012

theorem quiz_minimum_correct (x : ℕ) (hx : 7 * x + 14 ≥ 120) : x ≥ 16 := 
by sorry

end quiz_minimum_correct_l820_820012


namespace growth_rate_of_portfolio_l820_820029

-- Definitions of the problem conditions
def initial_portfolio := 80
def added_amount := 28
def second_year_growth := 1.10
def final_portfolio := 132

-- Main theorem statement
theorem growth_rate_of_portfolio (r : ℝ) (h : (initial_portfolio * (1 + r) + added_amount) * second_year_growth = final_portfolio) : r = 0.15 :=
sorry

end growth_rate_of_portfolio_l820_820029


namespace smallest_N_marked_cells_l820_820600

theorem smallest_N_marked_cells (k : ℕ) (h : k > 0) : 
  ∃ N : ℕ, (N >= ⌊(k + 1) / 2⌋ * ⌊(k + 2) / 2⌋) ∧ 
           ∀ grid : ℕ × ℕ → bool, 
            (∃ init_marked_cells : set (ℕ × ℕ), 
              (init_marked_cells.card = N) /\
              (∀ A : ℕ × ℕ, 
                (∃ marked_cells : set (ℕ × ℕ),
                  (∀ B : ℕ × ℕ, B ∈ marked_cells → grid B = tt) →
                  (∀ C : ℕ × ℕ, C ∈ grid.cross A → C ∈ marked_cells → grid C = tt → grid A = tt)) →
                 grid.all_marked)) := 
sorry

end smallest_N_marked_cells_l820_820600


namespace minimum_distinct_b_values_l820_820330

def gcd_of_rest (a : Fin 100 → ℕ) (i : Fin 100) : ℕ :=
  Nat.gcd (List.foldr Nat.gcd 0 (List.map (fun j : Fin 100 => if j ≠ i then a j else 0) (Fin.elems 100)))

theorem minimum_distinct_b_values (a : Fin 100 → ℕ) (h : Function.injective a) :
  let b : Fin 100 → ℕ := fun i => a i + gcd_of_rest a i
  ∃ distinct_b : Fin 100 → ℕ, (Set.toFinset (Set.univ : Set (Fin 100))) = 99 :=
sorry

end minimum_distinct_b_values_l820_820330


namespace arccos_sin_1_5_eq_pi_over_2_minus_1_5_l820_820546

-- Define the problem statement in Lean 4.
theorem arccos_sin_1_5_eq_pi_over_2_minus_1_5 : 
  Real.arccos (Real.sin 1.5) = (Real.pi / 2) - 1.5 :=
by
  sorry

end arccos_sin_1_5_eq_pi_over_2_minus_1_5_l820_820546


namespace smallest_y_squared_l820_820040

noncomputable def is_isosceles_trapezoid (P Q R S : Type) (PQ RS PR QS : ℝ) :=
  PQ = 100 ∧ RS = 25 ∧ PR = QS

theorem smallest_y_squared (P Q R S : Type) (PQ PR QS RS y : ℝ)
  (h1 : is_isosceles_trapezoid P Q R S PQ RS PR QS)
  (h2 : PR = QS)
  (h3 : PQ = 100)
  (h4 : RS = 25)
  (h5 : PQ / 2 = 50)
  (h6 : y = Math.sqrt (1875))
  :
  (y * y = 1875) := by
  have PR_def : PR = QS, from h2
  have PR_sq : PR^2 = y^2, by rw [PR_def]
  have def_y : y^2 = 1875, from by sorry
  exact def_y

end smallest_y_squared_l820_820040


namespace speed_of_student_B_l820_820480

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l820_820480


namespace angle_equality_l820_820842

noncomputable def quadrilateral_angles (A B C D : Type*) (angle_CAB angle_ACD angle_ACB angle_CAD : ℝ) (CB CD : ℝ) : Prop :=
  angle_ACD = 2 * angle_CAB ∧
  angle_ACB = 2 * angle_CAD ∧ 
  CB = CD 

theorem angle_equality
  (A B C D : Type*)
  (angle_CAB angle_CAD : ℝ)
  (h1 : quadrilateral_angles A B C D angle_CAB (2 * angle_CAB) (2 * angle_CAD) angle_CAD (CB CD))
  : angle_CAB = angle_CAD :=
by
  sorry

end angle_equality_l820_820842


namespace range_of_b_l820_820630

noncomputable def f (x : ℝ) (b : ℝ) := log x + (x - b)^2

theorem range_of_b (b : ℝ) :
  (∃ I : set ℝ, I ⊆ set.Icc (1/2) 2 ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f x b < f y b) ↔ b < 9/4 :=
by
  sorry

end range_of_b_l820_820630
