import Mathlib

namespace find_other_number_l383_38329

theorem find_other_number (lcm_ab : Nat) (gcd_ab : Nat) (a b : Nat) 
  (hlcm : Nat.lcm a b = lcm_ab) 
  (hgcd : Nat.gcd a b = gcd_ab) 
  (ha : a = 210) 
  (hlcm_ab : lcm_ab = 2310) 
  (hgcd_ab : gcd_ab = 55) 
  : b = 605 := 
by 
  sorry

end find_other_number_l383_38329


namespace barry_shirt_discount_l383_38306

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end barry_shirt_discount_l383_38306


namespace f_diff_ineq_l383_38367

variable {f : ℝ → ℝ}
variable (deriv_f : ∀ x > 0, x * (deriv f x) > 1)

theorem f_diff_ineq (h : ∀ x > 0, x * (deriv f x) > 1) : f 2 - f 1 > Real.log 2 := by 
  sorry

end f_diff_ineq_l383_38367


namespace fourth_triangle_exists_l383_38301

theorem fourth_triangle_exists (a b c d : ℝ)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (h4 : a + b > d) (h5 : a + d > b) (h6 : b + d > a)
  (h7 : a + c > d) (h8 : a + d > c) (h9 : c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b :=
by
  -- I skip the proof with "sorry"
  sorry

end fourth_triangle_exists_l383_38301


namespace f_zero_eq_zero_l383_38346

-- Define the problem conditions
variable {f : ℝ → ℝ}
variables (h_odd : ∀ x : ℝ, f (-x) = -f (x))
variables (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variables (h_eq : ∀ x : ℝ, f (1 - x) - f (1 + x) + 2 * x = 0)
variables (h_mono : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂)

-- State the theorem
theorem f_zero_eq_zero : f 0 = 0 :=
by sorry

end f_zero_eq_zero_l383_38346


namespace ribbon_segment_length_l383_38351

theorem ribbon_segment_length :
  ∀ (ribbon_length : ℚ) (segments : ℕ), ribbon_length = 4/5 → segments = 3 → 
  (ribbon_length / segments) = 4/15 :=
by
  intros ribbon_length segments h1 h2
  sorry

end ribbon_segment_length_l383_38351


namespace insurance_covers_80_percent_l383_38349

def total_cost : ℝ := 300
def out_of_pocket_cost : ℝ := 60
def insurance_coverage : ℝ := 0.8  -- Representing 80%

theorem insurance_covers_80_percent :
  (total_cost - out_of_pocket_cost) / total_cost = insurance_coverage := by
  sorry

end insurance_covers_80_percent_l383_38349


namespace option_C_holds_l383_38384

theorem option_C_holds (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a - b / a > b - a / b := 
  sorry

end option_C_holds_l383_38384


namespace isosceles_triangle_l383_38361

theorem isosceles_triangle 
  (α β γ : ℝ) 
  (a b : ℝ) 
  (h_sum : a + b = (Real.tan (γ / 2)) * (a * (Real.tan α) + b * (Real.tan β)))
  (h_sum_angles : α + β + γ = π) 
  (zero_lt_γ : 0 < γ ∧ γ < π) 
  (zero_lt_α : 0 < α ∧ α < π / 2) 
  (zero_lt_β : 0 < β ∧ β < π / 2) : 
  α = β := 
sorry

end isosceles_triangle_l383_38361


namespace bees_process_2_77_kg_nectar_l383_38375

noncomputable def nectar_to_honey : ℝ :=
  let percent_other_in_nectar : ℝ := 0.30
  let other_mass_in_honey : ℝ := 0.83
  other_mass_in_honey / percent_other_in_nectar

theorem bees_process_2_77_kg_nectar :
  nectar_to_honey = 2.77 :=
by
  sorry

end bees_process_2_77_kg_nectar_l383_38375


namespace div_by_3_l383_38356

theorem div_by_3 (a b : ℤ) : 
  (∃ (k : ℤ), a = 3 * k) ∨ 
  (∃ (k : ℤ), b = 3 * k) ∨ 
  (∃ (k : ℤ), a + b = 3 * k) ∨ 
  (∃ (k : ℤ), a - b = 3 * k) :=
sorry

end div_by_3_l383_38356


namespace pages_read_on_wednesday_l383_38344

theorem pages_read_on_wednesday (W : ℕ) (h : 18 + W + 23 = 60) : W = 19 :=
by {
  sorry
}

end pages_read_on_wednesday_l383_38344


namespace compare_neg_two_powers_l383_38334

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end compare_neg_two_powers_l383_38334


namespace find_n_l383_38352

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l383_38352


namespace total_cost_proof_l383_38308

noncomputable def cost_of_4kg_mangos_3kg_rice_5kg_flour (M R F : ℝ) : ℝ :=
  4 * M + 3 * R + 5 * F

theorem total_cost_proof
  (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 22) :
  cost_of_4kg_mangos_3kg_rice_5kg_flour M R F = 941.6 :=
  sorry

end total_cost_proof_l383_38308


namespace combined_towel_weight_l383_38327

/-
Given:
1. Mary has 5 times as many towels as Frances.
2. Mary has 3 times as many towels as John.
3. The total weight of their towels is 145 pounds.
4. Mary has 60 towels.

To prove: 
The combined weight of Frances's and John's towels is 22.863 kilograms.
-/

theorem combined_towel_weight (total_weight_pounds : ℝ) (mary_towels frances_towels john_towels : ℕ) 
  (conversion_factor : ℝ) (combined_weight_kilograms : ℝ) :
  mary_towels = 60 →
  mary_towels = 5 * frances_towels →
  mary_towels = 3 * john_towels →
  total_weight_pounds = 145 →
  conversion_factor = 0.453592 →
  combined_weight_kilograms = 22.863 :=
by
  sorry

end combined_towel_weight_l383_38327


namespace train_speed_l383_38337

theorem train_speed (v : ℕ) :
  (∀ (d : ℕ), d = 480 → ∀ (ship_speed : ℕ), ship_speed = 60 → 
  (∀ (ship_time : ℕ), ship_time = d / ship_speed →
  (∀ (train_time : ℕ), train_time = ship_time + 2 →
  v = d / train_time))) → v = 48 :=
by
  sorry

end train_speed_l383_38337


namespace calculate_solution_volume_l383_38382

theorem calculate_solution_volume (V : ℝ) (h : 0.35 * V = 1.4) : V = 4 :=
sorry

end calculate_solution_volume_l383_38382


namespace age_difference_l383_38339

noncomputable def years_older (A B : ℕ) : ℕ :=
A - B

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) :
  years_older A B = 9 :=
by
  rw [years_older]
  rw [h1] at h2
  sorry

end age_difference_l383_38339


namespace clever_seven_year_count_l383_38369

def isCleverSevenYear (y : Nat) : Bool :=
  let d1 := y / 1000
  let d2 := (y % 1000) / 100
  let d3 := (y % 100) / 10
  let d4 := y % 10
  d1 + d2 + d3 + d4 = 7

theorem clever_seven_year_count : 
  ∃ n, n = 21 ∧ ∀ y, 2000 ≤ y ∧ y ≤ 2999 → isCleverSevenYear y = true ↔ n = 21 :=
by 
  sorry

end clever_seven_year_count_l383_38369


namespace solve_system_of_equations_l383_38345

variables {a1 a2 a3 a4 : ℝ}

theorem solve_system_of_equations (h_distinct: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :
  ∃ (x1 x2 x3 x4 : ℝ),
    (|a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1) ∧
    (|a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1) ∧
    (|a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1) ∧
    (|a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) ∧
    (x1 = 1 / (a1 - a4)) ∧ (x2 = 0) ∧ (x3 = 0) ∧ (x4 = 1 / (a1 - a4)) :=
sorry

end solve_system_of_equations_l383_38345


namespace pie_split_l383_38336

theorem pie_split (initial_pie : ℚ) (number_of_people : ℕ) (amount_taken_by_each : ℚ) 
  (h1 : initial_pie = 5/6) (h2 : number_of_people = 4) : amount_taken_by_each = 5/24 :=
by
  sorry

end pie_split_l383_38336


namespace sin_cos_identity_l383_38323

theorem sin_cos_identity (α : ℝ) (hα_cos : Real.cos α = 3/5) (hα_sin : Real.sin α = 4/5) : Real.sin α + 2 * Real.cos α = 2 :=
by
  -- Proof omitted
  sorry

end sin_cos_identity_l383_38323


namespace min_value_f_l383_38385

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l383_38385


namespace avg_speed_is_20_l383_38393

-- Define the total distance and total time
def total_distance : ℕ := 100
def total_time : ℕ := 5

-- Define the average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The theorem to prove the average speed given the distance and time
theorem avg_speed_is_20 : average_speed total_distance total_time = 20 :=
by
  sorry

end avg_speed_is_20_l383_38393


namespace parking_lot_total_spaces_l383_38378

theorem parking_lot_total_spaces (ratio_fs_cc : ℕ) (ratio_cc_fs : ℕ) (fs_spaces : ℕ) (total_spaces : ℕ) 
  (h1 : ratio_fs_cc = 11) (h2 : ratio_cc_fs = 4) (h3 : fs_spaces = 330) :
  total_spaces = 450 :=
by
  sorry

end parking_lot_total_spaces_l383_38378


namespace minimum_expression_value_l383_38343

noncomputable def expr (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  (2 * (Real.sin x₁)^2 + 1 / (Real.sin x₁)^2) *
  (2 * (Real.sin x₂)^2 + 1 / (Real.sin x₂)^2) *
  (2 * (Real.sin x₃)^2 + 1 / (Real.sin x₃)^2) *
  (2 * (Real.sin x₄)^2 + 1 / (Real.sin x₄)^2)

theorem minimum_expression_value :
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  x₁ + x₂ + x₃ + x₄ = Real.pi →
  expr x₁ x₂ x₃ x₄ ≥ 81 := sorry

end minimum_expression_value_l383_38343


namespace wrapping_paper_per_present_l383_38398

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l383_38398


namespace A_can_complete_work_in_28_days_l383_38319
noncomputable def work_days_for_A (x : ℕ) (h : 4 / x = 1 / 21) : ℕ :=
  x / 3

theorem A_can_complete_work_in_28_days (x : ℕ) (h : 4 / x = 1 / 21) :
  work_days_for_A x h = 28 :=
  sorry

end A_can_complete_work_in_28_days_l383_38319


namespace time_to_meet_l383_38304

variables {v_g : ℝ} -- speed of Petya and Vasya on the dirt road
variable {v_a : ℝ} -- speed of Petya on the paved road
variable {t : ℝ} -- time from start until Petya and Vasya meet
variable {x : ℝ} -- distance from the starting point to the bridge

-- Conditions
axiom Petya_speed : v_a = 3 * v_g
axiom Petya_bridge_time : x / v_a = 1
axiom Vasya_speed : v_a ≠ 0 ∧ v_g ≠ 0

-- Statement
theorem time_to_meet (h1 : v_a = 3 * v_g) (h2 : x / v_a = 1) (h3 : v_a ≠ 0 ∧ v_g ≠ 0) : t = 2 :=
by
  have h4 : x = 3 * v_g := sorry
  have h5 : (2 * x - 2 * v_g) / (2 * v_g) = 1 := sorry
  exact sorry

end time_to_meet_l383_38304


namespace simplify_and_multiply_expression_l383_38350

variable (b : ℝ)

theorem simplify_and_multiply_expression :
  (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 :=
by
  sorry

end simplify_and_multiply_expression_l383_38350


namespace exists_solution_real_l383_38383

theorem exists_solution_real (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3 / 2 :=
by
  sorry

end exists_solution_real_l383_38383


namespace number_of_balls_in_last_box_l383_38309

noncomputable def box_question (b : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2010 → b i + b (i + 1) = 14 + i) ∧
  (b 1 + b 2011 = 1023)

theorem number_of_balls_in_last_box (b : ℕ → ℕ) (h : box_question b) : b 2011 = 1014 :=
by
  sorry

end number_of_balls_in_last_box_l383_38309


namespace topless_cubical_box_l383_38333

def squares : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def valid_placement (s : Char) : Bool :=
  match s with
  | 'A' => true
  | 'B' => true
  | 'C' => true
  | 'D' => false
  | 'E' => false
  | 'F' => true
  | 'G' => true
  | 'H' => false
  | _ => false

def valid_configurations : List Char := squares.filter valid_placement

theorem topless_cubical_box:
  valid_configurations.length = 5 := by
  sorry

end topless_cubical_box_l383_38333


namespace floor_e_eq_2_l383_38330

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l383_38330


namespace sin_sum_leq_3_sqrt_3_div_2_l383_38394

theorem sin_sum_leq_3_sqrt_3_div_2 (A B C : ℝ) (h_sum : A + B + C = Real.pi) (h_pos : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_leq_3_sqrt_3_div_2_l383_38394


namespace students_liking_both_l383_38354

theorem students_liking_both (total_students sports_enthusiasts music_enthusiasts neither : ℕ)
  (h1 : total_students = 55)
  (h2: sports_enthusiasts = 43)
  (h3: music_enthusiasts = 34)
  (h4: neither = 4) : 
  ∃ x, ((sports_enthusiasts - x) + x + (music_enthusiasts - x) = total_students - neither) ∧ (x = 22) :=
by
  sorry -- Proof omitted

end students_liking_both_l383_38354


namespace area_ratio_l383_38355

variable (A_shape A_triangle : ℝ)

-- Condition: The area ratio given.
axiom ratio_condition : A_shape / A_triangle = 2

-- Theorem statement
theorem area_ratio (A_shape A_triangle : ℝ) (h : A_shape / A_triangle = 2) : A_shape / A_triangle = 2 :=
by
  exact h

end area_ratio_l383_38355


namespace cole_avg_speed_back_home_l383_38348

noncomputable def avg_speed_back_home 
  (speed_to_work : ℚ) 
  (total_round_trip_time : ℚ) 
  (time_to_work : ℚ) 
  (time_in_minutes : ℚ) :=
  let time_to_work_hours := time_to_work / time_in_minutes
  let distance_to_work := speed_to_work * time_to_work_hours
  let time_back_home := total_round_trip_time - time_to_work_hours
  distance_to_work / time_back_home

theorem cole_avg_speed_back_home :
  avg_speed_back_home 75 1 (35/60) 60 = 105 := 
by 
  -- The proof is omitted
  sorry

end cole_avg_speed_back_home_l383_38348


namespace red_flowers_needed_l383_38387

-- Define the number of white and red flowers
def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

-- Define the problem statement.
theorem red_flowers_needed : red_flowers + 208 = white_flowers := by
  -- The proof goes here.
  sorry

end red_flowers_needed_l383_38387


namespace max_A_l383_38366

theorem max_A (A : ℝ) : (∀ (x y : ℕ), 0 < x → 0 < y → 3 * x^2 + y^2 + 1 ≥ A * (x^2 + x * y + x)) ↔ A ≤ 5 / 3 := by
  sorry

end max_A_l383_38366


namespace susan_strawberries_l383_38399

def strawberries_picked (total_in_basket : ℕ) (handful_size : ℕ) (eats_per_handful : ℕ) : ℕ :=
  let strawberries_per_handful := handful_size - eats_per_handful
  (total_in_basket / strawberries_per_handful) * handful_size

theorem susan_strawberries : strawberries_picked 60 5 1 = 75 := by
  sorry

end susan_strawberries_l383_38399


namespace project_completion_advance_l383_38300

variables (a : ℝ) -- efficiency of each worker (units of work per day)
variables (total_days : ℕ) (initial_workers added_workers : ℕ) (fraction_completed : ℝ)
variables (initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency : ℝ)

-- Conditions
def conditions : Prop :=
  total_days = 100 ∧
  initial_workers = 10 ∧
  initial_days = 30 ∧
  fraction_completed = 1 / 5 ∧
  added_workers = 10 ∧
  total_initial_work = initial_workers * initial_days * a * 5 ∧ 
  total_remaining_work = total_initial_work - (initial_workers * initial_days * a) ∧
  total_workers_efficiency = (initial_workers + added_workers) * a ∧
  remaining_days = total_remaining_work / total_workers_efficiency

-- Proof statement
theorem project_completion_advance (h : conditions a total_days initial_workers added_workers fraction_completed initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency) :
  total_days - (initial_days + remaining_days) = 10 :=
  sorry

end project_completion_advance_l383_38300


namespace arithmetic_mean_median_l383_38313

theorem arithmetic_mean_median (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : a = 0) (h4 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end arithmetic_mean_median_l383_38313


namespace erasers_difference_l383_38396

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end erasers_difference_l383_38396


namespace square_free_condition_l383_38341

/-- Define square-free integer -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

/-- Define the problem in Lean -/
theorem square_free_condition (p : ℕ) (hp : p ≥ 3 ∧ Nat.Prime p) :
  (∀ q : ℕ, Nat.Prime q ∧ q < p → square_free (p - (p / q) * q)) ↔
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 13 := by
  sorry

end square_free_condition_l383_38341


namespace randy_trip_total_distance_l383_38365

-- Definition of the problem condition
def randy_trip_length (x : ℝ) : Prop :=
  x / 3 + 20 + x / 5 = x

-- The total length of Randy's trip
theorem randy_trip_total_distance : ∃ x : ℝ, randy_trip_length x ∧ x = 300 / 7 :=
by
  sorry

end randy_trip_total_distance_l383_38365


namespace students_just_passed_l383_38376

theorem students_just_passed (total_students first_div_percent second_div_percent : ℝ)
  (h_total_students: total_students = 300)
  (h_first_div_percent: first_div_percent = 0.29)
  (h_second_div_percent: second_div_percent = 0.54)
  (h_no_failures : total_students = 300) :
  ∃ passed_students, passed_students = total_students - (first_div_percent * total_students + second_div_percent * total_students) ∧ passed_students = 51 :=
by
  sorry

end students_just_passed_l383_38376


namespace sum_tens_ones_digit_l383_38340

theorem sum_tens_ones_digit (a : ℕ) (b : ℕ) (n : ℕ) (h : a - b = 3) :
  let d := (3^n)
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  ones_digit + tens_digit = 9 :=
by 
  let d := 3^17
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  sorry

end sum_tens_ones_digit_l383_38340


namespace average_sleep_hours_l383_38342

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end average_sleep_hours_l383_38342


namespace find_a_and_vertices_find_y_range_find_a_range_l383_38314

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

end find_a_and_vertices_find_y_range_find_a_range_l383_38314


namespace number_of_chickens_l383_38307

theorem number_of_chickens (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 :=
by
  sorry

end number_of_chickens_l383_38307


namespace min_value_frac_inv_l383_38311

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l383_38311


namespace product_of_roots_l383_38325

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ^ 2 - 2 * x1 = 2) (h2 : x2 ^ 2 - 2 * x2 = 2) (hne : x1 ≠ x2) :
  x1 * x2 = -2 := 
sorry

end product_of_roots_l383_38325


namespace hari_joins_l383_38335

theorem hari_joins {x : ℕ} :
  let praveen_start := 3500
  let hari_start := 9000
  let total_months := 12
  (praveen_start * total_months) * 3 = (hari_start * (total_months - x)) * 2
  → x = 5 :=
by
  intros
  sorry

end hari_joins_l383_38335


namespace no_three_integers_exist_l383_38374

theorem no_three_integers_exist (x y z : ℤ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  ((x^2 - 1) % y = 0) ∧ ((x^2 - 1) % z = 0) ∧
  ((y^2 - 1) % x = 0) ∧ ((y^2 - 1) % z = 0) ∧
  ((z^2 - 1) % x = 0) ∧ ((z^2 - 1) % y = 0) → false :=
by
  sorry

end no_three_integers_exist_l383_38374


namespace motorcyclist_initial_speed_l383_38315

theorem motorcyclist_initial_speed (x : ℝ) : 
  (120 = x * (120 / x)) ∧
  (120 = x + 6) → 
  (120 / x = 1 + 1/6 + (120 - x) / (x + 6)) →
  (x = 48) :=
by
  sorry

end motorcyclist_initial_speed_l383_38315


namespace bob_speed_before_construction_l383_38363

theorem bob_speed_before_construction:
  ∀ (v : ℝ),
    (1.5 * v + 2 * 45 = 180) →
    v = 60 :=
by
  intros v h
  sorry

end bob_speed_before_construction_l383_38363


namespace integer_sum_l383_38373

theorem integer_sum {p q r s : ℤ} 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 22 := 
sorry

end integer_sum_l383_38373


namespace cat_and_dog_positions_l383_38318

def cat_position_after_365_moves : Nat :=
  let cycle_length := 9
  365 % cycle_length

def dog_position_after_365_moves : Nat :=
  let cycle_length := 16
  365 % cycle_length

theorem cat_and_dog_positions :
  cat_position_after_365_moves = 5 ∧ dog_position_after_365_moves = 13 :=
by
  sorry

end cat_and_dog_positions_l383_38318


namespace cost_price_of_computer_table_l383_38359

/-- The cost price \(C\) of a computer table is Rs. 7000 -/
theorem cost_price_of_computer_table : 
  ∃ (C : ℝ), (S = 1.20 * C) ∧ (S = 8400) → C = 7000 := 
by 
  sorry

end cost_price_of_computer_table_l383_38359


namespace average_words_written_l383_38372

def total_words : ℕ := 50000
def total_hours : ℕ := 100
def average_words_per_hour : ℕ := total_words / total_hours

theorem average_words_written :
  average_words_per_hour = 500 := 
by
  sorry

end average_words_written_l383_38372


namespace total_wheels_combined_l383_38358

-- Define the counts of vehicles and wheels per vehicle in each storage area
def bicycles_A : ℕ := 16
def tricycles_A : ℕ := 7
def unicycles_A : ℕ := 10
def four_wheelers_A : ℕ := 5

def bicycles_B : ℕ := 12
def tricycles_B : ℕ := 5
def unicycles_B : ℕ := 8
def four_wheelers_B : ℕ := 3

def wheels_bicycle : ℕ := 2
def wheels_tricycle : ℕ := 3
def wheels_unicycle : ℕ := 1
def wheels_four_wheeler : ℕ := 4

-- Calculate total wheels in Storage Area A
def total_wheels_A : ℕ :=
  bicycles_A * wheels_bicycle + tricycles_A * wheels_tricycle + unicycles_A * wheels_unicycle + four_wheelers_A * wheels_four_wheeler
  
-- Calculate total wheels in Storage Area B
def total_wheels_B : ℕ :=
  bicycles_B * wheels_bicycle + tricycles_B * wheels_tricycle + unicycles_B * wheels_unicycle + four_wheelers_B * wheels_four_wheeler

-- Theorem stating that the combined total number of wheels in both storage areas is 142
theorem total_wheels_combined : total_wheels_A + total_wheels_B = 142 := by
  sorry

end total_wheels_combined_l383_38358


namespace minimum_rubles_to_reverse_order_of_chips_100_l383_38347

noncomputable def minimum_rubles_to_reverse_order_of_chips (n : ℕ) : ℕ :=
if n = 100 then 61 else 0

theorem minimum_rubles_to_reverse_order_of_chips_100 :
  minimum_rubles_to_reverse_order_of_chips 100 = 61 :=
by sorry

end minimum_rubles_to_reverse_order_of_chips_100_l383_38347


namespace total_profit_calculation_l383_38364

theorem total_profit_calculation (A B C : ℕ) (C_share total_profit : ℕ) 
  (hA : A = 27000) 
  (hB : B = 72000) 
  (hC : C = 81000) 
  (hC_share : C_share = 36000) 
  (h_ratio : C_share * 20 = total_profit * 9) :
  total_profit = 80000 := by
  sorry

end total_profit_calculation_l383_38364


namespace compare_m_n_l383_38310

theorem compare_m_n (b m n : ℝ) :
  m = -3 * (-2) + b ∧ n = -3 * (3) + b → m > n :=
by
  sorry

end compare_m_n_l383_38310


namespace power_identity_l383_38390

theorem power_identity (x : ℝ) : (x ^ 10 = 25 ^ 5) → x = 5 := by
  sorry

end power_identity_l383_38390


namespace geometric_sequence_a2_a6_l383_38353

theorem geometric_sequence_a2_a6 (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = r * a n) (h₄ : a 4 = 4) :
  a 2 * a 6 = 16 :=
sorry

end geometric_sequence_a2_a6_l383_38353


namespace negation_of_all_squares_positive_l383_38326

theorem negation_of_all_squares_positive :
  ¬ (∀ x : ℝ, x * x > 0) ↔ ∃ x : ℝ, x * x ≤ 0 :=
by sorry

end negation_of_all_squares_positive_l383_38326


namespace negative_subtraction_result_l383_38388

theorem negative_subtraction_result : -2 - 1 = -3 := 
by
  -- The proof is not required by the prompt, so we use "sorry" to indicate the unfinished proof.
  sorry

end negative_subtraction_result_l383_38388


namespace bob_walking_rate_is_12_l383_38389

-- Definitions for the problem
def yolanda_distance := 24
def yolanda_rate := 3
def bob_distance_when_met := 12
def time_yolanda_walked := 2

-- The theorem we need to prove
theorem bob_walking_rate_is_12 : 
  (bob_distance_when_met / (time_yolanda_walked - 1) = 12) :=
by sorry

end bob_walking_rate_is_12_l383_38389


namespace D_72_eq_22_l383_38312

def D(n : ℕ) : ℕ :=
  if n = 72 then 22 else 0 -- the actual function logic should define D properly

theorem D_72_eq_22 : D 72 = 22 :=
  by sorry

end D_72_eq_22_l383_38312


namespace siblings_total_weekly_water_l383_38305

noncomputable def Theo_daily : ℕ := 8
noncomputable def Mason_daily : ℕ := 7
noncomputable def Roxy_daily : ℕ := 9

noncomputable def daily_to_weekly (daily : ℕ) : ℕ := daily * 7

theorem siblings_total_weekly_water :
  daily_to_weekly Theo_daily + daily_to_weekly Mason_daily + daily_to_weekly Roxy_daily = 168 := by
  sorry

end siblings_total_weekly_water_l383_38305


namespace machine_worked_yesterday_l383_38391

noncomputable def shirts_made_per_minute : ℕ := 3
noncomputable def shirts_made_yesterday : ℕ := 9

theorem machine_worked_yesterday : 
  (shirts_made_yesterday / shirts_made_per_minute) = 3 :=
sorry

end machine_worked_yesterday_l383_38391


namespace intersection_S_T_eq_T_l383_38357

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l383_38357


namespace pinning_7_nails_l383_38324

theorem pinning_7_nails {n : ℕ} (circles : Fin n → Set (ℝ × ℝ)) :
  (∀ i j : Fin n, i ≠ j → ∃ p : ℝ × ℝ, p ∈ circles i ∧ p ∈ circles j) →
  ∃ s : Finset (ℝ × ℝ), s.card ≤ 7 ∧ ∀ i : Fin n, ∃ p : ℝ × ℝ, p ∈ s ∧ p ∈ circles i :=
by sorry

end pinning_7_nails_l383_38324


namespace area_at_stage_8_l383_38362

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l383_38362


namespace sum_of_squares_l383_38397

theorem sum_of_squares (x : ℚ) (hx : 7 * x = 15) : 
  (x^2 + (2 * x)^2 + (4 * x)^2 = 4725 / 49) := by
  sorry

end sum_of_squares_l383_38397


namespace molly_age_l383_38303

variable (S M : ℕ)

theorem molly_age (h1 : S / M = 4 / 3) (h2 : S + 6 = 38) : M = 24 :=
by
  sorry

end molly_age_l383_38303


namespace toys_per_day_l383_38316

theorem toys_per_day (total_toys_per_week : ℕ) (days_worked_per_week : ℕ)
  (production_rate_constant : Prop) (h1 : total_toys_per_week = 8000)
  (h2 : days_worked_per_week = 4)
  (h3 : production_rate_constant)
  : (total_toys_per_week / days_worked_per_week) = 2000 :=
by
  sorry

end toys_per_day_l383_38316


namespace screen_to_body_ratio_increases_l383_38338

theorem screen_to_body_ratio_increases
  (a b m : ℝ)
  (h1 : a > b)
  (h2 : 0 < m)
  (h3 : m < 1) :
  (b + m) / (a + m) > b / a :=
by
  sorry

end screen_to_body_ratio_increases_l383_38338


namespace inequality_proof_l383_38395

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end inequality_proof_l383_38395


namespace composite_sum_pow_l383_38360

theorem composite_sum_pow (a b c d : ℕ) (h_pos : a > b ∧ b > c ∧ c > d)
    (h_div : (a + b - c + d) ∣ (a * c + b * d)) (m : ℕ) (h_m_pos : 0 < m) 
    (n : ℕ) (h_n_odd : n % 2 = 1) : ∃ k : ℕ, k > 1 ∧ k ∣ (a ^ n * b ^ m + c ^ m * d ^ n) :=
by
  sorry

end composite_sum_pow_l383_38360


namespace minimum_value_of_f_l383_38317

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - a - 15|

theorem minimum_value_of_f {a : ℝ} (h0 : 0 < a) (h1 : a < 15) : ∃ Q, (∀ x, a ≤ x ∧ x ≤ 15 → f x a ≥ Q) ∧ Q = 15 := by
  sorry

end minimum_value_of_f_l383_38317


namespace cyclic_determinant_zero_l383_38377

open Matrix

-- Define the roots of the polynomial and the polynomial itself.
variables {α β γ δ : ℂ} -- We assume the roots are complex numbers.
variable (p q r : ℂ) -- Coefficients of the polynomial x^4 + px^2 + qx + r = 0

-- Define the matrix whose determinant we want to compute
def cyclic_matrix (α β γ δ : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![
    ![α, β, γ, δ],
    ![β, γ, δ, α],
    ![γ, δ, α, β],
    ![δ, α, β, γ]
  ]

-- Statement of the theorem
theorem cyclic_determinant_zero :
  ∀ (α β γ δ : ℂ) (p q r : ℂ),
  (∀ x : ℂ, x ^ 4 + p * x ^ 2 + q * x + r = 0 → x = α ∨ x = β ∨ x = γ ∨ x = δ) →
  det (cyclic_matrix α β γ δ) = 0 :=
by
  intros α β γ δ p q r hRoots
  sorry

end cyclic_determinant_zero_l383_38377


namespace k_h_neg3_l383_38380

-- Definitions of h and k
def h (x : ℝ) : ℝ := 4 * x^2 - 12

variable (k : ℝ → ℝ) -- function k with range an ℝ

-- Given k(h(3)) = 16
axiom k_h_3 : k (h 3) = 16

-- Prove that k(h(-3)) = 16
theorem k_h_neg3 : k (h (-3)) = 16 :=
sorry

end k_h_neg3_l383_38380


namespace probability_cheryl_same_color_l383_38370

theorem probability_cheryl_same_color :
  let total_marble_count := 12
  let marbles_per_color := 3
  let carol_draw := 3
  let claudia_draw := 3
  let cheryl_draw := total_marble_count - carol_draw - claudia_draw
  let num_colors := 4

  0 < marbles_per_color ∧ marbles_per_color * num_colors = total_marble_count ∧
  0 < carol_draw ∧ carol_draw <= total_marble_count ∧
  0 < claudia_draw ∧ claudia_draw <= total_marble_count - carol_draw ∧
  0 < cheryl_draw ∧ cheryl_draw <= total_marble_count - carol_draw - claudia_draw ∧
  num_colors * (num_colors - 1) > 0
  →
  ∃ (p : ℚ), p = 2 / 55 := 
sorry

end probability_cheryl_same_color_l383_38370


namespace sum_of_two_numbers_l383_38392

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 :=
by
  sorry

end sum_of_two_numbers_l383_38392


namespace highest_price_per_shirt_l383_38321

theorem highest_price_per_shirt (x : ℝ) 
  (num_shirts : ℕ := 20)
  (total_money : ℝ := 180)
  (entrance_fee : ℝ := 5)
  (sales_tax : ℝ := 0.08)
  (whole_number: ∀ p : ℝ, ∃ n : ℕ, p = n) :
  (∀ (price_per_shirt : ℕ), price_per_shirt ≤ 8) :=
by
  sorry

end highest_price_per_shirt_l383_38321


namespace sequence_general_formula_l383_38381

theorem sequence_general_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) > a n)
  (h3 : ∀ n : ℕ, n > 0 → (a (n + 1))^2 - 2 * a n * a (n + 1) + (a n)^2 = 1) :
  ∀ n : ℕ, n > 0 → a n = n :=
by 
  sorry

end sequence_general_formula_l383_38381


namespace union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l383_38379

open Set

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | x^2 - 12*x + 20 < 0 }
def C (a : ℝ) : Set ℝ := { x | x < a }

theorem union_of_A_and_B :
  A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
sorry

theorem complement_of_A_intersect_B :
  ((univ \ A) ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
sorry

theorem intersection_of_A_and_C (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a > 3 :=
sorry

end union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l383_38379


namespace expression_S_max_value_S_l383_38328

section
variable (x t : ℝ)
def f (x : ℝ) := -3 * x^2 + 6 * x

-- Define the integral expression for S(t)
noncomputable def S (t : ℝ) := ∫ x in t..(t + 1), f x

-- Assert the expression for S(t)
theorem expression_S (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) :
  S t = -3 * t^2 + 3 * t + 2 :=
by
  sorry

-- Assert the maximum value of S(t)
theorem max_value_S :
  ∀ t, (0 ≤ t ∧ t ≤ 2) → S t ≤ 5 / 4 :=
by
  sorry

end

end expression_S_max_value_S_l383_38328


namespace mr_roper_lawn_cuts_l383_38322

theorem mr_roper_lawn_cuts (x : ℕ) (h_apr_sep : ℕ → ℕ) (h_total_cuts : 12 * 9 = 108) :
  (6 * x + 6 * 3 = 108) → x = 15 :=
by
  -- The proof is not needed as per the instructions, hence we use sorry.
  sorry

end mr_roper_lawn_cuts_l383_38322


namespace at_least_one_corner_square_selected_l383_38368

theorem at_least_one_corner_square_selected :
  let total_squares := 16
  let total_corners := 4
  let total_non_corners := 12
  let ways_to_select_3_from_total := Nat.choose total_squares 3
  let ways_to_select_3_from_non_corners := Nat.choose total_non_corners 3
  let probability_no_corners := (ways_to_select_3_from_non_corners : ℚ) / ways_to_select_3_from_total
  let probability_at_least_one_corner := 1 - probability_no_corners
  probability_at_least_one_corner = (17 / 28 : ℚ) :=
by
  sorry

end at_least_one_corner_square_selected_l383_38368


namespace repeated_three_digit_divisible_101_l383_38332

theorem repeated_three_digit_divisible_101 (abc : ℕ) (h1 : 100 ≤ abc) (h2 : abc < 1000) :
  (1000000 * abc + 1000 * abc + abc) % 101 = 0 :=
by
  sorry

end repeated_three_digit_divisible_101_l383_38332


namespace range_of_a_l383_38320

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2) 2 → x₂ ∈ Set.Icc (1/2) 2 → (a / x₁ + x₁ * Real.log x₁ ≥ x₂^3 - x₂^2 - 3)) →
  a ∈ Set.Ici 1 :=
by
  sorry

end range_of_a_l383_38320


namespace ratio_is_two_l383_38331

noncomputable def ratio_of_altitude_to_base (area base : ℕ) : ℕ :=
  have h : ℕ := area / base
  h / base

theorem ratio_is_two (area base : ℕ) (h : ℕ)  (h_area : area = 288) (h_base : base = 12) (h_altitude : h = area / base) : ratio_of_altitude_to_base area base = 2 :=
  by
    sorry 

end ratio_is_two_l383_38331


namespace ZYX_syndrome_diagnosis_l383_38386

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end ZYX_syndrome_diagnosis_l383_38386


namespace find_lawn_width_l383_38302

/-- Given a rectangular lawn with a length of 80 m and roads each 10 m wide,
    one running parallel to the length and the other running parallel to the width,
    with a total travel cost of Rs. 3300 at Rs. 3 per sq m, prove that the width of the lawn is 30 m. -/
theorem find_lawn_width (w : ℕ) (h_area_road : 10 * w + 10 * 80 = 1100) : w = 30 :=
by {
  sorry
}

end find_lawn_width_l383_38302


namespace total_money_divided_l383_38371

theorem total_money_divided (A B C : ℝ) (h1 : A = (1 / 2) * B) (h2 : B = (1 / 2) * C) (h3 : C = 208) :
  A + B + C = 364 := 
sorry

end total_money_divided_l383_38371
