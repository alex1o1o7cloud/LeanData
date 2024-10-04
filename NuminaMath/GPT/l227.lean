import Mathlib

namespace triangle_properties_l227_227308

theorem triangle_properties (a b c : ℝ) (h1 : a / b = 5 / 12) (h2 : b / c = 12 / 13) (h3 : a + b + c = 60) :
  (a^2 + b^2 = c^2) ∧ ((1 / 2) * a * b > 100) :=
by
  sorry

end triangle_properties_l227_227308


namespace greatest_a_for_x2_plus_ax_eq_neg24_l227_227022

theorem greatest_a_for_x2_plus_ax_eq_neg24 (a : ℕ) (h : ∃ x : ℤ, x^2 + (a : ℤ) * x = -24) : a ≤ 25 :=
begin
  sorry
end

example (h : ∃ a : ℕ, ∀ x : ℤ, x^2 + (a : ℤ) * x = -24 → a ≤ 25) : true :=
begin
  trivial
end

end greatest_a_for_x2_plus_ax_eq_neg24_l227_227022


namespace trajectory_of_center_l227_227940

-- Define the given conditions
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

def tangent_y_axis (x : ℝ) : Prop := x = 0

-- Define the theorem with the given conditions and the desired conclusion
theorem trajectory_of_center (x y : ℝ) (h1 : tangent_circle x y) (h2 : tangent_y_axis x) :
  (y^2 = 8 * x) ∨ (y = 0 ∧ x ≤ 0) :=
sorry

end trajectory_of_center_l227_227940


namespace range_of_a_intersection_nonempty_range_of_a_intersection_A_l227_227146

noncomputable def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a_intersection_nonempty (a : ℝ) : (A a ∩ B ≠ ∅) ↔ (a < -1 ∨ a > 2) :=
sorry

theorem range_of_a_intersection_A (a : ℝ) : (A a ∩ B = A a) ↔ (a < -4 ∨ a > 5) :=
sorry

end range_of_a_intersection_nonempty_range_of_a_intersection_A_l227_227146


namespace share_pizza_l227_227162

variable (Yoojung_slices Minyoung_slices total_slices : ℕ)
variable (Y : ℕ)

theorem share_pizza :
  Yoojung_slices = Y ∧
  Minyoung_slices = Y + 2 ∧
  total_slices = 10 ∧
  Yoojung_slices + Minyoung_slices = total_slices →
  Y = 4 :=
by
  sorry

end share_pizza_l227_227162


namespace linear_inequalities_solution_range_l227_227161

theorem linear_inequalities_solution_range (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) ↔ m > 2 / 3 :=
by
  sorry

end linear_inequalities_solution_range_l227_227161


namespace necessary_but_not_sufficient_condition_l227_227457

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l227_227457


namespace find_y_l227_227891

theorem find_y (x y : ℕ) (hx_positive : 0 < x) (hy_positive : 0 < y) (hmod : x % y = 9) (hdiv : (x : ℝ) / (y : ℝ) = 96.25) : y = 36 :=
sorry

end find_y_l227_227891


namespace ratio_of_u_to_v_l227_227081

theorem ratio_of_u_to_v (b : ℚ) (hb : b ≠ 0) (u v : ℚ)
  (hu : u = -b / 8) (hv : v = -b / 12) :
  u / v = 3 / 2 :=
by sorry

end ratio_of_u_to_v_l227_227081


namespace leif_fruit_weight_difference_l227_227711

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

end leif_fruit_weight_difference_l227_227711


namespace dog_catches_fox_at_120m_l227_227906

theorem dog_catches_fox_at_120m :
  let initial_distance := 30
  let dog_leap := 2
  let fox_leap := 1
  let dog_leap_frequency := 2
  let fox_leap_frequency := 3
  let dog_distance_per_time_unit := dog_leap * dog_leap_frequency
  let fox_distance_per_time_unit := fox_leap * fox_leap_frequency
  let relative_closure_rate := dog_distance_per_time_unit - fox_distance_per_time_unit
  let time_units_to_catch := initial_distance / relative_closure_rate
  let total_dog_distance := time_units_to_catch * dog_distance_per_time_unit
  total_dog_distance = 120 := sorry

end dog_catches_fox_at_120m_l227_227906


namespace negation_of_universal_abs_nonneg_l227_227007

theorem negation_of_universal_abs_nonneg :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by
  sorry

end negation_of_universal_abs_nonneg_l227_227007


namespace dagger_computation_l227_227362

def dagger (m n p q : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n)) + ((p : ℚ) / m)

theorem dagger_computation :
  dagger 5 9 6 2 (by norm_num) (by norm_num) = 518 / 15 :=
sorry

end dagger_computation_l227_227362


namespace gcd_765432_654321_l227_227587

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227587


namespace sum_of_digits_l227_227318

theorem sum_of_digits (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (h : 10 * x + 6 * x = 16) : x + 6 * x = 7 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l227_227318


namespace parents_survey_l227_227317

theorem parents_survey (W M : ℚ) 
  (h1 : 3/4 * W + 9/10 * M = 84) 
  (h2 : W + M = 100) :
  W = 40 :=
by
  sorry

end parents_survey_l227_227317


namespace square_area_l227_227915

theorem square_area (p : ℝ) (h : p = 20) : (p / 4) ^ 2 = 25 :=
by
  sorry

end square_area_l227_227915


namespace gcd_765432_654321_l227_227600

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227600


namespace int_values_satisfying_inequality_l227_227067

theorem int_values_satisfying_inequality : 
  ∃ (N : ℕ), N = 15 ∧ ∀ (x : ℕ), 9 < x ∧ x < 25 → x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} →
  set.size {x | 9 < x ∧ x < 25 ∧ x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}} = N :=
by
  sorry

end int_values_satisfying_inequality_l227_227067


namespace track_length_eq_900_l227_227252

/-- 
Bruce and Bhishma are running on a circular track. 
The speed of Bruce is 30 m/s and that of Bhishma is 20 m/s.
They start from the same point at the same time in the same direction.
They meet again for the first time after 90 seconds. 
Prove that the length of the track is 900 meters.
-/
theorem track_length_eq_900 :
  let speed_bruce := 30 -- [m/s]
  let speed_bhishma := 20 -- [m/s]
  let time_meet := 90 -- [s]
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  track_length = 900 :=
by
  let speed_bruce := 30
  let speed_bhishma := 20
  let time_meet := 90
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  have : track_length = 900 := by
    sorry
  exact this

end track_length_eq_900_l227_227252


namespace gcd_of_765432_and_654321_l227_227578

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227578


namespace area_ratio_of_circles_l227_227817

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l227_227817


namespace tara_spent_more_on_icecream_l227_227228

def iceCreamCount : ℕ := 19
def yoghurtCount : ℕ := 4
def iceCreamCost : ℕ := 7
def yoghurtCost : ℕ := 1

theorem tara_spent_more_on_icecream :
  (iceCreamCount * iceCreamCost) - (yoghurtCount * yoghurtCost) = 129 := 
  sorry

end tara_spent_more_on_icecream_l227_227228


namespace negation_of_at_most_four_l227_227527

theorem negation_of_at_most_four (n : ℕ) : ¬(n ≤ 4) → n ≥ 5 := 
by
  sorry

end negation_of_at_most_four_l227_227527


namespace frac_mul_eq_l227_227924

theorem frac_mul_eq : (2/3) * (3/8) = 1/4 := 
by 
  sorry

end frac_mul_eq_l227_227924


namespace triangle_inequality_l227_227520

theorem triangle_inequality (a b c : ℝ) (h : a^2 = b^2 + c^2) : 
  (b - c)^2 * (a^2 + 4 * b * c)^2 ≤ 2 * a^6 :=
by
  sorry

end triangle_inequality_l227_227520


namespace calories_consumed_Jean_l227_227330

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l227_227330


namespace trajectory_of_P_is_line_l227_227687

noncomputable def P_trajectory_is_line (a m : ℝ) (P : ℝ × ℝ) : Prop :=
  let A := (-a, 0)
  let B := (a, 0)
  let PA := (P.1 + a) ^ 2 + P.2 ^ 2
  let PB := (P.1 - a) ^ 2 + P.2 ^ 2
  PA - PB = m → P.1 = m / (4 * a)

theorem trajectory_of_P_is_line (a m : ℝ) (h : a ≠ 0) :
  ∀ (P : ℝ × ℝ), (P_trajectory_is_line a m P) := sorry

end trajectory_of_P_is_line_l227_227687


namespace A_inter_B_eq_l227_227513

def A := {x : ℤ | 1 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 3}
def B := {x : ℤ | 5 ≤ x ∧ x < 9}

theorem A_inter_B_eq : A ∩ B = {5, 6, 7} :=
by sorry

end A_inter_B_eq_l227_227513


namespace five_times_seven_divided_by_ten_l227_227611

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end five_times_seven_divided_by_ten_l227_227611


namespace overall_average_of_25_results_l227_227534

theorem overall_average_of_25_results (first_12_avg last_12_avg thirteenth_result : ℝ) 
  (h1 : first_12_avg = 14) (h2 : last_12_avg = 17) (h3 : thirteenth_result = 78) :
  (12 * first_12_avg + thirteenth_result + 12 * last_12_avg) / 25 = 18 :=
by
  sorry

end overall_average_of_25_results_l227_227534


namespace intersection_polar_coords_l227_227810

-- Definitions from conditions
def parametric_c1 (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, 1 + Real.sin θ)
def polar_c2 (ρ : ℝ) : Prop := ρ = 1

-- Lean proof problem
theorem intersection_polar_coords : 
  (∀ (θ : ℝ), (parametric_c1 θ).fst = 1 + Real.cos θ ∧ (parametric_c1 θ).snd = 1 + Real.sin θ) ∧
  polar_c2 1 ∧
  (∀ x y, (x - 1)^2 + (y - 1)^2 = 1 ↔ ∃ θ, (x, y) = parametric_c1 θ) ∧
  (∀ x y, x^2 + y^2 = 1 ↔ ∃ ρ θ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ polar_c2 ρ) →
  (∃ (θ1 θ2 : ℝ), (1 + Real.cos θ1, 1 + Real.sin θ1) = (1, 0) ∧ (1, 0) = (1 * Real.cos 0, 1 * Real.sin 0) ∧
                  (1 + Real.cos θ2, 1 + Real.sin θ2) = (0, 1) ∧ (0, 1) = (1 * Real.cos (Real.pi / 2), 1 * Real.sin (Real.pi / 2))) :=
begin
  sorry
end

end intersection_polar_coords_l227_227810


namespace remainder_of_division_l227_227446

variable (a : ℝ) (b : ℝ)

theorem remainder_of_division : a = 28 → b = 10.02 → ∃ r : ℝ, 0 ≤ r ∧ r < b ∧ ∃ q : ℤ, a = q * b + r ∧ r = 7.96 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end remainder_of_division_l227_227446


namespace spider_paths_l227_227416

theorem spider_paths : (Nat.choose (7 + 3) 3) = 210 := 
by
  sorry

end spider_paths_l227_227416


namespace spending_50_dollars_l227_227973

-- Defining the conditions as per the problem
def receiving (x : ℤ) := x
def spending (x : ℤ) := -x

-- Stating the theorem to be proved
theorem spending_50_dollars :
  receiving 80 = 80 → spending 50 = -50 :=
begin
  intros h,
  -- Leaving the proof for now
  sorry,
end

end spending_50_dollars_l227_227973


namespace gcd_10293_29384_l227_227781

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end gcd_10293_29384_l227_227781


namespace darry_steps_l227_227112

theorem darry_steps (f_steps : ℕ) (f_times : ℕ) (s_steps : ℕ) (s_times : ℕ) (no_other_steps : ℕ)
  (hf : f_steps = 11)
  (hf_times : f_times = 10)
  (hs : s_steps = 6)
  (hs_times : s_times = 7)
  (h_no_other : no_other_steps = 0) :
  (f_steps * f_times + s_steps * s_times + no_other_steps = 152) :=
by
  sorry

end darry_steps_l227_227112


namespace they_met_on_wednesday_l227_227346

theorem they_met_on_wednesday 
  (D : ℕ) -- total distance of princess's journey
  (d : ℕ) -- number of days traveled by princess to cover 1/5 of her journey
  (prince_days : ℕ) -- days traveled by prince
  (total_days : ℕ) -- total days from meeting to arrival at castle
  (start_day : ℕ) -- start day of princess's journey (Friday)
  (meet_day : ℕ) -- day they met
  
  (h1 : prince_days = 2)
  (h2 : d = 2)
  (h3 : total_days = 11)
  (h4 : start_day = 5) -- Assume 5 represents Friday as start day
  (h5 : meet_day = start_day + 2) -- After 2 days
  
  : meet_day % 7 = 3 :=
by
  sorry

end they_met_on_wednesday_l227_227346


namespace no_solution_ineq_l227_227829

theorem no_solution_ineq (m : ℝ) : 
  (∀ x : ℝ, x - m ≥ 0 → ¬(0.5 * x + 0.5 < 2)) → m ≥ 3 :=
by
  sorry

end no_solution_ineq_l227_227829


namespace fraction_value_l227_227608

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end fraction_value_l227_227608


namespace find_certain_number_l227_227902

theorem find_certain_number (x : ℝ) : 
  ((2 * (x + 5)) / 5 - 5 = 22) → x = 62.5 :=
by
  intro h
  -- Proof goes here
  sorry

end find_certain_number_l227_227902


namespace count_integers_satisfying_sqrt_condition_l227_227046

noncomputable def count_integers_in_range (lower upper : ℕ) : ℕ :=
    (upper - lower + 1)

/- Proof statement for the given problem -/
theorem count_integers_satisfying_sqrt_condition :
  let conditions := (∀ x : ℕ, 5 > Real.sqrt x ∧ Real.sqrt x > 3) in
  count_integers_in_range 10 24 = 15 :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l227_227046


namespace number_of_integers_inequality_l227_227048

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end number_of_integers_inequality_l227_227048


namespace labor_union_trees_l227_227237

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end labor_union_trees_l227_227237


namespace monthlyShoeSales_l227_227335

-- Defining the conditions
def pairsSoldLastWeek := 27
def pairsSoldThisWeek := 12
def pairsNeededToMeetGoal := 41

-- Defining the question as a statement to prove
theorem monthlyShoeSales : pairsSoldLastWeek + pairsSoldThisWeek + pairsNeededToMeetGoal = 80 := by
  sorry

end monthlyShoeSales_l227_227335


namespace correct_completion_l227_227923

-- Definitions of conditions
def sentence_template := "By the time he arrives, all the work ___, with ___ our teacher will be content."
def option_A := ("will be accomplished", "that")
def option_B := ("will have been accomplished", "which")
def option_C := ("will have accomplished", "it")
def option_D := ("had been accomplished", "him")

-- The actual proof statement
theorem correct_completion : (option_B.fst = "will have been accomplished") ∧ (option_B.snd = "which") :=
by
  sorry

end correct_completion_l227_227923


namespace correct_propositions_l227_227769

def P (x : ℝ) : Prop := x^2 + x + 1 < 0
def neg_P (x : ℝ) : Prop := x^2 + x + 1 ≥ 0

-- Statement that represents the problem and asserts that proposition ② and ③ are correct
theorem correct_propositions :
  (∀ (x : ℝ), x^2 + x + 1 ≥ 0) ∧
  (∀ P Q : Prop, ¬P → (P ∨ Q) → Q) :=
by
  -- Proposition ②
  have prop_2 : ∀ (x : ℝ), neg_P x := sorry
  
  -- Proposition ③
  have prop_3 : ∀ P Q : Prop, (¬P → (P ∨ Q) → Q) := sorry
  
  exact ⟨prop_2, prop_3⟩

end correct_propositions_l227_227769


namespace intersection_of_A_and_B_l227_227288

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 5} := by
  sorry

end intersection_of_A_and_B_l227_227288


namespace gcd_765432_654321_l227_227593

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227593


namespace fractional_part_frustum_l227_227424

noncomputable def base_edge : ℝ := 24
noncomputable def original_altitude : ℝ := 18
noncomputable def smaller_altitude : ℝ := original_altitude / 3

noncomputable def volume_pyramid (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * altitude

noncomputable def volume_original : ℝ := volume_pyramid base_edge original_altitude
noncomputable def similarity_ratio : ℝ := (smaller_altitude / original_altitude) ^ 3
noncomputable def volume_smaller : ℝ := similarity_ratio * volume_original
noncomputable def volume_frustum : ℝ := volume_original - volume_smaller

noncomputable def fractional_volume_frustum : ℝ := volume_frustum / volume_original

theorem fractional_part_frustum : fractional_volume_frustum = 26 / 27 := by
  sorry

end fractional_part_frustum_l227_227424


namespace side_length_of_square_l227_227394

theorem side_length_of_square (total_length : ℝ) (sides : ℕ) (h1 : total_length = 100) (h2 : sides = 4) :
  (total_length / (sides : ℝ) = 25) :=
by
  sorry

end side_length_of_square_l227_227394


namespace find_a_l227_227949

noncomputable def p (a : ℝ) : Prop := 3 < a ∧ a < 7/2
noncomputable def q (a : ℝ) : Prop := a > 3 ∧ a ≠ 7/2
theorem find_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) (hpq : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : a > 7/2 :=
sorry

end find_a_l227_227949


namespace c_work_rate_l227_227225

noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem c_work_rate (A B C: ℝ) 
  (h1 : A + B = work_rate 28) 
  (h2 : A + B + C = work_rate 21) : C = work_rate 84 := by
  -- Proof will go here
  sorry

end c_work_rate_l227_227225


namespace parking_lot_length_l227_227182

theorem parking_lot_length (W : ℝ) (U : ℝ) (A_car : ℝ) (N_cars : ℕ) (H_w : W = 400) (H_u : U = 0.80) (H_Acar : A_car = 10) (H_Ncars : N_cars = 16000) :
  (U * (W * L) = N_cars * A_car) → (L = 500) :=
by
  sorry

end parking_lot_length_l227_227182


namespace cos_alpha_solution_l227_227127

open Real

theorem cos_alpha_solution
  (α : ℝ)
  (h1 : π < α)
  (h2 : α < 3 * π / 2)
  (h3 : tan α = 2) :
  cos α = -sqrt (1 / (1 + 2^2)) :=
by
  sorry

end cos_alpha_solution_l227_227127


namespace sequence_diff_ge_abs_m_l227_227991

-- Define the conditions and theorem in Lean

theorem sequence_diff_ge_abs_m
    (m : ℤ) (h_m : |m| ≥ 2)
    (a : ℕ → ℤ)
    (h_seq_not_zero : ¬ (a 1 = 0 ∧ a 2 = 0))
    (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - m * a n)
    (r s : ℕ) (h_r : r > s) (h_s : s ≥ 2)
    (h_equal : a r = a 1 ∧ a s = a 1) :
    r - s ≥ |m| :=
by
  sorry

end sequence_diff_ge_abs_m_l227_227991


namespace no_2000_digit_perfect_square_with_1999_digits_of_5_l227_227115

theorem no_2000_digit_perfect_square_with_1999_digits_of_5 :
  ¬ (∃ n : ℕ,
      (Nat.digits 10 n).length = 2000 ∧
      ∃ k : ℕ, n = k * k ∧
      (Nat.digits 10 n).count 5 ≥ 1999) :=
sorry

end no_2000_digit_perfect_square_with_1999_digits_of_5_l227_227115


namespace arithmetic_sequence_sum_l227_227706

variable (a_n : ℕ → ℕ)

theorem arithmetic_sequence_sum (h1: a_n 1 + a_n 2 = 5) (h2 : a_n 3 + a_n 4 = 7) (arith : ∀ n, a_n (n + 1) - a_n n = a_n 2 - a_n 1) :
  a_n 5 + a_n 6 = 9 := 
sorry

end arithmetic_sequence_sum_l227_227706


namespace specific_clothing_choice_probability_l227_227482

noncomputable def probability_of_specific_clothing_choice : ℚ :=
  let total_clothing := 4 + 5 + 6
  let total_ways_to_choose_3 := Nat.choose 15 3
  let ways_to_choose_specific_3 := 4 * 5 * 6
  let probability := ways_to_choose_specific_3 / total_ways_to_choose_3
  probability

theorem specific_clothing_choice_probability :
  probability_of_specific_clothing_choice = 24 / 91 :=
by
  -- proof here 
  sorry

end specific_clothing_choice_probability_l227_227482


namespace invalid_transformation_of_equation_l227_227895

theorem invalid_transformation_of_equation (x y m : ℝ) (h : x = y) :
  (m = 0 → (x = y → x / m = y / m)) = false :=
by
  sorry

end invalid_transformation_of_equation_l227_227895


namespace increasing_function_greater_at_a_squared_plus_one_l227_227849

variable (f : ℝ → ℝ) (a : ℝ)

def strictly_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_greater_at_a_squared_plus_one :
  strictly_increasing f → f (a^2 + 1) > f a :=
by
  sorry

end increasing_function_greater_at_a_squared_plus_one_l227_227849


namespace prism_faces_vertices_l227_227102

theorem prism_faces_vertices {L E F V : ℕ} (hE : E = 21) (hEdges : E = 3 * L) 
    (hF : F = L + 2) (hV : V = L) : F = 9 ∧ V = 7 :=
by
  sorry

end prism_faces_vertices_l227_227102


namespace integer_values_count_l227_227073

theorem integer_values_count (x : ℕ) : (∃ y : ℤ, 10 ≤ y ∧ y ≤ 24) ↔ (∑ y in (finset.interval 10 24), 1) = 15 :=
by
  sorry

end integer_values_count_l227_227073


namespace smallest_cars_number_l227_227980

theorem smallest_cars_number :
  ∃ N : ℕ, N > 2 ∧ (N % 5 = 2) ∧ (N % 6 = 2) ∧ (N % 7 = 2) ∧ N = 212 := by
  sorry

end smallest_cars_number_l227_227980


namespace six_lines_regions_l227_227014

def number_of_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

theorem six_lines_regions (h1 : 6 > 0) : 
    number_of_regions 6 = 22 :=
by 
  -- Use the formula for calculating number of regions:
  -- number_of_regions n = 1 + n + (n * (n - 1) / 2)
  sorry

end six_lines_regions_l227_227014


namespace intersection_M_N_l227_227475

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

theorem intersection_M_N : M ∩ N = {0, 3} := by
  sorry

end intersection_M_N_l227_227475


namespace ratio_of_areas_of_circles_l227_227821

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l227_227821


namespace cricket_team_members_l227_227631

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

end cricket_team_members_l227_227631


namespace simplify_and_evaluate_expr_l227_227349

theorem simplify_and_evaluate_expr (x : Real) (h : x = Real.sqrt 3 - 1) :
  1 - (x / (x + 1)) / (x / (x ^ 2 - 1)) = 3 - Real.sqrt 3 :=
sorry

end simplify_and_evaluate_expr_l227_227349


namespace x_intercept_of_line_l227_227936

theorem x_intercept_of_line : ∃ x : ℚ, 3 * x + 5 * 0 = 20 ∧ (x, 0) = (20/3, 0) :=
by
  sorry

end x_intercept_of_line_l227_227936


namespace prove_seq_properties_l227_227674

theorem prove_seq_properties (a b : ℕ → ℕ) (S T : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_sum : ∀ n, 2 * S n = a n ^ 2 + n)
  (h_b : ∀ n, b n = a (n + 1) * 2 ^ n)
  : (∀ n, a n = n) ∧ (∀ n, T n = n * 2 ^ (n + 1)) :=
sorry

end prove_seq_properties_l227_227674


namespace gcd_765432_654321_l227_227595

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227595


namespace max_sum_x_y_l227_227806

theorem max_sum_x_y (x y : ℝ) (h1 : x^2 + y^2 = 7) (h2 : x^3 + y^3 = 10) : x + y ≤ 4 :=
sorry

end max_sum_x_y_l227_227806


namespace reciprocal_of_2023_l227_227374

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l227_227374


namespace faster_pipe_rate_l227_227718

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

end faster_pipe_rate_l227_227718


namespace cost_of_computer_game_is_90_l227_227341

-- Define the costs of individual items
def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

-- Define the number of items
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_quantity : ℕ := 1

-- Calculate the total cost before rebate
def total_cost_before_rebate : ℕ :=
  total_cost_after_rebate + rebate

-- Calculate the total cost of polo shirts and necklaces
def total_cost_polo_necklaces : ℕ :=
  (polo_shirt_quantity * polo_shirt_price) + (necklace_quantity * necklace_price)

-- Define the unknown cost of the computer game
def computer_game_price : ℕ :=
  total_cost_before_rebate - total_cost_polo_necklaces

-- Prove the cost of the computer game
theorem cost_of_computer_game_is_90 : computer_game_price = 90 := by
  -- The following line is a placeholder for the actual proof
  sorry

end cost_of_computer_game_is_90_l227_227341


namespace correct_regression_equation_l227_227375

variable (x y : ℝ)

-- Assume that y is negatively correlated with x
axiom negative_correlation : x * y ≤ 0

-- The candidate regression equations
def regression_A : ℝ := -2 * x - 100
def regression_B : ℝ := 2 * x - 100
def regression_C : ℝ := -2 * x + 100
def regression_D : ℝ := 2 * x + 100

-- Prove that the correct regression equation reflecting the negative correlation is regression_C
theorem correct_regression_equation : regression_C x = -2 * x + 100 := by
  sorry

end correct_regression_equation_l227_227375


namespace miriam_pushups_l227_227855

theorem miriam_pushups :
  let p_M := 5
  let p_T := 7
  let p_W := 2 * p_T
  let p_Th := (p_M + p_T + p_W) / 2
  let p_F := p_M + p_T + p_W + p_Th
  p_F = 39 := by
  sorry

end miriam_pushups_l227_227855


namespace no_real_solution_for_x_l227_227463

theorem no_real_solution_for_x
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/x = 1/3) :
  false :=
by
  sorry

end no_real_solution_for_x_l227_227463


namespace simple_interest_rate_problem_l227_227425

noncomputable def simple_interest_rate (P : ℝ) (T : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - P) * 100 / (P * T)

theorem simple_interest_rate_problem
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : T = 2)
  (h2 : final_amount = (7 / 6) * P)
  (h3 : simple_interest_rate P T final_amount = R) : 
  R = 100 / 12 := sorry

end simple_interest_rate_problem_l227_227425


namespace vector_parallel_eq_l227_227298

theorem vector_parallel_eq (m : ℝ) : 
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  a.1 * b.2 = a.2 * b.1 -> m = -6 := 
by 
  sorry

end vector_parallel_eq_l227_227298


namespace even_integer_squares_l227_227114

noncomputable def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 25

theorem even_integer_squares (x : ℤ) (hx : x % 2 = 0) :
  (∃ (a : ℤ), Q x = a ^ 2) → x = 8 :=
by
  sorry

end even_integer_squares_l227_227114


namespace number_of_integers_satisfying_sqrt_condition_l227_227040

noncomputable def count_integers_satisfying_sqrt_condition : ℕ :=
  let S := {x : ℕ | 3 < real.sqrt x ∧ real.sqrt x < 5}
  finset.card (finset.filter (λ x, 3 < real.sqrt x ∧ real.sqrt x < 5) (finset.range 26))

theorem number_of_integers_satisfying_sqrt_condition :
  count_integers_satisfying_sqrt_condition = 15 :=
sorry

end number_of_integers_satisfying_sqrt_condition_l227_227040


namespace price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars_l227_227364

-- Defining the known quantities
def price_of_two_kg_sugar_and_five_kg_salt : ℝ := 5.50
def price_per_kg_sugar : ℝ := 1.50

-- Defining the variables for the proof
def price_per_kg_salt := (price_of_two_kg_sugar_and_five_kg_salt - 2 * price_per_kg_sugar) / 5

def price_of_three_kg_sugar_and_one_kg_salt := 3 * price_per_kg_sugar + price_per_kg_salt

-- The theorem stating the result
theorem price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars :
  price_of_three_kg_sugar_and_one_kg_salt = 5.00 :=
by
  -- Calculate intermediary values for sugar and salt costs
  let price_of_two_kg_sugar := 2 * price_per_kg_sugar
  let price_of_five_kg_salt := price_of_two_kg_sugar_and_five_kg_salt - price_of_two_kg_sugar
  let price_per_kg_salt' := price_of_five_kg_salt / 5

  -- Calculate final price for verification
  let price_of_three_kg_sugar := 3 * price_per_kg_sugar
  let final_price := price_of_three_kg_sugar + price_per_kg_salt'

  -- Assert the final price is $5.00
  have h1 : price_of_two_kg_sugar = 3.00 := by sorry
  have h2 : price_of_five_kg_salt = 2.50 := by sorry
  have h3 : price_per_kg_salt' = 0.50 := by sorry
  have h4 : final_price = 5.00 := by sorry

  -- Conclude the proof
  exact h4

end price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars_l227_227364


namespace same_number_of_friends_l227_227753

open Finset

variable {A : Type*} [DecidableEq A] [Fintype A]

def knows (G : A → A → Prop) (a1 a2 : A) : Prop := G a1 a2

def common_friends (G : A → A → Prop) (a1 a2 : A) : Finset A :=
  {x | G a1 x ∧ G a2 x}.toFinset

noncomputable def friends (G : A → A → Prop) (a : A) : Finset A :=
  {x | G a x}.toFinset

theorem same_number_of_friends 
  (G : A → A → Prop)
  (h_symmetric : ∀ a b, G a b → G b a)
  (h_known : ∀ a b c, ¬ G a b → G a c → G b c → G c a)
  {a1 a2 : A}
  (h_a1_a2 : knows G a1 a2)
  (h_no_common_friends : common_friends G a1 a2 = ∅) :
  friends G a1 = friends G a2 :=
sorry

end same_number_of_friends_l227_227753


namespace one_greater_than_17_over_10_l227_227333

theorem one_greater_than_17_over_10 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a + b + c = a * b * c) : 
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
by
  sorry

end one_greater_than_17_over_10_l227_227333


namespace gcd_proof_l227_227554

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227554


namespace problem1_problem2_problem3_problem4_l227_227774

variable (a b c : ℝ)

theorem problem1 : a^4 * (a^2)^3 = a^10 :=
by
  sorry

theorem problem2 : 2 * a^3 * b^2 * c / (1 / 3 * a^2 * b) = 6 * a * b * c :=
by
  sorry

theorem problem3 : 6 * a * (1 / 3 * a * b - b) - (2 * a * b + b) * (a - 1) = -5 * a * b + b :=
by
  sorry

theorem problem4 : (a - 2)^2 - (3 * a + 2 * b) * (3 * a - 2 * b) = -8 * a^2 - 4 * a + 4 + 4 * b^2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l227_227774


namespace blue_polygons_exceed_red_polygons_by_1770_l227_227515

theorem blue_polygons_exceed_red_polygons_by_1770:
  ∃ red_points blue_point,
  set.card red_points = 60 ∧ 
  blue_point ∉ red_points ∧
  ∀ polygons, polygons ⊆ red_points ∨ polygons ⊆ red_points ∪ {blue_point} →
  (count_blue_polygons red_points blue_point) - (count_red_polygons red_points) = 1770 :=
by
  sorry

end blue_polygons_exceed_red_polygons_by_1770_l227_227515


namespace find_n_coins_l227_227621

def num_coins : ℕ := 5

theorem find_n_coins (n : ℕ) (h : (n^2 + n + 2) = 2^n) : n = num_coins :=
by {
  -- Proof to be filled in
  sorry
}

end find_n_coins_l227_227621


namespace cookies_left_l227_227502

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end cookies_left_l227_227502


namespace positive_integer_prime_condition_l227_227754

theorem positive_integer_prime_condition (n : ℕ) 
  (h1 : 0 < n)
  (h2 : ∀ (k : ℕ), k < n → Nat.Prime (4 * k^2 + n)) : 
  n = 3 ∨ n = 7 := 
sorry

end positive_integer_prime_condition_l227_227754


namespace negation_is_correct_l227_227868

-- Define the original proposition as a predicate on real numbers.
def original_prop : Prop := ∀ x : ℝ, 4*x^2 - 3*x + 2 < 0

-- State the negation of the original proposition
def negation_of_original_prop : Prop := ∃ x : ℝ, 4*x^2 - 3*x + 2 ≥ 0

-- The theorem to prove the correctness of the negation of the original proposition
theorem negation_is_correct : ¬original_prop ↔ negation_of_original_prop := by
  sorry

end negation_is_correct_l227_227868


namespace average_students_present_l227_227494

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

end average_students_present_l227_227494


namespace student_correct_answers_l227_227092

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 73) : C = 91 :=
sorry

end student_correct_answers_l227_227092


namespace sqrt_meaningful_range_l227_227828

theorem sqrt_meaningful_range (x : ℝ) (h : x - 2 ≥ 0) : x ≥ 2 :=
by {
  sorry
}

end sqrt_meaningful_range_l227_227828


namespace John_and_Rose_work_together_l227_227710

theorem John_and_Rose_work_together (John_work_days : ℕ) (Rose_work_days : ℕ) (combined_work_days: ℕ) 
  (hJohn : John_work_days = 10) (hRose : Rose_work_days = 40) :
  combined_work_days = 8 :=
by 
  sorry

end John_and_Rose_work_together_l227_227710


namespace exponent_of_5_in_30_factorial_l227_227175

open Nat

theorem exponent_of_5_in_30_factorial : padic_val_nat 5 (factorial 30) = 7 := 
  sorry

end exponent_of_5_in_30_factorial_l227_227175


namespace average_of_integers_l227_227740

theorem average_of_integers (A B C D : ℤ) (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D = 90) (h5 : 5 ≤ A) (h6 : A ≠ B ∧ B ≠ C ∧ C ≠ D) :
  (A + B + C + D) / 4 = 27 :=
by
  sorry

end average_of_integers_l227_227740


namespace total_legs_of_collection_l227_227339

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end total_legs_of_collection_l227_227339


namespace perimeter_of_playground_l227_227638

theorem perimeter_of_playground 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 900) 
  (h2 : x * y = 216) : 
  2 * (x + y) = 72 := 
by 
  sorry

end perimeter_of_playground_l227_227638


namespace conclusion_1_conclusion_2_l227_227449

open Function

-- Conclusion ①
theorem conclusion_1 {f : ℝ → ℝ} (h : StrictMono f) :
  ∀ {x1 x2 : ℝ}, f x1 ≤ f x2 ↔ x1 ≤ x2 := 
by
  intros x1 x2
  exact h.le_iff_le

-- Conclusion ②
theorem conclusion_2 {f : ℝ → ℝ} (h : ∀ x, f x ^ 2 = f (-x) ^ 2) :
  ¬ (∀ x, f (-x) = f x ∨ f (-x) = -f x) :=
by
  sorry

end conclusion_1_conclusion_2_l227_227449


namespace find_f_1000_l227_227024

noncomputable def f : ℕ → ℕ := sorry

axiom f_property1 : ∀ n : ℕ, 0 < n → f(f(n)) = 2*n
axiom f_property2 : ∀ n : ℕ, 0 < n → f(3*n + 1) = 3*n + 2

theorem find_f_1000 : f(1000) = 1008 :=
by {
  have h0 : 0 < 1000 := by norm_num,
  sorry
}

end find_f_1000_l227_227024


namespace trigonometric_identity_l227_227948

variable (α : Real)

theorem trigonometric_identity :
  (Real.tan (α - Real.pi / 4) = 1 / 2) →
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) :=
by
  intro h
  sorry

end trigonometric_identity_l227_227948


namespace additional_track_length_l227_227636

theorem additional_track_length (elevation_gain : ℝ) (orig_grade new_grade : ℝ) (Δ_track : ℝ) :
  elevation_gain = 800 ∧ orig_grade = 0.04 ∧ new_grade = 0.015 ∧ Δ_track = ((elevation_gain / new_grade) - (elevation_gain / orig_grade)) ->
  Δ_track = 33333 :=
by sorry

end additional_track_length_l227_227636


namespace circle_area_ratio_is_correct_l227_227825

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l227_227825


namespace alcohol_percentage_after_additions_l227_227897

/-
Problem statement:
A 40-liter solution of alcohol and water is 5% alcohol. If 4.5 liters of alcohol and 5.5 liters of water are added to this solution, what percent of the solution produced is alcohol?

Conditions:
1. Initial solution volume = 40 liters
2. Initial percentage of alcohol = 5%
3. Volume of alcohol added = 4.5 liters
4. Volume of water added = 5.5 liters

Correct answer:
The percent of the solution that is alcohol after the additions is 13%.
-/

theorem alcohol_percentage_after_additions (initial_volume : ℝ) (initial_percentage : ℝ) 
  (alcohol_added : ℝ) (water_added : ℝ) :
  initial_volume = 40 ∧ initial_percentage = 5 ∧ alcohol_added = 4.5 ∧ water_added = 5.5 →
  ((initial_percentage / 100 * initial_volume + alcohol_added) / (initial_volume + alcohol_added + water_added) * 100) = 13 :=
by simp; sorry

end alcohol_percentage_after_additions_l227_227897


namespace store_profit_in_february_l227_227420

variable (C : ℝ)

def initialSellingPrice := C * 1.20
def secondSellingPrice := initialSellingPrice C * 1.25
def finalSellingPrice := secondSellingPrice C * 0.88

theorem store_profit_in_february
  (initialSellingPrice_eq : initialSellingPrice C = C * 1.20)
  (secondSellingPrice_eq : secondSellingPrice C = initialSellingPrice C * 1.25)
  (finalSellingPrice_eq : finalSellingPrice C = secondSellingPrice C * 0.88)
  : finalSellingPrice C - C = 0.32 * C :=
sorry

end store_profit_in_february_l227_227420


namespace option_C_is_always_odd_l227_227391

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_C_is_always_odd (k : ℤ) : is_odd (2007 + 2 * k ^ 2) :=
sorry

end option_C_is_always_odd_l227_227391


namespace smallest_three_digit_multiple_of_6_5_8_9_eq_360_l227_227388

theorem smallest_three_digit_multiple_of_6_5_8_9_eq_360 :
  ∃ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧ n = 360 := 
by
  sorry

end smallest_three_digit_multiple_of_6_5_8_9_eq_360_l227_227388


namespace find_f_1000_l227_227023

theorem find_f_1000 (f : ℕ → ℕ) 
    (h1 : ∀ n : ℕ, 0 < n → f (f n) = 2 * n) 
    (h2 : ∀ n : ℕ, 0 < n → f (3 * n + 1) = 3 * n + 2) : 
    f 1000 = 1008 :=
by
  sorry

end find_f_1000_l227_227023


namespace fruits_total_l227_227793

def remaining_fruits (frank_apples susan_blueberries henry_apples karen_grapes : ℤ) : ℤ :=
  let frank_remaining := 36 - (36 / 3)
  let susan_remaining := 120 - (120 / 2)
  let henry_collected := 2 * 120
  let henry_after_eating := henry_collected - (henry_collected / 4)
  let henry_remaining := henry_after_eating - (henry_after_eating / 10)
  let karen_collected := henry_collected / 2
  let karen_after_spoilage := karen_collected - (15 * karen_collected / 100)
  let karen_after_giving_away := karen_after_spoilage - (karen_after_spoilage / 3)
  let karen_remaining := karen_after_giving_away - (Int.sqrt karen_after_giving_away)
  frank_remaining + susan_remaining + henry_remaining + karen_remaining

theorem fruits_total : remaining_fruits 36 120 240 120 = 254 :=
by sorry

end fruits_total_l227_227793


namespace gcd_20244_46656_l227_227660

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end gcd_20244_46656_l227_227660


namespace divisor_unique_l227_227491

theorem divisor_unique {b : ℕ} (h1 : 826 % b = 7) (h2 : 4373 % b = 8) : b = 9 :=
sorry

end divisor_unique_l227_227491


namespace quadratic_inequality_solution_range_l227_227158

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x, 1 < x ∧ x < 4 ∧ x^2 - 4 * x - 2 - a > 0) → a < -2 :=
sorry

end quadratic_inequality_solution_range_l227_227158


namespace initial_plank_count_l227_227432

def Bedroom := 8
def LivingRoom := 20
def Kitchen := 11
def DiningRoom := 13
def Hallway := 4
def GuestBedroom := Bedroom - 2
def Study := GuestBedroom + 3
def BedroomReplacements := 3
def LivingRoomReplacements := 2
def StudyReplacements := 1
def LeftoverPlanks := 7

def TotalPlanksUsed := 
  (Bedroom + BedroomReplacements) +
  (LivingRoom + LivingRoomReplacements) +
  (Kitchen) +
  (DiningRoom) +
  (GuestBedroom + BedroomReplacements) +
  (Hallway * 2) +
  (Study + StudyReplacements)

theorem initial_plank_count : 
  TotalPlanksUsed + LeftoverPlanks = 91 := 
by
  sorry

end initial_plank_count_l227_227432


namespace prairie_total_area_l227_227239

theorem prairie_total_area (acres_dust_storm : ℕ) (acres_untouched : ℕ) (h₁ : acres_dust_storm = 64535) (h₂ : acres_untouched = 522) : acres_dust_storm + acres_untouched = 65057 :=
by
  sorry

end prairie_total_area_l227_227239


namespace decrease_in_sales_percentage_l227_227995

theorem decrease_in_sales_percentage (P Q : Real) :
  let P' := 1.40 * P
  let R := P * Q
  let R' := 1.12 * R
  ∃ (D : Real), Q' = Q * (1 - D / 100) ∧ R' = P' * Q' → D = 20 :=
by
  sorry

end decrease_in_sales_percentage_l227_227995


namespace minimum_balls_l227_227101

/-- Given that tennis balls are stored in big boxes containing 25 balls each 
    and small boxes containing 20 balls each, and the least number of balls 
    that can be left unboxed is 5, prove that the least number of 
    freshly manufactured balls is 105.
-/
theorem minimum_balls (B S : ℕ) : 
  ∃ (n : ℕ), 25 * B + 20 * S = n ∧ n % 25 = 5 ∧ n % 20 = 5 ∧ n = 105 := 
sorry

end minimum_balls_l227_227101


namespace horse_catch_up_l227_227838

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end horse_catch_up_l227_227838


namespace nancy_water_intake_l227_227005

theorem nancy_water_intake (water_intake body_weight : ℝ) (h1 : water_intake = 54) (h2 : body_weight = 90) : 
  (water_intake / body_weight) * 100 = 60 :=
by
  -- using the conditions h1 and h2
  rw [h1, h2]
  -- skipping the proof
  sorry

end nancy_water_intake_l227_227005


namespace time_2556_hours_from_now_main_l227_227532

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

end time_2556_hours_from_now_main_l227_227532


namespace point_A_coordinates_l227_227358

variable (a x y : ℝ)

def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

theorem point_A_coordinates (h1 : ∃ t : ℝ, ∀ x : ℝ, f a x = t * x + t) (h2 : x = 0) : (0, 2) = (0, f a 0) :=
by
  sorry

end point_A_coordinates_l227_227358


namespace bottles_produced_l227_227614

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

end bottles_produced_l227_227614


namespace peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l227_227859

-- Define the conditions
variable (a b c : ℕ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)

-- Part 1
theorem peter_can_transfer_all_money_into_two_accounts :
  ∃ x y, (x + y = a + b + c ∧ y = 0) ∨
          (∃ z, (a + b + c = x + y + z ∧ y = 0 ∧ z = 0)) :=
  sorry

-- Part 2
theorem peter_cannot_always_transfer_all_money_into_one_account :
  ((a + b + c) % 2 = 1 → ¬ ∃ x, x = a + b + c) :=
  sorry

end peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l227_227859


namespace maximum_cows_l227_227911

theorem maximum_cows (s c : ℕ) (h1 : 30 * s + 33 * c = 1300) (h2 : c > 2 * s) : c ≤ 30 :=
by
  -- Proof would go here
  sorry

end maximum_cows_l227_227911


namespace eval_expr_at_3_l227_227655

theorem eval_expr_at_3 : (3^2 - 5 * 3 + 6) / (3 - 2) = 0 := by
  sorry

end eval_expr_at_3_l227_227655


namespace equilateral_triangle_in_ellipse_l227_227641

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

end equilateral_triangle_in_ellipse_l227_227641


namespace num_five_digit_numbers_is_correct_l227_227300

-- Define the set of digits and their repetition as given in the conditions
def digits : Multiset ℕ := {1, 3, 3, 5, 8}

-- Calculate the permutation with repetitions
noncomputable def num_five_digit_numbers : ℕ := (digits.card.factorial) / 
  (Multiset.count 1 digits).factorial / 
  (Multiset.count 3 digits).factorial / 
  (Multiset.count 5 digits).factorial / 
  (Multiset.count 8 digits).factorial

-- Theorem stating the final result
theorem num_five_digit_numbers_is_correct : num_five_digit_numbers = 60 :=
by
  -- Proof is omitted
  sorry

end num_five_digit_numbers_is_correct_l227_227300


namespace savings_per_month_l227_227910

-- Define the monthly earnings, total needed for car, and total earnings
def monthly_earnings : ℤ := 4000
def total_needed_for_car : ℤ := 45000
def total_earnings : ℤ := 360000

-- Define the number of months it takes to save the required amount using total earnings and monthly earnings
def number_of_months : ℤ := total_earnings / monthly_earnings

-- Define the monthly savings based on the total needed and number of months
def monthly_savings : ℤ := total_needed_for_car / number_of_months

-- Prove that the monthly savings is £500
theorem savings_per_month : monthly_savings = 500 := by
  -- Placeholder for the proof
  sorry

end savings_per_month_l227_227910


namespace m_add_n_equals_19_l227_227460

theorem m_add_n_equals_19 (n m : ℕ) (A_n_m : ℕ) (C_n_m : ℕ) (h1 : A_n_m = 272) (h2 : C_n_m = 136) :
  m + n = 19 :=
by
  sorry

end m_add_n_equals_19_l227_227460


namespace calculate_value_l227_227000

def f (x : ℕ) : ℕ := 2 * x - 3
def g (x : ℕ) : ℕ := x^2 + 1

theorem calculate_value : f (1 + g 3) = 19 := by
  sorry

end calculate_value_l227_227000


namespace general_formula_for_a_n_l227_227027

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Defining a_n as a function of n assuming it's an arithmetic sequence.
noncomputable def a (x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 then 2 * n - 4 else if x = 3 then 4 - 2 * n else 0

theorem general_formula_for_a_n (x : ℝ) (n : ℕ) (h1 : a x 1 = f (x + 1))
  (h2 : a x 2 = 0) (h3 : a x 3 = f (x - 1)) :
  (x = 1 → a x n = 2 * n - 4) ∧ (x = 3 → a x n = 4 - 2 * n) :=
by sorry

end general_formula_for_a_n_l227_227027


namespace sum_first_11_terms_of_arithmetic_sequence_l227_227397

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 an : ℤ) : ℤ :=
  n * (a1 + an) / 2

theorem sum_first_11_terms_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : S n = sum_arithmetic_sequence n (a 1) (a n))
  (h2 : a 3 + a 6 + a 9 = 60) : S 11 = 220 :=
sorry

end sum_first_11_terms_of_arithmetic_sequence_l227_227397


namespace simplify_eval_expression_l227_227013

theorem simplify_eval_expression (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  3 * (2 * x^2 + x * y + 1 / 3) - (3 * x^2 + 4 * x * y - y^2) = 11 :=
by
  rw [hx, hy]
  sorry

end simplify_eval_expression_l227_227013


namespace tomato_price_per_kilo_l227_227191

theorem tomato_price_per_kilo 
  (initial_money: ℝ) (money_left: ℝ)
  (potato_price_per_kilo: ℝ) (potato_kilos: ℝ)
  (cucumber_price_per_kilo: ℝ) (cucumber_kilos: ℝ)
  (banana_price_per_kilo: ℝ) (banana_kilos: ℝ)
  (tomato_kilos: ℝ)
  (spent_on_potatoes: initial_money - money_left = potato_price_per_kilo * potato_kilos)
  (spent_on_cucumbers: initial_money - money_left = cucumber_price_per_kilo * cucumber_kilos)
  (spent_on_bananas: initial_money - money_left = banana_price_per_kilo * banana_kilos)
  (total_spent: initial_money - money_left = 74)
  : (74 - (potato_price_per_kilo * potato_kilos + cucumber_price_per_kilo * cucumber_kilos + banana_price_per_kilo * banana_kilos)) / tomato_kilos = 3 := 
sorry

end tomato_price_per_kilo_l227_227191


namespace calories_consumed_l227_227327

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l227_227327


namespace range_of_a_l227_227398

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by
  sorry

end range_of_a_l227_227398


namespace problem1_problem2_l227_227958

section Problem
variables (a : ℝ) (x : ℝ) (x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem problem1 (ha : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem problem2 (ha : a ≥ 0) (h1 : x1 ∈ Icc (-1 : ℝ) (Real.inf)) (h2 : x2 ∈ Icc (-1 : ℝ) (Real.inf)) (h : f a x1 = g a x1 - g a x2) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end Problem

end problem1_problem2_l227_227958


namespace mark_weekly_reading_l227_227993

-- Using the identified conditions
def daily_reading_hours : ℕ := 2
def additional_weekly_hours : ℕ := 4

-- Prove the total number of hours Mark wants to read per week is 18 hours
theorem mark_weekly_reading : (daily_reading_hours * 7 + additional_weekly_hours) = 18 := by
  -- Placeholder for proof
  sorry

end mark_weekly_reading_l227_227993


namespace concurrency_of_perpendiculars_l227_227508

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

end concurrency_of_perpendiculars_l227_227508


namespace probability_of_color_difference_l227_227931

noncomputable def probability_of_different_colors (n m : ℕ) : ℚ :=
  (Nat.choose n m : ℚ) * (1/2)^n

theorem probability_of_color_difference :
  probability_of_different_colors 8 4 = 35/128 :=
by
  sorry

end probability_of_color_difference_l227_227931


namespace domain_of_g_l227_227522

def f (x : ℝ) : Prop := x ∈ Set.Icc (-12.0) 6.0

def g (x : ℝ) : Prop := f (3 * x)

theorem domain_of_g : Set.Icc (-4.0) 2.0 = {x : ℝ | g x} := 
by 
    sorry

end domain_of_g_l227_227522


namespace sampling_is_simple_random_l227_227913

-- Definitions based on conditions
def total_students := 200
def students_sampled := 20
def sampling_method := "Simple Random Sampling"

-- The problem: given the random sampling of 20 students from 200, prove that the method is simple random sampling.
theorem sampling_is_simple_random :
  (total_students = 200 ∧ students_sampled = 20) → sampling_method = "Simple Random Sampling" := 
by
  sorry

end sampling_is_simple_random_l227_227913


namespace probability_S7_eq_3_l227_227315

theorem probability_S7_eq_3 :
  let a_n (n : ℕ) : ℤ := if n % 3 = 0 then -1 else 1 
  let S_n (n : ℕ) : ℤ := ∑ i in finset.range n, a_n i
  let probability (x : ℕ) (y : ℕ) := nat.choose 7 5 * (1/3)^5 * (2/3)^2 
  (S_n 7 = 3) → probability 5 2 = C_7^5 \frac{1}{3}^5 \frac{2}{3}^2 :=
by
  sorry

end probability_S7_eq_3_l227_227315


namespace sarah_took_correct_amount_l227_227117

-- Definition of the conditions
def total_cookies : Nat := 150
def neighbors_count : Nat := 15
def correct_amount_per_neighbor : Nat := 10
def remaining_cookies : Nat := 8
def first_neighbors_count : Nat := 14
def last_neighbor : String := "Sarah"

-- Calculations based on conditions
def total_cookies_taken : Nat := total_cookies - remaining_cookies
def correct_cookies_taken : Nat := first_neighbors_count * correct_amount_per_neighbor
def extra_cookies_taken : Nat := total_cookies_taken - correct_cookies_taken
def sarah_cookies : Nat := correct_amount_per_neighbor + extra_cookies_taken

-- Proof statement: Sarah took 12 cookies
theorem sarah_took_correct_amount : sarah_cookies = 12 := by
  sorry

end sarah_took_correct_amount_l227_227117


namespace harry_worked_34_hours_l227_227444

noncomputable def Harry_hours_worked (x : ℝ) : ℝ := 34

theorem harry_worked_34_hours (x : ℝ)
  (H : ℝ) (James_hours : ℝ) (Harry_pay James_pay: ℝ) 
  (h1 : Harry_pay = 18 * x + 1.5 * x * (H - 18)) 
  (h2 : James_pay = 40 * x + 2 * x * (James_hours - 40)) 
  (h3 : James_hours = 41) 
  (h4 : Harry_pay = James_pay) : 
  H = Harry_hours_worked x :=
by
  sorry

end harry_worked_34_hours_l227_227444


namespace linda_coats_l227_227334

variable (wall_area : ℝ) (cover_per_gallon : ℝ) (gallons_bought : ℝ)

theorem linda_coats (h1 : wall_area = 600)
                    (h2 : cover_per_gallon = 400)
                    (h3 : gallons_bought = 3) :
  (gallons_bought / (wall_area / cover_per_gallon)) = 2 :=
by
  sorry

end linda_coats_l227_227334


namespace int_values_satisfying_inequality_l227_227066

theorem int_values_satisfying_inequality : 
  ∃ (N : ℕ), N = 15 ∧ ∀ (x : ℕ), 9 < x ∧ x < 25 → x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} →
  set.size {x | 9 < x ∧ x < 25 ∧ x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}} = N :=
by
  sorry

end int_values_satisfying_inequality_l227_227066


namespace value_of_w_l227_227704

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

end value_of_w_l227_227704


namespace determine_a_l227_227137

theorem determine_a (x : ℝ) (n : ℕ) (h : x > 0) (h_ineq : x + a / x^n ≥ n + 1) : a = n^n := by
  sorry

end determine_a_l227_227137


namespace count_integers_between_bounds_l227_227062

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end count_integers_between_bounds_l227_227062


namespace find_number_l227_227899

def exceeding_condition (x : ℝ) : Prop :=
  x = 0.16 * x + 84

theorem find_number : ∃ x : ℝ, exceeding_condition x ∧ x = 100 :=
by
  -- Proof goes here, currently omitted.
  sorry

end find_number_l227_227899


namespace sphere_surface_area_of_given_volume_l227_227877

-- Definition of the problem conditions
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem to be proved
theorem sphere_surface_area_of_given_volume :
  (∃ (r : ℝ), volume_of_sphere r = 72 * π ∧ surface_area_of_sphere r = 36 * π * 2 ^ (2 / 3)) :=
sorry

end sphere_surface_area_of_given_volume_l227_227877


namespace sphere_surface_area_of_given_volume_l227_227878

-- Definition of the problem conditions
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem to be proved
theorem sphere_surface_area_of_given_volume :
  (∃ (r : ℝ), volume_of_sphere r = 72 * π ∧ surface_area_of_sphere r = 36 * π * 2 ^ (2 / 3)) :=
sorry

end sphere_surface_area_of_given_volume_l227_227878


namespace least_number_subtracted_divisible_l227_227782

theorem least_number_subtracted_divisible (n : ℕ) (d : ℕ) (h : n = 1234567) (k : d = 37) :
  n % d = 13 :=
by 
  rw [h, k]
  sorry

end least_number_subtracted_divisible_l227_227782


namespace fuel_a_added_l227_227612

theorem fuel_a_added (capacity : ℝ) (ethanolA : ℝ) (ethanolB : ℝ) (total_ethanol : ℝ) (x : ℝ) : 
  capacity = 200 ∧ ethanolA = 0.12 ∧ ethanolB = 0.16 ∧ total_ethanol = 28 →
  0.12 * x + 0.16 * (200 - x) = 28 → x = 100 :=
sorry

end fuel_a_added_l227_227612


namespace count_even_numbers_is_320_l227_227479

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end count_even_numbers_is_320_l227_227479


namespace set_intersection_nonempty_l227_227476

theorem set_intersection_nonempty {a : ℕ} (h : ({0, a} ∩ {1, 2} : Set ℕ) ≠ ∅) :
  a = 1 ∨ a = 2 := by
  sorry

end set_intersection_nonempty_l227_227476


namespace reversible_triangle_inequality_l227_227990

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def reversible_triangle (a b c : ℝ) : Prop :=
  (is_triangle a b c) ∧ 
  (is_triangle (1 / a) (1 / b) (1 / c)) ∧
  (a ≤ b) ∧ (b ≤ c)

theorem reversible_triangle_inequality {a b c : ℝ} (h : reversible_triangle a b c) :
  a > (3 - Real.sqrt 5) / 2 * c :=
sorry

end reversible_triangle_inequality_l227_227990


namespace complement_intersection_l227_227901

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

-- Define the complements of A and B with respect to U
def complement_U (s : Set ℕ) := { x ∈ U | x ∉ s }

-- The theorem to prove the intersection of the complements
theorem complement_intersection :
  (complement_U A) ∩ (complement_U B) = {7, 9} :=
sorry

end complement_intersection_l227_227901


namespace x_plus_y_equals_two_l227_227989

variable (x y : ℝ)

def condition1 : Prop := (x - 1) ^ 2017 + 2013 * (x - 1) = -1
def condition2 : Prop := (y - 1) ^ 2017 + 2013 * (y - 1) = 1

theorem x_plus_y_equals_two (h1 : condition1 x) (h2 : condition2 y) : x + y = 2 :=
  sorry

end x_plus_y_equals_two_l227_227989


namespace infinite_solutions_b_l227_227452

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end infinite_solutions_b_l227_227452


namespace find_a_extreme_value_at_2_l227_227468

noncomputable def f (x : ℝ) (a : ℝ) := (2 / 3) * x^3 + a * x^2

theorem find_a_extreme_value_at_2 (a : ℝ) :
  (∀ x : ℝ, x ≠ 2 -> 0 = 2 * x^2 + 2 * a * x) ->
  (2 * 2^2 + 2 * a * 2 = 0) ->
  a = -2 :=
by {
  sorry
}

end find_a_extreme_value_at_2_l227_227468


namespace jenny_reading_time_l227_227708

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

end jenny_reading_time_l227_227708


namespace solution_interval_l227_227775

theorem solution_interval (x : ℝ) : (x^2 / (x - 5)^2 > 0) ↔ (x ∈ Set.Iio 0 ∪ Set.Ioi 0 ∩ Set.Iio 5 ∪ Set.Ioi 5) :=
by
  sorry

end solution_interval_l227_227775


namespace sticker_distribution_probability_l227_227246

theorem sticker_distribution_probability :
  let p := 32
  let q := 50050
  p + q = 50082 :=
sorry

end sticker_distribution_probability_l227_227246


namespace girls_picked_more_l227_227320

variable (N I A V : ℕ)

theorem girls_picked_more (h1 : N > A) (h2 : N > V) (h3 : N > I)
                         (h4 : I ≥ A) (h5 : I ≥ V) (h6 : A > V) :
  N + I > A + V := by
  sorry

end girls_picked_more_l227_227320


namespace form_x2_sub_2y2_l227_227933

theorem form_x2_sub_2y2 (x y : ℤ) (hx : x % 2 = 1) : (x^2 - 2*y^2) % 8 = 1 ∨ (x^2 - 2*y^2) % 8 = -1 := 
sorry

end form_x2_sub_2y2_l227_227933


namespace minimum_distance_l227_227942

def curve1 (x y : ℝ) : Prop := y^2 - 9 + 2*y*x - 12*x - 3*x^2 = 0
def curve2 (x y : ℝ) : Prop := y^2 + 3 - 4*x - 2*y + x^2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem minimum_distance 
  (A B : ℝ × ℝ) 
  (hA : curve1 A.1 A.2) 
  (hB : curve2 B.1 B.2) : 
  ∃ d, d = 2 * Real.sqrt 2 ∧ (∀ P Q : ℝ × ℝ, curve1 P.1 P.2 → curve2 Q.1 Q.2 → distance P.1 P.2 Q.1 Q.2 ≥ d) :=
sorry

end minimum_distance_l227_227942


namespace oranges_distributed_l227_227415

theorem oranges_distributed :
  ∀ (total_students : ℕ) (initial_oranges : ℕ) (bad_oranges : ℕ),
  total_students = 12 →
  initial_oranges = 108 →
  bad_oranges = 36 →
  let good_oranges := initial_oranges - bad_oranges in
  (initial_oranges / total_students - good_oranges / total_students) = 3 :=
by
  intros total_students initial_oranges bad_oranges ts_eq io_eq bo_eq
  let good_oranges := initial_oranges - bad_oranges
  sorry

end oranges_distributed_l227_227415


namespace total_amount_division_l227_227916

variables (w x y z : ℝ)

theorem total_amount_division (h_w : w = 2)
                              (h_x : x = 0.75)
                              (h_y : y = 1.25)
                              (h_z : z = 0.85)
                              (h_share_y : y * Rs48_50 = Rs48_50) :
                              total_amount = 4.85 * 38.80 := sorry

end total_amount_division_l227_227916


namespace Ronald_eggs_initially_l227_227862

def total_eggs_shared (friends eggs_per_friend : Nat) : Nat :=
  friends * eggs_per_friend

theorem Ronald_eggs_initially (eggs : Nat) (candies : Nat) (friends : Nat) (eggs_per_friend : Nat)
  (h1 : friends = 8) (h2 : eggs_per_friend = 2) (h_share : total_eggs_shared friends eggs_per_friend = 16) :
  eggs = 16 := by
  sorry

end Ronald_eggs_initially_l227_227862


namespace log_sum_l227_227670

variable (m a b : ℝ)
variable (m_pos : 0 < m)
variable (m_ne_one : m ≠ 1)
variable (h1 : m^2 = a)
variable (h2 : m^3 = b)

theorem log_sum (m_pos : 0 < m) (m_ne_one : m ≠ 1) (h1 : m^2 = a) (h2 : m^3 = b) :
  2 * Real.log (a) / Real.log (m) + Real.log (b) / Real.log (m) = 7 := 
sorry

end log_sum_l227_227670


namespace inequality_holds_for_all_x_iff_a_in_interval_l227_227736

theorem inequality_holds_for_all_x_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by sorry

end inequality_holds_for_all_x_iff_a_in_interval_l227_227736


namespace even_perfect_square_factors_l227_227692

theorem even_perfect_square_factors :
  let factors := 2^6 * 5^4 * 7^3
  ∃ (count : ℕ), count = (3 * 3 * 2) ∧
  ∀ (a b c : ℕ), (0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ c ∧ c ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 
  a % 2 = 0 ∧ 2 ≤ a ∧ c % 2 = 0 ∧ b % 2 = 0) → 
  a * b * c < count :=
by
  sorry

end even_perfect_square_factors_l227_227692


namespace find_sachin_age_l227_227093

variables (S R : ℕ)

def sachin_young_than_rahul_by_4_years (S R : ℕ) : Prop := R = S + 4
def ratio_of_ages (S R : ℕ) : Prop := 7 * R = 9 * S

theorem find_sachin_age (S R : ℕ) (h1 : sachin_young_than_rahul_by_4_years S R) (h2 : ratio_of_ages S R) : S = 14 := 
by sorry

end find_sachin_age_l227_227093


namespace juanita_sunscreen_cost_l227_227844

theorem juanita_sunscreen_cost:
  let bottles_per_month := 1
  let months_in_year := 12
  let cost_per_bottle := 30.0
  let discount_rate := 0.30
  let total_bottles := bottles_per_month * months_in_year
  let total_cost_before_discount := total_bottles * cost_per_bottle
  let discount_amount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  total_cost_after_discount = 252.00 := 
by
  sorry

end juanita_sunscreen_cost_l227_227844


namespace fraction_addition_correct_l227_227917

theorem fraction_addition_correct : (3 / 5 : ℚ) + (2 / 5) = 1 := 
by
  sorry

end fraction_addition_correct_l227_227917


namespace solve_quadratic_eq_l227_227197

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 7 * x + 6 = 0 ↔ x = 1 ∨ x = 6 :=
by
  sorry

end solve_quadratic_eq_l227_227197


namespace f_at_count_l227_227666

def f (a b c : ℕ) : ℕ := (a * b * c) / (Nat.gcd (Nat.gcd a b) c * Nat.lcm (Nat.lcm a b) c)

def is_f_at (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≤ 60 ∧ y ≤ 60 ∧ z ≤ 60 ∧ f x y z = n

theorem f_at_count : ∃ (n : ℕ), n = 70 ∧ ∀ k, is_f_at k → k ≤ 70 := 
sorry

end f_at_count_l227_227666


namespace geometric_series_evaluation_l227_227186

theorem geometric_series_evaluation (c d : ℝ) (h : (∑' n : ℕ, c / d^(n + 1)) = 3) :
  (∑' n : ℕ, c / (c + 2 * d)^(n + 1)) = (3 * d - 3) / (5 * d - 4) :=
sorry

end geometric_series_evaluation_l227_227186


namespace min_holiday_days_l227_227500

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

end min_holiday_days_l227_227500


namespace sphere_surface_area_l227_227875

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l227_227875


namespace four_real_solutions_l227_227661

-- Definitions used in the problem
def P (x : ℝ) : Prop := (6 * x) / (x^2 + 2 * x + 5) + (4 * x) / (x^2 - 4 * x + 5) = -2 / 3

-- Statement of the problem
theorem four_real_solutions : ∃ (x1 x2 x3 x4 : ℝ), P x1 ∧ P x2 ∧ P x3 ∧ P x4 ∧ 
  ∀ x, P x → (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :=
sorry

end four_real_solutions_l227_227661


namespace fraction_exponent_multiplication_l227_227545

theorem fraction_exponent_multiplication :
  ( (8/9 : ℚ)^2 * (1/3 : ℚ)^2 = (64/729 : ℚ) ) :=
by
  -- here we would write out the detailed proof
  sorry

end fraction_exponent_multiplication_l227_227545


namespace smallest_possible_sum_l227_227282

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l227_227282


namespace tangent_segrment_sum_eq_l227_227772

/-- Given a cyclic quadrilateral ABCD inscribed in a circle Γ,
let E be the intersection of AD and BC, and F be the intersection of AB and DC.
Let segments EG and FH be tangents to the circle Γ. -/
def cyclic_quadrilateral (A B C D E F G H : Point) : Prop :=
  cyclic (A, B, C, D) ∧
  collinear [A, D, E] ∧ 
  collinear [B, C, E] ∧
  collinear [A, B, F] ∧
  collinear [D, C, F] ∧
  tangent_to (E, G, Γ) ∧
  tangent_to (F, H, Γ)

/-- Prove that for the given cyclic quadrilateral and tangents, the equality EG^2 + FH^2 = EF^2 holds. -/
theorem tangent_segrment_sum_eq (A B C D E F G H : Point) (Γ : Circle) 
  (hcyclic : cyclic_quadrilateral A B C D E F G H)
  (tangent_EG : tangent_to (E, G, Γ))
  (tangent_FH : tangent_to (F, H, Γ)) :
  dist E G ^ 2 + dist F H ^ 2 = dist E F ^ 2 :=
sorry

end tangent_segrment_sum_eq_l227_227772


namespace john_less_than_anna_l227_227183

theorem john_less_than_anna (J A L T : ℕ) (h1 : A = 50) (h2: L = 3) (h3: T = 82) (h4: T + L = A + J) : A - J = 15 :=
by
  sorry

end john_less_than_anna_l227_227183


namespace solve_real_number_pairs_l227_227658

theorem solve_real_number_pairs (x y : ℝ) :
  (x^2 + y^2 - 48 * x - 29 * y + 714 = 0 ∧ 2 * x * y - 29 * x - 48 * y + 756 = 0) ↔
  (x = 31.5 ∧ y = 10.5) ∨ (x = 20 ∧ y = 22) ∨ (x = 28 ∧ y = 7) ∨ (x = 16.5 ∧ y = 18.5) :=
by
  sorry

end solve_real_number_pairs_l227_227658


namespace find_m_l227_227953

theorem find_m (m : ℝ) :
  (∃ m : ℝ, ∀ x y : ℝ, x + y - m = 0 ∧ x + (3 - 2 * m) * y = 0 → 
     (m = 1)) := 
sorry

end find_m_l227_227953


namespace no_equilateral_integer_coords_l227_227108

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end no_equilateral_integer_coords_l227_227108


namespace expression_in_terms_of_p_q_l227_227986

variables {α β γ δ p q : ℝ}

-- Let α and β be the roots of x^2 - 2px + 1 = 0
axiom root_α_β : ∀ x, (x - α) * (x - β) = x^2 - 2 * p * x + 1

-- Let γ and δ be the roots of x^2 + qx + 2 = 0
axiom root_γ_δ : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2

-- Expression to be proved
theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2 * (p - q) ^ 2 :=
sorry

end expression_in_terms_of_p_q_l227_227986


namespace jim_saves_money_by_buying_gallon_l227_227709

theorem jim_saves_money_by_buying_gallon :
  let gallon_price := 8
  let bottle_price := 3
  let ounces_per_gallon := 128
  let ounces_per_bottle := 16
  (ounces_per_gallon / ounces_per_bottle) * bottle_price - gallon_price = 16 :=
by
  sorry

end jim_saves_money_by_buying_gallon_l227_227709


namespace dan_initial_money_l227_227258

theorem dan_initial_money (cost_candy : ℕ) (cost_chocolate : ℕ) (total_spent: ℕ) (hc : cost_candy = 7) (hch : cost_chocolate = 6) (hs : total_spent = 13) 
  (h : total_spent = cost_candy + cost_chocolate) : total_spent = 13 := by
  sorry

end dan_initial_money_l227_227258


namespace gcd_of_765432_and_654321_l227_227585

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227585


namespace actual_average_speed_l227_227238

theorem actual_average_speed 
  (v t : ℝ)
  (h : v * t = (v + 21) * (2/3) * t) : 
  v = 42 :=
by
  sorry

end actual_average_speed_l227_227238


namespace ratio_of_final_to_initial_l227_227976

theorem ratio_of_final_to_initial (P : ℝ) (R : ℝ) (T : ℝ) (hR : R = 0.02) (hT : T = 50) :
  let SI := P * R * T
  let A := P + SI
  A / P = 2 :=
by
  sorry

end ratio_of_final_to_initial_l227_227976


namespace Johnson_Smith_tied_end_May_l227_227020

def home_runs_Johnson : List ℕ := [2, 12, 15, 8, 14, 11, 9, 16]
def home_runs_Smith : List ℕ := [5, 9, 10, 12, 15, 12, 10, 17]

def total_without_June (runs: List ℕ) : Nat := List.sum (runs.take 5 ++ runs.drop 5)
def estimated_June (total: Nat) : Nat := total / 8

theorem Johnson_Smith_tied_end_May :
  let total_Johnson := total_without_June home_runs_Johnson;
  let total_Smith := total_without_June home_runs_Smith;
  let estimated_June_Johnson := estimated_June total_Johnson;
  let estimated_June_Smith := estimated_June total_Smith;
  let total_with_June_Johnson := total_Johnson + estimated_June_Johnson;
  let total_with_June_Smith := total_Smith + estimated_June_Smith;
  (List.sum (home_runs_Johnson.take 5) = List.sum (home_runs_Smith.take 5)) :=
by
  sorry

end Johnson_Smith_tied_end_May_l227_227020


namespace sequence_uniquely_determined_l227_227518

theorem sequence_uniquely_determined (a : ℕ → ℝ) (p q : ℝ) (a0 a1 : ℝ)
  (h : ∀ n, a (n + 2) = p * a (n + 1) + q * a n)
  (h0 : a 0 = a0)
  (h1 : a 1 = a1) :
  ∀ n, ∃! a_n, a n = a_n :=
sorry

end sequence_uniquely_determined_l227_227518


namespace max_abs_sum_l227_227484

-- Define the condition for the ellipse equation
def ellipse_condition (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Prove that the largest possible value of |x| + |y| given the condition is 2√3
theorem max_abs_sum (x y : ℝ) (h : ellipse_condition x y) : |x| + |y| ≤ 2 * Real.sqrt 3 :=
sorry

end max_abs_sum_l227_227484


namespace amount_received_from_mom_l227_227447

-- Defining the problem conditions
def receives_from_dad : ℕ := 5
def spends : ℕ := 4
def has_more_from_mom_after_spending (M : ℕ) : Prop := 
  (receives_from_dad + M - spends = receives_from_dad + 2)

-- Lean theorem statement
theorem amount_received_from_mom (M : ℕ) (h : has_more_from_mom_after_spending M) : M = 6 := 
by
  sorry

end amount_received_from_mom_l227_227447


namespace square_division_l227_227181

theorem square_division (n k : ℕ) (m : ℕ) (h : n * k = m * m) :
  ∃ u v d : ℕ, (gcd u v = 1) ∧ (n = d * u * u) ∧ (k = d * v * v) ∧ (m = d * u * v) :=
by sorry

end square_division_l227_227181


namespace f_zero_eq_f_expression_alpha_value_l227_227808

noncomputable def f (ω x : ℝ) : ℝ :=
  3 * Real.sin (ω * x + Real.pi / 6)

theorem f_zero_eq (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  f ω 0 = 3 / 2 :=
by
  sorry

theorem f_expression (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  ∀ x : ℝ, f ω x = f 4 x :=
by
  sorry

theorem alpha_value (f_4 : ℝ → ℝ) (α : ℝ) (hα : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_f4 : ∀ x : ℝ, f_4 x = 3 * Real.sin (4 * x + Real.pi / 6)) (h_fα : f_4 (α / 2) = 3 / 2) :
  α = Real.pi / 3 :=
by
  sorry

end f_zero_eq_f_expression_alpha_value_l227_227808


namespace inverse_at_neg_two_l227_227470

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_at_neg_two :
  g (-2) = -43 :=
by
  -- sorry here to skip the proof, as instructed.
  sorry

end inverse_at_neg_two_l227_227470


namespace number_of_students_l227_227493

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

end number_of_students_l227_227493


namespace determine_x_l227_227261

theorem determine_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  have : ∀ y : ℝ, (5 * y + 1) * (2 * x - 3) = 0 := 
    sorry
  have : (2 * x - 3) = 0 := 
    sorry
  show x = 3 / 2
  sorry

end determine_x_l227_227261


namespace shaded_region_volume_l227_227871

theorem shaded_region_volume :
  let r1 := 4   -- radius of the first cylinder
  let h1 := 2   -- height of the first cylinder
  let r2 := 1   -- radius of the second cylinder
  let h2 := 5   -- height of the second cylinder
  let V1 := π * r1^2 * h1 -- volume of the first cylinder
  let V2 := π * r2^2 * h2 -- volume of the second cylinder
  V1 + V2 = 37 * π :=
by
  sorry

end shaded_region_volume_l227_227871


namespace area_ratio_of_circles_l227_227816

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l227_227816


namespace points_lie_on_line_l227_227126

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t
  let y := (t - 1) / t
  x + y = 2 := by
  sorry

end points_lie_on_line_l227_227126


namespace cookies_left_l227_227503

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end cookies_left_l227_227503


namespace fraction_addition_l227_227483

theorem fraction_addition (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end fraction_addition_l227_227483


namespace polynomial_evaluation_l227_227671

theorem polynomial_evaluation (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2005 = 2006 :=
sorry

end polynomial_evaluation_l227_227671


namespace solve_for_y_l227_227814

theorem solve_for_y (y : ℕ) (h : 9^y = 3^12) : y = 6 :=
by {
  sorry
}

end solve_for_y_l227_227814


namespace determine_quarters_given_l227_227715

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

end determine_quarters_given_l227_227715


namespace Moe_has_least_amount_of_money_l227_227106

variables (Money : Type) [LinearOrder Money]
variables (Bo Coe Flo Jo Moe Zoe : Money)
variables (Bo_lt_Flo : Bo < Flo) (Jo_lt_Flo : Jo < Flo)
variables (Moe_lt_Bo : Moe < Bo) (Moe_lt_Coe : Moe < Coe)
variables (Moe_lt_Jo : Moe < Jo) (Jo_lt_Bo : Jo < Bo)
variables (Moe_lt_Zoe : Moe < Zoe) (Zoe_lt_Jo : Zoe < Jo)

theorem Moe_has_least_amount_of_money : ∀ x, x ≠ Moe → Moe < x := by
  sorry

end Moe_has_least_amount_of_money_l227_227106


namespace cone_cannot_have_rectangular_projection_l227_227160

def orthographic_projection (solid : Type) : Type := sorry

theorem cone_cannot_have_rectangular_projection :
  (∀ (solid : Type), orthographic_projection solid = Rectangle → solid ≠ Cone) :=
sorry

end cone_cannot_have_rectangular_projection_l227_227160


namespace complex_roots_real_power_six_count_l227_227380

theorem complex_roots_real_power_six_count :
  let solutions := {z : ℂ | z ^ 24 = 1}
  (real_solutions := {z : ℂ | z ∈ solutions ∧ z ^ 6 ∈ ℝ}) 
  in
  solutions.card = 24 ∧ real_solutions.card = 12 :=
by
  sorry

end complex_roots_real_power_six_count_l227_227380


namespace probability_sin_cos_in_range_l227_227909

noncomputable def probability_sin_cos_interval : ℝ :=
  let interval_length := (Real.pi / 2 + Real.pi / 6)
  let valid_length := (Real.pi / 2 - 0)
  valid_length / interval_length

theorem probability_sin_cos_in_range :
  probability_sin_cos_interval = 3 / 4 :=
sorry

end probability_sin_cos_in_range_l227_227909


namespace solution_set_of_inequality_l227_227874

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - abs x - 2 < 0) ↔ (-2 < x ∧ x < 2) := 
sorry

end solution_set_of_inequality_l227_227874


namespace nate_age_is_14_l227_227652

def nate_current_age (N : ℕ) : Prop :=
  ∃ E : ℕ, E = N / 2 ∧ N - E = 7

theorem nate_age_is_14 : nate_current_age 14 :=
by {
  sorry
}

end nate_age_is_14_l227_227652


namespace rth_term_of_arithmetic_progression_l227_227664

noncomputable def Sn (n : ℕ) : ℕ := 2 * n + 3 * n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) : 
  (Sn r - Sn (r - 1)) = 3 * r^2 + 5 * r - 2 :=
by sorry

end rth_term_of_arithmetic_progression_l227_227664


namespace second_puppy_weight_l227_227887

variables (p1 p2 c1 c2 : ℝ)

-- Conditions from the problem statement
axiom h1 : p1 + p2 + c1 + c2 = 36
axiom h2 : p1 + c2 = 3 * c1
axiom h3 : p1 + c1 = c2
axiom h4 : p2 = 1.5 * p1

-- The question to prove: how much does the second puppy weigh
theorem second_puppy_weight : p2 = 108 / 11 :=
by sorry

end second_puppy_weight_l227_227887


namespace x0_range_l227_227144

noncomputable def f (x : ℝ) := (1 / 2) ^ x - Real.log x

theorem x0_range (x0 : ℝ) (h : f x0 > 1 / 2) : 0 < x0 ∧ x0 < 1 :=
by
  sorry

end x0_range_l227_227144


namespace best_fitting_model_l227_227319

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.98)
  (h2 : R2_2 = 0.80)
  (h3 : R2_3 = 0.50)
  (h4 : R2_4 = 0.25) :
  R2_1 = 0.98 ∧ R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by { sorry }

end best_fitting_model_l227_227319


namespace decrease_percent_revenue_l227_227229

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.10 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 12 := by
  sorry

end decrease_percent_revenue_l227_227229


namespace distinct_even_numbers_between_100_and_999_l227_227477

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end distinct_even_numbers_between_100_and_999_l227_227477


namespace length_of_chord_MN_l227_227360

theorem length_of_chord_MN 
  (m n : ℝ)
  (h1 : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * M.1 + M.2 * M.2 + m * M.1 + n * M.2 - 4 = 0 ∧ N.1 * N.1 + N.2 * N.2 + m * N.1 + n * N.2 - 4 = 0 
    ∧ N.2 = M.1 ∧ N.1 = M.2) 
  (h2 : x + y = 0)
  : length_of_chord = 4 := sorry

end length_of_chord_MN_l227_227360


namespace investment_value_change_l227_227668

theorem investment_value_change (k m : ℝ) : 
  let increaseFactor := 1 + k / 100
  let decreaseFactor := 1 - m / 100 
  let overallFactor := increaseFactor * decreaseFactor 
  let changeFactor := overallFactor - 1
  let percentageChange := changeFactor * 100 
  percentageChange = k - m - (k * m) / 100 := 
by 
  sorry

end investment_value_change_l227_227668


namespace area_shaded_region_l227_227733

theorem area_shaded_region (r R : ℝ) (h1 : 0 < r) (h2 : r < R)
  (h3 : 60 = 2 * sqrt (R^2 - r^2)) :
  π * (R^2 - r^2) = 900 * π :=
by
  sorry

end area_shaded_region_l227_227733


namespace profit_per_meal_A_and_B_l227_227970

theorem profit_per_meal_A_and_B (x y : ℝ) 
  (h1 : x + 2 * y = 35) 
  (h2 : 2 * x + 3 * y = 60) : 
  x = 15 ∧ y = 10 :=
sorry

end profit_per_meal_A_and_B_l227_227970


namespace equation_of_line_l227_227698

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

end equation_of_line_l227_227698


namespace number_of_integers_inequality_l227_227049

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end number_of_integers_inequality_l227_227049


namespace find_f_inv_128_l227_227696

open Function

theorem find_f_inv_128 (f : ℕ → ℕ) 
  (h₀ : f 5 = 2) 
  (h₁ : ∀ x, f (2 * x) = 2 * f x) : 
  f⁻¹' {128} = {320} :=
by
  sorry

end find_f_inv_128_l227_227696


namespace difference_of_fractions_l227_227890

theorem difference_of_fractions (h₁ : 1/10 * 8000 = 800) (h₂ : (1/20) / 100 * 8000 = 4) : 800 - 4 = 796 :=
by
  sorry

end difference_of_fractions_l227_227890


namespace num_integers_satisfying_sqrt_ineq_l227_227039

theorem num_integers_satisfying_sqrt_ineq:
  {x : ℕ} (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) →
  Finset.card (Finset.filter (λ x, 3 < Real.sqrt x ∧ Real.sqrt x < 5) (Finset.range 25)) = 15 :=
by
  sorry

end num_integers_satisfying_sqrt_ineq_l227_227039


namespace bobby_shoes_multiple_l227_227922

theorem bobby_shoes_multiple (B M : ℕ) (hBonny : 13 = 2 * B - 5) (hBobby : 27 = M * B) : 
  M = 3 :=
by 
  sorry

end bobby_shoes_multiple_l227_227922


namespace fractions_order_l227_227256

theorem fractions_order : (23 / 18) < (21 / 16) ∧ (21 / 16) < (25 / 19) :=
by
  sorry

end fractions_order_l227_227256


namespace school_students_l227_227211

theorem school_students (x y : ℕ) (h1 : x + y = 432) (h2 : x - 16 = (y + 16) + 24) : x = 244 ∧ y = 188 := by
  sorry

end school_students_l227_227211


namespace time_to_paint_one_room_l227_227761

theorem time_to_paint_one_room (total_rooms : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) (rooms_left : ℕ) :
  total_rooms = 9 ∧ rooms_painted = 5 ∧ time_remaining = 32 ∧ rooms_left = total_rooms - rooms_painted → time_remaining / rooms_left = 8 :=
by
  intros h
  sorry

end time_to_paint_one_room_l227_227761


namespace count_integer_values_l227_227034

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end count_integer_values_l227_227034


namespace gas_mixture_pressure_l227_227969

theorem gas_mixture_pressure
  (m : ℝ) -- mass of each gas
  (p : ℝ) -- initial pressure
  (T : ℝ) -- initial temperature
  (V : ℝ) -- volume of the container
  (R : ℝ) -- ideal gas constant
  (mu_He : ℝ := 4) -- molar mass of helium
  (mu_N2 : ℝ := 28) -- molar mass of nitrogen
  (is_ideal : True) -- assumption that the gases are ideal
  (temp_doubled : True) -- assumption that absolute temperature is doubled
  (N2_dissociates : True) -- assumption that nitrogen dissociates into atoms
  : (9 / 4) * p = p' :=
by
  sorry

end gas_mixture_pressure_l227_227969


namespace expression_in_terms_of_x_difference_between_x_l227_227697

variable (E x : ℝ)

theorem expression_in_terms_of_x (h1 : E / (2 * x + 15) = 3) : E = 6 * x + 45 :=
by 
  sorry

variable (x1 x2 : ℝ)

theorem difference_between_x (h1 : E / (2 * x1 + 15) = 3) (h2: E / (2 * x2 + 15) = 3) (h3 : x2 - x1 = 12) : True :=
by 
  sorry

end expression_in_terms_of_x_difference_between_x_l227_227697


namespace valid_twenty_letter_words_l227_227017

noncomputable def number_of_valid_words : ℕ := sorry

theorem valid_twenty_letter_words :
  number_of_valid_words = 3 * 2^18 := sorry

end valid_twenty_letter_words_l227_227017


namespace inequality_proof_l227_227192

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) : 
  (1 - a) * (1 - b) ≤ 25/36 :=
by
  sorry

end inequality_proof_l227_227192


namespace divide_plane_into_four_quadrants_l227_227841

-- Definitions based on conditions
def perpendicular_axes (x y : ℝ → ℝ) : Prop :=
  (∀ t : ℝ, x t = t ∨ x t = 0) ∧ (∀ t : ℝ, y t = t ∨ y t = 0) ∧ ∀ t : ℝ, x t ≠ y t

-- The mathematical proof statement
theorem divide_plane_into_four_quadrants (x y : ℝ → ℝ) (hx : perpendicular_axes x y) :
  ∃ quadrants : ℕ, quadrants = 4 :=
by
  sorry

end divide_plane_into_four_quadrants_l227_227841


namespace gcd_765432_654321_l227_227591

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227591


namespace min_value_a_l227_227185

theorem min_value_a (a b c d : ℚ) (h₀ : a > 0)
  (h₁ : ∀ n : ℕ, (a * n^3 + b * n^2 + c * n + d).den = 1) :
  a = 1/6 := by
  -- Proof goes here
  sorry

end min_value_a_l227_227185


namespace distance_between_A_and_B_is_45_kilometers_l227_227720

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

end distance_between_A_and_B_is_45_kilometers_l227_227720


namespace inequality_solution_l227_227378

-- Define the inequality condition
def fraction_inequality (x : ℝ) : Prop :=
  (3 * x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 / 3 ≤ x ∧ x < 2

-- The theorem to prove that the inequality's solution matches the given solution set
theorem inequality_solution (x : ℝ) (h : fraction_inequality x) : solution_set x :=
  sorry

end inequality_solution_l227_227378


namespace probability_of_drawing_3_one_color_and_1_another_l227_227412

open Nat

def combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_of_drawing_3_one_color_and_1_another : 
  let total_balls := 12 + 8
  let choose_4_from_total := combinations total_balls 4
  let choose_3_black_1_white := combinations 12 3 * combinations 8 1
  let choose_1_black_3_white := combinations 12 1 * combinations 8 3
  let favorable_outcomes := choose_3_black_1_white + choose_1_black_3_white
  let numerator := favorable_outcomes
  let denominator := choose_4_from_total
  numerator / denominator = 1 / 3 := 
by 
  sorry

end probability_of_drawing_3_one_color_and_1_another_l227_227412


namespace direct_proportion_function_l227_227699

theorem direct_proportion_function (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = k * x) (h2 : f 3 = 6) : ∀ x, f x = 2 * x := by
  sorry

end direct_proportion_function_l227_227699


namespace general_formula_for_a_n_l227_227028

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Defining a_n as a function of n assuming it's an arithmetic sequence.
noncomputable def a (x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 then 2 * n - 4 else if x = 3 then 4 - 2 * n else 0

theorem general_formula_for_a_n (x : ℝ) (n : ℕ) (h1 : a x 1 = f (x + 1))
  (h2 : a x 2 = 0) (h3 : a x 3 = f (x - 1)) :
  (x = 1 → a x n = 2 * n - 4) ∧ (x = 3 → a x n = 4 - 2 * n) :=
by sorry

end general_formula_for_a_n_l227_227028


namespace cos_identity_l227_227669

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, (Real.sqrt 3) / 2)
  let b := (Real.sin (x - Real.pi / 3), 1)
  a.1 * b.1 + a.2 * b.2

theorem cos_identity (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3))
  (hf : f x0 = 4 / 5) :
  Real.cos (2 * x0 - Real.pi / 12) = -7 * Real.sqrt 2 / 10 :=
sorry

end cos_identity_l227_227669


namespace smallest_integer_k_no_real_roots_l227_227270

def quadratic_no_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c < 0

theorem smallest_integer_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, quadratic_no_real_roots (2 * k - 1) (-8) 6) ∧ (k = 2) :=
by
  sorry

end smallest_integer_k_no_real_roots_l227_227270


namespace find_certain_number_l227_227695

theorem find_certain_number (h1 : 2994 / 14.5 = 171) (h2 : ∃ x : ℝ, x / 1.45 = 17.1) : ∃ x : ℝ, x = 24.795 :=
by
  sorry

end find_certain_number_l227_227695


namespace probability_divisor_of_60_l227_227603

theorem probability_divisor_of_60 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ a b c : ℕ, n = 2 ^ a * 3 ^ b * 5 ^ c ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) → 
  ∃ p : ℚ, p = 1 / 5 :=
by
  sorry

end probability_divisor_of_60_l227_227603


namespace parallel_case_perpendicular_case_l227_227691

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

end parallel_case_perpendicular_case_l227_227691


namespace min_book_corner_cost_l227_227079

theorem min_book_corner_cost :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧
  80 * x + 30 * (30 - x) ≤ 1900 ∧
  50 * x + 60 * (30 - x) ≤ 1620 ∧
  860 * x + 570 * (30 - x) = 22320 := sorry

end min_book_corner_cost_l227_227079


namespace sum_ratio_l227_227985

noncomputable def arithmetic_seq (a₁ d n : ℕ) : ℕ := a₁ + n * d

noncomputable def sum_first_n_terms (a₁ d n : ℕ) : ℕ := 
  n * (2 * a₁ + (n-1) * d) / 2

theorem sum_ratio 
  (a₁ d : ℕ) 
  (h : (a₁ + 4 * d) / (a₁ + 2 * d) = 2) 
  : (sum_first_n_terms a₁ d 9) / (sum_first_n_terms a₁ d 5) = 18 / 5 :=
by
  sorry

end sum_ratio_l227_227985


namespace sum_of_powers_l227_227606

theorem sum_of_powers : (-1: ℤ) ^ 2006 - (-1) ^ 2007 + 1 ^ 2008 + 1 ^ 2009 - 1 ^ 2010 = 3 := by
  sorry

end sum_of_powers_l227_227606


namespace no_equilateral_integer_coords_l227_227107

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end no_equilateral_integer_coords_l227_227107


namespace stadium_surface_area_correct_l227_227738

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

end stadium_surface_area_correct_l227_227738


namespace marbles_count_l227_227833

variable (r b : ℕ)

theorem marbles_count (hr1 : 8 * (r - 1) = r + b - 2) (hr2 : 4 * r = r + b - 3) : r + b = 9 := 
by sorry

end marbles_count_l227_227833


namespace number_decomposition_l227_227623

theorem number_decomposition (n : ℕ) : n = 6058 → (n / 1000 = 6) ∧ ((n % 100) / 10 = 5) ∧ (n % 10 = 8) :=
by
  -- Actual proof will go here
  sorry

end number_decomposition_l227_227623


namespace unique_zero_of_f_inequality_of_x1_x2_l227_227962

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l227_227962


namespace gcd_divisors_remainders_l227_227749

theorem gcd_divisors_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end gcd_divisors_remainders_l227_227749


namespace opposite_of_point_one_l227_227528

theorem opposite_of_point_one : ∃ x : ℝ, 0.1 + x = 0 ∧ x = -0.1 :=
by
  sorry

end opposite_of_point_one_l227_227528


namespace reciprocal_of_2023_l227_227370

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l227_227370


namespace number_of_integers_between_10_and_24_l227_227054

theorem number_of_integers_between_10_and_24 : 
  (set.count (set_of (λ x : ℤ, 9 < x ∧ x < 25))) = 15 := 
sorry

end number_of_integers_between_10_and_24_l227_227054


namespace reciprocal_of_2023_l227_227371

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l227_227371


namespace simplify_fraction_l227_227196

theorem simplify_fraction :
  (144 : ℤ) / (1296 : ℤ) = 1 / 9 := 
by sorry

end simplify_fraction_l227_227196


namespace andy_wrong_questions_l227_227919

variables (a b c d : ℕ)

theorem andy_wrong_questions 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 6) 
  (h3 : c = 7) : 
  a = 20 :=
sorry

end andy_wrong_questions_l227_227919


namespace sum_on_simple_interest_is_1750_l227_227615

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

end sum_on_simple_interest_is_1750_l227_227615


namespace total_cans_collected_l227_227719

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

end total_cans_collected_l227_227719


namespace total_cost_cardshop_l227_227730

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

end total_cost_cardshop_l227_227730


namespace range_of_a_l227_227947

variable (a : ℝ)

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a - 1) ^ x < (2 * a - 1) ^ y
def q (a : ℝ) : Prop := ∀ x : ℝ, 2 * a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (0 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a) :=
by
  sorry

end range_of_a_l227_227947


namespace circle_area_ratio_is_correct_l227_227826

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l227_227826


namespace find_C_D_l227_227657

theorem find_C_D (x C D : ℚ) 
  (h : 7 * x - 5 ≠ 0) -- Added condition to avoid zero denominator
  (hx : x^2 - 8 * x - 48 = (x - 12) * (x + 4))
  (h_eq : 7 * x - 5 = C * (x + 4) + D * (x - 12))
  (h_c : C = 79 / 16)
  (h_d : D = 33 / 16)
: 7 * x - 5 = 79 / 16 * (x + 4) + 33 / 16 * (x - 12) :=
by sorry

end find_C_D_l227_227657


namespace goods_train_speed_l227_227751

theorem goods_train_speed:
  let speed_mans_train := 100   -- in km/h
  let length_goods_train := 280 -- in meters
  let passing_time := 9         -- in seconds
  ∃ speed_goods_train: ℝ, 
  (speed_mans_train + speed_goods_train) * (5 / 18) * passing_time = length_goods_train ↔ speed_goods_train = 12 :=
by
  sorry

end goods_train_speed_l227_227751


namespace oranges_less_per_student_l227_227414

def total_students : ℕ := 12
def total_oranges : ℕ := 108
def bad_oranges : ℕ := 36

theorem oranges_less_per_student :
  (total_oranges / total_students) - ((total_oranges - bad_oranges) / total_students) = 3 :=
by
  sorry

end oranges_less_per_student_l227_227414


namespace emily_pen_selections_is_3150_l227_227263

open Function

noncomputable def emily_pen_selections : ℕ :=
  (Nat.choose 10 4) * (Nat.choose 6 2)

theorem emily_pen_selections_is_3150 : emily_pen_selections = 3150 :=
by
  sorry

end emily_pen_selections_is_3150_l227_227263


namespace constant_term_of_expansion_l227_227019

theorem constant_term_of_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ ∀ r : ℕ, r = 1 → (Nat.choose 5 r * 3^r * x^((5-5*r)/2) = c)) :=
by
  sorry

end constant_term_of_expansion_l227_227019


namespace base_of_square_eq_l227_227622

theorem base_of_square_eq (b : ℕ) (h : b > 6) : 
  (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 → b = 7 :=
by
  sorry

end base_of_square_eq_l227_227622


namespace maximum_area_of_rectangle_with_given_perimeter_l227_227331

theorem maximum_area_of_rectangle_with_given_perimeter {x y : ℕ} (h₁ : 2 * x + 2 * y = 160) : 
  (∃ x y : ℕ, 2 * x + 2 * y = 160 ∧ x * y = 1600) := 
sorry

end maximum_area_of_rectangle_with_given_perimeter_l227_227331


namespace line_equation_l227_227939

theorem line_equation (b : ℝ) :
  (∃ b, (∀ x y, y = (3/4) * x + b) ∧ 
  (1/2) * |b| * |- (4/3) * b| = 6 →
  (3 * x - 4 * y + 12 = 0 ∨ 3 * x - 4 * y - 12 = 0)) := 
sorry

end line_equation_l227_227939


namespace count_integers_satisfying_sqrt_condition_l227_227045

noncomputable def count_integers_in_range (lower upper : ℕ) : ℕ :=
    (upper - lower + 1)

/- Proof statement for the given problem -/
theorem count_integers_satisfying_sqrt_condition :
  let conditions := (∀ x : ℕ, 5 > Real.sqrt x ∧ Real.sqrt x > 3) in
  count_integers_in_range 10 24 = 15 :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l227_227045


namespace gcd_proof_l227_227561

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227561


namespace problem1_problem2_l227_227957

section Problem
variables (a : ℝ) (x : ℝ) (x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem problem1 (ha : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem problem2 (ha : a ≥ 0) (h1 : x1 ∈ Icc (-1 : ℝ) (Real.inf)) (h2 : x2 ∈ Icc (-1 : ℝ) (Real.inf)) (h : f a x1 = g a x1 - g a x2) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end Problem

end problem1_problem2_l227_227957


namespace raft_travel_time_l227_227903

noncomputable def downstream_speed (x y : ℝ) : ℝ := x + y
noncomputable def upstream_speed (x y : ℝ) : ℝ := x - y

theorem raft_travel_time {x y : ℝ} 
  (h1 : 7 * upstream_speed x y = 5 * downstream_speed x y) : (35 : ℝ) = (downstream_speed x y) * 7 / 4 := by sorry

end raft_travel_time_l227_227903


namespace fraction_identity_l227_227133

theorem fraction_identity (a b c : ℕ) (h : (a : ℚ) / (36 - a) + (b : ℚ) / (48 - b) + (c : ℚ) / (72 - c) = 9) : 
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 := 
by 
  sorry

end fraction_identity_l227_227133


namespace water_depth_correct_l227_227918

noncomputable def water_depth (ron_height : ℝ) (dean_shorter_by : ℝ) : ℝ :=
  let dean_height := ron_height - dean_shorter_by
  2.5 * dean_height + 3

theorem water_depth_correct :
  water_depth 14.2 8.3 = 17.75 :=
by
  let ron_height := 14.2
  let dean_shorter_by := 8.3
  let dean_height := ron_height - dean_shorter_by
  let depth := 2.5 * dean_height + 3
  simp [water_depth, dean_height, depth]
  sorry

end water_depth_correct_l227_227918


namespace minimum_value_of_quadratic_l227_227682

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = -p / 2 ∧ (∀ y : ℝ, (y - x) ^ 2 + 2*q ≥ (x ^ 2 + p * x + 2*q)) :=
by
  sorry

end minimum_value_of_quadratic_l227_227682


namespace john_apartment_number_l227_227501

variable (k d m : ℕ)

theorem john_apartment_number (h1 : k = m) (h2 : d + m = 239) (h3 : 10 * (k - 1) + 1 ≤ d) (h4 : d ≤ 10 * k) : d = 217 := 
by 
  sorry

end john_apartment_number_l227_227501


namespace number_is_12_l227_227619

theorem number_is_12 (x : ℝ) (h : 4 * x - 3 = 9 * (x - 7)) : x = 12 :=
by
  sorry

end number_is_12_l227_227619


namespace gcd_765432_654321_l227_227599

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227599


namespace arnaldo_billion_difference_l227_227920

theorem arnaldo_billion_difference :
  (10 ^ 12) - (10 ^ 9) = 999000000000 :=
by
  sorry

end arnaldo_billion_difference_l227_227920


namespace gcd_765432_654321_eq_3_l227_227575

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227575


namespace original_price_of_article_l227_227421

theorem original_price_of_article (SP : ℝ) (profit_rate : ℝ) (P : ℝ) (h1 : SP = 550) (h2 : profit_rate = 0.10) (h3 : SP = P * (1 + profit_rate)) : P = 500 :=
by
  sorry

end original_price_of_article_l227_227421


namespace rabbit_calories_l227_227433

theorem rabbit_calories (C : ℕ) :
  (6 * 300 = 2 * C + 200) → C = 800 :=
by
  intro h
  sorry

end rabbit_calories_l227_227433


namespace find_remainder_l227_227355

theorem find_remainder :
  ∀ (D d q r : ℕ), 
    D = 18972 → 
    d = 526 → 
    q = 36 → 
    D = d * q + r → 
    r = 36 :=
by 
  intros D d q r hD hd hq hEq
  sorry

end find_remainder_l227_227355


namespace solution_set_of_inequality_l227_227213

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1 / 3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l227_227213


namespace solve_for_x2_plus_9y2_l227_227487

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_l227_227487


namespace keith_initial_cards_l227_227506

theorem keith_initial_cards (new_cards : ℕ) (cards_after_incident : ℕ) (total_cards : ℕ) :
  new_cards = 8 →
  cards_after_incident = 46 →
  total_cards = 2 * cards_after_incident →
  (total_cards - new_cards) = 84 :=
by
  intros
  sorry

end keith_initial_cards_l227_227506


namespace gcd_of_expression_l227_227440

theorem gcd_of_expression 
  (a b c d : ℕ) :
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a - b) (c - d)) (a - c)) (b - d)) (a - d)) (b - c) = 12 :=
sorry

end gcd_of_expression_l227_227440


namespace gcd_765432_654321_eq_3_l227_227572

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227572


namespace M_squared_is_odd_l227_227714

theorem M_squared_is_odd (a b : ℤ) (h1 : a = b + 1) (c : ℤ) (h2 : c = a * b) (M : ℤ) (h3 : M^2 = a^2 + b^2 + c^2) : M^2 % 2 = 1 := 
by
  sorry

end M_squared_is_odd_l227_227714


namespace magnitude_of_c_is_correct_l227_227473

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
noncomputable def c : ℝ × ℝ := (a.1 - (dot_product a b) * b.1, a.2 - (dot_product a b) * b.2)

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt ((u.1 ^ 2) + (u.2 ^ 2))

theorem magnitude_of_c_is_correct :
  magnitude c = 8 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_c_is_correct_l227_227473


namespace good_horse_catchup_l227_227839

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end good_horse_catchup_l227_227839


namespace tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l227_227289

open Real

theorem tan_alpha_plus_pi (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  tan (α + π) = -3 / 4 :=
sorry

theorem cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  cos (α - π / 2) * sin (α + 3 * π / 2) = 12 / 25 :=
sorry

end tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l227_227289


namespace total_goals_is_50_l227_227116

def team_a_first_half_goals := 8
def team_b_first_half_goals := team_a_first_half_goals / 2
def team_c_first_half_goals := 2 * team_b_first_half_goals
def team_a_first_half_missed_penalty := 1
def team_c_first_half_missed_penalty := 2

def team_a_second_half_goals := team_c_first_half_goals
def team_b_second_half_goals := team_a_first_half_goals
def team_c_second_half_goals := team_b_second_half_goals + 3
def team_a_second_half_successful_penalty := 1
def team_b_second_half_successful_penalty := 2

def total_team_a_goals := team_a_first_half_goals + team_a_second_half_goals + team_a_second_half_successful_penalty
def total_team_b_goals := team_b_first_half_goals + team_b_second_half_goals + team_b_second_half_successful_penalty
def total_team_c_goals := team_c_first_half_goals + team_c_second_half_goals

def total_goals := total_team_a_goals + total_team_b_goals + total_team_c_goals

theorem total_goals_is_50 : total_goals = 50 := by
  unfold total_goals
  unfold total_team_a_goals total_team_b_goals total_team_c_goals
  unfold team_a_first_half_goals team_b_first_half_goals team_c_first_half_goals
  unfold team_a_second_half_goals team_b_second_half_goals team_c_second_half_goals
  unfold team_a_second_half_successful_penalty team_b_second_half_successful_penalty
  sorry

end total_goals_is_50_l227_227116


namespace limit_value_l227_227681

noncomputable def f (x : ℝ) := 2 * Real.log(3 * x) + 8 * x + 1

theorem limit_value :
  tendsto (λ Δx, (f (1 - 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-20)) :=
sorry

end limit_value_l227_227681


namespace electronics_weight_l227_227365

theorem electronics_weight (B C E : ℝ) (h1 : B / C = 5 / 4) (h2 : B / E = 5 / 2) (h3 : B / (C - 9) = 10 / 4) : E = 9 := 
by 
  sorry

end electronics_weight_l227_227365


namespace find_point_on_parabola_l227_227805

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def positive_y (y : ℝ) : Prop := y > 0
def distance_to_focus (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = (5/2)^2 

theorem find_point_on_parabola (x y : ℝ) :
  parabola x y ∧ positive_y y ∧ distance_to_focus x y → (x = 1 ∧ y = Real.sqrt 6) :=
by
  sorry

end find_point_on_parabola_l227_227805


namespace count_integers_between_bounds_l227_227061

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end count_integers_between_bounds_l227_227061


namespace gcd_765432_654321_l227_227601

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227601


namespace remainder_of_f_x10_mod_f_l227_227001

def f (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

theorem remainder_of_f_x10_mod_f (x : ℤ) : (f (x ^ 10)) % (f x) = 5 :=
by
  sorry

end remainder_of_f_x10_mod_f_l227_227001


namespace complex_z_1000_l227_227140

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end complex_z_1000_l227_227140


namespace boat_length_in_steps_l227_227653

theorem boat_length_in_steps (L E S : ℝ) 
  (h1 : 250 * E = L + 250 * S) 
  (h2 : 50 * E = L - 50 * S) :
  L = 83 * E :=
by sorry

end boat_length_in_steps_l227_227653


namespace smallest_sum_minimum_l227_227283

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l227_227283


namespace count_integers_satisfying_condition_l227_227071

theorem count_integers_satisfying_condition :
  (card {x : ℤ | 9 < x ∧ x < 25} = 15) :=
by
  sorry

end count_integers_satisfying_condition_l227_227071


namespace count_integers_satisfying_condition_l227_227068

theorem count_integers_satisfying_condition :
  (card {x : ℤ | 9 < x ∧ x < 25} = 15) :=
by
  sorry

end count_integers_satisfying_condition_l227_227068


namespace range_of_x_when_m_eq_4_range_of_m_given_conditions_l227_227458

-- Definitions of p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Question 1: Given m = 4 and conditions p ∧ q being true, prove the range of x is 4 < x < 5
theorem range_of_x_when_m_eq_4 (x m : ℝ) (h_m : m = 4) (h : p x ∧ q x m) : 4 < x ∧ x < 5 := 
by
  sorry

-- Question 2: Given conditions ⟪¬q ⟫is a sufficient but not necessary condition for ⟪¬p ⟫and constraints, prove the range of m is 5/3 ≤ m ≤ 2
theorem range_of_m_given_conditions (m : ℝ) (h_sufficient : ∀ (x : ℝ), ¬q x m → ¬p x) (h_constraints : m > 0) : 5 / 3 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_x_when_m_eq_4_range_of_m_given_conditions_l227_227458


namespace verna_sherry_total_weight_l227_227889

theorem verna_sherry_total_weight (haley verna sherry : ℕ)
  (h1 : verna = haley + 17)
  (h2 : verna = sherry / 2)
  (h3 : haley = 103) :
  verna + sherry = 360 :=
by
  sorry

end verna_sherry_total_weight_l227_227889


namespace integer_values_count_l227_227075

theorem integer_values_count (x : ℕ) : (∃ y : ℤ, 10 ≤ y ∧ y ≤ 24) ↔ (∑ y in (finset.interval 10 24), 1) = 15 :=
by
  sorry

end integer_values_count_l227_227075


namespace min_value_when_a_equals_1_range_of_a_for_f_geq_a_l227_227143

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem min_value_when_a_equals_1 : 
  ∃ x, f x 1 = 1 :=
by
  sorry

theorem range_of_a_for_f_geq_a (a : ℝ) :
  (∀ x, x ≥ -1 → f x a ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end min_value_when_a_equals_1_range_of_a_for_f_geq_a_l227_227143


namespace gcd_765432_654321_eq_3_l227_227573

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227573


namespace minimum_a_l227_227157

theorem minimum_a (a : ℝ) : (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (a / x + 4 / y) ≥ 16) → a ≥ 4 :=
by
  intros h
  -- We would provide a detailed mathematical proof here, but we use sorry for now.
  sorry

end minimum_a_l227_227157


namespace restaurant_tip_difference_l227_227896

theorem restaurant_tip_difference
  (a b : ℝ)
  (h1 : 0.15 * a = 3)
  (h2 : 0.25 * b = 3)
  : a - b = 8 := 
sorry

end restaurant_tip_difference_l227_227896


namespace translate_line_up_l227_227744

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := 2 * x - 4

-- Define the new line equation after translating upwards by 5 units
def new_line (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement to prove the translation result
theorem translate_line_up (x : ℝ) : original_line x + 5 = new_line x :=
by
  -- This would normally be where the proof goes, but we'll insert a placeholder
  sorry

end translate_line_up_l227_227744


namespace vector_addition_l227_227690

-- Definitions for the vectors
def a : ℝ × ℝ := (5, 2)
def b : ℝ × ℝ := (1, 6)

-- Proof statement (Note: "theorem" is used here instead of "def" because we are stating something to be proven)
theorem vector_addition : a + b = (6, 8) := by
  sorry

end vector_addition_l227_227690


namespace intervals_of_monotonicity_max_min_values_l227_227466

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 - 2

theorem intervals_of_monotonicity :
  (∀ x: ℝ, x < 0 → deriv f x < 0) ∧
  (∀ x: ℝ, 0 < x ∧ x < 2 → deriv f x > 0) ∧
  (∀ x: ℝ, x > 2 → deriv f x < 0) :=
sorry

theorem max_min_values :
  ∃ c d: ℝ, (c = Sup (set_of(continuous_on f (set.Icc (-2) 2))) ∧
              d = Inf (set_of(continuous_on f (set.Icc (-2) 2))) ∧
              c = 18 ∧ d = -2) :=
sorry

end intervals_of_monotonicity_max_min_values_l227_227466


namespace trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l227_227937

namespace Trapezoid

def isosceles_trapezoid (AD BC : ℝ) := 
  AD = 20 ∧ BC = 12

def diagonal (AD BC : ℝ) (AC : ℝ) := 
  isosceles_trapezoid AD BC → AC = 8 * Real.sqrt 5

def leg (AD BC : ℝ) (CD : ℝ) := 
  isosceles_trapezoid AD BC → CD = 4 * Real.sqrt 5

theorem trapezoid_diagonal_is_8sqrt5 (AD BC AC : ℝ) : 
  diagonal AD BC AC :=
by
  intros
  sorry

theorem trapezoid_leg_is_4sqrt5 (AD BC CD : ℝ) : 
  leg AD BC CD :=
by
  intros
  sorry

end Trapezoid

end trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l227_227937


namespace prime_p_and_cube_l227_227881

noncomputable def p : ℕ := 307

theorem prime_p_and_cube (a : ℕ) (h : a^3 = 16 * p + 1) : 
  Nat.Prime p := by
  sorry

end prime_p_and_cube_l227_227881


namespace find_a_l227_227156

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a^2 - 1

theorem find_a (a : ℝ) (h : ∀ x ∈ (Set.Icc 1 2), f x a ≤ 16 ∧ ∃ y ∈ (Set.Icc 1 2), f y a = 16) : a = 3 ∨ a = -3 :=
by
  sorry

end find_a_l227_227156


namespace quadratic_solutions_l227_227865

theorem quadratic_solutions (x : ℝ) :
  (4 * x^2 - 6 * x = 0) ↔ (x = 0) ∨ (x = 3 / 2) :=
sorry

end quadratic_solutions_l227_227865


namespace set_union_intersection_l227_227851

-- Definitions
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}
def C : Set ℤ := {1, 2}

-- Theorem statement
theorem set_union_intersection : (A ∩ B ∪ C) = {0, 1, 2} :=
by
  sorry

end set_union_intersection_l227_227851


namespace pairs_of_polygons_with_angle_ratio_l227_227888

theorem pairs_of_polygons_with_angle_ratio :
  ∃ n, n = 2 ∧ (∀ {k r : ℕ}, (k > 2 ∧ r > 2) → 
  (4 * (180 * r - 360) = 3 * (180 * k - 360) →
  ((k = 3 ∧ r = 18) ∨ (k = 2 ∧ r = 6)))) :=
by
  -- The proof should be provided here, but we skip it
  sorry

end pairs_of_polygons_with_angle_ratio_l227_227888


namespace ab_range_l227_227847

theorem ab_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + b + 8) : a * b ≥ 16 :=
sorry

end ab_range_l227_227847


namespace average_marks_l227_227646

theorem average_marks (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) 
                      (chemistry_marks : ℕ) (biology_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 →
  chemistry_marks = 87 → biology_marks = 81 → 
  (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / 5 = 85 :=
by
  intros
  sorry

end average_marks_l227_227646


namespace maximize_profit_l227_227443
-- Importing the entire necessary library

-- Definitions and conditions
def cost_price : ℕ := 40
def minimum_selling_price : ℕ := 44
def maximum_profit_margin : ℕ := 30
def sales_at_minimum_price : ℕ := 300
def price_increase_effect : ℕ := 10
def max_profit_price := 52
def max_profit := 2640

-- Function relationship between y and x
def sales_volume (x : ℕ) : ℕ := 300 - 10 * (x - 44)

-- Range of x
def valid_price (x : ℕ) : Prop := 44 ≤ x ∧ x ≤ 52

-- Statement of the problem
theorem maximize_profit (x : ℕ) (hx : valid_price x) : 
  sales_volume x = 300 - 10 * (x - 44) ∧
  44 ≤ x ∧ x ≤ 52 ∧
  x = 52 → 
  (x - cost_price) * (sales_volume x) = max_profit :=
sorry

end maximize_profit_l227_227443


namespace B_listing_method_l227_227685

-- Definitions for given conditions
def A : Set ℤ := {-2, -1, 1, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t*t}

-- The mathematically equivalent proof problem
theorem B_listing_method :
  B = {4, 1, 9, 16} := 
by {
  sorry
}

end B_listing_method_l227_227685


namespace gcd_765432_654321_eq_3_l227_227570

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227570


namespace Rohit_is_to_the_east_of_starting_point_l227_227010

-- Define the conditions and the problem statement.
def Rohit's_movements_proof
  (distance_south : ℕ) (distance_first_left : ℕ) (distance_second_left : ℕ) (distance_right : ℕ)
  (final_distance : ℕ) : Prop :=
  distance_south = 25 ∧
  distance_first_left = 20 ∧
  distance_second_left = 25 ∧
  distance_right = 15 ∧
  final_distance = 35 →
  (direction : String) → (distance : ℕ) →
  direction = "east" ∧ distance = final_distance

-- We can now state the theorem
theorem Rohit_is_to_the_east_of_starting_point :
  Rohit's_movements_proof 25 20 25 15 35 :=
by
  sorry

end Rohit_is_to_the_east_of_starting_point_l227_227010


namespace remaining_speed_l227_227752
open Real

theorem remaining_speed
  (D T : ℝ) (h1 : 40 * (T / 3) = (2 / 3) * D)
  (h2 : (T / 3) * 3 = T) :
  (D / 3) / ((2 * ((2 / 3) * D) / (40) / (3)) * 2 / 3) = 10 :=
by
  sorry

end remaining_speed_l227_227752


namespace exponent_of_5_in_30_factorial_l227_227169

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l227_227169


namespace weeks_jake_buys_papayas_l227_227167

theorem weeks_jake_buys_papayas
  (jake_papayas : ℕ)
  (brother_papayas : ℕ)
  (father_papayas : ℕ)
  (total_papayas : ℕ)
  (h1 : jake_papayas = 3)
  (h2 : brother_papayas = 5)
  (h3 : father_papayas = 4)
  (h4 : total_papayas = 48) :
  (total_papayas / (jake_papayas + brother_papayas + father_papayas) = 4) :=
by
  sorry

end weeks_jake_buys_papayas_l227_227167


namespace normal_distribution_probability_l227_227672

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

end normal_distribution_probability_l227_227672


namespace part1_part2_l227_227951

-- Part 1
theorem part1 (x : ℝ) (h1 : 2 * x = 3 * x - 1) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h2 : x < 0) (h3 : |2 * x| + |3 * x - 1| = 16) : x = -3 :=
by
  sorry

end part1_part2_l227_227951


namespace jean_total_calories_l227_227323

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l227_227323


namespace new_volume_of_cylinder_l227_227530

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

end new_volume_of_cylinder_l227_227530


namespace ratio_of_areas_l227_227820

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l227_227820


namespace correct_random_error_causes_l227_227750

-- Definitions based on conditions
def is_random_error_cause (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

-- Theorem: Valid causes of random errors are options (1), (2), and (3)
theorem correct_random_error_causes :
  (is_random_error_cause 1) ∧ (is_random_error_cause 2) ∧ (is_random_error_cause 3) :=
by
  sorry

end correct_random_error_causes_l227_227750


namespace factorize_expression_l227_227266

theorem factorize_expression (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) :=
by
  sorry

end factorize_expression_l227_227266


namespace passenger_capacity_passenger_capacity_at_5_max_profit_l227_227379

section SubwayProject

-- Define the time interval t and the passenger capacity function p(t)
def p (t : ℕ) : ℕ :=
  if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2
  else if 10 ≤ t ∧ t ≤ 20 then 500
  else 0

-- Define the net profit function Q(t)
def Q (t : ℕ) : ℚ :=
  if 2 ≤ t ∧ t < 10 then (8 * p t - 2656) / t - 60
  else if 10 ≤ t ∧ t ≤ 20 then (1344 : ℚ) / t - 60
  else 0

-- Statement 1: Prove the correct expression for p(t) and its value at t = 5
theorem passenger_capacity (t : ℕ) (ht1 : 2 ≤ t) (ht2 : t ≤ 20) :
  (p t = if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2 else 500) :=
sorry

theorem passenger_capacity_at_5 : p 5 = 450 :=
sorry

-- Statement 2: Prove the time interval t and the maximum value of Q(t)
theorem max_profit : ∃ t : ℕ, 2 ≤ t ∧ t ≤ 10 ∧ Q t = 132 ∧ (∀ u : ℕ, 2 ≤ u ∧ u ≤ 10 → Q u ≤ Q t) :=
sorry

end SubwayProject

end passenger_capacity_passenger_capacity_at_5_max_profit_l227_227379


namespace alejandro_rearrangement_l227_227707

noncomputable def rearrange_alejandro : Nat :=
  let X_choices := 2
  let total_letters := 8
  let repeating_letter_factorial := 2
  X_choices * Nat.factorial total_letters / Nat.factorial repeating_letter_factorial

theorem alejandro_rearrangement : rearrange_alejandro = 40320 := by
  sorry

end alejandro_rearrangement_l227_227707


namespace greatest_common_divisor_546_180_l227_227387

theorem greatest_common_divisor_546_180 : 
  ∃ d, d < 70 ∧ d > 0 ∧ d ∣ 546 ∧ d ∣ 180 ∧ ∀ x, x < 70 ∧ x > 0 ∧ x ∣ 546 ∧ x ∣ 180 → x ≤ d → x = 6 :=
by
  sorry

end greatest_common_divisor_546_180_l227_227387


namespace expand_polynomial_l227_227656

variable {R : Type*} [CommRing R]

-- Define the polynomial expression
def polynomial_expansion (x : R) : R := (7 * x^2 + 5 * x + 8) * 3 * x

-- The theorem to expand the expression
theorem expand_polynomial (x : R) :
  polynomial_expansion x = 21 * x^3 + 15 * x^2 + 24 * x :=
by {
  sorry
}

end expand_polynomial_l227_227656


namespace arithmetic_sequence_sum_l227_227514

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) 
  (h2 : S m = 0) 
  (h3 : S (m + 1) = 3) : 
  m = 5 :=
by sorry

end arithmetic_sequence_sum_l227_227514


namespace hyperbola_range_l227_227154

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (|m| - 1) - y^2 / (m - 2) = 1)) → (-1 < m ∧ m < 1) ∨ (m > 2) := by
  sorry

end hyperbola_range_l227_227154


namespace circle_equation_coefficients_l227_227203

theorem circle_equation_coefficients (a : ℝ) (x y : ℝ) : 
  (a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
by 
  sorry

end circle_equation_coefficients_l227_227203


namespace gcd_765432_654321_l227_227565

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227565


namespace prime_exponent_of_5_in_30_factorial_l227_227171

theorem prime_exponent_of_5_in_30_factorial : 
  (nat.factorial 30).factor_count 5 = 7 := 
by
  sorry

end prime_exponent_of_5_in_30_factorial_l227_227171


namespace gcd_765432_654321_l227_227592

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227592


namespace task_force_allocation_l227_227759

-- Define conditions
def subsidiaries := 6
def task_force_size := 8
def required_personnel (subsidiaries: ℕ) (task_force_size: ℕ) : Prop :=
  ∃ count_from_subs : fin subsidiaries → ℕ, (∀ i, 1 ≤ count_from_subs i) ∧ (∑ i, count_from_subs i) = task_force_size

-- Define the problem statement
theorem task_force_allocation :
  required_personnel subsidiaries task_force_size → 
  ∃ n, n = 21 :=
sorry

end task_force_allocation_l227_227759


namespace checkerboard_probability_l227_227858

-- Define the number of squares in the checkerboard and the number on the perimeter
def total_squares : Nat := 10 * 10
def perimeter_squares : Nat := 10 + 10 + (10 - 2) + (10 - 2)

-- The number of squares not on the perimeter
def inner_squares : Nat := total_squares - perimeter_squares

-- The probability that a randomly chosen square does not touch the outer edge
def probability_not_on_perimeter : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  probability_not_on_perimeter = 16 / 25 :=
by
  -- proof goes here
  sorry

end checkerboard_probability_l227_227858


namespace valid_paths_l227_227978

theorem valid_paths : 
  ∀ (paths_from : ℕ → ℕ → ℕ), 
  ∀ (x y : ℕ), 
  (∀ (x y : ℕ), x = 0 ∧ y = 0 → paths_from x y = 1) → 
  (∀ (x y : ℕ), x > 0 → paths_from x y = paths_from (x - 1) y + paths_from x (y - 1)) → 
  (paths_from 3 2) - (paths_from 1 1 * paths_from (3 - 1) (2 - 1)) = 4 :=
by
  sorry

end valid_paths_l227_227978


namespace proj_onto_w_equals_correct_l227_227663

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

end proj_onto_w_equals_correct_l227_227663


namespace Marcus_ate_more_than_John_l227_227077

theorem Marcus_ate_more_than_John:
  let John_eaten := 28
  let Marcus_eaten := 40
  Marcus_eaten - John_eaten = 12 :=
by
  sorry

end Marcus_ate_more_than_John_l227_227077


namespace solve_equation_l227_227124

theorem solve_equation :
  ∀ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 105 :=
by
  intros x h
  sorry

end solve_equation_l227_227124


namespace no_common_multiples_of_3_l227_227348

-- Define the sets X and Y
def SetX : Set ℤ := {n | 1 ≤ n ∧ n ≤ 24 ∧ n % 2 = 1}
def SetY : Set ℤ := {n | 0 ≤ n ∧ n ≤ 40 ∧ n % 2 = 0}

-- Define the condition for being a multiple of 3
def isMultipleOf3 (n : ℤ) : Prop := n % 3 = 0

-- Define the intersection of SetX and SetY that are multiples of 3
def intersectionMultipleOf3 : Set ℤ := {n | n ∈ SetX ∧ n ∈ SetY ∧ isMultipleOf3 n}

-- Prove that the set is empty
theorem no_common_multiples_of_3 : intersectionMultipleOf3 = ∅ := by
  sorry

end no_common_multiples_of_3_l227_227348


namespace chandler_bike_purchase_weeks_l227_227435

theorem chandler_bike_purchase_weeks (bike_cost birthday_money weekly_earnings total_weeks : ℕ) 
  (h_bike_cost : bike_cost = 600)
  (h_birthday_money : birthday_money = 60 + 40 + 20 + 30)
  (h_weekly_earnings : weekly_earnings = 18)
  (h_total_weeks : total_weeks = 25) :
  birthday_money + weekly_earnings * total_weeks = bike_cost :=
by {
  sorry
}

end chandler_bike_purchase_weeks_l227_227435


namespace find_f_expression_l227_227676

theorem find_f_expression (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = x^2 + 2 * x + 1 :=
by
  sorry

end find_f_expression_l227_227676


namespace angle_complement_half_supplement_is_zero_l227_227780

theorem angle_complement_half_supplement_is_zero (x : ℝ) 
  (h_complement: x - 90 = (1 / 2) * (x - 180)) : x = 0 := 
sorry

end angle_complement_half_supplement_is_zero_l227_227780


namespace line_contains_point_l227_227260

theorem line_contains_point {
    k : ℝ
} :
  (2 - k * 3 = -4 * 1) → k = 2 :=
by
  sorry

end line_contains_point_l227_227260


namespace order_of_three_numbers_l227_227026

theorem order_of_three_numbers :
  let a := (7 : ℝ) ^ (0.3 : ℝ)
  let b := (0.3 : ℝ) ^ (7 : ℝ)
  let c := Real.log (0.3 : ℝ)
  a > b ∧ b > c ∧ a > c :=
by
  sorry

end order_of_three_numbers_l227_227026


namespace first_tray_holds_260_cups_l227_227516

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

end first_tray_holds_260_cups_l227_227516


namespace number_of_integers_satisfying_sqrt_condition_l227_227042

noncomputable def count_integers_satisfying_sqrt_condition : ℕ :=
  let S := {x : ℕ | 3 < real.sqrt x ∧ real.sqrt x < 5}
  finset.card (finset.filter (λ x, 3 < real.sqrt x ∧ real.sqrt x < 5) (finset.range 26))

theorem number_of_integers_satisfying_sqrt_condition :
  count_integers_satisfying_sqrt_condition = 15 :=
sorry

end number_of_integers_satisfying_sqrt_condition_l227_227042


namespace simplify_expression_l227_227724

theorem simplify_expression (a b : ℕ) (h : a / b = 1 / 3) : 
    1 - (a - b) / (a - 2 * b) / ((a ^ 2 - b ^ 2) / (a ^ 2 - 4 * a * b + 4 * b ^ 2)) = 3 / 4 := 
by sorry

end simplify_expression_l227_227724


namespace satisfactory_fraction_is_28_over_31_l227_227968

-- Define the number of students for each grade
def students_with_grade_A := 8
def students_with_grade_B := 7
def students_with_grade_C := 6
def students_with_grade_D := 4
def students_with_grade_E := 3
def students_with_grade_F := 3

-- Calculate the total number of students with satisfactory grades
def satisfactory_grades := students_with_grade_A + students_with_grade_B + students_with_grade_C + students_with_grade_D + students_with_grade_E

-- Calculate the total number of students
def total_students := satisfactory_grades + students_with_grade_F

-- Define the fraction of satisfactory grades
def satisfactory_fraction : ℚ := satisfactory_grades / total_students

-- The main proposition that the satisfactory fraction is 28/31
theorem satisfactory_fraction_is_28_over_31 : satisfactory_fraction = 28 / 31 := by {
  sorry
}

end satisfactory_fraction_is_28_over_31_l227_227968


namespace possible_initial_triangles_l227_227011

-- Define the triangle types by their angles in degrees
inductive TriangleType
| T45T45T90
| T30T60T90
| T30T30T120
| T60T60T60

-- Define a Lean statement to express the problem
theorem possible_initial_triangles (T : TriangleType) :
  T = TriangleType.T45T45T90 ∨
  T = TriangleType.T30T60T90 ∨
  T = TriangleType.T30T30T120 ∨
  T = TriangleType.T60T60T60 :=
sorry

end possible_initial_triangles_l227_227011


namespace chess_tournament_points_l227_227166

theorem chess_tournament_points (boys girls : ℕ) (total_points : ℝ) 
  (total_matches : ℕ)
  (matches_among_boys points_among_boys : ℕ)
  (matches_among_girls points_among_girls : ℕ)
  (matches_between points_between : ℕ)
  (total_players : ℕ := boys + girls)
  (H1 : boys = 9) (H2 : girls = 3) (H3 : total_players = 12)
  (H4 : total_matches = total_players * (total_players - 1) / 2) 
  (H5 : total_points = total_matches) 
  (H6 : matches_among_boys = boys * (boys - 1) / 2) 
  (H7 : points_among_boys = matches_among_boys)
  (H8 : matches_among_girls = girls * (girls - 1) / 2) 
  (H9 : points_among_girls = matches_among_girls) 
  (H10 : matches_between = boys * girls) 
  (H11 : points_between = matches_between) :
  ¬ ∃ (P_B P_G : ℝ) (x : ℝ),
    P_B = points_among_boys + x ∧
    P_G = points_among_girls + (points_between - x) ∧
    P_B = P_G := by
  sorry

end chess_tournament_points_l227_227166


namespace papers_delivered_to_sunday_only_houses_l227_227845

-- Define the number of houses in the route and the days
def houses_in_route : ℕ := 100
def days_monday_to_saturday : ℕ := 6

-- Define the number of customers that do not get the paper on Sunday
def non_customers_sunday : ℕ := 10
def total_papers_per_week : ℕ := 720

-- Define the required number of papers delivered on Sunday to houses that only get the paper on Sunday
def papers_only_on_sunday : ℕ :=
  total_papers_per_week - (houses_in_route * days_monday_to_saturday) - (houses_in_route - non_customers_sunday)

theorem papers_delivered_to_sunday_only_houses : papers_only_on_sunday = 30 :=
by
  sorry

end papers_delivered_to_sunday_only_houses_l227_227845


namespace solve_system_of_equations_l227_227726

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2^(x + 2*y) + 2^x = 3 * 2^y) ∧ (2^(2*x + y) + 2 * 2^y = 4 * 2^x) ∧ (x = 1 / 2) ∧ (y = 1 / 2) := 
by
  let x := (1:ℝ) / 2
  let y := (1:ℝ) / 2
  have h1 : 2^(x + 2*y) + 2^x = 3 * 2^y := sorry
  have h2 : 2^(2*x + y) + 2 * 2^y = 4 * 2^x := sorry
  exact ⟨x, y, h1, h2, rfl, rfl⟩

end solve_system_of_equations_l227_227726


namespace gcd_ab_a2b2_eq_one_or_two_l227_227857

-- Definitions and conditions
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Problem statement
theorem gcd_ab_a2b2_eq_one_or_two (a b : ℕ) (h : coprime a b) : 
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_eq_one_or_two_l227_227857


namespace ratio_of_larger_to_smaller_l227_227215

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) : x / y = 2 :=
sorry

end ratio_of_larger_to_smaller_l227_227215


namespace count_integers_in_interval_l227_227059

theorem count_integers_in_interval :
  ∃ (n : ℕ), (∀ x : ℤ, 25 > x ∧ x > 9 → 10 ≤ x ∧ x ≤ 24 → x ∈ (Finset.range (25 - 10 + 1)).map (λ i, i + 10)) ∧ n = (Finset.range (25 - 10 + 1)).card :=
sorry

end count_integers_in_interval_l227_227059


namespace water_level_after_opening_valve_l227_227384

-- Define the initial conditions and final height to be proved
def initial_water_height_cm : ℝ := 40
def initial_oil_height_cm : ℝ := 40
def water_density : ℝ := 1000
def oil_density : ℝ := 700
def final_water_height_cm : ℝ := 34

-- The proof that the final height of water after equilibrium will be 34 cm
theorem water_level_after_opening_valve :
  ∀ (h_w h_o : ℝ),
  (water_density * h_w = oil_density * h_o) ∧ (h_w + h_o = initial_water_height_cm + initial_oil_height_cm) →
  h_w = final_water_height_cm :=
by
  -- Here goes the proof, skipped with sorry
  sorry

end water_level_after_opening_valve_l227_227384


namespace tickets_per_ride_factor_l227_227885

theorem tickets_per_ride_factor (initial_tickets spent_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 40) 
  (h2 : spent_tickets = 28) 
  (h3 : remaining_tickets = initial_tickets - spent_tickets) : 
  ∃ k : ℕ, remaining_tickets = 12 ∧ (∀ m : ℕ, m ∣ remaining_tickets → m = k) → (k ∣ 12) :=
by
  sorry

end tickets_per_ride_factor_l227_227885


namespace Xiaolong_dad_age_correct_l227_227393
noncomputable def Xiaolong_age (x : ℕ) : ℕ := x
noncomputable def mom_age (x : ℕ) : ℕ := 9 * x
noncomputable def dad_age (x : ℕ) : ℕ := 9 * x + 3
noncomputable def dad_age_next_year (x : ℕ) : ℕ := 9 * x + 4
noncomputable def Xiaolong_age_next_year (x : ℕ) : ℕ := x + 1
noncomputable def dad_age_predicated_next_year (x : ℕ) : ℕ := 8 * (x + 1)

theorem Xiaolong_dad_age_correct (x : ℕ) (h : 9 * x + 4 = 8 * (x + 1)) : dad_age x = 39 := by
  sorry

end Xiaolong_dad_age_correct_l227_227393


namespace coprime_divides_product_l227_227863

theorem coprime_divides_product {a b n : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a ∣ n) (h3 : b ∣ n) : ab ∣ n :=
by
  sorry

end coprime_divides_product_l227_227863


namespace exists_bounding_constant_M_l227_227946

variable (α : ℝ) (a : ℕ → ℝ)
variable (hα : α > 1)
variable (h_seq : ∀ n : ℕ, n > 0 →
  a n.succ = a n + (a n / n) ^ α)

theorem exists_bounding_constant_M (h_a1 : 0 < a 1 ∧ a 1 < 1) : 
  ∃ M, ∀ n > 0, a n ≤ M := 
sorry

end exists_bounding_constant_M_l227_227946


namespace applicantA_stability_l227_227104

noncomputable def applicantA_probs : ℕ → ℚ := λ n, 
  match n with
  | 1 => 1 / 5
  | 2 => 3 / 5
  | 3 => 1 / 5
  | _ => 0
  end

noncomputable def applicantB_probs : ℕ → ℚ := λ n, 
  match n with
  | 0 => 1 / 27
  | 1 => 2 / 9
  | 2 => 4 / 9
  | 3 => 8 / 27
  | _ => 0
  end

theorem applicantA_stability (E_X : ℚ) (D_X : ℚ) (E_Y : ℚ) (D_Y : ℚ) :
  E_X = 2 → E_Y = 2 → D_X = 2 / 5 → D_Y = 2 / 3 → D_X < D_Y → 
  "Applicant A has a higher probability of passing the interview due to greater stability in performance." := 
  by
  intro h1 h2 h3 h4 h5
  sorry

end applicantA_stability_l227_227104


namespace cos_alpha_value_l227_227812

theorem cos_alpha_value (α β : Real) (hα1 : 0 < α) (hα2 : α < π / 2) 
    (hβ1 : π / 2 < β) (hβ2 : β < π) (hcosβ : Real.cos β = -1/3)
    (hsin_alpha_beta : Real.sin (α + β) = 1/3) : 
    Real.cos α = 4 * Real.sqrt 2 / 9 := by
  sorry

end cos_alpha_value_l227_227812


namespace min_value_x3y3z2_is_1_over_27_l227_227850

noncomputable def min_value_x3y3z2 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h' : 1 / x + 1 / y + 1 / z = 9) : ℝ :=
  x^3 * y^3 * z^2

theorem min_value_x3y3z2_is_1_over_27 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z)
  (h' : 1 / x + 1 / y + 1 / z = 9) : min_value_x3y3z2 x y z h h' = 1 / 27 :=
sorry

end min_value_x3y3z2_is_1_over_27_l227_227850


namespace dawn_bananas_l227_227438

-- Definitions of the given conditions
def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60
def donna_bananas : ℕ := 40

-- Proof that Dawn has 100 bananas
theorem dawn_bananas : (total_bananas - donna_bananas) - lydia_bananas = 100 := by
  sorry

end dawn_bananas_l227_227438


namespace find_total_price_l227_227363

-- Define the cost parameters
variables (sugar_price salt_price : ℝ)

-- Define the given conditions
def condition_1 : Prop := 2 * sugar_price + 5 * salt_price = 5.50
def condition_2 : Prop := sugar_price = 1.50

-- Theorem to be proven
theorem find_total_price (h1 : condition_1 sugar_price salt_price) (h2 : condition_2 sugar_price) : 
  3 * sugar_price + 1 * salt_price = 5.00 :=
by
  sorry

end find_total_price_l227_227363


namespace line_equation_of_point_and_slope_angle_l227_227205

theorem line_equation_of_point_and_slope_angle 
  (p : ℝ × ℝ) (θ : ℝ)
  (h₁ : p = (-1, 2))
  (h₂ : θ = 45) :
  ∃ (a b c : ℝ), a * (p.1) + b * (p.2) + c = 0 ∧ (a * 1 + b * 1 = c) :=
sorry

end line_equation_of_point_and_slope_angle_l227_227205


namespace frog_climbs_out_l227_227633

theorem frog_climbs_out (d climb slip : ℕ) (h : d = 20) (h_climb : climb = 3) (h_slip : slip = 2) :
  ∃ n : ℕ, n = 20 ∧ d ≤ n * (climb - slip) + climb :=
sorry

end frog_climbs_out_l227_227633


namespace det_B_l227_227713

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

end det_B_l227_227713


namespace gcd_of_765432_and_654321_l227_227584

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227584


namespace last_digit_of_S_l227_227648

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_S : last_digit (54 ^ 2020 + 28 ^ 2022) = 0 :=
by 
  -- The Lean proof steps would go here
  sorry

end last_digit_of_S_l227_227648


namespace find_b_squared_l227_227357

theorem find_b_squared
    (b : ℝ)
    (c_ellipse c_hyperbola a_ellipse a2_hyperbola b2_hyperbola : ℝ)
    (h1: a_ellipse^2 = 25)
    (h2 : b2_hyperbola = 9 / 4)
    (h3 : a2_hyperbola = 4)
    (h4 : c_hyperbola = Real.sqrt (a2_hyperbola + b2_hyperbola))
    (h5 : c_ellipse = c_hyperbola)
    (h6 : b^2 = a_ellipse^2 - c_ellipse^2)
: b^2 = 75 / 4 :=
sorry

end find_b_squared_l227_227357


namespace int_values_satisfying_inequality_l227_227065

theorem int_values_satisfying_inequality : 
  ∃ (N : ℕ), N = 15 ∧ ∀ (x : ℕ), 9 < x ∧ x < 25 → x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} →
  set.size {x | 9 < x ∧ x < 25 ∧ x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}} = N :=
by
  sorry

end int_values_satisfying_inequality_l227_227065


namespace dice_sum_surface_l227_227402

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l227_227402


namespace geometric_series_sum_condition_l227_227314

def geometric_series_sum (a q n : ℕ) : ℕ := a * (1 - q^n) / (1 - q)

theorem geometric_series_sum_condition (S : ℕ → ℕ) (a : ℕ) (q : ℕ) (h1 : a = 1) 
  (h2 : ∀ n, S n = geometric_series_sum a q n)
  (h3 : S 7 - 4 * S 6 + 3 * S 5 = 0) : 
  S 4 = 40 := 
by 
  sorry

end geometric_series_sum_condition_l227_227314


namespace jo_integer_max_l227_227843
noncomputable def jo_integer : Nat :=
  let n := 166
  n

theorem jo_integer_max (n : Nat) (h1 : n < 200) (h2 : ∃ k : Nat, n + 2 = 9 * k) (h3 : ∃ l : Nat, n + 4 = 10 * l) : n ≤ jo_integer := 
by
  unfold jo_integer
  sorry

end jo_integer_max_l227_227843


namespace range_of_a_l227_227286

noncomputable def has_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0

def holds_for_all_x (a : ℝ) : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^2 - 3*a - x + 1 ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ((has_real_roots a) ∧ (holds_for_all_x a))) ∧ (¬ (¬ (holds_for_all_x a))) → (1 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l227_227286


namespace new_circle_contains_center_prob_l227_227413

noncomputable def probability_new_circle_contains_center (R : ℝ) (C : set (ℝ × ℝ)) (O : ℝ × ℝ)
(center_in_C : ∀ x ∈ C, ∃ (r : ℝ), 0 ≤ r ∧ r ≤ R - dist O x ∧ ball x r ⊆ C)
: ℝ :=
let prob := ∫ x in 0..R, (1 - x / R) in prob / R

theorem new_circle_contains_center_prob :
  ∀ (R : ℝ) (C : set (ℝ × ℝ)) (O : ℝ × ℝ)
  (center_in_C : ∀ x ∈ C, ∃ (r : ℝ), 0 ≤ r ∧ r ≤ R - dist O x ∧ ball x r ⊆ C),
  probability_new_circle_contains_center R C O center_in_C = 1 / 4 :=
begin
  assume R C O h,
  rw probability_new_circle_contains_center,
  sorry
end

end new_circle_contains_center_prob_l227_227413


namespace solution_inequality_l227_227873

theorem solution_inequality {x : ℝ} : x - 1 > 0 ↔ x > 1 := 
by
  sorry

end solution_inequality_l227_227873


namespace gcd_of_765432_and_654321_l227_227581

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227581


namespace num_integers_satisfying_sqrt_ineq_l227_227037

theorem num_integers_satisfying_sqrt_ineq:
  {x : ℕ} (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) →
  Finset.card (Finset.filter (λ x, 3 < Real.sqrt x ∧ Real.sqrt x < 5) (Finset.range 25)) = 15 :=
by
  sorry

end num_integers_satisfying_sqrt_ineq_l227_227037


namespace sum_of_digits_eq_28_l227_227975

theorem sum_of_digits_eq_28 (A B C D E : ℕ) 
  (hA : 0 ≤ A ∧ A ≤ 9) 
  (hB : 0 ≤ B ∧ B ≤ 9) 
  (hC : 0 ≤ C ∧ C ≤ 9) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (unique_digits : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E)) 
  (h : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 28 :=
sorry

end sum_of_digits_eq_28_l227_227975


namespace geometric_sequence_sum_l227_227834

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (a1 : a 1 = 3)
  (a4 : a 4 = 24)
  (h_geo : ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 1)) :
  a 3 + a 4 + a 5 = 84 :=
by
  sorry

end geometric_sequence_sum_l227_227834


namespace fair_game_expected_winnings_l227_227217

theorem fair_game_expected_winnings (num_players : ℕ) (total_pot : ℝ) 
  (p : ℕ → ℝ) (stakes : ℕ → ℝ) :
  num_players = 36 →
  (∀ k, p k = (35 / 36) ^ (k - 1) * p 1) →
  (∀ k, stakes k = total_pot * p k) →
  (∀ k, let L_k := stakes k in total_pot * p k - L_k * p k - L_k + L_k * p k = 0) :=
sorry

end fair_game_expected_winnings_l227_227217


namespace prob_not_lose_when_A_plays_l227_227016

def appearance_prob_center_forward : ℝ := 0.3
def appearance_prob_winger : ℝ := 0.5
def appearance_prob_attacking_midfielder : ℝ := 0.2

def lose_prob_center_forward : ℝ := 0.3
def lose_prob_winger : ℝ := 0.2
def lose_prob_attacking_midfielder : ℝ := 0.2

theorem prob_not_lose_when_A_plays : 
    (appearance_prob_center_forward * (1 - lose_prob_center_forward) + 
    appearance_prob_winger * (1 - lose_prob_winger) + 
    appearance_prob_attacking_midfielder * (1 - lose_prob_attacking_midfielder)) = 0.77 := 
by
  sorry

end prob_not_lose_when_A_plays_l227_227016


namespace molecular_weight_of_one_mole_l227_227602

-- Conditions
def molecular_weight_6_moles : ℤ := 1404
def num_moles : ℤ := 6

-- Theorem
theorem molecular_weight_of_one_mole : (molecular_weight_6_moles / num_moles) = 234 := by
  sorry

end molecular_weight_of_one_mole_l227_227602


namespace value_of_fraction_l227_227485

theorem value_of_fraction (x y : ℤ) (h : x / y = 7 / 2) : (x - 2 * y) / y = 3 / 2 := by
  sorry

end value_of_fraction_l227_227485


namespace maximum_items_6_yuan_l227_227536

theorem maximum_items_6_yuan :
  ∃ (x : ℕ), (∀ (x' : ℕ), (∃ (y z : ℕ), 6 * x' + 4 * y + 2 * z = 60 ∧ x' + y + z = 16) →
    x' ≤ 7) → x = 7 :=
by
  sorry

end maximum_items_6_yuan_l227_227536


namespace intersection_M_N_l227_227296

noncomputable def set_M : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def set_N : Set ℝ := {x | Real.log x ≥ 0}

theorem intersection_M_N :
  {x | x ∈ set_M ∧ x ∈ set_N} = {x | 1 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_M_N_l227_227296


namespace Lee_payment_total_l227_227792

theorem Lee_payment_total 
  (ticket_price : ℝ := 10.00)
  (booking_fee : ℝ := 1.50)
  (youngest_discount : ℝ := 0.40)
  (oldest_discount : ℝ := 0.30)
  (middle_discount : ℝ := 0.20)
  (youngest_tickets : ℕ := 3)
  (oldest_tickets : ℕ := 3)
  (middle_tickets : ℕ := 4) :
  (youngest_tickets * (ticket_price * (1 - youngest_discount)) + 
   oldest_tickets * (ticket_price * (1 - oldest_discount)) + 
   middle_tickets * (ticket_price * (1 - middle_discount)) + 
   (youngest_tickets + oldest_tickets + middle_tickets) * booking_fee) = 86.00 :=
by 
  sorry

end Lee_payment_total_l227_227792


namespace count_integer_values_l227_227033

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end count_integer_values_l227_227033


namespace length_of_train_l227_227639

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

end length_of_train_l227_227639


namespace bcdeq65_l227_227803

theorem bcdeq65 (a b c d e f : ℝ)
  (h₁ : a * b * c = 130)
  (h₂ : c * d * e = 500)
  (h₃ : d * e * f = 250)
  (h₄ : (a * f) / (c * d) = 1) :
  b * c * d = 65 :=
sorry

end bcdeq65_l227_227803


namespace proving_four_digit_number_l227_227867

def distinct (a b c d : Nat) : Prop :=
a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def same_parity (x y : Nat) : Prop :=
(x % 2 = 0 ∧ y % 2 = 0) ∨ (x % 2 = 1 ∧ y % 2 = 1)

def different_parity (x y : Nat) : Prop :=
¬same_parity x y

theorem proving_four_digit_number :
  ∃ (A B C D : Nat),
    distinct A B C D ∧
    (different_parity A B → B ≠ 4) ∧
    (different_parity B C → C ≠ 3) ∧
    (different_parity C D → D ≠ 2) ∧
    (different_parity D A → A ≠ 1) ∧
    A + D < B + C ∧
    1000 * A + 100 * B + 10 * C + D = 2341 :=
by
  sorry

end proving_four_digit_number_l227_227867


namespace bicycle_has_four_wheels_l227_227680

-- Define the universe and properties of cars
axiom Car : Type
axiom Bicycle : Car
axiom has_four_wheels : Car → Prop
axiom all_cars_have_four_wheels : ∀ c : Car, has_four_wheels c

-- Define the theorem
theorem bicycle_has_four_wheels : has_four_wheels Bicycle :=
by
  sorry

end bicycle_has_four_wheels_l227_227680


namespace amount_of_medication_B_l227_227091

def medicationAmounts (x y : ℝ) : Prop :=
  (x + y = 750) ∧ (0.40 * x + 0.20 * y = 215)

theorem amount_of_medication_B (x y : ℝ) (h : medicationAmounts x y) : y = 425 :=
  sorry

end amount_of_medication_B_l227_227091


namespace hazel_salmon_caught_l227_227149

-- Define the conditions
def father_salmon_caught : Nat := 27
def total_salmon_caught : Nat := 51

-- Define the main statement to be proved
theorem hazel_salmon_caught : total_salmon_caught - father_salmon_caught = 24 := by
  sorry

end hazel_salmon_caught_l227_227149


namespace count_integers_satisfying_condition_l227_227070

theorem count_integers_satisfying_condition :
  (card {x : ℤ | 9 < x ∧ x < 25} = 15) :=
by
  sorry

end count_integers_satisfying_condition_l227_227070


namespace number_of_storks_joined_l227_227537

theorem number_of_storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (total_birds_and_storks : ℕ) 
    (h1 : initial_birds = 3) (h2 : initial_storks = 4) (h3 : total_birds_and_storks = 13) : 
    (total_birds_and_storks - (initial_birds + initial_storks)) = 6 := 
by
  sorry

end number_of_storks_joined_l227_227537


namespace labor_union_trees_l227_227236

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end labor_union_trees_l227_227236


namespace avg_price_of_racket_l227_227243

theorem avg_price_of_racket (total_revenue : ℝ) (pairs_sold : ℝ) (h1 : total_revenue = 686) (h2 : pairs_sold = 70) : 
  total_revenue / pairs_sold = 9.8 := by
  sorry

end avg_price_of_racket_l227_227243


namespace solution_set_of_inequality_l227_227271

theorem solution_set_of_inequality :
  {x : ℝ | 4*x^2 - 9*x > 5} = {x : ℝ | x < -1/4} ∪ {x : ℝ | x > 5} :=
by
  sorry

end solution_set_of_inequality_l227_227271


namespace remainder_1234_mul_2047_mod_600_l227_227605

theorem remainder_1234_mul_2047_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end remainder_1234_mul_2047_mod_600_l227_227605


namespace model_x_computers_used_l227_227090

theorem model_x_computers_used
    (x_rate : ℝ)
    (y_rate : ℝ)
    (combined_rate : ℝ)
    (num_computers : ℝ) :
    x_rate = 1 / 72 →
    y_rate = 1 / 36 →
    combined_rate = num_computers * (x_rate + y_rate) →
    combined_rate = 1 →
    num_computers = 24 := by
  intros h1 h2 h3 h4
  sorry

end model_x_computers_used_l227_227090


namespace negation_of_forall_x_squared_nonnegative_l227_227208

theorem negation_of_forall_x_squared_nonnegative :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_forall_x_squared_nonnegative_l227_227208


namespace calories_consumed_Jean_l227_227328

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l227_227328


namespace cross_product_correct_l227_227659

def v : ℝ × ℝ × ℝ := (-3, 4, 5)
def w : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1,
 a.2.2 * b.1 - a.1 * b.2.2,
 a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct : cross_product v w = (21, 22, -5) :=
by
  sorry

end cross_product_correct_l227_227659


namespace domain_of_composite_l227_227952

theorem domain_of_composite (f : ℝ → ℝ) (x : ℝ) (hf : ∀ y, (0 ≤ y ∧ y ≤ 1) → f y = f y) :
  (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) →
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ 2*x ∧ 2*x ≤ 1 ∧ 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 →
  0 ≤ x ∧ x ≤ 1/2 :=
by
  intro h1 h2 h3 h4
  have h5: 0 ≤ 2*x ∧ 2*x ≤ 1 := sorry
  have h6: 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 := sorry
  sorry

end domain_of_composite_l227_227952


namespace largest_number_is_A_l227_227893

-- Definitions of the numbers
def numA := 8.45678
def numB := 8.456777777 -- This should be represented properly with an infinite sequence in a real formal proof
def numC := 8.456767676 -- This should be represented properly with an infinite sequence in a real formal proof
def numD := 8.456756756 -- This should be represented properly with an infinite sequence in a real formal proof
def numE := 8.456745674 -- This should be represented properly with an infinite sequence in a real formal proof

-- Lean statement to prove that numA is the largest number
theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE :=
by
  -- Proof not provided, sorry to skip
  sorry

end largest_number_is_A_l227_227893


namespace theater_total_cost_l227_227426

theorem theater_total_cost 
  (cost_orchestra : ℕ) (cost_balcony : ℕ)
  (total_tickets : ℕ) (ticket_difference : ℕ)
  (O B : ℕ)
  (h1 : cost_orchestra = 12)
  (h2 : cost_balcony = 8)
  (h3 : total_tickets = 360)
  (h4 : ticket_difference = 140)
  (h5 : O + B = total_tickets)
  (h6 : B = O + ticket_difference) :
  12 * O + 8 * B = 3320 :=
by
  sorry

end theater_total_cost_l227_227426


namespace find_sandwich_cost_l227_227385

theorem find_sandwich_cost (S : ℝ) :
  3 * S + 2 * 4 = 26 → S = 6 :=
by
  intro h
  sorry

end find_sandwich_cost_l227_227385


namespace spending_50_dollars_l227_227972

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end spending_50_dollars_l227_227972


namespace ratio_of_line_cutting_median_lines_l227_227417

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem ratio_of_line_cutting_median_lines (A B C P Q : ℝ × ℝ) 
    (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (0, 0)) 
    (h_mid_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
    (h_mid_BC : Q = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) 
    (h_ratio : (Real.sqrt (P.1^2 + P.2^2) / Real.sqrt (Q.1^2 + Q.2^2)) = (Real.sqrt (Q.1^2 + Q.2^2) / Real.sqrt (P.1^2 + P.2^2))) :
  (P.1 / Q.1) = golden_ratio :=
by 
  sorry

end ratio_of_line_cutting_median_lines_l227_227417


namespace radius_of_inscribed_circle_l227_227496

theorem radius_of_inscribed_circle (r1 r2 : ℝ) (AC BC AB : ℝ) 
  (h1 : AC = 2 * r1)
  (h2 : BC = 2 * r2)
  (h3 : AB = 2 * Real.sqrt (r1^2 + r2^2)) : 
  (r1 + r2 - Real.sqrt (r1^2 + r2^2)) = ((2 * r1 + 2 * r2 - 2 * Real.sqrt (r1^2 + r2^2)) / 2) := 
by
  sorry

end radius_of_inscribed_circle_l227_227496


namespace sheela_monthly_income_l227_227227

theorem sheela_monthly_income (d : ℝ) (p : ℝ) (income : ℝ) (h1 : d = 4500) (h2 : p = 0.28) (h3 : d = p * income) : 
  income = 16071.43 :=
by
  sorry

end sheela_monthly_income_l227_227227


namespace no_integer_solutions_l227_227193

theorem no_integer_solutions (x y z : ℤ) (h : ¬ (x = 0 ∧ y = 0 ∧ z = 0)) : 2 * x^4 + y^4 ≠ 7 * z^4 :=
sorry

end no_integer_solutions_l227_227193


namespace maximum_PM_minus_PN_l227_227422

noncomputable def x_squared_over_9_minus_y_squared_over_16_eq_1 (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 4

noncomputable def circle2 (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 1

theorem maximum_PM_minus_PN :
  ∀ (P M N : ℝ × ℝ),
    x_squared_over_9_minus_y_squared_over_16_eq_1 P.1 P.2 →
    circle1 M.1 M.2 →
    circle2 N.1 N.2 →
    (|dist P M - dist P N| ≤ 9) := sorry

end maximum_PM_minus_PN_l227_227422


namespace gcd_765432_654321_eq_3_l227_227571

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227571


namespace fraction_identity_l227_227151

theorem fraction_identity (a b : ℝ) (h : a / b = 5 / 2) : (a + 2 * b) / (a - b) = 3 :=
by sorry

end fraction_identity_l227_227151


namespace shara_savings_l227_227244

theorem shara_savings 
  (original_price : ℝ)
  (discount1 : ℝ := 0.08)
  (discount2 : ℝ := 0.05)
  (sales_tax : ℝ := 0.06)
  (final_price : ℝ := 184)
  (h : (original_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) = final_price) :
  original_price - final_price = 25.78 :=
sorry

end shara_savings_l227_227244


namespace find_larger_number_l227_227094

theorem find_larger_number (hc_f : ℕ) (factor1 factor2 : ℕ)
(h_hcf : hc_f = 63)
(h_factor1 : factor1 = 11)
(h_factor2 : factor2 = 17)
(lcm := hc_f * factor1 * factor2)
(A := hc_f * factor1)
(B := hc_f * factor2) :
max A B = 1071 := by
  sorry

end find_larger_number_l227_227094


namespace num_male_students_selected_l227_227517

def total_students := 220
def male_students := 60
def selected_female_students := 32

def selected_male_students (total_students male_students selected_female_students : Nat) : Nat :=
  (selected_female_students * male_students) / (total_students - male_students)

theorem num_male_students_selected : selected_male_students total_students male_students selected_female_students = 12 := by
  unfold selected_male_students
  sorry

end num_male_students_selected_l227_227517


namespace cos_thirteen_pi_over_three_l227_227120

theorem cos_thirteen_pi_over_three : Real.cos (13 * Real.pi / 3) = 1 / 2 := 
by
  sorry

end cos_thirteen_pi_over_three_l227_227120


namespace dice_surface_sum_l227_227401

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l227_227401


namespace min_x_plus_y_l227_227279

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l227_227279


namespace ratio_of_areas_l227_227818

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l227_227818


namespace distinct_sums_is_98_l227_227673

def arithmetic_sequence_distinct_sums (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :=
  (∀ n : ℕ, S n = (n * (2 * a_n 0 + (n - 1) * d)) / 2) ∧
  S 5 = 0 ∧
  d ≠ 0 →
  (∃ distinct_count : ℕ, distinct_count = 98 ∧
   ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 ∧ S i = S j → i = j)

theorem distinct_sums_is_98 (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence_distinct_sums a_n S d) :
  ∃ distinct_count : ℕ, distinct_count = 98 :=
sorry

end distinct_sums_is_98_l227_227673


namespace sum_a_n_l227_227788

def a_n (n : ℕ) : ℕ :=
  if n % 30 = 0 then 15
  else if n % 60 = 0 then 10
  else if n % 60 = 0 then 12
  else 0

theorem sum_a_n : (∑ n in Finset.range 1500, a_n n) = 1263 := by
  sorry

end sum_a_n_l227_227788


namespace magnitude_of_z_l227_227275

open Complex

noncomputable def z : ℂ := (1 - I) / (1 + I) + 2 * I

theorem magnitude_of_z : Complex.abs z = 1 := by
  sorry

end magnitude_of_z_l227_227275


namespace ratio_of_areas_of_circles_l227_227823

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l227_227823


namespace tamara_is_17_over_6_times_taller_than_kim_l227_227866

theorem tamara_is_17_over_6_times_taller_than_kim :
  ∀ (T K : ℕ), T = 68 → T + K = 92 → (T : ℚ) / K = 17 / 6 :=
by
  intros T K hT hSum
  -- proof steps go here, but we use sorry to skip the proof
  sorry

end tamara_is_17_over_6_times_taller_than_kim_l227_227866


namespace total_peaches_l227_227347

-- Definitions of conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- Proof problem statement
theorem total_peaches : initial_peaches + picked_peaches = 68 :=
by
  -- Including sorry to skip the actual proof
  sorry

end total_peaches_l227_227347


namespace cube_root_expression_l227_227084

theorem cube_root_expression : 
  (∛(5^7 + 5^7 + 5^7) = 25 * ∛(25)) :=
by sorry

end cube_root_expression_l227_227084


namespace factorization_of_expression_l227_227934

noncomputable def factorized_form (x : ℝ) : ℝ :=
  (x + 5 / 2 + Real.sqrt 13 / 2) * (x + 5 / 2 - Real.sqrt 13 / 2)

theorem factorization_of_expression (x : ℝ) :
  x^2 - 5 * x + 3 = factorized_form x :=
by
  sorry

end factorization_of_expression_l227_227934


namespace proof_of_k_bound_l227_227853

noncomputable def sets_with_nonempty_intersection_implies_k_bound (k : ℝ) : Prop :=
  let M := {x : ℝ | -1 ≤ x ∧ x < 2}
  let N := {x : ℝ | x ≤ k + 3}
  M ∩ N ≠ ∅ → k ≥ -4

theorem proof_of_k_bound (k : ℝ) : sets_with_nonempty_intersection_implies_k_bound k := by
  intro h
  have : -1 ≤ k + 3 := sorry
  linarith

end proof_of_k_bound_l227_227853


namespace total_students_university_l227_227080

theorem total_students_university :
  ∀ (sample_size freshmen sophomores other_sample other_total total_students : ℕ),
  sample_size = 500 →
  freshmen = 200 →
  sophomores = 100 →
  other_sample = 200 →
  other_total = 3000 →
  total_students = (other_total * sample_size) / other_sample →
  total_students = 7500 :=
by
  intros sample_size freshmen sophomores other_sample other_total total_students
  sorry

end total_students_university_l227_227080


namespace volume_of_sphere_eq_4_sqrt3_pi_l227_227292

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

theorem volume_of_sphere_eq_4_sqrt3_pi
  (r : ℝ) (h : 4 * Real.pi * r ^ 2 = 2 * Real.sqrt 3 * Real.pi * (2 * r)) :
  volume_of_sphere r = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_sphere_eq_4_sqrt3_pi_l227_227292


namespace compute_expression_l227_227111

theorem compute_expression : (-9 * 3 - (-7 * -4) + (-11 * -6) = 11) := by
  sorry

end compute_expression_l227_227111


namespace number_of_integers_between_10_and_24_l227_227055

theorem number_of_integers_between_10_and_24 : 
  (set.count (set_of (λ x : ℤ, 9 < x ∧ x < 25))) = 15 := 
sorry

end number_of_integers_between_10_and_24_l227_227055


namespace number_of_connections_l227_227538

theorem number_of_connections (n m : ℕ) (h1 : n = 30) (h2 : m = 4) :
    (n * m) / 2 = 60 := by
  -- Since each switch is connected to 4 others,
  -- and each connection is counted twice, 
  -- the number of unique connections is 60.
  sorry

end number_of_connections_l227_227538


namespace max_value_x_minus_2y_l227_227701

open Real

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 
  x - 2*y ≤ 10 :=
sorry

end max_value_x_minus_2y_l227_227701


namespace bromine_is_liquid_at_25C_1atm_l227_227892

-- Definitions for the melting and boiling points
def melting_point (element : String) : Float :=
  match element with
  | "Br" => -7.2
  | "Kr" => -157.4 -- Not directly used, but included for completeness
  | "P" => 44.1 -- Not directly used, but included for completeness
  | "Xe" => -111.8 -- Not directly used, but included for completeness
  | _ => 0.0 -- default case; not used

def boiling_point (element : String) : Float :=
  match element with
  | "Br" => 58.8
  | "Kr" => -153.4
  | "P" => 280.5 -- Not directly used, but included for completeness
  | "Xe" => -108.1
  | _ => 0.0 -- default case; not used

-- Define the condition of the problem
def is_liquid_at (element : String) (temperature : Float) (pressure : Float) : Bool :=
  melting_point element < temperature ∧ temperature < boiling_point element

-- Goal statement
theorem bromine_is_liquid_at_25C_1atm : is_liquid_at "Br" 25 1 = true :=
by
  sorry

end bromine_is_liquid_at_25C_1atm_l227_227892


namespace find_x_l227_227396

theorem find_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 4) : x = 2 :=
by
  sorry

end find_x_l227_227396


namespace unique_zero_of_f_inequality_of_x1_x2_l227_227961

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l227_227961


namespace cube_root_simplified_l227_227085

noncomputable def cube_root_3 : Real := Real.cbrt 3
noncomputable def cube_root_5 : Real := Real.cbrt (5^7)

theorem cube_root_simplified :
  Real.cbrt (3 * 5^7) = 3^(1 / 3) * 5^(7 / 3) :=
by
  sorry

end cube_root_simplified_l227_227085


namespace sixth_graders_forgot_homework_percentage_l227_227230

-- Definitions of the conditions
def num_students_A : ℕ := 20
def num_students_B : ℕ := 80
def percent_forgot_A : ℚ := 20 / 100
def percent_forgot_B : ℚ := 15 / 100

-- Statement to be proven
theorem sixth_graders_forgot_homework_percentage :
  (num_students_A * percent_forgot_A + num_students_B * percent_forgot_B) /
  (num_students_A + num_students_B) = 16 / 100 :=
by
  sorry

end sixth_graders_forgot_homework_percentage_l227_227230


namespace difference_is_1365_l227_227354

-- Define the conditions as hypotheses
def difference_between_numbers (L S : ℕ) : Prop :=
  L = 1637 ∧ L = 6 * S + 5

-- State the theorem to prove the difference is 1365
theorem difference_is_1365 {L S : ℕ} (h₁ : L = 1637) (h₂ : L = 6 * S + 5) :
  L - S = 1365 :=
by
  sorry

end difference_is_1365_l227_227354


namespace count_integers_satisfying_condition_l227_227069

theorem count_integers_satisfying_condition :
  (card {x : ℤ | 9 < x ∧ x < 25} = 15) :=
by
  sorry

end count_integers_satisfying_condition_l227_227069


namespace triangle_area_solutions_l227_227105

theorem triangle_area_solutions (ABC BDE : ℝ) (k : ℝ) (h₁ : BDE = k^2) : 
  S >= 4 * k^2 ∧ (if S = 4 * k^2 then solutions = 1 else solutions = 2) :=
by
  sorry

end triangle_area_solutions_l227_227105


namespace count_integers_in_interval_l227_227057

theorem count_integers_in_interval :
  ∃ (n : ℕ), (∀ x : ℤ, 25 > x ∧ x > 9 → 10 ≤ x ∧ x ≤ 24 → x ∈ (Finset.range (25 - 10 + 1)).map (λ i, i + 10)) ∧ n = (Finset.range (25 - 10 + 1)).card :=
sorry

end count_integers_in_interval_l227_227057


namespace solve_inequality_l227_227777

theorem solve_inequality : 
  {x : ℝ | (x^3 - x^2 - 6 * x) / (x^2 - 3 * x + 2) > 0} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x)} :=
sorry

end solve_inequality_l227_227777


namespace nontrivial_solution_fraction_l227_227448

theorem nontrivial_solution_fraction (x y z : ℚ)
  (h₁ : x - 6 * y + 3 * z = 0)
  (h₂ : 3 * x - 6 * y - 2 * z = 0)
  (h₃ : x + 6 * y - 5 * z = 0)
  (hne : x ≠ 0) :
  (y * z) / (x^2) = 2 / 3 :=
by
  sorry

end nontrivial_solution_fraction_l227_227448


namespace Mary_and_Sandra_solution_l227_227994

theorem Mary_and_Sandra_solution (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  (2 * 40 + 3 * 60) * n / (5 * n) = (4 * 30 * n + 80 * m) / (4 * n + m) →
  m + n = 29 :=
by
  intro h
  sorry

end Mary_and_Sandra_solution_l227_227994


namespace smallest_sum_minimum_l227_227284

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l227_227284


namespace inequality_solution_l227_227725

open Real

theorem inequality_solution (a x : ℝ) :
  (a = 0 ∧ x > 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a = 1 ∧ ∀ x, ¬ (a * x^2 - (2 * a + 2) * x + 4 > 0)) ∨
  (a < 0 ∧ (x < 2/a ∨ x > 2) ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (0 < a ∧ a < 1 ∧ 2 < x ∧ x < 2/a ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a > 1 ∧ 2/a < x ∧ x < 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) := 
sorry

end inequality_solution_l227_227725


namespace no_int_coords_equilateral_l227_227109

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end no_int_coords_equilateral_l227_227109


namespace greatest_root_of_g_l227_227941

noncomputable def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x := 
by
  sorry

end greatest_root_of_g_l227_227941


namespace complex_numbers_count_l227_227490

theorem complex_numbers_count (z : ℂ) (h1 : z^24 = 1) (h2 : ∃ r : ℝ, z^6 = r) : ℕ :=
  sorry -- Proof goes here

end complex_numbers_count_l227_227490


namespace exists_xy_for_cube_difference_l227_227195

theorem exists_xy_for_cube_difference (a : ℕ) (h : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 :=
sorry

end exists_xy_for_cube_difference_l227_227195


namespace digit_sum_multiple_exists_ck_l227_227789

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum  

theorem digit_sum_multiple_exists_ck 
  (k : ℕ) 
  (hk : k > 1) 
  (n : ℕ) 
  (hn : n > 0) :
  ∃ (c_k : ℝ), c_k > 0 ∧ ∀ (n : ℕ), n > 0 → digit_sum (k * n) ≥ (c_k * digit_sum n) ↔ 
  ∀ p, p.prime → p ∣ k → p = 2 ∨ p = 5 :=
sorry

end digit_sum_multiple_exists_ck_l227_227789


namespace nuts_distributive_problem_l227_227886

theorem nuts_distributive_problem (x y : ℕ) (h1 : 70 ≤ x + y) (h2 : x + y ≤ 80) (h3 : (3 / 4 : ℚ) * x + (1 / 5 : ℚ) * (y + (1 / 4 : ℚ) * x) = (x : ℚ) + 1) :
  x = 36 ∧ y = 41 :=
by
  sorry

end nuts_distributive_problem_l227_227886


namespace problem_f_2004_l227_227467

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_f_2004 (a α b β : ℝ) 
  (h_non_zero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0) 
  (h_condition : f 2003 a α b β = 6) : 
  f 2004 a α b β = 2 := 
by
  sorry

end problem_f_2004_l227_227467


namespace fibonacci_expression_equality_l227_227356

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Statement to be proven
theorem fibonacci_expression_equality :
  (fibonacci 0 * fibonacci 2 + fibonacci 1 * fibonacci 3 + fibonacci 2 * fibonacci 4 +
  fibonacci 3 * fibonacci 5 + fibonacci 4 * fibonacci 6 + fibonacci 5 * fibonacci 7)
  - (fibonacci 1 ^ 2 + fibonacci 2 ^ 2 + fibonacci 3 ^ 2 + fibonacci 4 ^ 2 + fibonacci 5 ^ 2 + fibonacci 6 ^ 2)
  = 0 :=
by
  sorry

end fibonacci_expression_equality_l227_227356


namespace line_form_l227_227418

-- Given vector equation for a line
def line_eq (x y : ℝ) : Prop :=
  (3 * (x - 4) + 7 * (y - 14)) = 0

-- Prove that the line can be written in the form y = mx + b
theorem line_form (x y : ℝ) (h : line_eq x y) :
  y = (-3/7) * x + (110/7) :=
sorry

end line_form_l227_227418


namespace area_of_shaded_region_l227_227734

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

end area_of_shaded_region_l227_227734


namespace calories_consumed_Jean_l227_227329

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l227_227329


namespace vertex_angle_of_obtuse_isosceles_triangle_l227_227497

noncomputable def isosceles_obtuse_triangle (a b h : ℝ) (φ : ℝ) : Prop :=
  a^2 = 2 * b * h ∧
  b = 2 * a * Real.cos ((180 - φ) / 2) ∧
  h = a * Real.sin ((180 - φ) / 2) ∧
  90 < φ ∧ φ < 180

theorem vertex_angle_of_obtuse_isosceles_triangle (a b h : ℝ) (φ : ℝ) :
  isosceles_obtuse_triangle a b h φ → φ = 150 :=
by
  sorry

end vertex_angle_of_obtuse_isosceles_triangle_l227_227497


namespace areaOfPolarCurve_l227_227253

noncomputable def polarArea (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ φ in a..b, (f φ)^2

def polarEq (φ : ℝ) : ℝ := 1 + Real.sqrt 2 * Real.sin φ

theorem areaOfPolarCurve :
  polarArea polarEq (-Real.pi / 2) (Real.pi / 2) = 2 * Real.pi :=
by
  sorry

end areaOfPolarCurve_l227_227253


namespace sum_of_first_ten_primes_with_units_digit_3_l227_227786

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

def sum_first_ten_primes_units_digit_3 : ℕ :=
  first_ten_primes_units_digit_3.sum

theorem sum_of_first_ten_primes_with_units_digit_3 :
  sum_first_ten_primes_units_digit_3 = 793 := by
  -- Here we provide the steps as a placeholder, but in real practice,
  -- a proof should be constructed to verify this calculation.
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l227_227786


namespace relationship_between_a_and_b_l227_227456

theorem relationship_between_a_and_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x : ℝ, |(2 * x + 2)| < a → |(x + 1)| < b) : b ≥ a / 2 :=
by
  -- The proof steps will be inserted here
  sorry

end relationship_between_a_and_b_l227_227456


namespace sum_of_cubes_consecutive_integers_l227_227529

theorem sum_of_cubes_consecutive_integers (x : ℕ) (h1 : 0 < x) (h2 : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 :=
by
  -- proof will go here
  sorry

end sum_of_cubes_consecutive_integers_l227_227529


namespace range_k_domain_f_l227_227307

theorem range_k_domain_f :
  (∀ x : ℝ, x^2 - 6*k*x + k + 8 ≥ 0) ↔ (-8/9 ≤ k ∧ k ≤ 1) :=
sorry

end range_k_domain_f_l227_227307


namespace multiplier_for_obsolete_books_l227_227654

theorem multiplier_for_obsolete_books 
  (x : ℕ) 
  (total_books_removed number_of_damaged_books : ℕ) 
  (h1 : total_books_removed = 69) 
  (h2 : number_of_damaged_books = 11) 
  (h3 : number_of_damaged_books + (x * number_of_damaged_books - 8) = total_books_removed) 
  : x = 6 := 
by 
  sorry

end multiplier_for_obsolete_books_l227_227654


namespace number_of_integers_between_10_and_24_l227_227052

theorem number_of_integers_between_10_and_24 : 
  (set.count (set_of (λ x : ℤ, 9 < x ∧ x < 25))) = 15 := 
sorry

end number_of_integers_between_10_and_24_l227_227052


namespace gcd_765432_654321_l227_227588

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227588


namespace gcd_765432_654321_l227_227590

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227590


namespace gcd_765432_654321_l227_227546

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227546


namespace find_value_at_frac_one_third_l227_227677

theorem find_value_at_frac_one_third
  (f : ℝ → ℝ) 
  (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 1 / 4) :
  f (1 / 3) = 9 := 
  sorry

end find_value_at_frac_one_third_l227_227677


namespace jean_total_calories_l227_227324

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l227_227324


namespace ratio_of_areas_of_circles_l227_227822

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l227_227822


namespace solve_for_X_l227_227694

variable (X Y : ℝ)

def diamond (X Y : ℝ) := 4 * X + 3 * Y + 7

theorem solve_for_X (h : diamond X 5 = 75) : X = 53 / 4 :=
by
  sorry

end solve_for_X_l227_227694


namespace intersection_point_finv_l227_227359

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b

theorem intersection_point_finv (a b : ℤ) : 
  (∀ x : ℝ, f (f x b) b = x) → 
  (∀ y : ℝ, f (f y b) b = y) → 
  (f (-4) b = a) → 
  (f a b = -4) → 
  a = -4 := 
by
  intros
  sorry

end intersection_point_finv_l227_227359


namespace problem_l227_227153

variable (a b : ℝ)

theorem problem (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_l227_227153


namespace ways_to_distribute_soccer_balls_l227_227930

noncomputable def count_ways (n r s₁ s₂ : ℕ) : ℕ :=
  ∑ k in finset.range (r + 1), (-1)^k * nat.choose r k *
    nat.choose (n + r - s₁ * r - (s₂ - s₁ + 1) * k - 1) (r - 1)

theorem ways_to_distribute_soccer_balls (n r s₁ s₂ : ℕ)
  (h₁ : r * s₁ ≤ n)
  (h₂ : n ≤ r * s₂) :
  count_ways n r s₁ s₂ = 
  ∑ k in finset.range (r + 1), (-1)^k * nat.choose r k *
    nat.choose (n + r - s₁ * r - (s₂ - s₁ + 1) * k - 1) (r - 1) := sorry

end ways_to_distribute_soccer_balls_l227_227930


namespace dice_sum_surface_l227_227404

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l227_227404


namespace find_sum_l227_227898

theorem find_sum 
  (R : ℝ) -- Original interest rate
  (P : ℝ) -- Principal amount
  (h: (P * (R + 3) * 3 / 100) = ((P * R * 3 / 100) + 81)): 
  P = 900 :=
sorry

end find_sum_l227_227898


namespace inequality_solution_l227_227521

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

end inequality_solution_l227_227521


namespace find_2x_2y_2z_l227_227488

theorem find_2x_2y_2z (x y z : ℝ) 
  (h1 : y + z = 10 - 2 * x)
  (h2 : x + z = -12 - 4 * y)
  (h3 : x + y = 5 - 2 * z) : 
  2 * x + 2 * y + 2 * z = 3 :=
by
  sorry

end find_2x_2y_2z_l227_227488


namespace elderly_people_pears_l227_227831

theorem elderly_people_pears (x y : ℕ) :
  (y = x + 1) ∧ (2 * x = y + 2) ↔
  (x = y - 1) ∧ (2 * x = y + 2) := by
  sorry

end elderly_people_pears_l227_227831


namespace reciprocal_of_2023_l227_227369

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l227_227369


namespace cistern_emptying_time_l227_227905

noncomputable def cistern_time_without_tap (tap_rate : ℕ) (empty_time_with_tap : ℕ) (cistern_volume : ℕ) : ℕ := 
  let tap_total := tap_rate * empty_time_with_tap
  let leaked_volume := cistern_volume - tap_total
  let leak_rate := leaked_volume / empty_time_with_tap
  cistern_volume / leak_rate

theorem cistern_emptying_time :
  cistern_time_without_tap 4 24 480 = 30 := 
by
  unfold cistern_time_without_tap
  norm_num

end cistern_emptying_time_l227_227905


namespace smallest_sum_minimum_l227_227285

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l227_227285


namespace dice_surface_sum_l227_227400

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l227_227400


namespace Bella_age_l227_227921

theorem Bella_age (B : ℕ) (h₁ : ∃ n : ℕ, n = B + 9) (h₂ : B + (B + 9) = 19) : B = 5 := 
by
  sorry

end Bella_age_l227_227921


namespace a_10_eq_505_l227_227132

-- The sequence definition
def a (n : ℕ) : ℕ :=
  let start := (n * (n - 1)) / 2 + 1
  List.sum (List.range' start n)

-- Theorem that the 10th term of the sequence is 505
theorem a_10_eq_505 : a 10 = 505 := 
by
  sorry

end a_10_eq_505_l227_227132


namespace unique_zero_f_x1_minus_2x2_l227_227965

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l227_227965


namespace parabola_vertex_l227_227204

theorem parabola_vertex (x y : ℝ) :
  (x^2 - 4 * x + 3 * y + 8 = 0) → (x, y) = (2, -4 / 3) :=
by
  sorry

end parabola_vertex_l227_227204


namespace closest_ratio_l227_227489

theorem closest_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : (x + y) / 2 = 3 * Real.sqrt (x * y)) :
  abs (x / y - 34) < abs (x / y - n) :=
by sorry

end closest_ratio_l227_227489


namespace roots_rational_l227_227519

/-- Prove that the roots of the equation x^2 + px + q = 0 are always rational,
given the rational numbers p and q, and a rational n where p = n + q / n. -/
theorem roots_rational
  (n p q : ℚ)
  (hp : p = n + q / n)
  : ∃ x y : ℚ, x^2 + p * x + q = 0 ∧ y^2 + p * y + q = 0 ∧ x ≠ y :=
sorry

end roots_rational_l227_227519


namespace evaluate_expression_l227_227264

theorem evaluate_expression : 
  (Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3)) = Real.sqrt 3 + 3 + 5 / 6 := by
  sorry

end evaluate_expression_l227_227264


namespace smallest_possible_sum_l227_227280

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l227_227280


namespace area_of_plot_is_correct_l227_227240

-- Define the side length of the square plot
def side_length : ℝ := 50.5

-- Define the area of the square plot
def area_of_square (s : ℝ) : ℝ := s * s

-- Theorem stating that the area of a square plot with side length 50.5 m is 2550.25 m²
theorem area_of_plot_is_correct : area_of_square side_length = 2550.25 := by
  sorry

end area_of_plot_is_correct_l227_227240


namespace gcd_proof_l227_227560

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227560


namespace football_game_spectators_l227_227705

-- Define the conditions and the proof goals
theorem football_game_spectators 
  (A C : ℕ) 
  (h_condition_1 : 2 * A + 2 * C + 40 = 310) 
  (h_condition_2 : C = A / 2) : 
  A = 90 ∧ C = 45 ∧ (A + C + 20) = 155 := 
by 
  sorry

end football_game_spectators_l227_227705


namespace dice_sum_surface_l227_227403

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l227_227403


namespace equal_lead_concentration_l227_227944

theorem equal_lead_concentration (x : ℝ) (h1 : 0 < x) (h2 : x < 6) (h3 : x < 12) 
: (x / 6 = (12 - x) / 12) → x = 4 := by
  sorry

end equal_lead_concentration_l227_227944


namespace gcd_of_765432_and_654321_l227_227580

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227580


namespace probability_neither_event_l227_227870

-- Definitions of given probabilities
def P_soccer_match : ℚ := 5 / 8
def P_science_test : ℚ := 1 / 4

-- Calculations of the complements
def P_no_soccer_match : ℚ := 1 - P_soccer_match
def P_no_science_test : ℚ := 1 - P_science_test

-- Independence of events implies the probability of neither event is the product of their complements
theorem probability_neither_event :
  (P_no_soccer_match * P_no_science_test) = 9 / 32 :=
by
  sorry

end probability_neither_event_l227_227870


namespace unique_positive_real_solution_l227_227302

theorem unique_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ (x^8 + 5 * x^7 + 10 * x^6 + 2023 * x^5 - 2021 * x^4 = 0) := sorry

end unique_positive_real_solution_l227_227302


namespace exponent_of_five_in_30_factorial_l227_227177

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l227_227177


namespace height_relationship_l227_227746

theorem height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
sorry

end height_relationship_l227_227746


namespace reciprocal_of_2023_l227_227373

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l227_227373


namespace most_frequent_data_is_mode_l227_227731

def most_frequent_data_name (dataset : Type) : String := "Mode"

theorem most_frequent_data_is_mode (dataset : Type) :
  most_frequent_data_name dataset = "Mode" :=
by
  sorry

end most_frequent_data_is_mode_l227_227731


namespace solution_l227_227929

noncomputable def problem_statement (a b : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)

theorem solution (a b : ℝ) :
  problem_statement a b ↔ (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a' b' : ℤ, (a : ℝ) = a' ∧ (b : ℝ) = b')) :=
by
  sorry

end solution_l227_227929


namespace min_x_plus_y_l227_227278

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l227_227278


namespace trig_eq_solutions_l227_227232

open Real

theorem trig_eq_solutions (x : ℝ) :
  2 * sin x ^ 3 + 2 * sin x ^ 2 * cos x - sin x * cos x ^ 2 - cos x ^ 3 = 0 ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨ (∃ k : ℤ, x = arctan (sqrt 2 / 2) + k * π) ∨ (∃ m : ℤ, x = -arctan (sqrt 2 / 2) + m * π) :=
by
  sorry

end trig_eq_solutions_l227_227232


namespace complex_eq_z100_zReciprocal_l227_227950

theorem complex_eq_z100_zReciprocal
  (z : ℂ)
  (h : z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + z⁻¹^100 = -2 * Real.cos (40 * Real.pi / 180) :=
by
  sorry

end complex_eq_z100_zReciprocal_l227_227950


namespace depth_of_canal_l227_227617

/-- The cross-section of a canal is a trapezium with a top width of 12 meters, 
a bottom width of 8 meters, and an area of 840 square meters. 
Prove that the depth of the canal is 84 meters.
-/
theorem depth_of_canal (top_width bottom_width area : ℝ) (h : ℝ) :
  top_width = 12 → bottom_width = 8 → area = 840 → 1 / 2 * (top_width + bottom_width) * h = area → h = 84 :=
by
  intros ht hb ha h_area
  sorry

end depth_of_canal_l227_227617


namespace new_paint_intensity_l227_227351

-- Definition of the given conditions
def original_paint_intensity : ℝ := 0.15
def replacement_paint_intensity : ℝ := 0.25
def fraction_replaced : ℝ := 1.5
def original_volume : ℝ := 100

-- Proof statement
theorem new_paint_intensity :
  (original_volume * original_paint_intensity + original_volume * fraction_replaced * replacement_paint_intensity) /
  (original_volume + original_volume * fraction_replaced) = 0.21 :=
by
  sorry

end new_paint_intensity_l227_227351


namespace gcd_765432_654321_eq_3_l227_227577

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227577


namespace solution_set_of_quadratic_inequality_l227_227377

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | 2 - x - x^2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_quadratic_inequality_l227_227377


namespace avg_cards_removed_until_prime_l227_227790

theorem avg_cards_removed_until_prime:
  let prime_count := 13
  let cards_count := 42
  let non_prime_count := cards_count - prime_count
  let groups_count := prime_count + 1
  let avg_non_prime_per_group := (non_prime_count: ℚ) / (groups_count: ℚ)
  (groups_count: ℚ) > 0 →
  avg_non_prime_per_group + 1 = (43: ℚ) / (14: ℚ) :=
by
  sorry

end avg_cards_removed_until_prime_l227_227790


namespace tens_digit_of_7_pow_35_l227_227222

theorem tens_digit_of_7_pow_35 : 
  (7 ^ 35) % 100 / 10 % 10 = 4 :=
by
  sorry

end tens_digit_of_7_pow_35_l227_227222


namespace original_price_of_shirt_l227_227872

theorem original_price_of_shirt (discounted_price : ℝ) (discount_percentage : ℝ) 
  (h_discounted_price : discounted_price = 780) (h_discount_percentage : discount_percentage = 0.20) 
  : (discounted_price / (1 - discount_percentage) = 975) := by
  sorry

end original_price_of_shirt_l227_227872


namespace final_amount_l227_227498

-- Definitions for the initial amount, price per pound, and quantity purchased.
def initial_amount : ℕ := 20
def price_per_pound : ℕ := 2
def quantity_purchased : ℕ := 3

-- Formalizing the statement
theorem final_amount (A P Q : ℕ) (hA : A = initial_amount) (hP : P = price_per_pound) (hQ : Q = quantity_purchased) :
  A - P * Q = 14 :=
by
  sorry

end final_amount_l227_227498


namespace joe_total_time_l227_227499

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

end joe_total_time_l227_227499


namespace value_of_a_l227_227469

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - 3 * x - Real.log x + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem value_of_a (a x0 : ℝ) (h : f x0 a = 3) : a = 1 - Real.log 2 :=
by
  sorry

end value_of_a_l227_227469


namespace union_of_M_and_N_l227_227798

namespace SetOperations

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3, 4} :=
sorry

end SetOperations

end union_of_M_and_N_l227_227798


namespace count_integers_satisfying_sqrt_condition_l227_227044

noncomputable def count_integers_in_range (lower upper : ℕ) : ℕ :=
    (upper - lower + 1)

/- Proof statement for the given problem -/
theorem count_integers_satisfying_sqrt_condition :
  let conditions := (∀ x : ℕ, 5 > Real.sqrt x ∧ Real.sqrt x > 3) in
  count_integers_in_range 10 24 = 15 :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l227_227044


namespace product_of_real_roots_l227_227967

theorem product_of_real_roots (x1 x2 : ℝ) (h1 : x1^2 - 6 * x1 + 8 = 0) (h2 : x2^2 - 6 * x2 + 8 = 0) :
  x1 * x2 = 8 := 
sorry

end product_of_real_roots_l227_227967


namespace find_c_l227_227429

theorem find_c (a b c : ℤ) (h1 : a + b * c = 2017) (h2 : b + c * a = 8) :
  c = -6 ∨ c = 0 ∨ c = 2 ∨ c = 8 :=
by 
  sorry

end find_c_l227_227429


namespace PE_bisects_CD_given_conditions_l227_227509

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

end PE_bisects_CD_given_conditions_l227_227509


namespace gcd_proof_l227_227556

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227556


namespace number_of_female_students_l227_227201

noncomputable def total_students : ℕ := 1600
noncomputable def sample_size : ℕ := 200
noncomputable def sampled_males : ℕ := 110
noncomputable def sampled_females := sample_size - sampled_males
noncomputable def total_males := (sampled_males * total_students) / sample_size
noncomputable def total_females := total_students - total_males

theorem number_of_female_students : total_females = 720 := 
sorry

end number_of_female_students_l227_227201


namespace sin_cos_identity_l227_227273

theorem sin_cos_identity (a : ℝ) (h : Real.sin (π - a) = -2 * Real.sin (π / 2 + a)) : 
  Real.sin a * Real.cos a = -2 / 5 :=
by
  sorry

end sin_cos_identity_l227_227273


namespace cos_double_angle_l227_227130

variables {α β : ℝ}

theorem cos_double_angle (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_l227_227130


namespace sum_of_first_ten_primes_ending_in_3_is_671_l227_227785

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l227_227785


namespace problem_translation_l227_227846

variables {a : ℕ → ℤ} (S : ℕ → ℤ)

-- Definition of the arithmetic sequence and its sum function
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ (d : ℤ), ∀ (n m : ℕ), a (n + 1) = a n + d

-- Sum of the first n terms defined recursively
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 0 then 0 else a n + sum_first_n_terms a (n - 1)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : S 5 > S 6

-- To be proved: Option D does not necessarily hold
theorem problem_translation : ¬(a 3 + a 6 + a 12 < 2 * a 7) := sorry

end problem_translation_l227_227846


namespace surface_area_of_sphere_l227_227880

noncomputable def volume : ℝ := 72 * Real.pi

theorem surface_area_of_sphere (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = volume) :
  4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  sorry

end surface_area_of_sphere_l227_227880


namespace shenzhen_vaccination_count_l227_227523

theorem shenzhen_vaccination_count :
  2410000 = 2.41 * 10^6 :=
  sorry

end shenzhen_vaccination_count_l227_227523


namespace complete_the_square_correct_l227_227389

noncomputable def complete_the_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 1 = 0 ↔ (x - 1)^2 = 2

theorem complete_the_square_correct : ∀ x : ℝ, complete_the_square x := by
  sorry

end complete_the_square_correct_l227_227389


namespace compute_expression_l227_227257

theorem compute_expression :
  (-9 * 5 - (-7 * -2) + (-11 * -4)) = -15 :=
by
  sorry

end compute_expression_l227_227257


namespace gcd_lcm_sum_l227_227607

theorem gcd_lcm_sum (a b : ℕ) (h₁ : a = 120) (h₂ : b = 3507) :
  Nat.gcd a b + Nat.lcm a b = 140283 := by 
  sorry

end gcd_lcm_sum_l227_227607


namespace nth_term_arithmetic_sequence_l227_227125

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 4 * n + 5 * n^2

theorem nth_term_arithmetic_sequence :
  (S r) - (S (r-1)) = 10 * r - 1 :=
by
  sorry

end nth_term_arithmetic_sequence_l227_227125


namespace gcd_765432_654321_l227_227569

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227569


namespace total_legs_of_collection_l227_227340

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end total_legs_of_collection_l227_227340


namespace integer_values_count_l227_227072

theorem integer_values_count (x : ℕ) : (∃ y : ℤ, 10 ≤ y ∧ y ≤ 24) ↔ (∑ y in (finset.interval 10 24), 1) = 15 :=
by
  sorry

end integer_values_count_l227_227072


namespace ellipse_eccentricity_l227_227679

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c^2 = a^2 - b^2)
  (h4 : let F := (-c, 0) in let A := (F.1 / 2, (sqrt 3 / 2) * F.1) in (A.1^2 / a^2) + (A.2^2 / b^2) = 1)
  : ∃ e : ℝ, e = sqrt 3 - 1 :=
by
  sorry

end ellipse_eccentricity_l227_227679


namespace reciprocal_2023_l227_227367

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l227_227367


namespace determine_p_l227_227647

noncomputable def roots (p : ℝ) : ℝ × ℝ :=
  let discr := p ^ 2 - 48
  ((-p + Real.sqrt discr) / 2, (-p - Real.sqrt discr) / 2)

theorem determine_p (p : ℝ) :
  let (x1, x2) := roots p
  (x1 - x2 = 1) → (p = 7 ∨ p = -7) :=
by
  intros
  sorry

end determine_p_l227_227647


namespace morning_snowfall_l227_227832

theorem morning_snowfall (total_snowfall afternoon_snowfall morning_snowfall : ℝ) 
  (h1 : total_snowfall = 0.625) 
  (h2 : afternoon_snowfall = 0.5) 
  (h3 : total_snowfall = morning_snowfall + afternoon_snowfall) : 
  morning_snowfall = 0.125 :=
by
  sorry

end morning_snowfall_l227_227832


namespace tetrahedron_in_cube_l227_227945

theorem tetrahedron_in_cube (a x : ℝ) (h : a = 6) :
  (∃ x, x = 6 * Real.sqrt 2) :=
sorry

end tetrahedron_in_cube_l227_227945


namespace k_range_correct_l227_227721

noncomputable def k_range (k : ℝ) : Prop :=
  (∀ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
  (∀ x : ℝ, k * x ^ 2 + k * x + 1 > 0) ∧
  ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∨
   (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0)) ∧
  ¬ ((∃ x : ℝ, ¬ (x ^ 2 + k * x + 9 / 4 = 0)) ∧
    (∃ x : ℝ, k * x ^ 2 + k * x + 1 > 0))

theorem k_range_correct (k : ℝ) : k_range k ↔ (-3 < k ∧ k < 0) ∨ (3 ≤ k ∧ k < 4) :=
sorry

end k_range_correct_l227_227721


namespace oranges_weight_l227_227226

theorem oranges_weight (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := 
by 
  sorry

end oranges_weight_l227_227226


namespace dice_surface_sum_l227_227407

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l227_227407


namespace min_ratio_number_l227_227395

theorem min_ratio_number (H T U : ℕ) (h1 : H - T = 8 ∨ T - H = 8) (hH : 1 ≤ H ∧ H ≤ 9) (hT : 0 ≤ T ∧ T ≤ 9) (hU : 0 ≤ U ∧ U ≤ 9) :
  100 * H + 10 * T + U = 190 :=
by sorry

end min_ratio_number_l227_227395


namespace exponent_of_5_in_30_fact_l227_227180

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l227_227180


namespace total_population_l227_227316

-- Define the predicates for g, b, and s based on t
variables (g b t s : ℕ)

-- The conditions given in the problem
def condition1 : Prop := g = 4 * t
def condition2 : Prop := b = 6 * g
def condition3 : Prop := s = t / 2

-- The theorem stating the total population is equal to (59 * t) / 2
theorem total_population (g b t s : ℕ) (h1 : condition1 g t) (h2 : condition2 b g) (h3 : condition3 s t) :
  b + g + t + s = 59 * t / 2 :=
by sorry

end total_population_l227_227316


namespace gcd_proof_l227_227559

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227559


namespace correct_coefficient_l227_227087

-- Definitions based on given conditions
def isMonomial (expr : String) : Prop := true

def coefficient (expr : String) : ℚ :=
  if expr = "-a/3" then -1/3 else 0

-- Statement to prove
theorem correct_coefficient : coefficient "-a/3" = -1/3 :=
by
  sorry

end correct_coefficient_l227_227087


namespace trigonometric_identity_l227_227459

open Real

variable (α : ℝ)
variable (h1 : π < α)
variable (h2 : α < 2 * π)
variable (h3 : cos (α - 7 * π) = -3 / 5)

theorem trigonometric_identity :
  sin (3 * π + α) * tan (α - 7 * π / 2) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l227_227459


namespace proof_problem_l227_227303

noncomputable def a : ℝ := 0.85 * 250
noncomputable def b : ℝ := 0.75 * 180
noncomputable def c : ℝ := 0.90 * 320

theorem proof_problem :
  (a - b = 77.5) ∧ (77.5 < c) :=
by
  sorry

end proof_problem_l227_227303


namespace distance_from_apex_to_larger_cross_section_l227_227762

namespace PyramidProof

variables (As Al : ℝ) (d h : ℝ)

theorem distance_from_apex_to_larger_cross_section 
  (As_eq : As = 256 * Real.sqrt 2) 
  (Al_eq : Al = 576 * Real.sqrt 2) 
  (d_eq : d = 12) :
  h = 36 := 
sorry

end PyramidProof

end distance_from_apex_to_larger_cross_section_l227_227762


namespace problem_proof_l227_227799

noncomputable def ellipse_equation_exists (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

noncomputable def equation_of_ellipse_C : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  (let C := λ x y, (x^2 / a^2) + (y^2 / b^2) = 1 in
  C 1 (3/2) ∧
  (∃ c : ℝ, b = sqrt 3 * c ∧ 1 / a^2 + 9 / (4 * b^2) = 1 ∧ a^2 = b^2 + c^2) ∧
  a^2 = 4 ∧ b^2 = 3 ∧ C x y)

noncomputable def line_l_exists_and_bisects (k m : ℝ) (h : (k = 1/2 ∨ k = -1/2) ∧ m = sqrt 21 / 7) : Prop :=
  ∃ x1 x2 : ℝ, 
  (let ellipse_eqn := λ x y, (x^2 / 4) + (y^2 / 3) = 1 in
  ∀ l : ℝ → ℝ, l = λ x, k * x + m →
  ∀ (N M : ℝ × ℝ), N = (-(m / k), 0) ∧ M = (0, m) →
  ∀ (P Q : ℝ × ℝ), P = ((m / k), 2 * m) ∧ Q = ((m / k), -2 * m) →
  ∀ (A B : ℝ × ℝ), A = (x1, (k * x1 + m)) ∧ B = (x2, (-3 * k * x2 + m)) →
  ∀ (A1 B1 : ℝ × ℝ), A1 = (x1, 0) ∧ B1 = (x2, 0) →
  (N.1 = (A1.1 + B1.1) / 2))

theorem problem_proof :
  exists (a b : ℝ), (a > b) ∧ (b > 0) ∧
  ellipse_equation_exists a b ∧
  equation_of_ellipse_C ∧
  exists (k m : ℝ), line_l_exists_and_bisects k m sorry :=
begin
  sorry
end

end problem_proof_l227_227799


namespace candy_bar_calories_unit_l227_227535

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

end candy_bar_calories_unit_l227_227535


namespace triangle_area_of_parabola_hyperbola_l227_227199

-- Definitions for parabola and hyperbola
def parabola_directrix (a : ℕ) (x y : ℝ) : Prop := x^2 = 16 * y
def hyperbola_asymptotes (a b : ℕ) (x y : ℝ) : Prop := x^2 / (a^2) - y^2 / (b^2) = 1

-- Theorem stating the area of the triangle formed by the intersections of the asymptotes with the directrix
theorem triangle_area_of_parabola_hyperbola (a b : ℕ) (h : a = 1) (h' : b = 1) : 
  ∃ (area : ℝ), area = 16 :=
sorry

end triangle_area_of_parabola_hyperbola_l227_227199


namespace complex_number_problem_l227_227511

variables {a b c x y z : ℂ}

theorem complex_number_problem (h1 : a = (b + c) / (x - 2))
    (h2 : b = (c + a) / (y - 2))
    (h3 : c = (a + b) / (z - 2))
    (h4 : x * y + y * z + z * x = 67)
    (h5 : x + y + z = 2010) :
    x * y * z = -5892 :=
sorry

end complex_number_problem_l227_227511


namespace dice_surface_sum_l227_227399

noncomputable def surface_sum_of_dice (n : ℕ) := 28175 + 2 * n

theorem dice_surface_sum (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6):
  ∃ s, s ∈ {28177, 28179, 28181, 28183, 28185, 28187} ∧ s = surface_sum_of_dice X := by
  use surface_sum_of_dice X
  have : 28175 + 2 * X ∈ {28177, 28179, 28181, 28183, 28185, 28187} := by
    interval_cases X
    all_goals simp [surface_sum_of_dice]
  exact ⟨this, rfl⟩
sorry

end dice_surface_sum_l227_227399


namespace find_overtime_hours_l227_227766

theorem find_overtime_hours
  (pay_rate_ordinary : ℝ := 0.60)
  (pay_rate_overtime : ℝ := 0.90)
  (total_pay : ℝ := 32.40)
  (total_hours : ℕ := 50) :
  ∃ y : ℕ, pay_rate_ordinary * (total_hours - y) + pay_rate_overtime * y = total_pay ∧ y = 8 := 
by
  sorry

end find_overtime_hours_l227_227766


namespace proposition_q_false_for_a_lt_2_l227_227008

theorem proposition_q_false_for_a_lt_2 (a : ℝ) (h : a < 2) : 
  ¬ ∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1 :=
sorry

end proposition_q_false_for_a_lt_2_l227_227008


namespace maximum_area_right_triangle_in_rectangle_l227_227423

theorem maximum_area_right_triangle_in_rectangle :
  ∃ (area : ℕ), 
  (∀ (a b : ℕ), a = 12 ∧ b = 5 → area = 1 / 2 * a * b) :=
by
  use 30
  sorry

end maximum_area_right_triangle_in_rectangle_l227_227423


namespace general_formula_arithmetic_sequence_l227_227029

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

theorem general_formula_arithmetic_sequence (x : ℝ) (a : ℕ → ℝ) 
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n : ℕ, (a n = 2 * n - 4) ∨ (a n = 4 - 2 * n) :=
by
  sorry

end general_formula_arithmetic_sequence_l227_227029


namespace trapezoid_area_equal_l227_227495

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

end trapezoid_area_equal_l227_227495


namespace sum_of_first_ten_primes_with_units_digit_three_l227_227783

-- Define the problem to prove the sum of the first 10 primes ending in 3 is 639
theorem sum_of_first_ten_primes_with_units_digit_three : 
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  in list.sum primes_with_units_digit_three = 639 := 
by 
  -- We define the primes with the units digit 3 as given and check the sum
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  show list.sum primes_with_units_digit_three = 639 from sorry

end sum_of_first_ten_primes_with_units_digit_three_l227_227783


namespace pyramid_area_ratio_l227_227025

theorem pyramid_area_ratio (S S1 S2 : ℝ) (h1 : S1 = (99 / 100)^2 * S) (h2 : S2 = (1 / 100)^2 * S) :
  S1 / S2 = 9801 := by
  sorry

end pyramid_area_ratio_l227_227025


namespace different_colors_at_minus_plus_1990_l227_227651

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

end different_colors_at_minus_plus_1990_l227_227651


namespace repeat_45_fraction_repeat_245_fraction_l227_227434

-- Define the repeating decimal 0.454545... == n / d
def repeating_45_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.45454545 = (n : ℚ) / (d : ℚ))

-- First problem statement: 0.4545... == 5 / 11
theorem repeat_45_fraction : 0.45454545 = (5 : ℚ) / (11 : ℚ) :=
by
  sorry

-- Define the repeating decimal 0.2454545... == n / d
def repeating_245_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.2454545 = (n : ℚ) / (d : ℚ))

-- Second problem statement: 0.2454545... == 27 / 110
theorem repeat_245_fraction : 0.2454545 = (27 : ℚ) / (110 : ℚ) :=
by
  sorry

end repeat_45_fraction_repeat_245_fraction_l227_227434


namespace exists_c_d_rel_prime_l227_227848

theorem exists_c_d_rel_prime (a b : ℤ) :
  ∃ c d : ℤ, ∀ n : ℤ, gcd (a * n + c) (b * n + d) = 1 :=
sorry

end exists_c_d_rel_prime_l227_227848


namespace scientific_notation_of_138000_l227_227006

noncomputable def scientific_notation_equivalent (n : ℕ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * (10:ℝ)^exp

theorem scientific_notation_of_138000 : scientific_notation_equivalent 138000 1.38 5 :=
by
  sorry

end scientific_notation_of_138000_l227_227006


namespace negation_of_universal_l227_227361

theorem negation_of_universal {x : ℝ} : ¬ (∀ x > 0, x^2 - x ≤ 0) ↔ ∃ x > 0, x^2 - x > 0 :=
by
  sorry

end negation_of_universal_l227_227361


namespace probability_all_same_color_l227_227624

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

end probability_all_same_color_l227_227624


namespace race_outcomes_210_l227_227248

-- Define the participants
def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fern", "Grace"]

-- The question is to prove the number of different 1st-2nd-3rd place outcomes is 210.
theorem race_outcomes_210 (h : participants.length = 7) : (7 * 6 * 5 = 210) :=
  by sorry

end race_outcomes_210_l227_227248


namespace incorrect_conclusion_l227_227272

theorem incorrect_conclusion (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a < 1/b ∧ 1/b < 0) : ¬ (ab > b^2) :=
by
  { sorry }

end incorrect_conclusion_l227_227272


namespace distinct_terms_in_expansion_l227_227301

theorem distinct_terms_in_expansion :
  let n1 := 2 -- number of terms in (x + y)
  let n2 := 3 -- number of terms in (a + b + c)
  let n3 := 3 -- number of terms in (d + e + f)
  (n1 * n2 * n3) = 18 :=
by
  sorry

end distinct_terms_in_expansion_l227_227301


namespace total_legs_correct_l227_227337

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end total_legs_correct_l227_227337


namespace total_annual_gain_l227_227640

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

end total_annual_gain_l227_227640


namespace associate_professor_pencils_l227_227642

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

end associate_professor_pencils_l227_227642


namespace trajectory_of_M_l227_227159

theorem trajectory_of_M
  (x y : ℝ)
  (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
sorry

end trajectory_of_M_l227_227159


namespace intersection_of_sets_l227_227686

def setA : Set ℝ := {x | x^2 - 1 ≥ 0}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets : (setA ∩ setB) = {x | 1 ≤ x ∧ x < 4} := 
by 
  sorry

end intersection_of_sets_l227_227686


namespace correct_time_fraction_l227_227411

theorem correct_time_fraction : (3 / 4 : ℝ) * (3 / 4 : ℝ) = (9 / 16 : ℝ) :=
by
  sorry

end correct_time_fraction_l227_227411


namespace max_slope_tangent_eqn_l227_227122

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem max_slope_tangent_eqn (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
    (∃ m b, m = Real.sqrt 2 ∧ b = -Real.sqrt 2 * (Real.pi / 4) ∧ 
    (∀ y, y = m * x + b)) :=
sorry

end max_slope_tangent_eqn_l227_227122


namespace infinite_solutions_of_system_l227_227727

theorem infinite_solutions_of_system :
  ∃x y : ℝ, (3 * x - 4 * y = 10 ∧ 6 * x - 8 * y = 20) :=
by
  sorry

end infinite_solutions_of_system_l227_227727


namespace ice_cubes_total_l227_227928

theorem ice_cubes_total (initial_cubes made_cubes : ℕ) (h_initial : initial_cubes = 2) (h_made : made_cubes = 7) : initial_cubes + made_cubes = 9 :=
by
  sorry

end ice_cubes_total_l227_227928


namespace sum_f_values_l227_227813

noncomputable def f (x : ℤ) : ℤ := (x - 1)^3 + 1

theorem sum_f_values :
  (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7) = 13 :=
by
  sorry

end sum_f_values_l227_227813


namespace cylinder_lateral_surface_area_l227_227737

theorem cylinder_lateral_surface_area 
  (diameter height : ℝ) 
  (h1 : diameter = 2) 
  (h2 : height = 2) : 
  2 * Real.pi * (diameter / 2) * height = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l227_227737


namespace find_x_such_that_g_inverse_of_x_is_neg2_l227_227471

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 3

theorem find_x_such_that_g_inverse_of_x_is_neg2 : g (-2) = -43 :=
by 
  rw [g]
  simp
  norm_num
  sorry

end find_x_such_that_g_inverse_of_x_is_neg2_l227_227471


namespace outfit_count_l227_227966

def num_shirts := 8
def num_hats := 8
def num_pants := 4

def shirt_colors := 6
def hat_colors := 6
def pants_colors := 4

def total_possible_outfits := num_shirts * num_hats * num_pants

def same_color_restricted_outfits := 4 * 8 * 7

def num_valid_outfits := total_possible_outfits - same_color_restricted_outfits

theorem outfit_count (h1 : num_shirts = 8) (h2 : num_hats = 8) (h3 : num_pants = 4)
                     (h4 : shirt_colors = 6) (h5 : hat_colors = 6) (h6 : pants_colors = 4)
                     (h7 : total_possible_outfits = 256) (h8 : same_color_restricted_outfits = 224) :
  num_valid_outfits = 32 :=
by
  sorry

end outfit_count_l227_227966


namespace circle_area_ratio_is_correct_l227_227824

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l227_227824


namespace salary_May_l227_227018

theorem salary_May
  (J F M A M' : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + M') / 4 = 8400)
  (h3 : J = 4900) :
  M' = 6500 :=
  by
  sorry

end salary_May_l227_227018


namespace snow_shoveling_l227_227221

noncomputable def volume_of_snow_shoveled (length1 length2 width depth1 depth2 : ℝ) : ℝ :=
  (length1 * width * depth1) + (length2 * width * depth2)

theorem snow_shoveling :
  volume_of_snow_shoveled 15 15 4 1 (1 / 2) = 90 :=
by
  sorry

end snow_shoveling_l227_227221


namespace travel_west_l227_227492

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

end travel_west_l227_227492


namespace exponent_of_5_in_30_factorial_l227_227176

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l227_227176


namespace johns_cookies_left_l227_227505

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end johns_cookies_left_l227_227505


namespace valid_grid_iff_divisible_by_9_l227_227620

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

end valid_grid_iff_divisible_by_9_l227_227620


namespace number_of_integers_between_10_and_24_l227_227053

theorem number_of_integers_between_10_and_24 : 
  (set.count (set_of (λ x : ℤ, 9 < x ∧ x < 25))) = 15 := 
sorry

end number_of_integers_between_10_and_24_l227_227053


namespace third_cyclist_speed_l227_227794

theorem third_cyclist_speed (s1 s3 : ℝ) :
  (∃ s1 s3 : ℝ,
    (∀ t : ℝ, t > 0 → (s1 > s3) ∧ (20 = abs (10 * t - s1 * t)) ∧ (5 = abs (s1 * t - s3 * t)) ∧ (s1 ≥ 10))) →
  (s3 = 25 ∨ s3 = 5) :=
by sorry

end third_cyclist_speed_l227_227794


namespace housewife_spending_l227_227241

theorem housewife_spending (P R A : ℝ) (h1 : R = 34.2) (h2 : R = 0.8 * P) (h3 : A / R - A / P = 4) :
  A = 683.45 :=
by
  sorry

end housewife_spending_l227_227241


namespace fruit_salad_cherries_l227_227908

variable (b r g c : ℕ)

theorem fruit_salad_cherries :
  (b + r + g + c = 350) ∧
  (r = 3 * b) ∧
  (g = 4 * c) ∧
  (c = 5 * r) →
  c = 66 :=
by
  sorry

end fruit_salad_cherries_l227_227908


namespace total_legs_correct_l227_227338

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end total_legs_correct_l227_227338


namespace final_price_of_coat_after_discounts_l227_227247

def original_price : ℝ := 120
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.20

theorem final_price_of_coat_after_discounts : 
    (1 - second_discount) * (1 - first_discount) * original_price = 72 := 
by
    sorry

end final_price_of_coat_after_discounts_l227_227247


namespace johns_cookies_left_l227_227504

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end johns_cookies_left_l227_227504


namespace prove_weight_of_a_l227_227616

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

end prove_weight_of_a_l227_227616


namespace mean_score_of_seniors_l227_227745

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

end mean_score_of_seniors_l227_227745


namespace infinite_solutions_l227_227451

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end infinite_solutions_l227_227451


namespace divides_b_n_minus_n_l227_227441

theorem divides_b_n_minus_n (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ n : ℕ, n > 0 ∧ a ∣ (b^n - n) :=
by
  sorry

end divides_b_n_minus_n_l227_227441


namespace probability_no_intersecting_chords_l227_227184

open Nat

def double_factorial (n : Nat) : Nat :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

def catalan_number (n : Nat) : Nat :=
  (factorial (2 * n)) / (factorial n * factorial (n + 1))

theorem probability_no_intersecting_chords (n : Nat) (h : n > 0) :
  (catalan_number n) / (double_factorial (2 * n - 1)) = 2^n / (factorial (n + 1)) :=
by
  sorry

end probability_no_intersecting_chords_l227_227184


namespace linear_regression_equation_demand_prediction_l227_227525

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

end linear_regression_equation_demand_prediction_l227_227525


namespace eccentricity_of_ellipse_l227_227938

theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : c = Real.sqrt (a^2 - b^2))
  (h4 : e = c / a) :
  e = 4 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l227_227938


namespace max_x_value_l227_227543

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

end max_x_value_l227_227543


namespace total_pieces_gum_is_correct_l227_227231

-- Define the number of packages and pieces per package
def packages : ℕ := 27
def pieces_per_package : ℕ := 18

-- Define the total number of pieces of gum Robin has
def total_pieces_gum : ℕ :=
  packages * pieces_per_package

-- State the theorem and proof obligation
theorem total_pieces_gum_is_correct : total_pieces_gum = 486 := by
  -- Proof omitted
  sorry

end total_pieces_gum_is_correct_l227_227231


namespace outfit_choices_l227_227693

theorem outfit_choices:
  let shirts := 8
  let pants := 8
  let hats := 8
  -- Each has 8 different colors
  -- No repetition of color within type of clothing
  -- Refuse to wear same color shirt and pants
  (shirts * pants * hats) - (shirts * hats) = 448 := 
sorry

end outfit_choices_l227_227693


namespace sin_add_pi_over_2_l227_227675

theorem sin_add_pi_over_2 (θ : ℝ) (h : Real.cos θ = -3 / 5) : Real.sin (θ + π / 2) = -3 / 5 :=
sorry

end sin_add_pi_over_2_l227_227675


namespace number_of_integers_inequality_l227_227050

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end number_of_integers_inequality_l227_227050


namespace surface_area_of_sphere_l227_227879

noncomputable def volume : ℝ := 72 * Real.pi

theorem surface_area_of_sphere (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = volume) :
  4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  sorry

end surface_area_of_sphere_l227_227879


namespace walmart_total_sales_l227_227544

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

end walmart_total_sales_l227_227544


namespace gcd_765432_654321_l227_227549

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227549


namespace scientific_notation_of_508_billion_yuan_l227_227209

-- Definition for a billion in the international system.
def billion : ℝ := 10^9

-- The amount of money given in the problem.
def amount_in_billion (n : ℝ) : ℝ := n * billion

-- The Lean theorem statement to prove.
theorem scientific_notation_of_508_billion_yuan :
  amount_in_billion 508 = 5.08 * 10^11 :=
by
  sorry

end scientific_notation_of_508_billion_yuan_l227_227209


namespace unique_zero_of_f_inequality_of_x1_x2_l227_227960

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l227_227960


namespace union_sets_l227_227142

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_sets : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_sets_l227_227142


namespace sum_of_surface_points_l227_227410

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l227_227410


namespace better_sequence_is_BAB_l227_227743

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

end better_sequence_is_BAB_l227_227743


namespace greatest_possible_value_of_a_l227_227021

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ x : ℤ, x * (x + a) = -24 → x * (x + a) = -24) ∧ (∀ b : ℕ, (∀ x : ℤ, x * (x + b) = -24 → x * (x + b) = -24) → b ≤ a) ∧ a = 25 :=
sorry

end greatest_possible_value_of_a_l227_227021


namespace mia_stops_in_quarter_C_l227_227344

def track_circumference : ℕ := 100 -- The circumference of the track in feet.
def total_distance_run : ℕ := 10560 -- The total distance Mia runs in feet.

-- Define the function to determine the quarter of the circle Mia stops in.
def quarter_mia_stops : ℕ :=
  let quarters := track_circumference / 4 -- Each quarter's length.
  let complete_laps := total_distance_run / track_circumference
  let remaining_distance := total_distance_run % track_circumference
  if remaining_distance < quarters then 1 -- Quarter A
  else if remaining_distance < 2 * quarters then 2 -- Quarter B
  else if remaining_distance < 3 * quarters then 3 -- Quarter C
  else 4 -- Quarter D

theorem mia_stops_in_quarter_C : quarter_mia_stops = 3 := by
  sorry

end mia_stops_in_quarter_C_l227_227344


namespace nails_to_buy_l227_227383

-- Define the initial number of nails Tom has
def initial_nails : ℝ := 247

-- Define the number of nails found in the toolshed
def toolshed_nails : ℝ := 144

-- Define the number of nails found in a drawer
def drawer_nails : ℝ := 0.5

-- Define the number of nails given by the neighbor
def neighbor_nails : ℝ := 58.75

-- Define the total number of nails needed for the project
def total_needed_nails : ℝ := 625.25

-- Define the total number of nails Tom already has
def total_existing_nails : ℝ := 
  initial_nails + toolshed_nails + drawer_nails + neighbor_nails

-- Prove that Tom needs to buy 175 more nails
theorem nails_to_buy :
  total_needed_nails - total_existing_nails = 175 := by
  sorry

end nails_to_buy_l227_227383


namespace geese_in_marsh_l227_227078

theorem geese_in_marsh (number_of_ducks : ℕ) (total_number_of_birds : ℕ) (number_of_geese : ℕ) (h1 : number_of_ducks = 37) (h2 : total_number_of_birds = 95) : 
  number_of_geese = 58 := 
by
  sorry

end geese_in_marsh_l227_227078


namespace find_intersection_A_B_find_range_t_l227_227297

-- Define sets A, B, C
def A : Set ℝ := {y | ∃ x, (1 ≤ x ∧ x ≤ 2) ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2 * t}

-- Theorem 1: Finding A ∩ B
theorem find_intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := 
by
  sorry

-- Theorem 2: If A ∩ C = C, find the range of values for t
theorem find_range_t (t : ℝ) (h : A ∩ C t = C t) : t ≤ 2 :=
by
  sorry

end find_intersection_A_B_find_range_t_l227_227297


namespace distance_from_point_A_l227_227764

theorem distance_from_point_A :
  ∀ (A : ℝ) (area : ℝ) (white_area : ℝ) (black_area : ℝ), area = 18 →
  (black_area = 2 * white_area) →
  A = (12 * Real.sqrt 2) / 5 := by
  intros A area white_area black_area h1 h2
  sorry

end distance_from_point_A_l227_227764


namespace valid_fraction_l227_227431

theorem valid_fraction (x: ℝ) : x^2 + 1 ≠ 0 :=
by
  sorry

end valid_fraction_l227_227431


namespace polynomial_problem_l227_227712

noncomputable def F (x : ℝ) : ℝ := sorry

theorem polynomial_problem
  (F : ℝ → ℝ)
  (h1 : F 4 = 22)
  (h2 : ∀ x : ℝ, (F (2 * x) / F (x + 2) = 4 - (16 * x + 8) / (x^2 + x + 1))) :
  F 8 = 1078 / 9 := sorry

end polynomial_problem_l227_227712


namespace find_second_derivative_at_1_l227_227128

-- Define the function f(x) and its second derivative
noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def f'' (x : ℝ) := (x + 2) * Real.exp x

-- State the theorem to be proved
theorem find_second_derivative_at_1 : f'' 1 = 2 * Real.exp 1 := by
  sorry

end find_second_derivative_at_1_l227_227128


namespace reciprocal_2023_l227_227366

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l227_227366


namespace range_of_m_l227_227134

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 0 < m) 
  (h4 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : m ≥ 4 :=
sorry

end range_of_m_l227_227134


namespace number_of_adults_l227_227015

-- Given constants
def children : ℕ := 200
def price_child (price_adult : ℕ) : ℕ := price_adult / 2
def total_amount : ℕ := 16000

-- Based on the problem conditions
def price_adult := 32

-- The generated proof problem
theorem number_of_adults 
    (price_adult_gt_0 : price_adult > 0)
    (h_price_adult : price_adult = 32)
    (h_total_amount : total_amount = 16000) 
    (h_price_relation : ∀ price_adult, price_adult / 2 * 2 = price_adult) :
  ∃ A : ℕ, 32 * A + 16 * 200 = 16000 ∧ price_child price_adult = 16 := by
  sorry

end number_of_adults_l227_227015


namespace min_x_plus_y_l227_227277

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l227_227277


namespace count_integers_between_bounds_l227_227060

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end count_integers_between_bounds_l227_227060


namespace sphere_surface_area_l227_227876

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l227_227876


namespace right_angled_triangle_l227_227894

-- Define the lengths of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove that these lengths form a right-angled triangle
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l227_227894


namespace gcd_765432_654321_l227_227563

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227563


namespace quadratic_no_solution_l227_227702

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_no_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  0 < a ∧ discriminant a b c ≤ 0 :=
by
  sorry

end quadratic_no_solution_l227_227702


namespace problem_proof_l227_227190

open Set

theorem problem_proof :
  let U : Set ℕ := {1, 2, 3, 4, 5, 6}
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5}
  P ∩ (U \ Q) = {1, 2} :=
by
  let U : Set ℕ := {1, 2, 3, 4, 5, 6}
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5}
  show P ∩ (U \ Q) = {1, 2}
  sorry

end problem_proof_l227_227190


namespace gcd_765432_654321_l227_227586

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227586


namespace circle_center_and_radius_l227_227747

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * y - 1 = 0 ↔ (x, y) = (0, 2) ∧ 5 = (0 - x)^2 + (2 - y)^2 :=
by sorry

end circle_center_and_radius_l227_227747


namespace exponent_of_5_in_30_factorial_l227_227174

-- Definition for the exponent of prime p in n!
def prime_factor_exponent (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else if p = 1 then 0
  else
    let rec compute (m acc : ℕ) : ℕ :=
      if m = 0 then acc else compute (m / p) (acc + m / p)
    in compute n 0

theorem exponent_of_5_in_30_factorial :
  prime_factor_exponent 5 30 = 7 :=
by {
  -- The proof is omitted.
  sorry
}

end exponent_of_5_in_30_factorial_l227_227174


namespace gcd_765432_654321_l227_227547

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227547


namespace chord_length_of_circle_l227_227524

theorem chord_length_of_circle (x y : ℝ) :
  (x^2 + y^2 - 4 * x - 4 * y - 1 = 0) ∧ (y = x + 2) → 
  2 * Real.sqrt 7 = 2 * Real.sqrt 7 :=
by sorry

end chord_length_of_circle_l227_227524


namespace x_minus_y_eq_11_l227_227306

theorem x_minus_y_eq_11 (x y : ℝ) (h : |x - 6| + |y + 5| = 0) : x - y = 11 := by
  sorry

end x_minus_y_eq_11_l227_227306


namespace middle_number_is_10_l227_227539

theorem middle_number_is_10 (x y z : ℤ) (hx : x < y) (hy : y < z) 
    (h1 : x + y = 18) (h2 : x + z = 25) (h3 : y + z = 27) : y = 10 :=
by 
  sorry

end middle_number_is_10_l227_227539


namespace andy_wrong_questions_l227_227771

/-- Andy, Beth, Charlie, and Daniel take a test. Andy and Beth together get the same number of 
    questions wrong as Charlie and Daniel together. Andy and Daniel together get four more 
    questions wrong than Beth and Charlie do together. Charlie gets five questions wrong. 
    Prove that Andy gets seven questions wrong. -/
theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 4) (h3 : c = 5) :
  a = 7 :=
by
  sorry

end andy_wrong_questions_l227_227771


namespace count_integers_in_interval_l227_227056

theorem count_integers_in_interval :
  ∃ (n : ℕ), (∀ x : ℤ, 25 > x ∧ x > 9 → 10 ≤ x ∧ x ≤ 24 → x ∈ (Finset.range (25 - 10 + 1)).map (λ i, i + 10)) ∧ n = (Finset.range (25 - 10 + 1)).card :=
sorry

end count_integers_in_interval_l227_227056


namespace select_more_stable_athlete_l227_227643

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

end select_more_stable_athlete_l227_227643


namespace problem_solution_l227_227255

theorem problem_solution :
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := 
  by
  sorry

end problem_solution_l227_227255


namespace triangle_area_is_96_l227_227757

-- Definitions of radii and sides being congruent
def tangent_circles (radius1 radius2 : ℝ) : Prop :=
  ∃ (O O' : ℝ × ℝ), dist O O' = radius1 + radius2

-- Given conditions
def radius_small : ℝ := 2
def radius_large : ℝ := 4
def sides_congruent (AB AC : ℝ) : Prop :=
  AB = AC

-- Theorem stating the goal
theorem triangle_area_is_96 
  (O O' : ℝ × ℝ)
  (AB AC : ℝ)
  (circ_tangent : tangent_circles radius_small radius_large)
  (sides_tangent : sides_congruent AB AC) :
  ∃ (BC : ℝ), ∃ (AF : ℝ), (1/2) * BC * AF = 96 := 
by
  sorry

end triangle_area_is_96_l227_227757


namespace max_surface_area_of_rectangular_solid_l227_227131

theorem max_surface_area_of_rectangular_solid {r a b c : ℝ} (h_sphere : 4 * π * r^2 = 4 * π)
  (h_diagonal : a^2 + b^2 + c^2 = (2 * r)^2) :
  2 * (a * b + a * c + b * c) ≤ 8 :=
by
  sorry

end max_surface_area_of_rectangular_solid_l227_227131


namespace tom_sold_4_books_l227_227220

-- Definitions based on conditions from the problem
def initial_books : ℕ := 5
def new_books : ℕ := 38
def final_books : ℕ := 39

-- The number of books Tom sold
def books_sold (S : ℕ) : Prop := initial_books - S + new_books = final_books

-- Our goal is to prove that Tom sold 4 books
theorem tom_sold_4_books : books_sold 4 :=
  by
    -- Implicitly here would be the proof, but we use sorry to skip it
    sorry

end tom_sold_4_books_l227_227220


namespace performance_arrangement_l227_227242

def num_ways_to_arrange_performances : ℕ :=
  let perms := finEnum.enumerate (Fin 6) -- Total arrangements of 6 performances
  perms.filter (λ perm,
    perm.head ≠ skit # and
    (0 until 4).all (λ i, perm.nth (i + 1) ≠ perm.nth i + 1) # all adjacent checks 
  ).size 

theorem performance_arrangement (first_not_skit : ∀ (arr : List ℕ), arr.head ≠ 5)
 (no_adj_sings : ∀ (arr : List ℕ), ∀ (i : ℕ), i < arr.length - 1 → arr[i]! > 1 ∨ arr[i + 1]! < 1 ∨ arr[i]! < 3 ∨ arr[i + 1]! > 2)
: num_ways_to_arrange_performances = 408 :=
sorry

end performance_arrangement_l227_227242


namespace general_formula_arithmetic_sequence_l227_227030

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

theorem general_formula_arithmetic_sequence (x : ℝ) (a : ℕ → ℝ) 
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n : ℕ, (a n = 2 * n - 4) ∨ (a n = 4 - 2 * n) :=
by
  sorry

end general_formula_arithmetic_sequence_l227_227030


namespace div_by_5_l227_227082

theorem div_by_5 (a b : ℕ) (h: 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  -- Proof by contradiction
  -- Assume the negation of the conclusion
  have h_nand : ¬ (5 ∣ a) ∧ ¬ (5 ∣ b) := sorry

  -- Derive a contradiction based on the assumptions
  sorry

end div_by_5_l227_227082


namespace domain_of_f_l227_227259

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3 * x + 2)

theorem domain_of_f :
  {x : ℝ | (x < 1) ∨ (1 < x ∧ x < 2) ∨ (x > 2)} = 
  {x : ℝ | f x ≠ 0} :=
sorry

end domain_of_f_l227_227259


namespace calories_consumed_l227_227325

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l227_227325


namespace integer_values_count_l227_227074

theorem integer_values_count (x : ℕ) : (∃ y : ℤ, 10 ≤ y ∧ y ≤ 24) ↔ (∑ y in (finset.interval 10 24), 1) = 15 :=
by
  sorry

end integer_values_count_l227_227074


namespace parallel_condition_sufficient_not_necessary_l227_227299

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x + 1, 3)

theorem parallel_condition_sufficient_not_necessary (x : ℝ) :
  (x = 2) → (a x = b x) ∨ (a (-2) = b (-2)) :=
by sorry

end parallel_condition_sufficient_not_necessary_l227_227299


namespace find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l227_227123

def isOdd (n : ℕ) : Prop := n % 2 = 1
def isInRange (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 50
def hasRemainderTwo (n : ℕ) : Prop := n % 7 = 2

theorem find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7 :
  ∃ n : ℕ, isInRange n ∧ isOdd n ∧ hasRemainderTwo n ∧ n = 37 :=
by
  sorry

end find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l227_227123


namespace total_cost_nancy_spends_l227_227625

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

end total_cost_nancy_spends_l227_227625


namespace sum_of_surface_points_l227_227408

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l227_227408


namespace one_percent_as_decimal_l227_227613

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := 
by 
  sorry

end one_percent_as_decimal_l227_227613


namespace labor_union_tree_equation_l227_227234

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end labor_union_tree_equation_l227_227234


namespace smallest_x_value_l227_227943

theorem smallest_x_value :
  ∃ x, (x ≠ 9) ∧ (∀ y, (y ≠ 9) → ((x^2 - x - 72) / (x - 9) = 3 / (x + 6)) → x ≤ y) ∧ x = -9 :=
by
  sorry

end smallest_x_value_l227_227943


namespace eq1_solution_eq2_solution_l227_227864

theorem eq1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 :=
by
  sorry

theorem eq2_solution (x : ℝ) (h : (1 / 2) * x - 6 = (3 / 4) * x) : x = -24 :=
by
  sorry

end eq1_solution_eq2_solution_l227_227864


namespace min_value_frac_eq_nine_halves_l227_227683

theorem min_value_frac_eq_nine_halves {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 2) :
  ∃ (x y : ℝ), 2 / x + 1 / y = 9 / 2 := by
  sorry

end min_value_frac_eq_nine_halves_l227_227683


namespace range_of_f_l227_227269

noncomputable def f (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 1)

theorem range_of_f : Set.Icc (-1/4 : ℝ) (1/4) = Set.range f :=
by
  sorry

end range_of_f_l227_227269


namespace JungMinBoughtWire_l227_227981

theorem JungMinBoughtWire
  (side_length : ℕ)
  (number_of_sides : ℕ)
  (remaining_wire : ℕ)
  (total_wire_bought : ℕ)
  (h1 : side_length = 13)
  (h2 : number_of_sides = 5)
  (h3 : remaining_wire = 8)
  (h4 : total_wire_bought = side_length * number_of_sides + remaining_wire) :
    total_wire_bought = 73 :=
by {
  sorry
}

end JungMinBoughtWire_l227_227981


namespace area_triangle_AEB_l227_227836

theorem area_triangle_AEB :
  ∀ (A B C D F G E : Type)
    (AB AD BC CD : ℝ) 
    (AF BG : ℝ) 
    (triangle_AEB : ℝ),
  (AB = 7) →
  (BC = 4) →
  (CD = 7) →
  (AD = 4) →
  (DF = 2) →
  (GC = 1) →
  (triangle_AEB = 1/2 * 7 * (4 + 16/3)) →
  (triangle_AEB = 98 / 3) :=
by
  intros A B C D F G E AB AD BC CD AF BG triangle_AEB
  sorry

end area_triangle_AEB_l227_227836


namespace unique_zero_f_x1_minus_2x2_l227_227964

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l227_227964


namespace Alice_more_nickels_l227_227250

-- Define quarters each person has
def Alice_quarters (q : ℕ) : ℕ := 10 * q + 2
def Bob_quarters (q : ℕ) : ℕ := 2 * q + 10

-- Prove that Alice has 40(q - 1) more nickels than Bob
theorem Alice_more_nickels (q : ℕ) : 
  (5 * (Alice_quarters q - Bob_quarters q)) = 40 * (q - 1) :=
by
  sorry

end Alice_more_nickels_l227_227250


namespace probability_of_selecting_one_second_class_product_l227_227163

def total_products : ℕ := 100
def first_class_products : ℕ := 90
def second_class_products : ℕ := 10
def selected_products : ℕ := 3
def exactly_one_second_class_probability : ℚ :=
  (Nat.choose first_class_products 2 * Nat.choose second_class_products 1) / Nat.choose total_products selected_products

theorem probability_of_selecting_one_second_class_product :
  exactly_one_second_class_probability = 0.25 := 
  sorry

end probability_of_selecting_one_second_class_product_l227_227163


namespace gcd_765432_654321_l227_227562

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227562


namespace sqrt7_sub_m_div_n_gt_inv_mn_l227_227189

variables (m n : ℤ)
variables (h_m_nonneg : m ≥ 1) (h_n_nonneg : n ≥ 1)
variables (h_ineq : Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 0)

theorem sqrt7_sub_m_div_n_gt_inv_mn : 
  Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 1 / ((m : ℝ) * (n : ℝ)) :=
by
  sorry

end sqrt7_sub_m_div_n_gt_inv_mn_l227_227189


namespace gcd_of_765432_and_654321_l227_227583

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227583


namespace range_of_a_l227_227852

noncomputable def f (x a : ℝ) := Real.exp (-x) - 2 * x - a

def curve (x : ℝ) := x ^ 3 + x

def y_in_range (x : ℝ) := x >= -2 ∧ x <= 2

theorem range_of_a : ∀ (a : ℝ), (∃ x, y_in_range (curve x) ∧ f (curve x) a = curve x) ↔ a ∈ Set.Icc (Real.exp (-2) - 6) (Real.exp 2 + 6) := by
  sorry

end range_of_a_l227_227852


namespace polynomial_roots_problem_l227_227540

theorem polynomial_roots_problem (a b c d e : ℝ) (h1 : a ≠ 0) 
    (h2 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
    (h3 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
    (h4 : a + b + c + d + e = 0) :
    (b + c + d) / a = -7 := 
sorry

end polynomial_roots_problem_l227_227540


namespace Shara_borrowed_6_months_ago_l227_227012

theorem Shara_borrowed_6_months_ago (X : ℝ) (h1 : ∃ n : ℕ, (X / 2 - 4 * 10 = 20) ∧ (X / 2 = n * 10)) :
  ∃ m : ℕ, m * 10 = X / 2 → m = 6 := 
sorry

end Shara_borrowed_6_months_ago_l227_227012


namespace x_coordinate_D_l227_227717

noncomputable def find_x_coordinate_D (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ := 
  let l := -a * b
  let x := l / c
  x

theorem x_coordinate_D (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (D_on_parabola : d^2 = (a + b) * (d) + l)
  (lines_intersect_y_axis : ∃ l : ℝ, (a^2 = (b + a) * a + l) ∧ (b^2 = (b + a) * b + l) ∧ (c^2 = (d + c) * c + l)) :
  d = (a * b) / c :=
by sorry

end x_coordinate_D_l227_227717


namespace sample_and_size_correct_l227_227098

structure SchoolSurvey :=
  (students_selected : ℕ)
  (classes_selected : ℕ)

def survey_sample (survey : SchoolSurvey) : String :=
  "the physical condition of " ++ toString survey.students_selected ++ " students"

def survey_sample_size (survey : SchoolSurvey) : ℕ :=
  survey.students_selected

theorem sample_and_size_correct (survey : SchoolSurvey)
  (h_selected : survey.students_selected = 190)
  (h_classes : survey.classes_selected = 19) :
  survey_sample survey = "the physical condition of 190 students" ∧ 
  survey_sample_size survey = 190 :=
by
  sorry

end sample_and_size_correct_l227_227098


namespace tangent_line_eq_l227_227445

theorem tangent_line_eq {f : ℝ → ℝ} (hf : ∀ x, f x = x - 2 * Real.log x) :
  ∃ m b, (m = -1) ∧ (b = 2) ∧ (∀ x, f x = m * x + b) :=
by
  sorry

end tangent_line_eq_l227_227445


namespace common_root_iff_cond_l227_227009

theorem common_root_iff_cond (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2) * (p1 * q2 - q1 * p2) = 0 :=
by
  sorry

end common_root_iff_cond_l227_227009


namespace find_x_for_equation_l227_227739

theorem find_x_for_equation 
  (x : ℝ)
  (h : (32 : ℝ)^(x-2) / (8 : ℝ)^(x-2) = (512 : ℝ)^(3 * x)) : 
  x = -4/25 :=
by
  sorry

end find_x_for_equation_l227_227739


namespace first_year_after_2020_with_digit_sum_18_l227_227748

theorem first_year_after_2020_with_digit_sum_18 : 
  ∃ (y : ℕ), y > 2020 ∧ (∃ a b c : ℕ, (2 + a + b + c = 18 ∧ y = 2000 + 100 * a + 10 * b + c)) ∧ y = 2799 := 
sorry

end first_year_after_2020_with_digit_sum_18_l227_227748


namespace john_toy_store_fraction_l227_227811

theorem john_toy_store_fraction
  (allowance : ℝ)
  (spent_at_arcade_fraction : ℝ)
  (remaining_allowance : ℝ)
  (spent_at_candy_store : ℝ)
  (spent_at_toy_store : ℝ)
  (john_allowance : allowance = 3.60)
  (arcade_fraction : spent_at_arcade_fraction = 3 / 5)
  (arcade_amount : remaining_allowance = allowance - (spent_at_arcade_fraction * allowance))
  (candy_store_amount : spent_at_candy_store = 0.96)
  (remaining_after_candy_store : spent_at_toy_store = remaining_allowance - spent_at_candy_store)
  : spent_at_toy_store / remaining_allowance = 1 / 3 :=
by
  sorry

end john_toy_store_fraction_l227_227811


namespace hyperbola_focus_y_axis_l227_227155

theorem hyperbola_focus_y_axis (m : ℝ) :
  (∀ x y : ℝ, (m + 1) * x^2 + (2 - m) * y^2 = 1) → m < -1 :=
sorry

end hyperbola_focus_y_axis_l227_227155


namespace scientific_notation_of_508_billion_l227_227210

theorem scientific_notation_of_508_billion:
  (508 * (10:ℝ)^9) = (5.08 * (10:ℝ)^11) := 
begin
  sorry
end

end scientific_notation_of_508_billion_l227_227210


namespace smallest_possible_AC_l227_227926

theorem smallest_possible_AC 
    (AB AC CD : ℤ) 
    (BD_squared : ℕ) 
    (h_isosceles : AB = AC)
    (h_point_D : ∃ D : ℤ, D = CD)
    (h_perpendicular : BD_squared = 85) 
    (h_integers : ∃ x y : ℤ, AC = x ∧ CD = y) 
    : AC = 11 :=
by
  sorry

end smallest_possible_AC_l227_227926


namespace no_int_coords_equilateral_l227_227110

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end no_int_coords_equilateral_l227_227110


namespace deck_card_count_l227_227165

theorem deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 :=
by
  sorry

end deck_card_count_l227_227165


namespace incircle_hexagon_area_ratio_l227_227644

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

end incircle_hexagon_area_ratio_l227_227644


namespace smarties_division_l227_227119

theorem smarties_division (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end smarties_division_l227_227119


namespace magnitude_relationship_l227_227776

noncomputable def a : ℝ := 2 ^ 0.3
def b : ℝ := 0.3 ^ 2
noncomputable def c : ℝ := log 2 0.3

theorem magnitude_relationship : c < b ∧ b < a := by
  sorry

end magnitude_relationship_l227_227776


namespace total_books_after_loss_l227_227722

-- Define variables for the problem
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

-- Prove the final number of books together
theorem total_books_after_loss : (sandy_books + tim_books - benny_lost_books) = 19 := by
  sorry

end total_books_after_loss_l227_227722


namespace order_of_trig_values_l227_227188

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem order_of_trig_values : b < a ∧ a < d ∧ d < c :=
by
  sorry

end order_of_trig_values_l227_227188


namespace time_to_pass_platform_l227_227088

-- Definitions for the given conditions
def train_length := 1200 -- length of the train in meters
def tree_crossing_time := 120 -- time taken to cross a tree in seconds
def platform_length := 1200 -- length of the platform in meters

-- Calculation of speed of the train and distance to be covered
def train_speed := train_length / tree_crossing_time -- speed in meters per second
def total_distance_to_cover := train_length + platform_length -- total distance in meters

-- Proof statement that given the above conditions, the time to pass the platform is 240 seconds
theorem time_to_pass_platform : 
  total_distance_to_cover / train_speed = 240 :=
  by sorry

end time_to_pass_platform_l227_227088


namespace prime_cube_solution_l227_227935

theorem prime_cube_solution (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : p^3 = p^2 + q^2 + r^2) : 
  p = 3 ∧ q = 3 ∧ r = 3 :=
by
  sorry

end prime_cube_solution_l227_227935


namespace gcd_765432_654321_l227_227596

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227596


namespace sum_of_surface_points_l227_227409

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l227_227409


namespace find_a_and_b_l227_227804

theorem find_a_and_b (a b m : ℝ) 
  (h1 : (3 * a - 5)^(1 / 3) = -2)
  (h2 : ∀ x, x^2 = b → x = m ∨ x = 1 - 5 * m) : 
  a = -1 ∧ b = 1 / 16 :=
by
  sorry  -- proof to be constructed

end find_a_and_b_l227_227804


namespace number_of_integers_satisfying_sqrt_condition_l227_227043

noncomputable def count_integers_satisfying_sqrt_condition : ℕ :=
  let S := {x : ℕ | 3 < real.sqrt x ∧ real.sqrt x < 5}
  finset.card (finset.filter (λ x, 3 < real.sqrt x ∧ real.sqrt x < 5) (finset.range 26))

theorem number_of_integers_satisfying_sqrt_condition :
  count_integers_satisfying_sqrt_condition = 15 :=
sorry

end number_of_integers_satisfying_sqrt_condition_l227_227043


namespace exponent_of_five_in_factorial_l227_227179

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l227_227179


namespace problem_proof_l227_227262

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

noncomputable def probability_ratio_pq : ℕ :=
let p := binomial 10 2 * binomial 30 2 * binomial 28 2
let q := binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3
p / (q / (binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3))

theorem problem_proof :
  probability_ratio_pq = 7371 :=
sorry

end problem_proof_l227_227262


namespace g_of_neg5_eq_651_over_16_l227_227987

def f (x : ℝ) : ℝ := 4 * x + 6

def g (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 7

theorem g_of_neg5_eq_651_over_16 : g (-5) = 651 / 16 := by
  sorry

end g_of_neg5_eq_651_over_16_l227_227987


namespace find_x_l227_227267

variable (c d : ℝ)

theorem find_x (x : ℝ) (h : x^2 + 4 * c^2 = (3 * d - x)^2) : 
  x = (9 * d^2 - 4 * c^2) / (6 * d) :=
sorry

end find_x_l227_227267


namespace find_f_zero_l227_227735

theorem find_f_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y - x * y) 
  (h1 : f 1 = 1) : 
  f 0 = 0 := 
sorry

end find_f_zero_l227_227735


namespace age_problem_l227_227971

variable (A B x : ℕ)

theorem age_problem (h1 : A = B + 5) (h2 : B = 35) (h3 : A + x = 2 * (B - x)) : x = 10 :=
sorry

end age_problem_l227_227971


namespace parallel_lines_perpendicular_lines_l227_227689

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

end parallel_lines_perpendicular_lines_l227_227689


namespace azalea_paid_shearer_l227_227430

noncomputable def amount_paid_to_shearer (number_of_sheep wool_per_sheep price_per_pound profit : ℕ) : ℕ :=
  let total_wool := number_of_sheep * wool_per_sheep
  let total_revenue := total_wool * price_per_pound
  total_revenue - profit

theorem azalea_paid_shearer :
  let number_of_sheep := 200
  let wool_per_sheep := 10
  let price_per_pound := 20
  let profit := 38000
  amount_paid_to_shearer number_of_sheep wool_per_sheep price_per_pound profit = 2000 := 
by
  sorry

end azalea_paid_shearer_l227_227430


namespace gcd_765432_654321_l227_227552

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227552


namespace trees_planted_l227_227382

-- Definitions for the quantities of lindens (x) and birches (y)
variables (x y : ℕ)

-- Definitions matching the given problem conditions
def condition1 := x + y > 14
def condition2 := y + 18 > 2 * x
def condition3 := x > 2 * y

-- The theorem stating that if the conditions hold, then x = 11 and y = 5
theorem trees_planted (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : 
  x = 11 ∧ y = 5 := 
sorry

end trees_planted_l227_227382


namespace gcd_765432_654321_l227_227564

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227564


namespace factor_expression_l227_227925

theorem factor_expression (x : ℝ) : 
  (21 * x ^ 4 + 90 * x ^ 3 + 40 * x - 10) - (7 * x ^ 4 + 6 * x ^ 3 + 8 * x - 6) = 
  2 * x * (7 * x ^ 3 + 42 * x ^ 2 + 16) - 4 :=
by sorry

end factor_expression_l227_227925


namespace first_and_second_bags_l227_227882

def bags_apples (A B C : ℕ) : Prop :=
  (A + B + C = 24) ∧ (B + C = 18) ∧ (A + C = 19)

theorem first_and_second_bags (A B C : ℕ) (h : bags_apples A B C) :
  A + B = 11 :=
sorry

end first_and_second_bags_l227_227882


namespace find_a_evaluate_expr_l227_227802

-- Given polynomials A and B
def A (a x y : ℝ) : ℝ := a * x^2 + 3 * x * y + 2 * |a| * x
def B (x y : ℝ) : ℝ := 2 * x^2 + 6 * x * y + 4 * x + y + 1

-- Statement part (1)
theorem find_a (a : ℝ) (x y : ℝ) (h : (2 * A a x y - B x y) = (2 * a - 2) * x^2 + (4 * |a| - 4) * x - y - 1) : a = -1 := 
  sorry

-- Expression for part (2)
def expr (a : ℝ) : ℝ := 3 * (-3 * a^2 - 2 * a) - (a^2 - 2 * (5 * a - 4 * a^2 + 1) - 2 * a)

-- Statement part (2)
theorem evaluate_expr : expr (-1) = -22 := 
  sorry

end find_a_evaluate_expr_l227_227802


namespace gcd_proof_l227_227555

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227555


namespace annual_growth_rate_proof_l227_227628

-- Lean 4 statement for the given problem
theorem annual_growth_rate_proof (profit_2021 : ℝ) (profit_2023 : ℝ) (r : ℝ)
  (h1 : profit_2021 = 3000)
  (h2 : profit_2023 = 4320)
  (h3 : profit_2023 = profit_2021 * (1 + r) ^ 2) :
  r = 0.2 :=
by sorry

end annual_growth_rate_proof_l227_227628


namespace solution_l227_227249

noncomputable def problem : Prop :=
  let num_apprentices := 200
  let num_junior := 20
  let num_intermediate := 60
  let num_senior := 60
  let num_technician := 40
  let num_senior_technician := 20
  let total_technician := num_technician + num_senior_technician
  let sampling_ratio := 10 / num_apprentices
  
  -- Number of technicians (including both technician and senior technicians) in the exchange group
  let num_technicians_selected := total_technician * sampling_ratio

  -- Probability Distribution of X
  let P_X_0 := 7 / 24
  let P_X_1 := 21 / 40
  let P_X_2 := 7 / 40
  let P_X_3 := 1 / 120

  -- Expected value of X
  let E_X := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2) + (3 * P_X_3)
  E_X = 9 / 10

theorem solution : problem :=
  sorry

end solution_l227_227249


namespace cost_of_3000_pencils_l227_227627

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

end cost_of_3000_pencils_l227_227627


namespace area_ratio_of_squares_l227_227869

-- Definition of squares, and their perimeters' relationship
def perimeter (side_length : ℝ) := 4 * side_length

theorem area_ratio_of_squares (a b : ℝ) (h : perimeter a = 4 * perimeter b) : (a * a) = 16 * (b * b) :=
by
  -- We assume the given condition
  have ha : a = 4 * b := sorry
  -- We then prove the area ratio
  sorry

end area_ratio_of_squares_l227_227869


namespace find_equation_of_line_l227_227954

theorem find_equation_of_line
  (midpoint : ℝ × ℝ)
  (ellipse : ℝ → ℝ → Prop)
  (l_eq : ℝ → ℝ → Prop)
  (H_mid : midpoint = (1, 2))
  (H_ellipse : ∀ (x y : ℝ), ellipse x y ↔ x^2 / 64 + y^2 / 16 = 1)
  (H_line : ∀ (x y : ℝ), l_eq x y ↔ y - 2 = - (1/8) * (x - 1))
  : ∃ (a b c : ℝ), (a, b, c) = (1, 8, -17) ∧ (∀ (x y : ℝ), l_eq x y ↔ a * x + b * y + c = 0) :=
by 
  sorry

end find_equation_of_line_l227_227954


namespace number_of_desired_numbers_l227_227150

-- Define a predicate for a four-digit number with the thousands digit 3
def isDesiredNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n / 1000) % 10 = 3

-- Statement of the theorem
theorem number_of_desired_numbers : 
  ∃ k, k = 1000 ∧ (∀ n, isDesiredNumber n ↔ 3000 ≤ n ∧ n < 4000) := 
by
  -- Proof omitted, using sorry to skip the proof
  sorry

end number_of_desired_numbers_l227_227150


namespace expression_not_equal_33_l227_227999

theorem expression_not_equal_33 (x y : ℤ) :
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_not_equal_33_l227_227999


namespace infinite_solutions_b_l227_227453

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end infinite_solutions_b_l227_227453


namespace greatest_fourth_term_arith_seq_sum_90_l227_227076

theorem greatest_fourth_term_arith_seq_sum_90 :
  ∃ a d : ℕ, 6 * a + 15 * d = 90 ∧ (∀ n : ℕ, n < 6 → a + n * d > 0) ∧ (a + 3 * d = 17) :=
by
  sorry

end greatest_fourth_term_arith_seq_sum_90_l227_227076


namespace gcd_765432_654321_l227_227551

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227551


namespace distinct_even_numbers_between_100_and_999_l227_227478

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end distinct_even_numbers_between_100_and_999_l227_227478


namespace smallest_possible_sum_l227_227281

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l227_227281


namespace complex_z_power_l227_227139

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_z_power_l227_227139


namespace sequence_inequality_l227_227763

theorem sequence_inequality (a : ℕ → ℕ)
  (h1 : a 0 > 0) -- Ensure all entries are positive integers.
  (h2 : ∀ k l m n : ℕ, k * l = m * n → a k + a l = a m + a n)
  {p q : ℕ} (hpq : p ∣ q) :
  a p ≤ a q :=
sorry

end sequence_inequality_l227_227763


namespace tom_speed_from_A_to_B_l227_227542

theorem tom_speed_from_A_to_B (D S : ℝ) (h1 : 2 * D = S * (3 * D / 36 - D / 20))
  (h2 : S * (3 * D / 36 - D / 20) = 3 * D / 36 ∨ 3 * D / 36 = S * (3 * D / 36 - D / 20))
  (h3 : D > 0) : S = 60 :=
by { sorry }

end tom_speed_from_A_to_B_l227_227542


namespace exponent_of_5_in_30_factorial_l227_227168

theorem exponent_of_5_in_30_factorial :
  (nat.factorial 30).factors.count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l227_227168


namespace gcd_proof_l227_227558

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227558


namespace factorize_poly_l227_227778

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorize_poly_l227_227778


namespace find_n_l227_227688

theorem find_n (n : ℕ) (k : ℕ) (x : ℝ) (h1 : k = 1) (h2 : x = 180 - 360 / n) (h3 : 1.5 * x = 180 - 360 / (n + 1)) :
    n = 3 :=
by
  -- proof steps will be provided here
  sorry

end find_n_l227_227688


namespace ball_distribution_l227_227381

theorem ball_distribution :
  let white_combinations : ℕ := Nat.choose 5 2,
      red_combinations : ℕ := Nat.choose 6 2,
      yellow_combinations : ℕ := Nat.choose 7 2
  in white_combinations * red_combinations * yellow_combinations = 3150 := by
  sorry

end ball_distribution_l227_227381


namespace cups_of_ketchup_l227_227977

-- Define variables and conditions
variables (k : ℕ)
def vinegar : ℕ := 1
def honey : ℕ := 1
def sauce_per_burger : ℚ := 1 / 4
def sauce_per_pulled_pork : ℚ := 1 / 6
def burgers : ℕ := 8
def pulled_pork_sandwiches : ℕ := 18

-- Main theorem statement
theorem cups_of_ketchup (h : 8 * sauce_per_burger + 18 * sauce_per_pulled_pork = k + vinegar + honey) : k = 3 :=
  by
    sorry

end cups_of_ketchup_l227_227977


namespace condition_needs_l227_227900

theorem condition_needs (a b c d : ℝ) :
  a + c > b + d → (¬ (a > b ∧ c > d) ∧ (a > b ∧ c > d)) :=
by
  sorry

end condition_needs_l227_227900


namespace complex_z_1000_l227_227141

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end complex_z_1000_l227_227141


namespace fair_game_stakes_ratio_l227_227218

theorem fair_game_stakes_ratio (n : ℕ) (deck_size : ℕ) (player_count : ℕ)
  (L : ℕ → ℝ) : 
  deck_size = 36 → player_count = 36 → 
  (∀ k : ℕ, k < player_count - 1 → 
    (L (k + 1)) / (L k) = 35 / 36) :=
by
  intros h_deck_size h_player_count k hk
  simp [h_deck_size, h_player_count, hk]
  sorry

end fair_game_stakes_ratio_l227_227218


namespace gcd_of_765432_and_654321_l227_227582

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227582


namespace simone_fraction_per_day_l227_227723

theorem simone_fraction_per_day 
  (x : ℚ) -- Define the fraction of an apple Simone ate each day as x.
  (h1 : 16 * x + 15 * (1/3) = 13) -- Condition: Simone and Lauri together ate 13 apples.
  : x = 1/2 := 
 by 
  sorry

end simone_fraction_per_day_l227_227723


namespace number_of_integers_satisfying_sqrt_condition_l227_227041

noncomputable def count_integers_satisfying_sqrt_condition : ℕ :=
  let S := {x : ℕ | 3 < real.sqrt x ∧ real.sqrt x < 5}
  finset.card (finset.filter (λ x, 3 < real.sqrt x ∧ real.sqrt x < 5) (finset.range 26))

theorem number_of_integers_satisfying_sqrt_condition :
  count_integers_satisfying_sqrt_condition = 15 :=
sorry

end number_of_integers_satisfying_sqrt_condition_l227_227041


namespace dice_surface_sum_l227_227406

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l227_227406


namespace sum_even_numbers_from_2_to_60_l227_227254

noncomputable def sum_even_numbers_seq : ℕ :=
  let a₁ := 2
  let d := 2
  let aₙ := 60
  let n := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

theorem sum_even_numbers_from_2_to_60:
  sum_even_numbers_seq = 930 :=
by
  sorry

end sum_even_numbers_from_2_to_60_l227_227254


namespace total_cost_of_selling_watermelons_l227_227765

-- Definitions of the conditions:
def watermelon_weight : ℝ := 23.0
def daily_prices : List ℝ := [2.10, 1.90, 1.80, 2.30, 2.00, 1.95, 2.20]
def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def number_of_watermelons : ℕ := 18

-- The theorem statement:
theorem total_cost_of_selling_watermelons :
  let average_price := (daily_prices.sum / daily_prices.length)
  let total_weight := number_of_watermelons * watermelon_weight
  let initial_cost := total_weight * average_price
  let discounted_cost := if number_of_watermelons > discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  final_cost = 796.43 := by
    sorry

end total_cost_of_selling_watermelons_l227_227765


namespace count_integers_between_bounds_l227_227063

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end count_integers_between_bounds_l227_227063


namespace keith_cards_initial_count_l227_227507

theorem keith_cards_initial_count :
  ∃ (x : ℕ), let final_count := 46 in
  let cards_after_dog := 2 * final_count in
  let total_after_buying := cards_after_dog - 8 in
  (x = total_after_buying) ∧ (total_after_buying = 84) :=
begin
  sorry
end

end keith_cards_initial_count_l227_227507


namespace ab_divisibility_l227_227667

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end ab_divisibility_l227_227667


namespace gcd_of_765432_and_654321_l227_227579

open Nat

theorem gcd_of_765432_and_654321 : gcd 765432 654321 = 111111 :=
  sorry

end gcd_of_765432_and_654321_l227_227579


namespace triangle_area_l227_227295

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x^2 - sqrt 3

theorem triangle_area
  (A : ℝ) (b c : ℝ)
  (h1 : f A = 1)
  (h2 : b * c = 2) 
  (h3 : (b * cos A) * (c * cos A) = sqrt 2) : 
  (1 / 2 * b * c * sin A = sqrt 2 / 2) := 
sorry

end triangle_area_l227_227295


namespace problem_statement_l227_227932

def a : ℤ := 2020
def b : ℤ := 2022

theorem problem_statement : b^3 - a * b^2 - a^2 * b + a^3 = 16168 := by
  sorry

end problem_statement_l227_227932


namespace spending_50_dollars_opposite_meaning_l227_227974

theorem spending_50_dollars_opposite_meaning :
  (∀ (income expenditure : Int), income = 80 → expenditure = 50 → -income = - (expenditure)) :=
by
  intro income expenditure h_income h_expenditure
  rw [h_income, h_expenditure]
  rfl

end spending_50_dollars_opposite_meaning_l227_227974


namespace possible_ratios_of_distances_l227_227800

theorem possible_ratios_of_distances (a b : ℝ) (h : a > b) (h1 : ∃ points : Fin 4 → ℝ × ℝ, 
  ∀ (i j : Fin 4), i ≠ j → 
  (dist (points i) (points j) = a ∨ dist (points i) (points j) = b )) :
  a / b = Real.sqrt 2 ∨ 
  a / b = (1 + Real.sqrt 5) / 2 ∨ 
  a / b = Real.sqrt 3 ∨ 
  a / b = Real.sqrt (2 + Real.sqrt 3) :=
by 
  sorry

end possible_ratios_of_distances_l227_227800


namespace five_times_seven_divided_by_ten_l227_227610

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end five_times_seven_divided_by_ten_l227_227610


namespace root_equation_satisfies_expr_l227_227135

theorem root_equation_satisfies_expr (a : ℝ) (h : 2 * a ^ 2 - 7 * a - 1 = 0) :
  a * (2 * a - 7) + 5 = 6 :=
by
  sorry

end root_equation_satisfies_expr_l227_227135


namespace max_value_of_f_l227_227645

noncomputable def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

theorem max_value_of_f : ∃ x : ℝ, f x = 12.5 :=
by
  sorry

end max_value_of_f_l227_227645


namespace percentage_third_day_l227_227004

def initial_pieces : ℕ := 1000
def percentage_first_day : ℝ := 0.10
def percentage_second_day : ℝ := 0.20
def pieces_left_after_third_day : ℕ := 504

theorem percentage_third_day :
  let pieces_first_day := initial_pieces * percentage_first_day
  let remaining_after_first_day := initial_pieces - pieces_first_day
  let pieces_second_day := remaining_after_first_day * percentage_second_day
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day
  let pieces_third_day := remaining_after_second_day - pieces_left_after_third_day
  (pieces_third_day / remaining_after_second_day * 100 = 30) :=
by
  sorry

end percentage_third_day_l227_227004


namespace exponent_of_5_in_30_fact_l227_227173

theorem exponent_of_5_in_30_fact : nat.factorial 30 = ((5^7) * (nat.factorial (30/5))) := by
  sorry

end exponent_of_5_in_30_fact_l227_227173


namespace rona_age_l227_227861

theorem rona_age (R : ℕ) (hR1 : ∀ Rachel Collete : ℕ, Rachel = 2 * R ∧ Collete = R / 2 ∧ Rachel - Collete = 12) : R = 12 :=
sorry

end rona_age_l227_227861


namespace fair_betting_scheme_fair_game_l227_227216

noncomputable def fair_game_stakes (L L_k : ℕ → ℚ) : Prop :=
  ∀ k: ℕ, 1 < k → L_k (k+1) = (35/36) * L_k k

theorem fair_betting_scheme_fair_game :
  ∃ L_k : ℕ → ℚ, fair_game_stakes (λ k, L_k k) (λ k, L_k k) :=
begin
  let L_k : ℕ → ℚ := λ k, (35 / 36) ^ k,
  use L_k,
  unfold fair_game_stakes,
  intros k hk,
  rw [mul_assoc, ←mul_pow],
  ring,
end

end fair_betting_scheme_fair_game_l227_227216


namespace regular_price_correct_l227_227245

noncomputable def regular_price_of_one_tire (x : ℝ) : Prop :=
  3 * x + 5 - 10 = 302

theorem regular_price_correct (x : ℝ) : regular_price_of_one_tire x → x = 307 / 3 := by
  intro h
  sorry

end regular_price_correct_l227_227245


namespace remainder_of_expansion_l227_227604

theorem remainder_of_expansion (x : ℤ) : ((x + 1) ^ 2012) % (x^2 - x + 1) = 1 := 
  sorry

end remainder_of_expansion_l227_227604


namespace ratio_of_areas_l227_227632

theorem ratio_of_areas (r : ℝ) (w_smaller : ℝ) (h_smaller : ℝ) (h_semi : ℝ) :
  (5 / 4) * 40 = r + 40 →
  h_semi = 20 →
  w_smaller = 5 →
  h_smaller = 20 →
  2 * w_smaller * h_smaller / ((1 / 2) * π * h_semi^2) = 1 / π :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_l227_227632


namespace area_of_remaining_shape_l227_227454

/-- Define the initial 6x6 square grid with each cell of size 1 cm. -/
def initial_square_area : ℝ := 6 * 6

/-- Define the area of the combined dark gray triangles forming a 1x3 rectangle. -/
def dark_gray_area : ℝ := 1 * 3

/-- Define the area of the combined light gray triangles forming a 2x3 rectangle. -/
def light_gray_area : ℝ := 2 * 3

/-- Define the total area of the gray triangles cut out. -/
def total_gray_area : ℝ := dark_gray_area + light_gray_area

/-- Calculate the area of the remaining figure after cutting out the gray triangles. -/
def remaining_area : ℝ := initial_square_area - total_gray_area

/-- Proof that the area of the remaining shape is 27 square centimeters. -/
theorem area_of_remaining_shape : remaining_area = 27 := by
  sorry

end area_of_remaining_shape_l227_227454


namespace int_values_satisfying_inequality_l227_227064

theorem int_values_satisfying_inequality : 
  ∃ (N : ℕ), N = 15 ∧ ∀ (x : ℕ), 9 < x ∧ x < 25 → x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} →
  set.size {x | 9 < x ∧ x < 25 ∧ x ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}} = N :=
by
  sorry

end int_values_satisfying_inequality_l227_227064


namespace real_root_quadratic_complex_eq_l227_227684

open Complex

theorem real_root_quadratic_complex_eq (a : ℝ) :
  ∀ x : ℝ, a * (1 + I) * x^2 + (1 + a^2 * I) * x + (a^2 + I) = 0 →
  a = -1 :=
by
  intros x h
  -- We need to prove this, but we're skipping the proof for now.
  sorry

end real_root_quadratic_complex_eq_l227_227684


namespace count_integers_in_interval_l227_227058

theorem count_integers_in_interval :
  ∃ (n : ℕ), (∀ x : ℤ, 25 > x ∧ x > 9 → 10 ≤ x ∧ x ≤ 24 → x ∈ (Finset.range (25 - 10 + 1)).map (λ i, i + 10)) ∧ n = (Finset.range (25 - 10 + 1)).card :=
sorry

end count_integers_in_interval_l227_227058


namespace problem1_problem2_l227_227455

-- Problem 1: If a is parallel to b, then x = 4
theorem problem1 (x : ℝ) (u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  (a.1 / b.1 = a.2 / b.2) → x = 4 := 
by 
  intros a b h
  dsimp [a, b] at h
  sorry

-- Problem 2: If (u - 2 * v) is perpendicular to (u + v), then x = -6
theorem problem2 (x : ℝ) (a u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 + b.1, 2 * a.2 + b.2)
  ((u.1 - 2 * v.1) * (u.1 + v.1) + (u.2 - 2 * v.2) * (u.2 + v.2) = 0) → x = -6 := 
by 
  intros a b u v h
  dsimp [a, b, u, v] at h
  sorry

end problem1_problem2_l227_227455


namespace corn_pounds_l227_227437

theorem corn_pounds (c b : ℤ) (h1 : b + c = 20) (h2 : 75 * b + 99 * c = 1680) : c = 15 / 2 :=
by 
begin
  -- This is skipped.
  sorry
end

end corn_pounds_l227_227437


namespace rebus_puzzle_verified_l227_227113

-- Defining the conditions
def A := 1
def B := 1
def C := 0
def D := 1
def F := 1
def L := 1
def M := 0
def N := 1
def P := 0
def Q := 1
def T := 1
def G := 8
def H := 1
def K := 4
def W := 4
def X := 1

noncomputable def verify_rebus_puzzle : Prop :=
  (A * B * 10 = 110) ∧
  (6 * G / (10 * H + 7) = 4) ∧
  (L + N * 10 = 20) ∧
  (12 - K = 8) ∧
  (101 + 10 * W + X = 142)

-- Lean statement to verify the problem
theorem rebus_puzzle_verified : verify_rebus_puzzle :=
by {
  -- Values are already defined and will be concluded by Lean
  sorry
}

end rebus_puzzle_verified_l227_227113


namespace arithmetic_seq_problem_l227_227461

theorem arithmetic_seq_problem (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 19 = 57 →
  3 * (a 1 + 4 * d) - a 1 - (a 1 + 3 * d) = 3 :=
by
  sorry

end arithmetic_seq_problem_l227_227461


namespace solution_set_real_implies_conditions_l227_227214

variable {a b c : ℝ}

theorem solution_set_real_implies_conditions (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) : a < 0 ∧ (b^2 - 4 * a * c) < 0 := 
sorry

end solution_set_real_implies_conditions_l227_227214


namespace factor_x6_plus_8_l227_227927

theorem factor_x6_plus_8 : (x^2 + 2) ∣ (x^6 + 8) :=
by
  sorry

end factor_x6_plus_8_l227_227927


namespace gcd_765432_654321_eq_3_l227_227576

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227576


namespace total_games_played_l227_227842

theorem total_games_played (points_per_game_winner : ℕ) (points_per_game_loser : ℕ) (jack_games_won : ℕ)
  (jill_total_points : ℕ) (total_games : ℕ)
  (h1 : points_per_game_winner = 2)
  (h2 : points_per_game_loser = 1)
  (h3 : jack_games_won = 4)
  (h4 : jill_total_points = 10)
  (h5 : ∀ games_won_by_jill : ℕ, jill_total_points = games_won_by_jill * points_per_game_winner +
           (jack_games_won * points_per_game_loser)) :
  total_games = jack_games_won + (jill_total_points - jack_games_won * points_per_game_loser) / points_per_game_winner := by
  sorry

end total_games_played_l227_227842


namespace complete_remaining_parts_l227_227662

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

end complete_remaining_parts_l227_227662


namespace gcd_765432_654321_l227_227594

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227594


namespace passes_through_point_l227_227304

theorem passes_through_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end passes_through_point_l227_227304


namespace gcd_of_3150_and_9800_is_350_l227_227386

-- Definition of the two numbers
def num1 : ℕ := 3150
def num2 : ℕ := 9800

-- The greatest common factor of num1 and num2 is 350
theorem gcd_of_3150_and_9800_is_350 : Nat.gcd num1 num2 = 350 := by
  sorry

end gcd_of_3150_and_9800_is_350_l227_227386


namespace part_I_part_II_l227_227294

open Real

-- Part I
theorem part_I (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = exp x + a * exp (-x) - 2 * x)
  (h_odd : ∀ x : ℝ, f (-x) = - (f x)) : a = -1 ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Part II
theorem part_II (b : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x : ℝ, f x = exp x - exp (-x) - 2 * x)
  (h_g : ∀ x : ℝ, g x = f (2 * x) - 4 * b * f x)
  (h_pos : ∀ x : ℝ, x > 0 → g x > 0) : b ≤ 2 := by
  sorry

end part_I_part_II_l227_227294


namespace number_of_integers_inequality_l227_227051

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end number_of_integers_inequality_l227_227051


namespace molecular_weight_BaO_is_correct_l227_227268

-- Define the atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of BaO as the sum of atomic weights of Ba and O
def molecular_weight_BaO := atomic_weight_Ba + atomic_weight_O

-- Theorem stating the molecular weight of BaO
theorem molecular_weight_BaO_is_correct : molecular_weight_BaO = 153.33 := by
  -- Proof can be filled in
  sorry

end molecular_weight_BaO_is_correct_l227_227268


namespace longest_perimeter_l227_227233

theorem longest_perimeter 
  (x : ℝ) (h : x > 1)
  (pA : ℝ := 4 + 6 * x)
  (pB : ℝ := 2 + 10 * x)
  (pC : ℝ := 7 + 5 * x)
  (pD : ℝ := 6 + 6 * x)
  (pE : ℝ := 1 + 11 * x) :
  pE > pA ∧ pE > pB ∧ pE > pC ∧ pE > pD :=
by
  sorry

end longest_perimeter_l227_227233


namespace num_integers_satisfying_sqrt_ineq_l227_227036

theorem num_integers_satisfying_sqrt_ineq:
  {x : ℕ} (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) →
  Finset.card (Finset.filter (λ x, 3 < Real.sqrt x ∧ Real.sqrt x < 5) (Finset.range 25)) = 15 :=
by
  sorry

end num_integers_satisfying_sqrt_ineq_l227_227036


namespace max_area_equilateral_triangle_l227_227376

/-- The sides of rectangle ABCD have lengths 13 and 14. An equilateral triangle is drawn so that no point of the triangle lies outside ABCD. The maximum possible area of such a triangle can be written in the form p * sqrt q - r, where p, q, and r are positive integers, and q is not divisible by the square of any prime number. Prove that p + q + r = 732. -/
theorem max_area_equilateral_triangle (A B C D : ℝ) (AB BC : ℝ) (p q r : ℕ) 
  (h1 : AB = 13) (h2 : BC = 14)
  (h3 : q > 0) (h4 : ∀ (n : ℕ), (n * n ∣ q) → n = 1)
  (h5 : ∃ (s : ℝ), AB = s * sqrt q - r):
  p + q + r = 732 := 
sorry

end max_area_equilateral_triangle_l227_227376


namespace legoland_kangaroos_l227_227983

theorem legoland_kangaroos :
  ∃ (K R : ℕ), R = 5 * K ∧ K + R = 216 ∧ R = 180 := by
  sorry

end legoland_kangaroos_l227_227983


namespace general_term_of_A_inter_B_l227_227854

def setA : Set ℕ := { n*n + n | n : ℕ }
def setB : Set ℕ := { 3*m - 1 | m : ℕ }

theorem general_term_of_A_inter_B (k : ℕ) :
  let a_k := 9*k^2 - 9*k + 2
  a_k ∈ setA ∩ setB ∧ ∀ n ∈ setA ∩ setB, n = a_k :=
sorry

end general_term_of_A_inter_B_l227_227854


namespace ben_has_10_fewer_stickers_than_ryan_l227_227982

theorem ben_has_10_fewer_stickers_than_ryan :
  ∀ (Karl_stickers Ryan_stickers Ben_stickers total_stickers : ℕ),
    Karl_stickers = 25 →
    Ryan_stickers = Karl_stickers + 20 →
    total_stickers = Karl_stickers + Ryan_stickers + Ben_stickers →
    total_stickers = 105 →
    (Ryan_stickers - Ben_stickers) = 10 :=
by
  intros Karl_stickers Ryan_stickers Ben_stickers total_stickers h1 h2 h3 h4
  -- Conditions mentioned in a)
  exact sorry

end ben_has_10_fewer_stickers_than_ryan_l227_227982


namespace ferry_P_travel_time_l227_227795

-- Define the conditions based on the problem statement
variables (t : ℝ) -- travel time of ferry P
def speed_P := 6 -- speed of ferry P in km/h
def speed_Q := speed_P + 3 -- speed of ferry Q in km/h
def distance_P := speed_P * t -- distance traveled by ferry P in km
def distance_Q := 3 * distance_P -- distance traveled by ferry Q in km
def time_Q := t + 3 -- travel time of ferry Q

-- Theorem to prove that travel time t for ferry P is 3 hours
theorem ferry_P_travel_time : time_Q * speed_Q = distance_Q → t = 3 :=
by {
  -- Since you've mentioned to include the statement only and not the proof,
  -- Therefore, the proof body is left as an exercise or represented by sorry.
  sorry
}

end ferry_P_travel_time_l227_227795


namespace percentage_increase_l227_227313

variables {a b : ℝ} -- Assuming a and b are real numbers

-- Define the conditions explicitly
def initial_workers := a
def workers_left := b
def remaining_workers := a - b

-- Define the theorem for percentage increase in daily performance
theorem percentage_increase (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (100 * b) / (a - b) = (100 * a * b) / (a * (a - b)) :=
by
  sorry -- Proof will be filled in as needed

end percentage_increase_l227_227313


namespace ABC_books_sold_eq_4_l227_227251

/-- "TOP" book cost in dollars --/
def TOP_price : ℕ := 8

/-- "ABC" book cost in dollars --/
def ABC_price : ℕ := 23

/-- Number of "TOP" books sold --/
def TOP_books_sold : ℕ := 13

/-- Difference in earnings in dollars --/
def earnings_difference : ℕ := 12

/-- Prove the number of "ABC" books sold --/
theorem ABC_books_sold_eq_4 (x : ℕ) (h : TOP_books_sold * TOP_price - x * ABC_price = earnings_difference) : x = 4 :=
by
  sorry

end ABC_books_sold_eq_4_l227_227251


namespace xiao_ming_error_step_l227_227224

theorem xiao_ming_error_step (x : ℝ) :
  (1 / (x + 1) = (2 * x) / (3 * x + 3) - 1) → 
  3 = 2 * x - (3 * x + 3) → 
  (3 = 2 * x - 3 * x + 3) ↔ false := by
  sorry

end xiao_ming_error_step_l227_227224


namespace words_per_page_l227_227626

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 270 [MOD 221]) (h2 : p ≤ 120) : p = 107 :=
sorry

end words_per_page_l227_227626


namespace arc_length_of_circle_l227_227526

theorem arc_length_of_circle (r θ : ℝ) (h_r : r = 2) (h_θ : θ = 120) : 
  (θ / 180 * r * Real.pi) = (4 / 3) * Real.pi := by
  sorry

end arc_length_of_circle_l227_227526


namespace gcd_765432_654321_eq_3_l227_227574

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l227_227574


namespace least_months_exceed_tripled_borrowed_l227_227321

theorem least_months_exceed_tripled_borrowed :
  ∃ t : ℕ, (1.03 : ℝ)^t > 3 ∧ ∀ n < t, (1.03 : ℝ)^n ≤ 3 :=
sorry

end least_months_exceed_tripled_borrowed_l227_227321


namespace james_tip_percentage_l227_227979

theorem james_tip_percentage :
  let ticket_cost : ℝ := 100
  let dinner_cost : ℝ := 120
  let limo_cost_per_hour : ℝ := 80
  let limo_hours : ℕ := 6
  let total_cost_with_tip : ℝ := 836
  let total_cost_without_tip : ℝ := 2 * ticket_cost + limo_hours * limo_cost_per_hour + dinner_cost
  let tip : ℝ := total_cost_with_tip - total_cost_without_tip
  let percentage_tip : ℝ := (tip / dinner_cost) * 100
  percentage_tip = 30 :=
by
  sorry

end james_tip_percentage_l227_227979


namespace calories_consumed_l227_227326

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l227_227326


namespace fraction_value_l227_227609

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end fraction_value_l227_227609


namespace unique_zero_f_x1_minus_2x2_l227_227963

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l227_227963


namespace find_g_zero_l227_227512

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

end find_g_zero_l227_227512


namespace distance_travel_l227_227758

-- Definition of the parameters and the proof problem
variable (W_t : ℕ)
variable (R_c : ℕ)
variable (remaining_coal : ℕ)

-- Conditions
def rate_of_coal_consumption : Prop := R_c = 4 * W_t / 1000
def remaining_coal_amount : Prop := remaining_coal = 160

-- Theorem statement
theorem distance_travel (W_t : ℕ) (R_c : ℕ) (remaining_coal : ℕ) 
  (h1 : rate_of_coal_consumption W_t R_c) 
  (h2 : remaining_coal_amount remaining_coal) : 
  (remaining_coal * 1000 / 4 / W_t) = 40000 / W_t := 
by
  sorry

end distance_travel_l227_227758


namespace evaluate_expression_l227_227265

theorem evaluate_expression : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end evaluate_expression_l227_227265


namespace sales_function_maximize_profit_l227_227442

def y (x : ℝ) : ℝ := 300 - 10 * (x - 44)

theorem sales_function (x : ℝ) (h₁ : x ≥ 44) (h₂ : x ≤ 52) :
  y x = -10 * x + 740 :=
by 
  have : y x = 300 - 10 * (x - 44) := rfl
  rw [this, sub_mul, add_sub_cancel]
  norm_num

def profit (x : ℝ) : ℝ := (x - 40) * (300 - 10 * (x - 44))

theorem maximize_profit (x : ℝ) (h₁ : x = 52) :
  profit x = 2640 :=
by
  unfold profit
  rw [h₁, sub_self, zero_mul, add_zero, mul_comm, ← mul_assoc, mul_left_comm]
  norm_num

end sales_function_maximize_profit_l227_227442


namespace lattice_points_on_segment_l227_227649

theorem lattice_points_on_segment : 
  let x1 := 5 
  let y1 := 23 
  let x2 := 47 
  let y2 := 297 
  ∃ n, n = 3 ∧ ∀ p : ℕ × ℕ, (p = (x1, y1) ∨ p = (x2, y2) ∨ ∃ t : ℕ, p = (x1 + t * (x2 - x1) / 2, y1 + t * (y2 - y1) / 2)) := 
sorry

end lattice_points_on_segment_l227_227649


namespace fish_weight_l227_227907

theorem fish_weight (W : ℝ) (h : W = 2 + W / 3) : W = 3 :=
by
  sorry

end fish_weight_l227_227907


namespace value_of_b_pos_sum_for_all_x_l227_227703

noncomputable def f (b : ℝ) (x : ℝ) := 3 * x^2 - 2 * x + b
noncomputable def g (b : ℝ) (x : ℝ) := x^2 + b * x - 1
noncomputable def sum_f_g (b : ℝ) (x : ℝ) := f b x + g b x

theorem value_of_b (b : ℝ) (h : ∀ x : ℝ, (sum_f_g b x = 4 * x^2 + (b - 2) * x + (b - 1))) :
  b = 2 := 
sorry

theorem pos_sum_for_all_x :
  ∀ x : ℝ, 4 * x^2 + 1 > 0 := 
sorry

end value_of_b_pos_sum_for_all_x_l227_227703


namespace part1_part2_l227_227465

noncomputable def f (x : ℝ) : ℝ := (Real.exp (-x) - Real.exp x) / 2

theorem part1 (h_odd : ∀ x, f (-x) = -f x) (g : ℝ → ℝ) (h_even : ∀ x, g (-x) = g x)
  (h_g_def : ∀ x, g x = f x + Real.exp x) :
  ∀ x, f x = (Real.exp (-x) - Real.exp x) / 2 := sorry

theorem part2 : {x : ℝ | f x ≥ 3 / 4} = {x | x ≤ -Real.log 2} := sorry

end part1_part2_l227_227465


namespace min_m_n_sum_l227_227729

theorem min_m_n_sum (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 108 * m = n^3) : m + n = 8 :=
sorry

end min_m_n_sum_l227_227729


namespace necessary_but_not_sufficient_condition_l227_227474

-- Define the sets M and P
def M (x : ℝ) : Prop := x > 2
def P (x : ℝ) : Prop := x < 3

-- Statement of the problem
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (M x ∨ P x) → (x ∈ { y : ℝ | 2 < y ∧ y < 3 }) :=
sorry

end necessary_but_not_sufficient_condition_l227_227474


namespace find_number_eq_l227_227305

theorem find_number_eq : ∃ x : ℚ, (35 / 100) * x = (25 / 100) * 40 ∧ x = 200 / 7 :=
by
  sorry

end find_number_eq_l227_227305


namespace village_population_l227_227756

theorem village_population (P : ℝ) (h : 0.8 * P = 32000) : P = 40000 := by
  sorry

end village_population_l227_227756


namespace factorize_poly_l227_227779

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorize_poly_l227_227779


namespace count_distinct_m_in_right_triangle_l227_227436

theorem count_distinct_m_in_right_triangle (k : ℝ) (hk : k > 0) :
  ∃! m : ℝ, (m = -3/8 ∨ m = -3/4) :=
by
  sorry

end count_distinct_m_in_right_triangle_l227_227436


namespace find_length_of_first_dimension_of_tank_l227_227912

theorem find_length_of_first_dimension_of_tank 
    (w : ℝ) (h : ℝ) (cost_per_sq_ft : ℝ) (total_cost : ℝ) (l : ℝ) :
    w = 5 → h = 3 → cost_per_sq_ft = 20 → total_cost = 1880 → 
    1880 = (2 * l * w + 2 * l * h + 2 * w * h) * cost_per_sq_ft →
    l = 4 := 
by
  intros hw hh hcost htotal heq
  sorry

end find_length_of_first_dimension_of_tank_l227_227912


namespace value_decrease_proof_l227_227310

noncomputable def value_comparison (diana_usd : ℝ) (etienne_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  let etienne_usd := etienne_eur * eur_to_usd
  let percentage_decrease := ((diana_usd - etienne_usd) / diana_usd) * 100
  percentage_decrease

theorem value_decrease_proof :
  value_comparison 700 300 1.5 = 35.71 :=
by
  sorry

end value_decrease_proof_l227_227310


namespace praveen_hari_profit_ratio_l227_227345

theorem praveen_hari_profit_ratio
  (praveen_capital : ℕ := 3360)
  (hari_capital : ℕ := 8640)
  (time_praveen_invested : ℕ := 12)
  (time_hari_invested : ℕ := 7)
  (praveen_shares_full_time : ℕ := praveen_capital * time_praveen_invested)
  (hari_shares_full_time : ℕ := hari_capital * time_hari_invested)
  (gcd_common : ℕ := Nat.gcd praveen_shares_full_time hari_shares_full_time) :
  (praveen_shares_full_time / gcd_common) * 2 = 2 ∧ (hari_shares_full_time / gcd_common) * 2 = 3 := by
    sorry

end praveen_hari_profit_ratio_l227_227345


namespace determine_counterfeit_coin_l227_227773

-- Definitions and conditions
def coin_weight (coin : ℕ) : ℕ :=
  match coin with
  | 1 => 1 -- 1-kopek coin weighs 1 gram
  | 2 => 2 -- 2-kopeks coin weighs 2 grams
  | 3 => 3 -- 3-kopeks coin weighs 3 grams
  | 5 => 5 -- 5-kopeks coin weighs 5 grams
  | _ => 0 -- Invalid coin denomination, should not happen

def is_counterfeit (coin : ℕ) (actual_weight : ℕ) : Prop :=
  coin_weight coin ≠ actual_weight

-- Statement of the problem to be proved
theorem determine_counterfeit_coin (coins : List (ℕ × ℕ)) :
   (∀ (coin: ℕ) (weight: ℕ) (h : (coin, weight) ∈ coins),
      coin_weight coin = weight ∨ is_counterfeit coin weight) →
   (∃ (counterfeit_coin: ℕ) (weight: ℕ),
      (counterfeit_coin, weight) ∈ coins ∧ is_counterfeit counterfeit_coin weight) :=
sorry

end determine_counterfeit_coin_l227_227773


namespace white_balls_count_l227_227904

theorem white_balls_count
  (total_balls : ℕ)
  (white_balls blue_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls + blue_balls + red_balls = total_balls)
  (h3 : blue_balls = white_balls + 12)
  (h4 : red_balls = 2 * blue_balls) : white_balls = 16 := by
  sorry

end white_balls_count_l227_227904


namespace seventh_degree_solution_l227_227392

theorem seventh_degree_solution (a b x : ℝ) :
  (x^7 - 7 * a * x^5 + 14 * a^2 * x^3 - 7 * a^3 * x = b) ↔
  ∃ α β : ℝ, α + β = x ∧ α * β = a ∧ α^7 + β^7 = b :=
by
  sorry

end seventh_degree_solution_l227_227392


namespace time_to_count_envelopes_l227_227103

theorem time_to_count_envelopes (r : ℕ) : (r / 10 = 1) → (r * 60 / r = 60) ∧ (r * 90 / r = 90) :=
by sorry

end time_to_count_envelopes_l227_227103


namespace concyclic_iff_l227_227984

variables {A B C H O' N D : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H]
variables [MetricSpace O'] [MetricSpace N] [MetricSpace D]
variables (a b c R : ℝ)

-- Conditions from the problem
def is_orthocenter (H : Type*) (A B C : Type*) : Prop :=
  -- definition of orthocenter using suitable predicates (omitted for brevity) 
  sorry

def is_circumcenter (O' : Type*) (B H C : Type*) : Prop :=
  -- definition of circumcenter using suitable predicates (omitted for brevity) 
  sorry

def is_midpoint (N : Type*) (A O' : Type*) : Prop :=
  -- definition of midpoint using suitable predicates (omitted for brevity) 
  sorry

def is_reflection (N D : Type*) (B C : Type*) : Prop :=
  -- definition of reflection about the side BC (omitted for brevity) 
  sorry

-- Definition that points A, B, C, D are concyclic
def are_concyclic (A B C D : Type*) : Prop :=
  -- definition using suitable predicates (omitted for brevity)
  sorry

-- Main theorem statement
theorem concyclic_iff (h1 : is_orthocenter H A B C) (h2 : is_circumcenter O' B H C) 
                      (h3 : is_midpoint N A O') (h4 : is_reflection N D B C)
                      (ha : a = 1) (hb : b = 1) (hc : c = 1) (hR : R = 1) :
  are_concyclic A B C D ↔ b^2 + c^2 - a^2 = 3 * R^2 := 
sorry

end concyclic_iff_l227_227984


namespace product_evaluation_l227_227118

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 :=
by 
sorry

end product_evaluation_l227_227118


namespace mean_of_combined_set_is_52_over_3_l227_227207

noncomputable def mean_combined_set : ℚ := 
  let mean_set1 := 10
  let size_set1 := 4
  let mean_set2 := 21
  let size_set2 := 8
  let sum_set1 := mean_set1 * size_set1
  let sum_set2 := mean_set2 * size_set2
  let total_sum := sum_set1 + sum_set2
  let combined_size := size_set1 + size_set2
  let combined_mean := total_sum / combined_size
  combined_mean

theorem mean_of_combined_set_is_52_over_3 :
  mean_combined_set = 52 / 3 :=
by
  sorry

end mean_of_combined_set_is_52_over_3_l227_227207


namespace correct_statements_B_and_C_l227_227287

-- Given real numbers a, b, c satisfying the conditions
variables (a b c : ℝ)
variables (h1 : a > b)
variables (h2 : b > c)
variables (h3 : a + b + c = 0)

theorem correct_statements_B_and_C : (a - c > 2 * b) ∧ (a ^ 2 > b ^ 2) :=
by
  sorry

end correct_statements_B_and_C_l227_227287


namespace problem1_problem2_l227_227959

section Problem
variables (a : ℝ) (x : ℝ) (x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem problem1 (ha : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem problem2 (ha : a ≥ 0) (h1 : x1 ∈ Icc (-1 : ℝ) (Real.inf)) (h2 : x2 ∈ Icc (-1 : ℝ) (Real.inf)) (h : f a x1 = g a x1 - g a x2) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end Problem

end problem1_problem2_l227_227959


namespace cube_root_simplification_l227_227086

theorem cube_root_simplification : 
  (∛(5^7 + 5^7 + 5^7) = 225 * ∛15) :=
by
  sorry

end cube_root_simplification_l227_227086


namespace gcd_proof_l227_227557

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l227_227557


namespace N_intersect_M_complement_l227_227955

-- Definitions based on given conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def N : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def M_complement : Set ℝ := { x | x < -2 ∨ x > 3 }  -- complement of M in ℝ

-- Lean statement for the proof problem
theorem N_intersect_M_complement :
  N ∩ M_complement = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end N_intersect_M_complement_l227_227955


namespace not_perfect_square_7_301_l227_227390

theorem not_perfect_square_7_301 :
  ¬ ∃ x : ℝ, x^2 = 7^301 := sorry

end not_perfect_square_7_301_l227_227390


namespace trig_identity_l227_227787

theorem trig_identity :
  (Real.sin (17 * Real.pi / 180) * Real.cos (47 * Real.pi / 180) - 
   Real.sin (73 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = -1/2 := 
by
  sorry

end trig_identity_l227_227787


namespace num_integers_satisfying_sqrt_ineq_l227_227038

theorem num_integers_satisfying_sqrt_ineq:
  {x : ℕ} (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) →
  Finset.card (Finset.filter (λ x, 3 < Real.sqrt x ∧ Real.sqrt x < 5) (Finset.range 25)) = 15 :=
by
  sorry

end num_integers_satisfying_sqrt_ineq_l227_227038


namespace gcd_765432_654321_l227_227553

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227553


namespace jaylene_saves_fraction_l227_227194

-- Statement of the problem
theorem jaylene_saves_fraction (r_saves : ℝ) (j_saves : ℝ) (m_saves : ℝ) 
    (r_salary_fraction : r_saves = 2 / 5) 
    (m_salary_fraction : m_saves = 1 / 2) 
    (total_savings : 4 * (r_saves * 500 + j_saves * 500 + m_saves * 500) = 3000) : 
    j_saves = 3 / 5 := 
by 
  sorry

end jaylene_saves_fraction_l227_227194


namespace gp_condition_necessity_l227_227290

theorem gp_condition_necessity {a b c : ℝ} 
    (h_gp: ∃ r: ℝ, b = a * r ∧ c = a * r^2 ) : b^2 = a * c :=
by
  sorry

end gp_condition_necessity_l227_227290


namespace correct_meiosis_sequence_l227_227650

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

end correct_meiosis_sequence_l227_227650


namespace reciprocal_of_2023_l227_227372

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l227_227372


namespace Amanda_money_left_l227_227768

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l227_227768


namespace bus_capacity_total_kids_l227_227742

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

end bus_capacity_total_kids_l227_227742


namespace maintenance_cost_relation_maximize_average_profit_l227_227728

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

end maintenance_cost_relation_maximize_average_profit_l227_227728


namespace max_value_of_expression_l227_227002

noncomputable def maxExpression (x y : ℝ) :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  maxExpression x y ≤ (656^2 / 18) :=
by
  sorry

end max_value_of_expression_l227_227002


namespace fraction_of_cost_due_to_high_octane_is_half_l227_227089

theorem fraction_of_cost_due_to_high_octane_is_half :
  ∀ (cost_regular cost_high : ℝ) (units_high units_regular : ℕ),
    units_high * cost_high + units_regular * cost_regular ≠ 0 →
    cost_high = 3 * cost_regular →
    units_high = 1515 →
    units_regular = 4545 →
    (units_high * cost_high) / (units_high * cost_high + units_regular * cost_regular) = 1 / 2 :=
by
  intro cost_regular cost_high units_high units_regular h_total_cost_ne_zero h_cost_rel h_units_high h_units_regular
  -- skip the actual proof steps
  sorry

end fraction_of_cost_due_to_high_octane_is_half_l227_227089


namespace solve_equation_l227_227350

theorem solve_equation : ∀ x : ℝ, 4 * x + 4 - x - 2 * x + 2 - 2 - x + 2 + 6 = 0 → x = 0 :=
by 
  intro x h
  sorry

end solve_equation_l227_227350


namespace find_breadth_of_rectangular_plot_l227_227095

-- Define the conditions
def length_is_thrice_breadth (b l : ℕ) : Prop := l = 3 * b
def area_is_363 (b l : ℕ) : Prop := l * b = 363

-- State the theorem
theorem find_breadth_of_rectangular_plot : ∃ b : ℕ, ∀ l : ℕ, length_is_thrice_breadth b l ∧ area_is_363 b l → b = 11 := 
by
  sorry

end find_breadth_of_rectangular_plot_l227_227095


namespace intersection_complement_l227_227992

open Set

def UniversalSet := ℝ
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def CU_M : Set ℝ := compl M

theorem intersection_complement :
  N ∩ CU_M = {x | 1 < x ∧ x ≤ 2} :=
by sorry

end intersection_complement_l227_227992


namespace house_construction_days_l227_227637

theorem house_construction_days
  (D : ℕ) -- number of planned days to build the house
  (Hwork_done : 1000 + 200 * (D - 10) = 100 * (D + 90)) : 
  D = 110 :=
sorry

end house_construction_days_l227_227637


namespace exponent_of_5_in_30_factorial_l227_227170

theorem exponent_of_5_in_30_factorial : 
  let n := 30!
  let p := 5
  (n.factor_count p = 7) :=
by
  sorry

end exponent_of_5_in_30_factorial_l227_227170


namespace hall_ratio_l227_227533

open Real

theorem hall_ratio (w l : ℝ) (h_area : w * l = 288) (h_diff : l - w = 12) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l227_227533


namespace hannah_bananas_l227_227148

theorem hannah_bananas (B : ℕ) (h1 : B / 4 = 15 / 3) : B = 20 :=
by
  sorry

end hannah_bananas_l227_227148


namespace gcd_765432_654321_l227_227589

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l227_227589


namespace slope_tangent_line_l227_227136

variable {f : ℝ → ℝ}

-- Assumption: f is differentiable
def differentiable_at (f : ℝ → ℝ) (x : ℝ) := ∃ f', ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |(f (x + h) - f x) / h - f'| < ε

-- Hypothesis: limit condition
axiom limit_condition : (∀ x, differentiable_at f (1 - x)) → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε)

-- Theorem: the slope of the tangent line to the curve y = f(x) at (1, f(1)) is -2
theorem slope_tangent_line : differentiable_at f 1 → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε) → deriv f 1 = -2 :=
by
    intro h_diff h_lim
    sorry

end slope_tangent_line_l227_227136


namespace only_solution_l227_227202

theorem only_solution (x : ℝ) : (3 / (x - 3) = 5 / (x - 5)) ↔ (x = 0) := 
sorry

end only_solution_l227_227202


namespace min_value_fraction_l227_227274

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  ∃ c, (c = 9) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) → (1/x + 4/y ≥ c)) :=
by
  sorry

end min_value_fraction_l227_227274


namespace compare_abc_l227_227462

noncomputable def a : ℝ := Real.sin (145 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (52 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (47 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l227_227462


namespace a_plus_c_eq_neg_300_l227_227187

namespace Polynomials

variable {α : Type*} [LinearOrderedField α]

def f (a b x : α) := x^2 + a * x + b
def g (c d x : α) := x^2 + c * x + d

theorem a_plus_c_eq_neg_300 
  {a b c d : α}
  (h1 : ∀ x, f a b x ≥ -144) 
  (h2 : ∀ x, g c d x ≥ -144)
  (h3 : f a b 150 = -200) 
  (h4 : g c d 150 = -200)
  (h5 : ∃ x, (2*x + a = 0) ∧ g c d x = 0)
  (h6 : ∃ x, (2*x + c = 0) ∧ f a b x = 0) :
  a + c = -300 := 
sorry

end Polynomials

end a_plus_c_eq_neg_300_l227_227187


namespace trajectory_of_P_l227_227801

def point := ℝ × ℝ

-- Definitions for points A and F, and the circle equation
def A : point := (-1, 0)
def F (x y : ℝ) := (x - 1) ^ 2 + y ^ 2 = 16

-- Main theorem statement: proving the trajectory equation of point P
theorem trajectory_of_P : 
  (∀ (B : point), F B.1 B.2 → 
  (∃ P : point, ∃ (k : ℝ), (P.1 - B.1) * k = -(P.2 - B.2) ∧ (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0)) →
  (∃ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) :=
sorry

end trajectory_of_P_l227_227801


namespace sum_of_first_ten_primes_with_units_digit_3_l227_227784

def units_digit_3_and_prime (n : ℕ) : Prop :=
  (n % 10 = 3) ∧ (Prime n)

def first_ten_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

theorem sum_of_first_ten_primes_with_units_digit_3 :
  list.sum first_ten_primes_with_units_digit_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l227_227784


namespace first_term_of_geometric_series_l227_227770

theorem first_term_of_geometric_series (a r S : ℝ)
  (h_sum : S = a / (1 - r))
  (h_r : r = 1/3)
  (h_S : S = 18) :
  a = 12 :=
by
  sorry

end first_term_of_geometric_series_l227_227770


namespace abs_abc_eq_one_l227_227510

variable (a b c : ℝ)

-- Conditions
axiom distinct_nonzero : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)
axiom condition : a^2 + 1/(b^2) = b^2 + 1/(c^2) ∧ b^2 + 1/(c^2) = c^2 + 1/(a^2)

theorem abs_abc_eq_one : |a * b * c| = 1 :=
by
  sorry

end abs_abc_eq_one_l227_227510


namespace sum_a_b_c_d_eq_nine_l227_227665

theorem sum_a_b_c_d_eq_nine
  (a b c d : ℤ)
  (h : (Polynomial.X ^ 2 + (Polynomial.C a) * Polynomial.X + Polynomial.C b) *
       (Polynomial.X ^ 2 + (Polynomial.C c) * Polynomial.X + Polynomial.C d) =
       Polynomial.X ^ 4 + 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 11 * Polynomial.X + 6) :
  a + b + c + d = 9 :=
by
  sorry

end sum_a_b_c_d_eq_nine_l227_227665


namespace range_of_k_l227_227807

noncomputable def e := Real.exp 1

theorem range_of_k (k : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ e ^ (x1 - 1) = |k * x1| ∧ e ^ (x2 - 1) = |k * x2| ∧ e ^ (x3 - 1) = |k * x3|) : k^2 > 1 := sorry

end range_of_k_l227_227807


namespace ratio_of_areas_l227_227819

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l227_227819


namespace exponent_of_5_in_30_factorial_l227_227172

theorem exponent_of_5_in_30_factorial : 
  (nat.factorial 30).prime_factors.count 5 = 7 := 
begin
  sorry
end

end exponent_of_5_in_30_factorial_l227_227172


namespace college_girls_count_l227_227164

theorem college_girls_count 
  (B G : ℕ)
  (h1 : B / G = 8 / 5)
  (h2 : B + G = 455) : 
  G = 175 := 
sorry

end college_girls_count_l227_227164


namespace hours_per_day_is_8_l227_227541

-- Define the conditions
def hire_two_bodyguards (day_count : ℕ) (total_payment : ℕ) (hourly_rate : ℕ) (daily_hours : ℕ) : Prop :=
  2 * hourly_rate * day_count * daily_hours = total_payment

-- Define the correct answer
theorem hours_per_day_is_8 :
  hire_two_bodyguards 7 2240 20 8 :=
by
  -- Here, you would provide the step-by-step justification, but we use sorry since no proof is required.
  sorry

end hours_per_day_is_8_l227_227541


namespace line_intersects_circle_l227_227472

theorem line_intersects_circle (α : ℝ) (r : ℝ) (hα : true) (hr : r > 0) :
  (∃ x y : ℝ, (x * Real.cos α + y * Real.sin α = 1) ∧ (x^2 + y^2 = r^2)) → r > 1 :=
by
  sorry

end line_intersects_circle_l227_227472


namespace medicine_liquid_poured_l227_227630

theorem medicine_liquid_poured (x : ℝ) (h : 63 * (1 - x / 63) * (1 - x / 63) = 28) : x = 18 :=
by
  sorry

end medicine_liquid_poured_l227_227630


namespace solve_for_x2_plus_9y2_l227_227486

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_l227_227486


namespace number_of_men_in_larger_group_l227_227755

-- Define the constants and conditions
def men1 := 36         -- men in the first group
def days1 := 18        -- days taken by the first group
def men2 := 108       -- men in the larger group (what we want to prove)
def days2 := 6         -- days taken by the second group

-- Given conditions as lean definitions
def total_work (men : Nat) (days : Nat) := men * days
def condition1 := (total_work men1 days1 = 648)
def condition2 := (total_work men2 days2 = 648)

-- Problem statement 
-- proving that men2 is 108
theorem number_of_men_in_larger_group : condition1 → condition2 → men2 = 108 :=
by
  intros
  sorry

end number_of_men_in_larger_group_l227_227755


namespace convex_polygon_sides_eq_49_l227_227312

theorem convex_polygon_sides_eq_49 
  (n : ℕ)
  (hn : n > 0) 
  (h : (n * (n - 3)) / 2 = 23 * n) : n = 49 :=
sorry

end convex_polygon_sides_eq_49_l227_227312


namespace overall_average_mark_l227_227914

theorem overall_average_mark :
  let n1 := 70
  let mean1 := 50
  let n2 := 35
  let mean2 := 60
  let n3 := 45
  let mean3 := 55
  let n4 := 42
  let mean4 := 45
  (n1 * mean1 + n2 * mean2 + n3 * mean3 + n4 * mean4 : ℝ) / (n1 + n2 + n3 + n4) = 51.89 := 
by {
  sorry
}

end overall_average_mark_l227_227914


namespace g_432_l227_227988

theorem g_432 (g : ℕ → ℤ)
  (h_mul : ∀ x y : ℕ, 0 < x → 0 < y → g (x * y) = g x + g y)
  (h8 : g 8 = 21)
  (h18 : g 18 = 26) :
  g 432 = 47 :=
  sorry

end g_432_l227_227988


namespace sum_of_two_consecutive_negative_integers_l227_227212

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 812) (h_neg : n < 0 ∧ (n + 1) < 0) : 
  n + (n + 1) = -57 :=
sorry

end sum_of_two_consecutive_negative_integers_l227_227212


namespace find_x0_l227_227678

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

end find_x0_l227_227678


namespace complex_z_power_l227_227138

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_z_power_l227_227138


namespace greatest_common_ratio_l227_227741

theorem greatest_common_ratio {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : (b = (a + c) / 2 → b^2 = a * c) ∨ (c = (a + b) / 2 ∧ b = -a / 2)) :
  ∃ r : ℝ, r = -2 :=
by
  sorry

end greatest_common_ratio_l227_227741


namespace find_other_number_l227_227618

theorem find_other_number (A B : ℕ) (hcf : ℕ) (lcm : ℕ) 
  (H1 : hcf = 12) 
  (H2 : lcm = 312) 
  (H3 : A = 24) 
  (H4 : hcf * lcm = A * B) : 
  B = 156 :=
by sorry

end find_other_number_l227_227618


namespace brock_buys_7_cookies_l227_227336

variable (cookies_total : ℕ)
variable (sold_to_stone : ℕ)
variable (left_after_sale : ℕ)
variable (cookies_brock_buys : ℕ)
variable (cookies_katy_buys : ℕ)

theorem brock_buys_7_cookies
  (h1 : cookies_total = 5 * 12)
  (h2 : sold_to_stone = 2 * 12)
  (h3 : left_after_sale = 15)
  (h4 : cookies_total - sold_to_stone - (cookies_brock_buys + cookies_katy_buys) = left_after_sale)
  (h5 : cookies_katy_buys = 2 * cookies_brock_buys) :
  cookies_brock_buys = 7 :=
by
  -- Proof is skipped
  sorry

end brock_buys_7_cookies_l227_227336


namespace cistern_length_l227_227100

variable (L : ℝ) (width water_depth total_area : ℝ)

theorem cistern_length
  (h_width : width = 8)
  (h_water_depth : water_depth = 1.5)
  (h_total_area : total_area = 134) :
  11 * L + 24 = total_area → L = 10 :=
by
  intro h_eq
  have h_eq1 : 11 * L = 110 := by
    linarith
  have h_L : L = 10 := by
    linarith
  exact h_L

end cistern_length_l227_227100


namespace f_at_1_over_11_l227_227332

noncomputable def f : (ℝ → ℝ) := sorry

axiom f_domain : ∀ x, 0 < x → 0 < f x

axiom f_eq : ∀ x y, 0 < x → 0 < y → 10 * ((x + y) / (x * y)) = (f x) * (f y) - f (x * y) - 90

theorem f_at_1_over_11 : f (1 / 11) = 21 := by
  -- proof is omitted
  sorry

end f_at_1_over_11_l227_227332


namespace soccer_players_count_l227_227835

theorem soccer_players_count (total_socks : ℕ) (P : ℕ) 
  (h_total_socks : total_socks = 22)
  (h_each_player_contributes : ∀ p : ℕ, p = P → total_socks = 2 * P) :
  P = 11 :=
by
  sorry

end soccer_players_count_l227_227835


namespace good_horse_catchup_l227_227840

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end good_horse_catchup_l227_227840


namespace solve_quadratic_eq_l227_227198

theorem solve_quadratic_eq (x : ℝ) : x^2 + 8 * x = 9 ↔ x = -9 ∨ x = 1 :=
by
  sorry

end solve_quadratic_eq_l227_227198


namespace count_integer_values_l227_227035

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end count_integer_values_l227_227035


namespace product_not_perfect_square_l227_227343

theorem product_not_perfect_square :
  ¬ ∃ n : ℕ, n^2 = (2021^1004) * (6^3) :=
by
  sorry

end product_not_perfect_square_l227_227343


namespace determine_length_AY_l227_227860

noncomputable def length_of_AY 
  (A B C D Y : Point) (circle_diameter : ℝ)
  (h1 : OnCircle A circle_diameter) 
  (h2 : OnCircle B circle_diameter) 
  (h3 : OnCircle C circle_diameter) 
  (h4 : OnCircle D circle_diameter)
  (h5 : Y ∈ diameter (A, D)) 
  (h6 : distance B Y = distance C Y) 
  (h7 : angle (A, B, C) = 12 * (π / 180)) 
  (h8 : angle (B, Y, C) = 36 * (π / 180)): ℝ :=
  sin (12 * π / 180) * sin (12 * π / 180) * (csc (18 * π / 180))

theorem determine_length_AY 
  (A B C D Y : Point) (circle_diameter : ℝ)
  (h1 : OnCircle A circle_diameter) 
  (h2 : OnCircle B circle_diameter) 
  (h3 : OnCircle C circle_diameter) 
  (h4 : OnCircle D circle_diameter)
  (h5 : Y ∈ diameter (A, D)) 
  (h6 : distance B Y = distance C Y) 
  (h7 : angle (A, B, C) = 12 * (π / 180)) 
  (h8 : angle (B, Y, C) = 36 * (π / 180)) :
  AY = length_of_AY A B C D Y circle_diameter h1 h2 h3 h4 h5 h6 h7 h8 :=
sorry

end determine_length_AY_l227_227860


namespace gcd_765432_654321_l227_227568

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227568


namespace correct_location_l227_227223

-- Define the possible options
inductive Location
| A : Location
| B : Location
| C : Location
| D : Location

-- Define the conditions
def option_A : Prop := ¬(∃ d, d ≠ "right")
def option_B : Prop := ¬(∃ d, d ≠ 900)
def option_C : Prop := ¬(∃ d, d ≠ "west")
def option_D : Prop := (∃ d₁ d₂, d₁ = "west" ∧ d₂ = 900)

-- The objective is to prove that option D is the correct description of the location
theorem correct_location : ∃ l, l = Location.D → 
  (option_A ∧ option_B ∧ option_C ∧ option_D) :=
by
  sorry

end correct_location_l227_227223


namespace infinite_solutions_l227_227450

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end infinite_solutions_l227_227450


namespace Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l227_227997

-- Definition of the problem conditions
def initially_painted (m n : ℕ) : Prop :=
  ∃ i j : ℕ, i < m ∧ j < n
  
def odd_painted_neighbors (m n : ℕ) : Prop :=
  ∀ i j : ℕ, i < m ∧ j < n →
  (∃ k l : ℕ, (k = i+1 ∨ k = i-1 ∨ l = j+1 ∨ l = j-1) ∧ k < m ∧ l < n → true)

-- Part (a): 8x9 rectangle
theorem Sasha_can_paint_8x9 : (initially_painted 8 9 ∧ odd_painted_neighbors 8 9) → ∀ (i j : ℕ), i < 8 ∧ j < 9 :=
by
  -- Proof here
  sorry

-- Part (b): 8x10 rectangle
theorem Sasha_cannot_paint_8x10 : (initially_painted 8 10 ∧ odd_painted_neighbors 8 10) → ¬ (∀ (i j : ℕ), i < 8 ∧ j < 10) :=
by
  -- Proof here
  sorry

end Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l227_227997


namespace jean_total_calories_l227_227322

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l227_227322


namespace set_intersection_A_B_l227_227147

def A := {x : ℝ | 2 * x - x^2 > 0}
def B := {x : ℝ | x > 1}
def I := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_A_B :
  A ∩ B = I :=
sorry

end set_intersection_A_B_l227_227147


namespace Amanda_money_left_l227_227767

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l227_227767


namespace gcd_765432_654321_l227_227566

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227566


namespace coefficient_x4_in_expansion_l227_227291

theorem coefficient_x4_in_expansion:
  (∑ i in Finset.range (6), (binomial 5 i) * (if i = 2 then 3 else -1) * (if i = 4 then 1 else 0)) = 25 :=
by
  sorry

end coefficient_x4_in_expansion_l227_227291


namespace horse_catch_up_l227_227837

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end horse_catch_up_l227_227837


namespace students_no_A_l227_227311

theorem students_no_A
  (total_students : ℕ)
  (A_in_English : ℕ)
  (A_in_math : ℕ)
  (A_in_both : ℕ)
  (total_students_eq : total_students = 40)
  (A_in_English_eq : A_in_English = 10)
  (A_in_math_eq : A_in_math = 18)
  (A_in_both_eq : A_in_both = 6) :
  total_students - ((A_in_English + A_in_math) - A_in_both) = 18 :=
by
  sorry

end students_no_A_l227_227311


namespace axis_of_symmetry_of_quadratic_l227_227352

theorem axis_of_symmetry_of_quadratic (m : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * m * x - m^2 + 3 = -x^2 + 2 * m * x - m^2 + 3) ∧ (∃ x : ℝ, x + 2 = 0) → m = -2 :=
by
  sorry

end axis_of_symmetry_of_quadratic_l227_227352


namespace reciprocal_2023_l227_227368

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l227_227368


namespace compare_xyz_l227_227797

noncomputable def x := (0.5 : ℝ)^(0.5 : ℝ)
noncomputable def y := (0.5 : ℝ)^(1.3 : ℝ)
noncomputable def z := (1.3 : ℝ)^(0.5 : ℝ)

theorem compare_xyz : z > x ∧ x > y := by
  sorry

end compare_xyz_l227_227797


namespace problem_l227_227152

theorem problem (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 1) 
  (h3 : a + c + d = 16) 
  (h4 : b + c + d = 9) : 
  a * b + c * d = 734 / 9 := 
by 
  sorry

end problem_l227_227152


namespace gcd_765432_654321_l227_227597

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227597


namespace factors_of_48_are_multiples_of_6_l227_227481

theorem factors_of_48_are_multiples_of_6 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d, d ∣ 48 → (6 ∣ d ↔ d = 6 ∨ d = 12 ∨ d = 24 ∨ d = 48) := 
by { sorry }

end factors_of_48_are_multiples_of_6_l227_227481


namespace positive_value_of_m_l227_227827

theorem positive_value_of_m (m : ℝ) (h : (64 * m^2 - 60 * m) = 0) : m = 15 / 16 :=
sorry

end positive_value_of_m_l227_227827


namespace total_cost_l227_227716

-- Define the given conditions
def total_tickets : Nat := 10
def discounted_tickets : Nat := 4
def full_price : ℝ := 2.00
def discounted_price : ℝ := 1.60

-- Calculation of the total cost Martin spent
theorem total_cost : (discounted_tickets * discounted_price) + ((total_tickets - discounted_tickets) * full_price) = 18.40 := by
  sorry

end total_cost_l227_227716


namespace family_has_11_eggs_l227_227760

def initialEggs : ℕ := 10
def eggsUsed : ℕ := 5
def chickens : ℕ := 2
def eggsPerChicken : ℕ := 3

theorem family_has_11_eggs :
  (initialEggs - eggsUsed) + (chickens * eggsPerChicken) = 11 := by
  sorry

end family_has_11_eggs_l227_227760


namespace area_of_isosceles_trapezoid_l227_227996

theorem area_of_isosceles_trapezoid (R α : ℝ) (hR : R > 0) (hα1 : 0 < α) (hα2 : α < π) :
  let a := 2 * R
  let b := 2 * R * Real.sin (α / 2)
  let h := R * Real.cos (α / 2)
  (1 / 2) * (a + b) * h = R^2 * (1 + Real.sin (α / 2)) * Real.cos (α / 2) :=
by
  sorry

end area_of_isosceles_trapezoid_l227_227996


namespace gcd_765432_654321_l227_227567

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l227_227567


namespace gcd_765432_654321_l227_227548

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227548


namespace min_f_x_gt_2_solve_inequality_l227_227796

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 / (x + b)

theorem min_f_x_gt_2 (a b : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
∃ c, ∀ x > 2, f a b x ≥ c ∧ (∀ y, y > 2 → f a b y = c → y = 4 ∧ c = 8) :=
sorry

theorem solve_inequality (a b k : ℝ) (x : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
  f a b x < (k * (x - 1) + 1 - x^2) / (2 - x) ↔ 
  (x < 2 ∧ k = 0) ∨ 
  (-1 < k ∧ k < 0 ∧ 1 - 1 / k < x ∧ x < 2) ∨ 
  ((k > 0 ∨ k < -1) ∧ (1 - 1 / k < x ∧ x < 2) ∨ x > 2) ∨ 
  (k = -1 ∧ x ≠ 2) :=
sorry

end min_f_x_gt_2_solve_inequality_l227_227796


namespace count_even_numbers_is_320_l227_227480

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end count_even_numbers_is_320_l227_227480


namespace exponent_of_5_in_30_factorial_l227_227178

theorem exponent_of_5_in_30_factorial: 
  ∃ (n : ℕ), prime n ∧ n = 5 → 
  (∃ (e : ℕ), (e = (30 / 5).floor + (30 / 25).floor) ∧ 
  (∃ d : ℕ, d = 30! → ord n d = e)) :=
by sorry

end exponent_of_5_in_30_factorial_l227_227178


namespace ap_sub_aq_l227_227531

variable {n : ℕ} (hn : n > 0)

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) (hn : n > 0) : ℕ :=
S n - S (n - 1)

theorem ap_sub_aq (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : p - q = 5) :
  a p hp - a q hq = 20 :=
sorry

end ap_sub_aq_l227_227531


namespace sum_of_digits_of_triangular_number_2010_l227_227428

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_triangular_number_2010 (N : ℕ)
  (h₁ : triangular_number N = 2010) :
  sum_of_digits N = 9 :=
sorry

end sum_of_digits_of_triangular_number_2010_l227_227428


namespace area_ratio_of_circles_l227_227815

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l227_227815


namespace area_triangle_sum_l227_227830

theorem area_triangle_sum (AB : ℝ) (angle_BAC angle_ABC angle_ACB angle_EDC : ℝ) 
  (h_AB : AB = 1) (h_angle_BAC : angle_BAC = 70) (h_angle_ABC : angle_ABC = 50) 
  (h_angle_ACB : angle_ACB = 60) (h_angle_EDC : angle_EDC = 80) :
  let area_triangle := (1/2) * AB * (Real.sin angle_70 / Real.sin angle_60) * (Real.sin angle_60) 
  let area_CDE := (1/2) * (Real.sin angle_80)
  area_triangle + 2 * area_CDE = (Real.sin angle_70 + Real.sin angle_80) / 2 :=
sorry

end area_triangle_sum_l227_227830


namespace solution_set_of_inequality_l227_227031

theorem solution_set_of_inequality : 
  { x : ℝ | x^2 - 3*x - 4 < 0 } = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end solution_set_of_inequality_l227_227031


namespace find_obtuse_angle_l227_227464

-- Define the conditions
def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180

-- Lean statement assuming the needed conditions
theorem find_obtuse_angle (α : ℝ) (h1 : is_obtuse α) (h2 : 4 * α = 360 + α) : α = 120 :=
by sorry

end find_obtuse_angle_l227_227464


namespace points_enclosed_in_circle_l227_227096

open Set

variable (points : Set (ℝ × ℝ))
variable (radius : ℝ)
variable (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points → 
  ∃ (c : ℝ × ℝ), dist c A ≤ radius ∧ dist c B ≤ radius ∧ dist c C ≤ radius)

theorem points_enclosed_in_circle
  (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points →
    ∃ (c : ℝ × ℝ), dist c A ≤ 1 ∧ dist c B ≤ 1 ∧ dist c C ≤ 1) :
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ points → dist c p ≤ 1 :=
sorry

end points_enclosed_in_circle_l227_227096


namespace cos_double_angle_l227_227129

theorem cos_double_angle (α β : ℝ) (h1 : Real.sin (α - β) = 1 / 3) (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_double_angle_l227_227129


namespace cost_price_percentage_l227_227200

variables (CP MP SP : ℝ) (x : ℝ)

theorem cost_price_percentage (h1 : CP = (x / 100) * MP)
                             (h2 : SP = 0.5 * MP)
                             (h3 : SP = 2 * CP) :
                             x = 25 := by
  sorry

end cost_price_percentage_l227_227200


namespace Mrs_Early_speed_l227_227342

noncomputable def speed_to_reach_on_time (distance : ℝ) (ideal_time : ℝ) : ℝ := distance / ideal_time

theorem Mrs_Early_speed:
  ∃ (d t : ℝ), 
    (d = 50 * (t + 5/60)) ∧ 
    (d = 80 * (t - 7/60)) ∧ 
    (speed_to_reach_on_time d t = 59) := sorry

end Mrs_Early_speed_l227_227342


namespace distinct_real_roots_of_quadratic_l227_227700

theorem distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, x^2 - 4*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂)) ↔ m < 2 := by
sorry

end distinct_real_roots_of_quadratic_l227_227700


namespace largest_number_of_four_consecutive_whole_numbers_l227_227791

theorem largest_number_of_four_consecutive_whole_numbers 
  (a : ℕ) (h1 : a + (a + 1) + (a + 2) = 184)
  (h2 : a + (a + 1) + (a + 3) = 201)
  (h3 : a + (a + 2) + (a + 3) = 212)
  (h4 : (a + 1) + (a + 2) + (a + 3) = 226) : 
  a + 3 = 70 := 
by sorry

end largest_number_of_four_consecutive_whole_numbers_l227_227791


namespace equal_sets_implies_value_of_m_l227_227003

theorem equal_sets_implies_value_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {3, m}) (hB : B = {3 * m, 3}) (hAB : A = B) : m = 0 :=
by
  -- Proof goes here
  sorry

end equal_sets_implies_value_of_m_l227_227003


namespace points_equidistant_from_circle_and_tangents_l227_227629

noncomputable def circle_radius := 4
noncomputable def tangent_distance := 6

theorem points_equidistant_from_circle_and_tangents :
  ∃! (P : ℝ × ℝ), dist P (0, 0) = circle_radius ∧
                 dist P (0, tangent_distance) = tangent_distance - circle_radius ∧
                 dist P (0, -tangent_distance) = tangent_distance - circle_radius :=
by {
  sorry
}

end points_equidistant_from_circle_and_tangents_l227_227629


namespace yellow_jelly_bean_probability_l227_227634

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

end yellow_jelly_bean_probability_l227_227634


namespace seq_100_eq_11_div_12_l227_227439

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 3
  else if n ≥ 3 then (2 - seq (n - 1)) / (3 * seq (n - 2) + 1)
  else 0 -- This line handles the case n < 1, but shouldn't ever be used in practice.

theorem seq_100_eq_11_div_12 : seq 100 = 11 / 12 :=
  sorry

end seq_100_eq_11_div_12_l227_227439


namespace stratified_sampling_admin_staff_count_l227_227097

theorem stratified_sampling_admin_staff_count
  (total_staff : ℕ)
  (admin_staff : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 160)
  (h_admin : admin_staff = 32)
  (h_sample : sample_size = 20) :
  admin_staff * sample_size / total_staff = 4 :=
by
  sorry

end stratified_sampling_admin_staff_count_l227_227097


namespace labor_union_tree_equation_l227_227235

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end labor_union_tree_equation_l227_227235


namespace great_dane_weight_l227_227099

def weight_problem (C P G : ℝ) : Prop :=
  (P = 3 * C) ∧ (G = 3 * P + 10) ∧ (C + P + G = 439)

theorem great_dane_weight : ∃ (C P G : ℝ), weight_problem C P G ∧ G = 307 :=
by
  sorry

end great_dane_weight_l227_227099


namespace percentage_increase_l227_227353

theorem percentage_increase (N P : ℕ) (h1 : N = 40)
       (h2 : (N + (P / 100) * N) - (N - (30 / 100) * N) = 22) : P = 25 :=
by 
  have p1 := h1
  have p2 := h2
  sorry

end percentage_increase_l227_227353


namespace find_a_of_inequality_solution_set_l227_227309

theorem find_a_of_inequality_solution_set :
  (∃ (a : ℝ), (∀ (x : ℝ), |2*x - a| + a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ a = 1) :=
by sorry

end find_a_of_inequality_solution_set_l227_227309


namespace dice_surface_sum_l227_227405

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l227_227405


namespace part1_solution_sets_part2_solution_set_l227_227956

-- Define the function f(x)
def f (a x : ℝ) := x^2 + (1 - a) * x - a

-- Statement for part (1)
theorem part1_solution_sets (a x : ℝ) :
  (a < -1 → f a x < 0 ↔ a < x ∧ x < -1) ∧
  (a = -1 → ¬ (f a x < 0)) ∧
  (a > -1 → f a x < 0 ↔ -1 < x ∧ x < a) :=
sorry

-- Statement for part (2)
theorem part2_solution_set (x : ℝ) :
  (f 2 x) > 0 → (x^3 * f 2 x > 0 ↔ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end part1_solution_sets_part2_solution_set_l227_227956


namespace quadratic_roots_l227_227732

theorem quadratic_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + k*x1 + (k - 1) = 0) ∧ (x2^2 + k*x2 + (k - 1) = 0) :=
by
  sorry

end quadratic_roots_l227_227732


namespace magnitude_of_z_l227_227276

open Complex

theorem magnitude_of_z {z : ℂ} (h : z * (1 + I) = 1 - I) : abs z = 1 :=
sorry

end magnitude_of_z_l227_227276


namespace complement_U_A_l227_227293

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | |x - 1| > 1 }

theorem complement_U_A : (U \ A) = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l227_227293


namespace gcd_765432_654321_l227_227550

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l227_227550


namespace acid_solution_l227_227884

theorem acid_solution (n y : ℝ) (h : n > 30) (h1 : y = 15 * n / (n - 15)) :
  (n / 100) * n = ((n - 15) / 100) * (n + y) :=
by
  sorry

end acid_solution_l227_227884


namespace count_integer_values_l227_227032

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end count_integer_values_l227_227032


namespace sum_one_to_twenty_nine_l227_227083

theorem sum_one_to_twenty_nine : (29 / 2) * (1 + 29) = 435 := by
  -- proof
  sorry

end sum_one_to_twenty_nine_l227_227083


namespace minimum_m_value_l227_227809

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x + 1

theorem minimum_m_value :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (3 : ℝ) → x2 ∈ Set.Ici (3 : ℝ) → x1 ≠ x2 →
     ∃ a : ℝ, a ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧
     (f x1 a - f x2 a) / (x2 - x1) < m) →
  m ≥ -20 / 3 := sorry

end minimum_m_value_l227_227809


namespace gcd_765432_654321_l227_227598

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l227_227598


namespace sum_of_possible_values_l227_227427

-- Define the triangle's base and height
def triangle_base (x : ℝ) : ℝ := x - 2
def triangle_height (x : ℝ) : ℝ := x - 2

-- Define the parallelogram's base and height
def parallelogram_base (x : ℝ) : ℝ := x - 3
def parallelogram_height (x : ℝ) : ℝ := x + 4

-- Define the areas
def triangle_area (x : ℝ) : ℝ := 0.5 * (triangle_base x) * (triangle_height x)
def parallelogram_area (x : ℝ) : ℝ := (parallelogram_base x) * (parallelogram_height x)

-- Statement to prove
theorem sum_of_possible_values (x : ℝ) (h : parallelogram_area x = 3 * triangle_area x) : x = 8 ∨ x = 3 →
  (x = 8 ∨ x = 3) → 8 + 3 = 11 :=
by sorry

end sum_of_possible_values_l227_227427


namespace value_of_a_l227_227145

theorem value_of_a (a : ℝ) (h_neg : a < 0) (h_f : ∀ (x : ℝ), (0 < x ∧ x ≤ 1) → 
  (x + 4 * a / x - a < 0)) : a ≤ -1 / 3 := 
sorry

end value_of_a_l227_227145


namespace count_integers_satisfying_sqrt_condition_l227_227047

noncomputable def count_integers_in_range (lower upper : ℕ) : ℕ :=
    (upper - lower + 1)

/- Proof statement for the given problem -/
theorem count_integers_satisfying_sqrt_condition :
  let conditions := (∀ x : ℕ, 5 > Real.sqrt x ∧ Real.sqrt x > 3) in
  count_integers_in_range 10 24 = 15 :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l227_227047


namespace distance_P_to_AB_l227_227883

def point_P_condition (P : ℝ) : Prop :=
  P > 0 ∧ P < 1

def parallel_line_property (P : ℝ) (h : ℝ) : Prop :=
  h = 1 - P / 1

theorem distance_P_to_AB (P h : ℝ) (area_total : ℝ) (area_smaller : ℝ) :
  point_P_condition P →
  parallel_line_property P h →
  (area_smaller / area_total) = 1 / 3 →
  h = 2 / 3 :=
by
  intro hP hp hratio
  sorry

end distance_P_to_AB_l227_227883


namespace proof_line_eq_l227_227419

variable (a T : ℝ) (line : ℝ × ℝ → Prop)

def line_eq (point : ℝ × ℝ) : Prop := 
  point.2 = (-2 * T / a^2) * point.1 + (2 * T / a)

def correct_line_eq (point : ℝ × ℝ) : Prop :=
  -2 * T * point.1 + a^2 * point.2 + 2 * a * T = 0

theorem proof_line_eq :
  ∀ point : ℝ × ℝ, line_eq a T point ↔ correct_line_eq a T point :=
by
  sorry

end proof_line_eq_l227_227419


namespace cos_angle_value_l227_227121

noncomputable def cos_angle := Real.cos (19 * Real.pi / 4)

theorem cos_angle_value : cos_angle = -Real.sqrt 2 / 2 := by
  sorry

end cos_angle_value_l227_227121


namespace total_clips_correct_l227_227856

def clips_in_april : ℕ := 48
def clips_in_may : ℕ := clips_in_april / 2
def total_clips : ℕ := clips_in_april + clips_in_may

theorem total_clips_correct : total_clips = 72 := by
  sorry

end total_clips_correct_l227_227856


namespace parabola_shifted_left_and_down_l227_227206

-- Define the original parabolic equation
def original_parabola (x : ℝ) : ℝ := 2 * x ^ 2 - 1

-- Define the transformed parabolic equation
def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 1) ^ 2 - 3

-- Theorem statement
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 1) ^ 2 - 3 :=
by 
  -- Proof Left as an exercise.
  sorry

end parabola_shifted_left_and_down_l227_227206


namespace study_time_for_average_l227_227998

theorem study_time_for_average
    (study_time_exam1 score_exam1 : ℕ)
    (study_time_exam2 score_exam2 average_score desired_average : ℝ)
    (relation : score_exam1 = 20 * study_time_exam1)
    (direct_relation : score_exam2 = 20 * study_time_exam2)
    (total_exams : ℕ)
    (average_condition : (score_exam1 + score_exam2) / total_exams = desired_average) :
    study_time_exam2 = 4.5 :=
by
  have : total_exams = 2 := by sorry
  have : score_exam1 = 60 := by sorry
  have : desired_average = 75 := by sorry
  have : score_exam2 = 90 := by sorry
  sorry

end study_time_for_average_l227_227998


namespace fair_game_condition_l227_227219

variables (n : ℕ) (L : ℝ) {p : ℕ → ℝ}

-- Define the probability p_k for the k-th player.
def probability (k : ℕ) : ℝ := (35.0 / 36.0) ^ k

-- Define the expected value of the k-th player.
def expected_value (L : ℝ) (Lk : ℝ) (k : ℕ) : ℝ := L * probability k - Lk

-- Define the conditions of the fair game.
def fair_game := ∀ k, expected_value L (L * probability k) k = 0

-- Main theorem stating that the game is fair if stakes decrease proportionally by a factor of 35/36.
theorem fair_game_condition (k : ℕ) (L : ℝ) :
  fair_game :=
by
  sorry

end fair_game_condition_l227_227219


namespace merchant_cost_price_l227_227635

theorem merchant_cost_price (x : ℝ) (h₁ : x + (x^2 / 100) = 39) : x = 30 :=
sorry

end merchant_cost_price_l227_227635
