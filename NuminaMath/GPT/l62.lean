import Mathlib

namespace set_equality_l62_62280

theorem set_equality (A : Set ℕ) (h : {1} ∪ A = {1, 3, 5}) : 
  A = {1, 3, 5} ∨ A = {3, 5} :=
  sorry

end set_equality_l62_62280


namespace c_ge_a_plus_b_sin_half_C_l62_62781

-- Define a triangle with sides a, b, and c opposite to angles A, B, and C respectively, with C being the angle at vertex C
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)
  (angles_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (angles_sum : A + B + C = π)

namespace TriangleProveInequality

open Triangle

theorem c_ge_a_plus_b_sin_half_C (t : Triangle) :
  t.c ≥ (t.a + t.b) * Real.sin (t.C / 2) := sorry

end TriangleProveInequality

end c_ge_a_plus_b_sin_half_C_l62_62781


namespace symmetric_point_correct_l62_62392

def point : Type := ℝ × ℝ × ℝ

def symmetric_with_respect_to_y_axis (A : point) : point :=
  let (x, y, z) := A
  (-x, y, z)

def A : point := (-4, 8, 6)

theorem symmetric_point_correct :
  symmetric_with_respect_to_y_axis A = (4, 8, 6) := by
  sorry

end symmetric_point_correct_l62_62392


namespace eggs_per_day_second_store_l62_62608

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozen eggs supplied to the first store each day
def dozen_per_day_first_store : ℕ := 5

-- Define the number of eggs supplied to the first store each day
def eggs_per_day_first_store : ℕ := dozen_per_day_first_store * eggs_in_a_dozen

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Calculate the weekly supply to the first store
def weekly_supply_first_store : ℕ := eggs_per_day_first_store * days_in_week

-- Define the total weekly supply to both stores
def total_weekly_supply : ℕ := 630

-- Calculate the weekly supply to the second store
def weekly_supply_second_store : ℕ := total_weekly_supply - weekly_supply_first_store

-- Define the theorem to prove the number of eggs supplied to the second store each day
theorem eggs_per_day_second_store : weekly_supply_second_store / days_in_week = 30 := by
  sorry

end eggs_per_day_second_store_l62_62608


namespace Carmela_difference_l62_62499

theorem Carmela_difference (Cecil Catherine Carmela : ℤ) (X : ℤ) (h1 : Cecil = 600) 
(h2 : Catherine = 2 * Cecil - 250) (h3 : Carmela = 2 * Cecil + X) 
(h4 : Cecil + Catherine + Carmela = 2800) : X = 50 :=
by { sorry }

end Carmela_difference_l62_62499


namespace find_f_a_plus_1_l62_62210

def f (x : ℝ) : ℝ := x^2 + 1

theorem find_f_a_plus_1 (a : ℝ) : f (a + 1) = a^2 + 2 * a + 2 := by
  sorry

end find_f_a_plus_1_l62_62210


namespace maria_waist_size_in_cm_l62_62249

noncomputable def waist_size_in_cm (waist_size_inches : ℕ) (extra_inch : ℕ) (inches_per_foot : ℕ) (cm_per_foot : ℕ) : ℚ :=
  let total_inches := waist_size_inches + extra_inch
  let total_feet := (total_inches : ℚ) / inches_per_foot
  total_feet * cm_per_foot

theorem maria_waist_size_in_cm :
  waist_size_in_cm 28 1 12 31 = 74.9 :=
by
  sorry

end maria_waist_size_in_cm_l62_62249


namespace ratio_Nikki_to_Michael_l62_62517

theorem ratio_Nikki_to_Michael
  (M Joyce Nikki Ryn : ℕ)
  (h1 : Joyce = M + 2)
  (h2 : Nikki = 30)
  (h3 : Ryn = (4 / 5) * Nikki)
  (h4 : M + Joyce + Nikki + Ryn = 76) :
  Nikki / M = 3 :=
by {
  sorry
}

end ratio_Nikki_to_Michael_l62_62517


namespace seokgi_jumped_furthest_l62_62999

noncomputable def yooseung_jump : ℝ := 15 / 8
def shinyoung_jump : ℝ := 2
noncomputable def seokgi_jump : ℝ := 17 / 8

theorem seokgi_jumped_furthest :
  yooseung_jump < seokgi_jump ∧ shinyoung_jump < seokgi_jump :=
by
  sorry

end seokgi_jumped_furthest_l62_62999


namespace union_complement_eq_set_l62_62699

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end union_complement_eq_set_l62_62699


namespace average_rate_of_change_l62_62875

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l62_62875


namespace total_apples_picked_l62_62916

def Mike_apples : ℕ := 7
def Nancy_apples : ℕ := 3
def Keith_apples : ℕ := 6
def Jennifer_apples : ℕ := 5
def Tom_apples : ℕ := 8
def Stacy_apples : ℕ := 4

theorem total_apples_picked : 
  Mike_apples + Nancy_apples + Keith_apples + Jennifer_apples + Tom_apples + Stacy_apples = 33 :=
by
  sorry

end total_apples_picked_l62_62916


namespace mixture_weight_l62_62876

theorem mixture_weight :
  let weight_a_per_liter := 900 -- in gm
  let weight_b_per_liter := 750 -- in gm
  let ratio_a := 3
  let ratio_b := 2
  let total_volume := 4 -- in liters
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  let total_weight_kg := total_weight_gm / 1000 
  total_weight_kg = 3.36 :=
by
  sorry

end mixture_weight_l62_62876


namespace number_of_multiples_of_6_between_5_and_125_l62_62244

theorem number_of_multiples_of_6_between_5_and_125 : 
  ∃ k : ℕ, (5 < 6 * k ∧ 6 * k < 125) → k = 20 :=
sorry

end number_of_multiples_of_6_between_5_and_125_l62_62244


namespace function_passes_through_fixed_point_l62_62560

theorem function_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : (2 - a^(0 : ℝ) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l62_62560


namespace find_a_l62_62496

/-- Given function -/
def f (x: ℝ) : ℝ := (x + 1)^2 - 2 * (x + 1)

/-- Problem statement -/
theorem find_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := 
by
  sorry

end find_a_l62_62496


namespace Christina_driving_time_l62_62085

theorem Christina_driving_time 
  (speed_Christina : ℕ) 
  (speed_friend : ℕ) 
  (total_distance : ℕ)
  (friend_driving_time : ℕ) 
  (distance_by_Christina : ℕ) 
  (time_driven_by_Christina : ℕ) 
  (total_driving_time : ℕ)
  (h1 : speed_Christina = 30)
  (h2 : speed_friend = 40) 
  (h3 : total_distance = 210)
  (h4 : friend_driving_time = 3)
  (h5 : speed_friend * friend_driving_time = 120)
  (h6 : total_distance - 120 = distance_by_Christina)
  (h7 : distance_by_Christina = 90)
  (h8 : distance_by_Christina / speed_Christina = 3)
  (h9 : time_driven_by_Christina = 3)
  (h10 : time_driven_by_Christina * 60 = 180) :
    total_driving_time = 180 := 
by
  sorry

end Christina_driving_time_l62_62085


namespace sqrt_of_9_l62_62432

theorem sqrt_of_9 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_l62_62432


namespace find_remainder_l62_62346

theorem find_remainder (x : ℤ) (h : 0 < x ∧ 7 * x % 26 = 1) : (13 + 3 * x) % 26 = 6 :=
sorry

end find_remainder_l62_62346


namespace possible_original_numbers_l62_62300

def four_digit_original_number (N : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    N = 1000 * a + 100 * b + 10 * c + d ∧ 
    (a+1) * (b+2) * (c+3) * (d+4) = 234 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem possible_original_numbers : 
  four_digit_original_number 1109 ∨ four_digit_original_number 2009 :=
sorry

end possible_original_numbers_l62_62300


namespace find_two_digit_ab_l62_62503

def digit_range (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def different_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_two_digit_ab (A B C D : ℕ) (hA : digit_range A) (hB : digit_range B)
                         (hC : digit_range C) (hD : digit_range D)
                         (h_diff : different_digits A B C D)
                         (h_eq : (100 * A + 10 * B + C) * (10 * A + B) + C * D = 2017) :
  10 * A + B = 14 :=
sorry

end find_two_digit_ab_l62_62503


namespace acute_angle_probability_l62_62561

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

end acute_angle_probability_l62_62561


namespace first_digit_base9_650_l62_62735

theorem first_digit_base9_650 : ∃ d : ℕ, 
  d = 8 ∧ (∃ k : ℕ, 650 = d * 9^2 + k ∧ k < 9^2) :=
by {
  sorry
}

end first_digit_base9_650_l62_62735


namespace average_of_first_16_even_numbers_l62_62663

theorem average_of_first_16_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30 + 32) / 16 = 17 := 
by sorry

end average_of_first_16_even_numbers_l62_62663


namespace mass_percentage_Al_in_AlBr₃_l62_62616

theorem mass_percentage_Al_in_AlBr₃ :
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  (Al_mass / M_AlBr₃ * 100) = 10.11 :=
by 
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  have : (Al_mass / M_AlBr₃ * 100) = 10.11 := sorry
  assumption

end mass_percentage_Al_in_AlBr₃_l62_62616


namespace no_rational_roots_of_odd_coefficient_quadratic_l62_62704

theorem no_rational_roots_of_odd_coefficient_quadratic 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, r * r * a + r * b + c = 0 :=
by
  sorry

end no_rational_roots_of_odd_coefficient_quadratic_l62_62704


namespace amount_paid_l62_62857

def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def change_received : ℕ := 11

theorem amount_paid (h_cost : ℕ := hamburger_cost) (o_cost : ℕ := onion_rings_cost) (s_cost : ℕ := smoothie_cost) (change : ℕ := change_received) :
  h_cost + o_cost + s_cost + change = 20 := by
  sorry

end amount_paid_l62_62857


namespace second_order_arithmetic_sequence_a30_l62_62891

theorem second_order_arithmetic_sequence_a30 {a : ℕ → ℝ}
  (h₁ : ∀ n, a (n + 1) - a n - (a (n + 2) - a (n + 1)) = 20)
  (h₂ : a 10 = 23)
  (h₃ : a 20 = 23) :
  a 30 = 2023 := 
sorry

end second_order_arithmetic_sequence_a30_l62_62891


namespace area_of_given_polygon_l62_62482

def point := (ℝ × ℝ)

def vertices : List point := [(0,0), (5,0), (5,2), (3,2), (3,3), (2,3), (2,2), (0,2), (0,0)]

def polygon_area (vertices : List point) : ℝ := 
  -- Function to compute the area of the given polygon
  -- Implementation of the area computation is assumed to be correct
  sorry

theorem area_of_given_polygon : polygon_area vertices = 11 :=
sorry

end area_of_given_polygon_l62_62482


namespace probability_of_at_most_3_heads_l62_62874

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l62_62874


namespace part1_part2_l62_62570

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (x : ℝ) : f x 1 >= f x 1 := sorry

theorem part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) : b / a ≥ 0 := sorry

end part1_part2_l62_62570


namespace find_a_range_l62_62268

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- The main theorem stating the range of a
theorem find_a_range (a : ℝ) (h : ¬(∃ x : ℝ, p a x) → ¬(∃ x : ℝ, q x) ∧ ¬(¬(∃ x : ℝ, q x) → ¬(∃ x : ℝ, p a x))) : 1 < a ∧ a ≤ 2 := sorry

end find_a_range_l62_62268


namespace cos_double_angle_l62_62340
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end cos_double_angle_l62_62340


namespace pages_in_second_chapter_l62_62800

theorem pages_in_second_chapter
  (total_pages : ℕ)
  (first_chapter_pages : ℕ)
  (second_chapter_pages : ℕ)
  (h1 : total_pages = 93)
  (h2 : first_chapter_pages = 60)
  (h3: second_chapter_pages = total_pages - first_chapter_pages) :
  second_chapter_pages = 33 :=
by
  sorry

end pages_in_second_chapter_l62_62800


namespace buratino_spent_dollars_l62_62260

theorem buratino_spent_dollars (x y : ℕ) (h1 : x + y = 50) (h2 : 2 * x = 3 * y) : 
  (y * 5 - x * 3) = 10 :=
by
  sorry

end buratino_spent_dollars_l62_62260


namespace money_problem_solution_l62_62415

theorem money_problem_solution (a b : ℝ) (h1 : 7 * a + b < 100) (h2 : 4 * a - b = 40) (h3 : b = 0.5 * a) : 
  a = 80 / 7 ∧ b = 40 / 7 :=
by
  sorry

end money_problem_solution_l62_62415


namespace min_vertical_segment_length_l62_62927

noncomputable def f₁ (x : ℝ) : ℝ := |x|
noncomputable def f₂ (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem min_vertical_segment_length :
  ∃ m : ℝ, m = 3 ∧
            ∀ x : ℝ, abs (f₁ x - f₂ x) ≥ m :=
sorry

end min_vertical_segment_length_l62_62927


namespace oliver_cards_l62_62738

variable {MC AB BG : ℕ}

theorem oliver_cards : 
  (BG = 48) → 
  (BG = 3 * AB) → 
  (MC = 2 * AB) → 
  MC = 32 := 
by 
  intros h1 h2 h3
  sorry

end oliver_cards_l62_62738


namespace sum_of_consecutive_integers_with_product_272_l62_62274

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end sum_of_consecutive_integers_with_product_272_l62_62274


namespace mowing_time_l62_62996

/-- 
Rena uses a mower to trim her "L"-shaped lawn which consists of two rectangular sections 
sharing one $50$-foot side. One section is $120$-foot by $50$-foot and the other is $70$-foot by 
$50$-foot. The mower has a swath width of $35$ inches with overlaps by $5$ inches. 
Rena walks at the rate of $4000$ feet per hour. 
Prove that it takes 0.95 hours for Rena to mow the entire lawn.
-/
theorem mowing_time 
  (length1 length2 width mower_swath overlap : ℝ) 
  (Rena_speed : ℝ) (effective_swath : ℝ) (total_area total_strips total_distance : ℝ)
  (h1 : length1 = 120)
  (h2 : length2 = 70)
  (h3 : width = 50)
  (h4 : mower_swath = 35 / 12)
  (h5 : overlap = 5 / 12)
  (h6 : effective_swath = mower_swath - overlap)
  (h7 : Rena_speed = 4000)
  (h8 : total_area = length1 * width + length2 * width)
  (h9 : total_strips = (length1 + length2) / effective_swath)
  (h10 : total_distance = total_strips * width) : 
  (total_distance / Rena_speed = 0.95) :=
by sorry

end mowing_time_l62_62996


namespace arithmetic_sequence_common_diff_l62_62103

theorem arithmetic_sequence_common_diff (d : ℝ) (a : ℕ → ℝ) 
  (h_first_term : a 0 = 24) 
  (h_arithmetic_sequence : ∀ n, a (n + 1) = a n + d)
  (h_ninth_term_nonneg : 24 + 8 * d ≥ 0) 
  (h_tenth_term_neg : 24 + 9 * d < 0) : 
  -3 ≤ d ∧ d < -8/3 :=
by 
  sorry

end arithmetic_sequence_common_diff_l62_62103


namespace crayons_loss_difference_l62_62256

theorem crayons_loss_difference (crayons_given crayons_lost : ℕ) 
  (h_given : crayons_given = 90) 
  (h_lost : crayons_lost = 412) : 
  crayons_lost - crayons_given = 322 :=
by
  sorry

end crayons_loss_difference_l62_62256


namespace quadratic_has_single_real_root_l62_62681

theorem quadratic_has_single_real_root (n : ℝ) (h : (6 * n) ^ 2 - 4 * 1 * (2 * n) = 0) : n = 2 / 9 :=
by
  sorry

end quadratic_has_single_real_root_l62_62681


namespace find_b_plus_c_l62_62908

theorem find_b_plus_c (a b c d : ℝ) 
    (h₁ : a + d = 6) 
    (h₂ : a * b + a * c + b * d + c * d = 40) : 
    b + c = 20 / 3 := 
sorry

end find_b_plus_c_l62_62908


namespace deluxe_stereo_time_fraction_l62_62599

theorem deluxe_stereo_time_fraction (S : ℕ) (B : ℝ)
  (H1 : 2 / 3 > 0)
  (H2 : 1.6 > 0) :
  (1.6 / 3 * S * B) / (1.2 * S * B) = 4 / 9 :=
by
  sorry

end deluxe_stereo_time_fraction_l62_62599


namespace minimum_value_l62_62188

theorem minimum_value {a b c : ℝ} (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * b * c = 1 / 2) :
  ∃ x, x = a^2 + 4 * a * b + 9 * b^2 + 8 * b * c + 3 * c^2 ∧ x = 13.5 :=
sorry

end minimum_value_l62_62188


namespace probability_of_interval_is_one_third_l62_62030

noncomputable def probability_in_interval (total_start total_end inner_start inner_end : ℝ) : ℝ :=
  (inner_end - inner_start) / (total_end - total_start)

theorem probability_of_interval_is_one_third :
  probability_in_interval 1 7 5 8 = 1 / 3 :=
by
  sorry

end probability_of_interval_is_one_third_l62_62030


namespace total_pears_sold_l62_62135

theorem total_pears_sold (sold_morning : ℕ) (sold_afternoon : ℕ) (h_morning : sold_morning = 120) (h_afternoon : sold_afternoon = 240) :
  sold_morning + sold_afternoon = 360 :=
by
  sorry

end total_pears_sold_l62_62135


namespace solve_for_a_l62_62679

theorem solve_for_a (x a : ℤ) (h : 2 * x - a - 5 = 0) (hx : x = 3) : a = 1 :=
by sorry

end solve_for_a_l62_62679


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l62_62904

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l62_62904


namespace sum_leq_two_l62_62204

open Classical

theorem sum_leq_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 + b^3 = 2) : a + b ≤ 2 :=
by
  sorry

end sum_leq_two_l62_62204


namespace finite_solutions_to_equation_l62_62828

theorem finite_solutions_to_equation :
  ∃ n : ℕ, ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ (1 / (a:ℝ) + 1 / (b:ℝ) + 1 / (c:ℝ) = 1 / 1983) →
    a ≤ n ∧ b ≤ n ∧ c ≤ n :=
sorry

end finite_solutions_to_equation_l62_62828


namespace difference_of_numbers_l62_62671

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 22500) (h2 : b = 10 * a + 5) : b - a = 18410 :=
by
  sorry

end difference_of_numbers_l62_62671


namespace polynomial_even_iff_exists_Q_l62_62660

open Polynomial

noncomputable def exists_polynomial_Q (P : Polynomial ℂ) : Prop :=
  ∃ Q : Polynomial ℂ, ∀ z : ℂ, P.eval z = (Q.eval z) * (Q.eval (-z))

theorem polynomial_even_iff_exists_Q (P : Polynomial ℂ) :
  (∀ z : ℂ, P.eval z = P.eval (-z)) ↔ exists_polynomial_Q P :=
by 
  sorry

end polynomial_even_iff_exists_Q_l62_62660


namespace tan_sum_identity_l62_62826

theorem tan_sum_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.tan α + (1 / Real.tan α) = 3 :=
by
  sorry

end tan_sum_identity_l62_62826


namespace cost_of_four_pencils_and_four_pens_l62_62553

def pencil_cost : ℝ := sorry
def pen_cost : ℝ := sorry

axiom h1 : 8 * pencil_cost + 3 * pen_cost = 5.10
axiom h2 : 3 * pencil_cost + 5 * pen_cost = 4.95

theorem cost_of_four_pencils_and_four_pens : 4 * pencil_cost + 4 * pen_cost = 4.488 :=
by
  sorry

end cost_of_four_pencils_and_four_pens_l62_62553


namespace cylinder_lateral_surface_area_l62_62487

theorem cylinder_lateral_surface_area (r l : ℝ) (A : ℝ) (h_r : r = 1) (h_l : l = 2) : A = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l62_62487


namespace find_m_l62_62787

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l62_62787


namespace last_two_digits_sum_is_32_l62_62359

-- Definitions for digit representation
variables (z a r l m : ℕ)

-- Numbers definitions
def ZARAZA := z * 10^5 + a * 10^4 + r * 10^3 + a * 10^2 + z * 10 + a
def ALMAZ := a * 10^4 + l * 10^3 + m * 10^2 + a * 10 + z

-- Condition that ZARAZA is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Condition that ALMAZ is divisible by 28
def divisible_by_28 (n : ℕ) : Prop := n % 28 = 0

-- The theorem to prove
theorem last_two_digits_sum_is_32
  (hz4 : divisible_by_4 (ZARAZA z a r))
  (ha28 : divisible_by_28 (ALMAZ a l m z))
  : (ZARAZA z a r + ALMAZ a l m z) % 100 = 32 :=
by sorry

end last_two_digits_sum_is_32_l62_62359


namespace find_certain_number_l62_62524

theorem find_certain_number 
  (x : ℝ) 
  (h : ( (x + 2 - 6) * 3 ) / 4 = 3) 
  : x = 8 :=
by
  sorry

end find_certain_number_l62_62524


namespace total_spent_on_clothing_l62_62678

def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  -- Proof goes here.
  sorry

end total_spent_on_clothing_l62_62678


namespace original_prop_and_contrapositive_l62_62794

theorem original_prop_and_contrapositive (m : ℝ) (h : m > 0) : 
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0 ∨ ∃ x y : ℝ, x^2 + x - m = 0 ∧ y^2 + y - m = 0) :=
by
  sorry

end original_prop_and_contrapositive_l62_62794


namespace sin_half_alpha_l62_62559

theorem sin_half_alpha (α : ℝ) (h_cos : Real.cos α = -2/3) (h_range : π < α ∧ α < 3 * π / 2) :
  Real.sin (α / 2) = Real.sqrt 30 / 6 :=
by
  sorry

end sin_half_alpha_l62_62559


namespace relationship_between_y1_y2_y3_l62_62234

-- Define the parabola equation and points
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the points
def point1 := -2
def point2 := 0
def point3 := 5 / 3

-- Define the y values at these points
def y1 (c : ℝ) := parabola point1 c
def y2 (c : ℝ) := parabola point2 c
def y3 (c : ℝ) := parabola point3 c

-- Proof statement
theorem relationship_between_y1_y2_y3 (c : ℝ) : 
  y1 c > y2 c ∧ y2 c > y3 c :=
sorry

end relationship_between_y1_y2_y3_l62_62234


namespace hyperbola_eccentricity_is_2_l62_62881

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 4 * a
  let e := c / a
  e

theorem hyperbola_eccentricity_is_2
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = 2 := 
sorry

end hyperbola_eccentricity_is_2_l62_62881


namespace stratified_sampling_A_l62_62426

theorem stratified_sampling_A (A B C total_units : ℕ) (propA : A = 400) (propB : B = 300) (propC : C = 200) (units : total_units = 90) :
  let total_families := A + B + C
  let nA := (A * total_units) / total_families
  nA = 40 :=
by
  -- prove the theorem here
  sorry

end stratified_sampling_A_l62_62426


namespace arithmetic_sequence_sum_l62_62917

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1))
  (h2 : S 3 = 6)
  (h3 : a 3 = 3) :
  S 2023 / 2023 = 1012 := by
  sorry

end arithmetic_sequence_sum_l62_62917


namespace xyz_logarithm_sum_l62_62045

theorem xyz_logarithm_sum :
  ∃ (X Y Z : ℕ), X > 0 ∧ Y > 0 ∧ Z > 0 ∧
  Nat.gcd X (Nat.gcd Y Z) = 1 ∧ 
  (↑X * Real.log 3 / Real.log 180 + ↑Y * Real.log 5 / Real.log 180 = ↑Z) ∧ 
  (X + Y + Z = 4) :=
by
  sorry

end xyz_logarithm_sum_l62_62045


namespace diff_of_squares_l62_62404

theorem diff_of_squares (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
by
  sorry

end diff_of_squares_l62_62404


namespace opposite_2024_eq_neg_2024_l62_62414

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end opposite_2024_eq_neg_2024_l62_62414


namespace equality_or_neg_equality_of_eq_l62_62052

theorem equality_or_neg_equality_of_eq
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^3 / a = b^2 + a^3 / b) : a = b ∨ a = -b := 
  by
  sorry

end equality_or_neg_equality_of_eq_l62_62052


namespace sufficient_but_not_necessary_condition_l62_62182

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a < 1 / b) ∧ ¬ (1 / a < 1 / b → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l62_62182


namespace sparrow_population_decline_l62_62810

theorem sparrow_population_decline {P : ℕ} (initial_year : ℕ) (initial_population : ℕ) (decrease_by_half : ∀ year, year ≥ initial_year →  init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20) :
  ∃ year, year ≥ initial_year + 5 ∧ init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20 :=
by
  sorry

end sparrow_population_decline_l62_62810


namespace mom_younger_than_grandmom_l62_62932

def cara_age : ℕ := 40
def cara_younger_mom : ℕ := 20
def grandmom_age : ℕ := 75

def mom_age : ℕ := cara_age + cara_younger_mom
def age_difference : ℕ := grandmom_age - mom_age

theorem mom_younger_than_grandmom : age_difference = 15 := by
  sorry

end mom_younger_than_grandmom_l62_62932


namespace circle_passing_points_l62_62675

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l62_62675


namespace a_is_5_if_extreme_at_neg3_l62_62152

-- Define the function f with parameter a
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 3

-- Define the given condition that f reaches an extreme value at x = -3
def reaches_extreme_at (a : ℝ) : Prop := f_prime a (-3) = 0

-- Prove that a = 5 if f reaches an extreme value at x = -3
theorem a_is_5_if_extreme_at_neg3 : ∀ a : ℝ, reaches_extreme_at a → a = 5 :=
by
  intros a h
  -- Proof omitted
  sorry

end a_is_5_if_extreme_at_neg3_l62_62152


namespace simplify_fraction_l62_62867

theorem simplify_fraction :
  (1 / (3 / (Real.sqrt 5 + 2) + 4 / (Real.sqrt 7 - 2))) = (3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10)) :=
sorry

end simplify_fraction_l62_62867


namespace gwen_points_per_bag_l62_62974

theorem gwen_points_per_bag : 
  ∀ (total_bags recycled_bags total_points_per_bag points_per_bag : ℕ),
  total_bags = 4 → 
  recycled_bags = total_bags - 2 →
  total_points_per_bag = 16 →
  points_per_bag = (total_points_per_bag / total_bags) →
  points_per_bag = 4 :=
by
  intros
  sorry

end gwen_points_per_bag_l62_62974


namespace relationship_between_problems_geometry_problem_count_steve_questions_l62_62393

variable (x y W A G : ℕ)

def word_problems (x : ℕ) : ℕ := x / 2
def addition_and_subtraction_problems (x : ℕ) : ℕ := x / 3
def geometry_problems (x W A : ℕ) : ℕ := x - W - A

theorem relationship_between_problems :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x ∧
  G = geometry_problems x W A →
  W + A + G = x :=
by
  sorry

theorem geometry_problem_count :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x →
  G = geometry_problems x W A →
  G = x / 6 :=
by
  sorry

theorem steve_questions :
  y = x / 2 - 4 :=
by
  sorry

end relationship_between_problems_geometry_problem_count_steve_questions_l62_62393


namespace find_m_l62_62743

theorem find_m (x1 x2 m : ℝ) (h_eq : ∀ x, x^2 + x + m = 0 → (x = x1 ∨ x = x2))
  (h_abs : |x1| + |x2| = 3)
  (h_sum : x1 + x2 = -1)
  (h_prod : x1 * x2 = m) :
  m = -2 :=
sorry

end find_m_l62_62743


namespace hancho_milk_l62_62685

def initial_milk : ℝ := 1
def ye_seul_milk : ℝ := 0.1
def ga_young_milk : ℝ := ye_seul_milk + 0.2
def remaining_milk : ℝ := 0.3

theorem hancho_milk : (initial_milk - (ye_seul_milk + ga_young_milk + remaining_milk)) = 0.3 :=
by
  sorry

end hancho_milk_l62_62685


namespace domain_of_composed_function_l62_62632

theorem domain_of_composed_function
  (f : ℝ → ℝ)
  (dom_f : ∀ x, 0 ≤ x ∧ x ≤ 4 → f x ≠ 0) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → f (x^2) ≠ 0 :=
by
  sorry

end domain_of_composed_function_l62_62632


namespace determine_dresses_and_shoes_colors_l62_62774

variables (dress_color shoe_color : String → String)
variables (Tamara Valya Lida : String)

-- Conditions
def condition_1 : Prop := ∀ x : String, x ≠ Tamara → dress_color x ≠ shoe_color x
def condition_2 : Prop := shoe_color Valya = "white"
def condition_3 : Prop := dress_color Lida ≠ "red" ∧ shoe_color Lida ≠ "red"
def condition_4 : Prop := ∀ x : String, dress_color x ∈ ["white", "red", "blue"] ∧ shoe_color x ∈ ["white", "red", "blue"]

-- Desired conclusion
def conclusion : Prop :=
  dress_color Valya = "blue" ∧ shoe_color Valya = "white" ∧
  dress_color Lida = "white" ∧ shoe_color Lida = "blue" ∧
  dress_color Tamara = "red" ∧ shoe_color Tamara = "red"

theorem determine_dresses_and_shoes_colors
  (Tamara Valya Lida : String)
  (h1 : condition_1 dress_color shoe_color Tamara)
  (h2 : condition_2 shoe_color Valya)
  (h3 : condition_3 dress_color shoe_color Lida)
  (h4 : condition_4 dress_color shoe_color) :
  conclusion dress_color shoe_color Valya Lida Tamara :=
sorry

end determine_dresses_and_shoes_colors_l62_62774


namespace geometric_sequence_common_ratio_l62_62082

theorem geometric_sequence_common_ratio (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : a_n 3 = a_n 2 * q) 
  (h2 : a_n 2 * q - 3 * a_n 2 = 2) 
  (h3 : 5 * a_n 4 = (12 * a_n 3 + 2 * a_n 5) / 2) : 
  q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l62_62082


namespace eighth_term_of_arithmetic_sequence_l62_62309

noncomputable def arithmetic_sequence (n : ℕ) (a1 an : ℚ) (k : ℕ) : ℚ :=
  a1 + (k - 1) * ((an - a1) / (n - 1))

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a1 a30 : ℚ), a1 = 5 → a30 = 86 → 
  arithmetic_sequence 30 a1 a30 8 = 592 / 29 :=
by
  intros a1 a30 h_a1 h_a30
  rw [h_a1, h_a30]
  dsimp [arithmetic_sequence]
  sorry

end eighth_term_of_arithmetic_sequence_l62_62309


namespace minimum_rows_required_l62_62575

theorem minimum_rows_required (n : ℕ) : (3 * n * (n + 1)) / 2 ≥ 150 ↔ n ≥ 10 := 
by
  sorry

end minimum_rows_required_l62_62575


namespace geo_seq_sum_condition_l62_62397

noncomputable def geometric_seq (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^n

noncomputable def sum_geo_seq_3 (a : ℝ) (q : ℝ) : ℝ :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

noncomputable def sum_geo_seq_6 (a : ℝ) (q : ℝ) : ℝ :=
  sum_geo_seq_3 a q + geometric_seq a q 3 + geometric_seq a q 4 + geometric_seq a q 5

theorem geo_seq_sum_condition {a q S₃ S₆ : ℝ} (h_sum_eq : S₆ = 9 * S₃)
  (h_S₃_def : S₃ = sum_geo_seq_3 a q)
  (h_S₆_def : S₆ = sum_geo_seq_6 a q) :
  q = 2 :=
by
  sorry

end geo_seq_sum_condition_l62_62397


namespace arithmetic_sqrt_of_13_l62_62651

theorem arithmetic_sqrt_of_13 : Real.sqrt 13 = Real.sqrt 13 := by
  sorry

end arithmetic_sqrt_of_13_l62_62651


namespace problem_a_add_b_eq_five_l62_62528

variable {a b : ℝ}

theorem problem_a_add_b_eq_five
  (h1 : ∀ x, -2 < x ∧ x < 3 → ax^2 + x + b > 0)
  (h2 : a < 0) :
  a + b = 5 :=
sorry

end problem_a_add_b_eq_five_l62_62528


namespace golu_distance_after_turning_left_l62_62703

theorem golu_distance_after_turning_left :
  ∀ (a c b : ℝ), a = 8 → c = 10 → (c ^ 2 = a ^ 2 + b ^ 2) → b = 6 :=
by
  intros a c b ha hc hpyth
  rw [ha, hc] at hpyth
  sorry

end golu_distance_after_turning_left_l62_62703


namespace sin_segment_ratio_is_rel_prime_l62_62214

noncomputable def sin_segment_ratio : ℕ × ℕ :=
  let p := 1
  let q := 8
  (p, q)
  
theorem sin_segment_ratio_is_rel_prime :
  1 < 8 ∧ gcd 1 8 = 1 ∧ sin_segment_ratio = (1, 8) :=
by
  -- gcd 1 8 = 1
  have h1 : gcd 1 8 = 1 := by exact gcd_one_right 8
  -- 1 < 8
  have h2 : 1 < 8 := by decide
  -- final tuple
  have h3 : sin_segment_ratio = (1, 8) := by rfl
  exact ⟨h2, h1, h3⟩

end sin_segment_ratio_is_rel_prime_l62_62214


namespace friends_truth_l62_62982

-- Definitions for the truth values of the friends
def F₁_truth (a x₁ x₂ x₃ : Prop) : Prop := a ↔ ¬ (x₁ ∨ x₂ ∨ x₃)
def F₂_truth (b x₁ x₂ x₃ : Prop) : Prop := b ↔ (x₂ ∧ ¬ x₁ ∧ ¬ x₃)
def F₃_truth (c x₁ x₂ x₃ : Prop) : Prop := c ↔ x₃

-- Main theorem statement
theorem friends_truth (a b c x₁ x₂ x₃ : Prop) 
  (H₁ : F₁_truth a x₁ x₂ x₃) 
  (H₂ : F₂_truth b x₁ x₂ x₃) 
  (H₃ : F₃_truth c x₁ x₂ x₃)
  (H₄ : a ∨ b ∨ c) 
  (H₅ : ¬ (a ∧ b ∧ c)) : a ∧ ¬b ∧ ¬c ∨ ¬a ∧ b ∧ ¬c ∨ ¬a ∧ ¬b ∧ c :=
sorry

end friends_truth_l62_62982


namespace shooting_prob_l62_62041

theorem shooting_prob (p : ℝ) (h₁ : (1 / 3) * (1 / 2) * (1 - p) + (1 / 3) * (1 / 2) * p + (2 / 3) * (1 / 2) * p = 7 / 18) :
  p = 2 / 3 :=
sorry

end shooting_prob_l62_62041


namespace binom_multiplication_l62_62788

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l62_62788


namespace boys_other_communities_l62_62035

/-- 
In a school of 850 boys, 44% are Muslims, 28% are Hindus, 
10% are Sikhs, and the remaining belong to other communities.
Prove that the number of boys belonging to other communities is 153.
-/
theorem boys_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℚ)
  (h_total_boys : total_boys = 850)
  (h_percentage_muslims : percentage_muslims = 44)
  (h_percentage_hindus : percentage_hindus = 28)
  (h_percentage_sikhs : percentage_sikhs = 10) :
  let percentage_others := 100 - (percentage_muslims + percentage_hindus + percentage_sikhs)
  let number_others := (percentage_others / 100) * total_boys
  number_others = 153 := 
by
  sorry

end boys_other_communities_l62_62035


namespace tan_at_max_value_l62_62949

theorem tan_at_max_value : 
  ∃ x₀, (∀ x, 3 * Real.sin x₀ - 4 * Real.cos x₀ ≥ 3 * Real.sin x - 4 * Real.cos x) → Real.tan x₀ = 3/4 := 
sorry

end tan_at_max_value_l62_62949


namespace usual_time_is_75_l62_62358

variable (T : ℕ) -- let T be the usual time in minutes

theorem usual_time_is_75 (h1 : (6 * T) / 5 = T + 15) : T = 75 :=
by
  sorry

end usual_time_is_75_l62_62358


namespace acid_solution_mix_l62_62902

theorem acid_solution_mix (x : ℝ) (h₁ : 0.2 * x + 50 = 0.35 * (100 + x)) : x = 100 :=
by
  sorry

end acid_solution_mix_l62_62902


namespace minimum_value_quadratic_expression_l62_62043

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end minimum_value_quadratic_expression_l62_62043


namespace eq_infinite_solutions_pos_int_l62_62535

noncomputable def eq_has_inf_solutions_in_positive_integers (m : ℕ) : Prop :=
    ∀ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 → 
    ∃ (a' b' c' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem eq_infinite_solutions_pos_int (m : ℕ) (hm : m > 0) : eq_has_inf_solutions_in_positive_integers m := 
by 
  sorry

end eq_infinite_solutions_pos_int_l62_62535


namespace number_of_females_l62_62173

-- Definitions
variable (F : ℕ) -- ℕ = Natural numbers, ensuring F is a non-negative integer
variable (h_male : ℕ := 2 * F)
variable (h_total : F + 2 * F = 18)
variable (h_female_pos : F > 0)

-- Theorem
theorem number_of_females (F : ℕ) (h_male : ℕ := 2 * F) (h_total : F + 2 * F = 18) (h_female_pos : F > 0) : F = 6 := 
by 
  sorry

end number_of_females_l62_62173


namespace exists_pos_int_n_l62_62485

def sequence_x (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n + 2) = x n + (x (n + 1))^2

def sequence_y (y : ℕ → ℝ) : Prop :=
  ∀ n, y (n + 2) = y n^2 + y (n + 1)

def positive_initial_conditions (x y : ℕ → ℝ) : Prop :=
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

theorem exists_pos_int_n (x y : ℕ → ℝ) (hx : sequence_x x) (hy : sequence_y y) 
  (ini : positive_initial_conditions x y) : ∃ n, x n > y n := 
sorry

end exists_pos_int_n_l62_62485


namespace perpendicular_iff_zero_dot_product_l62_62640

open Real

def a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem perpendicular_iff_zero_dot_product (m : ℝ) :
  dot_product (a m) (b m) = 0 → m = -1 / 3 :=
by
  sorry

end perpendicular_iff_zero_dot_product_l62_62640


namespace paint_intensity_change_l62_62691

theorem paint_intensity_change (intensity_original : ℝ) (intensity_new : ℝ) (fraction_replaced : ℝ) 
  (h1 : intensity_original = 0.40) (h2 : intensity_new = 0.20) (h3 : fraction_replaced = 1) :
  intensity_new = 0.20 :=
by
  sorry

end paint_intensity_change_l62_62691


namespace exists_integers_for_linear_combination_l62_62582

theorem exists_integers_for_linear_combination 
  (a b c d b1 b2 : ℤ)
  (h1 : ad - bc ≠ 0)
  (h2 : ∃ k : ℤ, b1 = (ad - bc) * k)
  (h3 : ∃ q : ℤ, b2 = (ad - bc) * q) :
  ∃ x y : ℤ, a * x + b * y = b1 ∧ c * x + d * y = b2 :=
sorry

end exists_integers_for_linear_combination_l62_62582


namespace oliver_cycling_distance_l62_62157

/-- Oliver has a training loop for his weekend cycling. He starts by cycling due north for 3 miles. 
  Then he cycles northeast, making a 30° angle with the north for 2 miles, followed by cycling 
  southeast, making a 60° angle with the south for 2 miles. He completes his loop by cycling 
  directly back to the starting point. Prove that the distance of this final segment of his ride 
  is √(11 + 6√3) miles. -/
theorem oliver_cycling_distance :
  let north_displacement : ℝ := 3
  let northeast_displacement : ℝ := 2
  let northeast_angle : ℝ := 30
  let southeast_displacement : ℝ := 2
  let southeast_angle : ℝ := 60
  let north_northeast : ℝ := northeast_displacement * Real.cos (northeast_angle * Real.pi / 180)
  let east_northeast : ℝ := northeast_displacement * Real.sin (northeast_angle * Real.pi / 180)
  let south_southeast : ℝ := southeast_displacement * Real.cos (southeast_angle * Real.pi / 180)
  let east_southeast : ℝ := southeast_displacement * Real.sin (southeast_angle * Real.pi / 180)
  let total_north : ℝ := north_displacement + north_northeast - south_southeast
  let total_east : ℝ := east_northeast + east_southeast
  total_north = 2 + Real.sqrt 3 ∧ total_east = 1 + Real.sqrt 3
  → Real.sqrt (total_north^2 + total_east^2) = Real.sqrt (11 + 6 * Real.sqrt 3) :=
by
  sorry

end oliver_cycling_distance_l62_62157


namespace mixture_price_correct_l62_62751

noncomputable def priceOfMixture (x y : ℝ) (P : ℝ) : Prop :=
  P = (3.10 * x + 3.60 * y) / (x + y)

theorem mixture_price_correct {x y : ℝ} (h_proportion : x / y = 7 / 3) : priceOfMixture x (3 / 7 * x) 3.25 :=
by
  sorry

end mixture_price_correct_l62_62751


namespace inequality_proof_l62_62862

open Real

noncomputable def f (t x : ℝ) : ℝ := t * x - (t - 1) * log x - t

theorem inequality_proof (t x : ℝ) (h_t : t ≤ 0) (h_x : x > 1) : 
  f t x < exp (x - 1) - 1 :=
sorry

end inequality_proof_l62_62862


namespace how_many_candies_eaten_l62_62539

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l62_62539


namespace fraction_of_donations_l62_62873

def max_donation_amount : ℝ := 1200
def total_money_raised : ℝ := 3750000
def donations_from_500_people : ℝ := 500 * max_donation_amount
def fraction_of_money_raised : ℝ := 0.4 * total_money_raised
def num_donors : ℝ := 1500

theorem fraction_of_donations (f : ℝ) :
  donations_from_500_people + num_donors * f * max_donation_amount = fraction_of_money_raised → f = 1 / 2 :=
by
  sorry

end fraction_of_donations_l62_62873


namespace real_roots_for_all_a_b_l62_62305

theorem real_roots_for_all_a_b (a b : ℝ) : ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) :=
sorry

end real_roots_for_all_a_b_l62_62305


namespace clay_boys_proof_l62_62652

variable (total_students : ℕ)
variable (total_boys : ℕ)
variable (total_girls : ℕ)
variable (jonas_students : ℕ)
variable (clay_students : ℕ)
variable (birch_students : ℕ)
variable (jonas_boys : ℕ)
variable (birch_girls : ℕ)

noncomputable def boys_from_clay (total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls : ℕ) : ℕ :=
  let birch_boys := birch_students - birch_girls
  let clay_boys := total_boys - (jonas_boys + birch_boys)
  clay_boys

theorem clay_boys_proof (h1 : total_students = 180) (h2 : total_boys = 94) 
    (h3 : total_girls = 86) (h4 : jonas_students = 60) 
    (h5 : clay_students = 80) (h6 : birch_students = 40) 
    (h7 : jonas_boys = 30) (h8 : birch_girls = 24) : 
  boys_from_clay total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls = 48 := 
by 
  simp [boys_from_clay] 
  sorry

end clay_boys_proof_l62_62652


namespace numbers_lcm_sum_l62_62968

theorem numbers_lcm_sum :
  ∃ A : List ℕ, A.length = 100 ∧
    (A.count 1 = 89 ∧ A.count 2 = 8 ∧ [4, 5, 6] ⊆ A) ∧
    A.sum = A.foldr lcm 1 :=
by
  sorry

end numbers_lcm_sum_l62_62968


namespace distance_on_dirt_section_distance_on_mud_section_l62_62602

noncomputable def v_highway : ℝ := 120 -- km/h
noncomputable def v_dirt : ℝ := 40 -- km/h
noncomputable def v_mud : ℝ := 10 -- km/h
noncomputable def initial_distance : ℝ := 0.6 -- km

theorem distance_on_dirt_section : 
  ∃ s_1 : ℝ, 
  (s_1 = 0.2 * 1000 ∧ -- converting km to meters
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

theorem distance_on_mud_section : 
  ∃ s_2 : ℝ, 
  (s_2 = 50 ∧
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

end distance_on_dirt_section_distance_on_mud_section_l62_62602


namespace find_x_l62_62829

theorem find_x (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 :=
sorry

end find_x_l62_62829


namespace no_statement_implies_neg_p_or_q_l62_62106

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∨ ¬ q
def statement3 (p q : Prop) : Prop := ¬ p ∨ q
def statement4 (p q : Prop) : Prop := ¬ p ∧ q
def neg_p_or_q (p q : Prop) : Prop := ¬ (p ∨ q)

theorem no_statement_implies_neg_p_or_q (p q : Prop) :
  ¬ (statement1 p q → neg_p_or_q p q) ∧
  ¬ (statement2 p q → neg_p_or_q p q) ∧
  ¬ (statement3 p q → neg_p_or_q p q) ∧
  ¬ (statement4 p q → neg_p_or_q p q)
:= by
  sorry

end no_statement_implies_neg_p_or_q_l62_62106


namespace more_blue_blocks_than_red_l62_62389

theorem more_blue_blocks_than_red 
  (red_blocks : ℕ) 
  (yellow_blocks : ℕ) 
  (blue_blocks : ℕ) 
  (total_blocks : ℕ) 
  (h_red : red_blocks = 18) 
  (h_yellow : yellow_blocks = red_blocks + 7) 
  (h_total : total_blocks = red_blocks + yellow_blocks + blue_blocks) 
  (h_total_given : total_blocks = 75) :
  blue_blocks - red_blocks = 14 :=
by sorry

end more_blue_blocks_than_red_l62_62389


namespace jesses_room_total_area_l62_62554

-- Define the dimensions of the first rectangular part
def length1 : ℕ := 12
def width1 : ℕ := 8

-- Define the dimensions of the second rectangular part
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Define the areas of both parts
def area1 : ℕ := length1 * width1
def area2 : ℕ := length2 * width2

-- Define the total area
def total_area : ℕ := area1 + area2

-- Statement of the theorem we want to prove
theorem jesses_room_total_area : total_area = 120 :=
by
  -- We would provide the proof here
  sorry

end jesses_room_total_area_l62_62554


namespace union_of_A_B_l62_62744

open Set

variable {α : Type*} [LinearOrder α]

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem union_of_A_B : A ∪ B = { x | -1 < x ∧ x < 3 } :=
sorry

end union_of_A_B_l62_62744


namespace symmetry_axis_one_of_cos_2x_minus_sin_2x_l62_62270

noncomputable def symmetry_axis (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 2) - Real.pi / 8

theorem symmetry_axis_one_of_cos_2x_minus_sin_2x :
  symmetry_axis (-Real.pi / 8) :=
by
  use 0
  simp
  sorry

end symmetry_axis_one_of_cos_2x_minus_sin_2x_l62_62270


namespace vincent_back_to_A_after_5_min_p_plus_q_computation_l62_62878

def probability (n : ℕ) : ℚ :=
  if n = 0 then 1
  else 1 / 4 * (1 - probability (n - 1))

theorem vincent_back_to_A_after_5_min : 
  probability 5 = 51 / 256 :=
by sorry

theorem p_plus_q_computation :
  51 + 256 = 307 :=
by linarith

end vincent_back_to_A_after_5_min_p_plus_q_computation_l62_62878


namespace postage_unformable_l62_62686

theorem postage_unformable (n : ℕ) (h₁ : n > 0) (h₂ : 110 = 7 * n - 7 - n) :
  n = 19 := 
sorry

end postage_unformable_l62_62686


namespace largest_divisor_l62_62568

theorem largest_divisor (x : ℤ) (hx : x % 2 = 1) : 180 ∣ (15 * x + 3) * (15 * x + 9) * (10 * x + 5) := 
by
  sorry

end largest_divisor_l62_62568


namespace foreign_students_next_semester_l62_62141

theorem foreign_students_next_semester (total_students : ℕ) (percent_foreign : ℝ) (new_foreign_students : ℕ) 
  (h_total : total_students = 1800) (h_percent : percent_foreign = 0.30) (h_new : new_foreign_students = 200) : 
  (0.30 * 1800 + 200 : ℝ) = 740 := by
  sorry

end foreign_students_next_semester_l62_62141


namespace abs_neg_eight_l62_62149

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l62_62149


namespace average_weight_of_class_l62_62784

theorem average_weight_of_class (n_boys n_girls : ℕ) (avg_weight_boys avg_weight_girls : ℝ)
    (h_boys : n_boys = 5) (h_girls : n_girls = 3)
    (h_avg_weight_boys : avg_weight_boys = 60) (h_avg_weight_girls : avg_weight_girls = 50) :
    (n_boys * avg_weight_boys + n_girls * avg_weight_girls) / (n_boys + n_girls) = 56.25 := 
by
  sorry

end average_weight_of_class_l62_62784


namespace binary_to_decimal_l62_62502

theorem binary_to_decimal (b : ℕ) (h : b = 2^3 + 2^2 + 0 * 2^1 + 2^0) : b = 13 :=
by {
  -- proof is omitted
  sorry
}

end binary_to_decimal_l62_62502


namespace incorrect_expressions_l62_62093

theorem incorrect_expressions (x y : ℚ) (h : x / y = 2 / 5) :
    (x + 3 * y) / x ≠ 17 / 2 ∧ (x - y) / y ≠ 3 / 5 :=
by
  sorry

end incorrect_expressions_l62_62093


namespace police_speed_l62_62338

/-- 
A thief runs away from a location with a speed of 20 km/hr.
A police officer starts chasing him from a location 60 km away after 1 hour.
The police officer catches the thief after 4 hours.
Prove that the speed of the police officer is 40 km/hr.
-/
theorem police_speed
  (thief_speed : ℝ)
  (police_start_distance : ℝ)
  (police_chase_time : ℝ)
  (time_head_start : ℝ)
  (police_distance_to_thief : ℝ)
  (thief_distance_after_time : ℝ)
  (total_distance_police_officer : ℝ) :
  thief_speed = 20 ∧
  police_start_distance = 60 ∧
  police_chase_time = 4 ∧
  time_head_start = 1 ∧
  police_distance_to_thief = police_start_distance + 100 ∧
  thief_distance_after_time = thief_speed * police_chase_time + thief_speed * time_head_start ∧
  total_distance_police_officer = police_start_distance + (thief_speed * (police_chase_time + time_head_start)) →
  (total_distance_police_officer / police_chase_time) = 40 := by
  sorry

end police_speed_l62_62338


namespace inequality_bound_l62_62352

theorem inequality_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ( (2 * a + b + c)^2 / (2 * a ^ 2 + (b + c) ^2) + 
    (2 * b + c + a)^2 / (2 * b ^ 2 + (c + a) ^2) + 
    (2 * c + a + b)^2 / (2 * c ^ 2 + (a + b) ^2) ) ≤ 8 := 
sorry

end inequality_bound_l62_62352


namespace remainder_ab_div_48_is_15_l62_62362

noncomputable def remainder_ab_div_48 (a b : ℕ) (ha : a % 8 = 3) (hb : b % 6 = 5) : ℕ :=
  (a * b) % 48

theorem remainder_ab_div_48_is_15 {a b : ℕ} (ha : a % 8 = 3) (hb : b % 6 = 5) : remainder_ab_div_48 a b ha hb = 15 :=
  sorry

end remainder_ab_div_48_is_15_l62_62362


namespace Mrs_Hilt_bought_two_cones_l62_62943

def ice_cream_cone_cost : ℕ := 99
def total_spent : ℕ := 198

theorem Mrs_Hilt_bought_two_cones : total_spent / ice_cream_cone_cost = 2 :=
by
  sorry

end Mrs_Hilt_bought_two_cones_l62_62943


namespace equation_solutions_35_implies_n_26_l62_62912

theorem equation_solutions_35_implies_n_26 (n : ℕ) (h3x3y2z_eq_n : ∃ (s : Finset (ℕ × ℕ × ℕ)), (∀ t ∈ s, ∃ (x y z : ℕ), 
  t = (x, y, z) ∧ 3 * x + 3 * y + 2 * z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ s.card = 35) : n = 26 := 
sorry

end equation_solutions_35_implies_n_26_l62_62912


namespace ratio_of_blue_fish_to_total_fish_l62_62339

-- Define the given conditions
def total_fish : ℕ := 30
def blue_spotted_fish : ℕ := 5
def half (n : ℕ) : ℕ := n / 2

-- Calculate the number of blue fish using the conditions
def blue_fish : ℕ := blue_spotted_fish * 2

-- Define the ratio of blue fish to total fish
def ratio (num denom : ℕ) : ℚ := num / denom

-- The theorem to prove
theorem ratio_of_blue_fish_to_total_fish :
  ratio blue_fish total_fish = 1 / 3 := by
  sorry

end ratio_of_blue_fish_to_total_fish_l62_62339


namespace probability_at_least_3_l62_62386

noncomputable def probability_hitting_at_least_3_of_4 (p : ℝ) (n : ℕ) : ℝ :=
  let p3 := (Nat.choose n 3) * (p^3) * ((1 - p)^(n - 3))
  let p4 := (Nat.choose n 4) * (p^4)
  p3 + p4

theorem probability_at_least_3 (h : probability_hitting_at_least_3_of_4 0.8 4 = 0.8192) : 
   True :=
by trivial

end probability_at_least_3_l62_62386


namespace ratio_a_b_eq_neg_one_fifth_l62_62840

theorem ratio_a_b_eq_neg_one_fifth (x y a b : ℝ) (hb_ne_zero : b ≠ 0) 
    (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) : a / b = -1 / 5 :=
by {
  sorry
}

end ratio_a_b_eq_neg_one_fifth_l62_62840


namespace equation_has_two_solutions_l62_62095

theorem equation_has_two_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^x1 = x1^2 - 2*x1 - a ∧ a^x2 = x2^2 - 2*x2 - a :=
sorry

end equation_has_two_solutions_l62_62095


namespace perimeter_of_triangle_eq_28_l62_62649

-- Definitions of conditions
variables (p : ℝ)
def inradius : ℝ := 2.0
def area : ℝ := 28

-- Main theorem statement
theorem perimeter_of_triangle_eq_28 : p = 28 :=
  by
  -- The proof is omitted
  sorry

end perimeter_of_triangle_eq_28_l62_62649


namespace rectangle_area_increase_l62_62807

theorem rectangle_area_increase (L W : ℝ) (h1: L > 0) (h2: W > 0) :
   let original_area := L * W
   let new_length := 1.20 * L
   let new_width := 1.20 * W
   let new_area := new_length * new_width
   let percentage_increase := ((new_area - original_area) / original_area) * 100
   percentage_increase = 44 :=
by
  sorry

end rectangle_area_increase_l62_62807


namespace positive_difference_l62_62433

-- Define the conditions given in the problem
def conditions (x y : ℝ) : Prop :=
  x + y = 40 ∧ 3 * y - 4 * x = 20

-- The theorem to prove
theorem positive_difference (x y : ℝ) (h : conditions x y) : abs (y - x) = 11.42 :=
by
  sorry -- proof omitted

end positive_difference_l62_62433


namespace partition_exists_min_n_in_A_l62_62108

-- Definition of subsets and their algebraic properties
variable (A B C : Set ℕ)

-- The Initial conditions
axiom A_squared_eq_A : ∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)
axiom B_squared_eq_C : ∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)
axiom C_squared_eq_B : ∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)
axiom AB_eq_B : ∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)
axiom AC_eq_C : ∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)
axiom BC_eq_A : ∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)

-- Statement for the partition existence with given conditions
theorem partition_exists :
  ∃ A B C : Set ℕ, (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
               (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
               (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
               (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
               (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
               (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) :=
sorry

-- Statement for the minimum n in A such that n and n+1 are both in A is at most 77
theorem min_n_in_A :
  ∀ A B C : Set ℕ,
    (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
    (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
    (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
    (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
    (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
    (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) →
    ∃ n : ℕ, (n ∈ A) ∧ (n + 1 ∈ A) ∧ n ≤ 77 :=
sorry

end partition_exists_min_n_in_A_l62_62108


namespace distinct_arrangements_of_PHONE_l62_62951

-- Condition: The word PHONE consists of 5 distinct letters
def distinctLetters := 5

-- Theorem: The number of distinct arrangements of the letters in the word PHONE
theorem distinct_arrangements_of_PHONE : Nat.factorial distinctLetters = 120 := sorry

end distinct_arrangements_of_PHONE_l62_62951


namespace Elmer_vs_Milton_food_l62_62088

def Penelope_daily_food := 20  -- Penelope eats 20 pounds per day
def Greta_to_Penelope_ratio := 1 / 10  -- Greta eats 1/10 of what Penelope eats
def Milton_to_Greta_ratio := 1 / 100  -- Milton eats 1/100 of what Greta eats
def Elmer_to_Penelope_difference := 60  -- Elmer eats 60 pounds more than Penelope

def Greta_daily_food := Penelope_daily_food * Greta_to_Penelope_ratio
def Milton_daily_food := Greta_daily_food * Milton_to_Greta_ratio
def Elmer_daily_food := Penelope_daily_food + Elmer_to_Penelope_difference

theorem Elmer_vs_Milton_food :
  Elmer_daily_food = 4000 * Milton_daily_food := by
  sorry

end Elmer_vs_Milton_food_l62_62088


namespace segment_association_l62_62639

theorem segment_association (x y : ℝ) 
  (h1 : ∃ (D : ℝ), ∀ (P : ℝ), abs (P - D) ≤ 5) 
  (h2 : ∃ (D' : ℝ), ∀ (P' : ℝ), abs (P' - D') ≤ 9)
  (h3 : 3 * x - 2 * y = 6) : 
  x + y = 12 := 
by sorry

end segment_association_l62_62639


namespace geometric_sequence_solution_l62_62511

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, a n = a1 * r ^ (n - 1)

theorem geometric_sequence_solution :
  ∀ (a : ℕ → ℝ),
    (geometric_sequence a) →
    (∃ a2 a18, a2 + a18 = -6 ∧ a2 * a18 = 4 ∧ a 2 = a2 ∧ a 18 = a18) →
    a 4 * a 16 + a 10 = 6 :=
by
  sorry

end geometric_sequence_solution_l62_62511


namespace single_discount_equivalence_l62_62746

noncomputable def original_price : ℝ := 50
noncomputable def discount1 : ℝ := 0.15
noncomputable def discount2 : ℝ := 0.10
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)
noncomputable def effective_discount_price := 
  apply_discount (apply_discount original_price discount1) discount2
noncomputable def effective_discount :=
  (original_price - effective_discount_price) / original_price

theorem single_discount_equivalence :
  effective_discount = 0.235 := by
  sorry

end single_discount_equivalence_l62_62746


namespace correct_product_l62_62726

theorem correct_product (a b : ℕ) (a' : ℕ) (h1 : a' = (a % 10) * 10 + (a / 10)) 
  (h2 : a' * b = 143) (h3 : 10 ≤ a ∧ a < 100):
  a * b = 341 :=
sorry

end correct_product_l62_62726


namespace inequality_proof_l62_62395

variables (a b c : ℝ)

theorem inequality_proof
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (cond : a^2 + b^2 + c^2 + ab + bc + ca ≤ 2) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 := 
sorry

end inequality_proof_l62_62395


namespace parabola_vertex_eq_l62_62953

theorem parabola_vertex_eq : 
  ∃ (x y : ℝ), y = -3 * x^2 + 6 * x + 1 ∧ (x = 1) ∧ (y = 4) := 
by
  sorry

end parabola_vertex_eq_l62_62953


namespace deposits_exceed_10_on_second_Tuesday_l62_62590

noncomputable def deposits_exceed_10 (n : ℕ) : ℕ :=
2 * (2^n - 1)

theorem deposits_exceed_10_on_second_Tuesday :
  ∃ n, deposits_exceed_10 n > 1000 ∧ 1 + (n - 1) % 7 = 2 ∧ n < 21 :=
sorry

end deposits_exceed_10_on_second_Tuesday_l62_62590


namespace cost_prices_three_watches_l62_62195

theorem cost_prices_three_watches :
  ∃ (C1 C2 C3 : ℝ), 
    (0.9 * C1 + 210 = 1.04 * C1) ∧ 
    (0.85 * C2 + 180 = 1.03 * C2) ∧ 
    (0.95 * C3 + 250 = 1.06 * C3) ∧ 
    C1 = 1500 ∧ 
    C2 = 1000 ∧ 
    C3 = (25000 / 11) :=
by 
  sorry

end cost_prices_three_watches_l62_62195


namespace sum_of_cubes_of_nonneg_rationals_l62_62883

theorem sum_of_cubes_of_nonneg_rationals (n : ℤ) (h1 : n > 1) (h2 : ∃ a b : ℚ, a^3 + b^3 = n) :
  ∃ c d : ℚ, c ≥ 0 ∧ d ≥ 0 ∧ c^3 + d^3 = n :=
sorry

end sum_of_cubes_of_nonneg_rationals_l62_62883


namespace find_c_l62_62985

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end find_c_l62_62985


namespace log_expression_simplification_l62_62429

open Real

noncomputable def log_expr (a b c d x y z : ℝ) : ℝ :=
  log (a^2 / b) + log (b^2 / c) + log (c^2 / d) - log (a^2 * y * z / (d^2 * x))

theorem log_expression_simplification (a b c d x y z : ℝ) (h : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : d ≠ 0) (h5 : x ≠ 0) (h6 : y ≠ 0) (h7 : z ≠ 0) :
  log_expr a b c d x y z = log (bdx / yz) :=
by
  -- Proof goes here
  sorry

end log_expression_simplification_l62_62429


namespace find_p_geometric_progression_l62_62434

theorem find_p_geometric_progression (p : ℝ) : 
  (p = -1 ∨ p = 40 / 9) ↔ ((9 * p + 10), (3 * p), |p - 8|) ∈ 
  {gp | ∃ r : ℝ, gp = (r, r * r, r * r * r)} :=
by sorry

end find_p_geometric_progression_l62_62434


namespace sallys_dad_nickels_l62_62399

theorem sallys_dad_nickels :
  ∀ (initial_nickels mother's_nickels total_nickels nickels_from_dad : ℕ), 
    initial_nickels = 7 → 
    mother's_nickels = 2 →
    total_nickels = 18 →
    total_nickels = initial_nickels + mother's_nickels + nickels_from_dad →
    nickels_from_dad = 9 :=
by
  intros initial_nickels mother's_nickels total_nickels nickels_from_dad
  intros h1 h2 h3 h4
  sorry

end sallys_dad_nickels_l62_62399


namespace arithmetic_sequence_properties_l62_62419

variables {a : ℕ → ℤ} {S T : ℕ → ℤ}

theorem arithmetic_sequence_properties 
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)) -- Sum of first n terms of arithmetic sequence
  (h₄ : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1)) -- General term formula of arithmetic sequence
  : (∀ n, a n = -2 * n + 15) ∧
    ( (∀ n, 1 ≤ n ∧ n ≤ 7 → T n = -n^2 + 14 * n) ∧ 
      (∀ n, n ≥ 8 → T n = n^2 - 14 * n + 98)) :=
by
sorry

end arithmetic_sequence_properties_l62_62419


namespace remaining_stock_weight_l62_62474

def green_beans_weight : ℕ := 80
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 20
def flour_weight : ℕ := 2 * sugar_weight
def lentils_weight : ℕ := flour_weight - 10

def rice_remaining_weight : ℕ := rice_weight - rice_weight / 3
def sugar_remaining_weight : ℕ := sugar_weight - sugar_weight / 5
def flour_remaining_weight : ℕ := flour_weight - flour_weight / 4
def lentils_remaining_weight : ℕ := lentils_weight - lentils_weight / 6

def total_remaining_weight : ℕ :=
  rice_remaining_weight + sugar_remaining_weight + flour_remaining_weight + lentils_remaining_weight + green_beans_weight

theorem remaining_stock_weight :
  total_remaining_weight = 343 := by
  sorry

end remaining_stock_weight_l62_62474


namespace paint_amount_third_day_l62_62584

theorem paint_amount_third_day : 
  let initial_paint := 80
  let first_day_usage := initial_paint / 2
  let paint_after_first_day := initial_paint - first_day_usage
  let added_paint := 20
  let new_total_paint := paint_after_first_day + added_paint
  let second_day_usage := new_total_paint / 2
  let paint_after_second_day := new_total_paint - second_day_usage
  paint_after_second_day = 30 :=
by
  sorry

end paint_amount_third_day_l62_62584


namespace science_club_members_neither_l62_62741

theorem science_club_members_neither {S B C : ℕ} (total : S = 60) (bio : B = 40) (chem : C = 35) (both : ℕ := 25) :
    S - ((B - both) + (C - both) + both) = 10 :=
by
  sorry

end science_club_members_neither_l62_62741


namespace find_number_l62_62446

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l62_62446


namespace simplify_expression_l62_62915

theorem simplify_expression :
  (625: ℝ)^(1/4) * (256: ℝ)^(1/3) = 20 := 
sorry

end simplify_expression_l62_62915


namespace geom_seq_necessity_geom_seq_not_sufficient_l62_62557

theorem geom_seq_necessity (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    q > 1 ∨ q < -1 :=
  sorry

theorem geom_seq_not_sufficient (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    ¬ (q > 1 → a₁ < a₁ * q^2) :=
  sorry

end geom_seq_necessity_geom_seq_not_sufficient_l62_62557


namespace max_gold_coins_l62_62813

theorem max_gold_coins (n k : ℕ) 
  (h1 : n = 8 * k + 4)
  (h2 : n < 150) : 
  n = 148 :=
by
  sorry

end max_gold_coins_l62_62813


namespace Ivan_uses_more_paint_l62_62843

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l62_62843


namespace total_weekly_airflow_l62_62921

-- Definitions from conditions
def fanA_airflow : ℝ := 10  -- liters per second
def fanA_time_per_day : ℝ := 10 * 60  -- converted to seconds (10 minutes * 60 seconds/minute)

def fanB_airflow : ℝ := 15  -- liters per second
def fanB_time_per_day : ℝ := 20 * 60  -- converted to seconds (20 minutes * 60 seconds/minute)

def fanC_airflow : ℝ := 25  -- liters per second
def fanC_time_per_day : ℝ := 30 * 60  -- converted to seconds (30 minutes * 60 seconds/minute)

def days_in_week : ℝ := 7

-- Theorem statement to be proven
theorem total_weekly_airflow : fanA_airflow * fanA_time_per_day * days_in_week +
                               fanB_airflow * fanB_time_per_day * days_in_week +
                               fanC_airflow * fanC_time_per_day * days_in_week = 483000 := 
by
  -- skip the proof
  sorry

end total_weekly_airflow_l62_62921


namespace third_term_geometric_series_l62_62019

variable {b1 b3 q : ℝ}
variable (hb1 : b1 * (-1/4) = -1/2)
variable (hs : b1 / (1 - q) = 8/5)
variable (hq : |q| < 1)

theorem third_term_geometric_series (hb1 : b1 * (-1 / 4) = -1 / 2)
  (hs : b1 / (1 - q) = 8 / 5)
  (hq : |q| < 1)
  : b3 = b1 * q^2 := by
    sorry

end third_term_geometric_series_l62_62019


namespace find_satisfying_pairs_l62_62910

theorem find_satisfying_pairs (n p : ℕ) (prime_p : Nat.Prime p) :
  n ≤ 2 * p ∧ (p - 1)^n + 1 ≡ 0 [MOD n^2] →
  (n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end find_satisfying_pairs_l62_62910


namespace distinct_ordered_pair_count_l62_62877

theorem distinct_ordered_pair_count (x y : ℕ) (h1 : x + y = 50) (h2 : 1 ≤ x) (h3 : 1 ≤ y) : 
  ∃! (x y : ℕ), x + y = 50 ∧ 1 ≤ x ∧ 1 ≤ y :=
by
  sorry

end distinct_ordered_pair_count_l62_62877


namespace ratio_D_to_C_l62_62351

-- Defining the terms and conditions
def speed_ratio (C Ch D : ℝ) : Prop :=
  (C = 2 * Ch) ∧
  (D / Ch = 6)

-- The theorem statement
theorem ratio_D_to_C (C Ch D : ℝ) (h : speed_ratio C Ch D) : (D / C = 3) :=
by
  sorry

end ratio_D_to_C_l62_62351


namespace average_temperature_correct_l62_62944

theorem average_temperature_correct (W T : ℝ) :
  (38 + W + T) / 3 = 32 →
  44 = 44 →
  38 = 38 →
  (W + T + 44) / 3 = 34 :=
by
  intros h1 h2 h3
  sorry

end average_temperature_correct_l62_62944


namespace part_i_part_ii_l62_62125

-- Define the operations for the weird calculator.
def Dsharp (n : ℕ) : ℕ := 2 * n + 1
def Dflat (n : ℕ) : ℕ := 2 * n - 1

-- Define the initial starting point.
def initial_display : ℕ := 1

-- Define a function to execute a sequence of button presses.
def execute_sequence (seq : List (ℕ → ℕ)) (initial : ℕ) : ℕ :=
  seq.foldl (fun x f => f x) initial

-- Problem (i): Prove there is a sequence that results in 313 starting from 1 after eight presses.
theorem part_i : ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = 313 :=
by sorry

-- Problem (ii): Describe all numbers that can be achieved from exactly eight button presses starting from 1.
theorem part_ii : 
  ∀ n : ℕ, n % 2 = 1 ∧ n < 2^9 →
  ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = n :=
by sorry

end part_i_part_ii_l62_62125


namespace last_four_digits_of_5_pow_2011_l62_62013

theorem last_four_digits_of_5_pow_2011 :
  (5 ^ 5) % 10000 = 3125 ∧
  (5 ^ 6) % 10000 = 5625 ∧
  (5 ^ 7) % 10000 = 8125 →
  (5 ^ 2011) % 10000 = 8125 :=
by
  sorry

end last_four_digits_of_5_pow_2011_l62_62013


namespace seq_eventually_reaches_one_l62_62022

theorem seq_eventually_reaches_one (a : ℕ → ℤ) (h₁ : a 1 > 0) :
  (∀ n, n % 4 = 0 → a (n + 1) = a n / 2) →
  (∀ n, n % 4 = 1 → a (n + 1) = 3 * a n + 1) →
  (∀ n, n % 4 = 2 → a (n + 1) = 2 * a n - 1) →
  (∀ n, n % 4 = 3 → a (n + 1) = (a n + 1) / 4) →
  ∃ m, a m = 1 :=
by
  sorry

end seq_eventually_reaches_one_l62_62022


namespace midpoint_distance_from_school_l62_62977

def distance_school_kindergarten_km := 1
def distance_school_kindergarten_m := 700
def distance_kindergarten_house_m := 900

theorem midpoint_distance_from_school : 
  (1000 * distance_school_kindergarten_km + distance_school_kindergarten_m + distance_kindergarten_house_m) / 2 = 1300 := 
by
  sorry

end midpoint_distance_from_school_l62_62977


namespace expected_red_light_l62_62960

variables (n : ℕ) (p : ℝ)
def binomial_distribution : Type := sorry

noncomputable def expected_value (n : ℕ) (p : ℝ) : ℝ :=
n * p

theorem expected_red_light :
  expected_value 3 0.4 = 1.2 :=
by
  simp [expected_value]
  sorry

end expected_red_light_l62_62960


namespace temperature_on_friday_l62_62057

-- Define the temperatures on different days
variables (T W Th F : ℝ)

-- Define the conditions
def condition1 : Prop := (T + W + Th) / 3 = 32
def condition2 : Prop := (W + Th + F) / 3 = 34
def condition3 : Prop := T = 38

-- State the theorem to prove the temperature on Friday
theorem temperature_on_friday (h1 : condition1 T W Th) (h2 : condition2 W Th F) (h3 : condition3 T) : F = 44 :=
  sorry

end temperature_on_friday_l62_62057


namespace exp_gt_pow_l62_62318

theorem exp_gt_pow (x : ℝ) (h_pos : 0 < x) (h_ne : x ≠ Real.exp 1) : Real.exp x > x ^ Real.exp 1 := by
  sorry

end exp_gt_pow_l62_62318


namespace Merry_sold_470_apples_l62_62964

-- Define the conditions
def boxes_on_Saturday : Nat := 50
def boxes_on_Sunday : Nat := 25
def apples_per_box : Nat := 10
def boxes_left : Nat := 3

-- Define the question as the number of apples sold
theorem Merry_sold_470_apples :
  (boxes_on_Saturday - boxes_on_Sunday) * apples_per_box +
  (boxes_on_Sunday - boxes_left) * apples_per_box = 470 := by
  sorry

end Merry_sold_470_apples_l62_62964


namespace shorter_piece_length_l62_62918

theorem shorter_piece_length (L : ℝ) (k : ℝ) (shorter_piece : ℝ) : 
  L = 28 ∧ k = 2.00001 / 5 ∧ L = shorter_piece + k * shorter_piece → 
  shorter_piece = 20 :=
by
  sorry

end shorter_piece_length_l62_62918


namespace range_of_g_l62_62636

noncomputable def f (x : ℝ) : ℝ := 2 * x - 3

noncomputable def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → -29 ≤ g x ∧ g x ≤ 3) :=
sorry

end range_of_g_l62_62636


namespace Jenny_total_wins_l62_62168

theorem Jenny_total_wins :
  let games_against_mark := 10
  let mark_wins := 1
  let mark_losses := games_against_mark - mark_wins
  let games_against_jill := 2 * games_against_mark
  let jill_wins := (75 / 100) * games_against_jill
  let jenny_wins_against_jill := games_against_jill - jill_wins
  mark_losses + jenny_wins_against_jill = 14 :=
by
  sorry

end Jenny_total_wins_l62_62168


namespace range_of_a_l62_62545

theorem range_of_a (a : ℝ) :
  (∃ x : ℤ, 2 * (x : ℝ) - 1 > 3 ∧ x ≤ a) ∧ (∀ x : ℤ, 2 * (x : ℝ) - 1 > 3 → x ≤ a) → 5 ≤ a ∧ a < 6 :=
by
  sorry

end range_of_a_l62_62545


namespace solve_problem_l62_62037

noncomputable def find_z_values (x : ℝ) : ℝ :=
  (x - 3)^2 * (x + 4) / (2 * x - 4)

theorem solve_problem (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  find_z_values x = 64.8 ∨ find_z_values x = -10.125 :=
by
  sorry

end solve_problem_l62_62037


namespace arithmetic_sequence_50th_term_l62_62060

-- Define the arithmetic sequence parameters
def first_term : Int := 2
def common_difference : Int := 5

-- Define the formula to calculate the n-th term of the sequence
def nth_term (n : Nat) : Int :=
  first_term + (n - 1) * common_difference

-- Prove that the 50th term of the sequence is 247
theorem arithmetic_sequence_50th_term : nth_term 50 = 247 :=
  by
  -- Proof goes here
  sorry

end arithmetic_sequence_50th_term_l62_62060


namespace complex_equation_square_sum_l62_62525

-- Lean 4 statement of the mathematical proof problem
theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
    (h1 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 := by
  sorry

end complex_equation_square_sum_l62_62525


namespace jimmy_irene_total_payment_l62_62650

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end jimmy_irene_total_payment_l62_62650


namespace range_of_k_l62_62051

theorem range_of_k (k : ℤ) (a : ℤ → ℤ) (h_a : ∀ n : ℕ, a n = |n - k| + |n + 2 * k|)
  (h_a3_equal_a4 : a 3 = a 4) : k ≤ -2 ∨ k ≥ 4 :=
sorry

end range_of_k_l62_62051


namespace carpenter_needs_more_logs_l62_62096

-- Define the given conditions in Lean 4
def total_woodblocks_needed : ℕ := 80
def logs_on_hand : ℕ := 8
def woodblocks_per_log : ℕ := 5

-- Statement: Proving the number of additional logs the carpenter needs
theorem carpenter_needs_more_logs :
  let woodblocks_available := logs_on_hand * woodblocks_per_log
  let additional_woodblocks := total_woodblocks_needed - woodblocks_available
  additional_woodblocks / woodblocks_per_log = 8 :=
by
  sorry

end carpenter_needs_more_logs_l62_62096


namespace find_unit_prices_l62_62870

variable (x : ℝ)

def typeB_unit_price (priceB : ℝ) : Prop :=
  priceB = 15

def typeA_unit_price (priceA : ℝ) : Prop :=
  priceA = 40

def budget_condition : Prop :=
  900 / x = 3 * (800 / (x + 25))

theorem find_unit_prices (h : budget_condition x) :
  typeB_unit_price x ∧ typeA_unit_price (x + 25) :=
sorry

end find_unit_prices_l62_62870


namespace score_below_mean_l62_62154

theorem score_below_mean :
  ∃ (σ : ℝ), (74 - 2 * σ = 58) ∧ (98 - 74 = 3 * σ) :=
sorry

end score_below_mean_l62_62154


namespace total_distance_apart_l62_62896

def Jay_rate : ℕ := 1 / 15 -- Jay walks 1 mile every 15 minutes
def Paul_rate : ℕ := 3 / 30 -- Paul walks 3 miles every 30 minutes
def time_in_minutes : ℕ := 120 -- 2 hours converted to minutes

def Jay_distance (rate time : ℕ) : ℕ := rate * time / 15
def Paul_distance (rate time : ℕ) : ℕ := rate * time / 30

theorem total_distance_apart : 
  Jay_distance Jay_rate time_in_minutes + Paul_distance Paul_rate time_in_minutes = 20 :=
  by
  -- Proof here
  sorry

end total_distance_apart_l62_62896


namespace white_pairs_coincide_l62_62034

theorem white_pairs_coincide 
    (red_triangles : ℕ)
    (blue_triangles : ℕ)
    (white_triangles : ℕ)
    (red_pairs : ℕ)
    (blue_pairs : ℕ)
    (red_white_pairs : ℕ)
    (coinciding_white_pairs : ℕ) :
    red_triangles = 4 → 
    blue_triangles = 6 →
    white_triangles = 10 →
    red_pairs = 3 →
    blue_pairs = 4 →
    red_white_pairs = 3 →
    coinciding_white_pairs = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end white_pairs_coincide_l62_62034


namespace polynomial_root_cubic_sum_l62_62988

theorem polynomial_root_cubic_sum
  (a b c : ℝ)
  (h : ∀ x : ℝ, (Polynomial.eval x (3 * Polynomial.X^3 + 5 * Polynomial.X^2 - 150 * Polynomial.X + 7) = 0)
    → x = a ∨ x = b ∨ x = c) :
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 :=
  sorry

end polynomial_root_cubic_sum_l62_62988


namespace count_whole_numbers_between_4_and_18_l62_62198

theorem count_whole_numbers_between_4_and_18 :
  ∀ (x : ℕ), 4 < x ∧ x < 18 ↔ ∃ n : ℕ, n = 13 :=
by sorry

end count_whole_numbers_between_4_and_18_l62_62198


namespace interval_satisfies_ineq_l62_62328

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l62_62328


namespace GCF_30_90_75_l62_62585

theorem GCF_30_90_75 : Nat.gcd (Nat.gcd 30 90) 75 = 15 := by
  sorry

end GCF_30_90_75_l62_62585


namespace decreasing_line_implies_m_half_l62_62150

theorem decreasing_line_implies_m_half (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * m - 1) * x₁ + b > (2 * m - 1) * x₂ + b) → m < 1 / 2 :=
by
  intro h
  sorry

end decreasing_line_implies_m_half_l62_62150


namespace total_books_bought_l62_62151

-- Let x be the number of math books and y be the number of history books
variables (x y : ℕ)

-- Conditions
def math_book_cost := 4
def history_book_cost := 5
def total_price := 368
def num_math_books := 32

-- The total number of books bought is the sum of the number of math books and history books, which should result in 80
theorem total_books_bought : 
  y * history_book_cost + num_math_books * math_book_cost = total_price → 
  x = num_math_books → 
  x + y = 80 :=
by
  sorry

end total_books_bought_l62_62151


namespace tom_filled_balloons_l62_62158

theorem tom_filled_balloons :
  ∀ (Tom Luke Anthony : ℕ), 
    (Tom = 3 * Luke) →
    (Luke = Anthony / 4) →
    (Anthony = 44) →
    (Tom = 33) :=
by
  intros Tom Luke Anthony hTom hLuke hAnthony
  sorry

end tom_filled_balloons_l62_62158


namespace compute_2018_square_123_Delta_4_l62_62492

namespace custom_operations

def Delta (a b : ℕ) : ℕ := a * 10 ^ b + b
def Square (a b : ℕ) : ℕ := a * 10 + b

theorem compute_2018_square_123_Delta_4 : Square 2018 (Delta 123 4) = 1250184 :=
by
  sorry

end custom_operations

end compute_2018_square_123_Delta_4_l62_62492


namespace age_of_17th_student_is_75_l62_62572

variables (T A : ℕ)

def avg_17_students := 17
def avg_5_students := 14
def avg_9_students := 16
def total_17_students := 17 * avg_17_students
def total_5_students := 5 * avg_5_students
def total_9_students := 9 * avg_9_students
def age_17th_student : ℕ := total_17_students - (total_5_students + total_9_students)

theorem age_of_17th_student_is_75 :
  age_17th_student = 75 := by sorry

end age_of_17th_student_is_75_l62_62572


namespace team_overall_progress_is_89_l62_62928

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

def overall_progress (changes : List Int) : Int :=
  changes.sum

theorem team_overall_progress_is_89 :
  overall_progress yard_changes = 89 :=
by
  sorry

end team_overall_progress_is_89_l62_62928


namespace log_sqrt_defined_in_interval_l62_62301

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l62_62301


namespace quadratic_complete_square_l62_62619

theorem quadratic_complete_square (c n : ℝ) (h1 : ∀ x : ℝ, x^2 + c * x + 20 = (x + n)^2 + 12) (h2: 0 < c) : 
  c = 4 * Real.sqrt 2 :=
by
  sorry

end quadratic_complete_square_l62_62619


namespace find_num_managers_l62_62271

variable (num_associates : ℕ) (avg_salary_managers avg_salary_associates avg_salary_company : ℚ)
variable (num_managers : ℚ)

-- Define conditions based on given problem
def conditions := 
  num_associates = 75 ∧
  avg_salary_managers = 90000 ∧
  avg_salary_associates = 30000 ∧
  avg_salary_company = 40000

-- Proof problem statement
theorem find_num_managers (h : conditions num_associates avg_salary_managers avg_salary_associates avg_salary_company) :
  num_managers = 15 :=
sorry

end find_num_managers_l62_62271


namespace domain_of_f_x_minus_1_l62_62934

theorem domain_of_f_x_minus_1 (f : ℝ → ℝ) (h : ∀ x, x^2 + 1 ∈ Set.Icc 1 10 → x ∈ Set.Icc (-3 : ℝ) 2) :
  Set.Icc 2 (11 : ℝ) ⊆ {x : ℝ | x - 1 ∈ Set.Icc 1 10} :=
by
  sorry

end domain_of_f_x_minus_1_l62_62934


namespace total_travel_expenses_l62_62667

noncomputable def cost_of_fuel_tank := 45
noncomputable def miles_per_tank := 500
noncomputable def journey_distance := 2000
noncomputable def food_ratio := 3 / 5
noncomputable def hotel_cost_per_night := 80
noncomputable def number_of_hotel_nights := 3
noncomputable def fuel_cost_increase := 5

theorem total_travel_expenses :
  let number_of_refills := journey_distance / miles_per_tank
  let first_refill_cost := cost_of_fuel_tank
  let second_refill_cost := first_refill_cost + fuel_cost_increase
  let third_refill_cost := second_refill_cost + fuel_cost_increase
  let fourth_refill_cost := third_refill_cost + fuel_cost_increase
  let total_fuel_cost := first_refill_cost + second_refill_cost + third_refill_cost + fourth_refill_cost
  let total_food_cost := food_ratio * total_fuel_cost
  let total_hotel_cost := hotel_cost_per_night * number_of_hotel_nights
  let total_expenses := total_fuel_cost + total_food_cost + total_hotel_cost
  total_expenses = 576 := by sorry

end total_travel_expenses_l62_62667


namespace square_plot_area_l62_62237

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (s : ℝ) (A : ℝ)
  (h1 : price_per_foot = 58)
  (h2 : total_cost = 1160)
  (h3 : total_cost = 4 * s * price_per_foot)
  (h4 : A = s * s) :
  A = 25 := by
  sorry

end square_plot_area_l62_62237


namespace zero_is_neither_positive_nor_negative_l62_62783

theorem zero_is_neither_positive_nor_negative :
  ¬ (0 > 0) ∧ ¬ (0 < 0) :=
by
  sorry

end zero_is_neither_positive_nor_negative_l62_62783


namespace solve_equation1_solve_equation2_l62_62129

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  2 * x^2 = 3 * (2 * x + 1)

-- Define the solution set for the first equation
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2

-- Prove that the solutions for the first equation are correct
theorem solve_equation1 (x : ℝ) : equation1 x ↔ solution1 x :=
by
  sorry

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  3 * x * (x + 2) = 4 * x + 8

-- Define the solution set for the second equation
def solution2 (x : ℝ) : Prop :=
  x = -2 ∨ x = 4 / 3

-- Prove that the solutions for the second equation are correct
theorem solve_equation2 (x : ℝ) : equation2 x ↔ solution2 x :=
by
  sorry

end solve_equation1_solve_equation2_l62_62129


namespace find_angle_A_l62_62283

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : 2 * Real.sin B = Real.sqrt 3 * b) 
  (h2 : a = 2) (h3 : ∃ area : ℝ, area = Real.sqrt 3 ∧ area = (1 / 2) * b * c * Real.sin A) :
  A = Real.pi / 3 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_angle_A_l62_62283


namespace largest_angle_in_triangle_l62_62183

theorem largest_angle_in_triangle (a b c : ℝ)
  (h1 : a + b = (4 / 3) * 90)
  (h2 : b = a + 36)
  (h3 : a + b + c = 180) :
  max a (max b c) = 78 :=
sorry

end largest_angle_in_triangle_l62_62183


namespace sparrow_grains_l62_62187

theorem sparrow_grains (x : ℤ) : 9 * x < 1001 ∧ 10 * x > 1100 → x = 111 :=
by
  sorry

end sparrow_grains_l62_62187


namespace solution_set_of_inequality_system_l62_62971

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 2 ≤ 3 ∧ 1 + x > -2) ↔ (-3 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_of_inequality_system_l62_62971


namespace puzzle_pieces_missing_l62_62892

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end puzzle_pieces_missing_l62_62892


namespace no_perfect_squares_exist_l62_62012

theorem no_perfect_squares_exist (x y : ℕ) :
  ¬(∃ k1 k2 : ℕ, x^2 + y = k1^2 ∧ y^2 + x = k2^2) :=
sorry

end no_perfect_squares_exist_l62_62012


namespace evaluateExpression_at_1_l62_62947

noncomputable def evaluateExpression (x : ℝ) : ℝ :=
  (x^2 - 3 * x - 10) / (x - 5)

theorem evaluateExpression_at_1 : evaluateExpression 1 = 3 :=
by
  sorry

end evaluateExpression_at_1_l62_62947


namespace number_of_allowed_pairs_l62_62945

theorem number_of_allowed_pairs (total_books : ℕ) (prohibited_books : ℕ) : ℕ :=
  let total_pairs := (total_books * (total_books - 1)) / 2
  let prohibited_pairs := (prohibited_books * (prohibited_books - 1)) / 2
  total_pairs - prohibited_pairs

example : number_of_allowed_pairs 15 3 = 102 :=
by
  sorry

end number_of_allowed_pairs_l62_62945


namespace min_value_of_expression_l62_62163

theorem min_value_of_expression (x y : ℝ) (hposx : x > 0) (hposy : y > 0) (heq : 2 / x + 1 / y = 1) : 
  x + 2 * y ≥ 8 :=
sorry

end min_value_of_expression_l62_62163


namespace number_of_perpendicular_points_on_ellipse_l62_62728

theorem number_of_perpendicular_points_on_ellipse :
  ∃ (P : ℝ × ℝ), (P ∈ {P : ℝ × ℝ | (P.1^2 / 8) + (P.2^2 / 4) = 1})
  ∧ (∀ (F1 F2 : ℝ × ℝ), F1 ≠ F2 → ∀ (P : ℝ × ℝ), ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) = 0) :=
sorry

end number_of_perpendicular_points_on_ellipse_l62_62728


namespace union_of_M_and_N_l62_62782

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l62_62782


namespace find_common_difference_l62_62707

variable {a : ℕ → ℝ} (d : ℝ) (a₁ : ℝ)

-- defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ + n * d

-- condition for the sum of even indexed terms
def sum_even_terms (a : ℕ → ℝ) : ℝ := a 2 + a 4 + a 6 + a 8 + a 10

-- condition for the sum of odd indexed terms
def sum_odd_terms (a : ℕ → ℝ) : ℝ := a 1 + a 3 + a 5 + a 7 + a 9

-- main theorem to prove
theorem find_common_difference
  (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h_even_sum : sum_even_terms a = 30)
  (h_odd_sum : sum_odd_terms a = 25) :
  d = 1 := by
  sorry

end find_common_difference_l62_62707


namespace equation_of_line_through_A_parallel_to_given_line_l62_62720

theorem equation_of_line_through_A_parallel_to_given_line :
  ∃ c : ℝ, 
    (∀ x y : ℝ, 2 * x - y + c = 0 ↔ ∃ a b : ℝ, a = -1 ∧ b = 0 ∧ 2 * a - b + 1 = 0) :=
sorry

end equation_of_line_through_A_parallel_to_given_line_l62_62720


namespace inequality_proof_l62_62708

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l62_62708


namespace inequality_square_l62_62458

theorem inequality_square (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end inequality_square_l62_62458


namespace price_of_fruit_juice_l62_62046

theorem price_of_fruit_juice (F : ℝ)
  (Sandwich_price : ℝ := 2)
  (Hamburger_price : ℝ := 2)
  (Hotdog_price : ℝ := 1)
  (Selene_purchases : ℝ := 3 * Sandwich_price + F)
  (Tanya_purchases : ℝ := 2 * Hamburger_price + 2 * F)
  (Total_spent : Selene_purchases + Tanya_purchases = 16) :
  F = 2 :=
by
  sorry

end price_of_fruit_juice_l62_62046


namespace average_price_of_towels_l62_62290

-- Definitions based on the conditions
def cost_of_three_towels := 3 * 100
def cost_of_five_towels := 5 * 150
def cost_of_two_towels := 550
def total_cost := cost_of_three_towels + cost_of_five_towels + cost_of_two_towels
def total_number_of_towels := 3 + 5 + 2
def average_price := total_cost / total_number_of_towels

-- The theorem statement
theorem average_price_of_towels :
  average_price = 160 :=
by
  sorry

end average_price_of_towels_l62_62290


namespace sandy_tokens_ratio_l62_62463

theorem sandy_tokens_ratio :
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (difference : ℕ),
  total_tokens = 1000000 →
  num_siblings = 4 →
  difference = 375000 →
  ∃ (sandy_tokens : ℕ),
  sandy_tokens = (total_tokens - (num_siblings * ((total_tokens - difference) / (num_siblings + 1)))) ∧
  sandy_tokens / total_tokens = 1 / 2 :=
by 
  intros total_tokens num_siblings difference h1 h2 h3
  sorry

end sandy_tokens_ratio_l62_62463


namespace violet_balloons_count_l62_62166

-- Define the initial number of violet balloons
def initial_violet_balloons := 7

-- Define the number of violet balloons Jason lost
def lost_violet_balloons := 3

-- Define the remaining violet balloons after losing some
def remaining_violet_balloons := initial_violet_balloons - lost_violet_balloons

-- Prove that the remaining violet balloons is equal to 4
theorem violet_balloons_count : remaining_violet_balloons = 4 :=
by
  sorry

end violet_balloons_count_l62_62166


namespace school_band_fundraising_l62_62061

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end school_band_fundraising_l62_62061


namespace poly_a_c_sum_l62_62765

theorem poly_a_c_sum {a b c d : ℝ} (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 + a * x + b)
  (hg : ∀ x, g x = x^2 + c * x + d)
  (hv_f_root_g : g (-a / 2) = 0)
  (hv_g_root_f : f (-c / 2) = 0)
  (f_min : ∀ x, f x ≥ -25)
  (g_min : ∀ x, g x ≥ -25)
  (f_g_intersect : f 50 = -25 ∧ g 50 = -25) : a + c = -101 :=
by
  sorry

end poly_a_c_sum_l62_62765


namespace range_of_a_l62_62732

variable {x a : ℝ}

theorem range_of_a (h1 : x < 0) (h2 : 2 ^ x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l62_62732


namespace monotonic_increasing_quadratic_l62_62848

theorem monotonic_increasing_quadratic (b : ℝ) (c : ℝ) :
  (∀ x y : ℝ, (0 ≤ x → x ≤ y → (x^2 + b*x + c) ≤ (y^2 + b*y + c))) ↔ (b ≥ 0) :=
sorry  -- Proof is omitted

end monotonic_increasing_quadratic_l62_62848


namespace gcd_le_sqrt_sum_l62_62142

theorem gcd_le_sqrt_sum {a b : ℕ} (h : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  ↑(Nat.gcd a b) ≤ Real.sqrt (a + b) := sorry

end gcd_le_sqrt_sum_l62_62142


namespace circles_intersect_and_common_chord_l62_62138

open Real

def circle1 (x y : ℝ) := x^2 + y^2 - 6 * x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6 = 0

theorem circles_intersect_and_common_chord :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) ∧ (∀ x y : ℝ, circle1 x y → circle2 x y → 3 * x - 2 * y = 0) :=
by
  sorry

end circles_intersect_and_common_chord_l62_62138


namespace real_mul_eq_zero_iff_l62_62550

theorem real_mul_eq_zero_iff (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end real_mul_eq_zero_iff_l62_62550


namespace find_values_l62_62913

theorem find_values (a b : ℝ) 
  (h1 : a + b = 10)
  (h2 : a - b = 4) 
  (h3 : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ ab = 21 := 
by 
  sorry

end find_values_l62_62913


namespace correct_option_B_l62_62081

theorem correct_option_B (a : ℤ) : (2 * a) ^ 3 = 8 * a ^ 3 :=
by
  sorry

end correct_option_B_l62_62081


namespace fruit_basket_l62_62759

-- Define the quantities and their relationships
variables (O A B P : ℕ)

-- State the conditions
def condition1 : Prop := A = O - 2
def condition2 : Prop := B = 3 * A
def condition3 : Prop := P = B / 2
def condition4 : Prop := O + A + B + P = 28

-- State the theorem
theorem fruit_basket (h1 : condition1 O A) (h2 : condition2 A B) (h3 : condition3 B P) (h4 : condition4 O A B P) : O = 6 :=
sorry

end fruit_basket_l62_62759


namespace room_width_is_7_l62_62394

-- Define the conditions of the problem
def room_length : ℝ := 10
def room_height : ℝ := 5
def door_width : ℝ := 1
def door_height : ℝ := 3
def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5
def cost_per_sq_meter : ℝ := 3
def total_cost : ℝ := 474

-- Define the total cost to be painted
def total_area_painted (width : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height) + 2 * (width * room_height)
  let door_area := 2 * (door_width * door_height)
  let window_area := (window1_width * window1_height) + 2 * (window2_width * window2_height)
  wall_area - door_area - window_area

def cost_equation (width : ℝ) : Prop :=
  (total_cost / cost_per_sq_meter) = total_area_painted width

-- Prove that the width required to satisfy the painting cost equation is 7 meters
theorem room_width_is_7 : ∃ w : ℝ, cost_equation w ∧ w = 7 :=
by
  sorry

end room_width_is_7_l62_62394


namespace no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l62_62886

theorem no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 10000 ∧ (n % 10 = 0) ∧ (Prime n) → False :=
by
  sorry

end no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l62_62886


namespace square_area_l62_62072

theorem square_area (x : ℚ) (h : 3 * x - 12 = 15 - 2 * x) : (3 * (27 / 5) - 12)^2 = 441 / 25 :=
by
  sorry

end square_area_l62_62072


namespace total_cakes_served_l62_62481

-- Defining the values for cakes served during lunch and dinner
def lunch_cakes : ℤ := 6
def dinner_cakes : ℤ := 9

-- Stating the theorem that the total number of cakes served today is 15
theorem total_cakes_served : lunch_cakes + dinner_cakes = 15 :=
by
  sorry

end total_cakes_served_l62_62481


namespace sold_on_saturday_l62_62757

-- Define all the conditions provided in the question
def amount_sold_thursday : ℕ := 210
def amount_sold_friday : ℕ := 2 * amount_sold_thursday
def amount_sold_sunday (S : ℕ) : ℕ := (S / 2)
def total_planned_sold : ℕ := 500
def excess_sold : ℕ := 325

-- Total sold is the sum of sold amounts from Thursday to Sunday
def total_sold (S : ℕ) : ℕ := amount_sold_thursday + amount_sold_friday + S + amount_sold_sunday S

-- The theorem to prove
theorem sold_on_saturday : ∃ S : ℕ, total_sold S = total_planned_sold + excess_sold ∧ S = 130 :=
by
  sorry

end sold_on_saturday_l62_62757


namespace domain_of_function_l62_62830

theorem domain_of_function:
  {x : ℝ | x + 1 ≥ 0 ∧ 3 - x ≠ 0} = {x : ℝ | x ≥ -1 ∧ x ≠ 3} :=
by
  sorry

end domain_of_function_l62_62830


namespace polynomial_has_real_root_l62_62383

theorem polynomial_has_real_root (a b : ℝ) :
  ∃ x : ℝ, x^3 + a * x + b = 0 :=
sorry

end polynomial_has_real_root_l62_62383


namespace dvds_left_l62_62281

-- Define the initial conditions
def owned_dvds : Nat := 13
def sold_dvds : Nat := 6

-- Define the goal
theorem dvds_left (owned_dvds : Nat) (sold_dvds : Nat) : owned_dvds - sold_dvds = 7 :=
by
  sorry

end dvds_left_l62_62281


namespace intersection_A_B_l62_62241

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l62_62241


namespace neither_sufficient_nor_necessary_l62_62538

theorem neither_sufficient_nor_necessary (α β : ℝ) :
  (α + β = 90) ↔ ¬((α + β = 90) ↔ (Real.sin α + Real.sin β > 1)) :=
sorry

end neither_sufficient_nor_necessary_l62_62538


namespace find_b_l62_62161

theorem find_b (b : ℤ) (h_quad : ∃ m : ℤ, (x + m)^2 + 20 = x^2 + b * x + 56) (h_pos : b > 0) : b = 12 :=
sorry

end find_b_l62_62161


namespace min_value_geometric_sequence_l62_62853

theorem min_value_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h : 0 < q ∧ 0 < a 0) 
  (H : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) 
  (h_geom : ∀ n, a (n+1) = a n * q) : 
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end min_value_geometric_sequence_l62_62853


namespace corn_height_growth_l62_62288

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l62_62288


namespace scientific_notation_100000_l62_62598

theorem scientific_notation_100000 : ∃ a n, (1 ≤ a) ∧ (a < 10) ∧ (100000 = a * 10 ^ n) :=
by
  use 1, 5
  repeat { split }
  repeat { sorry }

end scientific_notation_100000_l62_62598


namespace income_of_A_l62_62009

theorem income_of_A (x y : ℝ) 
    (ratio_income : 5 * x = y * 4)
    (ratio_expenditure : 3 * x = y * 2)
    (savings_A : 5 * x - 3 * y = 1600)
    (savings_B : 4 * x - 2 * y = 1600) : 
    5 * x = 4000 := 
by
  sorry

end income_of_A_l62_62009


namespace max_sum_of_four_distinct_with_lcm_165_l62_62884

theorem max_sum_of_four_distinct_with_lcm_165 (a b c d : ℕ)
  (h1 : Nat.lcm a b = 165)
  (h2 : Nat.lcm a c = 165)
  (h3 : Nat.lcm a d = 165)
  (h4 : Nat.lcm b c = 165)
  (h5 : Nat.lcm b d = 165)
  (h6 : Nat.lcm c d = 165)
  (h7 : a ≠ b) (h8 : a ≠ c) (h9 : a ≠ d)
  (h10 : b ≠ c) (h11 : b ≠ d) (h12 : c ≠ d) :
  a + b + c + d ≤ 268 := sorry

end max_sum_of_four_distinct_with_lcm_165_l62_62884


namespace sum_x_coordinates_l62_62897

-- Define the equations of the line segments
def segment1 (x : ℝ) := 2 * x + 6
def segment2 (x : ℝ) := -0.5 * x - 1.5
def segment3 (x : ℝ) := 2 * x + 1
def segment4 (x : ℝ) := -0.5 * x + 3.5
def segment5 (x : ℝ) := 2 * x - 4

-- Definition of the problem
theorem sum_x_coordinates (h1 : segment1 (-5) = -4 ∧ segment1 (-3) = 0)
    (h2 : segment2 (-3) = 0 ∧ segment2 (-1) = -1)
    (h3 : segment3 (-1) = -1 ∧ segment3 (1) = 3)
    (h4 : segment4 (1) = 3 ∧ segment4 (3) = 2)
    (h5 : segment5 (3) = 2 ∧ segment5 (5) = 6)
    (hx1 : ∃ x1, segment3 x1 = 2.4 ∧ -1 ≤ x1 ∧ x1 ≤ 1)
    (hx2 : ∃ x2, segment4 x2 = 2.4 ∧ 1 ≤ x2 ∧ x2 ≤ 3)
    (hx3 : ∃ x3, segment5 x3 = 2.4 ∧ 3 ≤ x3 ∧ x3 ≤ 5) :
    (∃ (x1 x2 x3 : ℝ), segment3 x1 = 2.4 ∧ segment4 x2 = 2.4 ∧ segment5 x3 = 2.4 ∧ x1 = 0.7 ∧ x2 = 2.2 ∧ x3 = 3.2 ∧ x1 + x2 + x3 = 6.1) :=
sorry

end sum_x_coordinates_l62_62897


namespace find_principal_amount_l62_62629

noncomputable def principal_amount_loan (SI R T : ℝ) : ℝ :=
  SI / (R * T)

theorem find_principal_amount (SI R T : ℝ) (h_SI : SI = 6480) (h_R : R = 0.12) (h_T : T = 3) :
  principal_amount_loan SI R T = 18000 :=
by
  rw [principal_amount_loan, h_SI, h_R, h_T]
  norm_num

#check find_principal_amount

end find_principal_amount_l62_62629


namespace largest_number_is_b_l62_62798

noncomputable def a := 0.935
noncomputable def b := 0.9401
noncomputable def c := 0.9349
noncomputable def d := 0.9041
noncomputable def e := 0.9400

theorem largest_number_is_b : b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  -- proof can be filled in here
  sorry

end largest_number_is_b_l62_62798


namespace yellow_white_flowers_count_l62_62349

theorem yellow_white_flowers_count
    (RY RW : Nat)
    (hRY : RY = 17)
    (hRW : RW = 14)
    (hRedMoreThanWhite : (RY + RW) - (RW + YW) = 4) :
    ∃ YW, YW = 13 := 
by
  sorry

end yellow_white_flowers_count_l62_62349


namespace total_hotdogs_sold_l62_62604

theorem total_hotdogs_sold : 
  let small := 58.3
  let medium := 21.7
  let large := 35.9
  let extra_large := 15.4
  small + medium + large + extra_large = 131.3 :=
by 
  sorry

end total_hotdogs_sold_l62_62604


namespace rook_reaches_upper_right_in_expected_70_minutes_l62_62973

section RookMoves

noncomputable def E : ℝ := 70

-- Definition of expected number of minutes considering the row and column moves.
-- This is a direct translation from the problem's correct answer.
def rook_expected_minutes_to_upper_right (E_0 E_1 : ℝ) : Prop :=
  E_0 = (70 : ℝ) ∧ E_1 = (70 : ℝ)

theorem rook_reaches_upper_right_in_expected_70_minutes : E = 70 := sorry

end RookMoves

end rook_reaches_upper_right_in_expected_70_minutes_l62_62973


namespace max_ab_bc_cd_da_l62_62480

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
by sorry

end max_ab_bc_cd_da_l62_62480


namespace lean_proof_l62_62089

noncomputable def proof_problem (a b c d : ℝ) (habcd : a * b * c * d = 1) : Prop :=
  (1 + a * b) / (1 + a) ^ 2008 +
  (1 + b * c) / (1 + b) ^ 2008 +
  (1 + c * d) / (1 + c) ^ 2008 +
  (1 + d * a) / (1 + d) ^ 2008 ≥ 4

theorem lean_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_abcd : a * b * c * d = 1) : proof_problem a b c d h_abcd :=
  sorry

end lean_proof_l62_62089


namespace bobs_share_l62_62580

theorem bobs_share 
  (r : ℕ → ℕ → ℕ → Prop) (s : ℕ) 
  (h_ratio : r 1 2 3) 
  (bill_share : s = 300) 
  (hr : ∃ p, s = 2 * p) :
  ∃ b, b = 3 * (s / 2) ∧ b = 450 := 
by
  sorry

end bobs_share_l62_62580


namespace air_conditioning_price_november_l62_62994

noncomputable def price_in_november : ℝ :=
  let january_price := 470
  let february_price := january_price * (1 - 0.12)
  let march_price := february_price * (1 + 0.08)
  let april_price := march_price * (1 - 0.10)
  let june_price := april_price * (1 + 0.05)
  let august_price := june_price * (1 - 0.07)
  let october_price := august_price * (1 + 0.06)
  october_price * (1 - 0.15)

theorem air_conditioning_price_november : price_in_november = 353.71 := by
  sorry

end air_conditioning_price_november_l62_62994


namespace composite_has_at_least_three_factors_l62_62127

-- Definition of composite number in terms of its factors
def is_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Theorem stating that a composite number has at least 3 factors
theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : 
  (∃ f1 f2 f3, f1 ∣ n ∧ f2 ∣ n ∧ f3 ∣ n ∧ f1 ≠ 1 ∧ f1 ≠ n ∧ f2 ≠ 1 ∧ f2 ≠ n ∧ f3 ≠ 1 ∧ f3 ≠ n ∧ f1 ≠ f2 ∧ f2 ≠ f3) := 
sorry

end composite_has_at_least_three_factors_l62_62127


namespace optimal_fence_area_l62_62171

variables {l w : ℝ}

theorem optimal_fence_area
  (h1 : 2 * l + 2 * w = 400) -- Tiffany must use exactly 400 feet of fencing.
  (h2 : l ≥ 100) -- The length must be at least 100 feet.
  (h3 : w ≥ 50) -- The width must be at least 50 feet.
  : l * w ≤ 10000 :=      -- We need to prove that the area is at most 10000 square feet.
by
  sorry

end optimal_fence_area_l62_62171


namespace expression_approx_l62_62620

noncomputable def simplified_expression : ℝ :=
  (Real.sqrt 97 + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7)

theorem expression_approx : abs (simplified_expression - 3.002) < 0.001 :=
by
  -- Proof omitted
  sorry

end expression_approx_l62_62620


namespace repeating_decimal_fraction_sum_l62_62822

/-- The repeating decimal 3.171717... can be written as a fraction. When reduced to lowest
terms, the sum of the numerator and denominator of this fraction is 413. -/
theorem repeating_decimal_fraction_sum :
  let y := 3.17171717 -- The repeating decimal
  let frac_num := 314
  let frac_den := 99
  let sum := frac_num + frac_den
  y = frac_num / frac_den ∧ sum = 413 := by
  sorry

end repeating_decimal_fraction_sum_l62_62822


namespace max_vertices_of_divided_triangle_l62_62212

theorem max_vertices_of_divided_triangle (n : ℕ) (h : n ≥ 1) : 
  (∀ t : ℕ, t = 1000 → exists T : ℕ, T = (n + 2)) :=
by sorry

end max_vertices_of_divided_triangle_l62_62212


namespace monotonic_decreasing_interval_l62_62824

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 (Real.sqrt 3 / 3), (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l62_62824


namespace width_rectangular_box_5_cm_l62_62676

theorem width_rectangular_box_5_cm 
  (W : ℕ)
  (h_dim_wooden_box : (8 * 10 * 6 * 100 ^ 3) = 480000000) -- dimensions of the wooden box in cm³
  (h_dim_rectangular_box : (4 * W * 6) = (24 * W)) -- dimensions of the rectangular box in cm³
  (h_max_boxes : 4000000 * (24 * W) = 480000000) -- max number of boxes that fit in the wooden box
: 
  W = 5 := 
by
  sorry

end width_rectangular_box_5_cm_l62_62676


namespace apple_pies_count_l62_62438

def total_pies := 13
def pecan_pies := 4
def pumpkin_pies := 7
def apple_pies := total_pies - pecan_pies - pumpkin_pies

theorem apple_pies_count : apple_pies = 2 := by
  sorry

end apple_pies_count_l62_62438


namespace min_value_of_f_solve_inequality_l62_62440

noncomputable def f (x : ℝ) : ℝ := abs (x - 5/2) + abs (x - 1/2)

theorem min_value_of_f : (∀ x : ℝ, f x ≥ 2) ∧ (∃ x : ℝ, f x = 2) := by
  sorry

theorem solve_inequality (x : ℝ) : (f x ≤ x + 4) ↔ (-1/3 ≤ x ∧ x ≤ 7) := by
  sorry

end min_value_of_f_solve_inequality_l62_62440


namespace root_of_linear_eq_l62_62462

variable (a b : ℚ) -- Using rationals for coefficients

-- Define the linear equation
def linear_eq (x : ℚ) : Prop := a * x + b = 0

-- Define the root function
def root_function : ℚ := -b / a

-- State the goal
theorem root_of_linear_eq : linear_eq a b (root_function a b) :=
by
  unfold linear_eq
  unfold root_function
  sorry

end root_of_linear_eq_l62_62462


namespace fixed_point_of_f_l62_62455

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 4

theorem fixed_point_of_f (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  -- Skip the proof; it will be filled in the subsequent steps
  sorry

end fixed_point_of_f_l62_62455


namespace find_original_number_l62_62334

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end find_original_number_l62_62334


namespace number_of_citroens_submerged_is_zero_l62_62133

-- Definitions based on the conditions
variables (x y : ℕ) -- Define x as the number of Citroen and y as the number of Renault submerged
variables (r p c vr vp : ℕ) -- Define r as the number of Renault, p as the number of Peugeot, c as the number of Citroën

-- Given conditions translated
-- Condition 1: There were twice as many Renault cars as there were Peugeot cars
def condition1 (r p : ℕ) : Prop := r = 2 * p
-- Condition 2: There were twice as many Peugeot cars as there were Citroens
def condition2 (p c : ℕ) : Prop := p = 2 * c
-- Condition 3: As many Citroens as Renaults were submerged in the water
def condition3 (x y : ℕ) : Prop := y = x
-- Condition 4: Three times as many Renaults were in the water as there were Peugeots
def condition4 (r y : ℕ) : Prop := r = 3 * y
-- Condition 5: As many Peugeots visible in the water as there were Citroens
def condition5 (vp c : ℕ) : Prop := vp = c

-- The question to prove: The number of Citroen cars submerged is 0
theorem number_of_citroens_submerged_is_zero
  (h1 : condition1 r p) 
  (h2 : condition2 p c)
  (h3 : condition3 x y)
  (h4 : condition4 r y)
  (h5 : condition5 vp c) :
  x = 0 :=
sorry

end number_of_citroens_submerged_is_zero_l62_62133


namespace num_pens_l62_62228

theorem num_pens (pencils : ℕ) (students : ℕ) (pens : ℕ)
  (h_pencils : pencils = 520)
  (h_students : students = 40)
  (h_div : pencils % students = 0)
  (h_pens_per_student : pens = (pencils / students) * students) :
  pens = 520 := by
  sorry

end num_pens_l62_62228


namespace arithmetic_sequence_positive_l62_62099

theorem arithmetic_sequence_positive (d a_1 : ℤ) (n : ℤ) :
  (a_11 - a_8 = 3) -> 
  (S_11 - S_8 = 33) ->
  (n > 0) ->
  a_1 + (n-1) * d > 0 ->
  n = 10 :=
by
  sorry

end arithmetic_sequence_positive_l62_62099


namespace calculate_expression_l62_62657

theorem calculate_expression :
  (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end calculate_expression_l62_62657


namespace fourth_vertex_of_square_l62_62215

def A : ℂ := 2 - 3 * Complex.I
def B : ℂ := 3 + 2 * Complex.I
def C : ℂ := -3 + 2 * Complex.I

theorem fourth_vertex_of_square : ∃ D : ℂ, 
  (D - B) = (B - A) * Complex.I ∧ 
  (D - C) = (C - A) * Complex.I ∧ 
  (D = -3 + 8 * Complex.I) :=
sorry

end fourth_vertex_of_square_l62_62215


namespace range_of_a_zeros_of_g_l62_62566

-- Definitions for the original functions f and g and their corresponding conditions
noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

noncomputable def g (x x2 a : ℝ) : ℝ := f x a - (x2 / 2)

-- Proving the range of a
theorem range_of_a (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  0 < a ∧ a < 1 := 
sorry

-- Proving the number of zeros of g based on the value of a
theorem zeros_of_g (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  (0 < a ∧ a < 3 / Real.exp 2 → ∃ x3 x4, x3 ≠ x4 ∧ g x3 x2 a = 0 ∧ g x4 x2 a = 0) ∧
  (a = 3 / Real.exp 2 → ∃ x3, g x3 x2 a = 0) ∧
  (3 / Real.exp 2 < a ∧ a < 1 → ∀ x, g x x2 a ≠ 0) :=
sorry

end range_of_a_zeros_of_g_l62_62566


namespace find_common_difference_l62_62147

variable {a : ℕ → ℝ}
variable {p q : ℕ}
variable {d : ℝ}

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) (p : ℕ) := a p = 4
def condition2 (a : ℕ → ℝ) (q : ℕ) := a q = 2
def condition3 (p q : ℕ) := p = 4 + q

-- The goal statement
theorem find_common_difference
  (a_seq : arithmetic_sequence a d)
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q) :
  d = 1 / 2 :=
by
  sorry

end find_common_difference_l62_62147


namespace shortest_side_of_right_triangle_l62_62987

theorem shortest_side_of_right_triangle
  (a b c : ℝ)
  (h : a = 5) (k : b = 13) (rightangled : a^2 + c^2 = b^2) : c = 12 := 
sorry

end shortest_side_of_right_triangle_l62_62987


namespace steve_book_earning_l62_62820

theorem steve_book_earning
  (total_copies : ℕ)
  (advance_copies : ℕ)
  (total_kept : ℝ)
  (agent_cut_percentage : ℝ)
  (copies : ℕ)
  (money_kept : ℝ)
  (x : ℝ)
  (h1 : total_copies = 1000000)
  (h2 : advance_copies = 100000)
  (h3 : total_kept = 1620000)
  (h4 : agent_cut_percentage = 0.10)
  (h5 : copies = total_copies - advance_copies)
  (h6 : money_kept = copies * (1 - agent_cut_percentage) * x)
  (h7 : money_kept = total_kept) :
  x = 2 := 
by 
  sorry

end steve_book_earning_l62_62820


namespace max_product_of_two_integers_with_sum_180_l62_62803

theorem max_product_of_two_integers_with_sum_180 :
  ∃ x y : ℤ, (x + y = 180) ∧ (x * y = 8100) := by
  sorry

end max_product_of_two_integers_with_sum_180_l62_62803


namespace least_homeowners_l62_62923

theorem least_homeowners (M W : ℕ) (total_members : M + W = 150)
  (men_homeowners : ∃ n : ℕ, n = 10 * M / 100) 
  (women_homeowners : ∃ n : ℕ, n = 20 * W / 100) : 
  ∃ homeowners : ℕ, homeowners = 16 := 
sorry

end least_homeowners_l62_62923


namespace largest_number_l62_62514

theorem largest_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 42) (h_dvd_a : 42 ∣ a) (h_dvd_b : 42 ∣ b)
  (a_eq : a = 42 * 11) (b_eq : b = 42 * 12) : max a b = 504 := by
  sorry

end largest_number_l62_62514


namespace triangle_inscribed_and_arcs_l62_62418

theorem triangle_inscribed_and_arcs
  (PQ QR PR : ℝ) (X Y Z : ℝ)
  (QY XZ QX YZ PX RY : ℝ)
  (H1 : PQ = 26)
  (H2 : QR = 28) 
  (H3 : PR = 27)
  (H4 : QY = XZ)
  (H5 : QX = YZ)
  (H6 : PX = RY)
  (H7 : RY = PX + 1)
  (H8 : XZ = QX + 1)
  (H9 : QY = YZ + 2) :
  QX = 29 / 2 :=
by
  sorry

end triangle_inscribed_and_arcs_l62_62418


namespace girls_25_percent_less_false_l62_62512

theorem girls_25_percent_less_false (g b : ℕ) (h : b = g * 125 / 100) : (b - g) / b ≠ 25 / 100 := by
  sorry

end girls_25_percent_less_false_l62_62512


namespace athlete_running_minutes_l62_62317

theorem athlete_running_minutes (r w : ℕ) 
  (h1 : r + w = 60)
  (h2 : 10 * r + 4 * w = 450) : 
  r = 35 := 
sorry

end athlete_running_minutes_l62_62317


namespace pizza_slices_left_l62_62366

theorem pizza_slices_left (total_slices : ℕ) (angeli_slices : ℚ) (marlon_slices : ℚ) 
  (H1 : total_slices = 8) (H2 : angeli_slices = 3/2) (H3 : marlon_slices = 3/2) :
  total_slices - (angeli_slices + marlon_slices) = 5 :=
by
  sorry

end pizza_slices_left_l62_62366


namespace editors_min_count_l62_62050

theorem editors_min_count
  (writers : ℕ)
  (P : ℕ)
  (S : ℕ)
  (W : ℕ)
  (H1 : writers = 45)
  (H2 : P = 90)
  (H3 : ∀ x : ℕ, x ≤ 6 → (90 = (writers + W - x) + 2 * x) → W ≥ P - 51)
  : W = 39 := by
  sorry

end editors_min_count_l62_62050


namespace prob_c_not_adjacent_to_a_or_b_l62_62098

-- Definitions for the conditions
def num_students : ℕ := 7
def a_and_b_together : Prop := true
def c_on_edge : Prop := true

-- Main theorem: probability c not adjacent to a or b under given conditions
theorem prob_c_not_adjacent_to_a_or_b
  (h1 : a_and_b_together)
  (h2 : c_on_edge) :
  ∃ (p : ℚ), p = 0.8 := by
  sorry

end prob_c_not_adjacent_to_a_or_b_l62_62098


namespace suitable_for_systematic_sampling_l62_62191

def city_districts : ℕ := 2000
def student_ratio : List ℕ := [3, 2, 8, 2]
def sample_size_city : ℕ := 200
def total_components : ℕ := 2000

def condition_A : Prop := 
  city_districts = 2000 ∧ 
  student_ratio = [3, 2, 8, 2] ∧ 
  sample_size_city = 200

def condition_B : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 5

def condition_C : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 200

def condition_D : Prop := 
  ∃ (n : ℕ), n = 20 ∧ n = 5

theorem suitable_for_systematic_sampling : condition_C :=
by
  sorry

end suitable_for_systematic_sampling_l62_62191


namespace total_pencils_crayons_l62_62940

theorem total_pencils_crayons (r : ℕ) (p : ℕ) (c : ℕ) 
  (hp : p = 31) (hc : c = 27) (hr : r = 11) : 
  r * p + r * c = 638 := 
  by
  sorry

end total_pencils_crayons_l62_62940


namespace johns_brother_age_l62_62196

variable (B : ℕ)
variable (J : ℕ)

-- Conditions given in the problem
def condition1 : Prop := J = 6 * B - 4
def condition2 : Prop := J + B = 10

-- The statement we want to prove, which is the answer to the problem:
theorem johns_brother_age (h1 : condition1 B J) (h2 : condition2 B J) : B = 2 := 
by 
  sorry

end johns_brother_age_l62_62196


namespace shaded_area_is_correct_l62_62331

-- Defining the conditions
def grid_width : ℝ := 15 -- in units
def grid_height : ℝ := 5 -- in units
def total_grid_area : ℝ := grid_width * grid_height -- in square units

def larger_triangle_base : ℝ := grid_width -- in units
def larger_triangle_height : ℝ := grid_height -- in units
def larger_triangle_area : ℝ := 0.5 * larger_triangle_base * larger_triangle_height -- in square units

def smaller_triangle_base : ℝ := 3 -- in units
def smaller_triangle_height : ℝ := 2 -- in units
def smaller_triangle_area : ℝ := 0.5 * smaller_triangle_base * smaller_triangle_height -- in square units

-- The total area of the triangles that are not shaded
def unshaded_areas : ℝ := larger_triangle_area + smaller_triangle_area

-- The area of the shaded region
def shaded_area : ℝ := total_grid_area - unshaded_areas

-- The statement to be proven
theorem shaded_area_is_correct : shaded_area = 34.5 := 
by 
  -- This is a placeholder for the actual proof, which would normally go here
  sorry

end shaded_area_is_correct_l62_62331


namespace find_x_for_slope_l62_62119

theorem find_x_for_slope (x : ℝ) (h : (2 - 5) / (x - (-3)) = -1 / 4) : x = 9 :=
by 
  -- Proof skipped
  sorry

end find_x_for_slope_l62_62119


namespace angles_of_triangle_l62_62574

theorem angles_of_triangle 
  (α β γ : ℝ)
  (triangle_ABC : α + β + γ = 180)
  (median_bisector_height : (γ / 4) * 4 = 90) :
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end angles_of_triangle_l62_62574


namespace petya_purchase_cost_l62_62929

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l62_62929


namespace problem_293_l62_62459

theorem problem_293 (s : ℝ) (R' : ℝ) (rectangle1 : ℝ) (circle1 : ℝ) 
  (condition1 : s = 4) 
  (condition2 : rectangle1 = 2 * 4) 
  (condition3 : circle1 = Real.pi * 1^2) 
  (condition4 : R' = s^2 - (rectangle1 + circle1)) 
  (fraction_form : ∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n) : 
  (∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n ∧ m + n = 293) := 
sorry

end problem_293_l62_62459


namespace determine_squirrel_color_l62_62179

-- Define the types for Squirrel species and the nuts in hollows
inductive Squirrel
| red
| gray

def tells_truth (s : Squirrel) : Prop :=
  s = Squirrel.red

def lies (s : Squirrel) : Prop :=
  s = Squirrel.gray

-- Statements made by the squirrel in front of the second hollow
def statement1 (s : Squirrel) (no_nuts_in_first : Prop) : Prop :=
  tells_truth s → no_nuts_in_first ∧ (lies s → ¬no_nuts_in_first)

def statement2 (s : Squirrel) (nuts_in_either : Prop) : Prop :=
  tells_truth s → nuts_in_either ∧ (lies s → ¬nuts_in_either)

-- Given a squirrel that says the statements and the information about truth and lies
theorem determine_squirrel_color (s : Squirrel) (no_nuts_in_first : Prop) (nuts_in_either : Prop) :
  (statement1 s no_nuts_in_first) ∧ (statement2 s nuts_in_either) → s = Squirrel.red :=
by
  sorry

end determine_squirrel_color_l62_62179


namespace first_plane_passengers_l62_62476

-- Definitions and conditions
def speed_plane_empty : ℕ := 600
def slowdown_per_passenger : ℕ := 2
def second_plane_passengers : ℕ := 60
def third_plane_passengers : ℕ := 40
def average_speed : ℕ := 500

-- Definition of the speed of a plane given number of passengers
def speed (passengers : ℕ) : ℕ := speed_plane_empty - slowdown_per_passenger * passengers

-- The problem statement rewritten in Lean 4
theorem first_plane_passengers (P : ℕ) (h_avg : (speed P + speed second_plane_passengers + speed third_plane_passengers) / 3 = average_speed) : P = 50 :=
sorry

end first_plane_passengers_l62_62476


namespace solve_equation_l62_62991

theorem solve_equation : ∀ x y : ℤ, x^2 + y^2 = 3 * x * y → x = 0 ∧ y = 0 := by
  intros x y h
  sorry

end solve_equation_l62_62991


namespace obtain_angle_10_30_l62_62457

theorem obtain_angle_10_30 (a : ℕ) (h : 100 + a = 135) : a = 35 := 
by sorry

end obtain_angle_10_30_l62_62457


namespace final_lives_equals_20_l62_62549

def initial_lives : ℕ := 30
def lives_lost : ℕ := 12
def bonus_lives : ℕ := 5
def penalty_lives : ℕ := 3

theorem final_lives_equals_20 : (initial_lives - lives_lost + bonus_lives - penalty_lives) = 20 :=
by 
  sorry

end final_lives_equals_20_l62_62549


namespace solve_equation_l62_62611

theorem solve_equation (x : ℝ) (h : (x - 3) / 2 - (2 * x) / 3 = 1) : x = -15 := 
by 
  sorry

end solve_equation_l62_62611


namespace second_player_wins_when_2003_candies_l62_62558

def game_winning_strategy (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 2

theorem second_player_wins_when_2003_candies :
  game_winning_strategy 2003 = 2 :=
by 
  sorry

end second_player_wins_when_2003_candies_l62_62558


namespace walter_age_1999_l62_62871

variable (w g : ℕ) -- represents Walter's age (w) and his grandmother's age (g) in 1994
variable (birth_sum : ℕ) (w_age_1994 : ℕ) (g_age_1994 : ℕ)

axiom h1 : g = 2 * w
axiom h2 : (1994 - w) + (1994 - g) = 3838

theorem walter_age_1999 (w g : ℕ) (h1 : g = 2 * w) (h2 : (1994 - w) + (1994 - g) = 3838) : w + 5 = 55 :=
by
  sorry

end walter_age_1999_l62_62871


namespace rayden_spent_more_l62_62890

-- Define the conditions
def lily_ducks := 20
def lily_geese := 10
def lily_chickens := 5
def lily_pigeons := 30

def rayden_ducks := 3 * lily_ducks
def rayden_geese := 4 * lily_geese
def rayden_chickens := 5 * lily_chickens
def rayden_pigeons := lily_pigeons / 2

def duck_price := 15
def geese_price := 20
def chicken_price := 10
def pigeon_price := 5

def lily_total := lily_ducks * duck_price +
                  lily_geese * geese_price +
                  lily_chickens * chicken_price +
                  lily_pigeons * pigeon_price

def rayden_total := rayden_ducks * duck_price +
                    rayden_geese * geese_price +
                    rayden_chickens * chicken_price +
                    rayden_pigeons * pigeon_price

def spending_difference := rayden_total - lily_total

theorem rayden_spent_more : spending_difference = 1325 := 
by 
  unfold spending_difference rayden_total lily_total -- to simplify the definitions
  sorry -- Proof is omitted

end rayden_spent_more_l62_62890


namespace problem_solution_l62_62172

theorem problem_solution :
  0.45 * 0.65 + 0.1 * 0.2 = 0.3125 :=
by
  sorry

end problem_solution_l62_62172


namespace sum_of_digits_joey_age_l62_62845

def int.multiple (a b : ℕ) := ∃ k : ℕ, a = k * b

theorem sum_of_digits_joey_age (J C M n : ℕ) (h1 : J = C + 2) (h2 : M = 2) (h3 : ∃ k, C = k * M) (h4 : C = 12) (h5 : J + n = 26) : 
  (2 + 6 = 8) :=
by
  sorry

end sum_of_digits_joey_age_l62_62845


namespace cos_20_cos_10_minus_sin_160_sin_10_l62_62229

theorem cos_20_cos_10_minus_sin_160_sin_10 : 
  (Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
   Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 
   Real.cos (30 * Real.pi / 180) :=
by
  sorry

end cos_20_cos_10_minus_sin_160_sin_10_l62_62229


namespace neither_directly_nor_inversely_proportional_A_D_l62_62734

-- Definitions for the equations where y is neither directly nor inversely proportional to x
def equationA (x y : ℝ) : Prop := x^2 + x * y = 0
def equationD (x y : ℝ) : Prop := 4 * x + y^2 = 7

-- Definition for direct or inverse proportionality
def isDirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x
def isInverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Proposition that y is neither directly nor inversely proportional to x for equations A and D
theorem neither_directly_nor_inversely_proportional_A_D (x y : ℝ) :
  equationA x y ∧ equationD x y ∧ ¬isDirectlyProportional x y ∧ ¬isInverselyProportional x y :=
by sorry

end neither_directly_nor_inversely_proportional_A_D_l62_62734


namespace single_discount_percentage_l62_62337

noncomputable def original_price : ℝ := 9795.3216374269
noncomputable def sale_price : ℝ := 6700
noncomputable def discount_percentage (p₀ p₁ : ℝ) : ℝ := ((p₀ - p₁) / p₀) * 100

theorem single_discount_percentage :
  discount_percentage original_price sale_price = 31.59 := 
by
  sorry

end single_discount_percentage_l62_62337


namespace range_of_k_l62_62603

theorem range_of_k (k : ℝ) : 
  (∀ x, x ∈ {x | -3 ≤ x ∧ x ≤ 2} ∩ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1} ↔ x ∈ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1}) →
   -1 ≤ k ∧ k ≤ 1 / 2 :=
by sorry

end range_of_k_l62_62603


namespace score_difference_proof_l62_62075

variable (α β γ δ : ℝ)

theorem score_difference_proof
  (h1 : α + β = γ + δ + 17)
  (h2 : α = β - 4)
  (h3 : γ = δ + 5) :
  β - δ = 13 :=
by
  -- proof goes here
  sorry

end score_difference_proof_l62_62075


namespace total_population_expr_l62_62793

-- Definitions of the quantities
variables (b g t : ℕ)

-- Conditions
axiom boys_as_girls : b = 3 * g
axiom girls_as_teachers : g = 9 * t

-- Theorem to prove
theorem total_population_expr : b + g + t = 37 * b / 27 :=
by
  sorry

end total_population_expr_l62_62793


namespace find_value_l62_62710

theorem find_value (x : ℝ) (f₁ f₂ : ℝ) (p : ℝ) (y₁ y₂ : ℝ) 
  (h1 : x * f₁ = (p * x) * y₁)
  (h2 : x * f₂ = (p * x) * y₂)
  (hf₁ : f₁ = 1 / 3)
  (hx : x = 4)
  (hy₁ : y₁ = 8)
  (hf₂ : f₂ = 1 / 8):
  y₂ = 3 := by
sorry

end find_value_l62_62710


namespace bean_inside_inscribed_circle_l62_62044

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * a * a

noncomputable def inscribed_circle_radius (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * a

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r * r

noncomputable def probability_inside_circle (s_triangle s_circle : ℝ) : ℝ :=
  s_circle / s_triangle

theorem bean_inside_inscribed_circle :
  let a := 2
  let s_triangle := equilateral_triangle_area a
  let r := inscribed_circle_radius a
  let s_circle := circle_area r
  probability_inside_circle s_triangle s_circle = (Real.sqrt 3 * Real.pi / 9) :=
by
  sorry

end bean_inside_inscribed_circle_l62_62044


namespace survey_response_total_l62_62100

theorem survey_response_total
  (X Y Z : ℕ)
  (h_ratio : X / 4 = Y / 2 ∧ X / 4 = Z)
  (h_X : X = 200) :
  X + Y + Z = 350 :=
sorry

end survey_response_total_l62_62100


namespace geometric_sequence_sum_ratio_l62_62370

theorem geometric_sequence_sum_ratio (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_nonzero_q : q ≠ 0) 
  (a2 : a_n 2 = a_n 1 * q) (a5 : a_n 5 = a_n 1 * q^4) 
  (h_condition : 8 * a_n 2 + a_n 5 = 0)
  (h_sum : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) : 
  S 5 / S 2 = -11 :=
by 
  sorry

end geometric_sequence_sum_ratio_l62_62370


namespace problem1_correct_problem2_correct_l62_62310

noncomputable def problem1_solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

noncomputable def problem2_solution_set : Set ℝ := {x | (-3 ≤ x ∧ x < 1) ∨ (3 < x ∧ x ≤ 7)}

theorem problem1_correct (x : ℝ) :
  (4 - x) / (x^2 + x + 1) ≤ 1 ↔ x ∈ problem1_solution_set :=
sorry

theorem problem2_correct (x : ℝ) :
  (1 < |x - 2| ∧ |x - 2| ≤ 5) ↔ x ∈ problem2_solution_set :=
sorry

end problem1_correct_problem2_correct_l62_62310


namespace find_ab_l62_62542

theorem find_ab (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 :=
by 
  sorry

end find_ab_l62_62542


namespace roots_of_equation_l62_62079

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l62_62079


namespace tims_total_earnings_l62_62614

theorem tims_total_earnings (days_of_week : ℕ) (tasks_per_day : ℕ) (tasks_40_rate : ℕ) (tasks_30_rate1 : ℕ) (tasks_30_rate2 : ℕ)
    (rate_40 : ℝ) (rate_30_1 : ℝ) (rate_30_2 : ℝ) (bonus_per_50 : ℝ) (performance_bonus : ℝ)
    (total_earnings : ℝ) :
  days_of_week = 6 →
  tasks_per_day = 100 →
  tasks_40_rate = 40 →
  tasks_30_rate1 = 30 →
  tasks_30_rate2 = 30 →
  rate_40 = 1.2 →
  rate_30_1 = 1.5 →
  rate_30_2 = 2.0 →
  bonus_per_50 = 10 →
  performance_bonus = 20 →
  total_earnings = 1058 :=
by
  intros
  sorry

end tims_total_earnings_l62_62614


namespace books_left_correct_l62_62505

variable (initial_books : ℝ) (sold_books : ℝ)

def number_of_books_left (initial_books sold_books : ℝ) : ℝ :=
  initial_books - sold_books

theorem books_left_correct :
  number_of_books_left 51.5 45.75 = 5.75 :=
by
  sorry

end books_left_correct_l62_62505


namespace ratio_345_iff_arithmetic_sequence_l62_62407

-- Define the variables and the context
variables (a b c : ℕ) -- assuming non-negative integers for simplicity
variable (k : ℕ) -- scaling factor for the 3:4:5 ratio
variable (d : ℕ) -- common difference in the arithmetic sequence

-- Conditions given
def isRightAngledTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ a < b ∧ b < c

def is345Ratio (a b c : ℕ) : Prop :=
  ∃ k, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k

def formsArithmeticSequence (a b c : ℕ) : Prop :=
  ∃ d, b = a + d ∧ c = b + d 

-- The statement to prove: sufficiency and necessity
theorem ratio_345_iff_arithmetic_sequence 
  (h_triangle : isRightAngledTriangle a b c) :
  (is345Ratio a b c ↔ formsArithmeticSequence a b c) :=
sorry

end ratio_345_iff_arithmetic_sequence_l62_62407


namespace complex_number_in_fourth_quadrant_l62_62038

theorem complex_number_in_fourth_quadrant (i : ℂ) (z : ℂ) (hx : z = -2 * i + 1) (hy : (z.re, z.im) = (1, -2)) :
  (1, -2).1 > 0 ∧ (1, -2).2 < 0 :=
by
  sorry

end complex_number_in_fourth_quadrant_l62_62038


namespace ratio_of_allergic_to_peanut_to_total_l62_62361

def total_children : ℕ := 34
def children_not_allergic_to_cashew : ℕ := 10
def children_allergic_to_both : ℕ := 10
def children_allergic_to_cashew : ℕ := 18
def children_not_allergic_to_any : ℕ := 6
def children_allergic_to_peanut : ℕ := 20

theorem ratio_of_allergic_to_peanut_to_total :
  (children_allergic_to_peanut : ℚ) / (total_children : ℚ) = 10 / 17 :=
by
  sorry

end ratio_of_allergic_to_peanut_to_total_l62_62361


namespace prove_b_zero_l62_62330

variables {a b c : ℕ}

theorem prove_b_zero (h1 : ∃ (a b c : ℕ), a^5 + 4 * b^5 = c^5 ∧ c % 2 = 0) : b = 0 :=
sorry

end prove_b_zero_l62_62330


namespace problem_statement_l62_62827

theorem problem_statement (a b : ℕ) (ha : a = 55555) (hb : b = 66666) :
  55554 * 55559 * 55552 - 55556 * 55551 * 55558 =
  66665 * 66670 * 66663 - 66667 * 66662 * 66669 := 
by
  sorry

end problem_statement_l62_62827


namespace equation_solution_l62_62202

theorem equation_solution (x : ℝ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) := 
sorry

end equation_solution_l62_62202


namespace jerry_weekly_earnings_l62_62656

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end jerry_weekly_earnings_l62_62656


namespace smallest_x_l62_62220

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y < 10) (h5 : y - x = 5) :
  x = 4 :=
sorry

end smallest_x_l62_62220


namespace baggies_of_oatmeal_cookies_l62_62668

theorem baggies_of_oatmeal_cookies (total_cookies : ℝ) (chocolate_chip_cookies : ℝ) (cookies_per_baggie : ℝ) 
(h_total : total_cookies = 41)
(h_choc : chocolate_chip_cookies = 13)
(h_baggie : cookies_per_baggie = 9) : 
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_baggie⌋ = 3 := 
by 
  sorry

end baggies_of_oatmeal_cookies_l62_62668


namespace average_speed_ratio_l62_62398

theorem average_speed_ratio
  (time_eddy : ℕ)
  (time_freddy : ℕ)
  (distance_ab : ℕ)
  (distance_ac : ℕ)
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_ab = 570)
  (h4 : distance_ac = 300) :
  (distance_ab / time_eddy) / (distance_ac / time_freddy) = 38 / 15 := 
by
  sorry

end average_speed_ratio_l62_62398


namespace multiple_of_n_eventually_written_l62_62696

theorem multiple_of_n_eventually_written (a b n : ℕ) (h_a_pos: 0 < a) (h_b_pos: 0 < b)  (h_ab_neq: a ≠ b) (h_n_pos: 0 < n) :
  ∃ m : ℕ, m % n = 0 :=
sorry

end multiple_of_n_eventually_written_l62_62696


namespace arithmetic_sequence_general_term_absolute_sum_first_19_terms_l62_62925

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) :
  ∀ n : ℕ, a n = 28 - 3 * n := 
sorry

theorem absolute_sum_first_19_terms (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) (an_eq : ∀ n : ℕ, a n = 28 - 3 * n) :
  |a 1| + |a 3| + |a 5| + |a 7| + |a 9| + |a 11| + |a 13| + |a 15| + |a 17| + |a 19| = 150 := 
sorry

end arithmetic_sequence_general_term_absolute_sum_first_19_terms_l62_62925


namespace sum_of_perimeters_l62_62766

theorem sum_of_perimeters (A1 A2 : ℝ) (h1 : A1 + A2 = 145) (h2 : A1 - A2 = 25) :
  4 * Real.sqrt 85 + 4 * Real.sqrt 60 = 4 * Real.sqrt A1 + 4 * Real.sqrt A2 :=
by
  sorry

end sum_of_perimeters_l62_62766


namespace wheel_distance_covered_l62_62762

noncomputable def diameter : ℝ := 15
noncomputable def revolutions : ℝ := 11.210191082802547
noncomputable def pi : ℝ := Real.pi -- or you can use the approximate value if required: 3.14159
noncomputable def circumference : ℝ := pi * diameter
noncomputable def distance_covered : ℝ := circumference * revolutions

theorem wheel_distance_covered :
  distance_covered = 528.316820577 := 
by
  unfold distance_covered
  unfold circumference
  unfold diameter
  unfold revolutions
  norm_num
  sorry

end wheel_distance_covered_l62_62762


namespace find_original_number_l62_62508

theorem find_original_number
  (x : ℤ)
  (h : 3 * (2 * x + 5) = 123) :
  x = 18 := 
sorry

end find_original_number_l62_62508


namespace sqrt_7_irrational_l62_62894

theorem sqrt_7_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a: ℝ) / b = Real.sqrt 7 := by
  sorry

end sqrt_7_irrational_l62_62894


namespace find_omega_value_l62_62693

theorem find_omega_value (ω : ℝ) (h : ω > 0) (h_dist : (1/2) * (2 * π / ω) = π / 6) : ω = 6 :=
by
  sorry

end find_omega_value_l62_62693


namespace least_possible_value_of_smallest_integer_l62_62573

theorem least_possible_value_of_smallest_integer 
  (A B C D : ℤ) 
  (H_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (H_avg : (A + B + C + D) / 4 = 74)
  (H_max : D = 90) :
  A ≥ 31 :=
by sorry

end least_possible_value_of_smallest_integer_l62_62573


namespace most_likely_composition_l62_62529

def event_a : Prop := (1 / 3) * (1 / 3) * 2 = (2 / 9)
def event_d : Prop := 2 * (1 / 3 * 1 / 3) = (2 / 9)

theorem most_likely_composition :
  event_a ∧ event_d :=
by sorry

end most_likely_composition_l62_62529


namespace upper_limit_of_arun_weight_l62_62895

variable (w : ℝ)

noncomputable def arun_opinion (w : ℝ) := 62 < w ∧ w < 72
noncomputable def brother_opinion (w : ℝ) := 60 < w ∧ w < 70
noncomputable def average_weight := 64

theorem upper_limit_of_arun_weight 
  (h1 : ∀ w, arun_opinion w → brother_opinion w → 64 = (62 + w) / 2 ) 
  : ∀ w, arun_opinion w ∧ brother_opinion w → w ≤ 66 :=
sorry

end upper_limit_of_arun_weight_l62_62895


namespace dot_product_equivalence_l62_62638

variable (a : ℝ × ℝ) 
variable (b : ℝ × ℝ)

-- Given conditions
def condition_1 : Prop := a = (2, 1)
def condition_2 : Prop := a - b = (-1, 2)

-- Goal
theorem dot_product_equivalence (h1 : condition_1 a) (h2 : condition_2 a b) : a.1 * b.1 + a.2 * b.2 = 5 :=
  sorry

end dot_product_equivalence_l62_62638


namespace pencils_to_sell_l62_62954

/--
A store owner bought 1500 pencils at $0.10 each. 
Each pencil is sold for $0.25. 
He wants to make a profit of exactly $100. 
Prove that he must sell 1000 pencils to achieve this profit.
-/
theorem pencils_to_sell (total_pencils : ℕ) (cost_per_pencil : ℝ) (selling_price_per_pencil : ℝ) (desired_profit : ℝ)
  (h1 : total_pencils = 1500)
  (h2 : cost_per_pencil = 0.10)
  (h3 : selling_price_per_pencil = 0.25)
  (h4 : desired_profit = 100) :
  total_pencils * cost_per_pencil + desired_profit = 1000 * selling_price_per_pencil :=
by
  -- Since Lean code requires some proof content, we put sorry to skip it.
  sorry

end pencils_to_sell_l62_62954


namespace inv_matrix_eq_l62_62221

variable (a : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 1, a])
variable (A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![a, -3; -1, a])

theorem inv_matrix_eq : (A⁻¹ = A_inv) → (a = 2) := 
by 
  sorry

end inv_matrix_eq_l62_62221


namespace magician_identifies_card_l62_62936

def Grid : Type := Fin 6 → Fin 6 → Nat

def choose_card (g : Grid) (c : Fin 6) (r : Fin 6) : Nat := g r c

def rearrange_columns_to_rows (s : List Nat) : Grid :=
  λ r c => s.get! (r.val * 6 + c.val)

theorem magician_identifies_card (g : Grid) (c1 : Fin 6) (r2 : Fin 6) :
  ∃ (card : Nat), (choose_card g c1 r2 = card) :=
  sorry

end magician_identifies_card_l62_62936


namespace attendance_second_concert_l62_62445

-- Define the given conditions
def attendance_first_concert : ℕ := 65899
def additional_people : ℕ := 119

-- Prove the number of people at the second concert
theorem attendance_second_concert : 
  attendance_first_concert + additional_people = 66018 := 
by
  -- Placeholder for the proof
  sorry

end attendance_second_concert_l62_62445


namespace bowling_team_avg_weight_l62_62018

noncomputable def total_weight (weights : List ℕ) : ℕ :=
  weights.foldr (· + ·) 0

noncomputable def average_weight (weights : List ℕ) : ℚ :=
  total_weight weights / weights.length

theorem bowling_team_avg_weight :
  let original_weights := [76, 76, 76, 76, 76, 76, 76]
  let new_weights := [110, 60, 85, 65, 100]
  let combined_weights := original_weights ++ new_weights
  average_weight combined_weights = 79.33 := 
by 
  sorry

end bowling_team_avg_weight_l62_62018


namespace correct_proposition_four_l62_62222

universe u

-- Definitions
variable {Point : Type u} (A B : Point) (a α : Set Point)
variable (h5 : A ∉ α)
variable (h6 : a ⊂ α)

-- The statement to be proved
theorem correct_proposition_four : A ∉ a :=
sorry

end correct_proposition_four_l62_62222


namespace deductive_reasoning_correct_l62_62427

theorem deductive_reasoning_correct :
  (∀ (s : ℕ), s = 3 ↔
    (s == 1 → DeductiveReasoningGeneralToSpecific ∧
     s == 2 → alwaysCorrect ∧
     s == 3 → InFormOfSyllogism ∧
     s == 4 → ConclusionDependsOnPremisesAndForm)) :=
sorry

end deductive_reasoning_correct_l62_62427


namespace farmer_apples_l62_62990

theorem farmer_apples : 127 - 39 = 88 := by
  -- Skipping proof details
  sorry

end farmer_apples_l62_62990


namespace smallest_M_conditions_l62_62227

theorem smallest_M_conditions :
  ∃ M : ℕ, M > 0 ∧
  ((∃ k₁, M = 8 * k₁) ∨ (∃ k₂, M + 2 = 8 * k₂) ∨ (∃ k₃, M + 4 = 8 * k₃)) ∧
  ((∃ k₄, M = 9 * k₄) ∨ (∃ k₅, M + 2 = 9 * k₅) ∨ (∃ k₆, M + 4 = 9 * k₆)) ∧
  ((∃ k₇, M = 25 * k₇) ∨ (∃ k₈, M + 2 = 25 * k₈) ∨ (∃ k₉, M + 4 = 25 * k₉)) ∧
  M = 100 :=
sorry

end smallest_M_conditions_l62_62227


namespace regular_discount_rate_l62_62494

theorem regular_discount_rate (MSRP : ℝ) (s : ℝ) (sale_price : ℝ) (d : ℝ) :
  MSRP = 35 ∧ s = 0.20 ∧ sale_price = 19.6 → d = 0.3 :=
by
  intro h
  sorry

end regular_discount_rate_l62_62494


namespace find_angle_EHG_l62_62962

noncomputable def angle_EHG (angle_EFG : ℝ) (angle_GHE : ℝ) : ℝ := angle_GHE - angle_EFG
 
theorem find_angle_EHG : 
  ∀ (EF GH : Prop) (angle_EFG angle_GHE : ℝ), (EF ∧ GH) → 
    EF ∧ GH ∧ angle_EFG = 50 ∧ angle_GHE = 80 → angle_EHG angle_EFG angle_GHE = 30 := 
by 
  intros EF GH angle_EFG angle_GHE h1 h2
  sorry

end find_angle_EHG_l62_62962


namespace find_a_b_sum_specific_find_a_b_sum_l62_62627

-- Define the sets A and B based on the given inequalities
def set_A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def set_B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Intersect the sets A and B
def set_A_int_B : Set ℝ := set_A ∩ set_B

-- Define the inequality with parameters a and b
def quad_ineq (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

-- Define the parameters a and b based on the given condition
noncomputable def a : ℝ := -1
noncomputable def b : ℝ := -1

-- The statement to be proved
theorem find_a_b_sum : ∀ a b : ℝ, set_A ∩ set_B = {x | a * x^2 + b * x + 2 > 0} → a + b = -2 :=
by
  sorry

-- Fixing the parameters a and b for our specific proof condition
theorem specific_find_a_b_sum : a + b = -2 :=
by
  sorry

end find_a_b_sum_specific_find_a_b_sum_l62_62627


namespace carrots_planted_per_hour_l62_62533

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end carrots_planted_per_hour_l62_62533


namespace coords_of_point_P_l62_62444

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

theorem coords_of_point_P :
  ∀ (a : ℝ), 0 < a ∧ a ≠ 1 → ∃ P : ℝ × ℝ, (P = (1, -2) ∧ ∀ y, f (f a (-2)) y = y) :=
by
  sorry

end coords_of_point_P_l62_62444


namespace sequence_100th_term_eq_l62_62644

-- Definitions for conditions
def numerator (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominator (n : ℕ) : ℕ := 2 + (n - 1) * 3

-- The statement of the problem as a Lean 4 theorem
theorem sequence_100th_term_eq :
  (numerator 100) / (denominator 100) = 199 / 299 :=
by
  sorry

end sequence_100th_term_eq_l62_62644


namespace hexadecagon_area_l62_62882

theorem hexadecagon_area (r : ℝ) : 
  let θ := (360 / 16 : ℝ)
  let A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180)
  let total_area := 16 * A_triangle
  3 * r^2 = total_area :=
by
  sorry

end hexadecagon_area_l62_62882


namespace arithmetic_sequence_fifth_term_l62_62790

theorem arithmetic_sequence_fifth_term (x y : ℝ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x^2 + y^2
    let a2 := x^2 - y^2
    let a3 := x^2 * y^2
    let a4 := x^2 / y^2
    let d := a2 - a1
    let a5 := a4 + d
    a5 = 2 := by
  sorry

end arithmetic_sequence_fifth_term_l62_62790


namespace find_certain_number_l62_62254

theorem find_certain_number
  (t b c : ℝ)
  (average1 : (t + b + c + 14 + 15) / 5 = 12)
  (average2 : (t + b + c + x) / 4 = 15)
  (x : ℝ) :
  x = 29 :=
by
  sorry

end find_certain_number_l62_62254


namespace expression_equals_5_l62_62551

def expression_value : ℤ := 8 + 15 / 3 - 2^3

theorem expression_equals_5 : expression_value = 5 :=
by
  sorry

end expression_equals_5_l62_62551


namespace root_expression_of_cubic_l62_62815

theorem root_expression_of_cubic :
  ∀ a b c : ℝ, (a^3 - 2*a - 2 = 0) ∧ (b^3 - 2*b - 2 = 0) ∧ (c^3 - 2*c - 2 = 0)
    → a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -6 := 
by 
  sorry

end root_expression_of_cubic_l62_62815


namespace girls_select_same_colored_marble_l62_62384

def probability_same_color (total_white total_black girls boys : ℕ) : ℚ :=
  let prob_white := (total_white * (total_white - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  let prob_black := (total_black * (total_black - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  prob_white + prob_black

theorem girls_select_same_colored_marble :
  probability_same_color 2 2 2 2 = 1 / 3 :=
by
  sorry

end girls_select_same_colored_marble_l62_62384


namespace maxRegions100Parabolas_l62_62006

-- Define the number of parabolas of each type
def numberOfParabolas1 := 50
def numberOfParabolas2 := 50

-- Define the function that counts the number of regions formed by n parabolas intersecting at most m times
def maxRegions (n m : Nat) : Nat :=
  (List.range (m+1)).foldl (λ acc k => acc + Nat.choose n k) 0

-- Specify the intersection properties for each type of parabolas
def intersectionsParabolas1 := 2
def intersectionsParabolas2 := 2
def intersectionsBetweenSets := 4

-- Calculate the number of regions formed by each set of 50 parabolas
def regionsSet1 := maxRegions numberOfParabolas1 intersectionsParabolas1
def regionsSet2 := maxRegions numberOfParabolas2 intersectionsParabolas2

-- Calculate the additional regions created by intersections between the sets
def additionalIntersections := numberOfParabolas1 * numberOfParabolas2 * intersectionsBetweenSets

-- Combine the regions
def totalRegions := regionsSet1 + regionsSet2 + additionalIntersections + 1

-- Prove the final result
theorem maxRegions100Parabolas : totalRegions = 15053 :=
  sorry

end maxRegions100Parabolas_l62_62006


namespace simplify_cube_root_21952000_l62_62933

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem simplify_cube_root_21952000 : 
  cube_root 21952000 = 280 := 
by {
  sorry
}

end simplify_cube_root_21952000_l62_62933


namespace replace_digits_divisible_by_13_l62_62347

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem replace_digits_divisible_by_13 :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ 
  (3 * 10^6 + x * 10^4 + y * 10^2 + 3) % 13 = 0 ∧
  (x = 2 ∧ y = 3 ∨ 
   x = 5 ∧ y = 2 ∨ 
   x = 8 ∧ y = 1 ∨ 
   x = 9 ∧ y = 5 ∨ 
   x = 6 ∧ y = 6 ∨ 
   x = 3 ∧ y = 7 ∨ 
   x = 0 ∧ y = 8) :=
by
  sorry

end replace_digits_divisible_by_13_l62_62347


namespace four_identical_pairwise_differences_l62_62760

theorem four_identical_pairwise_differences (a : Fin 20 → ℕ) (h_distinct : Function.Injective a) (h_lt_70 : ∀ i, a i < 70) :
  ∃ d, ∃ (f g : Fin 20 × Fin 20), f ≠ g ∧ (a f.1 - a f.2 = d) ∧ (a g.1 - a g.2 = d) ∧
  ∃ (f1 f2 : Fin 20 × Fin 20), (f1 ≠ f ∧ f1 ≠ g) ∧ (f2 ≠ f ∧ f2 ≠ g) ∧ (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  ∃ (f3 : Fin 20 × Fin 20), (f3 ≠ f ∧ f3 ≠ g ∧ f3 ≠ f1 ∧ f3 ≠ f2) ∧ (a f3.1 - a f3.2 = d) := 
sorry

end four_identical_pairwise_differences_l62_62760


namespace each_squirrel_needs_more_acorns_l62_62065

noncomputable def acorns_needed : ℕ := 300
noncomputable def total_acorns_collected : ℕ := 4500
noncomputable def number_of_squirrels : ℕ := 20

theorem each_squirrel_needs_more_acorns : 
  (acorns_needed - total_acorns_collected / number_of_squirrels) = 75 :=
by
  sorry

end each_squirrel_needs_more_acorns_l62_62065


namespace paul_lost_crayons_l62_62937

theorem paul_lost_crayons :
  let total := 229
  let given_away := 213
  let lost := total - given_away
  lost = 16 :=
by
  sorry

end paul_lost_crayons_l62_62937


namespace find_certain_number_l62_62690

theorem find_certain_number (x : ℕ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  x = 58 := by
  sorry

end find_certain_number_l62_62690


namespace not_linear_eq_l62_62371

-- Representing the given equations
def eq1 (x : ℝ) : Prop := 5 * x + 3 = 3 * x - 7
def eq2 (x : ℝ) : Prop := 1 + 2 * x = 3
def eq4 (x : ℝ) : Prop := x - 7 = 0

-- The equation to verify if it's not linear
def eq3 (x : ℝ) : Prop := abs (2 * x) / 3 + 5 / x = 3

-- Stating the Lean statement to be proved
theorem not_linear_eq : ¬ (eq3 x) := by
  sorry

end not_linear_eq_l62_62371


namespace scientific_notation_123000_l62_62176

theorem scientific_notation_123000 : (123000 : ℝ) = 1.23 * 10^5 := by
  sorry

end scientific_notation_123000_l62_62176


namespace part1_part2_l62_62519

noncomputable section

variables (a x : ℝ)

def P : Prop := x^2 - 4*a*x + 3*a^2 < 0
def Q : Prop := abs (x - 3) ≤ 1

-- Part 1: If a=1 and P ∨ Q, prove the range of x is 1 < x ≤ 4
theorem part1 (h1 : a = 1) (h2 : P a x ∨ Q x) : 1 < x ∧ x ≤ 4 :=
sorry

-- Part 2: If ¬P is necessary but not sufficient for ¬Q, prove the range of a is 4/3 ≤ a ≤ 2
theorem part2 (h : (¬P a x → ¬Q x) ∧ (¬Q x → ¬P a x → False)) : 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end part1_part2_l62_62519


namespace cards_probability_comparison_l62_62230

noncomputable def probability_case_a : ℚ :=
  (Nat.choose 13 10) * (Nat.choose 39 3) / Nat.choose 52 13

noncomputable def probability_case_b : ℚ :=
  4 ^ 13 / Nat.choose 52 13

theorem cards_probability_comparison :
  probability_case_b > probability_case_a :=
  sorry

end cards_probability_comparison_l62_62230


namespace length_of_bridge_l62_62613

theorem length_of_bridge
  (T : ℕ) (t : ℕ) (s : ℕ)
  (hT : T = 250)
  (ht : t = 20)
  (hs : s = 20) :
  ∃ L : ℕ, L = 150 :=
by
  sorry

end length_of_bridge_l62_62613


namespace haley_marbles_l62_62775

theorem haley_marbles (m : ℕ) (k : ℕ) (h1 : k = 2) (h2 : m = 28) : m / k = 14 :=
by sorry

end haley_marbles_l62_62775


namespace base_case_proof_l62_62839

noncomputable def base_case_inequality := 1 + (1 / (2 ^ 3)) < 2 - (1 / 2)

theorem base_case_proof : base_case_inequality := by
  -- The proof would go here
  sorry

end base_case_proof_l62_62839


namespace am_gm_inequality_l62_62355

theorem am_gm_inequality (x y z : ℝ) (n : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0):
  x^n + y^n + z^n ≥ 1 / 3^(n-1) :=
by
  sorry

end am_gm_inequality_l62_62355


namespace percentage_of_circle_outside_triangle_l62_62248

theorem percentage_of_circle_outside_triangle (A : ℝ)
  (h₁ : 0 < A) -- Total area A is positive
  (A_inter : ℝ) (A_outside_tri : ℝ) (A_total_circle : ℝ)
  (h₂ : A_inter = 0.45 * A)
  (h₃ : A_outside_tri = 0.40 * A)
  (h₄ : A_total_circle = 0.60 * A) :
  100 * (1 - A_inter / A_total_circle) = 25 :=
by
  sorry

end percentage_of_circle_outside_triangle_l62_62248


namespace two_roots_range_a_l62_62251

noncomputable def piecewise_func (x : ℝ) : ℝ :=
if x ≤ 1 then (1/3) * x + 1 else Real.log x

theorem two_roots_range_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ piecewise_func x1 = a * x1 ∧ piecewise_func x2 = a * x2) ↔ (1/3 < a ∧ a < 1/Real.exp 1) :=
sorry

end two_roots_range_a_l62_62251


namespace cube_volume_l62_62137

theorem cube_volume (SA : ℕ) (h : SA = 294) : 
  ∃ V : ℕ, V = 343 := 
by
  sorry

end cube_volume_l62_62137


namespace swimming_pool_min_cost_l62_62144

theorem swimming_pool_min_cost (a : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x : ℝ), x > 0 → y = 2400 * a + 6 * (x + 1600 / x) * a) →
  (∃ (x : ℝ), x > 0 ∧ y = 2880 * a) :=
by
  sorry

end swimming_pool_min_cost_l62_62144


namespace max_amount_paul_received_l62_62441

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end max_amount_paul_received_l62_62441


namespace value_of_x_squared_plus_y_squared_l62_62186

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l62_62186


namespace cos2_add_2sin2_eq_64_over_25_l62_62866

theorem cos2_add_2sin2_eq_64_over_25 (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
sorry

end cos2_add_2sin2_eq_64_over_25_l62_62866


namespace hotel_flat_fee_l62_62336

theorem hotel_flat_fee
  (f n : ℝ)
  (h1 : f + 3 * n = 195)
  (h2 : f + 7 * n = 380) :
  f = 56.25 :=
by sorry

end hotel_flat_fee_l62_62336


namespace area_inside_quadrilateral_BCDE_outside_circle_l62_62364

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3) / 2 * side_length ^ 2

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

theorem area_inside_quadrilateral_BCDE_outside_circle :
  let side_length := 2
  let hex_area := hexagon_area side_length
  let hex_area_large := hexagon_area (2 * side_length)
  let circle_radius := 3
  let circle_area_A := circle_area circle_radius
  let total_area_of_interest := hex_area_large - circle_area_A
  let area_of_one_region := total_area_of_interest / 6
  area_of_one_region = 4 * Real.sqrt 3 - (3 / 2) * Real.pi :=
by
  sorry

end area_inside_quadrilateral_BCDE_outside_circle_l62_62364


namespace time_saved_by_both_trains_trainB_distance_l62_62177

-- Define the conditions
def trainA_speed_reduced := 360 / 12  -- 30 miles/hour
def trainB_speed_reduced := 360 / 8   -- 45 miles/hour

def trainA_speed := trainA_speed_reduced / (2 / 3)  -- 45 miles/hour
def trainB_speed := trainB_speed_reduced / (1 / 2)  -- 90 miles/hour

def trainA_time_saved := 12 - (360 / trainA_speed)  -- 4 hours
def trainB_time_saved := 8 - (360 / trainB_speed)   -- 4 hours

-- Prove that total time saved by both trains running at their own speeds is 8 hours
theorem time_saved_by_both_trains : trainA_time_saved + trainB_time_saved = 8 := by
  sorry

-- Prove that the distance between Town X and Town Y for Train B is 360 miles
theorem trainB_distance : 360 = 360 := by
  rfl

end time_saved_by_both_trains_trainB_distance_l62_62177


namespace find_f_inv_128_l62_62431

open Function

theorem find_f_inv_128 (f : ℕ → ℕ) 
  (h₀ : f 5 = 2) 
  (h₁ : ∀ x, f (2 * x) = 2 * f x) : 
  f⁻¹' {128} = {320} :=
by
  sorry

end find_f_inv_128_l62_62431


namespace log_ratio_squared_eq_nine_l62_62069

-- Given conditions
variable (x y : ℝ) 
variable (hx_pos : x > 0) 
variable (hy_pos : y > 0)
variable (hx_neq1 : x ≠ 1) 
variable (hy_neq1 : y ≠ 1)
variable (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
variable (heq : x * y = 243)

-- Prove that (\log_3(\tfrac x y))^2 = 9
theorem log_ratio_squared_eq_nine (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq1 : x ≠ 1) (hy_neq1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (heq : x * y = 243) : 
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 :=
sorry

end log_ratio_squared_eq_nine_l62_62069


namespace parabola_vertex_y_coord_l62_62059

theorem parabola_vertex_y_coord (a b c x y : ℝ) (h : a = 2 ∧ b = 16 ∧ c = 35 ∧ y = a*x^2 + b*x + c ∧ x = -b / (2 * a)) : y = 3 :=
by
  sorry

end parabola_vertex_y_coord_l62_62059


namespace train_speed_proof_l62_62412

noncomputable def speedOfTrain (lengthOfTrain : ℝ) (timeToCross : ℝ) (speedOfMan : ℝ) : ℝ :=
  let man_speed_m_per_s := speedOfMan * 1000 / 3600
  let relative_speed := lengthOfTrain / timeToCross
  let train_speed_m_per_s := relative_speed + man_speed_m_per_s
  train_speed_m_per_s * 3600 / 1000

theorem train_speed_proof :
  speedOfTrain 100 5.999520038396929 3 = 63 := by
  sorry

end train_speed_proof_l62_62412


namespace birgit_numbers_sum_l62_62410

theorem birgit_numbers_sum (a b c d : ℕ) 
  (h1 : a + b + c = 415) 
  (h2 : a + b + d = 442) 
  (h3 : a + c + d = 396) 
  (h4 : b + c + d = 325) : 
  a + b + c + d = 526 :=
by
  sorry

end birgit_numbers_sum_l62_62410


namespace metal_sheets_per_panel_l62_62900

-- Define the given conditions
def num_panels : ℕ := 10
def rods_per_sheet : ℕ := 10
def rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def total_rods_needed : ℕ := 380

-- Question translated to Lean statement
theorem metal_sheets_per_panel (S : ℕ) (h : 10 * (10 * S + 8) = 380) : S = 3 := 
  sorry

end metal_sheets_per_panel_l62_62900


namespace contrapositive_example_l62_62562

theorem contrapositive_example (a b : ℕ) (h : a = 0 → ab = 0) : ab ≠ 0 → a ≠ 0 :=
by sorry

end contrapositive_example_l62_62562


namespace find_y_l62_62509

def vectors_orthogonal_condition (y : ℝ) : Prop :=
  (1 * -2) + (-3 * y) + (-4 * -1) = 0

theorem find_y : vectors_orthogonal_condition (2 / 3) :=
by
  sorry

end find_y_l62_62509


namespace range_of_f_l62_62630

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem range_of_f : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ f x ∧ f x ≤ 2 :=
by
  intro x Hx
  sorry

end range_of_f_l62_62630


namespace range_of_a_l62_62245

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a ∈ [-1, 3]) := 
by
  sorry

end range_of_a_l62_62245


namespace molecular_weight_chlorous_acid_l62_62768

def weight_H : ℝ := 1.01
def weight_Cl : ℝ := 35.45
def weight_O : ℝ := 16.00

def molecular_weight_HClO2 := (1 * weight_H) + (1 * weight_Cl) + (2 * weight_O)

theorem molecular_weight_chlorous_acid : molecular_weight_HClO2 = 68.46 := 
  by
    sorry

end molecular_weight_chlorous_acid_l62_62768


namespace mersenne_prime_condition_l62_62804

theorem mersenne_prime_condition (a n : ℕ) (h_a : 1 < a) (h_n : 1 < n) (h_prime : Prime (a ^ n - 1)) : a = 2 ∧ Prime n :=
by
  sorry

end mersenne_prime_condition_l62_62804


namespace probability_of_first_good_product_on_third_try_l62_62224

-- Define the problem parameters
def pass_rate : ℚ := 3 / 4
def failure_rate : ℚ := 1 / 4
def epsilon := 3

-- The target probability statement
theorem probability_of_first_good_product_on_third_try :
  (failure_rate * failure_rate * pass_rate) = ((1 / 4) ^ 2 * (3 / 4)) :=
by
  sorry

end probability_of_first_good_product_on_third_try_l62_62224


namespace consumption_increase_percentage_l62_62475

theorem consumption_increase_percentage (T C : ℝ) (T_pos : 0 < T) (C_pos : 0 < C) :
  (0.7 * (1 + x / 100) * T * C = 0.84 * T * C) → x = 20 :=
by sorry

end consumption_increase_percentage_l62_62475


namespace product_cos_angles_l62_62712

theorem product_cos_angles :
  (Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = 1 / 128) :=
sorry

end product_cos_angles_l62_62712


namespace smallest_divisor_after_391_l62_62448

theorem smallest_divisor_after_391 (m : ℕ) (h₁ : 1000 ≤ m ∧ m < 10000) (h₂ : Even m) (h₃ : 391 ∣ m) : 
  ∃ d, d > 391 ∧ d ∣ m ∧ ∀ e, 391 < e ∧ e ∣ m → e ≥ d :=
by
  use 441
  sorry

end smallest_divisor_after_391_l62_62448


namespace marikas_father_age_twice_in_2036_l62_62748

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l62_62748


namespace non_integer_interior_angle_count_l62_62899

theorem non_integer_interior_angle_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n < 10 ∧ ¬(∃ k : ℕ, 180 * (n - 2) = n * k) :=
by sorry

end non_integer_interior_angle_count_l62_62899


namespace inequality_must_hold_l62_62320

theorem inequality_must_hold (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := 
sorry

end inequality_must_hold_l62_62320


namespace sum_division_l62_62402

theorem sum_division (x y z : ℝ) (total_share_y : ℝ) 
  (Hx : x = 1) 
  (Hy : y = 0.45) 
  (Hz : z = 0.30) 
  (share_y : total_share_y = 36) 
  : (x + y + z) * (total_share_y / y) = 140 := by
  sorry

end sum_division_l62_62402


namespace solve_inequality_l62_62302

noncomputable def P (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem solve_inequality (x : ℝ) : (P x > 0) ↔ (x < 1 ∨ x > 2) := 
  sorry

end solve_inequality_l62_62302


namespace problem_b_lt_a_lt_c_l62_62718

theorem problem_b_lt_a_lt_c (a b c : ℝ)
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : b < a ∧ a < c := by
  sorry

end problem_b_lt_a_lt_c_l62_62718


namespace parabola_symmetry_l62_62236

theorem parabola_symmetry (a h m : ℝ) (A_on_parabola : 4 = a * (-1 - 3)^2 + h) (B_on_parabola : 4 = a * (m - 3)^2 + h) : 
  m = 7 :=
by 
  sorry

end parabola_symmetry_l62_62236


namespace total_daisies_sold_l62_62709

-- Conditions Definitions
def first_day_sales : ℕ := 45
def second_day_sales : ℕ := first_day_sales + 20
def third_day_sales : ℕ := 2 * second_day_sales - 10
def fourth_day_sales : ℕ := 120

-- Question: Prove that the total sales over the four days is 350.
theorem total_daisies_sold :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = 350 := by
  sorry

end total_daisies_sold_l62_62709


namespace intersection_of_sets_l62_62521

def set_a : Set ℝ := { x | -x^2 + 2 * x ≥ 0 }
def set_b : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_sets : (set_a ∩ set_b) = set_intersection := by 
  sorry

end intersection_of_sets_l62_62521


namespace fraction_of_innocent_cases_l62_62170

-- Definitions based on the given conditions
def total_cases : ℕ := 17
def dismissed_cases : ℕ := 2
def delayed_cases : ℕ := 1
def guilty_cases : ℕ := 4

-- The remaining cases after dismissals
def remaining_cases : ℕ := total_cases - dismissed_cases

-- The remaining cases that are not innocent
def non_innocent_cases : ℕ := delayed_cases + guilty_cases

-- The innocent cases
def innocent_cases : ℕ := remaining_cases - non_innocent_cases

-- The fraction of the remaining cases that were ruled innocent
def fraction_innocent : Rat := innocent_cases / remaining_cases

-- The theorem we want to prove
theorem fraction_of_innocent_cases :
  fraction_innocent = 2 / 3 := by
  sorry

end fraction_of_innocent_cases_l62_62170


namespace halfway_between_one_eighth_and_one_third_is_correct_l62_62856

-- Define the fractions
def one_eighth : ℚ := 1 / 8
def one_third : ℚ := 1 / 3

-- Define the correct answer
def correct_answer : ℚ := 11 / 48

-- State the theorem to prove the halfway number is correct_answer
theorem halfway_between_one_eighth_and_one_third_is_correct : 
  (one_eighth + one_third) / 2 = correct_answer :=
sorry

end halfway_between_one_eighth_and_one_third_is_correct_l62_62856


namespace number_of_bricks_l62_62465

theorem number_of_bricks (b1_hours b2_hours combined_hours: ℝ) (reduction_rate: ℝ) (x: ℝ):
  b1_hours = 12 ∧ 
  b2_hours = 15 ∧ 
  combined_hours = 6 ∧ 
  reduction_rate = 15 ∧ 
  (combined_hours * ((x / b1_hours) + (x / b2_hours) - reduction_rate) = x) → 
  x = 1800 :=
by
  sorry

end number_of_bricks_l62_62465


namespace bus_speed_including_stoppages_l62_62864

theorem bus_speed_including_stoppages
  (speed_excluding_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ) :
  speed_excluding_stoppages = 64 ∧ stoppage_time_per_hour = 15 / 60 →
  (44 / 60) * speed_excluding_stoppages = 48 :=
by
  sorry

end bus_speed_including_stoppages_l62_62864


namespace solve_fractional_equation_l62_62742

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1) ↔ x = -5 ∨ x = 3 :=
sorry

end solve_fractional_equation_l62_62742


namespace exclude_13_code_count_l62_62080

/-- The number of 5-digit codes (00000 to 99999) that don't contain the sequence "13". -/
theorem exclude_13_code_count :
  let total_codes := 100000
  let excluded_codes := 3970
  total_codes - excluded_codes = 96030 :=
by
  let total_codes := 100000
  let excluded_codes := 3970
  have h : total_codes - excluded_codes = 96030 := by
    -- Provide mathematical proof or use sorry for placeholder
    sorry
  exact h

end exclude_13_code_count_l62_62080


namespace min_area_is_fifteen_l62_62178

variable (L W : ℕ)

def minimum_possible_area (L W : ℕ) : ℕ :=
  if L = 3 ∧ W = 5 then 3 * 5 else 0

theorem min_area_is_fifteen (hL : 3 ≤ L ∧ L ≤ 5) (hW : 5 ≤ W ∧ W ≤ 7) : 
  minimum_possible_area 3 5 = 15 := 
by
  sorry

end min_area_is_fifteen_l62_62178


namespace motorboat_max_distance_l62_62114

/-- Given a motorboat which, when fully fueled, can travel exactly 40 km against the current 
    or 60 km with the current, proves that the maximum distance it can travel up the river and 
    return to the starting point with the available fuel is 24 km. -/
theorem motorboat_max_distance (upstream_dist : ℕ) (downstream_dist : ℕ) : 
  upstream_dist = 40 → downstream_dist = 60 → 
  ∃ max_round_trip_dist : ℕ, max_round_trip_dist = 24 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end motorboat_max_distance_l62_62114


namespace abs_value_solution_l62_62727

theorem abs_value_solution (a : ℝ) : |-a| = |-5.333| → (a = 5.333 ∨ a = -5.333) :=
by
  sorry

end abs_value_solution_l62_62727


namespace effect_on_revenue_l62_62323

-- Define the conditions using parameters and variables

variables {P Q : ℝ} -- Original price and quantity of TV sets

def new_price (P : ℝ) : ℝ := P * 1.60 -- New price after 60% increase
def new_quantity (Q : ℝ) : ℝ := Q * 0.80 -- New quantity after 20% decrease

def original_revenue (P Q : ℝ) : ℝ := P * Q -- Original revenue
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q) -- New revenue

theorem effect_on_revenue
  (P Q : ℝ) :
  new_revenue P Q = original_revenue P Q * 1.28 :=
by
  sorry

end effect_on_revenue_l62_62323


namespace power_addition_proof_l62_62400

theorem power_addition_proof :
  (-2) ^ 48 + 3 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2) = 2 ^ 48 + 3 ^ 40 := 
by
  sorry

end power_addition_proof_l62_62400


namespace inequality_abc_l62_62192

theorem inequality_abc (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b * c = 8) :
  (a^2 / Real.sqrt ((1 + a^3) * (1 + b^3))) + (b^2 / Real.sqrt ((1 + b^3) * (1 + c^3))) +
  (c^2 / Real.sqrt ((1 + c^3) * (1 + a^3))) ≥ 4 / 3 :=
sorry

end inequality_abc_l62_62192


namespace problem_N_lowest_terms_l62_62004

theorem problem_N_lowest_terms :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 2500 ∧ ∃ k : ℕ, k ∣ 128 ∧ (n + 11) % k = 0 ∧ (Nat.gcd (n^2 + 7) (n + 11)) > 1) →
  ∃ cnt : ℕ, cnt = 168 :=
by
  sorry

end problem_N_lowest_terms_l62_62004


namespace problem_statement_l62_62333

noncomputable def U : Set Int := {-2, -1, 0, 1, 2}
noncomputable def A : Set Int := {x : Int | -2 ≤ x ∧ x < 0}
noncomputable def B : Set Int := {x : Int | (x = 0 ∨ x = 1)} -- since natural numbers typically include positive integers, adapting B contextually

theorem problem_statement : ((U \ A) ∩ B) = {0, 1} := by
  sorry

end problem_statement_l62_62333


namespace mason_father_age_l62_62506

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end mason_father_age_l62_62506


namespace boy_present_age_l62_62523

-- Define the boy's present age
variable (x : ℤ)

-- Conditions from the problem statement
def condition_one : Prop :=
  x + 4 = 2 * (x - 6)

-- Prove that the boy's present age is 16
theorem boy_present_age (h : condition_one x) : x = 16 := 
sorry

end boy_present_age_l62_62523


namespace range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l62_62116

-- Define the propositions p and q in terms of x and m
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- The range of x for p ∧ q when m = 1
theorem range_x_when_p_and_q_m_eq_1 : {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by sorry

-- The range of m where ¬p is a necessary but not sufficient condition for q
theorem range_m_for_not_p_necessary_not_sufficient_q : {m : ℝ | ∀ x, ¬p x m → q x} ∩ {m : ℝ | ∃ x, ¬p x m ∧ q x} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} :=
by sorry

end range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l62_62116


namespace can_vasya_obtain_400_mercedes_l62_62169

-- Define the types for the cars
inductive Car : Type
| Zh : Car
| V : Car
| M : Car

-- Define the initial conditions as exchange constraints
def exchange1 (Zh V M : ℕ) : Prop :=
  3 * Zh = V + M

def exchange2 (V Zh M : ℕ) : Prop :=
  3 * V = 2 * Zh + M

-- Define the initial number of Zhiguli cars Vasya has.
def initial_Zh : ℕ := 700

-- Define the target number of Mercedes cars Vasya wants.
def target_M : ℕ := 400

-- The proof goal: Vasya cannot exchange to get exactly 400 Mercedes cars.
theorem can_vasya_obtain_400_mercedes (Zh V M : ℕ) (h1 : exchange1 Zh V M) (h2 : exchange2 V Zh M) :
  initial_Zh = 700 → target_M = 400 → (Zh ≠ 0 ∨ V ≠ 0 ∨ M ≠ 400) := sorry

end can_vasya_obtain_400_mercedes_l62_62169


namespace real_solutions_of_equation_l62_62209

theorem real_solutions_of_equation : 
  ∃! x₁ x₂ : ℝ, (3 * x₁^2 - 10 * x₁ + 7 = 0) ∧ (3 * x₂^2 - 10 * x₂ + 7 = 0) ∧ x₁ ≠ x₂ :=
sorry

end real_solutions_of_equation_l62_62209


namespace melons_count_l62_62380

theorem melons_count (w_apples_total w_apple w_2apples w_watermelons w_total w_melons : ℕ) :
  w_apples_total = 4500 →
  9 * w_apple = w_apples_total →
  2 * w_apple = w_2apples →
  5 * 1050 = w_watermelons →
  w_total = w_2apples + w_melons →
  w_total = w_watermelons →
  w_melons / 850 = 5 :=
by
  sorry

end melons_count_l62_62380


namespace largest_integral_k_for_real_distinct_roots_l62_62555

theorem largest_integral_k_for_real_distinct_roots :
  ∃ k : ℤ, (k < 9) ∧ (∀ k' : ℤ, k' < 9 → k' ≤ k) :=
sorry

end largest_integral_k_for_real_distinct_roots_l62_62555


namespace max_marks_l62_62670

theorem max_marks (total_marks : ℕ) (obtained_marks : ℕ) (failed_by : ℕ) 
    (passing_percentage : ℝ) (passing_marks : ℝ) (H1 : obtained_marks = 125)
    (H2 : failed_by = 40) (H3 : passing_percentage = 0.33) 
    (H4 : passing_marks = obtained_marks + failed_by) 
    (H5 : passing_marks = passing_percentage * total_marks) : total_marks = 500 := by
  sorry

end max_marks_l62_62670


namespace gun_fan_image_equivalence_l62_62556

def gunPiercingImage : String := "point moving to form a line"
def foldingFanImage : String := "line moving to form a surface"

theorem gun_fan_image_equivalence :
  (gunPiercingImage = "point moving to form a line") ∧ 
  (foldingFanImage = "line moving to form a surface") := by
  -- Proof goes here
  sorry

end gun_fan_image_equivalence_l62_62556


namespace problem_statement_l62_62581

theorem problem_statement (a b c m : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0) (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 :=
sorry

end problem_statement_l62_62581


namespace johns_number_is_1500_l62_62689

def is_multiple_of (a b : Nat) : Prop := ∃ k, a = k * b

theorem johns_number_is_1500 (n : ℕ) (h1 : is_multiple_of n 125) (h2 : is_multiple_of n 30) (h3 : 1000 ≤ n ∧ n ≤ 3000) : n = 1500 :=
by
  -- proof structure goes here
  sorry

end johns_number_is_1500_l62_62689


namespace regular_polygon_of_45_deg_l62_62308

def is_regular_polygon (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 2 ∧ 360 % k = 0 ∧ n = 360 / k

def regular_polygon_is_octagon (angle : ℕ) : Prop :=
  is_regular_polygon 8 ∧ angle = 45

theorem regular_polygon_of_45_deg : regular_polygon_is_octagon 45 :=
  sorry

end regular_polygon_of_45_deg_l62_62308


namespace bicycle_route_total_length_l62_62316

theorem bicycle_route_total_length :
  let horizontal_length := 13 
  let vertical_length := 13 
  2 * horizontal_length + 2 * vertical_length = 52 :=
by
  let horizontal_length := 13
  let vertical_length := 13
  sorry

end bicycle_route_total_length_l62_62316


namespace train_travel_section_marked_l62_62989

-- Definition of the metro structure with the necessary conditions.
structure Metro (Station : Type) :=
  (lines : List (Station × Station))
  (travel_time : Station → Station → ℕ)
  (terminal_turnaround : Station → Station)
  (transfer_station : Station → Station)

variable {Station : Type}

/-- The function that defines the bipolar coloring of the metro stations. -/
def station_color (s : Station) : ℕ := sorry  -- Placeholder for actual coloring function.

theorem train_travel_section_marked 
  (metro : Metro Station)
  (initial_station : Station)
  (end_station : Station)
  (travel_time : ℕ)
  (marked_section : Station × Station)
  (h_start : initial_station = marked_section.fst)
  (h_end : end_station = marked_section.snd)
  (h_travel_time : travel_time = 2016)
  (h_condition : ∀ s1 s2, (s1, s2) ∈ metro.lines → metro.travel_time s1 s2 = 1 ∧ 
                metro.terminal_turnaround s1 ≠ s1 ∧ metro.transfer_station s1 ≠ s2) :
  ∃ (time : ℕ), time = 2016 ∧ ∃ s1 s2, (s1, s2) = marked_section :=
sorry

end train_travel_section_marked_l62_62989


namespace eggs_per_box_l62_62306

-- Conditions
def num_eggs : ℝ := 3.0
def num_boxes : ℝ := 2.0

-- Theorem statement
theorem eggs_per_box (h1 : num_eggs = 3.0) (h2 : num_boxes = 2.0) : (num_eggs / num_boxes = 1.5) :=
sorry

end eggs_per_box_l62_62306


namespace range_of_a_l62_62531

theorem range_of_a (a : ℝ) : (-1/3 ≤ a) ∧ (a ≤ 2/3) ↔ (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → y = a * x + 1/3) :=
by
  sorry

end range_of_a_l62_62531


namespace simple_fraction_pow_l62_62965

theorem simple_fraction_pow : (66666^4 / 22222^4) = 81 := by
  sorry

end simple_fraction_pow_l62_62965


namespace initial_apples_l62_62664

-- Definitions of the conditions
def Minseok_ate : Nat := 3
def Jaeyoon_ate : Nat := 3
def apples_left : Nat := 2

-- The proposition we need to prove
theorem initial_apples : Minseok_ate + Jaeyoon_ate + apples_left = 8 := by
  sorry

end initial_apples_l62_62664


namespace inequality_ge_one_l62_62576

open Nat

variable (p q : ℝ) (m n : ℕ)

def conditions := p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

theorem inequality_ge_one (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := 
by sorry

end inequality_ge_one_l62_62576


namespace range_of_a_l62_62307

noncomputable def f (x a : ℝ) := Real.log x + a / x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 0, x * (2 * Real.log a - Real.log x) ≤ a) : 
  0 < a ∧ a ≤ 1 / Real.exp 1 :=
by
  sorry

end range_of_a_l62_62307


namespace necessary_but_not_sufficient_l62_62442

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 0 = 0) →
  (∀ x : ℝ, f (-x) = -f x) →
  ¬∀ f' : ℝ → ℝ, (f' 0 = 0 → ∀ y : ℝ, f' (-y) = -f' y)
:= by
  sorry

end necessary_but_not_sufficient_l62_62442


namespace minimum_k_exists_l62_62844

theorem minimum_k_exists (k : ℕ) (h : k > 0) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ k = 6 :=
sorry

end minimum_k_exists_l62_62844


namespace remainder_5_7_9_6_3_5_mod_7_l62_62763

theorem remainder_5_7_9_6_3_5_mod_7 : (5^7 + 9^6 + 3^5) % 7 = 5 :=
by sorry

end remainder_5_7_9_6_3_5_mod_7_l62_62763


namespace gardenia_to_lilac_ratio_l62_62818

-- Defining sales of flowers
def lilacs_sold : Nat := 10
def roses_sold : Nat := 3 * lilacs_sold
def total_flowers_sold : Nat := 45
def gardenias_sold : Nat := total_flowers_sold - (roses_sold + lilacs_sold)

-- The ratio of gardenias to lilacs as a fraction
def ratio_gardenias_to_lilacs (gardenias lilacs : Nat) : Rat := gardenias / lilacs

-- Stating the theorem to prove
theorem gardenia_to_lilac_ratio :
  ratio_gardenias_to_lilacs gardenias_sold lilacs_sold = 1 / 2 :=
by
  sorry

end gardenia_to_lilac_ratio_l62_62818


namespace range_of_b_if_solution_set_contains_1_2_3_l62_62852

theorem range_of_b_if_solution_set_contains_1_2_3 
  (b : ℝ)
  (h : ∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) :
  5 < b ∧ b < 7 :=
sorry

end range_of_b_if_solution_set_contains_1_2_3_l62_62852


namespace num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l62_62648

open Nat

def num_arrangements_A_middle (n : ℕ) : ℕ :=
  if n = 4 then factorial 4 else 0

def num_arrangements_A_not_adj_B (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3) * (factorial 4 / factorial 2) else 0

def num_arrangements_A_B_not_ends (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3 / factorial 2) * factorial 3 else 0

theorem num_arrangements_thm1 : num_arrangements_A_middle 4 = 24 := 
  sorry

theorem num_arrangements_thm2 : num_arrangements_A_not_adj_B 5 = 72 := 
  sorry

theorem num_arrangements_thm3 : num_arrangements_A_B_not_ends 5 = 36 := 
  sorry

end num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l62_62648


namespace tailor_cut_skirt_l62_62715

theorem tailor_cut_skirt (cut_pants cut_skirt : ℝ) (h1 : cut_pants = 0.5) (h2 : cut_skirt = cut_pants + 0.25) : cut_skirt = 0.75 :=
by
  sorry

end tailor_cut_skirt_l62_62715


namespace grunters_win_all_five_l62_62578

theorem grunters_win_all_five (p : ℚ) (games : ℕ) (win_prob : ℚ) :
  games = 5 ∧ win_prob = 3 / 5 → 
  p = (win_prob) ^ games ∧ p = 243 / 3125 := 
by
  intros h
  cases h
  sorry

end grunters_win_all_five_l62_62578


namespace max_points_of_intersection_l62_62695

-- Define the lines and their properties
variable (L : Fin 150 → Prop)

-- Condition: L_5n are parallel to each other
def parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k

-- Condition: L_{5n-1} pass through a given point B
def passing_through_B (n : ℕ) :=
  ∃ k, n = 5 * k + 1

-- Condition: L_{5n-2} are parallel to another line not parallel to those in parallel_group
def other_parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k + 3

-- Total number of points of intersection of pairs of lines from the complete set
theorem max_points_of_intersection (L : Fin 150 → Prop)
  (h_distinct : ∀ i j : Fin 150, i ≠ j → L i ≠ L j)
  (h_parallel_group : ∀ i j : Fin 150, parallel_group i → parallel_group j → L i = L j)
  (h_through_B : ∀ i j : Fin 150, passing_through_B i → passing_through_B j → L i = L j)
  (h_other_parallel_group : ∀ i j : Fin 150, other_parallel_group i → other_parallel_group j → L i = L j)
  : ∃ P, P = 8071 := 
sorry

end max_points_of_intersection_l62_62695


namespace max_knights_between_knights_l62_62635

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l62_62635


namespace vegetarian_family_member_count_l62_62048

variable (total_family : ℕ) (vegetarian_only : ℕ) (non_vegetarian_only : ℕ)
variable (both_vegetarian_nonvegetarian : ℕ) (vegan_only : ℕ)
variable (pescatarian : ℕ) (specific_vegetarian : ℕ)

theorem vegetarian_family_member_count :
  total_family = 35 →
  vegetarian_only = 11 →
  non_vegetarian_only = 6 →
  both_vegetarian_nonvegetarian = 9 →
  vegan_only = 3 →
  pescatarian = 4 →
  specific_vegetarian = 2 →
  vegetarian_only + both_vegetarian_nonvegetarian + vegan_only + pescatarian + specific_vegetarian = 29 :=
by
  intros
  sorry

end vegetarian_family_member_count_l62_62048


namespace parabola_shift_right_l62_62452

theorem parabola_shift_right (x : ℝ) :
  let original_parabola := - (1 / 2) * x^2
  let shifted_parabola := - (1 / 2) * (x - 1)^2
  original_parabola = shifted_parabola :=
sorry

end parabola_shift_right_l62_62452


namespace angie_age_l62_62834

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l62_62834


namespace smallest_integer_solution_l62_62488

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end smallest_integer_solution_l62_62488


namespace minimum_value_of_expression_l62_62504

theorem minimum_value_of_expression (x : ℝ) (h : x > 2) : 
  ∃ y, (∀ z, z > 2 → (z^2 - 4 * z + 5) / (z - 2) ≥ y) ∧ 
       y = 2 :=
by
  sorry

end minimum_value_of_expression_l62_62504


namespace john_calories_eaten_l62_62039

def servings : ℕ := 3
def calories_per_serving : ℕ := 120
def fraction_eaten : ℚ := 1 / 2

theorem john_calories_eaten : 
  (servings * calories_per_serving : ℕ) * fraction_eaten = 180 :=
  sorry

end john_calories_eaten_l62_62039


namespace find_y_l62_62374

theorem find_y (x y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end find_y_l62_62374


namespace fewer_parking_spaces_on_fourth_level_l62_62447

theorem fewer_parking_spaces_on_fourth_level 
  (spaces_first_level : ℕ) (spaces_second_level : ℕ) (spaces_third_level : ℕ) (spaces_fourth_level : ℕ) 
  (total_spaces_garage : ℕ) (cars_parked : ℕ) 
  (h1 : spaces_first_level = 90)
  (h2 : spaces_second_level = spaces_first_level + 8)
  (h3 : spaces_third_level = spaces_second_level + 12)
  (h4 : total_spaces_garage = 299)
  (h5 : cars_parked = 100)
  (h6 : spaces_first_level + spaces_second_level + spaces_third_level + spaces_fourth_level = total_spaces_garage) :
  spaces_third_level - spaces_fourth_level = 109 := 
by
  sorry

end fewer_parking_spaces_on_fourth_level_l62_62447


namespace polynomial_remainder_l62_62315

theorem polynomial_remainder (x : ℂ) (hx : x^5 = 1) :
  (x^25 + x^20 + x^15 + x^10 + x^5 + 1) % (x^5 - 1) = 6 :=
by
  -- Proof will go here
  sorry

end polynomial_remainder_l62_62315


namespace children_attended_l62_62240

theorem children_attended (A C : ℕ) (h1 : C = 2 * A) (h2 : A + C = 42) : C = 28 :=
by
  sorry

end children_attended_l62_62240


namespace smallest_solution_correct_l62_62901

noncomputable def smallest_solution (x : ℝ) : ℝ :=
if (⌊ x^2 ⌋ - ⌊ x ⌋^2 = 17) then x else 0

theorem smallest_solution_correct :
  smallest_solution (7 * Real.sqrt 2) = 7 * Real.sqrt 2 :=
by sorry

end smallest_solution_correct_l62_62901


namespace root_expression_value_l62_62246

-- Define the root condition
def is_root (a : ℝ) : Prop := 2 * a^2 - 3 * a - 5 = 0

-- The main theorem statement
theorem root_expression_value {a : ℝ} (h : is_root a) : -4 * a^2 + 6 * a = -10 := by
  sorry

end root_expression_value_l62_62246


namespace square_of_integer_l62_62812

theorem square_of_integer (n : ℕ) (h : ∃ l : ℤ, l^2 = 1 + 12 * (n^2 : ℤ)) :
  ∃ m : ℤ, 2 + 2 * Int.sqrt (1 + 12 * (n^2 : ℤ)) = m^2 := by
  sorry

end square_of_integer_l62_62812


namespace bertha_descendants_no_children_l62_62298

-- Definitions based on the conditions of the problem.
def bertha_daughters : ℕ := 10
def total_descendants : ℕ := 40
def granddaughters : ℕ := total_descendants - bertha_daughters
def daughters_with_children : ℕ := 8
def children_per_daughter_with_children : ℕ := 4
def number_of_granddaughters : ℕ := daughters_with_children * children_per_daughter_with_children
def total_daughters_and_granddaughters : ℕ := bertha_daughters + number_of_granddaughters
def without_children : ℕ := total_daughters_and_granddaughters - daughters_with_children

-- Lean statement to prove the main question given the definitions.
theorem bertha_descendants_no_children : without_children = 34 := by
  -- Placeholder for the proof
  sorry

end bertha_descendants_no_children_l62_62298


namespace sequence_inequality_l62_62563

theorem sequence_inequality (a : ℕ → ℤ) (h₀ : a 1 > a 0) 
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n+1) = 3 * a n - 2 * a (n-1)) : 
  a 100 > 2^99 := 
sorry

end sequence_inequality_l62_62563


namespace exactly_one_solves_problem_l62_62942

theorem exactly_one_solves_problem (pA pB pC : ℝ) (hA : pA = 1 / 2) (hB : pB = 1 / 3) (hC : pC = 1 / 4) :
  (pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC) = 11 / 24 :=
by
  sorry

end exactly_one_solves_problem_l62_62942


namespace roger_and_friend_fraction_l62_62008

theorem roger_and_friend_fraction 
  (total_distance : ℝ) 
  (fraction_driven_before_lunch : ℝ) 
  (lunch_time : ℝ) 
  (total_time : ℝ) 
  (same_speed : Prop) 
  (driving_time_before_lunch : ℝ)
  (driving_time_after_lunch : ℝ) :
  total_distance = 200 ∧
  lunch_time = 1 ∧
  total_time = 5 ∧
  driving_time_before_lunch = 1 ∧
  driving_time_after_lunch = (total_time - lunch_time - driving_time_before_lunch) ∧
  same_speed = (total_distance * fraction_driven_before_lunch / driving_time_before_lunch = total_distance * (1 - fraction_driven_before_lunch) / driving_time_after_lunch) →
  fraction_driven_before_lunch = 1 / 4 :=
sorry

end roger_and_friend_fraction_l62_62008


namespace polynomial_evaluation_at_8_l62_62104

def P (x : ℝ) : ℝ := x^3 + 2*x^2 + x - 1

theorem polynomial_evaluation_at_8 : P 8 = 647 :=
by sorry

end polynomial_evaluation_at_8_l62_62104


namespace f_increasing_f_odd_function_l62_62946

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : ∀ (x1 x2 : ℝ), x1 < x2 → f a x1 < f a x2 :=
by
  sorry

theorem f_odd_function (a : ℝ) : f a 0 = 0 → (a = 1) :=
by
  sorry

end f_increasing_f_odd_function_l62_62946


namespace incorrect_statement_among_ABCD_l62_62226

theorem incorrect_statement_among_ABCD :
  ¬ (-3 = Real.sqrt ((-3)^2)) :=
by
  sorry

end incorrect_statement_among_ABCD_l62_62226


namespace ratio_M_N_l62_62893

theorem ratio_M_N (P Q M N : ℝ) (h1 : M = 0.30 * Q) (h2 : Q = 0.20 * P) (h3 : N = 0.50 * P) (hP_nonzero : P ≠ 0) :
  M / N = 3 / 25 := 
by 
  sorry

end ratio_M_N_l62_62893


namespace inverse_proportion_shift_l62_62011

theorem inverse_proportion_shift (x : ℝ) : 
  (∀ x, y = 6 / x) -> (y = 6 / (x - 3)) :=
by
  intro h
  sorry

end inverse_proportion_shift_l62_62011


namespace equation_solution_l62_62477

theorem equation_solution (x y : ℝ) (h : x^2 + (1 - y)^2 + (x - y)^2 = (1 / 3)) : 
  x = (1 / 3) ∧ y = (2 / 3) := 
  sorry

end equation_solution_l62_62477


namespace find_f_of_given_g_and_odd_l62_62040

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end find_f_of_given_g_and_odd_l62_62040


namespace ratio_of_roots_l62_62750

theorem ratio_of_roots 
  (a b c : ℝ) 
  (h : a * b * c ≠ 0)
  (x1 x2 : ℝ) 
  (root1 : x1 = 2022 * x2) 
  (root2 : a * x1 ^ 2 + b * x1 + c = 0) 
  (root3 : a * x2 ^ 2 + b * x2 + c = 0) : 
  2023 * a * c / b ^ 2 = 2022 / 2023 :=
by
  sorry

end ratio_of_roots_l62_62750


namespace problem_l62_62903

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : geometric_sequence a q)
  (h1 : a 0 + a 1 = 4 / 9)
  (h2 : a 2 + a 3 + a 4 + a 5 = 40) :
  (a 6 + a 7 + a 8) / 9 = 117 :=
sorry

end problem_l62_62903


namespace intersection_of_A_and_B_l62_62615

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | 1 ≤ x ∧ x < 4}

-- The theorem stating the problem
theorem intersection_of_A_and_B : A ∩ B = {1} :=
by
  sorry

end intersection_of_A_and_B_l62_62615


namespace sum_of_eight_numbers_l62_62786

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l62_62786


namespace zoo_visitors_l62_62325

theorem zoo_visitors (P : ℕ) (h : 3 * P = 3750) : P = 1250 :=
by 
  sorry

end zoo_visitors_l62_62325


namespace ratio_of_m_div_x_l62_62520

theorem ratio_of_m_div_x (a b : ℝ) (h1 : a / b = 4 / 5) (h2 : a > 0) (h3 : b > 0) :
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  (m / x) = 2 / 5 :=
by
  -- Define x and m
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  -- Include the steps or assumptions here if necessary
  sorry

end ratio_of_m_div_x_l62_62520


namespace percentage_equivalence_l62_62919

theorem percentage_equivalence (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 :=
sorry

end percentage_equivalence_l62_62919


namespace problem_statement_l62_62190

-- Define a set S
variable {S : Type*}

-- Define the binary operation on S
variable (mul : S → S → S)

-- Assume the given condition: (a * b) * a = b for all a, b in S
axiom given_condition : ∀ (a b : S), (mul (mul a b) a) = b

-- Prove that a * (b * a) = b for all a, b in S
theorem problem_statement : ∀ (a b : S), mul a (mul b a) = b :=
by
  sorry

end problem_statement_l62_62190


namespace train_speed_kmh_l62_62472

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end train_speed_kmh_l62_62472


namespace compute_a_l62_62976

theorem compute_a (a : ℝ) (h : 2.68 * 0.74 = a) : a = 1.9832 :=
by
  -- Here skip the proof steps
  sorry

end compute_a_l62_62976


namespace urn_problem_l62_62677

noncomputable def probability_of_two_black_balls : ℚ := (10 / 15) * (9 / 14)

theorem urn_problem : probability_of_two_black_balls = 3 / 7 := 
by
  sorry

end urn_problem_l62_62677


namespace gcd_g_x_l62_62091

noncomputable def g (x : ℕ) : ℕ :=
  (3 * x + 5) * (7 * x + 2) * (13 * x + 7) * (2 * x + 10)

theorem gcd_g_x (x : ℕ) (h : x % 19845 = 0) : Nat.gcd (g x) x = 700 :=
  sorry

end gcd_g_x_l62_62091


namespace stockholm_to_malmo_road_distance_l62_62955

-- Define constants based on the conditions
def map_distance_cm : ℕ := 120
def scale_factor : ℕ := 10
def road_distance_multiplier : ℚ := 1.15

-- Define the real distances based on the conditions
def straight_line_distance_km : ℕ :=
  map_distance_cm * scale_factor

def road_distance_km : ℚ :=
  straight_line_distance_km * road_distance_multiplier

-- Assert the final statement
theorem stockholm_to_malmo_road_distance :
  road_distance_km = 1380 := 
sorry

end stockholm_to_malmo_road_distance_l62_62955


namespace find_number_l62_62950

theorem find_number :
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  x = 24998465 :=
by
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  sorry

end find_number_l62_62950


namespace range_of_a_l62_62993

variable {R : Type} [LinearOrderedField R]

def is_even (f : R → R) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_non_neg (f : R → R) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem range_of_a 
  (f : R → R) 
  (even_f : is_even f)
  (mono_f : is_monotone_increasing_on_non_neg f)
  (ineq : ∀ a, f (a + 1) ≤ f 4) : 
  ∀ a, -5 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l62_62993


namespace problem_statement_l62_62071

theorem problem_statement : (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end problem_statement_l62_62071


namespace butterflies_count_l62_62023

theorem butterflies_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ) 
                          (h1 : total_black_dots = 4764) 
                          (h2 : black_dots_per_butterfly = 12) :
                          total_black_dots / black_dots_per_butterfly = 397 :=
by
  sorry

end butterflies_count_l62_62023


namespace brenda_ends_with_15_skittles_l62_62510

def initial_skittles : ℕ := 7
def skittles_bought : ℕ := 8

theorem brenda_ends_with_15_skittles : initial_skittles + skittles_bought = 15 := 
by {
  sorry
}

end brenda_ends_with_15_skittles_l62_62510


namespace matrix_identity_l62_62642

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 4, 3]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  B^4 = -3 • B + 2 • I :=
by
  sorry

end matrix_identity_l62_62642


namespace simplify_fraction_l62_62345

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) :=
by
  sorry

end simplify_fraction_l62_62345


namespace sufficient_but_not_necessary_condition_l62_62706

noncomputable def are_parallel (a : ℝ) : Prop :=
  (2 + a) * a * 3 * a = 3 * a * (a - 2)

theorem sufficient_but_not_necessary_condition :
  (are_parallel 4) ∧ (∃ a ≠ 4, are_parallel a) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l62_62706


namespace trees_planted_tomorrow_l62_62257

-- Definitions from the conditions
def current_trees := 39
def trees_planted_today := 41
def total_trees := 100

-- Theorem statement matching the proof problem
theorem trees_planted_tomorrow : 
  ∃ (trees_planted_tomorrow : ℕ), current_trees + trees_planted_today + trees_planted_tomorrow = total_trees ∧ trees_planted_tomorrow = 20 := 
by
  sorry

end trees_planted_tomorrow_l62_62257


namespace cos_alpha_second_quadrant_l62_62097

theorem cos_alpha_second_quadrant (α : ℝ) (h₁ : (π / 2) < α ∧ α < π) (h₂ : Real.sin α = 5 / 13) :
  Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_second_quadrant_l62_62097


namespace bajazet_winning_strategy_l62_62342

-- Define the polynomial P with place holder coefficients a, b, c (assuming they are real numbers)
def P (a b c : ℝ) (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + 1

-- The statement that regardless of how Alcina plays, Bajazet can ensure that P has a real root.
theorem bajazet_winning_strategy :
  ∃ (a b c : ℝ), ∃ (x : ℝ), P a b c x = 0 :=
sorry

end bajazet_winning_strategy_l62_62342


namespace checker_move_10_cells_checker_move_11_cells_l62_62422

noncomputable def F : ℕ → Nat 
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem checker_move_10_cells : F 10 = 89 := by
  sorry

theorem checker_move_11_cells : F 11 = 144 := by
  sorry

end checker_move_10_cells_checker_move_11_cells_l62_62422


namespace exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l62_62469

theorem exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4
  (n : ℕ) (hn₁ : Odd n) (hn₂ : 0 < n) :
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ p, p ∣ n ∧ Prime p ∧ p % 4 = 1 :=
by
  sorry

end exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l62_62469


namespace pam_number_of_bags_l62_62756

-- Definitions of the conditions
def apples_in_geralds_bag : Nat := 40
def pam_bags_ratio : Nat := 3
def total_pam_apples : Nat := 1200

-- Problem statement (Theorem)
theorem pam_number_of_bags :
  Pam_bags == total_pam_apples / (pam_bags_ratio * apples_in_geralds_bag) :=
by 
  sorry

end pam_number_of_bags_l62_62756


namespace collinear_points_l62_62101

variable (α β γ δ E : Type)
variables {A B C D K L P Q : α}
variables (convex : α → α → α → α → Prop)
variables (not_parallel : α → α → Prop)
variables (internal_bisector : α → α → α → Prop)
variables (external_bisector : α → α → α → Prop)
variables (collinear : α → α → α → α → Prop)

axiom convex_quad : convex A B C D
axiom AD_not_parallel_BC : not_parallel A D ∧ not_parallel B C

axiom internal_bisectors :
  internal_bisector A B K ∧ internal_bisector B A K ∧ internal_bisector C D P ∧ internal_bisector D C P

axiom external_bisectors :
  external_bisector A B L ∧ external_bisector B A L ∧ external_bisector C D Q ∧ external_bisector D C Q

theorem collinear_points : collinear K L P Q := 
sorry

end collinear_points_l62_62101


namespace main_theorem_l62_62145

-- Definitions based on conditions
variables (A P H M E C : ℕ) 
-- Thickness of an algebra book
def x := 1
-- Thickness of a history book (twice that of algebra)
def history_thickness := 2 * x
-- Length of shelf filled by books
def z := A * x

-- Condition equations based on shelf length equivalences
def equation1 := A = P
def equation2 := 2 * H * x = M * x
def equation3 := E * x + C * history_thickness = z

-- Prove the relationship
theorem main_theorem : C = (M * (A - E)) / (2 * A * H) :=
by
  sorry

end main_theorem_l62_62145


namespace inequality_proof_l62_62770

theorem inequality_proof (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ (a^c < b^c) ∧ (Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a) := 
sorry

end inequality_proof_l62_62770


namespace sum_of_consecutive_integers_l62_62067

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l62_62067


namespace last_term_of_sequence_l62_62206

theorem last_term_of_sequence (u₀ : ℤ) (diffs : List ℤ) (sum_diffs : ℤ) :
  u₀ = 0 → diffs = [2, 4, -1, 0, -5, -3, 3] → sum_diffs = diffs.sum → 
  u₀ + sum_diffs = 0 := by
  sorry

end last_term_of_sequence_l62_62206


namespace Travis_annual_cereal_cost_l62_62879

def cost_of_box_A : ℚ := 2.50
def cost_of_box_B : ℚ := 3.50
def cost_of_box_C : ℚ := 4.00
def cost_of_box_D : ℚ := 5.25
def cost_of_box_E : ℚ := 6.00

def quantity_of_box_A : ℚ := 1
def quantity_of_box_B : ℚ := 0.5
def quantity_of_box_C : ℚ := 0.25
def quantity_of_box_D : ℚ := 0.75
def quantity_of_box_E : ℚ := 1.5

def cost_week1 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week2 : ℚ :=
  let subtotal := 
    cost_of_box_A * quantity_of_box_A +
    cost_of_box_B * quantity_of_box_B +
    cost_of_box_C * quantity_of_box_C +
    cost_of_box_D * quantity_of_box_D +
    cost_of_box_E * quantity_of_box_E
  subtotal * 0.8

def cost_week3 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  0 +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week4 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  let discounted_box_E := cost_of_box_E * quantity_of_box_E * 0.85
  cost_of_box_A * quantity_of_box_A +
  discounted_box_E
  
def monthly_cost : ℚ :=
  cost_week1 + cost_week2 + cost_week3 + cost_week4

def annual_cost : ℚ :=
  monthly_cost * 12

theorem Travis_annual_cereal_cost :
  annual_cost = 792.24 := by
  sorry

end Travis_annual_cereal_cost_l62_62879


namespace remaining_files_calc_l62_62425

-- Definitions based on given conditions
def music_files : ℕ := 27
def video_files : ℕ := 42
def deleted_files : ℕ := 11

-- Theorem statement to prove the number of remaining files
theorem remaining_files_calc : music_files + video_files - deleted_files = 58 := by
  sorry

end remaining_files_calc_l62_62425


namespace inverse_cos_plus_one_l62_62289

noncomputable def f (x : ℝ) : ℝ := Real.cos x + 1

theorem inverse_cos_plus_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
    f (-(Real.arccos (x - 1))) = x :=
by
  sorry

end inverse_cos_plus_one_l62_62289


namespace linear_function_product_neg_l62_62470

theorem linear_function_product_neg (a1 b1 a2 b2 : ℝ) (hP : b1 = -3 * a1 + 4) (hQ : b2 = -3 * a2 + 4) :
  (a1 - a2) * (b1 - b2) < 0 :=
by
  sorry

end linear_function_product_neg_l62_62470


namespace line_parallel_through_point_l62_62939

theorem line_parallel_through_point (P : ℝ × ℝ) (a b c : ℝ) (ha : a = 3) (hb : b = -4) (hc : c = 6) (hP : P = (4, -1)) :
  ∃ d : ℝ, (d = -16) ∧ (∀ x y : ℝ, a * x + b * y + d = 0 ↔ 3 * x - 4 * y - 16 = 0) :=
by
  sorry

end line_parallel_through_point_l62_62939


namespace find_number_l62_62811

def digits_form_geometric_progression (x y z : ℕ) : Prop :=
  x * z = y * y

def swapped_hundreds_units (x y z : ℕ) : Prop :=
  100 * z + 10 * y + x = 100 * x + 10 * y + z - 594

def reversed_post_removal (x y z : ℕ) : Prop :=
  10 * z + y = 10 * y + z - 18

theorem find_number (x y z : ℕ) (h1 : digits_form_geometric_progression x y z) 
  (h2 : swapped_hundreds_units x y z) 
  (h3 : reversed_post_removal x y z) :
  100 * x + 10 * y + z = 842 := by
  sorry

end find_number_l62_62811


namespace interior_diagonals_of_dodecahedron_l62_62083

/-- Definition of a dodecahedron. -/
structure Dodecahedron where
  vertices : ℕ
  faces : ℕ
  vertices_per_face : ℕ
  faces_meeting_per_vertex : ℕ
  interior_diagonals : ℕ

/-- A dodecahedron has 12 pentagonal faces, 20 vertices, and 3 faces meet at each vertex. -/
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_meeting_per_vertex := 3,
    interior_diagonals := 160 }

theorem interior_diagonals_of_dodecahedron (d : Dodecahedron) :
    d.vertices = 20 → 
    d.faces = 12 →
    d.faces_meeting_per_vertex = 3 →
    d.interior_diagonals = 160 :=
by
  intros
  sorry

end interior_diagonals_of_dodecahedron_l62_62083


namespace power_of_power_l62_62607

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end power_of_power_l62_62607


namespace exists_divisible_by_2021_l62_62959

def concat_numbers (n m : ℕ) : ℕ :=
  -- function to concatenate numbers from n to m
  sorry

theorem exists_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concat_numbers n m :=
by
  sorry

end exists_divisible_by_2021_l62_62959


namespace remainder_when_divided_by_13_l62_62297

theorem remainder_when_divided_by_13 (N k : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 := by
  sorry

end remainder_when_divided_by_13_l62_62297


namespace student_marks_l62_62647

theorem student_marks (x : ℕ) :
  let total_questions := 60
  let correct_answers := 38
  let wrong_answers := total_questions - correct_answers
  let total_marks := 130
  let marks_from_correct := correct_answers * x
  let marks_lost := wrong_answers * 1
  let net_marks := marks_from_correct - marks_lost
  net_marks = total_marks → x = 4 :=
by
  intros
  sorry

end student_marks_l62_62647


namespace henry_classical_cds_l62_62586

variable (R C : ℕ)

theorem henry_classical_cds :
  (23 - 3 = R) →
  (R = 2 * C) →
  C = 10 :=
by
  intros h1 h2
  sorry

end henry_classical_cds_l62_62586


namespace john_weight_end_l62_62688

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l62_62688


namespace count_games_l62_62015

def total_teams : ℕ := 20
def games_per_pairing : ℕ := 7
def total_games := (total_teams * (total_teams - 1)) / 2 * games_per_pairing

theorem count_games : total_games = 1330 := by
  sorry

end count_games_l62_62015


namespace candy_difference_l62_62451

-- Defining the conditions as Lean hypotheses
variable (R K B M : ℕ)

-- Given conditions
axiom h1 : K = 4
axiom h2 : B = M - 6
axiom h3 : M = R + 2
axiom h4 : K = B + 2

-- Prove that Robert gets 2 more pieces of candy than Kate
theorem candy_difference : R - K = 2 :=
by {
  sorry
}

end candy_difference_l62_62451


namespace right_vs_oblique_prism_similarities_and_differences_l62_62478

-- Definitions of Prisms and their properties
structure Prism where
  parallel_bases : Prop
  congruent_bases : Prop
  parallelogram_faces : Prop

structure RightPrism extends Prism where
  rectangular_faces : Prop
  perpendicular_sides : Prop

structure ObliquePrism extends Prism where
  non_perpendicular_sides : Prop

theorem right_vs_oblique_prism_similarities_and_differences 
  (p1 : RightPrism) (p2 : ObliquePrism) : 
    (p1.parallel_bases ↔ p2.parallel_bases) ∧ 
    (p1.congruent_bases ↔ p2.congruent_bases) ∧ 
    (p1.parallelogram_faces ↔ p2.parallelogram_faces) ∧
    (p1.rectangular_faces ∧ p1.perpendicular_sides ↔ p2.non_perpendicular_sides) := 
by 
  sorry

end right_vs_oblique_prism_similarities_and_differences_l62_62478


namespace find_k_for_solutions_l62_62056

theorem find_k_for_solutions (k : ℝ) :
  (∀ x: ℝ, x = 3 ∨ x = 5 → k * x^2 - 8 * x + 15 = 0) → k = 1 :=
by
  sorry

end find_k_for_solutions_l62_62056


namespace distance_between_points_l62_62753

theorem distance_between_points : 
  let p1 := (0, 24)
  let p2 := (10, 0)
  dist p1 p2 = 26 := 
by
  sorry

end distance_between_points_l62_62753


namespace ellipse_symmetry_range_l62_62687

theorem ellipse_symmetry_range :
  ∀ (x₀ y₀ : ℝ), (x₀^2 / 4 + y₀^2 / 2 = 1) →
  ∃ (x₁ y₁ : ℝ), (x₁ = (4 * y₀ - 3 * x₀) / 5) ∧ (y₁ = (3 * y₀ + 4 * x₀) / 5) →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
by intros x₀ y₀ h_linearity; sorry

end ellipse_symmetry_range_l62_62687


namespace problem_arithmetic_sequence_l62_62411

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arithmetic_sequence (a : ℕ → ℝ) (d a2 a8 : ℝ) :
  arithmetic_sequence a d →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 450) →
  (a 1 + a 7 = 2 * a 4) →
  (a 2 + a 6 = 2 * a 4) →
  (a 2 + a 8 = 180) :=
by
  sorry

end problem_arithmetic_sequence_l62_62411


namespace tangent_y_axis_circle_eq_l62_62869

theorem tangent_y_axis_circle_eq (h k r : ℝ) (hc : h = -2) (kc : k = 3) (rc : r = abs h) :
  (x + h)^2 + (y - k)^2 = r^2 ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end tangent_y_axis_circle_eq_l62_62869


namespace equal_copper_content_alloy_l62_62654

theorem equal_copper_content_alloy (a b : ℝ) :
  ∃ x : ℝ, 0 < x ∧ x < 10 ∧
  (10 - x) * a + x * b = (15 - x) * b + x * a → x = 6 :=
by
  sorry

end equal_copper_content_alloy_l62_62654


namespace total_students_in_class_l62_62832

def total_students (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) : Nat :=
  (H / hands_per_student) + consider_teacher

theorem total_students_in_class (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) 
  (H_eq : H = 20) (hands_per_student_eq : hands_per_student = 2) (consider_teacher_eq : consider_teacher = 1) : 
  total_students H hands_per_student consider_teacher = 11 := by
  sorry

end total_students_in_class_l62_62832


namespace zack_traveled_countries_l62_62167

theorem zack_traveled_countries 
  (a : ℕ) (g : ℕ) (j : ℕ) (p : ℕ) (z : ℕ)
  (ha : a = 30)
  (hg : g = (3 / 5) * a)
  (hj : j = (1 / 3) * g)
  (hp : p = (4 / 3) * j)
  (hz : z = (5 / 2) * p) :
  z = 20 := 
sorry

end zack_traveled_countries_l62_62167


namespace distance_house_to_market_l62_62672

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end distance_house_to_market_l62_62672


namespace sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l62_62778

theorem sin_pi_six_minus_alpha_eq_one_third_cos_two_answer
  (α : ℝ) (h1 : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l62_62778


namespace find_dimensions_l62_62401

theorem find_dimensions (x y : ℝ) 
  (h1 : 90 = (2 * x + y) * (2 * y))
  (h2 : x * y = 10) : x = 2 ∧ y = 5 :=
by
  sorry

end find_dimensions_l62_62401


namespace rubiks_cube_repeats_l62_62437

theorem rubiks_cube_repeats (num_positions : ℕ) (H : num_positions = 43252003274489856000) 
  (moves : ℕ → ℕ) : 
  ∃ n, ∃ m, (∀ P, moves n = moves m → P = moves 0) :=
by
  sorry

end rubiks_cube_repeats_l62_62437


namespace sqrt_domain_condition_l62_62180

theorem sqrt_domain_condition (x : ℝ) : (2 * x - 6 ≥ 0) ↔ (x ≥ 3) :=
by
  sorry

end sqrt_domain_condition_l62_62180


namespace vacation_cost_per_person_l62_62185

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end vacation_cost_per_person_l62_62185


namespace sum_first_12_terms_l62_62772

-- Defining the basic sequence recurrence relation
def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + (-1 : ℝ) ^ n * a n = 2 * (n : ℝ) - 1

-- Theorem statement: Sum of the first 12 terms of the given sequence is 78
theorem sum_first_12_terms (a : ℕ → ℝ) (h : seq a) : 
  (Finset.range 12).sum a = 78 := 
sorry

end sum_first_12_terms_l62_62772


namespace cupcake_frosting_l62_62733

theorem cupcake_frosting :
  (let cagney_rate := (1 : ℝ) / 24
   let lacey_rate := (1 : ℝ) / 40
   let sammy_rate := (1 : ℝ) / 30
   let total_time := 12 * 60
   let combined_rate := cagney_rate + lacey_rate + sammy_rate
   total_time * combined_rate = 72) :=
by 
   -- Proof goes here
   sorry

end cupcake_frosting_l62_62733


namespace ramu_profit_percent_l62_62197

def ramu_bought_car : ℝ := 48000
def ramu_repair_cost : ℝ := 14000
def ramu_selling_price : ℝ := 72900

theorem ramu_profit_percent :
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  profit_percent = 17.58 := 
by
  -- Definitions and setting up the proof environment
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  sorry

end ramu_profit_percent_l62_62197


namespace minimum_fraction_l62_62935

theorem minimum_fraction (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m + 2 * n = 8) : 2 / m + 1 / n = 1 :=
by
  sorry

end minimum_fraction_l62_62935


namespace scientific_calculators_ordered_l62_62785

variables (x y : ℕ)

theorem scientific_calculators_ordered :
  (10 * x + 57 * y = 1625) ∧ (x + y = 45) → x = 20 :=
by
  -- proof goes here
  sorry

end scientific_calculators_ordered_l62_62785


namespace vanya_number_l62_62507

def S (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem vanya_number:
  (2014 + S 2014 = 2021) ∧ (1996 + S 1996 = 2021) := by
  sorry

end vanya_number_l62_62507


namespace minimum_value_a_plus_2b_l62_62930

theorem minimum_value_a_plus_2b {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) : a + 2 * b = 9 :=
by sorry

end minimum_value_a_plus_2b_l62_62930


namespace general_formula_a_n_sum_T_n_l62_62888

-- Definitions of the sequences
def a (n : ℕ) : ℕ := 4 + (n - 1) * 1
def S (n : ℕ) : ℕ := n / 2 * (2 * 4 + (n - 1) * 1)
def b (n : ℕ) : ℕ := 2 ^ (a n - 3)
def T (n : ℕ) : ℕ := 2 * (2 ^ n - 1)

-- Given conditions
axiom a4_eq_7 : a 4 = 7
axiom S2_eq_9 : S 2 = 9

-- Theorems to prove
theorem general_formula_a_n : ∀ n, a n = n + 3 := 
by sorry

theorem sum_T_n : ∀ n, T n = 2 ^ (n + 1) - 2 := 
by sorry

end general_formula_a_n_sum_T_n_l62_62888


namespace compute_g_neg_101_l62_62526

variable (g : ℝ → ℝ)

def functional_eqn := ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
def g_neg_one := g (-1) = 3
def g_one := g (1) = 1

theorem compute_g_neg_101 (g : ℝ → ℝ)
  (H1 : functional_eqn g)
  (H2 : g_neg_one g)
  (H3 : g_one g) :
  g (-101) = 103 := 
by
  sorry

end compute_g_neg_101_l62_62526


namespace jan_paid_amount_l62_62761

def number_of_roses (dozens : Nat) : Nat := dozens * 12

def total_cost (number_of_roses : Nat) (cost_per_rose : Nat) : Nat := number_of_roses * cost_per_rose

def discounted_price (total_cost : Nat) (discount_percentage : Nat) : Nat := total_cost * discount_percentage / 100

theorem jan_paid_amount :
  let dozens := 5
  let cost_per_rose := 6
  let discount_percentage := 80
  number_of_roses dozens = 60 →
  total_cost (number_of_roses dozens) cost_per_rose = 360 →
  discounted_price (total_cost (number_of_roses dozens) cost_per_rose) discount_percentage = 288 :=
by
  intros
  sorry

end jan_paid_amount_l62_62761


namespace janice_total_hours_worked_l62_62948

-- Declare the conditions as definitions
def hourly_rate_first_40_hours : ℝ := 10
def hourly_rate_overtime : ℝ := 15
def first_40_hours : ℕ := 40
def total_pay : ℝ := 700

-- Define the main theorem
theorem janice_total_hours_worked (H : ℕ) (O : ℕ) : 
  H = first_40_hours + O ∧ (hourly_rate_first_40_hours * first_40_hours + hourly_rate_overtime * O = total_pay) → H = 60 :=
by
  sorry

end janice_total_hours_worked_l62_62948


namespace fixed_point_PQ_passes_l62_62920

theorem fixed_point_PQ_passes (P Q : ℝ × ℝ) (x1 x2 : ℝ)
  (hP : P = (x1, x1^2))
  (hQ : Q = (x2, x2^2))
  (hC1 : x1 ≠ 0)
  (hC2 : x2 ≠ 0)
  (hSlopes : (x2 / x2^2 * (2 * x1)) = -2) :
  ∃ D : ℝ × ℝ, D = (0, 1) ∧
    ∀ (x y : ℝ), (y = x1^2 + (x1 - (1 / x1)) * (x - x1)) → ((x, y) = P ∨ (x, y) = Q) := sorry

end fixed_point_PQ_passes_l62_62920


namespace inhabitable_fraction_of_mars_surface_l62_62421

theorem inhabitable_fraction_of_mars_surface :
  (3 / 5 : ℚ) * (2 / 3) = (2 / 5) :=
by
  sorry

end inhabitable_fraction_of_mars_surface_l62_62421


namespace subtract_add_example_l62_62146

theorem subtract_add_example : (3005 - 3000) + 10 = 15 :=
by
  sorry

end subtract_add_example_l62_62146


namespace reduced_price_per_dozen_is_approx_2_95_l62_62094

noncomputable def original_price : ℚ := 16 / 39
noncomputable def reduced_price := 0.6 * original_price
noncomputable def reduced_price_per_dozen := reduced_price * 12

theorem reduced_price_per_dozen_is_approx_2_95 :
  abs (reduced_price_per_dozen - 2.95) < 0.01 :=
by
  sorry

end reduced_price_per_dozen_is_approx_2_95_l62_62094


namespace solve_x_if_alpha_beta_eq_8_l62_62461

variable (x : ℝ)

def alpha (x : ℝ) := 4 * x + 9
def beta (x : ℝ) := 9 * x + 6

theorem solve_x_if_alpha_beta_eq_8 (hx : alpha (beta x) = 8) : x = (-25 / 36) :=
by
  sorry

end solve_x_if_alpha_beta_eq_8_l62_62461


namespace investment_amount_l62_62975

noncomputable def total_investment (A T : ℝ) : Prop :=
  (0.095 * T = 0.09 * A + 2750) ∧ (T = A + 25000)

theorem investment_amount :
  ∃ T, ∀ A, total_investment A T ∧ T = 100000 :=
by
  sorry

end investment_amount_l62_62975


namespace train_speed_l62_62131

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) (h_distance : distance = 7.5) (h_time : time_minutes = 5) :
  speed = 90 :=
by
  sorry

end train_speed_l62_62131


namespace hindi_speaking_children_l62_62729

-- Condition Definitions
def total_children : ℕ := 90
def percent_only_english : ℝ := 0.25
def percent_only_hindi : ℝ := 0.15
def percent_only_spanish : ℝ := 0.10
def percent_english_hindi : ℝ := 0.20
def percent_english_spanish : ℝ := 0.15
def percent_hindi_spanish : ℝ := 0.10
def percent_all_three : ℝ := 0.05

-- Question translated to a Lean statement
theorem hindi_speaking_children :
  (percent_only_hindi + percent_english_hindi + percent_hindi_spanish + percent_all_three) * total_children = 45 :=
by
  sorry

end hindi_speaking_children_l62_62729


namespace max_distance_m_l62_62272

def circle_eq (x y : ℝ) := x^2 + y^2 - 4*x + 6*y - 3 = 0
def line_eq (m x y : ℝ) := m * x + y + m - 1 = 0
def center_circle (x y : ℝ) := circle_eq x y → (x = 2) ∧ (y = -3)

theorem max_distance_m :
  ∃ m : ℝ, line_eq m (-1) 1 ∧ ∀ x y t u : ℝ, center_circle x y → line_eq m t u → 
  -(4 / 3) * -m = -1 → m = -(3 / 4) :=
sorry

end max_distance_m_l62_62272


namespace nancy_total_spending_l62_62252

theorem nancy_total_spending :
  let this_month_games := 9
  let this_month_price := 5
  let last_month_games := 8
  let last_month_price := 4
  let next_month_games := 7
  let next_month_price := 6
  let total_cost := (this_month_games * this_month_price) +
                    (last_month_games * last_month_price) +
                    (next_month_games * next_month_price)
  total_cost = 119 :=
by
  sorry

end nancy_total_spending_l62_62252


namespace share_of_B_is_2400_l62_62303

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end share_of_B_is_2400_l62_62303


namespace common_root_unique_k_l62_62833

theorem common_root_unique_k (k : ℝ) (x : ℝ) 
  (h₁ : x^2 + k * x - 12 = 0) 
  (h₂ : 3 * x^2 - 8 * x - 3 * k = 0) 
  : k = 1 :=
sorry

end common_root_unique_k_l62_62833


namespace sum_of_angles_l62_62587

variables (A B C D E F : ℝ)

theorem sum_of_angles 
  (h : E = 30) :
  A + B + C + D + E + F = 420 :=
sorry

end sum_of_angles_l62_62587


namespace gcf_45_135_90_l62_62049

theorem gcf_45_135_90 : Nat.gcd (Nat.gcd 45 135) 90 = 45 := 
by
  sorry

end gcf_45_135_90_l62_62049


namespace trapezoid_area_l62_62540

theorem trapezoid_area:
  let vert1 := (10, 10)
  let vert2 := (15, 15)
  let vert3 := (0, 15)
  let vert4 := (0, 10)
  let base1 := 10
  let base2 := 15
  let height := 5
  ∃ (area : ℝ), area = 62.5 := by
  sorry

end trapezoid_area_l62_62540


namespace gcd_6Tn_nplus1_l62_62692

theorem gcd_6Tn_nplus1 (n : ℕ) (h : 0 < n) : gcd (3 * n * n + 3 * n) (n + 1) = 1 := by
  sorry

end gcd_6Tn_nplus1_l62_62692


namespace wyatt_headmaster_duration_l62_62821

def Wyatt_start_month : Nat := 3 -- March
def Wyatt_break_start_month : Nat := 7 -- July
def Wyatt_break_end_month : Nat := 12 -- December
def Wyatt_end_year : Nat := 2011

def months_worked_before_break : Nat := Wyatt_break_start_month - Wyatt_start_month -- March to June (inclusive, hence -1)
def break_duration : Nat := 6
def months_worked_after_break : Nat := 12 -- January to December 2011

def total_months_worked : Nat := months_worked_before_break + months_worked_after_break
theorem wyatt_headmaster_duration : total_months_worked = 16 :=
by
  sorry

end wyatt_headmaster_duration_l62_62821


namespace parity_sum_matches_parity_of_M_l62_62631

theorem parity_sum_matches_parity_of_M (N M : ℕ) (even_numbers odd_numbers : ℕ → ℤ)
  (hn : ∀ i, i < N → even_numbers i % 2 = 0)
  (hm : ∀ i, i < M → odd_numbers i % 2 ≠ 0) : 
  (N + M) % 2 = M % 2 := 
sorry

end parity_sum_matches_parity_of_M_l62_62631


namespace total_fruits_proof_l62_62208

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end total_fruits_proof_l62_62208


namespace leap_years_count_l62_62851

def is_leap_year (y : ℕ) : Bool :=
  if y % 800 = 300 ∨ y % 800 = 600 then true else false

theorem leap_years_count : 
  { y : ℕ // 1500 ≤ y ∧ y ≤ 3500 ∧ y % 100 = 0 ∧ is_leap_year y } = {y | y = 1900 ∨ y = 2200 ∨ y = 2700 ∨ y = 3000 ∨ y = 3500} :=
by
  sorry

end leap_years_count_l62_62851


namespace ratio_of_areas_l62_62661

noncomputable def length_field : ℝ := 16
noncomputable def width_field : ℝ := length_field / 2
noncomputable def area_field : ℝ := length_field * width_field
noncomputable def side_pond : ℝ := 4
noncomputable def area_pond : ℝ := side_pond * side_pond
noncomputable def ratio_area_pond_to_field : ℝ := area_pond / area_field

theorem ratio_of_areas :
  ratio_area_pond_to_field = 1 / 8 :=
  by
  sorry

end ratio_of_areas_l62_62661


namespace meaningful_range_l62_62680

   noncomputable def isMeaningful (x : ℝ) : Prop :=
     (3 - x ≥ 0) ∧ (x + 1 ≠ 0)

   theorem meaningful_range :
     ∀ x : ℝ, isMeaningful x ↔ (x ≤ 3 ∧ x ≠ -1) :=
   by
     sorry
   
end meaningful_range_l62_62680


namespace volunteer_selection_count_l62_62907

open Nat

theorem volunteer_selection_count :
  let boys : ℕ := 5
  let girls : ℕ := 2
  let total_ways := choose girls 1 * choose boys 2 + choose girls 2 * choose boys 1
  total_ways = 25 :=
by
  sorry

end volunteer_selection_count_l62_62907


namespace line_no_intersect_parabola_range_l62_62070

def parabola_eq (x : ℝ) : ℝ := x^2 + 4

def line_eq (m x : ℝ) : ℝ := m * (x - 10) + 6

theorem line_no_intersect_parabola_range (r s m : ℝ) :
  (m^2 - 40 * m + 8 = 0) →
  r < s →
  (∀ x, parabola_eq x ≠ line_eq m x) →
  r + s = 40 :=
by
  sorry

end line_no_intersect_parabola_range_l62_62070


namespace joyce_pencils_given_l62_62294

def original_pencils : ℕ := 51
def total_pencils_after : ℕ := 57

theorem joyce_pencils_given : total_pencils_after - original_pencils = 6 :=
by
  sorry

end joyce_pencils_given_l62_62294


namespace sum_of_cubes_l62_62201

variable (a b c : ℝ)

theorem sum_of_cubes (h1 : a^2 + 3 * b = 2) (h2 : b^2 + 5 * c = 3) (h3 : c^2 + 7 * a = 6) :
  a^3 + b^3 + c^3 = -0.875 :=
by
  sorry

end sum_of_cubes_l62_62201


namespace cider_production_l62_62755

theorem cider_production (gd_pint : ℕ) (pl_pint : ℕ) (gs_pint : ℕ) (farmhands : ℕ) (gd_rate : ℕ) (pl_rate : ℕ) (gs_rate : ℕ) (work_hours : ℕ) 
  (gd_total : ℕ) (pl_total : ℕ) (gs_total : ℕ) (gd_ratio : ℕ) (pl_ratio : ℕ) (gs_ratio : ℕ) 
  (gd_pint_val : gd_pint = 20) (pl_pint_val : pl_pint = 40) (gs_pint_val : gs_pint = 30)
  (farmhands_val : farmhands = 6) (gd_rate_val : gd_rate = 120) (pl_rate_val : pl_rate = 240) (gs_rate_val : gs_rate = 180) 
  (work_hours_val : work_hours = 5) 
  (gd_total_val : gd_total = farmhands * work_hours * gd_rate) 
  (pl_total_val : pl_total = farmhands * work_hours * pl_rate) 
  (gs_total_val : gs_total = farmhands * work_hours * gs_rate) 
  (gd_ratio_val : gd_ratio = 1) (pl_ratio_val : pl_ratio = 2) (gs_ratio_val : gs_ratio = 3/2) 
  (ratio_condition : gd_total / gd_ratio = pl_total / pl_ratio ∧ pl_total / pl_ratio = gs_total / gs_ratio) : 
  (gd_total / gd_pint) = 180 := 
sorry

end cider_production_l62_62755


namespace miles_per_hour_l62_62957

theorem miles_per_hour (total_distance : ℕ) (total_hours : ℕ) (h1 : total_distance = 81) (h2 : total_hours = 3) :
  total_distance / total_hours = 27 :=
by
  sorry

end miles_per_hour_l62_62957


namespace distinct_c_values_l62_62156

theorem distinct_c_values (c r s t : ℂ) 
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_unity : ∃ ω : ℂ, ω^3 = 1 ∧ r = 1 ∧ s = ω ∧ t = ω^2)
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t)) :
  ∃ (c_vals : Finset ℂ), c_vals.card = 3 ∧ ∀ (c' : ℂ), c' ∈ c_vals → c'^3 = 1 :=
by
  sorry

end distinct_c_values_l62_62156


namespace Keiko_speed_l62_62571

theorem Keiko_speed (a b s : ℝ) (h1 : 8 = 8) 
  (h2 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : 
  s = π / 3 :=
by
  sorry

end Keiko_speed_l62_62571


namespace vova_gave_pavlik_three_nuts_l62_62565

variable {V P k : ℕ}
variable (h1 : V > P)
variable (h2 : V - P = 2 * P)
variable (h3 : k ≤ 5)
variable (h4 : ∃ m : ℕ, V - k = 3 * m)

theorem vova_gave_pavlik_three_nuts (h1 : V > P) (h2 : V - P = 2 * P) (h3 : k ≤ 5) (h4 : ∃ m : ℕ, V - k = 3 * m) : k = 3 := by
  sorry

end vova_gave_pavlik_three_nuts_l62_62565


namespace xyz_value_l62_62207

variables {x y z : ℂ}

theorem xyz_value (h1 : x * y + 2 * y = -8)
                  (h2 : y * z + 2 * z = -8)
                  (h3 : z * x + 2 * x = -8) :
  x * y * z = 32 :=
by
  sorry

end xyz_value_l62_62207


namespace coffee_table_price_l62_62941

theorem coffee_table_price :
  let sofa := 1250
  let armchairs := 2 * 425
  let rug := 350
  let bookshelf := 200
  let subtotal_without_coffee_table := sofa + armchairs + rug + bookshelf
  let C := 429.24
  let total_before_discount_and_tax := subtotal_without_coffee_table + C
  let discounted_total := total_before_discount_and_tax * 0.90
  let final_invoice_amount := discounted_total * 1.06
  final_invoice_amount = 2937.60 :=
by
  sorry

end coffee_table_price_l62_62941


namespace abs_inequality_solution_l62_62719

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l62_62719


namespace scaled_polynomial_roots_l62_62102

noncomputable def polynomial_with_scaled_roots : Polynomial ℂ :=
  Polynomial.X^3 - 3*Polynomial.X^2 + 5

theorem scaled_polynomial_roots :
  (∃ r1 r2 r3 : ℂ, polynomial_with_scaled_roots.eval r1 = 0 ∧ polynomial_with_scaled_roots.eval r2 = 0 ∧ polynomial_with_scaled_roots.eval r3 = 0 ∧
  (∃ q : Polynomial ℂ, q = Polynomial.X^3 - 9*Polynomial.X^2 + 135 ∧
  ∀ y, (q.eval y = 0 ↔ (polynomial_with_scaled_roots.eval (y / 3) = 0)))) := sorry

end scaled_polynomial_roots_l62_62102


namespace ages_total_l62_62356

-- Define the variables and conditions
variables (A B C : ℕ)

-- State the conditions
def condition1 (B : ℕ) : Prop := B = 14
def condition2 (A B : ℕ) : Prop := A = B + 2
def condition3 (B C : ℕ) : Prop := B = 2 * C

-- The main theorem to prove
theorem ages_total (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 B C) : A + B + C = 37 :=
by
  sorry

end ages_total_l62_62356


namespace initial_money_l62_62385

theorem initial_money {M : ℝ} (h : (M - 10) - (M - 10) / 4 = 15) : M = 30 :=
sorry

end initial_money_l62_62385


namespace rulers_added_initially_46_finally_71_l62_62062

theorem rulers_added_initially_46_finally_71 : 
  ∀ (initial final added : ℕ), initial = 46 → final = 71 → added = final - initial → added = 25 :=
by
  intros initial final added h_initial h_final h_added
  rw [h_initial, h_final] at h_added
  exact h_added

end rulers_added_initially_46_finally_71_l62_62062


namespace zero_people_with_fewer_than_six_cards_l62_62267

theorem zero_people_with_fewer_than_six_cards (cards people : ℕ) (h_cards : cards = 60) (h_people : people = 9) :
  let avg := cards / people
  let remainder := cards % people
  remainder < people → ∃ n, n = 0 := by
  sorry

end zero_people_with_fewer_than_six_cards_l62_62267


namespace sample_variance_l62_62193

theorem sample_variance (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) :
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  sorry

end sample_variance_l62_62193


namespace determinant_of_roots_l62_62992

noncomputable def determinant_expr (a b c d s p q r : ℝ) : ℝ :=
  by sorry

theorem determinant_of_roots (a b c d s p q r : ℝ)
    (h1 : a + b + c + d = -s)
    (h2 : abcd = r)
    (h3 : abc + abd + acd + bcd = -q)
    (h4 : ab + ac + bc = p) :
    determinant_expr a b c d s p q r = r - q + pq + p :=
  by sorry

end determinant_of_roots_l62_62992


namespace find_sum_of_x_and_y_l62_62885

theorem find_sum_of_x_and_y (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end find_sum_of_x_and_y_l62_62885


namespace proposition_D_is_true_l62_62597

-- Define the propositions
def proposition_A : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def proposition_B : Prop := ∀ x : ℝ, 2^x > x^2
def proposition_C : Prop := ∀ a b : ℝ, (a + b = 0 ↔ a / b = -1)
def proposition_D : Prop := ∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1

-- Problem statement: Proposition D is true
theorem proposition_D_is_true : proposition_D := 
by sorry

end proposition_D_is_true_l62_62597


namespace sqrt_1_0201_eq_1_01_l62_62861

theorem sqrt_1_0201_eq_1_01 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 :=
by 
  sorry

end sqrt_1_0201_eq_1_01_l62_62861


namespace statement_b_statement_c_l62_62159
-- Import all of Mathlib to include necessary mathematical functions and properties

-- First, the Lean statement for Statement B
theorem statement_b (a b : ℝ) (h : a > |b|) : a^2 > b^2 := 
sorry

-- Second, the Lean statement for Statement C
theorem statement_c (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end statement_b_statement_c_l62_62159


namespace math_proof_problem_l62_62483

noncomputable def proof_problem (c d : ℝ) : Prop :=
  (∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  ∧ (∃ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3)
  ∧ 100 * c + d = 141
  
theorem math_proof_problem (c d : ℝ) 
  (h1 : ∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  (h2 : ∀ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3) :
  100 * c + d = 141 := 
sorry

end math_proof_problem_l62_62483


namespace seq_fixed_point_l62_62645

theorem seq_fixed_point (a_0 b_0 : ℝ) (a b : ℕ → ℝ)
  (h1 : a 0 = a_0)
  (h2 : b 0 = b_0)
  (h3 : ∀ n, a (n + 1) = a n + b n)
  (h4 : ∀ n, b (n + 1) = a n * b n) :
  a 2022 = a_0 ∧ b 2022 = b_0 ↔ b_0 = 0 := sorry

end seq_fixed_point_l62_62645


namespace find_m_if_polynomial_is_square_l62_62501

theorem find_m_if_polynomial_is_square (m : ℝ) :
  (∀ x, ∃ k : ℝ, x^2 + 2 * (m - 3) * x + 16 = (x + k)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_if_polynomial_is_square_l62_62501


namespace correct_polynomials_are_l62_62122

noncomputable def polynomial_solution (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p.eval (x^2) = (p.eval x) * (p.eval (x - 1))

theorem correct_polynomials_are (p : Polynomial ℝ) :
  polynomial_solution p ↔ ∃ n : ℕ, p = (Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℝ) * Polynomial.X + Polynomial.C (1 : ℝ)) ^ n :=
by
  sorry

end correct_polynomials_are_l62_62122


namespace new_ratio_is_one_half_l62_62327

theorem new_ratio_is_one_half (x : ℕ) (y : ℕ) (h1 : y = 4 * x) (h2 : y = 48) :
  (x + 12) / y = 1 / 2 :=
by
  sorry

end new_ratio_is_one_half_l62_62327


namespace intersection_A_B_l62_62155

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := 
by 
  sorry

end intersection_A_B_l62_62155


namespace remaining_black_cards_l62_62024

def total_black_cards_per_deck : ℕ := 26
def num_decks : ℕ := 5
def removed_black_face_cards : ℕ := 7
def removed_black_number_cards : ℕ := 12

theorem remaining_black_cards : total_black_cards_per_deck * num_decks - (removed_black_face_cards + removed_black_number_cards) = 111 :=
by
  -- proof will go here
  sorry

end remaining_black_cards_l62_62024


namespace carols_weight_l62_62739

variables (a c : ℝ)

theorem carols_weight (h1 : a + c = 220) (h2 : c - a = c / 3 + 10) : c = 138 :=
by
  sorry

end carols_weight_l62_62739


namespace binary_multiplication_addition_l62_62780

-- Define the binary representation of the given numbers
def b1101 : ℕ := 0b1101
def b111 : ℕ := 0b111
def b1011 : ℕ := 0b1011
def b1011010 : ℕ := 0b1011010

-- State the theorem
theorem binary_multiplication_addition :
  (b1101 * b111 + b1011) = b1011010 := 
sorry

end binary_multiplication_addition_l62_62780


namespace real_imaginary_part_above_x_axis_polynomial_solutions_l62_62806

-- Question 1: For what values of the real number m is (m^2 - 2m - 15) > 0
theorem real_imaginary_part_above_x_axis (m : ℝ) : 
  (m^2 - 2 * m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

-- Question 2: For what values of the real number m does 2m^2 + 3m - 4=0?
theorem polynomial_solutions (m : ℝ) : 
  (2 * m^2 + 3 * m - 4 = 0) ↔ (m = -3 ∨ m = 2) :=
sorry

end real_imaginary_part_above_x_axis_polynomial_solutions_l62_62806


namespace birds_in_tree_l62_62365

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (h : initial_birds = 21.0) (h_flew : birds_flew_away = 14.0) : 
initial_birds - birds_flew_away = 7.0 :=
by
  -- proof goes here
  sorry

end birds_in_tree_l62_62365


namespace pencils_pens_total_l62_62926

theorem pencils_pens_total (x : ℕ) (h1 : 4 * x + 1 = 7 * (5 * x - 1)) : 4 * x + 5 * x = 45 :=
by
  sorry

end pencils_pens_total_l62_62926


namespace billy_sleep_total_l62_62381

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l62_62381


namespace smallest_positive_integer_N_l62_62769

theorem smallest_positive_integer_N :
  ∃ N : ℕ, N > 0 ∧ (N % 7 = 5) ∧ (N % 8 = 6) ∧ (N % 9 = 7) ∧ (∀ M : ℕ, M > 0 ∧ (M % 7 = 5) ∧ (M % 8 = 6) ∧ (M % 9 = 7) → N ≤ M) :=
sorry

end smallest_positive_integer_N_l62_62769


namespace rope_cut_into_pieces_l62_62467

theorem rope_cut_into_pieces (length_of_rope_cm : ℕ) (num_equal_pieces : ℕ) (length_equal_piece_mm : ℕ) (length_remaining_piece_mm : ℕ) 
  (h1 : length_of_rope_cm = 1165) (h2 : num_equal_pieces = 150) (h3 : length_equal_piece_mm = 75) (h4 : length_remaining_piece_mm = 100) :
  (num_equal_pieces * length_equal_piece_mm + (11650 - num_equal_pieces * length_equal_piece_mm) / length_remaining_piece_mm = 154) :=
by
  sorry

end rope_cut_into_pieces_l62_62467


namespace hungarian_1905_l62_62200

open Nat

theorem hungarian_1905 (n p : ℕ) : (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p^z) ↔ 
  (p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ¬ ∃ k : ℕ, n = p^k) :=
by
  sorry

end hungarian_1905_l62_62200


namespace tan_alpha_eq_neg_one_l62_62789

theorem tan_alpha_eq_neg_one (α : ℝ) (h1 : |Real.sin α| = |Real.cos α|)
    (h2 : π / 2 < α ∧ α < π) : Real.tan α = -1 :=
sorry

end tan_alpha_eq_neg_one_l62_62789


namespace part1_part2_l62_62543

open Real

variables (α : ℝ) (A : (ℝ × ℝ)) (B : (ℝ × ℝ)) (C : (ℝ × ℝ))

def points_coordinates : Prop :=
A = (3, 0) ∧ B = (0, 3) ∧ C = (cos α, sin α) ∧ π / 2 < α ∧ α < 3 * π / 2

theorem part1 (h : points_coordinates α A B C) (h1 : dist (3, 0) (cos α, sin α) = dist (0, 3) (cos α, sin α)) : 
  α = 5 * π / 4 :=
sorry

theorem part2 (h : points_coordinates α A B C) (h2 : ((cos α - 3) * cos α + (sin α) * (sin α - 3)) = -1) : 
  (2 * sin α * sin α + sin (2 * α)) / (1 + tan α) = -5 / 9 :=
sorry

end part1_part2_l62_62543


namespace john_final_price_l62_62277

theorem john_final_price : 
  let goodA_price := 2500
  let goodA_rebate := 0.06 * goodA_price
  let goodA_price_after_rebate := goodA_price - goodA_rebate
  let goodA_sales_tax := 0.10 * goodA_price_after_rebate
  let goodA_final_price := goodA_price_after_rebate + goodA_sales_tax
  
  let goodB_price := 3150
  let goodB_rebate := 0.08 * goodB_price
  let goodB_price_after_rebate := goodB_price - goodB_rebate
  let goodB_sales_tax := 0.12 * goodB_price_after_rebate
  let goodB_final_price := goodB_price_after_rebate + goodB_sales_tax

  let goodC_price := 1000
  let goodC_rebate := 0.05 * goodC_price
  let goodC_price_after_rebate := goodC_price - goodC_rebate
  let goodC_sales_tax := 0.07 * goodC_price_after_rebate
  let goodC_final_price := goodC_price_after_rebate + goodC_sales_tax

  let total_amount := goodA_final_price + goodB_final_price + goodC_final_price

  let special_voucher_discount := 0.03 * total_amount
  let final_price := total_amount - special_voucher_discount
  let rounded_final_price := Float.round final_price

  rounded_final_price = 6642 := by
  sorry

end john_final_price_l62_62277


namespace first_line_shift_time_l62_62109

theorem first_line_shift_time (x y : ℝ) (h1 : (1 / x) + (1 / (x - 2)) + (1 / y) = 1.5 * ((1 / x) + (1 / (x - 2)))) 
  (h2 : x - 24 / 5 = (1 / ((1 / (x - 2)) + (1 / y)))) :
  x = 8 :=
sorry

end first_line_shift_time_l62_62109


namespace square_side_length_square_area_l62_62814

theorem square_side_length 
  (d : ℝ := 4) : (s : ℝ) = 2 * Real.sqrt 2 :=
  sorry

theorem square_area 
  (s : ℝ := 2 * Real.sqrt 2) : (A : ℝ) = 8 :=
  sorry

end square_side_length_square_area_l62_62814


namespace right_triangle_sides_l62_62628

theorem right_triangle_sides (x y z : ℕ) (h_sum : x + y + z = 156) (h_area : x * y = 2028) (h_pythagorean : z^2 = x^2 + y^2) :
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by
  admit -- proof goes here

-- Additional details for importing required libraries and setting up the environment
-- are intentionally simplified as per instruction to cover a broader import.

end right_triangle_sides_l62_62628


namespace distance_of_route_l62_62758

-- Define the conditions
def round_trip_time : ℝ := 1 -- in hours
def avg_speed : ℝ := 3 -- in miles per hour
def return_speed : ℝ := 6.000000000000002 -- in miles per hour

-- Problem statement to prove
theorem distance_of_route : 
  ∃ (D : ℝ), 
  2 * D = avg_speed * round_trip_time ∧ 
  D = 1.5 := 
by
  sorry

end distance_of_route_l62_62758


namespace scientific_notation_correct_l62_62567

theorem scientific_notation_correct :
  27600 = 2.76 * 10^4 :=
sorry

end scientific_notation_correct_l62_62567


namespace triangle_tan_inequality_l62_62779

theorem triangle_tan_inequality (A B C : ℝ) (hA : A + B + C = π) :
    (Real.tan A)^2 + (Real.tan B)^2 + (Real.tan C)^2 ≥ (Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + (Real.tan C) * (Real.tan A) :=
by
  sorry

end triangle_tan_inequality_l62_62779


namespace winner_won_by_288_votes_l62_62031

theorem winner_won_by_288_votes (V : ℝ) (votes_won : ℝ) (perc_won : ℝ) 
(h1 : perc_won = 0.60)
(h2 : votes_won = 864)
(h3 : votes_won = perc_won * V) : 
votes_won - (1 - perc_won) * V = 288 := 
sorry

end winner_won_by_288_votes_l62_62031


namespace inverse_g_167_is_2_l62_62854

def g (x : ℝ) := 5 * x^5 + 7

theorem inverse_g_167_is_2 : g⁻¹' {167} = {2} := by
  sorry

end inverse_g_167_is_2_l62_62854


namespace number_of_children_l62_62110

-- Definitions based on conditions
def numDogs : ℕ := 2
def numCats : ℕ := 1
def numLegsTotal : ℕ := 22
def numLegsDog : ℕ := 4
def numLegsCat : ℕ := 4
def numLegsHuman : ℕ := 2

-- Main theorem proving the number of children
theorem number_of_children :
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  numChildren = 4 :=
by
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  exact sorry

end number_of_children_l62_62110


namespace sum_of_first_five_primes_with_units_digit_3_l62_62849

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l62_62849


namespace div_by_1963_iff_odd_l62_62911

-- Define the given condition and statement
theorem div_by_1963_iff_odd (n : ℕ) :
  (1963 ∣ (82^n + 454 * 69^n)) ↔ (n % 2 = 1) :=
sorry

end div_by_1963_iff_odd_l62_62911


namespace find_number_of_As_l62_62143

variables (M L S : ℕ)

def number_of_As (M L S : ℕ) : Prop :=
  M + L = 23 ∧ S + M = 18 ∧ S + L = 15

theorem find_number_of_As (M L S : ℕ) (h : number_of_As M L S) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end find_number_of_As_l62_62143


namespace total_questions_on_test_l62_62484

theorem total_questions_on_test :
  ∀ (correct incorrect score : ℕ),
  (score = correct - 2 * incorrect) →
  (score = 76) →
  (correct = 92) →
  (correct + incorrect = 100) :=
by
  intros correct incorrect score grading_system score_eq correct_eq
  sorry

end total_questions_on_test_l62_62484


namespace circle_equation_correct_l62_62842

theorem circle_equation_correct (x y : ℝ) :
  let h : ℝ := -2
  let k : ℝ := 2
  let r : ℝ := 5
  ((x - h)^2 + (y - k)^2 = r^2) ↔ ((x + 2)^2 + (y - 2)^2 = 25) :=
by
  sorry

end circle_equation_correct_l62_62842


namespace damage_conversion_l62_62972

def usd_to_cad_conversion_rate : ℝ := 1.25
def damage_in_usd : ℝ := 60000000
def damage_in_cad : ℝ := 75000000

theorem damage_conversion :
  damage_in_usd * usd_to_cad_conversion_rate = damage_in_cad :=
sorry

end damage_conversion_l62_62972


namespace union_sets_l62_62287

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_sets : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_sets_l62_62287


namespace smaller_solution_l62_62922

theorem smaller_solution (x : ℝ) (h : x^2 + 9 * x - 22 = 0) : x = -11 :=
sorry

end smaller_solution_l62_62922


namespace exists_idempotent_l62_62495

-- Definition of the set M as the natural numbers from 1 to 1993
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1993 }

-- Operation * on M
noncomputable def star (a b : ℕ) : ℕ := sorry

-- Hypothesis: * is closed on M and (a * b) * a = b for any a, b in M
axiom star_closed (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star a b ∈ M
axiom star_property (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star (star a b) a = b

-- Goal: Prove that there exists a number a in M such that a * a = a
theorem exists_idempotent : ∃ a ∈ M, star a a = a := by
  sorry

end exists_idempotent_l62_62495


namespace price_of_first_oil_is_54_l62_62140

/-- Let x be the price per litre of the first oil.
Given that 10 litres of the first oil are mixed with 5 litres of second oil priced at Rs. 66 per litre,
resulting in a 15-litre mixture costing Rs. 58 per litre, prove that x = 54. -/
theorem price_of_first_oil_is_54 :
  (∃ x : ℝ, x = 54) ↔
  (10 * x + 5 * 66 = 15 * 58) :=
by
  sorry

end price_of_first_oil_is_54_l62_62140


namespace find_x_l62_62092

theorem find_x (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 :=
by
  sorry

end find_x_l62_62092


namespace first_term_arithmetic_sequence_l62_62634

theorem first_term_arithmetic_sequence
    (a: ℚ)
    (S_n S_2n: ℕ → ℚ)
    (n: ℕ) 
    (h1: ∀ n > 0, S_n n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2: ∀ n > 0, S_2n (2 * n) = ((2 * n) * (2 * a + ((2 * n) - 1) * 5)) / 2)
    (h3: ∀ n > 0, (S_2n (2 * n)) / (S_n n) = 4) :
  a = 5 / 2 :=
by
  sorry

end first_term_arithmetic_sequence_l62_62634


namespace isosceles_triangle_perimeter_eq_70_l62_62027

-- Define the conditions
def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 60
def isosceles_triangle_base : ℕ := 30

-- Calculate the side of equilateral triangle
def equilateral_triangle_side : ℕ := equilateral_triangle_perimeter / 3

-- Lean 4 statement
theorem isosceles_triangle_perimeter_eq_70 :
  ∃ (a b c : ℕ), is_equilateral_triangle a b c ∧ 
  a + b + c = equilateral_triangle_perimeter →
  (is_isosceles_triangle a a isosceles_triangle_base) →
  a + a + isosceles_triangle_base = 70 :=
by
  sorry -- proof is omitted

end isosceles_triangle_perimeter_eq_70_l62_62027


namespace larger_square_uncovered_area_l62_62796

theorem larger_square_uncovered_area :
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  (area_larger - area_smaller) = 84 :=
by
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  sorry

end larger_square_uncovered_area_l62_62796


namespace sqrt_x_div_sqrt_y_as_fraction_l62_62353

theorem sqrt_x_div_sqrt_y_as_fraction 
  (x y : ℝ)
  (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = 54 * x / 115 * y * ((1/5)^2 + (1/7)^2 + (1/8)^2)) : 
  (Real.sqrt x) / (Real.sqrt y) = 49 / 29 :=
by
  sorry

end sqrt_x_div_sqrt_y_as_fraction_l62_62353


namespace minimize_expr_l62_62001

theorem minimize_expr : ∃ c : ℝ, (∀ d : ℝ, (3/4 * c^2 - 9 * c + 5) ≤ (3/4 * d^2 - 9 * d + 5)) ∧ c = 6 :=
by
  use 6
  sorry

end minimize_expr_l62_62001


namespace animal_count_l62_62264

variable (H C D : Nat)

theorem animal_count :
  (H + C + D = 72) → 
  (2 * H + 4 * C + 2 * D = 212) → 
  (C = 34) → 
  (H + D = 38) :=
by
  intros h1 h2 hc
  sorry

end animal_count_l62_62264


namespace intersection_M_N_l62_62449

def M (x : ℝ) : Prop := 2 - x > 0
def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

theorem intersection_M_N:
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l62_62449


namespace find_x_plus_y_l62_62118

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2011 + Real.pi :=
sorry

end find_x_plus_y_l62_62118


namespace original_number_is_seven_l62_62460

theorem original_number_is_seven (x : ℤ) (h : 3 * x - 6 = 15) : x = 7 :=
by
  sorry

end original_number_is_seven_l62_62460


namespace jordyn_total_payment_l62_62405

theorem jordyn_total_payment :
  let price_cherries := 5
  let price_olives := 7
  let price_grapes := 11
  let num_cherries := 50
  let num_olives := 75
  let num_grapes := 25
  let discount_cherries := 0.12
  let discount_olives := 0.08
  let discount_grapes := 0.15
  let sales_tax := 0.05
  let service_charge := 0.02
  let total_cherries := num_cherries * price_cherries
  let total_olives := num_olives * price_olives
  let total_grapes := num_grapes * price_grapes
  let discounted_cherries := total_cherries * (1 - discount_cherries)
  let discounted_olives := total_olives * (1 - discount_olives)
  let discounted_grapes := total_grapes * (1 - discount_grapes)
  let subtotal := discounted_cherries + discounted_olives + discounted_grapes
  let taxed_amount := subtotal * (1 + sales_tax)
  let final_amount := taxed_amount * (1 + service_charge)
  final_amount = 1002.32 :=
by
  sorry

end jordyn_total_payment_l62_62405


namespace points_on_opposite_sides_of_line_l62_62409

theorem points_on_opposite_sides_of_line (a : ℝ) :
  let A := (3, 1)
  let B := (-4, 6)
  (3 * A.1 - 2 * A.2 + a) * (3 * B.1 - 2 * B.2 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  let A := (3, 1)
  let B := (-4, 6)
  have hA : 3 * A.1 - 2 * A.2 + a = 7 + a := by sorry
  have hB : 3 * B.1 - 2 * B.2 + a = -24 + a := by sorry
  exact sorry

end points_on_opposite_sides_of_line_l62_62409


namespace find_a_value_l62_62134

theorem find_a_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (max (a^1) (a^2) + min (a^1) (a^2)) = 12) : a = 3 :=
by
  sorry

end find_a_value_l62_62134


namespace min_triangle_perimeter_proof_l62_62435

noncomputable def min_triangle_perimeter (l m n : ℕ) : ℕ :=
  if l > m ∧ m > n ∧ (3^l % 10000 = 3^m % 10000) ∧ (3^m % 10000 = 3^n % 10000) then
    l + m + n
  else
    0

theorem min_triangle_perimeter_proof : ∃ (l m n : ℕ), l > m ∧ m > n ∧ 
  (3^l % 10000 = 3^m % 10000) ∧
  (3^m % 10000 = 3^n % 10000) ∧ min_triangle_perimeter l m n = 3003 :=
  sorry

end min_triangle_perimeter_proof_l62_62435


namespace savings_calculation_l62_62162

-- Define the conditions as given in the problem
def income_expenditure_ratio (income expenditure : ℝ) : Prop :=
  ∃ x : ℝ, income = 10 * x ∧ expenditure = 4 * x

def income_value : ℝ := 19000

-- The final statement for the savings, where we will prove the above question == answer
theorem savings_calculation (income expenditure savings : ℝ)
  (h_ratio : income_expenditure_ratio income expenditure)
  (h_income : income = income_value) : savings = 11400 :=
by
  sorry

end savings_calculation_l62_62162


namespace unique_n_value_l62_62837

theorem unique_n_value :
  ∃ (n : ℕ), n > 0 ∧ (∃ k : ℕ, k > 0 ∧ k < 10 ∧ 111 * k = (n * (n + 1) / 2)) ∧ ∀ (m : ℕ), m > 0 → (∃ j : ℕ, j > 0 ∧ j < 10 ∧ 111 * j = (m * (m + 1) / 2)) → m = 36 :=
by
  sorry

end unique_n_value_l62_62837


namespace fraction_addition_simplification_l62_62855

theorem fraction_addition_simplification :
  (2 / 5 : ℚ) + (3 / 15) = 3 / 5 :=
by
  sorry

end fraction_addition_simplification_l62_62855


namespace smallest_integer_value_of_m_l62_62537

theorem smallest_integer_value_of_m (x y m : ℝ) 
  (h1 : 3*x + y = m + 8) 
  (h2 : 2*x + 2*y = 2*m + 5) 
  (h3 : x - y < 1) : 
  m >= 3 := 
sorry

end smallest_integer_value_of_m_l62_62537


namespace part_a_part_b_l62_62527

-- Part (a) Equivalent Proof Problem
theorem part_a (k : ℤ) : 
  ∃ a b c : ℤ, 3 * k - 2 = a ^ 2 + b ^ 3 + c ^ 3 := 
sorry

-- Part (b) Equivalent Proof Problem
theorem part_b (n : ℤ) : 
  ∃ a b c d : ℤ, n = a ^ 2 + b ^ 3 + c ^ 3 + d ^ 3 := 
sorry

end part_a_part_b_l62_62527


namespace total_seats_in_stadium_l62_62343

theorem total_seats_in_stadium (people_at_game : ℕ) (empty_seats : ℕ) (total_seats : ℕ)
  (h1 : people_at_game = 47) (h2 : empty_seats = 45) :
  total_seats = people_at_game + empty_seats :=
by
  rw [h1, h2]
  show total_seats = 47 + 45
  sorry

end total_seats_in_stadium_l62_62343


namespace calculate_myOp_l62_62111

-- Define the operation
def myOp (x y : ℝ) : ℝ := x^3 - y

-- Given condition for h as a real number
variable (h : ℝ)

-- The theorem we need to prove
theorem calculate_myOp : myOp (2 * h) (myOp (2 * h) (2 * h)) = 2 * h := by
  sorry

end calculate_myOp_l62_62111


namespace sophia_lost_pawns_l62_62282

theorem sophia_lost_pawns
    (total_pawns : ℕ := 16)
    (start_pawns_each : ℕ := 8)
    (chloe_lost : ℕ := 1)
    (pawns_left : ℕ := 10)
    (chloe_pawns_left : ℕ := start_pawns_each - chloe_lost) :
    total_pawns = 2 * start_pawns_each → 
    ∃ (sophia_lost : ℕ), sophia_lost = start_pawns_each - (pawns_left - chloe_pawns_left) :=
by 
    intros _ 
    use 5 
    sorry

end sophia_lost_pawns_l62_62282


namespace part1_part2_part3_l62_62724

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end part1_part2_part3_l62_62724


namespace even_function_a_value_monotonicity_on_neg_infinity_l62_62265

noncomputable def f (x a : ℝ) : ℝ := ((x + 1) * (x + a)) / (x^2)

-- (1) Proving f(x) is even implies a = -1
theorem even_function_a_value (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = -1 :=
by
  sorry

-- (2) Proving monotonicity on (-∞, 0) for f(x) with a = -1
theorem monotonicity_on_neg_infinity (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ < 0) :
  (f x₁ (-1) > f x₂ (-1)) :=
by
  sorry

end even_function_a_value_monotonicity_on_neg_infinity_l62_62265


namespace find_tan_theta_l62_62773

open Real

theorem find_tan_theta (θ : ℝ) (h1 : sin θ + cos θ = 7 / 13) (h2 : 0 < θ ∧ θ < π) :
  tan θ = -12 / 5 :=
sorry

end find_tan_theta_l62_62773


namespace distance_between_foci_l62_62847

-- Given problem
def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 144

theorem distance_between_foci :
  ∀ (x y : ℝ),
    hyperbola_eq x y →
    2 * Real.sqrt ((137 / 9) + (137 / 16)) / 72 = 38 * Real.sqrt 7 / 72 :=
by
  intros x y h
  sorry

end distance_between_foci_l62_62847


namespace find_first_dimension_l62_62223

variable (w h cost_per_sqft total_cost : ℕ)

def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

def insulation_cost (A cost_per_sqft : ℕ) : ℕ := A * cost_per_sqft

theorem find_first_dimension 
  (w := 7) (h := 2) (cost_per_sqft := 20) (total_cost := 1640) : 
  (∃ l : ℕ, insulation_cost (surface_area l w h) cost_per_sqft = total_cost) → 
  l = 3 := 
sorry

end find_first_dimension_l62_62223


namespace unit_digit_of_square_l62_62823

theorem unit_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := sorry

end unit_digit_of_square_l62_62823


namespace fisherman_daily_earnings_l62_62450

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end fisherman_daily_earnings_l62_62450


namespace smallest_b_greater_than_5_perfect_cube_l62_62021

theorem smallest_b_greater_than_5_perfect_cube : ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 3 = n ^ 3 ∧ b = 6 := 
by 
  sorry

end smallest_b_greater_than_5_perfect_cube_l62_62021


namespace compute_div_mul_l62_62266

theorem compute_div_mul (x y z : Int) (h : y ≠ 0) (hx : x = -100) (hy : y = -25) (hz : z = -6) :
  (((-x) / (-y)) * -z) = -24 := by
  sorry

end compute_div_mul_l62_62266


namespace main_theorem_l62_62825

variable (x y z : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1)

theorem main_theorem (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1):
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := 
by
  sorry

end main_theorem_l62_62825


namespace min_value_of_quadratic_l62_62887

theorem min_value_of_quadratic (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) : 
  ∃ (x : ℝ), (∃ (y_min : ℝ), y_min = -( (abs (a - b)/2)^2 ) 
  ∧ ∀ (x : ℝ), (x - a)*(x - b) ≥ y_min) :=
sorry

end min_value_of_quadratic_l62_62887


namespace steaks_from_15_pounds_of_beef_l62_62682

-- Definitions for conditions
def pounds_to_ounces (pounds : ℕ) : ℕ := pounds * 16

def steaks_count (total_ounces : ℕ) (ounces_per_steak : ℕ) : ℕ := total_ounces / ounces_per_steak

-- Translate the problem to Lean statement
theorem steaks_from_15_pounds_of_beef : 
  steaks_count (pounds_to_ounces 15) 12 = 20 :=
by
  sorry

end steaks_from_15_pounds_of_beef_l62_62682


namespace Seulgi_second_round_need_l62_62112

def Hohyeon_first_round := 23
def Hohyeon_second_round := 28
def Hyunjeong_first_round := 32
def Hyunjeong_second_round := 17
def Seulgi_first_round := 27

def Hohyeon_total := Hohyeon_first_round + Hohyeon_second_round
def Hyunjeong_total := Hyunjeong_first_round + Hyunjeong_second_round

def required_total_for_Seulgi := Hohyeon_total + 1

theorem Seulgi_second_round_need (Seulgi_second_round: ℕ) :
  Seulgi_first_round + Seulgi_second_round ≥ required_total_for_Seulgi → Seulgi_second_round ≥ 25 :=
by
  sorry

end Seulgi_second_round_need_l62_62112


namespace not_perfect_squares_l62_62725

theorem not_perfect_squares :
  (∀ x : ℝ, x * x ≠ 8 ^ 2041) ∧ (∀ y : ℝ, y * y ≠ 10 ^ 2043) :=
by
  sorry

end not_perfect_squares_l62_62725


namespace proof_f_prime_at_2_l62_62749

noncomputable def f_prime (x : ℝ) (f_prime_2 : ℝ) : ℝ :=
  2 * x + 2 * f_prime_2 - (1 / x)

theorem proof_f_prime_at_2 :
  ∃ (f_prime_2 : ℝ), f_prime 2 f_prime_2 = -7 / 2 :=
by
  sorry

end proof_f_prime_at_2_l62_62749


namespace trig_expression_value_l62_62609

theorem trig_expression_value (θ : ℝ) (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 := 
by 
  sorry

end trig_expression_value_l62_62609


namespace sum_h_k_a_b_l62_62311

-- Defining h, k, a, and b with their respective given values
def h : Int := -4
def k : Int := 2
def a : Int := 5
def b : Int := 3

-- Stating the theorem to prove \( h + k + a + b = 6 \)
theorem sum_h_k_a_b : h + k + a + b = 6 := by
  /- Proof omitted as per instructions -/
  sorry

end sum_h_k_a_b_l62_62311


namespace range_of_a_l62_62250

variable (a : ℝ)

def p : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

def q : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

theorem range_of_a :
  (p a ∧ q a) → a ≤ -1 := by
  sorry

end range_of_a_l62_62250


namespace geom_seq_sum_l62_62396

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : a 0 + a 1 = 3 / 4)
  (h4 : a 2 + a 3 + a 4 + a 5 = 15) :
  a 6 + a 7 + a 8 = 112 := by
  sorry

end geom_seq_sum_l62_62396


namespace divisor_of_1058_l62_62600

theorem divisor_of_1058 :
  ∃ (d : ℕ), (∃ (k : ℕ), 1058 = d * k) ∧ (¬ ∃ (d : ℕ), (∃ (l : ℕ), 1 < d ∧ d < 1058 ∧ 1058 = d * l)) :=
by {
  sorry
}

end divisor_of_1058_l62_62600


namespace find_local_value_of_7_in_difference_l62_62493

-- Define the local value of 3 in the number 28943712.
def local_value_of_3_in_28943712 : Nat := 30000

-- Define the property that the local value of 7 in a number Y is 7000.
def local_value_of_7 (Y : Nat) : Prop := (Y / 1000 % 10) = 7

-- Define the unknown number X and its difference with local value of 3 in 28943712.
variable (X : Nat)

-- Assumption: The difference between X and local_value_of_3_in_28943712 results in a number whose local value of 7 is 7000.
axiom difference_condition : local_value_of_7 (X - local_value_of_3_in_28943712)

-- The proof problem statement to be solved.
theorem find_local_value_of_7_in_difference : local_value_of_7 (X - local_value_of_3_in_28943712) = true :=
by
  -- Proof is omitted.
  sorry

end find_local_value_of_7_in_difference_l62_62493


namespace factor_count_x9_minus_x_l62_62189

theorem factor_count_x9_minus_x :
  ∃ (factors : List (Polynomial ℤ)), x^9 - x = factors.prod ∧ factors.length = 5 :=
sorry

end factor_count_x9_minus_x_l62_62189


namespace polynomial_expansion_l62_62986

theorem polynomial_expansion :
  (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
  35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := 
by
  sorry

end polynomial_expansion_l62_62986


namespace determine_house_numbers_l62_62284

-- Definitions based on the conditions given
def even_numbered_side (n : ℕ) : Prop :=
  n % 2 = 0

def sum_balanced (n : ℕ) (house_numbers : List ℕ) : Prop :=
  let left_sum := house_numbers.take n |>.sum
  let right_sum := house_numbers.drop (n + 1) |>.sum
  left_sum = right_sum

def house_constraints (n : ℕ) : Prop :=
  50 < n ∧ n < 500

-- Main theorem statement
theorem determine_house_numbers : 
  ∃ (n : ℕ) (house_numbers : List ℕ), 
    even_numbered_side n ∧ 
    house_constraints n ∧ 
    sum_balanced n house_numbers :=
  sorry

end determine_house_numbers_l62_62284


namespace jeremy_oranges_l62_62153

theorem jeremy_oranges (M : ℕ) (h : M + 3 * M + 70 = 470) : M = 100 := 
by
  sorry

end jeremy_oranges_l62_62153


namespace nonzero_fraction_exponent_zero_l62_62880

theorem nonzero_fraction_exponent_zero (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : (a / b : ℚ)^0 = 1 := 
by 
  sorry

end nonzero_fraction_exponent_zero_l62_62880


namespace income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l62_62105

-- Define the conditions
def annual_income (year : ℕ) : ℝ := 0.0124 * (1 + 0.2) ^ (year - 1)
def annual_repayment : ℝ := 0.05

-- Proof Problem 1: Show that the subway's annual operating income exceeds the annual repayment at year 9
theorem income_exceeds_repayment_after_9_years :
  ∀ n ≥ 9, annual_income n > annual_repayment :=
by
  sorry

-- Define the cumulative payment function for the municipal government
def cumulative_payment (years : ℕ) : ℝ :=
  (annual_repayment * years) - (List.sum (List.map annual_income (List.range years)))

-- Proof Problem 2: Show the cumulative payment by the municipal government up to year 8 is 19,541,135 RMB
theorem cumulative_payment_up_to_year_8 :
  cumulative_payment 8 = 0.1954113485 :=
by
  sorry

end income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l62_62105


namespace gcd_m_n_is_one_l62_62014

open Int
open Nat

-- Define m and n based on the given conditions
def m : ℤ := 130^2 + 240^2 + 350^2
def n : ℤ := 129^2 + 239^2 + 351^2

-- State the theorem to be proven
theorem gcd_m_n_is_one : gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l62_62014


namespace negation_of_p_l62_62906

def proposition_p (n : ℕ) : Prop := 3^n ≥ n + 1

theorem negation_of_p : (∃ n0 : ℕ, 3^n0 < n0^2 + 1) :=
  by sorry

end negation_of_p_l62_62906


namespace solve_abs_inequality_l62_62332

theorem solve_abs_inequality (x : ℝ) :
  3 ≤ abs ((x - 3)^2 - 4) ∧ abs ((x - 3)^2 - 4) ≤ 7 ↔ 3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11 :=
sorry

end solve_abs_inequality_l62_62332


namespace find_contributions_before_johns_l62_62074

-- Definitions based on the conditions provided
def avg_contrib_size_after (A : ℝ) := A + 0.5 * A = 75
def johns_contribution := 100
def total_amount_before (n : ℕ) (A : ℝ) := n * A
def total_amount_after (n : ℕ) (A : ℝ) := (n * A + johns_contribution)

-- Proposition we need to prove
theorem find_contributions_before_johns (n : ℕ) (A : ℝ) :
  avg_contrib_size_after A →
  total_amount_before n A + johns_contribution = (n + 1) * 75 →
  n = 1 :=
by
  sorry

end find_contributions_before_johns_l62_62074


namespace isosceles_trapezoid_ratio_l62_62579

theorem isosceles_trapezoid_ratio (a b h : ℝ) 
  (h1: h = b / 2)
  (h2: a = 1 - ((1 - b) / 2))
  (h3 : 1 = ((a + 1) / 2)^2 + (b / 2)^2) :
  b / a = (-1 + Real.sqrt 7) / 2 := 
sorry

end isosceles_trapezoid_ratio_l62_62579


namespace gcd_of_228_and_1995_l62_62029

theorem gcd_of_228_and_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

end gcd_of_228_and_1995_l62_62029


namespace solve_for_q_l62_62534

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 3 * p + 5 * q = 8) : q = 19 / 16 :=
by
  sorry

end solve_for_q_l62_62534


namespace find_prime_n_l62_62641

def is_prime (p : ℕ) : Prop := 
  p > 1 ∧ (∀ n, n ∣ p → n = 1 ∨ n = p)

def prime_candidates : List ℕ := [11, 17, 23, 29, 41, 47, 53, 59, 61, 71, 83, 89]

theorem find_prime_n (n : ℕ) 
  (h1 : n ∈ prime_candidates) 
  (h2 : is_prime (n)) 
  (h3 : is_prime (n + 20180500)) : 
  n = 61 :=
by sorry

end find_prime_n_l62_62641


namespace range_of_m_in_first_quadrant_l62_62541

theorem range_of_m_in_first_quadrant (m : ℝ) : ((m - 1 > 0) ∧ (m + 2 > 0)) ↔ m > 1 :=
by sorry

end range_of_m_in_first_quadrant_l62_62541


namespace probability_at_least_one_of_each_color_l62_62577

theorem probability_at_least_one_of_each_color
  (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h_total : total_balls = 16)
  (h_black : black_balls = 8)
  (h_white : white_balls = 5)
  (h_red : red_balls = 3) :
  ((black_balls.choose 1) * (white_balls.choose 1) * (red_balls.choose 1) : ℚ) / total_balls.choose 3 = 3 / 14 :=
by
  sorry

end probability_at_least_one_of_each_color_l62_62577


namespace interest_rate_of_second_part_l62_62128

theorem interest_rate_of_second_part 
  (total_sum : ℝ) (P2 : ℝ) (interest1_rate : ℝ) 
  (time1 : ℝ) (time2 : ℝ) (interest2_value : ℝ) : 
  (total_sum = 2704) → 
  (P2 = 1664) → 
  (interest1_rate = 0.03) → 
  (time1 = 8) → 
  (interest2_value = interest1_rate * (total_sum - P2) * time1) → 
  (time2 = 3) → 
  1664 * r * time2 = interest2_value → 
  r = 0.05 := 
by sorry

end interest_rate_of_second_part_l62_62128


namespace ratio_ac_bd_l62_62546

theorem ratio_ac_bd (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end ratio_ac_bd_l62_62546


namespace exists_k_l62_62231

theorem exists_k (m n : ℕ) : ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end exists_k_l62_62231


namespace seventh_term_geometric_sequence_l62_62073

theorem seventh_term_geometric_sequence :
  ∃ (a₁ a₁₀ a₇ : ℕ) (r : ℕ),
    a₁ = 6 ∧ a₁₀ = 93312 ∧
    a₁₀ = a₁ * r^9 ∧
    a₇ = a₁ * r^6 ∧
    a₇ = 279936 :=
by
  sorry

end seventh_term_geometric_sequence_l62_62073


namespace pole_length_after_cut_l62_62838

theorem pole_length_after_cut (original_length : ℝ) (percentage_retained : ℝ) : 
  original_length = 20 → percentage_retained = 0.7 → 
  original_length * percentage_retained = 14 :=
by
  intros h0 h1
  rw [h0, h1]
  norm_num

end pole_length_after_cut_l62_62838


namespace symmetric_point_in_third_quadrant_l62_62413

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to find the symmetric point about the y-axis
def symmetric_about_y (P : Point) : Point :=
  Point.mk (-P.x) P.y

-- Define the original point P
def P : Point := { x := 3, y := -2 }

-- Define the symmetric point P' about the y-axis
def P' : Point := symmetric_about_y P

-- Define a condition to determine if a point is in the third quadrant
def is_in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- The theorem stating that the symmetric point of P about the y-axis is in the third quadrant
theorem symmetric_point_in_third_quadrant : is_in_third_quadrant P' :=
  by
  sorry

end symmetric_point_in_third_quadrant_l62_62413


namespace min_value_expression_l62_62805

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + a * c = 4) :
  ∃ m, m = 4 ∧ m ≤ 2 / a + 2 / (b + c) + 8 / (a + b + c) :=
by
  sorry

end min_value_expression_l62_62805


namespace car_mileage_l62_62846

/-- If a car needs 3.5 gallons of gasoline to travel 140 kilometers, it gets 40 kilometers per gallon. -/
theorem car_mileage (gallons_used : ℝ) (distance_traveled : ℝ) 
  (h : gallons_used = 3.5 ∧ distance_traveled = 140) : 
  distance_traveled / gallons_used = 40 :=
by
  sorry

end car_mileage_l62_62846


namespace value_of_expression_l62_62655

theorem value_of_expression (x : ℕ) (h : x = 3) : 2 * x + 3 = 9 :=
by 
  sorry

end value_of_expression_l62_62655


namespace triangle_tangency_perimeter_l62_62673

def triangle_perimeter (a b c : ℝ) (s : ℝ) (t : ℝ) (u : ℝ) : ℝ :=
  s + t + u

theorem triangle_tangency_perimeter (a b c : ℝ) (D E F : ℝ) (s : ℝ) (t : ℝ) (u : ℝ)
  (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) 
  (h4 : s + t + u = 3) : triangle_perimeter a b c s t u = 3 :=
by
  sorry

end triangle_tangency_perimeter_l62_62673


namespace age_of_cat_l62_62232

variables (cat_age rabbit_age dog_age : ℕ)

-- Conditions
def condition1 : Prop := rabbit_age = cat_age / 2
def condition2 : Prop := dog_age = 3 * rabbit_age
def condition3 : Prop := dog_age = 12

-- Question
def question (cat_age : ℕ) : Prop := cat_age = 8

theorem age_of_cat (h1 : condition1 cat_age rabbit_age) (h2 : condition2 rabbit_age dog_age) (h3 : condition3 dog_age) : question cat_age :=
by
  sorry

end age_of_cat_l62_62232


namespace find_start_time_l62_62379

def time_first_train_started 
  (distance_pq : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (start_time_train2 : ℝ) 
  (meeting_time : ℝ) 
  (T : ℝ) : ℝ :=
  T

theorem find_start_time 
  (distance_pq : ℝ := 200)
  (speed_train1 : ℝ := 20)
  (speed_train2 : ℝ := 25)
  (start_time_train2 : ℝ := 8)
  (meeting_time : ℝ := 12) 
  : time_first_train_started distance_pq speed_train1 speed_train2 start_time_train2 meeting_time 7 = 7 :=
by
  sorry

end find_start_time_l62_62379


namespace largest_five_digit_divisible_by_97_l62_62403

theorem largest_five_digit_divisible_by_97 :
  ∃ n, (99999 - n % 97) = 99930 ∧ n % 97 = 0 ∧ 10000 ≤ n ∧ n ≤ 99999 :=
by
  sorry

end largest_five_digit_divisible_by_97_l62_62403


namespace find_b_collinear_points_l62_62737

theorem find_b_collinear_points :
  ∃ b : ℚ, 4 * 11 - 6 * (-3 * b + 4) = 5 * (b + 3) - 1 * 4 ∧ b = 11 / 26 :=
by
  sorry

end find_b_collinear_points_l62_62737


namespace exists_pairs_of_stops_l62_62243

def problem := ∃ (A1 B1 A2 B2 : Fin 6) (h1 : A1 < B1) (h2 : A2 < B2),
  (A1 ≠ A2 ∧ A1 ≠ B2 ∧ B1 ≠ A2 ∧ B1 ≠ B2) ∧
  ¬(∃ (a b : Fin 6), A1 = a ∧ B1 = b ∧ A2 = a ∧ B2 = b) -- such that no passenger boards at A1 and alights at B1
                                                              -- and no passenger boards at A2 and alights at B2.

theorem exists_pairs_of_stops (n : ℕ) (stops : Fin n) (max_passengers : ℕ) 
  (h : n = 6 ∧ max_passengers = 5 ∧ 
  ∀ (a b : Fin n), a < b → a < stops ∧ b < stops) : problem :=
sorry

end exists_pairs_of_stops_l62_62243


namespace james_correct_take_home_pay_l62_62515

noncomputable def james_take_home_pay : ℝ :=
  let main_job_hourly_rate := 20
  let second_job_hourly_rate := main_job_hourly_rate * 0.8
  let main_job_hours := 30
  let main_job_overtime_hours := 5
  let second_job_hours := 15
  let side_gig_daily_rate := 100
  let side_gig_days := 2
  let tax_deductions := 200
  let federal_tax_rate := 0.18
  let state_tax_rate := 0.05

  let regular_main_job_hours := main_job_hours - main_job_overtime_hours
  let main_job_regular_pay := regular_main_job_hours * main_job_hourly_rate
  let main_job_overtime_pay := main_job_overtime_hours * main_job_hourly_rate * 1.5
  let total_main_job_pay := main_job_regular_pay + main_job_overtime_pay

  let total_second_job_pay := second_job_hours * second_job_hourly_rate
  let total_side_gig_pay := side_gig_daily_rate * side_gig_days

  let total_earnings := total_main_job_pay + total_second_job_pay + total_side_gig_pay
  let taxable_income := total_earnings - tax_deductions
  let federal_tax := taxable_income * federal_tax_rate
  let state_tax := taxable_income * state_tax_rate
  let total_taxes := federal_tax + state_tax
  total_earnings - total_taxes

theorem james_correct_take_home_pay : james_take_home_pay = 885.30 := by
  sorry

end james_correct_take_home_pay_l62_62515


namespace max_integer_is_twelve_l62_62909

theorem max_integer_is_twelve
  (a b c d e : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : (a + b + c + d + e) / 5 = 9)
  (h6 : ((a - 9)^2 + (b - 9)^2 + (c - 9)^2 + (d - 9)^2 + (e - 9)^2) / 5 = 4) :
  e = 12 := sorry

end max_integer_is_twelve_l62_62909


namespace smallest_positive_b_l62_62247

theorem smallest_positive_b (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 6 = 5) ↔ 
  b = 59 :=
by
  sorry

end smallest_positive_b_l62_62247


namespace election_required_percentage_l62_62643

def votes_cast : ℕ := 10000

def geoff_percentage : ℕ := 5
def geoff_received_votes := (geoff_percentage * votes_cast) / 1000

def extra_votes_needed : ℕ := 5000
def total_votes_needed := geoff_received_votes + extra_votes_needed

def required_percentage := (total_votes_needed * 100) / votes_cast

theorem election_required_percentage : required_percentage = 505 / 10 :=
by
  sorry

end election_required_percentage_l62_62643


namespace problem_l62_62716

open Real

noncomputable def f (ω a x : ℝ) := (1 / 2) * (sin (ω * x) + a * cos (ω * x))

theorem problem (a : ℝ) 
  (hω_range : 0 < ω ∧ ω ≤ 1)
  (h_f_sym1 : ∀ x, f ω a x = f ω a (π/3 - x))
  (h_f_sym2 : ∀ x, f ω a (x - π) = f ω a (x + π))
  (x1 x2 : ℝ) 
  (h_x_in_interval1 : -π/3 < x1 ∧ x1 < 5*π/3)
  (h_x_in_interval2 : -π/3 < x2 ∧ x2 < 5*π/3)
  (h_distinct : x1 ≠ x2)
  (h_f_neg_half1 : f ω a x1 = -1/2)
  (h_f_neg_half2 : f ω a x2 = -1/2) :
  (f 1 (sqrt 3) x = sin (x + π/3)) ∧ (x1 + x2 = 7*π/3) :=
by
  sorry

end problem_l62_62716


namespace ap_contains_sixth_power_l62_62420

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end ap_contains_sixth_power_l62_62420


namespace find_f_value_l62_62617

noncomputable def f (x y z : ℝ) : ℝ := 2 * x^3 * Real.sin y + Real.log (z^2)

theorem find_f_value :
  f 1 (Real.pi / 2) (Real.exp 2) = 8 →
  f 2 Real.pi (Real.exp 3) = 6 :=
by
  intro h
  unfold f
  sorry

end find_f_value_l62_62617


namespace widow_share_l62_62235

theorem widow_share (w d s : ℝ) (h_sum : w + 5 * s + 4 * d = 8000)
  (h1 : d = 2 * w)
  (h2 : s = 3 * d) :
  w = 8000 / 39 := by
sorry

end widow_share_l62_62235


namespace total_pages_written_is_24_l62_62255

def normal_letter_interval := 3
def time_per_normal_letter := 20
def time_per_page := 10
def additional_time_factor := 2
def time_spent_long_letter := 80
def days_in_month := 30

def normal_letters_written := days_in_month / normal_letter_interval
def pages_per_normal_letter := time_per_normal_letter / time_per_page
def total_pages_normal_letters := normal_letters_written * pages_per_normal_letter

def time_per_page_long_letter := additional_time_factor * time_per_page
def pages_long_letter := time_spent_long_letter / time_per_page_long_letter

def total_pages_written := total_pages_normal_letters + pages_long_letter

theorem total_pages_written_is_24 : total_pages_written = 24 := by
  sorry

end total_pages_written_is_24_l62_62255


namespace determine_k_for_linear_dependence_l62_62165

theorem determine_k_for_linear_dependence :
  ∃ k : ℝ, (∀ (a1 a2 : ℝ), a1 ≠ 0 ∧ a2 ≠ 0 → 
  a1 • (⟨1, 2, 3⟩ : ℝ × ℝ × ℝ) + a2 • (⟨4, k, 6⟩ : ℝ × ℝ × ℝ) = (⟨0, 0, 0⟩ : ℝ × ℝ × ℝ)) → k = 8 :=
by
  sorry

end determine_k_for_linear_dependence_l62_62165


namespace elena_bread_max_flour_l62_62777

variable (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
variable (available_butter available_sugar : ℕ)

def max_flour (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
  (available_butter available_sugar : ℕ) : ℕ :=
  min (available_butter * sugar / butter_per_cup_flour) (available_sugar * butter / sugar_per_cup_flour)

theorem elena_bread_max_flour : 
  max_flour 3 4 2 5 24 30 = 32 := sorry

end elena_bread_max_flour_l62_62777


namespace range_of_a_l62_62666

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∨ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) ∧ ¬ (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∧ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) →
  a ≤ -2 ∨ 1 ≤ a ∧ a < 2 :=
by {
  sorry
}

end range_of_a_l62_62666


namespace problem_statement_l62_62473

theorem problem_statement (a : ℕ → ℝ)
  (h_recur : ∀ n, n ≥ 1 → a (n + 1) = a (n - 1) / (1 + n * a (n - 1) * a n))
  (h_initial_0 : a 0 = 1)
  (h_initial_1 : a 1 = 1) :
  1 / (a 190 * a 200) = 19901 :=
by
  sorry

end problem_statement_l62_62473


namespace complement_of_M_in_U_l62_62299

noncomputable def U : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
noncomputable def M : Set ℝ := { y | ∃ x, x^2 + y^2 = 1 }

theorem complement_of_M_in_U :
  (U \ M) = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end complement_of_M_in_U_l62_62299


namespace exists_three_irrationals_l62_62391

theorem exists_three_irrationals
    (x1 x2 x3 : ℝ)
    (h1 : ¬ ∃ q : ℚ, x1 = q)
    (h2 : ¬ ∃ q : ℚ, x2 = q)
    (h3 : ¬ ∃ q : ℚ, x3 = q)
    (sum_integer : ∃ n : ℤ, x1 + x2 + x3 = n)
    (sum_reciprocals_integer : ∃ m : ℤ, (1/x1) + (1/x2) + (1/x3) = m) :
  true :=
sorry

end exists_three_irrationals_l62_62391


namespace find_a_l62_62625

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a_l62_62625


namespace prob_not_rain_correct_l62_62033

noncomputable def prob_not_rain_each_day (prob_rain : ℚ) : ℚ :=
  1 - prob_rain

noncomputable def prob_not_rain_four_days (prob_not_rain : ℚ) : ℚ :=
  prob_not_rain ^ 4

theorem prob_not_rain_correct :
  prob_not_rain_four_days (prob_not_rain_each_day (2/3)) = 1 / 81 :=
by 
  sorry

end prob_not_rain_correct_l62_62033


namespace value_of_a_l62_62683

theorem value_of_a {a : ℝ} : 
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 2 = 0 ∧ ∀ y : ℝ, (a - 1) * y^2 + 4 * y - 2 ≠ 0 → y = x) → 
  (a = 1 ∨ a = -1) :=
by 
  sorry

end value_of_a_l62_62683


namespace cost_of_items_l62_62518

theorem cost_of_items (M R F : ℝ)
  (h1 : 10 * M = 24 * R) 
  (h2 : F = 2 * R) 
  (h3 : F = 20.50) : 
  4 * M + 3 * R + 5 * F = 231.65 := 
by
  sorry

end cost_of_items_l62_62518


namespace ratio_of_combined_area_to_combined_perimeter_l62_62623

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def equilateral_triangle_perimeter (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_combined_area_to_combined_perimeter :
  (equilateral_triangle_area 6 + equilateral_triangle_area 8) / 
  (equilateral_triangle_perimeter 6 + equilateral_triangle_perimeter 8) = (25 * Real.sqrt 3) / 42 :=
by
  sorry

end ratio_of_combined_area_to_combined_perimeter_l62_62623


namespace find_k_l62_62702

theorem find_k (k : ℝ) (hk : 0 < k) (slope_eq : (2 - k) / (k - 1) = k^2) : k = 1 :=
by sorry

end find_k_l62_62702


namespace average_time_per_stop_l62_62199

-- Definitions from the conditions
def pizzas : Nat := 12
def stops_with_two_pizzas : Nat := 2
def total_delivery_time : Nat := 40

-- Using the conditions to define what needs to be proved
theorem average_time_per_stop : 
  let single_pizza_stops := pizzas - stops_with_two_pizzas * 2
  let total_stops := single_pizza_stops + stops_with_two_pizzas
  let average_time := total_delivery_time / total_stops
  average_time = 4 := by
  -- Proof to be provided
  sorry

end average_time_per_stop_l62_62199


namespace number_of_men_first_group_l62_62002

theorem number_of_men_first_group :
  (∃ M : ℕ, 30 * 3 * (M : ℚ) * (84 / 30) / 3 = 112 / 6) → ∃ M : ℕ, M = 20 := 
by
  sorry

end number_of_men_first_group_l62_62002


namespace n_fraction_sum_l62_62211

theorem n_fraction_sum {n : ℝ} {lst : List ℝ} (h_len : lst.length = 21) 
(h_mem : n ∈ lst) 
(h_avg : n = 4 * (lst.erase n).sum / 20) :
  n = (lst.sum) / 6 :=
by
  sorry

end n_fraction_sum_l62_62211


namespace remainder_7_pow_137_mod_11_l62_62610

theorem remainder_7_pow_137_mod_11 :
    (137 = 13 * 10 + 7) →
    (7^10 ≡ 1 [MOD 11]) →
    (7^137 ≡ 6 [MOD 11]) :=
by
  intros h1 h2
  sorry

end remainder_7_pow_137_mod_11_l62_62610


namespace unique_digit_sum_l62_62076

theorem unique_digit_sum (Y M E T : ℕ) (h1 : Y ≠ M) (h2 : Y ≠ E) (h3 : Y ≠ T)
    (h4 : M ≠ E) (h5 : M ≠ T) (h6 : E ≠ T) (h7 : 10 * Y + E = YE) (h8 : 10 * M + E = ME)
    (h9 : YE * ME = T * T * T) (hT_even : T % 2 = 0) : 
    Y + M + E + T = 10 :=
  sorry

end unique_digit_sum_l62_62076


namespace adiabatic_compression_work_l62_62436

noncomputable def adiabatic_work (p1 V1 V2 k : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) : ℝ :=
  (p1 * V1) / (k - 1) * (1 - (V1 / V2)^(k - 1))

theorem adiabatic_compression_work (p1 V1 V2 k W : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) :
  W = adiabatic_work p1 V1 V2 k h₁ h₂ h₃ :=
sorry

end adiabatic_compression_work_l62_62436


namespace adam_has_23_tattoos_l62_62863

-- Conditions as definitions
def tattoos_on_each_of_jason_arms := 2
def number_of_jason_arms := 2
def tattoos_on_each_of_jason_legs := 3
def number_of_jason_legs := 2

def jason_total_tattoos : Nat :=
  tattoos_on_each_of_jason_arms * number_of_jason_arms + tattoos_on_each_of_jason_legs * number_of_jason_legs

def adam_tattoos (jason_tattoos : Nat) : Nat :=
  2 * jason_tattoos + 3

-- The main theorem to be proved
theorem adam_has_23_tattoos : adam_tattoos jason_total_tattoos = 23 := by
  sorry

end adam_has_23_tattoos_l62_62863


namespace intersect_graphs_exactly_four_l62_62776

theorem intersect_graphs_exactly_four (A : ℝ) (hA : 0 < A) :
  (∃ x y : ℝ, y = A * x^2 ∧ x^2 + 2 * y^2 = A + 3) ↔ (∀ x1 y1 x2 y2 : ℝ, (y1 = A * x1^2 ∧ x1^2 + 2 * y1^2 = A + 3) ∧ (y2 = A * x2^2 ∧ x2^2 + 2 * y2^2 = A + 3) → (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end intersect_graphs_exactly_four_l62_62776


namespace relationship_of_magnitudes_l62_62377

noncomputable def is_ordered (x : ℝ) (A B C : ℝ) : Prop :=
  0 < x ∧ x < Real.pi / 4 ∧
  A = Real.cos (x ^ Real.sin (x ^ Real.sin x)) ∧
  B = Real.sin (x ^ Real.cos (x ^ Real.sin x)) ∧
  C = Real.cos (x ^ Real.sin (x * (x ^ Real.cos x))) ∧
  B < A ∧ A < C

theorem relationship_of_magnitudes (x A B C : ℝ) : 
  is_ordered x A B C := 
sorry

end relationship_of_magnitudes_l62_62377


namespace city_rentals_cost_per_mile_l62_62466

theorem city_rentals_cost_per_mile (x : ℝ)
  (h₁ : 38.95 + 150 * x = 41.95 + 150 * 0.29) :
  x = 0.31 :=
by sorry

end city_rentals_cost_per_mile_l62_62466


namespace quadratic_double_root_eq1_quadratic_double_root_eq2_l62_62622

theorem quadratic_double_root_eq1 :
  (∃ r : ℝ , ∃ s : ℝ, (r ≠ s) ∧ (
  (1 : ℝ) * r^2 + (-3 : ℝ) * r + (2 : ℝ) = 0 ∧
  (1 : ℝ) * s^2 + (-3 : ℝ) * s + (2 : ℝ) = 0 ∧
  (r = 2 * s ∨ s = 2 * r) 
  )) := 
  sorry

theorem quadratic_double_root_eq2 :
  (∃ a b : ℝ, a ≠ 0 ∧
  ((∃ r : ℝ, (-b / a = 2 + r) ∧ (-6 / a = 2 * r)) ∨ 
  ((-b / a = 2 + 1) ∧ (-6 / a = 2 * 1))) ∧ 
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9))) :=
  sorry

end quadratic_double_root_eq1_quadratic_double_root_eq2_l62_62622


namespace smallest_share_arith_seq_l62_62058

theorem smallest_share_arith_seq (a1 d : ℚ) (h1 : 5 * a1 + 10 * d = 100) (h2 : (3 * a1 + 9 * d) * (1 / 7) = 2 * a1 + d) : a1 = 5 / 3 :=
by
  sorry

end smallest_share_arith_seq_l62_62058


namespace find_a_l62_62868
open Real

theorem find_a (a : ℝ) (k : ℤ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x1^2 + y1^2 = 10 * (x1 * cos a + y1 * sin a) ∧
     x2^2 + y2^2 = 10 * (x2 * sin (3 * a) + y2 * cos (3 * a)) ∧
     (x2 - x1)^2 + (y2 - y1)^2 = 64)) ↔
  (∃ k : ℤ, a = π / 8 + k * π / 2) :=
sorry

end find_a_l62_62868


namespace speed_ratio_l62_62406

theorem speed_ratio (L tA tB : ℝ) (R : ℝ) (h1: A_speed = R * B_speed) 
  (h2: head_start = 0.35 * L) (h3: finish_margin = 0.25 * L)
  (h4: A_distance = L + head_start) (h5: B_distance = L)
  (h6: A_finish = A_distance / A_speed)
  (h7: B_finish = B_distance / B_speed)
  (h8: B_finish_time = A_finish + finish_margin / B_speed)
  : R = 1.08 :=
by
  sorry

end speed_ratio_l62_62406


namespace time_period_is_12_hours_l62_62595

-- Define the conditions in the problem
def birth_rate := 8 / 2 -- people per second
def death_rate := 6 / 2 -- people per second
def net_increase := 86400 -- people

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Total time period in seconds
def time_period_seconds := net_increase / net_increase_per_second

-- Convert the time period to hours
def time_period_hours := time_period_seconds / 3600

-- The theorem we want to state and prove
theorem time_period_is_12_hours : time_period_hours = 12 :=
by
  -- Proof goes here
  sorry

end time_period_is_12_hours_l62_62595


namespace certain_number_d_sq_l62_62341

theorem certain_number_d_sq (d n m : ℕ) (hd : d = 14) (h : n * d = m^2) : n = 14 :=
by
  sorry

end certain_number_d_sq_l62_62341


namespace ordered_triples_eq_l62_62591

theorem ordered_triples_eq :
  ∃! (x y z : ℤ), x + y = 4 ∧ xy - z^2 = 3 ∧ (x = 2 ∧ y = 2 ∧ z = 0) :=
by
  -- Proof goes here
  sorry

end ordered_triples_eq_l62_62591


namespace find_n_l62_62291

theorem find_n :
  ∃ n : ℤ, 3 ^ 3 - 7 = 4 ^ 2 + 2 + n ∧ n = 2 :=
by
  use 2
  sorry

end find_n_l62_62291


namespace first_term_of_geometric_series_l62_62423

theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) :
  r = 1 / 4 → S = 20 → S = a / (1 - r) → a = 15 :=
by
  intro hr hS hsum
  sorry

end first_term_of_geometric_series_l62_62423


namespace length_of_GH_l62_62522

variable (S_A S_C S_E S_F : ℝ)
variable (AB FE CD GH : ℝ)

-- Given conditions
axiom h1 : AB = 11
axiom h2 : FE = 13
axiom h3 : CD = 5

-- Relationships between the sizes of the squares
axiom h4 : S_A = S_C + AB
axiom h5 : S_C = S_E + CD
axiom h6 : S_E = S_F + FE
axiom h7 : GH = S_A - S_F

theorem length_of_GH : GH = 29 :=
by
  -- This is where the proof would go
  sorry

end length_of_GH_l62_62522


namespace jason_text_messages_per_day_l62_62000

theorem jason_text_messages_per_day
  (monday_messages : ℕ)
  (tuesday_messages : ℕ)
  (total_messages : ℕ)
  (average_per_day : ℕ)
  (messages_wednesday_friday_per_day : ℕ) :
  monday_messages = 220 →
  tuesday_messages = monday_messages / 2 →
  average_per_day = 96 →
  total_messages = 5 * average_per_day →
  total_messages - (monday_messages + tuesday_messages) = 3 * messages_wednesday_friday_per_day →
  messages_wednesday_friday_per_day = 50 :=
by
  intros
  sorry

end jason_text_messages_per_day_l62_62000


namespace houses_in_block_l62_62665

theorem houses_in_block (junk_per_house : ℕ) (total_junk : ℕ) (h_junk : junk_per_house = 2) (h_total : total_junk = 14) :
  total_junk / junk_per_house = 7 := by
  sorry

end houses_in_block_l62_62665


namespace problem1_solution_set_problem2_range_of_a_l62_62324

-- Definitions and statements for Problem 1
def f1 (x : ℝ) : ℝ := -12 * x ^ 2 - 2 * x + 2

theorem problem1_solution_set :
  (∃ a b : ℝ, a = -12 ∧ b = -2 ∧
    ∀ x : ℝ, f1 x > 0 → -1 / 2 < x ∧ x < 1 / 3) :=
by sorry

-- Definitions and statements for Problem 2
def f2 (x a : ℝ) : ℝ := a * x ^ 2 - x + 2

theorem problem2_range_of_a :
  (∃ b : ℝ, b = -1 ∧
    ∀ a : ℝ, (∀ x : ℝ, f2 x a < 0 → false) → a ≥ 1 / 8) :=
by sorry

end problem1_solution_set_problem2_range_of_a_l62_62324


namespace records_given_l62_62967

theorem records_given (X : ℕ) (started_with : ℕ) (bought : ℕ) (days_per_record : ℕ) (total_days : ℕ)
  (h1 : started_with = 8) (h2 : bought = 30) (h3 : days_per_record = 2) (h4 : total_days = 100) :
  X = 12 := by
  sorry

end records_given_l62_62967


namespace gcd_consecutive_terms_l62_62547

theorem gcd_consecutive_terms (n : ℕ) : 
  Nat.gcd (2 * Nat.factorial n + n) (2 * Nat.factorial (n + 1) + (n + 1)) = 1 :=
by
  sorry

end gcd_consecutive_terms_l62_62547


namespace proportional_segments_l62_62363

theorem proportional_segments (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d1 d2 d3 d4 : ℕ)
  (hA : a1 = 1 ∧ a2 = 2 ∧ a3 = 3 ∧ a4 = 4)
  (hB : b1 = 1 ∧ b2 = 2 ∧ b3 = 2 ∧ b4 = 4)
  (hC : c1 = 3 ∧ c2 = 5 ∧ c3 = 9 ∧ c4 = 13)
  (hD : d1 = 1 ∧ d2 = 2 ∧ d3 = 2 ∧ d4 = 3) :
  (b1 * b4 = b2 * b3) :=
by
  sorry

end proportional_segments_l62_62363


namespace latitude_approx_l62_62836

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l62_62836


namespace complement_of_M_in_U_l62_62344

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}
def C_U (M : Set ℕ) (U : Set ℕ) : Set ℕ := U \ M

theorem complement_of_M_in_U : C_U M U = {1, 4} :=
by
  sorry

end complement_of_M_in_U_l62_62344


namespace pirate_rick_digging_time_l62_62016

theorem pirate_rick_digging_time :
  ∀ (initial_depth rate: ℕ) (storm_factor tsunami_added: ℕ),
  initial_depth = 8 →
  rate = 2 →
  storm_factor = 2 →
  tsunami_added = 2 →
  (initial_depth / storm_factor + tsunami_added) / rate = 3 := 
by
  intros
  sorry

end pirate_rick_digging_time_l62_62016


namespace solution_set_x_squared_minus_3x_lt_0_l62_62304

theorem solution_set_x_squared_minus_3x_lt_0 : { x : ℝ | x^2 - 3 * x < 0 } = { x : ℝ | 0 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_x_squared_minus_3x_lt_0_l62_62304


namespace new_weekly_income_l62_62028

-- Define the conditions
def original_income : ℝ := 60
def raise_percentage : ℝ := 0.20

-- Define the question and the expected answer
theorem new_weekly_income : original_income * (1 + raise_percentage) = 72 := 
by
  sorry

end new_weekly_income_l62_62028


namespace max_drinks_amount_l62_62659

noncomputable def initial_milk : ℚ := 3 / 4
noncomputable def rachel_fraction : ℚ := 1 / 2
noncomputable def max_fraction : ℚ := 1 / 3

def amount_rachel_drinks (initial: ℚ) (fraction: ℚ) : ℚ := initial * fraction
def remaining_milk_after_rachel (initial: ℚ) (amount_rachel: ℚ) : ℚ := initial - amount_rachel
def amount_max_drinks (remaining: ℚ) (fraction: ℚ) : ℚ := remaining * fraction

theorem max_drinks_amount :
  amount_max_drinks (remaining_milk_after_rachel initial_milk (amount_rachel_drinks initial_milk rachel_fraction)) max_fraction = 1 / 8 := 
sorry

end max_drinks_amount_l62_62659


namespace stewarts_theorem_l62_62279

theorem stewarts_theorem 
  (a b b₁ a₁ d c : ℝ)
  (h₁ : b * b ≠ 0) 
  (h₂ : a * a ≠ 0) 
  (h₃ : b₁ * b₁ ≠ 0) 
  (h₄ : a₁ * a₁ ≠ 0) 
  (h₅ : d * d ≠ 0) 
  (h₆ : c = a₁ + b₁) :
  b * b * a₁ + a * a * b₁ - d * d * c = a₁ * b₁ * c :=
  sorry

end stewarts_theorem_l62_62279


namespace problem_inequality_l62_62694

theorem problem_inequality 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m + n = 1) : 
  (m + 1 / m) * (n + 1 / n) ≥ 25 / 4 := 
sorry

end problem_inequality_l62_62694


namespace actual_cost_l62_62860

theorem actual_cost (x : ℝ) (h : 0.80 * x = 200) : x = 250 :=
sorry

end actual_cost_l62_62860


namespace geom_seq_decreasing_l62_62003

variable {a : ℕ → ℝ}
variable {a₁ q : ℝ}

theorem geom_seq_decreasing (h : ∀ n, a n = a₁ * q^n) (h₀ : a₁ * (q - 1) < 0) (h₁ : q > 0) :
  ∀ n, a (n + 1) < a n := 
sorry

end geom_seq_decreasing_l62_62003


namespace Tori_current_height_l62_62084

   -- Define the original height and the height she grew
   def Tori_original_height : Real := 4.4
   def Tori_growth : Real := 2.86

   -- Prove that Tori's current height is 7.26 feet
   theorem Tori_current_height : Tori_original_height + Tori_growth = 7.26 := by
     sorry
   
end Tori_current_height_l62_62084


namespace range_of_m_for_distinct_real_roots_l62_62464

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4 * x1 - m = 0 ∧ x2^2 - 4 * x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_l62_62464


namespace jade_transactions_l62_62217

theorem jade_transactions 
    (mabel_transactions : ℕ)
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = cal_transactions + 14) : 
    jade_transactions = 80 :=
sorry

end jade_transactions_l62_62217


namespace company_employee_percentage_l62_62372

theorem company_employee_percentage (M : ℝ)
  (h1 : 0.20 * M + 0.40 * (1 - M) = 0.31000000000000007) :
  M = 0.45 :=
sorry

end company_employee_percentage_l62_62372


namespace brass_players_count_l62_62831

def marching_band_size : ℕ := 110
def woodwinds (b : ℕ) : ℕ := 2 * b
def percussion (w : ℕ) : ℕ := 4 * w
def total_members (b : ℕ) : ℕ := b + woodwinds b + percussion (woodwinds b)

theorem brass_players_count : ∃ b : ℕ, total_members b = marching_band_size ∧ b = 10 :=
by
  sorry

end brass_players_count_l62_62831


namespace find_ff_half_l62_62711

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x + 1 else -x + 3

theorem find_ff_half : f (f (1 / 2)) = 3 / 2 := 
by 
  sorry

end find_ff_half_l62_62711


namespace coefficient_of_xy6_eq_one_l62_62417

theorem coefficient_of_xy6_eq_one (a : ℚ) (h : (7 : ℚ) * a = 1) : a = 1 / 7 :=
by sorry

end coefficient_of_xy6_eq_one_l62_62417


namespace product_identity_l62_62931

variable (x y : ℝ)

theorem product_identity :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l62_62931


namespace find_k_l62_62238

theorem find_k 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 4 * x + 2)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 8)
  (intersect : ∃ x y : ℝ, x = -2 ∧ y = -6 ∧ 4 * x + 2 = y ∧ k * x - 8 = y) :
  k = -1 := 
sorry

end find_k_l62_62238


namespace find_m_value_l62_62115

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables (m : ℝ)
variables (A B C D : V)

-- Assuming vectors a and b are non-collinear
axiom non_collinear (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (∃ (k : ℝ), a = k • b)

-- Given vectors
axiom hAB : B - A = 9 • a + m • b
axiom hBC : C - B = -2 • a - 1 • b
axiom hDC : C - D = a - 2 • b

-- Collinearity condition for A, B, and D
axiom collinear (k : ℝ) : B - A = k • (B - D)

theorem find_m_value : m = -3 :=
by sorry

end find_m_value_l62_62115


namespace slower_speed_percentage_l62_62017

theorem slower_speed_percentage (S S' T T' D : ℝ) (h1 : T = 8) (h2 : T' = T + 24) (h3 : D = S * T) (h4 : D = S' * T') : 
  (S' / S) * 100 = 25 := by
  sorry

end slower_speed_percentage_l62_62017


namespace carol_mike_equal_savings_weeks_l62_62184

theorem carol_mike_equal_savings_weeks :
  ∃ x : ℕ, (60 + 9 * x = 90 + 3 * x) ↔ x = 5 := 
by
  sorry

end carol_mike_equal_savings_weeks_l62_62184


namespace union_M_N_l62_62795

def M := {x : ℝ | x^2 - 4*x + 3 ≤ 0}
def N := {x : ℝ | Real.log x / Real.log 2 ≤ 1}

theorem union_M_N :
  M ∪ N = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end union_M_N_l62_62795


namespace total_time_pushing_car_l62_62253

theorem total_time_pushing_car :
  let d1 := 3
  let s1 := 6
  let d2 := 3
  let s2 := 3
  let d3 := 4
  let s3 := 8
  let t1 := d1 / s1
  let t2 := d2 / s2
  let t3 := d3 / s3
  (t1 + t2 + t3) = 2 :=
by
  sorry

end total_time_pushing_car_l62_62253


namespace side_of_beef_weight_after_processing_l62_62998

theorem side_of_beef_weight_after_processing (initial_weight : ℝ) (lost_percentage : ℝ) (final_weight : ℝ) 
  (h1 : initial_weight = 400) 
  (h2 : lost_percentage = 0.4) 
  (h3 : final_weight = initial_weight * (1 - lost_percentage)) : 
  final_weight = 240 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end side_of_beef_weight_after_processing_l62_62998


namespace find_a_l62_62376

theorem find_a (a : ℝ) (h : ∀ x y : ℝ, ax + y - 4 = 0 → x + (a + 3/2) * y + 2 = 0 → True) : a = 1/2 :=
sorry

end find_a_l62_62376


namespace ratio_of_linear_combination_l62_62850

theorem ratio_of_linear_combination (a b x y : ℝ) (hb : b ≠ 0) 
  (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) :
  a / b = -2 / 5 :=
by {
  sorry
}

end ratio_of_linear_combination_l62_62850


namespace sum_a_b_l62_62730

theorem sum_a_b (a b : ℝ) (h₁ : 2 = a + b) (h₂ : 6 = a + b / 9) : a + b = 2 :=
by
  sorry

end sum_a_b_l62_62730


namespace value_of_c_l62_62292

noncomputable def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem value_of_c (a b c : ℤ) (ha: a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0)
  (hfa: f a a b c = a^3) (hfb: f b a b c = b^3) : c = 16 := by
    sorry

end value_of_c_l62_62292


namespace circumcircle_radius_min_cosA_l62_62242

noncomputable def circumcircle_radius (a b c : ℝ) (A B C : ℝ) :=
  a / (2 * (Real.sin A))

theorem circumcircle_radius_min_cosA
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : Real.sin C + Real.sin B = 4 * Real.sin A)
  (h3 : a^2 + b^2 - 2 * a * b * (Real.cos A) = c^2)
  (h4 : a^2 + c^2 - 2 * a * c * (Real.cos B) = b^2)
  (h5 : b^2 + c^2 - 2 * b * c * (Real.cos C) = a^2) :
  circumcircle_radius a b c A B C = 8 * Real.sqrt 15 / 15 :=
sorry

end circumcircle_radius_min_cosA_l62_62242


namespace emma_final_balance_correct_l62_62078

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l62_62078


namespace probability_john_david_chosen_l62_62313

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem probability_john_david_chosen :
  let total_workers := 6
  let choose_two := choose total_workers 2
  let favorable_outcomes := 1
  choose_two = 15 → (favorable_outcomes / choose_two : ℝ) = 1 / 15 :=
by
  intros
  sorry

end probability_john_david_chosen_l62_62313


namespace part1_general_formula_part2_sum_S_l62_62588

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => a n + 1

theorem part1_general_formula (n : ℕ) : a n = n + 1 := by
  sorry

noncomputable def b (n : ℕ) : ℝ := 1 / (↑n * ↑(n + 2))

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b (i + 1))

theorem part2_sum_S (n : ℕ) : 
  S n = (1/2) * ((3/2) - (1 / (n + 1)) - (1 / (n + 2))) := by
  sorry

end part1_general_formula_part2_sum_S_l62_62588


namespace total_beds_in_hotel_l62_62624

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end total_beds_in_hotel_l62_62624


namespace pages_per_day_difference_l62_62319

theorem pages_per_day_difference :
  let songhee_pages := 288
  let songhee_days := 12
  let eunju_pages := 243
  let eunju_days := 9
  let songhee_per_day := songhee_pages / songhee_days
  let eunju_per_day := eunju_pages / eunju_days
  eunju_per_day - songhee_per_day = 3 := by
  sorry

end pages_per_day_difference_l62_62319


namespace arithmetic_square_root_of_sqrt_16_l62_62285

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l62_62285


namespace find_speed_l62_62329

theorem find_speed (v : ℝ) (t : ℝ) (h : t = 5 * v^2) (ht : t = 20) : v = 2 :=
by
  sorry

end find_speed_l62_62329


namespace oil_cylinder_capacity_l62_62700

theorem oil_cylinder_capacity
  (C : ℚ) -- total capacity of the cylinder, given as a rational number
  (h1 : 3 / 4 * C + 4 = 4 / 5 * C) -- equation representing the condition of initial and final amounts of oil in the cylinder
  : C = 80 := -- desired result showing the total capacity

sorry

end oil_cylinder_capacity_l62_62700


namespace minute_hand_rotation_l62_62007

theorem minute_hand_rotation (minutes : ℕ) (degrees_per_minute : ℝ) (radian_conversion_factor : ℝ) : 
  minutes = 10 → 
  degrees_per_minute = 360 / 60 → 
  radian_conversion_factor = π / 180 → 
  (-(degrees_per_minute * minutes * radian_conversion_factor) = -(π / 3)) := 
by
  intros hminutes hdegrees hfactor
  rw [hminutes, hdegrees, hfactor]
  simp
  sorry

end minute_hand_rotation_l62_62007


namespace molecular_weight_N2O5_l62_62225

variable {x : ℕ}

theorem molecular_weight_N2O5 (hx : 10 * 108 = 1080) : (108 * x = 1080 * x / 10) :=
by
  sorry

end molecular_weight_N2O5_l62_62225


namespace find_pairs_l62_62995

def is_prime (p : ℕ) : Prop := (p ≥ 2) ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_pairs (a p : ℕ) (h_pos_a : a > 0) (h_prime_p : is_prime p) :
  (∀ m n : ℕ, 0 < m → 0 < n → (a ^ (2 ^ n) % p ^ n = a ^ (2 ^ m) % p ^ m ∧ a ^ (2 ^ n) % p ^ n ≠ 0))
  ↔ (∃ k : ℕ, a = 2 * k + 1 ∧ p = 2) :=
sorry

end find_pairs_l62_62995


namespace find_rho_squared_l62_62054

theorem find_rho_squared:
  ∀ (a b : ℝ), (0 < a) → (0 < b) →
  (a^2 - 2 * b^2 = 0) →
  (∃ (x y : ℝ), 
    (0 ≤ x ∧ x < a) ∧ 
    (0 ≤ y ∧ y < b) ∧ 
    (a^2 + y^2 = b^2 + x^2) ∧ 
    ((a - x)^2 + (b - y)^2 = b^2 + x^2) ∧ 
    (x^2 + y^2 = b^2)) → 
  (∃ (ρ : ℝ), ρ = a / b ∧ ρ^2 = 2) :=
by
  intros a b ha hb hab hsol
  sorry  -- Proof to be provided later

end find_rho_squared_l62_62054


namespace necessary_and_sufficient_condition_l62_62841

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (a > b) ↔ (a - 1/a > b - 1/b) :=
sorry

end necessary_and_sufficient_condition_l62_62841


namespace total_number_of_sheep_l62_62970

theorem total_number_of_sheep (a₁ a₂ a₃ a₄ a₅ a₆ a₇ d : ℤ)
    (h1 : a₂ = a₁ + d)
    (h2 : a₃ = a₁ + 2 * d)
    (h3 : a₄ = a₁ + 3 * d)
    (h4 : a₅ = a₁ + 4 * d)
    (h5 : a₆ = a₁ + 5 * d)
    (h6 : a₇ = a₁ + 6 * d)
    (h_sum : a₁ + a₂ + a₃ = 33)
    (h_seven: 2 * a₂ + 9 = a₇) :
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 133 := sorry

end total_number_of_sheep_l62_62970


namespace find_a10_l62_62213

-- Conditions
variables (S : ℕ → ℕ) (a : ℕ → ℕ)
variables (hS9 : S 9 = 81) (ha2 : a 2 = 3)

-- Arithmetic sequence sum definition
def arithmetic_sequence_sum (n : ℕ) (a1 : ℕ) (d : ℕ) :=
  n * (2 * a1 + (n - 1) * d) / 2

-- a_n formula definition
def a_n (n a1 d : ℕ) := a1 + (n - 1) * d

-- Proof statement
theorem find_a10 (a1 d : ℕ) (hS9' : 9 * (2 * a1 + 8 * d) / 2 = 81) (ha2' : a1 + d = 3) :
  a 10 = a1 + 9 * d :=
sorry

end find_a10_l62_62213


namespace Juwella_reads_pages_l62_62090

theorem Juwella_reads_pages (p1 p2 p3 p_total p_tonight : ℕ) 
                            (h1 : p1 = 15)
                            (h2 : p2 = 2 * p1)
                            (h3 : p3 = p2 + 5)
                            (h4 : p_total = 100) 
                            (h5 : p_total = p1 + p2 + p3 + p_tonight) :
  p_tonight = 20 := 
sorry

end Juwella_reads_pages_l62_62090


namespace EDTA_Ca2_complex_weight_l62_62497

-- Definitions of atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Ca : ℝ := 40.08

-- Number of atoms in EDTA
def num_atoms_C : ℝ := 10
def num_atoms_H : ℝ := 16
def num_atoms_N : ℝ := 2
def num_atoms_O : ℝ := 8

-- Molecular weight of EDTA
def molecular_weight_EDTA : ℝ :=
  num_atoms_C * atomic_weight_C +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N +
  num_atoms_O * atomic_weight_O

-- Proof that the molecular weight of the complex is 332.328 g/mol
theorem EDTA_Ca2_complex_weight : molecular_weight_EDTA + atomic_weight_Ca = 332.328 := by
  sorry

end EDTA_Ca2_complex_weight_l62_62497


namespace team_OT_matches_l62_62713

variable (T x M: Nat)

-- Condition: Team C played T matches in the first week.
def team_C_matches_T : Nat := T

-- Condition: Team C played x matches in the first week.
def team_C_matches_x : Nat := x

-- Condition: Team O played M matches in the first week.
def team_O_matches_M : Nat := M

-- Condition: Team C has not played against Team A.
axiom C_not_played_A : ¬ (team_C_matches_T = team_C_matches_x)

-- Condition: Team B has not played against a specified team (interpreted).
axiom B_not_played_specified : ∀ x, ¬ (team_C_matches_x = x)

-- The proof for the number of matches played by team \(\overrightarrow{OT}\).
theorem team_OT_matches : T = 4 := 
    sorry

end team_OT_matches_l62_62713


namespace total_amount_invested_l62_62064

-- Define the conditions and specify the correct answer
theorem total_amount_invested (x y : ℝ) (h8 : y = 600) 
  (h_income_diff : 0.10 * (x - 600) - 0.08 * 600 = 92) : 
  x + y = 2000 := sorry

end total_amount_invested_l62_62064


namespace sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l62_62723

variable {a b : ℝ}

theorem sufficient_condition_for_reciprocal_square :
  (b > a ∧ a > 0) → (1 / a^2 > 1 / b^2) :=
sorry

theorem not_necessary_condition_for_reciprocal_square :
  ¬((1 / a^2 > 1 / b^2) → (b > a ∧ a > 0)) :=
sorry

end sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l62_62723


namespace sum_of_abs_values_eq_12_l62_62286

theorem sum_of_abs_values_eq_12 (a b c d : ℝ) (h : 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) :
  abs a + abs b + abs c + abs d = 12 := sorry

end sum_of_abs_values_eq_12_l62_62286


namespace prime_pairs_l62_62898

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, 2 ≤ m → m ≤ n / 2 → n % m ≠ 0

theorem prime_pairs :
  ∀ (p q : ℕ), is_prime p → is_prime q →
  1 < p → p < 100 →
  1 < q → q < 100 →
  is_prime (p + 6) →
  is_prime (p + 10) →
  is_prime (q + 4) →
  is_prime (q + 10) →
  is_prime (p + q + 1) →
  (p, q) = (7, 3) ∨ (p, q) = (13, 3) ∨ (p, q) = (37, 3) ∨ (p, q) = (97, 3) :=
by
  sorry

end prime_pairs_l62_62898


namespace find_y_z_l62_62369

theorem find_y_z (y z : ℝ) : 
  (∃ k : ℝ, (1:ℝ) = -k ∧ (2:ℝ) = k * y ∧ (3:ℝ) = k * z) → y = -2 ∧ z = -3 :=
by
  sorry

end find_y_z_l62_62369


namespace interest_earned_l62_62593

-- Define the principal, interest rate, and number of years
def principal : ℝ := 1200
def annualInterestRate : ℝ := 0.12
def numberOfYears : ℕ := 4

-- Define the compound interest formula
def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the total interest earned
def totalInterest (P A : ℝ) : ℝ :=
  A - P

-- State the theorem
theorem interest_earned :
  totalInterest principal (compoundInterest principal annualInterestRate numberOfYears) = 688.224 :=
by
  sorry

end interest_earned_l62_62593


namespace value_at_2_l62_62958

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

theorem value_at_2 : f 2 = 0 := by
  sorry

end value_at_2_l62_62958


namespace bug_final_position_after_2023_jumps_l62_62130

open Nat

def bug_jump (pos : Nat) : Nat :=
  if pos % 2 = 1 then (pos + 2) % 6 else (pos + 1) % 6

noncomputable def final_position (n : Nat) : Nat :=
  (iterate bug_jump n 6) % 6

theorem bug_final_position_after_2023_jumps : final_position 2023 = 1 := by
  sorry

end bug_final_position_after_2023_jumps_l62_62130


namespace least_possible_b_l62_62175

theorem least_possible_b (a b : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (h_a_factors : ∃ k, a = p^k ∧ k + 1 = 3) (h_b_factors : ∃ m, b = p^m ∧ m + 1 = a) (h_divisible : b % a = 0) : 
  b = 8 := 
by 
  sorry

end least_possible_b_l62_62175


namespace solving_linear_equations_count_l62_62261

def total_problems : ℕ := 140
def algebra_percentage : ℝ := 0.40
def algebra_problems := (total_problems : ℝ) * algebra_percentage
def solving_linear_equations_percentage : ℝ := 0.50
def solving_linear_equations_problems := algebra_problems * solving_linear_equations_percentage

theorem solving_linear_equations_count :
  solving_linear_equations_problems = 28 :=
by
  sorry

end solving_linear_equations_count_l62_62261


namespace div_equal_octagons_l62_62453

-- Definitions based on the conditions
def squareArea (n : ℕ) := n * n
def isDivisor (m n : ℕ) := n % m = 0

-- Main statement
theorem div_equal_octagons (n : ℕ) (hn : n = 8) :
  (2 ∣ squareArea n) ∨ (4 ∣ squareArea n) ∨ (8 ∣ squareArea n) ∨ (16 ∣ squareArea n) :=
by
  -- We shall show the divisibility aspect later.
  sorry

end div_equal_octagons_l62_62453


namespace minimum_value_of_function_l62_62601

theorem minimum_value_of_function :
  ∃ x y : ℝ, 2 * x ^ 2 + 3 * x * y + 4 * y ^ 2 - 8 * x + y = 3.7391 := by
  sorry

end minimum_value_of_function_l62_62601


namespace perpendicular_tangents_at_x0_l62_62113

noncomputable def x0 := (36 : ℝ)^(1 / 3) / 6

theorem perpendicular_tangents_at_x0 :
  (∃ x0 : ℝ, (∃ f1 f2 : ℝ → ℝ,
    (∀ x, f1 x = x^2 - 1) ∧
    (∀ x, f2 x = 1 - x^3) ∧
    (2 * x0 * (-3 * x0^2) = -1)) ∧
    x0 = (36 : ℝ)^(1 / 3) / 6) := sorry

end perpendicular_tangents_at_x0_l62_62113


namespace blue_string_length_is_320_l62_62799

-- Define the lengths of the strings
def red_string_length := 8
def white_string_length := 5 * red_string_length
def blue_string_length := 8 * white_string_length

-- The main theorem to prove
theorem blue_string_length_is_320 : blue_string_length = 320 := by
  sorry

end blue_string_length_is_320_l62_62799


namespace minimum_value_of_polynomial_l62_62516

def polynomial (a b : ℝ) : ℝ := 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999

theorem minimum_value_of_polynomial : ∃ (a b : ℝ), polynomial a b = 1947 :=
by
  sorry

end minimum_value_of_polynomial_l62_62516


namespace lines_are_perpendicular_l62_62322

-- Define the first line equation
def line1 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x - y + 3 = 0

-- Definition to determine the perpendicularity of two lines
def are_perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem lines_are_perpendicular :
  are_perpendicular (-1) (1) := 
by
  sorry

end lines_are_perpendicular_l62_62322


namespace part1_part2_l62_62889

variable (m : ℝ)

def p (m : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x - 2 ≥ m^2 - 3 * m
def q (m : ℝ) : Prop := ∃ x0 : ℝ, -1 ≤ x0 ∧ x0 ≤ 1 ∧ m ≤ x0

theorem part1 (h : p m) : 1 ≤ m ∧ m ≤ 2 := sorry

theorem part2 (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m < 1) ∨ (1 < m ∧ m ≤ 2) := sorry

end part1_part2_l62_62889


namespace avg_problem_l62_62053

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Formulate the proof problem statement
theorem avg_problem : avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end avg_problem_l62_62053


namespace solve_for_x_l62_62120

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 3029) : x = 200.4 :=
by
  sorry

end solve_for_x_l62_62120


namespace least_total_acorns_l62_62378

theorem least_total_acorns :
  ∃ a₁ a₂ a₃ : ℕ,
    (∀ k : ℕ, (∃ a₁ a₂ a₃ : ℕ,
      (2 * a₁ / 3 + a₁ % 3 / 3 + a₂ + a₃ / 9) % 6 = 4 * k ∧
      (a₁ / 6 + a₂ / 3 + a₃ / 3 + 8 * a₃ / 18) % 6 = 3 * k ∧
      (a₁ / 6 + 5 * a₂ / 6 + a₃ / 9) % 6 = 2 * k) → k = 630) ∧
    (a₁ + a₂ + a₃) = 630 :=
sorry

end least_total_acorns_l62_62378


namespace max_popsicles_with_10_dollars_l62_62646

theorem max_popsicles_with_10_dollars :
  (∃ (single_popsicle_cost : ℕ) (four_popsicle_box_cost : ℕ) (six_popsicle_box_cost : ℕ) (budget : ℕ),
    single_popsicle_cost = 1 ∧
    four_popsicle_box_cost = 3 ∧
    six_popsicle_box_cost = 4 ∧
    budget = 10 ∧
    ∃ (max_popsicles : ℕ),
      max_popsicles = 14 ∧
      ∀ (popsicles : ℕ),
        popsicles ≤ 14 →
        ∃ (x y z : ℕ),
          popsicles = x + 4*y + 6*z ∧
          x * single_popsicle_cost + y * four_popsicle_box_cost + z * six_popsicle_box_cost ≤ budget
  ) :=
sorry

end max_popsicles_with_10_dollars_l62_62646


namespace sqrt_sum_simplification_l62_62808

theorem sqrt_sum_simplification : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) = 10 := by
  sorry

end sqrt_sum_simplification_l62_62808


namespace find_possible_m_values_l62_62360

theorem find_possible_m_values (m : ℕ) (a : ℕ) (h₀ : m > 1) (h₁ : m * a + (m * (m - 1) / 2) = 33) :
  m = 2 ∨ m = 3 ∨ m = 6 :=
by
  sorry

end find_possible_m_values_l62_62360


namespace factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l62_62293

variable {α : Type*} [CommRing α]

-- Problem 1
theorem factorize_2x2_minus_8 (x : α) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorize_ax2_minus_2ax_plus_a (a x : α) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
sorry

end factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l62_62293


namespace find_interest_rate_l62_62722

-- Defining the conditions
def P : ℝ := 5000
def A : ℝ := 5302.98
def t : ℝ := 1.5
def n : ℕ := 2

-- Statement of the problem in Lean 4
theorem find_interest_rate (P A t : ℝ) (n : ℕ) (hP : P = 5000) (hA : A = 5302.98) (ht : t = 1.5) (hn : n = 2) : 
  ∃ r : ℝ, r * 100 = 3.96 :=
sorry

end find_interest_rate_l62_62722


namespace inscribed_squares_ratio_l62_62375

theorem inscribed_squares_ratio (a b : ℝ) (h_triangle : 5^2 + 12^2 = 13^2)
    (h_square1 : a = 25 / 37) (h_square2 : b = 10) :
    a / b = 25 / 370 :=
by 
  sorry

end inscribed_squares_ratio_l62_62375


namespace graphs_intersect_once_l62_62357

theorem graphs_intersect_once : 
  ∃! (x : ℝ), |3 * x + 6| = -|4 * x - 3| :=
sorry

end graphs_intersect_once_l62_62357


namespace mike_taller_than_mark_l62_62924

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l62_62924


namespace count_squares_below_graph_l62_62914

theorem count_squares_below_graph (x y: ℕ) (h_eq : 12 * x + 180 * y = 2160) (h_first_quadrant : x ≥ 0 ∧ y ≥ 0) :
  let total_squares := 180 * 12
  let diagonal_squares := 191
  let below_squares := total_squares - diagonal_squares
  below_squares = 1969 :=
by
  sorry

end count_squares_below_graph_l62_62914


namespace percent_of_a_is_4b_l62_62835

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b / a) * 100 = 333.33 := by
  sorry

end percent_of_a_is_4b_l62_62835


namespace move_digit_produces_ratio_l62_62797

theorem move_digit_produces_ratio
  (a b : ℕ)
  (h_original_eq : ∃ x : ℕ, x = 10 * a + b)
  (h_new_eq : ∀ (n : ℕ), 10^n * b + a = (3 * (10 * a + b)) / 2):
  285714 = 10 * a + b :=
by
  -- proof steps would go here
  sorry

end move_digit_produces_ratio_l62_62797


namespace nail_polish_count_l62_62816

-- Definitions from conditions
def K : ℕ := 25
def H : ℕ := K + 8
def Ka : ℕ := K - 6
def L : ℕ := 2 * K
def S : ℕ := 13 + 10  -- Since 25 / 2 = 12.5, rounded to 13 for practical purposes

-- Statement to prove
def T : ℕ := H + Ka + L + S

theorem nail_polish_count : T = 125 := by
  sorry

end nail_polish_count_l62_62816


namespace twice_joan_more_than_karl_l62_62606

-- Define the conditions
def J : ℕ := 158
def total : ℕ := 400
def K : ℕ := total - J

-- Define the theorem to be proven
theorem twice_joan_more_than_karl :
  2 * J - K = 74 := by
    -- Skip the proof steps using 'sorry'
    sorry

end twice_joan_more_than_karl_l62_62606


namespace range_f_iff_l62_62548

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log ((m^2 - 3 * m + 2) * x^2 + 2 * (m - 1) * x + 5)

theorem range_f_iff (m : ℝ) :
  (∀ y ∈ Set.univ, ∃ x, f m x = y) ↔ (m = 1 ∨ (2 < m ∧ m ≤ 9/4)) := 
by
  sorry

end range_f_iff_l62_62548


namespace trigonometric_identities_l62_62583

open Real

theorem trigonometric_identities :
  (cos 75 * cos 75 = (2 - sqrt 3) / 4) ∧
  ((1 + tan 105) / (1 - tan 105) ≠ sqrt 3 / 3) ∧
  (tan 1 + tan 44 + tan 1 * tan 44 = 1) ∧
  (sin 70 * (sqrt 3 / tan 40 - 1) ≠ 2) :=
by
  sorry

end trigonometric_identities_l62_62583


namespace cross_out_number_l62_62032

theorem cross_out_number (n : ℤ) (h1 : 5 * n + 10 = 10085) : n = 2015 → (n + 5 = 2020) :=
by
  sorry

end cross_out_number_l62_62032


namespace greatest_three_digit_number_divisible_by_3_6_5_l62_62491

theorem greatest_three_digit_number_divisible_by_3_6_5 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 3 = 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 3 = 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → m ≤ n) ∧ n = 990 := 
by
  sorry

end greatest_three_digit_number_divisible_by_3_6_5_l62_62491


namespace trig_identity_l62_62416

open Real

theorem trig_identity (α : ℝ) (hα : α > -π ∧ α < -π/2) :
  (sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α))) = - 2 / tan α :=
by
  sorry

end trig_identity_l62_62416


namespace two_digit_factors_of_3_18_minus_1_l62_62086

theorem two_digit_factors_of_3_18_minus_1 : ∃ n : ℕ, n = 6 ∧ 
  ∀ x, x ∈ {y : ℕ | y ∣ 3^18 - 1 ∧ y > 9 ∧ y < 100} → 
  (x = 13 ∨ x = 26 ∨ x = 52 ∨ x = 14 ∨ x = 28 ∨ x = 91) :=
by
  use 6
  sorry

end two_digit_factors_of_3_18_minus_1_l62_62086


namespace differences_occur_10_times_l62_62160

variable (a : Fin 45 → Nat)

theorem differences_occur_10_times 
    (h : ∀ i j : Fin 44, i < j → a i < a j)
    (h_lt_125 : ∀ i : Fin 44, a i < 125) :
    ∃ i : Fin 43, ∃ j : Fin 43, i ≠ j ∧ (a (i + 1) - a i) = (a (j + 1) - a j) ∧ 
    (∃ k : Nat, k ≥ 10 ∧ (a (j + 1) - a j) = (a (k + 1) - a k)) :=
sorry

end differences_occur_10_times_l62_62160


namespace decaf_percentage_total_l62_62275

-- Defining the initial conditions
def initial_stock : ℝ := 400
def initial_decaf_percentage : ℝ := 0.30
def new_stock : ℝ := 100
def new_decaf_percentage : ℝ := 0.60

-- Given conditions
def amount_initial_decaf := initial_decaf_percentage * initial_stock
def amount_new_decaf := new_decaf_percentage * new_stock
def total_decaf := amount_initial_decaf + amount_new_decaf
def total_stock := initial_stock + new_stock

-- Prove the percentage of decaffeinated coffee in the total stock
theorem decaf_percentage_total : 
  (total_decaf / total_stock) * 100 = 36 := by
  sorry

end decaf_percentage_total_l62_62275


namespace hoseok_wire_length_l62_62637

theorem hoseok_wire_length (side_length : ℕ) (equilateral : Prop) (leftover_wire : ℕ) (total_wire : ℕ)  
  (eq_side : side_length = 19) (eq_leftover : leftover_wire = 15) 
  (eq_equilateral : equilateral) : total_wire = 72 :=
sorry

end hoseok_wire_length_l62_62637


namespace prob_square_l62_62792

def total_figures := 10
def num_squares := 3
def num_circles := 4
def num_triangles := 3

theorem prob_square : (num_squares : ℚ) / total_figures = 3 / 10 :=
by
  rw [total_figures, num_squares]
  exact sorry

end prob_square_l62_62792


namespace parabola_focus_distance_l62_62662

theorem parabola_focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ := (1, 0)) 
    (h_parabola : A.2^2 = 4 * A.1) (h_distance : dist A F = 3) :
    A = (2, 2 * Real.sqrt 2) ∨ A = (2, -2 * Real.sqrt 2) :=
by
  sorry

end parabola_focus_distance_l62_62662


namespace probability_of_odd_divisor_l62_62592

noncomputable def prime_factorization_15! : ℕ :=
  (2 ^ 11) * (3 ^ 6) * (5 ^ 3) * (7 ^ 2) * 11 * 13

def total_factors_15! : ℕ :=
  (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def odd_factors_15! : ℕ :=
  (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def probability_odd_divisor_15! : ℚ :=
  odd_factors_15! / total_factors_15!

theorem probability_of_odd_divisor : probability_odd_divisor_15! = 1 / 12 :=
by
  sorry

end probability_of_odd_divisor_l62_62592


namespace length_DE_l62_62368

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l62_62368


namespace numerical_puzzle_unique_solution_l62_62552

theorem numerical_puzzle_unique_solution :
  ∃ (A X Y P : ℕ), 
    A ≠ X ∧ A ≠ Y ∧ A ≠ P ∧ X ≠ Y ∧ X ≠ P ∧ Y ≠ P ∧
    (A * 10 + X) + (Y * 10 + X) = Y * 100 + P * 10 + A ∧
    A = 8 ∧ X = 9 ∧ Y = 1 ∧ P = 0 :=
sorry

end numerical_puzzle_unique_solution_l62_62552


namespace embankment_building_l62_62669

theorem embankment_building (days : ℕ) (workers_initial : ℕ) (workers_later : ℕ) (embankments : ℕ) :
  workers_initial = 75 → days = 4 → embankments = 2 →
  (∀ r : ℚ, embankments = workers_initial * r * days →
            embankments = workers_later * r * 5) :=
by
  intros h75 hd4 h2 r hr
  sorry

end embankment_building_l62_62669


namespace equiangular_polygons_unique_solution_l62_62181

theorem equiangular_polygons_unique_solution :
  ∃! (n1 n2 : ℕ), (n1 ≠ 0 ∧ n2 ≠ 0) ∧ (180 / n1 + 360 / n2 = 90) :=
by
  sorry

end equiangular_polygons_unique_solution_l62_62181


namespace find_element_atomic_mass_l62_62626

-- Define the atomic mass of bromine
def atomic_mass_br : ℝ := 79.904

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 267

-- Define the number of bromine atoms in the compound (assuming n = 1)
def n : ℕ := 1

-- Define the atomic mass of the unknown element X
def atomic_mass_x : ℝ := molecular_weight - n * atomic_mass_br

-- State the theorem to prove
theorem find_element_atomic_mass : atomic_mass_x = 187.096 :=
by
  -- placeholder for the proof
  sorry

end find_element_atomic_mass_l62_62626


namespace fraction_of_network_advertisers_l62_62969

theorem fraction_of_network_advertisers 
  (total_advertisers : ℕ := 20) 
  (percentage_from_uni_a : ℝ := 0.75)
  (advertisers_from_uni_a := total_advertisers * percentage_from_uni_a) :
  (advertisers_from_uni_a / total_advertisers) = (3 / 4) :=
by
  sorry

end fraction_of_network_advertisers_l62_62969


namespace sum_of_polynomials_l62_62321

open Polynomial

noncomputable def f : ℚ[X] := -4 * X^2 + 2 * X - 5
noncomputable def g : ℚ[X] := -6 * X^2 + 4 * X - 9
noncomputable def h : ℚ[X] := 6 * X^2 + 6 * X + 2

theorem sum_of_polynomials :
  f + g + h = -4 * X^2 + 12 * X - 12 :=
by sorry

end sum_of_polynomials_l62_62321


namespace tree_growth_period_l62_62107

theorem tree_growth_period (initial height growth_rate : ℕ) (H4 final_height years : ℕ) 
  (h_init : initial_height = 4) 
  (h_growth_rate : growth_rate = 1) 
  (h_H4 : H4 = initial_height + 4 * growth_rate)
  (h_final_height : final_height = H4 + H4 / 4) 
  (h_years : years = (final_height - initial_height) / growth_rate) :
  years = 6 :=
by
  sorry

end tree_growth_period_l62_62107


namespace target_hit_probability_l62_62123

-- Define the probabilities given in the problem
def prob_A_hits : ℚ := 9 / 10
def prob_B_hits : ℚ := 8 / 9

-- The required probability that at least one hits the target
def prob_target_hit : ℚ := 89 / 90

-- Theorem stating that the probability calculated matches the expected outcome
theorem target_hit_probability :
  1 - ((1 - prob_A_hits) * (1 - prob_B_hits)) = prob_target_hit :=
by
  sorry

end target_hit_probability_l62_62123


namespace roses_carnations_price_comparison_l62_62978

variables (x y : ℝ)

theorem roses_carnations_price_comparison
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y :=
sorry

end roses_carnations_price_comparison_l62_62978


namespace geometric_sequence_a7_eq_64_l62_62717

open Nat

theorem geometric_sequence_a7_eq_64 (a : ℕ → ℕ) (h1 : a 1 = 1) (hrec : ∀ n : ℕ, a (n + 1) = 2 * a n) : a 7 = 64 := by
  sorry

end geometric_sequence_a7_eq_64_l62_62717


namespace rational_expression_iff_rational_square_l62_62348

theorem rational_expression_iff_rational_square (x : ℝ) :
  (∃ r : ℚ, x^2 + (Real.sqrt (x^4 + 1)) - 1 / (x^2 + (Real.sqrt (x^4 + 1))) = r) ↔
  (∃ q : ℚ, x^2 = q) := by
  sorry

end rational_expression_iff_rational_square_l62_62348


namespace total_feathers_needed_l62_62077

theorem total_feathers_needed 
  (animals_group1 : ℕ) (feathers_group1 : ℕ)
  (animals_group2 : ℕ) (feathers_group2 : ℕ) 
  (total_feathers : ℕ) :
  animals_group1 = 934 →
  feathers_group1 = 7 →
  animals_group2 = 425 →
  feathers_group2 = 12 →
  total_feathers = 11638 :=
by sorry

end total_feathers_needed_l62_62077


namespace inclination_angle_tan_60_perpendicular_l62_62872

/-
The inclination angle of the line given by x = tan(60 degrees) is 90 degrees.
-/
theorem inclination_angle_tan_60_perpendicular : 
  ∀ (x : ℝ), x = Real.tan (60 *Real.pi / 180) → 
  ∃ θ : ℝ, θ = 90 :=
sorry

end inclination_angle_tan_60_perpendicular_l62_62872


namespace fraction_to_decimal_l62_62809

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 :=
by
  sorry

end fraction_to_decimal_l62_62809


namespace find_q_l62_62569

theorem find_q (a b m p q : ℚ) (h1 : a * b = 3) (h2 : a + b = m) 
  (h3 : (a + 1/b) * (b + 1/a) = q) : 
  q = 13 / 3 := by
  sorry

end find_q_l62_62569


namespace total_bill_is_89_l62_62258

-- Define the individual costs and quantities
def adult_meal_cost := 12
def child_meal_cost := 7
def fries_cost := 5
def drink_cost := 10

def num_adults := 4
def num_children := 3
def num_fries := 2
def num_drinks := 1

-- Calculate the total bill
def total_bill : Nat :=
  (num_adults * adult_meal_cost) + 
  (num_children * child_meal_cost) + 
  (num_fries * fries_cost) + 
  (num_drinks * drink_cost)

-- The proof statement
theorem total_bill_is_89 : total_bill = 89 := 
  by
  -- The proof will be provided here
  sorry

end total_bill_is_89_l62_62258


namespace real_solution_to_abs_equation_l62_62479

theorem real_solution_to_abs_equation :
  (∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8|) :=
by
  sorry

end real_solution_to_abs_equation_l62_62479


namespace volume_ratio_l62_62335

noncomputable def V_D (s : ℝ) := (15 + 7 * Real.sqrt 5) * s^3 / 4
noncomputable def a (s : ℝ) := s / 2 * (1 + Real.sqrt 5)
noncomputable def V_I (a : ℝ) := 5 * (3 + Real.sqrt 5) * a^3 / 12

theorem volume_ratio (s : ℝ) (h₁ : 0 < s) :
  V_I (a s) / V_D s = (5 * (3 + Real.sqrt 5) * (1 + Real.sqrt 5)^3) / (12 * 2 * (15 + 7 * Real.sqrt 5)) :=
by
  sorry

end volume_ratio_l62_62335


namespace find_area_of_triangle_l62_62490

noncomputable def triangle_area (a b: ℝ) (cosC: ℝ) : ℝ :=
  let sinC := Real.sqrt (1 - cosC^2)
  0.5 * a * b * sinC

theorem find_area_of_triangle :
  ∀ (a b cosC : ℝ), a = 3 * Real.sqrt 2 → b = 2 * Real.sqrt 3 → cosC = 1 / 3 →
  triangle_area a b cosC = 4 * Real.sqrt 3 :=
by
  intros a b cosC ha hb hcosC
  rw [ha, hb, hcosC]
  sorry

end find_area_of_triangle_l62_62490


namespace sin_neg_225_eq_sqrt2_div2_l62_62117

theorem sin_neg_225_eq_sqrt2_div2 :
  Real.sin (-225 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_225_eq_sqrt2_div2_l62_62117


namespace number_of_truthful_warriors_l62_62148

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l62_62148


namespace tangent_slope_at_point_x_eq_1_l62_62764

noncomputable def curve (x : ℝ) : ℝ := x^3 - 4 * x
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - 4

theorem tangent_slope_at_point_x_eq_1 : curve_derivative 1 = -1 :=
by {
  -- This is just the theorem statement, no proof is required as per the instructions.
  sorry
}

end tangent_slope_at_point_x_eq_1_l62_62764


namespace part_I_part_II_l62_62063

theorem part_I (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2) (h4 : a + b ≤ m) : m ≥ 3 := by
  sorry

theorem part_II (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2)
  (h4 : 2 * |x - 1| + |x| ≥ a + b) : (x ≤ -1 / 3 ∨ x ≥ 5 / 3) := by
  sorry

end part_I_part_II_l62_62063


namespace equal_red_B_black_C_l62_62740

theorem equal_red_B_black_C (a : ℕ) (h_even : a % 2 = 0) :
  ∃ (x y k j l i : ℕ), x + y = a ∧ y + i + j = a ∧ i + k = y ∧ k + j = x ∧ i = k := 
  sorry

end equal_red_B_black_C_l62_62740


namespace chess_group_players_l62_62278

theorem chess_group_players (n : ℕ) (H : n * (n - 1) / 2 = 435) : n = 30 :=
by
  sorry

end chess_group_players_l62_62278


namespace solution_in_range_for_fraction_l62_62736

theorem solution_in_range_for_fraction (a : ℝ) : 
  (∃ x : ℝ, (2 * x + a) / (x + 1) = 1 ∧ x < 0) ↔ (a > 1 ∧ a ≠ 2) :=
by
  sorry

end solution_in_range_for_fraction_l62_62736


namespace school_club_profit_l62_62296

def calculate_profit (bars_bought : ℕ) (cost_per_3_bars : ℚ) (bars_sold : ℕ) (price_per_4_bars : ℚ) : ℚ :=
  let cost_per_bar := cost_per_3_bars / 3
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_4_bars / 4
  let total_revenue := bars_sold * price_per_bar
  total_revenue - total_cost

theorem school_club_profit :
  calculate_profit 1200 1.50 1200 2.40 = 120 :=
by sorry

end school_club_profit_l62_62296


namespace repeating_decimal_exceeds_decimal_representation_l62_62536

noncomputable def repeating_decimal : ℚ := 71 / 99
def decimal_representation : ℚ := 71 / 100

theorem repeating_decimal_exceeds_decimal_representation :
  repeating_decimal - decimal_representation = 71 / 9900 := by
  sorry

end repeating_decimal_exceeds_decimal_representation_l62_62536


namespace first_car_speed_l62_62218

theorem first_car_speed
  (highway_length : ℝ)
  (second_car_speed : ℝ)
  (meeting_time : ℝ)
  (D1 D2 : ℝ) :
  highway_length = 45 → second_car_speed = 16 → meeting_time = 1.5 → D2 = second_car_speed * meeting_time → D1 + D2 = highway_length → D1 = 14 * meeting_time :=
by
  intros h_highway h_speed h_time h_D2 h_sum
  sorry

end first_car_speed_l62_62218


namespace average_height_40_girls_l62_62498

/-- Given conditions for a class of 50 students, where the average height of 40 girls is H,
    the average height of the remaining 10 girls is 167 cm, and the average height of the whole
    class is 168.6 cm, prove that the average height H of the 40 girls is 169 cm. -/
theorem average_height_40_girls (H : ℝ)
  (h1 : 0 < H)
  (h2 : (40 * H + 10 * 167) = 50 * 168.6) :
  H = 169 :=
by
  sorry

end average_height_40_girls_l62_62498


namespace tree_circumference_inequality_l62_62544

theorem tree_circumference_inequality (x : ℝ) : 
  (∀ t : ℝ, t = 10 + 3 * x ∧ t > 90 → x > 80 / 3) :=
by
  intro t ht
  obtain ⟨h_t_eq, h_t_gt_90⟩ := ht
  linarith

end tree_circumference_inequality_l62_62544


namespace avg_height_of_class_is_168_6_l62_62791

noncomputable def avgHeightClass : ℕ → ℕ → ℕ → ℕ → ℚ :=
  λ n₁ h₁ n₂ h₂ => (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂)

theorem avg_height_of_class_is_168_6 :
  avgHeightClass 40 169 10 167 = 168.6 := 
by 
  sorry

end avg_height_of_class_is_168_6_l62_62791


namespace gcd_lcm_45_150_l62_62205

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 :=
by
  sorry

end gcd_lcm_45_150_l62_62205


namespace fraction_meaningful_l62_62532

theorem fraction_meaningful (x : ℝ) : x - 3 ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_l62_62532


namespace total_fish_at_wedding_l62_62326

def num_tables : ℕ := 32
def fish_per_table_except_one : ℕ := 2
def fish_on_special_table : ℕ := 3
def number_of_special_tables : ℕ := 1
def number_of_regular_tables : ℕ := num_tables - number_of_special_tables

theorem total_fish_at_wedding : 
  (number_of_regular_tables * fish_per_table_except_one) + (number_of_special_tables * fish_on_special_table) = 65 :=
by
  sorry

end total_fish_at_wedding_l62_62326


namespace g_difference_l62_62721

-- Define the function g(n)
def g (n : ℤ) : ℚ := (1/2 : ℚ) * n^2 * (n + 3)

-- State the theorem
theorem g_difference (s : ℤ) : g s - g (s - 1) = (1/2 : ℚ) * (3 * s - 2) := by
  sorry

end g_difference_l62_62721


namespace percentage_increase_sale_l62_62216

theorem percentage_increase_sale (P S : ℝ) (hP : 0 < P) (hS : 0 < S) :
  let new_price := 0.65 * P
  let original_revenue := P * S
  let new_revenue := 1.17 * original_revenue
  let percentage_increase := 80 / 100
  let new_sales := S * (1 + percentage_increase)
  new_price * new_sales = new_revenue :=
by
  sorry

end percentage_increase_sale_l62_62216


namespace probability_of_chosen_primes_l62_62010

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def total_ways : ℕ := Nat.choose 30 2
def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
def primes_not_divisible_by_5 : List ℕ := [2, 3, 7, 11, 13, 17, 19, 23, 29]

def chosen_primes (s : Finset ℕ) : Prop :=
  s.card = 2 ∧
  (∀ n ∈ s, n ∈ primes_not_divisible_by_5)  ∧
  (∀ n ∈ s, n ≠ 5) -- (5 is already excluded in the prime list, but for completeness)

def favorable_ways : ℕ := Nat.choose 9 2  -- 9 primes not divisible by 5

def probability := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_chosen_primes:
  probability = (12 / 145 : ℚ) :=
by
  sorry

end probability_of_chosen_primes_l62_62010


namespace remainder_sum_mult_3_zero_mod_18_l62_62754

theorem remainder_sum_mult_3_zero_mod_18
  (p q r s : ℕ)
  (hp : p % 18 = 8)
  (hq : q % 18 = 11)
  (hr : r % 18 = 14)
  (hs : s % 18 = 15) :
  3 * (p + q + r + s) % 18 = 0 :=
by
  sorry

end remainder_sum_mult_3_zero_mod_18_l62_62754


namespace exp_add_l62_62633

theorem exp_add (a : ℝ) (x₁ x₂ : ℝ) : a^(x₁ + x₂) = a^x₁ * a^x₂ :=
sorry

end exp_add_l62_62633


namespace pollution_control_l62_62408

theorem pollution_control (x y : ℕ) (h1 : x - y = 5) (h2 : 2 * x + 3 * y = 45) : x = 12 ∧ y = 7 :=
by
  sorry

end pollution_control_l62_62408


namespace intersection_distance_eq_l62_62164

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end intersection_distance_eq_l62_62164


namespace monotonically_increasing_power_function_l62_62263

theorem monotonically_increasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m ^ 2 - 2 * m - 2) * x ^ (m - 2) > 0 → (m ^ 2 - 2 * m - 2) > 0 ∧ (m - 2) > 0) ↔ m = 3 := 
sorry

end monotonically_increasing_power_function_l62_62263


namespace ratio_of_first_to_fourth_term_l62_62865

theorem ratio_of_first_to_fourth_term (a d : ℝ) (h1 : (a + d) + (a + 3 * d) = 6 * a) (h2 : a + 2 * d = 10) :
  a / (a + 3 * d) = 1 / 4 :=
by
  sorry

end ratio_of_first_to_fourth_term_l62_62865


namespace total_meals_sold_l62_62705

-- Definitions based on the conditions
def ratio_kids_adult := 2 / 1
def kids_meals := 8

-- The proof problem statement
theorem total_meals_sold : (∃ adults_meals : ℕ, 2 * adults_meals = kids_meals) → (kids_meals + 4 = 12) := 
by 
  sorry

end total_meals_sold_l62_62705


namespace find_x_l62_62066

theorem find_x
  (a b x : ℝ)
  (h1 : a * (x + 2) + b * (x + 2) = 60)
  (h2 : a + b = 12) :
  x = 3 :=
by
  sorry

end find_x_l62_62066


namespace elizabeth_bananas_eaten_l62_62961

theorem elizabeth_bananas_eaten (initial_bananas remaining_bananas eaten_bananas : ℕ) 
    (h1 : initial_bananas = 12) 
    (h2 : remaining_bananas = 8) 
    (h3 : eaten_bananas = initial_bananas - remaining_bananas) :
    eaten_bananas = 4 := 
sorry

end elizabeth_bananas_eaten_l62_62961


namespace find_k_value_l62_62350

theorem find_k_value : 
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 16 * 12 ^ 1001 :=
by
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  sorry

end find_k_value_l62_62350


namespace smallest_q_for_5_in_range_l62_62387

theorem smallest_q_for_5_in_range : ∃ q, (q = 9) ∧ (∃ x, (x^2 - 4 * x + q = 5)) := 
by 
  sorry

end smallest_q_for_5_in_range_l62_62387


namespace number_of_true_propositions_l62_62745

variable (x : ℝ)

def original_proposition (x : ℝ) : Prop := (x = 5) → (x^2 - 8 * x + 15 = 0)
def converse_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 = 0) → (x = 5)
def inverse_proposition (x : ℝ) : Prop := (x ≠ 5) → (x^2 - 8 * x + 15 ≠ 0)
def contrapositive_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 ≠ 0) → (x ≠ 5)

theorem number_of_true_propositions : 
  (original_proposition x ∧ contrapositive_proposition x) ∧
  ¬(converse_proposition x) ∧ ¬(inverse_proposition x) ↔ true := sorry

end number_of_true_propositions_l62_62745


namespace min_value_shift_l62_62858

noncomputable def f (x : ℝ) (c : ℝ) := x^2 + 4 * x + 5 - c

theorem min_value_shift (c : ℝ) (h : ∀ x : ℝ, f x c ≥ 2) :
  ∀ x : ℝ, f (x - 2009) c ≥ 2 :=
sorry

end min_value_shift_l62_62858


namespace total_marbles_l62_62443

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l62_62443


namespace probability_correct_l62_62262

-- Definitions and conditions
def G : List Char := ['A', 'B', 'C', 'D']

-- Number of favorable arrangements where A is adjacent to B and C
def favorable_arrangements : ℕ := 4  -- ABCD, BCDA, DABC, and CDAB

-- Total possible arrangements of 4 people
def total_arrangements : ℕ := 24  -- 4!

-- Probability calculation
def probability_A_adjacent_B_C : ℚ := favorable_arrangements / total_arrangements

-- Prove that this probability equals 1/6
theorem probability_correct : probability_A_adjacent_B_C = 1 / 6 := by
  sorry

end probability_correct_l62_62262


namespace jordan_annual_income_l62_62456

theorem jordan_annual_income (q : ℝ) (I T : ℝ) 
  (h1 : T = q * 35000 + (q + 3) * (I - 35000))
  (h2 : T = (q + 0.4) * I) : 
  I = 40000 :=
by sorry

end jordan_annual_income_l62_62456


namespace quad_roots_expression_l62_62979

theorem quad_roots_expression (x1 x2 : ℝ) (h1 : x1 * x1 + 2019 * x1 + 1 = 0) (h2 : x2 * x2 + 2019 * x2 + 1 = 0) :
  x1 * x2 - x1 - x2 = 2020 :=
sorry

end quad_roots_expression_l62_62979


namespace sum_volumes_spheres_l62_62801

theorem sum_volumes_spheres (l : ℝ) (h_l : l = 2) : 
  ∑' (n : ℕ), (4 / 3) * π * ((1 / (3 ^ (n + 1))) ^ 3) = (2 * π / 39) :=
by
  sorry

end sum_volumes_spheres_l62_62801


namespace average_exp_Feb_to_Jul_l62_62997

theorem average_exp_Feb_to_Jul (x y z : ℝ) 
    (h1 : 1200 + x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) = 6 * 4200) 
    (h2 : 0 ≤ x) 
    (h3 : 0 ≤ z) : 
    (x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) + 1500) / 6 = 4250 :=
by
    sorry

end average_exp_Feb_to_Jul_l62_62997


namespace max_intersections_two_circles_three_lines_l62_62194

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l62_62194


namespace chessboard_movement_l62_62276

-- Defining the problem as described in the transformed proof problem

theorem chessboard_movement (pieces : Nat) (adjacent_empty_square : Nat → Nat → Bool) (visited_all_squares : Nat → Bool)
  (returns_to_starting_square : Nat → Bool) :
  (∃ (moment : Nat), ∀ (piece : Nat), ¬ returns_to_starting_square piece) :=
by
  -- Here we state that there exists a moment when each piece (checker) is not on its starting square
  sorry

end chessboard_movement_l62_62276


namespace solution_l62_62390

noncomputable def problem (x : ℕ) : Prop :=
  2 ^ 28 = 4 ^ x  -- Simplified form of the condition given

theorem solution : problem 14 :=
by
  sorry

end solution_l62_62390


namespace married_men_fraction_l62_62802

-- define the total number of women
def W : ℕ := 7

-- define the number of single women
def single_women (W : ℕ) : ℕ := 3

-- define the probability of picking a single woman
def P_s : ℚ := single_women W / W

-- define number of married women
def married_women (W : ℕ) : ℕ := W - single_women W

-- define number of married men
def married_men (W : ℕ) : ℕ := married_women W

-- define total number of people
def total_people (W : ℕ) : ℕ := W + married_men W

-- define fraction of married men
def married_men_ratio (W : ℕ) : ℚ := married_men W / total_people W

-- theorem to prove that the ratio is 4/11
theorem married_men_fraction : married_men_ratio W = 4 / 11 := 
by 
  sorry

end married_men_fraction_l62_62802


namespace kx2_kx_1_pos_l62_62428

theorem kx2_kx_1_pos (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end kx2_kx_1_pos_l62_62428


namespace fixed_point_l62_62312

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : func a 1 = 1 :=
by {
  -- We need to prove that func a 1 = 1 for any a > 0 and a ≠ 1
  sorry
}

end fixed_point_l62_62312


namespace arithmetic_sequence_max_min_b_l62_62354

-- Define the sequence a_n
def S (n : ℕ) : ℚ := (1/2) * n^2 - 2 * n
def a (n : ℕ) : ℚ := S n - S (n - 1)

-- Question 1: Prove that {a_n} is an arithmetic sequence with a common difference of 1
theorem arithmetic_sequence (n : ℕ) (hn : n ≥ 2) : 
  a n - a (n - 1) = 1 :=
sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (a n + 1) / a n

-- Question 2: Prove that b_3 is the maximum value and b_2 is the minimum value in {b_n}
theorem max_min_b (hn2 : 2 ≥ 1) (hn3 : 3 ≥ 1) : 
  b 3 = 3 ∧ b 2 = -1 :=
sorry

end arithmetic_sequence_max_min_b_l62_62354


namespace complex_fraction_simplify_l62_62984

variable (i : ℂ)
variable (h : i^2 = -1)

theorem complex_fraction_simplify :
  (1 - i) / ((1 + i) ^ 2) = -1/2 - i/2 :=
by
  sorry

end complex_fraction_simplify_l62_62984


namespace find_subtracted_value_l62_62025

theorem find_subtracted_value (n x : ℕ) (h1 : n = 120) (h2 : n / 6 - x = 5) : x = 15 := by
  sorry

end find_subtracted_value_l62_62025


namespace find_number_l62_62698

theorem find_number (x : ℝ) (h : 45 - 3 * x = 12) : x = 11 :=
sorry

end find_number_l62_62698


namespace discount_calc_l62_62658

noncomputable def discount_percentage 
    (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := cost_price + (markup_percentage / 100 * cost_price)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

theorem discount_calc :
  discount_percentage 540 15 460 = 25.92 :=
by
  sorry

end discount_calc_l62_62658


namespace find_m_values_l62_62952

theorem find_m_values (m : ℕ) : (m - 3) ^ m = 1 ↔ m = 0 ∨ m = 2 ∨ m = 4 := sorry

end find_m_values_l62_62952


namespace mr_brown_no_calls_in_2020_l62_62938

noncomputable def number_of_days_with_no_calls (total_days : ℕ) (calls_niece1 : ℕ) (calls_niece2 : ℕ) (calls_niece3 : ℕ) : ℕ := 
  let calls_2 := total_days / calls_niece1
  let calls_3 := total_days / calls_niece2
  let calls_4 := total_days / calls_niece3
  let calls_6 := total_days / (Nat.lcm calls_niece1 calls_niece2)
  let calls_12_ := total_days / (Nat.lcm calls_niece1 (Nat.lcm calls_niece2 calls_niece3))
  total_days - (calls_2 + calls_3 + calls_4 - calls_6 - calls_4 - (total_days / calls_niece2 / 4) + calls_12_)

theorem mr_brown_no_calls_in_2020 : number_of_days_with_no_calls 365 2 3 4 = 122 := 
  by 
    -- Proof steps would go here
    sorry

end mr_brown_no_calls_in_2020_l62_62938


namespace sum_of_possible_n_values_l62_62605

theorem sum_of_possible_n_values (m n : ℕ) 
  (h : 0 < m ∧ 0 < n)
  (eq1 : 1/m + 1/n = 1/5) : 
  n = 6 ∨ n = 10 ∨ n = 30 → 
  m = 30 ∨ m = 10 ∨ m = 6 ∨ m = 5 ∨ m = 25 ∨ m = 1 →
  (6 + 10 + 30 = 46) := 
by 
  sorry

end sum_of_possible_n_values_l62_62605


namespace find_line_eq_l62_62697

theorem find_line_eq (m b k : ℝ) (h1 : (2, 7) ∈ ⋃ x, {(x, m * x + b)}) (h2 : ∀ k, abs ((k^2 + 4 * k + 3) - (m * k + b)) = 4) (h3 : b ≠ 0) : (m = 10) ∧ (b = -13) := by
  sorry

end find_line_eq_l62_62697


namespace cube_painted_surface_l62_62513

theorem cube_painted_surface (n : ℕ) (hn : n > 2) 
: 6 * (n - 2) ^ 2 = (n - 2) ^ 3 → n = 8 :=
by
  sorry

end cube_painted_surface_l62_62513


namespace quadratic_coefficients_l62_62564

theorem quadratic_coefficients (x : ℝ) : 
  let a := 3
  let b := -5
  let c := 1
  3 * x^2 + 1 = 5 * x → a * x^2 + b * x + c = 0 := by
sorry

end quadratic_coefficients_l62_62564


namespace det_E_eq_25_l62_62905

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l62_62905


namespace quadrilateral_is_trapezoid_l62_62530

variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Define the type of vectors and vector space over the reals
variables (a b : V) -- Vectors a and b
variables (AB BC CD AD : V) -- Vectors representing sides of quadrilateral

-- Condition: vectors a and b are not collinear
def not_collinear (a b : V) : Prop := ∀ k : ℝ, k ≠ 0 → a ≠ k • b

-- Given Conditions
def conditions (a b AB BC CD : V) : Prop :=
  AB = a + 2 • b ∧
  BC = -4 • a - b ∧
  CD = -5 • a - 3 • b ∧
  not_collinear a b

-- The to-be-proven property
def is_trapezoid (AB BC CD AD : V) : Prop :=
  AD = 2 • BC

theorem quadrilateral_is_trapezoid 
  (a b AB BC CD : V) 
  (h : conditions a b AB BC CD)
  : is_trapezoid AB BC CD (AB + BC + CD) :=
sorry

end quadrilateral_is_trapezoid_l62_62530


namespace trivia_team_total_score_l62_62701

theorem trivia_team_total_score 
  (scores : List ℕ)
  (present_members : List ℕ)
  (H_score : scores = [4, 6, 2, 8, 3, 5, 10, 3, 7])
  (H_present : present_members = scores) :
  List.sum present_members = 48 := 
by
  sorry

end trivia_team_total_score_l62_62701


namespace square_side_length_l62_62314

theorem square_side_length (a b s : ℝ) 
  (h_area : a * b = 54) 
  (h_square_condition : 3 * a = b / 2) : 
  s = 9 :=
by 
  sorry

end square_side_length_l62_62314


namespace quadrilateral_ratio_l62_62731

theorem quadrilateral_ratio (AB CD AD BC IA IB IC ID : ℝ)
  (h_tangential : AB + CD = AD + BC)
  (h_IA : IA = 5)
  (h_IB : IB = 7)
  (h_IC : IC = 4)
  (h_ID : ID = 9) :
  AB / CD = 35 / 36 :=
by
  -- Proof will be provided here
  sorry

end quadrilateral_ratio_l62_62731


namespace min_value_expression_l62_62653

theorem min_value_expression (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  ∃ (z : ℝ), z = (1 / (2 * x) + x / (y + 1)) ∧ z = 5 / 4 :=
sorry

end min_value_expression_l62_62653


namespace maximum_value_g_on_interval_l62_62055

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximum_value_g_on_interval : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 := by
  sorry

end maximum_value_g_on_interval_l62_62055


namespace total_initial_collection_l62_62042

variable (marco strawberries father strawberries_lost : ℕ)
variable (marco : ℕ := 12)
variable (father : ℕ := 16)
variable (strawberries_lost : ℕ := 8)
variable (total_initial_weight : ℕ := marco + father + strawberries_lost)

theorem total_initial_collection : total_initial_weight = 36 :=
by
  sorry

end total_initial_collection_l62_62042


namespace emails_received_afternoon_is_one_l62_62596

-- Define the number of emails received by Jack in the morning
def emails_received_morning : ℕ := 4

-- Define the total number of emails received by Jack in a day
def total_emails_received : ℕ := 5

-- Define the number of emails received by Jack in the afternoon
def emails_received_afternoon : ℕ := total_emails_received - emails_received_morning

-- Prove the number of emails received by Jack in the afternoon
theorem emails_received_afternoon_is_one : emails_received_afternoon = 1 :=
by 
  -- Proof is neglected as per instructions.
  sorry

end emails_received_afternoon_is_one_l62_62596


namespace combined_score_210_l62_62594

-- Define the constants and variables
def total_questions : ℕ := 50
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5
def jose_extra_marks (alisson_score : ℕ) : ℕ := 40
def meghan_less_marks (jose_score : ℕ) : ℕ := 20

-- Define the total possible marks
def total_possible_marks : ℕ := total_questions * marks_per_question

-- Given the conditions, we need to prove the total combined score is 210
theorem combined_score_210 : 
  ∃ (jose_score meghan_score alisson_score combined_score : ℕ), 
  jose_score = total_possible_marks - (jose_wrong_questions * marks_per_question) ∧
  meghan_score = jose_score - meghan_less_marks jose_score ∧
  alisson_score = jose_score - jose_extra_marks alisson_score ∧
  combined_score = jose_score + meghan_score + alisson_score ∧
  combined_score = 210 := by
  sorry

end combined_score_210_l62_62594


namespace max_alpha_l62_62430

theorem max_alpha (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hSum : A + B + C = π)
  (hmin : ∀ alpha, alpha = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A))) :
  ∃ alpha, alpha = 2 * π / 9 := 
sorry

end max_alpha_l62_62430


namespace inequality_proof_l62_62981

-- Defining the conditions
variable (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (cond : 1 / a + 1 / b = 1)

-- Defining the theorem to be proved
theorem inequality_proof (n : ℕ) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by
  sorry

end inequality_proof_l62_62981


namespace product_three_consecutive_integers_divisible_by_six_l62_62471

theorem product_three_consecutive_integers_divisible_by_six
  (n : ℕ) (h_pos : 0 < n) : ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k :=
by sorry

end product_three_consecutive_integers_divisible_by_six_l62_62471


namespace measure_angle_F_l62_62612

theorem measure_angle_F :
  ∃ (F : ℝ), F = 18 ∧
  ∃ (D E : ℝ),
  D = 75 ∧
  E = 15 + 4 * F ∧
  D + E + F = 180 :=
by
  sorry

end measure_angle_F_l62_62612


namespace inscribed_polygon_sides_l62_62273

-- We start by defining the conditions of the problem in Lean.
def radius := 1
def side_length_condition (n : ℕ) : Prop :=
  1 < 2 * Real.sin (Real.pi / n) ∧ 2 * Real.sin (Real.pi / n) < Real.sqrt 2

-- Now we state the main theorem.
theorem inscribed_polygon_sides (n : ℕ) (h1 : side_length_condition n) : n = 5 :=
  sorry

end inscribed_polygon_sides_l62_62273


namespace garrett_total_spent_l62_62020

/-- Garrett bought 6 oatmeal raisin granola bars, each costing $1.25. -/
def oatmeal_bars_count : Nat := 6
def oatmeal_bars_cost_per_unit : ℝ := 1.25

/-- Garrett bought 8 peanut granola bars, each costing $1.50. -/
def peanut_bars_count : Nat := 8
def peanut_bars_cost_per_unit : ℝ := 1.50

/-- The total amount spent on granola bars is $19.50. -/
theorem garrett_total_spent : oatmeal_bars_count * oatmeal_bars_cost_per_unit + peanut_bars_count * peanut_bars_cost_per_unit = 19.50 :=
by
  sorry

end garrett_total_spent_l62_62020


namespace value_of_y_l62_62500

theorem value_of_y (x y : ℤ) (h1 : x^2 - 3 * x + 6 = y + 2) (h2 : x = -8) : y = 92 :=
by
  sorry

end value_of_y_l62_62500


namespace tyler_meal_combinations_is_720_l62_62136

-- Required imports for permutations and combinations
open Nat
open BigOperators

-- Assumptions based on the problem conditions
def meat_options  := 4
def veg_options := 4
def dessert_options := 5
def bread_options := 3

-- Using combinations and permutations for calculations
def comb(n k : ℕ) := Nat.choose n k
def perm(n k : ℕ) := n.factorial / (n - k).factorial

-- Number of ways to choose meals
def meal_combinations : ℕ :=
  meat_options * (comb veg_options 2) * dessert_options * (perm bread_options 2)

theorem tyler_meal_combinations_is_720 : meal_combinations = 720 := by
  -- We provide proof later; for now, put sorry to skip
  sorry

end tyler_meal_combinations_is_720_l62_62136


namespace medical_team_selection_l62_62239

theorem medical_team_selection : 
  let male_doctors := 6
  let female_doctors := 5
  let choose_male := Nat.choose male_doctors 2
  let choose_female := Nat.choose female_doctors 1
  choose_male * choose_female = 75 := 
by 
  sorry

end medical_team_selection_l62_62239


namespace opposite_of_num_l62_62233

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l62_62233


namespace initial_volume_mixture_l62_62068

theorem initial_volume_mixture (x : ℝ) :
  (4 * x) / (3 * x + 13) = 5 / 7 →
  13 * x = 65 →
  7 * x = 35 := 
by
  intro h1 h2
  sorry

end initial_volume_mixture_l62_62068


namespace cube_volume_l62_62388

/-- Given the perimeter of one face of a cube, proving the volume of the cube -/

theorem cube_volume (h : ∀ (s : ℝ), 4 * s = 28) : (∃ (v : ℝ), v = (7 : ℝ) ^ 3) :=
by
  sorry

end cube_volume_l62_62388


namespace smaller_of_x_y_l62_62486

theorem smaller_of_x_y (x y a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : x * y = c) (h6 : x^2 - b * x + a * y = 0) : min x y = c / a :=
by sorry

end smaller_of_x_y_l62_62486


namespace ratio_lcm_gcf_240_360_l62_62980

theorem ratio_lcm_gcf_240_360 : Nat.lcm 240 360 / Nat.gcd 240 360 = 60 :=
by
  sorry

end ratio_lcm_gcf_240_360_l62_62980


namespace range_of_m_l62_62817

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = 3 * m + 1)
  (h2 : x + 2 * y = 3)
  (h3 : 2 * x - y < 1) : 
  m < 1 := 
sorry

end range_of_m_l62_62817


namespace striped_turtles_adult_percentage_l62_62382

noncomputable def percentage_of_adult_striped_turtles (total_turtles : ℕ) (female_percentage : ℝ) (stripes_per_male : ℕ) (baby_stripes : ℕ) : ℝ :=
  let total_male := total_turtles * (1 - female_percentage)
  let total_striped_male := total_male / stripes_per_male
  let adult_striped_males := total_striped_male - baby_stripes
  (adult_striped_males / total_striped_male) * 100

theorem striped_turtles_adult_percentage :
  percentage_of_adult_striped_turtles 100 0.60 4 4 = 60 := 
  by
  -- proof omitted
  sorry

end striped_turtles_adult_percentage_l62_62382


namespace Sue_necklace_total_beads_l62_62139

theorem Sue_necklace_total_beads :
  ∃ (purple blue green red total : ℕ),
  purple = 7 ∧
  blue = 2 * purple ∧
  green = blue + 11 ∧
  (red : ℕ) = green / 2 ∧
  total = purple + blue + green + red ∧
  total % 2 = 0 ∧
  total = 58 := by
    sorry

end Sue_necklace_total_beads_l62_62139


namespace problem_statement_l62_62956

variable {f : ℝ → ℝ}
variable {a : ℝ}

def odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem problem_statement
  (h_odd : odd_function f)
  (h_periodic : periodic_function f 3)
  (h_f1 : f 1 < 1)
  (h_f2 : f 2 = a) :
  -1 < a ∧ a < 2 :=
sorry

end problem_statement_l62_62956


namespace bat_wings_area_l62_62036

-- Defining a rectangle and its properties.
structure Rectangle where
  PQ : ℝ
  QR : ℝ
  PT : ℝ
  TR : ℝ
  RQ : ℝ

-- Example rectangle from the problem
def PQRS : Rectangle := { PQ := 5, QR := 3, PT := 1, TR := 1, RQ := 1 }

-- Calculate area of "bat wings" if the rectangle is specified as in the above structure.
-- Expected result is 3.5
theorem bat_wings_area (r : Rectangle) (hPQ : r.PQ = 5) (hQR : r.QR = 3) 
    (hPT : r.PT = 1) (hTR : r.TR = 1) (hRQ : r.RQ = 1) : 
    ∃ area : ℝ, area = 3.5 :=
by
  -- Adding the proof would involve geometric calculations.
  -- Skipping the proof for now.
  sorry

end bat_wings_area_l62_62036


namespace zane_total_payment_l62_62454

open Real

noncomputable def shirt1_price := 50.0
noncomputable def shirt2_price := 50.0
noncomputable def discount1 := 0.4 * shirt1_price
noncomputable def discount2 := 0.3 * shirt2_price
noncomputable def price1_after_discount := shirt1_price - discount1
noncomputable def price2_after_discount := shirt2_price - discount2
noncomputable def total_before_tax := price1_after_discount + price2_after_discount
noncomputable def sales_tax := 0.08 * total_before_tax
noncomputable def total_cost := total_before_tax + sales_tax

-- We want to prove:
theorem zane_total_payment : total_cost = 70.20 := by sorry

end zane_total_payment_l62_62454


namespace geometric_sequence_product_l62_62087

variable {α : Type*} [LinearOrderedField α]

theorem geometric_sequence_product :
  ∀ (a r : α), (a^3 * r^6 = 3) → (a^3 * r^15 = 24) → (a^3 * r^24 = 192) :=
by
  intros a r h1 h2
  sorry

end geometric_sequence_product_l62_62087


namespace trains_at_start_2016_l62_62203

def traversal_time_red := 7
def traversal_time_blue := 8
def traversal_time_green := 9

def return_period_red := 2 * traversal_time_red
def return_period_blue := 2 * traversal_time_blue
def return_period_green := 2 * traversal_time_green

def train_start_pos_time := 2016
noncomputable def lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)

theorem trains_at_start_2016 :
  train_start_pos_time % lcm_period = 0 :=
by
  have return_period_red := 2 * traversal_time_red
  have return_period_blue := 2 * traversal_time_blue
  have return_period_green := 2 * traversal_time_green
  have lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)
  have train_start_pos_time := 2016
  exact sorry

end trains_at_start_2016_l62_62203


namespace three_times_sum_first_35_odd_l62_62373

/-- 
The sum of the first n odd numbers --/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

/-- Given that 69 is the 35th odd number --/
theorem three_times_sum_first_35_odd : 3 * sum_first_n_odd 35 = 3675 := by
  sorry

end three_times_sum_first_35_odd_l62_62373


namespace number_of_students_passed_l62_62439

theorem number_of_students_passed (total_students : ℕ) (failure_frequency : ℝ) (h1 : total_students = 1000) (h2 : failure_frequency = 0.4) : 
  (total_students - (total_students * failure_frequency)) = 600 :=
by
  sorry

end number_of_students_passed_l62_62439


namespace delivery_boxes_l62_62132

-- Define the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Define the total number of boxes
def total_boxes : ℕ := stops * boxes_per_stop

-- State the theorem
theorem delivery_boxes : total_boxes = 27 := by
  sorry

end delivery_boxes_l62_62132


namespace nora_muffin_price_l62_62126

theorem nora_muffin_price
  (cases : ℕ)
  (packs_per_case : ℕ)
  (muffins_per_pack : ℕ)
  (total_money : ℕ)
  (total_cases : ℕ)
  (h1 : total_money = 120)
  (h2 : packs_per_case = 3)
  (h3 : muffins_per_pack = 4)
  (h4 : total_cases = 5) :
  (total_money / (total_cases * packs_per_case * muffins_per_pack) = 2) :=
by
  sorry

end nora_muffin_price_l62_62126


namespace gomoku_black_pieces_l62_62859

/--
Two students, A and B, are preparing to play a game of Gomoku but find that 
the box only contains a certain number of black and white pieces, each of the
same quantity, and the total does not exceed 10. Then, they find 20 more pieces 
(only black and white) and add them to the box. At this point, the ratio of 
the total number of white to black pieces is 7:8. We want to prove that the total number
of black pieces in the box after adding is 16.
-/
theorem gomoku_black_pieces (x y : ℕ) (hx : x = 15 * y - 160) (h_total : x + y ≤ 5)
  (h_ratio : 7 * (x + y) = 8 * (x + (20 - y))) : (x + y = 16) :=
by
  sorry

end gomoku_black_pieces_l62_62859


namespace range_of_m_l62_62966

theorem range_of_m (x m : ℝ)
  (h1 : (x + 2) / (10 - x) ≥ 0)
  (h2 : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (h3 : m < 0)
  (h4 : ∀ (x : ℝ), (x + 2) / (10 - x) ≥ 0 → (x^2 - 2 * x + 1 - m^2 ≤ 0)) :
  -3 ≤ m ∧ m < 0 :=
sorry

end range_of_m_l62_62966


namespace range_of_a_l62_62026

noncomputable def f (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 2, (a - 1 / x) ≥ 0) ↔ (a ≥ 1 / 2) :=
by
  sorry

end range_of_a_l62_62026


namespace count_positive_integers_satisfying_inequality_l62_62489

theorem count_positive_integers_satisfying_inequality :
  ∃ n : ℕ, n = 4 ∧ ∀ x : ℕ, (10 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 50) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := 
by
  sorry

end count_positive_integers_satisfying_inequality_l62_62489


namespace solve_inequality_l62_62005

theorem solve_inequality (a : ℝ) : 
  (a > 0 → {x : ℝ | x < -a / 4 ∨ x > a / 3 } = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a = 0 → {x : ℝ | x ≠ 0} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a < 0 → {x : ℝ | x < a / 3 ∨ x > -a / 4} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) :=
sorry

end solve_inequality_l62_62005


namespace prices_of_books_book_purchasing_plans_l62_62121

-- Define the conditions
def cost_eq1 (x y : ℕ): Prop := 20 * x + 40 * y = 1520
def cost_eq2 (x y : ℕ): Prop := 20 * x - 20 * y = 440
def plan_conditions (x y : ℕ): Prop := (20 + y - x = 20) ∧ (x + y + 20 ≥ 72) ∧ (40 * x + 18 * (y + 20) ≤ 2000)

-- Prove price of each book
theorem prices_of_books : 
  ∃ (x y : ℕ), cost_eq1 x y ∧ cost_eq2 x y ∧ x = 40 ∧ y = 18 :=
by {
  sorry
}

-- Prove possible book purchasing plans
theorem book_purchasing_plans : 
  ∃ (x : ℕ), plan_conditions x (x + 20) ∧ 
  (x = 26 ∧ x + 20 = 46 ∨ 
   x = 27 ∧ x + 20 = 47 ∨ 
   x = 28 ∧ x + 20 = 48) :=
by {
  sorry
}

end prices_of_books_book_purchasing_plans_l62_62121


namespace math_problem_l62_62983

variable (x y : ℝ)

theorem math_problem (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0) (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) :
  x * y - 12 * x + 15 * y = 0 :=
  sorry

end math_problem_l62_62983


namespace abs_ineq_subs_ineq_l62_62468

-- Problem 1
theorem abs_ineq (x : ℝ) : -2 ≤ x ∧ x ≤ 2 ↔ |x - 1| + |x + 1| ≤ 4 := 
sorry

-- Problem 2
theorem subs_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a + b + c := 
sorry

end abs_ineq_subs_ineq_l62_62468


namespace probability_units_digit_odd_l62_62621

theorem probability_units_digit_odd :
  (1 / 2 : ℚ) = 5 / 10 :=
by {
  -- This is the equivalent mathematically correct theorem statement
  -- The proof is omitted as per instructions
  sorry
}

end probability_units_digit_odd_l62_62621


namespace quadratic_inequality_l62_62771

theorem quadratic_inequality (a x1 x2 : ℝ) (h_eq : x1 ^ 2 - a * x1 + a = 0) (h_eq' : x2 ^ 2 - a * x2 + a = 0) :
  x1^2 + x2^2 ≥ 2 * (x1 + x2) :=
sorry

end quadratic_inequality_l62_62771


namespace find_whole_wheat_pastry_flour_l62_62684

variable (x : ℕ) -- where x is the pounds of whole-wheat pastry flour Sarah already had

-- Conditions
def rye_flour := 5
def whole_wheat_bread_flour := 10
def chickpea_flour := 3
def total_flour := 20

-- Total flour bought
def total_flour_bought := rye_flour + whole_wheat_bread_flour + chickpea_flour

-- Proof statement
theorem find_whole_wheat_pastry_flour (h : total_flour = total_flour_bought + x) : x = 2 :=
by
  -- The proof is omitted
  sorry

end find_whole_wheat_pastry_flour_l62_62684


namespace recreation_spent_percent_l62_62367

variable (W : ℝ) -- Assume W is the wages last week

-- Conditions
def last_week_spent_on_recreation (W : ℝ) : ℝ := 0.25 * W
def this_week_wages (W : ℝ) : ℝ := 0.70 * W
def this_week_spent_on_recreation (W : ℝ) : ℝ := 0.50 * (this_week_wages W)

-- Proof statement
theorem recreation_spent_percent (W : ℝ) :
  (this_week_spent_on_recreation W / last_week_spent_on_recreation W) * 100 = 140 := by
  sorry

end recreation_spent_percent_l62_62367


namespace find_constant_l62_62714

-- Given function f satisfying the conditions
variable (f : ℝ → ℝ)

-- Define the given conditions
variable (h1 : ∀ x : ℝ, f x + 3 * f (c - x) = x)
variable (h2 : f 2 = 2)

-- Statement to prove the constant c
theorem find_constant (c : ℝ) : (f x + 3 * f (c - x) = x) → (f 2 = 2) → c = 8 :=
by
  intro h1 h2
  sorry

end find_constant_l62_62714


namespace trig_expression_zero_l62_62819

theorem trig_expression_zero (α : ℝ) (h : Real.tan α = 2) : 
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := 
by
  sorry

end trig_expression_zero_l62_62819


namespace central_angle_measure_l62_62674

theorem central_angle_measure (p : ℝ) (x : ℝ) (h1 : p = 1 / 8) (h2 : p = x / 360) : x = 45 :=
by
  -- skipping the proof
  sorry

end central_angle_measure_l62_62674


namespace right_angle_sides_of_isosceles_right_triangle_l62_62747

def is_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

theorem right_angle_sides_of_isosceles_right_triangle
  (C : ℝ × ℝ)
  (hyp_line : ℝ → ℝ → Prop)
  (side_AC side_BC : ℝ → ℝ → Prop)
  (H1 : C = (3, -2))
  (H2 : hyp_line = is_on_line 3 (-1) 2)
  (H3 : side_AC = is_on_line 2 1 (-4))
  (H4 : side_BC = is_on_line 1 (-2) (-7))
  (H5 : ∃ x y, side_BC (3) y ∧ side_AC x (-2)) :
  side_AC = is_on_line 2 1 (-4) ∧ side_BC = is_on_line 1 (-2) (-7) :=
by
  sorry

end right_angle_sides_of_isosceles_right_triangle_l62_62747


namespace probability_of_queen_is_correct_l62_62259

def deck_size : ℕ := 52
def queen_count : ℕ := 4

-- This definition denotes the probability calculation.
def probability_drawing_queen : ℚ := queen_count / deck_size

theorem probability_of_queen_is_correct :
  probability_drawing_queen = 1 / 13 :=
by
  sorry

end probability_of_queen_is_correct_l62_62259


namespace cos_angle_subtraction_l62_62752

open Real

theorem cos_angle_subtraction (A B : ℝ) (h1 : sin A + sin B = 3 / 2) (h2 : cos A + cos B = 1) :
  cos (A - B) = 5 / 8 :=
sorry

end cos_angle_subtraction_l62_62752


namespace sum_of_terms_7_8_9_l62_62295

namespace ArithmeticSequence

-- Define the sequence and its properties
variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 0 + n * (n - 1) / 2 * (a 1 - a 0)

def condition3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def condition5 (S : ℕ → ℤ) : Prop :=
  S 5 = 30

-- Main statement to prove
theorem sum_of_terms_7_8_9 :
  is_arithmetic_sequence a →
  (∀ n, S n = sum_first_n_terms a n) →
  condition3 S →
  condition5 S →
  a 7 + a 8 + a 9 = 63 :=
by
  sorry

end ArithmeticSequence

end sum_of_terms_7_8_9_l62_62295


namespace find_constants_l62_62047

open Matrix 

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem find_constants :
  ∃ c d : ℝ, c = 1/12 ∧ d = 1/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end find_constants_l62_62047


namespace fraction_not_equal_l62_62424

theorem fraction_not_equal : ¬ (7 / 5 = 1 + 4 / 20) :=
by
  -- We'll use simplification to demonstrate the inequality
  sorry

end fraction_not_equal_l62_62424


namespace number_of_hydrogen_atoms_l62_62963

/-- 
A compound has a certain number of Hydrogen, 1 Chromium, and 4 Oxygen atoms. 
The molecular weight of the compound is 118. How many Hydrogen atoms are in the compound?
-/
theorem number_of_hydrogen_atoms
  (H Cr O : ℕ)
  (mw_H : ℕ := 1)
  (mw_Cr : ℕ := 52)
  (mw_O : ℕ := 16)
  (H_weight : ℕ := H * mw_H)
  (Cr_weight : ℕ := 1 * mw_Cr)
  (O_weight : ℕ := 4 * mw_O)
  (total_weight : ℕ := 118)
  (weight_without_H : ℕ := Cr_weight + O_weight) 
  (H_weight_calculated : ℕ := total_weight - weight_without_H) :
  H = 2 :=
  by
    sorry

end number_of_hydrogen_atoms_l62_62963


namespace katie_spending_l62_62618

theorem katie_spending :
  let price_per_flower : ℕ := 6
  let number_of_roses : ℕ := 5
  let number_of_daisies : ℕ := 5
  let total_number_of_flowers := number_of_roses + number_of_daisies
  let total_spending := total_number_of_flowers * price_per_flower
  total_spending = 60 :=
by
  sorry

end katie_spending_l62_62618


namespace product_ends_in_36_l62_62219

theorem product_ends_in_36 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
  ((10 * a + 6) * (10 * b + 6)) % 100 = 36 ↔ (a + b = 0 ∨ a + b = 5 ∨ a + b = 10 ∨ a + b = 15) :=
by
  sorry

end product_ends_in_36_l62_62219


namespace area_of_triangle_formed_by_tangency_points_l62_62767

theorem area_of_triangle_formed_by_tangency_points :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  let O1O2 := r1 + r2
  let O2O3 := r2 + r3
  let O1O3 := r1 + r3
  let s := (O1O2 + O2O3 + O1O3) / 2
  let A := Real.sqrt (s * (s - O1O2) * (s - O2O3) * (s - O1O3))
  let r := A / s
  r^2 = 5 / 3 := 
by
  sorry

end area_of_triangle_formed_by_tangency_points_l62_62767


namespace moles_of_C2H5Cl_l62_62269

-- Define chemical entities as types
structure Molecule where
  name : String

-- Declare molecules involved in the reaction
def C2H6 := Molecule.mk "C2H6"
def Cl2  := Molecule.mk "Cl2"
def C2H5Cl := Molecule.mk "C2H5Cl"
def HCl := Molecule.mk "HCl"

-- Define number of moles as a non-negative integer
def moles (m : Molecule) : ℕ := sorry

-- Conditions
axiom initial_moles_C2H6 : moles C2H6 = 3
axiom initial_moles_Cl2 : moles Cl2 = 3

-- Balanced reaction equation: 1 mole of C2H6 reacts with 1 mole of Cl2 to form 1 mole of C2H5Cl
axiom reaction_stoichiometry : ∀ (x : ℕ), moles C2H6 = x → moles Cl2 = x → moles C2H5Cl = x

-- Proof problem
theorem moles_of_C2H5Cl : moles C2H5Cl = 3 := by
  apply reaction_stoichiometry
  exact initial_moles_C2H6
  exact initial_moles_Cl2

end moles_of_C2H5Cl_l62_62269


namespace calculation_l62_62589

def operation_e (x y z : ℕ) : ℕ := 3 * x * y * z

theorem calculation :
  operation_e 3 (operation_e 4 5 6) 1 = 3240 :=
by
  sorry

end calculation_l62_62589


namespace original_portion_al_l62_62174

variable (a b c : ℕ)

theorem original_portion_al :
  a + b + c = 1200 ∧
  a - 150 + 3 * b + 3 * c = 1800 ∧
  c = 2 * b →
  a = 825 :=
by
  sorry

end original_portion_al_l62_62174


namespace number_of_days_l62_62124

theorem number_of_days (m1 d1 m2 d2 : ℕ) (h1 : m1 * d1 = m2 * d2) (k : ℕ) 
(h2 : m1 = 10) (h3 : d1 = 6) (h4 : m2 = 15) (h5 : k = 60) : 
d2 = 4 :=
by
  have : 10 * 6 = 60 := by sorry
  have : 15 * d2 = 60 := by sorry
  exact sorry

end number_of_days_l62_62124
