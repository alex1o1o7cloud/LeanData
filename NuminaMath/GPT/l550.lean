import Mathlib

namespace find_alpha_l550_55093

theorem find_alpha (n : ℕ) (h : ∀ x : ℤ, x * x * x + α * x + 4 - 2 * 2016 ^ n = 0 → ∀ r : ℤ, x = r)
  : α = -3 :=
sorry

end find_alpha_l550_55093


namespace percentage_increase_is_2_l550_55075

def alan_price := 2000
def john_price := 2040
def percentage_increase (alan_price : ℕ) (john_price : ℕ) : ℕ := (john_price - alan_price) * 100 / alan_price

theorem percentage_increase_is_2 (alan_price john_price : ℕ) (h₁ : alan_price = 2000) (h₂ : john_price = 2040) :
  percentage_increase alan_price john_price = 2 := by
  rw [h₁, h₂]
  sorry

end percentage_increase_is_2_l550_55075


namespace smallest_number_divisible_l550_55060

theorem smallest_number_divisible (n : ℕ) : (∃ n : ℕ, (n + 3) % 27 = 0 ∧ (n + 3) % 35 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0) ∧ n = 4722 :=
by
  sorry

end smallest_number_divisible_l550_55060


namespace distance_diff_is_0_point3_l550_55063

def john_walk_distance : ℝ := 0.7
def nina_walk_distance : ℝ := 0.4
def distance_difference_john_nina : ℝ := john_walk_distance - nina_walk_distance

theorem distance_diff_is_0_point3 : distance_difference_john_nina = 0.3 :=
by
  -- proof goes here
  sorry

end distance_diff_is_0_point3_l550_55063


namespace fraction_zero_imp_x_eq_two_l550_55076
open Nat Real

theorem fraction_zero_imp_x_eq_two (x : ℝ) (h: (2 - abs x) / (x + 2) = 0) : x = 2 :=
by
  have h1 : 2 - abs x = 0 := sorry
  have h2 : x + 2 ≠ 0 := sorry
  sorry

end fraction_zero_imp_x_eq_two_l550_55076


namespace scientific_notation_of_634000000_l550_55054

theorem scientific_notation_of_634000000 :
  634000000 = 6.34 * 10 ^ 8 := 
sorry

end scientific_notation_of_634000000_l550_55054


namespace trig_problem_l550_55082

theorem trig_problem 
  (α : ℝ) 
  (h1 : Real.cos α = -1/2) 
  (h2 : 180 * (Real.pi / 180) < α ∧ α < 270 * (Real.pi / 180)) : 
  α = 240 * (Real.pi / 180) :=
sorry

end trig_problem_l550_55082


namespace one_cow_one_bag_l550_55079

-- Define parameters
def cows : ℕ := 26
def bags : ℕ := 26
def days_for_all_cows : ℕ := 26

-- Theorem to prove the number of days for one cow to eat one bag of husk
theorem one_cow_one_bag (cows bags days_for_all_cows : ℕ) (h : cows = bags) (h2 : days_for_all_cows = 26) : days_for_one_cow_one_bag = 26 :=
by {
    sorry -- Proof to be filled in
}

end one_cow_one_bag_l550_55079


namespace percent_of_150_is_60_l550_55050

def percent_is_correct (Part Whole : ℝ) : Prop :=
  (Part / Whole) * 100 = 250

theorem percent_of_150_is_60 :
  percent_is_correct 150 60 :=
by
  sorry

end percent_of_150_is_60_l550_55050


namespace total_food_items_donated_l550_55086

def FosterFarmsDonation : ℕ := 45
def AmericanSummitsDonation : ℕ := 2 * FosterFarmsDonation
def HormelDonation : ℕ := 3 * FosterFarmsDonation
def BoudinButchersDonation : ℕ := HormelDonation / 3
def DelMonteFoodsDonation : ℕ := AmericanSummitsDonation - 30

theorem total_food_items_donated :
  FosterFarmsDonation + AmericanSummitsDonation + HormelDonation + BoudinButchersDonation + DelMonteFoodsDonation = 375 :=
by
  sorry

end total_food_items_donated_l550_55086


namespace geometric_sequence_solve_a1_l550_55022

noncomputable def geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
    (h2 : a 2 = 1) (h3 : a 3 * a 9 = 2 * (a 5 ^ 2)) :=
  a 1 = (Real.sqrt 2) / 2

-- Define the main statement
theorem geometric_sequence_solve_a1 (a : ℕ → ℝ) (q : ℝ)
    (hq : 0 < q) (ha2 : a 2 = 1) (ha3_ha9 : a 3 * a 9 = 2 * (a 5 ^ 2)) :
    a 1 = (Real.sqrt 2) / 2 :=
sorry  -- The proof will be written here

end geometric_sequence_solve_a1_l550_55022


namespace count_congruent_3_mod_8_l550_55087

theorem count_congruent_3_mod_8 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 300) :
  ∃ k : ℕ, (1 ≤ 8 * k + 3 ∧ 8 * k + 3 ≤ 300) ∧ n = 38 :=
by
  sorry

end count_congruent_3_mod_8_l550_55087


namespace owen_work_hours_l550_55010

def total_hours := 24
def chores_hours := 7
def sleep_hours := 11

theorem owen_work_hours : total_hours - chores_hours - sleep_hours = 6 := by
  sorry

end owen_work_hours_l550_55010


namespace tiger_time_to_pass_specific_point_l550_55001

theorem tiger_time_to_pass_specific_point :
  ∀ (distance_tree : ℝ) (time_tree : ℝ) (length_tiger : ℝ),
  distance_tree = 20 →
  time_tree = 5 →
  length_tiger = 5 →
  (length_tiger / (distance_tree / time_tree)) = 1.25 :=
by
  intros distance_tree time_tree length_tiger h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tiger_time_to_pass_specific_point_l550_55001


namespace geese_population_1996_l550_55004

theorem geese_population_1996 (k x : ℝ) 
  (h1 : x - 39 = k * 60) 
  (h2 : 123 - 60 = k * x) : 
  x = 84 := 
by
  sorry

end geese_population_1996_l550_55004


namespace sum_first_75_odd_numbers_l550_55026

theorem sum_first_75_odd_numbers : (75^2) = 5625 :=
by
  sorry

end sum_first_75_odd_numbers_l550_55026


namespace total_number_of_chips_l550_55031

theorem total_number_of_chips 
  (viviana_chocolate : ℕ) (susana_chocolate : ℕ) (viviana_vanilla : ℕ) (susana_vanilla : ℕ)
  (manuel_vanilla : ℕ) (manuel_chocolate : ℕ)
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : susana_chocolate = 25)
  (h5 : manuel_vanilla = 2 * susana_vanilla)
  (h6 : manuel_chocolate = viviana_chocolate / 2) :
  viviana_chocolate + susana_chocolate + manuel_chocolate + viviana_vanilla + susana_vanilla + manuel_vanilla = 135 :=
sorry

end total_number_of_chips_l550_55031


namespace max_value_of_8q_minus_9p_is_zero_l550_55061

theorem max_value_of_8q_minus_9p_is_zero (p : ℝ) (q : ℝ) (h1 : 0 < p) (h2 : p < 1) (hq : q = 3 * p ^ 2 - 2 * p ^ 3) : 
  8 * q - 9 * p ≤ 0 :=
by
  sorry

end max_value_of_8q_minus_9p_is_zero_l550_55061


namespace Emily_walks_more_distance_than_Troy_l550_55036

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end Emily_walks_more_distance_than_Troy_l550_55036


namespace problem_I_problem_II_l550_55006

-- Definitions
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (a - 2) * x - Real.log x

-- Problem (I)
theorem problem_I (a : ℝ) (h_min : ∀ x : ℝ, function_f a 1 ≤ function_f a x) :
  a = 1 ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → (function_f a x < function_f a 1)) ∧ (∀ x : ℝ, x > 1 → (function_f a x > function_f a 1)) :=
sorry

-- Problem (II)
theorem problem_II (a x0 : ℝ) (h_a_gt_1 : a > 1) (h_x0_pos : 0 < x0) (h_x0_lt_1 : x0 < 1)
    (h_min : ∀ x : ℝ, function_f a (1/a) ≤ function_f a x) :
  ∀ x : ℝ, function_f a 0 > 0
:= sorry

end problem_I_problem_II_l550_55006


namespace division_problem_l550_55068

theorem division_problem (D : ℕ) (Quotient Dividend Remainder : ℕ) 
    (h1 : Quotient = 36) 
    (h2 : Dividend = 3086) 
    (h3 : Remainder = 26) 
    (h_div : Dividend = (D * Quotient) + Remainder) : 
    D = 85 := 
by 
  -- Steps to prove the theorem will go here
  sorry

end division_problem_l550_55068


namespace domain_of_f_zeros_of_f_l550_55039

def log_a (a : ℝ) (x : ℝ) : ℝ := sorry -- Assume definition of logarithm base 'a'.

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_a a (2 - x)

theorem domain_of_f (a : ℝ) : ∀ x : ℝ, 2 - x > 0 ↔ x < 2 :=
by
  sorry

theorem zeros_of_f (a : ℝ) : f a 1 = 0 :=
by
  sorry

end domain_of_f_zeros_of_f_l550_55039


namespace average_marks_first_class_l550_55045

theorem average_marks_first_class (A : ℝ) :
  let students_class1 := 55
  let students_class2 := 48
  let avg_class2 := 58
  let avg_all := 59.067961165048544
  let total_students := 103
  let total_marks := avg_all * total_students
  total_marks = (A * students_class1) + (avg_class2 * students_class2) 
  → A = 60 :=
by
  sorry

end average_marks_first_class_l550_55045


namespace proof_statements_correct_l550_55053

variable (candidates : Nat) (sample_size : Nat)

def is_sampling_survey (survey_type : String) : Prop :=
  survey_type = "sampling"

def is_population (pop_size sample_size : Nat) : Prop :=
  (pop_size = 60000) ∧ (sample_size = 1000)

def is_sample (sample_size pop_size : Nat) : Prop :=
  sample_size < pop_size

def sample_size_correct (sample_size : Nat) : Prop :=
  sample_size = 1000

theorem proof_statements_correct :
  ∀ (survey_type : String) (pop_size sample_size : Nat),
  is_sampling_survey survey_type →
  is_population pop_size sample_size →
  is_sample sample_size pop_size →
  sample_size_correct sample_size →
  survey_type = "sampling" ∧
  pop_size = 60000 ∧
  sample_size = 1000 :=
by
  intros survey_type pop_size sample_size hs hp hsamp hsiz
  sorry

end proof_statements_correct_l550_55053


namespace g_at_6_l550_55089

def g (x : ℝ) : ℝ := 2 * x^4 - 13 * x^3 + 28 * x^2 - 32 * x - 48

theorem g_at_6 : g 6 = 552 :=
by sorry

end g_at_6_l550_55089


namespace race_distance_l550_55019

theorem race_distance (Va Vb Vc : ℝ) (D : ℝ) :
    (Va / Vb = 10 / 9) →
    (Va / Vc = 80 / 63) →
    (Vb / Vc = 8 / 7) →
    (D - 100) / D = 7 / 8 → 
    D = 700 :=
by
  intros h1 h2 h3 h4 
  sorry

end race_distance_l550_55019


namespace number_minus_six_l550_55067

variable (x : ℤ)

theorem number_minus_six
  (h : x / 5 = 2) : x - 6 = 4 := 
sorry

end number_minus_six_l550_55067


namespace number_of_triples_l550_55088

theorem number_of_triples : 
  {n : ℕ // ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ n = 4} :=
sorry

end number_of_triples_l550_55088


namespace perpendicular_line_through_circle_center_l550_55000

theorem perpendicular_line_through_circle_center :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x - 8 = 0 → x + 2*y = 0 → a * x + b * y + c = 0) ∧
  a = 2 ∧ b = -1 ∧ c = -2 :=
by
  sorry

end perpendicular_line_through_circle_center_l550_55000


namespace fraction_habitable_earth_l550_55098

theorem fraction_habitable_earth (one_fifth_land: ℝ) (one_third_inhabitable: ℝ)
  (h_land_fraction : one_fifth_land = 1 / 5)
  (h_inhabitable_fraction : one_third_inhabitable = 1 / 3) :
  (one_fifth_land * one_third_inhabitable) = 1 / 15 :=
by
  sorry

end fraction_habitable_earth_l550_55098


namespace ticket_cost_is_25_l550_55096

-- Define the given conditions
def num_tickets_first_show : ℕ := 200
def num_tickets_second_show : ℕ := 3 * num_tickets_first_show
def total_tickets : ℕ := num_tickets_first_show + num_tickets_second_show
def total_revenue_in_dollars : ℕ := 20000

-- Claim to prove
theorem ticket_cost_is_25 : ∃ x : ℕ, total_tickets * x = total_revenue_in_dollars ∧ x = 25 :=
by
  -- sorry is used here to skip the proof
  sorry

end ticket_cost_is_25_l550_55096


namespace derivative_of_x_log_x_l550_55033

noncomputable def y (x : ℝ) := x * Real.log x

theorem derivative_of_x_log_x (x : ℝ) (hx : 0 < x) :
  (deriv y x) = Real.log x + 1 :=
sorry

end derivative_of_x_log_x_l550_55033


namespace find_constants_l550_55008

theorem find_constants (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 → (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5))
  ↔ (A = -1 ∧ B = -1 ∧ C = 3) :=
by
  sorry

end find_constants_l550_55008


namespace integer_inequality_l550_55012

theorem integer_inequality (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := 
  sorry

end integer_inequality_l550_55012


namespace number_of_white_tiles_l550_55003

theorem number_of_white_tiles (n : ℕ) : 
  ∃ a_n : ℕ, a_n = 4 * n + 2 :=
sorry

end number_of_white_tiles_l550_55003


namespace probability_same_color_boxes_l550_55085

def num_neckties := 6
def num_shirts := 5
def num_hats := 4
def num_socks := 3

def num_common_colors := 3

def total_combinations : ℕ := num_neckties * num_shirts * num_hats * num_socks

def same_color_combinations : ℕ := num_common_colors

def same_color_probability : ℚ :=
  same_color_combinations / total_combinations

theorem probability_same_color_boxes :
  same_color_probability = 1 / 120 :=
  by
    -- Proof would go here
    sorry

end probability_same_color_boxes_l550_55085


namespace restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l550_55028

-- Let P be the original price of the jacket
variable (P : ℝ)

-- The price of the jacket after successive reductions
def price_after_discount (P : ℝ) : ℝ := 0.60 * P

-- The price of the jacket after all discounts including the limited-time offer
def price_after_full_discount (P : ℝ) : ℝ := 0.54 * P

-- Prove that to restore 0.60P back to P a 66.67% increase is needed
theorem restore_to_original_without_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.60 * P) * (1 + 66.67 / 100) = P :=
by sorry

-- Prove that to restore 0.54P back to P an 85.19% increase is needed
theorem restore_to_original_with_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.54 * P) * (1 + 85.19 / 100) = P :=
by sorry

end restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l550_55028


namespace area_of_enclosed_region_l550_55038

theorem area_of_enclosed_region :
  ∃ (r : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 5 = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2) ∧ (π * r^2 = 14 * π) := by
  sorry

end area_of_enclosed_region_l550_55038


namespace find_x_floor_l550_55016

theorem find_x_floor : ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 29 / 4 ∧ x = 29 / 4 := 
by
  sorry

end find_x_floor_l550_55016


namespace grocery_cost_l550_55080

/-- Potatoes and celery costs problem. -/
theorem grocery_cost (a b : ℝ) (potato_cost_per_kg celery_cost_per_kg : ℝ) 
(h1 : potato_cost_per_kg = 1) (h2 : celery_cost_per_kg = 0.7) :
  potato_cost_per_kg * a + celery_cost_per_kg * b = a + 0.7 * b :=
by
  rw [h1, h2]
  sorry

end grocery_cost_l550_55080


namespace average_score_of_all_matches_is_36_l550_55095

noncomputable def average_score_of_all_matches
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) : ℝ :=
  (x + y + a + b + c) / 5

theorem average_score_of_all_matches_is_36
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) :
  average_score_of_all_matches x y a b c h1 h2 h3x h3y h3a h3b h3c h4 = 36 := 
  by 
  sorry

end average_score_of_all_matches_is_36_l550_55095


namespace not_all_crows_gather_on_one_tree_l550_55009

theorem not_all_crows_gather_on_one_tree :
  ∀ (crows : Fin 6 → ℕ), 
  (∀ i, crows i = 1) →
  (∀ t1 t2, abs (t1 - t2) = 1 → crows t1 = crows t1 - 1 ∧ crows t2 = crows t2 + 1) →
  ¬(∃ i, crows i = 6 ∧ (∀ j ≠ i, crows j = 0)) :=
by
  sorry

end not_all_crows_gather_on_one_tree_l550_55009


namespace find_q_l550_55070

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by
  sorry

end find_q_l550_55070


namespace problem_I_problem_II_l550_55020

variable (t a : ℝ)

-- Problem (I)
theorem problem_I (h1 : a = 1) (h2 : t^2 - 5 * a * t + 4 * a^2 < 0) (h3 : (t - 2) * (t - 6) < 0) : 2 < t ∧ t < 4 := 
by 
  sorry   -- Proof omitted as per instructions

-- Problem (II)
theorem problem_II (h1 : (t - 2) * (t - 6) < 0 → t^2 - 5 * a * t + 4 * a^2 < 0) : 3 / 2 ≤ a ∧ a ≤ 2 :=
by 
  sorry   -- Proof omitted as per instructions

end problem_I_problem_II_l550_55020


namespace calc_g_f_3_l550_55040

def f (x : ℕ) : ℕ := x^3 + 3

def g (x : ℕ) : ℕ := 2 * x^2 + 3 * x + 2

theorem calc_g_f_3 : g (f 3) = 1892 := by
  sorry

end calc_g_f_3_l550_55040


namespace ratio_of_S_to_R_l550_55023

noncomputable def find_ratio (total_amount : ℕ) (diff_SP : ℕ) (n : ℕ) (k : ℕ) (P : ℕ) (Q : ℕ) (R : ℕ) (S : ℕ) (ratio_SR : ℕ) :=
  Q = n ∧ R = n ∧ P = k * n ∧ S = ratio_SR * n ∧ P + Q + R + S = total_amount ∧ S - P = diff_SP

theorem ratio_of_S_to_R :
  ∃ n k ratio_SR, k = 2 ∧ ratio_SR = 4 ∧ 
  find_ratio 1000 250 n k 250 125 125 500 ratio_SR :=
by
  sorry

end ratio_of_S_to_R_l550_55023


namespace quadruple_solution_l550_55062

theorem quadruple_solution (x y z w : ℝ) (h1: x + y + z + w = 0) (h2: x^7 + y^7 + z^7 + w^7 = 0) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨ (x = -y ∧ z = -w) ∨ (x = -z ∧ y = -w) ∨ (x = -w ∧ y = -z) :=
by
  sorry

end quadruple_solution_l550_55062


namespace total_seashells_l550_55027

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end total_seashells_l550_55027


namespace time_to_empty_is_109_89_hours_l550_55090

noncomputable def calculate_time_to_empty_due_to_leak : ℝ :=
  let R := 1 / 10 -- filling rate in tank/hour
  let Reffective := 1 / 11 -- effective filling rate in tank/hour
  let L := R - Reffective -- leak rate in tank/hour
  1 / L -- time to empty in hours

theorem time_to_empty_is_109_89_hours : calculate_time_to_empty_due_to_leak = 109.89 :=
by
  rw [calculate_time_to_empty_due_to_leak]
  sorry -- Proof steps can be filled in later

end time_to_empty_is_109_89_hours_l550_55090


namespace solve_system_of_equations_l550_55091

theorem solve_system_of_equations :
  ∃ y : ℝ, (2 * 2 + y = 0) ∧ (2 + y = 3) :=
by
  sorry

end solve_system_of_equations_l550_55091


namespace mason_courses_not_finished_l550_55052

-- Each necessary condition is listed as a definition.
def coursesPerWall := 6
def bricksPerCourse := 10
def numOfWalls := 4
def totalBricksUsed := 220

-- Creating an entity to store the problem and prove it.
theorem mason_courses_not_finished : 
  (numOfWalls * coursesPerWall * bricksPerCourse - totalBricksUsed) / bricksPerCourse = 2 := 
by
  sorry

end mason_courses_not_finished_l550_55052


namespace loan_repayment_l550_55014

open Real

theorem loan_repayment
  (a r : ℝ) (h_r : 0 ≤ r) :
  ∃ x : ℝ, 
    x = (a * r * (1 + r)^5) / ((1 + r)^5 - 1) :=
sorry

end loan_repayment_l550_55014


namespace jimmy_eats_7_cookies_l550_55092

def cookies_and_calories (c: ℕ) : Prop :=
  50 * c + 150 = 500

theorem jimmy_eats_7_cookies : cookies_and_calories 7 :=
by {
  -- This would be where the proof steps go, but we replace it with:
  sorry
}

end jimmy_eats_7_cookies_l550_55092


namespace find_g_at_1_l550_55059

theorem find_g_at_1 (g : ℝ → ℝ) (h : ∀ x, x ≠ 1/2 → g x + g ((2*x + 1)/(1 - 2*x)) = x) : 
  g 1 = 15 / 7 :=
sorry

end find_g_at_1_l550_55059


namespace equilibrium_constant_l550_55081

theorem equilibrium_constant (C_NO2 C_O2 C_NO : ℝ) (h_NO2 : C_NO2 = 0.4) (h_O2 : C_O2 = 0.3) (h_NO : C_NO = 0.2) :
  (C_NO2^2 / (C_O2 * C_NO^2)) = 13.3 := by
  rw [h_NO2, h_O2, h_NO]
  sorry

end equilibrium_constant_l550_55081


namespace dash_cam_mounts_max_profit_l550_55073

noncomputable def monthly_profit (x t : ℝ) : ℝ :=
  (48 + t / (2 * x)) * x - 32 * x - 3 - t

theorem dash_cam_mounts_max_profit :
  ∃ (x t : ℝ), 1 < x ∧ x < 3 ∧ x = 3 - 2 / (t + 1) ∧
  monthly_profit x t = 37.5 := by
sorry

end dash_cam_mounts_max_profit_l550_55073


namespace sticker_ratio_l550_55066

variable (Dan Tom Bob : ℕ)

theorem sticker_ratio 
  (h1 : Dan = 2 * Tom) 
  (h2 : Tom = Bob) 
  (h3 : Bob = 12) 
  (h4 : Dan = 72) : 
  Tom = Bob :=
by
  sorry

end sticker_ratio_l550_55066


namespace puppies_per_cage_l550_55042

-- Conditions
variables (total_puppies sold_puppies cages initial_puppies per_cage : ℕ)
variables (h_total : total_puppies = 13)
variables (h_sold : sold_puppies = 7)
variables (h_cages : cages = 3)
variables (h_equal_cages : total_puppies - sold_puppies = cages * per_cage)

-- Question
theorem puppies_per_cage :
  per_cage = 2 :=
by {
  sorry
}

end puppies_per_cage_l550_55042


namespace expected_number_of_socks_l550_55057

noncomputable def expected_socks_to_pick (n : ℕ) : ℚ := (2 * (n + 1)) / 3

theorem expected_number_of_socks (n : ℕ) (h : n ≥ 2) : 
  (expected_socks_to_pick n) = (2 * (n + 1)) / 3 := 
by
  sorry

end expected_number_of_socks_l550_55057


namespace no_valid_a_exists_l550_55099

theorem no_valid_a_exists 
  (a : ℝ)
  (h1: ∀ x : ℝ, x^2 + 2*(a+1)*x - (a-1) = 0 → (1 < x ∨ x < 1)) :
  false := by
  sorry

end no_valid_a_exists_l550_55099


namespace part1_solution_set_part2_range_of_a_l550_55072

noncomputable def f (x a : ℝ) : ℝ := -x^2 + a * x + 4

def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem part1_solution_set (a : ℝ := 1) :
  {x : ℝ | f x a ≥ g x} = { x : ℝ | -1 ≤ x ∧ x ≤ (Real.sqrt 17 - 1) / 2 } :=
by
  sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x ∈ [-1,1], f x a ≥ g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end part1_solution_set_part2_range_of_a_l550_55072


namespace decryption_correct_l550_55071

theorem decryption_correct (a b : ℤ) (h1 : a - 2 * b = 1) (h2 : 2 * a + b = 7) : a = 3 ∧ b = 1 :=
by
  sorry

end decryption_correct_l550_55071


namespace total_movies_shown_l550_55083

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end total_movies_shown_l550_55083


namespace fish_per_bowl_l550_55018

theorem fish_per_bowl : 6003 / 261 = 23 := by
  sorry

end fish_per_bowl_l550_55018


namespace find_the_number_l550_55046

theorem find_the_number :
  ∃ x : ℤ, 65 + (x * 12) / (180 / 3) = 66 ∧ x = 5 :=
by
  existsi (5 : ℤ)
  sorry

end find_the_number_l550_55046


namespace find_reciprocal_sum_of_roots_l550_55024

theorem find_reciprocal_sum_of_roots
  {x₁ x₂ : ℝ}
  (h1 : 5 * x₁ ^ 2 - 3 * x₁ - 2 = 0)
  (h2 : 5 * x₂ ^ 2 - 3 * x₂ - 2 = 0)
  (h_diff : x₁ ≠ x₂) :
  (1 / x₁ + 1 / x₂) = -3 / 2 :=
by {
  sorry
}

end find_reciprocal_sum_of_roots_l550_55024


namespace problem_statement_false_adjacent_complementary_l550_55097

-- Definition of straight angle, supplementary angles, and complementary angles.
def is_straight_angle (θ : ℝ) : Prop := θ = 180
def are_supplementary (θ ψ : ℝ) : Prop := θ + ψ = 180
def are_complementary (θ ψ : ℝ) : Prop := θ + ψ = 90

-- Definition of adjacent angles (for completeness, though we don't use adjacency differently right now)
def are_adjacent (θ ψ : ℝ) : Prop := ∀ x, θ + x + ψ + x = θ + ψ -- Simplified

-- Additional conditions that could be true or false -- we need one of them to be false.
def false_statement_D (θ ψ : ℝ) : Prop :=
  are_complementary θ ψ → are_adjacent θ ψ

theorem problem_statement_false_adjacent_complementary :
  ∃ (θ ψ : ℝ), ¬ false_statement_D θ ψ :=
by
  sorry

end problem_statement_false_adjacent_complementary_l550_55097


namespace ratio_e_a_l550_55037

theorem ratio_e_a (a b c d e : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4) :
  e / a = 8 / 15 := 
by
  sorry

end ratio_e_a_l550_55037


namespace exists_h_not_divisible_l550_55005

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end exists_h_not_divisible_l550_55005


namespace number_of_students_l550_55025

theorem number_of_students 
  (N : ℕ)
  (avg_age : ℕ → ℕ)
  (h1 : avg_age N = 15)
  (h2 : avg_age 5 = 12)
  (h3 : avg_age 9 = 16)
  (h4 : N = 15 ∧ avg_age 1 = 21) : 
  N = 15 :=
by
  sorry

end number_of_students_l550_55025


namespace knights_win_35_l550_55056

noncomputable def Sharks : ℕ := sorry
noncomputable def Falcons : ℕ := sorry
noncomputable def Knights : ℕ := 35
noncomputable def Wolves : ℕ := sorry
noncomputable def Royals : ℕ := sorry

-- Conditions
axiom h1 : Sharks > Falcons
axiom h2 : Wolves > 25
axiom h3 : Wolves < Knights ∧ Knights < Royals

-- Prove: Knights won 35 games
theorem knights_win_35 : Knights = 35 := 
by sorry

end knights_win_35_l550_55056


namespace impossible_odd_n_m_l550_55058

theorem impossible_odd_n_m (n m : ℤ) (h : Even (n^2 + m + n * m)) : ¬ (Odd n ∧ Odd m) :=
by
  intro h1
  sorry

end impossible_odd_n_m_l550_55058


namespace revised_lemonade_calories_l550_55084

def lemonade (lemon_grams sugar_grams water_grams lemon_calories_per_50grams sugar_calories_per_100grams : ℕ) :=
  let lemon_cals := lemon_calories_per_50grams
  let sugar_cals := (sugar_grams / 100) * sugar_calories_per_100grams
  let water_cals := 0
  lemon_cals + sugar_cals + water_cals

def lemonade_weight (lemon_grams sugar_grams water_grams : ℕ) :=
  lemon_grams + sugar_grams + water_grams

def caloric_density (total_calories : ℕ) (total_weight : ℕ) := (total_calories : ℚ) / total_weight

def calories_in_serving (density : ℚ) (serving : ℕ) := density * serving

theorem revised_lemonade_calories :
  let lemon_calories := 32
  let sugar_calories := 579
  let total_calories := lemonade 50 150 300 lemon_calories sugar_calories
  let total_weight := lemonade_weight 50 150 300
  let density := caloric_density total_calories total_weight
  let serving_calories := calories_in_serving density 250
  serving_calories = 305.5 := sorry

end revised_lemonade_calories_l550_55084


namespace find_fraction_B_minus_1_over_A_l550_55077

variable (A B : ℝ) (a_n S_n : ℕ → ℝ)
variable (h1 : ∀ n, a_n n + S_n n = A * (n ^ 2) + B * n + 1)
variable (h2 : A ≠ 0)

theorem find_fraction_B_minus_1_over_A : (B - 1) / A = 3 := by
  sorry

end find_fraction_B_minus_1_over_A_l550_55077


namespace jane_mean_after_extra_credit_l550_55069

-- Define Jane's original scores
def original_scores : List ℤ := [82, 90, 88, 95, 91]

-- Define the extra credit points
def extra_credit : ℤ := 2

-- Define the mean calculation after extra credit
def mean_after_extra_credit (scores : List ℤ) (extra : ℤ) : ℚ :=
  let total_sum := scores.sum + (scores.length * extra)
  total_sum / scores.length

theorem jane_mean_after_extra_credit :
  mean_after_extra_credit original_scores extra_credit = 91.2 := by
  sorry

end jane_mean_after_extra_credit_l550_55069


namespace cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l550_55044

theorem cos_alpha_plus_5pi_over_12_eq_neg_1_over_3
  (α : ℝ)
  (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l550_55044


namespace min_spend_for_free_delivery_l550_55064

theorem min_spend_for_free_delivery : 
  let chicken_price := 1.5 * 6.00
  let lettuce_price := 3.00
  let tomato_price := 2.50
  let sweet_potato_price := 4 * 0.75
  let broccoli_price := 2 * 2.00
  let brussel_sprouts_price := 2.50
  let current_total := chicken_price + lettuce_price + tomato_price + sweet_potato_price + broccoli_price + brussel_sprouts_price
  let additional_needed := 11.00 
  let minimum_spend := current_total + additional_needed
  minimum_spend = 35.00 :=
by
  sorry

end min_spend_for_free_delivery_l550_55064


namespace work_days_together_l550_55035

theorem work_days_together (p_rate q_rate : ℝ) (fraction_left : ℝ) (d : ℝ) 
  (h₁ : p_rate = 1/15) (h₂ : q_rate = 1/20) (h₃ : fraction_left = 8/15)
  (h₄ : (p_rate + q_rate) * d = 1 - fraction_left) : d = 4 :=
by
  sorry

end work_days_together_l550_55035


namespace swimming_pool_surface_area_l550_55055

def length : ℝ := 20
def width : ℝ := 15

theorem swimming_pool_surface_area : length * width = 300 := 
by
  -- The mathematical proof would go here; we'll skip it with "sorry" per instructions.
  sorry

end swimming_pool_surface_area_l550_55055


namespace perpendicular_bisector_eq_l550_55047

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end perpendicular_bisector_eq_l550_55047


namespace Janet_saves_154_minutes_per_week_l550_55032

-- Definitions for the time spent on each activity daily
def timeLookingForKeys := 8 -- minutes
def timeComplaining := 3 -- minutes
def timeSearchingForPhone := 5 -- minutes
def timeLookingForWallet := 4 -- minutes
def timeSearchingForSunglasses := 2 -- minutes

-- Total time spent daily on these activities
def totalDailyTime := timeLookingForKeys + timeComplaining + timeSearchingForPhone + timeLookingForWallet + timeSearchingForSunglasses
-- Time savings calculation for a week
def weeklySaving := totalDailyTime * 7

-- The proof statement that Janet will save 154 minutes every week
theorem Janet_saves_154_minutes_per_week : weeklySaving = 154 := by
  sorry

end Janet_saves_154_minutes_per_week_l550_55032


namespace sequence_strictly_monotonic_increasing_l550_55043

noncomputable def a (n : ℕ) : ℝ := ((n + 1) ^ n * n ^ (2 - n)) / (7 * n ^ 2 + 1)

theorem sequence_strictly_monotonic_increasing :
  ∀ n : ℕ, a n < a (n + 1) := 
by {
  sorry
}

end sequence_strictly_monotonic_increasing_l550_55043


namespace columns_contain_all_numbers_l550_55048

def rearrange (n m k : ℕ) (a : ℕ → ℕ) : ℕ → ℕ :=
  λ i => if i < n - m then a (i + m + 1)
         else if i < n - k - m then a (i - (n - m) + k + 1)
         else a (i - (n - k))

theorem columns_contain_all_numbers
  (n m k: ℕ)
  (h1 : n > 0)
  (h2 : m < n)
  (h3 : k < n)
  (a : ℕ → ℕ)
  (h4 : ∀ i : ℕ, i < n → a i = i + 1) :
  ∀ j : ℕ, j < n → ∃ i : ℕ, i < n ∧ rearrange n m k a i = j + 1 :=
by
  sorry

end columns_contain_all_numbers_l550_55048


namespace difference_of_numbers_l550_55007

theorem difference_of_numbers (x y : ℕ) (h₁ : x + y = 50) (h₂ : Nat.gcd x y = 5) :
  (x - y = 20 ∨ y - x = 20 ∨ x - y = 40 ∨ y - x = 40) :=
sorry

end difference_of_numbers_l550_55007


namespace total_angles_sum_l550_55013

variables (A B C D E : Type)
variables (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ)

-- Conditions about the geometry
axiom angle_triangle_ABC : angle1 + angle2 + angle3 = 180
axiom angle_triangle_BDE : angle7 + angle4 + angle5 = 180
axiom shared_angle_B : angle2 + angle7 = 180 -- since they form a straight line at vertex B

-- Proof statement
theorem total_angles_sum (A B C D E : Type) (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle7 - 180 = 180 :=
by
  sorry

end total_angles_sum_l550_55013


namespace cheryl_walking_speed_l550_55002

theorem cheryl_walking_speed (H : 12 = 6 * v) : v = 2 := 
by
  -- proof here
  sorry

end cheryl_walking_speed_l550_55002


namespace name_tag_area_l550_55029

-- Define the side length of the square
def side_length : ℕ := 11

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- State the theorem: the area of a square with side length of 11 cm is 121 cm²
theorem name_tag_area : square_area side_length = 121 :=
by
  sorry

end name_tag_area_l550_55029


namespace rose_bushes_after_work_l550_55015

def initial_rose_bushes := 2
def planned_rose_bushes := 4
def planting_rate := 3
def removed_rose_bushes := 5

theorem rose_bushes_after_work :
  initial_rose_bushes + (planned_rose_bushes * planting_rate) - removed_rose_bushes = 9 :=
by
  sorry

end rose_bushes_after_work_l550_55015


namespace range_of_a_for_function_min_max_l550_55094

theorem range_of_a_for_function_min_max 
  (a : ℝ) 
  (h_min : ∀ x ∈ [-1, 1], x = -1 → x^2 + a * x + 3 ≤ y) 
  (h_max : ∀ x ∈ [-1, 1], x = 1 → x^2 + a * x + 3 ≥ y) : 
  2 ≤ a := 
sorry

end range_of_a_for_function_min_max_l550_55094


namespace clothes_donation_l550_55017

variable (initial_clothes : ℕ)
variable (clothes_thrown_away : ℕ)
variable (final_clothes : ℕ)
variable (x : ℕ)

theorem clothes_donation (h1 : initial_clothes = 100) 
                        (h2 : clothes_thrown_away = 15) 
                        (h3 : final_clothes = 65) 
                        (h4 : 4 * x = initial_clothes - final_clothes - clothes_thrown_away) :
  x = 5 := by
  sorry

end clothes_donation_l550_55017


namespace number_for_B_expression_l550_55011

-- Define the number for A as a variable
variable (a : ℤ)

-- Define the number for B in terms of a
def number_for_B (a : ℤ) : ℤ := 2 * a - 1

-- Statement to prove
theorem number_for_B_expression (a : ℤ) : number_for_B a = 2 * a - 1 := by
  sorry

end number_for_B_expression_l550_55011


namespace problem_statement_l550_55034

variable (x : ℝ)

-- Definitions based on the conditions
def a := 2005 * x + 2009
def b := 2005 * x + 2010
def c := 2005 * x + 2011

-- Assertion for the problem
theorem problem_statement : a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = 3 := by
  sorry

end problem_statement_l550_55034


namespace max_side_length_l550_55021

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l550_55021


namespace train_length_is_correct_l550_55074

-- Defining the initial conditions
def train_speed_km_per_hr : Float := 90.0
def time_seconds : Float := 5.0

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : Float) : Float :=
  speed_km_per_hr * (1000.0 / 3600.0)

-- Calculate the length of the train in meters
def length_of_train (speed_km_per_hr : Float) (time_s : Float) : Float :=
  km_per_hr_to_m_per_s speed_km_per_hr * time_s

-- Theorem statement
theorem train_length_is_correct : length_of_train train_speed_km_per_hr time_seconds = 125.0 :=
by
  sorry

end train_length_is_correct_l550_55074


namespace smallest_n_congruence_l550_55065

theorem smallest_n_congruence :
  ∃ n : ℕ+, 537 * (n : ℕ) % 30 = 1073 * (n : ℕ) % 30 ∧ (∀ m : ℕ+, 537 * (m : ℕ) % 30 = 1073 * (m : ℕ) % 30 → (m : ℕ) < n → false) :=
  sorry

end smallest_n_congruence_l550_55065


namespace stream_current_rate_proof_l550_55041

noncomputable def stream_current_rate (c : ℝ) : Prop :=
  ∃ (c : ℝ), (6 / (8 - c) + 6 / (8 + c) = 2) ∧ c = 4

theorem stream_current_rate_proof : stream_current_rate 4 :=
by {
  -- Proof to be provided here.
  sorry
}

end stream_current_rate_proof_l550_55041


namespace plain_b_area_l550_55078

theorem plain_b_area : 
  ∃ x : ℕ, (x + (x - 50) = 350) ∧ x = 200 :=
by
  sorry

end plain_b_area_l550_55078


namespace number_of_people_in_group_is_21_l550_55049

-- Definitions based directly on the conditions
def pins_contribution_per_day := 10
def pins_deleted_per_week_per_person := 5
def group_initial_pins := 1000
def final_pins_after_month := 6600
def weeks_in_a_month := 4

-- To be proved: number of people in the group is 21
theorem number_of_people_in_group_is_21 (P : ℕ)
  (h1 : final_pins_after_month - group_initial_pins = 5600)
  (h2 : weeks_in_a_month * (pins_contribution_per_day * 7 - pins_deleted_per_week_per_person) = 260)
  (h3 : 5600 / 260 = 21) :
  P = 21 := 
sorry

end number_of_people_in_group_is_21_l550_55049


namespace intersection_M_N_l550_55051

noncomputable def M : Set ℝ := { x | x^2 + x - 2 = 0 }
def N : Set ℝ := { x | x < 0 }

theorem intersection_M_N : M ∩ N = { -2 } := by
  sorry

end intersection_M_N_l550_55051


namespace sunny_lead_l550_55030

-- Define the context of the race
variables {s m : ℝ}  -- s: Sunny's speed, m: Misty's speed
variables (distance_first : ℝ) (distance_ahead_first : ℝ)
variables (additional_distance_sunny_second : ℝ) (correct_answer : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first = 400 ∧
  distance_ahead_first = 20 ∧
  additional_distance_sunny_second = 40 ∧
  correct_answer = 20 

-- The math proof problem in Lean 4
theorem sunny_lead (h : conditions distance_first distance_ahead_first additional_distance_sunny_second correct_answer) :
  ∀ s m : ℝ, s / m = (400 / 380 : ℝ) → 
  (s / m) * 400 + additional_distance_sunny_second = (m / s) * 440 + correct_answer :=
sorry

end sunny_lead_l550_55030
