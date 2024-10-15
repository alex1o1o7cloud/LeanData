import Mathlib

namespace NUMINAMATH_GPT_remainder_of_11_pow_2023_mod_33_l820_82092

theorem remainder_of_11_pow_2023_mod_33 : (11 ^ 2023) % 33 = 11 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_11_pow_2023_mod_33_l820_82092


namespace NUMINAMATH_GPT_concentration_third_flask_l820_82004

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end NUMINAMATH_GPT_concentration_third_flask_l820_82004


namespace NUMINAMATH_GPT_ratio_of_edges_l820_82034

theorem ratio_of_edges
  (V₁ V₂ : ℝ)
  (a b : ℝ)
  (hV : V₁ / V₂ = 8 / 1)
  (hV₁ : V₁ = a^3)
  (hV₂ : V₂ = b^3) :
  a / b = 2 / 1 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_edges_l820_82034


namespace NUMINAMATH_GPT_evaluate_expression_l820_82009

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l820_82009


namespace NUMINAMATH_GPT_grooming_time_correct_l820_82036

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end NUMINAMATH_GPT_grooming_time_correct_l820_82036


namespace NUMINAMATH_GPT_negation_of_forall_x_gt_1_l820_82040

theorem negation_of_forall_x_gt_1 : ¬(∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by
  sorry

end NUMINAMATH_GPT_negation_of_forall_x_gt_1_l820_82040


namespace NUMINAMATH_GPT_initial_slices_ham_l820_82013

def total_sandwiches : ℕ := 50
def slices_per_sandwich : ℕ := 3
def additional_slices_needed : ℕ := 119

-- Calculate the total number of slices needed to make 50 sandwiches.
def total_slices_needed : ℕ := total_sandwiches * slices_per_sandwich

-- Prove the initial number of slices of ham Anna has.
theorem initial_slices_ham : total_slices_needed - additional_slices_needed = 31 := by
  sorry

end NUMINAMATH_GPT_initial_slices_ham_l820_82013


namespace NUMINAMATH_GPT_percentage_material_B_in_final_mixture_l820_82065

-- Conditions
def percentage_material_A_in_Solution_X : ℝ := 20
def percentage_material_B_in_Solution_X : ℝ := 80
def percentage_material_A_in_Solution_Y : ℝ := 30
def percentage_material_B_in_Solution_Y : ℝ := 70
def percentage_material_A_in_final_mixture : ℝ := 22

-- Goal
theorem percentage_material_B_in_final_mixture :
  100 - percentage_material_A_in_final_mixture = 78 := by
  sorry

end NUMINAMATH_GPT_percentage_material_B_in_final_mixture_l820_82065


namespace NUMINAMATH_GPT_total_board_length_l820_82022

-- Defining the lengths of the pieces of the board
def shorter_piece_length : ℕ := 23
def longer_piece_length : ℕ := 2 * shorter_piece_length

-- Stating the theorem that the total length of the board is 69 inches
theorem total_board_length : shorter_piece_length + longer_piece_length = 69 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_GPT_total_board_length_l820_82022


namespace NUMINAMATH_GPT_radius_of_sphere_eq_l820_82035

theorem radius_of_sphere_eq (r : ℝ) : 
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_sphere_eq_l820_82035


namespace NUMINAMATH_GPT_boats_seating_problem_l820_82085

theorem boats_seating_problem 
  (total_boats : ℕ) (total_people : ℕ) 
  (big_boat_seats : ℕ) (small_boat_seats : ℕ) 
  (b s : ℕ) 
  (h1 : total_boats = 12) 
  (h2 : total_people = 58) 
  (h3 : big_boat_seats = 6) 
  (h4 : small_boat_seats = 4) 
  (h5 : b + s = 12) 
  (h6 : b * 6 + s * 4 = 58) 
  : b = 5 ∧ s = 7 :=
sorry

end NUMINAMATH_GPT_boats_seating_problem_l820_82085


namespace NUMINAMATH_GPT_solution_point_satisfies_inequalities_l820_82078

theorem solution_point_satisfies_inequalities:
  let x := -1/3
  let y := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧ x - 4 * y ≤ -3 :=
by
  let x := -1/3
  let y := 2/3
  sorry

end NUMINAMATH_GPT_solution_point_satisfies_inequalities_l820_82078


namespace NUMINAMATH_GPT_carolyn_total_monthly_practice_l820_82014

-- Define the constants and relationships given in the problem
def daily_piano_practice : ℕ := 20
def times_violin_practice : ℕ := 3
def days_week : ℕ := 6
def weeks_month : ℕ := 4
def daily_violin_practice : ℕ := daily_piano_practice * times_violin_practice
def total_daily_practice : ℕ := daily_piano_practice + daily_violin_practice
def weekly_practice_time : ℕ := total_daily_practice * days_week
def monthly_practice_time : ℕ := weekly_practice_time * weeks_month

-- The proof statement with the final result
theorem carolyn_total_monthly_practice : monthly_practice_time = 1920 := by
  sorry

end NUMINAMATH_GPT_carolyn_total_monthly_practice_l820_82014


namespace NUMINAMATH_GPT_solve_for_f_sqrt_2_l820_82066

theorem solve_for_f_sqrt_2 (f : ℝ → ℝ) (h : ∀ x, f x = 2 / (2 - x)) : f (Real.sqrt 2) = 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_f_sqrt_2_l820_82066


namespace NUMINAMATH_GPT_basic_astrophysics_budget_percent_l820_82083

theorem basic_astrophysics_budget_percent
  (total_degrees : ℝ := 360)
  (astrophysics_degrees : ℝ := 108) :
  (astrophysics_degrees / total_degrees) * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_basic_astrophysics_budget_percent_l820_82083


namespace NUMINAMATH_GPT_river_width_l820_82002

def bridge_length : ℕ := 295
def additional_length : ℕ := 192
def total_width : ℕ := 487

theorem river_width (h1 : bridge_length = 295) (h2 : additional_length = 192) : bridge_length + additional_length = total_width := by
  sorry

end NUMINAMATH_GPT_river_width_l820_82002


namespace NUMINAMATH_GPT_number_of_zeros_of_h_l820_82003

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 3 - x^2
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem number_of_zeros_of_h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 = 0 ∧ h x2 = 0 ∧ ∀ x, h x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_of_h_l820_82003


namespace NUMINAMATH_GPT_typing_lines_in_10_minutes_l820_82041

def programmers := 10
def total_lines_in_60_minutes := 60
def total_minutes := 60
def target_minutes := 10

theorem typing_lines_in_10_minutes :
  (total_lines_in_60_minutes / total_minutes) * programmers * target_minutes = 100 :=
by sorry

end NUMINAMATH_GPT_typing_lines_in_10_minutes_l820_82041


namespace NUMINAMATH_GPT_total_marbles_l820_82021

-- Definitions based on given conditions
def ratio_white := 2
def ratio_purple := 3
def ratio_red := 5
def ratio_blue := 4
def ratio_green := 6
def blue_marbles := 24

-- Definition of sum of ratio parts
def sum_of_ratio_parts := ratio_white + ratio_purple + ratio_red + ratio_blue + ratio_green

-- Definition of ratio of blue marbles to total
def ratio_blue_to_total := ratio_blue / sum_of_ratio_parts

-- Proof goal: total number of marbles
theorem total_marbles : blue_marbles / ratio_blue_to_total = 120 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l820_82021


namespace NUMINAMATH_GPT_unique_solution_for_system_l820_82025

theorem unique_solution_for_system (a : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 4 * y = 0 ∧ x + a * y + a * z - a = 0 →
    (a = 2 ∨ a = -2)) :=
by
  intros x y z h
  sorry

end NUMINAMATH_GPT_unique_solution_for_system_l820_82025


namespace NUMINAMATH_GPT_find_k_value_l820_82080

theorem find_k_value (x k : ℝ) (h : x = 2) (h_sol : (k / (x - 3)) - (1 / (3 - x)) = 1) : k = -2 :=
by
  -- sorry to suppress the actual proof
  sorry

end NUMINAMATH_GPT_find_k_value_l820_82080


namespace NUMINAMATH_GPT_f_is_odd_l820_82086

open Real

def f (x : ℝ) : ℝ := x^3 + x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_f_is_odd_l820_82086


namespace NUMINAMATH_GPT_first_investment_percentage_l820_82008

variable (P : ℝ)
variable (x : ℝ := 1400)  -- investment amount in the first investment
variable (y : ℝ := 600)   -- investment amount at 8 percent
variable (income_difference : ℝ := 92)
variable (total_investment : ℝ := 2000)
variable (rate_8_percent : ℝ := 0.08)
variable (exceed_by : ℝ := 92)

theorem first_investment_percentage :
  P * x - rate_8_percent * y = exceed_by →
  total_investment = x + y →
  P = 0.10 :=
by
  -- Solution steps can be filled here if needed
  sorry

end NUMINAMATH_GPT_first_investment_percentage_l820_82008


namespace NUMINAMATH_GPT_problem_statement_l820_82016

def h (x : ℝ) : ℝ := 3 * x + 2
def k (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (h (k (h 3))) / (k (h (k 3))) = 59 / 19 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l820_82016


namespace NUMINAMATH_GPT_data_variance_l820_82026

def data : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

theorem data_variance : variance data = 0.02 := by
  sorry

end NUMINAMATH_GPT_data_variance_l820_82026


namespace NUMINAMATH_GPT_star_equiv_l820_82062

variable {m n x y : ℝ}

def star (m n : ℝ) : ℝ := (3 * m - 2 * n) ^ 2

theorem star_equiv (x y : ℝ) : star ((3 * x - 2 * y) ^ 2) ((2 * y - 3 * x) ^ 2) = (3 * x - 2 * y) ^ 4 := 
by
  sorry

end NUMINAMATH_GPT_star_equiv_l820_82062


namespace NUMINAMATH_GPT_find_all_real_solutions_l820_82050

theorem find_all_real_solutions (x : ℝ) :
    (1 / ((x - 1) * (x - 2))) + (1 / ((x - 2) * (x - 3))) + (1 / ((x - 3) * (x - 4))) + (1 / ((x - 4) * (x - 5))) = 1 / 4 →
    x = 1 ∨ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_all_real_solutions_l820_82050


namespace NUMINAMATH_GPT_minimum_value_l820_82018

/-- 
Given \(a > 0\), \(b > 0\), and \(a + 2b = 1\),
prove that the minimum value of \(\frac{2}{a} + \frac{1}{b}\) is 8.
-/
theorem minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) : 
  (∀ a b : ℝ, (a > 0) → (b > 0) → (a + 2 * b = 1) → (∃ c : ℝ, c = 8 ∧ ∀ x y : ℝ, (x = a) → (y = b) → (c ≤ (2 / x) + (1 / y)))) :=
sorry

end NUMINAMATH_GPT_minimum_value_l820_82018


namespace NUMINAMATH_GPT_daisy_count_per_bouquet_l820_82027

-- Define the conditions
def roses_per_bouquet := 12
def total_bouquets := 20
def rose_bouquets := 10
def daisy_bouquets := total_bouquets - rose_bouquets
def total_flowers_sold := 190
def total_roses_sold := rose_bouquets * roses_per_bouquet
def total_daisies_sold := total_flowers_sold - total_roses_sold

-- Define the problem: prove that the number of daisies per bouquet is 7
theorem daisy_count_per_bouquet : total_daisies_sold / daisy_bouquets = 7 := by
  sorry

end NUMINAMATH_GPT_daisy_count_per_bouquet_l820_82027


namespace NUMINAMATH_GPT_triangle_area_l820_82051

theorem triangle_area (base height : ℝ) (h_base : base = 8.4) (h_height : height = 5.8) :
  0.5 * base * height = 24.36 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l820_82051


namespace NUMINAMATH_GPT_volume_of_cuboid_is_250_cm3_l820_82088

-- Define the edge length of the cube
def edge_length (a : ℕ) : ℕ := 5

-- Define the volume of a single cube
def cube_volume := (edge_length 5) ^ 3

-- Define the total volume of the cuboid formed by placing two such cubes in a line
def cuboid_volume := 2 * cube_volume

-- Theorem stating the volume of the cuboid formed
theorem volume_of_cuboid_is_250_cm3 : cuboid_volume = 250 := by
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_is_250_cm3_l820_82088


namespace NUMINAMATH_GPT_tagged_fish_in_second_catch_l820_82072

theorem tagged_fish_in_second_catch (N : ℕ) (initially_tagged second_catch : ℕ)
  (h1 : N = 1250)
  (h2 : initially_tagged = 50)
  (h3 : second_catch = 50) :
  initially_tagged / N * second_catch = 2 :=
by
  sorry

end NUMINAMATH_GPT_tagged_fish_in_second_catch_l820_82072


namespace NUMINAMATH_GPT_average_income_N_O_l820_82033

variable (M N O : ℝ)

-- Condition declaration
def condition1 : Prop := M + N = 10100
def condition2 : Prop := M + O = 10400
def condition3 : Prop := M = 4000

-- Theorem statement
theorem average_income_N_O (h1 : condition1 M N) (h2 : condition2 M O) (h3 : condition3 M) :
  (N + O) / 2 = 6250 :=
sorry

end NUMINAMATH_GPT_average_income_N_O_l820_82033


namespace NUMINAMATH_GPT_min_number_of_lucky_weights_l820_82099

-- Definitions and conditions
def weight (n: ℕ) := n -- A weight is represented as a natural number.

def is_lucky (weights: Finset ℕ) (w: ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a ≠ b ∧ w = a + b
-- w is "lucky" if it's the sum of two other distinct weights in the set.

def min_lucky_guarantee (weights: Finset ℕ) (k: ℕ) : Prop :=
  ∀ (w1 w2 : ℕ), w1 ∈ weights ∧ w2 ∈ weights →
    ∃ (lucky_weights : Finset ℕ), lucky_weights.card = k ∧
    (is_lucky weights w1 ∧ is_lucky weights w2 ∧ (w1 ≥ 3 * w2 ∨ w2 ≥ 3 * w1))
-- The minimum number k of "lucky" weights ensures there exist two weights 
-- such that their masses differ by at least a factor of three.

-- The theorem to be proven
theorem min_number_of_lucky_weights (weights: Finset ℕ) (h_distinct: weights.card = 100) :
  ∃ k, min_lucky_guarantee weights k ∧ k = 87 := 
sorry

end NUMINAMATH_GPT_min_number_of_lucky_weights_l820_82099


namespace NUMINAMATH_GPT_line_intersects_circle_l820_82067

theorem line_intersects_circle (k : ℝ) (h1 : k = 2) (radius : ℝ) (center_distance : ℝ) (eq_roots : ∀ x, x^2 - k * x + 1 = 0) :
  radius = 5 → center_distance = k → k < radius :=
by
  intros hradius hdistance
  have h_root_eq : k = 2 := h1
  have h_rad : radius = 5 := hradius
  have h_dist : center_distance = k := hdistance
  have kval : k = 2 := h1
  simp [kval, hradius, hdistance, h_rad, h_dist]
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l820_82067


namespace NUMINAMATH_GPT_number_of_solutions_l820_82079

theorem number_of_solutions (x y: ℕ) (hx : 0 < x) (hy : 0 < y) :
    (1 / (x + 1) + 1 / y + 1 / ((x + 1) * y) = 1 / 1991) →
    ∃! (n : ℕ), n = 64 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l820_82079


namespace NUMINAMATH_GPT_combined_cost_is_correct_l820_82084

-- Definitions based on the conditions
def dryer_cost : ℕ := 150
def washer_cost : ℕ := 3 * dryer_cost
def combined_cost : ℕ := dryer_cost + washer_cost

-- Statement to be proved
theorem combined_cost_is_correct : combined_cost = 600 :=
by
  sorry

end NUMINAMATH_GPT_combined_cost_is_correct_l820_82084


namespace NUMINAMATH_GPT_equation_of_line_passing_through_center_and_perpendicular_to_l_l820_82054

theorem equation_of_line_passing_through_center_and_perpendicular_to_l (a : ℝ) : 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  ∃ (b : ℝ), ∀ x y : ℝ, (x + y + 1 = 0) := 
by 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  use 1
  sorry

end NUMINAMATH_GPT_equation_of_line_passing_through_center_and_perpendicular_to_l_l820_82054


namespace NUMINAMATH_GPT_total_students_in_class_l820_82077

/-- 
There are 208 boys in the class.
There are 69 more girls than boys.
The total number of students in the class is the sum of boys and girls.
Prove that the total number of students in the graduating class is 485.
-/
theorem total_students_in_class (boys girls : ℕ) (h1 : boys = 208) (h2 : girls = boys + 69) : 
  boys + girls = 485 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l820_82077


namespace NUMINAMATH_GPT_find_r_for_f_of_3_eq_0_l820_82087

noncomputable def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

theorem find_r_for_f_of_3_eq_0 : ∃ r : ℝ, f 3 r = 0 ∧ r = -186 := by
  sorry

end NUMINAMATH_GPT_find_r_for_f_of_3_eq_0_l820_82087


namespace NUMINAMATH_GPT_sum_of_ages_l820_82055

variable (a b c : ℕ)

theorem sum_of_ages (h1 : a = 20 + b + c) (h2 : a^2 = 2000 + (b + c)^2) : a + b + c = 80 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l820_82055


namespace NUMINAMATH_GPT_distance_to_lake_l820_82011

theorem distance_to_lake (d : ℝ) :
  ¬ (d ≥ 10) → ¬ (d ≤ 9) → d ≠ 7 → d ∈ Set.Ioo 9 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_distance_to_lake_l820_82011


namespace NUMINAMATH_GPT_member_sum_or_double_exists_l820_82039

theorem member_sum_or_double_exists (n : ℕ) (k : ℕ) (P: ℕ → ℕ) (m: ℕ) 
  (h_mem : n = 1978)
  (h_countries : m = 6) : 
  ∃ k, (∃ i j, P i + P j = k ∧ P i = P j)
    ∨ (∃ i, 2 * P i = k) :=
sorry

end NUMINAMATH_GPT_member_sum_or_double_exists_l820_82039


namespace NUMINAMATH_GPT_age_difference_l820_82006

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l820_82006


namespace NUMINAMATH_GPT_danny_bottle_caps_after_collection_l820_82061

-- Definitions for the conditions
def initial_bottle_caps : ℕ := 69
def bottle_caps_thrown : ℕ := 60
def bottle_caps_found : ℕ := 58

-- Theorem stating the proof problem
theorem danny_bottle_caps_after_collection : 
  initial_bottle_caps - bottle_caps_thrown + bottle_caps_found = 67 :=
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_danny_bottle_caps_after_collection_l820_82061


namespace NUMINAMATH_GPT_pipe_A_fill_time_l820_82074

theorem pipe_A_fill_time (x : ℝ) (h₁ : x > 0) (h₂ : 1 / x + 1 / 15 = 1 / 6) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_pipe_A_fill_time_l820_82074


namespace NUMINAMATH_GPT_set_A_membership_l820_82089

theorem set_A_membership (U : Finset ℕ) (A : Finset ℕ) (B : Finset ℕ)
  (hU : U.card = 193)
  (hB : B.card = 49)
  (hneither : (U \ (A ∪ B)).card = 59)
  (hAandB : (A ∩ B).card = 25) :
  A.card = 110 := sorry

end NUMINAMATH_GPT_set_A_membership_l820_82089


namespace NUMINAMATH_GPT_card_trick_l820_82064

/-- A magician is able to determine the fifth card from a 52-card deck using a prearranged 
    communication system between the magician and the assistant, thus no supernatural 
    abilities are required. -/
theorem card_trick (deck : Finset ℕ) (h_deck : deck.card = 52) (chosen_cards : Finset ℕ)
  (h_chosen : chosen_cards.card = 5) (shown_cards : Finset ℕ) (h_shown : shown_cards.card = 4)
  (fifth_card : ℕ) (h_fifth_card : fifth_card ∈ chosen_cards \ shown_cards) :
  ∃ (prearranged_system : (Finset ℕ) → (Finset ℕ) → ℕ),
    ∀ (remaining : Finset ℕ), remaining.card = 1 → 
    prearranged_system shown_cards remaining = fifth_card := 
sorry

end NUMINAMATH_GPT_card_trick_l820_82064


namespace NUMINAMATH_GPT_compute_product_fraction_l820_82076

theorem compute_product_fraction :
  ( ((3 : ℚ)^4 - 1) / ((3 : ℚ)^4 + 1) *
    ((4 : ℚ)^4 - 1) / ((4 : ℚ)^4 + 1) * 
    ((5 : ℚ)^4 - 1) / ((5 : ℚ)^4 + 1) *
    ((6 : ℚ)^4 - 1) / ((6 : ℚ)^4 + 1) *
    ((7 : ℚ)^4 - 1) / ((7 : ℚ)^4 + 1)
  ) = (25 / 210) := 
  sorry

end NUMINAMATH_GPT_compute_product_fraction_l820_82076


namespace NUMINAMATH_GPT_inequality_problem_l820_82060

open Real

theorem inequality_problem {a b c d : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_ac : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_problem_l820_82060


namespace NUMINAMATH_GPT_min_balls_draw_l820_82010

def box1_red := 40
def box1_green := 30
def box1_yellow := 25
def box1_blue := 15

def box2_red := 35
def box2_green := 25
def box2_yellow := 20

def min_balls_to_draw_to_get_20_balls_of_single_color (totalRed totalGreen totalYellow totalBlue : ℕ) : ℕ :=
  let maxNoColor :=
    (min totalRed 19) + (min totalGreen 19) + (min totalYellow 19) + (min totalBlue 15)
  maxNoColor + 1

theorem  min_balls_draw {r1 r2 g1 g2 y1 y2 b1 : ℕ} :
  r1 = box1_red -> g1 = box1_green -> y1 = box1_yellow -> b1 = box1_blue ->
  r2 = box2_red -> g2 = box2_green -> y2 = box2_yellow ->
  min_balls_to_draw_to_get_20_balls_of_single_color (r1 + r2) (g1 + g2) (y1 + y2) b1 = 73 :=
by
  intros
  unfold min_balls_to_draw_to_get_20_balls_of_single_color
  sorry

end NUMINAMATH_GPT_min_balls_draw_l820_82010


namespace NUMINAMATH_GPT_find_a8_l820_82096

/-!
Let {a_n} be an arithmetic sequence, with S_n representing the sum of the first n terms.
Given:
1. S_6 = 8 * S_3
2. a_3 - a_5 = 8
Prove: a_8 = -26
-/

noncomputable def arithmetic_seq (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_seq (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem find_a8 (a_1 d : ℤ)
  (h1 : sum_arithmetic_seq a_1 d 6 = 8 * sum_arithmetic_seq a_1 d 3)
  (h2 : arithmetic_seq a_1 d 3 - arithmetic_seq a_1 d 5 = 8) :
  arithmetic_seq a_1 d 8 = -26 :=
  sorry

end NUMINAMATH_GPT_find_a8_l820_82096


namespace NUMINAMATH_GPT_portion_to_joe_and_darcy_eq_half_l820_82094

open Int

noncomputable def portion_given_to_joe_and_darcy : ℚ := 
let total_slices := 8
let portion_to_carl := 1 / 4
let slices_to_carl := portion_to_carl * total_slices
let slices_left := 2
let slices_given_to_joe_and_darcy := total_slices - slices_to_carl - slices_left
let portion_to_joe_and_darcy := slices_given_to_joe_and_darcy / total_slices
portion_to_joe_and_darcy

theorem portion_to_joe_and_darcy_eq_half :
  portion_given_to_joe_and_darcy = 1 / 2 :=
sorry

end NUMINAMATH_GPT_portion_to_joe_and_darcy_eq_half_l820_82094


namespace NUMINAMATH_GPT_danivan_drugstore_end_of_week_inventory_l820_82038

-- Define the initial conditions in Lean
def initial_inventory := 4500
def sold_monday := 2445
def sold_tuesday := 900
def sold_wednesday_to_sunday := 50 * 5
def supplier_delivery := 650

-- Define the statement of the proof problem
theorem danivan_drugstore_end_of_week_inventory :
  initial_inventory - (sold_monday + sold_tuesday + sold_wednesday_to_sunday) + supplier_delivery = 1555 :=
by
  sorry

end NUMINAMATH_GPT_danivan_drugstore_end_of_week_inventory_l820_82038


namespace NUMINAMATH_GPT_infinite_impossible_values_of_d_l820_82070

theorem infinite_impossible_values_of_d 
  (pentagon_perimeter square_perimeter : ℕ) 
  (d : ℕ) 
  (h1 : pentagon_perimeter = 5 * (d + ((square_perimeter) / 4)) )
  (h2 : square_perimeter > 0)
  (h3 : pentagon_perimeter - square_perimeter = 2023) :
  ∀ n : ℕ, n > 404 → ¬∃ d : ℕ, d = n :=
by {
  sorry
}

end NUMINAMATH_GPT_infinite_impossible_values_of_d_l820_82070


namespace NUMINAMATH_GPT_remainder_7_pow_700_div_100_l820_82001

theorem remainder_7_pow_700_div_100 : (7 ^ 700) % 100 = 1 := 
  by sorry

end NUMINAMATH_GPT_remainder_7_pow_700_div_100_l820_82001


namespace NUMINAMATH_GPT_rational_sum_eq_one_l820_82032

theorem rational_sum_eq_one (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := 
by
  sorry

end NUMINAMATH_GPT_rational_sum_eq_one_l820_82032


namespace NUMINAMATH_GPT_total_profit_l820_82017

variable (InvestmentA InvestmentB InvestmentTimeA InvestmentTimeB ShareA : ℝ)
variable (hA : InvestmentA = 150)
variable (hB : InvestmentB = 200)
variable (hTimeA : InvestmentTimeA = 12)
variable (hTimeB : InvestmentTimeB = 6)
variable (hShareA : ShareA = 60)

theorem total_profit (TotalProfit : ℝ) :
  (ShareA / 3) * 5 = TotalProfit := 
by
  sorry

end NUMINAMATH_GPT_total_profit_l820_82017


namespace NUMINAMATH_GPT_student_chose_number_l820_82046

theorem student_chose_number (x : ℕ) (h : 2 * x - 138 = 112) : x = 125 :=
by
  sorry

end NUMINAMATH_GPT_student_chose_number_l820_82046


namespace NUMINAMATH_GPT_train_speed_in_km_hr_l820_82015

-- Definitions based on conditions
def train_length : ℝ := 150  -- meters
def crossing_time : ℝ := 6  -- seconds

-- Definition for conversion factor
def meters_per_second_to_km_per_hour (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Main theorem
theorem train_speed_in_km_hr : meters_per_second_to_km_per_hour (train_length / crossing_time) = 90 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_hr_l820_82015


namespace NUMINAMATH_GPT_cost_of_chlorine_l820_82037

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_chlorine_l820_82037


namespace NUMINAMATH_GPT_shopper_saved_percentage_l820_82057

-- Definition of the problem conditions
def amount_saved : ℝ := 4
def amount_spent : ℝ := 36

-- Lean 4 statement to prove the percentage saved
theorem shopper_saved_percentage : (amount_saved / (amount_spent + amount_saved)) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_shopper_saved_percentage_l820_82057


namespace NUMINAMATH_GPT_sum_single_digits_l820_82068

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end NUMINAMATH_GPT_sum_single_digits_l820_82068


namespace NUMINAMATH_GPT_apples_harvested_l820_82020

variable (A P : ℕ)
variable (h₁ : P = 3 * A) (h₂ : P - A = 120)

theorem apples_harvested : A = 60 := 
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_apples_harvested_l820_82020


namespace NUMINAMATH_GPT_roots_in_arithmetic_progression_l820_82030

theorem roots_in_arithmetic_progression (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x2 = (x1 + x3) / 2) ∧ (x1 + x2 + x3 = -a) ∧ (x1 * x3 + x2 * (x1 + x3) = b) ∧ (x1 * x2 * x3 = -c)) ↔ 
  (27 * c = 3 * a * b - 2 * a^3 ∧ 3 * b ≤ a^2) :=
sorry

end NUMINAMATH_GPT_roots_in_arithmetic_progression_l820_82030


namespace NUMINAMATH_GPT_log5_of_15625_l820_82023

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end NUMINAMATH_GPT_log5_of_15625_l820_82023


namespace NUMINAMATH_GPT_correct_option_B_l820_82012

theorem correct_option_B (x : ℝ) : (1 - x)^2 = 1 - 2 * x + x^2 :=
sorry

end NUMINAMATH_GPT_correct_option_B_l820_82012


namespace NUMINAMATH_GPT_customer_payment_l820_82058

noncomputable def cost_price : ℝ := 4090.9090909090905
noncomputable def markup : ℝ := 0.32
noncomputable def selling_price : ℝ := cost_price * (1 + markup)

theorem customer_payment :
  selling_price = 5400 :=
by
  unfold selling_price
  unfold cost_price
  unfold markup
  sorry

end NUMINAMATH_GPT_customer_payment_l820_82058


namespace NUMINAMATH_GPT_vector_b_norm_range_l820_82073

variable (a b : ℝ × ℝ)
variable (norm_a : ‖a‖ = 1)
variable (norm_sum : ‖a + b‖ = 2)

theorem vector_b_norm_range : 1 ≤ ‖b‖ ∧ ‖b‖ ≤ 3 :=
sorry

end NUMINAMATH_GPT_vector_b_norm_range_l820_82073


namespace NUMINAMATH_GPT_product_gcd_lcm_eq_1296_l820_82053

theorem product_gcd_lcm_eq_1296 : (Int.gcd 24 54) * (Int.lcm 24 54) = 1296 := by
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_eq_1296_l820_82053


namespace NUMINAMATH_GPT_regular_price_of_tire_l820_82093

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 3 = 240) : x = 79 :=
by
  sorry

end NUMINAMATH_GPT_regular_price_of_tire_l820_82093


namespace NUMINAMATH_GPT_fraction_of_top_10_lists_l820_82024

theorem fraction_of_top_10_lists (total_members : ℝ) (min_top_10_lists : ℝ) (fraction : ℝ) 
  (h1 : total_members = 765) (h2 : min_top_10_lists = 191.25) : 
    min_top_10_lists / total_members = fraction := by
  have h3 : fraction = 0.25 := by sorry
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_fraction_of_top_10_lists_l820_82024


namespace NUMINAMATH_GPT_inverse_proposition_l820_82059

-- Definition of the proposition
def complementary_angles_on_same_side (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- The original proposition
def original_proposition (l m : Line) : Prop := complementary_angles_on_same_side l m → parallel_lines l m

-- The statement of the proof problem
theorem inverse_proposition (l m : Line) :
  (complementary_angles_on_same_side l m → parallel_lines l m) →
  (parallel_lines l m → complementary_angles_on_same_side l m) := sorry

end NUMINAMATH_GPT_inverse_proposition_l820_82059


namespace NUMINAMATH_GPT_permutation_value_l820_82043

theorem permutation_value (n : ℕ) (h : n * (n - 1) = 12) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_permutation_value_l820_82043


namespace NUMINAMATH_GPT_derivative_of_function_y_l820_82097

noncomputable def function_y (x : ℝ) : ℝ := (x^2) / (x + 3)

theorem derivative_of_function_y (x : ℝ) :
  deriv function_y x = (x^2 + 6 * x) / ((x + 3)^2) :=
by 
  -- sorry since the proof is not required
  sorry

end NUMINAMATH_GPT_derivative_of_function_y_l820_82097


namespace NUMINAMATH_GPT_part1_part2_l820_82005

noncomputable def choose (n : ℕ) (k : ℕ) : ℕ :=
  n.choose k

theorem part1 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let doctors_left := total_doctors - 1 - 1 -- as one internal medicine must participate and one surgeon cannot
  choose doctors_left (team_size - 1) = 3060 := by
  sorry

theorem part2 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let only_internal_medicine := choose internal_medicine_doctors team_size
  let only_surgeons := choose surgeons team_size
  let total_ways := choose total_doctors team_size
  total_ways - only_internal_medicine - only_surgeons = 14656 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l820_82005


namespace NUMINAMATH_GPT_tony_pool_filling_time_l820_82052

theorem tony_pool_filling_time
  (J S T : ℝ)
  (hJ : J = 1 / 30)
  (hS : S = 1 / 45)
  (hCombined : J + S + T = 1 / 15) :
  T = 1 / 90 :=
by
  -- the setup for proof would be here
  sorry

end NUMINAMATH_GPT_tony_pool_filling_time_l820_82052


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l820_82044

theorem sum_and_product_of_roots (m p : ℝ) 
    (h₁ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α + β = 9)
    (h₂ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α * β = 14) :
    m + p = 69 := 
sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l820_82044


namespace NUMINAMATH_GPT_race_dead_heat_l820_82000

theorem race_dead_heat 
  (L Vb : ℝ) 
  (speed_a : ℝ := (16/15) * Vb)
  (speed_c : ℝ := (20/15) * Vb) 
  (time_a : ℝ := L / speed_a)
  (time_b : ℝ := L / Vb)
  (time_c : ℝ := L / speed_c) :
  (1 / (16 / 15) = 3 / 4) → 
  (1 - 3 / 4) = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_race_dead_heat_l820_82000


namespace NUMINAMATH_GPT_number_of_pens_sold_l820_82031

variables (C N : ℝ) (gain_percentage : ℝ) (gain : ℝ)

-- Defining conditions given in the problem
def trader_gain_cost_pens (C N : ℝ) : ℝ := 30 * C
def gain_percentage_condition (gain_percentage : ℝ) : Prop := gain_percentage = 0.30
def gain_condition (C N : ℝ) : Prop := (0.30 * N * C) = 30 * C

-- Defining the theorem to prove
theorem number_of_pens_sold
  (h_gain_percentage : gain_percentage_condition gain_percentage)
  (h_gain : gain_condition C N) :
  N = 100 :=
sorry

end NUMINAMATH_GPT_number_of_pens_sold_l820_82031


namespace NUMINAMATH_GPT_min_product_value_max_product_value_l820_82063

open Real

noncomputable def min_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

noncomputable def max_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

theorem min_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ min_cos_sin_product x y z = 1 / 8 :=
sorry

theorem max_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ max_cos_sin_product x y z = (2 + sqrt 3) / 8 :=
sorry

end NUMINAMATH_GPT_min_product_value_max_product_value_l820_82063


namespace NUMINAMATH_GPT_range_of_x_max_y_over_x_l820_82042

-- Define the circle and point P(x,y) on the circle
def CircleEquation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9

theorem range_of_x (x y : ℝ) (h : CircleEquation x y) : 1 ≤ x ∧ x ≤ 7 :=
sorry

theorem max_y_over_x (x y : ℝ) (h : CircleEquation x y) : ∀ k : ℝ, (k = y / x) → 0 ≤ k ∧ k ≤ (24 / 7) :=
sorry

end NUMINAMATH_GPT_range_of_x_max_y_over_x_l820_82042


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l820_82075

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 30)
  (h2 : ∀ n, S n = n * (a 1 + (n - 1) / 2 * d))
  (h3 : S 12 = S 19) :
  d = -2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l820_82075


namespace NUMINAMATH_GPT_combined_original_price_l820_82091

theorem combined_original_price (S P : ℝ) 
  (hS : 0.25 * S = 6) 
  (hP : 0.60 * P = 12) :
  S + P = 44 :=
by
  sorry

end NUMINAMATH_GPT_combined_original_price_l820_82091


namespace NUMINAMATH_GPT_solve_nat_pairs_l820_82028

theorem solve_nat_pairs (n m : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end NUMINAMATH_GPT_solve_nat_pairs_l820_82028


namespace NUMINAMATH_GPT_pentagon_edges_same_color_l820_82095

theorem pentagon_edges_same_color
  (A B : Fin 5 → Fin 5)
  (C : (Fin 5 → Fin 5) × (Fin 5 → Fin 5) → Bool)
  (condition : ∀ (i j : Fin 5), ∀ (k l m : Fin 5), (C (i, j) = C (k, l) → C (i, j) ≠ C (k, m))) :
  (∀ (x : Fin 5), C (A x, A ((x + 1) % 5)) = C (B x, B ((x + 1) % 5))) :=
by
sorry

end NUMINAMATH_GPT_pentagon_edges_same_color_l820_82095


namespace NUMINAMATH_GPT_greatest_possible_value_q_minus_r_l820_82098

theorem greatest_possible_value_q_minus_r : ∃ q r : ℕ, 1025 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_possible_value_q_minus_r_l820_82098


namespace NUMINAMATH_GPT_sum_of_roots_3x2_minus_12x_plus_12_eq_4_l820_82071

def sum_of_roots_quadratic (a b : ℚ) (h : a ≠ 0) : ℚ := -b / a

theorem sum_of_roots_3x2_minus_12x_plus_12_eq_4 :
  sum_of_roots_quadratic 3 (-12) (by norm_num) = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_3x2_minus_12x_plus_12_eq_4_l820_82071


namespace NUMINAMATH_GPT_sum_of_roots_of_qubic_polynomial_l820_82049

noncomputable def Q (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_qubic_polynomial (a b c d : ℝ) 
  (h₁ : ∀ x : ℝ, Q a b c d (x^4 + x) ≥ Q a b c d (x^3 + 1))
  (h₂ : Q a b c d 1 = 0) : 
  -b / a = 3 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_qubic_polynomial_l820_82049


namespace NUMINAMATH_GPT_fox_can_eat_80_fox_cannot_eat_65_l820_82019
-- import the required library

-- Define the conditions for the problem.
def total_candies := 100
def piles := 3
def fox_eat_equalize (fox: ℕ) (pile1: ℕ) (pile2: ℕ): ℕ :=
  if pile1 = pile2 then fox + pile1 else fox + pile2 - pile1

-- Statement for part (a)
theorem fox_can_eat_80: ∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 80) ∨ 
              (fox_eat_equalize x c₁ c₂  = 80)) :=
sorry

-- Statement for part (b)
theorem fox_cannot_eat_65: ¬ (∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 65) ∨ 
              (fox_eat_equalize x c₁ c₂  = 65))) :=
sorry

end NUMINAMATH_GPT_fox_can_eat_80_fox_cannot_eat_65_l820_82019


namespace NUMINAMATH_GPT_difference_in_total_cost_l820_82047

theorem difference_in_total_cost
  (item_price : ℝ := 15)
  (tax_rate1 : ℝ := 0.08)
  (tax_rate2 : ℝ := 0.072)
  (discount : ℝ := 0.005)
  (correct_difference : ℝ := 0.195) :
  let discounted_tax_rate := tax_rate2 - discount
  let total_price_with_tax_rate1 := item_price * (1 + tax_rate1)
  let total_price_with_discounted_tax_rate := item_price * (1 + discounted_tax_rate)
  total_price_with_tax_rate1 - total_price_with_discounted_tax_rate = correct_difference := by
  sorry

end NUMINAMATH_GPT_difference_in_total_cost_l820_82047


namespace NUMINAMATH_GPT_area_of_circle_l820_82090

theorem area_of_circle (C : ℝ) (hC : C = 36 * Real.pi) : 
  ∃ k : ℝ, (∃ r : ℝ, r = 18 ∧ k = r^2 ∧ (pi * r^2 = k * pi)) ∧ k = 324 :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_l820_82090


namespace NUMINAMATH_GPT_proof_problem_l820_82045

-- Define the conditions: n is a positive integer and (n(n + 1) / 3) is a square
def problem_condition (n : ℕ) : Prop :=
  ∃ m : ℕ, n > 0 ∧ (n * (n + 1)) = 3 * m^2

-- Define the proof problem: given the condition, n is a multiple of 3, n+1 and n/3 are squares
theorem proof_problem (n : ℕ) (h : problem_condition n) : 
  (∃ a : ℕ, n = 3 * a^2) ∧ 
  (∃ b : ℕ, n + 1 = b^2) ∧ 
  (∃ c : ℕ, n = 3 * c^2) :=
sorry

end NUMINAMATH_GPT_proof_problem_l820_82045


namespace NUMINAMATH_GPT_sine_wave_solution_l820_82056

theorem sine_wave_solution (a b c : ℝ) (h_pos_a : a > 0) 
  (h_amp : a = 3) 
  (h_period : (2 * Real.pi) / b = Real.pi) 
  (h_peak : (Real.pi / (2 * b)) - (c / b) = Real.pi / 6) : 
  a = 3 ∧ b = 2 ∧ c = Real.pi / 6 :=
by
  -- Lean code to construct the proof will appear here
  sorry

end NUMINAMATH_GPT_sine_wave_solution_l820_82056


namespace NUMINAMATH_GPT_simon_removes_exactly_180_silver_coins_l820_82048

theorem simon_removes_exactly_180_silver_coins :
  ∀ (initial_total_coins initial_gold_percentage final_gold_percentage : ℝ) 
  (initial_silver_coins final_total_coins final_silver_coins silver_coins_removed : ℕ),
  initial_total_coins = 200 → 
  initial_gold_percentage = 0.02 →
  final_gold_percentage = 0.2 →
  initial_silver_coins = (initial_total_coins * (1 - initial_gold_percentage)) → 
  final_total_coins = (4 / final_gold_percentage) →
  final_silver_coins = (final_total_coins - 4) →
  silver_coins_removed = (initial_silver_coins - final_silver_coins) →
  silver_coins_removed = 180 :=
by
  intros initial_total_coins initial_gold_percentage final_gold_percentage 
         initial_silver_coins final_total_coins final_silver_coins silver_coins_removed
  sorry

end NUMINAMATH_GPT_simon_removes_exactly_180_silver_coins_l820_82048


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l820_82082

theorem initial_volume_of_mixture 
  (V : ℝ)
  (h1 : 0 < V) 
  (h2 : 0.20 * V = 0.15 * (V + 5)) :
  V = 15 :=
by 
  -- proof steps 
  sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l820_82082


namespace NUMINAMATH_GPT_red_apples_ordered_l820_82081

variable (R : ℕ)

theorem red_apples_ordered (h : R + 32 = 2 + 73) : R = 43 := by
  sorry

end NUMINAMATH_GPT_red_apples_ordered_l820_82081


namespace NUMINAMATH_GPT_oreo_solution_l820_82029

noncomputable def oreo_problem : Prop :=
∃ (m : ℤ), (11 + m * 11 + 3 = 36) → m = 2

theorem oreo_solution : oreo_problem :=
sorry

end NUMINAMATH_GPT_oreo_solution_l820_82029


namespace NUMINAMATH_GPT_find_f2_l820_82069

variable (f g : ℝ → ℝ) (a : ℝ)

-- Definitions based on conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def equation (f g : ℝ → ℝ) (a : ℝ) := ∀ x, f x + g x = a^x - a^(-x) + 2

-- Lean statement for the proof problem
theorem find_f2
  (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : equation f g a)
  (h4 : g 2 = a) : f 2 = 15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l820_82069


namespace NUMINAMATH_GPT_min_b_for_quadratic_factorization_l820_82007

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end NUMINAMATH_GPT_min_b_for_quadratic_factorization_l820_82007
