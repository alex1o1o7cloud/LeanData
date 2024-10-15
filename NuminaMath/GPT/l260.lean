import Mathlib

namespace NUMINAMATH_GPT_martha_cakes_l260_26071

theorem martha_cakes :
  ∀ (n : ℕ), (∀ (c : ℕ), c = 3 → (∀ (k : ℕ), k = 6 → n = c * k)) → n = 18 :=
by
  intros n h
  specialize h 3 rfl 6 rfl
  exact h

end NUMINAMATH_GPT_martha_cakes_l260_26071


namespace NUMINAMATH_GPT_expected_value_die_l260_26031

noncomputable def expected_value (P_Star P_Moon : ℚ) (win_Star lose_Moon : ℚ) : ℚ :=
  P_Star * win_Star + P_Moon * lose_Moon

theorem expected_value_die :
  expected_value (2/5) (3/5) 4 (-3) = -1/5 := by
  sorry

end NUMINAMATH_GPT_expected_value_die_l260_26031


namespace NUMINAMATH_GPT_Megan_popsicles_l260_26040

def minutes_in_hour : ℕ := 60

def total_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * minutes_in_hour + minutes

def popsicle_time : ℕ := 18

def popsicles_consumed (total_minutes : ℕ) (popsicle_time : ℕ) : ℕ :=
  total_minutes / popsicle_time

theorem Megan_popsicles (hours : ℕ) (minutes : ℕ) (popsicle_time : ℕ)
  (total_minutes : ℕ) (h_hours : hours = 5) (h_minutes : minutes = 36) (h_popsicle_time : popsicle_time = 18)
  (h_total_minutes : total_minutes = (5 * 60 + 36)) :
  popsicles_consumed 336 popsicle_time = 18 :=
by 
  sorry

end NUMINAMATH_GPT_Megan_popsicles_l260_26040


namespace NUMINAMATH_GPT_smallest_N_exists_l260_26005

def find_smallest_N (N : ℕ) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : ℕ),
  (N ≠ 0) ∧ 
  (c1 = 6 * c2 - 1) ∧ 
  (N + c2 = 6 * c3 - 2) ∧ 
  (2 * N + c3 = 6 * c4 - 3) ∧ 
  (3 * N + c4 = 6 * c5 - 4) ∧ 
  (4 * N + c5 = 6 * c6 - 5) ∧ 
  (5 * N + c6 = 6 * c1)

theorem smallest_N_exists : ∃ (N : ℕ), find_smallest_N N :=
sorry

end NUMINAMATH_GPT_smallest_N_exists_l260_26005


namespace NUMINAMATH_GPT_youngest_child_age_is_3_l260_26019

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end NUMINAMATH_GPT_youngest_child_age_is_3_l260_26019


namespace NUMINAMATH_GPT_passed_candidates_l260_26082

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end NUMINAMATH_GPT_passed_candidates_l260_26082


namespace NUMINAMATH_GPT_smallest_integer_x_l260_26032

-- Conditions
def condition1 (x : ℤ) : Prop := 7 - 5 * x < 25
def condition2 (x : ℤ) : Prop := ∃ y : ℤ, y = 10 ∧ y - 3 * x > 6

-- Statement
theorem smallest_integer_x : ∃ x : ℤ, condition1 x ∧ condition2 x ∧ ∀ z : ℤ, condition1 z ∧ condition2 z → x ≤ z :=
  sorry

end NUMINAMATH_GPT_smallest_integer_x_l260_26032


namespace NUMINAMATH_GPT_Mark_speeding_ticket_owed_amount_l260_26070

theorem Mark_speeding_ticket_owed_amount :
  let base_fine := 50
  let additional_penalty_per_mph := 2
  let mph_over_limit := 45
  let school_zone_multiplier := 2
  let court_costs := 300
  let lawyer_fee_per_hour := 80
  let lawyer_hours := 3
  let additional_penalty := additional_penalty_per_mph * mph_over_limit
  let pre_school_zone_fine := base_fine + additional_penalty
  let doubled_fine := pre_school_zone_fine * school_zone_multiplier
  let total_fine_with_court_costs := doubled_fine + court_costs
  let lawyer_total_fee := lawyer_fee_per_hour * lawyer_hours
  let total_owed := total_fine_with_court_costs + lawyer_total_fee
  total_owed = 820 :=
by
  sorry

end NUMINAMATH_GPT_Mark_speeding_ticket_owed_amount_l260_26070


namespace NUMINAMATH_GPT_truncated_cone_resistance_l260_26052

theorem truncated_cone_resistance (a b h : ℝ) (ρ : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_pos : 0 < h) :
  (∫ x in (0:ℝ)..h, ρ / (π * ((a + x * (b - a) / h) / 2) ^ 2)) = 4 * ρ * h / (π * a * b) := 
sorry

end NUMINAMATH_GPT_truncated_cone_resistance_l260_26052


namespace NUMINAMATH_GPT_find_k_l260_26079

noncomputable def expr_to_complete_square (x : ℝ) : ℝ :=
  x^2 - 6 * x

theorem find_k (x : ℝ) : ∃ a h k, expr_to_complete_square x = a * (x - h)^2 + k ∧ k = -9 :=
by
  use 1, 3, -9
  -- detailed steps of the proof would go here
  sorry

end NUMINAMATH_GPT_find_k_l260_26079


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l260_26001

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l260_26001


namespace NUMINAMATH_GPT_find_second_number_l260_26097

variable (A B : ℕ)

def is_LCM (a b lcm : ℕ) := Nat.lcm a b = lcm
def is_HCF (a b hcf : ℕ) := Nat.gcd a b = hcf

theorem find_second_number (h_lcm : is_LCM 330 B 2310) (h_hcf : is_HCF 330 B 30) : B = 210 := by
  sorry

end NUMINAMATH_GPT_find_second_number_l260_26097


namespace NUMINAMATH_GPT_total_weight_is_correct_l260_26078

noncomputable def A (B : ℝ) : ℝ := 12 + (1/2) * B
noncomputable def B (C : ℝ) : ℝ := 8 + (1/3) * C
noncomputable def C (A : ℝ) : ℝ := 20 + 2 * A
noncomputable def NewWeightB (A B : ℝ) : ℝ := B + 0.15 * A
noncomputable def NewWeightA (A C : ℝ) : ℝ := A - 0.10 * C

theorem total_weight_is_correct (B C : ℝ) (h1 : A B = (C - 20) / 2)
  (h2 : B = 8 + (1/3) * C) 
  (h3 : C = 20 + 2 * A B) 
  (h4 : NewWeightB (A B) B = 38.35) 
  (h5 : NewWeightA (A B) C = 21.2) :
  NewWeightA (A B) C + NewWeightB (A B) B + C = 139.55 :=
sorry

end NUMINAMATH_GPT_total_weight_is_correct_l260_26078


namespace NUMINAMATH_GPT_balloon_count_l260_26061

theorem balloon_count (total_balloons red_balloons blue_balloons black_balloons : ℕ) 
  (h_total : total_balloons = 180)
  (h_red : red_balloons = 3 * blue_balloons)
  (h_black : black_balloons = 2 * blue_balloons) :
  red_balloons = 90 ∧ blue_balloons = 30 ∧ black_balloons = 60 :=
by
  sorry

end NUMINAMATH_GPT_balloon_count_l260_26061


namespace NUMINAMATH_GPT_trader_gain_percentage_l260_26016

theorem trader_gain_percentage 
  (C : ℝ) -- cost of each pen
  (h1 : 250 * C ≠ 0) -- ensure the cost of 250 pens is non-zero
  (h2 : 65 * C > 0) -- ensure the gain is positive
  (h3 : 250 * C + 65 * C > 0) -- ensure the selling price is positive
  : (65 / 250) * 100 = 26 := 
sorry

end NUMINAMATH_GPT_trader_gain_percentage_l260_26016


namespace NUMINAMATH_GPT_oranges_per_tree_correct_l260_26075

-- Definitions for the conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def total_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * total_oranges
def seeds_planted := 2 * frank_oranges
def total_trees := seeds_planted
def total_oranges_picked := 810
def oranges_per_tree := total_oranges_picked / total_trees

-- Theorem statement
theorem oranges_per_tree_correct : oranges_per_tree = 5 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_oranges_per_tree_correct_l260_26075


namespace NUMINAMATH_GPT_mean_value_of_interior_angles_of_quadrilateral_l260_26012

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end NUMINAMATH_GPT_mean_value_of_interior_angles_of_quadrilateral_l260_26012


namespace NUMINAMATH_GPT_solution_set_f_x_gt_0_l260_26092

theorem solution_set_f_x_gt_0 (b : ℝ)
  (h_eq : ∀ x : ℝ, (x + 1) * (x - 3) = 0 → b = -2) :
  {x : ℝ | (x - 1)^2 > 0} = {x : ℝ | x ≠ 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_x_gt_0_l260_26092


namespace NUMINAMATH_GPT_total_teachers_correct_l260_26056

noncomputable def total_teachers (x : ℕ) : ℕ := 26 + 104 + x

theorem total_teachers_correct
    (x : ℕ)
    (h : (x : ℝ) / (26 + 104 + x) = 16 / 56) :
  total_teachers x = 182 :=
sorry

end NUMINAMATH_GPT_total_teachers_correct_l260_26056


namespace NUMINAMATH_GPT_liam_annual_income_l260_26042

theorem liam_annual_income (q : ℝ) (I : ℝ) (T : ℝ) 
  (h1 : T = (q + 0.5) * 0.01 * I) 
  (h2 : I > 50000) 
  (h3 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000)) : 
  I = 56000 :=
by
  sorry

end NUMINAMATH_GPT_liam_annual_income_l260_26042


namespace NUMINAMATH_GPT_Nero_speed_is_8_l260_26096

-- Defining the conditions
def Jerome_time := 6 -- in hours
def Nero_time := 3 -- in hours
def Jerome_speed := 4 -- in miles per hour

-- Calculation step
def Distance := Jerome_speed * Jerome_time

-- The theorem we need to prove (Nero's speed)
theorem Nero_speed_is_8 :
  (Distance / Nero_time) = 8 := by
  sorry

end NUMINAMATH_GPT_Nero_speed_is_8_l260_26096


namespace NUMINAMATH_GPT_scientific_notation_of_2200_l260_26013

-- Define scientific notation criteria
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ a ∧ a < 10

-- Problem statement
theorem scientific_notation_of_2200 : ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 2200 ∧ a = 2.2 ∧ n = 3 :=
by {
  -- Proof can be added here.
  sorry
}

end NUMINAMATH_GPT_scientific_notation_of_2200_l260_26013


namespace NUMINAMATH_GPT_Reeta_pencils_l260_26043

-- Let R be the number of pencils Reeta has
variable (R : ℕ)

-- Condition 1: Anika has 4 more than twice the number of pencils as Reeta
def Anika_pencils := 2 * R + 4

-- Condition 2: Together, Anika and Reeta have 64 pencils
def combined_pencils := R + Anika_pencils R

theorem Reeta_pencils (h : combined_pencils R = 64) : R = 20 :=
by
  sorry

end NUMINAMATH_GPT_Reeta_pencils_l260_26043


namespace NUMINAMATH_GPT_sum_of_coordinates_l260_26054

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 4) :
  let x := 4
  let y := (f⁻¹ x) / 4
  x + y = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l260_26054


namespace NUMINAMATH_GPT_repaved_today_l260_26025

theorem repaved_today (total before : ℕ) (h_total : total = 4938) (h_before : before = 4133) : total - before = 805 := by
  sorry

end NUMINAMATH_GPT_repaved_today_l260_26025


namespace NUMINAMATH_GPT_max_product_of_roots_of_quadratic_l260_26062

theorem max_product_of_roots_of_quadratic :
  ∃ k : ℚ, 6 * k^2 - 8 * k + (4 / 3) = 0 ∧ (64 - 48 * k) ≥ 0 ∧ (∀ k' : ℚ, (64 - 48 * k') ≥ 0 → (k'/3) ≤ (4/9)) :=
by
  sorry

end NUMINAMATH_GPT_max_product_of_roots_of_quadratic_l260_26062


namespace NUMINAMATH_GPT_larry_gave_52_apples_l260_26048

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end NUMINAMATH_GPT_larry_gave_52_apples_l260_26048


namespace NUMINAMATH_GPT_parabola_intercepts_l260_26007

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end NUMINAMATH_GPT_parabola_intercepts_l260_26007


namespace NUMINAMATH_GPT_customer_total_payment_l260_26033

def Riqing_Beef_Noodles_quantity : ℕ := 24
def Riqing_Beef_Noodles_price_per_bag : ℝ := 1.80
def Riqing_Beef_Noodles_discount : ℝ := 0.8

def Kang_Shifu_Ice_Red_Tea_quantity : ℕ := 6
def Kang_Shifu_Ice_Red_Tea_price_per_box : ℝ := 1.70
def Kang_Shifu_Ice_Red_Tea_discount : ℝ := 0.8

def Shanlin_Purple_Cabbage_Soup_quantity : ℕ := 5
def Shanlin_Purple_Cabbage_Soup_price_per_bag : ℝ := 3.40

def Shuanghui_Ham_Sausage_quantity : ℕ := 3
def Shuanghui_Ham_Sausage_price_per_bag : ℝ := 11.20
def Shuanghui_Ham_Sausage_discount : ℝ := 0.9

def total_price : ℝ :=
  (Riqing_Beef_Noodles_quantity * Riqing_Beef_Noodles_price_per_bag * Riqing_Beef_Noodles_discount) +
  (Kang_Shifu_Ice_Red_Tea_quantity * Kang_Shifu_Ice_Red_Tea_price_per_box * Kang_Shifu_Ice_Red_Tea_discount) +
  (Shanlin_Purple_Cabbage_Soup_quantity * Shanlin_Purple_Cabbage_Soup_price_per_bag) +
  (Shuanghui_Ham_Sausage_quantity * Shuanghui_Ham_Sausage_price_per_bag * Shuanghui_Ham_Sausage_discount)

theorem customer_total_payment :
  total_price = 89.96 :=
by
  unfold total_price
  sorry

end NUMINAMATH_GPT_customer_total_payment_l260_26033


namespace NUMINAMATH_GPT_quadratic_two_distinct_roots_l260_26036

theorem quadratic_two_distinct_roots :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1^2 - 3 = 0 ∧ 2 * x2^2 - 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_roots_l260_26036


namespace NUMINAMATH_GPT_count_positive_solutions_of_eq_l260_26099

theorem count_positive_solutions_of_eq : 
  (∃ x : ℝ, x^2 = -6 * x + 9 ∧ x > 0) ∧ (¬ ∃ y : ℝ, y^2 = -6 * y + 9 ∧ y > 0 ∧ y ≠ -3 + 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_count_positive_solutions_of_eq_l260_26099


namespace NUMINAMATH_GPT_johns_weekly_allowance_l260_26094

variable (A : ℝ)

theorem johns_weekly_allowance 
  (h1 : ∃ A : ℝ, A > 0) 
  (h2 : (4/15) * A = 0.75) : 
  A = 2.8125 := 
by 
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_johns_weekly_allowance_l260_26094


namespace NUMINAMATH_GPT_maximum_items_6_yuan_l260_26004

theorem maximum_items_6_yuan :
  ∃ (x : ℕ), (∀ (x' : ℕ), (∃ (y z : ℕ), 6 * x' + 4 * y + 2 * z = 60 ∧ x' + y + z = 16) →
    x' ≤ 7) → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_maximum_items_6_yuan_l260_26004


namespace NUMINAMATH_GPT_alcohol_added_amount_l260_26047

theorem alcohol_added_amount :
  ∀ (x : ℝ), (40 * 0.05 + x) = 0.15 * (40 + x + 4.5) -> x = 5.5 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_alcohol_added_amount_l260_26047


namespace NUMINAMATH_GPT_overlap_length_in_mm_l260_26088

theorem overlap_length_in_mm {sheets : ℕ} {length_per_sheet : ℝ} {perimeter : ℝ} 
  (h_sheets : sheets = 12)
  (h_length_per_sheet : length_per_sheet = 18)
  (h_perimeter : perimeter = 210) : 
  (length_per_sheet * sheets - perimeter) / sheets * 10 = 5 := by
  sorry

end NUMINAMATH_GPT_overlap_length_in_mm_l260_26088


namespace NUMINAMATH_GPT_cube_volume_l260_26034

theorem cube_volume (A : ℝ) (V : ℝ) (h : A = 64) : V = 512 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_l260_26034


namespace NUMINAMATH_GPT_function_property_l260_26093

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem function_property
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_property : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0)
  : f (-4) > f (-6) :=
sorry

end NUMINAMATH_GPT_function_property_l260_26093


namespace NUMINAMATH_GPT_minimum_votes_for_tall_to_win_l260_26027

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_votes_for_tall_to_win_l260_26027


namespace NUMINAMATH_GPT_calc_expression_l260_26051

theorem calc_expression : 
  abs (Real.sqrt 3 - 2) + (8:ℝ)^(1/3) - Real.sqrt 16 + (-1)^(2023:ℝ) = -(Real.sqrt 3) - 1 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l260_26051


namespace NUMINAMATH_GPT_solve_for_a_l260_26068

def op (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem solve_for_a (a : ℝ) : op a 3 = 15 → a = 11 :=
by
  intro h
  rw [op] at h
  sorry

end NUMINAMATH_GPT_solve_for_a_l260_26068


namespace NUMINAMATH_GPT_smallest_number_of_students_l260_26074

theorem smallest_number_of_students 
  (ninth_to_seventh : ℕ → ℕ → Prop)
  (ninth_to_sixth : ℕ → ℕ → Prop) 
  (r1 : ninth_to_seventh 3 2) 
  (r2 : ninth_to_sixth 7 4) : 
  ∃ n7 n6 n9, 
    ninth_to_seventh n9 n7 ∧ 
    ninth_to_sixth n9 n6 ∧ 
    n9 + n7 + n6 = 47 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_students_l260_26074


namespace NUMINAMATH_GPT_solve_equation_l260_26058

theorem solve_equation (x : ℝ) : 
  x^2 + 2 * x + 4 * Real.sqrt (x^2 + 2 * x) - 5 = 0 →
  (x = Real.sqrt 2 - 1) ∨ (x = - (Real.sqrt 2 + 1)) :=
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l260_26058


namespace NUMINAMATH_GPT_remainder_2n_div_14_l260_26029

theorem remainder_2n_div_14 (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_2n_div_14_l260_26029


namespace NUMINAMATH_GPT_cost_of_fencing_l260_26009

-- Definitions of ratio and area conditions
def sides_ratio (length width : ℕ) : Prop := length / width = 3 / 2
def area (length width : ℕ) : Prop := length * width = 3750

-- Define the cost per meter in paise
def cost_per_meter : ℕ := 70

-- Convert paise to rupees
def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

-- The main statement we want to prove
theorem cost_of_fencing (length width perimeter : ℕ)
  (H1 : sides_ratio length width)
  (H2 : area length width)
  (H3 : perimeter = 2 * length + 2 * width) :
  paise_to_rupees (perimeter * cost_per_meter) = 175 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l260_26009


namespace NUMINAMATH_GPT_min_dot_product_l260_26039

theorem min_dot_product (m n : ℝ) (x1 x2 : ℝ)
    (h1 : m ≠ 0) 
    (h2 : n ≠ 0)
    (h3 : (x1 + 2) * (x2 - x1) + m * x1 * (n - m * x1) = 0) :
    ∃ (x1 : ℝ), (x1 = -2 / (m^2 + 1)) → 
    (x1 + 2) * (x2 + 2) + m * n * x1 = 4 * m^2 / (m^2 + 1) := 
sorry

end NUMINAMATH_GPT_min_dot_product_l260_26039


namespace NUMINAMATH_GPT_length_segment_AB_l260_26053

theorem length_segment_AB (A B : ℝ) (hA : A = -5) (hB : B = 2) : |A - B| = 7 :=
by
  sorry

end NUMINAMATH_GPT_length_segment_AB_l260_26053


namespace NUMINAMATH_GPT_john_spent_fraction_l260_26064

theorem john_spent_fraction (initial_money snacks_left necessities_left snacks_fraction : ℝ)
  (h1 : initial_money = 20)
  (h2 : snacks_fraction = 1/5)
  (h3 : snacks_left = initial_money * snacks_fraction)
  (h4 : necessities_left = 4)
  (remaining_money : ℝ) (h5 : remaining_money = initial_money - snacks_left)
  (spent_on_necessities : ℝ) (h6 : spent_on_necessities = remaining_money - necessities_left) 
  (fraction_spent : ℝ) (h7 : fraction_spent = spent_on_necessities / remaining_money) : 
  fraction_spent = 3/4 := 
sorry

end NUMINAMATH_GPT_john_spent_fraction_l260_26064


namespace NUMINAMATH_GPT_greatest_integer_value_l260_26080

theorem greatest_integer_value (x : ℤ) : 3 * |x - 2| + 9 ≤ 24 → x ≤ 7 :=
by sorry

end NUMINAMATH_GPT_greatest_integer_value_l260_26080


namespace NUMINAMATH_GPT_point_in_second_quadrant_l260_26020

def in_second_quadrant (z : Complex) : Prop := 
  z.re < 0 ∧ z.im > 0

theorem point_in_second_quadrant : in_second_quadrant (Complex.ofReal (1) + 2 * Complex.I / (Complex.ofReal (1) - Complex.I)) :=
by sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l260_26020


namespace NUMINAMATH_GPT_problem_statement_l260_26014

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

theorem problem_statement : 3 * g 3 - g 9 = -48 - 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l260_26014


namespace NUMINAMATH_GPT_cylinder_twice_volume_l260_26069

theorem cylinder_twice_volume :
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  V_desired = pi * r^2 * h2 :=
by
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  show V_desired = pi * r^2 * h2
  sorry

end NUMINAMATH_GPT_cylinder_twice_volume_l260_26069


namespace NUMINAMATH_GPT_prod_of_real_roots_equation_l260_26083

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end NUMINAMATH_GPT_prod_of_real_roots_equation_l260_26083


namespace NUMINAMATH_GPT_binomial_7_4_eq_35_l260_26045

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end NUMINAMATH_GPT_binomial_7_4_eq_35_l260_26045


namespace NUMINAMATH_GPT_circumcircle_eq_of_triangle_ABC_l260_26022

noncomputable def circumcircle_equation (A B C : ℝ × ℝ) : String := sorry

theorem circumcircle_eq_of_triangle_ABC :
  circumcircle_equation (4, 1) (-6, 3) (3, 0) = "x^2 + y^2 + x - 9y - 12 = 0" :=
sorry

end NUMINAMATH_GPT_circumcircle_eq_of_triangle_ABC_l260_26022


namespace NUMINAMATH_GPT_juan_speed_l260_26057

-- Statement of given distances and time
def distance : ℕ := 80
def time : ℕ := 8

-- Desired speed in miles per hour
def expected_speed : ℕ := 10

-- Theorem statement: Speed is distance divided by time and should equal 10 miles per hour
theorem juan_speed : distance / time = expected_speed :=
  by
  sorry

end NUMINAMATH_GPT_juan_speed_l260_26057


namespace NUMINAMATH_GPT_max_value_of_m_l260_26038

noncomputable def f (x m n : ℝ) : ℝ := x^2 + m*x + n^2
noncomputable def g (x m n : ℝ) : ℝ := x^2 + (m+2)*x + n^2 + m + 1

theorem max_value_of_m (m n t : ℝ) :
  (∀(t : ℝ), f t m n ≥ 0 ∨ g t m n ≥ 0) → m ≤ 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_max_value_of_m_l260_26038


namespace NUMINAMATH_GPT_value_of_a_l260_26003

-- Definition of the function and the point
def graph_function (x : ℝ) : ℝ := -x^2
def point_lies_on_graph (a : ℝ) : Prop := (a, -9) ∈ {p : ℝ × ℝ | p.2 = graph_function p.1}

-- The theorem stating that if the point (a, -9) lies on the graph of y = -x^2, then a = ±3
theorem value_of_a (a : ℝ) (h : point_lies_on_graph a) : a = 3 ∨ a = -3 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l260_26003


namespace NUMINAMATH_GPT_arc_length_of_sector_l260_26026

theorem arc_length_of_sector (theta : ℝ) (r : ℝ) (h_theta : theta = 90) (h_r : r = 6) : 
  (theta / 360) * 2 * Real.pi * r = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l260_26026


namespace NUMINAMATH_GPT_rhind_papyrus_max_bread_l260_26059

theorem rhind_papyrus_max_bread
  (a1 a2 a3 a4 a5 : ℕ) (d : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 100)
  (h2 : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a4 = a1 + 3 * d)
  (h6 : a5 = a1 + 4 * d)
  (h7 : a3 + a4 + a5 = 3 * (a1 + a2)) :
  a5 = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_rhind_papyrus_max_bread_l260_26059


namespace NUMINAMATH_GPT_average_tickets_per_day_l260_26049

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end NUMINAMATH_GPT_average_tickets_per_day_l260_26049


namespace NUMINAMATH_GPT_num_balls_in_box_l260_26035

theorem num_balls_in_box (n : ℕ) (h1: 9 <= n) (h2: (9 : ℝ) / n = 0.30) : n = 30 :=
sorry

end NUMINAMATH_GPT_num_balls_in_box_l260_26035


namespace NUMINAMATH_GPT_algebraic_expression_value_l260_26017

variables (a b c d m : ℤ)

def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℤ) : Prop := c * d = 1
def abs_eq_2 (m : ℤ) : Prop := |m| = 2

theorem algebraic_expression_value {a b c d m : ℤ} 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : abs_eq_2 m) :
  (2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l260_26017


namespace NUMINAMATH_GPT_setB_can_form_triangle_l260_26086

theorem setB_can_form_triangle : 
  let a := 8
  let b := 6
  let c := 4
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  let a := 8
  let b := 6
  let c := 4
  have h1 : a + b > c := by sorry
  have h2 : a + c > b := by sorry
  have h3 : b + c > a := by sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_setB_can_form_triangle_l260_26086


namespace NUMINAMATH_GPT_pure_imaginary_complex_number_l260_26089

variable (a : ℝ)

theorem pure_imaginary_complex_number:
  (a^2 + 2*a - 3 = 0) ∧ (a^2 + a - 6 ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_GPT_pure_imaginary_complex_number_l260_26089


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l260_26072

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + a < 0) → (a < 11) ∧ ¬((a < 11) → (∃ x : ℝ, x^2 - 2*x + a < 0)) :=
by
  -- Sorry to bypass proof below, which is correct as per the problem statement requirements.
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l260_26072


namespace NUMINAMATH_GPT_range_of_a_l260_26060

variable {α : Type*} [LinearOrderedField α]

def setA (a : α) : Set α := {x | abs (x - a) < 1}
def setB : Set α := {x | 1 < x ∧ x < 5}

theorem range_of_a (a : α) (h : setA a ∩ setB = ∅) : a ≤ 0 ∨ a ≥ 6 :=
sorry

end NUMINAMATH_GPT_range_of_a_l260_26060


namespace NUMINAMATH_GPT_correct_transformation_l260_26095

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end NUMINAMATH_GPT_correct_transformation_l260_26095


namespace NUMINAMATH_GPT_probability_A_and_B_same_county_l260_26081

/-
We have four experts and three counties. We need to assign the experts to the counties such 
that each county has at least one expert. We need to prove that the probability of experts 
A and B being dispatched to the same county is 1/6.
-/

def num_experts : Nat := 4
def num_counties : Nat := 3

def total_possible_events : Nat := 36
def favorable_events : Nat := 6

theorem probability_A_and_B_same_county :
  (favorable_events : ℚ) / total_possible_events = 1 / 6 := by sorry

end NUMINAMATH_GPT_probability_A_and_B_same_county_l260_26081


namespace NUMINAMATH_GPT_front_view_l260_26077

def first_column_heights := [3, 2]
def middle_column_heights := [1, 4, 2]
def third_column_heights := [5]

theorem front_view (h1 : first_column_heights = [3, 2])
                   (h2 : middle_column_heights = [1, 4, 2])
                   (h3 : third_column_heights = [5]) :
    [3, 4, 5] = [
        first_column_heights.foldr max 0,
        middle_column_heights.foldr max 0,
        third_column_heights.foldr max 0
    ] :=
    sorry

end NUMINAMATH_GPT_front_view_l260_26077


namespace NUMINAMATH_GPT_heptagon_triangulation_count_l260_26085

/-- The number of ways to divide a regular heptagon (7-sided polygon) 
    into 5 triangles using non-intersecting diagonals is 4. -/
theorem heptagon_triangulation_count : ∃ (n : ℕ), n = 4 ∧ ∀ (p : ℕ), (p = 7 ∧ (∀ (k : ℕ), k = 5 → (n = 4))) :=
by {
  -- The proof is non-trivial and omitted here
  sorry
}

end NUMINAMATH_GPT_heptagon_triangulation_count_l260_26085


namespace NUMINAMATH_GPT_cylinder_volume_multiplication_factor_l260_26002

theorem cylinder_volume_multiplication_factor (r h : ℝ) (h_r_positive : r > 0) (h_h_positive : h > 0) :
  let V := π * r^2 * h
  let V' := π * (2.5 * r)^2 * (3 * h)
  let X := V' / V
  X = 18.75 :=
by
  -- Proceed with the proof here
  sorry

end NUMINAMATH_GPT_cylinder_volume_multiplication_factor_l260_26002


namespace NUMINAMATH_GPT_a7_of_expansion_x10_l260_26018

theorem a7_of_expansion_x10 : 
  (∃ (a : ℕ) (a1 : ℕ) (a2 : ℕ) (a3 : ℕ) 
     (a4 : ℕ) (a5 : ℕ) (a6 : ℕ) 
     (a8 : ℕ) (a9 : ℕ) (a10 : ℕ),
     ((x : ℕ) → x^10 = a + a1*(x-1) + a2*(x-1)^2 + a3*(x-1)^3 + 
                      a4*(x-1)^4 + a5*(x-1)^5 + a6*(x-1)^6 + 
                      120*(x-1)^7 + a8*(x-1)^8 + a9*(x-1)^9 + a10*(x-1)^10)) :=
  sorry

end NUMINAMATH_GPT_a7_of_expansion_x10_l260_26018


namespace NUMINAMATH_GPT_mike_total_time_spent_l260_26091

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_time_spent_l260_26091


namespace NUMINAMATH_GPT_ellipse_focus_m_eq_3_l260_26008

theorem ellipse_focus_m_eq_3 (m : ℝ) (h : m > 0) : 
  (∃ a c : ℝ, a = 5 ∧ c = 4 ∧ c^2 = a^2 - m^2)
  → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focus_m_eq_3_l260_26008


namespace NUMINAMATH_GPT_problem_statement_l260_26067

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : 0 < c) (h6 : c < 1) :
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) :=
sorry

end NUMINAMATH_GPT_problem_statement_l260_26067


namespace NUMINAMATH_GPT_find_b_l260_26037

theorem find_b (b : ℚ) : (-4 : ℚ) * (45 / 4) = -45 → (-4 + 45 / 4) = -b → b = -29 / 4 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_b_l260_26037


namespace NUMINAMATH_GPT_find_angle_C_l260_26076

variable {A B C a b c : ℝ}
variable (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variable (hTriangle : A + B + C = π)
variable (hSides : a > 0 ∧ b > 0 ∧ c > 0)
variable (hCondition : Real.sqrt 3 * a = 2 * c * Real.sin A)

theorem find_angle_C (hA_pos : A ≠ 0) : C = π / 3 :=
  sorry

end NUMINAMATH_GPT_find_angle_C_l260_26076


namespace NUMINAMATH_GPT_pollywogs_disappear_in_44_days_l260_26055

theorem pollywogs_disappear_in_44_days :
  ∀ (initial_count rate_mature rate_caught first_period_days : ℕ),
  initial_count = 2400 →
  rate_mature = 50 →
  rate_caught = 10 →
  first_period_days = 20 →
  (initial_count - first_period_days * (rate_mature + rate_caught)) / rate_mature + first_period_days = 44 := 
by
  intros initial_count rate_mature rate_caught first_period_days h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_pollywogs_disappear_in_44_days_l260_26055


namespace NUMINAMATH_GPT_value_of_ab_l260_26065

theorem value_of_ab (a b c : ℝ) (C : ℝ) (h1 : (a + b) ^ 2 - c ^ 2 = 4) (h2 : C = Real.pi / 3) : 
  a * b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_ab_l260_26065


namespace NUMINAMATH_GPT_Lizzy_money_after_loan_l260_26024

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end NUMINAMATH_GPT_Lizzy_money_after_loan_l260_26024


namespace NUMINAMATH_GPT_min_trips_correct_l260_26006

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end NUMINAMATH_GPT_min_trips_correct_l260_26006


namespace NUMINAMATH_GPT_find_a99_l260_26046

def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n ≥ 2, a n - a (n-1) = n + 1

theorem find_a99 (a : ℕ → ℕ) (h : seq a) : a 99 = 5049 :=
by
  have : seq a := h
  sorry

end NUMINAMATH_GPT_find_a99_l260_26046


namespace NUMINAMATH_GPT_sum_digits_base8_to_base4_l260_26000

theorem sum_digits_base8_to_base4 :
  ∀ n : ℕ, (n ≥ 512 ∧ n ≤ 4095) →
  (∃ d : ℕ, (4^d > n ∧ n ≥ 4^(d-1))) →
  (d = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_digits_base8_to_base4_l260_26000


namespace NUMINAMATH_GPT_tan_alpha_add_pi_over_4_l260_26063

open Real

theorem tan_alpha_add_pi_over_4 
  (α : ℝ)
  (h1 : tan α = sqrt 3) : 
  tan (α + π / 4) = -2 - sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_add_pi_over_4_l260_26063


namespace NUMINAMATH_GPT_smallest_sum_divisible_by_3_l260_26023

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def is_consecutive_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4 ∨ p2 = p1 + 6 ∨ p2 = p1 + 2) ∧
  (p3 = p2 + 2 ∨ p3 = p2 + 4) ∧
  (p4 = p3 + 2 ∨ p4 = p3 + 4)

def greater_than_5 (p : ℕ) : Prop := p > 5

theorem smallest_sum_divisible_by_3 :
  ∃ (p1 p2 p3 p4 : ℕ), is_consecutive_prime p1 p2 p3 p4 ∧
                      greater_than_5 p1 ∧
                      (p1 + p2 + p3 + p4) % 3 = 0 ∧
                      (p1 + p2 + p3 + p4) = 48 :=
by sorry

end NUMINAMATH_GPT_smallest_sum_divisible_by_3_l260_26023


namespace NUMINAMATH_GPT_solve_asterisk_l260_26011

theorem solve_asterisk (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end NUMINAMATH_GPT_solve_asterisk_l260_26011


namespace NUMINAMATH_GPT_solution_set_l260_26015

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

theorem solution_set :
  ∀ x y z : ℝ, system_of_equations x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = -1 ∧ z = -3) ∨ 
  (x = -2 ∧ y = 1 ∧ z = -3) ∨ (x = -2 ∧ y = -1 ∧ z = 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l260_26015


namespace NUMINAMATH_GPT_find_m_l260_26087

theorem find_m 
(x0 m : ℝ)
(h1 : m ≠ 0)
(h2 : x0^2 - x0 + m = 0)
(h3 : (2 * x0)^2 - 2 * x0 + 3 * m = 0)
: m = -2 :=
sorry

end NUMINAMATH_GPT_find_m_l260_26087


namespace NUMINAMATH_GPT_correct_proposition_is_B_l260_26030

variables {m n : Type} {α β : Type}

-- Define parallel and perpendicular relationships
def parallel (l₁ l₂ : Type) : Prop := sorry
def perpendicular (l₁ l₂ : Type) : Prop := sorry

def lies_in (l : Type) (p : Type) : Prop := sorry

-- The problem statement
theorem correct_proposition_is_B
  (H1 : perpendicular m α)
  (H2 : perpendicular n β)
  (H3 : perpendicular α β) :
  perpendicular m n :=
sorry

end NUMINAMATH_GPT_correct_proposition_is_B_l260_26030


namespace NUMINAMATH_GPT_yoojung_namjoon_total_flowers_l260_26050

theorem yoojung_namjoon_total_flowers
  (yoojung_flowers : ℕ)
  (namjoon_flowers : ℕ)
  (yoojung_condition : yoojung_flowers = 4 * namjoon_flowers)
  (yoojung_count : yoojung_flowers = 32) :
  yoojung_flowers + namjoon_flowers = 40 :=
by
  sorry

end NUMINAMATH_GPT_yoojung_namjoon_total_flowers_l260_26050


namespace NUMINAMATH_GPT_length_of_each_stone_l260_26073

-- Define the dimensions of the hall in decimeters
def hall_length_dm : ℕ := 36 * 10
def hall_breadth_dm : ℕ := 15 * 10

-- Define the width of each stone in decimeters
def stone_width_dm : ℕ := 5

-- Define the number of stones
def number_of_stones : ℕ := 1350

-- Define the total area of the hall
def hall_area : ℕ := hall_length_dm * hall_breadth_dm

-- Define the area of one stone
def stone_area : ℕ := hall_area / number_of_stones

-- Define the length of each stone and state the theorem
theorem length_of_each_stone : (stone_area / stone_width_dm) = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_each_stone_l260_26073


namespace NUMINAMATH_GPT_consumption_increase_l260_26098

variable (T C C' : ℝ)
variable (h1 : 0.8 * T * C' = 0.92 * T * C)

theorem consumption_increase (T C C' : ℝ) (h1 : 0.8 * T * C' = 0.92 * T * C) : C' = 1.15 * C :=
by
  sorry

end NUMINAMATH_GPT_consumption_increase_l260_26098


namespace NUMINAMATH_GPT_happy_children_count_l260_26041

-- Definitions of the conditions
def total_children : ℕ := 60
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4
def boys_neither_happy_nor_sad : ℕ := 10

-- The theorem we wish to prove
theorem happy_children_count :
  total_children - sad_children - neither_happy_nor_sad_children = 30 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_happy_children_count_l260_26041


namespace NUMINAMATH_GPT_combined_weight_l260_26084

theorem combined_weight (S R : ℝ) (h1 : S - 5 = 2 * R) (h2 : S = 75) : S + R = 110 :=
sorry

end NUMINAMATH_GPT_combined_weight_l260_26084


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l260_26090

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

-- Define the condition that vectors are parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- State the problem: if a and b are parallel, then x = 1/2
theorem parallel_vectors_x_value (x : ℝ) (h : is_parallel a (b x)) : x = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l260_26090


namespace NUMINAMATH_GPT_incorrect_statements_l260_26010

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def monotonically_decreasing_in_pos (f : ℝ → ℝ) : Prop :=
∀ x y, 0 < x ∧ x < y → f y ≤ f x

theorem incorrect_statements
  (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_decreasing : monotonically_decreasing_in_pos f) :
  ¬ (∀ a, f (2 * a) < f (-a)) ∧ ¬ (f π > f (-3)) ∧ ¬ (∀ a, f (a^2 + 1) < f 1) :=
by sorry

end NUMINAMATH_GPT_incorrect_statements_l260_26010


namespace NUMINAMATH_GPT_expression_evaluates_to_3_l260_26021

theorem expression_evaluates_to_3 :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 :=
sorry

end NUMINAMATH_GPT_expression_evaluates_to_3_l260_26021


namespace NUMINAMATH_GPT_cos_negative_570_equals_negative_sqrt3_div_2_l260_26066

theorem cos_negative_570_equals_negative_sqrt3_div_2 : Real.cos (-570 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_cos_negative_570_equals_negative_sqrt3_div_2_l260_26066


namespace NUMINAMATH_GPT_simplify_expression_l260_26044

variable (x y : ℕ)
variable (h_x : x = 5)
variable (h_y : y = 2)

theorem simplify_expression : (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l260_26044


namespace NUMINAMATH_GPT_pyramid_dihedral_angle_l260_26028

theorem pyramid_dihedral_angle 
  (k : ℝ) 
  (h_k_pos : 0 < k) :
  ∃ α : ℝ, α = 2 * Real.arccos (1 / Real.sqrt (Real.sqrt (4 * k))) :=
sorry

end NUMINAMATH_GPT_pyramid_dihedral_angle_l260_26028
