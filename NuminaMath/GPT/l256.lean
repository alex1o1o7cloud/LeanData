import Mathlib

namespace NUMINAMATH_GPT_log_order_l256_25614

theorem log_order (a b c : ℝ) (h_a : a = Real.log 6 / Real.log 2) 
  (h_b : b = Real.log 15 / Real.log 5) (h_c : c = Real.log 21 / Real.log 7) : 
  a > b ∧ b > c := by sorry

end NUMINAMATH_GPT_log_order_l256_25614


namespace NUMINAMATH_GPT_hemisphere_surface_area_l256_25666

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h : π * r^2 = 225 * π) : 
  2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l256_25666


namespace NUMINAMATH_GPT_total_students_in_classrooms_l256_25637

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end NUMINAMATH_GPT_total_students_in_classrooms_l256_25637


namespace NUMINAMATH_GPT_isabella_hair_length_after_haircut_cm_l256_25683

theorem isabella_hair_length_after_haircut_cm :
  let initial_length_in : ℝ := 18  -- initial length in inches
  let growth_rate_in_per_week : ℝ := 0.5  -- growth rate in inches per week
  let weeks : ℝ := 4  -- time in weeks
  let hair_trimmed_in : ℝ := 2.25  -- length of hair trimmed in inches
  let cm_per_inch : ℝ := 2.54  -- conversion factor from inches to centimeters
  let final_length_in := initial_length_in + growth_rate_in_per_week * weeks - hair_trimmed_in  -- final length in inches
  let final_length_cm := final_length_in * cm_per_inch  -- final length in centimeters
  final_length_cm = 45.085 := by
  sorry

end NUMINAMATH_GPT_isabella_hair_length_after_haircut_cm_l256_25683


namespace NUMINAMATH_GPT_girls_boys_ratio_l256_25635

theorem girls_boys_ratio (G B : ℕ) (h1 : G + B = 100) (h2 : 0.20 * (G : ℝ) + 0.10 * (B : ℝ) = 15) : G / B = 1 :=
by
  -- Proof steps are omitted
  sorry

end NUMINAMATH_GPT_girls_boys_ratio_l256_25635


namespace NUMINAMATH_GPT_sum_of_coordinates_l256_25684

-- Define the given conditions as hypotheses
def isThreeUnitsFromLine (x y : ℝ) : Prop := y = 18 ∨ y = 12
def isTenUnitsFromPoint (x y : ℝ) : Prop := (x - 5)^2 + (y - 15)^2 = 100

-- We aim to prove the sum of the coordinates of the points satisfying these conditions
theorem sum_of_coordinates (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : isThreeUnitsFromLine x1 y1 ∧ isTenUnitsFromPoint x1 y1)
  (h2 : isThreeUnitsFromLine x2 y2 ∧ isTenUnitsFromPoint x2 y2)
  (h3 : isThreeUnitsFromLine x3 y3 ∧ isTenUnitsFromPoint x3 y3)
  (h4 : isThreeUnitsFromLine x4 y4 ∧ isTenUnitsFromPoint x4 y4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 50 :=
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l256_25684


namespace NUMINAMATH_GPT_anna_money_ratio_l256_25602

theorem anna_money_ratio (total_money spent_furniture left_money given_to_Anna : ℕ)
  (h_total : total_money = 2000)
  (h_spent : spent_furniture = 400)
  (h_left : left_money = 400)
  (h_after_furniture : total_money - spent_furniture = given_to_Anna + left_money) :
  (given_to_Anna / left_money) = 3 :=
by
  have h1 : total_money - spent_furniture = 1600 := by sorry
  have h2 : given_to_Anna = 1200 := by sorry
  have h3 : given_to_Anna / left_money = 3 := by sorry
  exact h3

end NUMINAMATH_GPT_anna_money_ratio_l256_25602


namespace NUMINAMATH_GPT_algebra_problem_l256_25616

theorem algebra_problem 
  (a : ℝ) 
  (h : a^3 + 3 * a^2 + 3 * a + 2 = 0) :
  (a + 1) ^ 2008 + (a + 1) ^ 2009 + (a + 1) ^ 2010 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_algebra_problem_l256_25616


namespace NUMINAMATH_GPT_prime_number_condition_l256_25668

theorem prime_number_condition (n : ℕ) (h1 : n ≥ 2) :
  (∀ d : ℕ, d ∣ n → d > 1 → d^2 + n ∣ n^2 + d) → Prime n :=
sorry

end NUMINAMATH_GPT_prime_number_condition_l256_25668


namespace NUMINAMATH_GPT_apples_to_mangos_equivalent_l256_25604

-- Definitions and conditions
def apples_worth_mangos (a b : ℝ) : Prop := (5 / 4) * 16 * a = 10 * b

-- Theorem statement
theorem apples_to_mangos_equivalent : 
  ∀ (a b : ℝ), apples_worth_mangos a b → (3 / 4) * 12 * a = 4.5 * b :=
by
  intro a b
  intro h
  sorry

end NUMINAMATH_GPT_apples_to_mangos_equivalent_l256_25604


namespace NUMINAMATH_GPT_combined_age_l256_25678

-- Define the conditions given in the problem
def Hezekiah_age : Nat := 4
def Ryanne_age := Hezekiah_age + 7

-- The statement to prove
theorem combined_age : Ryanne_age + Hezekiah_age = 15 :=
by
  -- we would provide the proof here, but for now we'll skip it with 'sorry'
  sorry

end NUMINAMATH_GPT_combined_age_l256_25678


namespace NUMINAMATH_GPT_find_number_of_students_l256_25656

theorem find_number_of_students (N : ℕ) (T : ℕ) (hN : N ≠ 0) (hT : T = 80 * N) 
  (h_avg_excluded : (T - 200) / (N - 5) = 90) : N = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_students_l256_25656


namespace NUMINAMATH_GPT_parabola_intersection_points_l256_25653

theorem parabola_intersection_points :
  (∃ (x y : ℝ), y = 4 * x ^ 2 + 3 * x - 7 ∧ y = 2 * x ^ 2 - 5)
  ↔ ((-2, 3) = (x, y) ∨ (1/2, -4.5) = (x, y)) :=
by
   -- To be proved (proof omitted)
   sorry

end NUMINAMATH_GPT_parabola_intersection_points_l256_25653


namespace NUMINAMATH_GPT_total_value_of_coins_is_correct_l256_25634

-- Definitions for the problem conditions
def number_of_dimes : ℕ := 22
def number_of_quarters : ℕ := 10
def value_of_dime : ℝ := 0.10
def value_of_quarter : ℝ := 0.25
def total_value_of_dimes : ℝ := number_of_dimes * value_of_dime
def total_value_of_quarters : ℝ := number_of_quarters * value_of_quarter
def total_value : ℝ := total_value_of_dimes + total_value_of_quarters

-- Theorem statement
theorem total_value_of_coins_is_correct : total_value = 4.70 := sorry

end NUMINAMATH_GPT_total_value_of_coins_is_correct_l256_25634


namespace NUMINAMATH_GPT_mean_home_runs_l256_25682

theorem mean_home_runs :
  let players6 := 5
  let players8 := 6
  let players10 := 4
  let home_runs6 := players6 * 6
  let home_runs8 := players8 * 8
  let home_runs10 := players10 * 10
  let total_home_runs := home_runs6 + home_runs8 + home_runs10
  let total_players := players6 + players8 + players10
  total_home_runs / total_players = 118 / 15 :=
by
  sorry

end NUMINAMATH_GPT_mean_home_runs_l256_25682


namespace NUMINAMATH_GPT_area_of_diamond_l256_25676

theorem area_of_diamond (x y : ℝ) : (|x / 2| + |y / 2| = 1) → 
∃ (area : ℝ), area = 8 :=
by sorry

end NUMINAMATH_GPT_area_of_diamond_l256_25676


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l256_25644

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l256_25644


namespace NUMINAMATH_GPT_probability_at_least_60_cents_l256_25693

theorem probability_at_least_60_cents :
  let num_total_outcomes := Nat.choose 16 8
  let num_successful_outcomes := 
    (Nat.choose 4 2) * (Nat.choose 5 1) * (Nat.choose 7 5) +
    1 -- only one way to choose all 8 dimes
  num_successful_outcomes / num_total_outcomes = 631 / 12870 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_60_cents_l256_25693


namespace NUMINAMATH_GPT_interval_between_segments_l256_25629

def population_size : ℕ := 800
def sample_size : ℕ := 40

theorem interval_between_segments : population_size / sample_size = 20 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_interval_between_segments_l256_25629


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l256_25696

theorem arithmetic_expression_evaluation :
  12 / 4 - 3 - 6 + 3 * 5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l256_25696


namespace NUMINAMATH_GPT_ratio_current_to_past_l256_25645

-- Conditions
def current_posters : ℕ := 22
def posters_after_summer (p : ℕ) : ℕ := p + 6
def posters_two_years_ago : ℕ := 14

-- Proof problem statement
theorem ratio_current_to_past (h₁ : current_posters = 22) (h₂ : posters_two_years_ago = 14) : 
  (current_posters / Nat.gcd current_posters posters_two_years_ago) = 11 ∧ 
  (posters_two_years_ago / Nat.gcd current_posters posters_two_years_ago) = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_current_to_past_l256_25645


namespace NUMINAMATH_GPT_find_constant_c_and_t_l256_25610

noncomputable def exists_constant_c_and_t (c : ℝ) (t : ℝ) : Prop :=
∀ (x1 x2 m : ℝ), (x1^2 - m * x1 - c = 0) ∧ (x2^2 - m * x2 - c = 0) →
  (t = 1 / ((1 + m^2) * x1^2) + 1 / ((1 + m^2) * x2^2))

theorem find_constant_c_and_t : ∃ (c t : ℝ), exists_constant_c_and_t c t ∧ c = 2 ∧ t = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_constant_c_and_t_l256_25610


namespace NUMINAMATH_GPT_perpendicular_vectors_l256_25630

/-- If vectors a = (1, 2) and b = (x, 4) are perpendicular, then x = -8. -/
theorem perpendicular_vectors (x : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (x, 4)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : x = -8 :=
by {
  sorry
}

end NUMINAMATH_GPT_perpendicular_vectors_l256_25630


namespace NUMINAMATH_GPT_triangle_side_a_l256_25607

theorem triangle_side_a (a : ℝ) (h1 : 4 < a) (h2 : a < 10) : a = 8 :=
  by
  sorry

end NUMINAMATH_GPT_triangle_side_a_l256_25607


namespace NUMINAMATH_GPT_warehouse_capacity_l256_25655

theorem warehouse_capacity (total_bins : ℕ) (bins_20_tons : ℕ) (bins_15_tons : ℕ)
    (total_capacity : ℕ) (h1 : total_bins = 30) (h2 : bins_20_tons = 12) 
    (h3 : bins_15_tons = total_bins - bins_20_tons) 
    (h4 : total_capacity = (bins_20_tons * 20) + (bins_15_tons * 15)) : 
    total_capacity = 510 :=
by {
  sorry
}

end NUMINAMATH_GPT_warehouse_capacity_l256_25655


namespace NUMINAMATH_GPT_prob_first_3_heads_last_5_tails_eq_l256_25694

-- Define the conditions
def prob_heads : ℚ := 3/5
def prob_tails : ℚ := 1 - prob_heads
def heads_flips (n : ℕ) : ℚ := prob_heads ^ n
def tails_flips (n : ℕ) : ℚ := prob_tails ^ n
def first_3_heads_last_5_tails (first_n : ℕ) (last_m : ℕ) : ℚ := (heads_flips first_n) * (tails_flips last_m)

-- Specify the problem
theorem prob_first_3_heads_last_5_tails_eq :
  first_3_heads_last_5_tails 3 5 = 864/390625 := 
by
  -- conditions and calculation here
  sorry

end NUMINAMATH_GPT_prob_first_3_heads_last_5_tails_eq_l256_25694


namespace NUMINAMATH_GPT_find_ABC_plus_DE_l256_25619

theorem find_ABC_plus_DE (ABCDE : Nat) (h1 : ABCDE = 13579 * 6) : (ABCDE / 1000 + ABCDE % 1000 % 100) = 888 :=
by
  sorry

end NUMINAMATH_GPT_find_ABC_plus_DE_l256_25619


namespace NUMINAMATH_GPT_inequality_solution_l256_25636

theorem inequality_solution (x : ℝ) : |x - 3| + |x - 5| ≥ 4 → x ≥ 6 ∨ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l256_25636


namespace NUMINAMATH_GPT_uncovered_area_is_8_l256_25613

-- Conditions
def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side_length : ℕ := 4

-- Theorem to prove
theorem uncovered_area_is_8
  (sh_height : ℕ := shoebox_height)
  (sh_width : ℕ := shoebox_width)
  (bl_length : ℕ := block_side_length)
  : sh_height * sh_width - bl_length * bl_length = 8 :=
by {
  -- Placeholder for proof; we are not proving it as per instructions.
  sorry
}

end NUMINAMATH_GPT_uncovered_area_is_8_l256_25613


namespace NUMINAMATH_GPT_candidate_X_votes_l256_25649

theorem candidate_X_votes (Z : ℕ) (Y : ℕ) (X : ℕ) (hZ : Z = 25000) 
                          (hY : Y = Z - (2 / 5) * Z) 
                          (hX : X = Y + (1 / 2) * Y) : 
                          X = 22500 :=
by
  sorry

end NUMINAMATH_GPT_candidate_X_votes_l256_25649


namespace NUMINAMATH_GPT_length_of_la_l256_25697

variables {A b c l_a: ℝ}
variables (S_ABC S_ACA' S_ABA': ℝ)

axiom area_of_ABC: S_ABC = (1 / 2) * b * c * Real.sin A
axiom area_of_ACA: S_ACA' = (1 / 2) * b * l_a * Real.sin (A / 2)
axiom area_of_ABA: S_ABA' = (1 / 2) * c * l_a * Real.sin (A / 2)
axiom sin_double_angle: Real.sin A = 2 * Real.sin (A / 2) * Real.cos (A / 2)

theorem length_of_la :
  l_a = (2 * b * c * Real.cos (A / 2)) / (b + c) :=
sorry

end NUMINAMATH_GPT_length_of_la_l256_25697


namespace NUMINAMATH_GPT_largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l256_25662

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end NUMINAMATH_GPT_largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l256_25662


namespace NUMINAMATH_GPT_fraction_of_single_men_l256_25600

theorem fraction_of_single_men :
  ∀ (total_faculty : ℕ) (women_percentage : ℝ) (married_percentage : ℝ) (married_men_ratio : ℝ),
    women_percentage = 0.7 → married_percentage = 0.4 → married_men_ratio = 2/3 →
    (total_faculty * (1 - women_percentage)) * (1 - married_men_ratio) / 
    (total_faculty * (1 - women_percentage)) = 1/3 :=
by 
  intros total_faculty women_percentage married_percentage married_men_ratio h_women h_married h_men_marry
  sorry

end NUMINAMATH_GPT_fraction_of_single_men_l256_25600


namespace NUMINAMATH_GPT_find_larger_number_l256_25608

theorem find_larger_number (x y : ℕ) (h1 : x + y = 55) (h2 : x - y = 15) : x = 35 := by 
  -- proof will go here
  sorry

end NUMINAMATH_GPT_find_larger_number_l256_25608


namespace NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l256_25621

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l256_25621


namespace NUMINAMATH_GPT_correct_remove_parentheses_l256_25603

theorem correct_remove_parentheses (a b c d : ℝ) :
  (a - (5 * b - (2 * c - 1)) = a - 5 * b + 2 * c - 1) :=
by sorry

end NUMINAMATH_GPT_correct_remove_parentheses_l256_25603


namespace NUMINAMATH_GPT_fourth_guard_run_distance_l256_25674

-- Define the rectangle's dimensions
def length : ℝ := 300
def width : ℝ := 200

-- Define the perimeter of the rectangle
def perimeter : ℝ := 2 * (length + width)

-- Given the sum of the distances run by three guards
def sum_of_three_guards : ℝ := 850

-- The fourth guard's distance is what we need to prove
def fourth_guard_distance := perimeter - sum_of_three_guards

-- The proof goal: we need to show that the fourth guard's distance is 150 meters
theorem fourth_guard_run_distance : fourth_guard_distance = 150 := by
  sorry  -- This placeholder means that the proof is omitted

end NUMINAMATH_GPT_fourth_guard_run_distance_l256_25674


namespace NUMINAMATH_GPT_dividend_is_10_l256_25647

theorem dividend_is_10
  (q d r : ℕ)
  (hq : q = 3)
  (hd : d = 3)
  (hr : d = 3 * r) :
  (q * d + r = 10) :=
by
  sorry

end NUMINAMATH_GPT_dividend_is_10_l256_25647


namespace NUMINAMATH_GPT_flowers_given_l256_25642

theorem flowers_given (initial_flowers total_flowers flowers_given : ℕ) 
  (h1 : initial_flowers = 67) 
  (h2 : total_flowers = 90) 
  (h3 : total_flowers = initial_flowers + flowers_given) : 
  flowers_given = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_flowers_given_l256_25642


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_iff_a_in_range_l256_25661

theorem inequality_holds_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x > 2 * a * x + a) ↔ (-4 < a ∧ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_iff_a_in_range_l256_25661


namespace NUMINAMATH_GPT_volume_of_ABDH_is_4_3_l256_25631

-- Define the vertices of the cube
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def D : (ℝ × ℝ × ℝ) := (0, 2, 0)
def H : (ℝ × ℝ × ℝ) := (0, 0, 2)

-- Function to calculate the volume of the pyramid
noncomputable def volume_of_pyramid (A B D H : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * 2 * 2 * 2

-- Theorem stating the volume of the pyramid ABDH is 4/3 cubic units
theorem volume_of_ABDH_is_4_3 : volume_of_pyramid A B D H = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_volume_of_ABDH_is_4_3_l256_25631


namespace NUMINAMATH_GPT_final_price_of_book_l256_25633

theorem final_price_of_book (original_price : ℝ) (d1_percentage : ℝ) (d2_percentage : ℝ) 
  (first_discount : ℝ) (second_discount : ℝ) (new_price1 : ℝ) (final_price : ℝ) :
  original_price = 15 ∧ d1_percentage = 0.20 ∧ d2_percentage = 0.25 ∧
  first_discount = d1_percentage * original_price ∧ new_price1 = original_price - first_discount ∧
  second_discount = d2_percentage * new_price1 ∧ 
  final_price = new_price1 - second_discount → final_price = 9 := 
by 
  sorry

end NUMINAMATH_GPT_final_price_of_book_l256_25633


namespace NUMINAMATH_GPT_count_expressible_integers_l256_25673

theorem count_expressible_integers :
  ∃ (count : ℕ), count = 1138 ∧ (∀ n, (n ≤ 2000) → (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n)) :=
sorry

end NUMINAMATH_GPT_count_expressible_integers_l256_25673


namespace NUMINAMATH_GPT_no_right_triangle_l256_25620

theorem no_right_triangle (a b c : ℝ) (h₁ : a = Real.sqrt 3) (h₂ : b = 2) (h₃ : c = Real.sqrt 5) : 
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end NUMINAMATH_GPT_no_right_triangle_l256_25620


namespace NUMINAMATH_GPT_derivative_of_log_base2_inv_x_l256_25623

noncomputable def my_function (x : ℝ) : ℝ := (Real.log x⁻¹) / (Real.log 2)

theorem derivative_of_log_base2_inv_x : 
  ∀ x : ℝ, x > 0 → deriv my_function x = -1 / (x * Real.log 2) :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_derivative_of_log_base2_inv_x_l256_25623


namespace NUMINAMATH_GPT_probability_shadedRegion_l256_25667

noncomputable def triangleVertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (0, 5), (5, 0))

noncomputable def totalArea : ℝ :=
  12.5

noncomputable def shadedArea : ℝ :=
  4.5

theorem probability_shadedRegion (x y : ℝ) :
  let p := (x, y)
  let condition := x + y <= 3
  let totalArea := 12.5
  let shadedArea := 4.5
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 5 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 5}) →
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 + p.2 ≤ 3}) →
  (shadedArea / totalArea) = 9/25 :=
by
  sorry

end NUMINAMATH_GPT_probability_shadedRegion_l256_25667


namespace NUMINAMATH_GPT_initial_pencils_l256_25624

theorem initial_pencils (pencils_added initial_pencils total_pencils : ℕ) 
  (h1 : pencils_added = 3) 
  (h2 : total_pencils = 5) :
  initial_pencils = total_pencils - pencils_added := 
by 
  sorry

end NUMINAMATH_GPT_initial_pencils_l256_25624


namespace NUMINAMATH_GPT_red_balls_count_l256_25606

theorem red_balls_count (R W N_1 N_2 : ℕ) 
  (h1 : R - 2 * N_1 = 18) 
  (h2 : W = 3 * N_1) 
  (h3 : R - 5 * N_2 = 0) 
  (h4 : W - 3 * N_2 = 18)
  : R = 50 :=
sorry

end NUMINAMATH_GPT_red_balls_count_l256_25606


namespace NUMINAMATH_GPT_calculate_percentage_l256_25677

/-- A candidate got a certain percentage of the votes polled and he lost to his rival by 2000 votes.
There were 10,000.000000000002 votes cast. What percentage of the votes did the candidate get? --/

def candidate_vote_percentage (P : ℝ) (total_votes : ℝ) (rival_margin : ℝ) : Prop :=
  (P / 100 * total_votes = total_votes - rival_margin) → P = 80

theorem calculate_percentage:
  candidate_vote_percentage P 10000.000000000002 2000 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_percentage_l256_25677


namespace NUMINAMATH_GPT_bottle_cap_count_l256_25664

theorem bottle_cap_count (price_per_cap total_cost : ℕ) (h_price : price_per_cap = 2) (h_total : total_cost = 12) : total_cost / price_per_cap = 6 :=
by
  sorry

end NUMINAMATH_GPT_bottle_cap_count_l256_25664


namespace NUMINAMATH_GPT_sum_of_non_solutions_l256_25654

theorem sum_of_non_solutions (A B C x : ℝ) 
  (h : ∀ x, ((x + B) * (A * x + 32)) = 4 * ((x + C) * (x + 8))) :
  (x = -B ∨ x = -8) → x ≠ -B → -B ≠ -8 → x ≠ -8 → x + 8 + B = 0 := 
sorry

end NUMINAMATH_GPT_sum_of_non_solutions_l256_25654


namespace NUMINAMATH_GPT_shaded_area_l256_25652

-- Define the problem in Lean
theorem shaded_area (area_large_square area_small_square : ℝ) (H_large_square : area_large_square = 10) (H_small_square : area_small_square = 4) (diagonals_contain : True) : 
  (area_large_square - area_small_square) / 4 = 1.5 :=
by
  sorry -- proof not required

end NUMINAMATH_GPT_shaded_area_l256_25652


namespace NUMINAMATH_GPT_value_of_a_l256_25646

theorem value_of_a (x a : ℤ) (h : x = 3 ∧ x^2 = a) : a = 9 :=
sorry

end NUMINAMATH_GPT_value_of_a_l256_25646


namespace NUMINAMATH_GPT_factorize_expression_l256_25615

theorem factorize_expression (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l256_25615


namespace NUMINAMATH_GPT_patrick_savings_ratio_l256_25640

theorem patrick_savings_ratio (S : ℕ) (bike_cost : ℕ) (lent_amt : ℕ) (remaining_amt : ℕ)
  (h1 : bike_cost = 150)
  (h2 : lent_amt = 50)
  (h3 : remaining_amt = 25)
  (h4 : S = remaining_amt + lent_amt) :
  (S / bike_cost : ℚ) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_patrick_savings_ratio_l256_25640


namespace NUMINAMATH_GPT_total_shoes_l256_25695

variables (people : ℕ) (shoes_per_person : ℕ)

-- There are 10 people
axiom h1 : people = 10
-- Each person has 2 shoes
axiom h2 : shoes_per_person = 2

-- The total number of shoes kept outside the library is 10 * 2 = 20
theorem total_shoes (people shoes_per_person : ℕ) (h1 : people = 10) (h2 : shoes_per_person = 2) : people * shoes_per_person = 20 :=
by sorry

end NUMINAMATH_GPT_total_shoes_l256_25695


namespace NUMINAMATH_GPT_radius_is_independent_variable_l256_25627

theorem radius_is_independent_variable 
  (r C : ℝ)
  (h : C = 2 * Real.pi * r) : 
  ∃ r_independent, r_independent = r := 
by
  sorry

end NUMINAMATH_GPT_radius_is_independent_variable_l256_25627


namespace NUMINAMATH_GPT_reduced_price_l256_25687

theorem reduced_price (P Q : ℝ) (h : P ≠ 0) (h₁ : 900 = Q * P) (h₂ : 900 = (Q + 6) * (0.90 * P)) : 0.90 * P = 15 :=
by 
  sorry

end NUMINAMATH_GPT_reduced_price_l256_25687


namespace NUMINAMATH_GPT_avg_last_four_is_63_75_l256_25611

noncomputable def average_of_list (l : List ℝ) : ℝ :=
  l.sum / l.length

variable (l : List ℝ)
variable (h_lenl : l.length = 7)
variable (h_avg7 : average_of_list l = 60)
variable (h_l3 : List ℝ := l.take 3)
variable (h_l4 : List ℝ := l.drop 3)
variable (h_avg3 : average_of_list h_l3 = 55)

theorem avg_last_four_is_63_75 : average_of_list h_l4 = 63.75 :=
by
  sorry

end NUMINAMATH_GPT_avg_last_four_is_63_75_l256_25611


namespace NUMINAMATH_GPT_math_problem_l256_25639

noncomputable def proof_problem (n : ℝ) (A B : ℝ) : Prop :=
  A = n^2 ∧ B = n^2 + 1 ∧ (1 * n^4 + 2 * n^2 + 3 + 2 * (n^2 + 1) + 1 = 5 * (2 * n^2 + 1)) → 
  A + B = 7 + 4 * Real.sqrt 2

theorem math_problem (n : ℝ) (A B : ℝ) :
  proof_problem n A B :=
sorry

end NUMINAMATH_GPT_math_problem_l256_25639


namespace NUMINAMATH_GPT_average_monthly_increase_l256_25612

theorem average_monthly_increase (x : ℝ) (turnover_january turnover_march : ℝ)
  (h_jan : turnover_january = 2)
  (h_mar : turnover_march = 2.88)
  (h_growth : turnover_march = turnover_january * (1 + x) * (1 + x)) :
  x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_increase_l256_25612


namespace NUMINAMATH_GPT_correct_option_division_l256_25628

theorem correct_option_division (x : ℝ) : 
  (-6 * x^3) / (-2 * x^2) = 3 * x :=
by 
  sorry

end NUMINAMATH_GPT_correct_option_division_l256_25628


namespace NUMINAMATH_GPT_range_of_x_function_l256_25660

open Real

theorem range_of_x_function : 
  ∀ x : ℝ, (x + 1 >= 0) ∧ (x - 3 ≠ 0) ↔ (x >= -1) ∧ (x ≠ 3) := 
by 
  sorry 

end NUMINAMATH_GPT_range_of_x_function_l256_25660


namespace NUMINAMATH_GPT_cook_one_potato_l256_25689

theorem cook_one_potato (total_potatoes cooked_potatoes remaining_potatoes remaining_time : ℕ) 
  (h1 : total_potatoes = 15) 
  (h2 : cooked_potatoes = 6) 
  (h3 : remaining_time = 72)
  (h4 : remaining_potatoes = total_potatoes - cooked_potatoes) :
  (remaining_time / remaining_potatoes) = 8 :=
by
  sorry

end NUMINAMATH_GPT_cook_one_potato_l256_25689


namespace NUMINAMATH_GPT_largest_A_l256_25690

def F (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem largest_A :
  ∃ n₁ n₂ n₃ n₄ n₅ n₆ : ℕ,
  (0 < n₁ ∧ 0 < n₂ ∧ 0 < n₃ ∧ 0 < n₄ ∧ 0 < n₅ ∧ 0 < n₆) ∧
  ∀ a, (1 ≤ a ∧ a ≤ 53590) -> 
    (F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) :=
sorry

end NUMINAMATH_GPT_largest_A_l256_25690


namespace NUMINAMATH_GPT_extracurricular_books_counts_l256_25651

theorem extracurricular_books_counts 
  (a b c d : ℕ)
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by
  sorry

end NUMINAMATH_GPT_extracurricular_books_counts_l256_25651


namespace NUMINAMATH_GPT_problem_l256_25669

noncomputable def K : ℕ := 36
noncomputable def L : ℕ := 147
noncomputable def M : ℕ := 56

theorem problem (h1 : 4 / 7 = K / 63) (h2 : 4 / 7 = 84 / L) (h3 : 4 / 7 = M / 98) :
  (K + L + M) = 239 :=
by
  sorry

end NUMINAMATH_GPT_problem_l256_25669


namespace NUMINAMATH_GPT_tenth_term_is_98415_over_262144_l256_25632

def first_term : ℚ := 5
def common_ratio : ℚ := 3 / 4

def tenth_term_geom_seq (a r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem tenth_term_is_98415_over_262144 :
  tenth_term_geom_seq first_term common_ratio 10 = 98415 / 262144 :=
sorry

end NUMINAMATH_GPT_tenth_term_is_98415_over_262144_l256_25632


namespace NUMINAMATH_GPT_arctan_sum_l256_25672

theorem arctan_sum (θ₁ θ₂ : ℝ) (h₁ : θ₁ = Real.arctan (1/2))
                              (h₂ : θ₂ = Real.arctan 2) :
  θ₁ + θ₂ = Real.pi / 2 :=
by
  have : θ₁ + θ₂ + Real.pi / 2 = Real.pi := sorry
  linarith

end NUMINAMATH_GPT_arctan_sum_l256_25672


namespace NUMINAMATH_GPT_midpoint_uniqueness_l256_25622

-- Define a finite set of points in the plane
axiom S : Finset (ℝ × ℝ)

-- Define what it means for P to be the midpoint of a segment
def is_midpoint (P A A' : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + A'.1) / 2 ∧ P.2 = (A.2 + A'.2) / 2

-- Statement of the problem
theorem midpoint_uniqueness (P Q : ℝ × ℝ) :
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint P A A') →
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint Q A A') →
  P = Q :=
sorry

end NUMINAMATH_GPT_midpoint_uniqueness_l256_25622


namespace NUMINAMATH_GPT_cave_depth_l256_25605

theorem cave_depth (current_depth remaining_distance : ℕ) (h₁ : current_depth = 849) (h₂ : remaining_distance = 369) :
  current_depth + remaining_distance = 1218 :=
by
  sorry

end NUMINAMATH_GPT_cave_depth_l256_25605


namespace NUMINAMATH_GPT_samantha_hike_distance_l256_25643

theorem samantha_hike_distance :
  let A : ℝ × ℝ := (0, 0)  -- Samantha's starting point
  let B := (0, 3)           -- Point after walking northward 3 miles
  let C := (5 / (2 : ℝ) * Real.sqrt 2, 3) -- Point after walking 5 miles at 45 degrees eastward
  (dist A C = Real.sqrt 86 / 2) :=
by
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (5 / (2 : ℝ) * Real.sqrt 2, 3)
  show dist A C = Real.sqrt 86 / 2
  sorry

end NUMINAMATH_GPT_samantha_hike_distance_l256_25643


namespace NUMINAMATH_GPT_tom_helicopter_hours_l256_25650

theorem tom_helicopter_hours (total_cost : ℤ) (cost_per_hour : ℤ) (days : ℤ) (h : total_cost = 450) (c : cost_per_hour = 75) (d : days = 3) :
  total_cost / cost_per_hour / days = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tom_helicopter_hours_l256_25650


namespace NUMINAMATH_GPT_gcd_198_286_l256_25692

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end NUMINAMATH_GPT_gcd_198_286_l256_25692


namespace NUMINAMATH_GPT_victor_wins_ratio_l256_25698

theorem victor_wins_ratio (victor_wins friend_wins : ℕ) (hvw : victor_wins = 36) (fw : friend_wins = 20) : (victor_wins : ℚ) / friend_wins = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_victor_wins_ratio_l256_25698


namespace NUMINAMATH_GPT_part1_part2_l256_25626

noncomputable def f (x a : ℝ) := (x + 1) * Real.log x - a * (x - 1)

theorem part1 : (∀ x a : ℝ, (x + 1) * Real.log x - a * (x - 1) = x - 1 → a = 1) := 
by sorry

theorem part2 (x : ℝ) (h : 1 < x ∧ x < 2) : 
  ( 1 / Real.log x - 1 / Real.log (x - 1) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l256_25626


namespace NUMINAMATH_GPT_sin2a_minus_cos2a_half_l256_25688

theorem sin2a_minus_cos2a_half (a : ℝ) (h : Real.tan (a - Real.pi / 4) = 1 / 2) :
  Real.sin (2 * a) - Real.cos a ^ 2 = 1 / 2 := 
sorry

end NUMINAMATH_GPT_sin2a_minus_cos2a_half_l256_25688


namespace NUMINAMATH_GPT_total_sets_needed_l256_25617

-- Conditions
variable (n : ℕ)

-- Theorem statement
theorem total_sets_needed : 3 * n = 3 * n :=
by sorry

end NUMINAMATH_GPT_total_sets_needed_l256_25617


namespace NUMINAMATH_GPT_symmetry_of_transformed_graphs_l256_25663

variable (f : ℝ → ℝ)

theorem symmetry_of_transformed_graphs :
  (∀ x, f x = f (-x)) → (∀ x, f (1 + x) = f (1 - x)) :=
by
  intro h_symmetry
  intro x
  sorry

end NUMINAMATH_GPT_symmetry_of_transformed_graphs_l256_25663


namespace NUMINAMATH_GPT_round_robin_odd_game_count_l256_25665

theorem round_robin_odd_game_count (n : ℕ) (h17 : n = 17) :
  ∃ p : ℕ, p < n ∧ (p % 2 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_round_robin_odd_game_count_l256_25665


namespace NUMINAMATH_GPT_solve_equation_l256_25685

noncomputable def is_solution (x : ℝ) : Prop :=
  (x / (2 * Real.sqrt 2) + (5 * Real.sqrt 2) / 2) * Real.sqrt (x^3 - 64 * x + 200) = x^2 + 6 * x - 40

noncomputable def conditions (x : ℝ) : Prop :=
  (x^3 - 64 * x + 200) ≥ 0 ∧ x ≥ 4

theorem solve_equation :
  (∀ x, is_solution x → conditions x) = (x = 6 ∨ x = 1 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l256_25685


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l256_25657

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) :
  ∀ n : ℕ, a n = 5 - 2 * n :=
by
  sorry

theorem max_sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) (h_sum : ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2) :
  S 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l256_25657


namespace NUMINAMATH_GPT_avg_age_diff_l256_25658

noncomputable def avg_age_team : ℕ := 28
noncomputable def num_players : ℕ := 11
noncomputable def wicket_keeper_age : ℕ := avg_age_team + 3
noncomputable def total_age_team : ℕ := avg_age_team * num_players
noncomputable def age_captain : ℕ := avg_age_team

noncomputable def total_age_remaining_players : ℕ := total_age_team - age_captain - wicket_keeper_age
noncomputable def num_remaining_players : ℕ := num_players - 2
noncomputable def avg_age_remaining_players : ℕ := total_age_remaining_players / num_remaining_players

theorem avg_age_diff :
  avg_age_team - avg_age_remaining_players = 3 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_diff_l256_25658


namespace NUMINAMATH_GPT_fido_yard_area_reach_l256_25601

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end NUMINAMATH_GPT_fido_yard_area_reach_l256_25601


namespace NUMINAMATH_GPT_find_y_l256_25691

theorem find_y (x y : ℤ) (h1 : x^2 - 2 * x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end NUMINAMATH_GPT_find_y_l256_25691


namespace NUMINAMATH_GPT_product_of_numbers_l256_25681

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x * y = 375 :=
sorry

end NUMINAMATH_GPT_product_of_numbers_l256_25681


namespace NUMINAMATH_GPT_wheel_center_travel_distance_l256_25625

theorem wheel_center_travel_distance (radius : ℝ) (revolutions : ℝ) (flat_surface : Prop) 
  (h_radius : radius = 2) (h_revolutions : revolutions = 2) : 
  radius * 2 * π * revolutions = 8 * π :=
by
  rw [h_radius, h_revolutions]
  simp [mul_assoc, mul_comm]
  sorry

end NUMINAMATH_GPT_wheel_center_travel_distance_l256_25625


namespace NUMINAMATH_GPT_no_y_satisfies_both_inequalities_l256_25671

variable (y : ℝ)

theorem no_y_satisfies_both_inequalities :
  ¬ (3 * y^2 - 4 * y - 5 < (y + 1)^2 ∧ (y + 1)^2 < 4 * y^2 - y - 1) :=
by
  sorry

end NUMINAMATH_GPT_no_y_satisfies_both_inequalities_l256_25671


namespace NUMINAMATH_GPT_sqrt_multiplication_l256_25679

theorem sqrt_multiplication :
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 :=
by
  -- statement follows
  sorry

end NUMINAMATH_GPT_sqrt_multiplication_l256_25679


namespace NUMINAMATH_GPT_exercise_b_c_values_l256_25648

open Set

universe u

theorem exercise_b_c_values : 
  ∀ (b c : ℝ), let U : Set ℝ := {2, 3, 5}
               let A : Set ℝ := {x | x^2 + b * x + c = 0}
               (U \ A = {2}) → (b = -8 ∧ c = 15) :=
by
  intros b c U A H
  let U : Set ℝ := {2, 3, 5}
  let A : Set ℝ := {x | x^2 + b * x + c = 0}
  have H1 : U \ A = {2} := H
  sorry

end NUMINAMATH_GPT_exercise_b_c_values_l256_25648


namespace NUMINAMATH_GPT_total_seven_flights_time_l256_25618

def time_for_nth_flight (n : ℕ) : ℕ :=
  25 + (n - 1) * 8

def total_time_for_flights (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => time_for_nth_flight (k + 1))

theorem total_seven_flights_time :
  total_time_for_flights 7 = 343 :=
  by
    sorry

end NUMINAMATH_GPT_total_seven_flights_time_l256_25618


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l256_25641

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l256_25641


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l256_25659

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_seq : arithmetic_sequence a)
  (h1 : a 0 + a 1 = 1)
  (h2 : a 2 + a 3 = 9) :
  a 4 + a 5 = 17 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l256_25659


namespace NUMINAMATH_GPT_average_primes_30_50_l256_25609

/-- The theorem statement for proving the average of all prime numbers between 30 and 50 is 39.8 -/
theorem average_primes_30_50 : (31 + 37 + 41 + 43 + 47) / 5 = 39.8 :=
  by
  sorry

end NUMINAMATH_GPT_average_primes_30_50_l256_25609


namespace NUMINAMATH_GPT_odd_divisors_l256_25638

-- Define p_1, p_2, p_3 as distinct prime numbers greater than 3
variables {p_1 p_2 p_3 : ℕ}
-- Define k, a, b, c as positive integers
variables {n k a b c : ℕ}

-- The conditions
def distinct_primes (p_1 p_2 p_3 : ℕ) : Prop :=
  p_1 > 3 ∧ p_2 > 3 ∧ p_3 > 3 ∧ p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_2 ≠ p_3

def is_n (n k p_1 p_2 p_3 a b c : ℕ) : Prop :=
  n = 2^k * p_1^a * p_2^b * p_3^c

def conditions (a b c : ℕ) : Prop :=
  a + b > c ∧ 1 ≤ b ∧ b ≤ c

-- The main statement
theorem odd_divisors
  (h_prime : distinct_primes p_1 p_2 p_3)
  (h_n : is_n n k p_1 p_2 p_3 a b c)
  (h_cond : conditions a b c) : 
  ∃ d : ℕ, d = (a + 1) * (b + 1) * (c + 1) :=
by sorry

end NUMINAMATH_GPT_odd_divisors_l256_25638


namespace NUMINAMATH_GPT_weight_of_172_is_around_60_316_l256_25670

noncomputable def weight_prediction (x : ℝ) : ℝ := 0.849 * x - 85.712

theorem weight_of_172_is_around_60_316 :
  ∀ (x : ℝ), x = 172 → abs (weight_prediction x - 60.316) < 1 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_172_is_around_60_316_l256_25670


namespace NUMINAMATH_GPT_num_4_digit_odd_distinct_l256_25686

theorem num_4_digit_odd_distinct : 
  ∃ n : ℕ, n = 5 * 4 * 3 * 2 :=
sorry

end NUMINAMATH_GPT_num_4_digit_odd_distinct_l256_25686


namespace NUMINAMATH_GPT_cube_angle_diagonals_l256_25675

theorem cube_angle_diagonals (q : ℝ) (h : q = 60) : 
  ∃ (d : String), d = "space diagonals" :=
by
  sorry

end NUMINAMATH_GPT_cube_angle_diagonals_l256_25675


namespace NUMINAMATH_GPT_find_coeff_sum_l256_25680

def parabola_eq (a b c : ℚ) (y : ℚ) : ℚ := a*y^2 + b*y + c

theorem find_coeff_sum 
  (a b c : ℚ)
  (h_eq : ∀ y, parabola_eq a b c y = - ((y + 6)^2) / 3 + 7)
  (h_pass : parabola_eq a b c 0 = 5) :
  a + b + c = -32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_coeff_sum_l256_25680


namespace NUMINAMATH_GPT_num_starting_lineups_l256_25699

def total_players := 15
def chosen_players := 3 -- Ace, Zeppo, Buddy already chosen
def remaining_players := total_players - chosen_players
def players_to_choose := 2 -- remaining players to choose

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem num_starting_lineups : combinations remaining_players players_to_choose = 66 := by
  sorry

end NUMINAMATH_GPT_num_starting_lineups_l256_25699
