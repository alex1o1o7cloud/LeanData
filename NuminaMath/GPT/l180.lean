import Mathlib

namespace NUMINAMATH_GPT_fraction_zero_x_value_l180_18026

theorem fraction_zero_x_value (x : ℝ) (h1 : 2 * x = 0) (h2 : x + 3 ≠ 0) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_x_value_l180_18026


namespace NUMINAMATH_GPT_elizabeth_revenue_per_investment_l180_18005

theorem elizabeth_revenue_per_investment :
  ∀ (revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth : ℕ),
    revenue_per_investment_banks = 500 →
    total_investments_banks = 8 →
    total_investments_elizabeth = 5 →
    revenue_difference = 500 →
    ((revenue_per_investment_banks * total_investments_banks) + revenue_difference) / total_investments_elizabeth = 900 :=
by
  intros revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth
  intros h_banks_revenue h_banks_investments h_elizabeth_investments h_revenue_difference
  sorry

end NUMINAMATH_GPT_elizabeth_revenue_per_investment_l180_18005


namespace NUMINAMATH_GPT_petya_can_restore_numbers_if_and_only_if_odd_l180_18032

def can_restore_numbers (n : ℕ) : Prop :=
  ∀ (V : Fin n → ℕ) (S : ℕ),
    ∃ f : Fin n → ℕ, 
    (∀ i : Fin n, 
      (V i) = f i ∨ 
      (S = f i)) ↔ (n % 2 = 1)

theorem petya_can_restore_numbers_if_and_only_if_odd (n : ℕ) : can_restore_numbers n ↔ n % 2 = 1 := 
by sorry

end NUMINAMATH_GPT_petya_can_restore_numbers_if_and_only_if_odd_l180_18032


namespace NUMINAMATH_GPT_initial_tomatoes_l180_18037

def t_picked : ℕ := 83
def t_left : ℕ := 14
def t_total : ℕ := t_picked + t_left

theorem initial_tomatoes : t_total = 97 := by
  rw [t_total]
  rfl

end NUMINAMATH_GPT_initial_tomatoes_l180_18037


namespace NUMINAMATH_GPT_prob_one_side_of_tri_in_decagon_is_half_l180_18008

noncomputable def probability_one_side_of_tri_in_decagon : ℚ :=
  let num_vertices := 10
  let total_triangles := Nat.choose num_vertices 3
  let favorable_triangles := 10 * 6
  favorable_triangles / total_triangles

theorem prob_one_side_of_tri_in_decagon_is_half :
  probability_one_side_of_tri_in_decagon = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_prob_one_side_of_tri_in_decagon_is_half_l180_18008


namespace NUMINAMATH_GPT_divisibility_by_7_l180_18028

theorem divisibility_by_7 (A X : Nat) (h1 : A < 10) (h2 : X < 10) : (100001 * A + 100010 * X) % 7 = 0 := 
by
  sorry

end NUMINAMATH_GPT_divisibility_by_7_l180_18028


namespace NUMINAMATH_GPT_distance_ratio_l180_18033

variable (d_RB d_BC : ℝ)

theorem distance_ratio
    (h1 : d_RB / 60 + d_BC / 20 ≠ 0)
    (h2 : 36 * (d_RB / 60 + d_BC / 20) = d_RB + d_BC) : 
    d_RB / d_BC = 2 := 
sorry

end NUMINAMATH_GPT_distance_ratio_l180_18033


namespace NUMINAMATH_GPT_sphere_radius_squared_l180_18006

theorem sphere_radius_squared (R x y z : ℝ)
  (h1 : 2 * Real.sqrt (R^2 - x^2 - y^2) = 5)
  (h2 : 2 * Real.sqrt (R^2 - x^2 - z^2) = 6)
  (h3 : 2 * Real.sqrt (R^2 - y^2 - z^2) = 7) :
  R^2 = 15 :=
sorry

end NUMINAMATH_GPT_sphere_radius_squared_l180_18006


namespace NUMINAMATH_GPT_square_area_on_parabola_l180_18018

theorem square_area_on_parabola (s : ℝ) (h : 0 < s) (hG : (3 + s)^2 - 6 * (3 + s) + 5 = -2 * s) : 
  (2 * s) * (2 * s) = 24 - 8 * Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_square_area_on_parabola_l180_18018


namespace NUMINAMATH_GPT_sum_of_first_11_terms_is_minus_66_l180_18077

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d 

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a n + a 1)) / 2

theorem sum_of_first_11_terms_is_minus_66 
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a)
  (h_roots : ∃ a2 a10, (a2 = a 2 ∧ a10 = a 10) ∧ (a2 + a10 = -12) ∧ (a2 * a10 = -8)) 
  : sum_of_first_n_terms a 11 = -66 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_11_terms_is_minus_66_l180_18077


namespace NUMINAMATH_GPT_red_bowling_balls_count_l180_18058

theorem red_bowling_balls_count (G R : ℕ) (h1 : G = R + 6) (h2 : R + G = 66) : R = 30 :=
by
  sorry

end NUMINAMATH_GPT_red_bowling_balls_count_l180_18058


namespace NUMINAMATH_GPT_college_application_distributions_l180_18045

theorem college_application_distributions : 
  let total_students := 6
  let colleges := 3
  ∃ n : ℕ, n = 540 ∧ 
    (n = (colleges^total_students - colleges * (2^total_students) + 
      (colleges.choose 2) * 1)) := sorry

end NUMINAMATH_GPT_college_application_distributions_l180_18045


namespace NUMINAMATH_GPT_remainder_14_div_5_l180_18053

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_14_div_5_l180_18053


namespace NUMINAMATH_GPT_dodecagon_area_l180_18076

theorem dodecagon_area (s : ℝ) (n : ℕ) (angles : ℕ → ℝ)
  (h_s : s = 10) (h_n : n = 12) 
  (h_angles : ∀ i, angles i = if i % 3 == 2 then 270 else 90) :
  ∃ area : ℝ, area = 500 := 
sorry

end NUMINAMATH_GPT_dodecagon_area_l180_18076


namespace NUMINAMATH_GPT_sakshi_days_l180_18043

theorem sakshi_days (Sakshi_efficiency Tanya_efficiency : ℝ) (Sakshi_days Tanya_days : ℝ) (h_efficiency : Tanya_efficiency = 1.25 * Sakshi_efficiency) (h_days : Tanya_days = 8) : Sakshi_days = 10 :=
by
  sorry

end NUMINAMATH_GPT_sakshi_days_l180_18043


namespace NUMINAMATH_GPT_solve_problem_l180_18094

noncomputable def problem_statement : Prop :=
  ∀ (tons_to_pounds : ℕ) 
    (packet_weight_pounds : ℕ) 
    (packet_weight_ounces : ℕ)
    (num_packets : ℕ)
    (bag_capacity_tons : ℕ)
    (X : ℕ),
    tons_to_pounds = 2300 →
    packet_weight_pounds = 16 →
    packet_weight_ounces = 4 →
    num_packets = 1840 →
    bag_capacity_tons = 13 →
    X = (packet_weight_ounces * bag_capacity_tons * tons_to_pounds) / 
        ((bag_capacity_tons * tons_to_pounds) - (num_packets * packet_weight_pounds)) →
    X = 16

theorem solve_problem : problem_statement :=
by sorry

end NUMINAMATH_GPT_solve_problem_l180_18094


namespace NUMINAMATH_GPT_solve_for_xy_l180_18055

theorem solve_for_xy (x y : ℝ) (h1 : 3 * x ^ 2 - 9 * y ^ 2 = 0) (h2 : x + y = 5) :
    (x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
    (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_xy_l180_18055


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l180_18067

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l180_18067


namespace NUMINAMATH_GPT_triangle_angle_A_eq_60_l180_18073

theorem triangle_angle_A_eq_60 (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_tan : (Real.tan A) / (Real.tan B) = (2 * c - b) / b) : 
  A = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_A_eq_60_l180_18073


namespace NUMINAMATH_GPT_trapezoid_two_heights_l180_18061

-- Define trivially what a trapezoid is, in terms of having two parallel sides.
structure Trapezoid :=
(base1 base2 : ℝ)
(height1 height2 : ℝ)
(has_two_heights : height1 = height2)

theorem trapezoid_two_heights (T : Trapezoid) : ∃ h1 h2 : ℝ, h1 = h2 :=
by
  use T.height1
  use T.height2
  exact T.has_two_heights

end NUMINAMATH_GPT_trapezoid_two_heights_l180_18061


namespace NUMINAMATH_GPT_flute_player_count_l180_18089

-- Define the total number of people in the orchestra
def total_people : Nat := 21

-- Define the number of people in each section
def sebastian : Nat := 1
def brass : Nat := 4 + 2 + 1
def strings : Nat := 3 + 1 + 1
def woodwinds_excluding_flutes : Nat := 3
def maestro : Nat := 1

-- Calculate the number of accounted people
def accounted_people : Nat := sebastian + brass + strings + woodwinds_excluding_flutes + maestro

-- State the number of flute players
def flute_players : Nat := total_people - accounted_people

-- The theorem stating the number of flute players
theorem flute_player_count : flute_players = 4 := by
  unfold flute_players accounted_people total_people sebastian brass strings woodwinds_excluding_flutes maestro
  -- Need to evaluate the expressions step by step to reach the final number 4.
  -- (Or simply "sorry" since we are skipping the proof steps)
  sorry

end NUMINAMATH_GPT_flute_player_count_l180_18089


namespace NUMINAMATH_GPT_colleen_paid_more_l180_18078

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end NUMINAMATH_GPT_colleen_paid_more_l180_18078


namespace NUMINAMATH_GPT_students_bought_pencils_l180_18096

theorem students_bought_pencils (h1 : 2 * 2 + 6 * 3 + 2 * 1 = 24) : 
  2 + 6 + 2 = 10 := by
  sorry

end NUMINAMATH_GPT_students_bought_pencils_l180_18096


namespace NUMINAMATH_GPT_verify_optionD_is_correct_l180_18079

-- Define the equations as options
def optionA : Prop := -abs (-6) = 6
def optionB : Prop := -(-6) = -6
def optionC : Prop := abs (-6) = -6
def optionD : Prop := -(-6) = 6

-- The proof problem to verify option D is correct
theorem verify_optionD_is_correct : optionD :=
by
  sorry

end NUMINAMATH_GPT_verify_optionD_is_correct_l180_18079


namespace NUMINAMATH_GPT_find_number_l180_18099

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l180_18099


namespace NUMINAMATH_GPT_four_x_sq_plus_nine_y_sq_l180_18080

theorem four_x_sq_plus_nine_y_sq (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 9)
  (h2 : x * y = -12) : 
  4 * x^2 + 9 * y^2 = 225 := 
by
  sorry

end NUMINAMATH_GPT_four_x_sq_plus_nine_y_sq_l180_18080


namespace NUMINAMATH_GPT_final_result_after_subtracting_15_l180_18090

theorem final_result_after_subtracting_15 :
  ∀ (n : ℕ) (r : ℕ) (f : ℕ),
  n = 120 → 
  r = n / 6 → 
  f = r - 15 → 
  f = 5 :=
by
  intros n r f hn hr hf
  have h1 : n = 120 := hn
  have h2 : r = n / 6 := hr
  have h3 : f = r - 15 := hf
  sorry

end NUMINAMATH_GPT_final_result_after_subtracting_15_l180_18090


namespace NUMINAMATH_GPT_initial_kids_count_l180_18050

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_initial_kids_count_l180_18050


namespace NUMINAMATH_GPT_correctness_of_statements_l180_18093

theorem correctness_of_statements (p q : Prop) (x y : ℝ) : 
  (¬ (p ∧ q) → (p ∨ q)) ∧
  ((xy = 0) → ¬(x^2 + y^2 = 0)) ∧
  ¬(∀ (L P : ℝ → ℝ), (∃ x, L x = P x) ↔ (∃ x, L x = P x ∧ ∀ x₁ x₂, x₁ ≠ x₂ → L x₁ ≠ P x₂)) →
  (0 + 1 + 0 = 1) :=
by
  sorry

end NUMINAMATH_GPT_correctness_of_statements_l180_18093


namespace NUMINAMATH_GPT_zero_points_in_intervals_l180_18051

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x - Real.log x

theorem zero_points_in_intervals :
  (∀ x : ℝ, x ∈ Set.Ioo (1 / Real.exp 1) 1 → f x ≠ 0) ∧
  (∃ x : ℝ, x ∈ Set.Ioo 1 (Real.exp 1) ∧ f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_zero_points_in_intervals_l180_18051


namespace NUMINAMATH_GPT_food_bank_remaining_after_four_weeks_l180_18087

def week1_donated : ℝ := 40
def week1_given_out : ℝ := 0.6 * week1_donated
def week1_remaining : ℝ := week1_donated - week1_given_out

def week2_donated : ℝ := 1.5 * week1_donated
def week2_given_out : ℝ := 0.7 * week2_donated
def week2_remaining : ℝ := week2_donated - week2_given_out
def total_remaining_after_week2 : ℝ := week1_remaining + week2_remaining

def week3_donated : ℝ := 1.25 * week2_donated
def week3_given_out : ℝ := 0.8 * week3_donated
def week3_remaining : ℝ := week3_donated - week3_given_out
def total_remaining_after_week3 : ℝ := total_remaining_after_week2 + week3_remaining

def week4_donated : ℝ := 0.9 * week3_donated
def week4_given_out : ℝ := 0.5 * week4_donated
def week4_remaining : ℝ := week4_donated - week4_given_out
def total_remaining_after_week4 : ℝ := total_remaining_after_week3 + week4_remaining

theorem food_bank_remaining_after_four_weeks : total_remaining_after_week4 = 82.75 := by
  sorry

end NUMINAMATH_GPT_food_bank_remaining_after_four_weeks_l180_18087


namespace NUMINAMATH_GPT_Tony_can_add_4_pairs_of_underwear_l180_18088

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end NUMINAMATH_GPT_Tony_can_add_4_pairs_of_underwear_l180_18088


namespace NUMINAMATH_GPT_find_other_endpoint_l180_18025

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ) 
  (h_mid_x : x_m = (x_1 + x_2) / 2)
  (h_mid_y : y_m = (y_1 + y_2) / 2)
  (h_x_m : x_m = 3)
  (h_y_m : y_m = 4)
  (h_x_1 : x_1 = 0)
  (h_y_1 : y_1 = -1) :
  (x_2, y_2) = (6, 9) :=
sorry

end NUMINAMATH_GPT_find_other_endpoint_l180_18025


namespace NUMINAMATH_GPT_compute_expression_l180_18013

theorem compute_expression : 12 + 5 * (4 - 9)^2 - 3 = 134 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l180_18013


namespace NUMINAMATH_GPT_paint_floor_cost_l180_18065

theorem paint_floor_cost :
  ∀ (L : ℝ) (rate : ℝ)
  (condition1 : L = 3 * (L / 3))
  (condition2 : L = 19.595917942265423)
  (condition3 : rate = 5),
  rate * (L * (L / 3)) = 640 :=
by
  intros L rate condition1 condition2 condition3
  sorry

end NUMINAMATH_GPT_paint_floor_cost_l180_18065


namespace NUMINAMATH_GPT_determine_n_for_11111_base_n_is_perfect_square_l180_18007

theorem determine_n_for_11111_base_n_is_perfect_square:
  ∃ m : ℤ, m^2 = 3^4 + 3^3 + 3^2 + 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_n_for_11111_base_n_is_perfect_square_l180_18007


namespace NUMINAMATH_GPT_percent_same_grades_l180_18036

theorem percent_same_grades 
    (total_students same_A same_B same_C same_D same_E : ℕ)
    (h_total_students : total_students = 40)
    (h_same_A : same_A = 3)
    (h_same_B : same_B = 5)
    (h_same_C : same_C = 6)
    (h_same_D : same_D = 2)
    (h_same_E : same_E = 1):
    ((same_A + same_B + same_C + same_D + same_E : ℚ) / total_students * 100) = 42.5 :=
by
  sorry

end NUMINAMATH_GPT_percent_same_grades_l180_18036


namespace NUMINAMATH_GPT_Tobias_monthly_allowance_l180_18042

noncomputable def monthly_allowance (shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways : ℕ) : ℕ :=
  (shoes_cost + change - (num_lawns * lawn_charge + num_driveways * driveway_charge)) / monthly_saving_period

theorem Tobias_monthly_allowance :
  let shoes_cost := 95
  let monthly_saving_period := 3
  let lawn_charge := 15
  let driveway_charge := 7
  let change := 15
  let num_lawns := 4
  let num_driveways := 5
  monthly_allowance shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways = 5 :=
by
  sorry

end NUMINAMATH_GPT_Tobias_monthly_allowance_l180_18042


namespace NUMINAMATH_GPT_part_a_part_b_l180_18057

theorem part_a (a : Fin 10 → ℤ) : ∃ i j : Fin 10, i ≠ j ∧ 27 ∣ (a i)^3 - (a j)^3 := sorry
theorem part_b (b : Fin 8 → ℤ) : ∃ i j : Fin 8, i ≠ j ∧ 27 ∣ (b i)^3 - (b j)^3 := sorry

end NUMINAMATH_GPT_part_a_part_b_l180_18057


namespace NUMINAMATH_GPT_solve_for_x_l180_18060

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l180_18060


namespace NUMINAMATH_GPT_study_time_in_minutes_l180_18001

theorem study_time_in_minutes :
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60 = 540 :=
by
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  sorry

end NUMINAMATH_GPT_study_time_in_minutes_l180_18001


namespace NUMINAMATH_GPT_subproblem1_l180_18056

theorem subproblem1 (a : ℝ) : a^3 * a + (2 * a^2)^2 = 5 * a^4 := 
by sorry

end NUMINAMATH_GPT_subproblem1_l180_18056


namespace NUMINAMATH_GPT_carter_drum_sticks_l180_18064

def sets_per_show (used : ℕ) (tossed : ℕ) : ℕ := used + tossed

def total_sets (sets_per_show : ℕ) (num_shows : ℕ) : ℕ := sets_per_show * num_shows

theorem carter_drum_sticks :
  sets_per_show 8 10 * 45 = 810 :=
by
  sorry

end NUMINAMATH_GPT_carter_drum_sticks_l180_18064


namespace NUMINAMATH_GPT_farmer_potatoes_initial_l180_18021

theorem farmer_potatoes_initial (P : ℕ) (h1 : 175 + P - 172 = 80) : P = 77 :=
by {
  sorry
}

end NUMINAMATH_GPT_farmer_potatoes_initial_l180_18021


namespace NUMINAMATH_GPT_probability_of_two_red_balls_l180_18044

-- Define the total number of balls, number of red balls, and number of white balls
def total_balls := 6
def red_balls := 4
def white_balls := 2
def drawn_balls := 2

-- Define the combination formula
def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to choose 2 red balls from 4
def ways_to_choose_red := choose 4 2

-- The number of ways to choose any 2 balls from the total of 6
def ways_to_choose_any := choose 6 2

-- The corresponding probability
def probability := ways_to_choose_red / ways_to_choose_any

-- The theorem we want to prove
theorem probability_of_two_red_balls :
  probability = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_red_balls_l180_18044


namespace NUMINAMATH_GPT_largest_class_students_l180_18062

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 95) :
  x = 23 :=
by
  sorry

end NUMINAMATH_GPT_largest_class_students_l180_18062


namespace NUMINAMATH_GPT_diego_apples_weight_l180_18081

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end NUMINAMATH_GPT_diego_apples_weight_l180_18081


namespace NUMINAMATH_GPT_max_non_intersecting_diagonals_l180_18075

theorem max_non_intersecting_diagonals (n : ℕ) (h : n ≥ 3) :
  ∃ k, k ≤ n - 3 ∧ (∀ m, m > k → ¬(m ≤ n - 3)) :=
by
  sorry

end NUMINAMATH_GPT_max_non_intersecting_diagonals_l180_18075


namespace NUMINAMATH_GPT_find_f_5_l180_18012

-- Definitions from conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 - b * x + 2

-- Stating the theorem
theorem find_f_5 (a b : ℝ) (h : f (-5) a b = 17) : f 5 a b = -13 :=
by
  sorry

end NUMINAMATH_GPT_find_f_5_l180_18012


namespace NUMINAMATH_GPT_product_fraction_l180_18049

open Int

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem product_fraction :
  (product first_six_composites : ℚ) / (product (first_three_primes ++ next_three_composites) : ℚ) = 24 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_product_fraction_l180_18049


namespace NUMINAMATH_GPT_probability_both_truth_l180_18047

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_truth (hA : P_A = 0.55) (hB : P_B = 0.60) :
  P_A * P_B = 0.33 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_truth_l180_18047


namespace NUMINAMATH_GPT_carls_garden_area_is_correct_l180_18046

-- Define the conditions
def isRectangle (length width : ℕ) : Prop :=
∃ l w, l * w = length * width

def validFencePosts (shortSidePosts longSidePosts totalPosts : ℕ) : Prop :=
∃ x, totalPosts = 2 * x + 2 * (2 * x) - 4 ∧ x = shortSidePosts

def validSpacing (shortSideSpaces longSideSpaces : ℕ) : Prop :=
shortSideSpaces = 4 * (shortSideSpaces - 1) ∧ longSideSpaces = 4 * (longSideSpaces - 1)

def correctArea (shortSide longSide expectedArea : ℕ) : Prop :=
shortSide * longSide = expectedArea

-- Prove the conditions lead to the expected area
theorem carls_garden_area_is_correct :
  ∃ shortSide longSide,
  isRectangle shortSide longSide ∧
  validFencePosts 5 10 24 ∧
  validSpacing 5 10 ∧
  correctArea (4 * (5-1)) (4 * (10-1)) 576 :=
by
  sorry

end NUMINAMATH_GPT_carls_garden_area_is_correct_l180_18046


namespace NUMINAMATH_GPT_contradiction_method_assumption_l180_18095

-- Definitions for three consecutive positive integers
variables {a b c : ℕ}

-- Definitions for the proposition and its negation
def consecutive_integers (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1
def at_least_one_divisible_by_2 (a b c : ℕ) : Prop := a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0
def all_not_divisible_by_2 (a b c : ℕ) : Prop := a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem contradiction_method_assumption (a b c : ℕ) (h : consecutive_integers a b c) :
  (¬ at_least_one_divisible_by_2 a b c) ↔ all_not_divisible_by_2 a b c :=
by sorry

end NUMINAMATH_GPT_contradiction_method_assumption_l180_18095


namespace NUMINAMATH_GPT_sixth_grader_count_l180_18039

theorem sixth_grader_count : 
  ∃ x y : ℕ, (3 / 7) * x = (1 / 3) * y ∧ x + y = 140 ∧ x = 61 :=
by {
  sorry  -- Proof not required
}

end NUMINAMATH_GPT_sixth_grader_count_l180_18039


namespace NUMINAMATH_GPT_transportation_trucks_l180_18015

theorem transportation_trucks (boxes : ℕ) (total_weight : ℕ) (box_weight : ℕ) (truck_capacity : ℕ) :
  (total_weight = 10) → (∀ (b : ℕ), b ≤ boxes → box_weight ≤ 1) → (truck_capacity = 3) → 
  ∃ (trucks : ℕ), trucks = 5 :=
by
  sorry

end NUMINAMATH_GPT_transportation_trucks_l180_18015


namespace NUMINAMATH_GPT_fraction_cost_of_raisins_l180_18068

variable (cost_raisins cost_nuts total_cost_raisins total_cost_nuts total_cost : ℝ)

theorem fraction_cost_of_raisins (h1 : cost_nuts = 3 * cost_raisins)
                                 (h2 : total_cost_raisins = 4 * cost_raisins)
                                 (h3 : total_cost_nuts = 4 * cost_nuts)
                                 (h4 : total_cost = total_cost_raisins + total_cost_nuts) :
                                 (total_cost_raisins / total_cost) = (1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_fraction_cost_of_raisins_l180_18068


namespace NUMINAMATH_GPT_cheerleader_total_l180_18041

theorem cheerleader_total 
  (size2 : ℕ)
  (size6 : ℕ)
  (size12 : ℕ)
  (h1 : size2 = 4)
  (h2 : size6 = 10)
  (h3 : size12 = size6 / 2) :
  size2 + size6 + size12 = 19 :=
by
  sorry

end NUMINAMATH_GPT_cheerleader_total_l180_18041


namespace NUMINAMATH_GPT_sin2theta_plus_cos2theta_l180_18063

theorem sin2theta_plus_cos2theta (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin2theta_plus_cos2theta_l180_18063


namespace NUMINAMATH_GPT_each_person_paid_l180_18048

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end NUMINAMATH_GPT_each_person_paid_l180_18048


namespace NUMINAMATH_GPT_maisy_earnings_increase_l180_18074

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end NUMINAMATH_GPT_maisy_earnings_increase_l180_18074


namespace NUMINAMATH_GPT_max_sum_of_abc_l180_18023

theorem max_sum_of_abc (A B C : ℕ) (h1 : A * B * C = 1386) (h2 : A ≠ B) (h3 : A ≠ C) (h4 : B ≠ C) : 
  A + B + C ≤ 88 :=
sorry

end NUMINAMATH_GPT_max_sum_of_abc_l180_18023


namespace NUMINAMATH_GPT_v3_value_at_2_l180_18086

def f (x : ℝ) : ℝ :=
  x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

def v3 (x : ℝ) : ℝ :=
  ((x - 12) * x + 60) * x - 160

theorem v3_value_at_2 :
  v3 2 = -80 :=
by
  sorry

end NUMINAMATH_GPT_v3_value_at_2_l180_18086


namespace NUMINAMATH_GPT_player5_points_combination_l180_18040

theorem player5_points_combination :
  ∃ (two_point_shots three_pointers free_throws : ℕ), 
  (two_point_shots * 2 + three_pointers * 3 + free_throws * 1 = 14) :=
sorry

end NUMINAMATH_GPT_player5_points_combination_l180_18040


namespace NUMINAMATH_GPT_golf_money_l180_18071

-- Definitions based on conditions
def cost_per_round : ℤ := 80
def number_of_rounds : ℤ := 5

-- The theorem/problem statement
theorem golf_money : cost_per_round * number_of_rounds = 400 := 
by {
  -- Proof steps would go here, but to skip the proof, we use sorry
  sorry
}

end NUMINAMATH_GPT_golf_money_l180_18071


namespace NUMINAMATH_GPT_harmonic_mean_closest_integer_l180_18083

theorem harmonic_mean_closest_integer (a b : ℝ) (ha : a = 1) (hb : b = 2016) :
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  sorry

end NUMINAMATH_GPT_harmonic_mean_closest_integer_l180_18083


namespace NUMINAMATH_GPT_total_scarves_l180_18019

def total_yarns_red : ℕ := 2
def total_yarns_blue : ℕ := 6
def total_yarns_yellow : ℕ := 4
def scarves_per_yarn : ℕ := 3

theorem total_scarves : 
  (total_yarns_red * scarves_per_yarn) + 
  (total_yarns_blue * scarves_per_yarn) + 
  (total_yarns_yellow * scarves_per_yarn) = 36 := 
by
  sorry

end NUMINAMATH_GPT_total_scarves_l180_18019


namespace NUMINAMATH_GPT_compound_interest_amount_l180_18059

/-
Given:
- Principal amount P = 5000
- Annual interest rate r = 0.07
- Time period t = 15 years

We aim to prove:
A = 5000 * (1 + 0.07) ^ 15 = 13795.15
-/
theorem compound_interest_amount :
  let P : ℝ := 5000
  let r : ℝ := 0.07
  let t : ℝ := 15
  let A : ℝ := P * (1 + r) ^ t
  A = 13795.15 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_amount_l180_18059


namespace NUMINAMATH_GPT_amount_with_r_l180_18091

theorem amount_with_r (p q r : ℝ) (h₁ : p + q + r = 7000) (h₂ : r = (2 / 3) * (p + q)) : r = 2800 :=
  sorry

end NUMINAMATH_GPT_amount_with_r_l180_18091


namespace NUMINAMATH_GPT_ksyusha_travel_time_l180_18027

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end NUMINAMATH_GPT_ksyusha_travel_time_l180_18027


namespace NUMINAMATH_GPT_fraction_equality_l180_18085
-- Import the necessary library

-- The proof statement
theorem fraction_equality : (16 + 8) / (4 - 2) = 12 := 
by {
  -- Inserting 'sorry' to indicate that the proof is omitted
  sorry
}

end NUMINAMATH_GPT_fraction_equality_l180_18085


namespace NUMINAMATH_GPT_factor_expression_l180_18038

variable (a : ℝ)

theorem factor_expression : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l180_18038


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_minimal_triangle_area_eq_line_l180_18098

-- Part (1)
theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ M : ℝ × ℝ, M = (-1, -2) ∧
    (∀ m : ℝ, (2 + m) * (-1) + (1 - 2 * m) * (-2) + (4 - 3 * m) = 0) := by
  sorry

-- Part (2)
theorem minimal_triangle_area_eq_line :
  ∃ k : ℝ, k = -2 ∧ 
    (∀ x y : ℝ, y = k * (x + 1) - 2 ↔ y = 2 * x + 4) := by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_minimal_triangle_area_eq_line_l180_18098


namespace NUMINAMATH_GPT_vaishali_total_stripes_l180_18054

def total_stripes (hats_with_3_stripes hats_with_4_stripes hats_with_no_stripes : ℕ) 
  (hats_with_5_stripes hats_with_7_stripes hats_with_1_stripe : ℕ) 
  (hats_with_10_stripes hats_with_2_stripes : ℕ)
  (stripes_per_hat_with_3 stripes_per_hat_with_4 stripes_per_hat_with_no : ℕ)
  (stripes_per_hat_with_5 stripes_per_hat_with_7 stripes_per_hat_with_1 : ℕ)
  (stripes_per_hat_with_10 stripes_per_hat_with_2 : ℕ) : ℕ :=
  hats_with_3_stripes * stripes_per_hat_with_3 +
  hats_with_4_stripes * stripes_per_hat_with_4 +
  hats_with_no_stripes * stripes_per_hat_with_no +
  hats_with_5_stripes * stripes_per_hat_with_5 +
  hats_with_7_stripes * stripes_per_hat_with_7 +
  hats_with_1_stripe * stripes_per_hat_with_1 +
  hats_with_10_stripes * stripes_per_hat_with_10 +
  hats_with_2_stripes * stripes_per_hat_with_2

#eval total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2 -- 71

theorem vaishali_total_stripes : (total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2) = 71 :=
by
  sorry

end NUMINAMATH_GPT_vaishali_total_stripes_l180_18054


namespace NUMINAMATH_GPT_ellipse_perimeter_l180_18092

noncomputable def perimeter_of_triangle (a b : ℝ) (e : ℝ) : ℝ :=
  if (b = 4 ∧ e = 3 / 5 ∧ a = b / (1 - e^2) ^ (1 / 2))
  then 4 * a
  else 0

theorem ellipse_perimeter :
  let a : ℝ := 5
  let b : ℝ := 4
  let e : ℝ := 3 / 5
  4 * a = 20 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_perimeter_l180_18092


namespace NUMINAMATH_GPT_sequence_sum_S5_l180_18011

theorem sequence_sum_S5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 2 = 4)
  (h2 : ∀ n, a (n + 1) = 2 * S n + 1)
  (h3 : ∀ n, S (n + 1) - S n = a (n + 1)) :
  S 5 = 121 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_S5_l180_18011


namespace NUMINAMATH_GPT_evaluate_product_l180_18022

theorem evaluate_product (m : ℕ) (h : m = 3) : (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 :=
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_product_l180_18022


namespace NUMINAMATH_GPT_initial_points_l180_18016

theorem initial_points (n : ℕ) (h : 16 * n - 15 = 225) : n = 15 :=
sorry

end NUMINAMATH_GPT_initial_points_l180_18016


namespace NUMINAMATH_GPT_find_y_l180_18084

theorem find_y (x y : ℕ) (h_pos_y : 0 < y) (h_rem : x % y = 7) (h_div : x = 86 * y + (1 / 10) * y) :
  y = 70 :=
sorry

end NUMINAMATH_GPT_find_y_l180_18084


namespace NUMINAMATH_GPT_least_xy_value_l180_18031

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end NUMINAMATH_GPT_least_xy_value_l180_18031


namespace NUMINAMATH_GPT_largest_possible_package_size_l180_18024

theorem largest_possible_package_size :
  ∃ (p : ℕ), Nat.gcd 60 36 = p ∧ p = 12 :=
by
  use 12
  sorry -- The proof is skipped as per instructions

end NUMINAMATH_GPT_largest_possible_package_size_l180_18024


namespace NUMINAMATH_GPT_athena_total_spent_l180_18030

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end NUMINAMATH_GPT_athena_total_spent_l180_18030


namespace NUMINAMATH_GPT_find_range_of_t_l180_18069

variable {f : ℝ → ℝ}

-- Definitions for the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ ⦃x y⦄, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x > f y

-- Given the conditions, we need to prove the statement
theorem find_range_of_t (h_odd : is_odd_function f)
    (h_decreasing : is_decreasing_on f (-1) 1)
    (h_inequality : ∀ t : ℝ, -1 < t ∧ t < 1 → f (1 - t) + f (1 - t^2) < 0) :
  ∀ t, -1 < t ∧ t < 1 → 0 < t ∧ t < 1 :=
  by
  sorry

end NUMINAMATH_GPT_find_range_of_t_l180_18069


namespace NUMINAMATH_GPT_Malcom_cards_after_giving_away_half_l180_18020

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end NUMINAMATH_GPT_Malcom_cards_after_giving_away_half_l180_18020


namespace NUMINAMATH_GPT_smallest_fraction_numerator_l180_18003

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end NUMINAMATH_GPT_smallest_fraction_numerator_l180_18003


namespace NUMINAMATH_GPT_n_is_perfect_square_l180_18000

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k ^ 2

theorem n_is_perfect_square (a b c d : ℤ) (h : a + b + c + d = 0) : 
  is_perfect_square ((ab - cd) * (bc - ad) * (ca - bd)) := 
  sorry

end NUMINAMATH_GPT_n_is_perfect_square_l180_18000


namespace NUMINAMATH_GPT_original_price_l180_18004

theorem original_price (P : ℝ) (h_discount : 0.75 * P = 560): P = 746.68 :=
sorry

end NUMINAMATH_GPT_original_price_l180_18004


namespace NUMINAMATH_GPT_square_diagonal_y_coordinate_l180_18034

theorem square_diagonal_y_coordinate 
(point_vertex : ℝ × ℝ) 
(x_int : ℝ) 
(area_square : ℝ) 
(y_int : ℝ) :
(point_vertex = (-6, -4)) →
(x_int = 3) →
(area_square = 324) →
(y_int = 5) → 
y_int = 5 := 
by
  intros h1 h2 h3 h4
  exact h4

end NUMINAMATH_GPT_square_diagonal_y_coordinate_l180_18034


namespace NUMINAMATH_GPT_smallest_positive_integer_l180_18029

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 66666 * n = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l180_18029


namespace NUMINAMATH_GPT_slope_of_line_l180_18070

theorem slope_of_line {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : 5 / x + 4 / y = 0) :
  ∃ x₁ x₂ y₁ y₂, (5 / x₁ + 4 / y₁ = 0) ∧ (5 / x₂ + 4 / y₂ = 0) ∧ 
  (y₂ - y₁) / (x₂ - x₁) = -4 / 5 :=
sorry

end NUMINAMATH_GPT_slope_of_line_l180_18070


namespace NUMINAMATH_GPT_top_face_not_rotated_by_90_l180_18082

-- Define the cube and the conditions of rolling and returning
structure Cube :=
  (initial_top_face_orientation : ℕ) -- an integer representation of the orientation of the top face
  (position : ℤ × ℤ) -- (x, y) coordinates on a 2D plane

def rolls_over_edges (c : Cube) : Cube :=
  sorry -- placeholder for the actual rolling operation

def returns_to_original_position (c : Cube) (original : Cube) : Prop :=
  c.position = original.position ∧ c.initial_top_face_orientation = original.initial_top_face_orientation

-- The main theorem to prove
theorem top_face_not_rotated_by_90 {c : Cube} (original : Cube) :
  returns_to_original_position c original → c.initial_top_face_orientation ≠ (original.initial_top_face_orientation + 1) % 4 :=
sorry

end NUMINAMATH_GPT_top_face_not_rotated_by_90_l180_18082


namespace NUMINAMATH_GPT_polynomial_identity_l180_18066

theorem polynomial_identity (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end NUMINAMATH_GPT_polynomial_identity_l180_18066


namespace NUMINAMATH_GPT_ratio_PA_AB_l180_18009

theorem ratio_PA_AB (A B C P : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup P]
  (h1 : ∃ AC CB : ℕ, AC = 2 * CB)
  (h2 : ∃ PA AB : ℕ, PA = 4 * (AB / 5)) :
  PA / AB = 4 / 5 := sorry

end NUMINAMATH_GPT_ratio_PA_AB_l180_18009


namespace NUMINAMATH_GPT_quilt_cost_proof_l180_18072

-- Definitions for conditions
def length := 7
def width := 8
def cost_per_sq_foot := 40

-- Definition for the calculation of the area
def area := length * width

-- Definition for the calculation of the cost
def total_cost := area * cost_per_sq_foot

-- Theorem stating the final proof
theorem quilt_cost_proof : total_cost = 2240 := by
  sorry

end NUMINAMATH_GPT_quilt_cost_proof_l180_18072


namespace NUMINAMATH_GPT_min_paint_steps_l180_18017

-- Checkered square of size 2021x2021 where all cells initially white.
-- Ivan selects two cells and paints them black.
-- Cells with at least one black neighbor by side are painted black simultaneously each step.

-- Define a function to represent the steps required to paint the square black
noncomputable def min_steps_to_paint_black (n : ℕ) (a b : ℕ × ℕ) : ℕ :=
  sorry -- Placeholder for the actual function definition, as we're focusing on the statement.

-- Define the specific instance of the problem
def square_size := 2021
def initial_cells := ((505, 1010), (1515, 1010))

-- Theorem statement: Proving the minimal number of steps required is 1515
theorem min_paint_steps : min_steps_to_paint_black square_size initial_cells.1 initial_cells.2 = 1515 :=
sorry

end NUMINAMATH_GPT_min_paint_steps_l180_18017


namespace NUMINAMATH_GPT_parametric_to_cartesian_l180_18097

variable (R t : ℝ)

theorem parametric_to_cartesian (x y : ℝ) (h1 : x = R * Real.cos t) (h2 : y = R * Real.sin t) : 
  x^2 + y^2 = R^2 := 
by
  sorry

end NUMINAMATH_GPT_parametric_to_cartesian_l180_18097


namespace NUMINAMATH_GPT_trig_identity_l180_18002

open Real

theorem trig_identity 
  (θ : ℝ)
  (h : tan (π / 4 + θ) = 3) : 
  sin (2 * θ) - 2 * cos θ ^ 2 = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l180_18002


namespace NUMINAMATH_GPT_distance_from_origin_l180_18052

theorem distance_from_origin :
  ∃ (m : ℝ), m = Real.sqrt (108 + 8 * Real.sqrt 10) ∧
              (∃ (x y : ℝ), y = 8 ∧ 
                            (x - 2)^2 + (y - 5)^2 = 49 ∧ 
                            x = 2 + 2 * Real.sqrt 10 ∧ 
                            m = Real.sqrt ((x^2) + (y^2))) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_l180_18052


namespace NUMINAMATH_GPT_steps_A_l180_18014

theorem steps_A (t_A t_B : ℝ) (a e t : ℝ) :
  t_A = 3 * t_B →
  t_B = t / 75 →
  a + e * t = 100 →
  75 + e * t = 100 →
  a = 75 :=
by sorry

end NUMINAMATH_GPT_steps_A_l180_18014


namespace NUMINAMATH_GPT_unique_line_through_A_parallel_to_a_l180_18010

variables {Point Line Plane : Type}
variables {α β : Plane}
variables {a l : Line}
variables {A : Point}

-- Definitions are necessary from conditions in step a)
def parallel_to (a b : Line) : Prop := sorry -- Definition that two lines are parallel
def contains (p : Plane) (x : Point) : Prop := sorry -- Definition that a plane contains a point
def line_parallel_to_plane (a : Line) (p : Plane) : Prop := sorry -- Definition that a line is parallel to a plane

-- Given conditions in the proof problem
variable (a_parallel_α : line_parallel_to_plane a α)
variable (A_in_α : contains α A)

-- Statement to be proven: There is only one line that passes through point A and is parallel to line a, and that line is within plane α.
theorem unique_line_through_A_parallel_to_a : 
  ∃! l : Line, contains α A ∧ parallel_to l a := sorry

end NUMINAMATH_GPT_unique_line_through_A_parallel_to_a_l180_18010


namespace NUMINAMATH_GPT_minimum_y_value_y_at_4_eq_6_l180_18035

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 2)

theorem minimum_y_value (x : ℝ) (h : x > 2) : y x ≥ 6 :=
sorry

theorem y_at_4_eq_6 : y 4 = 6 :=
sorry

end NUMINAMATH_GPT_minimum_y_value_y_at_4_eq_6_l180_18035
