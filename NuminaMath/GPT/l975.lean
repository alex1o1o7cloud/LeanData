import Mathlib

namespace NUMINAMATH_GPT_triangle_area_l975_97559

/-- Given a triangle with a perimeter of 20 cm and an inradius of 2.5 cm,
prove that its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 20) (h2 : inradius = 2.5) :
  area = 25 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l975_97559


namespace NUMINAMATH_GPT_total_water_output_l975_97507

theorem total_water_output (flow_rate: ℚ) (time_duration: ℕ) (total_water: ℚ) :
  flow_rate = 2 + 2 / 3 → time_duration = 9 → total_water = 24 →
  flow_rate * time_duration = total_water :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_total_water_output_l975_97507


namespace NUMINAMATH_GPT_sam_dads_dimes_l975_97510

theorem sam_dads_dimes (original_dimes new_dimes given_dimes : ℕ) 
  (h1 : original_dimes = 9)
  (h2 : new_dimes = 16)
  (h3 : new_dimes = original_dimes + given_dimes) : 
  given_dimes = 7 := 
by 
  sorry

end NUMINAMATH_GPT_sam_dads_dimes_l975_97510


namespace NUMINAMATH_GPT_range_of_a_l975_97562

open Set

def real_intervals (a : ℝ) : Prop :=
  let S := {x : ℝ | (x - 2)^2 > 9}
  let T := Ioo a (a + 8)
  S ∪ T = univ → -3 < a ∧ a < -1

theorem range_of_a (a : ℝ) : real_intervals a :=
sorry

end NUMINAMATH_GPT_range_of_a_l975_97562


namespace NUMINAMATH_GPT_compare_constants_l975_97514

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 2 / 2
noncomputable def c := Real.log 3 / 3

theorem compare_constants : b < c ∧ c < a := by
  sorry

end NUMINAMATH_GPT_compare_constants_l975_97514


namespace NUMINAMATH_GPT_probability_diff_color_correct_l975_97573

noncomputable def probability_diff_color (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  (red_balls * yellow_balls) / ((total_balls * (total_balls - 1)) / 2)

theorem probability_diff_color_correct :
  probability_diff_color 5 3 2 = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_diff_color_correct_l975_97573


namespace NUMINAMATH_GPT_min_quadratic_expression_l975_97597

theorem min_quadratic_expression:
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_min_quadratic_expression_l975_97597


namespace NUMINAMATH_GPT_female_computer_literacy_l975_97592

variable (E F C M CM CF : ℕ)

theorem female_computer_literacy (hE : E = 1200) 
                                (hF : F = 720) 
                                (hC : C = 744) 
                                (hM : M = 480) 
                                (hCM : CM = 240) 
                                (hCF : CF = C - CM) : 
                                CF = 504 :=
by {
  sorry
}

end NUMINAMATH_GPT_female_computer_literacy_l975_97592


namespace NUMINAMATH_GPT_six_digit_number_count_correct_l975_97508

-- Defining the 6-digit number formation problem
def count_six_digit_numbers_with_conditions : Nat := 1560

-- Problem statement
theorem six_digit_number_count_correct :
  count_six_digit_numbers_with_conditions = 1560 :=
sorry

end NUMINAMATH_GPT_six_digit_number_count_correct_l975_97508


namespace NUMINAMATH_GPT_polar_line_equation_l975_97599

theorem polar_line_equation
  (rho theta : ℝ)
  (h1 : rho = 4 * Real.cos theta)
  (h2 : ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → x = 2)
  : rho * Real.cos theta = 2 :=
sorry

end NUMINAMATH_GPT_polar_line_equation_l975_97599


namespace NUMINAMATH_GPT_solution_inequality_l975_97553

open Real

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- State the theorem for the given proof problem
theorem solution_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_solution_inequality_l975_97553


namespace NUMINAMATH_GPT_length_of_first_video_l975_97503

theorem length_of_first_video
  (total_time : ℕ)
  (second_video_time : ℕ)
  (last_two_videos_time : ℕ)
  (first_video_time : ℕ)
  (total_seconds : total_time = 510)
  (second_seconds : second_video_time = 4 * 60 + 30)
  (last_videos_seconds : last_two_videos_time = 60 + 60)
  (total_watch_time : total_time = second_video_time + last_two_videos_time + first_video_time) :
  first_video_time = 120 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_video_l975_97503


namespace NUMINAMATH_GPT_find_m_l975_97517

theorem find_m (m n : ℤ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 0 :=
sorry

end NUMINAMATH_GPT_find_m_l975_97517


namespace NUMINAMATH_GPT_interior_angle_solution_l975_97575

noncomputable def interior_angle_of_inscribed_triangle (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) : ℝ :=
  (1 / 2) * (x + 80)

theorem interior_angle_solution (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) :
  interior_angle_of_inscribed_triangle x h = 64 :=
sorry

end NUMINAMATH_GPT_interior_angle_solution_l975_97575


namespace NUMINAMATH_GPT_polynomial_evaluation_l975_97511

theorem polynomial_evaluation 
  (x : ℝ) (h : x^2 - 3*x - 10 = 0 ∧ x > 0) :
  x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l975_97511


namespace NUMINAMATH_GPT_range_of_k_l975_97528

theorem range_of_k (k : ℝ) : (4 < k ∧ k < 9 ∧ k ≠ 13 / 2) ↔ (k ∈ Set.Ioo 4 (13 / 2) ∪ Set.Ioo (13 / 2) 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l975_97528


namespace NUMINAMATH_GPT_smallest_number_is_C_l975_97591

-- Define the conditions
def A := 18 + 38
def B := A - 26
def C := B / 3

-- Proof statement: C is the smallest number among A, B, and C
theorem smallest_number_is_C : C = min A (min B C) :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_is_C_l975_97591


namespace NUMINAMATH_GPT_water_leakage_l975_97588

theorem water_leakage (initial_quarts : ℚ) (remaining_gallons : ℚ)
  (conversion_rate : ℚ) (expected_leakage : ℚ) :
  initial_quarts = 4 ∧ remaining_gallons = 0.33 ∧ conversion_rate = 4 ∧ 
  expected_leakage = 2.68 →
  initial_quarts - remaining_gallons * conversion_rate = expected_leakage :=
by 
  sorry

end NUMINAMATH_GPT_water_leakage_l975_97588


namespace NUMINAMATH_GPT_exists_natural_numbers_solving_equation_l975_97505

theorem exists_natural_numbers_solving_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end NUMINAMATH_GPT_exists_natural_numbers_solving_equation_l975_97505


namespace NUMINAMATH_GPT_xy_square_sum_l975_97545

variable (x y : ℝ)

theorem xy_square_sum : (y + 6 = (x - 3)^2) →
                        (x + 6 = (y - 3)^2) →
                        (x ≠ y) →
                        x^2 + y^2 = 43 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_xy_square_sum_l975_97545


namespace NUMINAMATH_GPT_inequality_proof_l975_97583

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy: 0 < y) (hz : 0 < z):
  ( ( (x + y + z) / 3 ) ^ (x + y + z) ) ≤ x^x * y^y * z^z ∧ x^x * y^y * z^z ≤ ( (x^2 + y^2 + z^2) / (x + y + z) ) ^ (x + y + z) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l975_97583


namespace NUMINAMATH_GPT_present_age_of_B_l975_97561

theorem present_age_of_B 
    (a b : ℕ) 
    (h1 : a + 10 = 2 * (b - 10)) 
    (h2 : a = b + 12) : 
    b = 42 := by 
  sorry

end NUMINAMATH_GPT_present_age_of_B_l975_97561


namespace NUMINAMATH_GPT_megan_folders_l975_97513

def filesOnComputer : Nat := 93
def deletedFiles : Nat := 21
def filesPerFolder : Nat := 8

theorem megan_folders:
  let remainingFiles := filesOnComputer - deletedFiles
  (remainingFiles / filesPerFolder) = 9 := by
    sorry

end NUMINAMATH_GPT_megan_folders_l975_97513


namespace NUMINAMATH_GPT_one_neither_prime_nor_composite_l975_97555

/-- Definition of a prime number in the natural numbers -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of a composite number in the natural numbers -/
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n 

/-- Theorem stating that the number 1 is neither prime nor composite -/
theorem one_neither_prime_nor_composite : ¬is_prime 1 ∧ ¬is_composite 1 :=
sorry

end NUMINAMATH_GPT_one_neither_prime_nor_composite_l975_97555


namespace NUMINAMATH_GPT_count_integers_in_range_l975_97557

theorem count_integers_in_range : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℤ, (-7 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 8) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_count_integers_in_range_l975_97557


namespace NUMINAMATH_GPT_basketball_match_scores_l975_97586

theorem basketball_match_scores :
  ∃ (a r b d : ℝ), (a = b) ∧ (a * (1 + r + r^2 + r^3) < 120) ∧
  (4 * b + 6 * d < 120) ∧ ((a * (1 + r + r^2 + r^3) - (4 * b + 6 * d)) = 3) ∧
  a + b + (a * r + (b + d)) = 35.5 :=
sorry

end NUMINAMATH_GPT_basketball_match_scores_l975_97586


namespace NUMINAMATH_GPT_largest_pos_int_divisible_l975_97500

theorem largest_pos_int_divisible (n : ℕ) (h1 : n > 0) (h2 : n + 11 ∣ n^3 + 101) : n = 1098 :=
sorry

end NUMINAMATH_GPT_largest_pos_int_divisible_l975_97500


namespace NUMINAMATH_GPT_statement_not_always_true_l975_97580

theorem statement_not_always_true 
  (a b c d : ℝ)
  (h1 : (a + b) / (3 * a - b) = (b + c) / (3 * b - c))
  (h2 : (b + c) / (3 * b - c) = (c + d) / (3 * c - d))
  (h3 : (c + d) / (3 * c - d) = (d + a) / (3 * d - a))
  (h4 : (d + a) / (3 * d - a) = (a + b) / (3 * a - b)) :
  a^2 + b^2 + c^2 + d^2 ≠ ab + bc + cd + da :=
by {
  sorry
}

end NUMINAMATH_GPT_statement_not_always_true_l975_97580


namespace NUMINAMATH_GPT_solve_for_y_l975_97568

theorem solve_for_y {y : ℝ} : (y - 5)^4 = 16 → y = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l975_97568


namespace NUMINAMATH_GPT_positive_integer_condition_l975_97552

theorem positive_integer_condition (p : ℕ) (hp : 0 < p) : 
  (∃ k : ℤ, k > 0 ∧ 4 * p + 17 = k * (3 * p - 8)) ↔ p = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_integer_condition_l975_97552


namespace NUMINAMATH_GPT_quadratic_always_positive_l975_97546

theorem quadratic_always_positive (k : ℝ) :
  ∀ x : ℝ, x^2 - (k - 4) * x + k - 7 > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_always_positive_l975_97546


namespace NUMINAMATH_GPT_paperclips_exceed_target_in_days_l975_97530

def initial_paperclips := 3
def ratio := 2
def target_paperclips := 200

theorem paperclips_exceed_target_in_days :
  ∃ k : ℕ, initial_paperclips * ratio ^ k > target_paperclips ∧ k = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_paperclips_exceed_target_in_days_l975_97530


namespace NUMINAMATH_GPT_jerry_age_l975_97551

theorem jerry_age (M J : ℕ) (hM : M = 24) (hCond : M = 4 * J - 20) : J = 11 := by
  sorry

end NUMINAMATH_GPT_jerry_age_l975_97551


namespace NUMINAMATH_GPT_min_value_expr_l975_97509

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l975_97509


namespace NUMINAMATH_GPT_sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l975_97566

theorem sum_of_two_greatest_values_of_b (b : Real) 
  (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 ∨ b = -2.5 ∨ b = -2 :=
sorry

theorem sum_of_two_greatest_values (b1 b2 : Real)
  (hb1 : 4 * b1 ^ 4 - 41 * b1 ^ 2 + 100 = 0)
  (hb2 : 4 * b2 ^ 4 - 41 * b2 ^ 2 + 100 = 0) :
  b1 = 2.5 → b2 = 2 → b1 + b2 = 4.5 :=
sorry

end NUMINAMATH_GPT_sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l975_97566


namespace NUMINAMATH_GPT_fractions_equiv_conditions_l975_97518

theorem fractions_equiv_conditions (x y z : ℝ) (h₁ : 2 * x - z ≠ 0) (h₂ : z ≠ 0) : 
  ((2 * x + y) / (2 * x - z) = y / -z) ↔ (y = -z) :=
by
  sorry

end NUMINAMATH_GPT_fractions_equiv_conditions_l975_97518


namespace NUMINAMATH_GPT_sqrt_eight_simplify_l975_97526

theorem sqrt_eight_simplify : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_eight_simplify_l975_97526


namespace NUMINAMATH_GPT_handbag_monday_price_l975_97542

theorem handbag_monday_price (initial_price : ℝ) (primary_discount : ℝ) (additional_discount : ℝ)
(h_initial_price : initial_price = 250)
(h_primary_discount : primary_discount = 0.4)
(h_additional_discount : additional_discount = 0.1) :
(initial_price - initial_price * primary_discount) - ((initial_price - initial_price * primary_discount) * additional_discount) = 135 := by
  sorry

end NUMINAMATH_GPT_handbag_monday_price_l975_97542


namespace NUMINAMATH_GPT_additional_employees_hired_l975_97532

-- Conditions
def initial_employees : ℕ := 500
def hourly_wage : ℕ := 12
def daily_hours : ℕ := 10
def weekly_days : ℕ := 5
def weekly_hours := daily_hours * weekly_days
def monthly_weeks : ℕ := 4
def monthly_hours_per_employee := weekly_hours * monthly_weeks
def wage_per_employee_per_month := monthly_hours_per_employee * hourly_wage

-- Given new payroll
def new_monthly_payroll : ℕ := 1680000

-- Calculate the initial payroll
def initial_monthly_payroll := initial_employees * wage_per_employee_per_month

-- Statement of the proof problem
theorem additional_employees_hired :
  (new_monthly_payroll - initial_monthly_payroll) / wage_per_employee_per_month = 200 :=
by
  sorry

end NUMINAMATH_GPT_additional_employees_hired_l975_97532


namespace NUMINAMATH_GPT_find_a_l975_97582

def set_A : Set ℝ := {x | x^2 + x - 6 = 0}

def set_B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_a (a : ℝ) : set_A ∪ set_B a = set_A ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) := 
by
  sorry

end NUMINAMATH_GPT_find_a_l975_97582


namespace NUMINAMATH_GPT_painter_total_rooms_l975_97538

theorem painter_total_rooms (hours_per_room : ℕ) (rooms_already_painted : ℕ) (additional_painting_hours : ℕ) 
  (h1 : hours_per_room = 8) (h2 : rooms_already_painted = 8) (h3 : additional_painting_hours = 16) : 
  rooms_already_painted + (additional_painting_hours / hours_per_room) = 10 := by
  sorry

end NUMINAMATH_GPT_painter_total_rooms_l975_97538


namespace NUMINAMATH_GPT_pipe_B_time_l975_97515

theorem pipe_B_time (C : ℝ) (T : ℝ) 
    (h1 : 2 / 3 * C + C / 3 = C)
    (h2 : C / 36 + C / (3 * T) = C / 14.4) 
    (h3 : T > 0) : 
    T = 8 := 
sorry

end NUMINAMATH_GPT_pipe_B_time_l975_97515


namespace NUMINAMATH_GPT_count_ordered_pairs_no_distinct_real_solutions_l975_97567

theorem count_ordered_pairs_no_distinct_real_solutions :
  {n : Nat // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (4 * b^2 - 4 * c ≤ 0) ∧ (4 * c^2 - 4 * b ≤ 0) ∧ n = 1} :=
sorry

end NUMINAMATH_GPT_count_ordered_pairs_no_distinct_real_solutions_l975_97567


namespace NUMINAMATH_GPT_largest_garden_is_candace_and_difference_is_100_l975_97506

-- Define the dimensions of the gardens
def area_alice : Nat := 30 * 50
def area_bob : Nat := 35 * 45
def area_candace : Nat := 40 * 40

-- The proof goal
theorem largest_garden_is_candace_and_difference_is_100 :
  area_candace > area_alice ∧ area_candace > area_bob ∧ area_candace - area_alice = 100 := by
    sorry

end NUMINAMATH_GPT_largest_garden_is_candace_and_difference_is_100_l975_97506


namespace NUMINAMATH_GPT_no_integer_roots_p_eq_2016_l975_97571

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end NUMINAMATH_GPT_no_integer_roots_p_eq_2016_l975_97571


namespace NUMINAMATH_GPT_tom_current_yellow_tickets_l975_97548

-- Definitions based on conditions provided
def yellow_to_red (y : ℕ) : ℕ := y * 10
def red_to_blue (r : ℕ) : ℕ := r * 10
def yellow_to_blue (y : ℕ) : ℕ := (yellow_to_red y) * 10

def tom_red_tickets : ℕ := 3
def tom_blue_tickets : ℕ := 7

def tom_total_blue_tickets : ℕ := (red_to_blue tom_red_tickets) + tom_blue_tickets
def tom_needed_blue_tickets : ℕ := 163

-- Proving that Tom currently has 2 yellow tickets
theorem tom_current_yellow_tickets : (tom_total_blue_tickets + tom_needed_blue_tickets) / yellow_to_blue 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_tom_current_yellow_tickets_l975_97548


namespace NUMINAMATH_GPT_count_possible_x_values_l975_97537

theorem count_possible_x_values (x y : ℕ) (H : (x + 2) * (y + 2) - x * y = x * y) :
  (∃! x, ∃ y, (x - 2) * (y - 2) = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_possible_x_values_l975_97537


namespace NUMINAMATH_GPT_total_time_six_laps_l975_97572

-- Defining the constants and conditions
def total_distance : Nat := 500
def speed_part1 : Nat := 3
def distance_part1 : Nat := 150
def speed_part2 : Nat := 6
def distance_part2 : Nat := total_distance - distance_part1
def laps : Nat := 6

-- Calculating the times based on conditions
def time_part1 := distance_part1 / speed_part1
def time_part2 := distance_part2 / speed_part2
def time_per_lap := time_part1 + time_part2
def total_time := laps * time_per_lap

-- The goal is to prove the total time is 10 minutes and 48 seconds (648 seconds)
theorem total_time_six_laps : total_time = 648 :=
-- proof would go here
sorry

end NUMINAMATH_GPT_total_time_six_laps_l975_97572


namespace NUMINAMATH_GPT_base4_to_base10_conversion_l975_97520

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end NUMINAMATH_GPT_base4_to_base10_conversion_l975_97520


namespace NUMINAMATH_GPT_zoe_correct_percentage_l975_97577

noncomputable def t : ℝ := sorry  -- total number of problems
noncomputable def chloe_alone_correct : ℝ := 0.70 * (1/3 * t)  -- Chloe's correct answers alone
noncomputable def chloe_total_correct : ℝ := 0.85 * t  -- Chloe's overall correct answers
noncomputable def together_correct : ℝ := chloe_total_correct - chloe_alone_correct  -- Problems solved correctly together
noncomputable def zoe_alone_correct : ℝ := 0.85 * (1/3 * t)  -- Zoe's correct answers alone
noncomputable def zoe_total_correct : ℝ := zoe_alone_correct + together_correct  -- Zoe's total correct answers
noncomputable def zoe_percentage_correct : ℝ := (zoe_total_correct / t) * 100  -- Convert to percentage

theorem zoe_correct_percentage : zoe_percentage_correct = 90 := 
by
  sorry

end NUMINAMATH_GPT_zoe_correct_percentage_l975_97577


namespace NUMINAMATH_GPT_dog_food_vs_cat_food_l975_97501

-- Define the quantities of dog food and cat food
def dog_food : ℕ := 600
def cat_food : ℕ := 327

-- Define the problem as a statement asserting the required difference
theorem dog_food_vs_cat_food : dog_food - cat_food = 273 := by
  sorry

end NUMINAMATH_GPT_dog_food_vs_cat_food_l975_97501


namespace NUMINAMATH_GPT_a4_eq_12_l975_97576

-- Definitions of the sequences and conditions
def S (n : ℕ) : ℕ := 
  -- sum of the first n terms, initially undefined
  sorry  

def a (n : ℕ) : ℕ := 
  -- terms of the sequence, initially undefined
  sorry  

-- Given conditions
axiom a2_eq_3 : a 2 = 3
axiom Sn_recurrence : ∀ n ≥ 2, S (n + 1) = 2 * S n

-- Statement to prove
theorem a4_eq_12 : a 4 = 12 :=
  sorry

end NUMINAMATH_GPT_a4_eq_12_l975_97576


namespace NUMINAMATH_GPT_find_K_3_15_10_l975_97595

def K (x y z : ℚ) : ℚ := 
  x / y + y / z + z / x + (x + y) / z

theorem find_K_3_15_10 : K 3 15 10 = 41 / 6 := 
  by
  sorry

end NUMINAMATH_GPT_find_K_3_15_10_l975_97595


namespace NUMINAMATH_GPT_find_n_l975_97550

-- Declaring the necessary context and parameters.
variable (n : ℕ)

-- Defining the condition described in the problem.
def reposting_equation (n : ℕ) : Prop := 1 + n + n^2 = 111

-- Stating the theorem to prove that for n = 10, the reposting equation holds.
theorem find_n : ∃ (n : ℕ), reposting_equation n ∧ n = 10 :=
by
  use 10
  unfold reposting_equation
  sorry

end NUMINAMATH_GPT_find_n_l975_97550


namespace NUMINAMATH_GPT_find_two_digit_number_l975_97584

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b

theorem find_two_digit_number (a b : ℕ) (h1 : a = 2 * b) (h2 : original_number b a = original_number a b - 36) : original_number a b = 84 :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l975_97584


namespace NUMINAMATH_GPT_absolute_value_condition_l975_97533

theorem absolute_value_condition (x : ℝ) (h : |x| = 32) : x = 32 ∨ x = -32 :=
sorry

end NUMINAMATH_GPT_absolute_value_condition_l975_97533


namespace NUMINAMATH_GPT_candy_distribution_l975_97578

theorem candy_distribution :
  let bags := 4
  let candies := 9
  (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 :=
by
  -- define variables for bags and candies
  let bags := 4
  let candies := 9
  have h : (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 := sorry
  exact h

end NUMINAMATH_GPT_candy_distribution_l975_97578


namespace NUMINAMATH_GPT_n_four_plus_n_squared_plus_one_not_prime_l975_97549

theorem n_four_plus_n_squared_plus_one_not_prime (n : ℤ) (h : n ≥ 2) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end NUMINAMATH_GPT_n_four_plus_n_squared_plus_one_not_prime_l975_97549


namespace NUMINAMATH_GPT_equal_areas_of_ngons_l975_97560

noncomputable def area_of_ngon (n : ℕ) (sides : Fin n → ℝ) (radius : ℝ) (circumference : ℝ) : ℝ := sorry

theorem equal_areas_of_ngons 
  (n : ℕ) 
  (sides1 sides2 : Fin n → ℝ) 
  (radius : ℝ) 
  (circumference : ℝ)
  (h_sides : ∀ i : Fin n, ∃ j : Fin n, sides1 i = sides2 j)
  (h_inscribed1 : area_of_ngon n sides1 radius circumference = area_of_ngon n sides1 radius circumference)
  (h_inscribed2 : area_of_ngon n sides2 radius circumference = area_of_ngon n sides2 radius circumference) :
  area_of_ngon n sides1 radius circumference = area_of_ngon n sides2 radius circumference :=
sorry

end NUMINAMATH_GPT_equal_areas_of_ngons_l975_97560


namespace NUMINAMATH_GPT_income_of_deceased_l975_97590

def average_income (total_income : ℕ) (members : ℕ) : ℕ :=
  total_income / members

theorem income_of_deceased
  (total_income_before : ℕ) (members_before : ℕ) (avg_income_before : ℕ)
  (total_income_after : ℕ) (members_after : ℕ) (avg_income_after : ℕ) :
  total_income_before = members_before * avg_income_before →
  total_income_after = members_after * avg_income_after →
  members_before = 4 →
  members_after = 3 →
  avg_income_before = 735 →
  avg_income_after = 650 →
  total_income_before - total_income_after = 990 :=
by
  sorry

end NUMINAMATH_GPT_income_of_deceased_l975_97590


namespace NUMINAMATH_GPT_part1_extreme_value_at_2_part2_increasing_function_l975_97536

noncomputable def f (a x : ℝ) := a * x - a / x - 2 * Real.log x

theorem part1_extreme_value_at_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y : ℝ, f a x ≥ f a y) → a = 4 / 5 ∧ f a 1/2 = 2 * Real.log 2 - 6 / 5 := by
  sorry

theorem part2_increasing_function (a : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (f a) x ≥ 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_GPT_part1_extreme_value_at_2_part2_increasing_function_l975_97536


namespace NUMINAMATH_GPT_convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l975_97534

theorem convert_deg_to_rad1 : 780 * (Real.pi / 180) = (13 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad2 : -1560 * (Real.pi / 180) = -(26 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad3 : 67.5 * (Real.pi / 180) = (3 * Real.pi) / 8 := sorry
theorem convert_rad_to_deg1 : -(10 * Real.pi / 3) * (180 / Real.pi) = -600 := sorry
theorem convert_rad_to_deg2 : (Real.pi / 12) * (180 / Real.pi) = 15 := sorry
theorem convert_rad_to_deg3 : (7 * Real.pi / 4) * (180 / Real.pi) = 315 := sorry

end NUMINAMATH_GPT_convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l975_97534


namespace NUMINAMATH_GPT_intersecting_rectangles_area_l975_97594

-- Define the dimensions of the rectangles
def rect1_length : ℝ := 12
def rect1_width : ℝ := 4
def rect2_length : ℝ := 7
def rect2_width : ℝ := 5

-- Define the areas of the individual rectangles
def area_rect1 : ℝ := rect1_length * rect1_width
def area_rect2 : ℝ := rect2_length * rect2_width

-- Assume overlapping region area
def area_overlap : ℝ := rect1_width * rect2_width

-- Define the total shaded area
def shaded_area : ℝ := area_rect1 + area_rect2 - area_overlap

-- Prove the shaded area is 63 square units
theorem intersecting_rectangles_area : shaded_area = 63 :=
by 
  -- Insert proof steps here, we only provide the theorem statement and leave the proof unfinished
  sorry

end NUMINAMATH_GPT_intersecting_rectangles_area_l975_97594


namespace NUMINAMATH_GPT_original_integer_is_21_l975_97579

theorem original_integer_is_21 (a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 29) 
  (h2 : (a + b + d) / 3 + c = 23) 
  (h3 : (a + c + d) / 3 + b = 21) 
  (h4 : (b + c + d) / 3 + a = 17) : 
  d = 21 :=
sorry

end NUMINAMATH_GPT_original_integer_is_21_l975_97579


namespace NUMINAMATH_GPT_multiplication_identity_l975_97502

theorem multiplication_identity : 32519 * 9999 = 324857481 := by
  sorry

end NUMINAMATH_GPT_multiplication_identity_l975_97502


namespace NUMINAMATH_GPT_avg_of_other_two_l975_97524

-- Definitions and conditions from the problem
def avg (l : List ℕ) : ℕ := l.sum / l.length

variables {A B C D E : ℕ}
variables (h_avg_five : avg [A, B, C, D, E] = 20)
variables (h_sum_three : A + B + C = 48)
variables (h_twice : A = 2 * B)

-- Theorem to prove
theorem avg_of_other_two (A B C D E : ℕ) 
  (h_avg_five : avg [A, B, C, D, E] = 20)
  (h_sum_three : A + B + C = 48)
  (h_twice : A = 2 * B) :
  avg [D, E] = 26 := 
  sorry

end NUMINAMATH_GPT_avg_of_other_two_l975_97524


namespace NUMINAMATH_GPT_satisfies_natural_solution_l975_97556

theorem satisfies_natural_solution (m : ℤ) :
  (∃ x : ℕ, x = 6 / (m - 1)) → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 7) :=
by
  sorry

end NUMINAMATH_GPT_satisfies_natural_solution_l975_97556


namespace NUMINAMATH_GPT_alpha_pi_over_four_sufficient_not_necessary_l975_97565

theorem alpha_pi_over_four_sufficient_not_necessary :
  (∀ α : ℝ, (α = (Real.pi / 4) → Real.cos α = Real.sqrt 2 / 2)) ∧
  (∃ α : ℝ, (Real.cos α = Real.sqrt 2 / 2) ∧ α ≠ (Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_GPT_alpha_pi_over_four_sufficient_not_necessary_l975_97565


namespace NUMINAMATH_GPT_min_value_n_constant_term_l975_97519

-- Define the problem statement
theorem min_value_n_constant_term (n r : ℕ) (h : 2 * n = 5 * r) : n = 5 :=
by sorry

end NUMINAMATH_GPT_min_value_n_constant_term_l975_97519


namespace NUMINAMATH_GPT_min_attendees_l975_97554

-- Define the constants and conditions
def writers : ℕ := 35
def min_editors : ℕ := 39
def x_max : ℕ := 26

-- Define the total number of people formula based on inclusion-exclusion principle
-- and conditions provided
def total_people (x : ℕ) : ℕ := writers + min_editors - x + 2 * x

-- Theorem to prove that the minimum number of attendees is 126
theorem min_attendees : ∃ x, x ≤ x_max ∧ total_people x = 126 :=
by
  use x_max
  sorry

end NUMINAMATH_GPT_min_attendees_l975_97554


namespace NUMINAMATH_GPT_cherry_pie_probability_l975_97569

noncomputable def probability_of_cherry_pie : Real :=
  let packets := ["KK", "KV", "VV"]
  let prob :=
    (1/3 * 1/4) + -- Case KK broken, then picking from KV or VV
    (1/6 * 1/2) + -- Case KV broken (cabbage found), picking cherry from KV
    (1/3 * 1) + -- Case VV broken (cherry found), remaining cherry picked
    (1/6 * 0) -- Case KV broken (cherry found), remaining cabbage
  prob

theorem cherry_pie_probability : probability_of_cherry_pie = 2 / 3 :=
  sorry

end NUMINAMATH_GPT_cherry_pie_probability_l975_97569


namespace NUMINAMATH_GPT_number_of_toothpicks_l975_97547

def num_horizontal_toothpicks(lines width : Nat) : Nat := lines * width
def num_vertical_toothpicks(lines height : Nat) : Nat := lines * height

theorem number_of_toothpicks (high wide : Nat) (missing : Nat) 
  (h_high : high = 15) (h_wide : wide = 15) (h_missing : missing = 1) : 
  num_horizontal_toothpicks (high + 1) wide + num_vertical_toothpicks (wide + 1) high - missing = 479 := by
  sorry

end NUMINAMATH_GPT_number_of_toothpicks_l975_97547


namespace NUMINAMATH_GPT_Rikki_earnings_l975_97531

theorem Rikki_earnings
  (price_per_word : ℝ := 0.01)
  (words_per_5_minutes : ℕ := 25)
  (total_minutes : ℕ := 120)
  (earning : ℝ := 6)
  : price_per_word * (words_per_5_minutes * (total_minutes / 5)) = earning := by
  sorry

end NUMINAMATH_GPT_Rikki_earnings_l975_97531


namespace NUMINAMATH_GPT_woman_worked_days_l975_97598

theorem woman_worked_days :
  ∃ (W I : ℕ), (W + I = 25) ∧ (20 * W - 5 * I = 450) ∧ W = 23 := by
  sorry

end NUMINAMATH_GPT_woman_worked_days_l975_97598


namespace NUMINAMATH_GPT_adjacent_girl_pairs_l975_97525

variable (boyCount girlCount : ℕ) 
variable (adjacentBoyPairs adjacentGirlPairs: ℕ)

theorem adjacent_girl_pairs
  (h1 : boyCount = 10)
  (h2 : girlCount = 15)
  (h3 : adjacentBoyPairs = 5) :
  adjacentGirlPairs = 10 :=
sorry

end NUMINAMATH_GPT_adjacent_girl_pairs_l975_97525


namespace NUMINAMATH_GPT_larger_number_is_34_l975_97593

theorem larger_number_is_34 (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_is_34_l975_97593


namespace NUMINAMATH_GPT_inequality_always_holds_l975_97564

theorem inequality_always_holds (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_holds_l975_97564


namespace NUMINAMATH_GPT_passing_percentage_correct_l975_97574

-- Define the conditions
def max_marks : ℕ := 500
def candidate_marks : ℕ := 180
def fail_by : ℕ := 45

-- Define the passing_marks based on given conditions
def passing_marks : ℕ := candidate_marks + fail_by

-- Theorem to prove: the passing percentage is 45%
theorem passing_percentage_correct : 
  (passing_marks / max_marks) * 100 = 45 := 
sorry

end NUMINAMATH_GPT_passing_percentage_correct_l975_97574


namespace NUMINAMATH_GPT_percentage_sold_correct_l975_97527

variables 
  (initial_cost : ℝ) 
  (tripled_value : ℝ) 
  (selling_price : ℝ) 
  (percentage_sold : ℝ)

def game_sold_percentage (initial_cost tripled_value selling_price percentage_sold : ℝ) :=
  tripled_value = initial_cost * 3 ∧ 
  selling_price = 240 ∧ 
  initial_cost = 200 ∧ 
  percentage_sold = (selling_price / tripled_value) * 100

theorem percentage_sold_correct : game_sold_percentage 200 (200 * 3) 240 40 :=
  by simp [game_sold_percentage]; sorry

end NUMINAMATH_GPT_percentage_sold_correct_l975_97527


namespace NUMINAMATH_GPT_num_female_managers_l975_97512

-- Definitions based on the conditions
def total_employees : ℕ := 250
def female_employees : ℕ := 90
def total_managers : ℕ := 40
def male_associates : ℕ := 160

-- Proof statement that computes the number of female managers
theorem num_female_managers : 
  (total_managers - (total_employees - female_employees - male_associates)) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_num_female_managers_l975_97512


namespace NUMINAMATH_GPT_range_of_a_l975_97596

noncomputable def operation (x y : ℝ) := x * (1 - y)

theorem range_of_a
  (a : ℝ)
  (hx : ∀ x : ℝ, operation (x - a) (x + a) < 1) :
  -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l975_97596


namespace NUMINAMATH_GPT_B_pow_99_identity_l975_97522

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_99_identity : (B ^ 99) = 1 := by
  sorry

end NUMINAMATH_GPT_B_pow_99_identity_l975_97522


namespace NUMINAMATH_GPT_shipping_cost_correct_l975_97529

noncomputable def shipping_cost (W : ℝ) : ℕ := 7 + 5 * (⌈W⌉₊ - 1)

theorem shipping_cost_correct (W : ℝ) : shipping_cost W = 5 * ⌈W⌉₊ + 2 :=
by
  sorry

end NUMINAMATH_GPT_shipping_cost_correct_l975_97529


namespace NUMINAMATH_GPT_unique_x_intersect_l975_97523

theorem unique_x_intersect (m : ℝ) (h : ∀ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 → ∀ y : ℝ, (m - 4) * y^2 - 2 * m * y - m - 6 = 0 → x = y) :
  m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end NUMINAMATH_GPT_unique_x_intersect_l975_97523


namespace NUMINAMATH_GPT_ab_value_l975_97570

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by 
  sorry

end NUMINAMATH_GPT_ab_value_l975_97570


namespace NUMINAMATH_GPT_simple_interest_time_period_l975_97543

variable (SI P R T : ℝ)

theorem simple_interest_time_period (h₁ : SI = 4016.25) (h₂ : P = 8925) (h₃ : R = 9) :
  (P * R * T) / 100 = SI ↔ T = 5 := by
  sorry

end NUMINAMATH_GPT_simple_interest_time_period_l975_97543


namespace NUMINAMATH_GPT_bullets_shot_per_person_l975_97540

-- Definitions based on conditions
def num_people : ℕ := 5
def initial_bullets_per_person : ℕ := 25
def total_remaining_bullets : ℕ := 25

-- Statement to prove
theorem bullets_shot_per_person (x : ℕ) :
  (initial_bullets_per_person * num_people - num_people * x) = total_remaining_bullets → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_bullets_shot_per_person_l975_97540


namespace NUMINAMATH_GPT_sphere_surface_area_l975_97541

theorem sphere_surface_area (R h : ℝ) (R_pos : 0 < R) (h_pos : 0 < h) :
  ∃ A : ℝ, A = 2 * Real.pi * R * h := 
sorry

end NUMINAMATH_GPT_sphere_surface_area_l975_97541


namespace NUMINAMATH_GPT_trader_sold_45_meters_l975_97589

-- Definitions based on conditions
def selling_price_total : ℕ := 4500
def profit_per_meter : ℕ := 12
def cost_price_per_meter : ℕ := 88
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof goal to show that the trader sold 45 meters of cloth
theorem trader_sold_45_meters : ∃ x : ℕ, selling_price_per_meter * x = selling_price_total ∧ x = 45 := 
by
  sorry

end NUMINAMATH_GPT_trader_sold_45_meters_l975_97589


namespace NUMINAMATH_GPT_compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l975_97516

def a_n (p n : ℕ) : ℕ := (2 * n + 1) ^ p
def b_n (p n : ℕ) : ℕ := (2 * n) ^ p + (2 * n - 1) ^ p

theorem compare_magnitude_p2_for_n1 :
  b_n 2 1 < a_n 2 1 := sorry

theorem compare_magnitude_p2_for_n2 :
  b_n 2 2 = a_n 2 2 := sorry

theorem compare_magnitude_p2_for_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  b_n 2 n > a_n 2 n := sorry

theorem compare_magnitude_p_eq_n_for_all_n (n : ℕ) :
  a_n n n ≥ b_n n n := sorry

end NUMINAMATH_GPT_compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l975_97516


namespace NUMINAMATH_GPT_seungho_more_marbles_l975_97563

variable (S H : ℕ)

-- Seungho gave 273 marbles to Hyukjin
def given_marbles : ℕ := 273

-- After giving 273 marbles, Seungho has 477 more marbles than Hyukjin
axiom marbles_condition : S - given_marbles = (H + given_marbles) + 477

theorem seungho_more_marbles (S H : ℕ) (marbles_condition : S - 273 = (H + 273) + 477) : S = H + 1023 :=
by
  sorry

end NUMINAMATH_GPT_seungho_more_marbles_l975_97563


namespace NUMINAMATH_GPT_cans_per_bag_l975_97585

def total_cans : ℕ := 42
def bags_saturday : ℕ := 4
def bags_sunday : ℕ := 3
def total_bags : ℕ := bags_saturday + bags_sunday

theorem cans_per_bag (h1 : total_cans = 42) (h2 : total_bags = 7) : total_cans / total_bags = 6 :=
by {
    -- proof body to be filled
    sorry
}

end NUMINAMATH_GPT_cans_per_bag_l975_97585


namespace NUMINAMATH_GPT_boat_speed_greater_than_current_l975_97535

theorem boat_speed_greater_than_current (U V : ℝ) (hU_gt_V : U > V)
  (h_equation : 1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) :
  U - V = 1 :=
sorry

end NUMINAMATH_GPT_boat_speed_greater_than_current_l975_97535


namespace NUMINAMATH_GPT_train_speed_is_72_kmph_l975_97521

-- Define the given conditions in Lean
def crossesMan (L V : ℝ) : Prop := L = 19 * V
def crossesPlatform (L V : ℝ) : Prop := L + 220 = 30 * V

-- The main theorem which states that the speed of the train is 72 km/h under given conditions
theorem train_speed_is_72_kmph (L V : ℝ) (h1 : crossesMan L V) (h2 : crossesPlatform L V) :
  (V * 3.6) = 72 := by
  -- We will provide a full proof here later
  sorry

end NUMINAMATH_GPT_train_speed_is_72_kmph_l975_97521


namespace NUMINAMATH_GPT_weight_jordan_after_exercise_l975_97558

def initial_weight : ℕ := 250
def first_4_weeks_loss : ℕ := 3 * 4
def next_8_weeks_loss : ℕ := 2 * 8
def total_weight_loss : ℕ := first_4_weeks_loss + next_8_weeks_loss
def final_weight : ℕ := initial_weight - total_weight_loss

theorem weight_jordan_after_exercise : final_weight = 222 :=
by 
  sorry

end NUMINAMATH_GPT_weight_jordan_after_exercise_l975_97558


namespace NUMINAMATH_GPT_filling_time_calculation_l975_97504

namespace TankerFilling

-- Define the filling rates
def fill_rate_A : ℚ := 1 / 60
def fill_rate_B : ℚ := 1 / 40
def combined_fill_rate : ℚ := fill_rate_A + fill_rate_B

-- Define the time variable
variable (T : ℚ)

-- State the theorem to be proved
theorem filling_time_calculation
  (h_fill_rate_A : fill_rate_A = 1 / 60)
  (h_fill_rate_B : fill_rate_B = 1 / 40)
  (h_combined_fill_rate : combined_fill_rate = 1 / 24) :
  (fill_rate_B * (T / 2) + combined_fill_rate * (T / 2)) = 1 → T = 30 :=
by
  intros h
  -- Proof will go here
  sorry

end TankerFilling

end NUMINAMATH_GPT_filling_time_calculation_l975_97504


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l975_97581

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.4 * Real.log 0.3)

theorem relationship_among_a_b_c : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l975_97581


namespace NUMINAMATH_GPT_range_of_x_l975_97587

variables {x : Real}

def P (x : Real) : Prop := (x + 1) / (x - 3) ≥ 0
def Q (x : Real) : Prop := abs (1 - x/2) < 1

theorem range_of_x (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end NUMINAMATH_GPT_range_of_x_l975_97587


namespace NUMINAMATH_GPT_initial_contribution_amount_l975_97539

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_contribution_amount_l975_97539


namespace NUMINAMATH_GPT_tax_amount_is_correct_l975_97544

def camera_cost : ℝ := 200.00
def tax_rate : ℝ := 0.15

theorem tax_amount_is_correct :
  (camera_cost * tax_rate) = 30.00 :=
sorry

end NUMINAMATH_GPT_tax_amount_is_correct_l975_97544
