import Mathlib

namespace NUMINAMATH_GPT_bicycle_speed_l1728_172822

theorem bicycle_speed
  (dist : ℝ := 15) -- Distance between the school and the museum
  (bus_factor : ℝ := 1.5) -- Bus speed is 1.5 times the bicycle speed
  (time_diff : ℝ := 1 / 4) -- Bicycle students leave 1/4 hour earlier
  (x : ℝ) -- Speed of bicycles
  (h : (dist / x) - (dist / (bus_factor * x)) = time_diff) :
  x = 20 :=
sorry

end NUMINAMATH_GPT_bicycle_speed_l1728_172822


namespace NUMINAMATH_GPT_total_marbles_correct_l1728_172819

variable (r : ℝ) -- number of red marbles
variable (b : ℝ) -- number of blue marbles
variable (g : ℝ) -- number of green marbles

-- Conditions
def red_blue_ratio : Prop := r = 1.5 * b
def green_red_ratio : Prop := g = 1.8 * r

-- Total number of marbles
def total_marbles (r b g : ℝ) : ℝ := r + b + g

theorem total_marbles_correct (r b g : ℝ) (h1 : red_blue_ratio r b) (h2 : green_red_ratio r g) : 
  total_marbles r b g = 3.467 * r :=
by 
  sorry

end NUMINAMATH_GPT_total_marbles_correct_l1728_172819


namespace NUMINAMATH_GPT_inequality_abc_l1728_172843

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 3 / (1 + a * b * c) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_abc_l1728_172843


namespace NUMINAMATH_GPT_simplify_expression_l1728_172854

def operation (a b : ℚ) : ℚ := 2 * a - b

theorem simplify_expression (x y : ℚ) : 
  operation (operation (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1728_172854


namespace NUMINAMATH_GPT_vertical_throw_time_l1728_172858

theorem vertical_throw_time (h v g t : ℝ)
  (h_def: h = v * t - (1/2) * g * t^2)
  (initial_v: v = 25)
  (gravity: g = 10)
  (target_h: h = 20) :
  t = 1 ∨ t = 4 := 
by
  sorry

end NUMINAMATH_GPT_vertical_throw_time_l1728_172858


namespace NUMINAMATH_GPT_factor_polynomial_l1728_172826

theorem factor_polynomial (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) := by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1728_172826


namespace NUMINAMATH_GPT_michael_matchstick_houses_l1728_172825

theorem michael_matchstick_houses :
  ∃ n : ℕ, n = (600 / 2) / 10 ∧ n = 30 := 
sorry

end NUMINAMATH_GPT_michael_matchstick_houses_l1728_172825


namespace NUMINAMATH_GPT_no_integer_k_sq_plus_k_plus_one_divisible_by_101_l1728_172892

theorem no_integer_k_sq_plus_k_plus_one_divisible_by_101 (k : ℤ) : 
  (k^2 + k + 1) % 101 ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_no_integer_k_sq_plus_k_plus_one_divisible_by_101_l1728_172892


namespace NUMINAMATH_GPT_parabola_tangent_sum_l1728_172884

theorem parabola_tangent_sum (m n : ℕ) (hmn_coprime : Nat.gcd m n = 1)
    (h_tangent : ∃ (k : ℝ), ∀ (x y : ℝ), y = 4 * x^2 ↔ x = y^2 + (m / n)) :
    m + n = 19 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_sum_l1728_172884


namespace NUMINAMATH_GPT_Jerry_has_36_stickers_l1728_172877

variable (FredStickers GeorgeStickers JerryStickers CarlaStickers : ℕ)
variable (h1 : FredStickers = 18)
variable (h2 : GeorgeStickers = FredStickers - 6)
variable (h3 : JerryStickers = 3 * GeorgeStickers)
variable (h4 : CarlaStickers = JerryStickers + JerryStickers / 4)
variable (h5 : GeorgeStickers + FredStickers = CarlaStickers ^ 2)

theorem Jerry_has_36_stickers : JerryStickers = 36 := by
  sorry

end NUMINAMATH_GPT_Jerry_has_36_stickers_l1728_172877


namespace NUMINAMATH_GPT_average_charge_proof_l1728_172814

noncomputable def averageChargePerPerson
  (chargeFirstDay : ℝ)
  (chargeSecondDay : ℝ)
  (chargeThirdDay : ℝ)
  (chargeFourthDay : ℝ)
  (ratioFirstDay : ℝ)
  (ratioSecondDay : ℝ)
  (ratioThirdDay : ℝ)
  (ratioFourthDay : ℝ)
  : ℝ :=
  let totalRevenue := ratioFirstDay * chargeFirstDay + ratioSecondDay * chargeSecondDay + ratioThirdDay * chargeThirdDay + ratioFourthDay * chargeFourthDay
  let totalVisitors := ratioFirstDay + ratioSecondDay + ratioThirdDay + ratioFourthDay
  totalRevenue / totalVisitors

theorem average_charge_proof :
  averageChargePerPerson 25 15 7.5 2.5 3 7 11 19 = 7.75 := by
  simp [averageChargePerPerson]
  sorry

end NUMINAMATH_GPT_average_charge_proof_l1728_172814


namespace NUMINAMATH_GPT_fifth_scroll_age_l1728_172875

def scrolls_age (n : ℕ) : ℕ :=
  match n with
  | 0 => 4080
  | k+1 => (3 * scrolls_age k) / 2

theorem fifth_scroll_age : scrolls_age 4 = 20655 := sorry

end NUMINAMATH_GPT_fifth_scroll_age_l1728_172875


namespace NUMINAMATH_GPT_combined_weight_of_new_students_l1728_172891

theorem combined_weight_of_new_students 
  (avg_weight_orig : ℝ) (num_students_orig : ℝ) 
  (new_avg_weight : ℝ) (num_new_students : ℝ) 
  (total_weight_gain_orig : ℝ) (total_weight_loss_orig : ℝ)
  (total_weight_orig : ℝ := avg_weight_orig * num_students_orig) 
  (net_weight_change_orig : ℝ := total_weight_gain_orig - total_weight_loss_orig)
  (total_weight_after_change_orig : ℝ := total_weight_orig + net_weight_change_orig) 
  (total_students_after : ℝ := num_students_orig + num_new_students) 
  (total_weight_class_after : ℝ := new_avg_weight * total_students_after) : 
  total_weight_class_after - total_weight_after_change_orig = 586 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_new_students_l1728_172891


namespace NUMINAMATH_GPT_large_painting_area_l1728_172829

theorem large_painting_area :
  ∃ (large_painting : ℕ),
  (3 * (6 * 6) + 4 * (2 * 3) + large_painting = 282) → large_painting = 150 := by
  sorry

end NUMINAMATH_GPT_large_painting_area_l1728_172829


namespace NUMINAMATH_GPT_boys_count_l1728_172807

variable (B G : ℕ)

theorem boys_count (h1 : B + G = 466) (h2 : G = B + 212) : B = 127 := by
  sorry

end NUMINAMATH_GPT_boys_count_l1728_172807


namespace NUMINAMATH_GPT_problem1_problem2_l1728_172894

noncomputable def part1 (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4} ∩ {x | x ≤ 2 * a - 5}
noncomputable def part2 (a : ℝ) : Prop := ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 4) → (x ≤ 2 * a - 5)

theorem problem1 : part1 3 = {x | -2 ≤ x ∧ x ≤ 1} :=
by { sorry }

theorem problem2 : ∀ a : ℝ, (part2 a) ↔ (a ≥ 9/2) :=
by { sorry }

end NUMINAMATH_GPT_problem1_problem2_l1728_172894


namespace NUMINAMATH_GPT_proof_M_inter_N_eq_01_l1728_172899
open Set

theorem proof_M_inter_N_eq_01 :
  let M := {x : ℤ | x^2 = x}
  let N := {-1, 0, 1}
  M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_GPT_proof_M_inter_N_eq_01_l1728_172899


namespace NUMINAMATH_GPT_find_x_l1728_172805

theorem find_x (x : ℝ) (h : 3 * x = (20 - x) + 20) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_x_l1728_172805


namespace NUMINAMATH_GPT_ratio_of_juniors_to_freshmen_l1728_172886

variables (f j : ℕ) 

theorem ratio_of_juniors_to_freshmen (h1 : (1/4 : ℚ) * f = (1/2 : ℚ) * j) :
  j = f / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_juniors_to_freshmen_l1728_172886


namespace NUMINAMATH_GPT_factorize_9_minus_a_squared_l1728_172824

theorem factorize_9_minus_a_squared (a : ℤ) : 9 - a^2 = (3 + a) * (3 - a) :=
by
  sorry

end NUMINAMATH_GPT_factorize_9_minus_a_squared_l1728_172824


namespace NUMINAMATH_GPT_enrollment_difference_l1728_172850

theorem enrollment_difference :
  let Varsity := 1680
  let Northwest := 1170
  let Central := 1840
  let Greenbriar := 1090
  let Eastside := 1450
  Central - Greenbriar = 750 := 
by
  intros Varsity Northwest Central Greenbriar Eastside
  -- calculate the difference
  have h1 : 750 = 750 := rfl
  sorry

end NUMINAMATH_GPT_enrollment_difference_l1728_172850


namespace NUMINAMATH_GPT_state_A_selection_percentage_l1728_172881

theorem state_A_selection_percentage
  (candidates_A : ℕ)
  (candidates_B : ℕ)
  (x : ℕ)
  (selected_B_ratio : ℚ)
  (extra_B : ℕ)
  (h1 : candidates_A = 7900)
  (h2 : candidates_B = 7900)
  (h3 : selected_B_ratio = 0.07)
  (h4 : extra_B = 79)
  (h5 : 7900 * (7 / 100) + 79 = 7900 * (x / 100) + 79) :
  x = 7 := by
  sorry

end NUMINAMATH_GPT_state_A_selection_percentage_l1728_172881


namespace NUMINAMATH_GPT_parabola_focus_l1728_172817

theorem parabola_focus (F : ℝ × ℝ) :
  (∀ (x y : ℝ), y^2 = 4 * x → (x + 1)^2 + y^2 = ((x - F.1)^2 + (y - F.2)^2)) → 
  F = (1, 0) :=
sorry

end NUMINAMATH_GPT_parabola_focus_l1728_172817


namespace NUMINAMATH_GPT_smallest_solution_x4_50x2_576_eq_0_l1728_172831

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_solution_x4_50x2_576_eq_0_l1728_172831


namespace NUMINAMATH_GPT_real_value_of_b_l1728_172845

open Real

theorem real_value_of_b : ∃ x : ℝ, (x^2 - 2 * x + 1 = 0) ∧ (x^2 + x - 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_real_value_of_b_l1728_172845


namespace NUMINAMATH_GPT_triangle_sides_inequality_l1728_172838

-- Define the sides of a triangle and their sum
variables {a b c : ℝ}

-- Define the condition that they are sides of a triangle.
def triangle_sides (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition that their sum is 1
axiom sum_of_sides (a b c : ℝ) (h : triangle_sides a b c) : a + b + c = 1

-- Define the proof theorem for the inequality
theorem triangle_sides_inequality (h : triangle_sides a b c) (h_sum : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_sides_inequality_l1728_172838


namespace NUMINAMATH_GPT_g_ab_eq_zero_l1728_172893

def g (x : ℤ) : ℤ := x^2 - 2013 * x

theorem g_ab_eq_zero (a b : ℤ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_g_ab_eq_zero_l1728_172893


namespace NUMINAMATH_GPT_smallest_n_perfect_square_and_cube_l1728_172801

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end NUMINAMATH_GPT_smallest_n_perfect_square_and_cube_l1728_172801


namespace NUMINAMATH_GPT_farmer_turkeys_l1728_172848

variable (n c : ℝ)

theorem farmer_turkeys (h1 : n * c = 60) (h2 : (c + 0.10) * (n - 15) = 54) : n = 75 :=
sorry

end NUMINAMATH_GPT_farmer_turkeys_l1728_172848


namespace NUMINAMATH_GPT_total_pennies_l1728_172895

theorem total_pennies (R G K : ℕ) (h1 : R = 180) (h2 : G = R / 2) (h3 : K = G / 3) : R + G + K = 300 := by
  sorry

end NUMINAMATH_GPT_total_pennies_l1728_172895


namespace NUMINAMATH_GPT_maximum_marks_l1728_172827

theorem maximum_marks (M : ℝ)
  (pass_threshold_percentage : ℝ := 33)
  (marks_obtained : ℝ := 92)
  (marks_failed_by : ℝ := 40) :
  (marks_obtained + marks_failed_by) = (pass_threshold_percentage / 100) * M → M = 400 := by
  sorry

end NUMINAMATH_GPT_maximum_marks_l1728_172827


namespace NUMINAMATH_GPT_minimum_voters_needed_l1728_172810

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end NUMINAMATH_GPT_minimum_voters_needed_l1728_172810


namespace NUMINAMATH_GPT_car_grid_probability_l1728_172857

theorem car_grid_probability:
  let m := 11
  let n := 48
  100 * m + n = 1148 := by
  sorry

end NUMINAMATH_GPT_car_grid_probability_l1728_172857


namespace NUMINAMATH_GPT_rectangle_dimension_correct_l1728_172870

-- Definition of the Width and Length based on given conditions
def width := 3 / 2
def length := 3

-- Perimeter and Area conditions
def perimeter_condition (w l : ℝ) := 2 * (w + l) = 2 * (w * l)
def length_condition (w l : ℝ) := l = 2 * w

-- Main theorem statement
theorem rectangle_dimension_correct :
  ∃ (w l : ℝ), perimeter_condition w l ∧ length_condition w l ∧ w = width ∧ l = length :=
by {
  -- add sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_rectangle_dimension_correct_l1728_172870


namespace NUMINAMATH_GPT_lambda_range_l1728_172853

noncomputable def lambda (S1 S2 S3 S4: ℝ) (S: ℝ) : ℝ :=
  4 * (S1 + S2 + S3 + S4) / S

theorem lambda_range (S1 S2 S3 S4: ℝ) (S: ℝ) (h_max: S = max (max S1 S2) (max S3 S4)) :
  2 < lambda S1 S2 S3 S4 S ∧ lambda S1 S2 S3 S4 S ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_lambda_range_l1728_172853


namespace NUMINAMATH_GPT_find_t_given_conditions_l1728_172876

variables (p t j x y : ℝ)

theorem find_t_given_conditions
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p * (1 - t / 100))
  (h4 : x = 0.10 * t)
  (h5 : y = 0.50 * j)
  (h6 : x + y = 12) :
  t = 24 :=
by sorry

end NUMINAMATH_GPT_find_t_given_conditions_l1728_172876


namespace NUMINAMATH_GPT_find_n_l1728_172803

theorem find_n (a : ℝ) (x : ℝ) (y : ℝ) (h1 : 0 < a) (h2 : a * x + 0.6 * a * y = 5 / 10)
(h3 : 1.6 * a * x + 1.2 * a * y = 1 - 1 / 10) : 
∃ n : ℕ, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1728_172803


namespace NUMINAMATH_GPT_square_of_binomial_l1728_172898

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, (x^2 - 18 * x + k) = (x + b)^2) ↔ k = 81 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l1728_172898


namespace NUMINAMATH_GPT_general_solution_of_differential_eq_l1728_172836

theorem general_solution_of_differential_eq (x y : ℝ) (C : ℝ) :
  (x^2 - y^2) * (y * (1 - C^2)) - 2 * (y * x) * (x) = 0 → (x^2 + y^2 = C * y) := by
  sorry

end NUMINAMATH_GPT_general_solution_of_differential_eq_l1728_172836


namespace NUMINAMATH_GPT_solution_set_inequality_l1728_172842

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f e = 0) (h2 : ∀ x > 0, x * deriv f x < 2) :
    ∀ x, 0 < x → x ≤ e → f x + 2 ≥ 2 * log x :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1728_172842


namespace NUMINAMATH_GPT_candy_division_l1728_172864

theorem candy_division (total_candy num_students : ℕ) (h1 : total_candy = 344) (h2 : num_students = 43) : total_candy / num_students = 8 := by
  sorry

end NUMINAMATH_GPT_candy_division_l1728_172864


namespace NUMINAMATH_GPT_solution_set_unique_line_l1728_172868

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_unique_line_l1728_172868


namespace NUMINAMATH_GPT_proof_problem_l1728_172859

def p := 8 + 7 = 16
def q := Real.pi > 3

theorem proof_problem :
  (¬p ∧ q) ∧ ((p ∨ q) = true) ∧ ((p ∧ q) = false) ∧ ((¬p) = true) := sorry

end NUMINAMATH_GPT_proof_problem_l1728_172859


namespace NUMINAMATH_GPT_remaining_rice_l1728_172888

theorem remaining_rice {q_0 : ℕ} {c : ℕ} {d : ℕ} 
    (h_q0 : q_0 = 52) 
    (h_c : c = 9) 
    (h_d : d = 3) : 
    q_0 - (c * d) = 25 := 
  by 
    -- Proof to be written here
    sorry

end NUMINAMATH_GPT_remaining_rice_l1728_172888


namespace NUMINAMATH_GPT_total_watermelons_l1728_172846

def watermelons_grown_by_jason : ℕ := 37
def watermelons_grown_by_sandy : ℕ := 11

theorem total_watermelons : watermelons_grown_by_jason + watermelons_grown_by_sandy = 48 := by
  sorry

end NUMINAMATH_GPT_total_watermelons_l1728_172846


namespace NUMINAMATH_GPT_max_value_xy_xz_yz_l1728_172837

theorem max_value_xy_xz_yz (x y z : ℝ) (h : x + 2 * y + z = 6) :
  xy + xz + yz ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_value_xy_xz_yz_l1728_172837


namespace NUMINAMATH_GPT_no_two_primes_sum_to_10003_l1728_172800

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the specific numbers involved
def even_prime : ℕ := 2
def target_number : ℕ := 10003
def candidate : ℕ := target_number - even_prime

-- State the main proposition in question
theorem no_two_primes_sum_to_10003 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = target_number :=
sorry

end NUMINAMATH_GPT_no_two_primes_sum_to_10003_l1728_172800


namespace NUMINAMATH_GPT_cost_of_items_l1728_172865

theorem cost_of_items (e t b : ℝ) 
    (h1 : 3 * e + 4 * t = 3.20)
    (h2 : 4 * e + 3 * t = 3.50)
    (h3 : 5 * e + 5 * t + 2 * b = 5.70) :
    4 * e + 4 * t + 3 * b = 5.20 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_items_l1728_172865


namespace NUMINAMATH_GPT_evaluate_infinite_series_l1728_172833

noncomputable def infinite_series (n : ℕ) : ℝ := (n^2) / (3^n)

theorem evaluate_infinite_series :
  (∑' k : ℕ, infinite_series (k+1)) = 4.5 :=
by sorry

end NUMINAMATH_GPT_evaluate_infinite_series_l1728_172833


namespace NUMINAMATH_GPT_sandy_correct_sums_l1728_172835

theorem sandy_correct_sums :
  ∃ x y : ℕ, x + y = 30 ∧ 3 * x - 2 * y = 60 ∧ x = 24 :=
by
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l1728_172835


namespace NUMINAMATH_GPT_company_pays_per_month_l1728_172828

theorem company_pays_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1.08 * 10^6)
  (h5 : cost_per_box = 0.6) :
  (total_volume / (length * width * height) * cost_per_box) = 360 :=
by
  -- sorry to skip proof
  sorry

end NUMINAMATH_GPT_company_pays_per_month_l1728_172828


namespace NUMINAMATH_GPT_coefficient_a2_l1728_172873

theorem coefficient_a2 :
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
  (x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
  a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + 
  a_10 * (x + 1)^10) →
  a_2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_a2_l1728_172873


namespace NUMINAMATH_GPT_sqrt_factorial_sq_l1728_172855

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end NUMINAMATH_GPT_sqrt_factorial_sq_l1728_172855


namespace NUMINAMATH_GPT_range_of_3a_minus_b_l1728_172815

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 2 ≤ a + b ∧ a + b ≤ 5) (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
    -2 ≤ 3 * a - b ∧ 3 * a - b ≤ 7 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_3a_minus_b_l1728_172815


namespace NUMINAMATH_GPT_water_to_milk_ratio_l1728_172896

theorem water_to_milk_ratio 
  (V : ℝ) 
  (hV : V > 0) 
  (milk_volume1 : ℝ := (3 / 5) * V) 
  (water_volume1 : ℝ := (2 / 5) * V) 
  (milk_volume2 : ℝ := (4 / 5) * V) 
  (water_volume2 : ℝ := (1 / 5) * V)
  (total_milk_volume : ℝ := milk_volume1 + milk_volume2)
  (total_water_volume : ℝ := water_volume1 + water_volume2) :
  total_water_volume / total_milk_volume = (3 / 7) := 
  sorry

end NUMINAMATH_GPT_water_to_milk_ratio_l1728_172896


namespace NUMINAMATH_GPT_minimum_number_of_apples_l1728_172879

-- Define the problem conditions and the proof statement
theorem minimum_number_of_apples :
  ∃ p : Fin 6 → ℕ, (∀ i, p i > 0) ∧ (Function.Injective p) ∧ (Finset.univ.sum p * 4 = 100) ∧ (Finset.univ.sum p = 25 / 4) := 
sorry

end NUMINAMATH_GPT_minimum_number_of_apples_l1728_172879


namespace NUMINAMATH_GPT_candy_cooking_time_l1728_172890

def initial_temperature : ℝ := 60
def peak_temperature : ℝ := 240
def final_temperature : ℝ := 170
def heating_rate : ℝ := 5
def cooling_rate : ℝ := 7

theorem candy_cooking_time : ( (peak_temperature - initial_temperature) / heating_rate + (peak_temperature - final_temperature) / cooling_rate ) = 46 := by
  sorry

end NUMINAMATH_GPT_candy_cooking_time_l1728_172890


namespace NUMINAMATH_GPT_perp_bisector_eq_l1728_172887

noncomputable def C1 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.1 - 7 = 0 }
noncomputable def C2 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.2 - 27 = 0 }

theorem perp_bisector_eq :
  ∃ x y, ( (x, y) ∈ C1 ∧ (x, y) ∈ C2 ) -> ( x - y = 0 ) :=
by
  sorry

end NUMINAMATH_GPT_perp_bisector_eq_l1728_172887


namespace NUMINAMATH_GPT_birdseed_mixture_l1728_172867

theorem birdseed_mixture (x : ℝ) (h1 : 0.40 * x + 0.65 * (100 - x) = 50) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_birdseed_mixture_l1728_172867


namespace NUMINAMATH_GPT_haley_initial_cupcakes_l1728_172804

-- Define the conditions
def todd_eats : ℕ := 11
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 3

-- Initial cupcakes calculation
def initial_cupcakes := packages * cupcakes_per_package + todd_eats

-- The theorem to prove
theorem haley_initial_cupcakes : initial_cupcakes = 20 :=
by
  -- Mathematical proof would go here,
  -- but we leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_haley_initial_cupcakes_l1728_172804


namespace NUMINAMATH_GPT_factorize_polynomial_l1728_172840

theorem factorize_polynomial (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := 
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l1728_172840


namespace NUMINAMATH_GPT_exists_set_X_gcd_condition_l1728_172851

theorem exists_set_X_gcd_condition :
  ∃ (X : Finset ℕ), X.card = 2022 ∧
  (∀ (a b c : ℕ) (n : ℕ) (ha : a ∈ X) (hb : b ∈ X) (hc : c ∈ X) (hn_pos : 0 < n)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c),
  Nat.gcd (a^n + b^n) c = 1) :=
sorry

end NUMINAMATH_GPT_exists_set_X_gcd_condition_l1728_172851


namespace NUMINAMATH_GPT_sector_angle_l1728_172874

theorem sector_angle (r L : ℝ) (h1 : r = 1) (h2 : L = 4) : abs (L - 2 * r) = 2 :=
by 
  -- This is the statement of our proof problem
  -- and does not include the proof itself.
  sorry

end NUMINAMATH_GPT_sector_angle_l1728_172874


namespace NUMINAMATH_GPT_problem_l1728_172839

theorem problem (a : ℤ) (n : ℕ) : (a + 1) ^ (2 * n + 1) + a ^ (n + 2) ∣ a ^ 2 + a + 1 :=
sorry

end NUMINAMATH_GPT_problem_l1728_172839


namespace NUMINAMATH_GPT_inverse_graph_pass_point_l1728_172871

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

noncomputable def satisfies_inverse (f f_inv : ℝ → ℝ) : Prop :=
∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

theorem inverse_graph_pass_point
  (hf : satisfies_inverse f f_inv)
  (h_point : (1 : ℝ) - f 1 = 3) :
  f_inv (-2) + 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_inverse_graph_pass_point_l1728_172871


namespace NUMINAMATH_GPT_function_classification_l1728_172866

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  sorry

end NUMINAMATH_GPT_function_classification_l1728_172866


namespace NUMINAMATH_GPT_expected_value_boy_girl_adjacent_pairs_l1728_172856

/-- Considering 10 boys and 15 girls lined up in a row, we need to show that
    the expected number of adjacent positions where a boy and a girl stand next to each other is 12. -/
theorem expected_value_boy_girl_adjacent_pairs :
  let boys := 10
  let girls := 15
  let total_people := boys + girls
  let total_adjacent_pairs := total_people - 1
  let p_boy_then_girl := (boys / total_people) * (girls / (total_people - 1))
  let p_girl_then_boy := (girls / total_people) * (boys / (total_people - 1))
  let expected_T := total_adjacent_pairs * (p_boy_then_girl + p_girl_then_boy)
  expected_T = 12 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_boy_girl_adjacent_pairs_l1728_172856


namespace NUMINAMATH_GPT_delivery_payment_l1728_172806

-- Define the problem conditions and the expected outcome
theorem delivery_payment 
    (deliveries_Oula : ℕ) 
    (deliveries_Tona : ℕ) 
    (difference_in_pay : ℝ) 
    (P : ℝ) 
    (H1 : deliveries_Oula = 96) 
    (H2 : deliveries_Tona = 72) 
    (H3 : difference_in_pay = 2400) :
    96 * P - 72 * P = 2400 → P = 100 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_delivery_payment_l1728_172806


namespace NUMINAMATH_GPT_Dans_placed_scissors_l1728_172820

theorem Dans_placed_scissors (initial_scissors placed_scissors total_scissors : ℕ) 
  (h1 : initial_scissors = 39) 
  (h2 : total_scissors = initial_scissors + placed_scissors) 
  (h3 : total_scissors = 52) : placed_scissors = 13 := 
by 
  sorry

end NUMINAMATH_GPT_Dans_placed_scissors_l1728_172820


namespace NUMINAMATH_GPT_find_f_of_2_l1728_172852

theorem find_f_of_2 
  (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1/x) = x^2 + 1/x^2) : f 2 = 6 :=
sorry

end NUMINAMATH_GPT_find_f_of_2_l1728_172852


namespace NUMINAMATH_GPT_train_cross_time_l1728_172880

noncomputable def train_length : ℝ := 130
noncomputable def train_speed_kph : ℝ := 45
noncomputable def total_length : ℝ := 375

noncomputable def speed_mps := train_speed_kph * 1000 / 3600
noncomputable def distance := train_length + total_length

theorem train_cross_time : (distance / speed_mps) = 30 := by
  sorry

end NUMINAMATH_GPT_train_cross_time_l1728_172880


namespace NUMINAMATH_GPT_mod_inverse_35_36_l1728_172869

theorem mod_inverse_35_36 : ∃ a : ℤ, 0 ≤ a ∧ a < 36 ∧ (35 * a) % 36 = 1 :=
  ⟨35, by sorry⟩

end NUMINAMATH_GPT_mod_inverse_35_36_l1728_172869


namespace NUMINAMATH_GPT_imaginary_unit_multiplication_l1728_172889

-- Statement of the problem   
theorem imaginary_unit_multiplication (i : ℂ) (hi : i ^ 2 = -1) : i * (1 + i) = -1 + i :=
by sorry

end NUMINAMATH_GPT_imaginary_unit_multiplication_l1728_172889


namespace NUMINAMATH_GPT_prime_in_A_l1728_172849

def is_in_A (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 2 * b^2 ∧ b ≠ 0

theorem prime_in_A (p : ℕ) (hp : Nat.Prime p) (h : is_in_A (p^2)) : is_in_A p :=
by
  sorry

end NUMINAMATH_GPT_prime_in_A_l1728_172849


namespace NUMINAMATH_GPT_isosceles_right_triangle_angle_l1728_172897

-- Define the conditions given in the problem
def is_isosceles (a b c : ℝ) : Prop := 
(a = b ∨ b = c ∨ c = a)

def is_right_triangle (a b c : ℝ) : Prop := 
(a = 90 ∨ b = 90 ∨ c = 90)

def angles_sum_to_180 (a b c : ℝ) : Prop :=
a + b + c = 180

-- The Proof Problem
theorem isosceles_right_triangle_angle :
  ∀ (a b c x : ℝ), (is_isosceles a b c) → (is_right_triangle a b c) → (angles_sum_to_180 a b c) → (x = a ∨ x = b ∨ x = c) → x = 45 :=
by
  intros a b c x h_isosceles h_right h_sum h_x
  -- Proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_angle_l1728_172897


namespace NUMINAMATH_GPT_b_payment_l1728_172878

theorem b_payment (b_days : ℕ) (a_days : ℕ) (total_wages : ℕ) (b_payment : ℕ) :
  b_days = 10 →
  a_days = 15 →
  total_wages = 5000 →
  b_payment = 3000 :=
by
  intros h1 h2 h3
  -- conditions
  have hb := h1
  have ha := h2
  have ht := h3
  -- skipping proof
  sorry

end NUMINAMATH_GPT_b_payment_l1728_172878


namespace NUMINAMATH_GPT_tan_product_identity_l1728_172813

-- Lean statement for the mathematical problem
theorem tan_product_identity : 
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 := by
  sorry

end NUMINAMATH_GPT_tan_product_identity_l1728_172813


namespace NUMINAMATH_GPT_matrix_norm_min_l1728_172802

-- Definition of the matrix
def matrix_mul (a b c d : ℤ) : Option (ℤ × ℤ × ℤ × ℤ) :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 then
    some (a^2 + b * c, a * b + b * d, a * c + c * d, b * c + d^2)
  else
    none

-- Main theorem statement
theorem matrix_norm_min (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hc : c ≠ 0) (hd : d ≠ 0) :
  matrix_mul a b c d = some (8, 0, 0, 5) → 
  |a| + |b| + |c| + |d| = 9 :=
by
  sorry

end NUMINAMATH_GPT_matrix_norm_min_l1728_172802


namespace NUMINAMATH_GPT_sqrt_inequalities_l1728_172872

theorem sqrt_inequalities
  (a b c d e : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hc : 0 ≤ c ∧ c ≤ 1)
  (hd : 0 ≤ d ∧ d ≤ 1)
  (he : 0 ≤ e ∧ e ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_inequalities_l1728_172872


namespace NUMINAMATH_GPT_remainder_of_n_div_7_l1728_172847

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_of_n_div_7_l1728_172847


namespace NUMINAMATH_GPT_expected_value_of_expression_is_50_l1728_172883

def expected_value_single_digit : ℚ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 9

def expected_value_expression : ℚ :=
  (expected_value_single_digit + expected_value_single_digit + expected_value_single_digit +
   (expected_value_single_digit + expected_value_single_digit * expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit + expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit * expected_value_single_digit)) / 4

theorem expected_value_of_expression_is_50 :
  expected_value_expression = 50 := sorry

end NUMINAMATH_GPT_expected_value_of_expression_is_50_l1728_172883


namespace NUMINAMATH_GPT_polygon_sides_eq_seven_l1728_172862

-- Given conditions:
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360
def difference_in_angles (n : ℕ) : ℝ := sum_interior_angles n - sum_exterior_angles

-- Proof statement:
theorem polygon_sides_eq_seven (n : ℕ) (h : difference_in_angles n = 540) : n = 7 := sorry

end NUMINAMATH_GPT_polygon_sides_eq_seven_l1728_172862


namespace NUMINAMATH_GPT_cost_of_bread_l1728_172863

-- Definition of the conditions
def total_purchase_amount : ℕ := 205  -- in cents
def amount_given_to_cashier : ℕ := 700  -- in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def num_nickels_received : ℕ := 8

-- Statement of the problem
theorem cost_of_bread :
  (∃ (B C : ℕ), B + C = total_purchase_amount ∧
                  amount_given_to_cashier - total_purchase_amount = 
                  (quarter_value + dime_value + num_nickels_received * nickel_value + 420) ∧
                  B = 125) :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_cost_of_bread_l1728_172863


namespace NUMINAMATH_GPT_odd_square_sum_of_consecutive_l1728_172885

theorem odd_square_sum_of_consecutive (n : ℤ) (h_odd : n % 2 = 1) (h_gt : n > 1) : 
  ∃ (j : ℤ), n^2 = j + (j + 1) :=
by
  sorry

end NUMINAMATH_GPT_odd_square_sum_of_consecutive_l1728_172885


namespace NUMINAMATH_GPT_handshakes_count_l1728_172809

def women := 6
def teams := 3
def shakes_per_woman := 4
def total_handshakes := (6 * 4) / 2

theorem handshakes_count : total_handshakes = 12 := by
  -- We provide this theorem directly.
  rfl

end NUMINAMATH_GPT_handshakes_count_l1728_172809


namespace NUMINAMATH_GPT_triangle_shortest_side_l1728_172834

theorem triangle_shortest_side (x y z : ℝ) (h : x / y = 1 / 2) (h1 : x / z = 1 / 3) (hyp : x = 6) : z = 3 :=
sorry

end NUMINAMATH_GPT_triangle_shortest_side_l1728_172834


namespace NUMINAMATH_GPT_hcf_of_numbers_l1728_172811

theorem hcf_of_numbers (x y : ℕ) (hcf lcm : ℕ) 
    (h_sum : x + y = 45) 
    (h_lcm : lcm = 100)
    (h_reciprocal_sum : 1 / (x : ℝ) + 1 / (y : ℝ) = 0.3433333333333333) :
    hcf = 1 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_numbers_l1728_172811


namespace NUMINAMATH_GPT_product_of_two_primes_l1728_172821

theorem product_of_two_primes (p q z : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) 
    (h_p_range : 2 < p ∧ p < 6) 
    (h_q_range : 8 < q ∧ q < 24) 
    (h_z_def : z = p * q) 
    (h_z_range : 15 < z ∧ z < 36) : 
    z = 33 := 
by 
    sorry

end NUMINAMATH_GPT_product_of_two_primes_l1728_172821


namespace NUMINAMATH_GPT_exchange_silver_cards_l1728_172818

theorem exchange_silver_cards : 
  (∃ red gold silver : ℕ,
    (∀ (r g s : ℕ), ((2 * g = 5 * r) ∧ (g = r + s) ∧ (r = 3) ∧ (g = 3) → s = 7))) :=
by
  sorry

end NUMINAMATH_GPT_exchange_silver_cards_l1728_172818


namespace NUMINAMATH_GPT_hexagon_probability_same_length_l1728_172861

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end NUMINAMATH_GPT_hexagon_probability_same_length_l1728_172861


namespace NUMINAMATH_GPT_greatest_prime_factor_341_l1728_172841

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_341_l1728_172841


namespace NUMINAMATH_GPT_value_of_xy_l1728_172844

noncomputable def distinct_nonzero_reals (x y : ℝ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y

theorem value_of_xy (x y : ℝ) (h : distinct_nonzero_reals x y) (h_eq : x + 4 / x = y + 4 / y) :
  x * y = 4 :=
sorry

end NUMINAMATH_GPT_value_of_xy_l1728_172844


namespace NUMINAMATH_GPT_parabola_axis_symmetry_value_p_l1728_172860

theorem parabola_axis_symmetry_value_p (p : ℝ) (h_parabola : ∀ y x, y^2 = 2 * p * x) (h_axis_symmetry : ∀ (a: ℝ), a = -1 → a = -p / 2) : p = 2 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_axis_symmetry_value_p_l1728_172860


namespace NUMINAMATH_GPT_initial_bees_l1728_172823

theorem initial_bees (B : ℕ) (h : B + 7 = 23) : B = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_bees_l1728_172823


namespace NUMINAMATH_GPT_hexagon_triangle_count_l1728_172816

-- Definitions based on problem conditions
def numPoints : ℕ := 7
def totalTriangles := Nat.choose numPoints 3
def collinearCases : ℕ := 3

-- Proof problem
theorem hexagon_triangle_count : totalTriangles - collinearCases = 32 :=
by
  -- Calculation is expected here
  sorry

end NUMINAMATH_GPT_hexagon_triangle_count_l1728_172816


namespace NUMINAMATH_GPT_bookseller_loss_l1728_172812

theorem bookseller_loss (C S : ℝ) (h : 20 * C = 25 * S) : (C - S) / C * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_bookseller_loss_l1728_172812


namespace NUMINAMATH_GPT_first_digit_base12_1025_l1728_172830

theorem first_digit_base12_1025 : (1025 : ℕ) / (12^2 : ℕ) = 7 := by
  sorry

end NUMINAMATH_GPT_first_digit_base12_1025_l1728_172830


namespace NUMINAMATH_GPT_geometric_sequence_nec_suff_l1728_172882

theorem geometric_sequence_nec_suff (a b c : ℝ) : (b^2 = a * c) ↔ (∃ r : ℝ, b = a * r ∧ c = b * r) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_nec_suff_l1728_172882


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1728_172832

variables (A B : Prop)

theorem sufficient_not_necessary (h : B → A) : ¬(A → B) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1728_172832


namespace NUMINAMATH_GPT_determine_m_l1728_172808

open Set Real

theorem determine_m (m : ℝ) : (∀ x, x ∈ { x | x ≥ 3 } ∪ { x | x < m }) ∧ (∀ x, x ∉ { x | x ≥ 3 } ∩ { x | x < m }) → m = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_determine_m_l1728_172808
