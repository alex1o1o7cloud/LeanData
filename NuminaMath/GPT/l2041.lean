import Mathlib

namespace NUMINAMATH_GPT_no_valid_rook_placement_l2041_204193

theorem no_valid_rook_placement :
  ∀ (r b g : ℕ), r + b + g = 50 →
  (2 * r ≤ b) →
  (2 * b ≤ g) →
  (2 * g ≤ r) →
  False :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_no_valid_rook_placement_l2041_204193


namespace NUMINAMATH_GPT_find_angle_A_triangle_is_right_l2041_204102

theorem find_angle_A (A : ℝ) (h : 2 * Real.cos (Real.pi + A) + Real.sin (Real.pi / 2 + 2 * A) + 3 / 2 = 0) :
  A = Real.pi / 3 := 
sorry

theorem triangle_is_right (a b c : ℝ) (A : ℝ) (ha : c - b = (Real.sqrt 3) / 3 * a) (hA : A = Real.pi / 3) :
  c^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_find_angle_A_triangle_is_right_l2041_204102


namespace NUMINAMATH_GPT_total_number_of_numbers_l2041_204174

-- Definitions using the conditions from the problem
def sum_of_first_4_numbers : ℕ := 4 * 4
def sum_of_last_4_numbers : ℕ := 4 * 4
def average_of_all_numbers (n : ℕ) : ℕ := 3 * n
def fourth_number : ℕ := 11
def total_sum_of_numbers : ℕ := sum_of_first_4_numbers + sum_of_last_4_numbers - fourth_number

-- Theorem stating the problem
theorem total_number_of_numbers (n : ℕ) : total_sum_of_numbers = average_of_all_numbers n → n = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_number_of_numbers_l2041_204174


namespace NUMINAMATH_GPT_Keith_picked_zero_apples_l2041_204160

variable (M J T K_A : ℕ)

theorem Keith_picked_zero_apples (hM : M = 14) (hJ : J = 41) (hT : T = 55) (hTotalOranges : M + J = T) : K_A = 0 :=
by
  sorry

end NUMINAMATH_GPT_Keith_picked_zero_apples_l2041_204160


namespace NUMINAMATH_GPT_train_travel_time_change_l2041_204196

theorem train_travel_time_change 
  (t1 t2 : ℕ) (s1 s2 d : ℕ) 
  (h1 : t1 = 4) 
  (h2 : s1 = 50) 
  (h3 : s2 = 100) 
  (h4 : d = t1 * s1) :
  t2 = d / s2 → t2 = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_travel_time_change_l2041_204196


namespace NUMINAMATH_GPT_liam_comic_books_l2041_204130

theorem liam_comic_books (cost_per_book : ℚ) (total_money : ℚ) (n : ℚ) : cost_per_book = 1.25 ∧ total_money = 10 → n = 8 :=
by
  intros h
  cases h
  have h1 : 1.25 * n ≤ 10 := by sorry
  have h2 : n ≤ 10 / 1.25 := by sorry
  have h3 : n ≤ 8 := by sorry
  have h4 : n = 8 := by sorry
  exact h4

end NUMINAMATH_GPT_liam_comic_books_l2041_204130


namespace NUMINAMATH_GPT_original_price_of_dish_l2041_204157

-- Define the variables and conditions explicitly
variables (P : ℝ)

-- John's payment after discount and tip over original price
def john_payment : ℝ := 0.9 * P + 0.15 * P

-- Jane's payment after discount and tip over discounted price
def jane_payment : ℝ := 0.9 * P + 0.135 * P

-- Given condition that John's payment is $0.63 more than Jane's
def payment_difference : Prop := john_payment P - jane_payment P = 0.63

theorem original_price_of_dish (h : payment_difference P) : P = 42 :=
by sorry

end NUMINAMATH_GPT_original_price_of_dish_l2041_204157


namespace NUMINAMATH_GPT_johns_new_weekly_earnings_l2041_204126

-- Define the original weekly earnings and the percentage increase as given conditions:
def original_weekly_earnings : ℕ := 60
def percentage_increase : ℕ := 50

-- Prove that John's new weekly earnings after the raise is 90 dollars:
theorem johns_new_weekly_earnings : original_weekly_earnings + (percentage_increase * original_weekly_earnings / 100) = 90 := by
sorry

end NUMINAMATH_GPT_johns_new_weekly_earnings_l2041_204126


namespace NUMINAMATH_GPT_function_increasing_on_interval_l2041_204124

theorem function_increasing_on_interval :
  ∀ x : ℝ, (1 / 2 < x) → (x > 0) → (8 * x - 1 / (x^2)) > 0 :=
sorry

end NUMINAMATH_GPT_function_increasing_on_interval_l2041_204124


namespace NUMINAMATH_GPT_students_in_class_l2041_204198

theorem students_in_class (n : ℕ) (S : ℕ) (h_avg_students : S / n = 14) (h_avg_including_teacher : (S + 45) / (n + 1) = 15) : n = 30 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l2041_204198


namespace NUMINAMATH_GPT_border_collie_catches_ball_in_32_seconds_l2041_204168

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end NUMINAMATH_GPT_border_collie_catches_ball_in_32_seconds_l2041_204168


namespace NUMINAMATH_GPT_correct_answer_l2041_204159

theorem correct_answer (x : ℝ) (h1 : 2 * x = 60) : x / 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l2041_204159


namespace NUMINAMATH_GPT_percentage_decrease_in_sale_l2041_204156

theorem percentage_decrease_in_sale (P Q : ℝ) (D : ℝ)
  (h1 : 1.80 * P * Q * (1 - D / 100) = 1.44 * P * Q) : 
  D = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_sale_l2041_204156


namespace NUMINAMATH_GPT_pradeep_pass_percentage_l2041_204133

variable (marks_obtained : ℕ) (marks_short : ℕ) (max_marks : ℝ)

theorem pradeep_pass_percentage (h1 : marks_obtained = 150) (h2 : marks_short = 25) (h3 : max_marks = 500.00000000000006) :
  ((marks_obtained + marks_short) / max_marks) * 100 = 35 := 
by
  sorry

end NUMINAMATH_GPT_pradeep_pass_percentage_l2041_204133


namespace NUMINAMATH_GPT_total_questions_attempted_l2041_204145

theorem total_questions_attempted 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (total_questions : ℕ) (incorrect_answers : ℕ)
  (h_marks_per_correct : marks_per_correct = 4)
  (h_marks_lost_per_wrong : marks_lost_per_wrong = 1) 
  (h_total_marks : total_marks = 130) 
  (h_correct_answers : correct_answers = 36) 
  (h_score_eq : marks_per_correct * correct_answers - marks_lost_per_wrong * incorrect_answers = total_marks)
  (h_total_questions : total_questions = correct_answers + incorrect_answers) : 
  total_questions = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_questions_attempted_l2041_204145


namespace NUMINAMATH_GPT_cos_270_eq_zero_l2041_204127

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end NUMINAMATH_GPT_cos_270_eq_zero_l2041_204127


namespace NUMINAMATH_GPT_value_of_a_l2041_204143

theorem value_of_a {a : ℝ} 
  (h : ∀ x y : ℝ, ax - 2*y + 2 = 0 ↔ x + (a-3)*y + 1 = 0) : 
  a = 1 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l2041_204143


namespace NUMINAMATH_GPT_no_five_consecutive_integers_with_fourth_powers_sum_l2041_204172

theorem no_five_consecutive_integers_with_fourth_powers_sum:
  ∀ n : ℤ, n^4 + (n + 1)^4 + (n + 2)^4 + (n + 3)^4 ≠ (n + 4)^4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_no_five_consecutive_integers_with_fourth_powers_sum_l2041_204172


namespace NUMINAMATH_GPT_solve_for_x_l2041_204166

theorem solve_for_x (x : ℝ) :
  5 * (x - 9) = 7 * (3 - 3 * x) + 10 → x = 38 / 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2041_204166


namespace NUMINAMATH_GPT_find_first_prime_l2041_204109

theorem find_first_prime (p1 p2 z : ℕ) 
  (prime_p1 : Nat.Prime p1)
  (prime_p2 : Nat.Prime p2)
  (z_eq : z = p1 * p2)
  (z_val : z = 33)
  (p2_range : 8 < p2 ∧ p2 < 24)
  : p1 = 3 := 
sorry

end NUMINAMATH_GPT_find_first_prime_l2041_204109


namespace NUMINAMATH_GPT_mark_saves_5_dollars_l2041_204182

def cost_per_pair : ℤ := 50

def promotionA_total_cost (cost : ℤ) : ℤ :=
  cost + (cost / 2)

def promotionB_total_cost (cost : ℤ) : ℤ :=
  cost + (cost - 20)

def savings (totalB totalA : ℤ) : ℤ :=
  totalB - totalA

theorem mark_saves_5_dollars :
  savings (promotionB_total_cost cost_per_pair) (promotionA_total_cost cost_per_pair) = 5 := by
  sorry

end NUMINAMATH_GPT_mark_saves_5_dollars_l2041_204182


namespace NUMINAMATH_GPT_administrative_staff_drawn_in_stratified_sampling_l2041_204141

theorem administrative_staff_drawn_in_stratified_sampling
  (total_staff : ℕ)
  (full_time_teachers : ℕ)
  (administrative_staff : ℕ)
  (logistics_personnel : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 320)
  (h_teachers : full_time_teachers = 248)
  (h_admin : administrative_staff = 48)
  (h_logistics : logistics_personnel = 24)
  (h_sample : sample_size = 40)
  : (administrative_staff * (sample_size / total_staff) = 6) :=
by
  -- mathematical proof goes here
  sorry

end NUMINAMATH_GPT_administrative_staff_drawn_in_stratified_sampling_l2041_204141


namespace NUMINAMATH_GPT_Vasek_solved_18_problems_l2041_204140

variables (m v z : ℕ)

theorem Vasek_solved_18_problems (h1 : m + v = 25) (h2 : z + v = 32) (h3 : z = 2 * m) : v = 18 := by 
  sorry

end NUMINAMATH_GPT_Vasek_solved_18_problems_l2041_204140


namespace NUMINAMATH_GPT_steve_reading_pages_l2041_204132

theorem steve_reading_pages (total_pages: ℕ) (weeks: ℕ) (reading_days_per_week: ℕ) 
  (reads_on_monday: ℕ) (reads_on_wednesday: ℕ) (reads_on_friday: ℕ) :
  total_pages = 2100 → weeks = 7 → reading_days_per_week = 3 → 
  (reads_on_monday = reads_on_wednesday ∧ reads_on_wednesday = reads_on_friday) → 
  ((weeks * reading_days_per_week) > 0) → 
  (total_pages / (weeks * reading_days_per_week)) = reads_on_monday :=
by
  intro h_total_pages h_weeks h_reading_days_per_week h_reads_on_days h_nonzero
  sorry

end NUMINAMATH_GPT_steve_reading_pages_l2041_204132


namespace NUMINAMATH_GPT_combination_10_3_eq_120_l2041_204112

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_combination_10_3_eq_120_l2041_204112


namespace NUMINAMATH_GPT_pencils_in_stock_at_end_of_week_l2041_204176

def pencils_per_day : ℕ := 100
def days_per_week : ℕ := 5
def initial_pencils : ℕ := 80
def sold_pencils : ℕ := 350

theorem pencils_in_stock_at_end_of_week :
  (pencils_per_day * days_per_week + initial_pencils - sold_pencils) = 230 :=
by sorry  -- Proof will be filled in later

end NUMINAMATH_GPT_pencils_in_stock_at_end_of_week_l2041_204176


namespace NUMINAMATH_GPT_initial_average_runs_l2041_204170

theorem initial_average_runs (A : ℕ) (h : 10 * A + 87 = 11 * (A + 5)) : A = 32 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_runs_l2041_204170


namespace NUMINAMATH_GPT_fish_count_l2041_204122

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end NUMINAMATH_GPT_fish_count_l2041_204122


namespace NUMINAMATH_GPT_find_m_and_n_l2041_204125

namespace BinomialProof

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n m : ℕ) : Prop :=
  binom (n+1) (m+1) = binom (n+1) m

def condition2 (n m : ℕ) : Prop :=
  binom (n+1) m / binom (n+1) (m-1) = 5 / 3

-- Problem statement
theorem find_m_and_n : ∃ (m n : ℕ), 
  (condition1 n m) ∧ 
  (condition2 n m) ∧ 
  m = 3 ∧ n = 6 := sorry

end BinomialProof

end NUMINAMATH_GPT_find_m_and_n_l2041_204125


namespace NUMINAMATH_GPT_solve_for_wood_length_l2041_204138

theorem solve_for_wood_length (y x : ℝ) (h1 : y - x = 4.5) (h2 : x - (1/2) * y = 1) :
  ∃! (x y : ℝ), (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  -- The content of the proof is omitted
  sorry

end NUMINAMATH_GPT_solve_for_wood_length_l2041_204138


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2041_204128

noncomputable def average_speed_excluding_stoppages
  (speed_including_stoppages : ℝ)
  (stoppage_time_ratio : ℝ) : ℝ :=
  (speed_including_stoppages * 1) / (1 - stoppage_time_ratio)

theorem bus_speed_excluding_stoppages :
  average_speed_excluding_stoppages 15 (3/4) = 60 := 
by
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2041_204128


namespace NUMINAMATH_GPT_sum_of_two_numbers_l2041_204139

theorem sum_of_two_numbers (x y : ℤ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l2041_204139


namespace NUMINAMATH_GPT_exists_x0_l2041_204188

theorem exists_x0 : ∃ x0 : ℝ, x0^2 + 2*x0 + 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_exists_x0_l2041_204188


namespace NUMINAMATH_GPT_intersection_is_3_l2041_204177

open Set -- Open the Set namespace to use set notation

theorem intersection_is_3 {A B : Set ℤ} (hA : A = {1, 3}) (hB : B = {-1, 2, 3}) :
  A ∩ B = {3} :=
by {
-- Proof goes here
  sorry
}

end NUMINAMATH_GPT_intersection_is_3_l2041_204177


namespace NUMINAMATH_GPT_problem_proof_l2041_204151

theorem problem_proof (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
  (h2 : a + b + c + d = m^2) 
  (h3 : max (max a b) (max c d) = n^2) : 
  m = 9 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l2041_204151


namespace NUMINAMATH_GPT_sequence_inequality_l2041_204161

theorem sequence_inequality
  (n : ℕ) (h1 : 1 < n)
  (a : ℕ → ℕ)
  (h2 : ∀ i, i < n → a i < a (i + 1))
  (h3 : ∀ i, i < n - 1 → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) :
  a (n - 1) ≥ 2 * n ^ 2 - 1 :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l2041_204161


namespace NUMINAMATH_GPT_numerical_expression_as_sum_of_squares_l2041_204186

theorem numerical_expression_as_sum_of_squares : 
  2 * (2009:ℕ)^2 + 2 * (2010:ℕ)^2 = (4019:ℕ)^2 + (1:ℕ)^2 := 
by
  sorry

end NUMINAMATH_GPT_numerical_expression_as_sum_of_squares_l2041_204186


namespace NUMINAMATH_GPT_two_pow_n_plus_one_divisible_by_three_l2041_204104

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end NUMINAMATH_GPT_two_pow_n_plus_one_divisible_by_three_l2041_204104


namespace NUMINAMATH_GPT_students_voted_both_issues_l2041_204142

-- Define the total number of students.
def total_students : ℕ := 150

-- Define the number of students who voted in favor of the first issue.
def voted_first_issue : ℕ := 110

-- Define the number of students who voted in favor of the second issue.
def voted_second_issue : ℕ := 95

-- Define the number of students who voted against both issues.
def voted_against_both : ℕ := 15

-- Theorem: Number of students who voted in favor of both issues is 70.
theorem students_voted_both_issues : 
  ((voted_first_issue + voted_second_issue) - (total_students - voted_against_both)) = 70 :=
by
  sorry

end NUMINAMATH_GPT_students_voted_both_issues_l2041_204142


namespace NUMINAMATH_GPT_total_candies_darrel_took_l2041_204192

theorem total_candies_darrel_took (r b x : ℕ) (h1 : r = 3 * b)
  (h2 : r - x = 4 * (b - x))
  (h3 : r - x - 12 = 5 * (b - x - 12)) : 2 * x = 48 := sorry

end NUMINAMATH_GPT_total_candies_darrel_took_l2041_204192


namespace NUMINAMATH_GPT_cookies_per_bag_l2041_204113

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (H1 : total_cookies = 703) (H2 : num_bags = 37) : total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_GPT_cookies_per_bag_l2041_204113


namespace NUMINAMATH_GPT_distinct_real_roots_iff_l2041_204149

noncomputable def operation (a b : ℝ) : ℝ := a * b^2 - b 

theorem distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation 1 x1 = k ∧ operation 1 x2 = k) ↔ k > -1/4 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_iff_l2041_204149


namespace NUMINAMATH_GPT_rhombus_diagonal_sum_l2041_204153

theorem rhombus_diagonal_sum
  (d1 d2 : ℝ)
  (h1 : d1 ≤ 6)
  (h2 : 6 ≤ d2)
  (side_len : ℝ)
  (h_side : side_len = 5)
  (rhombus_relation : d1^2 + d2^2 = 4 * side_len^2) :
  d1 + d2 ≤ 14 :=
sorry

end NUMINAMATH_GPT_rhombus_diagonal_sum_l2041_204153


namespace NUMINAMATH_GPT_mira_jogging_distance_l2041_204136

def jogging_speed : ℝ := 5 -- speed in miles per hour
def jogging_hours_per_day : ℝ := 2 -- hours per day
def days_count : ℕ := 5 -- number of days

theorem mira_jogging_distance :
  (jogging_speed * jogging_hours_per_day * days_count : ℝ) = 50 :=
by
  sorry

end NUMINAMATH_GPT_mira_jogging_distance_l2041_204136


namespace NUMINAMATH_GPT_set_intersection_eq_l2041_204135

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

def B : Set ℝ := { x | x < -2 ∨ x > 5 }

def C_U (B : Set ℝ) : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }

theorem set_intersection_eq : A ∩ (C_U B) = { x | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

end NUMINAMATH_GPT_set_intersection_eq_l2041_204135


namespace NUMINAMATH_GPT_gcd_490_910_l2041_204195

theorem gcd_490_910 : Nat.gcd 490 910 = 70 :=
by
  sorry

end NUMINAMATH_GPT_gcd_490_910_l2041_204195


namespace NUMINAMATH_GPT_total_sold_l2041_204134

theorem total_sold (D C : ℝ) (h1 : D = 1.6 * C) (h2 : D = 168) : D + C = 273 :=
by
  sorry

end NUMINAMATH_GPT_total_sold_l2041_204134


namespace NUMINAMATH_GPT_expected_red_hair_americans_l2041_204184

theorem expected_red_hair_americans (prob_red_hair : ℝ) (sample_size : ℕ) :
  prob_red_hair = 1 / 6 → sample_size = 300 → (prob_red_hair * sample_size = 50) := by
  intros
  sorry

end NUMINAMATH_GPT_expected_red_hair_americans_l2041_204184


namespace NUMINAMATH_GPT_fraction_of_students_with_buddy_l2041_204137

variables (f e : ℕ)
-- Given:
axiom H1 : e / 4 = f / 3

-- Prove:
theorem fraction_of_students_with_buddy : 
  (e / 4 + f / 3) / (e + f) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_buddy_l2041_204137


namespace NUMINAMATH_GPT_cell_population_l2041_204197

variable (n : ℕ)

def a (n : ℕ) : ℕ :=
  if n = 1 then 5
  else 1 -- Placeholder for general definition

theorem cell_population (n : ℕ) : a n = 2^(n-1) + 4 := by
  sorry

end NUMINAMATH_GPT_cell_population_l2041_204197


namespace NUMINAMATH_GPT_largest_possible_s_l2041_204173

theorem largest_possible_s (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (h_angle : (101 : ℚ) / 97 * ((s - 2) * 180 / s : ℚ) = ((r - 2) * 180 / r : ℚ)) :
  s = 100 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_s_l2041_204173


namespace NUMINAMATH_GPT_chapatis_order_count_l2041_204150

theorem chapatis_order_count (chapati_cost rice_cost veg_cost total_paid chapati_count : ℕ) 
  (rice_plates veg_plates : ℕ)
  (H1 : chapati_cost = 6)
  (H2 : rice_cost = 45)
  (H3 : veg_cost = 70)
  (H4 : total_paid = 1111)
  (H5 : rice_plates = 5)
  (H6 : veg_plates = 7)
  (H7 : chapati_count = (total_paid - (rice_plates * rice_cost + veg_plates * veg_cost)) / chapati_cost) :
  chapati_count = 66 :=
by
  sorry

end NUMINAMATH_GPT_chapatis_order_count_l2041_204150


namespace NUMINAMATH_GPT_find_g_9_l2041_204181

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end NUMINAMATH_GPT_find_g_9_l2041_204181


namespace NUMINAMATH_GPT_three_digit_number_is_382_l2041_204106

theorem three_digit_number_is_382 
  (x : ℕ) 
  (h1 : x >= 100 ∧ x < 1000) 
  (h2 : 7000 + x - (10 * x + 7) = 3555) : 
  x = 382 :=
by 
  sorry

end NUMINAMATH_GPT_three_digit_number_is_382_l2041_204106


namespace NUMINAMATH_GPT_final_speed_of_ball_l2041_204115

/--
 A small rubber ball moves horizontally between two vertical walls. One wall is fixed, and the other wall moves away from it at a constant speed u.
 The ball's collisions are perfectly elastic. The initial speed of the ball is v₀. Prove that after 10 collisions with the moving wall, the ball's speed is 17 cm/s.
-/
theorem final_speed_of_ball
    (u : ℝ) (v₀ : ℝ) (n : ℕ)
    (u_val : u = 100) (v₀_val : v₀ = 2017) (n_val : n = 10) :
    v₀ - 2 * u * n = 17 := 
    by
    rw [u_val, v₀_val, n_val]
    sorry

end NUMINAMATH_GPT_final_speed_of_ball_l2041_204115


namespace NUMINAMATH_GPT_no_four_distinct_real_roots_l2041_204146

theorem no_four_distinct_real_roots (a b : ℝ) :
  ¬ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_no_four_distinct_real_roots_l2041_204146


namespace NUMINAMATH_GPT_pentagon_area_l2041_204144

theorem pentagon_area 
  (edge_length : ℝ) 
  (triangle_height : ℝ) 
  (n_pentagons : ℕ) 
  (equal_convex_pentagons : ℕ) 
  (pentagon_area : ℝ) : 
  edge_length = 5 ∧ triangle_height = 2 ∧ n_pentagons = 5 ∧ equal_convex_pentagons = 5 → pentagon_area = 30 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_area_l2041_204144


namespace NUMINAMATH_GPT_kittens_percentage_rounded_l2041_204163

theorem kittens_percentage_rounded (total_cats female_ratio kittens_per_female cats_sold : ℕ) (h1 : total_cats = 6)
  (h2 : female_ratio = 2)
  (h3 : kittens_per_female = 7)
  (h4 : cats_sold = 9) : 
  ((12 : ℤ) * 100 / (18 : ℤ)).toNat = 67 := by
  -- Historical reference and problem specific values involved 
  sorry

end NUMINAMATH_GPT_kittens_percentage_rounded_l2041_204163


namespace NUMINAMATH_GPT_cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l2041_204180

-- Definitions for angles A, B, and C forming an arithmetic sequence and their sum being 180 degrees
variables {A B C : ℝ}

-- Definitions for side lengths a, b, and c forming a geometric sequence
variables {a b c : ℝ}

-- Question 1: Prove that cos B = 1/2 under the given conditions
theorem cos_B_equals_half 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) : 
  Real.cos B = 1 / 2 :=
sorry

-- Question 2: Prove that sin A * sin C = 3/4 under the given conditions
theorem sin_A_mul_sin_C_equals_three_fourths 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) 
  (h3 : b^2 = a * c) : 
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

end NUMINAMATH_GPT_cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l2041_204180


namespace NUMINAMATH_GPT_correctness_of_solution_set_l2041_204111

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := { x | 3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9 }

-- Define the expected solution set derived from the problem
def expected_solution_set : Set ℝ := { x | -1 < x ∧ x ≤ 1 } ∪ { x | 2.5 < x ∧ x < 4.5 }

-- The proof statement
theorem correctness_of_solution_set : solution_set = expected_solution_set :=
  sorry

end NUMINAMATH_GPT_correctness_of_solution_set_l2041_204111


namespace NUMINAMATH_GPT_greatest_sum_l2041_204162

theorem greatest_sum (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_greatest_sum_l2041_204162


namespace NUMINAMATH_GPT_unique_handshakes_l2041_204155

-- Define the circular arrangement and handshakes conditions
def num_people := 30
def handshakes_per_person := 2

theorem unique_handshakes : 
  (num_people * handshakes_per_person) / 2 = 30 :=
by
  -- Sorry is used here as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_unique_handshakes_l2041_204155


namespace NUMINAMATH_GPT_product_of_integers_l2041_204164

theorem product_of_integers (x y : ℕ) (h1 : x + y = 72) (h2 : x - y = 18) : x * y = 1215 := 
sorry

end NUMINAMATH_GPT_product_of_integers_l2041_204164


namespace NUMINAMATH_GPT_soap_box_length_l2041_204147

def VolumeOfEachSoapBox (L : ℝ) := 30 * L
def VolumeOfCarton := 25 * 42 * 60
def MaximumSoapBoxes := 300

theorem soap_box_length :
  ∀ L : ℝ,
  MaximumSoapBoxes * VolumeOfEachSoapBox L = VolumeOfCarton → 
  L = 7 :=
by
  intros L h
  sorry

end NUMINAMATH_GPT_soap_box_length_l2041_204147


namespace NUMINAMATH_GPT_probability_not_win_l2041_204190

theorem probability_not_win (A B : Fin 16) : 
  (256 - 16) / 256 = 15 / 16 := 
by
  sorry

end NUMINAMATH_GPT_probability_not_win_l2041_204190


namespace NUMINAMATH_GPT_cost_of_each_taco_l2041_204129

variables (T E : ℝ)

-- Conditions
axiom condition1 : 2 * T + 3 * E = 7.80
axiom condition2 : 3 * T + 5 * E = 12.70

-- Question to prove
theorem cost_of_each_taco : T = 0.90 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_taco_l2041_204129


namespace NUMINAMATH_GPT_find_PQ_length_l2041_204123

-- Define the lengths of the sides of the triangles and the angle
def PQ_length : ℝ := 9
def QR_length : ℝ := 20
def PR_length : ℝ := 15
def ST_length : ℝ := 4.5
def TU_length : ℝ := 7.5
def SU_length : ℝ := 15
def angle_PQR : ℝ := 135
def angle_STU : ℝ := 135

-- Define the similarity condition
def triangles_similar (PQ QR PR ST TU SU angle_PQR angle_STU : ℝ) : Prop :=
  angle_PQR = angle_STU ∧ PQ / QR = ST / TU

-- Theorem statement
theorem find_PQ_length (PQ QR PR ST TU SU angle_PQR angle_STU: ℝ) 
  (H : triangles_similar PQ QR PR ST TU SU angle_PQR angle_STU) : PQ = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_PQ_length_l2041_204123


namespace NUMINAMATH_GPT_problem_solution_l2041_204167

def prop_p (a b c : ℝ) : Prop := a < b → a * c^2 < b * c^2

def prop_q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

theorem problem_solution : (p ∨ ¬q) := sorry

end NUMINAMATH_GPT_problem_solution_l2041_204167


namespace NUMINAMATH_GPT_find_missing_percentage_l2041_204178

theorem find_missing_percentage (P : ℝ) : (P * 50 = 2.125) → (P * 100 = 4.25) :=
by
  sorry

end NUMINAMATH_GPT_find_missing_percentage_l2041_204178


namespace NUMINAMATH_GPT_solve_system_of_equations_l2041_204100

theorem solve_system_of_equations :
  ∀ x y : ℝ,
  (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ (y - x + 1 = x^2 - 3*x) ∧ (x ≠ 0) ∧ (x ≠ 3) →
  (x, y) = (-1, 2) ∨ (x, y) = (2, -1) ∨ (x, y) = (-2, 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2041_204100


namespace NUMINAMATH_GPT_abs_eq_condition_l2041_204108

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end NUMINAMATH_GPT_abs_eq_condition_l2041_204108


namespace NUMINAMATH_GPT_nolan_total_savings_l2041_204101

-- Define the conditions given in the problem
def monthly_savings : ℕ := 3000
def number_of_months : ℕ := 12

-- State the equivalent proof problem in Lean 4
theorem nolan_total_savings : (monthly_savings * number_of_months) = 36000 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_nolan_total_savings_l2041_204101


namespace NUMINAMATH_GPT_domain_of_g_l2041_204105

theorem domain_of_g :
  {x : ℝ | -6*x^2 - 7*x + 8 >= 0} = 
  {x : ℝ | (7 - Real.sqrt 241) / 12 ≤ x ∧ x ≤ (7 + Real.sqrt 241) / 12} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_g_l2041_204105


namespace NUMINAMATH_GPT_range_of_f_l2041_204169

open Set

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

theorem range_of_f :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → 2 ≤ f x ∧ f x ≤ 3 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_range_of_f_l2041_204169


namespace NUMINAMATH_GPT_minimum_value_expression_l2041_204171

-- Define the conditions in the problem
variable (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
variable (h3 : 2 * m + 2 * n = 2)

-- State the theorem proving the minimum value of the given expression
theorem minimum_value_expression : (1 / m + 2 / n) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l2041_204171


namespace NUMINAMATH_GPT_steve_speed_back_l2041_204189

theorem steve_speed_back :
  ∀ (v : ℝ), v > 0 → (20 / v + 20 / (2 * v) = 6) → 2 * v = 10 := 
by
  intros v v_pos h
  sorry

end NUMINAMATH_GPT_steve_speed_back_l2041_204189


namespace NUMINAMATH_GPT_total_number_of_students_l2041_204165

theorem total_number_of_students (sample_size : ℕ) (first_year_selected : ℕ) (third_year_selected : ℕ) (second_year_students : ℕ) (second_year_selected : ℕ) (prob_selection : ℕ) :
  sample_size = 45 →
  first_year_selected = 20 →
  third_year_selected = 10 →
  second_year_students = 300 →
  second_year_selected = sample_size - first_year_selected - third_year_selected →
  prob_selection = second_year_selected / second_year_students →
  (sample_size / prob_selection) = 900 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_number_of_students_l2041_204165


namespace NUMINAMATH_GPT_Tim_is_65_l2041_204120

def James_age : Nat := 23
def John_age : Nat := 35
def Tim_age : Nat := 2 * John_age - 5

theorem Tim_is_65 : Tim_age = 65 := by
  sorry

end NUMINAMATH_GPT_Tim_is_65_l2041_204120


namespace NUMINAMATH_GPT_four_people_possible_l2041_204194

structure Person :=
(first_name : String)
(patronymic : String)
(surname : String)

def noThreePeopleShareSameAttribute (people : List Person) : Prop :=
  ∀ (attr : Person → String), ¬ ∃ (a b c : Person),
    a ∈ people ∧ b ∈ people ∧ c ∈ people ∧ (attr a = attr b) ∧ (attr b = attr c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def anyTwoPeopleShareAnAttribute (people : List Person) : Prop :=
  ∀ (a b : Person), a ∈ people ∧ b ∈ people ∧ a ≠ b →
    (a.first_name = b.first_name ∨ a.patronymic = b.patronymic ∨ a.surname = b.surname)

def validGroup (people : List Person) : Prop :=
  noThreePeopleShareSameAttribute people ∧ anyTwoPeopleShareAnAttribute people

theorem four_people_possible : ∃ (people : List Person), people.length = 4 ∧ validGroup people :=
sorry

end NUMINAMATH_GPT_four_people_possible_l2041_204194


namespace NUMINAMATH_GPT_yellow_marbles_count_l2041_204199

-- Definitions based on given conditions
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def black_marbles : ℕ := 1
def probability_black : ℚ := 1 / 28
def total_marbles : ℕ := 28

-- Problem statement to prove
theorem yellow_marbles_count :
  (total_marbles = blue_marbles + green_marbles + black_marbles + n) →
  (probability_black = black_marbles / total_marbles) →
  n = 12 :=
by
  intros; sorry

end NUMINAMATH_GPT_yellow_marbles_count_l2041_204199


namespace NUMINAMATH_GPT_multiples_of_4_l2041_204131

theorem multiples_of_4 (n : ℕ) (h : n + 23 * 4 = 112) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_4_l2041_204131


namespace NUMINAMATH_GPT_length_of_platform_is_180_l2041_204187

-- Define the train passing a platform and a man with given speeds and times
def train_pass_platform (speed : ℝ) (time_man time_platform : ℝ) (length_train length_platform : ℝ) :=
  time_man = length_train / speed ∧ 
  time_platform = (length_train + length_platform) / speed

-- Given conditions
noncomputable def train_length_platform :=
  ∃ length_platform,
    train_pass_platform 15 20 32 300 length_platform ∧
    length_platform = 180

-- The main theorem we want to prove
theorem length_of_platform_is_180 : train_length_platform :=
sorry

end NUMINAMATH_GPT_length_of_platform_is_180_l2041_204187


namespace NUMINAMATH_GPT_k_satisfies_triangle_condition_l2041_204110

theorem k_satisfies_triangle_condition (k : ℤ) 
  (hk_pos : 0 < k) (a b c : ℝ) (ha_pos : 0 < a) 
  (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (h_ineq : (k : ℝ) * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : k = 6 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end NUMINAMATH_GPT_k_satisfies_triangle_condition_l2041_204110


namespace NUMINAMATH_GPT_Larry_wins_game_probability_l2041_204116

noncomputable def winning_probability_Larry : ℚ :=
  ∑' n : ℕ, if n % 3 = 0 then (2 / 3) ^ (n / 3 * 3) * (1 / 3) else 0

theorem Larry_wins_game_probability : winning_probability_Larry = 9 / 19 :=
by
  sorry

end NUMINAMATH_GPT_Larry_wins_game_probability_l2041_204116


namespace NUMINAMATH_GPT_ab_eq_one_l2041_204117

theorem ab_eq_one (a b : ℝ) (h1 : a ≠ b) (h2 : abs (Real.log a) = abs (Real.log b)) : a * b = 1 := sorry

end NUMINAMATH_GPT_ab_eq_one_l2041_204117


namespace NUMINAMATH_GPT_scientific_notation_21500000_l2041_204118

/-- Express the number 21500000 in scientific notation. -/
theorem scientific_notation_21500000 : 21500000 = 2.15 * 10^7 := 
sorry

end NUMINAMATH_GPT_scientific_notation_21500000_l2041_204118


namespace NUMINAMATH_GPT_solve_equation_l2041_204103

theorem solve_equation : ∃ x : ℤ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2041_204103


namespace NUMINAMATH_GPT_fixed_point_on_line_AB_always_exists_l2041_204175

-- Define the line where P lies
def line (x y : ℝ) : Prop := x + 2 * y = 4

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4

-- Define the point P
def moving_point_P (x y : ℝ) : Prop := line x y

-- Define the function that checks if a point is a tangent to the ellipse
def is_tangent (x0 y0 x y : ℝ) : Prop :=
  moving_point_P x0 y0 → (x * x0 + 4 * y * y0 = 4)

-- Statement: There exists a fixed point (1, 1/2) through which the line AB always passes
theorem fixed_point_on_line_AB_always_exists :
  ∀ (P A B : ℝ × ℝ),
    moving_point_P P.1 P.2 →
    is_tangent P.1 P.2 A.1 A.2 →
    is_tangent P.1 P.2 B.1 B.2 →
    ∃ (F : ℝ × ℝ), F = (1, 1/2) ∧ (F.1 - A.1) / (F.2 - A.2) = (F.1 - B.1) / (F.2 - B.2) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_on_line_AB_always_exists_l2041_204175


namespace NUMINAMATH_GPT_average_distance_per_day_l2041_204107

def miles_monday : ℕ := 12
def miles_tuesday : ℕ := 18
def miles_wednesday : ℕ := 21
def total_days : ℕ := 3

def total_distance : ℕ := miles_monday + miles_tuesday + miles_wednesday

theorem average_distance_per_day : total_distance / total_days = 17 := by
  sorry

end NUMINAMATH_GPT_average_distance_per_day_l2041_204107


namespace NUMINAMATH_GPT_length_of_edge_l2041_204191

-- Define all necessary conditions
def is_quadrangular_pyramid (e : ℝ) : Prop :=
  (8 * e = 14.8)

-- State the main theorem which is the equivalent proof problem
theorem length_of_edge (e : ℝ) (h : is_quadrangular_pyramid e) : e = 1.85 :=
by
  sorry

end NUMINAMATH_GPT_length_of_edge_l2041_204191


namespace NUMINAMATH_GPT_nh3_oxidation_mass_l2041_204114

theorem nh3_oxidation_mass
  (initial_volume : ℚ)
  (initial_cl2_percentage : ℚ)
  (initial_n2_percentage : ℚ)
  (escaped_volume : ℚ)
  (escaped_cl2_percentage : ℚ)
  (escaped_n2_percentage : ℚ)
  (molar_volume : ℚ)
  (cl2_molar_mass : ℚ)
  (nh3_molar_mass : ℚ) :
  initial_volume = 1.12 →
  initial_cl2_percentage = 0.9 →
  initial_n2_percentage = 0.1 →
  escaped_volume = 0.672 →
  escaped_cl2_percentage = 0.5 →
  escaped_n2_percentage = 0.5 →
  molar_volume = 22.4 →
  cl2_molar_mass = 71 →
  nh3_molar_mass = 17 →
  ∃ (mass_nh3_oxidized : ℚ),
    mass_nh3_oxidized = 0.34 := 
by {
  sorry
}

end NUMINAMATH_GPT_nh3_oxidation_mass_l2041_204114


namespace NUMINAMATH_GPT_baker_total_cost_is_correct_l2041_204179

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end NUMINAMATH_GPT_baker_total_cost_is_correct_l2041_204179


namespace NUMINAMATH_GPT_negation_of_exists_l2041_204158

theorem negation_of_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l2041_204158


namespace NUMINAMATH_GPT_eval_f_g_at_4_l2041_204119

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem eval_f_g_at_4 : f (g 4) = (25 / 7) * Real.sqrt 21 := by
  sorry

end NUMINAMATH_GPT_eval_f_g_at_4_l2041_204119


namespace NUMINAMATH_GPT_car_owners_without_motorcycles_l2041_204148

theorem car_owners_without_motorcycles
  (total_adults : ℕ)
  (car_owners : ℕ)
  (motorcycle_owners : ℕ)
  (all_owners : total_adults = 400)
  (john_owns_cars : car_owners = 370)
  (john_owns_motorcycles : motorcycle_owners = 50)
  (all_adult_owners : total_adults = car_owners + motorcycle_owners - (car_owners - motorcycle_owners)) : 
  (car_owners - (car_owners + motorcycle_owners - total_adults) = 350) :=
by {
  sorry
}

end NUMINAMATH_GPT_car_owners_without_motorcycles_l2041_204148


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l2041_204185

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0 → ab > 0) ∧ (ab > 0 → a + b > 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l2041_204185


namespace NUMINAMATH_GPT_roots_purely_imaginary_l2041_204183

open Complex

/-- 
  If m is a purely imaginary number, then the roots of the equation 
  8z^2 + 4i * z - m = 0 are purely imaginary.
-/
theorem roots_purely_imaginary (m : ℂ) (hm : m.im ≠ 0 ∧ m.re = 0) : 
  ∀ z : ℂ, 8 * z^2 + 4 * Complex.I * z - m = 0 → z.im ≠ 0 ∧ z.re = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_purely_imaginary_l2041_204183


namespace NUMINAMATH_GPT_deck_cost_l2041_204121

variable (rareCount : ℕ := 19)
variable (uncommonCount : ℕ := 11)
variable (commonCount : ℕ := 30)
variable (rareCost : ℝ := 1.0)
variable (uncommonCost : ℝ := 0.5)
variable (commonCost : ℝ := 0.25)

theorem deck_cost : rareCount * rareCost + uncommonCount * uncommonCost + commonCount * commonCost = 32 := by
  sorry

end NUMINAMATH_GPT_deck_cost_l2041_204121


namespace NUMINAMATH_GPT_find_x_l2041_204152

theorem find_x (x : ℝ) (h : 0.90 * 600 = 0.50 * x) : x = 1080 :=
sorry

end NUMINAMATH_GPT_find_x_l2041_204152


namespace NUMINAMATH_GPT_arithmetic_seq_necessary_not_sufficient_l2041_204154

noncomputable def arithmetic_sequence (a b c : ℝ) : Prop :=
  a + c = 2 * b

noncomputable def proposition_B (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ (a / b) + (c / b) = 2

theorem arithmetic_seq_necessary_not_sufficient (a b c : ℝ) :
  (arithmetic_sequence a b c → proposition_B a b c) ∧ 
  (∃ a' b' c', arithmetic_sequence a' b' c' ∧ ¬ proposition_B a' b' c') := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_necessary_not_sufficient_l2041_204154
