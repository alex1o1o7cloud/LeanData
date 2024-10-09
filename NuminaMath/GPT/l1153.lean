import Mathlib

namespace binary_sum_to_decimal_l1153_115337

theorem binary_sum_to_decimal :
  let bin1 := "1101011"
  let bin2 := "1010110"
  let dec1 := 64 + 32 + 0 + 8 + 0 + 2 + 1 -- decimal value of "1101011"
  let dec2 := 64 + 0 + 16 + 0 + 4 + 2 + 0 -- decimal value of "1010110"
  dec1 + dec2 = 193 := by
  sorry

end binary_sum_to_decimal_l1153_115337


namespace polynomial_degree_bound_l1153_115366

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) 
  (hm_pos : 0 < m)
  (hn_pos : 0 < n)
  (hk_pos : 2 ≤ k)
  (hP_odd : ∀ i, P.coeff i % 2 = 1) 
  (h_div : (X - 1) ^ m ∣ P)
  (hm_bound : m ≥ 2 ^ k) :
  n ≥ 2 ^ (k + 1) - 1 := sorry

end polynomial_degree_bound_l1153_115366


namespace not_perfect_square_n_l1153_115390

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

theorem not_perfect_square_n (n : ℕ) : ¬ isPerfectSquare (4 * n^2 + 4 * n + 4) :=
sorry

end not_perfect_square_n_l1153_115390


namespace bart_pages_bought_l1153_115369

theorem bart_pages_bought (total_money : ℝ) (price_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_money = 10) (h2 : price_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_money / price_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_pages_bought_l1153_115369


namespace quiz_show_prob_l1153_115373

-- Definitions extracted from the problem conditions
def n : ℕ := 4 -- Number of questions
def p_correct : ℚ := 1 / 4 -- Probability of guessing a question correctly
def p_incorrect : ℚ := 3 / 4 -- Probability of guessing a question incorrectly

-- We need to prove that the probability of answering at least 3 out of 4 questions correctly 
-- by guessing randomly is 13/256.
theorem quiz_show_prob :
  (Nat.choose n 3 * (p_correct ^ 3) * (p_incorrect ^ 1) +
   Nat.choose n 4 * (p_correct ^ 4)) = 13 / 256 :=
by sorry

end quiz_show_prob_l1153_115373


namespace average_of_last_three_numbers_l1153_115351

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l1153_115351


namespace third_place_amount_l1153_115389

noncomputable def total_people : ℕ := 13
noncomputable def money_per_person : ℝ := 5
noncomputable def total_money : ℝ := total_people * money_per_person

noncomputable def first_place_percentage : ℝ := 0.65
noncomputable def second_third_place_percentage : ℝ := 0.35
noncomputable def split_factor : ℝ := 0.5

noncomputable def first_place_money : ℝ := first_place_percentage * total_money
noncomputable def second_third_place_money : ℝ := second_third_place_percentage * total_money
noncomputable def third_place_money : ℝ := split_factor * second_third_place_money

theorem third_place_amount : third_place_money = 11.38 := by
  sorry

end third_place_amount_l1153_115389


namespace evaluate_expression_l1153_115331

theorem evaluate_expression : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by sorry

end evaluate_expression_l1153_115331


namespace cos_identity_l1153_115398

theorem cos_identity (α : ℝ) (h : Real.cos (Real.pi / 8 - α) = 1 / 6) :
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end cos_identity_l1153_115398


namespace minimum_value_expr_l1153_115313

theorem minimum_value_expr (a : ℝ) (h₀ : 0 < a) (h₁ : a < 2) : 
  ∃ (m : ℝ), m = (4 / a + 1 / (2 - a)) ∧ m = 9 / 2 :=
by
  sorry

end minimum_value_expr_l1153_115313


namespace range_of_m_l1153_115350

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ x y, x + y/4 < m^2 - 3*m) : m < -1 ∨ m > 4 :=
sorry

end range_of_m_l1153_115350


namespace circle_radius_five_l1153_115370

theorem circle_radius_five (c : ℝ) : (∃ x y : ℝ, x^2 + 10 * x + y^2 + 8 * y + c = 0) ∧ 
                                     ((x + 5)^2 + (y + 4)^2 = 25) → c = 16 :=
by
  sorry

end circle_radius_five_l1153_115370


namespace find_x_l1153_115352

-- Define the conditions as given in the problem
def angle1 (x : ℝ) : ℝ := 6 * x
def angle2 (x : ℝ) : ℝ := 3 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 5 * x
def sum_of_angles (x : ℝ) : ℝ := angle1 x + angle2 x + angle3 x + angle4 x

-- State the problem: prove that x equals 24 given the sum of angles is 360 degrees
theorem find_x (x : ℝ) (h : sum_of_angles x = 360) : x = 24 :=
by
  sorry

end find_x_l1153_115352


namespace seq_sum_11_l1153_115377

noncomputable def S (n : ℕ) : ℕ := sorry

noncomputable def a (n : ℕ) : ℕ := sorry

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem seq_sum_11 :
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) ∧
  (is_arithmetic_sequence a) ∧
  (3 * (a 2 + a 4) + 2 * (a 6 + a 9 + a 12) = 12) →
  S 11 = 11 :=
by
  sorry

end seq_sum_11_l1153_115377


namespace circle_tangent_locus_l1153_115334

theorem circle_tangent_locus (a b : ℝ) :
  (∃ r : ℝ, (a ^ 2 + b ^ 2 = (r + 1) ^ 2) ∧ ((a - 3) ^ 2 + b ^ 2 = (5 - r) ^ 2)) →
  3 * a ^ 2 + 4 * b ^ 2 - 14 * a - 49 = 0 := by
  sorry

end circle_tangent_locus_l1153_115334


namespace eval_at_d_eq_4_l1153_115308

theorem eval_at_d_eq_4 : ((4: ℕ) ^ 4 - (4: ℕ) * ((4: ℕ) - 2) ^ 4) ^ 4 = 136048896 :=
by
  sorry

end eval_at_d_eq_4_l1153_115308


namespace arcsin_cos_eq_neg_pi_div_six_l1153_115330

theorem arcsin_cos_eq_neg_pi_div_six :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  sorry

end arcsin_cos_eq_neg_pi_div_six_l1153_115330


namespace peter_fish_caught_l1153_115339

theorem peter_fish_caught (n : ℕ) (h : 3 * n = n + 24) : n = 12 :=
sorry

end peter_fish_caught_l1153_115339


namespace op_op_k_l1153_115315

def op (x y : ℝ) : ℝ := x^3 + x - y

theorem op_op_k (k : ℝ) : op k (op k k) = k := sorry

end op_op_k_l1153_115315


namespace last_operation_ends_at_eleven_am_l1153_115324

-- Definitions based on conditions
def operation_duration : ℕ := 45 -- duration of each operation in minutes
def start_time : ℕ := 8 * 60 -- start time of the first operation in minutes since midnight
def interval : ℕ := 15 -- interval between operations in minutes
def total_operations : ℕ := 10 -- total number of operations

-- Compute the start time of the last operation (10th operation)
def start_time_last_operation : ℕ := start_time + interval * (total_operations - 1)

-- Compute the end time of the last operation
def end_time_last_operation : ℕ := start_time_last_operation + operation_duration

-- End time of the last operation expected to be 11:00 a.m. in minutes since midnight
def expected_end_time : ℕ := 11 * 60 

theorem last_operation_ends_at_eleven_am : 
  end_time_last_operation = expected_end_time := by
  sorry

end last_operation_ends_at_eleven_am_l1153_115324


namespace number_of_books_before_purchase_l1153_115393

theorem number_of_books_before_purchase (x : ℕ) (h1 : x + 140 = (27 / 25) * x) : x = 1750 :=
by
  sorry

end number_of_books_before_purchase_l1153_115393


namespace cookies_prepared_l1153_115311

theorem cookies_prepared (n_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1 : n_people = 25) (h2 : cookies_per_person = 45) : total_cookies = 1125 :=
by
  sorry

end cookies_prepared_l1153_115311


namespace hyperbola_b_value_l1153_115358

theorem hyperbola_b_value (b : ℝ) (h₁ : b > 0) 
  (h₂ : ∃ x y, x^2 - (y^2 / b^2) = 1 ∧ (∀ (c : ℝ), c = Real.sqrt (1 + b^2) → c / 1 = 2)) : b = Real.sqrt 3 :=
by { sorry }

end hyperbola_b_value_l1153_115358


namespace evaluate_expression_l1153_115371

theorem evaluate_expression (x y z : ℝ) (hxy : x > y ∧ y > 1) (hz : z > 0) :
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x / y)^(y - x) :=
by
  sorry

end evaluate_expression_l1153_115371


namespace sqrt_sum_eval_l1153_115388

theorem sqrt_sum_eval : 
  (Real.sqrt 50 + Real.sqrt 72) = 11 * Real.sqrt 2 := 
by 
  sorry

end sqrt_sum_eval_l1153_115388


namespace donut_combinations_l1153_115314

-- Define the problem statement where Bill needs to purchase 10 donuts,
-- with at least one of each of the 5 kinds, and calculate the combinations.

def count_donut_combinations : ℕ :=
  Nat.choose 9 4

theorem donut_combinations :
  count_donut_combinations = 126 :=
by
  -- Proof can be filled in here
  sorry

end donut_combinations_l1153_115314


namespace baseball_cards_given_l1153_115372

theorem baseball_cards_given
  (initial_cards : ℕ)
  (maria_take : ℕ)
  (peter_cards : ℕ)
  (paul_triples : ℕ)
  (final_cards : ℕ)
  (h1 : initial_cards = 15)
  (h2 : maria_take = (initial_cards + 1) / 2)
  (h3 : final_cards = 3 * (initial_cards - maria_take - peter_cards))
  (h4 : final_cards = 18) :
  peter_cards = 1 := 
sorry

end baseball_cards_given_l1153_115372


namespace length_PR_in_triangle_l1153_115310

/-- In any triangle PQR, given:
  PQ = 7, QR = 10, median PS = 5,
  the length of PR must be sqrt(149). -/
theorem length_PR_in_triangle (PQ QR PS : ℝ) (PQ_eq : PQ = 7) (QR_eq : QR = 10) (PS_eq : PS = 5) : 
  ∃ (PR : ℝ), PR = Real.sqrt 149 := 
sorry

end length_PR_in_triangle_l1153_115310


namespace total_cases_is_8_l1153_115301

def num_blue_cards : Nat := 3
def num_yellow_cards : Nat := 5

def total_cases : Nat := num_blue_cards + num_yellow_cards

theorem total_cases_is_8 : total_cases = 8 := by
  sorry

end total_cases_is_8_l1153_115301


namespace intersection_correct_l1153_115360

variable (A B : Set ℝ)  -- Define variables A and B as sets of real numbers

-- Define set A as {x | -3 ≤ x < 4}
def setA : Set ℝ := {x | -3 ≤ x ∧ x < 4}

-- Define set B as {x | -2 ≤ x ≤ 5}
def setB : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- The goal is to prove the intersection of A and B is {x | -2 ≤ x < 4}
theorem intersection_correct : setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} := sorry

end intersection_correct_l1153_115360


namespace sum_of_possible_values_l1153_115374

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 8) = 4) : ∃ S : ℝ, S = 8 :=
sorry

end sum_of_possible_values_l1153_115374


namespace increase_in_average_l1153_115338

theorem increase_in_average (A : ℤ) (avg_after_12 : ℤ) (score_12th_inning : ℤ) (A : ℤ) : 
  score_12th_inning = 75 → avg_after_12 = 64 → (11 * A + score_12th_inning = 768) → (avg_after_12 - A = 1) :=
by
  intros h_score h_avg h_total
  sorry

end increase_in_average_l1153_115338


namespace geometric_sequence_sum_l1153_115348

theorem geometric_sequence_sum (k : ℕ) (h1 : a_1 = 1) (h2 : a_k = 243) (h3 : q = 3) : S_k = 364 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end geometric_sequence_sum_l1153_115348


namespace find_concentration_of_second_mixture_l1153_115305

noncomputable def concentration_of_second_mixture (total_volume : ℝ) (final_percent : ℝ) (pure_antifreeze : ℝ) (pure_antifreeze_amount : ℝ) : ℝ :=
  let remaining_volume := total_volume - pure_antifreeze_amount
  let final_pure_amount := final_percent * total_volume
  let required_pure_antifreeze := final_pure_amount - pure_antifreeze
  (required_pure_antifreeze / remaining_volume) * 100

theorem find_concentration_of_second_mixture :
  concentration_of_second_mixture 55 0.20 6.11 6.11 = 10 :=
by
  simp [concentration_of_second_mixture]
  sorry

end find_concentration_of_second_mixture_l1153_115305


namespace Marie_speed_l1153_115329

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end Marie_speed_l1153_115329


namespace describe_T_l1153_115320

def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (common : ℝ), 
    (common = 5 ∧ p.1 + 3 = common ∧ p.2 - 6 ≤ common) ∨
    (common = 5 ∧ p.2 - 6 = common ∧ p.1 + 3 ≤ common) ∨
    (common = p.1 + 3 ∧ common = p.2 - 6 ∧ common ≤ 5)}

theorem describe_T :
  T = {(2, y) | y ≤ 11} ∪ { (x, 11) | x ≤ 2} ∪ { (x, x + 9) | x ≤ 2} :=
by
  sorry

end describe_T_l1153_115320


namespace interest_rate_calculation_l1153_115349

theorem interest_rate_calculation
  (P : ℕ) 
  (I : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (principal : P = 9200) 
  (time : T = 3) 
  (interest_diff : P - 5888 = I) 
  (interest_formula : I = P * R * T / 100) 
  : R = 12 :=
sorry

end interest_rate_calculation_l1153_115349


namespace car_distances_equal_600_l1153_115300

-- Define the variables
def time_R (t : ℝ) := t
def speed_R := 50
def time_P (t : ℝ) := t - 2
def speed_P := speed_R + 10
def distance (t : ℝ) := speed_R * time_R t

-- The Lean theorem statement
theorem car_distances_equal_600 (t : ℝ) (h : time_R t = t) (h1 : speed_R = 50) 
  (h2 : time_P t = t - 2) (h3 : speed_P = speed_R + 10) :
  distance t = 600 :=
by
  -- We would provide the proof here, but for now we use sorry to indicate the proof is omitted.
  sorry

end car_distances_equal_600_l1153_115300


namespace find_k_l1153_115394

-- Define the vector structures for i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define the vectors a and b based on i, j, and k
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Statement of the theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by sorry

end find_k_l1153_115394


namespace period_sine_transformed_l1153_115343

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end period_sine_transformed_l1153_115343


namespace karen_total_nuts_l1153_115378

variable (x y : ℝ)
variable (hx : x = 0.25)
variable (hy : y = 0.25)

theorem karen_total_nuts : x + y = 0.50 := by
  rw [hx, hy]
  norm_num

end karen_total_nuts_l1153_115378


namespace find_ordered_pair_l1153_115317

theorem find_ordered_pair:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 10 * m * n = 45 - 5 * m - 3 * n ∧ (m, n) = (1, 11) :=
by
  sorry

end find_ordered_pair_l1153_115317


namespace even_function_f3_l1153_115318

theorem even_function_f3 (a : ℝ) (h : ∀ x : ℝ, (x + 2) * (x - a) = (-x + 2) * (-x - a)) : (3 + 2) * (3 - a) = 5 := by
  sorry

end even_function_f3_l1153_115318


namespace product_of_05_and_2_3_is_1_3_l1153_115379

theorem product_of_05_and_2_3_is_1_3 : (0.5 * (2 / 3) = 1 / 3) :=
by sorry

end product_of_05_and_2_3_is_1_3_l1153_115379


namespace mollys_present_age_l1153_115367

theorem mollys_present_age (x : ℤ) (h : x + 18 = 5 * (x - 6)) : x = 12 := by
  sorry

end mollys_present_age_l1153_115367


namespace balls_in_boxes_l1153_115309

theorem balls_in_boxes : (2^7 = 128) := 
by
  -- number of balls
  let n : ℕ := 7
  -- number of boxes
  let b : ℕ := 2
  have h : b ^ n = 128 := by sorry
  exact h

end balls_in_boxes_l1153_115309


namespace least_positive_integer_condition_l1153_115319

theorem least_positive_integer_condition (n : ℕ) :
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9, 11], n % d = 1) → n = 10396 := 
by
  sorry

end least_positive_integer_condition_l1153_115319


namespace remainder_when_divided_by_63_l1153_115306

theorem remainder_when_divided_by_63 (x : ℤ) (h1 : ∃ q : ℤ, x = 63 * q + r ∧ 0 ≤ r ∧ r < 63) (h2 : ∃ k : ℤ, x = 9 * k + 2) :
  ∃ r : ℤ, 0 ≤ r ∧ r < 63 ∧ r = 7 :=
by
  sorry

end remainder_when_divided_by_63_l1153_115306


namespace abs_diff_of_two_numbers_l1153_115396

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : |x - y| = 3 :=
by
  sorry

end abs_diff_of_two_numbers_l1153_115396


namespace proof_a_squared_plus_b_squared_l1153_115355

theorem proof_a_squared_plus_b_squared (a b : ℝ) (h1 : (a + b) ^ 2 = 4) (h2 : a * b = 1) : a ^ 2 + b ^ 2 = 2 := 
by 
  sorry

end proof_a_squared_plus_b_squared_l1153_115355


namespace point_on_or_outside_circle_l1153_115382

theorem point_on_or_outside_circle (a : ℝ) : 
  let P := (a, 2 - a)
  let r := 2
  let center := (0, 0)
  let distance_square := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_square >= r := 
by
  sorry

end point_on_or_outside_circle_l1153_115382


namespace check_range_a_l1153_115354

open Set

def A : Set ℝ := {x | x < -1/2 ∨ x > 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0}

theorem check_range_a :
  (∃! x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ : ℝ) ∈ A ∩ B a ∧ (x₂ : ℝ) ∈ A ∩ B a) →
  a ∈ Icc (4/3 : ℝ) (15/8 : ℝ) :=
sorry

end check_range_a_l1153_115354


namespace fundraising_part1_fundraising_part2_l1153_115365

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end fundraising_part1_fundraising_part2_l1153_115365


namespace circle_condition_l1153_115345

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 5*m = 0) →
  (m < 1) :=
by
  sorry

end circle_condition_l1153_115345


namespace function_decreasing_interval_l1153_115322

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem function_decreasing_interval :
  ∃ I : Set ℝ, I = (Set.Ioo 0 2) ∧ ∀ x ∈ I, deriv f x < 0 :=
by
  sorry

end function_decreasing_interval_l1153_115322


namespace yellow_faces_of_cube_l1153_115375

theorem yellow_faces_of_cube (n : ℕ) (h : 6 * n^2 = (1 / 3) * (6 * n^3)) : n = 3 :=
by {
  sorry
}

end yellow_faces_of_cube_l1153_115375


namespace slices_per_person_l1153_115328

theorem slices_per_person (total_slices : ℕ) (total_people : ℕ) (h_slices : total_slices = 12) (h_people : total_people = 3) :
  total_slices / total_people = 4 :=
by
  sorry

end slices_per_person_l1153_115328


namespace radius_of_sector_l1153_115332

theorem radius_of_sector (l : ℝ) (α : ℝ) (R : ℝ) (h1 : l = 2 * π / 3) (h2 : α = π / 3) : R = 2 := by
  have : l = |α| * R := by sorry
  rw [h1, h2] at this
  sorry

end radius_of_sector_l1153_115332


namespace find_EQ_length_l1153_115385

theorem find_EQ_length (a b c d : ℕ) (parallel : Prop) (circle_tangent : Prop) :
  a = 105 ∧ b = 45 ∧ c = 21 ∧ d = 80 ∧ parallel ∧ circle_tangent → (∃ x : ℚ, x = 336 / 5) :=
by
  sorry

end find_EQ_length_l1153_115385


namespace minimum_value_of_product_l1153_115368

theorem minimum_value_of_product (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 30 := 
sorry

end minimum_value_of_product_l1153_115368


namespace flag_blue_area_l1153_115364

theorem flag_blue_area (A C₁ C₃ : ℝ) (h₀ : A = 1.0) (h₁ : C₁ + C₃ = 0.36 * A) :
  C₃ = 0.02 * A := by
  sorry

end flag_blue_area_l1153_115364


namespace range_of_a_l1153_115359

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by
  sorry

end range_of_a_l1153_115359


namespace repeating_decimal_sum_to_fraction_l1153_115346

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0045 : ℚ := 45 / 9999
def repeating_decimal_000678 : ℚ := 678 / 999999

theorem repeating_decimal_sum_to_fraction :
  repeating_decimal_123 + repeating_decimal_0045 + repeating_decimal_000678 = 128178 / 998001000 :=
by
  sorry

end repeating_decimal_sum_to_fraction_l1153_115346


namespace sequence_a4_value_l1153_115391

theorem sequence_a4_value :
  ∀ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 3) → a 4 = 29 :=
by sorry

end sequence_a4_value_l1153_115391


namespace product_of_terms_geometric_sequence_l1153_115356

variable {a : ℕ → ℝ}
variable {q : ℝ}
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem product_of_terms_geometric_sequence
  (ha: geometric_sequence a q)
  (h3_4: a 3 * a 4 = 6) :
  a 2 * a 5 = 6 :=
by
  sorry

end product_of_terms_geometric_sequence_l1153_115356


namespace fred_games_last_year_l1153_115342

def total_games : Nat := 47
def games_this_year : Nat := 36

def games_last_year (total games games this year : Nat) : Nat := total_games - games_this_year

theorem fred_games_last_year : games_last_year total_games games_this_year = 11 :=
by
  sorry

end fred_games_last_year_l1153_115342


namespace ordering_of_abc_l1153_115395

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem ordering_of_abc : b < a ∧ a < c := by
  sorry

end ordering_of_abc_l1153_115395


namespace total_salaries_proof_l1153_115316

def total_salaries (A_salary B_salary : ℝ) :=
  A_salary + B_salary

theorem total_salaries_proof : ∀ A_salary B_salary : ℝ,
  A_salary = 3000 →
  (0.05 * A_salary = 0.15 * B_salary) →
  total_salaries A_salary B_salary = 4000 :=
by
  intros A_salary B_salary h1 h2
  rw [h1] at h2
  sorry

end total_salaries_proof_l1153_115316


namespace point_on_y_axis_l1153_115384

theorem point_on_y_axis (y : ℝ) :
  let A := (1, 0, 2)
  let B := (1, -3, 1)
  let M := (0, y, 0)
  dist A M = dist B M → y = -1 :=
by sorry

end point_on_y_axis_l1153_115384


namespace exists_two_digit_number_N_l1153_115381

-- Statement of the problem
theorem exists_two_digit_number_N : 
  ∃ (N : ℕ), (∃ (a b : ℕ), N = 10 * a + b ∧ N = a * b + 2 * (a + b) ∧ 10 ≤ N ∧ N < 100) :=
by
  sorry

end exists_two_digit_number_N_l1153_115381


namespace num_true_statements_l1153_115341

theorem num_true_statements :
  (if (2 : ℝ) = 2 then (2 : ℝ)^2 - 4 = 0 else false) ∧
  ((∀ (x : ℝ), x^2 - 4 = 0 → x = 2) ∨ (∃ (x : ℝ), x^2 - 4 = 0 ∧ x ≠ 2)) ∧
  ((∀ (x : ℝ), x ≠ 2 → x^2 - 4 ≠ 0) ∨ (∃ (x : ℝ), x ≠ 2 ∧ x^2 - 4 = 0)) ∧
  ((∀ (x : ℝ), x^2 - 4 ≠ 0 → x ≠ 2) ∨ (∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ x = 2)) :=
sorry

end num_true_statements_l1153_115341


namespace solve_for_x_l1153_115323

def equation (x : ℝ) (y : ℝ) : Prop := 5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)

def y_condition (x : ℝ) : ℝ := 3 * x

theorem solve_for_x (x : ℝ) :
  equation x (y_condition x) ↔ (x = 1/3 ∨ x = -2/9) := by
  sorry

end solve_for_x_l1153_115323


namespace polynomial_degrees_l1153_115326

-- Define the degree requirement for the polynomial.
def polynomial_deg_condition (m n : ℕ) : Prop :=
  2 + m = 5 ∧ n - 2 = 0 ∧ 2 + 2 = 5

theorem polynomial_degrees (m n : ℕ) (h : polynomial_deg_condition m n) : m - n = 1 :=
by
  have h1 : 2 + m = 5 := h.1
  have h2 : n - 2 = 0 := h.2.1
  have h3 := h.2.2
  have : m = 3 := by linarith
  have : n = 2 := by linarith
  linarith

end polynomial_degrees_l1153_115326


namespace necessary_but_not_sufficient_l1153_115312

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 2 → (x^2 - x - 2 >= 0) ∨ (x >= -1 ∧ x < 2)) ∧ ((-1 < x ∧ x < 2) → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_l1153_115312


namespace total_share_proof_l1153_115340

variable (P R : ℕ) -- Parker's share and Richie's share
variable (total_share : ℕ) -- Total share

-- Define conditions
def ratio_condition : Prop := (P : ℕ) / 2 = (R : ℕ) / 3
def parker_share_condition : Prop := P = 50

-- Prove the total share is 125
theorem total_share_proof 
  (h1 : ratio_condition P R)
  (h2 : parker_share_condition P) : 
  total_share = 125 :=
sorry

end total_share_proof_l1153_115340


namespace prove_expression_l1153_115399

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 5)

lemma root_of_unity : omega^5 = 1 := sorry
lemma sum_of_roots : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sorry

noncomputable def z := omega + omega^2 + omega^3 + omega^4

theorem prove_expression : z^2 + z + 1 = 1 :=
by 
  have h1 : omega^5 = 1 := root_of_unity
  have h2 : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sum_of_roots
  show z^2 + z + 1 = 1
  {
    -- Proof omitted
    sorry
  }

end prove_expression_l1153_115399


namespace solve_for_x_l1153_115344

theorem solve_for_x (x : ℝ) : 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end solve_for_x_l1153_115344


namespace find_value_of_a20_l1153_115304

variable {α : Type*} [LinearOrder α] [Field α]

def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

def arithmetic_sum (a d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem find_value_of_a20 
  (a d : ℝ) 
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 5 = 4)
  (h2 : arithmetic_sum a d 15 = 60) :
  arithmetic_sequence a d 20 = 10 := 
sorry

end find_value_of_a20_l1153_115304


namespace tan_ratio_given_sin_equation_l1153_115387

theorem tan_ratio_given_sin_equation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (2*α + β) = (3/2) * Real.sin β) : 
  Real.tan (α + β) / Real.tan α = 5 :=
by
  -- Proof goes here
  sorry

end tan_ratio_given_sin_equation_l1153_115387


namespace solution_set_l1153_115333

def solve_inequalities (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * x ^ 2 - x - 1 > 0

theorem solution_set : { x : ℝ | solve_inequalities x } = { x : ℝ | (-2 ≤ x ∧ x < 1/2) ∨ (1 < x ∧ x < 6) } :=
by sorry

end solution_set_l1153_115333


namespace A_wins_if_N_is_perfect_square_l1153_115302

noncomputable def player_A_can_always_win (N : ℕ) : Prop :=
  ∀ (B_moves : ℕ → ℕ), ∃ (A_moves : ℕ → ℕ), A_moves 0 = N ∧
  (∀ n, B_moves n = 0 ∨ (A_moves n ∣ B_moves (n + 1) ∨ B_moves (n + 1) ∣ A_moves n))

theorem A_wins_if_N_is_perfect_square :
  ∀ N : ℕ, player_A_can_always_win N ↔ ∃ n : ℕ, N = n * n := sorry

end A_wins_if_N_is_perfect_square_l1153_115302


namespace cost_of_parts_per_tire_repair_is_5_l1153_115336

-- Define the given conditions
def charge_per_tire_repair : ℤ := 20
def num_tire_repairs : ℤ := 300
def charge_per_complex_repair : ℤ := 300
def num_complex_repairs : ℤ := 2
def cost_per_complex_repair_parts : ℤ := 50
def retail_shop_profit : ℤ := 2000
def fixed_expenses : ℤ := 4000
def total_profit : ℤ := 3000

-- Define the calculation for total revenue
def total_revenue : ℤ := 
    (charge_per_tire_repair * num_tire_repairs) + 
    (charge_per_complex_repair * num_complex_repairs) + 
    retail_shop_profit

-- Define the calculation for total expenses
def total_expenses : ℤ := total_revenue - total_profit

-- Define the calculation for parts cost of tire repairs
def parts_cost_tire_repairs : ℤ := 
    total_expenses - (cost_per_complex_repair_parts * num_complex_repairs) - fixed_expenses

def cost_per_tire_repair : ℤ := parts_cost_tire_repairs / num_tire_repairs

-- The statement to be proved
theorem cost_of_parts_per_tire_repair_is_5 : cost_per_tire_repair = 5 := by
    sorry

end cost_of_parts_per_tire_repair_is_5_l1153_115336


namespace roof_problem_l1153_115361

theorem roof_problem (w l : ℝ) (h1 : l = 4 * w) (h2 : l * w = 900) : l - w = 45 := 
by
  sorry

end roof_problem_l1153_115361


namespace area_of_rectangle_l1153_115327

theorem area_of_rectangle
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = 16)
  (h_c : c = 17)
  (h_diag : a^2 + b^2 = c^2) :
  abs (a * b - 91.9136) < 0.0001 :=
by
  sorry

end area_of_rectangle_l1153_115327


namespace roots_of_quadratic_l1153_115307

theorem roots_of_quadratic (x1 x2 : ℝ) (h : ∀ x, x^2 - 3 * x - 2 = 0 → x = x1 ∨ x = x2) :
  x1 * x2 + x1 + x2 = 1 :=
sorry

end roots_of_quadratic_l1153_115307


namespace function_increasing_iff_l1153_115386

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - a * x

theorem function_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 - a) ↔ a ≤ 0 :=
by
  sorry

end function_increasing_iff_l1153_115386


namespace action_figure_collection_complete_l1153_115383

theorem action_figure_collection_complete (act_figures : ℕ) (cost_per_fig : ℕ) (extra_money_needed : ℕ) (total_collection : ℕ) 
    (h1 : act_figures = 7) 
    (h2 : cost_per_fig = 8) 
    (h3 : extra_money_needed = 72) : 
    total_collection = 16 :=
by
  sorry

end action_figure_collection_complete_l1153_115383


namespace total_fruits_consumed_l1153_115353

def starting_cherries : ℝ := 16.5
def remaining_cherries : ℝ := 6.3

def starting_strawberries : ℝ := 10.7
def remaining_strawberries : ℝ := 8.4

def starting_blueberries : ℝ := 20.2
def remaining_blueberries : ℝ := 15.5

theorem total_fruits_consumed 
  (sc : ℝ := starting_cherries)
  (rc : ℝ := remaining_cherries)
  (ss : ℝ := starting_strawberries)
  (rs : ℝ := remaining_strawberries)
  (sb : ℝ := starting_blueberries)
  (rb : ℝ := remaining_blueberries) :
  (sc - rc) + (ss - rs) + (sb - rb) = 17.2 := by
  sorry

end total_fruits_consumed_l1153_115353


namespace find_smallest_positive_angle_l1153_115376

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem find_smallest_positive_angle :
  ∃ φ > 0, cos_deg φ = sin_deg 45 + cos_deg 37 - sin_deg 23 - cos_deg 11 ∧ φ = 53 := 
by
  sorry

end find_smallest_positive_angle_l1153_115376


namespace find_p_of_abs_sum_roots_eq_five_l1153_115363

theorem find_p_of_abs_sum_roots_eq_five (p : ℝ) : 
  (∃ x y : ℝ, x + y = -p ∧ x * y = -6 ∧ |x| + |y| = 5) → (p = 1 ∨ p = -1) := by
  sorry

end find_p_of_abs_sum_roots_eq_five_l1153_115363


namespace arithmetic_geometric_sequence_l1153_115321

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the first term, common difference and positions of terms in geometric sequence
def a1 : ℤ := -8
def d : ℤ := 2
def a3 := arithmetic_sequence a1 d 2
def a4 := arithmetic_sequence a1 d 3

-- Conditions for the terms forming a geometric sequence
def geometric_condition (a b c : ℤ) : Prop :=
  b^2 = a * c

-- Statement to prove
theorem arithmetic_geometric_sequence :
  geometric_condition a1 a3 a4 → a1 = -8 :=
by
  intro h
  -- Proof can be filled in here
  sorry

end arithmetic_geometric_sequence_l1153_115321


namespace sin_alpha_minus_3pi_l1153_115335

theorem sin_alpha_minus_3pi (α : ℝ) (h : Real.sin α = 3/5) : Real.sin (α - 3 * Real.pi) = -3/5 :=
by
  sorry

end sin_alpha_minus_3pi_l1153_115335


namespace solution_set_of_inequality_l1153_115303

theorem solution_set_of_inequality : {x : ℝ | -2 < x ∧ x < 1} = {x : ℝ | -x^2 - x + 2 > 0} :=
by
  sorry

end solution_set_of_inequality_l1153_115303


namespace train_crosses_bridge_in_30_seconds_l1153_115392

/--
A train 155 metres long, travelling at 45 km/hr, can cross a bridge with length 220 metres in 30 seconds.
-/
theorem train_crosses_bridge_in_30_seconds
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_km_per_hr : ℕ)
  (total_distance : ℕ)
  (speed_m_per_s : ℚ)
  (time_seconds : ℚ) 
  (h1 : length_train = 155)
  (h2 : length_bridge = 220)
  (h3 : speed_km_per_hr = 45)
  (h4 : total_distance = length_train + length_bridge)
  (h5 : speed_m_per_s = (speed_km_per_hr * 1000) / 3600)
  (h6 : time_seconds = total_distance / speed_m_per_s) :
  time_seconds = 30 :=
sorry

end train_crosses_bridge_in_30_seconds_l1153_115392


namespace gain_percent_calculation_l1153_115362

variable (CP SP : ℝ)
variable (gain gain_percent : ℝ)

theorem gain_percent_calculation
  (h₁ : CP = 900) 
  (h₂ : SP = 1180)
  (h₃ : gain = SP - CP)
  (h₄ : gain_percent = (gain / CP) * 100) :
  gain_percent = 31.11 := by
sorry

end gain_percent_calculation_l1153_115362


namespace range_of_m_l1153_115397

theorem range_of_m {x1 x2 y1 y2 m : ℝ} 
  (h1 : x1 > x2) 
  (h2 : y1 > y2) 
  (ha : y1 = (m - 3) * x1 - 4) 
  (hb : y2 = (m - 3) * x2 - 4) : 
  m > 3 :=
sorry

end range_of_m_l1153_115397


namespace division_expression_l1153_115380

theorem division_expression :
  (240 : ℚ) / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end division_expression_l1153_115380


namespace price_decrease_percentage_l1153_115357

variables (P Q : ℝ)
variables (Q' R R' : ℝ)

-- Condition: the number sold increased by 60%
def quantity_increase_condition : Prop :=
  Q' = Q * (1 + 0.60)

-- Condition: the total revenue increased by 28.000000000000025%
def revenue_increase_condition : Prop :=
  R' = R * (1 + 0.28000000000000025)

-- Definition: the original revenue R
def original_revenue : Prop :=
  R = P * Q

-- The new price P' after decreasing by x%
variables (P' : ℝ) (x : ℝ)
def new_price_condition : Prop :=
  P' = P * (1 - x / 100)

-- The new revenue R'
def new_revenue : Prop :=
  R' = P' * Q'

-- The proof problem
theorem price_decrease_percentage (P Q Q' R R' P' x : ℝ)
  (h1 : quantity_increase_condition Q Q')
  (h2 : revenue_increase_condition R R')
  (h3 : original_revenue P Q R)
  (h4 : new_price_condition P P' x)
  (h5 : new_revenue P' Q' R') :
  x = 20 :=
sorry

end price_decrease_percentage_l1153_115357


namespace total_students_class_is_63_l1153_115325

def num_tables : ℕ := 6
def students_per_table : ℕ := 3
def girls_bathroom : ℕ := 4
def times_canteen : ℕ := 4
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def germany_students : ℕ := 2
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 1

def total_students_in_class : ℕ :=
  (num_tables * students_per_table) +
  girls_bathroom +
  (times_canteen * girls_bathroom) +
  (group1_students + group2_students + group3_students) +
  (germany_students + france_students + norway_students + italy_students)

theorem total_students_class_is_63 : total_students_in_class = 63 :=
  by
    sorry

end total_students_class_is_63_l1153_115325


namespace september_first_2021_was_wednesday_l1153_115347

-- Defining the main theorem based on the conditions and the question
theorem september_first_2021_was_wednesday
  (doubledCapitalOnWeekdays : ∀ day : Nat, day = 0 % 7 → True)
  (sevenFiftyPercOnWeekends : ∀ day : Nat, day = 5 % 7 → True)
  (millionaireOnLastDayOfYear: ∀ day : Nat, day = 364 % 7 → True)
  : 1 % 7 = 3 % 7 := 
sorry

end september_first_2021_was_wednesday_l1153_115347
