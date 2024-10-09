import Mathlib

namespace line_param_func_l1537_153772

theorem line_param_func (t : ℝ) : 
    ∃ f : ℝ → ℝ, (∀ t, (20 * t - 14) = 2 * (f t) - 30) ∧ (f t = 10 * t + 8) := by
  sorry

end line_param_func_l1537_153772


namespace chocolate_truffles_sold_l1537_153720

def fudge_sold_pounds : ℕ := 20
def price_per_pound_fudge : ℝ := 2.50
def price_per_truffle : ℝ := 1.50
def pretzels_sold_dozen : ℕ := 3
def price_per_pretzel : ℝ := 2.00
def total_revenue : ℝ := 212.00

theorem chocolate_truffles_sold (dozens_of_truffles_sold : ℕ) :
  let fudge_revenue := (fudge_sold_pounds : ℝ) * price_per_pound_fudge
  let pretzels_revenue := (pretzels_sold_dozen : ℝ) * 12 * price_per_pretzel
  let truffles_revenue := total_revenue - fudge_revenue - pretzels_revenue
  let num_truffles_sold := truffles_revenue / price_per_truffle
  let dozens_of_truffles_sold := num_truffles_sold / 12
  dozens_of_truffles_sold = 5 :=
by
  sorry

end chocolate_truffles_sold_l1537_153720


namespace tangent_line_circle_sol_l1537_153732

theorem tangent_line_circle_sol (r : ℝ) (h_pos : r > 0)
  (h_tangent : ∀ x y : ℝ, x^2 + y^2 = 2 * r → x + 2 * y = r) : r = 10 := 
sorry

end tangent_line_circle_sol_l1537_153732


namespace necessary_sufficient_condition_l1537_153778

theorem necessary_sufficient_condition (A B C : ℝ)
    (h : ∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) :
    |A - B + C| ≤ 2 * Real.sqrt (A * C) := 
by sorry

end necessary_sufficient_condition_l1537_153778


namespace juniors_to_freshmen_ratio_l1537_153735

variable (f s j : ℕ)

def participated_freshmen := 3 * f / 7
def participated_sophomores := 5 * s / 7
def participated_juniors := j / 2

-- The statement
theorem juniors_to_freshmen_ratio
    (h1 : participated_freshmen = participated_sophomores)
    (h2 : participated_freshmen = participated_juniors) :
    j = 6 * f / 7 ∧ f = 7 * j / 6 :=
by
  sorry

end juniors_to_freshmen_ratio_l1537_153735


namespace PropositionA_PropositionD_l1537_153716

-- Proposition A: a > 1 is a sufficient but not necessary condition for 1/a < 1.
theorem PropositionA (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by sorry

-- PropositionD: a ≠ 0 is a necessary but not sufficient condition for ab ≠ 0.
theorem PropositionD (a b : ℝ) (h : a ≠ 0) : a * b ≠ 0 :=
by sorry
 
end PropositionA_PropositionD_l1537_153716


namespace johns_remaining_money_l1537_153763

theorem johns_remaining_money (H1 : ∃ (n : ℕ), n = 5376) (H2 : 5376 = 5 * 8^3 + 3 * 8^2 + 7 * 8^1 + 6) :
  (2814 - 1350 = 1464) :=
by {
  sorry
}

end johns_remaining_money_l1537_153763


namespace train_speed_in_kmph_l1537_153737

variable (L V : ℝ) -- L is the length of the train in meters, and V is the speed of the train in m/s.

-- Conditions given in the problem
def crosses_platform_in_30_seconds : Prop := L + 200 = V * 30
def crosses_man_in_20_seconds : Prop := L = V * 20

-- Length of the platform
def platform_length : ℝ := 200

-- The proof problem: Prove the speed of the train is 72 km/h
theorem train_speed_in_kmph 
  (h1 : crosses_man_in_20_seconds L V) 
  (h2 : crosses_platform_in_30_seconds L V) : 
  V * 3.6 = 72 := 
by 
  sorry

end train_speed_in_kmph_l1537_153737


namespace bobby_initial_candy_l1537_153785

theorem bobby_initial_candy (initial_candy : ℕ) (remaining_candy : ℕ) (extra_candy : ℕ) (total_eaten : ℕ)
  (h_candy_initial : initial_candy = 36)
  (h_candy_remaining : remaining_candy = 4)
  (h_candy_extra : extra_candy = 15)
  (h_candy_total_eaten : total_eaten = initial_candy - remaining_candy) :
  total_eaten - extra_candy = 17 :=
by
  sorry

end bobby_initial_candy_l1537_153785


namespace min_sum_x1_x2_x3_x4_l1537_153724

variables (x1 x2 x3 x4 : ℝ)

theorem min_sum_x1_x2_x3_x4 : 
  (x1 + x2 ≥ 12) → 
  (x1 + x3 ≥ 13) → 
  (x1 + x4 ≥ 14) → 
  (x3 + x4 ≥ 22) → 
  (x2 + x3 ≥ 23) → 
  (x2 + x4 ≥ 24) → 
  (x1 + x2 + x3 + x4 = 37) := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_sum_x1_x2_x3_x4_l1537_153724


namespace number_of_divisors_M_l1537_153711

def M : ℕ := 2^5 * 3^4 * 5^2 * 7^3 * 11^1

theorem number_of_divisors_M : (M.factors.prod.divisors.card = 720) :=
sorry

end number_of_divisors_M_l1537_153711


namespace double_square_area_l1537_153708

theorem double_square_area (a k : ℝ) (h : (k * a) ^ 2 = 2 * a ^ 2) : k = Real.sqrt 2 := 
by 
  -- Our goal is to prove that k = sqrt(2)
  sorry

end double_square_area_l1537_153708


namespace problem_1163_prime_and_16424_composite_l1537_153712

theorem problem_1163_prime_and_16424_composite :
  let x := 1910 * 10000 + 1112
  let a := 1163
  let b := 16424
  x = a * b →
  Prime a ∧ ¬ Prime b :=
by
  intros h
  sorry

end problem_1163_prime_and_16424_composite_l1537_153712


namespace one_thirds_of_nine_halfs_l1537_153789

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l1537_153789


namespace mouse_shortest_path_on_cube_l1537_153717

noncomputable def shortest_path_length (edge_length : ℝ) : ℝ :=
  2 * edge_length * Real.sqrt 2

theorem mouse_shortest_path_on_cube :
  shortest_path_length 2 = 4 * Real.sqrt 2 :=
by
  sorry

end mouse_shortest_path_on_cube_l1537_153717


namespace max_digit_sum_l1537_153762

-- Define the condition for the hours and minutes digits
def is_valid_hour (h : ℕ) := 0 ≤ h ∧ h < 24
def is_valid_minute (m : ℕ) := 0 ≤ m ∧ m < 60

-- Define the function to calculate the sum of the digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Main statement: Prove that the maximum sum of the digits in the display is 24
theorem max_digit_sum : ∃ h m: ℕ, is_valid_hour h ∧ is_valid_minute m ∧ 
  sum_of_digits h + sum_of_digits m = 24 :=
sorry

end max_digit_sum_l1537_153762


namespace largest_possible_b_l1537_153782

theorem largest_possible_b (b : ℚ) (h : (3 * b + 7) * (b - 2) = 9 * b) : b ≤ 2 :=
sorry

end largest_possible_b_l1537_153782


namespace triangle_cosine_sum_l1537_153729

theorem triangle_cosine_sum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hsum : A + B + C = π) : 
  (Real.cos A + Real.cos B + Real.cos C > 1) :=
sorry

end triangle_cosine_sum_l1537_153729


namespace complex_square_sum_eq_zero_l1537_153748

theorem complex_square_sum_eq_zero (i : ℂ) (h : i^2 = -1) : (1 + i)^2 + (1 - i)^2 = 0 :=
sorry

end complex_square_sum_eq_zero_l1537_153748


namespace n_four_minus_n_squared_l1537_153775

theorem n_four_minus_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
by 
  sorry

end n_four_minus_n_squared_l1537_153775


namespace grade_students_difference_condition_l1537_153780

variables (G1 G2 G5 : ℕ)

theorem grade_students_difference_condition (h : G1 + G2 = G2 + G5 + 30) : G1 - G5 = 30 :=
sorry

end grade_students_difference_condition_l1537_153780


namespace cistern_fill_time_l1537_153770

variable (A_rate : ℚ) (B_rate : ℚ) (C_rate : ℚ)
variable (total_rate : ℚ := A_rate + C_rate - B_rate)

theorem cistern_fill_time (hA : A_rate = 1/7) (hB : B_rate = 1/9) (hC : C_rate = 1/12) :
  (1/total_rate) = 252/29 :=
by
  rw [hA, hB, hC]
  sorry

end cistern_fill_time_l1537_153770


namespace arithmetic_sequence_terms_l1537_153744

theorem arithmetic_sequence_terms
  (a : ℕ → ℝ)
  (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 20)
  (h2 : a (n-2) + a (n-1) + a n = 130)
  (h3 : (n * (a 1 + a n)) / 2 = 200) :
  n = 8 := 
sorry

end arithmetic_sequence_terms_l1537_153744


namespace part1_part2_l1537_153730

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem part1 : A ∩ B = {3, 5} := by
  sorry

theorem part2 : (U \ A) ∪ B = {3, 4, 5, 6} := by
  sorry

end part1_part2_l1537_153730


namespace collin_savings_l1537_153756

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end collin_savings_l1537_153756


namespace point_A_symmetric_to_B_about_l_l1537_153768

variables {A B : ℝ × ℝ} {l : ℝ → ℝ → Prop}

-- define point B
def point_B := (1, 2)

-- define the line equation x + y + 3 = 0 as a property
def line_l (x y : ℝ) := x + y + 3 = 0

-- define that A is symmetric to B about line l
def symmetric_about (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  (∀ x y : ℝ, l x y → ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = -(x + y)))
  ∧ ((A.2 - B.2) / (A.1 - B.1) * -1 = -1)

theorem point_A_symmetric_to_B_about_l :
  A = (-5, -4) →
  symmetric_about A B line_l →
  A = (-5, -4) := by
  intros _ sym
  sorry

end point_A_symmetric_to_B_about_l_l1537_153768


namespace Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l1537_153746

def cost_supermarket_A (x : ℝ) : ℝ :=
  200 + 0.8 * (x - 200)

def cost_supermarket_B (x : ℝ) : ℝ :=
  100 + 0.85 * (x - 100)

theorem Li_Minghui_should_go_to_supermarket_B_for_300_yuan :
  cost_supermarket_B 300 < cost_supermarket_A 300 := by
  sorry

theorem cost_equal_for_500_yuan :
  cost_supermarket_A 500 = cost_supermarket_B 500 := by
  sorry

end Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l1537_153746


namespace gecko_insects_eaten_l1537_153795

theorem gecko_insects_eaten
    (G : ℕ)  -- Number of insects each gecko eats
    (H1 : 5 * G + 3 * (2 * G) = 66) :  -- Total insects eaten condition
    G = 6 :=  -- Expected number of insects each gecko eats
by
  sorry

end gecko_insects_eaten_l1537_153795


namespace division_dividend_l1537_153742

/-- In a division sum, the quotient is 40, the divisor is 72, and the remainder is 64. We need to prove that the dividend is 2944. -/
theorem division_dividend : 
  let Q := 40
  let D := 72
  let R := 64
  (D * Q + R = 2944) :=
by
  sorry

end division_dividend_l1537_153742


namespace simple_interest_calculation_l1537_153776

theorem simple_interest_calculation (P R T : ℝ) (H₁ : P = 8925) (H₂ : R = 9) (H₃ : T = 5) : 
  P * R * T / 100 = 4016.25 :=
by
  sorry

end simple_interest_calculation_l1537_153776


namespace find_function_l1537_153710

theorem find_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x y, (f x * f y - f (x * y)) / 4 = 2 * x + 2 * y + a) : a = -3 ∧ ∀ x, f x = x + 1 :=
by
  sorry

end find_function_l1537_153710


namespace line_intersects_parabola_at_one_point_l1537_153725

theorem line_intersects_parabola_at_one_point (k : ℝ) :
    (∃ y : ℝ, x = -3 * y^2 - 4 * y + 7) ↔ (x = k) := by
  sorry

end line_intersects_parabola_at_one_point_l1537_153725


namespace pencils_left_l1537_153794

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l1537_153794


namespace tangent_line_is_x_minus_y_eq_zero_l1537_153733

theorem tangent_line_is_x_minus_y_eq_zero : 
  ∀ (f : ℝ → ℝ) (x y : ℝ), 
  f x = x^3 - 2 * x → 
  (x, y) = (1, 1) → 
  (∃ (m : ℝ), m = 3 * (1:ℝ)^2 - 2 ∧ (y - 1) = m * (x - 1)) → 
  x - y = 0 :=
by
  intros f x y h_func h_point h_tangent
  sorry

end tangent_line_is_x_minus_y_eq_zero_l1537_153733


namespace amelia_painted_faces_l1537_153721

def faces_of_cuboid : ℕ := 6
def number_of_cuboids : ℕ := 6

theorem amelia_painted_faces : faces_of_cuboid * number_of_cuboids = 36 :=
by {
  sorry
}

end amelia_painted_faces_l1537_153721


namespace max_value_of_n_l1537_153743

theorem max_value_of_n : 
  ∃ n : ℕ, 
    (∀ m : ℕ, m ≤ n → (2 / 3)^(m - 1) * (1 / 3) ≥ 1 / 60) 
      ∧ 
    (∀ k : ℕ, k > n → (2 / 3)^(k - 1) * (1 / 3) < 1 / 60) 
      ∧ 
    n = 8 :=
by
  sorry

end max_value_of_n_l1537_153743


namespace symmetric_about_line_periodic_function_l1537_153718

section
variable {α : Type*} [LinearOrderedField α]

-- First proof problem
theorem symmetric_about_line (f : α → α) (a : α) (h : ∀ x, f (a + x) = f (a - x)) : 
  ∀ x, f (2 * a - x) = f x :=
sorry

-- Second proof problem
theorem periodic_function (f : α → α) (a b : α) (ha : a ≠ b)
  (hsymm_a : ∀ x, f (2 * a - x) = f x)
  (hsymm_b : ∀ x, f (2 * b - x) = f x) : 
  ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry
end

end symmetric_about_line_periodic_function_l1537_153718


namespace ratio_diagonals_of_squares_l1537_153788

variable (d₁ d₂ : ℝ)

theorem ratio_diagonals_of_squares (h : ∃ k : ℝ, d₂ = k * d₁) (h₁ : 1 < k ∧ k < 9) : 
  (∃ k : ℝ, 4 * (d₂ / Real.sqrt 2) = k * 4 * (d₁ / Real.sqrt 2)) → k = 5 := by
  sorry

end ratio_diagonals_of_squares_l1537_153788


namespace savings_percentage_l1537_153750

variable (I : ℝ) -- First year's income
variable (S : ℝ) -- Amount saved in the first year

-- Conditions
axiom condition1 (h1 : S = 0.05 * I) : Prop
axiom condition2 (h2 : S + 0.05 * I = 2 * S) : Prop
axiom condition3 (h3 : (I - S) + 1.10 * (I - S) = 2 * (I - S)) : Prop

-- Theorem that proves the man saved 5% of his income in the first year
theorem savings_percentage : S = 0.05 * I :=
by
  sorry -- Proof goes here

end savings_percentage_l1537_153750


namespace question_equals_answer_l1537_153704

def heartsuit (a b : ℤ) : ℤ := |a + b|

theorem question_equals_answer : heartsuit (-3) (heartsuit 5 (-8)) = 0 := 
by
  sorry

end question_equals_answer_l1537_153704


namespace min_sum_product_l1537_153767

theorem min_sum_product (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 9/n = 1) :
  m * n = 48 :=
sorry

end min_sum_product_l1537_153767


namespace least_N_bench_sections_l1537_153740

-- First, define the problem conditions
def bench_capacity_adult (N : ℕ) : ℕ := 7 * N
def bench_capacity_child (N : ℕ) : ℕ := 11 * N

-- Define the problem statement to be proven
theorem least_N_bench_sections :
  ∃ N : ℕ, (N > 0) ∧ (bench_capacity_adult N = bench_capacity_child N → N = 77) :=
sorry

end least_N_bench_sections_l1537_153740


namespace james_car_purchase_l1537_153752

/-- 
James sold his $20,000 car for 80% of its value, 
then bought a $30,000 sticker price car, 
and he was out of pocket $11,000. 
James bought the new car for 90% of its value. 
-/
theorem james_car_purchase (V_1 P_1 V_2 O P : ℝ)
  (hV1 : V_1 = 20000)
  (hP1 : P_1 = 80)
  (hV2 : V_2 = 30000)
  (hO : O = 11000)
  (hSaleOld : (P_1 / 100) * V_1 = 16000)
  (hDiff : 16000 + O = 27000)
  (hPurchase : (P / 100) * V_2 = 27000) :
  P = 90 := 
sorry

end james_car_purchase_l1537_153752


namespace land_per_person_l1537_153700

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end land_per_person_l1537_153700


namespace value_of_x_minus_y_l1537_153799

theorem value_of_x_minus_y (x y : ℝ) 
  (h1 : |x| = 2) 
  (h2 : y^2 = 9) 
  (h3 : x + y < 0) : 
  x - y = 1 ∨ x - y = 5 := 
by 
  sorry

end value_of_x_minus_y_l1537_153799


namespace value_of_w_over_y_l1537_153769

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3.25) : w / y = 0.75 :=
sorry

end value_of_w_over_y_l1537_153769


namespace range_of_m_minimum_value_l1537_153734

theorem range_of_m (m n : ℝ) (h : 2 * m - n = 3) (ineq : |m| + |n + 3| ≥ 9) : 
  m ≤ -3 ∨ m ≥ 3 := 
sorry

theorem minimum_value (m n : ℝ) (h : 2 * m - n = 3) : 
  ∃ c, c = 3 ∧ c = |(5 / 3) * m - (1 / 3) * n| + |(1 / 3) * m - (2 / 3) * n| := 
sorry

end range_of_m_minimum_value_l1537_153734


namespace piece_length_is_111_l1537_153706

-- Define the conditions
axiom condition1 : ∃ (x : ℤ), 9 * x ≤ 1000
axiom condition2 : ∃ (x : ℤ), 9 * x ≤ 1100

-- State the problem: Prove that the length of each piece is 111 centimeters
theorem piece_length_is_111 (x : ℤ) (h1 : 9 * x ≤ 1000) (h2 : 9 * x ≤ 1100) : x = 111 :=
by sorry

end piece_length_is_111_l1537_153706


namespace mixing_solutions_l1537_153705

theorem mixing_solutions (Vx : ℝ) :
  (0.10 * Vx + 0.30 * 900 = 0.25 * (Vx + 900)) ↔ Vx = 300 := by
  sorry

end mixing_solutions_l1537_153705


namespace white_squares_in_20th_row_l1537_153715

def num_squares_in_row (n : ℕ) : ℕ :=
  3 * n

def num_white_squares (n : ℕ) : ℕ :=
  (num_squares_in_row n - 2) / 2

theorem white_squares_in_20th_row: num_white_squares 20 = 30 := by
  -- Proof skipped
  sorry

end white_squares_in_20th_row_l1537_153715


namespace max_sin_a_l1537_153709

theorem max_sin_a (a b c : ℝ) (h1 : Real.cos a = Real.tan b) 
                                  (h2 : Real.cos b = Real.tan c) 
                                  (h3 : Real.cos c = Real.tan a) : 
  Real.sin a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := 
by
  sorry

end max_sin_a_l1537_153709


namespace sum_of_three_consecutive_odd_integers_l1537_153707

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end sum_of_three_consecutive_odd_integers_l1537_153707


namespace no_prime_divisible_by_56_l1537_153753

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end no_prime_divisible_by_56_l1537_153753


namespace solution_set_abs_inequality_l1537_153774

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end solution_set_abs_inequality_l1537_153774


namespace taxi_ride_total_cost_l1537_153723

theorem taxi_ride_total_cost :
  let base_fee := 1.50
  let cost_per_mile := 0.25
  let distance1 := 5
  let distance2 := 8
  let distance3 := 3
  let cost1 := base_fee + distance1 * cost_per_mile
  let cost2 := base_fee + distance2 * cost_per_mile
  let cost3 := base_fee + distance3 * cost_per_mile
  cost1 + cost2 + cost3 = 8.50 := sorry

end taxi_ride_total_cost_l1537_153723


namespace digit_theta_l1537_153736

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end digit_theta_l1537_153736


namespace sum_min_max_z_l1537_153703

theorem sum_min_max_z (x y : ℝ) 
  (h1 : x - y - 2 ≥ 0) 
  (h2 : x - 5 ≤ 0) 
  (h3 : y + 2 ≥ 0) :
  ∃ (z_min z_max : ℝ), z_min = 2 ∧ z_max = 34 ∧ z_min + z_max = 36 :=
by
  sorry

end sum_min_max_z_l1537_153703


namespace intercept_condition_slope_condition_l1537_153783

theorem intercept_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 - 2 * m - 3) * -3 + (2 * m^2 + m - 1) * 0 + (-2 * m + 6) = 0 → 
  m = -5 / 3 := 
  sorry

theorem slope_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 + 2 * m - 4) = 0 → 
  m = 4 / 3 := 
  sorry

end intercept_condition_slope_condition_l1537_153783


namespace equation_of_perpendicular_line_l1537_153787

theorem equation_of_perpendicular_line (x y : ℝ) (l1 : 2*x - 3*y + 4 = 0) (pt : x = -2 ∧ y = -3) :
  3*(-2) + 2*(-3) + 12 = 0 := by
  sorry

end equation_of_perpendicular_line_l1537_153787


namespace monomial_2023rd_l1537_153761

theorem monomial_2023rd : ∀ (x : ℝ), (2 * 2023 + 1) / 2023 * x ^ 2023 = (4047 / 2023) * x ^ 2023 :=
by
  intro x
  sorry

end monomial_2023rd_l1537_153761


namespace evaluate_g_l1537_153751

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_l1537_153751


namespace vertices_of_regular_hexagonal_pyramid_l1537_153765

-- Define a structure for a regular hexagonal pyramid
structure RegularHexagonalPyramid where
  baseVertices : Nat
  apexVertices : Nat

-- Define a specific regular hexagonal pyramid with given conditions
def regularHexagonalPyramid : RegularHexagonalPyramid :=
  { baseVertices := 6, apexVertices := 1 }

-- The theorem stating the number of vertices of the pyramid
theorem vertices_of_regular_hexagonal_pyramid : regularHexagonalPyramid.baseVertices + regularHexagonalPyramid.apexVertices = 7 := 
  by
  sorry

end vertices_of_regular_hexagonal_pyramid_l1537_153765


namespace simplify_expression_l1537_153759

variable (p : ℤ)

-- Defining the given expression
def initial_expression : ℤ := ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9)

-- Statement asserting the simplification
theorem simplify_expression : initial_expression p = 13 * p - 30 := 
sorry

end simplify_expression_l1537_153759


namespace tysons_speed_in_ocean_l1537_153796

theorem tysons_speed_in_ocean
  (speed_lake : ℕ) (half_races_lake : ℕ) (total_races : ℕ) (race_distance : ℕ) (total_time : ℕ)
  (speed_lake_val : speed_lake = 3)
  (half_races_lake_val : half_races_lake = 5)
  (total_races_val : total_races = 10)
  (race_distance_val : race_distance = 3)
  (total_time_val : total_time = 11) :
  ∃ (speed_ocean : ℚ), speed_ocean = 2.5 := 
by
  sorry

end tysons_speed_in_ocean_l1537_153796


namespace Billy_has_10_fish_l1537_153745

def Billy_has_fish (Bobby Sarah Tony Billy : ℕ) : Prop :=
  Bobby = 2 * Sarah ∧
  Sarah = Tony + 5 ∧
  Tony = 3 * Billy ∧
  Bobby + Sarah + Tony + Billy = 145

theorem Billy_has_10_fish : ∃ (Billy : ℕ), Billy_has_fish (2 * (3 * Billy + 5)) (3 * Billy + 5) (3 * Billy) Billy ∧ Billy = 10 :=
by
  sorry

end Billy_has_10_fish_l1537_153745


namespace triangle_probability_is_correct_l1537_153747

-- Define the total number of figures
def total_figures : ℕ := 8

-- Define the number of triangles among the figures
def number_of_triangles : ℕ := 3

-- Define the probability function for choosing a triangle
def probability_of_triangle : ℚ := number_of_triangles / total_figures

-- The theorem to be proved
theorem triangle_probability_is_correct :
  probability_of_triangle = 3 / 8 := by
  sorry

end triangle_probability_is_correct_l1537_153747


namespace smallest_whole_number_l1537_153755

theorem smallest_whole_number :
  ∃ a : ℕ, a % 3 = 2 ∧ a % 5 = 3 ∧ a % 7 = 3 ∧ ∀ b : ℕ, (b % 3 = 2 ∧ b % 5 = 3 ∧ b % 7 = 3 → a ≤ b) :=
sorry

end smallest_whole_number_l1537_153755


namespace joy_tape_deficit_l1537_153771

noncomputable def tape_needed_field (width length : ℕ) : ℕ :=
2 * (length + width)

noncomputable def tape_needed_trees (num_trees circumference : ℕ) : ℕ :=
num_trees * circumference

def tape_total_needed (tape_field tape_trees : ℕ) : ℕ :=
tape_field + tape_trees

theorem joy_tape_deficit (tape_has : ℕ) (tape_field tape_trees: ℕ) : ℤ :=
tape_has - (tape_field + tape_trees)

example : joy_tape_deficit 180 (tape_needed_field 35 80) (tape_needed_trees 3 5) = -65 := by
sorry

end joy_tape_deficit_l1537_153771


namespace sum_of_roots_equals_18_l1537_153713

-- Define the conditions
variable (f : ℝ → ℝ)
variable (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x))
variable (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0))

-- The theorem statement
theorem sum_of_roots_equals_18 (f : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x)) 
  (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0)) :
  ∀ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0) → xs.sum id = 18 :=
by
  sorry

end sum_of_roots_equals_18_l1537_153713


namespace result_l1537_153758

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l1537_153758


namespace find_range_m_l1537_153741

-- Definitions of the conditions
def p (m : ℝ) : Prop := ∃ x y : ℝ, (x + y - m = 0) ∧ ((x - 1)^2 + y^2 = 1)
def q (m : ℝ) : Prop := ∃ x : ℝ, (x^2 - x + m - 4 = 0) ∧ x ≠ 0 ∧ ∀ y : ℝ, (y^2 - y + m - 4 = 0) → x * y < 0

theorem find_range_m (m : ℝ) : (p m ∨ q m) ∧ ¬p m → (m ≤ 1 - Real.sqrt 2 ∨ 1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by
  sorry

end find_range_m_l1537_153741


namespace number_of_classes_l1537_153728

-- Define the conditions
def first_term : ℕ := 27
def common_diff : ℤ := -2
def total_students : ℕ := 115

-- Define and prove the main statement
theorem number_of_classes : ∃ n : ℕ, n > 0 ∧ (first_term + (n - 1) * common_diff) * n / 2 = total_students ∧ n = 5 :=
by
  sorry

end number_of_classes_l1537_153728


namespace no_solution_equation_l1537_153701

theorem no_solution_equation (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) = (x - m) / (x - 8) → false) ↔ m = 7 :=
by
  sorry

end no_solution_equation_l1537_153701


namespace workers_together_time_l1537_153738

theorem workers_together_time (A_time B_time : ℝ) (hA : A_time = 8) (hB : B_time = 10) :
  let rateA := 1 / A_time
  let rateB := 1 / B_time
  let combined_rate := rateA + rateB
  combined_rate * (40 / 9) = 1 :=
by 
  sorry

end workers_together_time_l1537_153738


namespace ribbon_per_box_l1537_153781

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l1537_153781


namespace problem_l1537_153722

variable {R : Type} [Field R]

def f1 (a b c d : R) : R := a + b + c + d
def f2 (a b c d : R) : R := (1 / a) + (1 / b) + (1 / c) + (1 / d)
def f3 (a b c d : R) : R := (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) + (1 / (1 - d))

theorem problem (a b c d : R) (h1 : f1 a b c d = 2) (h2 : f2 a b c d = 2) : f3 a b c d = 2 :=
by sorry

end problem_l1537_153722


namespace complete_the_square_b_26_l1537_153749

theorem complete_the_square_b_26 :
  ∃ (a b : ℝ), (∀ x : ℝ, x^2 + 10 * x - 1 = 0 ↔ (x + a)^2 = b) ∧ b = 26 :=
sorry

end complete_the_square_b_26_l1537_153749


namespace number_mod_conditions_l1537_153791

theorem number_mod_conditions :
  ∃ N, (N % 10 = 9) ∧ (N % 9 = 8) ∧ (N % 8 = 7) ∧ (N % 7 = 6) ∧
       (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧
       N = 2519 :=
by
  sorry

end number_mod_conditions_l1537_153791


namespace monthly_rent_l1537_153727

-- Definition
def total_amount_saved := 2225
def extra_amount_needed := 775
def deposit := 500

-- Total amount required
def total_amount_required := total_amount_saved + extra_amount_needed
def total_rent_plus_deposit (R : ℝ) := 2 * R + deposit

-- The statement to prove
theorem monthly_rent (R : ℝ) : total_rent_plus_deposit R = total_amount_required → R = 1250 :=
by
  intros h
  exact sorry -- Proof is omitted.

end monthly_rent_l1537_153727


namespace teresa_age_at_michiko_birth_l1537_153731

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end teresa_age_at_michiko_birth_l1537_153731


namespace compute_c_minus_d_squared_eq_0_l1537_153773

-- Defining conditions
def multiples_of_n_under_m (n m : ℕ) : ℕ :=
  (m - 1) / n

-- Defining the specific values
def c : ℕ := multiples_of_n_under_m 9 60
def d : ℕ := multiples_of_n_under_m 9 60  -- Since every multiple of 9 is a multiple of 3

theorem compute_c_minus_d_squared_eq_0 : (c - d) ^ 2 = 0 := by
  sorry

end compute_c_minus_d_squared_eq_0_l1537_153773


namespace at_least_one_greater_than_one_l1537_153739

open Classical

variable (x y : ℝ)

theorem at_least_one_greater_than_one (h : x + y > 2) : x > 1 ∨ y > 1 :=
by
  sorry

end at_least_one_greater_than_one_l1537_153739


namespace square_area_4900_l1537_153777

/-- If one side of a square is increased by 3.5 times and the other side is decreased by 30 cm, resulting in a rectangle that has twice the area of the square, then the area of the square is 4900 square centimeters. -/
theorem square_area_4900 (x : ℝ) (h1 : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 :=
sorry

end square_area_4900_l1537_153777


namespace double_acute_angle_is_less_than_180_degrees_l1537_153798

theorem double_acute_angle_is_less_than_180_degrees (alpha : ℝ) (h : 0 < alpha ∧ alpha < 90) : 2 * alpha < 180 :=
sorry

end double_acute_angle_is_less_than_180_degrees_l1537_153798


namespace students_number_l1537_153790

theorem students_number (C P S : ℕ) : C = 315 ∧ 121 + C = P * S -> S = 4 := by
  sorry

end students_number_l1537_153790


namespace power_expression_simplify_l1537_153760

theorem power_expression_simplify :
  (1 / (-5^2)^3) * (-5)^8 * Real.sqrt 5 = 5^(5/2) :=
by
  sorry

end power_expression_simplify_l1537_153760


namespace brad_read_more_books_l1537_153757

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end brad_read_more_books_l1537_153757


namespace find_clubs_l1537_153784

theorem find_clubs (S D H C : ℕ) (h1 : S + D + H + C = 13)
  (h2 : S + C = 7) 
  (h3 : D + H = 6) 
  (h4 : D = 2 * S) 
  (h5 : H = 2 * D) 
  : C = 6 :=
by
  sorry

end find_clubs_l1537_153784


namespace exercise_l1537_153779

-- Define the given expression.
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- Define the general form expression.
def g (x h k : ℝ) (a : ℝ) := a * (x - h)^2 + k

-- Prove that a + h + k = 6 when expressing f(x) in the form a(x-h)^2 + k.
theorem exercise : ∃ a h k : ℝ, (∀ x : ℝ, f x = g x h k a) ∧ (a + h + k = 6) :=
by
  sorry

end exercise_l1537_153779


namespace triangle_inequalities_l1537_153702

theorem triangle_inequalities (a b c h_a h_b h_c : ℝ) (ha_eq : h_a = b * Real.sin (arc_c)) (hb_eq : h_b = a * Real.sin (arc_c)) (hc_eq : h_c = a * Real.sin (arc_b)) (h : a > b) (h2 : b > c) :
  (a + h_a > b + h_b) ∧ (b + h_b > c + h_c) :=
by
  sorry

end triangle_inequalities_l1537_153702


namespace germination_relative_frequency_l1537_153786

theorem germination_relative_frequency {n m : ℕ} (h₁ : n = 1000) (h₂ : m = 1000 - 90) : 
  (m : ℝ) / (n : ℝ) = 0.91 := by
  sorry

end germination_relative_frequency_l1537_153786


namespace total_books_l1537_153766

/-- Define Tim’s and Sam’s number of books. -/
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52

/-- Prove that together they have 96 books. -/
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l1537_153766


namespace trigonometric_identity_l1537_153793

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (3 * Real.cos x - Real.sin x) = 3 := 
by
  sorry

end trigonometric_identity_l1537_153793


namespace kevin_speed_first_half_l1537_153754

-- Let's define the conditions as variables and constants
variable (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ)
variable (time_20mph : ℝ) (time_8mph : ℝ) (distance_first_half : ℕ)
variable (speed_first_half : ℝ)

-- Conditions from the problem
def conditions (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ) : Prop :=
  total_distance = 17 ∧ 
  distance_20mph = 20 * 1 / 2 ∧
  distance_8mph = 8 * 1 / 4

-- Proof objective based on conditions and correct answer
theorem kevin_speed_first_half (
  h : conditions total_distance distance_20mph distance_8mph
) : speed_first_half = 10 := by
  sorry

end kevin_speed_first_half_l1537_153754


namespace sufficient_but_not_necessary_condition_l1537_153719

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition :
  (a > 2 → a^2 > 2 * a)
  ∧ (¬(a^2 > 2 * a → a > 2)) := by
  sorry

end sufficient_but_not_necessary_condition_l1537_153719


namespace greatest_perimeter_l1537_153714

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l1537_153714


namespace combined_area_l1537_153726

noncomputable def diagonal : ℝ := 12 * Real.sqrt 2

noncomputable def side_of_square (d : ℝ) : ℝ := d / Real.sqrt 2

noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

theorem combined_area (d : ℝ) (h : d = diagonal) :
  let s := side_of_square d
  let area_sq := area_of_square s
  let r := radius_of_circle d
  let area_circ := area_of_circle r
  area_sq + area_circ = 144 + 72 * Real.pi :=
by
  sorry

end combined_area_l1537_153726


namespace how_long_to_grow_more_l1537_153797

def current_length : ℕ := 14
def length_to_donate : ℕ := 23
def desired_length_after_donation : ℕ := 12

theorem how_long_to_grow_more : 
  (desired_length_after_donation + length_to_donate - current_length) = 21 := 
by
  -- Leave the proof part for later
  sorry

end how_long_to_grow_more_l1537_153797


namespace sqrt_12_bounds_l1537_153764

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end sqrt_12_bounds_l1537_153764


namespace arithmetic_sequence_m_value_l1537_153792

theorem arithmetic_sequence_m_value (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) 
  (h_seq : ∀ n : ℕ, S n = (n + 1) / 2 * (2 * a₁ + n * d)) :
  m = 5 :=
by
  sorry

end arithmetic_sequence_m_value_l1537_153792
