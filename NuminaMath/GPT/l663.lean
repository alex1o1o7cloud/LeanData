import Mathlib

namespace factorize_polynomial_l663_66373

theorem factorize_polynomial (x : ℝ) :
  x^4 + 2 * x^3 - 9 * x^2 - 2 * x + 8 = (x + 4) * (x - 2) * (x + 1) * (x - 1) :=
sorry

end factorize_polynomial_l663_66373


namespace range_of_a_l663_66391

theorem range_of_a (x y : ℝ) (a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 4 = 2 * x * y) :
  x^2 + 2 * x * y + y^2 - a * x - a * y + 1 ≥ 0 ↔ a ≤ 17 / 4 := 
sorry

end range_of_a_l663_66391


namespace smallest_n_l663_66385

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l663_66385


namespace area_of_square_II_l663_66379

theorem area_of_square_II {a b : ℝ} (h : a > b) (d : ℝ) (h1 : d = a - b)
    (A1_A : ℝ) (h2 : A1_A = (a - b)^2 / 2) (A2_A : ℝ) (h3 : A2_A = 3 * A1_A) :
  A2_A = 3 * (a - b)^2 / 2 := by
  sorry

end area_of_square_II_l663_66379


namespace length_decrease_by_33_percent_l663_66346

theorem length_decrease_by_33_percent (L W L_new : ℝ) 
  (h1 : L * W = L_new * 1.5 * W) : 
  L_new = (2 / 3) * L ∧ ((1 - (2 / 3)) * 100 = 33.33) := 
by
  sorry

end length_decrease_by_33_percent_l663_66346


namespace range_of_a_l663_66335

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + a > 0

def proposition_q (a : ℝ) : Prop :=
  a - 1 > 1

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬ (proposition_p a ∧ proposition_q a) ↔ 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l663_66335


namespace maria_needs_flour_l663_66300

-- Definitions from conditions
def cups_of_flour_per_cookie (c : ℕ) (f : ℚ) : ℚ := f / c

def total_cups_of_flour (cps_per_cookie : ℚ) (num_cookies : ℕ) : ℚ := cps_per_cookie * num_cookies

-- Given values
def cookies_20 := 20
def flour_3 := 3
def cookies_100 := 100

theorem maria_needs_flour :
  total_cups_of_flour (cups_of_flour_per_cookie cookies_20 flour_3) cookies_100 = 15 :=
by
  sorry -- Proof is omitted

end maria_needs_flour_l663_66300


namespace distance_walked_hazel_l663_66388

theorem distance_walked_hazel (x : ℝ) (h : x + 2 * x = 6) : x = 2 :=
sorry

end distance_walked_hazel_l663_66388


namespace find_k_l663_66310

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : 15 * k^4 < 120) : k = 1 := 
  sorry

end find_k_l663_66310


namespace carrot_cakes_in_february_l663_66359

theorem carrot_cakes_in_february :
  (∃ (cakes_in_oct : ℕ) (cakes_in_nov : ℕ) (cakes_in_dec : ℕ) (cakes_in_jan : ℕ) (monthly_increase : ℕ),
      cakes_in_oct = 19 ∧
      cakes_in_nov = 21 ∧
      cakes_in_dec = 23 ∧
      cakes_in_jan = 25 ∧
      monthly_increase = 2 ∧
      cakes_in_february = cakes_in_jan + monthly_increase) →
  cakes_in_february = 27 :=
  sorry

end carrot_cakes_in_february_l663_66359


namespace min_value_a_plus_b_plus_c_l663_66317

-- Define the main conditions
variables {a b c : ℝ}
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
variables (h_eq : a^2 + 2*a*b + 4*b*c + 2*c*a = 16)

-- Define the theorem
theorem min_value_a_plus_b_plus_c : 
  (∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (a^2 + 2*a*b + 4*b*c + 2*c*a = 16) → a + b + c ≥ 4) :=
sorry

end min_value_a_plus_b_plus_c_l663_66317


namespace inequality_one_inequality_two_l663_66395

-- Definitions of the three positive real numbers and their sum of reciprocals squared is equal to 1
variables {a b c : ℝ}
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1)

-- First proof that (1/a + 1/b + 1/c) <= sqrt(3)
theorem inequality_one (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (1 / a) + (1 / b) + (1 / c) ≤ Real.sqrt 3 :=
sorry

-- Second proof that (a^2/b^4) + (b^2/c^4) + (c^2/a^4) >= 1
theorem inequality_two (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (a^2 / b^4) + (b^2 / c^4) + (c^2 / a^4) ≥ 1 :=
sorry

end inequality_one_inequality_two_l663_66395


namespace xiaoming_problem_l663_66374

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end xiaoming_problem_l663_66374


namespace blanket_cost_l663_66376

theorem blanket_cost (x : ℝ) 
    (h₁ : 200 + 750 + 2 * x = 1350) 
    (h₂ : 2 + 5 + 2 = 9) 
    (h₃ : (200 + 750 + 2 * x) / 9 = 150) : 
    x = 200 :=
by
    have h_total : 200 + 750 + 2 * x = 1350 := h₁
    have h_avg : (200 + 750 + 2 * x) / 9 = 150 := h₃
    sorry

end blanket_cost_l663_66376


namespace length_B1C1_l663_66382

variable (AC BC : ℝ) (A1B1 : ℝ) (T : ℝ)

/-- Given a right triangle ABC with legs AC = 3 and BC = 4, and transformations
  of points to A1, B1, and C1 where A1B1 = 1 and angle B1 = 90 degrees,
  prove that the length of B1C1 is 12. -/
theorem length_B1C1 (h1 : AC = 3) (h2 : BC = 4) (h3 : A1B1 = 1) 
  (TABC : T = 6) (right_triangle_ABC : true) (right_triangle_A1B1C1 : true) : 
  B1C1 = 12 := 
sorry

end length_B1C1_l663_66382


namespace find_c_l663_66390

theorem find_c (x c : ℤ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 7 = 1) : c = -8 :=
sorry

end find_c_l663_66390


namespace spending_record_l663_66306

-- Definitions based on conditions
def deposit_record (x : ℤ) : ℤ := x
def spend_record (x : ℤ) : ℤ := -x

-- Theorem statement
theorem spending_record (x : ℤ) (hx : x = 500) : spend_record x = -500 := by
  sorry

end spending_record_l663_66306


namespace intersection_range_l663_66312

noncomputable def f (a : ℝ) (x : ℝ) := a * x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := f a x - g x

theorem intersection_range (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = g x1 ∧ f a x2 = g x2) ↔
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end intersection_range_l663_66312


namespace part_a_part_b_l663_66380

noncomputable def arithmetic_progression_a (a₁: ℕ) (r: ℕ) : ℕ :=
  a₁ + 3 * r

theorem part_a (a₁: ℕ) (r: ℕ) (h_a₁ : a₁ = 2) (h_r : r = 3) : arithmetic_progression_a a₁ r = 11 := 
by 
  sorry

noncomputable def arithmetic_progression_formula (d: ℕ) (r: ℕ) (n: ℕ) : ℕ :=
  d + (n - 1) * r

theorem part_b (a3: ℕ) (a6: ℕ) (a9: ℕ) (a4_plus_a7_plus_a10: ℕ) (a_sum: ℕ) (h_a3 : a3 = 3) (h_a6 : a6 = 6) (h_a9 : a9 = 9) 
  (h_a4a7a10 : a4_plus_a7_plus_a10 = 207) (h_asum : a_sum = 553) 
  (h_eqn1: 3 * a3 + a6 * 2 = 207) (h_eqn2: a_sum = 553): 
  arithmetic_progression_formula 9 10 11 = 109 := 
by 
  sorry

end part_a_part_b_l663_66380


namespace quadratic_expression_value_l663_66305

-- Given conditions
variables (a : ℝ) (h : 2 * a^2 + 3 * a - 2022 = 0)

-- Prove the main statement
theorem quadratic_expression_value :
  2 - 6 * a - 4 * a^2 = -4042 :=
sorry

end quadratic_expression_value_l663_66305


namespace three_f_l663_66355

noncomputable def f (x : ℝ) : ℝ := sorry

theorem three_f (x : ℝ) (hx : 0 < x) (h : ∀ y > 0, f (3 * y) = 5 / (3 + y)) :
  3 * f x = 45 / (9 + x) :=
by
  sorry

end three_f_l663_66355


namespace sum_first_third_numbers_l663_66303

theorem sum_first_third_numbers (A B C : ℕ)
    (h1 : A + B + C = 98)
    (h2 : A * 3 = B * 2)
    (h3 : B * 8 = C * 5)
    (h4 : B = 30) :
    A + C = 68 :=
by
-- Data is sufficient to conclude that A + C = 68
sorry

end sum_first_third_numbers_l663_66303


namespace simplify_expression_l663_66384

theorem simplify_expression (w : ℝ) : 3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 :=
by
  sorry

end simplify_expression_l663_66384


namespace solve_modulo_problem_l663_66318

theorem solve_modulo_problem (n : ℤ) :
  0 ≤ n ∧ n < 19 ∧ 38574 % 19 = n % 19 → n = 4 := by
  sorry

end solve_modulo_problem_l663_66318


namespace arithmetic_sequence_d_range_l663_66313

theorem arithmetic_sequence_d_range (d : ℝ) :
  (10 + 4 * d > 0) ∧ (10 + 5 * d < 0) ↔ (-5/2 < d) ∧ (d < -2) :=
by
  sorry

end arithmetic_sequence_d_range_l663_66313


namespace arithmetic_sequence_problem_l663_66308

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
variable (h_S_n : ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2)

theorem arithmetic_sequence_problem
  (h1 : S_n 5 = 2 * a_n 5)
  (h2 : a_n 3 = -4) :
  a_n 9 = -22 := sorry

end arithmetic_sequence_problem_l663_66308


namespace regular_polygon_sides_l663_66339

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l663_66339


namespace sin_cos_eq_sqrt2_l663_66357

theorem sin_cos_eq_sqrt2 (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 2 * Real.pi) (h2 : Real.sin x - Real.cos x = Real.sqrt 2) :
  x = (3 * Real.pi) / 4 :=
sorry

end sin_cos_eq_sqrt2_l663_66357


namespace incorrect_option_l663_66301

noncomputable def f : ℝ → ℝ := sorry
def is_odd (g : ℝ → ℝ) := ∀ x, g (-(2 * x + 1)) = -g (2 * x + 1)
def is_even (g : ℝ → ℝ) := ∀ x, g (x + 2) = g (-x + 2)

theorem incorrect_option (h₁ : is_odd f) (h₂ : is_even f) (h₃ : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = 3 - x) :
  ¬ (∀ x, f x = f (-x - 2)) :=
by
  sorry

end incorrect_option_l663_66301


namespace problem1_problem2_l663_66309

theorem problem1 :
  (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (60 * Real.pi / 180) - abs (1 - Real.sqrt 3) = 3 :=
by 
  sorry

theorem problem2 (x : ℝ) :
  (2 / (x + 1) + 1 = x / (x - 1)) → x = 3 :=
by 
  sorry

end problem1_problem2_l663_66309


namespace must_be_negative_when_x_is_negative_l663_66347

open Real

theorem must_be_negative_when_x_is_negative (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := 
by
  sorry

end must_be_negative_when_x_is_negative_l663_66347


namespace joe_eats_different_fruits_l663_66351

noncomputable def joe_probability : ℚ :=
  let single_fruit_prob := (1 / 3) ^ 4
  let all_same_fruit_prob := 3 * single_fruit_prob
  let at_least_two_diff_fruits_prob := 1 - all_same_fruit_prob
  at_least_two_diff_fruits_prob

theorem joe_eats_different_fruits :
  joe_probability = 26 / 27 :=
by
  -- The proof is omitted for this task
  sorry

end joe_eats_different_fruits_l663_66351


namespace tan_3theta_eq_2_11_sin_3theta_eq_22_125_l663_66394

variable {θ : ℝ}

-- First, stating the condition \(\tan \theta = 2\)
axiom tan_theta_eq_2 : Real.tan θ = 2

-- Stating the proof problem for \(\tan 3\theta = \frac{2}{11}\)
theorem tan_3theta_eq_2_11 : Real.tan (3 * θ) = 2 / 11 :=
by 
  sorry

-- Stating the proof problem for \(\sin 3\theta = \frac{22}{125}\)
theorem sin_3theta_eq_22_125 : Real.sin (3 * θ) = 22 / 125 :=
by 
  sorry

end tan_3theta_eq_2_11_sin_3theta_eq_22_125_l663_66394


namespace problem_statement_l663_66377

variables {a c b d : ℝ} {x y q z : ℕ}

-- Given conditions:
def condition1 (a c : ℝ) (x q : ℕ) : Prop := a^(x + 1) = c^(q + 2)
def condition2 (a c : ℝ) (y z : ℕ) : Prop := c^(y + 3) = a^(z+ 4)

-- Goal statement
theorem problem_statement (a c : ℝ) (x y q z : ℕ) (h1 : condition1 a c x q) (h2 : condition2 a c y z) :
  (q + 2) * (z + 4) = (y + 3) * (x + 1) :=
sorry

end problem_statement_l663_66377


namespace bobby_toy_cars_in_5_years_l663_66367

noncomputable def toy_cars_after_n_years (initial_cars : ℕ) (percentage_increase : ℝ) (n : ℕ) : ℝ :=
initial_cars * (1 + percentage_increase)^n

theorem bobby_toy_cars_in_5_years :
  toy_cars_after_n_years 25 0.75 5 = 410 := by
  -- 25 * (1 + 0.75)^5 
  -- = 25 * (1.75)^5 
  -- ≈ 410.302734375
  -- After rounding
  sorry

end bobby_toy_cars_in_5_years_l663_66367


namespace radius_ratio_in_right_triangle_l663_66348

theorem radius_ratio_in_right_triangle (PQ QR PR PS SR : ℝ)
  (h₁ : PQ = 5) (h₂ : QR = 12) (h₃ : PR = 13)
  (h₄ : PS + SR = PR) (h₅ : PS / SR = 5 / 8)
  (r_p r_q : ℝ)
  (hr_p : r_p = (1 / 2 * PQ * PS / 3) / ((PQ + PS / 3 + PS) / 3))
  (hr_q : r_q = (1 / 2 * QR * SR) / ((PS / 3 + QR + SR) / 3)) :
  r_p / r_q = 175 / 576 :=
sorry

end radius_ratio_in_right_triangle_l663_66348


namespace closed_pipe_length_l663_66332

def speed_of_sound : ℝ := 333
def fundamental_frequency : ℝ := 440

theorem closed_pipe_length :
  ∃ l : ℝ, l = 0.189 ∧ fundamental_frequency = speed_of_sound / (4 * l) :=
by
  sorry

end closed_pipe_length_l663_66332


namespace ratio_of_areas_is_16_l663_66383

-- Definitions and conditions
variables (a b : ℝ)

-- Given condition: Perimeter of the larger square is 4 times the perimeter of the smaller square
def perimeter_relation (ha : a = 4 * b) : Prop := a = 4 * b

-- Theorem to prove: Ratio of the area of the larger square to the area of the smaller square is 16
theorem ratio_of_areas_is_16 (ha : a = 4 * b) : (a^2 / b^2) = 16 :=
by
  sorry

end ratio_of_areas_is_16_l663_66383


namespace total_bricks_proof_l663_66370

-- Define the initial conditions
def initial_courses := 3
def bricks_per_course := 400
def additional_courses := 2

-- Compute the number of bricks removed from the last course
def bricks_removed_from_last_course (bricks_per_course: ℕ) : ℕ :=
  bricks_per_course / 2

-- Calculate the total number of bricks
def total_bricks (initial_courses : ℕ) (bricks_per_course : ℕ) (additional_courses : ℕ) (bricks_removed : ℕ) : ℕ :=
  (initial_courses + additional_courses) * bricks_per_course - bricks_removed

-- Given values and the proof problem
theorem total_bricks_proof :
  total_bricks initial_courses bricks_per_course additional_courses (bricks_removed_from_last_course bricks_per_course) = 1800 :=
by
  sorry

end total_bricks_proof_l663_66370


namespace kevin_total_distance_l663_66337

def v1 : ℝ := 10
def t1 : ℝ := 0.5
def v2 : ℝ := 20
def t2 : ℝ := 0.5
def v3 : ℝ := 8
def t3 : ℝ := 0.25

theorem kevin_total_distance : v1 * t1 + v2 * t2 + v3 * t3 = 17 := by
  sorry

end kevin_total_distance_l663_66337


namespace range_of_a_l663_66330

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * x + (a - 1) * Real.log x

theorem range_of_a (a : ℝ) (h1 : 1 < a) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 > x2 → f a x1 - f a x2 > x2 - x1) ↔ (1 < a ∧ a ≤ 5) :=
by
  -- The proof is omitted
  sorry

end range_of_a_l663_66330


namespace store_earnings_correct_l663_66320

theorem store_earnings_correct :
  let graphics_cards_sold : ℕ := 10
  let hard_drives_sold : ℕ := 14
  let cpus_sold : ℕ := 8
  let ram_pairs_sold : ℕ := 4
  let graphics_card_price : ℝ := 600
  let hard_drive_price : ℝ := 80
  let cpu_price : ℝ := 200
  let ram_pair_price : ℝ := 60
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := 
by
  sorry

end store_earnings_correct_l663_66320


namespace tony_running_speed_l663_66326

theorem tony_running_speed :
  (∀ R : ℝ, (4 / 2 * 60) + 2 * ((4 / R) * 60) = 168 → R = 10) :=
sorry

end tony_running_speed_l663_66326


namespace find_all_k_l663_66333

theorem find_all_k :
  ∃ (k : ℝ), ∃ (v : ℝ × ℝ), v ≠ 0 ∧ (∃ (v₀ v₁ : ℝ), v = (v₀, v₁) 
  ∧ (3 * v₀ + 6 * v₁) = k * v₀ ∧ (4 * v₀ + 3 * v₁) = k * v₁) 
  ↔ k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6 :=
by
  -- here goes the proof
  sorry

end find_all_k_l663_66333


namespace Kimberley_collected_10_pounds_l663_66316

variable (K H E total : ℝ)

theorem Kimberley_collected_10_pounds (h_total : total = 35) (h_Houston : H = 12) (h_Ela : E = 13) :
    K + H + E = total → K = 10 :=
by
  intros h_sum
  rw [h_Houston, h_Ela] at h_sum
  linarith

end Kimberley_collected_10_pounds_l663_66316


namespace minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l663_66349

noncomputable def f (x : Real) : Real := Real.cos (2*x) - 2*Real.sin x + 1

theorem minimum_value_of_f : ∃ x : Real, f x = -2 := sorry

theorem symmetry_of_f : ∀ x : Real, f x = f (π - x) := sorry

theorem monotonic_decreasing_f : ∀ x y : Real, 0 < x ∧ x < y ∧ y < π / 2 → f y < f x := sorry

end minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l663_66349


namespace product_of_series_l663_66324

theorem product_of_series :
  (1 - 1/2^2) * (1 - 1/3^2) * (1 - 1/4^2) * (1 - 1/5^2) * (1 - 1/6^2) *
  (1 - 1/7^2) * (1 - 1/8^2) * (1 - 1/9^2) * (1 - 1/10^2) = 11 / 20 :=
by 
  sorry

end product_of_series_l663_66324


namespace marie_finishes_ninth_task_at_730PM_l663_66398

noncomputable def start_time : ℕ := 8 * 60 -- 8:00 AM in minutes
noncomputable def end_time_task_3 : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
noncomputable def total_tasks : ℕ := 9
noncomputable def tasks_done_by_1130AM : ℕ := 3
noncomputable def end_time_task_9 : ℕ := 19 * 60 + 30 -- 7:30 PM in minutes

theorem marie_finishes_ninth_task_at_730PM
    (h1 : start_time = 480) -- 8:00 AM
    (h2 : end_time_task_3 = 690) -- 11:30 AM
    (h3 : total_tasks = 9)
    (h4 : tasks_done_by_1130AM = 3)
    (h5 : end_time_task_9 = 1170) -- 7:30 PM
    : end_time_task_9 = start_time + ((end_time_task_3 - start_time) / tasks_done_by_1130AM) * total_tasks :=
sorry

end marie_finishes_ninth_task_at_730PM_l663_66398


namespace find_hansol_weight_l663_66321

variable (H : ℕ)

theorem find_hansol_weight (h : H + (H + 4) = 88) : H = 42 :=
by
  sorry

end find_hansol_weight_l663_66321


namespace parakeets_per_cage_is_2_l663_66319

variables (cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

def number_of_parakeets_each_cage : ℕ :=
  (total_birds - cages * parrots_per_cage) / cages

theorem parakeets_per_cage_is_2
  (hcages : cages = 4)
  (hparrots_per_cage : parrots_per_cage = 8)
  (htotal_birds : total_birds = 40) :
  number_of_parakeets_each_cage cages parrots_per_cage total_birds = 2 := 
by
  sorry

end parakeets_per_cage_is_2_l663_66319


namespace insurance_compensation_zero_l663_66361

noncomputable def insured_amount : ℝ := 500000
noncomputable def deductible : ℝ := 0.01
noncomputable def actual_damage : ℝ := 4000

theorem insurance_compensation_zero :
  actual_damage < insured_amount * deductible → 0 = 0 := by
sorry

end insurance_compensation_zero_l663_66361


namespace total_people_counted_l663_66389

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end total_people_counted_l663_66389


namespace not_square_n5_plus_7_l663_66341

theorem not_square_n5_plus_7 (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, k^2 = n^5 + 7 := 
by
  sorry

end not_square_n5_plus_7_l663_66341


namespace joan_mortgage_payoff_l663_66345

/-- Joan's mortgage problem statement. -/
theorem joan_mortgage_payoff (a r : ℕ) (total : ℕ) (n : ℕ) : a = 100 → r = 3 → total = 12100 → 
    total = a * (1 - r^n) / (1 - r) → n = 5 :=
by intros ha hr htotal hgeom; sorry

end joan_mortgage_payoff_l663_66345


namespace sum_of_s_and_t_eq_neg11_l663_66314

theorem sum_of_s_and_t_eq_neg11 (s t : ℝ) 
  (h1 : ∀ x, x = 3 → x^2 + s * x + t = 0)
  (h2 : ∀ x, x = -4 → x^2 + s * x + t = 0) :
  s + t = -11 :=
sorry

end sum_of_s_and_t_eq_neg11_l663_66314


namespace ac_bc_ratios_l663_66354

theorem ac_bc_ratios (A B C : ℝ) (m n : ℕ) (h : AC / BC = m / n) : 
  if m ≠ n then
    ((AC / AB = m / (m+n) ∧ BC / AB = n / (m+n)) ∨ 
     (AC / AB = m / (n-m) ∧ BC / AB = n / (n-m)))
  else 
    (AC / AB = 1 / 2 ∧ BC / AB = 1 / 2) := sorry

end ac_bc_ratios_l663_66354


namespace marge_funds_l663_66369

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end marge_funds_l663_66369


namespace inverse_propositions_l663_66331

-- Given conditions
lemma right_angles_equal : ∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90) :=
sorry

lemma equal_angles_right : ∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2) :=
sorry

-- Theorem to be proven
theorem inverse_propositions :
  (∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90)) ↔
  (∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2)) :=
sorry

end inverse_propositions_l663_66331


namespace meridian_students_l663_66362

theorem meridian_students
  (eighth_to_seventh_ratio : Nat → Nat → Prop)
  (seventh_to_sixth_ratio : Nat → Nat → Prop)
  (r1 : ∀ a b, eighth_to_seventh_ratio a b ↔ 7 * b = 4 * a)
  (r2 : ∀ b c, seventh_to_sixth_ratio b c ↔ 10 * c = 9 * b) :
  ∃ a b c, eighth_to_seventh_ratio a b ∧ seventh_to_sixth_ratio b c ∧ a + b + c = 73 :=
by
  sorry

end meridian_students_l663_66362


namespace value_of_x_plus_inv_x_l663_66397

theorem value_of_x_plus_inv_x (x : ℝ) (h : x + (1 / x) = v) (hr : x^2 + (1 / x)^2 = 23) : v = 5 :=
sorry

end value_of_x_plus_inv_x_l663_66397


namespace min_difference_xue_jie_ti_neng_li_l663_66329

theorem min_difference_xue_jie_ti_neng_li : 
  ∀ (shu hsue jie ti neng li zhan shi : ℕ), 
  shu = 8 ∧ hsue = 1 ∧ jie = 4 ∧ ti = 3 ∧ neng = 9 ∧ li = 5 ∧ zhan = 7 ∧ shi = 2 →
  (shu * 1000 + hsue * 100 + jie * 10 + ti) = 1842 →
  (neng * 10 + li) = 95 →
  1842 - 95 = 1747 := 
by
  intros shu hsue jie ti neng li zhan shi h_digits h_xue_jie_ti h_neng_li
  sorry

end min_difference_xue_jie_ti_neng_li_l663_66329


namespace cube_edge_length_l663_66307

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 96) : ∃ (edge_length : ℝ), edge_length = 4 := 
by 
  sorry

end cube_edge_length_l663_66307


namespace elise_hospital_distance_l663_66387

noncomputable def distance_to_hospital (total_fare: ℝ) (base_price: ℝ) (toll_price: ℝ) 
(tip_percent: ℝ) (cost_per_mile: ℝ) (increase_percent: ℝ) (toll_count: ℕ) : ℝ :=
let base_and_tolls := base_price + (toll_price * toll_count)
let fare_before_tip := total_fare / (1 + tip_percent)
let distance_fare := fare_before_tip - base_and_tolls
let original_travel_fare := distance_fare / (1 + increase_percent)
original_travel_fare / cost_per_mile

theorem elise_hospital_distance : distance_to_hospital 34.34 3 2 0.15 4 0.20 3 = 5 := 
sorry

end elise_hospital_distance_l663_66387


namespace simplify_expression_l663_66304

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (2 * x + 10) - (x + 3) * (3 * x - 2) = 3 * x^2 + 15 * x - 34 := 
by
  sorry

end simplify_expression_l663_66304


namespace geometric_sequence_inequality_l663_66363

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)

-- Conditions
def geometric_sequence (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ) : Prop :=
  a₂ = a₁ * q ∧
  a₃ = a₁ * q^2 ∧
  a₄ = a₁ * q^3 ∧
  a₅ = a₁ * q^4 ∧
  a₆ = a₁ * q^5 ∧
  a₇ = a₁ * q^6 ∧
  a₈ = a₁ * q^7

theorem geometric_sequence_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)
  (h_seq : geometric_sequence a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q)
  (h_a₁_pos : 0 < a₁)
  (h_q_ne_1 : q ≠ 1) :
  a₁ + a₈ > a₄ + a₅ :=
by 
-- Proof omitted
sorry

end geometric_sequence_inequality_l663_66363


namespace sum_of_max_values_l663_66364

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_max_values : (f π + f (3 * π)) = (Real.exp π + Real.exp (3 * π)) := 
by sorry

end sum_of_max_values_l663_66364


namespace fifth_equation_pattern_l663_66336

theorem fifth_equation_pattern :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by sorry

end fifth_equation_pattern_l663_66336


namespace solution_pairs_l663_66322

theorem solution_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
    (h_coprime: Nat.gcd (2 * a - 1) (2 * b + 1) = 1) 
    (h_divides : (a + b) ∣ (4 * a * b + 1)) :
    ∃ n : ℕ, a = n ∧ b = n + 1 :=
by
  -- statement
  sorry

end solution_pairs_l663_66322


namespace quadratic_inequality_solution_non_empty_l663_66353

theorem quadratic_inequality_solution_non_empty
  (a b c : ℝ) (h : a < 0) :
  ∃ x : ℝ, ax^2 + bx + c < 0 :=
sorry

end quadratic_inequality_solution_non_empty_l663_66353


namespace dance_team_recruitment_l663_66325

theorem dance_team_recruitment 
  (total_students choir_students track_field_students dance_students : ℕ)
  (h1 : total_students = 100)
  (h2 : choir_students = 2 * track_field_students)
  (h3 : dance_students = choir_students + 10)
  (h4 : total_students = track_field_students + choir_students + dance_students) : 
  dance_students = 46 :=
by {
  -- The proof goes here, but it is not required as per instructions
  sorry
}

end dance_team_recruitment_l663_66325


namespace janet_additional_money_needed_l663_66328

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def advance_months : ℕ := 2
def deposit : ℕ := 500

theorem janet_additional_money_needed :
  (advance_months * monthly_rent + deposit - janet_savings) = 775 :=
by
  sorry

end janet_additional_money_needed_l663_66328


namespace kim_branch_marking_l663_66378

theorem kim_branch_marking (L : ℝ) (rem_frac : ℝ) (third_piece : ℝ) (F : ℝ) :
  L = 3 ∧ rem_frac = 0.6 ∧ third_piece = 1 ∧ L * rem_frac = 1.8 → F = 1 / 15 :=
by sorry

end kim_branch_marking_l663_66378


namespace repetend_of_5_over_13_l663_66371

theorem repetend_of_5_over_13 : (∃ r : ℕ, r = 384615) :=
by
  let d := 13
  let n := 5
  let r := 384615
  -- Definitions to use:
  -- d is denominator 13
  -- n is numerator 5
  -- r is the repetend 384615
  sorry

end repetend_of_5_over_13_l663_66371


namespace part_I_solution_set_part_II_min_value_l663_66344

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Prove the solution set of the inequality f(x) ≤ 6 for x ≥ -1 is -1 ≤ x ≤ 4
theorem part_I_solution_set (x : ℝ) (h1 : x ≥ -1) : f x ≤ 6 ↔ (-1 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Define the condition for the minimum value of f(x)
def min_f := 4

-- Prove the minimum value of 2a + b under the given constraints
theorem part_II_min_value (a b : ℝ) (h2 : a > 0 ∧ b > 0) (h3 : 8 * a * b = a + 2 * b) : 2 * a + b ≥ 9 / 8 :=
by
  sorry

end part_I_solution_set_part_II_min_value_l663_66344


namespace determine_m_l663_66311

theorem determine_m 
  (f : ℝ → ℝ) 
  (m : ℕ) 
  (h_nat: 0 < m) 
  (h_f: ∀ x, f x = x ^ (m^2 - 2 * m - 3)) 
  (h_no_intersection: ∀ x, f x ≠ 0) 
  (h_symmetric_origin : ∀ x, f (-x) = -f x) : 
  m = 2 :=
by
  sorry

end determine_m_l663_66311


namespace find_remainder_l663_66358

def dividend : ℝ := 17698
def divisor : ℝ := 198.69662921348313
def quotient : ℝ := 89
def remainder : ℝ := 14

theorem find_remainder :
  dividend = (divisor * quotient) + remainder :=
by 
  -- Placeholder proof
  sorry

end find_remainder_l663_66358


namespace box_length_l663_66334

theorem box_length :
  ∃ (length : ℝ), 
  let box_height := 8
  let box_width := 10
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let num_blocks := 40
  let box_volume := box_height * box_width * length
  let block_volume := block_height * block_width * block_length
  num_blocks * block_volume = box_volume ∧ length = 12 := by
  sorry

end box_length_l663_66334


namespace final_score_is_83_l663_66386

def running_score : ℕ := 90
def running_weight : ℚ := 0.5

def fancy_jump_rope_score : ℕ := 80
def fancy_jump_rope_weight : ℚ := 0.3

def jump_rope_score : ℕ := 70
def jump_rope_weight : ℚ := 0.2

noncomputable def final_score : ℚ := 
  running_score * running_weight + 
  fancy_jump_rope_score * fancy_jump_rope_weight + 
  jump_rope_score * jump_rope_weight

theorem final_score_is_83 : final_score = 83 := 
  by
    sorry

end final_score_is_83_l663_66386


namespace algebraic_expression_value_l663_66327

theorem algebraic_expression_value (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
  (1 / (a^2 + 1) + 1 / (b^2 + 1)) = 1 :=
sorry

end algebraic_expression_value_l663_66327


namespace harly_initial_dogs_l663_66372

theorem harly_initial_dogs (x : ℝ) 
  (h1 : 0.40 * x + 0.60 * x + 5 = 53) : 
  x = 80 := 
by 
  sorry

end harly_initial_dogs_l663_66372


namespace blueberries_count_l663_66338

theorem blueberries_count
  (initial_apples : ℕ)
  (initial_oranges : ℕ)
  (initial_blueberries : ℕ)
  (apples_eaten : ℕ)
  (oranges_eaten : ℕ)
  (remaining_fruits : ℕ)
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 9)
  (h3 : apples_eaten = 1)
  (h4 : oranges_eaten = 1)
  (h5 : remaining_fruits = 26) :
  initial_blueberries = 5 := 
by
  sorry

end blueberries_count_l663_66338


namespace arithmeticSeqModulus_l663_66315

-- Define the arithmetic sequence
def arithmeticSeqSum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

-- The main theorem to prove
theorem arithmeticSeqModulus : arithmeticSeqSum 2 5 102 % 20 = 12 := by
  sorry

end arithmeticSeqModulus_l663_66315


namespace arithmetic_seq_sum_l663_66340

theorem arithmetic_seq_sum (a d : ℕ) (S : ℕ → ℕ) (n : ℕ) :
  S 6 = 36 →
  S 12 = 144 →
  S (6 * n) = 576 →
  (∀ m, S m = m * (2 * a + (m - 1) * d) / 2) →
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end arithmetic_seq_sum_l663_66340


namespace solve_x_l663_66393

theorem solve_x : ∃ x : ℝ, 2^(Real.log 5 / Real.log 2) = 3 * x + 4 ∧ x = 1 / 3 :=
by
  use 1 / 3
  sorry

end solve_x_l663_66393


namespace least_three_digit_with_product_l663_66396

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_product (n : ℕ) (p : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 * d2 * d3 = p

theorem least_three_digit_with_product (p : ℕ) : ∃ n : ℕ, is_three_digit n ∧ digits_product n p ∧ 
  ∀ m : ℕ, is_three_digit m ∧ digits_product m p → n ≤ m :=
by
  use 116
  sorry

end least_three_digit_with_product_l663_66396


namespace find_M_l663_66365

theorem find_M :
  ∃ (M : ℕ), 1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M ∧ M = 75 :=
by
  sorry

end find_M_l663_66365


namespace triangle_area_gt_half_l663_66323

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end triangle_area_gt_half_l663_66323


namespace geometric_sequence_vertex_property_l663_66360

theorem geometric_sequence_vertex_property (a b c d : ℝ) 
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r)
  (h_vertex : b = 1 ∧ c = 2) : a * d = b * c :=
by sorry

end geometric_sequence_vertex_property_l663_66360


namespace number_of_students_without_A_l663_66342

theorem number_of_students_without_A (total_students : ℕ) (A_chemistry : ℕ) (A_physics : ℕ) (A_both : ℕ) (h1 : total_students = 40)
    (h2 : A_chemistry = 10) (h3 : A_physics = 18) (h4 : A_both = 5) :
    total_students - (A_chemistry + A_physics - A_both) = 17 :=
by {
  sorry
}

end number_of_students_without_A_l663_66342


namespace rabbit_travel_time_l663_66368

theorem rabbit_travel_time :
  let distance := 2
  let speed := 5
  let hours_to_minutes := 60
  (distance / speed) * hours_to_minutes = 24 := by
sorry

end rabbit_travel_time_l663_66368


namespace great_dane_weight_l663_66375

def weight_problem (C P G : ℝ) : Prop :=
  (P = 3 * C) ∧ (G = 3 * P + 10) ∧ (C + P + G = 439)

theorem great_dane_weight : ∃ (C P G : ℝ), weight_problem C P G ∧ G = 307 :=
by
  sorry

end great_dane_weight_l663_66375


namespace trig_identity_proof_l663_66352

theorem trig_identity_proof :
  (Real.cos (10 * Real.pi / 180) * Real.sin (70 * Real.pi / 180) - Real.cos (80 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)) = (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_proof_l663_66352


namespace min_a2_b2_c2_l663_66350

theorem min_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 5 * c = 100) : 
  a^2 + b^2 + c^2 ≥ (5000 / 19) :=
by
  sorry

end min_a2_b2_c2_l663_66350


namespace team_plays_60_games_in_division_l663_66366

noncomputable def number_of_division_games (N M : ℕ) (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) : ℕ :=
  4 * N

theorem team_plays_60_games_in_division (N M : ℕ) 
  (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) 
  : number_of_division_games N M hNM hM h_total = 60 := 
sorry

end team_plays_60_games_in_division_l663_66366


namespace jerry_weighted_mean_l663_66356

noncomputable def weighted_mean (aunt uncle sister cousin friend1 friend2 friend3 friend4 friend5 : ℝ)
    (eur_to_usd gbp_to_usd cad_to_usd : ℝ) (family_weight friends_weight : ℝ) : ℝ :=
  let uncle_usd := uncle * eur_to_usd
  let friend3_usd := friend3 * eur_to_usd
  let friend4_usd := friend4 * gbp_to_usd
  let cousin_usd := cousin * cad_to_usd
  let family_sum := aunt + uncle_usd + sister + cousin_usd
  let friends_sum := friend1 + friend2 + friend3_usd + friend4_usd + friend5
  family_sum * family_weight + friends_sum * friends_weight

theorem jerry_weighted_mean : 
  weighted_mean 9.73 9.43 7.25 20.37 22.16 23.51 18.72 15.53 22.84 
               1.20 1.38 0.82 0.40 0.60 = 85.4442 := 
sorry

end jerry_weighted_mean_l663_66356


namespace corn_syrup_content_sport_formulation_l663_66343

def standard_ratio_flavoring : ℕ := 1
def standard_ratio_corn_syrup : ℕ := 12
def standard_ratio_water : ℕ := 30

def sport_ratio_flavoring_to_corn_syrup : ℕ := 3 * standard_ratio_flavoring
def sport_ratio_flavoring_to_water : ℕ := standard_ratio_flavoring / 2

def sport_ratio_flavoring : ℕ := 1
def sport_ratio_corn_syrup : ℕ := sport_ratio_flavoring * sport_ratio_flavoring_to_corn_syrup
def sport_ratio_water : ℕ := (sport_ratio_flavoring * standard_ratio_water) / 2

def water_content_sport_formulation : ℕ := 30

theorem corn_syrup_content_sport_formulation : 
  (sport_ratio_corn_syrup / sport_ratio_water) * water_content_sport_formulation = 2 :=
by
  sorry

end corn_syrup_content_sport_formulation_l663_66343


namespace find_acute_angle_x_l663_66399

def a_parallel_b (x : ℝ) : Prop :=
  let a := (Real.sin x, 3 / 4)
  let b := (1 / 3, 1 / 2 * Real.cos x)
  b.1 * a.2 = a.1 * b.2

theorem find_acute_angle_x (x : ℝ) (h : a_parallel_b x) : x = Real.pi / 4 :=
by
  sorry

end find_acute_angle_x_l663_66399


namespace find_ticket_cost_l663_66381

-- Define the initial amount Tony had
def initial_amount : ℕ := 20

-- Define the amount Tony paid for a hot dog
def hot_dog_cost : ℕ := 3

-- Define the amount Tony had after buying the ticket and the hot dog
def remaining_amount : ℕ := 9

-- Define the function to find the baseball ticket cost
def ticket_cost (t : ℕ) : Prop := initial_amount - t - hot_dog_cost = remaining_amount

-- The statement to prove
theorem find_ticket_cost : ∃ t : ℕ, ticket_cost t ∧ t = 8 := 
by 
  existsi 8
  unfold ticket_cost
  simp
  exact sorry

end find_ticket_cost_l663_66381


namespace race_prob_l663_66302

theorem race_prob :
  let pX := (1 : ℝ) / 8
  let pY := (1 : ℝ) / 12
  let pZ := (1 : ℝ) / 6
  pX + pY + pZ = (3 : ℝ) / 8 :=
by
  sorry

end race_prob_l663_66302


namespace find_number_of_girls_l663_66392

-- Definitions
variables (B G : ℕ)
variables (total children_holding_boys_hand children_holding_girls_hand : ℕ)
variables (children_counted_twice : ℕ)

-- Conditions
axiom cond1 : B + G = 40
axiom cond2 : children_holding_boys_hand = 22
axiom cond3 : children_holding_girls_hand = 30
axiom cond4 : total = 40

-- Goal
theorem find_number_of_girls (h : children_counted_twice = children_holding_boys_hand + children_holding_girls_hand - total) :
  G = 24 :=
sorry

end find_number_of_girls_l663_66392
