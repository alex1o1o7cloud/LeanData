import Mathlib

namespace NUMINAMATH_GPT_coefficient_x4_of_square_l1118_111860

theorem coefficient_x4_of_square (q : Polynomial ℝ) (hq : q = Polynomial.X^5 - 4 * Polynomial.X^2 + 3) :
  (Polynomial.coeff (q * q) 4 = 16) :=
by {
  sorry
}

end NUMINAMATH_GPT_coefficient_x4_of_square_l1118_111860


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l1118_111894

theorem arithmetic_sequence_value (a : ℕ → ℕ) (m : ℕ) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 4) 
  (h_a5 : a 5 = m) 
  (h_a7 : a 7 = 16) : 
  m = 10 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l1118_111894


namespace NUMINAMATH_GPT_intersection_A_B_l1118_111863

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1118_111863


namespace NUMINAMATH_GPT_total_money_made_l1118_111829

def dvd_price : ℕ := 240
def dvd_quantity : ℕ := 8
def washing_machine_price : ℕ := 898

theorem total_money_made : dvd_price * dvd_quantity + washing_machine_price = 240 * 8 + 898 :=
by
  sorry

end NUMINAMATH_GPT_total_money_made_l1118_111829


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1118_111808

-- Define a rhombus with one diagonal of 10 cm and a perimeter of 52 cm.
theorem rhombus_diagonal_length (d : ℝ) 
  (h1 : ∃ a b c : ℝ, a = 10 ∧ b = d ∧ c = 13) -- The diagonals and side of rhombus.
  (h2 : 52 = 4 * c) -- The perimeter condition.
  (h3 : c^2 = (d/2)^2 + (10/2)^2) -- The relationship from Pythagorean theorem.
  : d = 24 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1118_111808


namespace NUMINAMATH_GPT_find_y_l1118_111864

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ)
  (hx : x = 3 - 2 * t)
  (hy : y = 3 * t + 6)
  (hx_cond : x = -6) :
  y = 19.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1118_111864


namespace NUMINAMATH_GPT_slower_pump_time_l1118_111886

theorem slower_pump_time (R : ℝ) (hours : ℝ) (combined_rate : ℝ) (faster_rate_adj : ℝ) (time_both : ℝ) :
  (combined_rate = R * (1 + faster_rate_adj)) →
  (faster_rate_adj = 1.5) →
  (time_both = 5) →
  (combined_rate * time_both = 1) →
  (hours = 1 / R) →
  hours = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_slower_pump_time_l1118_111886


namespace NUMINAMATH_GPT_expression_divisible_512_l1118_111873

theorem expression_divisible_512 (n : ℤ) (h : n % 2 ≠ 0) : (n^12 - n^8 - n^4 + 1) % 512 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_expression_divisible_512_l1118_111873


namespace NUMINAMATH_GPT_maria_workday_ends_at_330_pm_l1118_111838

/-- 
Given:
1. Maria's workday is 8 hours long.
2. Her workday does not include her lunch break.
3. Maria starts work at 7:00 A.M.
4. She takes her lunch break at 11:30 A.M., lasting 30 minutes.
Prove that Maria's workday ends at 3:30 P.M.
-/
def maria_end_workday : Prop :=
  let start_time : Nat := 7 * 60 -- in minutes
  let lunch_start_time : Nat := 11 * 60 + 30 -- in minutes
  let lunch_duration : Nat := 30 -- in minutes
  let lunch_end_time : Nat := lunch_start_time + lunch_duration
  let total_work_minutes : Nat := 8 * 60
  let work_before_lunch : Nat := lunch_start_time - start_time
  let remaining_work : Nat := total_work_minutes - work_before_lunch
  let end_time : Nat := lunch_end_time + remaining_work
  end_time = 15 * 60 + 30

theorem maria_workday_ends_at_330_pm : maria_end_workday :=
  by
    sorry

end NUMINAMATH_GPT_maria_workday_ends_at_330_pm_l1118_111838


namespace NUMINAMATH_GPT_intersection_M_N_l1118_111876

noncomputable def set_M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
noncomputable def set_N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_M_N :
  (set_M ∩ set_N) = { x | -1 ≤ x ∧ x ≤ Real.sqrt 2 } := sorry

end NUMINAMATH_GPT_intersection_M_N_l1118_111876


namespace NUMINAMATH_GPT_triangle_area_l1118_111899

noncomputable def area_ABC (AB BC : ℝ) (angle_B : ℝ) : ℝ :=
  1/2 * AB * BC * Real.sin angle_B

theorem triangle_area
  (A B C : Type)
  (AB : ℝ) (A_eq : ℝ) (B_eq : ℝ)
  (h_AB : AB = 6)
  (h_A : A_eq = Real.pi / 6)
  (h_B : B_eq = 2 * Real.pi / 3) :
  area_ABC AB AB (2 * Real.pi / 3) = 9 * Real.sqrt 3 :=
by
  simp [area_ABC, h_AB, h_A, h_B]
  sorry

end NUMINAMATH_GPT_triangle_area_l1118_111899


namespace NUMINAMATH_GPT_quadrant_and_terminal_angle_l1118_111892

def alpha : ℝ := -1910 

noncomputable def normalize_angle (α : ℝ) : ℝ := 
  let β := α % 360
  if β < 0 then β + 360 else β

noncomputable def in_quadrant_3 (β : ℝ) : Prop :=
  180 ≤ β ∧ β < 270

noncomputable def equivalent_theta (α : ℝ) (θ : ℝ) : Prop :=
  (α % 360 = θ % 360) ∧ (-720 ≤ θ ∧ θ < 0)

theorem quadrant_and_terminal_angle :
  in_quadrant_3 (normalize_angle alpha) ∧ 
  (equivalent_theta alpha (-110) ∨ equivalent_theta alpha (-470)) :=
by 
  sorry

end NUMINAMATH_GPT_quadrant_and_terminal_angle_l1118_111892


namespace NUMINAMATH_GPT_words_on_each_page_l1118_111842

/-- Given a book with 150 pages, where each page has between 50 and 150 words, 
    and the total number of words in the book is congruent to 217 modulo 221, 
    prove that each page has 135 words. -/
theorem words_on_each_page (p : ℕ) (h1 : 50 ≤ p) (h2 : p ≤ 150) (h3 : 150 * p ≡ 217 [MOD 221]) : 
  p = 135 :=
by
  sorry

end NUMINAMATH_GPT_words_on_each_page_l1118_111842


namespace NUMINAMATH_GPT_Joey_age_digit_sum_l1118_111839

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end NUMINAMATH_GPT_Joey_age_digit_sum_l1118_111839


namespace NUMINAMATH_GPT_students_got_off_the_bus_l1118_111865

theorem students_got_off_the_bus
    (original_students : ℕ)
    (students_left : ℕ)
    (h_original : original_students = 10)
    (h_left : students_left = 7) :
    original_students - students_left = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_got_off_the_bus_l1118_111865


namespace NUMINAMATH_GPT_sum_of_square_areas_l1118_111844

theorem sum_of_square_areas (a b : ℝ)
  (h1 : a + b = 14)
  (h2 : a - b = 2) :
  a^2 + b^2 = 100 := by
  sorry

end NUMINAMATH_GPT_sum_of_square_areas_l1118_111844


namespace NUMINAMATH_GPT_function_is_constant_and_straight_line_l1118_111815

-- Define a function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition: The derivative of f is 0 everywhere
axiom derivative_zero_everywhere : ∀ x, deriv f x = 0

-- Conclusion: f is a constant function
theorem function_is_constant_and_straight_line : ∃ C : ℝ, ∀ x, f x = C := by
  sorry

end NUMINAMATH_GPT_function_is_constant_and_straight_line_l1118_111815


namespace NUMINAMATH_GPT_cos_double_angle_l1118_111811

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) : Real.cos (2 * α) = -1 / 3 := 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1118_111811


namespace NUMINAMATH_GPT_john_bathroom_uses_during_movie_and_intermissions_l1118_111818

-- Define the conditions
def uses_bathroom_interval := 50   -- John uses the bathroom every 50 minutes
def walking_time := 5              -- It takes him an additional 5 minutes to walk to and from the bathroom
def movie_length := 150            -- The movie length in minutes (2.5 hours)
def intermission_length := 15      -- Each intermission length in minutes
def intermission_count := 2        -- The number of intermissions

-- Derived condition
def effective_interval := uses_bathroom_interval + walking_time

-- Total movie time including intermissions
def total_movie_time := movie_length + (intermission_length * intermission_count)

-- Define the theorem to be proved
theorem john_bathroom_uses_during_movie_and_intermissions : 
  ∃ n : ℕ, n = 3 + 2 ∧ total_movie_time = 180 ∧ effective_interval = 55 :=
by
  sorry

end NUMINAMATH_GPT_john_bathroom_uses_during_movie_and_intermissions_l1118_111818


namespace NUMINAMATH_GPT_happy_number_part1_happy_number_part2_happy_number_part3_l1118_111871

section HappyEquations

def is_happy_eq (a b c : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, a ≠ 0 ∧ a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b ^ 2) / (4 * a)

def happy_to_each_other (a b c p q r : ℤ) : Prop :=
  let Fa : ℚ := happy_number a b c
  let Fb : ℚ := happy_number p q r
  |r * Fa - c * Fb| = 0

theorem happy_number_part1 :
  happy_number 1 (-2) (-3) = -4 :=
by sorry

theorem happy_number_part2 (m : ℤ) (h : 1 < m ∧ m < 6) :
  is_happy_eq 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) →
  m = 3 ∧ happy_number 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) = -25 / 4 :=
by sorry

theorem happy_number_part3 (m n : ℤ) :
  is_happy_eq 1 (-m) (m + 1) ∧ is_happy_eq 1 (-(n + 2)) (2 * n) →
  happy_to_each_other 1 (-m) (m + 1) 1 (-(n + 2)) (2 * n) →
  n = 0 ∨ n = 3 ∨ n = 3 / 2 :=
by sorry

end HappyEquations

end NUMINAMATH_GPT_happy_number_part1_happy_number_part2_happy_number_part3_l1118_111871


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1118_111823

def ellipse {a : ℝ} (h : a^2 - 4 = 4) : Prop :=
  ∃ c e : ℝ, (c = 2) ∧ (e = c / a) ∧ (e = (Real.sqrt 2) / 2)

theorem ellipse_eccentricity (a : ℝ) (h : a^2 - 4 = 4) : 
  ellipse h :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1118_111823


namespace NUMINAMATH_GPT_problem1_l1118_111887

open Real

theorem problem1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  ∃ (m : ℝ), m = 9 / 2 ∧ ∀ (u v : ℝ), 0 < u → 0 < v → u + v = 1 → (1 / u + 4 / (1 + v)) ≥ m := 
sorry

end NUMINAMATH_GPT_problem1_l1118_111887


namespace NUMINAMATH_GPT_value_of_d_l1118_111825

theorem value_of_d (d : ℝ) (h : x^2 - 60 * x + d = (x - 30)^2) : d = 900 :=
by { sorry }

end NUMINAMATH_GPT_value_of_d_l1118_111825


namespace NUMINAMATH_GPT_probability_of_event_3a_minus_1_gt_0_l1118_111852

noncomputable def probability_event : ℝ :=
if h : 0 <= 1 then (1 - 1/3) else 0

theorem probability_of_event_3a_minus_1_gt_0 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) : 
  probability_event = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_event_3a_minus_1_gt_0_l1118_111852


namespace NUMINAMATH_GPT_compute_expression_l1118_111854

theorem compute_expression :
  24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1118_111854


namespace NUMINAMATH_GPT_height_of_table_without_book_l1118_111809

-- Define the variables and assumptions
variables (l h w : ℝ) (b : ℝ := 6)

-- State the conditions from the problem
-- Condition 1: l + h - w = 40
-- Condition 2: w + h - l + b = 34

theorem height_of_table_without_book (hlw : l + h - w = 40) (whlb : w + h - l + b = 34) : h = 34 :=
by
  -- Since we are skipping the proof, we put sorry here
  sorry

end NUMINAMATH_GPT_height_of_table_without_book_l1118_111809


namespace NUMINAMATH_GPT_sequence_count_l1118_111816

theorem sequence_count (a : ℕ → ℤ) (h₁ : a 1 = 0) (h₂ : a 11 = 4) 
  (h₃ : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → |a (k + 1) - a k| = 1) : 
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end NUMINAMATH_GPT_sequence_count_l1118_111816


namespace NUMINAMATH_GPT_gold_weight_l1118_111879

theorem gold_weight:
  ∀ (G C A : ℕ), 
  C = 9 → 
  (A = (4 * G + C) / 5) → 
  A = 17 → 
  G = 19 :=
by
  intros G C A hc ha h17
  sorry

end NUMINAMATH_GPT_gold_weight_l1118_111879


namespace NUMINAMATH_GPT_triangle_inequality_part_a_l1118_111869

theorem triangle_inequality_part_a (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  a^2 + b^2 + c^2 + a * b * c < 8 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_part_a_l1118_111869


namespace NUMINAMATH_GPT_green_apples_count_l1118_111862

def red_apples := 33
def students_took := 21
def extra_apples := 35

theorem green_apples_count : ∃ G : ℕ, red_apples + G - students_took = extra_apples ∧ G = 23 :=
by
  use 23
  have h1 : 33 + 23 - 21 = 35 := by norm_num
  exact ⟨h1, rfl⟩

end NUMINAMATH_GPT_green_apples_count_l1118_111862


namespace NUMINAMATH_GPT_sum_first_n_geometric_terms_l1118_111841

theorem sum_first_n_geometric_terms (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 2 = 2) (h2 : S 6 = 4) :
  S 4 = 1 + Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_n_geometric_terms_l1118_111841


namespace NUMINAMATH_GPT_Tim_total_expenditure_l1118_111821

theorem Tim_total_expenditure 
  (appetizer_price : ℝ) (main_course_price : ℝ) (dessert_price : ℝ)
  (appetizer_tip_percentage : ℝ) (main_course_tip_percentage : ℝ) (dessert_tip_percentage : ℝ) :
  appetizer_price = 12.35 →
  main_course_price = 27.50 →
  dessert_price = 9.95 →
  appetizer_tip_percentage = 0.18 →
  main_course_tip_percentage = 0.20 →
  dessert_tip_percentage = 0.15 →
  appetizer_price * (1 + appetizer_tip_percentage) + 
  main_course_price * (1 + main_course_tip_percentage) + 
  dessert_price * (1 + dessert_tip_percentage) = 12.35 * 1.18 + 27.50 * 1.20 + 9.95 * 1.15 :=
  by sorry

end NUMINAMATH_GPT_Tim_total_expenditure_l1118_111821


namespace NUMINAMATH_GPT_symmetric_lines_a_b_l1118_111847

theorem symmetric_lines_a_b (x y a b : ℝ) (A : ℝ × ℝ) (hA : A = (1, 0))
  (h1 : x + 2 * y - 3 = 0)
  (h2 : a * x + 4 * y + b = 0)
  (h_slope : -1 / 2 = -a / 4)
  (h_point : a * 1 + 4 * 0 + b = 0) :
  a + b = 0 :=
sorry

end NUMINAMATH_GPT_symmetric_lines_a_b_l1118_111847


namespace NUMINAMATH_GPT_gcd_75_100_l1118_111849

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end NUMINAMATH_GPT_gcd_75_100_l1118_111849


namespace NUMINAMATH_GPT_opposite_of_neg_three_l1118_111891

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_l1118_111891


namespace NUMINAMATH_GPT_find_pairs_l1118_111851

def Point := (ℤ × ℤ)

def P : Point := (1, 1)
def Q : Point := (4, 5)
def valid_pairs : List Point := [(4, 1), (7, 5), (10, 9), (1, 5), (4, 9)]

def area (P Q R : Point) : ℚ :=
  (1 / 2 : ℚ) * ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)).natAbs : ℚ)

theorem find_pairs :
  {pairs : List Point // ∀ (a b : ℤ), (0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ area P Q (a, b) = 6) ↔ (a, b) ∈ pairs} :=
  ⟨valid_pairs, by sorry⟩

end NUMINAMATH_GPT_find_pairs_l1118_111851


namespace NUMINAMATH_GPT_one_third_of_6_3_eq_21_10_l1118_111822

theorem one_third_of_6_3_eq_21_10 : (6.3 / 3) = (21 / 10) := by
  sorry

end NUMINAMATH_GPT_one_third_of_6_3_eq_21_10_l1118_111822


namespace NUMINAMATH_GPT_minNumberOfRectangles_correct_l1118_111834

variable (k n : ℤ)

noncomputable def minNumberOfRectangles (k n : ℤ) : ℤ :=
  if 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1 then
    if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1)
  else 0 -- 0 if the conditions are not met

theorem minNumberOfRectangles_correct (k n : ℤ) (h : 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1) : 
  minNumberOfRectangles k n = 
  if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1) := 
by 
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_minNumberOfRectangles_correct_l1118_111834


namespace NUMINAMATH_GPT_range_of_mu_l1118_111806

noncomputable def problem_statement (a b μ : ℝ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < μ) ∧ (1 / a + 9 / b = 1) → (0 < μ ∧ μ ≤ 16)

theorem range_of_mu (a b μ : ℝ) : problem_statement a b μ :=
  sorry

end NUMINAMATH_GPT_range_of_mu_l1118_111806


namespace NUMINAMATH_GPT_bucket_capacity_l1118_111804

theorem bucket_capacity :
  (∃ (x : ℝ), 30 * x = 45 * 9) → 13.5 = 13.5 :=
by
  -- proof needed
  sorry

end NUMINAMATH_GPT_bucket_capacity_l1118_111804


namespace NUMINAMATH_GPT_probability_two_cities_less_than_8000_l1118_111805

-- Define the city names
inductive City
| Bangkok | CapeTown | Honolulu | London | NewYork
deriving DecidableEq, Inhabited

-- Define the distance between cities
def distance : City → City → ℕ
| City.Bangkok, City.CapeTown  => 6300
| City.Bangkok, City.Honolulu  => 6609
| City.Bangkok, City.London    => 5944
| City.Bangkok, City.NewYork   => 8650
| City.CapeTown, City.Bangkok  => 6300
| City.CapeTown, City.Honolulu => 11535
| City.CapeTown, City.London   => 5989
| City.CapeTown, City.NewYork  => 7800
| City.Honolulu, City.Bangkok  => 6609
| City.Honolulu, City.CapeTown => 11535
| City.Honolulu, City.London   => 7240
| City.Honolulu, City.NewYork  => 4980
| City.London, City.Bangkok    => 5944
| City.London, City.CapeTown   => 5989
| City.London, City.Honolulu   => 7240
| City.London, City.NewYork    => 3470
| City.NewYork, City.Bangkok   => 8650
| City.NewYork, City.CapeTown  => 7800
| City.NewYork, City.Honolulu  => 4980
| City.NewYork, City.London    => 3470
| _, _                         => 0

-- Prove the probability
theorem probability_two_cities_less_than_8000 :
  let pairs := [(City.Bangkok, City.CapeTown), (City.Bangkok, City.Honolulu), (City.Bangkok, City.London), (City.CapeTown, City.London), (City.CapeTown, City.NewYork), (City.Honolulu, City.London), (City.Honolulu, City.NewYork), (City.London, City.NewYork)]
  (pairs.length : ℚ) / 10 = 4 / 5 :=
sorry

end NUMINAMATH_GPT_probability_two_cities_less_than_8000_l1118_111805


namespace NUMINAMATH_GPT_inequality_correct_l1118_111881

theorem inequality_correct (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1/a) < (1/b) :=
sorry

end NUMINAMATH_GPT_inequality_correct_l1118_111881


namespace NUMINAMATH_GPT_min_expression_value_l1118_111826

theorem min_expression_value (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ (min_val : ℝ), min_val = 12 ∧ (∀ (x y : ℝ), (x > 1) → (y > 1) →
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ min_val))) :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l1118_111826


namespace NUMINAMATH_GPT_solution1_solution2_l1118_111859

-- Problem: Solving equations and finding their roots

-- Condition 1:
def equation1 (x : Real) : Prop := x^2 - 2 * x = -1

-- Condition 2:
def equation2 (x : Real) : Prop := (x + 3)^2 = 2 * x * (x + 3)

-- Correct answer 1
theorem solution1 : ∀ x : Real, equation1 x → x = 1 := 
by 
  sorry

-- Correct answer 2
theorem solution2 : ∀ x : Real, equation2 x → x = -3 ∨ x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_solution1_solution2_l1118_111859


namespace NUMINAMATH_GPT_origin_inside_ellipse_l1118_111819

theorem origin_inside_ellipse (k : ℝ) (h : k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) : 0 < |k| ∧ |k| < 1 :=
by
  sorry

end NUMINAMATH_GPT_origin_inside_ellipse_l1118_111819


namespace NUMINAMATH_GPT_sufficiency_not_necessity_condition_l1118_111878

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficiency_not_necessity_condition_l1118_111878


namespace NUMINAMATH_GPT_sage_reflection_day_l1118_111812

theorem sage_reflection_day 
  (day_of_reflection_is_jan_1 : Prop)
  (equal_days_in_last_5_years : Prop)
  (new_year_10_years_ago_was_friday : Prop)
  (reflections_in_21st_century : Prop) : 
  ∃ (day : String), day = "Thursday" :=
by
  sorry

end NUMINAMATH_GPT_sage_reflection_day_l1118_111812


namespace NUMINAMATH_GPT_average_speed_of_train_l1118_111856

-- Define conditions
def traveled_distance1 : ℝ := 240
def traveled_distance2 : ℝ := 450
def time_period1 : ℝ := 3
def time_period2 : ℝ := 5

-- Define total distance and total time based on the conditions
def total_distance : ℝ := traveled_distance1 + traveled_distance2
def total_time : ℝ := time_period1 + time_period2

-- Prove that the average speed is 86.25 km/h
theorem average_speed_of_train : total_distance / total_time = 86.25 := by
  -- Here should be the proof, but we put sorry since we only need the statement
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l1118_111856


namespace NUMINAMATH_GPT_percentage_problem_l1118_111800

variable (x : ℝ)
variable (y : ℝ)

theorem percentage_problem : 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 → x = 33.52 := by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1118_111800


namespace NUMINAMATH_GPT_compute_m_div_18_l1118_111846

noncomputable def ten_pow (n : ℕ) : ℕ := Nat.pow 10 n

def valid_digits (m : ℕ) : Prop :=
  ∀ d ∈ m.digits 10, d = 0 ∨ d = 8

def is_multiple_of_18 (m : ℕ) : Prop :=
  m % 18 = 0

theorem compute_m_div_18 :
  ∃ m, valid_digits m ∧ is_multiple_of_18 m ∧ m / 18 = 493827160 :=
by
  sorry

end NUMINAMATH_GPT_compute_m_div_18_l1118_111846


namespace NUMINAMATH_GPT_distance_sum_conditions_l1118_111801

theorem distance_sum_conditions (a : ℚ) (k : ℚ) :
  abs (20 * a - 20 * k - 190) = 4460 ∧ abs (20 * a^2 - 20 * k - 190) = 2755 →
  a = -37 / 2 ∨ a = 39 / 2 :=
sorry

end NUMINAMATH_GPT_distance_sum_conditions_l1118_111801


namespace NUMINAMATH_GPT_stars_substitution_correct_l1118_111843

-- Define x and y with given conditions
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5
def product_ends_in_25 (x y : ℕ) : Prop := (x * y) % 100 = 25
def tens_digit_even (n : ℕ) : Prop := (n / 10) % 2 = 0
def valid_tens_digit (n : ℕ) : Prop := (n / 10) % 10 ≤ 3

theorem stars_substitution_correct :
  ∃ (x y : ℕ), ends_in_5 x ∧ ends_in_5 y ∧ product_ends_in_25 x y ∧ tens_digit_even x ∧ valid_tens_digit y ∧ x * y = 9125 :=
sorry

end NUMINAMATH_GPT_stars_substitution_correct_l1118_111843


namespace NUMINAMATH_GPT_matrix_A_pow_50_l1118_111840

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 1], ![-16, -3]]

theorem matrix_A_pow_50 :
  A ^ 50 = ![![201, 50], ![-800, -199]] :=
sorry

end NUMINAMATH_GPT_matrix_A_pow_50_l1118_111840


namespace NUMINAMATH_GPT_pieces_per_sister_l1118_111850

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end NUMINAMATH_GPT_pieces_per_sister_l1118_111850


namespace NUMINAMATH_GPT_four_thirds_eq_36_l1118_111853

theorem four_thirds_eq_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := by
  sorry

end NUMINAMATH_GPT_four_thirds_eq_36_l1118_111853


namespace NUMINAMATH_GPT_gcd_of_1230_and_920_is_10_l1118_111898

theorem gcd_of_1230_and_920_is_10 : Int.gcd 1230 920 = 10 :=
sorry

end NUMINAMATH_GPT_gcd_of_1230_and_920_is_10_l1118_111898


namespace NUMINAMATH_GPT_polygon_sides_l1118_111824

theorem polygon_sides (n : ℕ) (h_sum : 180 * (n - 2) = 1980) : n = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_sides_l1118_111824


namespace NUMINAMATH_GPT_find_x_in_acute_triangle_l1118_111833

-- Definition of an acute triangle with given segment lengths due to altitudes
def acute_triangle_with_segments (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) : Prop :=
  BC = 4 + x ∧ AE = x ∧ BE = 8 ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- The theorem to prove
theorem find_x_in_acute_triangle (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) 
  (h : acute_triangle_with_segments A B C D E BC AE BE x) : 
  x = 4 :=
by
  -- As the focus is on the statement, we add sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_find_x_in_acute_triangle_l1118_111833


namespace NUMINAMATH_GPT_exists_three_cycle_l1118_111807

variable {α : Type}

def tournament (P : α → α → Prop) : Prop :=
  (∃ (participants : List α), participants.length ≥ 3) ∧
  (∀ x y, x ≠ y → P x y ∨ P y x) ∧
  (∀ x, ∃ y, P x y)

theorem exists_three_cycle {α : Type} (P : α → α → Prop) :
  tournament P → ∃ A B C, P A B ∧ P B C ∧ P C A :=
by
  sorry

end NUMINAMATH_GPT_exists_three_cycle_l1118_111807


namespace NUMINAMATH_GPT_escher_prints_consecutive_l1118_111883

noncomputable def probability_all_eschers_consecutive (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
if h : m = n + 3 ∧ k = 4 then 1 / (n * (n + 1) * (n + 2)) else 0

theorem escher_prints_consecutive :
  probability_all_eschers_consecutive 10 12 4 = 1 / 1320 :=
  by sorry

end NUMINAMATH_GPT_escher_prints_consecutive_l1118_111883


namespace NUMINAMATH_GPT_final_fish_stock_l1118_111866

def initial_stock : ℤ := 200 
def sold_fish : ℤ := 50 
def fraction_spoiled : ℚ := 1/3 
def new_stock : ℤ := 200 

theorem final_fish_stock : 
    initial_stock - sold_fish - (fraction_spoiled * (initial_stock - sold_fish)) + new_stock = 300 := 
by 
  sorry

end NUMINAMATH_GPT_final_fish_stock_l1118_111866


namespace NUMINAMATH_GPT_min_students_l1118_111820

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : (b + g) % 5 = 2) : 
  b + g = 57 :=
sorry

end NUMINAMATH_GPT_min_students_l1118_111820


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1118_111848

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1118_111848


namespace NUMINAMATH_GPT_problem_solution_l1118_111897

theorem problem_solution :
  (1/3⁻¹) - Real.sqrt 27 + 3 * Real.tan (Real.pi / 6) + (Real.pi - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1118_111897


namespace NUMINAMATH_GPT_xy_zero_iff_x_zero_necessary_not_sufficient_l1118_111832

theorem xy_zero_iff_x_zero_necessary_not_sufficient {x y : ℝ} : 
  (x * y = 0) → ((x = 0) ∨ (y = 0)) ∧ ¬((x = 0) → (x * y ≠ 0)) := 
sorry

end NUMINAMATH_GPT_xy_zero_iff_x_zero_necessary_not_sufficient_l1118_111832


namespace NUMINAMATH_GPT_find_minimal_sum_n_l1118_111802

noncomputable def minimal_sum_n {a : ℕ → ℤ} {S : ℕ → ℤ} (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : ℕ := 
     5

theorem find_minimal_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : minimal_sum_n h1 h2 h3 = 5 :=
    sorry

end NUMINAMATH_GPT_find_minimal_sum_n_l1118_111802


namespace NUMINAMATH_GPT_geometric_series_sum_l1118_111874

theorem geometric_series_sum (a r : ℚ) (ha : a = 1) (hr : r = 1/4) : 
  (∑' n:ℕ, a * r^n) = 4/3 :=
by
  rw [ha, hr]
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1118_111874


namespace NUMINAMATH_GPT_total_marks_math_physics_l1118_111828

variable (M P C : ℕ)

theorem total_marks_math_physics (h1 : C = P + 10) (h2 : (M + C) / 2 = 35) : M + P = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_marks_math_physics_l1118_111828


namespace NUMINAMATH_GPT_ones_digit_of_six_power_l1118_111814

theorem ones_digit_of_six_power (n : ℕ) (hn : n ≥ 1) : (6 ^ n) % 10 = 6 :=
by
  sorry

example : (6 ^ 34) % 10 = 6 :=
by
  have h : 34 ≥ 1 := by norm_num
  exact ones_digit_of_six_power 34 h

end NUMINAMATH_GPT_ones_digit_of_six_power_l1118_111814


namespace NUMINAMATH_GPT_remainder_sets_two_disjoint_subsets_l1118_111858

noncomputable def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem remainder_sets_two_disjoint_subsets (m : ℕ)
  (h : m = (3^12 - 2 * 2^12 + 1) / 2) : m % 1000 = 625 := 
by {
  -- math proof is omitted
  sorry
}

end NUMINAMATH_GPT_remainder_sets_two_disjoint_subsets_l1118_111858


namespace NUMINAMATH_GPT_find_theta_l1118_111861

def equilateral_triangle_angle : ℝ := 60
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def total_round_angle : ℝ := 360

theorem find_theta (θ : ℝ)
  (h_eq_tri : equilateral_triangle_angle = 60)
  (h_squ : square_angle = 90)
  (h_pen : pentagon_angle = 108)
  (h_round : total_round_angle = 360) :
  θ = total_round_angle - (equilateral_triangle_angle + square_angle + pentagon_angle) :=
sorry

end NUMINAMATH_GPT_find_theta_l1118_111861


namespace NUMINAMATH_GPT_weight_of_8_moles_CCl4_correct_l1118_111836

/-- The problem states that carbon tetrachloride (CCl4) is given, and we are to determine the weight of 8 moles of CCl4 based on its molar mass calculations. -/
noncomputable def weight_of_8_moles_CCl4 (molar_mass_C : ℝ) (molar_mass_Cl : ℝ) : ℝ :=
  let molar_mass_CCl4 := molar_mass_C + 4 * molar_mass_Cl
  8 * molar_mass_CCl4

/-- Given the molar masses of Carbon (C) and Chlorine (Cl), prove that the calculated weight of 8 moles of CCl4 matches the expected weight. -/
theorem weight_of_8_moles_CCl4_correct :
  let molar_mass_C := 12.01
  let molar_mass_Cl := 35.45
  weight_of_8_moles_CCl4 molar_mass_C molar_mass_Cl = 1230.48 := by
  sorry

end NUMINAMATH_GPT_weight_of_8_moles_CCl4_correct_l1118_111836


namespace NUMINAMATH_GPT_minimum_value_l1118_111857

noncomputable def min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :=
  a + 2 * b

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :
  min_value a b h₁ h₂ h₃ ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1118_111857


namespace NUMINAMATH_GPT_katherine_has_4_apples_l1118_111882

variable (A P : ℕ)

theorem katherine_has_4_apples
  (h1 : P = 3 * A)
  (h2 : A + P = 16) :
  A = 4 := 
sorry

end NUMINAMATH_GPT_katherine_has_4_apples_l1118_111882


namespace NUMINAMATH_GPT_arrangements_with_AB_together_l1118_111895

theorem arrangements_with_AB_together (n : ℕ) (A B: ℕ) (students: Finset ℕ) (h₁ : students.card = 6) (h₂ : A ∈ students) (h₃ : B ∈ students):
  ∃! (count : ℕ), count = 240 :=
by
  sorry

end NUMINAMATH_GPT_arrangements_with_AB_together_l1118_111895


namespace NUMINAMATH_GPT_percentage_orange_juice_l1118_111888

-- Definitions based on conditions
def total_volume : ℝ := 120
def watermelon_percentage : ℝ := 0.60
def grape_juice_volume : ℝ := 30
def watermelon_juice_volume : ℝ := watermelon_percentage * total_volume
def combined_watermelon_grape_volume : ℝ := watermelon_juice_volume + grape_juice_volume
def orange_juice_volume : ℝ := total_volume - combined_watermelon_grape_volume

-- Lean 4 statement to prove the percentage of orange juice
theorem percentage_orange_juice : (orange_juice_volume / total_volume) * 100 = 15 := by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_percentage_orange_juice_l1118_111888


namespace NUMINAMATH_GPT_total_balls_l1118_111831

theorem total_balls (blue red green yellow purple orange black white : ℕ) 
  (h1 : blue = 8)
  (h2 : red = 5)
  (h3 : green = 3 * (2 * blue - 1))
  (h4 : yellow = Nat.floor (2 * Real.sqrt (red * blue)))
  (h5 : purple = 4 * (blue + green))
  (h6 : orange = 7)
  (h7 : black + white = blue + red + green + yellow + purple + orange)
  (h8 : blue + red + green + yellow + purple + orange + black + white = 3 * (red + green + yellow + purple) + orange / 2)
  : blue + red + green + yellow + purple + orange + black + white = 829 :=
by
  sorry

end NUMINAMATH_GPT_total_balls_l1118_111831


namespace NUMINAMATH_GPT_parabola_translation_l1118_111817

theorem parabola_translation :
  (∀ x, y = x^2) →
  (∀ x, y = (x + 1)^2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_translation_l1118_111817


namespace NUMINAMATH_GPT_solution_of_inequality_system_l1118_111845

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end NUMINAMATH_GPT_solution_of_inequality_system_l1118_111845


namespace NUMINAMATH_GPT_cory_packs_l1118_111885

theorem cory_packs (total_money_needed cost_per_pack : ℕ) (h1 : total_money_needed = 98) (h2 : cost_per_pack = 49) : total_money_needed / cost_per_pack = 2 :=
by 
  sorry

end NUMINAMATH_GPT_cory_packs_l1118_111885


namespace NUMINAMATH_GPT_smaller_cube_volume_l1118_111877

theorem smaller_cube_volume
  (d : ℝ) (s : ℝ) (V : ℝ)
  (h1 : d = 12)  -- condition: diameter of the sphere equals the edge length of the larger cube
  (h2 : d = s * Real.sqrt 3)  -- condition: space diagonal of the smaller cube equals the diameter of the sphere
  (h3 : s = 12 / Real.sqrt 3)  -- condition: side length of the smaller cube
  (h4 : V = s^3)  -- condition: volume of the cube with side length s
  : V = 192 * Real.sqrt 3 :=  -- proving the volume of the smaller cube
sorry

end NUMINAMATH_GPT_smaller_cube_volume_l1118_111877


namespace NUMINAMATH_GPT_remainder_div_by_3_not_divisible_by_9_l1118_111855

theorem remainder_div_by_3 (x : ℕ) (h : x = 1493826) : x % 3 = 0 :=
by sorry

theorem not_divisible_by_9 (x : ℕ) (h : x = 1493826) : x % 9 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_remainder_div_by_3_not_divisible_by_9_l1118_111855


namespace NUMINAMATH_GPT_y_intercept_of_line_l1118_111896

theorem y_intercept_of_line (m : ℝ) (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : b = 0) (h_slope : m = 3) (h_x_intercept : (a, b) = (4, 0)) :
  ∃ y : ℝ, (0, y) = (0, -12) :=
by 
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1118_111896


namespace NUMINAMATH_GPT_linda_total_distance_l1118_111830

theorem linda_total_distance
  (miles_per_gallon : ℝ) (tank_capacity : ℝ) (initial_distance : ℝ) (refuel_amount : ℝ) (final_tank_fraction : ℝ)
  (fuel_used_first_segment : ℝ := initial_distance / miles_per_gallon)
  (initial_fuel_full : fuel_used_first_segment = tank_capacity)
  (total_fuel_after_refuel : ℝ := 0 + refuel_amount)
  (remaining_fuel_stopping : ℝ := final_tank_fraction * tank_capacity)
  (fuel_used_second_segment : ℝ := total_fuel_after_refuel - remaining_fuel_stopping)
  (distance_second_leg : ℝ := fuel_used_second_segment * miles_per_gallon) :
  initial_distance + distance_second_leg = 637.5 := by
  sorry

end NUMINAMATH_GPT_linda_total_distance_l1118_111830


namespace NUMINAMATH_GPT_cos_690_eq_sqrt3_div_2_l1118_111867

theorem cos_690_eq_sqrt3_div_2 : Real.cos (690 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_690_eq_sqrt3_div_2_l1118_111867


namespace NUMINAMATH_GPT_pyramid_height_l1118_111884

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end NUMINAMATH_GPT_pyramid_height_l1118_111884


namespace NUMINAMATH_GPT_expression_positive_for_all_integers_l1118_111872

theorem expression_positive_for_all_integers (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_positive_for_all_integers_l1118_111872


namespace NUMINAMATH_GPT_min_value_of_f_l1118_111810

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log x / Real.log 2) * (Real.log (2 * x) / Real.log 2) else 0

theorem min_value_of_f : ∃ x > 0, f x = -1/4 :=
sorry

end NUMINAMATH_GPT_min_value_of_f_l1118_111810


namespace NUMINAMATH_GPT_roses_formula_l1118_111870

open Nat

def total_roses (n : ℕ) : ℕ := 
  (choose n 4) + (choose (n - 1) 2)

theorem roses_formula (n : ℕ) (h : n ≥ 4) : 
  total_roses n = (choose n 4) + (choose (n - 1) 2) := 
by
  sorry

end NUMINAMATH_GPT_roses_formula_l1118_111870


namespace NUMINAMATH_GPT_envelopes_initial_count_l1118_111835

noncomputable def initialEnvelopes (given_per_friend : ℕ) (friends : ℕ) (left : ℕ) : ℕ :=
  given_per_friend * friends + left

theorem envelopes_initial_count
  (given_per_friend : ℕ) (friends : ℕ) (left : ℕ)
  (h_given_per_friend : given_per_friend = 3)
  (h_friends : friends = 5)
  (h_left : left = 22) :
  initialEnvelopes given_per_friend friends left = 37 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end NUMINAMATH_GPT_envelopes_initial_count_l1118_111835


namespace NUMINAMATH_GPT_min_value_expr_l1118_111827

/-- Given x > y > 0 and x^2 - y^2 = 1, we need to prove that the minimum value of 2x^2 + 3y^2 - 4xy is 1. -/
theorem min_value_expr {x y : ℝ} (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  2 * x^2 + 3 * y^2 - 4 * x * y = 1 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l1118_111827


namespace NUMINAMATH_GPT_probability_no_defective_pens_l1118_111890

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (non_defective_pens : ℕ) (prob_first_non_defective : ℚ) (prob_second_non_defective : ℚ) :
  total_pens = 12 →
  defective_pens = 4 →
  non_defective_pens = total_pens - defective_pens →
  prob_first_non_defective = non_defective_pens / total_pens →
  prob_second_non_defective = (non_defective_pens - 1) / (total_pens - 1) →
  prob_first_non_defective * prob_second_non_defective = 14 / 33 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end NUMINAMATH_GPT_probability_no_defective_pens_l1118_111890


namespace NUMINAMATH_GPT_unique_solution_positive_integers_l1118_111813

theorem unique_solution_positive_integers :
  ∀ (a b : ℕ), (0 < a ∧ 0 < b ∧ ∃ k m : ℤ, a^3 + 6 * a * b + 1 = k^3 ∧ b^3 + 6 * a * b + 1 = m^3) → (a = 1 ∧ b = 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_unique_solution_positive_integers_l1118_111813


namespace NUMINAMATH_GPT_good_students_l1118_111889

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end NUMINAMATH_GPT_good_students_l1118_111889


namespace NUMINAMATH_GPT_banker_gain_l1118_111837

theorem banker_gain :
  ∀ (t : ℝ) (r : ℝ) (TD : ℝ),
  t = 1 →
  r = 12 →
  TD = 65 →
  (TD * r * t) / (100 - (r * t)) = 8.86 :=
by
  intros t r TD ht hr hTD
  rw [ht, hr, hTD]
  sorry

end NUMINAMATH_GPT_banker_gain_l1118_111837


namespace NUMINAMATH_GPT_necessary_condition_x_squared_minus_x_lt_zero_l1118_111875

theorem necessary_condition_x_squared_minus_x_lt_zero (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧ ((-1 < x ∧ x < 1) → ¬ (x^2 - x < 0)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_x_squared_minus_x_lt_zero_l1118_111875


namespace NUMINAMATH_GPT_walkways_area_l1118_111893

theorem walkways_area (rows cols : ℕ) (bed_length bed_width walkthrough_width garden_length garden_width total_flower_beds bed_area total_bed_area total_garden_area : ℝ) 
  (h1 : rows = 4) (h2 : cols = 3) 
  (h3 : bed_length = 8) (h4 : bed_width = 3) 
  (h5 : walkthrough_width = 2)
  (h6 : garden_length = (cols * bed_length) + ((cols + 1) * walkthrough_width))
  (h7 : garden_width = (rows * bed_width) + ((rows + 1) * walkthrough_width))
  (h8 : total_garden_area = garden_length * garden_width)
  (h9 : total_flower_beds = rows * cols)
  (h10 : bed_area = bed_length * bed_width)
  (h11 : total_bed_area = total_flower_beds * bed_area)
  (h12 : total_garden_area - total_bed_area = 416) : 
  True := 
sorry

end NUMINAMATH_GPT_walkways_area_l1118_111893


namespace NUMINAMATH_GPT_incorrect_conversion_D_l1118_111868

-- Definition of base conversions as conditions
def binary_to_decimal (b : String) : ℕ := -- Converts binary string to decimal number
  sorry

def octal_to_decimal (o : String) : ℕ := -- Converts octal string to decimal number
  sorry

def decimal_to_base_n (d : ℕ) (n : ℕ) : String := -- Converts decimal number to base-n string
  sorry

-- Given conditions
axiom cond1 : binary_to_decimal "101" = 5
axiom cond2 : octal_to_decimal "27" = 25 -- Note: "27"_base(8) is 2*8 + 7 = 23 in decimal; there's a typo in question's option.
axiom cond3 : decimal_to_base_n 119 6 = "315"
axiom cond4 : decimal_to_base_n 13 2 = "1101" -- Note: correcting from 62 to "1101"_base(2) which is 13

-- Prove the incorrect conversion between number systems
theorem incorrect_conversion_D : decimal_to_base_n 31 4 ≠ "62" :=
  sorry

end NUMINAMATH_GPT_incorrect_conversion_D_l1118_111868


namespace NUMINAMATH_GPT_fourth_quarter_points_sum_l1118_111880

variable (a d b j : ℕ)

-- Conditions from the problem
def halftime_tied : Prop := 2 * a + d = 2 * b
def wildcats_won_by_four : Prop := 4 * a + 6 * d = 4 * b - 3 * j + 4

-- The proof goal to be established
theorem fourth_quarter_points_sum
  (h1 : halftime_tied a d b)
  (h2 : wildcats_won_by_four a d b j) :
  (a + 3 * d) + (b - 2 * j) = 28 :=
sorry

end NUMINAMATH_GPT_fourth_quarter_points_sum_l1118_111880


namespace NUMINAMATH_GPT_lucas_pay_per_window_l1118_111803

-- Conditions
def num_floors : Nat := 3
def windows_per_floor : Nat := 3
def days_to_finish : Nat := 6
def penalty_rate : Nat := 3
def penalty_amount : Nat := 1
def final_payment : Nat := 16

-- Theorem statement
theorem lucas_pay_per_window :
  let total_windows := num_floors * windows_per_floor
  let total_penalty := penalty_amount * (days_to_finish / penalty_rate)
  let original_payment := final_payment + total_penalty
  let payment_per_window := original_payment / total_windows
  payment_per_window = 2 :=
by
  sorry

end NUMINAMATH_GPT_lucas_pay_per_window_l1118_111803
