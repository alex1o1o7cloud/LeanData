import Mathlib

namespace NUMINAMATH_GPT_greatest_integer_b_l1858_185804

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ↔ b ≤ 6 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_b_l1858_185804


namespace NUMINAMATH_GPT_find_a_l1858_185830

theorem find_a (a : ℝ) (h : -2 * a + 1 = -1) : a = 1 :=
by sorry

end NUMINAMATH_GPT_find_a_l1858_185830


namespace NUMINAMATH_GPT_choose_president_vice_president_and_committee_l1858_185868

theorem choose_president_vice_president_and_committee :
  let num_ways : ℕ := 10 * 9 * (Nat.choose 8 2)
  num_ways = 2520 :=
by
  sorry

end NUMINAMATH_GPT_choose_president_vice_president_and_committee_l1858_185868


namespace NUMINAMATH_GPT_num_games_played_l1858_185897

theorem num_games_played (n : ℕ) (h : n = 14) : (n.choose 2) = 91 :=
by
  sorry

end NUMINAMATH_GPT_num_games_played_l1858_185897


namespace NUMINAMATH_GPT_c_negative_l1858_185825

theorem c_negative (a b c : ℝ) (h₁ : a + b + c < 0) (h₂ : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 :=
sorry

end NUMINAMATH_GPT_c_negative_l1858_185825


namespace NUMINAMATH_GPT_gcd_gx_x_multiple_of_18432_l1858_185879

def g (x : ℕ) : ℕ := (3*x + 5) * (7*x + 2) * (13*x + 7) * (2*x + 10)

theorem gcd_gx_x_multiple_of_18432 (x : ℕ) (h : ∃ k : ℕ, x = 18432 * k) : Nat.gcd (g x) x = 28 :=
by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_multiple_of_18432_l1858_185879


namespace NUMINAMATH_GPT_num_rectangles_grid_l1858_185834

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end NUMINAMATH_GPT_num_rectangles_grid_l1858_185834


namespace NUMINAMATH_GPT_total_pieces_l1858_185812

def gum_packages : ℕ := 28
def candy_packages : ℕ := 14
def pieces_per_package : ℕ := 6

theorem total_pieces : (gum_packages * pieces_per_package) + (candy_packages * pieces_per_package) = 252 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_l1858_185812


namespace NUMINAMATH_GPT_min_value_4a2_b2_plus_1_div_2a_minus_b_l1858_185873

variable (a b : ℝ)

theorem min_value_4a2_b2_plus_1_div_2a_minus_b (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a > b) (h4 : a * b = 1 / 2) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x > y → x * y = 1 / 2 → (4 * x^2 + y^2 + 1) / (2 * x - y) ≥ c) :=
sorry

end NUMINAMATH_GPT_min_value_4a2_b2_plus_1_div_2a_minus_b_l1858_185873


namespace NUMINAMATH_GPT_chord_length_l1858_185885

/-- Given two concentric circles with radii R and r, where the area of the annulus between them is 16π,
    a chord of the larger circle that is tangent to the smaller circle has a length of 8. -/
theorem chord_length {R r c : ℝ} 
  (h1 : R^2 - r^2 = 16)
  (h2 : (c / 2)^2 + r^2 = R^2) :
  c = 8 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l1858_185885


namespace NUMINAMATH_GPT_probability_from_first_to_last_l1858_185836

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end NUMINAMATH_GPT_probability_from_first_to_last_l1858_185836


namespace NUMINAMATH_GPT_inequality_abc_sum_one_l1858_185831

theorem inequality_abc_sum_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 1) :
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 +
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 +
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 +
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end NUMINAMATH_GPT_inequality_abc_sum_one_l1858_185831


namespace NUMINAMATH_GPT_difference_of_squares_l1858_185889

theorem difference_of_squares (n : ℕ) : (n+1)^2 - n^2 = 2*n + 1 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1858_185889


namespace NUMINAMATH_GPT_sum_of_two_primes_is_multiple_of_six_l1858_185815

theorem sum_of_two_primes_is_multiple_of_six
  (p q r : ℕ)
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) (hr_gt_3 : r > 3)
  (h_sum_prime : Nat.Prime (p + q + r)) : 
  (p + q) % 6 = 0 ∨ (p + r) % 6 = 0 ∨ (q + r) % 6 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_two_primes_is_multiple_of_six_l1858_185815


namespace NUMINAMATH_GPT_simplify_expression_l1858_185813

theorem simplify_expression (a b : ℤ) : 
  (18 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 40 * b) = 21 * a + 41 * b := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1858_185813


namespace NUMINAMATH_GPT_questions_ratio_l1858_185843

theorem questions_ratio (R A : ℕ) (H₁ : R + 6 + A = 24) :
  (R, 6, A) = (R, 6, A) :=
sorry

end NUMINAMATH_GPT_questions_ratio_l1858_185843


namespace NUMINAMATH_GPT_tourist_growth_rate_l1858_185863

theorem tourist_growth_rate (F : ℝ) (x : ℝ) 
    (hMarch : F * 0.6 = 0.6 * F)
    (hApril : F * 0.6 * 0.5 = 0.3 * F)
    (hMay : 2 * F = 2 * F):
    (0.6 * 0.5 * (1 + x) = 2) :=
by
  sorry

end NUMINAMATH_GPT_tourist_growth_rate_l1858_185863


namespace NUMINAMATH_GPT_sum_of_squares_expr_l1858_185887

theorem sum_of_squares_expr : 
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_expr_l1858_185887


namespace NUMINAMATH_GPT_sample_size_eq_36_l1858_185845

def total_population := 27 + 54 + 81
def ratio_elderly_total := 27 / total_population
def selected_elderly := 6
def sample_size := 36

theorem sample_size_eq_36 : 
  (selected_elderly : ℚ) / (sample_size : ℚ) = ratio_elderly_total → 
  sample_size = 36 := 
by 
sorry

end NUMINAMATH_GPT_sample_size_eq_36_l1858_185845


namespace NUMINAMATH_GPT_problem_l1858_185801

variable (a : ℕ → ℝ) -- {a_n} is a sequence
variable (S : ℕ → ℝ) -- S_n represents the sum of the first n terms
variable (d : ℝ) -- non-zero common difference
variable (a1 : ℝ) -- first term of the sequence

-- Define an arithmetic sequence with common difference d and first term a1
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (a1 : ℝ) 
  (h_non_zero : d ≠ 0)
  (h_sequence : is_arithmetic_sequence a d a1)
  (h_sum : sum_of_arithmetic_sequence S a)
  (h_S5_eq_S6 : S 5 = S 6) :
  S 11 = 0 := 
sorry

end NUMINAMATH_GPT_problem_l1858_185801


namespace NUMINAMATH_GPT_find_a_l1858_185861

theorem find_a (a : ℤ) (h_range : 0 ≤ a ∧ a < 13) (h_div : (51 ^ 2022 + a) % 13 = 0) : a = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l1858_185861


namespace NUMINAMATH_GPT_probability_of_prime_number_on_spinner_l1858_185864

-- Definitions of conditions
def spinner_sections : List ℕ := [2, 3, 4, 5, 7, 9, 10, 11]
def total_sectors : ℕ := 8
def prime_count : ℕ := List.filter Nat.Prime spinner_sections |>.length

-- Statement of the theorem we want to prove
theorem probability_of_prime_number_on_spinner :
  (prime_count : ℚ) / total_sectors = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_probability_of_prime_number_on_spinner_l1858_185864


namespace NUMINAMATH_GPT_total_cases_after_three_weeks_l1858_185822

theorem total_cases_after_three_weeks (week1_cases week2_cases week3_cases : ℕ) 
  (h1 : week1_cases = 5000)
  (h2 : week2_cases = week1_cases + week1_cases / 10 * 3)
  (h3 : week3_cases = week2_cases - week2_cases / 10 * 2) :
  week1_cases + week2_cases + week3_cases = 16700 := 
by
  sorry

end NUMINAMATH_GPT_total_cases_after_three_weeks_l1858_185822


namespace NUMINAMATH_GPT_num_bases_ending_in_1_l1858_185833

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end NUMINAMATH_GPT_num_bases_ending_in_1_l1858_185833


namespace NUMINAMATH_GPT_total_food_correct_l1858_185899

def max_food_per_guest : ℕ := 2
def min_guests : ℕ := 162
def total_food_cons : ℕ := min_guests * max_food_per_guest

theorem total_food_correct : total_food_cons = 324 := by
  sorry

end NUMINAMATH_GPT_total_food_correct_l1858_185899


namespace NUMINAMATH_GPT_number_of_intersections_l1858_185882

def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

theorem number_of_intersections : 
  ∃ (p1 p2 : ℝ × ℝ), 
  (line_eq p1.1 p1.2 ∧ circle_eq p1.1 p1.2) ∧ 
  (line_eq p2.1 p2.2 ∧ circle_eq p2.1 p2.2) ∧ 
  p1 ≠ p2 ∧ 
  ∀ p : ℝ × ℝ, 
    (line_eq p.1 p.2 ∧ circle_eq p.1 p.2) → (p = p1 ∨ p = p2) :=
sorry

end NUMINAMATH_GPT_number_of_intersections_l1858_185882


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_12_l1858_185872

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_12_l1858_185872


namespace NUMINAMATH_GPT_find_first_term_l1858_185877

theorem find_first_term (S_n : ℕ → ℝ) (a d : ℝ) (n : ℕ) (h₁ : ∀ n > 0, S_n n = n * (2 * a + (n - 1) * d) / 2)
  (h₂ : d = 3) (h₃ : ∃ c, ∀ n > 0, S_n (3 * n) / S_n n = c) : a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_first_term_l1858_185877


namespace NUMINAMATH_GPT_nesbitts_inequality_l1858_185844

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_nesbitts_inequality_l1858_185844


namespace NUMINAMATH_GPT_reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l1858_185883

theorem reciprocal_opposite_of_neg_neg_3_is_neg_one_third : 
  (1 / (-(-3))) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l1858_185883


namespace NUMINAMATH_GPT_company_C_more_than_A_l1858_185858

theorem company_C_more_than_A (A B C D: ℕ) (hA: A = 30) (hB: B = 2 * A)
    (hC: C = A + 10) (hD: D = C - 5) (total: A + B + C + D = 165) : C - A = 10 := 
by 
  sorry

end NUMINAMATH_GPT_company_C_more_than_A_l1858_185858


namespace NUMINAMATH_GPT_length_of_field_l1858_185802

variable (w l : ℝ)
variable (H1 : l = 2 * w)
variable (pond_area : ℝ := 64)
variable (field_area : ℝ := l * w)
variable (H2 : pond_area = (1 / 98) * field_area)

theorem length_of_field : l = 112 :=
by
  sorry

end NUMINAMATH_GPT_length_of_field_l1858_185802


namespace NUMINAMATH_GPT_total_revenue_correct_l1858_185828

def price_per_book : ℝ := 25
def books_sold_monday : ℕ := 60
def discount_monday : ℝ := 0.10
def books_sold_tuesday : ℕ := 10
def discount_tuesday : ℝ := 0.0
def books_sold_wednesday : ℕ := 20
def discount_wednesday : ℝ := 0.05
def books_sold_thursday : ℕ := 44
def discount_thursday : ℝ := 0.15
def books_sold_friday : ℕ := 66
def discount_friday : ℝ := 0.20

def revenue (books_sold: ℕ) (discount: ℝ) : ℝ :=
  (1 - discount) * price_per_book * books_sold

theorem total_revenue_correct :
  revenue books_sold_monday discount_monday +
  revenue books_sold_tuesday discount_tuesday +
  revenue books_sold_wednesday discount_wednesday +
  revenue books_sold_thursday discount_thursday +
  revenue books_sold_friday discount_friday = 4330 := by 
sorry

end NUMINAMATH_GPT_total_revenue_correct_l1858_185828


namespace NUMINAMATH_GPT_min_value_of_function_l1858_185869

noncomputable def func (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sin (2 * x)

theorem min_value_of_function : ∃ x : ℝ, func x = 1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_min_value_of_function_l1858_185869


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1858_185827

-- Problem (1)
theorem problem1 : (-8 - 6 + 24) = 10 :=
by sorry

-- Problem (2)
theorem problem2 : (-48 / 6 + -21 * (-1 / 3)) = -1 :=
by sorry

-- Problem (3)
theorem problem3 : ((1 / 8 - 1 / 3 + 1 / 4) * -24) = -1 :=
by sorry

-- Problem (4)
theorem problem4 : (-1^4 - (1 + 0.5) * (1 / 3) * (1 - (-2)^2)) = 0.5 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1858_185827


namespace NUMINAMATH_GPT_rulers_in_drawer_l1858_185810

-- conditions
def initial_rulers : ℕ := 46
def additional_rulers : ℕ := 25

-- question: total rulers in the drawer
def total_rulers : ℕ := initial_rulers + additional_rulers

-- proof statement: prove that total_rulers is 71
theorem rulers_in_drawer : total_rulers = 71 := by
  sorry

end NUMINAMATH_GPT_rulers_in_drawer_l1858_185810


namespace NUMINAMATH_GPT_monthly_income_A_l1858_185884

theorem monthly_income_A (A B C : ℝ) :
  A + B = 10100 ∧ B + C = 12500 ∧ A + C = 10400 →
  A = 4000 :=
by
  intro h
  have h1 : A + B = 10100 := h.1
  have h2 : B + C = 12500 := h.2.1
  have h3 : A + C = 10400 := h.2.2
  sorry

end NUMINAMATH_GPT_monthly_income_A_l1858_185884


namespace NUMINAMATH_GPT_matches_needed_eq_l1858_185852

def count_matches (n : ℕ) : ℕ :=
  let total_triangles := n * n
  let internal_matches := 3 * total_triangles
  let external_matches := 4 * n
  internal_matches - external_matches + external_matches

theorem matches_needed_eq (n : ℕ) : count_matches 10 = 320 :=
by
  sorry

end NUMINAMATH_GPT_matches_needed_eq_l1858_185852


namespace NUMINAMATH_GPT_total_teaching_time_l1858_185888

def teaching_times :=
  let eduardo_math_time := 3 * 60
  let eduardo_science_time := 4 * 90
  let eduardo_history_time := 2 * 120
  let total_eduardo_time := eduardo_math_time + eduardo_science_time + eduardo_history_time

  let frankie_math_time := 2 * (3 * 60)
  let frankie_science_time := 2 * (4 * 90)
  let frankie_history_time := 2 * (2 * 120)
  let total_frankie_time := frankie_math_time + frankie_science_time + frankie_history_time

  let georgina_math_time := 3 * (3 * 80)
  let georgina_science_time := 3 * (4 * 100)
  let georgina_history_time := 3 * (2 * 150)
  let total_georgina_time := georgina_math_time + georgina_science_time + georgina_history_time

  total_eduardo_time + total_frankie_time + total_georgina_time

theorem total_teaching_time : teaching_times = 5160 := by
  -- calculations omitted
  sorry

end NUMINAMATH_GPT_total_teaching_time_l1858_185888


namespace NUMINAMATH_GPT_jack_initial_yen_l1858_185890

theorem jack_initial_yen 
  (pounds yen_per_pound euros pounds_per_euro total_yen : ℕ)
  (h₁ : pounds = 42)
  (h₂ : euros = 11)
  (h₃ : pounds_per_euro = 2)
  (h₄ : yen_per_pound = 100)
  (h₅ : total_yen = 9400) : 
  ∃ initial_yen : ℕ, initial_yen = 3000 :=
by
  sorry

end NUMINAMATH_GPT_jack_initial_yen_l1858_185890


namespace NUMINAMATH_GPT_q_one_eq_five_l1858_185881

variable (q : ℝ → ℝ)
variable (h : q 1 = 5)

theorem q_one_eq_five : q 1 = 5 :=
by sorry

end NUMINAMATH_GPT_q_one_eq_five_l1858_185881


namespace NUMINAMATH_GPT_remainder_when_divided_by_5_l1858_185874

theorem remainder_when_divided_by_5 
  (k : ℕ)
  (h1 : k % 6 = 5)
  (h2 : k < 42)
  (h3 : k % 7 = 3) : 
  k % 5 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_5_l1858_185874


namespace NUMINAMATH_GPT_no_such_triplets_of_positive_reals_l1858_185839

-- Define the conditions that the problem states.
def satisfies_conditions (a b c : ℝ) : Prop :=
  a = b + c ∧ b = c + a ∧ c = a + b

-- The main theorem to prove.
theorem no_such_triplets_of_positive_reals :
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → satisfies_conditions a b c → false :=
by
  intro a b c
  intro ha hb hc
  intro habc
  sorry

end NUMINAMATH_GPT_no_such_triplets_of_positive_reals_l1858_185839


namespace NUMINAMATH_GPT_increasing_interval_of_f_l1858_185853

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x, x > 2 → ∀ y, y > x → f x < f y :=
sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l1858_185853


namespace NUMINAMATH_GPT_max_time_digit_sum_l1858_185896

-- Define the conditions
def is_valid_time (h m : ℕ) : Prop :=
  (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60)

-- Define the function to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n % 10 + n / 10

-- Define the function to calculate the sum of digits in the time display
def time_digit_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

-- The theorem to prove
theorem max_time_digit_sum : ∀ (h m : ℕ),
  is_valid_time h m → time_digit_sum h m ≤ 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_time_digit_sum_l1858_185896


namespace NUMINAMATH_GPT_solve_trig_eq_l1858_185867

theorem solve_trig_eq (x : ℝ) :
  (0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - Real.cos (2 * x) ^ 2 + Real.sin (3 * x) ^ 2 = 0) →
  (∃ k : ℤ, x = (Real.pi / 2) * (2 * k + 1) ∨ x = (2 * k * Real.pi / 11)) :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l1858_185867


namespace NUMINAMATH_GPT_inequality_system_solution_range_l1858_185859

theorem inequality_system_solution_range (x m : ℝ) :
  (∃ x : ℝ, (x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_range_l1858_185859


namespace NUMINAMATH_GPT_combination_indices_l1858_185838
open Nat

theorem combination_indices (x : ℕ) (h : choose 18 x = choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_combination_indices_l1858_185838


namespace NUMINAMATH_GPT_proof_problem_l1858_185818

-- Defining a right triangle ΔABC with ∠BCA=90°
structure RightTriangle :=
(a b c : ℝ)  -- sides a, b, c with c as the hypotenuse
(hypotenuse_eq : c^2 = a^2 + b^2)  -- Pythagorean relation

-- Define the circles K1 and K2 with radii r1 and r2 respectively
structure CirclesOnTriangle (Δ : RightTriangle) :=
(r1 r2 : ℝ)  -- radii of the circles K1 and K2

-- Prove the relationship r1 + r2 = a + b - c
theorem proof_problem (Δ : RightTriangle) (C : CirclesOnTriangle Δ) :
  C.r1 + C.r2 = Δ.a + Δ.b - Δ.c := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1858_185818


namespace NUMINAMATH_GPT_sin_of_cos_of_angle_l1858_185809

-- We need to assume that A is an angle of a triangle, hence A is in the range (0, π).
theorem sin_of_cos_of_angle (A : ℝ) (hA : 0 < A ∧ A < π) (h_cos : Real.cos A = -3/5) : Real.sin A = 4/5 := by
  sorry

end NUMINAMATH_GPT_sin_of_cos_of_angle_l1858_185809


namespace NUMINAMATH_GPT_find_term_ninth_term_l1858_185850

variable (a_1 d a_k a_12 : ℤ)
variable (S_20 : ℤ := 200)

-- Definitions of the given conditions
def term_n (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d

-- Problem Statement
theorem find_term_ninth_term :
  (∃ k, term_n a_1 d k + term_n a_1 d 12 = 20) ∧ 
  (S_20 = 10 * (2 * a_1 + 19 * d)) → 
  ∃ k, k = 9 :=
by sorry

end NUMINAMATH_GPT_find_term_ninth_term_l1858_185850


namespace NUMINAMATH_GPT_perpendicular_slope_l1858_185803

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l1858_185803


namespace NUMINAMATH_GPT_cakes_remaining_l1858_185817

theorem cakes_remaining (initial_cakes : ℕ) (bought_cakes : ℕ) (h1 : initial_cakes = 169) (h2 : bought_cakes = 137) : initial_cakes - bought_cakes = 32 :=
by
  sorry

end NUMINAMATH_GPT_cakes_remaining_l1858_185817


namespace NUMINAMATH_GPT_elaine_earnings_l1858_185893

variable (E P : ℝ)
variable (H1 : 0.30 * E * (1 + P / 100) = 2.025 * 0.20 * E)

theorem elaine_earnings : P = 35 :=
by
  -- We assume the conditions here and the proof is skipped by sorry.
  sorry

end NUMINAMATH_GPT_elaine_earnings_l1858_185893


namespace NUMINAMATH_GPT_p_and_q_work_together_l1858_185806

-- Given conditions
variable (Wp Wq : ℝ)

-- Condition that p is 50% more efficient than q
def efficiency_relation : Prop := Wp = 1.5 * Wq

-- Condition that p can complete the work in 25 days
def work_completion_by_p : Prop := Wp = 1 / 25

-- To be proved that p and q working together can complete the work in 15 days
theorem p_and_q_work_together (h1 : efficiency_relation Wp Wq)
                              (h2 : work_completion_by_p Wp) :
                              1 / (Wp + (Wp / 1.5)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_p_and_q_work_together_l1858_185806


namespace NUMINAMATH_GPT_length_XW_l1858_185807

theorem length_XW {XY XZ YZ XW : ℝ}
  (hXY : XY = 15)
  (hXZ : XZ = 17)
  (hAngle : XY^2 + YZ^2 = XZ^2)
  (hYZ : YZ = 8) :
  XW = 15 :=
by
  sorry

end NUMINAMATH_GPT_length_XW_l1858_185807


namespace NUMINAMATH_GPT_find_average_age_of_students_l1858_185848

-- Given conditions
variables (n : ℕ) (T : ℕ) (A : ℕ)

-- 20 students in the class
def students : ℕ := 20

-- Teacher's age is 42 years
def teacher_age : ℕ := 42

-- When the teacher's age is included, the average age increases by 1
def average_age_increase (A : ℕ) := A + 1

-- Proof problem statement in Lean 4
theorem find_average_age_of_students (A : ℕ) :
  20 * A + 42 = 21 * (A + 1) → A = 21 :=
by
  -- Here should be the proof steps, added sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_average_age_of_students_l1858_185848


namespace NUMINAMATH_GPT_find_W_l1858_185876

noncomputable def volume_of_space (r_sphere r_cylinder h_cylinder : ℝ) : ℝ :=
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h_cylinder
  let V_cone := (1 / 3) * Real.pi * r_cylinder^2 * h_cylinder
  V_sphere - V_cylinder - V_cone

theorem find_W : volume_of_space 6 4 10 = (224 / 3) * Real.pi := by
  sorry

end NUMINAMATH_GPT_find_W_l1858_185876


namespace NUMINAMATH_GPT_find_p_q_sum_l1858_185842

theorem find_p_q_sum (p q : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = 0 → 3 * x ^ 2 - p * x + q = 0) →
  p = 24 ∧ q = 45 ∧ p + q = 69 :=
by
  intros h
  have h3 := h 3 (by ring)
  have h5 := h 5 (by ring)
  sorry

end NUMINAMATH_GPT_find_p_q_sum_l1858_185842


namespace NUMINAMATH_GPT_solve_inequality_l1858_185871

theorem solve_inequality (x : ℝ) (h : x / 3 - 2 < 0) : x < 6 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1858_185871


namespace NUMINAMATH_GPT_bobby_candy_l1858_185891

theorem bobby_candy (C G : ℕ) (H : C + G = 36) (Hchoc: (2/3 : ℚ) * C = 12) (Hgummy: (3/4 : ℚ) * G = 9) : 
  (1/3 : ℚ) * C + (1/4 : ℚ) * G = 9 :=
by
  sorry

end NUMINAMATH_GPT_bobby_candy_l1858_185891


namespace NUMINAMATH_GPT_picnic_attendance_l1858_185832

theorem picnic_attendance (L x : ℕ) (h1 : L + x = 2015) (h2 : L - (x - 1) = 4) : x = 1006 := 
by
  sorry

end NUMINAMATH_GPT_picnic_attendance_l1858_185832


namespace NUMINAMATH_GPT_total_cans_collected_l1858_185870

theorem total_cans_collected (students_perez : ℕ) (half_perez_collected_20 : ℕ) (two_perez_collected_0 : ℕ) (remaining_perez_collected_8 : ℕ)
                             (students_johnson : ℕ) (third_johnson_collected_25 : ℕ) (three_johnson_collected_0 : ℕ) (remaining_johnson_collected_10 : ℕ)
                             (hp : students_perez = 28) (hc1 : half_perez_collected_20 = 28 / 2) (hc2 : two_perez_collected_0 = 2) (hc3 : remaining_perez_collected_8 = 12)
                             (hj : students_johnson = 30) (jc1 : third_johnson_collected_25 = 30 / 3) (jc2 : three_johnson_collected_0 = 3) (jc3 : remaining_johnson_collected_10 = 18) :
    (half_perez_collected_20 * 20 + two_perez_collected_0 * 0 + remaining_perez_collected_8 * 8
    + third_johnson_collected_25 * 25 + three_johnson_collected_0 * 0 + remaining_johnson_collected_10 * 10) = 806 :=
by
  sorry

end NUMINAMATH_GPT_total_cans_collected_l1858_185870


namespace NUMINAMATH_GPT_num_real_roots_of_abs_x_eq_l1858_185857

theorem num_real_roots_of_abs_x_eq (k : ℝ) (hk : 6 < k ∧ k < 7) 
  : (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (|x1| * x1 - 2 * x1 + 7 - k = 0) ∧ 
    (|x2| * x2 - 2 * x2 + 7 - k = 0) ∧
    (|x3| * x3 - 2 * x3 + 7 - k = 0)) ∧
  (¬ ∃ x4 : ℝ, x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ |x4| * x4 - 2 * x4 + 7 - k = 0) :=
sorry

end NUMINAMATH_GPT_num_real_roots_of_abs_x_eq_l1858_185857


namespace NUMINAMATH_GPT_percentage_increase_l1858_185800

theorem percentage_increase (a : ℕ) (x : ℝ) (b : ℝ) (r : ℝ) 
    (h1 : a = 1500) 
    (h2 : r = 0.6) 
    (h3 : b = 1080) 
    (h4 : a * (1 + x / 100) * r = b) : 
    x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l1858_185800


namespace NUMINAMATH_GPT_minimum_cost_l1858_185880

theorem minimum_cost (price_pen_A price_pen_B price_notebook_A price_notebook_B : ℕ) 
  (discount_B : ℚ) (num_pens num_notebooks : ℕ)
  (h_price_pen : price_pen_A = 10) (h_price_notebook : price_notebook_A = 2)
  (h_discount : discount_B = 0.9) (h_num_pens : num_pens = 4) (h_num_notebooks : num_notebooks = 24) :
  ∃ (min_cost : ℕ), min_cost = 76 :=
by
  -- The conditions should be used here to construct the min_cost
  sorry

end NUMINAMATH_GPT_minimum_cost_l1858_185880


namespace NUMINAMATH_GPT_rewrite_neg_multiplication_as_exponent_l1858_185875

theorem rewrite_neg_multiplication_as_exponent :
  -2 * 2 * 2 * 2 = - (2^4) :=
by
  sorry

end NUMINAMATH_GPT_rewrite_neg_multiplication_as_exponent_l1858_185875


namespace NUMINAMATH_GPT_game_points_l1858_185829

noncomputable def total_points (total_enemies : ℕ) (red_enemies : ℕ) (blue_enemies : ℕ) 
  (enemies_defeated : ℕ) (points_per_enemy : ℕ) (bonus_points : ℕ) 
  (hits_taken : ℕ) (points_lost_per_hit : ℕ) : ℕ :=
  (enemies_defeated * points_per_enemy + if enemies_defeated > 0 ∧ enemies_defeated < total_enemies then bonus_points else 0) - (hits_taken * points_lost_per_hit)

theorem game_points (h : total_points 6 3 3 4 3 5 2 2 = 13) : Prop := sorry

end NUMINAMATH_GPT_game_points_l1858_185829


namespace NUMINAMATH_GPT_purple_marble_probability_l1858_185820

theorem purple_marble_probability (P_blue P_green P_purple : ℝ) (h1 : P_blue = 0.35) (h2 : P_green = 0.45) (h3 : P_blue + P_green + P_purple = 1) :
  P_purple = 0.2 := 
by sorry

end NUMINAMATH_GPT_purple_marble_probability_l1858_185820


namespace NUMINAMATH_GPT_suitable_survey_set_l1858_185816

def Survey1 := "Investigate the lifespan of a batch of light bulbs"
def Survey2 := "Investigate the household income situation in a city"
def Survey3 := "Investigate the vision of students in a class"
def Survey4 := "Investigate the efficacy of a certain drug"

-- Define what it means for a survey to be suitable for sample surveys
def suitable_for_sample_survey (survey : String) : Prop :=
  survey = Survey1 ∨ survey = Survey2 ∨ survey = Survey4

-- The question is to prove that the surveys suitable for sample surveys include exactly (1), (2), and (4).
theorem suitable_survey_set :
  {Survey1, Survey2, Survey4} = {s : String | suitable_for_sample_survey s} :=
by
  sorry

end NUMINAMATH_GPT_suitable_survey_set_l1858_185816


namespace NUMINAMATH_GPT_sum_smallest_largest_2y_l1858_185835

variable (a n y : ℤ)

noncomputable def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
noncomputable def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

theorem sum_smallest_largest_2y 
  (h1 : is_odd a) 
  (h2 : n % 2 = 0) 
  (h3 : y = a + n) : 
  a + (a + 2 * n) = 2 * y := 
by 
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_2y_l1858_185835


namespace NUMINAMATH_GPT_total_gray_trees_l1858_185805

theorem total_gray_trees :
  (∃ trees_first trees_second trees_third gray1 gray2,
    trees_first = 100 ∧
    trees_second = 90 ∧
    trees_third = 82 ∧
    gray1 = trees_first - trees_third ∧
    gray2 = trees_second - trees_third ∧
    trees_first + trees_second - 2 * trees_third = gray1 + gray2) →
  (gray1 + gray2 = 26) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_gray_trees_l1858_185805


namespace NUMINAMATH_GPT_count_of_green_hats_l1858_185846

-- Defining the total number of hats
def total_hats : ℕ := 85

-- Defining the costs of each hat type
def blue_cost : ℕ := 6
def green_cost : ℕ := 7
def red_cost : ℕ := 8

-- Defining the total cost
def total_cost : ℕ := 600

-- Defining the ratio as 3:2:1
def ratio_blue : ℕ := 3
def ratio_green : ℕ := 2
def ratio_red : ℕ := 1

-- Defining the multiplication factor
def x : ℕ := 14

-- Number of green hats based on the ratio
def G : ℕ := ratio_green * x

-- Proving that we bought 28 green hats
theorem count_of_green_hats : G = 28 := by
  -- proof steps intention: sorry to skip the proof
  sorry

end NUMINAMATH_GPT_count_of_green_hats_l1858_185846


namespace NUMINAMATH_GPT_krishan_money_l1858_185898

theorem krishan_money 
  (R G K : ℝ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 490) : K = 2890 :=
sorry

end NUMINAMATH_GPT_krishan_money_l1858_185898


namespace NUMINAMATH_GPT_max_regular_hours_correct_l1858_185840

-- Define the conditions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_hours_worked : ℝ := 57
def total_compensation : ℝ := 1116

-- Define the maximum regular hours per week
def max_regular_hours : ℝ := 40

-- Define the compensation equation
def compensation (H : ℝ) : ℝ :=
  regular_rate * H + overtime_rate * (total_hours_worked - H)

-- The theorem that needs to be proved
theorem max_regular_hours_correct :
  compensation max_regular_hours = total_compensation :=
by
  -- skolemize the proof
  sorry

end NUMINAMATH_GPT_max_regular_hours_correct_l1858_185840


namespace NUMINAMATH_GPT_gcf_84_112_210_l1858_185854

theorem gcf_84_112_210 : gcd (gcd 84 112) 210 = 14 := by sorry

end NUMINAMATH_GPT_gcf_84_112_210_l1858_185854


namespace NUMINAMATH_GPT_probability_of_drawing_letter_in_name_l1858_185811

theorem probability_of_drawing_letter_in_name :
  let total_letters := 26
  let alonso_letters := ['a', 'l', 'o', 'n', 's']
  let number_of_alonso_letters := alonso_letters.length
  number_of_alonso_letters / total_letters = 5 / 26 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_letter_in_name_l1858_185811


namespace NUMINAMATH_GPT_total_amount_shared_l1858_185821

theorem total_amount_shared (J Jo B : ℝ) (r1 r2 r3 : ℝ)
  (H1 : r1 = 2) (H2 : r2 = 4) (H3 : r3 = 6) (H4 : J = 1600) (part_value : ℝ)
  (H5 : part_value = J / r1) (H6 : Jo = r2 * part_value) (H7 : B = r3 * part_value) :
  J + Jo + B = 9600 :=
sorry

end NUMINAMATH_GPT_total_amount_shared_l1858_185821


namespace NUMINAMATH_GPT_find_f_at_one_l1858_185878

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4 * x ^ 2 - m * x + 5

theorem find_f_at_one :
  (∀ x : ℝ, x ≥ -2 → f x (-16) ≥ f (-2) (-16)) ∧
  (∀ x : ℝ, x ≤ -2 → f x (-16) ≤ f (-2) (-16)) →
  f 1 (-16) = 25 :=
sorry

end NUMINAMATH_GPT_find_f_at_one_l1858_185878


namespace NUMINAMATH_GPT_expand_product_l1858_185892

theorem expand_product (x : ℝ) :
  (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_expand_product_l1858_185892


namespace NUMINAMATH_GPT_weight_of_each_bag_of_planks_is_14_l1858_185849

-- Definitions
def crate_capacity : Nat := 20
def num_crates : Nat := 15
def num_bags_nails : Nat := 4
def weight_bag_nails : Nat := 5
def num_bags_hammers : Nat := 12
def weight_bag_hammers : Nat := 5
def num_bags_planks : Nat := 10
def weight_to_leave_out : Nat := 80

-- Total weight calculations
def weight_nails := num_bags_nails * weight_bag_nails
def weight_hammers := num_bags_hammers * weight_bag_hammers
def total_weight_nails_hammers := weight_nails + weight_hammers
def total_crate_capacity := num_crates * crate_capacity
def weight_that_can_be_loaded := total_crate_capacity - weight_to_leave_out
def weight_available_for_planks := weight_that_can_be_loaded - total_weight_nails_hammers
def weight_each_bag_planks := weight_available_for_planks / num_bags_planks

-- Theorem statement
theorem weight_of_each_bag_of_planks_is_14 : weight_each_bag_planks = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_each_bag_of_planks_is_14_l1858_185849


namespace NUMINAMATH_GPT_root_of_equation_value_l1858_185894

theorem root_of_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2 * m^2 - 4 * m + 5 = 11 := 
by
  sorry

end NUMINAMATH_GPT_root_of_equation_value_l1858_185894


namespace NUMINAMATH_GPT_sufficient_drivers_and_ivan_petrovich_departure_l1858_185886

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_drivers_and_ivan_petrovich_departure_l1858_185886


namespace NUMINAMATH_GPT_total_number_of_students_l1858_185865

namespace StudentRanking

def rank_from_right := 17
def rank_from_left := 5
def total_students (rank_from_right rank_from_left : ℕ) := rank_from_right + rank_from_left - 1

theorem total_number_of_students : total_students rank_from_right rank_from_left = 21 :=
by
  sorry

end StudentRanking

end NUMINAMATH_GPT_total_number_of_students_l1858_185865


namespace NUMINAMATH_GPT_money_received_from_mom_l1858_185823

-- Define the given conditions
def initial_amount : ℕ := 48
def amount_spent : ℕ := 11
def amount_after_getting_money : ℕ := 58
def amount_left_after_spending : ℕ := initial_amount - amount_spent

-- Define the proof statement
theorem money_received_from_mom : (amount_after_getting_money - amount_left_after_spending) = 21 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_money_received_from_mom_l1858_185823


namespace NUMINAMATH_GPT_range_of_distance_l1858_185819

noncomputable def A (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def B (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

theorem range_of_distance (α β : ℝ) :
  1 ≤ Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ∧
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_distance_l1858_185819


namespace NUMINAMATH_GPT_calculation_of_cube_exponent_l1858_185866

theorem calculation_of_cube_exponent (a : ℤ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end NUMINAMATH_GPT_calculation_of_cube_exponent_l1858_185866


namespace NUMINAMATH_GPT_bed_length_l1858_185808

noncomputable def volume (length width height : ℝ) : ℝ :=
  length * width * height

theorem bed_length
  (width height : ℝ)
  (bags_of_soil soil_volume_per_bag total_volume : ℝ)
  (needed_bags : ℝ)
  (L : ℝ) :
  width = 4 →
  height = 1 →
  needed_bags = 16 →
  soil_volume_per_bag = 4 →
  total_volume = needed_bags * soil_volume_per_bag →
  total_volume = 2 * volume L width height →
  L = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bed_length_l1858_185808


namespace NUMINAMATH_GPT_number_increased_by_one_fourth_l1858_185841

theorem number_increased_by_one_fourth (n : ℕ) (h : 25 * 80 / 100 = 20) (h1 : 80 - 20 = 60) :
  n + n / 4 = 60 ↔ n = 48 :=
by
  -- Conditions
  have h2 : 80 - 25 * 80 / 100 = 60 := by linarith [h, h1]
  have h3 : n + n / 4 = 60 := sorry
  -- Assertion (Proof to show is omitted)
  sorry

end NUMINAMATH_GPT_number_increased_by_one_fourth_l1858_185841


namespace NUMINAMATH_GPT_units_digit_7_pow_2023_l1858_185837

-- Define a function to compute the units digit
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Main theorem to prove: the units digit of 7^2023 equals 3
theorem units_digit_7_pow_2023 : units_digit (7^2023) = 3 :=
by {
  -- It suffices to show that the units digits of powers of 7 repeat every 4 with the pattern [7, 9, 3, 1]
  sorry
}

end NUMINAMATH_GPT_units_digit_7_pow_2023_l1858_185837


namespace NUMINAMATH_GPT_hoseoks_social_studies_score_l1858_185847

theorem hoseoks_social_studies_score 
  (avg_three_subjects : ℕ) 
  (new_avg_with_social_studies : ℕ) 
  (total_score_three_subjects : ℕ) 
  (total_score_four_subjects : ℕ) 
  (S : ℕ)
  (h1 : avg_three_subjects = 89) 
  (h2 : new_avg_with_social_studies = 90) 
  (h3 : total_score_three_subjects = 3 * avg_three_subjects) 
  (h4 : total_score_four_subjects = 4 * new_avg_with_social_studies) :
  S = 93 :=
sorry

end NUMINAMATH_GPT_hoseoks_social_studies_score_l1858_185847


namespace NUMINAMATH_GPT_minimum_value_of_f_l1858_185862

noncomputable def f (x : ℝ) := 2 * x + 18 / x

theorem minimum_value_of_f :
  ∃ x > 0, f x = 12 ∧ ∀ y > 0, f y ≥ 12 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1858_185862


namespace NUMINAMATH_GPT_find_a2_l1858_185851

-- Define the geometric sequence and its properties
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions 
variables (a : ℕ → ℝ) (h_geom : is_geometric a)
variables (h_a1 : a 1 = 1/4)
variables (h_condition : a 3 * a 5 = 4 * (a 4 - 1))

-- The goal is to prove a 2 = 1/2
theorem find_a2 : a 2 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_l1858_185851


namespace NUMINAMATH_GPT_fencing_required_l1858_185826

theorem fencing_required
  (L : ℝ) (A : ℝ) (h_L : L = 20) (h_A : A = 400) : 
  (2 * (A / L) + L) = 60 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l1858_185826


namespace NUMINAMATH_GPT_part_a_l1858_185855

theorem part_a (x : ℝ) (hx : x ≥ 1) : x^3 - 5 * x^2 + 8 * x - 4 ≥ 0 := 
  sorry

end NUMINAMATH_GPT_part_a_l1858_185855


namespace NUMINAMATH_GPT_jimin_yuna_difference_l1858_185895

-- Definitions based on the conditions.
def seokjin_marbles : ℕ := 3
def yuna_marbles : ℕ := seokjin_marbles - 1
def jimin_marbles : ℕ := seokjin_marbles * 2

-- Theorem stating the problem we need to prove: the difference in marbles between Jimin and Yuna is 4.
theorem jimin_yuna_difference : jimin_marbles - yuna_marbles = 4 :=
by sorry

end NUMINAMATH_GPT_jimin_yuna_difference_l1858_185895


namespace NUMINAMATH_GPT_frogs_meet_time_proven_l1858_185860

-- Define the problem
def frogs_will_meet_at_time : Prop :=
  ∃ (meet_time : Nat),
    let initial_time := 12 * 60 -- 12:00 PM in minutes
    let initial_distance := 2015
    let green_frog_jump := 9
    let blue_frog_jump := 8 
    let combined_reduction := green_frog_jump + blue_frog_jump
    initial_distance % combined_reduction = 0 ∧
    meet_time == initial_time + (2 * (initial_distance / combined_reduction))

theorem frogs_meet_time_proven (h : frogs_will_meet_at_time) : meet_time = 15 * 60 + 56 :=
sorry

end NUMINAMATH_GPT_frogs_meet_time_proven_l1858_185860


namespace NUMINAMATH_GPT_M_subset_N_l1858_185814

def M (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 2) + (Real.pi / 4)
def N (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 4) + (Real.pi / 2)

theorem M_subset_N : ∀ x, M x → N x := 
by
  sorry

end NUMINAMATH_GPT_M_subset_N_l1858_185814


namespace NUMINAMATH_GPT_greatest_x_lcm_105_l1858_185824

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end NUMINAMATH_GPT_greatest_x_lcm_105_l1858_185824


namespace NUMINAMATH_GPT_Annika_three_times_Hans_in_future_l1858_185856

theorem Annika_three_times_Hans_in_future
  (hans_age_now : Nat)
  (annika_age_now : Nat)
  (x : Nat)
  (hans_future_age : Nat)
  (annika_future_age : Nat)
  (H1 : hans_age_now = 8)
  (H2 : annika_age_now = 32)
  (H3 : hans_future_age = hans_age_now + x)
  (H4 : annika_future_age = annika_age_now + x)
  (H5 : annika_future_age = 3 * hans_future_age) :
  x = 4 := 
  by
  sorry

end NUMINAMATH_GPT_Annika_three_times_Hans_in_future_l1858_185856
