import Mathlib

namespace NUMINAMATH_GPT_tan_alpha_value_l781_78126

open Real

-- Define the angle alpha in the third quadrant
variable {α : ℝ}

-- Given conditions
def third_quadrant (α : ℝ) : Prop :=  π < α ∧ α < 3 * π / 2
def sin_alpha (α : ℝ) : Prop := sin α = -4 / 5

-- Statement to prove
theorem tan_alpha_value (h1 : third_quadrant α) (h2 : sin_alpha α) : tan α = 4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l781_78126


namespace NUMINAMATH_GPT_area_of_right_triangle_l781_78116

-- Define a structure for the triangle with the given conditions
structure Triangle :=
(A B C : ℝ × ℝ)
(right_angle_at_C : (C.1 = 0 ∧ C.2 = 0))
(hypotenuse_length : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 50 ^ 2)
(median_A : ∀ x: ℝ, A.2 = A.1 + 5)
(median_B : ∀ x: ℝ, B.2 = 2 * B.1 + 2)

-- Theorem statement
theorem area_of_right_triangle (t : Triangle) : 
  ∃ area : ℝ, area = 500 :=
sorry

end NUMINAMATH_GPT_area_of_right_triangle_l781_78116


namespace NUMINAMATH_GPT_loss_equates_to_balls_l781_78199

theorem loss_equates_to_balls
    (SP_20 : ℕ) (CP_1: ℕ) (Loss: ℕ) (x: ℕ)
    (h1 : SP_20 = 720)
    (h2 : CP_1 = 48)
    (h3 : Loss = (20 * CP_1 - SP_20))
    (h4 : Loss = x * CP_1) :
    x = 5 :=
by
  sorry

end NUMINAMATH_GPT_loss_equates_to_balls_l781_78199


namespace NUMINAMATH_GPT_largest_both_writers_editors_l781_78184

-- Define the conditions
def writers : ℕ := 45
def editors_gt : ℕ := 38
def total_attendees : ℕ := 90
def both_writers_editors (x : ℕ) : ℕ := x
def neither_writers_editors (x : ℕ) : ℕ := x / 2

-- Define the main proof statement
theorem largest_both_writers_editors :
  ∃ x : ℕ, x ≤ 4 ∧
  (writers + (editors_gt + (0 : ℕ)) + neither_writers_editors x + both_writers_editors x = total_attendees) :=
sorry

end NUMINAMATH_GPT_largest_both_writers_editors_l781_78184


namespace NUMINAMATH_GPT_price_of_books_sold_at_lower_price_l781_78134

-- Define the conditions
variable (n m p q t : ℕ) (earnings price_high price_low : ℝ)

-- The given conditions
def total_books : ℕ := 10
def books_high_price : ℕ := 2 * total_books / 5 -- 2/5 of total books
def books_low_price : ℕ := total_books - books_high_price
def high_price : ℝ := 2.50
def total_earnings : ℝ := 22

-- The proposition to prove
theorem price_of_books_sold_at_lower_price
  (h_books_high_price : books_high_price = 4)
  (h_books_low_price : books_low_price = 6)
  (h_total_earnings : total_earnings = 22)
  (h_high_price : high_price = 2.50) :
  (price_low = 2) := 
-- Proof goes here 
sorry

end NUMINAMATH_GPT_price_of_books_sold_at_lower_price_l781_78134


namespace NUMINAMATH_GPT_neg_p_l781_78140

open Real

variable {f : ℝ → ℝ}

theorem neg_p :
  (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end NUMINAMATH_GPT_neg_p_l781_78140


namespace NUMINAMATH_GPT_photographer_max_photos_l781_78162

-- The initial number of birds of each species
def total_birds : ℕ := 20
def starlings : ℕ := 8
def wagtails : ℕ := 7
def woodpeckers : ℕ := 5

-- Define a function to count the remaining birds of each species after n photos
def remaining_birds (n : ℕ) (species : ℕ) : ℕ := species - (if species ≤ n then species else n)

-- Define the main theorem we want to prove
theorem photographer_max_photos (n : ℕ) (h1 : remaining_birds n starlings ≥ 4) (h2 : remaining_birds n wagtails ≥ 3) : 
  n ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_photographer_max_photos_l781_78162


namespace NUMINAMATH_GPT_binary_div_mul_l781_78104

-- Define the binary numbers
def a : ℕ := 0b101110
def b : ℕ := 0b110100
def c : ℕ := 0b110

-- Statement to prove the given problem
theorem binary_div_mul : (a * b) / c = 0b101011100 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_binary_div_mul_l781_78104


namespace NUMINAMATH_GPT_initial_salt_percentage_l781_78101

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end NUMINAMATH_GPT_initial_salt_percentage_l781_78101


namespace NUMINAMATH_GPT_function_equiv_proof_l781_78138

noncomputable def function_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

theorem function_equiv_proof : ∀ f : ℝ → ℝ,
  function_solution f ↔ (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = -x) := 
sorry

end NUMINAMATH_GPT_function_equiv_proof_l781_78138


namespace NUMINAMATH_GPT_birds_initially_sitting_l781_78156

theorem birds_initially_sitting (initial_birds birds_joined total_birds : ℕ) 
  (h1 : birds_joined = 4) (h2 : total_birds = 6) (h3 : total_birds = initial_birds + birds_joined) : 
  initial_birds = 2 :=
by
  sorry

end NUMINAMATH_GPT_birds_initially_sitting_l781_78156


namespace NUMINAMATH_GPT_ratio_of_lost_diaries_to_total_diaries_l781_78167

theorem ratio_of_lost_diaries_to_total_diaries 
  (original_diaries : ℕ)
  (bought_diaries : ℕ)
  (current_diaries : ℕ)
  (h1 : original_diaries = 8)
  (h2 : bought_diaries = 2 * original_diaries)
  (h3 : current_diaries = 18) :
  (original_diaries + bought_diaries - current_diaries) / gcd (original_diaries + bought_diaries - current_diaries) (original_diaries + bought_diaries) 
  = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_lost_diaries_to_total_diaries_l781_78167


namespace NUMINAMATH_GPT_find_largest_m_l781_78105

variables (a b c t : ℝ)
def f (x : ℝ) := a * x^2 + b * x + c

theorem find_largest_m (a_ne_zero : a ≠ 0)
  (cond1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x) ∧ f a b c x ≥ x)
  (cond2 : ∀ x : ℝ, 0 < x ∧ x < 2 → f a b c x ≤ ((x + 1) / 2)^2)
  (cond3 : ∃ x : ℝ, ∀ y : ℝ, f a b c y ≥ f a b c x ∧ f a b c x = 0) :
  ∃ m : ℝ, 1 < m ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x) ∧ m = 9 := sorry

end NUMINAMATH_GPT_find_largest_m_l781_78105


namespace NUMINAMATH_GPT_diagonals_diff_heptagon_octagon_l781_78151

-- Define the function to calculate the number of diagonals in a polygon with n sides
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_diff_heptagon_octagon : 
  let A := num_diagonals 7
  let B := num_diagonals 8
  B - A = 6 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_diff_heptagon_octagon_l781_78151


namespace NUMINAMATH_GPT_roots_polynomial_l781_78109

theorem roots_polynomial (n r s : ℚ) (c d : ℚ)
  (h1 : c * c - n * c + 3 = 0)
  (h2 : d * d - n * d + 3 = 0)
  (h3 : (c + 1/d) * (d + 1/c) = s)
  (h4 : c * d = 3) :
  s = 16/3 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_l781_78109


namespace NUMINAMATH_GPT_sum_area_of_R_eq_20_l781_78145

noncomputable def sum_m_n : ℝ := 
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  let m := 20
  let n := 12 * Real.sqrt 2
  m + n

theorem sum_area_of_R_eq_20 :
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  area_R = 20 + 12 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_area_of_R_eq_20_l781_78145


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l781_78197

-- Define the repeating decimal
def repeating_decimal := 3 + (127 / 999)

-- State the goal
theorem repeating_decimal_as_fraction : repeating_decimal = (3124 / 999) := 
by 
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l781_78197


namespace NUMINAMATH_GPT_double_variable_for_1600_percent_cost_l781_78114

theorem double_variable_for_1600_percent_cost (t b0 b1 : ℝ) (h : t ≠ 0) :
    (t * b1^4 = 16 * t * b0^4) → b1 = 2 * b0 :=
by
sorry

end NUMINAMATH_GPT_double_variable_for_1600_percent_cost_l781_78114


namespace NUMINAMATH_GPT_max_books_l781_78177

theorem max_books (cost_per_book : ℝ) (total_money : ℝ) (h_cost : cost_per_book = 8.75) (h_money : total_money = 250.0) :
  ∃ n : ℕ, n = 28 ∧ cost_per_book * n ≤ total_money ∧ ∀ m : ℕ, cost_per_book * m ≤ total_money → m ≤ 28 :=
by
  sorry

end NUMINAMATH_GPT_max_books_l781_78177


namespace NUMINAMATH_GPT_a_10_eq_18_l781_78112

variable {a : ℕ → ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a2 : a 2 = 2
axiom a3 : a 3 = 4
axiom arithmetic_seq : is_arithmetic_sequence a

-- problem: prove a_{10} = 18
theorem a_10_eq_18 : a 10 = 18 :=
sorry

end NUMINAMATH_GPT_a_10_eq_18_l781_78112


namespace NUMINAMATH_GPT_no_nat_k_divides_7_l781_78155

theorem no_nat_k_divides_7 (k : ℕ) : ¬ 7 ∣ (2^(2*k - 1) + 2^k + 1) := 
sorry

end NUMINAMATH_GPT_no_nat_k_divides_7_l781_78155


namespace NUMINAMATH_GPT_find_x_l781_78157

variable {a b x : ℝ}

-- Defining the given conditions
def is_linear_and_unique_solution (a b : ℝ) : Prop :=
  3 * a + 2 * b = 0 ∧ a ≠ 0

-- The proof problem: prove that x = 1.5, given the conditions.
theorem find_x (ha : is_linear_and_unique_solution a b) : x = 1.5 :=
  sorry

end NUMINAMATH_GPT_find_x_l781_78157


namespace NUMINAMATH_GPT_curve_has_axis_of_symmetry_l781_78160

theorem curve_has_axis_of_symmetry (x y : ℝ) :
  (x^2 - x * y + y^2 + x - y - 1 = 0) ↔ (x+y = 0) :=
sorry

end NUMINAMATH_GPT_curve_has_axis_of_symmetry_l781_78160


namespace NUMINAMATH_GPT_find_b_in_triangle_l781_78142

-- Given conditions
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a = 3)
variable (h2 : c = 2 * Real.sqrt 3)
variable (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6))

-- The proof goal
theorem find_b_in_triangle (h1 : a = 3) (h2 : c = 2 * Real.sqrt 3) (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6)) : b = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_b_in_triangle_l781_78142


namespace NUMINAMATH_GPT_f_monotone_decreasing_without_min_value_l781_78154

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_without_min_value :
  (∀ x y : ℝ, x < y → f y < f x) ∧ (∃ b : ℝ, ∀ x : ℝ, f x > b) :=
by
  sorry

end NUMINAMATH_GPT_f_monotone_decreasing_without_min_value_l781_78154


namespace NUMINAMATH_GPT_car_late_speed_l781_78152

theorem car_late_speed :
  ∀ (d : ℝ) (t_on_time : ℝ) (t_late : ℝ) (v_on_time : ℝ) (v_late : ℝ),
  d = 225 →
  v_on_time = 60 →
  t_on_time = d / v_on_time →
  t_late = t_on_time + 0.75 →
  v_late = d / t_late →
  v_late = 50 :=
by
  intros d t_on_time t_late v_on_time v_late hd hv_on_time ht_on_time ht_late hv_late
  sorry

end NUMINAMATH_GPT_car_late_speed_l781_78152


namespace NUMINAMATH_GPT_divided_scale_length_l781_78173

/-
  The problem definition states that we have a scale that is 6 feet 8 inches long, 
  and we need to prove that when the scale is divided into two equal parts, 
  each part is 3 feet 4 inches long.
-/

/-- Given length conditions in feet and inches --/
def total_length_feet : ℕ := 6
def total_length_inches : ℕ := 8

/-- Convert total length to inches --/
def total_length_in_inches := total_length_feet * 12 + total_length_inches

/-- Proof that if a scale is 6 feet 8 inches long and divided into 2 parts, each part is 3 feet 4 inches --/
theorem divided_scale_length :
  (total_length_in_inches / 2) = 40 ∧ (40 / 12 = 3 ∧ 40 % 12 = 4) :=
by
  sorry

end NUMINAMATH_GPT_divided_scale_length_l781_78173


namespace NUMINAMATH_GPT_library_charge_l781_78196

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end NUMINAMATH_GPT_library_charge_l781_78196


namespace NUMINAMATH_GPT_correct_statements_l781_78166

def f (x : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d
def f_prime (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem correct_statements (b c d : ℝ) :
  (∃ x : ℝ, f x b c d = 4 ∧ f_prime x b c = 0) ∧
  (∃ x : ℝ, f x b c d = 0 ∧ f_prime x b c = 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l781_78166


namespace NUMINAMATH_GPT_tenth_term_of_geometric_sequence_l781_78185

theorem tenth_term_of_geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) (tenth_term : ℚ) :
  a = 5 →
  r = 4 / 3 →
  n = 10 →
  tenth_term = a * r ^ (n - 1) →
  tenth_term = 1310720 / 19683 :=
by sorry

end NUMINAMATH_GPT_tenth_term_of_geometric_sequence_l781_78185


namespace NUMINAMATH_GPT_sufficient_condition_l781_78189

theorem sufficient_condition (x y : ℤ) (h : x + y ≠ 2) : x ≠ 1 ∧ y ≠ 1 := 
sorry

end NUMINAMATH_GPT_sufficient_condition_l781_78189


namespace NUMINAMATH_GPT_square_root_25_pm5_l781_78190

-- Define that a number x satisfies the equation x^2 = 25
def square_root_of_25 (x : ℝ) : Prop := x * x = 25

-- The theorem states that the square root of 25 is ±5
theorem square_root_25_pm5 : ∀ x : ℝ, square_root_of_25 x ↔ x = 5 ∨ x = -5 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_square_root_25_pm5_l781_78190


namespace NUMINAMATH_GPT_common_factor_l781_78122

-- Define the polynomials
def P1 (x : ℝ) : ℝ := x^3 + x^2
def P2 (x : ℝ) : ℝ := x^2 + 2*x + 1
def P3 (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem common_factor (x : ℝ) : ∃ (f : ℝ → ℝ), (f x = x + 1) ∧ (∃ g1 g2 g3 : ℝ → ℝ, P1 x = f x * g1 x ∧ P2 x = f x * g2 x ∧ P3 x = f x * g3 x) :=
sorry

end NUMINAMATH_GPT_common_factor_l781_78122


namespace NUMINAMATH_GPT_river_flow_volume_l781_78107

/-- Given a river depth of 7 meters, width of 75 meters, 
and flow rate of 4 kilometers per hour,
the volume of water running into the sea per minute 
is 35,001.75 cubic meters. -/
theorem river_flow_volume
  (depth : ℝ) (width : ℝ) (rate_kmph : ℝ)
  (depth_val : depth = 7)
  (width_val : width = 75)
  (rate_val : rate_kmph = 4) :
  ( width * depth * (rate_kmph * 1000 / 60) ) = 35001.75 :=
by
  rw [depth_val, width_val, rate_val]
  sorry

end NUMINAMATH_GPT_river_flow_volume_l781_78107


namespace NUMINAMATH_GPT_find_a_l781_78165

-- Define the function f
def f (a x : ℝ) := a * x^3 - 2 * x

-- State the theorem, asserting that if f passes through the point (-1, 4) then a = -2.
theorem find_a (a : ℝ) (h : f a (-1) = 4) : a = -2 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_a_l781_78165


namespace NUMINAMATH_GPT_root_of_function_is_four_l781_78161

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_function_is_four (a : ℝ) (h : f a = 0) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_root_of_function_is_four_l781_78161


namespace NUMINAMATH_GPT_remainder_of_6_pow_1234_mod_13_l781_78163

theorem remainder_of_6_pow_1234_mod_13 : 6 ^ 1234 % 13 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_6_pow_1234_mod_13_l781_78163


namespace NUMINAMATH_GPT_geometric_sequence_a5_l781_78120

variable (a : ℕ → ℝ) (q : ℝ)

axiom pos_terms : ∀ n, a n > 0

axiom a1a3_eq : a 1 * a 3 = 16
axiom a3a4_eq : a 3 + a 4 = 24

theorem geometric_sequence_a5 :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) → a 5 = 32 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l781_78120


namespace NUMINAMATH_GPT_find_sam_current_age_l781_78121

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end NUMINAMATH_GPT_find_sam_current_age_l781_78121


namespace NUMINAMATH_GPT_charlies_mother_cookies_l781_78144

theorem charlies_mother_cookies 
    (charlie_cookies : ℕ) 
    (father_cookies : ℕ) 
    (total_cookies : ℕ)
    (h_charlie : charlie_cookies = 15)
    (h_father : father_cookies = 10)
    (h_total : total_cookies = 30) : 
    (total_cookies - charlie_cookies - father_cookies = 5) :=
by {
    sorry
}

end NUMINAMATH_GPT_charlies_mother_cookies_l781_78144


namespace NUMINAMATH_GPT_angle_A_range_l781_78127

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x < y ∧ x ∈ I ∧ y ∈ I → f x < f y

theorem angle_A_range (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_strict_inc : strictly_increasing f {x | 0 < x})
  (h_f_half : f (1 / 2) = 0)
  (A : ℝ)
  (h_cos_A : f (Real.cos A) < 0) :
  (π / 3 < A ∧ A < π / 2) ∨ (2 * π / 3 < A ∧ A < π) :=
by
  sorry

end NUMINAMATH_GPT_angle_A_range_l781_78127


namespace NUMINAMATH_GPT_last_three_digits_7_pow_123_l781_78175

theorem last_three_digits_7_pow_123 : (7^123 % 1000) = 717 := sorry

end NUMINAMATH_GPT_last_three_digits_7_pow_123_l781_78175


namespace NUMINAMATH_GPT_wine_age_problem_l781_78143

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end NUMINAMATH_GPT_wine_age_problem_l781_78143


namespace NUMINAMATH_GPT_dodgeball_tournament_l781_78182

theorem dodgeball_tournament (N : ℕ) (points : ℕ) :
  points = 1151 →
  (∀ {G : ℕ}, G = N * (N - 1) / 2 →
    (∃ (win_points loss_points tie_points : ℕ), 
      win_points = 15 * (N * (N - 1) / 2 - tie_points) ∧ 
      tie_points = 11 * tie_points ∧ 
      points = win_points + tie_points + loss_points)) → 
  N = 12 :=
by
  intro h_points h_games
  sorry

end NUMINAMATH_GPT_dodgeball_tournament_l781_78182


namespace NUMINAMATH_GPT_square_perimeter_calculation_l781_78146

noncomputable def perimeter_of_square (radius: ℝ) : ℝ := 
  if radius = 4 then 64 * Real.sqrt 2 else 0

theorem square_perimeter_calculation :
  perimeter_of_square 4 = 64 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_calculation_l781_78146


namespace NUMINAMATH_GPT_triangle_inequality_l781_78132

theorem triangle_inequality (x : ℕ) (hx : x > 0) :
  (x ≥ 34) ↔ (x + (10 + x) > 24) ∧ (x + 24 > 10 + x) ∧ ((10 + x) + 24 > x) := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l781_78132


namespace NUMINAMATH_GPT_factorize_expression_l781_78108

theorem factorize_expression (a : ℝ) : 
  (a + 1) * (a + 2) + 1 / 4 = (a + 3 / 2)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l781_78108


namespace NUMINAMATH_GPT_anne_speed_l781_78195

-- Conditions
def time_hours : ℝ := 3
def distance_miles : ℝ := 6

-- Question with correct answer
theorem anne_speed : distance_miles / time_hours = 2 := by 
  sorry

end NUMINAMATH_GPT_anne_speed_l781_78195


namespace NUMINAMATH_GPT_findB_coords_l781_78117

namespace ProofProblem

-- Define point A with its coordinates.
def A : ℝ × ℝ := (-3, 2)

-- Define a property that checks if a line segment AB is parallel to the x-axis.
def isParallelToXAxis (A B : (ℝ × ℝ)) : Prop :=
  A.2 = B.2

-- Define a property that checks if the length of line segment AB is 4.
def hasLengthFour (A B : (ℝ × ℝ)) : Prop :=
  abs (A.1 - B.1) = 4

-- The proof problem statement.
theorem findB_coords :
  ∃ B : ℝ × ℝ, isParallelToXAxis A B ∧ hasLengthFour A B ∧ (B = (-7, 2) ∨ B = (1, 2)) :=
  sorry

end ProofProblem

end NUMINAMATH_GPT_findB_coords_l781_78117


namespace NUMINAMATH_GPT_find_fraction_l781_78171

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h : a + b + c = 1)

theorem find_fraction :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3 * (a - b)^2) / (a * b * (1 - a - b)) :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l781_78171


namespace NUMINAMATH_GPT_problem_l781_78102

theorem problem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 7 := 
sorry

end NUMINAMATH_GPT_problem_l781_78102


namespace NUMINAMATH_GPT_oranges_for_profit_l781_78194

theorem oranges_for_profit (cost_buy: ℚ) (number_buy: ℚ) (cost_sell: ℚ) (number_sell: ℚ)
  (desired_profit: ℚ) (h₁: cost_buy / number_buy = 3.75) (h₂: cost_sell / number_sell = 4.5)
  (h₃: desired_profit = 120) :
  ∃ (oranges_to_sell: ℚ), oranges_to_sell = 160 ∧ (desired_profit / ((cost_sell / number_sell) - (cost_buy / number_buy))) = oranges_to_sell :=
by
  sorry

end NUMINAMATH_GPT_oranges_for_profit_l781_78194


namespace NUMINAMATH_GPT_pizzas_needed_l781_78129

theorem pizzas_needed (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) (h_people : people = 18) (h_slices_per_person : slices_per_person = 3) (h_slices_per_pizza : slices_per_pizza = 9) :
  people * slices_per_person / slices_per_pizza = 6 :=
by
  sorry

end NUMINAMATH_GPT_pizzas_needed_l781_78129


namespace NUMINAMATH_GPT_stratified_sampling_seniors_l781_78198

theorem stratified_sampling_seniors
  (total_students : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)
  (senior_sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : seniors = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_sample_size = seniors * sample_size / total_students) :
  senior_sample_size = 100 :=
  sorry

end NUMINAMATH_GPT_stratified_sampling_seniors_l781_78198


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l781_78169

theorem relationship_between_a_b_c :
  let m := 2
  let n := 3
  let f (x : ℝ) := x^3
  let a := f (Real.sqrt 3 / 3)
  let b := f (Real.log Real.pi)
  let c := f (Real.sqrt 2 / 2)
  a < c ∧ c < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l781_78169


namespace NUMINAMATH_GPT_min_a_for_increasing_interval_l781_78186

def f (x a : ℝ) : ℝ := x^2 + (a - 2) * x - 1

theorem min_a_for_increasing_interval (a : ℝ) : (∀ x : ℝ, x ≥ 2 → f x a ≤ f (x + 1) a) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_GPT_min_a_for_increasing_interval_l781_78186


namespace NUMINAMATH_GPT_schedule_arrangements_l781_78106

-- Define the initial setup of the problem
def subjects : List String := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"]

def periods_morning : List String := ["P1", "P2", "P3", "P4"]
def periods_afternoon : List String := ["P5", "P6", "P7"]

-- Define the constraints
def are_consecutive (subj1 subj2 : String) : Bool := 
  (subj1 = "Chinese" ∧ subj2 = "Mathematics") ∨ 
  (subj1 = "Mathematics" ∧ subj2 = "Chinese")

def can_schedule_max_one_period (subject : String) : Bool :=
  subject = "English" ∨ subject = "Physics" ∨ subject = "Chemistry" ∨ subject = "Biology"

-- Define the math problem as a proof in Lean
theorem schedule_arrangements : 
  ∃ n : Nat, n = 336 :=
by
  -- The detailed proof steps would go here
  sorry

end NUMINAMATH_GPT_schedule_arrangements_l781_78106


namespace NUMINAMATH_GPT_shifted_quadratic_roots_l781_78164

theorem shifted_quadratic_roots {a h k : ℝ} (h_root_neg3 : a * (-3 + h) ^ 2 + k = 0)
                                 (h_root_2 : a * (2 + h) ^ 2 + k = 0) :
  (a * (-2 + h) ^ 2 + k = 0) ∧ (a * (3 + h) ^ 2 + k = 0) := by
  sorry

end NUMINAMATH_GPT_shifted_quadratic_roots_l781_78164


namespace NUMINAMATH_GPT_decimal_fraction_to_percentage_l781_78124

theorem decimal_fraction_to_percentage (d : ℝ) (h : d = 0.03) : d * 100 = 3 := by
  sorry

end NUMINAMATH_GPT_decimal_fraction_to_percentage_l781_78124


namespace NUMINAMATH_GPT_bob_average_speed_l781_78183

theorem bob_average_speed
  (lap_distance : ℕ) (lap1_time lap2_time lap3_time total_laps : ℕ)
  (h_lap_distance : lap_distance = 400)
  (h_lap1_time : lap1_time = 70)
  (h_lap2_time : lap2_time = 85)
  (h_lap3_time : lap3_time = 85)
  (h_total_laps : total_laps = 3) : 
  (lap_distance * total_laps) / (lap1_time + lap2_time + lap3_time) = 5 := by
    sorry

end NUMINAMATH_GPT_bob_average_speed_l781_78183


namespace NUMINAMATH_GPT_total_carrots_l781_78119

-- Define the number of carrots grown by Sally and Fred
def sally_carrots := 6
def fred_carrots := 4

-- Theorem: The total number of carrots grown by Sally and Fred
theorem total_carrots : sally_carrots + fred_carrots = 10 := by
  sorry

end NUMINAMATH_GPT_total_carrots_l781_78119


namespace NUMINAMATH_GPT_product_mod_32_l781_78193

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end NUMINAMATH_GPT_product_mod_32_l781_78193


namespace NUMINAMATH_GPT_area_below_line_l781_78103

noncomputable def circle_eqn (x y : ℝ) := 
  x^2 + 2 * x + (y^2 - 6 * y) + 50 = 0

noncomputable def line_eqn (x y : ℝ) := 
  y = x + 1

theorem area_below_line : 
  (∃ (x y : ℝ), circle_eqn x y ∧ y < x + 1) →
  ∃ (a : ℝ), a = 20 * π :=
by
  sorry

end NUMINAMATH_GPT_area_below_line_l781_78103


namespace NUMINAMATH_GPT_workbook_arrangement_l781_78158

-- Define the condition of having different Korean and English workbooks
variables (K1 K2 : Type) (E1 E2 : Type)

-- The main theorem statement
theorem workbook_arrangement :
  ∃ (koreanWorkbooks englishWorkbooks : List (Type)), 
  (koreanWorkbooks.length = 2) ∧
  (englishWorkbooks.length = 2) ∧
  (∀ wb ∈ (koreanWorkbooks ++ englishWorkbooks), wb ≠ wb) ∧
  (∃ arrangements : Nat,
    arrangements = 12) :=
  sorry

end NUMINAMATH_GPT_workbook_arrangement_l781_78158


namespace NUMINAMATH_GPT_orchid_bushes_total_l781_78148

def current_bushes : ℕ := 47
def bushes_today : ℕ := 37
def bushes_tomorrow : ℕ := 25

theorem orchid_bushes_total : current_bushes + bushes_today + bushes_tomorrow = 109 := 
by sorry

end NUMINAMATH_GPT_orchid_bushes_total_l781_78148


namespace NUMINAMATH_GPT_alan_has_5_20_cent_coins_l781_78188

theorem alan_has_5_20_cent_coins
  (a b c : ℕ)
  (h1 : a + b + c = 20)
  (h2 : ((400 - 15 * a - 10 * b) / 5) + 1 = 24) :
  c = 5 :=
by
  sorry

end NUMINAMATH_GPT_alan_has_5_20_cent_coins_l781_78188


namespace NUMINAMATH_GPT_deborah_international_letters_l781_78131

theorem deborah_international_letters (standard_postage : ℝ) 
                                      (additional_charge : ℝ) 
                                      (total_letters : ℕ) 
                                      (total_cost : ℝ) 
                                      (h_standard_postage: standard_postage = 1.08)
                                      (h_additional_charge: additional_charge = 0.14)
                                      (h_total_letters: total_letters = 4)
                                      (h_total_cost: total_cost = 4.60) :
                                      ∃ (x : ℕ), x = 2 :=
by
  sorry

end NUMINAMATH_GPT_deborah_international_letters_l781_78131


namespace NUMINAMATH_GPT_horner_evaluation_of_f_at_5_l781_78174

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_evaluation_of_f_at_5 : f 5 = 2015 :=
by sorry

end NUMINAMATH_GPT_horner_evaluation_of_f_at_5_l781_78174


namespace NUMINAMATH_GPT_find_x_l781_78181

theorem find_x : ∃ x, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l781_78181


namespace NUMINAMATH_GPT_sufficient_condition_l781_78111

theorem sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 2, (1/2 : ℝ) * x^2 - a ≥ 0) → a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l781_78111


namespace NUMINAMATH_GPT_sum_of_intercepts_l781_78172

theorem sum_of_intercepts (x y : ℝ) (h : y + 3 = -2 * (x + 5)) : 
  (- (13 / 2) : ℝ) + (- 13 : ℝ) = - (39 / 2) :=
by sorry

end NUMINAMATH_GPT_sum_of_intercepts_l781_78172


namespace NUMINAMATH_GPT_solve_equation_l781_78135

theorem solve_equation (x : ℝ) : (x - 2) ^ 2 = 9 ↔ x = 5 ∨ x = -1 :=
by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_solve_equation_l781_78135


namespace NUMINAMATH_GPT_pqrs_product_l781_78170

noncomputable def P := (Real.sqrt 2007 + Real.sqrt 2008)
noncomputable def Q := (-Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def R := (Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def S := (-Real.sqrt 2008 + Real.sqrt 2007)

theorem pqrs_product : P * Q * R * S = -1 := by
  sorry

end NUMINAMATH_GPT_pqrs_product_l781_78170


namespace NUMINAMATH_GPT_find_m_l781_78176

theorem find_m (m : ℕ) (h₁ : 0 < m) : 
  144^5 + 91^5 + 56^5 + 19^5 = m^5 → m = 147 := by
  -- Mathematically, we know the sum of powers equals a fifth power of 147
  -- 144^5 = 61917364224
  -- 91^5 = 6240321451
  -- 56^5 = 550731776
  -- 19^5 = 2476099
  -- => 61917364224 + 6240321451 + 550731776 + 2476099 = 68897423550
  -- Find the nearest  m such that m^5 = 68897423550
  sorry

end NUMINAMATH_GPT_find_m_l781_78176


namespace NUMINAMATH_GPT_increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l781_78179

def a_n (n : ℕ) : ℤ := 2 * n - 8

theorem increasing_a_n : ∀ n : ℕ, a_n (n + 1) > a_n n := 
by 
-- Assuming n >= 0
intro n
dsimp [a_n]
sorry

def n_a_n (n : ℕ) : ℤ := n * (2 * n - 8)

theorem not_increasing_n_a_n : ∀ n : ℕ, n > 0 → n_a_n (n + 1) ≤ n_a_n n :=
by
-- Assuming n > 0
intro n hn
dsimp [n_a_n]
sorry

def a_n_over_n (n : ℕ) : ℚ := (2 * n - 8 : ℚ) / n

theorem increasing_a_n_over_n : ∀ n > 0, a_n_over_n (n + 1) > a_n_over_n n :=
by 
-- Assuming n > 0
intro n hn
dsimp [a_n_over_n]
sorry

def a_n_sq (n : ℕ) : ℤ := (2 * n - 8) * (2 * n - 8)

theorem not_increasing_a_n_sq : ∀ n : ℕ, a_n_sq (n + 1) ≤ a_n_sq n :=
by
-- Assuming n >= 0
intro n
dsimp [a_n_sq]
sorry

end NUMINAMATH_GPT_increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l781_78179


namespace NUMINAMATH_GPT_parabola_vertex_l781_78149

theorem parabola_vertex (a b c : ℝ) :
  (∀ x, y = ax^2 + bx + c ↔ 
   y = a*((x+3)^2) + 4) ∧
   (∀ x y, (x, y) = ((1:ℝ), (2:ℝ))) →
   a + b + c = 3 := by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l781_78149


namespace NUMINAMATH_GPT_hillary_minutes_read_on_saturday_l781_78118

theorem hillary_minutes_read_on_saturday :
  let total_minutes := 60
  let friday_minutes := 16
  let sunday_minutes := 16
  total_minutes - (friday_minutes + sunday_minutes) = 28 := by
sorry

end NUMINAMATH_GPT_hillary_minutes_read_on_saturday_l781_78118


namespace NUMINAMATH_GPT_mark_notebooks_at_126_percent_l781_78147

variable (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)

def merchant_condition1 := C = 0.85 * L
def merchant_condition2 := C = 0.75 * S
def merchant_condition3 := S = 0.9 * M

theorem mark_notebooks_at_126_percent :
    merchant_condition1 L C →
    merchant_condition2 C S →
    merchant_condition3 S M →
    M = 1.259 * L := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_mark_notebooks_at_126_percent_l781_78147


namespace NUMINAMATH_GPT_minimum_cost_to_store_food_l781_78153

-- Define the problem setting
def total_volume : ℕ := 15
def capacity_A : ℕ := 2
def capacity_B : ℕ := 3
def price_A : ℕ := 13
def price_B : ℕ := 15
def cashback_threshold : ℕ := 3
def cashback : ℕ := 10

-- The mathematical theorem statement for the proof problem
theorem minimum_cost_to_store_food : 
  ∃ (x y : ℕ), 
    capacity_A * x + capacity_B * y = total_volume ∧ 
    (y = 5 ∧ price_B * y = 75) ∨ 
    (x = 3 ∧ y = 3 ∧ price_A * x + price_B * y - cashback = 74) :=
sorry

end NUMINAMATH_GPT_minimum_cost_to_store_food_l781_78153


namespace NUMINAMATH_GPT_quadratic_fraction_equality_l781_78128

theorem quadratic_fraction_equality (r : ℝ) (h1 : r ≠ 4) (h2 : r ≠ 6) (h3 : r ≠ 5) 
(h4 : r ≠ -4) (h5 : r ≠ -3): 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) →
  r = -7/4 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_fraction_equality_l781_78128


namespace NUMINAMATH_GPT_calculate_total_notebooks_given_to_tom_l781_78137

noncomputable def total_notebooks_given_to_tom : ℝ :=
  let initial_red := 15
  let initial_blue := 17
  let initial_white := 19
  let red_given_day1 := 4.5
  let blue_given_day1 := initial_blue / 3
  let remaining_red_day1 := initial_red - red_given_day1
  let remaining_blue_day1 := initial_blue - blue_given_day1
  let white_given_day2 := initial_white / 2
  let blue_given_day2 := remaining_blue_day1 * 0.25
  let remaining_white_day2 := initial_white - white_given_day2
  let remaining_blue_day2 := remaining_blue_day1 - blue_given_day2
  let red_given_day3 := 3.5
  let blue_given_day3 := (remaining_blue_day2 * 2) / 5
  let remaining_red_day3 := remaining_red_day1 - red_given_day3
  let remaining_blue_day3 := remaining_blue_day2 - blue_given_day3
  let white_kept_day3 := remaining_white_day2 / 4
  let remaining_white_day3 := initial_white - white_kept_day3
  let remaining_notebooks_day3 := remaining_red_day3 + remaining_blue_day3 + remaining_white_day3
  let notebooks_total_day3 := initial_red + initial_blue + initial_white - red_given_day1 - blue_given_day1 - white_given_day2 - blue_given_day2 - red_given_day3 - blue_given_day3 - white_kept_day3
  let tom_notebooks := red_given_day1 + blue_given_day1
  notebooks_total_day3

theorem calculate_total_notebooks_given_to_tom : total_notebooks_given_to_tom = 10.17 :=
  sorry

end NUMINAMATH_GPT_calculate_total_notebooks_given_to_tom_l781_78137


namespace NUMINAMATH_GPT_fishing_problem_l781_78180

theorem fishing_problem
  (P : ℕ) -- weight of the fish Peter caught
  (H1 : Ali_weight = 2 * P) -- Ali caught twice as much as Peter
  (H2 : Joey_weight = P + 1) -- Joey caught 1 kg more than Peter
  (H3 : P + 2 * P + (P + 1) = 25) -- Together they caught 25 kg
  : Ali_weight = 12 :=
by
  sorry

end NUMINAMATH_GPT_fishing_problem_l781_78180


namespace NUMINAMATH_GPT_option_C_is_quadratic_l781_78159

theorem option_C_is_quadratic : ∀ (x : ℝ), (x = x^2) ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0) := 
by
  sorry

end NUMINAMATH_GPT_option_C_is_quadratic_l781_78159


namespace NUMINAMATH_GPT_benny_final_comic_books_l781_78133

-- Define the initial number of comic books
def initial_comic_books : ℕ := 22

-- Define the comic books sold (half of the initial)
def comic_books_sold : ℕ := initial_comic_books / 2

-- Define the comic books left after selling half
def comic_books_left_after_sale : ℕ := initial_comic_books - comic_books_sold

-- Define the number of comic books bought
def comic_books_bought : ℕ := 6

-- Define the final number of comic books
def final_comic_books : ℕ := comic_books_left_after_sale + comic_books_bought

-- Statement to prove that Benny has 17 comic books at the end
theorem benny_final_comic_books : final_comic_books = 17 := by
  sorry

end NUMINAMATH_GPT_benny_final_comic_books_l781_78133


namespace NUMINAMATH_GPT_daffodil_bulb_cost_l781_78113

theorem daffodil_bulb_cost :
  let total_bulbs := 55
  let crocus_cost := 0.35
  let total_budget := 29.15
  let num_crocus_bulbs := 22
  let total_crocus_cost := num_crocus_bulbs * crocus_cost
  let remaining_budget := total_budget - total_crocus_cost
  let num_daffodil_bulbs := total_bulbs - num_crocus_bulbs
  remaining_budget / num_daffodil_bulbs = 0.65 := 
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_daffodil_bulb_cost_l781_78113


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l781_78141

theorem isosceles_triangle_sides (r R : ℝ) (a b c : ℝ) (h1 : r = 3 / 2) (h2 : R = 25 / 8)
  (h3 : a = c) (h4 : 5 = a) (h5 : 6 = b) : 
  ∃ a b c, a = 5 ∧ c = 5 ∧ b = 6 := by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l781_78141


namespace NUMINAMATH_GPT_minimum_value_a_plus_3b_plus_9c_l781_78168

open Real

theorem minimum_value_a_plus_3b_plus_9c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
sorry

end NUMINAMATH_GPT_minimum_value_a_plus_3b_plus_9c_l781_78168


namespace NUMINAMATH_GPT_fraction_simplification_l781_78115

theorem fraction_simplification :
  (1 / 330) + (19 / 30) = 7 / 11 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l781_78115


namespace NUMINAMATH_GPT_gasoline_price_increase_l781_78130

theorem gasoline_price_increase
  (P Q : ℝ) -- Prices and quantities
  (x : ℝ) -- The percentage increase in price
  (h1 : (P * (1 + x / 100)) * (Q * 0.95) = P * Q * 1.14) -- Given condition
  : x = 20 := 
sorry

end NUMINAMATH_GPT_gasoline_price_increase_l781_78130


namespace NUMINAMATH_GPT_plum_purchase_l781_78191

theorem plum_purchase
    (x : ℕ)
    (h1 : ∃ x, 5 * (6 * (4 * x) / 5) - 6 * ((5 * x) / 6) = -30) :
    2 * x = 60 := sorry

end NUMINAMATH_GPT_plum_purchase_l781_78191


namespace NUMINAMATH_GPT_total_tea_cups_l781_78100

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end NUMINAMATH_GPT_total_tea_cups_l781_78100


namespace NUMINAMATH_GPT_simplify_expression_l781_78136

theorem simplify_expression : (- (1 / 343 : ℝ)) ^ (-2 / 3 : ℝ) = 49 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l781_78136


namespace NUMINAMATH_GPT_some_base_value_l781_78125

noncomputable def some_base (x y : ℝ) (h1 : x * y = 1) (h2 : (some_base : ℝ) → (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : ℝ :=
  7

theorem some_base_value (x y : ℝ) (h1 : x * y = 1) (h2 : ∀ some_base : ℝ, (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : some_base x y h1 h2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_some_base_value_l781_78125


namespace NUMINAMATH_GPT_primer_cost_before_discount_l781_78150

theorem primer_cost_before_discount (primer_cost_after_discount : ℝ) (paint_cost : ℝ) (total_cost : ℝ) 
  (rooms : ℕ) (primer_discount : ℝ) (paint_cost_per_gallon : ℝ) :
  (primer_cost_after_discount = total_cost - (rooms * paint_cost_per_gallon)) →
  (rooms * (primer_cost - primer_discount * primer_cost) = primer_cost_after_discount) →
  primer_cost = 30 := by
  sorry

end NUMINAMATH_GPT_primer_cost_before_discount_l781_78150


namespace NUMINAMATH_GPT_range_of_a_l781_78192

variable (a x y : ℝ)

theorem range_of_a (h1 : 2 * x + y = 1 + 4 * a) (h2 : x + 2 * y = 2 - a) (h3 : x + y > 0) : a > -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l781_78192


namespace NUMINAMATH_GPT_computation_l781_78110

theorem computation : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end NUMINAMATH_GPT_computation_l781_78110


namespace NUMINAMATH_GPT_problem_l781_78178

theorem problem (a : ℤ) (ha : 0 ≤ a ∧ a < 13) (hdiv : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end NUMINAMATH_GPT_problem_l781_78178


namespace NUMINAMATH_GPT_quadratic_has_two_roots_l781_78139

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end NUMINAMATH_GPT_quadratic_has_two_roots_l781_78139


namespace NUMINAMATH_GPT_length_of_the_bridge_l781_78187

theorem length_of_the_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (cross_time_s : ℕ)
  (h_train_length : train_length = 120)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_cross_time_s : cross_time_s = 30) :
  ∃ bridge_length : ℕ, bridge_length = 255 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_the_bridge_l781_78187


namespace NUMINAMATH_GPT_sin_sum_angles_36_108_l781_78123

theorem sin_sum_angles_36_108 (A B C : ℝ) (h_sum : A + B + C = 180)
  (h_angle : A = 36 ∨ A = 108 ∨ B = 36 ∨ B = 108 ∨ C = 36 ∨ C = 108) :
  Real.sin (5 * A) + Real.sin (5 * B) + Real.sin (5 * C) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sin_sum_angles_36_108_l781_78123
