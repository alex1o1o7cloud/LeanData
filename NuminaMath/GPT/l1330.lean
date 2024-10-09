import Mathlib

namespace find_f_minus_two_l1330_133036

noncomputable def f : ℝ → ℝ := sorry

axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_one : f 1 = 2

theorem find_f_minus_two : f (-2) = 2 :=
by sorry

end find_f_minus_two_l1330_133036


namespace ellie_shoes_count_l1330_133045

variable (E R : ℕ)

def ellie_shoes (E R : ℕ) : Prop :=
  E + R = 13 ∧ E = R + 3

theorem ellie_shoes_count (E R : ℕ) (h : ellie_shoes E R) : E = 8 :=
  by sorry

end ellie_shoes_count_l1330_133045


namespace cheaper_lens_price_l1330_133052

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) 
  (h₁ : original_price = 300) 
  (h₂ : discount_rate = 0.20) 
  (h₃ : savings = 20) 
  (discounted_price : ℝ) 
  (cheaper_lens_price : ℝ)
  (discount_eq : discounted_price = original_price * (1 - discount_rate))
  (savings_eq : cheaper_lens_price = discounted_price - savings) :
  cheaper_lens_price = 220 := 
by sorry

end cheaper_lens_price_l1330_133052


namespace jackson_spends_on_school_supplies_l1330_133050

theorem jackson_spends_on_school_supplies :
  let num_students := 50
  let pens_per_student := 7
  let notebooks_per_student := 5
  let binders_per_student := 3
  let highlighters_per_student := 4
  let folders_per_student := 2
  let cost_pen := 0.70
  let cost_notebook := 1.60
  let cost_binder := 5.10
  let cost_highlighter := 0.90
  let cost_folder := 1.15
  let teacher_discount := 135
  let bulk_discount := 25
  let sales_tax_rate := 0.05
  let total_cost := 
    (num_students * pens_per_student * cost_pen) + 
    (num_students * notebooks_per_student * cost_notebook) + 
    (num_students * binders_per_student * cost_binder) + 
    (num_students * highlighters_per_student * cost_highlighter) + 
    (num_students * folders_per_student * cost_folder)
  let discounted_cost := total_cost - teacher_discount - bulk_discount
  let sales_tax := discounted_cost * sales_tax_rate
  let final_cost := discounted_cost + sales_tax
  final_cost = 1622.25 := by
  sorry

end jackson_spends_on_school_supplies_l1330_133050


namespace verify_number_of_true_props_l1330_133032

def original_prop (a : ℝ) : Prop := a > -3 → a > 0
def converse_prop (a : ℝ) : Prop := a > 0 → a > -3
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ 0
def contrapositive_prop (a : ℝ) : Prop := a ≤ 0 → a ≤ -3

theorem verify_number_of_true_props :
  (¬ original_prop a ∧ converse_prop a ∧ inverse_prop a ∧ ¬ contrapositive_prop a) → (2 = 2) := sorry

end verify_number_of_true_props_l1330_133032


namespace find_dividend_l1330_133082

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 8) (h2 : quotient = 8) (h3 : dividend = k * quotient) : dividend = 64 := 
by 
  sorry

end find_dividend_l1330_133082


namespace translate_A_coordinates_l1330_133042

-- Definitions
def A_initial : ℝ × ℝ := (-3, 2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

-- Final coordinates after transformation
def A' : ℝ × ℝ :=
  let A_translated := translate_right A_initial 4
  translate_down A_translated 3

-- Proof statement
theorem translate_A_coordinates :
  A' = (1, -1) :=
by
  simp [A', translate_right, translate_down, A_initial]
  sorry

end translate_A_coordinates_l1330_133042


namespace region_area_l1330_133015

/-- 
  Trapezoid has side lengths 10, 10, 10, and 22. 
  Each side of the trapezoid is the diameter of a semicircle 
  with the two semicircles on the two parallel sides of the trapezoid facing outside 
  and the other two semicircles facing inside the trapezoid.
  The region bounded by these four semicircles has area m + nπ, where m and n are positive integers.
  Prove that m + n = 188.5.
-/
theorem region_area (m n : ℝ) (h1: m = 128) (h2: n = 60.5) : m + n = 188.5 :=
by
  rw [h1, h2]
  norm_num -- simplifies the expression and checks it is equal to 188.5

end region_area_l1330_133015


namespace equilateral_triangle_perimeter_l1330_133059

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end equilateral_triangle_perimeter_l1330_133059


namespace expand_polynomials_l1330_133072

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end expand_polynomials_l1330_133072


namespace base3_addition_proof_l1330_133000

-- Define the base 3 numbers
def one_3 : ℕ := 1
def twelve_3 : ℕ := 1 * 3 + 2
def two_hundred_twelve_3 : ℕ := 2 * 3^2 + 1 * 3 + 2
def two_thousand_one_hundred_twenty_one_3 : ℕ := 2 * 3^3 + 1 * 3^2 + 2 * 3 + 1

-- Define the correct answer in base 3
def expected_sum_3 : ℕ := 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3 + 0

-- The proof problem
theorem base3_addition_proof :
  one_3 + twelve_3 + two_hundred_twelve_3 + two_thousand_one_hundred_twenty_one_3 = expected_sum_3 :=
by
  -- Proof goes here
  sorry

end base3_addition_proof_l1330_133000


namespace probability_palindrome_divisible_by_11_is_zero_l1330_133079

def is_palindrome (n : ℕ) :=
  3000 ≤ n ∧ n < 8000 ∧ ∃ (a b : ℕ), 3 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ (n : ℕ), is_palindrome n ∧ n % 11 = 0) → false := by sorry

end probability_palindrome_divisible_by_11_is_zero_l1330_133079


namespace emily_101st_card_is_10_of_Hearts_l1330_133057

def number_sequence : List String := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
def suit_sequence : List String := ["Hearts", "Diamonds", "Clubs", "Spades"]

-- Function to get the number of a specific card
def card_number (n : ℕ) : String :=
  number_sequence.get! (n % number_sequence.length)

-- Function to get the suit of a specific card
def card_suit (n : ℕ) : String :=
  suit_sequence.get! ((n / suit_sequence.length) % suit_sequence.length)

-- Definition to state the question and the answer
def emily_card (n : ℕ) : String := card_number n ++ " of " ++ card_suit n

-- Proving that the 101st card is "10 of Hearts"
theorem emily_101st_card_is_10_of_Hearts : emily_card 100 = "10 of Hearts" :=
by {
  sorry
}

end emily_101st_card_is_10_of_Hearts_l1330_133057


namespace find_v_l1330_133027

variables (a b c : ℝ)

def condition1 := (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -6
def condition2 := (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

theorem find_v (h1 : condition1 a b c) (h2 : condition2 a b c) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 17 / 2 :=
by
  sorry

end find_v_l1330_133027


namespace sin_minus_cos_l1330_133084

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l1330_133084


namespace missed_number_l1330_133055

/-
  A student finds the sum \(1 + 2 + 3 + \cdots\) as his patience runs out. 
  He found the sum as 575. When the teacher declared the result wrong, 
  the student realized that he missed a number.
  Prove that the number he missed is 20.
-/

theorem missed_number (n : ℕ) (S_incorrect S_correct S_missed : ℕ) 
  (h1 : S_incorrect = 575)
  (h2 : S_correct = n * (n + 1) / 2)
  (h3 : S_correct = 595)
  (h4 : S_missed = S_correct - S_incorrect) :
  S_missed = 20 :=
sorry

end missed_number_l1330_133055


namespace total_flowers_sold_l1330_133009

theorem total_flowers_sold :
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  flowers_mon + flowers_tue + flowers_wed + flowers_thu + flowers_fri + flowers_sat = 78 :=
by
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  sorry

end total_flowers_sold_l1330_133009


namespace triangles_xyz_l1330_133016

theorem triangles_xyz (A B C D P Q R : Type) 
    (u v w x : ℝ)
    (angle_ADB angle_BDC angle_CDA : ℝ)
    (h1 : angle_ADB = 120) 
    (h2 : angle_BDC = 120) 
    (h3 : angle_CDA = 120) :
    x = u + v + w :=
sorry

end triangles_xyz_l1330_133016


namespace dawn_wash_dishes_time_l1330_133046

theorem dawn_wash_dishes_time (D : ℕ) : 2 * D + 6 = 46 → D = 20 :=
by
  intro h
  sorry

end dawn_wash_dishes_time_l1330_133046


namespace angle_C_correct_l1330_133002

theorem angle_C_correct (A B C : ℝ) (h1 : A = 65) (h2 : B = 40) (h3 : A + B + C = 180) : C = 75 :=
sorry

end angle_C_correct_l1330_133002


namespace calculate_expression_l1330_133056

theorem calculate_expression : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end calculate_expression_l1330_133056


namespace common_divisors_9240_8820_l1330_133058

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l1330_133058


namespace total_canoes_by_end_of_april_l1330_133038

def N_F : ℕ := 4
def N_M : ℕ := 3 * N_F
def N_A : ℕ := 3 * N_M
def total_canoes : ℕ := N_F + N_M + N_A

theorem total_canoes_by_end_of_april : total_canoes = 52 := by
  sorry

end total_canoes_by_end_of_april_l1330_133038


namespace marina_more_fudge_l1330_133083

theorem marina_more_fudge (h1 : 4.5 * 16 = 72)
                          (h2 : 4 * 16 - 6 = 58) :
                          72 - 58 = 14 := by
  sorry

end marina_more_fudge_l1330_133083


namespace club_members_problem_l1330_133096

theorem club_members_problem 
    (T : ℕ) (C : ℕ) (D : ℕ) (B : ℕ) 
    (h_T : T = 85) (h_C : C = 45) (h_D : D = 32) (h_B : B = 18) :
    let Cₒ := C - B
    let Dₒ := D - B
    let N := T - (Cₒ + Dₒ + B)
    N = 26 :=
by
  sorry

end club_members_problem_l1330_133096


namespace algebraic_expression_value_l1330_133010

theorem algebraic_expression_value (x : ℝ) (h : x = 5) : (3 / (x - 4) - 24 / (x^2 - 16)) = (1 / 3) :=
by
  have hx : x = 5 := h
  sorry

end algebraic_expression_value_l1330_133010


namespace walnut_trees_total_l1330_133026

variable (current_trees : ℕ) (new_trees : ℕ)

theorem walnut_trees_total (h1 : current_trees = 22) (h2 : new_trees = 55) : current_trees + new_trees = 77 :=
by
  sorry

end walnut_trees_total_l1330_133026


namespace relationship_between_a_b_c_l1330_133065

noncomputable def a : ℝ := Real.exp (-2)

noncomputable def b : ℝ := a ^ a

noncomputable def c : ℝ := a ^ b

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by {
  sorry
}

end relationship_between_a_b_c_l1330_133065


namespace student_sums_attempted_l1330_133003

theorem student_sums_attempted (sums_right sums_wrong : ℕ) (h1 : sums_wrong = 2 * sums_right) (h2 : sums_right = 16) :
  sums_right + sums_wrong = 48 :=
by
  sorry

end student_sums_attempted_l1330_133003


namespace slope_of_parallel_line_l1330_133092

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l1330_133092


namespace terrell_lifting_problem_l1330_133001

theorem terrell_lifting_problem (w1 w2 w3 n1 n2 : ℕ) (h1 : w1 = 12) (h2 : w2 = 18) (h3 : w3 = 24) (h4 : n1 = 20) :
  60 * n2 = 3 * w1 * n1 → n2 = 12 :=
by
  intros h
  sorry

end terrell_lifting_problem_l1330_133001


namespace circle_radius_and_diameter_relations_l1330_133053

theorem circle_radius_and_diameter_relations
  (r_x r_y r_z A_x A_y A_z d_x d_z : ℝ)
  (hx_circumference : 2 * π * r_x = 18 * π)
  (hx_area : A_x = π * r_x^2)
  (hy_area_eq : A_y = A_x)
  (hz_area_eq : A_z = 4 * A_x)
  (hy_area : A_y = π * r_y^2)
  (hz_area : A_z = π * r_z^2)
  (dx_def : d_x = 2 * r_x)
  (dz_def : d_z = 2 * r_z)
  : r_y = r_z / 2 ∧ d_z = 2 * d_x := 
by 
  sorry

end circle_radius_and_diameter_relations_l1330_133053


namespace find_A_l1330_133019

theorem find_A (A : ℕ) (B : ℕ) (h₁ : 0 ≤ B ∧ B ≤ 999) (h₂ : 1000 * A + B = A * (A + 1) / 2) : A = 1999 :=
  sorry

end find_A_l1330_133019


namespace watch_correct_time_l1330_133031

-- Conditions
def initial_time_slow : ℕ := 4 -- minutes slow at 8:00 AM
def final_time_fast : ℕ := 6 -- minutes fast at 4:00 PM
def total_time_interval : ℕ := 480 -- total time interval in minutes from 8:00 AM to 4:00 PM
def rate_of_time_gain : ℚ := (initial_time_slow + final_time_fast) / total_time_interval

-- Statement to prove
theorem watch_correct_time : 
  ∃ t : ℕ, t = 11 * 60 + 12 ∧ 
  ((8 * 60 + t) * rate_of_time_gain = 4) := 
sorry

end watch_correct_time_l1330_133031


namespace log_expression_equality_l1330_133008

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_equality :
  Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + (log_base 2 5) * (log_base 5 8) = 5 := by
  sorry

end log_expression_equality_l1330_133008


namespace find_f_13_l1330_133013

variable (f : ℤ → ℤ)

def is_odd_function (f : ℤ → ℤ) := ∀ x : ℤ, f (-x) = -f (x)
def has_period_4 (f : ℤ → ℤ) := ∀ x : ℤ, f (x + 4) = f (x)

theorem find_f_13 (h1 : is_odd_function f) (h2 : has_period_4 f) (h3 : f (-1) = 2) : f 13 = -2 :=
by
  sorry

end find_f_13_l1330_133013


namespace more_pie_eaten_l1330_133014

theorem more_pie_eaten (erik_pie : ℝ) (frank_pie : ℝ)
  (h_erik : erik_pie = 0.6666666666666666)
  (h_frank : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 :=
by
  sorry

end more_pie_eaten_l1330_133014


namespace agnes_twice_jane_in_years_l1330_133091

def agnes_age := 25
def jane_age := 6

theorem agnes_twice_jane_in_years (x : ℕ) : 
  25 + x = 2 * (6 + x) → x = 13 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end agnes_twice_jane_in_years_l1330_133091


namespace savings_are_equal_and_correct_l1330_133066

-- Definitions of the given conditions
variables (I1 I2 E1 E2 : ℝ)
variables (S1 S2 : ℝ)
variables (rI : ℝ := 5/4) -- ratio of incomes
variables (rE : ℝ := 3/2) -- ratio of expenditures
variables (I1_val : ℝ := 3000) -- P1's income

-- Given conditions
def given_conditions : Prop :=
  I1 = I1_val ∧
  I1 / I2 = rI ∧
  E1 / E2 = rE ∧
  S1 = S2

-- Required proof
theorem savings_are_equal_and_correct (I2_val : I2 = (I1_val * 4/5)) (x : ℝ) (h1 : E1 = 3 * x) (h2 : E2 = 2 * x) (h3 : S1 = 1200) :
  S1 = S2 ∧ S1 = 1200 := by
  sorry

end savings_are_equal_and_correct_l1330_133066


namespace problem1_problem2_l1330_133097

-- For problem (1)
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := sorry

-- For problem (2)
theorem problem2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b^2 = a * c) :
  a^2 + b^2 + c^2 > (a - b + c)^2 := sorry

end problem1_problem2_l1330_133097


namespace inequal_min_value_l1330_133054

theorem inequal_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1/x + 4/y) ≥ 9/4 :=
sorry

end inequal_min_value_l1330_133054


namespace pigeons_problem_l1330_133007

theorem pigeons_problem
  (x y : ℕ)
  (h1 : 6 * y + 3 = x)
  (h2 : 8 * y = x + 5) : x = 27 := 
sorry

end pigeons_problem_l1330_133007


namespace right_triangle_legs_l1330_133098

theorem right_triangle_legs (a b : ℤ) (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a^2 + b^2 = 65^2) : 
  a = 16 ∧ b = 63 ∨ a = 63 ∧ b = 16 :=
sorry

end right_triangle_legs_l1330_133098


namespace passed_candidates_count_l1330_133051

theorem passed_candidates_count
    (average_total : ℝ)
    (number_candidates : ℕ)
    (average_passed : ℝ)
    (average_failed : ℝ)
    (total_marks : ℝ) :
    average_total = 35 →
    number_candidates = 120 →
    average_passed = 39 →
    average_failed = 15 →
    total_marks = average_total * number_candidates →
    (∃ P F, P + F = number_candidates ∧ 39 * P + 15 * F = total_marks ∧ P = 100) :=
by
  sorry

end passed_candidates_count_l1330_133051


namespace radius_triple_area_l1330_133039

variable (r n : ℝ)

theorem radius_triple_area (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n / 2) * (Real.sqrt 3 - 1) :=
sorry

end radius_triple_area_l1330_133039


namespace unit_square_BE_value_l1330_133044

theorem unit_square_BE_value
  (ABCD : ℝ × ℝ → Prop)
  (unit_square : ∀ (a b c d : ℝ × ℝ), ABCD a ∧ ABCD b ∧ ABCD c ∧ ABCD d → 
                  a.1 = 0 ∧ a.2 = 0 ∧ b.1 = 1 ∧ b.2 = 0 ∧ 
                  c.1 = 1 ∧ c.2 = 1 ∧ d.1 = 0 ∧ d.2 = 1)
  (E F G : ℝ × ℝ)
  (on_sides : E.1 = 1 ∧ F.2 = 1 ∧ G.1 = 0)
  (AE_perp_EF : ((E.1 - 0) * (F.2 - E.2)) + ((E.2 - 0) * (F.1 - E.1)) = 0)
  (EF_perp_FG : ((F.1 - E.1) * (G.2 - F.2)) + ((F.2 - E.2) * (G.1 - F.1)) = 0)
  (GA_val : (1 - G.1) = 404 / 1331) :
  ∃ BE, BE = 9 / 11 := 
sorry

end unit_square_BE_value_l1330_133044


namespace vec_mag_diff_eq_neg_one_l1330_133074

variables (a b : ℝ × ℝ)

def vec_add_eq := a + b = (2, 3)

def vec_sub_eq := a - b = (-2, 1)

theorem vec_mag_diff_eq_neg_one (h₁ : vec_add_eq a b) (h₂ : vec_sub_eq a b) :
  (a.1 ^ 2 + a.2 ^ 2) - (b.1 ^ 2 + b.2 ^ 2) = -1 :=
  sorry

end vec_mag_diff_eq_neg_one_l1330_133074


namespace journey_time_ratio_l1330_133033

theorem journey_time_ratio (D : ℝ) (hD_pos : D > 0) :
  let T1 := D / 45
  let T2 := D / 30
  (T2 / T1) = (3 / 2) := 
by
  sorry

end journey_time_ratio_l1330_133033


namespace age_problem_l1330_133018

variable (A B x : ℕ)

theorem age_problem (h1 : A = B + 5) (h2 : B = 35) (h3 : A + x = 2 * (B - x)) : x = 10 :=
sorry

end age_problem_l1330_133018


namespace right_triangle_ABC_l1330_133085

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Points definitions
def point_A : ℝ × ℝ := (1, 2)
def point_on_line : ℝ × ℝ := (5, -2)

-- Points B and C on the parabola with parameters t and s respectively
def point_B (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)
def point_C (s : ℝ) : ℝ × ℝ := (s^2, 2 * s)

-- Line equation passing through points B and C
def line_eq (s t : ℝ) (x y : ℝ) : Prop :=
  2 * x - (s + t) * y + 2 * s * t = 0

-- Proof goal: Show that triangle ABC is a right triangle
theorem right_triangle_ABC
  (t s : ℝ)
  (hB : parabola (point_B t).1 (point_B t).2)
  (hC : parabola (point_C s).1 (point_C s).2)
  (hlt : point_on_line.1 = (5 : ℝ))
  (hlx : line_eq s t point_on_line.1 point_on_line.2)
  : let A := point_A
    let B := point_B t
    let C := point_C s
    -- Conclusion: triangle ABC is a right triangle
    k_AB * k_AC = -1 :=
  sorry
  where k_AB := (2 * t - 2) / (t^2 - 1)
        k_AC := (2 * s - 2) / (s^2 - 1)
        rel_t_s := (s + 1) * (t + 1) = -4

end right_triangle_ABC_l1330_133085


namespace expression_eq_one_l1330_133090

theorem expression_eq_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
   a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
   b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 := 
by
  sorry

end expression_eq_one_l1330_133090


namespace choir_members_total_l1330_133077

theorem choir_members_total
  (first_group second_group third_group : ℕ)
  (h1 : first_group = 25)
  (h2 : second_group = 30)
  (h3 : third_group = 15) :
  first_group + second_group + third_group = 70 :=
by
  sorry

end choir_members_total_l1330_133077


namespace value_of_expression_l1330_133048

theorem value_of_expression (x : ℤ) (h : x = 5) : x^5 - 10 * x = 3075 := by
  sorry

end value_of_expression_l1330_133048


namespace at_least_one_success_l1330_133088

-- Define probabilities for A, B, and C
def pA : ℚ := 1 / 2
def pB : ℚ := 2 / 3
def pC : ℚ := 4 / 5

-- Define the probability that none succeed
def pNone : ℚ := (1 - pA) * (1 - pB) * (1 - pC)

-- Define the probability that at least one of them succeeds
def pAtLeastOne : ℚ := 1 - pNone

theorem at_least_one_success : pAtLeastOne = 29 / 30 := 
by sorry

end at_least_one_success_l1330_133088


namespace perfect_square_count_between_20_and_150_l1330_133063

theorem perfect_square_count_between_20_and_150 :
  let lower_bound := 20
  let upper_bound := 150
  let smallest_ps := 25
  let largest_ps := 144
  let count_squares (a b : Nat) := b - a
  count_squares 4 12 = 8 := sorry

end perfect_square_count_between_20_and_150_l1330_133063


namespace two_digit_number_sum_l1330_133080

theorem two_digit_number_sum (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by {
  sorry
}

end two_digit_number_sum_l1330_133080


namespace completing_the_square_result_l1330_133025

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l1330_133025


namespace alyosha_cube_cut_l1330_133061

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l1330_133061


namespace sheets_in_stack_l1330_133017

theorem sheets_in_stack 
  (num_sheets : ℕ) 
  (initial_thickness final_thickness : ℝ) 
  (t_per_sheet : ℝ) 
  (h_initial : num_sheets = 800) 
  (h_thickness : initial_thickness = 4) 
  (h_thickness_per_sheet : initial_thickness / num_sheets = t_per_sheet) 
  (h_final_thickness : final_thickness = 6) 
  : num_sheets * (final_thickness / t_per_sheet) = 1200 := 
by 
  sorry

end sheets_in_stack_l1330_133017


namespace angle_quadrant_l1330_133067

theorem angle_quadrant (θ : Real) (P : Real × Real) (h : P = (Real.sin θ * Real.cos θ, 2 * Real.cos θ) ∧ P.1 < 0 ∧ P.2 < 0) :
  π / 2 < θ ∧ θ < π :=
by
  sorry

end angle_quadrant_l1330_133067


namespace geometric_sequence_sum_l1330_133021

theorem geometric_sequence_sum (S : ℕ → ℝ) 
  (S5 : S 5 = 10)
  (S10 : S 10 = 50) :
  S 15 = 210 := 
by
  sorry

end geometric_sequence_sum_l1330_133021


namespace speed_of_boat_in_still_water_l1330_133078

variables (Vb Vs : ℝ)

-- Conditions
def condition_1 : Prop := Vb + Vs = 11
def condition_2 : Prop := Vb - Vs = 5

theorem speed_of_boat_in_still_water (h1 : condition_1 Vb Vs) (h2 : condition_2 Vb Vs) : Vb = 8 := 
by sorry

end speed_of_boat_in_still_water_l1330_133078


namespace width_of_Carols_rectangle_l1330_133029

theorem width_of_Carols_rectangle 
  (w : ℝ) 
  (h1 : 15 * w = 6 * 50) : w = 20 := 
by 
  sorry

end width_of_Carols_rectangle_l1330_133029


namespace minimum_focal_length_of_hyperbola_l1330_133012

noncomputable def minimum_focal_length (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
    (h₃ : (1/2) * a * (2 * b) = 8) : ℝ :=
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let focal_length := 2 * c
  focal_length

theorem minimum_focal_length_of_hyperbola 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : (1/2) * a * (2 * b) = 8) :
  minimum_focal_length a b h₁ h₂ h₃ = 8 :=
by
  sorry

end minimum_focal_length_of_hyperbola_l1330_133012


namespace smallest_x_solution_l1330_133086

theorem smallest_x_solution (x : ℚ) :
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) →
  (x = -7/3 ∨ x = -11/16) →
  x = -7/3 :=
by
  sorry

end smallest_x_solution_l1330_133086


namespace f_half_and_minus_half_l1330_133022

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem f_half_and_minus_half :
  f (1 / 2) + f (-1 / 2) = 2 := by
  sorry

end f_half_and_minus_half_l1330_133022


namespace wheel_speed_l1330_133076

theorem wheel_speed (r : ℝ) (c : ℝ) (ts tf : ℝ) 
  (h₁ : c = 13) 
  (h₂ : r * ts = c / 5280) 
  (h₃ : (r + 6) * (tf - 1/3 / 3600) = c / 5280) 
  (h₄ : tf = ts - 1 / 10800) :
  r = 12 :=
  sorry

end wheel_speed_l1330_133076


namespace total_salary_after_strict_manager_l1330_133023

-- Definitions based on conditions
def total_initial_salary (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  500 * x + (Finset.sum (Finset.range y) s) = 10000

def kind_manager_total (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  1500 * x + (Finset.sum (Finset.range y) s) + 1000 * y = 24000

def strict_manager_total (x y : ℕ) : ℕ :=
  500 * (x + y)

-- Lean statement to prove the required
theorem total_salary_after_strict_manager (x y : ℕ) (s : ℕ → ℕ) 
  (h_total_initial : total_initial_salary x y s) (h_kind_manager : kind_manager_total x y s) :
  strict_manager_total x y = 7000 := by
  sorry

end total_salary_after_strict_manager_l1330_133023


namespace find_n_l1330_133095

theorem find_n (n : ℕ) (h : n ≥ 2) : 
  (∀ (i j : ℕ), 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔ ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by
  sorry

end find_n_l1330_133095


namespace initial_salty_cookies_count_l1330_133081

-- Define initial conditions
def initial_sweet_cookies : ℕ := 9
def sweet_cookies_ate : ℕ := 36
def salty_cookies_left : ℕ := 3
def salty_cookies_ate : ℕ := 3

-- Theorem to prove the initial salty cookies count
theorem initial_salty_cookies_count (initial_salty_cookies : ℕ) 
    (initial_sweet_cookies : initial_sweet_cookies = 9) 
    (sweet_cookies_ate : sweet_cookies_ate = 36)
    (salty_cookies_ate : salty_cookies_ate = 3) 
    (salty_cookies_left : salty_cookies_left = 3) : 
    initial_salty_cookies = 6 := 
sorry

end initial_salty_cookies_count_l1330_133081


namespace product_of_repeating_decimal_and_five_l1330_133093

noncomputable def repeating_decimal : ℚ :=
  456 / 999

theorem product_of_repeating_decimal_and_five : 
  (repeating_decimal * 5) = 760 / 333 :=
by
  -- The proof is omitted.
  sorry

end product_of_repeating_decimal_and_five_l1330_133093


namespace slope_of_line_l1330_133099

def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (4, 5)

theorem slope_of_line : 
  let (x1, y1) := point1
  let (x2, y2) := point2
  (x2 - x1) ≠ 0 → (y2 - y1) / (x2 - x1) = 1 := by
  sorry

end slope_of_line_l1330_133099


namespace price_of_second_tea_l1330_133006

theorem price_of_second_tea (price_first_tea : ℝ) (mixture_price : ℝ) (required_ratio : ℝ) (price_second_tea : ℝ) :
  price_first_tea = 62 → mixture_price = 64.5 → required_ratio = 3 → price_second_tea = 65.33 :=
by
  intros h1 h2 h3
  sorry

end price_of_second_tea_l1330_133006


namespace equal_sum_sequence_a18_l1330_133043

def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_sequence_a18 (a : ℕ → ℕ) (h : equal_sum_sequence a 5) (h1 : a 1 = 2) : a 18 = 3 :=
  sorry

end equal_sum_sequence_a18_l1330_133043


namespace cube_side_length_is_30_l1330_133024

theorem cube_side_length_is_30
  (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (s : ℝ)
  (h1 : cost_per_kg = 40)
  (h2 : coverage_per_kg = 20)
  (h3 : total_cost = 10800)
  (total_surface_area : ℝ) (W : ℝ) (C : ℝ)
  (h4 : total_surface_area = 6 * s^2)
  (h5 : W = total_surface_area / coverage_per_kg)
  (h6 : C = W * cost_per_kg)
  (h7 : C = total_cost) :
  s = 30 :=
by
  sorry

end cube_side_length_is_30_l1330_133024


namespace sum_of_three_smallest_positive_solutions_equals_ten_and_half_l1330_133062

noncomputable def sum_three_smallest_solutions : ℚ :=
    let x1 : ℚ := 2.75
    let x2 : ℚ := 3 + (4 / 9)
    let x3 : ℚ := 4 + (5 / 16)
    x1 + x2 + x3

theorem sum_of_three_smallest_positive_solutions_equals_ten_and_half :
  sum_three_smallest_solutions = 10.5 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_equals_ten_and_half_l1330_133062


namespace sin_alpha_minus_pi_over_6_l1330_133004

variable (α : ℝ)

theorem sin_alpha_minus_pi_over_6 (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_alpha_minus_pi_over_6_l1330_133004


namespace units_digit_proof_l1330_133047

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end units_digit_proof_l1330_133047


namespace find_door_height_l1330_133087

theorem find_door_height :
  ∃ (h : ℝ), 
  let l := 25
  let w := 15
  let H := 12
  let A := 80 * H
  let W := 960 - (6 * h + 36)
  let cost := 4 * W
  cost = 3624 ∧ h = 3 := sorry

end find_door_height_l1330_133087


namespace roots_of_quadratic_l1330_133040

theorem roots_of_quadratic (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a + b + c = 0) (h₂ : a - b + c = 0) :
  (a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) ∧ (a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) :=
sorry

end roots_of_quadratic_l1330_133040


namespace f_sum_zero_l1330_133068

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ := sorry

-- Define hypotheses based on the problem's conditions
axiom f_cube (x : ℝ) : f (x ^ 3) = (f x) ^ 3
axiom f_inj (x1 x2 : ℝ) (h : x1 ≠ x2) : f x1 ≠ f x2

-- State the proof problem
theorem f_sum_zero : f 0 + f 1 + f (-1) = 0 :=
sorry

end f_sum_zero_l1330_133068


namespace sequence_general_term_l1330_133030

-- Define the sequence and the sum of the sequence
def Sn (n : ℕ) : ℕ := 3 + 2^n

def an (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n - 1)

-- Proposition stating the equivalence
theorem sequence_general_term (n : ℕ) : 
  (n = 1 → an n = 5) ∧ (n ≠ 1 → an n = 2^(n - 1)) :=
by 
  sorry

end sequence_general_term_l1330_133030


namespace janets_shampoo_days_l1330_133028

-- Definitions from the problem conditions
def rose_shampoo := 1 / 3
def jasmine_shampoo := 1 / 4
def daily_usage := 1 / 12

-- Define the total shampoo and the days lasts
def total_shampoo := rose_shampoo + jasmine_shampoo
def days_lasts := total_shampoo / daily_usage

-- The theorem to be proved
theorem janets_shampoo_days : days_lasts = 7 :=
by sorry

end janets_shampoo_days_l1330_133028


namespace library_growth_rate_l1330_133073

theorem library_growth_rate (C_2022 C_2024: ℝ) (h₁ : C_2022 = 100000) (h₂ : C_2024 = 144000) :
  ∃ x : ℝ, (1 + x) ^ 2 = C_2024 / C_2022 ∧ x = 0.2 := 
by {
  sorry
}

end library_growth_rate_l1330_133073


namespace range_of_a_l1330_133035

variable (a : ℝ)

def proposition_p := ∀ x : ℝ, a * x^2 - 2 * x + 1 > 0
def proposition_q := ∀ x : ℝ, x ∈ Set.Icc (1/2 : ℝ) (2 : ℝ) → x + (1 / x) > a

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l1330_133035


namespace value_of_2_pow_b_l1330_133037

theorem value_of_2_pow_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h1 : (2 ^ a) ^ b = 2 ^ 2) (h2 : 2 ^ a * 2 ^ b = 8) : 2 ^ b = 4 :=
by
  sorry

end value_of_2_pow_b_l1330_133037


namespace probability_of_high_value_hand_l1330_133075

noncomputable def bridge_hand_probability : ℚ :=
  let total_combinations : ℕ := Nat.choose 16 4
  let favorable_combinations : ℕ := 1 + 16 + 16 + 16 + 36 + 96 + 16
  favorable_combinations / total_combinations

theorem probability_of_high_value_hand : bridge_hand_probability = 197 / 1820 := by
  sorry

end probability_of_high_value_hand_l1330_133075


namespace least_cans_required_l1330_133049

def maaza : ℕ := 20
def pepsi : ℕ := 144
def sprite : ℕ := 368

def GCD (a b : ℕ) : ℕ := Nat.gcd a b

def total_cans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd_maaza_pepsi := GCD maaza pepsi
  let gcd_all := GCD gcd_maaza_pepsi sprite
  (maaza / gcd_all) + (pepsi / gcd_all) + (sprite / gcd_all)

theorem least_cans_required : total_cans maaza pepsi sprite = 133 := by
  sorry

end least_cans_required_l1330_133049


namespace double_apply_l1330_133005

def op1 (x : ℤ) : ℤ := 9 - x 
def op2 (x : ℤ) : ℤ := x - 9

theorem double_apply (x : ℤ) : op1 (op2 x) = 3 := by
  sorry

end double_apply_l1330_133005


namespace find_x_l1330_133041

def operation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) :
  operation 6 (operation 4 x) = 480 ↔ x = 5 := 
by
  sorry

end find_x_l1330_133041


namespace cheesecake_factory_hours_per_day_l1330_133020

theorem cheesecake_factory_hours_per_day
  (wage_per_hour : ℝ)
  (days_per_week : ℝ)
  (weeks : ℝ)
  (combined_savings : ℝ)
  (robbie_saves : ℝ)
  (jaylen_saves : ℝ)
  (miranda_saves : ℝ)
  (h : ℝ) :
  wage_per_hour = 10 → days_per_week = 5 → weeks = 4 → combined_savings = 3000 →
  robbie_saves = 2/5 → jaylen_saves = 3/5 → miranda_saves = 1/2 →
  (robbie_saves * (wage_per_hour * h * days_per_week) +
  jaylen_saves * (wage_per_hour * h * days_per_week) +
  miranda_saves * (wage_per_hour * h * days_per_week)) * weeks = combined_savings →
  h = 10 :=
by
  intros hwage hweek hweeks hsavings hrobbie hjaylen hmiranda heq
  sorry

end cheesecake_factory_hours_per_day_l1330_133020


namespace min_value_fraction_l1330_133034

-- We start by defining the geometric sequence and the given conditions
variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {a1 : ℝ} (h_pos : ∀ n, 0 < a n)
variable (h_geo : ∀ n, a (n + 1) = a n * r)
variable (h_a7 : a 7 = a 6 + 2 * a 5)
variable (h_am_an : ∃ m n, a m * a n = 16 * (a 1)^2)

theorem min_value_fraction : 
  ∃ (m n : ℕ), (a m * a n = 16 * (a 1)^2 ∧ (1/m) + (4/n) = 1) :=
sorry

end min_value_fraction_l1330_133034


namespace x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l1330_133069

-- Define the context and main statement
theorem x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta
  (θ : ℝ)
  (hθ₁ : 0 < θ)
  (hθ₂ : θ < (π / 2))
  {x : ℝ}
  (hx : x + 1 / x = 2 * Real.sin θ)
  (n : ℕ) (hn : 0 < n) :
  x^n + 1 / x^n = 2 * Real.sin (n * θ) :=
sorry

end x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l1330_133069


namespace right_triangle_with_integer_sides_l1330_133089

theorem right_triangle_with_integer_sides (k : ℤ) :
  ∃ (a b c : ℤ), a = 2*k+1 ∧ b = 2*k*(k+1) ∧ c = 2*k^2+2*k+1 ∧ (a^2 + b^2 = c^2) ∧ (c = a + 1) := by
  sorry

end right_triangle_with_integer_sides_l1330_133089


namespace solve_quadratic_l1330_133070

theorem solve_quadratic (x : ℚ) (h_pos : x > 0) (h_eq : 3 * x^2 + 8 * x - 35 = 0) : 
    x = 7/3 :=
by
    sorry

end solve_quadratic_l1330_133070


namespace arithmetic_seq_terms_greater_than_50_l1330_133071

theorem arithmetic_seq_terms_greater_than_50 :
  let a_n (n : ℕ) := 17 + (n-1) * 4
  let num_terms := (19 - 10) + 1
  ∀ (a_n : ℕ → ℕ), ((a_n 1 = 17) ∧ (∃ k, a_n k = 89) ∧ (∀ n, a_n (n + 1) = a_n n + 4)) →
  ∃ m, m = num_terms ∧ ∀ n, (10 ≤ n ∧ n ≤ 19) → a_n n > 50 :=
by
  sorry

end arithmetic_seq_terms_greater_than_50_l1330_133071


namespace complement_intersection_l1330_133060

-- Definitions of the sets as given in the problem
namespace ProofProblem

def U : Set ℤ := {-2, -1, 0, 1, 2}
def M : Set ℤ := {y | y > 0}
def N : Set ℤ := {x | x = -1 ∨ x = 2}

theorem complement_intersection :
  (U \ M) ∩ N = {-1} :=
by
  sorry

end ProofProblem

end complement_intersection_l1330_133060


namespace B_completes_in_40_days_l1330_133064

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end B_completes_in_40_days_l1330_133064


namespace parabola_focus_coordinates_l1330_133094

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 4 * x^2) : (0, 1/16) = (0, 1/16) :=
by
  sorry

end parabola_focus_coordinates_l1330_133094


namespace green_dots_fifth_row_l1330_133011

variable (R : ℕ → ℕ)

-- Define the number of green dots according to the pattern
def pattern (n : ℕ) : ℕ := 3 * n

-- Define conditions for rows
axiom row_1 : R 1 = 3
axiom row_2 : R 2 = 6
axiom row_3 : R 3 = 9
axiom row_4 : R 4 = 12

-- The theorem
theorem green_dots_fifth_row : R 5 = 15 :=
by
  -- Row 5 follows the pattern and should satisfy the condition R 5 = R 4 + 3
  sorry

end green_dots_fifth_row_l1330_133011
