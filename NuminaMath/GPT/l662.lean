import Mathlib

namespace NUMINAMATH_GPT_amount_A_received_l662_66217

-- Define the conditions
def total_amount : ℕ := 600
def ratio_a : ℕ := 1
def ratio_b : ℕ := 2

-- Define the total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b

-- Define the value of one part
def value_per_part : ℕ := total_amount / total_parts

-- Define the amount A gets
def amount_A_gets : ℕ := ratio_a * value_per_part

-- Lean statement to prove
theorem amount_A_received : amount_A_gets = 200 := by
  sorry

end NUMINAMATH_GPT_amount_A_received_l662_66217


namespace NUMINAMATH_GPT_polynomial_nonnegative_iff_eq_l662_66257

variable {R : Type} [LinearOrderedField R]

def polynomial_p (x a b c : R) : R :=
  (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem polynomial_nonnegative_iff_eq (a b c : R) :
  (∀ x : R, polynomial_p x a b c ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_polynomial_nonnegative_iff_eq_l662_66257


namespace NUMINAMATH_GPT_sum_of_coefficients_l662_66214

theorem sum_of_coefficients (a b : ℝ) (h : ∀ x : ℝ, (x > 1 ∧ x < 4) ↔ (ax^2 + bx - 2 > 0)) :
  a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l662_66214


namespace NUMINAMATH_GPT_painting_frame_ratio_proof_l662_66246

def framed_painting_ratio (x : ℝ) : Prop :=
  let width := 20
  let height := 20
  let side_border := x
  let top_bottom_border := 3 * x
  let framed_width := width + 2 * side_border
  let framed_height := height + 2 * top_bottom_border
  let painting_area := width * height
  let frame_area := painting_area
  let total_area := framed_width * framed_height - painting_area
  total_area = frame_area ∧ (width + 2 * side_border) ≤ (height + 2 * top_bottom_border) → 
  framed_width / framed_height = 4 / 7

theorem painting_frame_ratio_proof (x : ℝ) (h : framed_painting_ratio x) : (20 + 2 * x) / (20 + 6 * x) = 4 / 7 :=
  sorry

end NUMINAMATH_GPT_painting_frame_ratio_proof_l662_66246


namespace NUMINAMATH_GPT_participants_l662_66213

variable {A B C D : Prop}

theorem participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) :
  (¬A ∧ C ∧ B ∧ ¬D) ∨ ¬B :=
by
  -- The proof is not provided
  sorry

end NUMINAMATH_GPT_participants_l662_66213


namespace NUMINAMATH_GPT_max_value_of_a_l662_66243

theorem max_value_of_a :
  ∀ (m : ℚ) (x : ℤ),
    (0 < x ∧ x ≤ 50) →
    (1 / 2 < m ∧ m < 25 / 49) →
    (∀ k : ℤ, m * x + 3 ≠ k) →
  m < 25 / 49 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_l662_66243


namespace NUMINAMATH_GPT_kitchen_length_l662_66219

-- Define the conditions
def tile_area : ℕ := 6
def kitchen_width : ℕ := 48
def number_of_tiles : ℕ := 96

-- The total area is the number of tiles times the area of each tile
def total_area : ℕ := number_of_tiles * tile_area

-- Statement to prove the length of the kitchen
theorem kitchen_length : (total_area / kitchen_width) = 12 :=
by
  sorry

end NUMINAMATH_GPT_kitchen_length_l662_66219


namespace NUMINAMATH_GPT_negatively_added_marks_l662_66298

theorem negatively_added_marks 
  (correct_marks_per_question : ℝ) 
  (total_marks : ℝ) 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (x : ℝ) 
  (h1 : correct_marks_per_question = 4)
  (h2 : total_marks = 420)
  (h3 : total_questions = 150)
  (h4 : correct_answers = 120) 
  (h5 : total_marks = (correct_answers * correct_marks_per_question) - ((total_questions - correct_answers) * x)) :
  x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_negatively_added_marks_l662_66298


namespace NUMINAMATH_GPT_custom_op_example_l662_66290

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_example : (custom_op 7 4) - (custom_op 4 7) = -9 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_example_l662_66290


namespace NUMINAMATH_GPT_find_other_number_l662_66258

theorem find_other_number (A B : ℕ) (hcf : ℕ) (lcm : ℕ) 
  (H1 : hcf = 12) 
  (H2 : lcm = 312) 
  (H3 : A = 24) 
  (H4 : hcf * lcm = A * B) : 
  B = 156 :=
by sorry

end NUMINAMATH_GPT_find_other_number_l662_66258


namespace NUMINAMATH_GPT_height_of_picture_frame_l662_66240

-- Definitions of lengths and perimeter
def length : ℕ := 10
def perimeter : ℕ := 44

-- Perimeter formula for a rectangle
def rectangle_perimeter (L H : ℕ) : ℕ := 2 * (L + H)

-- Theorem statement: Proving the height is 12 inches based on given conditions
theorem height_of_picture_frame : ∃ H : ℕ, rectangle_perimeter length H = perimeter ∧ H = 12 := by
  sorry

end NUMINAMATH_GPT_height_of_picture_frame_l662_66240


namespace NUMINAMATH_GPT_diagonal_inequality_l662_66272

theorem diagonal_inequality (A B C D : ℝ × ℝ) (h1 : A.1 = 0) (h2 : B.1 = 0) (h3 : C.2 = 0) (h4 : D.2 = 0) 
  (ha : A.2 < B.2) (hd : D.1 < C.1) : 
  (Real.sqrt (A.2^2 + C.1^2)) * (Real.sqrt (B.2^2 + D.1^2)) > (Real.sqrt (A.2^2 + D.1^2)) * (Real.sqrt (B.2^2 + C.1^2)) :=
sorry

end NUMINAMATH_GPT_diagonal_inequality_l662_66272


namespace NUMINAMATH_GPT_fraction_exp_3_4_cubed_l662_66227

def fraction_exp (a b n : ℕ) : ℚ := (a : ℚ) ^ n / (b : ℚ) ^ n

theorem fraction_exp_3_4_cubed : fraction_exp 3 4 3 = 27 / 64 :=
by
  sorry

end NUMINAMATH_GPT_fraction_exp_3_4_cubed_l662_66227


namespace NUMINAMATH_GPT_geometric_sequence_product_l662_66223

-- Defining the geometric sequence and the equation
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def satisfies_quadratic_roots (a : ℕ → ℝ) : Prop :=
  (a 2 = -1 ∧ a 18 = -16 / (-1 + 16 / -1) ∨
  a 18 = -1 ∧ a 2 = -16 / (-1 + 16 / -1))

-- Problem statement
theorem geometric_sequence_product (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : satisfies_quadratic_roots a) : 
  a 3 * a 10 * a 17 = -64 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l662_66223


namespace NUMINAMATH_GPT_nuts_per_cookie_l662_66284

theorem nuts_per_cookie (h1 : (1/4:ℝ) * 60 = 15)
(h2 : (0.40:ℝ) * 60 = 24)
(h3 : 60 - 15 - 24 = 21)
(h4 : 72 / (15 + 21) = 2) :
72 / 36 = 2 := by
suffices h : 72 / 36 = 2 from h
exact h4

end NUMINAMATH_GPT_nuts_per_cookie_l662_66284


namespace NUMINAMATH_GPT_sum_of_number_and_conjugate_l662_66287

noncomputable def x : ℝ := 16 - Real.sqrt 2023
noncomputable def y : ℝ := 16 + Real.sqrt 2023

theorem sum_of_number_and_conjugate : x + y = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_conjugate_l662_66287


namespace NUMINAMATH_GPT_time_A_to_complete_race_l662_66224

noncomputable def km_race_time (V_B : ℕ) : ℚ :=
  940 / V_B

theorem time_A_to_complete_race : km_race_time 6 = 156.67 := by
  sorry

end NUMINAMATH_GPT_time_A_to_complete_race_l662_66224


namespace NUMINAMATH_GPT_length_of_ON_l662_66294

noncomputable def proof_problem : Prop :=
  let hyperbola := { x : ℝ × ℝ | x.1 ^ 2 - x.2 ^ 2 = 1 }
  ∃ (F1 F2 P : ℝ × ℝ) (O : ℝ × ℝ) (N : ℝ × ℝ),
    O = (0, 0) ∧
    P ∈ hyperbola ∧
    N = ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2) ∧
    dist P F1 = 5 ∧
    ∃ r : ℝ, r = 1.5 ∧ (dist O N = r)

theorem length_of_ON : proof_problem :=
sorry

end NUMINAMATH_GPT_length_of_ON_l662_66294


namespace NUMINAMATH_GPT_range_of_k_l662_66248

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Define the condition that the function does not pass through the third quadrant
def does_not_pass_third_quadrant (k : ℝ) : Prop :=
  ∀ x : ℝ, (x < 0 ∧ linear_function k x < 0) → false

-- Theorem statement proving the range of k
theorem range_of_k (k : ℝ) : does_not_pass_third_quadrant k ↔ (0 ≤ k ∧ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l662_66248


namespace NUMINAMATH_GPT_minimize_f_l662_66265

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + (Real.sin x)^2

theorem minimize_f :
  ∃ x : ℝ, (-π / 4 < x ∧ x ≤ π / 2) ∧
  ∀ y : ℝ, (-π / 4 < y ∧ y ≤ π / 2) → f y ≥ f x ∧ f x = 1 ∧ x = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_f_l662_66265


namespace NUMINAMATH_GPT_total_surfers_is_60_l662_66205

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end NUMINAMATH_GPT_total_surfers_is_60_l662_66205


namespace NUMINAMATH_GPT_find_A_l662_66299

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_A_l662_66299


namespace NUMINAMATH_GPT_neg_disj_imp_neg_conj_l662_66269

theorem neg_disj_imp_neg_conj (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end NUMINAMATH_GPT_neg_disj_imp_neg_conj_l662_66269


namespace NUMINAMATH_GPT_production_units_l662_66238

-- Define the production function U
def U (women hours days : ℕ) : ℕ := women * hours * days

-- State the theorem
theorem production_units (x z : ℕ) (hx : ¬ x = 0) :
  U z z z = (z^3 / x) :=
  sorry

end NUMINAMATH_GPT_production_units_l662_66238


namespace NUMINAMATH_GPT_triangle_side_lengths_l662_66247

theorem triangle_side_lengths (a b c r : ℕ) (h : a / b / c = 25 / 29 / 36) (hinradius : r = 232) :
  (a = 725 ∧ b = 841 ∧ c = 1044) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l662_66247


namespace NUMINAMATH_GPT_solution_set_equivalence_l662_66242

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative : ∀ x : ℝ, deriv f x > 1 - f x
axiom f_at_0 : f 0 = 3

theorem solution_set_equivalence :
  {x : ℝ | (Real.exp x) * f x > (Real.exp x) + 2} = {x : ℝ | x > 0} :=
by sorry

end NUMINAMATH_GPT_solution_set_equivalence_l662_66242


namespace NUMINAMATH_GPT_harry_james_payment_l662_66225

theorem harry_james_payment (x y H : ℝ) (h1 : H - 12 = 44 / y) (h2 : y > 1) (h3 : H != 12 + 44/3) : H = 23 ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_harry_james_payment_l662_66225


namespace NUMINAMATH_GPT_least_5_digit_number_divisible_by_15_25_40_75_125_140_l662_66253

theorem least_5_digit_number_divisible_by_15_25_40_75_125_140 : 
  ∃ n : ℕ, (10000 ≤ n) ∧ (n < 100000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ (125 ∣ n) ∧ (140 ∣ n) ∧ (n = 21000) :=
by
  sorry

end NUMINAMATH_GPT_least_5_digit_number_divisible_by_15_25_40_75_125_140_l662_66253


namespace NUMINAMATH_GPT_hurleys_age_l662_66283

-- Definitions and conditions
variable (H R : ℕ)
variable (cond1 : R - H = 20)
variable (cond2 : (R + 40) + (H + 40) = 128)

-- Theorem statement
theorem hurleys_age (H R : ℕ) (cond1 : R - H = 20) (cond2 : (R + 40) + (H + 40) = 128) : H = 14 := 
by
  sorry

end NUMINAMATH_GPT_hurleys_age_l662_66283


namespace NUMINAMATH_GPT_expand_and_simplify_fraction_l662_66235

theorem expand_and_simplify_fraction (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * ((7 / (x^2)) + 15 * (x^3) - 4 * x) = (3 / (x^2)) + (45 * (x^3) / 7) - (12 * x / 7) :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_fraction_l662_66235


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l662_66222

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l662_66222


namespace NUMINAMATH_GPT_dividend_rate_l662_66277

theorem dividend_rate (face_value market_value expected_interest interest_rate : ℝ)
  (h1 : face_value = 52)
  (h2 : expected_interest = 0.12)
  (h3 : market_value = 39)
  : ((expected_interest * market_value) / face_value) * 100 = 9 := by
  sorry

end NUMINAMATH_GPT_dividend_rate_l662_66277


namespace NUMINAMATH_GPT_clock_equiv_4_cubic_l662_66296

theorem clock_equiv_4_cubic :
  ∃ x : ℕ, x > 3 ∧ x % 12 = (x^3) % 12 ∧ (∀ y : ℕ, y > 3 ∧ y % 12 = (y^3) % 12 → x ≤ y) :=
by
  use 4
  sorry

end NUMINAMATH_GPT_clock_equiv_4_cubic_l662_66296


namespace NUMINAMATH_GPT_problem_l662_66210

def a : ℝ := (-2)^2002
def b : ℝ := (-2)^2003

theorem problem : a + b = -2^2002 := by
  sorry

end NUMINAMATH_GPT_problem_l662_66210


namespace NUMINAMATH_GPT_bob_raise_per_hour_l662_66271

theorem bob_raise_per_hour
  (hours_per_week : ℕ := 40)
  (monthly_housing_reduction : ℤ := 60)
  (weekly_earnings_increase : ℤ := 5)
  (weeks_per_month : ℕ := 4) :
  ∃ (R : ℚ), 40 * R - (monthly_housing_reduction / weeks_per_month) + weekly_earnings_increase = 0 ∧
              R = 0.25 := 
by
  sorry

end NUMINAMATH_GPT_bob_raise_per_hour_l662_66271


namespace NUMINAMATH_GPT_domain_of_c_is_all_reals_l662_66276

theorem domain_of_c_is_all_reals (k : ℝ) : 
  (∀ x : ℝ, -3 * x^2 + 5 * x + k ≠ 0) ↔ k < -(25 / 12) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_c_is_all_reals_l662_66276


namespace NUMINAMATH_GPT_range_of_a_for_function_is_real_l662_66273

noncomputable def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 4 * x + a - 3

theorem range_of_a_for_function_is_real :
  (∀ x : ℝ, quadratic_expr a x > 0) → 0 ≤ a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_function_is_real_l662_66273


namespace NUMINAMATH_GPT_rectangle_length_l662_66292

/--
The perimeter of a rectangle is 150 cm. The length is 15 cm greater than the width.
This theorem proves that the length of the rectangle is 45 cm under these conditions.
-/
theorem rectangle_length (P w l : ℝ) (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : l = 45 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l662_66292


namespace NUMINAMATH_GPT_log_sum_exp_log_sub_l662_66200

theorem log_sum : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1 := 
by sorry

theorem exp_log_sub : Real.exp (Real.log 3 / Real.log 2 * Real.log 2) - Real.exp (Real.log 8 / 3) = 1 := 
by sorry

end NUMINAMATH_GPT_log_sum_exp_log_sub_l662_66200


namespace NUMINAMATH_GPT_cube_surface_area_example_l662_66241

def cube_surface_area (V : ℝ) (S : ℝ) : Prop :=
  (∃ s : ℝ, s ^ 3 = V ∧ S = 6 * s ^ 2)

theorem cube_surface_area_example : cube_surface_area 8 24 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_example_l662_66241


namespace NUMINAMATH_GPT_value_of_v3_at_neg4_l662_66221

def poly (x : ℤ) : ℤ := (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem value_of_v3_at_neg4 : poly (-4) = -49 := 
by
  sorry

end NUMINAMATH_GPT_value_of_v3_at_neg4_l662_66221


namespace NUMINAMATH_GPT_inverse_function_point_l662_66288

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.LeftInverse f f⁻¹) (h_point : f 2 = -1) : f⁻¹ (-1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_point_l662_66288


namespace NUMINAMATH_GPT_max_integer_value_l662_66256

theorem max_integer_value (x : ℝ) : 
  ∃ m : ℤ, ∀ (x : ℝ), (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ m ∧ m = 41 :=
by sorry

end NUMINAMATH_GPT_max_integer_value_l662_66256


namespace NUMINAMATH_GPT_parity_of_f_find_a_l662_66232

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x + a * Real.exp (-x)

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a ↔ a = 1 ∨ a = -1) ∧
  (∀ x : ℝ, f (-x) a = -f x a ↔ a = -1) ∧
  (∀ x : ℝ, ¬(f (-x) a = f x a) ∧ ¬(f (-x) a = -f x a) ↔ ¬(a = 1 ∨ a = -1)) :=
by
  sorry

theorem find_a (h : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x a ≥ f 0 a) : 
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_parity_of_f_find_a_l662_66232


namespace NUMINAMATH_GPT_valve_XY_time_correct_l662_66209

-- Given conditions
def valve_rates (x y z : ℝ) := (x + y + z = 1/2 ∧ x + z = 1/4 ∧ y + z = 1/3)
def total_fill_time (t : ℝ) (x y : ℝ) := t = 1 / (x + y)

-- The proof problem
theorem valve_XY_time_correct (x y z : ℝ) (t : ℝ) 
  (h : valve_rates x y z) : total_fill_time t x y → t = 2.4 :=
by
  -- Assume h defines the rates
  have h1 : x + y + z = 1/2 := h.1
  have h2 : x + z = 1/4 := h.2.1
  have h3 : y + z = 1/3 := h.2.2
  
  sorry

end NUMINAMATH_GPT_valve_XY_time_correct_l662_66209


namespace NUMINAMATH_GPT_vehicle_flow_mod_15_l662_66216

theorem vehicle_flow_mod_15
  (vehicle_length : ℝ := 5)
  (max_speed : ℕ := 100)
  (speed_interval : ℕ := 10)
  (distance_multiplier : ℕ := 10)
  (N : ℕ := 2000) :
  (N % 15) = 5 := 
sorry

end NUMINAMATH_GPT_vehicle_flow_mod_15_l662_66216


namespace NUMINAMATH_GPT_min_AC_plus_BD_l662_66234

theorem min_AC_plus_BD (k : ℝ) (h : k ≠ 0) :
  (8 + 8 / k^2) + (8 + 2 * k^2) ≥ 24 :=
by
  sorry -- skipping the proof

end NUMINAMATH_GPT_min_AC_plus_BD_l662_66234


namespace NUMINAMATH_GPT_proof_min_k_l662_66211

-- Define the number of teachers
def num_teachers : ℕ := 200

-- Define what it means for a teacher to send a message to another teacher.
-- Represent this as a function where each teacher sends a message to exactly one other teacher.
def sends_message (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∀ i : Fin num_teachers, ∃ j : Fin num_teachers, teachers i = j

-- Define the main proposition: there exists a group of 67 teachers where no one sends a message to anyone else in the group.
def min_k (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∃ (k : ℕ) (reps : Fin k → Fin num_teachers), k ≥ 67 ∧
  ∀ (i j : Fin k), i ≠ j → teachers (reps i) ≠ reps j

theorem proof_min_k : ∀ (teachers : Fin num_teachers → Fin num_teachers),
  sends_message teachers → min_k teachers :=
sorry

end NUMINAMATH_GPT_proof_min_k_l662_66211


namespace NUMINAMATH_GPT_final_problem_l662_66244

-- Define the function f
def f (x p q : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition ①: When q=0, f(x) is an odd function
def prop1 (p : ℝ) : Prop :=
  ∀ x : ℝ, f x p 0 = - f (-x) p 0

-- Proposition ②: The graph of y=f(x) is symmetric with respect to the point (0,q)
def prop2 (p q : ℝ) : Prop :=
  ∀ x : ℝ, f x p q = f (-x) p q + 2 * q

-- Proposition ③: When p=0 and q > 0, the equation f(x)=0 has exactly one real root
def prop3 (q : ℝ) : Prop :=
  q > 0 → ∃! x : ℝ, f x 0 q = 0

-- Proposition ④: The equation f(x)=0 has at most two real roots
def prop4 (p q : ℝ) : Prop :=
  ∀ x1 x2 x3 : ℝ, f x1 p q = 0 ∧ f x2 p q = 0 ∧ f x3 p q = 0 → x1 = x2 ∨ x1 = x3 ∨ x2 = x3

-- The final problem to prove that propositions ①, ②, and ③ are true and proposition ④ is false
theorem final_problem (p q : ℝ) :
  prop1 p ∧ prop2 p q ∧ prop3 q ∧ ¬prop4 p q :=
sorry

end NUMINAMATH_GPT_final_problem_l662_66244


namespace NUMINAMATH_GPT_find_a_if_even_function_l662_66274

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_a_if_even_function_l662_66274


namespace NUMINAMATH_GPT_simplify_vectors_l662_66282

variables {Point : Type} [AddGroup Point] (A B C D : Point)

def vector (P Q : Point) : Point := Q - P

theorem simplify_vectors :
  vector A B + vector B C - vector A D = vector D C :=
by
  sorry

end NUMINAMATH_GPT_simplify_vectors_l662_66282


namespace NUMINAMATH_GPT_platform_length_l662_66207

theorem platform_length (speed_kmh : ℕ) (time_min : ℕ) (train_length_m : ℕ) (distance_covered_m : ℕ) : 
  speed_kmh = 90 → time_min = 1 → train_length_m = 750 → distance_covered_m = 1500 →
  train_length_m + (distance_covered_m - train_length_m) = 750 + (1500 - 750) :=
by sorry

end NUMINAMATH_GPT_platform_length_l662_66207


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_100_to_110_l662_66202

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_100_to_110_l662_66202


namespace NUMINAMATH_GPT_compute_R_at_3_l662_66215

def R (x : ℝ) := 3 * x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem compute_R_at_3 : R 3 = 283 := by
  sorry

end NUMINAMATH_GPT_compute_R_at_3_l662_66215


namespace NUMINAMATH_GPT_polygon_interior_plus_exterior_l662_66275

theorem polygon_interior_plus_exterior (n : ℕ) 
  (h : (n - 2) * 180 + 60 = 1500) : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_interior_plus_exterior_l662_66275


namespace NUMINAMATH_GPT_proof_simplify_expression_l662_66278

noncomputable def simplify_expression (a b : ℝ) : ℝ :=
  (a / b + b / a)^2 - 1 / (a^2 * b^2)

theorem proof_simplify_expression 
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = a + b) :
  simplify_expression a b = 2 / (a * b) := by
  sorry

end NUMINAMATH_GPT_proof_simplify_expression_l662_66278


namespace NUMINAMATH_GPT_length_of_bridge_l662_66249

noncomputable def speed_kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

def total_distance_covered (speed_mps : ℝ) (time_s : ℕ) : ℝ := speed_mps * time_s

def bridge_length (total_distance : ℝ) (train_length : ℝ) : ℝ := total_distance - train_length

theorem length_of_bridge (train_length : ℝ) (time_s : ℕ) (speed_kmh : ℕ) :
  bridge_length (total_distance_covered (speed_kmh_to_mps speed_kmh) time_s) train_length = 299.9 :=
by
  have speed_mps := speed_kmh_to_mps speed_kmh
  have total_distance := total_distance_covered speed_mps time_s
  have length_of_bridge := bridge_length total_distance train_length
  sorry

end NUMINAMATH_GPT_length_of_bridge_l662_66249


namespace NUMINAMATH_GPT_true_false_questions_count_l662_66291

/-- 
 In an answer key for a quiz, there are some true-false questions followed by 3 multiple-choice questions with 4 answer choices each. 
 The correct answers to all true-false questions cannot be the same. 
 There are 384 ways to write the answer key. How many true-false questions are there?
-/
theorem true_false_questions_count : 
  ∃ n : ℕ, 2^n - 2 = 6 ∧ (2^n - 2) * 4^3 = 384 := 
sorry

end NUMINAMATH_GPT_true_false_questions_count_l662_66291


namespace NUMINAMATH_GPT_unique_position_all_sequences_one_l662_66260

-- Define the main theorem
theorem unique_position_all_sequences_one (n : ℕ) (sequences : Fin (2^(n-1)) → Fin n → Bool) :
  (∀ a b c : Fin (2^(n-1)), ∃ p : Fin n, sequences a p = true ∧ sequences b p = true ∧ sequences c p = true) →
  ∃! p : Fin n, ∀ i : Fin (2^(n-1)), sequences i p = true :=
by
  sorry

end NUMINAMATH_GPT_unique_position_all_sequences_one_l662_66260


namespace NUMINAMATH_GPT_determine_g_l662_66218

def real_function (g : ℝ → ℝ) :=
  ∀ c d : ℝ, g (c + d) + g (c - d) = g (c) * g (d) + g (d)

def non_zero_function (g : ℝ → ℝ) :=
  ∃ x : ℝ, g x ≠ 0

theorem determine_g (g : ℝ → ℝ) (h1 : real_function g) (h2 : non_zero_function g) : g 0 = 1 ∧ ∀ x : ℝ, g (-x) = g x := 
sorry

end NUMINAMATH_GPT_determine_g_l662_66218


namespace NUMINAMATH_GPT_first_solution_carbonation_l662_66236

-- Definitions of given conditions in the problem
variable (C : ℝ) -- Percentage of carbonated water in the first solution
variable (L : ℝ) -- Percentage of lemonade in the first solution

-- The second solution is 55% carbonated water and 45% lemonade
def second_solution_carbonated : ℝ := 55
def second_solution_lemonade : ℝ := 45

-- The mixture is 65% carbonated water and 40% of the volume is the first solution
def mixture_carbonated : ℝ := 65
def first_solution_contribution : ℝ := 0.40
def second_solution_contribution : ℝ := 0.60

-- The relationship between the solution components
def equation := first_solution_contribution * C + second_solution_contribution * second_solution_carbonated = mixture_carbonated

-- The statement to prove: C = 80
theorem first_solution_carbonation :
  equation C →
  C = 80 :=
sorry

end NUMINAMATH_GPT_first_solution_carbonation_l662_66236


namespace NUMINAMATH_GPT_my_car_mpg_l662_66270

-- Definitions from the conditions.
def total_miles := 100
def total_gallons := 5

-- The statement we need to prove.
theorem my_car_mpg : (total_miles / total_gallons : ℕ) = 20 :=
by
  sorry

end NUMINAMATH_GPT_my_car_mpg_l662_66270


namespace NUMINAMATH_GPT_sum_of_squares_of_six_odds_not_2020_l662_66289

theorem sum_of_squares_of_six_odds_not_2020 :
  ¬ ∃ a1 a2 a3 a4 a5 a6 : ℤ, (∀ i ∈ [a1, a2, a3, a4, a5, a6], i % 2 = 1) ∧ (a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = 2020) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_six_odds_not_2020_l662_66289


namespace NUMINAMATH_GPT_hash_op_calculation_l662_66254

-- Define the new operation
def hash_op (a b : ℚ) : ℚ :=
  a^2 + a * b - 5

-- Prove that (-3) # 6 = -14
theorem hash_op_calculation : hash_op (-3) 6 = -14 := by
  sorry

end NUMINAMATH_GPT_hash_op_calculation_l662_66254


namespace NUMINAMATH_GPT_rachel_age_is_19_l662_66293

def rachel_and_leah_ages (R L : ℕ) : Prop :=
  (R = L + 4) ∧ (R + L = 34)

theorem rachel_age_is_19 : ∃ L : ℕ, rachel_and_leah_ages 19 L :=
by {
  sorry
}

end NUMINAMATH_GPT_rachel_age_is_19_l662_66293


namespace NUMINAMATH_GPT_solve_for_x_l662_66226

theorem solve_for_x (x y : ℝ) (h1 : 2 * x - 3 * y = 18) (h2 : x + 2 * y = 8) : x = 60 / 7 := sorry

end NUMINAMATH_GPT_solve_for_x_l662_66226


namespace NUMINAMATH_GPT_average_fuel_efficiency_round_trip_l662_66237

noncomputable def average_fuel_efficiency (d1 d2 mpg1 mpg2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let fuel_used := (d1 / mpg1) + (d2 / mpg2)
  total_distance / fuel_used

theorem average_fuel_efficiency_round_trip :
  average_fuel_efficiency 180 180 36 24 = 28.8 :=
by 
  sorry

end NUMINAMATH_GPT_average_fuel_efficiency_round_trip_l662_66237


namespace NUMINAMATH_GPT_quadratic_eq_equal_roots_l662_66263

theorem quadratic_eq_equal_roots (m x : ℝ) (h : (x^2 - m * x + m - 1 = 0) ∧ ((x - 1)^2 = 0)) : 
    m = 2 ∧ ((x = 1 ∧ x = 1)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_equal_roots_l662_66263


namespace NUMINAMATH_GPT_k_l_m_n_values_l662_66251

theorem k_l_m_n_values (k l m n : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m) (hn : 0 < n)
  (hklmn : k + l + m + n = k * m) (hln : k + l + m + n = l * n) :
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end NUMINAMATH_GPT_k_l_m_n_values_l662_66251


namespace NUMINAMATH_GPT_Joan_paid_158_l662_66239

theorem Joan_paid_158 (J K : ℝ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end NUMINAMATH_GPT_Joan_paid_158_l662_66239


namespace NUMINAMATH_GPT_find_sum_s_u_l662_66252

theorem find_sum_s_u (p r s u : ℝ) (q t : ℝ) 
  (h_q : q = 5) 
  (h_t : t = -p - r) 
  (h_sum_imaginary : q + s + u = 4) :
  s + u = -1 := 
sorry

end NUMINAMATH_GPT_find_sum_s_u_l662_66252


namespace NUMINAMATH_GPT_sum_of_first_three_tests_l662_66212

variable (A B C: ℕ)

def scores (A B C test4 : ℕ) : Prop := (A + B + C + test4) / 4 = 85

theorem sum_of_first_three_tests (h : scores A B C 100) : A + B + C = 240 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_first_three_tests_l662_66212


namespace NUMINAMATH_GPT_sound_heard_in_4_seconds_l662_66231

/-- Given the distance between a boy and his friend is 1200 meters,
    the speed of the car is 108 km/hr, and the speed of sound is 330 m/s,
    the duration after which the friend hears the whistle is 4 seconds. -/
theorem sound_heard_in_4_seconds :
  let distance := 1200  -- distance in meters
  let speed_of_car_kmh := 108  -- speed of car in km/hr
  let speed_of_sound := 330  -- speed of sound in m/s
  let speed_of_car := speed_of_car_kmh * 1000 / 3600  -- convert km/hr to m/s
  let effective_speed_of_sound := speed_of_sound - speed_of_car
  let time := distance / effective_speed_of_sound
  time = 4 := 
by
  sorry

end NUMINAMATH_GPT_sound_heard_in_4_seconds_l662_66231


namespace NUMINAMATH_GPT_jack_total_dollars_l662_66233

-- Constants
def initial_dollars : ℝ := 45
def euro_amount : ℝ := 36
def yen_amount : ℝ := 1350
def ruble_amount : ℝ := 1500
def euro_to_dollar : ℝ := 2
def yen_to_dollar : ℝ := 0.009
def ruble_to_dollar : ℝ := 0.013
def transaction_fee_rate : ℝ := 0.01
def spending_rate : ℝ := 0.1

-- Convert each foreign currency to dollars
def euros_to_dollars : ℝ := euro_amount * euro_to_dollar
def yen_to_dollars : ℝ := yen_amount * yen_to_dollar
def rubles_to_dollars : ℝ := ruble_amount * ruble_to_dollar

-- Calculate transaction fees for each currency conversion
def euros_fee : ℝ := euros_to_dollars * transaction_fee_rate
def yen_fee : ℝ := yen_to_dollars * transaction_fee_rate
def rubles_fee : ℝ := rubles_to_dollars * transaction_fee_rate

-- Subtract transaction fees from the converted amounts
def euros_after_fee : ℝ := euros_to_dollars - euros_fee
def yen_after_fee : ℝ := yen_to_dollars - yen_fee
def rubles_after_fee : ℝ := rubles_to_dollars - rubles_fee

-- Calculate total dollars after conversion and fees
def total_dollars_before_spending : ℝ := initial_dollars + euros_after_fee + yen_after_fee + rubles_after_fee

-- Calculate 10% expenditure
def spending_amount : ℝ := total_dollars_before_spending * spending_rate

-- Calculate final amount after spending
def final_amount : ℝ := total_dollars_before_spending - spending_amount

theorem jack_total_dollars : final_amount = 132.85 := by
  sorry

end NUMINAMATH_GPT_jack_total_dollars_l662_66233


namespace NUMINAMATH_GPT_problem_l662_66264

theorem problem (K : ℕ) : 16 ^ 3 * 8 ^ 3 = 2 ^ K → K = 21 := by
  sorry

end NUMINAMATH_GPT_problem_l662_66264


namespace NUMINAMATH_GPT_max_non_managers_l662_66245

theorem max_non_managers (N : ℕ) : (8 / N : ℚ) > 7 / 32 → N ≤ 36 :=
by sorry

end NUMINAMATH_GPT_max_non_managers_l662_66245


namespace NUMINAMATH_GPT_surface_area_LShape_l662_66259

-- Define the structures and conditions
structure UnitCube where
  x : ℕ
  y : ℕ
  z : ℕ

def LShape (cubes : List UnitCube) : Prop :=
  -- Condition 1: Exactly 7 unit cubes
  cubes.length = 7 ∧
  -- Condition 2: 4 cubes in a line along x-axis (bottom row)
  ∃ a b c d : UnitCube, 
    (a.x + 1 = b.x ∧ b.x + 1 = c.x ∧ c.x + 1 = d.x ∧
     a.y = b.y ∧ b.y = c.y ∧ c.y = d.y ∧
     a.z = b.z ∧ b.z = c.z ∧ c.z = d.z) ∧
  -- Condition 3: 3 cubes stacked along z-axis at one end of the row
  ∃ e f g : UnitCube,
    (d.x = e.x ∧ e.x = f.x ∧ f.x = g.x ∧
     d.y = e.y ∧ e.y = f.y ∧ f.y = g.y ∧
     e.z + 1 = f.z ∧ f.z + 1 = g.z)

-- Define the surface area function
def surfaceArea (cubes : List UnitCube) : ℕ :=
  4*7 - 2*3 + 4 -- correct answer calculation according to manual analysis of exposed faces

-- The theorem to be proven
theorem surface_area_LShape : 
  ∀ (cubes : List UnitCube), LShape cubes → surfaceArea cubes = 26 :=
by sorry

end NUMINAMATH_GPT_surface_area_LShape_l662_66259


namespace NUMINAMATH_GPT_Xiaohuo_books_l662_66268

def books_proof_problem : Prop :=
  ∃ (X_H X_Y X_Z : ℕ), 
    (X_H + X_Y + X_Z = 1248) ∧ 
    (X_H = X_Y + 64) ∧ 
    (X_Y = X_Z - 32) ∧ 
    (X_H = 448)

theorem Xiaohuo_books : books_proof_problem :=
by
  sorry

end NUMINAMATH_GPT_Xiaohuo_books_l662_66268


namespace NUMINAMATH_GPT_probability_of_sum_20_is_correct_l662_66261

noncomputable def probability_sum_20 : ℚ :=
  let total_outcomes := 12 * 12
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_20_is_correct :
  probability_sum_20 = 5 / 144 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sum_20_is_correct_l662_66261


namespace NUMINAMATH_GPT_train_length_proof_l662_66279

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 5 / 18
  speed_ms * time_s

theorem train_length_proof : train_length 144 16 = 640 := by
  sorry

end NUMINAMATH_GPT_train_length_proof_l662_66279


namespace NUMINAMATH_GPT_find_m_l662_66206

theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l662_66206


namespace NUMINAMATH_GPT_customer_paid_amount_l662_66204

theorem customer_paid_amount 
  (cost_price : ℝ) 
  (markup_percent : ℝ) 
  (customer_payment : ℝ)
  (h1 : cost_price = 1250) 
  (h2 : markup_percent = 0.60)
  (h3 : customer_payment = cost_price + (markup_percent * cost_price)) :
  customer_payment = 2000 :=
sorry

end NUMINAMATH_GPT_customer_paid_amount_l662_66204


namespace NUMINAMATH_GPT_sum_m_b_eq_neg_five_halves_l662_66203

theorem sum_m_b_eq_neg_five_halves : 
  let x1 := 1 / 2
  let y1 := -1
  let x2 := -1 / 2
  let y2 := 2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = -5 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_sum_m_b_eq_neg_five_halves_l662_66203


namespace NUMINAMATH_GPT_tank_loss_rate_after_first_repair_l662_66228

def initial_capacity : ℕ := 350000
def first_loss_rate : ℕ := 32000
def first_loss_duration : ℕ := 5
def second_loss_duration : ℕ := 10
def filling_rate : ℕ := 40000
def filling_duration : ℕ := 3
def missing_gallons : ℕ := 140000

noncomputable def first_repair_loss_rate := (initial_capacity - (first_loss_rate * first_loss_duration) + (filling_rate * filling_duration) - (initial_capacity - missing_gallons)) / second_loss_duration

theorem tank_loss_rate_after_first_repair : first_repair_loss_rate = 10000 := by sorry

end NUMINAMATH_GPT_tank_loss_rate_after_first_repair_l662_66228


namespace NUMINAMATH_GPT_stork_count_l662_66250

theorem stork_count (B S : ℕ) (h1 : B = 7) (h2 : B = S + 3) : S = 4 := 
by 
  sorry -- Proof to be filled in


end NUMINAMATH_GPT_stork_count_l662_66250


namespace NUMINAMATH_GPT_problem_solution_l662_66201

def f (x : ℤ) : ℤ := 3 * x + 1
def g (x : ℤ) : ℤ := 4 * x - 3

theorem problem_solution :
  (f (g (f 3))) / (g (f (g 3))) = 112 / 109 := by
sorry

end NUMINAMATH_GPT_problem_solution_l662_66201


namespace NUMINAMATH_GPT_minimum_selling_price_l662_66280

def monthly_sales : ℕ := 50
def base_cost : ℕ := 1200
def shipping_cost : ℕ := 20
def store_fee : ℕ := 10000
def repair_fee : ℕ := 5000
def profit_margin : ℕ := 20

def total_monthly_expenses : ℕ := store_fee + repair_fee
def total_cost_per_machine : ℕ := base_cost + shipping_cost + total_monthly_expenses / monthly_sales
def min_selling_price : ℕ := total_cost_per_machine * (1 + profit_margin / 100)

theorem minimum_selling_price : min_selling_price = 1824 := 
by
  sorry 

end NUMINAMATH_GPT_minimum_selling_price_l662_66280


namespace NUMINAMATH_GPT_range_of_b_not_strictly_decreasing_l662_66262

def f (b x : ℝ) : ℝ := -x^3 + b*x^2 - (2*b + 3)*x + 2 - b

theorem range_of_b_not_strictly_decreasing :
  {b : ℝ | ¬(∀ (x1 x2 : ℝ), x1 < x2 → f b x1 > f b x2)} = {b | b < -1 ∨ b > 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_not_strictly_decreasing_l662_66262


namespace NUMINAMATH_GPT_no_such_m_for_equivalence_existence_of_m_for_implication_l662_66297

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_such_m_for_equivalence :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
sorry

theorem existence_of_m_for_implication :
  ∃ m : ℝ, (∀ x : ℝ, S x m → P x) ∧ m ≤ 3 :=
sorry

end NUMINAMATH_GPT_no_such_m_for_equivalence_existence_of_m_for_implication_l662_66297


namespace NUMINAMATH_GPT_find_a_n_l662_66295

noncomputable def is_arithmetic_seq (a b : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = a + n * b

noncomputable def is_geometric_seq (b a : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = b * a ^ n

theorem find_a_n (a b : ℕ) 
  (a_positive : a > 1)
  (b_positive : b > 1)
  (a_seq : ℕ → ℕ)
  (b_seq : ℕ → ℕ)
  (arith_seq : is_arithmetic_seq a b a_seq)
  (geom_seq : is_geometric_seq b a b_seq)
  (init_condition : a_seq 0 < b_seq 0)
  (next_condition : b_seq 1 < a_seq 2)
  (relation_condition : ∀ n, ∃ m, a_seq m + 3 = b_seq n) :
  ∀ n, a_seq n = 5 * n - 3 :=
sorry

end NUMINAMATH_GPT_find_a_n_l662_66295


namespace NUMINAMATH_GPT_Alyssa_initial_puppies_l662_66267

theorem Alyssa_initial_puppies : 
  ∀ (a b c : ℕ), b = 7 → c = 5 → a = b + c → a = 12 := 
by
  intros a b c hb hc hab
  rw [hb, hc] at hab
  exact hab

end NUMINAMATH_GPT_Alyssa_initial_puppies_l662_66267


namespace NUMINAMATH_GPT_alicia_masks_left_l662_66266

theorem alicia_masks_left (T G L : ℕ) (hT : T = 90) (hG : G = 51) (hL : L = T - G) : L = 39 :=
by
  rw [hT, hG] at hL
  exact hL

end NUMINAMATH_GPT_alicia_masks_left_l662_66266


namespace NUMINAMATH_GPT_orthogonal_pairs_in_cube_is_36_l662_66229

-- Define a cube based on its properties, i.e., having vertices, edges, and faces.
structure Cube :=
(vertices : Fin 8 → Fin 3)
(edges : Fin 12 → (Fin 2 → Fin 8))
(faces : Fin 6 → (Fin 4 → Fin 8))

-- Define orthogonal pairs of a cube as an axiom.
axiom orthogonal_line_plane_pairs (c : Cube) : ℕ

-- The main theorem stating the problem's conclusion.
theorem orthogonal_pairs_in_cube_is_36 (c : Cube): orthogonal_line_plane_pairs c = 36 :=
by { sorry }

end NUMINAMATH_GPT_orthogonal_pairs_in_cube_is_36_l662_66229


namespace NUMINAMATH_GPT_solve_equation_solve_inequality_system_l662_66208

theorem solve_equation (x : ℝ) : x^2 - 2 * x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 :=
by
  sorry

theorem solve_inequality_system (x : ℝ) : (4 * (x - 1) < x + 2) ∧ ((x + 7) / 3 > x) ↔ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_solve_inequality_system_l662_66208


namespace NUMINAMATH_GPT_balloons_remaining_l662_66286

-- Define the initial conditions
def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

-- State the theorem
theorem balloons_remaining : initial_balloons - lost_balloons = 7 := by
  -- Add the solution proof steps here
  sorry

end NUMINAMATH_GPT_balloons_remaining_l662_66286


namespace NUMINAMATH_GPT_diana_can_paint_statues_l662_66281

theorem diana_can_paint_statues (total_paint : ℚ) (paint_per_statue : ℚ) 
  (h1 : total_paint = 3 / 6) (h2 : paint_per_statue = 1 / 6) : 
  total_paint / paint_per_statue = 3 :=
by
  sorry

end NUMINAMATH_GPT_diana_can_paint_statues_l662_66281


namespace NUMINAMATH_GPT_xsquared_plus_5x_minus_6_condition_l662_66285

theorem xsquared_plus_5x_minus_6_condition (x : ℝ) : 
  (x^2 + 5 * x - 6 > 0) → (x > 2) ∨ (((x > 1) ∨ (x < -6)) ∧ ¬(x > 2)) := 
sorry

end NUMINAMATH_GPT_xsquared_plus_5x_minus_6_condition_l662_66285


namespace NUMINAMATH_GPT_integer_solutions_to_equation_l662_66230

theorem integer_solutions_to_equation :
  ∃ (x y : ℤ), 2 * x^2 + 8 * y^2 = 17 * x * y - 423 ∧
               ((x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19)) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_equation_l662_66230


namespace NUMINAMATH_GPT_number_of_bricks_required_l662_66255

def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.10
def brick_height : ℝ := 0.075

def wall_length : ℝ := 25.0
def wall_width : ℝ := 2.0
def wall_height : ℝ := 0.75

def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_width * wall_height

theorem number_of_bricks_required :
  wall_volume / brick_volume = 25000 := by
  sorry

end NUMINAMATH_GPT_number_of_bricks_required_l662_66255


namespace NUMINAMATH_GPT_find_ab_l662_66220

variable (a b m n : ℝ)

theorem find_ab (h1 : (a + b)^2 = m) (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l662_66220
