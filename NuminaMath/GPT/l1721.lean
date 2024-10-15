import Mathlib

namespace NUMINAMATH_GPT_lcm_3_15_is_15_l1721_172182

theorem lcm_3_15_is_15 : Nat.lcm 3 15 = 15 :=
sorry

end NUMINAMATH_GPT_lcm_3_15_is_15_l1721_172182


namespace NUMINAMATH_GPT_solve_expression_l1721_172165

theorem solve_expression (x : ℝ) (h : 3 * x - 5 = 10 * x + 9) : 4 * (x + 7) = 20 :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l1721_172165


namespace NUMINAMATH_GPT_find_m_of_parallel_lines_l1721_172193

theorem find_m_of_parallel_lines (m : ℝ) 
  (H1 : ∃ x y : ℝ, m * x + 2 * y + 6 = 0) 
  (H2 : ∃ x y : ℝ, x + (m - 1) * y + m^2 - 1 = 0) : 
  m = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_m_of_parallel_lines_l1721_172193


namespace NUMINAMATH_GPT_symmetric_curve_equation_l1721_172178

theorem symmetric_curve_equation (y x : ℝ) :
  (y^2 = 4 * x) → (y^2 = 16 - 4 * x) :=
sorry

end NUMINAMATH_GPT_symmetric_curve_equation_l1721_172178


namespace NUMINAMATH_GPT_Vihaan_more_nephews_than_Alden_l1721_172113

theorem Vihaan_more_nephews_than_Alden :
  ∃ (a v : ℕ), (a = 100) ∧ (a + v = 260) ∧ (v - a = 60) := by
  sorry

end NUMINAMATH_GPT_Vihaan_more_nephews_than_Alden_l1721_172113


namespace NUMINAMATH_GPT_max_f_value_l1721_172101

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ := (S_n n : ℝ) / ((n + 32) * S_n (n + 1))

theorem max_f_value : ∃ n : ℕ, f n = 1 / 50 := by
  sorry

end NUMINAMATH_GPT_max_f_value_l1721_172101


namespace NUMINAMATH_GPT_ratio_of_volumes_l1721_172134

theorem ratio_of_volumes (r1 r2 : ℝ) (h : (4 * π * r1^2) / (4 * π * r2^2) = 4 / 9) :
  (4/3 * π * r1^3) / (4/3 * π * r2^3) = 8 / 27 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1721_172134


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1721_172160

theorem isosceles_triangle_base_length
  (a : ℕ) (b : ℕ)
  (ha : a = 7) 
  (p : ℕ)
  (hp : p = a + a + b) 
  (hp_perimeter : p = 21) : b = 7 :=
by 
  -- The actual proof will go here, using the provided conditions
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1721_172160


namespace NUMINAMATH_GPT_xy_range_l1721_172150

theorem xy_range (x y : ℝ)
  (h1 : x + y = 1)
  (h2 : 1 / 3 ≤ x ∧ x ≤ 2 / 3) :
  2 / 9 ≤ x * y ∧ x * y ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_xy_range_l1721_172150


namespace NUMINAMATH_GPT_cats_in_studio_count_l1721_172186

theorem cats_in_studio_count :
  (70 + 40 + 30 + 50
  - 25 - 15 - 20 - 28
  + 5 + 10 + 12
  - 8
  + 12) = 129 :=
by sorry

end NUMINAMATH_GPT_cats_in_studio_count_l1721_172186


namespace NUMINAMATH_GPT_other_endpoint_diameter_l1721_172185

theorem other_endpoint_diameter (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hO : O = (2, 3)) (hA : A = (-1, -1)) 
  (h_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : B = (5, 7) := by
  sorry

end NUMINAMATH_GPT_other_endpoint_diameter_l1721_172185


namespace NUMINAMATH_GPT_product_of_numbers_l1721_172148

theorem product_of_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 150)
  (h2 : 7 * x = n)
  (h3 : y - 10 = n)
  (h4 : z + 10 = n) : x * y * z = 48000 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1721_172148


namespace NUMINAMATH_GPT_triangle_obtuse_l1721_172118

def is_obtuse_triangle (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

theorem triangle_obtuse (A B C : ℝ) (h1 : A > 3 * B) (h2 : C < 2 * B) (h3 : A + B + C = 180) : is_obtuse_triangle A B C :=
by sorry

end NUMINAMATH_GPT_triangle_obtuse_l1721_172118


namespace NUMINAMATH_GPT_james_marbles_left_l1721_172197

theorem james_marbles_left :
  ∀ (initial_marbles bags remaining_bags marbles_per_bag left_marbles : ℕ),
  initial_marbles = 28 →
  bags = 4 →
  marbles_per_bag = initial_marbles / bags →
  remaining_bags = bags - 1 →
  left_marbles = remaining_bags * marbles_per_bag →
  left_marbles = 21 :=
by
  intros initial_marbles bags remaining_bags marbles_per_bag left_marbles
  sorry

end NUMINAMATH_GPT_james_marbles_left_l1721_172197


namespace NUMINAMATH_GPT_no_integer_solutions_l1721_172135

theorem no_integer_solutions (x y z : ℤ) (h₀ : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l1721_172135


namespace NUMINAMATH_GPT_range_of_f_l1721_172125

noncomputable def f (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 1)

theorem range_of_f : Set.Icc (-1/4 : ℝ) (1/4) = Set.range f :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1721_172125


namespace NUMINAMATH_GPT_max_value_of_sample_l1721_172115

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end NUMINAMATH_GPT_max_value_of_sample_l1721_172115


namespace NUMINAMATH_GPT_complex_number_conditions_l1721_172173

open Complex Real

noncomputable def is_real (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 = 0

noncomputable def is_imaginary (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 ≠ 0

noncomputable def is_purely_imaginary (a : ℝ) : Prop :=
a ^ 2 - 9 = 0 ∧ a ^ 2 - 2 * a - 15 ≠ 0

theorem complex_number_conditions (a : ℝ) :
  (is_real a ↔ (a = 5 ∨ a = -3))
  ∧ (is_imaginary a ↔ (a ≠ 5 ∧ a ≠ -3))
  ∧ (¬(∃ a : ℝ, is_purely_imaginary a)) :=
by
  sorry

end NUMINAMATH_GPT_complex_number_conditions_l1721_172173


namespace NUMINAMATH_GPT_lucas_fraction_to_emma_l1721_172136

variable (n : ℕ)

-- Define initial stickers
def noah_stickers := n
def emma_stickers := 3 * n
def lucas_stickers := 12 * n

-- Define the final state where each has the same number of stickers
def final_stickers_per_person := (16 * n) / 3

-- Lucas gives some stickers to Emma. Calculate the fraction of Lucas's stickers given to Emma
theorem lucas_fraction_to_emma :
  (7 * n / 3) / (12 * n) = 7 / 36 := by
  sorry

end NUMINAMATH_GPT_lucas_fraction_to_emma_l1721_172136


namespace NUMINAMATH_GPT_average_of_11_numbers_l1721_172112

theorem average_of_11_numbers (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ)
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = 58)
  (h2 : (a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 6 = 65)
  (h3 : a₆ = 78) : 
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 11 = 60 := 
by 
  sorry 

end NUMINAMATH_GPT_average_of_11_numbers_l1721_172112


namespace NUMINAMATH_GPT_intersection_point_l1721_172151

variable (x y : ℝ)

theorem intersection_point :
  (y = 9 / (x^2 + 3)) →
  (x + y = 3) →
  (x = 0) := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_intersection_point_l1721_172151


namespace NUMINAMATH_GPT_determine_range_of_a_l1721_172132

noncomputable def f (x a : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

noncomputable def g (x a : ℝ) : ℝ := f x a - 2*x

theorem determine_range_of_a (a : ℝ) :
  (∀ x, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) →
  (-1 ≤ a ∧ a < 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_range_of_a_l1721_172132


namespace NUMINAMATH_GPT_combined_transformation_matrix_l1721_172147

-- Definitions for conditions
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0], ![0, s]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- Theorem to be proven
theorem combined_transformation_matrix :
  (rotation_matrix_90_ccw * dilation_matrix 4) = ![![0, -4], ![4, 0]] :=
by
  sorry

end NUMINAMATH_GPT_combined_transformation_matrix_l1721_172147


namespace NUMINAMATH_GPT_sum_binomials_l1721_172117

-- Defining binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem sum_binomials : binom 12 4 + binom 10 3 = 615 :=
by
  -- Here we state the problem, and the proof will be left as 'sorry'.
  sorry

end NUMINAMATH_GPT_sum_binomials_l1721_172117


namespace NUMINAMATH_GPT_ratio_B_to_A_l1721_172149

def work_together_rate : Real := 0.75
def days_for_A : Real := 4

theorem ratio_B_to_A : 
  ∃ (days_for_B : Real), 
    (1/days_for_A + 1/days_for_B = work_together_rate) → 
    (days_for_B / days_for_A = 0.5) :=
by 
  sorry

end NUMINAMATH_GPT_ratio_B_to_A_l1721_172149


namespace NUMINAMATH_GPT_simplified_expression_correct_l1721_172190

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) :=
  sorry

end NUMINAMATH_GPT_simplified_expression_correct_l1721_172190


namespace NUMINAMATH_GPT_xy_sum_l1721_172166

theorem xy_sum (x y : ℝ) (h1 : x^3 + 6 * x^2 + 16 * x = -15) (h2 : y^3 + 6 * y^2 + 16 * y = -17) : x + y = -4 :=
by
  -- The proof can be skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_xy_sum_l1721_172166


namespace NUMINAMATH_GPT_monotonic_function_range_maximum_value_condition_function_conditions_l1721_172122

-- Part (1): Monotonicity condition
theorem monotonic_function_range (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0) ↔ (m ≥ 3) := sorry

-- Part (2): Maximum value condition
theorem maximum_value_condition (m : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4) ↔ (m = -2) := sorry

-- Combined statement (optional if you want to show entire problem in one go)
theorem function_conditions (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0 ∧ 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4)) ↔ (m = -2 ∨ m ≥ 3) := sorry

end NUMINAMATH_GPT_monotonic_function_range_maximum_value_condition_function_conditions_l1721_172122


namespace NUMINAMATH_GPT_three_digit_factorions_l1721_172170

def is_factorion (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  n = Nat.factorial a + Nat.factorial b + Nat.factorial c

theorem three_digit_factorions : ∀ n : ℕ, (100 ≤ n ∧ n < 1000) → is_factorion n → n = 145 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_factorions_l1721_172170


namespace NUMINAMATH_GPT_triangle_inequality_l1721_172121

theorem triangle_inequality (S : Finset (ℕ × ℕ)) (m n : ℕ) (hS : S.card = m)
  (h_ab : ∀ (a b : ℕ), (a, b) ∈ S → (1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ a ≠ b)) :
  ∃ (t : Finset (ℕ × ℕ × ℕ)),
    (t.card ≥ (4 * m / (3 * n)) * (m - (n^2) / 4)) ∧
    ∀ (a b c : ℕ), (a, b, c) ∈ t → (a, b) ∈ S ∧ (b, c) ∈ S ∧ (c, a) ∈ S := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1721_172121


namespace NUMINAMATH_GPT_points_on_line_l1721_172120

-- Define the points involved
def point1 : ℝ × ℝ := (4, 10)
def point2 : ℝ × ℝ := (-2, -8)
def candidate_points : List (ℝ × ℝ) := [(1, 1), (0, -1), (2, 3), (-1, -5), (3, 7)]
def correct_points : List (ℝ × ℝ) := [(1, 1), (-1, -5), (3, 7)]

-- Define a function to check if a point lies on the line defined by point1 and point2
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (10 - (-8)) / (4 - (-2))
  let b := 10 - m * 4
  p.2 = m * p.1 + b

-- Main theorem statement
theorem points_on_line :
  ∀ p ∈ candidate_points, p ∈ correct_points ↔ lies_on_line p :=
sorry

end NUMINAMATH_GPT_points_on_line_l1721_172120


namespace NUMINAMATH_GPT_simplify_120_div_180_l1721_172129

theorem simplify_120_div_180 : (120 : ℚ) / 180 = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_simplify_120_div_180_l1721_172129


namespace NUMINAMATH_GPT_gumball_difference_l1721_172139

theorem gumball_difference :
  ∀ C : ℕ, 19 ≤ (29 + C) / 3 ∧ (29 + C) / 3 ≤ 25 →
  (46 - 28) = 18 :=
by
  intros C h
  sorry

end NUMINAMATH_GPT_gumball_difference_l1721_172139


namespace NUMINAMATH_GPT_find_first_number_l1721_172167

theorem find_first_number 
  (second_number : ℕ)
  (increment : ℕ)
  (final_number : ℕ)
  (h1 : second_number = 45)
  (h2 : increment = 11)
  (h3 : final_number = 89)
  : ∃ first_number : ℕ, first_number + increment = second_number := 
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1721_172167


namespace NUMINAMATH_GPT_x_intercept_correct_l1721_172152

noncomputable def x_intercept_of_line : ℝ × ℝ :=
if h : (-4 : ℝ) ≠ 0 then (24 / (-4), 0) else (0, 0)

theorem x_intercept_correct : x_intercept_of_line = (-6, 0) := by
  -- proof will be given here
  sorry

end NUMINAMATH_GPT_x_intercept_correct_l1721_172152


namespace NUMINAMATH_GPT_samantha_tenth_finger_l1721_172106

def g (x : ℕ) : ℕ :=
  match x with
  | 2 => 2
  | _ => 0  -- Assume a simple piecewise definition for the sake of the example.

theorem samantha_tenth_finger : g (2) = 2 :=
by  sorry

end NUMINAMATH_GPT_samantha_tenth_finger_l1721_172106


namespace NUMINAMATH_GPT_find_x_l1721_172184

theorem find_x (x y : ℝ) (h₁ : 2 * x - y = 14) (h₂ : y = 2) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1721_172184


namespace NUMINAMATH_GPT_find_x_for_parallel_vectors_l1721_172175

theorem find_x_for_parallel_vectors :
  ∀ (x : ℚ), (∃ a b : ℚ × ℚ, a = (2 * x, 3) ∧ b = (1, 9) ∧ (∃ k : ℚ, (2 * x, 3) = (k * 1, k * 9))) ↔ x = 1 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_for_parallel_vectors_l1721_172175


namespace NUMINAMATH_GPT_remainder_invariance_l1721_172144

theorem remainder_invariance (S A K : ℤ) (h : ∃ B r : ℤ, S = A * B + r ∧ 0 ≤ r ∧ r < |A|) :
  (∃ B' r' : ℤ, S + A * K = A * B' + r' ∧ r' = r) ∧ (∃ B'' r'' : ℤ, S - A * K = A * B'' + r'' ∧ r'' = r) :=
by
  sorry

end NUMINAMATH_GPT_remainder_invariance_l1721_172144


namespace NUMINAMATH_GPT_trajectory_is_straight_line_l1721_172188

theorem trajectory_is_straight_line (x y : ℝ) (h : x + y = 0) : ∃ m b : ℝ, y = m * x + b :=
by
  use -1
  use 0
  sorry

end NUMINAMATH_GPT_trajectory_is_straight_line_l1721_172188


namespace NUMINAMATH_GPT_right_handed_total_l1721_172127

theorem right_handed_total
  (total_players : ℕ)
  (throwers : ℕ)
  (left_handed_non_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (non_throwers : ℕ)
  (right_handed_non_throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = non_throwers / 3 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  right_handed_throwers = throwers →
  right_handed_throwers + right_handed_non_throwers = 56 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_right_handed_total_l1721_172127


namespace NUMINAMATH_GPT_least_subtraction_divisible_l1721_172100

theorem least_subtraction_divisible (n : ℕ) (h : n = 3830) (lcm_val : ℕ) (hlcm : lcm_val = Nat.lcm (Nat.lcm 3 7) 11) 
(largest_multiple : ℕ) (h_largest : largest_multiple = (n / lcm_val) * lcm_val) :
  ∃ x : ℕ, x = n - largest_multiple ∧ x = 134 := 
by
  sorry

end NUMINAMATH_GPT_least_subtraction_divisible_l1721_172100


namespace NUMINAMATH_GPT_inequality_k_ge_2_l1721_172133

theorem inequality_k_ge_2 {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℤ) (h_k : k ≥ 2) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_k_ge_2_l1721_172133


namespace NUMINAMATH_GPT_inscribed_square_proof_l1721_172196

theorem inscribed_square_proof :
  (∃ (r : ℝ), 2 * π * r = 72 * π ∧ r = 36) ∧ 
  (∃ (s : ℝ), (2 * (36:ℝ))^2 = 2 * s ^ 2 ∧ s = 36 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_square_proof_l1721_172196


namespace NUMINAMATH_GPT_total_amount_in_wallet_l1721_172157

theorem total_amount_in_wallet
  (num_10_bills : ℕ)
  (num_20_bills : ℕ)
  (num_5_bills : ℕ)
  (amount_10_bills : ℕ)
  (num_20_bills_eq : num_20_bills = 4)
  (amount_10_bills_eq : amount_10_bills = 50)
  (total_num_bills : ℕ)
  (total_num_bills_eq : total_num_bills = 13)
  (num_10_bills_eq : num_10_bills = amount_10_bills / 10)
  (total_amount : ℕ)
  (total_amount_eq : total_amount = amount_10_bills + num_20_bills * 20 + num_5_bills * 5)
  (num_bills_accounted : ℕ)
  (num_bills_accounted_eq : num_bills_accounted = num_10_bills + num_20_bills)
  (num_5_bills_eq : num_5_bills = total_num_bills - num_bills_accounted)
  : total_amount = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_in_wallet_l1721_172157


namespace NUMINAMATH_GPT_final_mark_is_correct_l1721_172162

def term_mark : ℝ := 80
def term_weight : ℝ := 0.70
def exam_mark : ℝ := 90
def exam_weight : ℝ := 0.30

theorem final_mark_is_correct :
  (term_mark * term_weight + exam_mark * exam_weight) = 83 :=
by
  sorry

end NUMINAMATH_GPT_final_mark_is_correct_l1721_172162


namespace NUMINAMATH_GPT_train_speed_in_kmph_l1721_172158

-- Definitions for the given problem conditions
def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 240
def time_to_cross_bridge : ℝ := 20.99832013438925

-- Main theorem statement
theorem train_speed_in_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60.0084 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l1721_172158


namespace NUMINAMATH_GPT_ab_value_l1721_172119

theorem ab_value 
  (a b : ℝ) 
  (hx : 2 = b + 1) 
  (hy : a = -3) : 
  a * b = -3 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l1721_172119


namespace NUMINAMATH_GPT_arctan_sum_pi_over_four_l1721_172103

theorem arctan_sum_pi_over_four (a b c : ℝ) (C : ℝ) (h : Real.sin C = c / (a + b + c)) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_arctan_sum_pi_over_four_l1721_172103


namespace NUMINAMATH_GPT_students_prefer_windows_l1721_172108

theorem students_prefer_windows (total_students students_prefer_mac equally_prefer_both no_preference : ℕ) 
  (h₁ : total_students = 210)
  (h₂ : students_prefer_mac = 60)
  (h₃ : equally_prefer_both = 20)
  (h₄ : no_preference = 90) :
  total_students - students_prefer_mac - equally_prefer_both - no_preference = 40 := 
  by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_students_prefer_windows_l1721_172108


namespace NUMINAMATH_GPT_find_number_l1721_172126

theorem find_number (n : ℕ) (h1 : n % 20 = 1) (h2 : n / 20 = 9) : n = 181 := 
by {
  -- proof not required
  sorry
}

end NUMINAMATH_GPT_find_number_l1721_172126


namespace NUMINAMATH_GPT_find_multiple_of_son_age_l1721_172174

variable (F S k : ℕ)

theorem find_multiple_of_son_age
  (h1 : F = k * S + 4)
  (h2 : F + 4 = 2 * (S + 4) + 20)
  (h3 : F = 44) :
  k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_son_age_l1721_172174


namespace NUMINAMATH_GPT_max_non_managers_l1721_172163

theorem max_non_managers (n_mngrs n_non_mngrs : ℕ) (hmngrs : n_mngrs = 8) 
                (h_ratio : (5 : ℚ) / 24 < (n_mngrs : ℚ) / n_non_mngrs) :
                n_non_mngrs ≤ 38 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_non_managers_l1721_172163


namespace NUMINAMATH_GPT_ball_bounce_height_lt_one_l1721_172145

theorem ball_bounce_height_lt_one :
  ∃ (k : ℕ), 15 * (1/3:ℝ)^k < 1 ∧ k = 3 := 
sorry

end NUMINAMATH_GPT_ball_bounce_height_lt_one_l1721_172145


namespace NUMINAMATH_GPT_percentage_increase_first_year_l1721_172137

-- Assume the original price of the painting is P and the percentage increase during the first year is X
variable {P : ℝ} (X : ℝ)

-- Condition: The price decreases by 15% during the second year
def condition_decrease (price : ℝ) : ℝ := price * 0.85

-- Condition: The price at the end of the 2-year period was 93.5% of the original price
axiom condition_end_price : ∀ (P : ℝ), (P + (X/100) * P) * 0.85 = 0.935 * P

-- Proof problem: What was the percentage increase during the first year?
theorem percentage_increase_first_year : X = 10 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_first_year_l1721_172137


namespace NUMINAMATH_GPT_x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l1721_172180

theorem x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842
  (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end NUMINAMATH_GPT_x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l1721_172180


namespace NUMINAMATH_GPT_unit_digit_3_pow_2023_l1721_172102

def unit_digit_pattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0

theorem unit_digit_3_pow_2023 : unit_digit_pattern 2023 = 7 :=
by sorry

end NUMINAMATH_GPT_unit_digit_3_pow_2023_l1721_172102


namespace NUMINAMATH_GPT_turtles_remaining_on_log_l1721_172181

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end NUMINAMATH_GPT_turtles_remaining_on_log_l1721_172181


namespace NUMINAMATH_GPT_a_8_is_256_l1721_172172

variable (a : ℕ → ℕ)

axiom a_1 : a 1 = 2

axiom a_pq : ∀ p q : ℕ, a (p + q) = a p * a q

theorem a_8_is_256 : a 8 = 256 := by
  sorry

end NUMINAMATH_GPT_a_8_is_256_l1721_172172


namespace NUMINAMATH_GPT_pies_not_eaten_with_forks_l1721_172155

variables (apple_pe_forked peach_pe_forked cherry_pe_forked chocolate_pe_forked lemon_pe_forked : ℤ)
variables (total_pies types_of_pies : ℤ)

def pies_per_type (total_pies types_of_pies : ℤ) : ℤ :=
  total_pies / types_of_pies

def not_eaten_with_forks (percentage_forked : ℤ) (pies : ℤ) : ℤ :=
  pies - (pies * percentage_forked) / 100

noncomputable def apple_not_forked  := not_eaten_with_forks apple_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def peach_not_forked  := not_eaten_with_forks peach_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def cherry_not_forked := not_eaten_with_forks cherry_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def chocolate_not_forked := not_eaten_with_forks chocolate_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def lemon_not_forked := not_eaten_with_forks lemon_pe_forked (pies_per_type total_pies types_of_pies)

theorem pies_not_eaten_with_forks :
  (apple_not_forked = 128) ∧
  (peach_not_forked = 112) ∧
  (cherry_not_forked = 84) ∧
  (chocolate_not_forked = 76) ∧
  (lemon_not_forked = 140) :=
by sorry

end NUMINAMATH_GPT_pies_not_eaten_with_forks_l1721_172155


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_l1721_172161

theorem inequality_holds_for_all_real (a : ℝ) : a + a^3 - a^4 - a^6 < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_l1721_172161


namespace NUMINAMATH_GPT_problem_proof_l1721_172107

theorem problem_proof :
  (3 ∣ 18) ∧
  (17 ∣ 187 ∧ ¬ (17 ∣ 52)) ∧
  ¬ ((24 ∣ 72) ∧ (24 ∣ 67)) ∧
  ¬ (13 ∣ 26 ∧ ¬ (13 ∣ 52)) ∧
  (8 ∣ 160) :=
by 
  sorry

end NUMINAMATH_GPT_problem_proof_l1721_172107


namespace NUMINAMATH_GPT_distance_point_to_line_l1721_172187

theorem distance_point_to_line : 
  let x0 := 1
  let y0 := 0
  let A := 1
  let B := -2
  let C := 1 
  let dist := (A * x0 + B * y0 + C : ℝ) / Real.sqrt (A^2 + B^2)
  abs dist = 2 * Real.sqrt 5 / 5 :=
by
  -- Using basic principles of Lean and Mathlib to state the equality proof
  sorry

end NUMINAMATH_GPT_distance_point_to_line_l1721_172187


namespace NUMINAMATH_GPT_vector_identity_l1721_172198

-- Definitions of the vectors
variables {V : Type*} [AddGroup V]

-- Conditions as Lean definitions
def cond1 (AB BO AO : V) : Prop := AB + BO = AO
def cond2 (AO OM AM : V) : Prop := AO + OM = AM
def cond3 (AM MB AB : V) : Prop := AM + MB = AB

-- The main statement to be proved
theorem vector_identity (AB MB BO BC OM AO AM AC : V) 
  (h1 : cond1 AB BO AO) 
  (h2 : cond2 AO OM AM) 
  (h3 : cond3 AM MB AB) 
  : (AB + MB) + (BO + BC) + OM = AC :=
sorry

end NUMINAMATH_GPT_vector_identity_l1721_172198


namespace NUMINAMATH_GPT_x4_plus_y4_l1721_172105
noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

theorem x4_plus_y4 :
  (x^2 + (1 / x^2) = 7) →
  (x * y = 1) →
  (x^4 + y^4 = 47) :=
by
  intros h1 h2
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_x4_plus_y4_l1721_172105


namespace NUMINAMATH_GPT_problem_solution_l1721_172168

theorem problem_solution (x : ℝ) :
  (⌊|x^2 - 1|⌋ = 10) ↔ (x ∈ Set.Ioc (-2 * Real.sqrt 3) (-Real.sqrt 11) ∪ Set.Ico (Real.sqrt 11) (2 * Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1721_172168


namespace NUMINAMATH_GPT_fraction_of_x_by_110_l1721_172191

theorem fraction_of_x_by_110 (x : ℝ) (f : ℝ) (h1 : 0.6 * x = f * x + 110) (h2 : x = 412.5) : f = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_of_x_by_110_l1721_172191


namespace NUMINAMATH_GPT_prob_axisymmetric_and_centrally_symmetric_l1721_172169

theorem prob_axisymmetric_and_centrally_symmetric : 
  let card1 := "Line segment"
  let card2 := "Equilateral triangle"
  let card3 := "Parallelogram"
  let card4 := "Isosceles trapezoid"
  let card5 := "Circle"
  let cards := [card1, card2, card3, card4, card5]
  let symmetric_cards := [card1, card5]
  (symmetric_cards.length / cards.length : ℚ) = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_prob_axisymmetric_and_centrally_symmetric_l1721_172169


namespace NUMINAMATH_GPT_digimon_pack_price_l1721_172199

-- Defining the given conditions as Lean variables
variables (total_spent baseball_cost : ℝ)
variables (packs_of_digimon : ℕ)

-- Setting given values from the problem
def keith_total_spent : total_spent = 23.86 := sorry
def baseball_deck_cost : baseball_cost = 6.06 := sorry
def number_of_digimon_packs : packs_of_digimon = 4 := sorry

-- Stating the main theorem/problem to prove
theorem digimon_pack_price 
  (h1 : total_spent = 23.86)
  (h2 : baseball_cost = 6.06)
  (h3 : packs_of_digimon = 4) : 
  ∃ (price_per_pack : ℝ), price_per_pack = 4.45 :=
sorry

end NUMINAMATH_GPT_digimon_pack_price_l1721_172199


namespace NUMINAMATH_GPT_find_a_l1721_172104

open Set

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (h : (A a ∩ B a) = {2, 5}) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1721_172104


namespace NUMINAMATH_GPT_gain_percentage_is_30_l1721_172116

-- Definitions based on the conditions
def selling_price : ℕ := 195
def gain : ℕ := 45
def cost_price : ℕ := selling_price - gain
def gain_percentage : ℕ := (gain * 100) / cost_price

-- The statement to prove the gain percentage
theorem gain_percentage_is_30 : gain_percentage = 30 := 
by 
  -- Allow usage of fictive sorry for incomplete proof
  sorry

end NUMINAMATH_GPT_gain_percentage_is_30_l1721_172116


namespace NUMINAMATH_GPT_smaller_angle_at_seven_oclock_l1721_172194

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_seven_oclock_l1721_172194


namespace NUMINAMATH_GPT_triangle_area_is_9sqrt2_l1721_172164

noncomputable def triangle_area_with_given_medians_and_angle (CM BN : ℝ) (angle_BKM : ℝ) : ℝ :=
  let centroid_division_ratio := (2.0 / 3.0)
  let BK := centroid_division_ratio * BN
  let MK := (1.0 / 3.0) * CM
  let area_BKM := (1.0 / 2.0) * BK * MK * Real.sin angle_BKM
  6.0 * area_BKM

theorem triangle_area_is_9sqrt2 :
  triangle_area_with_given_medians_and_angle 6 4.5 (Real.pi / 4) = 9 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_9sqrt2_l1721_172164


namespace NUMINAMATH_GPT_ages_correct_l1721_172154

variables (Rehana_age Phoebe_age Jacob_age Xander_age : ℕ)

theorem ages_correct
  (h1 : Rehana_age = 25)
  (h2 : Rehana_age + 5 = 3 * (Phoebe_age + 5))
  (h3 : Jacob_age = 3 * Phoebe_age / 5)
  (h4 : Xander_age = Rehana_age + Jacob_age - 4) : 
  Rehana_age = 25 ∧ Phoebe_age = 5 ∧ Jacob_age = 3 ∧ Xander_age = 24 :=
by
  sorry

end NUMINAMATH_GPT_ages_correct_l1721_172154


namespace NUMINAMATH_GPT_value_of_c_l1721_172179

-- Define a structure representing conditions of the problem
structure ProblemConditions where
  c : Real

-- Define the problem in terms of given conditions and required proof
theorem value_of_c (conditions : ProblemConditions) : conditions.c = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_value_of_c_l1721_172179


namespace NUMINAMATH_GPT_maria_walk_to_school_l1721_172141

variable (w s : ℝ)

theorem maria_walk_to_school (h1 : 25 * w + 13 * s = 38) (h2 : 11 * w + 20 * s = 31) : 
  51 = 51 := by
  sorry

end NUMINAMATH_GPT_maria_walk_to_school_l1721_172141


namespace NUMINAMATH_GPT_smallest_fraction_greater_than_4_over_5_l1721_172156

theorem smallest_fraction_greater_than_4_over_5 :
  ∃ (b : ℕ), 10 ≤ b ∧ b < 100 ∧ 77 * 5 > 4 * b ∧ Int.gcd 77 b = 1 ∧
  ∀ (a : ℕ), 10 ≤ a ∧ a < 77 → ¬ ∃ (b' : ℕ), 10 ≤ b' ∧ b' < 100 ∧ a * 5 > 4 * b' ∧ Int.gcd a b' = 1 := by
  sorry

end NUMINAMATH_GPT_smallest_fraction_greater_than_4_over_5_l1721_172156


namespace NUMINAMATH_GPT_eq_fraction_l1721_172143

def f(x : ℤ) : ℤ := 3 * x + 4
def g(x : ℤ) : ℤ := 2 * x - 1

theorem eq_fraction : (f (g (f 3))) / (g (f (g 3))) = 79 / 37 := by
  sorry

end NUMINAMATH_GPT_eq_fraction_l1721_172143


namespace NUMINAMATH_GPT_equality_of_a_b_c_l1721_172130

theorem equality_of_a_b_c
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (eqn : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_equality_of_a_b_c_l1721_172130


namespace NUMINAMATH_GPT_min_a2_plus_b2_l1721_172192

theorem min_a2_plus_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
sorry

end NUMINAMATH_GPT_min_a2_plus_b2_l1721_172192


namespace NUMINAMATH_GPT_baylor_final_amount_l1721_172177

def CDA := 4000
def FCP := (1 / 2) * CDA
def SCP := FCP + (2 / 5) * FCP
def TCP := 2 * (FCP + SCP)
def FDA := CDA + FCP + SCP + TCP

theorem baylor_final_amount : FDA = 18400 := by
  sorry

end NUMINAMATH_GPT_baylor_final_amount_l1721_172177


namespace NUMINAMATH_GPT_beadshop_profit_on_wednesday_l1721_172189

theorem beadshop_profit_on_wednesday (total_profit profit_on_monday profit_on_tuesday profit_on_wednesday : ℝ)
  (h1 : total_profit = 1200)
  (h2 : profit_on_monday = total_profit / 3)
  (h3 : profit_on_tuesday = total_profit / 4)
  (h4 : profit_on_wednesday = total_profit - profit_on_monday - profit_on_tuesday) :
  profit_on_wednesday = 500 := 
sorry

end NUMINAMATH_GPT_beadshop_profit_on_wednesday_l1721_172189


namespace NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l1721_172183

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {4, 5, 6, 7, 8, 9}
def B : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem union_of_A_and_B : A ∪ B = U := by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {4, 5, 6} := by
  sorry

theorem complement_of_intersection : U \ (A ∩ B) = {1, 2, 3, 7, 8, 9} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l1721_172183


namespace NUMINAMATH_GPT_height_of_parallelogram_l1721_172146

noncomputable def parallelogram_height (base area : ℝ) : ℝ :=
  area / base

theorem height_of_parallelogram :
  parallelogram_height 8 78.88 = 9.86 :=
by
  -- This is where the proof would go, but it's being omitted as per instructions.
  sorry

end NUMINAMATH_GPT_height_of_parallelogram_l1721_172146


namespace NUMINAMATH_GPT_david_started_with_15_samsung_phones_l1721_172128

-- Definitions
def SamsungPhonesAtEnd : ℕ := 10 -- S_e
def IPhonesAtEnd : ℕ := 5 -- I_e
def SamsungPhonesThrownOut : ℕ := 2 -- S_d
def IPhonesThrownOut : ℕ := 1 -- I_d
def TotalPhonesSold : ℕ := 4 -- C

-- Number of iPhones sold
def IPhonesSold : ℕ := IPhonesThrownOut

-- Assume: The remaining phones sold are Samsung phones
def SamsungPhonesSold : ℕ := TotalPhonesSold - IPhonesSold

-- Calculate the number of Samsung phones David started the day with
def SamsungPhonesAtStart : ℕ := SamsungPhonesAtEnd + SamsungPhonesThrownOut + SamsungPhonesSold

-- Statement
theorem david_started_with_15_samsung_phones : SamsungPhonesAtStart = 15 := by
  sorry

end NUMINAMATH_GPT_david_started_with_15_samsung_phones_l1721_172128


namespace NUMINAMATH_GPT_probability_yellow_face_l1721_172131

-- Define the total number of faces and the number of yellow faces on the die
def total_faces := 12
def yellow_faces := 4

-- Define the probability calculation
def probability_of_yellow := yellow_faces / total_faces

-- State the theorem
theorem probability_yellow_face : probability_of_yellow = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_yellow_face_l1721_172131


namespace NUMINAMATH_GPT_cookies_in_each_batch_l1721_172140

theorem cookies_in_each_batch (batches : ℕ) (people : ℕ) (consumption_per_person : ℕ) (cookies_per_dozen : ℕ) 
  (total_batches : batches = 4) 
  (total_people : people = 16) 
  (cookies_per_person : consumption_per_person = 6) 
  (dozen_size : cookies_per_dozen = 12) :
  (6 * 16) / 4 / 12 = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_cookies_in_each_batch_l1721_172140


namespace NUMINAMATH_GPT_number_of_students_l1721_172153

noncomputable def is_handshakes_correct (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 
  (1 / 2 : ℚ) * (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) = 1020

theorem number_of_students (m n : ℕ) (h : is_handshakes_correct m n) : m * n = 280 := sorry

end NUMINAMATH_GPT_number_of_students_l1721_172153


namespace NUMINAMATH_GPT_soda_amount_l1721_172159

theorem soda_amount (S : ℝ) (h1 : S / 2 + 2000 = (S - (S / 2 + 2000)) / 2 + 2000) : S = 12000 :=
by
  sorry

end NUMINAMATH_GPT_soda_amount_l1721_172159


namespace NUMINAMATH_GPT_min_buses_needed_l1721_172124

-- Given definitions from conditions
def students_per_bus : ℕ := 45
def total_students : ℕ := 495

-- The proposition to prove
theorem min_buses_needed : ∃ n : ℕ, 45 * n ≥ 495 ∧ (∀ m : ℕ, 45 * m ≥ 495 → n ≤ m) :=
by
  -- Preliminary calculations that lead to the solution
  let n := total_students / students_per_bus
  have h : total_students % students_per_bus = 0 := by sorry
  
  -- Conclude that the minimum n so that 45 * n ≥ 495 is indeed 11
  exact ⟨n, by sorry, by sorry⟩

end NUMINAMATH_GPT_min_buses_needed_l1721_172124


namespace NUMINAMATH_GPT_initial_quantity_of_milk_in_A_l1721_172123

theorem initial_quantity_of_milk_in_A (A : ℝ) 
  (h1: ∃ C B: ℝ, B = 0.375 * A ∧ C = 0.625 * A) 
  (h2: ∃ M: ℝ, M = 0.375 * A + 154 ∧ M = 0.625 * A - 154) 
  : A = 1232 :=
by
  -- you can use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_initial_quantity_of_milk_in_A_l1721_172123


namespace NUMINAMATH_GPT_three_digit_odd_sum_count_l1721_172138

def countOddSumDigits : Nat :=
  -- Count of three-digit numbers with an odd sum formed by (1, 2, 3, 4, 5)
  24

theorem three_digit_odd_sum_count :
  -- Guarantees that the count of three-digit numbers meeting the criteria is 24
  ∃ n : Nat, n = countOddSumDigits :=
by
  use 24
  sorry

end NUMINAMATH_GPT_three_digit_odd_sum_count_l1721_172138


namespace NUMINAMATH_GPT_surface_area_of_prism_l1721_172195

theorem surface_area_of_prism (l w h : ℕ)
  (h_internal_volume : l * w * h = 24)
  (h_external_volume : (l + 2) * (w + 2) * (h + 2) = 120) :
  2 * ((l + 2) * (w + 2) + (w + 2) * (h + 2) + (h + 2) * (l + 2)) = 148 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_prism_l1721_172195


namespace NUMINAMATH_GPT_complex_product_conjugate_l1721_172171

theorem complex_product_conjugate : (1 + Complex.I) * (1 - Complex.I) = 2 := 
by 
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_complex_product_conjugate_l1721_172171


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l1721_172110

theorem geometric_sequence_third_term :
  ∃ (a : ℕ) (r : ℝ), a = 5 ∧ a * r^3 = 500 ∧ a * r^2 = 5 * 100^(2/3) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l1721_172110


namespace NUMINAMATH_GPT_landscaping_charges_l1721_172114

theorem landscaping_charges
    (x : ℕ)
    (h : 63 * x + 9 * 11 + 10 * 9 = 567) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_landscaping_charges_l1721_172114


namespace NUMINAMATH_GPT_Mark_paid_total_cost_l1721_172176

def length_of_deck : ℝ := 30
def width_of_deck : ℝ := 40
def cost_per_sq_ft_without_sealant : ℝ := 3
def additional_cost_per_sq_ft_sealant : ℝ := 1

def area (length width : ℝ) : ℝ := length * width
def total_cost (area cost_without_sealant cost_sealant : ℝ) : ℝ := 
  area * cost_without_sealant + area * cost_sealant

theorem Mark_paid_total_cost :
  total_cost (area length_of_deck width_of_deck) cost_per_sq_ft_without_sealant additional_cost_per_sq_ft_sealant = 4800 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_Mark_paid_total_cost_l1721_172176


namespace NUMINAMATH_GPT_part_i_part_ii_l1721_172142

-- Define the variables and conditions
variable (a b : ℝ)
variable (h₁ : a > 0)
variable (h₂ : b > 0)
variable (h₃ : a + b = 1 / a + 1 / b)

-- Prove the first part: a + b ≥ 2
theorem part_i : a + b ≥ 2 := by
  sorry

-- Prove the second part: It is impossible for both a² + a < 2 and b² + b < 2 simultaneously
theorem part_ii : ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end NUMINAMATH_GPT_part_i_part_ii_l1721_172142


namespace NUMINAMATH_GPT_mark_paid_more_than_anne_by_three_dollars_l1721_172109

theorem mark_paid_more_than_anne_by_three_dollars :
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  mark_total - anne_total = 3 :=
by
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  sorry

end NUMINAMATH_GPT_mark_paid_more_than_anne_by_three_dollars_l1721_172109


namespace NUMINAMATH_GPT_christine_speed_l1721_172111

def distance : ℕ := 20
def time : ℕ := 5

theorem christine_speed :
  (distance / time) = 4 := 
sorry

end NUMINAMATH_GPT_christine_speed_l1721_172111
