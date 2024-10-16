import Mathlib

namespace NUMINAMATH_CALUDE_max_divisibility_of_product_l752_75216

theorem max_divisibility_of_product (w x y z : ℕ) :
  w % 2 = 1 → x % 2 = 1 → y % 2 = 1 → z % 2 = 1 →
  w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
  w > 0 → x > 0 → y > 0 → z > 0 →
  ∃ (k : ℕ), (w^2 + x^2) * (y^2 + z^2) = 4 * k ∧
  ∀ (m : ℕ), m > 4 → (w^2 + x^2) * (y^2 + z^2) % m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_divisibility_of_product_l752_75216


namespace NUMINAMATH_CALUDE_trig_functions_right_triangle_l752_75221

/-- Define trigonometric functions for a right-angled triangle --/
theorem trig_functions_right_triangle 
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (A : ℝ), 
    Real.sin A = a / c ∧ 
    Real.cos A = b / c ∧ 
    Real.tan A = a / b :=
sorry

end NUMINAMATH_CALUDE_trig_functions_right_triangle_l752_75221


namespace NUMINAMATH_CALUDE_rationalize_denominator_l752_75242

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -1 ∧ B = -3 ∧ C = 1 ∧ D = 2 ∧ E = 33 ∧ F = 17 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l752_75242


namespace NUMINAMATH_CALUDE_rental_duration_proof_l752_75230

/-- Calculates the number of rental days given the daily rate, weekly rate, and total payment -/
def rentalDays (dailyRate weeklyRate totalPayment : ℕ) : ℕ :=
  let fullWeeks := totalPayment / weeklyRate
  let remainingPayment := totalPayment % weeklyRate
  let additionalDays := remainingPayment / dailyRate
  fullWeeks * 7 + additionalDays

/-- Proves that given the specified rates and payment, the rental duration is 11 days -/
theorem rental_duration_proof :
  rentalDays 30 190 310 = 11 := by
  sorry

end NUMINAMATH_CALUDE_rental_duration_proof_l752_75230


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l752_75255

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 + a 4 + a 5 + a 6 + a 8 = 25 →
  a 2 + a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l752_75255


namespace NUMINAMATH_CALUDE_total_sums_attempted_l752_75271

theorem total_sums_attempted (correct : ℕ) (wrong : ℕ) : 
  correct = 25 →
  wrong = 2 * correct →
  correct + wrong = 75 := by
sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l752_75271


namespace NUMINAMATH_CALUDE_min_additional_cars_l752_75267

/-- The number of cars Danica currently has -/
def initial_cars : ℕ := 37

/-- The number of cars required in each row -/
def cars_per_row : ℕ := 8

/-- The function to calculate the minimum number of additional cars needed -/
def additional_cars_needed (current : ℕ) (row_size : ℕ) : ℕ :=
  (row_size - (current % row_size)) % row_size

/-- Theorem stating that the minimum number of additional cars needed is 3 -/
theorem min_additional_cars :
  additional_cars_needed initial_cars cars_per_row = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_cars_l752_75267


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l752_75290

/-- A line passing through (2,3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (2,3)
  passes_through : a * 2 + b * 3 + c = 0
  -- The line has equal intercepts on both axes
  equal_intercepts : a ≠ 0 ∧ b ≠ 0 ∧ (c / a = c / b ∨ c = 0)

/-- The equation of an equal intercept line is either x+y-5=0 or 3x-2y=0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) ∨ (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l752_75290


namespace NUMINAMATH_CALUDE_father_age_problem_l752_75252

/-- The age problem -/
theorem father_age_problem (sebastian_age sister_age father_age : ℕ) : 
  sebastian_age = 40 →
  sister_age = sebastian_age - 10 →
  (sebastian_age - 5) + (sister_age - 5) = (3 * (father_age - 5)) / 4 →
  father_age = 90 := by
  sorry

end NUMINAMATH_CALUDE_father_age_problem_l752_75252


namespace NUMINAMATH_CALUDE_min_value_of_quartic_l752_75232

theorem min_value_of_quartic (x : ℝ) : 
  let y := (x - 16) * (x - 14) * (x + 14) * (x + 16)
  ∀ z : ℝ, y ≥ -900 ∧ ∃ x : ℝ, y = -900 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quartic_l752_75232


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l752_75220

/-- Given two points A(a, 3) and A'(2, b) that are symmetric with respect to the x-axis,
    prove that (a + b)^2023 = -1 -/
theorem symmetric_points_sum_power (a b : ℝ) : 
  (∃ (A A' : ℝ × ℝ), A = (a, 3) ∧ A' = (2, b) ∧ 
    (A.1 = A'.1 ∧ A.2 = -A'.2)) → (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l752_75220


namespace NUMINAMATH_CALUDE_a_range_l752_75224

-- Define proposition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

-- Define proposition q
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the theorem
theorem a_range (a : ℝ) : 
  (a < 0) → 
  (∀ x, ¬(p x a) → ¬(q x)) → 
  (∃ x, ¬(p x a) ∧ (q x)) → 
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_a_range_l752_75224


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l752_75203

def anna_lap_time : ℕ := 4
def stephanie_lap_time : ℕ := 7
def james_lap_time : ℕ := 6

theorem earliest_meeting_time :
  let meeting_time := lcm (lcm anna_lap_time stephanie_lap_time) james_lap_time
  meeting_time = 84 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l752_75203


namespace NUMINAMATH_CALUDE_initial_pens_l752_75288

theorem initial_pens (P : ℕ) : 2 * (P + 22) - 19 = 65 ↔ P = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_pens_l752_75288


namespace NUMINAMATH_CALUDE_square_sum_value_l752_75247

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l752_75247


namespace NUMINAMATH_CALUDE_certain_number_proof_l752_75283

theorem certain_number_proof (x : ℕ) : x > 72 ∧ x ∣ (72 * 14) ∧ (∀ y : ℕ, y > 72 ∧ y ∣ (72 * 14) → x ≤ y) → x = 84 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l752_75283


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l752_75231

/-- S_n is the sum of the reciprocals of the non-zero digits of the integers from 1 to 10^n inclusive -/
def S (n : ℕ+) : ℚ :=
  sorry

/-- 63 is the smallest positive integer n for which S_n is an integer -/
theorem smallest_n_for_integer_S :
  ∀ k : ℕ+, k < 63 → ¬ (S k).isInt ∧ (S 63).isInt := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l752_75231


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l752_75260

theorem binomial_expansion_coefficients 
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b) ^ n = 243)
  (h2 : (1 + |a|) ^ n = 32) : 
  a = 1 ∧ b = 2 ∧ n = 5 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l752_75260


namespace NUMINAMATH_CALUDE_borrowed_with_interest_l752_75269

/-- The amount to be returned after borrowing with interest -/
def amount_to_return (borrowed : ℝ) (interest_rate : ℝ) : ℝ :=
  borrowed * (1 + interest_rate)

/-- Theorem: Borrowing $100 with 10% interest results in returning $110 -/
theorem borrowed_with_interest :
  amount_to_return 100 0.1 = 110 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_with_interest_l752_75269


namespace NUMINAMATH_CALUDE_same_terminal_side_l752_75239

/-- Proves that -2π/3 has the same terminal side as 240° --/
theorem same_terminal_side : ∃ (k : ℤ), -2 * π / 3 = 240 * π / 180 + 2 * k * π := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l752_75239


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l752_75248

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l752_75248


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_a_value_l752_75200

-- Define the vectors m and n
def m : ℝ × ℝ := (-2, 3)
def n (a : ℝ) : ℝ × ℝ := (a + 1, 3)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem parallel_vectors_imply_a_value :
  parallel m (n a) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_a_value_l752_75200


namespace NUMINAMATH_CALUDE_sum_of_roots_is_51_l752_75245

-- Define the function f
def f (x : ℝ) : ℝ := 16 * x + 3

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (x - 3) / 16

-- Theorem statement
theorem sum_of_roots_is_51 :
  ∃ (x₁ x₂ : ℝ), 
    (f_inv x₁ = f ((2 * x₁)⁻¹)) ∧
    (f_inv x₂ = f ((2 * x₂)⁻¹)) ∧
    (∀ x : ℝ, f_inv x = f ((2 * x)⁻¹) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 51 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_51_l752_75245


namespace NUMINAMATH_CALUDE_complex_equation_solution_l752_75251

theorem complex_equation_solution (n : ℝ) : (2 / (1 - Complex.I) = 1 + n * Complex.I) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l752_75251


namespace NUMINAMATH_CALUDE_price_restoration_l752_75238

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) 
  (h1 : reduced_price = 0.9 * original_price) : 
  (11 + 1/9) / 100 * reduced_price = original_price := by
sorry

end NUMINAMATH_CALUDE_price_restoration_l752_75238


namespace NUMINAMATH_CALUDE_tan_three_pi_halves_minus_alpha_l752_75218

theorem tan_three_pi_halves_minus_alpha (α : Real) 
  (h : Real.cos (Real.pi - α) = -3/5) : 
  Real.tan (3/2 * Real.pi - α) = 3/4 ∨ Real.tan (3/2 * Real.pi - α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_pi_halves_minus_alpha_l752_75218


namespace NUMINAMATH_CALUDE_extremum_and_monotonicity_l752_75277

/-- The function f(x) defined as e^x - ln(x + m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (x + m)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / (x + m)

theorem extremum_and_monotonicity (m : ℝ) :
  (f_deriv m 0 = 0) →
  (m = 1) ∧
  (∀ x > 0, f_deriv m x > 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f_deriv m x < 0) := by
  sorry

#check extremum_and_monotonicity

end NUMINAMATH_CALUDE_extremum_and_monotonicity_l752_75277


namespace NUMINAMATH_CALUDE_cubic_equation_root_implies_m_range_l752_75289

theorem cubic_equation_root_implies_m_range :
  ∀ m : ℝ, 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^3 - 3*x + m = 0) →
  m ∈ Set.Icc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_implies_m_range_l752_75289


namespace NUMINAMATH_CALUDE_no_solution_l752_75273

/-- The function f(t) = t^3 + t -/
def f (t : ℚ) : ℚ := t^3 + t

/-- Iterative application of f, n times -/
def f_iter (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

/-- There do not exist rational numbers x and y and positive integers m and n
    such that xy = 3 and f^m(x) = f^n(y) -/
theorem no_solution :
  ¬ ∃ (x y : ℚ) (m n : ℕ+), x * y = 3 ∧ f_iter m x = f_iter n y := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l752_75273


namespace NUMINAMATH_CALUDE_product_count_in_range_l752_75264

theorem product_count_in_range (total_sample : ℕ) 
  (freq_96_100 : ℚ) (freq_98_104 : ℚ) (count_less_100 : ℕ) :
  freq_96_100 = 3/10 →
  freq_98_104 = 3/8 →
  count_less_100 = 36 →
  total_sample = count_less_100 / freq_96_100 →
  (freq_98_104 * total_sample : ℚ) = 60 :=
by sorry

end NUMINAMATH_CALUDE_product_count_in_range_l752_75264


namespace NUMINAMATH_CALUDE_ball_count_equals_hex_sum_ball_count_2010_l752_75208

/-- Converts a natural number to its hexadecimal representation -/
def toHex (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball-placing process for n steps -/
def ballCount (n : ℕ) : ℕ :=
  sorry

theorem ball_count_equals_hex_sum (n : ℕ) : 
  ballCount n = sumDigits (toHex n) := by
  sorry

theorem ball_count_2010 : 
  ballCount 2010 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_equals_hex_sum_ball_count_2010_l752_75208


namespace NUMINAMATH_CALUDE_sum_xy_values_l752_75266

theorem sum_xy_values (x y : ℕ) (hx : x < 20) (hy : y < 20) (hsum : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_values_l752_75266


namespace NUMINAMATH_CALUDE_f_negative_a_value_l752_75274

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem f_negative_a_value (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_value_l752_75274


namespace NUMINAMATH_CALUDE_problem_statement_l752_75265

theorem problem_statement (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁^2 + 5*x₂^2 = 10)
  (h2 : x₂*y₁ - x₁*y₂ = 5)
  (h3 : x₁*y₁ + 5*x₂*y₂ = Real.sqrt 105) :
  y₁^2 + 5*y₂^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l752_75265


namespace NUMINAMATH_CALUDE_mixed_grains_calculation_l752_75284

/-- Calculates the amount of mixed grains in a batch of rice -/
theorem mixed_grains_calculation (total_rice : ℝ) (sample_size : ℝ) (mixed_in_sample : ℝ) :
  total_rice * (mixed_in_sample / sample_size) = 150 :=
by
  -- Assuming total_rice = 1500, sample_size = 200, and mixed_in_sample = 20
  have h1 : total_rice = 1500 := by sorry
  have h2 : sample_size = 200 := by sorry
  have h3 : mixed_in_sample = 20 := by sorry
  
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mixed_grains_calculation_l752_75284


namespace NUMINAMATH_CALUDE_system_solution_l752_75285

theorem system_solution (x y : ℝ) : 
  (x = 5 ∧ y = -1) → (2 * x + 3 * y = 7 ∧ x = -2 * y + 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l752_75285


namespace NUMINAMATH_CALUDE_no_sequence_satisfying_inequality_l752_75295

theorem no_sequence_satisfying_inequality :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ), 
    (0 < α ∧ α < 1) ∧
    (∀ n, 0 < a n) ∧
    (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_satisfying_inequality_l752_75295


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l752_75211

/-- Two triangles are similar -/
structure SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop where
  similar : True  -- We don't need to define the full similarity conditions for this problem

/-- The length of a line segment between two points -/
def segmentLength (A B : ℝ × ℝ) : ℝ := sorry

theorem similar_triangles_segment_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z) 
  (h_PQ : segmentLength P Q = 8)
  (h_QR : segmentLength Q R = 16)
  (h_YZ : segmentLength Y Z = 24) :
  segmentLength X Y = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l752_75211


namespace NUMINAMATH_CALUDE_max_value_of_expression_l752_75287

theorem max_value_of_expression (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a ∈ ({2, 3, 6} : Set ℕ) →
  b ∈ ({2, 3, 6} : Set ℕ) →
  c ∈ ({2, 3, 6} : Set ℕ) →
  (a : ℚ) / ((b : ℚ) / (c : ℚ)) ≤ 9 →
  (∃ a' b' c' : ℕ, 
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    a' ∈ ({2, 3, 6} : Set ℕ) ∧
    b' ∈ ({2, 3, 6} : Set ℕ) ∧
    c' ∈ ({2, 3, 6} : Set ℕ) ∧
    (a' : ℚ) / ((b' : ℚ) / (c' : ℚ)) = 9) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l752_75287


namespace NUMINAMATH_CALUDE_triangle_with_120_degree_angle_l752_75292

theorem triangle_with_120_degree_angle : 
  ∃ (a b c : ℕ), 
    a = 3 ∧ b = 6 ∧ c = 7 ∧ 
    (a^2 + b^2 - c^2 : ℝ) / (2 * a * b : ℝ) = - (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_120_degree_angle_l752_75292


namespace NUMINAMATH_CALUDE_a_5_equals_one_l752_75279

/-- A geometric sequence with positive terms and common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem a_5_equals_one
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_a_5_equals_one_l752_75279


namespace NUMINAMATH_CALUDE_complete_square_constant_l752_75298

theorem complete_square_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_constant_l752_75298


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l752_75202

/-- Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k, h = -3/2 -/
theorem quadratic_vertex_form_h (x : ℝ) : 
  ∃ (a k : ℝ), 3*x^2 + 9*x + 20 = a*(x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l752_75202


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l752_75276

/-- The value of j that makes the line 4x + 7y + j = 0 tangent to the parabola y^2 = 32x -/
def tangent_j : ℝ := 98

/-- The line equation -/
def line (x y j : ℝ) : Prop := 4 * x + 7 * y + j = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- Theorem stating that tangent_j is the unique value making the line tangent to the parabola -/
theorem tangent_line_to_parabola :
  ∃! j : ℝ, ∀ x y : ℝ, line x y j ∧ parabola x y → 
    (∃! p : ℝ × ℝ, line p.1 p.2 j ∧ parabola p.1 p.2) ∧ j = tangent_j :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l752_75276


namespace NUMINAMATH_CALUDE_parabola_sum_l752_75257

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_sum (p : Parabola) : 
  p.y_coord 4 = 2 ∧  -- vertex (4,2)
  p.y_coord 1 = -4 ∧  -- point (1,-4)
  p.y_coord 7 = 0 ∧  -- point (7,0)
  (∀ x : ℝ, p.y_coord (8 - x) = p.y_coord x) →  -- vertical axis of symmetry at x = 4
  p.a + p.b + p.c = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l752_75257


namespace NUMINAMATH_CALUDE_exponential_inequality_l752_75272

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2^a + 2*a = 2^b + 3*b → a > b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l752_75272


namespace NUMINAMATH_CALUDE_grey_pairs_coincide_l752_75244

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  green : Nat
  yellow : Nat
  grey : Nat

/-- Represents the number of coinciding pairs of each type when folded -/
structure CoincidingPairs where
  green_green : Nat
  yellow_yellow : Nat
  green_grey : Nat
  grey_grey : Nat

/-- The main theorem statement -/
theorem grey_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.green = 4 ∧ 
  counts.yellow = 6 ∧ 
  counts.grey = 10 ∧
  pairs.green_green = 3 ∧
  pairs.yellow_yellow = 4 ∧
  pairs.green_grey = 3 →
  pairs.grey_grey = 5 := by
  sorry

end NUMINAMATH_CALUDE_grey_pairs_coincide_l752_75244


namespace NUMINAMATH_CALUDE_patients_arrangement_exists_l752_75296

theorem patients_arrangement_exists :
  ∃ (cow she_wolf beetle worm : ℝ),
    0 ≤ cow ∧ cow < she_wolf ∧ she_wolf < beetle ∧ beetle < worm ∧ worm = 6 ∧
    she_wolf - cow = 1 ∧
    beetle - cow = 2 ∧
    (she_wolf - cow) + (beetle - she_wolf) + (worm - beetle) = 7 ∧
    (beetle - cow) + (worm - beetle) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_patients_arrangement_exists_l752_75296


namespace NUMINAMATH_CALUDE_expression_value_l752_75246

theorem expression_value : 3^(2+4+6) - (3^2 + 3^4 + 3^6) + (3^2 * 3^4 * 3^6) = 1062242 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l752_75246


namespace NUMINAMATH_CALUDE_total_shares_sold_l752_75299

/-- Proves that the total number of shares sold is 300 given the specified conditions -/
theorem total_shares_sold (microtron_price dynaco_price avg_price : ℚ) (dynaco_shares : ℕ) : 
  microtron_price = 36 →
  dynaco_price = 44 →
  avg_price = 40 →
  dynaco_shares = 150 →
  ∃ (microtron_shares : ℕ), 
    (microtron_price * microtron_shares + dynaco_price * dynaco_shares) / (microtron_shares + dynaco_shares) = avg_price ∧
    microtron_shares + dynaco_shares = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_shares_sold_l752_75299


namespace NUMINAMATH_CALUDE_absolute_difference_l752_75233

theorem absolute_difference (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l752_75233


namespace NUMINAMATH_CALUDE_circle_area_ratio_l752_75226

/-- Given two circles X and Y, where an arc of 60° on X has the same length as an arc of 20° on Y,
    the ratio of the area of X to the area of Y is 9. -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) 
  (h : X * (60 / 360) = Y * (20 / 360)) : 
  (X^2 * Real.pi) / (Y^2 * Real.pi) = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l752_75226


namespace NUMINAMATH_CALUDE_M_not_subset_P_l752_75205

-- Define the sets M and P
def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem M_not_subset_P : ¬(M ⊆ P) := by sorry

end NUMINAMATH_CALUDE_M_not_subset_P_l752_75205


namespace NUMINAMATH_CALUDE_numerator_greater_than_denominator_l752_75228

theorem numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ 4 * x - 3 > 9 - 2 * x → 2 < x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_numerator_greater_than_denominator_l752_75228


namespace NUMINAMATH_CALUDE_system_solution_l752_75254

theorem system_solution : 
  ∃! (x y : ℚ), (3 * x - 4 * y = 10) ∧ (9 * x + 8 * y = 14) ∧ (x = 34/15) ∧ (y = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l752_75254


namespace NUMINAMATH_CALUDE_parallelogram_angle_l752_75249

/-- 
Given a parallelogram with the following properties:
- One angle exceeds the other by 40 degrees
- An inscribed circle touches the extended line of the smaller angle
- This touch point forms a triangle exterior to the parallelogram
- The angle at this point is 60 degrees less than double the smaller angle

Prove that the smaller angle of the parallelogram is 70 degrees.
-/
theorem parallelogram_angle (x : ℝ) : 
  x > 0 ∧ 
  x + 40 > x ∧
  x + (x + 40) = 180 ∧
  2 * x - 60 > 0 → 
  x = 70 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_l752_75249


namespace NUMINAMATH_CALUDE_cylindrical_to_cartesian_l752_75243

/-- Given a point M with cylindrical coordinates (√2, 5π/4, √2), 
    its Cartesian coordinates are (-1, -1, √2) -/
theorem cylindrical_to_cartesian :
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 4
  let z : ℝ := Real.sqrt 2
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  x = -1 ∧ y = -1 ∧ z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cylindrical_to_cartesian_l752_75243


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l752_75204

theorem negative_sixty_four_to_seven_thirds : (-64 : ℝ) ^ (7/3) = -65536 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l752_75204


namespace NUMINAMATH_CALUDE_second_square_area_l752_75253

/-- Represents an isosceles right triangle with inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the first inscribed square -/
  s : ℝ
  /-- Area of the first inscribed square is 484 cm² -/
  first_square_area : s^2 = 484
  /-- Side length of the second inscribed square -/
  x : ℝ
  /-- Relationship between side lengths of the triangle and the second square -/
  triangle_side_relation : 3 * x = 2 * s

/-- The area of the second inscribed square is 1936/9 cm² -/
theorem second_square_area (triangle : IsoscelesRightTriangleWithSquares) :
  triangle.x^2 = 1936 / 9 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_l752_75253


namespace NUMINAMATH_CALUDE_digits_left_of_264_divisible_by_4_l752_75281

theorem digits_left_of_264_divisible_by_4 : 
  (∀ n : ℕ, n < 10 → (n * 1000 + 264) % 4 = 0) ∧ 
  (∃ (S : Finset ℕ), S.card = 10 ∧ ∀ n ∈ S, n < 10 ∧ (n * 1000 + 264) % 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_digits_left_of_264_divisible_by_4_l752_75281


namespace NUMINAMATH_CALUDE_cells_intersected_303x202_l752_75209

/-- Represents a grid rectangle with diagonals --/
structure GridRectangle where
  length : Nat
  width : Nat

/-- Calculates the number of cells intersected by diagonals in a grid rectangle --/
def cells_intersected_by_diagonals (grid : GridRectangle) : Nat :=
  let small_rectangles := (grid.length / 3) * (grid.width / 2)
  let cells_per_diagonal := small_rectangles * 4
  let total_cells := cells_per_diagonal * 2
  total_cells - 2

/-- Theorem stating that in a 303 x 202 grid rectangle, 806 cells are intersected by diagonals --/
theorem cells_intersected_303x202 :
  cells_intersected_by_diagonals ⟨303, 202⟩ = 806 := by
  sorry

end NUMINAMATH_CALUDE_cells_intersected_303x202_l752_75209


namespace NUMINAMATH_CALUDE_line_slope_l752_75241

/-- Given a line l and two points on it, prove that the slope of the line is -3 -/
theorem line_slope (l : Set (ℝ × ℝ)) (a b : ℝ) 
  (h1 : (a, b) ∈ l) 
  (h2 : (a + 1, b - 3) ∈ l) : 
  ∃ (m : ℝ), m = -3 ∧ ∀ (x y : ℝ), (x, y) ∈ l → y = m * (x - a) + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_l752_75241


namespace NUMINAMATH_CALUDE_runner_picture_probability_l752_75282

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the state of the race at a given time -/
def RaceState :=
  ℕ  -- time in seconds

/-- Represents the camera setup -/
structure Camera where
  coverageFraction : ℚ
  centerPosition : ℚ  -- fraction of track from start line

/-- Calculate the position of a runner at a given time -/
def runnerPosition (r : Runner) (t : ℕ) : ℚ :=
  sorry

/-- Check if a runner is in the camera's view -/
def isInPicture (r : Runner) (t : ℕ) (c : Camera) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem runner_picture_probability :
  let alice : Runner := ⟨"Alice", 120, true⟩
  let ben : Runner := ⟨"Ben", 100, false⟩
  let camera : Camera := ⟨1/3, 0⟩
  let raceTime : ℕ := 900
  let totalOverlapTime : ℚ := 40/3
  (totalOverlapTime / 60 : ℚ) = 1333/6000 := by
  sorry

end NUMINAMATH_CALUDE_runner_picture_probability_l752_75282


namespace NUMINAMATH_CALUDE_chess_board_pawn_arrangements_l752_75286

theorem chess_board_pawn_arrangements (n : ℕ) (h : n = 5) : 
  (Finset.range n).card.factorial = 120 := by sorry

end NUMINAMATH_CALUDE_chess_board_pawn_arrangements_l752_75286


namespace NUMINAMATH_CALUDE_circle_trajectory_intersection_l752_75214

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the trajectory curve E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line that intersects E and C₁
def line (x y b : ℝ) : Prop := y = (1/2)*x + b

-- Define the condition for complementary angles
def complementary_angles (B D : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := B
  let (x₂, y₂) := D
  (y₁ / (x₁ - 1)) + (y₂ / (x₂ - 1)) = 0

theorem circle_trajectory_intersection :
  ∀ (A B C D : ℝ × ℝ) (b : ℝ),
  E B.1 B.2 → E D.1 D.2 →
  C₁ A.1 A.2 → C₁ C.1 C.2 →
  line A.1 A.2 b → line B.1 B.2 b → line C.1 C.2 b → line D.1 D.2 b →
  complementary_angles B D →
  ∃ (AB CD : ℝ), AB + CD = (36 * Real.sqrt 5) / 5 :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_intersection_l752_75214


namespace NUMINAMATH_CALUDE_prime_equation_solution_l752_75240

theorem prime_equation_solution :
  ∀ p : ℕ, Prime p →
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔
    (p = 2 ∨ p = 3 ∨ p = 7) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l752_75240


namespace NUMINAMATH_CALUDE_locus_of_R_l752_75262

/-- The ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Point A -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B -/
def B : ℝ × ℝ := (1, 0)

/-- Point C -/
def C : ℝ × ℝ := (3, 0)

/-- The locus equation -/
def locus_equation (x y : ℝ) : Prop := 45 * x^2 - 108 * y^2 = 20

/-- The theorem stating the locus of point R -/
theorem locus_of_R :
  ∀ (R : ℝ × ℝ),
  (∃ (P Q : ℝ × ℝ),
    ellipse P.1 P.2 ∧
    ellipse Q.1 Q.2 ∧
    Q.1 < P.1 ∧
    (∃ (t : ℝ), t > 0 ∧ Q.1 = C.1 - t * (C.1 - P.1) ∧ Q.2 = C.2 - t * (C.2 - P.2)) ∧
    (∃ (s : ℝ), A.1 + s * (P.1 - A.1) = R.1 ∧ A.2 + s * (P.2 - A.2) = R.2) ∧
    (∃ (u : ℝ), B.1 + u * (Q.1 - B.1) = R.1 ∧ B.2 + u * (Q.2 - B.2) = R.2)) →
  locus_equation R.1 R.2 ∧ 2/3 < R.1 ∧ R.1 < 4/3 :=
sorry

end NUMINAMATH_CALUDE_locus_of_R_l752_75262


namespace NUMINAMATH_CALUDE_largest_triangular_square_under_50_l752_75261

def isTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem largest_triangular_square_under_50 :
  ∃ n : ℕ, n ≤ 50 ∧ isTriangular n ∧ isPerfectSquare n ∧
  ∀ m : ℕ, m ≤ 50 → isTriangular m → isPerfectSquare m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_triangular_square_under_50_l752_75261


namespace NUMINAMATH_CALUDE_theo_homework_assignments_l752_75237

/-- Calculates the number of assignments for a given set number -/
def assignmentsPerSet (setNumber : Nat) : Nat :=
  2^(setNumber - 1)

/-- Calculates the total assignments for a given number of sets -/
def totalAssignments (sets : Nat) : Nat :=
  (List.range sets).map (fun i => 6 * assignmentsPerSet (i + 1)) |>.sum

theorem theo_homework_assignments :
  totalAssignments 5 = 186 := by
  sorry

#eval totalAssignments 5

end NUMINAMATH_CALUDE_theo_homework_assignments_l752_75237


namespace NUMINAMATH_CALUDE_total_bouncy_balls_l752_75234

def red_packs : ℕ := 4
def yellow_packs : ℕ := 8
def green_packs : ℕ := 4
def balls_per_pack : ℕ := 10

theorem total_bouncy_balls :
  (red_packs + yellow_packs + green_packs) * balls_per_pack = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_bouncy_balls_l752_75234


namespace NUMINAMATH_CALUDE_problem_1_l752_75259

theorem problem_1 : (1/3)⁻¹ + Real.sqrt 18 - 4 * Real.cos (π/4) = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l752_75259


namespace NUMINAMATH_CALUDE_marias_green_towels_l752_75236

theorem marias_green_towels :
  ∀ (green_towels : ℕ),
  (green_towels + 21 : ℕ) - 34 = 22 →
  green_towels = 35 :=
by sorry

end NUMINAMATH_CALUDE_marias_green_towels_l752_75236


namespace NUMINAMATH_CALUDE_area_decreasing_map_l752_75280

open Set
open MeasureTheory

-- Define a distance function for ℝ²
noncomputable def distance (x y : ℝ × ℝ) : ℝ := Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

-- Define the properties of function f
def is_distance_decreasing (f : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ x y : ℝ × ℝ, distance x y ≥ distance (f x) (f y)

-- Theorem statement
theorem area_decreasing_map
  (f : ℝ × ℝ → ℝ × ℝ)
  (h_inj : Function.Injective f)
  (h_surj : Function.Surjective f)
  (h_dist : is_distance_decreasing f)
  (A : Set (ℝ × ℝ)) :
  MeasureTheory.volume A ≥ MeasureTheory.volume (f '' A) :=
sorry

end NUMINAMATH_CALUDE_area_decreasing_map_l752_75280


namespace NUMINAMATH_CALUDE_smallest_integer_with_eight_factors_l752_75219

theorem smallest_integer_with_eight_factors : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (factors : Finset ℕ), factors.card = 8 ∧ 
    (∀ m ∈ factors, m > 0 ∧ n % m = 0) ∧
    (∀ m : ℕ, m > 0 → n % m = 0 → m ∈ factors)) ∧
  (∀ k : ℕ, k > 0 → k < n →
    ¬(∃ (factors : Finset ℕ), factors.card = 8 ∧ 
      (∀ m ∈ factors, m > 0 ∧ k % m = 0) ∧
      (∀ m : ℕ, m > 0 → k % m = 0 → m ∈ factors))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_eight_factors_l752_75219


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_l752_75235

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333333 / 1000000 = 1 / (3 * 1000000) := by
  sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_l752_75235


namespace NUMINAMATH_CALUDE_number_line_steps_l752_75263

/-- Given a number line with equally spaced markings, where 35 is reached in 7 steps from 0,
    prove that after 5 steps, the number reached is 25. -/
theorem number_line_steps (total_distance : ℝ) (total_steps : ℕ) (target_steps : ℕ) : 
  total_distance = 35 ∧ total_steps = 7 ∧ target_steps = 5 → 
  (total_distance / total_steps) * target_steps = 25 := by
  sorry

#check number_line_steps

end NUMINAMATH_CALUDE_number_line_steps_l752_75263


namespace NUMINAMATH_CALUDE_min_z_shapes_cover_min_z_shapes_necessary_l752_75293

/-- Represents a cell on the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a Z shape on the table -/
structure ZShape where
  base : Cell
  rotation : Nat

/-- The size of the table -/
def tableSize : Nat := 8

/-- Checks if a cell is within the table bounds -/
def isValidCell (c : Cell) : Prop :=
  c.row ≥ 1 ∧ c.row ≤ tableSize ∧ c.col ≥ 1 ∧ c.col ≤ tableSize

/-- Checks if a Z shape covers a given cell -/
def coversCell (z : ZShape) (c : Cell) : Prop :=
  match z.rotation % 4 with
  | 0 => c = z.base ∨ c = ⟨z.base.row, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 2⟩
  | 1 => c = z.base ∨ c = ⟨z.base.row + 1, z.base.col⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 2⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 2⟩
  | 2 => c = z.base ∨ c = ⟨z.base.row, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col + 2⟩
  | _ => c = z.base ∨ c = ⟨z.base.row - 1, z.base.col⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col - 1⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col - 2⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col - 2⟩

/-- The main theorem stating that 12 Z shapes are sufficient to cover the table -/
theorem min_z_shapes_cover (shapes : List ZShape) : 
  (∀ c : Cell, isValidCell c → ∃ z ∈ shapes, coversCell z c) → 
  shapes.length ≥ 12 :=
sorry

/-- The main theorem stating that 12 Z shapes are necessary to cover the table -/
theorem min_z_shapes_necessary : 
  ∃ shapes : List ZShape, shapes.length = 12 ∧ 
  ∀ c : Cell, isValidCell c → ∃ z ∈ shapes, coversCell z c :=
sorry

end NUMINAMATH_CALUDE_min_z_shapes_cover_min_z_shapes_necessary_l752_75293


namespace NUMINAMATH_CALUDE_sum_exponents_of_sqrt_largest_perfect_square_12_factorial_l752_75278

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largest_perfect_square_divisor (n : ℕ) : ℕ := sorry

def prime_factor_exponents (n : ℕ) : List ℕ := sorry

theorem sum_exponents_of_sqrt_largest_perfect_square_12_factorial : 
  (prime_factor_exponents (largest_perfect_square_divisor (factorial 12)).sqrt).sum = 8 := by sorry

end NUMINAMATH_CALUDE_sum_exponents_of_sqrt_largest_perfect_square_12_factorial_l752_75278


namespace NUMINAMATH_CALUDE_base_10_423_equals_base_5_3143_l752_75225

def base_10_to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_10_423_equals_base_5_3143 :
  base_10_to_base_5 423 = [3, 1, 4, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_10_423_equals_base_5_3143_l752_75225


namespace NUMINAMATH_CALUDE_courtyard_length_l752_75206

/-- Calculates the length of a rectangular courtyard given its width, number of bricks, and brick dimensions --/
theorem courtyard_length 
  (width : ℝ) 
  (num_bricks : ℕ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (h1 : width = 16) 
  (h2 : num_bricks = 20000) 
  (h3 : brick_length = 0.2) 
  (h4 : brick_width = 0.1) : 
  (num_bricks : ℝ) * brick_length * brick_width / width = 25 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l752_75206


namespace NUMINAMATH_CALUDE_meaningful_range_l752_75223

def is_meaningful (x : ℝ) : Prop :=
  x ≥ 3 ∧ x ≠ 4

theorem meaningful_range (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (x - 3) / (x - 4)) ↔ is_meaningful x :=
sorry

end NUMINAMATH_CALUDE_meaningful_range_l752_75223


namespace NUMINAMATH_CALUDE_middle_bead_value_is_92_l752_75294

/-- Represents a string of beads with specific properties -/
structure BeadString where
  total_beads : Nat
  middle_bead_index : Nat
  price_diff_left : Nat
  price_diff_right : Nat
  total_value : Nat

/-- Calculates the value of the middle bead in a BeadString -/
def middle_bead_value (bs : BeadString) : Nat :=
  sorry

/-- Theorem stating the value of the middle bead in the specific BeadString -/
theorem middle_bead_value_is_92 :
  let bs : BeadString := {
    total_beads := 31,
    middle_bead_index := 15,
    price_diff_left := 3,
    price_diff_right := 4,
    total_value := 2012
  }
  middle_bead_value bs = 92 := by sorry

end NUMINAMATH_CALUDE_middle_bead_value_is_92_l752_75294


namespace NUMINAMATH_CALUDE_simplify_expression_l752_75217

theorem simplify_expression : (27 ^ (1/6) - Real.sqrt (6 + 3/4)) ^ 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l752_75217


namespace NUMINAMATH_CALUDE_equation_solution_characterization_equation_unique_solution_characterization_l752_75291

/-- The equation has a solution -/
def has_solution (a b : ℝ) : Prop :=
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3)

/-- The solution is unique -/
def unique_solution (a b : ℝ) : Prop :=
  a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3

/-- The equation in question -/
def equation (x a b : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2

theorem equation_solution_characterization (a b : ℝ) :
  (∃ x, equation x a b) ↔ has_solution a b :=
sorry

theorem equation_unique_solution_characterization (a b : ℝ) :
  (∃! x, equation x a b) ↔ unique_solution a b :=
sorry

end NUMINAMATH_CALUDE_equation_solution_characterization_equation_unique_solution_characterization_l752_75291


namespace NUMINAMATH_CALUDE_charlie_and_diana_qualify_l752_75210

structure Person :=
  (name : String)
  (qualifies : Prop)

def Alice : Person := ⟨"Alice", sorry⟩
def Bob : Person := ⟨"Bob", sorry⟩
def Charlie : Person := ⟨"Charlie", sorry⟩
def Diana : Person := ⟨"Diana", sorry⟩

def Statements : Prop :=
  (Alice.qualifies → Bob.qualifies) ∧
  (Bob.qualifies → Charlie.qualifies) ∧
  (Charlie.qualifies → (Diana.qualifies ∧ ¬Alice.qualifies))

def ExactlyTwoQualify : Prop :=
  ∃! (p1 p2 : Person), p1 ≠ p2 ∧ p1.qualifies ∧ p2.qualifies ∧
    ∀ (p : Person), p.qualifies → (p = p1 ∨ p = p2)

theorem charlie_and_diana_qualify :
  Statements ∧ ExactlyTwoQualify →
  Charlie.qualifies ∧ Diana.qualifies ∧ ¬Alice.qualifies ∧ ¬Bob.qualifies :=
by sorry

end NUMINAMATH_CALUDE_charlie_and_diana_qualify_l752_75210


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l752_75270

theorem complex_magnitude_problem (z : ℂ) : 
  z + Complex.I = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l752_75270


namespace NUMINAMATH_CALUDE_corey_sunday_vs_saturday_l752_75213

/-- Corey's goal for finding golf balls -/
def goal : ℕ := 48

/-- Number of golf balls Corey found on Saturday -/
def saturdayBalls : ℕ := 16

/-- Number of golf balls Corey still needs to reach his goal -/
def stillNeeded : ℕ := 14

/-- Number of golf balls Corey found on Sunday -/
def sundayBalls : ℕ := goal - saturdayBalls - stillNeeded

theorem corey_sunday_vs_saturday : sundayBalls - saturdayBalls = 2 := by
  sorry

end NUMINAMATH_CALUDE_corey_sunday_vs_saturday_l752_75213


namespace NUMINAMATH_CALUDE_sisters_sandcastle_height_l752_75201

theorem sisters_sandcastle_height 
  (janet_height : Float) 
  (height_difference : Float) 
  (h1 : janet_height = 3.6666666666666665) 
  (h2 : height_difference = 1.3333333333333333) : 
  janet_height - height_difference = 2.333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_sisters_sandcastle_height_l752_75201


namespace NUMINAMATH_CALUDE_population_reaches_max_capacity_l752_75250

/-- The number of acres on the island of Nisos -/
def island_acres : ℕ := 36000

/-- The number of acres required per person -/
def acres_per_person : ℕ := 2

/-- The initial population in 2040 -/
def initial_population : ℕ := 300

/-- The number of years it takes for the population to quadruple -/
def quadruple_period : ℕ := 30

/-- The maximum capacity of the island -/
def max_capacity : ℕ := island_acres / acres_per_person

/-- The population after n periods -/
def population (n : ℕ) : ℕ := initial_population * 4^n

/-- The number of years from 2040 until the population reaches or exceeds the maximum capacity -/
theorem population_reaches_max_capacity : 
  ∃ n : ℕ, n * quadruple_period = 90 ∧ population n ≥ max_capacity ∧ population (n - 1) < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_max_capacity_l752_75250


namespace NUMINAMATH_CALUDE_equation_is_linear_l752_75227

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 1 = 20 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- Theorem: The equation 2x - 1 = 20 is a linear equation -/
theorem equation_is_linear : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l752_75227


namespace NUMINAMATH_CALUDE_sam_car_repair_cost_l752_75268

/-- Represents Sam's yard work earnings and expenses --/
structure SamFinances where
  march_to_august_earnings : ℕ
  march_to_august_hours : ℕ
  sept_to_feb_hours : ℕ
  console_cost : ℕ
  additional_hours_needed : ℕ

/-- Calculates the amount Sam spent on fixing his car --/
def car_repair_cost (s : SamFinances) : ℕ :=
  let hourly_rate := s.march_to_august_earnings / s.march_to_august_hours
  let total_earnings := s.march_to_august_earnings + 
                        hourly_rate * s.sept_to_feb_hours + 
                        hourly_rate * s.additional_hours_needed
  total_earnings - s.console_cost

/-- Theorem stating that Sam spent $340 on fixing his car --/
theorem sam_car_repair_cost :
  ∃ (s : SamFinances),
    s.march_to_august_earnings = 460 ∧
    s.march_to_august_hours = 23 ∧
    s.sept_to_feb_hours = 8 ∧
    s.console_cost = 600 ∧
    s.additional_hours_needed = 16 ∧
    car_repair_cost s = 340 :=
  sorry

end NUMINAMATH_CALUDE_sam_car_repair_cost_l752_75268


namespace NUMINAMATH_CALUDE_circle_equation_l752_75297

/-- A circle C with center (a, 0) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ

/-- The line l: y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The point P(0, 1) -/
def P : ℝ × ℝ := (0, 1)

/-- The circle C is tangent to the line l at point P -/
def is_tangent (C : Circle) : Prop :=
  C.r^2 = (C.a - P.1)^2 + (0 - P.2)^2 ∧
  (0 - P.2) / (C.a - P.1) = -1 / 2

theorem circle_equation (C : Circle) 
  (h1 : is_tangent C) : 
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 5 ↔ (x - C.a)^2 + y^2 = C.r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l752_75297


namespace NUMINAMATH_CALUDE_shopping_expense_l752_75229

theorem shopping_expense (total_spent shirt_cost : ℕ) (h1 : total_spent = 300) (h2 : shirt_cost = 97) :
  ∃ (shoe_cost : ℕ), 
    shoe_cost > 2 * shirt_cost ∧ 
    shirt_cost + shoe_cost = total_spent ∧ 
    shoe_cost - 2 * shirt_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_shopping_expense_l752_75229


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l752_75258

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (1 - x / (x + 1)) / ((x^2 - 2*x + 1) / (x^2 - 1)) = 1 / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l752_75258


namespace NUMINAMATH_CALUDE_unique_solution_system_l752_75215

theorem unique_solution_system : 
  ∃! (x y z : ℕ+), 
    (2 * x * z = y^2) ∧ 
    (x + z = 1987) ∧
    (x = 1458) ∧ 
    (y = 1242) ∧ 
    (z = 529) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l752_75215


namespace NUMINAMATH_CALUDE_drainpipe_emptying_time_l752_75212

theorem drainpipe_emptying_time 
  (fill_rate1 : ℝ) (fill_rate2 : ℝ) (drain_rate : ℝ) (combined_fill_time : ℝ) :
  fill_rate1 = 1 / 5 →
  fill_rate2 = 1 / 4 →
  drain_rate > 0 →
  combined_fill_time = 2.5 →
  fill_rate1 + fill_rate2 - drain_rate = 1 / combined_fill_time →
  1 / drain_rate = 20 := by
sorry

end NUMINAMATH_CALUDE_drainpipe_emptying_time_l752_75212


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_expansion_main_proof_l752_75256

theorem fraction_to_decimal : (7 : ℚ) / 200 = (35 : ℚ) / 1000 := by sorry

theorem decimal_expansion : (35 : ℚ) / 1000 = 0.035 := by sorry

theorem main_proof : (7 : ℚ) / 200 = 0.035 := by
  rw [fraction_to_decimal]
  exact decimal_expansion

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_expansion_main_proof_l752_75256


namespace NUMINAMATH_CALUDE_unique_abc_solution_l752_75222

/-- Represents a base-7 number with two digits -/
def Base7TwoDigit (a b : Nat) : Nat := 7 * a + b

/-- Represents a base-7 number with one digit -/
def Base7OneDigit (c : Nat) : Nat := c

/-- Represents a base-7 number with two digits, where the first digit is 'c' and the second is 0 -/
def Base7TwoDigitWithZero (c : Nat) : Nat := 7 * c

theorem unique_abc_solution :
  ∀ (A B C : Nat),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigitWithZero C →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
    A = 3 ∧ B = 2 ∧ C = 5 := by
  sorry

#check unique_abc_solution

end NUMINAMATH_CALUDE_unique_abc_solution_l752_75222


namespace NUMINAMATH_CALUDE_distance_between_points_l752_75207

/-- The distance between two points on a plane is the square root of the sum of squares of differences in their coordinates. -/
theorem distance_between_points (A B : ℝ × ℝ) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 26 :=
by
  -- Given points A(2,1) and B(-3,2)
  have hA : A = (2, 1) := by sorry
  have hB : B = (-3, 2) := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l752_75207


namespace NUMINAMATH_CALUDE_neil_cookies_fraction_l752_75275

theorem neil_cookies_fraction (total : ℕ) (remaining : ℕ) (h1 : total = 20) (h2 : remaining = 12) :
  (total - remaining : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_neil_cookies_fraction_l752_75275
