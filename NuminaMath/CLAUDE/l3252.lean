import Mathlib

namespace gym_signup_fee_l3252_325241

theorem gym_signup_fee 
  (cheap_monthly : ℕ)
  (expensive_monthly : ℕ)
  (expensive_signup : ℕ)
  (total_cost : ℕ)
  (h1 : cheap_monthly = 10)
  (h2 : expensive_monthly = 3 * cheap_monthly)
  (h3 : expensive_signup = 4 * expensive_monthly)
  (h4 : total_cost = 650)
  (h5 : total_cost = 12 * cheap_monthly + 12 * expensive_monthly + expensive_signup + cheap_signup) :
  cheap_signup = 50 := by
  sorry

end gym_signup_fee_l3252_325241


namespace rationalize_denominator_l3252_325289

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end rationalize_denominator_l3252_325289


namespace sandwiches_per_person_l3252_325248

theorem sandwiches_per_person (people : ℝ) (total_sandwiches : ℕ) 
  (h1 : people = 219) 
  (h2 : total_sandwiches = 657) : 
  (total_sandwiches : ℝ) / people = 3 := by
  sorry

end sandwiches_per_person_l3252_325248


namespace total_balloons_sam_dan_l3252_325209

-- Define the initial quantities
def sam_initial : ℝ := 46.5
def fred_receive : ℝ := 10.2
def gaby_receive : ℝ := 3.3
def dan_balloons : ℝ := 16.4

-- Define Sam's remaining balloons after distribution
def sam_remaining : ℝ := sam_initial - fred_receive - gaby_receive

-- Theorem statement
theorem total_balloons_sam_dan : 
  sam_remaining + dan_balloons = 49.4 := by sorry

end total_balloons_sam_dan_l3252_325209


namespace k_values_l3252_325229

theorem k_values (p q r s k : ℂ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0)
  (h_eq1 : p * k^3 + q * k^2 + r * k + s = 0)
  (h_eq2 : q * k^3 + r * k^2 + s * k + p = 0)
  (h_pqrs : p * q = r * s) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end k_values_l3252_325229


namespace necessary_not_sufficient_l3252_325271

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  (m > 1) ∧ (m < 3) ∧ (m ≠ 2)

/-- The condition given in the problem -/
def given_condition (m : ℝ) : Prop :=
  (m > 1) ∧ (m < 3)

/-- Theorem stating that the given condition is necessary but not sufficient -/
theorem necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → given_condition m) ∧
  ¬(∀ m : ℝ, given_condition m → is_ellipse m) :=
sorry

end necessary_not_sufficient_l3252_325271


namespace parabola_directrix_l3252_325214

/-- Given a parabola defined by x = -1/4 * y^2, its directrix is the line x = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (x = -(1/4) * y^2) → (∃ (k : ℝ), k = 1 ∧ k = x) := by sorry

end parabola_directrix_l3252_325214


namespace product_plus_one_is_square_l3252_325297

theorem product_plus_one_is_square (n : ℕ) : 
  ∃ m : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m ^ 2 := by
  sorry

#check product_plus_one_is_square 7321

end product_plus_one_is_square_l3252_325297


namespace compare_logarithmic_expressions_l3252_325270

open Real

theorem compare_logarithmic_expressions :
  let e := exp 1
  1/e > log (3^(1/3)) ∧ 
  log (3^(1/3)) > log π / π ∧ 
  log π / π > sqrt 15 * log 15 / 30 :=
by
  sorry

end compare_logarithmic_expressions_l3252_325270


namespace sum_of_fractions_l3252_325277

theorem sum_of_fractions : 
  (3 / 15 : ℚ) + (6 / 15 : ℚ) + (9 / 15 : ℚ) + (12 / 15 : ℚ) + (1 : ℚ) + 
  (18 / 15 : ℚ) + (21 / 15 : ℚ) + (24 / 15 : ℚ) + (27 / 15 : ℚ) + (5 : ℚ) = 14 := by
  sorry

end sum_of_fractions_l3252_325277


namespace modular_inverse_three_mod_seventeen_l3252_325268

theorem modular_inverse_three_mod_seventeen :
  ∃! x : ℕ, x ≤ 16 ∧ (3 * x) % 17 = 1 :=
by
  sorry

end modular_inverse_three_mod_seventeen_l3252_325268


namespace range_of_h_l3252_325284

noncomputable def h (t : ℝ) : ℝ := (t^2 + 5/4 * t) / (t^2 + 2)

theorem range_of_h :
  Set.range h = Set.Icc 0 (128/103) := by sorry

end range_of_h_l3252_325284


namespace minimum_value_problem_minimum_value_achievable_l3252_325286

theorem minimum_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 :=
by sorry

theorem minimum_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) = 4 :=
by sorry

end minimum_value_problem_minimum_value_achievable_l3252_325286


namespace min_value_expression_l3252_325290

theorem min_value_expression (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (m : ℝ), (∀ c d : ℝ, c ≠ 0 → d ≠ 0 → c^2 + d^2 + 4/c^2 + 2*d/c ≥ m) ∧
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ c^2 + d^2 + 4/c^2 + 2*d/c = m) ∧
  m = 2 * Real.sqrt 3 :=
sorry

end min_value_expression_l3252_325290


namespace equation_solution_l3252_325251

theorem equation_solution : ∃ x : ℕ, 5 + x = 10 + 20 ∧ x = 25 := by
  sorry

end equation_solution_l3252_325251


namespace diophantine_equation_solution_l3252_325218

theorem diophantine_equation_solution (t : ℤ) : 
  ∃ (x y : ℤ), x^4 + 2*x^3 + 8*x - 35*y + 9 = 0 ∧
  (x = 35*t + 6 ∨ x = 35*t - 4 ∨ x = 35*t - 9 ∨ 
   x = 35*t - 16 ∨ x = 35*t - 1 ∨ x = 35*t - 11) ∧
  y = (x^4 + 2*x^3 + 8*x + 9) / 35 :=
by sorry

end diophantine_equation_solution_l3252_325218


namespace two_digit_product_digits_l3252_325203

theorem two_digit_product_digits :
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (100 ≤ a * b ∧ a * b ≤ 9999) :=
by sorry

end two_digit_product_digits_l3252_325203


namespace weight_of_K2Cr2O7_l3252_325236

/-- The atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of chromium in g/mol -/
def atomic_weight_Cr : ℝ := 52.00

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of potassium atoms in K2Cr2O7 -/
def K_count : ℕ := 2

/-- The number of chromium atoms in K2Cr2O7 -/
def Cr_count : ℕ := 2

/-- The number of oxygen atoms in K2Cr2O7 -/
def O_count : ℕ := 7

/-- The number of moles of K2Cr2O7 -/
def moles : ℝ := 4

/-- The molecular weight of K2Cr2O7 in g/mol -/
def molecular_weight_K2Cr2O7 : ℝ := 
  K_count * atomic_weight_K + Cr_count * atomic_weight_Cr + O_count * atomic_weight_O

/-- The total weight of 4 moles of K2Cr2O7 in grams -/
theorem weight_of_K2Cr2O7 : moles * molecular_weight_K2Cr2O7 = 1176.80 := by
  sorry

end weight_of_K2Cr2O7_l3252_325236


namespace geometric_sequence_11th_term_l3252_325245

/-- Given a geometric sequence where the 5th term is 2 and the 8th term is 16,
    prove that the 11th term is 128. -/
theorem geometric_sequence_11th_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n m, a (n + 1) / a n = a (m + 1) / a m)  -- Geometric sequence condition
  (h_5th : a 5 = 2)  -- 5th term is 2
  (h_8th : a 8 = 16)  -- 8th term is 16
  : a 11 = 128 := by
  sorry

end geometric_sequence_11th_term_l3252_325245


namespace absolute_value_equation_solution_l3252_325280

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  use 4
  sorry

end absolute_value_equation_solution_l3252_325280


namespace confetti_area_sum_l3252_325259

/-- The sum of the areas of two square-shaped pieces of confetti, one with side length 11 cm and the other with side length 5 cm, is equal to 146 cm². -/
theorem confetti_area_sum : 
  let red_side : ℝ := 11
  let blue_side : ℝ := 5
  red_side ^ 2 + blue_side ^ 2 = 146 :=
by sorry

end confetti_area_sum_l3252_325259


namespace area_ratio_of_squares_l3252_325260

/-- Given four square regions with perimeters p₁, p₂, p₃, and p₄, 
    this theorem proves that the ratio of the area of the second square 
    to the area of the fourth square is 9/16 when p₁ = 16, p₂ = 36, p₃ = p₄ = 48. -/
theorem area_ratio_of_squares (p₁ p₂ p₃ p₄ : ℝ) 
    (h₁ : p₁ = 16) (h₂ : p₂ = 36) (h₃ : p₃ = 48) (h₄ : p₄ = 48) :
    (p₂ / 4)^2 / (p₄ / 4)^2 = 9 / 16 := by
  sorry

end area_ratio_of_squares_l3252_325260


namespace problem_solution_l3252_325223

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6}

theorem problem_solution (A B : Set ℤ) 
  (h1 : U = A ∪ B) 
  (h2 : A ∩ (U \ B) = {1, 3, 5}) : 
  B = {0, 2, 4, 6} := by
  sorry

end problem_solution_l3252_325223


namespace problem_statement_l3252_325212

theorem problem_statement :
  (∀ a : ℝ, a < (3/2) → 2*a + 4/(2*a - 3) + 3 ≤ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3*y = 2*x*y → x + 3*y ≥ 6) := by
sorry

end problem_statement_l3252_325212


namespace weight_problem_l3252_325225

theorem weight_problem (a b c d e f g h : ℝ) 
  (h1 : (a + b + c + f) / 4 = 80)
  (h2 : (a + b + c + d + e + f) / 6 = 82)
  (h3 : g = d + 5)
  (h4 : h = e - 4)
  (h5 : (c + d + e + f + g + h) / 6 = 83) :
  a + b = 167 := by
sorry

end weight_problem_l3252_325225


namespace plane_equation_correct_l3252_325267

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space represented by the equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- The point where the perpendicular meets the plane -/
def perpendicularPoint : Point3D := ⟨10, -2, 5⟩

/-- The plane in question -/
def targetPlane : Plane := ⟨10, -2, 5, -129⟩

/-- Vector from origin to perpendicularPoint -/
def normalVector : Point3D := perpendicularPoint

theorem plane_equation_correct :
  (∀ (p : Point3D), p.liesOn targetPlane ↔ 
    (p.x - perpendicularPoint.x) * normalVector.x + 
    (p.y - perpendicularPoint.y) * normalVector.y + 
    (p.z - perpendicularPoint.z) * normalVector.z = 0) ∧
  perpendicularPoint.liesOn targetPlane :=
sorry

end plane_equation_correct_l3252_325267


namespace bowling_ball_weight_l3252_325293

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (10 * bowling_ball_weight = 4 * canoe_weight) →
    (canoe_weight = 35) →
    (bowling_ball_weight = 14) :=
by
  sorry

end bowling_ball_weight_l3252_325293


namespace fraction_evaluation_l3252_325296

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by sorry

end fraction_evaluation_l3252_325296


namespace salary_solution_l3252_325205

def salary_problem (s : ℕ) : Prop :=
  s - s / 3 - s / 4 - s / 5 = 1760

theorem salary_solution : ∃ (s : ℕ), salary_problem s ∧ s = 812 := by
  sorry

end salary_solution_l3252_325205


namespace jack_socks_purchase_l3252_325249

/-- The number of pairs of socks Jack needs to buy -/
def num_socks : ℕ := 2

/-- The cost of each pair of socks in dollars -/
def sock_cost : ℚ := 9.5

/-- The cost of the shoes in dollars -/
def shoe_cost : ℕ := 92

/-- The total amount Jack needs in dollars -/
def total_amount : ℕ := 111

theorem jack_socks_purchase :
  sock_cost * num_socks + shoe_cost = total_amount :=
by sorry

end jack_socks_purchase_l3252_325249


namespace right_triangle_with_hypotenuse_41_l3252_325224

theorem right_triangle_with_hypotenuse_41 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →
  c = 41 →
  a < b →
  a = 9 :=
by
  sorry

end right_triangle_with_hypotenuse_41_l3252_325224


namespace cylinder_volume_change_l3252_325213

/-- Given a cylinder with original volume of 15 cubic feet, proves that tripling its radius and halving its height results in a new volume of 67.5 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) (h3 : π * r^2 * h = 15) :
  π * (3*r)^2 * (h/2) = 67.5 := by
  sorry

end cylinder_volume_change_l3252_325213


namespace largest_prime_divisor_of_1202102_base5_l3252_325202

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds the largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_1202102_base5 :
  largestPrimeDivisor (base5ToBase10 1202102) = 307 := by sorry

end largest_prime_divisor_of_1202102_base5_l3252_325202


namespace nine_integer_chords_l3252_325298

/-- A circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- The number of integer-length chords through P -/
def integerChordCount (c : CircleWithPoint) : ℕ :=
  sorry

theorem nine_integer_chords 
  (c : CircleWithPoint) 
  (h1 : c.radius = 20) 
  (h2 : c.distanceFromCenter = 12) : 
  integerChordCount c = 9 := by sorry

end nine_integer_chords_l3252_325298


namespace polynomial_equality_l3252_325285

/-- Given that 7x^5 + 4x^3 - 3x + p(x) = 2x^4 - 10x^3 + 5x - 2,
    prove that p(x) = -7x^5 + 2x^4 - 6x^3 + 2x - 2 -/
theorem polynomial_equality (x : ℝ) (p : ℝ → ℝ) 
  (h : ∀ x, 7 * x^5 + 4 * x^3 - 3 * x + p x = 2 * x^4 - 10 * x^3 + 5 * x - 2) : 
  p = fun x ↦ -7 * x^5 + 2 * x^4 - 6 * x^3 + 2 * x - 2 := by
  sorry

end polynomial_equality_l3252_325285


namespace second_number_proof_l3252_325211

theorem second_number_proof (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 := by
  sorry

end second_number_proof_l3252_325211


namespace base_sum_theorem_l3252_325282

theorem base_sum_theorem : ∃! (R_A R_B : ℕ), 
  (R_A > 0 ∧ R_B > 0) ∧
  ((4 * R_A + 5) * (R_B^2 - 1) = (3 * R_B + 6) * (R_A^2 - 1)) ∧
  ((5 * R_A + 4) * (R_B^2 - 1) = (6 * R_B + 3) * (R_A^2 - 1)) ∧
  (R_A + R_B = 19) := by
sorry

end base_sum_theorem_l3252_325282


namespace correct_operation_l3252_325252

theorem correct_operation (a : ℝ) : 3 * a - 2 * a = a := by sorry

end correct_operation_l3252_325252


namespace min_complex_sum_value_l3252_325226

theorem min_complex_sum_value (p q r : ℕ+) (ζ : ℂ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_ζ_fourth : ζ^4 = 1)
  (h_ζ_neq_one : ζ ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 7 ∧ 
    ∀ (p' q' r' : ℕ+) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ζ + r' * ζ^3) ≥ m :=
sorry

end min_complex_sum_value_l3252_325226


namespace tangent_lines_count_l3252_325216

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ f a (-a) + f a (3*a) = 2 * f a a

-- Define a tangent line from the origin
def tangent_from_origin (a : ℝ) (x₀ : ℝ) : Prop :=
  ∃ y₀ : ℝ, f a x₀ = y₀ ∧ y₀ = (3 * a * x₀^2 - 6 * x₀) * x₀

-- Main theorem
theorem tangent_lines_count (a : ℝ) (ha : a ≠ 0) :
  arithmetic_sequence a →
  ∃! (count : ℕ), count = 2 ∧ 
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
      tangent_from_origin a x₁ ∧ 
      tangent_from_origin a x₂ ∧
      ∀ (x : ℝ), tangent_from_origin a x → (x = x₁ ∨ x = x₂) :=
sorry

end tangent_lines_count_l3252_325216


namespace teacher_count_l3252_325273

theorem teacher_count (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) :
  total = 3000 →
  sample_size = 150 →
  students_in_sample = 140 →
  (total - (total * students_in_sample / sample_size) : ℕ) = 200 := by
  sorry

end teacher_count_l3252_325273


namespace triangle_theorem_l3252_325238

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The given condition relating sides and angles -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) / t.c = (Real.cos (t.A + t.C)) / (Real.cos t.C)

theorem triangle_theorem (t : Triangle) (h : triangle_condition t) :
  t.C = 2 * π / 3 ∧ 1 < (t.a + t.b) / t.c ∧ (t.a + t.b) / t.c ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end triangle_theorem_l3252_325238


namespace geometric_sequence_third_term_l3252_325219

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a1 : a 1 = -2)
  (h_a5 : a 5 = -8) :
  a 3 = -4 :=
sorry

end geometric_sequence_third_term_l3252_325219


namespace fraction_difference_zero_l3252_325207

theorem fraction_difference_zero (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  1 / ((x - 2) * (x - 3)) - 2 / ((x - 1) * (x - 3)) + 1 / ((x - 1) * (x - 2)) = 0 := by
  sorry

end fraction_difference_zero_l3252_325207


namespace necessary_but_not_sufficient_l3252_325279

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def distance (p q : Point) : ℝ := sorry

def is_hyperbola (trajectory : Set Point) (F₁ F₂ : Point) : Prop := sorry

def is_constant (f : Point → ℝ) : Prop := sorry

-- State the theorem
theorem necessary_but_not_sufficient 
  (M : Point) (F₁ F₂ : Point) (trajectory : Set Point) :
  (∀ M ∈ trajectory, is_hyperbola trajectory F₁ F₂) →
    is_constant (λ M => |distance M F₁ - distance M F₂|) ∧
  ∃ trajectory' : Set Point, 
    is_constant (λ M => |distance M F₁ - distance M F₂|) ∧
    ¬(is_hyperbola trajectory' F₁ F₂) :=
by
  sorry

end necessary_but_not_sufficient_l3252_325279


namespace smallest_k_is_three_l3252_325227

/-- A coloring of positive integers with k colors -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- Property (i): For all positive integers m, n of the same color, f(m+n) = f(m) + f(n) -/
def PropertyOne (f : ℕ+ → ℕ+) (c : Coloring k) :=
  ∀ m n : ℕ+, c m = c n → f (m + n) = f m + f n

/-- Property (ii): There exist positive integers m, n such that f(m+n) ≠ f(m) + f(n) -/
def PropertyTwo (f : ℕ+ → ℕ+) :=
  ∃ m n : ℕ+, f (m + n) ≠ f m + f n

/-- The main theorem statement -/
theorem smallest_k_is_three :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : ℕ+ → ℕ+, PropertyOne f c ∧ PropertyTwo f) ∧
  (∀ k : ℕ+, k < 3 → ¬∃ c : Coloring k, ∃ f : ℕ+ → ℕ+, PropertyOne f c ∧ PropertyTwo f) :=
sorry

end smallest_k_is_three_l3252_325227


namespace log_inequality_solution_l3252_325222

theorem log_inequality_solution (x : ℝ) : 
  (4 * (Real.log (Real.cos (2 * x)) / Real.log 16) + 
   2 * (Real.log (Real.sin x) / Real.log 4) + 
   Real.log (Real.cos x) / Real.log 2 + 3 < 0) ↔ 
  (0 < x ∧ x < Real.pi / 24) ∨ (5 * Real.pi / 24 < x ∧ x < Real.pi / 4) :=
sorry

end log_inequality_solution_l3252_325222


namespace geometric_series_common_ratio_l3252_325299

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 20/21
  let a₃ : ℚ := 100/63
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = (4/7) * (5/3)^(n-1)) →
  r = 5/3 := by
sorry

end geometric_series_common_ratio_l3252_325299


namespace shopkeeper_profit_margin_l3252_325278

theorem shopkeeper_profit_margin
  (C : ℝ) -- Current cost
  (S : ℝ) -- Selling price
  (y : ℝ) -- Original profit margin percentage
  (h1 : S = C * (1 + 0.01 * y)) -- Current profit margin equation
  (h2 : S = 0.9 * C * (1 + 0.01 * (y + 15))) -- New profit margin equation
  : y = 35 := by
  sorry

end shopkeeper_profit_margin_l3252_325278


namespace day2_to_day1_rain_ratio_l3252_325294

/-- Represents the rainfall data and conditions for a 4-day storm --/
structure RainfallData where
  capacity : ℝ  -- Capacity in inches
  drainRate : ℝ  -- Drain rate in inches per day
  day1Rain : ℝ  -- Rainfall on day 1 in inches
  day3Increase : ℝ  -- Percentage increase of day 3 rain compared to day 2
  day4Rain : ℝ  -- Rainfall on day 4 in inches

/-- Theorem stating the ratio of day 2 rain to day 1 rain --/
theorem day2_to_day1_rain_ratio (data : RainfallData) 
  (h1 : data.capacity = 72) -- 6 feet = 72 inches
  (h2 : data.drainRate = 3)
  (h3 : data.day1Rain = 10)
  (h4 : data.day3Increase = 1.5) -- 50% more
  (h5 : data.day4Rain = 21) :
  ∃ (x : ℝ), x = 2 ∧ 
    data.day1Rain + x * data.day1Rain + data.day3Increase * x * data.day1Rain + data.day4Rain = 
    data.capacity + 3 * data.drainRate := by
  sorry

#check day2_to_day1_rain_ratio

end day2_to_day1_rain_ratio_l3252_325294


namespace initial_blocks_l3252_325200

theorem initial_blocks (initial final added : ℕ) : 
  final = initial + added → 
  final = 65 → 
  added = 30 → 
  initial = 35 := by sorry

end initial_blocks_l3252_325200


namespace square_is_rectangle_and_rhombus_l3252_325266

-- Define a quadrilateral
structure Quadrilateral :=
  (sides : Fin 4 → ℝ)
  (angles : Fin 4 → ℝ)

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, q.angles i = 90 ∧ q.sides i = q.sides ((i + 2) % 4)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  is_rectangle q ∧ is_rhombus q

-- Theorem statement
theorem square_is_rectangle_and_rhombus (q : Quadrilateral) :
  is_square q → is_rectangle q ∧ is_rhombus q :=
sorry

end square_is_rectangle_and_rhombus_l3252_325266


namespace ten_thousand_one_hundred_one_l3252_325217

theorem ten_thousand_one_hundred_one (n : ℕ) : n = 10101 → n = 10000 + 100 + 1 := by
  sorry

end ten_thousand_one_hundred_one_l3252_325217


namespace acidic_solution_concentration_l3252_325230

/-- Proves that the initial volume of a 40% acidic solution is 27 liters
    when it becomes 60% acidic after removing 9 liters of water. -/
theorem acidic_solution_concentration (initial_volume : ℝ) : 
  initial_volume > 0 →
  (0.4 * initial_volume) / (initial_volume - 9) = 0.6 →
  initial_volume = 27 := by
  sorry

end acidic_solution_concentration_l3252_325230


namespace power_division_equality_l3252_325253

theorem power_division_equality (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := by
  sorry

end power_division_equality_l3252_325253


namespace cos_45_cos_15_plus_sin_45_sin_15_l3252_325276

theorem cos_45_cos_15_plus_sin_45_sin_15 :
  Real.cos (45 * π / 180) * Real.cos (15 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end cos_45_cos_15_plus_sin_45_sin_15_l3252_325276


namespace nikolai_silver_decrease_l3252_325206

/-- Represents the number of each type of coin --/
structure CoinCount where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Represents a transaction at the exchange point --/
inductive Transaction
  | Type1 : Transaction  -- 2 gold for 3 silver and 1 copper
  | Type2 : Transaction  -- 5 silver for 3 gold and 1 copper

/-- Applies a single transaction to a CoinCount --/
def applyTransaction (t : Transaction) (c : CoinCount) : CoinCount :=
  match t with
  | Transaction.Type1 => CoinCount.mk (c.gold - 2) (c.silver + 3) (c.copper + 1)
  | Transaction.Type2 => CoinCount.mk (c.gold + 3) (c.silver - 5) (c.copper + 1)

/-- Applies a list of transactions to an initial CoinCount --/
def applyTransactions (ts : List Transaction) (initial : CoinCount) : CoinCount :=
  ts.foldl (fun acc t => applyTransaction t acc) initial

theorem nikolai_silver_decrease (initialSilver : ℕ) :
  ∃ (ts : List Transaction),
    let final := applyTransactions ts (CoinCount.mk 0 initialSilver 0)
    final.gold = 0 ∧
    final.copper = 50 ∧
    initialSilver - final.silver = 10 := by
  sorry

end nikolai_silver_decrease_l3252_325206


namespace largest_prime_factor_of_5985_l3252_325210

theorem largest_prime_factor_of_5985 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 5985 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 5985 → q ≤ p :=
sorry

end largest_prime_factor_of_5985_l3252_325210


namespace xy_value_l3252_325254

theorem xy_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 5) : 2 * x * y / 3 = 2 := by
  sorry

end xy_value_l3252_325254


namespace rectangular_field_distance_l3252_325243

/-- The distance run around a rectangular field -/
def distance_run (length width : ℕ) (laps : ℕ) : ℕ :=
  2 * (length + width) * laps

/-- Theorem: Running 3 laps around a 75m by 15m rectangular field results in a total distance of 540m -/
theorem rectangular_field_distance :
  distance_run 75 15 3 = 540 := by
  sorry

end rectangular_field_distance_l3252_325243


namespace sin_210_degrees_l3252_325272

theorem sin_210_degrees : Real.sin (210 * π / 180) = -(1 / 2) := by
  sorry

end sin_210_degrees_l3252_325272


namespace sum_of_squares_of_roots_l3252_325292

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
  sorry

end sum_of_squares_of_roots_l3252_325292


namespace possible_perimeters_only_possible_perimeters_l3252_325275

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the possible ways to cut the original rectangle -/
inductive Cut
  | Vertical
  | Horizontal
  | Mixed

/-- The original rectangle -/
def originalRect : Rectangle := { length := 6, width := 3 }

/-- Theorem stating the possible perimeters of the resulting rectangles -/
theorem possible_perimeters :
  ∃ (c : Cut) (r : Rectangle),
    (c = Cut.Vertical ∧ perimeter r = 14) ∨
    (c = Cut.Horizontal ∧ perimeter r = 10) ∨
    (c = Cut.Mixed ∧ perimeter r = 10.5) :=
  sorry

/-- Theorem stating that these are the only possible perimeters -/
theorem only_possible_perimeters :
  ∀ (c : Cut) (r : Rectangle),
    (perimeter r ≠ 14 ∧ perimeter r ≠ 10 ∧ perimeter r ≠ 10.5) →
    ¬(∃ (r1 r2 : Rectangle), 
      perimeter r = perimeter r1 ∧
      perimeter r = perimeter r2 ∧
      r.length + r1.length + r2.length = originalRect.length ∧
      r.width = r1.width ∧ r.width = r2.width ∧ r.width = originalRect.width) :=
  sorry

end possible_perimeters_only_possible_perimeters_l3252_325275


namespace albert_has_two_snakes_l3252_325281

/-- Represents the number of snakes Albert has -/
def num_snakes : ℕ := 2

/-- Length of the garden snake in inches -/
def garden_snake_length : ℝ := 10.0

/-- Ratio of garden snake length to boa constrictor length -/
def snake_length_ratio : ℝ := 7.0

/-- Length of the boa constrictor in inches -/
def boa_constrictor_length : ℝ := 1.428571429

/-- Theorem stating that Albert has exactly 2 snakes given the conditions -/
theorem albert_has_two_snakes :
  num_snakes = 2 ∧
  garden_snake_length = 10.0 ∧
  boa_constrictor_length = garden_snake_length / snake_length_ratio ∧
  boa_constrictor_length = 1.428571429 :=
by sorry

end albert_has_two_snakes_l3252_325281


namespace function_inequality_implies_a_bound_l3252_325208

theorem function_inequality_implies_a_bound 
  (f g : ℝ → ℝ) 
  (h_f : ∀ x, f x = |x - a| + a) 
  (h_g : ∀ x, g x = 4 - x^2) 
  (h_exists : ∃ x, g x ≥ f x) : 
  a ≤ 17/8 := by
sorry

end function_inequality_implies_a_bound_l3252_325208


namespace teddy_pillows_l3252_325201

/-- The number of pounds in a ton -/
def pounds_per_ton : ℕ := 2000

/-- The amount of fluffy foam material Teddy has, in tons -/
def teddy_material : ℕ := 3

/-- The amount of fluffy foam material used for each pillow, in pounds -/
def material_per_pillow : ℕ := 5 - 3

/-- The number of pillows Teddy can make -/
def pillows_made : ℕ := (teddy_material * pounds_per_ton) / material_per_pillow

theorem teddy_pillows :
  pillows_made = 3000 := by sorry

end teddy_pillows_l3252_325201


namespace votes_for_both_policies_l3252_325264

-- Define the total number of students
def total_students : ℕ := 185

-- Define the number of students voting for the first policy
def first_policy_votes : ℕ := 140

-- Define the number of students voting for the second policy
def second_policy_votes : ℕ := 110

-- Define the number of students voting against both policies
def against_both : ℕ := 22

-- Define the number of students abstaining from both policies
def abstained : ℕ := 15

-- Theorem stating that the number of students voting for both policies is 102
theorem votes_for_both_policies : 
  first_policy_votes + second_policy_votes - total_students + against_both + abstained = 102 :=
by sorry

end votes_for_both_policies_l3252_325264


namespace my_circle_center_l3252_325228

/-- A circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def center (c : Circle) : ℝ × ℝ := sorry

/-- Our specific circle -/
def my_circle : Circle :=
  { equation := fun x y => (x + 2)^2 + y^2 = 5 }

/-- Theorem: The center of our specific circle is (-2, 0) -/
theorem my_circle_center :
  center my_circle = (-2, 0) := by sorry

end my_circle_center_l3252_325228


namespace speedster_convertibles_l3252_325247

theorem speedster_convertibles (total : ℕ) 
  (h1 : 2 * total = 3 * (total - 60))  -- 2/3 of total are Speedsters, 60 are not
  (h2 : 5 * (total - 60) = 3 * total)  -- Restating h1 in a different form
  : (4 * (total - 60)) / 5 = 96 := by  -- 4/5 of Speedsters are convertibles
  sorry

#check speedster_convertibles

end speedster_convertibles_l3252_325247


namespace right_triangle_area_l3252_325288

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 10) (h3 : a = 6) :
  (1/2) * a * b = 24 := by
sorry

end right_triangle_area_l3252_325288


namespace evaluate_expression_l3252_325220

theorem evaluate_expression : 16^3 + 3*(16^2) + 3*16 + 1 = 4913 := by
  sorry

end evaluate_expression_l3252_325220


namespace tens_digit_of_6_to_18_l3252_325257

/-- The tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The theorem stating that the tens digit of 6^18 is 1 -/
theorem tens_digit_of_6_to_18 : tens_digit (6^18) = 1 := by
  sorry

end tens_digit_of_6_to_18_l3252_325257


namespace fourth_number_proof_l3252_325291

theorem fourth_number_proof (x : ℝ) : 
  (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001 → x = 0.3 := by
  sorry

end fourth_number_proof_l3252_325291


namespace profit_percent_calculation_l3252_325231

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.4 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 150 := by
  sorry

end profit_percent_calculation_l3252_325231


namespace polynomial_equality_implies_b_value_l3252_325287

theorem polynomial_equality_implies_b_value 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, (4*x^2 - 2*x + 5/2)*(a*x^2 + b*x + c) = 
                 12*x^4 - 8*x^3 + 15*x^2 - 5*x + 5/2) : 
  b = -1/2 := by
sorry

end polynomial_equality_implies_b_value_l3252_325287


namespace daisy_germination_rate_l3252_325232

/-- Proves that the germination rate of daisy seeds is 60% given the problem conditions --/
theorem daisy_germination_rate :
  let daisy_seeds : ℕ := 25
  let sunflower_seeds : ℕ := 25
  let sunflower_germination_rate : ℚ := 80 / 100
  let flower_production_rate : ℚ := 80 / 100
  let total_flowering_plants : ℕ := 28
  ∃ (daisy_germination_rate : ℚ),
    daisy_germination_rate = 60 / 100 ∧
    (↑daisy_seeds * daisy_germination_rate * flower_production_rate +
     ↑sunflower_seeds * sunflower_germination_rate * flower_production_rate : ℚ) = total_flowering_plants :=
by sorry

end daisy_germination_rate_l3252_325232


namespace arithmetic_sequence_theorem_l3252_325246

def is_arithmetic_sequence (a b c d : ℚ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def sum_is_26 (a b c d : ℚ) : Prop :=
  a + b + c + d = 26

def middle_product_is_40 (b c : ℚ) : Prop :=
  b * c = 40

theorem arithmetic_sequence_theorem (a b c d : ℚ) :
  is_arithmetic_sequence a b c d →
  sum_is_26 a b c d →
  middle_product_is_40 b c →
  ((a = 2 ∧ b = 5 ∧ c = 8 ∧ d = 11) ∨ (a = 11 ∧ b = 8 ∧ c = 5 ∧ d = 2)) :=
by sorry

end arithmetic_sequence_theorem_l3252_325246


namespace child_ticket_cost_l3252_325265

theorem child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_price total_bill : ℚ) :
  num_adults = 10 →
  num_children = 11 →
  adult_ticket_price = 8 →
  total_bill = 124 →
  (total_bill - num_adults * adult_ticket_price) / num_children = 4 :=
by sorry

end child_ticket_cost_l3252_325265


namespace sum_first_150_remainder_l3252_325263

theorem sum_first_150_remainder (n : Nat) (h : n = 150) :
  (n * (n + 1) / 2) % 5000 = 1275 := by
  sorry

end sum_first_150_remainder_l3252_325263


namespace problem_solution_l3252_325234

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.cos y = 1010)
  (h2 : x + 1010 * Real.sin y = 1009)
  (h3 : π / 4 ≤ y ∧ y ≤ π / 2) :
  x + y = 1010 + π / 2 := by
  sorry

end problem_solution_l3252_325234


namespace student_count_equation_l3252_325258

/-- Represents the number of pens per box for the first type of pen -/
def pens_per_box_1 : ℕ := 8

/-- Represents the number of pens per box for the second type of pen -/
def pens_per_box_2 : ℕ := 12

/-- Represents the number of students without pens if x boxes of type 1 are bought -/
def students_without_pens : ℕ := 3

/-- Represents the number of fewer boxes that can be bought of type 2 -/
def fewer_boxes_type_2 : ℕ := 2

/-- Represents the number of pens left in the last box of type 2 -/
def pens_left_type_2 : ℕ := 1

theorem student_count_equation (x : ℕ) : 
  pens_per_box_1 * x + students_without_pens = 
  pens_per_box_2 * (x - fewer_boxes_type_2) - pens_left_type_2 := by
  sorry

end student_count_equation_l3252_325258


namespace inequality_solution_l3252_325240

open Real

theorem inequality_solution (x : ℝ) : 
  (2 * x + 3) / (x + 5) > (5 * x + 7) / (3 * x + 14) ↔ 
  (x > -103.86 ∧ x < -14/3) ∨ (x > -5 ∧ x < -0.14) :=
sorry

end inequality_solution_l3252_325240


namespace product_of_fractions_l3252_325255

theorem product_of_fractions : 
  (7 / 5 : ℚ) * (8 / 16 : ℚ) * (21 / 15 : ℚ) * (14 / 28 : ℚ) * 
  (35 / 25 : ℚ) * (20 / 40 : ℚ) * (49 / 35 : ℚ) * (32 / 64 : ℚ) = 2401 / 10000 := by
  sorry

end product_of_fractions_l3252_325255


namespace choose_positions_count_l3252_325269

def num_people : ℕ := 6
def num_positions : ℕ := 3

theorem choose_positions_count :
  (num_people.factorial) / ((num_people - num_positions).factorial) = 120 :=
sorry

end choose_positions_count_l3252_325269


namespace cone_lateral_surface_area_l3252_325250

/-- Given an equilateral cone with an inscribed sphere of volume 100 cm³,
    the lateral surface area of the cone is 6π * ∛(5625/π²) cm² -/
theorem cone_lateral_surface_area (v : ℝ) (r : ℝ) (l : ℝ) (P : ℝ) :
  v = 100 →  -- volume of the sphere
  v = (4/3) * π * r^3 →  -- volume formula of a sphere
  l = 2 * Real.sqrt 3 * (75/π)^(1/3) →  -- side length of the cone
  P = 6 * π * ((5625:ℝ)/π^2)^(1/3) →  -- lateral surface area of the cone
  P = 6 * π * ((75:ℝ)^2/π^2)^(1/3) :=
by sorry

end cone_lateral_surface_area_l3252_325250


namespace pairs_with_female_l3252_325215

theorem pairs_with_female (total : Nat) (males : Nat) (females : Nat) : 
  total = males + females → males = 3 → females = 3 → 
  (Nat.choose total 2) - (Nat.choose males 2) = 12 := by
  sorry

end pairs_with_female_l3252_325215


namespace increasing_equivalent_l3252_325295

/-- A function is increasing on an interval if its graph always rises when viewed from left to right. -/
def IncreasingOnInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem increasing_equivalent {f : ℝ → ℝ} {I : Set ℝ} :
  IncreasingOnInterval f I ↔
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end increasing_equivalent_l3252_325295


namespace triangle_is_obtuse_l3252_325283

theorem triangle_is_obtuse (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7 / 12) : π / 2 < A ∧ A < π := by
  sorry

end triangle_is_obtuse_l3252_325283


namespace f_properties_l3252_325261

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x + a) / x

def g : ℝ → ℝ := λ x => 1

theorem f_properties (a : ℝ) :
  (∀ x > 0, f a x ≤ Real.exp (a - 1)) ∧
  (∃ x > 0, x ≤ Real.exp 2 ∧ f a x = g x) ↔ a ≥ 1 := by
  sorry

end

end f_properties_l3252_325261


namespace ironman_age_relation_l3252_325204

/-- Represents the age relationship between superheroes -/
structure SuperheroAges where
  thor : ℝ
  captainAmerica : ℝ
  peterParker : ℝ
  ironman : ℝ

/-- The age relationships between the superheroes are valid -/
def validAgeRelationships (ages : SuperheroAges) : Prop :=
  ages.thor = 13 * ages.captainAmerica ∧
  ages.captainAmerica = 7 * ages.peterParker ∧
  ages.ironman = ages.peterParker + 32

/-- Theorem stating the relationship between Ironman's age and Thor's age -/
theorem ironman_age_relation (ages : SuperheroAges) 
  (h : validAgeRelationships ages) : 
  ages.ironman = ages.thor / 91 + 32 := by
  sorry

end ironman_age_relation_l3252_325204


namespace sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l3252_325244

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds :
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l3252_325244


namespace range_of_m_l3252_325274

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) * (x - 1) < 0}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < 1}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∅ ≠ B m) ∧ 
  (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ 
  (∃ y : ℝ, y ∈ A ∧ y ∉ B m) →
  -1 < m ∧ m < 1 :=
by sorry

end range_of_m_l3252_325274


namespace regular_polygon_sides_l3252_325235

theorem regular_polygon_sides (central_angle : ℝ) : 
  central_angle = 20 → (360 : ℝ) / central_angle = 18 := by
  sorry

end regular_polygon_sides_l3252_325235


namespace square_root_expression_l3252_325221

theorem square_root_expression (m n : ℝ) : 
  Real.sqrt ((m - 2*n - 3) * (m - 2*n + 3) + 9) = 
    if m ≥ 2*n then m - 2*n else 2*n - m := by
  sorry

end square_root_expression_l3252_325221


namespace initial_markup_percentage_l3252_325242

theorem initial_markup_percentage (C : ℝ) (h : C > 0) : 
  ∃ M : ℝ, 
    M ≥ 0 ∧ 
    C * (1 + M) * 1.25 * 0.93 = C * (1 + 0.395) ∧ 
    M = 0.2 := by
  sorry

end initial_markup_percentage_l3252_325242


namespace bitna_elementary_students_l3252_325233

/-- The number of pencils purchased by Bitna Elementary School -/
def total_pencils : ℕ := 10395

/-- The number of pencils distributed to each student -/
def pencils_per_student : ℕ := 11

/-- The number of students in Bitna Elementary School -/
def number_of_students : ℕ := total_pencils / pencils_per_student

theorem bitna_elementary_students : number_of_students = 945 := by
  sorry

end bitna_elementary_students_l3252_325233


namespace a_equals_one_sufficient_not_necessary_l3252_325237

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + a = 0

-- Define what it means for the equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem a_equals_one_sufficient_not_necessary :
  (represents_circle 1) ∧
  (∃ (a : ℝ), a ≠ 1 ∧ represents_circle a) :=
sorry

end a_equals_one_sufficient_not_necessary_l3252_325237


namespace g_25_l3252_325262

-- Define the function g
variable (g : ℝ → ℝ)

-- State the conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = y * g x
axiom g_50 : g 50 = 10

-- State the theorem to be proved
theorem g_25 : g 25 = 20 := by sorry

end g_25_l3252_325262


namespace min_intersection_size_l3252_325239

theorem min_intersection_size (total students_with_brown_eyes students_with_lunch_box : ℕ) 
  (h1 : total = 25)
  (h2 : students_with_brown_eyes = 15)
  (h3 : students_with_lunch_box = 18) :
  ∃ (intersection : ℕ), 
    intersection ≤ students_with_brown_eyes ∧ 
    intersection ≤ students_with_lunch_box ∧
    intersection ≥ students_with_brown_eyes + students_with_lunch_box - total ∧
    intersection = 8 :=
by sorry

end min_intersection_size_l3252_325239


namespace decimal_fraction_equality_l3252_325256

theorem decimal_fraction_equality (b : ℕ) : 
  b > 0 ∧ (5 * b + 22 : ℚ) / (7 * b + 15) = 87 / 100 → b = 8 := by
  sorry

end decimal_fraction_equality_l3252_325256
