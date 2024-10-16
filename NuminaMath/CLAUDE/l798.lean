import Mathlib

namespace NUMINAMATH_CALUDE_real_part_expression1_real_part_expression2_real_part_expression3_l798_79881

open Complex

-- Define the function f that returns the real part of a complex number
def f (z : ℂ) : ℝ := z.re

-- Theorem 1
theorem real_part_expression1 : f ((1 + 2*I)^2 + 3*(1 - I)) / (2 + I) = 1/5 := by sorry

-- Theorem 2
theorem real_part_expression2 : f (1 + (1 - I) / (1 + I)^2 + (1 + I) / (1 - I)^2) = -1 := by sorry

-- Theorem 3
theorem real_part_expression3 : f (1 + (1 - Complex.I * Real.sqrt 3) / (Real.sqrt 3 + I)^2) = 3/4 := by sorry

end NUMINAMATH_CALUDE_real_part_expression1_real_part_expression2_real_part_expression3_l798_79881


namespace NUMINAMATH_CALUDE_petya_vasya_divisibility_l798_79816

theorem petya_vasya_divisibility (n m : ℕ) (h : ∀ k ∈ Finset.range 100, ∃ j ∈ Finset.range 99, (m - j) ∣ (n + k)) :
  m > n^3 / 10000000 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_divisibility_l798_79816


namespace NUMINAMATH_CALUDE_nonnegative_rational_function_l798_79827

theorem nonnegative_rational_function (x : ℝ) :
  (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_rational_function_l798_79827


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_union_equals_A_iff_m_in_range_l798_79866

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 5}) ∧
  ((Aᶜ ∪ B 3) = {x | x < -2 ∨ 2 ≤ x}) :=
sorry

-- Theorem for part 2
theorem union_equals_A_iff_m_in_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_union_equals_A_iff_m_in_range_l798_79866


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_distance_l798_79886

/-- The ellipse with equation 9x² + 16y² = 114 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | 9 * p.1^2 + 16 * p.2^2 = 114}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The distance from a point to a line defined by two points -/
noncomputable def distanceToLine (p q r : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_perpendicular_distance :
  ∀ (P Q : ℝ × ℝ),
  P ∈ Ellipse →
  Q ∈ Ellipse →
  (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 0 →
  distanceToLine O P Q = 12/5 := by
    sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_distance_l798_79886


namespace NUMINAMATH_CALUDE_john_travel_money_l798_79895

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remainingMoney (savings : ℕ) (ticketCost : ℕ) : ℕ :=
  base8ToBase10 savings - ticketCost

theorem john_travel_money :
  remainingMoney 5555 1200 = 1725 := by sorry

end NUMINAMATH_CALUDE_john_travel_money_l798_79895


namespace NUMINAMATH_CALUDE_find_x_l798_79898

def A : Set ℝ := {0, 2, 3}
def B (x : ℝ) : Set ℝ := {x + 1, x^2 + 4}

theorem find_x : ∃ x : ℝ, A ∩ B x = {3} → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l798_79898


namespace NUMINAMATH_CALUDE_range_of_m_l798_79877

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l798_79877


namespace NUMINAMATH_CALUDE_htf_sequence_probability_l798_79858

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a specific sequence of three independent coin flips -/
def prob_sequence (p : ℝ) : ℝ := p * p * p

theorem htf_sequence_probability :
  ∀ p : ℝ, fair_coin p → prob_sequence p = 1/8 := by sorry

end NUMINAMATH_CALUDE_htf_sequence_probability_l798_79858


namespace NUMINAMATH_CALUDE_fred_cantelopes_count_l798_79821

/-- The number of cantelopes grown by Fred and Tim together -/
def total_cantelopes : ℕ := 82

/-- The number of cantelopes grown by Tim -/
def tim_cantelopes : ℕ := 44

/-- The number of cantelopes grown by Fred -/
def fred_cantelopes : ℕ := total_cantelopes - tim_cantelopes

theorem fred_cantelopes_count : fred_cantelopes = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_cantelopes_count_l798_79821


namespace NUMINAMATH_CALUDE_tangent_line_circle_a_value_l798_79870

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line equals the radius of the circle. -/
axiom line_tangent_to_circle_iff_distance_eq_radius {a b c d e f : ℝ} :
  (∀ x y, a*x + b*y + c = 0 → (x - d)^2 + (y - e)^2 = f^2) ↔
  |a*d + b*e + c| / Real.sqrt (a^2 + b^2) = f

/-- Given that the line 5x + 12y + a = 0 is tangent to the circle x^2 - 2x + y^2 = 0,
    prove that a = 8 or a = -18 -/
theorem tangent_line_circle_a_value :
  (∀ x y, 5*x + 12*y + a = 0 → x^2 - 2*x + y^2 = 0) →
  a = 8 ∨ a = -18 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_a_value_l798_79870


namespace NUMINAMATH_CALUDE_complex_magnitude_l798_79860

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l798_79860


namespace NUMINAMATH_CALUDE_champion_is_c_l798_79873

-- Define the athletes
inductive Athlete : Type
| a : Athlete
| b : Athlete
| c : Athlete

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the correctness of a statement
inductive Correctness : Type
| Correct : Correctness
| HalfCorrect : Correctness
| Incorrect : Correctness

-- Define the champion
def champion : Athlete := Athlete.c

-- Define the statements made by each student
def statement (s : Student) : Athlete × Athlete :=
  match s with
  | Student.A => (Athlete.b, Athlete.c)
  | Student.B => (Athlete.b, Athlete.a)
  | Student.C => (Athlete.c, Athlete.b)

-- Define the correctness of each student's statement
def studentCorrectness (s : Student) : Correctness :=
  match s with
  | Student.A => Correctness.Correct
  | Student.B => Correctness.HalfCorrect
  | Student.C => Correctness.Incorrect

-- Theorem to prove
theorem champion_is_c :
  (∀ s : Student, (statement s).1 ≠ champion → (statement s).2 = champion ↔ studentCorrectness s = Correctness.Correct) ∧
  (∃! s : Student, studentCorrectness s = Correctness.Correct) ∧
  (∃! s : Student, studentCorrectness s = Correctness.HalfCorrect) ∧
  (∃! s : Student, studentCorrectness s = Correctness.Incorrect) →
  champion = Athlete.c := by
  sorry

end NUMINAMATH_CALUDE_champion_is_c_l798_79873


namespace NUMINAMATH_CALUDE_simplify_expression_l798_79854

theorem simplify_expression : 5 * (18 / (-9)) * (24 / 36) = -20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l798_79854


namespace NUMINAMATH_CALUDE_binomial_16_13_l798_79809

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by sorry

end NUMINAMATH_CALUDE_binomial_16_13_l798_79809


namespace NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l798_79893

theorem roots_sum_of_sixth_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 7 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 7 + 1 = 0 →
  r^6 + s^6 = 389374 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l798_79893


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l798_79874

open Real

noncomputable def f (x : ℝ) := -log x + (1/2) * x^2

theorem tangent_line_and_max_value :
  (∀ x, x ∈ Set.Icc (1/Real.exp 1) (Real.sqrt (Real.exp 1)) →
    f x ≤ 1 + 1 / (2 * (Real.exp 1)^2)) ∧
  (∃ x, x ∈ Set.Icc (1/Real.exp 1) (Real.sqrt (Real.exp 1)) ∧
    f x = 1 + 1 / (2 * (Real.exp 1)^2)) ∧
  (3 * 2 - 2 * f 2 - 2 - 2 * log 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_max_value_l798_79874


namespace NUMINAMATH_CALUDE_garden_area_increase_l798_79848

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rectangular_length : ℝ := 60
  let rectangular_width : ℝ := 20
  let perimeter : ℝ := 2 * (rectangular_length + rectangular_width)
  let square_side : ℝ := perimeter / 4
  let rectangular_area : ℝ := rectangular_length * rectangular_width
  let square_area : ℝ := square_side * square_side
  square_area - rectangular_area = 400 :=
by sorry


end NUMINAMATH_CALUDE_garden_area_increase_l798_79848


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l798_79836

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- Theorem: In a rhombus with one diagonal of 25 m and an area of 625 m², the other diagonal is 50 m -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.d1 = 25) (h2 : r.area = 625) : r.d2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l798_79836


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l798_79829

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (3 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 3 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l798_79829


namespace NUMINAMATH_CALUDE_factorization_3x2_minus_27y2_l798_79802

theorem factorization_3x2_minus_27y2 (x y : ℝ) : 3 * x^2 - 27 * y^2 = 3 * (x + 3*y) * (x - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x2_minus_27y2_l798_79802


namespace NUMINAMATH_CALUDE_abc_equality_l798_79830

theorem abc_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (1 - b) = 1/4) (h2 : b * (1 - c) = 1/4) (h3 : c * (1 - a) = 1/4) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_abc_equality_l798_79830


namespace NUMINAMATH_CALUDE_bacon_tomato_difference_l798_79894

theorem bacon_tomato_difference (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 228)
  (h2 : bacon = 337)
  (h3 : tomatoes = 23) :
  bacon - tomatoes = 314 := by
  sorry

end NUMINAMATH_CALUDE_bacon_tomato_difference_l798_79894


namespace NUMINAMATH_CALUDE_jasons_pepper_spray_dilemma_l798_79823

theorem jasons_pepper_spray_dilemma :
  ¬ ∃ (raccoons squirrels opossums : ℕ),
    squirrels = 6 * raccoons ∧
    opossums = 2 * raccoons ∧
    raccoons + squirrels + opossums = 168 :=
by sorry

end NUMINAMATH_CALUDE_jasons_pepper_spray_dilemma_l798_79823


namespace NUMINAMATH_CALUDE_parabola_shift_l798_79808

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the horizontal shift
def horizontal_shift : ℝ := 2

-- Define the vertical shift
def vertical_shift : ℝ := 3

-- Define the resulting parabola after shifts
def shifted_parabola (x : ℝ) : ℝ := 5 * (x + horizontal_shift)^2 + vertical_shift

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l798_79808


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l798_79865

theorem simplify_sqrt_expression (y : ℝ) (h : y ≠ 0) : 
  Real.sqrt (4 + ((y^3 - 2) / (3 * y))^2) = (Real.sqrt (y^6 - 4*y^3 + 36*y^2 + 4)) / (3 * y) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l798_79865


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l798_79834

theorem arctan_sum_equals_pi_fourth (y : ℝ) : 
  2 * Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/y) = π/4 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l798_79834


namespace NUMINAMATH_CALUDE_files_deleted_l798_79850

theorem files_deleted (initial_files initial_apps final_files final_apps : ℕ) 
  (h1 : initial_files = 24)
  (h2 : initial_apps = 13)
  (h3 : final_files = 21)
  (h4 : final_apps = 17) :
  initial_files - final_files = 3 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l798_79850


namespace NUMINAMATH_CALUDE_accurate_estimation_l798_79882

/-- Represents a scale reading on a measuring device --/
structure ScaleReading where
  min : Float
  max : Float
  reading : Float
  min_le_reading : min ≤ reading
  reading_le_max : reading ≤ max

/-- The most accurate estimation for a scale reading --/
def mostAccurateEstimation (s : ScaleReading) : Float :=
  15.9

/-- Theorem stating that 15.9 is the most accurate estimation for the given scale reading --/
theorem accurate_estimation (s : ScaleReading) 
  (h1 : s.min = 15.75) 
  (h2 : s.max = 16.0) : 
  mostAccurateEstimation s = 15.9 := by
  sorry

end NUMINAMATH_CALUDE_accurate_estimation_l798_79882


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l798_79840

theorem cubic_equation_solution :
  ∀ x y : ℕ+, x^3 - y^3 = 999 ↔ (x = 12 ∧ y = 9) ∨ (x = 10 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l798_79840


namespace NUMINAMATH_CALUDE_exist_three_quadratics_with_specific_root_properties_l798_79826

theorem exist_three_quadratics_with_specific_root_properties :
  ∃ (p₁ p₂ p₃ : ℝ → ℝ),
    (∃ x₁, p₁ x₁ = 0) ∧
    (∃ x₂, p₂ x₂ = 0) ∧
    (∃ x₃, p₃ x₃ = 0) ∧
    (∀ x, p₁ x + p₂ x ≠ 0) ∧
    (∀ x, p₂ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x = (x - 1)^2) ∧
    (∀ x, p₂ x = x^2) ∧
    (∀ x, p₃ x = (x - 2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_exist_three_quadratics_with_specific_root_properties_l798_79826


namespace NUMINAMATH_CALUDE_two_true_propositions_l798_79872

theorem two_true_propositions :
  let prop1 := ∀ a : ℝ, a > -1 → a > -2
  let prop2 := ∀ a : ℝ, a > -2 → a > -1
  let prop3 := ∀ a : ℝ, a ≤ -1 → a ≤ -2
  let prop4 := ∀ a : ℝ, a ≤ -2 → a ≤ -1
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l798_79872


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l798_79807

theorem shoe_price_calculation (initial_money : ℝ) (sweater_price : ℝ) (tshirt_price : ℝ) (refund_percentage : ℝ) (final_money : ℝ) :
  initial_money = 74 →
  sweater_price = 9 →
  tshirt_price = 11 →
  refund_percentage = 0.9 →
  final_money = 51 →
  ∃ (shoe_price : ℝ),
    shoe_price = 30 ∧
    final_money = initial_money - sweater_price - tshirt_price - shoe_price + refund_percentage * shoe_price :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l798_79807


namespace NUMINAMATH_CALUDE_willies_stickers_l798_79806

theorem willies_stickers (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  given = 7 → remaining = 29 → initial = remaining + given :=
by
  sorry

end NUMINAMATH_CALUDE_willies_stickers_l798_79806


namespace NUMINAMATH_CALUDE_triangle_properties_l798_79879

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.sin t.A * Real.cos t.B = 2 * Real.sin t.C - Real.sin t.B)
  (h2 : t.a = 4 * Real.sqrt 3)
  (h3 : t.b + t.c = 8) :
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l798_79879


namespace NUMINAMATH_CALUDE_oliver_stickers_l798_79841

theorem oliver_stickers (S : ℕ) : 
  (3/5 : ℚ) * (2/3 : ℚ) * S = 54 → S = 135 := by
sorry

end NUMINAMATH_CALUDE_oliver_stickers_l798_79841


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l798_79869

theorem quadratic_equation_solution_difference : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 - 5*x₁ + 11 = x₁ + 27) → 
  (x₂^2 - 5*x₂ + 11 = x₂ + 27) → 
  x₁ ≠ x₂ →
  |x₁ - x₂| = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l798_79869


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_C_internal_tangency_of_circles_l798_79835

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - 2*m)^2 = m^2

-- Define the circle E
def circle_E (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ x = 0

-- Theorem for part (I)
theorem tangent_line_to_circle_C :
  ∀ x y : ℝ, circle_C 2 x y → tangent_line x y → (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0) :=
sorry

-- Theorem for part (II)
theorem internal_tangency_of_circles :
  ∃ x y : ℝ, circle_C ((Real.sqrt 29 - 1) / 4) x y ∧ circle_E x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_C_internal_tangency_of_circles_l798_79835


namespace NUMINAMATH_CALUDE_equal_angles_with_perpendicular_circle_l798_79847

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (passes_through : Circle → Point → Prop)
variable (tangent_to : Circle → Circle → Prop)
variable (perpendicular_to : Circle → Circle → Prop)
variable (angle_between : Circle → Circle → ℝ)

-- State the theorem
theorem equal_angles_with_perpendicular_circle
  (A B : Point) (S S₁ S₂ S₃ : Circle)
  (h1 : passes_through S₁ A ∧ passes_through S₁ B)
  (h2 : passes_through S₂ A ∧ passes_through S₂ B)
  (h3 : tangent_to S₁ S)
  (h4 : tangent_to S₂ S)
  (h5 : perpendicular_to S₃ S) :
  angle_between S₃ S₁ = angle_between S₃ S₂ :=
by sorry

end NUMINAMATH_CALUDE_equal_angles_with_perpendicular_circle_l798_79847


namespace NUMINAMATH_CALUDE_metal_price_calculation_l798_79842

/-- Given two metals mixed in a 3:1 ratio, prove the price of the first metal -/
theorem metal_price_calculation (price_second : ℚ) (price_alloy : ℚ) :
  price_second = 96 →
  price_alloy = 75 →
  ∃ (price_first : ℚ),
    price_first = 68 ∧
    (3 * price_first + 1 * price_second) / 4 = price_alloy :=
by sorry

end NUMINAMATH_CALUDE_metal_price_calculation_l798_79842


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l798_79875

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 60) :
  x^3 * y^4 ≤ (3/7 * 60)^3 * (4/7 * 60)^4 ∧
  x^3 * y^4 = (3/7 * 60)^3 * (4/7 * 60)^4 ↔ x = 3/7 * 60 ∧ y = 4/7 * 60 :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l798_79875


namespace NUMINAMATH_CALUDE_functional_equation_solution_l798_79859

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = 2 * x + 2 * y + 8) :
  ∀ x : ℝ, g x = -2 * x - 7 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l798_79859


namespace NUMINAMATH_CALUDE_determinant_special_matrix_l798_79857

open Matrix

theorem determinant_special_matrix (x : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x + 2, x, x; x, x + 2, x; x, x, x + 2]
  det A = 16 * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_determinant_special_matrix_l798_79857


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l798_79896

/-- An arithmetic sequence is a sequence where the difference between 
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l798_79896


namespace NUMINAMATH_CALUDE_sequence_with_nondivisible_sums_l798_79853

theorem sequence_with_nondivisible_sums (k : ℕ) (h : Even k) (h' : k > 0) :
  ∃ π : Fin (k - 1) → Fin (k - 1), Function.Bijective π ∧
    ∀ (i j : Fin (k - 1)), i ≤ j →
      ¬(k ∣ (Finset.sum (Finset.Icc i j) (fun n => (π n).val + 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_with_nondivisible_sums_l798_79853


namespace NUMINAMATH_CALUDE_tank_capacity_l798_79863

theorem tank_capacity (x : ℝ) 
  (h1 : x / 3 + 180 = 2 * x / 3) : x = 540 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l798_79863


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l798_79890

theorem ratio_of_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (sum_diff : a + b = 7 * (a - b)) (product : a * b = 50) :
  max a b / min a b = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l798_79890


namespace NUMINAMATH_CALUDE_equation_solutions_l798_79885

theorem equation_solutions :
  (∃ x : ℝ, 8 * (x + 1)^3 = 64 ∧ x = 1) ∧
  (∃ x y : ℝ, (x + 1)^2 = 100 ∧ (y + 1)^2 = 100 ∧ x = 9 ∧ y = -11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l798_79885


namespace NUMINAMATH_CALUDE_remainder_of_power_700_l798_79812

theorem remainder_of_power_700 (n : ℕ) (h : n^700 % 100 = 1) : n^700 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_700_l798_79812


namespace NUMINAMATH_CALUDE_three_digit_sum_not_always_three_digits_l798_79888

theorem three_digit_sum_not_always_three_digits : ∃ (a b : ℕ), 
  100 ≤ a ∧ a ≤ 999 ∧ 100 ≤ b ∧ b ≤ 999 ∧ 1000 ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_not_always_three_digits_l798_79888


namespace NUMINAMATH_CALUDE_ratio_problem_l798_79810

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3) 
  (h2 : c / b = 2) 
  (h3 : d / c = 4) : 
  (a + c) / (b + d) = 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l798_79810


namespace NUMINAMATH_CALUDE_first_hour_distance_l798_79891

/-- A structure representing a family road trip -/
structure RoadTrip where
  firstHourDistance : ℝ
  remainingDistance : ℝ
  totalTime : ℝ
  speed : ℝ

/-- Theorem: Given the conditions of the road trip, the distance traveled in the first hour is 100 miles -/
theorem first_hour_distance (trip : RoadTrip) 
  (h1 : trip.remainingDistance = 300)
  (h2 : trip.totalTime = 4)
  (h3 : trip.speed * 1 = trip.firstHourDistance)
  (h4 : trip.speed * 3 = trip.remainingDistance) : 
  trip.firstHourDistance = 100 := by
  sorry

#check first_hour_distance

end NUMINAMATH_CALUDE_first_hour_distance_l798_79891


namespace NUMINAMATH_CALUDE_probability_two_forks_two_spoons_l798_79839

/-- The number of forks in the drawer -/
def num_forks : ℕ := 8

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 10

/-- The number of knives in the drawer -/
def num_knives : ℕ := 6

/-- The total number of pieces of silverware -/
def total_silverware : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be randomly selected -/
def num_selected : ℕ := 4

/-- The probability of selecting two forks and two spoons when randomly choosing
    four pieces of silverware from the drawer -/
theorem probability_two_forks_two_spoons :
  (Nat.choose num_forks 2 * Nat.choose num_spoons 2 : ℚ) /
  (Nat.choose total_silverware num_selected) = 18 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_forks_two_spoons_l798_79839


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_7_factorial_plus_8_factorial_l798_79825

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

theorem largest_prime_factor_of_7_factorial_plus_8_factorial :
  largest_prime_factor (factorial 7 + factorial 8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_7_factorial_plus_8_factorial_l798_79825


namespace NUMINAMATH_CALUDE_complement_intersection_when_a_is_3_range_of_a_when_union_equals_B_range_of_a_when_intersection_is_empty_l798_79801

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Theorem 1: When a=3, ℂᴿ(A∩B) = {x | x < 2 or x > 4}
theorem complement_intersection_when_a_is_3 :
  (Set.univ \ (A 3 ∩ B)) = {x | x < 2 ∨ x > 4} := by sorry

-- Theorem 2: When A∪B=B, the range of a is (-∞,-2)∪[-1,3/2]
theorem range_of_a_when_union_equals_B :
  (∀ a, A a ∪ B = B) ↔ (∀ a, a < -2 ∨ (-1 ≤ a ∧ a ≤ 3/2)) := by sorry

-- Theorem 3: When A∩B=∅, the range of a is (-∞,-3/2)∪(5,+∞)
theorem range_of_a_when_intersection_is_empty :
  (∀ a, A a ∩ B = ∅) ↔ (∀ a, a < -3/2 ∨ a > 5) := by sorry

end NUMINAMATH_CALUDE_complement_intersection_when_a_is_3_range_of_a_when_union_equals_B_range_of_a_when_intersection_is_empty_l798_79801


namespace NUMINAMATH_CALUDE_characterization_of_f_l798_79862

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def NonNegative (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x ≥ 0

def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ a b c d : ℝ, a * b + b * c + c * d = 0 →
    f (a - b) + f (c - d) = f a + f (b + c) + f d

-- Main theorem
theorem characterization_of_f (f : RealFunction)
  (h1 : NonNegative f)
  (h2 : SatisfiesEquation f) :
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * x^2 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_f_l798_79862


namespace NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l798_79813

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem ratio_A_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l798_79813


namespace NUMINAMATH_CALUDE_first_box_weight_proof_l798_79817

/-- The weight of the first box given the conditions in the problem -/
def first_box_weight : ℝ := 24

/-- The weight of the third box -/
def third_box_weight : ℝ := 13

/-- The difference between the weight of the first and third box -/
def weight_difference : ℝ := 11

theorem first_box_weight_proof :
  first_box_weight = third_box_weight + weight_difference := by
  sorry

end NUMINAMATH_CALUDE_first_box_weight_proof_l798_79817


namespace NUMINAMATH_CALUDE_neglart_students_count_l798_79867

/-- Represents the number of toes on a Hoopit's hand -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands a Hoopit has -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes on a Neglart's hand -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands a Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that the number of Neglart students on the bus is 8 -/
theorem neglart_students_count : ∃ (n : ℕ), 
  n * (neglart_toes_per_hand * neglart_hands) + 
  hoopit_students * (hoopit_toes_per_hand * hoopit_hands) = total_toes ∧ 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_neglart_students_count_l798_79867


namespace NUMINAMATH_CALUDE_three_factors_for_cash_preference_l798_79805

/-- Represents an economic factor influencing payment preference --/
structure EconomicFactor where
  name : String
  description : String

/-- Represents a large retail chain --/
structure RetailChain where
  name : String
  prefersCash : Bool

/-- Determines if an economic factor contributes to cash preference --/
def contributesToCashPreference (factor : EconomicFactor) (chain : RetailChain) : Prop :=
  factor.description ≠ "" ∧ chain.prefersCash

/-- The main theorem stating that there are at least three distinct economic factors
    contributing to cash preference for large retail chains --/
theorem three_factors_for_cash_preference :
  ∃ (f1 f2 f3 : EconomicFactor) (chain : RetailChain),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    contributesToCashPreference f1 chain ∧
    contributesToCashPreference f2 chain ∧
    contributesToCashPreference f3 chain :=
  sorry

end NUMINAMATH_CALUDE_three_factors_for_cash_preference_l798_79805


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l798_79845

universe u

theorem complement_intersection_problem :
  let U : Set ℕ := {1, 2, 3, 4, 5}
  let M : Set ℕ := {3, 4, 5}
  let N : Set ℕ := {2, 3}
  (U \ N) ∩ M = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l798_79845


namespace NUMINAMATH_CALUDE_units_digit_of_17_power_2007_l798_79800

theorem units_digit_of_17_power_2007 : 17^2007 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_power_2007_l798_79800


namespace NUMINAMATH_CALUDE_volleyball_team_girls_l798_79880

/-- Given a volleyball team with the following properties:
  * The total number of team members is 30
  * 20 members attended the last meeting
  * One-third of the girls and all boys attended the meeting
  Prove that the number of girls on the team is 15 -/
theorem volleyball_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 20 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = attended →
  girls = 15 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_girls_l798_79880


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l798_79897

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m + 1, -2)
  let b : ℝ × ℝ := (-3, 3)
  parallel a b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l798_79897


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l798_79815

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (8, -3)

/-- The expected center of the reflected circle -/
def expected_reflected_center : ℝ × ℝ := (3, -8)

theorem reflection_of_circle_center :
  reflect_about_diagonal original_center = expected_reflected_center := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l798_79815


namespace NUMINAMATH_CALUDE_sum_after_100_operations_l798_79832

/-- The operation that inserts the difference between each pair of neighboring numbers -/
def insertDifferences (s : List Int) : List Int :=
  sorry

/-- Applies the insertDifferences operation n times to a list -/
def applyNTimes (s : List Int) (n : Nat) : List Int :=
  sorry

/-- The sum of a list of integers -/
def listSum (s : List Int) : Int :=
  sorry

theorem sum_after_100_operations :
  let initialSequence : List Int := [1, 9, 8, 8]
  listSum (applyNTimes initialSequence 100) = 726 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_100_operations_l798_79832


namespace NUMINAMATH_CALUDE_milk_tea_sales_l798_79818

-- Define the relationship between cups of milk tea and total sales price
def sales_price (x : ℕ) : ℕ := 10 * x + 2

-- Theorem stating the conditions and the result to be proved
theorem milk_tea_sales :
  (sales_price 1 = 12) →
  (sales_price 2 = 22) →
  (∃ x : ℕ, sales_price x = 822) →
  (∃ x : ℕ, sales_price x = 822 ∧ x = 82) :=
by sorry

end NUMINAMATH_CALUDE_milk_tea_sales_l798_79818


namespace NUMINAMATH_CALUDE_x_plus_y_equals_four_l798_79838

/-- Geometric configuration with segments AB and A'B' --/
structure GeometricConfiguration where
  AB : ℝ
  APB : ℝ
  P_distance_from_D : ℝ
  total_distance : ℝ

/-- Theorem stating that x + y = 4 in the given geometric configuration --/
theorem x_plus_y_equals_four (config : GeometricConfiguration) 
  (h1 : config.AB = 6)
  (h2 : config.APB = 10)
  (h3 : config.P_distance_from_D = 2)
  (h4 : config.total_distance = 12) :
  let D := config.AB / 2
  let D' := config.APB / 2
  let x := config.P_distance_from_D
  let y := config.total_distance - (D + x + D')
  x + y = 4 := by
  sorry


end NUMINAMATH_CALUDE_x_plus_y_equals_four_l798_79838


namespace NUMINAMATH_CALUDE_strictly_decreasing_function_l798_79822

/-- A function satisfying the given condition -/
noncomputable def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, ∃ (S : Finset ℝ), ∀ y ∈ S, y > 0 ∧ (x + f y) * (y + f x) ≤ 4

/-- The main theorem -/
theorem strictly_decreasing_function 
  (f : ℝ → ℝ) (h : SatisfiesCondition f) :
  ∀ x y, 0 < x ∧ x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_strictly_decreasing_function_l798_79822


namespace NUMINAMATH_CALUDE_triangle_circle_radii_relation_l798_79884

/-- Given a triangle with sides of consecutive natural numbers, 
    the radius of its circumcircle (R) and the radius of its incircle (r) 
    satisfy the equation R = 2r + 1/(2r) -/
theorem triangle_circle_radii_relation (n : ℕ) (R r : ℝ) 
    (h1 : n > 1) 
    (h2 : R = (n^2 - 1) / (6 * r)) 
    (h3 : r^2 = (n^2 - 4) / 12) : 
  R = 2*r + 1/(2*r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_radii_relation_l798_79884


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_hcf_lcm_l798_79851

theorem product_of_numbers_with_given_hcf_lcm :
  ∀ (a b : ℕ+),
  Nat.gcd a b = 33 →
  Nat.lcm a b = 2574 →
  a * b = 84942 :=
by sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_hcf_lcm_l798_79851


namespace NUMINAMATH_CALUDE_leo_balloon_distribution_l798_79883

theorem leo_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 144) 
  (h2 : num_friends = 9) :
  total_balloons % num_friends = 0 := by
  sorry

end NUMINAMATH_CALUDE_leo_balloon_distribution_l798_79883


namespace NUMINAMATH_CALUDE_kyles_rent_calculation_l798_79855

def monthly_income : ℕ := 3200

def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries_eating_out : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350
def gas_maintenance : ℕ := 350

def total_expenses : ℕ :=
  utilities + retirement_savings + groceries_eating_out + insurance +
  miscellaneous + car_payment + gas_maintenance

def rent : ℕ := monthly_income - total_expenses

theorem kyles_rent_calculation :
  rent = 1250 :=
sorry

end NUMINAMATH_CALUDE_kyles_rent_calculation_l798_79855


namespace NUMINAMATH_CALUDE_range_of_fraction_l798_79871

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2*x = 0) :
  -1 ≤ (y - x) / (x - 1) ∧ (y - x) / (x - 1) ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l798_79871


namespace NUMINAMATH_CALUDE_class_3_1_fairy_tales_l798_79819

theorem class_3_1_fairy_tales (andersen : ℕ) (grimm : ℕ) (both : ℕ) (total : ℕ) :
  andersen = 20 →
  grimm = 27 →
  both = 8 →
  total = 55 →
  andersen + grimm - both ≠ total :=
by
  sorry

end NUMINAMATH_CALUDE_class_3_1_fairy_tales_l798_79819


namespace NUMINAMATH_CALUDE_intersection_condition_l798_79828

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l798_79828


namespace NUMINAMATH_CALUDE_fish_pond_population_l798_79843

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 30 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged + 750) :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l798_79843


namespace NUMINAMATH_CALUDE_ratio_problem_l798_79864

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a / b = 1.40625 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l798_79864


namespace NUMINAMATH_CALUDE_uncle_zhang_age_uncle_zhang_age_proof_l798_79820

theorem uncle_zhang_age : Nat → Nat → Prop :=
  fun zhang_age li_age =>
    zhang_age + li_age = 56 ∧
    2 * (li_age - (li_age - zhang_age)) = li_age ∧
    zhang_age = 24

-- The proof is omitted
theorem uncle_zhang_age_proof : ∃ (zhang_age li_age : Nat), uncle_zhang_age zhang_age li_age :=
  sorry

end NUMINAMATH_CALUDE_uncle_zhang_age_uncle_zhang_age_proof_l798_79820


namespace NUMINAMATH_CALUDE_male_students_count_l798_79892

/-- Represents the number of students in a grade. -/
def total_students : ℕ := 800

/-- Represents the size of the stratified sample. -/
def sample_size : ℕ := 20

/-- Represents the number of female students in the sample. -/
def females_in_sample : ℕ := 8

/-- Calculates the number of male students in the entire grade based on stratified sampling. -/
def male_students_in_grade : ℕ := 
  (total_students * (sample_size - females_in_sample)) / sample_size

/-- Theorem stating that the number of male students in the grade is 480. -/
theorem male_students_count : male_students_in_grade = 480 := by sorry

end NUMINAMATH_CALUDE_male_students_count_l798_79892


namespace NUMINAMATH_CALUDE_town_population_problem_l798_79849

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1500) * 85 / 100) : ℕ) = original_population - 45 →
  original_population = 8800 := by
sorry

end NUMINAMATH_CALUDE_town_population_problem_l798_79849


namespace NUMINAMATH_CALUDE_f_two_values_l798_79868

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| = |x - y|

/-- Theorem stating the possible values of f(2) given the conditions -/
theorem f_two_values (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_two_values_l798_79868


namespace NUMINAMATH_CALUDE_total_cost_usd_l798_79803

/-- Calculate the total cost of items with discounts and tax -/
def calculate_total_cost (shirt_price : ℚ) (shoe_price_diff : ℚ) (dress_price : ℚ)
  (shoe_discount : ℚ) (dress_discount : ℚ) (sales_tax : ℚ) (exchange_rate : ℚ) : ℚ :=
  let shoe_price := shirt_price + shoe_price_diff
  let discounted_shoe_price := shoe_price * (1 - shoe_discount)
  let discounted_dress_price := dress_price * (1 - dress_discount)
  let subtotal := 2 * shirt_price + discounted_shoe_price + discounted_dress_price
  let bag_price := subtotal / 2
  let total_before_tax := subtotal + bag_price
  let tax_amount := total_before_tax * sales_tax
  let total_with_tax := total_before_tax + tax_amount
  total_with_tax * exchange_rate

/-- Theorem stating the total cost in USD -/
theorem total_cost_usd :
  calculate_total_cost 12 5 25 (1/10) (1/20) (7/100) (118/100) = 11942/100 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_usd_l798_79803


namespace NUMINAMATH_CALUDE_li_point_parabola_range_l798_79878

/-- A point (x, y) is a "Li point" if x and y have opposite signs -/
def is_li_point (x y : ℝ) : Prop := x * y < 0

/-- Parabola equation -/
def parabola (a c x : ℝ) : ℝ := a * x^2 - 7 * x + c

theorem li_point_parabola_range (a c : ℝ) :
  a > 1 →
  (∃! x : ℝ, is_li_point x (parabola a c x)) →
  0 < c ∧ c < 9 :=
by sorry

end NUMINAMATH_CALUDE_li_point_parabola_range_l798_79878


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l798_79804

theorem solve_exponential_equation (x y z : ℕ) :
  (3 : ℝ)^x * (4 : ℝ)^y / (2 : ℝ)^z = 59049 ∧ x - y + 2*z = 10 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l798_79804


namespace NUMINAMATH_CALUDE_complex_square_simplify_l798_79811

theorem complex_square_simplify :
  (4 - 3 * Complex.I) ^ 2 = 7 - 24 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplify_l798_79811


namespace NUMINAMATH_CALUDE_polynomial_positive_intervals_l798_79861

/-- The polynomial (x+1)(x-1)(x-3) is positive if and only if x is in the interval (-1, 1) or (3, ∞) -/
theorem polynomial_positive_intervals (x : ℝ) : 
  (x + 1) * (x - 1) * (x - 3) > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_positive_intervals_l798_79861


namespace NUMINAMATH_CALUDE_problem_statement_l798_79837

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12)
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) :
  a + b^2 + c^3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l798_79837


namespace NUMINAMATH_CALUDE_exactly_one_absent_l798_79833

-- Define the three guests
variable (B K Z : Prop)

-- B: Baba Yaga comes to the festival
-- K: Koschei comes to the festival
-- Z: Zmey Gorynych comes to the festival

-- Define the conditions
axiom condition1 : ¬B → K
axiom condition2 : ¬K → Z
axiom condition3 : ¬Z → B
axiom at_least_one_absent : ¬B ∨ ¬K ∨ ¬Z

-- Theorem to prove
theorem exactly_one_absent : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_absent_l798_79833


namespace NUMINAMATH_CALUDE_parabola_trajectory_l798_79852

/-- The trajectory of point M given the conditions of the parabola and vector relationship -/
theorem parabola_trajectory (x y t : ℝ) : 
  let F : ℝ × ℝ := (1, 0)  -- Focus of the parabola
  let P : ℝ → ℝ × ℝ := λ t => (t^2/4, t)  -- Point on the parabola
  let M : ℝ × ℝ := (x, y)  -- Point M
  (∀ t, (P t).2^2 = 4 * (P t).1) →  -- P is on the parabola y^2 = 4x
  ((P t).1 - F.1, (P t).2 - F.2) = (2*(x - F.1), 2*(y - F.2)) →  -- FP = 2FM
  y^2 = 2*x - 1  -- Trajectory equation
:= by sorry

end NUMINAMATH_CALUDE_parabola_trajectory_l798_79852


namespace NUMINAMATH_CALUDE_remaining_lives_l798_79856

def initial_lives : ℕ := 98
def lives_lost : ℕ := 25

theorem remaining_lives : initial_lives - lives_lost = 73 := by
  sorry

end NUMINAMATH_CALUDE_remaining_lives_l798_79856


namespace NUMINAMATH_CALUDE_no_real_solutions_count_l798_79876

theorem no_real_solutions_count : 
  ∀ b c : ℕ+, 
  (∃ x : ℝ, x^2 + (b:ℝ)*x + (c:ℝ) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c:ℝ)*x + (b:ℝ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_count_l798_79876


namespace NUMINAMATH_CALUDE_merry_go_round_cost_per_child_l798_79887

/-- The cost of a merry-go-round ride per child given the following conditions:
  - There are 5 children
  - 3 children rode the Ferris wheel
  - Ferris wheel cost is $5 per child
  - Everyone rode the merry-go-round
  - Each child bought 2 ice cream cones
  - Each ice cream cone costs $8
  - Total spent is $110
-/
theorem merry_go_round_cost_per_child 
  (num_children : ℕ)
  (ferris_wheel_riders : ℕ)
  (ferris_wheel_cost : ℚ)
  (ice_cream_cones_per_child : ℕ)
  (ice_cream_cone_cost : ℚ)
  (total_spent : ℚ)
  (h1 : num_children = 5)
  (h2 : ferris_wheel_riders = 3)
  (h3 : ferris_wheel_cost = 5)
  (h4 : ice_cream_cones_per_child = 2)
  (h5 : ice_cream_cone_cost = 8)
  (h6 : total_spent = 110) :
  (total_spent - (ferris_wheel_riders * ferris_wheel_cost) - (num_children * ice_cream_cones_per_child * ice_cream_cone_cost)) / num_children = 3 :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_cost_per_child_l798_79887


namespace NUMINAMATH_CALUDE_duck_count_l798_79846

theorem duck_count (total_animals : ℕ) (total_legs : ℕ) (duck_legs : ℕ) (horse_legs : ℕ) 
  (h1 : total_animals = 11)
  (h2 : total_legs = 30)
  (h3 : duck_legs = 2)
  (h4 : horse_legs = 4) :
  ∃ (ducks horses : ℕ),
    ducks + horses = total_animals ∧
    ducks * duck_legs + horses * horse_legs = total_legs ∧
    ducks = 7 :=
by sorry

end NUMINAMATH_CALUDE_duck_count_l798_79846


namespace NUMINAMATH_CALUDE_is_circle_center_l798_79814

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y - 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l798_79814


namespace NUMINAMATH_CALUDE_razor_blade_profit_equation_l798_79831

theorem razor_blade_profit_equation (x : ℝ) :
  (x ≥ 0) →                          -- number of razors sold is non-negative
  (30 : ℝ) * x +                     -- profit from razors
  (-0.5 : ℝ) * (2 * x) =             -- loss from blades (twice the number of razors)
  (5800 : ℝ)                         -- total profit
  := by sorry

end NUMINAMATH_CALUDE_razor_blade_profit_equation_l798_79831


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l798_79824

/-- Proposition p: For any x ∈ ℝ, x² - 2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: There exists x ∈ ℝ such that x² + 2ax + 2 - a = 0 -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := { a : ℝ | (a > -2 ∧ a < -1) ∨ a ≥ 1 }

theorem range_of_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l798_79824


namespace NUMINAMATH_CALUDE_arrow_symmetry_axis_l798_79899

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a geometric figure on a grid --/
structure GeometricFigure where
  points : Set GridPoint

/-- Represents a line on a grid --/
inductive GridLine
  | Vertical : Int → GridLine
  | Horizontal : Int → GridLine
  | DiagonalTopLeftToBottomRight : GridLine
  | DiagonalBottomLeftToTopRight : GridLine

/-- Predicate to check if a figure is arrow-shaped --/
def isArrowShaped (figure : GeometricFigure) : Prop := sorry

/-- Predicate to check if a line is an axis of symmetry for a figure --/
def isAxisOfSymmetry (line : GridLine) (figure : GeometricFigure) : Prop := sorry

/-- Theorem: An arrow-shaped figure with only one axis of symmetry has a vertical line through the center as its axis of symmetry --/
theorem arrow_symmetry_axis (figure : GeometricFigure) (h1 : isArrowShaped figure) 
    (h2 : ∃! (line : GridLine), isAxisOfSymmetry line figure) : 
    ∃ (x : Int), isAxisOfSymmetry (GridLine.Vertical x) figure := by
  sorry

end NUMINAMATH_CALUDE_arrow_symmetry_axis_l798_79899


namespace NUMINAMATH_CALUDE_walking_time_is_half_time_saved_l798_79844

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure HomeCommuteScenario where
  usual_arrival_time : ℕ  -- Time in minutes when they usually arrive home
  early_station_arrival : ℕ  -- Time in minutes the man arrives early at the station
  actual_arrival_time : ℕ  -- Time in minutes when they actually arrive home
  walking_time : ℕ  -- Time in minutes the man spends walking

/-- Theorem stating that the walking time is half of the time saved --/
theorem walking_time_is_half_time_saved (scenario : HomeCommuteScenario) 
  (h1 : scenario.early_station_arrival = 60)
  (h2 : scenario.usual_arrival_time - scenario.actual_arrival_time = 30) :
  scenario.walking_time = (scenario.usual_arrival_time - scenario.actual_arrival_time) / 2 := by
  sorry

#check walking_time_is_half_time_saved

end NUMINAMATH_CALUDE_walking_time_is_half_time_saved_l798_79844


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l798_79889

theorem fraction_sum_equality (n : ℕ) (hn : n > 1) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x ≤ y ∧ (1 : ℚ) / n = 1 / x - 1 / (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l798_79889
