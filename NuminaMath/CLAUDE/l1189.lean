import Mathlib

namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l1189_118926

/-- The minimum time required for blacksmiths to shoe horses -/
theorem minimum_shoeing_time 
  (num_blacksmiths : ℕ) 
  (num_horses : ℕ) 
  (time_per_horseshoe : ℕ) 
  (horseshoes_per_horse : ℕ) : 
  num_blacksmiths = 48 → 
  num_horses = 60 → 
  time_per_horseshoe = 5 → 
  horseshoes_per_horse = 4 → 
  (num_horses * horseshoes_per_horse * time_per_horseshoe) / num_blacksmiths = 25 := by
  sorry

end NUMINAMATH_CALUDE_minimum_shoeing_time_l1189_118926


namespace NUMINAMATH_CALUDE_line_intercept_sum_l1189_118972

/-- Given a line with equation 2x - 5y + 10 = 0, prove that the absolute value of the sum of its x and y intercepts is 3. -/
theorem line_intercept_sum (a b : ℝ) : 
  (2 * a - 5 * 0 + 10 = 0) →  -- x-intercept condition
  (2 * 0 - 5 * b + 10 = 0) →  -- y-intercept condition
  |a + b| = 3 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l1189_118972


namespace NUMINAMATH_CALUDE_quadratic_real_root_range_l1189_118962

theorem quadratic_real_root_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 4 = 0) ∨ 
  (∃ x : ℝ, x^2 + (a-2)*x + 4 = 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x + a^2 + 1 = 0) ↔ 
  a ≥ 4 ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_range_l1189_118962


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1189_118940

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 8) :
  a / c = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1189_118940


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1189_118983

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n - 2) * 180 : ℕ) / n = 144) → n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1189_118983


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1189_118986

theorem fraction_sum_equality : 
  1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 9 / 20 = -9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1189_118986


namespace NUMINAMATH_CALUDE_product_of_1011_2_and_102_3_l1189_118906

def base_2_to_10 (n : ℕ) : ℕ := sorry

def base_3_to_10 (n : ℕ) : ℕ := sorry

theorem product_of_1011_2_and_102_3 : 
  (base_2_to_10 1011) * (base_3_to_10 102) = 121 := by sorry

end NUMINAMATH_CALUDE_product_of_1011_2_and_102_3_l1189_118906


namespace NUMINAMATH_CALUDE_classroom_gpa_proof_l1189_118998

theorem classroom_gpa_proof (x : ℝ) : 
  (1/3 : ℝ) * x + (2/3 : ℝ) * 66 = 64 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_classroom_gpa_proof_l1189_118998


namespace NUMINAMATH_CALUDE_experimental_primary_school_students_l1189_118964

/-- The total number of students in Experimental Primary School -/
def total_students (num_classes : ℕ) (boys_in_class1 : ℕ) (girls_in_class1 : ℕ) : ℕ :=
  num_classes * (boys_in_class1 + girls_in_class1)

/-- Theorem: The total number of students in Experimental Primary School is 896 -/
theorem experimental_primary_school_students :
  total_students 28 21 11 = 896 := by
  sorry

end NUMINAMATH_CALUDE_experimental_primary_school_students_l1189_118964


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1189_118993

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = (2 + Real.sqrt 14) / 2 ∧ 
  x₂ = (2 - Real.sqrt 14) / 2 ∧ 
  2 * x₁^2 - 4 * x₁ - 5 = 0 ∧ 
  2 * x₂^2 - 4 * x₂ - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1189_118993


namespace NUMINAMATH_CALUDE_inequality_proof_l1189_118956

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  a * b > a * c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1189_118956


namespace NUMINAMATH_CALUDE_root_between_roots_l1189_118952

theorem root_between_roots (a b : ℝ) (α β : ℝ) 
  (hα : α^2 + a*α + b = 0) 
  (hβ : β^2 - a*β - b = 0) : 
  ∃ x, x ∈ Set.Icc α β ∧ x^2 - 2*a*x - 2*b = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_between_roots_l1189_118952


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1189_118927

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width +  -- bottom area
  2 * length * depth +  -- longer sides area
  2 * width * depth  -- shorter sides area

/-- Theorem: The wet surface area of a 7m x 5m cistern with 1.40m water depth is 68.6 m² -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 7 5 1.40 = 68.6 := by
  sorry

#eval wetSurfaceArea 7 5 1.40

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1189_118927


namespace NUMINAMATH_CALUDE_gcd_g_x_l1189_118997

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(11*x+7)*(4*x+11)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 17248 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l1189_118997


namespace NUMINAMATH_CALUDE_complex_product_imaginary_l1189_118915

theorem complex_product_imaginary (a : ℝ) : 
  (Complex.I * (1 + a * Complex.I) + (2 : ℂ) * (1 + a * Complex.I)).re = 0 → a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_imaginary_l1189_118915


namespace NUMINAMATH_CALUDE_annie_hamburger_cost_l1189_118970

/-- Calculates the cost of a single hamburger given the initial amount,
    cost of a milkshake, number of hamburgers and milkshakes bought,
    and the remaining amount after purchase. -/
def hamburger_cost (initial_amount : ℕ) (milkshake_cost : ℕ) 
                   (hamburgers_bought : ℕ) (milkshakes_bought : ℕ) 
                   (remaining_amount : ℕ) : ℕ :=
  (initial_amount - remaining_amount - milkshake_cost * milkshakes_bought) / hamburgers_bought

/-- Theorem stating that given Annie's purchases and finances, 
    each hamburger costs $4. -/
theorem annie_hamburger_cost : 
  hamburger_cost 132 5 8 6 70 = 4 := by
  sorry

end NUMINAMATH_CALUDE_annie_hamburger_cost_l1189_118970


namespace NUMINAMATH_CALUDE_expected_yield_for_80kg_fertilizer_l1189_118949

/-- Represents the regression line equation for rice yield based on fertilizer amount -/
def regression_line (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem stating that the expected rice yield is 650 kg when 80 kg of fertilizer is applied -/
theorem expected_yield_for_80kg_fertilizer : 
  regression_line 80 = 650 := by sorry

end NUMINAMATH_CALUDE_expected_yield_for_80kg_fertilizer_l1189_118949


namespace NUMINAMATH_CALUDE_distance_implies_abs_x_l1189_118946

theorem distance_implies_abs_x (x : ℝ) :
  |((3 + x) - (3 - x))| = 8 → |x| = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_implies_abs_x_l1189_118946


namespace NUMINAMATH_CALUDE_torn_page_numbers_l1189_118968

theorem torn_page_numbers (n : ℕ) (k : ℕ) : 
  n > 0 ∧ k > 1 ∧ k < n ∧ (n * (n + 1)) / 2 - (2 * k - 1) = 15000 → k = 113 := by
  sorry

end NUMINAMATH_CALUDE_torn_page_numbers_l1189_118968


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1189_118996

theorem isosceles_triangle (A B C : Real) (h1 : A + B + C = π) (h2 : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1189_118996


namespace NUMINAMATH_CALUDE_trailing_zeros_80_factorial_l1189_118982

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

theorem trailing_zeros_80_factorial :
  trailingZeros 73 = 16 → trailingZeros 80 = 18 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_80_factorial_l1189_118982


namespace NUMINAMATH_CALUDE_equation_solution_l1189_118987

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 54 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1189_118987


namespace NUMINAMATH_CALUDE_congruence_solution_and_sum_l1189_118929

theorem congruence_solution_and_sum (x : ℤ) : 
  (15 * x + 3) % 21 = 9 → 
  ∃ (a m : ℤ), x % m = a ∧ 
                a < m ∧ 
                m = 7 ∧ 
                a = 6 ∧ 
                a + m = 13 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_and_sum_l1189_118929


namespace NUMINAMATH_CALUDE_cosine_ratio_equals_negative_sqrt_three_l1189_118948

theorem cosine_ratio_equals_negative_sqrt_three : 
  (2 * Real.cos (80 * π / 180) + Real.cos (160 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_equals_negative_sqrt_three_l1189_118948


namespace NUMINAMATH_CALUDE_modulo_seventeen_residue_l1189_118934

theorem modulo_seventeen_residue : (352 + 6 * 68 + 8 * 221 + 3 * 34 + 5 * 17) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_seventeen_residue_l1189_118934


namespace NUMINAMATH_CALUDE_sum_of_first_53_odd_numbers_l1189_118947

theorem sum_of_first_53_odd_numbers : 
  (Finset.range 53).sum (fun n => 2 * n + 1) = 2809 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_53_odd_numbers_l1189_118947


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1189_118931

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define that ABC is a right-angled triangle
def IsRightAngled (A B C : ℝ × ℝ) := True

-- Define AD as an angle bisector
def IsAngleBisector (A B C D : ℝ × ℝ) := True

-- Define the lengths of the sides
def SideLength (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (A B C D : ℝ × ℝ) (x : ℝ) :
  Triangle A B C →
  IsRightAngled A B C →
  IsAngleBisector A B C D →
  SideLength A B = 100 →
  SideLength B C = x →
  SideLength A C = x + 10 →
  Int.floor (TriangleArea A D C + 0.5) = 20907 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1189_118931


namespace NUMINAMATH_CALUDE_line_through_C_parallel_to_AB_area_of_triangle_OMN_l1189_118919

-- Define the points
def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (1, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := x - y + 1 = 0

-- Define points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (0, 1)

-- Theorem for the line equation
theorem line_through_C_parallel_to_AB :
  line_equation C.1 C.2 ∧
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1) := by sorry

-- Theorem for the area of triangle OMN
theorem area_of_triangle_OMN :
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
  M.2 = 0 ∧ N.1 = 0 ∧
  (1 / 2 : ℝ) * abs M.1 * abs N.2 = (1 / 2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_line_through_C_parallel_to_AB_area_of_triangle_OMN_l1189_118919


namespace NUMINAMATH_CALUDE_custom_operation_equality_l1189_118921

/-- The custom operation ⊗ -/
def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y z : ℝ) : 
  otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x^2 + 2*x*z - y^2 - 2*z*y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l1189_118921


namespace NUMINAMATH_CALUDE_pie_division_l1189_118928

theorem pie_division (initial_pie : ℚ) (scrooge_share : ℚ) (num_friends : ℕ) : 
  initial_pie = 4/5 → scrooge_share = 1/5 → num_friends = 3 →
  (initial_pie - scrooge_share * initial_pie) / num_friends = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_pie_division_l1189_118928


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1189_118999

theorem least_addition_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (821562 + k) % 5 = 0) → 
  (∃ m : ℕ, m > 0 ∧ (821562 + m) % 5 = 0 ∧ ∀ l : ℕ, l > 0 → (821562 + l) % 5 = 0 → m ≤ l) →
  (821562 + 3) % 5 = 0 ∧ 
  ∀ k : ℕ, k > 0 → (821562 + k) % 5 = 0 → 3 ≤ k :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1189_118999


namespace NUMINAMATH_CALUDE_even_student_schools_count_l1189_118950

/-- Represents a school with its student count -/
structure School where
  name : String
  students : ℕ

/-- Checks if a number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Counts the number of schools with an even number of students -/
def countEvenStudentSchools (schools : List School) : ℕ :=
  (schools.filter (fun s => isEven s.students)).length

/-- The main theorem -/
theorem even_student_schools_count :
  let schools : List School := [
    ⟨"A", 786⟩,
    ⟨"B", 777⟩,
    ⟨"C", 762⟩,
    ⟨"D", 819⟩,
    ⟨"E", 493⟩
  ]
  countEvenStudentSchools schools = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_student_schools_count_l1189_118950


namespace NUMINAMATH_CALUDE_tourist_tax_theorem_l1189_118942

/-- Calculates the tax paid given the total value of goods -/
def tax_paid (total_value : ℝ) : ℝ :=
  0.08 * (total_value - 600)

/-- Theorem stating that if $89.6 tax is paid, the total value of goods is $1720 -/
theorem tourist_tax_theorem (total_value : ℝ) :
  tax_paid total_value = 89.6 → total_value = 1720 := by
  sorry

end NUMINAMATH_CALUDE_tourist_tax_theorem_l1189_118942


namespace NUMINAMATH_CALUDE_series_sum_equals_257_l1189_118951

def series_sum : ℕ := by
  -- Define the ranges for n, m, and p
  let n_range := Finset.range 12
  let m_range := Finset.range 3
  let p_range := Finset.range 2

  -- Define the summation functions
  let f_n (n : ℕ) := 2 * (n + 1)
  let f_m (m : ℕ) := 3 * (2 * m + 3)
  let f_p (p : ℕ) := 4 * (4 * p + 2)

  -- Calculate the sum
  exact (n_range.sum f_n) + (m_range.sum f_m) + (p_range.sum f_p)

theorem series_sum_equals_257 : series_sum = 257 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_257_l1189_118951


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1189_118960

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define point P on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  hyperbola a b 2 3

-- Define the condition for slope of MA being 1 and MF = AF
def slope_and_distance_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola a b x y ∧ (y - (-a)) / (x - (-a)) = 1 ∧
  (x - 2*a)^2 + y^2 = (3*a)^2

-- Define the perpendicularity condition
def perpendicular_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  point_on_hyperbola a b →
  slope_and_distance_condition a b →
  perpendicular_condition a b →
  (a = 1 ∧ b = Real.sqrt 3) ∧
  (∀ (k t : ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola 1 (Real.sqrt 3) x₁ y₁ ∧
    hyperbola 1 (Real.sqrt 3) x₂ y₂ ∧
    y₁ = k * x₁ + t ∧
    y₂ = k * x₂ + t) →
  |t| / Real.sqrt (1 + k^2) = Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1189_118960


namespace NUMINAMATH_CALUDE_power_equality_implies_x_equals_two_l1189_118925

theorem power_equality_implies_x_equals_two :
  ∀ x : ℝ, (2 : ℝ)^10 = 32^x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_x_equals_two_l1189_118925


namespace NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l1189_118954

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3}

theorem complement_of_B_relative_to_A : A \ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l1189_118954


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l1189_118990

/-- Given two employees with a total pay of 570 and one paid 150% of the other,
    prove that the lower-paid employee receives 228. -/
theorem employee_pay_calculation (total_pay : ℝ) (ratio : ℝ) :
  total_pay = 570 →
  ratio = 1.5 →
  ∃ (low_pay : ℝ), low_pay * (1 + ratio) = total_pay ∧ low_pay = 228 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l1189_118990


namespace NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_proof_l1189_118963

/-- The perimeter of a rectangle with width 16 and length 19 is 70 -/
theorem rectangle_perimeter : ℕ → ℕ → ℕ
  | 16, 19 => 70
  | _, _ => 0  -- Default case for other inputs

/-- The perimeter of a rectangle is twice the sum of its length and width -/
def perimeter (width length : ℕ) : ℕ := 2 * (width + length)

theorem rectangle_perimeter_proof (width length : ℕ) (h1 : width = 16) (h2 : length = 19) :
  perimeter width length = rectangle_perimeter width length := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_proof_l1189_118963


namespace NUMINAMATH_CALUDE_f_increasing_iff_l1189_118980

/-- A piecewise function f defined on ℝ --/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 1 then a^x else (3-a)*x + 1

/-- Theorem stating the condition for f to be increasing --/
theorem f_increasing_iff (a : ℝ) :
  StrictMono (f a) ↔ 2 ≤ a ∧ a < 3 :=
sorry

#check f_increasing_iff

end NUMINAMATH_CALUDE_f_increasing_iff_l1189_118980


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l1189_118922

theorem number_puzzle_solution : ∃ x : ℝ, 12 * x = x + 198 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l1189_118922


namespace NUMINAMATH_CALUDE_johns_friends_l1189_118918

theorem johns_friends (num_pizzas : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) :
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  slices_per_person = 4 →
  (num_pizzas * slices_per_pizza) / slices_per_person - 1 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_friends_l1189_118918


namespace NUMINAMATH_CALUDE_modulus_of_one_minus_i_l1189_118920

theorem modulus_of_one_minus_i :
  let z : ℂ := 1 - I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_minus_i_l1189_118920


namespace NUMINAMATH_CALUDE_find_y_value_l1189_118992

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1189_118992


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1189_118966

/-- The polynomial P(x) -/
def P (a : ℝ) (x : ℝ) : ℝ := x^1000 + a*x^2 + 9

/-- Theorem: P(x) is divisible by (x + 1) iff a = -10 -/
theorem polynomial_divisibility (a : ℝ) : 
  (∃ q : ℝ → ℝ, ∀ x, P a x = (x + 1) * q x) ↔ a = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1189_118966


namespace NUMINAMATH_CALUDE_ten_by_ten_grid_triangles_l1189_118943

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)

/-- Counts the number of triangles formed by drawing a diagonal in a square grid -/
def countTriangles (grid : SquareGrid) : ℕ :=
  (grid.size + 1) * (grid.size + 1) - (grid.size + 1)

/-- Theorem: In a 10 × 10 square grid with one diagonal drawn, 110 triangles are formed -/
theorem ten_by_ten_grid_triangles :
  countTriangles { size := 10 } = 110 := by
  sorry

end NUMINAMATH_CALUDE_ten_by_ten_grid_triangles_l1189_118943


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l1189_118903

/-- Represents the length of an edge in the pyramid --/
inductive EdgeLength
| ten : EdgeLength
| twenty : EdgeLength
| twentyFive : EdgeLength

/-- Represents a triangular face of the pyramid --/
structure TriangularFace where
  edge1 : EdgeLength
  edge2 : EdgeLength
  edge3 : EdgeLength

/-- Represents the pyramid WXYZ --/
structure Pyramid where
  faces : List TriangularFace
  edge_length_condition : ∀ f ∈ faces, f.edge1 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive] ∧
                                       f.edge2 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive] ∧
                                       f.edge3 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive]
  not_equilateral : ∀ f ∈ faces, f.edge1 ≠ f.edge2 ∨ f.edge1 ≠ f.edge3 ∨ f.edge2 ≠ f.edge3

/-- The surface area of a pyramid --/
def surfaceArea (p : Pyramid) : ℝ := sorry

/-- Theorem stating the surface area of the pyramid WXYZ --/
theorem pyramid_surface_area (p : Pyramid) : surfaceArea p = 100 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l1189_118903


namespace NUMINAMATH_CALUDE_quarter_capacity_at_6_l1189_118901

/-- Represents the volume of water in the pool as a fraction of its full capacity -/
def PoolVolume := Fin 9 → ℚ

/-- The pool's volume doubles every hour -/
def doubles (v : PoolVolume) : Prop :=
  ∀ i : Fin 8, v (i + 1) = 2 * v i

/-- The pool is full after 8 hours -/
def full_at_8 (v : PoolVolume) : Prop :=
  v 8 = 1

/-- The main theorem: If the pool's volume doubles every hour and is full after 8 hours,
    then it was at one quarter capacity after 6 hours -/
theorem quarter_capacity_at_6 (v : PoolVolume) 
  (h1 : doubles v) (h2 : full_at_8 v) : v 6 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quarter_capacity_at_6_l1189_118901


namespace NUMINAMATH_CALUDE_bike_shop_profit_l1189_118909

/-- The cost of parts for fixing a single bike tire -/
def tire_part_cost : ℝ := 5

theorem bike_shop_profit (tire_repair_price : ℝ) (tire_repairs : ℕ) 
  (complex_repair_price : ℝ) (complex_repair_cost : ℝ) (complex_repairs : ℕ)
  (retail_profit : ℝ) (fixed_expenses : ℝ) (total_profit : ℝ) :
  tire_repair_price = 20 →
  tire_repairs = 300 →
  complex_repair_price = 300 →
  complex_repair_cost = 50 →
  complex_repairs = 2 →
  retail_profit = 2000 →
  fixed_expenses = 4000 →
  total_profit = 3000 →
  tire_part_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_bike_shop_profit_l1189_118909


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1189_118932

theorem cubic_roots_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 4 * r^2 + 1500 * r + 3000 = 0) →
  (6 * s^3 + 4 * s^2 + 1500 * s + 3000 = 0) →
  (6 * t^3 + 4 * t^2 + 1500 * t + 3000 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992/27 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1189_118932


namespace NUMINAMATH_CALUDE_marks_speed_l1189_118933

/-- Given a distance of 24 miles and a time of 4 hours, prove that the speed is 6 miles per hour. -/
theorem marks_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 24 ∧ time = 4 ∧ speed = distance / time → speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_marks_speed_l1189_118933


namespace NUMINAMATH_CALUDE_vectors_collinear_l1189_118961

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (2, 4)

theorem vectors_collinear : ∃ k : ℝ, k • a = b + c := by sorry

end NUMINAMATH_CALUDE_vectors_collinear_l1189_118961


namespace NUMINAMATH_CALUDE_prop_1_prop_3_prop_4_l1189_118957

open Real

-- Define the second quadrant
def second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

-- Proposition 1
theorem prop_1 (θ : ℝ) (h : second_quadrant θ) : sin θ * tan θ < 0 := by
  sorry

-- Proposition 3
theorem prop_3 : sin 1 * cos 2 * tan 3 > 0 := by
  sorry

-- Proposition 4
theorem prop_4 (θ : ℝ) (h : 3*π/2 < θ ∧ θ < 2*π) : sin (π + θ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_prop_1_prop_3_prop_4_l1189_118957


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l1189_118981

theorem sin_two_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * (Real.cos α)^2 = Real.sin (π/4 - α)) :
  Real.sin (2*α) = 1 ∨ Real.sin (2*α) = -17/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l1189_118981


namespace NUMINAMATH_CALUDE_jimmy_cards_ratio_l1189_118994

def jimmy_cards_problem (initial_cards : ℕ) (cards_to_bob : ℕ) (cards_left : ℕ) : Prop :=
  let cards_to_mary := initial_cards - cards_left - cards_to_bob
  (cards_to_mary : ℚ) / cards_to_bob = 2 / 1

theorem jimmy_cards_ratio : jimmy_cards_problem 18 3 9 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_cards_ratio_l1189_118994


namespace NUMINAMATH_CALUDE_blake_change_l1189_118941

-- Define the quantities and prices
def num_lollipops : ℕ := 4
def num_chocolate_packs : ℕ := 6
def lollipop_price : ℕ := 2
def num_bills : ℕ := 6
def bill_value : ℕ := 10

-- Define the relationship between chocolate and lollipop prices
def chocolate_pack_price : ℕ := 4 * lollipop_price

-- Calculate the total cost
def total_cost : ℕ := num_lollipops * lollipop_price + num_chocolate_packs * chocolate_pack_price

-- Calculate the amount given
def amount_given : ℕ := num_bills * bill_value

-- Theorem to prove
theorem blake_change : amount_given - total_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_change_l1189_118941


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1189_118912

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 2/5, then the ratio of A's area to B's area is 4:25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 5) (h2 : b / d = 2 / 5) :
  (a * b) / (c * d) = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1189_118912


namespace NUMINAMATH_CALUDE_no_double_application_function_l1189_118976

theorem no_double_application_function :
  ¬∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l1189_118976


namespace NUMINAMATH_CALUDE_range_of_a_l1189_118904

-- Define the condition
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - a*x + 2*a > 0

-- State the theorem
theorem range_of_a : 
  {a : ℝ | always_positive a} = {a : ℝ | 0 < a ∧ a < 8} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1189_118904


namespace NUMINAMATH_CALUDE_imaginary_part_z2_l1189_118979

theorem imaginary_part_z2 (z₁ z₂ : ℂ) : 
  z₁ = 2 - 3*I → z₁ * z₂ = 1 + 2*I → z₂.im = 7/13 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_z2_l1189_118979


namespace NUMINAMATH_CALUDE_second_person_work_time_l1189_118984

/-- Given two persons who can finish a job in 8 days, where the first person alone can finish the job in 24 days, prove that the second person alone will take 12 days to finish the job. -/
theorem second_person_work_time (total_time : ℝ) (first_person_time : ℝ) (second_person_time : ℝ) : 
  total_time = 8 → first_person_time = 24 → second_person_time = 12 := by
  sorry

#check second_person_work_time

end NUMINAMATH_CALUDE_second_person_work_time_l1189_118984


namespace NUMINAMATH_CALUDE_carpet_inner_length_is_two_l1189_118905

/-- Represents a rectangular carpet with three concentric colored regions -/
structure Carpet where
  inner_length : ℝ
  inner_width : ℝ
  region_width : ℝ

/-- Calculates the area of the inner region -/
def inner_area (c : Carpet) : ℝ :=
  c.inner_length * c.inner_width

/-- Calculates the area of the middle region -/
def middle_area (c : Carpet) : ℝ :=
  (c.inner_length + 2 * c.region_width) * (c.inner_width + 2 * c.region_width) - c.inner_length * c.inner_width

/-- Calculates the area of the outer region -/
def outer_area (c : Carpet) : ℝ :=
  (c.inner_length + 4 * c.region_width) * (c.inner_width + 4 * c.region_width) - 
  (c.inner_length + 2 * c.region_width) * (c.inner_width + 2 * c.region_width)

/-- Checks if the areas form an arithmetic progression -/
def areas_in_arithmetic_progression (c : Carpet) : Prop :=
  2 * middle_area c = inner_area c + outer_area c

theorem carpet_inner_length_is_two (c : Carpet) 
  (h1 : c.inner_width = 1)
  (h2 : c.region_width = 1)
  (h3 : areas_in_arithmetic_progression c) :
  c.inner_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_inner_length_is_two_l1189_118905


namespace NUMINAMATH_CALUDE_rest_albums_count_l1189_118911

def total_pictures : ℕ := 25
def first_album_pictures : ℕ := 10
def pictures_per_remaining_album : ℕ := 3

theorem rest_albums_count : 
  (total_pictures - first_album_pictures) / pictures_per_remaining_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_rest_albums_count_l1189_118911


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1189_118917

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x - 2 < 0}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1189_118917


namespace NUMINAMATH_CALUDE_total_clients_l1189_118953

/-- Represents the number of clients needing vegan meals -/
def vegan : ℕ := 7

/-- Represents the number of clients needing kosher meals -/
def kosher : ℕ := 8

/-- Represents the number of clients needing both vegan and kosher meals -/
def both : ℕ := 3

/-- Represents the number of clients needing neither vegan nor kosher meals -/
def neither : ℕ := 18

/-- Theorem stating that the total number of clients is 30 -/
theorem total_clients : vegan + kosher - both + neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_clients_l1189_118953


namespace NUMINAMATH_CALUDE_suraya_caleb_difference_l1189_118959

/-- The number of apples picked by Kayla -/
def kayla_apples : ℕ := 20

/-- The number of apples picked by Caleb -/
def caleb_apples : ℕ := kayla_apples - 5

/-- The number of apples picked by Suraya -/
def suraya_apples : ℕ := kayla_apples + 7

/-- Theorem stating the difference between Suraya's and Caleb's apple count -/
theorem suraya_caleb_difference : suraya_apples - caleb_apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_suraya_caleb_difference_l1189_118959


namespace NUMINAMATH_CALUDE_angle_rewrite_and_terminal_sides_l1189_118902

theorem angle_rewrite_and_terminal_sides (α : Real) (h : α = 1200 * π / 180) :
  ∃ (β k : Real),
    α = β + 2 * k * π ∧
    0 ≤ β ∧ β < 2 * π ∧
    β = 2 * π / 3 ∧
    k = 3 ∧
    (2 * π / 3 ∈ Set.Icc (-2 * π) (2 * π)) ∧
    (-4 * π / 3 ∈ Set.Icc (-2 * π) (2 * π)) ∧
    ∃ (m n : ℤ),
      2 * π / 3 = α + 2 * m * π ∧
      -4 * π / 3 = α + 2 * n * π :=
by sorry

end NUMINAMATH_CALUDE_angle_rewrite_and_terminal_sides_l1189_118902


namespace NUMINAMATH_CALUDE_pudding_distribution_l1189_118989

theorem pudding_distribution (total_cups : ℕ) (additional_cups : ℕ) 
  (h1 : total_cups = 315)
  (h2 : additional_cups = 121)
  (h3 : ∀ (students : ℕ), students > 0 → 
    (total_cups + additional_cups) % students = 0 → 
    total_cups < students * ((total_cups + additional_cups) / students)) :
  ∃ (students : ℕ), students = 4 ∧ 
    (total_cups + additional_cups) % students = 0 ∧
    total_cups < students * ((total_cups + additional_cups) / students) :=
by sorry

end NUMINAMATH_CALUDE_pudding_distribution_l1189_118989


namespace NUMINAMATH_CALUDE_mask_production_optimization_l1189_118944

/-- Represents the production and profit parameters for a mask factory --/
structure MaskFactory where
  total_days : ℕ
  total_masks : ℕ
  min_type_a : ℕ
  daily_type_a : ℕ
  daily_type_b : ℕ
  profit_type_a : ℚ
  profit_type_b : ℚ

/-- The main theorem about mask production and profit optimization --/
theorem mask_production_optimization (f : MaskFactory) 
  (h_total_days : f.total_days = 8)
  (h_total_masks : f.total_masks = 50000)
  (h_min_type_a : f.min_type_a = 18000)
  (h_daily_type_a : f.daily_type_a = 6000)
  (h_daily_type_b : f.daily_type_b = 8000)
  (h_profit_type_a : f.profit_type_a = 1/2)
  (h_profit_type_b : f.profit_type_b = 3/10) :
  ∃ (profit_function : ℚ → ℚ) (x_range : Set ℚ) (max_profit : ℚ) (min_time : ℕ),
    (∀ x, profit_function x = 0.2 * x + 1.5) ∧
    x_range = {x | 1.8 ≤ x ∧ x ≤ 4.2} ∧
    max_profit = 2.34 ∧
    min_time = 7 :=
by sorry

#check mask_production_optimization

end NUMINAMATH_CALUDE_mask_production_optimization_l1189_118944


namespace NUMINAMATH_CALUDE_birds_on_fence_l1189_118988

theorem birds_on_fence : 
  let initial_birds : ℕ := 12
  let additional_birds : ℕ := 8
  let num_groups : ℕ := 3
  let birds_per_group : ℕ := 6
  initial_birds + additional_birds + num_groups * birds_per_group = 38 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1189_118988


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l1189_118923

theorem right_triangle_squares_area (x : ℝ) :
  let triangle_area := (1/2) * (3*x) * (4*x)
  let square1_area := (3*x)^2
  let square2_area := (4*x)^2
  let total_area := triangle_area + square1_area + square2_area
  (total_area = 1000) → (x = 10 * Real.sqrt 31 / 31) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l1189_118923


namespace NUMINAMATH_CALUDE_remaining_debt_percentage_l1189_118937

def original_debt : ℝ := 500
def initial_payment : ℝ := 125

theorem remaining_debt_percentage :
  (original_debt - initial_payment) / original_debt * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_remaining_debt_percentage_l1189_118937


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1189_118939

theorem quadratic_rewrite (b : ℝ) (n : ℝ) : 
  b < 0 ∧ 
  (∀ x : ℝ, x^2 + b*x + (1/4 : ℝ) = (x + n)^2 + (1/18 : ℝ)) →
  b = -(Real.sqrt 7)/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1189_118939


namespace NUMINAMATH_CALUDE_train_length_l1189_118916

theorem train_length (bridge_length : ℝ) (total_time : ℝ) (on_bridge_time : ℝ) :
  bridge_length = 600 →
  total_time = 30 →
  on_bridge_time = 20 →
  (bridge_length + (bridge_length * on_bridge_time / total_time)) / (total_time - on_bridge_time) = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1189_118916


namespace NUMINAMATH_CALUDE_rectangles_in_3x2_grid_l1189_118973

/-- The number of rectangles in a grid -/
def count_rectangles (m n : ℕ) : ℕ :=
  let one_by_one := m * n
  let one_by_two := m * (n - 1)
  let two_by_one := (m - 1) * n
  let two_by_two := (m - 1) * (n - 1)
  one_by_one + one_by_two + two_by_one + two_by_two

/-- Theorem: The number of rectangles in a 3x2 grid is 14 -/
theorem rectangles_in_3x2_grid :
  count_rectangles 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_3x2_grid_l1189_118973


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1189_118930

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 4*x - 1 = 0) :
  2*x^4 + 8*x^3 - 4*x^2 - 8*x + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1189_118930


namespace NUMINAMATH_CALUDE_equation_solution_l1189_118908

theorem equation_solution : ∃ x : ℚ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1189_118908


namespace NUMINAMATH_CALUDE_expression_value_l1189_118978

theorem expression_value : (19 + 43 / 151) * 151 = 2910 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1189_118978


namespace NUMINAMATH_CALUDE_product_segment_doubles_when_unit_halved_l1189_118985

/-- Theorem: Product segment length doubles when unit segment is halved -/
theorem product_segment_doubles_when_unit_halved 
  (a b e d : ℝ) 
  (h1 : d = a * b / e) 
  (e' : ℝ) 
  (h2 : e' = e / 2) 
  (d' : ℝ) 
  (h3 : d' = a * b / e') : 
  d' = 2 * d := by
sorry

end NUMINAMATH_CALUDE_product_segment_doubles_when_unit_halved_l1189_118985


namespace NUMINAMATH_CALUDE_clothing_store_problem_l1189_118975

-- Define the types of clothing
inductive ClothingType
| A
| B

-- Define the structure for clothing information
structure ClothingInfo where
  purchasePrice : ClothingType → ℕ
  sellingPrice : ClothingType → ℕ
  totalQuantity : ℕ

-- Define the problem conditions
def problemConditions (info : ClothingInfo) : Prop :=
  info.totalQuantity = 100 ∧
  2 * info.purchasePrice ClothingType.A + info.purchasePrice ClothingType.B = 260 ∧
  info.purchasePrice ClothingType.A + 3 * info.purchasePrice ClothingType.B = 380 ∧
  info.sellingPrice ClothingType.A = 120 ∧
  info.sellingPrice ClothingType.B = 150

-- Define the profit calculation function
def calculateProfit (info : ClothingInfo) (quantityA quantityB : ℕ) : ℕ :=
  (info.sellingPrice ClothingType.A - info.purchasePrice ClothingType.A) * quantityA +
  (info.sellingPrice ClothingType.B - info.purchasePrice ClothingType.B) * quantityB

-- Theorem statement
theorem clothing_store_problem (info : ClothingInfo) :
  problemConditions info →
  (info.purchasePrice ClothingType.A = 80 ∧
   info.purchasePrice ClothingType.B = 100 ∧
   calculateProfit info 50 50 = 4500 ∧
   (∀ m : ℕ, m ≤ 33 → (100 - m) ≥ 2 * m) ∧
   (∀ m : ℕ, m > 33 → (100 - m) < 2 * m) ∧
   calculateProfit info 67 33 = 4330) :=
by sorry


end NUMINAMATH_CALUDE_clothing_store_problem_l1189_118975


namespace NUMINAMATH_CALUDE_vector_magnitude_l1189_118967

def a : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (2, -4)

theorem vector_magnitude (x : ℝ) (b : ℝ × ℝ) 
  (h1 : b = (-1, x))
  (h2 : ∃ k : ℝ, b.1 = k * c.1 ∧ b.2 = k * c.2) :
  ‖a + b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1189_118967


namespace NUMINAMATH_CALUDE_power_multiplication_l1189_118995

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1189_118995


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1189_118955

theorem quadratic_root_problem (k : ℝ) :
  (∃ x : ℝ, x^2 + (k - 5) * x + (4 - k) = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 + (k - 5) * y + (4 - k) = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1189_118955


namespace NUMINAMATH_CALUDE_valid_table_iff_odd_l1189_118958

/-- A square table of size n × n -/
def SquareTable (n : ℕ) := Fin n → Fin n → ℚ

/-- The sum of numbers on a diagonal of a square table -/
def diagonalSum (table : SquareTable n) (d : ℕ) : ℚ :=
  sorry

/-- A square table is valid if the sum of numbers on each diagonal is 1 -/
def isValidTable (table : SquareTable n) : Prop :=
  ∀ d, d < 4*n - 2 → diagonalSum table d = 1

/-- There exists a valid square table of size n × n if and only if n is odd -/
theorem valid_table_iff_odd (n : ℕ) :
  (∃ (table : SquareTable n), isValidTable table) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_table_iff_odd_l1189_118958


namespace NUMINAMATH_CALUDE_tshirt_price_proof_l1189_118936

/-- The regular price of a T-shirt -/
def regular_price : ℝ := 14.50

/-- The cost of a discounted T-shirt -/
def discount_price : ℝ := 1

/-- The total number of T-shirts bought -/
def total_shirts : ℕ := 12

/-- The total cost of all T-shirts -/
def total_cost : ℝ := 120

/-- The number of T-shirts in a "lot" (2 regular + 1 discounted) -/
def lot_size : ℕ := 3

theorem tshirt_price_proof :
  regular_price * (2 * (total_shirts / lot_size)) + 
  discount_price * (total_shirts / lot_size) = total_cost :=
sorry

end NUMINAMATH_CALUDE_tshirt_price_proof_l1189_118936


namespace NUMINAMATH_CALUDE_inequality_proof_l1189_118910

theorem inequality_proof (a b m n p : ℝ) 
  (h1 : a > b) (h2 : m > n) (h3 : p > 0) : 
  n - a * p < m - b * p := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1189_118910


namespace NUMINAMATH_CALUDE_last_ball_is_red_l1189_118914

/-- Represents the color of a ball -/
inductive BallColor
  | Blue
  | Red
  | Green

/-- Represents the state of the bottle -/
structure BottleState where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents a single ball removal operation -/
inductive RemovalOperation
  | BlueGreen
  | RedGreen
  | TwoRed
  | Other

/-- Defines the initial state of the bottle -/
def initialState : BottleState :=
  { blue := 1001, red := 1000, green := 1000 }

/-- Applies a single removal operation to the bottle state -/
def applyOperation (state : BottleState) (op : RemovalOperation) : BottleState :=
  match op with
  | RemovalOperation.BlueGreen => { blue := state.blue - 1, red := state.red + 1, green := state.green - 1 }
  | RemovalOperation.RedGreen => { blue := state.blue, red := state.red, green := state.green - 1 }
  | RemovalOperation.TwoRed => { blue := state.blue + 2, red := state.red - 2, green := state.green }
  | RemovalOperation.Other => { blue := state.blue, red := state.red, green := state.green - 1 }

/-- Determines if the game has ended (only one ball left) -/
def isGameOver (state : BottleState) : Bool :=
  state.blue + state.red + state.green = 1

/-- Theorem: The last remaining ball is red -/
theorem last_ball_is_red :
  ∃ (operations : List RemovalOperation),
    let finalState := operations.foldl applyOperation initialState
    isGameOver finalState ∧ finalState.red = 1 :=
  sorry


end NUMINAMATH_CALUDE_last_ball_is_red_l1189_118914


namespace NUMINAMATH_CALUDE_death_rate_calculation_l1189_118907

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate (people per two seconds) -/
def birth_rate : ℕ := 7

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 259200

/-- Represents the death rate (people per two seconds) -/
def death_rate : ℕ := 1

theorem death_rate_calculation :
  (birth_rate - death_rate) * seconds_per_day / 2 = net_increase_per_day :=
sorry

end NUMINAMATH_CALUDE_death_rate_calculation_l1189_118907


namespace NUMINAMATH_CALUDE_hot_dog_discount_calculation_l1189_118900

theorem hot_dog_discount_calculation (num_hot_dogs : ℕ) (price_per_hot_dog : ℕ) (discount_rate : ℚ) :
  num_hot_dogs = 6 →
  price_per_hot_dog = 50 →
  discount_rate = 1/10 →
  (num_hot_dogs * price_per_hot_dog) * (1 - discount_rate) = 270 :=
by sorry

end NUMINAMATH_CALUDE_hot_dog_discount_calculation_l1189_118900


namespace NUMINAMATH_CALUDE_lace_cost_per_meter_l1189_118969

-- Define the lengths in centimeters
def cuff_length : ℝ := 50
def hem_length : ℝ := 300
def ruffle_length : ℝ := 20
def total_cost : ℝ := 36

-- Define the number of cuffs and ruffles
def num_cuffs : ℕ := 2
def num_ruffles : ℕ := 5

-- Define the conversion factor from cm to m
def cm_to_m : ℝ := 100

-- Theorem to prove
theorem lace_cost_per_meter :
  let total_length := num_cuffs * cuff_length + hem_length + (hem_length / 3) + num_ruffles * ruffle_length
  let total_length_m := total_length / cm_to_m
  total_cost / total_length_m = 6 := by
  sorry

end NUMINAMATH_CALUDE_lace_cost_per_meter_l1189_118969


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1189_118924

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_two : a + b + c + d = 2) :
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 
   1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 9 ∧
  ∃ (a' b' c' d' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    a' + b' + c' + d' = 2 ∧
    (1 / (a' + b') + 1 / (a' + c') + 1 / (a' + d') + 
     1 / (b' + c') + 1 / (b' + d') + 1 / (c' + d')) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1189_118924


namespace NUMINAMATH_CALUDE_rational_roots_count_l1189_118965

/-- The set of factors of a natural number -/
def factors (n : ℕ) : Finset ℤ :=
  sorry

/-- The set of possible rational roots for a polynomial with given leading coefficient and constant term -/
def possibleRationalRoots (leadingCoeff constTerm : ℤ) : Finset ℚ :=
  sorry

/-- Theorem stating that the number of different possible rational roots for the given polynomial form is 20 -/
theorem rational_roots_count :
  let leadingCoeff := 4
  let constTerm := 18
  (possibleRationalRoots leadingCoeff constTerm).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_rational_roots_count_l1189_118965


namespace NUMINAMATH_CALUDE_distance_from_speed_and_time_l1189_118938

/-- The distance between two points given average speed and travel time -/
theorem distance_from_speed_and_time 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 50) 
  (h2 : time = 15.8) : 
  speed * time = 790 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_speed_and_time_l1189_118938


namespace NUMINAMATH_CALUDE_lily_book_count_l1189_118945

/-- The number of books Lily read last month -/
def books_last_month : ℕ := 4

/-- The number of books Lily plans to read this month -/
def books_this_month : ℕ := 2 * books_last_month

/-- The total number of books Lily will read in two months -/
def total_books : ℕ := books_last_month + books_this_month

theorem lily_book_count : total_books = 12 := by
  sorry

end NUMINAMATH_CALUDE_lily_book_count_l1189_118945


namespace NUMINAMATH_CALUDE_total_hours_worked_l1189_118971

theorem total_hours_worked (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) :
  hours_per_day = 3 →
  days_worked = 5 →
  total_hours = hours_per_day * days_worked →
  total_hours = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_hours_worked_l1189_118971


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l1189_118935

/-- Given plane vectors a and b, if a + b is parallel to a - b, then the second component of b is -2√3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) :
  a = (1, -Real.sqrt 3) →
  b.1 = 2 →
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) →
  b.2 = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l1189_118935


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l1189_118991

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divisible_by_2005 :
  ∃ m : ℕ+, (3^100 * m.val + (3^100 - 1)) % 2005 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l1189_118991


namespace NUMINAMATH_CALUDE_average_of_numbers_l1189_118913

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125320.5 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1189_118913


namespace NUMINAMATH_CALUDE_corner_rectangles_area_sum_l1189_118974

/-- Given a square with side length 100 cm divided into 9 rectangles,
    where the central rectangle has dimensions 40 cm × 60 cm,
    the sum of the areas of the four corner rectangles is 2400 cm². -/
theorem corner_rectangles_area_sum (x y : ℝ) : x > 0 → y > 0 →
  x + 40 + (100 - x - 40) = 100 →
  y + 60 + (100 - y - 60) = 100 →
  x * y + (60 - x) * y + x * (40 - y) + (60 - x) * (40 - y) = 2400 := by
  sorry

#check corner_rectangles_area_sum

end NUMINAMATH_CALUDE_corner_rectangles_area_sum_l1189_118974


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1189_118977

/-- A polynomial of the form ax^2 + bx + c is a perfect square trinomial if there exists a real number r such that ax^2 + bx + c = (√a * x + r)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

/-- The main theorem: If x^2 - kx + 64 is a perfect square trinomial, then k = 16 or k = -16 -/
theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 64 → k = 16 ∨ k = -16 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1189_118977
