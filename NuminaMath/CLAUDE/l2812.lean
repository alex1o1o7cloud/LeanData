import Mathlib

namespace NUMINAMATH_CALUDE_ordering_abc_l2812_281201

theorem ordering_abc (a b c : ℝ) (ha : a = Real.log (11/10)) (hb : b = 1/10) (hc : c = 2/21) : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l2812_281201


namespace NUMINAMATH_CALUDE_tony_grocery_distance_l2812_281287

/-- Represents the distances Tony drives for his errands -/
structure TonyErrands where
  halfway_distance : ℕ
  haircut_distance : ℕ
  doctor_distance : ℕ

/-- Calculates the distance Tony drives to get groceries -/
def grocery_distance (e : TonyErrands) : ℕ :=
  2 * e.halfway_distance - (e.haircut_distance + e.doctor_distance)

/-- Theorem: Tony drives 10 miles to get groceries -/
theorem tony_grocery_distance :
  let e : TonyErrands := { halfway_distance := 15, haircut_distance := 15, doctor_distance := 5 }
  grocery_distance e = 10 := by sorry

end NUMINAMATH_CALUDE_tony_grocery_distance_l2812_281287


namespace NUMINAMATH_CALUDE_trihedral_acute_angles_l2812_281282

/-- A trihedral angle is an angle formed by three planes meeting at a point. -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  planeAngles : Fin 3 → ℝ
  /-- The three dihedral angles of the trihedral angle -/
  dihedralAngles : Fin 3 → ℝ

/-- A predicate to check if an angle is acute -/
def isAcute (angle : ℝ) : Prop := 0 < angle ∧ angle < Real.pi / 2

/-- The main theorem: if all dihedral angles of a trihedral angle are acute,
    then all its plane angles are also acute -/
theorem trihedral_acute_angles (t : TrihedralAngle) 
  (h : ∀ i : Fin 3, isAcute (t.dihedralAngles i)) :
  ∀ i : Fin 3, isAcute (t.planeAngles i) := by
  sorry

end NUMINAMATH_CALUDE_trihedral_acute_angles_l2812_281282


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l2812_281243

theorem sum_greater_than_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : 1/a + 1/b = 1) :
  a + b > 4 := by
sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l2812_281243


namespace NUMINAMATH_CALUDE_inequality_proof_l2812_281213

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2812_281213


namespace NUMINAMATH_CALUDE_equation_solutions_l2812_281207

theorem equation_solutions :
  (∃ x₁ x₂, 3 * (x₁ - 2)^2 = 27 ∧ 3 * (x₂ - 2)^2 = 27 ∧ x₁ = 5 ∧ x₂ = -1) ∧
  (∃ x, (x + 5)^3 + 27 = 0 ∧ x = -8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2812_281207


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2812_281250

def A : Set ℚ := {1, 2, 1/2}

def B : Set ℚ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2812_281250


namespace NUMINAMATH_CALUDE_base_number_proof_l2812_281218

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^18) 
  (h2 : n = 17) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_proof_l2812_281218


namespace NUMINAMATH_CALUDE_max_d_value_l2812_281272

/-- Represents a 6-digit number of the form 7d7,33e -/
def SixDigitNumber (d e : ℕ) : ℕ := 700000 + d * 10000 + 7000 + 330 + e

/-- Checks if a natural number is a digit (0-9) -/
def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

/-- The main theorem stating the maximum value of d -/
theorem max_d_value : 
  ∃ (d e : ℕ), isDigit d ∧ isDigit e ∧ 
  (SixDigitNumber d e) % 33 = 0 ∧
  ∀ (d' e' : ℕ), isDigit d' ∧ isDigit e' ∧ (SixDigitNumber d' e') % 33 = 0 → d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2812_281272


namespace NUMINAMATH_CALUDE_final_sum_after_fillings_l2812_281237

/-- Represents the state of the blackboard after each filling -/
structure BoardState :=
  (numbers : List Int)
  (sum : Int)

/-- Perform one filling operation on the board -/
def fill (state : BoardState) : BoardState :=
  sorry

/-- The initial state of the board -/
def initial_state : BoardState :=
  { numbers := [2, 0, 2, 3], sum := 7 }

/-- Theorem stating the final sum after 2023 fillings -/
theorem final_sum_after_fillings :
  (Nat.iterate fill 2023 initial_state).sum = 2030 :=
sorry

end NUMINAMATH_CALUDE_final_sum_after_fillings_l2812_281237


namespace NUMINAMATH_CALUDE_point_coordinates_theorem_l2812_281203

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- Predicate to check if a point is left of the y-axis -/
def isLeftOfYAxis (p : Point) : Prop := p.x < 0

theorem point_coordinates_theorem (B : Point) 
  (h1 : isLeftOfYAxis B)
  (h2 : distanceToXAxis B = 4)
  (h3 : distanceToYAxis B = 5) :
  (B.x = -5 ∧ B.y = 4) ∨ (B.x = -5 ∧ B.y = -4) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_theorem_l2812_281203


namespace NUMINAMATH_CALUDE_triangle_b_range_l2812_281204

open Real Set

-- Define the triangle and its properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions for the triangle
def TriangleConditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.A = π / 3

-- Define the condition for exactly one solution
def ExactlyOneSolution (t : Triangle) : Prop :=
  (t.a = t.b * Real.sin t.A) ∨ (t.a ≥ t.b ∧ t.a > t.b * Real.sin t.A)

-- Define the range of values for b
def BRange : Set ℝ := Ioc 0 (Real.sqrt 3) ∪ {2}

-- State the theorem
theorem triangle_b_range (t : Triangle) :
  TriangleConditions t → ExactlyOneSolution t → t.b ∈ BRange :=
by sorry

end NUMINAMATH_CALUDE_triangle_b_range_l2812_281204


namespace NUMINAMATH_CALUDE_root_equation_r_value_l2812_281262

theorem root_equation_r_value (a b m p r : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 2/b)^2 - p*(a + 2/b) + r = 0) → 
  ((b + 2/a)^2 - p*(b + 2/a) + r = 0) → 
  r = 32/3 := by sorry

end NUMINAMATH_CALUDE_root_equation_r_value_l2812_281262


namespace NUMINAMATH_CALUDE_max_value_of_g_l2812_281232

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 25 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2812_281232


namespace NUMINAMATH_CALUDE_tangent_line_at_one_minimum_value_of_f_l2812_281216

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + f 1 → (x - y - 1 = 0) :=
sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value_of_f :
  ∃ x, f x = -1 / Real.exp 1 ∧ ∀ y, f y ≥ -1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_minimum_value_of_f_l2812_281216


namespace NUMINAMATH_CALUDE_minervas_stamps_l2812_281265

/-- Given that Lizette has 813 stamps and 125 more stamps than Minerva,
    prove that Minerva has 688 stamps. -/
theorem minervas_stamps :
  let lizette_stamps : ℕ := 813
  let difference : ℕ := 125
  let minerva_stamps : ℕ := lizette_stamps - difference
  minerva_stamps = 688 := by
sorry

end NUMINAMATH_CALUDE_minervas_stamps_l2812_281265


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2812_281297

/-- The line l with equation x - y + √3 = 0 intersects the circle C with equation x² + (y - √2)² = 2 -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), (x - y + Real.sqrt 3 = 0) ∧ (x^2 + (y - Real.sqrt 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2812_281297


namespace NUMINAMATH_CALUDE_class_size_is_20_l2812_281239

/-- Represents the number of students in a class with specific age distributions. -/
def num_students : ℕ := by sorry

/-- The average age of all students in the class. -/
def average_age : ℝ := 20

/-- The average age of a group of 9 students. -/
def average_age_group1 : ℝ := 11

/-- The average age of a group of 10 students. -/
def average_age_group2 : ℝ := 24

/-- The age of the 20th student. -/
def age_20th_student : ℝ := 61

/-- Theorem stating that the number of students in the class is 20. -/
theorem class_size_is_20 : num_students = 20 := by sorry

end NUMINAMATH_CALUDE_class_size_is_20_l2812_281239


namespace NUMINAMATH_CALUDE_eleventh_term_ratio_l2812_281296

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDiff : ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.firstTerm + (n - 1) * seq.commonDiff) / 2

/-- nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.firstTerm + (n - 1) * seq.commonDiff

theorem eleventh_term_ratio
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, sumOfTerms seq1 n / sumOfTerms seq2 n = (7 * n + 1) / (4 * n + 27)) :
  nthTerm seq1 11 / nthTerm seq2 11 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_ratio_l2812_281296


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2812_281217

theorem cosine_sine_identity : Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2812_281217


namespace NUMINAMATH_CALUDE_number_problem_l2812_281277

theorem number_problem (n : ℝ) (h : (1/3) * (1/4) * n = 18) : (3/10) * n = 64.8 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2812_281277


namespace NUMINAMATH_CALUDE_expression_not_equal_77_l2812_281281

theorem expression_not_equal_77 (x y : ℤ) :
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_CALUDE_expression_not_equal_77_l2812_281281


namespace NUMINAMATH_CALUDE_rectangular_box_problem_l2812_281214

theorem rectangular_box_problem (m n r : ℕ) (hm : m > 0) (hn : n > 0) (hr : r > 0)
  (h_order : m ≤ n ∧ n ≤ r) (h_equation : (m-2)*(n-2)*(r-2) + 4*((m-2) + (n-2) + (r-2)) - 
  2*((m-2)*(n-2) + (m-2)*(r-2) + (n-2)*(r-2)) = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_problem_l2812_281214


namespace NUMINAMATH_CALUDE_distance_to_line_l2812_281247

/-- The distance from a point on the line y = ax - 2a + 5 to the line x - 2y + 3 = 0 is √5 -/
theorem distance_to_line : ∀ (a : ℝ), ∃ (A : ℝ × ℝ),
  (A.2 = a * A.1 - 2 * a + 5) ∧ 
  (|A.1 - 2 * A.2 + 3| / Real.sqrt (1 + 4) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_line_l2812_281247


namespace NUMINAMATH_CALUDE_real_part_of_z_l2812_281263

theorem real_part_of_z (i : ℂ) (h : i * i = -1) :
  (i * (1 - 2 * i)).re = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2812_281263


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l2812_281269

theorem smaller_root_of_equation :
  let f : ℝ → ℝ := λ x => (x - 1/3)^2 + (x - 1/3)*(x + 1/6)
  (f (1/12) = 0) ∧ (∀ y < 1/12, f y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l2812_281269


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2812_281210

/-- A linear function passing through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passes_through_quadrant (a b : ℝ) (quad : ℕ) : Prop :=
  ∃ x y : ℝ, 
    (quad = 1 → x > 0 ∧ y > 0) ∧
    (quad = 2 → x < 0 ∧ y > 0) ∧
    (quad = 3 → x < 0 ∧ y < 0) ∧
    (quad = 4 → x > 0 ∧ y < 0) ∧
    y = a * x + b

theorem linear_function_not_in_third_quadrant :
  ¬ passes_through_quadrant (-1/2) 1 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2812_281210


namespace NUMINAMATH_CALUDE_juice_cost_proof_l2812_281257

/-- The cost of 5 cans of juice during a store's anniversary sale -/
def cost_of_five_juice_cans : ℝ := by sorry

theorem juice_cost_proof (original_ice_cream_price : ℝ) 
                         (ice_cream_discount : ℝ) 
                         (total_cost : ℝ) : 
  original_ice_cream_price = 12 →
  ice_cream_discount = 2 →
  total_cost = 24 →
  2 * (original_ice_cream_price - ice_cream_discount) + 2 * cost_of_five_juice_cans = total_cost →
  cost_of_five_juice_cans = 2 := by sorry

end NUMINAMATH_CALUDE_juice_cost_proof_l2812_281257


namespace NUMINAMATH_CALUDE_no_such_function_exists_l2812_281253

theorem no_such_function_exists :
  ¬∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l2812_281253


namespace NUMINAMATH_CALUDE_area_of_fifth_rectangle_l2812_281291

/-- Given a rectangle divided into five smaller rectangles, prove the area of the fifth rectangle --/
theorem area_of_fifth_rectangle
  (x y n k m : ℝ)
  (a b c d : ℝ)
  (h1 : a = k * (y - n))
  (h2 : b = (m - k) * (y - n))
  (h3 : c = m * (y - n))
  (h4 : d = (x - m) * n)
  (h5 : 0 < x ∧ 0 < y ∧ 0 < n ∧ 0 < k ∧ 0 < m)
  (h6 : n < y ∧ k < m ∧ m < x) :
  x * y - a - b - c - d = x * y - x * n :=
sorry

end NUMINAMATH_CALUDE_area_of_fifth_rectangle_l2812_281291


namespace NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l2812_281258

-- Define a function to check if two angles have the same terminal side
def same_terminal_side (a b : Int) : Prop :=
  ∃ k : Int, a ≡ b + 360 * k [ZMOD 360]

-- State the theorem
theorem negative_390_same_terminal_side_as_330 :
  same_terminal_side (-390) 330 := by
  sorry

end NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l2812_281258


namespace NUMINAMATH_CALUDE_a_value_l2812_281208

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem a_value (a : ℝ) : A ∪ B a = {0, 1, 2, 4} → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l2812_281208


namespace NUMINAMATH_CALUDE_midpoint_linear_combination_l2812_281242

/-- Given two points A and B in the plane, prove that if C is their midpoint,
    then a specific linear combination of C's coordinates equals -21. -/
theorem midpoint_linear_combination (A B : ℝ × ℝ) (h : A = (20, 9) ∧ B = (4, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 6 * C.2 = -21 := by
  sorry

#check midpoint_linear_combination

end NUMINAMATH_CALUDE_midpoint_linear_combination_l2812_281242


namespace NUMINAMATH_CALUDE_smallest_divisible_by_72_l2812_281206

theorem smallest_divisible_by_72 (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(72 ∣ m * 40)) ∧ 
  (72 ∣ n * 40) ∧ 
  (n ≥ 5) ∧
  (∃ k : ℕ, n * 40 = 72 * k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_72_l2812_281206


namespace NUMINAMATH_CALUDE_det_E_l2812_281229

/-- A 2x2 matrix representing a dilation centered at the origin with scale factor 5 -/
def E : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

/-- Theorem: The determinant of E is 25 -/
theorem det_E : Matrix.det E = 25 := by sorry

end NUMINAMATH_CALUDE_det_E_l2812_281229


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_l2812_281248

theorem modulus_of_specific_complex : let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ‖z‖ = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_l2812_281248


namespace NUMINAMATH_CALUDE_possible_values_for_e_l2812_281225

def is_digit (n : ℕ) : Prop := n < 10

def distinct (a b c e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ e ∧ b ≠ c ∧ b ≠ e ∧ c ≠ e

def subtraction_equation (a b c e : ℕ) : Prop :=
  10000 * a + 1000 * b + 100 * b + 10 * c + b -
  (10000 * b + 1000 * c + 100 * a + 10 * c + b) =
  10000 * c + 1000 * e + 100 * b + 10 * e + e

theorem possible_values_for_e :
  ∃ (s : Finset ℕ),
    (∀ e ∈ s, is_digit e) ∧
    (∀ e ∈ s, ∃ (a b c : ℕ),
      is_digit a ∧ is_digit b ∧ is_digit c ∧
      distinct a b c e ∧
      subtraction_equation a b c e) ∧
    s.card = 10 :=
sorry

end NUMINAMATH_CALUDE_possible_values_for_e_l2812_281225


namespace NUMINAMATH_CALUDE_xiaolong_exam_score_l2812_281205

theorem xiaolong_exam_score (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℤ) 
  (xiaolong_score : ℕ) (max_answered : ℕ) :
  total_questions = 50 →
  correct_points = 3 →
  incorrect_points = -1 →
  xiaolong_score = 120 →
  max_answered = 48 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect ≤ max_answered ∧
    correct * correct_points + incorrect * incorrect_points = xiaolong_score ∧
    correct ≤ 42 ∧
    ∀ (c i : ℕ), 
      c + i ≤ max_answered →
      c * correct_points + i * incorrect_points = xiaolong_score →
      c ≤ 42 :=
by sorry

end NUMINAMATH_CALUDE_xiaolong_exam_score_l2812_281205


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2812_281255

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < B → B < π →
  a = b * Real.cos C + c * Real.sin B →
  B = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2812_281255


namespace NUMINAMATH_CALUDE_equal_chore_time_l2812_281220

/-- The time in minutes it takes to sweep one room -/
def sweep_time : ℕ := 3

/-- The time in minutes it takes to wash one dish -/
def dish_time : ℕ := 2

/-- The time in minutes it takes to do one load of laundry -/
def laundry_time : ℕ := 9

/-- The number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- The number of laundry loads Billy does -/
def billy_laundry : ℕ := 2

/-- The number of dishes Billy should wash -/
def billy_dishes : ℕ := 6

theorem equal_chore_time : 
  anna_rooms * sweep_time = billy_laundry * laundry_time + billy_dishes * dish_time := by
  sorry

end NUMINAMATH_CALUDE_equal_chore_time_l2812_281220


namespace NUMINAMATH_CALUDE_nap_time_is_three_hours_nap_time_in_hours_l2812_281202

/-- Calculates the remaining time for a nap given flight duration and time spent on activities -/
def remaining_nap_time (flight_duration : ℕ) (reading_time : ℕ) (movie_time : ℕ)
  (dinner_time : ℕ) (radio_time : ℕ) (game_time : ℕ) : ℕ :=
  flight_duration - (reading_time + movie_time + dinner_time + radio_time + game_time)

/-- Theorem stating that the remaining time for a nap is 3 hours -/
theorem nap_time_is_three_hours :
  remaining_nap_time 680 120 240 30 40 70 = 180 := by
  sorry

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

/-- Theorem stating that 180 minutes is equal to 3 hours -/
theorem nap_time_in_hours :
  minutes_to_hours (remaining_nap_time 680 120 240 30 40 70) = 3 := by
  sorry

end NUMINAMATH_CALUDE_nap_time_is_three_hours_nap_time_in_hours_l2812_281202


namespace NUMINAMATH_CALUDE_bella_roses_from_parents_l2812_281276

/-- The number of dancer friends Bella has -/
def num_friends : ℕ := 10

/-- The number of roses Bella received from each friend -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := 44

/-- The number of roses Bella received from her parents -/
def roses_from_parents : ℕ := total_roses - (num_friends * roses_per_friend)

theorem bella_roses_from_parents :
  roses_from_parents = 24 :=
sorry

end NUMINAMATH_CALUDE_bella_roses_from_parents_l2812_281276


namespace NUMINAMATH_CALUDE_amount_c_l2812_281274

/-- Given four amounts a, b, c, and d satisfying certain conditions, prove that c equals 225. -/
theorem amount_c (a b c d : ℕ) : 
  a + b + c + d = 750 →
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  c = 225 := by
  sorry


end NUMINAMATH_CALUDE_amount_c_l2812_281274


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l2812_281236

/-- The volume of a cylinder minus two congruent cones -/
theorem cylinder_minus_cones_volume (r h_cylinder h_cone : ℝ) 
  (hr : r = 10)
  (hh_cylinder : h_cylinder = 20)
  (hh_cone : h_cone = 9) :
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 1400 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l2812_281236


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2812_281256

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem unique_solution_factorial_equation : 
  ∃! n : ℕ, n * factorial n + 2 * factorial n = 5040 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2812_281256


namespace NUMINAMATH_CALUDE_melissa_pencils_count_l2812_281246

/-- The number of pencils Melissa wants to buy -/
def melissa_pencils : ℕ := 2

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants to buy -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants to buy -/
def robert_pencils : ℕ := 5

/-- The total amount spent by all students in cents -/
def total_spent : ℕ := 200

theorem melissa_pencils_count :
  melissa_pencils * pencil_price + tolu_pencils * pencil_price + robert_pencils * pencil_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_melissa_pencils_count_l2812_281246


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2812_281298

theorem sum_of_fractions_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2812_281298


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_proposition_4_l2812_281260

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Notation
local notation:50 l1:50 " ⊥ " l2:50 => perpendicular l1 l2
local notation:50 l1:50 " ∥ " l2:50 => parallel l1 l2
local notation:50 l:50 " ⊥ " p:50 => perpendicularToPlane l p
local notation:50 l:50 " ∥ " p:50 => parallelToPlane l p
local notation:50 p1:50 " ⊥ " p2:50 => perpendicularPlanes p1 p2
local notation:50 p1:50 " ∥ " p2:50 => parallelPlanes p1 p2

-- Theorem statements
theorem proposition_1 (m n : Line) (α β : Plane) :
  (m ⊥ α) → (n ⊥ β) → (m ⊥ n) → (α ⊥ β) := by sorry

theorem proposition_2 (m n : Line) (α β : Plane) :
  ¬ ((m ∥ α) → (n ∥ β) → (m ∥ n) → (α ∥ β)) := by sorry

theorem proposition_3 (m n : Line) (α β : Plane) :
  ¬ ((m ⊥ α) → (n ∥ β) → (m ⊥ n) → (α ⊥ β)) := by sorry

theorem proposition_4 (m n : Line) (α β : Plane) :
  (m ⊥ α) → (n ∥ β) → (m ∥ n) → (α ⊥ β) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_proposition_4_l2812_281260


namespace NUMINAMATH_CALUDE_candy_ratio_l2812_281288

theorem candy_ratio (chocolate_bars : ℕ) (m_and_ms : ℕ) (marshmallows : ℕ) :
  chocolate_bars = 5 →
  marshmallows = 6 * m_and_ms →
  chocolate_bars + m_and_ms + marshmallows = 250 →
  m_and_ms / chocolate_bars = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l2812_281288


namespace NUMINAMATH_CALUDE_angle_d_is_190_l2812_281271

/-- A quadrilateral with angles A, B, C, and D. -/
structure Quadrilateral where
  angleA : Real
  angleB : Real
  angleC : Real
  angleD : Real
  sum_360 : angleA + angleB + angleC + angleD = 360

/-- Theorem: In a quadrilateral ABCD, if ∠A = 70°, ∠B = 60°, and ∠C = 40°, then ∠D = 190°. -/
theorem angle_d_is_190 (q : Quadrilateral) 
  (hA : q.angleA = 70)
  (hB : q.angleB = 60)
  (hC : q.angleC = 40) : 
  q.angleD = 190 := by
  sorry

end NUMINAMATH_CALUDE_angle_d_is_190_l2812_281271


namespace NUMINAMATH_CALUDE_allison_video_uploads_l2812_281278

/-- Prove that Allison uploaded 10 one-hour videos daily during the first half of June. -/
theorem allison_video_uploads :
  ∀ (x : ℕ),
  (15 * x + 15 * (2 * x) = 450) →
  x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_allison_video_uploads_l2812_281278


namespace NUMINAMATH_CALUDE_terrence_earnings_l2812_281254

def total_earnings : ℕ := 90
def emilee_earnings : ℕ := 25

theorem terrence_earnings (jermaine_earnings terrence_earnings : ℕ) 
  (h1 : jermaine_earnings = terrence_earnings + 5)
  (h2 : jermaine_earnings + terrence_earnings + emilee_earnings = total_earnings) :
  terrence_earnings = 30 := by
  sorry

end NUMINAMATH_CALUDE_terrence_earnings_l2812_281254


namespace NUMINAMATH_CALUDE_shirts_made_yesterday_is_nine_l2812_281279

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 3

/-- The number of minutes the machine worked yesterday -/
def minutes_worked_yesterday : ℕ := 3

/-- The number of shirts made yesterday -/
def shirts_made_yesterday : ℕ := shirts_per_minute * minutes_worked_yesterday

theorem shirts_made_yesterday_is_nine : shirts_made_yesterday = 9 := by
  sorry

end NUMINAMATH_CALUDE_shirts_made_yesterday_is_nine_l2812_281279


namespace NUMINAMATH_CALUDE_increasing_condition_m_range_l2812_281233

-- Define the linear function
def y (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Part 1: y increases as x increases iff m > 2
theorem increasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y m x₁ < y m x₂) ↔ m > 2 :=
sorry

-- Part 2: Range of m when -2 ≤ x ≤ 4 and y ≤ 10
theorem m_range (m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → y m x ≤ 10) ↔ (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_increasing_condition_m_range_l2812_281233


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2812_281275

/-- An isosceles triangle with given perimeter and leg length -/
structure IsoscelesTriangle where
  perimeter : ℝ
  leg_length : ℝ

/-- The base length of an isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ :=
  t.perimeter - 2 * t.leg_length

/-- Theorem: The base length of an isosceles triangle with perimeter 62 and leg length 25 is 12 -/
theorem isosceles_triangle_base_length :
  let t : IsoscelesTriangle := { perimeter := 62, leg_length := 25 }
  base_length t = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2812_281275


namespace NUMINAMATH_CALUDE_car_speed_percentage_increase_l2812_281227

/-- Proves that given two cars driving toward each other, with the first car traveling at 100 km/h,
    a distance of 720 km between them, and meeting after 4 hours, the percentage increase in the
    speed of the first car compared to the second car is 25%. -/
theorem car_speed_percentage_increase
  (speed_first : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : speed_first = 100)
  (h2 : distance = 720)
  (h3 : time = 4)
  (h4 : speed_first * time + (distance / time) * time = distance) :
  (speed_first - (distance / time)) / (distance / time) * 100 = 25 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_percentage_increase_l2812_281227


namespace NUMINAMATH_CALUDE_triangle_cut_theorem_l2812_281295

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Represents the line PQ that cuts the triangle -/
structure CuttingLine where
  length : ℝ

/-- Theorem statement for the triangle problem -/
theorem triangle_cut_theorem 
  (triangle : IsoscelesTriangle) 
  (cutting_line : CuttingLine) : 
  triangle.height = 30 ∧ 
  triangle.base * triangle.height / 2 = 180 ∧
  triangle.base * triangle.height / 2 - 135 = 
    (triangle.base * triangle.height / 2) / 4 →
  cutting_line.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cut_theorem_l2812_281295


namespace NUMINAMATH_CALUDE_interest_rate_is_six_percent_l2812_281270

/-- Calculates the simple interest rate given the principal, final amount, and time period. -/
def simple_interest_rate (principal : ℚ) (final_amount : ℚ) (time : ℚ) : ℚ :=
  ((final_amount - principal) * 100) / (principal * time)

/-- Theorem stating that for the given conditions, the simple interest rate is 6% -/
theorem interest_rate_is_six_percent :
  simple_interest_rate 12500 15500 4 = 6 := by
  sorry

#eval simple_interest_rate 12500 15500 4

end NUMINAMATH_CALUDE_interest_rate_is_six_percent_l2812_281270


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2812_281280

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0

-- Define the point A
def A : ℝ × ℝ := (3, 2)

-- Define the property of two lines being parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the property of two lines being perpendicular
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g y (-x)

-- State the theorem
theorem perpendicular_line_equation :
  ∃ (m : ℝ), parallel (l₁ m) (l₂ m) →
  ∃ (f : ℝ → ℝ → Prop),
    perpendicular (l₁ m) f ∧
    f A.1 A.2 ∧
    ∀ (x y : ℝ), f x y ↔ 2 * x - y - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2812_281280


namespace NUMINAMATH_CALUDE_value_of_a_l2812_281283

theorem value_of_a (a : ℝ) : (0.005 * a = 0.75) → a = 150 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2812_281283


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2812_281252

theorem adjacent_sum_divisible_by_four (n : ℕ) (h : n = 2006) :
  ∀ (board : Fin n → Fin n → ℕ),
  (∀ i j, board i j ∈ Finset.range (n^2 + 1)) →
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → board i j ≠ board k l) →
  ∃ (i j k l : Fin n),
    (((i = k ∧ j.val + 1 = l.val) ∨
      (i = k ∧ j.val = l.val + 1) ∨
      (i.val + 1 = k.val ∧ j = l) ∨
      (i.val = k.val + 1 ∧ j = l) ∨
      (i.val + 1 = k.val ∧ j.val + 1 = l.val) ∨
      (i.val + 1 = k.val ∧ j.val = l.val + 1) ∨
      (i.val = k.val + 1 ∧ j.val + 1 = l.val) ∨
      (i.val = k.val + 1 ∧ j.val = l.val + 1)) ∧
     (board i j + board k l) % 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2812_281252


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l2812_281259

theorem smallest_x_abs_equation : 
  (∃ x : ℝ, |x - 8| = 9) ∧ (∀ x : ℝ, |x - 8| = 9 → x ≥ -1) ∧ |-1 - 8| = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l2812_281259


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_l2812_281221

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + a*|x - 1|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
by sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ |x - 2|) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_l2812_281221


namespace NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l2812_281290

/-- Represents a player in the token game -/
inductive Player : Type
| A
| B
| C

/-- The state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)

/-- The initial state of the game -/
def initial_state : GameState :=
  { tokens := fun p => match p with
    | Player.A => 15
    | Player.B => 14
    | Player.C => 13 }

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def game_ended (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def count_rounds (state : GameState) : ℕ :=
  sorry

theorem token_game_ends_in_37_rounds :
  count_rounds initial_state = 37 :=
sorry

end NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l2812_281290


namespace NUMINAMATH_CALUDE_first_group_number_is_five_l2812_281249

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  groupSize : ℕ
  numberFromGroup17 : ℕ

/-- The systematic sampling scheme for the given problem -/
def problemSampling : SystematicSampling :=
  { totalStudents := 140
  , sampleSize := 20
  , groupSize := 7
  , numberFromGroup17 := 117
  }

/-- The number drawn from the first group in a systematic sampling -/
def firstGroupNumber (s : SystematicSampling) : ℕ :=
  s.numberFromGroup17 - s.groupSize * (17 - 1)

/-- Theorem stating that the number drawn from the first group is 5 -/
theorem first_group_number_is_five :
  firstGroupNumber problemSampling = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_group_number_is_five_l2812_281249


namespace NUMINAMATH_CALUDE_ratio_problem_l2812_281223

theorem ratio_problem (a b c : ℚ) 
  (h1 : a / b = (-5/4) / (3/2))
  (h2 : b / c = (2/3) / (-5)) :
  a / c = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2812_281223


namespace NUMINAMATH_CALUDE_prob_sum_five_l2812_281234

/-- A uniformly dense cubic die -/
structure Die :=
  (faces : Fin 6)

/-- The result of throwing a die twice -/
def TwoThrows := Die × Die

/-- The sum of points from two throws -/
def sum_points (t : TwoThrows) : ℕ :=
  t.1.faces.val + 1 + t.2.faces.val + 1

/-- The set of all possible outcomes when throwing a die twice -/
def all_outcomes : Finset TwoThrows :=
  sorry

/-- The set of outcomes where the sum of points is 5 -/
def sum_five : Finset TwoThrows :=
  sorry

/-- The probability of an event occurring when throwing a die twice -/
def prob (event : Finset TwoThrows) : ℚ :=
  (event.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_sum_five :
  prob sum_five = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_five_l2812_281234


namespace NUMINAMATH_CALUDE_circle_radius_l2812_281245

theorem circle_radius (x y : ℝ) :
  x^2 - 8*x + y^2 + 4*y + 9 = 0 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2812_281245


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l2812_281228

/-- Given two vectors a and b in ℝ², if a + x*b is perpendicular to b, then x = -2/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) 
  (ha : a = (3, 4))
  (hb : b = (2, -1))
  (h_perp : (a.1 + x * b.1, a.2 + x * b.2) • b = 0) :
  x = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l2812_281228


namespace NUMINAMATH_CALUDE_weekly_allowance_calculation_l2812_281292

/-- Represents the daily calorie allowance for a person in their 60's. -/
def daily_allowance : ℕ := 2000

/-- Represents the number of days in a week. -/
def days_in_week : ℕ := 7

/-- Calculates the weekly calorie allowance based on the daily allowance. -/
def weekly_allowance : ℕ := daily_allowance * days_in_week

/-- Proves that the weekly calorie allowance for a person in their 60's
    with an average daily allowance of 2000 calories is equal to 10500 calories. -/
theorem weekly_allowance_calculation :
  weekly_allowance = 10500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_allowance_calculation_l2812_281292


namespace NUMINAMATH_CALUDE_random_events_identification_l2812_281212

-- Define the type for events
inductive Event : Type
  | draw_glasses : Event
  | guess_digit : Event
  | electric_charges : Event
  | lottery_win : Event

-- Define what it means for an event to be random
def is_random_event (e : Event) : Prop :=
  ∀ (outcome : Prop), ¬(outcome ∧ ¬outcome)

-- State the theorem
theorem random_events_identification :
  (is_random_event Event.draw_glasses) ∧
  (is_random_event Event.guess_digit) ∧
  (is_random_event Event.lottery_win) ∧
  (¬is_random_event Event.electric_charges) := by
  sorry

end NUMINAMATH_CALUDE_random_events_identification_l2812_281212


namespace NUMINAMATH_CALUDE_town_population_problem_l2812_281251

theorem town_population_problem :
  ∃ n : ℕ, 
    (∃ a b : ℕ, 
      n * (n + 1) / 2 + 121 = a^2 ∧
      n * (n + 1) / 2 + 121 + 144 = b^2) ∧
    n * (n + 1) / 2 = 2280 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l2812_281251


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2812_281293

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (q > 0) →  -- q is positive
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (a 3 * a 9 = 2 * (a 5)^2) →  -- given condition
  q = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2812_281293


namespace NUMINAMATH_CALUDE_distributive_property_l2812_281226

theorem distributive_property (m : ℝ) : m * (m - 1) = m^2 - m := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l2812_281226


namespace NUMINAMATH_CALUDE_euros_to_rubles_conversion_l2812_281222

/-- Exchange rate from euros to US dollars -/
def euro_to_usd_rate : ℚ := 12 / 10

/-- Exchange rate from US dollars to rubles -/
def usd_to_ruble_rate : ℚ := 60

/-- Cost of the travel package in euros -/
def travel_package_cost : ℚ := 600

/-- Theorem stating the equivalence of 600 euros to 43200 rubles given the exchange rates -/
theorem euros_to_rubles_conversion :
  (travel_package_cost * euro_to_usd_rate * usd_to_ruble_rate : ℚ) = 43200 := by
  sorry


end NUMINAMATH_CALUDE_euros_to_rubles_conversion_l2812_281222


namespace NUMINAMATH_CALUDE_sum_a_2b_is_zero_l2812_281240

theorem sum_a_2b_is_zero (a b : ℝ) (h : (a^2 + 4*a + 6)*(2*b^2 - 4*b + 7) ≤ 10) : 
  a + 2*b = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_a_2b_is_zero_l2812_281240


namespace NUMINAMATH_CALUDE_mushroom_pickers_l2812_281211

theorem mushroom_pickers (n : ℕ) (A V S R : ℚ) : 
  (∀ i : Fin n, i.val ≠ 0 ∧ i.val ≠ 1 ∧ i.val ≠ 2 → A / 2 = V + A / 2) →  -- Condition 1
  (S + A = R + V + A) →                                                   -- Condition 2
  (A > 0) →                                                               -- Anya has mushrooms
  (n > 3) →                                                               -- At least 4 children
  (n : ℚ) * (A / 2) = A + V + S + R →                                     -- Total mushrooms
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_mushroom_pickers_l2812_281211


namespace NUMINAMATH_CALUDE_one_seventh_comparison_l2812_281299

theorem one_seventh_comparison : (1 : ℚ) / 7 - 142857142857 / 1000000000000 = 1 / (7 * 1000000000000) := by
  sorry

end NUMINAMATH_CALUDE_one_seventh_comparison_l2812_281299


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2812_281224

/-- The x-intercept of the line -4x + 6y = 24 is (-6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  -4 * x + 6 * y = 24 → y = 0 → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2812_281224


namespace NUMINAMATH_CALUDE_car_travel_distance_l2812_281294

/-- Proves that a car traveling for 12 hours at 68 km/h covers 816 km -/
theorem car_travel_distance (travel_time : ℝ) (average_speed : ℝ) (h1 : travel_time = 12) (h2 : average_speed = 68) : travel_time * average_speed = 816 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l2812_281294


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1879_l2812_281244

theorem smallest_prime_factor_of_1879 :
  Nat.minFac 1879 = 17 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1879_l2812_281244


namespace NUMINAMATH_CALUDE_last_digits_of_powers_l2812_281219

theorem last_digits_of_powers : 
  (∃ n : ℕ, 1989^1989 ≡ 9 [MOD 10]) ∧
  (∃ n : ℕ, 1989^1992 ≡ 1 [MOD 10]) ∧
  (∃ n : ℕ, 1992^1989 ≡ 2 [MOD 10]) ∧
  (∃ n : ℕ, 1992^1992 ≡ 6 [MOD 10]) :=
by sorry

end NUMINAMATH_CALUDE_last_digits_of_powers_l2812_281219


namespace NUMINAMATH_CALUDE_grace_pool_volume_l2812_281238

/-- The volume of water in Grace's pool -/
def pool_volume (first_hose_rate : ℝ) (first_hose_time : ℝ) (second_hose_rate : ℝ) (second_hose_time : ℝ) : ℝ :=
  first_hose_rate * first_hose_time + second_hose_rate * second_hose_time

/-- Theorem stating that Grace's pool contains 390 gallons of water -/
theorem grace_pool_volume :
  let first_hose_rate : ℝ := 50
  let first_hose_time : ℝ := 5
  let second_hose_rate : ℝ := 70
  let second_hose_time : ℝ := 2
  pool_volume first_hose_rate first_hose_time second_hose_rate second_hose_time = 390 :=
by
  sorry


end NUMINAMATH_CALUDE_grace_pool_volume_l2812_281238


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2812_281231

theorem not_necessarily_right_triangle (a b c : ℝ) : 
  a^2 = 5 → b^2 = 12 → c^2 = 13 → 
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2812_281231


namespace NUMINAMATH_CALUDE_student_arrangement_count_l2812_281200

/-- The number of ways to arrange students into communities --/
def arrange_students (total_students : ℕ) (selected_students : ℕ) (communities : ℕ) (min_per_community : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem student_arrangement_count :
  arrange_students 7 6 2 2 = 350 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l2812_281200


namespace NUMINAMATH_CALUDE_expression_evaluation_l2812_281273

theorem expression_evaluation :
  let a : ℚ := -3/2
  let expr := 1 + (1 - a) / a / ((a^2 - 1) / (a^2 + 2*a))
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2812_281273


namespace NUMINAMATH_CALUDE_line_point_x_coordinate_l2812_281289

/-- Given a line with slope -3.5 and y-intercept 1.5, 
    the x-coordinate of the point with y-coordinate 1025 is -1023.5 / 3.5 -/
theorem line_point_x_coordinate 
  (slope : ℝ) 
  (y_intercept : ℝ) 
  (y : ℝ) 
  (h1 : slope = -3.5) 
  (h2 : y_intercept = 1.5) 
  (h3 : y = 1025) : 
  (y - y_intercept) / (-slope) = -1023.5 / 3.5 := by
  sorry

#eval -1023.5 / 3.5

end NUMINAMATH_CALUDE_line_point_x_coordinate_l2812_281289


namespace NUMINAMATH_CALUDE_perpendicular_equal_magnitude_vectors_l2812_281284

/-- Given two vectors m and n in ℝ², prove that if n is obtained by swapping and negating one component of m, then m and n are perpendicular and have equal magnitudes. -/
theorem perpendicular_equal_magnitude_vectors
  (a b : ℝ) :
  let m : ℝ × ℝ := (a, b)
  let n : ℝ × ℝ := (b, -a)
  (m.1 * n.1 + m.2 * n.2 = 0) ∧ 
  (m.1^2 + m.2^2 = n.1^2 + n.2^2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_equal_magnitude_vectors_l2812_281284


namespace NUMINAMATH_CALUDE_expression_value_l2812_281266

theorem expression_value : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2812_281266


namespace NUMINAMATH_CALUDE_garden_path_width_l2812_281267

/-- Given two concentric circles with a difference in circumference of 20π meters,
    the width of the path between them is 10 meters. -/
theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) :
  R - r = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_path_width_l2812_281267


namespace NUMINAMATH_CALUDE_max_dominoes_20x19_grid_l2812_281241

/-- Represents a rectangular grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a domino --/
structure Domino where
  length : ℕ
  width : ℕ

/-- The maximum number of dominoes that can be placed on a grid --/
def max_dominoes (g : Grid) (d : Domino) : ℕ :=
  (g.rows * g.cols) / (d.length * d.width)

/-- The theorem stating the maximum number of 3×1 dominoes on a 20×19 grid --/
theorem max_dominoes_20x19_grid :
  let grid : Grid := ⟨20, 19⟩
  let domino : Domino := ⟨3, 1⟩
  max_dominoes grid domino = 126 := by
  sorry

#eval max_dominoes ⟨20, 19⟩ ⟨3, 1⟩

end NUMINAMATH_CALUDE_max_dominoes_20x19_grid_l2812_281241


namespace NUMINAMATH_CALUDE_difference_divisible_by_99_l2812_281285

/-- Represents a three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.c

/-- The value of a three-digit number with hundreds and units digits exchanged -/
def exchangedValue (n : ThreeDigitNumber) : ℕ :=
  100 * n.c + 10 * n.b + n.a

/-- The difference between the exchanged value and the original value -/
def difference (n : ThreeDigitNumber) : ℤ :=
  (exchangedValue n : ℤ) - (value n : ℤ)

/-- Theorem stating that the difference is always divisible by 99 -/
theorem difference_divisible_by_99 (n : ThreeDigitNumber) :
  ∃ k : ℤ, difference n = 99 * k := by
  sorry


end NUMINAMATH_CALUDE_difference_divisible_by_99_l2812_281285


namespace NUMINAMATH_CALUDE_doll_collection_increase_l2812_281268

theorem doll_collection_increase (original_count : ℕ) (increase : ℕ) (final_count : ℕ) :
  original_count + increase = final_count →
  final_count = 10 →
  increase = 2 →
  (increase : ℚ) / (original_count : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l2812_281268


namespace NUMINAMATH_CALUDE_largest_fraction_l2812_281215

theorem largest_fraction : 
  let a := (1 / 17 - 1 / 19) / 20
  let b := (1 / 15 - 1 / 21) / 60
  let c := (1 / 13 - 1 / 23) / 100
  let d := (1 / 11 - 1 / 25) / 140
  d > a ∧ d > b ∧ d > c := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l2812_281215


namespace NUMINAMATH_CALUDE_expression_simplification_l2812_281209

theorem expression_simplification (x : ℚ) (h : x = 3) : 
  (((x - 1) / (x + 2) + 1) / ((x - 1) / (x + 2) - 1)) = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2812_281209


namespace NUMINAMATH_CALUDE_sally_initial_peaches_l2812_281235

/-- The number of peaches Sally picked from the orchard -/
def picked_peaches : ℕ := 42

/-- The final number of peaches at Sally's stand -/
def final_peaches : ℕ := 55

/-- The initial number of peaches at Sally's stand -/
def initial_peaches : ℕ := final_peaches - picked_peaches

theorem sally_initial_peaches : initial_peaches = 13 := by
  sorry

end NUMINAMATH_CALUDE_sally_initial_peaches_l2812_281235


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l2812_281230

/-- The number of oak trees in the park after planting -/
def trees_after_planting (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem: There will be 11 oak trees after planting -/
theorem oak_trees_after_planting :
  trees_after_planting 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l2812_281230


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l2812_281264

/-- Given a ratio of blue paint to white paint and the amount of white paint used,
    calculate the amount of blue paint required. -/
def blue_paint_required (blue_ratio : ℚ) (white_ratio : ℚ) (white_amount : ℚ) : ℚ :=
  (blue_ratio / white_ratio) * white_amount

/-- Theorem stating that given a 5:6 ratio of blue to white paint and 18 quarts of white paint,
    15 quarts of blue paint are required. -/
theorem blue_paint_calculation :
  let blue_ratio : ℚ := 5
  let white_ratio : ℚ := 6
  let white_amount : ℚ := 18
  blue_paint_required blue_ratio white_ratio white_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l2812_281264


namespace NUMINAMATH_CALUDE_base_number_proof_l2812_281261

theorem base_number_proof (x : ℝ) (n : ℕ) 
  (h1 : 4 * x^(2*n) = 4^18) (h2 : n = 17) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2812_281261


namespace NUMINAMATH_CALUDE_inequality_proof_l2812_281286

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧ 
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2812_281286
