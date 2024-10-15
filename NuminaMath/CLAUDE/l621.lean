import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l621_62163

theorem percentage_problem (n : ℝ) (p : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 10 →
  (p / 100) * n = 120 →
  p = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l621_62163


namespace NUMINAMATH_CALUDE_max_triangle_area_l621_62110

theorem max_triangle_area (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (area : ℝ), area ≤ 1 ∧ 
  ∀ (area' : ℝ), (∃ (a' b' c' : ℝ), 
    0 ≤ a' ∧ a' ≤ 1 ∧ 
    1 ≤ b' ∧ b' ≤ 2 ∧ 
    2 ≤ c' ∧ c' ≤ 3 ∧ 
    a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a' ∧
    area' = (a' + b' + c') * (a' + b' - c') * (a' - b' + c') * (-a' + b' + c') / 16) → 
  area' ≤ area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l621_62110


namespace NUMINAMATH_CALUDE_f_max_value_l621_62136

/-- The function f(x) = 8x - 3x^2 -/
def f (x : ℝ) : ℝ := 8 * x - 3 * x^2

/-- The maximum value of f(x) for any real x is 16/3 -/
theorem f_max_value : ∃ (M : ℝ), M = 16/3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l621_62136


namespace NUMINAMATH_CALUDE_strawberries_left_l621_62134

/-- Given an initial amount of strawberries and amounts eaten over two days, 
    calculate the remaining amount. -/
def remaining_strawberries (initial : ℝ) (eaten_day1 : ℝ) (eaten_day2 : ℝ) : ℝ :=
  initial - eaten_day1 - eaten_day2

/-- Theorem stating that given the specific amounts in the problem, 
    the remaining amount of strawberries is 0.5 kg. -/
theorem strawberries_left : 
  remaining_strawberries 1.6 0.8 0.3 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_left_l621_62134


namespace NUMINAMATH_CALUDE_triangle_properties_l621_62146

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * (Real.cos (t.B / 2))^2 + t.b * (Real.cos (t.A / 2))^2 = (3/2) * t.c)
  (h2 : t.a = 2 * t.b)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15) :
  (t.A > π / 2) ∧ (t.b = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l621_62146


namespace NUMINAMATH_CALUDE_unfinished_courses_l621_62124

/-- Given the conditions of a construction project, calculate the number of unfinished courses in the last wall. -/
theorem unfinished_courses
  (courses_per_wall : ℕ)
  (bricks_per_course : ℕ)
  (total_walls : ℕ)
  (bricks_used : ℕ)
  (h1 : courses_per_wall = 6)
  (h2 : bricks_per_course = 10)
  (h3 : total_walls = 4)
  (h4 : bricks_used = 220) :
  (courses_per_wall * bricks_per_course * total_walls - bricks_used) / bricks_per_course = 2 :=
by sorry

end NUMINAMATH_CALUDE_unfinished_courses_l621_62124


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l621_62161

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 8 * x + 1 < 0 ↔ (4 - Real.sqrt 19) / 3 < x ∧ x < (4 + Real.sqrt 19) / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l621_62161


namespace NUMINAMATH_CALUDE_product_difference_equality_l621_62193

theorem product_difference_equality : 2012.25 * 2013.75 - 2010.25 * 2015.75 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equality_l621_62193


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l621_62118

theorem common_roots_cubic_polynomials :
  ∀ (a b : ℝ),
  (∃ (r s : ℝ), r ≠ s ∧
    (r^3 + a*r^2 + 15*r + 10 = 0) ∧
    (r^3 + b*r^2 + 18*r + 12 = 0) ∧
    (s^3 + a*s^2 + 15*s + 10 = 0) ∧
    (s^3 + b*s^2 + 18*s + 12 = 0)) →
  a = 6 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l621_62118


namespace NUMINAMATH_CALUDE_sum_of_complex_unit_magnitude_l621_62167

theorem sum_of_complex_unit_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^2 / (b*c) + b^2 / (a*c) + c^2 / (a*b) = 3)
  (h5 : a + b + c ≠ 0) :
  Complex.abs (a + b + c) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_complex_unit_magnitude_l621_62167


namespace NUMINAMATH_CALUDE_min_value_theorem_l621_62150

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 5/b = 1) :
  a + 5*b ≥ 36 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 5/b₀ = 1 ∧ a₀ + 5*b₀ = 36 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l621_62150


namespace NUMINAMATH_CALUDE_rationality_of_expressions_l621_62157

theorem rationality_of_expressions :
  (∃ (a b : ℤ), b ≠ 0 ∧ (1.728 : ℚ) = a / b) ∧
  (∃ (c d : ℤ), d ≠ 0 ∧ (0.0032 : ℚ) = c / d) ∧
  (∃ (e f : ℤ), f ≠ 0 ∧ (-8 : ℚ) = e / f) ∧
  (∃ (g h : ℤ), h ≠ 0 ∧ (0.25 : ℚ) = g / h) ∧
  ¬(∃ (i j : ℤ), j ≠ 0 ∧ Real.pi = (i : ℚ) / j) :=
by sorry

end NUMINAMATH_CALUDE_rationality_of_expressions_l621_62157


namespace NUMINAMATH_CALUDE_average_weight_problem_l621_62127

theorem average_weight_problem (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 46 →
  b = 37 →
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l621_62127


namespace NUMINAMATH_CALUDE_translated_line_y_axis_intersection_l621_62147

/-- The intersection point of a line translated upward with the y-axis -/
theorem translated_line_y_axis_intersection
  (original_line : ℝ → ℝ)
  (h_original : ∀ x, original_line x = x - 3)
  (translation : ℝ)
  (h_translation : translation = 2)
  (translated_line : ℝ → ℝ)
  (h_translated : ∀ x, translated_line x = original_line x + translation)
  : translated_line 0 = -1 :=
by sorry

end NUMINAMATH_CALUDE_translated_line_y_axis_intersection_l621_62147


namespace NUMINAMATH_CALUDE_continuity_at_zero_l621_62132

noncomputable def f (x : ℝ) : ℝ := 
  (Real.rpow (1 + x) (1/3) - 1) / (Real.sqrt (4 + x) - 2)

theorem continuity_at_zero : 
  Filter.Tendsto f (nhds 0) (nhds (4/3)) := by sorry

end NUMINAMATH_CALUDE_continuity_at_zero_l621_62132


namespace NUMINAMATH_CALUDE_expression_value_l621_62141

theorem expression_value (a b c k : ℤ) 
  (ha : a = 30) (hb : b = 10) (hc : c = 7) (hk : k = 3) : 
  k * ((a - (b - c)) - ((a - b) - c)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l621_62141


namespace NUMINAMATH_CALUDE_red_button_probability_main_theorem_l621_62130

/-- Represents a jar containing buttons of different colors -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Represents the state of the jars after Carla's action -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

/-- The probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / j.total

/-- The initial state of Jar A -/
def initialJarA : Jar := ⟨6, 10⟩

/-- Theorem stating the probability of selecting red buttons from both jars -/
theorem red_button_probability (state : JarState) : 
  redProbability state.jarA * redProbability state.jarB = 1/6 :=
sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem (state : JarState) :
  initialJarA.total = 16 →
  state.jarA.total = (3/4 : ℚ) * initialJarA.total →
  state.jarB.total = initialJarA.total - state.jarA.total →
  state.jarB.red = state.jarB.blue →
  state.jarA.red + state.jarB.red = initialJarA.red →
  state.jarA.blue + state.jarB.blue = initialJarA.blue →
  redProbability state.jarA * redProbability state.jarB = 1/6 :=
sorry

end NUMINAMATH_CALUDE_red_button_probability_main_theorem_l621_62130


namespace NUMINAMATH_CALUDE_fencing_rate_proof_l621_62104

/-- Given a rectangular plot with the following properties:
  - The length is 10 meters more than the width
  - The perimeter is 300 meters
  - The total fencing cost is 1950 Rs
  Prove that the rate per meter for fencing is 6.5 Rs -/
theorem fencing_rate_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) (total_cost : ℝ) :
  length = width + 10 →
  perimeter = 300 →
  perimeter = 2 * (length + width) →
  total_cost = 1950 →
  total_cost / perimeter = 6.5 := by
sorry

end NUMINAMATH_CALUDE_fencing_rate_proof_l621_62104


namespace NUMINAMATH_CALUDE_largest_common_term_under_300_l621_62191

-- Define the first arithmetic progression
def seq1 (n : ℕ) : ℕ := 3 * n + 1

-- Define the second arithmetic progression
def seq2 (n : ℕ) : ℕ := 10 * n + 2

-- Define a function to check if a number is in both sequences
def isCommonTerm (x : ℕ) : Prop :=
  ∃ n m : ℕ, seq1 n = x ∧ seq2 m = x

-- Theorem statement
theorem largest_common_term_under_300 :
  (∀ x : ℕ, x < 300 → isCommonTerm x → x ≤ 290) ∧
  isCommonTerm 290 := by sorry

end NUMINAMATH_CALUDE_largest_common_term_under_300_l621_62191


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l621_62123

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 - a*b + b^2

-- Theorem statement
theorem custom_mult_four_three : custom_mult 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l621_62123


namespace NUMINAMATH_CALUDE_rectangle_area_with_equal_perimeter_to_triangle_l621_62197

/-- The area of a rectangle with equal perimeter to a specific triangle -/
theorem rectangle_area_with_equal_perimeter_to_triangle : 
  ∀ (rectangle_side1 rectangle_side2 : ℝ),
  rectangle_side1 = 12 →
  2 * (rectangle_side1 + rectangle_side2) = 10 + 12 + 15 →
  rectangle_side1 * rectangle_side2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_equal_perimeter_to_triangle_l621_62197


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l621_62192

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 4

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one white ball when randomly selecting 2 balls from a bag containing 4 red balls and 2 white balls -/
theorem probability_of_white_ball : 
  (1 : ℚ) - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l621_62192


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l621_62175

theorem contrapositive_equivalence (a b c d : ℝ) :
  ((a = b ∧ c = d) → a + c = b + d) ↔ (a + c ≠ b + d → a ≠ b ∨ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l621_62175


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l621_62145

/-- A circle tangent to the parabola y = x^2 at two points and lying inside the parabola --/
structure TangentCircle where
  /-- The x-coordinate of one tangent point (the other is at -a) --/
  a : ℝ
  /-- The y-coordinate of the circle's center --/
  b : ℝ
  /-- The radius of the circle --/
  r : ℝ
  /-- The circle lies inside the parabola --/
  inside : b > a^2
  /-- The circle is tangent to the parabola at (a, a^2) and (-a, a^2) --/
  tangent : (a^2 + (a^2 - b)^2 = r^2) ∧ (a^2 + (a^2 - b)^2 = r^2)

/-- The difference between the y-coordinate of the circle's center and the y-coordinate of either tangent point is 1/2 --/
theorem tangent_circle_height_difference (c : TangentCircle) : c.b - c.a^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l621_62145


namespace NUMINAMATH_CALUDE_intersection_with_complement_l621_62171

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {1, 2, 5, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l621_62171


namespace NUMINAMATH_CALUDE_work_completion_time_l621_62199

/-- Given that:
  - B can do a work in 24 days
  - A and B working together can finish the work in 8 days
  Prove that A can do the work alone in 12 days -/
theorem work_completion_time (work : ℝ) (a_rate b_rate combined_rate : ℝ) :
  work / b_rate = 24 →
  work / combined_rate = 8 →
  combined_rate = a_rate + b_rate →
  work / a_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l621_62199


namespace NUMINAMATH_CALUDE_curve_composition_l621_62120

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt (-y^2 + 2*y + 8)) * Real.sqrt (x - y) = 0

-- Define the line segment
def line_segment (x y : ℝ) : Prop :=
  x = y ∧ -2 ≤ y ∧ y ≤ 4

-- Define the minor arc
def minor_arc (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 9 ∧ x ≥ 0

-- Theorem stating that the curve consists of a line segment and a minor arc
theorem curve_composition :
  ∀ x y : ℝ, curve_equation x y ↔ (line_segment x y ∨ minor_arc x y) :=
sorry

end NUMINAMATH_CALUDE_curve_composition_l621_62120


namespace NUMINAMATH_CALUDE_relationship_abc_l621_62135

theorem relationship_abc : 
  let a := (1/2)^(2/3)
  let b := (1/3)^(1/3)
  let c := Real.log 3
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l621_62135


namespace NUMINAMATH_CALUDE_translated_circle_center_l621_62107

/-- Given a point A(1,1) and a point P(m,n) on the circle centered at A, 
    if P is symmetric with P' with respect to the origin after translation,
    then the coordinates of A' are (1-2m, 1-2n) -/
theorem translated_circle_center (m n : ℝ) : 
  let A : ℝ × ℝ := (1, 1)
  let P : ℝ × ℝ := (m, n)
  let O : ℝ × ℝ := (0, 0)
  ∃ (A' : ℝ × ℝ), 
    (∃ (P' : ℝ × ℝ), P'.1 = -P.1 ∧ P'.2 = -P.2) →  -- P and P' are symmetric about origin
    (∃ (r : ℝ), (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2) →  -- P is on circle centered at A
    A' = (1 - 2*m, 1 - 2*n) :=
sorry

end NUMINAMATH_CALUDE_translated_circle_center_l621_62107


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l621_62152

/-- The line y = 2 is tangent to the circle (x - 2)² + y² = 4 -/
theorem line_tangent_to_circle :
  ∃ (x y : ℝ), y = 2 ∧ (x - 2)^2 + y^2 = 4 ∧
  ∀ (x' y' : ℝ), y' = 2 → (x' - 2)^2 + y'^2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l621_62152


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l621_62131

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l621_62131


namespace NUMINAMATH_CALUDE_sum_of_transformed_numbers_l621_62178

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_transformed_numbers_l621_62178


namespace NUMINAMATH_CALUDE_sum_of_series_equals_two_l621_62125

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-2)/(3^n) is equal to 2. -/
theorem sum_of_series_equals_two :
  ∑' n : ℕ, (4 * n - 2 : ℝ) / (3 ^ n) = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_two_l621_62125


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l621_62122

/-- Given a line l: 2x + y - 5 = 0 and a point M(-1, 2), 
    this function returns the coordinates of the symmetric point Q with respect to l. -/
def symmetricPoint (l : ℝ → ℝ → Prop) (M : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := M
  -- Define the symmetric point Q
  let Q := (3, 4)
  Q

/-- The theorem states that the symmetric point of M(-1, 2) with respect to
    the line 2x + y - 5 = 0 is (3, 4). -/
theorem symmetric_point_coordinates :
  let l : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y - 5 = 0
  let M : ℝ × ℝ := (-1, 2)
  symmetricPoint l M = (3, 4) := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l621_62122


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l621_62183

theorem polynomial_division_quotient : 
  let dividend := fun (z : ℝ) => 5*z^5 - 3*z^4 + 4*z^3 - 7*z^2 + 2*z - 1
  let divisor := fun (z : ℝ) => 3*z^2 + 4*z + 1
  let quotient := fun (z : ℝ) => (5/3)*z^3 - (29/9)*z^2 + (71/27)*z - 218/81
  ∀ z : ℝ, dividend z = (divisor z) * (quotient z) + (dividend z % divisor z) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l621_62183


namespace NUMINAMATH_CALUDE_added_amount_after_doubling_l621_62111

theorem added_amount_after_doubling (x y : ℕ) : 
  x = 17 → 3 * (2 * x + y) = 117 → y = 5 := by sorry

end NUMINAMATH_CALUDE_added_amount_after_doubling_l621_62111


namespace NUMINAMATH_CALUDE_geometric_configurations_l621_62184

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (passes_through : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem geometric_configurations 
  (α β : Plane) (l m : Line) 
  (h1 : passes_through α l)
  (h2 : passes_through α m)
  (h3 : passes_through β l)
  (h4 : passes_through β m)
  (h5 : skew l m)
  (h6 : perpendicular l m) :
  (∃ (α' β' : Plane) (l' m' : Line), 
    passes_through α' l' ∧ 
    passes_through α' m' ∧ 
    passes_through β' l' ∧ 
    passes_through β' m' ∧ 
    skew l' m' ∧ 
    perpendicular l' m' ∧
    ((parallel α' β') ∨ 
     (perpendicular_planes α' β') ∨ 
     (parallel_line_plane l' β') ∨ 
     (perpendicular_line_plane m' α'))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_configurations_l621_62184


namespace NUMINAMATH_CALUDE_sum_inequality_l621_62176

theorem sum_inequality (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a : ℚ) / (b + c^2) = (a + c^2 : ℚ) / b) : 
  a + b + c ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sum_inequality_l621_62176


namespace NUMINAMATH_CALUDE_triangle_rotation_theorem_l621_62139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices O, P, and Q -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point) : ℝ := sorry

/-- Rotates a point 90 degrees counterclockwise around the origin -/
def rotate90 (p : Point) : Point := 
  { x := -p.y, y := p.x }

theorem triangle_rotation_theorem (t : Triangle) : 
  t.O = ⟨0, 0⟩ → 
  t.Q = ⟨6, 0⟩ → 
  t.P.x > 0 → 
  t.P.y > 0 → 
  angle ⟨t.P.x - t.Q.x, t.P.y - t.Q.y⟩ ⟨t.O.x - t.Q.x, t.O.y - t.Q.y⟩ = π / 2 →
  angle ⟨t.P.x - t.O.x, t.P.y - t.O.y⟩ ⟨t.Q.x - t.O.x, t.Q.y - t.O.y⟩ = π / 4 →
  t.P = ⟨6, 6⟩ ∧ rotate90 t.P = ⟨-6, 6⟩ := by
sorry

end NUMINAMATH_CALUDE_triangle_rotation_theorem_l621_62139


namespace NUMINAMATH_CALUDE_units_digit_37_pow_37_l621_62151

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 37^37 is 7 -/
theorem units_digit_37_pow_37 : unitsDigit (37^37) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_37_pow_37_l621_62151


namespace NUMINAMATH_CALUDE_power_product_equals_l621_62160

theorem power_product_equals : 2^4 * 3^2 * 5^2 * 11 = 39600 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_l621_62160


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l621_62140

theorem cos_pi_fourth_plus_alpha (α : ℝ) (h : Real.sin (π/4 - α) = 1/3) : 
  Real.cos (π/4 + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l621_62140


namespace NUMINAMATH_CALUDE_coordinate_uniqueness_l621_62144

/-- A type representing a location description -/
inductive LocationDescription
| Coordinates (longitude : Real) (latitude : Real)
| CityLandmark (city : String) (landmark : String)
| Direction (angle : Real)
| VenueSeat (venue : String) (seat : String)

/-- Function to check if a location description uniquely determines a location -/
def uniquelyDeterminesLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

/-- Theorem stating that only coordinate-based descriptions uniquely determine locations -/
theorem coordinate_uniqueness 
  (descriptions : List LocationDescription) 
  (h_contains_coordinates : ∃ (long lat : Real), LocationDescription.Coordinates long lat ∈ descriptions) :
  ∃! (desc : LocationDescription), desc ∈ descriptions ∧ uniquelyDeterminesLocation desc :=
sorry

end NUMINAMATH_CALUDE_coordinate_uniqueness_l621_62144


namespace NUMINAMATH_CALUDE_circle_equation_proof_l621_62142

-- Define a circle with center (1, 1) passing through (0, 0)
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, circle_equation x y ↔ 
    ((x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 0) → 
      (x - 1)^2 + (y - 1)^2 = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l621_62142


namespace NUMINAMATH_CALUDE_range_of_a_for_positive_x_l621_62108

theorem range_of_a_for_positive_x (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2*x - a = 3*x - 4) ↔ a < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_positive_x_l621_62108


namespace NUMINAMATH_CALUDE_perfect_square_sum_in_pile_l621_62170

theorem perfect_square_sum_in_pile (n : ℕ) (h : n ≥ 100) :
  ∀ (S₁ S₂ : Set ℕ), 
    (∀ k, n ≤ k ∧ k ≤ 2*n → k ∈ S₁ ∨ k ∈ S₂) →
    (S₁ ∩ S₂ = ∅) →
    (∃ (a b : ℕ), (a ∈ S₁ ∧ b ∈ S₁ ∧ a ≠ b ∧ ∃ (m : ℕ), a + b = m^2) ∨
                   (a ∈ S₂ ∧ b ∈ S₂ ∧ a ≠ b ∧ ∃ (m : ℕ), a + b = m^2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_in_pile_l621_62170


namespace NUMINAMATH_CALUDE_function_properties_l621_62162

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_properties (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_shift : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∃ p, p > 0 ∧ ∀ x, f (x + p) = f x) ∧
  (∀ x, f (2 - x) = f x) ∧
  f 2 = f 0 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l621_62162


namespace NUMINAMATH_CALUDE_gravelingCostIs3600_l621_62148

/-- Represents the dimensions and cost parameters of a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  costPerSqm : ℝ

/-- Calculates the total cost of graveling two intersecting roads in a rectangular lawn -/
def totalGravelingCost (lawn : LawnWithRoads) : ℝ :=
  let lengthRoadArea := lawn.length * lawn.roadWidth
  let widthRoadArea := (lawn.width - lawn.roadWidth) * lawn.roadWidth
  let totalArea := lengthRoadArea + widthRoadArea
  totalArea * lawn.costPerSqm

/-- Theorem stating that the total cost of graveling for the given lawn is 3600 -/
theorem gravelingCostIs3600 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.width = 50)
  (h3 : lawn.roadWidth = 10)
  (h4 : lawn.costPerSqm = 3) :
  totalGravelingCost lawn = 3600 := by
  sorry

#eval totalGravelingCost { length := 80, width := 50, roadWidth := 10, costPerSqm := 3 }

end NUMINAMATH_CALUDE_gravelingCostIs3600_l621_62148


namespace NUMINAMATH_CALUDE_earnings_ratio_l621_62180

/-- Proves that given Mork's tax rate of 30%, Mindy's tax rate of 20%, and their combined tax rate of 22.5%, the ratio of Mindy's earnings to Mork's earnings is 3:1. -/
theorem earnings_ratio (mork_earnings mindy_earnings : ℝ) 
  (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) (combined_tax_rate : ℝ)
  (h1 : mork_tax_rate = 0.3)
  (h2 : mindy_tax_rate = 0.2)
  (h3 : combined_tax_rate = 0.225)
  (h4 : mork_earnings > 0)
  (h5 : mindy_earnings > 0)
  (h6 : mindy_tax_rate * mindy_earnings + mork_tax_rate * mork_earnings = 
        combined_tax_rate * (mindy_earnings + mork_earnings)) :
  mindy_earnings / mork_earnings = 3 := by
  sorry


end NUMINAMATH_CALUDE_earnings_ratio_l621_62180


namespace NUMINAMATH_CALUDE_shepherd_puzzle_l621_62153

/-- The number of sheep after passing through a gate, given the number before the gate -/
def sheep_after_gate (n : ℕ) : ℕ := n / 2 + 1

/-- The number of sheep after passing through n gates, given the initial number -/
def sheep_after_gates (initial : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | n + 1 => sheep_after_gate (sheep_after_gates initial n)

theorem shepherd_puzzle :
  ∃ initial : ℕ, sheep_after_gates initial 6 = 2 ∧ initial = 2 := by sorry

end NUMINAMATH_CALUDE_shepherd_puzzle_l621_62153


namespace NUMINAMATH_CALUDE_problem_solution_l621_62174

open Real

noncomputable def α : ℝ := sorry

-- Given conditions
axiom cond1 : (sin (π/2 - α) + sin (-π - α)) / (3 * cos (2*π + α) + cos (3*π/2 - α)) = 3
axiom cond2 : ∃ (a : ℝ), ∀ (x y : ℝ), (x - a)^2 + y^2 = 7 → y = 0
axiom cond3 : ∃ (a : ℝ), abs (2*a) / sqrt 5 = sqrt 5
axiom cond4 : ∃ (a r : ℝ), r > 0 ∧ (2*sqrt 2)^2 + (sqrt 5)^2 = (2*r)^2 ∧ ∀ (x y : ℝ), (x - a)^2 + y^2 = r^2

-- Theorem to prove
theorem problem_solution :
  (sin α - 3*cos α) / (sin α + cos α) = -1/3 ∧
  ∃ (a : ℝ), (∀ (x y : ℝ), (x - a)^2 + y^2 = 7 ∨ (x + a)^2 + y^2 = 7) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l621_62174


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l621_62187

theorem binomial_coefficient_divisibility (m n : ℕ) (h1 : m > 0) (h2 : n > 1) :
  (∀ k : ℕ, 1 ≤ k ∧ k < m → n ∣ Nat.choose m k) →
  ∃ (p u : ℕ), Prime p ∧ u > 0 ∧ m = p^u ∧ n = p :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l621_62187


namespace NUMINAMATH_CALUDE_child_money_distribution_l621_62115

/-- Prove that for three children with shares in the ratio 2:3:4, 
    where the second child's share is $300, the total amount is $900. -/
theorem child_money_distribution (a b c : ℕ) : 
  a + b + c = 9 ∧ 
  2 * b = 3 * a ∧ 
  4 * b = 3 * c ∧ 
  b = 300 → 
  a + b + c = 900 := by
sorry

end NUMINAMATH_CALUDE_child_money_distribution_l621_62115


namespace NUMINAMATH_CALUDE_cookies_for_lunch_is_five_l621_62138

/-- Calculates the number of cookies needed to reach the target calorie count for lunch -/
def cookiesForLunch (totalCalories burgerCalories carrotCalories cookieCalories : ℕ) 
                    (numCarrots : ℕ) : ℕ :=
  let remainingCalories := totalCalories - burgerCalories - (carrotCalories * numCarrots)
  remainingCalories / cookieCalories

/-- Proves that the number of cookies each kid gets is 5 -/
theorem cookies_for_lunch_is_five :
  cookiesForLunch 750 400 20 50 5 = 5 := by
  sorry

#eval cookiesForLunch 750 400 20 50 5

end NUMINAMATH_CALUDE_cookies_for_lunch_is_five_l621_62138


namespace NUMINAMATH_CALUDE_converse_parallel_supplementary_true_converse_vertical_angles_false_converse_squares_equal_false_converse_sum_squares_positive_false_only_parallel_supplementary_has_true_converse_l621_62181

-- Define the concept of vertical angles
def vertical_angles (a b : Angle) : Prop := sorry

-- Define the concept of consecutive interior angles
def consecutive_interior_angles (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary (a b : Angle) : Prop := sorry

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem converse_parallel_supplementary_true :
  ∀ (l1 l2 : Line) (a b : Angle),
    parallel l1 l2 → consecutive_interior_angles a b l1 l2 → supplementary a b := by sorry

theorem converse_vertical_angles_false :
  ∃ (a b : Angle), a = b ∧ ¬(vertical_angles a b) := by sorry

theorem converse_squares_equal_false :
  ∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b := by sorry

theorem converse_sum_squares_positive_false :
  ∃ (a b : ℝ), a^2 + b^2 > 0 ∧ (a ≤ 0 ∨ b ≤ 0) := by sorry

theorem only_parallel_supplementary_has_true_converse :
  (∀ (l1 l2 : Line) (a b : Angle),
    parallel l1 l2 → consecutive_interior_angles a b l1 l2 → supplementary a b) ∧
  (∃ (a b : Angle), a = b ∧ ¬(vertical_angles a b)) ∧
  (∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b) ∧
  (∃ (a b : ℝ), a^2 + b^2 > 0 ∧ (a ≤ 0 ∨ b ≤ 0)) := by sorry

end NUMINAMATH_CALUDE_converse_parallel_supplementary_true_converse_vertical_angles_false_converse_squares_equal_false_converse_sum_squares_positive_false_only_parallel_supplementary_has_true_converse_l621_62181


namespace NUMINAMATH_CALUDE_zoo_open_hours_proof_l621_62113

/-- The number of hours the zoo is open in one day -/
def zoo_open_hours : ℕ := 8

/-- The number of new visitors entering the zoo every hour -/
def visitors_per_hour : ℕ := 50

/-- The percentage of total visitors who go to the gorilla exhibit -/
def gorilla_exhibit_percentage : ℚ := 80 / 100

/-- The number of visitors who go to the gorilla exhibit in one day -/
def gorilla_exhibit_visitors : ℕ := 320

/-- Theorem stating that the zoo is open for 8 hours given the conditions -/
theorem zoo_open_hours_proof :
  zoo_open_hours * visitors_per_hour * gorilla_exhibit_percentage = gorilla_exhibit_visitors :=
by sorry

end NUMINAMATH_CALUDE_zoo_open_hours_proof_l621_62113


namespace NUMINAMATH_CALUDE_john_completion_time_l621_62143

/-- The number of days it takes Jane to complete the task alone -/
def jane_days : ℝ := 12

/-- The total number of days it took to complete the task -/
def total_days : ℝ := 10.8

/-- The number of days Jane was indisposed before the work was completed -/
def jane_indisposed : ℝ := 6

/-- The number of days it takes John to complete the task alone -/
def john_days : ℝ := 18

theorem john_completion_time :
  (jane_indisposed / john_days) + 
  ((total_days - jane_indisposed) * (1 / john_days + 1 / jane_days)) = 1 :=
sorry

#check john_completion_time

end NUMINAMATH_CALUDE_john_completion_time_l621_62143


namespace NUMINAMATH_CALUDE_germination_expectation_l621_62169

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.8

/-- The number of seeds sown -/
def seeds_sown : ℕ := 100

/-- The expected number of germinated seeds -/
def expected_germinated_seeds : ℝ := germination_rate * seeds_sown

theorem germination_expectation :
  expected_germinated_seeds = 80 := by sorry

end NUMINAMATH_CALUDE_germination_expectation_l621_62169


namespace NUMINAMATH_CALUDE_max_k_no_intersection_l621_62159

noncomputable def f (x : ℝ) : ℝ := x - 1 + (Real.exp x)⁻¹

theorem max_k_no_intersection : 
  (∃ k : ℝ, ∀ x : ℝ, f x ≠ k * x - 1) ∧ 
  (∀ k : ℝ, k > 1 → ∃ x : ℝ, f x = k * x - 1) :=
sorry

end NUMINAMATH_CALUDE_max_k_no_intersection_l621_62159


namespace NUMINAMATH_CALUDE_area_right_triangle_45_deg_l621_62185

theorem area_right_triangle_45_deg (a : ℝ) (h1 : a = 8) (h2 : a > 0) : 
  (1 / 2 : ℝ) * a * a = 32 := by
sorry

end NUMINAMATH_CALUDE_area_right_triangle_45_deg_l621_62185


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l621_62133

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five : 
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l621_62133


namespace NUMINAMATH_CALUDE_function_range_l621_62116

theorem function_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = |x - 3| + |x - a|) →
  (∀ x : ℝ, f x ≥ 4) →
  (a ≤ -1 ∨ a ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l621_62116


namespace NUMINAMATH_CALUDE_some_ai_in_machines_l621_62128

-- Define the sets
variable (Robot : Type) -- Set of all robots
variable (Machine : Type) -- Set of all machines
variable (AdvancedAI : Type) -- Set of all advanced AI systems

-- Define the relations
variable (has_ai : Robot → AdvancedAI → Prop) -- Relation: robot has advanced AI
variable (is_machine : Robot → Machine → Prop) -- Relation: robot is a machine

-- State the theorem
theorem some_ai_in_machines 
  (h1 : ∀ (r : Robot), ∃ (ai : AdvancedAI), has_ai r ai) -- All robots have advanced AI
  (h2 : ∃ (r : Robot) (m : Machine), is_machine r m) -- Some robots are machines
  : ∃ (ai : AdvancedAI) (m : Machine), 
    ∃ (r : Robot), has_ai r ai ∧ is_machine r m :=
by sorry

end NUMINAMATH_CALUDE_some_ai_in_machines_l621_62128


namespace NUMINAMATH_CALUDE_f_eq_g_g_is_right_shift_f_is_right_shift_of_x_squared_l621_62188

/-- The original quadratic function -/
def f (x : ℝ) := x^2 - 2*x + 1

/-- The shifted quadratic function -/
def g (x : ℝ) := (x - 1)^2

/-- Theorem stating that f and g are equivalent -/
theorem f_eq_g : ∀ x, f x = g x := by sorry

/-- Theorem stating that g is a right shift of x^2 by 1 unit -/
theorem g_is_right_shift : ∀ x, g x = (x - 1)^2 := by sorry

/-- Main theorem: f is a right shift of x^2 by 1 unit -/
theorem f_is_right_shift_of_x_squared : 
  ∃ h : ℝ, h > 0 ∧ (∀ x, f x = (x - h)^2) := by sorry

end NUMINAMATH_CALUDE_f_eq_g_g_is_right_shift_f_is_right_shift_of_x_squared_l621_62188


namespace NUMINAMATH_CALUDE_f_lower_bound_l621_62189

noncomputable section

def f (x t : ℝ) : ℝ := ((x + t) / (x - 1)) * Real.exp (x - 1)

theorem f_lower_bound (x t : ℝ) (hx : x > 1) (ht : t > -1) :
  f x t > Real.sqrt x * (1 + (1/2) * Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l621_62189


namespace NUMINAMATH_CALUDE_coach_spending_difference_l621_62155

-- Define the purchases and discounts for each coach
def coach_A_basketballs : Nat := 10
def coach_A_basketball_price : ℝ := 29
def coach_A_soccer_balls : Nat := 5
def coach_A_soccer_ball_price : ℝ := 15
def coach_A_discount : ℝ := 0.05

def coach_B_baseballs : Nat := 14
def coach_B_baseball_price : ℝ := 2.50
def coach_B_baseball_bats : Nat := 1
def coach_B_baseball_bat_price : ℝ := 18
def coach_B_hockey_sticks : Nat := 4
def coach_B_hockey_stick_price : ℝ := 25
def coach_B_hockey_masks : Nat := 1
def coach_B_hockey_mask_price : ℝ := 72
def coach_B_discount : ℝ := 0.10

def coach_C_volleyball_nets : Nat := 8
def coach_C_volleyball_net_price : ℝ := 32
def coach_C_volleyballs : Nat := 12
def coach_C_volleyball_price : ℝ := 12
def coach_C_discount : ℝ := 0.07

-- Define the theorem
theorem coach_spending_difference :
  let coach_A_total := (1 - coach_A_discount) * (coach_A_basketballs * coach_A_basketball_price + coach_A_soccer_balls * coach_A_soccer_ball_price)
  let coach_B_total := (1 - coach_B_discount) * (coach_B_baseballs * coach_B_baseball_price + coach_B_baseball_bats * coach_B_baseball_bat_price + coach_B_hockey_sticks * coach_B_hockey_stick_price + coach_B_hockey_masks * coach_B_hockey_mask_price)
  let coach_C_total := (1 - coach_C_discount) * (coach_C_volleyball_nets * coach_C_volleyball_net_price + coach_C_volleyballs * coach_C_volleyball_price)
  coach_A_total - (coach_B_total + coach_C_total) = -227.75 := by
  sorry

end NUMINAMATH_CALUDE_coach_spending_difference_l621_62155


namespace NUMINAMATH_CALUDE_pattern_D_cannot_fold_into_cube_only_pattern_D_cannot_fold_into_cube_l621_62190

-- Define a type for the patterns
inductive Pattern : Type
  | A : Pattern
  | B : Pattern
  | C : Pattern
  | D : Pattern

-- Define a predicate to check if a pattern can be folded into a cube
def can_fold_into_cube (p : Pattern) : Prop :=
  match p with
  | Pattern.A => true
  | Pattern.B => true
  | Pattern.C => true
  | Pattern.D => false

-- Theorem stating that Pattern D cannot be folded into a cube
theorem pattern_D_cannot_fold_into_cube :
  ¬(can_fold_into_cube Pattern.D) :=
by sorry

-- Theorem stating that Pattern D is the only pattern that cannot be folded into a cube
theorem only_pattern_D_cannot_fold_into_cube :
  ∀ (p : Pattern), ¬(can_fold_into_cube p) ↔ p = Pattern.D :=
by sorry

end NUMINAMATH_CALUDE_pattern_D_cannot_fold_into_cube_only_pattern_D_cannot_fold_into_cube_l621_62190


namespace NUMINAMATH_CALUDE_probability_of_two_as_median_l621_62156

def S : Finset ℕ := {2, 0, 1, 5}

def is_median (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c) ∨ (c ≤ b ∧ b ≤ a)

def favorable_outcomes : Finset (ℕ × ℕ × ℕ) :=
  {(0, 2, 5), (1, 2, 5)}

def total_outcomes : Finset (ℕ × ℕ × ℕ) :=
  {(0, 1, 2), (0, 1, 5), (0, 2, 5), (1, 2, 5)}

theorem probability_of_two_as_median :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card total_outcomes : ℚ) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_as_median_l621_62156


namespace NUMINAMATH_CALUDE_ava_lily_trees_l621_62119

/-- The number of apple trees planted by Ava and Lily -/
def total_trees (ava_trees lily_trees : ℕ) : ℕ :=
  ava_trees + lily_trees

/-- Theorem stating the total number of apple trees planted by Ava and Lily -/
theorem ava_lily_trees :
  ∀ (ava_trees lily_trees : ℕ),
    ava_trees = 9 →
    ava_trees = lily_trees + 3 →
    total_trees ava_trees lily_trees = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ava_lily_trees_l621_62119


namespace NUMINAMATH_CALUDE_opposite_sides_line_condition_l621_62194

theorem opposite_sides_line_condition (a : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = -1 ∧ y2 = -4 ∧ 
    ((a * x1 + 3 * y1 + 1) * (a * x2 + 3 * y2 + 1) < 0)) ↔ 
  (a < -11 ∨ a > -10) := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_condition_l621_62194


namespace NUMINAMATH_CALUDE_joanna_money_problem_l621_62106

/-- Joanna's money problem -/
theorem joanna_money_problem (J : ℚ) 
  (h1 : J + 3 * J + J / 2 = 36) : J = 8 := by
  sorry

end NUMINAMATH_CALUDE_joanna_money_problem_l621_62106


namespace NUMINAMATH_CALUDE_horners_first_step_l621_62101

-- Define the polynomial coefficients
def a₅ : ℝ := 0.5
def a₄ : ℝ := 4
def a₃ : ℝ := 0
def a₂ : ℝ := -3
def a₁ : ℝ := 1
def a₀ : ℝ := -1

-- Define the polynomial
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Define the point at which to evaluate the polynomial
def x : ℝ := 3

-- State the theorem
theorem horners_first_step :
  a₅ * x + a₄ = 5.5 :=
sorry

end NUMINAMATH_CALUDE_horners_first_step_l621_62101


namespace NUMINAMATH_CALUDE_integer_x_is_seven_l621_62196

theorem integer_x_is_seven (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 8)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_x_is_seven_l621_62196


namespace NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l621_62117

theorem no_integer_cube_equals_3n2_plus_3n_plus_7 :
  ¬ ∃ (x n : ℤ), x^3 = 3*n^2 + 3*n + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l621_62117


namespace NUMINAMATH_CALUDE_sin_shift_l621_62154

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 4) + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l621_62154


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l621_62126

theorem solution_to_linear_equation :
  ∃ (x y : ℤ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l621_62126


namespace NUMINAMATH_CALUDE_graveling_cost_l621_62103

/-- The cost of graveling two intersecting roads on a rectangular lawn -/
theorem graveling_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 4 →
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * cost_per_sqm = 5200 := by
  sorry

end NUMINAMATH_CALUDE_graveling_cost_l621_62103


namespace NUMINAMATH_CALUDE_range_of_a_l621_62112

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 2

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc a 3, f x ∈ Set.Icc (-1) 3) ∧
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a 3, f x = y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l621_62112


namespace NUMINAMATH_CALUDE_probability_three_black_balls_l621_62121

theorem probability_three_black_balls (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + black_balls + white_balls →
  red_balls = 10 →
  black_balls = 8 →
  white_balls = 3 →
  (Nat.choose black_balls 3 : ℚ) / (Nat.choose total_balls 3 : ℚ) = 4 / 95 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_black_balls_l621_62121


namespace NUMINAMATH_CALUDE_john_taxes_l621_62165

/-- Calculate the total tax given a progressive tax system and taxable income -/
def calculate_tax (taxable_income : ℕ) : ℕ :=
  let tax1 := min taxable_income 20000 * 10 / 100
  let tax2 := min (max (taxable_income - 20000) 0) 30000 * 15 / 100
  let tax3 := min (max (taxable_income - 50000) 0) 50000 * 20 / 100
  let tax4 := max (taxable_income - 100000) 0 * 25 / 100
  tax1 + tax2 + tax3 + tax4

/-- John's financial situation -/
theorem john_taxes :
  let main_job := 75000
  let freelance := 25000
  let rental := 15000
  let dividends := 10000
  let mortgage_deduction := 32000
  let retirement_deduction := 15000
  let charitable_deduction := 10000
  let education_credit := 3000
  let total_income := main_job + freelance + rental + dividends
  let total_deductions := mortgage_deduction + retirement_deduction + charitable_deduction + education_credit
  let taxable_income := total_income - total_deductions
  taxable_income = 65000 ∧ calculate_tax taxable_income = 9500 := by
  sorry


end NUMINAMATH_CALUDE_john_taxes_l621_62165


namespace NUMINAMATH_CALUDE_pencil_distribution_l621_62100

theorem pencil_distribution (total_pencils : Nat) (pencils_per_box : Nat) : 
  total_pencils = 48297858 → pencils_per_box = 6 → total_pencils % pencils_per_box = 0 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l621_62100


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l621_62177

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l621_62177


namespace NUMINAMATH_CALUDE_timi_ears_count_l621_62172

-- Define the inhabitants
structure Inhabitant where
  name : String
  ears_seen : Nat

-- Define the problem setup
def zog_problem : List Inhabitant :=
  [{ name := "Imi", ears_seen := 8 },
   { name := "Dimi", ears_seen := 7 },
   { name := "Timi", ears_seen := 5 }]

-- Theorem: Timi has 5 ears
theorem timi_ears_count (problem : List Inhabitant) : 
  problem = zog_problem → 
  (problem.find? (fun i => i.name = "Timi")).map (fun i => 
    List.sum (problem.map (fun j => j.ears_seen)) / 2 - i.ears_seen) = some 5 := by
  sorry

end NUMINAMATH_CALUDE_timi_ears_count_l621_62172


namespace NUMINAMATH_CALUDE_infinitely_many_composites_l621_62168

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def sequence_property (p : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, last_digit (p (i + 1)) ≠ 9 ∧ remove_last_digit (p (i + 1)) = p i

theorem infinitely_many_composites (p : ℕ → ℕ) (h : sequence_property p) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_composite (p n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composites_l621_62168


namespace NUMINAMATH_CALUDE_sum_of_decimals_l621_62105

def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_2 : ℚ := 2/9

theorem sum_of_decimals : 
  repeating_decimal_6 - repeating_decimal_2 + (1/4 : ℚ) = 25/36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l621_62105


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l621_62129

theorem chocolate_gain_percent :
  ∀ (C S : ℝ),
  C > 0 →
  S > 0 →
  24 * C = 16 * S →
  (S - C) / C * 100 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l621_62129


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l621_62137

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l621_62137


namespace NUMINAMATH_CALUDE_part1_part2_l621_62198

/-- Defines the sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℤ
| 0 => 3  -- We define a₀ = 3 to match a₁ = 3 in the original problem
| n + 1 => 2 * a n + n^2 - 4*n + 1

/-- The arithmetic sequence b_n -/
def b (n : ℕ) : ℤ := -2*n + 3

theorem part1 : ∀ n : ℕ, a n = 2^n - n^2 + 2*n := by sorry

theorem part2 (h : ∀ n : ℕ, a n = (n + 1) * b (n + 1) - n * b n) : 
  a 0 = 1 ∧ ∀ n : ℕ, b n = -2*n + 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l621_62198


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l621_62179

theorem triangle_third_side_length 
  (a b : ℝ) 
  (angle : ℝ) 
  (ha : a = 9) 
  (hb : b = 8) 
  (hangle : angle = 150 * π / 180) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*(Real.cos angle) ∧ 
            c = Real.sqrt (145 + 72 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l621_62179


namespace NUMINAMATH_CALUDE_mike_initial_marbles_l621_62109

/-- The number of marbles Mike gave to Sam -/
def marbles_given : ℕ := 4

/-- The number of marbles Mike has left -/
def marbles_left : ℕ := 4

/-- The initial number of marbles Mike had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem mike_initial_marbles : initial_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_mike_initial_marbles_l621_62109


namespace NUMINAMATH_CALUDE_division_multiplication_example_l621_62182

theorem division_multiplication_example : (180 / 6) * 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_example_l621_62182


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l621_62195

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 2
def l (x y : ℝ) : Prop := x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := (x-2)^2 + (y-4)^2 = 20

-- Define the ray
def ray (x y : ℝ) : Prop := 2*x - y = 0 ∧ x ≥ 0

-- Theorem statement
theorem circle_and_line_problem :
  -- Given conditions
  (∀ x y, C₁ x y → l x y → (x = 1 ∧ y = 1)) →  -- l is tangent to C₁ at (1,1)
  (∃ a b, ray a b ∧ ∀ x y, C₂ x y → (x - a)^2 + (y - b)^2 = (x^2 + y^2)) →  -- Center of C₂ is on the ray and C₂ passes through origin
  (∃ x₁ y₁ x₂ y₂, C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 48) →  -- Chord length is 4√3
  -- Conclusion
  (∀ x y, l x y ↔ x + y - 2 = 0) ∧
  (∀ x y, C₂ x y ↔ (x-2)^2 + (y-4)^2 = 20) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l621_62195


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_bound_l621_62102

/-- A triangle in a 2D plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a triangle --/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a rectangle --/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle is inscribed in a triangle --/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Theorem: The area of a rectangle inscribed in a triangle does not exceed half the area of the triangle --/
theorem inscribed_rectangle_area_bound (t : Triangle) (r : Rectangle) :
  isInscribed r t → rectangleArea r ≤ (1/2) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_bound_l621_62102


namespace NUMINAMATH_CALUDE_kristy_ate_two_cookies_l621_62114

def cookies_problem (total_baked : ℕ) (brother_took : ℕ) (friend1_took : ℕ) (friend2_took : ℕ) (friend3_took : ℕ) (cookies_left : ℕ) : Prop :=
  total_baked = 22 ∧
  brother_took = 1 ∧
  friend1_took = 3 ∧
  friend2_took = 5 ∧
  friend3_took = 5 ∧
  cookies_left = 6 ∧
  total_baked - (brother_took + friend1_took + friend2_took + friend3_took + cookies_left) = 2

theorem kristy_ate_two_cookies :
  ∀ (total_baked brother_took friend1_took friend2_took friend3_took cookies_left : ℕ),
  cookies_problem total_baked brother_took friend1_took friend2_took friend3_took cookies_left →
  total_baked - (brother_took + friend1_took + friend2_took + friend3_took + cookies_left) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_kristy_ate_two_cookies_l621_62114


namespace NUMINAMATH_CALUDE_joan_total_games_l621_62164

/-- The total number of football games Joan attended over two years -/
def total_games (this_year last_year : ℕ) : ℕ :=
  this_year + last_year

/-- Theorem: Joan attended 9 football games in total over two years -/
theorem joan_total_games : total_games 4 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_games_l621_62164


namespace NUMINAMATH_CALUDE_complex_number_problem_l621_62158

theorem complex_number_problem (m : ℝ) (z z₁ : ℂ) :
  z₁ = m * (m - 1) + (m - 1) * Complex.I ∧
  z₁.re = 0 ∧
  z₁.im ≠ 0 ∧
  (3 + z₁) * z = 4 + 2 * Complex.I →
  m = 0 ∧ z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l621_62158


namespace NUMINAMATH_CALUDE_remaining_packs_eq_26_l621_62186

/-- The number of cookie packs Tory needs to sell -/
def total_goal : ℕ := 50

/-- The number of cookie packs Tory sold to his grandmother -/
def sold_to_grandmother : ℕ := 12

/-- The number of cookie packs Tory sold to his uncle -/
def sold_to_uncle : ℕ := 7

/-- The number of cookie packs Tory sold to a neighbor -/
def sold_to_neighbor : ℕ := 5

/-- The number of remaining cookie packs Tory needs to sell -/
def remaining_packs : ℕ := total_goal - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor)

theorem remaining_packs_eq_26 : remaining_packs = 26 := by
  sorry

end NUMINAMATH_CALUDE_remaining_packs_eq_26_l621_62186


namespace NUMINAMATH_CALUDE_set_difference_equiv_l621_62149

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x) ∧ -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

theorem set_difference_equiv : A \ B = {x | x < 0} := by sorry

end NUMINAMATH_CALUDE_set_difference_equiv_l621_62149


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_cone_l621_62173

theorem lateral_surface_area_of_cone (slant_height base_radius : ℝ) 
  (h1 : slant_height = 4)
  (h2 : base_radius = 2) :
  (1/2) * slant_height * (2 * Real.pi * base_radius) = 8 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_cone_l621_62173


namespace NUMINAMATH_CALUDE_sum_of_specific_S_values_l621_62166

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S_values : S 17 + S 33 + S 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_values_l621_62166
