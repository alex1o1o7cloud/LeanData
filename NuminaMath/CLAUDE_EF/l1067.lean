import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l1067_106725

/-- Two lines in a plane -/
structure Lines where
  m : ℝ
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  h₁ : ∀ x y, l₁ x y ↔ x + m * y + 1 = 0
  h₂ : ∀ x y, l₂ x y ↔ (m - 3) * x - 2 * y + (13 - 7 * m) = 0

/-- Perpendicular lines -/
def perpendicular (L : Lines) : Prop :=
  1 * (L.m - 3) - 2 * L.m = 0

/-- Parallel lines -/
def parallel (L : Lines) : Prop :=
  L.m * (L.m - 3) + 2 = 0

/-- Distance between parallel lines -/
noncomputable def distance (L : Lines) : ℝ :=
  |(-3 - 1)| / Real.sqrt (1^2 + 1^2)

theorem lines_theorem (L : Lines) :
  (perpendicular L → L.m = -3) ∧
  (parallel L → distance L = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l1067_106725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1067_106772

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 2 * y) - f (3 * x - 2 * y) = 2 * y - x

-- State the theorem
theorem expression_value (f : ℝ → ℝ) (h : satisfies_property f) :
  ∀ t : ℝ, f (4 * t) - f (3 * t) ≠ 0 →
    (f (5 * t) - f t) / (f (4 * t) - f (3 * t)) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1067_106772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1067_106796

theorem quartic_equation_solutions :
  {z : ℂ | z^4 - 8*z^2 + 15 = 0} = {-Complex.I * Real.sqrt 5, -Complex.I * Real.sqrt 3, Complex.I * Real.sqrt 3, Complex.I * Real.sqrt 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1067_106796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_25_not_multiple_of_15_l1067_106723

def numbers : List ℕ := [150, 25, 30, 45, 60]

theorem only_25_not_multiple_of_15 : 
  ∃! n, n ∈ numbers ∧ ¬(∃ k : ℕ, n = 15 * k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_25_not_multiple_of_15_l1067_106723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_triangle_area_l1067_106762

/-- Two perpendicular lines intersecting at A(8,6) with y-intercepts summing to 4 form a triangle APQ of area 80 --/
theorem perpendicular_lines_triangle_area :
  ∀ (m₁ m₂ b₁ b₂ : ℝ),
  m₁ * m₂ = -1 →  -- perpendicular lines condition
  6 = 8 * m₁ + b₁ →  -- line 1 passes through A(8,6)
  6 = 8 * m₂ + b₂ →  -- line 2 passes through A(8,6)
  b₁ + b₂ = 4 →  -- sum of y-intercepts
  let P : ℝ × ℝ := (0, b₁)
  let Q : ℝ × ℝ := (0, b₂)
  let A : ℝ × ℝ := (8, 6)
  (1/2) * 8 * |b₂ - b₁| = 80 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_triangle_area_l1067_106762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_equals_16_l1067_106719

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^a else |x - 2|

-- State the theorem
theorem f_4_equals_16 (a : ℝ) :
  (f a (-2) = f a 2) → f a 4 = 16 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_equals_16_l1067_106719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1067_106792

/-- The point we want to prove is equidistant from the given points -/
noncomputable def P : Fin 3 → ℝ := ![35/16, 25/16, 0]

/-- The three given points -/
noncomputable def A : Fin 3 → ℝ := ![1, -2, 0]
noncomputable def B : Fin 3 → ℝ := ![0, 1, 3]
noncomputable def C : Fin 3 → ℝ := ![4, 0, -2]

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2)

/-- Theorem stating that P is equidistant from A, B, and C -/
theorem equidistant_point : 
  distance P A = distance P B ∧ distance P A = distance P C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1067_106792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goal_count_possibilities_l1067_106743

def is_valid_goal_count (x : ℕ) : Prop :=
  (x > 10 ∧ x < 17) ∨ (x > 11 ∧ x < 18) ∨ x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (is_valid_goal_count x) ∧
  (((x > 10 ∧ x < 17) ∧ (x > 11 ∧ x < 18) ∧ ¬(x % 2 = 1)) ∨
   ((x > 10 ∧ x < 17) ∧ ¬(x > 11 ∧ x < 18) ∧ (x % 2 = 1)) ∨
   (¬(x > 10 ∧ x < 17) ∧ (x > 11 ∧ x < 18) ∧ (x % 2 = 1)))

theorem goal_count_possibilities :
  ∀ x : ℕ, exactly_two_correct x ↔ x ∈ ({11, 12, 14, 16, 17} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goal_count_possibilities_l1067_106743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1067_106734

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope k passing through point (x₀, y₀) -/
structure Line where
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: Given an ellipse with eccentricity 1/2 and left focus at (-1, 0),
    if a line passing through the left focus intersects the ellipse at points A and B,
    intersects the y-axis at point C on line segment AB, and |AF₁| = |CB|,
    then the equation of the ellipse is x²/4 + y²/3 = 1,
    and the equation of line l is either y = (√3/2)(x+1) or y = -(√3/2)(x+1). -/
theorem ellipse_and_line_theorem (G : Ellipse) (l : Line) :
  eccentricity G = 1/2 →
  focal_distance G = 1 →
  l.x₀ = -1 ∧ l.y₀ = 0 →
  ∃ (A B C : ℝ × ℝ),
    (A.1^2 / G.a^2 + A.2^2 / G.b^2 = 1) ∧
    (B.1^2 / G.a^2 + B.2^2 / G.b^2 = 1) ∧
    (C.2 = l.k * (C.1 + 1)) ∧
    (C.1 = 0) ∧
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1-t) • A + t • B) ∧
    (Real.sqrt ((A.1 + 1)^2 + A.2^2) = Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) →
  (G.a = 2 ∧ G.b = Real.sqrt 3) ∧
  (l.k = Real.sqrt 3 / 2 ∨ l.k = -Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1067_106734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1067_106701

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm,
    and a height of 15 cm, is 285 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 15 = 285 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1067_106701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_theorem_l1067_106754

/-- Calculates the number of minutes a bus stops per hour given its average speeds with and without stoppages. -/
noncomputable def bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : ℝ :=
  60 * (speed_without_stops - speed_with_stops) / speed_without_stops

/-- Theorem stating that a bus with given average speeds stops for 40 minutes per hour. -/
theorem bus_stop_theorem (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 60)
  (h2 : speed_with_stops = 20) :
  bus_stop_time speed_without_stops speed_with_stops = 40 := by
  sorry

/-- Computable version for evaluation -/
def bus_stop_time_rat (speed_without_stops : ℚ) (speed_with_stops : ℚ) : ℚ :=
  60 * (speed_without_stops - speed_with_stops) / speed_without_stops

#eval bus_stop_time_rat 60 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_theorem_l1067_106754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_x_intercepts_eq_5730_l1067_106730

/-- The number of x-intercepts for y = sin(1/x) in the interval (0.00005, 0.0005) -/
noncomputable def num_x_intercepts : ℕ :=
  (⌊(20000 : ℝ) / Real.pi⌋ - ⌊(2000 : ℝ) / Real.pi⌋).toNat

/-- Theorem stating that the number of x-intercepts is 5730 -/
theorem num_x_intercepts_eq_5730 : num_x_intercepts = 5730 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_x_intercepts_eq_5730_l1067_106730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1067_106736

noncomputable def f (x : ℝ) := 2 * Real.sin (x - Real.pi/6) * Real.sin (x + Real.pi/3)

theorem triangle_ratio (C : ℝ) (h1 : 0 < C) (h2 : C < Real.pi/2) 
  (h3 : f (C/2 + Real.pi/6) = 1/2) : 
  Real.sin (Real.pi/4) / Real.sin C = Real.sqrt 2 := by
  sorry

#check triangle_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1067_106736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_b_value_l1067_106733

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
noncomputable def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
noncomputable def ratio_a : ℝ := 3
noncomputable def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
noncomputable def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
noncomputable def total_weight : ℝ := 3520

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
noncomputable def weight_b : ℝ := (total_weight - ratio_a * weight_a) / ratio_b

theorem weight_b_value : weight_b = 410 := by
  -- Unfold the definition of weight_b
  unfold weight_b
  -- Perform the calculation
  simp [total_weight, ratio_a, weight_a, ratio_b]
  -- The proof is complete
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_b_value_l1067_106733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_three_l1067_106769

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x + 3) / (2^x - 1)

-- Define the property of being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- Theorem statement
theorem odd_function_implies_a_equals_three :
  ∀ a : ℝ, is_odd_function (f a) → a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_three_l1067_106769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_after_rotations_l1067_106708

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the diagonal length of a rectangle using the Pythagorean theorem -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.a^2 + r.b^2)

/-- Calculates the arc length of a 90° rotation with given radius -/
noncomputable def arcLength90 (radius : ℝ) : ℝ :=
  (Real.pi / 2) * radius

/-- Theorem: The length of the path traveled by point B after two 90° rotations is 4π -/
theorem path_length_after_rotations (r : Rectangle) (h1 : r.a = 3) (h2 : r.b = 4) :
  arcLength90 r.diagonal + arcLength90 r.a = 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_after_rotations_l1067_106708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1067_106765

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, t - 2)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ₁ : ℝ, circle_C θ₁ = A) ∧
    (∃ θ₂ : ℝ, circle_C θ₂ = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1067_106765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1067_106737

-- Define the ellipse equation
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  y = x + 3

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

-- Theorem statement
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃! p : ℝ × ℝ, ellipse_equation p.1 p.2 a b ∧ line_equation p.1 p.2) →
  eccentricity a b = Real.sqrt 5 / 5 →
  (∀ x y : ℝ, ellipse_equation x y a b ↔ x^2 / 5 + y^2 / 4 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1067_106737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_equal_area_division_l1067_106789

/-- An isosceles triangle with two sides of length 8.5 and one side of length 5 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b ∧ a = 8.5 ∧ c = 5

/-- A point inside the triangle -/
structure InnerPoint (t : IsoscelesTriangle) where
  x : ℝ
  y : ℝ
  inside : True  -- Placeholder condition, replace with actual condition when implementing

/-- Perpendicular distances from the inner point to the sides of the triangle -/
noncomputable def perpendicular_distances (t : IsoscelesTriangle) (p : InnerPoint t) : ℝ × ℝ × ℝ :=
  sorry

/-- The areas of the three subtriangles formed by the perpendiculars -/
noncomputable def subtriangle_areas (t : IsoscelesTriangle) (p : InnerPoint t) : ℝ × ℝ × ℝ :=
  sorry

/-- The theorem statement -/
theorem isosceles_triangle_equal_area_division (t : IsoscelesTriangle) :
  ∃ (p : InnerPoint t),
    let (a1, a2, a3) := subtriangle_areas t p
    let (d1, d2, d3) := perpendicular_distances t p
    a1 = a2 ∧ a2 = a3 ∧
    d1 = 4 * Real.sqrt 3 / 3 ∧
    d2 = 4 * Real.sqrt 3 / 3 ∧
    d3 = (9 - 5 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_equal_area_division_l1067_106789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_22_l1067_106712

/-- An arithmetic sequence with given first and fourth terms -/
structure ArithmeticSequence where
  first_term : ℚ
  fourth_term : ℚ

/-- The seventh term of an arithmetic sequence -/
def seventh_term (seq : ArithmeticSequence) : ℚ :=
  let d := (seq.fourth_term - seq.first_term) / 3
  seq.first_term + 6 * d

/-- Theorem: The seventh term of the specific arithmetic sequence is 22 -/
theorem seventh_term_is_22 (seq : ArithmeticSequence) 
    (h1 : seq.first_term = 10)
    (h2 : seq.fourth_term = 16) : 
  seventh_term seq = 22 := by
  sorry

#eval seventh_term { first_term := 10, fourth_term := 16 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_22_l1067_106712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4alpha_value_l1067_106797

theorem tan_4alpha_value (α : Real) 
  (h1 : Real.sin (π/4 + α) * Real.sin (π/4 - α) = 1/6)
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (4*α) = 4*Real.sqrt 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4alpha_value_l1067_106797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_tails_probability_l1067_106783

/-- The probability of getting exactly k successes in n trials of a binomial experiment -/
noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def num_flips : ℕ := 10

/-- The number of tails we're interested in -/
def num_tails : ℕ := 3

/-- The probability of getting tails on a single flip -/
noncomputable def prob_tails : ℝ := 2/3

theorem exactly_three_tails_probability :
  binomial_probability num_flips num_tails prob_tails = 960/6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_tails_probability_l1067_106783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_228_l1067_106784

/-- A convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  qe : ℝ
  qf : ℝ
  qg : ℝ
  qh : ℝ
  convex : Bool
  interior_q : Bool
  bisecting_diag : Bool
  h_area : area = 2500
  h_qe : qe = 30
  h_qf : qf = 40
  h_qg : qg = 35
  h_qh : qh = 50
  h_convex : convex = true
  h_interior_q : interior_q = true
  h_bisecting_diag : bisecting_diag = true

/-- Calculate the perimeter of the quadrilateral -/
noncomputable def perimeter (q : ConvexQuadrilateral) : ℝ :=
  let ef := Real.sqrt (q.qe^2 + q.qf^2)
  let fg := Real.sqrt (q.qf^2 + q.qg^2)
  let gh := Real.sqrt (q.qg^2 + q.qh^2)
  let he := Real.sqrt (q.qh^2 + q.qe^2)
  ef + fg + gh + he

/-- Theorem stating that the perimeter of the quadrilateral is 228 -/
theorem perimeter_is_228 (q : ConvexQuadrilateral) : perimeter q = 228 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_228_l1067_106784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_evaluation_at_zero_l1067_106768

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := 
  (x - 1) / (x^2 - 1) / ((x - 2) / (x^2 + 2*x + 1)) - 2*x / (x - 2)

-- Define the simplified expression
noncomputable def g (x : ℝ) : ℝ := (1 - x) / (x - 2)

-- State the theorem
theorem expression_simplification {x : ℝ} (h1 : -1 ≤ x) (h2 : x ≤ 2) 
  (h3 : x ≠ -1) (h4 : x ≠ 1) (h5 : x ≠ 2) : 
  f x = g x := by
  sorry

-- Evaluate at x = 0
theorem evaluation_at_zero : g 0 = -1/2 := by
  simp [g]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_evaluation_at_zero_l1067_106768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_five_l1067_106717

def point_a : Fin 3 → ℝ := ![2, 0, -1]
def point_b : Fin 3 → ℝ := ![1, 3, 2]
def point_c : Fin 3 → ℝ := ![0, -1, 5]

def direction_vector : Fin 3 → ℝ := ![
  point_b 0 - point_c 0,
  point_b 1 - point_c 1,
  point_b 2 - point_c 2
]

def line_equation (t : ℝ) : Fin 3 → ℝ := ![
  point_b 0 + t * direction_vector 0,
  point_b 1 + t * direction_vector 1,
  point_b 2 + t * direction_vector 2
]

noncomputable def distance_to_line : ℝ := sorry

theorem distance_is_five : distance_to_line = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_five_l1067_106717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karan_loan_principal_l1067_106790

/-- Represents the loan details and repayment --/
structure Loan where
  rate : ℚ  -- Annual interest rate as a rational number
  time : ℚ  -- Loan duration in years
  total_repaid : ℚ  -- Total amount repaid after the loan term

/-- Calculates the initial borrowed amount given loan details --/
noncomputable def calculate_principal (loan : Loan) : ℚ :=
  loan.total_repaid / (1 + loan.rate * loan.time)

/-- Theorem stating that for the given loan conditions, the calculated principal is approximately 5461 --/
theorem karan_loan_principal :
  let loan : Loan := { rate := 6/100, time := 9, total_repaid := 8410 }
  ∃ ε > 0, |calculate_principal loan - 5461| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karan_loan_principal_l1067_106790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1067_106770

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge's length. -/
noncomputable def train_length (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_ms : ℝ := train_speed * 1000 / 3600
  let total_distance : ℝ := speed_ms * time_to_pass
  total_distance - bridge_length

/-- Theorem stating that a train traveling at 50 km/h and taking 36 seconds to pass a 140m bridge is approximately 360m long. -/
theorem train_length_calculation :
  let train_speed : ℝ := 50
  let time_to_pass : ℝ := 36
  let bridge_length : ℝ := 140
  |train_length train_speed time_to_pass bridge_length - 360| < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1067_106770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radius_l1067_106722

-- Define the Circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the necessary functions
def inscribed_in_semicircle (c s : Circle) : Prop := sorry
def circles_touch (c₁ c₂ : Circle) : Prop := sorry
def circles_touch_diameter (c s : Circle) : Prop := sorry

theorem inscribed_circles_radius (R r : ℝ) (hR : R > 0) (hr : r > 0) (hr_lt_R : r < R) :
  let x := (r * R * (3 * R - 2 * r + 2 * Real.sqrt (2 * R * (R - 2 * r)))) / ((R + 2 * r)^2)
  ∃ (c₁ c₂ s : Circle),
    s.radius = R ∧
    c₁.radius = r ∧
    c₂.radius = x ∧
    inscribed_in_semicircle c₁ s ∧
    inscribed_in_semicircle c₂ s ∧
    circles_touch c₁ c₂ ∧
    circles_touch_diameter c₁ s ∧
    circles_touch_diameter c₂ s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radius_l1067_106722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_x_distance_after_y_start_l1067_106721

/-- Represents the scenario of two cars traveling --/
structure TwoCarScenario where
  speed_x : ℝ  -- Speed of Car X in miles per hour
  speed_y : ℝ  -- Speed of Car Y in miles per hour
  head_start : ℝ  -- Head start time for Car X in hours

/-- Calculates the distance traveled by Car X after Car Y starts --/
noncomputable def distance_x_after_y_start (scenario : TwoCarScenario) : ℝ :=
  let total_time := (scenario.speed_y * scenario.head_start) / (scenario.speed_x - scenario.speed_y)
  scenario.speed_x * total_time

/-- Theorem stating that Car X travels 245 miles after Car Y starts --/
theorem car_x_distance_after_y_start :
  let scenario : TwoCarScenario := {
    speed_x := 35,
    speed_y := 39,
    head_start := 48 / 60
  }
  distance_x_after_y_start scenario = 245 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_x_distance_after_y_start_l1067_106721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_c_value_l1067_106703

/-- A cubic function with a specific derivative and maximum value. -/
def CubicFunction (a c : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + c

theorem cubic_function_c_value (a c : ℝ) :
  (∃ f : ℝ → ℝ, f = CubicFunction a c) →
  (∃ f' : ℝ → ℝ, ∀ x, f' x = deriv (CubicFunction a c) x) →
  (∃ f' : ℝ → ℝ, f' 1 = 6) →
  (∀ x ∈ Set.Icc 1 2, CubicFunction a c x ≤ 20) →
  (∃ x ∈ Set.Icc 1 2, CubicFunction a c x = 20) →
  c = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_c_value_l1067_106703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_intersection_l1067_106793

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4, 5}

theorem number_of_proper_subsets_of_intersection : 
  (Finset.powerset (A ∩ B)).card - 1 = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_intersection_l1067_106793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_theorem_l1067_106758

noncomputable def task_completion_time (time_together time_person1 : ℝ) : ℝ :=
  (time_together * time_person1) / (time_person1 - time_together)

theorem task_completion_theorem (time_together time_person1 : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_person1 > time_together)
  (h3 : time_together = 18)
  (h4 : time_person1 = 45) :
  task_completion_time time_together time_person1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_theorem_l1067_106758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_income_l1067_106714

/-- Represents the tax structure and Linda's income --/
structure TaxSystem where
  p : ℚ  -- Base tax rate as a rational number
  income : ℚ  -- Linda's annual income
  threshold : ℚ  -- Income threshold for higher tax rate
  effectiveTaxRate : ℚ  -- Linda's effective tax rate as a rational number

/-- Calculates the total tax paid given the tax system --/
def totalTax (ts : TaxSystem) : ℚ :=
  ts.p * min ts.income ts.threshold + 
  (ts.p + 3/100) * max (ts.income - ts.threshold) 0

/-- Theorem stating that Linda's income is $42000 --/
theorem linda_income (ts : TaxSystem) 
  (h1 : ts.threshold = 35000)
  (h2 : ts.effectiveTaxRate = ts.p + 5/1000)
  (h3 : totalTax ts = ts.effectiveTaxRate * ts.income) : 
  ts.income = 42000 := by
  sorry

#eval 42000  -- Expected output: 42000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_income_l1067_106714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_leftward_shift_l1067_106763

theorem min_leftward_shift :
  ∃ (shift : ℝ), shift ≥ 0 ∧
  (∀ (t : ℝ), Real.sin (2*t) + Real.cos (2*t) = Real.sqrt 2 * Real.cos (2*(t - shift))) ∧
  (∀ (s : ℝ), s ≥ 0 →
    (∀ (t : ℝ), Real.sin (2*t) + Real.cos (2*t) = Real.sqrt 2 * Real.cos (2*(t - s))) →
    s ≥ shift) ∧
  shift = Real.pi/8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_leftward_shift_l1067_106763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_2_1_part_2_2_l1067_106704

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (1/2) * a * x^2
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * x

-- Define the derivative of g
def g_prime (a : ℝ) (x : ℝ) : ℝ := a / x

theorem part_2_1 (k : ℝ) :
  (∀ x > 1, f_prime 3 x ≤ k * g 3 x) → k ≥ -1 :=
by sorry

theorem part_2_2 :
  (∃ a > 0, ∃ x₀ > 0, Real.sqrt (f_prime a x₀) ≥ g_prime a x₀) →
  (∃ a ≥ 16, ∃ x₀ > 0, Real.sqrt (f_prime a x₀) ≥ g_prime a x₀) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_2_1_part_2_2_l1067_106704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sqrt_10x_l1067_106710

theorem probability_sqrt_10x : 
  ∃ (P : Set ℝ → ℝ), 
    (∀ x : ℝ, 0 ≤ x ∧ x < 1000 ∧ ⌊Real.sqrt x⌋ = 14 →
      P {y | ⌊Real.sqrt (10 * y)⌋ = 44} = 13/58) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sqrt_10x_l1067_106710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_overtime_multiplier_is_24_l1067_106774

/-- Represents the hourly rate structure for Harry and James -/
structure PayStructure where
  baseRate : ℝ  -- Base hourly rate (x dollars)
  harryBaseHours : ℕ  -- Number of hours at base rate for Harry
  jamesBaseHours : ℕ  -- Number of hours at base rate for James
  jamesOvertimeMultiplier : ℝ  -- Overtime multiplier for James

/-- Calculates James's total pay for the week -/
def jamesPay (ps : PayStructure) (totalHours : ℕ) : ℝ :=
  ps.baseRate * (ps.jamesBaseHours : ℝ) + 
  ps.jamesOvertimeMultiplier * ps.baseRate * ((totalHours : ℝ) - (ps.jamesBaseHours : ℝ))

/-- Calculates Harry's total pay for the week -/
def harryPay (ps : PayStructure) (overtimeMultiplier : ℝ) (totalHours : ℕ) : ℝ :=
  ps.baseRate * (ps.harryBaseHours : ℝ) + 
  overtimeMultiplier * ps.baseRate * ((totalHours : ℝ) - (ps.harryBaseHours : ℝ))

/-- Theorem stating that Harry's overtime multiplier is 24 -/
theorem harry_overtime_multiplier_is_24 (ps : PayStructure) (harryHours : ℕ) :
  ps.harryBaseHours = 18 →
  ps.jamesBaseHours = 40 →
  ps.jamesOvertimeMultiplier = 2 →
  jamesPay ps 41 = harryPay ps 24 harryHours →
  harryHours > ps.harryBaseHours →
  24 = (jamesPay ps 41 - ps.baseRate * (ps.harryBaseHours : ℝ)) / (ps.baseRate * ((harryHours : ℝ) - (ps.harryBaseHours : ℝ))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_overtime_multiplier_is_24_l1067_106774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_symbol_is_d_l1067_106753

def flowchart_decision_symbol : String := "D"

theorem decision_symbol_is_d : flowchart_decision_symbol = "D" := by
  rfl

#check decision_symbol_is_d

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_symbol_is_d_l1067_106753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l1067_106715

/-- The distance at which two trains meet, given their departure times and speeds -/
noncomputable def meeting_distance (departure_time_1 departure_time_2 : ℕ) (speed_1 speed_2 : ℝ) : ℝ :=
  let time_difference := (departure_time_2 - departure_time_1 : ℝ)
  let distance_covered_by_train_1 := speed_1 * time_difference
  let relative_speed := speed_2 - speed_1
  let catch_up_time := distance_covered_by_train_1 / relative_speed
  speed_2 * catch_up_time

/-- Theorem stating that the meeting distance for the given problem is 600 km -/
theorem train_meeting_distance :
  meeting_distance 9 14 30 40 = 600 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l1067_106715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_basis_is_basis_l1067_106764

def e₁ : ℝ × ℝ × ℝ := (1, 0, 0)
def e₂ : ℝ × ℝ × ℝ := (0, 1, 0)
def e₃ : ℝ × ℝ × ℝ := (0, 0, 1)

theorem standard_basis_is_basis :
  LinearIndependent ℝ (λ i : Fin 3 => if i = 0 then e₁ else if i = 1 then e₂ else e₃) ∧
  Submodule.span ℝ (Set.range (λ i : Fin 3 => if i = 0 then e₁ else if i = 1 then e₂ else e₃)) = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_basis_is_basis_l1067_106764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1067_106732

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The condition that f must satisfy for all positive real x and y -/
def SatisfiesCondition (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val x / y^2 - f.val y / x^2 ≤ (1/x - 1/y)^2

/-- The theorem stating that any function satisfying the condition must be of the form C/x^2 -/
theorem function_characterization (f : PositiveRealFunction) 
  (h : SatisfiesCondition f) : 
  ∃ C > 0, ∀ x > 0, f.val x = C / x^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1067_106732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1067_106779

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem min_value_of_f :
  ∀ x : ℝ, x > 2 → f x ≥ 4 ∧ (f x = 4 ↔ x = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1067_106779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l1067_106756

/-- Calculate the price after applying a discount percentage to an initial price -/
noncomputable def apply_discount (initial_price : ℝ) (discount_percentage : ℝ) : ℝ :=
  initial_price * (1 - discount_percentage / 100)

/-- Calculate the final price after applying two successive discounts -/
noncomputable def final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  apply_discount (apply_discount list_price discount1) discount2

theorem article_price_after_discounts :
  let list_price : ℝ := 68
  let discount1 : ℝ := 10
  let discount2 : ℝ := 8.235294117647069
  abs (final_price list_price discount1 discount2 - 56.16) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l1067_106756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_problem_l1067_106767

/-- The length of a circular track given two runners' speeds and meeting frequency -/
noncomputable def track_length (speed_a speed_b : ℝ) (time meeting_count : ℕ) : ℝ :=
  (speed_a + speed_b) * (time : ℝ) / (meeting_count : ℝ)

/-- Theorem stating the track length for the given problem -/
theorem track_length_problem : 
  track_length 180 240 30 24 = 525 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_problem_l1067_106767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1067_106759

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- State the theorem
theorem f_strictly_increasing :
  ∀ x y, x ∈ Set.Icc (-Real.pi/6) 0 → y ∈ Set.Icc (-Real.pi/6) 0 →
    x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1067_106759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1067_106794

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2 + Real.pi / 6) + 3

theorem function_properties :
  let period : ℝ := 4 * Real.pi
  let amplitude : ℝ := 3
  let initial_phase : ℝ := Real.pi / 6
  let max_value : ℝ := 3 * (Real.sqrt 2 + Real.sqrt 6) / 4 + 3
  let min_value : ℝ := 3 * Real.sin (Real.pi / 6) + 3
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    (∃ k : ℤ, f (x + period) = f x) ∧
    (abs (f x - 3) ≤ amplitude) ∧
    (f (0 - initial_phase) = 3) ∧
    (f x ≤ max_value) ∧
    (f x ≥ min_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1067_106794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_l1067_106778

/-- Given a rectangle with dimensions 2a by b, where b ≤ 2a, 
    the total area wasted after cutting out the largest possible circular piece 
    and then cutting out the largest possible square piece from that circle 
    is equal to 2ab - b²/2. -/
theorem metal_waste (a b : ℝ) (h : b ≤ 2*a) : 
  2*a*b - (b/Real.sqrt 2)^2 = 2*a*b - b^2/2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_l1067_106778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_proof_l1067_106742

def ball_distribution_count : Nat → Nat → Nat
| 5, 4 => 240
| _, _ => 0  -- Default case for all other inputs

theorem ball_distribution_proof :
  ∀ (n m : Nat), n = 5 ∧ m = 4 →
  ball_distribution_count n m = 240 := by
  intros n m h
  cases h with
  | intro hn hm =>
    rw [hn, hm]
    rfl

#eval ball_distribution_count 5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_proof_l1067_106742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_angles_pentagon_l1067_106782

/-- The number of sides in a regular pentagon -/
def n : ℕ := 5

/-- The angle of rotation in degrees -/
def θ : ℝ := 40

/-- The measure of an exterior angle of a regular pentagon in degrees -/
noncomputable def exterior_angle : ℝ := 360 / n

/-- The measure of each newly formed adjacent angle in degrees -/
noncomputable def adjacent_angle : ℝ := 180 - (exterior_angle + θ)

/-- The theorem stating the sum of newly formed adjacent angles in a pentagon -/
theorem sum_of_adjacent_angles_pentagon :
  (n : ℝ) * adjacent_angle = 340 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_angles_pentagon_l1067_106782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_increase_l1067_106731

theorem investment_increase (original_investment increase_percentage : ℝ) 
  (h1 : original_investment = 12500)
  (h2 : increase_percentage = 215) : 
  original_investment * (1 + increase_percentage / 100) = 39375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_increase_l1067_106731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_l1067_106728

/-- The intersection point of two lines lies on a fixed circle -/
theorem intersection_point_on_circle (m : ℝ) :
  ∃ P : ℝ × ℝ,
    (m * P.1 - P.2 = 0) ∧
    (P.1 + m * P.2 - m - 2 = 0) ∧
    ((P.1 - 1)^2 + (P.2 - 1/2)^2 = 5/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_l1067_106728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l1067_106773

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (x + 4) / (x + 3)

-- State the theorem
theorem h_properties :
  -- h is strictly decreasing on (-∞, -3) and (-3, +∞)
  (∀ x y : ℝ, x < y ∧ x < -3 → h x > h y) ∧
  (∀ x y : ℝ, -3 < x ∧ x < y → h x > h y) ∧
  -- (-3, 1) is the center of symmetry
  (∀ x : ℝ, h ((-3) - x) + 1 = h ((-3) + x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l1067_106773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_of_A_l1067_106776

noncomputable def cost_price_A : ℝ := 154
noncomputable def selling_price_C : ℝ := 231
noncomputable def profit_percentage_B : ℝ := 25

noncomputable def selling_price_B : ℝ := selling_price_C / (1 + profit_percentage_B / 100)
noncomputable def selling_price_A : ℝ := selling_price_B
noncomputable def profit_A : ℝ := selling_price_A - cost_price_A
noncomputable def profit_percentage_A : ℝ := (profit_A / cost_price_A) * 100

theorem profit_percentage_of_A :
  profit_percentage_A = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_of_A_l1067_106776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_intersection_frequency_l1067_106752

/-- A sinusoidal function with amplitude A, angular frequency ω, and phase φ. -/
noncomputable def sinusoidal (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- Theorem: If a sinusoidal function with positive amplitude and angular frequency
intersects a horizontal line at three consecutive x-coordinates π/6, π/3, and 2π/3,
then the angular frequency ω equals 4. -/
theorem sinusoidal_intersection_frequency
  (A ω φ m : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_intersect₁ : sinusoidal A ω φ (π / 6) = m)
  (h_intersect₂ : sinusoidal A ω φ (π / 3) = m)
  (h_intersect₃ : sinusoidal A ω φ ((2 * π) / 3) = m) :
  ω = 4 := by
  sorry

#check sinusoidal_intersection_frequency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_intersection_frequency_l1067_106752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1067_106755

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) - 1/2

theorem f_properties :
  ∀ α k : ℝ,
  0 < α → α < π/2 →
  Real.sin α = Real.sqrt 2 / 2 →
  f α = 1/2 ∧
  (∀ x : ℝ, -3*π/8 + k*π ≤ x → x ≤ π/8 + k*π → 
    ∀ y : ℝ, -3*π/8 + k*π ≤ y → y ≤ x → f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1067_106755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1067_106775

theorem calculation_proofs :
  (∀ x y z : ℝ, x = Real.sqrt 6 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 3 → 
    x * (y + z) - Real.sqrt 27 = 3 * y - z) ∧
  (∀ a b c : ℝ, a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 27 → 
    (a - b)^2 + (2 * c - Real.sqrt 48) / a = 7 - 2 * Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1067_106775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_focus_l1067_106760

/-- Parabola defined by parametric equations x = 4t² and y = 4t -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P on the parabola -/
def P (m : ℝ) : ℝ × ℝ := (3, m)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_distance_to_focus (m : ℝ) :
  ∃ (para : Parabola), P m = (para.x, para.y) → distance (P m) focus = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_focus_l1067_106760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l1067_106766

/-- The function c(x) with parameter m -/
noncomputable def c (m : ℝ) (x : ℝ) : ℝ := (3 * m * x^2 - 4 * x + 1) / (7 * x^2 - 6 * x + m)

/-- The domain of c(x) is all real numbers if and only if m > 9/7 -/
theorem domain_c_all_reals (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, c m x = y) ↔ m > 9/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l1067_106766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1067_106749

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define M as the midpoint of AB
noncomputable def M (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define P and Q on AB
variable (P Q : ℝ × ℝ)

-- Define D and E on BC
variable (D E : ℝ × ℝ)

-- Define N as the midpoint of MB
noncomputable def N (A B : ℝ × ℝ) : ℝ × ℝ := 
  let M := M A B
  ((M.1 + B.1) / 2, (M.2 + B.2) / 2)

-- Axioms
axiom A_P_Q_M_order (A B P Q : ℝ × ℝ) : A.1 < P.1 ∧ P.1 < Q.1 ∧ Q.1 < (M A B).1

axiom MD_parallel_PC (A B C P D : ℝ × ℝ) : 
  (D.2 - (M A B).2) * (C.1 - P.1) = (C.2 - P.2) * (D.1 - (M A B).1)

axiom NE_parallel_QC (A B C Q E : ℝ × ℝ) : 
  (E.2 - (N A B).2) * (C.1 - Q.1) = (C.2 - Q.2) * (E.1 - (N A B).1)

-- Define area function
noncomputable def area (X Y Z : ℝ × ℝ) : ℝ := 
  abs ((Y.1 - X.1) * (Z.2 - X.2) - (Z.1 - X.1) * (Y.2 - X.2)) / 2

-- Theorem to prove
theorem area_ratio (A B C : ℝ × ℝ) : 
  area B (N A B) E / area A B C = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1067_106749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powerTower_3_mod_60_l1067_106795

def powerTower (a : ℕ) : ℕ → ℕ
  | 0 => 1
  | 1 => a
  | n + 1 => a ^ (powerTower a n)

theorem powerTower_3_mod_60 :
  powerTower 3 (powerTower 3 (powerTower 3 3)) % 60 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powerTower_3_mod_60_l1067_106795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1067_106747

/-- Calculates the length of a bridge given train parameters and crossing time. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Theorem stating that a bridge's length is 255 meters given specific train parameters. -/
theorem bridge_length_calculation :
  bridge_length 120 45 30 = 255 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1067_106747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1067_106741

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))

-- Theorem statement
theorem ellipse_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hM : (1, 3/2) ∈ Ellipse a b)
  (hF : ∃ (F₁ F₂ : ℝ × ℝ), F₁ ∈ Ellipse a b ∧ F₂ ∈ Ellipse a b ∧
    distance (1, 3/2) F₁ + distance (1, 3/2) F₂ = 4) :
  (∃ (P Q : ℝ × ℝ), P ∈ Ellipse 2 (Real.sqrt 3) ∧ Q ∈ Ellipse 2 (Real.sqrt 3) ∧ P ≠ Q ∧
    (0, 0) = ((1, 3/2) + P + Q) / 3 ∧
    triangleArea (1, 3/2) P Q = 9/2) ∧
  Ellipse 2 (Real.sqrt 3) = Ellipse a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1067_106741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l1067_106745

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 2 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 8}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := {x | 2 < x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l1067_106745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_necessary_not_sufficient_l1067_106726

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the condition of constant sum of distances
def constantSumCondition (p : Point) (f1 f2 : Point) (c : ℝ) : Prop :=
  distance p f1 + distance p f2 = c

-- Define what it means for a point to be on an ellipse
def isOnEllipse (p : Point) (f1 f2 : Point) (a : ℝ) : Prop :=
  distance p f1 + distance p f2 = 2 * a

-- Theorem statement
theorem constant_sum_necessary_not_sufficient :
  (∀ p f1 f2 : Point, ∀ a : ℝ, isOnEllipse p f1 f2 a → ∃ c, constantSumCondition p f1 f2 c) ∧
  ¬(∀ p f1 f2 : Point, ∀ c : ℝ, constantSumCondition p f1 f2 c → ∃ a, isOnEllipse p f1 f2 a) := by
  sorry

#check constant_sum_necessary_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_necessary_not_sufficient_l1067_106726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1067_106780

/-- A hyperbola is a type of conic section -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a² - y²/b² = 1) -/
  equation : (ℝ → ℝ → Prop)

/-- Defines the concept of a focus for a hyperbola -/
def has_focus (h : Hyperbola) (x y : ℝ) : Prop :=
  ∃ (c : ℝ), x^2 + y^2 = c^2

/-- Defines the concept of asymptotes for a hyperbola -/
def has_asymptotes (h : Hyperbola) (m : ℝ) : Prop :=
  ∃ (a b : ℝ), h.equation = (λ x y ↦ x^2/a^2 - y^2/b^2 = 1) ∧ m = b/a

/-- The main theorem: given a hyperbola with one focus at (5,0) and asymptotes y = ± 4/3x, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation 
  (h : Hyperbola) 
  (focus : has_focus h 5 0) 
  (asymp : has_asymptotes h (4/3)) : 
  h.equation = (λ x y ↦ x^2/9 - y^2/16 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1067_106780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_theorem_l1067_106705

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1/2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_and_range_theorem 
  (a : ℝ) 
  (h_symmetry : ∀ x, f (a + x) = f (a - x)) :
  (g (2 * a) = 1/2) ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
    1/2 ≤ f x + g x ∧ f x + g x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_theorem_l1067_106705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_planes_intersection_lines_l1067_106771

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Represents the intersection of two planes -/
def intersection (p1 p2 : Plane) : Option Line := sorry

/-- Counts the number of distinct lines of intersection among the given planes -/
def countIntersectionLines (planes : List Plane) : Nat := sorry

/-- Counts the number of parts that the given planes divide the space into -/
def countSpaceParts (planes : List Plane) : Nat := sorry

/-- Theorem: Three non-overlapping planes that divide space into six parts have exactly three lines of intersection -/
theorem three_planes_intersection_lines (p1 p2 p3 : Plane) :
  let planes := [p1, p2, p3]
  (countSpaceParts planes = 6) → (countIntersectionLines planes = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_planes_intersection_lines_l1067_106771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_sum_l1067_106750

theorem max_pairs_sum (n : ℕ) (h : n = 4019) :
  let S := Finset.range n
  ∃ (k : ℕ) (f : Fin k → S × S),
    (∀ i : Fin k, (f i).1 < (f i).2) ∧
    (∀ i j : Fin k, i ≠ j → (f i).1 ≠ (f j).1 ∧ (f i).1 ≠ (f j).2 ∧ (f i).2 ≠ (f j).1 ∧ (f i).2 ≠ (f j).2) ∧
    (∀ i j : Fin k, i ≠ j → (f i).1.val + (f i).2.val ≠ (f j).1.val + (f j).2.val) ∧
    (∀ i : Fin k, (f i).1.val + (f i).2.val ≤ n) ∧
    k = 1607 ∧
    (∀ m : ℕ, m > k →
      ¬∃ (g : Fin m → S × S),
        (∀ i : Fin m, (g i).1 < (g i).2) ∧
        (∀ i j : Fin m, i ≠ j → (g i).1 ≠ (g j).1 ∧ (g i).1 ≠ (g j).2 ∧ (g i).2 ≠ (g j).1 ∧ (g i).2 ≠ (g j).2) ∧
        (∀ i j : Fin m, i ≠ j → (g i).1.val + (g i).2.val ≠ (g j).1.val + (g j).2.val) ∧
        (∀ i : Fin m, (g i).1.val + (g i).2.val ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_sum_l1067_106750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_translates_existence_l1067_106707

theorem disjoint_translates_existence 
  (n k m : ℕ) 
  (A : Finset ℕ) 
  (h1 : A.card = k)
  (h2 : ∀ x ∈ A, x ≤ n)
  (h3 : m > 0)
  (h4 : n > (m - 1) * (Nat.choose k 2 + 1)) :
  ∃ t : Fin m → ℕ, 
    ∀ i j : Fin m, i ≠ j → 
      (A.image (λ x ↦ (x + t i) % (n + 1))) ∩ 
      (A.image (λ x ↦ (x + t j) % (n + 1))) = ∅ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_translates_existence_l1067_106707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1067_106720

theorem triangle_tangent (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (condition : a^2 + b^2 - c^2 = -2/3 * a * b) : 
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1067_106720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_of_digits_l1067_106751

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a = 5 ∨ a = 6 ∨ a = 7 ∨ a = 8) ∧
  (b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) ∧
  (c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8) ∧
  (d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8)

def product_of_arrangement (a b c d : ℕ) : ℕ :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product_of_digits :
  ∀ a b c d : ℕ, is_valid_arrangement a b c d →
  product_of_arrangement a b c d ≥ 3876 := by
  sorry

#check smallest_product_of_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_of_digits_l1067_106751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_original_price_l1067_106761

/-- Given a sale price and a percent decrease, calculates the original price. -/
noncomputable def original_price (sale_price : ℝ) (percent_decrease : ℝ) : ℝ :=
  sale_price / (1 - percent_decrease)

/-- Theorem: Given a sale price of $40 and a percent decrease of 60%, 
    the original price of the trouser is $100. -/
theorem trouser_original_price :
  original_price 40 0.60 = 100 := by
  -- Unfold the definition of original_price
  unfold original_price
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_original_price_l1067_106761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1067_106706

/-- The standard equation of a hyperbola with the same asymptotes as x^2 - y^2/4 = 1
    and passing through the point (2, 2) is x^2/3 - y^2/12 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ k : ℝ, x^2 - y^2/4 = k ∧ 2^2 - 2^2/4 = k) →
  (x^2/3 - y^2/12 = 1) ↔ (x^2 - y^2/4 = 1 ∨ (x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1067_106706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_at_least_4_pow_m_l1067_106757

/-- Represents the state of the sheets of paper --/
structure SheetState (m : ℕ) where
  numbers : Fin (2^m) → ℕ

/-- The operation performed on two sheets --/
def update_sheets (state : SheetState m) (i j : Fin (2^m)) : SheetState m :=
  { numbers := λ k =>
      if k = i ∨ k = j then
        state.numbers i + state.numbers j
      else
        state.numbers k }

/-- The initial state with all sheets containing 1 --/
def initial_state (m : ℕ) : SheetState m :=
  { numbers := λ _ => 1 }

/-- The final state after performing m * 2^(m-1) operations --/
noncomputable def final_state (m : ℕ) : SheetState m :=
  sorry  -- The implementation of this function is not required for the theorem statement

/-- The sum of all numbers in a state --/
def sum_of_numbers (state : SheetState m) : ℕ :=
  (Finset.univ.sum state.numbers)

/-- The main theorem --/
theorem sum_at_least_4_pow_m (m : ℕ) :
  sum_of_numbers (final_state m) ≥ 4^m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_at_least_4_pow_m_l1067_106757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l1067_106798

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x)^2 - 1/2

noncomputable def g (x : ℝ) : ℝ := f (x + 5*Real.pi/12)

theorem range_of_g_on_interval :
  Set.range (fun x ↦ g x) ∩ Set.Icc (-Real.pi/12) (Real.pi/3) = Set.Icc (-1) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l1067_106798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_equation_l1067_106700

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the curve C
def C : Set (ℝ × ℝ) := {P | ‖P - A‖ = Real.sqrt 2 * ‖P - B‖}

-- Define the line l
def l : Set (ℝ × ℝ) := {Q | Q.1 + Q.2 - 4 = 0}

-- Define a placeholder for the area calculation
noncomputable def area_OEQF (O E Q F : ℝ × ℝ) : ℝ := sorry

-- Define the minimization condition
def is_min_area (E F Q : ℝ × ℝ) : Prop :=
  E ∈ C ∧ F ∈ C ∧ Q ∈ l ∧
  (∀ E' F' Q', E' ∈ C → F' ∈ C → Q' ∈ l →
    area_OEQF (0, 0) E Q F ≤ area_OEQF (0, 0) E' Q' F')

-- State the theorem
theorem min_area_line_equation :
  ∀ E F Q : ℝ × ℝ,
  is_min_area E F Q →
  ∃ k : ℝ, E.1 + E.2 - 1 = 0 ∧ F.1 + F.2 - 1 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_equation_l1067_106700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_equal_rounding_l1067_106724

noncomputable def K₁ : ℝ := 1842 * Real.sqrt 2 + 863 * Real.sqrt 7
noncomputable def K₂ : ℝ := 3519 + 559 * Real.sqrt 6

noncomputable def round_to_n_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋ : ℝ) / 10^n

theorem largest_equal_rounding :
  (∀ n : ℕ, n ≤ 4 → round_to_n_places K₁ n = round_to_n_places K₂ n) ∧
  (∀ n : ℕ, n > 4 → round_to_n_places K₁ n ≠ round_to_n_places K₂ n) := by
  sorry

#check largest_equal_rounding

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_equal_rounding_l1067_106724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1067_106777

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the condition for the ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∀ M : ℝ × ℝ, distance M F₁ + distance M F₂ = 2*m + 1

-- Define the condition for the hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∀ N : ℝ × ℝ, |distance N F₁ - distance N F₂| = 2*m - 1

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (is_ellipse m ∧ is_hyperbola m) → (5/2 < m ∧ m < 7/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1067_106777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_dividing_line_l1067_106785

/-- A polar curve representing a circle -/
noncomputable def circle_curve (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- A line in polar coordinates -/
noncomputable def dividing_line (θ : ℝ) (a : ℝ) : ℝ := a / (Real.sin θ - Real.cos θ)

/-- Predicate stating that a line divides a region into equal areas -/
def divides_equally (line : ℝ → ℝ) (curve : ℝ → ℝ) : Prop := sorry

theorem equal_area_dividing_line :
  ∃ a : ℝ, divides_equally (dividing_line · a) circle_curve ∧ a = -1 := by
  sorry

#check equal_area_dividing_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_dividing_line_l1067_106785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_at_check_is_100_l1067_106791

/-- Represents the road construction project parameters and progress --/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  initialWorkers : ℚ
  completedLength : ℚ
  additionalWorkers : ℚ

/-- Calculates the number of days passed when progress was checked --/
def daysPassedAtCheck (project : RoadProject) : ℚ :=
  (project.completedLength * project.totalDays) / project.totalLength

/-- Theorem stating that for the given project parameters, 100 days had passed when progress was checked --/
theorem days_passed_at_check_is_100 (project : RoadProject) 
  (h1 : project.totalLength = 10)
  (h2 : project.totalDays = 300)
  (h3 : project.initialWorkers = 30)
  (h4 : project.completedLength = 2)
  (h5 : project.additionalWorkers = 30) :
  daysPassedAtCheck project = 100 := by
  sorry

def main : IO Unit := do
  let project : RoadProject := {
    totalLength := 10,
    totalDays := 300,
    initialWorkers := 30,
    completedLength := 2,
    additionalWorkers := 30
  }
  IO.println s!"Days passed at check: {daysPassedAtCheck project}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_at_check_is_100_l1067_106791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_l1067_106727

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- C is at right angle (90°)
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 5
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 25 ∧
  -- AC = 12
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 144

-- Define sin B
noncomputable def sin_B (A B C : ℝ × ℝ) : ℝ :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB / BC

-- Theorem statement
theorem sin_B_value (A B C : ℝ × ℝ) :
  triangle_ABC A B C → sin_B A B C = 5/13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_l1067_106727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_properties_l1067_106718

def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}

theorem A_properties (a b : ℝ) :
  (b = 2 → (∀ x y, x ∈ A a b → y ∈ A a b → x = y) ↔ (a = 0 ∨ a ≥ 1)) ∧
  (A a b = ∅ ↔ (a = 0 ∧ b = 0) ∨ (b^2 - 4*a < 0 ∧ a ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_properties_l1067_106718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_car_average_speed_example_l1067_106799

/-- Calculates the average speed of a car given its uphill and downhill speeds and distances --/
theorem car_average_speed 
  (uphill_speed uphill_distance downhill_speed downhill_distance : ℝ) 
  (h1 : uphill_speed > 0)
  (h2 : downhill_speed > 0)
  (h3 : uphill_distance > 0)
  (h4 : downhill_distance > 0) :
  let total_distance := uphill_distance + downhill_distance
  let uphill_time := uphill_distance / uphill_speed
  let downhill_time := downhill_distance / downhill_speed
  let total_time := uphill_time + downhill_time
  total_distance / total_time = (uphill_distance + downhill_distance) / (uphill_distance / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

/-- Specific instance of the car_average_speed theorem --/
theorem car_average_speed_example :
  let uphill_speed : ℝ := 30
  let uphill_distance : ℝ := 100
  let downhill_speed : ℝ := 80
  let downhill_distance : ℝ := 50
  let average_speed := (uphill_distance + downhill_distance) / (uphill_distance / uphill_speed + downhill_distance / downhill_speed)
  ∃ ε > 0, |average_speed - 37.92| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_car_average_speed_example_l1067_106799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_20_factorial_l1067_106746

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def highest_power_dividing (base : ℕ) : ℕ → ℕ
| 0 => 0
| n + 1 => if (n + 1) % base = 0 
           then 1 + highest_power_dividing base n
           else highest_power_dividing base n

def highest_power_of_10 (n : ℕ) : ℕ :=
  min (highest_power_dividing 2 n) (highest_power_dividing 5 n)

def highest_power_of_6 (n : ℕ) : ℕ :=
  min (highest_power_dividing 2 n) (highest_power_dividing 3 n)

theorem sum_of_powers_20_factorial :
  highest_power_of_10 (factorial 20) + highest_power_of_6 (factorial 20) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_20_factorial_l1067_106746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polyhedron_space_diagonals_l1067_106702

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - (Q.pentagonal_faces * 5)

/-- Theorem stating that a convex polyhedron with specific properties has 303 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 42,
    triangular_faces := 30,
    pentagonal_faces := 12
  }
  space_diagonals Q = 303 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polyhedron_space_diagonals_l1067_106702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_allowance_l1067_106713

/-- The student's weekly allowance in dollars -/
noncomputable def weekly_allowance : ℝ := 4.50

/-- The amount spent at the arcade -/
noncomputable def arcade_spent : ℝ := (3/5) * weekly_allowance

/-- The amount remaining after spending at the arcade -/
noncomputable def remaining_after_arcade : ℝ := weekly_allowance - arcade_spent

/-- The amount spent at the toy store -/
noncomputable def toy_store_spent : ℝ := (1/3) * remaining_after_arcade

/-- The amount remaining after spending at the toy store -/
noncomputable def remaining_after_toy_store : ℝ := remaining_after_arcade - toy_store_spent

/-- The amount spent at the candy store -/
def candy_store_spent : ℝ := 1.20

theorem student_allowance : 
  weekly_allowance = 4.50 ∧ 
  remaining_after_toy_store = candy_store_spent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_allowance_l1067_106713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_symmetry_axis_l1067_106744

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)
noncomputable def g (x : ℝ) := 2 * (Real.cos (x - Real.pi / 6))^2 + 1

theorem same_symmetry_axis (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ) 
  (h3 : φ < Real.pi) 
  (h4 : ∀ x, ∃ k, f ω φ x = f ω φ (k * Real.pi / ω + (Real.pi - 2 * φ) / (2 * ω) - x)) 
  (h5 : ∀ x, ∃ k, g x = g (k * Real.pi / 2 + Real.pi / 6 - x)) :
  ω = 2 ∧ φ = Real.pi / 6 ∧ 
  StrictMonoOn (f ω φ) (Set.Icc (-Real.pi / 3) (Real.pi / 6)) :=
sorry

#check same_symmetry_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_symmetry_axis_l1067_106744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_girls_at_event_l1067_106716

/-- Represents a school with a given number of students and ratio of boys to girls -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def num_girls (s : School) : ℕ :=
  s.total_students * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- The Maple Grove Middle School -/
def maple_grove : School :=
  { total_students := 300, boys_ratio := 3, girls_ratio := 2 }

/-- The Pine Ridge Middle School -/
def pine_ridge : School :=
  { total_students := 240, boys_ratio := 1, girls_ratio := 3 }

/-- The fraction of girls attending the social event -/
theorem fraction_of_girls_at_event :
  (num_girls maple_grove + num_girls pine_ridge) * 9 =
  (maple_grove.total_students + pine_ridge.total_students) * 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_girls_at_event_l1067_106716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_zero_l1067_106781

theorem logarithm_expression_equals_zero :
  1 + Real.log 2 / Real.log 10 * Real.log 5 / Real.log 10 
  - Real.log 2 / Real.log 10 * Real.log 50 / Real.log 10 
  - Real.log 5 / Real.log 3 * Real.log 9 / Real.log 25 * Real.log 5 / Real.log 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_zero_l1067_106781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_B_highest_no_car_percent_l1067_106748

structure Ship where
  name : String
  round_trip_percent : ℚ
  car_percent : ℚ

def no_car_percent (s : Ship) : ℚ :=
  s.round_trip_percent * (1 - s.car_percent / 100)

def ships : List Ship := [
  { name := "A", round_trip_percent := 30, car_percent := 25 },
  { name := "B", round_trip_percent := 50, car_percent := 15 },
  { name := "C", round_trip_percent := 20, car_percent := 35 }
]

theorem ship_B_highest_no_car_percent :
  ∀ s ∈ ships, s.name ≠ "B" → no_car_percent s ≤ no_car_percent (ships[1]) :=
by
  intro s s_in_ships s_not_B
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_B_highest_no_car_percent_l1067_106748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_equals_thirteen_over_two_l1067_106735

-- Define the velocity function
def v (t : ℝ) : ℝ := 3 * t + 2

-- Theorem statement
theorem distance_covered_equals_thirteen_over_two :
  ∫ t in (1 : ℝ)..(2 : ℝ), v t = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_equals_thirteen_over_two_l1067_106735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_reciprocal_l1067_106738

theorem exponent_sum_reciprocal (a b c : ℝ) (h1 : (15 : ℝ)^a = 25) (h2 : (5 : ℝ)^b = 25) (h3 : (3 : ℝ)^c = 25) :
  1/a + 1/b - 1/c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_reciprocal_l1067_106738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l1067_106739

-- Define the hyperbola C
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line l
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the circle centered at A(0, -1)
def circle_at_A (x y r : ℝ) : Prop := x^2 + (y + 1)^2 = r^2

-- Main theorem
theorem hyperbola_intersection_theorem 
  (a b k m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hk : k ≠ 0) 
  (hm : m ≠ 0) 
  (h_focal : 4 = 2 * Real.sqrt (a^2 + b^2))
  (h_eccentricity : Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 3 / 3)
  (h_intersection : ∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    hyperbola x1 y1 a b ∧ 
    hyperbola x2 y2 a b ∧ 
    line x1 y1 k m ∧ 
    line x2 y2 k m)
  (h_circle : ∃ (r : ℝ), 
    circle_at_A x1 y1 r ∧ 
    circle_at_A x2 y2 r) :
  (a^2 = 3 ∧ b^2 = 1) ∧ 
  (m > -1/4 ∧ m < 0 ∨ m > 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l1067_106739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l1067_106711

noncomputable section

-- Define the given points and lines
def A : ℝ × ℝ := (7, 1)
def B : ℝ × ℝ := (4, -2)
def N : ℝ × ℝ := (4, 2)

def k : ℝ := 1/2

def l1 (x y : ℝ) : Prop := y - A.2 = k * (x - A.1)
def l2 (x y : ℝ) : Prop := x + 2*y + 3 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - M.1)^2 + (y - M.2)^2 = 9

-- Define the tangent lines
def tangent1 (x : ℝ) : Prop := x = 4
def tangent2 (x y : ℝ) : Prop := 7*x - 24*y + 20 = 0

theorem circle_and_tangents :
  (∀ x y, l1 x y ∧ l2 x y ↔ (x, y) = M) ∧
  C B.1 B.2 ∧
  (∀ x y, (tangent1 x ∨ tangent2 x y) →
    (x, y) = N ∨ (C x y ∧ ∃ ε > 0, ∀ δ x' y', 0 < δ ∧ δ < ε →
      ((x' - x)^2 + (y' - y)^2 = δ^2) → ¬C x' y')) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l1067_106711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_different_digits_sum_32_l1067_106709

-- Define a function to check if all digits in a number are different
def allDigitsDifferent (n : Nat) : Bool :=
  let digits := Nat.digits 10 n
  digits.length = digits.toFinset.card

-- Define a function to calculate the sum of digits
def sumOfDigits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

-- State the theorem
theorem smallest_number_with_different_digits_sum_32 :
  (∀ m : Nat, m < 26789 → ¬(allDigitsDifferent m ∧ sumOfDigits m = 32)) ∧
  allDigitsDifferent 26789 ∧
  sumOfDigits 26789 = 32 := by
  sorry

#eval allDigitsDifferent 26789
#eval sumOfDigits 26789

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_different_digits_sum_32_l1067_106709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l1067_106786

theorem sqrt_difference_comparison : Real.sqrt 7 - Real.sqrt 6 < Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l1067_106786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1067_106740

/-- The area of a circular sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (angle : ℝ) : ℝ :=
  (angle / 360) * Real.pi * radius^2

theorem sector_area_calculation (radius angle : ℝ) 
    (h_radius : radius = 12)
    (h_angle : angle = 42) :
  sectorArea radius angle = (7/60) * Real.pi * 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1067_106740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_bases_formula_l1067_106787

/-- An isosceles trapezoid with given height and diagonals. -/
structure IsoscelesTrapezoid where
  /-- Height of the trapezoid -/
  h : ℝ
  /-- Length of one diagonal -/
  a : ℝ
  /-- Length of the other diagonal -/
  b : ℝ
  /-- Height is positive -/
  h_pos : 0 < h
  /-- Diagonals are longer than height -/
  a_gt_h : h < a
  b_gt_h : h < b

/-- The sum of the bases of an isosceles trapezoid. -/
noncomputable def sumOfBases (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt (t.a^2 - t.h^2) + Real.sqrt (t.b^2 - t.h^2)

/-- Theorem: The sum of the bases of an isosceles trapezoid is equal to
    the sum of the square roots of the differences between the squares of
    the diagonals and the square of the height. -/
theorem sum_of_bases_formula (t : IsoscelesTrapezoid) :
  ∃ (base1 base2 : ℝ), base1 + base2 = sumOfBases t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_bases_formula_l1067_106787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_sum_equals_negative_25_l1067_106788

-- Define the points A, B, and C as vectors in ℝ²
variable (A B C : ℝ × ℝ)

-- Define the vectors AB, BC, and CA
def AB (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def CA (C A : ℝ × ℝ) : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude (length) of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem dot_product_sum_equals_negative_25 
  (h1 : magnitude (AB A B) = 3)
  (h2 : magnitude (BC B C) = 4)
  (h3 : magnitude (CA C A) = 5) :
  dot_product (AB A B) (BC B C) + 
  dot_product (BC B C) (CA C A) + 
  dot_product (CA C A) (AB A B) = -25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_sum_equals_negative_25_l1067_106788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1067_106729

def M : Set ℝ := {x | -1 < x ∧ x < 1}

def N : Set ℝ := {x | ∃ n : ℤ, x = n ∧ -1 < n ∧ n < 2}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1067_106729
