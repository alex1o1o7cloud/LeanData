import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_session_on_thursday_l1228_122882

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents Victoria's gym schedule -/
def is_gym_day (d : Day) : Bool :=
  match d with
  | Day.Sunday => false
  | _ => true

/-- Number of gym days in a two-week cycle -/
def gym_days_per_cycle : Nat := 6

/-- Total number of gym sessions -/
def total_sessions : Nat := 30

/-- The day Victoria starts her gym sessions -/
def start_day : Day := Day.Monday

/-- Function to calculate the next gym day -/
def next_gym_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Wednesday
  | Day.Tuesday => Day.Thursday
  | Day.Wednesday => Day.Friday
  | Day.Thursday => Day.Saturday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Monday
  | Day.Sunday => Day.Monday

/-- Function to apply next_gym_day n times -/
def apply_next_gym_day (n : Nat) (d : Day) : Day :=
  match n with
  | 0 => d
  | n + 1 => next_gym_day (apply_next_gym_day n d)

/-- Theorem stating that Victoria's last gym session will be on Thursday -/
theorem last_session_on_thursday :
  ∃ n : Nat, (n * gym_days_per_cycle = total_sessions) ∧
  (apply_next_gym_day (n * gym_days_per_cycle - 1) start_day = Day.Thursday) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_session_on_thursday_l1228_122882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_in_range_l1228_122876

theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x < -1 → (m - m^2) * (4 : ℝ)^x + (2 : ℝ)^x + 1 > 0) ↔ -2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_in_range_l1228_122876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1228_122888

-- Define the function (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log (1/2)

-- Define the domain of the function
def domain : Set ℝ := {x | x < 1 ∨ x > 2}

-- Statement to prove
theorem monotonic_decreasing_interval :
  ∀ x y, x ∈ domain → y ∈ domain → x > y → x > 2 → y > 2 → f x < f y :=
by
  -- We use 'sorry' to skip the proof
  sorry

#check monotonic_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1228_122888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_cellos_l1228_122885

/-- The number of cellos in a music store, given certain conditions. -/
theorem number_of_cellos (violas : ℕ) (matching_pairs : ℕ) (probability : ℚ) : ℕ :=
  let cellos := 800
  by
    have h1 : violas = 600 := by sorry
    have h2 : matching_pairs = 100 := by sorry
    have h3 : probability = 1 / 4800 := by sorry
    have h4 : probability = matching_pairs / (cellos * violas) := by sorry
    -- The main proof goes here
    sorry

-- Remove the #eval statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_cellos_l1228_122885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_minus_four_equals_twentyfive_l1228_122824

theorem x_squared_minus_four_equals_twentyfive (x : ℝ) : 
  ((2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x = 256) → (x + 2) * (x - 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_minus_four_equals_twentyfive_l1228_122824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1228_122862

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Define distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define dot product
def dot_product (p q r : ℝ × ℝ) : ℝ :=
  (p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)

theorem ellipse_properties :
  ∀ (p : ℝ × ℝ), is_on_ellipse p.1 p.2 →
    (∀ (d : ℝ), d = distance p right_focus → 1 ≤ d ∧ d ≤ 3) ∧
    (2 ≤ dot_product p left_focus right_focus ∧ 
     dot_product p left_focus right_focus ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1228_122862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l1228_122878

theorem complex_number_in_fourth_quadrant (m : ℝ) (h : 2/3 < m ∧ m < 1) :
  let z : ℂ := m * (3 + Complex.I) - (2 + Complex.I)
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l1228_122878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_to_hundredth_l1228_122820

noncomputable def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem rounding_to_hundredth :
  round_to_nearest_hundredth 34.5539999 = 34.55 ∧
  round_to_nearest_hundredth 34.561 = 34.56 ∧
  round_to_nearest_hundredth 34.558 = 34.56 ∧
  round_to_nearest_hundredth 34.5601 = 34.56 ∧
  round_to_nearest_hundredth 34.56444 = 34.56 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_to_hundredth_l1228_122820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equality_l1228_122860

/-- Given three non-empty sets satisfying the specified condition, at least two of them are equal -/
theorem sets_equality (S₁ S₂ S₃ : Set ℝ) 
  (h_nonempty : Set.Nonempty S₁ ∧ Set.Nonempty S₂ ∧ Set.Nonempty S₃)
  (h_condition : ∀ (i j k : Fin 3) (x y : ℝ), 
    x ∈ (match i with | 0 => S₁ | 1 => S₂ | 2 => S₃ | _ => ∅) →
    y ∈ (match j with | 0 => S₁ | 1 => S₂ | 2 => S₃ | _ => ∅) →
    x - y ∈ (match k with | 0 => S₁ | 1 => S₂ | 2 => S₃ | _ => ∅)) :
  S₁ = S₂ ∨ S₂ = S₃ ∨ S₁ = S₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equality_l1228_122860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_graph_structure_five_is_smallest_special_graph_l1228_122803

/-- A graph with n points satisfying specific conditions --/
structure SpecialGraph (n : ℕ) where
  -- The vertex set
  V : Finset (Fin n)
  -- The edge relation
  E : Fin n → Fin n → Bool
  -- No point has edges to all other points
  no_full_degree : ∀ v : Fin n, ∃ w : Fin n, v ≠ w ∧ ¬E v w
  -- No triangles
  no_triangles : ∀ a b c : Fin n, E a b = true → E b c = true → E a c = true → a = b ∨ b = c ∨ a = c
  -- For any two non-adjacent points, there's exactly one point connected to both
  unique_connection : ∀ a b : Fin n, a ≠ b → E a b = false →
    ∃! c : Fin n, c ≠ a ∧ c ≠ b ∧ E a c = true ∧ E b c = true

/-- The degree of a vertex in the graph --/
def vertex_degree (G : SpecialGraph n) (v : Fin n) : ℕ :=
  (G.V.filter (fun w => G.E v w)).card

/-- The main theorem about SpecialGraph --/
theorem special_graph_structure (n : ℕ) (G : SpecialGraph n) :
  (∃ m : ℕ, n = m^2 + 1 ∧ ∀ v : Fin n, vertex_degree G v = m) ∧
  (∀ n' < n, ¬∃ G' : SpecialGraph n', True) :=
sorry

/-- The smallest possible n for a SpecialGraph --/
def smallest_special_graph : ℕ := 5

/-- Proof that 5 is the smallest possible n for a SpecialGraph --/
theorem five_is_smallest_special_graph :
  ∃ (G : SpecialGraph smallest_special_graph), True ∧
  ∀ n < smallest_special_graph, ¬∃ G' : SpecialGraph n, True :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_graph_structure_five_is_smallest_special_graph_l1228_122803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_compound_figure_l1228_122879

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- Theorem: Area of shaded region in compound figure -/
theorem shaded_area_in_compound_figure 
  (rect1 rect2 rect3 : Rectangle)
  (h1 : rect1.width = 4 ∧ rect1.height = 4)
  (h2 : rect2.width = 12 ∧ rect2.height = 12)
  (h3 : rect3.width = 16 ∧ rect3.height = 16)
  : triangleArea rect3.width (rect3.height * (rect1.width / (rect1.width + rect2.width + rect3.width))) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_compound_figure_l1228_122879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_of_Q_l1228_122817

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Defines the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Defines the right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse) : Point :=
  ⟨Real.sqrt (e.a^2 - e.b^2), 0⟩

/-- Theorem: The x-coordinate of Q lies in [-9/8, 9/8] -/
theorem x_coordinate_range_of_Q (e : Ellipse) (P : Point) :
  e.a = 4 ∧ e.b = 2 ∧
  eccentricity e = Real.sqrt 3 / 2 ∧
  P = ⟨0, 3/2⟩ ∧
  distance P (right_focus e) = Real.sqrt 57 / 2 →
  ∃ Q : Point, Q.y = 0 ∧ -9/8 ≤ Q.x ∧ Q.x ≤ 9/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_of_Q_l1228_122817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l1228_122855

/-- Definition of a right circular cone -/
structure RightCircularCone where
  base_radius : ℝ
  slant_height : ℝ
  axis_section_isosceles_right : slant_height = Real.sqrt 2 * base_radius

/-- The central angle of the unfolded side of a right circular cone -/
noncomputable def central_angle (cone : RightCircularCone) : ℝ :=
  2 * Real.pi * cone.base_radius / cone.slant_height

/-- Theorem: The central angle of the unfolded side of a right circular cone is √2π -/
theorem right_circular_cone_central_angle (cone : RightCircularCone) :
  central_angle cone = Real.sqrt 2 * Real.pi := by
  sorry

#check right_circular_cone_central_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l1228_122855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60deg_l1228_122872

/-- The area of a figure formed by rotating a semicircle about one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ := (2 * Real.pi * R^2 * α) / (2 * Real.pi)

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R 
    about one of its ends by an angle of 60° is equal to (2πR²)/3 -/
theorem rotated_semicircle_area_60deg (R : ℝ) (h : R > 0) : 
  rotated_semicircle_area R (Real.pi / 3) = (2 * Real.pi * R^2) / 3 := by
  sorry

#check rotated_semicircle_area_60deg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60deg_l1228_122872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l1228_122811

/-- Given a circle tangent to perpendicular axes with radius 10 and a tangent line AB,
    if the area of triangle OAB is 600 and OA > OB, then A, B, and P have specific coordinates -/
theorem circle_tangent_line (O A B P : ℝ × ℝ) : 
  let R : ℝ := 10
  let circleCenter : ℝ × ℝ := (R, R)
  -- Circle is tangent to axes
  (circleCenter.1 = R ∧ circleCenter.2 = R) →
  -- Line AB is tangent to circle at P
  (∃ (k : ℝ), (A.2 - B.2) = k * (A.1 - B.1) ∧ 
     (P.2 - circleCenter.2) = -1/k * (P.1 - circleCenter.1) ∧
     (P.1 - circleCenter.1)^2 + (P.2 - circleCenter.2)^2 = R^2) →
  -- Area of triangle OAB is 600
  (A.1 * B.2 / 2 = 600) →
  -- OA > OB
  (A.1 > B.2) →
  -- Coordinates of O
  (O = (0, 0)) →
  A = (40, 0) ∧ B = (0, 30) ∧ P = (16, 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l1228_122811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coronavirus_diameter_scientific_notation_l1228_122857

theorem coronavirus_diameter_scientific_notation :
  (0.00000006 : ℝ) = 6 * 10^(-8 : ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coronavirus_diameter_scientific_notation_l1228_122857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_apples_is_20_l1228_122804

/-- Represents a distribution of apples among 5 people -/
structure AppleDistribution where
  percentages : Fin 5 → Nat
  sum_to_100 : (Finset.univ.sum (fun i => percentages i)) = 100
  all_positive : ∀ i, percentages i > 0
  all_distinct : ∀ i j, i ≠ j → percentages i ≠ percentages j

/-- The minimum number of apples that can be collected -/
def min_apples : Nat := 20

/-- Theorem stating that 20 is the minimum number of apples that can be collected -/
theorem min_apples_is_20 :
  ∀ n : Nat, (∃ d : AppleDistribution, ∀ i, n % (d.percentages i) = 0) → n ≥ min_apples := by
  sorry

#check min_apples_is_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_apples_is_20_l1228_122804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1228_122840

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 12 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

/-- The distance function between a point (x, y) and the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x + 4*y - 2| / Real.sqrt (3^2 + 4^2)

/-- The minimum distance from the circle to the line is 2 -/
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 ∧
  ∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1228_122840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_rectangular_solid_l1228_122812

/-- A rectangular solid with given surface area and sum of edge lengths -/
structure RectangularSolid where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area_eq : a * b + b * c + a * c = 45 / 4
  sum_edges_eq : a + b + c = 24 / 4

/-- The angle between the body diagonal and an edge of a rectangular solid -/
noncomputable def body_diagonal_angle (solid : RectangularSolid) : ℝ :=
  Real.arccos (min solid.a (min solid.b solid.c) / Real.sqrt (solid.a^2 + solid.b^2 + solid.c^2))

/-- The maximum angle between the body diagonal and an edge of a rectangular solid -/
noncomputable def max_body_diagonal_angle : ℝ := Real.arccos (Real.sqrt 6 / 9)

/-- Theorem stating the maximum angle for a rectangular solid with given conditions -/
theorem max_angle_rectangular_solid (solid : RectangularSolid) :
  body_diagonal_angle solid ≤ max_body_diagonal_angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_rectangular_solid_l1228_122812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1228_122843

/-- Represents a line in the form x/m + y/n = 1 --/
structure Line where
  m : ℝ
  n : ℝ
  m_pos : m > 0
  n_pos : n > 0

/-- The area of the triangle formed by a line and the positive x and y axes --/
noncomputable def triangle_area (l : Line) : ℝ := l.m * l.n / 2

/-- The line passes through the point (1, 2) --/
def passes_through_point (l : Line) : Prop :=
  1 / l.m + 2 / l.n = 1

theorem min_triangle_area (l : Line) (h : passes_through_point l) :
  4 ≤ triangle_area l ∧ ∃ l', passes_through_point l' ∧ triangle_area l' = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1228_122843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1228_122870

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x - Real.pi/3) - Real.sqrt 3

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), (0 < T' ∧ T' < T) → ∃ (x : ℝ), f (x + T') ≠ f x) ∧
    (∀ (k : ℤ), ∃ (c : ℝ), c = Real.pi/6 + k * Real.pi/2 ∧ ∀ (x : ℝ), f (c + x) = f (c - x)) ∧
    (∀ (x y : ℝ), Real.pi/6 < x ∧ x < y ∧ y < 5*Real.pi/6 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1228_122870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_probability_l1228_122835

-- Define the triangle ABC
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

-- Define the properties of the isosceles right triangle
def IsoscelesRightTriangle (t : Triangle) : Prop :=
  t.AB = 8 ∧ t.AC = 8 ∧ t.BC = 8 * Real.sqrt 2

-- Define a point P within the triangle
structure PointInTriangle (t : Triangle) where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ t.AB

-- Define the area of a triangle
noncomputable def TriangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

-- Define the probability function
noncomputable def Probability (t : Triangle) (p : PointInTriangle t) : ℝ :=
  (Real.sqrt 2 / 3) ^ 2

-- Theorem statement
theorem isosceles_right_triangle_probability 
  (t : Triangle) 
  (h : IsoscelesRightTriangle t) 
  (p : PointInTriangle t) : 
  Probability t p = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_probability_l1228_122835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l1228_122849

theorem repeating_decimal_to_fraction (x : ℚ) : 
  (∃ (n : ℕ), x = (27 : ℚ) / (99 * 10^n)) → 
  ∃ (a b : ℤ), x = a / b ∧ b = 11 ∧ Int.gcd a.natAbs b = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l1228_122849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_l1228_122850

noncomputable def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem g_of_5 : g 5 = 17 / 3 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_l1228_122850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1228_122898

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 - 6 * Real.sin x + 2

-- State the theorem
theorem sum_of_max_min_f : 
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1228_122898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_constant_l1228_122826

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (n : ℕ+) (k : ℝ) : ℝ := 3 * (2 : ℝ)^(n : ℝ) + k

/-- Represents the nth term of the geometric sequence -/
noncomputable def a (n : ℕ+) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k
  else S n k - S (n-1) k

theorem geometric_sequence_sum_constant (k : ℝ) :
  (∀ n : ℕ+, a n k = 3 * (2 : ℝ)^((n-1) : ℝ)) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_constant_l1228_122826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_two_position_l1228_122889

-- Define the sequence as noncomputable due to the use of Real.sqrt
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (4 * n + 2)

-- State the theorem
theorem fifth_root_two_position :
  ∃ (n : ℕ), n > 0 ∧ a n = 5 * Real.sqrt 2 ↔ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_two_position_l1228_122889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_f_increasing_on_interval_m_range_for_inequality_l1228_122865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 4 / (x - a)

theorem odd_function_implies_a_zero (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 0 := by sorry

theorem f_increasing_on_interval :
  ∀ x y, 2 ≤ x → x < y → f 0 x < f 0 y := by sorry

theorem m_range_for_inequality (m : ℝ) :
  (∀ x, 2 ≤ x → x ≤ 4 → (f 0 x)^2 - 3*(f 0 x) + m ≤ 0) →
  m ≤ -10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_f_increasing_on_interval_m_range_for_inequality_l1228_122865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parabola_l1228_122895

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the circle
def circle_equation (x y b r : ℝ) : Prop := x^2 + (y - b)^2 = r^2

-- Define tangency condition
def is_tangent (a b r : ℝ) : Prop :=
  circle_equation a (parabola a) b r ∧
  circle_equation (-a) (parabola (-a)) b r

-- Theorem statement
theorem circle_tangent_parabola (a b r : ℝ) :
  is_tangent a b r →
  r = Real.sqrt ((1 + 4 * a^2)^2 / 16 - a^4) :=
by
  sorry

#check circle_tangent_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parabola_l1228_122895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cassini_lemniscate_properties_l1228_122809

/-- Cassini lemniscate -/
structure CassiniLemniscate where
  c : ℝ
  a : ℝ

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : Point) : ℝ :=
  Real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2)

/-- Check if a point is on the Cassini lemniscate -/
def isOnLemniscate (cl : CassiniLemniscate) (p : Point) : Prop :=
  let f₁ : Point := ⟨-cl.c, 0⟩
  let f₂ : Point := ⟨cl.c, 0⟩
  distance p f₁ * distance p f₂ = cl.a^2

/-- Properties of Cassini lemniscate -/
theorem cassini_lemniscate_properties (cl : CassiniLemniscate) :
  (∀ (p : Point), isOnLemniscate cl p → isOnLemniscate cl ⟨-p.x, p.y⟩) ∧ 
  (∀ (p : Point), isOnLemniscate cl p → isOnLemniscate cl ⟨p.x, -p.y⟩) ∧
  (∀ (p : Point), isOnLemniscate cl p → isOnLemniscate cl ⟨-p.x, -p.y⟩) ∧
  (cl.a = cl.c → isOnLemniscate cl ⟨0, 0⟩) ∧
  (0 < cl.a ∧ cl.a < cl.c → ¬∃ (p : Point), isOnLemniscate cl p) ∧
  (0 < cl.c ∧ cl.c < cl.a → ∀ (p : Point), isOnLemniscate cl p → 
    cl.a^2 - cl.c^2 ≤ p.x^2 + p.y^2 ∧ p.x^2 + p.y^2 ≤ cl.a^2 + cl.c^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cassini_lemniscate_properties_l1228_122809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_earnings_in_five_weeks_l1228_122880

/-- Calculate the earnings of a piano teacher over a given number of weeks -/
def teacher_earnings (cost_per_half_hour : ℕ) (hours_per_lesson : ℕ) (lessons_per_week : ℕ) (weeks : ℕ) : ℕ :=
  cost_per_half_hour * (2 * hours_per_lesson) * lessons_per_week * weeks

/-- Prove that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_in_five_weeks :
  teacher_earnings 10 1 1 5 = 100 := by
  unfold teacher_earnings
  ring

#eval teacher_earnings 10 1 1 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_earnings_in_five_weeks_l1228_122880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_C_D_l1228_122827

def C : ℤ := 2*3 + 4*5 + 6*7 + 8*9 + 10*11 + 12*13 + 14*15 + 16*17 + 18*19 + 20*21 + 
           22*23 + 24*25 + 26*27 + 28*29 + 30*31 + 32*33 + 34*35 + 36*37 + 38*39 + 40

def D : ℤ := 2 + 3*4 + 5*6 + 7*8 + 9*10 + 11*12 + 13*14 + 15*16 + 17*18 + 19*20 + 
           21*22 + 23*24 + 25*26 + 27*28 + 29*30 + 31*32 + 33*34 + 35*36 + 37*38 + 39*40

theorem difference_C_D : |C - D| = 1159 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_C_D_l1228_122827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l1228_122858

theorem congruent_integers_count : 
  (Finset.filter (fun n : ℕ => n < 500 ∧ n % 7 = 3) (Finset.range 500)).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l1228_122858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_value_x_min_is_negative_f_x_min_equals_minimum_l1228_122841

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - (3 * Real.sqrt 2) / (2 * x)

-- State the theorem
theorem f_has_minimum_value :
  ∃ (x_min : ℝ), x_min < 0 ∧
  (∀ (x : ℝ), x < 0 → f x ≥ f x_min) ∧
  f x_min = (3/2) * (9 : ℝ)^(1/3) := by
  sorry

-- Define the minimum point
noncomputable def x_min : ℝ := -(18/16)^(1/6)

-- State that x_min is negative
theorem x_min_is_negative : x_min < 0 := by
  sorry

-- State that f(x_min) equals the minimum value
theorem f_x_min_equals_minimum :
  f x_min = (3/2) * (9 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_value_x_min_is_negative_f_x_min_equals_minimum_l1228_122841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_triangle_condition_l1228_122893

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + 2*m

-- Define the interval [1/3, 3]
def I : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 3}

-- Define the condition for three numbers to form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem f_triangle_condition (m : ℝ) :
  (∃ a b c, a ∈ I ∧ b ∈ I ∧ c ∈ I ∧ is_triangle (f m a) (f m b) (f m c)) → m > 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_triangle_condition_l1228_122893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_4_5_7_l1228_122884

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  max / divisor

theorem probability_multiple_4_5_7 (total_cards : ℕ) (h : total_cards = 150) :
  (count_multiples total_cards 4 + count_multiples total_cards 5 + count_multiples total_cards 7 -
   (count_multiples total_cards 20 + count_multiples total_cards 28 + count_multiples total_cards 35) +
   count_multiples total_cards 140 : ℚ) / total_cards = 73 / 150 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_4_5_7_l1228_122884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l1228_122856

theorem cosine_of_angle (α : Real) : 
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x * Real.cos α = x ∧ y * Real.sin α = y) → 
  Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l1228_122856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l1228_122838

noncomputable def angle_between {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] (x y : n) : ℝ :=
  Real.arccos ((inner x y) / (norm x * norm y))

theorem vector_magnitude_problem {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (h1 : angle_between a b = π / 3) (h2 : norm a = 1) (h3 : norm (2 • a - b) = 2 * Real.sqrt 3) :
  norm b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l1228_122838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_circle_intersect_l1228_122833

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ := (2 * t, 1 + 4 * t)

noncomputable def circle_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin θ

def line_equation (x y : ℝ) : Prop := 2 * x - y + 1 = 0

noncomputable def circle_cartesian (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 2)^2 = 2

noncomputable def distance_point_line (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

theorem curve_circle_intersect :
  ∃ (x y : ℝ), line_equation x y ∧ circle_cartesian x y :=
by
  sorry

#check curve_circle_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_circle_intersect_l1228_122833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_derivative_l1228_122808

noncomputable def f (x : ℝ) : ℝ :=
  x^2 * (69.28 * 0.004^x - Real.log (27*x)) / (0.03 * Real.cos (55 * Real.pi / 180))

theorem exists_zero_derivative :
  ∃ x : ℝ, ContinuousOn f Set.univ → (deriv f x = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_derivative_l1228_122808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_not_fully_submerged_l1228_122869

-- Define the container and balls
def container_diameter : ℝ := 22
def water_volume : ℝ := 5000
def ball1_diameter : ℝ := 10
def ball2_diameter : ℝ := 14

-- Define the function to calculate the required water volume
noncomputable def required_water_volume : ℝ :=
  let container_radius : ℝ := container_diameter / 2
  let ball1_radius : ℝ := ball1_diameter / 2
  let ball2_radius : ℝ := ball2_diameter / 2
  let water_height : ℝ := ball2_radius + 2 * Real.sqrt 11 + ball1_radius
  let cylinder_volume : ℝ := Real.pi * container_radius^2 * water_height
  let spheres_volume : ℝ := (4/3) * Real.pi * (ball1_radius^3 + ball2_radius^3)
  cylinder_volume - spheres_volume

-- Theorem statement
theorem balls_not_fully_submerged : required_water_volume > water_volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_not_fully_submerged_l1228_122869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l1228_122847

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_power_of_18_dividing_30_factorial : 
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ m : ℕ, (18^m : ℕ) ∣ factorial 30 → m ≤ n) ∧
  ((18^n : ℕ) ∣ factorial 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l1228_122847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1228_122868

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_decreasing : ∀ x y, x < y → f y < f x
axiom f_domain : ∀ x, f x ≠ 0 → x ≤ 3

-- Define the inequality condition
axiom inequality_condition : ∀ (a x : ℝ), f (a^2 - Real.sin x) ≤ f (a + 1 + Real.cos x^2)

-- Theorem statement
theorem range_of_a :
  ∃ (a : ℝ), (∀ x, f (a^2 - Real.sin x) ≤ f (a + 1 + Real.cos x^2)) ↔ 
  (a ≥ -Real.sqrt 2 ∧ a ≤ (1 - Real.sqrt 10) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1228_122868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_terms_l1228_122887

/-- An arithmetic sequence with a_1 = 1 and a_4 = 7 -/
noncomputable def arithmetic_seq (n : ℕ) : ℝ :=
  let d := (7 - 1) / 3  -- Common difference
  1 + (n - 1 : ℝ) * d

/-- The sum of the first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  n * (arithmetic_seq 1 + arithmetic_seq n) / 2

theorem sum_of_first_four_terms :
  S 4 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_terms_l1228_122887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_characterization_l1228_122897

def has_only_integer_roots (p q : ℕ) : Prop :=
  ∃ x y : ℤ, x^2 - (p*q:ℤ)*x + (p+q:ℤ) = 0 ∧ 
             y^2 - (p*q:ℤ)*y + (p+q:ℤ) = 0 ∧ 
  ∀ z : ℝ, z^2 - (p*q:ℝ)*z + (p+q:ℝ) = 0 → ∃ n : ℤ, z = n

theorem integer_roots_characterization (p q : ℕ) :
  has_only_integer_roots p q ↔ 
  (p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1) ∨ (p = 2 ∧ q = 2) ∨ 
  (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_characterization_l1228_122897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_sum_of_squares_greater_than_sum_of_products_l1228_122851

-- Part I
theorem sqrt_inequality : Real.sqrt 11 - 2 * Real.sqrt 3 > 3 - Real.sqrt 10 := by sorry

-- Part II
theorem sum_of_squares_greater_than_sum_of_products 
  {a b c : ℝ} (h : ¬(a = b ∧ b = c)) : 
  a^2 + b^2 + c^2 > a*b + b*c + c*a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_sum_of_squares_greater_than_sum_of_products_l1228_122851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_eccentricity_l1228_122894

/-- Given an ellipse with foci at (-1,0) and (1,0) that intersects with the line x + y - 3 = 0,
    the maximum eccentricity of the ellipse is √5/5 -/
theorem ellipse_max_eccentricity :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ x + y = 3) →
  (∀ (x : ℝ), x^2/a^2 + ((3-x)^2)/b^2 ≤ 1) →
  (a^2 - b^2 = 4) →
  (∃ (e : ℝ), e = Real.sqrt (a^2 - b^2) / a ∧ e ≤ Real.sqrt 5 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_eccentricity_l1228_122894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1228_122877

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

-- State the theorem
theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x > 0, f a x = 0 → x = x₁ ∨ x = x₂) →  -- f has exactly two zeros
  (0 < x₁) →
  (x₁ < x₂) →
  (x₁ / Real.log x₁ = a) →
  (x₂ / Real.log x₂ = a) →
  (a > Real.exp 1) ∧ (x₁ / Real.log x₁ < 2 * x₂ - x₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1228_122877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l1228_122822

/-- The length of an altitude in a triangle -/
def AltitudeLength (triangle : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

/-- The radius of the inscribed circle in a triangle -/
def InscribedCircleRadius (triangle : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

/-- The length of the base of a triangle -/
def BaseLength (triangle : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

/-- 
Given an isosceles triangle with an altitude of 25 units to the base
and an inscribed circle with radius 8 units, the length of the base is 80/3 units.
-/
theorem isosceles_triangle_base_length 
  (triangle : Set ℝ × Set ℝ × Set ℝ) 
  (h_isosceles : IsIsosceles triangle)
  (h_altitude : AltitudeLength triangle = 25)
  (h_inscribed_radius : InscribedCircleRadius triangle = 8) :
  BaseLength triangle = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l1228_122822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_5_l1228_122874

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Theorem: The area of triangle ABC is 13.5 -/
theorem triangle_area_is_13_5 (C : Point) (h : C.x + C.y = 6) :
  triangleArea ⟨-2, 1⟩ ⟨1, 4⟩ C = 13.5 := by
  sorry

#check triangle_area_is_13_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_5_l1228_122874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jog_to_coffee_shop_time_l1228_122846

/-- Represents the jogging scenario with given parameters -/
structure JoggingScenario where
  total_time : ℚ  -- Time to jog to the park in minutes
  total_distance : ℚ  -- Distance to the park in miles
  half_distance : ℚ  -- Distance to the coffee shop in miles

/-- Calculates the time to jog to the coffee shop given a JoggingScenario -/
def time_to_coffee_shop (scenario : JoggingScenario) : ℚ :=
  (scenario.total_time * scenario.half_distance) / scenario.total_distance

/-- Theorem stating that under the given conditions, it takes 6 minutes to jog to the coffee shop -/
theorem jog_to_coffee_shop_time 
  (scenario : JoggingScenario)
  (h1 : scenario.total_time = 12)
  (h2 : scenario.total_distance = 3/2)
  (h3 : scenario.half_distance = scenario.total_distance / 2) :
  time_to_coffee_shop scenario = 6 := by
  sorry

#eval time_to_coffee_shop { total_time := 12, total_distance := 3/2, half_distance := 3/4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jog_to_coffee_shop_time_l1228_122846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1228_122830

theorem m_range (m : ℝ) : 
  (((0.9 : ℝ)^(1.1 : ℝ))^m < ((1.1 : ℝ)^(0.9 : ℝ))^m) ↔ (m > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1228_122830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_no_intersection_condition_intersection_chord_length_l1228_122837

-- Define the locus of points M
def locus (x y : ℝ) : Prop :=
  (x - 8)^2 + y^2 = 4 * ((x - 2)^2 + y^2)

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the line L
def line_L (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 5

-- Define the intersecting circle
def circle_I (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 8*y + 16 = 0

theorem locus_is_circle :
  ∀ x y : ℝ, locus x y ↔ circle_C x y := by sorry

theorem no_intersection_condition :
  ∀ k : ℝ, (∀ x y : ℝ, ¬(circle_C x y ∧ line_L k x y)) ↔ -3/4 < k ∧ k < 3/4 := by sorry

theorem intersection_chord_length :
  ∃ A B : ℝ × ℝ, 
    (circle_C A.1 A.2 ∧ circle_I A.1 A.2) ∧ 
    (circle_C B.1 B.2 ∧ circle_I B.1 B.2) ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_no_intersection_condition_intersection_chord_length_l1228_122837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_decreasing_l1228_122866

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem f_is_odd_and_decreasing :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_decreasing_l1228_122866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_equals_3H_l1228_122842

noncomputable def H (x : ℝ) : ℝ := Real.log ((2 + x) / (2 - x))

noncomputable def sub (x : ℝ) : ℝ := (4 * x - x^3) / (1 + 4 * x^2)

theorem K_equals_3H : ∀ x : ℝ, 
  x ≠ 2 ∧ x ≠ -2 ∧ 1 + 4 * x^2 ≠ 0 → 
  H (sub x) = 3 * H x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_equals_3H_l1228_122842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_base7_remainder_theorem_l1228_122896

def base7_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 7^2 + d₁ * 7 + d₀

theorem sum_base7_remainder_theorem :
  let a := base7_to_base10 0 2 4  -- 24₇
  let b := base7_to_base10 3 6 4  -- 364₇
  let c := base7_to_base10 0 4 3  -- 43₇
  let d := base7_to_base10 0 1 2  -- 12₇
  let e := base7_to_base10 0 0 3  -- 3₇
  let f := base7_to_base10 0 0 1  -- 1₇
  (a + b + c + d + e + f) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_base7_remainder_theorem_l1228_122896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joanne_part_time_rate_l1228_122818

/-- Joanne's work schedule and earnings --/
structure WorkSchedule where
  main_job_hourly_rate : ℚ
  main_job_hours_per_day : ℚ
  part_time_hours_per_day : ℚ
  days_per_week : ℚ
  total_weekly_earnings : ℚ

/-- Calculate Joanne's part-time hourly rate --/
def part_time_hourly_rate (w : WorkSchedule) : ℚ :=
  let main_job_weekly_earnings := w.main_job_hourly_rate * w.main_job_hours_per_day * w.days_per_week
  let part_time_weekly_earnings := w.total_weekly_earnings - main_job_weekly_earnings
  let part_time_weekly_hours := w.part_time_hours_per_day * w.days_per_week
  part_time_weekly_earnings / part_time_weekly_hours

/-- Theorem: Joanne's part-time hourly rate is $13.50 --/
theorem joanne_part_time_rate :
  let w : WorkSchedule := {
    main_job_hourly_rate := 16
    main_job_hours_per_day := 8
    part_time_hours_per_day := 2
    days_per_week := 5
    total_weekly_earnings := 775
  }
  part_time_hourly_rate w = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joanne_part_time_rate_l1228_122818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_one_half_l1228_122883

noncomputable def f (x : ℝ) : ℝ := x^2 / (2*x + 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| (n+1) => λ x => f (f_n n x)

theorem f_10_one_half :
  f_n 10 (1/2) = 1 / (3^1024 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_one_half_l1228_122883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_pyramid_max_cross_section_sides_l1228_122801

/-- A convex polyhedron formed by two regular pyramids -/
structure DoublePyramid (n : ℕ) :=
  (base_sides : Fin (2*n))
  (is_convex : Bool)
  (all_faces_triangular : Bool)

/-- The maximum number of sides in a planar cross-section of a double pyramid -/
def max_cross_section_sides (n : ℕ) : ℕ := 2*(n+1)

/-- Theorem stating the maximum number of sides in a planar cross-section -/
theorem double_pyramid_max_cross_section_sides (n : ℕ) (h : n ≥ 2) 
  (dp : DoublePyramid n) (hc : dp.is_convex = true) (ht : dp.all_faces_triangular = true) :
  ∀ (cross_section : Finset (Fin (2*n))), cross_section.card ≤ max_cross_section_sides n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_pyramid_max_cross_section_sides_l1228_122801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_odd_f_and_g_are_odd_sorry_l1228_122800

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := x + 1/x
def g (x : ℝ) : ℝ := x^3 + 2*x

-- Theorem statement
theorem f_and_g_are_odd :
  IsOdd f ∧ IsOdd g := by
  constructor
  · -- Proof for f
    intro x
    simp [IsOdd, f]
    ring
  · -- Proof for g
    intro x
    simp [IsOdd, g]
    ring

-- Alternative theorem statement with sorry
theorem f_and_g_are_odd_sorry :
  IsOdd f ∧ IsOdd g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_odd_f_and_g_are_odd_sorry_l1228_122800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheets_for_million_l1228_122828

/-- Calculates the total number of characters needed to print numbers from 1 to n --/
def total_characters (n : ℕ) : ℕ := sorry

/-- Calculates the number of sheets needed given the total number of characters --/
def sheets_needed (total_chars : ℕ) (chars_per_sheet : ℕ) : ℕ := sorry

/-- The main theorem stating the number of sheets needed --/
theorem sheets_for_million : 
  (let total_chars := total_characters 1000000 + 1999998
   let chars_per_sheet := 30 * 60
   sheets_needed total_chars chars_per_sheet) = 4383 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheets_for_million_l1228_122828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1228_122815

noncomputable section

open Real

def f (x : ℝ) : ℝ := -|x|

def a : ℝ := f (log (1 / π))

def b : ℝ := f (log (1 / (exp 1)) / log π)

def c : ℝ := f (log (1 / π^2) / log (1 / (exp 1)))

theorem relationship_abc : b > a ∧ a > c := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1228_122815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inflation_time_p_q_l1228_122890

/-- Represents the time taken to inflate a balloon with given valves open -/
noncomputable def inflation_time (p q r : ℝ) : ℕ → ℝ
| 0 => 0  -- placeholder for unused case
| 1 => 1 / (p + q)
| 2 => 1 / (p + r)
| 3 => 1 / (q + r)
| 4 => 1 / (p + q + r)
| _ => 0  -- placeholder for other cases

/-- The main theorem stating the inflation time with valves P and Q open -/
theorem inflation_time_p_q
  (p q r : ℝ)
  (h1 : p > 0) (h2 : q > 0) (h3 : r > 0)
  (h4 : inflation_time p q r 4 = 2)
  (h5 : inflation_time p q r 2 = 3)
  (h6 : inflation_time p q r 3 = 6) :
  inflation_time p q r 1 = 2 := by
  sorry

#check inflation_time_p_q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inflation_time_p_q_l1228_122890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_to_complex_correspondence_l1228_122839

-- Define a vector in R^2
def vector_AB : ℝ × ℝ := (2, -3)

-- Define the corresponding complex number
def complex_AB : ℂ := 2 - 3 * Complex.I

-- Theorem stating the correspondence
theorem vector_to_complex_correspondence :
  Complex.ofReal vector_AB.1 + Complex.ofReal vector_AB.2 * Complex.I = complex_AB := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_to_complex_correspondence_l1228_122839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1228_122802

/-- The time taken for a train to cross a pole -/
noncomputable def time_to_cross_pole (train_speed_kmph : ℝ) (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  train_length_m / train_speed_mps

/-- Theorem: The time taken for a train of length 800.064 meters, 
    travelling at 160 kmph, to cross a pole is approximately 18.00144 seconds -/
theorem train_crossing_pole_time :
  ∃ ε > 0, |time_to_cross_pole 160 800.064 - 18.00144| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1228_122802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1228_122813

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) 
    and an asymptote 2x - √3y = 0, 
    prove that the eccentricity of the hyperbola is √21/3 -/
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (asymptote : ∀ x y : ℝ, 2 * x - Real.sqrt 3 * y = 0 → 
    x^2 / a^2 - y^2 / b^2 = 1) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1228_122813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_egg_count_l1228_122829

theorem goose_egg_count (total_eggs : ℕ) : 
  (((((((1 : ℚ) / 4 * total_eggs) * 4 / 5) * 3 / 4) * 7 / 8) * 3 / 7) * 3 / 5) * 9 / 10 = 120 →
  total_eggs ≥ 659 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_egg_count_l1228_122829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_sum_l1228_122814

theorem two_digit_number_sum (a b : ℤ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a ≠ b →
  |((10 * a + b) - (10 * b + a))| = 3 * |a - b| →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_sum_l1228_122814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1228_122881

/-- A function that checks if a 7-digit number is valid according to the problem conditions -/
def is_valid_number (n : ℕ) : Bool :=
  (1000000 ≤ n) && (n < 10000000) &&  -- 7-digit number
  (n % 9 = 0) &&                      -- divisible by 9
  ((n / 10) % 10 = 5)                 -- second to last digit is 5

/-- The count of valid numbers according to the problem conditions -/
def count_valid_numbers : ℕ := 
  (Finset.filter (fun n => is_valid_number n = true) (Finset.range 10000000)).card

/-- The theorem stating that the count of valid numbers is 100,000 -/
theorem valid_numbers_count : count_valid_numbers = 100000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1228_122881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redesigned_survey_respondents_l1228_122819

/-- Calculates the number of respondents to a redesigned survey given the data from an original survey and the increase in response rate. -/
theorem redesigned_survey_respondents
  (original_total : ℕ)
  (original_respondents : ℕ)
  (redesigned_total : ℕ)
  (response_rate_increase : ℚ)
  (h1 : original_total = 80)
  (h2 : original_respondents = 7)
  (h3 : redesigned_total = 63)
  (h4 : response_rate_increase = 5 / 100) :
  ∃ (n : ℕ), n = round ((original_respondents / original_total + response_rate_increase) * redesigned_total) ∧ n = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_redesigned_survey_respondents_l1228_122819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roque_bike_trips_l1228_122807

/-- Represents the number of times Roque rides his bike to and from work per week -/
def bike_trips : ℕ → Prop := sorry

/-- Time it takes Roque to walk to work one way (in hours) -/
def walk_time : ℕ := 2

/-- Time it takes Roque to ride his bike to work one way (in hours) -/
def bike_time : ℕ := 1

/-- Number of times Roque walks to and from work per week -/
def walk_trips : ℕ := 3

/-- Total time Roque spends commuting (walking and biking) per week (in hours) -/
def total_commute_time : ℕ := 16

theorem roque_bike_trips : 
  bike_trips 2 ↔ 
  walk_trips * (2 * walk_time) + 2 * (2 * bike_time) = total_commute_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roque_bike_trips_l1228_122807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_calculation_l1228_122844

noncomputable def bob_incorrect (y : ℝ) : Prop := (y - 7) / 5 = 47

noncomputable def bob_correct (y : ℝ) : ℝ := (y - 5) / 7

theorem bob_calculation (y : ℝ) (h : bob_incorrect y) : 
  Int.floor (bob_correct y) = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_calculation_l1228_122844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_four_l1228_122806

theorem cube_root_of_four (x : ℝ) : x^3 = 4 → x = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_four_l1228_122806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_for_logarithmic_equation_l1228_122854

theorem solution_count_for_logarithmic_equation (a : ℝ) :
  let equation := fun x => Real.log (2 * x) / Real.log (x + a) = 2
  ((∀ x, ¬equation x) ↔ a ≥ (1 / 2)) ∧
  ((∃! x, equation x) ↔ a ≤ 0) ∧
  ((∃ x y, x ≠ y ∧ equation x ∧ equation y) ↔ 0 < a ∧ a < (1 / 2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_for_logarithmic_equation_l1228_122854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_equal_chord_l1228_122867

theorem central_angle_of_equal_chord (O A B : EuclideanSpace ℝ (Fin 2)) (r α : ℝ) :
  (∀ P, ‖O - P‖ = r) →  -- O is the center of a circle with radius r
  ‖A - B‖ = r →         -- The length of chord AB is equal to the radius
  ‖O - A‖ = r →         -- A is on the circle
  ‖O - B‖ = r →         -- B is on the circle
  α = Real.arccos ((2 * r^2 - ‖A - B‖^2) / (2 * r^2)) →  -- α is the central angle corresponding to chord AB
  α = π / 3 :=           -- The central angle is π/3 radians
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_equal_chord_l1228_122867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_can_prevent_B_from_winning_l1228_122836

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents a move in the game -/
structure Move where
  digit : Nat
  position : Bool  -- true for right, false for left

/-- Represents the state of the game -/
structure GameState where
  board : List Nat
  currentPlayer : Player

/-- A strategy for player A -/
def Strategy := GameState → Move

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { board := if move.position
              then state.board ++ [move.digit]
              else move.digit :: state.board,
    currentPlayer := match state.currentPlayer with
                     | Player.A => Player.B
                     | Player.B => Player.A }

/-- Converts the board to a number -/
def boardToNumber (board : List Nat) : Nat :=
  board.foldl (fun acc d => acc * 10 + d) 0

/-- Applies a list of moves to the initial game state -/
def applyMoves : List Move → GameState
| [] => GameState.mk [] Player.A
| (move :: moves) => applyMove (applyMoves moves) move

theorem player_A_can_prevent_B_from_winning :
  ∃ (strategy : Strategy),
    ∀ (game : List Move),
      (game.length % 2 = 1) →  -- B's turn just ended
      ¬(isPerfectSquare (boardToNumber (applyMoves game).board)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_can_prevent_B_from_winning_l1228_122836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1228_122816

noncomputable def f (x : ℝ) : ℝ := 2 * (x - 1 / x) - 2 * Real.log x

theorem tangent_line_at_one (x y : ℝ) :
  f 1 = 0 ∧ 
  (∀ x, HasDerivAt f (2 + 2 / x^2 - 2 / x) x) →
  2 * x - y - 2 = 0 ↔ y = f x + (2 + 2 - 2) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1228_122816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1228_122864

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (sin A + sin C) / (c - b) = sin B / (c - a) →
  a = 2 * sqrt 3 →
  (1/2) * b * c * sin A = 2 * sqrt 3 →
  -- Conclusions
  A = π/3 ∧ a + b + c = 6 + 2 * sqrt 3 :=
by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1228_122864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_central_symmetry_distance_l1228_122861

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (A A' A'' O O' : V)

-- Define central symmetry
def centralSymmetry (center point : V) : V :=
  2 • center - point

-- Define the problem statement
theorem double_central_symmetry_distance 
  (h1 : A' = centralSymmetry O A)
  (h2 : A'' = centralSymmetry O' A')
  (h3 : ‖O - O'‖ = a)
  : ‖A - A''‖ = 2 * a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_central_symmetry_distance_l1228_122861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1228_122892

noncomputable section

-- Define the semicircle
def semicircle (x y : ℝ) : Prop := x^2 + y^2 = 2 ∧ x ≥ 0

-- Define point A on the semicircle with slope angle 45°
def point_A : ℝ × ℝ := (1, 1)

-- Define point H as the foot of the perpendicular from A to x-axis
def point_H : ℝ × ℝ := (1, 0)

-- Define the slope of line HB
def slope_HB : ℝ := 1

-- Define point B as the intersection of HB and the semicircle
def point_B : ℝ × ℝ := ((1 + Real.sqrt 3) / 2, (Real.sqrt 3 - 1) / 2)

-- Theorem statement
theorem line_AB_equation :
  ∀ (x y : ℝ),
  semicircle (point_A.1) (point_A.2) →
  (point_B.2 - point_A.2) * (x - point_A.1) = (point_B.1 - point_A.1) * (y - point_A.2) →
  Real.sqrt 3 * x + y - Real.sqrt 3 - 1 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1228_122892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_set_l1228_122891

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 5 ∧
  (∀ n ∈ s, 1 ≤ n ∧ n ≤ 30) ∧
  (∃ k : ℕ, s.prod id = 2^k) ∧
  (∃ a d : ℕ, s = {a, a + d, a + 2*d, a + 3*d, a + 4*d})

theorem unique_valid_set :
  ∃! s : Finset ℕ, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_set_l1228_122891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1228_122832

noncomputable def line_point (t : ℝ) : ℝ × ℝ × ℝ := (3 - t, 2 + 4*t, 2 - 2*t)

def target_point : ℝ × ℝ × ℝ := (5, 1, 6)

noncomputable def closest_point : ℝ × ℝ × ℝ := (47/19, 78/19, 18/19)

theorem closest_point_on_line :
  ∃ (t : ℝ), 
    (line_point t = closest_point) ∧ 
    (∀ (s : ℝ), ‖line_point s - target_point‖ ≥ ‖closest_point - target_point‖) :=
by
  sorry

#check closest_point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1228_122832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a4_value_l1228_122825

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem max_a4_value (seq : ArithmeticSequence) 
  (h1 : S seq 4 ≥ 10) (h2 : S seq 5 ≤ 15) : 
  seq.a 4 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a4_value_l1228_122825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1228_122848

/-- The speed of a train in km/h, given its length in meters and time to cross a fixed point in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem: A train 640 m long that takes 16 seconds to cross a telegraph post has a speed of 144 km/h. -/
theorem train_speed_theorem :
  train_speed 640 16 = 144 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp [div_div]
  -- Perform the calculation
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check train_speed 640 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1228_122848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PJ1J2_l1228_122852

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 28 ∧ qr = 26 ∧ pr = 30

-- Define the orthocenter
def Orthocenter (P Q R Y : ℝ × ℝ) : Prop :=
  -- Y is on the altitude from P
  (Y.2 - P.2) * (Q.1 - R.1) = (Y.1 - P.1) * (Q.2 - R.2) ∧
  -- Y is on the altitude from Q
  (Y.2 - Q.2) * (P.1 - R.1) = (Y.1 - Q.1) * (P.2 - R.2) ∧
  -- Y is on the altitude from R
  (Y.2 - R.2) * (P.1 - Q.1) = (Y.1 - R.1) * (P.2 - Q.2)

-- Define the incenter
def Incenter (A B C I : ℝ × ℝ) : Prop :=
  -- I is equidistant from the sides of the triangle
  let d1 := (I.2 - B.2) * (A.1 - C.1) - (I.1 - B.1) * (A.2 - C.2)
  let d2 := (I.2 - C.2) * (B.1 - A.1) - (I.1 - C.1) * (B.2 - A.2)
  let d3 := (I.2 - A.2) * (C.1 - B.1) - (I.1 - A.1) * (C.2 - B.2)
  abs d1 = abs d2 ∧ abs d2 = abs d3

-- Define the area of a triangle
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Main theorem
theorem area_of_triangle_PJ1J2 (P Q R Y J1 J2 : ℝ × ℝ) :
  Triangle P Q R →
  Orthocenter P Q R Y →
  Incenter P Q Y J1 →
  Incenter P R Y J2 →
  TriangleArea P J1 J2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PJ1J2_l1228_122852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_with_remainder_three_l1228_122805

theorem two_digit_integers_with_remainder_three : 
  (Finset.filter (fun n : ℕ => 
    10 ≤ n ∧ n < 100 ∧ n % 7 = 3) (Finset.range 100)).card = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_with_remainder_three_l1228_122805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_trip_tax_percentage_l1228_122863

/-- Calculates the total tax percentage given spending percentages and tax rates -/
noncomputable def totalTaxPercentage (clothingPercentage foodPercentage otherPercentage : ℝ)
                       (clothingTaxRate foodTaxRate otherTaxRate : ℝ) : ℝ :=
  (clothingPercentage * clothingTaxRate + 
   foodPercentage * foodTaxRate + 
   otherPercentage * otherTaxRate) / 100

/-- Theorem stating that the total tax percentage is 5.5% given the specified conditions -/
theorem shopping_trip_tax_percentage :
  totalTaxPercentage 50 20 30 5 0 10 = 5.5 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_trip_tax_percentage_l1228_122863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_circle_l1228_122853

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Checks if a point is within or on the circle x^2 + y^2 = 16 -/
def isWithinCircle (p : Point) : Bool :=
  p.x ^ 2 + p.y ^ 2 ≤ 16

/-- Generates all possible points from throwing two dice -/
def allPoints : List Point :=
  List.join (List.map (fun x =>
    List.map (fun y =>
      Point.mk x y) [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6])

/-- Counts the number of points within or on the circle -/
def countPointsWithinCircle : Nat :=
  (List.filter isWithinCircle allPoints).length

theorem probability_within_circle :
  (countPointsWithinCircle : ℚ) / (allPoints.length : ℚ) = 2 / 9 := by
  sorry

#eval countPointsWithinCircle
#eval allPoints.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_circle_l1228_122853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_hypothesis_test_l1228_122834

/-- Test statistic for variance hypothesis test -/
noncomputable def chi_square_statistic (n : ℕ) (s_squared : ℝ) (sigma_squared : ℝ) : ℝ :=
  (n - 1 : ℝ) * s_squared / sigma_squared

/-- Critical value for right-tailed chi-square test -/
noncomputable def chi_square_critical (alpha : ℝ) (df : ℕ) : ℝ := sorry

/-- Theorem: The null hypothesis is not rejected for the given sample -/
theorem variance_hypothesis_test (n : ℕ) (s_squared : ℝ) (sigma_squared : ℝ) (alpha : ℝ) :
  n = 21 →
  s_squared = 16.2 →
  sigma_squared = 15 →
  alpha = 0.01 →
  chi_square_statistic n s_squared sigma_squared < chi_square_critical alpha (n - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_hypothesis_test_l1228_122834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_zero_l1228_122810

-- Define a real-valued function f with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the inverse function
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the problem conditions
axiom f_property : ∀ x : ℝ, f (-x) + f x = 3

-- State the theorem to be proved
theorem inverse_sum_zero : 
  ∀ x : ℝ, f_inv f (x - 1) + f_inv f (4 - x) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_zero_l1228_122810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_is_four_l1228_122859

-- Define the curves C and M
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

noncomputable def curve_M (t : ℝ) : ℝ × ℝ :=
  ((2 / (2 * t - 1)), (2 * t / (2 * t - 1)))

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ θ t, curve_C θ = p ∧ curve_M t = p ∧ t > 1/2}

-- Define the slope of a line from origin to a point
noncomputable def slope_from_origin (p : ℝ × ℝ) : ℝ :=
  p.2 / p.1

-- State the theorem
theorem sum_of_slopes_is_four :
  ∀ A B, A ∈ intersection_points → B ∈ intersection_points →
  A ≠ B →
  slope_from_origin A + slope_from_origin B = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_is_four_l1228_122859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_sum_theorem_stated_result_l1228_122845

/-- The probability that the straight-line distance between two randomly chosen points 
    on the sides of a unit square is at least 1/2 -/
noncomputable def probability_distance_at_least_half : ℝ :=
  (26 - Real.pi) / 32

/-- Theorem stating that the probability is equal to (26-π)/32 -/
theorem probability_theorem :
  probability_distance_at_least_half = (26 - Real.pi) / 32 := by
  -- The proof is omitted for now
  sorry

/-- Compute the sum of a, b, and c -/
def sum_a_b_c : ℕ := 26 + 1 + 32

/-- Theorem stating that the sum of a, b, and c is 59 -/
theorem sum_theorem : sum_a_b_c = 59 := by
  -- The proof is trivial as it's a direct computation
  rfl

-- We can't use #eval for noncomputable definitions, so we'll just state the result
theorem stated_result : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  probability_distance_at_least_half = (a - b * Real.pi) / c ∧
  a + b + c = 59 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_sum_theorem_stated_result_l1228_122845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_third_l1228_122821

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem derivative_f_at_pi_third :
  deriv f (Real.pi / 3) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_third_l1228_122821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1228_122831

noncomputable section

/-- A function f is monotonically increasing if for all x and y, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x - (1/3)sin(2x) + a*sin(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

theorem monotone_f_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → -1/3 ≤ a ∧ a ≤ 1/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1228_122831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_power_series_solution_l1228_122886

/-- The sum of the infinite geometric power series -/
noncomputable def S (x : ℝ) : ℝ := 1 + x + 2*x^2 + 3*x^3 + 4*x^4 + ∑' n, (n+5)*x^(n+4)

theorem geometric_power_series_solution :
  ∃ (x : ℝ), S x = 16 ∧ x = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_power_series_solution_l1228_122886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l1228_122875

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

-- Define the inverse function of f
noncomputable def f_inv (b : ℝ) (y : ℝ) : ℝ := 
  (b / y + 4) / 3

-- State the theorem
theorem product_of_b_values :
  ∃ b₁ b₂ : ℝ, b₁ ≠ b₂ ∧ 
    b₁ ≠ 0 ∧ b₁ ≠ 20/3 ∧
    b₂ ≠ 0 ∧ b₂ ≠ 20/3 ∧
    f b₁ 3 = f_inv b₁ (2*b₁ - 1) ∧
    f b₂ 3 = f_inv b₂ (2*b₂ - 1) ∧
    b₁ * b₂ = 10/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l1228_122875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_with_more_multiples_l1228_122871

theorem max_interval_length_with_more_multiples (m n : ℕ) :
  m < n →
  (∀ k ∈ Set.Ico m n, k % 2021 = 0 → k % 2000 ≠ 0) →
  (∃ k ∈ Set.Ico m n, k % 2021 = 0) →
  n - m ≤ 191999 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_with_more_multiples_l1228_122871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_set_l1228_122873

theorem sin_cos_equation_solution_set (a : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 - (2 * a + 1) * Real.cos x - a ^ 2 = 0) ↔ 
  -5/4 ≤ a ∧ a ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_set_l1228_122873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_polynomial_l1228_122899

-- Define a polynomial type
structure MyPolynomial (α : Type*) where
  coeffs : List (α × ℕ)

-- Define a function to check if a polynomial has a negative coefficient
def has_negative_coefficient (p : MyPolynomial ℝ) : Prop :=
  ∃ (c : ℝ) (n : ℕ), (c, n) ∈ p.coeffs ∧ c < 0

-- Define a function to check if all coefficients of a polynomial are positive
def all_coefficients_positive (p : MyPolynomial ℝ) : Prop :=
  ∀ (c : ℝ) (n : ℕ), (c, n) ∈ p.coeffs → c > 0

-- Define the power operation for polynomials
def pow_poly (p : MyPolynomial ℝ) (n : ℕ) : MyPolynomial ℝ :=
  sorry

-- The main theorem
theorem exists_special_polynomial :
  ∃ (p : MyPolynomial ℝ),
    has_negative_coefficient p ∧
    ∀ (n : ℕ), n > 1 → all_coefficients_positive (pow_poly p n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_polynomial_l1228_122899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_accessible_area_l1228_122823

noncomputable section

/-- The area accessible to a point tied to the corner of a rectangle --/
def accessibleArea (width length radius : ℝ) : ℝ :=
  (3/4) * Real.pi * radius^2 + 
  (1/2) * Real.pi * (max (radius - width) 0)^2 +
  (1/2) * Real.pi * (max (radius - length) 0)^2

/-- Theorem stating the accessible area for the given dimensions --/
theorem goat_accessible_area :
  accessibleArea 3 4 4 = 12.5 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_accessible_area_l1228_122823
