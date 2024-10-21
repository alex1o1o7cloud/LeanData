import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_from_rectangle_l1007_100744

/-- The volume of a cylinder formed by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (short_side long_side : ℝ) 
  (short_side_pos : 0 < short_side) (long_side_pos : 0 < long_side) 
  (long_side_longer : short_side ≤ long_side) : 
  let radius := short_side / 2
  let height := long_side
  let volume := π * radius^2 * height
  short_side = 8 ∧ long_side = 16 → volume = 256 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_from_rectangle_l1007_100744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1007_100749

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse a b) : ℝ × ℝ := (-(Real.sqrt (a^2 - b^2)), Real.sqrt (a^2 - b^2))

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse a b) : Type :=
  (x y : ℝ)
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)

/-- Condition that AF₁ is perpendicular to F₁F₂ -/
def perpendicular_condition (e : Ellipse a b) (A : PointOnEllipse e) : Prop :=
  let (F₁, F₂) := foci e
  (A.x - F₁) * (F₂ - F₁) + A.y * 0 = 0

/-- Condition that AF₂ = 3F₂B -/
def vector_condition (e : Ellipse a b) (A B : PointOnEllipse e) : Prop :=
  let (_, F₂) := foci e
  (A.x - F₂) = 3 * (F₂ - B.x) ∧ (A.y - 0) = 3 * (0 - B.y)

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) (e : Ellipse a b) 
  (A B : PointOnEllipse e) 
  (h_perp : perpendicular_condition e A)
  (h_vec : vector_condition e A B) :
  let (F₁, _) := foci e
  (Real.sqrt ((A.x - F₁)^2 + A.y^2) = 2*a/3) ∧
  (Real.sqrt (a^2 - b^2) / a = Real.sqrt 3 / 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1007_100749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_6_not_exists_l1007_100757

/-- The function f calculates the remainder when x^2 is divided by 13 -/
def f (x : ℕ) : ℕ := x^2 % 13

/-- The n-th iteration of f -/
def f_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

/-- The order of 6 with respect to f does not exist -/
theorem order_of_6_not_exists : ∀ n : ℕ, f_iter n 6 ≠ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_6_not_exists_l1007_100757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_zero_in_interval_l1007_100704

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.log x)

-- Define the interval
def interval : Set ℝ := {x | 1 < x ∧ x < Real.exp Real.pi}

-- Theorem statement
theorem g_has_one_zero_in_interval : 
  ∃! x, x ∈ interval ∧ g x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_zero_in_interval_l1007_100704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_distance_l1007_100719

-- Define the line l
def line (a : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x + (2 - Real.sqrt 3) * a

-- Define the circle C
def curve (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Define the distance from a point to the line
noncomputable def distance_to_line (a x y : ℝ) : ℝ :=
  |Real.sqrt 3 * x - Real.sqrt 3 * a + 2 * a - y| / 2

-- Theorem statement
theorem unique_point_distance (a : ℝ) :
  (∃! p : ℝ × ℝ, curve p.1 p.2 ∧ distance_to_line a p.1 p.2 = Real.sqrt 3 - 1) →
  (a = -6 * Real.sqrt 3 - 9 ∨ a = 2 * Real.sqrt 3 + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_distance_l1007_100719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_is_135_l1007_100776

noncomputable def a : ℝ := ∫ x in (0)..(2), (2*x + 1)

noncomputable def coefficient_x_squared : ℝ := 
  (Nat.choose 6 2) * (-a/2)^2

theorem coefficient_is_135 : coefficient_x_squared = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_is_135_l1007_100776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l1007_100777

noncomputable section

-- Define the points A and B
def A : ℝ × ℝ := (0, 14)
def B : ℝ × ℝ := (0, 4)

-- Define the hyperbola
def hyperbola (x : ℝ) : ℝ := 1 / x

-- Define the parallel lines
def line1 (k : ℝ) (x : ℝ) : ℝ := k * x + 14
def line2 (k : ℝ) (x : ℝ) : ℝ := k * x + 4

-- Define the intersection points
noncomputable def K (k : ℝ) : ℝ × ℝ := sorry
noncomputable def L (k : ℝ) : ℝ × ℝ := sorry
noncomputable def M (k : ℝ) : ℝ × ℝ := sorry
noncomputable def N (k : ℝ) : ℝ × ℝ := sorry

-- Define the distances
noncomputable def AL (k : ℝ) : ℝ := sorry
noncomputable def AK (k : ℝ) : ℝ := sorry
noncomputable def BN (k : ℝ) : ℝ := sorry
noncomputable def BM (k : ℝ) : ℝ := sorry

-- Theorem statement
theorem intersection_ratio : 
  ∀ k : ℝ, (AL k - AK k) / (BN k - BM k) = 3.5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l1007_100777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1007_100730

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Statement of the theorem -/
theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = 2 * Real.sqrt 2 / 3) 
  (h4 : b = 1) :
  (∃ (major_axis : ℝ), major_axis = 6) ∧ 
  (∃ (T : ℝ × ℝ), 
    T.1 = -19 * Real.sqrt 2 / 9 ∧ T.2 = 0 ∧
    ∀ (A B : ℝ × ℝ), 
      ellipse a b A.1 A.2 → ellipse a b B.1 B.2 →
      ((A.1 - T.1) * (B.1 - T.1) + (A.2 - T.2) * (B.2 - T.2) = -7/81)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1007_100730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cube_equation_solution_l1007_100722

theorem square_cube_equation_solution (a b c : ℕ+) 
  (h1 : ∃ k : ℕ+, a * b = k ^ 2)
  (h2 : (a : ℝ) + b + c - 3 * ((a * b * c : ℝ) ^ (1/3)) = 1) :
  ∃ d : ℕ+, a = d ^ 2 ∧ b = d ^ 2 + 2 * d + 1 ∧ c = d ^ 2 + d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cube_equation_solution_l1007_100722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_representation_l1007_100789

-- Define the diamond
def is_in_diamond (x y : ℝ) : Prop := abs x + abs y ≤ 2 * Real.sqrt (x^2 + y^2)

-- Define the circle
def is_in_circle (x y : ℝ) : Prop := 2 * Real.sqrt (x^2 + y^2) ≤ 3

-- Define the square
def is_in_square (x y : ℝ) : Prop := 3 * max (abs x) (abs y) ≤ 3

-- Theorem statement
theorem inequality_representation :
  ∀ x y : ℝ,
  (is_in_diamond x y ∧ is_in_circle x y ∧ is_in_square x y) ↔
  (x^2 + y^2 ≤ (3/2)^2 ∧ abs x + abs y ≤ 2*Real.sqrt 2 ∧ max (abs x) (abs y) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_representation_l1007_100789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l1007_100703

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

-- State the theorem
theorem t_100_mod_7 : T 100 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l1007_100703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_password_count_l1007_100702

def is_valid_password (p : Fin 10 × Fin 10 × Fin 10 × Fin 10) : Prop :=
  ¬(p.1 = 1 ∧ p.2.1 = 2 ∧ p.2.2.1 = 3) ∨
  (p.1 = 1 ∧ p.2.1 = 2 ∧ p.2.2.1 = 3 ∧ p.2.2.2 ≠ 4 ∧ p.2.2.2 ≠ 5)

instance : DecidablePred is_valid_password :=
  fun p => by
    cases p
    apply Or.decidable

theorem valid_password_count :
  (Finset.filter is_valid_password (Finset.univ : Finset (Fin 10 × Fin 10 × Fin 10 × Fin 10))).card = 9992 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_password_count_l1007_100702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1007_100764

theorem product_remainder (a b c : ℕ) :
  a % 7 = 2 → b % 7 = 3 → c % 7 = 4 → (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1007_100764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_same_color_at_distance_l1007_100723

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem two_points_same_color_at_distance (x : ℝ) (h : x > 0) :
  ∃ (c : Color) (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_same_color_at_distance_l1007_100723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1007_100773

/-- The circle C with equation x^2 + y^2 - 2x - 15 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- The radius of circle C -/
noncomputable def radius_C : ℝ := 4

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : Ellipse) :
  e.a = radius_C →
  eccentricity e = 1/2 →
  ∀ x y : ℝ, standard_equation e x y ↔ x^2/4 + y^2/3 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1007_100773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jaco_gift_budget_l1007_100795

/-- Calculate the total budget for Jaco's gifts --/
theorem jaco_gift_budget : 
  (8 : ℕ) * (9 : ℕ) + (2 : ℕ) * (14 : ℕ) = (100 : ℕ) := by
  -- Evaluate the left-hand side
  calc
    (8 : ℕ) * (9 : ℕ) + (2 : ℕ) * (14 : ℕ)
    = 72 + 28 := by ring
    _ = 100 := by rfl

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jaco_gift_budget_l1007_100795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1007_100762

/-- The area of a circular sector with radius r and central angle α (in radians) is (1/2) * r^2 * α -/
noncomputable def sector_area (r : ℝ) (α : ℝ) : ℝ := (1/2) * r^2 * α

/-- Theorem: The area of a sector with radius 3 cm and central angle 2 radians is 9 cm^2 -/
theorem sector_area_example : sector_area 3 2 = 9 := by
  -- Unfold the definition of sector_area
  unfold sector_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1007_100762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_l1007_100724

theorem triangle_geometric_sequence (a b c : ℝ) (B : ℝ) :
  a > 0 → b > 0 → c > 0 →
  b^2 = a * c →  -- Condition for geometric sequence
  c = 2 * a →
  Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c) →  -- Cosine law
  Real.cos B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_l1007_100724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_sum_of_m_values_l1007_100700

/-- The piecewise function g(x) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x^2 + 4*x + 3 else 3*x + 9

/-- Theorem stating that the sum of all values of m that make g(x) continuous is -1 -/
theorem sum_of_continuity_points (m : ℝ) :
  (∀ x : ℝ, ContinuousAt (g m) x) → (m = 2 ∨ m = -3) :=
by sorry

/-- The main theorem proving that the sum of all values of m that make g(x) continuous is -1 -/
theorem sum_of_m_values :
  ∃ m₁ m₂ : ℝ, (∀ x : ℝ, ContinuousAt (g m₁) x) ∧ (∀ x : ℝ, ContinuousAt (g m₂) x) ∧ m₁ + m₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_sum_of_m_values_l1007_100700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_common_divisors_l1007_100763

/-- The number of positive integer divisors common to both 90 and 150 -/
def common_divisors_count : ℕ := 8

/-- 90 and 150 are the numbers we're considering -/
def n1 : ℕ := 90
def n2 : ℕ := 150

/-- The theorem stating that the count of common divisors of 90 and 150 is 8 -/
theorem count_common_divisors :
  (Finset.filter (λ x => x ∣ n1 ∧ x ∣ n2) (Finset.range (min n1 n2 + 1))).card = common_divisors_count :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_common_divisors_l1007_100763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1007_100710

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 4

-- Define the line
def line_eq (x y : ℝ) : Prop := 3*x + 4*y - 16 = 0

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |3*x + 4*y - 16| / Real.sqrt (3^2 + 4^2)

-- Theorem statement
theorem circle_line_intersection :
  ∃! (s : Finset (ℝ × ℝ)), 
    (∀ p ∈ s, circle_eq p.1 p.2 ∧ distance_to_line p.1 p.2 = 1) ∧ 
    s.card = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1007_100710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_screen_time_is_two_hours_l1007_100765

/-- Represents the screen time in minutes -/
def ScreenTime := ℕ

/-- Converts minutes to hours -/
def minutesToHours (minutes : ℕ) : ℚ :=
  minutes / 60

/-- The morning screen time in minutes -/
def morningScreenTime : ℕ := 45

/-- The evening screen time in minutes -/
def eveningScreenTime : ℕ := 75

/-- The total screen time in hours -/
def totalScreenTimeHours : ℚ := minutesToHours (morningScreenTime + eveningScreenTime)

/-- Theorem stating that the total screen time is 2 hours -/
theorem total_screen_time_is_two_hours : totalScreenTimeHours = 2 := by
  -- Unfold definitions
  unfold totalScreenTimeHours minutesToHours morningScreenTime eveningScreenTime
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_screen_time_is_two_hours_l1007_100765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l1007_100718

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2/9 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem max_distance_between_curves :
  ∃ (M : ℝ), M = 8 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    distance x₁ y₁ x₂ y₂ ≤ M) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l1007_100718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1007_100769

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.sin α = Real.sqrt 6 / 3) : 
  Real.tan α = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1007_100769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_odd_sections_polyhedron_l1007_100758

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Add necessary fields for a convex polyhedron

/-- A plane in 3D space. -/
structure Plane where
  -- Add necessary fields for a plane

/-- A polygon. -/
structure Polygon where
  sides : ℕ

/-- Represents whether a plane passes through a vertex of a polyhedron. -/
def passes_through_vertex (p : Plane) (c : ConvexPolyhedron) : Prop :=
  sorry

/-- The section of a polyhedron by a plane. -/
def polyhedron_section (c : ConvexPolyhedron) (p : Plane) : Polygon :=
  sorry

/-- States that all sections of a polyhedron by planes not passing through vertices
    are polygons with an odd number of sides. -/
def all_sections_odd (c : ConvexPolyhedron) : Prop :=
  ∀ p : Plane, ¬(passes_through_vertex p c) → Odd (polyhedron_section c p).sides

/-- Theorem stating that there does not exist a convex polyhedron
    such that all its sections by planes not passing through vertices
    are polygons with an odd number of sides. -/
theorem no_all_odd_sections_polyhedron :
  ¬∃ c : ConvexPolyhedron, all_sections_odd c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_odd_sections_polyhedron_l1007_100758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_seven_sum_of_squares_l1007_100775

theorem cos_power_seven_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + 
                        b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + 
                        b₇ * Real.cos (7 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 1716 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_seven_sum_of_squares_l1007_100775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1007_100748

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

-- State the theorem
theorem inequality_solution :
  ∀ x : ℝ, x ≠ -2 ∧ x ≠ 2 →
  (f x ≤ 0 ↔ x < -2 ∨ (0 < x ∧ x ≤ 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1007_100748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_powers_of_three_l1007_100738

theorem two_digit_powers_of_three : 
  ∃! k : ℕ, k = (Finset.filter (λ n : ℕ ↦ 10 ≤ 3^n ∧ 3^n ≤ 99) (Finset.range 100)).card ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_powers_of_three_l1007_100738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt8_same_type_as_sqrt2_l1007_100735

-- Define a function to represent "same type" relation
def same_type (a b : ℝ) : Prop := ∃ (q : ℚ), a = q * b

-- Define the square roots we're considering
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt4 : ℝ := Real.sqrt 4
noncomputable def sqrt6 : ℝ := Real.sqrt 6
noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def sqrt10 : ℝ := Real.sqrt 10

-- Theorem statement
theorem sqrt8_same_type_as_sqrt2 : same_type sqrt8 sqrt2 := by
  -- Existential introduction
  use 2
  -- Simplify the goal
  simp [same_type, sqrt8, sqrt2]
  -- The proof of sqrt 8 = 2 * sqrt 2 would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt8_same_type_as_sqrt2_l1007_100735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_condition_l1007_100705

/-- The function f(x) = x^2 + a*x + b*cos(x) -/
noncomputable def f (a b x : ℝ) : ℝ := x^2 + a*x + b*(Real.cos x)

/-- The set of real solutions to f(x) = 0 -/
def S (a b : ℝ) : Set ℝ := {x : ℝ | f a b x = 0}

/-- The set of real solutions to f(f(x)) = 0 -/
def T (a b : ℝ) : Set ℝ := {x : ℝ | f a b (f a b x) = 0}

theorem function_equality_condition (a b : ℝ) :
  (S a b = T a b ∧ S a b ≠ ∅) ↔ (0 ≤ a ∧ a < 4 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_condition_l1007_100705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_inequality_l1007_100785

def S : Finset ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3}

theorem count_satisfying_inequality :
  (S.filter (fun x => -3 * x^2 < -14)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_inequality_l1007_100785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_result_proof_l1007_100780

def correct_answer : Nat := 555681

def incorrect_result : Nat := 995681

theorem incorrect_result_proof :
  (correct_answer = 555681) →
  (incorrect_result.repr.count '9' = 2) →
  (∀ d : Char, d ≠ '9' → d.isDigit → 
    (incorrect_result.repr.count d = correct_answer.repr.count d)) →
  incorrect_result = 995681 := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#eval incorrect_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_result_proof_l1007_100780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_rate_calculation_l1007_100760

/-- Calculates the monthly interest rate on a loan given the following conditions:
  * loan_amount: The amount of the loan in rupees
  * deposit_amount: The amount of the fixed deposit in rupees
  * deposit_rate: The monthly interest rate on the fixed deposit as a decimal
  * weekly_interest: The interest amount for 7 days in rupees
  * days_in_month: The number of days in a month
Returns the monthly interest rate on the loan as a decimal. -/
noncomputable def calculate_loan_rate (loan_amount deposit_amount : ℝ) (deposit_rate : ℝ) 
  (weekly_interest : ℝ) (days_in_month : ℕ) : ℝ :=
  let monthly_interest := weekly_interest * (days_in_month : ℝ) / 7
  monthly_interest / loan_amount

/-- Theorem stating that under the given conditions, the monthly interest rate
    on the loan is approximately 0.051614 (5.1614%). -/
theorem loan_rate_calculation (loan_amount deposit_amount : ℝ) (deposit_rate : ℝ) 
  (weekly_interest : ℝ) (days_in_month : ℕ) :
  loan_amount = 15000 →
  deposit_amount = 10000 →
  deposit_rate = 0.095 →
  weekly_interest = 180.83 →
  days_in_month = 30 →
  abs (calculate_loan_rate loan_amount deposit_amount deposit_rate weekly_interest days_in_month - 0.051614) < 0.000001 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_rate_calculation_l1007_100760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_2_f_max_value_f_min_period_l1007_100766

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.cos (2 * x) + 1

-- Theorem 1: f(π/2) = 2
theorem f_at_pi_over_2 : f (Real.pi / 2) = 2 := by sorry

-- Theorem 2: The maximum value of f(x) is √2 + 1
theorem f_max_value : ∃ x : ℝ, f x = Real.sqrt 2 + 1 ∧ ∀ y : ℝ, f y ≤ Real.sqrt 2 + 1 := by sorry

-- Theorem 3: The minimum positive period of f(x) is π
theorem f_min_period : ∀ x : ℝ, f (x + Real.pi) = f x ∧ 
  ∀ p : ℝ, 0 < p ∧ p < Real.pi → ∃ y : ℝ, f (y + p) ≠ f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_2_f_max_value_f_min_period_l1007_100766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_explosion_velocity_l1007_100756

/-- The magnitude of the velocity of the second fragment after a firecracker explosion -/
theorem firecracker_explosion_velocity (u v_x1 : ℝ) (t g : ℝ) : 
  u = 20 →
  t = 3 →
  g = 10 →
  v_x1 = 48 →
  Real.sqrt ((-(u - g * t) - v_x1)^2 + (-(u - g * t))^2) = Real.sqrt 2404 :=
by
  intros h_u h_t h_g h_v_x1
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_explosion_velocity_l1007_100756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_divisibility_l1007_100728

theorem largest_n_divisibility : 
  ∃ (n : ℕ), n = 6 ∧ 
  (∀ (m : ℕ), m > n → ¬(Nat.factorial (Nat.factorial (Nat.factorial m)) ∣ Nat.factorial (Nat.factorial 2004))) ∧
  (Nat.factorial (Nat.factorial (Nat.factorial n)) ∣ Nat.factorial (Nat.factorial 2004)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_divisibility_l1007_100728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1007_100788

-- Define the function x(t)
noncomputable def x (t a : ℝ) : ℝ := 5 * (t + 1)^2 + a / (t + 1)^5

-- State the theorem
theorem min_a_value :
  let min_a := 2 * Real.sqrt ((24/7)^7)
  ∀ a : ℝ, (∀ t : ℝ, t ≥ 0 → x t a ≥ 24) ↔ a ≥ min_a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1007_100788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_factorial_digits_l1007_100708

theorem seventeen_factorial_digits (a b : ℕ) : 
  (a < 10) → 
  (b < 10) → 
  (Nat.factorial 17 = 355687 * 10000 + a * 1000 + b * 100 + 8096000) → 
  (a * 10 + b = 75) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_factorial_digits_l1007_100708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_achievable_is_two_l1007_100716

/-- The smallest achievable integer in the number reduction game -/
def smallest_achievable (n : ℕ) : ℕ := 2

/-- The game operation of replacing two numbers with their arithmetic mean -/
def arithmetic_mean_replacement (a b : ℚ) : ℚ := (a + b) / 2

/-- Predicate to check if a list of lists represents a valid sequence of moves -/
def valid_sequence (n : ℕ) (sequence : List (List ℕ)) : Prop :=
  sequence.head? = some (List.range' 1 n) ∧
  sequence.getLast? ≠ none ∧
  sequence.getLast?.get!.length = 1 ∧
  ∀ (i : ℕ) (hi : i < sequence.length - 1),
    ∃ (a b : ℕ) (rest : List ℕ),
      sequence[i]! = a :: b :: rest ∧
      sequence[i+1]! = Int.toNat (⌊(arithmetic_mean_replacement a b)⌋) :: rest

theorem smallest_achievable_is_two (n : ℕ) (h : n ≥ 3) :
  ∀ (final : ℕ), (∃ (sequence : List (List ℕ)),
    valid_sequence n sequence ∧
    sequence.getLast?.get! = [final]) →
  final ≥ smallest_achievable n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_achievable_is_two_l1007_100716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_at_600m_l1007_100768

-- Define the atmospheric pressure function
noncomputable def atmospheric_pressure (c k x : ℝ) : ℝ := c * Real.exp (k * x)

-- Define the theorem
theorem pressure_at_600m (c k : ℝ) :
  -- Given conditions
  atmospheric_pressure c k 0 = 1.01e5 →
  atmospheric_pressure c k 1000 = 0.90e5 →
  -- Conclusion
  ∃ ε > 0, |atmospheric_pressure c k 600 - 9.43e4| < ε :=
by
  -- Proof goes here
  sorry

#check pressure_at_600m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_at_600m_l1007_100768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_after_ice_l1007_100761

/-- The unoccupied volume in a cube container after adding ice cubes -/
theorem unoccupied_volume_after_ice (container_side : ℝ) (ice_cube_side : ℝ) 
  (num_ice_cubes : ℕ) (initial_water_fraction : ℝ) : 
  container_side = 12 →
  ice_cube_side = 1.5 →
  num_ice_cubes = 20 →
  initial_water_fraction = 1/3 →
  let container_volume := container_side ^ 3
  let initial_water_volume := initial_water_fraction * container_volume
  let ice_cube_volume := ice_cube_side ^ 3
  let total_ice_volume := (num_ice_cubes : ℝ) * ice_cube_volume
  let occupied_volume := initial_water_volume + total_ice_volume
  container_volume - occupied_volume = 1084.5 := by
  sorry

#check unoccupied_volume_after_ice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_after_ice_l1007_100761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l1007_100741

/-- The point on the line 4x + 3y = 24 that is closest to (2,1) -/
noncomputable def closest_point : ℝ × ℝ := (152/25, 64/25)

/-- The line 4x + 3y = 24 -/
def line (p : ℝ × ℝ) : Prop := 4 * p.1 + 3 * p.2 = 24

/-- The point (2,1) -/
def given_point : ℝ × ℝ := (2, 1)

/-- Theorem stating that closest_point is on the line and is closest to given_point -/
theorem closest_point_is_correct :
  line closest_point ∧
  ∀ p : ℝ × ℝ, line p → ‖p - given_point‖ ≥ ‖closest_point - given_point‖ := by
  sorry

#check closest_point_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l1007_100741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1007_100782

theorem trigonometric_identity (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : α - β = π / 3) : 
  Real.sin α ^ 2 + Real.cos β ^ 2 - Real.sqrt 3 * Real.sin α * Real.cos β = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1007_100782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_p_or_greater_sqrt_l1007_100786

variable (p : ℕ) (hp : Prime p)
variable (x y : ℤ)

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => x * a (n + 1) + y * a n + 1

theorem gcd_a_p_or_greater_sqrt (p : ℕ) (hp : Prime p) (x y : ℤ) :
  (Int.gcd (a x y p) (a x y (p + 1)) = 1) ∨
  (Int.gcd (a x y p) (a x y (p + 1)) > Int.floor (Real.sqrt (p : ℝ))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_p_or_greater_sqrt_l1007_100786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l1007_100746

/-- The Law of Sines for a triangle inscribed in a circle -/
theorem law_of_sines (A B C : ℝ) (a b c R : ℝ) : 
  ∃ (O : ℂ), 
    (A + B + C = π) →
    (a = 2 * R * Real.sin A) →
    (b = 2 * R * Real.sin B) →
    (c = 2 * R * Real.sin C) →
    (a / Real.sin A = b / Real.sin B) ∧ 
    (b / Real.sin B = c / Real.sin C) ∧
    (a / Real.sin A = 2 * R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l1007_100746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l1007_100779

/-- Calculates the speed of a train in km/hr given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Proves that the speed of a train is 30 km/hr given specific conditions -/
theorem train_speed_proof (length : ℝ) (time : ℝ) 
    (h1 : length = 75) 
    (h2 : time = 9) : 
  train_speed length time = 30 := by
  sorry

-- Use #eval only for nat, but we're using real numbers here
-- So we'll use #check instead to verify the type and structure
#check train_speed 75 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l1007_100779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_power_identity_l1007_100794

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem rotation_matrix_power_identity :
  ∃ (n : ℕ), n > 0 ∧ rotation_matrix (2 * Real.pi / 3)^n = 1 ∧
  ∀ (m : ℕ), 0 < m ∧ m < n → rotation_matrix (2 * Real.pi / 3)^m ≠ 1 :=
by sorry

#check rotation_matrix_power_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_power_identity_l1007_100794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_b_1_b_value_when_a_2_range_of_b_over_a_l1007_100784

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^2 - abs (a * x - b)

-- Theorem for part 1
theorem min_value_when_a_b_1 :
  ∀ x : ℝ, f 1 1 x ≥ -5/4 ∧ ∃ x₀ : ℝ, f 1 1 x₀ = -5/4 :=
sorry

-- Theorem for part 2
theorem b_value_when_a_2 :
  ∀ b : ℝ, b ≥ 2 → (∀ x : ℝ, x ∈ Set.Icc 1 b → f 2 b x ∈ Set.Icc 1 b) → b = 2 :=
sorry

-- Theorem for part 3
theorem range_of_b_over_a :
  ∀ a b : ℝ, a > 0 →
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f a b x₁ = 1 ∧ f a b x₂ = 1) →
  1 < b / a ∧ b / a < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_b_1_b_value_when_a_2_range_of_b_over_a_l1007_100784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1007_100754

open Real

/-- Definition of a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < A ∧ 0 < B ∧ 0 < C
  h_sum_angles : A + B + C = π

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : t.a * sin t.A + t.b * sin t.B = t.c * sin t.C + Real.sqrt 2 * t.a * sin t.B) :
  t.C = π / 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 π → Real.sqrt 3 * sin x - cos (π - x + π / 4) ≤ 2) := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1007_100754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_change_calculation_l1007_100714

def cappuccino_price : ℚ := 2
def iced_tea_price : ℚ := 3
def cafe_latte_price : ℚ := 3/2
def espresso_price : ℚ := 1
def mocha_price : ℚ := 5/2
def hot_chocolate_price : ℚ := 2

def cappuccino_count : ℕ := 4
def iced_tea_count : ℕ := 3
def cafe_latte_count : ℕ := 5
def espresso_count : ℕ := 3
def mocha_count : ℕ := 2
def hot_chocolate_count : ℕ := 2

def custom_tip : ℚ := 5
def bill_amount : ℚ := 60

theorem sandy_change_calculation :
  (let total_cost := cappuccino_price * cappuccino_count +
                    iced_tea_price * iced_tea_count +
                    cafe_latte_price * cafe_latte_count +
                    espresso_price * espresso_count +
                    mocha_price * mocha_count +
                    hot_chocolate_price * hot_chocolate_count
   let total_with_tip := total_cost + custom_tip
   let change := bill_amount - total_with_tip
   change) = 37/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_change_calculation_l1007_100714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_assignment_l1007_100701

-- Define the people and professions
inductive Person : Type
  | A | B | C

inductive Profession : Type
  | Worker | Farmer | Intellectual

-- Define a function to assign professions to people
variable (assignment : Person → Profession)

-- Define the age relation
variable (older_than : Person → Person → Prop)

-- State the conditions
axiom one_of_each : 
  ∃! (w f i : Person), assignment w = Profession.Worker ∧ 
                       assignment f = Profession.Farmer ∧ 
                       assignment i = Profession.Intellectual

axiom c_older_than_intellectual : 
  ∀ i, assignment i = Profession.Intellectual → older_than Person.C i

axiom a_different_from_farmer : 
  ∀ f, assignment f = Profession.Farmer → ¬(older_than Person.A f ∧ older_than f Person.A)

axiom farmer_younger_than_b : 
  ∀ f, assignment f = Profession.Farmer → older_than Person.B f

-- State the theorem
theorem correct_assignment : 
  assignment Person.A = Profession.Intellectual ∧ 
  assignment Person.B = Profession.Worker ∧ 
  assignment Person.C = Profession.Farmer :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_assignment_l1007_100701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_mean_unchanged_l1007_100712

def original_set : Finset ℚ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset ℚ) : ℚ := (s.sum id) / s.card

def variance (s : Finset ℚ) : ℚ :=
  (s.sum (λ x => x^2) / s.card) - (mean s)^2

def new_set1 : Finset ℚ := {-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5}
def new_set2 : Finset ℚ := {-5, -4, -3, -2, -1, -1, 0, 1, 2, 3, 5, 5}

theorem variance_mean_unchanged :
  (mean original_set = mean new_set1) ∧
  (variance original_set = variance new_set1) ∧
  (mean original_set = mean new_set2) ∧
  (variance original_set = variance new_set2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_mean_unchanged_l1007_100712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_g_l1007_100793

/-- Given a function f(x) = a*sin(x) + cos(x) that is symmetric about x = π/6,
    prove that g(x) = sin(x) + a*cos(x) is symmetric about x = π/3 -/
theorem symmetry_of_g (a : ℝ) :
  (∀ x : ℝ, a * Real.sin x + Real.cos x = a * Real.sin (π / 3 - x) + Real.cos (π / 3 - x)) →
  (∀ x : ℝ, Real.sin x + a * Real.cos x = Real.sin (2 * π / 3 - x) + a * Real.cos (2 * π / 3 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_g_l1007_100793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l1007_100752

-- Define the arrival times as Real numbers between 0 and 3
def ArrivalTime : Type := { t : ℝ // 0 ≤ t ∧ t ≤ 3 }

-- Define the condition for the meeting to take place
def meetingOccurs (x y z : ArrivalTime) : Prop :=
  ∃ (x' y' z' : ℝ), x.val = x' ∧ y.val = y' ∧ z.val = z' ∧
  abs (x' - y') ≤ 1.5 ∧ z' > max x' y'

-- State the theorem
theorem meeting_probability :
  ∃ (P : Set (ArrivalTime × ArrivalTime × ArrivalTime) → ℝ),
    P (Set.univ : Set (ArrivalTime × ArrivalTime × ArrivalTime)) = 1 ∧
    P { (x, y, z) | meetingOccurs x y z } = 625 / 2700 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l1007_100752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1007_100707

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h_geom : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.q^n) / (1 - seq.q)

theorem geometric_sequence_problem (seq : GeometricSequence) 
    (h1 : sum_n seq 3 = seq.a 2 + 10 * seq.a 1)
    (h2 : seq.a 5 = 9) :
  seq.a 1 = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1007_100707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_order_correct_l1007_100709

-- Define the set of coins
inductive Coin : Type
  | A | B | C | D | E
  deriving BEq, Repr

-- Define the relation "is above"
def IsAbove : Coin → Coin → Prop := sorry

-- Define the properties of the coin arrangement
axiom c_top : ∀ x, x ≠ Coin.C → IsAbove Coin.C x
axiom e_below_c : IsAbove Coin.C Coin.E
axiom e_above_a_d : IsAbove Coin.E Coin.A ∧ IsAbove Coin.E Coin.D
axiom a_above_b : IsAbove Coin.A Coin.B
axiom d_above_b : IsAbove Coin.D Coin.B
axiom b_bottom : ∀ x, x ≠ Coin.B → IsAbove x Coin.B

-- Define a correct order of coins
def CorrectOrder : List Coin := [Coin.C, Coin.E, Coin.D, Coin.A, Coin.B]

-- Theorem to prove
theorem coin_order_correct : 
  ∀ (x y : Coin), x ≠ y → 
  (CorrectOrder.indexOf x < CorrectOrder.indexOf y ↔ IsAbove x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_order_correct_l1007_100709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l1007_100737

/-- A parabola with equation y² = 2px (p > 0) -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

/-- The focus of the parabola -/
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: If a line passing through the focus of the parabola y² = 2px (p > 0)
    intersects the parabola at points P and Q such that x₁ + x₂ = 6 and PQ = 10,
    then p = 4. -/
theorem parabola_theorem (para : Parabola) (P Q : ParabolaPoint para)
  (h_sum : P.x + Q.x = 6)
  (h_dist : distance (P.x, P.y) (Q.x, Q.y) = 10) :
  para.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l1007_100737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1007_100713

theorem equation_solution (a b : ℝ) (h : a ≠ 0) : 
  ∃ y : ℝ, y^2 + (2 * b)^2 = (3 * a - y)^2 ∧ y = (9 * a^2 - 4 * b^2) / (6 * a) := by
  let y := (9 * a^2 - 4 * b^2) / (6 * a)
  use y
  constructor
  · ring
    field_simp [h]
    ring
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1007_100713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_sum_exists_l1007_100797

theorem zero_sum_exists (n : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ k, k ≤ n → a k ≤ k)
  (h2 : Even (Finset.sum (Finset.range n) a)) :
  ∃ s : ℕ → Int, (∀ k, k ≤ n → s k = 1 ∨ s k = -1) ∧ 
    (Finset.sum (Finset.range n) (λ k ↦ s k * a k) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_sum_exists_l1007_100797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1007_100742

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (Real.sqrt x)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1007_100742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_unit_radian_chord_arc_length_equals_inverse_sin_half_l1007_100715

/-- 
Given a circle where the chord length corresponding to a central angle of 1 radian is 2,
prove that the arc length corresponding to this central angle is 1 / sin(1/2).
-/
theorem arc_length_for_unit_radian_chord (r : ℝ) : 
  r * Real.sin (1/2 : ℝ) = 1 → r = 1 / Real.sin (1/2 : ℝ) := by sorry

/-- 
The arc length corresponding to a central angle of 1 radian 
in a circle where the chord length for this angle is 2.
-/
def arc_length (r : ℝ) : r * Real.sin (1/2 : ℝ) = 1 → ℝ :=
  λ h => 1 * r

theorem arc_length_equals_inverse_sin_half (r : ℝ) (h : r * Real.sin (1/2 : ℝ) = 1) :
  arc_length r h = 1 / Real.sin (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_unit_radian_chord_arc_length_equals_inverse_sin_half_l1007_100715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_less_than_three_l1007_100799

/-- The function f(x) = -1/3 * x^2 + 2x -/
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^2 + 2*x

/-- The sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => f (a n)

/-- Theorem: For all n ≥ 1, a_n < 3 -/
theorem a_n_less_than_three : ∀ n : ℕ, n ≥ 1 → a n < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_less_than_three_l1007_100799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_in_cube_volume_ratio_l1007_100731

/-- The ratio of the volume of a cylinder inscribed in a cube to the volume of the cube,
    given that the height and diameter of the cylinder are equal to the side length of the cube. -/
theorem cylinder_in_cube_volume_ratio (s : ℝ) (h : s > 0) :
  let r : ℝ := s / 2
  let cylinder_volume : ℝ := π * r^2 * s
  let cube_volume : ℝ := s^3
  cylinder_volume / cube_volume = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_in_cube_volume_ratio_l1007_100731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1007_100772

/-- The ellipse C₁ -/
def C₁ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The circle C₃ -/
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The ellipse C₂ -/
def C₂ (a b x y : ℝ) : Prop :=
  x^2 / (((a^2 - b^2) / (a^2 + b^2))^2 * a^2) + y^2 / (((a^2 - b^2) / (a^2 + b^2))^2 * b^2) = 1

/-- Line through two points -/
def line_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {R | ∃ t : ℝ, R = (1 - t) • P + t • Q}

/-- The main theorem -/
theorem ellipse_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (A B C D E F : ℝ × ℝ) (hA : C₁ a b A.1 A.2)
  (hB : B = (-A.1, A.2)) (hC : C = (-A.1, -A.2)) (hD : D = (A.1, -A.2))
  (hE : C₁ a b E.1 E.2) (hAE_perp_AC : (E.1 - A.1) * (C.1 - A.1) + (E.2 - A.2) * (C.2 - A.2) = 0)
  (hF : F ∈ (line_through B D) ∩ (line_through C E))
  (h_eq : (a^2 + b^2)^3 = a^2 * b^2 * (a^2 - b^2)^2)
  (A₁ B₁ : ℝ × ℝ) (hA₁B₁_on_C₂ : C₂ a b A₁.1 A₁.2 ∧ C₂ a b B₁.1 B₁.2)
  (hA₁B₁_tangent_C₃ : ∃ (t : ℝ), C₃ ((1-t)*A₁.1 + t*B₁.1) ((1-t)*A₁.2 + t*B₁.2) ∧
    ∀ (s : ℝ), s ≠ t → ¬C₃ ((1-s)*A₁.1 + s*B₁.1) ((1-s)*A₁.2 + s*B₁.2)) :
  (C₂ a b F.1 F.2) ∧ (A₁.1 * B₁.1 + A₁.2 * B₁.2 = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1007_100772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EG_l1007_100796

/-- Square with side length 5 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 5 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

/-- Point E -/
def E : ℝ × ℝ := (0, 5)

/-- Point G -/
def G : ℝ × ℝ := (5, 0)

/-- Point H -/
def H : ℝ × ℝ := (0, 0)

/-- Midpoint N of GH -/
def N : ℝ × ℝ := (2.5, 0)

/-- Circle centered at N with radius 2.5 -/
def CircleN : Set (ℝ × ℝ) :=
  {p | (p.1 - 2.5)^2 + p.2^2 = 2.5^2}

/-- Circle centered at E with radius 5 -/
def CircleE : Set (ℝ × ℝ) :=
  {p | p.1^2 + (p.2 - 5)^2 = 5^2}

/-- Point Q is an intersection of CircleN and CircleE, different from H -/
def Q : ℝ × ℝ := (4, 2)

/-- Distance from a point to a line -/
noncomputable def distToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  Real.sqrt (p.2^2)

/-- Line EG -/
def LineEG : Set (ℝ × ℝ) :=
  {p | p.2 = p.1}

theorem distance_Q_to_EG :
  distToLine Q LineEG = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EG_l1007_100796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_inequality_l1007_100711

/-- A point with rational coordinates in the plane -/
structure RatPoint where
  x : ℚ
  y : ℚ

/-- A circle in the plane -/
structure Circle where
  center : ℚ × ℚ
  radius : ℚ

/-- Distance squared between two points -/
def distanceSquared (p q : RatPoint) : ℚ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Check if a point lies on a circle -/
def onCircle (p : RatPoint) (c : Circle) : Prop :=
  distanceSquared p ⟨c.center.1, c.center.2⟩ = c.radius^2

/-- Main theorem -/
theorem triangle_circle_inequality (A B C : RatPoint) (K : Circle)
    (hA : onCircle A K) (hB : onCircle B K) (hC : onCircle C K) :
    distanceSquared A B * distanceSquared B C * distanceSquared C A ≥ 8 * K.radius^3 ∧
    (K.center = (0, 0) → distanceSquared A B * distanceSquared B C * distanceSquared C A ≥ 64 * K.radius^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_inequality_l1007_100711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_3_vs_base_8_digits_l1007_100751

/-- The number of digits required to represent a positive integer n in base b -/
noncomputable def numDigits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.floor (Real.log (n : ℝ) / Real.log (b : ℝ)) + 1

theorem base_3_vs_base_8_digits :
  numDigits 5000 3 - numDigits 5000 8 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_3_vs_base_8_digits_l1007_100751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1007_100725

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then Real.exp x + a
  else if x < 0 then -(Real.exp (-x) + a)
  else 0

-- State the theorem
theorem min_value_of_a :
  ∀ a : ℝ,
  (∀ x : ℝ, f a x = -f a (-x)) → -- f is odd
  (∀ x y : ℝ, x < y → f a x < f a y) → -- f is strictly increasing
  a ≥ -1 ∧ ∀ b : ℝ, b < -1 → ¬(∀ x y : ℝ, x < y → f b x < f b y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1007_100725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_19_eq_1_1443_l1007_100798

def f : ℕ → ℚ
  | 0 => 0  -- Add a case for 0
  | 1 => 1/3
  | (n+2) => (2*n + 1)/(2*n + 5) * f (n+1)

theorem f_19_eq_1_1443 : f 19 = 1/1443 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_19_eq_1_1443_l1007_100798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_water_usage_l1007_100727

/-- Water bill calculation function -/
noncomputable def waterBill (x : ℝ) : ℝ :=
  if x ≤ 7 then 3 * x
  else if x ≤ 11 then 6 * x - 21
  else 9 * x - 54

/-- Theorem: March water usage is 10 tons -/
theorem march_water_usage 
  (jan_usage : ℝ) 
  (feb_usage : ℝ) 
  (total_bill : ℝ) 
  (h1 : jan_usage = 9) 
  (h2 : feb_usage = 12) 
  (h3 : total_bill = 126) 
  (h4 : ∃ (march_usage : ℝ), 
    0 ≤ march_usage ∧ 
    march_usage ≤ 15 ∧ 
    waterBill jan_usage + waterBill feb_usage + waterBill march_usage = total_bill) :
  ∃ (march_usage : ℝ), march_usage = 10 ∧ 
    waterBill jan_usage + waterBill feb_usage + waterBill march_usage = total_bill :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_water_usage_l1007_100727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangles_different_lengths_l1007_100787

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles_different_lengths :
  ∃ (a b c d : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ 
    (a + b > c ∧ a + c > b ∧ b + c > a) ∧
    (a + b > d ∧ a + d > b ∧ b + d > a) ∧
    (a + c > d ∧ a + d > c ∧ c + d > a) ∧
    (b + c > d ∧ b + d > c ∧ c + d > b) ∧
    (area_triangle a b c = area_triangle a b d) ∧
    (area_triangle a b c = area_triangle a c d) ∧
    (area_triangle a b c = area_triangle b c d) ∧
    ¬(a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangles_different_lengths_l1007_100787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_iff_m_eq_neg_one_l1007_100774

/-- A power function of the form y = (m^2 - 5m - 5)x^(2m+1) -/
noncomputable def powerFunction (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5*m - 5) * (x^(2*m + 1))

/-- The function is decreasing on (0,+∞) -/
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f y < f x

theorem power_function_decreasing_iff_m_eq_neg_one :
  ∀ m : ℝ, isDecreasing (powerFunction m) ↔ m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_iff_m_eq_neg_one_l1007_100774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1007_100717

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := x / (x^k + 1)

-- State the theorem
theorem range_of_g (k : ℝ) (h_k : k > 0) :
  Set.range (fun x => g k x) = Set.Icc 0 (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1007_100717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l1007_100783

/-- A circle with center (3,k) where k > 8 is tangent to y=x, y=-x, and y=8. Its radius is 73/6. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 8) :
  let center := (3, k)
  let is_tangent_to (line : ℝ → ℝ) := ∃ p : ℝ × ℝ, p.2 = line p.1 ∧ Real.sqrt ((p.1 - 3)^2 + (p.2 - k)^2) = k - 8
  is_tangent_to (λ x ↦ x) ∧ 
  is_tangent_to (λ x ↦ -x) ∧ 
  is_tangent_to (λ _ ↦ 8) →
  k - 8 = 73 / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l1007_100783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1007_100720

/-- The function f(x) = 2sin(ωx + φ) + 1 -/
noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 1

/-- The theorem stating the range of φ given the conditions -/
theorem phi_range (ω φ : ℝ) : 
  ω > 0 → 
  |φ| ≤ π / 2 → 
  (∀ x : ℝ, f ω φ (x + π / ω) = f ω φ x) → 
  (∀ x : ℝ, x ∈ Set.Ioo (-π/12) (π/3) → f ω φ x > 1) → 
  φ ∈ Set.Icc (π/6) (π/3) := by
  sorry

#check phi_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1007_100720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l1007_100750

structure Plane where
  -- Placeholder for plane properties

structure Line where
  -- Placeholder for line properties

def parallel_to_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of a line parallel to a plane

def within_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of a line within a plane

def parallel_lines (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallel lines

def skew_lines (l1 l2 : Line) : Prop :=
  sorry -- Definition of skew lines

theorem line_plane_relationship (a b : Line) (α : Plane) :
  parallel_to_plane a α → within_plane b α →
  (parallel_lines a b ∨ skew_lines a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l1007_100750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_angle_in_set_l1007_100770

open Real

-- Define the set of angles
noncomputable def angle_set : Set ℝ := {15 * π / 180, 45 * π / 180, 135 * π / 180, 165 * π / 180, 255 * π / 180, 285 * π / 180}

-- Define the system of equations
def system (x y α : ℝ) : Prop :=
  y^3 ≥ 3 * x^2 * y ∧ (x - cos α)^2 + (y - sin α)^2 = (2 - sqrt 3) / 4

-- Define the theorem
theorem unique_solution_iff_angle_in_set :
  ∀ α ∈ Set.Icc 0 (2 * π),
    (∃! p : ℝ × ℝ, system p.1 p.2 α) ↔ α ∈ angle_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_angle_in_set_l1007_100770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_existence_uniqueness_l1007_100747

-- Define the basic types
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define lines and points
variable (l₁ l₂ : Set V)
variable (A : V)

-- Define the ratio
variable (m n : ℝ)

-- Define the property of being a line
def is_line (l : Set V) : Prop := sorry

-- Define the property of a point being on a line
def on_line (P : V) (l : Set V) : Prop := sorry

-- Define the property of two lines being parallel
def parallel (l₁ l₂ : Set V) : Prop := sorry

-- Define the distance between two points
noncomputable def distance (P Q : V) : ℝ := ‖P - Q‖

-- Theorem statement
theorem intersection_line_existence_uniqueness :
  (∃! l : Set V, is_line l ∧ A ∈ l ∧
    (∃ B C : V, B ∈ l₁ ∧ C ∈ l₂ ∧ B ∈ l ∧ C ∈ l ∧
      distance A B * n = distance A C * m)) ↔
  ¬ parallel l₁ l₂ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_existence_uniqueness_l1007_100747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_packs_split_l1007_100753

/-- Given information about sticker packs and costs, prove the number of packs split. -/
theorem sticker_packs_split (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) (james_payment : ℚ) :
  stickers_per_pack = 30 →
  cost_per_sticker = 1/10 →
  james_payment = 6 →
  (james_payment * 2) / (cost_per_sticker * stickers_per_pack) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_packs_split_l1007_100753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1007_100732

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum :
  let a : ℝ := -2
  let r : ℝ := 3
  let n : ℕ := 8
  geometric_sum a r n = -6560 := by
  -- Unfold the definition of geometric_sum
  unfold geometric_sum
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1007_100732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1007_100726

-- Define the function f(x) = 2x / (x - 1)
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x - 1)

-- Statement: The domain of f is all real numbers except 1
theorem f_domain : {x : ℝ | x ≠ 1} = {x : ℝ | IsRegular (f x)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1007_100726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_values_l1007_100706

theorem function_g_values (g : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, g (x + 3) - g x = 6 * x + 9)
  (h2 : ∀ x : ℤ, g (x^2 - 4) = (g x - x)^2 + x^2 - 8) :
  g 0 = 0 ∧ g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_values_l1007_100706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_uniqueness_l1007_100740

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define a projective transformation
structure ProjectiveTransformation where
  matrix : Matrix (Fin 3) (Fin 3) ℝ
  det_nonzero : matrix.det ≠ 0

-- Define the concept of "inside a circle"
def inside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define the concept of "mapping to a circle"
def maps_to_circle (t : ProjectiveTransformation) (s : Circle) : Prop :=
  sorry

-- Define the concept of "mapping to center"
def maps_to_center (t : ProjectiveTransformation) (p : Point) (c : Circle) : Prop :=
  sorry

-- Define the concept of "mapping to infinity"
def maps_to_infinity (t : ProjectiveTransformation) (l : Line) : Prop :=
  sorry

-- State the theorem
theorem polar_line_uniqueness (s : Circle) (o : Point) (h : inside o s) :
  ∃! l : Line,
    ∀ t : ProjectiveTransformation,
      maps_to_circle t s → maps_to_center t o s →
        maps_to_infinity t l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_uniqueness_l1007_100740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_measurement_greater_than_actual_l1007_100739

/-- Represents a balance scale with unequal arms -/
structure UnequalArmScale where
  leftArmLength : ℝ
  rightArmLength : ℝ
  leftArmLength_pos : leftArmLength > 0
  rightArmLength_pos : rightArmLength > 0
  unequalArms : leftArmLength ≠ rightArmLength

/-- Measures the apparent mass of an object on the left pan -/
noncomputable def measureLeft (scale : UnequalArmScale) (actualMass : ℝ) : ℝ :=
  actualMass * (scale.rightArmLength / scale.leftArmLength)

/-- Measures the apparent mass of an object on the right pan -/
noncomputable def measureRight (scale : UnequalArmScale) (actualMass : ℝ) : ℝ :=
  actualMass * (scale.leftArmLength / scale.rightArmLength)

/-- The average of two measurements is always greater than the actual mass -/
theorem average_measurement_greater_than_actual
  (scale : UnequalArmScale) (actualMass : ℝ) (actualMass_pos : actualMass > 0) :
  (measureLeft scale actualMass + measureRight scale actualMass) / 2 > actualMass := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_measurement_greater_than_actual_l1007_100739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_M_with_inclination_angle_l1007_100767

-- Define the point M
def M : ℝ × ℝ := (2, -3)

-- Define the inclination angle in radians
noncomputable def α : ℝ := Real.pi / 4  -- 45° in radians

-- Define the slope of the line
noncomputable def m : ℝ := Real.tan α

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := x - y - 5 = 0

-- Theorem statement
theorem line_passes_through_M_with_inclination_angle :
  line_equation M.1 M.2 ∧ m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_M_with_inclination_angle_l1007_100767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l1007_100792

open Real Set

-- Define the ellipse
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the right focus
noncomputable def rightFocus (a b : ℝ) (h : a > b ∧ b > 0) : ℝ × ℝ :=
  (Real.sqrt (a^2 - b^2), 0)

-- Define the right directrix
noncomputable def rightDirectrix (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = a^2 / Real.sqrt (a^2 - b^2)}

-- Define point P
noncomputable def pointP (a b : ℝ) (h : a > b ∧ b > 0) : ℝ × ℝ :=
  (a^2 / Real.sqrt (a^2 - b^2), 0)

-- Define a line through a point
def lineThrough (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ t : ℝ, q = (p.1 + t, p.2 + t)}

-- Define reflection with respect to x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Theorem statement
theorem points_collinear
  (a b : ℝ) (h : a > b ∧ b > 0)
  (A B : ℝ × ℝ)
  (hA : A ∈ ellipse a b h)
  (hB : B ∈ ellipse a b h)
  (hAB : A ∈ lineThrough (rightFocus a b h) ∧ B ∈ lineThrough (rightFocus a b h))
  (Q : ℝ × ℝ)
  (hQ : Q = reflectX A) :
  ∃ (m : ℝ), (pointP a b h).2 - Q.2 = m * ((pointP a b h).1 - Q.1) ∧
             B.2 - Q.2 = m * (B.1 - Q.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l1007_100792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1007_100781

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((floor x) * y) = (floor (f y)) * (f x)

theorem function_characterization (f : ℝ → ℝ) (h : satisfies_property f) :
  (∀ x : ℝ, f x = 0) ∨ 
  (∃ c : ℝ, (∀ x : ℝ, f x = c) ∧ floor c = 1) :=
by
  sorry

#check function_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1007_100781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l1007_100743

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem min_shift_for_even_function :
  ∃ m : ℝ, m > 0 ∧ is_even (g m) ∧ ∀ m' : ℝ, m' > 0 ∧ is_even (g m') → m ≤ m' ∧ m = 5 * Real.pi / 6 := by
  sorry

#check min_shift_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l1007_100743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_last_four_l1007_100791

/-- Calculates the average score for the last 4 matches in a cricket series -/
theorem cricket_average_last_four (total_matches first_matches : ℕ) 
  (total_average first_average : ℚ) : 
  total_matches = 10 →
  first_matches = 6 →
  total_average = 389/10 →
  first_average = 41 →
  let last_matches := total_matches - first_matches
  let last_average := (total_average * total_matches - first_average * first_matches) / last_matches
  last_average = 143/4 := by
    intro h1 h2 h3 h4
    sorry

#check cricket_average_last_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_last_four_l1007_100791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_persimmon_count_l1007_100759

theorem persimmon_count (n k m : ℕ) 
  (h1 : n = 6) -- Total number of baskets is 6
  (h2 : k = 5) -- 5 baskets have 5 persimmons each
  (h3 : m = 8) -- The remaining basket has 8 persimmons
  : k * 5 + m = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_persimmon_count_l1007_100759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_count_l1007_100771

-- Define the point A
def A : ℝ × ℝ := (1, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function to check if a triangle is isosceles
def is_isosceles (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let d12 := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let d23 := Real.sqrt ((p2.1 - p3.1)^2 + (p2.2 - p3.2)^2)
  let d31 := Real.sqrt ((p3.1 - p1.1)^2 + (p3.2 - p1.2)^2)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

-- Define the set of points P on the x-axis that form an isosceles triangle with A and O
def isosceles_points : Set (ℝ × ℝ) :=
  {P | P.2 = 0 ∧ is_isosceles A O P}

-- Theorem statement
theorem isosceles_triangle_count :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 4 ∧ ∀ P, P ∈ S ↔ P ∈ isosceles_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_count_l1007_100771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_a_circumference_l1007_100790

/-- The circumference of a circular tank given its volume and height. -/
noncomputable def tank_circumference (volume : ℝ) (height : ℝ) : ℝ :=
  Real.sqrt ((4 * Real.pi * volume) / height)

/-- Theorem: Given the conditions of the problem, the circumference of tank A is 8 meters. -/
theorem tank_a_circumference (height_a height_b circumference_b : ℝ)
    (volume_ratio : ℝ) :
    height_a = 10 →
    height_b = 8 →
    circumference_b = 10 →
    volume_ratio = 0.8000000000000001 →
    tank_circumference (volume_ratio * (Real.pi * (circumference_b / (2 * Real.pi))^2 * height_b)) height_a = 8 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_a_circumference_l1007_100790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_arithmetic_progression_probability_l1007_100745

def dice_outcomes : Nat := 1296  -- 6^4

def valid_sequences : List (List Nat) := [[1,2,3,4], [2,3,4,5], [3,4,5,6]]

def arrangements_per_sequence : Nat := 24  -- 4!

theorem dice_arithmetic_progression_probability : 
  (valid_sequences.length * arrangements_per_sequence : Rat) / dice_outcomes = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_arithmetic_progression_probability_l1007_100745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_cubic_curve_l1007_100734

/-- The equation of the tangent line to y = x³ - 2x at (1, -1) is x - y - 2 = 0 -/
theorem tangent_line_cubic_curve : 
  let f (x : ℝ) := x^3 - 2*x
  let point : ℝ × ℝ := (1, -1)
  let tangent_line (x y : ℝ) := x - y - 2
  (∀ x y, tangent_line x y = 0 ↔ 
    y - f point.1 = (deriv f point.1) * (x - point.1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_cubic_curve_l1007_100734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1007_100733

def p (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ - a = 0 ∧ x₂^2 + 2*x₂ - a = 0

def q (a : ℝ) : Prop :=
  ∀ m : ℝ, -2 ≤ m ∧ m ≤ 4 → a^2 - a ≥ 4 - m

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (-1 < a ∧ a < 3) ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1007_100733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_five_l1007_100755

noncomputable def g (x : ℝ) : ℝ := 1 / (1 - x)

theorem g_composition_five : g (g (g (g (g (g (g 5)))))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_five_l1007_100755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l1007_100736

def A : Set ℤ := {x | (2 / (x + 1 : ℚ)).isInt}

theorem set_A_elements : A = {-3, -2, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l1007_100736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1007_100721

-- Define the function f(x) = x - ln x
noncomputable def f (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem monotonic_increase_interval :
  (StrictMonoOn f (Set.Ioi 1)) ∧ (∀ y ∈ Set.Ioo 0 1, ¬ StrictMonoOn f (Set.Ioo y 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1007_100721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1007_100729

-- Define the ellipse
def is_ellipse (p : ℝ × ℝ) : Prop := 4 * p.1^2 + 5 * p.2^2 = 1

-- Define the foci
def is_focus (F F' : ℝ × ℝ) (is_ellipse : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (c : ℝ), F.1 = -c ∧ F'.1 = c ∧ F.2 = 0 ∧ F'.2 = 0 ∧
  ∀ (p : ℝ × ℝ), is_ellipse p → (p.1 - F.1)^2 + p.2^2 + (p.1 - F'.1)^2 + p.2^2 = 1

-- Define points on the ellipse
def on_ellipse (P : ℝ × ℝ) (is_ellipse : (ℝ × ℝ) → Prop) : Prop :=
  is_ellipse P

-- Define the line through F' intersecting the ellipse at M and N
def line_intersects (F' M N : ℝ × ℝ) (is_ellipse : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (t₁ t₂ : ℝ), M = F' + t₁ • (M - F') ∧ N = F' + t₂ • (N - F') ∧
  on_ellipse M is_ellipse ∧ on_ellipse N is_ellipse

-- Main theorem
theorem ellipse_triangle_perimeter
  (F F' M N : ℝ × ℝ)
  (h_ellipse : ∀ (p : ℝ × ℝ), is_ellipse p ↔ 4 * p.1^2 + 5 * p.2^2 = 1)
  (h_foci : is_focus F F' is_ellipse)
  (h_intersect : line_intersects F' M N is_ellipse) :
  dist F M + dist M N + dist N F = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1007_100729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_with_large_prime_divisor_l1007_100778

theorem infinite_n_with_large_prime_divisor :
  ∃ f : ℕ → ℕ, StrictMono f ∧
    ∀ i : ℕ, ∃ p : ℕ,
      Nat.Prime p ∧
      p ∣ (f i)^2 + 1 ∧
      (p : ℝ) > 2 * (f i) + Real.sqrt (5 * (f i) + 2011) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_with_large_prime_divisor_l1007_100778
