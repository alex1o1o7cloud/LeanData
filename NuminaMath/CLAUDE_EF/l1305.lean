import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_ice_cream_depth_l1305_130577

/-- The volume of a sphere with radius r is (4/3) * π * r^3 -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h is π * r^2 * h -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The height of a cylinder with radius 9 inches, having the same volume as a sphere with radius 3 inches, is 4/9 inches -/
theorem melted_ice_cream_depth :
  let sphere_radius : ℝ := 3
  let cylinder_radius : ℝ := 9
  let cylinder_height : ℝ := 4/9
  sphere_volume sphere_radius = cylinder_volume cylinder_radius cylinder_height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_ice_cream_depth_l1305_130577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1305_130529

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 9 = 2 * (a 5)^2 →
  a 2 = 1 →
  a 1 = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1305_130529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1305_130571

-- Define the function f(x) = x + sin x
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1305_130571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_bound_l1305_130564

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool  -- We assume this property is true for the polygon

/-- Checks if a point is inside the unit square -/
noncomputable def inside_unit_square (p : Real × Real) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1

/-- Calculates the sum of squares of the sides of a polygon -/
noncomputable def sum_of_squares_of_sides (p : ConvexPolygon) : Real :=
  sorry  -- Implementation details omitted

/-- Theorem: The sum of squares of sides of a convex polygon inside a unit square is at most 4 -/
theorem sum_of_squares_bound (p : ConvexPolygon) 
    (h : ∀ v ∈ p.vertices, inside_unit_square v) : 
  sum_of_squares_of_sides p ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_bound_l1305_130564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_coordinates_l1305_130518

/-- Given two points A and B symmetric about a line, prove their coordinates. -/
theorem symmetric_points_coordinates :
  ∀ (a b : ℝ),
  let A := (a + 2, b + 2)
  let B := (b - a, -b)
  let line (p : ℝ × ℝ) := 4 * p.1 + 3 * p.2 = 11
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let perpendicular := (A.2 - B.2) * 4 = -(A.1 - B.1) * 3
  (line midpoint ∧ perpendicular) → a = 4 ∧ b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_coordinates_l1305_130518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1305_130507

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 6) / (2*x - 4)

-- Define the domain
def domain (x : ℝ) : Prop := -2 < x ∧ x < 5

-- State the theorem
theorem f_min_max :
  ∃ (x_min x_max : ℝ), domain x_min ∧ domain x_max ∧
  (∀ x, domain x → f x_min ≤ f x ∧ f x ≤ f x_max) ∧
  f x_min = -3/2 ∧ f x_max = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1305_130507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_function_is_sine_l1305_130524

open Real

-- Define the transformation functions
def double_x (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (2 * x)
def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x ↦ f (x + a)

-- State the theorem
theorem original_function_is_sine (f : ℝ → ℝ) :
  (double_x (shift_left f (π/3))) = (λ x ↦ sin (x - π/4)) →
  f = (λ x ↦ sin (x/2 + π/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_function_is_sine_l1305_130524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_theorem_l1305_130588

open Complex

-- Define the domain (upper half of the unit circle)
def upperHalfUnitCircle (z : ℂ) : Prop :=
  abs z < 1 ∧ z.im > 0

-- Define the target (upper half-plane)
def upperHalfPlane (w : ℂ) : Prop :=
  w.im > 0

-- Define the mapping function
noncomputable def f (z : ℂ) : ℂ :=
  ((1 + z) / (1 - z)) ^ 2

-- Theorem statement
theorem mapping_theorem :
  ∀ z : ℂ, upperHalfUnitCircle z → upperHalfPlane (f z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_theorem_l1305_130588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_ln_l1305_130502

theorem negation_of_forall_ln :
  (¬ ∀ x > 3, Real.log x > 1) ↔ (∃ x > 3, Real.log x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_ln_l1305_130502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_specific_pyramid_l1305_130580

/-- The surface area of a regular pyramid with given base edge length and lateral edge length -/
noncomputable def surface_area_regular_pyramid (base_edge : ℝ) (lateral_edge : ℝ) : ℝ :=
  let slant_height := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  4 * (1 / 2 * base_edge * slant_height) + base_edge ^ 2

/-- Theorem: The surface area of a regular pyramid with base edge length 4 and lateral edge length √13 is 40 -/
theorem surface_area_specific_pyramid :
  surface_area_regular_pyramid 4 (Real.sqrt 13) = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_specific_pyramid_l1305_130580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_polar_representation_l1305_130594

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * t, Real.sqrt 3 - t)

noncomputable def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = 2

theorem curve_C_polar_representation :
  ∀ (x y : ℝ), ∃ (t : ℝ), curve_C t = (x, y) →
  ∃ (ρ θ : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ polar_equation ρ θ := by
  sorry

#check curve_C_polar_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_polar_representation_l1305_130594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_neg_twenty_pi_thirds_l1305_130508

theorem sin_neg_twenty_pi_thirds : Real.sin (-20 * π / 3) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_neg_twenty_pi_thirds_l1305_130508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_l1305_130583

-- Define the points
variable (A B C D E K O : EuclideanSpace ℝ (Fin 2))

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (B - A) = (D - C) ∧ (C - B) = (D - A)

-- Define that E is on the extension of AB
def on_extension_AB (A B E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, t > 1 ∧ E = A + t • (B - A)

-- Define that K is on the extension of AD
def on_extension_AD (A D K : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, t > 1 ∧ K = A + t • (D - A)

-- Define that BK and DE intersect at O
def intersect_at (B K D E O : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t s : ℝ, 0 < t ∧ 0 < s ∧ O = B + t • (K - B) ∧ O = D + s • (E - D)

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (P Q R S : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- Theorem statement
theorem equal_areas
  (h1 : is_parallelogram A B C D)
  (h2 : on_extension_AB A B E)
  (h3 : on_extension_AD A D K)
  (h4 : intersect_at B K D E O) :
  area_quadrilateral A B O D = area_quadrilateral E C K O :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_l1305_130583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_even_marbles_distribution_l1305_130591

/-- Represents the initial distribution of marbles among friends -/
def initial_marbles : List Nat := [14, 20, 19, 7, 23, 11]

/-- The number of friends -/
def num_friends : Nat := 6

/-- Theorem stating the maximum even number of marbles each friend can have -/
theorem max_even_marbles_distribution :
  let total_marbles := initial_marbles.sum
  let max_marbles := total_marbles / num_friends
  let max_even_marbles := if max_marbles % 2 = 0 then max_marbles else max_marbles - 1
  max_even_marbles = 14 := by
  sorry

#eval initial_marbles.sum
#eval initial_marbles.sum / num_friends

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_even_marbles_distribution_l1305_130591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1305_130562

noncomputable def f (x : ℝ) : ℝ := (3 * x - 5) / (x + 4)

theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1305_130562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_point_relation_l1305_130525

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the points A, B, C
noncomputable def A : ℝ × ℝ := (-3, f (-3))
noncomputable def B : ℝ × ℝ := (1/2, f (1/2))
noncomputable def C : ℝ × ℝ := (2, f 2)

-- Extract y-coordinates
noncomputable def y₁ : ℝ := A.2
noncomputable def y₂ : ℝ := B.2
noncomputable def y₃ : ℝ := C.2

-- State the theorem
theorem quadratic_point_relation : y₂ < y₁ ∧ y₁ < y₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_point_relation_l1305_130525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1305_130520

/-- The length of a train given its speed, time to cross a platform, and the platform's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (platform_length : ℝ) : 
  train_speed = 60 → crossing_time = 20 → platform_length = 213.36 → 
  ∃ (train_length : ℝ), abs (train_length - 120.04) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1305_130520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_equation_l1305_130527

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the intersection point A
def point_A : ℝ × ℝ := (2, 0)

-- Define the perpendicular lines l₁ and l₂ (implicitly defined by their intersection at A)
def perpendicular_lines (l₁ l₂ : ℝ → ℝ) : Prop :=
  ∃ (m₁ m₂ : ℝ), (∀ x, l₁ x = m₁ * (x - 2)) ∧ (∀ x, l₂ x = m₂ * (x - 2)) ∧ m₁ * m₂ = -1

-- Define the tangency conditions for circle M
def tangent_conditions (M : ℝ × ℝ) (m : ℝ) : Prop :=
  m > 0 ∧
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- Tangent to circle C
    circle_C x₁ y₁ ∧ (x₁ - M.1)^2 + (y₁ - M.2)^2 = m^2 ∧
    -- Tangent to line l₁
    (∃ l₁, perpendicular_lines l₁ (λ x ↦ 0) ∧ l₁ x₂ = y₂ ∧ (x₂ - M.1)^2 + (y₂ - M.2)^2 = m^2) ∧
    -- Tangent to line l₂
    (∃ l₂, perpendicular_lines (λ x ↦ 0) l₂ ∧ l₂ x₃ = y₃ ∧ (x₃ - M.1)^2 + (y₃ - M.2)^2 = m^2)

-- Theorem statement
theorem circle_M_equation (M : ℝ × ℝ) (m : ℝ) :
  tangent_conditions M m →
  ∀ (x y : ℝ), (x - 1)^2 + (y - Real.sqrt 7)^2 = 4 ↔ (x - M.1)^2 + (y - M.2)^2 = m^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_M_equation_l1305_130527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1305_130593

theorem equation_solution :
  ∃ t : ℝ, (2 * 2^t + Real.sqrt (4 * 4^t) = 12) ∧ (t = Real.log 3 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1305_130593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_l1305_130556

noncomputable def g (x : ℝ) : ℝ := (3 * x + 7) / (x - 2)

theorem g_of_8 : g 8 = 31 / 6 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the numerator and denominator
  simp [mul_add, add_div]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_l1305_130556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l1305_130558

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

/-- Definition of the right branch of the hyperbola -/
def is_on_right_branch (x y : ℝ) : Prop := is_on_hyperbola x y ∧ x > 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem to prove -/
theorem hyperbola_min_value (x y xf1 yf1 xf2 yf2 : ℝ) :
  is_on_right_branch x y →
  (xf1, yf1) = (-3, 0) →  -- Left focus
  (xf2, yf2) = (3, 0) →   -- Right focus
  ∃ (m : ℝ), (distance x y xf1 yf1)^2 - (distance x y xf2 yf2) ≥ 23 * (distance x y xf2 yf2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l1305_130558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_through_points_and_plane_radius_calculation_correct_l1305_130579

/-- The radius of a sphere passing through two points and intersecting a plane -/
theorem sphere_radius_through_points_and_plane (AB : ℝ) (angle : ℝ) (ratio : ℝ) : 
  AB = 8 →
  angle = 30 * π / 180 →
  ratio = 1 / 3 →
  ∃ (R : ℝ), R = 2 * Real.sqrt 7 :=
by
  sorry

/-- Function to calculate the radius of the sphere -/
noncomputable def radius_of_sphere_through_points_and_plane (AB : ℝ) (angle : ℝ) (ratio : ℝ) : ℝ :=
  2 * Real.sqrt 7

/-- Theorem stating that our function calculates the correct radius -/
theorem radius_calculation_correct (AB : ℝ) (angle : ℝ) (ratio : ℝ) :
  AB = 8 →
  angle = 30 * π / 180 →
  ratio = 1 / 3 →
  radius_of_sphere_through_points_and_plane AB angle ratio = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_through_points_and_plane_radius_calculation_correct_l1305_130579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_sale_items_l1305_130569

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card > 0 →
  (∀ x y, x ∈ prices → y ∈ prices → x ≠ y → x ≠ radio_price → y ≠ radio_price → x ≠ y) →
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = 14 →
  (prices.filter (λ x => x < radio_price)).card = 21 →
  prices.card = 36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_sale_items_l1305_130569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1305_130595

/-- The function f(x) = 2x^3 - 6x^2 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem max_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f m x ≤ f m y) ∧
  (∃ x ∈ Set.Icc 1 3, f m x = 2 ∧ ∀ y ∈ Set.Icc 1 3, f m y ≥ 2) →
  ∃ z ∈ Set.Icc 1 3, ∀ w ∈ Set.Icc 1 3, f m w ≤ f m z ∧ f m z = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1305_130595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_gender_selections_twenty_ten_ten_l1305_130532

/-- Represents a group of people with boys and girls -/
structure MyGroup where
  total : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of ways to select two people of different genders -/
def differentGenderSelections (g : MyGroup) : Nat :=
  g.total * g.girls

/-- Theorem stating that for a group of 20 people with 10 boys and 10 girls,
    the number of ways to select two people of different genders is 200 -/
theorem different_gender_selections_twenty_ten_ten :
  let g : MyGroup := { total := 20, boys := 10, girls := 10 }
  differentGenderSelections g = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_gender_selections_twenty_ten_ten_l1305_130532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_theorem_l1305_130545

def PositiveNat := {n : ℕ // n > 0}

def divides (a b : ℕ) := ∃ k, b = a * k

theorem identity_function_theorem (f : PositiveNat → PositiveNat) :
  (∀ m n : PositiveNat, divides ((f m).val^2 + (f n).val) ((m.val^2 + n.val)^2)) →
  (∀ n : PositiveNat, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_theorem_l1305_130545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1305_130551

/-- The distance from a point (a, b) to a line y = mx + c --/
noncomputable def distancePointToLine (a b m c : ℝ) : ℝ :=
  (|m * a - b + c|) / Real.sqrt (m^2 + 1)

theorem circle_line_intersection (r : ℝ) (hr : r > 0) :
  let circle := fun (x y : ℝ) ↦ x^2 + (y - 3)^2 = r^2
  let line := fun (x : ℝ) ↦ Real.sqrt 3 * x + 1
  let center_distance := distancePointToLine 0 3 (Real.sqrt 3) 1
  (∃ x y, circle x y ∧ y = line x) ∧ center_distance < r → r = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1305_130551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1305_130534

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the equation
def equation (m n : ℤ) : Prop :=
  floor (↑n * Real.sqrt 2) = (2 : ℤ) + floor (↑m * Real.sqrt 2)

-- State the theorem
theorem solution_exists : ∃ m n : ℤ, equation m n ∧ m = 1 ∧ n = 2 := by
  -- We use 'sorry' to skip the proof for now
  sorry

-- Example of a specific solution
example : equation 1 2 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1305_130534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1305_130566

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem problem_solution (A b c : ℝ) (hA : 0 < A ∧ A < π/2) (hb : b = Real.sqrt 2) (hc : c = 3) (hf : f A = 1) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, StrictMono (f ∘ (λ x ↦ x + k * π - π/8)) ∧ StrictMono (f ∘ (λ x ↦ -x + k * π + 3*π/8))) ∧
  A = π/4 ∧
  (Real.sqrt 5) ^ 2 = b^2 + c^2 - 2*b*c*Real.cos A :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1305_130566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_C_completion_time_l1305_130592

/-- A proof that worker C takes 10 days to complete the remaining work. -/
theorem worker_C_completion_time : ∃ (completion_time : ℝ),
  let work_rate_A : ℝ := 1 / 30
  let work_rate_B : ℝ := 1 / 30
  let work_rate_C : ℝ := 1 / 29.999999999999996
  let work_done_A : ℝ := work_rate_A * 10
  let work_done_B : ℝ := work_rate_B * 10
  let remaining_work : ℝ := 1 - (work_done_A + work_done_B)
  remaining_work = work_rate_C * completion_time ∧ completion_time = 10 :=
by
  -- The proof goes here
  sorry

#check worker_C_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_C_completion_time_l1305_130592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l1305_130581

theorem sin_2theta_value (θ : ℝ) 
  (h1 : Real.cos θ + Real.sin θ = 7/5) 
  (h2 : Real.cos θ - Real.sin θ = 1/5) : 
  Real.sin (2 * θ) = 48/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l1305_130581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l1305_130516

theorem smallest_odd_five_prime_factors : 
  ∀ n : ℕ, Odd n → (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₄ ≠ p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅) → n ≥ 15015 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l1305_130516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l1305_130500

-- Define the function f on the given domain
noncomputable def f (e a : ℝ) (x : ℝ) : ℝ := 
  if x < 0 ∧ x ≥ -e then a * x + Real.log (-x)
  else if x > 0 ∧ x ≤ e then a * x - Real.log x
  else 0  -- undefined outside the domain

-- State the theorem
theorem odd_function_extension (e a : ℝ) (h : e > 0) :
  (∀ x ∈ Set.Icc (-e) 0, f e a x = a * x + Real.log (-x)) →
  (∀ x ∈ Set.Ioo (-e) e, f e a x = -(f e a (-x))) →
  (∀ x ∈ Set.Icc 0 e, f e a x = a * x - Real.log x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l1305_130500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1305_130530

def f (x : ℝ) := x^2 - 4*x + 3

def g (x : ℝ) := 3*x - 2

def M : Set ℝ := {x | f (g x) > 0}

def N : Set ℝ := {x | g x < 2}

theorem intersection_M_N : M ∩ N = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1305_130530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_rotation_volume_l1305_130542

/-- Represents a semicircle with radius R -/
def semicircle (R : ℝ) : Set (ℝ × ℝ) := sorry

/-- Represents the tangent line parallel to the diameter of the semicircle -/
def tangent_line (R : ℝ) : Set (ℝ × ℝ) := sorry

/-- Calculates the volume of a solid of revolution -/
def volume_of_solid_of_revolution (shape : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ := sorry

/-- The volume of a solid formed by rotating a semicircle around its tangent line -/
theorem semicircle_rotation_volume (R : ℝ) (h : R > 0) :
  ∃ V : ℝ, V = (π / 3) * (3 * π - 4) * R^3 ∧
  V = volume_of_solid_of_revolution (semicircle R) (tangent_line R) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_rotation_volume_l1305_130542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_sum_range_l1305_130505

/-- Given that m and n are real numbers and the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1, prove that m+n is in the set (-∞, 2-2√2] ∪ [2+2√2, +∞) -/
theorem tangent_line_circle_sum_range (m n : ℝ) 
  (h_tangent : ∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → 
    ((x - 1)^2 + (y - 1)^2 = 1 → 
      ∀ ε > 0, ∃ x' y' : ℝ, (x' - x)^2 + (y' - y)^2 < ε^2 ∧ 
        (m + 1) * x' + (n + 1) * y' - 2 ≠ 0 ∧ (x' - 1)^2 + (y' - 1)^2 > 1)) :
  m + n ∈ Set.Iic (2 - 2 * Real.sqrt 2) ∪ Set.Ici (2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_sum_range_l1305_130505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_body_surface_area_main_result_l1305_130599

-- Define a spatial geometric body
structure SpatialBody where
  radius : ℝ

-- Define the conditions
def isSpecialBody (body : SpatialBody) : Prop :=
  body.radius = 5

-- Define the surface area
noncomputable def surfaceArea (body : SpatialBody) : ℝ :=
  4 * Real.pi * body.radius ^ 2

-- State the theorem
theorem special_body_surface_area (body : SpatialBody) :
  isSpecialBody body → surfaceArea body = 100 * Real.pi := by
  intro h
  unfold isSpecialBody at h
  unfold surfaceArea
  rw [h]
  ring

-- Proof of the main result
theorem main_result :
  ∃ (body : SpatialBody), isSpecialBody body ∧ surfaceArea body = 100 * Real.pi := by
  use ⟨5⟩
  constructor
  · rfl
  · exact special_body_surface_area ⟨5⟩ rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_body_surface_area_main_result_l1305_130599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_covering_theorem_l1305_130510

/-- A polygon in the plane -/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  sides : ℕ

/-- A covering of the plane -/
structure PlaneCovering where
  polygons : Set Polygon
  covers_plane : ∀ (p : ℝ × ℝ), ∃ (poly : Polygon), p ∈ poly.vertices

/-- Convexity of a polygon -/
def is_convex (p : Polygon) : Prop := sorry

/-- Congruence of polygons -/
def are_congruent (p1 p2 : Polygon) : Prop := sorry

/-- Main theorem -/
theorem plane_covering_theorem (n : ℕ) (h : n ≥ 7) :
  (∃ (c : PlaneCovering), ∀ (p : Polygon), p ∈ c.polygons → p.sides = n ∧ is_convex p) ∧
  (∃ (c : PlaneCovering), ∀ (p1 p2 : Polygon), p1 ∈ c.polygons → p2 ∈ c.polygons → p1.sides = n ∧ are_congruent p1 p2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_covering_theorem_l1305_130510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1305_130573

/-- The function f(x) defined for all real numbers x -/
noncomputable def f (x : ℝ) : ℝ := x^2/8 + x * Real.cos x + Real.cos (2*x)

/-- Theorem stating that f(x) has a minimum value of -1 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1305_130573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l1305_130536

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem sine_function_value (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  f ω (Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l1305_130536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_approximation_l1305_130568

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem root_approximation (h1 : f 0.64 < 0) (h2 : f 0.72 > 0) (h3 : f 0.68 < 0) 
  (h_continuous : Continuous f) :
  ∃ x : ℝ, x ∈ Set.Ioo 0.68 0.72 ∧ f x = 0 ∧ |x - 0.7| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_approximation_l1305_130568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l1305_130590

theorem integral_bounds (n : ℕ) (hn : n > 2) :
  (1/2 : ℝ) < ∫ x in (0 : ℝ)..(1/2), 1 / Real.sqrt (1 - x^n) ∧
  ∫ x in (0 : ℝ)..(1/2), 1 / Real.sqrt (1 - x^n) < π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l1305_130590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approx_l1305_130515

/-- Calculates the total payment for Plan 1 -/
noncomputable def plan1Payment (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  let balanceAt4Years := principal * (1 + rate / 12) ^ (12 * 4)
  let paymentAt4Years := balanceAt4Years / 3
  let remainingBalance := balanceAt4Years - paymentAt4Years
  let finalPayment := remainingBalance * (1 + rate / 12) ^ (12 * 8)
  paymentAt4Years + finalPayment

/-- Calculates the total payment for Plan 2 -/
noncomputable def plan2Payment (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * Real.exp (rate * years)

/-- Theorem stating the difference between Plan 2 and Plan 1 payments -/
theorem payment_difference_approx (principal : ℝ) (rate : ℝ) (years : ℝ) 
    (h_principal : principal = 12000)
    (h_rate : rate = 0.08)
    (h_years : years = 12) : 
  ∃ ε > 0, |plan2Payment principal rate years - plan1Payment principal rate years - 12124| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approx_l1305_130515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_monotonicity_condition_l1305_130567

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-5 : ℝ) 5

-- Part I: Minimum and maximum values when a = -1
theorem min_max_values :
  (∃ x ∈ domain, ∀ y ∈ domain, f (-1) x ≤ f (-1) y) ∧
  (∃ x ∈ domain, ∀ y ∈ domain, f (-1) y ≤ f (-1) x) ∧
  (∀ x ∈ domain, 1 ≤ f (-1) x) ∧
  (∃ x ∈ domain, f (-1) x = 37) ∧
  (∀ x ∈ domain, f (-1) x ≤ 37) :=
by sorry

-- Part II: Monotonicity condition
theorem monotonicity_condition :
  ∀ a : ℝ, (∀ x y : ℝ, x ∈ domain → y ∈ domain → x < y → f a x < f a y) ∨ 
            (∀ x y : ℝ, x ∈ domain → y ∈ domain → x < y → f a y < f a x) ↔ 
  a ≤ -5 ∨ 5 ≤ a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_monotonicity_condition_l1305_130567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_increasing_in_interval_l1305_130559

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (x - Real.pi / 4) ^ 2

-- Theorem for the smallest positive period
theorem smallest_positive_period (x : ℝ) : 
  ∃ T : ℝ, T > 0 ∧ (∀ y : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ y : ℝ, f (y + S) = f y) → T ≤ S) ∧ 
  T = Real.pi :=
sorry

-- Theorem for increasing function in the given interval
theorem increasing_in_interval : 
  ∀ x y : ℝ, -Real.pi/6 ≤ x ∧ x < y ∧ y ≤ Real.pi/6 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_increasing_in_interval_l1305_130559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1305_130552

/-- Given a function f(x) with an axis of symmetry, prove its interval of monotonic increase -/
theorem monotonic_increase_interval
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2)
  (h2 : ∃ k : ℤ, ∀ x, f (π / 3 - x) = f (π / 3 + x)) :
  ∃ k : ℤ, ∀ x y, x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6) →
                  y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6) →
                  x ≤ y → f x ≤ f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1305_130552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l1305_130553

/-- Definition of the ellipse -/
noncomputable def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of centroid being at origin -/
def centroid_at_origin (P A B : ℝ × ℝ) : Prop :=
  P.1 + A.1 + B.1 = 0 ∧ P.2 + A.2 + B.2 = 0

/-- Area of a triangle given three points -/
noncomputable def triangle_area (P A B : ℝ × ℝ) : ℝ :=
  abs ((P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2)) / 2)

/-- Main theorem -/
theorem constant_triangle_area (P A B : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 →
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  centroid_at_origin P A B →
  triangle_area P A B = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l1305_130553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_xoy_plane_l1305_130528

/-- The symmetric point of (x, y, z) with respect to the xOy plane is (x, y, -z) -/
def symmetricPointXOY (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2.1, -p.2.2)

theorem symmetric_point_xoy_plane :
  let P : ℝ × ℝ × ℝ := (1, 2, -3)
  symmetricPointXOY P = (1, 2, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_xoy_plane_l1305_130528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_removed_volume_l1305_130570

/-- The edge length of the regular tetrahedron -/
noncomputable def tetrahedron_edge_length : ℝ := 2

/-- The length of each side of the regular hexagon formed on each face -/
noncomputable def hexagon_side_length : ℝ := (2 * (Real.sqrt 3 - 2)) / (2 + Real.sqrt 3)

/-- The volume of a single removed tetrahedron -/
noncomputable def single_removed_tetrahedron_volume : ℝ :=
  (1 / 3) * (Real.sqrt 3 / 4) * (Real.sqrt 3 - 2)^2 * (2 - Real.sqrt 3 * (Real.sqrt 3 - 2))

/-- The number of corners removed from the regular tetrahedron -/
def num_removed_corners : ℕ := 4

/-- The theorem stating the total volume of removed tetrahedrons -/
theorem total_removed_volume :
  (↑num_removed_corners : ℝ) * single_removed_tetrahedron_volume =
  (Real.sqrt 3 / 3) * ((Real.sqrt 3 - 2)^2) * (2 - Real.sqrt 3 * (Real.sqrt 3 - 2)) := by
  sorry

#eval num_removed_corners

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_removed_volume_l1305_130570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l1305_130506

/-- Represents the angle between clock hands at a given time -/
structure ClockAngle where
  hours : Nat
  minutes : Nat
  smallerAngle : Float
  largerAngle : Float
  deriving Repr

/-- Calculates the angles between clock hands at 3:20 -/
def clockAngleAt3_20 : ClockAngle :=
  { hours := 3
  , minutes := 20
  , smallerAngle := 160.0
  , largerAngle := 200.0 }

/-- Theorem: The angles between clock hands at 3:20 are 160° and 200° -/
theorem clock_angle_at_3_20 :
  let angle := clockAngleAt3_20
  angle.smallerAngle = 160.0 ∧ angle.largerAngle = 200.0 := by
  sorry

#eval clockAngleAt3_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l1305_130506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_tangent_line_proof_intersection_product_constant_l1305_130512

noncomputable def circle_O : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 4}

def line_intersecting (x y : ℝ) : Prop :=
  3*x - y + Real.sqrt 5 = 0

noncomputable def chord_length (l : Set (ℝ × ℝ)) : ℝ :=
  Real.sqrt 14

def tangent_line (x y : ℝ) : Prop :=
  x + y - 2 * Real.sqrt 2 = 0

def point_on_circle (p : ℝ × ℝ) : Prop :=
  p ∈ circle_O

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

noncomputable def x_intersection (p q : ℝ × ℝ) : ℝ :=
  (p.1 * q.2 - q.1 * p.2) / (q.2 - p.2)

theorem circle_equation_proof (l : Set (ℝ × ℝ)) 
  (h1 : ∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ≠ q ∧ line_intersecting p.1 p.2 ∧ line_intersecting q.1 q.2)
  (h2 : chord_length l = Real.sqrt 14) : 
  circle_O = {p | p.1^2 + p.2^2 = 4} := by sorry

theorem tangent_line_proof :
  ∃ l : Set (ℝ × ℝ), ∀ p ∈ l, tangent_line p.1 p.2 ∧ 
  (∀ q : ℝ × ℝ, q.2 ≥ 0 → point_on_circle q → q ∈ l) ∧
  (∀ m : Set (ℝ × ℝ), (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ m = {p | p.1/a + p.2/b = 1}) →
    (∃ d e : ℝ × ℝ, d.2 = 0 ∧ e.1 = 0 ∧ d ∈ m ∧ e ∈ m →
      (d.1 - e.1)^2 + (d.2 - e.2)^2 ≥ (2 * Real.sqrt 2)^2)) := by sorry

theorem intersection_product_constant (m p : ℝ × ℝ) 
  (hm : point_on_circle m) (hp : point_on_circle p) :
  let n := symmetric_point m
  let x_m := x_intersection m p
  let x_n := x_intersection n p
  x_m * x_n = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_tangent_line_proof_intersection_product_constant_l1305_130512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gina_earnings_per_hour_l1305_130514

/-- Represents a painting order with cup counts, payment, and time limit -/
structure PaintingOrder where
  roses : ℕ
  lilies : ℕ
  sunflowers : ℕ
  orchids : ℕ
  payment : ℝ
  timeLimit : ℝ

/-- Represents Gina's painting rates and work schedule -/
structure PainterInfo where
  roseRate : ℝ
  lilyRate : ℝ
  sunflowerRate : ℝ
  orchidRate : ℝ
  totalHours : ℝ
  hoursPerDay : ℝ
  days : ℕ

def order1 : PaintingOrder := {
  roses := 6, lilies := 14, sunflowers := 4, orchids := 0,
  payment := 120, timeLimit := 6
}

def order2 : PaintingOrder := {
  roses := 2, lilies := 0, sunflowers := 0, orchids := 10,
  payment := 80, timeLimit := 3
}

def order3 : PaintingOrder := {
  roses := 0, lilies := 0, sunflowers := 8, orchids := 4,
  payment := 70, timeLimit := 4
}

def ginaInfo : PainterInfo := {
  roseRate := 6, lilyRate := 7, sunflowerRate := 5, orchidRate := 8,
  totalHours := 12, hoursPerDay := 4, days := 3
}

/-- Calculate the earnings per hour given painter info and orders -/
noncomputable def earningsPerHour (info : PainterInfo) (orders : List PaintingOrder) : ℝ :=
  sorry

theorem gina_earnings_per_hour :
  ∃ (eps : ℝ), eps > 0 ∧ abs (earningsPerHour ginaInfo [order1, order2, order3] - 36.10) < eps :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gina_earnings_per_hour_l1305_130514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_sqrt_5_l1305_130572

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
noncomputable def b (x : ℝ) : Fin 2 → ℝ := ![x, -4]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 2), v i = k * w i

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

-- The theorem to be proved
theorem magnitude_of_sum_equals_sqrt_5 (x : ℝ) 
  (h : parallel a (b x)) : 
  magnitude (λ i => a i + b x i) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_sqrt_5_l1305_130572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_quadrilateral_equality_l1305_130563

/-- A circle is tangent to three sides of a quadrilateral -/
structure CircleTangentToSides (r : ℝ) (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop where
  -- We'll leave this empty for now, as the specific conditions are not provided
  mk :: 

/-- A quadrilateral is convex -/
structure ConvexQuadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop where
  -- We'll leave this empty for now, as the specific conditions are not provided
  mk ::

/-- Given a convex quadrilateral ABCD and four circles with radii r₁, r₂, r₃, r₄
    tangent to its sides as described, prove the equality of ratios. -/
theorem tangent_circles_quadrilateral_equality
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (r₁ r₂ r₃ r₄ : ℝ)
  (convex : ConvexQuadrilateral A B C D)
  (tangent₁ : CircleTangentToSides r₁ D A B C)
  (tangent₂ : CircleTangentToSides r₂ A B C D)
  (tangent₃ : CircleTangentToSides r₃ B C D A)
  (tangent₄ : CircleTangentToSides r₄ C D A B) :
  (dist A B / r₁ + dist C D / r₃ = dist B C / r₂ + dist A D / r₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_quadrilateral_equality_l1305_130563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l1305_130598

theorem perfect_square_condition (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 0 < a^2 + b^2 - a*b*c ∧ a^2 + b^2 - a*b*c ≤ c + 1) :
  ∃ k : ℕ, a^2 + b^2 - a*b*c = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l1305_130598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_equation_l1305_130597

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  area : ℝ

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), ((P.1 - t.A.1)^2 + (P.2 - t.A.2)^2 = r^2) ∧
                   ((P.1 - t.B.1)^2 + (P.2 - t.B.2)^2 = r^2) ∧
                   ((P.1 - t.C.1)^2 + (P.2 - t.C.2)^2 = r^2)}

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem circumcircle_equation (t : Triangle) :
  ∀ P ∈ circumcircle t,
    (distance P t.A)^2 * Real.sin (2 * t.α) +
    (distance P t.B)^2 * Real.sin (2 * t.β) +
    (distance P t.C)^2 * Real.sin (2 * t.γ) = 4 * t.area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_equation_l1305_130597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1305_130574

theorem sqrt_inequality : Real.sqrt 6 - Real.sqrt 5 > 2 * Real.sqrt 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1305_130574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l1305_130555

def investment_A : ℕ := 6500
def investment_B : ℕ := 8400
def investment_C : ℕ := 10000

def months_A : ℕ := 6
def months_B : ℕ := 5
def months_C : ℕ := 3

def working_partner_percentage : ℚ := 5 / 100

def C_share : ℕ := 1900

theorem total_profit_calculation :
  let capital_months_A := investment_A * months_A
  let capital_months_B := investment_B * months_B
  let capital_months_C := investment_C * months_C
  let total_capital_months := capital_months_A + capital_months_B + capital_months_C
  let profit_share_C := C_share
  let total_profit : ℚ := (profit_share_C * total_capital_months) / (capital_months_C * (1 - working_partner_percentage))
  ⌊total_profit⌋ = 24667 := by sorry

#eval ⌊(C_share * (investment_A * months_A + investment_B * months_B + investment_C * months_C) : ℚ) / (investment_C * months_C * (1 - working_partner_percentage))⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l1305_130555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_odd_function_phi_l1305_130531

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (x + φ)

theorem cos_odd_function_phi (φ : ℝ) (h1 : 0 ≤ φ) (h2 : φ ≤ π) :
  (∀ x, f φ x = -f φ (-x)) → φ = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_odd_function_phi_l1305_130531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_calculation_l1305_130584

/-- Calculate the final amount after compound interest -/
def finalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Round a real number to the nearest integer -/
noncomputable def roundToNearestInt (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem final_amount_calculation :
  let principal : ℝ := 933.3333333333334
  let rate : ℝ := 0.05
  let time : ℝ := 4
  roundToNearestInt (finalAmount principal rate time) = 1120 := by
  sorry

#eval finalAmount 933.3333333333334 0.05 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_calculation_l1305_130584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l1305_130589

/-- A permutation of the sequence 1,2,3,4,5,6 -/
def Permutation := Fin 6 → Fin 6

/-- Checks if three consecutive terms in a permutation are increasing or decreasing -/
def hasThreeConsecutiveIncreasingOrDecreasing (p : Permutation) : Prop :=
  ∃ i : Fin 4, (p i < p (i + 1) ∧ p (i + 1) < p (i + 2)) ∨ 
               (p i > p (i + 1) ∧ p (i + 1) > p (i + 2))

/-- Checks if 1 and 2 are adjacent in a permutation -/
def oneAndTwoAdjacent (p : Permutation) : Prop :=
  ∃ i : Fin 5, (p i = 0 ∧ p (i + 1) = 1) ∨ (p i = 1 ∧ p (i + 1) = 0)

/-- The set of valid permutations according to the problem conditions -/
def ValidPermutations : Set Permutation :=
  {p | ¬hasThreeConsecutiveIncreasingOrDecreasing p ∧ oneAndTwoAdjacent p}

/-- Prove that ValidPermutations is finite -/
instance : Fintype ValidPermutations := by
  sorry

/-- The main theorem stating that the number of valid permutations is 96 -/
theorem valid_permutations_count : Fintype.card ValidPermutations = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l1305_130589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l1305_130578

/-- A circle in polar coordinates defined by ρ = 2cos θ -/
noncomputable def polar_circle (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- A line in polar coordinates defined by 3ρcos θ + 4ρsin θ + a = 0 -/
def polar_line (ρ θ a : ℝ) : Prop := 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0

/-- The condition for a circle and line to be tangent in polar coordinates -/
def is_tangent (a : ℝ) : Prop := 
  ∃ θ : ℝ, polar_line (polar_circle θ) θ a ∧
    ∀ φ : ℝ, φ ≠ θ → ¬ polar_line (polar_circle φ) φ a

/-- Theorem stating that if the circle and line are tangent, then a = 2 or a = -8 -/
theorem circle_tangent_line (a : ℝ) : 
  is_tangent a → a = 2 ∨ a = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l1305_130578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_iterations_of_f_on_two_l1305_130549

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem six_iterations_of_f_on_two :
  f (f (f (f (f (f 2))))) = 2 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_iterations_of_f_on_two_l1305_130549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1305_130522

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (M : ℝ), M = Real.sqrt 3 + 1 ∧
    ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M ∧
    ∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ Real.pi / 2 ∧ f x₀ = M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1305_130522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_XY_is_80_l1305_130513

-- Define the points
variable (X Y G H I J : ℝ)

-- Define the midpoint relation
def is_midpoint (a b c : ℝ) : Prop := c = (a + b) / 2

-- State the theorem
theorem length_XY_is_80 
  (hG : is_midpoint X Y G)
  (hH : is_midpoint X G H)
  (hI : is_midpoint X H I)
  (hJ : is_midpoint X I J)
  (hXJ : X - J = 5) : 
  X - Y = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_XY_is_80_l1305_130513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1305_130521

-- Define the line l
def line_equation (x y θ : ℝ) : Prop := x - y * Real.sin θ + 2 = 0

-- Define the slope angle α
noncomputable def slope_angle (θ : ℝ) : ℝ :=
  if Real.sin θ = 0 then Real.pi / 2
  else Real.arctan (1 / Real.sin θ)

-- Theorem statement
theorem slope_angle_range :
  ∀ θ : ℝ, ∃ α : ℝ, line_equation 0 0 θ → Real.pi / 4 ≤ slope_angle θ ∧ slope_angle θ ≤ 3 * Real.pi / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1305_130521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_negative_product_l1305_130565

def sequence_a (n : ℕ) : ℚ :=
  15 - 2/3 * (n - 1)

theorem adjacent_negative_product :
  ∀ n : ℕ, n < 23 → sequence_a n * sequence_a (n + 1) > 0 ∧
  sequence_a 23 * sequence_a 24 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_negative_product_l1305_130565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_l1305_130596

/-- Given a set of observations with an incorrect value, calculate the corrected mean -/
theorem corrected_mean (n : ℕ) (original_mean incorrect_value correct_value : ℚ) :
  n > 0 →
  let original_sum := n * original_mean
  let corrected_sum := original_sum - incorrect_value + correct_value
  corrected_sum / n = 36.46 :=
by
  intro h_n_pos
  -- The proof would go here
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_l1305_130596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_southbound_vehicle_count_l1305_130585

/-- The speed of vehicles in miles per hour -/
noncomputable def speed : ℝ := 70

/-- The observation time in hours -/
noncomputable def observationTime : ℝ := 10 / 60

/-- The number of southbound vehicles observed -/
def observedVehicles : ℕ := 30

/-- The length of the highway section in miles -/
noncomputable def sectionLength : ℝ := 150

/-- The number of southbound vehicles in the given section -/
def southboundVehicles : ℕ := 193

theorem southbound_vehicle_count :
  let relativeSpeed := 2 * speed
  let relativeDistance := relativeSpeed * observationTime
  let vehicleDensity := (observedVehicles : ℝ) / relativeDistance
  ⌊vehicleDensity * sectionLength⌋₊ = southboundVehicles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_southbound_vehicle_count_l1305_130585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1305_130526

theorem complex_absolute_value (ω : ℂ) (h : ω = 5 + 3*Complex.I) : 
  Complex.abs (ω^2 + 10*ω + 40) = 4 * Real.sqrt 1066 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1305_130526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_doors_l1305_130586

theorem apartment_doors (num_buildings : ℕ) (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) (doors_per_apartment : ℕ) : 
  num_buildings * floors_per_building * apartments_per_floor * doors_per_apartment = 1008 :=
  by
  -- Define the given conditions
  have h1 : num_buildings = 2 := by sorry
  have h2 : floors_per_building = 12 := by sorry
  have h3 : apartments_per_floor = 6 := by sorry
  have h4 : doors_per_apartment = 7 := by sorry

  -- Calculate the total number of doors
  calc
    num_buildings * floors_per_building * apartments_per_floor * doors_per_apartment
    = 2 * 12 * 6 * 7 := by sorry
    _ = 1008 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_doors_l1305_130586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l1305_130560

theorem triangle_sine_inequality (a b c : ℝ) (γ : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 ≤ γ) (h5 : γ < 2 * Real.pi) :
  Real.sin (γ / 2) ≤ c / (a + b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l1305_130560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1305_130550

/-- A parabola with vertex at the origin, symmetric about the y-axis, and passing through (√3, -2√3) has the equation x² = -√3/2 * y -/
theorem parabola_equation :
  ∀ (x y p : ℝ),
  (∀ (x₀ y₀ : ℝ), x₀^2 = -2*p*y₀ → (-x₀)^2 = -2*p*y₀) →  -- symmetry about y-axis
  0^2 = -2*p*0 →  -- vertex at origin
  (Real.sqrt 3)^2 = -2*p*(-2*Real.sqrt 3) →  -- passes through (√3, -2√3)
  (x^2 = -2*p*y ↔ x^2 = -(Real.sqrt 3)/2*y) :=  -- standard equation
by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1305_130550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1305_130582

noncomputable def inverse_proportion (x : ℝ) : ℝ := -6 / x

theorem inverse_proportion_quadrants :
  ∀ x y : ℝ, y = inverse_proportion x →
  (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1305_130582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_needed_to_reap_l1305_130519

/-- Calculates the number of men needed to reap a given area of land in a specific number of days,
    given a reference group's performance. -/
noncomputable def calculate_men (reference_men : ℕ) (reference_acres : ℝ) (reference_days : ℕ)
                  (target_acres : ℝ) (target_days : ℕ) : ℝ :=
  (reference_men : ℝ) * target_acres * (reference_days : ℝ) / (reference_acres * (target_days : ℝ))

theorem men_needed_to_reap (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (n : ℕ), abs (n - calculate_men 12 120 36 413.33333333333337 62) < ε ∧ n = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_needed_to_reap_l1305_130519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l1305_130541

-- Define the map M as a simple graph
structure CityMap where
  cities : Type
  routes : cities → cities → Prop

-- Define M_a as a subgraph of M
def subgraph_M_a (M : CityMap) (a : M.cities) : CityMap where
  cities := {b : M.cities | M.routes a b}
  routes := λ b c => M.routes b c

-- Define the cut of a graph
noncomputable def cut (G : CityMap) : ℕ :=
  sorry -- Maximum number of routes between two disjoint sets of cities

-- Define the number of routes in a graph
noncomputable def number_of_routes (G : CityMap) : ℕ :=
  sorry -- Total number of routes in the graph

-- Condition on the cut of M_a
axiom cut_condition (M : CityMap) (a : M.cities) :
  (cut (subgraph_M_a M a) : ℚ) < (2/3 : ℚ) * (number_of_routes (subgraph_M_a M a) : ℚ)

-- Two-coloring of routes
def two_coloring (M : CityMap) : M.cities → M.cities → Bool :=
  sorry

-- Monochromatic triangle
def monochromatic_triangle (M : CityMap) (coloring : M.cities → M.cities → Bool) : Prop :=
  ∃ a b c : M.cities, 
    M.routes a b ∧ M.routes b c ∧ M.routes c a ∧
    coloring a b = coloring b c ∧ coloring b c = coloring c a

-- The main theorem
theorem monochromatic_triangle_exists (M : CityMap) :
  ∀ coloring : M.cities → M.cities → Bool, monochromatic_triangle M coloring :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l1305_130541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1305_130523

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (Real.sqrt (a^2 + b^2) / a = 3) →
  (∀ x y : ℝ, (x = 2 * Real.sqrt 2 * y ∨ x = -2 * Real.sqrt 2 * y) ↔ y = a/b * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1305_130523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_power_of_matrix_l1305_130548

theorem det_power_of_matrix {n : Type*} [Fintype n] [DecidableEq n] 
  (M : Matrix n n ℝ) (h : Matrix.det M = 3) : Matrix.det (M^5) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_power_of_matrix_l1305_130548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_proof_l1305_130511

/-- The altitude of an airplane given two observers' positions and angles of elevation -/
noncomputable def airplane_altitude (distance_between_observers : ℝ) 
                      (angle_alice : ℝ) 
                      (angle_bob : ℝ) : ℝ :=
  distance_between_observers / 3

theorem airplane_altitude_proof 
  (distance_between_observers : ℝ) 
  (angle_alice : ℝ) 
  (angle_bob : ℝ) 
  (h_distance : distance_between_observers = 12)
  (h_angle_alice : angle_alice = Real.pi / 4)  -- 45 degrees in radians
  (h_angle_bob : angle_bob = Real.pi / 6)  -- 30 degrees in radians
  : airplane_altitude distance_between_observers angle_alice angle_bob = 4 := by
  unfold airplane_altitude
  rw [h_distance]
  norm_num

#check airplane_altitude_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_proof_l1305_130511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_subset_implies_a_range_l1305_130517

noncomputable def f (x : ℝ) := Real.sqrt ((x + 1) / (x - 2))

noncomputable def g (a : ℝ) (x : ℝ) := 1 / Real.sqrt (x^2 - (2*a + 1)*x + a^2 + a)

def domain_f : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ -1}

def domain_g (a : ℝ) : Set ℝ := {x : ℝ | x > a + 1 ∨ x < a}

theorem domain_subset_implies_a_range (a : ℝ) :
  domain_f ⊆ domain_g a → -1 < a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_subset_implies_a_range_l1305_130517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_l1305_130501

/-- Calculate population after two years of growth -/
theorem population_growth (initial_population : ℕ) (growth_rate1 : ℚ) (growth_rate2 : ℚ) : 
  initial_population = 13000 ∧ growth_rate1 = 1/10 ∧ growth_rate2 = 3/20 →
  (initial_population : ℚ) * (1 + growth_rate1) * (1 + growth_rate2) = 16445 := by
  intro h
  sorry

#check population_growth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_l1305_130501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l1305_130557

noncomputable def h (t : ℝ) : ℝ := (t^2 + (5/4)*t) / (t^2 + 2)

theorem range_of_h :
  Set.range h = Set.Icc (-1/8 : ℝ) (25/16 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l1305_130557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fourth_term_l1305_130575

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_fourth_term
  (seq : ArithmeticSequence)
  (h : S seq 7 = 28) :
  seq.a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fourth_term_l1305_130575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_col_products_not_equal_l1305_130576

/-- Represents a 10x10 table filled with natural numbers from 108 to 207 -/
def Table := Fin 10 → Fin 10 → Nat

/-- Predicate to check if a table is valid according to the problem conditions -/
def is_valid_table (t : Table) : Prop :=
  (∀ i j, 108 ≤ t i j ∧ t i j ≤ 207) ∧
  (∀ i j i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j')

/-- Calculate the product of a row -/
def row_product (t : Table) (i : Fin 10) : Nat :=
  (Finset.univ : Finset (Fin 10)).prod (λ j ↦ t i j)

/-- Calculate the product of a column -/
def col_product (t : Table) (j : Fin 10) : Nat :=
  (Finset.univ : Finset (Fin 10)).prod (λ i ↦ t i j)

/-- The set of row products -/
def row_products (t : Table) : Finset Nat :=
  (Finset.univ : Finset (Fin 10)).image (row_product t)

/-- The set of column products -/
def col_products (t : Table) : Finset Nat :=
  (Finset.univ : Finset (Fin 10)).image (col_product t)

/-- Main theorem: The sets of row products and column products cannot be identical -/
theorem row_col_products_not_equal (t : Table) (h : is_valid_table t) :
  row_products t ≠ col_products t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_col_products_not_equal_l1305_130576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_lemon_production_l1305_130503

/-- The number of lemons produced by a normal lemon tree per year -/
def normal_lemon_production : ℕ := 60

/-- The percentage increase in lemon production for Jim's trees -/
def jim_tree_increase : ℚ := 1/2

/-- The width of Jim's lemon grove in number of trees -/
def grove_width : ℕ := 50

/-- The length of Jim's lemon grove in number of trees -/
def grove_length : ℕ := 30

/-- The number of years for which we calculate lemon production -/
def production_years : ℕ := 5

/-- Theorem stating that Jim's lemon production over 5 years is 675,000 lemons -/
theorem jim_lemon_production :
  (grove_width * grove_length) *
  (normal_lemon_production * (1 + jim_tree_increase)).floor *
  production_years = 675000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_lemon_production_l1305_130503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1305_130547

/-- The length of a train given its speed, the speed and length of another train 
    running in the opposite direction, and the time they take to cross each other. -/
noncomputable def train_length (speed1 speed2 : ℝ) (length2 time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * time
  total_distance - length2

/-- Theorem stating that under the given conditions, the length of Train 1 
    is approximately 180 meters. -/
theorem train_length_problem :
  let speed1 := (120 : ℝ) -- km/h
  let speed2 := (80 : ℝ)  -- km/h
  let time := (9 : ℝ)     -- seconds
  let length2 := (320.04 : ℝ) -- meters
  abs (train_length speed1 speed2 length2 time - 180) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1305_130547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_students_count_l1305_130561

/-- Represents the types of students in the class -/
inductive StudentType
  | Excellent
  | Average
  | Poor

/-- Represents the class of students -/
structure ClassComposition where
  total : Nat
  excellent : Nat
  average : Nat
  poor : Nat

/-- The conditions of the problem -/
structure ProblemConditions where
  c : ClassComposition
  excellent_yes : Nat
  average_yes : Nat
  poor_yes : Nat

/-- The main theorem to prove -/
theorem average_students_count (cond : ProblemConditions) : cond.c.average = 20 :=
  by
  -- Assuming the following conditions
  have h1 : cond.c.total = 30 := by sorry
  have h2 : cond.c.excellent + cond.c.average + cond.c.poor = cond.c.total := by sorry
  have h3 : cond.excellent_yes = 19 := by sorry
  have h4 : cond.average_yes = 12 := by sorry
  have h5 : cond.poor_yes = 9 := by sorry
  have h6 : cond.excellent_yes = cond.c.excellent + cond.c.poor + (cond.c.average / 2) := by sorry
  have h7 : cond.average_yes = cond.c.poor + (cond.c.average / 2) := by sorry
  have h8 : cond.poor_yes = cond.c.average / 2 := by sorry
  
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_students_count_l1305_130561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_first_group_length_value_l1305_130546

/-- A sequence type representing a random arrangement of 19 ones and 49 zeros -/
def Sequence := Fin 68 → Fin 2

/-- The expected length of the first group in a sequence -/
noncomputable def expected_first_group_length (seq : Sequence) : ℝ :=
  19 * (1 / 50) + 49 * (1 / 20)

/-- Theorem stating the expected length of the first group -/
theorem expected_first_group_length_value (seq : Sequence) :
  expected_first_group_length seq = 2.83 := by
  unfold expected_first_group_length
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_first_group_length_value_l1305_130546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_time_approx_l1305_130509

/-- Represents a trapezoidal field with given dimensions and running speeds -/
structure TrapezoidalField where
  shortSide : ℝ
  longSide : ℝ
  nonParallelSide : ℝ
  speedShortSide : ℝ
  speedNonParallelSide : ℝ
  speedLongSide : ℝ

/-- Calculates the time to run around the trapezoidal field -/
noncomputable def runningTime (field : TrapezoidalField) : ℝ :=
  field.shortSide / (field.speedShortSide * 5 / 18) +
  2 * field.nonParallelSide / (field.speedNonParallelSide * 5 / 18) +
  field.longSide / (field.speedLongSide * 5 / 18)

/-- Theorem stating the running time for the given trapezoidal field -/
theorem running_time_approx (field : TrapezoidalField)
  (h1 : field.shortSide = 30)
  (h2 : field.longSide = 40)
  (h3 : field.nonParallelSide = 35)
  (h4 : field.speedShortSide = 5)
  (h5 : field.speedNonParallelSide = 4)
  (h6 : field.speedLongSide = 7) :
  ∃ ε > 0, |runningTime field - 105.17| < ε := by
  sorry

#check running_time_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_time_approx_l1305_130509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l1305_130538

-- Define the ellipse
def isOnEllipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + P.2^2 = 1

-- Define the foci
def foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem min_sum_of_squares (P : ℝ × ℝ) (h : isOnEllipse P) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (Q : ℝ × ℝ), isOnEllipse Q →
    (distance Q foci.1)^2 + (distance Q foci.2)^2 ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l1305_130538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_net_change_l1305_130539

/-- Represents the sequence of percentage changes applied to a salary -/
def salary_changes : List ℝ := [1.15, 0.90, 1.20, 0.95]

/-- Calculates the net change factor after applying a sequence of percentage changes -/
def net_change_factor (changes : List ℝ) : ℝ :=
  changes.foldl (· * ·) 1

/-- Theorem: The net change in salary after the given changes is equal to a 3.55% increase -/
theorem salary_net_change :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |net_change_factor salary_changes - 1.0355| < ε := by
  sorry

#eval net_change_factor salary_changes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_net_change_l1305_130539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_event_proof_l1305_130587

-- Define the set I
def I : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool := n % 2 = 0

-- Define the event A: at least two numbers are even
def eventA (s : Finset Nat) : Prop := (s.filter (fun n => isEven n)).card ≥ 2

-- Define the complementary event of A
def complementEventA (s : Finset Nat) : Prop := (s.filter (fun n => isEven n)).card ≤ 1

-- Theorem statement
theorem complementary_event_proof :
  ∀ s : Finset Nat, s ⊆ I → s.card = 5 →
  (¬ eventA s ↔ complementEventA s) := by
  sorry

#check complementary_event_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_event_proof_l1305_130587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_difference_l1305_130554

/-- Represents a school with students and teachers. -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculates the average number of students per class from a teacher's perspective. -/
noncomputable def teacher_perspective (s : School) : ℝ :=
  (s.num_students : ℝ) / (s.num_teachers : ℝ)

/-- Calculates the average number of students per class from a student's perspective. -/
noncomputable def student_perspective (s : School) : ℝ :=
  (s.class_sizes.map (λ n => (n : ℝ) * (n : ℝ) / (s.num_students : ℝ))).sum

/-- The main theorem stating the difference between perspectives. -/
theorem perspective_difference (s : School) 
  (h1 : s.num_students = 120)
  (h2 : s.num_teachers = 6)
  (h3 : s.class_sizes = [60, 30, 20, 5, 3, 2])
  (h4 : s.class_sizes.sum = s.num_students) :
  teacher_perspective s - student_perspective s = -21.148 := by
  sorry

#eval toString (-21.148 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_difference_l1305_130554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_with_at_least_7_black_count_l1305_130537

/-- Represents a square on the checkerboard --/
structure Square where
  size : Nat
  position : Nat × Nat

/-- Calculates the number of black squares in a given square --/
def blackSquaresCount (s : Square) : Nat :=
  sorry

/-- Checks if a square is valid (fits on the 10x10 board) --/
def isValidSquare (s : Square) : Bool :=
  sorry

/-- The set of all valid squares on the 10x10 board --/
def allValidSquares : Finset Square :=
  sorry

/-- The set of squares containing at least 7 black squares --/
def squaresWithAtLeast7Black : Finset Square :=
  sorry

/-- Main theorem: The number of squares with at least 7 black squares is 140 --/
theorem squares_with_at_least_7_black_count :
  squaresWithAtLeast7Black.card = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_with_at_least_7_black_count_l1305_130537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l1305_130544

noncomputable def f (x : ℝ) := 2^x + x * Real.log (1/4)

theorem max_value_f : 
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧ 
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y ≤ f x) ∧
  f x = 1/4 + 4 * Real.log 2 := by
  sorry

#check max_value_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l1305_130544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_32_over_3_l1305_130535

/-- The line function y = 2x + 3 -/
def line (x : ℝ) : ℝ := 2 * x + 3

/-- The parabola function y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The area enclosed by the line y = 2x + 3 and the parabola y = x^2 -/
noncomputable def enclosed_area : ℝ := ∫ x in (-1)..3, (line x - parabola x)

theorem area_enclosed_is_32_over_3 : enclosed_area = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_32_over_3_l1305_130535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_T_l1305_130533

-- Define the function f(x) = 1 + x ln x
noncomputable def f (x : ℝ) : ℝ := 1 + x * Real.log x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 1 + Real.log x

-- Property T theorem
theorem property_T : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' x₁ * f' x₂ = -1 := by
  -- We need to find two values of x where f'(x₁) * f'(x₂) = -1
  -- Let's choose x₁ = 1/e and x₂ = e
  let x₁ : ℝ := Real.exp (-1)
  let x₂ : ℝ := Real.exp 1
  
  have h1 : x₁ ≠ x₂ := by
    apply ne_of_lt
    exact Real.exp_lt_exp.mpr (by norm_num)
  
  have h2 : f' x₁ * f' x₂ = -1 := by
    -- f'(1/e) = 1 + log(1/e) = 1 - 1 = 0
    -- f'(e) = 1 + log(e) = 1 + 1 = 2
    -- 0 * 2 = 0 ≠ -1, so this choice doesn't work
    sorry -- We need to find the correct values of x₁ and x₂
  
  exact ⟨x₁, x₂, h1, h2⟩

-- The actual proof of property T for this function is more complex
-- and requires finding the correct values of x₁ and x₂.
-- The above is just a skeleton of the proof structure.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_T_l1305_130533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfactory_fraction_is_three_fourths_l1305_130540

/-- Represents the grades in a class --/
inductive Grade
| A
| B
| C
| D
| F

/-- Determines if a grade is satisfactory --/
def is_satisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | Grade.D => true
  | Grade.F => false

/-- Represents the number of students for each grade --/
def grade_count : Grade → Nat
| Grade.A => 5
| Grade.B => 4
| Grade.C => 3
| Grade.D => 3
| Grade.F => 5

/-- The total number of students --/
def total_students : Nat :=
  grade_count Grade.A + grade_count Grade.B + grade_count Grade.C +
  grade_count Grade.D + grade_count Grade.F

/-- The number of students with satisfactory grades --/
def satisfactory_students : Nat :=
  grade_count Grade.A + grade_count Grade.B + grade_count Grade.C + grade_count Grade.D

theorem satisfactory_fraction_is_three_fourths :
  (satisfactory_students : ℚ) / (total_students : ℚ) = 3 / 4 := by
  sorry

#eval satisfactory_students
#eval total_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfactory_fraction_is_three_fourths_l1305_130540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_read_all_three_l1305_130543

def TotalWorkers : ℕ := 120

def SaramagoReaders : ℚ := 1/4
def KureishiReaders : ℚ := 5/8
def DuboisReaders : ℚ := 3/10

def WorkersReadAllThree (x : ℕ) : Prop :=
  x = (DuboisReaders * ↑TotalWorkers).floor / 2 ∧
  x = (TotalWorkers - (SaramagoReaders * ↑TotalWorkers).floor - (KureishiReaders * ↑TotalWorkers).floor + x)

theorem workers_read_all_three :
  ∃ x : ℕ, WorkersReadAllThree x ∧ x = 18 := by
  sorry

#eval (DuboisReaders * ↑TotalWorkers).floor / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_read_all_three_l1305_130543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_approximation_l1305_130504

/-- The perimeter of a semicircle in centimeters -/
noncomputable def semicircle_perimeter : ℝ := 126

/-- The radius of the semicircle in centimeters -/
noncomputable def semicircle_radius : ℝ := semicircle_perimeter / (Real.pi + 2)

/-- Theorem stating that the radius of a semicircle with perimeter 126 cm is approximately 24.5 cm -/
theorem semicircle_radius_approximation :
  abs (semicircle_radius - 24.5) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_approximation_l1305_130504
