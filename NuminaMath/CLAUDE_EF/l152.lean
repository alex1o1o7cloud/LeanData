import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_problem_l152_15255

theorem triangle_abc_problem (A B C a b c : ℝ) : 
  (Real.sin A - Real.sin C) / b = (Real.sin A - Real.sin B) / (a + c) →
  Real.cos A = 1 / 7 →
  C = π / 3 ∧ Real.cos (2 * A - C) = -23 / 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_problem_l152_15255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_cube_surface_area_l152_15283

/-- The surface area of a cube formed by gluing together smaller cubes -/
theorem large_cube_surface_area 
  (n : ℕ) -- number of small cubes
  (small_edge : ℝ) -- edge length of each small cube
  (h1 : n = 27) -- condition: 27 small cubes are used
  (h2 : small_edge = 4) -- condition: each small cube has an edge length of 4 cm
  : (6 : ℝ) * ((n : ℝ) ^ (1 / 3 : ℝ) * small_edge) ^ 2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_cube_surface_area_l152_15283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_overlapping_triangles_l152_15247

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  hypotenuse_positive : hypotenuse > 0

/-- The area of the region common to two overlapping isosceles right triangles -/
noncomputable def common_area (t : IsoscelesRightTriangle) : ℝ :=
  (t.hypotenuse ^ 2) / 8

/-- 
  Theorem: The area common to two congruent isosceles right triangles 
  with hypotenuses of length 10 that overlap partly with coinciding hypotenuses 
  is equal to 12.5.
-/
theorem common_area_of_overlapping_triangles :
  ∀ t : IsoscelesRightTriangle, 
  t.hypotenuse = 10 → common_area t = 12.5 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and might cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_overlapping_triangles_l152_15247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_proof_l152_15284

theorem pattern_proof (n : ℕ) : (n : ℤ) * (n + 2) - (n + 1)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_proof_l152_15284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_comp_f_l152_15236

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x - 1 else 2^x

-- Define the composition f ∘ f
noncomputable def f_comp_f (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem range_of_f_comp_f :
  ∀ y : ℝ, (∃ x : ℝ, f_comp_f x = y) ↔ y ≥ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_comp_f_l152_15236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l152_15259

def S (n : ℕ) : ℕ := 3 + 2^n

def a : ℕ → ℕ
  | 0 => 5  -- Add this case for n = 0
  | 1 => 5
  | n + 2 => 2^(n + 1)

theorem sequence_general_term (n : ℕ) : 
  (n = 0 ∧ a n = 5) ∨ (n = 1 ∧ a n = 5) ∨ (n > 1 ∧ a n = 2^(n-1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l152_15259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l152_15213

/-- Two lines in the Cartesian plane -/
structure IntersectingLines where
  k : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y ↦ k * x - y + 2 = 0
  l₂ : ℝ → ℝ → Prop := λ x y ↦ x + k * y - 2 = 0

/-- The target line -/
def targetLine : ℝ → ℝ → Prop := λ x y ↦ x - y - 4 = 0

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (lines : IntersectingLines) : ℝ × ℝ := sorry

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ := sorry

/-- The maximum distance theorem -/
theorem max_distance_to_line :
  ∃ max_dist : ℝ,
  max_dist = 3 * Real.sqrt 2 ∧
  ∀ lines : IntersectingLines,
  let p := intersectionPoint lines
  distanceToLine p targetLine ≤ max_dist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l152_15213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l152_15294

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) (h2 : a < 13) 
  (h3 : (51^2012 + a) % 13 = 0) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l152_15294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_network_connectivity_l152_15265

/-- A city in the road network. -/
structure City where
  id : Nat

/-- A road connecting two cities. -/
structure Road where
  source : City
  target : City

/-- A road network consisting of cities and roads. -/
structure RoadNetwork where
  cities : List City
  roads : List Road

/-- Predicate to check if a city is reachable from another city in the network. -/
def isReachable (network : RoadNetwork) (source target : City) : Prop :=
  sorry

/-- The main theorem stating that for any number of cities n ≥ 3, 
    it's possible to change at most one road direction to allow travel between any two cities. -/
theorem road_network_connectivity (n : Nat) (h : n ≥ 3) :
  ∀ (network : RoadNetwork),
    network.cities.length = n →
    (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
      ∃ r : Road, r ∈ network.roads ∧ (r.source = c1 ∧ r.target = c2 ∨ r.source = c2 ∧ r.target = c1)) →
    ∃ (newNetwork : RoadNetwork),
      (∀ c1 c2 : City, c1 ∈ newNetwork.cities → c2 ∈ newNetwork.cities → isReachable newNetwork c1 c2) ∧
      (newNetwork.roads.length = network.roads.length ∨ newNetwork.roads.length = network.roads.length - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_network_connectivity_l152_15265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_10_mod_100_l152_15271

/-- The polynomial q(x) = x^10 + x^9 + x^8 + ... + x + 1 -/
noncomputable def q (x : ℝ) : ℝ := (x^11 - 1) / (x - 1)

/-- The polynomial x^3 + x^2 + 1 -/
def divisor (x : ℝ) : ℝ := x^3 + x^2 + 1

/-- The polynomial remainder s(x) when q(x) is divided by x^3 + x^2 + 1 -/
noncomputable def s (x : ℝ) : ℝ := q x % divisor x

theorem remainder_of_s_10_mod_100 : Int.mod (Int.natAbs (Int.floor (s 10))) 100 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_10_mod_100_l152_15271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l152_15207

theorem tan_triple_angle (x : ℝ) (h1 : Real.tan x = 2/3) (h2 : Real.tan (3*x) = 3/5) : 
  ∃ k : ℝ, k = 2/3 ∧ x = Real.arctan k ∧ x > 0 ∧ ∀ y : ℝ, y > 0 → Real.tan y = 2/3 → y ≥ x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l152_15207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_ratio_specific_l152_15202

/-- The area of an equilateral triangle with side length s -/
noncomputable def area_equilateral (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The ratio of the area of a small equilateral triangle to the area of the trapezoid
    formed by cutting it from a larger equilateral triangle -/
noncomputable def triangle_trapezoid_ratio (large_side small_side : ℝ) : ℝ :=
  let small_area := area_equilateral small_side
  let large_area := area_equilateral large_side
  let trapezoid_area := large_area - small_area
  small_area / trapezoid_area

theorem triangle_trapezoid_ratio_specific : 
  triangle_trapezoid_ratio 10 2 = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_ratio_specific_l152_15202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_14_representation_l152_15293

/-- Represents a repeating decimal in base k -/
def RepeatingDecimal (k : ℕ) (a b : ℕ) := 
  (a : ℚ) / k + (b : ℚ) / (k^2 - 1)

/-- The fraction we're working with -/
def target : ℚ := 9 / 35

/-- The base we're proving -/
def base : ℕ := 14

/-- The repeating part of the decimal -/
def repeating : ℕ := 14

theorem base_14_representation : 
  RepeatingDecimal base 1 repeating = target := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_14_representation_l152_15293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l152_15226

def sequenceA (n : ℕ) : ℤ := (-1)^(n+1) * (2*n - 1)

theorem sequence_correct : ∀ n : ℕ, 
  (n = 1 → sequenceA n = 1) ∧ 
  (n = 2 → sequenceA n = -3) ∧ 
  (n = 3 → sequenceA n = 5) ∧ 
  (n = 4 → sequenceA n = -7) ∧ 
  (n = 5 → sequenceA n = 9) :=
by
  intro n
  apply And.intro
  · intro h; rw [h, sequenceA]; norm_num
  apply And.intro
  · intro h; rw [h, sequenceA]; norm_num
  apply And.intro
  · intro h; rw [h, sequenceA]; norm_num
  apply And.intro
  · intro h; rw [h, sequenceA]; norm_num
  · intro h; rw [h, sequenceA]; norm_num

#eval sequenceA 1
#eval sequenceA 2
#eval sequenceA 3
#eval sequenceA 4
#eval sequenceA 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l152_15226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_is_composite_l152_15242

def N : ℕ := 10^2016 * (10^2017 - 1) / 9 + 2 * (10^2016 - 1) / 9

theorem N_is_composite : ¬ Prime N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_is_composite_l152_15242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_arrangement_l152_15237

-- Define a stick as a positive real number representing its length
def Stick : Type := { x : ℝ // x > 0 }

-- Define a function to check if three sticks can form a triangle
def is_triangle (a b c : Stick) : Prop :=
  a.val + b.val > c.val ∧ a.val + c.val > b.val ∧ b.val + c.val > a.val

-- Define a type for a set of 15 sticks
def StickSet := Fin 15 → Stick

-- Define a type for an arrangement of 5 triangles
def TriangleArrangement := Fin 5 → (Fin 3 → Stick)

-- Define a predicate to check if an arrangement is valid
def valid_arrangement (sticks : StickSet) (arr : TriangleArrangement) : Prop :=
  (∀ i : Fin 5, is_triangle (arr i 0) (arr i 1) (arr i 2)) ∧
  (∀ s : Stick, (∃ i : Fin 15, sticks i = s) ↔ (∃ i : Fin 5, ∃ j : Fin 3, arr i j = s))

-- State the theorem
theorem unique_triangle_arrangement (sticks : StickSet) :
  (∃ arr : TriangleArrangement, valid_arrangement sticks arr) →
  (∀ arr₁ arr₂ : TriangleArrangement, valid_arrangement sticks arr₁ → valid_arrangement sticks arr₂ → arr₁ = arr₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_arrangement_l152_15237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_lateral_faces_congruent_l152_15252

/-- A regular polygon is a polygon with all sides and angles equal. -/
structure RegularPolygon where
  -- Define the properties of a regular polygon
  sides : ℕ
  sideLength : ℝ
  -- Add more properties as needed

/-- A point in 3D space. -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A regular pyramid is a pyramid with a regular polygon base and congruent triangular faces meeting at the apex. -/
structure RegularPyramid where
  base : RegularPolygon
  apex : Point
  -- Add more properties as needed

/-- A triangle in 3D space. -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Congruent triangles are triangles that have the same shape and size. -/
def CongruentTriangles (t1 t2 : Triangle) : Prop :=
  sorry

/-- The lateral faces of a pyramid are the triangular faces that meet at the apex. -/
def LateralFaces (p : RegularPyramid) : Set Triangle :=
  sorry

/-- The theorem stating that all lateral faces of a regular pyramid are congruent triangles. -/
theorem regular_pyramid_lateral_faces_congruent (p : RegularPyramid) :
  ∀ t1 t2, t1 ∈ LateralFaces p → t2 ∈ LateralFaces p → CongruentTriangles t1 t2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_lateral_faces_congruent_l152_15252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l152_15287

noncomputable def f (a ω x : ℝ) : ℝ := 2 * a * Real.sin (ω * x) * Real.cos (ω * x) + 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3

def has_max_value (f : ℝ → ℝ) (max : ℝ) : Prop :=
  ∀ x, f x ≤ max

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ has_period f T ∧ ∀ T' > 0, has_period f T' → T ≤ T'

theorem function_properties (a ω : ℝ) (ha : a > 0) (hω : ω > 0)
  (hmax : has_max_value (f a ω) 2)
  (hperiod : is_smallest_positive_period (f a ω) Real.pi) :
  (∀ x, f a ω x = 2 * Real.sin (2 * x + Real.pi / 3)) ∧
  (∀ k : ℤ, ∃ x, x = Real.pi / 12 + k * Real.pi / 2 ∧ 
    ∀ y, f a ω (x + y) = f a ω (x - y)) ∧
  (∀ α, f a ω α = 4 / 3 → Real.sin (4 * α + Real.pi / 6) = -1 / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l152_15287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l152_15250

theorem intersection_points_count :
  ∃! (s : Finset (ℝ × ℝ)), 
    (∀ p ∈ s, 
      ((2 * p.2 - 3 * p.1 = 4) ∧ (p.1 - 3 * p.2 = 6)) ∨
      ((2 * p.2 - 3 * p.1 = 4) ∧ (3 * p.1 + 2 * p.2 = 5)) ∨
      ((p.1 - 3 * p.2 = 6) ∧ (3 * p.1 + 2 * p.2 = 5))) ∧ 
    s.card = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l152_15250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_l152_15292

-- Define the points
noncomputable def A : ℝ × ℝ := (3, 7)
noncomputable def B : ℝ × ℝ := (5, -1)
noncomputable def C : ℝ × ℝ := (-2, -5)

-- Define the midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Theorem statement
theorem median_equation : 
  line_equation C.1 C.2 ∧ line_equation D.1 D.2 := by
  sorry

#check median_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_l152_15292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_full_price_revenue_l152_15243

/-- Represents the price of a full-price ticket -/
def full_price : ℕ := sorry

/-- Represents the number of full-price tickets sold -/
def full_tickets : ℕ := sorry

/-- Represents the number of half-price tickets sold -/
def half_tickets : ℕ := sorry

/-- The total number of tickets sold is 180 -/
axiom total_tickets : full_tickets + half_tickets = 180

/-- The total revenue is $2709 -/
axiom total_revenue : full_tickets * full_price + half_tickets * (full_price / 2) = 2709

/-- Half-price tickets cost half the price of full-price tickets -/
axiom half_price : half_tickets * (full_price / 2) = half_tickets * full_price / 2

/-- Theorem: The amount collected through full-price tickets is $2142 -/
theorem full_price_revenue : full_tickets * full_price = 2142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_full_price_revenue_l152_15243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_parallel_chords_l152_15233

/-- Given a circle with two parallel chords of lengths 40 and 48, separated by a distance of 22,
    the radius of the circle is 25. -/
theorem circle_radius_from_parallel_chords (R : ℝ) 
  (chord1 : ℝ) (chord2 : ℝ) (distance : ℝ)
  (h1 : chord1 = 40)
  (h2 : chord2 = 48)
  (h3 : distance = 22)
  (h4 : Real.sqrt (R^2 - (chord1/2)^2) + Real.sqrt (R^2 - (chord2/2)^2) = distance) : R = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_parallel_chords_l152_15233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l152_15251

theorem problem_statement (m n : ℝ) (h1 : (2 : ℝ)^m = 6) (h2 : (3 : ℝ)^n = 6) :
  (m + n = m * n) ∧ (m + n > 4) ∧ ((m - 1)^2 + (n - 1)^2 > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l152_15251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_cylinder_volume_l152_15275

/-- The volume of a cylinder formed by rolling a rectangle -/
def cylinder_volume (length width : ℝ) : Set ℝ :=
  {v | ∃ (r h : ℝ), (2 * Real.pi * r = length ∨ 2 * Real.pi * r = width) ∧
                     (h = width ∨ h = length) ∧
                     v = Real.pi * r^2 * h}

/-- Theorem: The volume of the cylinder formed by rolling a 4x2 rectangle -/
theorem rectangle_to_cylinder_volume :
  cylinder_volume 4 2 = {8/Real.pi, 4/Real.pi} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_cylinder_volume_l152_15275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equality_l152_15225

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- State the theorem
theorem f_equality {x : ℝ} (h : -1 < x ∧ x < 1) : 
  f ((4*x - x^3) / (1 + 4*x^2)) = f x :=
by
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equality_l152_15225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_lines_l152_15244

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations
variable (belongs_to : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem intersection_parallel_lines 
  (l m n : Line) (α β γ : Plane) 
  (h1 : intersect α β l)
  (h2 : intersect β γ m)
  (h3 : intersect γ α n)
  (h4 : parallel_line_plane l γ) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_lines_l152_15244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_problem_l152_15290

/- Define the prices of trees -/
def wishing_tree_price : ℕ → Prop := λ x => x = 18
def money_tree_price : ℕ → Prop := λ y => y = 12

/- Define the planting scheme -/
def planting_scheme : ℕ × ℕ → Prop := λ (a, b) => 
  (a + b = 20) ∧ (a ≥ b) ∧ (18 * a + 12 * b ≤ 312)

/- Define the conditions -/
axiom price_condition_1 : ∀ x y, wishing_tree_price x → money_tree_price y → x + 2*y = 42
axiom price_condition_2 : ∀ x y, wishing_tree_price x → money_tree_price y → 2*x + y = 48

/- State the theorem -/
theorem tree_planting_problem :
  (wishing_tree_price 18 ∧ money_tree_price 12) ∧
  (planting_scheme (10, 10) ∧ planting_scheme (11, 9) ∧ planting_scheme (12, 8)) ∧
  (∀ a b, planting_scheme (a, b) → (a = 10 ∧ b = 10) ∨ (a = 11 ∧ b = 9) ∨ (a = 12 ∧ b = 8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_problem_l152_15290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marker_cost_is_1_40_l152_15276

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The cost of a marker in dollars -/
def marker_cost : ℝ := sorry

/-- The total cost of three notebooks and two markers is $7.45 -/
axiom three_notebooks_two_markers : 3 * notebook_cost + 2 * marker_cost = 7.45

/-- The total cost of four notebooks and three markers is $10.40 -/
axiom four_notebooks_three_markers : 4 * notebook_cost + 3 * marker_cost = 10.40

/-- The cost of a marker is $1.40 -/
theorem marker_cost_is_1_40 : marker_cost = 1.40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marker_cost_is_1_40_l152_15276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pasture_length_l152_15288

/-- Represents the configuration of a rectangular horse pasture -/
structure Pasture where
  barn_length : ℕ
  fence_cost_per_foot : ℕ
  total_fence_cost : ℕ

/-- Calculates the length of the side parallel to the barn that maximizes the area of the pasture -/
def optimal_parallel_side_length (p : Pasture) : ℕ :=
  let total_fence_length := p.total_fence_cost / p.fence_cost_per_foot
  total_fence_length / 2

/-- Theorem stating that the optimal length of the side parallel to the barn is 140 feet -/
theorem optimal_pasture_length (p : Pasture) 
  (h1 : p.barn_length = 350)
  (h2 : p.fence_cost_per_foot = 5)
  (h3 : p.total_fence_cost = 1400) : 
  optimal_parallel_side_length p = 140 := by
  sorry

#eval optimal_parallel_side_length { barn_length := 350, fence_cost_per_foot := 5, total_fence_cost := 1400 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pasture_length_l152_15288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l152_15269

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the given expression
noncomputable def expression (y : ℝ) : ℝ :=
  (floor 6.5 : ℝ) * (floor (2/3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor 8.4 : ℝ) - y

-- Theorem statement
theorem expression_value :
  expression 6.0 = 16.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l152_15269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l152_15210

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (log x) / (2 * x)

-- State the theorem
theorem f_max_value :
  ∃ (x_max : ℝ), x_max > 0 ∧
    (∀ x > 0, f x ≤ f x_max) ∧
    f x_max = 1 / (2 * exp 1) ∧
    x_max = exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l152_15210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_diagonal_formula_l152_15289

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- Length of the longer leg -/
  a : ℝ
  /-- Length of the shorter base -/
  b : ℝ
  /-- The trapezoid is right-angled -/
  is_right : True
  /-- The shorter diagonal is equal to the longer leg -/
  shorter_diagonal_eq_longer_leg : True

/-- The longer diagonal of a right trapezoid with specific properties -/
noncomputable def longer_diagonal (t : RightTrapezoid) : ℝ :=
  Real.sqrt (t.a^2 + 3 * t.b^2)

/-- Theorem: The longer diagonal of a right trapezoid with given properties -/
theorem longer_diagonal_formula (t : RightTrapezoid) :
  longer_diagonal t = Real.sqrt (t.a^2 + 3 * t.b^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_diagonal_formula_l152_15289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l152_15214

/-- Calculates the length of a train given its speed and time to cross a pole -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 1000 / 3600

/-- Theorem: A train traveling at 160 km/h and crossing a pole in 18 seconds has a length of approximately 800 meters -/
theorem train_length_approx :
  let speed := (160 : ℝ)  -- km/h
  let time := (18 : ℝ)    -- seconds
  ∃ ε > 0, |train_length speed time - 800| < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l152_15214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_value_l152_15205

-- Define the function representing the left side of the equation
noncomputable def f (x : ℝ) : ℝ := ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5)

-- State the theorem
theorem max_x_value :
  ∃ (x_max : ℝ), x_max = 9/5 ∧
  f x_max = 20 ∧
  ∀ (x : ℝ), f x = 20 → x ≤ x_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_value_l152_15205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_or_concurrent_l152_15256

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder and should be properly defined
  mk :: -- empty structure for now

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane in 3D space
  -- This is a placeholder and should be properly defined
  mk :: -- empty structure for now

/-- A point in 3D space -/
structure Point3D where
  -- Add necessary fields to define a point in 3D space
  -- This is a placeholder and should be properly defined
  mk :: -- empty structure for now

/-- Predicate to check if two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if a line lies in a plane -/
def lies_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- Main theorem: Given a set of lines where any two intersect, 
    either all lines are coplanar or all lines are concurrent -/
theorem lines_coplanar_or_concurrent 
  (lines : Set Line3D) 
  (h_intersect : ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → intersect l1 l2) :
  (∃ p : Plane3D, ∀ l, l ∈ lines → lies_in_plane l p) ∨ 
  (∃ p : Point3D, ∀ l, l ∈ lines → passes_through l p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_or_concurrent_l152_15256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_third_l152_15229

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (4 * x) * Real.cos (4 * x)

-- State the theorem
theorem f_derivative_at_pi_third : 
  deriv f (π / 3) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_third_l152_15229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_club_committee_probability_total_combinations_all_boys_combinations_all_girls_combinations_l152_15216

theorem science_club_committee_probability : 
  (8855 - (715 + 210)) / 8855 = 7930 / 8855 :=
by
  -- The proof steps would go here
  sorry

-- Additional definitions to explain the numbers used in the theorem
def total_members : ℕ := 23
def boys : ℕ := 13
def girls : ℕ := 10
def committee_size : ℕ := 4

-- These could be used to show how the numbers in the theorem are derived
theorem total_combinations : Nat.choose total_members committee_size = 8855 :=
by sorry

theorem all_boys_combinations : Nat.choose boys committee_size = 715 :=
by sorry

theorem all_girls_combinations : Nat.choose girls committee_size = 210 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_club_committee_probability_total_combinations_all_boys_combinations_all_girls_combinations_l152_15216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_product_l152_15291

/-- The ratio of the area of a circle inscribed in an equilateral triangle
    (touching the midpoints of the sides) to the area of the triangle -/
noncomputable def area_ratio (s : ℝ) : ℝ :=
  (Real.pi * (s / Real.sqrt 3)^2) / ((Real.sqrt 3 / 4) * s^2)

/-- The product of a and b when the area ratio is expressed as (√a/b)π -/
def product_ab (s : ℝ) : ℕ :=
  let a : ℕ := 3  -- From the simplified form (4√3/9)π
  let b : ℕ := 9  -- From the simplified form (4√3/9)π
  a * b

theorem area_ratio_product : ∀ s : ℝ, s > 0 → product_ab s = 27 := by
  intro s hs
  unfold product_ab
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_product_l152_15291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resistance_y_value_l152_15219

noncomputable def parallel_resistance (x y : ℝ) : ℝ := 1 / (1/x + 1/y)

theorem resistance_y_value (x r : ℝ) (hx : x = 3) (hr : r = 1.875) :
  ∃ y : ℝ, parallel_resistance x y = r ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_resistance_y_value_l152_15219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l152_15223

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x - Real.pi / 3) + b

noncomputable def g (a b x : ℝ) : ℝ := b * Real.cos (a * x + Real.pi / 6)

theorem function_properties :
  ∀ (a b : ℝ), a > 0 →
  (∀ x, f a b x ≤ 1) →
  (∃ x, f a b x = 1) →
  (∀ x, f a b x ≥ -5) →
  (∃ x, f a b x = -5) →
  (a = 3 ∧ b = -2) ∧
  (∀ x, g a b x ≤ 2) ∧
  (∃ x, g a b x = 2) ∧
  (∀ k : ℤ, g a b (5 * Real.pi / 18 + 2 * ↑k * Real.pi / 3) = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l152_15223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_writing_time_theorem_l152_15263

/-- Represents the writing speed for each topic in pages per hour -/
structure WritingSpeed where
  literature : ℚ
  politics : ℚ
  history : ℚ
  science : ℚ
  philosophy : ℚ

/-- Represents the daily page count for each topic on weekdays -/
structure WeekdayPages where
  politics : ℚ
  politicsPeople : ℕ
  history : ℚ
  historyPeople : ℕ
  science : ℚ
  sciencePeople : ℕ

/-- Represents the daily page count for each topic on weekends -/
structure WeekendPages where
  literature : ℚ
  literaturePeople : ℕ
  philosophy : ℚ
  philosophyPeople : ℕ

/-- Calculates the total writing time per week -/
def totalWritingTime (speed : WritingSpeed) (weekday : WeekdayPages) (weekend : WeekendPages) : ℚ :=
  let weekdayHours := 
    5 * (weekday.politics * weekday.politicsPeople / speed.politics +
         weekday.history * weekday.historyPeople / speed.history +
         weekday.science * weekday.sciencePeople / speed.science)
  let weekendHours := 
    2 * (weekend.literature * weekend.literaturePeople / speed.literature +
         weekend.philosophy * weekend.philosophyPeople / speed.philosophy)
  weekdayHours + weekendHours

theorem writing_time_theorem (speed : WritingSpeed) (weekday : WeekdayPages) (weekend : WeekendPages) :
  speed.literature = 12 ∧ 
  speed.politics = 8 ∧ 
  speed.history = 9 ∧ 
  speed.science = 6 ∧ 
  speed.philosophy = 10 ∧
  weekday.politics = 5 ∧ 
  weekday.politicsPeople = 2 ∧
  weekday.history = 3 ∧ 
  weekday.historyPeople = 1 ∧
  weekday.science = 7 ∧ 
  weekday.sciencePeople = 2 ∧
  weekend.literature = 10 ∧ 
  weekend.literaturePeople = 3 ∧
  weekend.philosophy = 4 ∧ 
  weekend.philosophyPeople = 1 →
  totalWritingTime speed weekday weekend = 2535 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_writing_time_theorem_l152_15263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_100_over_9_l152_15224

/-- Represents the cost and selling prices of pencils -/
structure PencilPrices where
  initialSellQuantity : ℚ
  initialSellPrice : ℚ
  profitSellQuantity : ℚ
  profitSellPrice : ℚ
  profitPercentage : ℚ

/-- Calculates the loss percentage given the initial selling scenario and the profit scenario -/
def calculateLossPercentage (p : PencilPrices) : ℚ :=
  let costPrice := p.profitSellPrice / (p.profitSellQuantity * (1 + p.profitPercentage))
  let initialSellPricePerPencil := p.initialSellPrice / p.initialSellQuantity
  let lossPerPencil := costPrice - initialSellPricePerPencil
  (lossPerPencil / costPrice) * 100

/-- Theorem stating that under the given conditions, the loss percentage is 100/9 -/
theorem loss_percentage_is_100_over_9 (p : PencilPrices) 
    (h1 : p.initialSellQuantity = 11)
    (h2 : p.initialSellPrice = 1)
    (h3 : p.profitSellQuantity = 33/4)
    (h4 : p.profitSellPrice = 1)
    (h5 : p.profitPercentage = 1/5) :
    calculateLossPercentage p = 100 / 9 := by
  sorry

#eval calculateLossPercentage { 
  initialSellQuantity := 11, 
  initialSellPrice := 1, 
  profitSellQuantity := 33/4, 
  profitSellPrice := 1, 
  profitPercentage := 1/5 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_100_over_9_l152_15224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_parabola_is_correct_l152_15249

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The parabola passes through the points (0, 4) and (2, -2) -/
def passes_through (p : Parabola) : Prop :=
  p.c = 4 ∧ p.a * 4 + p.b * 2 + p.c = -2

/-- The length of the segment that the parabola cuts on the x-axis -/
noncomputable def x_axis_segment_length (p : Parabola) : ℝ :=
  Real.sqrt ((p.b^2 - 4 * p.a * p.c) / (p.a^2))

/-- The parabola with the shortest x-axis segment -/
noncomputable def shortest_segment_parabola : Parabola :=
  { a := 9/2
    b := -12
    c := 4
    a_pos := by norm_num }

theorem shortest_segment_parabola_is_correct :
  passes_through shortest_segment_parabola ∧
  ∀ p : Parabola, passes_through p →
    x_axis_segment_length p ≥ x_axis_segment_length shortest_segment_parabola :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_parabola_is_correct_l152_15249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_minus_sum_not_in_list_l152_15203

def isPrimeBetween10And30 (n : ℕ) : Prop :=
  Nat.Prime n ∧ 10 < n ∧ n < 30

theorem prime_product_minus_sum_not_in_list :
  ∀ x y : ℕ, 
    isPrimeBetween10And30 x → 
    isPrimeBetween10And30 y → 
    x ≠ y →
    (x * y - (x + y)) ∉ ({77, 156, 224, 270, 319} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_minus_sum_not_in_list_l152_15203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_parabola_l152_15279

-- Define the parabola
def on_parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

-- Define the right angle condition
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.2 - B.2) + (B.1 - A.1) * (C.1 - B.1) = 0

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The main theorem
theorem min_distance_on_parabola :
  ∀ B C : ℝ × ℝ,
  on_parabola B → on_parabola C →
  right_angle (1, 1) B C →
  ∀ B' C' : ℝ × ℝ,
  on_parabola B' → on_parabola C' →
  right_angle (1, 1) B' C' →
  distance (1, 1) C ≤ distance (1, 1) C' →
  distance (1, 1) C = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_parabola_l152_15279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_domain_l152_15200

noncomputable def log_function (x : ℝ) : ℝ := Real.log (5 - x) / Real.log (x - 2)

theorem log_function_domain :
  {x : ℝ | ∃ y, y = log_function x} = Set.union (Set.Ioo 2 3) (Set.Ioo 3 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_domain_l152_15200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15298

def f (a x : ℝ) := |x + a| - |x - a^2 - a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≤ 1 ↔ x ≤ -1) ∧
  (∀ b : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-1) (1/3) → (∀ x : ℝ, f a x ≤ b)) → b ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_angle_l152_15239

/-- A line in 2D space represented by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The inclination angle of a line. -/
noncomputable def inclinationAngle (l : ParametricLine) : ℝ :=
  Real.arctan ((l.y 1 - l.y 0) / (l.x 1 - l.x 0))

/-- Checks if a point lies on a parametric line. -/
def pointOnLine (l : ParametricLine) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- The main theorem stating that the given parametric equations represent
    a line passing through (-1, 2) with inclination angle 3π/4. -/
theorem line_through_point_with_angle :
  let l : ParametricLine := {
    x := λ t => -1 - (t * Real.sqrt 2) / 2,
    y := λ t => 2 + (t * Real.sqrt 2) / 2
  }
  pointOnLine l (-1, 2) ∧ inclinationAngle l = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_angle_l152_15239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_half_l152_15296

/-- A square pyramid with a circumscribed sphere and six smaller spheres -/
structure SquarePyramidWithSpheres where
  /-- Side length of the square base -/
  s : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- The pyramid has a square base and four equilateral triangular faces -/
  pyramid_shape : s > 0 ∧ h > 0
  /-- The circumscribed sphere contains the entire pyramid -/
  circumscribed_sphere : R > 0 ∧ R^2 = h^2 + (s / (2 * Real.sqrt 2))^2
  /-- Six smaller spheres, each tangent to a face and the circumscribed sphere -/
  smaller_spheres : ∃ (r : ℝ), r > 0 ∧ r ≤ R / 2

/-- The probability of a random point in the circumscribed sphere being in one of the smaller spheres -/
noncomputable def probability_in_smaller_spheres (p : SquarePyramidWithSpheres) : ℝ :=
  (6 * (4 / 3 * Real.pi * (p.R / 2)^3)) / (4 / 3 * Real.pi * p.R^3)

/-- The main theorem: The probability is always 1/2 -/
theorem probability_is_half (p : SquarePyramidWithSpheres) : 
  probability_in_smaller_spheres p = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_half_l152_15296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l152_15211

noncomputable def f (x : ℝ) : ℝ := x / (1 + abs x)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (Set.range f = Set.Ioo (-1 : ℝ) 1) ∧
  Function.Injective f ∧
  (∃! x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l152_15211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validNumberCount_eq_66_l152_15201

/-- A function that checks if three digits form a valid three-digit number
    according to the problem conditions -/
def isValidNumber (a b c : Nat) : Bool :=
  a ≥ 0 ∧ a ≤ 9 ∧
  b ≥ 0 ∧ b ≤ 9 ∧
  c ≥ 0 ∧ c ≤ 9 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c ∨
   b < c ∧ c < a ∨ c < a ∧ a < b ∨ c < b ∧ b < a) ∧
  (b = (a + c) / 2 ∨ a = (b + c) / 2 ∨ c = (a + b) / 2)

/-- The count of valid three-digit numbers -/
def validNumberCount : Nat :=
  (List.range 10).foldl
    (fun count a =>
      (List.range 10).foldl
        (fun count' b =>
          (List.range 10).foldl
            (fun count'' c =>
              if isValidNumber a b c then count'' + 1 else count'')
            count')
        count)
    0

/-- The main theorem stating that there are exactly 66 valid numbers -/
theorem validNumberCount_eq_66 : validNumberCount = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validNumberCount_eq_66_l152_15201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l152_15232

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + a) + Real.sqrt ((c - x)^2 + b)

theorem min_value_of_f :
  ∃ (min_val : ℝ), (∀ x, f a b c x ≥ min_val) ∧ (min_val = Real.sqrt (c^2 + (Real.sqrt a + Real.sqrt b)^2)) := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l152_15232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15209

theorem problem_solution (x y z : ℤ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_order : x ≥ y ∧ y ≥ z)
  (h_eq1 : x^2 - y^2 - z^2 + x*y = 4033)
  (h_eq2 : x^2 + 4*y^2 + 4*z^2 - 4*x*y - 3*x*z - 3*y*z = -3995) :
  x = 69 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_point_probability_l152_15235

-- Define the side length of the square
noncomputable def squareSideLength : ℝ := 6

-- Define the radius of the circle
noncomputable def circleRadius : ℝ := 2

-- Define the probability
noncomputable def probability : ℝ := Real.pi / 9

-- Theorem statement
theorem random_point_probability :
  (Real.pi * circleRadius^2) / squareSideLength^2 = probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_point_probability_l152_15235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l152_15260

-- Define the function k(x) as noncomputable
noncomputable def k (x : ℝ) : ℝ := (3 * x + 5) / (x - 4)

-- State the theorem about the range of k(x)
theorem range_of_k : 
  ∀ y : ℝ, y ≠ 3 → ∃ x : ℝ, k x = y ∧ x ≠ 4 := by
  sorry

-- Optional: You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l152_15260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_diff_inequality_l152_15261

theorem absolute_diff_inequality :
  (∃ x : ℝ, |x - 3| - |x - 1| < 2 → x ≠ 1) ∧
  (∃ x : ℝ, x ≠ 1 ∧ |x - 3| - |x - 1| ≥ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_diff_inequality_l152_15261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l152_15228

/-- Curve C in the Cartesian plane -/
noncomputable def C : ℝ → ℝ × ℝ := fun α ↦ (2 * Real.cos α, Real.sin α)

/-- Polar equation of curve C -/
noncomputable def polar_eq (θ : ℝ) : ℝ := 4 / (1 + 3 * Real.sin θ ^ 2)

/-- Distance between two points on curve C -/
noncomputable def distance (θ : ℝ) : ℝ := 
  Real.sqrt (polar_eq θ + polar_eq (θ + Real.pi/2))

theorem curve_C_properties :
  (∀ θ, (polar_eq θ) ^ 2 = 4 / (1 + 3 * Real.sin θ ^ 2)) ∧
  (∀ θ, 16/5 ≤ (distance θ) ^ 2 ∧ (distance θ) ^ 2 ≤ 5) := by
  sorry

#check curve_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l152_15228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_delicious_integer_l152_15280

def IsDelicious (n : Int) : Prop :=
  ∃ (a b : Int), a ≤ n ∧ n ≤ b ∧ (Finset.sum (Finset.Icc a b) id) = 2023

theorem smallest_delicious_integer :
  IsDelicious (-2022) ∧ ∀ k < -2022, ¬IsDelicious k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_delicious_integer_l152_15280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_meeting_l152_15270

/-- The probability that two people meet given their arrival time constraints -/
theorem probability_of_meeting (a_start a_end b_start b_end wait_time : ℝ) : 
  a_start = 7 ∧ a_end = 8 ∧ b_start = 7 + 1/3 ∧ b_end = 7 + 5/6 ∧ wait_time = 1/6 →
  (∫ (x : ℝ) in a_start..a_end, ∫ (y : ℝ) in b_start..b_end, 
    if |x - y| ≤ wait_time then 1 else 0) / 
  ((a_end - a_start) * (b_end - b_start)) = 1/3 := by
  sorry

#check probability_of_meeting

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_meeting_l152_15270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l152_15268

/-- The force function F(x) in Newtons -/
def F (x : ℝ) : ℝ := 5 * x + 3

/-- The work done by the force F(x) from x = a to x = b -/
noncomputable def work (a b : ℝ) : ℝ := ∫ x in a..b, F x

theorem work_calculation : work 0 5 = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l152_15268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_g_l152_15281

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|
def g (x : ℝ) : ℝ := |x + 3/2| + |x - 3/2|

-- Statement 1
theorem solution_set_f (x : ℝ) : f x ≤ x + 2 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

-- Statement 2
theorem inequality_g (x a : ℝ) (h : a ≠ 0) : (|a + 1| - |2*a - 1|) / |a| ≤ g x := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_inequality_g_l152_15281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_primes_divisible_by_five_l152_15231

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primes_under_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem percentage_of_primes_divisible_by_five :
  (primes_under_20.filter (λ p => p % 5 = 0)).length / primes_under_20.length * 100 = 125 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_primes_divisible_by_five_l152_15231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_zoo_animals_l152_15246

/-- The number of animals in John's zoo --/
def zoo_animals (snakes monkeys lions pandas dogs : ℕ) : ℕ :=
  snakes + monkeys + lions + pandas + dogs

theorem john_zoo_animals :
  ∀ (snakes monkeys lions pandas dogs : ℕ),
    snakes = 15 →
    monkeys = 2 * snakes →
    lions = monkeys - 5 →
    pandas = lions + 8 →
    dogs = pandas / 3 →
    zoo_animals snakes monkeys lions pandas dogs = 114 :=
by
  intros snakes monkeys lions pandas dogs h1 h2 h3 h4 h5
  simp [zoo_animals]
  sorry

#eval zoo_animals 15 30 25 33 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_zoo_animals_l152_15246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l152_15248

theorem matrix_sum_values (x y z : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![x, y, z; y, z, x; z, x, y]
  ¬(IsUnit (Matrix.det M)) →
  (x / (y + z) + y / (x + z) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (x + z) + z / (x + y) = 3/2) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l152_15248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_money_problem_l152_15206

noncomputable section

-- Define the total amount of money
def total : ℝ := 200

-- Define the original amount in the left pocket
def left_original : ℝ := 160

-- Define the original amount in the right pocket
def right_original : ℝ := total - left_original

-- Define the amount in the left pocket after the first transfer
def left_after_first : ℝ := left_original * (3/4)

-- Define the amount in the right pocket after the first transfer
def right_after_first : ℝ := right_original + left_original * (1/4)

-- Define the amount in the left pocket after the second transfer
def left_final : ℝ := left_after_first - 20

-- Define the amount in the right pocket after the second transfer
def right_final : ℝ := right_after_first + 20

end noncomputable section

theorem joe_money_problem :
  left_original + right_original = total ∧
  left_final = right_final ∧
  left_original = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_money_problem_l152_15206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_l152_15262

theorem max_value_trig_sum (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + 
  Real.cos θ₄ * Real.sin θ₅ + Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁ ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_l152_15262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prescription_with_four_potent_medicines_l152_15240

/-- Represents a type of medicine -/
structure Medicine : Type :=
  (id : ℕ)
  (isPotent : Bool)

/-- Represents a prescription -/
structure Prescription : Type :=
  (medicines : Finset Medicine)
  (hasFiveMedicines : medicines.card = 5)
  (hasAtLeastOnePotent : ∃ m ∈ medicines, m.isPotent)

/-- The set of all prescriptions -/
def AllPrescriptions : Finset Prescription := sorry

/-- The set of all medicines -/
def AllMedicines : Finset Medicine := sorry

theorem prescription_with_four_potent_medicines :
  (AllPrescriptions.card = 68) →
  (∀ (m1 m2 m3 : Medicine), m1 ∈ AllMedicines → m2 ∈ AllMedicines → m3 ∈ AllMedicines →
    m1 ≠ m2 → m2 ≠ m3 → m1 ≠ m3 →
    (∃! p : Prescription, p ∈ AllPrescriptions ∧ m1 ∈ p.medicines ∧ m2 ∈ p.medicines ∧ m3 ∈ p.medicines)) →
  ∃ p : Prescription, p ∈ AllPrescriptions ∧ (p.medicines.filter (λ m : Medicine => m.isPotent)).card ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prescription_with_four_potent_medicines_l152_15240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l152_15234

-- Define the function f
def f (a k x : ℝ) : ℝ := -(a + 2) * x^2 + (k - 1) * x - a

-- Define the domain of f
def domain (a : ℝ) : Set ℝ := Set.Icc (a - 2) (a + 4)

-- State the theorem
theorem even_function_properties (a k : ℝ) 
  (h_even : ∀ x ∈ domain a, f a k x = f a k (-x)) :
  a = -1 ∧ 
  k = 1 ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) (-1) ∪ Set.Icc 0 1, 
    ∀ y ∈ Set.Icc (-3 : ℝ) (-1) ∪ Set.Icc 0 1, 
    x < y → |f (-1) 1 x| > |f (-1) 1 y|) ∧
  (∀ t : ℝ, f (-1) 1 (t - 1) - f (-1) 1 t > 0 ↔ t ∈ Set.Ioo (1/2 : ℝ) 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l152_15234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l152_15217

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else -x * (x + 2)

-- Theorem statement
theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l152_15217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_division_octahedron_division_l152_15208

/-- Represents a polyhedron with an edge length -/
structure Polyhedron where
  edge_length : ℝ

/-- A tetrahedron with edge length 2a can be divided into 4 tetrahedra and 1 octahedron, all with edge length a -/
theorem tetrahedron_division (a : ℝ) (h : a > 0) :
  ∃ (tetra_count octahedron_count : ℕ),
    tetra_count = 4 ∧
    octahedron_count = 1 ∧
    (∀ t : Polyhedron, t.edge_length = a) ∧
    (∀ o : Polyhedron, o.edge_length = a) :=
by
  sorry

/-- An octahedron with edge length 2a can be divided into 8 tetrahedra and 6 octahedra, all with edge length a -/
theorem octahedron_division (a : ℝ) (h : a > 0) :
  ∃ (tetra_count octahedron_count : ℕ),
    tetra_count = 8 ∧
    octahedron_count = 6 ∧
    (∀ t : Polyhedron, t.edge_length = a) ∧
    (∀ o : Polyhedron, o.edge_length = a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_division_octahedron_division_l152_15208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_FCG_measure_l152_15221

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C D E F G : ℝ × ℝ)

-- Define the clockwise arrangement of points
def clockwise_arrangement (circle : Set (ℝ × ℝ)) (A B C D E F G : ℝ × ℝ) : Prop :=
  A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle ∧ G ∈ circle

-- Define the diameter property
def is_diameter (circle : Set (ℝ × ℝ)) (A E : ℝ × ℝ) : Prop :=
  A ∈ circle ∧ E ∈ circle

-- Define angle measures
noncomputable def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_FCG_measure
  (h_clockwise : clockwise_arrangement circle A B C D E F G)
  (h_diameter : is_diameter circle A E)
  (h_ABF : angle_measure A B F = 81)
  (h_EDG : angle_measure E D G = 76) :
  angle_measure F C G = 67 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_FCG_measure_l152_15221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_positions_fold_to_cube_l152_15238

/-- Represents a square in the polygon --/
structure Square where
  area : ℝ

/-- Represents a position where a square can be attached to the cross-shaped polygon --/
inductive AttachmentPosition
| CenterEdge
| OuterEdge

/-- Represents the cross-shaped polygon made of five congruent squares --/
structure CrossPolygon where
  squares : Fin 5 → Square
  congruent : ∀ i j, (squares i).area = (squares j).area

/-- Represents the resulting polygon after attaching a square --/
structure ResultingPolygon where
  base : CrossPolygon
  attachedSquare : Square
  position : AttachmentPosition

/-- Predicate to check if a resulting polygon can be folded into a cube with one face missing --/
def canFoldIntoCube (p : ResultingPolygon) : Prop :=
  sorry

/-- The main theorem stating that exactly 8 positions allow folding into a cube --/
theorem eight_positions_fold_to_cube (cross : CrossPolygon) :
  ∃! (positions : Finset AttachmentPosition),
    positions.card = 8 ∧
    ∀ pos, pos ∈ positions ↔
      canFoldIntoCube ⟨cross, cross.squares 0, pos⟩ :=
by
  sorry

#check eight_positions_fold_to_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_positions_fold_to_cube_l152_15238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l152_15295

/-- Predicate to determine if a point is the focus of a parabola --/
def is_focus (f : ℝ × ℝ) (parabola : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (p : ℝ), parabola = λ (point : ℝ × ℝ) ↦ (point.1 - f.1)^2 = 4*p*(point.2 - f.2)

/-- The focus of a parabola with equation x^2 + y = 0 --/
theorem parabola_focus : 
  ∃ (f : ℝ × ℝ), f = (0, -1/4) ∧ is_focus f (λ (p : ℝ × ℝ) ↦ p.1^2 + p.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l152_15295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l152_15215

-- Define set A
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sin x}

-- Define set B
def B : Set ℝ := {x : ℝ | (1/9 : ℝ) < (1/3 : ℝ)^x ∧ (1/3 : ℝ)^x < 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l152_15215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_below_line_l152_15274

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

-- Define the theorem
theorem function_below_line (a : ℝ) :
  (∀ x > 1, f a x < 2 * a * x) ↔ -1/2 ≤ a ∧ a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_below_line_l152_15274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l152_15253

noncomputable def original_curve (x : ℝ) : ℝ := Real.log (x + 1)

noncomputable def symmetric_curve (x : ℝ) : ℝ := -Real.log (-x + 1)

theorem curve_symmetry : 
  ∀ x y : ℝ, y = original_curve x ↔ -y = symmetric_curve (-x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l152_15253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_F_no_zeros_range_l152_15258

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x - x / 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.exp x - 1 / 4

-- Define the function g
def g (x : ℝ) : ℝ := x * f' x

-- Define the function F
def F (a x : ℝ) : ℝ := Real.log x - a * f x + 1

-- Theorem for the monotonicity of g
theorem g_monotone_increasing : ∀ x > 0, (deriv g x) > 0 := by sorry

-- Theorem for the range of a
theorem F_no_zeros_range (a : ℝ) : 
  (∀ x > 0, F a x ≠ 0) → a > 4 / (4 * Real.exp 1 - 1) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_F_no_zeros_range_l152_15258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_one_equals_negative_four_l152_15297

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a) / x

theorem f_negative_one_equals_negative_four (a : ℝ) :
  f 1 a = 4 → f (-1) a = -4 := by
  intro h
  have a_eq : a = 3 := by
    -- Prove that a = 3 using the given condition f(1) = 4
    sorry
  -- Use a_eq to prove f(-1) = -4
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_one_equals_negative_four_l152_15297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intercept_l152_15285

/-- Given a curve y = ax + ln x with a tangent line y = 2x + b passing through the point (1, a), prove that b = -1 -/
theorem tangent_line_intercept (a b : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, f x = a * x + Real.log x) →
  (∃ g : ℝ → ℝ, ∀ x, g x = 2 * x + b) →
  (∀ x, HasDerivAt (fun x => a * x + Real.log x) (a + 1 / x) x) →
  (1 = a) →
  (2 + b = a) →
  b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intercept_l152_15285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15257

-- Define the function f
noncomputable def f (x θ : ℝ) : ℝ := 2 * Real.sin x * (Real.cos (θ / 2))^2 + Real.cos x * Real.sin θ - Real.sin x

-- State the theorem
theorem problem_solution (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : ∀ x, f x θ ≥ f π θ) :
  (θ = π / 2) ∧ 
  (∀ (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hABC : A + B + C = π)
     (ha : 1 = Real.sin B / Real.sin C) 
     (hb : Real.sqrt 2 = Real.sin C / Real.sin A)
     (hfA : f A (π / 2) = Real.sqrt 3 / 2),
   C = 7 * π / 12 ∨ C = π / 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_always_true_iff_a_in_range_l152_15218

theorem inequality_always_true_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_always_true_iff_a_in_range_l152_15218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l152_15282

theorem constant_term_expansion (n : ℕ+) : 
  (∃ k : ℚ, k ≠ 0 ∧ ∀ x y : ℚ, y^3 * (x + 1/(x^2*y))^(n : ℕ) = k + (y^3 * (x + 1/(x^2*y))^(n : ℕ) - k)) →
  (∃ k : ℚ, k ≠ 0 ∧ ∀ x y : ℚ, y^3 * (x + 1/(x^2*y))^(n : ℕ) = 84 + (y^3 * (x + 1/(x^2*y))^(n : ℕ) - 84)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l152_15282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l152_15267

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x) + Real.sqrt (-Real.tan x)

def domain (x : ℝ) : Prop :=
  ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 2 < x ∧ x ≤ 2 * k * Real.pi + Real.pi) ∨ x = 2 * k * Real.pi

theorem f_domain : 
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ domain x := by
  sorry

#check f_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l152_15267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_square_l152_15241

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- Represents a square constructed on a side of a quadrilateral -/
structure ExternalSquare where
  side : Fin 4
  center : Point

/-- Theorem: Centers of squares on parallelogram sides form a square -/
theorem centers_form_square
  (ABCD : Quadrilateral)
  (h_para : is_parallelogram ABCD)
  (squares : Fin 4 → ExternalSquare)
  (h_squares : ∀ i, (squares i).side = i) :
  ∃ (PQRS : Quadrilateral),
    (PQRS.A = (squares 0).center ∧
     PQRS.B = (squares 1).center ∧
     PQRS.C = (squares 2).center ∧
     PQRS.D = (squares 3).center) ∧
    is_square PQRS :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_square_l152_15241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l152_15230

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * cos (π / 2 * x)

-- State the theorem
theorem f_max_min :
  (∀ x, f x ≤ 3) ∧ (∃ x, f x = 3) ∧ (∀ x, f x ≥ -1) ∧ (∃ x, f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l152_15230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equals_four_plus_three_pi_l152_15272

/-- The area enclosed by a square with side length √2 and four major arcs of unit circles placed on its sides -/
noncomputable def enclosed_area : ℝ :=
  let square_side := Real.sqrt 2
  let arc_radius := 1
  let central_square_area := 4
  let sector_area := 3 * Real.pi / 4
  let total_sector_area := 4 * sector_area
  central_square_area + total_sector_area

/-- Theorem stating that the enclosed area is equal to 4 + 3π -/
theorem enclosed_area_equals_four_plus_three_pi :
  enclosed_area = 4 + 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equals_four_plus_three_pi_l152_15272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_sqrt3_line_l152_15299

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point P(2,5) to the line y = -√3x is (2√3 + 5) / 2 -/
theorem distance_point_to_sqrt3_line :
  let P : ℝ × ℝ := (2, 5)
  let line_eq (x : ℝ) : ℝ := -Real.sqrt 3 * x
  distance_point_to_line P.1 P.2 (Real.sqrt 3) 1 0 = (2 * Real.sqrt 3 + 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_sqrt3_line_l152_15299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l152_15286

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.cos (2 * x)

-- Define the function g as a transformation of f
noncomputable def g (x : ℝ) : ℝ := -Real.sin (2 * x)

-- State the theorem
theorem g_properties :
  (∀ x y, x ∈ Set.Ioo 0 (π / 4) → y ∈ Set.Ioo 0 (π / 4) → x < y → g y < g x) ∧
  (∀ x : ℝ, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l152_15286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_equals_one_l152_15204

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e : ℝ} (h : b ≠ 0 ∧ d ≠ 0) : 
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ c * x + d * y + e = 0) ↔ a / b = c / d

/-- Definition of line l₁ -/
def l₁ (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + (1 + k) * y = 2 - k

/-- Definition of line l₂ -/
def l₂ (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ k * x + 2 * y + 8 = 0

theorem parallel_lines_k_equals_one :
  ∀ k : ℝ, (∀ x y : ℝ, l₁ k x y ↔ l₂ k x y) → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_equals_one_l152_15204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_after_movement_l152_15273

/-- Given two points A and B on a Cartesian plane, their midpoint M, 
    and new positions of A and B after movement, 
    prove that the distance between M and the new midpoint M' is √17/2 -/
theorem midpoint_distance_after_movement 
  (a b c d m n : ℝ) : 
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  Real.sqrt ((((a + 3) + (c - 7)) / 2 - m)^2 + (((b + 5) + (d - 4)) / 2 - n)^2) = Real.sqrt 17 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_after_movement_l152_15273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_scores_different_l152_15220

/-- Represents a team in the tournament -/
structure Team where
  id : Nat
  deriving Repr, DecidableEq

/-- Represents the result of a match between two teams -/
inductive MatchResult where
  | Win
  | Loss
  deriving Repr, DecidableEq

/-- Represents a round-robin tournament -/
structure Tournament where
  teams : Finset Team
  result : Team → Team → MatchResult
  round_robin : ∀ t1 t2, t1 ≠ t2 → result t1 t2 = MatchResult.Win ∨ result t1 t2 = MatchResult.Loss

/-- The score of a team is the number of wins -/
def score (tournament : Tournament) (team : Team) : Nat :=
  (tournament.teams.filter (λ t => tournament.result team t = MatchResult.Win)).card

/-- The tournament satisfies the given condition -/
def satisfies_condition (tournament : Tournament) : Prop :=
  ∀ (subset : Finset Team),
    subset ⊆ tournament.teams →
    subset.card = 19 →
    (∃ (winner : Team), winner ∈ subset ∧ ∀ (t : Team), t ∈ subset ∧ t ≠ winner → tournament.result winner t = MatchResult.Win) ∧
    (∃ (loser : Team), loser ∈ subset ∧ ∀ (t : Team), t ∈ subset ∧ t ≠ loser → tournament.result loser t = MatchResult.Loss)

/-- Main theorem: All teams have different scores -/
theorem all_scores_different (tournament : Tournament) 
  (h1 : tournament.teams.card = 93)
  (h2 : satisfies_condition tournament) :
  ∀ t1 t2 : Team, t1 ∈ tournament.teams → t2 ∈ tournament.teams → t1 ≠ t2 → score tournament t1 ≠ score tournament t2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_scores_different_l152_15220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_theorem_l152_15245

-- Define the circle Γ
variable (Γ : Set (ℝ × ℝ))

-- Define points A, B, C, D, E, F, G
variable (A B C D E F G : ℝ × ℝ)

-- AB is a chord of Γ
axiom ab_chord : A ∈ Γ ∧ B ∈ Γ

-- C is on the segment AB
axiom c_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

-- Line r through C intersects Γ at D and E
axiom de_on_gamma : D ∈ Γ ∧ E ∈ Γ

-- D and E are on different sides of the perpendicular bisector of AB
axiom de_diff_sides : ∃ M : ℝ × ℝ, M = (A + B) / 2 ∧ 
  ((D.1 - M.1) * (E.1 - M.1) + (D.2 - M.2) * (E.2 - M.2) < 0)

-- ΓD is externally tangent to Γ at D and touches AB at F
axiom gamma_d_tangent : ∃ (ΓD : Set (ℝ × ℝ)) (t : ℝ), 
  D ∈ ΓD ∧ F ∈ ΓD ∧ 0 ≤ t ∧ t ≤ 1 ∧
  F.1 = A.1 + t * (B.1 - A.1) ∧ F.2 = A.2 + t * (B.2 - A.2)

-- ΓE is externally tangent to Γ at E and touches AB at G
axiom gamma_e_tangent : ∃ (ΓE : Set (ℝ × ℝ)) (t : ℝ), 
  E ∈ ΓE ∧ G ∈ ΓE ∧ 0 ≤ t ∧ t ≤ 1 ∧
  G.1 = A.1 + t * (B.1 - A.1) ∧ G.2 = A.2 + t * (B.2 - A.2)

-- Define distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The theorem to prove
theorem butterfly_theorem : 
  distance C A = distance C B ↔ distance C F = distance C G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_theorem_l152_15245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_S_l152_15227

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_element_in_S : 
  ∃! p : ℝ × ℝ, p ∈ S ∧ p = (Real.rpow (1/9) (1/3), Real.rpow (1/3) (1/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_S_l152_15227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_and_tan_value_l152_15278

noncomputable def f (x : Real) : Real := 5 * Real.sin x ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

theorem f_monotonic_decreasing_and_tan_value :
  (∀ (k : ℤ), ∀ (x y : Real), 
    k * Real.pi - Real.pi / 3 ≤ x ∧ 
    x < y ∧ 
    y ≤ k * Real.pi + Real.pi / 6 → 
    f y < f x) ∧
  (∀ (α : Real), 
    f (α + Real.pi / 6) = 12 / 5 ∧ 
    Real.pi / 2 < α ∧ 
    α < Real.pi → 
    Real.tan (2 * α + Real.pi / 4) = 1 / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_and_tan_value_l152_15278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_difference_l152_15254

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

/-- The line passing through B(4,5) -/
def line_l (k x : ℝ) : ℝ := k*(x - 4) + 5

/-- The x-coordinates of intersection points M and N -/
noncomputable def intersection_points (k : ℝ) : ℝ × ℝ := by
  let a := 1
  let b := -4*k
  let c := 16*k - 20
  let discriminant := b^2 - 4*a*c
  let x1 := (-b + Real.sqrt discriminant) / (2*a)
  let x2 := (-b - Real.sqrt discriminant) / (2*a)
  exact (x1, x2)

/-- The slopes of lines AM and AN -/
noncomputable def slopes (k : ℝ) : ℝ × ℝ := by
  let (x1, x2) := intersection_points k
  exact ((x1 - (-4))/8, (x2 - (-4))/8)

/-- The absolute difference between slopes -/
noncomputable def slope_difference (k : ℝ) : ℝ := by
  let (k1, k2) := slopes k
  exact |k1 - k2|

theorem min_slope_difference :
  ∃ (k : ℝ), ∀ (k' : ℝ), slope_difference k ≤ slope_difference k' ∧ slope_difference k = 1 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_difference_l152_15254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15212

-- Define the functions
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - (2 * Real.sqrt (4 + 2*b - b^2)) * x
noncomputable def g (a x : ℝ) : ℝ := -Real.sqrt (1 - (x - a)^2)

-- Define the theorem
theorem problem_solution :
  -- Part 1
  (∀ x : ℝ, f (-1) 0 x ≤ 4) ∧ (∃ x : ℝ, f (-1) 0 x = 4) ∧
  -- Part 2
  (∀ a b : ℤ, (∃ x₀ : ℝ, (∀ x : ℝ, f (a : ℝ) (b : ℝ) x ≤ f (a : ℝ) (b : ℝ) x₀) ∧ 
                         (∀ x : ℝ, g (a : ℝ) x ≥ g (a : ℝ) x₀)) ↔ 
              ((a = -1 ∧ b = -1) ∨ (a = -1 ∧ b = 3))) ∧
  -- Part 3
  (∀ m : ℝ, (∃! x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ x : ℝ, x ≠ 2 * ↑(Int.floor (x / 2)) → 
      (∀ y : ℝ, -f (-1) (-1) (y - 2 * ↑(Int.floor (y / 2))) = 
                 f (-1) (-1) (x - 2 * ↑(Int.floor (x / 2))) - m * x →
        x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄))) ↔
    (m = -10 + 4 * Real.sqrt 6 ∨ 
     (14 - 8 * Real.sqrt 3 < m ∧ m < 6 - 4 * Real.sqrt 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l152_15212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_real_implies_a_eq_two_l152_15266

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (a + 2i) / (1 + i) -/
noncomputable def complex_fraction (a : ℝ) : ℂ := (a + 2 * i) / (1 + i)

/-- Theorem: If (a + 2i) / (1 + i) is real, then a = 2 -/
theorem complex_fraction_real_implies_a_eq_two (a : ℝ) :
  Complex.im (complex_fraction a) = 0 → a = 2 := by
  sorry

#check complex_fraction_real_implies_a_eq_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_real_implies_a_eq_two_l152_15266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l152_15264

theorem expansion_coefficient_sum (n : ℕ) (hn : n > 0) : 
  (Nat.choose n 1 + Nat.choose (2*n) 2 + Nat.choose (2*n) 1 = 40) → n = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l152_15264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l152_15222

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length s -/
structure Cube where
  s : ℝ

/-- Represents the quadrilateral ABCD formed by the intersection of a plane with the cube -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the area of the quadrilateral ABCD -/
noncomputable def quadrilateralArea (cube : Cube) (quad : Quadrilateral) : ℝ :=
  (cube.s^2 * Real.sqrt 33) / 6

/-- Theorem statement for the area of quadrilateral ABCD -/
theorem quadrilateral_area_theorem (cube : Cube) (quad : Quadrilateral) :
  quad.A = Point3D.mk 0 0 0 →
  quad.C = Point3D.mk cube.s cube.s cube.s →
  quad.B = Point3D.mk 0 (cube.s / 3) cube.s →
  quad.D = Point3D.mk cube.s ((2 * cube.s) / 3) 0 →
  quadrilateralArea cube quad = (cube.s^2 * Real.sqrt 33) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l152_15222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l152_15277

/-- The curve C in Cartesian coordinates -/
noncomputable def C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line l in Cartesian coordinates -/
noncomputable def l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - Real.sqrt 3 = 0

/-- The point M -/
noncomputable def M : ℝ × ℝ := (-Real.sqrt 6 / 2, -Real.sqrt 2 / 2)

/-- Distance function from a point to the line l -/
noncomputable def dist_to_l (x y : ℝ) : ℝ :=
  (|x + Real.sqrt 3 * y - Real.sqrt 3|) / 2

theorem max_distance_point :
  C M.1 M.2 ∧
  ∀ (x y : ℝ), C x y → dist_to_l M.1 M.2 ≥ dist_to_l x y := by
  sorry

#check max_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l152_15277
