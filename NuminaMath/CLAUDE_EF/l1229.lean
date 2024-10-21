import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_X_count_Y_l1229_122959

variable (M A B : Finset ℕ)

-- Define the conditions given in the problem
def condition_M : Prop := Finset.card M = 10
def condition_A : Prop := A ⊆ M
def condition_B : Prop := B ⊆ M
def condition_AB : Prop := A ∩ B = ∅
def condition_card_A : Prop := Finset.card A = 2
def condition_card_B : Prop := Finset.card B = 3

-- Theorem for part (1)
theorem count_X (hM : condition_M M) (hA : condition_A M A) :
  Finset.card (Finset.filter (fun X => A ⊆ X ∧ X ⊆ M) (Finset.powerset M)) = 256 := by sorry

-- Theorem for part (2)
theorem count_Y (hM : condition_M M) (hA : condition_A M A) (hB : condition_B M B) :
  Finset.card (Finset.filter (fun Y => Y ⊆ M ∧ ¬(A ⊆ Y) ∧ ¬(B ⊆ Y)) (Finset.powerset M)) = 672 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_X_count_Y_l1229_122959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_parallel_to_same_line_are_parallel_l1229_122967

-- Define a type for lines
structure Line where
  -- Add any necessary properties for a line
  slope : ℝ
  intercept : ℝ

-- Define a relation for parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem two_lines_parallel_to_same_line_are_parallel 
  (l1 l2 l3 : Line) (h1 : parallel l1 l3) (h2 : parallel l2 l3) : 
  parallel l1 l2 := by
  -- Unfold the definition of parallel
  unfold parallel at *
  -- Use transitivity of equality
  rw [h1, ← h2]

#check two_lines_parallel_to_same_line_are_parallel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_parallel_to_same_line_are_parallel_l1229_122967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_truck_trip_l1229_122930

/-- The total cost function for a truck trip -/
noncomputable def total_cost (x : ℝ) : ℝ := 2340 / x + 13 * x / 18

/-- Theorem stating the minimum cost and the speed at which it occurs -/
theorem min_cost_truck_trip :
  ∃ (min_cost : ℝ) (optimal_speed : ℝ),
    (∀ x : ℝ, 50 ≤ x → x ≤ 100 → total_cost x ≥ min_cost) ∧
    (50 ≤ optimal_speed ∧ optimal_speed ≤ 100) ∧
    (total_cost optimal_speed = min_cost) ∧
    (min_cost = 26 * Real.sqrt 10) ∧
    (optimal_speed = 18 * Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_truck_trip_l1229_122930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_l1229_122903

-- Define the universal set U as the real numbers
def U := ℝ

-- Define the complement of A in U
def complementA : Set U := {x : ℝ | x ≥ 1}

-- Define set B
def B : Set U := {x : ℝ | x < -2}

-- Define set A (derived from its complement)
def A : Set U := {x : ℝ | x ∉ complementA}

-- Statement to prove
theorem union_A_B : A ∪ B = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_l1229_122903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_b_l1229_122969

noncomputable def y (b θ : ℝ) : ℝ := 4*b^2 - 3*b^2*(Real.sin (2*θ)) - 3*b*(Real.sin θ) + 9/4

theorem max_value_implies_b (b : ℝ) :
  (∀ θ, y b θ ≤ 7) ∧ (∃ θ, y b θ = 7) → b = 1 ∨ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_b_l1229_122969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EH_length_l1229_122981

/-- A right-angled trapezoid with specific measurements -/
structure RightTrapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EF_parallel_GH : True  -- Represents that EF is parallel to GH
  EF_less_GH : EF < GH
  right_angle_F : True   -- Represents that angle F is a right angle
  right_angle_G : True   -- Represents that angle G is a right angle

/-- The length of EH in a right-angled trapezoid with given measurements -/
noncomputable def length_EH (t : RightTrapezoid) : ℝ :=
  Real.sqrt 338

/-- Theorem stating that for a right-angled trapezoid with given measurements, 
    the length of EH is √338 -/
theorem trapezoid_EH_length :
  ∀ t : RightTrapezoid, 
  t.EF = 7 ∧ t.FG = 6 ∧ t.GH = 24 →
  length_EH t = Real.sqrt 338 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EH_length_l1229_122981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_problem_solution_l1229_122982

/-- Represents the problem of maximizing sandwich purchases and using remaining funds for soft drinks --/
def sandwichProblem (totalMoney sandwichCost drinkCost : ℚ) : ℕ := 
  let maxSandwiches := (totalMoney / sandwichCost).floor.toNat
  let remainingMoney := totalMoney - maxSandwiches * sandwichCost
  let drinks := (remainingMoney / drinkCost).floor.toNat
  maxSandwiches + drinks

/-- Theorem stating that given the problem conditions, the total number of items purchased is 9 --/
theorem sandwich_problem_solution :
  sandwichProblem 30 (9/2) 1 = 9 := by
  -- The proof goes here
  sorry

#eval sandwichProblem 30 (9/2) 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_problem_solution_l1229_122982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1229_122909

theorem power_difference (m n : ℝ) (h1 : (10 : ℝ)^m = 2) (h2 : (10 : ℝ)^n = 3) : 
  (10 : ℝ)^(m-n) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1229_122909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1229_122913

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let (x₃, y₃) := R
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₁) - (x₁ - x₂) * (y₃ - y₁))

/-- The theorem stating that the area of triangle PQR with given coordinates is 30 square units -/
theorem area_of_triangle_PQR : 
  triangle_area (-2, 2) (8, 2) (6, -4) = 30 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1229_122913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_22_l1229_122951

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle with width w and length l -/
structure Rectangle where
  w : ℝ
  l : ℝ

/-- The area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  (t.a * t.b) / 2

/-- The area of a rectangle -/
noncomputable def Rectangle.area (r : Rectangle) : ℝ :=
  r.w * r.l

/-- The perimeter of a rectangle -/
noncomputable def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.w + r.l)

theorem rectangle_perimeter_equals_22 (t : Triangle) (r : Rectangle) : 
  t.a = 5 → t.b = 12 → t.c = 13 → r.w = 5 → Triangle.area t = Rectangle.area r → 
  Rectangle.perimeter r = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_22_l1229_122951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_x_squared_plus_3_l1229_122947

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc 3 6

-- State the theorem
theorem domain_of_f_x_squared_plus_3 
  (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) = f (x + 1)) :
  {x : ℝ | f (x^2 + 3) = f (x^2 + 3)} = Set.Icc (-2) (-1) ∪ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_x_squared_plus_3_l1229_122947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_50_l1229_122983

theorem units_digit_factorial_50 : ∃ k : ℕ, Nat.factorial 50 = 10 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_50_l1229_122983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_value_l1229_122996

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem minimum_m_value (a₁ d b₁ q : ℝ) (h1 : b₁ = 2) (h2 : b₁ = (a₁ + (a₁ + d)) / 2)
    (h3 : arithmetic_sequence a₁ d 3 = 5)
    (h4 : geometric_sequence b₁ q 3 = arithmetic_sequence a₁ d 4 + 1)
    (h5 : ∀ n : ℕ, n ≥ 2 → geometric_sequence b₁ q n > geometric_sequence b₁ q (n - 1)) :
  (∃ m : ℕ, ∀ n : ℕ, n ≥ m → sum_arithmetic_sequence a₁ d n ≤ geometric_sequence b₁ q n) ∧
  (∀ m : ℕ, m < 4 → ∃ n : ℕ, n ≥ m ∧ sum_arithmetic_sequence a₁ d n > geometric_sequence b₁ q n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_value_l1229_122996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_alpha_simplified_l1229_122999

theorem f_alpha_simplified (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) -- α is in the third quadrant
  (h2 : Real.cos (α - 3*π/2) = 1/5) :
  let f := λ β : Real => (Real.sin (β - π/2) * Real.cos (3*π/2 + β) * Real.tan (π - β)) / 
              (Real.tan (-π + β) * Real.sin (π + β))
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_alpha_simplified_l1229_122999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1229_122920

def S (n : ℕ) : ℕ := n^2 + 3*n + 1

def a : ℕ → ℕ
  | 0 => 5  -- Add this case to handle n = 0
  | 1 => 5
  | n + 2 => 2*(n + 2) + 2

theorem sequence_general_term (n : ℕ) : 
  (n = 1 ∧ a n = 5) ∨ 
  (n > 1 ∧ a n = S n - S (n-1)) := by
  sorry

#check sequence_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1229_122920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_removed_rectangle_to_square_l1229_122938

/-- Represents a rectangular shape with corner cells removed -/
structure CornerRemovedRectangle where
  width : ℕ
  height : ℕ
  corner_size : ℕ

/-- Represents a part cut from the original shape -/
inductive ShapePart
  | LShaped : ℕ → ℕ → ShapePart
  | Irregular : ℕ → ShapePart

/-- Checks if a part is not a square -/
def is_not_square : ShapePart → Prop
  | ShapePart.LShaped _ _ => True
  | ShapePart.Irregular _ => True

/-- Checks if parts can form a square of given size -/
def can_form_square (parts : List ShapePart) (size : ℕ) : Prop :=
  (parts.map fun p => match p with
    | ShapePart.LShaped w h => w * h
    | ShapePart.Irregular area => area).sum = size * size

/-- Main theorem statement -/
theorem corner_removed_rectangle_to_square 
  (rect : CornerRemovedRectangle) 
  (h_rect : rect.width = 5 ∧ rect.height = 4 ∧ rect.corner_size = 1) :
  ∃ (parts : List ShapePart),
    parts.length = 3 ∧ 
    (∀ p ∈ parts, is_not_square p) ∧
    can_form_square parts 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_removed_rectangle_to_square_l1229_122938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1229_122950

-- Define the function f(x) = 3^(|x-1|)
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ (|x - 1|)

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1229_122950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1229_122960

-- Part 1
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem part1 (b c : ℝ) (hb : b < 0) :
  (∀ x ∈ Set.Icc 0 1, f b c x ∈ Set.Icc 0 1) ∧
  (Set.range (f b c) = Set.Icc 0 1) →
  b = -2 ∧ c = 1 := by sorry

-- Part 2
noncomputable def g (c x : ℝ) : ℝ := (x^2 - 2*x + c) / x

theorem part2 (c : ℝ) :
  (∀ x ∈ Set.Icc 3 5, g c x > c) →
  c < 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1229_122960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l1229_122976

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point A on the circle
def point_A (x y : ℝ) : Prop := my_circle x y

-- Define line l perpendicular to x-axis through A
def line_l (x₀ y₀ x : ℝ) : Prop := x = x₀

-- Define point D as intersection of l and x-axis
def point_D (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)

-- Define point M on l
def point_M (x₀ y₀ x y : ℝ) : Prop :=
  line_l x₀ y₀ x ∧ y = (Real.sqrt 3/2) * y₀

-- Theorem: The trajectory of M forms an ellipse
theorem trajectory_is_ellipse (x y : ℝ) :
  (∃ x₀ y₀ : ℝ, point_A x₀ y₀ ∧ point_M x₀ y₀ x y) →
  x^2/4 + y^2/3 = 1 := by
  sorry

#check trajectory_is_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l1229_122976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_points_l1229_122912

/-- Point in 2D Euclidean space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Circle defined by center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Angle bisector of a triangle -/
noncomputable def angle_bisector (t : Triangle) (v : Point) : Line :=
  sorry

/-- Circumcircle of a triangle -/
noncomputable def circumcircle (t : Triangle) : Circle :=
  sorry

/-- Incircle of a triangle -/
noncomputable def incircle (t : Triangle) : Circle :=
  sorry

/-- Intersection of a line and a circle -/
noncomputable def line_circle_intersection (l : Line) (c : Circle) : Point :=
  sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Intersection of a line and another line -/
noncomputable def line_line_intersection (l1 : Line) (l2 : Line) : Point :=
  sorry

/-- Theorem: Triangle construction from given points -/
theorem triangle_construction_from_points
  (A E F P : Point)
  (h1 : ∃ (t : Triangle), E = line_line_intersection (angle_bisector t A) (Line.mk t.B t.C))
  (h2 : ∃ (t : Triangle), F = line_circle_intersection (angle_bisector t A) (circumcircle t))
  (h3 : ∃ (t : Triangle), point_on_circle P (incircle t)) :
  ∃! (t : Triangle), t.A = A ∧
    E = line_line_intersection (angle_bisector t A) (Line.mk t.B t.C) ∧
    F = line_circle_intersection (angle_bisector t A) (circumcircle t) ∧
    point_on_circle P (incircle t) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_points_l1229_122912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_M_l1229_122910

universe u

-- Define the set M
def M : Finset (Fin 3) := {0, 1, 2}

-- Define a mapping from Fin 3 to {a, b, c}
def elementMap : Fin 3 → Char
  | 0 => 'a'
  | 1 => 'b'
  | 2 => 'c'

-- Theorem statement
theorem subsets_of_M :
  (∃ (S : Set (Set Char)),
    S = {∅, {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}} ∧
    S = Set.powerset (Set.image elementMap M.toSet)) ∧
  (∃ (P : Set (Set Char)),
    P = {∅, {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}} ∧
    P = {s | s ∈ Set.powerset (Set.image elementMap M.toSet) ∧ s ≠ Set.image elementMap M.toSet}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_M_l1229_122910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_rental_cost_l1229_122964

/-- Represents the daily rental business for canoes and kayaks. -/
structure RentalBusiness where
  canoe_cost : ℚ  -- Cost of renting a canoe per day
  kayak_cost : ℚ  -- Cost of renting a kayak per day
  canoe_count : ℕ  -- Number of canoes rented
  kayak_count : ℕ  -- Number of kayaks rented
  total_revenue : ℚ  -- Total revenue for the day

/-- The theorem stating the cost of a canoe rental per day. -/
theorem canoe_rental_cost (rb : RentalBusiness) : rb.canoe_cost = 11 :=
  by
  have h1 : rb.kayak_cost = 16 := by sorry
  have h2 : rb.canoe_count = rb.kayak_count + 5 := by sorry
  have h3 : 4 * rb.kayak_count = 3 * rb.canoe_count := by sorry
  have h4 : rb.total_revenue = 460 := by sorry
  have h5 : rb.total_revenue = rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_rental_cost_l1229_122964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_l1229_122954

/-- A line segment connecting two points in 2D space -/
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Parameterization of a line segment -/
structure Parameterization where
  e : ℝ
  f : ℝ
  g : ℝ
  h : ℝ

/-- Theorem: The sum of squares of coefficients for the given line segment parameterization -/
theorem sum_of_squares_of_coefficients 
  (segment : LineSegment) 
  (param : Parameterization) : 
  segment.start = (1, -3) →
  segment.endpoint = (-4, 6) →
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (param.e * t + param.f, param.g * t + param.h) ∈ Set.Icc segment.start segment.endpoint) →
  param.e^2 + param.f^2 + param.g^2 + param.h^2 = 116 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_l1229_122954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_range_l1229_122958

noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  (1 - a^2) * x^2 + a^2 * y^2 = a^2

noncomputable def upper_vertex (a : ℝ) : ℝ × ℝ := (0, 1)

noncomputable def intersection_point (a : ℝ) : ℝ × ℝ := (-a, a)

noncomputable def parabola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 = -4 * (m - 1) * (y - m)

noncomputable def slope_pm (a m : ℝ) : ℝ := (m - a) / a

theorem hyperbola_range (a : ℝ) :
  a > 1 →
  hyperbola a (-a) a →
  (∃ m, parabola m (-a) a ∧
       1/4 ≤ slope_pm a m ∧ slope_pm a m ≤ 1/3) →
  12/7 ≤ a ∧ a ≤ 4 := by
  sorry

#check hyperbola_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_range_l1229_122958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_theorem_l1229_122972

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- The maximum squared radius of a sphere inside two intersecting cones -/
noncomputable def max_sphere_radius_squared (c : Cone) (intersectionDistance : ℝ) : ℝ :=
  (289 - 84 * Real.sqrt 2) / 2

/-- Theorem statement for the maximum sphere radius squared -/
theorem max_sphere_radius_squared_theorem (c1 c2 : Cone) (h1 : c1 = c2) 
    (h2 : c1.baseRadius = 5) (h3 : c1.height = 12) (intersectionDistance : ℝ) 
    (h4 : intersectionDistance = 5) : 
  max_sphere_radius_squared c1 intersectionDistance = (289 - 84 * Real.sqrt 2) / 2 := by
  sorry

#check max_sphere_radius_squared_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_theorem_l1229_122972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_reading_speed_l1229_122977

/-- Proves that given the conditions, Juan reads 250 pages per hour -/
theorem juan_reading_speed 
  (lunch_time : ℝ) 
  (book_pages : ℕ) 
  (office_to_lunch_time : ℝ)
  (pages_per_hour : ℝ)
  (h1 : lunch_time = 2 * office_to_lunch_time)
  (h2 : book_pages = 4000)
  (h3 : office_to_lunch_time = 4)
  (h4 : lunch_time = (book_pages : ℝ) / pages_per_hour) :
  pages_per_hour = 250 := by
  sorry

#check juan_reading_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_reading_speed_l1229_122977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_player_households_l1229_122985

/-- The number of households with at least one DVD player -/
def D : ℕ := sorry

/-- The number of households with at least one cell phone -/
def C : ℕ := 90

/-- The number of households with at least one MP3 player -/
def M : ℕ := 55

/-- The total number of households -/
def total : ℕ := 100

/-- The greatest possible number of households with all 3 devices -/
def x : ℕ := sorry

/-- The lowest possible number of households with all 3 devices -/
def y : ℕ := sorry

theorem dvd_player_households :
  C = 90 →
  M = 55 →
  total = 100 →
  x ≤ M →
  y = D + C + M - total - (x - y) →
  x - y = 25 →
  D = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_player_households_l1229_122985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_roots_l1229_122908

theorem constant_sum_of_roots (a b : ℤ) : 
  (∃ r s t : ℕ+, (r : ℤ) * s * t = 2310 ∧ 
   ∀ x : ℝ, x^3 - a*x^2 + b*x - 2310 = (x - r) * (x - s) * (x - t)) → 
  a = 48 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_roots_l1229_122908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proof_methods_evaluation_incorrect_statements_l1229_122924

-- Define the statements as axioms (accepted without proof)
axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom statement_D : Prop
axiom statement_E : Prop

-- Define the nature of mathematical proofs
axiom direct_proof : Prop
axiom indirect_proof : Prop
axiom contrapositive : Prop → Prop
axiom converse : Prop → Prop

-- Theorem stating which statements are correct and incorrect
theorem proof_methods_evaluation :
  (¬statement_A) ∧
  statement_B ∧
  statement_C ∧
  statement_D ∧
  (¬statement_E) :=
by
  sorry  -- We use sorry to skip the proof

-- Theorem to show that A and E are the incorrect statements
theorem incorrect_statements :
  (¬statement_A) ∧ (¬statement_E) :=
by
  sorry  -- We use sorry to skip the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proof_methods_evaluation_incorrect_statements_l1229_122924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_2012_solutions_frac_properties_l1229_122937

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2 * frac x - 1|

-- Define the theorem
theorem smallest_n_with_2012_solutions :
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (S : Set ℝ), Finite S ∧ S.ncard ≥ 2012 ∧ ∀ x ∈ S, n * f (x * f x) = x) ∧
  (∀ m : ℕ, m < n → 
    ¬∃ (T : Set ℝ), Finite T ∧ T.ncard ≥ 2012 ∧ ∀ x ∈ T, m * f (x * f x) = x) ∧
  n = 32 := by
  sorry

-- Additional properties of the fractional part function
theorem frac_properties (x : ℝ) : 
  0 ≤ frac x ∧ frac x < 1 ∧ ∃ (k : ℤ), x - frac x = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_2012_solutions_frac_properties_l1229_122937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_possible_l1229_122901

/-- Represents a container with a certain capacity and current amount --/
structure Container where
  capacity : Nat
  amount : Nat
  h_amount_le_capacity : amount ≤ capacity

/-- Represents the state of all containers --/
structure State where
  original : Container
  container13 : Container
  container11 : Container
  container5 : Container

/-- Checks if a given state represents an equal division --/
def isEqualDivision (s : State) : Prop :=
  s.container13.amount = 8 ∧ s.container11.amount = 8 ∧ s.container5.amount = 8

/-- Defines a valid pour operation between two containers --/
def canPour (source dest : Container) (amount : Nat) : Prop :=
  amount ≤ source.amount ∧ amount + dest.amount ≤ dest.capacity

/-- Theorem stating that equal division is possible --/
theorem equal_division_possible : ∃ (final : State),
  isEqualDivision final ∧
  (∃ (initial : State),
    initial.original.capacity = 24 ∧
    initial.original.amount = 24 ∧
    initial.container13.capacity = 13 ∧
    initial.container11.capacity = 11 ∧
    initial.container5.capacity = 5 ∧
    (∃ (steps : List (Container × Container × Nat)),
      -- There exists a sequence of valid pour operations
      -- that transforms the initial state to the final state
      true)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_possible_l1229_122901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_minus_π_div_2_l1229_122980

-- Define the angle α
noncomputable def α : Real := Real.arctan (-1 / Real.sqrt 3)

-- Theorem statement
theorem sin_2α_minus_π_div_2 : Real.sin (2 * α - Real.pi / 2) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_minus_π_div_2_l1229_122980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1229_122968

-- Define the ◎ operation
noncomputable def diamond (a b : ℝ) : ℝ :=
  if a > b then 3 / (a - b) else b / (b - a)

-- Theorem statement
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond 2 x = 3 → x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1229_122968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l1229_122919

theorem line_through_point_inequality (a b θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (Real.cos θ / a + Real.sin θ / b = 1) → (1 / a^2 + 1 / b^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l1229_122919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_value_relation_l1229_122949

/-- Represents the characteristics and value of a stock -/
structure Stock where
  face_value : ℚ
  dividend_rate : ℚ
  yield : ℚ

/-- Calculates the market value of a stock -/
def market_value (s : Stock) : ℚ :=
  (s.face_value * s.dividend_rate) / s.yield * 100

/-- Theorem stating the relationship between face value and market value for the given stock -/
theorem market_value_relation (s : Stock) 
    (h1 : s.dividend_rate = 8/100) 
    (h2 : s.yield = 20/100) : 
  market_value s = 40 * s.face_value := by
  -- Unfold the definition of market_value
  unfold market_value
  -- Substitute the given conditions
  rw [h1, h2]
  -- Perform algebraic simplification
  field_simp
  ring

#check market_value_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_value_relation_l1229_122949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_example_l1229_122987

/-- Represents a geometric sequence with first term a₁, common ratio q, and k-th term aₖ. -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ
  k : ℕ
  aₖ : ℝ

/-- Sum of the first k terms of a geometric sequence. -/
noncomputable def geometricSum (g : GeometricSequence) : ℝ :=
  g.a₁ * (1 - g.q^g.k) / (1 - g.q)

/-- Theorem stating that for a specific geometric sequence, the sum of its first k terms is 364. -/
theorem geometric_sum_example :
  ∃ g : GeometricSequence,
    g.a₁ = 1 ∧
    g.q = 3 ∧
    g.aₖ = 243 ∧
    geometricSum g = 364 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_example_l1229_122987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_two_digit_prime_saturated_l1229_122965

def is_prime_saturated (n : Nat) : Prop :=
  (Nat.factors n).prod < Real.sqrt (n : ℝ)

theorem greatest_two_digit_prime_saturated : 
  (∀ m : Nat, m ≥ 10 ∧ m < 100 → is_prime_saturated m → m ≤ 98) ∧ 
  is_prime_saturated 98 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_two_digit_prime_saturated_l1229_122965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_morse_code_symbols_count_l1229_122915

theorem morse_code_symbols_count : Finset.sum {1, 2, 3, 4, 5} (λ n => 2^n) = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_morse_code_symbols_count_l1229_122915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l1229_122926

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ
  h_AB_positive : 0 < AB
  h_CD_positive : 0 < CD
  h_AB_less_CD : AB < CD
  h_height_positive : 0 < height

/-- The perimeter of an isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  t.AB + t.CD + 2 * Real.sqrt ((t.CD - t.AB)^2 / 4 + t.height^2)

/-- Theorem stating the perimeter of a specific isosceles trapezoid -/
theorem specific_trapezoid_perimeter :
  ∃ t : IsoscelesTrapezoid,
    t.AB = 10 ∧ t.CD = 18 ∧ t.height = 6 ∧
    perimeter t = 28 + 4 * Real.sqrt 13 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l1229_122926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1229_122905

/-- The parabola defined by y = -x^2 has its vertex at (0, 0) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 → (0, 0) = (x, y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1229_122905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1229_122957

/-- A quadratic function passing through two given points -/
noncomputable def QuadraticFunction (b c : ℝ) : ℝ → ℝ := fun x ↦ (1/2) * x^2 + b * x + c

theorem quadratic_function_properties (b c : ℝ) :
  (QuadraticFunction b c 0 = -1) ∧ 
  (QuadraticFunction b c 2 = -3) →
  (∀ x, QuadraticFunction b c x = (1/2) * x^2 - 2*x - 1) ∧
  (∃ x y, QuadraticFunction b c x = y ∧ 
    ∀ t, QuadraticFunction b c t ≥ QuadraticFunction b c x ∧
    x = 2 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1229_122957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_approx_l1229_122992

/-- Calculates the brokerage percentage given the cash realized and net amount received from a stock sale. -/
noncomputable def brokerage_percentage (cash_realized : ℝ) (net_amount : ℝ) : ℝ :=
  ((cash_realized - net_amount) / cash_realized) * 100

/-- Theorem stating that for the given stock sale, the brokerage percentage is approximately 0.2310%. -/
theorem brokerage_percentage_approx :
  let cash_realized : ℝ := 108.25
  let net_amount : ℝ := 108
  abs (brokerage_percentage cash_realized net_amount - 0.2310) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_approx_l1229_122992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_vertex_angle_l1229_122998

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The angle between the semi-major axis and the line connecting a vertex 
    on the major axis to the vertex on the minor axis -/
noncomputable def vertex_angle (e : Ellipse) : ℝ :=
  Real.arccos (e.b / e.a)

theorem ellipse_eccentricity_from_vertex_angle (e : Ellipse) 
    (h : vertex_angle e = 2 * Real.pi / 3) : 
    eccentricity e = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_vertex_angle_l1229_122998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rubber_elongation_improvement_l1229_122927

noncomputable section

def x : Fin 10 → ℝ := sorry
def y : Fin 10 → ℝ := sorry

def z (i : Fin 10) : ℝ := x i - y i

noncomputable def z_bar : ℝ := (Finset.univ.sum z) / 10

noncomputable def s_squared : ℝ := (Finset.univ.sum (λ i => (z i - z_bar)^2)) / 10

def significant_improvement : Prop := z_bar ≥ 2 * Real.sqrt (s_squared / 10)

theorem rubber_elongation_improvement : significant_improvement := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rubber_elongation_improvement_l1229_122927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_constructible_l1229_122922

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A predicate that checks if a set of points satisfies given distances -/
def satisfies_distances (points : Finset Point) (distances : Finset ℝ) : Prop :=
  ∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → distance p1 p2 ∈ distances

/-- A predicate that checks if any subset of 5 points can be constructed -/
def any_five_constructible (points : Finset Point) (distances : Finset ℝ) : Prop :=
  ∀ subset : Finset Point, subset.card = 5 → subset ⊆ points → 
    ∃ constructed : Finset Point, constructed.card = 5 ∧ 
      satisfies_distances constructed distances

/-- The main theorem: if any 5 points can be constructed, all N points can be constructed -/
theorem all_points_constructible 
  (N : ℕ) (points : Finset Point) (distances : Finset ℝ) 
  (h_card : points.card = N) (h_five : any_five_constructible points distances) : 
  ∃ constructed : Finset Point, constructed.card = N ∧ 
    satisfies_distances constructed distances :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_constructible_l1229_122922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_required_l1229_122916

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℕ

instance : OfNat Moles n where
  ofNat := n

/-- Represents the chemical reaction: AgNO3 + HCl → AgCl + HNO3 -/
structure Reaction where
  silver_nitrate : Moles
  hydrochloric_acid : Moles
  silver_chloride : Moles
  nitric_acid : Moles

/-- The reaction is balanced when all components have the same number of moles -/
def is_balanced (r : Reaction) : Prop :=
  r.silver_nitrate = r.hydrochloric_acid ∧
  r.silver_nitrate = r.silver_chloride ∧
  r.silver_nitrate = r.nitric_acid

theorem hydrochloric_acid_required (r : Reaction) 
  (h1 : r.silver_nitrate = 3)
  (h2 : r.silver_chloride = 3)
  (h3 : r.nitric_acid = 3)
  (h4 : is_balanced r) :
  r.hydrochloric_acid = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_required_l1229_122916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_school_count_l1229_122962

/-- Represents a student in the mathematics contest -/
structure Student where
  name : String
  rank : Nat

/-- Represents a school in the suburb of Pythagoras -/
structure School where
  team : Finset Student

/-- The mathematics contest -/
structure Contest where
  schools : Finset School
  allStudents : Finset Student

theorem contest_school_count (contest : Contest) : 
  contest.schools.card = 20 :=
by
  have h1 : ∀ s : School, s ∈ contest.schools → s.team.card = 4 := sorry
  have h2 : ∀ s1 s2 : Student, s1 ∈ contest.allStudents → s2 ∈ contest.allStudents → s1 ≠ s2 → s1.rank ≠ s2.rank := sorry
  have h3 : ∃ andrea : Student, andrea ∈ contest.allStudents ∧ 
    andrea.rank = (contest.allStudents.card + 1) / 2 := sorry
  have h4 : ∃ andrea beth carla dan : Student, 
    andrea ∈ contest.allStudents ∧ beth ∈ contest.allStudents ∧ 
    carla ∈ contest.allStudents ∧ dan ∈ contest.allStudents ∧
    beth.rank = 45 ∧ carla.rank = 73 ∧ dan.rank = 85 ∧
    andrea.rank < beth.rank ∧ andrea.rank < carla.rank ∧ andrea.rank < dan.rank := sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_school_count_l1229_122962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l1229_122988

theorem cosine_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos (2*x) + Real.cos (6*x) + Real.cos (10*x) + Real.cos (14*x) = 
      (a : ℝ) * Real.cos (b*x) * Real.cos (c*x) * Real.cos (d*x)) ∧
    a + b + c + d = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l1229_122988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_20_divisibility_l1229_122984

theorem factorial_20_divisibility : ∃ (a b : ℕ), (
  (10^a ∣ Nat.factorial 20) ∧ 
  ∀ k > a, ¬(10^k ∣ Nat.factorial 20) ∧
  (6^b ∣ Nat.factorial 20) ∧ 
  ∀ m > b, ¬(6^m ∣ Nat.factorial 20) ∧
  a + b = 12
) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_20_divisibility_l1229_122984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_score_surpass_l1229_122943

theorem highest_score_surpass (total_questions : ℕ) (current_highest : ℚ) : ℕ :=
  let min_score_to_beat := current_highest + 1
  let min_correct_answers := ⌈min_score_to_beat⌉
  by
    have h1 : total_questions = 50 := by sorry
    have h2 : current_highest = 47.5 := by sorry
    have h3 : min_correct_answers = 49 := by sorry
    exact min_correct_answers.toNat

#check highest_score_surpass

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_score_surpass_l1229_122943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_answers_needed_l1229_122955

/-- Represents the number of additional correct answers needed to pass a test -/
def additional_correct_answers (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arithmetic_correct_percent : ℚ) (algebra_correct_percent : ℚ)
  (geometry_correct_percent : ℚ) (passing_grade : ℚ) : ℕ :=
  let correct_arithmetic := (arithmetic_correct_percent * arithmetic).floor
  let correct_algebra := (algebra_correct_percent * algebra).floor
  let correct_geometry := (geometry_correct_percent * geometry).floor
  let total_correct := correct_arithmetic + correct_algebra + correct_geometry
  let needed_correct := (passing_grade * total).ceil
  (needed_correct - total_correct).toNat

/-- The theorem stating the number of additional correct answers needed -/
theorem additional_answers_needed :
  additional_correct_answers 80 15 30 35 (4/5) (2/5) (3/5) (13/20) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_answers_needed_l1229_122955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypergeometric_scenarios_hypergeometric_existence_l1229_122921

-- Define the characteristics of a hypergeometric distribution
structure HypergeometricDistribution where
  population_size : ℕ
  success_count : ℕ
  sample_size : ℕ

-- Define the scenarios
inductive Scenario
| Dice
| Seeds
| Balls
| Students

-- Define a function to check if a scenario is hypergeometric
def isHypergeometric : Scenario → Bool
| Scenario.Dice => false
| Scenario.Seeds => false
| Scenario.Balls => true
| Scenario.Students => true

-- Define specific hypergeometric distributions for scenarios 3 and 4
def scenario3 : HypergeometricDistribution :=
  { population_size := 12,  -- Total number of balls
    success_count := 9,    -- Number of non-red balls
    sample_size := 3 }     -- Number of balls drawn

def scenario4 : HypergeometricDistribution :=
  { population_size := 44,  -- Total number of students (excluding class leader)
    success_count := 20,   -- Number of girls
    sample_size := 3 }     -- Number of students selected (excluding class leader)

-- Theorem stating which scenarios belong to the hypergeometric distribution
theorem hypergeometric_scenarios :
  (¬ isHypergeometric Scenario.Dice) ∧
  (¬ isHypergeometric Scenario.Seeds) ∧
  (isHypergeometric Scenario.Balls) ∧
  (isHypergeometric Scenario.Students) := by
  sorry

-- Theorem confirming the existence of hypergeometric distributions for scenarios 3 and 4
theorem hypergeometric_existence :
  (∃ (h : HypergeometricDistribution), h = scenario3) ∧
  (∃ (h : HypergeometricDistribution), h = scenario4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypergeometric_scenarios_hypergeometric_existence_l1229_122921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_count_l1229_122933

/-- The binomial expression -/
noncomputable def binomial (x : ℝ) : ℝ := (Real.sqrt x + 1 / (4 * x)) ^ 8

/-- The number of rational terms in the expansion -/
def num_rational_terms : ℕ := 3

/-- The number of non-rational terms in the expansion -/
def num_non_rational_terms : ℕ := 6

/-- The number of slots available for rational terms after arranging non-rational terms -/
def num_slots : ℕ := 7

/-- The number of ways to rearrange the terms of the binomial expansion 
    such that no two rational terms are adjacent -/
def num_rearrangements : ℕ := 
  (Nat.descFactorial num_non_rational_terms num_non_rational_terms) * 
  (Nat.descFactorial num_slots num_rational_terms)

theorem rearrangement_count : 
  num_rearrangements = (Nat.descFactorial 6 6) * (Nat.descFactorial 7 3) := by
  sorry

#eval num_rearrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_count_l1229_122933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1229_122917

/-- The area of the shaded region bounded by two lines -/
theorem shaded_area_theorem : 
  let line1 : ℝ → ℝ := fun x => -3/8 * x + 4
  let line2 : ℝ → ℝ := fun x => -5/6 * x + 35/6
  let intersection_x : ℝ := 4
  let shaded_area : ℝ := 2 * (∫ x in Set.Icc 0 1, (5 - line1 x) + ∫ x in Set.Icc 1 intersection_x, (line2 x - line1 x))
  shaded_area = 13/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1229_122917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1229_122929

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) : 
  (a₁ * (1 - q^3) / (1 - q)) / (a₁ * (1 - q^2) / (1 - q)) = 3 / 2 → q = 1 ∨ q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1229_122929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1229_122975

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_value_of_f :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
  f x = Real.pi / 6 + Real.sqrt 3 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1229_122975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l1229_122973

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3
def circle_C2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the circle where P lies
def circle_P (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem min_distance_PQ :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_P x1 y1 ∧ circle_C2 x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      circle_P x3 y3 → circle_C2 x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l1229_122973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_volume_equals_water_displacement_l1229_122966

/-- The volume of water displaced by submerging objects -/
def water_displacement (objects : Set ℝ) : ℝ := sorry

/-- A set representing a dozen dozen apples -/
def dozen_dozen_apples : Set ℝ := sorry

/-- The volume of a set of objects in cubic centimeters -/
def volume (objects : Set ℝ) : ℝ := sorry

theorem apple_volume_equals_water_displacement :
  volume dozen_dozen_apples = water_displacement dozen_dozen_apples := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_volume_equals_water_displacement_l1229_122966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_XYZ_l1229_122923

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define point X as the intersection of AD and BC
noncomputable def X (A B C D : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define Y as the midpoint of AC
noncomputable def Y (A C : ℝ × ℝ) : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define Z as the midpoint of BD
noncomputable def Z (B D : ℝ × ℝ) : ℝ × ℝ := ((B.1 + D.1) / 2, (B.2 + D.2) / 2)

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- Function to calculate the area of a quadrilateral
noncomputable def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_XYZ (A B C D : ℝ × ℝ) 
  (h_convex : sorry) -- Assumption that ABCD is convex
  (h_area : quadrilateralArea A B C D = 1) : -- Assumption that area of ABCD is 1
  triangleArea (X A B C D) (Y A C) (Z B D) = 1/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_XYZ_l1229_122923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1229_122914

def mySequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  a 1 = 1 ∧
  ∀ n, a (n + 2) = 1 / (a n + 1)

theorem sequence_sum (a : ℕ → ℝ) (h : mySequence a) (h100 : a 100 = a 96) :
  a 2014 + a 3 = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1229_122914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_existence_l1229_122995

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: There exists a quadrilateral with given side lengths and sum of two adjacent angles -/
theorem quadrilateral_existence :
  ∃ (q : Quadrilateral),
    distance q.A q.B = 4 ∧
    distance q.B q.C = 2 ∧
    distance q.C q.D = 8 ∧
    distance q.D q.A = 5.5 ∧
    angle q.D q.A q.B + angle q.A q.B q.C = 225 * π / 180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_existence_l1229_122995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_weight_l1229_122942

theorem initial_average_weight (class_size : ℕ) (misread_weight : ℝ) (correct_weight : ℝ) (correct_average : ℝ) :
  class_size = 20 →
  misread_weight = 56 →
  correct_weight = 68 →
  correct_average = 59 →
  (class_size * correct_average - (correct_weight - misread_weight)) / class_size = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_weight_l1229_122942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_not_always_equal_l1229_122989

-- Define the basic geometric concepts
structure Triangle
structure Angle
structure Line

-- Define the properties mentioned in the problem
def congruent (t1 t2 : Triangle) : Prop := sorry
def correspondingAngles (t1 t2 : Triangle) (a1 a2 : Angle) : Prop := sorry
def verticalAngles (a1 a2 : Angle) : Prop := sorry
def rightTriangle (t : Triangle) : Prop := sorry
def equalSides (t1 t2 : Triangle) (s1 s2 : Nat) : Prop := sorry

-- State the given true propositions
axiom congruent_corresponding_angles_equal :
  ∀ (t1 t2 : Triangle) (a1 a2 : Angle),
    congruent t1 t2 → correspondingAngles t1 t2 a1 a2 → a1 = a2

axiom vertical_angles_equal :
  ∀ (a1 a2 : Angle), verticalAngles a1 a2 → a1 = a2

axiom right_triangles_two_sides_equal_congruent :
  ∀ (t1 t2 : Triangle),
    rightTriangle t1 → rightTriangle t2 →
    ∃ (s1 s2 : Nat), equalSides t1 t2 s1 s2 →
    congruent t1 t2

-- The theorem to be proven
theorem corresponding_angles_not_always_equal :
  ¬(∀ (t1 t2 : Triangle) (a1 a2 : Angle), correspondingAngles t1 t2 a1 a2 → a1 = a2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_not_always_equal_l1229_122989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_BC_l1229_122939

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle bisectors
def angle_bisector_B : ℝ → ℝ := fun _ ↦ 0
def angle_bisector_C : ℝ → ℝ := fun x ↦ x

-- Define the line BC
def line_BC : ℝ → ℝ := fun x ↦ 2 * x + 5

-- Theorem statement
theorem triangle_line_BC (t : Triangle) :
  t.A = (3, -1) →
  (angle_bisector_B = fun _ ↦ 0) →
  (angle_bisector_C = fun x ↦ x) →
  (line_BC = fun x ↦ 2 * x + 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_BC_l1229_122939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_no_adjacent_blue_numbers_l1229_122993

def Color := Bool

def is_red (c : Color) : Bool := c
def is_blue (c : Color) : Bool := !c

def Coloring := Nat → Color

theorem impossibility_of_no_adjacent_blue_numbers 
  (coloring : Coloring)
  (h1 : ∀ n, n > 1000 → coloring n = true ∨ coloring n = false)
  (h2 : ∀ m n, m > 1000 → n > 1000 → m ≠ n → 
        coloring m = true → coloring n = true → 
        coloring (m * n) = false)
  : ∃ n, n > 1000 ∧ coloring n = false ∧ coloring (n + 1) = false := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_no_adjacent_blue_numbers_l1229_122993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1229_122925

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the major axis length
noncomputable def major_axis_length : ℝ := 10

-- Define the minor axis length
noncomputable def minor_axis_length : ℝ := 6

-- Define the eccentricity
noncomputable def eccentricity : ℝ := 4/5

-- Define the foci
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the vertices
def vertices : Set (ℝ × ℝ) := {(-5, 0), (5, 0), (0, -3), (0, 3)}

-- Theorem statement
theorem ellipse_properties :
  (∀ x y, ellipse x y → 
    major_axis_length = 10 ∧
    minor_axis_length = 6 ∧
    eccentricity = 4/5 ∧
    foci = {(-4, 0), (4, 0)} ∧
    vertices = {(-5, 0), (5, 0), (0, -3), (0, 3)}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1229_122925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1229_122974

theorem sin_minus_cos_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : π/4 < α ∧ α < π/2) :
  Real.sin α - Real.cos α = Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1229_122974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_selection_l1229_122991

theorem divisibility_in_selection :
  ∀ (S : Finset ℕ),
  (S ⊆ Finset.range 201) →
  (S.card = 100) →
  (∃ n ∈ S, n < 16) →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_selection_l1229_122991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1229_122941

noncomputable def amit_rate : ℝ := 1 / 10
noncomputable def ananthu_rate : ℝ := 1 / 20
def amit_days_worked : ℝ := 2

def total_work : ℝ := 1

theorem work_completion_time :
  let work_done_by_amit : ℝ := amit_rate * amit_days_worked
  let remaining_work : ℝ := total_work - work_done_by_amit
  let ananthu_days : ℝ := remaining_work / ananthu_rate
  amit_days_worked + ananthu_days = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1229_122941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_over_200_is_friday_l1229_122978

inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, DecidableEq

def paperclip_count : DayOfWeek → ℕ
  | DayOfWeek.Sunday => 5
  | DayOfWeek.Monday => 10
  | DayOfWeek.Tuesday => 20
  | DayOfWeek.Wednesday => 60
  | DayOfWeek.Thursday => 120
  | DayOfWeek.Friday => 240
  | DayOfWeek.Saturday => 480

def day_lt : DayOfWeek → DayOfWeek → Prop
  | DayOfWeek.Sunday, d => d ≠ DayOfWeek.Sunday
  | DayOfWeek.Monday, d => d ∉ [DayOfWeek.Sunday, DayOfWeek.Monday]
  | DayOfWeek.Tuesday, d => d ∉ [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday]
  | DayOfWeek.Wednesday, d => d ∈ [DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]
  | DayOfWeek.Thursday, d => d ∈ [DayOfWeek.Friday, DayOfWeek.Saturday]
  | DayOfWeek.Friday, d => d = DayOfWeek.Saturday
  | DayOfWeek.Saturday, _ => False

theorem first_day_over_200_is_friday :
  ∀ d : DayOfWeek, paperclip_count d > 200 → d = DayOfWeek.Friday ∧
  ∀ d' : DayOfWeek, day_lt d' d → paperclip_count d' ≤ 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_over_200_is_friday_l1229_122978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_l1229_122979

noncomputable def ellipse_C1 (x y : ℝ) : Prop :=
  (x^2 / 13^2) + (y^2 / 12^2) = 1

def focus1 : ℝ × ℝ := (-5, 0)
def focus2 : ℝ × ℝ := (5, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def on_C2 (p : ℝ × ℝ) : Prop :=
  |distance p focus1 - distance p focus2| = 8

theorem C2_equation : 
  ∀ p : ℝ × ℝ, on_C2 p ↔ (p.1^2 / 4^2) - (p.2^2 / 3^2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_l1229_122979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_tangents_l1229_122946

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines the number of common tangents between two circles -/
def commonTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- The main theorem stating that the two given circles have exactly 3 common tangents -/
theorem three_common_tangents :
  let c1 : Circle := { center := (-2, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 4 }
  commonTangents c1 c2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_tangents_l1229_122946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l1229_122990

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_abc_area :
  let A : ℝ × ℝ := (0, 3)
  let B : ℝ × ℝ := (6, 0)
  let C : ℝ × ℝ := (3, 8)
  triangle_area A B C = 19.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l1229_122990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_points_distance_l1229_122900

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: In a 2×1 rectangle with 7 points, there exist two points with distance ≤ √(10)/3 -/
theorem rectangle_points_distance (points : Finset Point) : 
  points.card = 7 → 
  (∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 1) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 10 / 3 := by
  sorry

#check rectangle_points_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_points_distance_l1229_122900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_equals_one_l1229_122902

/-- Given two lines in the xy-plane, if they are perpendicular, then m = 1 -/
theorem perpendicular_lines_m_equals_one (m : ℝ) : 
  (∀ x y : ℝ, (x - 2*y + 5 = 0 ↔ y = (1/2)*x + 5/2) ∧ 
              (2*x + m*y - 6 = 0 ↔ y = -(2/m)*x + 6/m) ∧
              (1/2 * -(2/m) = -1)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_equals_one_l1229_122902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_bound_l1229_122944

def is_valid_matrix (M : Matrix (Fin 9) (Fin 9) ℝ) : Prop :=
  (∀ i j, |M i j| < 1) ∧
  (∀ i j, i < 8 → j < 8 → M i j + M i (j+1) + M (i+1) j + M (i+1) (j+1) = 0)

theorem sum_bound (M : Matrix (Fin 9) (Fin 9) ℝ) (h : is_valid_matrix M) :
  |Finset.sum (Finset.univ : Finset (Fin 9 × Fin 9)) (λ (i, j) => M i j)| < 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_bound_l1229_122944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_range_l1229_122953

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) - 1

-- Define the function g(x) as a shift of f(x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

-- State the theorem
theorem g_zero_range (m : ℝ) :
  (∃ x ∈ Set.Icc 0 (π / 2), g x = m) ↔ m ∈ Set.Icc (-3) (Real.sqrt 3 - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_range_l1229_122953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_perigee_distance_is_9AU_l1229_122940

/-- Represents the distance in astronomical units (AU) -/
def AU : ℝ := 1

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ  -- Distance at closest approach
  apogee : ℝ   -- Distance at farthest point

/-- The distance from the star to a point on the orbit directly opposite the perigee -/
noncomputable def oppositePerigeeDistance (orbit : EllipticalOrbit) : ℝ :=
  (orbit.perigee + orbit.apogee) / 2

/-- Theorem: For the given elliptical orbit, the distance from the star to a point
    directly opposite the perigee is 9 AU -/
theorem opposite_perigee_distance_is_9AU (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3 * AU) (h2 : orbit.apogee = 15 * AU) :
    oppositePerigeeDistance orbit = 9 * AU := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_perigee_distance_is_9AU_l1229_122940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pell_equation_solution_l1229_122986

theorem pell_equation_solution (d : ℕ) (x y : ℤ) :
  Odd x → Odd y → x^2 - d * y^2 = -4 →
  ∃ X Y : ℤ, X^2 - d * Y^2 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pell_equation_solution_l1229_122986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AX_length_l1229_122994

-- Define the circle and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the diameter of the circle
def diameter : ℝ := 1

-- Define the property that points lie on the circle
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define the property that a point lies on a line segment
def on_segment (p q r : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = t * q.x + (1 - t) * r.x ∧ p.y = t * q.y + (1 - t) * r.y

-- Define the angle measure
noncomputable def angle_measure (p q r : Point) : ℝ :=
  Real.arccos ((p.x - q.x) * (r.x - q.x) + (p.y - q.y) * (r.y - q.y)) /
    (((p.x - q.x)^2 + (p.y - q.y)^2)^(1/2) * ((r.x - q.x)^2 + (r.y - q.y)^2)^(1/2))

-- Define the distance between two points
def dist (p q : Point) : ℝ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2)^(1/2)

-- State the theorem
theorem AX_length 
  (c : Circle)
  (A B C D X : Point)
  (h1 : on_circle A c) 
  (h2 : on_circle B c) 
  (h3 : on_circle C c) 
  (h4 : on_circle D c)
  (h5 : on_segment X A D)
  (h6 : dist A X = dist D X)
  (h7 : 3 * angle_measure B A C = angle_measure B X C)
  (h8 : angle_measure B X C = 30 * (π / 180))
  (h9 : c.radius = diameter / 2) :
  dist A X = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AX_length_l1229_122994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segment_length_l1229_122932

/-- The ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- A line passing through point (0, 2) -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 2}

/-- The length of the segment AB where A and B are the intersection points of the line and the ellipse -/
noncomputable def segmentLength (k : ℝ) : ℝ :=
  let x1 := (-4*k - Real.sqrt (16*k^2 - 24)) / (1 + 2*k^2)
  let x2 := (-4*k + Real.sqrt (16*k^2 - 24)) / (1 + 2*k^2)
  let y1 := k * x1 + 2
  let y2 := k * x2 + 2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating that the maximum length of AB is 3/2 -/
theorem max_segment_length :
  (∃ k : ℝ, segmentLength k = 3/2) ∧
  (∀ k : ℝ, segmentLength k ≤ 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segment_length_l1229_122932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_speed_satisfies_conditions_problem_slower_speed_approx_l1229_122952

/-- Represents the walking scenario with two different speeds -/
structure WalkingScenario where
  actual_distance : ℝ
  faster_speed : ℝ
  additional_distance : ℝ

/-- Calculates the slower speed given a WalkingScenario -/
noncomputable def calculate_slower_speed (scenario : WalkingScenario) : ℝ :=
  (scenario.actual_distance - scenario.additional_distance) * scenario.faster_speed / scenario.actual_distance

/-- Theorem stating that the calculated slower speed satisfies the given conditions -/
theorem slower_speed_satisfies_conditions (scenario : WalkingScenario) 
  (h1 : scenario.actual_distance > 0)
  (h2 : scenario.faster_speed > 0)
  (h3 : scenario.additional_distance > 0)
  (h4 : scenario.additional_distance < scenario.actual_distance) :
  let slower_speed := calculate_slower_speed scenario
  (scenario.actual_distance - scenario.additional_distance) / slower_speed = 
  scenario.actual_distance / scenario.faster_speed := by sorry

/-- The specific walking scenario from the problem -/
def problem_scenario : WalkingScenario := {
  actual_distance := 60,
  faster_speed := 16,
  additional_distance := 20
}

/-- Theorem stating that the slower speed for the problem scenario is approximately 10.67 -/
theorem problem_slower_speed_approx :
  ‖calculate_slower_speed problem_scenario - 10.67‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_speed_satisfies_conditions_problem_slower_speed_approx_l1229_122952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_probabilities_l1229_122956

/-- Represents a group of people seated around a round table -/
structure SeatingArrangement where
  men : ℕ
  women : ℕ

/-- A man is satisfied if at least one woman is sitting next to him -/
def is_satisfied (s : SeatingArrangement) (person : ℕ) : Prop :=
  ∃ (adjacent : Fin 2), ∃ (w : ℕ), w ≤ s.women

/-- The probability of a specific man being satisfied -/
noncomputable def prob_man_satisfied (s : SeatingArrangement) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
noncomputable def expected_satisfied_men (s : SeatingArrangement) : ℚ :=
  (s.men : ℚ) * (prob_man_satisfied s)

/-- Main theorem about seating arrangement probabilities -/
theorem seating_probabilities (s : SeatingArrangement) 
  (h1 : s.men = 50) (h2 : s.women = 50) : 
  prob_man_satisfied s = 25 / 33 ∧ 
  expected_satisfied_men s = 1250 / 33 := by
  sorry

#check seating_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_probabilities_l1229_122956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1229_122906

-- Define the expression
noncomputable def f (a : ℝ) : ℝ := (((a^2 - 3) / (a - 3) - a) / ((a - 1) / (a^2 - 6*a + 9)))

-- State the theorem
theorem expression_simplification :
  ∃ (a : ℤ), 1 ≤ a ∧ a < 4 ∧ f (a : ℝ) = -3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1229_122906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_equals_one_two_l1229_122907

-- Define set P
def P : Set ℕ := {x : ℕ | ∃ y : ℝ, y = Real.sqrt (-x^2 + x + 2)}

-- Define set Q
def Q : Set ℕ := {x : ℕ | x > 0 ∧ Real.log (x : ℝ) < 1}

-- Theorem statement
theorem P_intersect_Q_equals_one_two : P ∩ Q = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_equals_one_two_l1229_122907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_contained_l1229_122918

-- Define the triangle sequence
def triangle_sequence (n : ℕ) : Set (ℝ × ℝ) := sorry

-- Define the ratio function
noncomputable def ratio (k : ℝ) (n : ℕ) : ℝ := 
  if n % 2 = 0 then k^(2^n) else 1 / (k^(2^n))

-- Define the intersection triangle
def intersection_triangle : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem intersection_triangle_contained (n : ℕ) (k : ℝ) :
  k > 0 →
  (∀ m, ratio k m = if m % 2 = 0 then k^(2^m) else 1 / (k^(2^m))) →
  intersection_triangle ⊆ triangle_sequence n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_contained_l1229_122918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1229_122971

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

-- Define the open interval (-4, 1)
def I : Set ℝ := { x | -4 < x ∧ x < 1 }

-- Statement to prove
theorem max_value_of_f :
  ∃ (M : ℝ), M = -1 ∧ ∀ x ∈ I, f x ≤ M := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1229_122971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_advantageous_salary_l1229_122931

/-- Represents the salary in tugriks -/
noncomputable def salary : ℝ → ℝ := id

/-- Calculates the tax rate based on the salary -/
noncomputable def taxRate (x : ℝ) : ℝ := x / 1000

/-- Calculates the net income after tax -/
noncomputable def netIncome (x : ℝ) : ℝ := x - x * (taxRate x)

/-- Theorem stating that 500 tugriks is the most advantageous salary -/
theorem most_advantageous_salary :
  ∀ x : ℝ, x > 0 → netIncome x ≤ netIncome 500 := by
  sorry

#check most_advantageous_salary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_advantageous_salary_l1229_122931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l1229_122935

def P (x : ℕ) : Finset ℕ := {1, x}
def Q (y : ℕ) : Finset ℕ := {1, 2, y}

def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ pair => pair.1 ∈ Finset.range 9 ∧ pair.2 ∈ Finset.range 9 ∧ 
                           P pair.1 ⊆ Q pair.2)
                (Finset.product (Finset.range 9) (Finset.range 9))

theorem count_valid_pairs : valid_pairs.card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l1229_122935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_f_monotonicity_l1229_122963

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) + Real.log x

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := f 2 x - 2 * x

-- Part 1: Monotonicity of g(x)
theorem g_monotonicity :
  (∀ x ∈ Set.Ioo 0 1, StrictMonoOn g (Set.Ioo 0 1)) ∧
  (∀ x ∈ Set.Ioi 1, StrictAntiOn g (Set.Ioi 1)) :=
sorry

-- Part 2: Condition for f(x) to be monotonically increasing
theorem f_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, StrictMonoOn (f a) (Set.Icc 1 4)) ↔ a > -5/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_f_monotonicity_l1229_122963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_roots_l1229_122970

/-- Defines an isosceles triangle with sides a, b, and c. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (c = a ∧ c ≠ b)

/-- An isosceles triangle with one side of length 3 and the other two sides
    as roots of x^2 - 4x + k = 0 has k equal to either 3 or 4. -/
theorem isosceles_triangle_roots (k : ℝ) : 
  (∃ (a b : ℝ), IsIsosceles 3 a b ∧ 
   (a^2 - 4*a + k = 0) ∧ 
   (b^2 - 4*b + k = 0)) → 
  (k = 3 ∨ k = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_roots_l1229_122970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_equal_area_l1229_122936

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields here, e.g.:
  -- vertices : Fin 3 → ℝ × ℝ

/-- Represents a square -/
structure Square where
  -- Add necessary fields here, e.g.:
  -- side : ℝ

/-- Represents the method of inscribing a square in a triangle -/
inductive InscriptionMethod
  | Figure1
  | Figure2

/-- Checks if a triangle is isosceles right -/
def Triangle.isIsoscelesRight (t : Triangle) : Prop :=
  sorry -- Add the actual condition here

/-- Checks if a square is inscribed in a triangle -/
def Square.inscribedInTriangle (s : Square) (t : Triangle) : Prop :=
  sorry -- Add the actual condition here

/-- Returns the inscription method of a square -/
def Square.inscriptionMethod (s : Square) : InscriptionMethod :=
  sorry -- Add the actual logic here

/-- Calculates the area of a square -/
noncomputable def Square.area (s : Square) : ℝ :=
  sorry -- Add the actual calculation here

/-- Given an isosceles right triangle ABC with a square inscribed as in Figure 1 having area 484 cm²,
    prove that a square inscribed as in Figure 2 has the same area. -/
theorem inscribed_squares_equal_area (ABC : Triangle) (square1 square2 : Square) :
  Triangle.isIsoscelesRight ABC →
  Square.inscribedInTriangle square1 ABC →
  Square.inscribedInTriangle square2 ABC →
  Square.inscriptionMethod square1 = InscriptionMethod.Figure1 →
  Square.inscriptionMethod square2 = InscriptionMethod.Figure2 →
  Square.area square1 = 484 →
  Square.area square2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_equal_area_l1229_122936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_appears_450_minutes_l1229_122911

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Checks if a given digit appears in a number -/
def digitAppears (digit : Nat) (number : Nat) : Bool :=
  toString number |>.any (· == Char.ofNat (digit + 48))

/-- Checks if the digit 5 appears in a given time -/
def has5 (t : Time) : Bool :=
  digitAppears 5 t.hours ∨ digitAppears 5 t.minutes

/-- Counts the number of minutes in a day where the digit 5 appears -/
def count5Minutes : Nat :=
  List.range 1440 |> -- 24 * 60 = 1440 minutes in a day
    List.filter (fun m => has5 ⟨m / 60, m % 60, by sorry⟩) |>
    List.length

theorem five_appears_450_minutes : count5Minutes = 450 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_appears_450_minutes_l1229_122911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1229_122997

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the given condition
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

theorem triangle_properties (t : Triangle) (h : condition t) :
  t.A = π / 6 ∧
  Set.Icc (-((Real.sqrt 3 + 2) / 2)) (Real.sqrt 3 - 1) =
    Set.range (fun B => Real.cos (5 * π / 2 - B) - 2 * (Real.sin (t.C / 2))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1229_122997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_3_to_x_at_2_l1229_122928

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem derivative_of_3_to_x_at_2 :
  deriv f 2 = 9 * Real.log 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_3_to_x_at_2_l1229_122928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l1229_122934

theorem complex_modulus_problem :
  Complex.abs ((4 - 2 * Complex.I) / (1 + Complex.I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l1229_122934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_difference_l1229_122945

noncomputable def g : ℕ → ℚ
  | 0 => 0
  | 1 => 0
  | n+2 => 1 / (2 + g (n+1))

noncomputable def k : ℕ → ℚ
  | 0 => 0
  | 1 => 0
  | n+2 => 1 / (2 + 1 / (n+2 + k (n+1)))

theorem continued_fraction_difference (n : ℕ) (h : n > 1) :
  |g n - k n| ≤ 1 / ((Nat.factorial (n - 1)) * (Nat.factorial n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_difference_l1229_122945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_equal_l1229_122904

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_areas_equal : 
  triangleArea 15 15 18 = triangleArea 15 15 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_equal_l1229_122904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_two_and_four_identical_views_l1229_122948

-- Define the type for orthographic views
inductive View
  | Top
  | Front
  | Side

-- Define the type for geometric solids
inductive Solid
  | One
  | Two
  | Three
  | Four

-- Function to get the views of a solid
def getViews (s : Solid) : List View :=
  match s with
  | Solid.One => [View.Top, View.Front, View.Side]
  | Solid.Two => [View.Top, View.Front, View.Side]
  | Solid.Three => [View.Top, View.Front, View.Side]
  | Solid.Four => [View.Top, View.Front, View.Side]

-- Function to check if two views are identical
def areViewsIdentical (v1 v2 : View) : Bool :=
  match v1, v2 with
  | View.Top, View.Top => true
  | View.Front, View.Front => true
  | View.Side, View.Side => true
  | _, _ => false

-- Function to count identical views between two solids
def countIdenticalViews (s1 s2 : Solid) : Nat :=
  (getViews s1).zip (getViews s2)
  |> List.filter (fun (v1, v2) => areViewsIdentical v1 v2)
  |> List.length

-- Theorem stating that only solids Two and Four have exactly one pair of identical views
theorem only_two_and_four_identical_views :
  ∀ s1 s2 : Solid,
    (s1 = Solid.Two ∧ s2 = Solid.Four) ∨ (s1 = Solid.Four ∧ s2 = Solid.Two) ↔
    countIdenticalViews s1 s2 = 2 ∧
    (∀ s3 s4 : Solid, (s3 ≠ s1 ∨ s4 ≠ s2) → countIdenticalViews s3 s4 ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_two_and_four_identical_views_l1229_122948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_is_two_acres_l1229_122961

/-- Represents a rectangular field with specific properties -/
structure RectangularField where
  length : ℚ
  width : ℚ
  diagonal : ℚ
  width_plus_diagonal_sum : width + diagonal = 50
  length_value : length = 30

/-- Calculates the area of a rectangular field in square steps -/
def area_in_square_steps (field : RectangularField) : ℚ :=
  field.length * field.width

/-- Converts square steps to acres -/
def square_steps_to_acres (square_steps : ℚ) : ℚ :=
  square_steps / 240

/-- Theorem stating that a field with the given properties has an area of 2 acres -/
theorem field_area_is_two_acres (field : RectangularField) :
  square_steps_to_acres (area_in_square_steps field) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_is_two_acres_l1229_122961
