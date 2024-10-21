import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_lightning_l1249_124986

-- Define the constants
def time_elapsed : ℚ := 12
def speed_of_sound : ℚ := 1050
def feet_per_mile : ℚ := 5280

-- Define the function to calculate distance
def calculate_distance (time : ℚ) (speed : ℚ) : ℚ :=
  time * speed

-- Define the function to convert feet to miles
def feet_to_miles (feet : ℚ) (feet_per_mile : ℚ) : ℚ :=
  feet / feet_per_mile

-- Define the function to round to nearest half-mile
noncomputable def round_to_half_mile (miles : ℚ) : ℚ :=
  ⌊(miles * 2 + 1/2)⌋ / 2

-- Theorem statement
theorem distance_from_lightning :
  round_to_half_mile (feet_to_miles (calculate_distance time_elapsed speed_of_sound) feet_per_mile) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_lightning_l1249_124986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_arcsin_range_l1249_124980

theorem cot_arcsin_range (a : ℝ) :
  (∃ x : ℝ, Real.arctan (Real.sqrt (1 - x^2) / x) = Real.sqrt (a^2 - x^2)) →
  a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_arcsin_range_l1249_124980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l1249_124930

theorem roots_of_equation (x : ℝ) :
  (3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 7) ↔
  (x = ((7 + Real.sqrt 13) / 6)^2 ∨ x = ((7 - Real.sqrt 13) / 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l1249_124930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1249_124961

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  3 * (Real.sin t.C)^2 + 8 * (Real.sin t.A)^2 = 11 * Real.sin t.A * Real.sin t.C ∧
  t.c < 2 * t.a ∧
  t.a = t.c

-- Define the additional conditions for the second part
def additionalConditions (t : Triangle) : Prop :=
  (1/2) * t.a * t.b * Real.sin t.C = 8 * Real.sqrt 15 ∧
  Real.sin t.B = Real.sqrt 15 / 4

-- Define the median
noncomputable def median (t : Triangle) : Real := 
  Real.sqrt ((1/4) * (2 * t.a^2 + 2 * t.c^2 - t.b^2))

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : additionalConditions t) : 
  median t = 8 ∨ median t = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1249_124961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_ngon_l1249_124984

/-- A convex n-gon with colored vertices, edges, and diagonals. -/
structure ColoredNgon (n : ℕ) where
  (color : Fin n → Fin n)  -- Color assignment function
  (vertex_color : Fin n → Fin n)  -- Color of each vertex
  (segment_color : Fin n → Fin n → Fin n)  -- Color of each segment (including edges and diagonals)

/-- The coloring satisfies the given conditions. -/
def valid_coloring (ngon : ColoredNgon n) : Prop :=
  -- Condition 1: Any two segments emerging from the same vertex must be different colors
  ∀ i j k : Fin n, j ≠ k → ngon.segment_color i j ≠ ngon.segment_color i k
  ∧
  -- Condition 2: The color of any vertex must be different from the color of any segment emerging from it
  ∀ i j : Fin n, i ≠ j → ngon.vertex_color i ≠ ngon.segment_color i j

/-- The minimum number of colors required is n. -/
theorem min_colors_ngon (n : ℕ) (hn : n > 0) :
  (∃ (ngon : ColoredNgon n), valid_coloring ngon) ∧
  (∀ (ngon : ColoredNgon n), valid_coloring ngon → Fintype.card (Set.range ngon.color) = n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_ngon_l1249_124984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_T6_l1249_124913

def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def T (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).prod (λ i => a (i + 1))

theorem geometric_progression_T6 (a : ℕ → ℝ) :
  geometric_progression a →
  (∀ n : ℕ, a n > 0) →
  Real.sqrt (a 3 * a 4) = ∫ x in Set.Icc (Real.exp (-1)) (Real.exp 1), 1 / x →
  T a 6 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_T6_l1249_124913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_seventh_is_two_sum_of_first_2007_l1249_124952

/-- Defines the sequence based on the given rule -/
def mySequence : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => 
  let k := (n + 2).sqrt
  if n + 2 = k * (k + 1) / 2 then 1 else 2

/-- The 2007th number in the sequence is 2 -/
theorem two_thousand_seventh_is_two : mySequence 2006 = 2 := by sorry

/-- The sum of the first 2007 numbers in the sequence is 3952 -/
theorem sum_of_first_2007 : (Finset.range 2007).sum (fun i => mySequence i) = 3952 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_seventh_is_two_sum_of_first_2007_l1249_124952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1249_124924

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the function g
def g (a x : ℝ) : ℝ := f x - (4 + 2*a)*x + 2

-- State the theorem
theorem quadratic_function_properties :
  (∀ x : ℝ, f x = x^2 + 2*x) ∧
  (f (-3) = f 1) ∧
  (f 0 = 0) ∧
  (∀ a : ℝ, ∀ x ∈ Set.Icc 1 2,
    (a ≤ 0 → (
      IsMinOn (g a) (Set.Icc 1 2) (g a 1) ∧
      IsMaxOn (g a) (Set.Icc 1 2) (g a 2)
    )) ∧
    (0 < a ∧ a < 1/2 → (
      IsMinOn (g a) (Set.Icc 1 2) (g a (1 + a)) ∧
      IsMaxOn (g a) (Set.Icc 1 2) (g a 2)
    )) ∧
    (a = 1/2 → (
      IsMinOn (g a) (Set.Icc 1 2) (g a (3/2)) ∧
      IsMaxOn (g a) (Set.Icc 1 2) (g a 1)
    )) ∧
    (1/2 < a ∧ a < 1 → (
      IsMinOn (g a) (Set.Icc 1 2) (g a (1 + a)) ∧
      IsMaxOn (g a) (Set.Icc 1 2) (g a 1)
    )) ∧
    (1 ≤ a → (
      IsMinOn (g a) (Set.Icc 1 2) (g a 2) ∧
      IsMaxOn (g a) (Set.Icc 1 2) (g a 1)
    ))
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1249_124924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_upper_bound_of_k_l1249_124982

/-- The function f(x) defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x * Real.cos x + Real.sin x * (Real.sqrt 3 * Real.cos x)) + 1

/-- Theorem stating the range of f(x) -/
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc 1 4 := by sorry

/-- Theorem stating the upper bound of k -/
theorem upper_bound_of_k (k : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2),
   ∀ α ∈ Set.Icc (Real.pi / 12) (Real.pi / 3),
   k * Real.sqrt (1 + Real.sin (2 * α)) - Real.sin (2 * α) ≤ f x + 1) →
  k ≤ 5 * Real.sqrt 6 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_upper_bound_of_k_l1249_124982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1249_124900

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a = c - 2 * a * Real.cos B →
  c = 5 →
  3 * a = 2 * b →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1249_124900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_parallelogram_area_theorem_l1249_124968

/-- Configuration of connecting division points -/
inductive Configuration
  | A
  | B

/-- Represents a parallelogram with sides divided into equal parts -/
structure DividedParallelogram where
  area : ℝ
  n : ℕ
  m : ℕ

/-- Calculates the area of each small parallelogram based on the configuration -/
noncomputable def smallParallelogramArea (p : DividedParallelogram) (config : Configuration) : ℝ :=
  match config with
  | Configuration.A => p.area / (p.m * p.n + 1)
  | Configuration.B => p.area / (p.m * p.n - 1)

/-- Theorem statement for the area of small parallelograms -/
theorem small_parallelogram_area_theorem (p : DividedParallelogram) 
  (h : p.area = 1) (hn : p.n > 0) (hm : p.m > 0) :
  (smallParallelogramArea p Configuration.A = 1 / (p.m * p.n + 1)) ∧
  (smallParallelogramArea p Configuration.B = 1 / (p.m * p.n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_parallelogram_area_theorem_l1249_124968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_garden_area_l1249_124996

/-- Represents a rectangular garden with length and width -/
structure Garden where
  length : ℚ
  width : ℚ

/-- Calculates the area of a garden -/
def Garden.area (g : Garden) : ℚ := g.length * g.width

/-- The original garden -/
def original_garden : Garden :=
  { length := 30
    width := 30 / 2 }

/-- The new garden after extension -/
def new_garden : Garden :=
  { length := original_garden.length + 10
    width := original_garden.width + 5 }

/-- Theorem: The area of the new garden is 800 square meters -/
theorem new_garden_area : new_garden.area = 800 := by
  -- Unfold definitions
  unfold Garden.area
  unfold new_garden
  unfold original_garden
  -- Simplify
  simp
  -- Prove equality
  norm_num

#eval new_garden.area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_garden_area_l1249_124996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l1249_124985

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- Calculate the x-intercept of a line -/
noncomputable def Line.x_intercept (l : Line) : ℝ :=
  -l.y_intercept / l.slope

/-- Calculate the area of a triangle formed by a line and the coordinate axes -/
noncomputable def triangle_area (l : Line) : ℝ :=
  (1/2) * abs (l.x_intercept * l.y_intercept)

/-- The main theorem -/
theorem line_equation_theorem (l : Line) :
  l.contains (-5) (-4) ∧
  l.x_intercept ≠ 0 ∧
  l.y_intercept ≠ 0 ∧
  triangle_area l = 5 →
  (l.slope = 8/5 ∧ l.y_intercept = -4) ∨
  (l.slope = 2/5 ∧ l.y_intercept = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l1249_124985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1249_124912

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 8*y + 4 = 0

-- Define the centers and radii of the circles
def center_C1 : ℝ × ℝ := (-1, 0)
def radius_C1 : ℝ := 1
def center_C2 : ℝ × ℝ := (2, -4)
def radius_C2 : ℝ := 4

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((2 - (-1))^2 + (-4 - 0)^2)

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius_C1 + radius_C2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1249_124912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_x_l1249_124920

/-- Given that q = (69.28 × x) / 0.03 and q ≈ 9.237333333333334, prove that x ≈ 0.004 -/
theorem approximate_x (q x : ℝ) (h1 : q = (69.28 * x) / 0.03) 
  (h2 : abs (q - 9.237333333333334) < 0.000000001) : abs (x - 0.004) < 0.000000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_x_l1249_124920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1249_124934

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 6) 
  (hb : b % 15 = 8) 
  (hc : c % 15 = 11) : 
  (a + b + c) % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1249_124934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_square_l1249_124929

/-- Represents the number composed of m 3's followed by n 6's -/
def specialNumber (m n : ℕ) : ℕ :=
  3 * (10^n - 1) / 9 * 10^n + 6 * (10^n - 1) / 9

/-- The theorem stating that (1, 1) is the only solution -/
theorem unique_special_square :
  ∀ m n : ℕ, (∃ k : ℕ, specialNumber m n = k^2) ↔ (m = 1 ∧ n = 1) :=
sorry

#check unique_special_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_square_l1249_124929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l1249_124972

theorem trigonometric_equality (a b m n : ℝ) (α : ℝ) 
  (h1 : a * Real.sin α + b * Real.cos α = m)
  (h2 : b * Real.tan α - n / Real.cos α = a) :
  a^2 + b^2 = m^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l1249_124972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_of_given_grid_l1249_124926

/-- Represents a square grid -/
structure SquareGrid where
  size : Nat
  shaded : Nat

/-- Calculates the percentage of shaded area in a square grid -/
noncomputable def shadedPercentage (grid : SquareGrid) : Real :=
  (grid.shaded : Real) / ((grid.size * grid.size) : Real) * 100

/-- The given 7x7 grid with 25 shaded squares -/
def givenGrid : SquareGrid :=
  { size := 7, shaded := 25 }

theorem shaded_percentage_of_given_grid :
  shadedPercentage givenGrid = (25 : Real) / (49 : Real) * 100 := by
  unfold shadedPercentage givenGrid
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_of_given_grid_l1249_124926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_given_points_and_tangent_intersection_l1249_124939

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- The y-axis -/
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- Tangent line to a circle at a point -/
def TangentLine (c : Circle) (p : Point) : Set (ℝ × ℝ) := sorry

/-- Intersection of two sets -/
def Intersect (s t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := {p | p ∈ s ∧ p ∈ t}

/-- Area of a circle -/
noncomputable def area (c : Circle) : ℝ := Real.pi * c.radius ^ 2

theorem circle_area_given_points_and_tangent_intersection (ω : Circle) (A B : Point) :
  A ∈ {p : ℝ × ℝ | (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2} →
  B ∈ {p : ℝ × ℝ | (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2} →
  A = (7, 15) →
  B = (13, 9) →
  (Intersect (TangentLine ω A) (TangentLine ω B)) ∩ yAxis ≠ ∅ →
  area ω = 237.62 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_given_points_and_tangent_intersection_l1249_124939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_D_value_l1249_124915

/-- Triangle DEF with sides DE and DF -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  D : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.DE * t.DF * Real.sin t.D

/-- The geometric mean between DE and DF -/
noncomputable def geometric_mean (t : Triangle) : ℝ := Real.sqrt (t.DE * t.DF)

theorem sin_D_value (t : Triangle) 
  (h_area : area t = 100)
  (h_gm : geometric_mean t = 15) :
  Real.sin t.D = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_D_value_l1249_124915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_specific_triangle_l1249_124911

-- Define the triangle DEF
structure RightTriangle where
  D : ℝ
  E : ℝ
  F : ℝ
  is_right_angle : F = 90
  angle_D : D = 45
  side_DF : ℝ
  side_DF_length : side_DF = 8

-- Define the incircle radius
noncomputable def incircle_radius (t : RightTriangle) : ℝ :=
  4 - 2 * Real.sqrt 2

-- Theorem statement
theorem incircle_radius_of_specific_triangle (t : RightTriangle) :
  incircle_radius t = 4 - 2 * Real.sqrt 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_specific_triangle_l1249_124911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_line_segment_length_l1249_124955

/-- Represents a trapezoid with bases of lengths a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- Represents a line segment MN in the trapezoid -/
noncomputable def line_segment (t : Trapezoid) : ℝ := 2 * t.a * t.b / (t.a + t.b)

/-- 
Theorem: In a trapezoid with bases of lengths a and b, if a line parallel 
to the bases is drawn through the intersection point of the diagonals, 
intersecting the non-parallel sides at M and N, then the length of MN 
is equal to (2ab)/(a + b).
-/
theorem trapezoid_line_segment_length (t : Trapezoid) : 
  ∃ (MN : ℝ), MN = line_segment t := by
  -- The proof goes here
  sorry

#check trapezoid_line_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_line_segment_length_l1249_124955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt_reciprocal_equality_condition_l1249_124953

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x ≥ 5 * (2 : ℝ)^(1/3) :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x = 5 * (2 : ℝ)^(1/3) ↔ x = 2^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt_reciprocal_equality_condition_l1249_124953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removing_six_maximizes_pairs_l1249_124928

def original_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def count_pairs (l : List Int) : Nat :=
  (l.filterMap (fun x => 
    if x ≠ 6 ∧ (12 - x) ∈ l ∧ x ≠ (12 - x) then some x else none
  )).length

def remove_element (l : List Int) (n : Int) : List Int :=
  l.filter (fun x => x ≠ n)

theorem removing_six_maximizes_pairs :
  ∀ n ∈ original_list, n ≠ 6 →
    count_pairs (remove_element original_list 6) ≥ 
    count_pairs (remove_element original_list n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removing_six_maximizes_pairs_l1249_124928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_last_theorem_extension_l1249_124965

-- Define Fermat's Last Theorem for a given exponent
def fermatLastTheorem (n : ℕ) : Prop :=
  ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x^n + y^n ≠ z^n

-- Define what it means for Fermat's Last Theorem to hold for all prime exponents
def fermatLastTheoremForPrimes : Prop :=
  ∀ p : ℕ, Nat.Prime p → fermatLastTheorem p

-- The theorem to be proved
theorem fermat_last_theorem_extension :
  fermatLastTheoremForPrimes →
  ∀ n : ℕ, n > 2 → ¬Nat.Prime n → fermatLastTheorem n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_last_theorem_extension_l1249_124965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_in_225_divisors_l1249_124905

def divisors_of_225 : List Nat :=
  [3, 5, 9, 15, 25, 45, 75, 225]

-- A function to check if two numbers have a common factor greater than 1
def has_common_factor (a b : Nat) : Bool :=
  (Nat.gcd a b) > 1

-- A predicate to check if a list represents a valid circular arrangement
def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i, has_common_factor (arr[i % arr.length]!) (arr[(i + 1) % arr.length]!)

theorem adjacent_sum_in_225_divisors :
  ∃ (arr : List Nat), 
    arr.toFinset = divisors_of_225.toFinset ∧ 
    is_valid_arrangement arr ∧
    ∃ i, arr[i % arr.length]! + arr[(i + 1) % arr.length]! = 120 := by
  sorry

#eval divisors_of_225

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_in_225_divisors_l1249_124905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_is_fifty_cents_l1249_124998

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the problem parameters -/
structure ProblemParameters where
  boxDim : BoxDimensions
  totalVolume : ℝ
  totalCost : ℝ

/-- Calculates the number of boxes needed -/
noncomputable def numberOfBoxes (p : ProblemParameters) : ℝ :=
  p.totalVolume / boxVolume p.boxDim

/-- Calculates the cost per box -/
noncomputable def costPerBox (p : ProblemParameters) : ℝ :=
  p.totalCost / numberOfBoxes p

/-- The main theorem stating that the cost per box is $0.50 -/
theorem cost_per_box_is_fifty_cents (p : ProblemParameters) 
  (h1 : p.boxDim = { length := 20, width := 20, height := 15 })
  (h2 : p.totalVolume = 3060000)
  (h3 : p.totalCost = 255) : 
  costPerBox p = 0.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_is_fifty_cents_l1249_124998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_property_l1249_124932

theorem absolute_value_property (x : ℝ) :
  (abs x + abs (abs x - 1) = 1) → ((x + 1) * (x - 1) ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_property_l1249_124932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1249_124943

def range : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 100) (Finset.range 101)

def multiples_of_four : Finset ℕ := Finset.filter (λ n => n % 4 = 0) range

def probability_at_least_one_multiple_of_four : ℚ :=
  1 - (1 - (multiples_of_four.card : ℚ) / (range.card : ℚ))^2

theorem probability_theorem :
  probability_at_least_one_multiple_of_four = 7/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1249_124943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_positive_l1249_124991

theorem negation_of_universal_sin_positive :
  (¬ ∀ x : ℝ, Real.sin x > 0) ↔ (∃ x : ℝ, Real.sin x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_positive_l1249_124991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_odd_increasing_function_l1249_124988

open Set
open Function

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem solution_set_of_odd_increasing_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_incr : is_increasing_on f (Ioi 0))
  (h_f2 : f 2 = 0) :
  {x : ℝ | (f x - f (-x)) / x < 0} = Ioo (-2) 0 ∪ Ioo 0 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_odd_increasing_function_l1249_124988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l1249_124931

def M : ℕ := 2^5 * 3^3 * 5^2 * 7^1

theorem number_of_factors_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l1249_124931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_midpoint_l1249_124958

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define what it means for a point to be on a line
def is_on_line (a b c x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the midpoint property
def is_midpoint (m : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  m.1 = (x₁ + x₂)/2 ∧ m.2 = (y₁ + y₂)/2

-- State the theorem
theorem unique_line_through_midpoint :
  ∃! (a b c : ℝ), 
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      is_on_ellipse x₁ y₁ ∧
      is_on_ellipse x₂ y₂ ∧
      is_on_line a b c x₁ y₁ ∧
      is_on_line a b c x₂ y₂ ∧
      is_on_line a b c M.1 M.2 ∧
      is_midpoint M x₁ y₁ x₂ y₂) ∧
    a = 1 ∧ b = 2 ∧ c = -4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_midpoint_l1249_124958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l1249_124994

/-- The radius of the larger circle -/
def R : ℝ := 5

/-- The radius of the smaller circle -/
noncomputable def r : ℝ := 5 * Real.sqrt 3 / 3

/-- The area of the smaller circle -/
noncomputable def A₁ : ℝ := Real.pi * r^2

/-- The difference between the areas of the larger and smaller circles -/
noncomputable def A₂ : ℝ := Real.pi * R^2 - A₁

/-- Theorem stating that given the conditions, the radius of the smaller circle is 5√3/3 -/
theorem smaller_circle_radius :
  (A₁ + A₂ = Real.pi * R^2) ∧ 
  (A₂ = (A₁ + (A₁ + A₂)) / 2) →
  r = 5 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l1249_124994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_of_parabola_l1249_124903

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The distance from the focus to the directrix of a parabola -/
noncomputable def focus_directrix_distance (parab : Parabola) : ℝ := parab.p

/-- The x-coordinate of the focus of a parabola -/
noncomputable def focus_x (parab : Parabola) : ℝ := parab.p / 2

/-- The y-coordinate of the focus of a parabola -/
noncomputable def focus_y (_parab : Parabola) : ℝ := 0

theorem focus_coordinates_of_parabola (parab : Parabola) 
  (h_dist : focus_directrix_distance parab = 4) :
  (focus_x parab, focus_y parab) = (2, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_of_parabola_l1249_124903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacon_cost_l1249_124981

/-- The cost of a slice of bacon at a fundraiser, given the following conditions:
  * The cost of a stack of pancakes is $4.00
  * 60 stacks of pancakes were sold
  * 90 slices of bacon were sold
  * The total revenue raised was $420
-/
theorem bacon_cost (pancake_cost pancake_stacks bacon_slices total_revenue bacon_cost : ℝ)
  (h1 : pancake_cost = 4)
  (h2 : pancake_stacks = 60)
  (h3 : bacon_slices = 90)
  (h4 : total_revenue = 420)
  (h5 : total_revenue = pancake_cost * pancake_stacks + bacon_slices * bacon_cost) :
  bacon_cost = 2 := by
  sorry

#check bacon_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacon_cost_l1249_124981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1249_124944

-- Define the ages of A, B, and C as integers
variable (A B C : ℤ)

-- Define the condition that the total age of A and B is 12 years more than the total age of B and C
def age_condition (A B C : ℤ) : Prop := A + B = B + C + 12

-- Theorem to prove
theorem age_difference {A B C : ℤ} (h : age_condition A B C) : A - C = 12 := by
  -- Unfold the definition of age_condition
  unfold age_condition at h
  -- Subtract B from both sides of the equation
  have h1 : A = C + 12 := by
    linarith
  -- Subtract C from both sides
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1249_124944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_before_root_l1249_124963

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- State the theorem
theorem f_negative_before_root (x₀ x₁ : ℝ) (h_root : f x₀ = 0) (h_order : 0 < x₁ ∧ x₁ < x₀) : f x₁ < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_before_root_l1249_124963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_and_points_l1249_124951

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x / 2) - Real.sqrt 3 / 2

theorem g_minimum_value_and_points :
  (∀ x : ℝ, g x ≥ -1) ∧
  (∀ k : ℤ, g (2 * Real.pi * ↑k - Real.pi / 6) = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_and_points_l1249_124951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l1249_124946

/-- Represents a vessel with a certain capacity and alcohol concentration -/
structure Vessel where
  capacity : ℚ
  alcohol_concentration : ℚ

/-- Calculates the new alcohol concentration after mixing and adding water -/
def new_concentration (vessels : List Vessel) (final_capacity : ℚ) : ℚ :=
  let total_alcohol := vessels.map (λ v => v.capacity * v.alcohol_concentration) |>.sum
  let total_liquid := vessels.map (λ v => v.capacity) |>.sum
  let water_added := final_capacity - total_liquid
  total_alcohol / final_capacity

/-- Theorem stating that the new concentration of the mixture is 34% -/
theorem mixture_concentration : 
  let vessels := [
    { capacity := 2, alcohol_concentration := 1/5 },
    { capacity := 6, alcohol_concentration := 11/20 },
    { capacity := 4, alcohol_concentration := 7/20 }
  ]
  let final_capacity := 15
  new_concentration vessels final_capacity = 17/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l1249_124946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cary_calorie_deficit_l1249_124956

/-- Calculates the net calorie deficit for Cary's grocery store trip -/
noncomputable def net_calorie_deficit (round_trip_distance : ℝ) (initial_speed : ℝ) (candy_calories : ℝ)
  (weight_calorie_increase : ℝ) (return_speed : ℝ) (base_burn_rate : ℝ) : ℝ :=
  let one_way_distance := round_trip_distance / 2
  let initial_calories := one_way_distance * base_burn_rate
  let return_burn_rate := base_burn_rate * (return_speed / initial_speed) * (1 + weight_calorie_increase)
  let return_calories := one_way_distance * return_burn_rate
  let total_calories := initial_calories + return_calories
  total_calories - candy_calories

/-- Theorem stating that Cary's net calorie deficit is 385 calories -/
theorem cary_calorie_deficit :
  net_calorie_deficit 3 3 200 0.2 4 150 = 385 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cary_calorie_deficit_l1249_124956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_properties_l1249_124921

/-- Given an obtuse triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure ObtuseTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  obtuse : A + B + C = π ∧ max A (max B C) > π/2
  side_angle_relation : b = a * Real.tan B

theorem obtuse_triangle_properties (t : ObtuseTriangle) :
  t.A - t.B = π/2 ∧ 
  -Real.sqrt 2 / 2 < Real.cos (2 * t.B) - Real.sin t.A ∧ 
  Real.cos (2 * t.B) - Real.sin t.A < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_properties_l1249_124921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_rectangle_perimeter_l1249_124908

-- Define the rectangle and its components
noncomputable def rectangle_width (y : ℝ) : ℝ := 2 * y
noncomputable def rectangle_height (y : ℝ) : ℝ := y
noncomputable def square_side (x : ℝ) : ℝ := x

-- Define the congruent rectangle dimensions
noncomputable def congruent_rectangle_width (y x : ℝ) : ℝ := (2 * y - x) / 2
noncomputable def congruent_rectangle_height (x : ℝ) : ℝ := x

-- Theorem statement
theorem congruent_rectangle_perimeter 
  (x y : ℝ) 
  (h : x ≤ y) : 
  2 * (congruent_rectangle_width y x + congruent_rectangle_height x) = 2 * y + x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_rectangle_perimeter_l1249_124908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_bounded_l1249_124948

/-- Product of digits in the decimal representation of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- The sequence n_k defined recursively -/
def seq (k : ℕ) : ℕ :=
  match k with
  | 0 => 1  -- n_1 is some arbitrary natural number, we choose 1 for simplicity
  | k + 1 => seq k + digit_product (seq k)

/-- The theorem stating that the sequence is bounded -/
theorem sequence_is_bounded : ∃ (M : ℕ), ∀ (k : ℕ), seq k ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_bounded_l1249_124948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storms_eye_area_l1249_124925

/- Define the storm's eye region -/
def storms_eye (r₁ r₂ : ℝ) (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^2 + p.2^2 ≤ r₁^2 ∧ 
               (p.1 - center.1)^2 + (p.2 - center.2)^2 ≥ r₂^2}

/- Define the area of a quarter circle -/
noncomputable def quarter_circle_area (r : ℝ) : ℝ := (Real.pi * r^2) / 4

/- Define the area of a full circle -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/- Theorem: The area of the storm's eye is 9π/4 -/
theorem storms_eye_area : 
  quarter_circle_area 5 - circle_area 2 = (9 * Real.pi) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_storms_eye_area_l1249_124925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_area_ratio_l1249_124941

theorem concentric_circles_area_ratio (r R : ℝ) 
  (hr : r > 0) (hR : R > 0) (hrR : r < R)
  (h : ∃ (θ : ℝ), 0 < θ ∧ θ < π ∧ θ / (2 * π) = 1 / 6 ∧ r / R = Real.cos (θ / 2)) :
  (π * r^2) / (π * R^2) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_area_ratio_l1249_124941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_8_l1249_124989

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2 * frac x - 3|

-- Define the property that f oscillates between 0 and 3 over any interval of length 1
axiom f_oscillates (x : ℝ) : ∃ y ∈ Set.Icc x (x + 1), f y = 0 ∧ ∃ z ∈ Set.Icc x (x + 1), f z = 3

-- Define the property that f transforms into three main intervals within each unit interval
axiom f_intervals (x : ℝ) : 
  (∃ y ∈ Set.Icc x (x + 1), f y ∈ Set.Icc 0 1.5) ∧
  (∃ y ∈ Set.Icc x (x + 1), f y ∈ Set.Icc 1.5 2) ∧
  (∃ y ∈ Set.Icc x (x + 1), f y ∈ Set.Icc 2 3)

-- Define the equation
def has_solution (m : ℕ) (x : ℝ) : Prop := m * f (x * f x) = x

-- Define the property of having at least 100 real solutions
def has_100_solutions (m : ℕ) : Prop := ∃ s : Set ℝ, s.Finite ∧ Nat.card s ≥ 100 ∧ ∀ x ∈ s, has_solution m x

-- The theorem to prove
theorem smallest_m_is_8 : 
  (∀ m < 8, ¬ has_100_solutions m) ∧ has_100_solutions 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_8_l1249_124989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l1249_124969

theorem sum_remainder_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l1249_124969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_zero_determinant_binary_matrices_l1249_124976

/-- A 2x2 matrix with elements in {0,1} -/
def BinaryMatrix := Matrix (Fin 2) (Fin 2) Bool

/-- The determinant of a 2x2 binary matrix -/
def det (m : BinaryMatrix) : Bool :=
  (m 0 0 && m 1 1) ≠ (m 0 1 && m 1 0)

/-- The set of all 2x2 binary matrices with determinant 0 -/
def ZeroDeterminantBinaryMatrices : Set BinaryMatrix :=
  {m : BinaryMatrix | det m = false}

/-- Proof that ZeroDeterminantBinaryMatrices is finite -/
instance : Fintype ZeroDeterminantBinaryMatrices := by
  sorry

theorem count_zero_determinant_binary_matrices :
  Fintype.card ZeroDeterminantBinaryMatrices = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_zero_determinant_binary_matrices_l1249_124976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l1249_124902

-- Define the custom operation
def otimes (a b : ℚ) : ℚ := a * b + |a| - b

-- Theorem for part (1)
theorem part_one : otimes (-5) 4 = -19 := by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify
  simp [abs_of_neg]
  -- Evaluate
  norm_num

-- Theorem for part (2)
theorem part_two : otimes (otimes 2 (-3)) 4 = -7 := by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify
  simp [abs_of_pos, abs_of_neg]
  -- Evaluate
  norm_num

-- Theorem for part (3)
theorem part_three : otimes 3 (-2) > otimes (-2) 3 := by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify
  simp [abs_of_pos, abs_of_neg]
  -- Evaluate
  norm_num

#check part_one
#check part_two
#check part_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l1249_124902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_theorem_l1249_124964

/-- Regular tetrahedron with base edge length 4 and side edge length 2√3 -/
structure RegularTetrahedron where
  base_edge : ℝ
  side_edge : ℝ
  base_edge_eq : base_edge = 4
  side_edge_eq : side_edge = 2 * Real.sqrt 3

/-- Sphere with center at a vertex of the tetrahedron and radius 2 -/
structure IntersectingSphere where
  radius : ℝ
  radius_eq : radius = 2

/-- The volume of intersection between the sphere and tetrahedron -/
noncomputable def intersection_volume (t : RegularTetrahedron) (s : IntersectingSphere) : ℝ := 
  16 * Real.pi / 9

/-- Theorem stating the volume of intersection -/
theorem intersection_volume_theorem (t : RegularTetrahedron) (s : IntersectingSphere) :
  intersection_volume t s = 16 * Real.pi / 9 := by
  -- Unfold the definition of intersection_volume
  unfold intersection_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_theorem_l1249_124964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_more_likely_from_second_machine_l1249_124923

/-- Represents a manufacturing machine --/
inductive Machine
| first
| second

/-- The probability that a part comes from a given machine --/
noncomputable def machine_probability (m : Machine) : ℝ :=
  match m with
  | Machine.first => 0.8
  | Machine.second => 0.2

/-- The probability that a part from a given machine is defective --/
noncomputable def defect_probability (m : Machine) : ℝ :=
  match m with
  | Machine.first => 0.01
  | Machine.second => 0.05

/-- The probability that a part is defective --/
noncomputable def total_defect_probability : ℝ :=
  (machine_probability Machine.first * defect_probability Machine.first) +
  (machine_probability Machine.second * defect_probability Machine.second)

/-- The probability that a defective part came from a given machine --/
noncomputable def defective_from_machine (m : Machine) : ℝ :=
  (machine_probability m * defect_probability m) / total_defect_probability

theorem defective_more_likely_from_second_machine :
  defective_from_machine Machine.second > defective_from_machine Machine.first := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_more_likely_from_second_machine_l1249_124923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_medians_hypotenuse_l1249_124916

/-- A right triangle with medians 5 and √40 from acute angles has hypotenuse 2√13 -/
theorem right_triangle_medians_hypotenuse : ∀ (a b c : ℝ),
  -- Right triangle condition
  a^2 + b^2 = c^2 →
  -- Median conditions (using the formula for medians in a right triangle)
  Real.sqrt (b^2 + (a/2)^2) = Real.sqrt 40 →
  Real.sqrt (a^2 + (b/2)^2) = 5 →
  -- Conclusion: hypotenuse is 2√13
  c = 2 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_medians_hypotenuse_l1249_124916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1249_124999

theorem problem_statement (x : ℝ) (m : ℝ) (a b c d : ℤ) : x ≠ 0 → m = 1 / x → m ≠ 1 →
  a > b ∧ b > c ∧ c > d →
  a * b * c * d = 1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7 →
  (((1 / x - 1) * (1 / x^7 + 1 / x^6 + 1 / x^5 + 1 / x^4 + 1 / x^3 + 1 / x^2 + 1 / x + 1) = 1 / x^8 - 1) ∧
  (m^7 + m^6 + m^5 + m^4 + m^3 + m^2 + m + 1 = (m + 1) * (m^2 + 1) * (m^4 + 1)) ∧
  ((-b / (c * d : ℝ))^2 / (-5 * b / (17 * c : ℝ)) * (6 * d / a : ℝ) = -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1249_124999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_property_implies_logarithmic_function_l1249_124936

/-- A function satisfying the logarithmic property --/
def LogarithmicProperty (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y

/-- Theorem: A function satisfying the logarithmic property is a logarithmic function --/
theorem logarithmic_property_implies_logarithmic_function (f : ℝ → ℝ) 
  (h : LogarithmicProperty f) : 
  ∃ (b : ℝ), b > 0 ∧ b ≠ 1 ∧ (∀ x, x > 0 → f x = Real.log x / Real.log b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_property_implies_logarithmic_function_l1249_124936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_even_l1249_124978

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi/4 + x) * Real.sin (Real.pi/4 - x)

theorem f_periodic_even : 
  (∀ x, f (x + Real.pi) = f x) ∧ 
  (∀ x, f (-x) = f x) := by
  sorry

#check f_periodic_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_even_l1249_124978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_or_q_l1249_124950

theorem proposition_p_or_q (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) (-1), x^2 + a*x - 2 > 0) ∨ 
  (∃! x, x < 0 ∧ a*x^2 + 2*x + 1 = 0) ↔ 
  a ≤ 0 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_or_q_l1249_124950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l1249_124987

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define the line l
def l (x y m : ℝ) : Prop := (m + 1) * x + (m - 1) * y - 2 = 0

-- Define the chord length function
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem min_chord_length :
  ∃ (m : ℝ), ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ m ∧ l x₂ y₂ m →
    chord_length x₁ y₁ x₂ y₂ ≥ 4 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l1249_124987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monochromatic_rectangles_l1249_124992

/-- A color is represented as a natural number -/
def Color := Fin 256  -- Using Fin for a finite set of colors

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- A rectangle in the plane -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorVertices (c : Coloring) (r : Rectangle) : Prop :=
  c r.p1 = c r.p2 ∧ c r.p2 = c r.p3 ∧ c r.p3 = c r.p4

/-- The main theorem -/
theorem infinite_monochromatic_rectangles 
  (n : ℕ) 
  (h : n ≥ 2) 
  (c : Coloring) 
  (hc : ∀ p, (c p).val < n) : 
  ∃ (S : Set Rectangle), Set.Infinite S ∧ ∀ r ∈ S, SameColorVertices c r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monochromatic_rectangles_l1249_124992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_of_1990_eq_11_l1249_124949

/-- Sum of digits of a natural number in decimal representation -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Function f as defined in the problem -/
def f (n : Nat) : Nat := sumOfDigits (n^2 + 1)

/-- Recursive definition of f_k -/
def f_k : Nat → Nat → Nat
  | 0, n => n
  | 1, n => f n
  | (k+1), n => f (f_k k n)

/-- The main theorem to prove -/
theorem f_100_of_1990_eq_11 : f_k 100 1990 = 11 := by
  sorry

#eval f_k 100 1990  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_of_1990_eq_11_l1249_124949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l1249_124927

-- Define the slope of line l
noncomputable def slope_l : ℝ := Real.tan (45 * Real.pi / 180)

-- Define the coordinates of points A and B
def point_A : ℝ × ℝ := (3, 2)
def point_B (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define the slope of line l₁
noncomputable def slope_l1 (a : ℝ) : ℝ := (point_A.2 - (point_B a).2) / (point_A.1 - (point_B a).1)

-- Define the condition that l₁ is perpendicular to l
def perpendicular_condition (a : ℝ) : Prop := slope_l * slope_l1 a = -1

-- Define the slope of line l₂
noncomputable def slope_l2 (b : ℝ) : ℝ := -2 / b

-- Define the condition that l₁ is parallel to l₂
def parallel_condition (a b : ℝ) : Prop := slope_l1 a = slope_l2 b

-- The theorem to prove
theorem sum_of_a_and_b (a b : ℝ) : 
  perpendicular_condition a → parallel_condition a b → a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l1249_124927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l1249_124967

/-- The profit function for a manufacturing company -/
noncomputable def profit (x : ℝ) : ℝ := -1/3 * x^3 + 81 * x - 234

/-- The derivative of the profit function -/
noncomputable def profit_derivative (x : ℝ) : ℝ := -x^2 + 81

/-- Theorem stating that the annual output of 9 (ten thousand units) maximizes the company's annual profit -/
theorem max_profit_at_nine :
  ∃ (x_max : ℝ), x_max = 9 ∧
  ∀ (x : ℝ), x > 0 → profit x ≤ profit x_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l1249_124967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solutions_l1249_124938

theorem unique_integer_solutions : 
  {(n, m) : ℕ × ℕ | (2 : ℕ)^((3 : ℕ)^n) = (3 : ℕ)^((2 : ℕ)^m) - 1} = {(0, 0), (1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solutions_l1249_124938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1249_124935

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define a point on the right branch of the hyperbola
variable (P : ℝ × ℝ)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem hyperbola_triangle_area :
  hyperbola P.1 P.2 →  -- P is on the hyperbola
  P.1 > 0 →  -- P is on the right branch
  distance P F₂ = 8/15 * distance F₁ F₂ →  -- Given condition
  1/2 * distance P F₁ * distance F₁ F₂ * Real.sqrt (1 - ((distance P F₁)^2 + (distance F₁ F₂)^2 - (distance P F₂)^2)^2 / (4 * (distance P F₁)^2 * (distance F₁ F₂)^2)) = 80/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1249_124935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_after_purchase_l1249_124960

/-- Calculates the change after a purchase with sales tax -/
theorem change_after_purchase
  (initial_amount : ℚ)
  (item_cost : ℚ)
  (tax_rate : ℚ)
  (h1 : initial_amount = 5)
  (h2 : item_cost = 428 / 100)
  (h3 : tax_rate = 7 / 100) :
  initial_amount - (item_cost + (item_cost * tax_rate).ceil / 100) = 42 / 100 := by
  sorry

#check change_after_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_after_purchase_l1249_124960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sibling_product_in_specific_family_l1249_124907

/-- Represents a family with siblings -/
structure Family where
  girls : ℕ
  boys : ℕ

/-- Calculates the number of sisters and brothers for a sibling -/
def sibling_count (f : Family) (is_girl : Bool) : ℕ × ℕ :=
  if is_girl then
    (f.girls - 1, f.boys)
  else
    (f.girls, f.boys - 1)

theorem sibling_product_in_specific_family : 
  ∀ (f : Family), 
  f.girls = 5 → f.boys = 7 → 
  ∃ (is_girl : Bool), 
  let (sisters, brothers) := sibling_count f is_girl
  sisters * brothers = 24 := by
  sorry

#check sibling_product_in_specific_family

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sibling_product_in_specific_family_l1249_124907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l1249_124906

/-- Calculates the profit percent for a car sale -/
noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the profit percent for the given car sale is approximately 54.05% -/
theorem car_sale_profit_percent : 
  let purchase_price : ℝ := 36400
  let repair_cost : ℝ := 8000
  let selling_price : ℝ := 68400
  abs (profit_percent purchase_price repair_cost selling_price - 54.05) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l1249_124906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1249_124937

/-- Calculates the time for two trains to cross each other under specific conditions -/
theorem train_crossing_time (length1 length2 : ℝ) (initial_speed1 initial_speed2 : ℝ) 
  (acceleration1 deceleration2 : ℝ) (acceleration_time1 : ℝ) (final_speed2 : ℝ) 
  (wind_resistance track_reduction : ℝ) : ℝ :=
  by
  -- Assumptions
  have h1 : length1 = 210 := by sorry
  have h2 : length2 = 260 := by sorry
  have h3 : initial_speed1 = 60 := by sorry
  have h4 : initial_speed2 = 40 := by sorry
  have h5 : acceleration1 = 3 := by sorry
  have h6 : deceleration2 = 2 := by sorry
  have h7 : acceleration_time1 = 20 := by sorry
  have h8 : final_speed2 = 20 := by sorry
  have h9 : wind_resistance = 0.05 := by sorry
  have h10 : track_reduction = 0.03 := by sorry

  -- Placeholder for the actual calculation
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1249_124937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crushing_load_calculation_l1249_124971

/-- The crushing load formula for square pillars -/
noncomputable def L (T H C : ℝ) : ℝ := (15 * T^3) / (H^2 + C)

/-- Proof that the crushing load L is equal to 1875/103 for given values -/
theorem crushing_load_calculation :
  L 5 10 3 = 1875 / 103 := by
  -- Unfold the definition of L
  unfold L
  -- Simplify the numerical expressions
  simp [pow_two, pow_three]
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crushing_load_calculation_l1249_124971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acai_berry_juice_cost_is_3104_35_l1249_124966

/-- The cost per litre of açaí berry juice given the following conditions:
  - The superfruit juice cocktail costs $1399.45 per litre to make.
  - The mixed fruit juice costs $262.85 per litre.
  - 33 litres of mixed fruit juice are used.
  - 22 litres of açaí berry juice are added.
-/
noncomputable def acai_berry_juice_cost : ℝ :=
  let superfruit_cost_per_litre : ℝ := 1399.45
  let mixed_fruit_cost_per_litre : ℝ := 262.85
  let mixed_fruit_volume : ℝ := 33
  let acai_berry_volume : ℝ := 22
  let total_volume : ℝ := mixed_fruit_volume + acai_berry_volume
  let total_cost : ℝ := total_volume * superfruit_cost_per_litre
  let mixed_fruit_cost : ℝ := mixed_fruit_volume * mixed_fruit_cost_per_litre
  let acai_berry_total_cost : ℝ := total_cost - mixed_fruit_cost
  acai_berry_total_cost / acai_berry_volume

theorem acai_berry_juice_cost_is_3104_35 : 
  acai_berry_juice_cost = 3104.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acai_berry_juice_cost_is_3104_35_l1249_124966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l1249_124933

-- Define a quadratic equation
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the equations from the problem
noncomputable def eq_A (x : ℝ) : ℝ := x - 1 / (x - 1)
noncomputable def eq_B (x : ℝ) : ℝ := 7 * x^2 + 1 / x^2 - 1
def eq_C (x : ℝ) : ℝ := x^2
def eq_D (x : ℝ) : ℝ := (x + 1) * (x - 2) - x * (x + 1)

-- Theorem stating that eq_C is quadratic while others are not
theorem quadratic_equation_identification :
  is_quadratic_equation eq_C ∧
  ¬is_quadratic_equation eq_A ∧
  ¬is_quadratic_equation eq_B ∧
  ¬is_quadratic_equation eq_D :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l1249_124933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_rotation_proof_l1249_124959

noncomputable def rotate_vector (v : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  let (x, y) := v
  (x * Real.cos angle - y * Real.sin angle,
   x * Real.sin angle + y * Real.cos angle)

theorem vector_rotation_proof (a : ℝ × ℝ) (h : a = (Real.sqrt 3, 1)) :
  let neg_2a := (-2 * a.1, -2 * a.2)
  let b := rotate_vector neg_2a (2 * Real.pi / 3)
  b = (2 * Real.sqrt 3, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_rotation_proof_l1249_124959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l1249_124979

/-- Represents Jason's work schedule and earnings --/
structure WorkSchedule where
  after_school_rate : ℚ
  saturday_rate : ℚ
  total_hours : ℚ
  total_earnings : ℚ

/-- Calculates the number of hours worked on Saturday given a work schedule --/
noncomputable def saturday_hours (schedule : WorkSchedule) : ℚ :=
  (schedule.total_earnings - schedule.after_school_rate * schedule.total_hours) /
  (schedule.saturday_rate - schedule.after_school_rate)

/-- Theorem stating that Jason worked 8 hours on Saturday --/
theorem jason_saturday_hours :
  let schedule : WorkSchedule := {
    after_school_rate := 4,
    saturday_rate := 6,
    total_hours := 18,
    total_earnings := 88
  }
  saturday_hours schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l1249_124979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1249_124954

theorem sin_double_angle_special_case (α : ℝ) 
  (h1 : Real.sin (π/2 - α) = -3/5)
  (h2 : 0 < α)
  (h3 : α < π) :
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1249_124954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rectangle_division_l1249_124975

theorem no_rectangle_division : ¬ ∃ (a b : ℝ) (n : ℕ),
  (a > 0 ∧ b > 0) ∧
  (∃ (m : ℕ), a * b = (3 * Real.sqrt 3 / 2) + m * (Real.sqrt 3 / 2)) ∧
  (∃ (k l : ℕ), a = k + l * Real.sqrt 3 ∧ b = (k + l * Real.sqrt 3 : ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rectangle_division_l1249_124975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_bound_l1249_124942

theorem cubic_root_bound (a b c : ℝ) (z : ℂ) 
  (h1 : 1 ≥ a) (h2 : a ≥ b) (h3 : b ≥ c) (h4 : c ≥ 0)
  (h5 : z^3 + a*z^2 + b*z + c = 0) : 
  Complex.abs z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_bound_l1249_124942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_variance_calculation_l1249_124910

-- Define the binomial distribution as a predicate
def binomial_distribution (n : ℕ) (p : ℝ) (X : ℝ → ℝ) : Prop := sorry

-- Define expected value for binomial distribution
def expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

-- Define variance for binomial distribution
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_calculation :
  ∀ (X : ℝ → ℝ) (n : ℕ) (p : ℝ),
    n = 8 →
    binomial_distribution n p X →
    expected_value n p = 1.6 →
    variance n p = 1.28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_variance_calculation_l1249_124910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1249_124919

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ)

noncomputable def f_derivative (x θ : ℝ) : ℝ := 2 * Real.cos (2 * x + θ)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem tan_theta_value (θ : ℝ) :
  is_odd (λ x => f x θ + f_derivative x θ) →
  Real.tan θ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1249_124919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_friday_hours_l1249_124917

/-- Sandy's work scenario --/
structure WorkScenario where
  hourly_rate : ℚ
  saturday_hours : ℚ
  sunday_hours : ℚ
  total_earnings : ℚ

/-- Calculate Friday hours based on work scenario --/
def friday_hours (scenario : WorkScenario) : ℚ :=
  (scenario.total_earnings - scenario.hourly_rate * (scenario.saturday_hours + scenario.sunday_hours)) / scenario.hourly_rate

/-- Theorem: Sandy worked 10 hours on Friday --/
theorem sandy_friday_hours :
  let scenario : WorkScenario := {
    hourly_rate := 15,
    saturday_hours := 6,
    sunday_hours := 14,
    total_earnings := 450
  }
  friday_hours scenario = 10 := by
  -- The proof goes here
  sorry

#eval friday_hours {
  hourly_rate := 15,
  saturday_hours := 6,
  sunday_hours := 14,
  total_earnings := 450
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_friday_hours_l1249_124917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1249_124973

/-- The parabola C with focus F and directrix d -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  directrix : ℝ
  eq : (y : ℝ) → (x : ℝ) → Prop

/-- A point on the x-axis -/
def XAxisPoint (x : ℝ) : ℝ × ℝ := (x, 0)

/-- The origin -/
def Origin : ℝ × ℝ := (0, 0)

theorem parabola_directrix 
  (C : Parabola)
  (h_p_pos : C.p > 0)
  (h_focus : C.focus = (C.p / 2, 0))
  (P : ℝ × ℝ)
  (h_P_on_C : C.eq P.2 P.1)
  (h_PF_perp_x : P.2 = C.p)
  (Q : ℝ)
  (h_Q_on_x : (Q, 0) = XAxisPoint Q)
  (h_PQ_perp_OP : (P.2 - 0) * (P.1 - Q) = -(P.1 - 0) * (P.2 - 0))
  (h_FQ_dist : |Q - C.p / 2| = 6)
  : C.directrix = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1249_124973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1249_124945

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-8) 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x + 1) / (x + 2)

-- Define the domain of g
def dom_g : Set ℝ := Set.Ioo (-9/2) (-2) ∪ Set.Ico (-2) 0

theorem domain_of_g : 
  ∀ x : ℝ, x ∈ dom_g ↔ (2 * x + 1 ∈ dom_f ∧ x + 2 ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1249_124945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_group_l1249_124909

-- Define the group
structure MyGroup where
  women : ℕ
  men : ℕ
  womenAgeSum : ℕ
  menAgeSum : ℕ

-- Define the properties of the group
def validGroup (g : MyGroup) : Prop :=
  g.women * 8 = g.men * 9 ∧
  g.womenAgeSum = g.women * 30 ∧
  g.menAgeSum = g.men * 36

-- Theorem statement
theorem average_age_of_group (g : MyGroup) (h : validGroup g) :
  (g.womenAgeSum + g.menAgeSum : ℚ) / (g.women + g.men : ℚ) = 32 + 14/17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_group_l1249_124909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_10cm_l1249_124977

/-- The amount of metal wasted when cutting shapes from a square -/
noncomputable def metal_waste (s : ℝ) : ℝ :=
  let original_square_area := s^2
  let circle_radius := s / 2
  let circle_area := Real.pi * circle_radius^2
  let inner_square_side := s / Real.sqrt 2
  let inner_square_area := inner_square_side^2
  let triangle_side := inner_square_side
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  original_square_area - circle_area + (circle_area - inner_square_area) + (inner_square_area - triangle_area)

/-- Theorem stating the amount of metal wasted for a 10 cm square -/
theorem metal_waste_10cm :
  metal_waste 10 = 100 - 25 * Real.pi + 50 - (25 * Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_10cm_l1249_124977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_journey_time_l1249_124957

/-- Calculates the time taken for a boat to travel a given distance downstream -/
noncomputable def time_downstream (boat_speed stream_speed distance : ℝ) : ℝ :=
  distance / (boat_speed + stream_speed)

/-- Proves that the total time taken by the boat to cover all three segments is approximately 7.0436 hours -/
theorem boat_journey_time (boat_speed : ℝ) (total_distance : ℝ) 
  (segment1_distance segment1_stream_speed : ℝ)
  (segment2_distance segment2_stream_speed : ℝ)
  (segment3_distance segment3_stream_speed : ℝ)
  (h1 : boat_speed = 16)
  (h2 : total_distance = 147)
  (h3 : segment1_distance = 47)
  (h4 : segment1_stream_speed = 5)
  (h5 : segment2_distance = 50)
  (h6 : segment2_stream_speed = 7)
  (h7 : segment3_distance = 50)
  (h8 : segment3_stream_speed = 3)
  (h9 : total_distance = segment1_distance + segment2_distance + segment3_distance) :
  ∃ (ε : ℝ), ε > 0 ∧ 
  |time_downstream boat_speed segment1_stream_speed segment1_distance +
   time_downstream boat_speed segment2_stream_speed segment2_distance +
   time_downstream boat_speed segment3_stream_speed segment3_distance - 7.0436| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_journey_time_l1249_124957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_51_good_not_very_good_good_2010_implies_very_good_l1249_124990

-- Define the polynomial P(x) = ax^3 + bx
def P (a b x : ℤ) : ℤ := a * x^3 + b * x

-- Define n-good
def is_n_good (a b n : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ (P a b m - P a b k) → n ∣ (m - k)

-- Define very good
def is_very_good (a b : ℤ) : Prop :=
  ∃ f : ℕ → ℕ, Monotone f ∧ StrictMono f ∧ ∀ i : ℕ, is_n_good a b (f i)

-- Theorem 1
theorem pair_51_good_not_very_good :
  is_n_good 1 (-51^2) 51 ∧ ¬ is_very_good 1 (-51^2) := by sorry

-- Theorem 2
theorem good_2010_implies_very_good :
  ∀ a b : ℤ, is_n_good a b 2010 → is_very_good a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_51_good_not_very_good_good_2010_implies_very_good_l1249_124990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_real_roots_l1249_124962

/-- The function f(x) = 2^x - x^2 - 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - x^2 - 1

/-- The equation 2^x = 1 + x^2 has exactly three real roots -/
theorem three_real_roots : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_real_roots_l1249_124962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_lateral_surface_area_l1249_124997

/-- The lateral surface area of a frustum of a right circular cone. -/
noncomputable def frustumLateralSurfaceArea (R r h : ℝ) : ℝ :=
  let l := Real.sqrt ((R - r)^2 + h^2)
  Real.pi * (R + r) * l

/-- Theorem: The lateral surface area of a specific frustum. -/
theorem specific_frustum_lateral_surface_area :
  frustumLateralSurfaceArea 10 4 9 = 14 * Real.pi * Real.sqrt 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_lateral_surface_area_l1249_124997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l1249_124974

/-- Given a square, circle, and rectangle with specific properties, prove the breadth of the rectangle --/
theorem rectangle_breadth (square_area : ℝ) (circle_radius : ℝ) (rect_length : ℝ) (rect_area : ℝ) 
    (h1 : square_area = 1600)
    (h2 : circle_radius = Real.sqrt square_area)
    (h3 : rect_length = (2/5) * circle_radius)
    (h4 : rect_area = 160)
    (h5 : rect_area = rect_length * (rect_area / rect_length)) :
  rect_area / rect_length = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l1249_124974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_recycled_pounds_l1249_124995

/-- The number of pounds of paper needed to earn one point -/
def pounds_per_point : ℕ := 6

/-- The number of pounds Chloe recycled -/
def chloe_pounds : ℕ := 28

/-- The total number of points earned by Chloe and her friends -/
def total_points : ℕ := 5

/-- The number of pounds Chloe's friends recycled -/
def friends_pounds : ℕ := 6

theorem friends_recycled_pounds :
  (chloe_pounds / pounds_per_point) * pounds_per_point + friends_pounds = 
  total_points * pounds_per_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_recycled_pounds_l1249_124995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_712_factorial_plus_one_l1249_124947

theorem not_prime_712_factorial_plus_one (h : Nat.Prime 719) : 
  ∃ k : ℕ, (Nat.factorial 712 + 1) = 719 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_712_factorial_plus_one_l1249_124947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_power_divisors_l1249_124970

theorem fifth_power_divisors (x d : ℕ) (n : ℕ+) : 
  x = n ^ 5 → d = (Finset.filter (λ i => i ∣ x) (Finset.range (x + 1))).card → d % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_power_divisors_l1249_124970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_sum_sine_l1249_124918

-- Define the line equation
def line (x : ℝ) : ℝ := 3 * x

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ y = line x ∧ circle_eq x y}

-- Define the angle function
noncomputable def angle (p : ℝ × ℝ) : ℝ := Real.arctan (p.2 / p.1)

-- Theorem statement
theorem intersection_angles_sum_sine :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    A ≠ B ∧ Real.sin (angle A + angle B) = -3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_sum_sine_l1249_124918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radii_ratio_value_l1249_124901

/-- A right-angled sector with two tangent circles -/
structure TangentCirclesSector where
  /-- The radius of the sector -/
  R : ℝ
  /-- The radius of circle ω -/
  r_omega : ℝ
  /-- The radius of circle ω' -/
  r_omega_prime : ℝ
  /-- Circle ω is tangent to arc AB, arc OC, and line OA -/
  h_omega_tangent : True
  /-- Circle ω' is tangent to arc OC, line OA, and circle ω -/
  h_omega_prime_tangent : True
  /-- The sector is right-angled -/
  h_right_angled : True
  /-- R is equal to 6 -/
  h_R_eq_6 : R = 6

/-- The ratio of the radii of circles ω and ω' in a TangentCirclesSector -/
noncomputable def radii_ratio (s : TangentCirclesSector) : ℝ := s.r_omega / s.r_omega_prime

/-- The theorem stating the ratio of the radii of circles ω and ω' -/
theorem radii_ratio_value (s : TangentCirclesSector) :
  radii_ratio s = (7 + 2 * Real.sqrt 6) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radii_ratio_value_l1249_124901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l1249_124993

theorem trigonometric_simplification : 
  ∃ (a b : ℕ), 
    (0 < b) ∧ 
    (b < 90) ∧ 
    (1000 * Real.sin (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180) = 
     (a : ℝ) * Real.sin (b * π / 180)) ∧
    (100 * a + b = 12560) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l1249_124993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1249_124904

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.sin t.A = t.a * Real.sin (2 * t.B) ∧
  t.b = Real.sqrt 10 ∧
  t.a + t.c = t.a * t.c

-- Helper function to calculate triangle area
noncomputable def triangle_area (t : Triangle) : Real :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 3 ∧ triangle_area t = 5 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1249_124904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unique_coefficients_9th_degree_polynomial_l1249_124922

/-- Represents a polynomial of degree n with non-zero coefficients -/
structure MyPolynomial (n : ℕ) where
  coeffs : Fin (n + 1) → ℝ
  nonzero : ∀ i, coeffs i ≠ 0

/-- Calculates the derivative of a polynomial -/
noncomputable def derivative (p : MyPolynomial n) : MyPolynomial (n - 1) := sorry

/-- Counts the number of unique coefficients in a polynomial and its derivatives -/
noncomputable def countUniqueCoefficients (p : MyPolynomial n) : ℕ := sorry

/-- Theorem: The minimum number of unique coefficients for a 9th degree polynomial
    and its derivatives down to a constant is 9 -/
theorem min_unique_coefficients_9th_degree_polynomial :
  ∀ p : MyPolynomial 9, countUniqueCoefficients p ≥ 9 ∧
  ∃ p : MyPolynomial 9, countUniqueCoefficients p = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unique_coefficients_9th_degree_polynomial_l1249_124922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_duration_l1249_124914

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
noncomputable def bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : ℝ :=
  let distance_difference := speed_without_stops - speed_with_stops
  distance_difference / speed_without_stops * 60

/-- Theorem stating that a bus with given speeds stops for approximately 15.7 minutes per hour -/
theorem bus_stop_duration (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 65)
  (h2 : speed_with_stops = 48) :
  ∃ ε > 0, |bus_stop_time speed_without_stops speed_with_stops - 15.7| < ε := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_duration_l1249_124914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_short_connected_network_l1249_124940

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Represents a road network connecting four villages -/
structure RoadNetwork where
  a : Point
  b : Point
  c : Point
  d : Point
  roads : List (Point × Point)

/-- Checks if the road network allows travel between any two villages -/
def isConnected (network : RoadNetwork) : Prop :=
  ∀ p q : Point, p ∈ [network.a, network.b, network.c, network.d] →
                 q ∈ [network.a, network.b, network.c, network.d] →
                 ∃ path : List Point, path.head? = some p ∧ path.getLast? = some q ∧
                 ∀ i : Fin (path.length - 1), (path[i.val], path[i.val+1]) ∈ network.roads ∨
                                              (path[i.val+1], path[i.val]) ∈ network.roads

/-- Calculates the total length of roads in the network -/
noncomputable def totalLength (network : RoadNetwork) : ℝ :=
  network.roads.foldl (λ acc road => acc + distance road.1 road.2) 0

/-- The main theorem to be proved -/
theorem exists_short_connected_network :
  ∃ (network : RoadNetwork),
    network.a = ⟨0, 0⟩ ∧
    network.b = ⟨2, 0⟩ ∧
    network.c = ⟨2, 2⟩ ∧
    network.d = ⟨0, 2⟩ ∧
    isConnected network ∧
    totalLength network < 5.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_short_connected_network_l1249_124940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_parallel_l1249_124983

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- The given vectors from option D -/
def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (6, -4)

/-- Theorem: Vectors a and b are parallel -/
theorem vectors_are_parallel : are_parallel a b := by
  use (-1/2)
  simp [a, b, are_parallel]
  constructor
  · norm_num
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_are_parallel_l1249_124983
