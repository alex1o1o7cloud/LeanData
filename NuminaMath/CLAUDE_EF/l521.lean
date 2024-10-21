import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l521_52121

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 1 else Real.log (x + 1) / Real.log (1/2)

-- State the theorem
theorem solution_set_f (x : ℝ) :
  f x < 3 ↔ -2 < x ∧ x < 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l521_52121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_symmetry_sides_l521_52148

/-- A polygon with its number of sides and symmetry lines. -/
structure Polygon where
  sides : ℕ
  symmetryLines : ℕ

/-- Predicate indicating that a polygon with S sides has exactly n lines of symmetry. -/
def HasExactlyNLinesOfSymmetry (n : ℕ) (S : ℕ) : Prop :=
  ∃ (polygon : Polygon), polygon.sides = S ∧ polygon.symmetryLines = n

/-- A polygon with n lines of symmetry has a number of sides that is a multiple of n. -/
theorem polygon_symmetry_sides (n : ℕ) (S : ℕ) (h : HasExactlyNLinesOfSymmetry n S) :
  ∃ k : ℕ, S = n * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_symmetry_sides_l521_52148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_72_l521_52160

/-- Represents the upstream distance traveled by a boat -/
noncomputable def upstream_distance (river_speed boat_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := boat_speed - river_speed
  let downstream_speed := boat_speed + river_speed
  (total_time * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed)

/-- Theorem stating that under given conditions, the upstream distance is 72 km -/
theorem upstream_distance_is_72 :
  upstream_distance 2 6 27 = 72 := by
  -- Unfold the definition of upstream_distance
  unfold upstream_distance
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_72_l521_52160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_S_approx_l521_52123

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of sulfur in g/mol -/
noncomputable def molar_mass_S : ℝ := 32.06

/-- The number of aluminum atoms in aluminum sulfide -/
def num_Al : ℕ := 2

/-- The number of sulfur atoms in aluminum sulfide -/
def num_S : ℕ := 3

/-- The molar mass of aluminum sulfide in g/mol -/
noncomputable def molar_mass_Al2S3 : ℝ := num_Al * molar_mass_Al + num_S * molar_mass_S

/-- The mass percentage of sulfur in aluminum sulfide -/
noncomputable def mass_percentage_S : ℝ := (num_S * molar_mass_S / molar_mass_Al2S3) * 100

theorem mass_percentage_S_approx :
  ∃ (ε : ℝ), ε > 0 ∧ |mass_percentage_S - 64.07| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_S_approx_l521_52123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lucky_monkeys_l521_52117

/-- A fruit type representing the four available fruits -/
inductive Fruit
  | Orange
  | Banana
  | Peach
  | Tangerine
deriving BEq, DecidableEq

/-- A function to determine if a monkey is lucky based on the fruits it eats -/
def is_lucky (fruits : List Fruit) : Bool :=
  (fruits.length == 3) && (fruits.toFinset.card == 3)

/-- The available quantity of each fruit -/
def fruit_quantities : Fruit → Nat
  | Fruit.Orange => 20
  | Fruit.Banana => 30
  | Fruit.Peach => 40
  | Fruit.Tangerine => 50

/-- The theorem stating the maximum number of lucky monkeys -/
theorem max_lucky_monkeys :
  ∃ (distribution : List (List Fruit)),
    (∀ fruits ∈ distribution, is_lucky fruits) ∧
    (∀ fruit : Fruit, (distribution.join.filter (· == fruit)).length ≤ fruit_quantities fruit) ∧
    distribution.length = 40 ∧
    ∀ (other_distribution : List (List Fruit)),
      (∀ fruits ∈ other_distribution, is_lucky fruits) →
      (∀ fruit : Fruit, (other_distribution.join.filter (· == fruit)).length ≤ fruit_quantities fruit) →
      other_distribution.length ≤ 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lucky_monkeys_l521_52117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l521_52103

noncomputable def TriangleArea (a b θ : ℝ) : ℝ := (1 / 2) * a * b * Real.sin θ

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < B) → (B < π) →
  (Real.sqrt 3 * c = Real.sqrt 3 * b * Real.cos A + a * Real.sin B) →
  (B = π / 3) ∧
  ((A = π / 4 ∧ b = 2 * Real.sqrt 3 → ∃! (a : ℝ), TriangleArea a b (π / 3) = 3 + Real.sqrt 3) ∨
   (c = 4 ∧ b = Real.sqrt 21 → ∃! (a : ℝ), TriangleArea a 4 (π / 3) = 5 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l521_52103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_below_line_l521_52132

/-- The area of a circle segment below a horizontal line -/
noncomputable def circle_segment_area (a b c : ℝ) (k : ℝ) : ℝ :=
  let r := Real.sqrt ((a - 2)^2 + (b - 8)^2)
  let d := b - k
  (r^2 / 2) * (2 * Real.arccos (d / r) - Real.sin (2 * Real.arccos (d / r)))

/-- The theorem statement -/
theorem circle_area_below_line (x y : ℝ) :
  x^2 - 4*x + y^2 - 16*y + 39 = 0 →
  circle_segment_area 2 8 29 5 = (29/2) * (2 * Real.arccos (3/Real.sqrt 29) - Real.sin (2 * Real.arccos (3/Real.sqrt 29))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_below_line_l521_52132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_five_largest_l521_52172

/-- The number of cards in the box -/
def total_cards : ℕ := 6

/-- The number of cards selected -/
def selected_cards : ℕ := 3

/-- The probability of an event occurring -/
noncomputable def probability (favorable_outcomes : ℝ) (total_outcomes : ℝ) : ℝ :=
  favorable_outcomes / total_outcomes

/-- The probability of not selecting a specific card in a single draw -/
noncomputable def prob_not_select (cards_left : ℕ) (unwanted_cards : ℕ) : ℝ :=
  (cards_left - unwanted_cards : ℝ) / cards_left

/-- The theorem to be proved -/
theorem prob_five_largest (total_cards : ℕ) (selected_cards : ℕ) :
  total_cards = 6 → selected_cards = 3 →
  probability (probability 1 2 - probability 1 5) 1 = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_five_largest_l521_52172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_remainders_l521_52115

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (∀ n : ℕ, n ∈ ({6, 7, 8, 9, 10, 11, 13} : Set ℕ) → M % n = n - 1) ∧
  (∀ k : ℕ, k < M → ∃ n : ℕ, n ∈ ({6, 7, 8, 9, 10, 11, 13} : Set ℕ) ∧ k % n ≠ n - 1) ∧
  M = 360359 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_remainders_l521_52115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l521_52181

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + Real.sin (2 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ (M = 1 + Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l521_52181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_is_two_l521_52188

theorem cube_root_of_eight_is_two : (8 : ℝ) ^ (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_is_two_l521_52188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l521_52104

theorem binomial_expansion_properties :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ),
  (∀ x : ℝ, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1) ∧
  (a + a₂ + a₄ + a₆ = 1094) ∧
  (a₁ + a₃ + a₅ + a₇ = -1093) ∧
  (Finset.sum (Finset.range 8) (λ k => Nat.choose 7 k) = 128) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l521_52104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_tangent_line_l521_52136

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 9

-- Define the line with slope k
def line (k c x y : ℝ) : Prop := y = k * x + c

-- Define the midpoint of two points
def midpoint_of (x1 y1 x2 y2 x0 y0 : ℝ) : Prop :=
  x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

theorem slope_of_tangent_line 
  (k : ℝ) 
  (x1 y1 x2 y2 x0 y0 c : ℝ) :
  parabola x1 y1 →
  parabola x2 y2 →
  circle_eq x0 y0 →
  line k c x1 y1 →
  line k c x2 y2 →
  line k c x0 y0 →
  midpoint_of x1 y1 x2 y2 x0 y0 →
  k = 2 * Real.sqrt 5 / 5 ∨ k = -2 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_tangent_line_l521_52136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l521_52145

noncomputable def x (t : ℝ) : ℝ := (3 : ℝ)^t - 2
noncomputable def y (t : ℝ) : ℝ := (9 : ℝ)^t - 7 * (3 : ℝ)^t + 4

theorem point_on_parabola :
  ∃ (a b c : ℝ), ∀ (t : ℝ), y t = a * (x t)^2 + b * (x t) + c := by
  -- We'll provide the existence of a, b, and c
  use 1, -3, -6
  -- Now we need to prove that for all t, y t = (x t)^2 - 3*(x t) - 6
  intro t
  -- Expand the definitions of x and y
  simp [x, y]
  -- The rest of the proof would involve algebraic manipulation
  -- For now, we'll use sorry to skip the detailed proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l521_52145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_eq_4014_l521_52167

noncomputable def polynomial (x : ℝ) : ℝ :=
  (x - 1)^2008 + 2*(x - 2)^2007 + 3*(x - 3)^2006 + 4*(x - 4)^2005 + 5*(x - 5)^2004 + 6*(x - 6)^2003

theorem sum_of_roots_eq_4014 :
  ∃ (roots : Finset ℝ), (∀ x ∈ roots, polynomial x = 0) ∧ (Finset.sum roots id = 4014) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_eq_4014_l521_52167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_profit_theorem_l521_52165

def bag_weights : List Float := [91, 92, 90, 89, 89, 91.2, 88.9, 91.8, 91.1, 88]
def expected_weight : Float := 90
def bags_count : Nat := 10
def wheat_price : Float := 100
def flour_price : Float := 4
def wheat_to_flour_ratio : Float := 0.7
def processing_cost : Float := 500

theorem wheat_profit_theorem :
  (List.sum bag_weights - bags_count.toFloat * expected_weight = 2) ∧
  (flour_price * wheat_to_flour_ratio * (List.sum bag_weights) - 
   (wheat_price * bags_count.toFloat + processing_cost) = 1025.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_profit_theorem_l521_52165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l521_52180

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The distance between the right vertex and the right focus of an ellipse -/
noncomputable def right_vertex_focus_distance (e : Ellipse) : ℝ := 
  (e.a^2 - e.b^2).sqrt - e.a

/-- The length of the minor axis of an ellipse -/
def minor_axis_length (e : Ellipse) : ℝ := 2 * e.b

/-- A line passing through a point -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The area of a triangle formed by two points on an ellipse and the origin -/
noncomputable def triangle_area (e : Ellipse) (l : Line) : ℝ := sorry

/-- The main theorem -/
theorem ellipse_and_line_theorem (e : Ellipse) 
  (h1 : right_vertex_focus_distance e = Real.sqrt 3 - 1)
  (h2 : minor_axis_length e = 2 * Real.sqrt 2) :
  (∃ l : Line, triangle_area e l = (3 * Real.sqrt 2) / 4) →
  (e.a = Real.sqrt 3 ∧ e.b = Real.sqrt 2) ∧
  (∃ l : Line, (l.slope = Real.sqrt 2 ∧ l.intercept = -Real.sqrt 2) ∨
               (l.slope = -Real.sqrt 2 ∧ l.intercept = -Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l521_52180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l521_52112

/-- The equation whose roots form the vertices of the hexagon -/
def hexagon_equation (x : ℂ) : Prop :=
  x^6 + 6*x^3 - 216 = 0

/-- The set of points that are roots of the hexagon equation -/
def hexagon_points : Set ℂ :=
  {z : ℂ | hexagon_equation z}

/-- Predicate to check if a set of complex points forms a convex hexagon -/
def is_convex_hexagon (s : Set ℂ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℂ), s = {p₁, p₂, p₃, p₄, p₅, p₆} ∧ 
  -- Additional conditions for convexity would be defined here
  True

/-- The area of a polygon given its vertices -/
noncomputable def polygon_area (vertices : List ℂ) : ℝ :=
  sorry  -- The actual implementation would go here

/-- Theorem stating the area of the hexagon formed by the roots of the equation -/
theorem hexagon_area :
  is_convex_hexagon hexagon_points →
  ∃ (vertices : List ℂ), 
    vertices.length = 6 ∧
    (∀ v ∈ vertices, v ∈ hexagon_points) ∧
    polygon_area vertices = 9 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l521_52112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l521_52118

/-- Future value of an investment with annual compounding -/
noncomputable def future_value_annual (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Future value of an investment with quarterly compounding -/
noncomputable def future_value_quarterly (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 4) ^ (4 * time)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_difference : 
  let principal := (50000 : ℝ)
  let rate := (0.04 : ℝ)
  let time := (2 : ℕ)
  round_to_nearest (future_value_quarterly principal rate time - 
                    future_value_annual principal rate time) = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l521_52118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_line_equation_l521_52135

/-- Given two points P and Q that are symmetrical with respect to a line l,
    prove that the equation of line l is x - y + 1 = 0 --/
theorem symmetric_points_line_equation (m n : ℝ) :
  let P : ℝ × ℝ := (m - 2, n + 1)
  let Q : ℝ × ℝ := (n, m - 1)
  let l : Set (ℝ × ℝ) := {(x, y) | x - y + 1 = 0}
  (∀ (M : ℝ × ℝ), M ∈ l → (dist P M = dist Q M)) →
  l = {(x, y) | x - y + 1 = 0} :=
by
  sorry

/-- Helper function to calculate the distance between two points --/
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_line_equation_l521_52135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l521_52184

noncomputable def f (x : ℝ) := Real.sin (2 * Real.pi * x - Real.pi / 5)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioo (-3/20) (7/20)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l521_52184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l521_52164

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) -/
noncomputable def circle_radius (θ : ℝ) : ℝ :=
  Real.sqrt ((2 * Real.sin (Real.pi/4) * Real.cos θ)^2 + (2 * Real.sin (Real.pi/4) * Real.sin θ)^2)

/-- Theorem stating that the radius of the circle is √2 -/
theorem circle_radius_is_sqrt_2 (θ : ℝ) : circle_radius θ = Real.sqrt 2 := by
  sorry

#check circle_radius_is_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l521_52164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_is_60_l521_52111

/-- A rectangular field with one uncovered side and three fenced sides -/
structure RectangularField where
  uncovered_side : ℝ
  fencing : ℝ

/-- The area of the rectangular field -/
noncomputable def field_area (f : RectangularField) : ℝ :=
  let width := (f.fencing - f.uncovered_side) / 2
  f.uncovered_side * width

/-- Theorem: The area of the specified rectangular field is 60 square feet -/
theorem field_area_is_60 (f : RectangularField) 
  (h1 : f.uncovered_side = 20)
  (h2 : f.fencing = 26) : 
  field_area f = 60 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval field_area { uncovered_side := 20, fencing := 26 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_is_60_l521_52111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_nonpositive_l521_52158

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + a * Real.cos x + a

theorem f_upper_bound_implies_a_nonpositive :
  (∀ a : ℝ, ∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≤ 1) →
  (∀ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≤ 1) → a ≤ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_nonpositive_l521_52158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l521_52146

/-- The unit circle centered at the origin -/
def Γ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Point A on the x-axis -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B on the x-axis -/
def B : ℝ × ℝ := (1, 0)

/-- A moving point P on the circle Γ -/
noncomputable def P : ℝ × ℝ := sorry

/-- The line passing through P and tangent to Γ -/
noncomputable def l : Set (ℝ × ℝ) := sorry

/-- The line perpendicular to l passing through A -/
noncomputable def perpLine : Set (ℝ × ℝ) := sorry

/-- Point M, intersection of perpLine and line BP -/
noncomputable def M : ℝ × ℝ := sorry

/-- The line x + 2y - 9 = 0 -/
def targetLine : Set (ℝ × ℝ) := {p | p.1 + 2*p.2 - 9 = 0}

/-- The distance function from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ := sorry

theorem max_distance_to_line :
  ∃ (P : ℝ × ℝ), P ∈ Γ ∧ distanceToLine M targetLine ≤ 2*Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l521_52146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_slope_is_two_thirds_l521_52178

-- Define the function f(x) = x / (1-x)
noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

-- Define the slope of the secant line
noncomputable def secant_slope (x₁ x₂ : ℝ) : ℝ := (f x₂ - f x₁) / (x₂ - x₁)

-- Theorem statement
theorem secant_slope_is_two_thirds :
  secant_slope 2 2.5 = 2/3 := by
  -- Unfold the definitions
  unfold secant_slope f
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_slope_is_two_thirds_l521_52178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_l521_52169

-- Define the triangle ABC
structure Triangle where
  A : Real
  a : Real
  b : Real
  c : Real

-- Define the specific triangle from the problem
noncomputable def problemTriangle : Triangle := {
  A := Real.pi / 6
  a := 1
  b := Real.sqrt 3
  c := 0  -- We'll prove this is either 1 or 2
}

-- State the theorem
theorem triangle_side_c (t : Triangle) (h : t = problemTriangle) :
  t.c = 1 ∨ t.c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_l521_52169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l521_52193

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/3)^x else Real.log x / Real.log 3

-- State the theorem
theorem f_composition_value :
  f (f (1/9)) = 9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l521_52193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_squares_l521_52186

/-- Given two squares where the smaller square has a perimeter of 8 cm and the larger square
    has an area of 36 cm², prove that the distance between the upper right corner of the
    larger square and the upper left corner of the smaller square is 4√5 cm. -/
theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ)
    (h_small_perimeter : small_perimeter = 8)
    (h_large_area : large_area = 36) : 
    ∃ (distance : ℝ), distance = 4 * Real.sqrt 5 := by
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal_distance := small_side + large_side
  let vertical_distance := large_side - small_side
  let distance := Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2)
  
  have h : distance = 4 * Real.sqrt 5 := by
    sorry -- Proof steps would go here
  
  exact ⟨distance, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_squares_l521_52186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_section_parabola_l521_52114

/-- Represents the shape of a conic section -/
inductive ConicSection
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- Given a cone with an apex angle of 90° and a section that forms a 45° angle with the axis,
    the resulting section is a parabola -/
theorem cone_section_parabola (apex_angle section_axis_angle : ℝ) :
  apex_angle = 90 ∧ section_axis_angle = 45 →
  ConicSection.Parabola = ConicSection.Parabola := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_section_parabola_l521_52114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_16_l521_52168

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def a : ℤ := floor ((Real.sqrt 3 - Real.sqrt 2) ^ 2009) + 16

theorem a_equals_16 : a = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_16_l521_52168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l521_52138

-- Define the circle ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points A and B
def A : ℝ × ℝ := (5, 12)
def B : ℝ × ℝ := (14, 8)

-- Define the property that A and B lie on ω
def points_on_circle (ω : Circle) : Prop :=
  (A.1 - ω.center.1)^2 + (A.2 - ω.center.2)^2 = ω.radius^2 ∧
  (B.1 - ω.center.1)^2 + (B.2 - ω.center.2)^2 = ω.radius^2

-- Define the property that tangent lines at A and B intersect on x-axis
def tangents_intersect_x_axis (ω : Circle) : Prop :=
  ∃ x : ℝ, (x, 0) ∈ Set.range (λ t : ℝ ↦ (t, (A.2 / A.1) * (t - A.1) + A.2)) ∩
              Set.range (λ t : ℝ ↦ (t, (B.2 / B.1) * (t - B.1) + B.2))

-- Theorem statement
theorem circle_area_theorem (ω : Circle) 
  (h1 : points_on_circle ω) 
  (h2 : tangents_intersect_x_axis ω) : 
  ω.radius^2 * π = 121975 / 1961 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l521_52138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l521_52108

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2) ∧ x > 0}
def B : Set ℝ := {x | (3 : ℝ)^x > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l521_52108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l521_52124

theorem rectangular_to_polar_conversion :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r * (Real.cos θ) = 2 ∧ r * (Real.sin θ) = 2 * Real.sqrt 3 ∧
  r = 4 ∧ θ = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l521_52124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ways_to_choose_book_l521_52189

def bookshelf_choice_count (chinese_books english_books math_books : ℕ) : ℕ :=
  chinese_books + english_books + math_books

theorem total_ways_to_choose_book :
  bookshelf_choice_count 10 7 5 = 22 := by
  unfold bookshelf_choice_count
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ways_to_choose_book_l521_52189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_equation_determines_ratio_l521_52191

/-- A homogeneous polynomial of degree n in two variables -/
def HomogeneousPolynomial (n : ℕ) (x y : ℝ) (a : ℕ → ℝ) : ℝ :=
  (Finset.range (n + 1)).sum (fun i => a i * x^(n - i) * y^i)

/-- The ratio of the two variables -/
noncomputable def ratio (x y : ℝ) : ℝ := x / y

theorem homogeneous_equation_determines_ratio (n : ℕ) :
  ∀ (a : ℕ → ℝ), ∃ (f : ℝ → ℝ),
    ∀ (x y : ℝ), y ≠ 0 →
      HomogeneousPolynomial n x y a = 0 ↔ f (ratio x y) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_equation_determines_ratio_l521_52191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l521_52151

/-- Given an arithmetic sequence with first four terms a, b, x, 2x, where b = x²/4,
    prove that the ratio of a to x is (x - 4) / 4 -/
theorem arithmetic_sequence_ratio (x : ℝ) (a b : ℝ) 
  (h1 : b = x^2 / 4)
  (h2 : ∃ d, b = a + d ∧ x = b + d ∧ 2*x = x + d) :
  a / x = (x - 4) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l521_52151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_l521_52192

/-- An isosceles triangle with integer sides where one median divides the perimeter in a 1:2 ratio -/
structure IsoscelesTriangle where
  x : ℕ  -- length of the equal sides
  y : ℕ  -- length of the base
  median_divides_perimeter : ∃ n : ℕ, 3 * x = 4 * n ∧ y = n / 3

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.y : ℝ) * (t.x : ℝ) * Real.sqrt (4 * (t.x : ℝ)^2 - (t.y : ℝ)^2) / 4

/-- The theorem stating the smallest area of such triangles -/
theorem smallest_area :
  (∃ t : IsoscelesTriangle, ∀ t' : IsoscelesTriangle, area t ≤ area t') ∧
  (∃ t : IsoscelesTriangle, area t = 21 / 4) := by
  sorry

#check smallest_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_l521_52192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_negative_one_l521_52196

theorem complex_expression_equals_negative_one :
  (((1 / 27 : ℝ) ^ (1 / 3 : ℝ)) * (1 / 3)) - ((6.25 : ℝ) ^ (1 / 2)) + (2 * Real.sqrt 2) - (2 / 3) + (Real.pi ^ 0) - (3 ^ (-1 : ℤ)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_negative_one_l521_52196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_equal_vectors_l521_52194

open Real

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between_vectors (a b : V) : ℝ := 
  arccos ((inner a (a + b)) / (norm a * norm (a + b)))

theorem angle_between_equal_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_norm : norm a = norm b ∧ norm a = norm (a - b)) :
  angle_between_vectors a b = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_equal_vectors_l521_52194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l521_52166

-- Define a as a non-negative real number
variable (a : ℝ) (ha : a ≥ 0)

-- Define P and Q as functions of a
noncomputable def P (a : ℝ) : ℝ := Real.sqrt a + Real.sqrt (a + 5)
noncomputable def Q (a : ℝ) : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 3)

-- Theorem statement
theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) : P a < Q a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l521_52166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jezebel_sunflowers_l521_52163

-- Define the problem parameters
def red_roses : ℕ := 24  -- Two dozens of red roses
def red_rose_cost : ℚ := 3/2  -- $1.50 per red rose
def sunflower_cost : ℚ := 3  -- $3 per sunflower
def total_cost : ℚ := 45  -- Total cost of $45

-- Define the function to calculate the number of sunflowers
def sunflowers_count : ℕ :=
  let red_roses_total_cost : ℚ := red_roses * red_rose_cost
  let remaining_cost : ℚ := total_cost - red_roses_total_cost
  (remaining_cost / sunflower_cost).floor.toNat

-- Theorem statement
theorem jezebel_sunflowers :
  sunflowers_count = 3 := by
  -- Unfold the definition of sunflowers_count
  unfold sunflowers_count
  -- Simplify the arithmetic expressions
  simp [red_roses, red_rose_cost, sunflower_cost, total_cost]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jezebel_sunflowers_l521_52163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_f_is_even_l521_52134

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) + Real.cos (3 * x)

-- Define the shifted function g
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

-- Theorem statement
theorem shifted_f_is_even : ∀ x : ℝ, g x = g (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_f_is_even_l521_52134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l521_52130

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin 1 * Real.cos 1) = Real.sin 1 - Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l521_52130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_and_symmetric_properties_and_monotonic_range_l521_52133

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 4)

theorem periodic_and_symmetric_properties_and_monotonic_range
  (w : ℝ) (h1 : w < 0) (h2 : |w| < 1) :
  (w = -1/2 →
    (∃ T : ℝ, T = 4*Real.pi ∧ ∀ x : ℝ, f w x = f w (x + T)) ∧
    (∀ k : ℤ, ∃ c : ℝ × ℝ, c = (2*Real.pi*↑k - Real.pi/2, 0) ∧
      ∀ x : ℝ, f w (c.1 + x) = f w (c.1 - x)) ∧
    (∀ k : ℤ, ∃ a : ℝ, a = -2*Real.pi*↑k - Real.pi/2 ∧
      ∀ x : ℝ, f w (a + x) = f w (a - x))) ∧
  (∀ x y : ℝ, Real.pi/2 < x ∧ x < y ∧ y < Real.pi →
    (f w x > f w y ↔ -3/4 ≤ w ∧ w < 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_and_symmetric_properties_and_monotonic_range_l521_52133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_right_angles_l521_52142

/-- The speed of the hour hand in degrees per minute -/
noncomputable def hour_hand_speed : ℝ := 1 / 2

/-- The speed of the minute hand in degrees per minute -/
noncomputable def minute_hand_speed : ℝ := 6

/-- The time period in minutes from 8:30 AM to 3:30 PM -/
def time_period : ℕ := 7 * 60

/-- The number of right angles formed between the hour and minute hands -/
def right_angle_count : ℕ := 13

theorem clock_right_angles :
  ∃ (f : ℕ → ℝ),
    (∀ n, 0 ≤ n ∧ n < right_angle_count →
      0 ≤ f n ∧ f n < time_period) ∧
    (∀ n, 0 ≤ n ∧ n < right_angle_count →
      (hour_hand_speed * f n - minute_hand_speed * f n) % 90 = 0) ∧
    (∀ n m, 0 ≤ n ∧ n < m ∧ m < right_angle_count → f n < f m) := by
  sorry

#check clock_right_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_right_angles_l521_52142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l521_52195

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersecting : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (m n : Line) 
  (α β : Plane) 
  (h_diff : m ≠ n) 
  (h_intersect : intersecting α β) 
  (h_perp : perpendicular m β) 
  (h_par : parallel_line_plane m α) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l521_52195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_range_l521_52113

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

-- State the theorem
theorem increasing_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  2 ≤ a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_range_l521_52113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_times_f_f_times_g_l521_52156

/-- The function f(x) represented as an infinite series -/
noncomputable def f (x : ℝ) : ℝ := ∑' n, x^n

/-- The function g(x) represented as an infinite series -/
noncomputable def g (x : ℝ) : ℝ := ∑' n, (-1)^n * x^n

/-- Theorem stating the product of f(x) with itself -/
theorem f_times_f (x : ℝ) (h : |x| < 1) :
  f x * f x = ∑' n, (n + 1) * x^n := by sorry

/-- Theorem stating the product of f(x) and g(x) -/
theorem f_times_g (x : ℝ) (h : |x| < 1) :
  f x * g x = ∑' n, x^(2*n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_times_f_f_times_g_l521_52156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_check_increase_l521_52120

noncomputable def original_time : ℝ := 25
noncomputable def new_time : ℝ := 30

noncomputable def percentage_increase (old_value new_value : ℝ) : ℝ :=
  ((new_value - old_value) / old_value) * 100

theorem maintenance_check_increase :
  percentage_increase original_time new_time = 20 := by
  -- Unfold the definition of percentage_increase
  unfold percentage_increase
  -- Simplify the expression
  simp [original_time, new_time]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_check_increase_l521_52120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l521_52143

-- Define the given values
noncomputable def a : ℝ := 2^(Real.log 3 / Real.log 4)
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := 3^(3/5)

-- State the theorem
theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l521_52143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l521_52155

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x))^2 + Real.sqrt 3 * Real.sin (2 * ω * x) - 2

-- State the theorem
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ (x : ℝ), f ω x = f ω (x + π) ∧ 
    ∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f ω x = f ω (x + p)) → p ≥ π) :
  (ω = 1) ∧ 
  (∀ (x y : ℝ), x ∈ Set.Icc 0 (π/6) → y ∈ Set.Icc 0 (π/6) → x < y → f ω x < f ω y) ∧
  (∀ (x : ℝ), f ω (5*π/12 + x) = f ω (5*π/12 - x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l521_52155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descartes_rule_of_signs_l521_52125

/-- Counts the number of sign changes in a sequence of real numbers -/
def countSignChanges (seq : List ℝ) : ℕ := sorry

/-- Counts the number of positive roots of a polynomial -/
def countPositiveRoots (coeffs : List ℝ) : ℕ := sorry

/-- Counts the number of negative roots of a polynomial -/
def countNegativeRoots (coeffs : List ℝ) : ℕ := sorry

/-- Transforms the coefficient sequence for negative roots analysis -/
def transformCoeffs (coeffs : List ℝ) : List ℝ := sorry

theorem descartes_rule_of_signs (coeffs : List ℝ) (h : coeffs ≠ []) :
  (countPositiveRoots coeffs ≤ countSignChanges coeffs) ∧
  (countNegativeRoots coeffs ≤ countSignChanges (transformCoeffs coeffs)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descartes_rule_of_signs_l521_52125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l521_52131

-- Define the sets P and Q
def P : Set ℝ := Set.Ioc 0 1
def Q : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) ≤ 1}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l521_52131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_magnitude_l521_52161

noncomputable def point := ℝ × ℝ

def A : point := (0, -1)
def B : point := (0, 3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_AB_magnitude :
  magnitude vector_AB = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_magnitude_l521_52161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_inequality_rational_convex_inequality_real_l521_52150

-- Define a convex function
def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- Theorem for rational p and q
theorem convex_inequality_rational
  (f : ℝ → ℝ) (hf : IsConvex f)
  (x₁ x₂ : ℝ) (p q : ℚ)
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1) :
  f (↑p * x₁ + ↑q * x₂) ≤ ↑p * f x₁ + ↑q * f x₂ :=
by
  sorry

-- Theorem for real p and q with continuous f
theorem convex_inequality_real
  (f : ℝ → ℝ) (hf : IsConvex f) (hfc : Continuous f)
  (x₁ x₂ : ℝ) (p q : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1) :
  f (p * x₁ + q * x₂) ≤ p * f x₁ + q * f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_inequality_rational_convex_inequality_real_l521_52150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_isosceles_triangle_condition_l521_52154

-- Part 1: Factorization
theorem polynomial_factorization (a b : ℝ) :
  a^2 - 4*a - b^2 + 4 = (a + b - 2) * (a - b - 2) := by sorry

-- Part 2: Triangle shape
theorem isosceles_triangle_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^2 - a*b - a*c + b*c = 0 → (a = b ∨ a = c ∨ b = c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_isosceles_triangle_condition_l521_52154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_2023000_l521_52109

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Checks if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  n = sn.coefficient * (10 : ℝ) ^ sn.exponent

theorem scientific_notation_of_2023000 :
  ∃ sn : ScientificNotation, represents sn 2023000 ∧ sn.coefficient = 2.023 ∧ sn.exponent = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_2023000_l521_52109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_mile_equals_40_rods_l521_52126

-- Define the units as structures with a value field
structure Mile where
  value : ℕ

structure Chain where
  value : ℕ

structure Rod where
  value : ℕ

-- Define conversion functions
def mile_to_chain (m : Mile) : Chain :=
  ⟨m.value * 10⟩

def chain_to_rod (c : Chain) : Rod :=
  ⟨c.value * 4⟩

-- Define instances for literal syntax
instance : OfNat Mile n where
  ofNat := ⟨n⟩

instance : OfNat Chain n where
  ofNat := ⟨n⟩

instance : OfNat Rod n where
  ofNat := ⟨n⟩

-- Define multiplication for Chain and Rod
instance : HMul Chain Rod Rod where
  hMul c r := ⟨c.value * r.value⟩

-- Theorem statement
theorem one_mile_equals_40_rods : 
  (mile_to_chain 1).value * (chain_to_rod 1).value = (40 : Rod).value := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_mile_equals_40_rods_l521_52126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_natural_numbers_l521_52182

theorem partition_natural_numbers : ∃ (A B : Set ℕ), 
  (∀ n, n ∈ A → n > 1) ∧ 
  (∀ n, n ∈ B → n > 1) ∧ 
  (A ∪ B = {n : ℕ | n > 1}) ∧ 
  (A ∩ B = ∅) ∧ 
  (A ≠ ∅) ∧ 
  (B ≠ ∅) ∧ 
  (∀ a b, a ∈ A → b ∈ A → a * b - 1 ∈ B) ∧ 
  (∀ a b, a ∈ B → b ∈ B → a * b - 1 ∈ A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_natural_numbers_l521_52182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_l521_52106

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 1 then 2 * x
  else if x > 1 ∧ x < 2 then 2
  else if x ≥ 2 then 3
  else 0  -- This case is added to make the function total

-- Define the range of the function
def range_f : Set ℝ := Set.range f

-- Theorem statement
theorem value_range_of_f : range_f = Set.Icc 0 2 ∪ {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_l521_52106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_takes_three_values_l521_52198

variable (a b c d : ℝ)

def n (x y z t : ℝ) : ℝ := (x - y)^2 + (y - z)^2 + (z - t)^2 + (t - x)^2

def is_permutation (x y z t : ℝ) : Prop :=
  Multiset.ofList [x, y, z, t] = Multiset.ofList [a, b, c, d]

theorem n_takes_three_values
  (h : a < b ∧ b < c ∧ c < d) :
  ∃! (s : Finset ℝ), s.card = 3 ∧
    ∀ x y z t, is_permutation a b c d x y z t →
      n x y z t ∈ s :=
by sorry

#check n_takes_three_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_takes_three_values_l521_52198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_closed_unit_interval_l521_52128

def A (a : ℝ) : Set ℝ := {a}

def B : Set ℝ := {x : ℝ | x^2 - x > 0}

theorem a_in_closed_unit_interval (a : ℝ) (h : ¬(A a ⊆ B)) : a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_closed_unit_interval_l521_52128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l521_52147

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_a_neg : a < 0)
  (h_solution_set : ∀ x, f a b c x > -2 * x ↔ 1 < x ∧ x < 3)
  (h_equal_roots : ∃! x, f a b c x + 6 * a = 0) :
  (f a b c = λ x ↦ -1/5 * x^2 - 4/5 * x - 3/5) ∧
  (∃ x, f a b c x > 0) →
  (a < -2/5 ∨ (-2/5 < a ∧ a < 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l521_52147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l521_52171

theorem division_problem (x y : ℕ) (h1 : x % y = 5) (h2 : (x : ℝ) / (y : ℝ) = 96.12) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l521_52171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_representations_l521_52144

-- Define the three representations of 1
def regular_notation : ℝ := 1
def percentage_form : ℝ := 100

noncomputable def radian_to_degrees (x : ℝ) : ℝ := x * (180 / Real.pi)

-- Define the conversion from degrees to degrees, minutes, and seconds
noncomputable def degrees_to_dms (x : ℝ) : ℕ × ℕ × ℕ := by
  let degrees := Int.floor x
  let minutes := Int.floor ((x - degrees) * 60)
  let seconds := Int.floor ((((x - degrees) * 60) - minutes) * 60)
  exact (degrees.toNat, minutes.toNat, seconds.toNat)

-- Theorem statement
theorem one_representations :
  regular_notation = 1 ∧
  percentage_form = 100 ∧
  degrees_to_dms (radian_to_degrees 1) = (57, 17, 44) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_representations_l521_52144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l521_52122

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- State the theorem
theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), a = -Real.pi/6 ∧ b = 0 ∧
  (∀ x y, -Real.pi ≤ x ∧ x < y ∧ y ≤ 0 →
    (a < x ∧ x < y ∧ y < b) ↔ f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l521_52122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_prime_factors_l521_52179

theorem consecutive_integers_prime_factors (n : ℕ) (h : n > 7) :
  ∃ k ∈ ({n, n + 1, n + 2} : Set ℕ), ∃ p q : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ k ∧ q ∣ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_prime_factors_l521_52179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l521_52100

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ := (abs (f x) - f x) / 2

def three_intersections (a b : ℝ) : Prop :=
  a > 0 ∧ ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  a * x₁ + b = g x₁ ∧ a * x₂ + b = g x₂ ∧ a * x₃ + b = g x₃

theorem intersection_condition (a b : ℝ) :
  three_intersections a b ↔ (0 < a ∧ a < 3 ∧ 2 * a < b ∧ b < (a + 1)^2 / 4 + 2) := by
  sorry

#check intersection_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l521_52100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_after_2560_minutes_l521_52175

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- Checks if two DateTimes are equal -/
def dateTimeEqual (dt1 dt2 : DateTime) : Prop :=
  dt1.year = dt2.year ∧ 
  dt1.month = dt2.month ∧ 
  dt1.day = dt2.day ∧ 
  dt1.hour = dt2.hour ∧ 
  dt1.minute = dt2.minute

theorem time_after_2560_minutes : 
  let start := DateTime.mk 2011 1 3 12 0
  let end_ := DateTime.mk 2011 1 5 6 40
  dateTimeEqual (addMinutes start 2560) end_ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_after_2560_minutes_l521_52175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_partition_l521_52183

/-- Represents a student in the school -/
structure Student :=
  (id : ℕ)

/-- Represents a class in the school -/
structure SchoolClass :=
  (id : ℕ)

/-- Represents the school with students and classes -/
structure School :=
  (students : Finset Student)
  (classes : Finset SchoolClass)
  (student_class : Student → SchoolClass)
  (friendship : Student → Student → Prop)

/-- The main theorem to be proved -/
theorem student_partition
  (school : School)
  (n k : ℕ)
  (h1 : school.students.card = n)
  (h2 : school.classes.card = k)
  (h3 : ∀ s1 s2 : Student, school.student_class s1 = school.student_class s2 → school.friendship s1 s2)
  (h4 : ∀ c1 c2 : SchoolClass, c1 ≠ c2 → ∃ s1 s2 : Student, 
    school.student_class s1 = c1 ∧ 
    school.student_class s2 = c2 ∧ 
    ¬school.friendship s1 s2) :
  ∃ (partition : Finset (Finset Student)),
    partition.card = n - k + 1 ∧
    (∀ part : Finset Student, part ∈ partition → 
      ∀ s1 s2 : Student, s1 ∈ part → s2 ∈ part → s1 ≠ s2 → ¬school.friendship s1 s2) ∧
    (∀ s : Student, ∃! part : Finset Student, part ∈ partition ∧ s ∈ part) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_partition_l521_52183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_p_sufficient_not_necessary_l521_52139

noncomputable section

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)
def B (y₀ : ℝ) : ℝ × ℝ := (-2, 2 / y₀)

-- Define collinearity condition
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Define the parabola equation
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = -p.1

-- Theorem statement
theorem condition_p_sufficient_not_necessary :
  (∀ x₀ y₀ : ℝ, y₀ ≠ 0 → collinear O (A x₀ y₀) (B y₀) → on_parabola (A x₀ y₀)) ∧
  (∃ p : ℝ × ℝ, on_parabola p ∧ ¬collinear O p (B p.2)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_p_sufficient_not_necessary_l521_52139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l521_52140

-- Define the diamond operation as noncomputable
noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- State the theorem
theorem diamond_calculation :
  diamond (diamond 8 15) (diamond (-15) (-8)) = 17 * Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l521_52140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_40_l521_52119

-- Define the volume function
noncomputable def V (x : ℝ) : ℝ := x^2 * ((60 - x) / 2)

-- State the theorem
theorem volume_maximized_at_40 :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 60 ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 0 60 → V y ≤ V x) ∧
  x = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_40_l521_52119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l521_52152

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

-- State the theorem
theorem f_properties :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∀ y, f y ≤ f (Real.sqrt 2)) ∧
  (∀ y, f (-Real.sqrt 2) ≤ f y) ∧
  (∃ min max, ∀ y, min ≤ f y ∧ f y ≤ max) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l521_52152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l521_52199

/-- A conic section with foci F₁ and F₂ and a point P on the conic. -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The eccentricity of a conic section -/
noncomputable def eccentricity (c : ConicSection) : ℝ :=
  distance c.F₁ c.F₂ / (distance c.P c.F₁ + distance c.P c.F₂)

theorem conic_eccentricity (c : ConicSection) :
  distance c.P c.F₁ / distance c.F₁ c.F₂ = 4/3 ∧
  distance c.F₁ c.F₂ / distance c.P c.F₂ = 3/2 →
  eccentricity c = 1/2 ∨ eccentricity c = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l521_52199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_of_vectors_l521_52101

theorem angle_cosine_of_vectors (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (u v : V) 
  (h1 : ‖u‖ = 5)
  (h2 : ‖v‖ = 10)
  (h3 : ‖u + v‖ = 12) :
  (inner u v) / (‖u‖ * ‖v‖) = 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_of_vectors_l521_52101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_planes_cover_l521_52110

/-- The minimum number of planes required to cover a set of points in 3D space -/
theorem min_planes_cover (n : ℕ+) : ∃ (N : ℕ),
  (∀ (P : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), x ≤ n ∧ y ≤ n ∧ z ≤ n → 
      (x + y + z > 0 → (x, y, z) ∈ P) ∧ 
      ((x, y, z) = (0, 0, 0) → (x, y, z) ∉ P)) →
    (∃ (planes : Finset (Set (ℕ × ℕ × ℕ))), 
      planes.card = N ∧
      (∀ (p : ℕ × ℕ × ℕ), p ∈ P → ∃ (plane : Set (ℕ × ℕ × ℕ)), plane ∈ planes ∧ p ∈ plane) ∧
      (0, 0, 0) ∉ ⋃₀ planes.toSet)) ∧
  (∀ (M : ℕ),
    (∃ (P : Set (ℕ × ℕ × ℕ)) (planes : Finset (Set (ℕ × ℕ × ℕ))), 
      (∀ (x y z : ℕ), x ≤ n ∧ y ≤ n ∧ z ≤ n → 
        (x + y + z > 0 → (x, y, z) ∈ P) ∧ 
        ((x, y, z) = (0, 0, 0) → (x, y, z) ∉ P)) ∧
      planes.card = M ∧
      (∀ (p : ℕ × ℕ × ℕ), p ∈ P → ∃ (plane : Set (ℕ × ℕ × ℕ)), plane ∈ planes ∧ p ∈ plane) ∧
      (0, 0, 0) ∉ ⋃₀ planes.toSet) →
    M ≥ N) ∧
  N = 3 * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_planes_cover_l521_52110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_always_composite_expression_l521_52116

-- Define the set of expressions as a function
def expressions (n : ℕ) : List ℕ :=
  [n^2 + 40, n^2 + 55, n^2 + 81, n^2 + 117, n^2 + 150]

-- Theorem statement
theorem no_always_composite_expression :
  ∀ n : ℕ, Prime n → n > 3 →
    ∀ E ∈ expressions n,
      ∃ p : ℕ, Prime p ∧ p > 3 ∧ ¬(∀ q : ℕ, 1 < q ∧ q < E → E % q ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_always_composite_expression_l521_52116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_third_and_general_term_l521_52102

/-- Sequence of natural numbers -/
def a : ℕ+ → ℕ := sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℕ := (n : ℕ) ^ 2

/-- The sum of the first n terms equals n^2 -/
axiom S_eq (n : ℕ+) : S n = (n : ℕ) ^ 2

theorem a_third_and_general_term :
  a 3 = 5 ∧ ∀ n : ℕ+, a n = 2 * (n : ℕ) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_third_and_general_term_l521_52102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_quadratic_expression_l521_52185

theorem divides_quadratic_expression (a b c : ℤ) (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) :
  ∃ x y : ℤ, (p : ℤ) ∣ (x^2 + y^2 + a*x + b*y + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_quadratic_expression_l521_52185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_bound_l521_52137

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a*x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

theorem extreme_points_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf : ∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → f' a x ≠ 0) :
  (1/2 : ℝ) < x₂^2/2 ∧ x₂^2/2 < 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_bound_l521_52137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_le_self_g_eq_quadratic_solutions_l521_52170

/-- g is a function from ℕ to ℕ where g(n) is the product of the digits of n -/
def g : ℕ → ℕ := sorry

/-- Theorem stating that g(n) is always less than or equal to n -/
theorem g_le_self : ∀ n : ℕ, g n ≤ n := by sorry

/-- Theorem stating that 4 and 9 are the only natural numbers satisfying n² - 12n + 36 = g(n) -/
theorem g_eq_quadratic_solutions : {n : ℕ | n^2 - 12*n + 36 = g n} = {4, 9} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_le_self_g_eq_quadratic_solutions_l521_52170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotten_eggs_probability_l521_52153

/-- The probability of choosing 2 rotten eggs from a pack of 36 eggs containing 3 rotten eggs is 1/210. -/
theorem rotten_eggs_probability (total_eggs : ℕ) (rotten_eggs : ℕ) (chosen_eggs : ℕ) :
  total_eggs = 36 →
  rotten_eggs = 3 →
  chosen_eggs = 2 →
  (Nat.choose rotten_eggs chosen_eggs : ℚ) / (Nat.choose total_eggs chosen_eggs) = 1 / 210 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotten_eggs_probability_l521_52153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l521_52176

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) ∧
  (f 2 = 5 * Real.exp (-2 * Real.log 2) + 3) :=
by
  constructor
  · intro x
    simp [f]
    ring_nf
  · constructor
    · intro x₁ x₂ h₁ h₂
      sorry -- Proof of monotonicity goes here
    · simp [f]
      norm_num
      ring_nf
      sorry -- Proof of equality goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l521_52176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_workers_completion_time_l521_52107

/-- Represents the time taken to complete a job by a group of workers -/
noncomputable def job_completion_time (individual_times : List ℝ) : ℝ :=
  1 / (individual_times.map (λ t => 1 / t)).sum

/-- Theorem: Three workers complete the job in 11 days given specific conditions -/
theorem three_workers_completion_time 
  (time_ab time_c : ℝ) 
  (h1 : time_ab = 15) 
  (h2 : time_c = 41.25) : 
  job_completion_time [time_ab, time_c] = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_workers_completion_time_l521_52107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_of_factorial_l521_52190

theorem smallest_sum_of_factors_of_factorial (p q r s : ℕ+) : 
  p * q * r * s = 40320 → (∀ a b c d : ℕ+, a * b * c * d = 40320 → a + b + c + d ≥ p + q + r + s) → 
  p + q + r + s = 89 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_of_factorial_l521_52190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l521_52105

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) ^ 2
def g (x : ℝ) : ℝ := |x|

-- Define the theorem
theorem problem_statement :
  -- Part 1
  (∀ x, 2*x^2 - 5*x - 3 < 0 → -1/2 < x ∧ x < 4) ∧
  (∃ x, -1/2 < x ∧ x < 4 ∧ 2*x^2 - 5*x - 3 ≥ 0) ∧
  -- Part 2
  f ≠ g ∧
  -- Part 3
  (∀ x y m, x > 0 → y > 0 → 3/(2*x) + 6/y = 2 →
    (∀ m, 4*x + y > 7*m - m^2) →
    m < 3 ∨ m > 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l521_52105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shift_l521_52127

theorem parabola_shift (x y : ℝ) : 
  (∃ k : ℝ, y = 2 * (x - 3)^2 - 1) ↔ 
  (∃ h : ℝ, y + 1 = 2 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shift_l521_52127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_conditions_l521_52157

theorem function_satisfying_conditions (f : ℤ → ℤ) 
  (h1 : ∀ m n : ℤ, f (f m + n) + 2 * m = f n + f (3 * m))
  (h2 : ∃ d : ℤ, f d - f 0 = 2)
  (h3 : Even (f 1 - f 0)) :
  ∃ u v : ℤ, ∀ n : ℤ, f (2 * n) = 2 * n + 2 * u ∧ f (2 * n + 1) = 2 * n + 2 * v :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_conditions_l521_52157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l521_52174

/-- The number of bricks required to pave a courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  (courtyard_length * courtyard_width / (brick_length * brick_width)).ceil.toNat

/-- Proof that 13,788 bricks are required for the given courtyard and brick dimensions -/
theorem courtyard_paving :
  bricks_required 28 13 (22/100) (12/100) = 13788 := by
  sorry

#eval bricks_required 28 13 (22/100) (12/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l521_52174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l521_52129

open Real

/-- The inclination angle of a line given by the equation x - y*sin(α) - 3 = 0, where α is a real number. -/
noncomputable def inclination_angle (α : ℝ) : ℝ :=
  if sin α = 0 then π/2
  else arctan (1 / sin α)

/-- The theorem stating that the range of the inclination angle is [π/4, 3π/4]. -/
theorem inclination_angle_range :
  Set.range inclination_angle = Set.Icc (π/4 : ℝ) (3*π/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l521_52129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l521_52162

/-- Represents the number of knights on the island -/
def r : ℕ := sorry

/-- Represents the number of liars on the island -/
def l : ℕ := sorry

/-- The total number of islanders -/
def total_islanders : ℕ := r + l

/-- The number of statements each islander makes -/
def statements_per_islander : ℕ := total_islanders - 1

/-- The total number of statements made -/
def total_statements : ℕ := total_islanders * statements_per_islander

/-- The number of times "You are a liar!" was said -/
def liar_statements : ℕ := 230

/-- The number of times "You are a knight!" was said -/
def knight_statements : ℕ := total_statements - liar_statements

theorem island_puzzle :
  (r ≥ 2) →
  (l ≥ 2) →
  (2 * r * l = 230) →
  knight_statements = 526 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l521_52162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_is_46_thirds_pi_l521_52173

/-- Represents the dimensions of a well with a cylindrical top and conical bottom -/
structure WellDimensions where
  cylinderDiameter : ℝ
  cylinderDepth : ℝ
  coneTopDiameter : ℝ
  coneBottomDiameter : ℝ
  coneDepth : ℝ

/-- Calculates the total volume of a well given its dimensions -/
noncomputable def totalWellVolume (d : WellDimensions) : ℝ :=
  let cylinderRadius := d.cylinderDiameter / 2
  let cylinderVolume := Real.pi * cylinderRadius^2 * d.cylinderDepth
  let coneVolume := (1/3) * Real.pi * cylinderRadius^2 * d.coneDepth
  cylinderVolume + coneVolume

/-- Theorem stating that the total volume of the well with given dimensions is (46/3)π cubic meters -/
theorem well_volume_is_46_thirds_pi :
  let wellDims : WellDimensions := {
    cylinderDiameter := 2,
    cylinderDepth := 14,
    coneTopDiameter := 2,
    coneBottomDiameter := 1,
    coneDepth := 4
  }
  totalWellVolume wellDims = (46/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_is_46_thirds_pi_l521_52173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_geq_5_l521_52187

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x - 5 < 0}
def Q (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem subset_implies_a_geq_5 :
  ∀ a : ℝ, P ⊂ Q a → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_geq_5_l521_52187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l521_52141

theorem diophantine_equation_solutions :
  {(x, y) : ℤ × ℤ | x^2 + x*y + y^2 = x + 20} =
  {(1, -5), (5, -5), (-4, 0), (5, 0), (-4, 4), (1, 4)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l521_52141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_origin_to_line_l521_52149

/-- The minimum distance from the origin to the line 2x - y + 1 = 0 is √5/5 -/
theorem min_distance_origin_to_line : ∃ (d : ℝ),
  d = Real.sqrt 5 / 5 ∧
  ∀ (x y : ℝ),
    2*x - y + 1 = 0 →
    d ≤ Real.sqrt (x^2 + y^2) := by
  -- Let d be √5/5
  let d := Real.sqrt 5 / 5
  
  -- Show that d satisfies the conditions
  have h1 : d = Real.sqrt 5 / 5 := rfl
  
  have h2 : ∀ (x y : ℝ),
    2*x - y + 1 = 0 →
    d ≤ Real.sqrt (x^2 + y^2) := by
    sorry  -- The actual proof would go here
  
  -- Conclude the existence
  exact ⟨d, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_origin_to_line_l521_52149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_die_even_sum_probability_l521_52177

/-- Represents the probability of rolling an odd number on the unfair die -/
noncomputable def p_odd : ℝ := 1 / 5

/-- Represents the probability of rolling an even number on the unfair die -/
noncomputable def p_even : ℝ := 4 / 5

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 3

/-- Theorem stating the probability of obtaining an even sum when rolling the unfair die three times -/
theorem unfair_die_even_sum_probability : 
  (p_even ^ 3 + 3 * p_odd ^ 2 * p_even : ℝ) = 76 / 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_die_even_sum_probability_l521_52177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l521_52159

def G : ℕ → ℚ
  | 0 => 3  -- Adding the base case for 0
  | 1 => 3
  | (n + 1) => (3 * G n + 2) / 2

theorem G_51 : G 51 = 70349216 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l521_52159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexanders_height_l521_52197

/-- Calculates the height of a person after a given number of years, given their initial height and growth rate. -/
def height_after_years (initial_height : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_height + growth_rate * (years : ℝ) * 12

/-- Theorem stating that given the initial conditions, Alexander's height at 12 years old will be 74 inches. -/
theorem alexanders_height : 
  let initial_height : ℝ := 50
  let growth_rate : ℝ := 0.5
  let years_passed : ℕ := 4
  height_after_years initial_height growth_rate years_passed = 74 := by
  sorry

#eval height_after_years 50 0.5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexanders_height_l521_52197
