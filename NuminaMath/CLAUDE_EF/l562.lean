import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_opposite_side_specific_l562_56248

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  -- Base side length
  baseLength : ℝ
  -- Exterior angle at the apex
  exteriorAngle : ℝ

/-- The distance from the base vertex to the opposite side in the triangle -/
noncomputable def distanceToOppositeSide (t : IsoscelesTriangle) : ℝ :=
  t.baseLength / 2

/-- Theorem: In an isosceles triangle with base 37 and exterior angle 60°,
    the distance from a base vertex to the opposite side is 18.5 -/
theorem distance_to_opposite_side_specific :
  let t : IsoscelesTriangle := { baseLength := 37, exteriorAngle := 60 }
  distanceToOppositeSide t = 18.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_opposite_side_specific_l562_56248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secants_construction_l562_56265

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problem_setup (c1 c2 : Circle) (P : Point) (α : ℝ) : Prop :=
  -- Circles do not intersect
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > (c1.radius + c2.radius)^2 ∧
  -- P is outside both circles
  (P.x - c1.center.1)^2 + (P.y - c1.center.2)^2 > c1.radius^2 ∧
  (P.x - c2.center.1)^2 + (P.y - c2.center.2)^2 > c2.radius^2

-- Define the existence of secants
def secants_exist (c1 c2 : Circle) (P : Point) (α : ℝ) : Prop :=
  ∃ (s1 s2 : ℝ → ℝ → Prop),
    -- s1 and s2 are lines passing through P
    s1 P.x P.y ∧ s2 P.x P.y ∧
    -- s1 and s2 intersect both circles
    (∃ (x1 y1 x2 y2 : ℝ), s1 x1 y1 ∧ s1 x2 y2 ∧ (x1 - c1.center.1)^2 + (y1 - c1.center.2)^2 = c1.radius^2 ∧
                          (x2 - c1.center.1)^2 + (y2 - c1.center.2)^2 = c1.radius^2) ∧
    (∃ (x3 y3 x4 y4 : ℝ), s2 x3 y3 ∧ s2 x4 y4 ∧ (x3 - c2.center.1)^2 + (y3 - c2.center.2)^2 = c2.radius^2 ∧
                          (x4 - c2.center.1)^2 + (y4 - c2.center.2)^2 = c2.radius^2) ∧
    -- Chords have equal length
    ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
      (x1 - x2)^2 + (y1 - y2)^2 = (x3 - x4)^2 + (y3 - y4)^2 ∧
    -- s1 and s2 form angle α
    ∃ (m1 m2 : ℝ), |Real.arctan m1 - Real.arctan m2| = α

-- Theorem statement
theorem secants_construction (c1 c2 : Circle) (P : Point) (α : ℝ) :
  problem_setup c1 c2 P α → secants_exist c1 c2 P α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secants_construction_l562_56265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_moving_circle_center_l562_56267

-- Define the line L
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -1}

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 2)^2 = 1}

-- Define the property of being tangent to a line
def TangentToLine (circle : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of being externally tangent to another circle
def ExternallyTangent (circle1 circle2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the center of a circle
noncomputable def CenterOf (circle : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the locus of points
def Locus (points : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem locus_of_moving_circle_center 
  (M : Set (ℝ × ℝ)) -- Moving circle M
  (h1 : TangentToLine M L)
  (h2 : ExternallyTangent M C) :
  Locus {CenterOf M} = {p : ℝ × ℝ | p.1^2 = 8 * p.2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_moving_circle_center_l562_56267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l562_56228

theorem complex_equation_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let z : ℂ := Complex.ofReal a + Complex.I * Complex.ofReal b
  (z * (z + Complex.I) * (z + 3 * Complex.I) = 2002 * Complex.I) →
  (a^3 - a * (b^2 + 4*b + 3) - (b + 4) * (b^2 + 4*b + 3) = 0 ∧
   a^2 * (b + 4) - b * (b^2 + 4*b + 3) = 2002) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l562_56228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_terms_l562_56230

noncomputable def expansion (x : ℝ) : ℝ → ℝ := fun n ↦ (x^2 - 1/x)^n

def sum_of_coefficients (n : ℕ) : ℕ := 2^n

noncomputable def general_term (n k : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n k) * (x^2)^(n-k) * (-1/x)^k

theorem largest_coefficient_terms (x : ℝ) :
  sum_of_coefficients 7 = 128 →
  ∀ k, 0 ≤ k ∧ k ≤ 7 →
    (abs (general_term 7 3 x) ≥ abs (general_term 7 k x) ∧
     abs (general_term 7 4 x) ≥ abs (general_term 7 k x)) ∧
    general_term 7 3 x = -35 * x^5 ∧
    general_term 7 4 x = 35 * x^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_terms_l562_56230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_main_theorem_l562_56253

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper definition for equilateral triangle -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C

/-- The given conditions for the triangle -/
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c ∧
  t.b + t.c = 2 * t.a ∧
  t.b + t.c = 2 * Real.sqrt 3

theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ t.a = t.b ∧ t.b = t.c := by
  sorry

/-- Main theorem combining both parts of the problem -/
theorem main_theorem (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ IsEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_main_theorem_l562_56253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l562_56227

/-- The equation of a circle in polar coordinates -/
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = -4 * Real.cos θ

/-- The center of the circle in polar coordinates -/
noncomputable def circle_center : ℝ × ℝ := (2, Real.pi)

/-- Theorem stating that the given point is the center of the circle -/
theorem circle_center_correct :
  ∀ ρ θ : ℝ, polar_circle_equation ρ θ → 
  ∃ (r α : ℝ), (r, α) = circle_center ∧ 
  (r * Real.cos α + 4 * Real.cos θ)^2 + (r * Real.sin α)^2 = ρ^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l562_56227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l562_56216

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 8

-- Define the line L
def line_L (x y a b : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the symmetry condition
def symmetry (a b : ℝ) : Prop := -2*a + 2*b + 6 = 0

-- Define the tangent length function
noncomputable def tangent_length (a b : ℝ) : ℝ := Real.sqrt ((a + 1)^2 + (b - 2)^2 - 8)

-- Statement of the theorem
theorem min_tangent_length (a b : ℝ) :
  symmetry a b →
  (∃ (x y : ℝ), circle_C x y ∧ line_L x y a b) →
  (∀ (a' b' : ℝ), tangent_length a' b' ≥ Real.sqrt 10) ∧
  (∃ (a₀ b₀ : ℝ), tangent_length a₀ b₀ = Real.sqrt 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l562_56216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l562_56260

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 1/x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x - 1/x^2

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' point.fst

theorem tangent_line_equation :
  ∀ x y : ℝ, (x - point.fst) * m = y - point.snd ↔ x - y - 1 = 0 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l562_56260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_269_l562_56255

/-- Sequence definition -/
def a (n : ℕ) : ℤ := 16^n + (-1)^n

/-- Theorem statement -/
theorem infinitely_many_divisible_by_269 :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, 269 ∣ a (f k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_269_l562_56255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_D_l562_56258

-- Define the set C
def C : Set ℂ := {w : ℂ | w^4 - 16 = 0}

-- Define the set D
def D : Set ℂ := {w : ℂ | w^4 - 4*w^3 + 16 = 0}

-- Define the distance function between two complex numbers
noncomputable def distance (z₁ z₂ : ℂ) : ℝ := Complex.abs (z₁ - z₂)

-- Theorem statement
theorem max_distance_C_D : 
  ∃ (c : ℂ) (d : ℂ), c ∈ C ∧ d ∈ D ∧ 
    (∀ (c' : ℂ) (d' : ℂ), c' ∈ C → d' ∈ D → distance c d ≥ distance c' d') ∧ 
    distance c d = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_D_l562_56258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_range_of_a_l562_56240

-- Define the sets A, B, and C
def A : Set ℝ := {x | x < -2 ∨ x > 0}
def B : Set ℝ := {x | (1/3 : ℝ)^x ≥ 3}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 1}

-- Theorem for part (I)
theorem union_A_B : A ∪ B = {x | x ≤ -1 ∨ x > 0} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) : A ∩ C a = C a → a < -3 ∨ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_range_of_a_l562_56240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l562_56250

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

def g_odd (g : ℝ → ℝ) : Prop :=
  ∀ x, g x - 1 = -(g (-x) - 1)

def intersection_points (f g : ℝ → ℝ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃ ∧ f x₄ = g x₄

theorem intersection_sum (g : ℝ → ℝ) (x₁ x₂ x₃ x₄ : ℝ) :
  g_odd g →
  intersection_points f g x₁ x₂ x₃ x₄ →
  g x₁ + g x₂ + g x₃ + g x₄ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l562_56250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l562_56276

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1/2 then x/3
  else if 1/2 < x ∧ x ≤ 1 then 2*x^3/(x+1)
  else 0  -- undefined for x outside [0,1]

noncomputable def g (a x : ℝ) : ℝ := a*x - a/2 + 3

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧
    (∀ x₁ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 1 →
      ∃ x₂ : ℝ, 0 ≤ x₂ ∧ x₂ ≤ 1/2 ∧ f x₁ = g a x₂)) →
  a ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l562_56276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l562_56261

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 30

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (4, -3)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (-6, 9)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The distance between the foci of the given ellipse is 2√61 -/
theorem distance_between_foci :
  distance focus1 focus2 = 2 * Real.sqrt 61 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l562_56261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_l562_56235

-- Define the curve C parametrically
noncomputable def C (t : ℝ) : ℝ × ℝ := (2 + (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

-- Theorem stating the equivalence of the parametric and ordinary equations
theorem curve_C_equation : 
  ∀ (x y : ℝ), (∃ t : ℝ, C t = (x, y)) ↔ x - y - 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_l562_56235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_l562_56243

theorem placemat_length (R : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) : 
  R = 5 → n = 8 → w = 1 →
  (2 * R * Real.sin (π / (2 * n)))^2 = w^2 + x^2 →
  R^2 = (w/2)^2 + (x + R * Real.cos (π / n) - w/2)^2 →
  x = Real.sqrt 24.75 - 5 * Real.sqrt ((1 + Real.sqrt 2 / 2) / 2) + 1 / 2 := by
  sorry

#check placemat_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_l562_56243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l562_56264

theorem trig_fraction_value (α : Real) (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l562_56264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_750_l562_56215

/-- Regular hexagon ABCDEF is the base of right pyramid PABCDEF. -/
structure RegularHexagonPyramid where
  /-- Side length of the regular hexagon -/
  side_length : ℝ
  /-- Height of the pyramid -/
  height : ℝ

/-- Volume of a pyramid given its base area and height -/
noncomputable def pyramid_volume (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

/-- Area of a regular hexagon given its side length -/
noncomputable def regular_hexagon_area (side_length : ℝ) : ℝ :=
  6 * ((Real.sqrt 3 / 4) * side_length^2)

/-- Height of an equilateral triangle given its side length -/
noncomputable def equilateral_triangle_height (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * side_length

/-- Theorem stating the volume of the pyramid PABCDEF -/
theorem pyramid_volume_is_750 (p : RegularHexagonPyramid) 
    (h1 : p.side_length = 10) 
    (h2 : p.height = equilateral_triangle_height p.side_length) : 
  pyramid_volume (regular_hexagon_area p.side_length) p.height = 750 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_750_l562_56215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_power_l562_56275

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power : ((1 + i) / (1 - i))^2013 = i := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_power_l562_56275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l562_56203

theorem tan_alpha_value (α : Real) 
  (h1 : α > Real.pi/2 ∧ α < Real.pi) 
  (h2 : Real.cos (2*α) = -3/5) : 
  Real.tan α = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l562_56203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_of_a_l562_56284

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - 2*a*x - 5 else a/x

theorem increasing_f_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-2) (-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_of_a_l562_56284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_l562_56299

theorem sin_cos_equation (x : ℝ) (h : Real.sin x - 5 * Real.cos x = 2) :
  Real.cos x + 5 * Real.sin x = Real.sqrt 46 ∨ Real.cos x + 5 * Real.sin x = -Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_l562_56299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_relationship_l562_56278

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 12 = 0
def C₂ (x y k : ℝ) : Prop := x^2 + y^2 - 2*x - 14*y + k = 0

-- Define the relationship between the circles based on k
def circles_touch (k : ℝ) : Prop := k = 14 ∨ k = 34
def circles_intersect (k : ℝ) : Prop := 14 < k ∧ k < 34
def circles_separate (k : ℝ) : Prop := k < 14 ∨ (34 < k ∧ k < 50)

-- Theorem statement
theorem circle_relationship (k : ℝ) :
  ((∃ x y, C₁ x y ∧ C₂ x y k) ↔ (circles_touch k ∨ circles_intersect k)) ∧
  ((¬∃ x y, C₁ x y ∧ C₂ x y k) ↔ circles_separate k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_relationship_l562_56278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_taxi_charge_for_3_6_miles_l562_56295

/-- A taxi service with a fixed initial fee and per-distance charge -/
structure TaxiService where
  initialFee : ℚ
  chargePerIncrement : ℚ
  incrementDistance : ℚ

/-- Calculate the total charge for a given distance -/
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  let increments := distance / service.incrementDistance
  service.initialFee + service.chargePerIncrement * increments

/-- Jim's taxi service -/
def jimTaxi : TaxiService := {
  initialFee := 235/100
  chargePerIncrement := 35/100
  incrementDistance := 2/5
}

theorem jim_taxi_charge_for_3_6_miles :
  totalCharge jimTaxi (36/10) = 550/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_taxi_charge_for_3_6_miles_l562_56295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l562_56229

/-- Represents the state of the game board -/
structure GameBoard :=
  (eq1 : ℝ → ℝ → Prop)
  (eq2 : ℝ → ℝ → ℝ → Prop)
  (eq3 : ℝ → ℝ → ℝ → ℝ → Prop)
  (eq4 : ℝ → ℝ → ℝ → ℝ → ℝ → Prop)
  (eq5 : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop)
  (eq6 : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop)

/-- Represents a player's move -/
inductive Move
  | Fill : Nat → ℝ → Move

/-- Represents the game state -/
structure GameState :=
  (board : GameBoard)
  (moves : List Move)

/-- Defines a winning strategy for the first player -/
def winning_strategy : GameState → Move :=
  λ _ => sorry

/-- Simulates the game given strategies for both players -/
def simulate_game (initial_board : GameBoard) 
  (first_player_strategy : GameState → Move) 
  (second_player_strategy : GameState → Move) : GameState :=
  sorry

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins (initial_board : GameBoard) :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      let game_result := simulate_game initial_board strategy opponent_strategy
      ∀ (a b c d e f g h i j k l m n o p q r s t u v w x y z aa : ℝ),
        (initial_board.eq1 a b) ∧
        (initial_board.eq2 c d e) ∧
        (initial_board.eq3 f g h i) ∧
        (initial_board.eq4 j k l m n) ∧
        (initial_board.eq5 o p q r s t) ∧
        (initial_board.eq6 u v w x y z aa) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l562_56229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_triangle_area_theorem_l562_56234

/-- Represents a parallelogram with a given area -/
structure Parallelogram where
  area : ℝ

/-- Represents a configuration of four parallelograms forming a larger parallelogram -/
structure ParallelogramConfiguration where
  A : Parallelogram
  B : Parallelogram
  C : Parallelogram
  D : Parallelogram

/-- The area of the triangle formed by the diagonal of the larger parallelogram -/
noncomputable def diagonal_triangle_area (config : ParallelogramConfiguration) : ℝ :=
  (config.A.area + config.B.area) / 2

theorem diagonal_triangle_area_theorem (config : ParallelogramConfiguration) 
  (h1 : config.A.area = 30)
  (h2 : config.B.area = 15)
  (h3 : config.C.area = 20) :
  diagonal_triangle_area config = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_triangle_area_theorem_l562_56234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closest_to_longest_side_is_half_l562_56269

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  angle_inclination : ℝ
  angle_BCD : ℝ
  h : ℝ
  area : ℝ
  parallel : Prop
  longest_side : Prop

/-- The fraction of the area closest to the longest side of the trapezoid -/
noncomputable def fraction_closest_to_longest_side (t : Trapezoid) : ℝ :=
  1 / 2

/-- Theorem stating that the fraction of the area closest to the longest side is 1/2 -/
theorem fraction_closest_to_longest_side_is_half (t : Trapezoid) 
  (h_AB : t.AB = 150)
  (h_CD : t.CD = 200)
  (h_angle_incl : t.angle_inclination = 70 * Real.pi / 180)
  (h_angle_BCD : t.angle_BCD = 135 * Real.pi / 180)
  (h_parallel : t.parallel)
  (h_longest_side : t.longest_side) :
  fraction_closest_to_longest_side t = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closest_to_longest_side_is_half_l562_56269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l562_56280

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angle_at_Q (PQRS : Quadrilateral) : Prop :=
  let (xP, yP) := PQRS.P
  let (xQ, yQ) := PQRS.Q
  let (xR, yR) := PQRS.R
  (xR - xQ) * (xP - xQ) + (yR - yQ) * (yP - yQ) = 0

def is_PR_perpendicular_to_RS (PQRS : Quadrilateral) : Prop :=
  let (xP, yP) := PQRS.P
  let (xR, yR) := PQRS.R
  let (xS, yS) := PQRS.S
  (xR - xP) * (xS - xR) + (yR - yP) * (yS - yR) = 0

noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  let (xA, yA) := A
  let (xB, yB) := B
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

noncomputable def perimeter (PQRS : Quadrilateral) : ℝ :=
  length PQRS.P PQRS.Q + length PQRS.Q PQRS.R + length PQRS.R PQRS.S + length PQRS.S PQRS.P

theorem quadrilateral_perimeter (PQRS : Quadrilateral) :
  is_right_angle_at_Q PQRS →
  is_PR_perpendicular_to_RS PQRS →
  length PQRS.P PQRS.Q = 24 →
  length PQRS.Q PQRS.R = 28 →
  length PQRS.R PQRS.S = 16 →
  perimeter PQRS = 108.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l562_56280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l562_56225

theorem integral_inequality (f : ℝ → ℝ) (n : ℕ) 
  (hf : ContinuousOn f (Set.Icc 0 1)) (hn : n > 0) :
  ∫ x in Set.Icc 0 1, f x ≤ (n + 1 : ℝ) * ∫ x in Set.Icc 0 1, x^n * f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l562_56225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_x_axis_symmetry_proof_l562_56204

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if two lines are symmetric about the X-axis -/
def IsSymmetricAboutXAxis (l1 l2 : Line) : Prop :=
  (l1.a = l2.a) ∧ (l1.b = -l2.b) ∧ (l1.c = l2.c)

/-- Given two lines, this theorem states that they are symmetric with respect to the X-axis -/
theorem symmetric_lines_x_axis (l1 l2 : Line) : 
  IsSymmetricAboutXAxis l1 l2 → 
  (l1.a * l2.a + l1.b * l2.b ≠ 0) →
  True := by
  sorry

def original_line : Line := { a := 1, b := -2, c := 3 }
def symmetric_line : Line := { a := 1, b := 2, c := 3 }

theorem symmetry_proof : 
  IsSymmetricAboutXAxis original_line symmetric_line := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_x_axis_symmetry_proof_l562_56204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_difference_p_q_l562_56209

/-- The efficiency difference between two workers --/
noncomputable def efficiency_difference (p_days q_days combined_days : ℝ) : ℝ :=
  let p_rate := 1 / p_days
  let combined_rate := 1 / combined_days
  let q_rate := combined_rate - p_rate
  ((p_rate - q_rate) / q_rate) * 100

/-- Theorem stating the efficiency difference between p and q --/
theorem efficiency_difference_p_q :
  efficiency_difference 21 (11 * 21 / 10) 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_difference_p_q_l562_56209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_pay_is_16450_l562_56259

/-- Represents the types of workers --/
inductive WorkerType
| A
| B
| C
| D

/-- Daily pay rate for each worker type --/
def dailyPayRate (wt : WorkerType) : ℕ :=
  match wt with
  | .A => 200
  | .B => 250
  | .C => 300
  | .D => 350

/-- Number of workers for each type in the new team --/
def newTeamComposition (wt : WorkerType) : ℕ :=
  match wt with
  | .A => 3
  | .B => 2
  | .C => 3
  | .D => 1

/-- Hours worked per day by the new team --/
def hoursWorkedPerDay : ℕ := 6

/-- Number of working days in a week --/
def workingDaysPerWeek : ℕ := 7

/-- Calculates the weekly pay for the new team --/
def weeklyPayForNewTeam : ℕ :=
  (dailyPayRate WorkerType.A * newTeamComposition WorkerType.A +
   dailyPayRate WorkerType.B * newTeamComposition WorkerType.B +
   dailyPayRate WorkerType.C * newTeamComposition WorkerType.C +
   dailyPayRate WorkerType.D * newTeamComposition WorkerType.D) * workingDaysPerWeek

/-- Theorem: The weekly pay for the new team is 16450 --/
theorem weekly_pay_is_16450 : weeklyPayForNewTeam = 16450 := by
  sorry

#eval weeklyPayForNewTeam

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_pay_is_16450_l562_56259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_games_l562_56210

/-- Represents the number of teams in the basketball league -/
def x : ℕ := sorry

/-- Represents the total number of games in the league -/
def total_games : ℕ := 45

/-- Theorem stating the relationship between the number of teams and total games in a round-robin format -/
theorem round_robin_games : (x * (x - 1)) / 2 = total_games := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_games_l562_56210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l562_56214

/-- The sum of an infinite geometric series with first term a and common ratio r, where |r| < 1 -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio -2/3 and sum 24, the first term is 40 -/
theorem first_term_of_geometric_series :
  ∃ (a : ℝ), infiniteGeometricSum a (-2/3) = 24 ∧ a = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l562_56214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l562_56247

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 2)

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2),
    x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l562_56247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PRQ_in_regular_heptagon_l562_56287

/-- The measure of an angle in a regular heptagon -/
noncomputable def regular_heptagon_angle : ℝ := 180 * 5 / 7

/-- The measure of angle PRQ in a regular heptagon LMNOPQR -/
noncomputable def angle_PRQ : ℝ := (180 - regular_heptagon_angle) / 2

/-- Theorem: In a regular heptagon LMNOPQR, the measure of angle PRQ is (180° - (180° * 5 / 7)) / 2 -/
theorem angle_PRQ_in_regular_heptagon :
  angle_PRQ = (180 - (180 * 5 / 7)) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PRQ_in_regular_heptagon_l562_56287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_distance_formula_third_circle_radius_smallest_nice_c_l562_56232

/-- Three circles with radii 0 < c < b < a are tangent to each other and to a line. -/
structure TangentCircles where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < c
  h2 : c < b
  h3 : b < a

/-- The distance between the points of tangency of two circles to the line. -/
noncomputable def tangencyDistance (circles : TangentCircles) : ℝ := 2 * Real.sqrt (circles.a * circles.b)

/-- The radius of the third circle when a = 16 and b = 4. -/
noncomputable def thirdCircleRadius (circles : TangentCircles) : ℝ := 16 / 9

/-- A configuration is nice if all radii are integers. -/
def isNice (circles : TangentCircles) : Prop :=
  ∃ (x y z : ℕ), (circles.a = x) ∧ (circles.b = y) ∧ (circles.c = z)

theorem tangency_distance_formula (circles : TangentCircles) :
  tangencyDistance circles = 2 * Real.sqrt (circles.a * circles.b) := by
  sorry

theorem third_circle_radius (circles : TangentCircles) (h4 : circles.a = 16) (h5 : circles.b = 4) :
  thirdCircleRadius circles = 16 / 9 := by
  sorry

theorem smallest_nice_c (circles : TangentCircles) (h4 : isNice circles) :
  circles.c ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_distance_formula_third_circle_radius_smallest_nice_c_l562_56232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_implies_scalar_multiple_l562_56207

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem parallel_implies_scalar_multiple (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h_parallel : parallel a b) :
  ∃! (k : ℝ), a = k • b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_implies_scalar_multiple_l562_56207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_side_isosceles_l562_56231

/-- Angle of a triangle given side lengths -/
noncomputable def angle_of_triangle (a b c : ℝ) : ℝ :=
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

/-- Area of a triangle given side lengths -/
noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- For a triangle ABC with fixed angle α and area S, the side length BC is minimized when AB = AC -/
theorem smallest_side_isosceles (α : ℝ) (S : ℝ) (h_α : 0 < α ∧ α < π) (h_S : S > 0) :
  ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  α = angle_of_triangle a b c →
  S = area_of_triangle a b c →
  ∀ (a' b' c' : ℝ),
  0 < a' ∧ 0 < b' ∧ 0 < c' →
  α = angle_of_triangle a' b' c' →
  S = area_of_triangle a' b' c' →
  b = c →
  a ≤ a' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_side_isosceles_l562_56231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_equals_one_l562_56251

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

-- Define fn recursively
noncomputable def fn : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | n + 1 => λ x => fn n (f x)

-- State the theorem
theorem f_100_equals_one :
  fn 100 (-1/99) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_equals_one_l562_56251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swim_club_membership_l562_56245

theorem swim_club_membership (T : ℕ) : 
  (⌊(T : ℝ) * 0.45⌋ = (T : ℝ) * 0.45) →  -- 45% of T is a whole number
  (⌊(T : ℝ) * 0.30⌋ = (T : ℝ) * 0.30) →  -- 30% of T is a whole number
  (⌊(T : ℝ) * 0.09⌋ = (T : ℝ) * 0.09) →  -- 9% of T is a whole number (20% of 45%)
  ((T : ℝ) * 0.45 - (T : ℝ) * 0.09 + (T : ℝ) * 0.30 - (T : ℝ) * 0.09 + (T : ℝ) * 0.09 + 64 = T) →
  T = 188 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swim_club_membership_l562_56245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_palindromes_in_ap_l562_56274

theorem infinitely_many_palindromes_in_ap : 
  ∃ f : ℕ → ℕ, StrictMono f ∧ 
  ∀ n : ℕ, ∃ i : ℕ, (10^(f n) - 1) / 9 = 18 + 19 * i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_palindromes_in_ap_l562_56274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l562_56218

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 6*x + 13) + Real.sqrt (x^2 - 14*x + 58)

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), m ≤ f x) ∧ (m = Real.sqrt 41) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l562_56218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martas_walking_distance_l562_56224

/-- Marta's walking distance in miles after 15 minutes, given her rate of 1.5 miles in 36 minutes -/
def martas_distance : ℚ :=
  (1.5 * 15) / 36

/-- Rounds a rational number to the nearest tenth -/
def round_to_tenth (x : ℚ) : ℚ :=
  ⌊x * 10 + 1/2⌋ / 10

theorem martas_walking_distance :
  round_to_tenth martas_distance = 6/10 := by
  -- Proof goes here
  sorry

#eval round_to_tenth martas_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martas_walking_distance_l562_56224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l562_56233

/-- A hyperbola with foci and a special point -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Special point on the right branch
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_equation : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) = P
  h_right_branch : P.1 > 0
  h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0
  h_arithmetic_sequence : ∃ (d : ℝ), 
    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) - d ∧
    Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) - 2*d

/-- The eccentricity of a hyperbola with the given properties is 5 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt ((h.F₂.1 - h.F₁.1)^2 + (h.F₂.2 - h.F₁.2)^2) / (2 * h.a) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l562_56233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_james_hours_relation_l562_56220

/-- Represents the payment structure for an employee -/
structure PaymentStructure where
  baseHours : ℕ
  baseRate : ℚ
  overtimeRate : ℚ

/-- Calculates the total pay for an employee given their payment structure and hours worked -/
def totalPay (ps : PaymentStructure) (hoursWorked : ℚ) : ℚ :=
  if hoursWorked ≤ ps.baseHours then
    hoursWorked * ps.baseRate
  else
    ps.baseHours * ps.baseRate + (hoursWorked - ps.baseHours) * ps.overtimeRate

theorem harry_james_hours_relation (x : ℚ) (j : ℚ) :
  let harry := PaymentStructure.mk 30 x (2 * x)
  let james := PaymentStructure.mk 40 x (1.5 * x)
  let h := 5 + 0.75 * j
  totalPay harry h = totalPay james j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_james_hours_relation_l562_56220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_C_value_l562_56289

noncomputable def A : Fin 3 → ℝ := ![Real.sqrt 3 / 3, Real.sqrt 3 / 3, Real.sqrt 3 / 3]

noncomputable def B (a b c : ℝ) : Fin 4 → ℝ := ![0, a, b, c]

def C (x y : Fin 3 → ℝ) : ℝ := (Finset.univ.sum fun i => x i * y i)

def is_subarray (S : Fin 3 → ℝ) (B : Fin 4 → ℝ) : Prop :=
  ∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    S = ![B i, B j, B k]

theorem max_C_value (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (∀ S : Fin 3 → ℝ, is_subarray S (B a b c) → C A S ≤ 1) ∧
  (∃ S : Fin 3 → ℝ, is_subarray S (B a b c) ∧ C A S = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_C_value_l562_56289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_probability_theorem_l562_56211

def average_letters_per_week : ℝ := 6
def average_foreign_letters_per_week : ℝ := 2

def domestic_rate : ℝ := average_letters_per_week - average_foreign_letters_per_week
def foreign_rate : ℝ := average_foreign_letters_per_week

def weeks : ℕ := 2

def domestic_parameter : ℝ := domestic_rate * (weeks : ℝ)
def foreign_parameter : ℝ := foreign_rate * (weeks : ℝ)

def total_letters : ℕ := 13
def foreign_letters : ℕ := 5
def domestic_letters : ℕ := total_letters - foreign_letters

noncomputable def poisson_probability (lambda : ℝ) (k : ℕ) : ℝ :=
  (lambda ^ k * Real.exp (-lambda)) / k.factorial

theorem letter_probability_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧
  abs (poisson_probability domestic_parameter domestic_letters *
       poisson_probability foreign_parameter foreign_letters - 0.0218) < ε := by
  sorry

#eval domestic_parameter
#eval foreign_parameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_probability_theorem_l562_56211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_maximizes_score_l562_56213

/-- Score function for a subject given weeks of study -/
def score (subject : Fin 3) (weeks : Nat) : Nat :=
  match subject, weeks with
  | 0, w => [20, 40, 55, 65, 72, 78, 80, 82, 83, 84, 85].get! (min w 10)
  | 1, w => [30, 45, 53, 58, 62, 65, 68, 70, 72, 74, 75].get! (min w 10)
  | 2, w => [50, 70, 85, 90, 93, 95, 96, 96, 96, 96, 96].get! (min w 10)

/-- Total score for a given allocation of weeks -/
def totalScore (allocation : Fin 3 → Nat) : Nat :=
  (Finset.sum Finset.univ fun i => score i (allocation i))

/-- The optimal allocation of weeks -/
def optimalAllocation : Fin 3 → Nat :=
  ![5, 4, 2]

theorem optimal_allocation_maximizes_score :
  ∀ allocation : Fin 3 → Nat,
    (Finset.sum Finset.univ allocation = 11) →
    totalScore allocation ≤ totalScore optimalAllocation := by
  sorry

#eval totalScore optimalAllocation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_maximizes_score_l562_56213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zenon_minor_axis_distance_l562_56266

/-- Represents the distance of a planet from its star in astronomical units (AU) -/
def Distance := ℝ

/-- Represents an elliptical orbit of a planet around its star -/
structure EllipticalOrbit where
  perigee : Distance
  apogee : Distance

/-- Calculates the distance from the focus to a vertex of the minor axis of an elliptical orbit -/
noncomputable def distanceToMinorAxisVertex (orbit : EllipticalOrbit) : Distance :=
  sorry

/-- Theorem stating that for the given orbit of Zenon, the distance to a minor axis vertex is 9 AU -/
theorem zenon_minor_axis_distance (zenon_orbit : EllipticalOrbit) 
  (h1 : zenon_orbit.perigee = (3 : ℝ))
  (h2 : zenon_orbit.apogee = (15 : ℝ)) :
  distanceToMinorAxisVertex zenon_orbit = (9 : ℝ) := by
  sorry

#check zenon_minor_axis_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zenon_minor_axis_distance_l562_56266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l562_56277

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)

/-- Line l in parametric form -/
def line_l (t : ℝ) : ℝ × ℝ := (-2 + 4*t, 3*t)

/-- Distance from a point to line l -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  abs (3*x - 4*y + 6) / 5

theorem max_distance_curve_to_line :
  ∃ (θ : ℝ), ∀ (φ : ℝ), distance_to_line (curve_C θ) ≥ distance_to_line (curve_C φ) ∧
  distance_to_line (curve_C θ) = 14/5 := by
  sorry

#check max_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l562_56277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_range_l562_56212

-- Define the fixed circle B
noncomputable def circle_B (x y : ℝ) : Prop := x^2 + y^2 + 2 * Real.sqrt 5 * x - 31 = 0

-- Define point A
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the range of x + 2y
def range_sum (x y : ℝ) : Prop := -5 ≤ x + 2*y ∧ x + 2*y ≤ 5

-- Theorem statement
theorem trajectory_and_range :
  ∀ x y : ℝ,
  (∃ r : ℝ, r > 0 ∧
    (∀ x' y' : ℝ, (x' - x)^2 + (y' - y)^2 = r^2 →
      (∃ x'' y'' : ℝ, circle_B x'' y'' ∧
        (x' - x'')^2 + (y' - y'')^2 = 0))) ∧
  ((x - point_A.1)^2 + (y - point_A.2)^2 = 0 ∨
   (∃ r : ℝ, r > 0 ∧ (x - point_A.1)^2 + (y - point_A.2)^2 = r^2)) →
  trajectory_E x y ∧ range_sum x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_range_l562_56212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l562_56272

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_property (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_eq : (Real.sin A - Real.sin B + Real.sin C) / Real.sin C = b / (a + b - c))
  (h_circum : ∀ X, Real.sin X / (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) = 1) :
  A = Real.pi/3 ∧ a + b + c ≤ 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l562_56272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_between_squares_and_cubes_l562_56296

theorem exists_x0_between_squares_and_cubes 
  (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 1 2))
  (h_integral : ∫ x in (1:ℝ)..(2:ℝ), f x = 73/24) :
  ∃ x₀ ∈ Set.Ioo 1 2, x₀^2 < f x₀ ∧ f x₀ < x₀^3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_between_squares_and_cubes_l562_56296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_valid_l562_56226

open Real MeasureTheory

variable (C : ℝ) (x : ℝ)

noncomputable def y (x : ℝ) : ℝ := C * (1 / x^3) * Real.exp (-1 / x)

theorem solution_valid (hx : x > 0) :
  x * ∫ t in Set.Icc 0 x, y C t = (x + 1) * ∫ t in Set.Icc 0 x, t * y C t := by
  sorry

#check solution_valid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_valid_l562_56226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l562_56257

def a : ℕ → ℚ
| 0 => 1
| n + 1 => 2 / (n + 2 : ℚ)

theorem sequence_formula (n : ℕ) :
  (a 1 = 1) ∧
  (∀ k : ℕ, k ≥ 1 → a (k + 1) = (2 * a k) / (a k + 2)) ∧
  (a (n + 1) = 2 / (n + 2 : ℚ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l562_56257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2015_l562_56252

def sequence_a : ℕ → ℤ
  | 0 => 1  -- We define a_0 as 1 to match a_1 in the problem
  | 1 => 4  -- This corresponds to a_2 in the problem
  | 2 => 9  -- This corresponds to a_3 in the problem
  | n+3 => sequence_a (n+2) + sequence_a (n+1) - sequence_a n

theorem sequence_a_2015 : sequence_a 2014 = 8057 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2015_l562_56252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l562_56205

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 2 * (Real.cos x) ^ 2 - 1

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 1 / 2)
  (h2 : b - a = c - b)  -- arithmetic sequence condition
  (h3 : b * c * Real.cos A = 9)  -- AB · AC = 9
  : a = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l562_56205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_square_side_length_l562_56294

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a square -/
structure Square where
  topLeft : Point
  sideLength : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  P : Point
  Q : Point
  R : Point
  S : Point
  M : Point
  N : Point
  largeSquare : Square
  smallSquare : Square

/-- The main theorem to be proved -/
theorem small_square_side_length 
  (setup : ProblemSetup)
  (h1 : setup.largeSquare.sideLength = 1)
  (h2 : setup.M.x = setup.R.x ∧ setup.M.y = setup.Q.y)
  (h3 : setup.N.x = setup.R.x ∧ setup.N.y = setup.S.y)
  (h4 : distance setup.P setup.M = distance setup.P setup.N ∧ 
        distance setup.P setup.M = distance setup.M setup.N)
  (h5 : setup.smallSquare.topLeft = setup.Q)
  (h6 : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
        setup.smallSquare.topLeft.x + setup.smallSquare.sideLength = 
        setup.P.x * (1 - t) + setup.M.x * t)
  : setup.smallSquare.sideLength = (3 - Real.sqrt 3) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_square_side_length_l562_56294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l562_56219

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity 2, prove that its asymptotes are y = ±√3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.sqrt (1 + b^2 / a^2) = 2) →
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ = Real.sqrt 3 * x ∧ y₂ = -Real.sqrt 3 * x ∧ 
    (∀ ε > 0, ∃ x₀ > 0, ∀ x' ≥ x₀, 
      |y₁ - (b/a) * Real.sqrt (x'^2 - a^2)| < ε ∧ 
      |y₂ + (b/a) * Real.sqrt (x'^2 - a^2)| < ε)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l562_56219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_has_nonlosing_strategy_l562_56241

/-- Represents a card in the game -/
structure Card where
  number : Nat
  deriving Repr

/-- Represents the game state -/
structure GameState where
  cards : List Card
  aliceTurn : Bool
  aliceSum : Nat
  bobSum : Nat

/-- A strategy for Alice to choose a card -/
def AliceStrategy := GameState → Bool

/-- Simulates playing the game with a given strategy and initial cards -/
def playGame (strategy : AliceStrategy) (initialCards : List Card) : GameState :=
  sorry -- Implementation of game simulation goes here

theorem alice_has_nonlosing_strategy (n : Nat) (h : n > 1) :
  ∃ (strategy : AliceStrategy),
    ∀ (initialCards : List Card),
      (initialCards.length = 2 * n) →
      (∀ c ∈ initialCards, c.number ≥ 1 ∧ c.number ≤ 2 * n) →
      (∀ i ∈ Finset.range (2 * n), ∃ c ∈ initialCards, c.number = i + 1) →
      let finalState := playGame strategy initialCards
      finalState.aliceSum ≥ finalState.bobSum :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_has_nonlosing_strategy_l562_56241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l562_56208

/-- Given an ellipse and a circle with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b c : ℝ) (C A B : ℝ × ℝ) :
  c > 0 →  -- Assume c is positive for the right focus
  a^2 = b^2 + c^2 →  -- Relation between a, b, and c in an ellipse
  C = (2, 1) →  -- Point C coordinates
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - 2)^2 + (y - 1)^2 ≥ ((2*x + y - 4) / Real.sqrt 5)^2) →  -- Tangent line condition
  (A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4) →  -- A and B are on the circle
  (2*A.1 + A.2 = 4 ∧ 2*B.1 + B.2 = 4) →  -- A and B are on the line 2x + y = 4
  (c, 0) ∈ {x : ℝ × ℝ | 2*x.1 + x.2 = 4} →  -- Right focus is on the line
  (0, b) ∈ {x : ℝ × ℝ | 2*x.1 + x.2 = 4} →  -- Top vertex is on the line
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/20 + y^2/16 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l562_56208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_probability_l562_56236

def primes_30 : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def has_integer_root (p q r : Nat) : Prop :=
  (p - q + r = 0) ∨ (p * r = q - 1)

def valid_triples : Finset (Nat × Nat × Nat) :=
  {(2, 5, 3), (2, 7, 5), (2, 13, 11), (2, 19, 17), (2, 5, 2), (2, 7, 3), (2, 11, 5), (2, 23, 11),
   (3, 5, 2), (5, 7, 2), (11, 13, 2), (17, 19, 2), (2, 3, 5),
   (5, 2, 3), (7, 2, 5), (13, 2, 11), (19, 2, 17)}

theorem integer_root_probability :
  (Finset.card valid_triples : Rat) / (Finset.card primes_30 ^ 3 : Rat) = 3 / 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_probability_l562_56236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_theorem_l562_56286

/-- Represents the walking problem with Erin, Susan, and Daniel -/
structure WalkingProblem where
  total_time : ℝ
  combined_speed : ℝ
  susan_speed : ℝ
  daniel_speed : ℝ
  erin_speed : ℝ

/-- The solution to the walking problem -/
noncomputable def walking_solution (p : WalkingProblem) : ℝ :=
  p.susan_speed * p.total_time / 3

/-- Theorem stating the conditions and solution of the walking problem -/
theorem walking_theorem (p : WalkingProblem) 
  (h1 : p.total_time = 8)
  (h2 : p.combined_speed = 6)
  (h3 : p.daniel_speed = p.susan_speed / 2)
  (h4 : p.erin_speed = p.susan_speed - 3 * 3 / 8)
  (h5 : p.erin_speed + p.susan_speed + p.daniel_speed = p.combined_speed) :
  walking_solution p = 9.6 := by
  sorry

/-- Example calculation (marked as noncomputable) -/
noncomputable example : ℝ := 
  walking_solution { 
    total_time := 8, 
    combined_speed := 6, 
    susan_speed := 18/5, 
    daniel_speed := 9/5, 
    erin_speed := 12/5 
  }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_theorem_l562_56286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_vector_sum_l562_56298

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- In a quadrilateral ABCD, if E is the midpoint of AC and F is the midpoint of BD, 
    then EF = 1/2 * (AB + CD) -/
theorem midpoint_vector_sum (A B C D E F : V) 
  (h1 : E = (1 / 2 : ℝ) • (A + C)) 
  (h2 : F = (1 / 2 : ℝ) • (B + D)) : 
  F - E = (1 / 2 : ℝ) • ((B - A) + (D - C)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_vector_sum_l562_56298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_purchased_l562_56270

def budget : ℚ := 50
def pastry_cost : ℚ := 7
def juice_cost : ℚ := 2

def max_pastries : ℕ := Int.toNat ((budget / pastry_cost).floor)

def remaining_money : ℚ := budget - (max_pastries : ℚ) * pastry_cost

def max_juices : ℕ := Int.toNat ((remaining_money / juice_cost).floor)

theorem total_items_purchased : max_pastries + max_juices = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_purchased_l562_56270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_theorem_l562_56249

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 27 = 0

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem tangent_angle_theorem :
  ∃ (A B : ℝ × ℝ),
    (circle_equation A.1 A.2) ∧
    (circle_equation B.1 B.2) ∧
    (∀ (x y : ℝ), circle_equation x y → (x - origin.1)^2 + (y - origin.2)^2 ≥ (A.1 - origin.1)^2 + (A.2 - origin.2)^2) ∧
    (∀ (x y : ℝ), circle_equation x y → (x - origin.1)^2 + (y - origin.2)^2 ≥ (B.1 - origin.1)^2 + (B.2 - origin.2)^2) ∧
    (A ≠ B) ∧
    (let angle := Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2)));
     angle = π / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_theorem_l562_56249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_complete_residue_function_l562_56238

/-- A function is a complete residue function if for every prime p and natural number x,
    the set {x, f(x), ..., f^(p-1)(x)} forms a complete residue system modulo p. -/
def IsCompleteResidueFunction (f : ℕ → ℕ) : Prop :=
  ∀ (p : ℕ) (x : ℕ), Nat.Prime p →
    (Finset.range p).card = p ∧
    (Finset.range p).card = (Finset.image (fun k => (f^[k] x) % p) (Finset.range p)).card

/-- The successor function is the only complete residue function. -/
theorem unique_complete_residue_function :
  ∃! f : ℕ → ℕ, IsCompleteResidueFunction f ∧ ∀ x, f x = x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_complete_residue_function_l562_56238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l562_56297

/-- The function representing the given graph -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 3) / (x^2 - 4*x + 4)

/-- The vertical asymptote of the graph -/
def vertical_asymptote : ℝ := 2

/-- The horizontal asymptote of the graph -/
def horizontal_asymptote : ℝ := 1

/-- Theorem: The point of intersection of the asymptotes is (2, 1) -/
theorem asymptotes_intersection :
  (vertical_asymptote, horizontal_asymptote) = (2, 1) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l562_56297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_in_set_l562_56281

theorem unique_a_in_set (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_in_set_l562_56281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l562_56292

/-- The equation of the directrix of a parabola y = ax^2 + bx + c, where a ≠ 0 -/
noncomputable def directrix (a b c : ℝ) : ℝ := 
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  k - 1 / (4 * a)

/-- Theorem: The directrix of the parabola y = -3x^2 + 6x - 5 is y = -25/12 -/
theorem parabola_directrix : directrix (-3 : ℝ) 6 (-5) = -25/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l562_56292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_plus_i_l562_56263

open Complex

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I))) :
  Complex.abs (z + I) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_plus_i_l562_56263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_l562_56246

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^2 else 2 * Real.cos x

theorem f_solutions (x : ℝ) :
  f x = 1 ↔ x = -2 ∨ x = π/3 ∨ x = 5*π/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_l562_56246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_balls_count_l562_56254

/-- The number of purple balls in a bin with green and purple balls -/
def k : ℕ := sorry

/-- The number of green balls in the bin -/
def green_balls : ℕ := 7

/-- The amount won when drawing a green ball -/
def green_win : ℤ := 3

/-- The amount lost when drawing a purple ball -/
def purple_loss : ℤ := 1

/-- The expected value of the game -/
def expected_value : ℚ := 1

/-- Theorem stating that k equals 7 given the conditions of the game -/
theorem purple_balls_count : k = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_balls_count_l562_56254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_display_screen_signals_l562_56291

/-- Represents a display screen with 4 holes -/
structure DisplayScreen :=
  (holes : Fin 4 → Bool)

/-- Checks if two holes are adjacent -/
def are_adjacent (a b : Fin 4) : Bool :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

/-- Checks if a pair of holes is valid (not adjacent) -/
def is_valid_pair (a b : Fin 4) : Bool :=
  a ≠ b ∧ ¬(are_adjacent a b)

/-- Counts the number of valid signals -/
def count_valid_signals : Nat :=
  Fintype.card (Fin 4) * (Fintype.card (Fin 4) - 1) / 2 * 4

/-- The main theorem stating that the number of valid signals is 12 -/
theorem display_screen_signals :
  count_valid_signals = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_display_screen_signals_l562_56291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_half_angle_identities_l562_56200

theorem triangle_half_angle_identities (A B C : Real) (h : A + B + C = Real.pi) :
  (Real.cos (A/2))^2 + (Real.cos (B/2))^2 + (Real.cos (C/2))^2 = 2 * (1 + Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) ∧
  (Real.sin (A/2))^2 + (Real.sin (B/2))^2 + (Real.sin (C/2))^2 = 1 - 2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_half_angle_identities_l562_56200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l562_56256

theorem min_value_of_f :
  ∃ (min_val : ℝ) (min_x : ℝ),
    (∀ x : ℝ, x ≥ 1 → (4 * x^2 - 2 * x + 16) / (2 * x - 1) ≥ min_val) ∧
    (4 * min_x^2 - 2 * min_x + 16) / (2 * min_x - 1) = min_val ∧
    min_val = 9 ∧
    min_x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l562_56256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_squares_same_pattern_l562_56217

-- Define the number of 2x2 squares and possible color patterns
def num_squares : Nat := 49
def num_patterns : Nat := 16

-- Define a function that assigns a color pattern to each 2x2 square
def color_assignment : Fin num_squares → Fin num_patterns := sorry

-- Theorem statement
theorem two_squares_same_pattern :
  ∃ (i j : Fin num_squares), i ≠ j ∧ color_assignment i = color_assignment j := by
  sorry

#check two_squares_same_pattern

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_squares_same_pattern_l562_56217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_l562_56282

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.sin x) ^ 2

-- Define the interval bounds
noncomputable def lower_bound (k : ℤ) : ℝ := (5 * Real.pi / 12) + k * Real.pi
noncomputable def upper_bound (k : ℤ) : ℝ := (11 * Real.pi / 12) + k * Real.pi

-- State the theorem
theorem f_monotonically_decreasing (k : ℤ) :
  ∀ x y, lower_bound k ≤ x ∧ x < y ∧ y ≤ upper_bound k → f y < f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_l562_56282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_power_function_l562_56279

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (-2 * m^2 + m + 2) * x^(-2 * m + 1)

theorem even_power_function (m : ℝ) :
  (∀ x, f m x = f m (-x)) → f m = λ x ↦ x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_power_function_l562_56279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gemini_functions_l562_56293

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem gemini_functions :
  (∃ (c : ℝ), c = 1 / Real.exp 1 ∧
    (∀ x > 0, f x ≤ c) ∧
    (∃ x > 0, f x = c)) ∧
  (∃ (c : ℝ), c = 1 / Real.exp 1 ∧
    (∀ x : ℝ, g x ≤ c) ∧
    (∃ x : ℝ, g x = c)) ∧
  (∀ x > 0, Real.exp x - Real.log x > 2 * x) ∧
  (∀ m n : ℝ, 0 < m → m < 1 → n = Real.log m →
    m * n ≥ -1 / Real.exp 1 ∧
    (∃ m₀ : ℝ, 0 < m₀ ∧ m₀ < 1 ∧ m₀ * Real.log m₀ = -1 / Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gemini_functions_l562_56293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_to_cos_product_l562_56290

theorem sin_squared_sum_to_cos_product :
  (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (2*x))^2 + (Real.sin (3*x))^2 + (Real.sin (4*x))^2 = 2) ↔
  (∀ x : ℝ, Real.cos x * Real.cos (2*x) * Real.cos (5*x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_to_cos_product_l562_56290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_pay_is_2_60_l562_56201

/-- Represents the characteristics of a slow clock and payment rates -/
structure SlowClockPayment where
  slow_clock_cycle : ℚ  -- Time for minute and hour hands to meet in minutes
  regular_rate : ℚ      -- Regular pay rate per hour
  overtime_rate : ℚ     -- Overtime pay rate per hour
  regular_hours : ℚ     -- Regular work schedule in hours

/-- Calculates the extra overtime pay for a given slow clock and payment structure -/
def calculate_overtime_pay (s : SlowClockPayment) : ℚ :=
  let actual_minutes_in_day : ℚ := 12 * 60
  let slow_clock_minutes_in_day : ℚ := s.slow_clock_cycle * 11
  let slow_clock_minutes_in_regular_hours : ℚ := (s.regular_hours * slow_clock_minutes_in_day) / 12
  let overtime_minutes : ℚ := slow_clock_minutes_in_regular_hours - (s.regular_hours * 60)
  let overtime_hours : ℚ := overtime_minutes / 60
  overtime_hours * s.overtime_rate

/-- Theorem stating that the extra overtime pay for the given conditions is $2.60 -/
theorem extra_pay_is_2_60 (s : SlowClockPayment) 
  (h1 : s.slow_clock_cycle = 69)
  (h2 : s.regular_rate = 4)
  (h3 : s.overtime_rate = 6)
  (h4 : s.regular_hours = 8) :
  calculate_overtime_pay s = 13/5 := by
  sorry

#eval calculate_overtime_pay { slow_clock_cycle := 69, regular_rate := 4, overtime_rate := 6, regular_hours := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_pay_is_2_60_l562_56201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_furniture_time_l562_56222

/-- The time Rachel spent putting together her furniture -/
theorem rachel_furniture_time 
  (chairs : ℕ) (tables : ℕ) (time_per_piece : ℕ) (total_time : ℕ) :
  chairs = 7 →
  tables = 3 →
  time_per_piece = 4 →
  total_time = (chairs + tables) * time_per_piece →
  total_time = 40 :=
by
  intros h_chairs h_tables h_time h_total
  rw [h_chairs, h_tables, h_time] at h_total
  norm_num at h_total
  exact h_total

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_furniture_time_l562_56222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shark_ratio_problem_l562_56285

/-- The ratio of sharks at Dana Point beach to Newport Beach --/
def shark_ratio (dana_point : ℕ) (newport : ℕ) : ℚ :=
  dana_point / newport

/-- The problem statement --/
theorem shark_ratio_problem (dana_point newport : ℕ) : 
  newport = 22 →
  dana_point + newport = 110 →
  shark_ratio dana_point newport = 4 := by
  intros h1 h2
  unfold shark_ratio
  -- The proof steps would go here
  sorry

#check shark_ratio_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shark_ratio_problem_l562_56285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_theorem_l562_56223

/-- Represents the speed of a man rowing a boat in different conditions. -/
structure RowingSpeed where
  withStream : ℚ
  againstStream : ℚ

/-- Calculates the man's rate in still water given his speeds with and against the stream. -/
def manRate (speed : RowingSpeed) : ℚ :=
  (speed.withStream + speed.againstStream) / 2

/-- Theorem stating that if a man rows at 16 km/h with the stream and 4 km/h against the stream,
    his rate in still water is 10 km/h. -/
theorem man_rate_theorem (speed : RowingSpeed)
    (h1 : speed.withStream = 16)
    (h2 : speed.againstStream = 4) :
    manRate speed = 10 := by
  sorry

#eval manRate { withStream := 16, againstStream := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_theorem_l562_56223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_sum_l562_56242

-- Define the square ABCD
def square_ABCD : Set (ℝ × ℝ) := {p | ∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 ∧ p = (x, y)}

-- Define points A, B, C, D
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (0, 2)

-- Define points E and F
noncomputable def E : ℝ × ℝ := sorry
noncomputable def F : ℝ × ℝ := sorry

-- Define the smaller square
noncomputable def smaller_square : Set (ℝ × ℝ) := sorry

-- Define the side length of the smaller square
noncomputable def s : ℝ := sorry

-- State the theorem
theorem square_side_length_sum (h1 : E.1 = 2 ∧ 0 ≤ E.2 ∧ E.2 ≤ 2)
                               (h2 : F.1 = 0 ∧ 0 ≤ F.2 ∧ F.2 ≤ 2)
                               (h3 : (A.1 - E.1)^2 + (A.2 - E.2)^2 = (A.1 - F.1)^2 + (A.2 - F.2)^2)
                               (h4 : ∃ p ∈ smaller_square, p.1 = 2 ∧ p.2 = 0)
                               (h5 : ∃ p ∈ smaller_square, 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ p.2 = 0)
                               (h6 : ∃ (a b c : ℕ), s = (a + Real.sqrt b) / c ∧ 
                                     (∀ (p : ℕ), Prime p → ¬(p^2 ∣ b))) :
  ∃ (a b c : ℕ), s = (a + Real.sqrt b) / c ∧ a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_sum_l562_56242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l562_56283

noncomputable def polar_point := ℝ × ℝ

noncomputable def distance (p₁ p₂ : polar_point) : ℝ :=
  Real.sqrt ((p₁.1 * Real.cos p₁.2 - p₂.1 * Real.cos p₂.2)^2 + 
             (p₁.1 * Real.sin p₁.2 - p₂.1 * Real.sin p₂.2)^2)

theorem distance_between_polar_points 
  (A B : polar_point) 
  (h₁ : A.1 = 4) 
  (h₂ : B.1 = 6) 
  (h₃ : A.2 - B.2 = π / 3) : 
  distance A B = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l562_56283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_fourth_vertex_l562_56221

/-- A parallelogram with three known vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (1, 0)
  v2 : ℝ × ℝ := (5, 8)
  v3 : ℝ × ℝ := (7, -4)

/-- The possible fourth vertices of the parallelogram -/
def possible_fourth_vertices : List (ℝ × ℝ) :=
  [(11, 4), (-1, 12), (3, -12)]

/-- Predicate to check if four points form a parallelogram -/
def is_parallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1 ∧ b.2 - a.2 = d.2 - c.2) ∨
  (c.1 - a.1 = d.1 - b.1 ∧ c.2 - a.2 = d.2 - b.2)

/-- Theorem stating that the fourth vertex of the parallelogram is one of the possible vertices -/
theorem parallelogram_fourth_vertex (p : Parallelogram) :
  ∃ v4 : ℝ × ℝ, v4 ∈ possible_fourth_vertices ∧ 
  (is_parallelogram p.v1 p.v2 p.v3 v4 ∨ 
   is_parallelogram p.v1 p.v2 v4 p.v3 ∨ 
   is_parallelogram p.v1 v4 p.v2 p.v3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_fourth_vertex_l562_56221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l562_56262

open Complex Real

-- Define the complex number -1 + i
def z_eq : ℂ := -1 + I

-- Define the equation z^6 = -1 + i
def equation (z : ℂ) : Prop := z^6 = z_eq

-- Define the set of angles θ that satisfy the equation
def valid_angles : Set ℝ :=
  {θ | 0 ≤ θ ∧ θ < 2*Real.pi ∧ equation (exp (I * θ))}

-- Theorem statement
theorem sum_of_angles :
  ∃ (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ),
    θ₁ ∈ valid_angles ∧
    θ₂ ∈ valid_angles ∧
    θ₃ ∈ valid_angles ∧
    θ₄ ∈ valid_angles ∧
    θ₅ ∈ valid_angles ∧
    θ₆ ∈ valid_angles ∧
    θ₁ ≠ θ₂ ∧ θ₁ ≠ θ₃ ∧ θ₁ ≠ θ₄ ∧ θ₁ ≠ θ₅ ∧ θ₁ ≠ θ₆ ∧
    θ₂ ≠ θ₃ ∧ θ₂ ≠ θ₄ ∧ θ₂ ≠ θ₅ ∧ θ₂ ≠ θ₆ ∧
    θ₃ ≠ θ₄ ∧ θ₃ ≠ θ₅ ∧ θ₃ ≠ θ₆ ∧
    θ₄ ≠ θ₅ ∧ θ₄ ≠ θ₆ ∧
    θ₅ ≠ θ₆ ∧
    θ₁ + θ₂ + θ₃ + θ₄ + θ₅ + θ₆ = (1125 * Real.pi) / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l562_56262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_roots_l562_56206

theorem sum_of_reciprocals_of_roots : 
  ∀ r₁ r₂ : ℝ, 
  r₁^2 - 17*r₁ + 8 = 0 →
  r₂^2 - 17*r₂ + 8 = 0 →
  r₁ ≠ r₂ →
  1/r₁ + 1/r₂ = 17/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_roots_l562_56206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_investment_gain_percentage_l562_56268

/-- Calculates the total gain percentage for three flats given their costs and individual gain/loss percentages. -/
noncomputable def total_gain_percentage (cost1 cost2 cost3 : ℝ) (gain1 loss2 gain3 : ℝ) : ℝ :=
  let gain_amount1 := cost1 * gain1
  let loss_amount2 := cost2 * loss2
  let gain_amount3 := cost3 * gain3
  let total_gain := gain_amount1 + gain_amount3
  let total_loss := loss_amount2
  let net_gain := total_gain - total_loss
  let total_cost := cost1 + cost2 + cost3
  (net_gain / total_cost) * 100

/-- Theorem stating that the total gain percentage for the given flat costs and gain/loss percentages is approximately 3.608%. -/
theorem flat_investment_gain_percentage :
  let cost1 := (675958 : ℝ)
  let cost2 := (995320 : ℝ)
  let cost3 := (837492 : ℝ)
  let gain1 := (0.11 : ℝ)
  let loss2 := (0.11 : ℝ)
  let gain3 := (0.15 : ℝ)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |total_gain_percentage cost1 cost2 cost3 gain1 loss2 gain3 - 3.608| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_investment_gain_percentage_l562_56268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_card_distribution_l562_56202

/-- Proves that Rick gave cards to 8 friends given the conditions of the problem -/
theorem rick_card_distribution (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (cards_per_friend : ℕ) (num_sisters : ℕ) (cards_per_sister : ℕ) : 
  total_cards = 130 ∧ kept_cards = 15 ∧ miguel_cards = 13 ∧ 
  cards_per_friend = 12 ∧ num_sisters = 2 ∧ cards_per_sister = 3 →
  (total_cards - kept_cards - miguel_cards - num_sisters * cards_per_sister) / cards_per_friend = 8 := by
  intro h
  sorry

#eval (130 - 15 - 13 - 2 * 3) / 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_card_distribution_l562_56202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_segments_l562_56239

/-- Given four points E, F, G, H on a line in that order, with EF = 3, FG = 6, and EH = 16,
    prove that the ratio of EG to FH is 9/13. -/
theorem ratio_of_segments (E F G H : ℝ) : 
  F ∈ Set.Icc E H → G ∈ Set.Icc F H → 
  F - E = 3 → G - F = 6 → H - E = 16 → 
  (G - E) / (H - F) = 9 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_segments_l562_56239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2012_equals_2_l562_56271

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add a case for 0
  | 1 => 1/2
  | n + 1 => 1 / (1 - sequence_a n)

theorem a_2012_equals_2 : sequence_a 2012 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2012_equals_2_l562_56271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_l562_56237

/-- Represents the vote counts for three candidates in an election -/
structure ElectionResults where
  candidate1 : Nat
  candidate2 : Nat
  candidate3 : Nat

/-- Calculates the total number of votes in the election -/
def totalVotes (results : ElectionResults) : Nat :=
  results.candidate1 + results.candidate2 + results.candidate3

/-- Determines the number of votes for the winning candidate -/
def winningVotes (results : ElectionResults) : Nat :=
  max results.candidate1 (max results.candidate2 results.candidate3)

/-- Calculates the percentage of votes for the winning candidate -/
noncomputable def winningPercentage (results : ElectionResults) : Real :=
  (winningVotes results : Real) / (totalVotes results : Real) * 100

/-- Theorem stating that in the given election, the winning candidate received approximately 67.23% of the votes -/
theorem winning_percentage_approx (results : ElectionResults)
  (h1 : results.candidate1 = 1036)
  (h2 : results.candidate2 = 4636)
  (h3 : results.candidate3 = 11628) :
  abs (winningPercentage results - 67.23) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_l562_56237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_relations_l562_56288

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the projection operation
def project (l : Line) (p : Plane) : Line := sorry

-- Define the perpendicular and parallel relations
def perpendicular (l1 l2 : Line) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem projection_relations :
  ∀ (a b : Line) (α : Plane),
    let a1 := project a α
    let b1 := project b α
    (¬ (perpendicular a b → perpendicular a1 b1)) ∧
    (¬ (perpendicular a1 b1 → perpendicular a b)) ∧
    (¬ (parallel a b → parallel a1 b1)) ∧
    (¬ (parallel a1 b1 → parallel a b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_relations_l562_56288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_fixed_point_l562_56244

-- Define the circles and points
def circle_radius : ℝ := 1  -- We can use any positive real number for the radius
def circle_center_O : ℝ × ℝ := (0, 0)
def circle_center_C : ℝ × ℝ := (-2 * circle_radius, 0)

-- α is a constant angle, β is a variable angle
variable (α β : ℝ)

-- Define points A and B
noncomputable def point_A (β : ℝ) : ℝ × ℝ := 
  (circle_radius * Real.cos β, circle_radius * Real.sin β)

noncomputable def point_B (α β : ℝ) : ℝ × ℝ := 
  (-2 * circle_radius + circle_radius * Real.cos (α + β), 
   circle_radius * Real.sin (α + β))

-- Define the fixed point D
noncomputable def point_D (α : ℝ) : ℝ × ℝ := 
  (-circle_radius, -circle_radius * Real.tan (α / 2)⁻¹)

-- Define the perpendicular bisector of AB
noncomputable def perp_bisector (α β : ℝ) : ℝ → ℝ := 
  λ x ↦ (x - (point_A β).1 + (point_B α β).1) / 2 * 
    (Real.cos (α + β) - Real.cos β - 2) / (Real.sin (α + β) - Real.sin β) + 
    ((point_A β).2 + (point_B α β).2) / 2

-- Theorem statement
theorem perpendicular_bisector_fixed_point :
  ∀ α β, perp_bisector α β (point_D α).1 = (point_D α).2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_fixed_point_l562_56244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_children_ages_l562_56273

theorem sum_of_children_ages (n : ℕ) (T : ℕ) : 
  n > 0 →  -- number of adults is positive
  (∀ (age : ℕ), age > 1) →  -- all ages are greater than 1
  156 = n * (156 / n) →  -- sum of adult ages is 156
  T = 2 * n * ((156 + T) / (3 * n) / 5) →  -- mean age of children is 20% of whole group mean
  T = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_children_ages_l562_56273
