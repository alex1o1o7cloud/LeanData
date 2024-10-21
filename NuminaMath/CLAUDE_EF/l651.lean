import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l651_65192

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l651_65192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_grey_triangle_l651_65153

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a triangle with three sides -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Checks if a triangle is right-angled -/
def Triangle.is_right_triangle (t : Triangle) : Prop :=
  t.side1^2 + t.side2^2 = t.side3^2 ∨ t.side1^2 + t.side3^2 = t.side2^2 ∨ t.side2^2 + t.side3^2 = t.side1^2

/-- Checks if a triangle is isosceles -/
def Triangle.is_isosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side1 = t.side3 ∨ t.side2 = t.side3

/-- Calculates the smallest angle of a triangle in radians -/
noncomputable def Triangle.smallest_angle (t : Triangle) : ℝ :=
  Real.arccos ((t.side2^2 + t.side3^2 - t.side1^2) / (2 * t.side2 * t.side3))

theorem smallest_angle_grey_triangle : 
  ∀ (square : Rectangle) (white_triangle : Triangle) (grey_triangle : Triangle),
    -- The square has side length 2
    square.length = 2 ∧ square.width = 2 →
    -- The white triangle is isosceles and right-angled with leg length 1
    white_triangle.is_right_triangle ∧ 
    white_triangle.is_isosceles ∧
    white_triangle.side1 = 1 →
    -- The grey triangle is formed between two adjacent white triangles
    grey_triangle.is_right_triangle ∧
    grey_triangle.side3 = 2 ∧
    grey_triangle.side1 = 1 →
    -- The smallest angle in the grey triangle is 15 degrees
    grey_triangle.smallest_angle = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_grey_triangle_l651_65153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_alpha_when_OP_minimized_l651_65102

noncomputable def P (t : ℝ) : ℝ × ℝ := (t/2 + 2/t, 1)

theorem cosine_alpha_when_OP_minimized :
  ∀ t : ℝ, t < 0 →
  let p := P t
  let magnitude := Real.sqrt ((p.1)^2 + (p.2)^2)
  (∀ s : ℝ, s < 0 → Real.sqrt ((P s).1^2 + (P s).2^2) ≥ magnitude) →
  magnitude = Real.sqrt 5 ∧ p.1 / magnitude = -2 / Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_alpha_when_OP_minimized_l651_65102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_ratio_l651_65134

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : 0 < side_length

/-- The overlap region between two triangles in a regular hexagon -/
def overlap (h : RegularHexagon) : Set (Fin 2 → ℝ) :=
  sorry

/-- The area of a regular hexagon -/
noncomputable def area_hexagon (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * h.side_length ^ 2

/-- The area of the overlap region -/
noncomputable def area_overlap (h : RegularHexagon) : ℝ :=
  sorry

/-- Theorem: The ratio of the overlap area to the hexagon area is 1/3 -/
theorem overlap_area_ratio (h : RegularHexagon) :
  area_overlap h / area_hexagon h = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_ratio_l651_65134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l651_65144

/-- Given a triangle ABC with the following properties:
  AC = 6, cos B = 4/5, C = π/4
  Prove the length of AB and the value of cos(A - π/6) -/
theorem triangle_abc_properties (A B C : ℝ) (AC AB : ℝ) :
  AC = 6 →
  Real.cos B = 4/5 →
  C = π/4 →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  A + B + C = π →
  AB = 5 * Real.sqrt 2 ∧
  Real.cos (A - π/6) = (7 * Real.sqrt 2 - 3 * Real.sqrt 6) / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l651_65144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_right_triangle_l651_65150

/-- Given a hyperbola and an ellipse with eccentricities whose product is 1,
    prove that the triangle formed by their parameters is a right triangle. -/
theorem hyperbola_ellipse_right_triangle
  (a b m : ℝ)
  (hm : m > b)
  (hb : b > 0)
  (hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (ellipse : ∀ (x y : ℝ), x^2 / m^2 + y^2 / b^2 = 1)
  (eccentricity_product : (Real.sqrt (a^2 + b^2) / a) * (Real.sqrt (m^2 - b^2) / m) = 1) :
  a^2 + b^2 = m^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_right_triangle_l651_65150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_sqrt_36_l651_65136

theorem square_root_of_sqrt_36 : ∀ x : ℝ, x^2 = 36 → x = 6 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_sqrt_36_l651_65136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_area_properties_l651_65149

/-- A curve C in the plane -/
structure Curve where
  -- The predicate that defines the curve
  contains : ℝ × ℝ → Prop

/-- The distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The distance from a point to a vertical line -/
def distanceToVerticalLine (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  |p.1 - x|

/-- The curve C defined by the problem conditions -/
def curveC : Curve where
  contains := fun p => distance p (2, 0) + 2 = distanceToVerticalLine p (-4)

/-- The theorem stating the properties of the curve and the minimum area -/
theorem curve_and_area_properties (a : ℝ) (h : a > 0) :
  -- The equation of curve C is y^2 = 8x
  (∀ p, curveC.contains p ↔ p.2^2 = 8 * p.1) ∧
  -- The minimum area of triangle AOB is 2a√(2a)
  (∃ minArea : ℝ, minArea = 2 * a * Real.sqrt (2 * a) ∧
    ∀ A B : ℝ × ℝ, curveC.contains A → curveC.contains B →
      (∃ m : ℝ, A.1 = m * A.2 + a ∧ B.1 = m * B.2 + a) →
      minArea ≤ (1/2) * a * |A.2 - B.2|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_area_properties_l651_65149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_transform_correct_l651_65190

def scale_matrix (sx sy : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  !![sx, 0; 0, sy]

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

noncomputable def combined_transform (sx sy θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (rotation_matrix θ) * (scale_matrix sx sy)

theorem combined_transform_correct : 
  combined_transform (-3) 2 (π/4) = !![(-3/Real.sqrt 2), (-2/Real.sqrt 2); (3/Real.sqrt 2), (2/Real.sqrt 2)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_transform_correct_l651_65190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_in_dodecagon_l651_65108

noncomputable section

-- Definitions for the theorem
def ConvexHull (𝕜 : Type*) [NormedAddCommGroup 𝕜] [NormedSpace ℝ 𝕜] (S : Set 𝕜) : Set 𝕜 := sorry

def ConsecutiveVertices (P : Set ℂ) (A B C : ℂ) : Prop := sorry

def AreaTriangle (A B C : ℂ) : ℝ := sorry

theorem min_area_triangle_in_dodecagon : ∃ (z : ℂ), 
  (z - 5) ^ 12 = 144 →
  let vertices := {w : ℂ | (w - 5) ^ 12 = 144}
  let dodecagon := ConvexHull ℂ vertices
  ∃ (D E F : ℂ), D ∈ vertices ∧ E ∈ vertices ∧ F ∈ vertices ∧
    ConsecutiveVertices dodecagon D E F ∧
    AreaTriangle D E F = 
      (12 * Real.sin (Real.pi / 12) ^ 2 * Real.sin (Real.pi / 6)) / 2 ∧
    ∀ (A B C : ℂ), A ∈ vertices → B ∈ vertices → C ∈ vertices →
      ConsecutiveVertices dodecagon A B C →
      AreaTriangle A B C ≥ 
        (12 * Real.sin (Real.pi / 12) ^ 2 * Real.sin (Real.pi / 6)) / 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_in_dodecagon_l651_65108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l651_65188

-- Define points A, B, C in R^2
def A : Fin 2 → ℝ := ![1, 0]
def B : Fin 2 → ℝ := ![3, 0]
def C : Fin 2 → ℝ := ![-1, 4]

-- Define the condition for point P
def P_condition (P : Fin 2 → ℝ) : Prop :=
  (P 0 - A 0)^2 + (P 1 - A 1)^2 + (P 0 - B 0)^2 + (P 1 - B 1)^2 = 10

-- Define the locus of P
def P_locus (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

-- Define the area of triangle PAB
noncomputable def area_PAB (P : Fin 2 → ℝ) : ℝ :=
  abs ((P 0 - A 0) * (P 1 - B 1) - (P 1 - A 1) * (P 0 - B 0)) / 2

-- Define the distance between C and P
noncomputable def dist_CP (P : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((P 0 - C 0)^2 + (P 1 - C 1)^2)

-- Main theorem
theorem main_theorem :
  (∀ P, P_condition P → P_locus (P 0) (P 1)) ∧
  (∃ P, P_condition P ∧ area_PAB P = 2 ∧ ∀ Q, P_condition Q → area_PAB Q ≤ 2) ∧
  (∃ P, P_condition P ∧ dist_CP P = 3 ∧ ∀ Q, P_condition Q → dist_CP Q ≥ 3) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 3 → 3/m + 1/n ≥ (4 + 2 * Real.sqrt 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l651_65188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l651_65189

theorem absolute_value_nested_expression : 
  abs (abs (-(abs (-2 + 3))) - 2 + 2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l651_65189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_proof_l651_65196

noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem symmetric_function_proof (f : ℝ → ℝ) :
  (∀ x y, f y = x ↔ g x = y) →
  (∀ x, f x = (1/2) * Real.exp x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_proof_l651_65196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l651_65169

noncomputable def binomial_expansion (x : ℝ) := (x^(1/2) - 2/x^2)^8

noncomputable def max_binomial_coeff_term (x : ℝ) := 1120/x^6

noncomputable def max_coeff_term (x : ℝ) := 1792/x^11

noncomputable def min_coeff_term (x : ℝ) := -1792/x^(17/2)

theorem binomial_expansion_properties (x : ℝ) (hx : x > 0) :
  (∃ (term : ℝ → ℝ), term = max_binomial_coeff_term ∧ 
    ∀ (other_term : ℝ → ℝ), other_term ≠ term → 
      ∃ (k : ℕ), binomial_expansion x = term x + other_term x + k) ∧
  (max_coeff_term x + min_coeff_term x = 
    1792/x^11 - 1792/x^(17/2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l651_65169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l651_65167

noncomputable def f (φ : ℝ) (x : ℝ) := Real.cos (x + φ)

noncomputable def g (φ : ℝ) (x : ℝ) := f φ x + (deriv (f φ)) x

def is_even (h : ℝ → ℝ) := ∀ x, h x = h (-x)

theorem cosine_function_properties 
  (φ : ℝ) 
  (h1 : -Real.pi < φ) 
  (h2 : φ < 0) 
  (h3 : is_even (g φ)) :
  φ = -Real.pi / 4 ∧ 
  ∃ (M : ℝ), M = (Real.sqrt 2 + 1) / 2 ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), (f φ x) * (g φ x) ≤ M) ∧
  ∃ x ∈ Set.Icc 0 (Real.pi / 4), (f φ x) * (g φ x) = M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l651_65167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removal_percentage_l651_65128

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℝ

/-- Calculates the volume of a rectangular box -/
noncomputable def boxVolume (b : BoxDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a cube -/
noncomputable def cubeVolume (c : CubeDimensions) : ℝ :=
  c.side ^ 3

/-- Calculates the percentage of volume removed -/
noncomputable def percentageVolumeRemoved (box : BoxDimensions) (cube : CubeDimensions) (numCubesRemoved : ℕ) : ℝ :=
  (numCubesRemoved * cubeVolume cube / boxVolume box) * 100

theorem volume_removal_percentage :
  let box : BoxDimensions := ⟨24, 16, 12⟩
  let cube : CubeDimensions := ⟨2⟩
  let numCubesRemoved : ℕ := 8
  ∃ (ε : ℝ), ε > 0 ∧ abs (percentageVolumeRemoved box cube numCubesRemoved - 1.3889) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removal_percentage_l651_65128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l651_65160

/-- Calculates the speed of a train in km/hr given its length, the bridge length, and the time to cross the bridge. -/
noncomputable def train_speed (train_length bridge_length : ℝ) (time_to_cross : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time_to_cross
  speed_ms * 3.6

/-- Theorem stating that a train of length 250 meters crossing a bridge of length 350 meters in 30 seconds has a speed of 72 km/hr. -/
theorem train_speed_proof :
  train_speed 250 350 30 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l651_65160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l651_65183

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c^2 = 1 ∧ 
  ((F₁.1 = c ∧ F₁.2 = 0) ∨ (F₁.1 = -c ∧ F₁.2 = 0)) ∧
  ((F₂.1 = c ∧ F₂.2 = 0) ∨ (F₂.1 = -c ∧ F₂.2 = 0)) ∧
  F₁ ≠ F₂

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

-- Define the perimeter of a triangle
noncomputable def triangle_perimeter (A B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) +
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (F₁ F₂ P Q : ℝ × ℝ) :
  foci F₁ F₂ →
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  collinear P Q F₁ →
  triangle_perimeter P Q F₂ = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l651_65183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_calculation_l651_65103

-- Define constants
noncomputable def curved_surface_area : ℝ := 989.6016858807849
def slant_height : ℝ := 15

-- Define the theorem
theorem cone_radius_calculation (ε : ℝ) (h_ε : ε > 0) :
  ∃ (r : ℝ), abs (r - curved_surface_area / (Real.pi * slant_height)) < ε ∧ abs (r - 21) < ε :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_calculation_l651_65103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l651_65119

-- Define the lines L₁ and L₂
def L₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x + (1 - a) * y = 3
def L₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (a - 1) * x + (2 * a + 3) * y = 2

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    f x₁ y₁ ∧ f x₂ y₂ ∧ g x₃ y₃ ∧ g x₄ y₄ ∧ 
    x₁ ≠ x₂ ∧ x₃ ≠ x₄ → 
    (y₂ - y₁) * (y₄ - y₃) = -(x₂ - x₁) * (x₄ - x₃)

-- The main theorem
theorem perpendicular_lines (a : ℝ) : 
  perpendicular (L₁ a) (L₂ a) ↔ a = 1 ∨ a = -3 := by
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l651_65119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_account_percentage_l651_65159

def income : ℝ := 1200000
def children_share : ℝ := 0.2
def num_children : ℕ := 3
def orphan_donation_rate : ℝ := 0.05
def final_amount : ℝ := 60000

theorem wife_account_percentage : 
  let remaining_after_children := income * (1 - children_share * num_children)
  let remaining_after_donation := remaining_after_children * (1 - orphan_donation_rate)
  let wife_deposit := remaining_after_donation - final_amount
  (wife_deposit / income) * 100 = 33 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_account_percentage_l651_65159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l651_65127

theorem angle_relations (α β : ℝ) (h_acute_α : 0 < α ∧ α < Real.pi / 2) (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_cos_α : Real.cos α = 2 * Real.sqrt 5 / 5) (h_cos_β : Real.cos β = 3 * Real.sqrt 10 / 10) :
  Real.tan (α - β) = 1 / 7 ∧ α + β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l651_65127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_premium_ticket_price_approx_total_cost_matches_l651_65111

/-- Represents the price of tickets and associated calculations for a group circus outing. -/
structure CircusTickets where
  adult_price : ℝ
  num_adults : ℕ := 4
  num_kids : ℕ := 6
  family_discount : ℝ := 0.1
  tax_rate : ℝ := 0.05
  premium_charge : ℝ := 2
  total_cost : ℝ := 100

/-- Calculates the price of a kid's premium ticket based on the given conditions. -/
noncomputable def kids_premium_ticket_price (tickets : CircusTickets) : ℝ :=
  let kid_price := tickets.adult_price / 2
  let adult_total := tickets.num_adults * tickets.adult_price * (1 - tickets.family_discount)
  let kid_total := tickets.num_kids * kid_price
  let subtotal := adult_total + kid_total
  let with_tax := subtotal * (1 + tickets.tax_rate)
  let total_with_premium := with_tax + (tickets.num_adults + tickets.num_kids) * tickets.premium_charge
  kid_price + tickets.premium_charge

/-- Theorem stating that the kid's premium ticket price is approximately $7.77. -/
theorem kids_premium_ticket_price_approx (tickets : CircusTickets) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |kids_premium_ticket_price tickets - 7.77| < ε := by
  sorry

/-- Theorem stating that the calculated total cost matches the given total cost. -/
theorem total_cost_matches (tickets : CircusTickets) :
  let kid_price := tickets.adult_price / 2
  let adult_total := tickets.num_adults * tickets.adult_price * (1 - tickets.family_discount)
  let kid_total := tickets.num_kids * kid_price
  let subtotal := adult_total + kid_total
  let with_tax := subtotal * (1 + tickets.tax_rate)
  let total_with_premium := with_tax + (tickets.num_adults + tickets.num_kids) * tickets.premium_charge
  total_with_premium = tickets.total_cost := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_premium_ticket_price_approx_total_cost_matches_l651_65111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_dimes_calculation_l651_65100

/-- The number of dimes Sam had initially -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Sam's dad gave him -/
def dimes_from_dad : ℕ := 7

/-- The total number of dimes Sam has now -/
def total_dimes : ℕ := 16

/-- Theorem stating that the initial number of dimes equals the total minus those from dad -/
theorem initial_dimes_calculation : initial_dimes = total_dimes - dimes_from_dad := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_dimes_calculation_l651_65100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l651_65166

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x / Real.log 2 else -(Real.log x / Real.log 2)

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  (∃ (min : ℝ), ∀ (x y : ℝ), 0 < x ∧ x < y ∧ f x = f y → 1/x + 4/y ≥ min) ∧
  (∃ (min : ℝ), ∀ (x y : ℝ), 0 < x ∧ x < y ∧ f x = f y → 1/x + 4/y ≥ min ∧ (1/a + 4/b = min)) ∧
  f (a + b) = 1 - 2 * (Real.log 2 / Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l651_65166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l651_65151

-- Define the equation
def equation (x : ℝ) : Prop :=
  (((63 - 3*x) ^ (1/4 : ℝ)) + ((27 + 3*x) ^ (1/4 : ℝ))) = 5

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l651_65151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_surface_area_and_angle_l651_65158

noncomputable section

open Real

/-- The volume of a cone given its surface area and central angle of the lateral surface --/
theorem cone_volume_from_surface_area_and_angle 
  (S : ℝ) (θ : ℝ) (h_S : S = 15 * π) (h_θ : θ = π / 3) :
  ∃ V : ℝ, V = (25 * Real.sqrt 3) / 7 * π := by
  -- Let r be the radius of the base and l be the slant height
  let r : ℝ := Real.sqrt (15 / 7)
  let l : ℝ := 6 * r

  have h1 : 2 * π * r = θ * l := by sorry
  have h2 : S = π * r^2 + π * r * l := by sorry
  have h3 : l^2 = r^2 + (Real.sqrt 35 * r)^2 := by sorry

  -- Calculate the height h
  let h : ℝ := Real.sqrt 35 * r

  -- Calculate the volume V
  let V : ℝ := (1/3) * π * r^2 * h

  -- Show that V equals the expected value
  have h_V : V = (25 * Real.sqrt 3) / 7 * π := by sorry

  exact ⟨V, h_V⟩

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_surface_area_and_angle_l651_65158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_person_with_many_acquaintances_l651_65156

/-- Represents a person in the population -/
structure Person where
  id : Nat
  believes_in_santa : Bool
  acquaintances : Finset Nat

/-- The population of the country -/
def Population : Finset Nat := Finset.range 1000000

/-- Axiom: The total population is 1,000,000 -/
axiom total_population : Population.card = 1000000

/-- Axiom: Everyone knows at least one other person -/
axiom everyone_knows_someone (p : Nat) (h : p ∈ Population) : 
  ∃ q : Nat, q ∈ Population ∧ q ≠ p ∧ q ∈ (Person.mk p false ∅).acquaintances

/-- Axiom: 90% of the population believes in Santa Claus -/
axiom believers_percentage : 
  (Population.filter (λ p => (Person.mk p true ∅).believes_in_santa)).card = 900000

/-- Axiom: For each person, 10% of their acquaintances believe in Santa Claus -/
axiom acquaintance_believers_percentage (p : Nat) (h : p ∈ Population) :
  let person := Person.mk p false ∅
  (person.acquaintances.filter (λ a => (Person.mk a true ∅).believes_in_santa)).card = 
    (person.acquaintances.card / 10)

/-- Theorem: There exists a person who knows at least 810 people -/
theorem exists_person_with_many_acquaintances :
  ∃ p : Nat, p ∈ Population ∧ (Person.mk p false ∅).acquaintances.card ≥ 810 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_person_with_many_acquaintances_l651_65156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_det_projection_matrix_3_5_zero_l651_65121

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_v := Real.sqrt (v.1^2 + v.2^2)
  let a := v.1 / norm_v
  let b := v.2 / norm_v
  ![![a^2, a*b],
    ![a*b, b^2]]

theorem det_projection_matrix_zero (v : ℝ × ℝ) :
  Matrix.det (projection_matrix v) = 0 := by
  sorry

theorem det_projection_matrix_3_5_zero :
  Matrix.det (projection_matrix (3, 5)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_det_projection_matrix_3_5_zero_l651_65121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_allowance_proof_l651_65197

/-- The student's weekly allowance in dollars -/
def weekly_allowance : ℝ := 3.75

theorem weekly_allowance_proof :
  ∃ (a : ℝ),
    a > 0 ∧
    let arcade_spent := (3/5) * a;
    let remaining_after_arcade := a - arcade_spent;
    let toy_store_spent := (1/3) * remaining_after_arcade;
    let remaining_after_toy_store := remaining_after_arcade - toy_store_spent;
    remaining_after_toy_store = 1 ∧
    a = weekly_allowance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_allowance_proof_l651_65197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_f_zero_points_l651_65157

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (k - 1) * x + k + 5

-- Define the maximum value function
noncomputable def f_max (k : ℝ) : ℝ := if k < -7/2 then k + 5 else 7 * k + 26

-- Theorem for the maximum value of f(x) on [0, 3]
theorem f_maximum (k : ℝ) : 
  ∀ x ∈ Set.Icc 0 3, f k x ≤ f_max k := by
  sorry

-- Theorem for the range of k when f(x) has zero points on [0, 3]
theorem f_zero_points (k : ℝ) : 
  (∃ x ∈ Set.Icc 0 3, f k x = 0) ↔ k ∈ Set.Icc (-5) (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_f_zero_points_l651_65157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_walking_distance_l651_65140

theorem friend_walking_distance (trail_length : ℝ) (speed_ratio : ℝ) 
  (h1 : trail_length = 33)
  (h2 : speed_ratio = 1.2) : 
  trail_length * speed_ratio / (1 + speed_ratio) = 18 :=
by
  -- Replace this with the actual proof steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_walking_distance_l651_65140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_equals_five_l651_65112

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x else x + 1

-- Theorem statement
theorem f_composition_negative_two_equals_five :
  f (f (-2)) = 5 := by
  -- Evaluate f(-2)
  have h1 : f (-2) = 4 := by
    rw [f]
    simp [Real.rpow_neg]
    norm_num
  
  -- Evaluate f(4)
  have h2 : f 4 = 5 := by
    rw [f]
    simp
    norm_num
  
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_equals_five_l651_65112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_minimum_distance_l651_65198

/-- A race with a wall and a river obstacle -/
theorem race_minimum_distance 
  (wall_length : ℝ) 
  (a_to_b_vertical : ℝ) 
  (a_to_river : ℝ) 
  (river_width : ℝ) 
  (h1 : wall_length = 1300)
  (h2 : a_to_b_vertical = 800)
  (h3 : a_to_river = 100)
  (h4 : river_width = 50) :
  ∃ (min_distance : ℝ), 
    abs (min_distance - 1570) < 1 ∧ 
    (∀ (path : ℝ), 
      (path ≥ a_to_river + river_width) → 
      (path ≥ min_distance)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_minimum_distance_l651_65198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_exist_l651_65120

/-- Represents an algorithm --/
structure Algorithm where
  steps : List String
  is_finite : steps.length < ω
  is_clear : ∀ s ∈ steps, s ≠ ""  -- Simplified representation of clarity
  has_clear_output : String  -- Simplified representation of output

/-- Represents a problem that can be solved by algorithms --/
structure Problem where
  description : String

/-- Predicate to check if an algorithm solves a given problem --/
def SolvesProblem (a : Algorithm) (p : Problem) : Prop := sorry

/-- Theorem stating that multiple algorithms can exist for a single problem --/
theorem multiple_algorithms_exist (p : Problem) : 
  ∃ a₁ a₂ : Algorithm, a₁ ≠ a₂ ∧ SolvesProblem a₁ p ∧ SolvesProblem a₂ p := by
  sorry

#check multiple_algorithms_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_exist_l651_65120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_proof_l651_65164

theorem y_value_proof (b y : ℝ) (hb : b > 2) (hy : y > 0) 
  (heq : (3 * y) ^ (Real.log 3 / Real.log b) - (5 * y) ^ (Real.log 5 / Real.log b) = 0) : 
  y = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_proof_l651_65164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_volume_relation_helium_pressure_change_l651_65143

/-- Given a gas at constant temperature, prove that p₁ * v₁ = p₂ * v₂ -/
theorem pressure_volume_relation
  (p₁ : ℝ) (v₁ : ℝ) (p₂ : ℝ) (v₂ : ℝ)
  (h₁ : p₁ > 0) (h₂ : v₁ > 0) (h₃ : v₂ > 0)
  (h₄ : p₂ = p₁ * v₁ / v₂) :
  p₁ * v₁ = p₂ * v₂ :=
by sorry

/-- Given initial pressure and volume, and a new volume, calculate the new pressure -/
noncomputable def calculate_new_pressure (p₁ v₁ v₂ : ℝ) : ℝ :=
  p₁ * v₁ / v₂

/-- Prove that for the given initial conditions and new volume, the new pressure is 4 kPa -/
theorem helium_pressure_change
  (p₁ : ℝ) (v₁ : ℝ) (v₂ : ℝ)
  (h₁ : p₁ = 8) (h₂ : v₁ = 3.5) (h₃ : v₂ = 7) :
  calculate_new_pressure p₁ v₁ v₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_volume_relation_helium_pressure_change_l651_65143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_asymptote_distance_is_2sqrt2_l651_65141

/-- Represents a hyperbola with equation (y^2 / 8) - (x^2 / b^2) = 1 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt 2

/-- The distance from the focus to the asymptote of the hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ := 2 * Real.sqrt 2

/-- Theorem stating that the distance from the focus to the asymptote is 2√2 -/
theorem focus_asymptote_distance_is_2sqrt2 (h : Hyperbola) :
  focus_to_asymptote_distance h = 2 * Real.sqrt 2 := by
  -- Unfold the definition of focus_to_asymptote_distance
  unfold focus_to_asymptote_distance
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_asymptote_distance_is_2sqrt2_l651_65141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l651_65116

/-- An ellipse with given properties and a line intersecting it. -/
structure EllipseAndLine where
  /-- Semi-major axis of the ellipse -/
  a : ℝ
  /-- Semi-minor axis of the ellipse -/
  b : ℝ
  /-- Slope of the intersecting line -/
  m : ℝ
  /-- a > b > 0 -/
  h₁ : a > b ∧ b > 0
  /-- Eccentricity of the ellipse is √3/2 -/
  h₂ : Real.sqrt (a^2 - b^2) / a = Real.sqrt 3 / 2
  /-- The ellipse passes through point (1, -√3/2) -/
  h₃ : 1 / a^2 + 3 / (4 * b^2) = 1
  /-- The line intersects the ellipse at two distinct points -/
  h₄ : -Real.sqrt 5 < m ∧ m < Real.sqrt 5
  /-- The area of the triangle formed by the intersection points and origin is 1 -/
  h₅ : 2 * abs m * Real.sqrt (5 - m^2) = 5

/-- The main theorem about the ellipse and intersecting line. -/
theorem ellipse_and_line_properties (el : EllipseAndLine) :
  (el.a = 2 ∧ el.b = 1) ∧ (el.m = Real.sqrt 10 / 2 ∨ el.m = -Real.sqrt 10 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l651_65116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_imply_a_range_l651_65135

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

-- State the theorem
theorem monotonic_intervals_imply_a_range :
  ∀ a : ℝ, 
    a > 0 →
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a / 3 → f x < f y) →
    (∀ x y : ℝ, 2 * a ≤ x ∧ x < y ∧ y ≤ 4 * Real.pi / 3 → f x < f y) →
    5 * Real.pi / 12 ≤ a ∧ a ≤ Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_imply_a_range_l651_65135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_25pi_div_6_f_range_l651_65172

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x - Real.sqrt 3, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1 + Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_value_at_25pi_div_6 : f (25 * π / 6) = 0 := by sorry

theorem f_range (x : ℝ) (h : x ∈ Set.Icc (-π / 3) (π / 6)) :
  f x ∈ Set.Icc (-Real.sqrt 3) (1 - Real.sqrt 3 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_25pi_div_6_f_range_l651_65172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_cycle_edge_bound_l651_65193

/-- A simple graph with no self-loops or multiple edges -/
structure MySimpleGraph (V : Type*) where
  adj : V → V → Prop
  symm : ∀ {u v}, adj u v → adj v u
  loopless : ∀ v, ¬adj v v

/-- The number of vertices in a graph -/
def numVertices {V : Type*} (g : MySimpleGraph V) : ℕ := sorry

/-- The number of edges in a graph -/
def numEdges {V : Type*} (g : MySimpleGraph V) : ℕ := sorry

/-- A 4-cycle in a graph -/
def hasFourCycle {V : Type*} (g : MySimpleGraph V) : Prop := sorry

theorem no_four_cycle_edge_bound {V : Type*} (g : MySimpleGraph V) 
  (n : ℕ) (m : ℕ) 
  (hn : numVertices g = n) 
  (hm : numEdges g = m) 
  (h_no_four_cycle : ¬hasFourCycle g) : 
  (m : ℝ) ≤ (n / 4 : ℝ) * (1 + Real.sqrt (4 * n - 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_cycle_edge_bound_l651_65193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_factorial_base_nine_zeroes_l651_65178

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 12 factorial -/
def factorial12 : ℕ := Nat.factorial 12

theorem twelve_factorial_base_nine_zeroes :
  trailingZeroes factorial12 9 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_factorial_base_nine_zeroes_l651_65178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l651_65115

/-- Calculates the time taken for a train to cross a man walking in the same direction. -/
noncomputable def time_to_cross (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / ((train_speed - man_speed) * 1000 / 3600)

/-- Theorem stating that a 1500m train moving at 95 km/hr takes 60 seconds to cross a man walking at 5 km/hr in the same direction. -/
theorem train_crossing_time :
  time_to_cross 1500 95 5 = 60 := by
  -- Unfold the definition of time_to_cross
  unfold time_to_cross
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l651_65115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_of_exponents_for_1562_l651_65137

def sumOfDistinctPowersOf2 (n : ℕ) : ℕ := 
  let rec go (m : ℕ) (sum : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then sum
    else if m = 0 then sum
    else
      let k := Nat.log2 m
      go (m - 2^k) (sum + k) (fuel - 1)
  go n 0 (Nat.log2 n + 1)

theorem least_sum_of_exponents_for_1562 :
  sumOfDistinctPowersOf2 1562 = 27 := by
  rfl

#eval sumOfDistinctPowersOf2 1562

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_of_exponents_for_1562_l651_65137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_special_vectors_l651_65110

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between (a b : V) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem angle_between_special_vectors (a b : V) 
  (ha : norm a = 2)
  (hb : norm b = 1)
  (hab : norm (a - (2 : ℝ) • b) = 2 * Real.sqrt 3) :
  angle_between a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_special_vectors_l651_65110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_odd_l651_65165

theorem not_all_odd (a₁ a₂ a₃ a₄ a₅ b : ℤ) 
  (h : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = b^2) : 
  ¬(Odd a₁ ∧ Odd a₂ ∧ Odd a₃ ∧ Odd a₄ ∧ Odd a₅ ∧ Odd b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_odd_l651_65165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l651_65162

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l651_65162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_89_l651_65125

theorem decimal_to_binary_89 : 
  Nat.digits 2 89 = [1, 0, 1, 1, 0, 0, 1] :=
by
  -- The proof goes here
  sorry

#eval Nat.digits 2 89

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_89_l651_65125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_increases_with_radius_l651_65179

-- Define the volume of a sphere as a function of its radius
noncomputable def sphereVolume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

-- Theorem statement
theorem volume_increases_with_radius (R1 R2 : ℝ) (h1 : R1 > 1) (h2 : R2 > R1) :
  sphereVolume R1 < sphereVolume R2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_increases_with_radius_l651_65179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_quadratic_l651_65185

-- Define the concept of a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the given functions
noncomputable def f1 (x : ℝ) : ℝ := 3 * (x - 1)^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := x + 1/x
noncomputable def f3 (x : ℝ) : ℝ := 8 * x^2 + 1
noncomputable def f4 (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2

-- Theorem statement
theorem exactly_two_quadratic : 
  (is_quadratic f1 ∧ is_quadratic f3) ∧ 
  (¬is_quadratic f2 ∧ ¬is_quadratic f4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_quadratic_l651_65185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l651_65155

def satisfies_conditions (S : Set ℝ) : Prop :=
  (1 ∈ S) ∧
  (∀ x y, x ∈ S → y ∈ S → x > y → Real.sqrt (x^2 - y^2) ∈ S) ∧
  (∀ z, z ∈ S → z ≥ 1)

def sqrt_set : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.sqrt n}

def bounded_sqrt_set (n : ℕ) : Set ℝ :=
  {x | ∃ k : ℕ, k ≤ n ∧ x = Real.sqrt k}

theorem characterization_of_S :
  ∀ S : Set ℝ, satisfies_conditions S ↔
    (S = sqrt_set ∨ ∃ n : ℕ, S = bounded_sqrt_set n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l651_65155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_tangent_l651_65195

/-- The circle C with equation x^2 + y^2 + ax + 2ay + 2a^2 - a - 1 = 0 -/
def circleC (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 - a - 1 = 0

/-- The point P -/
def P : ℝ × ℝ := (-1, -2)

/-- The condition that there is exactly one tangent line from P to the circle C -/
def unique_tangent (a : ℝ) : Prop :=
  ∃! l : Set (ℝ × ℝ), (P ∈ l) ∧ (∀ p, p ∈ l → circleC a p.1 p.2 → ∃! q, q ∈ l ∧ circleC a q.1 q.2)

/-- There exists a unique value of a for which there is a unique tangent line -/
theorem unique_a_for_tangent : ∃! a : ℝ, unique_tangent a :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_tangent_l651_65195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_mean_weight_Y_Z_l651_65133

/-- Represents a pile of bricks -/
structure BrickPile where
  weight : ℝ  -- Total weight of the pile
  count : ℝ   -- Number of bricks in the pile (using ℝ for simplicity)

/-- Calculate the mean weight of a pile of bricks -/
noncomputable def meanWeight (pile : BrickPile) : ℝ :=
  pile.weight / pile.count

/-- Calculate the mean weight of two combined piles of bricks -/
noncomputable def combinedMeanWeight (pile1 pile2 : BrickPile) : ℝ :=
  (pile1.weight + pile2.weight) / (pile1.count + pile2.count)

theorem smallest_mean_weight_Y_Z (X Y Z : BrickPile)
  (hX : meanWeight X = 60)
  (hY : meanWeight Y = 70)
  (hXY : combinedMeanWeight X Y = 65)
  (hXZ : combinedMeanWeight X Z = 67) :
  combinedMeanWeight Y Z ≥ 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_mean_weight_Y_Z_l651_65133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l651_65105

/-- The area of a triangle given its three vertices -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- The theorem stating that the area of triangle PQR is 20 square units -/
theorem area_of_triangle_PQR :
  let P : ℝ × ℝ := (-2, 3)
  let Q : ℝ × ℝ := (6, 3)
  let R : ℝ × ℝ := (4, -2)
  triangle_area P Q R = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l651_65105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_inv_at_five_unique_solution_l651_65173

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x - 5

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (x + 5) / 2

-- Theorem stating that f(x) = f⁻¹(x) when x = 5
theorem f_equals_f_inv_at_five :
  f 5 = f_inv 5 := by
  -- Expand the definitions of f and f_inv
  simp [f, f_inv]
  -- Perform the arithmetic
  norm_num

-- Theorem stating that 5 is the only solution
theorem unique_solution (x : ℝ) :
  f x = f_inv x ↔ x = 5 := by
  -- Split into two directions
  constructor
  -- Forward direction
  · intro h
    -- Expand definitions and simplify
    simp [f, f_inv] at h
    -- Solve the resulting equation
    linarith
  -- Backward direction
  · intro h
    -- Substitute x = 5
    subst h
    -- Use the previous theorem
    exact f_equals_f_inv_at_five


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_inv_at_five_unique_solution_l651_65173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_food_expense_l651_65163

/-- Represents Rohan's monthly financial breakdown -/
structure RohanFinances where
  salary : ℚ
  savings : ℚ
  house_rent_percent : ℚ
  entertainment_percent : ℚ
  conveyance_percent : ℚ

/-- Calculates the percentage of salary spent on food -/
def food_expense_percent (r : RohanFinances) : ℚ :=
  100 - (r.house_rent_percent + r.entertainment_percent + r.conveyance_percent + r.savings / r.salary * 100)

/-- Theorem stating that Rohan spends 40% of his salary on food -/
theorem rohan_food_expense (r : RohanFinances) 
  (h1 : r.salary = 12500)
  (h2 : r.savings = 2500)
  (h3 : r.house_rent_percent = 20)
  (h4 : r.entertainment_percent = 10)
  (h5 : r.conveyance_percent = 10) :
  food_expense_percent r = 40 := by
  sorry

#eval food_expense_percent { 
  salary := 12500, 
  savings := 2500, 
  house_rent_percent := 20, 
  entertainment_percent := 10, 
  conveyance_percent := 10 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_food_expense_l651_65163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_max_distance_l651_65117

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem parabola_intersection_max_distance 
  (E : Parabola) 
  (F : Point)
  (l₁ : Line)
  (P Q : Point)
  (h1 : E.a = 1/4 ∧ E.h = 0 ∧ E.k = 0)  -- Equation of E: x² = 4y
  (h2 : F.x = 0 ∧ F.y = 1)              -- Focus at (0,1)
  (h3 : l₁.slope = 0 ∧ l₁.intercept = -1) -- Directrix: y = -1
  (h4 : P.x^2 = 4 * P.y ∧ Q.x^2 = 4 * Q.y) -- P and Q are on E
  (h5 : (P.y + Q.y) / 2 = 2)            -- Midpoint of PQ has y-coordinate 2
  : distance P Q ≤ 6 := by
  sorry

#check parabola_intersection_max_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_max_distance_l651_65117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l651_65118

noncomputable def f (x : ℝ) := (1/3) * x^3 + x^2 - 3*x - 4

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 2 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-4 : ℝ) 2 → f x ≤ f y) ∧
  f x = -17/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l651_65118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l651_65122

noncomputable def sequence_a (n : ℕ) : ℝ := 3^n

noncomputable def sequence_b (n : ℕ) : ℝ := 2 * sequence_a n - 3 * n

noncomputable def sum_S (n : ℕ) : ℝ := (sequence_a n - 1) / 2

noncomputable def sum_T (n : ℕ) : ℝ := 3^(n+1) - (3*n^2)/2 - (3*n)/2 - 3

theorem sequence_property (n : ℕ) :
  2 * (sum_S n) = 3 * (sequence_a n) - 3 ∧
  sequence_a n = 3^n ∧
  sum_T n = 3^(n+1) - (3*n^2)/2 - (3*n)/2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l651_65122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l651_65114

/-- Given the heights and shadow lengths of two objects measured at the same time and place,
    prove that the height of the second object is 20m. -/
theorem flagpole_height
  (h_xiao_ming : ℝ)
  (s_xiao_ming : ℝ)
  (s_flagpole : ℝ)
  (h_flagpole : ℝ)
  (h_xiao_ming_pos : h_xiao_ming > 0)
  (s_xiao_ming_pos : s_xiao_ming > 0)
  (s_flagpole_pos : s_flagpole > 0)
  (h_xiao_ming_val : h_xiao_ming = 1.6)
  (s_xiao_ming_val : s_xiao_ming = 0.4)
  (s_flagpole_val : s_flagpole = 5)
  (ratio_eq : h_xiao_ming / s_xiao_ming = h_flagpole / s_flagpole) :
  h_flagpole = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l651_65114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_all_distinct_l651_65174

theorem sequence_not_all_distinct (a : ℕ → ℚ) 
  (h : ∀ m n : ℕ, a m + a n = a (m * n))
  (nonneg : ∀ n : ℕ, 0 ≤ a n) : 
  ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_all_distinct_l651_65174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l651_65152

noncomputable def harry_pay (x : ℝ) (h : ℝ) : ℝ :=
  if h ≤ 24 then x * h
  else if h ≤ 35 then x * 24 + 1.5 * x * (h - 24)
  else x * 24 + 1.5 * x * 11 + 2 * x * (h - 35)

noncomputable def james_pay (x : ℝ) (h : ℝ) : ℝ :=
  if h ≤ 40 then x * h
  else if h ≤ 50 then x * 40 + 2 * x * (h - 40)
  else x * 40 + 2 * x * 10 + 2.5 * x * (h - 50)

-- Theorem statement
theorem harry_hours_worked (x : ℝ) (h_pos : x > 0) :
  harry_pay x 41 = james_pay x 47 ∧
  ∀ h : ℝ, h ≥ 0 ∧ h ≠ 41 → harry_pay x h ≠ james_pay x 47 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l651_65152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l651_65154

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3  -- Add this case to handle x < 0
  else if x < 1 then 2 * x^2
  else if x < 2 then 2
  else 3

-- Define the range of f
def range_f : Set ℝ := Set.range f

-- Theorem statement
theorem f_range : range_f = Set.union (Set.Icc 0 2) {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l651_65154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_quadratic_equation_1_solve_quadratic_equation_2_l651_65130

-- First equation
theorem solve_quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 / 2 ∧ x₂ = 1 - Real.sqrt 2 / 2 ∧
  ∀ x : ℝ, 2 * x^2 - 4 * x + 1 = 0 ↔ x = x₁ ∨ x = x₂ := by sorry

-- Second equation
theorem solve_quadratic_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁ = -3/2 ∧ x₂ = -1/2 ∧
  ∀ x : ℝ, (2*x + 3)^2 - 4*x - 6 = 0 ↔ x = x₁ ∨ x = x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_quadratic_equation_1_solve_quadratic_equation_2_l651_65130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_from_condition_l651_65187

open Matrix Complex

theorem matrix_equality_from_condition (n : ℕ) (A : Matrix (Fin n) (Fin n) ℂ) :
  A + Aᴴ = A ^ 2 * Aᴴ → A = Aᴴ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_from_condition_l651_65187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_line_equation_l651_65146

-- Define the given line
noncomputable def given_line (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

-- Define the angle between two lines
noncomputable def angle_between_lines (m1 m2 : ℝ) : ℝ := Real.arctan ((m2 - m1) / (1 + m1 * m2))

-- Define the condition for the desired line
noncomputable def desired_line_condition (x y : ℝ) : Prop :=
  (x = -1 ∧ y = Real.sqrt 3) ∧
  (∃ m : ℝ, angle_between_lines (Real.sqrt 3) m = Real.pi / 6 ∨
                angle_between_lines (Real.sqrt 3) m = -Real.pi / 6)

-- Theorem statement
theorem desired_line_equation (x y : ℝ) :
  desired_line_condition x y →
  (x + 1 = 0 ∨ x - Real.sqrt 3 * y + 4 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_line_equation_l651_65146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_points_line_segments_l651_65113

/-- A point in a plane with a color --/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Fin 4

/-- A line segment connecting two points --/
structure LineSegment where
  p1 : ColoredPoint
  p2 : ColoredPoint

/-- Predicate to check if two line segments intersect --/
def intersect (l1 l2 : LineSegment) : Prop :=
  sorry

/-- Theorem: Given 20 points in a plane colored with 4 different colors,
    where each color is assigned to exactly 5 points, it is always possible
    to select 4 non-intersecting line segments such that each segment connects
    two points of the same color, and the endpoints of different segments
    have different colors. --/
theorem colored_points_line_segments
  (points : Finset ColoredPoint)
  (h1 : points.card = 20)
  (h2 : ∀ c : Fin 4, (points.filter (λ p => p.color = c)).card = 5) :
  ∃ (segments : Finset LineSegment),
    segments.card = 4 ∧
    (∀ l, l ∈ segments → l.p1.color = l.p2.color) ∧
    (∀ l1 l2, l1 ∈ segments → l2 ∈ segments → l1 ≠ l2 → ¬(intersect l1 l2)) ∧
    (∀ l1 l2, l1 ∈ segments → l2 ∈ segments → l1 ≠ l2 → l1.p1.color ≠ l2.p1.color) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_points_line_segments_l651_65113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l651_65109

theorem exam_students_count (N : ℕ) (avg_all avg_excluded avg_remaining : ℝ) : ℕ :=
  let excluded : ℕ := 5
  have avg_all_eq : avg_all = 72 := by sorry
  have excluded_eq : excluded = 5 := by rfl
  have avg_excluded_eq : avg_excluded = 40 := by sorry
  have avg_remaining_eq : avg_remaining = 92 := by sorry
  have h : (N * avg_all - excluded * avg_excluded) / (N - excluded) = avg_remaining := by sorry
  have : N = 13 := by
    sorry
  13

#check exam_students_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l651_65109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_breadth_increase_l651_65138

/-- Proves that the percentage increase in breadth for the second carpet is 25% -/
theorem carpet_breadth_increase (breadth_1 : ℝ) (length_1 : ℝ) (length_2 : ℝ) 
  (cost_2 : ℝ) (rate : ℝ) : 
  breadth_1 = 6 →
  length_1 = 1.44 * breadth_1 →
  length_2 = length_1 * 1.4 →
  cost_2 = 4082.4 →
  rate = 45 →
  (cost_2 / rate / length_2 - breadth_1) / breadth_1 * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_breadth_increase_l651_65138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_g_omega_range_l651_65132

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + Real.pi / 6)

theorem monotonic_decreasing_g_omega_range :
  ∀ ω : ℝ, 
    (ω > 0 ∧ 
     (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi / 2 → g ω x₁ > g ω x₂)) 
    ↔ 
    (0 < ω ∧ ω ≤ 5 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_g_omega_range_l651_65132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_eq_l651_65148

/-- The side length of the largest square that can be inscribed in the space inside a square 
    with side length 15, but outside of two congruent equilateral triangles drawn as described. -/
noncomputable def largest_inscribed_square_side : ℝ :=
  (30 - 15 * Real.sqrt 3) / 2

/-- The outer square has a side length of 15. -/
def outer_square_side : ℝ := 15

/-- The side length of the equilateral triangles. -/
def triangle_side : ℝ := outer_square_side

/-- The height of the equilateral triangles. -/
noncomputable def triangle_height : ℝ := (Real.sqrt 3 / 2) * triangle_side

theorem largest_inscribed_square_side_eq :
  largest_inscribed_square_side = outer_square_side - triangle_height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_eq_l651_65148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_already_added_marys_flour_problem_l651_65171

/-- Given a recipe that requires a certain number of cups of flour and the number of cups still needed to be added, calculate the number of cups already put in. -/
theorem flour_already_added (total_required : ℕ) (cups_to_add : ℕ) 
  (h : total_required ≥ cups_to_add) : 
  total_required - cups_to_add = total_required - cups_to_add := by
  rfl

/-- Solve Mary's baking problem -/
theorem marys_flour_problem : 
  let total_required : ℕ := 12
  let cups_to_add : ℕ := 1
  total_required - cups_to_add = 11 := by
  norm_num

#eval 12 - 1  -- This will output 11, confirming our calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_already_added_marys_flour_problem_l651_65171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_change_is_24_l651_65142

/-- The exchange rate of the dollar on January 1, 2014, in rubles -/
def initial_rate : ℝ := 32.6587

/-- The exchange rate of the dollar on December 31, 2014, in rubles -/
def final_rate : ℝ := 56.2584

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  Int.floor (x + 0.5)

/-- The change in dollar exchange rate from January 1, 2014, to December 31, 2014, rounded to the nearest whole number -/
noncomputable def exchange_rate_change : ℤ :=
  round_to_nearest (final_rate - initial_rate)

theorem exchange_rate_change_is_24 :
  exchange_rate_change = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_change_is_24_l651_65142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_is_36km_l651_65181

/-- Represents the walking scenario with two speeds and a distance difference -/
structure WalkingScenario where
  slow_speed : ℝ
  fast_speed : ℝ
  distance_difference : ℝ

/-- Calculates the actual distance traveled given a WalkingScenario -/
noncomputable def actual_distance (scenario : WalkingScenario) : ℝ :=
  let time := scenario.distance_difference / (scenario.fast_speed - scenario.slow_speed)
  scenario.slow_speed * time

/-- Theorem stating that for the given scenario, the actual distance is 36 km -/
theorem actual_distance_is_36km : 
  let scenario := WalkingScenario.mk 12 20 24
  actual_distance scenario = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_is_36km_l651_65181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cross_section_eccentricity_30_degrees_l651_65107

noncomputable def cylinder_cross_section_eccentricity (R : ℝ) (θ : ℝ) : ℝ :=
  Real.sqrt (1 - (Real.cos θ) ^ 2)

theorem cylinder_cross_section_eccentricity_30_degrees (R : ℝ) :
  cylinder_cross_section_eccentricity R (π / 6) = 1 / 2 :=
by
  unfold cylinder_cross_section_eccentricity
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cross_section_eccentricity_30_degrees_l651_65107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l651_65199

-- Define the function h
noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3*x - 30

-- State the theorem
theorem h_composition_equality {a : ℝ} (ha : a < 0) :
  h (h (h 6)) = h (h (h a)) → a = -14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l651_65199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_fifteen_l651_65124

/-- Represents the road construction project -/
structure RoadProject where
  totalDays : ℕ
  initialWorkers : ℕ
  daysElapsed : ℕ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total length of the road given the project parameters -/
noncomputable def calculateRoadLength (project : RoadProject) : ℝ :=
  let totalWorkers := (project.initialWorkers + project.extraWorkers : ℝ)
  let remainingDays := (project.totalDays - project.daysElapsed : ℝ)
  let dailyRate := project.completedLength / project.daysElapsed
  project.completedLength + (totalWorkers / project.initialWorkers) * dailyRate * remainingDays

/-- Theorem stating that the calculated road length for the given project is 15 km -/
theorem road_length_is_fifteen (project : RoadProject) 
    (h1 : project.totalDays = 300)
    (h2 : project.initialWorkers = 30)
    (h3 : project.daysElapsed = 100)
    (h4 : project.completedLength = 2.5)
    (h5 : project.extraWorkers = 45) : 
  calculateRoadLength project = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_fifteen_l651_65124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_C₁_C₂_l651_65101

-- Define curve C₁
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, Real.sin φ)

-- Define curve C₂ (center and radius in Cartesian coordinates)
def C₂_center : ℝ × ℝ := (0, 3)
def C₂_radius : ℝ := 1

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_range_C₁_C₂ :
  ∀ (φ : ℝ) (N : ℝ × ℝ),
  (distance C₂_center N = C₂_radius) →
  1 ≤ distance (C₁ φ) N ∧ distance (C₁ φ) N ≤ 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_C₁_C₂_l651_65101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l651_65184

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  |r₁ - r₂| = 6 ↔ a = 1 ∧ b = 42 ∧ c = 408 := by
  sorry

#check root_difference_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l651_65184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_exclusion_l651_65104

-- Define the polynomial P(x)
def P (a : ℕ → ℤ) (k : ℕ) (x : ℤ) : ℤ := 
  (Finset.range (k + 1)).sum (λ i ↦ a i * x^i)

-- State the theorem
theorem polynomial_value_exclusion 
  (a : ℕ → ℤ) (k : ℕ) 
  (x₁ x₂ x₃ x₄ : ℤ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h_roots : P a k x₁ = 2 ∧ P a k x₂ = 2 ∧ P a k x₃ = 2 ∧ P a k x₄ = 2) :
  ∀ (x : ℤ), ∀ (m : Fin 5), P a k x ≠ 2 * (m : ℤ) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_exclusion_l651_65104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l651_65194

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 4)

-- Theorem statement
theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (-π / 8 + x) = f (-π / 8 - x) :=
by
  intro x
  -- Expand the definition of f
  simp [f]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l651_65194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_tournament_probability_l651_65170

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  matches_per_team : Nat
  win_probability : ℚ
  points_for_win : Nat

/-- The probability that one team finishes with more points than another,
    given they've already won their direct match -/
def probability_of_more_points (t : SoccerTournament) : ℚ :=
  3172 / 4096

/-- The main theorem to prove -/
theorem soccer_tournament_probability 
  (t : SoccerTournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.matches_per_team = 7)
  (h3 : t.win_probability = 1/2)
  (h4 : t.points_for_win = 1) :
  probability_of_more_points t = 3172 / 4096 := by
  sorry

#eval probability_of_more_points { num_teams := 8, matches_per_team := 7, win_probability := 1/2, points_for_win := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_tournament_probability_l651_65170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_can_escape_l651_65177

structure Garden where
  side_length : ℝ
  rabbit_speed : ℝ
  wolf_speed : ℝ
  wolf_speed_ratio : wolf_speed = 1.4 * rabbit_speed

theorem rabbit_can_escape (g : Garden) (h : g.wolf_speed < Real.sqrt 2 * g.rabbit_speed) :
  ∃ (escape_path : ℝ → ℝ × ℝ), 
    (escape_path 0 = (0, 0)) ∧ 
    (∃ (t : ℝ), t > 0 ∧ (|((escape_path t).1)| = g.side_length / 2 ∨ |((escape_path t).2)| = g.side_length / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_can_escape_l651_65177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_when_sum_maximized_l651_65176

theorem min_difference_when_sum_maximized (x : Fin 9 → ℕ) 
  (h_order : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.sum Finset.univ (λ i => x i)) = 220)
  (h_max_sum : ∀ y : Fin 9 → ℕ, 
    (∀ i j, i < j → y i < y j) → 
    ((Finset.sum Finset.univ (λ i => y i)) = 220) → 
    (Finset.sum (Finset.range 5) (λ i => y i) ≤ Finset.sum (Finset.range 5) (λ i => x i))) :
  x 8 - x 0 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_when_sum_maximized_l651_65176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semifinalists_count_l651_65175

theorem semifinalists_count (n : ℕ) (h : Nat.choose (n - 2) 3 = 56) : n = 10 := by
  -- Define the number of finalists
  let finalists := n - 2

  -- Define the number of medals awarded
  let medals := 3

  -- Define the number of possible groups of medal winners
  let medal_groups := 56

  -- The combination of finalists choosing medals equals the number of medal groups
  have : Nat.choose finalists medals = medal_groups := h

  -- Prove that n = 10
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semifinalists_count_l651_65175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_l651_65186

-- Define the polar coordinates
noncomputable def rho : ℝ := 2
noncomputable def theta : ℝ := (4 * Real.pi) / 3

-- Define the rectangular coordinates
noncomputable def x : ℝ := -1
noncomputable def y : ℝ := -Real.sqrt 3

-- Theorem statement
theorem polar_to_rectangular :
  (x * x + y * y = rho * rho) ∧
  (Real.tan theta = y / x) ∧
  (x < 0) ∧ (y < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_l651_65186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_value_ratio_example_l651_65147

/-- The ratio of place values for digits in a decimal number -/
def placeValueRatio (n : ℚ) (d1 d2 : ℕ) : ℚ :=
  let s := toString n
  let digits := s.toList
  let d1_index := digits.indexOf (toString d1).front
  let d2_index := digits.indexOf (toString d2).front
  (10 : ℚ) ^ (digits.length - d1_index - 2 - (digits.length - d2_index - 2))

theorem place_value_ratio_example :
  placeValueRatio 52674.1892 6 8 = 10000 := by
  sorry

#eval placeValueRatio 52674.1892 6 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_value_ratio_example_l651_65147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_area_l651_65168

-- Define the function g on a domain of three points
noncomputable def g : Fin 3 → ℝ := sorry

-- Define the area of the triangle formed by the graph of g
noncomputable def area_g : ℝ := sorry

-- Define the area of the triangle formed by the graph of 4g(3x)
noncomputable def area_transformed : ℝ := sorry

-- Theorem statement
theorem transformed_area (h : area_g = 45) : area_transformed = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_area_l651_65168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2018_l651_65131

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => (1 + sequence_a (n + 1)) / (1 - sequence_a (n + 1))

theorem sequence_a_2018 : sequence_a 2018 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2018_l651_65131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cargo_truck_min_cost_l651_65145

/-- The total cost function for the cargo truck trip -/
noncomputable def total_cost (x : ℝ) : ℝ := 2340 / x + 13 * x / 18

/-- The theorem stating the minimum cost and optimal speed -/
theorem cargo_truck_min_cost :
  ∃ (x : ℝ), 50 ≤ x ∧ x ≤ 100 ∧
  (∀ y : ℝ, 50 ≤ y ∧ y ≤ 100 → total_cost x ≤ total_cost y) ∧
  x = 18 * Real.sqrt 10 ∧
  total_cost x = 26 * Real.sqrt 10 := by
  sorry

#check cargo_truck_min_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cargo_truck_min_cost_l651_65145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l651_65180

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let triangle_area := (1/2) * a * b * Real.sin C
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  c = 2 ∧
  Real.sqrt 3 * a = 2 * c * Real.sin A ∧
  triangle_area = Real.sqrt 3 →
  C = Real.pi/3 ∧ a = 2 ∧ b = 2

theorem triangle_theorem :
  ∀ (a b c : ℝ) (A B C : ℝ),
  triangle_proof a b c A B C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l651_65180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l651_65182

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) := by sorry

-- Define the set of real numbers satisfying f(x) ≥ 6 when a = 1
def solution_set : Set ℝ := {x : ℝ | f 1 x ≥ 6}

-- Define the set of real numbers a for which f(x) > -a for all x
def a_range : Set ℝ := {a : ℝ | ∀ x, f a x > -a}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l651_65182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_l651_65161

def is_valid_solution (x y z : ℕ) : Prop :=
  (x * y) % z = 2 ∧
  (y * z) % x = 2 ∧
  (z * x) % y = 2

theorem congruence_solutions :
  ∀ x y z : ℕ,
    x > 0 → y > 0 → z > 0 →
    is_valid_solution x y z ↔
      ((x, y, z) = (3, 8, 22) ∨
       (x, y, z) = (3, 10, 14) ∨
       (x, y, z) = (4, 5, 18) ∨
       (x, y, z) = (4, 6, 11) ∨
       (x, y, z) = (6, 14, 82) ∨
       (x, y, z) = (6, 22, 26)) :=
by sorry

#check congruence_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_l651_65161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_uniqueness_l651_65106

/-- A quadratic function with specific properties -/
structure QuadraticFunction (f : ℝ → ℝ) : Prop where
  exists_abc : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (3 + x) = f (3 - x)
  min_value : f 3 = -2
  passes_through : f 0 = 1

/-- The specific quadratic function we want to prove -/
noncomputable def TargetFunction (x : ℝ) : ℝ :=
  (1/3) * x^2 - 2 * x + 1

/-- Theorem stating that the quadratic function with given properties is unique -/
theorem quadratic_function_uniqueness :
  ∀ f : ℝ → ℝ, QuadraticFunction f → f = TargetFunction := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_uniqueness_l651_65106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fuel_consumption_l651_65123

/-- Fuel consumption function (liters per hour) --/
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

/-- Total fuel consumed for a 100 km journey at speed x --/
noncomputable def total_fuel (x : ℝ) : ℝ := fuel_consumption x * (100 / x)

/-- The minimum fuel consumption occurs at 80 km/h and is 11.25 liters --/
theorem min_fuel_consumption :
  ∃ (x : ℝ), x > 0 ∧ x ≤ 120 ∧
  (∀ (y : ℝ), y > 0 → y ≤ 120 → total_fuel x ≤ total_fuel y) ∧
  x = 80 ∧ total_fuel x = 11.25 := by
  sorry

#check min_fuel_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fuel_consumption_l651_65123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l651_65129

def is_valid_permutation (a b c d : ℕ) : Prop :=
  Multiset.ofList [a, b, c, d] = Multiset.ofList [2, 3, 4, 5]

def product_sum (a b c d : ℕ) : ℕ :=
  a * b + a * c + b * d + c * d

theorem max_product_sum :
  ∀ a b c d : ℕ, is_valid_permutation a b c d →
  product_sum a b c d ≤ 49 ∧
  ∃ a' b' c' d' : ℕ, is_valid_permutation a' b' c' d' ∧ product_sum a' b' c' d' = 49 :=
by
  sorry

#eval product_sum 2 3 4 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l651_65129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_distance_for_given_trapezoid_l651_65191

/-- Represents a rectangular trapezoid ABCD -/
structure RectangularTrapezoid where
  ab : ℝ
  ad : ℝ
  dc : ℝ

/-- Calculates the area of the rectangular trapezoid -/
noncomputable def area (t : RectangularTrapezoid) : ℝ :=
  t.ab * t.ad + (t.dc - t.ab) * t.ad / 2

/-- Calculates the distance from D to divide the trapezoid into two equal areas -/
noncomputable def divisionDistance (t : RectangularTrapezoid) : ℝ :=
  area t / (2 * t.ad)

theorem division_distance_for_given_trapezoid :
  let t : RectangularTrapezoid := { ab := 30, ad := 20, dc := 45 }
  divisionDistance t = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_distance_for_given_trapezoid_l651_65191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missouri_to_newyork_distance_l651_65139

/-- The distance from Missouri to New York by car -/
noncomputable def distance_missouri_to_newyork (flying_distance : ℝ) (driving_increase : ℝ) : ℝ :=
  (flying_distance * (1 + driving_increase)) / 2

/-- Theorem: The distance from Missouri to New York by car is 1400 miles -/
theorem missouri_to_newyork_distance :
  distance_missouri_to_newyork 2000 0.4 = 1400 := by
  -- Unfold the definition of distance_missouri_to_newyork
  unfold distance_missouri_to_newyork
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_missouri_to_newyork_distance_l651_65139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l651_65126

theorem constant_term_expansion (a : ℝ) : 
  (∃ c : ℝ → ℝ, c 0 = 14 ∧ 
   ∀ x : ℝ, x > 0 → c x = (x^(1/6) - a/x^(1/2))^8) → 
  a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l651_65126
