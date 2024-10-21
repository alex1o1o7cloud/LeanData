import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l620_62066

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l620_62066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l620_62005

theorem quadratic_coefficient (a b c : ℤ) (f : ℝ → ℝ) :
  f = (λ x : ℝ => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)) →
  (∀ x : ℝ, f x = (a : ℝ) * (x - 2)^2 + 5) →
  f 3 = 4 →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l620_62005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equal_sum_greater_than_two_l620_62068

open Real

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := log x - x

-- State the theorem
theorem f_equal_sum_greater_than_two (x₁ x₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : 0 < x₂) (h3 : x₁ < x₂) (h4 : f x₁ = f x₂) : 
  x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equal_sum_greater_than_two_l620_62068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l620_62015

open Real

/-- The radius of an inscribed sphere in a regular triangular pyramid -/
noncomputable def inscribed_sphere_radius (a : ℝ) (α : ℝ) : ℝ :=
  (a / 6) * sqrt (3 * sin (π / 3 - α / 2) / sin (π / 3 + α / 2))

/-- Theorem: The radius of the sphere inscribed in a regular triangular pyramid
    with base side length a and apex dihedral angle α is given by the formula. -/
theorem inscribed_sphere_radius_formula (a α : ℝ) 
    (h_a : a > 0) 
    (h_α : 0 < α ∧ α < π) : 
  inscribed_sphere_radius a α = 
    (a / 6) * sqrt (3 * sin (π / 3 - α / 2) / sin (π / 3 + α / 2)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l620_62015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_length_in_square_with_equilateral_triangles_l620_62083

noncomputable section

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- An equilateral triangle with base on the x-axis -/
structure EquilateralTriangle where
  base : ℝ
  height : ℝ
  apex : Point

/-- A square with side length 1 -/
structure UnitSquare where
  a : Point
  b : Point
  c : Point
  d : Point

/-- The main theorem -/
theorem xy_length_in_square_with_equilateral_triangles 
  (square : UnitSquare)
  (triangle1 : EquilateralTriangle)
  (triangle2 : EquilateralTriangle)
  (h1 : square.a = ⟨0, 0⟩)
  (h2 : square.b = ⟨1, 0⟩)
  (h3 : square.c = ⟨1, 1⟩)
  (h4 : square.d = ⟨0, 1⟩)
  (h5 : triangle1.base = 1)
  (h6 : triangle1.height = Real.sqrt 3 / 2)
  (h7 : triangle1.apex = ⟨1/2, Real.sqrt 3 / 2⟩)
  (h8 : triangle2.base = 1)
  (h9 : triangle2.height = Real.sqrt 3 / 2)
  (h10 : triangle2.apex = ⟨1/2, 1 - Real.sqrt 3 / 2⟩) :
  distance triangle1.apex triangle2.apex = Real.sqrt 3 - 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_length_in_square_with_equilateral_triangles_l620_62083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l620_62086

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0)
    and an asymptote equation 2x - y = 0, 
    the eccentricity of C is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ C : Set (ℝ × ℝ), C = {(x, y) | x^2/a^2 - y^2/b^2 = 1}) →
  (∃ asymptote : Set (ℝ × ℝ), asymptote = {(x, y) | 2*x - y = 0}) →
  (∃ e : ℝ, e = Real.sqrt 5 ∧ e = Real.sqrt (1 + (b/a)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l620_62086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l620_62000

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (A B F₁ F₂ : Point)
  (h_right_branch : A.x > 0 ∧ B.x > 0)
  (h_F₂_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    F₂.x = A.x + t * (B.x - A.x) ∧ 
    F₂.y = A.y + t * (B.y - A.y))
  (h_AF₁_perp_AB : (A.x - F₁.x) * (B.x - A.x) + (A.y - F₁.y) * (B.y - A.y) = 0)
  (h_distance_prop : 4 * distance A F₁ = 3 * distance A B) :
  eccentricity h = Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l620_62000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_triangle_AQR_l620_62010

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if one circle is internally tangent to another at a given point -/
def is_internally_tangent (ω Γ : Circle) (P : Point) : Prop := sorry

/-- Checks if two points form a chord of a given circle -/
def is_chord_of (Γ : Circle) (A B : Point) : Prop := sorry

/-- Checks if a line segment is tangent to a circle at a given point -/
def is_tangent_to (ω : Circle) (A B Q : Point) : Prop := sorry

/-- Checks if a point is on a line defined by two other points -/
def on_line (P Q R : Point) : Prop := sorry

/-- Checks if a point is on a given circle -/
def on_circle (Γ : Circle) (P : Point) : Prop := sorry

/-- Calculates the distance between two points -/
noncomputable def distance (A B : Point) : ℝ := sorry

/-- Calculates the circumradius of a triangle given its three vertices -/
noncomputable def circumradius (A Q R : Point) : ℝ := sorry

/-- Given two circles ω and Γ, where ω is internally tangent to Γ at point P,
    and a chord AB of Γ tangent to ω at point Q, with R ≠ P being the second
    intersection of line PQ with Γ, prove that the circumradius of triangle AQR
    is √170 when the radius of Γ is 17, the radius of ω is 7, and AQ/BQ = 3. -/
theorem circumradius_of_triangle_AQR
  (ω Γ : Circle)
  (P Q R A B : Point)
  (h_tangent : is_internally_tangent ω Γ P)
  (h_chord : is_chord_of Γ A B)
  (h_tangent_chord : is_tangent_to ω A B Q)
  (h_intersection : R ≠ P ∧ on_line P Q R ∧ on_circle Γ R)
  (h_radius_Γ : Γ.radius = 17)
  (h_radius_ω : ω.radius = 7)
  (h_ratio : distance A Q / distance B Q = 3)
  : circumradius A Q R = Real.sqrt 170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_triangle_AQR_l620_62010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annes_female_cat_weight_l620_62078

/-- The weight of Anne's female cat in kilograms -/
def female_cat_weight : ℝ := sorry

/-- The weight of Anne's male cat in kilograms -/
def male_cat_weight : ℝ := sorry

/-- The total weight of both cats in kilograms -/
def total_weight : ℝ := sorry

/-- Theorem stating the weight of Anne's female cat -/
theorem annes_female_cat_weight :
  female_cat_weight > 0 ∧
  male_cat_weight = 2 * female_cat_weight ∧
  total_weight = 6 ∧
  total_weight = female_cat_weight + male_cat_weight →
  female_cat_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annes_female_cat_weight_l620_62078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l620_62090

noncomputable section

/-- A function f defined as f(x) = 2sin(ωx + π/4) -/
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)

/-- Theorem stating that if the distance between highest and lowest points of f is 5, then ω = π/3 -/
theorem omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ x y : ℝ, f ω x = 2 ∧ f ω y = -2 ∧ |x - y| = 5) : 
  ω = Real.pi / 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l620_62090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burglary_sentence_calculation_l620_62096

noncomputable def total_sentence (stolen_value : ℝ) (value_per_year : ℝ) : ℝ :=
  (stolen_value / value_per_year) * 1.25 + 2

theorem burglary_sentence_calculation :
  ∃ (value_per_year : ℝ),
    value_per_year > 0 ∧
    total_sentence 40000 value_per_year = 12 ∧
    value_per_year = 5000 := by
  use 5000
  constructor
  · norm_num
  constructor
  · norm_num
    rw [total_sentence]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_burglary_sentence_calculation_l620_62096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_yield_calculation_l620_62020

/-- Represents the corn yield problem with given conditions -/
structure CornYield where
  total_acres : ℚ
  good_soil_yield : ℚ
  clay_rich_soil_fraction : ℚ
  good_soil_fraction : ℚ

/-- Calculates the total corn yield based on the given conditions -/
def total_yield (cy : CornYield) : ℚ :=
  let good_soil_area := cy.good_soil_fraction * cy.total_acres
  let clay_rich_soil_area := cy.clay_rich_soil_fraction * cy.total_acres
  let clay_rich_soil_yield := cy.good_soil_yield / 2
  good_soil_area * cy.good_soil_yield + clay_rich_soil_area * clay_rich_soil_yield

/-- Theorem stating that the total corn yield is 20000 bushels -/
theorem corn_yield_calculation (cy : CornYield) 
    (h1 : cy.total_acres = 60)
    (h2 : cy.good_soil_yield = 400)
    (h3 : cy.clay_rich_soil_fraction = 1/3)
    (h4 : cy.good_soil_fraction = 2/3) :
  total_yield cy = 20000 := by
  sorry

#eval total_yield { total_acres := 60, good_soil_yield := 400, clay_rich_soil_fraction := 1/3, good_soil_fraction := 2/3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_yield_calculation_l620_62020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_zero_two_four_l620_62041

noncomputable def absolute_value (x : ℝ) : ℝ := max x (-x)

noncomputable def equation (x : ℝ) : ℝ :=
  2 * 88 * absolute_value (absolute_value (absolute_value (absolute_value (x - 1) - 1) - 1) - 1)

theorem solutions_zero_two_four :
  equation 0 = 0 ∧ equation 2 = 0 ∧ equation 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_zero_two_four_l620_62041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_and_empty_intersection_l620_62088

def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem subset_intersection_and_empty_intersection (a : ℝ) :
  (Set.Nonempty (A a ∩ B)) ∧ (A a ∩ C = ∅) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_and_empty_intersection_l620_62088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_l620_62042

-- Define the functions f and h
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 4)

noncomputable def h (x : ℝ) : ℝ := 2 * (f⁻¹ x)

-- State the theorem
theorem h_equals_20 : 
  ∃ x : ℝ, h x = 20 ∧ x = 15/7 := by
  sorry

-- If you want to add some steps to the proof (still using sorry), you can do:
-- exists 15/7
-- apply And.intro
-- · sorry -- Prove h (15/7) = 20
-- · rfl -- Prove 15/7 = 15/7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_l620_62042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l620_62026

noncomputable def data : List ℝ := [-2, -1, 0, 3, 5]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem variance_of_data : variance data = 34/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l620_62026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_of_negative_seven_equals_three_fourths_l620_62085

-- Define the functions h and k
def h (x : ℝ) : ℝ := 4 * x - 9

noncomputable def k (x : ℝ) : ℝ := 
  let y := (x + 9) / 4  -- This is h⁻¹(x)
  3 * y^2 + 4 * y - 2

-- Theorem statement
theorem k_of_negative_seven_equals_three_fourths :
  k (-7) = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_of_negative_seven_equals_three_fourths_l620_62085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_one_l620_62062

theorem trig_expression_equals_one :
  Real.sin (50 * π / 180) * (Real.tan (45 * π / 180) + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_one_l620_62062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combine_with_3sqrt2_l620_62036

-- Define the concept of "can be combined"
def can_be_combined (x y : ℝ) : Prop :=
  ∃ (k : ℤ), x = k * y

-- Define the square roots we're working with
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt3 : ℝ := Real.sqrt 3
noncomputable def sqrt6 : ℝ := Real.sqrt 6
noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def sqrt12 : ℝ := Real.sqrt 12
noncomputable def sqrt_one_third : ℝ := Real.sqrt (1/3)

-- State the theorem
theorem combine_with_3sqrt2 :
  can_be_combined sqrt8 (3 * sqrt2) ∧
  ¬can_be_combined sqrt6 (3 * sqrt2) ∧
  ¬can_be_combined sqrt_one_third (3 * sqrt2) ∧
  ¬can_be_combined sqrt12 (3 * sqrt2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combine_with_3sqrt2_l620_62036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_share_l620_62008

theorem partnership_profit_share 
  (a b c : ℚ) 
  (total_profit : ℚ)
  (h1 : a = 6)
  (h2 : b = 2)
  (h3 : c = 9)
  (h4 : total_profit = 11000) :
  c / (a + b + c) * total_profit = 9 / 17 * 11000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_share_l620_62008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l620_62034

/-- The asymptotic line equations for the hyperbola x²/8 - y²/6 = 1 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 8 - y^2 / 6 = 1) →
  (∃ (k : ℝ), k = Real.sqrt 3 / 2 ∧ (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l620_62034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_when_cost_is_99_percent_of_selling_l620_62094

noncomputable def profit_percentage (cost_price : ℝ) (selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

theorem profit_percentage_when_cost_is_99_percent_of_selling :
  ∀ (selling_price : ℝ),
  selling_price > 0 →
  let cost_price := 0.99 * selling_price
  abs (profit_percentage cost_price selling_price - 1.01) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_when_cost_is_99_percent_of_selling_l620_62094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l620_62089

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the x-intercept of a line --/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.a

/-- Calculates the y-intercept of a line --/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

/-- Calculates the slope of a line --/
noncomputable def line_slope (l : Line) : ℝ := -l.a / l.b

/-- Theorem stating the properties of the line 4x - 5y - 20 = 0 --/
theorem line_properties :
  let l : Line := { a := 4, b := -5, c := -20 }
  x_intercept l = 5 ∧ y_intercept l = -4 ∧ line_slope l = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l620_62089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l620_62073

theorem inequality_proof (n : ℕ+) (x : ℝ) (hx : x > 0) :
  x + (n : ℝ)^(n : ℕ) / x^(n : ℕ) ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l620_62073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_can_be_partitioned_into_four_congruent_parts_l620_62013

/-- Represents a point on a grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a shape on a grid --/
structure GridShape where
  points : Set GridPoint

/-- Represents a partition of a grid shape --/
structure GridPartition where
  parts : Finset (Set GridPoint)

/-- Checks if a partition is valid for a given shape --/
def isValidPartition (shape : GridShape) (partition : GridPartition) : Prop :=
  (partition.parts.card = 4) ∧
  (∀ p, p ∈ partition.parts → p ⊆ shape.points) ∧
  (∀ p q, p ∈ partition.parts → q ∈ partition.parts → p ≠ q → p ∩ q = ∅) ∧
  (⋃₀ partition.parts.toSet = shape.points)

/-- Checks if two sets of grid points are congruent --/
def areCongruent (p q : Set GridPoint) : Prop :=
  ∃ dx dy : Int, ∀ point, point ∈ p → { x := point.x + dx, y := point.y + dy } ∈ q

/-- The main theorem stating that the shape can be partitioned into 4 congruent parts --/
theorem shape_can_be_partitioned_into_four_congruent_parts 
  (shape : GridShape) : 
  ∃ partition : GridPartition, 
    isValidPartition shape partition ∧
    ∀ p q, p ∈ partition.parts → q ∈ partition.parts → areCongruent p q :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_can_be_partitioned_into_four_congruent_parts_l620_62013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l620_62069

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- Define the theorem
theorem range_of_a : 
  ∃ S : Set ℝ, S = Set.Iic (-2) ∧ 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l620_62069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_reciprocal_real_negative_F_reciprocal_imag_negative_F_reciprocal_outside_unit_circle_l620_62060

/-- A complex number inside the unit circle with negative real and positive imaginary components -/
noncomputable def F : ℂ :=
  sorry

/-- F is inside the unit circle -/
axiom F_inside_unit_circle : Complex.abs F < 1

/-- F has negative real part -/
axiom F_real_negative : F.re < 0

/-- F has positive imaginary part -/
axiom F_imag_positive : F.im > 0

/-- The reciprocal of F -/
noncomputable def F_reciprocal : ℂ := F⁻¹

/-- Theorem: The reciprocal of F has negative real part -/
theorem F_reciprocal_real_negative : F_reciprocal.re < 0 := by
  sorry

/-- Theorem: The reciprocal of F has negative imaginary part -/
theorem F_reciprocal_imag_negative : F_reciprocal.im < 0 := by
  sorry

/-- Theorem: The reciprocal of F lies outside the unit circle -/
theorem F_reciprocal_outside_unit_circle : Complex.abs F_reciprocal > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_reciprocal_real_negative_F_reciprocal_imag_negative_F_reciprocal_outside_unit_circle_l620_62060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_sum_from_interests_l620_62043

/-- Given compound and simple interest for a 2-year period, find the principal sum -/
theorem principal_sum_from_interests (CI SI : ℚ) (h1 : CI = 11730) (h2 : SI = 10200) : ℚ :=
  let P := 34000
  let R := 15
  let t := 2
  have h_SI : SI = P * R * t / 100 := by sorry
  have h_CI : CI = P * (1 + R/100)^2 - P := by sorry
  P

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_sum_from_interests_l620_62043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_l620_62075

noncomputable def f (x m : ℝ) : ℝ :=
  if x < m then 2 * x^2 + 3 else 3 * x + 4

def is_continuous_at_m (m : ℝ) : Prop :=
  2 * m^2 + 3 = 3 * m + 4

theorem sum_of_continuity_points :
  ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ is_continuous_at_m m₁ ∧ is_continuous_at_m m₂ ∧ m₁ + m₂ = 3/2 := by
  sorry

#check sum_of_continuity_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_l620_62075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tray_height_is_fourth_root_l620_62023

/-- The height of a paper tray formed from a square sheet -/
noncomputable def paperTrayHeight (sideLength : ℝ) (cutDistance : ℝ) (cutAngle : ℝ) : ℝ :=
  let diagonalCutLength := cutDistance * Real.sqrt 2
  let heightToBase := (cutDistance + diagonalCutLength) / Real.sqrt 2
  heightToBase

theorem paper_tray_height_is_fourth_root (sideLength : ℝ) (cutDistance : ℝ) (cutAngle : ℝ) :
  sideLength = 120 →
  cutDistance = Real.sqrt 13 →
  cutAngle = π / 4 →
  paperTrayHeight sideLength cutDistance cutAngle = Real.sqrt (Real.sqrt 507) := by
  sorry

#check paper_tray_height_is_fourth_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tray_height_is_fourth_root_l620_62023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_point_distance_l620_62049

theorem right_triangle_point_distance (h d x : ℝ) (h_pos : h > 0) (d_pos : d > 0) :
  let triangle_ABC : Set (ℝ × ℝ) := {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^2 + p.2^2 ≤ h^2}
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2*d, 0)
  let C : ℝ × ℝ := (0, h)
  let M : ℝ × ℝ := (0, x)
  (∀ p ∈ triangle_ABC, Dist.dist p A + Dist.dist C p = Dist.dist B A) →
  x = (h - Real.sqrt (h^2 - 8*d^2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_point_distance_l620_62049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_plus_v_between_60_and_61_l620_62009

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def problem_conditions (u v : ℝ) : Prop :=
  v = 4 * (floor u) + 5 ∧
  v = 5 * (floor (u - 3)) + 9 ∧
  ¬(∃ n : ℤ, u = n)

theorem u_plus_v_between_60_and_61 :
  ∀ u v : ℝ, problem_conditions u v → 60 < u + v ∧ u + v < 61 := by
  sorry

#check u_plus_v_between_60_and_61

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_plus_v_between_60_and_61_l620_62009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_climbing_time_ratio_l620_62016

/-- The ratio of Tom's climbing time to Elizabeth's climbing time is 4:1 -/
theorem climbing_time_ratio : 
  (120 : ℚ) / 30 = 4 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_climbing_time_ratio_l620_62016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_is_zero_l620_62031

noncomputable def complex_sum : ℂ :=
  Complex.exp (5 * Real.pi * Complex.I / 40) +
  Complex.exp (15 * Real.pi * Complex.I / 40) +
  Complex.exp (25 * Real.pi * Complex.I / 40) +
  Complex.exp (35 * Real.pi * Complex.I / 40)

theorem complex_sum_arg_is_zero :
  0 ≤ Complex.arg complex_sum ∧ Complex.arg complex_sum < 2 * Real.pi →
  Complex.arg complex_sum = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_is_zero_l620_62031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotone_increasing_range_l620_62055

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_monotone_increasing_range :
  ∀ α : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → power_function α x₁ < power_function α x₂) ↔ 0 < α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotone_increasing_range_l620_62055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_p_range_of_a_when_p_or_q_and_not_p_and_q_l620_62064

-- Define the functions
noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := Real.log ((a * x) / (a - 2)) / Real.log (3 * a)
noncomputable def f₂ (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 4*x + 5*a)

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x y, x < y → x < 0 → y < 0 → f₁ a x > f₁ a y
def q (a : ℝ) : Prop := Set.range (f₂ a) = Set.Ici 0

-- Theorem statements
theorem range_of_a_when_p (a : ℝ) : p a ↔ a ∈ Set.Ioo (1/3) 2 := by sorry

theorem range_of_a_when_p_or_q_and_not_p_and_q (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ a ∈ Set.Iic (1/3) ∪ Set.Ioo (4/5) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_p_range_of_a_when_p_or_q_and_not_p_and_q_l620_62064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_leftover_raisins_l620_62095

/-- Represents the cost of items in an arbitrary unit -/
structure Cost where
  cream_puff : ℝ
  kofola_dl : ℝ
  yogurt_raisins_dkg : ℝ

/-- Represents Martin's purchase options and final choice -/
structure MartinsPurchase where
  option1 : Cost
  option2 : Cost
  option3 : Cost
  final_purchase : Cost

/-- Theorem stating that given Martin's purchase options and final choice, 
    he has enough money left for 60 grams of yogurt raisins -/
theorem martin_leftover_raisins (p : MartinsPurchase) : ℝ := by
  let option1 := p.option1
  let option2 := p.option2
  let option3 := p.option3
  let final := p.final_purchase
  have h1 : 3 * option1.cream_puff + 3 * option1.kofola_dl = 18 * option1.yogurt_raisins_dkg := sorry
  have h2 : 12 * option3.yogurt_raisins_dkg + 5 * option3.kofola_dl = 
            option3.cream_puff + 6 * option3.kofola_dl := sorry
  have h3 : final.cream_puff = option1.cream_puff ∧ 
            final.kofola_dl = option1.kofola_dl ∧ 
            final.yogurt_raisins_dkg = option1.yogurt_raisins_dkg := sorry
  exact 60

#check martin_leftover_raisins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_leftover_raisins_l620_62095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_range_l620_62077

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_range
  (a₁ : ℝ) (d : ℝ) (h₁ : a₁ > 0) (h₂ : d < 0)
  (h₃ : ∀ n : ℕ, n ≠ 8 → sum_arithmetic_sequence a₁ d n < sum_arithmetic_sequence a₁ d 8) :
  ∃ (l u : ℝ), l = -30 ∧ u = -18 ∧
  ∀ x : ℝ, (∃ (S₁₂ : ℝ), S₁₂ = sum_arithmetic_sequence a₁ d 12 ∧ x = S₁₂ / d) → l < x ∧ x < u :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_range_l620_62077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l620_62059

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 4 →
  e = -2*a - c →
  a + b*Complex.I + c + d*Complex.I + e + f*Complex.I = 3*Complex.I →
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l620_62059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_cylinder_volume_l620_62056

-- Define the constants and variables
noncomputable def π : ℝ := Real.pi
variable (h : ℝ) -- Height of both cylinders
variable (r₁ : ℝ) -- Radius of the first cylinder
variable (r₂ : ℝ) -- Radius of the second cylinder

-- Define the volume of a cylinder
noncomputable def cylinderVolume (r : ℝ) (h : ℝ) : ℝ := π * r^2 * h

-- State the theorem
theorem second_cylinder_volume (h : ℝ) (r₁ : ℝ) (r₂ : ℝ) : 
  r₂ = 3 * r₁ →  -- Radii are in the ratio 1:3
  cylinderVolume r₁ h = 40 → -- Volume of the first cylinder is 40 cc
  cylinderVolume r₂ h = 360 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_cylinder_volume_l620_62056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_numbers_l620_62084

theorem mean_of_remaining_numbers
  (numbers : Fin 5 → ℝ)
  (mean_condition : (Finset.sum Finset.univ numbers) / 5 = 100)
  (max_condition : ∃ i, numbers i = 120 ∧ ∀ j, numbers j ≤ numbers i) :
  ((Finset.sum Finset.univ numbers - 120) / 4) = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_numbers_l620_62084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_specific_l620_62079

/-- A right triangle ABC with angle A = 45° and AC = 12 -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle_at_C : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  angle_A_45 : Real.cos (45 * π / 180) = (B.1 - C.1) / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 12

/-- The radius of the incircle of a right triangle -/
noncomputable def incircle_radius (t : RightTriangle) : ℝ :=
  let s := (Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) +
            Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) +
            Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)) / 2
  let area := Real.sqrt (s * (s - Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)) *
                             (s - Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)) *
                             (s - Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)))
  area / s

/-- The radius of the incircle of a right triangle with angle A = 45° and AC = 12 is 6 - 3√2 -/
theorem incircle_radius_specific (t : RightTriangle) : incircle_radius t = 6 - 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_specific_l620_62079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_6374_to_hundredth_l620_62033

-- Define a function to round a number to the nearest hundredth
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

-- Theorem statement
theorem round_24_6374_to_hundredth :
  round_to_hundredth 24.6374 = 24.64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_6374_to_hundredth_l620_62033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_trips_equal_time_l620_62018

/-- Represents a trip with distance and speed -/
structure Trip where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a trip -/
noncomputable def time (t : Trip) : ℝ := t.distance / t.speed

theorem jenny_trips_equal_time (first_trip second_trip : Trip) 
  (h1 : first_trip.distance = 60)
  (h2 : second_trip.distance = 240)
  (h3 : second_trip.speed = 4 * first_trip.speed) :
  time second_trip = time first_trip := by
  sorry

#check jenny_trips_equal_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_trips_equal_time_l620_62018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_tank_l620_62048

/-- The volume of a cylindrical tank -/
noncomputable def cylindrical_tank_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

/-- Theorem: The volume of a cylindrical tank with diameter 20 feet and depth 10 feet is 1000π cubic feet -/
theorem volume_of_specific_tank :
  cylindrical_tank_volume 20 10 = 1000 * Real.pi := by
  -- Unfold the definition of cylindrical_tank_volume
  unfold cylindrical_tank_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_tank_l620_62048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_functions_l620_62044

-- Define the domain of the functions
def Domain : Set ℝ := {x : ℝ | x < 0 ∨ x > 0}

-- Define the functions
def f (x : Domain) : ℝ := (x : ℝ)^2

noncomputable def g (x : Domain) : ℝ := Real.sqrt (abs (x : ℝ))

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 2) = (a (n + 1))^2

-- State the theorem
theorem geometric_sequence_preserving_functions
  (a : ℕ → Domain) (h : is_geometric_sequence (λ n => (a n : ℝ))) :
  (is_geometric_sequence (λ n => f (a n))) ∧
  (is_geometric_sequence (λ n => g (a n))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_functions_l620_62044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l620_62074

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_c_sq : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The distance from the right focus to the right directrix -/
noncomputable def focus_to_directrix (e : Ellipse) : ℝ := e.a^2 / e.c - e.c

/-- Theorem about the standard equation of the ellipse and intersection points -/
theorem ellipse_properties (e : Ellipse)
  (h_ecc : eccentricity e = Real.sqrt 3 / 2)
  (h_dist : focus_to_directrix e = Real.sqrt 3 / 3) :
  (∃ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (∃ d : ℝ, d = 8 / 5 ∧
    d = Real.sqrt 2 * Real.sqrt ((8 * Real.sqrt 3 / 5)^2 - 4 * (8 / 5))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l620_62074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l620_62012

-- Define the quadratic function f
def f (x : ℝ) : ℝ := -x^2 + x + 6

-- Define the function g
def g (x : ℝ) : ℝ := x + 5 - f x

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) 3) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ 4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 4) ∧
  (∀ m : ℝ, (∀ x ≤ (-3/4 : ℝ), g (x/m) - g (x-1) ≤ 4*(m^2 * g x + g m)) ↔
            m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l620_62012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angles_count_l620_62046

/-- The set of basic angles provided in the problem -/
def basic_angles : Finset ℕ := {30, 45, 60, 90}

/-- A function that generates all possible angles from the basic angles -/
def generate_angles (angles : Finset ℕ) : Finset ℕ :=
  sorry

/-- The main theorem to prove -/
theorem unique_angles_count : 
  let all_angles := generate_angles basic_angles
  (all_angles.filter (λ a => 0 < a ∧ a < 176)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angles_count_l620_62046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fractions_with_denominator_2007_l620_62080

theorem count_fractions_with_denominator_2007 : 
  let n : ℕ := 2007
  let is_proper (a : ℕ) : Prop := a < n
  let is_coprime (a : ℕ) : Prop := Nat.Coprime a n
  (Finset.filter (λ a => is_proper a ∧ is_coprime a) (Finset.range n)).card = 1332 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fractions_with_denominator_2007_l620_62080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_coefficient_sum_l620_62014

noncomputable def ellipse_x (t : ℝ) : ℝ := (3 * (Real.sin t + 2)) / (3 - Real.cos t)
noncomputable def ellipse_y (t : ℝ) : ℝ := (4 * (Real.cos t - 6)) / (3 - Real.cos t)

def are_integers (A B C D E F : ℤ) : Prop := True

def gcd_condition (A B C D E F : ℤ) : Prop := 
  Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) 
    (Nat.gcd (Int.natAbs D) (Nat.gcd (Int.natAbs E) (Int.natAbs F))))) = 1

theorem ellipse_coefficient_sum :
  ∃ (A B C D E F : ℤ),
    (∀ (x y : ℝ), (∃ t, x = ellipse_x t ∧ y = ellipse_y t) ↔ 
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧
    are_integers A B C D E F ∧
    gcd_condition A B C D E F ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 733) := by
  sorry

#check ellipse_coefficient_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_coefficient_sum_l620_62014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_diophantine_equation_l620_62093

theorem unique_solution_for_diophantine_equation :
  ∃! (x y z t : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    2^y + 2^z * 5^t - 5^x = 1 ∧ x = 2 ∧ y = 4 ∧ z = 1 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_diophantine_equation_l620_62093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_in_interval_l620_62087

noncomputable def rootCount (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  Nat.card {x : ℝ | a ≤ x ∧ x ≤ b ∧ f x = 0}

theorem min_roots_in_interval
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (3 - x))
  (h2 : ∀ x, f (8 + x) = f (8 - x))
  (h3 : f 0 = 0) :
  rootCount f (-1500) 1500 ≥ 601 := by
  sorry

#check min_roots_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_in_interval_l620_62087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_difference_l620_62072

/-- Parabola E: y^2 = 4x -/
def parabola_E (x y : ℝ) : Prop := y^2 = 4*x

/-- Circle C: x^2 + y^2 - 2ax + a^2 - 4 = 0 with a = 2 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The focus of the parabola E -/
def focus_F : ℝ × ℝ := (1, 0)

/-- A line l that intersects parabola E at points A and B, and is tangent to circle C -/
structure Line_l where
  k : ℝ  -- slope
  b : ℝ  -- y-intercept
  tangent_to_C : 4 * k * b + b^2 = 4  -- condition for tangency to circle C

/-- Points A and B where line l intersects parabola E -/
noncomputable def intersection_points (l : Line_l) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (2 * l.k * l.b - 4 + Real.sqrt (4 * l.b^2)) / (2 * l.k^2)
  let x2 := (2 * l.k * l.b - 4 - Real.sqrt (4 * l.b^2)) / (2 * l.k^2)
  ((x1, l.k * x1 + l.b), (x2, l.k * x2 + l.b))

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: |FA| + |FB| - |AB| is constant and equal to 2 -/
theorem constant_difference (l : Line_l) : 
  let (A, B) := intersection_points l
  distance focus_F A + distance focus_F B - distance A B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_difference_l620_62072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l620_62082

def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x - a|

theorem problem_statement (a : ℝ) :
  (∀ x, f a x > -4) →
  a ∈ Set.Ioo (-1) 7 ∧
  ((∀ t ∈ Set.Ioo 0 1, ∀ x, f a x ≤ 1/t + 9/(1-t)) →
   a ∈ Set.Icc (-13) 19) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l620_62082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l620_62097

noncomputable def sequence_a (n : ℕ) : ℝ := n + 2

noncomputable def sequence_b (n : ℕ) : ℝ := 2 * n + 3

noncomputable def sequence_c (n : ℕ) : ℝ := 1 / (4 * (2 * n - 1)) - 1 / (4 * (2 * n + 1))

noncomputable def sum_s (n : ℕ) : ℝ := 1/2 * n^2 + 5/2 * n

noncomputable def sum_t (n : ℕ) : ℝ := (1 - 1 / (2 * n + 1)) / 4

theorem sequence_properties :
  (∀ n : ℕ, sum_s n = (1/2 * n^2 + 5/2 * n)) ∧
  (∀ n : ℕ, sequence_b n + sequence_b (n + 2) = 2 * sequence_b (n + 1)) ∧
  (sequence_b 4 = 11) ∧
  (sequence_b 1 + sequence_b 2 + sequence_b 3 + sequence_b 4 + sequence_b 5 = 45) →
  (∀ n : ℕ, sequence_a n = n + 2) ∧
  (∀ n : ℕ, sequence_b n = 2 * n + 3) ∧
  (∀ k : ℕ, k ≤ 8 → ∀ n : ℕ, sum_t n > k / 54) ∧
  (∀ k : ℕ, k > 8 → ∃ n : ℕ, sum_t n ≤ k / 54) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l620_62097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l620_62035

structure Intersection where
  line : ℝ → ℝ → ℝ → Prop
  circle : ℝ → ℝ → Prop
  intersects : Prop
  chord_length : ℝ

def valid_a (a : ℝ) (i : Intersection) : Prop :=
  i.line = (λ x y a' ↦ x - y + a' = 0) ∧
  i.circle = (λ x y ↦ x^2 + y^2 + 2*x - 4*y + 2 = 0) ∧
  i.intersects ∧
  i.chord_length = 2 ∧
  (a = 1 ∨ a = 5)

theorem intersection_theorem :
  ∀ i : Intersection, ∃ a : ℝ, valid_a a i := by
  sorry

#check intersection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l620_62035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_characterization_l620_62092

-- Define the set of polynomials over real numbers
def MyPolynomial := ℝ → ℝ

-- Define the property that P satisfies the given equation
def SatisfiesEquation (P : MyPolynomial) : Prop :=
  ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

-- Define the set of polynomials of the form (x+1)^n
def PowerPolynomial (n : ℕ) : MyPolynomial :=
  λ x ↦ (x + 1)^n

theorem polynomial_equation_characterization :
  ∀ P : MyPolynomial, (∀ x, P x ≠ 0) →
    (SatisfiesEquation P ↔ ∃ n : ℕ, n > 0 ∧ P = PowerPolynomial n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_characterization_l620_62092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l620_62039

theorem angle_between_vectors (a b : ℝ × ℝ) (θ : ℝ) : 
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 4 →
  Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 3 →
  (a.1 + 2 * b.1) * (a.1 - b.1) + (a.2 + 2 * b.2) * (a.2 - b.2) = 4 →
  θ = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (4 * 3)) →
  θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l620_62039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_theorem_l620_62002

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the bag after each draw -/
structure BagState where
  white : ℕ
  black : ℕ

/-- Represents the result of a single draw -/
inductive DrawResult
| White
| Black
| TwoBlacks
| FiveDraws

/-- Represents the ball drawing process -/
noncomputable def drawProcess (initialState : BagState) (maxDraws : ℕ) : List DrawResult → ℝ :=
  sorry

/-- The probability of exactly drawing two black balls in the 4th draw -/
def probTwoBlacksFourthDraw : ℚ :=
  129 / 1000

/-- The probability distribution of X (number of draws when drawing stops) -/
def probDistributionX : ℕ → ℚ
| 2 => 1 / 10
| 3 => 13 / 100
| 4 => 129 / 1000
| 5 => 641 / 1000
| _ => 0

/-- Helper function to convert the result of drawProcess to a probability -/
noncomputable def drawProcessProb (initialState : BagState) (maxDraws : ℕ) (n : ℕ) : ℝ :=
  sorry

theorem ball_drawing_theorem (initialState : BagState) :
  initialState.white = 2 ∧ initialState.black = 2 →
  (drawProcess initialState 5 [DrawResult.Black, DrawResult.White, DrawResult.White, DrawResult.Black] = probTwoBlacksFourthDraw) ∧
  (∀ n, (probDistributionX n : ℝ) = drawProcessProb initialState 5 n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_theorem_l620_62002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colorings_count_l620_62070

def valid_colorings (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n - 3)) (λ k => n - k - 3)

theorem colorings_count :
  valid_colorings 50 = 1128 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colorings_count_l620_62070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l620_62030

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 3
def b : ℝ := 1
noncomputable def e : ℝ := Real.sqrt 6 / 3

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line passing through (0,2) with slope k
def line (k x : ℝ) : ℝ := k * x + 2

-- Define the intersection points of the line and the ellipse
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | is_on_ellipse x y ∧ y = line k x}

-- Define the condition for ∠AOB to be acute
def angle_AOB_acute (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  x₁ * x₂ + y₁ * y₂ > 0

-- State the theorem
theorem ellipse_and_line_intersection :
  (a > b) ∧ (b > 0) ∧ (e = Real.sqrt 6 / 3) ∧
  (∀ (x y : ℝ), is_on_ellipse x y → 
    (Real.sqrt ((x - Real.sqrt 2)^2 + y^2) + Real.sqrt ((x + Real.sqrt 2)^2 + y^2) = 2 * Real.sqrt 3)) →
  (∀ (k : ℝ), (intersection_points k).Nonempty ∧ 
    (∀ A B, A ∈ intersection_points k → B ∈ intersection_points k → A ≠ B → angle_AOB_acute A B) →
    (k ∈ Set.Ioo (-Real.sqrt 39 / 3) (-1) ∪ Set.Ioo 1 (Real.sqrt 39 / 3))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l620_62030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_reciprocal_lower_bound_l620_62025

/-- Given a > 1, m is the root of a^x + x - 4 = 0, and n is the root of log_a(x) + x - 4 = 0 -/
theorem root_sum_reciprocal_lower_bound (a m n : ℝ) (ha : a > 1) 
  (hm : a^m + m - 4 = 0) (hn : (Real.log n) / (Real.log a) + n - 4 = 0) : 
  1/m + 1/n ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_reciprocal_lower_bound_l620_62025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_product_l620_62011

def p (x : ℝ) : ℝ := x^3 + x^2 + 3
def q (x : ℝ) : ℝ := 2*x^4 + x^3 + 7

theorem constant_term_product : 
  (p 0) * (q 0) = 21 := by
  simp [p, q]
  norm_num

#eval (p 0) * (q 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_product_l620_62011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_peter_equality_l620_62019

theorem ivan_peter_equality (total_population : ℕ) 
  (h1 : (total_population / 10 : ℚ) = ((total_population : ℚ) * (1 : ℚ) / 10))
  (h2 : (total_population / 20 : ℚ) = ((total_population : ℚ) * (1 : ℚ) / 20)) :
  ((total_population : ℚ) * (1 : ℚ) / 200) = ((total_population : ℚ) * (1 : ℚ) / 200) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_peter_equality_l620_62019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l620_62037

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧ 
  (∃ x, (x - m + 1) * (x - m - 1) ≤ 0 ∧ |x - 3| > 2)) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l620_62037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_encounters_l620_62022

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the state of the race -/
structure RaceState where
  petya : Runner
  misha : Runner
  track_length : ℝ
  misha_distance : ℝ  -- distance Misha has run in current direction

/-- Determines if Misha can change direction -/
def can_change_direction (state : RaceState) : Prop :=
  state.misha_distance ≥ state.track_length / 2

/-- Counts the number of encounters between Petya and Misha -/
def count_encounters (initial_state : RaceState) : ℕ → Prop :=
  sorry

/-- Main theorem to prove -/
theorem three_encounters 
  (initial_state : RaceState)
  (h1 : initial_state.petya.direction = true)
  (h2 : initial_state.misha.direction = true)
  (h3 : initial_state.misha.speed = 1.02 * initial_state.petya.speed)
  (h4 : initial_state.misha_distance = 0) :
  ∃ (n : ℕ), count_encounters initial_state n ∧ n = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_encounters_l620_62022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l620_62029

/-- Square with side length 6 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

/-- Point E -/
def E : ℝ × ℝ := (0, 6)

/-- Point G -/
def G : ℝ × ℝ := (0, 0)

/-- Point H -/
def H : ℝ × ℝ := (6, 0)

/-- Midpoint N of GH -/
def N : ℝ × ℝ := (3, 0)

/-- Circle centered at N with radius 4 -/
def CircleN (p : ℝ × ℝ) : Prop :=
  (p.1 - 3)^2 + p.2^2 = 16

/-- Circle centered at E with radius 3 -/
def CircleE (p : ℝ × ℝ) : Prop :=
  p.1^2 + (p.2 - 6)^2 = 9

/-- Point Q is an intersection of CircleN and CircleE, different from G -/
noncomputable def Q : ℝ × ℝ :=
  (25 / 6, -1 / 3)

/-- Distance from a point to a horizontal line -/
def distToHorizontalLine (p : ℝ × ℝ) (y : ℝ) : ℝ :=
  |p.2 - y|

theorem distance_Q_to_EH :
  distToHorizontalLine Q 6 = 19 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l620_62029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l620_62006

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x - 5) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x > 5 ∨ x < -1

-- Define the monotonicity of t = x² - 4x - 5
def t_increasing (x : ℝ) : Prop := x > 5 → ∀ y, y > x → (y^2 - 4*y - 5) > (x^2 - 4*x - 5)
def t_decreasing (x : ℝ) : Prop := x < -1 → ∀ y, y < x → (y^2 - 4*y - 5) < (x^2 - 4*x - 5)

-- Define the monotonicity of y = log₁/₂(t)
def log_half_decreasing (t : ℝ) : Prop := ∀ s, s > t → Real.log s / Real.log (1/2) < Real.log t / Real.log (1/2)

-- State the theorem
theorem f_decreasing_interval :
  (∀ x, domain x → t_increasing x) →
  (∀ x, domain x → t_decreasing x) →
  (∀ t, t > 0 → log_half_decreasing t) →
  ∀ x y, x > 5 → y > x → f y < f x :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l620_62006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bhanu_petrol_percentage_l620_62065

/-- Represents Bhanu's financial situation --/
structure BhanuFinances where
  income : ℚ
  petrolExpense : ℚ
  rentExpense : ℚ

/-- Calculates the percentage of income spent on petrol --/
noncomputable def petrolPercentage (b : BhanuFinances) : ℚ :=
  (b.petrolExpense / b.income) * 100

/-- Theorem stating that Bhanu spends 30% of his income on petrol --/
theorem bhanu_petrol_percentage :
  ∀ (b : BhanuFinances),
    b.petrolExpense = 300 ∧
    b.rentExpense = 70 ∧
    b.rentExpense = (1/10) * (b.income - b.petrolExpense) →
    petrolPercentage b = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bhanu_petrol_percentage_l620_62065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_values_sin_A_value_l620_62054

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sqrt 3 / 2 * Real.sin x - 1 / 2 * Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 / 2 * Real.sin x - 1 / 2 * Real.cos x)

-- Define the function f as the dot product of m and n
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Theorem for the maximum value of f(x)
theorem f_max_values (k : ℤ) :
  IsLocalMax f (k * Real.pi - Real.pi / 12) := by
  sorry

-- Theorem for the value of sin A in the triangle
theorem sin_A_value (A B C : ℝ) :
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  0 < A ∧ A < Real.pi / 2 →  -- A is acute
  0 < B ∧ B < Real.pi / 2 →  -- B is acute
  0 < C ∧ C < Real.pi / 2 →  -- C is acute
  Real.cos B = 3 / 5 →
  f C = -1 / 4 →
  Real.sin A = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_values_sin_A_value_l620_62054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_beads_removed_l620_62067

def initial_white : ℕ := 105
def initial_black : ℕ := 210
def initial_blue : ℕ := 60

def first_round_black_fraction : ℚ := 2/7
def first_round_white_fraction : ℚ := 3/7
def first_round_blue_fraction : ℚ := 1/4

def second_round_add_white : ℕ := 45
def second_round_add_black : ℕ := 80

def second_round_black_fraction : ℚ := 3/8
def second_round_white_fraction : ℚ := 1/3

theorem total_beads_removed : ∃ (
  first_round_black first_round_white first_round_blue
  remaining_black remaining_white
  second_round_black second_round_white : ℕ),
  first_round_black = (first_round_black_fraction * initial_black).floor ∧
  first_round_white = (first_round_white_fraction * initial_white).floor ∧
  first_round_blue = (first_round_blue_fraction * initial_blue).floor ∧
  remaining_black = initial_black - first_round_black ∧
  remaining_white = initial_white - first_round_white ∧
  second_round_black = (second_round_black_fraction * (remaining_black + second_round_add_black)).floor ∧
  second_round_white = (second_round_white_fraction * second_round_add_white).floor ∧
  first_round_black + first_round_white + first_round_blue + second_round_black + second_round_white = 221 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_beads_removed_l620_62067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_duration_is_three_hours_l620_62099

/-- Represents the hike described in the problem -/
structure Hike where
  total_distance : ℝ
  initial_water : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_6_miles_rate : ℝ

/-- Calculates the duration of the hike -/
noncomputable def hike_duration (h : Hike) : ℝ :=
  (h.initial_water - h.remaining_water - 
   (h.first_6_miles_rate * 6 + h.last_mile_consumption)) / h.leak_rate

/-- Theorem stating that the hike duration is 3 hours -/
theorem hike_duration_is_three_hours (h : Hike) 
  (h_total_distance : h.total_distance = 7)
  (h_initial_water : h.initial_water = 11)
  (h_remaining_water : h.remaining_water = 2)
  (h_leak_rate : h.leak_rate = 1)
  (h_last_mile_consumption : h.last_mile_consumption = 3)
  (h_first_6_miles_rate : h.first_6_miles_rate = 0.5) :
  hike_duration h = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_duration_is_three_hours_l620_62099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_one_plus_sqrt_one_minus_x_squared_l620_62071

theorem integral_one_plus_sqrt_one_minus_x_squared :
  ∫ x in Set.Icc 0 1, (1 + Real.sqrt (1 - x^2)) = 1 + π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_one_plus_sqrt_one_minus_x_squared_l620_62071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_age_order_l620_62098

-- Define the friends
inductive Friend : Type
| Amy : Friend
| Bill : Friend
| Celine : Friend
| David : Friend
deriving BEq, Repr

-- Define the age ordering
def AgeOrder : List Friend := [Friend.Amy, Friend.David, Friend.Celine, Friend.Bill]

-- Define the statements
def StatementI (order : List Friend) : Prop := order.head? = some Friend.Bill
def StatementII (order : List Friend) : Prop := order.head? ≠ some Friend.Amy
def StatementIII (order : List Friend) : Prop := order.reverse.head? ≠ some Friend.Celine
def StatementIV (order : List Friend) : Prop := 
  (order.indexOf Friend.David < order.indexOf Friend.Amy) ∧ 
  (order.indexOf Friend.David > order.indexOf Friend.Celine)

-- Main theorem
theorem correct_age_order : 
  (∀ (f1 f2 : Friend), f1 ≠ f2 → AgeOrder.indexOf f1 ≠ AgeOrder.indexOf f2) ∧ 
  (StatementI AgeOrder ∨ StatementII AgeOrder ∨ StatementIII AgeOrder ∨ StatementIV AgeOrder) ∧
  (¬(StatementI AgeOrder ∧ StatementII AgeOrder) ∧ 
   ¬(StatementI AgeOrder ∧ StatementIII AgeOrder) ∧ 
   ¬(StatementI AgeOrder ∧ StatementIV AgeOrder) ∧ 
   ¬(StatementII AgeOrder ∧ StatementIII AgeOrder) ∧ 
   ¬(StatementII AgeOrder ∧ StatementIV AgeOrder) ∧ 
   ¬(StatementIII AgeOrder ∧ StatementIV AgeOrder)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_age_order_l620_62098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l620_62063

/-- A polynomial of degree 4 with real coefficients -/
def polynomial (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- Theorem stating that if a polynomial with real coefficients has 3i and 1+i as roots, 
    then the sum of its coefficients is 9 -/
theorem polynomial_root_sum (a b c d : ℝ) : 
  let g := polynomial a b c d
  g (3*I) = 0 ∧ g (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l620_62063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_period_calculation_l620_62061

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time / 100

theorem interest_period_calculation (principal : ℝ) (principal_positive : principal > 0) :
  let rate : ℝ := 2.5
  let interest : ℝ := principal / 5
  ∃ time : ℝ, simple_interest principal rate time = interest ∧ time = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_period_calculation_l620_62061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_accurate_l620_62027

/-- Represents a method for testing relationships between categorical features -/
inductive TestMethod
| ThreeDimensionalBarChart
| TwoDimensionalBarChart
| ContourBarChart
| IndependenceTest

/-- Represents a categorical feature -/
structure CategoricalFeature where
  label : String

/-- Represents a two-way classified contingency table -/
structure ContingencyTable where
  feature1 : CategoricalFeature
  feature2 : CategoricalFeature
  data : Matrix (Fin 2) (Fin 2) ℕ

/-- Represents the accuracy of a test method -/
noncomputable def accuracy : TestMethod → ℝ := sorry

/-- The set of commonly used test methods -/
def commonlyUsedMethods : Set TestMethod :=
  {TestMethod.ThreeDimensionalBarChart, TestMethod.TwoDimensionalBarChart,
   TestMethod.ContourBarChart, TestMethod.IndependenceTest}

/-- Theorem stating that the independence test is the most accurate method
    among commonly used ones for testing relationships in a contingency table -/
theorem independence_test_most_accurate
  (table : ContingencyTable)
  (h_common : TestMethod.IndependenceTest ∈ commonlyUsedMethods) :
  ∀ m ∈ commonlyUsedMethods,
    accuracy TestMethod.IndependenceTest ≥ accuracy m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_accurate_l620_62027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l620_62017

/-- A hyperbola with foci on the x-axis and asymptote equations y = ± √3 x has eccentricity 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = Real.sqrt 3) : 
  Real.sqrt ((a^2 + b^2) / a^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l620_62017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_number_l620_62047

theorem find_number (A B : ℕ) (h1 : Nat.gcd A B = 16) (h2 : Nat.lcm A B = 312) (h3 : A = 24) : B = 208 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_number_l620_62047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_overlap_theorem_l620_62052

/-- Represents a rug on the floor -/
structure Rug where
  area : ℝ
  position : ℝ × ℝ

/-- Represents the floor and the rugs on it -/
structure Floor where
  area : ℝ
  rugs : List Rug

/-- Calculate the overlap between two rugs -/
noncomputable def rugOverlap (r1 r2 : Rug) : ℝ :=
  sorry

theorem rug_overlap_theorem (floor : Floor) 
  (h1 : floor.area = 3)
  (h2 : floor.rugs.length = 5)
  (h3 : ∀ r, r ∈ floor.rugs → r.area = 1) :
  ∃ r1 r2, r1 ∈ floor.rugs ∧ r2 ∈ floor.rugs ∧ r1 ≠ r2 ∧ rugOverlap r1 r2 ≥ 0.2 :=
by
  sorry

#check rug_overlap_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_overlap_theorem_l620_62052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_travel_time_l620_62050

/-- Represents the time taken by a squirrel to travel a given distance at a constant speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Converts hours to minutes -/
noncomputable def hours_to_minutes (hours : ℝ) : ℝ :=
  hours * 60

theorem squirrel_travel_time :
  let distance : ℝ := 2
  let speed : ℝ := 5
  hours_to_minutes (travel_time distance speed) = 24 := by
  -- Unfold the definitions
  unfold hours_to_minutes travel_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_travel_time_l620_62050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_problem_l620_62040

/-- Represents the composition of a mixture --/
structure Mixture where
  milk : ℚ
  water : ℚ
  juice : ℚ

/-- The initial mixture --/
def initial_mixture : Mixture :=
  { milk := 70, water := 30, juice := 50 }

/-- The amount removed from each component --/
def removed_amount : ℚ := 10

/-- The new mixture after removal --/
def new_mixture : Mixture :=
  { milk := initial_mixture.milk - removed_amount,
    water := initial_mixture.water - removed_amount,
    juice := initial_mixture.juice - removed_amount }

/-- The removed portion --/
def removed_portion : Mixture :=
  { milk := removed_amount,
    water := removed_amount,
    juice := removed_amount }

/-- Calculates the total volume of a mixture --/
def total_volume (m : Mixture) : ℚ :=
  m.milk + m.water + m.juice

/-- Calculates the ratio of components in a mixture --/
noncomputable def ratio (m : Mixture) : Mixture :=
  let total := total_volume m
  { milk := m.milk / total,
    water := m.water / total,
    juice := m.juice / total }

theorem mixture_problem :
  (total_volume new_mixture = 120) ∧
  (ratio removed_portion = ratio initial_mixture) ∧
  (∃ (x : ℚ), ratio { milk := removed_portion.milk + x,
                      water := removed_portion.water + x,
                      juice := removed_portion.juice + x } =
               { milk := 1/3, water := 1/3, juice := 1/3 }) := by
  sorry

#eval total_volume new_mixture

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_problem_l620_62040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_is_16_l620_62081

-- Define the set of colors
inductive Color : Type
| Red : Color
| Blue : Color

-- Define a coloring function
def Coloring := ℤ × ℤ → Color

-- Define the rotation functions
def φ₁ (p : ℤ × ℤ) : ℤ × ℤ := (-1 - p.2, p.1 + 1)
def φ₂ (p : ℤ × ℤ) : ℤ × ℤ := (1 - p.2, p.1 - 1)

-- Define the property that a coloring satisfies the condition
def SatisfiesCondition (c : Coloring) : Prop :=
  ∀ (a b : ℤ), c (a, b) = c (φ₁ (a, b)) ∧ c (a, b) = c (φ₂ (a, b))

-- Define the set of valid colorings
def ValidColorings : Set Coloring :=
  { c | SatisfiesCondition c }

-- Axiom: There are exactly 16 valid colorings
axiom valid_colorings_count : ∃ (f : Fin 16 → ValidColorings), Function.Bijective f

-- The main theorem
theorem valid_colorings_count_is_16 : ∃ (n : ℕ), n = 16 ∧ ∃ (f : Fin n → ValidColorings), Function.Bijective f := by
  exists 16
  constructor
  · rfl
  · exact valid_colorings_count


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_is_16_l620_62081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_not_less_than_twelve_l620_62045

/-- Represents the number of distinct arrangements of n objects from a set of 12 objects with an unknown color distribution -/
def distinctArrangements (n : ℕ) : ℕ := sorry

/-- The number of objects in the set -/
def totalObjects : ℕ := 12

/-- Theorem stating that the number of distinct arrangements of 11 objects is not less than the number of distinct arrangements of 12 objects -/
theorem eleven_not_less_than_twelve :
  distinctArrangements 11 ≥ distinctArrangements totalObjects :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_not_less_than_twelve_l620_62045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_time_theorem_l620_62024

/-- Represents the time spent on each pet activity in hours -/
structure PetTime where
  dog_walk_play : ℚ
  dog_feed : ℚ
  dog_groom : ℚ
  cat_play : ℚ
  cat_feed : ℚ
  hamster_care : ℚ
  parrot_clean : ℚ
  parrot_play : ℚ
  parrot_feed : ℚ

/-- Calculates the average daily time spent on pets in minutes -/
noncomputable def average_daily_pet_time (pt : PetTime) : ℚ :=
  let daily_time := 2 * pt.dog_walk_play + pt.dog_feed + 2 * pt.cat_play + pt.cat_feed +
                    pt.hamster_care + pt.parrot_clean + pt.parrot_play + pt.parrot_feed
  let weekly_groom_time := 3 * pt.dog_groom
  let total_weekly_time := 7 * daily_time + weekly_groom_time
  (total_weekly_time / 7) * 60

/-- The main theorem stating the average daily time spent on pets -/
theorem pet_time_theorem (pt : PetTime)
  (h1 : pt.dog_walk_play = 1/2)
  (h2 : pt.dog_feed = 1/5)
  (h3 : pt.dog_groom = 1/10)
  (h4 : pt.cat_play = 1/4)
  (h5 : pt.cat_feed = 1/10)
  (h6 : pt.hamster_care = 1/12)
  (h7 : pt.parrot_clean = 3/20)
  (h8 : pt.parrot_play = 1/9)
  (h9 : pt.parrot_feed = 1/6) :
  ∃ (ε : ℚ), ε > 0 ∧ |average_daily_pet_time pt - 141.24| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_time_theorem_l620_62024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_and_range_l620_62001

-- Define the function h
noncomputable def h : ℝ → ℝ := sorry

-- Define the properties of h
axiom h_domain : ∀ x ∈ Set.Icc 1 4, h x ∈ Set.Icc 0 1

-- Define the function k
noncomputable def k (x : ℝ) : ℝ := (1 - h (x - 1))^2

-- Theorem statement
theorem k_domain_and_range :
  (∀ x, x ∈ Set.Icc 2 5 ↔ k x ∈ Set.Icc 0 1) ∧
  (∀ y ∈ Set.Icc 0 1, ∃ x ∈ Set.Icc 2 5, k x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_and_range_l620_62001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l620_62076

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l620_62076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_quadratic_lines_l620_62007

/-- Represents the angle between two lines given by a quadratic equation. -/
noncomputable def angle_between_lines (eq : ℝ → ℝ → ℝ) : ℝ :=
  sorry

/-- Given the equation x^2 + (b+2)xy + by^2 = 0, where b ∈ ℝ, representing two lines,
    the angle θ between these lines satisfies: θ ∈ [arctan(2√5/5), π/2]. -/
theorem angle_range_for_quadratic_lines (b : ℝ) :
  let θ := angle_between_lines (fun x y => x^2 + (b+2)*x*y + b*y^2)
  θ ∈ Set.Icc (Real.arctan ((2 * Real.sqrt 5) / 5)) (π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_quadratic_lines_l620_62007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_places_of_fraction_l620_62053

theorem decimal_places_of_fraction : ∃ (n : ℕ), n = 5 ∧ 
  (∃ (a b : ℚ), (4^6 : ℚ) / (7^4 * 5^6 : ℚ) = a + b ∧ 
   0 ≤ b ∧ b < 10^(-n : ℤ) ∧ (a * 10^n).floor ≠ ((a * 10^n + 1).floor)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_places_of_fraction_l620_62053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_dihedral_angles_gt_360_l620_62004

/-- A tetrahedron is a three-dimensional geometric object with four triangular faces. -/
structure Tetrahedron where
  mk :: -- Empty structure for this problem

/-- The dihedral angle is the angle between two intersecting planes. -/
def dihedral_angle (t : Tetrahedron) : ℝ → ℝ := 
  fun _ => 0 -- Placeholder function

/-- The sum of all dihedral angles in a tetrahedron -/
def sum_dihedral_angles (t : Tetrahedron) : ℝ :=
  0 -- Placeholder value

/-- Theorem: The sum of the dihedral angles of a tetrahedron is greater than 360° -/
theorem sum_dihedral_angles_gt_360 (t : Tetrahedron) :
  sum_dihedral_angles t > 360 := by
  sorry

#check sum_dihedral_angles_gt_360

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_dihedral_angles_gt_360_l620_62004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_range_l620_62032

-- Define the expression as noncomputable due to its dependence on real numbers
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (7 - x) / (2 * x - 6)

-- State the theorem
theorem meaningful_range (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ (x ≤ 7 ∧ x ≠ 3) :=
by
  -- The proof would go here, but we'll use sorry as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_range_l620_62032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_above_perfect_square_and_cube_l620_62028

-- Define perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Define perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define the set of numbers that are both perfect squares and perfect cubes
def perfect_square_and_cube (n : ℕ) : Prop := is_perfect_square n ∧ is_perfect_cube n

-- Assert that there are exactly 6 such numbers
axiom exactly_six : ∃! (s : Finset ℕ), (∀ n ∈ s, perfect_square_and_cube n) ∧ s.card = 6

theorem smallest_number_above_perfect_square_and_cube : ℕ := by
  -- The theorem to prove
  have h : ∃ m : ℕ, (∀ n : ℕ, perfect_square_and_cube n → n < m) ∧ 
                    (∀ k : ℕ, (∀ n : ℕ, perfect_square_and_cube n → n < k) → m ≤ k) := by
    sorry

  -- The smallest such number is 117649
  exact 117649

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_above_perfect_square_and_cube_l620_62028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l620_62057

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a spherical marble -/
structure Marble where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (m : Marble) : ℝ := (4/3) * Real.pi * m.radius^3

/-- Calculates the new height of liquid in a cone after adding a marble -/
noncomputable def newHeight (c : Cone) (m : Marble) : ℝ :=
  c.height * ((coneVolume c + sphereVolume m) / coneVolume c)^(1/3)

/-- Theorem: The ratio of liquid level rise in two cones is 4:1 -/
theorem liquid_rise_ratio :
  ∀ (c1 c2 : Cone) (m : Marble),
    c1.radius = 4 →
    c2.radius = 8 →
    m.radius = 2 →
    coneVolume c1 = coneVolume c2 →
    (newHeight c1 m - c1.height) / (newHeight c2 m - c2.height) = 4 := by
  sorry

#check liquid_rise_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l620_62057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_count_lower_bound_l620_62021

/-- Triangle property: For any four numbers a, b, c, d where a is above b, 
    and c and d are below b on the respective columns, a * d < b * c holds -/
def triangle_property (t : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j k l, i < j → k ≠ l → 
    t i k * t j l < t j k * t i l

/-- A triangle of size n satisfying the triangle property -/
structure SpecialTriangle (n : ℕ) where
  t : ℕ → ℕ → ℕ
  property : triangle_property t

/-- The number of distinct elements in a triangle -/
def distinct_count {n : ℕ} (triangle : SpecialTriangle n) : ℕ :=
  Finset.card (Finset.image (fun (i j : Fin n) ↦ triangle.t i.val j.val) Finset.univ)

/-- Theorem: The number of distinct elements in a SpecialTriangle is at least n / 4 -/
theorem distinct_count_lower_bound {n : ℕ} (triangle : SpecialTriangle n) :
  distinct_count triangle ≥ n / 4 := by
  sorry

#check distinct_count_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_count_lower_bound_l620_62021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l620_62038

/-- Given a magnification factor and the magnified diameter of a circular tissue,
    calculate the actual diameter of the tissue. -/
noncomputable def actual_diameter (magnification : ℝ) (magnified_diameter : ℝ) : ℝ :=
  magnified_diameter / magnification

/-- Theorem stating that for a magnification of 1000 and a magnified diameter of 0.3 cm,
    the actual diameter is 0.0003 cm. -/
theorem tissue_diameter_calculation :
  let magnification : ℝ := 1000
  let magnified_diameter : ℝ := 0.3
  actual_diameter magnification magnified_diameter = 0.0003 := by
  -- Unfold the definition of actual_diameter
  unfold actual_diameter
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l620_62038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_in_range_l620_62058

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1
  else (a * x + 1) / (x + a)

-- State the theorem
theorem monotone_increasing_f_implies_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_in_range_l620_62058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_sd_l620_62003

/-- The standard deviation of a normal distribution -/
noncomputable def standard_deviation (mean : ℝ) (value_2sd_below : ℝ) : ℝ :=
  (mean - value_2sd_below) / 2

theorem normal_distribution_sd (mean value_2sd_below : ℝ)
  (h1 : mean = 16.5)
  (h2 : value_2sd_below = 13.5) :
  standard_deviation mean value_2sd_below = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_sd_l620_62003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_condition_minimum_value_condition_l620_62091

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 7 * a / x

theorem function_zero_condition (a : ℝ) :
  (∃! x, x > 0 ∧ f a x = 0) → a ∈ Set.Iic 0 ∪ {1 / (7 * Real.exp 1)} := by
  sorry

theorem minimum_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a x ≥ 3) ∧
  (∃ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a x = 3) →
  a = (Real.exp 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_condition_minimum_value_condition_l620_62091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l620_62051

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) : 
  (6 * (1.3 * s)^2 - 6 * s^2) / (6 * s^2) = 0.69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l620_62051
