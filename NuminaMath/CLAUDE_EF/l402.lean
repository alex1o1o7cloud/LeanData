import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l402_40296

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem inverse_function_value (g : ℝ → ℝ) (hg : Function.LeftInverse g f ∧ Function.RightInverse g f) :
  g (1/2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l402_40296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_10_l402_40264

def alternating_sequence : ℕ → ℕ
  | 0 => 8^7
  | m + 1 => if m % 2 = 0 then (alternating_sequence m) / 4 else (alternating_sequence m) * 3

theorem alternating_sequence_10 :
  alternating_sequence 10 = 2^11 * 3^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_10_l402_40264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l402_40215

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + (p.2 - 3)^2) = 10}

-- Define the points A, B, and C
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 6, 1)
def point_B : ℝ × ℝ := (1, 0)
def point_C : ℝ × ℝ := (3, 2)

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x = 3 ∨ 4*x - 3*y - 6 = 0

-- Define the chord length
def chord_length : ℝ := 2

-- Main theorem
theorem circle_and_line_properties :
  point_A ∈ circle_M ∧ 
  point_B ∈ circle_M ∧ 
  point_C ∈ circle_M ∧
  ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, line_l x y ↔ l x y) ∧
    l point_C.1 point_C.2 ∧
    (∀ p q : ℝ × ℝ, p ∈ circle_M ∧ q ∈ circle_M ∧ l p.1 p.2 ∧ l q.1 q.2 → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) :=
by sorry

#check circle_and_line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l402_40215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l402_40256

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) : ℝ :=
  4 * (base_edge * (Real.sqrt ((lateral_edge^2) - ((base_edge/2)^2)))) / 2

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid 
    with base edges of 10 units and lateral edges of 7 units is equal to 40√6 square units -/
theorem pyramid_area_theorem : 
  pyramid_face_area 10 7 = 40 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l402_40256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yardwork_earnings_distribution_l402_40221

theorem yardwork_earnings_distribution :
  let earnings : List ℝ := [18, 22, 30, 36, 50]
  let total : ℝ := earnings.sum
  let equal_share : ℝ := total / 5
  let highest_earner : ℝ := earnings.maximum.getD 0
  highest_earner - equal_share = 18.80 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yardwork_earnings_distribution_l402_40221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_bicycle_count_l402_40220

def bicycle_count (tricycles : ℕ) (total_wheels : ℕ) : ℕ :=
  let bicycle_wheels := 2
  let tricycle_wheels := 3
  (total_wheels - tricycle_wheels * tricycles) / bicycle_wheels

theorem correct_bicycle_count :
  bicycle_count 14 90 = 24 := by
  rfl

#eval bicycle_count 14 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_bicycle_count_l402_40220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_natural_numbers_12_to_53_l402_40244

theorem average_of_natural_numbers_12_to_53 :
  let start : ℕ := 12
  let stop : ℕ := 53
  let sequence := Finset.range (stop - start + 1)
  (sequence.sum (λ i => start + i) : ℚ) / sequence.card = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_natural_numbers_12_to_53_l402_40244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l402_40299

open Real

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * (tan t.A + tan t.B) = tan t.A / cos t.B + tan t.B / cos t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : given_condition t) : 
  (t.a + t.b = 2 * t.c) ∧ 
  (∀ (t' : Triangle), given_condition t' → cos t'.C ≥ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l402_40299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l402_40251

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  (a * b * Real.sin C = 20 * Real.sin B) →
  (a^2 + c^2 = 41) →
  (8 * Real.cos B = 1) →
  (b = 6) ∧
  (∃ x y, ((x = A ∧ y = B) ∨ (x = B ∧ y = C) ∨ (x = C ∧ y = A)) ∧ x = 2 * y) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l402_40251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l402_40238

noncomputable def f (x : ℝ) := Real.log (2 * x - 1) / Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l402_40238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_travel_distance_l402_40283

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem martin_travel_distance :
  let start : point := (3, -4)
  let middle : point := (0, 0)
  let end_ : point := (-5, 7)
  distance start middle + distance middle end_ = 5 + Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_travel_distance_l402_40283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_of_2_eq_2_l402_40224

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * (1 - x)
  else if 1 < x ∧ x ≤ 2 then x - 1
  else 0  -- This else case is added to make the function total

-- Define the n-fold composition of f
noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_n n x)

-- State the theorem
theorem f_2016_of_2_eq_2 : f_n 2016 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_of_2_eq_2_l402_40224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l402_40202

theorem angle_sum (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2)
  (h4 : Real.cos (α - β) = Real.sqrt 5 / 5) (h5 : Real.cos (2 * α) = Real.sqrt 10 / 10) :
  α + β = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l402_40202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_reduction_l402_40208

theorem computer_price_reduction : 
  ∀ (original_price : ℝ), original_price > 0 →
  (original_price - (original_price * (1 - 0.3) * (1 - 0.2))) / original_price = 0.44 :=
by
  intro original_price h_positive
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_reduction_l402_40208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l402_40232

/-- Triangle ABC with side lengths a, b, c satisfying the given equation -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0

/-- The radius of the circumcircle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := 
  t.c / 2

theorem triangle_properties (t : Triangle) : 
  t.a = 3 ∧ t.b = 4 ∧ t.c = 5 ∧ circumradius t = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l402_40232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_vector_l402_40277

/-- Given two vectors in ℝ², prove that the projection of one onto the other is equal to the vector being projected onto. -/
theorem projection_equals_vector (a b : ℝ × ℝ) :
  a = (1, 0) →
  b = (1, Real.sqrt 3) →
  ((b.1 * a.1 + b.2 * a.2) / (a.1^2 + a.2^2)) • a = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_vector_l402_40277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_transformation_l402_40258

-- Define the initial line
def initial_line (x y : ℝ) : Prop := y = 3 * x

-- Define the rotation transformation
def rotate_90_ccw (x y : ℝ) : ℝ × ℝ := (-y, x)

-- Define the translation transformation
def translate_right (x y : ℝ) : ℝ × ℝ := (x + 1, y)

-- Define the final line
def final_line (x y : ℝ) : Prop := y = -1/3 * x + 1/3

-- Theorem statement
theorem line_transformation :
  ∀ x y x' y' x'' y'' : ℝ, 
    initial_line x y →
    (x', y') = rotate_90_ccw x y →
    (x'', y'') = translate_right x' y' →
    final_line x'' y'' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_transformation_l402_40258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_time_sum_l402_40255

-- Define the constants from the problem
def path_distance : ℝ := 300
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 1
def building_diameter : ℝ := 200

-- Define a function to represent the time when Jenny and Kenny can see each other again
def visibility_time (path_distance kenny_speed jenny_speed building_diameter : ℝ) : ℝ := 
  sorry

-- Define a function to represent the fraction in lowest terms
def lowest_terms (x : ℝ) : ℤ × ℤ := 
  sorry

-- Theorem statement
theorem visibility_time_sum :
  ∃ (n d : ℤ), 
    lowest_terms (visibility_time path_distance kenny_speed jenny_speed building_diameter) = (n, d) ∧ 
    n + d = 245 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_time_sum_l402_40255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l402_40223

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- Represents a circle with center at the right focus of the ellipse -/
structure FocalCircle (e : Ellipse) where
  center : ℝ × ℝ  -- Right focus of the ellipse
  radius : ℝ
  passes_through_center : (0, 0) ∈ Metric.sphere center radius
  intersects_ellipse : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M ∈ Metric.sphere center radius ∧ N ∈ Metric.sphere center radius ∧
                        (M.1 / e.a) ^ 2 + (M.2 / e.b) ^ 2 = 1 ∧ (N.1 / e.a) ^ 2 + (N.2 / e.b) ^ 2 = 1

/-- The line from the left focus to point M is tangent to the circle -/
def tangent_condition (e : Ellipse) (c : FocalCircle e) (M : ℝ × ℝ) : Prop :=
  let left_focus := (-e.a * Real.sqrt (e.a^2 - e.b^2) / e.a, 0)
  ∃ t : ℝ, (left_focus.1 + t * (M.1 - left_focus.1), left_focus.2 + t * (M.2 - left_focus.2)) ∈ Metric.sphere c.center c.radius

/-- The main theorem: given an ellipse with the specified conditions, its eccentricity is √3 - 1 -/
theorem ellipse_eccentricity (e : Ellipse) (c : FocalCircle e) 
  (h : ∃ M : ℝ × ℝ, M ∈ Metric.sphere c.center c.radius ∧ (M.1 / e.a) ^ 2 + (M.2 / e.b) ^ 2 = 1 ∧ tangent_condition e c M) :
  (e.a^2 - e.b^2) / e.a^2 = (Real.sqrt 3 - 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l402_40223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l402_40254

-- Define the constants
noncomputable def a : ℝ := Real.log 0.2 / Real.log 10
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.sqrt 5

-- State the theorem
theorem ordering_abc : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l402_40254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_shangri_la_to_atlantis_l402_40267

/-- The distance between two points in a complex plane -/
noncomputable def complex_distance (z₁ z₂ : ℂ) : ℝ :=
  Real.sqrt ((z₁.re - z₂.re)^2 + (z₁.im - z₂.im)^2)

/-- Atlantis location -/
noncomputable def atlantis : ℂ := 0

/-- Shangri-La location -/
noncomputable def shangri_la : ℂ := Complex.mk 1260 1680

theorem distance_shangri_la_to_atlantis :
  complex_distance shangri_la atlantis = 2100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_shangri_la_to_atlantis_l402_40267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l402_40291

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the hyperbola C -/
noncomputable def C (a b : ℝ) : Hyperbola a b := sorry

/-- Left focus of the hyperbola -/
noncomputable def F₁ (a b : ℝ) : Point := sorry

/-- Right focus of the hyperbola -/
noncomputable def F₂ (a b : ℝ) : Point := sorry

/-- Point on the right branch of the hyperbola -/
noncomputable def P (a b : ℝ) : Point := sorry

/-- Point where PF₁ intersects the left branch of the hyperbola -/
noncomputable def Q (a b : ℝ) : Point := sorry

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Perpendicularity of line segments -/
def perpendicular (p q r : Point) : Prop := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Main theorem: The eccentricity of hyperbola C is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) :
  perpendicular (P a b) (F₁ a b) (F₂ a b) →
  distance (P a b) (F₂ a b) = 3/2 * distance (Q a b) (F₁ a b) →
  eccentricity (C a b) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l402_40291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_annual_income_is_4300_l402_40234

/-- Calculates the annual income from a stock investment -/
noncomputable def annual_income (investment : ℚ) (dividend_rate : ℚ) (price : ℚ) : ℚ :=
  (investment / price) * (dividend_rate * 100)

/-- Theorem: The total annual income from the given investments is $4300 -/
theorem total_annual_income_is_4300 :
  let investment1 : ℚ := 6800
  let dividend_rate1 : ℚ := 40 / 100
  let price1 : ℚ := 136
  let investment2 : ℚ := 3500
  let dividend_rate2 : ℚ := 50 / 100
  let price2 : ℚ := 125
  let investment3 : ℚ := 4500
  let dividend_rate3 : ℚ := 30 / 100
  let price3 : ℚ := 150
  annual_income investment1 dividend_rate1 price1 +
  annual_income investment2 dividend_rate2 price2 +
  annual_income investment3 dividend_rate3 price3 = 4300 := by
  sorry

#check total_annual_income_is_4300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_annual_income_is_4300_l402_40234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_properties_l402_40225

/-- A random variable following a Bernoulli distribution -/
structure BernoulliDist where
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- A random variable following a binomial distribution -/
structure BinomialDist where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a Bernoulli distribution -/
def expectedValueBernoulli (X : BernoulliDist) : ℝ := X.p

/-- Variance of a Bernoulli distribution -/
def varianceBernoulli (X : BernoulliDist) : ℝ := X.p * (1 - X.p)

/-- Expected value of a binomial distribution -/
def expectedValueBinomial (Y : BinomialDist) : ℝ := (Y.n : ℝ) * Y.p

/-- Variance of a binomial distribution -/
def varianceBinomial (Y : BinomialDist) : ℝ := (Y.n : ℝ) * Y.p * (1 - Y.p)

theorem distribution_properties :
  let X : BernoulliDist := ⟨0.7, by norm_num⟩
  let Y : BinomialDist := ⟨10, 0.8, by norm_num⟩
  (expectedValueBernoulli X = 0.7) ∧
  (varianceBernoulli X = 0.21) ∧
  (expectedValueBinomial Y = 8) ∧
  (varianceBinomial Y = 1.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_properties_l402_40225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l402_40257

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_condition : ∀ x, x < 0 → f x + x * (deriv f x) < 0

-- Define a, b, c
noncomputable def a : ℝ := (3 : ℝ)^(3/10) * f ((3 : ℝ)^(3/10))
noncomputable def b : ℝ := (Real.log 3 / Real.log Real.pi) * f (Real.log 3 / Real.log Real.pi)
noncomputable def c : ℝ := (Real.log (1/9) / Real.log 3) * f (Real.log (1/9) / Real.log 3)

-- State the theorem
theorem abc_relationship (f : ℝ → ℝ) : c f > a f ∧ a f > b f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l402_40257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_equation_l402_40286

theorem bus_speed_equation (x : ℝ) (h1 : x > 0) : 
  (12 / x) - (12 / (1.2 * x)) = 3 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_equation_l402_40286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_criterion_l402_40263

/-- A triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The semiperimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- The area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := 
  (t.a * t.b * t.c) / (4 * area t)

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ := 
  area t / semiperimeter t

/-- A triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- The main theorem: 2R + r = p iff the triangle is right-angled -/
theorem right_angle_criterion (t : Triangle) : 
  2 * circumradius t + inradius t = semiperimeter t ↔ is_right_angled t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_criterion_l402_40263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l402_40270

/-- The number of even natural-number factors of 2^2 * 3^1 * 7^2 * 5^1 -/
def num_even_factors : ℕ := 24

/-- The prime factorization of n -/
def n : ℕ := 2^2 * 3^1 * 7^2 * 5^1

/-- Theorem stating that the number of even natural-number factors of n is equal to num_even_factors -/
theorem count_even_factors :
  (Finset.filter (fun x => x ∣ n ∧ Even x) (Finset.range (n + 1))).card = num_even_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l402_40270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_distances_l402_40252

/-- Line l in parametric form -/
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

/-- Curve C in polar form -/
noncomputable def curve_C (p : ℝ) (θ : ℝ) : ℝ :=
  p / (1 - Real.cos θ)

/-- Theorem stating the sum of reciprocals of distances from origin to intersection points -/
theorem sum_reciprocal_distances (α p : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : p > 0) :
  ∃ A B : ℝ × ℝ,
    (∃ t1 t2 : ℝ, line_l α t1 = A ∧ line_l α t2 = B) ∧
    (∃ θ1 θ2 : ℝ, curve_C p θ1 = Real.sqrt (A.1^2 + A.2^2) ∧
                     curve_C p θ2 = Real.sqrt (B.1^2 + B.2^2)) →
    1 / Real.sqrt (A.1^2 + A.2^2) + 1 / Real.sqrt (B.1^2 + B.2^2) = 2 / p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_distances_l402_40252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barry_sotter_magic_l402_40219

theorem barry_sotter_magic (n : ℕ) : n = 147 := by
  -- Define the increase factor for day k
  let increase_factor (k : ℕ) : ℚ :=
    if k = 0 then 4/3 else (k + 3 : ℚ) / (k + 2)

  -- Define the cumulative product of increase factors up to day n
  let cumulative_product (n : ℕ) : ℚ :=
    (Finset.range (n + 1)).prod (λ k => increase_factor k)

  -- State that the cumulative product equals 50
  have h : cumulative_product n = 50 := by sorry

  -- Prove that n = 147
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barry_sotter_magic_l402_40219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contradiction_proof_l402_40289

noncomputable def sequence_x (x₁ : ℝ) : ℕ → ℝ
  | 0 => x₁
  | n + 1 => (sequence_x x₁ n * ((sequence_x x₁ n)^2 + 3)) / (3 * (sequence_x x₁ n)^2 + 1)

theorem contradiction_proof (x₁ : ℝ) (h1 : x₁ > 0) (h2 : x₁ ≠ 1) :
  (∃ k : ℕ, k > 0 ∧ sequence_x x₁ k ≤ sequence_x x₁ (k + 1)) →
  ¬(∀ n : ℕ, n > 0 → sequence_x x₁ n > sequence_x x₁ (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contradiction_proof_l402_40289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_fraction_sum_l402_40226

theorem decimal_fraction_sum : ∃ (a b c d : ℕ), 
  (a = 32 ∨ a = 22 ∨ a = 23) ∧ 
  (b = 32 ∨ b = 22 ∨ b = 23) ∧ 
  (c = 32 ∨ c = 22 ∨ c = 23) ∧ 
  (d = 32 ∨ d = 22 ∨ d = 23) ∧ 
  (a : ℚ) / 100 + (b : ℚ) / 100 + (c : ℚ) / 100 + (d : ℚ) / 100 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_fraction_sum_l402_40226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bound_is_ten_l402_40203

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define a partition of the interval [0, 4]
def IsPartition (p : List ℝ) : Prop :=
  p.length > 1 ∧ 
  (∀ i, i < p.length - 1 → p[i]! < p[i+1]!) ∧
  p.head! = 0 ∧ 
  p.getLast! = 4

-- Define the sum of absolute differences
def SumAbsDiff (p : List ℝ) : ℝ :=
  (List.zip p (List.tail! p)).foldr (λ pair acc => acc + |f pair.2 - f pair.1|) 0

-- Theorem statement
theorem min_bound_is_ten :
  ∃ (M : ℝ), (∀ (p : List ℝ), IsPartition p → SumAbsDiff p ≤ M) ∧
             (∀ (M' : ℝ), (∀ (p : List ℝ), IsPartition p → SumAbsDiff p ≤ M') → M ≤ M') ∧
             M = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bound_is_ten_l402_40203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_value_l402_40261

/-- Represents a cell in the grid -/
structure Cell where
  value : ℕ
  neighbors : List ℕ

/-- Represents the grid configuration -/
structure Grid where
  cells : List Cell
  hasFour : ∃ c, c ∈ cells ∧ c.value = 4
  hasFive : ∃ c, c ∈ cells ∧ c.value = 5

/-- Checks if a cell satisfies the multiple condition -/
def isValidCell (cell : Cell) : Prop :=
  ∃ k : ℕ, k * cell.value = cell.neighbors.sum

/-- Checks if all cells in the grid are valid -/
def isValidGrid (grid : Grid) : Prop :=
  ∀ cell, cell ∈ grid.cells → isValidCell cell

/-- Checks if all numbers from 1 to 9 are used exactly once -/
def usesAllNumbers (grid : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃! cell, cell ∈ grid.cells ∧ cell.value = n

/-- The main theorem -/
theorem max_x_value (grid : Grid) 
  (h1 : isValidGrid grid) 
  (h2 : usesAllNumbers grid) : 
  ∃ x : ℕ, x ≤ 6 ∧ 
    ∃ cell, cell ∈ grid.cells ∧ cell.value = x ∧ 
    ∀ cell', cell' ∈ grid.cells → cell'.value ≤ x :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_value_l402_40261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shuffle_returns_to_original_l402_40275

/-- Represents a sequence of 2n elements -/
def Sequence (α : Type) (n : ℕ) := Fin (2 * n) → α

/-- Performs a shuffle transformation on a sequence -/
def shuffleTransform {α : Type} {n : ℕ} (seq : Sequence α n) : Sequence α n :=
  fun i => 
    if i.val % 2 = 0 
    then seq ⟨(i.val / 2 + n) % (2 * n), by sorry⟩ 
    else seq ⟨((i.val - 1) / 2 + 1) % (2 * n), by sorry⟩

/-- Applies m shuffle transformations to a sequence -/
def applyShuffles {α : Type} {n : ℕ} (m : ℕ) (seq : Sequence α n) : Sequence α n :=
  (List.range m).foldl (fun s _ => shuffleTransform s) seq

theorem shuffle_returns_to_original {α : Type} {n : ℕ} (seq : Sequence α n) :
  applyShuffles (Nat.totient (2 * n - 1)) seq = seq := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shuffle_returns_to_original_l402_40275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l402_40227

def polynomial (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem polynomial_roots :
  (∀ x : ℝ, polynomial x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (deriv polynomial 1 ≠ 0) ∧
  (deriv polynomial 2 ≠ 0) ∧
  (deriv polynomial 3 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l402_40227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l402_40206

/-- Definition of the sequence x_k -/
def x (k : ℕ) : ℤ := (-1) ^ (k + 1)

/-- Definition of the function g -/
noncomputable def g (n : ℕ+) : ℚ :=
  |((Finset.range n.val).sum (λ k => x (k + 1)) : ℚ) / n.val|

/-- Theorem stating the possible values of g(n) -/
theorem g_values (n : ℕ+) : g n = 0 ∨ g n = 1 / n :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l402_40206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l402_40228

theorem simplify_expression : 
  (-2.4) - (-4.7) - (0.5) + (-3.5) = -2.4 + 4.7 - 0.5 - 3.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l402_40228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_l402_40253

theorem sum_of_solutions_quadratic :
  let a : ℝ := 18
  let b : ℝ := -27
  let c : ℝ := -45
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  sum_of_roots = 3/2 := by
  -- Proof goes here
  sorry

#eval (3 : ℚ) / 2  -- This will evaluate to 1.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_l402_40253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equation_l402_40260

theorem power_sum_equation (x : ℝ) : 
  (2 : ℝ)^x + (2 : ℝ)^x + (2 : ℝ)^x + (2 : ℝ)^x + (2 : ℝ)^x + (2 : ℝ)^x = 4096 → x = 9.415 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equation_l402_40260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_wins_probability_value_log2_denominator_l402_40229

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The number of games each team plays -/
def games_per_team : ℕ := num_teams - 1

/-- The total number of games in the tournament -/
def total_games : ℕ := num_teams * games_per_team / 2

/-- The probability of winning a single game -/
def win_probability : ℚ := 1 / 2

/-- The probability that all teams win a unique number of games -/
noncomputable def unique_wins_probability : ℚ := (Nat.factorial num_teams : ℚ) / (2 ^ total_games)

theorem unique_wins_probability_value :
  unique_wins_probability = (Nat.factorial 30 : ℚ) / (2 ^ 435) := by
  sorry

theorem log2_denominator :
  ∃ n : ℕ, unique_wins_probability = (Nat.factorial 30 : ℚ) / (2 ^ n) ∧ n = 409 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_wins_probability_value_log2_denominator_l402_40229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_c_most_stable_l402_40200

/-- Represents a packaging machine with its sample variance -/
structure PackagingMachine where
  name : String
  variance : Float
  deriving Repr

/-- Theorem: Given three packaging machines A, B, and C with their respective sample variances,
    machine C has the smallest variance and is therefore the most stable in terms of weight. -/
theorem machine_c_most_stable (machineA machineB machineC : PackagingMachine)
    (hA : machineA.name = "A" ∧ machineA.variance = 10.3)
    (hB : machineB.name = "B" ∧ machineB.variance = 6.9)
    (hC : machineC.name = "C" ∧ machineC.variance = 3.5) :
    machineC.variance < machineA.variance ∧ machineC.variance < machineB.variance := by
  sorry

#check machine_c_most_stable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_c_most_stable_l402_40200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l402_40266

/-- Represents a triangle as a triple of real numbers (side lengths) -/
def Triangle := ℝ × ℝ × ℝ

/-- Checks if a triangle is isosceles -/
def IsIsosceles (t : Triangle) : Prop :=
  let (a, b, c) := t
  a = b ∨ b = c ∨ c = a

/-- Calculates the perimeter of a triangle -/
def PerimeterTriangle (t : Triangle) : ℝ :=
  let (a, b, c) := t
  a + b + c

/-- Returns the set of sides of a triangle -/
def SidesOfTriangle (t : Triangle) : Set ℝ :=
  let (a, b, c) := t
  {a, b, c}

/-- Returns the base of an isosceles triangle -/
noncomputable def BaseOfIsoscelesTriangle (t : Triangle) (h : IsIsosceles t) : ℝ :=
  let (a, b, c) := t
  if a = b then c
  else if b = c then a
  else b

theorem isosceles_triangle_base_length 
  (t : Triangle)
  (is_isosceles : IsIsosceles t)
  (perimeter : PerimeterTriangle t = 8)
  (one_side : ∃ s ∈ SidesOfTriangle t, s = 2) :
  BaseOfIsoscelesTriangle t is_isosceles = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l402_40266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_strip_length_for_reverse_order_l402_40230

/-- Represents a checker on the strip -/
structure Checker where
  position : ℕ

/-- Represents the state of the strip -/
structure Strip where
  length : ℕ
  checkers : List Checker

/-- Defines a valid move for a checker -/
def validMove (s : Strip) (fromPos toPos : ℕ) : Prop :=
  toPos > fromPos ∧ toPos ≤ s.length ∧
  (toPos = fromPos + 1 ∨ (∃ c : Checker, c ∈ s.checkers ∧ c.position = fromPos + 1 ∧ toPos = fromPos + 2))

/-- Defines the initial state of the strip -/
def initialStrip (n : ℕ) : Strip :=
  { length := n
  , checkers := List.range 25 |>.map (λ i => ⟨i + 1⟩) }

/-- Defines the final state of the strip -/
def finalStrip (n : ℕ) : Strip :=
  { length := n
  , checkers := List.range 25 |>.map (λ i => ⟨n - i⟩) }

/-- Theorem: The minimum N for rearranging 25 checkers in reverse order is 50 -/
theorem min_strip_length_for_reverse_order : 
  ∀ n : ℕ, n ≥ 50 → 
  ∃ (moves : List (ℕ × ℕ)), 
    (∀ (fromPos toPos : ℕ), (fromPos, toPos) ∈ moves → validMove (initialStrip n) fromPos toPos) ∧
    (List.foldl (λ s (fromPos, toPos) => 
      { length := s.length
      , checkers := s.checkers.map (λ c => if c.position = fromPos then { position := toPos } else c)
      }) (initialStrip n) moves) = finalStrip n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_strip_length_for_reverse_order_l402_40230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_given_radii_l402_40285

/-- The radius of an inscribed circle within three mutually externally tangent circles -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem: The radius of the inscribed circle for given external circle radii -/
theorem inscribed_circle_radius_for_given_radii :
  let a : ℝ := 5
  let b : ℝ := 10
  let c : ℝ := 20
  let r := inscribed_circle_radius a b c
  abs (r - 1.381) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_given_radii_l402_40285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_stable_number_count_stable_bases_correct_l402_40216

/-- A number is stable in base B if it satisfies the stability condition --/
def is_stable (n : ℕ) (B : ℕ) : Prop :=
  ∃ (x y z t : ℕ),
    n = x * B^3 + y * B^2 + z * B + t ∧
    x < B ∧ y < B ∧ z < B ∧ t < B ∧
    ∃ (a b c d : ℕ),
      a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
      Finset.toSet {a, b, c, d} = Finset.toSet {x, y, z, t} ∧
      x * B^3 + y * B^2 + z * B + t = (d * B^3 + c * B^2 + b * B + a) - (a * B^3 + b * B^2 + c * B + d)

/-- Theorem: The only stable number in any base B ≥ 2 is 0000_B --/
theorem unique_stable_number (B : ℕ) (h : B ≥ 2) :
  ∀ n : ℕ, is_stable n B ↔ n = 0 := by
  sorry

/-- The count of bases B ≤ 1985 with stable numbers --/
def count_stable_bases : ℕ := 1984

/-- Theorem: The count of bases B ≤ 1985 with stable numbers is 1984 --/
theorem count_stable_bases_correct :
  count_stable_bases = 1984 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_stable_number_count_stable_bases_correct_l402_40216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_cube_series_volume_l402_40269

/-- Given a cube with edge length a, this function returns the edge length of the next cube in the series -/
noncomputable def nextCubeEdgeLength (a : ℝ) : ℝ := a / 3

/-- The volume of a cube with edge length a -/
noncomputable def cubeVolume (a : ℝ) : ℝ := a^3

/-- The sum of the volumes of all cubes in the infinite series -/
noncomputable def totalVolume (a : ℝ) : ℝ := (27 / 26) * a^3

/-- Theorem stating that the sum of the volumes of all cubes in the infinite series
    is equal to (27/26) * a^3, where a is the edge length of the first cube -/
theorem infinite_cube_series_volume (a : ℝ) (h : a > 0) :
  ∑' n, cubeVolume (a / 3^n) = totalVolume a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_cube_series_volume_l402_40269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gardensquare_to_madison_map_scale_l402_40245

/-- The scale of a map given distance on map, travel time, and average speed -/
noncomputable def map_scale (map_distance : ℝ) (travel_time : ℝ) (average_speed : ℝ) : ℝ :=
  map_distance / (travel_time * average_speed)

/-- Theorem: The scale of the map from Gardensquare to Madison is 1/18 inches per mile -/
theorem gardensquare_to_madison_map_scale :
  let map_distance : ℝ := 5 -- inches
  let travel_time : ℝ := 1.5 -- hours
  let average_speed : ℝ := 60 -- miles per hour
  map_scale map_distance travel_time average_speed = 1 / 18 := by
  -- Unfold the definition of map_scale
  unfold map_scale
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gardensquare_to_madison_map_scale_l402_40245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l402_40294

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / 2) * x^2 + (a + 1) * x + 2 * Real.log (x - 1)

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x > 1, (6 : ℝ) * x - 3 * f a 2 - 10 = 0 → 
    HasDerivAt (f a) ((6 : ℝ) / 3) 2) ∧
  (a ≥ 0 → ∀ x > 1, (deriv (f a)) x > 0) ∧
  (a < 0 → (∀ x, 1 < x ∧ x < (a - 1) / a → (deriv (f a)) x > 0) ∧
           (∀ x > (a - 1) / a, (deriv (f a)) x < 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l402_40294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l402_40240

noncomputable section

open Real EuclideanGeometry

theorem triangle_side_lengths 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_angle_A : angle B A C = π / 4)
  (h_angle_B : angle A B C = π / 3)
  (h_BC : dist B C = 2) :
  dist A C = sqrt 6 ∧ dist A B = 1 + sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l402_40240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l402_40268

-- Define the function g based on the graph
noncomputable def g : ℝ → ℝ :=
  fun x => if x < -1 then 2*x + 4
           else if x < 3 then 4/3*x + 2
           else -x/2 + 5/2

-- Define the property we want to prove
def satisfies_equation (x : ℝ) : Prop :=
  g (g x) = 4

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, satisfies_equation x) ∧ s.card = 3 := by
  sorry

#check exactly_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l402_40268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_pipe_fill_rate_l402_40280

/-- The rate at which the tank empties when only the leak is open, in litres per hour -/
noncomputable def leak_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) : ℝ :=
  tank_capacity / leak_empty_time

/-- The net rate at which the tank empties when both inlet and leak are open, in litres per hour -/
noncomputable def net_rate (tank_capacity : ℝ) (both_empty_time : ℝ) : ℝ :=
  tank_capacity / both_empty_time

/-- The rate at which the inlet pipe fills the tank, in litres per hour -/
noncomputable def inlet_rate (leak_rate : ℝ) (net_rate : ℝ) : ℝ :=
  leak_rate - net_rate

/-- Convert rate from litres per hour to litres per minute -/
noncomputable def to_litres_per_minute (rate : ℝ) : ℝ :=
  rate / 60

theorem inlet_pipe_fill_rate 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (both_empty_time : ℝ) :
  tank_capacity = 2160 ∧ 
  leak_empty_time = 4 ∧ 
  both_empty_time = 12 → 
  to_litres_per_minute (inlet_rate (leak_rate tank_capacity leak_empty_time) (net_rate tank_capacity both_empty_time)) = 12 := by
  sorry

#check inlet_pipe_fill_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_pipe_fill_rate_l402_40280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_cassini_oval_l402_40284

/-- The Cassini oval curve in the Cartesian plane -/
noncomputable def cassini_oval (x y : ℝ) : Prop :=
  x^2 + y^2 + 1 = 2 * Real.sqrt (x^2 + 1)

/-- The distance from the origin to a point (x, y) -/
noncomputable def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- The maximum value of OP for points on the Cassini oval -/
theorem max_distance_cassini_oval : 
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ 
  (∀ x y : ℝ, cassini_oval x y → distance_from_origin x y ≤ M) ∧
  (∃ x y : ℝ, cassini_oval x y ∧ distance_from_origin x y = M) := by
  sorry

#check max_distance_cassini_oval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_cassini_oval_l402_40284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aqua_park_earnings_l402_40212

/-- Calculates the total earnings of an aqua park given specific conditions. -/
theorem aqua_park_earnings (admission_fee tour_fee : ℕ) 
  (group1_size group2_size : ℕ) : 
  admission_fee = 12 →
  tour_fee = 6 →
  group1_size = 10 →
  group2_size = 5 →
  (group1_size * (admission_fee + tour_fee) + group2_size * admission_fee) = 240 := by
  intros h1 h2 h3 h4
  sorry

#check aqua_park_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aqua_park_earnings_l402_40212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l402_40259

theorem shopkeeper_profit (total_apples : ℝ) (profit1 profit2 portion1 portion2 : ℝ) :
  total_apples = 280 ∧
  portion1 = 0.4 ∧
  portion2 = 0.6 ∧
  profit1 = 0.2 ∧
  profit2 = 0.3 →
  let apples1 := total_apples * portion1
  let apples2 := total_apples * portion2
  let cost1 := apples1
  let cost2 := apples2
  let profit_amount1 := cost1 * profit1
  let profit_amount2 := cost2 * profit2
  let total_profit := profit_amount1 + profit_amount2
  let total_cost := cost1 + cost2
  total_profit / total_cost = 0.26 := by
  intro h
  -- Proof steps would go here
  sorry

#check shopkeeper_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l402_40259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_hydrangea_start_year_l402_40287

def hydrangea_cost : ℚ := 20
def total_spent : ℚ := 640
def end_year : ℕ := 2021

def start_year : ℕ := end_year - (total_spent / hydrangea_cost - 1).floor.toNat

theorem lily_hydrangea_start_year :
  start_year = 1990 := by
  -- Unfold the definition of start_year
  unfold start_year
  -- Simplify the arithmetic
  simp [hydrangea_cost, total_spent, end_year]
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_hydrangea_start_year_l402_40287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b6_value_l402_40209

noncomputable section

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem b6_value (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  sum_arithmetic a 9 = -36 →
  sum_arithmetic a 13 = -104 →
  b 5 = a 5 →
  b 7 = a 7 →
  (b 6 = 4 * Real.sqrt 2 ∨ b 6 = -4 * Real.sqrt 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b6_value_l402_40209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ships_same_direction_time_l402_40250

/-- Represents the travel time of ships in a river with different upstream and downstream speeds -/
structure RiverTravel where
  downstream_speed : ℝ
  upstream_speed : ℝ
  total_time : ℝ

/-- Calculates the time ships travel in the same direction -/
noncomputable def same_direction_time (travel : RiverTravel) : ℝ :=
  let x := travel.total_time * travel.upstream_speed / (travel.downstream_speed + travel.upstream_speed)
  travel.total_time - 2 * x

/-- Theorem stating that under specific conditions, ships travel in the same direction for 1 hour -/
theorem ships_same_direction_time :
  ∀ (travel : RiverTravel),
  travel.downstream_speed = 8 ∧
  travel.upstream_speed = 4 ∧
  travel.total_time = 3 →
  same_direction_time travel = 1 := by
  sorry

-- Remove the #eval statement as it's not necessary for compilation
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ships_same_direction_time_l402_40250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l402_40222

-- Define the function y
def y (x a : ℝ) : ℝ := x^2 - 2*a*x + a

-- Define set A
def A : Set ℝ := {x | x^2 + 4*x = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | y x a + a^2 - a = -(4*a + 2)*x + 1}

-- Part 1
theorem part_1 (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x₁ x₂, y x₁ a = 0 ∧ y x₂ a = 0 ∧ x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 6*x₁*x₂ - 3 →
  a = 3/2 := by sorry

-- Part 2
theorem part_2 (a : ℝ) :
  A ∪ B a = A →
  a ∈ Set.Iic (-1) ∪ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l402_40222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_remaining_eq_cookies_taken_l402_40231

/-- Represents the number of cookies in the cookie jar -/
def total_cookies : ℕ := sorry

/-- Represents the number of cookies Lou Senior took -/
def lou_senior_cookies : ℕ := 4

/-- Represents the number of cookies Louie Junior took -/
def louie_junior_cookies : ℕ := 7

/-- Represents the total number of cookies taken -/
def cookies_taken : ℕ := lou_senior_cookies + louie_junior_cookies

/-- Represents the number of cookies remaining in the jar -/
def cookies_remaining : ℕ := sorry

/-- States that half of the cookies were taken -/
axiom half_cookies_taken : 2 * cookies_taken = total_cookies

/-- Theorem: The number of cookies remaining is equal to the number of cookies taken -/
theorem cookies_remaining_eq_cookies_taken : cookies_remaining = cookies_taken := by
  sorry

#check cookies_remaining_eq_cookies_taken

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_remaining_eq_cookies_taken_l402_40231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l402_40236

theorem equation_solutions_count : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ s ↔ m > 0 ∧ n > 0 ∧ (6 : ℚ) / m + (3 : ℚ) / n = 1) ∧ 
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l402_40236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_exp_neg_x_minus_one_l402_40233

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g (the intermediate function before translation)
def g : ℝ → ℝ := sorry

-- State the theorem
theorem f_is_exp_neg_x_minus_one :
  (∀ x, f x = g (x - 1)) →  -- Condition 1: f is g translated 1 unit right
  (∀ x, g x = exp (-x)) →   -- Condition 2: g is symmetric to exp x w.r.t. y-axis
  ∀ x, f x = exp (-x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_exp_neg_x_minus_one_l402_40233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_complementary_angle_l402_40207

theorem sin_eq_cos_complementary_angle (θ : ℝ) :
  Real.cos ((5 * π) / 12 - θ) = 1 / 3 →
  Real.sin ((π / 12) + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_complementary_angle_l402_40207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_intersection_distance_l402_40243

/-- A circle in which a quadrilateral is inscribed -/
structure InscribedQuadrilateralCircle where
  /-- The circle -/
  circle : Set (Fin 2 → ℝ)
  /-- The inscribed quadrilateral -/
  quad : Set (Fin 2 → ℝ)
  /-- The quadrilateral is inscribed in the circle -/
  inscribed : quad ⊆ circle
  /-- P is the intersection of one pair of opposite sides -/
  P : Fin 2 → ℝ
  /-- Q is the intersection of the other pair of opposite sides -/
  Q : Fin 2 → ℝ
  /-- Length of tangent from P to the circle -/
  a : ℝ
  /-- Length of tangent from Q to the circle -/
  b : ℝ
  /-- P is outside the circle -/
  P_outside : P ∉ circle
  /-- Q is outside the circle -/
  Q_outside : Q ∉ circle

/-- The theorem stating the relation between PQ, a, and b -/
theorem inscribed_quadrilateral_intersection_distance
  (c : InscribedQuadrilateralCircle) :
  ‖c.P - c.Q‖ = Real.sqrt (c.a^2 + c.b^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_intersection_distance_l402_40243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l402_40278

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point A(3, 2) to the line x + y + 3 = 0 is 4√2 -/
theorem distance_point_to_line :
  distancePointToLine 3 2 1 1 3 = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l402_40278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_negative_reals_exists_a_for_odd_g_l402_40241

-- Define the function f(x) = x + 1/x
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Define the function g(x, a) = x/((2x+1)(x-a))
noncomputable def g (x a : ℝ) : ℝ := x / ((2*x + 1) * (x - a))

-- Theorem 1: f(x) has a maximum value of -2 on (-∞, 0)
theorem f_max_negative_reals : 
  ∀ x < 0, f x ≤ -2 ∧ ∃ y < 0, f y = -2 := by
  sorry

-- Theorem 2: There exists an 'a' such that g(x, a) is an odd function
theorem exists_a_for_odd_g :
  ∃ a : ℝ, ∀ x : ℝ, g (-x) a = -(g x a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_negative_reals_exists_a_for_odd_g_l402_40241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_squared_l402_40272

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem intersection_chord_length_squared
  (c1 c2 : Circle)
  (h1 : c1.radius = 8)
  (h2 : c2.radius = 6)
  (h3 : distance c1.center c2.center = 12)
  (P : Point)
  (h4 : distance c1.center P = c1.radius ∧ distance c2.center P = c2.radius)
  (Q R : Point)
  (h5 : distance P Q = distance P R)
  (h6 : distance c1.center Q = c1.radius)
  (h7 : distance c2.center R = c2.radius) :
  (distance P Q)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_squared_l402_40272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l402_40290

noncomputable def f (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x

theorem f_properties :
  (∀ x y, x ∈ Set.Icc (-π/6) (5*π/6) → y ∈ Set.Icc (-π/6) (5*π/6) → x < y → f x < f y) ∧
  f (4*π/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l402_40290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l402_40213

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 6 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the center of circle N
def center_N : ℝ × ℝ := (0, 0)

-- Define the tangency condition
def tangent_circles : Prop := ∃ (x y : ℝ), circle_M x y ∧ circle_N x y

-- Define points E and F on x-axis
noncomputable def point_E : ℝ × ℝ := (-Real.sqrt 8, 0)
noncomputable def point_F : ℝ × ℝ := (Real.sqrt 8, 0)

-- Define the geometric sequence condition
def geometric_sequence (D : ℝ × ℝ) : Prop :=
  let x := D.1
  let y := D.2
  x^2 + y^2 = (x + Real.sqrt 8) * (Real.sqrt 8 - x)

-- Define the complementary angle condition for slopes
def complementary_slopes (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

-- Main theorem
theorem circle_problem :
  ∀ (A B : ℝ × ℝ),
  tangent_circles →
  (∀ D : ℝ × ℝ, D.1^2 + D.2^2 < 8 → geometric_sequence D) →
  (∃ k₁ k₂, complementary_slopes k₁ k₂ ∧
    (A.2 - 1 = k₁ * (A.1 - 1)) ∧
    (B.2 - 1 = k₂ * (B.1 - 1))) →
  (∀ x y, circle_N x y ↔ x^2 + y^2 = 8) ∧
  (∀ D : ℝ × ℝ, D.1^2 + D.2^2 < 8 → -1 ≤ Real.sqrt (D.1^2 + D.2^2) ∧ Real.sqrt (D.1^2 + D.2^2) < 0) ∧
  ((1 - A.1) * (B.2 - 1) = (1 - B.1) * (A.2 - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l402_40213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_problem_l402_40211

theorem jellybean_problem (initial_jellybeans : ℕ) (shelby_eaten : ℕ) (final_jellybeans : ℕ) 
  (samantha_taken : ℕ) :
  initial_jellybeans = 90 →
  shelby_eaten = 12 →
  final_jellybeans = 72 →
  final_jellybeans = initial_jellybeans - (samantha_taken + shelby_eaten) + 
    (samantha_taken + shelby_eaten) / 2 →
  samantha_taken = 24 := by
  intro h1 h2 h3 h4
  sorry

#check jellybean_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_problem_l402_40211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_squared_difference_l402_40273

theorem divisibility_of_squared_difference (S : Finset ℕ) : 
  S.card = 43 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 100 ∣ (a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_squared_difference_l402_40273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l402_40247

theorem imaginary_part_of_z (z : ℂ) (h : (4 + 3*Complex.I)*z = -Complex.I) : z.im = -4/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l402_40247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l402_40279

-- Define complex numbers V and Z
def V : ℂ := 2 + 3*Complex.I
def Z : ℂ := 2 - 2*Complex.I

-- Define the current I as V/Z
noncomputable def I : ℂ := V / Z

-- Theorem to prove
theorem current_calculation : I = -1/4 + 5/4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l402_40279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_equals_27_l402_40293

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 3 * Real.exp (x - 1)
  else x ^ 3

-- State the theorem
theorem f_of_f_one_equals_27 : f (f 1) = 27 := by
  -- First, calculate f(1)
  have h1 : f 1 = 3 := by
    -- Evaluate f(1) using the definition
    rw [f]
    simp [Real.exp_zero]
  
  -- Then, calculate f(f(1)) = f(3)
  have h2 : f 3 = 27 := by
    -- Evaluate f(3) using the definition
    rw [f]
    simp
    norm_num
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_equals_27_l402_40293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_l402_40282

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 4 * ((x^2 * (1 + x) / (1 - x))^(1/3))

-- State the theorem
theorem functional_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  f x * 2 * f ((1 - x) / (1 + x)) = 64 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_l402_40282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salami_coverage_is_half_l402_40237

/-- Represents a circular pizza with salami slices -/
structure PizzaWithSalami where
  pizza_diameter : ℝ
  salami_count_across : ℕ
  total_salami_count : ℕ

/-- Calculates the fraction of the pizza covered by salami -/
noncomputable def salami_coverage_fraction (p : PizzaWithSalami) : ℝ :=
  let salami_diameter := p.pizza_diameter / p.salami_count_across
  let salami_radius := salami_diameter / 2
  let salami_area := Real.pi * salami_radius^2
  let total_salami_area := p.total_salami_count • salami_area
  let pizza_radius := p.pizza_diameter / 2
  let pizza_area := Real.pi * pizza_radius^2
  total_salami_area / pizza_area

/-- Theorem stating that the fraction of the pizza covered by salami is 1/2 -/
theorem salami_coverage_is_half (p : PizzaWithSalami) 
    (h1 : p.pizza_diameter = 16)
    (h2 : p.salami_count_across = 8)
    (h3 : p.total_salami_count = 32) : 
    salami_coverage_fraction p = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salami_coverage_is_half_l402_40237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_from_tan_l402_40298

theorem cos_from_tan (A : ℝ) :
  Real.tan A = -5/12 → Real.cos A = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_from_tan_l402_40298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_evaluation_l402_40248

noncomputable def product_expression : ℤ :=
  (List.range 8).foldl (fun acc n => acc * (⌊-(n : ℝ) - 0.5⌋ * ⌈(n : ℝ) + 0.5⌉)) 1

theorem product_evaluation :
  product_expression = -1625702400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_evaluation_l402_40248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l402_40249

/-- The area of the shape enclosed by y = x², y = x, and y = 2x is 7/6 -/
theorem area_enclosed_by_curves : 
  ∃ (S : ℝ), 
    (S = ∫ (x : ℝ) in Set.Icc 0 1, (2*x - x) + ∫ (x : ℝ) in Set.Icc 1 2, (2*x - x^2)) ∧
    (S = 7/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l402_40249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_quotient_value_l402_40295

theorem greatest_quotient_value (n : ℕ) (x : ℝ) (h : 0 < x ∧ x < 1) :
  (1 - x^n - (1-x)^n) / (x*(1-x)^n + (1-x)*x^n) ≤ 2^n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_quotient_value_l402_40295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_tangent_property_l402_40262

/-- A triangle with sides satisfying specific conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  cond1 : a^2 = 3 * (b^2 - c^2)
  cond2 : b^2 = 5 * (a^2 - c^2)
  cond3 : c^2 = 7 * (b^2 - a^2)

/-- The angles of the triangle -/
noncomputable def angles (t : SpecialTriangle) : Fin 3 → ℝ :=
  λ i => match i with
    | 0 => Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
    | 1 => Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c))
    | 2 => Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

/-- The theorem stating the special property of the triangle's tangents -/
theorem special_triangle_tangent_property (t : SpecialTriangle) :
  ∃ (i j k : Fin 3) (hi : i ≠ j) (hj : j ≠ k) (hk : k ≠ i),
    Real.tan (angles t i) = (Real.tan (angles t j) + Real.tan (angles t k)) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_tangent_property_l402_40262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_18pi_l402_40235

-- Define the diameter of each semicircle in inches
def semicircle_diameter : ℝ := 3

-- Define the length of the pattern in inches
def pattern_length : ℝ := 24

-- Define the number of rows (top and bottom)
def num_rows : ℕ := 2

-- Function to calculate the area of the shaded region
noncomputable def shaded_area (diameter : ℝ) (length : ℝ) (rows : ℕ) : ℝ :=
  let num_semicircles := (length / diameter) * (rows : ℝ)
  let semicircle_area := (Real.pi * (diameter / 2)^2) / 2
  num_semicircles * semicircle_area

-- Theorem statement
theorem shaded_area_equals_18pi :
  shaded_area semicircle_diameter pattern_length num_rows = 18 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_18pi_l402_40235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_a_greater_b_ratio_equality_values_l402_40242

def a : ℕ+ → ℝ := sorry
def b : ℕ+ → ℝ := sorry
def S : ℕ+ → ℝ := sorry
def T : ℕ+ → ℝ := sorry

axiom a_1 : a 1 = 1
axiom S_2 : S 2 = 4
axiom S_relation : ∀ n : ℕ+, 3 * S (n + 1) = 2 * S n + S (n + 2) + a n
axiom b_arithmetic : ∃ d : ℝ, ∀ n : ℕ+, b (n + 1) = b n + d
axiom S_greater_T : ∀ n : ℕ+, S n > T n
axiom b_geometric : ∃ r : ℝ, ∀ n : ℕ+, b (n + 1) = r * b n
axiom b_1_eq_a_1 : b 1 = a 1
axiom b_2_eq_a_2 : b 2 = a 2

theorem a_formula : ∀ n : ℕ+, a n = 2 * n - 1 := by sorry

theorem a_greater_b : ∀ n : ℕ+, a n > b n := by sorry

theorem ratio_equality_values : 
  ∀ n : ℕ+, (∃ k : ℕ+, (a n + 2 * T n) / (b n + 2 * S n) = a k) ↔ (n = 1 ∨ n = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_a_greater_b_ratio_equality_values_l402_40242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_intersection_range_l402_40210

-- Define the circle C
def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 5}

-- Define the lines
def line_l₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}
def line_l₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.sqrt 3 * p.1 - p.2 + 1 - Real.sqrt 3 = 0}
def line_l₃ (m a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m * p.1 - p.2 + Real.sqrt a + 1 = 0}

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

theorem circle_equation_and_intersection_range :
  ∃ (center : ℝ × ℝ),
    center ∈ line_l₁ ∧
    (Int.floor center.1 = center.1 ∧ Int.floor center.2 = center.2) ∧
    circle_C center = {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 5} ∧
    M ∈ circle_C center ∧ N ∈ circle_C center ∧
    M ∈ line_l₂ ∧ N ∈ line_l₂ ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = 17 ∧
    ∀ (a : ℝ), (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ p ∈ line_l₃ m a) ↔ 0 ≤ a ∧ a ≤ 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_intersection_range_l402_40210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_coefficient_value_l402_40292

noncomputable def binomial_expansion (x : ℝ) := (x + 2 / x^2)^5

theorem coefficient_of_x_squared (x : ℝ) : 
  ∃ (c : ℝ), binomial_expansion x = c * x^2 + (λ y ↦ (binomial_expansion y - c * y^2)) x :=
by
  sorry

theorem coefficient_value : 
  ∃ (c : ℝ), (∀ x : ℝ, binomial_expansion x = c * x^2 + (λ y ↦ (binomial_expansion y - c * y^2)) x) ∧ c = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_coefficient_value_l402_40292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_midpoint_d_value_l402_40297

/-- The value of d for a line 2x + 3y = d that intersects the midpoint of a line segment -/
theorem line_intersects_midpoint_d_value :
  let point1 : ℝ × ℝ := (3, 4)
  let point2 : ℝ × ℝ := (7, 10)
  let midpoint : ℝ × ℝ := ((point1.1 + point2.1) / 2, (point1.2 + point2.2) / 2)
  let d : ℝ := 2 * midpoint.1 + 3 * midpoint.2
  d = 31 := by
  -- Proof goes here
  sorry

#check line_intersects_midpoint_d_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_midpoint_d_value_l402_40297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_in_triangle_l402_40217

theorem cos_B_in_triangle (A B C : ℝ) : 
  A = 120 * Real.pi / 180 →
  B = 45 * Real.pi / 180 →
  C = 15 * Real.pi / 180 →
  A + B + C = Real.pi →
  Real.cos B = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_in_triangle_l402_40217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_investment_l402_40281

def investment_problem (raghu_investment : ℝ) : Prop :=
  let trishul_investment := 0.9 * raghu_investment
  let vishal_investment := 1.1 * trishul_investment
  raghu_investment + trishul_investment + vishal_investment = 6647 ∧
  abs (raghu_investment - 2299.65) < 0.01

theorem solve_investment : ∃ raghu_investment : ℝ, investment_problem raghu_investment := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_investment_l402_40281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_complement_A_B_l402_40271

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for (∁A) ∪ B
theorem union_complement_A_B : (Set.univ \ A) ∪ B = {x : ℝ | x ≤ 2 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_complement_A_B_l402_40271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l402_40204

/-- Given a parabola and a line intersecting it, with specific conditions, 
    prove the values of p and k. -/
theorem parabola_line_intersection 
  (p : ℝ) 
  (k : ℝ) 
  (h_p_pos : p > 0) 
  (C : Set (ℝ × ℝ)) 
  (l : Set (ℝ × ℝ)) 
  (A B : ℝ × ℝ) 
  (h_C : C = {(x, y) | x^2 = 2*p*y}) 
  (h_l : l = {(x, y) | y = k*x + 2}) 
  (h_intersect : A ∈ C ∩ l ∧ B ∈ C ∩ l) 
  (h_perpendicular : (A.1 * B.1 + A.2 * B.2 = 0)) 
  (P : ℝ × ℝ) 
  (h_P : P = (2, 2)) 
  (h_complementary : 
    (A.2 - P.2) / (A.1 - P.1) + (B.2 - P.2) / (B.1 - P.1) = 0) : 
  p = 1 ∧ k = -2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l402_40204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l402_40201

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y+1)^2 = 5

-- Define the line
def line_eq (x y : ℝ) : Prop := 2*x - y + 9 = 0

-- Define the distance function between a point (x, y) and the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2*x - y + 9) / Real.sqrt (2^2 + (-1)^2)

-- Statement of the theorem
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_eq x y ∧ 
    (∀ (x' y' : ℝ), circle_eq x' y' → distance_to_line x y ≥ distance_to_line x' y') ∧
    distance_to_line x y = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l402_40201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_proof_l402_40288

theorem complex_division_proof : 
  (1 + Complex.I) / (-2 * Complex.I) = -(1/2 : ℂ) + (1/2 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_proof_l402_40288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_decomposition_l402_40274

-- Define a structure for functions with domain symmetric about the origin
structure SymmetricDomainFunction (α : Type) [Ring α] where
  f : α → α
  symmetric : ∀ x, f x = f (-x)

-- Define properties for odd and even functions
def IsOdd {α : Type} [Ring α] (g : α → α) : Prop := ∀ x, g (-x) = -g x
def IsEven {α : Type} [Ring α] (h : α → α) : Prop := ∀ x, h (-x) = h x

-- State the theorem
theorem odd_even_decomposition
  {α : Type} [Field α]
  (f : SymmetricDomainFunction α)
  (g h : α → α)
  (h1 : ∀ x, f.f x = g x + h x)
  (h2 : IsOdd g)
  (h3 : IsEven h) :
  g = λ x ↦ (f.f x - f.f (-x)) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_decomposition_l402_40274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l402_40205

/-- Circle M with center (1, -2) and radius 1 -/
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 1

/-- Circle N with center (2, 2) and radius 3 -/
def circle_N (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 9

/-- Distance between the centers of circles M and N -/
noncomputable def center_distance : ℝ := Real.sqrt 17

/-- Function to calculate the number of common tangents between two circles -/
def number_of_common_tangents (c1 c2 : ℝ → ℝ → Prop) : ℕ := sorry

/-- Theorem stating that the number of common tangents between circles M and N is 4 -/
theorem common_tangents_count :
  (∀ x y : ℝ, ¬(circle_M x y ∧ circle_N x y)) →  -- Circles are disjoint
  center_distance > 4 →                         -- Centers are further apart than sum of radii
  number_of_common_tangents circle_M circle_N = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l402_40205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l402_40265

-- Define the sets M and N
def M : Set ℝ := {x | (2 : ℝ)^(x + 1) > 1}
def N : Set ℝ := {x | x > 0 ∧ Real.log x ≤ 1}

-- Statement to prove
theorem intersection_M_N : M ∩ N = Set.Ioc 0 (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l402_40265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_triangles_l402_40246

theorem area_ratio_of_triangles (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hpqr_sum : p + q + r = 3/4) (hpqr_sum_sq : p^2 + q^2 + r^2 = 1/2) :
  let area_ratio := 1 - (p + q + r) + (p*q + q*r + r*p)
  area_ratio = 385/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_triangles_l402_40246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l402_40276

-- Define the participants
inductive Participant
| Olya
| Oleg
| Pasha

-- Define the possible rankings
inductive Ranking
| First
| Second
| Third

-- Define a function to represent the actual ranking of each participant
def actual_ranking : Participant → Ranking := sorry

-- Define a function to represent whether a participant is telling the truth
def is_truthful : Participant → Prop := sorry

-- Define a function to represent whether a ranking is odd
def is_odd_ranking : Ranking → Bool
  | Ranking.First => true
  | Ranking.Third => true
  | _ => false

-- Define a function to represent whether a participant is a boy
def is_boy : Participant → Bool
  | Participant.Olya => false
  | _ => true

-- State the theorem
theorem competition_result :
  (∀ p, is_truthful p ↔ (
    (actual_ranking p = Ranking.First) ∧
    (p = Participant.Olya → ∀ r, is_odd_ranking r = true → ∀ p', actual_ranking p' = r → is_boy p' = true) ∧
    (p = Participant.Oleg → ¬(∀ r, is_odd_ranking r = true → ∀ p', actual_ranking p' = r → is_boy p' = true))
  )) →
  (actual_ranking Participant.Oleg = Ranking.First ∧
   actual_ranking Participant.Pasha = Ranking.Second ∧
   actual_ranking Participant.Olya = Ranking.Third) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l402_40276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_area_l402_40218

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt ((e.a^2 - e.b^2) / e.a^2)

def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def intersects (l : Line) (e : Ellipse) : Prop :=
  ∃ (p : Point), on_ellipse p e ∧ l.a * p.x + l.b * p.y + l.c = 0

theorem ellipse_equation_and_max_area 
  (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 6 / 3)
  (h_point : on_ellipse ⟨-3, -1⟩ e)
  (l : Line)
  (h_line : l.a = 1 ∧ l.b = -1 ∧ l.c = -2)
  (h_intersect : intersects l e) :
  (e.a^2 = 12 ∧ e.b^2 = 4) ∧
  (∃ (max_area : ℝ), max_area = 9 ∧ 
    ∀ (p : Point), on_ellipse p e → 
      ∃ (area : ℝ), area ≤ max_area) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_area_l402_40218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l402_40214

theorem polynomial_problem (p : ℝ → ℝ) 
  (h1 : p 1 = 3)
  (h2 : p 2 = 8)
  (h3 : p 3 = 15)
  (h4 : p 4 = 24)
  (h_poly : ∀ x, p x = x^2 + 2*x + 1) :
  (∀ x, p x = x^2 + 2*x + 1) ∧ p 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l402_40214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclone_damage_conversion_l402_40239

/-- Converts Indian rupees to British pounds based on the given exchange rate -/
noncomputable def rupees_to_pounds (rupees : ℝ) (exchange_rate : ℝ) : ℝ :=
  rupees / exchange_rate

/-- Theorem stating the conversion of cyclone damage from Indian rupees to British pounds -/
theorem cyclone_damage_conversion :
  let damage_rupees : ℝ := 75000000
  let exchange_rate : ℝ := 75
  rupees_to_pounds damage_rupees exchange_rate = 1000000 := by
  -- Unfold the definitions
  unfold rupees_to_pounds
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclone_damage_conversion_l402_40239
