import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l447_44709

-- Define the constants for the containers' dimensions
def john_diameter : ℝ := 8
def john_height : ℝ := 15
def emma_diameter : ℝ := 10
def emma_height : ℝ := 12

-- Define a function to calculate the volume of a cylinder
noncomputable def cylinder_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * height

-- Theorem statement
theorem container_volume_ratio :
  (cylinder_volume john_diameter john_height) / (cylinder_volume emma_diameter emma_height) = 4/5 := by
  -- Unfold the definition of cylinder_volume
  unfold cylinder_volume
  -- Simplify the expression
  simp [john_diameter, john_height, emma_diameter, emma_height]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l447_44709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l447_44715

/-- The sum of the infinite series ∑(n=1 to ∞) 3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1)) -/
noncomputable def infiniteSeries : ℝ := ∑' n, (3 : ℝ) ^ n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

/-- The sum of the infinite series is equal to 1/4 -/
theorem infiniteSeriesSum : infiniteSeries = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l447_44715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l447_44725

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem
theorem complement_of_M :
  Set.compl M = Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l447_44725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_comparison_l447_44711

noncomputable def merchant_a_price (x : ℝ) : ℝ := x

noncomputable def merchant_b_price (x : ℝ) : ℝ := (0.2 * 2 * x + 0.3 * 3 * x + 0.4 * 4 * x) / 3

noncomputable def merchant_a_discounted_price (x : ℝ) : ℝ := 0.9 * x

noncomputable def merchant_b_price_after_discount (x : ℝ) : ℝ := merchant_b_price (merchant_a_discounted_price x)

theorem merchant_comparison (x : ℝ) (h : x > 0) :
  merchant_b_price x < merchant_a_price x ∧
  merchant_a_discounted_price x < merchant_b_price x ∧
  merchant_b_price_after_discount x < merchant_a_discounted_price x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_comparison_l447_44711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculation_l447_44782

theorem arithmetic_calculation : (-8) - 3 + (-6) - (-10) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculation_l447_44782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus60_similar_l447_44788

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  side : ℝ
  side_pos : side > 0

/-- Two rhombuses with 60° angles are similar -/
theorem rhombus60_similar (r1 r2 : Rhombus60) : ∃ (k : ℝ), k > 0 ∧ r1.side = k * r2.side := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus60_similar_l447_44788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_properties_l447_44798

def MagicSquare : Type := Fin 3 → Fin 3 → Fin 9

def is_magic_square (s : MagicSquare) : Prop :=
  (∀ i : Fin 3, (Finset.univ.sum (λ j ↦ s i j)) = 15) ∧
  (∀ j : Fin 3, (Finset.univ.sum (λ i ↦ s i j)) = 15) ∧
  ((Finset.univ.sum (λ i ↦ s i i)) = 15) ∧
  ((Finset.univ.sum (λ i ↦ s i (2 - i))) = 15)

def contains_all_numbers (s : MagicSquare) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, s i j = n

theorem magic_square_properties (s : MagicSquare) 
  (h1 : is_magic_square s) (h2 : contains_all_numbers s) :
  (s 1 1 = 5) ∧ 
  (∃ i : Fin 3, (s i 0 = 1 ∨ s i 2 = 1 ∨ s 0 i = 1 ∨ s 2 i = 1)) ∧
  (∃ i j : Fin 3, (i = 0 ∨ i = 2) ∧ (j = 0 ∨ j = 2) ∧ s i j = 8) := by
  sorry

#check magic_square_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_properties_l447_44798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_229_l447_44712

/-- The number of sides in the regular polygon -/
def n : ℕ := 16

/-- The radius of the circle -/
def r : ℝ := 16

/-- The central angle between consecutive vertices -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The length of each side of the polygon -/
noncomputable def side_length : ℝ := 2 * r * Real.sin (θ / 2)

/-- The length of a diagonal connecting every fourth vertex -/
def diagonal_length : ℝ := 2 * r

/-- The number of diagonals included in the sum -/
def num_diagonals : ℕ := 4

/-- The sum of lengths of all sides and selected diagonals -/
noncomputable def total_sum : ℝ := n * side_length + num_diagonals * diagonal_length

/-- Theorem stating the sum of a, b, and c equals 229 -/
theorem sum_abc_equals_229 : ∃ (a b c : ℕ), 
  a > 0 ∧ b ≥ 0 ∧ c > 0 ∧ 
  (total_sum = a + b * Real.sqrt 2 + c * Real.sqrt 4) ∧
  a + b + c = 229 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_229_l447_44712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_grazing_months_l447_44726

/-- Represents the number of months b put his oxen for grazing -/
def b_months (x : ℚ) : Prop := x = 5

/-- The total rent of the pasture -/
def total_rent : ℚ := 175

/-- c's share of the rent -/
def c_share : ℚ := 45

/-- Calculates the share of rent based on oxen and months -/
def rent_share (oxen : ℚ) (months : ℚ) : ℚ := oxen * months

/-- The theorem stating the number of months b put his oxen for grazing -/
theorem b_grazing_months : b_months 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_grazing_months_l447_44726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l447_44770

def is_valid_arrangement (a b c d : Nat) : Prop :=
  ({a, b, c, d} : Finset Nat) = {1, 2, 3, 4} ∧
  (a = 1 ∨ b ≠ 1 ∨ c = 2 ∨ d ≠ 4) ∧
  ¬(a = 1 ∧ b ≠ 1) ∧ ¬(a = 1 ∧ c = 2) ∧ ¬(a = 1 ∧ d ≠ 4) ∧
  ¬(b ≠ 1 ∧ c = 2) ∧ ¬(b ≠ 1 ∧ d ≠ 4) ∧ ¬(c = 2 ∧ d ≠ 4)

theorem valid_arrangements_count :
  ∃! (arrangements : List (Nat × Nat × Nat × Nat)),
    arrangements.length = 6 ∧
    (∀ (a b c d : Nat), (a, b, c, d) ∈ arrangements ↔ is_valid_arrangement a b c d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l447_44770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l447_44731

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Predicate for two circles intersecting -/
def intersecting (c1 c2 : Circle) : Prop :=
  let d := distance c1.center c2.center
  d > abs (c1.radius - c2.radius) ∧ d < c1.radius + c2.radius

/-- The first circle: (x-1)^2 + y^2 = 1 -/
def circle1 : Circle :=
  { center := (1, 0), radius := 1 }

/-- The second circle: x^2 + (y-1)^2 = 2 -/
noncomputable def circle2 : Circle :=
  { center := (0, 1), radius := Real.sqrt 2 }

/-- Theorem stating that the two given circles are intersecting -/
theorem circles_intersect : intersecting circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l447_44731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_concentration_is_35_percent_l447_44733

-- Define the vessels and their properties
noncomputable def vessel1_capacity : ℝ := 2
noncomputable def vessel1_alcohol_percentage : ℝ := 25
noncomputable def vessel2_capacity : ℝ := 6
noncomputable def vessel2_alcohol_percentage : ℝ := 50
noncomputable def total_liquid : ℝ := 8
noncomputable def final_vessel_capacity : ℝ := 10

-- Define the function to calculate the new concentration
noncomputable def new_concentration : ℝ :=
  let vessel1_alcohol := vessel1_capacity * (vessel1_alcohol_percentage / 100)
  let vessel2_alcohol := vessel2_capacity * (vessel2_alcohol_percentage / 100)
  let total_alcohol := vessel1_alcohol + vessel2_alcohol
  let water_added := final_vessel_capacity - total_liquid
  (total_alcohol / final_vessel_capacity) * 100

-- Theorem statement
theorem new_concentration_is_35_percent :
  new_concentration = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_concentration_is_35_percent_l447_44733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l447_44717

/-- The equation of circle D -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 4 = -y^2 + 6*x + 16

/-- The center of circle D -/
def center : ℝ × ℝ := (3, 2)

/-- The radius of circle D -/
noncomputable def radius : ℝ := Real.sqrt 33

theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - 3)^2 + (y - 2)^2 = 33) ∧
  (center.1 + center.2 + radius = 5 + Real.sqrt 33) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l447_44717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l447_44766

/-- An acute-angled triangle with sides a, b, and c. -/
def acute_angled_triangle (a b c : ℝ) : Prop := sorry

/-- A perpendicular from the vertex opposite side a divides a into two segments,
    with p being the length of the smaller segment. -/
def perpendicular_divides_side (a p : ℝ) : Prop := sorry

/-- Theorem about the ratio of sides in an acute-angled triangle with a perpendicular. -/
theorem triangle_side_ratio (a b c p : ℝ) 
  (h_acute : acute_angled_triangle a b c)
  (h_perp : perpendicular_divides_side a p) :
  a / (c + b) = (c - b) / (a - 2 * p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l447_44766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_tangents_l447_44795

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 2*y + 1 = 0

/-- The external point P -/
def P : ℝ × ℝ := (3, 2)

/-- The angle between the two tangents -/
noncomputable def angle_between_tangents : ℝ := sorry

/-- Theorem stating the cosine of the angle between the tangents -/
theorem cosine_of_angle_between_tangents :
  Real.cos angle_between_tangents = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_tangents_l447_44795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l447_44729

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side a
  b : ℝ  -- side b
  c : ℝ  -- side c

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : t.c > t.b)
  (h2 : t.b * t.c * Real.cos t.A = 20)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 10 * Real.sqrt 3)
  (h4 : let O := circumcenter t
        (O.1 * t.b) * (O.1 * t.c) + (O.2 * t.b) * (O.2 * t.c) = -49/6) :
  t.A = Real.pi / 3 ∧ t.a = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l447_44729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l447_44791

open Set
open Function
open Topology

theorem periodic_function_property :
  ∀ d : ℝ, d ∈ Ioc 0 1 →
  (∀ f : ℝ → ℝ, Continuous f ∧ (∀ x ∈ Icc 0 1, f x = f (x + 1)) →
    ∃ x₀ ∈ Icc 0 (1 - d), f x₀ = f (x₀ + d)) ↔
  ∃ k : ℕ, d = 1 / k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l447_44791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l447_44704

def unit_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def M : ℝ × ℝ := (3, 4)

theorem min_distance_to_circle : 
  ∃ q ∈ unit_circle, ∀ r ∈ unit_circle, 
    Real.sqrt ((q.1 - M.1)^2 + (q.2 - M.2)^2) ≤ Real.sqrt ((r.1 - M.1)^2 + (r.2 - M.2)^2) ∧
    Real.sqrt ((q.1 - M.1)^2 + (q.2 - M.2)^2) = 4 :=
by sorry

#check min_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l447_44704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_specific_primes_l447_44780

-- Define the two smallest two-digit primes
def smallest_two_digit_prime_1 : ℕ := 11
def smallest_two_digit_prime_2 : ℕ := 13

-- Define the largest three-digit prime
def largest_three_digit_prime : ℕ := 997

-- State the theorem
theorem product_of_specific_primes :
  smallest_two_digit_prime_1 * smallest_two_digit_prime_2 * largest_three_digit_prime = 142571 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_specific_primes_l447_44780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_invariants_l447_44741

-- Define the hyperbola
def hyperbola (x y θ : ℝ) : Prop :=
  x^2 - y^2 / 3 = Real.cos θ^2

-- Define the condition for θ
def θ_condition (θ : ℝ) : Prop :=
  ∀ k : ℤ, θ ≠ k * Real.pi

-- Define the asymptote equation
def asymptote_eq (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Define the eccentricity
def eccentricity : ℝ := 2

theorem hyperbola_invariants (x y θ : ℝ) 
  (h_hyperbola : hyperbola x y θ) (h_θ : θ_condition θ) :
  (asymptote_eq x y ∧ eccentricity = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_invariants_l447_44741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_theorem_l447_44742

def total_books : ℕ := 13
def arithmetic_books : ℕ := 4
def algebra_books : ℕ := 6
def geometry_books : ℕ := 3

theorem book_arrangement_theorem :
  (∀ (n : ℕ), Nat.factorial n = n.factorial) →
  (Nat.factorial total_books = 6227020800) ∧
  (Nat.factorial (total_books - arithmetic_books + 1) * Nat.factorial arithmetic_books = 87091200) ∧
  (Nat.factorial (total_books - arithmetic_books - algebra_books + 2) * Nat.factorial arithmetic_books * Nat.factorial algebra_books = 2073600) ∧
  (Nat.factorial (total_books - arithmetic_books - algebra_books - geometry_books + 3) * Nat.factorial arithmetic_books * Nat.factorial algebra_books * Nat.factorial geometry_books = 622080) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_theorem_l447_44742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_three_polynomials_l447_44730

-- Define a helper function to count real roots (this is a placeholder)
noncomputable def number_of_real_roots (f : ℝ → ℝ) : ℕ :=
  sorry

theorem max_real_roots_three_polynomials (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (n : ℕ), n ≤ 4 ∧
  (∀ (m : ℕ), 
    (∃ (x y z : ℕ), 
      x = (number_of_real_roots (λ t ↦ a*t^2 + b*t + c)) ∧
      y = (number_of_real_roots (λ t ↦ b*t^2 + c*t + a)) ∧
      z = (number_of_real_roots (λ t ↦ c*t^2 + a*t + b)) ∧
      m = x + y + z) →
    m ≤ n) ∧
  ∃ (x y z : ℕ), 
    x = (number_of_real_roots (λ t ↦ a*t^2 + b*t + c)) ∧
    y = (number_of_real_roots (λ t ↦ b*t^2 + c*t + a)) ∧
    z = (number_of_real_roots (λ t ↦ c*t^2 + a*t + b)) ∧
    n = x + y + z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_three_polynomials_l447_44730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_of_3_mod_197_l447_44751

theorem modular_inverse_of_3_mod_197 : ∃ x : ℕ, x ≤ 196 ∧ (3 * x) % 197 = 1 :=
by
  use 66
  constructor
  · norm_num
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_of_3_mod_197_l447_44751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_combination_l447_44786

theorem max_value_trig_combination :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧ ∀ x : ℝ, 2 * Real.cos x + 3 * Real.sin x ≤ M :=
by
  -- We'll use M = √13 as our maximum value
  let M := Real.sqrt 13
  
  -- Prove that this M satisfies the conditions
  have h1 : M = Real.sqrt 13 := rfl
  
  have h2 : ∀ x : ℝ, 2 * Real.cos x + 3 * Real.sin x ≤ M := by
    intro x
    -- The proof steps would go here, but we'll use sorry for now
    sorry
  
  -- Combine the two parts to prove the existence
  exact ⟨M, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_combination_l447_44786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_divisors_congruent_to_one_mod_four_l447_44761

theorem more_divisors_congruent_to_one_mod_four (n : ℕ) :
  let divisors := Finset.filter (fun d => d ∣ n^2) (Finset.range (n^2 + 1))
  let divisors_mod_1 := Finset.filter (fun d => d % 4 = 1) divisors
  let divisors_mod_3 := Finset.filter (fun d => d % 4 = 3) divisors
  n > 0 → Finset.card divisors_mod_1 > Finset.card divisors_mod_3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_divisors_congruent_to_one_mod_four_l447_44761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l447_44750

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 3 + Real.cos (2 * x) - Real.cos x ^ 2 - Real.sin x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5/27 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l447_44750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_tangent_l447_44784

-- Define a triangle ABC with given side lengths
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (bc : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 14)
  (ca : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15)

-- Define a point P inside the triangle
def InsidePoint (t : Triangle) := { P : ℝ × ℝ // 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
  P = (a * t.A.1 + b * t.B.1 + c * t.C.1, a * t.A.2 + b * t.B.2 + c * t.C.2) }

-- Define the angle function (placeholder)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the property of congruent angles
def CongruentAngles (t : Triangle) (P : InsidePoint t) :=
  ∃ ω : ℝ, 
    angle t.A P.val t.B = ω ∧
    angle t.B P.val t.C = ω ∧
    angle t.C P.val t.A = ω

-- State the theorem
theorem special_point_tangent (t : Triangle) (P : InsidePoint t) 
  (h : CongruentAngles t P) : 
  Real.tan (angle t.A P.val t.B) = 168 / 295 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_tangent_l447_44784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BFDE_eq_4_32_l447_44734

/-- Given a rhombus ABCD with diagonals d₁ and d₂, and altitudes BE and BF from vertex B,
    this function calculates the area of quadrilateral BFDE. -/
noncomputable def area_BFDE (d₁ d₂ : ℝ) : ℝ :=
  let S := d₁ * d₂ / 2
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  let BE := S / side
  let DE := Real.sqrt (d₁^2 - BE^2)
  BE * DE

/-- Theorem stating that for a rhombus with diagonals 3 cm and 4 cm,
    the area of quadrilateral BFDE is 4.32 cm². -/
theorem area_BFDE_eq_4_32 :
  area_BFDE 3 4 = 4.32 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BFDE_eq_4_32_l447_44734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_from_cos_sum_and_tan_sum_l447_44776

theorem sin_sum_from_cos_sum_and_tan_sum 
  (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = Real.sqrt 2 / 4) 
  (h2 : Real.tan (α + β) = 4 / 3) : 
  Real.sin α + Real.sin β = -Real.sqrt 2 / 2 ∨ Real.sin α + Real.sin β = Real.sqrt 2 / 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_from_cos_sum_and_tan_sum_l447_44776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_point_zero_incorrect_l447_44757

/-- Represents a number with its decimal representation and claimed accuracy --/
structure ApproximateNumber where
  value : Float
  accuracy : Nat

/-- Checks if a number is correctly represented according to its claimed accuracy --/
def isCorrectlyRepresented (n : ApproximateNumber) : Prop :=
  match n.accuracy with
  | 0 => n.value = n.value.floor  -- Units place
  | 1 => n.value = (n.value * 10).round / 10  -- Tenths place
  | 2 => n.value = (n.value * 100).round / 100  -- Hundredths place
  | 3 => n.value = (n.value * 1000).round / 1000  -- Thousandths place
  | _ => False  -- We don't consider other cases for this problem

/-- The theorem stating that 5.0 is incorrectly represented when claimed to be accurate to the units place --/
theorem five_point_zero_incorrect : 
  ¬(isCorrectlyRepresented { value := 5.0, accuracy := 0 }) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_point_zero_incorrect_l447_44757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_number_l447_44735

def is_nine_digit (n : ℕ) : Prop := 
  100000000 ≤ n ∧ n < 1000000000

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 n) → d ≠ 0

def all_digits (n : ℕ) : Finset ℕ :=
  (Nat.digits 10 n).toFinset

def all_remainders_different (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ all_digits n → d₂ ∈ all_digits n → d₁ ≠ d₂ → 
    n % d₁ ≠ n % d₂

theorem no_such_number : 
  ¬ ∃ n : ℕ, is_nine_digit n ∧ has_no_zero_digit n ∧ all_remainders_different n :=
by
  sorry

#check no_such_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_number_l447_44735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ae_length_l447_44789

-- Define the quadrilateral ABCD and point E
structure Quadrilateral :=
  (A B C D E : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let d := Euclidean.dist
  d q.A q.B = 12 ∧
  d q.C q.D = 15 ∧
  d q.A q.C = 20 ∧
  -- Placeholder for intersection point check
  true ∧
  -- Placeholder for area ratio
  true ∧
  -- Placeholder for perimeter ratio
  true

-- Define the theorem
theorem quadrilateral_ae_length (q : Quadrilateral) :
  is_valid_quadrilateral q →
  Euclidean.dist q.A q.E = 20 * Real.sqrt 15 / (5 + Real.sqrt 15) :=
by
  sorry

#check quadrilateral_ae_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ae_length_l447_44789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_CD_l447_44739

noncomputable section

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (-2, -1)
def D : ℝ × ℝ := (2, 2)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_CD : ℝ × ℝ := (D.1 - C.1, D.2 - C.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / vector_magnitude w

theorem projection_AB_CD : vector_projection vector_AB vector_CD = 11/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_CD_l447_44739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_longest_side_l447_44759

theorem median_to_longest_side (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  let m := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  m = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_longest_side_l447_44759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_negative_five_halves_l447_44787

theorem sum_of_solutions_is_negative_five_halves :
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, |x - 1| = 3 * |x + 1|) ∧ 
    (∀ y : ℝ, |y - 1| = 3 * |y + 1| → y ∈ S) ∧
    (S.sum id) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_negative_five_halves_l447_44787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_unit_circle_l447_44748

noncomputable section

open Real

theorem triangle_in_unit_circle 
  (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC is inscribed in a unit circle
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  sin A / a = sin B / b ∧ sin B / b = sin C / c →
  -- Given condition
  2 * a * cos A = c * cos B + b * cos C →
  -- Additional condition
  b^2 + c^2 = 4 →
  -- Prove cos A = 1/2
  cos A = 1/2 ∧
  -- Prove area of triangle ABC is √3/4
  (1/2) * b * c * sin A = sqrt 3 / 4 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_unit_circle_l447_44748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_bowls_theorem_l447_44783

theorem marble_bowls_theorem (capacity_ratio : ℚ) (second_bowl_marbles : ℕ) : 
  capacity_ratio = 3/4 →
  second_bowl_marbles = 600 →
  (capacity_ratio * second_bowl_marbles) + second_bowl_marbles = 1050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_bowls_theorem_l447_44783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l447_44705

noncomputable section

-- Define the hyperbola C
def C : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) - (p.2^2 / 16) = 1}

-- Define the center, left focus, and eccentricity
def center : ℝ × ℝ := (0, 0)
noncomputable def leftFocus : ℝ × ℝ := (-2 * Real.sqrt 5, 0)
noncomputable def eccentricity : ℝ := Real.sqrt 5

-- Define vertices
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)

-- Define the line passing through (-4, 0)
def intersectingLine (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 = m * p.2 - 4}

-- Define the point P
noncomputable def P (M N : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual computation of P is not needed for the statement

-- State the theorem
theorem hyperbola_properties :
  (∀ p ∈ C, (p.1^2 / 4) - (p.2^2 / 16) = 1) ∧
  (∀ m : ℝ, ∀ M N : ℝ × ℝ,
    M ∈ C ∧ N ∈ C ∧
    M ∈ intersectingLine m ∧ N ∈ intersectingLine m ∧
    M.1 < 0 ∧ M.2 > 0 →
    (P M N).1 = -1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l447_44705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l447_44749

theorem point_in_third_quadrant (x : ℝ) :
  (Real.sin x - Real.cos x < 0 ∧ -3 < 0) ↔ x ∈ Set.Ioo (-3 * Real.pi / 4) (Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l447_44749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l447_44744

/-- The area of a trapezoid given circle radius, longer base, and base angle -/
noncomputable def trapezoid_area (r : ℝ) (b : ℝ) (θ : ℝ) : ℝ :=
  let y := 10 * (Real.sqrt 2 - 1)
  let h := 5 * Real.sqrt 2 * (3 - Real.sqrt 2)
  (b + y) * h / 2

/-- The area of an isosceles trapezoid circumscribed around a circle -/
theorem isosceles_trapezoid_area (r : ℝ) (b : ℝ) (θ : ℝ) (h : 0 < r) :
  r = 5 ∧ b = 20 ∧ θ = π / 4 →
  ∃ (A : ℝ), A = trapezoid_area r b θ ∧ A = 90 * Real.sqrt 2 - 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l447_44744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_flips_l447_44762

/-- The probability of getting the first head on the nth flip for a fair coin. -/
noncomputable def prob_first_head_on_nth_flip (n : ℕ) : ℝ := (1/2) ^ n

/-- The probability that all four players get their first head on the nth flip. -/
noncomputable def prob_all_four_on_nth_flip (n : ℕ) : ℝ := (prob_first_head_on_nth_flip n) ^ 4

/-- The sum of probabilities for all possible numbers of flips. -/
noncomputable def total_probability : ℝ := ∑' n, prob_all_four_on_nth_flip n

/-- Theorem stating that the probability of all four players flipping the same number of times is 1/15. -/
theorem probability_same_flips : total_probability = 1/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_flips_l447_44762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l447_44796

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- The given trapezium -/
def givenTrapezium : Trapezium :=
  { side1 := 24
    side2 := 18
    height := 15 }

theorem trapezium_properties :
  (area givenTrapezium = 315) ∧
  (max givenTrapezium.side1 givenTrapezium.side2 = 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l447_44796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_after_ten_steps_l447_44723

def transform (triple : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ :=
  let (x, y, z) := triple
  (y + z - x, z + x - y, x + y - z)

def iterate_transform (n : ℕ) (triple : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => triple
  | n + 1 => iterate_transform n (transform triple)

theorem negative_after_ten_steps
  (a b c : ℤ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 2013) :
  ∃ x, x < 0 ∧ (x = (iterate_transform 10 (a, b, c)).1 ∨
               x = (iterate_transform 10 (a, b, c)).2.1 ∨
               x = (iterate_transform 10 (a, b, c)).2.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_after_ten_steps_l447_44723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l447_44792

-- Define the line
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- State the theorem
theorem min_distance_line_circle :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    line_eq x₁ y₁ →
    circle_eq x₂ y₂ →
    d ≤ Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l447_44792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_inequality_solution_set_l447_44764

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4|

-- Part 1: Prove that a = ±2
theorem min_value_implies_a (a : ℝ) : 
  (∀ x : ℝ, f (2*x + a) + f (2*x - a) ≥ 4) ∧ 
  (∃ x : ℝ, f (2*x + a) + f (2*x - a) = 4) → 
  a = 2 ∨ a = -2 := by
  sorry

-- Part 2: Prove the solution set of the inequality
theorem inequality_solution_set (x : ℝ) :
  f x > 1 - (1/2) * x ↔ x > -2 ∨ x < -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_inequality_solution_set_l447_44764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_inclination_range_l447_44760

noncomputable def inclination_angle (k : ℝ) := Real.arctan k * (180 / Real.pi)

theorem slope_to_inclination_range (k : ℝ) (h : -1 ≤ k ∧ k < 1) :
  let α := inclination_angle k
  (0 ≤ α ∧ α < 45) ∨ (135 ≤ α ∧ α < 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_inclination_range_l447_44760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_curve_decreasing_function_l447_44732

-- Define the functions
noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (a b : ℝ) (x : ℝ) := (1/2) * a * x + b
noncomputable def h (m : ℝ) (x : ℝ) := m * (x - 1) / (x + 1) - Real.log x

-- Theorem for part 1
theorem tangent_curve (a b : ℝ) :
  (∀ x, f x ≤ g a b x) ∧ (f 2 = g a b 2) ∧ (deriv f 2 = deriv (g a b) 2) →
  ∀ x, g a b x = x + Real.log 2 - 1 := by
  sorry

-- Theorem for part 2
theorem decreasing_function (m : ℝ) :
  (∀ x > 0, ∀ y > x, h m y < h m x) →
  m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_curve_decreasing_function_l447_44732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_rational_l447_44778

theorem triangle_area_rational (y₁ y₂ y₃ : ℤ) :
  ∃ (q : ℚ), (1 / 2 : ℚ) * |(y₁ + 1) * (y₂ - y₃) + (y₂ - 1) * (y₃ - y₁) + (y₃ + 2) * (y₁ - y₂)| = q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_rational_l447_44778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l447_44700

-- Define the variables as noncomputable
noncomputable def a : ℝ := Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
noncomputable def b : ℝ := -Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
noncomputable def d : ℝ := -Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5

-- State the theorem
theorem sum_of_reciprocals_squared :
  (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l447_44700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_minus_alice_difference_l447_44745

/-- The amount paid by Alice -/
noncomputable def alice_paid : ℝ := 150

/-- The amount paid by Bob -/
noncomputable def bob_paid : ℝ := 180

/-- The amount paid by Cindy -/
noncomputable def cindy_paid : ℝ := 210

/-- The amount paid by Dan -/
noncomputable def dan_paid : ℝ := 120

/-- The total amount paid by all four people -/
noncomputable def total_paid : ℝ := alice_paid + bob_paid + cindy_paid + dan_paid

/-- The fair share that each person should pay -/
noncomputable def fair_share : ℝ := total_paid / 4

/-- The amount Alice gives to Dan -/
noncomputable def a : ℝ := fair_share - alice_paid

/-- The amount Bob gives to Dan -/
noncomputable def b : ℝ := bob_paid - fair_share

/-- The amount Cindy gives to Dan -/
noncomputable def c : ℝ := cindy_paid - fair_share

theorem bob_minus_alice_difference : b - a = 7.5 := by
  -- Expand definitions
  unfold b a fair_share total_paid
  -- Simplify the expression
  simp [alice_paid, bob_paid, cindy_paid, dan_paid]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_minus_alice_difference_l447_44745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_power_digits_l447_44701

/-- A function that returns true if the number consists of the digits 1, 2, 3, 3, 7, 9 in any order -/
def has_required_digits (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
    Multiset.toFinset {a, b, c, d, e, f} = Multiset.toFinset {1, 2, 3, 3, 7, 9}

theorem fifth_power_digits :
  ∃! (n : ℕ), has_required_digits (n^5) ∧ n = 13 :=
by
  sorry

#check fifth_power_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_power_digits_l447_44701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_marks_correct_l447_44772

/-- Calculates the marks in the fifth subject given the marks in four subjects and the average of all five subjects. -/
def calculate_fifth_subject_marks (english_marks mathematics_marks physics_marks chemistry_marks : ℕ) (average_marks : ℚ) : ℕ :=
  let total_known_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks
  let total_marks := (average_marks * 5).floor.toNat
  if total_marks ≥ total_known_marks then
    total_marks - total_known_marks
  else
    0

/-- Theorem stating that the calculated marks for the fifth subject (Biology) are correct. -/
theorem biology_marks_correct (english_marks mathematics_marks physics_marks chemistry_marks : ℕ) (average_marks : ℚ) :
  calculate_fifth_subject_marks english_marks mathematics_marks physics_marks chemistry_marks average_marks = 65 :=
by
  sorry

#eval calculate_fifth_subject_marks 70 60 78 60 (666/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_marks_correct_l447_44772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangle_with_magnitude_two_l447_44773

open Complex

/-- A complex number z forms an equilateral triangle with 0 and z^3 -/
def formsEquilateralTriangle (z : ℂ) : Prop :=
  Complex.abs (z - 0) = Complex.abs (z^3 - 0) ∧
  Complex.abs (z - 0) = Complex.abs (z^3 - z) ∧
  Complex.abs (z^3 - 0) = Complex.abs (z^3 - z)

/-- There are no nonzero complex numbers z such that z forms an equilateral triangle with 0 and z^3, and |z| = 2 -/
theorem no_equilateral_triangle_with_magnitude_two :
  ¬∃ (z : ℂ), z ≠ 0 ∧ formsEquilateralTriangle z ∧ Complex.abs z = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangle_with_magnitude_two_l447_44773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vectors_sum_of_squared_norms_bound_l447_44737

theorem unit_vectors_sum_of_squared_norms_bound 
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] [Finite n]
  (a b c d : n) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ∃ (m : ℝ), m ≤ 14 ∧ 
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ m ∧
  ∃ (a' b' c' d' : n), 
    ‖a'‖ = 1 ∧ ‖b'‖ = 1 ∧ ‖c'‖ = 1 ∧ ‖d'‖ = 1 ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖a' - d'‖^2 + ‖b' - c'‖^2 + ‖b' - d'‖^2 + ‖c' - d'‖^2 = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vectors_sum_of_squared_norms_bound_l447_44737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_l447_44714

def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (2*k, 3)

theorem perpendicular_vectors_k (k : ℝ) : 
  (a.1 * (2*a.1 + (b k).1) + a.2 * (2*a.2 + (b k).2) = 0) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_l447_44714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_l447_44718

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x - 1) + (x + 2) ^ 0

-- Theorem stating when the expression is meaningful
theorem f_meaningful (x : ℝ) : x ≠ 1 ↔ (∃ y, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_l447_44718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l447_44722

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length crossing_time : Real) :
  train_length = 156 →
  bridge_length = 344.04 →
  crossing_time = 40 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45.0036 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l447_44722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_correct_l447_44706

/-- The volume of a cone-shaped container without a lid, made from a semicircular sheet of iron with radius R. -/
noncomputable def cone_volume (R : ℝ) : ℝ :=
  (Real.sqrt 3 / 24) * Real.pi * R^3

/-- Theorem stating that the volume of the cone-shaped container is correctly calculated by cone_volume. -/
theorem cone_volume_correct (R : ℝ) (h : R > 0) : 
  (1/3) * Real.pi * (R/2)^2 * (Real.sqrt (R^2 - (R/2)^2)) = cone_volume R :=
by
  -- Expand the definition of cone_volume
  unfold cone_volume
  -- Simplify the right-hand side
  simp [Real.sqrt_sq, h]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_correct_l447_44706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophie_picked_correct_number_l447_44747

noncomputable def sophies_number (product theo : ℂ) : ℂ := product / theo

theorem sophie_picked_correct_number (product theo : ℂ) 
  (h1 : product = 80 - 24 * Complex.I) 
  (h2 : theo = 7 + 4 * Complex.I) : 
  sophies_number product theo = 464 / 65 - 488 / 65 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophie_picked_correct_number_l447_44747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_product_l447_44707

-- Define the parabola
noncomputable def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the dot product condition
def dot_product_condition (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = -4

-- Define the area of a triangle
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem parabola_triangle_area_product (a b : ℝ × ℝ) :
  parabola a → parabola b → dot_product_condition a b →
  (triangle_area (0, 0) focus a) * (triangle_area (0, 0) focus b) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_product_l447_44707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polynomial_for_C_l447_44713

/-- C(n) is the number of representations of n as a sum of nonincreasing powers of 2,
    where no power can be used more than three times. -/
def C (n : ℕ+) : ℕ := sorry

/-- Predicate to check if a function is a polynomial -/
def IsPolynomial (f : ℝ → ℝ) : Prop := sorry

/-- The theorem states that there exists a polynomial P such that
    for all positive integers n, C(n) = ⌊P(n)⌋. -/
theorem exists_polynomial_for_C :
  ∃ (P : ℝ → ℝ), IsPolynomial P ∧
    ∀ (n : ℕ+), C n = ⌊P n⌋ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polynomial_for_C_l447_44713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l447_44785

noncomputable def f (x : ℝ) : ℝ := (10*x^4 + 3*x^3 + 7*x^2 + 6*x + 4) / (2*x^4 + 5*x^3 + 4*x^2 + 2*x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l447_44785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l447_44763

-- Define the points A, B, C in R²
variable (A B C : ℝ × ℝ)
-- Define k and p as real numbers
variable (k p : ℝ)

-- Define the conditions
def midpoint_BC (A B C : ℝ × ℝ) (k p : ℝ) : Prop := (k, 0) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def midpoint_AC (A B C : ℝ × ℝ) (k p : ℝ) : Prop := (0, p) = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
def midpoint_AB (A B C : ℝ × ℝ) (k p : ℝ) : Prop := (0, 0) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define distance squared function
def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Theorem statement
theorem triangle_ratio (A B C : ℝ × ℝ) (k p : ℝ)
                       (h1 : midpoint_BC A B C k p)
                       (h2 : midpoint_AC A B C k p)
                       (h3 : midpoint_AB A B C k p) :
  (dist_squared A B + dist_squared A C + dist_squared B C) / (k^2 + p^2) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l447_44763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_fraction_is_one_sixth_l447_44779

-- Define the initial setup
def initial_tea : ℚ := 6
def initial_milk : ℚ := 6
def initial_honey : ℚ := 3
def cup_capacity : ℚ := 12

-- Define the pouring actions
def pour_tea_to_milk (tea : ℚ) : ℚ := tea / 2
def pour_mixture_to_tea (mixture : ℚ) : ℚ := mixture / 2
def pour_to_honey : ℚ := 3

-- Define the fraction of tea in the final mixture
def tea_fraction (final_tea : ℚ) (final_total : ℚ) : ℚ := final_tea / final_total

-- Theorem statement
theorem tea_fraction_is_one_sixth :
  let tea_to_milk := pour_tea_to_milk initial_tea
  let remaining_tea := initial_tea - tea_to_milk
  let mixed_volume := initial_milk + tea_to_milk
  let mixture_to_tea := pour_mixture_to_tea mixed_volume
  let tea_in_mixture := tea_to_milk / mixed_volume * mixture_to_tea
  let final_tea_cup1 := remaining_tea + tea_in_mixture
  let final_volume_cup1 := remaining_tea + mixture_to_tea
  let tea_to_honey := final_tea_cup1 / final_volume_cup1 * pour_to_honey
  let final_volume_cup3 := initial_honey + pour_to_honey
  tea_fraction tea_to_honey final_volume_cup3 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_fraction_is_one_sixth_l447_44779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l447_44793

noncomputable def f (x : ℝ) := (1/6) * x^3 + Real.sin x - x

theorem function_properties :
  let f := f
  ∃ (slope : ℝ),
    (slope = (1/2) * π^2 - 2) ∧
    (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc π (2*π) → x₂ ∈ Set.Icc π (2*π) → 
      |f x₂ - f x₁| ≤ (7/6) * π^3 - π) ∧
    (∀ a : ℝ, (∀ x : ℝ, x ≥ π → f x ≥ a * x^2) ↔ a ≤ π/6 - 1/π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l447_44793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_bounds_l447_44736

noncomputable def x : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => Real.sqrt (x (n + 1) * x n + (n + 2) / 2)

theorem exists_a_bounds : ∃ a : ℝ, ∀ n : ℕ, a * n < x n ∧ x n < a * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_bounds_l447_44736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_album_cost_l447_44777

theorem album_cost : ∃ (cost : ℕ), cost = 35 ∧
  cost - 2 ≥ 0 ∧
  cost - 34 ≥ 0 ∧
  cost - 35 ≥ 0 ∧
  (cost - 2) + (cost - 34) + (cost - 35) < cost :=
by
  -- We claim that the cost is 35
  use 35
  
  -- Now we prove each part of the conjunction
  constructor
  · -- First, we show that 35 = 35 (trivial)
    rfl
  
  constructor
  · -- 35 - 2 ≥ 0
    norm_num
  
  constructor
  · -- 35 - 34 ≥ 0
    norm_num
  
  constructor
  · -- 35 - 35 ≥ 0
    norm_num
  
  -- Finally, we show (35 - 2) + (35 - 34) + (35 - 35) < 35
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_album_cost_l447_44777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dry_grapes_weight_l447_44768

/-- The weight of dry grapes obtained from fresh grapes -/
noncomputable def weight_dry_grapes (fresh_weight : ℝ) (fresh_water_percent : ℝ) (dry_water_percent : ℝ) : ℝ :=
  let solid_weight := fresh_weight * (1 - fresh_water_percent)
  solid_weight / (1 - dry_water_percent)

/-- Theorem stating the weight of dry grapes obtained from 25 kg of fresh grapes -/
theorem dry_grapes_weight :
  weight_dry_grapes 25 0.9 0.2 = 3.125 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval weight_dry_grapes 25 0.9 0.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dry_grapes_weight_l447_44768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l447_44746

theorem log_equation_solution : 
  ∃ y : ℝ, y > 0 ∧ Real.log 16 / Real.log y = Real.log 5 / Real.log 125 ∧ y = 4096 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l447_44746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l447_44728

theorem find_b (a b c : ℕ) (h1 : a * b + b * c - c * a = 0) (h2 : a - c = 101) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : b = 2550 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l447_44728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_in_S_l447_44724

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (x - 8*x^2 + 16*x^3) / (9 - x^2)

-- Define the set of x values for which the expression is nonnegative
def S : Set ℝ := Set.union (Set.Icc (-3) 0) (Set.Icc 0 (1/4))

-- Theorem statement
theorem f_nonnegative_iff_in_S :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_in_S_l447_44724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_management_fee_percentage_max_management_fee_percentage_is_ten_l447_44769

/-- The maximum management fee percentage that satisfies the given conditions -/
theorem max_management_fee_percentage : ℝ :=
  let initial_price : ℝ := 70
  let initial_sales : ℝ := 11.8
  let price_increase (x : ℝ) : ℝ := (initial_price * x / 100) / (1 - x / 100)
  let sales_decrease (x : ℝ) : ℝ := x / 1000
  let second_year_sales (x : ℝ) : ℝ := initial_sales - sales_decrease x
  let second_year_price (x : ℝ) : ℝ := initial_price + price_increase x
  let management_fee (x : ℝ) : ℝ := (second_year_price x * second_year_sales x * x) / 10000
  let management_fee_constraint (x : ℝ) : Prop := management_fee x ≥ 140
  let max_x : ℝ := 10
  have h1 : ∀ x, 0 < x → x ≤ max_x → management_fee_constraint x := by sorry
  have h2 : ∀ x, x > max_x → ¬management_fee_constraint x := by sorry
  max_x

/-- Proof that the maximum management fee percentage is 10 -/
theorem max_management_fee_percentage_is_ten : max_management_fee_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_management_fee_percentage_max_management_fee_percentage_is_ten_l447_44769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_difference_l447_44771

theorem gcd_power_difference (a b m n : ℕ) (h : Nat.gcd a b = 1) :
  Nat.gcd (a^m - b^m) (a^n - b^n) = a^(Nat.gcd m n) - b^(Nat.gcd m n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_difference_l447_44771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equivalent_l447_44738

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2^(x+1) * x^2

-- State the theorem
theorem g_equivalent : ∀ x : ℝ, g x = 2^(x+1) * (x^2 - 4*x - 2) := by
  intro x
  -- Unfold the definition of g
  unfold g
  -- Simplify the right-hand side
  simp [Real.rpow_add, Real.rpow_one]
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equivalent_l447_44738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_on_zero_one_l447_44781

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -Real.log x + a * x^2 + (1 - 2*a) * x + a - 1

theorem f_positive_on_zero_one (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f a x > 0) ↔ a ≥ -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_on_zero_one_l447_44781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_negative_one_point_five_l447_44775

/-- Represents the outcome of rolling a die -/
inductive DieOutcome
| Low  : DieOutcome  -- Represents rolling 1, 2, or 3
| High : DieOutcome  -- Represents rolling 4, 5, or 6

/-- The probability of rolling a low number (1, 2, or 3) -/
noncomputable def prob_low : ℝ := 1/2

/-- The probability of rolling a high number (4, 5, or 6) -/
noncomputable def prob_high : ℝ := 1/2

/-- The financial outcome of rolling a low number -/
def gain_low : ℝ := 2

/-- The financial outcome of rolling a high number -/
def gain_high : ℝ := -5

/-- The expected value of a single roll of the biased die -/
noncomputable def expected_value : ℝ := prob_low * gain_low + prob_high * gain_high

/-- Theorem stating that the expected value of a single roll is -$1.50 -/
theorem expected_value_is_negative_one_point_five :
  expected_value = -1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_negative_one_point_five_l447_44775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_time_allocation_l447_44753

/-- Represents the time spent on type A problems in an examination -/
noncomputable def time_for_type_A (total_time : ℝ) (total_questions : ℕ) (type_A_count : ℕ) : ℝ :=
  let type_B_count := total_questions - type_A_count
  let x := total_time / (2 * type_A_count + type_B_count)
  2 * x * type_A_count

/-- The theorem states that given the examination conditions, 
    the time spent on type A problems is approximately 25.116 minutes -/
theorem exam_time_allocation :
  let total_time : ℝ := 3 * 60 -- 3 hours in minutes
  let total_questions : ℕ := 200
  let type_A_count : ℕ := 15
  abs (time_for_type_A total_time total_questions type_A_count - 25.116) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_time_allocation_l447_44753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_result_l447_44703

structure Reaction where
  agno3 : ℚ
  naoh : ℚ

def limiting_reagent (r : Reaction) : String :=
  if r.agno3 < r.naoh then "AgNO₃" else "NaOH"

def agoh_formed (r : Reaction) : ℚ :=
  min r.agno3 r.naoh

theorem reaction_result (r : Reaction) (h1 : r.agno3 = 1/2) (h2 : r.naoh = 7/20) :
  limiting_reagent r = "NaOH" ∧ agoh_formed r = 7/20 := by
  sorry

#check reaction_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_result_l447_44703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l447_44755

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 9) / (x - 3)

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, x ≠ 3 ∧ f x = 3 * x - 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l447_44755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_123_l447_44756

def f : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | n+3 => f (n+1) + f (n+2)

theorem f_10_equals_123 : f 10 = 123 := by
  -- Compute the values step by step
  have h3 : f 3 = 4 := rfl
  have h4 : f 4 = 7 := rfl
  have h5 : f 5 = 11 := rfl
  have h6 : f 6 = 18 := rfl
  have h7 : f 7 = 29 := rfl
  have h8 : f 8 = 47 := rfl
  have h9 : f 9 = 76 := rfl
  have h10 : f 10 = 123 := rfl
  exact h10

#eval f 10  -- This will evaluate f 10 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_123_l447_44756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inflection_point_of_f_l447_44727

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x) + (1 / 3) * x

def is_inflection_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f'' : ℝ → ℝ), (deriv^[2] f) x₀ = 0

theorem inflection_point_of_f (x₀ y₀ : ℝ) :
  is_inflection_point f x₀ →
  -π/4 < x₀ →
  x₀ < 0 →
  f x₀ = y₀ →
  y₀ = -π/24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inflection_point_of_f_l447_44727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_sum_l447_44758

theorem triangle_angles_sum (A B C : ℝ) : 
  -- A, B, C are angles of a triangle
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- C is obtuse
  Real.pi / 2 < C →
  -- Given equations
  Real.cos A ^ 2 + Real.cos C ^ 2 + 2 * Real.sin A * Real.sin C * Real.cos B = 17 / 9 →
  Real.cos C ^ 2 + Real.cos B ^ 2 + 2 * Real.sin C * Real.sin B * Real.cos A = 16 / 7 →
  -- Conclusion
  ∃ (p q r s : ℕ), 
    -- p, q, r, s are positive integers
    0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧
    -- The equation holds
    Real.cos B ^ 2 + Real.cos A ^ 2 + 2 * Real.sin B * Real.sin A * Real.cos C = (p - q * Real.sqrt r) / s ∧
    -- p+q and s are coprime
    Nat.Coprime (p + q) s ∧
    -- r is not divisible by the square of any prime
    ∀ (prime : ℕ), Nat.Prime prime → ¬(prime ^ 2 ∣ r) ∧
    -- Sum of p, q, r, s
    p + q + r + s = 673 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_sum_l447_44758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_temperature_count_l447_44719

def fahrenheit_to_celsius (f : ℤ) : ℚ :=
  (5 : ℚ) / 9 * (f - 32)

def celsius_to_fahrenheit (c : ℚ) : ℚ :=
  (9 : ℚ) / 5 * c + 32

def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

def temperature_conversion_cycle (t : ℤ) : ℤ :=
  round_to_nearest (celsius_to_fahrenheit (round_to_nearest (fahrenheit_to_celsius t)))

def unchanged_temperatures : Finset ℤ :=
  Finset.filter (λ t ↦ t ∈ Finset.Icc 32 1000 ∧ temperature_conversion_cycle t = t) (Finset.Icc 32 1000)

theorem unchanged_temperature_count :
  Finset.card unchanged_temperatures = 539 := by
  sorry

#eval Finset.card unchanged_temperatures

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_temperature_count_l447_44719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l447_44794

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoidPerpendicularDiagonals where
  base1 : ℝ
  base2 : ℝ
  is_isosceles : True
  diagonals_perpendicular : True

/-- The area of an isosceles trapezoid with perpendicular diagonals -/
noncomputable def area (t : IsoscelesTrapezoidPerpendicularDiagonals) : ℝ :=
  ((t.base1 + t.base2) / 2) ^ 2

/-- Theorem: The area of an isosceles trapezoid with bases 40 cm and 24 cm, 
    and perpendicular diagonals, is 1024 cm² -/
theorem isosceles_trapezoid_area : 
  ∀ t : IsoscelesTrapezoidPerpendicularDiagonals, 
  t.base1 = 40 ∧ t.base2 = 24 → area t = 1024 := by
  intro t ⟨h1, h2⟩
  unfold area
  rw [h1, h2]
  norm_num
  -- The proof is completed by norm_num, but we can add sorry if needed
  -- sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l447_44794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l447_44790

/-- Given a parabola with equation y^2 = 6x, its focus has coordinates (3/2, 0) -/
theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 = 6*x → (x, y) = (3/2, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l447_44790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l447_44767

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 5)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem equidistant_point : 
  ∃ x : ℝ, distance (x, 0) A = distance (x, 0) B ∧ x = 33/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l447_44767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l447_44754

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + Real.log x + b

-- State the theorem
theorem f_properties (a b : ℝ) 
  (h1 : ∀ x, x > 0 → (4 * x + 4 * f a b x + 1 = 0) ↔ x = 1) :
  (∃ x_max > 0, ∀ x > 0, f a b x ≤ f a b x_max ∧ f a b x_max = -(3 + 2 * Real.log 2) / 4) ∧
  (∀ x > 0, f a b x < x^3 - 2 * x^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l447_44754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l447_44716

theorem integers_between_cubes : 
  (Finset.range (Int.toNat (Int.floor ((10.4 : ℝ)^3) - Int.ceil ((10.1 : ℝ)^3) + 1))).card = 94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l447_44716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l447_44710

/-- Definition of an ellipse (C) -/
def Ellipse (a b : ℝ) (x y : ℝ → ℝ) :=
  ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of a point being on the ellipse -/
def OnEllipse (a b x y : ℝ) :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Main theorem about the ellipse and point P -/
theorem ellipse_theorem (a b : ℝ) (x y : ℝ → ℝ) 
    (h1 : a > b) (h2 : b > 0)
    (h3 : Eccentricity a b = Real.sqrt 2 / 2)
    (h4 : Ellipse a b x y)
    (h5 : x 0 = 1 ∧ y 0 = 0)  -- Right focus F(1,0)
    (M N : ℝ × ℝ) (h6 : OnEllipse a b M.1 M.2)
    (h7 : OnEllipse a b N.1 N.2) (h8 : M.1 = N.1 ∧ M.2 = -N.2)
    (Q : ℝ × ℝ) (h9 : Q = (2, 0))
    (P : ℝ × ℝ) (h10 : ∃ t, (1 - t) * M + t * (1, 0) = P)
    (h11 : ∃ s, (1 - s) * N + s * Q = P) :
  (a = Real.sqrt 2 ∧ b = 1) ∧ OnEllipse (Real.sqrt 2) 1 P.1 P.2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l447_44710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_500_l447_44708

theorem closest_perfect_square_to_500 :
  ∀ n : ℤ, n ≠ 484 → n^2 ≠ 484 → |500 - 484| ≤ |500 - n^2| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_500_l447_44708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wastewater_2013_calculation_l447_44740

/-- The amount of wastewater treated in the first quarter of 2014, in cubic meters -/
noncomputable def wastewater_2014 : ℝ := 38000

/-- The percentage increase from 2013 to 2014 -/
noncomputable def percentage_increase : ℝ := 60

/-- The amount of wastewater treated in the first quarter of 2013, in cubic meters -/
noncomputable def wastewater_2013 : ℝ := wastewater_2014 / (1 + percentage_increase / 100)

theorem wastewater_2013_calculation :
  wastewater_2013 = 23750 := by
  -- Expand the definition of wastewater_2013
  unfold wastewater_2013
  -- Expand the definitions of wastewater_2014 and percentage_increase
  unfold wastewater_2014 percentage_increase
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wastewater_2013_calculation_l447_44740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l447_44797

/-- Given a parabola y = 4ax^2 where a ≠ 0, its focus coordinates are (0, 1/(16a)) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let parabola := {p : ℝ × ℝ | p.2 = 4 * a * p.1^2}
  ∃ (f : ℝ × ℝ), f = (0, 1 / (16 * a)) ∧ f ∈ parabola :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l447_44797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redSequence2018_l447_44799

/-- Represents the sequence of red numbers --/
def redSequence : ℕ → ℕ := sorry

/-- The nth group in the sequence --/
def groupNumber (n : ℕ) : ℕ := n

/-- The starting number of the nth group --/
def groupStart (n : ℕ) : ℕ := (n - 1)^2 + 1

/-- The number of elements in the nth group --/
def groupSize (n : ℕ) : ℕ := n

/-- The index of the last element in the nth group --/
def lastIndexInGroup (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 2018th element in the sequence --/
def element2018 : ℕ := 3972

/-- The group that contains the 2018th element --/
def group2018 : ℕ := 64

/-- The position of the 2018th element within its group --/
def positionInGroup2018 : ℕ := 2

/-- Theorem stating that the 2018th element in the redSequence is 3972 --/
theorem redSequence2018 : redSequence 2018 = 3972 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_redSequence2018_l447_44799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_x_in_open_interval_max_value_of_g_l447_44720

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - 3

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sqrt (x + 4) + 4 * Real.sqrt (|x - 6|)

-- Statement for part I
theorem f_negative_iff_x_in_open_interval :
  ∀ x : ℝ, f x < 0 ↔ x ∈ Set.Ioo (-1) 5 := by
  sorry

-- Statement for part II
theorem max_value_of_g :
  ∃ M : ℝ, M = 5 * Real.sqrt 10 ∧ 
  (∀ x ∈ Set.Ioo (-1) 5, g x ≤ M) ∧
  (∃ x₀ ∈ Set.Ioo (-1) 5, g x₀ = M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_x_in_open_interval_max_value_of_g_l447_44720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_spheres_l447_44721

/-- Regular tetrahedron with inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  inscribed_radius : ℝ
  circumscribed_radius : ℝ
  face_sphere_radius : ℝ

/-- Conditions for the tetrahedron -/
def tetrahedron_conditions (t : RegularTetrahedron) : Prop :=
  t.circumscribed_radius = 2 * t.inscribed_radius ∧
  t.face_sphere_radius = t.inscribed_radius

/-- Volume of a sphere given its radius -/
noncomputable def sphere_volume (radius : ℝ) : ℝ :=
  (4 / 3) * Real.pi * radius ^ 3

/-- Probability calculation -/
noncomputable def probability (t : RegularTetrahedron) : ℝ :=
  (sphere_volume t.inscribed_radius + 4 * sphere_volume t.face_sphere_radius) /
  sphere_volume t.circumscribed_radius

/-- Main theorem -/
theorem probability_in_spheres (t : RegularTetrahedron) 
  (h : tetrahedron_conditions t) : probability t = 5/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_spheres_l447_44721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_distance_l447_44702

/-- Given a square PQRS with side length t and quarter-circle arcs with radii t
    drawn from corners P and Q, the distance of the intersection point Y
    of these arcs from side RS is t(2 - √3)/2 -/
theorem intersection_point_distance (t : ℝ) (h : t > 0) :
  let square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ t ∧ 0 ≤ p.2 ∧ p.2 ≤ t}
  let arc_p := {p : ℝ × ℝ | p.1^2 + (p.2 - t)^2 = t^2 ∧ 0 ≤ p.1 ∧ p.1 ≤ t ∧ t/2 ≤ p.2 ∧ p.2 ≤ t}
  let arc_q := {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - t)^2 = t^2 ∧ 0 ≤ p.1 ∧ p.1 ≤ t ∧ t/2 ≤ p.2 ∧ p.2 ≤ t}
  let y := arc_p ∩ arc_q
  ∃ (p : ℝ × ℝ), p ∈ y ∧ p.2 = t * (2 - Real.sqrt 3) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_distance_l447_44702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_y_formula_l447_44774

theorem tan_y_formula (a b y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < y) (h4 : y < π / 2)
  (h5 : Real.sin y = (3 * a * b) / Real.sqrt (a^6 + 3 * a^3 * b^3 + b^6)) :
  Real.tan y = (3 * a * b) / Real.sqrt (a^6 + 3 * a^3 * b^3 + b^6 - 9 * a^2 * b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_y_formula_l447_44774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equation_solutions_l447_44765

theorem binomial_equation_solutions : 
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, (Nat.choose 16 (Nat.floor (x^2 - x)) = Nat.choose 16 (Nat.floor (5*x - 5))) ∧ 
               (0 ≤ x^2 - x) ∧ (x^2 - x ≤ 16) ∧ 
               (0 ≤ 5*x - 5) ∧ (5*x - 5 ≤ 16)) ∧ 
    s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equation_solutions_l447_44765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l447_44743

/-- Given a line segment with one endpoint at (15, -8) and midpoint at (10, -3),
    the sum of the coordinates of the other endpoint is 7. -/
theorem endpoint_coordinate_sum : ∀ (x y mx my : ℝ),
  (mx = (x + 15) / 2 ∧ my = (y - 8) / 2) →
  (mx = 10 ∧ my = -3) →
  x + y = 7 := by
  intros x y mx my h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l447_44743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_specific_quadratic_has_two_real_roots_l447_44752

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  discriminant > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

/-- The quadratic equation 5x^2 + 14x + 5 = 0 has two distinct real roots. -/
theorem specific_quadratic_has_two_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 5*x₁^2 + 14*x₁ + 5 = 0 ∧ 5*x₂^2 + 14*x₂ + 5 = 0 :=
by
  have h : (5 : ℝ) ≠ 0 := by norm_num
  apply quadratic_equation_roots 5 14 5 h
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_specific_quadratic_has_two_real_roots_l447_44752
