import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_difference_l123_12302

/-- The parabola defined by x^2 = 2y -/
def Parabola : Set (ℝ × ℝ) := {p | p.1^2 = 2 * p.2}

/-- The locus E of point M -/
def LocusE : Set (ℝ × ℝ) := {p | p.1^2 = 4 * p.2}

/-- Point S -/
def S : ℝ × ℝ := (-4, 4)

/-- Point N -/
def N : ℝ × ℝ := (4, 5)

/-- Line l passing through N -/
def LineL (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 - 4) + 5}

/-- Slope of line SA -/
noncomputable def k1 (A : ℝ × ℝ) : ℝ := (A.2 - S.2) / (A.1 - S.1)

/-- Slope of line SB -/
noncomputable def k2 (B : ℝ × ℝ) : ℝ := (B.2 - S.2) / (B.1 - S.1)

/-- The theorem to be proved -/
theorem min_slope_difference :
  ∀ k : ℝ,
  ∀ A B : ℝ × ℝ,
  A ∈ LocusE → B ∈ LocusE →
  A ∈ LineL k → B ∈ LineL k →
  A ≠ B →
  |k1 A - k2 B| ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_difference_l123_12302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l123_12360

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  2 * (a * b + a * c + b * c) > a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l123_12360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_bush_berries_l123_12338

def berries_picked : ℕ → ℕ
  | 0 => 2  -- First bush (we use 0-indexing here)
  | n + 1 => 
    if n % 2 = 0 then
      berries_picked n * 2  -- Odd bushes: multiply by 2
    else
      berries_picked n + 1  -- Even bushes: add 1

theorem sixth_bush_berries : berries_picked 5 = 15 := by
  -- Unfold the definition for the first 6 bushes
  unfold berries_picked
  simp
  -- The rest of the proof would go here
  sorry

#eval berries_picked 5  -- This should output 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_bush_berries_l123_12338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_f_eq_g_l123_12361

/-- The infinite series representation of the given equation -/
noncomputable def f (x : ℝ) : ℝ := ∑' n, (-1)^n * (n + 1) * x^n

/-- The closed form of the infinite series for |x| < 1 -/
noncomputable def g (x : ℝ) : ℝ := 1 / (1 + x)^2

/-- The equation to be solved -/
def equation (x : ℝ) : Prop := x = f x

/-- The theorem stating the approximate solution -/
theorem equation_solution : 
  ∃ x : ℝ, equation x ∧ abs x < 1/2 ∧ abs (x + 0.5436890127) < 1e-10 := by
  sorry

/-- The theorem stating that f and g are equal for |x| < 1 -/
theorem f_eq_g (x : ℝ) (h : abs x < 1) : f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_f_eq_g_l123_12361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_right_triangle_l123_12318

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a right angle in a triangle -/
def RightAngle (A B C : Point) : Prop :=
  sorry

/-- Represents the angle between three points -/
noncomputable def angle (A B C : Point) : ℝ :=
  sorry

/-- The eccentricity of a hyperbola given its foci and a point on the curve -/
noncomputable def HyperbolaEccentricity (A B C : Point) : ℝ :=
  sorry

/-- Given a right triangle ABC with cos C = (2√5) / 5, the eccentricity of the hyperbola
    passing through point C with A and B as foci is 2 + √5 -/
theorem hyperbola_eccentricity_right_triangle 
  (A B C : Point) (h_right : RightAngle A B C) 
  (h_cos : Real.cos (angle A C B) = (2 * Real.sqrt 5) / 5) : 
  HyperbolaEccentricity A B C = 2 + Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_right_triangle_l123_12318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l123_12303

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos x * (2 * a * Real.sin x - Real.cos x) + Real.sin x ^ 2

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : ∀ x, f a x ≤ 2) :
  a = Real.sqrt 3 ∧
  (∀ k : ℤ, IsClosed (Set.Icc (π / 3 + k * π) (5 * π / 6 + k * π)) ∧
    ∀ x ∈ Set.Icc (π / 3 + k * π) (5 * π / 6 + k * π), StrictAntiOn (f a) (Set.Icc (π / 3 + k * π) (5 * π / 6 + k * π))) ∧
  (∀ A B C a b c : ℝ,
    (a^2 + c^2 - b^2) / (a^2 + b^2 - c^2) = c / (2*a - c) →
    B = π / 3) ∧
  Set.Icc 1 2 = Set.image (f a) (Set.Icc (π / 3) (π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l123_12303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stops_45_minutes_l123_12381

/-- Calculates the number of minutes a bus stops per hour given its speeds with and without stoppages -/
noncomputable def bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : ℝ :=
  let distance_lost := speed_without_stops - speed_with_stops
  distance_lost / (speed_without_stops / 60)

/-- Theorem stating that a bus with given speeds stops for 45 minutes per hour -/
theorem bus_stops_45_minutes :
  bus_stop_time 60 15 = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stops_45_minutes_l123_12381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_and_reciprocal_l123_12339

theorem max_sum_and_reciprocal (n : ℕ) (h : n = 3011) (s : ℝ) (hs : s = 3012) : 
  ∃ (f : Fin n → ℝ), 
    (∀ i, f i > 0) ∧ 
    (Finset.sum Finset.univ f = s) ∧
    (Finset.sum Finset.univ (λ i => 1 / f i) = s) ∧
    (∀ x ∈ Finset.image f Finset.univ, x + 1 / x ≤ 12073 / 3012) :=
by
  sorry

#check max_sum_and_reciprocal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_and_reciprocal_l123_12339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_through_A_l123_12354

-- Define the basic geometric elements
variable {Ω : Type*} [NormedAddCommGroup Ω] [InnerProductSpace ℝ Ω] [CompleteSpace Ω]
variable (A P B C : Ω)
variable (circle1 circle2 : Set Ω)
variable (line : Set Ω)

-- Define the conditions
variable (h1 : A ∈ circle1 ∩ circle2)
variable (h2 : P ∈ circle1 ∩ circle2)
variable (h3 : B ∈ circle1 ∩ line)
variable (h4 : C ∈ circle2 ∩ line)
variable (h5 : P ∈ line)
variable (h6 : line ≠ {A, P})

-- Define the triangle ABC
def triangle_ABC : Set Ω := {A, B, C}

-- Define the centers of the triangle
noncomputable def centroid : Ω := sorry
noncomputable def circumcenter : Ω := sorry
noncomputable def orthocenter : Ω := sorry
noncomputable def incenter : Ω := sorry

-- Define the locus of centers
noncomputable def locus_of_centers : Set Ω := sorry

-- Theorem statement
theorem locus_is_circle_through_A :
  ∃ (center : Ω) (radius : ℝ), 
    locus_of_centers = {X | dist X center = radius} ∧ 
    A ∈ locus_of_centers :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_through_A_l123_12354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factor_l123_12364

theorem perfect_square_factor (p q r a : ℕ+) (h1 : p * q = r * a^2) (h2 : Nat.Prime r) (h3 : Nat.Coprime p q) :
  ∃ x : ℕ+, (p = x^2 ∨ q = x^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factor_l123_12364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l123_12335

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100}

def valid_subset (H : Set ℕ) : Prop :=
  H ⊆ S ∧ ∀ x ∈ H, 10 * x ∉ H

theorem max_subset_size :
  ∃ H : Set ℕ, valid_subset H ∧ H.Finite ∧ Nat.card H = 91 ∧
    ∀ H' : Set ℕ, valid_subset H' → H'.Finite → Nat.card H' ≤ 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l123_12335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equals_sin_l123_12308

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3)
noncomputable def g (x : ℝ) : ℝ := sin (2 * (x + π / 12))

-- State the theorem
theorem cos_shift_equals_sin (x : ℝ) : f x = g x := by
  -- Expand the definitions of f and g
  unfold f g
  -- Use trigonometric identities to prove the equality
  -- This step would require several intermediate steps and trigonometric theorems
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equals_sin_l123_12308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_23_l123_12366

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Theorem: If a_6 + a_9 = a_3 + 4 in an arithmetic sequence, then S_23 = 92 -/
theorem arithmetic_sequence_sum_23 (seq : ArithmeticSequence) 
  (h : seq.a 6 + seq.a 9 = seq.a 3 + 4) : 
  S seq 23 = 92 := by
  sorry

#check arithmetic_sequence_sum_23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_23_l123_12366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_equation_correct_l123_12323

/-- Represents the painting scenario with Alice and Bob --/
structure PaintingScenario where
  alice_rate : ℚ := 1 / 6  -- Alice's painting rate (room per hour)
  bob_rate : ℚ := 1 / 8    -- Bob's painting rate (room per hour)
  break_time : ℚ := 2      -- Break time in hours

/-- The equation that represents the painting scenario --/
def painting_equation (x t : ℚ) : Prop :=
  (1 / 8) * x + (1 / 6) * t = 4 / 3

/-- Theorem stating that the painting equation correctly represents the scenario --/
theorem painting_equation_correct :
  ∃ x t : ℚ, x > 0 ∧ t > 2 ∧ painting_equation x t := by
  sorry

#check painting_equation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_equation_correct_l123_12323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l123_12304

/-- Represents a parabola in the form ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  p.a * x^2 + p.b * x + p.c = y

/-- Calculates the vertex of the parabola -/
noncomputable def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  let x := -p.b / (2 * p.a)
  let y := p.a * x^2 + p.b * x + p.c
  (x, y)

/-- Checks if the parabola has a vertical axis of symmetry -/
def Parabola.hasVerticalAxisOfSymmetry (p : Parabola) : Prop :=
  p.a ≠ 0

theorem parabola_equation : 
  ∃ (p : Parabola), 
    p.a = 4 ∧ 
    p.b = -24 ∧ 
    p.c = 34 ∧ 
    p.vertex = (3, -2) ∧ 
    p.hasVerticalAxisOfSymmetry ∧ 
    p.contains 4 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l123_12304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_subsequence_property_l123_12398

noncomputable def sequence_a (l m : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => (l * (sequence_a l m n)^2 + m * sequence_a l m n + 4) / (sequence_a l m n + 2)

theorem sequence_a_properties (l m : ℝ) (hl : l ≠ 0) (hm : m ≠ 0) :
  (∀ n : ℕ, sequence_a 3 8 n = 2 * 3^(n-1) - 1) ∧
  (∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, sequence_a l m (n+1) - sequence_a l m n = d) →
    l = 1 ∧ m = 4 :=
by sorry

def S (n : ℕ) : ℕ := n^2

theorem subsequence_property :
  ∃! (a b c : ℕ), a < b ∧ b < c ∧ 
    S 1 < S a ∧ S a < S b ∧ S b < S c ∧
    S a + S b = 2017 ∧
    (∀ x y z : ℕ, x < y ∧ y < z ∧ 
      S 1 < S x ∧ S x < S y ∧ S y < S z ∧
      S x + S y = 2017 →
      (x = a ∧ y = b ∧ z = c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_subsequence_property_l123_12398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l123_12396

/-- Given a principal amount and an interest rate, calculates the total interest
    after 10 years when the principal is trebled after 5 years. -/
noncomputable def total_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  let interest_5_years := (principal * rate * 5) / 100
  let interest_next_5_years := (3 * principal * rate * 5) / 100
  interest_5_years + interest_next_5_years

/-- Theorem stating that under the given conditions, the total interest is 2000. -/
theorem interest_calculation (principal : ℝ) (rate : ℝ) 
    (h : (principal * rate * 10) / 100 = 800) : 
  total_interest principal rate = 2000 := by
  sorry

#check interest_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l123_12396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonessa_tax_percentage_l123_12317

/-- The percentage of pay that goes to tax, given total pay and take-home pay -/
noncomputable def tax_percentage (total_pay : ℚ) (take_home_pay : ℚ) : ℚ :=
  ((total_pay - take_home_pay) / total_pay) * 100

/-- Jonessa's tax percentage problem -/
theorem jonessa_tax_percentage :
  tax_percentage 500 450 = 10 := by
  -- Unfold the definition of tax_percentage
  unfold tax_percentage
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonessa_tax_percentage_l123_12317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_in_region_l123_12321

theorem integer_points_in_region : 
  let S := {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ 4 * x + 3 * y < 12}
  Finset.card (Finset.filter (fun (x, y) => x > 0 ∧ y > 0 ∧ 4 * x + 3 * y < 12) (Finset.product (Finset.range 12) (Finset.range 12))) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_in_region_l123_12321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_small_spheres_l123_12332

/-- The amount of paint needed for a sphere is proportional to its surface area -/
def paint_prop_to_surface (r : ℝ) (paint : ℝ) : Prop := ∃ k, paint = k * (4 * Real.pi * r^2)

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem paint_for_small_spheres (R r : ℝ) (paint_large : ℝ) (paint_small : ℝ) :
  R > 0 → r > 0 →
  sphere_volume R = 64 * sphere_volume r →
  paint_prop_to_surface R paint_large →
  paint_prop_to_surface r paint_small →
  paint_large = 2.4 →
  paint_small = 9.6 := by
  sorry

#check paint_for_small_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_small_spheres_l123_12332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_sum_l123_12378

open Real

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1)

theorem min_value_of_f_sum (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : f x₁ + f x₂ = 1) :
  ∀ y, f (x₁ + x₂) ≤ y → 4/5 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_sum_l123_12378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relationship_l123_12320

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2 * y = 3

-- Define the angles of inclination
noncomputable def α : ℝ := Real.arctan 2
noncomputable def β : ℝ := Real.arctan (-1/2) + Real.pi

-- Define the relationship between the lines and angles
axiom slope_line1 : ∃ m : ℝ, m = 2 ∧ ∀ x y : ℝ, line1 x y → y = m * x + 1
axiom slope_line2 : ∃ m : ℝ, m = -1/2 ∧ ∀ x y : ℝ, line2 x y → y = m * x + 3/2

-- Define the perpendicularity of the lines
axiom perpendicular : ∃ m1 m2 : ℝ, 
  (∀ x y : ℝ, line1 x y → y = m1 * x + 1) ∧ 
  (∀ x y : ℝ, line2 x y → y = m2 * x + 3/2) ∧ 
  m1 * m2 = -1

-- Define the nature of the angles
axiom α_acute : 0 < α ∧ α < Real.pi/2
axiom β_obtuse : Real.pi/2 < β ∧ β < Real.pi

-- Theorem to prove
theorem angle_relationship : β = Real.pi/2 + α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relationship_l123_12320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l123_12315

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_difference_magnitude (a b : V) 
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 2)
  (hab : ‖a + b‖ = 4) :
  ‖a - b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l123_12315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_real_range_real_domain_restricted_decreasing_on_interval_range_composition_unique_solution_upper_bound_on_interval_l123_12330

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*(2*a-1)*x + 8) / Real.log (1/2)

-- Theorem 1: Domain of f is ℝ
theorem domain_real (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ 1/2 - Real.sqrt 2 < a ∧ a < 1/2 + Real.sqrt 2 :=
sorry

-- Theorem 2: Range of f is ℝ
theorem range_real (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ≤ 1/2 - Real.sqrt 2 ∨ a ≥ 1/2 + Real.sqrt 2 :=
sorry

-- Theorem 3: f is defined on [-1, +∞]
theorem domain_restricted (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → ∃ y : ℝ, f a x = y) ↔ -7/4 < a ∧ a < 1/2 + Real.sqrt 2 :=
sorry

-- Theorem 4: f is decreasing on [a, +∞]
theorem decreasing_on_interval (a : ℝ) :
  (∀ x y : ℝ, a ≤ x ∧ x < y → f a x > f a y) ↔ -4/3 < a ∧ a ≤ 1 :=
sorry

-- Theorem 5: Range of f(sin(2x - π/3)) when a = 3/4 and x ∈ [π/12, π/2]
theorem range_composition (x : ℝ) :
  x ∈ Set.Icc (Real.pi/12) (Real.pi/2) →
  f (3/4) (Real.sin (2*x - Real.pi/3)) ∈ Set.Icc (Real.log (35/4) / Real.log (1/2)) (Real.log (31/4) / Real.log (1/2)) :=
sorry

-- Theorem 6: f(x) = -1 + log_(1/2)(x+3) has exactly one solution in [1, 3]
theorem unique_solution (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Icc 1 3 ∧ f a x = -1 + Real.log (x+3) / Real.log (1/2)) ↔
  (a = Real.sqrt 2 / 2 ∨ (3/4 < a ∧ a ≤ 11/12)) :=
sorry

-- Theorem 7: f(x) ≤ -1 for all x ∈ [2, 3]
theorem upper_bound_on_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → f a x ≤ -1) ↔ a ≤ (Real.sqrt 6 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_real_range_real_domain_restricted_decreasing_on_interval_range_composition_unique_solution_upper_bound_on_interval_l123_12330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_temperature_in_fahrenheit_l123_12350

-- Define the boiling point of water in Celsius and Kelvin
noncomputable def boiling_point_celsius : ℝ := 100
noncomputable def boiling_point_kelvin : ℝ := 373.15

-- Define the conversion functions
noncomputable def celsius_to_kelvin (c : ℝ) : ℝ := c + 273.15
noncomputable def kelvin_to_celsius (k : ℝ) : ℝ := k - 273.15
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

-- Theorem statement
theorem target_temperature_in_fahrenheit :
  let target_kelvin := boiling_point_kelvin - 40
  let target_celsius := kelvin_to_celsius target_kelvin
  celsius_to_fahrenheit target_celsius = 140 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_temperature_in_fahrenheit_l123_12350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_equals_third_l123_12363

theorem sum_of_two_equals_third (n : ℕ) :
  ∀ (S : Finset ℕ), S.card = n + 2 → (∀ x, x ∈ S → x ≤ 2*n) →
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_equals_third_l123_12363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l123_12391

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  downstream_distance : ℝ
  time : ℝ
  still_water_speed : ℝ

/-- Calculates the upstream distance given a swimming scenario -/
noncomputable def upstream_distance (s : SwimmingScenario) : ℝ :=
  (s.still_water_speed - (s.downstream_distance / s.time - s.still_water_speed)) * s.time

/-- Theorem stating that the upstream distance is 18 km under the given conditions -/
theorem upstream_distance_is_18 (s : SwimmingScenario) 
  (h1 : s.downstream_distance = 30)
  (h2 : s.time = 3)
  (h3 : s.still_water_speed = 8) :
  upstream_distance s = 18 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval upstream_distance { downstream_distance := 30, time := 3, still_water_speed := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l123_12391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l123_12316

/-- The number of hours Kate charged to the project -/
noncomputable def kate_hours : ℝ := 216 / 9.75

/-- The number of hours Pat charged to the project -/
noncomputable def pat_hours : ℝ := 2 * kate_hours

/-- The number of hours Mark charged to the project -/
noncomputable def mark_hours : ℝ := 6 * kate_hours

/-- The number of hours Alice charged to the project -/
noncomputable def alice_hours : ℝ := 3/4 * kate_hours

/-- The total number of hours charged to the project -/
def total_hours : ℝ := 216

theorem project_hours_difference :
  ∃ ε > 0, |mark_hours - (kate_hours + alice_hours) - 94.14| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l123_12316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_calculation_l123_12346

theorem sum_reciprocal_calculation : 15 * ((1/3 : ℚ) + 1/4 + 1/6)⁻¹ = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_calculation_l123_12346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_fifth_power_decomposition_l123_12327

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 4, 3]

theorem A_fifth_power_decomposition :
  A^5 = 293 • A + 72 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_fifth_power_decomposition_l123_12327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_sector_area_l123_12376

/-- The area of a circular sector given the radius of the circle and the probability of the sector -/
noncomputable def sectorArea (radius : ℝ) (probability : ℝ) : ℝ :=
  probability * Real.pi * radius^2

theorem win_sector_area :
  let radius : ℝ := 8
  let winProbability : ℝ := 3/8
  sectorArea radius winProbability = 24 * Real.pi := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_sector_area_l123_12376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l123_12347

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set ℕ := {x : ℕ | x^2 ∈ A}

theorem A_intersect_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l123_12347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_bound_l123_12340

theorem max_product_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 27) :
  max (a * b) (max (b * c) (c * a)) ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_bound_l123_12340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_graph_is_finite_distinct_points_l123_12372

/-- Represents the cost of oranges in cents -/
def orange_cost (n : ℕ) : ℕ := 25 * n

/-- The set of points representing the cost graph for 1 to 20 oranges -/
def cost_graph : Set (ℕ × ℕ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, orange_cost n)}

theorem cost_graph_is_finite_distinct_points :
  Set.Finite cost_graph ∧ ∀ p q : ℕ × ℕ, p ∈ cost_graph → q ∈ cost_graph → p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_graph_is_finite_distinct_points_l123_12372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_consumed_in_two_weeks_l123_12365

/-- Represents the two types of breakfast Jason eats -/
inductive Breakfast
  | omelet
  | scrambleWithSide

/-- Returns the number of eggs used for a given breakfast type -/
def eggsUsed (b : Breakfast) : ℕ :=
  match b with
  | .omelet => 3
  | .scrambleWithSide => 3

/-- Returns the breakfast type for a given day, assuming Jason starts with an omelet -/
def breakfastOnDay (day : ℕ) : Breakfast :=
  if day % 2 = 1 then .omelet else .scrambleWithSide

/-- Calculates the total number of eggs consumed over a given number of days -/
def totalEggsConsumed (days : ℕ) : ℕ :=
  List.range days |>.map (fun d => eggsUsed (breakfastOnDay (d + 1))) |>.sum

theorem eggs_consumed_in_two_weeks :
  totalEggsConsumed 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_consumed_in_two_weeks_l123_12365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_parametric_line_l123_12397

/-- The slope angle of a line given by parametric equations x = 1 + 2√3t and y = 3 - 2t -/
theorem slope_angle_of_parametric_line : 
  Real.arctan (Real.sqrt 3 / 3) = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_parametric_line_l123_12397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l123_12362

-- Define the functions for the curve and line
noncomputable def f (x : ℝ) : ℝ := 2 / x
def g (x : ℝ) : ℝ := x - 1

-- Define the bounds of integration
def lower_bound : ℝ := 1
def upper_bound : ℝ := 2

-- State the theorem
theorem area_of_closed_figure :
  ∫ x in lower_bound..upper_bound, (f x - g x) = 2 * Real.log 2 - 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l123_12362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_proof_l123_12377

/-- The smallest positive angle in radians with the same terminal side as -600° -/
noncomputable def smallest_positive_angle : ℝ := 2 * Real.pi / 3

/-- Converts degrees to radians -/
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ := degrees * Real.pi / 180

theorem smallest_positive_angle_proof :
  smallest_positive_angle = degrees_to_radians 120 ∧
  smallest_positive_angle > 0 ∧
  ∃ (k : ℤ), degrees_to_radians (-600) = smallest_positive_angle + 2 * Real.pi * k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_proof_l123_12377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l123_12386

noncomputable def h (x : ℝ) : ℝ := (3*x + 1) / (x^2 - 9)

theorem h_domain :
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l123_12386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_postal_code_l123_12328

def phone_number : List Nat := [3, 4, 6, 2, 7, 8, 9]

def sum_digits (digits : List Nat) : Nat :=
  digits.sum

def is_valid_postal_code (code : List Nat) : Prop :=
  code.length = 5 ∧ code.Nodup ∧ sum_digits code = sum_digits phone_number

def is_largest_postal_code (code : List Nat) : Prop :=
  is_valid_postal_code code ∧
  ∀ other : List Nat, is_valid_postal_code other → 
    (List.foldl (fun acc d => acc * 10 + d) 0 other) ≤ 
    (List.foldl (fun acc d => acc * 10 + d) 0 code)

theorem largest_postal_code :
  is_largest_postal_code [9, 8, 7, 6, 5] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_postal_code_l123_12328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_plan_correct_actual_sales_correct_l123_12395

-- Define the cost and pricing parameters
def cost : ℚ := 24
def price1 : ℚ := 32
def price2 : ℚ := 28
def monthly_fee : ℚ := 2400

-- Define profit functions for each plan
def profit1 (x : ℚ) : ℚ := (price1 - cost) * x - monthly_fee
def profit2 (x : ℚ) : ℚ := (price2 - cost) * x

-- Define the optimal plan choice function
def optimal_plan (x : ℚ) : String :=
  if x > 600 then "Plan 1"
  else if x < 600 then "Plan 2"
  else "Either"

-- Define the reported sales and profits for each month
def reported_sales : List ℚ := [550, 600, 1400]
def reported_profits : List ℚ := [2000, 2400, 5600]

-- Define the actual sales for each month
def actual_sales : List ℚ := [500, 600, 1000]

-- Theorem: The optimal plan choice is correct
theorem optimal_plan_correct (x : ℚ) :
  optimal_plan x = "Plan 1" → profit1 x > profit2 x ∧
  optimal_plan x = "Plan 2" → profit2 x > profit1 x ∧
  optimal_plan x = "Either" → profit1 x = profit2 x := by
  sorry

-- Theorem: The actual total sales volume for the first quarter is correct
theorem actual_sales_correct :
  actual_sales.sum = 2100 ∧
  (∀ i : Fin 3, 
    (if i.val = 1 then profit2 (actual_sales.get i) = reported_profits.get i
     else profit1 (actual_sales.get i) = reported_profits.get i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_plan_correct_actual_sales_correct_l123_12395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quad_area_l123_12306

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Calculates the intersection point of two lines -/
noncomputable def intersect (l1 l2 : Line) : Point :=
  { x := (l2.b - l1.b) / (l1.m - l2.m),
    y := l1.m * ((l2.b - l1.b) / (l1.m - l2.m)) + l1.b }

/-- Calculates the area of a quadrilateral given its four vertices -/
noncomputable def quadArea (a b c d : Point) : ℝ :=
  (1/2) * |a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y -
           (a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x)|

/-- States that the area of the inner quadrilateral in a unit square
    with lines drawn from vertices to midpoints is 1/5 -/
theorem inner_quad_area :
  let l1 : Line := { m := 1/2, b := 0 }
  let l2 : Line := { m := -2, b := 1 }
  let l3 : Line := { m := 2, b := 0 }
  let l4 : Line := { m := -1/2, b := 1/2 }
  let l5 : Line := { m := 2, b := -1 }
  let l6 : Line := { m := -1/2, b := 3/2 }
  let l7 : Line := { m := -2, b := 2 }
  let l8 : Line := { m := -1/2, b := 1 }
  let a := intersect l1 l2
  let b := intersect l3 l4
  let c := intersect l5 l6
  let d := intersect l7 l8
  quadArea a b c d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quad_area_l123_12306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mersenne_divisors_l123_12351

def has_n_divisors (n : ℕ) : Prop :=
  (Finset.filter (λ d ↦ (2^n - 1) % d = 0) (Finset.range (2^n - 1 + 1))).card = n

theorem mersenne_divisors :
  {n : ℕ | n ≥ 1 ∧ has_n_divisors n} = {1, 2, 4, 6, 8, 16, 32} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mersenne_divisors_l123_12351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l123_12341

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: If S, S₃, S₅ form an arithmetic sequence, then q = -2 -/
theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = geometric_sum a₁ q n) →
  (∃ d, S 4 - S 3 = d ∧ S 5 - S 4 = d) →
  q = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l123_12341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l123_12383

/-- Represents a square pyramid --/
structure SquarePyramid where
  base_edge : ℝ
  altitude : ℝ

/-- Calculates the volume of a square pyramid --/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.base_edge^2 * p.altitude

/-- Theorem: The volume of the frustum after cutting a similar pyramid 
    with 1/3 altitude is 26/27 of the original volume --/
theorem frustum_volume_ratio (p : SquarePyramid) :
  let small_pyramid := SquarePyramid.mk (p.base_edge / 3) (p.altitude / 3)
  let frustum_volume := pyramid_volume p - pyramid_volume small_pyramid
  frustum_volume / pyramid_volume p = 26 / 27 := by
  sorry

/-- Example with given dimensions --/
def example_pyramid : SquarePyramid :=
  SquarePyramid.mk 40 18

-- Remove the #eval statement as it's not necessary for building
-- #eval frustum_volume_ratio example_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l123_12383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_C_R_A_B_l123_12353

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, x^2 - y^2 = 1}
def B : Set ℝ := {y | ∃ x, x^2 = 4*y}

-- Define the complement of A in the real numbers
def C_R_A : Set ℝ := {x | x ∉ A}

-- State the theorem
theorem intersection_C_R_A_B : C_R_A ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_C_R_A_B_l123_12353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l123_12357

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define the line l
def l (x y : ℝ) : Prop := y = -x + 3

-- Define point T
def T : ℝ × ℝ := (2, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem ellipse_intersection_property :
  ∃ (A B P : ℝ × ℝ) (l' : ℝ → ℝ),
    E A.1 A.2 ∧ E B.1 B.2 ∧ l P.1 P.2 ∧
    (∀ x, l' x = (T.2 - O.2) / (T.1 - O.1) * (x - O.1) + O.2) ∧
    (l' A.1 = A.2) ∧ (l' B.1 = B.2) ∧
    (distance P T)^2 = (4/5) * distance P A * distance P B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l123_12357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_vector_acute_angle_condition_l123_12385

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, -2)

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_perpendicular_condition (k : ℝ) :
  perpendicular ((k * a.1 + b.1, k * a.2 + b.2)) ((a.1 - 3 * b.1, a.2 - 3 * b.2)) ↔ k = 23 / 13 :=
by sorry

def acute_angle (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 > 0

theorem vector_acute_angle_condition (l : ℝ) :
  acute_angle a (a.1 + l * b.1, a.2 + l * b.2) ↔ l < 0 ∨ (l > 0 ∧ l < 5 / 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_vector_acute_angle_condition_l123_12385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l123_12309

theorem polynomial_remainder :
  ∃ q : Polynomial ℤ, (X : Polynomial ℤ)^101 = (X + 1)^4 * q + 
    (166650 * X^3 - 3520225 * X^2 + 67605570 * X - 1165299255) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l123_12309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_kmph_l123_12305

-- Define the distance covered by the train
noncomputable def distance : ℝ := 20

-- Define the time taken by the train in minutes
noncomputable def time_minutes : ℝ := 20

-- Define the conversion factor from minutes to hours
noncomputable def minutes_to_hours : ℝ := 1 / 60

-- Define the speed of the train
noncomputable def train_speed : ℝ := distance / (time_minutes * minutes_to_hours)

-- Theorem statement
theorem train_speed_is_60_kmph : train_speed = 60 := by
  -- Unfold definitions
  unfold train_speed distance time_minutes minutes_to_hours
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_kmph_l123_12305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_zero_f_monotone_increasing_l123_12342

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 3 else ((-x)^2 - 4*(-x) + 3)

-- State the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_def_nonneg : ∀ x, x ≥ 0 → f x = x^2 - 4*x + 3

-- Theorem 1: f(-3) = 0
theorem f_neg_three_eq_zero : f (-3) = 0 := by
  sorry

-- Theorem 2: f is monotonically increasing on [-2, 0] and [2, +∞)
theorem f_monotone_increasing :
  (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_zero_f_monotone_increasing_l123_12342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_duration_is_four_hours_l123_12374

/-- Represents a flight schedule --/
structure FlightSchedule where
  departureTime : Int
  arrivalTime : Int

/-- Calculates the duration of a flight in Moscow time --/
def flightDuration (schedule : FlightSchedule) (timeDifference : Int) : Int :=
  (schedule.arrivalTime + timeDifference * 100) - schedule.departureTime

/-- Proves that the duration of each flight is 4 hours --/
theorem flight_duration_is_four_hours 
  (flight608 : FlightSchedule)
  (flight607 : FlightSchedule)
  (timeDifference : Int)
  (h1 : flight608.departureTime = 1200)
  (h2 : flight608.arrivalTime = 1800)
  (h3 : flight607.departureTime = 800)
  (h4 : flight607.arrivalTime = 1000)
  (h5 : timeDifference = 3)
  : flightDuration flight608 (-timeDifference) = 400 ∧ 
    flightDuration flight607 timeDifference = 400 :=
by
  sorry

#check flight_duration_is_four_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_duration_is_four_hours_l123_12374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additive_homomorphism_iff_weighted_sum_l123_12313

/-- The set of all integer sequences -/
def IntegerSequence := ℕ → ℤ

/-- The property that a function f: A → ℤ satisfies f(x + y) = f(x) + f(y) for all x, y ∈ A -/
def IsAdditiveHomomorphism (f : IntegerSequence → ℤ) : Prop :=
  ∀ x y : IntegerSequence, f (fun n => x n + y n) = f x + f y

/-- The form of the function as a sum of weighted elements -/
def HasWeightedSum (f : IntegerSequence → ℤ) : Prop :=
  ∃ a : ℕ → ℤ, ∀ x : IntegerSequence, f x = ∑' n, a n * x n

/-- The main theorem stating the equivalence between being an additive homomorphism
    and having a weighted sum form -/
theorem additive_homomorphism_iff_weighted_sum (f : IntegerSequence → ℤ) :
    IsAdditiveHomomorphism f ↔ HasWeightedSum f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additive_homomorphism_iff_weighted_sum_l123_12313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_difference_l123_12325

theorem equation_solution_difference : ∃ (x y : ℝ), 
  ((5 - x^2 / 3)^(1/3 : ℝ) = -3 ∧ 
   (5 - y^2 / 3)^(1/3 : ℝ) = -3 ∧ 
   x ≠ y ∧
   |x - y| = 8 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_difference_l123_12325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_exists_unique_k_for_dot_product_distance_mn_when_k_is_one_l123_12392

-- Define the line l: y = kx + 1
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle C: (x-2)^2 + (y-3)^2 = 1
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the intersection points M and N
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ y = line k x ∧ circle_eq x y}

-- Theorem 1: Range of k
theorem range_of_k :
  ∀ k : ℝ, (intersection_points k).Nonempty ↔ (4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3 :=
sorry

-- Theorem 2: Existence of k for dot product = 12
theorem exists_unique_k_for_dot_product :
  ∃! k : ℝ, ∀ m n : ℝ × ℝ, m ∈ intersection_points k → n ∈ intersection_points k →
    m.1 * n.1 + m.2 * n.2 = 12 :=
sorry

-- Theorem 3: Distance between M and N when k = 1
theorem distance_mn_when_k_is_one :
  ∀ m n : ℝ × ℝ, m ∈ intersection_points 1 → n ∈ intersection_points 1 →
    (m.1 - n.1)^2 + (m.2 - n.2)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_exists_unique_k_for_dot_product_distance_mn_when_k_is_one_l123_12392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_mod_84_l123_12337

def a : ℕ := 5555555555555555555555555555555555555555555555555555555

theorem a_mod_84 : a % 84 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_mod_84_l123_12337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XMYN_is_54_l123_12387

/-- Given a triangle XYZ with medians XM and YN intersecting at Q,
    this function calculates the area of quadrilateral XMYN. -/
noncomputable def areaXMYN (qn qm mn : ℝ) : ℝ :=
  let ym := 2 * qn
  let xn := 2 * qm
  (1 / 2) * (xn * qn + qm * qn + ym * qm + xn * ym)

/-- Theorem stating that the area of quadrilateral XMYN is 54
    given the specific measurements in the problem. -/
theorem area_XMYN_is_54 :
  areaXMYN 3 4 5 = 54 := by
  -- Unfold the definition of areaXMYN
  unfold areaXMYN
  -- Simplify the arithmetic
  simp [mul_add, add_mul, mul_comm, mul_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XMYN_is_54_l123_12387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_contains_arithmetic_progression_l123_12390

def sequence_a : ℕ → ℕ
  | 0 => 1000000  -- Add this case for 0
  | 1 => 1000000
  | n + 2 => (n + 1) * ((sequence_a (n + 1)) / (n + 1)) + (n + 1)

theorem sequence_a_contains_arithmetic_progression :
  ∃ (d : ℕ) (k : ℕ), ∀ (i : ℕ),
    ∃ (n : ℕ), sequence_a (n + i * d) = sequence_a n + i * k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_contains_arithmetic_progression_l123_12390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separated_distance_between_centers_is_5_l123_12356

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 9

-- Define the centers and radii
def center_O₁ : ℝ × ℝ := (0, 0)
def center_O₂ : ℝ × ℝ := (3, -4)
def radius_O₁ : ℝ := 1
def radius_O₂ : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2)

-- Theorem statement
theorem circles_are_separated :
  distance_between_centers > radius_O₁ + radius_O₂ := by
  -- Proof goes here
  sorry

-- Additional theorem to show the exact value of distance_between_centers
theorem distance_between_centers_is_5 :
  distance_between_centers = 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separated_distance_between_centers_is_5_l123_12356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_l123_12359

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -1 ∧ x ≤ 2 then 3 - x^2
  else if x > 2 ∧ x ≤ 5 then x - 3
  else 0  -- Define a default value for x outside the given intervals

-- Theorem statement
theorem solutions_of_f (x : ℝ) :
  f x = 1 ↔ x = Real.sqrt 2 ∨ x = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_l123_12359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_face_value_example_l123_12380

/-- The face value of a stock given its cost price, discount, and brokerage fee -/
noncomputable def stock_face_value (cost_price : ℝ) (discount_rate : ℝ) (brokerage_rate : ℝ) : ℝ :=
  cost_price / ((1 - discount_rate) * (1 + brokerage_rate))

/-- Theorem: The face value of a stock with a cost price of 93.2, 
    7% discount, and 0.2% brokerage fee is approximately 99.89 -/
theorem stock_face_value_example : 
  ∃ ε > 0, |stock_face_value 93.2 0.07 0.002 - 99.89| < ε :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_face_value_example_l123_12380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_common_point_eccentricity_l123_12329

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a conic section (ellipse or hyperbola) -/
structure ConicSection where
  foci : Point × Point
  eccentricity : ℝ

/-- Calculates the angle between three points -/
noncomputable def angle (a b c : Point) : ℝ := sorry

/-- Checks if a point is on a conic section -/
def isOnConic (p : Point) (c : ConicSection) : Prop := sorry

theorem hyperbola_ellipse_common_point_eccentricity 
  (h : ConicSection) 
  (e : ConicSection) 
  (p : Point) :
  h.eccentricity > 1 →
  e.eccentricity = Real.sqrt 2 / 2 →
  h.foci = e.foci →
  isOnConic p h →
  isOnConic p e →
  angle h.foci.fst p h.foci.snd = π / 3 →
  h.eccentricity = Real.sqrt 6 / 2 := by
  sorry

#check hyperbola_ellipse_common_point_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_common_point_eccentricity_l123_12329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l123_12314

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ (m : ℝ), m = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l123_12314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_axes_l123_12379

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := -Real.cos x

-- Define the area calculation function
noncomputable def area_under_curve (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

-- Theorem statement
theorem area_enclosed_by_curve_and_axes : 
  area_under_curve (π/2) (3*π/2) - area_under_curve 0 (π/2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_axes_l123_12379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_downstream_approx_120_l123_12389

/-- Calculates the distance traveled downstream given the rowing speed in still water,
    the current speed, and the time taken. -/
noncomputable def distance_downstream (rowing_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (rowing_speed + current_speed) * 1000 / 3600 * time

/-- Proves that the distance traveled downstream is approximately 120 meters
    given the specified conditions. -/
theorem distance_downstream_approx_120 :
  let rowing_speed := (15 : ℝ)
  let current_speed := (3 : ℝ)
  let time := (23.998080153587715 : ℝ)
  abs (distance_downstream rowing_speed current_speed time - 120) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_downstream_approx_120_l123_12389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteers_2022_l123_12326

/-- The number of volunteers in a community garden over three years -/
def volunteers_count (initial : ℕ) (increase_2021 : ℚ) (increase_2022 : ℚ) : ℕ :=
  (((initial : ℚ) * (1 + increase_2021) * (1 + increase_2022)).floor).toNat

/-- Theorem stating the expected number of volunteers in 2022 -/
theorem volunteers_2022 :
  volunteers_count 1200 (15 / 100) (30 / 100) = 1794 := by
  -- Unfold the definition of volunteers_count
  unfold volunteers_count
  -- Simplify the arithmetic expressions
  simp [Nat.cast_ofNat, Rat.mul_def, Rat.add_def]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteers_2022_l123_12326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l123_12384

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Define the slope of the tangent line at x = 1
noncomputable def m : ℝ := f' 1

-- Define the y-intercept of the tangent line
noncomputable def c : ℝ := f 1 - m * 1

-- Define the x-intercept of the tangent line
noncomputable def x_intercept : ℝ := -c / m

-- Define the y-intercept of the tangent line
noncomputable def y_intercept : ℝ := c

-- Define the area of the triangle
noncomputable def triangle_area : ℝ := (1/2) * x_intercept * (abs y_intercept)

-- Theorem statement
theorem tangent_line_triangle_area : triangle_area = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l123_12384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_tangents_l123_12307

/-- The distance between the tangents of the parabolas y = x^2 + 1 and x = y^2 + 1
    that are parallel to the line y = x -/
theorem distance_between_parallel_tangents :
  let f (x : ℝ) := x^2 + 1
  let g (y : ℝ) := y^2 + 1
  let l (x : ℝ) := x
  ∃ (t₁ t₂ : ℝ → ℝ),
    (∀ x, HasDerivAt t₁ 1 x) ∧
    (∀ y, HasDerivAt t₂ 1 y) ∧
    (∃ x₁, f x₁ = t₁ x₁ ∧ ∀ x, x ≠ x₁ → f x ≠ t₁ x) ∧
    (∃ y₁, g y₁ = t₂ y₁ ∧ ∀ y, y ≠ y₁ → g y ≠ t₂ y) ∧
    (∃ d, d = |t₁ 0 - t₂ 0| ∧ d = 3 * Real.sqrt 2 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_tangents_l123_12307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangent_line_l123_12311

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by the equation (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle2D where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculate the distance from a point to a line -/
noncomputable def point_to_line_distance (x y : ℝ) (l : Line2D) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line is tangent to a circle -/
def is_tangent_to_circle (l : Line2D) (c : Circle2D) : Prop :=
  point_to_line_distance c.h c.k l = c.r

theorem parallel_tangent_line : 
  let l1 : Line2D := ⟨2, 1, 1⟩
  let l2 : Line2D := ⟨2, 1, 5⟩
  let c : Circle2D := ⟨0, 0, Real.sqrt 5⟩
  are_parallel l1 l2 ∧ is_tangent_to_circle l2 c := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangent_line_l123_12311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_origin_l123_12370

/-- The circle C₂ defined by the equation x² + y² + 4x - 4y = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- A point M on the circle C₂ -/
def M : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_from_origin :
  ∃ M_max : ℝ × ℝ, C₂ M_max.1 M_max.2 ∧
  distance O M_max = 4 * Real.sqrt 2 ∧
  ∀ M' : ℝ × ℝ, C₂ M'.1 M'.2 → distance O M' ≤ distance O M_max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_origin_l123_12370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_arrangement_l123_12382

theorem optimal_arrangement (a₁ a₂ a₃ a₄ : ℝ) (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) :
  let Φ (i₁ i₂ i₃ i₄ : ℝ) := (i₁ - i₂)^2 + (i₂ - i₃)^2 + (i₃ - i₄)^2 + (i₄ - i₁)^2
  ∀ (σ : Equiv.Perm (Fin 4)), Φ a₁ a₂ a₄ a₃ ≤ Φ (σ 0) (σ 1) (σ 2) (σ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_arrangement_l123_12382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_odd_prime_and_power_l123_12343

/-- The distance between a real number and the nearest integer -/
noncomputable def dist_to_nearest_int (x : ℝ) : ℝ := |x - round x|

/-- The main theorem -/
theorem exists_odd_prime_and_power (a b : ℕ+) : 
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ Odd p ∧ 
    dist_to_nearest_int (a / p^k : ℝ) + 
    dist_to_nearest_int (b / p^k : ℝ) + 
    dist_to_nearest_int ((a + b) / p^k : ℝ) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_odd_prime_and_power_l123_12343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l123_12345

-- Define the ellipse parameters
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 2

-- Define the focal length
noncomputable def focal_length : ℝ := 2 * Real.sqrt 2

-- Define the slope of line l
variable (k : ℝ)

-- Define the point through which line l passes
noncomputable def point_on_l : ℝ × ℝ := (-Real.sqrt 2, 1)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line equation
def is_on_line (x y : ℝ) : Prop := y = k * x - 2

-- Define the intersection points
noncomputable def A : ℝ → ℝ × ℝ := sorry
noncomputable def B : ℝ → ℝ × ℝ := sorry

-- Define point P
noncomputable def P (k : ℝ) : ℝ × ℝ := (2 / k, 0)

-- Define point C (reflection of A over x-axis)
noncomputable def C (k : ℝ) : ℝ × ℝ := ((A k).1, -(A k).2)

-- Define point Q
noncomputable def Q (k : ℝ) : ℝ × ℝ := (2 * k, 0)

-- Theorem to prove
theorem constant_product (k : ℝ) (h : k^2 > 1/2) : |((P k).1)| * |((Q k).1)| = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l123_12345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_for_positive_f_l123_12322

-- Define the function f
noncomputable def f (x θ : ℝ) : ℝ := 1 / (x - 2)^2 - 2*x + Real.cos (2*θ) - 3*Real.sin θ + 2

-- State the theorem
theorem theta_range_for_positive_f :
  ∀ θ : ℝ, 0 < θ ∧ θ < π ∧
  (∀ x : ℝ, x < 2 → f x θ > 0) →
  θ < π/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_for_positive_f_l123_12322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l123_12312

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

theorem train_speed_approximation :
  let train_length : ℝ := 400
  let bridge_length : ℝ := 300
  let crossing_time : ℝ := 45
  abs (train_speed train_length bridge_length crossing_time - 15.56) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l123_12312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_shape_is_circle_l123_12369

-- Define the basic geometric objects
structure Plane : Type
structure Cylinder : Type
structure Sphere : Type

-- Define the intersection shape
inductive IntersectionShape
| Circle
| Ellipse
| Other

-- Define the intersection function
def intersect : Plane → (Cylinder ⊕ Sphere) → IntersectionShape
| _, _ => IntersectionShape.Circle  -- Simplified for this example

-- Theorem statement
theorem intersection_shape_is_circle 
  (p : Plane) (c : Cylinder) (s : Sphere) :
  intersect p (Sum.inl c) = intersect p (Sum.inr s) → 
  intersect p (Sum.inr s) = IntersectionShape.Circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_shape_is_circle_l123_12369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_N_l123_12358

/-- Definition of N as the sum of numbers consisting of all 9s with lengths from 1 to 400 -/
def N : ℕ := (Finset.range 400).sum (fun i => 10^(i+1) - 1)

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of N is 405 -/
theorem sum_of_digits_N : sum_of_digits N = 405 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_N_l123_12358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l123_12344

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => arithmeticSequence a₁ d n + d

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℤ) 
  (h_arithmetic : ∃ a₁ d : ℤ, ∀ n, a n = arithmeticSequence a₁ d n) 
  (h_first : a 1 = -11) 
  (h_sum : a 4 + a 6 = -6) : 
  a 3 = -7 := by
  sorry

#check arithmetic_sequence_third_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l123_12344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l123_12368

/-- Represents a class of students -/
structure MyClass where
  boys : ℕ
  girls : ℕ

/-- The average number of items made per student is 18 -/
def average_items (c : MyClass) : ℚ := 18

/-- Each girl makes 20 items -/
def items_per_girl : ℕ := 20

/-- Each boy makes 15 items -/
def items_per_boy : ℕ := 15

/-- The theorem stating the ratio of boys to girls -/
theorem boys_to_girls_ratio (c : MyClass) :
  (average_items c * (c.boys + c.girls : ℚ) = items_per_boy * c.boys + items_per_girl * c.girls) →
  3 * c.boys = 2 * c.girls :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l123_12368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l123_12331

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- The height from vertex B to side BC -/
noncomputable def height_from_B (t : Triangle) : ℝ := (2 * area t) / t.a

/-- Main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 3)
  (h2 : t.c = 2)
  (h3 : area t = (3 * Real.sqrt 3) / 2)
  (h4 : t.A > π / 2) :  -- A is obtuse
  Real.sin t.A = Real.sqrt 3 / 2 ∧ 
  height_from_B t = (3 * Real.sqrt 57) / 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l123_12331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l123_12324

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.exp x - b * x

theorem problem_solution (a b : ℝ) :
  (∀ x > 0, f a x + x ≤ (x / b) * g b x) →
  (∀ x, (deriv (f a)) x * (1 : ℝ) = -1) →
  (a = 2 ∧ 
   (b ≤ 0 → Monotone (g b)) ∧
   (b > 0 → 
     (StrictMono (fun x => g b x) ∧
      StrictMono (fun x => -g b x))) ∧
   (0 ≤ b ∧ b ≤ Real.exp 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l123_12324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_water_tank_height_l123_12336

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  volume : ℝ

/-- Calculates the scale factor between two water tanks based on their volumes -/
noncomputable def scaleFactor (original : WaterTank) (scaled : WaterTank) : ℝ :=
  (original.volume / scaled.volume) ^ (1/3 : ℝ)

/-- Theorem: The height of a scaled-down cylindrical water tank -/
theorem scaled_water_tank_height 
  (original : WaterTank)
  (scaled : WaterTank)
  (h1 : original.height = 50)
  (h2 : original.volume = 200000)
  (h3 : scaled.volume = 0.2)
  : scaled.height = 0.5 := by
  sorry

#check scaled_water_tank_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_water_tank_height_l123_12336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_fifth_terms_l123_12388

/-- A sequence where each term is 1/4 of the previous term -/
noncomputable def geometric_sequence (a₀ : ℝ) : ℕ → ℝ :=
  λ n => a₀ * (1/4)^n

theorem sum_of_fourth_and_fifth_terms :
  ∃ a₀ : ℝ, (geometric_sequence a₀ 5 = 4) ∧
    (geometric_sequence a₀ 3 + geometric_sequence a₀ 4 = 80) := by
  use 4096
  constructor
  · simp [geometric_sequence]
    norm_num
  · simp [geometric_sequence]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_fifth_terms_l123_12388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l123_12399

/-- A circle passes through the point (0,2) and is tangent to the parabola y = x^2 at (1,1).
    Its center is at (1/3, 4/3). -/
theorem circle_center (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) : 
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1^2 + center.2^2)) →  -- C is a circle
  (0, 2) ∈ C →  -- C passes through (0,2)
  (1, 1) ∈ C →  -- C passes through (1,1)
  (∀ (x : ℝ), x ≠ 1 → ((x^2 - 1) / (x - 1) - 2*x + 1)^2 + (x - 1)^2 > (center.1^2 + center.2^2)) →  -- C is tangent to y = x^2 at (1,1)
  center = (1/3, 4/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l123_12399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_sum_5_to_35_l123_12375

noncomputable def ceiling_sqrt_sum (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (fun i => (Int.ceil (Real.sqrt (i + a : ℝ))).toNat)

theorem ceiling_sqrt_sum_5_to_35 :
  ceiling_sqrt_sum 5 35 = 148 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_sum_5_to_35_l123_12375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplifies_to_four_l123_12334

-- Define the expression as a function of a
noncomputable def expression (a : ℝ) : ℝ := 
  (3*a - 3)/a / ((a^2 - 2*a + 1)/a^2) - a/(a - 1)

-- State the theorem
theorem expression_simplifies_to_four : 
  expression 2 = 4 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplifies_to_four_l123_12334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l123_12319

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos α = Real.sqrt 5 / 5) (h4 : Real.sin (α + β) = 3/5) : 
  Real.cos β = 2 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l123_12319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l123_12371

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle ABC -/
def triangle_ABC : Triangle :=
  { A := (4, 0),
    B := (6, 7),
    C := (0, 3) }

/-- Altitude from A to BC -/
def altitude_equation (t : Triangle) : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * x + 2 * y - 12 = 0

/-- Line through B bisecting the area of triangle ABC -/
def area_bisector_equation (t : Triangle) : ℝ → ℝ → Prop :=
  λ x y ↦ 11 * x - 8 * y - 10 = 0

/-- Main theorem stating the equations of the altitude and area bisector -/
theorem triangle_properties (t : Triangle) (h : t = triangle_ABC) :
  (∀ x y, altitude_equation t x y ↔ 3 * x + 2 * y - 12 = 0) ∧
  (∀ x y, area_bisector_equation t x y ↔ 11 * x - 8 * y - 10 = 0) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l123_12371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_prime_factors_l123_12300

def expression : ℕ → ℕ
| 2 => 23
| 3 => 19
| 5 => 17
| 7 => 13
| 11 => 11
| 13 => 9
| 17 => 7
| 19 => 5
| 23 => 3
| 29 => 2
| _ => 0

theorem total_prime_factors :
  (List.sum (List.map expression [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])) = 109 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_prime_factors_l123_12300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l123_12301

/-- The speed of a train in km/hr given its length and time to pass a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A 240-meter long train that passes a fixed point in 8 seconds has a speed of 108 km/hr -/
theorem train_speed_theorem :
  train_speed 240 8 = 108 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [div_mul_eq_mul_div]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l123_12301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_solution_range_l123_12355

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + a*x + a)

-- Theorem for part I
theorem monotonicity_intervals (x : ℝ) :
  let a := 1
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -2 → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, -2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ -1 → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) :=
by sorry

-- Theorem for part II
theorem solution_range (a : ℝ) :
  (∃ x, x ≥ a ∧ f a x ≤ Real.exp a) ↔ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_solution_range_l123_12355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_age_combinations_l123_12393

/-- Represents the set of digits used to form the plant's age -/
def digits : Finset ℕ := {3, 4, 8, 9}

/-- The total number of digits used -/
def total_digits : ℕ := 6

/-- The number of times the digit 3 appears -/
def count_threes : ℕ := 3

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- The number of valid ages for the plant -/
def valid_ages : ℕ := 20

theorem plant_age_combinations :
  (Finset.filter (fun n => n % 2 = 0) digits).card * (Nat.factorial (total_digits - 1)) / (Nat.factorial count_threes) = valid_ages :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_age_combinations_l123_12393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equal_1_functions_equal_2_functions_equal_3_l123_12349

-- Define the functions
noncomputable def f1 (x : ℝ) := |x|
noncomputable def g1 (x : ℝ) := Real.sqrt (x^2)

noncomputable def f2 (x : ℝ) := x^0
noncomputable def g2 (x : ℝ) := 1 / (x^0)

def f3 (x : ℝ) := x^2 - 2*x - 1
def g3 (t : ℝ) := t^2 - 2*t - 1

-- State the theorems
theorem functions_equal_1 : ∀ x : ℝ, f1 x = g1 x := by sorry

theorem functions_equal_2 : ∀ x : ℝ, x ≠ 0 → f2 x = g2 x := by sorry

theorem functions_equal_3 : ∀ x : ℝ, f3 x = g3 x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equal_1_functions_equal_2_functions_equal_3_l123_12349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l123_12352

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define proposition p
def prop_p (Line Plane : Type) 
           (subset : Line → Plane → Prop) 
           (parallel_planes : Plane → Plane → Prop) : Prop :=
  ∀ (m : Line) (α β : Plane),
    subset m α → parallel_planes α β → parallel_planes α β

-- Define proposition q
def prop_q (Line Plane : Type) 
           (perpendicular : Line → Plane → Prop) 
           (parallel_lines : Line → Line → Prop)
           (parallel_planes : Plane → Plane → Prop) : Prop :=
  ∀ (l m : Line) (α β : Plane),
    perpendicular m α → perpendicular l β → parallel_planes α β → parallel_lines m l

-- State the theorem
theorem problem_statement : 
  ∀ (Line Plane : Type)
    (subset : Line → Plane → Prop)
    (parallel_lines : Line → Line → Prop)
    (parallel_planes : Plane → Plane → Prop)
    (perpendicular : Line → Plane → Prop),
  ¬(prop_p Line Plane subset parallel_planes) ∧ 
   (prop_q Line Plane perpendicular parallel_lines parallel_planes) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l123_12352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l123_12310

/-- Given functions f and g, prove that if h(x) = f(x) - g(x) satisfies
    (h(x₁) - h(x₂)) / (x₁ - x₂) > 2 for any two unequal positive numbers x₁ and x₂,
    then a ≤ -4. -/
theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    (let f : ℝ → ℝ := λ x ↦ (1/2) * x^2 - 2*x
     let g : ℝ → ℝ := λ x ↦ a * Real.log x
     let h : ℝ → ℝ := λ x ↦ f x - g x
     (h x₁ - h x₂) / (x₁ - x₂) > 2)) →
  a ≤ -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l123_12310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_area_ratio_l123_12348

/-- Given a rectangle with length 4 times its width, a circle inscribed in the rectangle,
    a square inscribed in this circle, and a smaller circle inscribed in the square,
    the ratio of the area of the smaller circle to the area of the rectangle is π/16. -/
theorem inscribed_circles_area_ratio (w : ℝ) (hw : w > 0) : 
  (π * (w / 2)^2) / (4 * w^2) = π / 16 := by
  -- Expand the left-hand side
  have h1 : (π * (w / 2)^2) / (4 * w^2) = (π * w^2 / 4) / (4 * w^2) := by
    ring
  -- Simplify the fraction
  have h2 : (π * w^2 / 4) / (4 * w^2) = π / 16 := by
    field_simp
    ring
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_area_ratio_l123_12348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_when_f_nonpositive_range_a_when_g_has_max_less_than_zero_l123_12367

/-- Given function f(x) = ln x - (1/2)ax + 1, where a is a real number -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x + 1

/-- Function g(x) = xf(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * f a x

/-- Theorem stating the minimum value of a when f(x) ≤ 0 for all x > 0 -/
theorem min_a_when_f_nonpositive (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → a ≥ 2 := by
  sorry

/-- Theorem stating the range of a when g(x) has a maximum value less than 0 -/
theorem range_a_when_g_has_max_less_than_zero (a : ℝ) :
  (∃ x₀ > 0, ∀ x > 0, g a x ≤ g a x₀) →
  (∃ x₁ > 0, g a x₁ < 0) →
  2 < a ∧ a < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_when_f_nonpositive_range_a_when_g_has_max_less_than_zero_l123_12367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_g_inequality_l123_12333

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

-- State the theorems
theorem f_positive : ∀ x > -1, f x > 0 := by sorry

theorem g_inequality : ∃ a : ℝ, a > 0 ∧ (∀ x > -1, g x ≤ a * x + 1) ∧ 
  (∀ b : ℝ, b > 0 → (∀ x > -1, g x ≤ b * x + 1) → b = a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_g_inequality_l123_12333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_first_three_miles_l123_12394

/-- Represents the hike scenario with given parameters --/
structure HikeScenario where
  totalDistance : ℚ
  initialWater : ℚ
  finalWater : ℚ
  timeTaken : ℚ
  leakRate : ℚ
  waterLastMile : ℚ

/-- Calculates the water consumption per mile for the first 3 miles of the hike --/
def waterPerMileFirstThreeMiles (h : HikeScenario) : ℚ :=
  (h.initialWater - h.leakRate * h.timeTaken - h.finalWater - h.waterLastMile) / (h.totalDistance - 1)

/-- Theorem stating that given the specific hike conditions, Harry drank 1/3 cup per mile during the first 3 miles --/
theorem water_consumption_first_three_miles :
  let h : HikeScenario := {
    totalDistance := 4
    initialWater := 10
    finalWater := 2
    timeTaken := 2
    leakRate := 1
    waterLastMile := 3
  }
  waterPerMileFirstThreeMiles h = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_first_three_miles_l123_12394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_final_money_l123_12373

def weekly_allowance : ℕ := 8
def movie_spending_fraction : ℚ := 1/2
def car_washing_earnings : ℕ := 8

theorem alyssa_final_money : 
  weekly_allowance - (weekly_allowance * movie_spending_fraction).floor + car_washing_earnings = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_final_money_l123_12373
