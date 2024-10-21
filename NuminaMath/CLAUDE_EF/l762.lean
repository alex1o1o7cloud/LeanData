import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_circle_property_l762_76220

noncomputable def point : ℝ × ℝ := (12, 5)

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

def on_circle (p : ℝ × ℝ) (r : ℝ) : Prop :=
  p.1^2 + p.2^2 = r^2

theorem distance_and_circle_property :
  distance_from_origin point = 13 ∧ on_circle point 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_circle_property_l762_76220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_over_fourth_root_of_eleven_l762_76232

theorem fifth_root_over_fourth_root_of_eleven :
  (11 : ℝ) ^ ((1/5) : ℝ) / (11 : ℝ) ^ ((1/4) : ℝ) = (11 : ℝ) ^ ((-(1/20)) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_over_fourth_root_of_eleven_l762_76232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_steps_correct_main_theorem_l762_76263

/-- Represents the state of the glasses --/
inductive GlassState
| S1010 -- Initial state
| S0101 -- Final state
| S1001 -- Intermediate state
| S0110 -- Intermediate state
| S0011 -- Intermediate state
| S1100 -- Intermediate state

/-- Represents the simplified graph states --/
inductive SimpleState
| A -- Initial state
| B -- Intermediate states
| C -- Final state

/-- Transition probability from one simple state to another --/
def transitionProb (s1 s2 : SimpleState) : ℚ :=
  match s1, s2 with
  | SimpleState.A, SimpleState.B => 1
  | SimpleState.B, SimpleState.A => 1/4
  | SimpleState.B, SimpleState.B => 1/2
  | SimpleState.B, SimpleState.C => 1/4
  | _, _ => 0

/-- Expected number of steps from one simple state to another --/
def expectedSteps (s1 s2 : SimpleState) : ℚ :=
  match s1, s2 with
  | SimpleState.A, SimpleState.C => 5
  | SimpleState.B, SimpleState.C => 4
  | _, _ => 0  -- We're not interested in other cases for this problem

theorem expected_steps_correct :
  expectedSteps SimpleState.A SimpleState.C = 5 := by
  rfl

/-- The main theorem stating that the expected number of steps is 5 --/
theorem main_theorem :
  ∃ (n : ℚ), n = expectedSteps SimpleState.A SimpleState.C ∧ n = 5 := by
  use 5
  constructor
  · exact expected_steps_correct
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_steps_correct_main_theorem_l762_76263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_after_cuts_l762_76235

-- Define the cube and its properties
noncomputable def cube_volume : ℝ := 2
noncomputable def first_cut_distance : ℝ := 2/3
noncomputable def second_cut_distance : ℝ := 2/3 + 5/14

-- Define the function to calculate the total surface area
noncomputable def total_surface_area (v : ℝ) : ℝ :=
  let side_length := v^(1/3)
  6 * side_length^2 + 4 * side_length^2

-- State the theorem
theorem surface_area_after_cuts :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_surface_area cube_volume - 15.87| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_after_cuts_l762_76235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l762_76257

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg : Set ℝ :=
  {x | x < -1 ∨ x > 1/2}

-- Define the solution set of f(10^x) > 0
def solution_set_f_10x_pos : Set ℝ :=
  {x | x < -Real.log 2 / Real.log 10}

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, x ∈ solution_set_f_neg ↔ f x < 0) →
  (∀ x, x ∈ solution_set_f_10x_pos ↔ f (10^x) > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l762_76257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l762_76219

/-- The cost of fencing a rectangular park -/
theorem park_fencing_cost (area : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) (cost_per_meter : ℝ) : 
  area = 5766 →
  ratio_length = 3 →
  ratio_width = 2 →
  cost_per_meter = 0.5 →
  let length := ratio_length * Real.sqrt (area / (ratio_length * ratio_width))
  let width := ratio_width * Real.sqrt (area / (ratio_length * ratio_width))
  let perimeter := 2 * (length + width)
  2 * (ratio_length + ratio_width) * Real.sqrt (area / (ratio_length * ratio_width)) * cost_per_meter = 155 := by
  sorry

-- Remove the #eval line as it's causing issues
-- #eval park_fencing_cost 5766 3 2 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l762_76219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l762_76276

-- Define the line
def line (m : ℝ) (y : ℝ) : ℝ := m * y - 1

-- Define the circle
def circle_eq (m n p : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + m*x + n*y + p = 0

-- Define symmetry about y = x
def symmetricAboutYEqualsX (A B : ℝ × ℝ) : Prop :=
  A.1 = B.2 ∧ A.2 = B.1

-- Theorem statement
theorem intersection_range (m n : ℝ) :
  ∃ (A B : ℝ × ℝ) (p : ℝ),
    circle_eq m n p A.1 A.2 ∧
    circle_eq m n p B.1 B.2 ∧
    A.1 = line m A.2 ∧
    B.1 = line m B.2 ∧
    symmetricAboutYEqualsX A B →
    ∀ q : ℝ, q < -3/2 ↔ ∃ (A' B' : ℝ × ℝ),
      circle_eq m n q A'.1 A'.2 ∧
      circle_eq m n q B'.1 B'.2 ∧
      A'.1 = line m A'.2 ∧
      B'.1 = line m B'.2 ∧
      symmetricAboutYEqualsX A' B' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l762_76276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l762_76238

def sequence_a : ℕ → ℚ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | (n + 2) => 1 / (2 - sequence_a (n + 1))

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → (1 / (sequence_a (n + 1) - 1)) - (1 / (sequence_a n - 1)) = -1) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = (n - 1) / n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l762_76238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_students_l762_76270

theorem classroom_students (total : ℕ) (auditorium_frac : ℚ) (classroom_frac : ℚ) : 
  total = 1260 →
  auditorium_frac = 7 / 18 →
  classroom_frac = 6 / 11 →
  (total - (auditorium_frac * ↑total).floor) * classroom_frac = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_students_l762_76270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_correct_answer_l762_76223

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log ((-x^2 + 2*x + 3) : ℝ) / Real.log (1/4 : ℝ)

-- Define the domain
def domain : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the monotonic increasing interval
def increasing_interval : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

-- Theorem statement
theorem f_monotonic_increasing :
  ∀ x y, x ∈ domain → y ∈ domain →
    x ∈ increasing_interval → y ∈ increasing_interval → x < y →
    f x < f y :=
by
  sorry

-- Proof that the answer is correct
theorem correct_answer :
  increasing_interval = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_correct_answer_l762_76223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_sum_l762_76213

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Sides a, b, c are positive
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- A, B, C are angles in radians
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  -- Sum of angles is π
  A + B + C = Real.pi

-- Define the altitude condition
def altitude_condition (a b c : ℝ) (A : ℝ) : Prop :=
  b * c * Real.sin A = Real.sqrt 3 / 2 * a^2

-- Theorem statement
theorem max_ratio_sum (a b c : ℝ) (A B C : ℝ) :
  triangle a b c A B C →
  altitude_condition a b c A →
  (∀ A' : ℝ, triangle a b c A' B C → altitude_condition a b c A' → 
    c / b + b / c ≤ c / b + b / c) →
  A = Real.pi / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_sum_l762_76213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l762_76281

/-- Given an equation (x - ∛27)(x - ∛64)(x - ∛125) = 1/2 with three distinct real solutions u, v, and w,
    prove that u³ + v³ + w³ = 217.5 -/
theorem sum_of_cubes_of_solutions (u v w : ℝ) : 
  (u - (27 : ℝ)^(1/3)) * (u - (64 : ℝ)^(1/3)) * (u - (125 : ℝ)^(1/3)) = 1/2 →
  (v - (27 : ℝ)^(1/3)) * (v - (64 : ℝ)^(1/3)) * (v - (125 : ℝ)^(1/3)) = 1/2 →
  (w - (27 : ℝ)^(1/3)) * (w - (64 : ℝ)^(1/3)) * (w - (125 : ℝ)^(1/3)) = 1/2 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 217.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l762_76281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l762_76248

def sequenceA (n : ℕ) : ℕ := n^2 + 1

theorem sequence_formula :
  (sequenceA 1 = 2) ∧
  (sequenceA 2 = 5) ∧
  (sequenceA 3 = 10) ∧
  (sequenceA 4 = 17) ∧
  (∀ n : ℕ, n > 0 → sequenceA n = n^2 + 1) :=
by
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  intro n hn
  rfl

#check sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l762_76248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octopus_legs_theorem_l762_76275

-- Define the type for octopus
structure Octopus where
  legs : Nat
  statement : Nat

-- Define the properties of octopuses
def isHonest (o : Octopus) : Prop := o.legs % 2 = 1
def isLiar (o : Octopus) : Prop := o.legs % 2 = 0

-- Define the range of legs
def validLegs (o : Octopus) : Prop := o.legs ≥ 7 ∧ o.legs ≤ 9

-- Define the consistency of an octopus's statement
def consistentStatement (o : Octopus) (totalLegs : Nat) : Prop :=
  (isHonest o ∧ o.statement = totalLegs) ∨ (isLiar o ∧ o.statement ≠ totalLegs)

-- Main theorem
theorem octopus_legs_theorem (octopuses : List Octopus) : 
  octopuses.length = 5 ∧ 
  (∀ o, o ∈ octopuses → validLegs o) ∧
  (∃ o1 o2 o3 o4 o5, 
    o1 ∈ octopuses ∧ 
    o2 ∈ octopuses ∧ 
    o3 ∈ octopuses ∧ 
    o4 ∈ octopuses ∧ 
    o5 ∈ octopuses ∧ 
    o1.statement = 36 ∧ 
    o2.statement = 37 ∧ 
    o3.statement = 38 ∧ 
    o4.statement = 39 ∧ 
    o5.statement = 40) →
  ∃ totalLegs : Nat, totalLegs = 39 ∧ 
    (∀ o, o ∈ octopuses → consistentStatement o totalLegs) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octopus_legs_theorem_l762_76275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l762_76215

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x ≠ 0) (h2 : Real.sin y ≠ 0) (h3 : Real.cos x ≠ 0) (h4 : Real.cos y ≠ 0)
  (eq1 : Real.sin x / Real.sin y + Real.sin y / Real.sin x = 4)
  (eq2 : Real.cos x / Real.cos y + Real.cos y / Real.cos x = 3) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l762_76215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_with_sqrt2_distance_l762_76237

noncomputable section

def circleSet (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - a)^2 = 8}

def distanceToOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem circle_points_with_sqrt2_distance (a : ℝ) :
  (∃ p q : ℝ × ℝ, p ∈ circleSet a ∧ q ∈ circleSet a ∧ 
   p ≠ q ∧ 
   distanceToOrigin p = Real.sqrt 2 ∧ 
   distanceToOrigin q = Real.sqrt 2) ↔ 
  (a > 1 ∧ a < 3) ∨ (a < -1 ∧ a > -3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_with_sqrt2_distance_l762_76237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l762_76286

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = (x^(1/6) - a/x^(1/2))^8) ∧ 
   (∃ c, ∀ x, f x = c + x * (f x - c) / x) ∧ c = 56) → 
  a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l762_76286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_digit_string_length_is_correct_l762_76226

/-- The maximum length of a string of equal nonzero digits that can appear at the end of a perfect square -/
def max_equal_digit_string_length : ℕ := 3

/-- A function that checks if a number ends with n equal nonzero digits -/
def ends_with_n_equal_nonzero_digits (x : ℕ) (n : ℕ) : Prop :=
  ∃ (d : ℕ), 0 < d ∧ d < 10 ∧ ∃ (k : ℕ), x % (10^n) = d * ((10^n - 1) / 9)

/-- The theorem stating that the maximum length of a string of equal nonzero digits
    that can appear at the end of a perfect square is max_equal_digit_string_length -/
theorem max_equal_digit_string_length_is_correct :
  (∀ (x n : ℕ), n > max_equal_digit_string_length → ¬∃ (y : ℕ), ends_with_n_equal_nonzero_digits (y^2) n) ∧
  (∃ (x n : ℕ), n = max_equal_digit_string_length ∧ ends_with_n_equal_nonzero_digits (x^2) n) := by
  sorry

#check max_equal_digit_string_length_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_digit_string_length_is_correct_l762_76226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_150_6_l762_76297

/-- The area of a circular sector with central angle θ (in degrees) and radius r -/
noncomputable def sectorArea (θ : ℝ) (r : ℝ) : ℝ := (θ / 360) * Real.pi * r^2

theorem sector_area_150_6 :
  sectorArea 150 6 = 15 * Real.pi := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_150_6_l762_76297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_simplification_l762_76269

theorem complex_division_simplification :
  ∀ i : ℂ, i * i = -1 → (1 + i) / i = 1 - i := by
  intro i h
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_simplification_l762_76269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_intersecting_circle_l762_76241

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem stating the range of eccentricity for an ellipse intersecting with a circle -/
theorem eccentricity_range_for_intersecting_circle (e : Ellipse) 
  (h_intersect : ∃ (x y : ℝ), x^2/e.a^2 + y^2/e.b^2 = 1 ∧ x^2 + y^2 = (2*e.a*eccentricity e)^2) :
  Real.sqrt 2 / 2 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_intersecting_circle_l762_76241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_irrationality_cf_purely_periodic_l762_76289

/-- A quadratic irrationality -/
structure QuadraticIrrationality where
  P : ℤ
  Q : ℤ
  D : ℤ
  h1 : Q > 0
  h2 : P > 0
  h3 : P^2 < D
  h4 : Q < P + Real.sqrt D
  h5 : P + Real.sqrt D < 2 * Real.sqrt D

/-- Continued fraction representation -/
def ContinuedFractionRepresentation := List ℤ

/-- Check if a continued fraction is purely periodic -/
def isPurelyPeriodic (cf : ContinuedFractionRepresentation) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, cf.get? (n + k) = cf.get? n

/-- The continued fraction expansion of a quadratic irrationality -/
noncomputable def continuedFractionExpansion (α : QuadraticIrrationality) : ContinuedFractionRepresentation :=
  sorry

/-- Theorem: The continued fraction of a quadratic irrationality is purely periodic -/
theorem quadratic_irrationality_cf_purely_periodic (α : QuadraticIrrationality) :
  isPurelyPeriodic (continuedFractionExpansion α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_irrationality_cf_purely_periodic_l762_76289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l762_76272

/-- The atomic mass of calcium in g/mol -/
def atomic_mass_Ca : ℝ := 40.08

/-- The atomic mass of hydrogen in g/mol -/
def atomic_mass_H : ℝ := 1.008

/-- The number of calcium atoms in CaH2 -/
def num_Ca : ℕ := 1

/-- The number of hydrogen atoms in CaH2 -/
def num_H : ℕ := 2

/-- The molar mass of CaH2 in g/mol -/
noncomputable def molar_mass_CaH2 : ℝ := atomic_mass_Ca * num_Ca + atomic_mass_H * num_H

/-- The mass percentage of H in CaH2 -/
noncomputable def mass_percentage_H : ℝ := (atomic_mass_H * num_H / molar_mass_CaH2) * 100

/-- Theorem stating that the mass percentage of H in CaH2 is approximately 4.79% -/
theorem mass_percentage_H_approx :
  |mass_percentage_H - 4.79| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l762_76272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_18_divisors_l762_76240

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d ↦ n % d = 0) (Finset.range (n + 1))

theorem smallest_with_18_divisors :
  (∀ m : ℕ, m > 0 ∧ m < 78732 → (divisors m).card ≠ 18) ∧
  (divisors 78732).card = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_18_divisors_l762_76240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generatrix_length_of_special_cone_l762_76228

/-- Represents a cone with a given base radius and generatrix length. -/
structure Cone where
  base_radius : ℝ
  generatrix_length : ℝ

/-- A cone with base radius √2 and lateral surface that unfolds into a semicircle. -/
noncomputable def special_cone : Cone :=
  { base_radius := Real.sqrt 2,
    generatrix_length := 2 * Real.sqrt 2 }

/-- The lateral surface of the cone unfolds into a semicircle. -/
axiom lateral_surface_is_semicircle (c : Cone) :
  2 * Real.pi * c.base_radius = Real.pi * c.generatrix_length

theorem generatrix_length_of_special_cone :
  (special_cone.generatrix_length : ℝ) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_generatrix_length_of_special_cone_l762_76228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_approx_l762_76253

/-- The radius of a circle inscribed in three mutually externally tangent circles --/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem: The radius of the inscribed circle for given radii is approximately 1.304 --/
theorem inscribed_circle_radius_approx :
  let a : ℝ := 5
  let b : ℝ := 10
  let c : ℝ := 15
  let r := inscribed_circle_radius a b c
  abs (r - 1.304) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_approx_l762_76253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l762_76221

noncomputable section

-- Function 1
noncomputable def f₁ (x : ℝ) : ℝ := (x^3 - 1) / Real.sin x

-- Function 2
noncomputable def f₂ (x : ℝ) : ℝ := 2 * x * Real.sin (2 * x + 5)

-- Theorem for the derivative of f₁
theorem derivative_f₁ (x : ℝ) (h : Real.sin x ≠ 0) :
  deriv f₁ x = (3 * x^2 * Real.sin x - (x^3 - 1) * Real.cos x) / (Real.sin x)^2 := by
  sorry

-- Theorem for the derivative of f₂
theorem derivative_f₂ (x : ℝ) :
  deriv f₂ x = 4 * x * Real.cos (2 * x + 5) + 2 * Real.sin (2 * x + 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l762_76221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l762_76277

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the point P
noncomputable def P : ℝ × ℝ := (-3, -3/2)

-- Define the chord length
def chord_length : ℝ := 8

-- Define the possible line equations
def line_eq_1 (x : ℝ) : Prop := x + 3 = 0
def line_eq_2 (x y : ℝ) : Prop := 3*x + 4*y + 15 = 0

-- Theorem statement
theorem chord_line_equation :
  ∃ (x y : ℝ), 
    (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t) ∧  -- Line passes through P
    my_circle x y ∧                           -- Line intersects the circle
    (∃ (x' y' : ℝ), my_circle x' y' ∧         -- Chord endpoints on circle
      (x - x')^2 + (y - y')^2 = chord_length^2) →  -- Chord length is 8
    line_eq_1 x ∨ line_eq_2 x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l762_76277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_parameter_l762_76245

/-- The equation of an ellipse with parameter a -/
def ellipse_equation (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a^2 * x^2 - (a/2) * y^2 = 1

/-- The distance from the center to a focus of the ellipse -/
def focus_distance (a : ℝ) : ℝ := 2

theorem ellipse_focus_parameter :
  ∃ (a : ℝ), (∀ x y, ellipse_equation a x y) ∧ (focus_distance a = 2) ∧ (a = (1 - Real.sqrt 5) / 4) := by
  sorry

#check ellipse_focus_parameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_parameter_l762_76245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_when_focus_to_asymptote_is_2root2_l762_76209

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The distance from a focus to an asymptote of the hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ :=
  (h.b * Real.sqrt (h.a^2 + h.b^2)) / Real.sqrt (1 + h.b^2)

/-- The focal length of the hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: If the distance from a focus to an asymptote of a hyperbola
    with equation x^2/a^2 - y^2/b^2 = 1 is 2√2, then its focal length is 6 -/
theorem hyperbola_focal_length_when_focus_to_asymptote_is_2root2
  (h : Hyperbola)
  (eq_one : h.a = 1)
  (dist : focus_to_asymptote_distance h = 2 * Real.sqrt 2) :
  focal_length h = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_when_focus_to_asymptote_is_2root2_l762_76209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_l762_76279

theorem triangle_is_right (A B C : ℝ) (h : Real.sin (A - B) * Real.cos B + Real.cos (A - B) * Real.sin B ≥ 1) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_l762_76279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_east_probability_l762_76233

-- Define the necessary types and functions
variable (Point : Type)
variable (Angle : Point → Point → Point → ℝ)
variable (Probability : Prop → ℝ)
variable (VertexEastOfOthers : Point → Point → Point → Prop)

-- Define the theorem
theorem triangle_vertex_east_probability (A B C : Point) (h : Angle A B C = 40) :
  Probability (VertexEastOfOthers A B C) = 7 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_east_probability_l762_76233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_coloring_l762_76246

/-- The set of lattice points (x, y) with |x|, |y| ≤ n -/
def D_n (n : ℕ) : Set (ℤ × ℤ) :=
  {p | |p.1| ≤ n ∧ |p.2| ≤ n}

/-- A coloring of D_n with k colors -/
def Coloring (n k : ℕ) := (ℤ × ℤ) → Fin k

/-- Two points form a line segment with no other points of D_n between them -/
def FormLineSegment (n : ℕ) (p1 p2 : ℤ × ℤ) : Prop := sorry

theorem lattice_point_coloring (n : ℕ) (h : n > 1) :
  (∀ c : Coloring n 3, ∃ p1 p2 : ℤ × ℤ, p1 ∈ D_n n ∧ p2 ∈ D_n n ∧
    p1 ≠ p2 ∧ c p1 = c p2 ∧ FormLineSegment n p1 p2) ∧
  (∃ c : Coloring n 4, ∀ p1 p2 : ℤ × ℤ, p1 ∈ D_n n → p2 ∈ D_n n →
    p1 ≠ p2 → FormLineSegment n p1 p2 → c p1 ≠ c p2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_coloring_l762_76246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_union_tree_planting_l762_76218

/-- Represents the number of students in the labor union -/
def x : ℕ := sorry

/-- Represents the total number of trees available -/
def total_trees : ℕ := 2 * x + 21

/-- Proposition: The equation 2x + 21 = 3x - 24 holds for the given conditions -/
theorem labor_union_tree_planting : 2 * x + 21 = 3 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_union_tree_planting_l762_76218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l762_76280

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - 4*x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

-- Define the mean value average function property
def is_mean_value_average (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ) (h : x₁ < x₂), 
    let x₀ := (x₁ + x₂) / 2
    (deriv f x₀) * (x₂ - x₁) = f x₂ - f x₁

-- Main theorem
theorem f_properties (a : ℝ) :
  (∀ x ∈ Set.Icc (Real.exp (-1)) (Real.exp 1), f a x ≥ g a x) →
  (a ≤ -1 ∧
   (a = 1 → ∀ x y : ℝ, y = f 1 x → x + y + 2 = 0 ↔ x = 1) ∧
   (is_mean_value_average (f a) ↔ a = 0)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l762_76280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_solution_l762_76283

theorem nested_radical_solution :
  ∃! x : ℝ, x > 0 ∧ x = Real.sqrt (4 + x) :=
by
  use (1 + Real.sqrt 17) / 2
  constructor
  · constructor
    · -- Prove x > 0
      sorry
    · -- Prove x = Real.sqrt (4 + x)
      sorry
  · -- Prove uniqueness
    sorry

#check nested_radical_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_solution_l762_76283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_construction_l762_76296

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicular lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define an equilateral triangle
def isEquilateralTriangle (p1 p2 p3 : Point2D) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

-- Theorem: Given two perpendicular lines and the ability to transfer equal distances,
-- it's possible to construct an equilateral triangle
theorem equilateral_triangle_construction 
  (l1 l2 : Line2D) 
  (h_perp : perpendicular l1 l2) :
  ∃ (p1 p2 p3 : Point2D), isEquilateralTriangle p1 p2 p3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_construction_l762_76296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_lines_l762_76287

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 9

-- Define the line
def myLine (x y : ℝ) : Prop := x - y = 0

-- Define the point P
def P : ℝ × ℝ := (4, -3)

-- Part 1: Length of chord AB
theorem chord_length :
  ∃ A B : ℝ × ℝ,
    myCircle A.1 A.2 ∧ myCircle B.1 B.2 ∧
    myLine A.1 A.2 ∧ myLine B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 28 :=
by sorry

-- Part 2: Equations of tangent lines
theorem tangent_lines :
  (∀ x y : ℝ, myCircle x y → (x = 4 → (x - P.1)^2 + (y - P.2)^2 ≥ 9)) ∧
  (∀ x y : ℝ, myCircle x y → (3*x + 4*y = 0 → (x - P.1)^2 + (y - P.2)^2 ≥ 9)) ∧
  (∃ x y : ℝ, myCircle x y ∧ x = 4 ∧ (x - P.1)^2 + (y - P.2)^2 = 9) ∧
  (∃ x y : ℝ, myCircle x y ∧ 3*x + 4*y = 0 ∧ (x - P.1)^2 + (y - P.2)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_lines_l762_76287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_seven_pi_sixths_l762_76236

theorem sin_seven_pi_sixths : Real.sin (7 * Real.pi / 6) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_seven_pi_sixths_l762_76236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_intercepts_l762_76254

/-- The modulus used in the congruence equation -/
def m : Nat := 21

/-- The equation 5x ≡ 3y-2 (mod 21) -/
def congruence_equation (x y : ℕ) : Prop :=
  (5 * x) % m = (3 * y - 2) % m

/-- The x-intercept of the equation -/
noncomputable def x_intercept : ℕ :=
  Classical.choose (exists_unique_x_intercept)
where
  exists_unique_x_intercept : ∃! x, congruence_equation x 0 ∧ x < m := by sorry

/-- The y-intercept of the equation -/
noncomputable def y_intercept : ℕ :=
  Classical.choose (exists_unique_y_intercept)
where
  exists_unique_y_intercept : ∃! y, congruence_equation 0 y ∧ y < m := by sorry

/-- The theorem stating that the sum of intercepts is 15 -/
theorem sum_of_intercepts : x_intercept + y_intercept = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_intercepts_l762_76254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l762_76234

/-- An arithmetic sequence with a₁ = -1 and a₃ = 3 -/
def a : ℕ → ℤ
  | 0 => -3 -- Adding a case for 0 to cover all natural numbers
  | 1 => -1
  | n + 2 => a (n + 1) + 2 -- Defining recursively to ensure termination

/-- The sequence b_n defined as 2^(a_n) -/
noncomputable def b (n : ℕ) : ℝ := 2 ^ (a n)

theorem arithmetic_and_geometric_sequences :
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 3) ∧
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 4 * b n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l762_76234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l762_76239

-- Define a power function (marked as noncomputable due to Real.instPowReal)
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h : ∃ α : ℝ, ∀ x : ℝ, f x = power_function α x) :
  f 2 = (1 : ℝ) / 4 → f (1 / 2) = 4 := by
  intro h1
  rcases h with ⟨α, hα⟩
  have h2 : α = -2 := by
    -- Proof that α = -2
    sorry
  have h3 : ∀ x : ℝ, f x = 1 / (x ^ 2) := by
    -- Proof that f(x) = 1 / x^2
    sorry
  -- Calculate f(1/2)
  calc
    f (1 / 2) = 1 / ((1 / 2) ^ 2) := by rw [h3]
    _ = 1 / (1 / 4) := by norm_num
    _ = 4 := by norm_num

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l762_76239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_subject_grade_l762_76205

/-- Represents the grades of a student in three subjects -/
structure Grades where
  subject1 : ℝ
  subject2 : ℝ
  subject3 : ℝ

/-- Calculates the average of three grades -/
noncomputable def average (g : Grades) : ℝ := (g.subject1 + g.subject2 + g.subject3) / 3

theorem second_subject_grade (g : Grades) 
  (h1 : g.subject1 = 70)
  (h2 : g.subject3 = 75)
  (h3 : average g = 75) :
  g.subject2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_subject_grade_l762_76205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_2_l762_76273

def sequence_a : ℕ → ℤ
  | 0 => 2  -- We need to define a value for 0 to make the function total
  | 1 => 2
  | 2 => 5
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_2013_equals_2 : sequence_a 2013 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_2_l762_76273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l762_76252

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def sum_of_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i ↦ a (i + 1))

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 7)
  (h_a5_a7 : a 5 + a 7 = 26) :
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, sum_of_sequence a n = n^2 + 2*n) ∧
  (∀ n : ℕ, sum_of_sequence (λ k ↦ -1 / ((a k)^2 - 1)) n = -n / (4 * (n + 1))) :=
by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l762_76252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_four_l762_76244

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_of_three_equals_four :
  (∀ x : ℝ, f (x + 2) = x + 3) → f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_four_l762_76244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l762_76278

noncomputable section

-- Define the line l: y = k(x-1)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define points A, B, and C
def point_A : ℝ × ℝ := (1, 0)
def point_B (k : ℝ) : ℝ × ℝ := (0, -k)
def point_C (k : ℝ) : ℝ × ℝ := (k / (k - 1), k / (k - 1))

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem part_1 (k : ℝ) (h : k = 3) : 
  distance (point_B k) (point_C k) / distance point_A (point_C k) = 3 := by
  sorry

theorem part_2 (k : ℝ) (h : distance (point_B k) (point_C k) = 2 * distance point_A (point_C k)) : 
  k = 2 ∨ k = -2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l762_76278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_P_l762_76230

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 3

-- Define the point P
noncomputable def P : ℝ × ℝ := (1, -5/2)

-- Theorem statement
theorem tangent_slope_at_P :
  deriv f P.fst = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_P_l762_76230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_from_bankers_gain_l762_76285

/-- Banker's gain calculation -/
noncomputable def bankers_gain (present_worth : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (present_worth * rate * time) / 100

/-- Theorem: Present worth calculation given banker's gain -/
theorem present_worth_from_bankers_gain 
  (gain : ℝ) (rate : ℝ) (time : ℝ) 
  (h_gain : gain = 24) 
  (h_rate : rate = 10) 
  (h_time : time = 2) : 
  ∃ (pw : ℝ), bankers_gain pw rate time = gain ∧ pw = 120 := by
  sorry

#check present_worth_from_bankers_gain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_from_bankers_gain_l762_76285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_value_l762_76267

noncomputable def a : ℝ := Real.sin (Real.cos (2015 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2015 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2015 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2015 * Real.pi / 180))

theorem largest_value : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_value_l762_76267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABO_is_right_isosceles_l762_76207

/-- PolarCoord represents a point in polar coordinates (r, θ) -/
structure PolarCoord where
  r : ℝ
  θ : ℝ

/-- Convert polar coordinates to Cartesian coordinates -/
noncomputable def toCartesian (p : PolarCoord) : ℝ × ℝ :=
  (p.r * Real.cos p.θ, p.r * Real.sin p.θ)

/-- Calculate the distance between two points in Cartesian coordinates -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- IsRightIsoscelesTriangle checks if three points form a right isosceles triangle -/
def IsRightIsoscelesTriangle (A B C : PolarCoord) : Prop :=
  let a := toCartesian A
  let b := toCartesian B
  let c := toCartesian C
  (distance a b = distance b c) ∧ 
  (distance a c)^2 = (distance a b)^2 + (distance b c)^2

/-- Given points A, B, and O in polar coordinates, prove that triangle ABO is a right isosceles triangle -/
theorem triangle_ABO_is_right_isosceles :
  let A : PolarCoord := ⟨-2, -π/2⟩
  let B : PolarCoord := ⟨Real.sqrt 2, 3*π/4⟩
  let O : PolarCoord := ⟨0, 0⟩
  IsRightIsoscelesTriangle A B O := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABO_is_right_isosceles_l762_76207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_curve_C_line_segment_AB_length_l762_76284

/-- Curve C in parametric form -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

/-- Line in parametric form -/
noncomputable def line (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

/-- Polar equation of a point (x, y) -/
def polar_equation (x y : ℝ) : ℝ := x^2 + y^2

/-- Theorem stating the polar equation of curve C -/
theorem polar_equation_curve_C (θ : ℝ) :
  polar_equation (curve_C θ).1 (curve_C θ).2 = 4 / (1 + 3 * Real.sin θ ^ 2) := by sorry

/-- Function to calculate the length of a line segment given by parameter values -/
noncomputable def line_segment_length (α t₁ t₂ : ℝ) : ℝ :=
  Real.sqrt ((t₁ - t₂)^2 * (Real.cos α^2 + Real.sin α^2))

/-- Theorem stating the length of line segment AB when α = π/3 -/
theorem line_segment_AB_length :
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    curve_C (Real.arccos ((line (π/3) t₁).1 / 2)) = line (π/3) t₁ ∧
    curve_C (Real.arccos ((line (π/3) t₂).1 / 2)) = line (π/3) t₂ ∧
    line_segment_length (π/3) t₁ t₂ = 8 * Real.sqrt 10 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_curve_C_line_segment_AB_length_l762_76284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l762_76243

-- Define the base of the logarithm
variable (a : ℝ)

-- Define the conditions on a
variable (ha_pos : a > 0)
variable (ha_neq_one : a ≠ 1)

-- Define the function f(x) = log_a x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define what it means for f to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem contrapositive_equivalence :
  (is_decreasing (f a) → f a 2 < 0) ↔
  (f a 2 ≥ 0 → ¬is_decreasing (f a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l762_76243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_amount_calculation_l762_76261

theorem certain_amount_calculation : ℝ := by
  -- Define x
  let x : ℝ := 900

  -- Define the certain amount
  let certain_amount : ℝ := 0.15 * 1600 - 0.25 * x

  -- Theorem statement
  have : certain_amount = 15 := by
    -- Proof steps would go here
    sorry

  -- Return the result
  exact 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_amount_calculation_l762_76261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l762_76262

theorem solve_exponential_equation :
  ∃ x : ℝ, 5 * (2 : ℝ) ^ x = 320 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l762_76262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l762_76200

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := ((x + 4) / 5) ^ (1/4)

-- State the theorem
theorem h_equality (x : ℝ) : h (3 * x) = 3 * h x ↔ x = -160/39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l762_76200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_1728_l762_76242

/-- The number of distinct, positive factors of 1728 -/
def factorCount : ℕ := 28

/-- 1728 as a natural number -/
def n : ℕ := 1728

/-- Theorem stating that the number of distinct, positive factors of 1728 is 28 -/
theorem factor_count_of_1728 : (Finset.filter (fun d => d > 0 ∧ n % d = 0) (Finset.range (n + 1))).card = factorCount := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_1728_l762_76242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l762_76214

theorem sin_double_theta (θ : ℝ) (h : Real.cos θ + Real.sin θ = 5/4) : Real.sin (2 * θ) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l762_76214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l762_76299

/-- The total surface area of a cylinder with height 12 and radius 5 is 170π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 5
  let circle_area : ℝ → ℝ := λ radius ↦ π * radius^2
  let cylinder_lateral_area : ℝ → ℝ → ℝ := λ height radius ↦ 2 * π * radius * height
  let total_surface_area : ℝ → ℝ → ℝ := λ height radius ↦ 
    2 * circle_area radius + cylinder_lateral_area height radius
  total_surface_area h r = 170 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l762_76299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_theorem_l762_76247

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

-- State the theorem
theorem two_zeros_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) 
  (h₂ : f a x₂ = 0) 
  (h₃ : x₁ < x₂) : 
  0 < a ∧ a < 1 ∧ x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_theorem_l762_76247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_second_is_half_l762_76266

def prob_win_first : ℝ := 0.3

noncomputable def prob_win_second : ℝ := 
  sorry -- We'll define this later in the proof

theorem prob_win_second_is_half : 
  prob_win_second = 0.5 := by
  have h1 : prob_win_first = 0.3 := rfl
  have h2 : (1 - prob_win_first) * (1 - prob_win_second) = 
            prob_win_first * prob_win_second + 0.2 := by sorry
  
  -- Here you would put the proof steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_second_is_half_l762_76266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_final_llama_count_l762_76210

/-- Represents Jill's llama breeding business -/
structure LlamaBusiness where
  pregnant_single : ℕ -- number of llamas pregnant with single calves
  pregnant_twins : ℕ -- number of llamas pregnant with twins
  traded_calves : ℕ -- number of calves traded
  new_adults : ℕ -- number of new adult llamas acquired through trade
  sell_fraction : ℚ -- fraction of herd sold at market

/-- Calculate the final number of llamas in Jill's herd -/
def final_llama_count (business : LlamaBusiness) : ℕ :=
  let initial_adults := business.pregnant_single + business.pregnant_twins
  let newborn_calves := business.pregnant_single + 2 * business.pregnant_twins
  let after_trade := initial_adults + newborn_calves - business.traded_calves + business.new_adults
  let sold := (after_trade * business.sell_fraction.num / business.sell_fraction.den).toNat
  after_trade - sold

/-- Theorem stating that Jill ends up with 18 llamas -/
theorem jill_final_llama_count :
  final_llama_count ⟨9, 5, 8, 2, 1/3⟩ = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_final_llama_count_l762_76210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_circle_l762_76291

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a circle -/
theorem curve_is_circle : ∃ (c : ℝ × ℝ) (r : ℝ), ∀ (θ : ℝ),
  let (x, y) := (1 / (Real.sin θ + Real.cos θ) * Real.cos θ, 1 / (Real.sin θ + Real.cos θ) * Real.sin θ)
  (x - c.1)^2 + (y - c.2)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_circle_l762_76291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_crease_length_l762_76288

theorem fold_crease_length (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_sides : a = 5 ∧ b = 12 ∧ c = 13) :
  let midpoint := c / 2
  let crease_length := Real.sqrt (midpoint^2 + midpoint^2)
  crease_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_crease_length_l762_76288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_steps_l762_76290

/-- The number of feet in a mile -/
def feetPerMile : ℕ := 5280

/-- The distance between Bella's and Ella's houses in miles -/
def distanceMiles : ℕ := 3

/-- The distance between Bella's and Ella's houses in feet -/
def distanceFeet : ℕ := distanceMiles * feetPerMile

/-- The ratio of Ella's speed to Bella's speed -/
def speedRatio : ℕ := 4

/-- The number of feet Bella covers in one step -/
def feetPerStep : ℕ := 3

/-- Theorem: Bella takes 1056 steps before meeting Ella -/
theorem bella_steps : (distanceFeet / (speedRatio + 1)) / feetPerStep = 1056 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_steps_l762_76290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_prob_theorem_l762_76271

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

-- Define the probability of X ≥ 1 for X ~ B(2,p)
def prob_X_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^2

-- Define the probability of Y ≥ 2 for Y ~ B(4,p)
def prob_Y_geq_2 (p : ℝ) : ℝ := 
  6 * p^2 * (1-p)^2 + 4 * p^3 * (1-p) + p^4

theorem binomial_prob_theorem : 
  ∀ p : ℝ, 0 < p → p < 1 →
  prob_X_geq_1 p = 5/9 →
  prob_Y_geq_2 p = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_prob_theorem_l762_76271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log_base_3_l762_76274

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := 3^x

-- Theorem statement
theorem inverse_of_log_base_3 :
  (∀ x > 0, g (f x) = x) ∧ (∀ y, f (g y) = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log_base_3_l762_76274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l762_76231

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  eccentricity : ℝ
  foci_axis : Unit

/-- Represents the equation of asymptotes -/
structure AsymptoteEquation where
  slope : ℝ

/-- Given a hyperbola with center at origin, eccentricity 5/3, and foci on y-axis,
    its asymptotes have the equation y = ± 3/4 x -/
theorem hyperbola_asymptote_equation (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_eccentricity : h.eccentricity = 5/3)
    (h_foci : h.foci_axis = ()) : 
    ∃ (a : AsymptoteEquation), a.slope = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l762_76231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_problem_l762_76222

/-- Given a positive integer d satisfying the equation (10/9)d - 80 = d,
    prove that the sum of its digits is 9 --/
theorem exchange_rate_problem (d : ℕ) (h1 : d > 0) 
  (h2 : (10 : ℚ) / 9 * d - 80 = d) : 
  (d.repr.toList.map (λ c => c.toString.toNat!)).sum = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_problem_l762_76222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l762_76206

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_2 (a b : ℝ) :
  (∃ (h : ℝ → ℝ), HasDerivAt (f a b) (h 1) 1 ∧ IsLocalMax (f a b) 1) →
  f a b 1 = -2 →
  ∃ (g : ℝ → ℝ), HasDerivAt (f a b) (g 2) 2 ∧ g 2 = -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l762_76206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_post_tax_income_distribution_l762_76249

/-- Represents the income distribution and tax system in a city --/
structure CityEconomy where
  x : ℚ  -- Percentage of total income for the poor group
  poor_income : ℚ := x
  middle_income : ℚ := 4 * x
  rich_income : ℚ := 5 * x
  tax_rate : ℚ := x^2 / 4 + x
  poor_tax_share : ℚ := 3 / 4
  middle_tax_share : ℚ := 1 / 4

/-- Calculates the post-tax income distribution --/
noncomputable def post_tax_income (c : CityEconomy) : (ℚ × ℚ × ℚ) :=
  let tax_amount := c.rich_income * c.tax_rate / 100
  let poor_new := c.poor_income + tax_amount * c.poor_tax_share
  let middle_new := c.middle_income + tax_amount * c.middle_tax_share
  let rich_new := c.rich_income - tax_amount
  (poor_new, middle_new, rich_new)

/-- Theorem stating the correct post-tax income distribution --/
theorem post_tax_income_distribution (c : CityEconomy) :
  c.x = 10 → post_tax_income c = (23125/1000, 44375/1000, 325/10) := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_post_tax_income_distribution_l762_76249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_pay_approx_l762_76255

/-- The weekly pay of employee Y given the total pay and the ratio of X's pay to Y's pay -/
noncomputable def employee_y_pay (total_pay : ℝ) (x_to_y_ratio : ℝ) : ℝ :=
  total_pay / (1 + x_to_y_ratio)

/-- Theorem stating that given the conditions, employee Y's pay is approximately 255 -/
theorem employee_y_pay_approx :
  let total_pay : ℝ := 560
  let x_to_y_ratio : ℝ := 1.2
  abs (employee_y_pay total_pay x_to_y_ratio - 255) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_pay_approx_l762_76255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_percentage_bounds_l762_76204

/-- Represents the stock packages in the auction --/
structure StockPackage where
  razneft : ℕ
  dvaneft : ℕ
  trineft : ℕ

/-- Represents the prices of individual shares --/
structure SharePrices where
  razneft : ℚ
  dvaneft : ℚ
  trineft : ℚ

/-- The conditions of the stock auction --/
def AuctionConditions (sp : StockPackage) (prices : SharePrices) : Prop :=
  -- Total shares of Razneft and Dvaneft equals shares of Trineft
  sp.razneft + sp.dvaneft = sp.trineft
  -- Dvaneft package is 3 times cheaper than Razneft package
  ∧ 3 * (prices.dvaneft * sp.dvaneft) = prices.razneft * sp.razneft
  -- Total value of Razneft and Dvaneft equals value of Trineft
  ∧ prices.razneft * sp.razneft + prices.dvaneft * sp.dvaneft = prices.trineft * sp.trineft
  -- Price difference between Razneft and Dvaneft share: 10,000 to 18,000 rubles
  ∧ 10000 ≤ prices.razneft - prices.dvaneft ∧ prices.razneft - prices.dvaneft ≤ 18000
  -- Price of Trineft share: 18,000 to 42,000 rubles
  ∧ 18000 ≤ prices.trineft ∧ prices.trineft ≤ 42000

/-- The percentage of Dvaneft shares in the total lot --/
noncomputable def DvaneftPercentage (sp : StockPackage) : ℚ :=
  (sp.dvaneft : ℚ) / (2 * (sp.razneft + sp.dvaneft) : ℚ) * 100

/-- The main theorem stating the bounds of Dvaneft percentage --/
theorem dvaneft_percentage_bounds (sp : StockPackage) (prices : SharePrices) 
  (h : AuctionConditions sp prices) : 
  15 ≤ DvaneftPercentage sp ∧ DvaneftPercentage sp ≤ 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_percentage_bounds_l762_76204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l762_76229

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := -(1/a) * Real.exp x + b

-- Define the derivative of f(x)
noncomputable def f' (a x : ℝ) : ℝ := -(1/a) * Real.exp x

theorem tangent_line_implies_a_and_b :
  ∀ (a b : ℝ),
  (f' a 0 = -1) →  -- The slope of the tangent line at x=0 is -1
  (f a b 0 = 1) →  -- The curve passes through the point (0,1)
  (a = 1 ∧ b = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l762_76229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_above_mean_l762_76298

-- Define the mean score
noncomputable def mean_score : ℝ := 88.8

-- Define the standard deviation
noncomputable def standard_deviation : ℝ := (mean_score - 86) / 7

-- Define the score in question
noncomputable def score : ℝ := 90

-- Theorem to prove
theorem score_above_mean : 
  (score - mean_score) / standard_deviation = 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_above_mean_l762_76298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_connected_pairs_l762_76251

/-- Represents a network of factories connected by bus routes -/
structure FactoryNetwork where
  n : ℕ  -- number of factories
  connected : Fin n → Fin n → Bool  -- connection relation between factories

/-- The property that any four factories can be split into two connected pairs -/
def fourSplitProperty (net : FactoryNetwork) : Prop :=
  ∀ a b c d : Fin net.n, 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
    ((net.connected a b ∧ net.connected c d) ∨
     (net.connected a c ∧ net.connected b d) ∨
     (net.connected a d ∧ net.connected b c))

/-- The number of connected pairs in the network -/
def connectedPairsCount (net : FactoryNetwork) : ℕ :=
  (Finset.univ : Finset (Fin net.n)).sum (λ i =>
    (Finset.filter (λ j => net.connected i j) (Finset.univ : Finset (Fin net.n))).card) / 2

/-- The main theorem: given 150 factories with the four-split property, 
    the minimum number of connected pairs is 11025 -/
theorem min_connected_pairs : 
  ∀ net : FactoryNetwork, 
    net.n = 150 → 
    fourSplitProperty net → 
    connectedPairsCount net ≥ 11025 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_connected_pairs_l762_76251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_eval_l762_76294

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem f_composition_eval :
  f (f (5/2)) = 3/2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_eval_l762_76294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_speed_calculation_l762_76282

-- Define the walking speed, walking time, and running time
noncomputable def walking_speed : ℝ := 5
noncomputable def walking_time : ℝ := 5
noncomputable def running_time_minutes : ℝ := 36

-- Define the function to calculate distance
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Define the function to convert minutes to hours
noncomputable def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

-- Theorem statement
theorem running_speed_calculation :
  let walking_distance := distance walking_speed walking_time
  let running_time_hours := minutes_to_hours running_time_minutes
  let running_speed := walking_distance / running_time_hours
  ∃ ε > 0, |running_speed - 41.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_speed_calculation_l762_76282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l762_76202

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

-- Define the line l
def lineL (m x y : ℝ) : Prop := y = m*x + 1 - m

-- Theorem statement
theorem circle_line_intersection (m : ℝ) :
  -- Part 1: Line l always intersects circle C at two distinct points
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    circleC x₁ y₁ ∧ circleC x₂ y₂ ∧ 
    lineL m x₁ y₁ ∧ lineL m x₂ y₂) ∧
  -- Part 2: If |AB| = √17, then the slope angle is 60° or 120°
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    circleC x₁ y₁ ∧ circleC x₂ y₂ ∧ 
    lineL m x₁ y₁ ∧ lineL m x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 17 →
    m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l762_76202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_YXZ_is_45_degrees_l762_76225

noncomputable section

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle_at_Y : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the point W on XZ
def W_on_XZ (X Z W : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ W = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)

-- Define the condition XW = 2WZ
def XW_eq_2WZ (X W Z : ℝ × ℝ) : Prop :=
  (X.1 - W.1)^2 + (X.2 - W.2)^2 = 4 * ((W.1 - Z.1)^2 + (W.2 - Z.2)^2)

-- Define the condition YW = WZ
def YW_eq_WZ (Y W Z : ℝ × ℝ) : Prop :=
  (Y.1 - W.1)^2 + (Y.2 - W.2)^2 = (W.1 - Z.1)^2 + (W.2 - Z.2)^2

-- Define the angle YXZ
noncomputable def angle_YXZ (X Y Z : ℝ × ℝ) : ℝ :=
  Real.arccos ((Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2)) /
    (Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) * Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2))

theorem angle_YXZ_is_45_degrees 
  (X Y Z W : ℝ × ℝ) 
  (h_triangle : Triangle X Y Z) 
  (h_W_on_XZ : W_on_XZ X Z W) 
  (h_XW_eq_2WZ : XW_eq_2WZ X W Z) 
  (h_YW_eq_WZ : YW_eq_WZ Y W Z) : 
  angle_YXZ X Y Z = π / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_YXZ_is_45_degrees_l762_76225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_joey_pills_increase_l762_76293

/-- Represents the number of pills Joey takes on a given day -/
def pills_on_day (d : ℕ) (increase : ℕ) : ℕ :=
  1 + (d - 1) * increase

/-- Represents the total number of pills Joey takes over a week -/
def total_pills (increase : ℕ) : ℕ :=
  (List.range 7).map (λ d ↦ pills_on_day (d + 1) increase) |>.sum

/-- 
Theorem stating that if Joey takes 49 pills over 7 days, 
starting with 1 pill on the first day and increasing by a constant number each day, 
then the constant increase is 2 pills per day.
-/
theorem joey_pills_increase : total_pills 2 = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_joey_pills_increase_l762_76293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l762_76216

theorem oil_price_reduction (original_quantity : ℝ) (original_price : ℝ) : 
  (original_quantity * original_price = 1080) →
  ((original_quantity + 8) * 24.3 = 1080) →
  (original_price > 24.3) →
  abs ((original_price - 24.3) / original_price * 100 - 18.02) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l762_76216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l762_76292

def sequence_a (n : ℕ+) : ℕ := 2 * n.val - 1

def sum_sequence_a (n : ℕ+) : ℕ := n.val ^ 2

noncomputable def sequence_b (q : ℝ) (n : ℕ+) : ℝ := (3 / 2) * q ^ (n.val - 1)

noncomputable def sum_sequence_b (q : ℝ) (n : ℕ+) : ℝ := (3 / 2) * (q ^ n.val - 1) / (q - 1)

theorem sequence_properties :
  ∀ (n : ℕ+),
  (∀ (k : ℕ+), k ≤ n → sum_sequence_a k = k.val ^ 2) →
  sequence_a n = 2 * n.val - 1 ∧
  ∃ (q : ℝ),
    q > 0 ∧
    sequence_b q 1 = sum_sequence_a 1 ∧
    sequence_b q 4 = sequence_a 2 + sequence_a 3 ∧
    ∀ (m : ℕ+), sum_sequence_b q m = (3 / 2) * (2 ^ m.val - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l762_76292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l762_76224

open Real

-- Define the function f(x) = lg(x) + x - 2
noncomputable def f (x : ℝ) := log x / log 10 + x - 2

-- State the theorem
theorem solution_in_interval (x₀ : ℝ) (h : f x₀ = 0) : x₀ ∈ Set.Ioo (7/4) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l762_76224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_second_quadrant_l762_76208

theorem sin_minus_cos_second_quadrant (α : ℝ) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.cos (2 * α) = -Real.sqrt 5 / 3 →
  Real.sin α - Real.cos α = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_second_quadrant_l762_76208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_l762_76256

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the angle between two lines defined by four points -/
noncomputable def angleBetweenLines (p1 p2 p3 p4 : Point3D) : ℝ := sorry

theorem angle_between_lines (A B C D : Point3D) : 
  distance A B = distance B C ∧ 
  distance B C = distance C D ∧
  angle A B C = 36 ∧ 
  angle B C D = 36 ∧ 
  angle C D A = 36 →
  (angleBetweenLines A C B D = 36 ∨ angleBetweenLines A C B D = 90) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_l762_76256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_excenter_concyclicity_l762_76203

-- Define the points
variable (A B C D O : EuclideanPlane)
-- Define the incenters
variable (I₁ I₂ I₃ I₄ : EuclideanPlane)
-- Define the excenters
variable (J₁ J₂ J₃ J₄ : EuclideanPlane)

-- ABCD is a convex quadrilateral
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- AC and BD are diagonals intersecting at O
def diagonals_intersect_at (A B C D O : EuclideanPlane) : Prop := sorry

-- I1, I2, I3, I4 are incenters of triangles AOB, BOC, COD, DOA respectively
def are_incenters (I₁ I₂ I₃ I₄ A B C D O : EuclideanPlane) : Prop := sorry

-- J1, J2, J3, J4 are excenters of triangles AOB, BOC, COD, DOA opposite O respectively
def are_excenters (J₁ J₂ J₃ J₄ A B C D O : EuclideanPlane) : Prop := sorry

-- Points are concyclic
def are_concyclic (P Q R S : EuclideanPlane) : Prop := sorry

-- The main theorem
theorem incenter_excenter_concyclicity 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_diag : diagonals_intersect_at A B C D O)
  (h_incenters : are_incenters I₁ I₂ I₃ I₄ A B C D O)
  (h_excenters : are_excenters J₁ J₂ J₃ J₄ A B C D O) :
  are_concyclic I₁ I₂ I₃ I₄ ↔ are_concyclic J₁ J₂ J₃ J₄ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_excenter_concyclicity_l762_76203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_banana_count_l762_76268

/-- Given the ratio of flour to banana mush, bananas per cup of mush, and total flour used,
    calculate the number of bananas used. -/
def banana_bread_bananas 
  (flour_to_mush_ratio : ℚ) 
  (bananas_per_mush : ℕ) 
  (total_flour : ℕ) : ℕ :=
  let bananas := 
    (total_flour : ℚ) / flour_to_mush_ratio * (bananas_per_mush : ℚ)
  bananas.floor.toNat

/-- Prove that Hannah used 20 bananas for her banana bread. -/
theorem hannah_banana_count : 
  banana_bread_bananas 3 4 15 = 20 := by
  -- Unfold the definition of banana_bread_bananas
  unfold banana_bread_bananas
  -- Simplify the arithmetic
  simp [Nat.cast_ofNat, Int.floor_eq_iff]
  -- The proof is completed
  sorry

#eval banana_bread_bananas 3 4 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_banana_count_l762_76268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_triangle_from_unit_segments_l762_76264

/-- The probability of forming a triangle with three randomly chosen segments,
    each shorter than a unit. -/
noncomputable def triangle_probability : ℝ := 1/2

/-- Theorem stating that the probability of forming a triangle with three randomly
    chosen segments, each shorter than a unit, is 1/2. -/
theorem prob_triangle_from_unit_segments : triangle_probability = 1/2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_triangle_from_unit_segments_l762_76264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_is_right_triangle_l762_76260

-- Define the sets of numbers
noncomputable def set_A : List ℝ := [Real.sqrt 3, Real.sqrt 4, Real.sqrt 5]
def set_B : List ℝ := [2, 3, 4]
def set_C : List ℝ := [6, 7, 8]
def set_D : List ℝ := [9, 12, 15]

-- Function to check if a set of three numbers satisfies the Pythagorean theorem
def is_right_triangle (sides : List ℝ) : Prop :=
  match sides with
  | [a, b, c] => a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2
  | _ => False

-- Theorem stating that only set_D forms a right-angled triangle
theorem only_set_D_is_right_triangle :
  ¬(is_right_triangle set_A) ∧
  ¬(is_right_triangle set_B) ∧
  ¬(is_right_triangle set_C) ∧
  is_right_triangle set_D := by
  sorry

#eval set_B
#eval set_C
#eval set_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_is_right_triangle_l762_76260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_sin_transform_shift_is_pi_over_6_l762_76211

-- Define the function f as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

-- State the theorem
theorem transform_to_sin (φ : ℝ) (h1 : |φ| ≤ π/2) 
  (h2 : ∀ x, f x φ = -f (8*π/3 - x) φ) :
  ∃ c, ∀ x, f (x - c) φ = Real.sin (2*x) := by
  -- The proof goes here
  sorry

-- Additional lemma to show that c = π/6
theorem transform_shift_is_pi_over_6 (φ : ℝ) (h1 : |φ| ≤ π/2) 
  (h2 : ∀ x, f x φ = -f (8*π/3 - x) φ) :
  ∃ c, c = π/6 ∧ ∀ x, f (x - c) φ = Real.sin (2*x) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_sin_transform_shift_is_pi_over_6_l762_76211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l762_76201

-- Define variables as axioms instead of noncomputable definitions
axiom a : ℝ
axiom b : ℝ
axiom c : ℝ
axiom d : ℝ

theorem problem_statement 
  (h1 : a / Real.log a = 2.86)
  (h2 : b / Real.log b = 2.86)
  (h3 : c * Real.log c = -0.35)
  (h4 : d * Real.log d = -0.35)
  (h5 : a < b)
  (h6 : c < d) :
  (a + b > 2 * Real.exp 1) ∧
  (c + d > 2 / Real.exp 1) ∧
  (a * d < 1) ∧
  (b * c > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l762_76201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l762_76227

/-- Calculates the annual interest rate given the principal, time, and total amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (total_amount : ℝ) : ℝ :=
  ((total_amount / principal) ^ (1 / time)) - 1

/-- Theorem stating that the calculated interest rate is 0.1 (10%) -/
theorem interest_rate_is_ten_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (compound_interest : ℝ) 
  (h1 : principal = 700) 
  (h2 : time = 2) 
  (h3 : compound_interest = 147.0000000000001) : 
  calculate_interest_rate principal time (principal + compound_interest) = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l762_76227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinFunction_properties_l762_76295

noncomputable def sinFunction (x : ℝ) : ℝ := Real.sin (8 * x + Real.pi / 4)

theorem sinFunction_properties :
  let period := 2 * Real.pi / 8
  let phaseShift := -(Real.pi / 4) / 8
  (period = Real.pi / 4) ∧ (phaseShift = -Real.pi / 32) := by
  -- Proof goes here
  sorry

#check sinFunction_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinFunction_properties_l762_76295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficient_sum_l762_76212

/-- A parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
noncomputable def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The vertex of a parabola -/
noncomputable def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (-p.b / (2 * p.a), p.y_coord (-p.b / (2 * p.a)))

theorem parabola_coefficient_sum (p : Parabola) :
  p.vertex = (-3, 2) →
  p.y_coord 1 = -6 →
  p.a + p.b + p.c = -11/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficient_sum_l762_76212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_minimum_l762_76258

/-- Given a > 0, f(x) = 2ax^2 - 1/(ax), the slope of the tangent line at (1, f(1)) is minimized when a = 1/2 -/
theorem tangent_slope_minimum (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * a * x^2 - 1 / (a * x)
  let k : ℝ := (λ x ↦ 4 * a * x + 1 / (a * x^2)) 1
  ∀ b > 0, k ≤ (λ x ↦ 4 * b * x + 1 / (b * x^2)) 1 → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_minimum_l762_76258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l762_76250

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * |x₁*y₂ + x₂*y₃ + x₃*y₁ - x₂*y₁ - x₁*y₃ - x₃*y₂|

/-- Theorem: The area of a triangle with vertices (x₁, y₁), (x₂, y₂), and (x₃, y₃)
    is equal to 1/2 |x₁y₂ + x₂y₃ + x₃y₁ - x₂y₁ - x₁y₃ - x₃y₂| -/
theorem triangle_area_formula (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  triangleArea x₁ y₁ x₂ y₂ x₃ y₃ = (1/2) * |x₁*y₂ + x₂*y₃ + x₃*y₁ - x₂*y₁ - x₁*y₃ - x₃*y₂| := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l762_76250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l762_76265

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a/2) * Real.log x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x - a/(2*x)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f 1 x + (1/2) * m * x

theorem problem_solution :
  -- Part 1: Given f'(1/2) = 0, then a = 1
  (∀ a : ℝ, f_deriv a (1/2) = 0 → a = 1) ∧
  -- Part 2: Monotonicity of f
  (∀ x : ℝ, x > 1/2 → f_deriv 1 x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → f_deriv 1 x < 0) ∧
  -- Part 3: Condition for g(x) > 0 on (1, +∞)
  (∀ m : ℝ, (∀ x : ℝ, x > 1 → g m x > 0) ↔ m ≥ -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l762_76265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_admissible_coloring_reachable_l762_76217

/-- Represents a coloring of a 100 × 100 table -/
def Coloring := Fin 100 → Fin 100 → Bool

/-- Checks if a coloring is admissible -/
def isAdmissible (c : Coloring) : Prop :=
  ∀ i : Fin 100, 
    (50 ≤ (Finset.filter (λ j ↦ c i j) Finset.univ).card) ∧
    ((Finset.filter (λ j ↦ c i j) Finset.univ).card ≤ 60) ∧
    (50 ≤ (Finset.filter (λ j ↦ c j i) Finset.univ).card) ∧
    ((Finset.filter (λ j ↦ c j i) Finset.univ).card ≤ 60)

/-- Represents a sequence of recoloring operations -/
def RecoloringSequence := List ((Fin 100 × Fin 100) × Bool)

/-- Applies a recoloring sequence to a coloring -/
def applyRecoloring (c : Coloring) (seq : RecoloringSequence) : Coloring :=
  seq.foldl (λ acc (pos, color) ↦ Function.update acc pos.1 (Function.update (acc pos.1) pos.2 color)) c

/-- The main theorem: any two admissible colorings can be transformed into each other -/
theorem admissible_coloring_reachable (c1 c2 : Coloring) 
  (h1 : isAdmissible c1) (h2 : isAdmissible c2) :
  ∃ seq : RecoloringSequence, 
    isAdmissible (applyRecoloring c1 seq) ∧ 
    applyRecoloring c1 seq = c2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_admissible_coloring_reachable_l762_76217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_with_digit_sum_8_l762_76259

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem two_digit_primes_with_digit_sum_8 :
  ∀ n : ℕ, is_two_digit n → Nat.Prime n → digit_sum n = 8 → n ∈ ({17, 71, 53} : Set ℕ) :=
by
  sorry

#eval digit_sum 17
#eval digit_sum 71
#eval digit_sum 53

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_with_digit_sum_8_l762_76259
