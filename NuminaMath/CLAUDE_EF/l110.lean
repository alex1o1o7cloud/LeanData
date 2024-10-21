import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l110_11032

theorem square_root_problem (x y : ℝ) :
  (Real.sqrt (2*x + y - 2) * (x - y + 3)^2 ≤ 0) ∧ 
  (Real.sqrt (2*x + y - 2) ≠ 0 ∨ (x - y + 3)^2 ≠ 0) →
  Real.sqrt (x^2 + y) = 5/3 ∨ Real.sqrt (x^2 + y) = -5/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l110_11032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statistical_study_properties_l110_11061

-- Define the type for a student
structure Student where
  id : Nat

-- Define the population
def population : Finset Student := sorry

-- Define the sample
def sample : Finset Student := sorry

-- State the theorem
theorem statistical_study_properties :
  -- The population size is 1000
  Finset.card population = 1000 ∧
  -- The sample size is 100
  Finset.card sample = 100 ∧
  -- The sample is a subset of the population
  sample ⊆ population ∧
  -- Each element in the population is an individual student
  ∀ s ∈ population, s.id ≤ 1000 ∧
  -- The sample was randomly selected from the population
  (Finset.card sample : Real) / (Finset.card population : Real) = 0.1 →
  -- Prove the following:
  -- 1. The 1000 students constitute the population
  Finset.card population = 1000 ∧
  -- 2. Each student is considered an individual
  ∀ s ∈ population, s ∈ {s : Student | s.id ≤ 1000} ∧
  -- 3. The sample size is 100
  Finset.card sample = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statistical_study_properties_l110_11061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sine_fourier_transform_of_F_l110_11020

-- Define the Fourier sine transform F(p)
noncomputable def F (p : ℝ) : ℝ := 
  if 0 < p ∧ p < 1 then 1 else 
  if 1 < p then 0 else 0

-- Define the inverse sine Fourier transform
noncomputable def inverseSineFourierTransform (F : ℝ → ℝ) (x : ℝ) : ℝ :=
  Real.sqrt (2 / Real.pi) * ∫ (p : ℝ) in Set.Ioi 0, F p * Real.sin (p * x)

-- State the theorem
theorem inverse_sine_fourier_transform_of_F (x : ℝ) (hx : x ≠ 0) :
  inverseSineFourierTransform F x = Real.sqrt (2 / Real.pi) * (1 - Real.cos x) / x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sine_fourier_transform_of_F_l110_11020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_boy_girl_different_hair_not_always_all_different_not_always_three_with_at_most_one_common_l110_11019

-- Define the types for attributes
inductive Gender : Type
| Boy : Gender
| Girl : Gender

inductive HairColor : Type
| Blonde : HairColor
| Brown : HairColor

inductive Height : Type
| Short : Height
| Tall : Height

-- Define a student as a tuple of attributes
def Student := Gender × HairColor × Height

-- Define the class properties
structure ClassProperties where
  students : List Student
  has_boys : ∃ s ∈ students, s.1 = Gender.Boy
  has_girls : ∃ s ∈ students, s.1 = Gender.Girl
  has_blonde : ∃ s ∈ students, s.2.1 = HairColor.Blonde
  has_brown : ∃ s ∈ students, s.2.1 = HairColor.Brown
  has_short : ∃ s ∈ students, s.2.2 = Height.Short
  has_tall : ∃ s ∈ students, s.2.2 = Height.Tall

-- Theorem 1: There exists a boy-girl pair with different hair colors
theorem exists_boy_girl_different_hair (c : ClassProperties) :
  ∃ s1 s2, s1 ∈ c.students ∧ s2 ∈ c.students ∧ s1.1 ≠ s2.1 ∧ s1.2.1 ≠ s2.2.1 :=
sorry

-- Theorem 2: It is not always true that there are two students with all attributes different
theorem not_always_all_different :
  ¬(∀ c : ClassProperties, ∃ s1 s2, s1 ∈ c.students ∧ s2 ∈ c.students ∧ s1.1 ≠ s2.1 ∧ s1.2.1 ≠ s2.2.1 ∧ s1.2.2 ≠ s2.2.2) :=
sorry

-- Theorem 3: It is not always true that there are three students with at most one common attribute
theorem not_always_three_with_at_most_one_common :
  ¬(∀ c : ClassProperties, ∃ s1 s2 s3, s1 ∈ c.students ∧ s2 ∈ c.students ∧ s3 ∈ c.students ∧
    (s1.1 = s2.1 → s1.2.1 ≠ s2.2.1 ∧ s1.2.2 ≠ s2.2.2) ∧
    (s1.1 = s3.1 → s1.2.1 ≠ s3.2.1 ∧ s1.2.2 ≠ s3.2.2) ∧
    (s2.1 = s3.1 → s2.2.1 ≠ s3.2.1 ∧ s2.2.2 ≠ s3.2.2) ∧
    (s1.2.1 = s2.2.1 → s1.1 ≠ s2.1 ∧ s1.2.2 ≠ s2.2.2) ∧
    (s1.2.1 = s3.2.1 → s1.1 ≠ s3.1 ∧ s1.2.2 ≠ s3.2.2) ∧
    (s2.2.1 = s3.2.1 → s2.1 ≠ s3.1 ∧ s2.2.2 ≠ s3.2.2) ∧
    (s1.2.2 = s2.2.2 → s1.1 ≠ s2.1 ∧ s1.2.1 ≠ s2.2.1) ∧
    (s1.2.2 = s3.2.2 → s1.1 ≠ s3.1 ∧ s1.2.1 ≠ s3.2.1) ∧
    (s2.2.2 = s3.2.2 → s2.1 ≠ s3.1 ∧ s2.2.1 ≠ s3.2.1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_boy_girl_different_hair_not_always_all_different_not_always_three_with_at_most_one_common_l110_11019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l110_11012

theorem expression_simplification (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
  ((4 - 2*x + x^2) / (4 - 2*x) + (6*x^2 + 8 + 12*x) / (4 - x^2) - (x^2 + 2*x + 4) / (2*x + 4))^(-(1/3 : ℝ)) * (x + 2) = 
  (4 - x^2)^(1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l110_11012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l110_11074

/-- Represents the average test score of a group of students -/
structure GroupScore where
  portion : Rat  -- Portion of the class (between 0 and 1)
  average : Rat  -- Average score (between 0 and 1)

/-- Calculates the overall class average given group scores -/
def classAverage (scores : List GroupScore) : Rat :=
  (scores.map (fun s => s.portion * s.average)).sum / (scores.map (fun s => s.portion)).sum

/-- Theorem stating the overall class average given specific group scores -/
theorem class_average_theorem (scores : List GroupScore) 
  (h1 : scores = [⟨1/5, 4/5⟩, ⟨1/2, 3/5⟩, ⟨3/10, 2/5⟩]) : 
  classAverage scores = 29/50 := by
  sorry

#eval classAverage [⟨1/5, 4/5⟩, ⟨1/2, 3/5⟩, ⟨3/10, 2/5⟩]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l110_11074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_sum_21_l110_11042

theorem pigeonhole_sum_21 (S : Finset ℕ) : 
  S ⊆ Finset.range 20 → S.card = 11 → ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x + y = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_sum_21_l110_11042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximable_implies_polynomial_l110_11082

/-- A function that can be approximated by polynomials with an error bound of c * x^1998 -/
def ApproximableByPolynomials (f : ℝ → ℝ) : Prop :=
  ∀ c > 0, ∃ P : Polynomial ℝ, ∀ x, |f x - P.eval x| ≤ c * x^1998

/-- Main theorem: If a function can be approximated by polynomials with an error bound of c * x^1998,
    then it is a polynomial -/
theorem approximable_implies_polynomial (f : ℝ → ℝ) (h : ApproximableByPolynomials f) :
  ∃ P : Polynomial ℝ, f = P.eval :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximable_implies_polynomial_l110_11082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bear_weight_gain_l110_11048

-- Define the total weight the bear needs to gain
variable (W : ℝ)

-- Define the weight gained from each source
noncomputable def weight_from_berries (W : ℝ) : ℝ := W / 5
noncomputable def weight_from_acorns (W : ℝ) : ℝ := 2 * W / 5
noncomputable def weight_from_salmon (W : ℝ) : ℝ := W / 5
def weight_from_small_animals : ℝ := 200

-- Theorem statement
theorem bear_weight_gain (W : ℝ) :
  weight_from_berries W + weight_from_acorns W + weight_from_salmon W + weight_from_small_animals = W →
  W = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bear_weight_gain_l110_11048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_conduction_solution_l110_11098

/-- Heat conduction equation solution -/
theorem heat_conduction_solution 
  (u : ℝ → ℝ → ℝ) 
  (h1 : ∀ (x t : ℝ), t > 0 → 
    (deriv (fun τ => u x τ)) t = (deriv (fun ξ => deriv (fun ζ => u ζ t) ξ)) x)
  (h2 : ∀ x : ℝ, u x 0 = Real.exp (-x)) :
  ∀ (x t : ℝ), t > 0 → u x t = Real.exp (t - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_conduction_solution_l110_11098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l110_11055

/-- Given a function f(x) = sin(ωx) where ω > 0, prove that the minimum value of ω is 3/4
    when the graph of f(x) shifted left by π/2 is symmetrical about x = π/6 -/
theorem min_omega_value (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, Real.sin (ω * (x + Real.pi/2)) = Real.sin (ω * (Real.pi/3 - x + Real.pi/2))) → ω ≥ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l110_11055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_width_theorem_l110_11022

/-- The width of a circular road around a circular garden -/
noncomputable def road_width (inner_radius outer_radius : ℝ) : ℝ :=
  outer_radius - inner_radius

/-- The sum of circumferences of two concentric circles -/
noncomputable def sum_of_circumferences (inner_radius outer_radius : ℝ) : ℝ :=
  2 * Real.pi * inner_radius + 2 * Real.pi * outer_radius

theorem road_width_theorem (inner_radius outer_radius : ℝ) 
  (h1 : sum_of_circumferences inner_radius outer_radius = 88)
  (h2 : inner_radius = (1/3) * outer_radius) :
  road_width inner_radius outer_radius = 22 / Real.pi := by
  sorry

#check road_width_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_width_theorem_l110_11022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_value_l110_11011

theorem complex_fraction_value : 
  (1 + Complex.I)^2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_value_l110_11011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l110_11036

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3) + 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≤ 3) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Ioo (-Real.pi/12 + k*Real.pi) (5*Real.pi/12 + k*Real.pi))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l110_11036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_equation_l110_11096

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 16

-- Define the intersecting line
def intersecting_line_eq (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the diameter line
def diameter_line_eq (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define midpoint of a chord
def is_midpoint_of_chord (m : Point) (circle_eq : ℝ → ℝ → Prop) (line_eq : ℝ → ℝ → Prop) : Prop :=
  circle_eq m.x m.y ∧ line_eq m.x m.y

-- Define diameter passing through a point
def diameter_passes_through (center : Point) (p : Point) : Prop :=
  (p.x - center.x) * (p.y - center.y) = 8 -- This is a simplification, adjust as needed

-- Theorem statement
theorem diameter_equation (x y : ℝ) :
  circle_eq x y ∧ 
  intersecting_line_eq x y ∧ 
  (∃ (m : Point), is_midpoint_of_chord m circle_eq intersecting_line_eq ∧ 
              diameter_passes_through ⟨2, -1⟩ m) →
  diameter_line_eq x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_equation_l110_11096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_no_common_point_existence_of_non_skew_no_common_point_l110_11044

/-- Represents a line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne_points : point1 ≠ point2

/-- The set of points on a line -/
def set_of_points (l : Line3D) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines do not have a common point -/
def no_common_point (l1 l2 : Line3D) : Prop :=
  ∀ (p : ℝ × ℝ × ℝ), p ∈ set_of_points l1 → p ∉ set_of_points l2

/-- Two lines are skew -/
def skew_lines (l1 l2 : Line3D) : Prop :=
  no_common_point l1 l2 ∧ ¬ parallel l1 l2

theorem skew_lines_no_common_point :
  ∀ (l1 l2 : Line3D), skew_lines l1 l2 → no_common_point l1 l2 :=
by
  sorry

theorem existence_of_non_skew_no_common_point :
  ∃ (l1 l2 : Line3D), no_common_point l1 l2 ∧ ¬ skew_lines l1 l2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_no_common_point_existence_of_non_skew_no_common_point_l110_11044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_values_l110_11030

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

-- State the theorem
theorem f_min_max_values :
  ∃ (x_min x_max : ℝ), x_min ∈ Set.Icc 0 (2 * π) ∧ x_max ∈ Set.Icc 0 (2 * π) ∧
  (∀ x ∈ Set.Icc 0 (2 * π), f x_min ≤ f x ∧ f x ≤ f x_max) ∧
  f x_min = -3 * π / 2 ∧ f x_max = π / 2 + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_values_l110_11030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_factorial_base_16_trailing_zeros_l110_11057

/-- The number of trailing zeros in n! when expressed in base b -/
def trailing_zeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 8 factorial -/
def eight_factorial : ℕ := 40320  -- Explicitly calculate 8!

/-- Base 16 -/
def base_16 : ℕ := 16

theorem eight_factorial_base_16_trailing_zeros :
  trailing_zeros eight_factorial base_16 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_factorial_base_16_trailing_zeros_l110_11057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_D_is_midpoint_sum_slope_y_intercept_l110_11095

-- Define the points
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)
def D : ℝ × ℝ := (0, 4)

-- Define D as the midpoint of AB
theorem D_is_midpoint : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) := by sorry

-- Define the slope of line CD
noncomputable def slope_CD : ℝ := (D.2 - C.2) / (D.1 - C.1)

-- Define the y-intercept of line CD
def y_intercept : ℝ := D.2

-- Theorem to prove
theorem sum_slope_y_intercept : slope_CD + y_intercept = 3.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_D_is_midpoint_sum_slope_y_intercept_l110_11095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_XYZ_min_triples_optimal_l110_11021

/-- The set of all triples (x,y,z) with non-negative integers x, y, z < 10 -/
def T : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 < 10 ∧ t.2.1 < 10 ∧ t.2.2 < 10}

/-- A's reply function for a given triple (X,Y,Z) and a named triple (a,b,c) -/
def reply (X Y Z a b c : ℕ) : ℕ :=
  Int.natAbs (X + Y - a - b) + Int.natAbs (Y + Z - b - c) + Int.natAbs (Z + X - c - a)

/-- The minimum number of triples needed to identify (X,Y,Z) -/
def min_triples : ℕ := 3

theorem identify_XYZ (X Y Z : ℕ) (h : (X, Y, Z) ∈ T) :
  ∃ (t₁ t₂ t₃ : ℕ × ℕ × ℕ),
    t₁ ∈ T ∧ t₂ ∈ T ∧ t₃ ∈ T ∧
    ∀ (X' Y' Z' : ℕ), (X', Y', Z') ∈ T →
      (reply X Y Z t₁.1 t₁.2.1 t₁.2.2 = reply X' Y' Z' t₁.1 t₁.2.1 t₁.2.2 ∧
       reply X Y Z t₂.1 t₂.2.1 t₂.2.2 = reply X' Y' Z' t₂.1 t₂.2.1 t₂.2.2 ∧
       reply X Y Z t₃.1 t₃.2.1 t₃.2.2 = reply X' Y' Z' t₃.1 t₃.2.1 t₃.2.2) →
      X = X' ∧ Y = Y' ∧ Z = Z' :=
by sorry

theorem min_triples_optimal :
  ∀ n < min_triples, ∃ (X Y Z X' Y' Z' : ℕ),
    (X, Y, Z) ∈ T ∧ (X', Y', Z') ∈ T ∧ (X, Y, Z) ≠ (X', Y', Z') ∧
    ∀ (t : ℕ × ℕ × ℕ) (i : Fin n), t ∈ T →
      reply X Y Z t.1 t.2.1 t.2.2 = reply X' Y' Z' t.1 t.2.1 t.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_XYZ_min_triples_optimal_l110_11021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linearFunctionIntegralEquality_quadraticFunctionIntegralEquality_l110_11043

noncomputable section

-- Define the intervals
def blackIntervals1 : List (Real × Real) := [(-1, -1/2), (1/2, 1)]
def whiteInterval1 : Real × Real := (-1/2, 1/2)

def blackIntervals2 : List (Real × Real) := [(-3/4, -1/4), (0, 1/4), (3/4, 1)]
def whiteIntervals2 : List (Real × Real) := [(-1, -3/4), (-1/4, 0), (1/4, 3/4)]

-- Define linear and quadratic functions
def linearFunction (a b : Real) : Real → Real := fun x ↦ a * x + b
def quadraticFunction (a b c : Real) : Real → Real := fun x ↦ a * x^2 + b * x + c

-- Define the integral of a function over an interval
noncomputable def integralOverInterval (f : Real → Real) (interval : Real × Real) : Real :=
  (interval.2 - interval.1) * (f interval.2 + f interval.1) / 2

-- Define the sum of integrals over a list of intervals
noncomputable def sumIntegralsOverIntervals (f : Real → Real) (intervals : List (Real × Real)) : Real :=
  intervals.foldl (fun acc interval ↦ acc + integralOverInterval f interval) 0

-- Theorem statements
theorem linearFunctionIntegralEquality (a b : Real) :
  sumIntegralsOverIntervals (linearFunction a b) blackIntervals1 =
  integralOverInterval (linearFunction a b) whiteInterval1 := by
  sorry

theorem quadraticFunctionIntegralEquality (a b c : Real) :
  sumIntegralsOverIntervals (quadraticFunction a b c) blackIntervals2 =
  sumIntegralsOverIntervals (quadraticFunction a b c) whiteIntervals2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linearFunctionIntegralEquality_quadraticFunctionIntegralEquality_l110_11043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_infection_equation_l110_11040

/-- Represents the average number of people infected by each person in a round -/
def x : ℝ := sorry

/-- The total number of people infected after two rounds -/
def total_infected : ℝ := 1 + x + x * (1 + x)

/-- The given total number of infected people -/
def given_total : ℝ := 36

/-- Theorem stating that the equation for the total number of infected people after two rounds is correct -/
theorem flu_infection_equation : total_infected = given_total := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_infection_equation_l110_11040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_book_purchase_proof_l110_11084

def joes_book_purchase (initial_amount : ℕ) (notebooks : ℕ) (notebook_cost : ℕ) (book_cost : ℕ) (remaining : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining
  let notebook_total := notebooks * notebook_cost
  let book_total := total_spent - notebook_total
  book_total / book_cost

theorem joes_book_purchase_proof :
  joes_book_purchase 56 7 4 7 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_book_purchase_proof_l110_11084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_l110_11063

theorem tan_inequality : Real.tan (-13 * Real.pi / 4) > Real.tan (-17 * Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_l110_11063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_pump_theorem_l110_11062

/-- Represents the well pump system -/
structure WellPump where
  tube_diameter : ℝ
  lever_ratio : ℝ
  pulls_per_minute : ℕ
  long_arm_descent : ℝ
  friction_factor : ℝ

/-- Calculates the mass of water pumped per hour -/
def mass_pumped_per_hour (pump : WellPump) : ℝ :=
  sorry

/-- Calculates the power exerted by the person in horsepower -/
def power_exerted (pump : WellPump) : ℝ :=
  sorry

/-- Main theorem: Proves the mass of water pumped and power exerted -/
theorem well_pump_theorem (pump : WellPump) 
  (h1 : pump.tube_diameter = 12)
  (h2 : pump.lever_ratio = 1/5)
  (h3 : pump.pulls_per_minute = 40)
  (h4 : pump.long_arm_descent = 60)
  (h5 : pump.friction_factor = 1/12) :
  (abs (mass_pumped_per_hour pump - 3258) < 1) ∧ 
  (abs (power_exerted pump - 3.72) < 0.01) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_pump_theorem_l110_11062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juniors_in_program_prove_juniors_in_program_l110_11072

theorem juniors_in_program (total_students : ℕ) 
  (junior_club_percent senior_club_percent : ℚ)
  (club_ratio_juniors club_ratio_seniors : ℕ) : Prop :=
  -- Total number of students in the program
  total_students = 40 ∧
  -- Percentage of juniors and seniors in the programming club
  junior_club_percent = 30 / 100 ∧
  senior_club_percent = 20 / 100 ∧
  -- Ratio of juniors to seniors in the programming club
  club_ratio_juniors = 3 ∧
  club_ratio_seniors = 2 ∧
  -- The number of juniors and seniors in the club are unequal
  club_ratio_juniors ≠ club_ratio_seniors ∧
  -- The theorem to prove
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total_students ∧
    juniors * club_ratio_seniors = seniors * club_ratio_juniors ∧
    (juniors : ℚ) * junior_club_percent = (club_ratio_juniors : ℚ) ∧
    (seniors : ℚ) * senior_club_percent = (club_ratio_seniors : ℚ) ∧
    juniors = 20

-- Proof of the theorem
theorem prove_juniors_in_program :
  juniors_in_program 40 (30/100) (20/100) 3 2 := by
  -- We use 'sorry' to skip the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juniors_in_program_prove_juniors_in_program_l110_11072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_div_1998_count_l110_11056

noncomputable def floor_div_1998 (n : ℕ) : ℤ := Int.floor ((n : ℚ)^2 / 1998)

theorem distinct_floor_div_1998_count :
  Finset.card (Finset.image floor_div_1998 (Finset.range 1997)) = 1498 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_div_1998_count_l110_11056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocentric_tetrahedron_equal_edges_isohedral_tetrahedron_rectangular_parallelepiped_l110_11089

structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

structure Parallelepiped where
  vertices : Fin 8 → ℝ × ℝ × ℝ

def is_orthocentric (t : Tetrahedron) : Prop :=
  sorry

def is_isohedral (t : Tetrahedron) : Prop :=
  sorry

def form_parallelepiped (t : Tetrahedron) : Parallelepiped :=
  sorry

def all_edges_equal (p : Parallelepiped) : Prop :=
  sorry

def is_rectangular (p : Parallelepiped) : Prop :=
  sorry

theorem orthocentric_tetrahedron_equal_edges 
  (t : Tetrahedron) (p : Parallelepiped) : 
  is_orthocentric t → p = form_parallelepiped t → all_edges_equal p := by
  sorry

theorem isohedral_tetrahedron_rectangular_parallelepiped 
  (t : Tetrahedron) (p : Parallelepiped) : 
  is_isohedral t → p = form_parallelepiped t → is_rectangular p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocentric_tetrahedron_equal_edges_isohedral_tetrahedron_rectangular_parallelepiped_l110_11089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_red_balls_bound_l110_11053

/-- The number of boxes -/
def n : ℕ := 2 -- You can change this to any positive natural number

/-- The probability of drawing a red ball from box i -/
noncomputable def P (i : ℕ) : ℝ :=
  1/2 + 1/2 * (1/3)^i

/-- The expected number of red balls in the nth box -/
noncomputable def E (n : ℕ) : ℝ :=
  3/2 + 1/2 * (1/3)^(n-1)

/-- Theorem: The expected number of red balls in the nth box is between 3/2 and 2 -/
theorem expected_red_balls_bound (n : ℕ) (h : n > 0) : 
  3/2 < E n ∧ E n ≤ 2 := by
  sorry

#check expected_red_balls_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_red_balls_bound_l110_11053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fish_catch_l110_11049

theorem min_fish_catch (a : Fin 5 → ℕ) 
  (h_desc : ∀ i j, i < j → a i ≥ a j)
  (h_sum : (Finset.univ.sum a) = 100) :
  a 0 + a 2 + a 4 ≥ 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fish_catch_l110_11049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l110_11024

def mySequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2010 ∧ 
  a 2 = 2012 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n + 2

theorem sequence_1000th_term (a : ℕ → ℕ) (h : mySequence a) : a 1000 = 2343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l110_11024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_cos_2a_plus_pi_3_cos_2a_special_l110_11093

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/2 * Real.cos (2*x) + Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2

-- Theorem statements
theorem f_max_value : ∃ (M : ℝ), M = 0 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

theorem cos_2a_plus_pi_3 (a : ℝ) (h : f a = -1/5) : Real.cos (2*a + π/3) = -4/5 := by sorry

theorem cos_2a_special (a : ℝ) (h1 : π/3 < a) (h2 : a < π/2) (h3 : f a = -1/5) : 
  Real.cos (2*a) = (-3 * Real.sqrt 3 + 4) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_cos_2a_plus_pi_3_cos_2a_special_l110_11093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_in_triangle_arrangement_l110_11086

noncomputable def energy_between_charges (distance : ℝ) (base_energy : ℝ) : ℝ :=
  base_energy * (1 / distance)

def total_energy (e_ab e_bc e_ac : ℝ) : ℝ :=
  e_ab + e_bc + e_ac

theorem energy_increase_in_triangle_arrangement (d : ℝ) (h_d_pos : d > 0) :
  let initial_energy_per_pair := 18 / 3
  let initial_total_energy := total_energy initial_energy_per_pair initial_energy_per_pair initial_energy_per_pair
  let new_e_ab := initial_energy_per_pair
  let new_e_ac := energy_between_charges (3 * d / 4) initial_energy_per_pair
  let new_e_bc := energy_between_charges (d / 4) initial_energy_per_pair
  let new_total_energy := total_energy new_e_ab new_e_ac new_e_bc
  new_total_energy - initial_total_energy = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_in_triangle_arrangement_l110_11086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_theorem_l110_11001

def is_valid_permutation (p : Fin 102 → Fin 102) : Prop :=
  Function.Bijective p ∧ ∀ k : Fin 102, (p k : ℕ) % (k : ℕ) = 0

def valid_permutations : List (Fin 102 → Fin 102) :=
  [
    (λ k ↦ if k = 1 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 2 else if k = 2 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 3 else if k = 3 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 6 else if k = 6 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 17 else if k = 17 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 34 else if k = 34 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 51 else if k = 51 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 2 else if k = 2 then 6 else if k = 6 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 2 else if k = 2 then 34 else if k = 34 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 3 else if k = 3 then 6 else if k = 6 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 3 else if k = 3 then 51 else if k = 51 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 17 else if k = 17 then 34 else if k = 34 then 102 else if k = 102 then 1 else k),
    (λ k ↦ if k = 1 then 17 else if k = 17 then 51 else if k = 51 then 102 else if k = 102 then 1 else k)
  ]

theorem permutation_theorem :
  ∀ p : Fin 102 → Fin 102, is_valid_permutation p ↔ p ∈ valid_permutations := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_theorem_l110_11001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_transformation_l110_11028

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s,
    prove that x^2 - (a+b)x + abc = 0 has roots a(r+b) and a(s+b) -/
theorem quadratic_root_transformation (a b c r s : ℝ) 
  (hr : a * r^2 + b * r + c = 0)
  (hs : a * s^2 + b * s + c = 0)
  (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - (a + b) * x + a * b * c
  (f (a * (r + b)) = 0) ∧ (f (a * (s + b)) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_transformation_l110_11028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l110_11016

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := y^2/4 + x^2/3 = 1

/-- Point A -/
def A : ℝ × ℝ := (1, 1)

/-- Point B -/
def B : ℝ × ℝ := (0, -1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_sum :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 →
    distance P A + distance P B ≤ 5 ∧
    ∃ Q : ℝ × ℝ, ellipse Q.1 Q.2 ∧ distance Q A + distance Q B = 5 :=
by
  sorry

#check max_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l110_11016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_zero_necessary_not_sufficient_for_sin_one_l110_11035

theorem cos_zero_necessary_not_sufficient_for_sin_one :
  (∀ α : Real, Real.sin α = 1 → Real.cos α = 0) ∧
  (∃ α : Real, Real.cos α = 0 ∧ Real.sin α ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_zero_necessary_not_sufficient_for_sin_one_l110_11035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l110_11017

theorem factorial_simplification : (13 * 12 * Nat.factorial 11) / (Nat.factorial 11 + 2 * Nat.factorial 10) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l110_11017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_ratio_of_cubes_l110_11099

/-- Represents a cube with mass, volume, and side length -/
structure Cube where
  mass : ℝ
  volume : ℝ
  side_length : ℝ

/-- The density of a cube -/
noncomputable def density (c : Cube) : ℝ := c.mass / c.volume

theorem mass_ratio_of_cubes (c1 c2 : Cube)
  (h1 : density c2 = 2 * density c1)
  (h2 : c2.side_length = 2 * c1.side_length)
  (h3 : c1.volume = c1.side_length ^ 3)
  (h4 : c2.volume = c2.side_length ^ 3) :
  c2.mass = 16 * c1.mass := by
  sorry

#check mass_ratio_of_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_ratio_of_cubes_l110_11099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l110_11023

-- Define the function g
noncomputable def g (B : ℝ) : ℝ := 2 + Real.sin B ^ 4

-- State the theorem
theorem range_of_g :
  ∀ B : ℝ, B ∉ {n : ℝ | ∃ k : ℤ, n = k * Real.pi} →
  ∃ y ∈ Set.Icc 2 3, g B = y ∧
  ∀ z, g B = z → z ∈ Set.Icc 2 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l110_11023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_supremum_l110_11047

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/2 * x^2 - 2*x + 5

def domain : Set ℝ := Set.Icc (-1) 2

theorem f_supremum :
  ∃ (m : ℝ), m = 7 ∧ IsLUB (Set.image f domain) m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_supremum_l110_11047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l110_11069

/-- Given two circles A and B, where a 90° arc on A has the same length as a 60° arc on B,
    the ratio of the area of A to the area of B is 4/9. -/
theorem circle_area_ratio (A B : ℝ → ℝ × ℝ) (R_A R_B : ℝ) :
  (∀ t, A t = (R_A * Real.cos t, R_A * Real.sin t)) →
  (∀ t, B t = (R_B * Real.cos t, R_B * Real.sin t)) →
  R_A * (π / 2) = R_B * (π / 3) →
  (π * R_A ^ 2) / (π * R_B ^ 2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l110_11069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l110_11067

/-- Calculates the time taken for two trains to cross each other when moving in opposite directions -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_distance := length1 + length2
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  total_distance / relative_speed

/-- Theorem stating that the time taken for the given trains to cross each other is 9 seconds -/
theorem trains_crossing_time : 
  let length1 : ℝ := 270
  let length2 : ℝ := 230.04
  let speed1 : ℝ := 120
  let speed2 : ℝ := 80
  Int.floor (time_to_cross length1 length2 speed1 speed2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l110_11067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_negative_roots_l110_11050

theorem unique_positive_negative_roots :
  ∃! (p n : ℝ), p > 0 ∧ n < 0 ∧
  (p^4 + 5*p^3 + 15*p - 9 = 0) ∧
  (n^4 + 5*n^3 + 15*n - 9 = 0) ∧
  (∀ x : ℝ, x^4 + 5*x^3 + 15*x - 9 = 0 → (x = p ∨ x = n ∨ x ≠ p ∧ x ≠ n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_negative_roots_l110_11050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2001_l110_11052

def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ a 5 = 8 ∧ ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 7

theorem sequence_2001 (a : ℕ → ℤ) (h : my_sequence a) : a 2001 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2001_l110_11052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OFA_l110_11092

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus F of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Line l passing through F with inclination angle 60° -/
noncomputable def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

/-- Point A is the intersection of line l and the parabola on the positive x-axis -/
noncomputable def point_A : ℝ × ℝ := (1 + Real.sqrt 3, 2 * Real.sqrt 3)

/-- The area of triangle OFA -/
noncomputable def triangle_area (O F A : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((F.1 - O.1) * (A.2 - O.2) - (A.1 - O.1) * (F.2 - O.2))

/-- Theorem stating that the area of triangle OFA is √3 -/
theorem area_of_triangle_OFA : 
  triangle_area (0, 0) focus point_A = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OFA_l110_11092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l110_11054

/-- Proves that a train with given length and speed takes the calculated time to pass a bridge of given length -/
theorem train_bridge_passing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 460)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 140) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 48 := by
  -- Convert given values to reals
  have train_length_real : ℝ := 460
  have train_speed_kmh_real : ℝ := 45
  have bridge_length_real : ℝ := 140

  -- Calculate total distance
  let total_distance : ℝ := train_length_real + bridge_length_real

  -- Convert speed from km/h to m/s
  let train_speed_ms : ℝ := train_speed_kmh_real * 1000 / 3600

  -- Calculate time
  let time : ℝ := total_distance / train_speed_ms

  -- Prove the equality
  sorry

#check train_bridge_passing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l110_11054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l110_11009

/-- The circle with center (0, 1) and radius 2 -/
def my_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

/-- The line 3x - 4y - 1 = 0 -/
def my_line (x y : ℝ) : Prop := 3*x - 4*y - 1 = 0

/-- The chord length intercepted by the circle on the line -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

theorem chord_length_is_correct :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
  my_line x₁ y₁ ∧ my_line x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l110_11009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tesseract_equality_l110_11004

noncomputable section

-- Define the edge lengths of the tesseract
def edge_length (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the 3-dimensional surface area
def surface_area (x : ℝ) : ℝ :=
  2 * (
    edge_length 3 x * edge_length 6 x +
    edge_length 3 x * edge_length 9 x +
    edge_length 3 x * edge_length 12 x +
    edge_length 6 x * edge_length 9 x +
    edge_length 6 x * edge_length 12 x +
    edge_length 9 x * edge_length 12 x
  )

-- Define the 4-dimensional volume
def volume (x : ℝ) : ℝ :=
  edge_length 3 x * edge_length 6 x * edge_length 9 x * edge_length 12 x

-- Theorem statement
theorem tesseract_equality (x : ℝ) (h : x > 0) :
  surface_area x = volume x → x = 432 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tesseract_equality_l110_11004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l110_11073

/-- Given a class with two groups of students, this theorem proves
    the average score of the second group based on known information. -/
theorem class_average_problem (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) :
  n₁ = 15 →
  n₂ = 10 →
  avg₁ = 80 / 100 →
  avg_total = 84 / 100 →
  (n₁ + n₂ : ℚ) * avg_total - n₁ * avg₁ = n₂ * (90 / 100) := by
  intro h1 h2 h3 h4
  norm_num
  ring
  sorry

#check class_average_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l110_11073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l110_11068

/-- The equation of a hyperbola in the form ((2y-2)^2)/5^2 - ((3x-4)^2)/4^2 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  ((2*y - 2)^2) / 5^2 - ((3*x - 4)^2) / 4^2 = 1

/-- The center of a hyperbola -/
noncomputable def hyperbola_center : ℝ × ℝ := (4/3, 1)

/-- Theorem stating that the center of the given hyperbola is (4/3, 1) -/
theorem hyperbola_center_correct :
  ∀ x y : ℝ, hyperbola_equation x y → (x, y) = hyperbola_center :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l110_11068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_ratio_l110_11007

/-- Represents the cycling scenario with Jack and Peter -/
structure CyclingScenario where
  speed : ℝ
  distance_home_to_store : ℝ
  distance_store_to_peter : ℝ
  total_distance : ℝ

/-- The specific cycling scenario described in the problem -/
def problem_scenario : CyclingScenario where
  speed := 1  -- We set an arbitrary positive speed
  distance_home_to_store := 100  -- Calculated from the problem
  distance_store_to_peter := 50
  total_distance := 250

/-- The theorem stating the ratio of cycling times -/
theorem cycling_time_ratio (s : CyclingScenario) (h1 : s.speed > 0) 
    (h2 : s.distance_store_to_peter = 50)
    (h3 : s.total_distance = s.distance_home_to_store + 2 * s.distance_store_to_peter + s.distance_store_to_peter)
    (h4 : s.total_distance = 250) :
  (s.distance_home_to_store / s.speed) / (s.distance_store_to_peter / s.speed) = 2 := by
  sorry

#check cycling_time_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_ratio_l110_11007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_difference_implies_value_l110_11013

-- Define the condition that the difference is a monomial
def is_monomial_difference (m n : ℤ) : Prop :=
  ∃ (k : ℤ) (c : ℝ), ∀ (x y : ℝ), 
    3 * (x^(m+2)) * (y^4) - 8 * (x^6) * (y^(n+1)) = c * (x^k) * (y^k)

-- State the theorem
theorem monomial_difference_implies_value (m n : ℤ) :
  is_monomial_difference m n → (-1 : ℤ) * m^(n.toNat) = -64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_difference_implies_value_l110_11013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_x_equals_g_l110_11079

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -4 ∧ x ≤ -1 then -3 - x
  else if x > -1 ∧ x < 1 then -Real.sqrt (4 - x^2) - 3
  else if x ≥ 1 ∧ x ≤ 4 then 2*x - 6
  else 0  -- Default value for x outside the defined intervals

-- Define g(x) = f(-x)
noncomputable def g (x : ℝ) : ℝ := f (-x)

-- Theorem statement
theorem f_neg_x_equals_g (x : ℝ) :
  (x ≥ 1 ∧ x ≤ 4 → g x = -3 + x) ∧
  (x ≥ -1 ∧ x ≤ 1 → g x = -Real.sqrt (4 - x^2) - 3) ∧
  (x ≥ -4 ∧ x ≤ -1 → g x = -2*x - 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_x_equals_g_l110_11079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_rational_triples_l110_11002

theorem inequality_and_rational_triples :
  (∀ x y z : ℝ, x ≠ 1 → y ≠ 1 → z ≠ 1 → x * y * z = 1 →
    x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1) ∧
  (∃ f : ℕ → ℚ × ℚ × ℚ, ∀ n : ℕ,
    let (x, y, z) := f n;
    x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ x * y * z = 1 ∧
    x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_rational_triples_l110_11002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_l110_11087

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

-- Define the solutions
noncomputable def solution1 : ℝ := 1 + Real.sqrt 2
noncomputable def solution2 : ℝ := 1 - Real.sqrt 2

-- Theorem statement
theorem quadratic_solutions :
  (quadratic_equation solution1) ∧ (quadratic_equation solution2) :=
by
  sorry -- Skip the proof for now

#check quadratic_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_l110_11087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tangent_function_l110_11090

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

theorem period_of_tangent_function :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tangent_function_l110_11090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumsphere_radius_l110_11025

/-- The radius of the circumscribed sphere of a tetrahedron P-ABC -/
noncomputable def circumsphere_radius (PA AB AC BC : ℝ) : ℝ :=
  let r := BC / (2 * Real.sqrt 3)
  let d := (PA^2 + r^2 - (PA - r)^2) / (2 * PA)
  Real.sqrt (d^2 + r^2)

/-- Theorem: The radius of the circumscribed sphere of the given tetrahedron is 4 -/
theorem tetrahedron_circumsphere_radius :
  circumsphere_radius 4 (2 * Real.sqrt 3) (2 * Real.sqrt 3) 6 = 4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval circumsphere_radius 4 (2 * Real.sqrt 3) (2 * Real.sqrt 3) 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumsphere_radius_l110_11025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l110_11039

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is inside a rectangle --/
def is_inside (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.x_max ∧ 0 ≤ p.y ∧ p.y ≤ r.y_max

/-- Predicate to check if x > 4y for a point --/
def satisfies_condition (p : Point) : Prop :=
  p.x > 4 * p.y

/-- The probability of selecting a point satisfying the condition --/
noncomputable def probability (r : Rectangle) : ℝ :=
  (3 : ℝ) / 32

theorem probability_theorem (r : Rectangle) (h1 : r.x_max = 3000) (h2 : r.y_max = 4000) :
  probability r = (3 : ℝ) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l110_11039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pogo_leaves_four_footprints_per_meter_main_result_l110_11037

/-- Represents the number of footprints Pogo leaves per meter -/
def pogo_footprints_per_meter : ℝ := 4

/-- Represents the number of footprints Grimzi leaves per meter -/
def grimzi_footprints_per_meter : ℝ := 0.5

/-- The distance traveled by both Pogo and Grimzi -/
def distance_traveled : ℝ := 6000

/-- The total number of footprints left by both creatures -/
def total_footprints : ℝ := 27000

/-- Theorem stating that Pogo leaves 4 footprints per meter -/
theorem pogo_leaves_four_footprints_per_meter :
  pogo_footprints_per_meter = 4 :=
by
  rfl

/-- Main theorem proving the result based on given conditions -/
theorem main_result :
  pogo_footprints_per_meter * distance_traveled +
  grimzi_footprints_per_meter * distance_traveled = total_footprints ∧
  pogo_footprints_per_meter = 4 :=
by
  apply And.intro
  · simp [pogo_footprints_per_meter, grimzi_footprints_per_meter, distance_traveled, total_footprints]
    norm_num
  · exact pogo_leaves_four_footprints_per_meter


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pogo_leaves_four_footprints_per_meter_main_result_l110_11037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_volume_cylinder_l110_11077

/-- The volume of a cylinder with given radius and height -/
noncomputable def cylinderVolume (radius : ℝ) (height : ℝ) : ℝ := Real.pi * radius^2 * height

theorem triple_volume_cylinder :
  let r₁ : ℝ := 8
  let h₁ : ℝ := 7
  let r₂ : ℝ := 8
  let h₂ : ℝ := 21
  cylinderVolume r₂ h₂ = 3 * cylinderVolume r₁ h₁ := by
  -- Unfold the definitions
  unfold cylinderVolume
  -- Simplify the expressions
  simp [Real.pi]
  -- The proof is complete
  sorry

#check triple_volume_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_volume_cylinder_l110_11077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_x_l110_11018

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 
  (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))

noncomputable def y (t : ℝ) : ℝ := 
  t / Real.sqrt (1 - t^2)

-- State the theorem
theorem derivative_y_x (t : ℝ) (h : t ≠ 0 ∧ t ≠ 1 ∧ t ≠ -1) : 
  deriv y t / deriv x t = 1 / Real.arcsin t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_x_l110_11018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l110_11064

-- Define the conic section C
def conic_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 + Real.sin θ^2)

-- Define the line l
def line_equation (t x y : ℝ) : Prop :=
  x = -1 + t ∧ y = Real.sqrt 3 * t

-- Define the intersection points M and N
def intersection_points (M N : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    line_equation t₁ M.1 M.2 ∧
    line_equation t₂ N.1 N.2 ∧
    conic_equation (Real.sqrt (M.1^2 + M.2^2)) (Real.arctan (M.2 / M.1)) ∧
    conic_equation (Real.sqrt (N.1^2 + N.2^2)) (Real.arctan (N.2 / N.1))

-- Define the left focus F1
noncomputable def F1 : ℝ × ℝ := sorry

-- Theorem statement
theorem sum_of_distances (M N : ℝ × ℝ) (h : intersection_points M N) :
  Real.sqrt ((M.1 - F1.1)^2 + (M.2 - F1.2)^2) +
  Real.sqrt ((N.1 - F1.1)^2 + (N.2 - F1.2)^2) = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l110_11064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l110_11060

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c : ℝ) (k : ℝ) : ℝ :=
  |k| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between x + 3y - 4 = 0 and 2x + 6y - 9 = 0 is √10 / 20 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines 1 3 (-4) 1 = Real.sqrt 10 / 20 := by
  sorry

#check distance_specific_parallel_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l110_11060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l110_11070

-- Define the curves
noncomputable def C₁ (x : ℝ) : ℝ := Real.sin x
def C₂ (x y r : ℝ) : Prop := x^2 + (y + r)^2 = r^2

-- State the theorem
theorem curves_intersection (r : ℝ) (hr : r > 0) :
  (∃ x y : ℝ, C₁ x = y ∧ C₂ x y r) ∧
  ¬(∃ n : ℕ, ∀ x y : ℝ, (C₁ x = y ∧ C₂ x y r) → 
    (∃ l : List (ℝ × ℝ), l.length = n ∧ ∀ p ∈ l, C₁ p.1 = p.2 ∧ C₂ p.1 p.2 r)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l110_11070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_us_land_distribution_l110_11075

/-- The population of the United States in 2020 --/
def us_population : ℕ := 331000000

/-- The total land area of the United States in square miles --/
def us_land_area : ℕ := 3796742

/-- The number of square feet in a square mile --/
def sq_feet_per_sq_mile : ℕ := 5280^2

/-- Approximate average number of square feet per person --/
def approx_sq_feet_per_person : ℕ := 320000

/-- Tolerance for approximation --/
def tolerance : ℚ := 1/100

theorem us_land_distribution :
  |(us_land_area * sq_feet_per_sq_mile : ℚ) / us_population - approx_sq_feet_per_person| ≤ tolerance * approx_sq_feet_per_person := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_us_land_distribution_l110_11075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_numbers_count_l110_11097

def T : Finset Nat := Finset.range 100

theorem remaining_numbers_count : 
  (T.filter (λ n => ¬(4 ∣ n) ∧ ¬(5 ∣ n))).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_numbers_count_l110_11097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_line_equation_l110_11008

/-- The slope of the first line -/
noncomputable def m₁ : ℝ := 2

/-- The y-intercept of the first line -/
noncomputable def b₁ : ℝ := 2

/-- The y-coordinate of the point where the second line intersects the first line -/
noncomputable def y₀ : ℝ := 520

/-- The equation of the first line -/
noncomputable def line₁ (x : ℝ) : ℝ := m₁ * x + b₁

/-- The x-coordinate of the intersection point -/
noncomputable def x₀ : ℝ := (y₀ - b₁) / m₁

/-- The slope of the second line -/
noncomputable def m₂ : ℝ := -1 / m₁

/-- The equation of the second line -/
noncomputable def line₂ (x : ℝ) : ℝ := m₂ * (x - x₀) + y₀

theorem second_line_equation :
  ∀ x : ℝ, line₂ x = -1/2 * x + 649.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_line_equation_l110_11008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_evaluation_l110_11029

theorem ceiling_evaluation : ⌈(4 : ℝ) * (8 - 3/4)⌉ = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_evaluation_l110_11029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l110_11041

/-- A line in the xy-plane defined by the equation 4x + 7y = 28 -/
def line (x y : ℝ) : Prop := 4 * x + 7 * y = 28

/-- The slope of the line -/
def line_slope : ℚ := -4/7

/-- The x-intercept of the line -/
def x_intercept : ℝ := 7

/-- Theorem stating that the slope and x-intercept of the line are correct -/
theorem line_properties :
  (∀ x y : ℝ, line x y → y = (line_slope : ℝ) * x + 4) ∧
  line x_intercept 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l110_11041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_simplification_l110_11026

theorem complex_square_simplification :
  (4 - 3 * Complex.I)^2 + 2 = 9 - 24 * Complex.I :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_simplification_l110_11026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_all_ones_existence_l110_11076

theorem quadratic_polynomial_all_ones_existence :
  ∃ (a b c : ℤ), ∀ (k : ℕ), k > 0 →
    let n : ℕ := (10^k - 1) / 9
    let P (x : ℕ) := a * x^2 + b * x + c
    (P n : ℤ) = (10^(2*k + 1) - 1) / 9 :=
by
  -- We'll use a = 90, b = 20, c = 1
  use 90, 20, 1
  intro k hk
  -- Define n and P as per the let bindings
  let n := (10^k - 1) / 9
  let P (x : ℕ) := 90 * x^2 + 20 * x + 1
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_all_ones_existence_l110_11076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_subadditive_l110_11015

/-- A function satisfying certain properties on non-negative real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a, a ≥ 0 → (f a = 0 ↔ a = 0)) ∧
  (∀ a b, a ≥ 0 → b ≥ 0 → f (a * b) = f a * f b) ∧
  (∀ a b, a ≥ 0 → b ≥ 0 → f (a + b) ≤ 2 * max (f a) (f b))

/-- The main theorem stating that for a special function f, 
    f(a+b) ≤ f(a) + f(b) for all non-negative real numbers a and b -/
theorem special_function_subadditive {f : ℝ → ℝ} (hf : special_function f) :
  ∀ a b : ℝ, a ≥ 0 → b ≥ 0 → f (a + b) ≤ f a + f b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_subadditive_l110_11015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l110_11071

-- Define the function f
def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∀ x < 0, (deriv (f a b c)) x < 0) →  -- f is decreasing on (-∞,0)
  (∀ x ∈ Set.Ioo 0 1, (deriv (f a b c)) x > 0) →  -- f is increasing on (0,1)
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →  -- f has three distinct zeros
  f a b c 1 = 0 →  -- 1 is one of the zeros
  b = 0 ∧ f a b c 2 > -5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l110_11071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_6_l110_11083

noncomputable def f (x : ℝ) : ℝ := 1 + x^2 + Real.log x / Real.log 2

theorem inverse_f_at_6 : 
  ∃ (y : ℝ), f y = 6 ∧ y = 2 :=
by
  use 2
  constructor
  · simp [f]
    sorry -- Proof of f 2 = 6
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_6_l110_11083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_tangent_line_l110_11046

-- Define the quadratic function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + a * x - 2 * b

-- Define the derivative of f(x)
def f_prime (a x : ℝ) : ℝ := 2 * a * x + a

-- Define the function h(x)
noncomputable def h (a b x : ℝ) : ℝ := x * Real.log x + f a b x

-- Define the derivative of h(x)
noncomputable def h_prime (a b x : ℝ) : ℝ := Real.log x + 1 - 2 * x - 1

theorem quadratic_function_and_tangent_line :
  ∃ (a b : ℝ),
    (f a b 2 = -4) ∧
    (f_prime a 1 = -3) ∧
    (a = -1) ∧
    (b = -1) ∧
    (∀ (x y : ℝ), y = h_prime a b 1 * (x - 1) ↔ 2 * x + y - 2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_tangent_line_l110_11046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_exists_l110_11027

def square_sides : List ℕ := [2, 8, 14, 16, 18, 20, 28, 30, 36]

def total_area : ℕ := (square_sides.map (λ x => x * x)).sum

theorem rectangle_exists : ∃ (length width : ℕ), 
  length * width = total_area ∧ 
  length ≥ (square_sides.maximum?.getD 0) ∧ 
  width ≥ (square_sides.maximum?.getD 0) ∧
  length ≥ width := by
  sorry

#eval total_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_exists_l110_11027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_result_l110_11005

-- Define the operations
def square_op : ℚ → ℚ → ℚ := (·*·)
def circle_op : ℚ → ℚ → ℚ := (·+·)

-- Define the given equations
axiom eq1 : circle_op (square_op (3 + 1/11) (3 + 15/17)) 1 = 13
axiom eq2 : circle_op (square_op 2 2) 1 = 5

-- Define A and B
def A : ℚ := circle_op (square_op 2 (3 + 15/17)) 1
def B : ℚ := circle_op (square_op (3 + 1/11) 2) 1

-- Theorem to prove
theorem result : circle_op A B = 2982/187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_result_l110_11005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_problem_l110_11000

theorem power_of_three_problem (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_problem_l110_11000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_solution_l110_11006

theorem complex_inequality_solution (m : ℝ) : 
  (Complex.I * (m^2 - 4*m + 3) + 10 - (m^2 - Complex.I * (m^2 - 3*m))).re > 0 ↔ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_solution_l110_11006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scramble_language_word_count_l110_11058

/-- The number of letters in the Scramble alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length in the Scramble language -/
def max_word_length : ℕ := 5

/-- Calculate the number of words of a given length using all letters -/
def words_of_length (n : ℕ) : ℕ := alphabet_size ^ n

/-- Calculate the number of words of a given length without using the letter B -/
def words_without_b_of_length (n : ℕ) : ℕ := (alphabet_size - 1) ^ n

/-- The total number of words in the Scramble language -/
def total_words : ℕ := Finset.sum (Finset.range (max_word_length + 1)) words_of_length

/-- The total number of words without the letter B -/
def total_words_without_b : ℕ := Finset.sum (Finset.range (max_word_length + 1)) words_without_b_of_length

/-- The number of valid words in the Scramble language -/
def valid_words : ℕ := total_words - total_words_without_b

theorem scramble_language_word_count :
  valid_words = 1692701 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scramble_language_word_count_l110_11058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crosses_is_18_l110_11031

/-- A configuration of crosses on a 10x10 grid -/
def Configuration := Fin 10 → Fin 10 → Bool

/-- A cross is unique in its row or column -/
def is_valid (config : Configuration) : Prop :=
  ∀ i j : Fin 10, config i j → 
    ((∀ k : Fin 10, k ≠ j → ¬config i k) ∨ 
     (∀ k : Fin 10, k ≠ i → ¬config k j))

/-- Count the number of crosses in a configuration -/
def count_crosses (config : Configuration) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 10)) fun j =>
      if config i j then 1 else 0)

/-- The maximum number of crosses in a valid configuration is 18 -/
theorem max_crosses_is_18 :
  (∃ config : Configuration, is_valid config ∧ count_crosses config = 18) ∧
  (∀ config : Configuration, is_valid config → count_crosses config ≤ 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crosses_is_18_l110_11031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_baseball_cards_l110_11010

def initial_cards : ℕ := 97
def torn_cards : ℕ := 8
def sam_purchase : ℕ := 15
def alex_purchase_fraction : ℚ := 1/4
def new_gift_cards : ℕ := 6

def dans_final_cards : ℕ := 62

theorem dans_baseball_cards : 
  (initial_cards - torn_cards - sam_purchase - 
   ((initial_cards - torn_cards - sam_purchase : ℚ) * alex_purchase_fraction).floor + 
   new_gift_cards) = dans_final_cards := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_baseball_cards_l110_11010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l110_11034

noncomputable def vector_a : Fin 2 → ℝ := ![1, 2]
noncomputable def vector_b (x : ℝ) : Fin 2 → ℝ := ![3, x]
noncomputable def vector_c (y : ℝ) : Fin 2 → ℝ := ![2, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * u i

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

noncomputable def angle (u v : Fin 2 → ℝ) : ℝ :=
  Real.arccos ((u 0 * v 0 + u 1 * v 1) / (Real.sqrt (u 0^2 + u 1^2) * Real.sqrt (v 0^2 + v 1^2)))

theorem vector_problem (x y : ℝ) :
  parallel vector_a (vector_b x) →
  perpendicular vector_a (vector_c y) →
  (x = 6 ∧ y = -1) ∧
  angle (![-1, -2]) (![3, 1]) = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l110_11034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log23_not_expressible_l110_11003

-- Define the given logarithm values
noncomputable def log2 : ℝ := Real.log 2
noncomputable def log5 : ℝ := Real.log 5

-- Define a function to check if a logarithm can be expressed in terms of log2 and log5
def expressible (x : ℝ) : Prop :=
  ∃ (a b : ℚ), x = a * log2 + b * log5

-- State the theorem
theorem log23_not_expressible :
  expressible (Real.log (1/5)) ∧
  expressible (Real.log 45) ∧
  expressible (Real.log 250) ∧
  expressible (Real.log 0.2) ∧
  ¬ expressible (Real.log 23) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log23_not_expressible_l110_11003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alfreds_scooter_gain_percent_l110_11051

/-- Calculates the gain percent given the purchase price, repair cost, and selling price of an item. -/
noncomputable def gain_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100

/-- Theorem stating that Alfred's gain percent on the scooter is approximately 1.75% -/
theorem alfreds_scooter_gain_percent :
  let purchase_price : ℝ := 4700
  let repair_cost : ℝ := 1000
  let selling_price : ℝ := 5800
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |gain_percent purchase_price repair_cost selling_price - 1.75| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alfreds_scooter_gain_percent_l110_11051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_implies_equality_l110_11085

/-- A linear transformation in 2D space -/
def LinearTransformation (a b c d : ℝ) : ℝ × ℝ → ℝ × ℝ :=
  fun v => (a * v.1 + b * v.2, c * v.1 + d * v.2)

/-- The origin point -/
def origin : ℝ × ℝ := (0, 0)

/-- Distance from a point to the origin -/
noncomputable def distanceToOrigin (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

theorem constant_ratio_implies_equality (a b c d : ℝ) :
  (∀ p : ℝ × ℝ, p ≠ origin →
    ∃ k : ℝ, distanceToOrigin (LinearTransformation a b c d p) = k * distanceToOrigin p) →
  a^2 + c^2 = b^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_implies_equality_l110_11085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l110_11094

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (4 - x)) / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | 0 ≤ x ∧ x < 1} ∪ {x | 1 < x ∧ x ≤ 4}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l110_11094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l110_11081

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)

noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

theorem vector_properties (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ Real.pi) :
  (∃ (k : ℝ), a = k • b θ → θ = Real.pi / 6) ∧
  (∃ (proj : ℝ × ℝ), proj = (((a.1 * (b θ).1 + a.2 * (b θ).2) / ((b θ).1^2 + (b θ).2^2)) • b θ) ∧
                      proj = -(Real.sqrt 3) • b θ → b θ = (-1, 0)) := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l110_11081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l110_11080

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 7 = 0

-- Define the centers and radii of the circles
def center_C1 : ℝ × ℝ := (0, 0)
def radius_C1 : ℝ := 1
def center_C2 : ℝ × ℝ := (0, 4)
def radius_C2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius_C1 + radius_C2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l110_11080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_adjacent_sum_l110_11033

def odd_numbers : List Nat := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 10 ∧ arr.toFinset = odd_numbers.toFinset

def adjacent_sum (arr : List Nat) (i : Nat) : Nat :=
  arr[i % arr.length]! + arr[(i + 1) % arr.length]! + arr[(i + 2) % arr.length]!

def max_adjacent_sum (arr : List Nat) : Nat :=
  (List.range arr.length).map (adjacent_sum arr) |>.maximum?.getD 0

theorem min_max_adjacent_sum :
  ∃ m : Nat, (∀ arr : List Nat, is_valid_arrangement arr →
    max_adjacent_sum arr ≥ m) ∧ (∃ arr : List Nat, is_valid_arrangement arr ∧ max_adjacent_sum arr = m) ∧ m = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_adjacent_sum_l110_11033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_bound_l110_11038

/-- Predicate indicating that three points form a triangle -/
def IsTriangle (A B C : EuclideanPlane) : Prop := sorry

/-- Predicate indicating that a point is an excircle point of a side of a triangle -/
def IsExcirclePoint (P X Y : EuclideanPlane) : Prop := sorry

/-- Function to calculate the perimeter of a triangle given its vertices -/
def perimeter (A B C : EuclideanPlane) : ℝ := sorry

/-- Given a triangle ABC with excircles touching sides BC, CA, AB at D, E, F respectively,
    the perimeter of triangle ABC is at most twice the perimeter of triangle DEF. -/
theorem triangle_perimeter_bound (A B C D E F : EuclideanPlane) : 
  IsTriangle A B C →
  IsExcirclePoint D B C →
  IsExcirclePoint E C A →
  IsExcirclePoint F A B →
  perimeter A B C ≤ 2 * perimeter D E F := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_bound_l110_11038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l110_11066

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y < -1 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l110_11066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l110_11091

/-- The radius of spheres in a cone given specific conditions -/
theorem sphere_radius_in_cone (R H : ℝ) (r : ℝ) : 
  R = 6 → -- Base radius of the cone
  H = 10 → -- Height of the cone
  (H - r)^2 + R^2 = (2 * Real.sqrt 34 - r)^2 → -- Relationship derived from tangency conditions
  r = 100 / (20 - 4 * Real.sqrt 34) := by
  sorry

#check sphere_radius_in_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l110_11091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_given_inequality_l110_11088

open Real

-- Define the quadratic function
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem range_of_x_given_inequality :
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x a > 0) →
  {x : ℝ | x < 1 ∨ x > 3} = {x | ∀ a ∈ Set.Icc (-1) 1, f x a > 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_given_inequality_l110_11088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_very_special_numbers_l110_11045

/-- A very special number is a positive real number with a decimal representation
    consisting entirely of digits 0 and 5. -/
def is_very_special (x : ℝ) : Prop :=
  x > 0 ∧ ∃ (n : ℕ) (d : ℕ → Fin 2), x = ∑' i, (d i).val * (5 : ℝ) / 10^(i + 1)

/-- A number with up to three decimal places -/
def has_up_to_three_decimal_places (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = ↑n / 1000

/-- The statement to be proven -/
theorem smallest_sum_of_very_special_numbers :
  ∃ (a b : ℝ),
    is_very_special a ∧
    is_very_special b ∧
    has_up_to_three_decimal_places a ∧
    has_up_to_three_decimal_places b ∧
    a + b = 1 ∧
    (∀ (x y z : ℝ),
      is_very_special x ∧
      is_very_special y ∧
      is_very_special z ∧
      has_up_to_three_decimal_places x ∧
      has_up_to_three_decimal_places y ∧
      has_up_to_three_decimal_places z →
      x + y + z ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_very_special_numbers_l110_11045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_condition_l110_11078

/-- A sequence of positive real numbers -/
def Sequence : Type := ℕ → ℝ

/-- Definition of the sequence a_n -/
def a (b c k : ℝ) : Sequence :=
  fun n => match n with
  | 0 => b  -- Add this case to handle n = 0
  | 1 => b
  | 2 => c
  | n + 3 => k * (a b c k (n + 1)) * (a b c k (n + 2))

/-- Proposition: The sequence a_n is a geometric progression iff b = c -/
theorem geometric_progression_condition (b c k : ℝ) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (∃ r : ℝ, ∀ n : ℕ, a b c k (n + 1) = r * a b c k n) ↔ b = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_condition_l110_11078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_tail_three_tosses_fair_coin_l110_11059

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ → Prop) : Prop :=
  p 0.5 ∧ ∀ x, p x ↔ x = 0.5

/-- The probability of getting exactly one tails in three tosses of a fair coin. -/
noncomputable def prob_one_tail_in_three_tosses (p : ℝ → Prop) : ℝ :=
  3 / 8

/-- Theorem stating that the probability of getting exactly one tails in three tosses of a fair coin is 3/8. -/
theorem prob_one_tail_three_tosses_fair_coin (p : ℝ → Prop) :
  fair_coin p → prob_one_tail_in_three_tosses p = 3 / 8 :=
by
  sorry

#check prob_one_tail_three_tosses_fair_coin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_tail_three_tosses_fair_coin_l110_11059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l110_11014

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem unique_solution_exists :
  ∃! m : ℤ, m - floor (m / 2005 : ℝ) = 2005 ∧ m = 2006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l110_11014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2007_equals_4_l110_11065

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 5
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_2007_equals_4 : sequence_a 2007 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2007_equals_4_l110_11065
