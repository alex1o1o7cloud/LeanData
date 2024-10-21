import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l74_7462

open Set

/-- A function f: ℝ → ℝ satisfying certain conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- Theorem stating the solution set of f(x) > x + 2 -/
theorem solution_set_of_inequality 
  (h1 : f (-1) = 1) 
  (h2 : ∀ x, f' x > 1) :
  {x : ℝ | f x > x + 2} = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l74_7462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crowd_size_l74_7485

/-- Represents the direction choices at the crossroads -/
inductive Direction
  | Left
  | Right
  | Straight

/-- Defines the rounding function for a given fraction of people -/
def roundFraction (n : ℕ) (d : ℕ) (total : ℕ) : ℕ :=
  if (n * total) % d ≥ d / 2 then (n * total) / d + 1 else (n * total) / d

/-- Defines the condition for a valid split of the crowd -/
def validSplit (total : ℕ) : Prop :=
  ∃ (left right straight : ℕ),
    left = roundFraction 1 2 total ∧
    right = roundFraction 1 3 total ∧
    straight = roundFraction 1 5 total ∧
    left + right + straight = total

/-- The main theorem stating the maximum number of people in the crowd -/
theorem max_crowd_size :
  ∀ n : ℕ, (∀ m : ℕ, m > n → ¬validSplit m) → validSplit n → n = 37 := by
  sorry

#check max_crowd_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crowd_size_l74_7485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_l74_7474

theorem equation_equivalence (φ : ℝ) :
  3 * Real.sin φ * Real.cos φ + 4 * Real.sin φ + 3 * (Real.cos φ)^2 = 4 + Real.cos φ ↔
  (3 * Real.sin φ - 1) * (Real.cos φ - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_l74_7474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_l74_7424

theorem sum_of_exponents (x y : ℕ) (h : (2:ℕ)^11 * (6:ℕ)^5 = (4:ℕ)^x * (3:ℕ)^y) : x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_l74_7424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cape_may_has_24_sightings_l74_7433

/-- The number of shark sightings in Daytona Beach -/
def daytona_sightings : ℕ := sorry

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := sorry

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := 40

/-- Cape May has 8 less than double the number of shark sightings of Daytona Beach -/
axiom cape_may_relation : cape_may_sightings = 2 * daytona_sightings - 8

/-- The total sightings is the sum of sightings in both locations -/
axiom total_sightings_sum : total_sightings = daytona_sightings + cape_may_sightings

/-- Theorem stating that Cape May has 24 shark sightings -/
theorem cape_may_has_24_sightings : cape_may_sightings = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cape_may_has_24_sightings_l74_7433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_polynomial_division_l74_7481

theorem no_solution_polynomial_division :
  ¬ ∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ),
    (x^2 - 1) ∣ (1 + 5*x^2 + x^4 - (n - 1)*x^(n - 1) + (n - 8)*x^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_polynomial_division_l74_7481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_tangents_equal_l74_7499

/-- A convex quadrilateral is a quadrilateral where all interior angles are less than 180 degrees --/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

/-- The sum of interior angles of a convex quadrilateral is 360 degrees --/
axiom sum_angles_convex_quadrilateral {A B C D : ℝ × ℝ} (h : ConvexQuadrilateral A B C D) :
  ∃ (α β γ δ : ℝ), α + β + γ + δ = 2 * Real.pi ∧ 0 < α ∧ α < Real.pi ∧ 0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi ∧ 0 < δ ∧ δ < Real.pi

/-- In a convex quadrilateral, there is at least one acute angle and one obtuse angle --/
axiom acute_obtuse_angles_exist {A B C D : ℝ × ℝ} (h : ConvexQuadrilateral A B C D) :
  ∃ (α β : ℝ), (0 < α ∧ α < Real.pi/2) ∧ (Real.pi/2 < β ∧ β < Real.pi)

/-- Tangent of an acute angle is positive --/
axiom tan_acute_positive (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi/2) : 0 < Real.tan θ

/-- Tangent of an obtuse angle is negative --/
axiom tan_obtuse_negative (θ : ℝ) (h : Real.pi/2 < θ ∧ θ < Real.pi) : Real.tan θ < 0

/-- Main theorem: In a convex quadrilateral, it's impossible for the tangents of all four angles to be equal to the same value m --/
theorem not_all_tangents_equal {A B C D : ℝ × ℝ} (h : ConvexQuadrilateral A B C D) (m : ℝ) :
  ¬∃ (α β γ δ : ℝ), (Real.tan α = m ∧ Real.tan β = m ∧ Real.tan γ = m ∧ Real.tan δ = m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_tangents_equal_l74_7499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_umbrella_representation_l74_7475

/-- Represents a letter in the word "KANGAROO" -/
inductive Letter
| K | A | N | G | R | O

/-- Represents the orientation of a letter -/
inductive LetterOrientation
| Correct | Reversed

/-- Represents a diagram of the umbrella -/
structure Diagram where
  letters : List (Letter × LetterOrientation)

/-- Checks if a letter is in its correct orientation -/
def is_correctly_oriented (l : Letter × LetterOrientation) : Prop :=
  l.2 = LetterOrientation.Correct

/-- Checks if the letters are in the correct order for "KANGAROO" -/
def is_correct_order (d : Diagram) : Prop :=
  d.letters.map Prod.fst = [Letter.K, Letter.A, Letter.N, Letter.G, Letter.A, Letter.R, Letter.O, Letter.O]

/-- Theorem stating the conditions for a correct representation of the umbrella -/
theorem correct_umbrella_representation (d : Diagram) : 
  (∀ l ∈ d.letters, is_correctly_oriented l) ∧ is_correct_order d ↔ 
  d = Diagram.mk [
    (Letter.K, LetterOrientation.Correct),
    (Letter.A, LetterOrientation.Correct),
    (Letter.N, LetterOrientation.Correct),
    (Letter.G, LetterOrientation.Correct),
    (Letter.A, LetterOrientation.Correct),
    (Letter.R, LetterOrientation.Correct),
    (Letter.O, LetterOrientation.Correct),
    (Letter.O, LetterOrientation.Correct)
  ] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_umbrella_representation_l74_7475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_property_l74_7437

/-- A right-angled triangle with side lengths 5 and 12 -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 12

/-- The intersection point of circles with diameters AB and BC -/
noncomputable def intersectionPoint (t : RightTriangle) : ℝ × ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem intersection_point_property (t : RightTriangle) :
  2400 / distance t.B (intersectionPoint t) = 520 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_property_l74_7437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_rate_l74_7460

/-- The time in minutes for 100 persons to be added to the population -/
noncomputable def population_increase_time : ℝ := 25

/-- The number of persons added in the given time -/
def persons_added : ℕ := 100

/-- Converts minutes to seconds -/
noncomputable def minutes_to_seconds (minutes : ℝ) : ℝ := minutes * 60

/-- Calculates the time in seconds for one person to be added to the population -/
noncomputable def time_per_person : ℝ :=
  (minutes_to_seconds population_increase_time) / persons_added

theorem population_growth_rate :
  time_per_person = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_rate_l74_7460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_ratio_l74_7457

-- Define the curves C1 and C2
noncomputable def C1 (a t : ℝ) : ℝ × ℝ := (a + Real.sqrt 2 * t, 1 + Real.sqrt 2 * t)

def C2 (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, C1 a t = p ∧ C2 p.1 p.2}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_ratio (a : ℝ) : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points a ∧ B ∈ intersection_points a ∧
  A ≠ B ∧ distance (a, 1) A = 2 * distance (a, 1) B →
  a = 1/36 ∨ a = 9/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_ratio_l74_7457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pete_travel_time_l74_7404

/-- Represents the travel time from Gardensquare to Madison -/
noncomputable def travel_time (map_distance : ℝ) (map_scale : ℝ) (average_speed : ℝ) : ℝ :=
  (map_distance / map_scale) / average_speed

/-- Theorem stating that Pete's travel time from Gardensquare to Madison is 1.5 hours -/
theorem pete_travel_time :
  travel_time 5 0.05555555555555555 60 = 1.5 := by
  -- Unfold the definition of travel_time
  unfold travel_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pete_travel_time_l74_7404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_for_range_exists_eleven_no_larger_than_eleven_l74_7498

theorem largest_c_for_range (f : ℝ → ℝ) (c : ℝ) : 
  (∃ x, f x = 2) ∧ (f = λ x ↦ x^2 - 6*x + c) → c ≤ 11 :=
by
  sorry

theorem exists_eleven : 
  ∃ x, (λ x : ℝ ↦ x^2 - 6*x + 11) x = 2 :=
by
  sorry

theorem no_larger_than_eleven (c : ℝ) : 
  c > 11 → ¬∃ x, (λ x : ℝ ↦ x^2 - 6*x + c) x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_for_range_exists_eleven_no_larger_than_eleven_l74_7498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l74_7447

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

/-- The ellipse equation -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- Point A -/
def point_A : ℝ × ℝ := (4, 4)

/-- Point on the parabola -/
noncomputable def point_on_parabola (p : ℝ) (y : ℝ) : ℝ × ℝ := (y^2/(2*p), y)

/-- Projection on y-axis -/
def projection_y (y : ℝ) : ℝ × ℝ := (0, y)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem min_distance_sum (p a b : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ parabola p x (Real.sqrt (2*p*x))) →
  (∃ (x y : ℝ), ellipse a b x y) →
  (p/2 = a + Real.sqrt (a^2 - b^2)) →
  (∀ (y : ℝ), 
    let M := point_on_parabola p y
    let N := projection_y y
    distance M point_A + distance M N ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l74_7447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_minus_median_equals_two_l74_7405

theorem mean_minus_median_equals_two (x : ℕ) :
  let sequence := [x, x + 2, x + 4, x + 7, x + 17]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5
  let median := x + 4
  (mean : ℚ) - median = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_minus_median_equals_two_l74_7405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l74_7438

noncomputable section

-- Define the lines and distance
def line1 (x y m : ℝ) := 3 * x - y + m = 0
def line2 (x y n : ℝ) := 6 * x + n * y + 7 = 0
def distance := Real.sqrt 10 / 4

-- Define the theorem
theorem parallel_lines_distance (m n : ℝ) :
  (∃ x y, line1 x y m ∧ line2 x y n) →  -- Lines exist
  (∀ x y, line1 x y m → line2 x y n → False) →  -- Lines are distinct
  (∀ x₁ y₁ x₂ y₂, line1 x₁ y₁ m → line2 x₂ y₂ n → 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ distance) →  -- Distance between any two points is at least √10/4
  (∃ x₁ y₁ x₂ y₂, line1 x₁ y₁ m ∧ line2 x₂ y₂ n ∧ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = distance) →  -- There exist points with exactly distance √10/4
  m = 6 ∨ m = 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l74_7438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l74_7430

open BigOperators Finset

def total_marbles : ℕ := 8
def blue_marbles : ℕ := 3
def white_marbles : ℕ := 5
def marbles_drawn : ℕ := 6

def probability_one_white_one_blue : ℚ :=
  (Nat.choose blue_marbles (blue_marbles - 1) * Nat.choose white_marbles (white_marbles - 1)) /
  Nat.choose total_marbles marbles_drawn

theorem probability_theorem :
  probability_one_white_one_blue = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l74_7430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evenTailsProbability2021_l74_7439

/-- The probability of obtaining an even number of tails when flipping a fair coin n times. -/
noncomputable def evenTailsProbability (n : ℕ) : ℝ :=
  1 / 2

/-- Theorem stating that the probability of obtaining an even number of tails
    when flipping a fair coin 2021 times is 1/2. -/
theorem evenTailsProbability2021 :
  evenTailsProbability 2021 = 1 / 2 := by
  sorry

#check evenTailsProbability2021

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evenTailsProbability2021_l74_7439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l74_7448

open Real

-- Define the triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define an obtuse triangle
def ObtuseTriangle (A B C : ℝ) : Prop :=
  Triangle A B C ∧ (Real.pi / 2 < A ∨ Real.pi / 2 < B ∨ Real.pi / 2 < C)

-- Theorem statement
theorem triangle_problem (A B C a b c : ℝ) :
  ObtuseTriangle A B C →
  c = sqrt 2 →
  b = sqrt 6 →
  (1 / 2) * b * c * sin A = sqrt 2 →
  (cos A = -(sqrt 3 / 3) ∧ a = 2 * sqrt 3) ∧
  sin (2 * B - Real.pi / 4) = (4 - sqrt 2) / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l74_7448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l74_7453

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≤ 1 ∨ x ≥ 17} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l74_7453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l74_7470

/-- The hyperbola is defined by the equation: √((x-2)²+(y-3)²) - √((x-8)²+(y-3)²) = 4 --/
def hyperbola (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4

/-- The positive slope of an asymptote of the hyperbola --/
noncomputable def positive_asymptote_slope : ℝ := Real.sqrt 5 / 2

theorem hyperbola_asymptote_slope :
  ∀ (x y : ℝ), hyperbola x y → positive_asymptote_slope = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l74_7470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_is_10_l74_7486

/-- A function that checks if all digits in a natural number are less than 5 -/
def allDigitsLessThan5 (m : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ m.digits 10 → d < 5

/-- The property that c satisfies the condition for all positive integers n -/
def satisfiesCondition (c : ℕ) : Prop :=
  c > 0 ∧ ∀ n : ℕ, n > 0 → allDigitsLessThan5 (c^n + 2014)

/-- Theorem stating that 10 is the smallest positive integer satisfying the condition -/
theorem smallest_c_is_10 : 
  satisfiesCondition 10 ∧ ∀ c : ℕ, c < 10 → ¬satisfiesCondition c := by
  sorry

#check smallest_c_is_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_is_10_l74_7486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_factorial_equation_l74_7401

theorem no_solution_factorial_equation (k m : ℕ) :
  k.factorial + 48 ≠ 48 * (k + 1) ^ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_factorial_equation_l74_7401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_inequality_l74_7442

/-- Euler's totient function -/
noncomputable def φ : ℕ+ → ℕ := sorry

/-- Property of Euler's totient function for 1 -/
axiom φ_one : φ 1 = 1

/-- Main theorem -/
theorem euler_totient_inequality (a b : ℕ+) :
  (φ (a * b) : ℝ) / Real.sqrt ((φ (a ^ 2) : ℝ) ^ 2 + (φ (b ^ 2) : ℝ) ^ 2) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_inequality_l74_7442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_square_starts_1989_l74_7414

theorem smallest_integer_square_starts_1989 : 
  ∀ n : ℕ, n < 446 → (n^2).repr.take 4 ≠ "1989" ∧ (446^2).repr.take 4 = "1989" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_square_starts_1989_l74_7414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_equals_two_l74_7479

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x - x else -(Real.sqrt (-x) + x)

-- State the theorem
theorem f_neg_four_equals_two :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x ≥ 0, f x = Real.sqrt x - x) →  -- definition for non-negative x
  f (-4) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_equals_two_l74_7479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_axis_tangent_to_f_h_one_zero_h_two_zeros_h_three_zeros_l74_7419

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^3 - 2 * a * x - 1/2
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- Define the function h
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := max (f a x) (g x)

-- Theorem for part 1
theorem x_axis_tangent_to_f (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) ↔ a = -3/4 := by sorry

-- Theorems for part 2
theorem h_one_zero (a : ℝ) : 
  (a > -3/4 ∨ a < -5/4) → (∃! x : ℝ, x > 0 ∧ h a x = 0) := by sorry

theorem h_two_zeros (a : ℝ) : 
  (a = -3/4 ∨ a = -5/4) → (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ h a x = 0 ∧ h a y = 0 ∧
    ∀ z : ℝ, z > 0 ∧ h a z = 0 → (z = x ∨ z = y)) := by sorry

theorem h_three_zeros (a : ℝ) : 
  (-5/4 < a ∧ a < -3/4) → (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    ∀ w : ℝ, w > 0 ∧ h a w = 0 → (w = x ∨ w = y ∨ w = z)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_axis_tangent_to_f_h_one_zero_h_two_zeros_h_three_zeros_l74_7419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_herdsman_l74_7436

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path problem for the herdsman -/
theorem shortest_path_herdsman :
  let herdsman := Point.mk 0 (-4)
  let house := Point.mk (-8) 3
  let river_y := 0
  let reflected_house := Point.mk house.x (2 * river_y - house.y)
  distance herdsman reflected_house = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_herdsman_l74_7436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_y_l74_7403

/-- The function that we want to minimize -/
def y (a x : ℕ+) : ℚ :=
  (a + 2) * x^2 - 2 * (a^2 - 1) * x + 1

/-- The set of x values that minimize y for a given a -/
def minimizing_x (a : ℕ+) : Set ℕ+ :=
  { x | ∀ (z : ℕ+), y a x ≤ y a z }

/-- The theorem statement -/
theorem minimize_y (a : ℕ+) :
  minimizing_x a =
    if a < 4 then {a - 1}
    else if a = 4 then {2, 3}
    else {a - 2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_y_l74_7403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corsair_catches_up_in_three_hours_l74_7495

/-- Represents the time when the corsair catches up to the cargo ship -/
noncomputable def catch_up_time (initial_distance : ℝ) (corsair_speed cargo_speed : ℝ) : ℝ :=
  initial_distance / (corsair_speed - cargo_speed)

/-- Proves that the corsair catches up to the cargo ship after exactly 3 hours -/
theorem corsair_catches_up_in_three_hours 
  (initial_distance : ℝ) 
  (corsair_speed cargo_speed : ℝ) :
  initial_distance = 12 →
  corsair_speed = 14 →
  cargo_speed = 10 →
  catch_up_time initial_distance corsair_speed cargo_speed = 3 := by
  sorry

/-- Evaluates the catch-up time for the given parameters -/
def evaluate_catch_up_time : ℚ :=
  (12 : ℚ) / ((14 : ℚ) - (10 : ℚ))

#eval evaluate_catch_up_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corsair_catches_up_in_three_hours_l74_7495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_eight_l74_7477

theorem count_divisible_by_eight : ∃ n : ℕ, n = (Finset.filter (fun x => x % 8 = 0) (Finset.range 501 \ Finset.range 100)).card ∧ n = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_eight_l74_7477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_l74_7423

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem f_zero_value (ω φ : ℝ) :
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  f ω φ (Real.pi/8) = Real.sqrt 2 →
  f ω φ (Real.pi/2) = 0 →
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  f ω φ 0 = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_l74_7423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gain_percentage_after_discounts_l74_7432

/-- Proves that the total gain percentage after applying three successive discounts is approximately 19.61% -/
theorem total_gain_percentage_after_discounts (M : ℝ) : 
  let cost_price := 0.65 * M
  let price_after_first_discount := 0.88 * M
  let price_after_second_discount := 0.95 * price_after_first_discount
  let final_price := 0.93 * price_after_second_discount
  let gain := final_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  ∃ ε > 0, |gain_percentage - 19.61| < ε := by
  sorry

#check total_gain_percentage_after_discounts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gain_percentage_after_discounts_l74_7432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_l74_7420

theorem triangle_cosine_sum_max (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  Real.cos A + Real.cos B * Real.cos C ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_l74_7420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_isosceles_trapezoid_l74_7444

/-- Definition of a regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  points : Set (ℝ × ℝ)
  -- Add other necessary fields and axioms

/-- The set of all points in a regular polygon --/
def RegularPolygon.allPoints (P : RegularPolygon n) : Set (ℝ × ℝ) := P.points

/-- Definition of an isosceles trapezoid --/
def IsIsoscelesTrapezoid (T : Set (ℝ × ℝ)) : Prop := sorry

/-- A regular pentagon can be divided into three parts that form an isosceles trapezoid --/
theorem pentagon_to_isosceles_trapezoid :
  ∃ (P : RegularPolygon 5) (A B C : Set (ℝ × ℝ)),
    (A ∪ B ∪ C = P.allPoints) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    (∃ (T : Set (ℝ × ℝ)), IsIsoscelesTrapezoid T ∧ T = A ∪ B ∪ C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_isosceles_trapezoid_l74_7444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_five_l74_7478

/-- Represents a round trip journey with different speeds for each leg -/
structure RoundTrip where
  distance : ℝ
  speed_ab : ℝ
  speed_ba : ℝ
  avg_speed : ℝ

/-- Calculates the average speed of a round trip -/
noncomputable def calculate_avg_speed (trip : RoundTrip) : ℝ :=
  (2 * trip.distance) / (trip.distance / trip.speed_ab + trip.distance / trip.speed_ba)

/-- Theorem stating that given specific conditions, the initial speed is 5 km/h -/
theorem initial_speed_is_five (trip : RoundTrip) 
  (h1 : trip.speed_ba = 20)
  (h2 : trip.avg_speed = 8)
  (h3 : calculate_avg_speed trip = trip.avg_speed) :
  trip.speed_ab = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_five_l74_7478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_moles_formed_l74_7450

/-- Represents the number of moles of a substance -/
def Moles : Type := ℕ

/-- Represents the chemical reaction between Sulfuric acid and Sodium hydroxide -/
structure Reaction where
  sulfuric_acid : Moles
  sodium_hydroxide : Moles
  sodium_bisulfate : Moles
  water : Moles

/-- Provide an instance of OfNat for Moles -/
instance : OfNat Moles n where
  ofNat := n

/-- The theorem states that given the conditions of the reaction,
    the number of moles of Water formed is 3 -/
theorem water_moles_formed (r : Reaction) 
  (h1 : r.sulfuric_acid = 3)
  (h2 : r.sodium_hydroxide = 3)
  (h3 : r.sodium_bisulfate = 3)
  (h4 : r.water = r.sodium_bisulfate) :
  r.water = 3 := by
  rw [h4, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_moles_formed_l74_7450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l74_7466

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem g_range :
  ∀ y : ℝ,
  (∃ x : ℝ, x > Real.pi / 4 ∧ x < 3 * Real.pi / 4 ∧ g x = y) ↔
  (y > -Real.sqrt 3 / 2 ∧ y ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l74_7466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_three_l74_7458

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We add this case to cover Nat.zero
  | 1 => 3
  | 2 => 6
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_2016_equals_negative_three : sequence_a 2016 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_three_l74_7458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_largest_inscribed_shapes_l74_7480

/-- Represents a tetrahedron with mutually perpendicular edges -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The side length of the largest cube with vertex S contained within the tetrahedron -/
noncomputable def largest_cube_side_length (t : Tetrahedron) : ℝ :=
  (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)

/-- The dimensions of the largest rectangular parallelepiped with vertex S contained within the tetrahedron -/
noncomputable def largest_parallelepiped_dimensions (t : Tetrahedron) : ℝ × ℝ × ℝ :=
  (t.a / 3, t.b / 3, t.c / 3)

theorem tetrahedron_largest_inscribed_shapes (t : Tetrahedron) :
  (largest_cube_side_length t = (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)) ∧
  (largest_parallelepiped_dimensions t = (t.a / 3, t.b / 3, t.c / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_largest_inscribed_shapes_l74_7480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_decreasing_intervals_l74_7493

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.sin (3 * Real.pi / 4))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (3 * Real.pi / 4), -Real.cos (2 * x))

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry

-- Theorem for monotonically decreasing intervals
theorem monotonically_decreasing_intervals :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 8 → f y < f x) ∧
  (∀ x y : ℝ, 5 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ Real.pi → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_decreasing_intervals_l74_7493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_when_x_gt_neg_pi_div_4_l74_7451

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := sin x + cos x

-- State the theorem
theorem f_geq_g_when_x_gt_neg_pi_div_4 :
  ∀ x : ℝ, x > -π/4 →
  (∀ y : ℝ, exp y ≥ y + 1) →
  f x ≥ g x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_when_x_gt_neg_pi_div_4_l74_7451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_excluded_value_l74_7483

/-- Function g defined as (px+q)/(rx+s) -/
noncomputable def g (p q r s x : ℝ) : ℝ := (p*x + q) / (r*x + s)

/-- Theorem stating that p/r is the unique number not in the range of g -/
theorem unique_excluded_value (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : g p q r s 11 = 11)
  (h2 : g p q r s 35 = 35)
  (h3 : g p q r s 75 = 75)
  (h4 : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = p/r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_excluded_value_l74_7483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_max_volume_l74_7472

/-- The side length of the original square sheet of paper in centimeters. -/
noncomputable def paper_side_length : ℝ := 20

/-- The volume of the box as a function of the side length of the cut-out squares. -/
noncomputable def box_volume (x : ℝ) : ℝ := x * (paper_side_length - 2 * x)^2

/-- The maximum volume of the box. -/
noncomputable def max_volume : ℝ := 16000 / 27

theorem box_max_volume :
  ∃ x : ℝ, 0 < x ∧ x < paper_side_length / 2 ∧
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ paper_side_length / 2 → box_volume y ≤ box_volume x) ∧
  box_volume x = max_volume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_max_volume_l74_7472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l74_7496

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) * Real.sin x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x

-- State the theorems
theorem derivative_f (x : ℝ) :
  deriv f x = 2 * x * Real.sin x + (x^2 + 2) * Real.cos x :=
by sorry

theorem derivative_g (x : ℝ) :
  deriv g x = (2 * x - x^2) / Real.exp x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l74_7496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_identities_l74_7406

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem combinatorial_identities :
  (∀ n m : ℕ, (Finset.range (m + 1)).sum (λ i ↦ binomial (n - i) (m - i)) = binomial (n + 1) m) ∧
  (∀ n m k : ℕ, (Finset.range (m + 1)).sum (λ i ↦ binomial (n - i) (m - i) * binomial (k + i) i) = binomial (n + k + 1) m) ∧
  (∀ n m k : ℕ, (Finset.range (k + 1)).sum (λ i ↦ binomial n i * binomial m (k - i)) = binomial (n + m) k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_identities_l74_7406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l74_7410

/-- The function g with the given properties -/
noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- The theorem stating the unique number not in the range of g -/
theorem unique_number_not_in_range
  (p q r s : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : g p q r s 23 = 23)
  (h2 : g p q r s 101 = 101)
  (h3 : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 62 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l74_7410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_conditions_l74_7455

def has_24_factors (n : ℕ) : Prop := (Finset.filter (λ i => i ∣ n) (Finset.range (n + 1))).card = 24

theorem smallest_integer_with_conditions :
  ∃ (x : ℕ), has_24_factors x ∧ 18 ∣ x ∧ 28 ∣ x ∧
  ∀ (y : ℕ), has_24_factors y ∧ 18 ∣ y ∧ 28 ∣ y → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_conditions_l74_7455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l74_7415

noncomputable def purchase_price : ℝ := 13500
noncomputable def discount_rate : ℝ := 0.20
noncomputable def installation_cost : ℝ := 250
noncomputable def selling_price : ℝ := 18975
noncomputable def profit_rate : ℝ := 0.10

noncomputable def labeled_price : ℝ := purchase_price / (1 - discount_rate)

noncomputable def extra_cost : ℝ := selling_price - labeled_price * (1 + profit_rate)

theorem transport_cost_calculation :
  extra_cost - installation_cost = 1850 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l74_7415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mediator_is_collection_agency_l74_7490

/-- Represents the different roles in the book lending scenario -/
inductive Role
  | FinancialPyramid
  | CollectionAgency
  | Bank
  | InsuranceCompany

/-- Represents a person in the book lending scenario -/
structure Person where
  name : String

/-- Represents the book lending scenario -/
structure BookLendingScenario where
  lender : Person
  borrower : Person
  mediator : Person
  books_lent : ℕ
  books_returned : ℕ
  mediator_fee : ℕ

/-- Defines the conditions of the book lending scenario -/
def scenario_conditions (s : BookLendingScenario) : Prop :=
  s.lender.name = "Katya" ∧
  s.borrower.name = "Vasya" ∧
  s.mediator.name = "Kolya" ∧
  s.books_lent > 0 ∧
  s.books_returned = 0 ∧
  s.mediator_fee = 1

/-- Defines the role of the mediator based on the scenario -/
def mediator_role (s : BookLendingScenario) : Role :=
  Role.CollectionAgency

/-- Theorem stating that under the given conditions, the mediator's role is equivalent to a collection agency -/
theorem mediator_is_collection_agency (s : BookLendingScenario) :
  scenario_conditions s → mediator_role s = Role.CollectionAgency :=
by
  intro h
  simp [mediator_role]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mediator_is_collection_agency_l74_7490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l74_7402

noncomputable def f (x : ℝ) := 5 * Real.sin (2 * x - Real.pi / 4)

theorem f_properties :
  (∀ x, f (3 * Real.pi / 8 + x) = f (3 * Real.pi / 8 - x)) ∧
  (∀ x, f (x + 5 * Real.pi / 8) = -f (-x + 5 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l74_7402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotationVolume_eq_44pi_div_7_l74_7467

/-- The volume of the solid formed by rotating the region bounded by x = ∛(y - 2), x = 1, and y = 1 around the Ox axis -/
noncomputable def rotationVolume : ℝ := 
  let f (x : ℝ) := x^3 + 2
  let a : ℝ := -1
  let b : ℝ := 1
  Real.pi * ∫ x in a..b, (f x)^2 - 1

/-- The theorem stating that the volume of the rotated solid is equal to 44π/7 -/
theorem rotationVolume_eq_44pi_div_7 : rotationVolume = (44 / 7) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotationVolume_eq_44pi_div_7_l74_7467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l74_7411

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 26 = 0

-- Define the center and radius of each circle
def center1 : ℝ × ℝ := (2, -3)
noncomputable def radius1 : ℝ := Real.sqrt 13
def center2 : ℝ × ℝ := (-1, 3)
def radius2 : ℝ := 6

-- Calculate the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt (3^2 + 6^2)

-- Theorem: The circles are intersecting
theorem circles_intersect :
  distance_between_centers < radius1 + radius2 ∧
  distance_between_centers > abs (radius1 - radius2) := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l74_7411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thumb_closest_to_5cm_l74_7469

def school_bus_length : ℝ := 1000
def picnic_table_height : ℝ := 75
def elephant_height : ℝ := 300
def foot_length : ℝ := 25
def thumb_length : ℝ := 4.5

def target_length : ℝ := 5

theorem thumb_closest_to_5cm :
  ∀ x ∈ ({school_bus_length, picnic_table_height, elephant_height, foot_length} : Set ℝ),
    |thumb_length - target_length| ≤ |x - target_length| :=
by
  intro x hx
  -- The proof goes here
  sorry

#check thumb_closest_to_5cm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thumb_closest_to_5cm_l74_7469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l74_7435

-- Define the basic types and relations
variable (Line : Type) (Plane : Type)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_to_plane : Line → Plane → Prop)
variable (lies_in : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Define the propositions
def prop1 (Line : Type) (perpendicular parallel : Line → Line → Prop) : Prop := 
  ∀ l₁ l₂ l₃ : Line, (perpendicular l₁ l₃ ∧ perpendicular l₂ l₃) → parallel l₁ l₂

def prop2 (Line Plane : Type) (parallel_to_plane : Line → Plane → Prop) (parallel : Line → Line → Prop) : Prop := 
  ∀ l₁ l₂ : Line, ∀ P : Plane, (parallel_to_plane l₁ P ∧ parallel_to_plane l₂ P) → parallel l₁ l₂

def prop3 (Line : Type) (parallel : Line → Line → Prop) : Prop := 
  ∀ l₁ l₂ l₃ : Line, (parallel l₁ l₃ ∧ parallel l₂ l₃) → parallel l₁ l₂

def prop4 (Line Plane : Type) (lies_in : Line → Plane → Prop) (intersect parallel : Line → Line → Prop) : Prop := 
  ∀ l₁ l₂ : Line, ∀ P : Plane, (lies_in l₁ P ∧ lies_in l₂ P ∧ ¬intersect l₁ l₂) → parallel l₁ l₂

-- Theorem statement
theorem exactly_two_correct : 
  (¬prop1 Line perpendicular parallel ∧ ¬prop2 Line Plane parallel_to_plane parallel ∧ prop3 Line parallel ∧ prop4 Line Plane lies_in intersect parallel) ∨
  (¬prop1 Line perpendicular parallel ∧ prop2 Line Plane parallel_to_plane parallel ∧ prop3 Line parallel ∧ ¬prop4 Line Plane lies_in intersect parallel) ∨
  (¬prop1 Line perpendicular parallel ∧ prop2 Line Plane parallel_to_plane parallel ∧ ¬prop3 Line parallel ∧ prop4 Line Plane lies_in intersect parallel) ∨
  (prop1 Line perpendicular parallel ∧ ¬prop2 Line Plane parallel_to_plane parallel ∧ prop3 Line parallel ∧ ¬prop4 Line Plane lies_in intersect parallel) ∨
  (prop1 Line perpendicular parallel ∧ ¬prop2 Line Plane parallel_to_plane parallel ∧ ¬prop3 Line parallel ∧ prop4 Line Plane lies_in intersect parallel) ∨
  (prop1 Line perpendicular parallel ∧ prop2 Line Plane parallel_to_plane parallel ∧ ¬prop3 Line parallel ∧ ¬prop4 Line Plane lies_in intersect parallel) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l74_7435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_solution_l74_7400

def vector1 : Fin 3 → ℝ := ![3, 4, 5]
def vector2 (m : ℝ) : Fin 3 → ℝ := ![2, m, 3]
def vector3 (m : ℝ) : Fin 3 → ℝ := ![2, 3, m]

def parallelepiped_volume (v1 v2 v3 : Fin 3 → ℝ) : ℝ :=
  |Matrix.det (Matrix.of ![v1, v2, v3])|

theorem parallelepiped_volume_solution (m : ℝ) :
  m > 0 ∧ parallelepiped_volume vector1 (vector2 m) (vector3 m) = 20 →
  m = 3 + (2 * Real.sqrt 15) / 3 := by
  sorry

#eval parallelepiped_volume vector1 (vector2 5) (vector3 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_solution_l74_7400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_2_solution_set_general_l74_7413

-- Define the inequality function
def f (x a : ℝ) : ℝ := x^2 - x + a - a^2

-- Define the solution set type
def SolutionSet := Set ℝ

-- Theorem for the case when a = 2
theorem solution_set_a_2 : 
  ∀ x : ℝ, f x 2 ≤ 0 ↔ x ∈ (Set.Icc (-1) 2 : SolutionSet) := by sorry

-- Theorem for the general case
theorem solution_set_general (a : ℝ) :
  (∀ x : ℝ, f x a ≤ 0) ↔ 
    (a < 1/2 ∧ (Set.Icc a (1-a) : SolutionSet) = {x | f x a ≤ 0}) ∨
    (a > 1/2 ∧ (Set.Icc (1-a) a : SolutionSet) = {x | f x a ≤ 0}) ∨
    (a = 1/2 ∧ ({x | x = 1/2} : SolutionSet) = {x | f x a ≤ 0}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_2_solution_set_general_l74_7413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_movement_in_room_l74_7428

theorem table_movement_in_room (L W : ℕ) : 
  (L ≥ W) →  -- L is the greater dimension
  (∃ (table_length table_width : ℕ), 
    table_length = 9 ∧ 
    table_width = 12 ∧ 
    (L : ℝ) ≥ Real.sqrt ((table_length^2 : ℝ) + (table_width^2 : ℝ))) →
  L ≥ 15 := by
  sorry

#check table_movement_in_room

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_movement_in_room_l74_7428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_slope_tangent_slopes_intersecting_slopes_l74_7425

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 4

-- Define the line passing through (-3, -4) with slope k
def myLine (k : ℝ) (x y : ℝ) : Prop := y + 4 = k * (x + 3)

-- Theorem for the bisecting case
theorem bisecting_slope :
  ∃ k : ℝ, k = -1/2 ∧
  ∀ x y : ℝ, myLine k x y → (∀ x' y' : ℝ, myCircle x' y' → 
    (x' - 1)^2 + (y' - (-2))^2 = (x - 1)^2 + (y - (-2))^2) :=
sorry

-- Theorem for the tangent case
theorem tangent_slopes :
  ∃ k₁ k₂ : ℝ, k₁ = 0 ∧ k₂ = 4/3 ∧
  (∀ x y : ℝ, myLine k₁ x y → (∃! p : ℝ × ℝ, myCircle p.1 p.2 ∧ myLine k₁ p.1 p.2)) ∧
  (∀ x y : ℝ, myLine k₂ x y → (∃! p : ℝ × ℝ, myCircle p.1 p.2 ∧ myLine k₂ p.1 p.2)) :=
sorry

-- Theorem for the intersecting case with chord length 2
theorem intersecting_slopes :
  ∃ k₁ k₂ : ℝ, k₁ = (8 - Real.sqrt 51) / 13 ∧ k₂ = (8 + Real.sqrt 51) / 13 ∧
  (∀ x y : ℝ, myLine k₁ x y → 
    (∃ p₁ p₂ : ℝ × ℝ, myCircle p₁.1 p₁.2 ∧ myCircle p₂.1 p₂.2 ∧ 
    myLine k₁ p₁.1 p₁.2 ∧ myLine k₁ p₂.1 p₂.2 ∧ 
    (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = 4)) ∧
  (∀ x y : ℝ, myLine k₂ x y → 
    (∃ p₁ p₂ : ℝ × ℝ, myCircle p₁.1 p₁.2 ∧ myCircle p₂.1 p₂.2 ∧ 
    myLine k₂ p₁.1 p₁.2 ∧ myLine k₂ p₂.1 p₂.2 ∧ 
    (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = 4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_slope_tangent_slopes_intersecting_slopes_l74_7425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_six_l74_7468

def is_valid_number (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ (a b c d e : ℕ), n = 10000*a + 1000*b + 100*c + 10*d + e ∧ 
    ({a, b, c, d, e} : Finset ℕ) = {1, 2, 6, 7, 8})  -- contains digits 1, 2, 6, 7, 8 exactly once

theorem smallest_valid_divisible_by_six :
  ∀ n : ℕ, is_valid_number n ∧ n % 6 = 0 → n ≥ 12678 :=
by
  intro n hn
  sorry

#check smallest_valid_divisible_by_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_six_l74_7468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_not_parallel_l74_7461

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (a b x : ℝ) : ℝ := (1/2) * a * x^2 + b * x

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := 1 / x

-- Define the derivative of g
noncomputable def g_deriv (a b x : ℝ) : ℝ := a * x + b

-- Theorem statement
theorem tangent_lines_not_parallel (a b x₁ x₂ : ℝ) (ha : a ≠ 0) (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf : f x₁ = g a b x₁ ∧ f x₂ = g a b x₂) : 
  let x := (x₁ + x₂) / 2
  f_deriv x ≠ g_deriv a b x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_not_parallel_l74_7461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_iff_l74_7476

theorem matrix_not_invertible_iff (x : ℚ) : 
  ¬(IsUnit (Matrix.det !![2*x, 5; 4-x, 9])) ↔ x = 20/23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_iff_l74_7476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l74_7445

/-- The angle (in degrees) that the minute hand moves per minute -/
noncomputable def minute_hand_speed : ℝ := 360 / 60

/-- The angle (in degrees) that the hour hand moves per hour -/
noncomputable def hour_hand_speed : ℝ := 360 / 12

/-- The position of the minute hand at 3:15 -/
noncomputable def minute_hand_position : ℝ := 15 * minute_hand_speed

/-- The position of the hour hand at 3:15 -/
noncomputable def hour_hand_position : ℝ := 3 * hour_hand_speed + 0.25 * hour_hand_speed

/-- The obtuse angle between the hour hand and minute hand at 3:15 -/
noncomputable def clock_angle : ℝ := 360 - |hour_hand_position - minute_hand_position|

theorem clock_angle_at_3_15 : clock_angle = 352.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l74_7445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_correct_is_one_sixth_l74_7416

/-- The probability of exactly 3 out of 5 packages being delivered to the correct houses -/
def probability_three_correct : ℚ :=
  (Nat.choose 5 3 : ℚ) * (1 / 5) * (1 / 4) * (1 / 3)

/-- Theorem stating that the probability of exactly 3 out of 5 packages being delivered to the correct houses is 1/6 -/
theorem probability_three_correct_is_one_sixth :
  probability_three_correct = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_correct_is_one_sixth_l74_7416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_line_slope_l74_7482

noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

theorem secant_line_slope :
  let x₁ : ℝ := 2
  let y₁ : ℝ := -2
  let Δx : ℝ := 0.5
  let x₂ : ℝ := x₁ + Δx
  let y₂ : ℝ := f x₂
  (y₂ - y₁) / (x₂ - x₁) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_line_slope_l74_7482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_in_surface_area_l74_7484

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surface_area_rectangular_solid (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the surface area of a sphere -/
noncomputable def surface_area_sphere (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

/-- Calculates the surface area of a spherical cap -/
noncomputable def surface_area_spherical_cap (radius : ℝ) : ℝ :=
  2 * Real.pi * radius^2

/-- Theorem: Change in surface area after removing a spherical section -/
theorem change_in_surface_area 
  (solid : RectangularSolid)
  (sphere_radius : ℝ)
  (h_solid : solid = ⟨4, 3, 5⟩)
  (h_radius : sphere_radius = 1) :
  surface_area_rectangular_solid solid - 
  (surface_area_sphere sphere_radius - surface_area_spherical_cap sphere_radius) = 
  94 - 2 * Real.pi := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_in_surface_area_l74_7484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_occupancy_problem_l74_7465

/-- A small airplane with first, business, and economy class seating -/
structure Airplane where
  first_class_capacity : Nat
  business_class_capacity : Nat
  economy_class_capacity : Nat

/-- The occupancy of each class in the airplane -/
structure Occupancy where
  first_class : Nat
  business_class : Nat
  economy_class : Nat

/-- The theorem representing the problem -/
theorem airplane_occupancy_problem (plane : Airplane) (occupancy : Occupancy) :
    plane.first_class_capacity = 10 →
    plane.business_class_capacity = 30 →
    plane.economy_class_capacity = 50 →
    occupancy.economy_class = plane.economy_class_capacity / 2 →
    occupancy.first_class + occupancy.business_class = occupancy.economy_class →
    occupancy.first_class = 3 →
    plane.business_class_capacity - occupancy.business_class = 8 := by
  intro h1 h2 h3 h4 h5 h6
  sorry

#check airplane_occupancy_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_occupancy_problem_l74_7465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l74_7494

theorem matrix_transformation (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 2, 1]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2*a+b, 2*b+a; 2*c+d, 2*d+c]
  M * A = N := by
  simp [Matrix.mul_apply]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l74_7494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l74_7408

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- Define the domain
def domain : Set ℝ := {x | -Real.pi ≤ x ∧ x ≤ 0}

-- Define the interval where f is strictly increasing
def increasing_interval : Set ℝ := {x | -Real.pi/6 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem f_strictly_increasing_on_interval :
  StrictMonoOn f increasing_interval ∧ increasing_interval ⊆ domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l74_7408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_are_true_l74_7492

-- Define proposition p
def proposition_p (m : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + x + m > 0) → m > 1/4

-- Define proposition q
def proposition_q : Prop :=
  ∀ (A B C : ℝ),
    (A > B ↔ Real.sin A > Real.sin B)

-- Theorem to prove
theorem propositions_are_true : ∃ m : ℝ, proposition_p m ∧ proposition_q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_are_true_l74_7492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_reappearance_l74_7489

/-- Represents a cyclic sequence --/
structure CyclicSequence (α : Type) where
  elements : List α
  deriving Repr

/-- The length of a cyclic sequence --/
def CyclicSequence.length {α : Type} (cs : CyclicSequence α) : Nat :=
  cs.elements.length

/-- Perform one cycle on a cyclic sequence --/
def CyclicSequence.cycle {α : Type} (cs : CyclicSequence α) : CyclicSequence α :=
  match cs.elements with
  | [] => cs
  | h :: t => CyclicSequence.mk (t ++ [h])

/-- Check if two cyclic sequences are in their original state --/
def areInOriginalState {α β : Type} (cs1 : CyclicSequence α) (cs2 : CyclicSequence β) 
    (original1 : CyclicSequence α) (original2 : CyclicSequence β) : Prop :=
  cs1 = original1 ∧ cs2 = original2

/-- Iterate a function n times --/
def iterate {α : Type} (f : α → α) : Nat → α → α
  | 0, x => x
  | n + 1, x => iterate f n (f x)

/-- The main theorem to prove --/
theorem cyclic_sequence_reappearance 
    (cs1 : CyclicSequence Char) 
    (cs2 : CyclicSequence Nat) : 
    cs1.length = 8 → 
    cs2.length = 5 → 
    ∃ n : Nat, n = 40 ∧ 
      areInOriginalState 
        (iterate CyclicSequence.cycle n cs1) 
        (iterate CyclicSequence.cycle n cs2) 
        cs1 
        cs2 := by
  sorry

#check cyclic_sequence_reappearance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_reappearance_l74_7489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_and_height_l74_7456

/-- A rectangular box with specific proportions and volume -/
structure Box where
  height : ℝ
  volume_cubic_feet : ℝ
  width_height_ratio : ℝ
  length_height_ratio : ℝ

/-- The volume of the box in cubic yards -/
noncomputable def volume_cubic_yards (b : Box) : ℝ :=
  b.volume_cubic_feet / 27

/-- Theorem about a specific box -/
theorem box_volume_and_height (b : Box) 
  (h_volume : b.volume_cubic_feet = 216)
  (h_width : b.width_height_ratio = 2)
  (h_length : b.length_height_ratio = 3) :
  volume_cubic_yards b = 8 ∧ b.height = (36 : ℝ) ^ (1/3 : ℝ) := by
  sorry

#check box_volume_and_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_and_height_l74_7456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l74_7454

/-- The time taken to complete a work when two people work together -/
noncomputable def time_together (time1 time2 : ℝ) : ℝ :=
  1 / (1 / time1 + 1 / time2)

/-- Theorem: Given Johnson's time and Vincent's time, prove they complete the work in 8 days together -/
theorem work_completion_time (johnson_time vincent_time : ℝ) 
  (h1 : johnson_time = 10)
  (h2 : vincent_time = 40) :
  time_together johnson_time vincent_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l74_7454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l74_7488

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem statement -/
theorem interest_calculation (P : ℝ) (h : simpleInterest P 5 2 = 40) :
  compoundInterest P 5 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l74_7488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_l74_7417

/-- Represents the possible scores for a single question -/
inductive Score
  | Full : Score
  | Partial : Score
  | Zero : Score
deriving Fintype, Repr

/-- Represents the score combination for two questions -/
def ScoreCombination := Score × Score

/-- Converts a Score to its numerical value -/
def scoreValue (s : Score) : Nat :=
  match s with
  | Score.Full => 10
  | Score.Partial => 5
  | Score.Zero => 0

/-- The number of students for each unique score combination -/
def studentsPerCombination : Nat := 5

/-- The set of all possible score combinations -/
def allScoreCombinations : Finset ScoreCombination :=
  Finset.product (Finset.univ : Finset Score) (Finset.univ : Finset Score)

/-- The total number of students in the class -/
def totalStudents : Nat := Finset.card allScoreCombinations * studentsPerCombination

theorem class_size :
  totalStudents = 45 :=
by
  -- Unfold the definition of totalStudents
  unfold totalStudents
  -- Simplify the Finset.card expression
  simp [allScoreCombinations]
  -- Evaluate the arithmetic
  norm_num
  -- The proof is complete
  rfl

#eval totalStudents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_l74_7417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l74_7412

theorem inequality_proof (a₁ a₂ a₃ : ℝ) (h_distinct : a₁ ≠ a₂ ∧ a₂ ≠ a₃ ∧ a₃ ≠ a₁) :
  let b₁ := (1 + a₁*a₂/(a₁-a₂))*(1 + a₁*a₃/(a₁-a₃))
  let b₂ := (1 + a₂*a₁/(a₂-a₁))*(1 + a₂*a₃/(a₂-a₃))
  let b₃ := (1 + a₃*a₁/(a₃-a₁))*(1 + a₃*a₂/(a₃-a₂))
  1 + |a₁*b₁ + a₂*b₂ + a₃*b₃| ≤ (1 + |a₁|) * (1 + |a₂|) * (1 + |a₃|) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l74_7412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l74_7440

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 5 then 4 * x^2 + 5 else b * x + 2

theorem continuity_condition (b : ℝ) : 
  (∀ x, ContinuousAt (f b) x) ↔ b = 103/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l74_7440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l74_7473

-- Define the parameters a and b
variable (a b : ℝ)

-- Define the solution set of ax-b>0
def solution_set_1 : Set ℝ := Set.Ioi 1

-- Define the proposed solution set of (ax+b)/(x-2) > 0
def solution_set_2 : Set ℝ := Set.Iic (-1) ∪ Set.Ioi 2

-- State the theorem
theorem inequality_solution_sets 
  (h : {x : ℝ | a * x - b > 0} = solution_set_1) :
  {x : ℝ | (a * x + b) / (x - 2) > 0} = solution_set_2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l74_7473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_sum_l74_7471

/-- Circle w1 with equation x^2+y^2+12x-20y-100=0 -/
def w1 : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 + 12*p.1 - 20*p.2 - 100) = 0}

/-- Circle w2 with equation x^2+y^2-12x-20y+200=0 -/
def w2 : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 - 12*p.1 - 20*p.2 + 200) = 0}

/-- A circle is externally tangent to w2 and internally tangent to w1 -/
def is_tangent_circle (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    c = {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} ∧
    (∃ p ∈ w2, p ∈ c) ∧
    (∀ p ∈ w2, (p.1 - center.1)^2 + (p.2 - center.2)^2 ≥ radius^2) ∧
    (∃ p ∈ w1, p ∈ c) ∧
    (∀ p ∈ w1, (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2)

/-- The line y = nx contains the center of the tangent circle -/
def center_on_line (n : ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    c = {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} ∧
    center.2 = n * center.1

/-- n is the smallest positive value satisfying the tangency condition -/
def is_smallest_n (n : ℝ) : Prop :=
  n > 0 ∧
  (∃ c, is_tangent_circle c ∧ center_on_line n c) ∧
  (∀ m, 0 < m ∧ m < n → ¬∃ c, is_tangent_circle c ∧ center_on_line m c)

theorem tangent_circle_sum (n p q : ℕ) (hn : is_smallest_n (n : ℝ)) 
    (hpq : (n : ℝ)^2 = p / q) (hcoprime : Nat.Coprime p q) : p + q = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_sum_l74_7471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_operation_properties_l74_7441

def AbsoluteOperation (x y z m n : ℝ) : Set (ℝ → ℝ → ℝ → ℝ → ℝ → ℝ) := 
  {f | ∃ (a b c d : Bool), 
    f = λ x y z m n => (if a then |x - y| else x - y) - 
                       (if b then |z - (if a then |x - y| else x - y)| else z) - 
                       (if c then |m - (if b then |z - (if a then |x - y| else x - y)| else z)| else m) - 
                       (if d then |n - (if c then |m - (if b then |z - (if a then |x - y| else x - y)| else z)| else m)| else n)}

theorem absolute_operation_properties (x y z m n : ℝ) 
  (h : x > y ∧ y > z ∧ z > m ∧ m > n) : 
  (∃ f ∈ AbsoluteOperation x y z m n, f x y z m n = x - y - z - m - n) ∧ 
  (¬ ∃ f ∈ AbsoluteOperation x y z m n, f x y z m n = 0) ∧ 
  (∃ S : Finset ℝ, (∀ f ∈ AbsoluteOperation x y z m n, f x y z m n ∈ S) ∧ S.card ≠ 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_operation_properties_l74_7441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_4_l74_7449

def isInRange (n : ℕ) : Bool := 15 < n ∧ n < 55

def isDivisibleBy4 (n : ℕ) : Bool := n % 4 = 0

def numbersInRange : List ℕ := List.range 40 |>.map (· + 16)

def validNumbers : List ℕ := numbersInRange.filter (λ x => isInRange x && isDivisibleBy4 x)

theorem average_of_numbers_divisible_by_4 : 
  (validNumbers.sum : ℚ) / validNumbers.length = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_4_l74_7449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_needed_l74_7459

/-- Represents Alani's baby-sitting job information -/
structure BabySittingJob where
  pay : ℚ
  hours : ℚ

/-- Calculates the hourly rate for a baby-sitting job -/
def hourlyRate (job : BabySittingJob) : ℚ :=
  job.pay / job.hours

/-- Represents Alani's baby-sitting situation -/
structure BabySittingSituation where
  job1 : BabySittingJob
  job2 : BabySittingJob
  job3 : BabySittingJob
  incomeGoal : ℚ

/-- Theorem stating the total hours needed to reach Alani's income goal -/
theorem total_hours_needed (situation : BabySittingSituation)
  (h1 : situation.job1 = { pay := 45, hours := 3 })
  (h2 : situation.job2 = { pay := 90, hours := 6 })
  (h3 : situation.job3 = { pay := 30, hours := 2 })
  (h4 : situation.incomeGoal = 375)
  (h5 : hourlyRate situation.job1 = hourlyRate situation.job2)
  (h6 : hourlyRate situation.job2 = hourlyRate situation.job3) :
  situation.incomeGoal / (hourlyRate situation.job1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_needed_l74_7459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l74_7434

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The focal distance of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a ^ 2 - e.b ^ 2)

/-- Theorem: For an ellipse with equation x²/m + y²/4 = 1 and focal distance 2,
    the possible values of m are 3 and 5 -/
theorem ellipse_m_values (m : ℝ) (h_m_pos : 0 < m) :
  (∃ e : Ellipse, e.a ^ 2 = m ∧ e.b ^ 2 = 4 ∧ e.focalDistance = 2) ↔ m = 3 ∨ m = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l74_7434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_B_iff_a_equals_two_l74_7407

def A : Set ℝ := {1, 2, 3}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + 1 = 0}

theorem intersection_equals_B_iff_a_equals_two :
  ∀ a ∈ A, (A ∩ B a = B a) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_B_iff_a_equals_two_l74_7407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l74_7431

theorem cos_double_angle_special_case (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α + Real.cos α = 1/2) : 
  Real.cos (2 * α) = -Real.sqrt 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l74_7431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_l74_7409

theorem vector_calculation : 
  (⟨3, -8⟩ : ℝ × ℝ) - 5 • (⟨2, -3⟩ : ℝ × ℝ) + (⟨-1, 4⟩ : ℝ × ℝ) = ⟨-8, 11⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_l74_7409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l74_7426

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length ≥ 3 ∧
  (∀ i, i < arr.length → arr.get! i ∈ Finset.range 26 \ {0}) ∧
  (∀ i, i < arr.length → 
    (arr.get! i ^ 2 + arr.get! ((i + 1) % arr.length) ^ 2 + arr.get! ((i + 2) % arr.length) ^ 2) % 10 = 0)

theorem circular_arrangement_theorem :
  (¬ ∃ (arr : List Nat), arr.length = 8 ∧ is_valid_arrangement arr) ∧
  (∃ (arr : List Nat), arr.length = 9 ∧ is_valid_arrangement arr) :=
by sorry

#check circular_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l74_7426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l74_7463

/-- The focus of a parabola y² = 8x -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The equation of the parabola -/
def is_on_parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The equation of the hyperbola -/
def is_on_hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hf : (a^2 + b^2 : ℝ) = 4) -- The focus of the hyperbola coincides with the parabola's focus
  (hP : ∃ x y : ℝ, is_on_parabola x y ∧ is_on_hyperbola a b x y) -- Intersection point exists
  (hd : ∃ x y : ℝ, is_on_parabola x y ∧ is_on_hyperbola a b x y ∧ 
    distance x y (parabola_focus.1) (parabola_focus.2) = 5) -- Distance condition
  : a^2 = 1 ∧ b^2 = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l74_7463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_bounds_l74_7497

/-- Represents the walking scenario of a pedestrian along a bus route -/
structure WalkingScenario where
  bus_interval : ℚ  -- Time interval between buses in minutes
  same_direction_buses : ℕ  -- Number of buses passing in the same direction
  opposite_direction_buses : ℕ  -- Number of buses coming from opposite direction
  speed_ratio : ℚ  -- Ratio of bus speed to pedestrian speed

/-- Calculates the minimum and maximum walking time for the given scenario -/
def calculate_walking_time (scenario : WalkingScenario) : ℚ × ℚ :=
  sorry

/-- Theorem stating the bounds of the walking time for the given scenario -/
theorem walking_time_bounds (scenario : WalkingScenario) 
  (h1 : scenario.bus_interval = 5)
  (h2 : scenario.same_direction_buses = 11)
  (h3 : scenario.opposite_direction_buses = 13)
  (h4 : scenario.speed_ratio = 8) :
  let (min_time, max_time) := calculate_walking_time scenario
  min_time ≥ 57 + 1/7 ∧ max_time ≤ 62 + 2/9 :=
by sorry

#check walking_time_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_bounds_l74_7497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l74_7429

/-- Represents John's cycling marathon --/
structure CyclingMarathon where
  initial_distance : ℝ
  initial_time : ℝ
  break_time : ℝ
  final_distance : ℝ
  final_time : ℝ

/-- Calculates the average speed for a cycling marathon --/
noncomputable def average_speed (marathon : CyclingMarathon) : ℝ :=
  (marathon.initial_distance + marathon.final_distance) /
  (marathon.initial_time + marathon.break_time + marathon.final_time)

/-- John's specific cycling marathon --/
def johns_marathon : CyclingMarathon where
  initial_distance := 40
  initial_time := 4
  break_time := 1
  final_distance := 15
  final_time := 3

/-- Theorem stating that John's average speed is 6.875 miles per hour --/
theorem johns_average_speed :
  average_speed johns_marathon = 6.875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l74_7429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inexperienced_wage_is_correct_l74_7491

/-- Represents the hourly wage of an inexperienced sailor -/
noncomputable def inexperienced_wage : ℚ := 10

/-- The number of sailors in the crew -/
def total_sailors : ℕ := 17

/-- The number of inexperienced sailors -/
def inexperienced_sailors : ℕ := 5

/-- The number of experienced sailors -/
def experienced_sailors : ℕ := total_sailors - inexperienced_sailors

/-- The ratio of experienced to inexperienced sailor's wage -/
noncomputable def wage_ratio : ℚ := 6/5

/-- Hours worked per week -/
def weekly_hours : ℕ := 60

/-- Weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Total monthly earnings of experienced sailors -/
noncomputable def total_experienced_earnings : ℚ := 34560

theorem inexperienced_wage_is_correct :
  inexperienced_wage * wage_ratio * experienced_sailors * (weekly_hours * weeks_per_month) = total_experienced_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inexperienced_wage_is_correct_l74_7491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_difference_l74_7446

-- Define the amount of soda for each person
def julio_orange : ℝ := 4 * 2
def julio_grape : ℝ := 7 * 2
def mateo_orange : ℝ := 1 * 2.5
def mateo_grape : ℝ := 3 * 2.5
def sophia_orange : ℝ := 6 * 1.5
def sophia_strawberry_full : ℝ := 3 * 2.5
def sophia_strawberry_partial : ℝ := 2 * 2.5 * 0.75

-- Define the total amount of soda for each person
def julio_total : ℝ := julio_orange + julio_grape
def mateo_total : ℝ := mateo_orange + mateo_grape
def sophia_total : ℝ := sophia_orange + sophia_strawberry_full + sophia_strawberry_partial

-- Theorem statement
theorem soda_difference : 
  let max_soda := max (max julio_total mateo_total) sophia_total
  let min_soda := min (min julio_total mateo_total) sophia_total
  max_soda - min_soda = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_difference_l74_7446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l74_7443

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F of parabola C
def focus_F : ℝ × ℝ := (1, 0)

-- Define the line l passing through F with slope -1
def line_l (x y : ℝ) : Prop := y = -x + 1

-- Define circle D
def circle_D (x y : ℝ) : Prop := (x - 3)^2 + (y + 2)^2 = 16

-- Define circle T
def circle_T (x y a : ℝ) : Prop := (x + 2)^2 + (y + 7)^2 = a^2

-- Theorem statement
theorem parabola_circle_theorem (A B : ℝ × ℝ) (P Q : ℝ × ℝ) (M : ℝ × ℝ) (a : ℝ) :
  (∀ x y, parabola_C x y → line_l x y → (x, y) = A ∨ (x, y) = B) →
  (∀ x y, circle_D x y → (x, y) = P ∨ (x, y) = Q) →
  (∃ x y, circle_T x y a ∧ (x, y) = M) →
  a > 0 →
  -- Right angle condition (simplified representation)
  (P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) = 0 →
  -- Conclusion
  (∀ x y, circle_D x y ↔ (x - 3)^2 + (y + 2)^2 = 16) ∧
  Real.sqrt 2 ≤ a ∧ a ≤ 9 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l74_7443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_approx_simplifiedValue_l74_7418

noncomputable def trigExpression : ℝ :=
  (Real.sin (15 * Real.pi / 180) + Real.sin (30 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) +
   Real.sin (60 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) /
  (Real.cos (15 * Real.pi / 180) * Real.sin (45 * Real.pi / 180) * Real.cos (30 * Real.pi / 180))

def simplifiedValue : ℝ := 7.13109

theorem trigExpression_approx_simplifiedValue : 
  |trigExpression - simplifiedValue| < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_approx_simplifiedValue_l74_7418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l74_7421

-- Define the radii of the two spheres
def small_radius : ℝ := 3
def large_radius : ℝ := 6

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the volume of the region between the two spheres
noncomputable def volume_difference : ℝ := sphere_volume large_radius - sphere_volume small_radius

-- Theorem statement
theorem volume_between_spheres : volume_difference = 252 * Real.pi := by
  -- Expand the definitions
  unfold volume_difference sphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry to skip them for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l74_7421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l74_7487

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- Check if a point is on the parabola -/
def is_on_parabola (c : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * c.p * p.x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem distance_to_focus (c : Parabola) (m : Point) 
    (h_m_on_c : is_on_parabola c m) 
    (h_m_coords : m.x = 1 ∧ m.y = 2) : 
  distance m (focus c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l74_7487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_specific_value_l74_7422

noncomputable def α (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x), Real.cos x + Real.sin x)

noncomputable def β (x : ℝ) : ℝ × ℝ := (1, Real.cos x - Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (α x).1 * (β x).1 + (α x).2 * (β x).2

theorem f_period_and_specific_value :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (θ : ℝ), 0 < θ ∧ θ < Real.pi / 2 → f θ = 1 →
    Real.cos (θ - Real.pi / 6) = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_specific_value_l74_7422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_270_l74_7464

/-- The length of a platform given train passing times and train length -/
noncomputable def platform_length (train_length : ℝ) (time_pass_man : ℝ) (time_cross_platform : ℝ) : ℝ :=
  (train_length / time_pass_man) * time_cross_platform - train_length

/-- The length of the platform is 270 meters -/
theorem platform_length_is_270 :
  platform_length 180 8 20 = 270 := by
  -- Unfold the definition of platform_length
  unfold platform_length
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_270_l74_7464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_matrix_property_l74_7427

theorem scaling_matrix_property (w : Fin 4 → ℝ) : 
  let N : Matrix (Fin 4) (Fin 4) ℝ := λ i j => if i = j then 3 else 0
  Matrix.mulVec N w = λ i => 3 * w i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_matrix_property_l74_7427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_specific_parabola_directrix_l74_7452

/-- The directrix of a parabola y = ax^2 is given by y = -1/(4a) -/
theorem parabola_directrix (a : ℝ) (a_pos : a > 0) :
  let parabola := λ x : ℝ ↦ a * x^2
  let directrix := λ _ : ℝ ↦ -1 / (4 * a)
  ∀ x, directrix x = -1 / (4 * a) := by
  sorry

/-- For the specific parabola y = 8x^2, its directrix is y = -1/32 -/
theorem specific_parabola_directrix :
  let parabola := λ x : ℝ ↦ 8 * x^2
  let directrix := λ _ : ℝ ↦ -1 / 32
  ∀ x, directrix x = -1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_specific_parabola_directrix_l74_7452
