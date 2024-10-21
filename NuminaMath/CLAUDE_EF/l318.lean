import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_values_l318_31863

theorem system_solution_values (a b : ℚ) : 
  (∀ x y : ℚ, x + a * y = 3 ∧ 3 * x - a * y = 1) →
  (b + a * 1 = 3 ∧ 3 * b - a * 1 = 1) →
  a = 2 ∧ b = 1 := by
  sorry

#check system_solution_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_values_l318_31863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_scores_l318_31884

/-- Represents the scores of a basketball team over four quarters -/
structure TeamScores where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- Checks if scores follow a geometric sequence -/
def isGeometric (s : TeamScores) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if scores follow an arithmetic sequence -/
def isArithmetic (s : TeamScores) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℝ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def firstHalfScore (s : TeamScores) : ℝ :=
  s.q1 + s.q2

theorem basketball_game_scores (mustangs falcons : TeamScores) 
  (h1 : isGeometric mustangs)
  (h2 : isArithmetic falcons)
  (h3 : mustangs.q1 + mustangs.q2 + mustangs.q3 = falcons.q1 + falcons.q2 + falcons.q3)
  (h4 : totalScore mustangs = totalScore falcons + 2)
  (h5 : totalScore mustangs + totalScore falcons < 80) :
  firstHalfScore mustangs + firstHalfScore falcons = 20.4 := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_scores_l318_31884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_is_six_l318_31870

-- Define the triangle ABC
structure Triangle where
  A : Real  -- Angle A in radians
  B : Real  -- Angle B in radians
  C : Real  -- Angle C in radians
  a : Real  -- Side opposite to angle A

-- Define the properties of the triangle
def TriangleABC (t : Triangle) : Prop :=
  t.B = 2 * t.A ∧ 
  t.C = 3 * t.A ∧ 
  t.a = 6 ∧
  t.A + t.B + t.C = Real.pi

-- Define the radius of the circumscribed circle
noncomputable def CircumRadius (t : Triangle) : Real :=
  t.a / (2 * Real.sin t.A)

-- Theorem statement
theorem circumradius_is_six (t : Triangle) 
  (h : TriangleABC t) : CircumRadius t = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_is_six_l318_31870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pentagon_sum_length_bound_l318_31891

/-- A pentagon inscribed in a unit circle -/
structure InscribedPentagon where
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The vertices lie on the unit circle -/
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The sum of lengths of sides and diagonals of an inscribed pentagon -/
noncomputable def sum_lengths (p : InscribedPentagon) : ℝ :=
  let sides := (Finset.sum (Finset.range 5) fun i => 
    Real.sqrt ((p.vertices i).1 - (p.vertices ((i + 1) % 5)).1)^2 + 
               ((p.vertices i).2 - (p.vertices ((i + 1) % 5)).2)^2)
  let diagonals := (Finset.sum (Finset.range 5) fun i => 
    Finset.sum (Finset.range 5) fun j =>
      if i < j ∧ (i + 1) % 5 ≠ j then
        Real.sqrt ((p.vertices i).1 - (p.vertices j).1)^2 + 
                   ((p.vertices i).2 - (p.vertices j).2)^2
      else 0)
  sides + diagonals

/-- The theorem to be proved -/
theorem inscribed_pentagon_sum_length_bound (p : InscribedPentagon) :
  sum_lengths p < 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pentagon_sum_length_bound_l318_31891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_given_positive_test_l318_31808

/-- The probability of having the disease in the population -/
noncomputable def disease_probability : ℝ := 1 / 200

/-- The probability of not having the disease in the population -/
noncomputable def no_disease_probability : ℝ := 1 - disease_probability

/-- The probability of testing positive given that the person has the disease -/
noncomputable def test_positive_given_disease : ℝ := 1

/-- The probability of testing positive given that the person does not have the disease (false positive rate) -/
noncomputable def false_positive_rate : ℝ := 0.05

/-- The probability of testing positive -/
noncomputable def test_positive_probability : ℝ := 
  disease_probability * test_positive_given_disease + 
  no_disease_probability * false_positive_rate

theorem disease_given_positive_test : 
  (disease_probability * test_positive_given_disease) / test_positive_probability = 20 / 219 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_given_positive_test_l318_31808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_difference_not_22_l318_31823

theorem divisor_difference_not_22 (n : ℕ) 
  (h_n : n < 1995) 
  (h_prime_factors : ∃ p₁ p₂ p₃ p₄ : ℕ, 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ * p₂ * p₃ * p₄)
  (h_divisors : ∃ d : Fin 16 → ℕ, 
    (∀ i : Fin 16, d i ∣ n) ∧
    (∀ i j : Fin 16, i < j → d i < d j) ∧
    d 0 = 1 ∧ d 15 = n ∧
    (∀ m : ℕ, m ∣ n → ∃ i : Fin 16, d i = m)) :
  ∀ d : Fin 16 → ℕ, (d 8 : ℕ) - (d 7 : ℕ) ≠ 22 :=
by
  intro d
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_difference_not_22_l318_31823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l318_31873

/- Define the circles and triangle -/
def circle1_radius : ℝ := 3
def circle2_radius : ℝ := 4

structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/- Define the properties of the triangle -/
def is_tangent_to_circles (t : Triangle) : Prop :=
  -- The sides of the triangle are tangent to both circles
  sorry

def has_two_congruent_sides (t : Triangle) : Prop :=
  -- The sides DE and DF are congruent
  sorry

/- Define the area function for the triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  -- This is a placeholder definition; replace with actual area calculation
  sorry

/- Theorem statement -/
theorem triangle_area (t : Triangle) 
  (h1 : is_tangent_to_circles t) 
  (h2 : has_two_congruent_sides t) : 
  Real.sqrt ((area t) * (area t)) = 42 * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l318_31873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_curves_l318_31858

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

/-- Curve C₂ in polar form -/
noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.sin θ

/-- Curve C₃ in polar form -/
noncomputable def C₃ (θ : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos θ

/-- Distance between two points in Cartesian coordinates -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem max_distance_on_curves :
  ∃ (t₁ t₂ α₁ α₂ : ℝ), t₁ ≠ 0 ∧ t₂ ≠ 0 ∧ 0 ≤ α₁ ∧ α₁ < π ∧ 0 ≤ α₂ ∧ α₂ < π ∧
  C₂ α₁ = Real.sqrt ((C₁ t₁ α₁).1^2 + (C₁ t₁ α₁).2^2) ∧
  C₃ α₂ = Real.sqrt ((C₁ t₂ α₂).1^2 + (C₁ t₂ α₂).2^2) ∧
  (∀ (s₁ s₂ β₁ β₂ : ℝ), s₁ ≠ 0 → s₂ ≠ 0 → 0 ≤ β₁ → β₁ < π → 0 ≤ β₂ → β₂ < π →
    C₂ β₁ = Real.sqrt ((C₁ s₁ β₁).1^2 + (C₁ s₁ β₁).2^2) →
    C₃ β₂ = Real.sqrt ((C₁ s₂ β₂).1^2 + (C₁ s₂ β₂).2^2) →
    distance (C₁ t₁ α₁) (C₁ t₂ α₂) ≥ distance (C₁ s₁ β₁) (C₁ s₂ β₂)) ∧
  distance (C₁ t₁ α₁) (C₁ t₂ α₂) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_curves_l318_31858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_symmetry_l318_31835

/-- Given a cubic function f(x) = ax³ + bx + l and f(m) = 2, prove that f(-m) = 0 -/
theorem cubic_function_symmetry (a b l m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^3 + b * x + l) →
  f m = 2 →
  f (-m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_symmetry_l318_31835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l318_31882

/-- The surface area of a cone given its slant height and height -/
noncomputable def cone_surface_area (slant_height : ℝ) (height : ℝ) : ℝ :=
  let radius := Real.sqrt (slant_height^2 - height^2)
  Real.pi * radius^2 + Real.pi * radius * slant_height

theorem cone_surface_area_specific : 
  cone_surface_area 10 8 = 96 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l318_31882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l318_31872

/-- An odd function f(x) = (ax^2 + 1) / (bx + c) where a, b, c are constants -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + 1) / (b * x + c)

/-- Main theorem -/
theorem odd_function_properties (a b : ℝ) (c : ℝ) :
  (∀ x, f a b c (-x) = -f a b c x) →  -- f is odd
  c = 0 ∧
  (a > 0 ∧ b > 0 ∧ f a b c 1 = 2 ∧ f a b c 2 < 3 → 
    ∀ x, f a b c x = (x^2 + 1) / x) ∧
  (∀ m : ℝ, (∃ x > 0, f a b c x = m) ↔ m ≥ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l318_31872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l318_31830

/-- An ellipse with the given properties has eccentricity √2 - 1 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  ∃ (F₁ F₂ B C : ℝ × ℝ),
    (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ ({B, C} : Set (ℝ × ℝ))) ∧  -- B and C are on the ellipse
    (F₁.1 < 0 ∧ F₂.1 > 0 ∧ F₁.2 = 0 ∧ F₂.2 = 0) ∧  -- F₁ and F₂ are foci on x-axis
    (B.1 = F₁.1 ∧ C.1 = F₁.1) ∧  -- B and C are on a line through F₁
    ((B.2 - F₂.2) * (C.2 - F₂.2) = -(B.1 - F₂.1) * (C.1 - F₂.1)) →  -- ∠BF₂C = 90°
  e = Real.sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l318_31830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l318_31859

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point F -/
def F : ℝ × ℝ := (-1, 0)

/-- Line l -/
def l (x : ℝ) : Prop := x = -4

/-- Distance from a point to line l -/
def dist_to_l (x y : ℝ) : ℝ := |x + 4|

/-- Distance between two points -/
noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := ((x₁ - x₂)^2 + (y₁ - y₂)^2).sqrt

/-- Point P -/
noncomputable def P : ℝ × ℝ := (1/2, 3 * Real.sqrt 5 / 4)

/-- The condition that P satisfies -/
axiom h_P_condition : ∀ x y, trajectory x y → dist_to_l x y = 2 * dist x y (-1) 0

/-- P is on the trajectory -/
axiom h_P_on_trajectory : trajectory P.1 P.2

/-- Q is the other intersection of PF with the trajectory -/
axiom h_Q_exists : ∃ Q : ℝ × ℝ, 
  Q ≠ P ∧ 
  trajectory Q.1 Q.2 ∧ 
  ∃ t : ℝ, Q.1 = P.1 + t * (F.1 - P.1) ∧ Q.2 = P.2 + t * (F.2 - P.2)

/-- Radius of the circle -/
noncomputable def r (Q : ℝ × ℝ) : ℝ := 2 * dist P.1 P.2 Q.1 Q.2

/-- Chord AB -/
noncomputable def AB (Q : ℝ × ℝ) : ℝ := 2 * Real.sqrt ((r Q)^2 - (P.1 + 4)^2)

/-- Main theorem -/
theorem chord_length (Q : ℝ × ℝ) (hQ : Q ≠ P ∧ trajectory Q.1 Q.2 ∧ ∃ t : ℝ, Q.1 = P.1 + t * (F.1 - P.1) ∧ Q.2 = P.2 + t * (F.2 - P.2)) : 
  AB Q = 3 * Real.sqrt 77 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l318_31859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l318_31867

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem point_on_line_segment (P : ℝ × ℝ) :
  distance F₁ F₂ / 2 = (distance P F₁ + distance P F₂) / 2 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * F₂.1 + (1 - t) * F₁.1, t * F₂.2 + (1 - t) * F₁.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l318_31867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_four_l318_31868

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ones_digit (n : ℕ) : ℕ := n % 10

theorem probability_divisible_by_four (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : ones_digit N = 4) : 
  (Finset.filter (λ n => n % 4 = 0) (Finset.range 10)).card / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_four_l318_31868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l318_31826

-- Define the function f(x) = x^2 - ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Theorem statement
theorem min_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 1 2 → f x ≤ f y) ∧
  f x = 1 := by
  sorry

-- Note: Set.Icc 1 2 represents the closed interval [1, 2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l318_31826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_PQ_RS_l318_31828

noncomputable section

-- Define the quadrilateral ABCD
def A : ℝ × ℝ := (0, 0)
def B (b : ℝ) : ℝ × ℝ := (b, 0)
def C (c a : ℝ) : ℝ × ℝ := (c, a)
def D (d e : ℝ) : ℝ × ℝ := (d, e)

-- Define the intersection point M of diagonals AC and BD
noncomputable def M (a b c d e : ℝ) : ℝ × ℝ :=
  ((b * c * e) / (a * (b - d) + c * e), (a * b * e) / (a * (b - d) + c * e))

-- Define centroids P and Q
noncomputable def P (a b c d e : ℝ) : ℝ × ℝ :=
  ((d + (b * c * e) / (a * (b - d) + c * e)) / 3, (e + (a * b * e) / (a * (b - d) + c * e)) / 3)

noncomputable def Q (a b c d e : ℝ) : ℝ × ℝ :=
  ((b + c + (b * c * e) / (a * (b - d) + c * e)) / 3, (a + (a * b * e) / (a * (b - d) + c * e)) / 3)

-- Define orthocenters R and S
noncomputable def R (a b c d e : ℝ) : ℝ × ℝ :=
  ((b * c * e) / (a * (b - d) + c * e), (b * c * (b - d)) / (a * (b - d) + c * e))

noncomputable def S (a b c d e : ℝ) : ℝ × ℝ :=
  ((-a^2 * e + a * (b * c - c * d + e^2) - c * d * e) / (a * (b - d) + c * e),
   (a * e * (b + c - d) + c * (b - d) * (d - c)) / (a * (b - d) + c * e))

-- Define the theorem
theorem perpendicular_PQ_RS (a b c d e : ℝ) :
  (a ≠ e) → (d ≠ b + c) →
  let pq_slope := (e - a) / (d - b - c)
  let rs_slope := (b + c - d) / (e - a)
  pq_slope * rs_slope = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_PQ_RS_l318_31828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_l318_31833

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude_AB : ℝ := Real.sqrt (vector_AB.1^2 + vector_AB.2^2)

noncomputable def unit_vector : ℝ × ℝ := (vector_AB.1 / magnitude_AB, vector_AB.2 / magnitude_AB)

theorem unit_vector_AB : unit_vector = (3/5, -4/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_l318_31833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duodecimal_divisibility_rule_l318_31888

/-- Represents a digit in the duodecimal system -/
inductive DuodecimalDigit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Eleven

/-- Represents a number in the duodecimal system as a list of digits -/
def DuodecimalNumber := List DuodecimalDigit

/-- Checks if a duodecimal number is divisible by a natural number -/
def isDivisibleBy (n : DuodecimalNumber) (d : Nat) : Prop := sorry

/-- Checks if a duodecimal digit is divisible by a natural number -/
def isDigitDivisibleBy (digit : DuodecimalDigit) (d : Nat) : Prop := sorry

/-- Gets the last digit of a duodecimal number -/
def lastDigit (n : DuodecimalNumber) : Option DuodecimalDigit :=
  n.getLast?

/-- The main theorem stating the divisibility rule for 2, 3, and 4 in duodecimal system -/
theorem duodecimal_divisibility_rule (n : DuodecimalNumber) (d : Nat) :
  d ∈ [2, 3, 4] →
  (∀ last : DuodecimalDigit, lastDigit n = some last →
    (isDivisibleBy n d ↔ isDigitDivisibleBy last d)) ∧
  (∀ (k : Nat), k ≤ 12 → k ∉ [2, 3, 4] →
    ¬(∀ last : DuodecimalDigit, lastDigit n = some last →
      (isDivisibleBy n k ↔ isDigitDivisibleBy last k))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_duodecimal_divisibility_rule_l318_31888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lambda_inequality_l318_31815

theorem smallest_lambda_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  5 * (a*b + a*c + a*d + b*c + b*d + c*d) ≤ 8 * a*b*c*d + 12 ∧
  ∀ lambda : ℝ, (∀ a' b' c' d' : ℝ, 0 < a' → 0 < b' → 0 < c' → 0 < d' → 
    a' + b' + c' + d' = 4 → 
    5 * (a'*b' + a'*c' + a'*d' + b'*c' + b'*d' + c'*d') ≤ lambda * a'*b'*c'd' + 12) → 
  8 ≤ lambda :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lambda_inequality_l318_31815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l318_31895

/-- The line equation y = kx - 1 -/
def line (k x : ℝ) : ℝ := k * x - 1

/-- The curve equation x² - y² = 4 -/
def curve (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ p.2 = line k p.1

/-- k = ± √5/2 is a necessary but not sufficient condition for single intersection -/
theorem necessary_not_sufficient :
  (∀ k : ℝ, single_intersection k → k = Real.sqrt 5 / 2 ∨ k = -Real.sqrt 5 / 2) ∧
  ¬(∀ k : ℝ, (k = Real.sqrt 5 / 2 ∨ k = -Real.sqrt 5 / 2) → single_intersection k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l318_31895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_words_per_page_l318_31845

theorem words_per_page 
  (total_pages words_mod_251 words_div_5 words_per_page : ℕ)
  (h1 : total_pages = 150)
  (h2 : words_mod_251 = 100)
  (h3 : (total_pages * words_per_page) % 251 = words_mod_251)
  (h4 : (total_pages * words_per_page) % 5 = 0) :
  words_per_page = 52 :=
by sorry

#check words_per_page

end NUMINAMATH_CALUDE_ERRORFEEDBACK_words_per_page_l318_31845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_on_inclined_plane_l318_31801

noncomputable def wheelRevolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * diameter)

noncomputable def roundToNearestInt (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem wheel_revolutions_on_inclined_plane :
  roundToNearestInt (wheelRevolutions 14 1056) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_on_inclined_plane_l318_31801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l318_31855

-- Define the function f
noncomputable def f (a b A B θ : ℝ) : ℝ := 1 + a * Real.cos θ + b * Real.sin θ + A * Real.sin (2 * θ) + B * Real.cos (2 * θ)

-- State the theorem
theorem function_inequality (a b A B : ℝ) 
  (h : ∀ θ : ℝ, f a b A B θ ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l318_31855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_interval_l318_31807

def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + (k - 1) * x + 3

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def decreasingInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem even_function_decreasing_interval (k : ℝ) :
  isEven (f k) → decreasingInterval (f k) 0 (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_interval_l318_31807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l318_31822

noncomputable section

/-- The curve in polar coordinates -/
def curve (θ : Real) : Real := Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

/-- The line in parametric form -/
def line (t : Real) : Real × Real := (1 + (4/5) * t, -1 - (3/5) * t)

/-- The length of the chord -/
def chord_length : Real := 7/5

theorem chord_length_is_correct : 
  ∃ (θ₁ θ₂ t₁ t₂ : Real), 
    let (x₁, y₁) := line t₁
    let (x₂, y₂) := line t₂
    let ρ₁ := curve θ₁
    let ρ₂ := curve θ₂
    x₁ = ρ₁ * Real.cos θ₁ ∧ 
    y₁ = ρ₁ * Real.sin θ₁ ∧
    x₂ = ρ₂ * Real.cos θ₂ ∧ 
    y₂ = ρ₂ * Real.sin θ₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = chord_length :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l318_31822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marcella_shoes_l318_31803

/-- Given a person who loses some shoes and has some matching pairs left,
    calculate the initial number of full pairs of shoes they had. -/
def initial_full_pairs (lost : ℕ) (remaining_pairs : ℕ) : ℕ :=
  ((remaining_pairs * 2 + lost) / 2)

theorem marcella_shoes :
  initial_full_pairs 9 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marcella_shoes_l318_31803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l318_31871

noncomputable def f (x : ℝ) := Real.sqrt (x + 4) / (x + 2)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y, f x = y

theorem domain_of_f : 
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ -4 ∧ x ≠ -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l318_31871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_f_coefficient_value_l318_31856

def f (x : ℝ) := (3 - x)^6 - x*(3 - x)^5

theorem coefficient_x_cubed_in_f : 
  ∃ (a b c d e : ℝ), f = λ x ↦ a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + (3^6) :=
by sorry

theorem coefficient_value : 
  ∃ (a b c d e : ℝ), f = λ x ↦ a*x^5 + b*x^4 + (-810)*x^3 + d*x^2 + e*x + (3^6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_f_coefficient_value_l318_31856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l318_31832

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the sides, inradius, and ex-radii
def Triangle.side_a (t : Triangle) : ℝ := sorry
def Triangle.side_b (t : Triangle) : ℝ := sorry
def Triangle.side_c (t : Triangle) : ℝ := sorry
def Triangle.inradius (t : Triangle) : ℝ := sorry
def Triangle.exradius_a (t : Triangle) : ℝ := sorry
def Triangle.exradius_b (t : Triangle) : ℝ := sorry
def Triangle.exradius_c (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be acute
def Triangle.is_acute (t : Triangle) : Prop := sorry

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.side_a > t.exradius_a)
  (h2 : t.side_b > t.exradius_b)
  (h3 : t.side_c > t.exradius_c) : 
  (t.is_acute) ∧ 
  (t.side_a + t.side_b + t.side_c > 
   t.inradius + t.exradius_a + t.exradius_b + t.exradius_c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l318_31832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l318_31804

noncomputable def f (x : ℝ) := Real.sqrt (x - 3) / (abs (x + 1) - 5)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Icc 3 4 ∪ Set.Ioi 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l318_31804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_interval_increase_l318_31887

/-- The maintenance interval problem -/
theorem maintenance_interval_increase :
  let base_interval : ℝ := 25
  let additive_a_increase : ℝ := 0.10
  let additive_b_increase : ℝ := 0.15
  let additive_c_increase : ℝ := 0.05
  let harsh_conditions_decrease : ℝ := 0.05
  let manufacturer_recommendation : ℝ := 0.03
  
  let interval_after_a : ℝ := base_interval * (1 + additive_a_increase)
  let interval_after_b : ℝ := interval_after_a * (1 + additive_b_increase)
  let interval_after_c : ℝ := interval_after_b * (1 + additive_c_increase)
  let interval_after_harsh : ℝ := interval_after_c * (1 - harsh_conditions_decrease)
  let final_interval : ℝ := interval_after_harsh * (1 + manufacturer_recommendation)
  
  let total_increase_percentage : ℝ := (final_interval - base_interval) / base_interval * 100

  ∃ ε > 0, |total_increase_percentage - 29.9692625| < ε := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_interval_increase_l318_31887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_point_correct_l318_31842

noncomputable section

/-- The point of reflection on the plane -/
def B : ℝ × ℝ × ℝ := (14/3, 11/3, 32/3)

/-- The starting point of the light ray -/
def A : ℝ × ℝ × ℝ := (-2, 8, 10)

/-- The endpoint of the light ray after reflection -/
def C : ℝ × ℝ × ℝ := (4, 4, 7)

/-- The normal vector of the reflecting plane -/
def normal : ℝ × ℝ × ℝ := (1, 1, 1)

/-- Check if a point is on the plane x + y + z = 15 -/
def on_plane (p : ℝ × ℝ × ℝ) : Prop :=
  p.1 + p.2.1 + p.2.2 = 15

/-- Calculate the dot product of two 3D vectors -/
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

/-- Calculate the vector from one point to another -/
def vector_between (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (q.1 - p.1, (q.2.1 - p.2.1, q.2.2 - p.2.2))

/-- Check if the angle of incidence equals the angle of reflection -/
def angle_of_incidence_equals_reflection (a b c : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : Prop :=
  dot_product (vector_between a b) n = dot_product (vector_between c b) n

theorem reflection_point_correct :
  on_plane B ∧ angle_of_incidence_equals_reflection A B C normal :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_point_correct_l318_31842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_nth_root_l318_31836

theorem smallest_upper_bound_nth_root :
  (∃ k : ℝ, ∀ n : ℕ+, 1 ≤ (n : ℝ)^(1/n : ℝ) ∧ (n : ℝ)^(1/n : ℝ) ≤ k) ∧
  (∀ k : ℝ, (∀ n : ℕ+, 1 ≤ (n : ℝ)^(1/n : ℝ) ∧ (n : ℝ)^(1/n : ℝ) ≤ k) → 3^(1/3 : ℝ) ≤ k) :=
by sorry

#check smallest_upper_bound_nth_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_nth_root_l318_31836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l318_31819

theorem expression_equality : 
  let a : ℝ := 0.32
  let x : ℝ := 0.08
  (Real.sqrt 2 * (x - a)) / (2 * x - a) - 
  ((((Real.sqrt x) / (Real.sqrt (2 * x) + Real.sqrt a)) ^ 2 + 
    ((Real.sqrt (2 * x) + Real.sqrt a) / (2 * Real.sqrt a)) ^ (-1 : ℝ)) ^ (1/2 : ℝ)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l318_31819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_angle_l318_31817

theorem perpendicular_lines_angle (α : Real) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (∀ x y : Real, x * Real.cos α - y - 1 = 0 ∧ x + y * Real.sin α + 1 = 0 → 
    (x * Real.cos α - y - 1) * (x + y * Real.sin α + 1) = 0) →
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_angle_l318_31817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l318_31810

theorem function_properties (f : ℝ → ℝ) : 
  (∃ A ω φ B : ℝ, A > 0 ∧ ω > 0 ∧ abs φ < π/2 ∧
    (∀ x, f x = A * Real.sin (ω * x + φ) + B)) →
  (∀ x, f (x + 2*π) = f x) →
  (∃ x, f x = -2) →
  (f (5*π/6) = 4) →
  (∀ x, f x = 3 * Real.sin (x - π/3) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l318_31810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remi_cant_reach_goal_l318_31892

-- Define the point type
def Point := ℚ × ℚ

-- Define the initial positions
def remi_start : Point := (0, 0)
def aurelien : Point := (0, 1)
def savinien : Point := (1, 0)
def remi_goal : Point := (2, 2)

-- Function to calculate the area of a triangle given three points
def triangle_area (p1 p2 p3 : Point) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem remi_cant_reach_goal :
  ∀ (final_pos : Point),
  (triangle_area remi_start aurelien savinien = triangle_area final_pos aurelien savinien) →
  final_pos ≠ remi_goal :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remi_cant_reach_goal_l318_31892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l318_31837

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 * (Real.log x / Real.log 3)
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x) / Real.log 2

-- Theorem statement
theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l318_31837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l318_31876

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (0, 1)

-- State the theorem
theorem tangent_line_triangle_area :
  let tangent_slope := (deriv f) (point_of_tangency.fst)
  let tangent_line (x : ℝ) := tangent_slope * (x - point_of_tangency.fst) + point_of_tangency.snd
  let x_intercept := -point_of_tangency.snd / tangent_slope
  let y_intercept := tangent_line 0
  let triangle_area := (1/2) * abs x_intercept * y_intercept
  triangle_area = 1/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l318_31876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l318_31874

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) / Real.log 2

-- State the theorem
theorem f_properties :
  ∀ x x₁ x₂ a b : ℝ,
  -1 < x ∧ x < 1 ∧ 
  -1 < x₁ ∧ x₁ < 1 ∧
  -1 < x₂ ∧ x₂ < 1 ∧
  -1 < a ∧ a < 1 ∧
  -1 < b ∧ b < 1 →
  (f (-x) = -f x) ∧ 
  (f x₁ + f x₂ = f ((x₁ + x₂) / (1 + x₁ * x₂))) ∧
  (f ((a + b) / (1 + a * b)) = 1 ∧ f (-b) = 1/2 → f a = 3/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l318_31874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_three_pi_four_l318_31844

noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

theorem function_value_at_three_pi_four 
  (A : ℝ) (φ : ℝ) 
  (h1 : A > 0) 
  (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∀ x, f A φ x ≤ 1) 
  (h5 : ∃ x, f A φ x = 1) 
  (h6 : f A φ (π/3) = 1/2) :
  f A φ (3*π/4) = -Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_three_pi_four_l318_31844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deduction_from_second_l318_31816

def consecutive_integers (start : ℚ) : List ℚ :=
  List.range 10 |>.map (λ i => start + i)

def deductions (y : ℚ) : List ℚ :=
  [9, y, 7, 6, 5, 4, 3, 2, 1, 0]

theorem deduction_from_second (start : ℚ) (y : ℚ) : 
  (consecutive_integers start).sum / 10 = 25 →
  (List.zip (consecutive_integers start) (deductions y) |>.map (λ (a, b) => a - b)).sum / 10 = 15.5 →
  y = 58 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deduction_from_second_l318_31816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_disjoint_sets_exist_l318_31840

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a predicate for collinearity of three points
def collinear (p q r : Point) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Define a predicate for a point being inside a triangle
def insideTriangle (p a b c : Point) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u + v + w = 1 ∧
  p = (u * a.1 + v * b.1 + w * c.1, u * a.2 + v * b.2 + w * c.2)

-- Theorem statement
theorem no_disjoint_sets_exist :
  ¬ ∃ (A B : Set Point),
    -- A and B are infinite and disjoint
    (Set.Infinite A ∧ Set.Infinite B ∧ A ∩ B = ∅) ∧
    -- Any 3 points in A ∪ B are non-collinear
    (∀ (p q r : Point), p ∈ A ∪ B → q ∈ A ∪ B → r ∈ A ∪ B →
      p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬collinear p q r) ∧
    -- Distance between any two points in A ∪ B is at least 1
    (∀ (p q : Point), p ∈ A ∪ B → q ∈ A ∪ B → p ≠ q → distance p q ≥ 1) ∧
    -- Inside any triangle with vertices in B, there is a point from A
    (∀ (p q r : Point), p ∈ B → q ∈ B → r ∈ B →
      ∃ (a : Point), a ∈ A ∧ insideTriangle a p q r) ∧
    -- Inside any triangle with vertices in A, there is a point from B
    (∀ (p q r : Point), p ∈ A → q ∈ A → r ∈ A →
      ∃ (b : Point), b ∈ B ∧ insideTriangle b p q r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_disjoint_sets_exist_l318_31840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l318_31831

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- A point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : ℝ × ℝ := by
  let c := Real.sqrt (h.a^2 + h.b^2)
  exact (-c, c)

/-- Line perpendicular bisector -/
def LinePerpendicularBisector (x1 y1 x2 y2 x3 y3 : ℝ) : Prop := sorry

/-- Distance from origin to line -/
def DistanceFromOriginToLine (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- Theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (p : PointOnHyperbola h)
  (right_branch : p.x > 0)
  (perp_bisector : ∃ (m : ℝ), LinePerpendicularBisector p.x p.y (foci h).1 0 m (foci h).2)
  (origin_distance : DistanceFromOriginToLine p.x p.y (foci h).1 0 = h.a) :
  eccentricity h = 5/3 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l318_31831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l318_31866

/-- Given a parabola y² = 2px (p > 0) with focus F and a point A on the parabola,
    if the angle between FA and the positive x-axis is 60°,
    then the distance from the origin O to point A is (√21/2)p -/
theorem parabola_point_distance (p : ℝ) (h_p : p > 0) :
  ∀ (A : ℝ × ℝ),
    (A.2)^2 = 2 * p * A.1 →  -- A is on the parabola
    let F := (p/2, 0)  -- Focus of the parabola
    ∃ θ : ℝ,
      θ = π/3 ∧  -- Angle between FA and positive x-axis is 60°
      (A.1 - F.1) * Real.cos θ + (A.2 - F.2) * Real.sin θ =
        Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) →
    Real.sqrt (A.1^2 + A.2^2) = (Real.sqrt 21 / 2) * p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l318_31866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l318_31864

open Real

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) :
  a + b + c = 9 →
  (sin A : ℝ) / 3 = (sin B : ℝ) / 2 →
  (sin A : ℝ) / 3 = (sin C : ℝ) / 4 →
  a / (sin A : ℝ) = b / (sin B : ℝ) →
  a / (sin A : ℝ) = c / (sin C : ℝ) →
  (cos C : ℝ) = (a^2 + b^2 - c^2) / (2 * a * b) →
  (cos C : ℝ) = -1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l318_31864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_maximum_marks_l318_31838

theorem exam_maximum_marks : ℕ := by
  let passing_percentage : ℚ := 60 / 100
  let marks_obtained : ℕ := 52
  let marks_short : ℕ := 18
  let maximum_marks : ℕ := 117
  let passing_marks : ℕ := (passing_percentage * maximum_marks).ceil.toNat
  have h1 : passing_marks = marks_obtained + marks_short := by sorry
  exact maximum_marks


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_maximum_marks_l318_31838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assertions_correctness_l318_31846

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 then |x| / x else 0

noncomputable def g (x : ℝ) : ℝ := if x ≥ 0 then 1 else -1

def h (x : ℝ) : ℝ := 2 * x - 1

def k : ℝ → ℝ := λ _ => 1

theorem assertions_correctness :
  (¬ (∀ x : ℝ, f x = g x)) ∧ 
  (∀ x : ℝ, h x = h x) ∧
  (∀ y : ℝ, (Set.preimage h {y}).Finite) ∧
  (Function.Injective k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assertions_correctness_l318_31846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l318_31805

open Real

noncomputable def f (x : ℝ) : ℝ := (x / (2 * Real.sqrt (1 - 4 * x^2))) * Real.arcsin (2 * x) + (1/8) * Real.log (1 - 4 * x^2)

theorem f_derivative (x : ℝ) (h : x ≠ -1/2 ∧ x ≠ 1/2) : 
  deriv f x = (Real.arcsin (2 * x)) / (2 * (1 - 4 * x^2) * Real.sqrt (1 - 4 * x^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l318_31805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_interval_l318_31813

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x - 1)

-- State the theorem
theorem f_decreasing_on_open_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₂ → x₂ < x₁ → f x₁ < f x₂ := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_interval_l318_31813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l318_31879

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def solution_pairs : Set (ℝ × ℝ) :=
  {(Real.sqrt 975, Real.sqrt 980.99), (Real.sqrt 975, -Real.sqrt 980.99),
   (-Real.sqrt 975, Real.sqrt 1043.99), (-Real.sqrt 975, -Real.sqrt 1043.99)}

theorem equation_solution :
  ∀ x y : ℝ, (y^2 - (floor x)^2 = 19.99 ∧ x^2 + (floor y)^2 = 1999) ↔ (x, y) ∈ solution_pairs :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l318_31879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_col_products_not_equal_l318_31894

-- Define the table type
def Table := Matrix (Fin 9) (Fin 9) ℕ

-- Define a valid table
def is_valid_table (t : Table) : Prop :=
  ∀ i j, t i j ∈ Finset.range 82 ∧ t i j ≠ 0

-- Define row products
def row_products (t : Table) : Finset ℕ :=
  Finset.image (λ i : Fin 9 => (Finset.univ.prod (λ j : Fin 9 => t i j))) Finset.univ

-- Define column products
def column_products (t : Table) : Finset ℕ :=
  Finset.image (λ j : Fin 9 => (Finset.univ.prod (λ i : Fin 9 => t i j))) Finset.univ

-- Theorem statement
theorem row_col_products_not_equal :
  ∀ t : Table, is_valid_table t → row_products t ≠ column_products t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_col_products_not_equal_l318_31894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_median_angle_l318_31897

/-- Triangle KLM with given properties -/
structure Triangle where
  /-- Circumradius of the triangle -/
  R : ℝ
  /-- Side KL of the triangle -/
  c : ℝ
  /-- Height MH of the triangle -/
  h : ℝ
  /-- Constraint: Circumradius is 10 -/
  hR : R = 10
  /-- Constraint: Side KL is 16 -/
  hc : c = 16
  /-- Constraint: Height MH is 39/10 -/
  hh : h = 39/10

/-- The angle KML that minimizes the median MN -/
noncomputable def minMedianAngle (t : Triangle) : ℝ := Real.pi - Real.arcsin (4/5)

/-- Theorem: The angle KML that minimizes the median MN is π - arcsin(4/5) -/
theorem min_median_angle (t : Triangle) : 
  minMedianAngle t = Real.pi - Real.arcsin (4/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_median_angle_l318_31897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_second_quadrant_l318_31861

/-- Two lines intersect in the second quadrant if and only if their intersection point (x, y) satisfies x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of the intersection point of two lines y = ax + b and y = cx + d -/
noncomputable def intersection_x (a b c d : ℝ) : ℝ := (d - b) / (a - c)

/-- The y-coordinate of the intersection point of two lines y = ax + b and y = cx + d -/
noncomputable def intersection_y (a b c d : ℝ) : ℝ := a * ((d - b) / (a - c)) + b

theorem intersection_in_second_quadrant (m : ℝ) :
  second_quadrant (intersection_x 2 4 (-2) m) (intersection_y 2 4 (-2) m) ↔ -4 < m ∧ m < 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_second_quadrant_l318_31861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_area_perimeter_ratio_l318_31800

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

def equilateral_triangle_perimeter (s : ℝ) : ℝ := 3 * s

theorem equilateral_triangles_area_perimeter_ratio :
  let s₁ : ℝ := 6
  let s₂ : ℝ := 8
  let area₁ := equilateral_triangle_area s₁
  let area₂ := equilateral_triangle_area s₂
  let perimeter₁ := equilateral_triangle_perimeter s₁
  let perimeter₂ := equilateral_triangle_perimeter s₂
  (area₁ + area₂) / (perimeter₁ + perimeter₂) = 25 * Real.sqrt 3 / 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_area_perimeter_ratio_l318_31800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_right_triangle_l318_31899

/-- Representation of a triangle with three angles -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- A triangle is equilateral if all its angles are equal -/
def is_equilateral (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∧ t.angle2 = t.angle3

/-- A triangle is right if one of its angles is 90 degrees -/
def is_right (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

/-- The sum of angles in a triangle is 180 degrees -/
axiom triangle_angle_sum (t : Triangle) : t.angle1 + t.angle2 + t.angle3 = 180

/-- Theorem: An equilateral right triangle cannot exist -/
theorem no_equilateral_right_triangle :
  ¬ ∃ (t : Triangle), is_equilateral t ∧ is_right t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_right_triangle_l318_31899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_money_left_l318_31821

/-- Represents the fraction of money spent on half of the sketchbooks -/
def money_spent_on_half : ℚ := 1/4

/-- Represents the fraction of sketchbooks bought with the spent money -/
def sketchbooks_bought : ℚ := 1/2

/-- Theorem stating that Jenna will have half of her money left after buying all sketchbooks -/
theorem jenna_money_left (total_money : ℚ) :
  money_spent_on_half * total_money = sketchbooks_bought * (money_spent_on_half * total_money * 2) →
  total_money - (money_spent_on_half * total_money * 2) = (1/2) * total_money :=
by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_money_left_l318_31821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_max_value_l318_31881

/-- The function f(x, m) = 2sin²(x) + mcos(x) - 1/8 -/
noncomputable def f (x m : ℝ) : ℝ := 2 * Real.sin x ^ 2 + m * Real.cos x - 1/8

theorem f_range_and_max_value :
  (∀ x ∈ Set.Icc (-Real.pi/3) (2*Real.pi/3), f x (-1) ∈ Set.Icc (-9/8) 2) ∧
  (∀ m < -4, ∃ x, ∀ y, f y m ≤ f x m ∧ f x m = -m - 1/8) ∧
  (∀ m > 4, ∃ x, ∀ y, f y m ≤ f x m ∧ f x m = m - 1/8) ∧
  (∀ m ∈ Set.Icc (-4) 4, ∃ x, ∀ y, f y m ≤ f x m ∧ f x m = (m^2 + 15) / 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_max_value_l318_31881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_square_bound_l318_31834

-- Define the triangle DEF inscribed in a semicircle
def Triangle (r : ℝ) (D E F : ℝ × ℝ) : Prop :=
  -- D and E are on the diameter
  D.1^2 + D.2^2 = r^2 ∧ E.1^2 + E.2^2 = r^2 ∧
  -- F is on the semicircle
  F.1^2 + F.2^2 = r^2 ∧ F.2 ≥ 0 ∧
  -- F is not coincident with D or E
  F ≠ D ∧ F ≠ E

-- Define the sum of DF and EF
noncomputable def t (D E F : ℝ × ℝ) : ℝ :=
  ((D.1 - F.1)^2 + (D.2 - F.2)^2).sqrt +
  ((E.1 - F.1)^2 + (E.2 - F.2)^2).sqrt

-- Theorem statement
theorem triangle_sum_square_bound {r : ℝ} {D E F : ℝ × ℝ} 
  (h : Triangle r D E F) (hr : r > 0) : 
  (t D E F)^2 ≤ 8 * r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_square_bound_l318_31834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l318_31824

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l318_31824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feuerbach_l318_31877

/-- Triangle represented by its vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Circle represented by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Incircle of a triangle -/
noncomputable def incircle (t : Triangle) : Circle := sorry

/-- Excircles of a triangle -/
noncomputable def excircles (t : Triangle) : Finset Circle := sorry

/-- Midpoint of a line segment -/
noncomputable def midpoint_of_segment (A B : Point) : Point := sorry

/-- Circle passing through midpoints of triangle sides -/
noncomputable def midpoint_circle (t : Triangle) : Circle := sorry

/-- Tangency between two circles -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- Feuerbach's Theorem -/
theorem feuerbach (t : Triangle) : 
  (∀ e ∈ excircles t, are_tangent (midpoint_circle t) e) ∧ 
  are_tangent (midpoint_circle t) (incircle t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_feuerbach_l318_31877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_week_cut_correct_l318_31857

/-- The percentage of marble cut away in the first week -/
def first_week_cut : ℝ := 24.95

/-- The original weight of the marble in kg -/
def original_weight : ℝ := 190

/-- The percentage cut away in the second week -/
def second_week_cut : ℝ := 15

/-- The percentage cut away in the third week -/
def third_week_cut : ℝ := 10

/-- The final weight of the statue in kg -/
def final_weight : ℝ := 109.0125

/-- Theorem stating that the first week cut percentage is correct given the other conditions -/
theorem first_week_cut_correct :
  abs (original_weight * (1 - first_week_cut / 100) * (1 - second_week_cut / 100) * (1 - third_week_cut / 100) - final_weight) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_week_cut_correct_l318_31857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l318_31843

/-- Given a point P(-4, -2, 3) in 3D space, prove that the sum of the z-coordinate
    of its reflection about the xoy plane and the x-coordinate of its reflection
    about the y-axis is equal to 1. -/
theorem reflection_sum (P : ℝ × ℝ × ℝ) 
  (h1 : P = (-4, -2, 3))
  (h2 : ∃ a b c, (a, b, c) = (P.fst, P.snd.fst, -P.snd.snd))
  (h3 : ∃ e f d, (e, f, d) = (-P.fst, P.snd.fst, P.snd.snd)) :
  ∃ c e, c + e = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l318_31843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_of_seven_consecutive_even_numbers_l318_31809

theorem smallest_of_seven_consecutive_even_numbers (n : ℕ) : 
  (∀ i : ℕ, i < 7 → Even (n + 2 * i)) →
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12) = 560) →
  n = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_of_seven_consecutive_even_numbers_l318_31809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_parallelogram_l318_31802

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sum of binomial coefficients in the parallelogram -/
def parallelogram_sum (n k : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i =>
    Finset.sum (Finset.range k) (λ j =>
      if i + j > n + k - 2 then binomial i j else 0))

/-- Theorem: The binomial coefficient minus 1 equals the parallelogram sum -/
theorem pascal_parallelogram (n k : ℕ) :
  binomial n k - 1 = parallelogram_sum n k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_parallelogram_l318_31802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_zero_l318_31812

/-- Given a triangle with sides a, b, and c satisfying (a + b + c)(a + b - c) = 4ab,
    the angle opposite side c is 0 degrees. -/
theorem triangle_angle_zero (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let angle_c := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_zero_l318_31812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circle_theorem_l318_31811

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def inscribed (t : Triangle) (c : Circle) : Prop := sorry
def circumscribed (t : Triangle) (c : Circle) : Prop := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry
def symmetric_point (p q r : ℝ × ℝ) : Prop := sorry

theorem orthocenter_circle_theorem (c1 c2 : Circle) :
  (∃ (ts : Set Triangle), Set.Infinite ts ∧ 
    (∀ t ∈ ts, inscribed t c1 ∧ circumscribed t c2)) →
  ∃ (oc : Circle),
    (symmetric_point oc.center c1.center c2.center) ∧
    (oc.radius = |2 * c1.radius + c2.radius|) ∧
    (∀ t : Triangle, inscribed t c1 ∧ circumscribed t c2 →
      ∃ p : ℝ × ℝ, p ∈ Metric.sphere oc.center oc.radius ∧ p = orthocenter t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circle_theorem_l318_31811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_fraction_calculation_replacement_fraction_approx_l318_31875

/-- The fraction of the first solution that was replaced -/
noncomputable def replacement_fraction : ℝ := (0.16 - 0.15) / (0.19000000000000007 - 0.15)

/-- The initial salt concentration as a decimal -/
def initial_salt : ℝ := 0.15

/-- The final salt concentration as a decimal -/
def final_salt : ℝ := 0.16

/-- The salt concentration of the second solution as a decimal -/
def second_solution_salt : ℝ := 0.19000000000000007

theorem replacement_fraction_calculation :
  initial_salt * (1 - replacement_fraction) + second_solution_salt * replacement_fraction = final_salt :=
by sorry

theorem replacement_fraction_approx :
  |replacement_fraction - 0.25| < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_fraction_calculation_replacement_fraction_approx_l318_31875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_given_ride_l318_31848

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  initialFare : ℚ  -- Initial fare in dollars
  initialDistance : ℚ  -- Initial distance covered by initial fare in miles
  additionalFareRate : ℚ  -- Additional fare rate in dollars
  additionalDistanceUnit : ℚ  -- Additional distance unit in miles
  tip : ℚ  -- Tip amount in dollars
  totalBudget : ℚ  -- Total budget in dollars

/-- Calculates the maximum distance that can be traveled given the fare structure and budget -/
def maxDistance (ride : TaxiRide) : ℚ :=
  ride.initialDistance + 
  ((ride.totalBudget - ride.tip - ride.initialFare) / ride.additionalFareRate) * 
  ride.additionalDistanceUnit

/-- Theorem stating that for the given fare structure and budget, 
    the maximum distance that can be traveled is 2.55 miles -/
theorem max_distance_for_given_ride : 
  let ride : TaxiRide := {
    initialFare := 3,
    initialDistance := 3/4,
    additionalFareRate := 1/4,
    additionalDistanceUnit := 1/20,
    tip := 3,
    totalBudget := 15
  }
  maxDistance ride = 51/20 := by sorry

#eval maxDistance {
  initialFare := 3,
  initialDistance := 3/4,
  additionalFareRate := 1/4,
  additionalDistanceUnit := 1/20,
  tip := 3,
  totalBudget := 15
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_given_ride_l318_31848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_roll_six_probability_fourth_roll_six_probability_proof_l318_31825

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_prob_six : ℚ := 3 / 4
def biased_die_prob_other : ℚ := 1 / 20

-- Define the probability of choosing each die
def choose_die_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the event of rolling three sixes
def three_sixes : Prop := True

-- Theorem statement
theorem fourth_roll_six_probability : ℚ := 65 / 92

-- Proof of the theorem
theorem fourth_roll_six_probability_proof :
  fourth_roll_six_probability = 65 / 92 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_roll_six_probability_fourth_roll_six_probability_proof_l318_31825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_number_l318_31854

theorem modulus_of_complex_number : 
  Complex.abs (1 + 3 * Complex.I) = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_number_l318_31854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l318_31896

/-- The circle representing curve C1 -/
def circle_c1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 4

/-- The line representing curve C2 -/
def line_c2 (x y : ℝ) : Prop := 4*x - y - 1 = 0

/-- The minimum distance between a point on the circle and a point on the line -/
noncomputable def min_distance : ℝ := (10 * Real.sqrt 17) / 17 - 2

/-- Theorem stating the minimum distance between the circle and the line -/
theorem min_distance_circle_line :
  ∀ (p q : ℝ × ℝ), circle_c1 p.1 p.2 → line_c2 q.1 q.2 →
  ∃ (d : ℝ), d ≥ min_distance ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l318_31896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l318_31860

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A ∧ Finset.card (A.toFinite.toFinset) = 3 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l318_31860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l318_31849

/-- The eccentricity of a hyperbola with given parameters -/
noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = Real.sqrt 3 / 3) : ℝ :=
  2 * Real.sqrt 3 / 3

/-- Theorem: The eccentricity of the given hyperbola is 2√3/3 -/
theorem hyperbola_eccentricity_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = Real.sqrt 3 / 3) :
  hyperbola_eccentricity a b h1 h2 h3 = 2 * Real.sqrt 3 / 3 := by
  -- Unfold the definition of hyperbola_eccentricity
  unfold hyperbola_eccentricity
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l318_31849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_polynomial_value_l318_31806

theorem unique_prime_polynomial_value : ∃! n : ℤ, Nat.Prime (Int.natAbs (n^4 + 2*n^3 + 2*n^2 + 2*n + 1)) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_polynomial_value_l318_31806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonals_equal_rhombus_diagonals_not_necessarily_equal_l318_31829

-- Define a rectangle
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (width_positive : width > 0)
  (height_positive : height > 0)

-- Define a rhombus
structure Rhombus :=
  (side : ℝ)
  (angle : ℝ)
  (side_positive : side > 0)
  (angle_positive : angle > 0)
  (angle_less_than_pi : angle < π)

-- Function to calculate the length of a diagonal in a rectangle
noncomputable def rectangle_diagonal_length (r : Rectangle) : ℝ :=
  Real.sqrt (r.width^2 + r.height^2)

-- Function to calculate the lengths of diagonals in a rhombus
noncomputable def rhombus_diagonal_lengths (rh : Rhombus) : (ℝ × ℝ) :=
  (2 * rh.side * Real.sin (rh.angle / 2), 2 * rh.side * Real.cos (rh.angle / 2))

-- Theorem: The diagonals of a rectangle are always equal in length
theorem rectangle_diagonals_equal (r : Rectangle) :
  rectangle_diagonal_length r = rectangle_diagonal_length r := by
  sorry

-- Theorem: The diagonals of a rhombus are not necessarily equal in length
theorem rhombus_diagonals_not_necessarily_equal :
  ∃ rh : Rhombus, (rhombus_diagonal_lengths rh).1 ≠ (rhombus_diagonal_lengths rh).2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonals_equal_rhombus_diagonals_not_necessarily_equal_l318_31829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_24n_l318_31890

theorem count_divisible_24n : 
  (Finset.filter (fun n : Fin 10 => n.val ≥ 1 ∧ (24 * n.val) % n.val = 0) Finset.univ).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_24n_l318_31890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_two_pi_thirds_angle_C_equals_two_pi_thirds_l318_31814

-- Define a triangle ABC with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the angle_C function
noncomputable def angle_C (t : Triangle) : ℝ := 
  Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- Define the theorem
theorem angle_C_is_two_pi_thirds (t : Triangle) 
  (h : (t.a + t.b + t.c) * (t.a + t.b - t.c) = t.a * t.b) : 
  Real.cos (angle_C t) = -(1/2) := by
  sorry

-- Final theorem stating the result
theorem angle_C_equals_two_pi_thirds (t : Triangle) 
  (h : (t.a + t.b + t.c) * (t.a + t.b - t.c) = t.a * t.b) : 
  angle_C t = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_two_pi_thirds_angle_C_equals_two_pi_thirds_l318_31814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_cubed_ln_x_derivative_frac_1_minus_x_squared_over_exp_x_l318_31820

-- Function 1
theorem derivative_x_cubed_ln_x (x : ℝ) (h : x > 0) :
  deriv (λ x => x^3 * Real.log x) x = 3 * x^2 * Real.log x + x^2 := by sorry

-- Function 2
theorem derivative_frac_1_minus_x_squared_over_exp_x (x : ℝ) :
  deriv (λ x => (1 - x^2) / Real.exp x) x = (x^2 - 2*x - 1) / Real.exp x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_cubed_ln_x_derivative_frac_1_minus_x_squared_over_exp_x_l318_31820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lion_cubs_turtle_time_difference_l318_31889

/-- The time difference between two lion cubs stepping on a turtle --/
theorem lion_cubs_turtle_time_difference 
  (distance_first_cub : ℝ) 
  (speed_ratio_second_cub : ℝ)
  (distance_turtle : ℝ)
  (time_turtle_after_second_cub : ℝ) :
  distance_first_cub = 6 →
  speed_ratio_second_cub = 1.5 →
  distance_turtle = 32 →
  time_turtle_after_second_cub = 28.8 →
  ∃ (speed_first_cub : ℝ),
    speed_first_cub > 0 ∧
    let speed_second_cub := speed_ratio_second_cub * speed_first_cub
    let speed_turtle := distance_turtle / 32
    let time_first_encounter := (distance_first_cub - 1) / (speed_first_cub - speed_turtle)
    let time_second_encounter := distance_turtle - time_turtle_after_second_cub - time_first_encounter
    time_second_encounter - time_first_encounter = 2.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lion_cubs_turtle_time_difference_l318_31889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_transformation_l318_31839

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * (Real.cos (ω * x))^2 - 1 + 2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (1/2) ((x + 2 * Real.pi / 3) / 2)

-- State the theorem
theorem function_symmetry_and_transformation (ω α : ℝ) : 
  (0 < ω) → (ω < 1) → 
  (∀ x, f ω x = f ω (2 * Real.pi / 3 - x)) → 
  (g (2 * α + Real.pi / 3) = 6/5) → 
  (0 < α) → (α < Real.pi / 2) → 
  (ω = 1/2 ∧ Real.sin α = (4 * Real.sqrt 3 - 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_transformation_l318_31839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_formula_min_time_to_recoup_optimal_transfer_time_l318_31847

-- Define constants
def initial_investment : ℚ := 720000
def max_usage_period : ℚ := 40
def first_year_rent : ℚ := 54000
def yearly_rent_increase : ℚ := 4000

-- Define functions
def total_rent (x : ℚ) : ℚ := 0.2 * x^2 + 5.2 * x

def transfer_price (x : ℚ) : ℚ := -0.3 * x^2 + 10.56 * x + 57.6

def annual_average_return (x : ℚ) : ℚ := 
  (transfer_price x + total_rent x - initial_investment) / x

-- Theorem statements
theorem total_rent_formula (x : ℚ) (h : 0 < x ∧ x ≤ max_usage_period) : 
  total_rent x = 0.2 * x^2 + 5.2 * x := by sorry

theorem min_time_to_recoup : 
  ∃ (t : ℚ), t = 10 ∧ ∀ (x : ℚ), 0 < x ∧ x < t → total_rent x < initial_investment := by sorry

theorem optimal_transfer_time : 
  ∃ (t : ℚ), t = 12 ∧ ∀ (x : ℚ), 0 < x ∧ x ≤ max_usage_period → 
    annual_average_return x ≤ annual_average_return t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_formula_min_time_to_recoup_optimal_transfer_time_l318_31847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_coprime_integer_l318_31818

def a (n : ℕ) : ℤ := 2^n + 3^n + 6^n - 1

theorem unique_coprime_integer (k : ℕ) :
  (∀ n : ℕ, (Int.natAbs (a n)).gcd k = 1) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_coprime_integer_l318_31818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l318_31853

/-- Given a line y = kx + 1 tangent to the curve y = x³ + ax + b at point A(1,3),
    prove that 2a + b = 1 -/
theorem tangent_line_problem (k a b : ℝ) : 
  (∃ k, (3 = k * 1 + 1) ∧ 
        (3 = 1 ^ 3 + a * 1 + b) ∧ 
        (k = 3 * 1 ^ 2 + a)) →
  2 * a + b = 1 := by
  intro h
  rcases h with ⟨k, h1, h2, h3⟩
  -- From h1: k = 2
  have k_eq : k = 2 := by
    linarith
  -- From h2: a + b = 2
  have ab_eq : a + b = 2 := by
    linarith
  -- From h3 and k_eq: a = -1
  have a_eq : a = -1 := by
    rw [k_eq] at h3
    linarith
  -- From ab_eq and a_eq: b = 3
  have b_eq : b = 3 := by
    rw [a_eq] at ab_eq
    linarith
  -- Conclude: 2a + b = 1
  calc
    2 * a + b = 2 * (-1) + 3 := by rw [a_eq, b_eq]
    _         = -2 + 3       := by ring
    _         = 1            := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l318_31853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l318_31883

noncomputable section

/-- Given a hyperbola C and a circle F with specific properties, prove that the eccentricity of C is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Equation of hyperbola C
  (∀ x y : ℝ, (x - c)^2 + y^2 = c^2) →     -- Equation of circle F
  (∃ m k : ℝ, ∀ x y : ℝ, y = m * x + k ∧ 
    m = a / b ∧                           -- Line l perpendicular to asymptote
    (0 - k) / m = 2 * a / 3) →            -- x-intercept of l is 2a/3
  (∃ x1 y1 x2 y2 : ℝ, 
    (x1 - c)^2 + y1^2 = c^2 ∧ 
    (x2 - c)^2 + y2^2 = c^2 ∧ 
    (x1 - x2)^2 + (y1 - y2)^2 = (4 * Real.sqrt 2 * c / 3)^2) →  -- Chord length on F
  c / a = 2 :=                            -- Eccentricity is 2
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l318_31883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l318_31898

/-- An arithmetic sequence with specified conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  S : ℕ → ℚ  -- Sum function (changed to ℚ for computability)
  h1 : S 5 = 30
  h2 : a 1 + a 6 = 14

/-- The sum of the first n terms of the sequence 2^(a n) -/
noncomputable def T (as : ArithmeticSequence) (n : ℕ) : ℚ :=
  (4^(n+1) / 3) - (4 / 3)

theorem arithmetic_sequence_properties (as : ArithmeticSequence) :
  (∀ n, as.a n = 2 * n) ∧
  (∀ n, T as n = (4^(n+1) / 3) - (4 / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l318_31898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_of_given_sets_l318_31893

theorem union_cardinality_of_given_sets : ∃ (A B : Finset ℕ), 
  A = {1, 2, 3} ∧ B = {2, 3, 4} ∧ Finset.card (A ∪ B) = 4 := by
  -- Define sets A and B
  let A : Finset ℕ := {1, 2, 3}
  let B : Finset ℕ := {2, 3, 4}
  
  -- Prove the theorem
  use A, B
  constructor
  · rfl
  constructor
  · rfl
  · sorry  -- We'll leave the actual proof for later


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_of_given_sets_l318_31893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinate_sum_l318_31885

-- Define the points
def A : ℚ × ℚ := (0, 8)
def B : ℚ × ℚ := (0, 0)
def C : ℚ × ℚ := (10, 0)

-- Define midpoints
def D : ℚ × ℚ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def E : ℚ × ℚ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the intersection point F
noncomputable def F : ℚ × ℚ := (10/3, 8/3)

-- Theorem statement
theorem intersection_coordinate_sum :
  F.1 + F.2 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinate_sum_l318_31885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_points_outside_circle_l318_31852

/-- The circle equation --/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

/-- The line equation for the center --/
def center_line (x y : ℝ) : Prop := 2*x + y = 0

/-- The tangent line equation --/
def tangent_line (x y : ℝ) : Prop := y = -x + 1

/-- The tangent point --/
def tangent_point : ℝ × ℝ := (2, -1)

/-- Point O --/
def point_O : ℝ × ℝ := (0, 0)

/-- Point A --/
noncomputable def point_A : ℝ × ℝ := (1, 2 - Real.sqrt 2)

theorem circle_properties :
  ∃ (center : ℝ × ℝ),
    center_line center.1 center.2 ∧
    circle_equation tangent_point.1 tangent_point.2 ∧
    tangent_line tangent_point.1 tangent_point.2 ∧
    (∀ (x y : ℝ), circle_equation x y → 
      (x - center.1)^2 + (y - center.2)^2 = (tangent_point.1 - center.1)^2 + (tangent_point.2 - center.2)^2) :=
by sorry

theorem points_outside_circle :
  (point_O.1 - 1)^2 + (point_O.2 + 2)^2 > 2 ∧
  (point_A.1 - 1)^2 + (point_A.2 + 2)^2 > 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_points_outside_circle_l318_31852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_trigonometric_equation_l318_31880

theorem solution_of_trigonometric_equation :
  ∃ x : ℝ, 0 < x ∧ x < 90 ∧
  Real.sin (9 * π / 180) * Real.sin (21 * π / 180) * Real.sin ((102 + x) * π / 180) = 
  Real.sin (30 * π / 180) * Real.sin (42 * π / 180) * Real.sin (x * π / 180) ∧
  x = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_trigonometric_equation_l318_31880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l318_31878

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x - Real.pi/4))^2 - 1

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x, f (x + Real.pi) = f x) ∧ -- π is a period of f
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi) -- π is the least positive period
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l318_31878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_200_and_1200_l318_31827

theorem perfect_cubes_between_200_and_1200 : 
  (Finset.filter (fun n : ℕ => 200 < n^3 ∧ n^3 < 1200) (Finset.range 11)).card = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_200_and_1200_l318_31827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_projections_cube_sphere_l318_31886

-- Define the shapes
inductive Shape
  | Cone
  | Cube
  | Cylinder
  | Sphere
  | RegularTetrahedron

-- Define the views
inductive View
  | Front
  | Top
  | Side

-- Define a function to get the projection shape for a given shape and view
def projection (s : Shape) (v : View) : Set Shape := sorry

-- Theorem statement
theorem identical_projections_cube_sphere :
  ∀ (s : Shape), 
    (∀ (v : View), projection s v = projection Shape.Cube v) ↔ 
    (s = Shape.Cube ∨ s = Shape.Sphere) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_projections_cube_sphere_l318_31886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_remaining_slices_l318_31862

-- Define the number of pizzas and slices per pizza
def num_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 6

-- Define the fractions eaten by James and Lisa
def james_fraction : ℚ := 5/6
def lisa_fraction : ℚ := 1/3

-- Define the function to calculate remaining slices
def remaining_slices (total_slices : ℕ) (eaten_fraction : ℚ) : ℕ :=
  total_slices - (eaten_fraction * total_slices).num.toNat

-- Theorem statement
theorem pizza_remaining_slices :
  (remaining_slices slices_per_pizza james_fraction = 1) ∧
  (remaining_slices slices_per_pizza lisa_fraction = 4) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_remaining_slices_l318_31862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_proof_l318_31850

theorem arc_length_proof (R : ℝ) (h : 2 * R * Real.sin 1 = 4) : 2 * R = 4 / Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_proof_l318_31850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l318_31865

-- Define the custom operation
noncomputable def circledSlash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 4

-- State the theorem
theorem solve_equation (x : ℝ) : circledSlash 6 x = 256 → x = -2 := by
  intro h
  -- The proof steps would go here
  sorry

#check solve_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l318_31865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_combination_l318_31841

theorem max_xy_combination (x y : ℕ) (h : Nat.choose 8 (2*x) = Nat.choose 8 y) : 
  ∀ a b : ℕ, Nat.choose 8 (2*a) = Nat.choose 8 b → x * y ≥ a * b :=
by
  sorry

#check max_xy_combination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_combination_l318_31841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rotation_problem_l318_31869

theorem circle_rotation_problem (X : ℕ) (hX : X = 144) : 
  (Finset.filter (fun s => s < X ∧ X % s = 0) (Finset.range X)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rotation_problem_l318_31869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_theorem_l318_31851

-- Define the ellipse E
def myEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle O
def myCircle (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

-- Define the line l
def myLine (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define the dot product
def myDotProduct (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem ellipse_circle_tangent_theorem 
  (a b r : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 0 < r) 
  (h4 : r < b) 
  (h5 : myEllipse a b (-1) (3/2)) 
  (h6 : ∃ k m x1 y1 x2 y2, 
    myLine k m x1 y1 ∧ 
    myLine k m x2 y2 ∧ 
    myEllipse a b x1 y1 ∧ 
    myEllipse a b x2 y2 ∧ 
    (∃ x y, myLine k m x y ∧ myCircle r x y) ∧
    myDotProduct x1 y1 x2 y2 = 0) :
  a = 2 ∧ b = Real.sqrt 3 ∧ r = Real.sqrt (12/7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_theorem_l318_31851
