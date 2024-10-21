import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_statements_l820_82005

/-- Represents a statement about linear regression -/
inductive LinearRegressionStatement
  | Temporality
  | SampleRangeAffectsApplicability
  | PrecisePredictedValues
  | AlwaysYieldsEquation

/-- Determines if a given statement about linear regression is correct -/
def is_correct (statement : LinearRegressionStatement) : Bool :=
  match statement with
  | .Temporality => true
  | .SampleRangeAffectsApplicability => true
  | .PrecisePredictedValues => false
  | .AlwaysYieldsEquation => false

/-- The list of all statements about linear regression -/
def all_statements : List LinearRegressionStatement :=
  [.Temporality, .SampleRangeAffectsApplicability, .PrecisePredictedValues, .AlwaysYieldsEquation]

theorem two_correct_statements :
  (all_statements.filter is_correct).length = 2 := by
  -- Proof goes here
  sorry

#eval (all_statements.filter is_correct).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_statements_l820_82005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_club_recipes_l820_82027

/-- Calculates the number of full recipes needed for a Math Club event --/
def recipes_needed (total_students : ℕ) (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) (attendance_rate : ℚ) : ℕ :=
  let expected_attendance := (total_students : ℚ) * attendance_rate
  let total_cookies_needed := expected_attendance * cookies_per_student
  (total_cookies_needed / cookies_per_recipe).ceil.toNat

/-- Proves that 14 full recipes are needed for the Math Club event --/
theorem math_club_recipes :
  recipes_needed 150 3 20 (3/5) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_club_recipes_l820_82027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_problem_l820_82037

theorem linear_equation_problem :
  ∀ (a b c : ℝ),
  (∃ x y : ℤ, a * x + b * y = c) →
  (a * (-1) + b * 0 = c) →
  (a * 0 + b * 1 = c) →
  (∀ x y : ℝ, a * x + b * y = c ↔ x - y = -1) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = -1 ∧ 2*x - y = -m) ↔ m < 1) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 3 → x + 1 < 2*x + m) ↔ m ≥ -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_problem_l820_82037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_lines_concurrent_l820_82098

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point where two circles are tangent -/
noncomputable def TangencyPoint (c1 c2 : Circle) : ℝ × ℝ := sorry

/-- Represents a line connecting two tangency points -/
structure TangencyLine where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if three lines are concurrent -/
def areConcurrent (l1 l2 l3 : TangencyLine) : Prop := sorry

/-- Main theorem: Given four circles in a plane where any two circles are tangent to each other,
    the lines connecting the tangency points of different pairs of circles are concurrent -/
theorem tangency_lines_concurrent (c1 c2 c3 c4 : Circle) 
  (h : ∀ (i j : Fin 4), i ≠ j → ∃ (p : ℝ × ℝ), p = TangencyPoint (match i with
    | ⟨0, _⟩ => c1
    | ⟨1, _⟩ => c2
    | ⟨2, _⟩ => c3
    | ⟨3, _⟩ => c4
    | _ => c1 -- This case should never occur due to Fin 4
  ) (match j with
    | ⟨0, _⟩ => c1
    | ⟨1, _⟩ => c2
    | ⟨2, _⟩ => c3
    | ⟨3, _⟩ => c4
    | _ => c1 -- This case should never occur due to Fin 4
  )) :
  ∃ (l1 l2 l3 : TangencyLine), areConcurrent l1 l2 l3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_lines_concurrent_l820_82098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_center_of_gravity_displacement_rod_center_of_gravity_displacement_80cm_l820_82056

/-- Represents a uniform straight rod -/
structure Rod where
  length : ℝ
  uniformity : Unit

/-- Calculates the displacement of the center of gravity when a piece is cut from a rod -/
noncomputable def centerOfGravityDisplacement (rod : Rod) (cutLength : ℝ) : ℝ :=
  cutLength / 2

/-- Theorem: The displacement of the center of gravity is half the length of the cut piece -/
theorem rod_center_of_gravity_displacement (rod : Rod) (s : ℝ) 
  (h₁ : s > 0) (h₂ : s < rod.length) :
  centerOfGravityDisplacement rod s = s / 2 := by
  sorry

/-- Corollary: For a rod with a piece of 80 cm cut off, the center of gravity displaces by 40 cm -/
theorem rod_center_of_gravity_displacement_80cm (rod : Rod) 
  (h : rod.length > 80) :
  centerOfGravityDisplacement rod 80 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_center_of_gravity_displacement_rod_center_of_gravity_displacement_80cm_l820_82056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_radius_l820_82057

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem sphere_intersection_radius (s : Sphere) 
    (h1 : distance s.center ⟨3, 0, 5⟩ = s.radius)
    (h2 : s.center.x = 3)
    (h3 : s.center.z = 5)
    (h4 : distance ⟨3, 0, 5⟩ ⟨3, 2, 5⟩ = 2) : 
    distance s.center ⟨0, s.center.y, 5⟩ = 3 := by
  sorry

#check sphere_intersection_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_radius_l820_82057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l820_82095

noncomputable section

-- Define the ellipse and its properties
def Ellipse (a b : ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧ F₁.1 < F₂.1 ∧ F₁.2 = F₂.2 ∧ F₂.1 - F₁.1 = 2 * Real.sqrt (a^2 - b^2)

-- Define a point on the ellipse
def OnEllipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the midpoint of a line segment
def Midpoint (M Q F₂ : ℝ × ℝ) : Prop :=
  M = ((Q.1 + F₂.1) / 2, (Q.2 + F₂.2) / 2)

-- Theorem statement
theorem ellipse_properties 
  (a b : ℝ) (F₁ F₂ Q M : ℝ × ℝ) 
  (h_ellipse : Ellipse a b F₁ F₂)
  (h_Q_on_ellipse : OnEllipse Q.1 Q.2 a b)
  (h_Q : Q = (Real.sqrt 2, 1))
  (h_M_on_y : M.1 = 0)
  (h_midpoint : Midpoint M Q F₂) :
  ∃ (P : ℝ × ℝ),
    OnEllipse P.1 P.2 a b ∧ 
    (P.1^2 / 4 + P.2^2 / 2 = 1) ∧
    (let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2);
     let d₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2);
     d₁ * d₂ / 2 = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l820_82095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_call_cost_l820_82014

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Define the cost function
noncomputable def g (t : ℝ) : ℝ :=
  1.06 * (0.75 * (ceiling t : ℝ) + 1)

-- State the theorem
theorem phone_call_cost (t : ℝ) (h : t > 0) :
  g 5.5 = 5.83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_call_cost_l820_82014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l820_82062

noncomputable section

def f (x : ℝ) := 2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  -- The minimum positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The interval of monotonic increase
  (∀ (k : ℤ), ∀ (x y : ℝ),
    k * Real.pi - Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 5 * Real.pi / 12 → f x < f y) ∧
  -- Range of m
  (∀ (m : ℝ),
    (∃ (x : ℝ), Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x - m = 2) ↔ 0 ≤ m ∧ m ≤ 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l820_82062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_sequence_properties_l820_82068

/-- Definition of a T point sequence -/
noncomputable def is_T_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) > a (n + 1) - a n

/-- The specific sequence given in the problem -/
noncomputable def specific_sequence (n : ℕ) : ℝ := 1 / n

/-- Theorem stating the properties of T point sequences -/
theorem T_sequence_properties 
  (a : ℕ → ℝ) 
  (h_T : is_T_sequence a) 
  (h_a2_gt_a1 : a 2 > a 1) :
  (is_T_sequence specific_sequence) ∧ 
  (∀ k : ℕ, (a (k + 2) - a (k + 1))^2 + (a (k + 1) - a k)^2 < 1) ∧
  (∀ k l m : ℕ, k < l → l < m → 
    (a (m + k) - a l) > (a m - a (l - k))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_sequence_properties_l820_82068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_from_line_l820_82067

/-- The distance from a point (x, y) to the line ax + by + c = 0 is given by
    |ax + by + c| / √(a² + b²) --/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

theorem point_distance_from_line (a : ℝ) :
  distance_point_to_line a 6 3 (-4) 2 = 4 ↔ a = 2 ∨ a = 46/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_from_line_l820_82067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_rate_l820_82077

/-- Proves that the population growth rate in the first year is 25% given the initial and final populations and the second year's decrease rate. -/
theorem population_growth_rate (initial_population final_population : ℕ) (second_year_decrease : ℚ) :
  initial_population = 415600 →
  final_population = 363650 →
  second_year_decrease = 30 / 100 →
  ∃ (first_year_increase : ℚ),
    first_year_increase = 25 / 100 ∧
    final_population = (initial_population : ℚ) * (1 + first_year_increase) * (1 - second_year_decrease) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_rate_l820_82077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_f_l820_82010

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (x - m)^2 + (Real.log x - 2*m)^2

-- Theorem statement
theorem minimize_f :
  ∃ (m : ℝ), ∀ (x : ℝ), x > 0 → f x m ≤ f x (1/10 - 2/5 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_f_l820_82010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cyclic_polynomial_permutation_l820_82041

theorem no_cyclic_polynomial_permutation 
  (a b c : ℤ) (P : ℤ → ℤ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∀ x : ℤ, ∃ Q : Polynomial ℤ, P x = Q.eval x) →
  ¬(P a = b ∧ P b = c ∧ P c = a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cyclic_polynomial_permutation_l820_82041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_multiple_of_11_l820_82046

/-- The alternating sum of digits for a five-digit number abcde -/
def alternating_sum_of_digits (n : ℕ) : ℤ :=
  let digits := n.digits 10
  match digits with
  | [a, b, c, d, e] => (a + c + e) - (b + d)
  | _ => 0

/-- A number is a multiple of 11 if and only if the alternating sum of its digits is divisible by 11 -/
axiom multiple_of_11 (n : ℕ) : 
  11 ∣ n ↔ 11 ∣ (alternating_sum_of_digits n)

/-- The five-digit number formed by 5678d -/
def number (d : ℕ) : ℕ := 
  56780 + d

theorem five_digit_multiple_of_11 (d : ℕ) (h : d < 10) : 
  11 ∣ (number d) ↔ d = 2 := by
  sorry

#eval alternating_sum_of_digits (number 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_multiple_of_11_l820_82046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_l820_82063

def A : Set ℤ := {x | ∃ k : ℤ, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2*k}

theorem negation_of_universal : 
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_l820_82063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_eulerian_circuits_l820_82088

/-- An Eulerian circuit in a graph. -/
def EulerianCircuit {V : Type} (G : SimpleGraph V) (c : List V) : Prop :=
  -- Define properties of an Eulerian circuit
  sorry

/-- The number of distinct Eulerian circuits in a graph. -/
def numDistinctEulerianCircuits {V : Type} (G : SimpleGraph V) : ℕ :=
  -- Define how to count distinct Eulerian circuits
  sorry

/-- Theorem: A simple graph cannot have exactly three distinct Eulerian circuits. -/
theorem no_three_eulerian_circuits {V : Type} (G : SimpleGraph V) :
  (∃ c, EulerianCircuit G c) → numDistinctEulerianCircuits G ≠ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_eulerian_circuits_l820_82088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_ratio_in_mixture_l820_82032

/-- Given two jars with different volumes and alcohol-to-water ratios,
    this theorem proves the ratio of alcohol to water in the mixture. -/
theorem alcohol_water_ratio_in_mixture
  (V₁ V₂ r s : ℝ)
  (h₁ : V₁ > 0)
  (h₂ : V₂ > 0)
  (h₃ : r > 0)
  (h₄ : s > 0) :
  (r / (r + 1) * V₁ + s / (s + 1) * V₂) / (1 / (r + 1) * V₁ + 1 / (s + 1) * V₂) =
  (r / (r + 1) * V₁ + s / (s + 1) * V₂) / (1 / (r + 1) * V₁ + 1 / (s + 1) * V₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_ratio_in_mixture_l820_82032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curves_l820_82078

/-- The area bounded by two curves C₁ and C₂ -/
noncomputable def bounded_area (r : ℝ) : ℝ :=
  2 * ∫ x in Set.Icc 0 1, (Real.sqrt (r^2 - x^2) - (2 * x^2) / (x^2 + 1))

/-- The condition for perpendicular tangent lines -/
def perpendicular_tangents (r : ℝ) (x : ℝ) : Prop :=
  (4 * x) / ((x^2 + 1)^2) * (-x / Real.sqrt (r^2 - x^2)) = -1

/-- The main theorem stating the area bounded by the curves -/
theorem area_of_curves (r : ℝ) (hr : r > 0) :
  (∃ x : ℝ, perpendicular_tangents r x) →
  bounded_area r = π / 2 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curves_l820_82078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_properties_l820_82048

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number with digit sum 2013, prove that the next number has digit sum 2005 and is not divisible by 4 -/
theorem next_number_properties (N : ℕ) : 
  (digit_sum N = 2013) → 
  (digit_sum (N + 1) = 2005) ∧ 
  ¬(4 ∣ (N + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_properties_l820_82048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_two_l820_82079

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : lg 4 + 2 * lg 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_two_l820_82079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_4_l820_82090

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The parabola y² = 4x -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- The distance from a point to the line x = -2 -/
def distance_to_line (x : ℝ) : ℝ :=
  |x + 2|

theorem distance_to_focus_is_4 (x y : ℝ) :
  on_parabola x y →
  distance_to_line x = 5 →
  distance x y 1 0 = 4 := by
  sorry

#check distance_to_focus_is_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_4_l820_82090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_prob_less_than_half_l820_82071

/-- The probability of drawing n red apples in a row from a box with initial_red red apples
    and initial_total total apples, where each drawn apple is removed. -/
def prob_n_red_draws (initial_red : ℕ) (initial_total : ℕ) (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ i => (initial_red - i : ℚ) / (initial_total - i))

/-- 6 is the smallest positive integer n such that the probability of drawing
    n red apples in a row from a box initially containing 9 red apples and
    10 total apples, where each drawn apple is removed, is less than 0.5. -/
theorem smallest_n_for_prob_less_than_half :
  (∀ k < 6, prob_n_red_draws 9 10 k ≥ (1/2 : ℚ)) ∧
  prob_n_red_draws 9 10 6 < (1/2 : ℚ) := by
  sorry

#eval prob_n_red_draws 9 10 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_prob_less_than_half_l820_82071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l820_82045

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 is given by
    |ax₀ + by₀ + c| / √(a² + b²) -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- The distance from the point (0, -1) to the line x + 2y - 3 = 0 is √5 -/
theorem distance_to_line : distance_point_to_line 0 (-1) 1 2 (-3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l820_82045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l820_82017

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a = 2 ∧
  t.a + t.b + t.c = 6 ∧
  3 * t.a * (-2 * Real.sin t.B) + 3 * t.b = 0

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = Real.pi/6 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : ℝ) = 6 - 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l820_82017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l820_82023

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (3/2) * Real.sin (ω * x) + (Real.sqrt 3 / 2) * Real.cos (ω * x)

-- Part 1
theorem part1 (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + 3 * Real.pi) = f ω x) (h3 : ∀ T, T > 0 → T < 3 * Real.pi → ∃ x, f ω (x + T) ≠ f ω x) :
  ω = 2/3 := by sorry

-- Part 2
theorem part2 (k : ℤ) :
  StrictMonoOn (f (2/3)) (Set.Icc (-Real.pi + 3 * ↑k * Real.pi) (Real.pi/2 + 3 * ↑k * Real.pi)) := by sorry

-- Part 3
theorem part3 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : f 2 α = 3/2) :
  α = Real.pi/4 ∨ α = Real.pi/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l820_82023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_largest_to_largest_volume_ratio_l820_82051

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a slice of a cone -/
structure ConeSlice where
  topRadius : ℝ
  bottomRadius : ℝ
  height : ℝ

/-- Calculate the volume of a cone slice -/
noncomputable def coneSliceVolume (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.height * (slice.bottomRadius^2 + slice.topRadius^2 + slice.bottomRadius * slice.topRadius)

/-- Theorem: The ratio of the volume of the second-largest piece to the largest piece
    in a right circular cone sliced into five equal-height pieces is 37/187 -/
theorem second_largest_to_largest_volume_ratio
  (cone : RightCircularCone)
  (h1 : cone.height > 0)
  (h2 : cone.baseRadius > 0)
  (slices : Fin 5 → ConeSlice)
  (h3 : ∀ i, (slices i).height = cone.height / 5)
  (h4 : ∀ i, (slices i).bottomRadius = (i + 1 : ℝ) * cone.baseRadius / 5)
  (h5 : ∀ i, (slices i).topRadius = i * cone.baseRadius / 5) :
  let secondLargestVolume := coneSliceVolume (slices 3)
  let largestVolume := coneSliceVolume (slices 4)
  secondLargestVolume / largestVolume = 37 / 187 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_largest_to_largest_volume_ratio_l820_82051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l820_82084

/-- The function f(x) = x² --/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 2 * x

/-- The slope of the tangent line at x=1 --/
def tangent_slope : ℝ := f' 1

/-- The y-intercept of the tangent line --/
noncomputable def y_intercept : ℝ := -1

/-- The x-intercept of the tangent line --/
noncomputable def x_intercept : ℝ := 1/2

/-- The area of the triangle formed by the tangent line and coordinate axes --/
noncomputable def triangle_area : ℝ := (1/2) * x_intercept * 1

theorem tangent_line_triangle_area :
  triangle_area = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l820_82084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l820_82047

noncomputable def I : Set ℝ := Set.univ
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}
def A : Set ℝ := (I \ M) ∩ N

def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}

theorem problem_solution :
  (∀ x, x ∈ ((I \ M) ∩ N) ↔ x = 2) ∧
  (∀ a, B a ∪ A = A ↔ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l820_82047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_solution_trigonometric_equation_l820_82073

theorem greatest_solution_trigonometric_equation :
  ∃ x : ℝ,
    x ∈ Set.Icc 0 (10 * Real.pi) ∧
    |2 * Real.sin x - 1| + |2 * Real.cos (2 * x) - 1| = 0 ∧
    (∀ y ∈ Set.Icc 0 (10 * Real.pi),
      |2 * Real.sin y - 1| + |2 * Real.cos (2 * y) - 1| = 0 → y ≤ x) ∧
    |x - 27.7| < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_solution_trigonometric_equation_l820_82073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_foci_coincide_l820_82007

/-- The squared semi-major axis of the ellipse -/
noncomputable def a_squared_ellipse : ℝ := 25

/-- The squared semi-major axis of the hyperbola -/
noncomputable def a_squared_hyperbola : ℝ := 225 / 36

/-- The squared semi-minor axis of the hyperbola -/
noncomputable def b_squared_hyperbola : ℝ := 144 / 36

/-- The focal distance of the hyperbola -/
noncomputable def c_hyperbola : ℝ := Real.sqrt (a_squared_hyperbola + b_squared_hyperbola)

/-- The theorem stating that if the foci of the ellipse and hyperbola coincide, 
    then the squared semi-minor axis of the ellipse is 14.75 -/
theorem ellipse_hyperbola_foci_coincide : 
  let b_squared_ellipse := a_squared_ellipse - c_hyperbola^2
  b_squared_ellipse = 14.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_foci_coincide_l820_82007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l820_82050

/-- Parabola represented by the equation y^2 = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_line_intersection
  (C : Parabola)
  (l : Line)
  (A B : Point)
  (hC : C.equation = fun x y => y^2 = 2*x)
  (hF : C.focus = (1/2, 0))
  (hl : l.intercept = -l.slope/2)  -- Line passes through focus
  (hA : C.equation A.x A.y ∧ A.y = l.slope * (A.x - 1/2))
  (hB : C.equation B.x B.y ∧ B.y = l.slope * (B.x - 1/2))
  (hDist : distance A ⟨C.focus.1, C.focus.2⟩ = 3 * distance B ⟨C.focus.1, C.focus.2⟩) :
  l.slope = Real.sqrt 3 ∨ l.slope = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l820_82050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_l820_82064

theorem max_sin_a : 
  (∀ a b : Real, Real.cos (2 * a + b) = Real.cos a * Real.cos b - Real.sin a * Real.sin b) → 
  (∃ a : Real, Real.sin a = Real.sqrt 2 / 2 ∧ ∀ a' : Real, Real.sin a' ≤ Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_l820_82064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_tree_problem_l820_82092

theorem bird_tree_problem :
  ∃ (x y z : ℕ), z % 3 = 0 ∧ 29 + x - y + z / 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_tree_problem_l820_82092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_price_increase_l820_82074

/-- The percentage increase in price between two pairs of shoes --/
noncomputable def percentage_increase (price1 price2 : ℝ) : ℝ :=
  (price2 - price1) / price1 * 100

theorem shoe_price_increase (price1 price2 total : ℝ) 
  (h1 : price1 = 22)
  (h2 : price2 > price1)
  (h3 : total = price1 + price2)
  (h4 : total = 55) :
  percentage_increase price1 price2 = 50 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_price_increase_l820_82074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l820_82066

theorem triangle_cosine_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  (Real.sqrt 3 * b - c) * (Real.cos A) = a * (Real.cos C) →
  Real.cos A = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l820_82066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_4_l820_82036

noncomputable section

-- Define the curve C
def curve_C (θ : Real) : Real × Real :=
  (1 + Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the line l
def line_l (θ : Real) (ρ : Real) : Prop :=
  ρ * Real.cos (θ - Real.pi/6) = 3 * Real.sqrt 3

-- Define the ray OT
def ray_OT (θ : Real) (ρ : Real) : Prop :=
  θ = Real.pi/3 ∧ ρ > 0

-- Theorem statement
theorem length_of_AB_is_4 :
  ∃ (ρ_A ρ_B : Real),
    -- A is on curve C and ray OT
    curve_C (Real.pi/3) = (ρ_A * Real.cos (Real.pi/3), ρ_A * Real.sin (Real.pi/3)) ∧
    ray_OT (Real.pi/3) ρ_A ∧
    -- B is on line l and ray OT
    line_l (Real.pi/3) ρ_B ∧
    ray_OT (Real.pi/3) ρ_B ∧
    -- The length of AB is 4
    ρ_B - ρ_A = 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_4_l820_82036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_special_function_l820_82072

/-- The derivative of (cos(ln 7) * sin²(7x)) / (7 * cos(14x)) with respect to x -/
theorem derivative_special_function (x : ℝ) :
  deriv (λ x ↦ (Real.cos (Real.log 7) * Real.sin (7 * x)^2) / (7 * Real.cos (14 * x))) x
  = (Real.cos (Real.log 7) * Real.tan (14 * x)) / Real.cos (14 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_special_function_l820_82072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l820_82019

/-- The time (in seconds) required for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_bridge_passing_time :
  let ε := 0.1  -- Allow for small numerical discrepancy
  |time_to_pass_bridge 360 140 50 - 36| ≤ ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l820_82019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l820_82021

-- Define the function f(x) = x³ + 3ax² + 3bx + c
noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

-- Theorem statement
theorem max_min_difference :
  ∃ a b c : ℝ, 
    (f' a b 2 = 0) ∧ 
    (f' a b 1 = -3) → 
    ∃ max min : ℝ, 
      (∀ x : ℝ, f a b c x ≤ max ∧ f a b c x ≥ min) ∧ 
      max - min = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l820_82021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_b_l820_82075

noncomputable def u (b : ℝ) : ℕ → ℝ
  | 0 => b  -- Adding case for 0
  | 1 => b
  | n + 1 => -1 / (u b n + 2)

theorem u_16_equals_b (b : ℝ) (h : b > 1) : u b 16 = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_b_l820_82075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l820_82016

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

-- Define the derivative of the function
noncomputable def f_deriv (x : ℝ) : ℝ := (x * Real.cos x - Real.sin x) / (x^2)

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 2 * Real.pi
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - 2 * Real.pi * y = 2 * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l820_82016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_primes_product_three_times_sum_l820_82040

theorem three_primes_product_three_times_sum :
  ∃ (a b c : ℕ), Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ a * b * c = 3 * (a + b + c) :=
by
  -- The solution is (2, 3, 5)
  use 2, 3, 5
  constructor
  · exact Nat.prime_two
  constructor
  · exact Nat.prime_three
  constructor
  · exact Nat.prime_five
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_primes_product_three_times_sum_l820_82040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_height_for_equal_volume_and_base_area_l820_82085

/-- Represents a cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- The base area of a cylinder -/
noncomputable def cylinderBaseArea (c : Cylinder) : ℝ := Real.pi * c.radius^2

/-- The volume of a rectangular solid -/
def rectangularSolidVolume (r : RectangularSolid) : ℝ := r.length * r.width * r.height

/-- The base area of a rectangular solid -/
def rectangularSolidBaseArea (r : RectangularSolid) : ℝ := r.length * r.width

theorem equal_height_for_equal_volume_and_base_area
  (c : Cylinder) (r : RectangularSolid)
  (h_vol : cylinderVolume c = rectangularSolidVolume r)
  (h_base : cylinderBaseArea c = rectangularSolidBaseArea r) :
  c.height = r.height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_height_for_equal_volume_and_base_area_l820_82085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l820_82069

/-- The volume of a regular triangular pyramid -/
noncomputable def pyramid_volume (a : ℝ) (α : ℝ) : ℝ :=
  (Real.sqrt 3 * a^3 * Real.sqrt ((1 + 4 * Real.tan α ^ 2) ^ 3)) / (4 * Real.tan α ^ 2)

/-- Theorem: Volume of a regular triangular pyramid given perpendicular length and inclination angle -/
theorem regular_triangular_pyramid_volume 
  (a : ℝ) 
  (α : ℝ) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < Real.pi / 2) :
  ∃ (V : ℝ), V = pyramid_volume a α ∧ V > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l820_82069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_S_over_2n_l820_82060

noncomputable section

/-- An arithmetic sequence with first term 1 and non-zero common difference d -/
def arithmeticSequence (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1 : ℝ) * d

/-- The sum of the first n terms of the arithmetic sequence -/
def S (d : ℝ) (n : ℕ) : ℝ := n * (2 + (n - 1 : ℝ) * d) / 2

/-- The condition that a_2, a_3, a_6 form a geometric sequence -/
def geometricCondition (d : ℝ) : Prop :=
  (arithmeticSequence d 2) ^ 2 = (arithmeticSequence d 1) * (arithmeticSequence d 5)

/-- The main theorem stating the minimum value of S_n / 2^n -/
theorem min_value_S_over_2n (d : ℝ) (h1 : d ≠ 0) (h2 : geometricCondition d) :
  ∃ (n : ℕ), ∀ (m : ℕ), S d n / (2 ^ n) ≤ S d m / (2 ^ m) ∧ S d n / (2 ^ n) = -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_S_over_2n_l820_82060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_transformed_data_l820_82034

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)

noncomputable def variance (a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  (1/5) * (a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 - 80)

noncomputable def transformed_mean (a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  (1/5) * ((2*a₁+1) + (2*a₂+1) + (2*a₃+1) + (2*a₄+1) + (2*a₅+1))

theorem mean_of_transformed_data (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  variance a₁ a₂ a₃ a₄ a₅ ≥ 0 →
  transformed_mean a₁ a₂ a₃ a₄ a₅ = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_transformed_data_l820_82034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimals_l820_82029

/-- Represents a repeating decimal with a single digit repeat. -/
def SingleDigitRepeatingDecimal (whole : ℚ) (r : ℕ) : ℚ :=
  whole + (r : ℚ) / 9

/-- Represents a repeating decimal with a two-digit repeat. -/
def TwoDigitRepeatingDecimal (whole : ℚ) (r : ℕ) : ℚ :=
  whole + (r : ℚ) / 99

theorem product_of_repeating_decimals :
  (SingleDigitRepeatingDecimal 0 3) * (TwoDigitRepeatingDecimal 0 45) = 5 / 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimals_l820_82029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_difference_1998_l820_82025

theorem smallest_factor_difference_1998 :
  (∀ a b : ℕ, a > 0 → b > 0 → a * b = 1998 → |Int.ofNat a - Int.ofNat b| ≥ 17) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a * b = 1998 ∧ |Int.ofNat a - Int.ofNat b| = 17) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_difference_1998_l820_82025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptic_equation_solution_l820_82006

/-- Represents a digit in the given base -/
def Digit (β : ℕ) := Fin β

/-- Represents a number in the given base -/
def Number (β : ℕ) := List (Digit β)

/-- Converts a Number to its decimal representation -/
def toDecimal (β : ℕ) (n : Number β) : ℕ := sorry

/-- Checks if all digits in a Number are distinct -/
def allDistinct (n : Number β) : Prop := sorry

theorem cryptic_equation_solution :
  ∃ (β : ℕ) (G R E A T : Digit β),
    β ≥ 10 ∧
    (∀ (n : Number β), n.length ≤ 5) ∧
    allDistinct [G, R, E, A, T] ∧
    (let great := [G, R, E, A, T]
     let treat := [T, R, E, A, T]
     let rate  := [R, A, T, E]
     let art   := [A, R, T]
     let ag    := [A, G]
     let re    := [R, E]
     toDecimal β great * toDecimal β treat -
     toDecimal β rate * toDecimal β great =
     toDecimal β art * toDecimal β ag * toDecimal β re ∧
     toDecimal β rate = 1233) :=
by sorry

#check cryptic_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptic_equation_solution_l820_82006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_exp_curve_l820_82053

/-- A line y = x + b is tangent to the curve y = e^x if and only if b = 1 -/
theorem tangent_line_to_exp_curve (b : ℝ) : 
  (∃ x₀ : ℝ, (x₀ + b = Real.exp x₀) ∧ 
             (∀ x : ℝ, x + b ≤ Real.exp x) ∧
             (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |Real.exp x - (x + b)| < ε))
  ↔ b = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_exp_curve_l820_82053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_base_angle_value_l820_82031

/-- A regular triangular pyramid with pairwise perpendicular lateral faces -/
structure RegularTriangularPyramid where
  /-- The side length of the base triangle -/
  base_side : ℝ
  /-- The lateral faces are pairwise perpendicular -/
  lateral_faces_perpendicular : True

/-- The angle between a lateral edge and the base plane of the pyramid -/
noncomputable def lateral_edge_base_angle (pyramid : RegularTriangularPyramid) : ℝ :=
  Real.arccos (Real.sqrt 6 / 3)

/-- Theorem: The angle between a lateral edge and the base plane of a regular triangular pyramid
    with pairwise perpendicular lateral faces is arccos(√6/3) -/
theorem lateral_edge_base_angle_value (pyramid : RegularTriangularPyramid) :
  lateral_edge_base_angle pyramid = Real.arccos (Real.sqrt 6 / 3) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_base_angle_value_l820_82031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l820_82080

/-- Distance formula in 2D -/
noncomputable def distance_2d (A B C x₀ y₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Distance formula in 3D -/
noncomputable def distance_3d (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

/-- The distance from (2, 4, 1) to the line x + 2y + 2z + 3 = 0 is 5 -/
theorem distance_to_line : distance_3d 1 2 2 3 2 4 1 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l820_82080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_railway_crossings_l820_82091

/-- The distance between villages A and B along the railway track in kilometers. -/
def railway_length : ℝ := 10

/-- The equation that holds for any point P on the road connecting A and B. -/
def road_equation (BH PH : ℝ) : Prop :=
  20 * BH + 13 * BH^3 + 300 * PH = PH^4 + 32 * PH^2

/-- The number of points where the road crosses the railway track. -/
def crossing_points : ℕ := 10

/-- Theorem stating that the number of crossing points is equal to 10. -/
theorem road_railway_crossings :
  ∃ (road : ℝ → ℝ × ℝ),
    (∀ t, let (BH, PH) := road t; road_equation BH PH) ∧
    (∃ (crossings : Finset ℝ),
      crossings.card = crossing_points ∧
      (∀ t ∈ crossings, (road t).2 = 0) ∧
      (∀ t, (road t).2 = 0 → t ∈ crossings)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_railway_crossings_l820_82091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_disappear_l820_82030

/-- Represents a cell in the grid -/
structure Cell where
  x : ℤ
  y : ℤ

/-- The set of black cells at time t -/
def BlackCells (t : ℕ) : Set Cell := sorry

/-- The initial set of black cells -/
def InitialBlackCells : Set Cell := sorry

/-- The recoloring rule for a cell -/
def recolor (c : Cell) (t : ℕ) : Bool := sorry

/-- The number of initially black cells -/
def n : ℕ := sorry

/-- Theorem stating that all black cells disappear after at most n steps -/
theorem black_cells_disappear :
  (∀ t, t ≥ n → BlackCells t = ∅) ∧
  (∃ t, t ≤ n ∧ BlackCells t = ∅) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_disappear_l820_82030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_45_jogger_speed_is_9_initial_distance_is_200_train_length_is_200_passing_time_is_40_l820_82008

/-- The speed of a train given specific conditions about a jogger and the train's movement. -/
def train_speed (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed : ℝ := 45
  train_speed

/-- The main theorem stating that the train's speed is 45 km/hr under the given conditions. -/
theorem train_speed_is_45 : train_speed 9 200 200 40 = 45 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- The result follows directly from the definition
  rfl

/-- Auxiliary lemmas to represent the given conditions -/
theorem jogger_speed_is_9 : 9 = 9 := by rfl
theorem initial_distance_is_200 : 200 = 200 := by rfl
theorem train_length_is_200 : 200 = 200 := by rfl
theorem passing_time_is_40 : 40 = 40 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_45_jogger_speed_is_9_initial_distance_is_200_train_length_is_200_passing_time_is_40_l820_82008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_smaller_part_l820_82001

theorem fraction_of_smaller_part : 
  ∀ (larger smaller : ℝ),
  larger + smaller = 66 →
  larger = 50 →
  0.40 * larger = (5/8) * smaller + 10 →
  smaller = 16 :=
by
  intros larger smaller sum_eq larger_eq equation
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_smaller_part_l820_82001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_product_of_random_choices_l820_82052

noncomputable def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

noncomputable def random_choice (s : Set ℕ) : ℝ := 
  (Finset.sum (Finset.range 10) (λ i => (i + 1 : ℝ))) / 10

theorem expected_product_of_random_choices :
  (random_choice S) * (random_choice S) = 30.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_product_of_random_choices_l820_82052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_five_l820_82061

def is_valid_number (n : ℕ) : Prop :=
  (100000 ≤ n) ∧ (n < 1000000) ∧
  (∃ a b c d e f : ℕ, n = 100000*a + 10000*b + 1000*c + 100*d + 10*e + f ∧
    ({a, b, c, d, e, f} : Finset ℕ) = {0, 1, 2, 3, 4, 5})

theorem smallest_valid_divisible_by_five :
  ∀ n : ℕ, is_valid_number n ∧ n % 5 = 0 → n ≥ 123450 :=
by
  sorry

#check smallest_valid_divisible_by_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_five_l820_82061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l820_82096

noncomputable section

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The perpendicular line to a given line passing through a point -/
noncomputable def perpendicularLine (l : Line) (p : ℝ × ℝ) : Line :=
  { slope := -1 / l.slope,
    intercept := p.2 - (-1 / l.slope) * p.1 }

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

theorem intersection_of_perpendicular_lines :
  let l1 : Line := { slope := -3, intercept := -2 }
  let p : ℝ × ℝ := (3, -3)
  let l2 := perpendicularLine l1 p
  intersectionPoint l1 l2 = (3/5, -19/5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l820_82096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_equals_radius_l820_82028

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a tangent line
structure TangentLine where
  point : Point
  direction : ℝ × ℝ

-- Function to check if a point is outside the circle
def isOutside (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Function to check if two lines are perpendicular
def arePerpendicular (l1 l2 : TangentLine) : Prop :=
  let (dx1, dy1) := l1.direction
  let (dx2, dy2) := l2.direction
  dx1 * dx2 + dy1 * dy2 = 0

-- Function to calculate the length of a line segment
noncomputable def lineLength (p1 p2 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem tangent_length_equals_radius 
  (c : Circle) 
  (p : Point) 
  (t1 t2 : TangentLine) : 
  c.radius = 10 →
  isOutside c p →
  t1.point = p →
  t2.point = p →
  arePerpendicular t1 t2 →
  ∃ (a b : Point), 
    lineLength p a = 10 ∧
    lineLength p b = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_equals_radius_l820_82028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_triangle_area_equality_l820_82009

/-- Given a circle with center C and radius r, a tangent line AB at point A,
    and another radius CD at angle θ with the horizontal,
    prove that the area of sector ACD equals the area of triangle AMC
    if and only if tan 2θ = 2θ. -/
theorem sector_triangle_area_equality (θ : ℝ) (r : ℝ) :
  let sector_area := (θ * r^2) / 2
  let triangle_area := (r^2 * Real.sin (2 * θ)) / 4
  sector_area = triangle_area ↔ Real.tan (2 * θ) = 2 * θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_triangle_area_equality_l820_82009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_l820_82015

/-- A right triangle ABC where angle C is 90° and the ratio of BC to CA is 3:4 -/
structure RightTriangle where
  A : ℝ  -- Length of side CA
  B : ℝ  -- Length of side BC
  C : ℝ  -- Length of side AB (hypotenuse)
  angle_C_is_right : A^2 + B^2 = C^2  -- Pythagorean theorem for right angle at C
  BC_CA_ratio : B / A = 3 / 4  -- Ratio of BC to CA is 3:4

/-- The sine of angle A in the right triangle ABC is 3/5 -/
theorem sin_A_value (triangle : RightTriangle) : Real.sin (Real.arctan (triangle.B / triangle.A)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_l820_82015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l820_82049

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x - Real.pi / 3) - Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h1 : 1 < ω) (h2 : ω < 2)
  (h3 : ∀ x, f ω x = f ω (2 * Real.pi - x)) :
  (∃ T : ℝ, T > 0 ∧ T = 6 * Real.pi / 5 ∧ ∀ x, f ω x = f ω (x + T)) ∧
  (∃ A B C : ℝ, 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
    1 = Real.sin B * Real.sin C ∧
    f ω (3 / 5 * A) = 1 / 2 ∧
    ∀ a b c : ℝ, a = 1 → c = Real.sin A → b = Real.sin C / Real.sin B →
      1 / 2 * a * b * Real.sin C ≤ Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l820_82049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l820_82035

-- Define the angle α
noncomputable def α : Real := Real.arctan (-2 / (-1))

-- Define the point on the terminal side of the angle
def terminal_point : ℝ × ℝ := (-1, -2)

-- Theorem statement
theorem sin_alpha_value : 
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l820_82035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l820_82026

/-- The solution to the differential equation y'' = y' * cos(y) with initial conditions x = 1, y = π/2, y' = 1 -/
noncomputable def solution (x : ℝ) : ℝ := 2 * Real.arctan (Real.exp (x - 1))

/-- The differential equation y'' = y' * cos(y) -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x = (deriv y x) * Real.cos (y x)

theorem solution_satisfies_equation :
  differential_equation solution ∧
  solution 1 = π / 2 ∧
  deriv solution 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l820_82026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_max_min_l820_82097

-- Define the function u (noncomputable due to logarithm)
noncomputable def u (x y : ℝ) : ℝ := Real.log (8 * x * y + 4 * y^2 + 1) / Real.log 0.75

-- State the theorem
theorem u_max_min (x y : ℝ) (hx : x > 0) (hy : y ≥ 0) (hxy : x + 2*y = 1/2) :
  (∃ (x_max y_max : ℝ), x_max > 0 ∧ y_max ≥ 0 ∧ x_max + 2*y_max = 1/2 ∧
    u x_max y_max = 0 ∧ ∀ (x' y' : ℝ), x' > 0 → y' ≥ 0 → x' + 2*y' = 1/2 → u x' y' ≤ 0) ∧
  (∃ (x_min y_min : ℝ), x_min > 0 ∧ y_min ≥ 0 ∧ x_min + 2*y_min = 1/2 ∧
    u x_min y_min = -1 ∧ ∀ (x' y' : ℝ), x' > 0 → y' ≥ 0 → x' + 2*y' = 1/2 → u x' y' ≥ -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_max_min_l820_82097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_average_grades_l820_82012

theorem increase_average_grades (init_avg_A init_avg_B : ℝ) 
  (init_count_A init_count_B : ℕ) (grade_K grade_S : ℝ) :
  init_avg_A = 44.2 →
  init_avg_B = 38.8 →
  init_count_A = 10 →
  init_count_B = 10 →
  grade_K = 41 →
  grade_S = 44 →
  (init_avg_A * ↑init_count_A - grade_K - grade_S) / (↑init_count_A - 2) > init_avg_A ∧
  (init_avg_B * ↑init_count_B + grade_K + grade_S) / (↑init_count_B + 2) > init_avg_B :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_average_grades_l820_82012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l820_82011

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

def side_ratio_condition (t : Triangle) : Prop :=
  t.a / t.b = Real.sqrt 2 / Real.sqrt 3

def side_c_condition (t : Triangle) : Prop :=
  t.c = 2

-- Helper function to calculate triangle area
noncomputable def area_triangle (t : Triangle) : Real :=
  1/2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : angles_in_arithmetic_progression t)
  (h3 : side_ratio_condition t)
  (h4 : side_c_condition t) :
  t.A = Real.pi/4 ∧ t.B = Real.pi/3 ∧ t.C = 5*Real.pi/12 ∧
  area_triangle t = 3 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l820_82011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_candy_coloring_is_20ml_l820_82094

/-- Represents the amount of food colouring used for each type of candy and the total amount used. -/
structure CandyColoring where
  lollipop_coloring : ℚ
  lollipop_count : ℕ
  hard_candy_count : ℕ
  total_coloring : ℚ

/-- Calculates the amount of food colouring needed for each hard candy. -/
noncomputable def hard_candy_coloring (c : CandyColoring) : ℚ :=
  (c.total_coloring - c.lollipop_coloring * c.lollipop_count) / c.hard_candy_count

/-- Theorem stating that each hard candy needs 20 ml of food colouring. -/
theorem hard_candy_coloring_is_20ml (c : CandyColoring)
  (h1 : c.lollipop_coloring = 5)
  (h2 : c.lollipop_count = 100)
  (h3 : c.hard_candy_count = 5)
  (h4 : c.total_coloring = 600) :
  hard_candy_coloring c = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_candy_coloring_is_20ml_l820_82094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_with_triangle_area_8_l820_82081

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- The area of the triangle ODE formed by the origin and the intersection points
    of the line x = a with the asymptotes of the hyperbola -/
def triangle_area (h : Hyperbola) : ℝ := h.a * h.b

/-- The focal length of the hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- Theorem stating that the minimum focal length of a hyperbola with triangle area 8 is 8 -/
theorem min_focal_length_with_triangle_area_8 (h : Hyperbola) 
  (area_eq : triangle_area h = 8) :
  8 ≤ focal_length h ∧ ∃ (h' : Hyperbola), triangle_area h' = 8 ∧ focal_length h' = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_with_triangle_area_8_l820_82081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l820_82018

-- Define the room dimensions in feet
def room_length : ℚ := 15
def room_width : ℚ := 20

-- Define the tile dimensions in feet
def tile_length : ℚ := 3 / 12
def tile_width : ℚ := 9 / 12

-- Define the function to calculate the number of tiles
def tiles_required (room_l room_w tile_l tile_w : ℚ) : ℕ :=
  (room_l * room_w / (tile_l * tile_w)).ceil.toNat

-- State the theorem
theorem tiles_for_room :
  tiles_required room_length room_width tile_length tile_width = 1600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l820_82018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_at_3_l820_82039

/-- The polynomial f(x) = x^6 - x^3 - x^2 - 1 -/
def f (x : ℤ) : ℤ := x^6 - x^3 - x^2 - 1

/-- q₁ is an irreducible monic factor of f with integer coefficients -/
noncomputable def q₁ : Polynomial ℤ := sorry

/-- q₂ is an irreducible monic factor of f with integer coefficients -/
noncomputable def q₂ : Polynomial ℤ := sorry

/-- f factors as the product of q₁ and q₂ -/
axiom f_factorization (x : ℤ) : f x = (q₁.eval x) * (q₂.eval x)

/-- q₁ and q₂ are monic -/
axiom q₁_monic : Polynomial.Monic q₁
axiom q₂_monic : Polynomial.Monic q₂

/-- q₁ and q₂ are irreducible -/
axiom q₁_irreducible : Irreducible q₁
axiom q₂_irreducible : Irreducible q₂

theorem sum_of_factors_at_3 : q₁.eval 3 + q₂.eval 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_at_3_l820_82039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l820_82059

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (1 / (3 * m))^(-3 : ℤ) * (3 * m)^4 = (3 * m)^7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l820_82059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l820_82070

-- Define the curve function
noncomputable def curve (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + 1

-- Define the condition for no common points
def no_common_points (b : ℝ) : Prop :=
  ∀ x : ℝ, curve x ≠ b ∧ curve x ≠ -b

-- Theorem statement
theorem range_of_b :
  ∀ b : ℝ, no_common_points b ↔ -1 < b ∧ b < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l820_82070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_has_winning_strategy_l820_82020

/-- Represents a move in the domino gluing game -/
structure Move where
  row1 : Fin 100
  col1 : Fin 100
  row2 : Fin 100
  col2 : Fin 100
  isAdjacent : (row1 = row2 ∧ |col1 - col2| = 1) ∨ (col1 = col2 ∧ |row1 - row2| = 1)

/-- Represents the game state -/
structure GameState where
  board : Matrix (Fin 100) (Fin 100) Nat
  currentPlayer : Nat

/-- Determines if a move results in a fully connected board -/
def isConnected (state : GameState) (move : Move) : Bool :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Determines if a strategy is a winning strategy for Player 2 -/
def isWinningStrategyForPlayer2 (strategy : Strategy) : Prop :=
  ∀ (initialState : GameState),
    initialState.currentPlayer = 1 →
    ∃ (move : Move),
      ¬isConnected initialState move ∧
      ∀ (nextState : GameState),
        nextState.currentPlayer = 2 →
        ∃ (nextMove : Move),
          ¬isConnected nextState nextMove

/-- The main theorem stating that Player 2 has a winning strategy -/
theorem player2_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategyForPlayer2 strategy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_has_winning_strategy_l820_82020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_is_193_l820_82055

/-- Represents the cost structure and production quantity for an electronic component manufacturer -/
structure ManufacturerData where
  productionCost : ℚ
  shippingCost : ℚ
  fixedCosts : ℚ
  quantity : ℕ

/-- Calculates the lowest price per component that covers all costs -/
def lowestPricePerComponent (data : ManufacturerData) : ℚ :=
  (data.productionCost + data.shippingCost + data.fixedCosts / data.quantity)

/-- Theorem stating that the lowest price per component is $193 for the given manufacturer data -/
theorem lowest_price_is_193 (data : ManufacturerData) 
  (h1 : data.productionCost = 80)
  (h2 : data.shippingCost = 3)
  (h3 : data.fixedCosts = 16500)
  (h4 : data.quantity = 150) :
  lowestPricePerComponent data = 193 := by
  sorry

#eval lowestPricePerComponent { productionCost := 80, shippingCost := 3, fixedCosts := 16500, quantity := 150 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_is_193_l820_82055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l820_82043

theorem exponent_equation_solution (y : ℝ) : (3 : ℝ)^(y - 2) = (9 : ℝ)^3 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l820_82043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_share_2022_approx_gold_share_change_approx_l820_82076

/-- Total NWF funds as of December 1, 2022 (in billion rubles) -/
def total_nwf : ℝ := 1388.01

/-- Euro allocation in NWF (in billion rubles) -/
def euro_allocation : ℝ := 41.89

/-- British pound allocation in NWF (in billion rubles) -/
def pound_allocation : ℝ := 2.77

/-- Japanese yen allocation in NWF (in billion rubles) -/
def yen_allocation : ℝ := 478.48

/-- Chinese yuan allocation in NWF (in billion rubles) -/
def yuan_allocation : ℝ := 309.72

/-- Other currencies allocation in NWF (in billion rubles) -/
def other_allocation : ℝ := 0.24

/-- Gold share in NWF as of December 1, 2021 (in percentage) -/
def gold_share_2021 : ℝ := 31.8

/-- Calculates the gold allocation in NWF as of December 1, 2022 -/
def gold_allocation : ℝ := total_nwf - euro_allocation - pound_allocation - yen_allocation - yuan_allocation - other_allocation

/-- Theorem: The share of gold in the NWF as of December 1, 2022, is approximately 39.98% -/
theorem gold_share_2022_approx : 
  ∃ ε > 0, abs ((gold_allocation / total_nwf) * 100 - 39.98) < ε := by
  sorry

/-- Theorem: The change in gold share from 2021 to 2022 is approximately 8.2 percentage points -/
theorem gold_share_change_approx : 
  ∃ ε > 0, abs (((gold_allocation / total_nwf) * 100 - gold_share_2021) - 8.2) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_share_2022_approx_gold_share_change_approx_l820_82076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_k_value_l820_82000

/-- Given two unit vectors e₁ and e₂ with an angle of 2π/3 between them,
    and vectors a and b defined in terms of e₁ and e₂,
    prove that the value of k that makes a · b = 0 is 5/4. -/
theorem dot_product_zero_k_value 
  (e₁ e₂ : ℝ × ℝ × ℝ) 
  (h_unit_e₁ : ‖e₁‖ = 1) 
  (h_unit_e₂ : ‖e₂‖ = 1) 
  (h_angle : e₁ • e₂ = -1/2) 
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = e₁ - 2 • e₂) 
  (h_b : ∃ k : ℝ, b = k • e₁ + e₂) :
  ∃ k : ℝ, b = k • e₁ + e₂ ∧ a • b = 0 ↔ k = 5/4 := by
  sorry

#check dot_product_zero_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_k_value_l820_82000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l820_82065

def A (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a^2 + a < 0}

def B : Set ℝ := {x | x^2 + Real.log x ≤ 1000}

theorem range_of_a :
  ∀ a : ℝ, (A a ∩ (Set.univ \ B) = ∅) ↔ (1/1000 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l820_82065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_irrational_numbers_l820_82044

noncomputable def special_sequence : ℝ := sorry -- Represents -0.1010010001... (with one more 0 between each pair of 1)

noncomputable def number_list : List ℝ := [1/7, -Real.pi, -Real.sqrt 3, 0.3, special_sequence, -Real.sqrt 49]

def is_irrational (x : ℝ) : Bool := sorry -- We'll use a Bool instead of Prop for List.filter

theorem count_irrational_numbers : 
  (number_list.filter is_irrational).length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_irrational_numbers_l820_82044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l820_82099

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given triangle
noncomputable def givenTriangle : Triangle where
  A := 30 * Real.pi / 180  -- Convert 30° to radians
  a := 2
  b := 2 * Real.sqrt 3
  -- Other fields are not specified, so we'll use arbitrary values
  B := 0
  C := 0
  c := 0

-- State the theorem
theorem triangle_angle_B (t : Triangle) (h1 : t.A = givenTriangle.A) 
    (h2 : t.a = givenTriangle.a) (h3 : t.b = givenTriangle.b) : 
  t.B = 60 * Real.pi / 180 ∨ t.B = 120 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l820_82099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_director_salary_equals_avg_non_director_salary_l820_82054

/-- Represents a company with employees and salaries -/
structure Company where
  num_employees : ℕ
  director_salary : ℝ
  total_non_director_salary : ℝ

/-- The average salary of non-director employees in a company -/
noncomputable def avg_non_director_salary (c : Company) : ℝ :=
  c.total_non_director_salary / (c.num_employees - 1)

/-- The average salary of all employees in a company -/
noncomputable def avg_total_salary (c : Company) : ℝ :=
  (c.director_salary + c.total_non_director_salary) / c.num_employees

theorem director_salary_equals_avg_non_director_salary
  (horns hooves : Company)
  (h1 : horns.num_employees ≠ hooves.num_employees)
  (h2 : horns.director_salary = hooves.director_salary)
  (h3 : avg_non_director_salary horns = avg_non_director_salary hooves)
  (h4 : avg_total_salary horns = avg_total_salary hooves) :
  horns.director_salary = avg_non_director_salary horns :=
by
  sorry

#check director_salary_equals_avg_non_director_salary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_director_salary_equals_avg_non_director_salary_l820_82054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l820_82058

/-- Calculates the length of a train given the conditions of two trains passing each other. -/
noncomputable def calculate_train_length (length1 : ℝ) (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  relative_speed * time - length1

/-- Theorem stating the length of the second train given the problem conditions. -/
theorem second_train_length :
  let length1 : ℝ := 290
  let speed1 : ℝ := 120
  let speed2 : ℝ := 80
  let time : ℝ := 9
  ∃ ε > 0, |calculate_train_length length1 speed1 speed2 time - 209.95| < ε := by
  sorry

-- Using #eval with noncomputable functions is not possible, so we'll comment it out
-- #eval calculate_train_length 290 120 80 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l820_82058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_fifth_element_and_factorial_l820_82022

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The 15th row of Pascal's Triangle (indexed from 0) -/
def pascal_row_15 : List ℕ := List.map (binomial 15) (List.range 16)

theorem pascal_fifth_element_and_factorial :
  (pascal_row_15[4]! = 1365) ∧
  (factorial (pascal_row_15[4]!) = factorial 1365) := by
  sorry

#eval pascal_row_15[4]!
#eval factorial 1365

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_fifth_element_and_factorial_l820_82022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_l820_82042

theorem rectangle_ratio (w : ℚ) : 
  w > 0 → 
  2 * w + 2 * 10 = 30 → 
  w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_l820_82042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_climbs_out_in_four_days_l820_82083

/-- The number of days it takes for a snail to climb out of a well -/
noncomputable def snail_climb_days (well_depth : ℝ) (day_climb : ℝ) (night_slip : ℝ) : ℕ :=
  let net_daily_progress := day_climb - night_slip
  let last_day_climb := day_climb
  let distance_before_last_day := well_depth - last_day_climb
  let full_days := Int.floor (distance_before_last_day / net_daily_progress)
  full_days.toNat + 1

/-- Theorem: A snail in a 1.1m deep well, climbing 0.4m up during the day
    and slipping 0.2m down at night, takes 4 days to climb out -/
theorem snail_climbs_out_in_four_days :
  snail_climb_days 1.1 0.4 0.2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_climbs_out_in_four_days_l820_82083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_multiplication_result_l820_82033

theorem correct_multiplication_result (f : ℕ) : 
  (f * 153 = 102325 ∧ ∃ (a b : ℕ), a ≠ 2 ∧ b ≠ 2 ∧ f * 153 = 100000 + a * 1000 + 300 + b * 10 + 5) → 
  f * 153 = 102357 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_multiplication_result_l820_82033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_side_b_range_l820_82086

/-- Define the triangle ABC -/
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

/-- Define the given condition -/
def given_condition (t : Triangle) : Prop :=
  (Real.cos t.B - 2 * Real.cos t.A) / Real.cos t.C = (2 * t.a - t.b) / t.c

/-- Theorem 1: a/b = 2 -/
theorem side_ratio (t : Triangle) (h : given_condition t) : t.a / t.b = 2 := by
  sorry

/-- Theorem 2: If angle A is obtuse and c = 3, then √3 < b < 3 -/
theorem side_b_range (t : Triangle) (h1 : given_condition t) (h2 : t.A > Real.pi / 2) (h3 : t.c = 3) :
  Real.sqrt 3 < t.b ∧ t.b < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_side_b_range_l820_82086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_range_l820_82087

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

-- Define the points
variable (A B : C₁) (P : C₂)

-- Define the distance between A and B
axiom AB_distance : dist A B = Real.sqrt 3

-- Define the vector sum
noncomputable def vector_sum (A B P : ℝ × ℝ) : ℝ :=
  ‖(P.1 - A.1, P.2 - A.2) + (P.1 - B.1, P.2 - B.2)‖

-- Theorem statement
theorem vector_sum_range :
  ∀ A B P, A ∈ C₁ → B ∈ C₁ → P ∈ C₂ → dist A B = Real.sqrt 3 →
  7 ≤ vector_sum A B P ∧ vector_sum A B P ≤ 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_range_l820_82087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l820_82004

/-- The number of ways to arrange 6 people in a row under various conditions -/
theorem arrangement_counts :
  let n : ℕ := 6  -- Total number of people
  (n - 2) * (Nat.factorial (n - 1)) = 480 ∧  -- (1) Person A not at ends
  2 * (Nat.factorial (n - 1)) = 240 ∧  -- (2) Person A and B adjacent
  n * (Nat.factorial (n - 1)) - 2 * (Nat.factorial (n - 1)) = 480 ∧  -- (3) Person A and B not adjacent
  n * (Nat.factorial (n - 1)) - 3 * (Nat.factorial (n - 1)) = 360 ∧  -- (4) Person A before B (not necessarily adjacent)
  2 * (Nat.factorial (n - 2)) = 48  -- (5) Person A and B at ends
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l820_82004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_equals_cube_l820_82093

variable (f g h k : ℤ)

def u (f g h k : ℤ) : ℤ := f * (f^2 + 3 * g^2) - h * (h^2 + 3 * k^2)
def t (f g h k : ℤ) : ℤ := 3 * k * (h^2 + 3 * k^2) - 3 * g * (f^2 + 3 * g^2)

def p (f g h k : ℤ) : ℤ := f * t f g h k + 3 * g * u f g h k
def q (f g h k : ℤ) : ℤ := g * t f g h k - f * u f g h k
def r (f g h k : ℤ) : ℤ := k * t f g h k - h * u f g h k
def s (f g h k : ℤ) : ℤ := h * t f g h k + 3 * k * u f g h k

def x (f g h k : ℤ) : ℤ := p f g h k + q f g h k
def y (f g h k : ℤ) : ℤ := p f g h k - q f g h k
def z (f g h k : ℤ) : ℤ := r f g h k - s f g h k

theorem cube_sum_equals_cube (f g h k : ℤ) :
  (x f g h k)^3 + (y f g h k)^3 + (z f g h k)^3 = (r f g h k + s f g h k)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_equals_cube_l820_82093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subset_with_common_gcd_l820_82082

/-- A set of integers where each element is a product of at most 1987 primes -/
def LimitedPrimeFactorSet (A : Set ℤ) : Prop :=
  ∀ a ∈ A, ∃ (primes : Finset ℕ) (multiplicities : ℕ → ℕ),
    (∀ p ∈ primes, Nat.Prime p) ∧
    (∀ p ∉ primes, multiplicities p = 0) ∧
    (Finset.sum primes multiplicities ≤ 1987) ∧
    a = Finset.prod primes (fun p => p ^ multiplicities p)

theorem infinite_subset_with_common_gcd
  (A : Set ℤ) (hA : Set.Infinite A) (hLPF : LimitedPrimeFactorSet A) :
  ∃ (B : Set ℤ) (b : ℕ), Set.Infinite B ∧ B ⊆ A ∧
    ∀ (x y : ℤ), x ∈ B → y ∈ B → Nat.gcd (Int.natAbs x) (Int.natAbs y) = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subset_with_common_gcd_l820_82082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_difference_l820_82013

theorem polynomial_value_difference (a : ℝ) (n : ℕ) (P : Polynomial ℝ) :
  a ≥ 3 →
  P.natDegree = n →
  ∃ i : Fin (n + 2), |a ^ (i : ℕ) - P.eval ((i : ℕ) : ℝ)| ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_difference_l820_82013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_proof_l820_82003

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digits_to_num (d : List ℕ) : ℕ := d.foldl (fun acc x => acc * 10 + x) 0

def num_to_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

def reverse_num (n : ℕ) : ℕ := digits_to_num (num_to_digits n).reverse

theorem phone_number_proof :
  ∃! n : ℕ,
    (num_to_digits n).length = 5 ∧
    is_prime (List.head! (num_to_digits n)) ∧
    is_prime (digits_to_num (List.take 2 (List.drop 1 (num_to_digits n)))) ∧
    is_perfect_square (digits_to_num (List.take 2 (List.drop 3 (num_to_digits n)))) ∧
    Even (reverse_num n) ∧
    n = 26116 := by
  sorry

#check phone_number_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_proof_l820_82003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l820_82024

open Real

variable (a b c A B C : ℝ)

-- Triangle ABC with sides a, b, c opposite to angles A, B, C
axiom triangle_exists : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Law of cosines
axiom law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- Law of sines
axiom law_of_sines : a / (Real.sin A) = b / (Real.sin B)

theorem triangle_properties :
  (a * (Real.cos B) = b * (Real.cos A) → A = B) ∧
  (b * (Real.cos A) + (a - 2*c) * (Real.cos B) = 0 → B = Real.pi/3) ∧
  (A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2 → 
    (a^2 + b^2 - c^2) * (Real.sin A) > (a^2 + b^2 - c^2) * (Real.cos B)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l820_82024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_removal_l820_82002

/-- Represents a position on a chessboard --/
structure Position where
  x : Fin 20
  y : Fin 20

/-- Represents a knight on the chessboard --/
structure Knight where
  pos : Position

/-- Defines the concept of a knight controlling a square --/
def controls (k : Knight) (p : Position) : Prop := sorry

/-- The set of all positions on the 20x20 chessboard --/
def all_positions : Set Position := sorry

/-- The initial set of 220 knights --/
def initial_knights : Finset Knight := sorry

/-- Axiom: The initial knights control all positions --/
axiom initial_control : ∀ p : Position, ∃ k ∈ initial_knights, controls k p

/-- The theorem to be proved --/
theorem knights_removal :
  ∃ (remaining_knights : Finset Knight),
    remaining_knights ⊆ initial_knights ∧
    remaining_knights.card = 200 ∧
    (∀ p : Position, ∃ k ∈ remaining_knights, controls k p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_removal_l820_82002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_proof_l820_82038

noncomputable def v1 : ℝ × ℝ := (3, -2)
noncomputable def v2 : ℝ × ℝ := (2, 5)
noncomputable def q : ℝ × ℝ := (133/50, 39/50)

theorem projection_vector_proof :
  ∃ (a b : ℝ), q = a • v1 + b • v2 ∧
  (q.1 * (v2.1 - v1.1) + q.2 * (v2.2 - v1.2) = 0) := by
  sorry

#check projection_vector_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_proof_l820_82038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boundary_line_exists_l820_82089

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2
noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.log x)

def is_boundary_line (k m : ℝ) : Prop :=
  (∀ x, f x ≥ k * x + m) ∧ 
  (∀ x, x > 0 → g x ≤ k * x + m)

theorem boundary_line_exists : 
  ∃ k m, is_boundary_line k m ∧ k = Real.sqrt (Real.exp 1) ∧ m = -(Real.exp 1)/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boundary_line_exists_l820_82089
