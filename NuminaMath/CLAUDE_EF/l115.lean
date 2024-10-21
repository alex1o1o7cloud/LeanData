import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_fraction_l115_11540

theorem original_fraction (n d : ℚ) 
  (h1 : (n + 2) / (d + 1) = 1)
  (h2 : (n + 4) / (d + 2) = 1/2)
  (h3 : (n + 6) / (d + 3) = 2/3) :
  n / d = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_fraction_l115_11540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_specific_l115_11508

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Calculate the volume of a tetrahedron given its four vertices -/
noncomputable def tetrahedronVolume (a b c d : Point3D) : ℝ :=
  (1 / 6) * abs (
    (b.x - a.x) * ((c.y - a.y) * (d.z - a.z) - (c.z - a.z) * (d.y - a.y)) -
    (b.y - a.y) * ((c.x - a.x) * (d.z - a.z) - (c.z - a.z) * (d.x - a.x)) +
    (b.z - a.z) * ((c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x))
  )

theorem tetrahedron_volume_specific : 
  ∃ (a b c : Point3D),
    a.y = 0 ∧ a.z = 0 ∧ a.x > 0 ∧
    b.x = 0 ∧ b.z = 0 ∧ b.y > 0 ∧
    c.x = 0 ∧ c.y = 0 ∧ c.z > 0 ∧
    distance a b = 7 ∧
    distance b c = 8 ∧
    distance c a = 9 ∧
    tetrahedronVolume origin a b c = 2 * Real.sqrt 176 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_specific_l115_11508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donovans_test_score_l115_11537

-- Define the number of fully correct, incorrect, and partially correct answers
def fully_correct : ℕ := 35
def incorrect : ℕ := 13
def partially_correct : ℕ := 7

-- Define the point value for each type of answer
def full_point_value : ℚ := 1
def partial_point_value : ℚ := 1/2

-- Calculate the total points earned
def total_points_earned : ℚ :=
  fully_correct * full_point_value + partially_correct * partial_point_value

-- Calculate the total number of questions
def total_questions : ℕ := fully_correct + incorrect + partially_correct

-- Calculate the percentage of correct answers
def percentage_correct : ℚ := (total_points_earned / total_questions) * 100

-- Theorem to prove
theorem donovans_test_score :
  (round (percentage_correct * 100) : ℚ) / 100 = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donovans_test_score_l115_11537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l115_11579

/-- An increasing arithmetic sequence starting with 1 -/
def arithmetic_sequence (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_sequence d n + d

/-- An increasing geometric sequence starting with 1 -/
def geometric_sequence (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_sequence r n * r

/-- The sum of the arithmetic and geometric sequences -/
def c_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

theorem c_k_value (d r k : ℕ) :
  d > 0 ∧ r > 1 ∧ k > 2 ∧
  c_sequence d r (k - 1) = 200 ∧
  c_sequence d r (k + 1) = 400 →
  c_sequence d r k = 322 := by
  sorry

#eval c_sequence 98 3 4  -- This should output 322

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l115_11579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_on_percentage_l115_11595

theorem operation_on_percentage : ∃ (x : ℝ → ℝ), 
  (2 * (x 0.04) = 0.02) ∧ (x = (λ y ↦ y / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_on_percentage_l115_11595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_sets_l115_11525

theorem unique_integer_sets : 
  {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | 
    1 < a ∧ a < b ∧ b < c ∧ c < d ∧ 
    ∃ (k : ℕ), k * ((a - 1) * (b - 1) * (c - 1) * (d - 1)) = a * b * c * d - 1} = 
  {(3, 5, 17, 255), (2, 4, 10, 80)} := by
  sorry

#check unique_integer_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_sets_l115_11525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_correct_l115_11583

/-- Horner's Rule for polynomial evaluation -/
theorem horner_rule_correct (n : ℕ) (a : ℕ → ℝ) (x₀ : ℝ) :
  let f (x : ℝ) := (Finset.range (n + 1)).sum (λ i ↦ a i * x^i)
  let horner_eval := 
    Nat.recOn n
      (a 0)
      (λ k acc ↦ acc * x₀ + a (k + 1))
  f x₀ = horner_eval :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_correct_l115_11583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintain_magic_power_l115_11570

/-- Represents the actions that the Little Humpbacked Horse can perform -/
inductive HorseAction
  | Eat
  | Sleep

/-- Represents the state of the Little Humpbacked Horse -/
structure HorseState where
  lastAction : HorseAction
  daysWithoutEating : Nat
  daysWithoutSleeping : Nat
  hasMagicPower : Bool

/-- Determines if the Horse has magic power based on its state -/
def hasMagicPower (state : HorseState) : Bool :=
  state.daysWithoutEating < 7 && state.daysWithoutSleeping < 7

/-- Updates the Horse's state after performing an action -/
def updateState (state : HorseState) (action : HorseAction) : HorseState :=
  match action with
  | HorseAction.Eat => {
      lastAction := HorseAction.Eat,
      daysWithoutEating := 0,
      daysWithoutSleeping := state.daysWithoutSleeping + 1,
      hasMagicPower := hasMagicPower { 
        lastAction := HorseAction.Eat,
        daysWithoutEating := 0,
        daysWithoutSleeping := state.daysWithoutSleeping + 1,
        hasMagicPower := state.hasMagicPower
      }
    }
  | HorseAction.Sleep => {
      lastAction := HorseAction.Sleep,
      daysWithoutEating := state.daysWithoutEating + 1,
      daysWithoutSleeping := 0,
      hasMagicPower := hasMagicPower {
        lastAction := HorseAction.Sleep,
        daysWithoutEating := state.daysWithoutEating + 1,
        daysWithoutSleeping := 0,
        hasMagicPower := state.hasMagicPower
      }
    }

/-- Theorem: To maintain magic power, the Horse must perform the action
    opposite to the last action before the 7-day period -/
theorem maintain_magic_power (initialState : HorseState)
  (h : initialState.daysWithoutEating = 7 ∨ initialState.daysWithoutSleeping = 7) :
  let oppositeAction := match initialState.lastAction with
    | HorseAction.Eat => HorseAction.Sleep
    | HorseAction.Sleep => HorseAction.Eat
  (updateState initialState oppositeAction).hasMagicPower = true := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintain_magic_power_l115_11570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_α_plus_β_equals_three_fourths_π_l115_11562

/-- Two acute angles in the Cartesian coordinate system -/
noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

/-- Points A and B on the unit circle -/
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

/-- Conditions for the problem -/
axiom acute_α : 0 < α ∧ α < Real.pi / 2
axiom acute_β : 0 < β ∧ β < Real.pi / 2
axiom A_on_unit_circle : A.1^2 + A.2^2 = 1
axiom B_on_unit_circle : B.1^2 + B.2^2 = 1
axiom A_x_coord : A.1 = Real.sqrt 5 / 5
axiom B_y_coord : B.2 = Real.sqrt 2 / 10

/-- The theorem to be proved -/
theorem two_α_plus_β_equals_three_fourths_π : 2 * α + β = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_α_plus_β_equals_three_fourths_π_l115_11562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_slope_intercept_product_problem_statement_l115_11554

/-- A type to represent indeterminate values -/
inductive Indeterminate : Type
  | indeterminate : Indeterminate

/-- Given two points with the same x-coordinate but different y-coordinates,
    the product of the slope and y-intercept of the line passing through these points
    is indeterminate. -/
theorem vertical_line_slope_intercept_product (x y₁ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let A : ℝ × ℝ := (x, y₁)
  let B : ℝ × ℝ := (x, y₂)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  let y_intercept := A.2 - slope * A.1
  Indeterminate := by sorry

/-- The specific case for points (5, 10) and (5, 20) -/
theorem problem_statement :
  let A : ℝ × ℝ := (5, 10)
  let B : ℝ × ℝ := (5, 20)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  let y_intercept := A.2 - slope * A.1
  Indeterminate := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_slope_intercept_product_problem_statement_l115_11554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_long_distance_call_cost_decrease_l115_11580

/-- The average cost of a long-distance call in cents per minute for a given year -/
def average_cost (year : Nat) : ℚ :=
  match year with
  | 1990 => 35
  | 2000 => 15
  | 2010 => 5
  | _ => 0

/-- Calculate the percent decrease between two years -/
def percent_decrease (year1 year2 : Nat) : ℚ :=
  ((average_cost year1 - average_cost year2) / average_cost year1) * 100

theorem long_distance_call_cost_decrease :
  (Int.floor (percent_decrease 1990 2010)) = 85 ∧
  (Int.floor (percent_decrease 2000 2010)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_long_distance_call_cost_decrease_l115_11580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_equation_l115_11544

theorem prime_power_equation (a p n : ℕ) (h_prime : Nat.Prime p) 
  (h_eq : p^a - 1 = 2^n * (p - 1)) :
  (a = 2 ∧ ∃ m : ℕ, p = 2^m - 1 ∧ Nat.Prime (2^m - 1) ∧ m = n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_equation_l115_11544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_dihedral_sphere_l115_11519

-- Define the dihedral angle
def DihedralAngle (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define a function to calculate the perimeter of a triangle
noncomputable def trianglePerimeter (a b c : Point3D) : ℝ :=
  distance a b + distance b c + distance c a

-- Define membership for Point3D in Sphere
def Point3DInSphere (p : Point3D) (s : Sphere) : Prop :=
  distance p s.center = s.radius

-- Define membership for Point3D in Plane
def Point3DInPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- State the theorem
theorem min_perimeter_dihedral_sphere 
  (α : ℝ) 
  (dihedral : DihedralAngle α)
  (s : Sphere) 
  (p1 p2 : Plane) 
  (A : Point3D) 
  (AG : ℝ) :
  ∃ (B C : Point3D), 
    Point3DInSphere A s ∧ 
    Point3DInPlane B p1 ∧ 
    Point3DInPlane C p2 ∧ 
    (∀ (B' C' : Point3D), Point3DInPlane B' p1 → Point3DInPlane C' p2 → 
      trianglePerimeter A B C ≤ trianglePerimeter A B' C') ∧
    trianglePerimeter A B C = 2 * AG * Real.sin α :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_dihedral_sphere_l115_11519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l115_11506

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 3 * (4 ^ n)

noncomputable def S : ℕ → ℝ
  | 0 => 1
  | n + 1 => S n + sequence_a (n + 1)

noncomputable def sequence_b (n : ℕ) : ℝ :=
  9 * sequence_a n / ((sequence_a n + 3) * (sequence_a (n + 1) + 3))

noncomputable def T : ℕ → ℝ
  | 0 => sequence_b 0
  | n + 1 => T n + sequence_b (n + 1)

theorem sequence_properties :
  (∀ n ≥ 2, sequence_a (n + 1) * S (n - 1) - sequence_a n * S n = 0) →
  (∀ n : ℕ, S n = 4^n) ∧
  (∀ n : ℕ, sequence_a n = if n = 0 then 1 else if n = 1 then 3 else 3 * 4^(n - 2)) ∧
  (∀ n : ℕ, T n = 7/8 - 1/(4^n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l115_11506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_l115_11505

-- Define the function f(x) as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x - 1)

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a ≤ 0 := by
  sorry

-- You can add more lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_l115_11505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tire_repair_problem_l115_11501

theorem tire_repair_problem (repair_cost : ℚ) (sales_tax : ℚ) (final_cost : ℚ) :
  repair_cost = 7 →
  sales_tax = 1/2 →
  final_cost = 30 →
  (final_cost / (repair_cost + sales_tax)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tire_repair_problem_l115_11501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_correlation_without_3_10_l115_11507

-- Define the set of points
def points : List (ℝ × ℝ) := [(1, 2), (2, 4), (4, 5), (3, 10), (10, 12)]

-- Function to calculate the correlation coefficient
noncomputable def correlationCoefficient (pts : List (ℝ × ℝ)) : ℝ :=
  sorry

-- Function to get all subsets of size 4
def subsetsOfSizeFour (pts : List (ℝ × ℝ)) : List (List (ℝ × ℝ)) :=
  sorry

-- Theorem statement
theorem highest_correlation_without_3_10 :
  let subsets := subsetsOfSizeFour points
  let correlations := subsets.map correlationCoefficient
  let maxCorrelation := correlations.maximum?
  maxCorrelation = some (correlationCoefficient (points.filter (· ≠ (3, 10)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_correlation_without_3_10_l115_11507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_head_on_third_flip_l115_11528

-- Define a fair coin
noncomputable def fair_coin : ℝ := 1/2

-- Define the probability of getting tails on a single flip
noncomputable def prob_tails (p : ℝ) : ℝ := p

-- Define the probability of getting heads on a single flip
noncomputable def prob_heads (p : ℝ) : ℝ := p

-- Theorem statement
theorem first_head_on_third_flip (p : ℝ) (h : p = fair_coin) : 
  prob_tails p * prob_tails p * prob_heads p = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_head_on_third_flip_l115_11528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l115_11504

theorem inequality_holds_iff_a_leq_neg_two (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l115_11504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_equidistant_from_m_and_n_p_unique_equidistant_point_l115_11565

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- P is on the Z-axis -/
def P : Point3D := ⟨0, 0, -3⟩

/-- Point M -/
def M : Point3D := ⟨1, 0, 2⟩

/-- Point N -/
def N : Point3D := ⟨1, -3, 1⟩

/-- Theorem: P is equidistant from M and N -/
theorem p_equidistant_from_m_and_n : distance P M = distance P N := by
  sorry

/-- Main theorem: P(0, 0, -3) is the unique point on the Z-axis equidistant from M and N -/
theorem p_unique_equidistant_point :
  ∀ Q : Point3D, Q.x = 0 ∧ Q.y = 0 → (distance Q M = distance Q N ↔ Q = P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_equidistant_from_m_and_n_p_unique_equidistant_point_l115_11565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotonicity_intervals_monotonic_increasing_condition_l115_11597

noncomputable section

variable (k : ℝ)

def f (x : ℝ) := x * Real.exp (k * x)

theorem tangent_line_at_zero (h : k ≠ 0) :
  (fun x => x) = fun x => (deriv (f k)) 0 * x + f k 0 :=
by sorry

theorem monotonicity_intervals (h : k ≠ 0) :
  (∀ x y, x < y → x < -1/k → y < -1/k → f k x > f k y) ∧
  (∀ x y, x < y → x > -1/k → y > -1/k → f k x < f k y) :=
by sorry

theorem monotonic_increasing_condition (h : k ≠ 0) :
  (∀ x y, -1 < x → x < y → y < 1 → f k x < f k y) ↔ 
  (k ∈ Set.Icc (-1) 0 ∪ Set.Ioo 0 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotonicity_intervals_monotonic_increasing_condition_l115_11597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_tan_sum_side_sum_l115_11511

noncomputable section

-- Define the triangle
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle : a > 0 ∧ b > 0 ∧ c > 0
axiom geometric_sequence : b^2 = a * c
axiom cos_B : Real.cos B = 3/4
axiom dot_product : a * c * Real.cos B = 3/2

-- Theorem 1
theorem inverse_tan_sum :
  1 / Real.tan A + 1 / Real.tan C = 4 / Real.sqrt 7 :=
sorry

-- Theorem 2
theorem side_sum :
  a + c = 3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_tan_sum_side_sum_l115_11511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l115_11577

-- Define a power function
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point :
  ∀ α : ℝ, powerFunction α 2 = (1 : ℝ) / 4 → α = -2 := by
  intro α h
  -- The proof goes here
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l115_11577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l115_11587

/-- Given f(x) = xe^x, the equation of the tangent line to the graph of f(x) at (0, f(0)) is x - y = 0 -/
theorem tangent_line_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x * Real.exp x) :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 0 ∧ y = f 0) ∨ y - f 0 = ((deriv f) 0) * (x - 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l115_11587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2023_pi_4_minus_2alpha_l115_11599

theorem tan_2023_pi_4_minus_2alpha (α : Real) :
  (∃ P : Real × Real, P = (1, -5) ∧ Real.tan α = -5) →
  Real.tan ((2023 / 4) * Real.pi - 2 * α) = -17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2023_pi_4_minus_2alpha_l115_11599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_set_l115_11541

def is_valid_set (S : Set ℕ) : Prop :=
  S.Nonempty ∧ S.Finite ∧ 
  ∀ i j, i ∈ S → j ∈ S → (i + j) / Nat.gcd i j ∈ S

theorem unique_valid_set : 
  ∃! S : Set ℕ, is_valid_set S ∧ S = {2} := by
  sorry

#check unique_valid_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_set_l115_11541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_triangle_area_l115_11556

/-- The hyperbola Γ: x^2 - y^2 = 1 -/
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2 = 1}

/-- The asymptotes of the hyperbola -/
def asymptotes : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 ∨ p.2 = -p.1}

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Tangent line to Γ at point P -/
noncomputable def tangentLine (P : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Intersection points of tangent line with asymptotes -/
noncomputable def intersectionPoints (P : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_tangent_triangle_area :
  ∀ P ∈ Γ, (let (A, B) := intersectionPoints P; triangleArea O A B = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_triangle_area_l115_11556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l115_11561

/-- Given an article with a cost price, marked price, and discount percentage,
    calculate the profit percentage on the cost price. -/
theorem profit_percentage_calculation
  (cost_price : ℝ)
  (marked_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 85.5)
  (h2 : marked_price = 112.5)
  (h3 : discount_percentage = 5) :
  (let discount := (discount_percentage / 100) * marked_price
   let selling_price := marked_price - discount
   let profit := selling_price - cost_price
   let profit_percentage := (profit / cost_price) * 100
   profit_percentage) = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l115_11561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_specific_l115_11574

/-- Given a triangle with sides a, b, and c, where a line parallel to the base
    divides the height in a ratio of m:n (counting from the vertex),
    calculates the area of the resulting trapezoid. -/
noncomputable def trapezoid_area (a b c m n : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let small_triangle_area := triangle_area * (m / (m + n))^2
  triangle_area - small_triangle_area

/-- Theorem stating that for a triangle with sides 30, 26, and 28,
    where a line divides the height in a 2:3 ratio,
    the area of the resulting trapezoid is approximately 322.56. -/
theorem trapezoid_area_specific : 
  ‖trapezoid_area 28 26 30 2 3 - 322.56‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_specific_l115_11574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l115_11500

-- Define the slopes of the two lines
noncomputable def slope1 (m : ℝ) : ℝ := -1 / (m^2 - m)
def slope2 : ℝ := 2

-- Define the perpendicularity condition
def perpendicular (m : ℝ) : Prop := slope1 m * slope2 = -1

-- Theorem statement
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, perpendicular m → m = -1 ∨ m = 2 := by
  intro m h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check perpendicular_lines_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l115_11500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_omega_for_increasing_f_l115_11523

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  4 * Real.cos (ω * x - Real.pi / 6) * Real.sin (ω * x) - Real.cos (2 * ω * x + Real.pi)

-- Theorem for the range of f
theorem range_of_f (ω : ℝ) (h : ω > 0) :
  Set.range (f ω) = Set.Icc (1 - Real.sqrt 3) (1 + Real.sqrt 3) := by
  sorry

-- Theorem for the maximum value of ω
theorem max_omega_for_increasing_f :
  (∃ (ω : ℝ), ω > 0 ∧ 
    (∀ x y : ℝ, -3*Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 → f ω x < f ω y) ∧
    (∀ ω' : ℝ, ω' > ω → ∃ x y : ℝ, -3*Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 ∧ f ω' x ≥ f ω' y)) ∧
  (∀ ω : ℝ, (∀ x y : ℝ, -3*Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 → f ω x < f ω y) → ω ≤ 1/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_omega_for_increasing_f_l115_11523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_from_sine_l115_11578

theorem cosine_value_from_sine (α : ℝ) : 
  Real.sin (π / 3 - α) = 1 / 4 → Real.cos (π / 3 + 2 * α) = -7 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_from_sine_l115_11578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_say_dislike_but_like_approx_l115_11558

/-- Represents the fraction of students who like dancing -/
noncomputable def like_dancing : ℝ := 0.7

/-- Represents the fraction of students who dislike dancing -/
noncomputable def dislike_dancing : ℝ := 1 - like_dancing

/-- Represents the fraction of students who like dancing and accurately say so -/
noncomputable def accurate_like : ℝ := 0.75

/-- Represents the fraction of students who dislike dancing and accurately say so -/
noncomputable def accurate_dislike : ℝ := 0.85

/-- The fraction of students who say they dislike dancing but actually like it -/
noncomputable def fraction_say_dislike_but_like : ℝ :=
  (like_dancing * (1 - accurate_like)) / 
  (like_dancing * (1 - accurate_like) + dislike_dancing * accurate_dislike)

theorem fraction_say_dislike_but_like_approx :
  |fraction_say_dislike_but_like - 0.407| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_say_dislike_but_like_approx_l115_11558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l115_11502

/-- Proposition p: There exists a > 0 such that f(x) = ax^2 - 4x is monotonically decreasing on (-∞, 2] -/
def p : Prop := ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x ≤ 2 → ∀ y : ℝ, y ≤ 2 → x < y → a * x^2 - 4 * x > a * y^2 - 4 * y

/-- Proposition q: There exists a ∈ ℝ such that for x ∈ ℝ, 16x^2 - 16(a-1)x + 1 ≠ 0 -/
def q : Prop := ∃ a : ℝ, ∀ x : ℝ, 16 * x^2 - 16 * (a - 1) * x + 1 ≠ 0

theorem range_of_a (h : p ∧ q) : ∃ a : ℝ, 1/2 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l115_11502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_five_l115_11532

/-- The set of digits to choose from -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function that checks if a number is divisible by 5 -/
def divisible_by_five (n : Nat) : Bool := n % 5 = 0

/-- A function that checks if a number has three distinct digits from the given set -/
def valid_number (n : Nat) : Bool :=
  n ≥ 100 && n < 1000 &&
  (n / 100) ∈ digits &&
  ((n / 10) % 10) ∈ digits &&
  (n % 10) ∈ digits &&
  (n / 100) ≠ ((n / 10) % 10) &&
  (n / 100) ≠ (n % 10) &&
  ((n / 10) % 10) ≠ (n % 10)

/-- The main theorem to be proved -/
theorem count_divisible_by_five :
  (Finset.filter (λ n => valid_number n && divisible_by_five n) (Finset.range 1000)).card = 36 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_five_l115_11532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equality_l115_11568

/-- Given an angle θ in standard position with its terminal side on the line 3x - 5y = 0,
    prove that tan θ + sin(7π/2 + 2θ) = 11/85 -/
theorem angle_sum_equality (θ : ℝ) 
  (h : ∃ (x y : ℝ), x ≠ 0 ∧ 3*x - 5*y = 0 ∧ Real.tan θ = y / x) : 
  Real.tan θ + Real.sin (7 * Real.pi / 2 + 2 * θ) = 11 / 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equality_l115_11568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_B_l115_11549

-- Define the set A
def A : Finset Char := {'a', 'b', 'c', 'd'}

-- Define the properties of set B
def is_valid_B (B : Finset Char) : Prop :=
  B ⊆ A ∧
  B.card = 2 ∧
  ('a' ∈ B → 'c' ∈ B) ∧
  ('d' ∉ B → 'c' ∉ B) ∧
  ('d' ∈ B → 'b' ∉ B)

-- Theorem stating that the only valid B is {c, d}
theorem unique_valid_B :
  ∀ B : Finset Char, is_valid_B B ↔ B = {'c', 'd'} :=
by
  sorry

#check unique_valid_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_B_l115_11549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_example_l115_11518

/-- The area of a rhombus given the lengths of its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals of length 20 and 30 is 300 -/
theorem rhombus_area_example : rhombusArea 20 30 = 300 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_example_l115_11518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_next_multiple_sum_of_digits_l115_11513

/-- Represents a person's age -/
def Age := ℕ

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Finds the next multiple of a number greater than a given value -/
def next_multiple (base : ℕ) (n : ℕ) : ℕ := 
  ((n + base - 1) / base) * base

theorem joey_age_next_multiple_sum_of_digits 
  (joey chloe zoe : ℕ) 
  (h1 : joey = chloe + 2)
  (h2 : zoe = 1)
  (h3 : ∃ (n : ℕ), n = 9 ∧ joey = (Nat.factors joey).length)
  (h4 : joey = 36) : 
  sum_of_digits (next_multiple joey zoe) = 9 := by
  sorry

#eval sum_of_digits (next_multiple 36 1)  -- Should output 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_next_multiple_sum_of_digits_l115_11513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_l115_11576

/-- Represents a trapezium with the given properties -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  nonParallelSide : ℝ
  angle : ℝ
  parallelSidesAreValid : side1 > 0 ∧ side2 > 0
  nonParallelSideIsValid : nonParallelSide > 0
  angleIsValid : 0 < angle ∧ angle < π

/-- Calculates the area of the trapezium -/
noncomputable def areaOfTrapezium (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.nonParallelSide * Real.sin t.angle / 4

/-- The main theorem stating the area of the specific trapezium -/
theorem trapezium_area :
  ∀ t : Trapezium,
    t.side1 = 20 ∧
    t.side2 = 18 ∧
    t.nonParallelSide = 15 ∧
    t.angle = π / 3 →
    areaOfTrapezium t = 285 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_l115_11576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l115_11563

/-- The cost price of a book given its selling price and profit percentage -/
noncomputable def cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem stating that the cost price of a book sold for $290 with 20% profit is $241.67 -/
theorem book_cost_price : 
  let selling_price : ℝ := 290
  let profit_percentage : ℝ := 20
  abs (cost_price selling_price profit_percentage - 241.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l115_11563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equality_and_abs_difference_l115_11512

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_equality_and_abs_difference :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) :=
by
  -- Proof is omitted using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equality_and_abs_difference_l115_11512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_L_l115_11575

/-- The side length of the cube -/
noncomputable def cube_side : ℝ := 3

/-- The surface area of the cube -/
noncomputable def cube_surface_area : ℝ := 6 * cube_side^2

/-- The radius of the sphere with the same surface area as the cube -/
noncomputable def sphere_radius : ℝ := (cube_surface_area / (4 * Real.pi))^(1/2)

/-- The volume of the sphere -/
noncomputable def sphere_volume : ℝ := (4/3) * Real.pi * sphere_radius^3

/-- The value L in the volume expression -/
noncomputable def L : ℝ := (sphere_volume * Real.pi^(1/2)) / Real.sqrt 15

theorem sphere_volume_equals_L (h : sphere_volume = L * Real.sqrt 15 / Real.pi^(1/2)) : 
  L = 108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_L_l115_11575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_l115_11592

/-- Represents the polynomial p(x) = 2x^n - 3x^{n-1} + 4x^{n-2} - ... + (-1)^{n-1}⋅(n+1)x + (-1)^n⋅(n + 2) -/
def p (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range (n + 1)).sum (fun k => (-1)^k * (k + 2 : ℝ) * x^(n - k))

/-- The maximum number of real roots for the polynomial p is n -/
theorem max_real_roots (n : ℕ) : 
  ∃ (r : ℕ), r ≤ n ∧ ∀ (s : ℕ), (∃ (roots : Finset ℝ), (∀ x ∈ roots, p n x = 0) ∧ roots.card = s) → s ≤ r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_l115_11592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_intersection_points_l115_11538

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - x^2 + a * x

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a * x - m

-- State the theorem
theorem tangent_line_and_intersection_points (a : ℝ) :
  -- Part 1: Tangent line equation when a = 2
  (∀ x y : ℝ, a = 2 → x = 1 → y = f 2 x → y = 2 * x - 1) ∧
  -- Part 2: Conditions for two intersection points
  (∀ m : ℝ, 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      Real.exp (-1) ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧
      Real.exp (-1) ≤ x₂ ∧ x₂ ≤ Real.exp 1 ∧
      f a x₁ = g a m x₁ ∧ f a x₂ = g a m x₂)
    ↔ 1 ≤ m ∧ m ≤ 2 + (Real.exp 2)⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_intersection_points_l115_11538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l115_11582

/-- The sequence sum function -/
def S (a : ℝ) (n : ℕ+) : ℝ := (n : ℝ)^2 + 2*a*|(n : ℝ) - 2016|

/-- The sequence term function -/
def a_n (a : ℝ) (n : ℕ+) : ℝ :=
  if n = 1 then S a 1
  else S a n - S a (n-1)

/-- The theorem statement -/
theorem max_a_value (a : ℝ) :
  (a > 0) →
  (∀ n : ℕ+, a_n a n ≤ a_n a (n+1)) ↔
  (a ≤ 1/2016) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l115_11582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l115_11517

/-- Represents the ingredients of the lemonade -/
structure Ingredients :=
  (lime_juice : ℚ)
  (sugar : ℚ)
  (water : ℚ)
  (mint : ℚ)

/-- Represents the calorie content per 100g of each ingredient -/
structure CalorieContent :=
  (lime_juice : ℚ)
  (sugar : ℚ)
  (water : ℚ)
  (mint : ℚ)

/-- Calculates the total calories in the lemonade mixture -/
def total_calories (i : Ingredients) (c : CalorieContent) : ℚ :=
  i.lime_juice * c.lime_juice / 100 +
  i.sugar * c.sugar / 100 +
  i.water * c.water / 100 +
  i.mint * c.mint / 100

/-- Calculates the total weight of the lemonade mixture -/
def total_weight (i : Ingredients) : ℚ :=
  i.lime_juice + i.sugar + i.water + i.mint

/-- Theorem: 300g of the lemonade contains approximately 276 calories -/
theorem lemonade_calories 
  (i : Ingredients) 
  (c : CalorieContent) 
  (h1 : i = { lime_juice := 150, sugar := 200, water := 500, mint := 50 })
  (h2 : c = { lime_juice := 30, sugar := 390, water := 0, mint := 7 }) :
  (300 * total_calories i c / total_weight i).floor = 276 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l115_11517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangle_weight_l115_11521

/-- Represents an equilateral triangle with given side length and weight -/
structure EquilateralTriangle where
  sideLength : ℝ
  weight : ℝ

/-- Calculates the area of an equilateral triangle given its side length -/
noncomputable def triangleArea (a : ℝ) : ℝ := (a^2 * Real.sqrt 3) / 4

theorem second_triangle_weight (t1 t2 : EquilateralTriangle) 
  (h1 : t1.sideLength = 4)
  (h2 : t1.weight = 16)
  (h3 : t2.sideLength = 6)
  (h4 : triangleArea t1.sideLength / t1.weight = triangleArea t2.sideLength / t2.weight) :
  t2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangle_weight_l115_11521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_obtuse_triangle_l115_11536

-- Define the triangle
def Triangle (A B C : ℝ) := 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define an obtuse triangle
def ObtuseTriangle (A B C : ℝ) := 
  Triangle A B C ∧ (Real.pi/2 < A ∨ Real.pi/2 < B ∨ Real.pi/2 < C)

-- Define the theorem
theorem max_value_in_obtuse_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  ObtuseTriangle A B C →
  A = 3*Real.pi/4 →
  c = 1 →
  (∀ x : ℝ, 2*Real.sqrt 2*|a| + 3*|b| ≤ x → x ≤ Real.sqrt 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_obtuse_triangle_l115_11536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_five_digit_number_tens_place_l115_11535

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (n.digits 10).length = 5 ∧
  (n.digits 10).toFinset = {0, 3, 5, 6, 8}

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem smallest_even_five_digit_number_tens_place :
  ∃ (n : ℕ), is_valid_number n ∧ is_even n ∧
  (∀ (m : ℕ), is_valid_number m ∧ is_even m → n ≤ m) ∧
  ((n / 10) % 10 = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_five_digit_number_tens_place_l115_11535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_perm_price_l115_11552

/-- Represents Rebecca's hair salon pricing and daily earnings -/
structure RebeccaSalon where
  haircut_price : ℕ
  dye_job_price : ℕ
  dye_box_cost : ℕ
  perm_price : ℕ
  haircuts : ℕ
  dye_jobs : ℕ
  perms : ℕ
  tips : ℕ
  total_earnings : ℕ
  haircut_price_eq : haircut_price = 30
  dye_job_price_eq : dye_job_price = 60
  dye_box_cost_eq : dye_box_cost = 10
  haircuts_eq : haircuts = 4
  dye_jobs_eq : dye_jobs = 2
  perms_eq : perms = 1
  tips_eq : tips = 50
  total_earnings_eq : total_earnings = 310
  earnings_eq : total_earnings = haircut_price * haircuts + dye_job_price * dye_jobs - dye_box_cost * dye_jobs + perm_price * perms + tips

/-- Proves that Rebecca charges $40 for a perm -/
theorem rebecca_perm_price (salon : RebeccaSalon) : salon.perm_price = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_perm_price_l115_11552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l115_11586

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x / 4 - Real.pi / 6) - 2 * (Real.cos (Real.pi * x / 8))^2 + 1

theorem f_properties :
  -- The smallest positive period is 8
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
   (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧ T = 8) ∧
  -- The maximum value is √3
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = Real.sqrt 3) ∧
  -- The function is increasing on intervals [8k + 2/3, 8k + 10/3] for any integer k
  (∀ (k : ℤ), ∀ (x y : ℝ), 8 * ↑k + 2/3 ≤ x ∧ x < y ∧ y ≤ 8 * ↑k + 10/3 → f x < f y) ∧
  -- The function is decreasing on intervals [8k + 10/3, 8k + 22/3] for any integer k
  (∀ (k : ℤ), ∀ (x y : ℝ), 8 * ↑k + 10/3 ≤ x ∧ x < y ∧ y ≤ 8 * ↑k + 22/3 → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l115_11586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_proof_l115_11567

/-- The ratio of the volume of an octahedron formed by the centers of a unit cube's faces to the volume of the cube -/
noncomputable def octahedron_cube_volume_ratio : ℝ := Real.sqrt 3 / 9

/-- The vertices of a unit cube -/
def unit_cube_vertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0], ![1, 0, 0], ![0, 1, 0], ![0, 0, 1],
  ![1, 1, 0], ![1, 0, 1], ![0, 1, 1], ![1, 1, 1]
]

/-- The centers of a unit cube's faces -/
def cube_face_centers : List (Fin 3 → ℝ) := [
  ![0.5, 0.5, 0], ![0, 0.5, 0.5], ![0.5, 0, 0.5],
  ![1, 0.5, 0.5], ![0.5, 1, 0.5], ![0.5, 0.5, 1]
]

/-- The volume of a unit cube -/
def unit_cube_volume : ℝ := 1

/-- Proof of the octahedron to cube volume ratio -/
theorem octahedron_cube_volume_ratio_proof :
  let octahedron_volume := (Real.sqrt 3) / 3
  octahedron_volume / unit_cube_volume = octahedron_cube_volume_ratio := by
  sorry

#check octahedron_cube_volume_ratio_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_proof_l115_11567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_from_tan_l115_11547

theorem sin_double_angle_from_tan (θ : ℝ) (h : Real.tan θ = -3/5) : 
  Real.sin (2 * θ) = -15/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_from_tan_l115_11547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_arrangements_l115_11522

/-- A type representing a valid arrangement of numbers in an n × 3 rectangle -/
def ValidArrangement (n : ℕ) := 
  Fin n → Fin 3 → Fin (3*n)

/-- Predicate to check if an arrangement satisfies the required properties -/
def IsValidArrangement (n : ℕ) (arr : ValidArrangement n) : Prop :=
  let rowSum (i : Fin n) := (Finset.sum Finset.univ (λ j => (arr i j).val + 1))
  let colSum (j : Fin 3) := (Finset.sum Finset.univ (λ i => (arr i j).val + 1))
  (∀ i j : Fin n, rowSum i = rowSum j) ∧ 
  (∀ i j : Fin 3, colSum i = colSum j) ∧
  (∀ i : Fin n, 6 ∣ rowSum i) ∧
  (∀ j : Fin 3, 6 ∣ colSum j) ∧
  (Finset.card (Finset.image (λ p : Fin n × Fin 3 => (arr p.1 p.2).val + 1) 
    (Finset.univ.product Finset.univ)) = 3*n)

/-- The main theorem statement -/
theorem infinitely_many_valid_arrangements :
  ∀ k : ℕ, ∃ n > k, n ≡ 9 [MOD 12] ∧ 
  ∃ arr : ValidArrangement n, IsValidArrangement n arr := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_arrangements_l115_11522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_theorem_l115_11560

/-- Calculates the percentage change between two values -/
noncomputable def percentageChange (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem membership_change_theorem :
  let initialMembers : ℝ := 100
  let fallMembers := initialMembers * (1 + 0.05)
  let springMembers := fallMembers * (1 - 0.19)
  abs (percentageChange initialMembers springMembers + 19.05) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_theorem_l115_11560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l115_11520

theorem triangle_sine_ratio (A B C : ℝ) (a b c : ℝ) : 
  a = 2 → b = 3 → c = 4 → 
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) → 
  (b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) → 
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) → 
  (Real.sin (2*A)) / (Real.sin B) = 7/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l115_11520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l115_11559

-- Define the basic elements
variable (α : Type) -- Plane α
variable (a b : Type) -- Lines a and b

-- Define the relationships
def parallel (l p : Type) : Prop := sorry
def intersects (l1 l2 : Type) : Prop := sorry

-- State the theorem
theorem line_plane_relationship 
  (α : Type) (a b : Type)
  (h1 : intersects a b) 
  (h2 : parallel a α) : 
  parallel b α ∨ intersects b α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l115_11559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problem_l115_11589

theorem sqrt_problem (h : Real.sqrt 102.01 = 10.1) : 
  Set.image (fun x => x * Real.sqrt 1.0201) {-1, 1} = Set.image (fun x => x * 1.01) {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problem_l115_11589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_position_l115_11594

/-- The x-coordinate of the vertex of a quadratic function f(x) = ax^2 + bx + c -/
noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

/-- Parabola 1: f(x) = x^2 - 1/2x + 2 -/
noncomputable def f (x : ℝ) : ℝ := x^2 - 1/2 * x + 2

/-- Parabola 2: g(x) = x^2 + 1/2x + 2 -/
noncomputable def g (x : ℝ) : ℝ := x^2 + 1/2 * x + 2

theorem parabola_position :
  vertex_x 1 (-1/2) > vertex_x 1 (1/2) :=
by
  -- Unfold the definition of vertex_x
  unfold vertex_x
  -- Simplify the expressions
  simp
  -- The inequality becomes 1/4 > -1/4, which is true
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_position_l115_11594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l115_11584

-- Define the three lines
def line1 (x : ℝ) : ℝ := x + 4
def line2 (x : ℝ) : ℝ := -3 * x + 9
def line3 : ℝ := 2

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ 
    ((y = line1 x) ∨ (y = line2 x) ∨ (y = line3)) ∧
    (y ≤ line1 x) ∧ (y ≤ line2 x) ∧ (y ≥ line3)}

-- State the theorem
theorem triangle_area : MeasureTheory.volume triangle = 169 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l115_11584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_theorem_l115_11515

/-- A function that satisfies the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x ≠ 0 → x * (f (2 * f y - x)) + y^2 * (f (2 * x - f y)) = (f x)^2 / x + f (y * f y)

/-- The theorem stating that any function satisfying the equation must be x² or 0 -/
theorem solution_theorem (f : ℤ → ℤ) (h : SatisfiesEquation f) :
  (∀ x : ℤ, f x = x^2) ∨ (∀ x : ℤ, f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_theorem_l115_11515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_array_count_l115_11564

/-- Represents a 4x4 array with entries from 1 to 16 -/
def Array4x4 := Fin 4 → Fin 4 → Fin 16

/-- Checks if the entries in a row are in increasing order -/
def RowIncreasing (a : Array4x4) (row : Fin 4) : Prop :=
  ∀ i j : Fin 4, i < j → a row i < a row j

/-- Checks if the entries in a column are in increasing order -/
def ColumnIncreasing (a : Array4x4) (col : Fin 4) : Prop :=
  ∀ i j : Fin 4, i < j → a i col < a j col

/-- Checks if all entries in the array are distinct -/
def AllDistinct (a : Array4x4) : Prop :=
  ∀ i j k l : Fin 4, (i ≠ k ∨ j ≠ l) → a i j ≠ a k l

/-- Defines a valid array according to the problem conditions -/
def ValidArray (a : Array4x4) : Prop :=
  (∀ row : Fin 4, RowIncreasing a row) ∧
  (∀ col : Fin 4, ColumnIncreasing a col) ∧
  AllDistinct a

/-- The main theorem stating the number of valid arrays -/
theorem valid_array_count :
  ∃ s : Finset Array4x4, s.card = 13824 ∧ ∀ a ∈ s, ValidArray a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_array_count_l115_11564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grouping_theorem_l115_11591

/-- The number of ways to divide 4 men and 5 women into three groups of three people,
    with at least one man and one woman in each group, considering identically sized
    groups as indistinguishable. -/
def grouping_ways : ℕ := 180

/-- The total number of men -/
def num_men : ℕ := 4

/-- The total number of women -/
def num_women : ℕ := 5

/-- The size of each group -/
def group_size : ℕ := 3

/-- The number of groups -/
def num_groups : ℕ := 3

/-- Theorem stating that the number of ways to divide the people into groups
    under the given conditions is equal to grouping_ways -/
theorem grouping_theorem :
  (num_men = 4) →
  (num_women = 5) →
  (group_size = 3) →
  (num_groups = 3) →
  (∀ g : Fin num_groups, ∃ m w : ℕ, m ≥ 1 ∧ w ≥ 1 ∧ m + w = group_size) →
  grouping_ways = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grouping_theorem_l115_11591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_twenty_equals_negative_340_l115_11533

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  third_fifth_product : a 3 * a 5 = 12
  second_term_zero : a 2 = 0
  first_term_positive : 0 < a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem sum_twenty_equals_negative_340 (seq : ArithmeticSequence) : 
  sum_n seq 20 = -340 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_twenty_equals_negative_340_l115_11533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisible_by_g_iff_coprime_l115_11516

/-- Polynomial f(x) = 1 + x^n + x^(2n) + ... + x^(mn) -/
def f (m n : ℕ) (x : ℝ) : ℝ := (Finset.range (m + 1)).sum (λ i ↦ x^(i * n))

/-- Polynomial g(x) = 1 + x + x^2 + ... + x^m -/
def g (m : ℕ) (x : ℝ) : ℝ := (Finset.range (m + 1)).sum (λ i ↦ x^i)

/-- f is divisible by g if and only if n and m+1 are coprime -/
theorem f_divisible_by_g_iff_coprime (m n : ℕ) :
  (∀ x : ℝ, ∃ q : ℝ, f m n x = g m x * q) ↔ Nat.Coprime n (m + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisible_by_g_iff_coprime_l115_11516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_factor_proof_l115_11585

theorem other_factor_proof (w : ℕ) : 
  w > 0 →
  (936 * w).factorization 2 ≥ 5 →
  (936 * w).factorization 3 ≥ 3 →
  w ≥ 156 →
  (936 * 156) / w = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_factor_proof_l115_11585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ajay_walking_time_l115_11534

/-- Ajay's walking speed in km/hour -/
noncomputable def walking_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem ajay_walking_time 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 10) 
  (h2 : initial_time = 5) 
  (h3 : final_distance = 50) :
  walking_speed initial_distance initial_time * final_distance = 25 := by
  sorry

#check ajay_walking_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ajay_walking_time_l115_11534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l115_11553

/-- Definition of the sequence a_n -/
noncomputable def a (n : ℕ+) : ℝ := 3 * (n : ℝ)

/-- Definition of S_n (sum of first n terms of a_n) -/
noncomputable def S (n : ℕ+) : ℝ := (3/2) * (n : ℝ)^2 + (3/2) * (n : ℝ)

/-- Definition of b_n -/
noncomputable def b (n : ℕ+) : ℝ := a n + (2 : ℝ)^(n : ℝ)

/-- Definition of T_n (sum of first n terms of b_n) -/
noncomputable def T (n : ℕ+) : ℝ := (3 * (n : ℝ) * (1 + (n : ℝ))) / 2 + (2 : ℝ)^((n : ℝ) + 1) - 2

/-- Main theorem -/
theorem sequence_properties (n : ℕ+) :
  (a n = 3 * (n : ℝ)) ∧ 
  (T n = (3 * (n : ℝ) * (1 + (n : ℝ))) / 2 + (2 : ℝ)^((n : ℝ) + 1) - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l115_11553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_4_l115_11566

-- Define the semi-circle C
noncomputable def semi_circle (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l
def line (ρ θ : ℝ) : Prop := ρ * (Real.sin θ + Real.sqrt 3 * Real.cos θ) = 5 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := θ = Real.pi / 3

-- Theorem statement
theorem length_PQ_is_4 :
  ∃ (ρ_P ρ_Q : ℝ),
    semi_circle (Real.pi / 3) = ρ_P ∧
    line ρ_Q (Real.pi / 3) ∧
    ray_OM (Real.pi / 3) ∧
    ρ_Q - ρ_P = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_4_l115_11566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approx_14_l115_11539

/-- The depth of a cylindrical well given its diameter, cost per cubic meter, and total cost -/
noncomputable def well_depth (diameter : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let volume := total_cost / cost_per_cubic_meter
  let radius := diameter / 2
  volume / (Real.pi * radius^2)

/-- Theorem stating that the depth of the well is approximately 14 meters -/
theorem well_depth_approx_14 :
  ∃ ε > 0, |well_depth 3 19 1880.2432031734913 - 14| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approx_14_l115_11539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_negative_implies_a_range_l115_11542

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + (1 + a) / x - a * Real.log x

theorem function_negative_implies_a_range (a : ℝ) (h_a : a > -1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 e ∧ f a x₀ < 0) →
  a > (e^2 + 1) / (e - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_negative_implies_a_range_l115_11542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_gain_percentage_l115_11510

/-- Baker's cake ingredients and pricing --/
structure BakerData where
  milk_cost : ℚ
  flour_cost : ℚ
  sugar_cost : ℚ
  water_percent : ℚ
  num_cakes : ℕ
  selling_price : ℚ

/-- Calculate the percentage of gain for the baker --/
def calculate_gain_percentage (data : BakerData) : ℚ :=
  let total_cost := data.milk_cost + data.flour_cost + data.sugar_cost
  let cost_per_cake := total_cost / data.num_cakes
  let gain_per_cake := data.selling_price - cost_per_cake
  (gain_per_cake / cost_per_cake) * 100

/-- Theorem stating that the percentage of gain is 900% --/
theorem baker_gain_percentage :
  let data := BakerData.mk 12 8 10 20 5 60
  calculate_gain_percentage data = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_gain_percentage_l115_11510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yeast_count_correct_l115_11573

/-- Represents the dimensions and properties of a hemocytometer -/
structure Hemocytometer where
  gridSize : ℝ
  smallSquares : ℕ
  thickness : ℝ

/-- Represents the experimental conditions for yeast counting -/
structure YeastExperiment where
  hemocytometer : Hemocytometer
  dilutionFactor : ℕ
  avgYeastPerSmallSquare : ℕ
  sampleVolume : ℝ

/-- Calculates the total number of yeast cells in the given sample volume -/
noncomputable def calculateTotalYeast (experiment : YeastExperiment) : ℝ :=
  let smallSquareVolume := experiment.hemocytometer.gridSize * experiment.hemocytometer.gridSize * 
                           experiment.hemocytometer.thickness / experiment.hemocytometer.smallSquares
  (experiment.avgYeastPerSmallSquare : ℝ) / smallSquareVolume * 
  experiment.dilutionFactor * experiment.sampleVolume

theorem yeast_count_correct (experiment : YeastExperiment) 
  (h1 : experiment.hemocytometer.gridSize = 1)
  (h2 : experiment.hemocytometer.smallSquares = 400)
  (h3 : experiment.hemocytometer.thickness = 0.1)
  (h4 : experiment.dilutionFactor = 10)
  (h5 : experiment.avgYeastPerSmallSquare = 5)
  (h6 : experiment.sampleVolume = 10) :
  calculateTotalYeast experiment = 2e9 := by
  sorry

-- Note: This example will not compute due to the noncomputable definition
example : calculateTotalYeast {
  hemocytometer := { gridSize := 1, smallSquares := 400, thickness := 0.1 },
  dilutionFactor := 10,
  avgYeastPerSmallSquare := 5,
  sampleVolume := 10
} = 2e9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yeast_count_correct_l115_11573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_historical_sets_l115_11546

/-- A historical set is a set of three non-negative integers {x, y, z} where x < y < z and {z - y, y - x} = {a, b} for some given constants 0 < a < b. -/
def IsHistoricalSet (s : Set ℕ) (a b : ℕ) : Prop :=
  ∃ x y z : ℕ, s = {x, y, z} ∧ x < y ∧ y < z ∧ ({z - y, y - x} : Set ℕ) = {a, b} ∧ 0 < a ∧ a < b

/-- A partition of a set S is a collection of non-empty, pairwise disjoint subsets of S whose union is S. -/
def IsPartition (P : Set (Set ℕ)) (S : Set ℕ) : Prop :=
  (∀ s, s ∈ P → s.Nonempty) ∧ 
  (∀ s t, s ∈ P → t ∈ P → s ≠ t → s ∩ t = ∅) ∧
  (⋃₀ P = S)

/-- There exists a partition of the set of non-negative integers into historical sets. -/
theorem partition_into_historical_sets (a b : ℕ) (h : 0 < a ∧ a < b) : 
  ∃ P : Set (Set ℕ), IsPartition P (Set.univ : Set ℕ) ∧ 
    ∀ s ∈ P, IsHistoricalSet s a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_historical_sets_l115_11546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l115_11596

theorem solve_exponential_equation :
  ∀ x : ℝ, (2 : ℝ)^(x - 3) = 16 → x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l115_11596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_le_two_l115_11557

/-- A function f(x) that depends on a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + x

/-- Theorem stating that if f is increasing on (0,1), then a ≤ 2 -/
theorem f_increasing_implies_a_le_two (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, StrictMonoOn (f a) (Set.Ioo 0 1)) →
  a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_le_two_l115_11557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_l115_11543

noncomputable def cycling_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

noncomputable def total_time (distances : List ℝ) (speeds : List ℝ) : ℝ :=
  (List.zip distances speeds).map (λ (d, s) => cycling_time d s) |> List.sum

theorem cycling_time_difference : 
  let constant_speed : ℝ := 5
  let total_distance : ℝ := 4 * 3
  let actual_distances : List ℝ := [3, 3, 3, 3]
  let actual_speeds : List ℝ := [6, 4, 5, 3]
  let constant_time := cycling_time total_distance constant_speed
  let actual_time := total_time actual_distances actual_speeds
  (constant_time - actual_time) * 60 = 27 := by
  sorry

#check cycling_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_l115_11543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_lattice_points_count_l115_11581

/-- A lattice point is a point (a,b) such that both a and b are integers. -/
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (a b : ℤ), p = (↑a, ↑b)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The set of lattice points that are exactly twice as close to (0,0) as they are to (15,0) -/
def SpecialLatticePoints : Set (ℝ × ℝ) :=
  {p | LatticePoint p ∧ distance p (0, 0) = (1/2) * distance p (15, 0)}

theorem special_lattice_points_count :
  Finset.card (Finset.filter (fun p => (p.1 + 5)^2 + p.2^2 = 100) (Finset.product (Finset.range 31) (Finset.range 21))) = 12 :=
by sorry

#eval Finset.card (Finset.filter (fun p => (p.1 + 5)^2 + p.2^2 = 100) (Finset.product (Finset.range 31) (Finset.range 21)))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_lattice_points_count_l115_11581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l115_11569

/-- The function f(x) = 2sin(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

/-- The theorem stating that under given conditions, φ = π/3 -/
theorem phi_value (ω φ : ℝ) :
  ω > 0 →
  |φ| < π/2 →
  (∃ (d : ℝ), d = π/3 ∧ 
    ∀ (x y : ℝ), f ω φ x = 1 → f ω φ y = 1 → x ≠ y → |x - y| ≥ d) →
  (∀ (x : ℝ), f ω φ x ≤ f ω φ (π/12)) →
  φ = π/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l115_11569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_inequality_solution_set_l115_11503

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x - 5/2| + |x - 1/2|

-- Theorem for the minimum value of f
theorem f_min_value : ∀ x : ℝ, f x ≥ 2 := by
  sorry

-- Theorem for the solution set of the inequality
theorem f_inequality_solution_set :
  ∀ x : ℝ, f x ≤ x + 4 ↔ -1/3 ≤ x ∧ x ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_inequality_solution_set_l115_11503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l115_11527

noncomputable section

def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x + Real.pi/3) + Real.sqrt 3

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/6 → f x ≤ 2) ∧
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/6 → f x ≥ -1) ∧
  (f (Real.pi/12) = 2) ∧
  (f (-Real.pi/4) = -1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l115_11527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_four_thirds_l115_11529

/-- The area enclosed by y=x^2-1, x=2, and y=0 -/
noncomputable def enclosed_area : ℝ := ∫ x in (1)..(2), (x^2 - 1)

/-- Theorem stating that the enclosed area is equal to 4/3 -/
theorem area_is_four_thirds : enclosed_area = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_four_thirds_l115_11529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decomposition_l115_11598

/-- A function h is even if h(-x) = h(x) for all x -/
def IsEven (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

/-- The main theorem -/
theorem function_decomposition (h g : ℝ → ℝ) 
  (h_even : IsEven h) (g_odd : IsOdd g) 
  (hg_bound : ∀ x : ℝ, x ≠ 1 → h x + g x ≤ 1 / (x - 1)) :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → h x = 1 / (x^2 - 1) ∧ g x = x / (x^2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decomposition_l115_11598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_coloring_theorem_l115_11524

theorem subset_coloring_theorem :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    ∀ coloring : Finset (Fin n) → Fin 1391,
    ∃ A B : Finset (Fin n),
      A ≠ ∅ ∧ B ≠ ∅ ∧ A ∩ B = ∅ ∧
      coloring A = coloring B ∧ coloring A = coloring (A ∪ B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_coloring_theorem_l115_11524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_l115_11571

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the constant a
def a : ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_nonneg : ∀ x, x ≥ 0 → f x = 1 / (2^x) + a

-- Theorem to prove
theorem f_neg_one : f (-1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_l115_11571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_specific_a_l115_11514

theorem no_solution_for_specific_a (a : ℝ) : 
  a ∈ ({-3, -1/2, 2} : Set ℝ) → 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → 
  1 - (2*x + 2*a - 2) / (x^2 - 1) ≠ (x + a) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_specific_a_l115_11514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l115_11551

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x + 1 = 0

-- Define the roots
noncomputable def root1 : ℝ := (-3 + Real.sqrt 5) / 2
noncomputable def root2 : ℝ := (-3 - Real.sqrt 5) / 2

-- Theorem statement
theorem quadratic_roots :
  (∃ (x y : ℝ), x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y) ∧
  quadratic_equation root1 ∧
  quadratic_equation root2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l115_11551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l115_11550

/-- The fixed cost of production -/
def fixed_cost : ℝ := 7500

/-- The additional cost per instrument -/
def additional_cost : ℝ := 100

/-- The total revenue function -/
noncomputable def H (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 200 then 400 * x - x^2
  else 40000

/-- The profit function -/
noncomputable def profit (x : ℝ) : ℝ :=
  H x - (fixed_cost + additional_cost * x)

/-- The theorem stating the maximum profit and the corresponding production volume -/
theorem max_profit :
  ∃ (max_x : ℝ), max_x = 150 ∧
  ∀ (x : ℝ), profit x ≤ profit max_x ∧ profit max_x = 15000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l115_11550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_complement_range_l115_11526

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define function f(x) = lg(x-1)
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 10

-- Define function g(x) = √(x^2 + 2x + 10)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 10)

-- Define set A as the domain of f
def A : Set ℝ := {x | x > 1}

-- Define set B as the range of g
def B : Set ℝ := {y | ∃ x, g x = y}

-- Theorem to prove
theorem domain_intersection_complement_range : 
  A ∩ (Set.compl B) = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_complement_range_l115_11526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l115_11531

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: Given a man who can row upstream at 22 kmph and downstream at 42 kmph, 
    his speed in still water is 32 kmph -/
theorem mans_speed_in_still_water :
  speed_in_still_water 22 42 = 32 := by
  unfold speed_in_still_water
  norm_num

-- Note: We remove the #eval line as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l115_11531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_to_focus_l115_11593

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The minimum sum of distances from two points on the parabola to the focus is 4 -/
theorem min_sum_distances_to_focus :
  ∀ A B : Point,
  A ∈ Parabola → B ∈ Parabola →
  ∃ (line : ℝ → Point),
  line 0 = focus ∧ 
  (∃ t1 t2 : ℝ, line t1 = A ∧ line t2 = B) →
  4 ≤ distance A focus + distance B focus :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_to_focus_l115_11593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_breaking_load_specific_case_l115_11590

/-- Breaking load formula for cylindrical bars -/
noncomputable def breaking_load (T : ℝ) (H : ℝ) : ℝ := (30 * T^3) / H^2

/-- Theorem stating that the breaking load is 22.5 for T = 3 and H = 6 -/
theorem breaking_load_specific_case : breaking_load 3 6 = 22.5 := by
  -- Unfold the definition of breaking_load
  unfold breaking_load
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_breaking_load_specific_case_l115_11590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_of_sequence_l115_11509

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / (3 * sequence_a (n + 1) + 1)

theorem twentieth_term_of_sequence :
  sequence_a 20 = 1 / 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_of_sequence_l115_11509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_equation_solutions_l115_11555

theorem sin_squared_equation_solutions (x : ℝ) :
  (Real.sin x)^2 + (Real.sin (2*x))^2 + (Real.sin (3*x))^2 = 2 ↔
  (∃ n : ℤ, x = π/4 + π*n/2 ∨ x = π/2 + π*n ∨ x = π/6 + π*n ∨ x = -π/6 + π*n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_equation_solutions_l115_11555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l115_11530

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x^2 - m * x + 2)

def has_full_domain (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - m * x + 2 ≥ 0

def M : Set ℝ := {m : ℝ | has_full_domain m}

theorem range_of_m : M = Set.Icc 0 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l115_11530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lists_count_l115_11545

def valid_list (n : ℕ) : Prop :=
  ∃ (a : Fin n → ℕ),
    (∀ i, a i ∈ Finset.range n) ∧
    (∀ i : Fin n, i.val ≥ 2 → 
      ∃ j : Fin n, j.val < i.val ∧ (a j = a i + 1 ∨ a j = a i - 1))

def count_valid_lists : ℕ → ℕ
  | 0 => 1  -- Base case for 0
  | 1 => 1  -- Base case for 1
  | 2 => 2
  | n+1 => 2 * count_valid_lists n

theorem valid_lists_count : count_valid_lists 15 = 16384 := by
  -- Unfold the definition and calculate
  unfold count_valid_lists
  -- You would typically prove this by induction, but for now, we'll use sorry
  sorry

-- Optional: You can add a test to check the result
#eval count_valid_lists 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lists_count_l115_11545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_179_l115_11548

def sequence_prop (a₁ a₂ a₃ a₄ a₅ : ℕ) : Prop :=
  (a₁ = 3) ∧
  (a₄ = 48) ∧
  (a₂ = (a₁ + a₃) / 4) ∧
  (a₃ = (a₂ + a₄) / 4) ∧
  (a₄ = (a₃ + a₅) / 4)

theorem fifth_term_is_179 (a₁ a₂ a₃ a₄ a₅ : ℕ) :
  sequence_prop a₁ a₂ a₃ a₄ a₅ → a₅ = 179 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_179_l115_11548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_two_l115_11572

/-- Represents the number of black balls in the urn -/
def n : ℕ := 5

/-- Represents the number of white balls drawn -/
def X : ℕ → ℝ := sorry

/-- The probability of drawing a white ball -/
noncomputable def p : ℝ := 5 / (n + 5)

/-- The number of trials (draws) -/
def num_trials : ℕ := 4

/-- The variance of X is 1 -/
axiom h_variance : num_trials * p * (1 - p) = 1

/-- The expected value of X -/
noncomputable def E_X : ℝ := num_trials * p

/-- Theorem: If the variance of X is 1, then the expected value of X is 2 -/
theorem expected_value_is_two : E_X = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_two_l115_11572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_theorem_l115_11588

/-- The side length of the equilateral triangle -/
noncomputable def s : ℝ := 800

/-- The circumradius of an equilateral triangle -/
noncomputable def R (s : ℝ) : ℝ := (s * Real.sqrt 3) / 3

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Given an equilateral triangle ABC with side length s, 
    points P and Q outside the plane of ABC on the same side, 
    such that PA = PB = PC and QA = QB = QC, 
    and the planes of triangle PAB and triangle QAB form a 90° dihedral angle,
    there exists a point O equidistant from A, B, C, P, and Q,
    and this distance is equal to the circumradius of triangle ABC -/
theorem equidistant_point_theorem 
  (ABC : Triangle3D) 
  (P Q : Point3D) 
  (h_equilateral : ABC.A.x = ABC.B.x ∧ ABC.A.y = ABC.B.y ∧ ABC.A.z = ABC.B.z)
  (h_side_length : (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 + (ABC.A.z - ABC.B.z)^2 = s^2)
  (h_P_equidistant : (P.x - ABC.A.x)^2 + (P.y - ABC.A.y)^2 + (P.z - ABC.A.z)^2 =
                     (P.x - ABC.B.x)^2 + (P.y - ABC.B.y)^2 + (P.z - ABC.B.z)^2)
  (h_Q_equidistant : (Q.x - ABC.A.x)^2 + (Q.y - ABC.A.y)^2 + (Q.z - ABC.A.z)^2 =
                     (Q.x - ABC.B.x)^2 + (Q.y - ABC.B.y)^2 + (Q.z - ABC.B.z)^2)
  (h_dihedral_angle : True) : -- Placeholder for 90° dihedral angle condition
  ∃ (O : Point3D), 
    (O.x - ABC.A.x)^2 + (O.y - ABC.A.y)^2 + (O.z - ABC.A.z)^2 = (R s)^2 ∧
    (O.x - ABC.B.x)^2 + (O.y - ABC.B.y)^2 + (O.z - ABC.B.z)^2 = (R s)^2 ∧
    (O.x - ABC.C.x)^2 + (O.y - ABC.C.y)^2 + (O.z - ABC.C.z)^2 = (R s)^2 ∧
    (O.x - P.x)^2 + (O.y - P.y)^2 + (O.z - P.z)^2 = (R s)^2 ∧
    (O.x - Q.x)^2 + (O.y - Q.y)^2 + (O.z - Q.z)^2 = (R s)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_theorem_l115_11588
