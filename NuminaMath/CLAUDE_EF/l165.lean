import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l165_16515

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + (1 - a) * x

-- Theorem for part (I)
theorem part_one : {a : ℝ | f a 2 < 0} = Set.Ioi (4/3) := by sorry

-- Theorem for part (II)
theorem part_two : {a : ℝ | ∀ x, f a x ≥ 0} = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l165_16515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_80_by_150_percent_l165_16547

/-- Increases a number by a given percentage. -/
noncomputable def increase_by_percentage (n : ℝ) (p : ℝ) : ℝ := n * (1 + p / 100)

/-- Theorem: Increasing 80 by 150% results in 200. -/
theorem increase_80_by_150_percent :
  increase_by_percentage 80 150 = 200 := by
  -- Unfold the definition of increase_by_percentage
  unfold increase_by_percentage
  -- Simplify the expression
  simp [mul_add, mul_div_cancel']
  -- Check that the result is equal to 200
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_80_by_150_percent_l165_16547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l165_16506

theorem function_characterization (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f (f m) + f n) :
  (∀ n, f n = 0) ∨ 
  (∀ n, f n = n) ∨ 
  (∃ n₀ : ℕ, n₀ ≥ 2 ∧ 
    ∃ a : ℕ → ℕ, a 0 = 0 ∧ 
    ∀ n, ∃ K r : ℕ, r < n₀ ∧ f n = a r * n₀ + K * n₀) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l165_16506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l165_16556

theorem smallest_integer_in_set (n : ℤ) : 
  (∀ i : ℕ, i < 7 → (n + i : ℤ) ≤ n + 6) →
  (n + 6 : ℚ) < 3 * ((7 * n + 21) / 7) →
  n ≥ -1 :=
by
  intros h1 h2
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l165_16556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_is_22_5_main_theorem_l165_16518

/-- Represents a two-leg relay race with given times for each runner -/
structure RelayRace where
  dawson_time : ℚ
  henry_time : ℚ

/-- Calculates the average time per leg for a given relay race -/
def average_time_per_leg (race : RelayRace) : ℚ :=
  (race.dawson_time + race.henry_time) / 2

/-- Theorem stating that for the given race times, the average time per leg is 22.5 seconds -/
theorem average_time_is_22_5 (race : RelayRace) 
  (h1 : race.dawson_time = 38)
  (h2 : race.henry_time = 7) : 
  average_time_per_leg race = 45/2 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ (race : RelayRace), 
  race.dawson_time = 38 ∧ 
  race.henry_time = 7 ∧ 
  average_time_per_leg race = 45/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_is_22_5_main_theorem_l165_16518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_value_a_n_minus_one_f_decreasing_f_minimum_l165_16579

-- Define the sequence a_n
def a : ℕ → ℝ
  | 0 => 4  -- Add a case for 0 to cover all natural numbers
  | 1 => 4
  | n + 1 => 3 * a n - 2

-- Define the property that k must satisfy
def satisfies_inequality (k : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → k * (a n - 1) ≥ 2 * n - 5

-- State the theorem
theorem minimum_k_value :
  ∃ k : ℝ, k = 1/27 ∧ satisfies_inequality k ∧
  ∀ k' : ℝ, k' < k → ¬ satisfies_inequality k' :=
sorry

-- Prove that a_n - 1 = 3^n for n ≥ 1
theorem a_n_minus_one (n : ℕ) (h : n ≥ 1) : a n - 1 = 3^n := by
  sorry

-- Prove that f(n) = (2n - 5) / 3^n is decreasing for n ≥ 3
theorem f_decreasing (n : ℕ) (h : n ≥ 3) :
  (2 * (n + 1) - 5) / 3^(n + 1) ≤ (2 * n - 5) / 3^n := by
  sorry

-- Prove that the minimum value of f(n) for n ≥ 3 occurs at n = 3
theorem f_minimum :
  ∀ n : ℕ, n ≥ 3 → (2 * n - 5) / 3^n ≥ 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_value_a_n_minus_one_f_decreasing_f_minimum_l165_16579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l165_16544

/-- The smallest positive value t such that the graph of r = 2sin(θ) for 0 ≤ θ ≤ t 
    forms a complete circle is π. -/
theorem smallest_t_for_complete_circle (θ : ℝ) : 
  let r : ℝ → ℝ := λ θ => 2 * Real.sin θ
  ∃ t : ℝ, t > 0 ∧ t = Real.pi ∧ 
    (∀ θ₁ θ₂, 0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ t → r θ₁ ≠ r θ₂) ∧
    (∀ θ', 0 ≤ θ' ∧ θ' ≤ t → ∃ θ, 0 ≤ θ ∧ θ ≤ 2*Real.pi ∧ r θ' = r θ) ∧
    (∀ t' > t, ∃ θ₁ θ₂, 0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ t' ∧ r θ₁ = r θ₂) :=
by
  sorry

/-- The graph of r = 2sin(θ) is a circle. -/
axiom is_circle (θ : ℝ) : 
  let r : ℝ → ℝ := λ θ => 2 * Real.sin θ
  ∃ c : ℝ × ℝ, ∃ radius : ℝ, ∀ θ, (r θ * Real.cos θ - c.1)^2 + (r θ * Real.sin θ - c.2)^2 = radius^2

/-- The function r = 2sin(θ) is periodic with period 2π. -/
axiom is_periodic (θ : ℝ) :
  let r : ℝ → ℝ := λ θ => 2 * Real.sin θ
  ∀ θ, r (θ + 2*Real.pi) = r θ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l165_16544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_rate_is_four_l165_16550

/-- The walking scenario with Yolanda and Bob --/
structure WalkingScenario where
  totalDistance : ℝ
  yolandaRate : ℝ
  bobDistance : ℝ
  timeDifference : ℝ

/-- Bob's walking rate given the scenario --/
noncomputable def bobRate (scenario : WalkingScenario) : ℝ :=
  scenario.bobDistance / (scenario.totalDistance / scenario.yolandaRate - scenario.timeDifference)

/-- Theorem stating that Bob's walking rate is 4 miles per hour --/
theorem bob_rate_is_four (scenario : WalkingScenario)
  (h1 : scenario.totalDistance = 31)
  (h2 : scenario.yolandaRate = 3)
  (h3 : scenario.bobDistance = 16)
  (h4 : scenario.timeDifference = 1) :
  bobRate scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_rate_is_four_l165_16550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l165_16565

/-- The number of days it takes for p and q to complete a work together,
    given p's efficiency and completion time. -/
noncomputable def days_to_complete_together (p_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  let q_efficiency := p_efficiency / 1.5
  let combined_efficiency := p_efficiency + q_efficiency
  1 / combined_efficiency

/-- Theorem stating that p and q working together will complete the work in 15 days,
    given the conditions from the problem. -/
theorem work_completion_time :
  days_to_complete_together (1 / 25) 25 = 15 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l165_16565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l165_16548

/-- Represents the possible outcomes of a coin flip -/
inductive CoinFlip
| Heads
| Tails
deriving Repr, DecidableEq

/-- Represents the set of coins -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
deriving Repr, DecidableEq

/-- Calculates the value of heads in cents -/
def heads_value (cs : CoinSet) : ℕ :=
  (if cs.penny = CoinFlip.Heads then 1 else 0) +
  (if cs.nickel = CoinFlip.Heads then 5 else 0) +
  (if cs.dime = CoinFlip.Heads then 10 else 0) +
  (if cs.quarter = CoinFlip.Heads then 25 else 0) +
  (if cs.half_dollar = CoinFlip.Heads then 50 else 0)

/-- Defines the set of all possible outcomes -/
def all_outcomes : Finset CoinSet := sorry

/-- Defines the set of favorable outcomes (at least 40 cents in heads) -/
def favorable_outcomes : Finset CoinSet :=
  all_outcomes.filter (λ cs => heads_value cs ≥ 40)

/-- The main theorem: probability of getting at least 40 cents in heads -/
theorem probability_at_least_40_cents :
  (favorable_outcomes.card : ℚ) / all_outcomes.card = 19 / 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l165_16548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l165_16582

noncomputable def calculate_savings (initial_amount : ℝ) : ℝ :=
  let wife_share := (2/5) * initial_amount
  let remaining_after_wife := initial_amount - wife_share
  let first_son_share := (3/10) * remaining_after_wife
  let remaining_after_first_son := remaining_after_wife - first_son_share
  let second_son_share := 0.30 * remaining_after_first_son
  let remaining_after_second_son := remaining_after_first_son - second_son_share
  let third_son_share := 0.20 * remaining_after_second_son
  let remaining_after_third_son := remaining_after_second_son - third_son_share
  let charity_donation := 200
  let remaining_after_donation := remaining_after_third_son - charity_donation
  let tax := 0.05 * remaining_after_donation
  remaining_after_donation - tax

theorem savings_calculation (initial_amount : ℝ) 
  (h : initial_amount = 5000) : 
  calculate_savings initial_amount = 927.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l165_16582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangles_ratio_l165_16585

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Real × Real
  sideLength : Real

/-- Rotates a triangle about its center by a given angle -/
def rotateTriangle (t : EquilateralTriangle) (α : Real) : EquilateralTriangle :=
  { t with center := t.center }  -- Placeholder implementation

/-- Gets the three equilateral triangles formed by the intersection of corresponding side lines -/
def getIntersectionTriangles (t1 t2 : EquilateralTriangle) : Array EquilateralTriangle :=
  #[t1, t2, t1]  -- Placeholder implementation

/-- Calculates the ratio of side lengths of the given triangles -/
def calculateSideLengthRatio (triangles : Array EquilateralTriangle) : Array Real :=
  #[1, 1, 1]  -- Placeholder implementation

/-- Given two equilateral triangles ABC and A'B'C' where A'B'C' is formed by rotating ABC 
    about its center O by an angle α, the ratio of the side lengths of the three equilateral 
    triangles formed by the intersection points of corresponding side lines is:
    1 / cos(60° - α/2) : 1 / cos(α/2) : 1 / cos(60° + α/2) -/
theorem intersection_triangles_ratio (α : Real) :
  let ABC : EquilateralTriangle := { center := (0, 0), sideLength := 1 }
  let A'B'C' := rotateTriangle ABC α
  let intersectionTriangles := getIntersectionTriangles ABC A'B'C'
  let ratio := calculateSideLengthRatio intersectionTriangles
  ratio = #[1 / Real.cos (π/3 - α/2), 1 / Real.cos (α/2), 1 / Real.cos (π/3 + α/2)] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangles_ratio_l165_16585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bhanu_petrol_percentage_l165_16523

/-- Represents Bhanu's financial situation -/
structure BhanuFinances where
  income : ℚ
  petrolExpenditure : ℚ
  houseRentExpenditure : ℚ
  houseRentPercentage : ℚ

/-- Calculates the percentage of income spent on petrol -/
def petrolPercentage (b : BhanuFinances) : ℚ :=
  (b.petrolExpenditure / b.income) * 100

/-- Theorem stating that Bhanu spends 30% of his income on petrol -/
theorem bhanu_petrol_percentage (b : BhanuFinances) 
  (h1 : b.petrolExpenditure = 300)
  (h2 : b.houseRentExpenditure = 98)
  (h3 : b.houseRentPercentage = 14)
  (h4 : b.houseRentExpenditure = (b.houseRentPercentage / 100) * (b.income - b.petrolExpenditure)) :
  petrolPercentage b = 30 := by
  sorry

#check bhanu_petrol_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bhanu_petrol_percentage_l165_16523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_circle_through_foci_and_ellipse_l165_16597

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 9 * x^2 + 36 * y^2 = 36

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the foci of the ellipse
def foci (x : ℝ) : Prop := x^2 = 3

-- Theorem statement
theorem min_radius_circle_through_foci_and_ellipse :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), ellipse x y → circle_eq x y r → (x = 0 ∨ y = 0)) ∧
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (ellipse x₁ y₁ ∧ circle_eq x₁ y₁ r) ∧
    (ellipse x₂ y₂ ∧ circle_eq x₂ y₂ r) ∧
    (ellipse x₃ y₃ ∧ circle_eq x₃ y₃ r) ∧
    (ellipse x₄ y₄ ∧ circle_eq x₄ y₄ r) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧
    (x₁ ≠ x₄ ∨ y₁ ≠ y₄) ∧
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₂ ≠ x₄ ∨ y₂ ≠ y₄) ∧
    (x₃ ≠ x₄ ∨ y₃ ≠ y₄)) ∧
  (∀ (x : ℝ), foci x → circle_eq x 0 r) ∧
  (∀ (r' : ℝ), r' > 0 →
    (∀ (x y : ℝ), ellipse x y → circle_eq x y r' → (x = 0 ∨ y = 0)) →
    (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
      (ellipse x₁ y₁ ∧ circle_eq x₁ y₁ r') ∧
      (ellipse x₂ y₂ ∧ circle_eq x₂ y₂ r') ∧
      (ellipse x₃ y₃ ∧ circle_eq x₃ y₃ r') ∧
      (ellipse x₄ y₄ ∧ circle_eq x₄ y₄ r') ∧
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
      (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧
      (x₁ ≠ x₄ ∨ y₁ ≠ y₄) ∧
      (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
      (x₂ ≠ x₄ ∨ y₂ ≠ y₄) ∧
      (x₃ ≠ x₄ ∨ y₃ ≠ y₄)) →
    (∀ (x : ℝ), foci x → circle_eq x 0 r') →
    r' ≥ r) →
  r = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_circle_through_foci_and_ellipse_l165_16597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l165_16512

/-- The volume of a cone with radius r and height h is (1/3) * π * r^2 * h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_volume_ratio :
  let r_C : ℝ := 15.6
  let h_C : ℝ := 29.5
  let r_D : ℝ := 29.5
  let h_D : ℝ := 15.6
  (cone_volume r_C h_C) / (cone_volume r_D h_D) = 156 / 295 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l165_16512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_solutions_area_enclosed_by_curves_l165_16583

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := x - (x + 1) * log (x + 1)

theorem f_monotonicity_and_solutions (t : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, deriv f x > 0) ∧
  (∀ x ∈ Set.Ioi 0, deriv f x < 0) ∧
  (∃ x y, x ∈ Set.Icc (-1/2 : ℝ) 1 ∧ y ∈ Set.Icc (-1/2 : ℝ) 1 ∧ x ≠ y ∧ f x = t ∧ f y = t ↔ 
    t ∈ Set.Ioc (-1/2 + 1/2 * log 2) 0) :=
sorry

theorem area_enclosed_by_curves :
  ∃ m ∈ Set.Icc (0 : ℝ) (1/2), 
    (∫ (x : ℝ) in Set.Icc 0 m, (deriv f x - log (x + 1/6))) = 1 + 2/3 * log 2 - log 3 ∧
    (∫ (x : ℝ) in Set.Icc 0 0, (deriv f x - log (x + 1/6))) = 1 + 2/3 * log 2 - log 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_solutions_area_enclosed_by_curves_l165_16583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_projection_l165_16549

/-- A line in 3D space -/
structure Line3D where
  -- Define the line properties here
  -- For simplicity, we'll just use a placeholder
  placeholder : Unit

/-- A rectangle in 3D space -/
structure Rectangle3D where
  -- Define the rectangle properties here
  parallel_side : Line3D
  -- Add other necessary properties

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane properties here
  -- For simplicity, we'll just use a placeholder
  placeholder : Unit

/-- Represents the projection of a shape onto a plane -/
inductive Projection
  | Rectangle
  | LineSegment

/-- Checks if a line is parallel to a plane -/
def isParallelTo (line : Line3D) (plane : Plane3D) : Prop :=
  sorry -- The actual implementation would go here

/-- The theorem stating that the projection of a rectangle with one side parallel to a plane
    is either a rectangle or a line segment -/
theorem rectangle_projection (rect : Rectangle3D) (plane : Plane3D) 
    (h : isParallelTo rect.parallel_side plane) :
    ∃ (proj : Projection), proj = Projection.Rectangle ∨ proj = Projection.LineSegment := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_projection_l165_16549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l165_16587

/-- The annual profit function for a manufacturer -/
noncomputable def annual_profit (x : ℝ) : ℝ := -1/3 * x^3 + 81*x - 23

/-- The derivative of the annual profit function -/
noncomputable def annual_profit_derivative (x : ℝ) : ℝ := -x^2 + 81

theorem max_profit_at_nine :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → annual_profit y ≤ annual_profit x) ∧
  x = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l165_16587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_multiple_l165_16525

theorem unique_non_multiple (S : Set ℕ) : S = {16, 21, 28, 34, 45} →
  ∃! x, x ∈ S ∧ ¬(∃ k : ℕ, x = 3 * k ∨ x = 4 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_multiple_l165_16525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_tangent_a_l165_16594

/-- A circle with center (a, 0) and radius 2 -/
def myCircle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- The line x - y + √2 = 0 -/
def myLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + Real.sqrt 2 = 0}

/-- Two sets are tangent if they intersect at exactly one point -/
def isTangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ S ∩ T

theorem circle_line_tangent_a (a : ℝ) :
  isTangent (myCircle a) myLine → a = Real.sqrt 2 ∨ a = -3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_tangent_a_l165_16594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_y_plus_z_eq_p_plus_one_l165_16567

theorem abs_y_plus_z_eq_p_plus_one (x y z : ℤ) (p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_eq : x^2 + x*z - x*y - y*z = -(p : ℤ)) : 
  |y + z| = (p : ℤ) + 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_y_plus_z_eq_p_plus_one_l165_16567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_dimension_proof_l165_16577

theorem room_dimension_proof (x : ℝ) : 
  let room_length : ℝ := x
  let room_width : ℝ := 15
  let room_height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 4
  let total_cost : ℝ := 3624
  let wall_area : ℝ := 2 * (x * room_width + x * room_height)
  let non_whitewash_area : ℝ := door_area + num_windows * window_area
  let whitewash_area : ℝ := wall_area - non_whitewash_area
  whitewash_area * whitewash_cost_per_sqft = total_cost →
  x = 18 := by
  intro h
  -- The proof steps would go here
  sorry

#check room_dimension_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_dimension_proof_l165_16577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l165_16507

theorem sin_cos_equation_solution (x : ℝ) :
  (Real.sin (7 * x) + (1 - (Real.cos (3 * x))^11 * (Real.cos (7 * x))^2)^(1/4) = 0) ↔
  (∃ n : ℤ, x = -π/14 + 2*π*n/7) ∨ (∃ s : ℤ, x = 2*π*s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l165_16507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decrease_l165_16569

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- State the theorem
theorem f_monotonic_decrease :
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 0 → ∃ y, f (y + 1) = x^2 - 2*x + 1) →
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decrease_l165_16569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labeling_exists_l165_16571

def RegularNGon (n : ℕ) := Fin n

structure Side (n : ℕ) where
  endpoints : RegularNGon n × RegularNGon n
  different : endpoints.1 ≠ endpoints.2

structure Diagonal (n : ℕ) where
  endpoints : RegularNGon n × RegularNGon n
  different : endpoints.1 ≠ endpoints.2
  not_adjacent : (endpoints.2.val - endpoints.1.val) % n ≠ 1 ∧ (endpoints.1.val - endpoints.2.val) % n ≠ 1

def EdgeOrDiagonal (n : ℕ) := Side n ⊕ Diagonal n

structure Labeling (n : ℕ) where
  label : EdgeOrDiagonal n → Fin n
  distinct_from_endpoints : ∀ e : EdgeOrDiagonal n, 
    match e with
    | Sum.inl s => label e ≠ s.endpoints.1 ∧ label e ≠ s.endpoints.2
    | Sum.inr d => label e ≠ d.endpoints.1 ∧ label e ≠ d.endpoints.2
  distinct_at_vertex : ∀ v : RegularNGon n, 
    ∀ e1 e2 : EdgeOrDiagonal n, 
    (match e1 with
     | Sum.inl s => s.endpoints.1 = v ∨ s.endpoints.2 = v
     | Sum.inr d => d.endpoints.1 = v ∨ d.endpoints.2 = v) → 
    (match e2 with
     | Sum.inl s => s.endpoints.1 = v ∨ s.endpoints.2 = v
     | Sum.inr d => d.endpoints.1 = v ∨ d.endpoints.2 = v) → 
    e1 ≠ e2 → label e1 ≠ label e2

theorem labeling_exists (n : ℕ) (h1 : n > 3) (h2 : Odd n) : 
  ∃ l : Labeling n, True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_labeling_exists_l165_16571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_words_count_l165_16593

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of words of length n that do not contain the letter B -/
def words_without_b (n : ℕ) : ℕ := (alphabet_size - 1) ^ n

/-- The total number of words of length n -/
def total_words (n : ℕ) : ℕ := alphabet_size ^ n

/-- The number of words of length n that contain the letter B at least once -/
def words_with_b (n : ℕ) : ℕ := total_words n - words_without_b n

/-- The total number of valid words (5 letters or less, containing B at least once) -/
def total_valid_words : ℕ :=
  (List.range max_word_length).map (λ n => words_with_b (n + 1)) |>.sum

theorem valid_words_count :
  total_valid_words = 1860701 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_words_count_l165_16593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_catches_alice_catch_up_time_in_minutes_l165_16574

/-- The time it takes for Tom to catch up with Alice -/
noncomputable def catchUpTime (alice_speed : ℝ) (tom_speed : ℝ) (initial_distance : ℝ) (eastward_time : ℝ) : ℝ :=
  (initial_distance) / (tom_speed - alice_speed)

/-- The main theorem stating that Tom catches up with Alice in 60 minutes -/
theorem tom_catches_alice : 
  let alice_speed : ℝ := 6
  let tom_speed : ℝ := 9
  let initial_distance : ℝ := 3
  let eastward_time : ℝ := 1/6
  catchUpTime alice_speed tom_speed initial_distance eastward_time = 1 := by
  sorry

/-- Converting the catch-up time to minutes -/
theorem catch_up_time_in_minutes :
  (catchUpTime 6 9 3 (1/6)) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_catches_alice_catch_up_time_in_minutes_l165_16574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_k_l165_16572

-- Define the function f(x) = 4x + c
def f (c : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + c

-- State the theorem
theorem intersection_point_k (c : ℤ) (k : ℤ) : 
  (∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv (f c) ∧ Function.RightInverse f_inv (f c)) →  -- f has an inverse
  (f c 2 = k) →                                        -- (2, k) is on the graph of f
  (f c k = 2) →                                        -- (2, k) is on the graph of f^(-1)
  k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_k_l165_16572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l165_16586

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 2

-- State the theorem
theorem root_interval (n : ℤ) :
  (∃ r : ℝ, f r = 0 ∧ n < r ∧ r < n + 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l165_16586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_added_approx_3_5_l165_16575

-- Define the initial solution volume
def initial_volume : ℝ := 40

-- Define the initial alcohol percentage
def initial_alcohol_percentage : ℝ := 0.05

-- Define the amount of water added
def water_added : ℝ := 6.5

-- Define the final alcohol percentage
def final_alcohol_percentage : ℝ := 0.11

-- Define the function to calculate the amount of alcohol added
def alcohol_added (x : ℝ) : ℝ := x

-- Define the function to calculate the final volume
def final_volume (x : ℝ) : ℝ := initial_volume + alcohol_added x + water_added

-- Define the function to calculate the initial amount of alcohol
def initial_alcohol : ℝ := initial_volume * initial_alcohol_percentage

-- Define the function to calculate the final amount of alcohol
def final_alcohol (x : ℝ) : ℝ := final_volume x * final_alcohol_percentage

-- Theorem statement
theorem alcohol_added_approx_3_5 :
  ∃ x : ℝ, x > 0 ∧ x < 4 ∧ 
  initial_alcohol + alcohol_added x = final_alcohol x ∧
  |x - 3.5| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_added_approx_3_5_l165_16575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_focus_l165_16531

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define a point on the left branch of the hyperbola
def left_branch_point (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ P.1 < 0

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the foci of the hyperbola
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem distance_to_left_focus (P : ℝ × ℝ) :
  left_branch_point P →
  distance P F1 + distance P F2 = 8 →
  distance P F1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_focus_l165_16531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_bear_incorrect_l165_16542

/-- Represents the color of a bear -/
inductive BearColor
| White
| Brown
| Black

/-- Represents the row of 1000 bears -/
def BearRow := Fin 1000 → BearColor

/-- The condition that among any three consecutive bears, there is at least one of each color -/
def validBearArrangement (row : BearRow) : Prop :=
  ∀ i : Fin 998, ∃ (c1 c2 c3 : BearColor), 
    ({c1, c2, c3} : Set BearColor) = {BearColor.White, BearColor.Brown, BearColor.Black} ∧
    row i = c1 ∧ row (i + 1) = c2 ∧ row (i + 2) = c3

/-- Iskander's guesses about the bears' colors -/
def iskanderGuesses (row : BearRow) : Prop :=
  row 1 = BearColor.White ∧
  row 19 = BearColor.Brown ∧
  row 399 = BearColor.Black ∧
  row 599 = BearColor.Brown ∧
  row 799 = BearColor.White

/-- The theorem stating that the 20th bear's color must be the incorrect guess -/
theorem twentieth_bear_incorrect (row : BearRow) 
  (h1 : validBearArrangement row) 
  (h2 : iskanderGuesses row) : 
  row 19 ≠ BearColor.Brown :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_bear_incorrect_l165_16542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l165_16581

open Real

theorem trigonometric_inequality (x : ℝ) (n : ℤ) :
  (Real.sin (2 * x) ≠ 0) →
  (8.66 * Real.sin (4 * x) + Real.cos (4 * x) * (Real.cos (2 * x) / Real.sin (2 * x)) > 1) ↔
  (x > n * π / 2 ∧ x < (4 * n + 1) * π / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l165_16581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_log_inequality_l165_16560

noncomputable section

def f (a x : ℝ) : ℝ := Real.log x - (x + a) / x

theorem f_monotonicity (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((x + a) / x^2) x) ∧
  ((a ≥ 0 ∧ ∀ x > 0, (x + a) / x^2 > 0) ∨
   (a < 0 ∧ ∃ x₀ > 0, ∀ x > 0, 
     (x < x₀ → (x + a) / x^2 < 0) ∧ 
     (x > x₀ → (x + a) / x^2 > 0))) :=
sorry

theorem log_inequality : ∀ x > 0, 1 / (x + 1) < Real.log (x + 1) / x ∧ Real.log (x + 1) / x < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_log_inequality_l165_16560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l165_16563

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 1)

-- Define the function g in terms of f and a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Define symmetry about the line y = x + 1
def symmetricAboutXPlusOne (h : ℝ → ℝ) : Prop :=
  ∀ x, h (h x + 1) = x + 1

-- State the theorem
theorem symmetry_condition (a : ℝ) :
  symmetricAboutXPlusOne (g a) ↔ a = 4 := by
  sorry

#check symmetry_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l165_16563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_m_value_l165_16520

-- Define the vectors
noncomputable def a : Fin 2 → ℝ := ![1, Real.sqrt 3]
def b (m : ℝ) : Fin 2 → ℝ := ![3, m]

-- Define the perpendicularity condition
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

-- State the theorem
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, perpendicular a (b m) → m = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_m_value_l165_16520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_segment_ratio_l165_16588

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the necessary functions
def intersect (c1 c2 : Circle) : Set Point := sorry
def secant (p1 p2 : Point) : Line := sorry
def segment_length (p1 p2 : Point) : ℝ := sorry
def distance_point_to_line (p : Point) (l : Line) : ℝ := sorry

-- Define membership instances
instance : Membership Point Circle where
  mem := λ p c => sorry

instance : Membership Point Line where
  mem := λ p l => sorry

-- Main theorem
theorem secant_segment_ratio 
  (c1 c2 : Circle) 
  (A B : Point) 
  (C D E F : Point) :
  A ∈ intersect c1 c2 →
  B ∈ intersect c1 c2 →
  C ∈ c1 →
  D ∈ c2 →
  E ∈ c1 →
  F ∈ c2 →
  A ∈ secant C D →
  A ∈ secant E F →
  segment_length E F / segment_length C D = 
    distance_point_to_line B (secant E F) / distance_point_to_line B (secant C D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_segment_ratio_l165_16588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_triple_exponential_l165_16513

theorem x_range_for_triple_exponential (x : ℝ) : (3 : ℝ)^((3 : ℝ)^x) = 333 → 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_triple_exponential_l165_16513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analects_reasoning_is_deductive_l165_16527

-- Define the structure of the Analects statement
structure AnalectsStatement :=
  (implications : List (Prop → Prop))

-- Define deductive reasoning
def IsDeductiveReasoning (s : AnalectsStatement) : Prop :=
  s.implications.length > 0 ∧
  ∀ i j, i < j → j < s.implications.length →
    ∃ (p q r : Prop), s.implications[i]! = λ x => (p → q) ∧ s.implications[j]! = λ x => (q → r)

-- Define the propositions used in the Analects statement
def names_correct : Prop := sorry
def language_accordance_with_truth : Prop := sorry
def affairs_success : Prop := sorry
def rituals_and_music_flourish : Prop := sorry
def punishments_properly_executed : Prop := sorry
def people_have_nowhere_to_put_hands_and_feet : Prop := sorry

-- Theorem statement
theorem analects_reasoning_is_deductive (s : AnalectsStatement)
  (h : s.implications = [
    λ _ => ¬names_correct → ¬language_accordance_with_truth,
    λ _ => ¬language_accordance_with_truth → ¬affairs_success,
    λ _ => ¬affairs_success → ¬rituals_and_music_flourish,
    λ _ => ¬rituals_and_music_flourish → ¬punishments_properly_executed,
    λ _ => ¬punishments_properly_executed → people_have_nowhere_to_put_hands_and_feet
  ]) :
  IsDeductiveReasoning s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_analects_reasoning_is_deductive_l165_16527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l165_16501

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) / b n = q

/-- Sum of first n terms of a sequence -/
def sum_n (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) s

/-- Main theorem -/
theorem sequence_problem (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 1 = 2 →
  b 1 = 2 →
  sum_n a 5 = 30 →
  b 5 = 32 →
  (∀ n : ℕ, a n = 2 * n) ∧
  ((∀ n : ℕ, b n = 2^n) ∨ (∀ n : ℕ, b n = 2 * (-2)^(n-1))) ∧
  ((∀ n : ℕ, sum_n (λ k ↦ a k + b k) n = n^2 + n - 2 + 2^(n+1)) ∨
   (∀ n : ℕ, sum_n (λ k ↦ a k + b k) n = n^2 + n + 2 * (1 - (-2)^n) / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l165_16501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_southern_car_speed_is_60_l165_16528

/-- The speed of the southern car that satisfies the given conditions -/
def southern_car_speed (v : ℝ) : Prop :=
  let initial_ns_distance : ℝ := 300
  let northern_car_speed : ℝ := 20
  let time : ℝ := 5
  let final_distance : ℝ := 500
  let northern_car_distance : ℝ := northern_car_speed * time
  let southern_car_distance : ℝ := v * time
  let final_ew_distance : ℝ := northern_car_distance + southern_car_distance
  initial_ns_distance ^ 2 + final_ew_distance ^ 2 = final_distance ^ 2

theorem southern_car_speed_is_60 : 
  southern_car_speed 60 := by
  -- Unfold the definition of southern_car_speed
  unfold southern_car_speed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check southern_car_speed_is_60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_southern_car_speed_is_60_l165_16528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_problem_verify_tax_equation_l165_16590

/-- The total value of an imported item given the tax rate, tax threshold, and tax paid. -/
noncomputable def total_value (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) : ℝ :=
  tax_threshold + tax_paid / tax_rate

/-- Theorem stating that given the specific conditions, the total value of the item is $2579.43. -/
theorem import_tax_problem :
  let tax_rate : ℝ := 0.07
  let tax_threshold : ℝ := 1000
  let tax_paid : ℝ := 110.60
  total_value tax_rate tax_threshold tax_paid = 2579.43 := by
  sorry

/-- Verifies that the calculated total value satisfies the original tax equation. -/
theorem verify_tax_equation :
  let tax_rate : ℝ := 0.07
  let tax_threshold : ℝ := 1000
  let tax_paid : ℝ := 110.60
  let total : ℝ := total_value tax_rate tax_threshold tax_paid
  tax_rate * (total - tax_threshold) = tax_paid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_problem_verify_tax_equation_l165_16590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_even_numbers_l165_16519

/-- The number of distinct even numbers that can be formed by reorganizing the digits of 124467 -/
def distinct_even_numbers : ℕ := 240

/-- The original number from which we reorganize digits -/
def original_number : ℕ := 124467

/-- Function that returns the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- Predicate that checks if a number is even -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Function that counts the number of permutations of a list of digits that satisfy a given predicate -/
def number_of_permutations (l : List ℕ) (p : ℕ → Prop) : ℕ := sorry

/-- Theorem stating that the number of distinct even numbers formed by reorganizing the digits of 124467 is 240 -/
theorem count_distinct_even_numbers :
  distinct_even_numbers = 240 ∧ 
  distinct_even_numbers = 
    (number_of_permutations (digits original_number) is_even) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_even_numbers_l165_16519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l165_16502

noncomputable def data : List ℝ := [10, 7, 7, 7, 9]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l165_16502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l165_16543

-- Define the circle
def circle_center : ℝ × ℝ := (6, 6)
def circle_radius : ℝ := 6

-- Define point P
def point_p : ℝ × ℝ := (12, 10)

-- Define a point Q on the circle
noncomputable def point_q (θ : ℝ) : ℝ × ℝ :=
  (circle_center.1 + circle_radius * Real.cos θ,
   circle_center.2 + circle_radius * Real.sin θ)

-- Define the midpoint M of PQ
noncomputable def midpoint_m (θ : ℝ) : ℝ × ℝ :=
  ((point_p.1 + (point_q θ).1) / 2,
   (point_p.2 + (point_q θ).2) / 2)

-- Theorem statement
theorem midpoint_locus :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (9, 8) ∧
    radius = 3 ∧
    ∀ θ, ((midpoint_m θ).1 - center.1)^2 + ((midpoint_m θ).2 - center.2)^2 = radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l165_16543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_19_factors_four_six_zero_eight_has_19_factors_four_six_zero_eight_is_smallest_l165_16508

def count_factors (n : ℕ) : ℕ := (Finset.filter (λ i ↦ n % i = 0) (Finset.range (n + 1))).card

theorem smallest_number_with_19_factors : 
  ∀ n : ℕ, n > 0 → count_factors n = 19 → n ≥ 3 → n ≥ 4608 :=
by
  sorry

theorem four_six_zero_eight_has_19_factors : 
  count_factors 4608 = 19 :=
by
  sorry

theorem four_six_zero_eight_is_smallest : 
  ∀ n : ℕ, n > 0 → count_factors n = 19 → n ≥ 3 → n ≥ 4608 ∧ 
  (n = 4608 ∨ count_factors n ≠ 19 ∨ n < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_19_factors_four_six_zero_eight_has_19_factors_four_six_zero_eight_is_smallest_l165_16508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_theorem_l165_16591

-- Define a structure for participants
structure Participant where
  id : ℕ
  solved_problems : Finset ℕ

theorem math_competition_theorem 
  (n : ℕ) 
  (first_part second_part : Finset ℕ) 
  (participants : Finset Participant) :
  n = 28 →
  first_part ∪ second_part = Finset.range n →
  first_part ∩ second_part = ∅ →
  (∀ p : ℕ, p ∈ Finset.range n → ∃! (solvers : Finset Participant), solvers.card = 2 ∧ ∀ s ∈ solvers, p ∈ s.solved_problems) →
  (∀ participant ∈ participants, participant.solved_problems.card = 7) →
  ∃ participant ∈ participants, 
    (participant.solved_problems ∩ first_part).card = 0 ∨ 
    (participant.solved_problems ∩ first_part).card ≥ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_theorem_l165_16591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grill_runtime_theorem_l165_16551

/-- Represents the burning rate of coals in a grill -/
structure BurningRate where
  coals : ℕ
  minutes : ℕ

/-- Represents a bag of coals -/
structure CoalBag where
  coals : ℕ

noncomputable def grill_runtime (rate : BurningRate) (bags : ℕ) (bag_size : ℕ) : ℝ :=
  (bags * bag_size : ℝ) / rate.coals * rate.minutes / 60

theorem grill_runtime_theorem (rate : BurningRate) (bags : ℕ) (bag_size : ℕ) :
  rate.coals = 15 → rate.minutes = 20 → bags = 3 → bag_size = 60 →
  grill_runtime rate bags bag_size = 4 := by
  sorry

#check grill_runtime_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grill_runtime_theorem_l165_16551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_numbers_theorem_l165_16511

/-- A move replaces two numbers with their sum and non-negative difference -/
def is_valid_move (a b c d : ℕ) : Prop :=
  (c = a + b ∧ d = a.sub b) ∨ (d = a + b ∧ c = a.sub b)

/-- A sequence of moves is valid if each move is valid -/
def valid_move_sequence (initial final : List ℕ) : Prop :=
  ∃ (moves : List (List ℕ)), 
    moves.head? = some initial ∧
    moves.getLast? = some final ∧
    ∀ i < moves.length - 1, 
      ∃ a b c d, is_valid_move a b c d ∧
        moves[i]?.isSome ∧ moves[i+1]?.isSome ∧
        ∃ l1 l2 : List ℕ, 
          moves[i]? = some (l1 ++ [a, b] ++ l2) ∧
          moves[i+1]? = some (l1 ++ [c, d] ++ l2)

/-- The theorem states that for n ≥ 3, the only possible values of k
    such that all numbers from 1 to n can be made equal to k through
    a series of valid moves are powers of 2 greater than or equal to n -/
theorem blackboard_numbers_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ k : ℕ, 
    (∃ final : List ℕ, 
      valid_move_sequence (List.range n) final ∧ 
      ∀ x ∈ final, x = k) ↔ 
    (∃ m : ℕ, k = 2^m ∧ k ≥ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_numbers_theorem_l165_16511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_listening_time_approx_57_l165_16573

/-- Represents the distribution of audience listening times for a 90-minute talk --/
structure AudienceDistribution where
  total_duration : ℚ
  full_listeners_percent : ℚ
  non_listeners_percent : ℚ
  half_listeners_percent : ℚ
  two_thirds_listeners_percent : ℚ

/-- Calculates the average listening time for the audience --/
noncomputable def average_listening_time (dist : AudienceDistribution) : ℚ :=
  let remaining_percent := 1 - dist.full_listeners_percent - dist.non_listeners_percent
  let full_time := dist.full_listeners_percent * dist.total_duration
  let half_time := remaining_percent * dist.half_listeners_percent * (dist.total_duration / 2)
  let two_thirds_time := remaining_percent * (1 - dist.half_listeners_percent) * (2 * dist.total_duration / 3)
  full_time + half_time + two_thirds_time

/-- Theorem stating that the average listening time is approximately 57 minutes --/
theorem average_listening_time_approx_57 (dist : AudienceDistribution) 
    (h1 : dist.total_duration = 90)
    (h2 : dist.full_listeners_percent = 3/10)
    (h3 : dist.non_listeners_percent = 15/100)
    (h4 : dist.half_listeners_percent = 2/5)
    (h5 : dist.two_thirds_listeners_percent = 1 - dist.half_listeners_percent) :
  ∃ ε : ℚ, ε > 0 ∧ |average_listening_time dist - 57| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_listening_time_approx_57_l165_16573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l165_16578

/-- Given a rational function r(x)/s(x) with specific properties, 
    prove that r(x) + s(x) has a specific form. -/
theorem rational_function_sum (r s : ℝ → ℝ) : 
  (∀ x, x ≠ 2 → r x / s x ≠ 0) →  -- horizontal asymptote at y=0
  (∃ ε > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < ε → r x / s x = 0) →  -- hole at x=2
  (∃ a b c : ℝ, ∀ x, s x = a*x^2 + b*x + c) →  -- s(x) is quadratic
  r 1 = 2 →  -- r(1) = 2
  s (-1) = -1 →  -- s(-1) = -1
  ∀ x, r x + s x = -1/3*x^2 - 2*x + 16/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l165_16578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l165_16530

-- Define the proposition for part 1
def P (a b : ℝ) : Prop := a > b → (2 : ℝ)^a > (2 : ℝ)^b

-- Define evenness
def even (n : ℤ) : Prop := ∃ k, n = 2 * k

-- Define the proposition for part 2
def Q (a b : ℤ) : Prop := even a ∧ even b → even (a + b)

-- Define sufficient and necessary conditions
def sufficient (p q : Prop) : Prop := p → q
def necessary (p q : Prop) : Prop := q → p

-- Define the equation for part 4
def has_unique_solution (a : ℝ) : Prop := ∃! x, a * x^2 + x + a = 0

theorem problem_statement :
  (∀ a b : ℝ, ¬(P a b)) ∧
  (∀ a b : ℤ, ¬(even (a + b)) → ¬(even a ∧ even b)) ∧
  (∀ p q : Prop, sufficient p q ∧ ¬(necessary p q) → necessary (¬p) (¬q) ∧ ¬(sufficient (¬p) (¬q))) ∧
  (∀ a : ℝ, has_unique_solution a ↔ (a = 1/2 ∨ a = -1/2 ∨ a = 0)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l165_16530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melanie_final_plums_l165_16532

def initial_plums : ℕ := 12
def eaten_plums : ℕ := 2
def fraction_given : ℚ := 1/3

def remaining_plums : ℕ := initial_plums - eaten_plums

noncomputable def plums_given_away : ℕ := Int.toNat ((remaining_plums : ℚ) * fraction_given).floor

theorem melanie_final_plums :
  remaining_plums - plums_given_away = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melanie_final_plums_l165_16532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_coefficient_l165_16538

/-- A quadratic trinomial with coefficient -1 for the quadratic term -/
def quadratic_trinomial (x : ℝ) : ℝ := -x^2 + 3*x - 1

/-- Theorem stating that the quadratic trinomial has coefficient -1 for the quadratic term -/
theorem quadratic_trinomial_coefficient :
  ∃ (a b : ℝ), quadratic_trinomial = λ x ↦ -x^2 + a*x + b := by
  use 3, -1
  ext x
  simp [quadratic_trinomial]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_coefficient_l165_16538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l165_16526

theorem simplify_expression : 4 - 3 - (-7) + (-2) = 4 - 3 + 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l165_16526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_acute_angle_between_lines_l165_16535

/-- The cosine of the acute angle between two lines with given direction vectors -/
theorem cosine_acute_angle_between_lines (v1 v2 : ℝ × ℝ) : 
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  v1 = (-2, 5) → v2 = (6, -1) → 
  (abs dot_product) / (magnitude1 * magnitude2) = 17 / Real.sqrt (29 * 37) := by
  sorry

#check cosine_acute_angle_between_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_acute_angle_between_lines_l165_16535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speeding_percentage_calculation_l165_16589

/-- The percentage of motorists who receive speeding tickets -/
noncomputable def ticket_percentage : ℝ := 10

/-- The percentage of speeding motorists who do not receive tickets -/
noncomputable def no_ticket_percentage : ℝ := 20

/-- The total percentage of motorists exceeding the speed limit -/
noncomputable def speeding_percentage : ℝ := ticket_percentage / (1 - no_ticket_percentage / 100)

theorem speeding_percentage_calculation :
  speeding_percentage = 12.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speeding_percentage_calculation_l165_16589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l165_16599

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 2)

theorem f_properties :
  /- Part 1: Minimum value of f(x) for x > 2 is 8 -/
  (∀ x > 2, f x ≥ 8) ∧ (∃ x > 2, f x = 8) ∧
  /- Part 2: Solutions to the inequality depending on k -/
  (∀ k : ℝ,
    (k = 0 → ∀ x, f x < (k*(x-1) + 1 - x^2)/(2-x) ↔ x < 2) ∧
    (-1 < k ∧ k < 0 → ∀ x, f x < (k*(x-1) + 1 - x^2)/(2-x) ↔ 1 - 1/k < x ∧ x < 2) ∧
    (k > 0 → ∀ x, f x < (k*(x-1) + 1 - x^2)/(2-x) ↔ 1 - 1/k < x ∧ x < 2) ∧
    (k < -1 → ∀ x, f x < (k*(x-1) + 1 - x^2)/(2-x) ↔ (x < 1 - 1/k ∨ x > 2)) ∧
    (k = -1 → ∀ x, f x < (k*(x-1) + 1 - x^2)/(2-x) ↔ x ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l165_16599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l165_16540

theorem right_triangle_area (A B C : ℝ × ℝ) 
  (h_right : A = (0, 0) ∧ B = (0, (C.2)) ∧ C.1 > 0)
  (h_angle : (C.2 / C.1) = 1) 
  (h_hypotenuse : Real.sqrt (C.1^2 + C.2^2) = 20) : 
  (1/2 : ℝ) * C.1 * C.2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l165_16540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sums_l165_16514

/-- The sequence aₙ -/
def a (n : ℕ+) : ℚ := n / (2^(n.val+1))

/-- The partial sum Sₙ of the sequence aₙ -/
def S (n : ℕ+) : ℚ := -a n + 1 - 1 / (2^n.val)

/-- The sum of the first n terms of the sequence Sₙ -/
def T (n : ℕ+) : ℚ := n.val - 2 + (n.val + 4) / (2^(n.val+1))

/-- Theorem stating the relationship between Sₙ and aₙ, and the formulas for aₙ and Tₙ -/
theorem sequence_sums (n : ℕ+) : 
  S n = -a n + 1 - 1 / (2^n.val) ∧ 
  a n = n / (2^(n.val+1)) ∧ 
  T n = n.val - 2 + (n.val + 4) / (2^(n.val+1)) := by
  sorry

#check sequence_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sums_l165_16514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_only_one_proposition_true_l165_16596

-- Define the types for planes and lines
variable (α β γ : Set (Fin 3 → ℝ))
variable (m n l : Set (Fin 3 → ℝ))

-- Define the necessary relations
def is_plane (p : Set (Fin 3 → ℝ)) : Prop := sorry
def is_line (l : Set (Fin 3 → ℝ)) : Prop := sorry
def perpendicular (a b : Set (Fin 3 → ℝ)) : Prop := sorry
def parallel (a b : Set (Fin 3 → ℝ)) : Prop := sorry
def lies_in (l p : Set (Fin 3 → ℝ)) : Prop := sorry

-- State the theorem
theorem perpendicular_planes 
  (h_α_plane : is_plane α) 
  (h_β_plane : is_plane β) 
  (h_γ_plane : is_plane γ) 
  (h_m_line : is_line m) 
  (h_n_line : is_line n) 
  (h_l_line : is_line l) 
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α) 
  (h_non_coincident : m ≠ n ∧ n ≠ l ∧ l ≠ m) 
  (h_m_perp_α : perpendicular m α) 
  (h_m_parallel_n : parallel m n) 
  (h_n_in_β : lies_in n β) : 
  perpendicular α β :=
by sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ m n l, is_line m → is_line n → is_line l →
  perpendicular m l → perpendicular n l → parallel m n

def proposition_2 : Prop :=
  ∀ α β γ, is_plane α → is_plane β → is_plane γ →
  perpendicular α γ → perpendicular β γ → parallel α β

def proposition_3 : Prop :=
  ∀ m n α β, is_line m → is_line n → is_plane α → is_plane β →
  perpendicular m α → parallel m n → lies_in n β → perpendicular α β

def proposition_4 : Prop :=
  ∀ m n α β, is_line m → is_line n → is_plane α → is_plane β →
  parallel m α → (∃ x, x ∈ α ∧ x ∈ β ∧ x ∈ n) → parallel m n

-- Theorem stating that only one proposition is true
theorem only_one_proposition_true :
  (proposition_1 ↔ False) ∧
  (proposition_2 ↔ False) ∧
  (proposition_3 ↔ True) ∧
  (proposition_4 ↔ False) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_only_one_proposition_true_l165_16596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_50_div_x_minus_1_cubed_l165_16592

theorem remainder_x_50_div_x_minus_1_cubed (x : ℝ) : 
  ∃ q : Polynomial ℝ, X^50 = (X - 1)^3 * q + (1225*X^2 - 2400*X + 1176) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_50_div_x_minus_1_cubed_l165_16592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_l165_16546

/-- Sequence of numbers following the pattern 1, 19, 199, 1999, etc. -/
def sequenceNum : ℕ → ℕ
  | 0 => 1
  | n + 1 => 10 * sequenceNum n + 9

/-- Predicate to check if a number has all digits as twos except for one digit -/
def allTwoExceptOne (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ d ≠ 2 ∧ ∃ (m : ℕ), n = (Finset.range m).sum (fun i => 2 * 10^i) + d * 10^m

theorem sequence_sum_property :
  ∃ (S : Finset ℕ) (k : ℕ),
    S.card ≥ 3 ∧
    (∀ i ∈ S, ∃ n, sequenceNum n = i) ∧
    allTwoExceptOne (S.sum id) ∧
    (k = 0 ∨ k = 1) ∧
    ∃ (m : ℕ), S.sum id = (Finset.range m).sum (fun i => 2 * 10^i) + k * 10^m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_l165_16546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_x_intersection_is_six_l165_16559

-- Define the circle
noncomputable def circle_center : ℝ × ℝ := ((2 + 10) / 2, (2 + 8) / 2)
noncomputable def circle_radius : ℝ := Real.sqrt ((2 - 10)^2 + (2 - 8)^2) / 2

-- Define the function for the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

-- Theorem statement
theorem second_x_intersection_is_six :
  ∃ (x : ℝ), x ≠ 2 ∧ circle_equation x 0 ∧ x = 6 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_x_intersection_is_six_l165_16559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cube_volume_l165_16595

-- Define the total surface area of the cube
noncomputable def cube_surface_area : ℝ := 6

-- Define the volume of a sphere given its radius
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4/3) * Real.pi * (radius ^ 3)

-- Theorem statement
theorem sphere_in_cube_volume :
  ∃ (sphere_radius : ℝ),
    sphere_radius > 0 ∧
    cube_surface_area = 6 * (2 * sphere_radius) ^ 2 ∧
    sphere_volume sphere_radius = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cube_volume_l165_16595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tape_pieces_count_l165_16554

/-- Calculates the number of tape pieces given the individual tape length, overlap, and total connected length -/
noncomputable def calculate_tape_pieces (tape_length : ℝ) (overlap : ℝ) (total_length : ℝ) : ℕ :=
  Int.toNat ⌈(total_length - tape_length) / (tape_length - overlap) + 1⌉

/-- Theorem stating that given the specific measurements, the number of tape pieces is 34 -/
theorem tape_pieces_count :
  calculate_tape_pieces 8.8 0.5 282.7 = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tape_pieces_count_l165_16554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_inequality_l165_16558

/-- Represents a convex polygon with n vertices -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- The length of the side between two vertices -/
noncomputable def sideLength (p : ConvexPolygon n) (i j : Fin n) : ℝ :=
  let (x₁, y₁) := p.vertices i
  let (x₂, y₂) := p.vertices j
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The angle at a vertex -/
noncomputable def angle (p : ConvexPolygon n) (i : Fin n) : ℝ :=
  sorry -- Definition of angle calculation

/-- Statement of the theorem -/
theorem polygon_inequality {n : ℕ} (A B : ConvexPolygon (n + 3)) :
  (∀ i j : Fin (n + 3), i ≠ j → (i ≠ 0 ∨ j ≠ (n + 2)) → sideLength A i j = sideLength B i j) →
  (∀ i : Fin (n + 3), i ≠ 0 → i ≠ (n + 2) → angle A i ≥ angle B i) →
  (∃ i : Fin (n + 3), i ≠ 0 ∧ i ≠ (n + 2) ∧ angle A i > angle B i) →
  sideLength A 0 (n + 2) > sideLength B 0 (n + 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_inequality_l165_16558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_squares_l165_16533

/-- The set of points with integer coordinates (x, y) where |x| ≤ 5 and |y| ≤ 5 -/
def H : Set (ℤ × ℤ) :=
  {p | |p.1| ≤ 5 ∧ |p.2| ≤ 5}

/-- A square with side length 4 and vertices in H -/
def ValidSquare (v1 v2 v3 v4 : ℤ × ℤ) : Prop :=
  v1 ∈ H ∧ v2 ∈ H ∧ v3 ∈ H ∧ v4 ∈ H ∧
  (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = 16 ∧
  (v2.1 - v3.1)^2 + (v2.2 - v3.2)^2 = 16 ∧
  (v3.1 - v4.1)^2 + (v3.2 - v4.2)^2 = 16 ∧
  (v4.1 - v1.1)^2 + (v4.2 - v1.2)^2 = 16

theorem count_valid_squares : 
  ∃! (squares : Finset ((ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ))), 
    squares.card = 4 ∧ 
    ∀ s ∈ squares, ValidSquare s.1 s.2.1 s.2.2.1 s.2.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_squares_l165_16533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_l165_16536

theorem derivative_reciprocal (x : ℝ) (h : x ≠ 0) : 
  deriv (λ y => 1 / y) x = -1 / x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_l165_16536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_problem_l165_16580

/-- The value of x in the problem -/
noncomputable def x : ℝ := 1649.999999999999

/-- The percentage we're looking for -/
noncomputable def P : ℝ := (1/3 * x + 110) / x * 100

theorem percentage_problem :
  (P/100) * x = (1/3) * x + 110 ∧ 
  abs (P - 40) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_problem_l165_16580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_result_l165_16598

/-- A complex number transformation that applies a rotation and dilation -/
noncomputable def transform (z : ℂ) : ℂ :=
  2 * Complex.exp (Complex.I * Real.pi / 3) * z

theorem transform_result : 
  transform (-3 - 8 * Complex.I) = (8 * Real.sqrt 3 - 3) - (3 * Real.sqrt 3 + 8) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_result_l165_16598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_odd_multiple_of_three_on_die_l165_16509

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem one_odd_multiple_of_three_on_die :
  (standard_die.filter (λ n => n % 2 = 1 ∧ n % 3 = 0)).card = 1 := by
  simp [standard_die]
  norm_num
  rfl

#eval (standard_die.filter (λ n => n % 2 = 1 ∧ n % 3 = 0)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_odd_multiple_of_three_on_die_l165_16509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l165_16505

/-- Geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 1 else a 1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (geometric_sum a q 3 = 3 * a 3) →
  (q = 1 ∨ q = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l165_16505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l165_16537

/-- Represents a point on an ellipse -/
structure EllipsePoint (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / a^2 + y^2 / b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The slope product condition for a point on the ellipse -/
def slope_product_condition (a b : ℝ) (p : EllipsePoint a b) : Prop :=
  (p.y / (p.x + a)) * (p.y / (p.x - a)) = -1/4

theorem ellipse_eccentricity_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ p : EllipsePoint a b, slope_product_condition a b p) →
  eccentricity a b = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l165_16537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_difference_is_seven_l165_16522

/-- The number of steps Ginny takes between consecutive markers -/
def ginny_steps : ℕ := 70

/-- The number of jumps Lenny takes between consecutive markers -/
def lenny_jumps : ℕ := 22

/-- The total number of markers on the trail -/
def total_markers : ℕ := 51

/-- The distance in feet between the first and last marker -/
def total_distance : ℕ := 10560

/-- Ginny's step length in feet -/
def ginny_step_length : ℚ := total_distance / (ginny_steps * (total_markers - 1))

/-- Lenny's jump length in feet -/
def lenny_jump_length : ℚ := total_distance / (lenny_jumps * (total_markers - 1))

/-- The difference between Lenny's jump length and Ginny's step length -/
def length_difference : ℚ := lenny_jump_length - ginny_step_length

theorem length_difference_is_seven : 
  Int.floor length_difference = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_difference_is_seven_l165_16522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_inequality_l165_16584

/-- Volumes of geometric shapes with radius r -/
noncomputable def cone_volume (r : ℝ) : ℝ := (1/3) * Real.pi * r^3

noncomputable def cylinder_volume (r : ℝ) : ℝ := 2 * Real.pi * r^3

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- Theorem stating that the volume of a hemisphere plus twice the volume of a cone
    is not equal to the volume of a cylinder, all with the same radius r -/
theorem volume_inequality (r : ℝ) (hr : r > 0) :
  hemisphere_volume r + 2 * cone_volume r ≠ cylinder_volume r := by
  -- Expand definitions
  unfold hemisphere_volume cone_volume cylinder_volume
  -- Simplify the expressions
  simp [Real.pi]
  -- The proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_inequality_l165_16584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_polynomial_solutions_l165_16504

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the polynomial g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_polynomial_solutions :
  (∀ x, f (g x) = 9 * x^2 + 12 * x + 4) →
  (∀ x, g x = 3 * x + 2 ∨ g x = -3 * x - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_polynomial_solutions_l165_16504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l165_16564

-- Define the function f(x)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2^x - a else b - x

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a b 6 = 0) →  -- 6 is a zero of the function
  (∃ x y : ℝ, f a b x = 0 ∧ f a b y = 0 ∧ x * y = 0) →  -- product of zeros is 0
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = 5/2 ∧ f a b x₂ = 5/2) →  -- equation f(x) = 5/2 has two roots
  f a b 4 ≠ 0  -- 4 is not a zero of the function
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l165_16564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_ratio_l165_16553

theorem field_area_ratio (total_area smaller_area : ℝ) 
  (h1 : total_area = 900)
  (h2 : smaller_area = 405)
  (h3 : smaller_area ≤ total_area / 2) : 
  (total_area - smaller_area - smaller_area) / ((total_area - smaller_area + smaller_area) / 2) = 1 / 5 := by
  sorry

#check field_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_ratio_l165_16553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l165_16510

/-- Rectangle EFGH with given dimensions and circles -/
structure Rectangle :=
  (ef : ℝ)
  (fg : ℝ)
  (radius_e : ℝ)
  (radius_f : ℝ)
  (radius_g : ℝ)

/-- The area inside the rectangle but outside the quarter circles -/
noncomputable def area_outside_circles (r : Rectangle) : ℝ :=
  r.ef * r.fg - (Real.pi / 4) * (r.radius_e^2 + r.radius_f^2 + r.radius_g^2)

/-- The theorem stating the approximate area -/
theorem area_approximation (r : Rectangle) 
  (h1 : r.ef = 4)
  (h2 : r.fg = 6)
  (h3 : r.radius_e = 2)
  (h4 : r.radius_f = Real.sqrt 2)
  (h5 : r.radius_g = 3) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |area_outside_circles r - 12.2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l165_16510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_surface_area_theorem_l165_16541

/-- Represents a truncated cone circumscribed around a sphere with its bases being great circles of two other spheres -/
structure TruncatedCone where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  h : ℝ  -- height of the truncated cone

/-- The sum of the surface areas of the three spheres -/
noncomputable def spheres_surface_area (tc : TruncatedCone) : ℝ :=
  4 * Real.pi * (tc.R^2 + tc.R * tc.r + tc.r^2)

/-- The surface area of the truncated cone -/
noncomputable def truncated_cone_surface_area (tc : TruncatedCone) : ℝ :=
  Real.pi * (tc.R + tc.r) * (tc.R - tc.r)^2 / (tc.R - tc.r) + 
  2 * Real.pi * (tc.R * tc.R + tc.r * tc.r)

/-- Theorem stating that the surface area of the truncated cone is half the sum of the surface areas of the three spheres -/
theorem truncated_cone_surface_area_theorem (tc : TruncatedCone) : 
  truncated_cone_surface_area tc = (spheres_surface_area tc) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_surface_area_theorem_l165_16541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximately_30_kmh_l165_16516

-- Define constants
noncomputable def train_length : ℝ := 400
noncomputable def crossing_time : ℝ := 59.99520038396929
noncomputable def man_speed_kmh : ℝ := 6

-- Define conversion factors
noncomputable def km_to_m : ℝ := 1000
noncomputable def hr_to_s : ℝ := 3600

-- Define functions
noncomputable def km_per_hr_to_m_per_s (speed_kmh : ℝ) : ℝ :=
  speed_kmh * km_to_m / hr_to_s

noncomputable def m_per_s_to_km_per_hr (speed_ms : ℝ) : ℝ :=
  speed_ms * hr_to_s / km_to_m

noncomputable def relative_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

-- Theorem statement
theorem train_speed_approximately_30_kmh :
  let man_speed_ms := km_per_hr_to_m_per_s man_speed_kmh
  let rel_speed := relative_speed train_length crossing_time
  let train_speed_ms := rel_speed + man_speed_ms
  let train_speed_kmh := m_per_s_to_km_per_hr train_speed_ms
  abs (train_speed_kmh - 30) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximately_30_kmh_l165_16516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l165_16557

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + 2*x

/-- Part 1 of the theorem -/
theorem part_one : ∀ x : ℝ, x > 1 → f 1 x > x - 1 := by
  sorry

/-- Part 2 of the theorem -/
theorem part_two : ∀ a : ℝ, (∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f a y > f a x)) → a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l165_16557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l165_16539

noncomputable def f (x y : ℝ) : ℝ := (x - y)^2 + (4 + Real.sqrt (1 - x^2) + Real.sqrt (1 - y^2 / 9))^2

theorem max_value_of_f :
  ∃ (max : ℝ), max = 28 + 6 * Real.sqrt 3 ∧
  ∀ (x y : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ -3 ≤ y ∧ y ≤ 3 →
  f x y ≤ max := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l165_16539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l165_16568

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the locus of points
def locus : Set (ℝ × ℝ) :=
  {p | distance p F₁ + distance p F₂ = 10}

-- Theorem stating the equation of the ellipse
theorem locus_is_ellipse :
  locus = {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l165_16568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_slope_relation_l165_16500

/-- An ellipse C in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (C : Ellipse) : ℝ := Real.sqrt (1 - C.b^2 / C.a^2)

/-- The length of the line segment cut by the ellipse on y = x -/
noncomputable def segment_length (C : Ellipse) : ℝ := 2 * C.a * Real.sqrt (2 / (C.a^2 + C.b^2))

/-- Point on the ellipse -/
structure Point (C : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / C.a^2 + y^2 / C.b^2 = 1

theorem ellipse_equation_and_slope_relation (C : Ellipse)
  (h_ecc : eccentricity C = Real.sqrt 3 / 2)
  (h_seg : segment_length C = 4 * Real.sqrt 10 / 5) :
  (∃ (A B D : Point C),
    A.x ≠ 0 ∧ A.y ≠ 0 ∧
    B.x = -A.x ∧ B.y = -A.y ∧
    (D.y - A.y) * (B.x - A.x) = -(D.x - A.x) * (B.y - A.y) →
    C.a = 2 ∧ C.b = 1 ∧
    (let k₁ := (D.y + A.y) / (D.x + A.x);
     let k₂ := -A.y / (3 * A.x);
     k₁ = -1/2 * k₂)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_slope_relation_l165_16500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_covered_l165_16545

/-- Represents a 6x6 grid --/
def Grid := Fin 6 → Fin 6 → ℕ

/-- A rectangle covers exactly 2 cells --/
structure Rectangle where
  cell1 : Fin 6 × Fin 6
  cell2 : Fin 6 × Fin 6
  covers_two_cells : cell1 ≠ cell2

/-- No two rectangles share a common edge --/
def no_shared_edges (r1 r2 : Rectangle) : Prop :=
  (r1.cell1.1 ≠ r2.cell1.1 ∨ r1.cell1.2 ≠ r2.cell1.2) ∧
  (r1.cell1.1 ≠ r2.cell2.1 ∨ r1.cell1.2 ≠ r2.cell2.2) ∧
  (r1.cell2.1 ≠ r2.cell1.1 ∨ r1.cell2.2 ≠ r2.cell1.2) ∧
  (r1.cell2.1 ≠ r2.cell2.1 ∨ r1.cell2.2 ≠ r2.cell2.2)

/-- A valid placement of rectangles on the grid --/
def ValidPlacement (placement : List Rectangle) : Prop :=
  ∀ r1 r2, r1 ∈ placement → r2 ∈ placement → r1 ≠ r2 → no_shared_edges r1 r2

/-- The sum of numbers covered by a placement of rectangles --/
def sum_covered (grid : Grid) (placement : List Rectangle) : ℕ :=
  placement.foldl (λ sum r => sum + grid r.cell1.1 r.cell1.2 + grid r.cell2.1 r.cell2.2) 0

/-- The main theorem: maximum sum of covered numbers is 342 --/
theorem max_sum_covered (grid : Grid) :
  ∃ placement : List Rectangle,
    ValidPlacement placement ∧
    (∀ other : List Rectangle, ValidPlacement other →
      sum_covered grid other ≤ sum_covered grid placement) ∧
    sum_covered grid placement = 342 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_covered_l165_16545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_in_combined_vessel_l165_16555

-- Define the vessels and their volume ratios
def vessel1_volume : ℚ := 3
def vessel2_volume : ℚ := 5

-- Define the milk to water ratios in each vessel
def vessel1_milk_ratio : ℚ := 1
def vessel1_water_ratio : ℚ := 2
def vessel2_milk_ratio : ℚ := 6
def vessel2_water_ratio : ℚ := 4

-- Theorem statement
theorem milk_water_ratio_in_combined_vessel :
  (vessel1_volume * vessel1_milk_ratio / (vessel1_milk_ratio + vessel1_water_ratio) +
   vessel2_volume * vessel2_milk_ratio / (vessel2_milk_ratio + vessel2_water_ratio)) =
  (vessel1_volume * vessel1_water_ratio / (vessel1_milk_ratio + vessel1_water_ratio) +
   vessel2_volume * vessel2_water_ratio / (vessel2_milk_ratio + vessel2_water_ratio)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_in_combined_vessel_l165_16555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_theorem_l165_16503

/-- Given a quadrilateral with sides a, b, c, d and diagonals e, f, where e starts from the common endpoint of a and d,
    prove that e^4 - f^4 = ((a+c)/(a-c)) * [d^2 * (2ac + d^2) - b^2 * (2ac + b^2)]
    under the conditions that a > c, b ≥ d, and a is parallel to c. -/
theorem quadrilateral_diagonal_theorem
  (a b c d e f : ℝ)
  (h1 : a > c)
  (h2 : b ≥ d)
  (h3 : ∃ k : ℝ, a = k * c) :  -- Representing parallelism as a scalar multiple
  e^4 - f^4 = ((a+c)/(a-c)) * (d^2 * (2*a*c + d^2) - b^2 * (2*a*c + b^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_theorem_l165_16503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_correct_l165_16521

noncomputable section

-- Define the original curve
def original_curve (x : ℝ) : ℝ := 2 * Real.sin (3 * x)

-- Define the transformed curve
def transformed_curve (x' : ℝ) : ℝ := Real.sin x'

-- Define the scaling transformation
def scaling_x (x : ℝ) : ℝ := x / 3
def scaling_y (y : ℝ) : ℝ := y / 2

theorem scaling_transformation_correct :
  ∀ x : ℝ, transformed_curve (scaling_x x) = scaling_y (original_curve x) := by
  intro x
  simp [transformed_curve, scaling_x, scaling_y, original_curve]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_correct_l165_16521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_trip_day3_miles_l165_16576

/-- Represents Carrie's four-day trip --/
structure CarrieTrip where
  day1_miles : ℕ
  day2_miles : ℕ
  day3_miles : ℕ
  day4_miles : ℕ
  charge_interval : ℕ
  total_charges : ℕ

/-- Theorem stating the conditions of Carrie's trip and the miles driven on the third day --/
theorem carrie_trip_day3_miles (trip : CarrieTrip) : 
  trip.day1_miles = 135 ∧ 
  trip.day2_miles = trip.day1_miles + 124 ∧
  trip.day4_miles = 189 ∧
  trip.charge_interval = 106 ∧
  trip.total_charges = 7 →
  trip.day3_miles = 159 := by
  intro h
  -- The proof steps would go here
  sorry

#check carrie_trip_day3_miles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_trip_day3_miles_l165_16576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_half_sum_min_l165_16534

theorem sin_cos_half_sum_min : 
  ∃ (B : ℝ), (∀ A : ℝ, Real.sin (A / 2) + Real.cos (A / 2) ≥ -2) ∧ 
  Real.sin (B / 2) + Real.cos (B / 2) = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_half_sum_min_l165_16534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_negative_fraction_power_l165_16517

theorem simplify_negative_fraction_power : (-1/27 : ℝ)^(-3/4 : ℝ) = 3^(3/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_negative_fraction_power_l165_16517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l165_16566

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  4 / (3 * x^(-3 : ℤ)) * (3 * x^2) / 2 * x^(-1 : ℤ) / 5 = 2 * x^4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l165_16566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_sea_turtles_l165_16529

theorem baby_sea_turtles (total : ℕ) (swept_fraction : ℚ) (remaining : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  remaining = total - (swept_fraction * ↑total).floor → 
  remaining = 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_sea_turtles_l165_16529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_equal_tangents_l165_16561

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 8*y + 31 = 0

-- Define the length of a tangent from a point to a circle
noncomputable def tangent_length (px py cx cy r : ℝ) : ℝ :=
  Real.sqrt ((px - cx)^2 + (py - cy)^2 - r^2)

-- Theorem statement
theorem trajectory_of_equal_tangents (x y : ℝ) :
  tangent_length x y 0 0 1 = tangent_length x y 4 4 1 →
  x + y - 4 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_equal_tangents_l165_16561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sin_plus_3cos_l165_16570

theorem range_of_sin_plus_3cos : 
  ∀ y : ℝ, (∃ x : ℝ, Real.sin x + 3 * Real.cos x = y) ↔ -Real.sqrt 10 ≤ y ∧ y ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sin_plus_3cos_l165_16570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_of_f_l165_16552

-- Define the function f
def f (S : Set ℝ) : ℝ → ℝ := sorry

-- Define the property of the function
def has_property (S : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ S, (1 / x) ∈ S ∧ f x + f (1 / x) = x

-- Define the largest set that satisfies the property
def largest_set : Set ℝ := {-1, 1}

-- Theorem statement
theorem largest_domain_of_f :
  ∀ S : Set ℝ, has_property S (f S) → S ⊆ largest_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_of_f_l165_16552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_count_l165_16562

def floor_square_div (n : ℕ) : ℕ := 
  Int.toNat ⌊(n^2 : ℚ) / 500⌋

def distinct_numbers (n : ℕ) : Finset ℕ :=
  Finset.image floor_square_div (Finset.range (n + 1))

theorem distinct_numbers_count :
  Finset.card (distinct_numbers 1000) = 876 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_count_l165_16562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_elastic_plate_vibration_solution_l165_16524

/-- Given an integral equation for thin elastic plate vibrations -/
theorem thin_elastic_plate_vibration_solution 
  (b : ℝ) (ψ : ℝ → ℝ) (f : ℝ → ℝ) :
  (∀ t > 0, ψ t = (1 / (2 * b * t)) * ∫ x in Set.Ici 0, x * f x * Real.sin (x^2 / (4 * b * t))) →
  (∀ x, f x = (2 / Real.pi) * ∫ t in Set.Ici 0, (ψ t / t) * Real.sin (x^2 / (4 * b * t))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_elastic_plate_vibration_solution_l165_16524
