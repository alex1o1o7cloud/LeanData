import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_612_55_l1312_131249

/-- Represents the fleet composition and fuel costs for different vehicle types --/
structure FleetData where
  serviceCost : ℝ
  minivanCount : ℕ
  truckCount : ℕ
  sedanCount : ℕ
  motorcycleCount : ℕ
  minivanFuelCost : ℝ
  truckFuelCost : ℝ
  sedanFuelCost : ℝ
  motorcycleFuelCost : ℝ
  minivanTankCapacity : ℝ
  sedanTankCapacity : ℝ
  motorcycleTankCapacity : ℝ
  truckTankMultiplier : ℝ

/-- Calculates the total cost for filling up and servicing all vehicles --/
def totalCost (data : FleetData) : ℝ :=
  let truckTankCapacity := data.minivanTankCapacity * data.truckTankMultiplier
  let totalVehicles := data.minivanCount + data.truckCount + data.sedanCount + data.motorcycleCount
  let fuelCost :=
    data.minivanFuelCost * data.minivanTankCapacity * (data.minivanCount : ℝ) +
    data.truckFuelCost * truckTankCapacity * (data.truckCount : ℝ) +
    data.sedanFuelCost * data.sedanTankCapacity * (data.sedanCount : ℝ) +
    data.motorcycleFuelCost * data.motorcycleTankCapacity * (data.motorcycleCount : ℝ)
  let serviceCost := data.serviceCost * (totalVehicles : ℝ)
  fuelCost + serviceCost

/-- Theorem stating that the total cost for the given fleet data is $612.55 --/
theorem total_cost_is_612_55 (data : FleetData)
  (h1 : data.serviceCost = 2.30)
  (h2 : data.minivanCount = 4)
  (h3 : data.truckCount = 2)
  (h4 : data.sedanCount = 3)
  (h5 : data.motorcycleCount = 5)
  (h6 : data.minivanFuelCost = 0.70)
  (h7 : data.truckFuelCost = 0.85)
  (h8 : data.sedanFuelCost = 0.75)
  (h9 : data.motorcycleFuelCost = 0.60)
  (h10 : data.minivanTankCapacity = 65)
  (h11 : data.sedanTankCapacity = 45)
  (h12 : data.motorcycleTankCapacity = 18)
  (h13 : data.truckTankMultiplier = 2.2) :
  totalCost data = 612.55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_612_55_l1312_131249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_components_produced_is_150_l1312_131207

/-- Represents the manufacturer's production and sales scenario --/
structure ManufacturerScenario where
  production_cost : ℚ
  shipping_cost : ℚ
  fixed_cost : ℚ
  selling_price : ℚ

/-- Calculates the number of components produced and sold per month --/
def components_produced_and_sold (scenario : ManufacturerScenario) : ℚ :=
  scenario.fixed_cost / (scenario.selling_price - scenario.production_cost - scenario.shipping_cost)

/-- Theorem stating that the number of components produced and sold is 150 --/
theorem components_produced_is_150 (scenario : ManufacturerScenario)
  (h1 : scenario.production_cost = 80)
  (h2 : scenario.shipping_cost = 2)
  (h3 : scenario.fixed_cost = 16200)
  (h4 : scenario.selling_price = 190) :
  components_produced_and_sold scenario = 150 := by
  sorry

def example_scenario : ManufacturerScenario := {
  production_cost := 80,
  shipping_cost := 2,
  fixed_cost := 16200,
  selling_price := 190
}

#eval components_produced_and_sold example_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_components_produced_is_150_l1312_131207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_60_l1312_131260

/-- The sum of the positive factors of 60 is 168. -/
theorem sum_of_factors_60 : (Finset.filter (λ x ↦ 60 % x = 0) (Finset.range 61)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_60_l1312_131260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l1312_131216

def satisfies_conditions (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    1100 ≤ n ∧ n < 2200 ∧
    d = b + c ∧
    b = a - 1 ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

-- Define a decidable version of the predicate
def satisfies_conditions_decidable (n : ℕ) : Bool :=
  match n with
  | n => 
    let a := n / 1000
    let b := (n / 100) % 10
    let c := (n / 10) % 10
    let d := n % 10
    1100 ≤ n ∧ n < 2200 ∧
    d = b + c ∧
    b = a - 1 ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

theorem count_special_numbers :
  (Finset.filter (λ n => satisfies_conditions_decidable n) (Finset.range 2200)).card = 19 :=
by
  sorry -- The proof is omitted for brevity

#eval (Finset.filter (λ n => satisfies_conditions_decidable n) (Finset.range 2200)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l1312_131216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_sum_max_l1312_131246

/-- A quadrilateral with ordered side lengths -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_order : d ≤ c ∧ c ≤ b ∧ b ≤ a

/-- The sum of lengths of lines drawn from an interior point to the sides -/
def interior_sum (q : Quadrilateral) (p : ℝ) : ℝ := sorry

/-- The maximum value of the interior sum is bounded by the sum of all sides -/
theorem interior_sum_max (q : Quadrilateral) : 
  ∀ p, interior_sum q p ≤ q.a + q.b + q.c + q.d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_sum_max_l1312_131246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelograms_in_triangular_grid_l1312_131288

/-- The number of parallelograms in a triangular grid of side n. -/
def number_of_parallelograms (n : ℕ) : ℕ := 
  3 * Nat.choose (n + 2) 4

/-- 
An equilateral triangle of side n can be divided into n^2 equilateral triangles of side 1.
This function represents that division.
-/
def divide_triangle (n : ℕ) : ℕ := n^2

/-- 
Given a triangular grid obtained by dividing an equilateral triangle of side n into n^2 
equilateral triangles of side 1, the number of parallelograms in this grid is 3 * binomial(n+2, 4).
-/
theorem parallelograms_in_triangular_grid (n : ℕ) : 
  (number_of_parallelograms n) = 3 * Nat.choose (n + 2) 4 := by
  -- Unfold the definition of number_of_parallelograms
  unfold number_of_parallelograms
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelograms_in_triangular_grid_l1312_131288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_is_32_l1312_131284

/-- The time to walk to the bus stop at the usual speed -/
def usual_time : ℝ := sorry

/-- The time to walk to the bus stop at 4/5 of the usual speed -/
def slower_time : ℝ := sorry

/-- The relationship between usual_time and slower_time -/
axiom time_relation : slower_time = usual_time + 8

/-- The speed relationship -/
axiom speed_relation : (4 / 5) * usual_time = slower_time

theorem usual_time_is_32 : usual_time = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_is_32_l1312_131284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_growth_rate_is_half_inch_l1312_131297

/-- Represents the grass cutting scenario -/
structure GrassCutting where
  initial_height : ℚ  -- Initial grass height after cutting (in inches)
  cut_height : ℚ      -- Height at which grass is cut (in inches)
  cost_per_cut : ℚ    -- Cost per cut (in dollars)
  annual_cost : ℚ     -- Annual cost for grass cutting (in dollars)

/-- Calculates the monthly grass growth rate -/
def monthly_growth_rate (gc : GrassCutting) : ℚ :=
  let cuts_per_year := gc.annual_cost / gc.cost_per_cut
  let growth_per_cut := gc.cut_height - gc.initial_height
  let annual_growth := growth_per_cut * cuts_per_year
  annual_growth / 12

/-- Theorem stating that the monthly growth rate is 0.5 inches -/
theorem monthly_growth_rate_is_half_inch (gc : GrassCutting) 
    (h1 : gc.initial_height = 2)
    (h2 : gc.cut_height = 4)
    (h3 : gc.cost_per_cut = 100)
    (h4 : gc.annual_cost = 300) : 
  monthly_growth_rate gc = 1/2 := by
  sorry

#eval monthly_growth_rate { initial_height := 2, cut_height := 4, cost_per_cut := 100, annual_cost := 300 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_growth_rate_is_half_inch_l1312_131297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1312_131227

/-- The circle C with center (2, 1) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Tangent line from a point to a circle -/
def isTangent (p q : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop :=
  q ∈ s ∧ ∀ r ∈ s, r ≠ q → distance p r > distance p q

theorem tangent_length :
  ∀ P ∈ C, isTangent O P C → distance O P = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1312_131227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_condition_l1312_131259

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem count_integers_satisfying_condition :
  ∃ (S : Finset ℕ), (∀ x ∈ S, floor ((x + 1 : ℝ) / 3) = 3) ∧ S.card = 3 ∧
   ∀ y : ℕ, floor ((y + 1 : ℝ) / 3) = 3 → y ∈ S :=
by
  -- Proof goes here
  sorry

#check count_integers_satisfying_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_condition_l1312_131259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1312_131291

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.c^2 - t.a^2 = 5 * t.b)
  (h2 : 3 * Real.sin t.A * Real.cos t.C = Real.cos t.A * Real.sin t.C) :
  t.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1312_131291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_intersection_product_l1312_131241

noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1/2 + (Real.sqrt 3 / 2) * t, 1/2 + (1/2) * t)

def circle_C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

axiom intersect_points_exist : ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  let (x₁, y₁) := line_l t₁
  let (x₂, y₂) := line_l t₂
  circle_C_cartesian x₁ y₁ ∧ circle_C_cartesian x₂ y₂

axiom point_A_exists : ∃ (t_A : ℝ), ∀ (t₁ t₂ : ℝ),
  (circle_C_cartesian (line_l t₁).1 (line_l t₁).2 ∧
   circle_C_cartesian (line_l t₂).1 (line_l t₂).2) →
  t_A ≠ t₁ ∧ t_A ≠ t₂

theorem circle_equation_and_intersection_product :
  (∀ θ : ℝ, (circle_C θ * Real.cos θ)^2 + (circle_C θ * Real.sin θ)^2 = (2 * Real.cos θ)^2) ∧
  (∃ A P Q : ℝ × ℝ, 
    (∃ t_A t_P t_Q : ℝ, A = line_l t_A ∧ P = line_l t_P ∧ Q = line_l t_Q) ∧
    circle_C_cartesian P.1 P.2 ∧ circle_C_cartesian Q.1 Q.2 ∧
    A ≠ P ∧ A ≠ Q ∧
    ((A.1 - P.1)^2 + (A.2 - P.2)^2) * ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_intersection_product_l1312_131241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_ln_l1312_131281

open Real MeasureTheory

/-- The integrand function -/
noncomputable def f (x : ℝ) : ℝ := (5 * tan x + 2) / (2 * sin (2 * x) + 5)

/-- Theorem stating the equality of the definite integral and its result -/
theorem integral_equals_ln : 
  ∫ x in Set.Icc 0 (π/4), f x = (1/2) * log (14/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_ln_l1312_131281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1312_131204

/-- The function f(x) = 1 / (x^2 + 2) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 2)

/-- The range of f is (0, 1/2] -/
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 0 < y ∧ y ≤ 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1312_131204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_25_6_pi_equals_half_l1312_131293

theorem sin_25_6_pi_equals_half : Real.sin (25 / 6 * π) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_25_6_pi_equals_half_l1312_131293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_theorem_l1312_131220

/-- A line segment represented by its endpoints -/
structure Segment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- Two segments intersect if they have a point in common -/
def intersect (s1 s2 : Segment) : Prop :=
  ∃ x : ℝ, s1.left ≤ x ∧ x ≤ s1.right ∧ s2.left ≤ x ∧ x ≤ s2.right

/-- A set of segments is pairwise disjoint if no two distinct segments intersect -/
def pairwiseDisjoint (S : Set Segment) : Prop :=
  ∀ s1 s2, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → ¬(intersect s1 s2)

/-- Main theorem: Given 50 segments, either 8 share a point or 8 are pairwise disjoint -/
theorem segment_intersection_theorem (segments : Finset Segment) (h : segments.card = 50) :
    (∃ (point : ℝ) (S : Finset Segment), S ⊆ segments ∧ S.card = 8 ∧ ∀ s ∈ S, s.left ≤ point ∧ point ≤ s.right) ∨
    (∃ S : Finset Segment, S ⊆ segments ∧ S.card = 8 ∧ pairwiseDisjoint S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_theorem_l1312_131220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1312_131266

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp (a * x) + b * x

-- State the theorem
theorem function_properties :
  ∀ a b : ℝ, a < 0 →
  (∀ x : ℝ, (Real.exp (a * 0) + b * 0) + (a * Real.exp (a * 0) + b) * x = 5 * x + 1) →
  (Real.exp (a * 1) + b * 1) + (a * Real.exp (a * 1) + b) = 12 →
  (∃ c : ℝ, c = 6 - 6 * Real.log 6) →
  (∀ x : ℝ, f (-1) 6 x = Real.exp (-x) + 6 * x) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f (-1) 6 x ≥ f (-1) 6 x_min ∧ f (-1) 6 x_min = 6 - 6 * Real.log 6) ∧
  (¬∃ x_max : ℝ, ∀ x : ℝ, f (-1) 6 x ≤ f (-1) 6 x_max) ∧
  (∀ m : ℕ, m > 5 → ∃ x : ℝ, x ∈ Set.Icc (1 : ℝ) (m : ℝ) ∧ f (-1) 6 x ≤ x^2 + 3) ∧
  (∀ x : ℝ, x ∈ Set.Icc (1 : ℝ) (5 : ℝ) → f (-1) 6 x > x^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1312_131266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_extrema_f_geq_g_implies_a_l1312_131273

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (a x : ℝ) : ℝ := a - 4 * Real.sqrt 3 * Real.sin x
noncomputable def h (a x : ℝ) : ℝ := f x + g a x

-- Part 1
theorem h_extrema :
  (∀ x, h 0 x ≤ 5) ∧ 
  (∃ x, h 0 x = 5) ∧ 
  (∀ x, h 0 x ≥ -4 - 4 * Real.sqrt 3) ∧ 
  (∃ x, h 0 x = -4 - 4 * Real.sqrt 3) :=
sorry

-- Part 2
theorem f_geq_g_implies_a (n m : ℝ) :
  m - n = 5 * Real.pi / 3 →
  (∀ x ∈ Set.Icc n m, f x ≥ g (-7) x) →
  (-7 : ℝ) = -7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_extrema_f_geq_g_implies_a_l1312_131273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1312_131209

def M : Set ℝ := {x | x = 1 ∨ ∃ y : ℝ, x = y^2}

theorem range_of_x : {x : ℝ | x ≠ 1 ∧ x ≠ -1} = {x : ℝ | ∃ y ∈ M, x = Real.sqrt y ∨ x = -Real.sqrt y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1312_131209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1312_131253

/-- Definition of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of the problem conditions -/
def ProblemConditions (C₁ C₂ : Circle) (m : ℝ) (a b c : ℕ) : Prop :=
  ∃ (x₀ y₀ : ℝ),
    -- Circles intersect at (10, 8)
    (x₀ - 10)^2 + (y₀ - 8)^2 = C₁.radius^2 ∧
    (x₀ - 10)^2 + (y₀ - 8)^2 = C₂.radius^2 ∧
    -- Product of radii is 75
    C₁.radius * C₂.radius = 75 ∧
    -- x-axis is tangent to both circles
    C₁.center.2 = C₁.radius ∧
    C₂.center.2 = C₂.radius ∧
    -- Line y = 2mx is tangent to both circles
    (∃ (x₁ y₁ : ℝ), y₁ = 2 * m * x₁ ∧ (x₁ - C₁.center.1)^2 + (y₁ - C₁.center.2)^2 = C₁.radius^2) ∧
    (∃ (x₂ y₂ : ℝ), y₂ = 2 * m * x₂ ∧ (x₂ - C₂.center.1)^2 + (y₂ - C₂.center.2)^2 = C₂.radius^2) ∧
    -- m is positive
    m > 0 ∧
    -- m can be expressed as a√b/c
    m = (a : ℝ) * Real.sqrt (b : ℝ) / c ∧
    -- b is not divisible by the square of any prime
    ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ b) ∧
    -- a and c are relatively prime
    Nat.Coprime a c

/-- The main theorem -/
theorem problem_solution (C₁ C₂ : Circle) (m : ℝ) (a b c : ℕ) 
  (h : ProblemConditions C₁ C₂ m a b c) : a + b + c = 152 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1312_131253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_all_sums_is_eleven_l1312_131278

/-- Represents a four-digit number formed by distinct, non-consecutive digits --/
structure FourDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  h_digit : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10
  h_nonconsecutive : (a + 1 ≠ b ∧ a ≠ b + 1) ∧ 
                     (a + 1 ≠ c ∧ a ≠ c + 1) ∧ 
                     (a + 1 ≠ d ∧ a ≠ d + 1) ∧ 
                     (b + 1 ≠ c ∧ b ≠ c + 1) ∧ 
                     (b + 1 ≠ d ∧ b ≠ d + 1) ∧ 
                     (c + 1 ≠ d ∧ c ≠ d + 1)

/-- The sum of a four-digit number and its reverse --/
def sum_with_reverse (n : FourDigitNumber) : ℕ :=
  1001 * (n.a + n.d) + 110 * (n.b + n.c)

/-- The theorem stating that the GCD of all possible sums is 11 --/
theorem gcd_of_all_sums_is_eleven :
  ∃ (k : ℕ), k > 1 ∧ ∀ (n : FourDigitNumber), k ∣ sum_with_reverse n ∧
  ∀ (m : ℕ), (∀ (n : FourDigitNumber), m ∣ sum_with_reverse n) → m ∣ k :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_all_sums_is_eleven_l1312_131278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1312_131231

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem omega_value (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x y, x ∈ Set.Icc 0 π → y ∈ Set.Icc 0 π → x < y → 
    (f A ω φ x < f A ω φ y ∨ f A ω φ x > f A ω φ y)) →
  f A ω φ (-π) = f A ω φ 0 ∧ f A ω φ 0 = -f A ω φ (π/2) →
  ω = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1312_131231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_fifteen_eighths_l1312_131203

theorem ceiling_sum_fifteen_eighths : 
  ⌈(15/8 : ℚ).sqrt⌉ + ⌈(15/8 : ℚ)⌉ + ⌈(15/8 : ℚ)^2⌉ + ⌈(15/8 : ℚ)^3⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_fifteen_eighths_l1312_131203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_convex_20gon_l1312_131242

/-- The number of sides in the polygon -/
def n : ℕ := 20

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℕ := (n - 2) * 180

/-- Predicate to check if a list of angles is valid for a convex polygon -/
def is_valid_convex_polygon (angles : List ℕ) : Prop :=
  angles.length = n ∧ 
  angles.sum = interior_angle_sum n ∧
  ∀ a ∈ angles.toFinset, 0 < a ∧ a < 180

/-- Predicate to check if a list is an increasing arithmetic sequence -/
def is_increasing_arithmetic_seq (seq : List ℕ) : Prop :=
  seq.length > 1 ∧ 
  ∃ d : ℕ, d > 0 ∧ ∀ i : Fin (seq.length - 1), seq[i.val + 1]! = seq[i.val]! + d

theorem smallest_angle_in_convex_20gon (angles : List ℕ) :
  is_valid_convex_polygon angles →
  is_increasing_arithmetic_seq angles →
  angles[0]! = 161 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_convex_20gon_l1312_131242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_1680_l1312_131238

def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  [(2, 4), (3, 1), (5, 1), (7, 1)]

def is_perfect_square (factors : List (ℕ × ℕ)) : Bool :=
  factors.all (fun (_, exp) => exp % 2 = 0)

def count_perfect_square_factors (n : ℕ) : ℕ :=
  let factors := prime_factorization n
  let all_subfactors := factors.map (fun (base, max_exp) =>
    List.range (max_exp + 1) |>.filter (fun exp => exp % 2 = 0))
  all_subfactors.map List.length |>.foldl (· * ·) 1

theorem perfect_square_factors_of_1680 :
  count_perfect_square_factors 1680 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_1680_l1312_131238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_mass_is_6975_l1312_131262

/-- Represents the number of trees for each variety -/
def num_trees : Fin 4 → ℕ
| 0 => 20  -- Gala apple trees
| 1 => 10  -- Fuji apple trees
| 2 => 30  -- Redhaven peach trees
| 3 => 15  -- Elberta peach trees

/-- Represents the average yield (in kg) for each variety -/
def avg_yield : Fin 4 → ℕ
| 0 => 120  -- Gala apple trees
| 1 => 180  -- Fuji apple trees
| 2 => 55   -- Redhaven peach trees
| 3 => 75   -- Elberta peach trees

/-- Calculates the total mass of fruit harvested in the orchard -/
def total_mass : ℕ := (Finset.sum (Finset.range 4) fun i => num_trees i * avg_yield i)

/-- Theorem stating that the total mass of fruit harvested is 6975 kg -/
theorem total_mass_is_6975 : total_mass = 6975 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_mass_is_6975_l1312_131262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_is_one_l1312_131213

/-- A regular octahedron -/
structure RegularOctahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Surface area of a regular octahedron -/
noncomputable def surfaceAreaOctahedron (o : RegularOctahedron) : ℝ :=
  2 * Real.sqrt 3 * o.sideLength ^ 2

/-- Surface area of a regular tetrahedron -/
noncomputable def surfaceAreaTetrahedron (t : RegularTetrahedron) : ℝ :=
  Real.sqrt 3 * t.sideLength ^ 2

/-- The theorem stating that the ratio of surface areas is 1 -/
theorem surface_area_ratio_is_one
  (o : RegularOctahedron)
  (t : RegularTetrahedron)
  (h : t.sideLength = o.sideLength * Real.sqrt 2) :
  surfaceAreaOctahedron o / surfaceAreaTetrahedron t = 1 := by
  sorry

#check surface_area_ratio_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_is_one_l1312_131213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_A_l1312_131218

def A : Set ℕ := {x | 1 < x ∧ x < 4}

theorem proper_subsets_of_A : Finset.card (Finset.powerset {2, 3} \ {{2, 3}}) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_A_l1312_131218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gambler_win_percentage_l1312_131232

/-- The percentage of games won by a gambler -/
noncomputable def percentage_won (games_won : ℕ) (total_games : ℕ) : ℝ :=
  (games_won : ℝ) / (total_games : ℝ) * 100

/-- The number of games won given a winning percentage and total games -/
noncomputable def games_won (win_percentage : ℝ) (total_games : ℕ) : ℝ :=
  win_percentage / 100 * (total_games : ℝ)

theorem gambler_win_percentage :
  let initial_games : ℕ := 40
  let additional_games : ℕ := 40
  let final_win_percentage : ℝ := 60
  let new_win_percentage : ℝ := 80
  let initial_win_percentage : ℝ := percentage_won 16 initial_games
  
  (games_won final_win_percentage (initial_games + additional_games) -
   games_won new_win_percentage additional_games) = 16 ∧
  initial_win_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gambler_win_percentage_l1312_131232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1312_131258

theorem trigonometric_problem (α : ℝ)
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.sin (π / 4 - α) = Real.sqrt 10 / 10) :
  Real.tan (2 * α) = 4 / 5 ∧
  Real.sin (α + π / 4) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = Real.sqrt 10 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1312_131258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_surface_area_fraction_l1312_131228

theorem white_surface_area_fraction (large_cube_edge : ℕ) (small_cube_edge : ℕ) 
  (total_small_cubes : ℕ) (white_cubes : ℕ) (black_cubes : ℕ) :
  large_cube_edge = 4 →
  small_cube_edge = 1 →
  total_small_cubes = 64 →
  white_cubes = 48 →
  black_cubes = 16 →
  white_cubes + black_cubes = total_small_cubes →
  (large_cube_edge : ℚ) / (small_cube_edge : ℚ) = (total_small_cubes : ℚ)^(1/3) →
  (3 : ℚ) / 4 = (6 * (large_cube_edge^2 : ℚ) - 24) / (6 * (large_cube_edge^2 : ℚ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_surface_area_fraction_l1312_131228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_range_l1312_131244

/-- The function f(x) defined on [1, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x + 1 / x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / x + a - 1 / (x^2)

/-- Theorem stating the range of a for which f(x) is monotonically decreasing -/
theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (f_derivative a x ≤ 0)) ↔ a ≤ -1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_range_l1312_131244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_double_indices_l1312_131282

def my_sequence (a : ℕ → ℕ) : Prop :=
  a 1 > 10 ∧ ∀ n > 1, a n = a (n - 1) + Nat.gcd n (a (n - 1))

theorem infinite_double_indices (a : ℕ → ℕ) (h : my_sequence a) :
  (∃ i, a i = 2 * i) → ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, a (f k) = 2 * f k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_double_indices_l1312_131282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l1312_131248

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((x^2 + a) / x) / Real.log a

theorem minimum_value_implies_a (a : ℝ) :
  a > 0 →
  (∃ (min : ℝ), min = 1 ∧ (∀ x > 0, f a x ≥ min)) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l1312_131248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1312_131272

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 3) + 1 / (x - 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici (3/2) \ {3} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1312_131272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_six_numbers_l1312_131264

theorem average_of_six_numbers (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) :
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₃ + n₄) / 2 = 3.85 →
  (n₅ + n₆) / 2 = 4.600000000000001 →
  |((n₁ + n₂) / 2 - 3.4)| < 0.000000000000001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_six_numbers_l1312_131264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pa_range_l1312_131280

/-- Given a line segment AB of length 2 and a point P such that |PA| + |PB| = 6,
    prove that the minimum value of |PA| is 2 and the maximum value of |PA| is 4. -/
theorem pa_range (A B P : ℝ × ℝ) : 
  let d := λ (x y : ℝ × ℝ) ↦ Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)
  (d A B = 2) → (d A P + d B P = 6) → 
  (∀ P', d A P' + d B P' = 6 → 2 ≤ d A P' ∧ d A P' ≤ 4) :=
by
  sorry

#check pa_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pa_range_l1312_131280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_area_not_equal_to_special_number_l1312_131256

theorem dodecagon_area_not_equal_to_special_number : ¬ ∃ (a : ℕ), 
  (3 * a^2) / 2 = 2 * (10^2017 - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_area_not_equal_to_special_number_l1312_131256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_diagonal_polygon_is_quad_or_pent_l1312_131277

/-- A convex polygon with n sides and all diagonals equal -/
structure EqualDiagonalPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)
  n_ge_4 : n ≥ 4
  equal_diagonals : ∀ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ (i, j) ≠ (k, l) →
    dist (vertices i) (vertices j) = dist (vertices k) (vertices l)

/-- The theorem stating that a polygon with equal diagonals must be either a quadrilateral or a pentagon -/
theorem equal_diagonal_polygon_is_quad_or_pent (F : EqualDiagonalPolygon) :
  F.n = 4 ∨ F.n = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_diagonal_polygon_is_quad_or_pent_l1312_131277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l1312_131279

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

theorem min_shift_value (m : ℝ) (h₁ : m > 0) (h₂ : g m 0 = 1) :
  ∀ k : ℝ, k > 0 ∧ g k 0 = 1 → m ≤ k →
  m = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l1312_131279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_locus_is_ellipse_l1312_131265

/-- The set of complex numbers Z satisfying |Z+i|+|Z-i|=4 forms an ellipse -/
theorem complex_locus_is_ellipse :
  {Z : ℂ | Complex.abs (Z + Complex.I) + Complex.abs (Z - Complex.I) = 4} =
  {Z : ℂ | ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
    (Z.re / a)^2 + (Z.im / b)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_locus_is_ellipse_l1312_131265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_extrema_l1312_131215

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

theorem cubic_function_extrema (a b : ℝ) : 
  (∀ x : ℝ, (deriv (f a b)) x = 3*x^2 + 2*a*x + b) →
  (deriv (f a b)) (-1) = 0 →
  (deriv (f a b)) 3 = 0 →
  a = -3 ∧ b = -9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_extrema_l1312_131215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cabbage_increase_l1312_131219

theorem cabbage_increase (n : ℕ) (h : n = 8281) :
  n - (Int.floor (Real.sqrt (n - 1 : ℝ)))^2 = 1720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cabbage_increase_l1312_131219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_fourth_power_l1312_131287

theorem det_B_fourth_power {n : Type*} [Fintype n] [DecidableEq n] 
  (B : Matrix n n ℝ) (h : Matrix.det B = -3) :
  Matrix.det (B ^ 4) = 81 := by
  have h1 : Matrix.det (B ^ 4) = (Matrix.det B) ^ 4 := by
    exact Matrix.det_pow B 4
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_fourth_power_l1312_131287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_increase_l1312_131212

/-- Represents the speed increase of a car per hour -/
noncomputable def speed_increase : ℝ → ℝ := λ x =>
  let first_hour := 50
  let total_hours := 12
  let total_distance := 732
  let distance_sum := total_hours / 2 * (2 * first_hour + (total_hours - 1) * x)
  distance_sum

/-- Theorem stating that the speed increase is 2 km/h -/
theorem car_speed_increase :
  ∃ (x : ℝ), speed_increase x = 732 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_increase_l1312_131212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l1312_131254

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem first_term_of_geometric_sequence 
  (a₁ : ℝ) (q : ℝ) (h_q : q ≠ 1) :
  geometric_sum a₁ q 4 = 240 → 
  geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 = 180 →
  a₁ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l1312_131254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1312_131217

/-- Given that the coefficient of x³ in ((ax-1)⁵) is 80, prove that the coefficient of x² is -40 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 5 2 : ℝ) * a^3 = 80 → (Nat.choose 5 3 : ℝ) * a^2 * (-1)^3 = -40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1312_131217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_theorem_l1312_131214

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_inequality_theorem (t : Triangle) (n : ℕ) (hn : n > 0) :
  (t.a^n / (t.b + t.c) + t.b^n / (t.c + t.a) + t.c^n / (t.a + t.b)) ≥ 
  (2/3)^(n-2) * ((t.a + t.b + t.c) / 2)^(n-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_theorem_l1312_131214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_and_normal_l1312_131299

noncomputable section

/-- Definition of the curve -/
def curve (t : Real) : Real × Real :=
  (Real.sqrt 3 * Real.cos t, Real.sin t)

/-- The parameter value -/
def t₀ : Real := Real.pi / 3

/-- The point on the curve at t₀ -/
def point : Real × Real := curve t₀

/-- The tangent line equation -/
def tangent_line (x : Real) : Real :=
  -1/3 * x + 2 * Real.sqrt 3 / 3

/-- The normal line equation -/
def normal_line (x : Real) : Real :=
  3 * x - Real.sqrt 3

/-- Theorem stating the equations of the tangent and normal lines -/
theorem curve_tangent_and_normal :
  (∀ x, tangent_line x = -1/3 * x + 2 * Real.sqrt 3 / 3) ∧
  (∀ x, normal_line x = 3 * x - Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_and_normal_l1312_131299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_S_approx_64_07_l1312_131202

-- Define the molar masses and composition
noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_S : ℝ := 32.06
def Al_atoms : ℕ := 2
def S_atoms : ℕ := 3

-- Define the molar mass of Al2S3
noncomputable def molar_mass_Al2S3 : ℝ := Al_atoms * molar_mass_Al + S_atoms * molar_mass_S

-- Define the mass percentage calculation
noncomputable def mass_percentage_S : ℝ := (S_atoms * molar_mass_S / molar_mass_Al2S3) * 100

-- Theorem to prove
theorem mass_percentage_S_approx_64_07 :
  abs (mass_percentage_S - 64.07) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_S_approx_64_07_l1312_131202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1312_131257

/-- The time taken for two workers to complete a job together, given their individual completion times -/
noncomputable def time_together (time_b : ℝ) (time_a : ℝ) : ℝ :=
  1 / (1 / time_b + 1 / time_a)

/-- Theorem: Given worker b can complete a job in 30 days, and worker a is twice as fast as b,
    the time taken for a and b to complete the job together is 10 days -/
theorem job_completion_time :
  let time_b : ℝ := 30
  let time_a : ℝ := time_b / 2
  time_together time_b time_a = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1312_131257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_to_f_g_geq_abs_x_minus_1_solution_c_inequality_l1312_131252

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1

noncomputable def g (x : ℝ) : ℝ := 2 * x - 1

noncomputable def solution_set : Set ℝ := {x : ℝ | x ≥ 2/3}

def c_upper_bound : ℚ := -1/2

theorem g_symmetric_to_f : ∀ x : ℝ, g x = -f (-x) := by sorry

theorem g_geq_abs_x_minus_1_solution : 
  {x : ℝ | g x ≥ |x - 1|} = solution_set := by sorry

theorem c_inequality : 
  ∀ c : ℝ, (∀ x : ℝ, |g x| - c ≥ |x - 1|) ↔ c ≤ c_upper_bound := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_to_f_g_geq_abs_x_minus_1_solution_c_inequality_l1312_131252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1312_131239

noncomputable def f (x : ℝ) : ℝ := (x^3 - 4*x) / (x^2 - 4*x + 3)

theorem inequality_solution_set :
  {x : ℝ | f x > 0} = 
    Set.Ioi (-2) ∩ Set.Iic (-2) ∪ 
    Set.Ioo (-2) 0 ∪ 
    Set.Ioo 1 2 ∪ 
    Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1312_131239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sin_squared_l1312_131223

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := Real.arcsin (-Real.sqrt x)

-- State the theorem
theorem inverse_of_sin_squared :
  ∀ x ∈ Set.Ioo (-Real.pi/2 : ℝ) 0,
  ∀ y ∈ Set.Ioo (0 : ℝ) 1,
  f x = y ↔ f_inv y = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sin_squared_l1312_131223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_tetrahedron_proof_l1312_131222

/-- The surface area of a regular tetrahedron with edge length a -/
noncomputable def surface_area_tetrahedron (a : ℝ) : ℝ := Real.sqrt 3 * a^2

/-- A regular tetrahedron consists of 4 equilateral triangles -/
def tetrahedron_faces : ℕ := 4

/-- The area of an equilateral triangle with side length a -/
noncomputable def area_equilateral_triangle (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2

theorem surface_area_tetrahedron_proof (a : ℝ) (h : a > 0) :
  surface_area_tetrahedron a = (tetrahedron_faces : ℝ) * area_equilateral_triangle a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_tetrahedron_proof_l1312_131222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_divides_difference_of_three_digit_and_reverse_l1312_131261

/-- Given a three-digit number ABC and its reverse CBA, where A ≠ C,
    11 is a factor of their difference. -/
theorem eleven_divides_difference_of_three_digit_and_reverse (A B C : ℤ) :
  A ≠ C →
  0 ≤ A ∧ A < 10 → 0 ≤ B ∧ B < 10 → 0 ≤ C ∧ C < 10 →
  11 ∣ ((100 * A + 10 * B + C) - (100 * C + 10 * B + A)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_divides_difference_of_three_digit_and_reverse_l1312_131261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_and_expression_value_l1312_131205

noncomputable section

-- Define the circle and points
def unit_circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}

-- Define the angle θ
noncomputable def θ : ℝ := Real.arcsin (4/5)

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (-3/5, 4/5)

theorem point_B_coordinates_and_expression_value :
  -- Point A is on the positive x-axis
  A ∈ unit_circle ∧ A.1 > 0 ∧ A.2 = 0 ∧
  -- Point B is in the second quadrant
  B ∈ unit_circle ∧ B.1 < 0 ∧ B.2 > 0 ∧
  -- sin θ = 4/5
  Real.sin θ = 4/5 →
  -- 1. The coordinates of point B are (-3/5, 4/5)
  B = (-3/5, 4/5) ∧
  -- 2. The value of the expression is -5/3
  (Real.sin (Real.pi + θ) + 2 * Real.sin (Real.pi/2 - θ)) / (2 * Real.cos (Real.pi - θ)) = -5/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_and_expression_value_l1312_131205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_imply_t_range_l1312_131226

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.log x

-- Define the domain of f
def domain (x : ℝ) : Prop := (0 < x ∧ x < 1) ∨ (x > 1)

-- Define the equation
def equation (t x : ℝ) : Prop := t * f x - x = 0

-- Define the interval for zeros
def zero_interval (x : ℝ) : Prop := (1/Real.exp 1 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ Real.exp 2)

-- Theorem statement
theorem zeros_imply_t_range (t : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ zero_interval x₁ ∧ zero_interval x₂ ∧ equation t x₁ ∧ equation t x₂) →
  (2 / Real.exp 2 ≤ t ∧ t < 1 / Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_imply_t_range_l1312_131226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1312_131211

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (3 - x)) / Real.log ((x - 3)^2)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x | 0 ≤ x ∧ x < 2} ∪ {x | 2 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1312_131211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lidless_cylinder_area_l1312_131268

/-- The surface area of a lidless cylinder -/
noncomputable def lidless_cylinder_surface_area (d h : ℝ) : ℝ :=
  Real.pi * (d / 2)^2 + Real.pi * d * h

/-- Theorem: The surface area of a lidless cylinder with base diameter 1dm and height 5dm is 1648.5 cm² -/
theorem lidless_cylinder_area :
  let d : ℝ := 10  -- 1dm = 10cm
  let h : ℝ := 50  -- 5dm = 50cm
  lidless_cylinder_surface_area d h = 1648.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lidless_cylinder_area_l1312_131268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_l1312_131275

theorem cricket_average (initial_innings : ℕ) (next_innings_runs : ℕ) (average_increase : ℕ) 
  (h1 : initial_innings = 20)
  (h2 : next_innings_runs = 137)
  (h3 : average_increase = 5) :
  ∃ initial_average : ℚ,
    (initial_average * initial_innings + next_innings_runs) / (initial_innings + 1 : ℚ) = 
    initial_average + average_increase ∧
    initial_average = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_l1312_131275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approximately_8463_855_l1312_131237

-- Define the circumferences of the two circles
def c1 : ℝ := 132
def c2 : ℝ := 352

-- Define pi as a constant
noncomputable def π : ℝ := Real.pi

-- Define the radii of the circles
noncomputable def r1 : ℝ := c1 / (2 * π)
noncomputable def r2 : ℝ := c2 / (2 * π)

-- Define the areas of the circles
noncomputable def a1 : ℝ := π * r1^2
noncomputable def a2 : ℝ := π * r2^2

-- Theorem statement
theorem area_difference_approximately_8463_855 :
  abs (a2 - a1 - 8463.855) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approximately_8463_855_l1312_131237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_exclusion_l1312_131274

theorem average_after_exclusion (numbers : Fin 5 → ℝ) 
  (h_avg : (Finset.sum Finset.univ (λ i => numbers i)) / 5 = 12)
  (h_exclude : numbers 4 = 20) :
  ((Finset.sum Finset.univ (λ i => numbers i) - numbers 4) / 4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_exclusion_l1312_131274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_radius_l1312_131286

/-- The radius of the inscribed circle in an isosceles triangle with two sides of length 8 and base of length 5 -/
noncomputable def inscribed_circle_radius (a b : ℝ) : ℝ :=
  let s := (2 * a + b) / 2
  let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  area / s

theorem isosceles_triangle_inscribed_circle_radius :
  inscribed_circle_radius 8 5 = 76 * Real.sqrt 10 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_radius_l1312_131286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_four_l1312_131251

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 2^(-x) * (1 - a^x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (2:ℝ)^(-x) * (1 - a^x)

/-- Theorem: If f(x) = 2^(-x) * (1 - a^x) is an odd function, where a > 0 and a ≠ 1, then a = 4 -/
theorem odd_function_implies_a_equals_four (a : ℝ) 
    (h1 : a > 0) 
    (h2 : a ≠ 1) 
    (h3 : IsOdd (f a)) : 
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_four_l1312_131251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reroll_two_is_one_sixth_l1312_131263

/-- A fair standard six-sided die -/
def Die : Type := Fin 6

/-- The result of rolling three dice -/
def RollResult : Type := Die × Die × Die

/-- The optimal strategy for rerolling dice to get a sum of 9 -/
def optimal_strategy (roll : RollResult) : Finset Die := sorry

/-- The probability of an event occurring -/
noncomputable def probability (event : Set RollResult) : ℚ := sorry

/-- The set of all possible roll results -/
def all_rolls : Set RollResult := sorry

/-- The set of roll results where rerolling exactly two dice is optimal -/
def reroll_two_optimal : Set RollResult :=
  {roll ∈ all_rolls | Finset.card (optimal_strategy roll) = 2}

/-- The main theorem stating that the probability of rerolling exactly two dice is 1/6 -/
theorem probability_reroll_two_is_one_sixth :
  probability reroll_two_optimal = 1/6 := by
  sorry

#check probability_reroll_two_is_one_sixth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reroll_two_is_one_sixth_l1312_131263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1312_131224

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := 3 * cos (2 * x - Real.pi / 5)

-- Define the translated function g
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 3)

-- Define the axis of symmetry
noncomputable def axis_of_symmetry (k : ℤ) : ℝ := k * Real.pi / 2 + 13 * Real.pi / 30

-- Theorem statement
theorem symmetry_axis_of_g :
  ∀ (k : ℤ), ∀ (x : ℝ),
    g (axis_of_symmetry k + x) = g (axis_of_symmetry k - x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1312_131224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_rate_l1312_131206

/-- Represents the growth rate of bacteria over time -/
noncomputable def growth_rate (initial : ℝ) (time : ℕ) (final : ℝ) : ℝ :=
  (final / initial) ^ (1 / time : ℝ)

/-- Theorem stating the growth rate of bacteria under given conditions -/
theorem bacteria_growth_rate :
  ∀ (initial : ℝ) (capacity : ℝ),
  initial > 0 → capacity > 0 →
  growth_rate initial 25 (capacity / 32) = growth_rate initial 30 capacity →
  growth_rate initial 30 capacity = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_rate_l1312_131206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_paint_cost_l1312_131210

/-- The cost to paint a cube with given dimensions and paint properties -/
noncomputable def paint_cost (cube_side : ℝ) (paint_cost_per_kg : ℝ) (paint_coverage_per_kg : ℝ) : ℝ :=
  let surface_area := 6 * cube_side^2
  let paint_needed := surface_area / paint_coverage_per_kg
  paint_needed * paint_cost_per_kg

/-- Theorem stating the cost to paint the cube under given conditions -/
theorem cube_paint_cost :
  paint_cost 8 36.5 16 = 876 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_paint_cost_l1312_131210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_six_in_ten_steps_l1312_131230

/-- A random walk on a number line -/
def RandomWalk := List Int

/-- The probability of a specific walk occurring -/
def probability (walk : RandomWalk) : ℚ :=
  (1 / 2) ^ walk.length

/-- Check if a walk reaches or crosses a certain position -/
def reachesOrCrosses (walk : RandomWalk) (position : Int) : Bool :=
  walk.scanl (·+·) 0 |>.any (fun x => x ≥ position)

/-- All possible walks of a given length -/
def allWalks (n : ℕ) : List RandomWalk :=
  sorry

/-- The probability of reaching or crossing a position in n steps -/
noncomputable def probReachOrCross (n : ℕ) (position : Int) : ℚ :=
  (allWalks n).filter (fun w => reachesOrCrosses w position)
    |>.map probability
    |>.sum

theorem prob_reach_six_in_ten_steps :
  probReachOrCross 10 6 = 45 / 512 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_six_in_ten_steps_l1312_131230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_reach_single_pawn_l1312_131240

/-- Represents a position on the chessboard -/
structure Position :=
  (pawns : Set (ℤ × ℤ))

/-- Represents a valid move in the game -/
inductive Move : Position → Position → Prop
  | slide {p q : Position} {i j : ℤ} :
    (i, j) ∈ p.pawns →
    (i + 1, j) ∈ p.pawns →
    (i + 2, j) ∉ p.pawns →
    q.pawns = p.pawns \ {(i, j), (i + 1, j)} ∪ {(i + 2, j)} →
    Move p q
  | slide_vertical {p q : Position} {i j : ℤ} :
    (i, j) ∈ p.pawns →
    (i, j + 1) ∈ p.pawns →
    (i, j + 2) ∉ p.pawns →
    q.pawns = p.pawns \ {(i, j), (i, j + 1)} ∪ {(i, j + 2)} →
    Move p q

/-- Represents a sequence of valid moves -/
def Reachable (start finish : Position) : Prop :=
  ∃ (n : ℕ) (seq : ℕ → Position),
    seq 0 = start ∧
    seq (n - 1) = finish ∧
    ∀ i : ℕ, i < n - 1 → Move (seq i) (seq (i + 1))

/-- Initial position with a 3k × n rectangle filled with pawns -/
def initial_position (k n : ℕ) : Position :=
  ⟨{(i, j) | 0 ≤ i ∧ i < 3 * k ∧ 0 ≤ j ∧ j < n}⟩

/-- Final position with only one pawn -/
def final_position : Position :=
  ⟨{(0, 0)}⟩

theorem impossible_to_reach_single_pawn (k n : ℕ) (h1 : k > 0) (h2 : n > 0) :
  ¬ Reachable (initial_position k n) final_position := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_reach_single_pawn_l1312_131240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_a_geq_four_l1312_131292

-- Define the function f(x) = ax - x³
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

-- Define the open interval (0, 1)
def openUnitInterval : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∀ (x₁ x₂ : ℝ), x₁ ∈ openUnitInterval → x₂ ∈ openUnitInterval → x₁ < x₂ → 
    f a x₂ - f a x₁ > x₂ - x₁) → 
  a ≥ 4 :=
by
  sorry

-- Proof that the minimum value of a is indeed 4
theorem a_geq_four (a : ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), x₁ ∈ openUnitInterval → x₂ ∈ openUnitInterval → x₁ < x₂ → 
    f a x₂ - f a x₁ > x₂ - x₁) : 
  a ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_a_geq_four_l1312_131292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_sum_l1312_131234

theorem power_of_three_sum (x : ℝ) : (3 : ℝ)^x + (3 : ℝ)^x + (3 : ℝ)^x + (3 : ℝ)^x = 6561 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_sum_l1312_131234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_existence_l1312_131285

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x < y → f x < f y

/-- The property that f(f(n)) = 2n + 1 for all positive integers n -/
def FunctionProperty (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, f (f n) = 2 * n + 1

/-- The inequalities that should hold for the function -/
def FunctionInequalities (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, (4 * ↑n + 1 : ℚ) / 3 ≤ ↑(f n) ∧ ↑(f n) ≤ (3 * ↑n + 1 : ℚ) / 2

/-- There exist infinitely many n for which the lower equality holds -/
def LowerEqualityInfinite (f : ℕ+ → ℕ+) : Prop :=
  ∀ m : ℕ+, ∃ n : ℕ+, n ≥ m ∧ ↑(f n) = (4 * ↑n + 1 : ℚ) / 3

/-- There exist infinitely many n for which the upper equality holds -/
def UpperEqualityInfinite (f : ℕ+ → ℕ+) : Prop :=
  ∀ m : ℕ+, ∃ n : ℕ+, n ≥ m ∧ ↑(f n) = (3 * ↑n + 1 : ℚ) / 2

/-- The main theorem stating the existence and uniqueness of the function with all required properties -/
theorem unique_function_existence : 
  ∃! f : ℕ+ → ℕ+, 
    StrictlyIncreasing f ∧ 
    FunctionProperty f ∧ 
    FunctionInequalities f ∧ 
    LowerEqualityInfinite f ∧ 
    UpperEqualityInfinite f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_existence_l1312_131285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l1312_131200

noncomputable section

-- Define g as a noncomputable function from ℝ to ℝ
noncomputable def g : ℝ → ℝ := sorry

-- Area between y = g(x) and x-axis
def area_g : ℝ := 15

-- Area between y = 2g(x + 3) and x-axis
noncomputable def area_transformed : ℝ := ∫ x, 2 * g (x + 3)

theorem area_transformation (h : ∫ x, g x = area_g) : 
  area_transformed = 2 * area_g := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l1312_131200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l1312_131221

-- Define the functions
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2^x + m - 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log m

-- Define the properties
def has_zero_points (m : ℝ) : Prop := ∃ x, f m x = 0
def is_decreasing (m : ℝ) : Prop := ∀ x y, 0 < x → x < y → g m y < g m x

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_decreasing m → has_zero_points m) ∧
  (∃ m : ℝ, has_zero_points m ∧ ¬is_decreasing m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l1312_131221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l1312_131229

/-- Represents the number of seventh graders -/
def s : ℕ := 1  -- We assign a concrete value to s for demonstration

/-- The average number of minutes run per day by sixth graders -/
def sixth_grade_avg : ℚ := 18

/-- The average number of minutes run per day by seventh graders -/
def seventh_grade_avg : ℚ := 12

/-- The average number of minutes run per day by eighth graders -/
def eighth_grade_avg : ℚ := 16

/-- The number of sixth graders -/
def sixth_grade_count : ℕ := 3 * s

/-- The number of seventh graders -/
def seventh_grade_count : ℕ := s

/-- The number of eighth graders -/
def eighth_grade_count : ℕ := s

/-- The total number of students -/
def total_students : ℕ := sixth_grade_count + seventh_grade_count + eighth_grade_count

/-- The total minutes run by all students -/
def total_minutes : ℚ := 
  sixth_grade_avg * (sixth_grade_count : ℚ) + 
  seventh_grade_avg * (seventh_grade_count : ℚ) + 
  eighth_grade_avg * (eighth_grade_count : ℚ)

/-- Theorem: The average number of minutes run per day by all students is 82/5 -/
theorem average_minutes_run : 
  total_minutes / (total_students : ℚ) = 82 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l1312_131229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1312_131298

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 2*x + 3)

theorem f_range : Set.range f = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1312_131298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sequence_a_l1312_131250

def sequence_a : ℕ → ℚ
  | 0 => 8
  | n + 1 => sequence_a n + n

theorem min_value_sequence_a : 
  ∀ n : ℕ, n > 0 → sequence_a n / n ≥ 3.5 ∧ ∃ m : ℕ, m > 0 ∧ sequence_a m / m = 3.5 := by
  sorry

#eval sequence_a 4 / 4  -- This should evaluate to 3.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sequence_a_l1312_131250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1312_131201

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x ∨ f x = 2 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1312_131201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1312_131289

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (3 * x + Real.pi / 6)

theorem smallest_positive_period : 
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  T = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1312_131289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_gamma_value_l1312_131235

theorem unique_gamma_value (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α ≤ π/2)
  (h2 : 0 ≤ β ∧ β ≤ π/2)
  (h3 : 0 ≤ γ ∧ γ ≤ π/2)
  (h4 : Real.sin α - Real.cos β = Real.tan γ)
  (h5 : Real.sin β - Real.cos α = 1 / Real.tan γ) :
  γ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_gamma_value_l1312_131235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_exists_solution_l1312_131255

theorem negation_of_all_exists_solution : 
  ¬(∀ (a : ℝ), ∃ (x : ℝ), Real.sqrt x - a * x = 0) ↔ 
  (∃ (a : ℝ), ∀ (x : ℝ), Real.sqrt x - a * x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_exists_solution_l1312_131255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_one_fifth_and_one_third_l1312_131270

-- State the theorem
theorem midpoint_of_one_fifth_and_one_third :
  (1/5 : ℚ) + ((1/3 : ℚ) - (1/5 : ℚ)) / 2 = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_one_fifth_and_one_third_l1312_131270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l1312_131283

/-- The angle between two lines given by their equations -/
noncomputable def angle_between_lines (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ :=
  Real.arccos (|a1 * a2 + b1 * b2| / (Real.sqrt (a1^2 + b1^2) * Real.sqrt (a2^2 + b2^2)))

/-- Theorem: The angle between l₁: √3x - y + 2 = 0 and l₂: 3x + √3y - 5 = 0 is π/3 -/
theorem angle_between_specific_lines :
  angle_between_lines (Real.sqrt 3) (-1) 2 3 (Real.sqrt 3) (-5) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l1312_131283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_l1312_131208

noncomputable def f (x : ℝ) : ℝ := -4 / x

theorem inverse_proportion_decreasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁ ≠ 0 → x₂ ≠ 0 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_l1312_131208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_problem_l1312_131290

def A (x y : ℝ) : Set ℝ := {x, y}
def B (x : ℝ) : Set ℝ := {x + 1, 5}

theorem set_union_problem (x y : ℝ) (h : A x y ∩ B x = {2}) : A x y ∪ B x = {1, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_problem_l1312_131290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1312_131269

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_properties :
  -- 1. g(x) = -sin(2x)
  (∀ x, g x = -Real.sin (2 * x)) ∧
  -- 2. g(x) is monotonically decreasing on (0, π/4)
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 4 → g y < g x) ∧
  -- 3. g(x) is an odd function
  (∀ x, g (-x) = -g x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1312_131269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1312_131225

-- Define the pattern function
def pattern (n : ℕ) : ℚ → ℚ → Prop :=
  λ x y ↦ Real.sqrt (n + x) = n * Real.sqrt y

-- State the theorem
theorem solve_equation (a b : ℕ) :
  (∀ n : ℕ, pattern n (n / (2^n - 1)) (n / (2^n - 1))) →
  pattern 8 (b / a) (b / a) →
  a = 63 ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1312_131225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_minimum_speed_l1312_131296

/-- The minimum speed Alice needs to exceed to arrive before Bob -/
noncomputable def minimum_speed_alice (distance : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) : ℝ :=
  distance / (distance / bob_speed - alice_delay)

/-- Theorem stating the minimum speed Alice needs to exceed -/
theorem alice_minimum_speed :
  let distance := (30 : ℝ) -- miles
  let bob_speed := (40 : ℝ) -- mph
  let alice_delay := (0.5 : ℝ) -- hours (30 minutes)
  minimum_speed_alice distance bob_speed alice_delay > 60 := by
  sorry

-- Use #eval with a function that returns a rational number instead
def minimum_speed_alice_rat (distance : ℚ) (bob_speed : ℚ) (alice_delay : ℚ) : ℚ :=
  distance / (distance / bob_speed - alice_delay)

#eval minimum_speed_alice_rat 30 40 (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_minimum_speed_l1312_131296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_increasing_and_quadratic_real_roots_l1312_131267

theorem log_increasing_and_quadratic_real_roots (a : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → Real.log x₁ < Real.log x₂) ∧
  (∃ x : ℝ, x^2 - 2*a*x + 4 = 0) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_increasing_and_quadratic_real_roots_l1312_131267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_sixteen_l1312_131233

/-- 
Given a series S(x) = 1 + 3x + 5x^2 + 7x^3 + ..., where the coefficients form an arithmetic sequence 
with first term 1 and common difference 2, prove that if S(x) = 16, then x = 3/4.
-/
theorem series_sum_equals_sixteen (x : ℝ) : 
  (∑' n : ℕ, (2*n + 1) * x^n) = 16 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_sixteen_l1312_131233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_reaches_35m_at_10_7s_l1312_131245

/-- The height (in meters) of a projectile as a function of time (in seconds) -/
def projectile_height (t : ℝ) : ℝ := -4.9 * t^2 + 30.4 * t

/-- The time at which the projectile reaches a specific height -/
def time_at_height (h : ℝ) : Set ℝ := {t : ℝ | projectile_height t = h}

theorem projectile_reaches_35m_at_10_7s :
  ∃ (t : ℝ), t ∈ time_at_height 35 ∧ t > 0 ∧ ∀ (s : ℝ), s ∈ time_at_height 35 ∧ s > 0 → t ≤ s ∧ t = 10/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_reaches_35m_at_10_7s_l1312_131245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1312_131294

noncomputable def curve (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

def symmetry_line (x : ℝ) : ℝ := 2 * x

theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, ∃ y : ℝ, curve a b c d x = y ∧ curve a b c d (symmetry_line y) = x) →
  b + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1312_131294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_lines_through_point_with_intercept_sum_l1312_131295

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Get the x-intercept of a line -/
noncomputable def Line.x_intercept (l : Line) : ℝ := -l.c / l.a

/-- Get the y-intercept of a line -/
noncomputable def Line.y_intercept (l : Line) : ℝ := -l.c / l.b

theorem line_through_point_parallel (l : Line) :
  l.contains 2 1 ∧ l.parallel { a := 2, b := 3, c := 0 } →
  l = { a := 2, b := 3, c := -7 } := by
  sorry

theorem lines_through_point_with_intercept_sum (l : Line) :
  l.contains (-3) 1 ∧ l.x_intercept + l.y_intercept = -4 →
  (l = { a := 1, b := -3, c := -6 } ∨ l = { a := 1, b := 1, c := -2 }) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_lines_through_point_with_intercept_sum_l1312_131295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l1312_131276

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := -x^3/900 + 300*x - 20000

-- Define the derivative of the profit function
noncomputable def profit_derivative (x : ℝ) : ℝ := -x^2/300 + 300

-- State the theorem
theorem max_profit_at_300 :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 390 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 390 → profit y ≤ profit x ∧
  x = 300 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l1312_131276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_all_same_l1312_131236

-- Define the set of colors
inductive Color
| Red
| Yellow
| Green

-- Define a function to represent a single draw
def draw : Type := Color

-- Define a function to represent three draws
def threeDraw : Type := (draw × draw × draw)

-- Define the probability space
def Ω : Type := threeDraw

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event of not all balls being the same color
def notAllSame : Set Ω := {x : Ω | ∃ (c1 c2 : Color), c1 ≠ c2 ∧ 
  ((x.1 = c1 ∨ x.2.1 = c1 ∨ x.2.2 = c1) ∧ (x.1 = c2 ∨ x.2.1 = c2 ∨ x.2.2 = c2))}

-- Theorem statement
theorem probability_not_all_same : P notAllSame = 8/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_all_same_l1312_131236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_split_condition_l1312_131271

/-- A triangle is isosceles if two of its sides are equal -/
def Isosceles (T : Triangle) : Prop := sorry

/-- The vertex angle of a triangle -/
def VertexAngle (T : Triangle) : Real := sorry

/-- A line splits a triangle into two smaller triangles -/
def Splits (T R S : Triangle) : Prop := sorry

/-- Two triangles are similar -/
def Similar (T1 T2 : Triangle) : Prop := sorry

/-- An isosceles triangle with vertex angle α can be split into two smaller
    isosceles triangles, neither similar to the original, if and only if α = 36° -/
theorem isosceles_split_condition (α : Real) : 
  (∃ (T R S : Triangle),
    Isosceles T ∧ 
    VertexAngle T = α ∧
    Splits T R S ∧
    Isosceles R ∧
    Isosceles S ∧
    ¬ Similar R T ∧
    ¬ Similar S T) ↔ 
  α = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_split_condition_l1312_131271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_and_probability_l1312_131247

-- Define the store data
noncomputable def store_data : List (ℝ × ℝ) := [
  (2, 30), (3, 34), (5, 40), (6, 45), (8, 50), (12, 60)
]

-- Define the sum of x*y
noncomputable def sum_xy : ℝ := 1752

-- Define the linear regression coefficients
noncomputable def b_hat : ℝ := 3
noncomputable def a_hat : ℝ := 151 / 6

-- Define the minimum advertising cost for sales to exceed 100,000 yuan
def min_cost : ℕ := 25

-- Define the probability of selecting at least one efficient store
noncomputable def prob_efficient : ℚ := 4 / 5

-- Theorem statement
theorem linear_regression_and_probability :
  let x := store_data.map Prod.fst
  let y := store_data.map Prod.snd
  let n := store_data.length
  let x_mean := (x.sum) / n
  let y_mean := (y.sum) / n
  let x_squared_sum := (x.map (λ xi => xi ^ 2)).sum
  b_hat = (sum_xy - n * x_mean * y_mean) / (x_squared_sum - n * x_mean ^ 2) ∧
  a_hat = y_mean - b_hat * x_mean ∧
  (min_cost : ℝ) * b_hat + a_hat > 100 ∧
  ((min_cost - 1 : ℕ) : ℝ) * b_hat + a_hat ≤ 100 ∧
  prob_efficient = (Nat.choose 6 3 - Nat.choose 4 3) / Nat.choose 6 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_and_probability_l1312_131247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_range_l1312_131243

/-- The game operation that multiplies by 2 and subtracts 12 -/
noncomputable def operation1 (x : ℝ) : ℝ := 2 * x - 12

/-- The game operation that divides by 2 and adds 12 -/
noncomputable def operation2 (x : ℝ) : ℝ := x / 2 + 12

/-- The set of possible values for a₃ after two operations -/
noncomputable def possible_a3 (a1 : ℝ) : Finset ℝ :=
  {operation1 (operation1 a1),
   operation1 (operation2 a1),
   operation2 (operation1 a1),
   operation2 (operation2 a1)}

/-- The probability of Player A winning -/
noncomputable def prob_A_wins (a1 : ℝ) : ℝ :=
  (Finset.filter (λ a3 => a3 > a1) (possible_a3 a1)).card / 4

theorem game_winning_range :
  {a1 : ℝ | prob_A_wins a1 = 3/4} = Set.Iic 12 ∪ Set.Ici 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_range_l1312_131243
