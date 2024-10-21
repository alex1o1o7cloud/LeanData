import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decomposition_KClO3_to_O2_l1008_100851

/-- The molar mass of potassium chlorate (KClO3) in g/mol -/
noncomputable def molar_mass_KClO3 : ℝ := 122.6

/-- The mass of potassium chlorate (KClO3) in grams -/
noncomputable def mass_KClO3 : ℝ := 245

/-- The stoichiometric ratio of O2 to KClO3 in the decomposition reaction -/
noncomputable def stoichiometric_ratio : ℝ := 3 / 2

/-- Calculates the number of moles of a substance given its mass and molar mass -/
noncomputable def moles (mass : ℝ) (molar_mass : ℝ) : ℝ := mass / molar_mass

/-- The theorem states that decomposing 245 g of KClO3 produces 3.00 moles of O2 -/
theorem decomposition_KClO3_to_O2 : 
  moles mass_KClO3 molar_mass_KClO3 * stoichiometric_ratio = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decomposition_KClO3_to_O2_l1008_100851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_sale_cash_realized_l1008_100864

/-- Calculates the cash realized on selling a stock given the total amount before brokerage and the brokerage rate. -/
def cash_realized (total_amount : ℚ) (brokerage_rate : ℚ) : ℚ :=
  total_amount - (total_amount * brokerage_rate / 100).floor / 100

/-- Theorem stating that given a total amount of 101 before brokerage and a brokerage rate of 1/4%, 
    the cash realized on selling the stock is 100.75. -/
theorem stock_sale_cash_realized :
  cash_realized 101 (1/4) = 100.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_sale_cash_realized_l1008_100864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conductor_is_wang_l1008_100800

-- Define the set of surnames
inductive Surname
| Zhang
| Wang
| Li

-- Define the set of cities
inductive City
| Beijing
| Tianjin
| Shanghai

-- Define the set of roles
inductive Role
| Driver
| Conductor
| Police

-- Define a person type
structure Person where
  surname : Surname
  role : Option Role
  city : City
  likesSports : Bool

-- Define the problem setup
def problemSetup (zhang wang li : Person) : Prop :=
  -- Condition 1 and 2 are implicitly defined by the existence of zhang, wang, and li
  -- Condition 3
  li.city = City.Beijing
  -- Condition 4
  ∧ ∃ (conductor : Person), conductor.role = some Role.Conductor ∧ conductor.city = City.Tianjin
  -- Condition 5
  ∧ ¬wang.likesSports
  -- Condition 6
  ∧ ∃ (conductor passenger : Person), 
      conductor.role = some Role.Conductor 
      ∧ conductor.surname = passenger.surname 
      ∧ passenger.city = City.Shanghai
  -- Condition 7
  ∧ ∃ (conductor basketballFan : Person),
      conductor.role = some Role.Conductor
      ∧ basketballFan.likesSports
      ∧ (conductor.city = City.Tianjin ∧ basketballFan.city = City.Shanghai
         ∨ conductor.city = City.Shanghai ∧ basketballFan.city = City.Tianjin)

-- The theorem to prove
theorem conductor_is_wang (zhang wang li : Person) :
  problemSetup zhang wang li → 
  ∃ (conductor : Person), conductor.role = some Role.Conductor ∧ conductor.surname = Surname.Wang :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conductor_is_wang_l1008_100800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_work_hours_correct_l1008_100890

/-- The number of hours Bill works when painting a line with Jill -/
noncomputable def billWorkHours (B J : ℝ) : ℝ :=
  (B * (J + 1)) / (B + J)

/-- Theorem stating that billWorkHours gives the correct number of hours Bill works -/
theorem bill_work_hours_correct (B J : ℝ) (hB : B > 0) (hJ : J > 0) :
  let t := billWorkHours B J
  (1 / B) + t * (1 / B + 1 / J) = 1 := by
  sorry

#check bill_work_hours_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_work_hours_correct_l1008_100890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_l1008_100832

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n + 3^x + 2*x

theorem tangent_slope_at_point (n : ℝ) :
  (deriv (f n)) 1 = 3 + 3 * Real.log 3 ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_l1008_100832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_distances_l1008_100802

-- Define the unit circle
def unit_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- Define the perpendicularity condition
def perpendicular (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - b.1) + (b.2 - a.2) * (c.2 - b.2) = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the sum of distances from P to A, B, and C
noncomputable def sum_distances (p a b c : ℝ × ℝ) : ℝ :=
  distance p a + distance p b + distance p c

-- Theorem statement
theorem range_of_sum_distances :
  ∀ (a b c : ℝ × ℝ),
  unit_circle a ∧ unit_circle b ∧ unit_circle c ∧
  perpendicular a b c →
  ∃ (min max : ℝ),
    min = 5 ∧ max = 7 ∧
    (∀ (x : ℝ), min ≤ x ∧ x ≤ max ↔ ∃ (a' b' c' : ℝ × ℝ),
      unit_circle a' ∧ unit_circle b' ∧ unit_circle c' ∧
      perpendicular a' b' c' ∧
      x = sum_distances (2, 0) a' b' c') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_distances_l1008_100802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_implies_x_squared_range_l1008_100834

theorem cube_root_difference_implies_x_squared_range (x : ℝ) :
  (x + 9) ^ (1/3) - (x - 9) ^ (1/3) = 3 → (75 < x^2 ∧ x^2 < 85) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_implies_x_squared_range_l1008_100834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exponential_equation_l1008_100858

/-- There are infinitely many ordered pairs (x,y) of real numbers that satisfy the equation:
    25^(x^2 + x + y) + 25^(x + y^2 + y) = 100 -/
theorem infinite_solutions_exponential_equation :
  ∃ S : Set (ℝ × ℝ), (Set.Infinite S) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ S ↔ (25 : ℝ)^(x^2 + x + y) + (25 : ℝ)^(x + y^2 + y) = 100) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exponential_equation_l1008_100858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_seven_expansion_l1008_100804

theorem cos_power_seven_expansion (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ) :
  (∀ θ : ℝ, (Real.cos θ)^7 = b₁ * Real.cos θ + b₂ * Real.cos (2*θ) + b₃ * Real.cos (3*θ) + 
                        b₄ * Real.cos (4*θ) + b₅ * Real.cos (5*θ) + b₆ * Real.cos (6*θ) + 
                        b₇ * Real.cos (7*θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429/1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_seven_expansion_l1008_100804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l1008_100806

-- Define the curves C1 and C2
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

def C2 (θ ρ : ℝ) : Prop := ρ * (Real.cos θ)^2 = Real.sin θ

-- Define the ray l
noncomputable def l (α t : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

-- Define the intersection points
noncomputable def OA (α : ℝ) : ℝ := 4 * Real.cos α

noncomputable def OB (α : ℝ) : ℝ := Real.sin α / (Real.cos α)^2

-- Theorem statement
theorem intersection_product_range :
  ∀ α : ℝ, π/6 < α ∧ α ≤ π/4 →
  4*Real.sqrt 3/3 < OA α * OB α ∧ OA α * OB α ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l1008_100806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l1008_100870

theorem unique_solution_for_exponential_equation :
  ∀ a b : ℕ,
  a > b →
  a > 0 →
  b > 0 →
  (a - b : ℕ) ^ (a * b) = a ^ b * b ^ a →
  a = 4 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l1008_100870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1008_100879

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (60 - x)) + Real.sqrt (x * (4 - x))

-- State the theorem
theorem max_value_of_g :
  ∃ (x₁ : ℝ), 
    0 ≤ x₁ ∧ x₁ ≤ 4 ∧
    g x₁ = 16 ∧
    (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ 16) ∧
    x₁ = 15 / 4 := by
  sorry

#check max_value_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1008_100879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l1008_100887

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := 180 * (n - 2)  -- formula for sum of interior angles
  let one_angle : ℝ := sum_of_interior_angles / n  -- measure of one interior angle
  have : one_angle = 135 := by
    -- Proof goes here
    sorry
  exact one_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l1008_100887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_face_in_last_round_probability_is_correct_l1008_100838

/-- 
Represents a tournament with the given properties:
- n is a positive integer
- There are 2^(n+1) players
- Each player has a 50% chance of winning against any other player
- The tournament has n+1 rounds
- In each round, players are paired randomly and losers are eliminated
-/
structure Tournament (n : ℕ) where
  players : Fin (2^(n+1)) → Unit
  win_probability : ℚ
  rounds : Fin (n+2) → Unit
  elimination : Fin (2^(n+1)) → Fin (n+2) → Bool

/-- The probability of two specific players facing each other in the last round -/
def face_in_last_round_probability (n : ℕ) : ℚ :=
  1 / 4^n

/-- 
Theorem stating that the probability of two specific players 
facing each other in the last round is 1/4^n 
-/
theorem face_in_last_round_probability_is_correct (n : ℕ) (t : Tournament n) :
  t.win_probability = 1/2 → face_in_last_round_probability n = 1 / 4^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_face_in_last_round_probability_is_correct_l1008_100838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l1008_100846

/-- The length of the chord cut from a circle by a line -/
noncomputable def chord_length (r : ℝ) (a b c : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (c / Real.sqrt (a^2 + b^2))^2)

/-- The circle equation x^2 + y^2 = r^2 -/
def circle_equation (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

/-- The line equation ax + by + c = 0 -/
def line_equation (x y a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem chord_length_circle_line :
  ∀ (t : ℝ),
  circle_equation (2*t - 1) (t + 1) 3 →
  line_equation (2*t - 1) (t + 1) 1 (-2) 3 →
  chord_length 3 1 (-2) 3 = 12 * Real.sqrt 5 / 5 := by
  sorry

#check chord_length_circle_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l1008_100846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_weight_theorem_l1008_100860

/-- Represents a cylindrical water tank with its properties and contents -/
structure WaterTank where
  height : ℝ
  diameter : ℝ
  capacity : ℝ
  emptyWeight : ℝ
  mixtureRatio : ℝ × ℝ
  fillPercentage : ℝ
  waterWeight : ℝ
  liquidXWeight : ℝ
  cubicFtToGallon : ℝ

/-- Calculates the total weight of the tank with its contents -/
noncomputable def totalWeight (tank : WaterTank) : ℝ :=
  let totalVolume := tank.capacity * tank.fillPercentage
  let waterVolume := totalVolume * (tank.mixtureRatio.2 / (tank.mixtureRatio.1 + tank.mixtureRatio.2))
  let liquidXVolume := totalVolume * (tank.mixtureRatio.1 / (tank.mixtureRatio.1 + tank.mixtureRatio.2))
  let waterWeightTotal := waterVolume * tank.waterWeight
  let liquidXWeightTotal := liquidXVolume * tank.liquidXWeight
  tank.emptyWeight + waterWeightTotal + liquidXWeightTotal

/-- Theorem stating that the total weight of the tank with the given conditions is 1184 pounds -/
theorem tank_weight_theorem (tank : WaterTank)
    (h1 : tank.height = 10)
    (h2 : tank.diameter = 4)
    (h3 : tank.capacity = 200)
    (h4 : tank.emptyWeight = 80)
    (h5 : tank.mixtureRatio = (3, 7))
    (h6 : tank.fillPercentage = 0.6)
    (h7 : tank.waterWeight = 8)
    (h8 : tank.liquidXWeight = 12)
    (h9 : tank.cubicFtToGallon = 7.48) :
    totalWeight tank = 1184 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_weight_theorem_l1008_100860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_opposite_faces_is_eighteen_l1008_100830

/-- Represents a cube with marked points on its faces -/
structure MarkedCube where
  faces : Fin 6 → ℕ
  start_with_five : faces 0 = 5
  sequential_increase : ∀ i : Fin 5, faces (i.succ) = faces i + 1

/-- The maximum number of points on two opposite faces of a marked cube -/
def max_opposite_faces (cube : MarkedCube) : ℕ :=
  max (cube.faces 0 + cube.faces 5) (max (cube.faces 1 + cube.faces 4) (cube.faces 2 + cube.faces 3))

/-- Theorem stating that the maximum number of points on two opposite faces is 18 -/
theorem max_opposite_faces_is_eighteen (cube : MarkedCube) : 
  max_opposite_faces cube = 18 := by
  sorry

#eval max_opposite_faces { 
  faces := fun i => 5 + i,
  start_with_five := rfl,
  sequential_increase := by
    intro i
    simp [Fin.succ]
    rfl
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_opposite_faces_is_eighteen_l1008_100830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_primes_l1008_100811

/-- A function that returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- A function that returns true if n is odd, false otherwise -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- A function that checks if p satisfies the given conditions -/
def satisfiesConditions (p : ℕ) : Prop :=
  ∃ a b : ℤ, (p : ℤ)^2 = a^3 + b^2 ∧ ¬(p ∣ b.natAbs) ∧ Even b

/-- The theorem statement -/
theorem sum_of_smallest_primes :
  ∃ p₁ p₂ : ℕ,
    p₁ < p₂ ∧
    isPrime p₁ ∧
    isPrime p₂ ∧
    isOdd p₁ ∧
    isOdd p₂ ∧
    satisfiesConditions p₁ ∧
    satisfiesConditions p₂ ∧
    (∀ p : ℕ, p < p₁ → ¬(isPrime p ∧ isOdd p ∧ satisfiesConditions p)) ∧
    (∀ p : ℕ, p₁ < p ∧ p < p₂ → ¬(isPrime p ∧ isOdd p ∧ satisfiesConditions p)) ∧
    p₁ + p₂ = 122 :=
  by
    -- Proof goes here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_primes_l1008_100811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1008_100841

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (A + B + C = Real.pi) →
  (Real.cos A = (b * Real.cos C + c * Real.cos B) / (2 * a)) →
  (b = c + 2) →
  (1/2 * b * c * Real.sin A = 15 * Real.sqrt 3 / 4) →
  (A = Real.pi/3 ∧ a = Real.sqrt 19) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1008_100841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_2_value_and_not_integer_l1008_100889

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 2)) + (8 / Real.pi) * Real.arctan (Real.sqrt (abs x))

theorem F_2_value_and_not_integer :
  F 2 = 2 + (8 / Real.pi) * Real.arctan (Real.sqrt 2) ∧ ¬ (∃ n : ℤ, F 2 = n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_2_value_and_not_integer_l1008_100889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l1008_100880

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 5^(abs x)
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

-- State the theorem
theorem function_composition_equality (a : ℝ) :
  f (g a 1) = 1 → a = 1 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l1008_100880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_through_origin_l1008_100835

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 3*m + 3) * x^((m^2 - m - 2)/2)

-- Theorem statement
theorem power_function_not_through_origin (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f m x ≠ 0) → (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_through_origin_l1008_100835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1008_100899

open Real

theorem triangle_problem (A B C a b c : ℝ) (h1 : Real.sqrt 3 * Real.sin A - Real.cos (B + C) = 1)
  (h2 : Real.sin B + Real.sin C = 8/7 * Real.sin A) (h3 : a = 7) :
  A = 2*Real.pi/3 ∧ (1/2 * b * c * Real.sin A = 15*Real.sqrt 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1008_100899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1008_100881

def sequenceDigit (n : ℕ) : ℕ :=
  [2, 9, 4, 7, 3, 6].get! (n % 6)

def sumOfDigits (n : ℕ) : ℕ :=
  (List.range n).map sequenceDigit |>.sum

theorem sequence_properties :
  (sequenceDigit 99 = 7) ∧ (sumOfDigits 100 = 518) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1008_100881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_tour_days_l1008_100894

theorem johns_tour_days : ∃ (days : ℕ) (daily_expense : ℕ),
  days * daily_expense = 800 ∧
  (days + 7) * (daily_expense - 5) = 800 ∧
  days = 28 := by
  -- Define the total budget
  let total_budget : ℕ := 800

  -- Define the extension days
  let extension_days : ℕ := 7

  -- Define the daily expense reduction
  let daily_reduction : ℕ := 5

  -- Define the function for the original tour cost
  let original_cost (days : ℕ) (daily_expense : ℕ) : ℕ :=
    days * daily_expense

  -- Define the function for the extended tour cost
  let extended_cost (days : ℕ) (daily_expense : ℕ) : ℕ :=
    (days + extension_days) * (daily_expense - daily_reduction)

  -- Provide the solution
  use 28, 800 / 28

  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_tour_days_l1008_100894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1008_100876

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ d * x + e * y + f = 0) → a * e = b * d

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + m * y + 6 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (m - 2) * x + 3 * y + 2 * m = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1008_100876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beans_l1008_100808

/-- Represents the amount of beans in pounds -/
def beans : ℝ → Prop := sorry

/-- Represents the amount of rice in pounds -/
def rice : ℝ → Prop := sorry

/-- The amount of rice is at least 8 pounds more than a third of the amount of beans -/
axiom rice_lower_bound : ∀ b r, beans b → rice r → r ≥ 8 + b / 3

/-- The amount of rice is no more than three times the amount of beans -/
axiom rice_upper_bound : ∀ b r, beans b → rice r → r ≤ 3 * b

/-- The minimum amount of beans that satisfies the conditions is 3 pounds -/
theorem min_beans : ∀ b, beans b → b ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beans_l1008_100808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1008_100892

-- Pair B
noncomputable def f_B (x : ℝ) : ℝ := if x ≠ 0 then |x| / x else 0

noncomputable def g_B (x : ℝ) : ℝ := if x > 0 then 1 else if x < 0 then -1 else 0

-- Pair C
def f_C (x : ℝ) : ℝ := 2 * x^2 + 1
def g_C (t : ℝ) : ℝ := 2 * t^2 + 1

-- Pair D
def f_D (x : ℝ) : ℝ := x
noncomputable def g_D (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem function_equality :
  (∀ x : ℝ, x ≠ 0 → f_B x = g_B x) ∧
  (∀ x : ℝ, f_C x = g_C x) ∧
  (∀ x : ℝ, f_D x = g_D x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1008_100892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1008_100831

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_quadrilateral (q : Quadrilateral) : Prop :=
  let (wx, wy) := q.W
  let (xx, xy) := q.X
  let (yx, yy) := q.Y
  let (zx, zy) := q.Z
  (wx - xx) * (wy - zy) + (wy - yy) * (zx - xx) = 0 ∧ 
  (yx - xx) * (yy - zy) + (yy - wy) * (zx - yx) = 0

def diagonal_perpendicular_to_side (q : Quadrilateral) : Prop :=
  let (wx, wy) := q.W
  let (xx, xy) := q.X
  let (yx, yy) := q.Y
  let (zx, zy) := q.Z
  (wx - yx) * (xx - zx) + (wy - yy) * (xy - zy) = 0

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  side_length q.W q.X + side_length q.X q.Z + side_length q.Z q.Y + side_length q.Y q.W

-- Theorem statement
theorem quadrilateral_perimeter (q : Quadrilateral) :
  is_right_angled_quadrilateral q →
  diagonal_perpendicular_to_side q →
  side_length q.W q.Z = 15 →
  side_length q.W q.X = 20 →
  side_length q.X q.Z = 9 →
  perimeter q = 44 + Real.sqrt 706 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1008_100831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1008_100848

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1008_100848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_has_21_sides_l1008_100895

/-- The number of sides in a convex polygon -/
def num_sides : ℕ := 21

/-- The sum of all angles except one in the polygon -/
def sum_angles_except_one : ℝ := 3330

/-- Axiom: The sum of all angles except one is 3330° -/
axiom sum_angles_except_one_val : sum_angles_except_one = 3330

/-- Theorem: The polygon has 21 sides -/
theorem polygon_has_21_sides : num_sides = 21 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_has_21_sides_l1008_100895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l1008_100855

theorem sum_remainder_mod_30 (x y z : ℕ) 
  (hx : x % 30 = 14)
  (hy : y % 30 = 5)
  (hz : z % 30 = 21) :
  (x + y + z) % 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l1008_100855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_start_time_is_three_l1008_100871

/-- Represents the scenario of a police chase --/
structure PoliceChase where
  thief_speed : ℝ
  police_initial_distance : ℝ
  police_speed : ℝ
  catch_time : ℝ

/-- The time after which the police officer started chasing the thief --/
noncomputable def chase_start_time (chase : PoliceChase) : ℝ :=
  (chase.police_initial_distance - chase.catch_time * (chase.police_speed - chase.thief_speed)) / chase.thief_speed

/-- Theorem stating that the chase start time is 3 hours for the given scenario --/
theorem chase_start_time_is_three (chase : PoliceChase) 
  (h1 : chase.thief_speed = 20)
  (h2 : chase.police_initial_distance = 60)
  (h3 : chase.police_speed = 40)
  (h4 : chase.catch_time = 4) :
  chase_start_time chase = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_start_time_is_three_l1008_100871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_l1008_100816

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 2*x - 4*y + 8

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 4)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_to_center :
  distance circle_center given_point = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_l1008_100816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sets_satisfying_conditions_l1008_100828

theorem count_sets_satisfying_conditions : 
  let S : Set (Finset ℤ) := {A | A ∩ ({-1, 0, 1} : Finset ℤ) = ({0, 1} : Finset ℤ) ∧ 
                                A ∪ ({-2, 0, 2} : Finset ℤ) = ({-2, 0, 1, 2} : Finset ℤ)}
  Finset.card (Finset.filter (fun A => A ∈ S) (Finset.powerset {-2, -1, 0, 1, 2})) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sets_satisfying_conditions_l1008_100828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_probability_l1008_100845

/-- The number of ways to deliver 5 packages to 5 houses -/
def total_deliveries : ℕ := 120  -- 5! = 120

/-- The number of ways to choose 2 packages out of 5 -/
def choose_two_correct : ℕ := 10  -- 5 choose 2 = 10

/-- The number of derangements of 3 items -/
def derangements_three : ℕ := 2  -- 3! * (1 - 1 + 1/2 - 1/6) = 2

/-- The probability of exactly 2 packages being delivered correctly -/
def prob_two_correct : ℚ := (choose_two_correct * derangements_three : ℚ) / total_deliveries

theorem correct_probability : prob_two_correct = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_probability_l1008_100845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_4_l1008_100866

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_4 :
  ∀ r : ℝ,
  (geometricSum 1 r 1 = 1) →
  (geometricSum 1 r 3 = 3/4) →
  (geometricSum 1 r 4 = 5/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_4_l1008_100866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l1008_100822

noncomputable section

-- Define the circle C
def circle_C (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

-- Define the polar equation
def polar_equation (θ : ℝ) : ℝ :=
  2 * Real.cos θ

-- Define the sum of distances for points P and Q
def sum_distances (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 2 * Real.cos (θ + Real.pi/3)

theorem circle_C_properties :
  -- Part I: Prove that the polar equation of C is ρ = 2cos(θ)
  (∀ θ, ∃ φ, circle_C φ = (polar_equation θ * Real.cos θ, polar_equation θ * Real.sin θ)) ∧
  -- Part II: Prove that the maximum value of |OP| + |OQ| is 2√3
  (∃ θ, sum_distances θ = 2 * Real.sqrt 3 ∧
        ∀ θ', sum_distances θ' ≤ 2 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l1008_100822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_eight_points_no_square_l1008_100850

/-- A simple graph with no self-loops or multiple edges -/
structure SimpleGraph' (V : Type*) where
  adj : V → V → Prop
  symm : ∀ {u v}, adj u v → adj v u
  irrefl : ∀ v, ¬adj v v

/-- A square in a graph is a set of 4 distinct points A, B, C, D where A and C are both connected to B and D -/
def HasSquare {V : Type*} (G : SimpleGraph' V) : Prop :=
  ∃ (A B C D : V), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    G.adj A B ∧ G.adj A D ∧ G.adj C B ∧ G.adj C D

/-- The number of edges in a graph -/
def EdgeCount {V : Type*} [Fintype V] (G : SimpleGraph' V) : ℕ :=
  ((Fintype.card V * (Fintype.card V - 1)) / 2)

/-- The main theorem stating that a graph with 8 points and no squares has at most 11 edges -/
theorem max_edges_eight_points_no_square {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph' V) 
    (h_card : Fintype.card V = 8) (h_no_square : ¬HasSquare G) : 
    EdgeCount G ≤ 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_eight_points_no_square_l1008_100850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_g_is_shifted_f_g_is_odd_l1008_100883

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Statement 1: f is decreasing on [π/3, π/2]
theorem f_decreasing : 
  ∀ x y, π/3 ≤ x ∧ x < y ∧ y ≤ π/2 → f y < f x := by
  sorry

-- Statement 2: g is obtained by shifting f to the left by π/12
theorem g_is_shifted_f : 
  ∀ x, g x = f (x + π/12) := by
  sorry

-- Statement 3: g is an odd function
theorem g_is_odd : 
  ∀ x, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_g_is_shifted_f_g_is_odd_l1008_100883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_car_efficiency_l1008_100869

-- Define the total distance Bob's car can travel
noncomputable def total_distance : ℝ := 100

-- Define the total amount of gas used
noncomputable def total_gas : ℝ := 10

-- Define the fuel efficiency of Bob's car
noncomputable def fuel_efficiency : ℝ := total_distance / total_gas

-- Theorem statement
theorem bobs_car_efficiency :
  fuel_efficiency = 10 := by
  -- Unfold the definitions
  unfold fuel_efficiency total_distance total_gas
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_car_efficiency_l1008_100869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_distance_properties_l1008_100824

-- Define the symmetry operation with respect to Ozx plane
def symmetry_Ozx (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

-- Define the symmetry operation with respect to y-axis
def symmetry_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2.1, p.2.2)

-- Define the distance from a point to Oyz plane
def distance_to_Oyz (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.1|

-- Define a vector in ℝ³
def vector (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

theorem symmetry_and_distance_properties :
  (symmetry_Ozx (1, -2, 3) = (1, 2, 3)) ∧
  (symmetry_y_axis (1/2, 1, -3) = (-1/2, 1, -3)) ∧
  (distance_to_Oyz (2, -1, 3) = 2) ∧
  (vector 3 (-2) 4 = (3, -2, 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_distance_properties_l1008_100824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_pq_l1008_100805

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  c : ℝ
  a : ℝ
  b : ℝ
  h : c > 0
  h1 : a^2 + b^2 = c^2
  h2 : b^2 = 6
  h3 : c = 3 * a^2 / c

/-- Theorem stating the equation of the hyperbola and the equation of line PQ -/
theorem hyperbola_and_line_pq (H : Hyperbola) :
  (∃ (x y : ℝ), x^2 / 3 - y^2 / 6 = 1) ∧
  (∃ (k : ℝ), k = Real.sqrt 2 / 2 ∨ k = -(Real.sqrt 2 / 2)) ∧
  (∃ (x y : ℝ), x - Real.sqrt 2 * y - 3 = 0 ∨ x + Real.sqrt 2 * y - 3 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_pq_l1008_100805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_terms_count_l1008_100875

-- Define the binomial expansion term
def binomial_term (r : ℕ) (x y : ℝ) : ℝ :=
  (Nat.choose 16 r) * (-1)^r * y^(16 - (3/2)*r) * x^((3/2)*r - 8)

-- Define when a term is integral
def is_integral_term (r : ℕ) : Prop :=
  ∃ (m n : ℤ), (16 - (3/2)*r : ℚ) = m ∧ ((3/2)*r - 8 : ℚ) = n

-- Theorem statement
theorem integral_terms_count :
  (∃ (S : Finset ℕ), S.card = 3 ∧ (∀ r ∈ S, is_integral_term r) ∧
   (∀ r ∉ S, ¬is_integral_term r)) :=
by
  sorry

#check integral_terms_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_terms_count_l1008_100875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_and_helicopter_speed_l1008_100882

/-- The speed of a goods train and helicopter given specific conditions --/
theorem train_and_helicopter_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (downdraft_increase : ℝ)
  (ε : ℝ) -- epsilon for approximation
  (hε : ε > 0) :
  let relative_speed := goods_train_length / passing_time
  let v := (relative_speed * 3600 / 1000 - man_train_speed) / (1 + downdraft_increase)
  let goods_train_speed := v
  let goods_train_speed_with_downdraft := v * (1 + downdraft_increase)
  let helicopter_speed := goods_train_speed_with_downdraft
  man_train_speed = 55 →
  goods_train_length = 320 →
  passing_time = 10 →
  downdraft_increase = 0.1 →
  (abs (goods_train_speed - 54.73) < ε ∧
   abs (goods_train_speed_with_downdraft - 60.203) < ε ∧
   abs (helicopter_speed - 60.203) < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_and_helicopter_speed_l1008_100882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_assignment_count_l1008_100849

/-- The number of ways to assign 6 friends to 6 rooms with at most 2 friends per room -/
def hotel_assignments : ℕ := 49320

/-- The number of rooms in the hotel -/
def num_rooms : ℕ := 6

/-- The number of friends staying at the hotel -/
def num_friends : ℕ := 6

/-- The maximum number of friends allowed in any room -/
def max_per_room : ℕ := 2

/-- Theorem stating that the number of ways to assign 6 friends to 6 rooms,
    with no more than 2 friends per room, is equal to 49320 -/
theorem hotel_assignment_count :
  (Fintype.card {assignment : Fin num_friends → Fin num_rooms |
    ∀ room, (Finset.filter (λ friend => assignment friend = room) Finset.univ).card ≤ max_per_room}) = hotel_assignments :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_assignment_count_l1008_100849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_exists_l1008_100874

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
def UnitSquare : Set Point :=
  { p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 }

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

/-- Three points are collinear if the area of the triangle they form is zero -/
def collinear (p1 p2 p3 : Point) : Prop :=
  triangleArea p1 p2 p3 = 0

theorem small_triangle_exists 
  (points : Finset Point) 
  (h1 : points.card = 101) 
  (h2 : ∀ p, p ∈ points → p ∈ UnitSquare) 
  (h3 : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
       p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) :
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_exists_l1008_100874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernice_third_oldest_l1008_100852

-- Define the type for students
inductive Student : Type
| Adyant : Student
| Bernice : Student
| Cici : Student
| Dara : Student
| Ellis : Student

-- Define the age relation
def olderThan : Student → Student → Prop := sorry

-- Define the conditions
axiom adyant_older_bernice : olderThan Student.Adyant Student.Bernice
axiom dara_youngest : ∀ s : Student, s ≠ Student.Dara → olderThan s Student.Dara
axiom bernice_older_ellis : olderThan Student.Bernice Student.Ellis
axiom bernice_younger_cici : olderThan Student.Cici Student.Bernice
axiom cici_not_oldest : ∃ s : Student, olderThan s Student.Cici

-- Define the property of being the third oldest
def isThirdOldest (s : Student) : Prop :=
  ∃ (a b : Student), a ≠ b ∧ a ≠ s ∧ b ≠ s ∧
    olderThan a s ∧ olderThan b s ∧
    (∀ c : Student, c ≠ a ∧ c ≠ b ∧ c ≠ s → olderThan s c)

-- Theorem to prove
theorem bernice_third_oldest : isThirdOldest Student.Bernice := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernice_third_oldest_l1008_100852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_area_difference_l1008_100857

/-- The difference in area between a circle with diameter 10 and a square with diagonal 10 -/
theorem circle_square_area_difference : 
  let square_diagonal : ℝ := 10
  let circle_diameter : ℝ := 10
  let square_area : ℝ := (square_diagonal^2) / 2
  let circle_area : ℝ := π * (circle_diameter / 2)^2
  ‖(circle_area - square_area) - 28.5‖ < 0.05 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_area_difference_l1008_100857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_p_15_l1008_100829

theorem min_abs_p_15 (a b c d : ℤ) (p : ℝ → ℝ) :
  p = (fun x ↦ (a : ℝ) * x^3 + (b : ℝ) * x^2 + (c : ℝ) * x + (d : ℝ)) →
  p 5 + p 25 = 1906 →
  ∃ (m : ℤ), (∀ (a' b' c' d' : ℤ) (q : ℝ → ℝ),
    q = (fun x ↦ (a' : ℝ) * x^3 + (b' : ℝ) * x^2 + (c' : ℝ) * x + (d' : ℝ)) →
    q 5 + q 25 = 1906 →
    m ≤ |q 15|) ∧
  m = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_p_15_l1008_100829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l1008_100818

/-- The function f(x) = (6x^2 - 11) / (4x^2 + 6x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3)

/-- p and q are the x-coordinates of the vertical asymptotes of f -/
def is_vertical_asymptote (x : ℝ) : Prop :=
  (4 * x^2 + 6 * x + 3 = 0) ∧ (∀ y : ℝ, f x ≠ y)

/-- Theorem: The sum of the x-coordinates of the vertical asymptotes of f is -2 -/
theorem vertical_asymptotes_sum :
  ∃ (p q : ℝ), is_vertical_asymptote p ∧ is_vertical_asymptote q ∧ p + q = -2 := by
  sorry

#check vertical_asymptotes_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l1008_100818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_complement_l1008_100872

universe u

open Set

def symmetricDifference {U : Type u} (A B : Set U) : Set U :=
  (A ∪ B) \ (A ∩ B)

infix:70 " ∆ " => symmetricDifference

theorem symmetric_difference_complement {U : Type u} (A B : Set U) :
  (A ∆ B) = (Aᶜ ∆ Bᶜ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_complement_l1008_100872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l1008_100825

theorem cos_2alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 4) = 3 / 5) :
  Real.cos (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l1008_100825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_slope_l1008_100854

-- Define the circle
def circleEquation (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define a line passing through (2, 0) with slope k
def lineEquation (x y k : ℝ) : Prop := y = k * (x - 2)

-- Define the distance from origin to a line y = kx + b
noncomputable def distanceToLine (k b : ℝ) : ℝ := |b| / Real.sqrt (k^2 + 1)

-- Define the condition for maximum area (isosceles right triangle)
def maxAreaCondition (k : ℝ) : Prop := distanceToLine k (-2*k) = 1

-- State the theorem
theorem max_area_slope :
  ∃ k : ℝ, maxAreaCondition k ∧ (k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_slope_l1008_100854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1008_100878

theorem angle_values (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi)
  (h2 : 0 < β ∧ β < Real.pi)
  (h3 : Real.cos (Real.pi/2 - α) = Real.sqrt 2 * Real.cos (3*Real.pi/2 + β))
  (h4 : Real.sqrt 3 * Real.sin (3*Real.pi/2 - α) = -Real.sqrt 2 * Real.sin (Real.pi/2 + β)) :
  (α = Real.pi/4 ∧ β = Real.pi/6) ∨ (α = 3*Real.pi/4 ∧ β = 5*Real.pi/6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1008_100878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1008_100862

def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![1, 1]

theorem perpendicular_vectors (lambda : ℝ) :
  (∀ i : Fin 2, (a + lambda • b) i = a i + lambda * b i) →
  (∀ i : Fin 2, ((a + lambda • b) i) * (b i) = 0) →
  lambda = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1008_100862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_area_l1008_100847

/-- Given a triangle ABC with vertices A(1, 10), B(3, 0), C(10, 0), 
    and a horizontal line y=t intersecting AB at T and AC at U, 
    prove that if the area of triangle ATU is 15, then t ≈ 3.22. -/
theorem triangle_intersection_area (t : ℝ) : 
  let A : ℝ × ℝ := (1, 10)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (10, 0)
  let T : ℝ × ℝ := ((15 - t) / 5, t)
  let U : ℝ × ℝ := (10 - 9*t/10, t)
  let area_ATU := (1/2) * (U.1 - T.1) * (10 - t)
  area_ATU = 15 → abs (t - 3.22) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_area_l1008_100847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_equal_intercept_problem_l1008_100853

/-- Circle C with equation x^2 + y^2 = 4 -/
def Circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Point M with coordinates (1, m) -/
def Point_M (m : ℝ) : ℝ × ℝ := (1, m)

/-- Tangent line to circle C passing through point (x, y) -/
def tangent_line (x y : ℝ) : Set (ℝ × ℝ) := {p | x * p.1 + y * p.2 = 4}

/-- Line with equal intercepts passing through (1, m) -/
def equal_intercept_line (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1 + m}

/-- Theorem for the first part of the problem -/
theorem tangent_line_problem (m : ℝ) :
  (Point_M m ∈ Circle_C ∧ 
   ∃! l : Set (ℝ × ℝ), l = tangent_line 1 m ∧ Point_M m ∈ l) →
  (m = Real.sqrt 3 ∨ m = -Real.sqrt 3) ∧
  (tangent_line 1 m = {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 = 4} ∨
   tangent_line 1 m = {p : ℝ × ℝ | p.1 - Real.sqrt 3 * p.2 = 4}) :=
by sorry

/-- Theorem for the second part of the problem -/
theorem equal_intercept_problem (m : ℝ) :
  (∃ l : Set (ℝ × ℝ), l = equal_intercept_line m ∧
   Point_M m ∈ l ∧
   ∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ Circle_C ∧ q ∈ Circle_C ∧
   Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 3) →
  m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_equal_intercept_problem_l1008_100853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_20_consecutive_even_integers_sum_6400_l1008_100896

def consecutive_even_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => start + 2 * i)

theorem largest_of_20_consecutive_even_integers_sum_6400 :
  ∃ start : ℤ,
    let sequence := consecutive_even_integers start 20
    (sequence.sum = 6400) ∧
    (sequence.getLast? = some 339) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_20_consecutive_even_integers_sum_6400_l1008_100896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1008_100891

theorem calculation_proof : 
  ((-8 : ℝ) ^ (1/3)) * (-1)^2023 - 6 / 2 + (1/2)^0 = 0 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1008_100891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_div_n_l1008_100801

/-- The sequence a_n defined by the given conditions -/
def a : ℕ → ℚ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => 15
  | (n + 2) => a (n + 1) + 2 * (n + 1)

/-- The theorem stating the minimum value of a_n / n -/
theorem min_value_a_div_n :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n > 0 → a n / n ≥ 27 / 4 ∧ a k / k = 27 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_div_n_l1008_100801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l1008_100868

theorem money_distribution (gil loki moe nick : ℚ) :
  gil > 0 ∧ loki > 0 ∧ moe > 0 ∧ nick > 0 →
  (1/4) * gil = (1/3) * loki ∧
  (1/3) * loki = (1/6) * moe ∧
  (1/6) * moe = (1/2) * nick →
  ((1/4) * gil + (1/3) * loki + (1/6) * moe + (1/2) * nick) / (gil + loki + moe + nick) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l1008_100868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_sum_l1008_100821

theorem quadratic_root_difference_sum (m n : ℤ) : 
  (∃ (r₁ r₂ : ℝ), r₁ > r₂ ∧ 2 * r₁^2 - 5 * r₁ - 12 = 0 ∧ 2 * r₂^2 - 5 * r₂ - 12 = 0 ∧ r₁ - r₂ = Real.sqrt (m.toNat) / n.toNat) →
  (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ m.toNat)) →
  m + n = 123 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_sum_l1008_100821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_k_l1008_100844

theorem unit_digit_of_k (k : ℤ) (a : ℝ) :
  k > 1 →
  a^2 - k*a + 1 = 0 →
  (∀ n : ℕ, n > 10 → Int.floor (a^(2^n) + a^(-(2^n : ℤ))) % 10 = 7) →
  k % 10 = 3 ∨ k % 10 = 5 ∨ k % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_k_l1008_100844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1008_100819

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * (1 - seq.commonRatio^n) / (1 - seq.commonRatio)

theorem geometric_sequence_sum (seq : GeometricSequence) :
  sumGeometric seq 6033 = 600 →
  sumGeometric seq 12066 = 1140 →
  sumGeometric seq 18099 = 1626 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1008_100819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_f_h_l1008_100820

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The function h(x) = √(x-2) - 2 for x ≥ 2 -/
noncomputable def h (x : ℝ) : ℝ := Real.sqrt (x - 2) - 2

/-- The distance between two points (x₁, y₁) and (x₂, y₂) in ℝ² -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_f_h :
  ∃ (d : ℝ), d = 7 * Real.sqrt 2 / 4 ∧
  ∀ (x₁ x₂ : ℝ), x₁ ≥ 0 → x₂ ≥ 2 →
    distance x₁ (f x₁) x₂ (h x₂) ≥ d ∧
    ∃ (x₁' x₂' : ℝ), x₁' ≥ 0 ∧ x₂' ≥ 2 ∧
      distance x₁' (f x₁') x₂' (h x₂') = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_f_h_l1008_100820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_specific_triangle_l1008_100865

/-- The distance between the centers of the incircle and excircle of a triangle -/
noncomputable def distance_between_centers (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  let R := (a * b * c) / (4 * area)
  2 * R - r

theorem distance_centers_specific_triangle :
  distance_between_centers 12 15 17 = 1235 / 11 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_specific_triangle_l1008_100865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_constraint_l1008_100839

noncomputable def f (B C : ℤ) (x : ℝ) : ℝ := (x^2 + 1) / (3 * x^2 + B * x + C)

theorem function_constraint (B C : ℤ) :
  (∀ x > -1, f B C x > 0.5) → B = 6 ∧ C = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_constraint_l1008_100839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_sphere_radius_is_16_l1008_100826

/-- The radius of a new sphere formed from a sphere with a cylindrical hole -/
noncomputable def new_sphere_radius (R : ℝ) (r : ℝ) : ℝ :=
  let original_volume := (4 / 3) * Real.pi * R^3
  let cylinder_volume := Real.pi * r^2 * (2 * R)
  let cap_volume := (2 * Real.pi * 4^2 * (3 * R - 4)) / 3
  let remaining_volume := original_volume - cylinder_volume - cap_volume
  (3 * remaining_volume / (4 * Real.pi)) ^ (1/3)

/-- Theorem stating that the radius of the new sphere is 16 -/
theorem new_sphere_radius_is_16 :
  new_sphere_radius 20 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_sphere_radius_is_16_l1008_100826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_to_focus_l1008_100809

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → ℝ := λ _ => -2

-- Define a point P on the parabola
def P : ℝ × ℝ := sorry

-- Define the angle of inclination of PF
noncomputable def angle_PF : ℝ := 2*(Real.pi)/3  -- 120° in radians

-- Theorem statement
theorem parabola_distance_to_focus :
  parabola P.1 P.2 →
  (P.2 - directrix P.1)^2 = (P.1 - focus.1)^2 + (P.2 - focus.2)^2 →
  angle_PF = Real.arctan ((P.2 - focus.2) / (P.1 - focus.1)) →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_to_focus_l1008_100809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_y_l1008_100863

-- Define the ⊘ operator
noncomputable def circleSlash (a b : ℝ) : ℝ := (Real.sqrt (a^2 + 2*b))^3

-- State the theorem
theorem solve_y (y : ℝ) : circleSlash 3 y = 64 → y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_y_l1008_100863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1008_100886

/-- Time taken for two trains to cross each other -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * (5/18))

/-- Theorem: The time taken for the trains to cross each other is approximately 10.44 seconds -/
theorem trains_crossing_time :
  let length1 : ℝ := 140  -- length of first train in meters
  let length2 : ℝ := 150  -- length of second train in meters
  let speed1 : ℝ := 60    -- speed of first train in km/hr
  let speed2 : ℝ := 40    -- speed of second train in km/hr
  abs (time_to_cross length1 length2 speed1 speed2 - 10.44) < 0.01
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1008_100886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_students_count_l1008_100833

/-- Represents the change in student population over 5 years --/
def student_population (X : ℝ) : ℝ :=
  let year2 := X * 1.20
  let year3 := year2 * 0.90
  let year4 := year3 * 1.15
  let year5 := year4 * 0.95
  year5 + 150 - 50

/-- Theorem stating that the initial number of students is approximately 746 --/
theorem initial_students_count : 
  ∃ X : ℝ, student_population X = 980 ∧ abs (X - 746) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_students_count_l1008_100833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tangent_ratio_l1008_100859

/-- In an acute triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if b/a + a/b = 6cos(C), then tan(C)/tan(A) + tan(C)/tan(B) = 4 -/
theorem acute_triangle_tangent_ratio 
  (a b c : ℝ) (A B C : ℝ) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_opposite : Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c)
  (h_condition : b / a + a / b = 6 * Real.cos C) :
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tangent_ratio_l1008_100859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_six_l1008_100867

-- Define the pyramid
structure Pyramid where
  base_leg : ℝ
  height : ℝ

-- Define the volume function for the pyramid
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  (1 / 3) * (1 / 2) * p.base_leg * p.base_leg * p.height

-- Theorem statement
theorem pyramid_volume_is_six :
  ∃ (p : Pyramid), p.base_leg = 3 ∧ p.height = 4 ∧ pyramid_volume p = 6 :=
by
  -- Construct the pyramid
  let p : Pyramid := ⟨3, 4⟩
  
  -- Show that this pyramid satisfies the conditions
  have h1 : p.base_leg = 3 := rfl
  have h2 : p.height = 4 := rfl
  
  -- Calculate the volume
  have h3 : pyramid_volume p = 6 := by
    unfold pyramid_volume
    simp [h1, h2]
    norm_num
  
  -- Prove the existence
  exact ⟨p, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_six_l1008_100867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l1008_100897

def b : ℕ → ℚ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 2
  | n + 3 => (1/4) * b (n + 2) + (2/5) * b (n + 1)

theorem sum_of_sequence_b : ∑' n, b n = 85/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l1008_100897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_rational_equation_l1008_100836

theorem unique_solution_for_rational_equation :
  ∃! ℓ : ℚ, ∃! x : ℚ, (x + 3) / (ℓ * x + 2) = x ∧ ℓ * x + 2 ≠ 0 :=
by
  -- We claim that ℓ = -1/12 and x = 6 satisfy the conditions
  use (-1/12 : ℚ)
  constructor
  · -- First, prove existence of a unique x for ℓ = -1/12
    use (6 : ℚ)
    constructor
    · -- Show that x = 6 satisfies the equation
      simp [add_div]
      -- The rest of the proof is omitted
      sorry
    · -- Show uniqueness of x
      sorry
  · -- Then, prove uniqueness of ℓ
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_rational_equation_l1008_100836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1008_100803

theorem right_triangle_hypotenuse (P Q R : ℝ × ℝ) (PQ PR : ℝ) :
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →  -- right angle at P
  ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 : ℝ) = PQ^2 →  -- PQ length
  ((R.1 - P.1)^2 + (R.2 - P.2)^2 : ℝ) = PR^2 →  -- PR length
  (Q.1 - P.1) / Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) = 4/5 →  -- cos Q
  PQ = 12 →
  PR = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1008_100803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1008_100884

/-- Time for a train to pass a person moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 385 →
  train_speed = 60 * (1000 / 3600) →
  person_speed = 6 * (1000 / 3600) →
  let relative_speed := train_speed + person_speed
  ∃ t : ℝ, (t ≥ 20 ∧ t ≤ 22) ∧ t * relative_speed = train_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1008_100884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_a_in_range_l1008_100817

noncomputable section

-- Define the function f(x) piecewise
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then (1 - a) * x + 1 else x + 4 / x - 4

-- Define the property of f having a minimum value
def has_minimum (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), ∀ (x : ℝ), f m ≤ f x

-- State the theorem
theorem f_has_minimum_iff_a_in_range (a : ℝ) :
  has_minimum (f a) ↔ 1 ≤ a ∧ a ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_a_in_range_l1008_100817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1008_100812

open Real

theorem trigonometric_equation_solution :
  ∀ t : ℝ,
  (Real.cos (2 * t - 18 * π / 180) * Real.tan (50 * π / 180) + Real.sin (2 * t - 18 * π / 180) = 1 / (2 * Real.cos (130 * π / 180))) ↔
  (∃ k : ℤ, t = (-31 * π / 180 + k * π) ∨ t = (89 * π / 180 + k * π)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1008_100812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_8pi_l1008_100842

/-- Piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-2) 0 then Real.sqrt (4 - x^2)
  else if x ∈ Set.Icc 0 2 then 2 - x
  else 0

/-- Volume of the solid of revolution -/
noncomputable def volume_of_revolution (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.pi * ∫ x in a..b, (f x)^2

/-- Theorem stating that the volume of the solid generated by rotating y=f(x) about the x-axis is 8π -/
theorem volume_equals_8pi :
  volume_of_revolution f (-2) 2 = 8 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_8pi_l1008_100842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_ratio_l1008_100856

theorem rice_mixture_ratio (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  (16 * x + 24 * y + 30 * z) / (x + y + z) = 18 →
  x = 9 * y ∧ x = 9 * z := by
  sorry

#check rice_mixture_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_ratio_l1008_100856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_station_probability_l1008_100813

/-- The probability of Alex arriving while the train is at the station -/
theorem train_station_probability : ∃ (probability : ℝ), probability = 1/6 := by
  -- Define constants
  let train_wait_time : ℕ := 15
  let alex_arrival_range : ℕ := 90
  let train_arrival_range : ℕ := 60

  -- Calculate probability
  let probability : ℝ := (train_arrival_range * train_wait_time : ℕ) / (alex_arrival_range * train_arrival_range : ℕ)

  -- Prove the theorem
  use probability
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_station_probability_l1008_100813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_and_rounding_l1008_100823

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The result of subtracting 45.239 from 96.865 and rounding to the nearest tenth is 51.6 -/
theorem subtraction_and_rounding :
  round_to_tenth (96.865 - 45.239) = 51.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_and_rounding_l1008_100823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_minimum_l1008_100898

/-- The perimeter of a circular sector as a function of its radius, given a fixed area of 100 -/
noncomputable def sectorPerimeter (r : ℝ) : ℝ := 2 * r + 200 / r

/-- Theorem stating that the perimeter of a circular sector with area 100 is minimized when the radius is 10 -/
theorem sector_perimeter_minimum : 
  ∀ r : ℝ, r > 0 → sectorPerimeter r ≥ sectorPerimeter 10 := by
  sorry

#check sector_perimeter_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_minimum_l1008_100898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1008_100873

noncomputable section

variable (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 2*a*(Real.log x)

theorem f_extrema_and_monotonicity (h1 : a > 0) (h2 : x > 0) :
  (a = 2 → ∃ (max min : ℝ),
    IsLocalMax (f a) 1 ∧ f a 1 = max ∧ max = -5 ∧
    IsLocalMin (f a) 2 ∧ f a 2 = min ∧ min = 4 * Real.log 2 - 8) ∧
  (∃ (x1 x2 : ℝ), x1 = a ∧ x2 = 1 ∧
    ∀ y, y ≠ x1 ∧ y ≠ x2 → (deriv (f a)) y ≠ 0) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1008_100873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1008_100810

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the sides a, b, c
def side_lengths (a b c : ℝ) (A B C : ℝ) : Prop :=
  triangle_ABC A B C ∧ 
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C

-- Define the area of the triangle
def area (S a b c : ℝ) (C : ℝ) : Prop :=
  S = (1/2) * a * b * Real.sin C

theorem triangle_ratio (A B C a b c S : ℝ) :
  triangle_ABC A B C →
  side_lengths a b c A B C →
  area S a b c C →
  A = Real.pi/3 →
  b = 1 →
  S = Real.sqrt 3 →
  c / Real.sin C = 2 * Real.sqrt 39 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1008_100810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_lens_total_cost_l1008_100837

/-- Calculate the total cost of a new camera and discounted lens, including sales tax -/
theorem camera_lens_total_cost
  (old_camera_cost : ℝ)
  (new_camera_increase_rate : ℝ)
  (lens_original_price : ℝ)
  (lens_discount : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : old_camera_cost = 4000)
  (h2 : new_camera_increase_rate = 0.3)
  (h3 : lens_original_price = 400)
  (h4 : lens_discount = 200)
  (h5 : sales_tax_rate = 0.08) :
  (old_camera_cost * (1 + new_camera_increase_rate) + 
   (lens_original_price - lens_discount)) * 
  (1 + sales_tax_rate) = 5832 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_lens_total_cost_l1008_100837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_of_g_l1008_100861

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x + 2 * Real.tan x + 5 / Real.tan x

theorem smallest_positive_root_of_g :
  ∃ s : ℝ,
    s > 0 ∧
    g s = 0 ∧
    (∀ x > 0, g x = 0 → x ≥ s) ∧
    Real.pi < s ∧ s < 3 * Real.pi / 2 ∧
    Int.floor s = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_of_g_l1008_100861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_displacement_l1008_100843

-- Define the velocity function
def v (t : ℝ) : ℝ := 27 - 0.9 * t

-- Define the time when the train stops
def stop_time : ℝ := 30

-- Theorem statement
theorem train_displacement :
  ∫ t in (0 : ℝ)..(stop_time), v t = 405 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_displacement_l1008_100843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_l1008_100893

theorem systematic_sampling (total sample_size first_drawn range_start range_end : Nat) : 
  total = 800 → 
  sample_size = 50 → 
  first_drawn = 7 → 
  range_start = 33 → 
  range_end = 48 → 
  ∃ (interval : Nat), 
    interval = total / sample_size ∧ 
    ∃ (group : Nat), 
      group * interval < range_start ∧ 
      (group + 1) * interval ≥ range_end ∧
      first_drawn + group * interval = 39 := by
  sorry

#check systematic_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_l1008_100893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1008_100840

/-- Expansion operation as defined in the problem -/
noncomputable def expand (a b : ℝ) : ℝ := a * b + a + b

/-- Perform one operation by expanding the two larger numbers -/
noncomputable def operate (a b : ℝ) : ℝ :=
  let c := expand a b
  if c ≥ a ∧ a ≥ b then expand c a
  else if c ≥ b ∧ b ≥ a then expand c b
  else expand a b

/-- Perform n operations -/
noncomputable def operate_n (a b : ℝ) : ℕ → ℝ
  | 0 => max a b
  | n + 1 => operate (operate_n a b n) (max a b)

theorem part1 : operate_n 1 3 3 = 255 := by sorry

theorem part2 {p q : ℝ} (h : p > q ∧ q > 0) :
  ∃ (m n : ℕ), operate_n p q 6 = (q + 1) ^ m * (p + 1) ^ n - 1 ∧ m = 8 ∧ n = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1008_100840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_form_l1008_100827

/-- A function f: ℝ → ℝ satisfying the given conditions has the form f(x) = x + c for some constant c. -/
theorem function_form (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y → f x < f y)  -- f is strictly increasing
  (h2 : ∃ g : ℝ → ℝ, (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x))  -- g is the composition inverse of f
  (h3 : ∀ x, f x + (Classical.choose h2) x = 2 * x)  -- functional equation
  : ∃ c : ℝ, ∀ x, f x = x + c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_form_l1008_100827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_x_plus_5_range_of_x_for_inequality_l1008_100877

-- Define the function f(x)
def f (x : ℝ) := |x + 1| + |x - 2|

-- Theorem for part 1
theorem solution_set_of_f_leq_x_plus_5 :
  {x : ℝ | f x ≤ x + 5} = Set.Icc (-4/3) 6 := by sorry

-- Theorem for part 2
theorem range_of_x_for_inequality :
  {x : ℝ | ∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |3*a - 1|) / |a|} = 
    Set.Iic (-3/2) ∪ Set.Ici (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_x_plus_5_range_of_x_for_inequality_l1008_100877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_similar_lattice_triangle_circumcenter_not_lattice_l1008_100807

-- Define a lattice point
def LatticePoint (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define a lattice triangle
def LatticeTriangle (A B C : ℝ × ℝ) : Prop :=
  LatticePoint A ∧ LatticePoint B ∧ LatticePoint C

-- Define similarity between triangles
def SimilarTriangles (A B C A' B' C' : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = k * ((B'.1 - A'.1)^2 + (B'.2 - A'.2)^2) ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = k * ((C'.1 - A'.1)^2 + (C'.2 - A'.2)^2) ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = k * ((C'.1 - B'.1)^2 + (C'.2 - B'.2)^2)

-- Define the circumcenter of a triangle
noncomputable def Circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- Definition of circumcenter

-- Define the area of a triangle
noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  sorry -- Definition of area

-- Theorem statement
theorem smallest_similar_lattice_triangle_circumcenter_not_lattice :
  ∀ (A B C : ℝ × ℝ),
    LatticeTriangle A B C →
    (∀ (A' B' C' : ℝ × ℝ),
      LatticeTriangle A' B' C' →
      SimilarTriangles A B C A' B' C' →
      area A B C ≤ area A' B' C') →
    ¬ LatticePoint (Circumcenter A B C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_similar_lattice_triangle_circumcenter_not_lattice_l1008_100807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_one_third_l1008_100814

/-- The sum of the infinite series ∑(n=1 to ∞) [(n^2 + 2n - 2) / (n+3)!] -/
noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (n^2 + 2*n - 2 : ℝ) / (n + 3).factorial

/-- Theorem: The sum of the infinite series ∑(n=1 to ∞) [(n^2 + 2n - 2) / (n+3)!] is equal to 1/3 -/
theorem infinite_series_sum_equals_one_third : infinite_series_sum = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_one_third_l1008_100814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1008_100815

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) / 3

-- Define the point of tangency
def x₀ : ℝ := 3

-- Define the slope of the tangent line
noncomputable def m : ℝ := (2 * x₀ - 3) / 3

-- Define the y-coordinate of the point of tangency
noncomputable def y₀ : ℝ := f x₀

-- Theorem: The equation of the tangent line is y = x - 2
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = x - 2 := by
  sorry

#eval x₀  -- This line is added to ensure some computable part exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1008_100815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_min_value_F_main_result_l1008_100885

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

-- Define the max function
noncomputable def max_abs (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Theorem 1
theorem solution_set_f (m n : ℝ) (h : m + n = 7) :
  {x : ℝ | f x ≥ (m + n) * x} = {x : ℝ | x ≤ 0} := by sorry

-- Theorem 2
theorem min_value_F (m n : ℝ) (h : m + n = 7) :
  ∀ x y : ℝ, max_abs (|x^2 - 4*y + m|) (|y^2 - 2*x + n|) ≥ 1 := by sorry

-- Theorem 3 (combining both results)
theorem main_result (m n : ℝ) (h : m + n = 7) :
  ({x : ℝ | f x ≥ (m + n) * x} = {x : ℝ | x ≤ 0}) ∧
  (∀ x y : ℝ, max_abs (|x^2 - 4*y + m|) (|y^2 - 2*x + n|) ≥ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_min_value_F_main_result_l1008_100885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_A_l1008_100888

-- Define the function A as noncomputable
noncomputable def A (α : Real) : Real :=
  (Real.sin (15 * Real.pi / 8 - 4 * α))^2 - (Real.sin (17 * Real.pi / 8 - 4 * α))^2

-- State the theorem
theorem max_value_of_A :
  ∃ (α_max : Real),
    0 ≤ α_max ∧ α_max ≤ Real.pi / 8 ∧
    A α_max = 1 / Real.sqrt 2 ∧
    ∀ (α : Real), 0 ≤ α ∧ α ≤ Real.pi / 8 → A α ≤ 1 / Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

#check max_value_of_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_A_l1008_100888
