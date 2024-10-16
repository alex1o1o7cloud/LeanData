import Mathlib

namespace NUMINAMATH_CALUDE_initial_interest_rate_l3268_326855

/-- Given a deposit with specific interest conditions, prove the initial interest rate -/
theorem initial_interest_rate (P : ℝ) (r : ℝ) : 
  (P * r / 100 = 202.50) →
  (P * (r + 5) / 100 = 225) →
  r = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_interest_rate_l3268_326855


namespace NUMINAMATH_CALUDE_total_selling_price_calculation_l3268_326808

def calculate_total_selling_price (item1_cost item2_cost item3_cost : ℚ)
  (loss1 loss2 loss3 tax_rate : ℚ) (overhead : ℚ) : ℚ :=
  let total_purchase := item1_cost + item2_cost + item3_cost
  let tax := tax_rate * total_purchase
  let selling_price1 := item1_cost * (1 - loss1)
  let selling_price2 := item2_cost * (1 - loss2)
  let selling_price3 := item3_cost * (1 - loss3)
  let total_selling := selling_price1 + selling_price2 + selling_price3
  total_selling + overhead + tax

theorem total_selling_price_calculation :
  calculate_total_selling_price 750 1200 500 0.1 0.15 0.05 0.05 300 = 2592.5 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_calculation_l3268_326808


namespace NUMINAMATH_CALUDE_initial_girls_percentage_l3268_326807

theorem initial_girls_percentage 
  (initial_total : ℕ)
  (new_boys : ℕ)
  (new_girls_percentage : ℚ)
  (h1 : initial_total = 20)
  (h2 : new_boys = 5)
  (h3 : new_girls_percentage = 32 / 100) :
  let initial_girls := (new_girls_percentage * (initial_total + new_boys)).floor
  let initial_girls_percentage := initial_girls / initial_total
  initial_girls_percentage = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_initial_girls_percentage_l3268_326807


namespace NUMINAMATH_CALUDE_probability_blue_after_removal_l3268_326862

/-- Probability of pulling a blue ball after removal -/
theorem probability_blue_after_removal (initial_total : ℕ) (initial_blue : ℕ) (removed_blue : ℕ) :
  initial_total = 18 →
  initial_blue = 6 →
  removed_blue = 3 →
  (initial_blue - removed_blue : ℚ) / (initial_total - removed_blue) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_blue_after_removal_l3268_326862


namespace NUMINAMATH_CALUDE_four_intersections_iff_l3268_326827

/-- The number of intersection points between x^2 + y^2 = a^2 and y = x^2 - a - 1 -/
def intersection_count (a : ℝ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the condition for exactly four intersection points -/
theorem four_intersections_iff (a : ℝ) :
  intersection_count a = 4 ↔ a > -1/2 :=
by sorry

end NUMINAMATH_CALUDE_four_intersections_iff_l3268_326827


namespace NUMINAMATH_CALUDE_lily_tennis_balls_l3268_326894

theorem lily_tennis_balls (brian frodo lily : ℕ) 
  (h1 : brian = 2 * frodo) 
  (h2 : frodo = lily + 8) 
  (h3 : brian = 22) : 
  lily = 3 := by
sorry

end NUMINAMATH_CALUDE_lily_tennis_balls_l3268_326894


namespace NUMINAMATH_CALUDE_pizza_combinations_l3268_326876

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3268_326876


namespace NUMINAMATH_CALUDE_johns_allowance_l3268_326840

def weekly_allowance : ℝ → Prop :=
  λ A => 
    let arcade_spent := (3/5) * A
    let remaining_after_arcade := A - arcade_spent
    let toy_store_spent := (1/3) * remaining_after_arcade
    let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
    remaining_after_toy_store = 0.60

theorem johns_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 2.25 := by sorry

end NUMINAMATH_CALUDE_johns_allowance_l3268_326840


namespace NUMINAMATH_CALUDE_andy_ant_position_l3268_326867

/-- Represents a coordinate point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction (North, East, South, West) -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of Andy the Ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat

/-- Calculates the next direction after a left turn -/
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.East => Direction.North
  | Direction.South => Direction.East
  | Direction.West => Direction.South

/-- Calculates the movement distance for a given move number -/
def moveDistance (n : Nat) : Int :=
  2 * n + 1

/-- Performs a single move and updates the ant's state -/
def move (state : AntState) : AntState :=
  let dist := moveDistance state.moveCount
  let newPos := match state.direction with
    | Direction.North => { x := state.position.x, y := state.position.y + dist }
    | Direction.East => { x := state.position.x + dist, y := state.position.y }
    | Direction.South => { x := state.position.x, y := state.position.y - dist }
    | Direction.West => { x := state.position.x - dist, y := state.position.y }
  { position := newPos,
    direction := nextDirection state.direction,
    moveCount := state.moveCount + 1 }

/-- Performs n moves and returns the final state -/
def nMoves (n : Nat) (initialState : AntState) : AntState :=
  if n = 0 then initialState else nMoves (n - 1) (move initialState)

/-- The main theorem to prove -/
theorem andy_ant_position :
  let initialState : AntState := {
    position := { x := 10, y := -10 },
    direction := Direction.East,
    moveCount := 0
  }
  let finalState := nMoves 2022 initialState
  finalState.position = { x := 12, y := 4038 } := by sorry

end NUMINAMATH_CALUDE_andy_ant_position_l3268_326867


namespace NUMINAMATH_CALUDE_cruise_liner_travelers_l3268_326856

theorem cruise_liner_travelers :
  ∃ a : ℕ,
    250 ≤ a ∧ a ≤ 400 ∧
    a % 15 = 8 ∧
    a % 25 = 17 ∧
    (a = 292 ∨ a = 367) :=
by sorry

end NUMINAMATH_CALUDE_cruise_liner_travelers_l3268_326856


namespace NUMINAMATH_CALUDE_quadratic_inequality_counterexample_l3268_326828

theorem quadratic_inequality_counterexample :
  ∃ (a b c : ℝ), b^2 - 4*a*c ≤ 0 ∧ ∃ (x : ℝ), a*x^2 + b*x + c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_counterexample_l3268_326828


namespace NUMINAMATH_CALUDE_softball_team_composition_l3268_326814

theorem softball_team_composition :
  ∀ (men women : ℕ),
  men + women = 16 →
  (men : ℚ) / (women : ℚ) = 7 / 9 →
  women - men = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_composition_l3268_326814


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3268_326834

/-- The sum of the digits of (10^40 - 46) is 369. -/
theorem sum_of_digits_of_large_number : 
  (let k := 10^40 - 46
   Finset.sum (Finset.range 41) (λ i => (k / 10^i) % 10)) = 369 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3268_326834


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l3268_326810

theorem quadratic_rational_solutions_product : ∃ (d₁ d₂ : ℕ+),
  (∀ (d : ℕ+), (∃ (x : ℚ), 8 * x^2 + 16 * x + d.val = 0) ↔ (d = d₁ ∨ d = d₂)) ∧
  d₁.val * d₂.val = 48 :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l3268_326810


namespace NUMINAMATH_CALUDE_exists_zero_sum_assignment_l3268_326818

/-- A regular 2n-gon -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- An arrow assignment for a regular 2n-gon -/
def ArrowAssignment (n : ℕ) := 
  (Fin (2*n) × Fin (2*n)) → ℝ × ℝ

/-- The sum of vectors in an arrow assignment -/
def sumVectors (n : ℕ) (assignment : ArrowAssignment n) : ℝ × ℝ := sorry

/-- Theorem stating the existence of a zero-sum arrow assignment -/
theorem exists_zero_sum_assignment (n : ℕ) (polygon : RegularPolygon n) :
  ∃ (assignment : ArrowAssignment n), sumVectors n assignment = (0, 0) := by sorry

end NUMINAMATH_CALUDE_exists_zero_sum_assignment_l3268_326818


namespace NUMINAMATH_CALUDE_not_square_or_cube_of_2pow_minus_1_l3268_326875

theorem not_square_or_cube_of_2pow_minus_1 (n : ℕ) (h : n > 1) :
  ¬∃ (a : ℤ), (2^n - 1 : ℤ) = a^2 ∨ (2^n - 1 : ℤ) = a^3 := by
  sorry

end NUMINAMATH_CALUDE_not_square_or_cube_of_2pow_minus_1_l3268_326875


namespace NUMINAMATH_CALUDE_circle_tangency_l3268_326857

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 - r2)^2

theorem circle_tangency (m : ℝ) :
  let c1 : ℝ × ℝ := (0, 0)
  let r1 : ℝ := Real.sqrt m
  let c2 : ℝ × ℝ := (-3, 4)
  let r2 : ℝ := 6
  internally_tangent c1 c2 r1 r2 → m = 1 ∨ m = 121 := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_l3268_326857


namespace NUMINAMATH_CALUDE_value_after_two_years_theorem_l3268_326878

/-- Calculates the value of an amount after two years, considering annual increases and inflation rates -/
def value_after_two_years (initial_amount : ℝ) (annual_increase_rate : ℝ) (inflation_rate_year1 : ℝ) (inflation_rate_year2 : ℝ) : ℝ :=
  let amount_year1 := initial_amount * (1 + annual_increase_rate)
  let value_year1 := amount_year1 * (1 - inflation_rate_year1)
  let amount_year2 := value_year1 * (1 + annual_increase_rate)
  let value_year2 := amount_year2 * (1 - inflation_rate_year2)
  value_year2

/-- Theorem stating that the value after two years is approximately 3771.36 -/
theorem value_after_two_years_theorem :
  let initial_amount : ℝ := 3200
  let annual_increase_rate : ℝ := 1/8
  let inflation_rate_year1 : ℝ := 3/100
  let inflation_rate_year2 : ℝ := 4/100
  abs (value_after_two_years initial_amount annual_increase_rate inflation_rate_year1 inflation_rate_year2 - 3771.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_value_after_two_years_theorem_l3268_326878


namespace NUMINAMATH_CALUDE_triangle_properties_l3268_326871

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the altitude line
def altitude_line (x y : ℝ) : Prop := 2 * x - 3 * y + 14 = 0

-- Define the equidistant lines
def equidistant_line1 (x y : ℝ) : Prop := 7 * x - 6 * y + 4 = 0
def equidistant_line2 (x y : ℝ) : Prop := 3 * x + 2 * y - 44 = 0

-- Theorem statement
theorem triangle_properties :
  -- 1. The altitude from A to BC
  (∀ x y : ℝ, altitude_line x y ↔ 
    (x - A.1) * (B.1 - C.1) + (y - A.2) * (B.2 - C.2) = 0 ∧ 
    ∃ t : ℝ, x = A.1 + t * (B.2 - C.2) ∧ y = A.2 - t * (B.1 - C.1)) ∧
  -- 2. The lines through B equidistant from A and C
  (∀ x y : ℝ, (equidistant_line1 x y ∨ equidistant_line2 x y) ↔
    abs ((y - A.2) * (B.1 - A.1) - (x - A.1) * (B.2 - A.2)) = 
    abs ((y - C.2) * (B.1 - C.1) - (x - C.1) * (B.2 - C.2))) :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3268_326871


namespace NUMINAMATH_CALUDE_flora_initial_daily_milk_l3268_326845

def total_milk : ℕ := 105
def weeks : ℕ := 3
def days_per_week : ℕ := 7
def brother_additional : ℕ := 2

theorem flora_initial_daily_milk :
  let total_days : ℕ := weeks * days_per_week
  let flora_initial_think : ℕ := total_milk / total_days
  flora_initial_think = 5 := by sorry

end NUMINAMATH_CALUDE_flora_initial_daily_milk_l3268_326845


namespace NUMINAMATH_CALUDE_tangent_ratio_problem_l3268_326825

theorem tangent_ratio_problem (α : ℝ) (h : Real.tan (π - α) = 1/3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_problem_l3268_326825


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3268_326819

theorem more_girls_than_boys (boys girls : ℕ) : 
  boys = 40 →
  girls * 5 = boys * 13 →
  girls > boys →
  girls - boys = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3268_326819


namespace NUMINAMATH_CALUDE_walk_distance_proof_l3268_326872

/-- Given a total distance walked and a distance walked before rest,
    calculate the distance walked after rest. -/
def distance_after_rest (total : Real) (before_rest : Real) : Real :=
  total - before_rest

/-- Theorem: Given a total distance of 1 mile and a distance of 0.75 mile
    walked before rest, the distance walked after rest is 0.25 mile. -/
theorem walk_distance_proof :
  let total_distance : Real := 1
  let before_rest : Real := 0.75
  distance_after_rest total_distance before_rest = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l3268_326872


namespace NUMINAMATH_CALUDE_functional_equation_iff_forms_l3268_326885

/-- The functional equation that f and g must satisfy for all real x and y -/
def functional_equation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Real.sin x + Real.cos y = f x + f y + g x - g y

/-- The proposed form of function f -/
def f_form (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (Real.sin x + Real.cos x) / 2

/-- The proposed form of function g, with an arbitrary constant C -/
def g_form (g : ℝ → ℝ) : Prop :=
  ∃ C : ℝ, ∀ x : ℝ, g x = (Real.sin x - Real.cos x) / 2 + C

/-- The main theorem stating the equivalence between the functional equation and the proposed forms of f and g -/
theorem functional_equation_iff_forms (f g : ℝ → ℝ) :
  functional_equation f g ↔ (f_form f ∧ g_form g) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_iff_forms_l3268_326885


namespace NUMINAMATH_CALUDE_mean_twice_mode_iff_x_21_l3268_326868

def is_valid_list (x : ℕ) : Prop :=
  x > 0 ∧ x ≤ 100

def mean_of_list (x : ℕ) : ℚ :=
  (31 + 58 + 98 + 3 * x) / 6

def mode_of_list (x : ℕ) : ℕ := x

theorem mean_twice_mode_iff_x_21 :
  ∀ x : ℕ, is_valid_list x →
    (mean_of_list x = 2 * mode_of_list x) ↔ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_mean_twice_mode_iff_x_21_l3268_326868


namespace NUMINAMATH_CALUDE_oldest_babysat_age_jane_l3268_326879

/-- Represents a person with their current age and baby-sitting history. -/
structure Person where
  currentAge : ℕ
  babySittingStartAge : ℕ
  babySittingEndAge : ℕ

/-- Calculates the maximum age of a child that a person could have babysat. -/
def maxBabysatChildAge (p : Person) : ℕ :=
  p.babySittingEndAge / 2

/-- Calculates the current age of the oldest person that could have been babysat. -/
def oldestBabysatPersonCurrentAge (p : Person) : ℕ :=
  maxBabysatChildAge p + (p.currentAge - p.babySittingEndAge)

/-- Theorem stating the age of the oldest person Jane could have babysat. -/
theorem oldest_babysat_age_jane :
  let jane : Person := {
    currentAge := 32,
    babySittingStartAge := 18,
    babySittingEndAge := 20
  }
  oldestBabysatPersonCurrentAge jane = 22 := by
  sorry


end NUMINAMATH_CALUDE_oldest_babysat_age_jane_l3268_326879


namespace NUMINAMATH_CALUDE_brendans_dad_fish_count_l3268_326812

theorem brendans_dad_fish_count :
  ∀ (morning afternoon thrown_back total dad_catch : ℕ),
    morning = 8 →
    afternoon = 5 →
    thrown_back = 3 →
    total = 23 →
    dad_catch = total - (morning + afternoon - thrown_back) →
    dad_catch = 13 := by
  sorry

end NUMINAMATH_CALUDE_brendans_dad_fish_count_l3268_326812


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l3268_326864

/-- Given a function f(x) = ax³ + bx - 2 where f(2017) = 10, prove that f(-2017) = -14 -/
theorem cubic_function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 2
  f 2017 = 10 → f (-2017) = -14 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l3268_326864


namespace NUMINAMATH_CALUDE_cannot_finish_fourth_l3268_326833

-- Define the set of runners
inductive Runner : Type
  | A | B | C | D | E | F | G

-- Define the race result as a function from Runner to Nat (position)
def RaceResult := Runner → Nat

-- Define the conditions of the race
def ValidRaceResult (result : RaceResult) : Prop :=
  (result Runner.A < result Runner.B) ∧
  (result Runner.A < result Runner.C) ∧
  (result Runner.B < result Runner.D) ∧
  (result Runner.C < result Runner.E) ∧
  (result Runner.A < result Runner.F) ∧ (result Runner.F < result Runner.B) ∧
  (result Runner.B < result Runner.G) ∧ (result Runner.G < result Runner.C)

-- Theorem to prove
theorem cannot_finish_fourth (result : RaceResult) 
  (h : ValidRaceResult result) : 
  result Runner.A ≠ 4 ∧ result Runner.F ≠ 4 ∧ result Runner.G ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_cannot_finish_fourth_l3268_326833


namespace NUMINAMATH_CALUDE_gum_cost_l3268_326838

/-- The cost of gum in cents -/
def cost_per_piece : ℕ := 2

/-- The number of pieces of gum -/
def num_pieces : ℕ := 500

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 500 pieces of gum is 1000 cents and 10 dollars -/
theorem gum_cost :
  (num_pieces * cost_per_piece = 1000) ∧
  (num_pieces * cost_per_piece / cents_per_dollar = 10) :=
by sorry

end NUMINAMATH_CALUDE_gum_cost_l3268_326838


namespace NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l3268_326803

theorem multiple_of_nine_implies_multiple_of_three (n : ℤ) :
  (∀ k : ℤ, 9 ∣ k → 3 ∣ k) →
  9 ∣ n →
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l3268_326803


namespace NUMINAMATH_CALUDE_extremum_sum_l3268_326809

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_sum (a b : ℝ) : 
  (f a b 1 = 10) ∧ (f' a b 1 = 0) → a + b = -7 :=
by
  sorry

#check extremum_sum

end NUMINAMATH_CALUDE_extremum_sum_l3268_326809


namespace NUMINAMATH_CALUDE_rational_cube_sum_zero_l3268_326881

theorem rational_cube_sum_zero (x y z : ℚ) 
  (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_zero_l3268_326881


namespace NUMINAMATH_CALUDE_candy_store_sales_theorem_l3268_326880

/-- Represents the sales data of a candy store -/
structure CandyStoreSales where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  trufflePrice : ℚ
  pretzelsDozens : ℕ
  pretzelPrice : ℚ

/-- Calculates the total money made by the candy store -/
def totalMoney (sales : CandyStoreSales) : ℚ :=
  sales.fudgePounds * sales.fudgePrice +
  sales.trufflesDozens * 12 * sales.trufflePrice +
  sales.pretzelsDozens * 12 * sales.pretzelPrice

/-- Theorem stating that the candy store made $212.00 -/
theorem candy_store_sales_theorem (sales : CandyStoreSales) 
  (h1 : sales.fudgePounds = 20)
  (h2 : sales.fudgePrice = 5/2)
  (h3 : sales.trufflesDozens = 5)
  (h4 : sales.trufflePrice = 3/2)
  (h5 : sales.pretzelsDozens = 3)
  (h6 : sales.pretzelPrice = 2) :
  totalMoney sales = 212 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_sales_theorem_l3268_326880


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3268_326874

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 4*I
  let z₂ : ℂ := 2 - 4*I
  (z₁ / z₂ - z₂ / z₁) = (4:ℝ)/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3268_326874


namespace NUMINAMATH_CALUDE_no_solutions_inequality_l3268_326858

theorem no_solutions_inequality : ¬∃ (n k : ℕ), n ≤ n! - k^n ∧ n! - k^n ≤ k * n := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_inequality_l3268_326858


namespace NUMINAMATH_CALUDE_complex_quadrant_l3268_326802

theorem complex_quadrant (z : ℂ) (h : (3 + 4*I)*z = 25) : 
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3268_326802


namespace NUMINAMATH_CALUDE_inequality_proof_l3268_326861

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a + b) * (b + c) * (c + a) = 1) : 
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3268_326861


namespace NUMINAMATH_CALUDE_binomial_recurrence_l3268_326843

theorem binomial_recurrence (n k : ℕ) (h1 : k ≤ n) (h2 : ¬(n = 0 ∧ k = 0)) :
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_recurrence_l3268_326843


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_7_l3268_326846

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year < 10000

theorem first_year_after_2020_with_digit_sum_7 :
  ∃ (year : ℕ), is_valid_year year ∧ 
    year > 2020 ∧ 
    sum_of_digits year = 7 ∧
    (∀ y, is_valid_year y → y > 2020 → y < year → sum_of_digits y ≠ 7) ∧
    year = 2021 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_7_l3268_326846


namespace NUMINAMATH_CALUDE_albert_purchase_cost_l3268_326889

/-- The total cost of horses and cows bought by Albert --/
def total_cost (num_horses num_cows : ℕ) (horse_cost cow_cost : ℕ) : ℕ :=
  num_horses * horse_cost + num_cows * cow_cost

/-- The profit from selling an item at a certain percentage --/
def profit_from_sale (cost : ℕ) (profit_percentage : ℚ) : ℚ :=
  (cost : ℚ) * profit_percentage

theorem albert_purchase_cost :
  ∃ (cow_cost : ℕ),
    let num_horses : ℕ := 4
    let num_cows : ℕ := 9
    let horse_cost : ℕ := 2000
    let horse_profit_percentage : ℚ := 1/10
    let cow_profit_percentage : ℚ := 1/5
    let total_profit : ℕ := 1880
    (num_horses : ℚ) * profit_from_sale horse_cost horse_profit_percentage +
    (num_cows : ℚ) * profit_from_sale cow_cost cow_profit_percentage = total_profit ∧
    total_cost num_horses num_cows horse_cost cow_cost = 13400 :=
by sorry


end NUMINAMATH_CALUDE_albert_purchase_cost_l3268_326889


namespace NUMINAMATH_CALUDE_lcm_of_12_and_16_l3268_326849

theorem lcm_of_12_and_16 :
  let n : ℕ := 12
  let m : ℕ := 16
  let gcf : ℕ := 4
  Nat.gcd n m = gcf →
  Nat.lcm n m = 48 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_16_l3268_326849


namespace NUMINAMATH_CALUDE_total_players_count_l3268_326890

/-- The number of players who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of players who play kho kho only -/
def kho_kho_only_players : ℕ := 40

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabadi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 45 := by sorry

end NUMINAMATH_CALUDE_total_players_count_l3268_326890


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3268_326844

theorem chess_tournament_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * (n - 1)) / 2 = 190 ∧ 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3268_326844


namespace NUMINAMATH_CALUDE_pin_purchase_cost_l3268_326806

theorem pin_purchase_cost (num_pins : ℕ) (original_price : ℚ) (discount_percent : ℚ) :
  num_pins = 10 →
  original_price = 20 →
  discount_percent = 15 / 100 →
  (num_pins : ℚ) * (original_price * (1 - discount_percent)) = 170 :=
by sorry

end NUMINAMATH_CALUDE_pin_purchase_cost_l3268_326806


namespace NUMINAMATH_CALUDE_chord_arithmetic_sequence_l3268_326883

theorem chord_arithmetic_sequence (n : ℕ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5*x}
  let point := (5/2, 3/2)
  let shortest_chord := 4
  let longest_chord := 5
  ∀ d : ℝ, 1/6 < d ∧ d ≤ 1/3 →
    (n > 0 ∧ 
     shortest_chord + (n - 1) * d = longest_chord ∧
     point ∈ circle) →
    n ∈ ({4, 5, 6} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_chord_arithmetic_sequence_l3268_326883


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_system_l3268_326830

theorem smallest_solution_congruence_system :
  ∃ (x : ℕ), x > 0 ∧
    x % 17 = 15 ∧
    x % 13 = 11 ∧
    x % 10 = 3 ∧
    (∀ (y : ℕ), y > 0 ∧ y % 17 = 15 ∧ y % 13 = 11 ∧ y % 10 = 3 → y ≥ x) ∧
    x = 1103 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_system_l3268_326830


namespace NUMINAMATH_CALUDE_f_composition_value_l3268_326847

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_composition_value : f (3 * f (-1)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3268_326847


namespace NUMINAMATH_CALUDE_dot_path_length_on_rolled_cube_l3268_326832

/-- The path length of a dot on a cube when rolled twice --/
theorem dot_path_length_on_rolled_cube : 
  let cube_edge_length : ℝ := 2
  let dot_distance_from_center : ℝ := cube_edge_length / 4
  let roll_count : ℕ := 2
  let radius : ℝ := Real.sqrt (1^2 + dot_distance_from_center^2)
  let path_length : ℝ := roll_count * 2 * π * radius
  path_length = 2.236 * π := by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rolled_cube_l3268_326832


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3268_326866

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + 1 > 0) → (-2 ≤ m ∧ m ≤ 2) ∧
  ¬((-2 ≤ m ∧ m ≤ 2) → (∀ x : ℝ, x^2 - m*x + 1 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3268_326866


namespace NUMINAMATH_CALUDE_odds_against_C_winning_l3268_326887

-- Define the type for horses
inductive Horse : Type
| A
| B
| C

-- Define the function for odds against winning
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 4 / 1
  | Horse.B => 3 / 2
  | Horse.C => 3 / 2  -- This is what we want to prove

-- State the theorem
theorem odds_against_C_winning :
  (∀ h₁ h₂ : Horse, h₁ ≠ h₂ → oddsAgainst h₁ ≠ oddsAgainst h₂) →  -- No ties
  oddsAgainst Horse.A = 4 / 1 →
  oddsAgainst Horse.B = 3 / 2 →
  oddsAgainst Horse.C = 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_odds_against_C_winning_l3268_326887


namespace NUMINAMATH_CALUDE_ellipse_equation_from_line_through_focus_and_vertex_l3268_326895

/-- Represents an ellipse in standard form -/
structure StandardEllipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: If a line with equation x - 2y + 2 = 0 passes through a focus and a vertex of an ellipse,
    then the standard equation of the ellipse is either x²/5 + y² = 1 or x²/4 + y²/5 = 1 -/
theorem ellipse_equation_from_line_through_focus_and_vertex 
  (l : Line) 
  (hl : l.a = 1 ∧ l.b = -2 ∧ l.c = 2) 
  (passes_through_focus_and_vertex : ∃ (e : StandardEllipse), 
    (∃ (x y : ℝ), x - 2*y + 2 = 0 ∧ 
      ((x = e.a ∧ y = 0) ∨ (x = 0 ∧ y = e.b) ∨ (x = -e.a ∧ y = 0) ∨ (x = 0 ∧ y = -e.b)) ∧
      ((x^2 / e.a^2 + y^2 / e.b^2 = 1) ∨ (y^2 / e.a^2 + x^2 / e.b^2 = 1)))) :
  ∃ (e : StandardEllipse), (e.a^2 = 5 ∧ e.b^2 = 1) ∨ (e.a^2 = 4 ∧ e.b^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_line_through_focus_and_vertex_l3268_326895


namespace NUMINAMATH_CALUDE_circles_intersect_l3268_326816

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- Definition of intersection for two circles -/
def intersect (c : TwoCircles) : Prop :=
  let d := Real.sqrt ((c.center1.1 - c.center2.1)^2 + (c.center1.2 - c.center2.2)^2)
  d < c.radius1 + c.radius2 ∧ d > abs (c.radius1 - c.radius2)

/-- The main theorem: the given circles intersect -/
theorem circles_intersect : 
  let c := TwoCircles.mk (0, 0) 2 (-3, 4) 4
  intersect c := by sorry


end NUMINAMATH_CALUDE_circles_intersect_l3268_326816


namespace NUMINAMATH_CALUDE_solution_set_inequality_for_a_b_l3268_326813

-- Define the inequality
def satisfies_inequality (x : ℝ) : Prop := abs (x + 1) + abs (x + 3) < 4

-- Theorem for the solution set
theorem solution_set :
  ∀ x : ℝ, satisfies_inequality x ↔ -4 < x ∧ x < 0 := by sorry

-- Theorem for the inequality between a and b
theorem inequality_for_a_b (a b : ℝ) 
  (ha : satisfies_inequality a) (hb : satisfies_inequality b) :
  2 * abs (a - b) < abs (a * b + 2 * a + 2 * b) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_for_a_b_l3268_326813


namespace NUMINAMATH_CALUDE_cartesian_product_eq_expected_set_l3268_326899

-- Define the set of possible x and y values
def X : Set ℕ := {1, 2}
def Y : Set ℕ := {1, 2}

-- Define the Cartesian product set
def cartesianProduct : Set (ℕ × ℕ) := {p | p.1 ∈ X ∧ p.2 ∈ Y}

-- Define the expected result set
def expectedSet : Set (ℕ × ℕ) := {(1, 1), (1, 2), (2, 1), (2, 2)}

-- Theorem stating that the Cartesian product is equal to the expected set
theorem cartesian_product_eq_expected_set : cartesianProduct = expectedSet := by
  sorry

end NUMINAMATH_CALUDE_cartesian_product_eq_expected_set_l3268_326899


namespace NUMINAMATH_CALUDE_limit_at_one_l3268_326820

def f (x : ℝ) : ℝ := 2 * x^3

theorem limit_at_one (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (1 + Δx) - f 1) / Δx) - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_at_one_l3268_326820


namespace NUMINAMATH_CALUDE_function_property_l3268_326859

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def symmetric_about_origin (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f (-x - 1)

-- State the theorem
theorem function_property (h1 : is_even f) (h2 : symmetric_about_origin f) (h3 : f 0 = 1) :
  f (-1) + f 2 = -1 := by sorry

end NUMINAMATH_CALUDE_function_property_l3268_326859


namespace NUMINAMATH_CALUDE_shaded_area_is_ten_l3268_326805

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents the configuration of two adjacent squares -/
structure TwoSquares where
  small : Square
  large : Square
  adjacent : True  -- This is a placeholder for the adjacency condition

/-- Calculates the area of the shaded region formed by the diagonal of the larger square
    overlapping with the smaller square in a TwoSquares configuration -/
def shaded_area (squares : TwoSquares) : ℝ :=
  sorry

/-- Theorem stating that for a TwoSquares configuration with sides 4 and 12,
    the shaded area is 10 square units -/
theorem shaded_area_is_ten (squares : TwoSquares)
  (h1 : squares.small.side = 4)
  (h2 : squares.large.side = 12) :
  shaded_area squares = 10 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_ten_l3268_326805


namespace NUMINAMATH_CALUDE_race_win_probability_l3268_326893

/-- Represents the weather conditions -/
inductive Weather
  | Sunny
  | Rainy

/-- Represents a car in the race -/
structure Car where
  winProb : ℚ  -- Winning probability as a rational number

/-- Calculates the total winning probability for the given cars under the specified weather condition -/
def totalWinProb (cars : List Car) (weather : Weather) : ℚ :=
  cars.map (fun c => c.winProb) |>.sum

theorem race_win_probability (carA carB carC carD : Car) (carE : Weather → Car) :
  carA.winProb = 1/8 →
  carB.winProb = 1/12 →
  carC.winProb = 1/6 →
  carD.winProb = 1/10 →
  (carE Weather.Sunny).winProb = 1/20 →
  (carE Weather.Rainy).winProb = 1/15 →
  totalWinProb [carA, carB, carC, carD, carE Weather.Sunny] Weather.Sunny = 21/40 ∧
  totalWinProb [carA, carB, carC, carD, carE Weather.Rainy] Weather.Rainy = 13/24 := by
  sorry

#check race_win_probability

end NUMINAMATH_CALUDE_race_win_probability_l3268_326893


namespace NUMINAMATH_CALUDE_sin_4x_eq_sin_2x_solution_l3268_326815

open Set
open Real

def solution_set : Set ℝ := {π/6, π/2, π, 5*π/6, 7*π/6}

theorem sin_4x_eq_sin_2x_solution (x : ℝ) :
  x ∈ Ioo 0 (3*π/2) →
  (sin (4*x) = sin (2*x)) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_sin_4x_eq_sin_2x_solution_l3268_326815


namespace NUMINAMATH_CALUDE_number_divisibility_l3268_326821

theorem number_divisibility :
  (∀ a : ℕ, 100 ≤ a ∧ a < 1000 → (7 ∣ 1001 * a) ∧ (11 ∣ 1001 * a) ∧ (13 ∣ 1001 * a)) ∧
  (∀ b : ℕ, 1000 ≤ b ∧ b < 10000 → (73 ∣ 10001 * b) ∧ (137 ∣ 10001 * b)) :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l3268_326821


namespace NUMINAMATH_CALUDE_book_pages_equation_l3268_326854

theorem book_pages_equation (x : ℝ) : 
  x > 0 → 
  20 + (1/2) * (x - 20) + 15 = x := by
  sorry

end NUMINAMATH_CALUDE_book_pages_equation_l3268_326854


namespace NUMINAMATH_CALUDE_fred_initial_cards_l3268_326891

theorem fred_initial_cards (cards_bought cards_left : ℕ) : 
  cards_bought = 3 → cards_left = 2 → cards_bought + cards_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_fred_initial_cards_l3268_326891


namespace NUMINAMATH_CALUDE_unique_product_list_l3268_326831

/-- Extended alphabet mapping function -/
def alphabetValue (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26 | '#' => 27
  | _ => 0

/-- Calculate the product of a list of characters -/
def listProduct (s : List Char) : Nat :=
  s.foldl (fun acc c => acc * alphabetValue c) 1

/-- Check if a list is valid (four characters from the extended alphabet) -/
def isValidList (s : List Char) : Prop :=
  s.length = 4 ∧ s.all (fun c => alphabetValue c ≠ 0)

theorem unique_product_list : 
  ∀ (s : List Char), 
    isValidList s → 
    listProduct s = listProduct ['T', 'W', 'X', '#'] → 
    s = ['E', 'V', 'W', '#'] ∨ s = ['T', 'W', 'X', '#'] :=
sorry

end NUMINAMATH_CALUDE_unique_product_list_l3268_326831


namespace NUMINAMATH_CALUDE_alice_bob_number_game_l3268_326870

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem alice_bob_number_game (alice_num bob_num : ℕ) : 
  (1 ≤ alice_num ∧ alice_num ≤ 50) →
  (1 ≤ bob_num ∧ bob_num ≤ 50) →
  (alice_num ≠ 1) →
  (is_prime bob_num) →
  (∃ m : ℕ, 100 * bob_num + alice_num = m * m) →
  (alice_num = 24 ∨ alice_num = 61) :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_number_game_l3268_326870


namespace NUMINAMATH_CALUDE_tagged_ratio_is_one_thirtieth_l3268_326877

/-- Represents the fish population in a pond -/
structure FishPopulation where
  initialTagged : ℕ
  secondCatchTotal : ℕ
  secondCatchTagged : ℕ
  estimatedTotal : ℕ

/-- Calculates the ratio of tagged fish to total fish in the second catch -/
def taggedRatio (fp : FishPopulation) : ℚ :=
  fp.secondCatchTagged / fp.secondCatchTotal

/-- The specific fish population described in the problem -/
def pondPopulation : FishPopulation :=
  { initialTagged := 60
  , secondCatchTotal := 60
  , secondCatchTagged := 2
  , estimatedTotal := 1800 }

/-- Theorem stating that the ratio of tagged fish to total fish in the second catch is 1/30 -/
theorem tagged_ratio_is_one_thirtieth :
  taggedRatio pondPopulation = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_tagged_ratio_is_one_thirtieth_l3268_326877


namespace NUMINAMATH_CALUDE_a_range_l3268_326892

def f (a x : ℝ) := x^2 - 2*a*x + 7

theorem a_range (a : ℝ) : 
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) → 
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_a_range_l3268_326892


namespace NUMINAMATH_CALUDE_extremum_values_l3268_326837

/-- The function f(x) = x^3 - ax^2 - bx + a^2 has an extremum of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∨
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1)

/-- The main theorem -/
theorem extremum_values (a b : ℝ) :
  has_extremum a b ∧ (1^3 - a*1^2 - b*1 + a^2 = 10) →
  (a = 3 ∧ b = -3) ∨ (a = -4 ∧ b = 11) := by sorry


end NUMINAMATH_CALUDE_extremum_values_l3268_326837


namespace NUMINAMATH_CALUDE_simplify_fraction_l3268_326898

theorem simplify_fraction (a : ℚ) (h : a = 2) : 24 * a^5 / (72 * a^3) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3268_326898


namespace NUMINAMATH_CALUDE_weed_eater_string_cost_is_seven_l3268_326839

-- Define the number of lawnmower blades
def num_blades : ℕ := 4

-- Define the cost per blade in dollars
def cost_per_blade : ℕ := 8

-- Define the total spent on supplies in dollars
def total_spent : ℕ := 39

-- Define the cost of the weed eater string
def weed_eater_string_cost : ℕ := total_spent - (num_blades * cost_per_blade)

-- Theorem statement
theorem weed_eater_string_cost_is_seven :
  weed_eater_string_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_weed_eater_string_cost_is_seven_l3268_326839


namespace NUMINAMATH_CALUDE_eight_power_y_equals_one_eighth_of_two_power_36_l3268_326800

theorem eight_power_y_equals_one_eighth_of_two_power_36 :
  ∀ y : ℝ, (1/8 : ℝ) * (2^36) = 8^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_y_equals_one_eighth_of_two_power_36_l3268_326800


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3268_326860

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 6 = 0, 
    its center is at (-1, 2) and its radius is √11 -/
theorem circle_center_and_radius :
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y - 6 = 0)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = Real.sqrt 11 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3268_326860


namespace NUMINAMATH_CALUDE_classroom_ratio_l3268_326835

theorem classroom_ratio :
  ∀ (num_boys num_girls : ℕ),
  num_boys > 0 →
  num_girls > 0 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 
    3 * ((num_girls : ℚ) / (num_boys + num_girls : ℚ)) / 5 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3/8 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l3268_326835


namespace NUMINAMATH_CALUDE_this_year_cabbage_production_l3268_326836

/-- Represents a square garden where cabbages are grown -/
structure CabbageGarden where
  side : ℕ -- Side length of the square garden

/-- Calculates the number of cabbages in a square garden -/
def cabbageCount (garden : CabbageGarden) : ℕ := garden.side ^ 2

/-- Theorem stating the number of cabbages produced this year -/
theorem this_year_cabbage_production 
  (last_year : CabbageGarden) 
  (this_year : CabbageGarden) 
  (h1 : cabbageCount this_year - cabbageCount last_year = 211) :
  cabbageCount this_year = 11236 := by
  sorry


end NUMINAMATH_CALUDE_this_year_cabbage_production_l3268_326836


namespace NUMINAMATH_CALUDE_line_AC_passes_through_fixed_point_l3268_326888

-- Define the moving circle M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (center : ℝ × ℝ), 
    ((p.1 - center.1)^2 + (p.2 - center.2)^2 = (p.2 + 1)^2) ∧
    ((0 - center.1)^2 + (1 - center.2)^2 = (1 + 1)^2)}

-- Define the trajectory of M's center
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the moving line l
def l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 - 2}

-- Define points A and B as intersections of l and trajectory
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  trajectory ∩ l k

-- Define point C as symmetric to B with respect to y-axis
def C (B : ℝ × ℝ) : ℝ × ℝ :=
  (-B.1, B.2)

-- Theorem statement
theorem line_AC_passes_through_fixed_point :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    A ∈ intersectionPoints k →
    B ∈ intersectionPoints k →
    A ≠ B →
    (0, 2) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A + t • C B} :=
sorry

end NUMINAMATH_CALUDE_line_AC_passes_through_fixed_point_l3268_326888


namespace NUMINAMATH_CALUDE_inequality_solution_l3268_326841

theorem inequality_solution (x : ℝ) : 
  (3 - 1 / (3 * x + 2) < 5) ↔ (x < -5/3 ∨ x > -2/3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3268_326841


namespace NUMINAMATH_CALUDE_q_age_is_40_l3268_326863

/-- Represents the ages of two people p and q --/
structure Ages where
  p : ℕ
  q : ℕ

/-- The condition stated by p --/
def age_condition (ages : Ages) : Prop :=
  ages.p = 3 * (ages.q - (ages.p - ages.q))

/-- The sum of their present ages is 100 --/
def age_sum (ages : Ages) : Prop :=
  ages.p + ages.q = 100

/-- Theorem stating that given the conditions, q's present age is 40 --/
theorem q_age_is_40 (ages : Ages) 
  (h1 : age_condition ages) 
  (h2 : age_sum ages) : 
  ages.q = 40 := by
  sorry

end NUMINAMATH_CALUDE_q_age_is_40_l3268_326863


namespace NUMINAMATH_CALUDE_water_in_bucket_A_l3268_326804

/-- Given two buckets A and B, prove that the original amount of water in bucket A is 20 kg. -/
theorem water_in_bucket_A (bucket_A bucket_B : ℝ) : 
  (0.2 * bucket_A = 0.4 * bucket_B) → 
  (0.6 * bucket_B = 6) → 
  bucket_A = 20 := by sorry

end NUMINAMATH_CALUDE_water_in_bucket_A_l3268_326804


namespace NUMINAMATH_CALUDE_factor_expression_l3268_326822

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3268_326822


namespace NUMINAMATH_CALUDE_fifth_plot_excess_tiles_l3268_326882

def plot_width (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def plot_length (n : ℕ) : ℕ := 4 + 3 * (n - 1)
def plot_area (n : ℕ) : ℕ := plot_width n * plot_length n

theorem fifth_plot_excess_tiles : plot_area 5 - plot_area 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_fifth_plot_excess_tiles_l3268_326882


namespace NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l3268_326850

/-- Converts an octal number to decimal --/
def octal_to_decimal (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- Proves that the octal number 123₈ is equal to the decimal number 83 --/
theorem octal_123_equals_decimal_83 : octal_to_decimal 1 2 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l3268_326850


namespace NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l3268_326865

theorem two_digit_perfect_squares_divisible_by_four :
  (∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ 
    (10 ≤ n^2 ∧ n^2 ≤ 99) ∧ 4 ∣ n^2) ∧
  (∃ s, (∀ n, n ∈ s ↔ 
    (10 ≤ n^2 ∧ n^2 ≤ 99) ∧ 4 ∣ n^2) ∧ 
    Finset.card s = 3) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l3268_326865


namespace NUMINAMATH_CALUDE_distance_between_points_l3268_326853

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (-5, 3)
  let pointB : ℝ × ℝ := (6, 3)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3268_326853


namespace NUMINAMATH_CALUDE_power_of_half_equals_one_l3268_326817

theorem power_of_half_equals_one (a b : ℕ) : 
  (2^a : ℕ) ∣ 300 ∧ 
  (∀ k : ℕ, k > a → ¬((2^k : ℕ) ∣ 300)) ∧ 
  (3^b : ℕ) ∣ 300 ∧ 
  (∀ k : ℕ, k > b → ¬((3^k : ℕ) ∣ 300)) → 
  (1/2 : ℚ)^(b - a + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_half_equals_one_l3268_326817


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3268_326869

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the only function satisfying the functional equation is f(z) = 1 - z²/2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ z : ℝ, f z = 1 - z^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3268_326869


namespace NUMINAMATH_CALUDE_earliest_retirement_year_l3268_326811

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Employee's age in a given year -/
def age_in_year (hire_year : ℕ) (hire_age : ℕ) (current_year : ℕ) : ℕ :=
  (current_year - hire_year) + hire_age

/-- Employee's years of employment in a given year -/
def years_employed (hire_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - hire_year

theorem earliest_retirement_year 
  (hire_year : ℕ) 
  (hire_age : ℕ) 
  (retirement_year : ℕ) :
  hire_year = 1988 →
  hire_age = 32 →
  retirement_year = 2007 →
  rule_of_70 (age_in_year hire_year hire_age retirement_year) 
             (years_employed hire_year retirement_year) ∧
  ∀ y : ℕ, y < retirement_year →
    ¬(rule_of_70 (age_in_year hire_year hire_age y) 
                 (years_employed hire_year y)) :=
by sorry

end NUMINAMATH_CALUDE_earliest_retirement_year_l3268_326811


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l3268_326823

theorem max_value_of_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  (∀ x y : ℝ, x * y = 1 → x > y → y ≥ 2/3 → (x - y) / (x^2 + y^2) ≤ (a - b) / (a^2 + b^2)) →
  (a - b) / (a^2 + b^2) = 30/97 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l3268_326823


namespace NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l3268_326826

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

theorem specific_square_root_squared : (Real.sqrt 625681) ^ 2 = 625681 := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l3268_326826


namespace NUMINAMATH_CALUDE_candy_distribution_l3268_326842

/-- Given that Frank has a total of 16 pieces of candy and divides them equally into 2 bags,
    prove that each bag contains 8 pieces of candy. -/
theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 16 → num_bags = 2 → total_candy = num_bags * candy_per_bag → candy_per_bag = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3268_326842


namespace NUMINAMATH_CALUDE_james_calories_per_minute_l3268_326848

/-- Represents the number of calories burned per minute in a spinning class -/
def calories_per_minute (classes_per_week : ℕ) (hours_per_class : ℚ) (total_calories_per_week : ℕ) : ℚ :=
  let minutes_per_week : ℚ := classes_per_week * hours_per_class * 60
  total_calories_per_week / minutes_per_week

/-- Proves that James burns 7 calories per minute in his spinning class -/
theorem james_calories_per_minute :
  calories_per_minute 3 (3/2) 1890 = 7 := by
sorry

end NUMINAMATH_CALUDE_james_calories_per_minute_l3268_326848


namespace NUMINAMATH_CALUDE_age_problem_l3268_326824

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3268_326824


namespace NUMINAMATH_CALUDE_inequality_proof_l3268_326852

theorem inequality_proof (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ (1/3) * (x / z + z / x + 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3268_326852


namespace NUMINAMATH_CALUDE_prob_green_is_five_sevenths_l3268_326873

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def prob_green (containers : List Container) : ℚ :=
  let total_prob := containers.map (λ c => (c.green : ℚ) / (c.red + c.green))
  (total_prob.sum) / containers.length

/-- Theorem: The probability of selecting a green ball is 5/7 -/
theorem prob_green_is_five_sevenths : 
  let containers := [
    Container.mk 8 4,  -- Container I
    Container.mk 3 4,  -- Container II
    Container.mk 3 4   -- Container III
  ]
  prob_green containers = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_is_five_sevenths_l3268_326873


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3268_326884

theorem arithmetic_to_geometric_sequence (a₁ a₂ a₃ a₄ d : ℝ) : 
  d ≠ 0 →
  a₂ = a₁ + d →
  a₃ = a₁ + 2*d →
  a₄ = a₁ + 3*d →
  ((a₂^2 = a₁ * a₃) ∨ (a₂^2 = a₁ * a₄) ∨ (a₃^2 = a₁ * a₄) ∨ (a₃^2 = a₂ * a₄)) →
  (a₁ / d = -4 ∨ a₁ / d = 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3268_326884


namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l3268_326851

-- System of equations
theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 7 ∧ 3*x + y = 6 ∧ x = 1 ∧ y = 3 := by sorry

-- System of inequalities
theorem system_of_inequalities_solution :
  ∀ x : ℝ, (2*(x - 1) + 1 > -3 ∧ x - 1 ≤ (1 + x) / 3) ↔ (-1 < x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l3268_326851


namespace NUMINAMATH_CALUDE_price_increase_2008_2009_l3268_326886

/-- Given a 60% increase from 2006 to 2008 and a 20% annual average growth rate
    from 2006 to 2009, the increase from 2008 to 2009 is 8%. -/
theorem price_increase_2008_2009 
  (price_2006 : ℝ) 
  (price_2008 : ℝ) 
  (price_2009 : ℝ) 
  (h1 : price_2008 = price_2006 * (1 + 0.60))
  (h2 : price_2009 = price_2006 * (1 + 0.20)^3) :
  price_2009 = price_2008 * (1 + 0.08) :=
by sorry

end NUMINAMATH_CALUDE_price_increase_2008_2009_l3268_326886


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3268_326801

theorem system_of_equations_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (solutions.card = 8) ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      (x = 2 * y^2 - 1 ∧ y = 2 * z^2 - 1 ∧ z = 2 * x^2 - 1)) :=
by sorry


end NUMINAMATH_CALUDE_system_of_equations_solutions_l3268_326801


namespace NUMINAMATH_CALUDE_remainder_theorem_l3268_326897

theorem remainder_theorem (n : ℤ) (k : ℤ) : 
  (∃ k, n = 100 * k - 2) → (n^2 + 4*n + 10) % 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3268_326897


namespace NUMINAMATH_CALUDE_square_root_difference_l3268_326896

theorem square_root_difference : Real.sqrt (49 + 36) - Real.sqrt (36 - 0) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_l3268_326896


namespace NUMINAMATH_CALUDE_tangents_and_line_of_tangency_l3268_326829

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

-- Define point P
def P : ℝ × ℝ := (-2, 3)

-- Define the tangent lines
def tangent1 (x y : ℝ) : Prop := (Real.sqrt 3 + 6) * x - 4 * y + 2 * Real.sqrt 3 - 3 = 0
def tangent2 (x y : ℝ) : Prop := (3 + Real.sqrt 3) * x + 4 * y - 6 + 2 * Real.sqrt 3 = 0

-- Define the line passing through points of tangency
def tangencyLine (x y : ℝ) : Prop := 3 * x - 2 * y - 3 = 0

theorem tangents_and_line_of_tangency :
  ∃ (M N : ℝ × ℝ),
    M ∈ C ∧ N ∈ C ∧
    (tangent1 M.1 M.2 ∨ tangent2 M.1 M.2) ∧
    (tangent1 N.1 N.2 ∨ tangent2 N.1 N.2) ∧
    tangencyLine M.1 M.2 ∧
    tangencyLine N.1 N.2 :=
by sorry

end NUMINAMATH_CALUDE_tangents_and_line_of_tangency_l3268_326829
