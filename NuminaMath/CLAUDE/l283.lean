import Mathlib

namespace NUMINAMATH_CALUDE_parabola_directrix_l283_28375

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop :=
  y = -17/4

/-- Theorem: The directrix of the given parabola is y = -17/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l283_28375


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l283_28318

/-- A quadratic equation in x is of the form ax² + bx + c = 0 where a ≠ 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α
  a_nonzero : a ≠ 0

/-- The condition for ax² + bx + c = 0 to be a quadratic equation in x -/
theorem quadratic_equation_condition {α : Type*} [Field α] (a b c : α) :
  ∃ (eq : QuadraticEquation α), eq.a = a ∧ eq.b = b ∧ eq.c = c ↔ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l283_28318


namespace NUMINAMATH_CALUDE_inequality_solution_solution_set_complete_l283_28303

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 + 3*x + 4 < 0

-- Define the solution set
def solution_set : Set ℝ := {x | x > 4 ∨ x < -1}

-- Theorem stating that the solution set satisfies the inequality
theorem inequality_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

-- Theorem stating that the solution set is complete
theorem solution_set_complete :
  ∀ x : ℝ, inequality x → x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_solution_set_complete_l283_28303


namespace NUMINAMATH_CALUDE_total_wheels_eq_160_l283_28381

/-- The number of bicycles -/
def num_bicycles : ℕ := 50

/-- The number of tricycles -/
def num_tricycles : ℕ := 20

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- The total number of wheels on all bicycles and tricycles -/
def total_wheels : ℕ := num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_160 : total_wheels = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_eq_160_l283_28381


namespace NUMINAMATH_CALUDE_speed_in_miles_per_minute_l283_28332

-- Define the speed in kilometers per hour
def speed_km_per_hour : ℝ := 600

-- Define the conversion factor from km to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem to prove
theorem speed_in_miles_per_minute :
  (speed_km_per_hour * km_to_miles) / minutes_per_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_miles_per_minute_l283_28332


namespace NUMINAMATH_CALUDE_norma_laundry_problem_l283_28320

theorem norma_laundry_problem (t : ℕ) (s : ℕ) : 
  s = 2 * t →                    -- Twice as many sweaters as T-shirts
  (t + s) - (t + 3) = 15 →       -- 15 items missing
  t = 9 :=                       -- Prove that t (number of T-shirts) is 9
by sorry

end NUMINAMATH_CALUDE_norma_laundry_problem_l283_28320


namespace NUMINAMATH_CALUDE_x_coordinate_is_nineteen_thirds_l283_28359

/-- The x-coordinate of a point on a line --/
def x_coordinate_on_line (x1 y1 x2 y2 y : ℚ) : ℚ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  (y - b) / m

/-- Theorem: The x-coordinate of the point (x, 8) on the line passing through (3, 3) and (1, 0) is 19/3 --/
theorem x_coordinate_is_nineteen_thirds :
  x_coordinate_on_line 3 3 1 0 8 = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_is_nineteen_thirds_l283_28359


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l283_28369

theorem matrix_equation_solution (N : Matrix (Fin 2) (Fin 2) ℝ) :
  N * !![2, -3; 4, -1] = !![-8, 7; 20, -11] →
  N = !![-20, -10; 24, 38] := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l283_28369


namespace NUMINAMATH_CALUDE_game_ends_in_22_rounds_l283_28389

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : Vector Player 3
  round : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that the game ends after exactly 22 rounds -/
theorem game_ends_in_22_rounds :
  let initialState : GameState := {
    players := ⟨[{tokens := 16}, {tokens := 15}, {tokens := 14}], rfl⟩,
    round := 0
  }
  ∃ (finalState : GameState),
    finalState.round = 22 ∧
    gameEnded finalState ∧
    (∀ (intermediateState : GameState),
      intermediateState.round < 22 →
      ¬gameEnded intermediateState) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_in_22_rounds_l283_28389


namespace NUMINAMATH_CALUDE_sin_alpha_value_l283_28322

theorem sin_alpha_value (α β : Real) (a b : Fin 2 → Real) :
  a 0 = Real.cos α ∧ a 1 = Real.sin α ∧
  b 0 = Real.cos β ∧ b 1 = Real.sin β ∧
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 2 * Real.sqrt 5 / 5 ∧
  0 < α ∧ α < Real.pi / 2 ∧
  -Real.pi / 2 < β ∧ β < 0 ∧
  Real.cos (5 * Real.pi / 2 - β) = -5 / 13 →
  Real.sin α = 33 / 65 := by sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l283_28322


namespace NUMINAMATH_CALUDE_original_ratio_first_term_l283_28346

/-- Given a ratio where the second term is 11, and adding 5 to both terms results
    in a ratio of 3:4, prove that the first term of the original ratio is 7. -/
theorem original_ratio_first_term :
  ∀ x : ℚ,
  (x + 5) / (11 + 5) = 3 / 4 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_original_ratio_first_term_l283_28346


namespace NUMINAMATH_CALUDE_perspective_square_area_l283_28362

/-- A square whose perspective drawing is a parallelogram -/
structure PerspectiveSquare where
  /-- The side length of the parallelogram in the perspective drawing -/
  parallelogram_side : ℝ
  /-- The side length of the original square -/
  square_side : ℝ

/-- The theorem stating the possible areas of the square -/
theorem perspective_square_area (s : PerspectiveSquare) (h : s.parallelogram_side = 4) :
  s.square_side ^ 2 = 16 ∨ s.square_side ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_perspective_square_area_l283_28362


namespace NUMINAMATH_CALUDE_coin_array_problem_l283_28331

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The problem statement -/
theorem coin_array_problem :
  ∃ (N : ℕ), triangular_sum N = 3003 ∧ sum_of_digits N = 14 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l283_28331


namespace NUMINAMATH_CALUDE_return_trip_times_l283_28367

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- wind speed
  time_against_wind : ℝ  -- time flying against the wind

/-- Conditions of the flight scenario -/
def flight_conditions (scenario : FlightScenario) : Prop :=
  scenario.time_against_wind = 1.4 ∧  -- 1.4 hours against the wind
  scenario.d = 1.4 * (scenario.p - scenario.w) ∧  -- distance equation
  scenario.d / (scenario.p + scenario.w) = scenario.d / scenario.p - 0.25  -- return trip equation

/-- Theorem stating the possible return trip times -/
theorem return_trip_times (scenario : FlightScenario) 
  (h : flight_conditions scenario) : 
  (scenario.d / (scenario.p + scenario.w) = 12 / 60) ∨ 
  (scenario.d / (scenario.p + scenario.w) = 69 / 60) := by
  sorry


end NUMINAMATH_CALUDE_return_trip_times_l283_28367


namespace NUMINAMATH_CALUDE_carousel_candy_leftover_l283_28390

theorem carousel_candy_leftover (num_clowns num_children num_parents num_vendors : ℕ)
  (initial_supply leftover_candies : ℕ)
  (clown_candies child_candies parent_candies vendor_candies : ℕ)
  (prize_candies bulk_purchase_children bulk_purchase_candies : ℕ)
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : num_parents = 10)
  (h4 : num_vendors = 5)
  (h5 : initial_supply = 2000)
  (h6 : leftover_candies = 700)
  (h7 : clown_candies = 10)
  (h8 : child_candies = 20)
  (h9 : parent_candies = 15)
  (h10 : vendor_candies = 25)
  (h11 : prize_candies = 150)
  (h12 : bulk_purchase_children = 20)
  (h13 : bulk_purchase_candies = 350) :
  initial_supply - (num_clowns * clown_candies + num_children * child_candies +
    num_parents * parent_candies + num_vendors * vendor_candies +
    prize_candies + bulk_purchase_candies) = 685 :=
by sorry

end NUMINAMATH_CALUDE_carousel_candy_leftover_l283_28390


namespace NUMINAMATH_CALUDE_milk_problem_l283_28360

/-- The original amount of milk in liters -/
def original_milk : ℝ := 1.15

/-- The fraction of milk grandmother drank -/
def grandmother_drank : ℝ := 0.4

/-- The amount of milk remaining in liters -/
def remaining_milk : ℝ := 0.69

theorem milk_problem :
  original_milk * (1 - grandmother_drank) = remaining_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_problem_l283_28360


namespace NUMINAMATH_CALUDE_units_digit_base_6_l283_28305

theorem units_digit_base_6 : ∃ (n : ℕ), (67^2 * 324) = 6 * n :=
sorry

end NUMINAMATH_CALUDE_units_digit_base_6_l283_28305


namespace NUMINAMATH_CALUDE_lilly_buys_seven_flowers_l283_28377

/-- Calculates the number of flowers Lilly can buy for Maria's birthday --/
def flowers_for_maria (days : ℕ) (daily_savings : ℕ) (wrapping_cost : ℕ) (other_costs : ℕ) (flower_cost : ℕ) : ℕ :=
  let total_savings := days * daily_savings
  let total_expenses := wrapping_cost + other_costs
  let money_for_flowers := total_savings - total_expenses
  money_for_flowers / flower_cost

/-- Theorem stating that Lilly can buy 7 flowers for Maria's birthday --/
theorem lilly_buys_seven_flowers :
  flowers_for_maria 22 2 6 10 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lilly_buys_seven_flowers_l283_28377


namespace NUMINAMATH_CALUDE_jerrys_shelf_difference_l283_28344

/-- Given Jerry's initial book and action figure counts, and the number of action figures added,
    prove that there are 2 more books than action figures on the shelf. -/
theorem jerrys_shelf_difference (initial_books : ℕ) (initial_figures : ℕ) (added_figures : ℕ)
    (h1 : initial_books = 7)
    (h2 : initial_figures = 3)
    (h3 : added_figures = 2) :
    initial_books - (initial_figures + added_figures) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_shelf_difference_l283_28344


namespace NUMINAMATH_CALUDE_floor_equation_solution_range_l283_28321

theorem floor_equation_solution_range (a : ℝ) : 
  (∃ x : ℕ+, ⌊(x + a) / 3⌋ = 2) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_range_l283_28321


namespace NUMINAMATH_CALUDE_tangent_line_equation_l283_28394

-- Define the function f(x) = x^3 + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the point of interest
def point_of_interest : ℝ := 1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (∃ y : ℝ, y = f point_of_interest ∧
      ∀ x : ℝ, abs (f x - (m * x + b)) ≤ abs (m * (x - point_of_interest)) * abs (x - point_of_interest)) ∧
    m * point_of_interest - f point_of_interest + b = 0 ∧
    m = 4 ∧ b = -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l283_28394


namespace NUMINAMATH_CALUDE_line_curve_intersection_m_bound_l283_28368

/-- Given a straight line and a curve with a common point, prove that m ≥ 3 -/
theorem line_curve_intersection_m_bound (k : ℝ) (m : ℝ) :
  (∃ x y : ℝ, y = k * x - k + 1 ∧ x^2 + 2 * y^2 = m) →
  m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_line_curve_intersection_m_bound_l283_28368


namespace NUMINAMATH_CALUDE_cat_cafe_cool_cats_cat_cafe_cool_cats_proof_l283_28363

theorem cat_cafe_cool_cats : ℕ → ℕ → ℕ → Prop :=
  fun cool paw meow =>
    paw = 2 * cool ∧
    meow = 3 * paw ∧
    meow + paw = 40 →
    cool = 5

-- Proof
theorem cat_cafe_cool_cats_proof : cat_cafe_cool_cats 5 10 30 :=
by
  sorry

end NUMINAMATH_CALUDE_cat_cafe_cool_cats_cat_cafe_cool_cats_proof_l283_28363


namespace NUMINAMATH_CALUDE_polygon_with_five_triangles_has_fourteen_diagonals_l283_28328

/-- A polygon is a shape with a finite number of straight sides. -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 0

/-- The number of triangles formed when drawing diagonals from one vertex of a polygon. -/
def triangles_from_vertex (p : Polygon) : ℕ := p.sides - 2

/-- The number of diagonals in a polygon. -/
def num_diagonals (p : Polygon) : ℕ := p.sides * (p.sides - 3) / 2

/-- Theorem: A polygon that can be divided into at most 5 triangles by drawing diagonals from one vertex has 14 diagonals. -/
theorem polygon_with_five_triangles_has_fourteen_diagonals (p : Polygon) 
  (h : triangles_from_vertex p ≤ 5) : 
  num_diagonals p = 14 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_five_triangles_has_fourteen_diagonals_l283_28328


namespace NUMINAMATH_CALUDE_house_problem_theorem_l283_28312

def house_problem (original_price selling_price years_owned broker_commission_rate
                   property_tax_rate yearly_maintenance mortgage_interest_rate : ℝ) : Prop :=
  let profit_rate := (selling_price - original_price) / original_price
  let broker_commission := broker_commission_rate * original_price
  let total_property_tax := property_tax_rate * original_price * years_owned
  let total_maintenance := yearly_maintenance * years_owned
  let total_mortgage_interest := mortgage_interest_rate * original_price * years_owned
  let total_costs := broker_commission + total_property_tax + total_maintenance + total_mortgage_interest
  let net_profit := selling_price - original_price - total_costs
  (original_price = 80000) ∧
  (years_owned = 5) ∧
  (profit_rate = 0.2) ∧
  (broker_commission_rate = 0.05) ∧
  (property_tax_rate = 0.012) ∧
  (yearly_maintenance = 1500) ∧
  (mortgage_interest_rate = 0.04) →
  net_profit = -16300

theorem house_problem_theorem :
  ∀ (original_price selling_price years_owned broker_commission_rate
     property_tax_rate yearly_maintenance mortgage_interest_rate : ℝ),
  house_problem original_price selling_price years_owned broker_commission_rate
                 property_tax_rate yearly_maintenance mortgage_interest_rate :=
by
  sorry

end NUMINAMATH_CALUDE_house_problem_theorem_l283_28312


namespace NUMINAMATH_CALUDE_students_drawn_from_C_l283_28314

/-- Represents the total number of students in the college -/
def total_students : ℕ := 1500

/-- Represents the planned sample size -/
def sample_size : ℕ := 150

/-- Represents the number of students in major A -/
def students_major_A : ℕ := 420

/-- Represents the number of students in major B -/
def students_major_B : ℕ := 580

/-- Theorem stating the number of students to be drawn from major C -/
theorem students_drawn_from_C : 
  (sample_size : ℚ) / total_students * (total_students - students_major_A - students_major_B) = 50 :=
by sorry

end NUMINAMATH_CALUDE_students_drawn_from_C_l283_28314


namespace NUMINAMATH_CALUDE_percentage_difference_l283_28392

theorem percentage_difference (w u y z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l283_28392


namespace NUMINAMATH_CALUDE_special_circle_six_radii_l283_28353

/-- A circle with integer radius and specific geometric properties -/
structure SpecialCircle where
  -- Center of the circle
  H : ℝ × ℝ
  -- Radius of the circle (which we want to prove has 6 possible integer values)
  r : ℕ
  -- Point F outside the circle
  F : ℝ × ℝ
  -- Point G on the circle and on line FH
  G : ℝ × ℝ
  -- Point I on the circle where FI is tangent
  I : ℝ × ℝ
  -- Distance FG is an integer
  FG_integer : ∃ (n : ℕ), Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) = n
  -- FI = FG + 6
  FI_eq_FG_plus_6 : Real.sqrt ((F.1 - I.1)^2 + (F.2 - I.2)^2) = 
                    Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) + 6
  -- G is on circle
  G_on_circle : (G.1 - H.1)^2 + (G.2 - H.2)^2 = r^2
  -- I is on circle
  I_on_circle : (I.1 - H.1)^2 + (I.2 - H.2)^2 = r^2
  -- FI is tangent to circle at I
  FI_tangent : (F.1 - I.1) * (I.1 - H.1) + (F.2 - I.2) * (I.2 - H.2) = 0
  -- G is on line FH
  G_on_FH : ∃ (t : ℝ), G = (F.1 + t * (H.1 - F.1), F.2 + t * (H.2 - F.2))

/-- There are exactly 6 possible values for the radius of a SpecialCircle -/
theorem special_circle_six_radii : ∃! (S : Finset ℕ), (∀ c : SpecialCircle, c.r ∈ S) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_special_circle_six_radii_l283_28353


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_l283_28326

/-- The curve function f(x) = ax^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2

/-- The derivative of f(x) = ax^2 -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

/-- The slope of the line 4x - y + 4 = 0 -/
def line_slope : ℝ := 4

theorem tangent_perpendicular_implies_a (a : ℝ) :
  f a 2 = 4 * a ∧
  f_derivative a 2 * line_slope = -1 →
  a = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_l283_28326


namespace NUMINAMATH_CALUDE_partition_to_magic_square_l283_28349

/-- Represents a 3x3 square of integers -/
def Square : Type := Fin 3 → Fin 3 → ℤ

/-- Checks if a square is a magic square -/
def isMagicSquare (s : Square) : Prop :=
  let rowSum (i : Fin 3) := (s i 0) + (s i 1) + (s i 2)
  let colSum (j : Fin 3) := (s 0 j) + (s 1 j) + (s 2 j)
  let diagSum1 := (s 0 0) + (s 1 1) + (s 2 2)
  let diagSum2 := (s 0 2) + (s 1 1) + (s 2 0)
  ∀ i j : Fin 3, rowSum i = colSum j ∧ rowSum i = diagSum1 ∧ rowSum i = diagSum2

/-- Represents a partition of numbers from 1 to 360 into 9 subsets -/
def Partition : Type := Fin 9 → List ℕ

/-- Checks if a partition is valid (consecutive integers, sum to 360) -/
def isValidPartition (p : Partition) : Prop :=
  (∀ i : Fin 9, List.Chain' (·+1=·) (p i)) ∧
  (List.sum (List.join (List.ofFn p)) = 360) ∧
  (∀ i : Fin 9, p i ≠ [])

/-- The sum of a subset in the partition -/
def subsetSum (p : Partition) (i : Fin 9) : ℤ :=
  List.sum (p i)

/-- Theorem: It is possible to arrange the sums of a valid partition into a magic square -/
theorem partition_to_magic_square :
  ∃ (p : Partition) (s : Square), isValidPartition p ∧ isMagicSquare s ∧
  ∀ i : Fin 9, ∃ j k : Fin 3, s j k = subsetSum p i :=
sorry

end NUMINAMATH_CALUDE_partition_to_magic_square_l283_28349


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l283_28327

/-- The modulus for our congruences -/
def m : ℕ := 9

/-- First congruence equation -/
def eq1 (x y : ℤ) : Prop := y ≡ 2 * x + 3 [ZMOD m]

/-- Second congruence equation -/
def eq2 (x y : ℤ) : Prop := y ≡ 7 * x + 6 [ZMOD m]

/-- The x-coordinate of the intersection point -/
def intersection_x : ℤ := 3

theorem intersection_point_x_coordinate :
  ∃ y : ℤ, eq1 intersection_x y ∧ eq2 intersection_x y :=
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l283_28327


namespace NUMINAMATH_CALUDE_old_cars_less_than_half_after_three_years_l283_28345

/-- Represents the state of the car fleet after a certain number of years -/
structure FleetState where
  years : ℕ
  oldCars : ℕ
  newCars : ℕ

/-- Updates the fleet state for one year -/
def updateFleet (state : FleetState) : FleetState :=
  { years := state.years + 1,
    oldCars := max (state.oldCars - 5) 0,
    newCars := state.newCars + 6 }

/-- Calculates the fleet state after a given number of years -/
def fleetAfterYears (years : ℕ) : FleetState :=
  (List.range years).foldl (fun state _ => updateFleet state) { years := 0, oldCars := 20, newCars := 0 }

/-- Theorem: After 3 years, the number of old cars is less than 50% of the total fleet -/
theorem old_cars_less_than_half_after_three_years :
  let state := fleetAfterYears 3
  state.oldCars < (state.oldCars + state.newCars) / 2 := by
  sorry


end NUMINAMATH_CALUDE_old_cars_less_than_half_after_three_years_l283_28345


namespace NUMINAMATH_CALUDE_square_difference_l283_28364

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l283_28364


namespace NUMINAMATH_CALUDE_problem_statement_l283_28385

theorem problem_statement (x₁ x₂ : ℝ) (h1 : |x₁ - 2| < 1) (h2 : |x₂ - 2| < 1) (h3 : x₁ ≠ x₂) : 
  let f := fun x => x^2 - x + 1
  2 < x₁ + x₂ ∧ x₁ + x₂ < 6 ∧ 
  |x₁ - x₂| < 2 ∧
  |x₁ - x₂| < |f x₁ - f x₂| ∧ |f x₁ - f x₂| < 5 * |x₁ - x₂| := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l283_28385


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_root_l283_28317

theorem quadratic_equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_root_l283_28317


namespace NUMINAMATH_CALUDE_solution_value_l283_28307

/-- The function F as defined in the problem -/
def F (a b c : ℚ) : ℚ := a * b^3 + c

/-- Theorem stating that -5/19 is the solution to the equation -/
theorem solution_value : ∃ a : ℚ, F a 3 8 = F a 2 3 ∧ a = -5/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l283_28307


namespace NUMINAMATH_CALUDE_min_value_expression_l283_28347

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 ∧
  (∃ x₀ : ℝ, x₀ > 1 ∧ (x₀ + 10) / Real.sqrt (x₀ - 1) = 2 * Real.sqrt 11 ∧ x₀ = 12) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l283_28347


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l283_28341

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = Set.Icc (-1) 2 := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a > 5 ∨ a < -3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l283_28341


namespace NUMINAMATH_CALUDE_smallest_seven_digit_divisible_l283_28330

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

theorem smallest_seven_digit_divisible :
  (is_seven_digit 7207200) ∧
  (is_divisible_by 7207200 35) ∧
  (is_divisible_by 7207200 112) ∧
  (is_divisible_by 7207200 175) ∧
  (is_divisible_by 7207200 288) ∧
  (is_divisible_by 7207200 429) ∧
  (is_divisible_by 7207200 528) ∧
  (∀ n : ℕ, (is_seven_digit n) ∧
            (is_divisible_by n 35) ∧
            (is_divisible_by n 112) ∧
            (is_divisible_by n 175) ∧
            (is_divisible_by n 288) ∧
            (is_divisible_by n 429) ∧
            (is_divisible_by n 528) →
            n ≥ 7207200) :=
by sorry

end NUMINAMATH_CALUDE_smallest_seven_digit_divisible_l283_28330


namespace NUMINAMATH_CALUDE_right_triangle_area_l283_28333

theorem right_triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l283_28333


namespace NUMINAMATH_CALUDE_pencil_sharpening_l283_28340

theorem pencil_sharpening (original_length sharpened_length : ℕ) 
  (h1 : original_length = 31)
  (h2 : sharpened_length = 14) :
  original_length - sharpened_length = 17 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l283_28340


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l283_28365

theorem sqrt_nine_factorial_over_ninety : 
  Real.sqrt (Nat.factorial 9 / 90) = 4 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l283_28365


namespace NUMINAMATH_CALUDE_parabola_vertex_l283_28382

/-- The vertex of a parabola given by the equation y^2 + 8y + 4x + 9 = 0 is (7/4, -4) -/
theorem parabola_vertex (x y : ℝ) : 
  y^2 + 8*y + 4*x + 9 = 0 → (x, y) = (7/4, -4) ∨ ∃ t : ℝ, (x, y) = (7/4 - t^2, -4 + t) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l283_28382


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l283_28384

/-- Represents a multiple choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- The number of ways to complete a test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question,
    there is exactly 1 way to complete it with all questions unanswered -/
theorem unanswered_test_completion_ways 
  (test : MultipleChoiceTest)
  (h1 : test.num_questions = 4)
  (h2 : test.choices_per_question = 5) :
  ways_to_complete_unanswered test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l283_28384


namespace NUMINAMATH_CALUDE_latte_cost_is_2_50_l283_28355

/-- The cost of Sean's Sunday purchases -/
def seans_purchase (latte_cost : ℚ) : Prop :=
  let almond_croissant := (4.5 : ℚ)
  let salami_cheese_croissant := (4.5 : ℚ)
  let plain_croissant := (3 : ℚ)
  let focaccia := (4 : ℚ)
  let num_lattes := (2 : ℚ)
  let total_spent := (21 : ℚ)
  almond_croissant + salami_cheese_croissant + plain_croissant + focaccia + num_lattes * latte_cost = total_spent

/-- Theorem stating that each latte costs $2.50 -/
theorem latte_cost_is_2_50 : ∃ (latte_cost : ℚ), seans_purchase latte_cost ∧ latte_cost = (2.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_latte_cost_is_2_50_l283_28355


namespace NUMINAMATH_CALUDE_natural_number_representation_l283_28319

def representable (n : ℕ) : Prop :=
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  (x * y + y * z + z * x : ℚ) / (x + y + z : ℚ) = n

theorem natural_number_representation :
  ∀ n : ℕ, n > 1 → representable n ∧ ¬representable 1 := by sorry

end NUMINAMATH_CALUDE_natural_number_representation_l283_28319


namespace NUMINAMATH_CALUDE_equation_equivalence_and_solutions_l283_28315

theorem equation_equivalence_and_solutions (x y z : ℤ) : 
  (x * (x - y) + y * (y - x) + z * (z - y) = 1) ↔ 
  ((x - y)^2 + (z - y)^2 = 1) ∧
  ((x = y - 1 ∧ z = y + 1) ∨ (x = y ∧ z = y + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_and_solutions_l283_28315


namespace NUMINAMATH_CALUDE_martas_fruit_bags_l283_28350

/-- Proves that each bag can hold 49 ounces of fruit given the conditions of Marta's fruit purchase --/
theorem martas_fruit_bags 
  (apple_weight : ℕ) 
  (orange_weight : ℕ) 
  (num_bags : ℕ) 
  (total_apple_weight : ℕ) : 
  apple_weight = 4 →
  orange_weight = 3 →
  num_bags = 3 →
  total_apple_weight = 84 →
  ∃ (bag_capacity : ℕ),
    bag_capacity = 49 ∧
    (total_apple_weight / apple_weight) * (apple_weight + orange_weight) = num_bags * bag_capacity :=
by sorry

end NUMINAMATH_CALUDE_martas_fruit_bags_l283_28350


namespace NUMINAMATH_CALUDE_nickel_chocolates_l283_28376

theorem nickel_chocolates (robert : ℕ) (difference : ℕ) (h1 : robert = 12) (h2 : difference = 9) :
  robert - difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l283_28376


namespace NUMINAMATH_CALUDE_fraction_transformation_l283_28383

theorem fraction_transformation (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 2 / 5 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l283_28383


namespace NUMINAMATH_CALUDE_total_socks_is_51_l283_28313

/-- Calculates the total number of socks John and Mary have after throwing away and buying new socks -/
def totalSocksAfterChanges (johnInitial : ℕ) (maryInitial : ℕ) (johnThrowAway : ℕ) (johnBuy : ℕ) (maryThrowAway : ℕ) (maryBuy : ℕ) : ℕ :=
  (johnInitial - johnThrowAway + johnBuy) + (maryInitial - maryThrowAway + maryBuy)

/-- Proves that John and Mary have 51 socks in total after the changes -/
theorem total_socks_is_51 :
  totalSocksAfterChanges 33 20 19 13 6 10 = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_socks_is_51_l283_28313


namespace NUMINAMATH_CALUDE_well_depth_solution_l283_28354

def well_depth_problem (d : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    d = 20 * t₁^2 ∧  -- Stone's fall distance
    d = 1100 * t₂ ∧  -- Sound's travel distance
    t₁ + t₂ = 10 ∧   -- Total time
    d = 122500       -- Depth to prove

theorem well_depth_solution :
  ∃ d : ℝ, well_depth_problem d :=
sorry

end NUMINAMATH_CALUDE_well_depth_solution_l283_28354


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l283_28379

theorem arithmetic_expression_equality : 2 + 3 * 5 + 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l283_28379


namespace NUMINAMATH_CALUDE_blue_balls_count_l283_28397

theorem blue_balls_count (total : ℕ) (p : ℚ) (h_total : total = 12) (h_prob : p = 1 / 22) :
  ∃ b : ℕ, b ≤ total ∧ 
    (b : ℚ) * (b - 1) / (total * (total - 1)) = p ∧
    b = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l283_28397


namespace NUMINAMATH_CALUDE_arithmetic_sequence_probability_l283_28358

/-- The set of numbers from which we select -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

/-- Predicate to check if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop := 2 * b = a + c

/-- The number of ways to select 3 different numbers from S -/
def totalSelections : ℕ := Nat.choose 20 3

/-- The number of ways to select 3 different numbers from S that form an arithmetic sequence -/
def validSelections : ℕ := 90

/-- The probability of selecting 3 different numbers from S that form an arithmetic sequence -/
def probability : ℚ := validSelections / totalSelections

theorem arithmetic_sequence_probability : probability = 1 / 38 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_probability_l283_28358


namespace NUMINAMATH_CALUDE_project_exceeds_budget_l283_28356

def field_area : ℝ := 3136
def wire_cost_per_meter : ℝ := 1.10
def gate_width : ℝ := 1
def gate_height : ℝ := 2
def iron_cost_per_kg : ℝ := 350
def gate_weight : ℝ := 25
def labor_cost_per_day : ℝ := 1500
def work_days : ℝ := 2
def budget : ℝ := 10000

theorem project_exceeds_budget :
  let field_side := Real.sqrt field_area
  let perimeter := 4 * field_side
  let wire_length := perimeter - 2 * gate_width
  let wire_cost := wire_length * wire_cost_per_meter
  let gates_cost := 2 * gate_weight * iron_cost_per_kg
  let labor_cost := work_days * labor_cost_per_day
  let total_cost := wire_cost + gates_cost + labor_cost
  total_cost > budget := by sorry

end NUMINAMATH_CALUDE_project_exceeds_budget_l283_28356


namespace NUMINAMATH_CALUDE_dance_lesson_cost_l283_28302

/-- The cost of each dance lesson given the total number of lessons, 
    number of free lessons, and total cost paid. -/
theorem dance_lesson_cost 
  (total_lessons : ℕ) 
  (free_lessons : ℕ) 
  (total_cost : ℚ) : 
  total_lessons = 10 → 
  free_lessons = 2 → 
  total_cost = 80 → 
  (total_cost / (total_lessons - free_lessons : ℚ)) = 10 := by
sorry

end NUMINAMATH_CALUDE_dance_lesson_cost_l283_28302


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l283_28300

theorem octagon_area_in_circle (r : ℝ) (h : r = 2) :
  let octagon_area := 8 * r^2 * Real.sin (π / 4)
  octagon_area = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l283_28300


namespace NUMINAMATH_CALUDE_test_subjects_count_l283_28361

def choose (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def number_of_colors : ℕ := 5
def colors_per_code : ℕ := 2
def unidentified_subjects : ℕ := 6

theorem test_subjects_count : 
  choose number_of_colors colors_per_code + unidentified_subjects = 16 := by
  sorry

end NUMINAMATH_CALUDE_test_subjects_count_l283_28361


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l283_28324

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking_height * water_density

/-- Theorem stating the mass of the man in the given problem. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 9
  let boat_breadth : ℝ := 3
  let boat_sinking_height : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth boat_sinking_height water_density = 270 :=
by sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l283_28324


namespace NUMINAMATH_CALUDE_number_difference_proof_l283_28334

theorem number_difference_proof :
  ∃ x : ℚ, (1 / 3 : ℚ) * x - (1 / 4 : ℚ) * x = 3 ∧ x = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l283_28334


namespace NUMINAMATH_CALUDE_no_valid_grid_exists_l283_28370

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if two positions in the grid are adjacent -/
def isAdjacent (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j'.val + 1 = j.val)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i'.val + 1 = i.val))

/-- Checks if a grid satisfies the prime sum condition -/
def satisfiesPrimeSum (g : Grid) : Prop :=
  ∀ i j i' j' : Fin 3, isAdjacent i j i' j' → isPrime (g i j + g i' j')

/-- Checks if a grid contains all numbers from 1 to 9 exactly once -/
def containsAllNumbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃! i j : Fin 3, g i j = n.val + 1

/-- The main theorem stating that no valid grid exists -/
theorem no_valid_grid_exists : ¬∃ g : Grid, satisfiesPrimeSum g ∧ containsAllNumbers g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_exists_l283_28370


namespace NUMINAMATH_CALUDE_math_book_cost_l283_28372

-- Define the total number of books
def total_books : ℕ := 90

-- Define the number of math books
def math_books : ℕ := 53

-- Define the cost of each history book
def history_book_cost : ℕ := 5

-- Define the total price of all books
def total_price : ℕ := 397

-- Theorem to prove
theorem math_book_cost :
  ∃ (x : ℕ), x * math_books + (total_books - math_books) * history_book_cost = total_price ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_math_book_cost_l283_28372


namespace NUMINAMATH_CALUDE_concyclicity_theorem_l283_28378

-- Define the points
variable (A B C D A' B' C' D' : Point)

-- Define the concyclicity property
def areConcyclic (P Q R S : Point) : Prop := sorry

-- State the theorem
theorem concyclicity_theorem 
  (h1 : areConcyclic A B C D)
  (h2 : areConcyclic A A' B B')
  (h3 : areConcyclic B B' C C')
  (h4 : areConcyclic C C' D D')
  (h5 : areConcyclic D D' A A') :
  areConcyclic A' B' C' D' := by sorry

end NUMINAMATH_CALUDE_concyclicity_theorem_l283_28378


namespace NUMINAMATH_CALUDE_cookie_combinations_theorem_l283_28373

/-- The number of different combinations of cookies that can be purchased
    given the specified conditions. -/
def cookieCombinations : ℕ := 34

/-- The number of types of cookies available. -/
def cookieTypes : ℕ := 4

/-- The total number of cookies to be purchased. -/
def totalCookies : ℕ := 8

/-- The minimum number of each type of cookie to be purchased. -/
def minEachType : ℕ := 1

theorem cookie_combinations_theorem :
  (cookieTypes = 4) →
  (totalCookies = 8) →
  (minEachType = 1) →
  (cookieCombinations = 34) := by sorry

end NUMINAMATH_CALUDE_cookie_combinations_theorem_l283_28373


namespace NUMINAMATH_CALUDE_union_P_complement_Q_l283_28374

def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

theorem union_P_complement_Q : P ∪ (Set.univ \ Q) = {x : ℝ | -2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_P_complement_Q_l283_28374


namespace NUMINAMATH_CALUDE_scored_at_least_once_and_not_scored_both_times_mutually_exclusive_l283_28357

-- Define the sample space for two shots
inductive ShotOutcome
  | Score
  | Miss

-- Define the event of scoring at least once
def scoredAtLeastOnce (outcome : ShotOutcome × ShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Score ∨ outcome.2 = ShotOutcome.Score

-- Define the event of not scoring both times
def notScoredBothTimes (outcome : ShotOutcome × ShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∨ outcome.2 = ShotOutcome.Miss

-- Theorem stating that the events are mutually exclusive
theorem scored_at_least_once_and_not_scored_both_times_mutually_exclusive :
  ∀ (outcome : ShotOutcome × ShotOutcome),
    ¬(scoredAtLeastOnce outcome ∧ notScoredBothTimes outcome) := by
  sorry


end NUMINAMATH_CALUDE_scored_at_least_once_and_not_scored_both_times_mutually_exclusive_l283_28357


namespace NUMINAMATH_CALUDE_multiplication_correction_l283_28309

theorem multiplication_correction (a b c d e : ℕ) : 
  (a = 4 ∧ b = 5 ∧ c = 4 ∧ d = 5 ∧ e = 4) →
  (a * b * c * d * e = 2247) →
  ∃ (x : ℕ), (x = 5 ∨ x = 7) ∧
    ((4 * x * 4 * 5 * 4 = 2240) ∨ (4 * 5 * 4 * x * 4 = 2240)) :=
by sorry

end NUMINAMATH_CALUDE_multiplication_correction_l283_28309


namespace NUMINAMATH_CALUDE_fib_consecutive_coprime_fib_gcd_l283_28323

/-- Definition of Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- Theorem: Consecutive Fibonacci numbers are coprime -/
theorem fib_consecutive_coprime (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by sorry

/-- Theorem: GCD of Fibonacci numbers -/
theorem fib_gcd (m n : ℕ) : Nat.gcd (fib m) (fib n) = fib (Nat.gcd m n) := by sorry

end NUMINAMATH_CALUDE_fib_consecutive_coprime_fib_gcd_l283_28323


namespace NUMINAMATH_CALUDE_tuition_calculation_l283_28391

theorem tuition_calculation (discount : ℕ) (total_cost : ℕ) : 
  discount = 15 → total_cost = 75 → (total_cost + discount) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_tuition_calculation_l283_28391


namespace NUMINAMATH_CALUDE_find_set_B_l283_28338

open Set

def U : Set ℕ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℕ := {1, 2, 3}

theorem find_set_B : 
  ∃! B : Set ℕ, (U \ (A ∩ B) = {1, 2, 4, 5, 6, 7}) ∧ B ⊆ U ∧ B = {3, 4, 5} :=
sorry

end NUMINAMATH_CALUDE_find_set_B_l283_28338


namespace NUMINAMATH_CALUDE_intersection_segment_length_l283_28343

-- Define the polar curve C₁
def C₁ (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 8

-- Define the line l
def l (t x y : ℝ) : Prop := x = 1 + (Real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t

-- Define the Cartesian form of C₁
def C₁_cartesian (x y : ℝ) : Prop := x^2 - y^2 = 8

-- Theorem statement
theorem intersection_segment_length :
  ∃ (t₁ t₂ : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      l t₁ x₁ y₁ ∧ l t₂ x₂ y₂ ∧
      C₁_cartesian x₁ y₁ ∧ C₁_cartesian x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 68) :=
sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l283_28343


namespace NUMINAMATH_CALUDE_gcd_max_digits_l283_28352

theorem gcd_max_digits (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 →
  10000000 ≤ b ∧ b < 100000000 →
  Nat.lcm a b = 1000000000000 →
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_max_digits_l283_28352


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l283_28366

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_product (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a (n + 1) > a n) →
  a 6 * a 7 = 15 →
  a 1 = 2 →
  a 4 * a 9 = 234 / 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l283_28366


namespace NUMINAMATH_CALUDE_locus_of_points_l283_28388

noncomputable def circle_locus (d : ℝ) : Set (ℝ × ℝ) :=
  if 0 < d ∧ d < 0.5 then {p : ℝ × ℝ | p.1^2 + p.2^2 = 1 - 2*d}
  else if d = 0.5 then {(0, 0)}
  else ∅

theorem locus_of_points (k : Set (ℝ × ℝ)) (O P Q : ℝ × ℝ) (d : ℝ) :
  (k = {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = 1}) →
  (O = (0, 0)) →
  (P = (-1, 0)) →
  (Q = (1, 0)) →
  (∀ e f : ℝ, 
    (abs (e - f) = d) → 
    (∀ E₁ F₁ F₂ : ℝ × ℝ,
      (E₁ ∈ k ∧ E₁.1 = e) →
      ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (E₁.1 - P.1)^2 + (E₁.2 - P.2)^2) →
      (F₁.1 = f) →
      ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (E₁.1 - P.1)^2 + (E₁.2 - P.2)^2) →
      (F₂.1 = f) →
      (F₁ ∈ circle_locus d ∧ F₂ ∈ circle_locus d))) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_points_l283_28388


namespace NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l283_28306

/-- For an arithmetic sequence with common difference d ≠ 0, 
    if a_3 + a_9 = a_10 - a_8, then a_n = 0 when n = 5 -/
theorem arithmetic_sequence_zero_term 
  (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_d_neq_0 : d ≠ 0) 
  (h_eq : a 3 + a 9 = a 10 - a 8) :
  ∃ n, a n = 0 ∧ n = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l283_28306


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l283_28393

theorem complex_arithmetic_equation : 
  -4^2 * ((1 - 7) / 6)^3 + ((-5)^3 - 3) / (-2)^3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l283_28393


namespace NUMINAMATH_CALUDE_vector_decomposition_l283_28396

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![11, -1, 4]
def p : Fin 3 → ℝ := ![1, -1, 2]
def q : Fin 3 → ℝ := ![3, 2, 0]
def r : Fin 3 → ℝ := ![-1, 1, 1]

/-- Theorem stating the decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = λ i => 3 * p i + 2 * q i - 2 * r i := by
  sorry


end NUMINAMATH_CALUDE_vector_decomposition_l283_28396


namespace NUMINAMATH_CALUDE_quadratic_inequality_relationship_l283_28348

theorem quadratic_inequality_relationship (x : ℝ) :
  (∀ x, x^2 - 2*x < 0 → 0 < x ∧ x < 4) ∧
  (∃ x, 0 < x ∧ x < 4 ∧ ¬(x^2 - 2*x < 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relationship_l283_28348


namespace NUMINAMATH_CALUDE_total_yards_two_days_l283_28342

-- Define the basic throw distance at 50°F
def base_distance : ℕ := 20

-- Define the temperature effect
def temp_effect (temp : ℕ) : ℕ → ℕ :=
  λ d => if temp = 80 then 2 * d else d

-- Define the wind effect
def wind_effect (wind_speed : ℤ) : ℕ → ℤ :=
  λ d => d + wind_speed * 5 / 10

-- Calculate total distance for a day
def total_distance (temp : ℕ) (wind_speed : ℤ) (throws : ℕ) : ℕ :=
  (wind_effect wind_speed (temp_effect temp base_distance)).toNat * throws

-- Theorem statement
theorem total_yards_two_days :
  total_distance 50 (-1) 20 + total_distance 80 3 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_yards_two_days_l283_28342


namespace NUMINAMATH_CALUDE_lawn_length_is_70_l283_28387

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  roadWidth : ℝ
  costPerSquareMeter : ℝ
  totalCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width

/-- Theorem stating the length of the lawn given specific conditions -/
theorem lawn_length_is_70 (l : LawnWithRoads) 
  (h1 : l.width = 50)
  (h2 : l.roadWidth = 10)
  (h3 : l.costPerSquareMeter = 3)
  (h4 : l.totalCost = 3600)
  (h5 : roadArea l = l.totalCost / l.costPerSquareMeter) :
  l.length = 70 := by
  sorry


end NUMINAMATH_CALUDE_lawn_length_is_70_l283_28387


namespace NUMINAMATH_CALUDE_number_problem_l283_28399

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 14 → (40/100 : ℝ) * N = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l283_28399


namespace NUMINAMATH_CALUDE_plane_perpendicular_sufficient_condition_l283_28336

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (containedIn : Line → Plane → Prop)

-- Define the intersecting relation for lines
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem plane_perpendicular_sufficient_condition
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : m ≠ n)
  (h2 : containedIn m α)
  (h3 : containedIn n α)
  (h4 : containedIn l₁ β)
  (h5 : containedIn l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : perpendicular l₁ m)
  (h8 : perpendicular l₂ m) :
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_sufficient_condition_l283_28336


namespace NUMINAMATH_CALUDE_smallest_y_congruence_l283_28311

theorem smallest_y_congruence : 
  ∃ y : ℕ, y > 0 ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, z > 0 → (26 * z + 8) % 16 = 4 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_congruence_l283_28311


namespace NUMINAMATH_CALUDE_count_even_factors_l283_28337

def n : ℕ := 2^3 * 3^2 * 7^1 * 11^1

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 36 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l283_28337


namespace NUMINAMATH_CALUDE_double_radius_ellipse_iff_l283_28371

/-- An ellipse is a "double-radius ellipse" if there exists a point P on the ellipse
    such that the ratio of the distances from P to the two foci is 2:1 -/
def is_double_radius_ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧ a ≤ 3 * Real.sqrt (a^2 - b^2)

/-- Theorem: Characterization of a double-radius ellipse -/
theorem double_radius_ellipse_iff (a b : ℝ) :
  is_double_radius_ellipse a b ↔ 
  (a > 0 ∧ b > 0 ∧ a ≥ b) ∧ (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ 
    ∃ (d1 d2 : ℝ), d1 = 2*d2 ∧ 
      d1 + d2 = 2*a ∧
      d1^2 = (x + Real.sqrt (a^2 - b^2))^2 + y^2 ∧
      d2^2 = (x - Real.sqrt (a^2 - b^2))^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_double_radius_ellipse_iff_l283_28371


namespace NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_sufficient_not_necessary_l283_28335

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem 1
theorem range_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

-- Theorem 2
theorem range_a_for_sufficient_not_necessary :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_sufficient_not_necessary_l283_28335


namespace NUMINAMATH_CALUDE_range_of_a_l283_28304

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-2) 3 ∧ 2 * x - x^2 ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l283_28304


namespace NUMINAMATH_CALUDE_line_through_point_l283_28329

theorem line_through_point (k : ℚ) : 
  (2 * k * (-3/2) - 3 * 4 = k + 2) → k = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l283_28329


namespace NUMINAMATH_CALUDE_irreducible_polynomial_l283_28339

def S : Finset ℕ := {54, 72, 36, 108}

def is_permutation (b₀ b₁ b₂ b₃ : ℕ) : Prop :=
  {b₀, b₁, b₂, b₃} = S

def polynomial (b₀ b₁ b₂ b₃ : ℕ) (x : ℤ) : ℤ :=
  x^5 + b₃*x^3 + b₂*x^2 + b₁*x + b₀

theorem irreducible_polynomial (b₀ b₁ b₂ b₃ : ℕ) :
  is_permutation b₀ b₁ b₂ b₃ →
  Irreducible (polynomial b₀ b₁ b₂ b₃) :=
by sorry

end NUMINAMATH_CALUDE_irreducible_polynomial_l283_28339


namespace NUMINAMATH_CALUDE_pool_water_increase_l283_28380

theorem pool_water_increase (total_capacity : ℝ) (additional_water : ℝ) 
  (h1 : total_capacity = 1857.1428571428573)
  (h2 : additional_water = 300)
  (h3 : additional_water + (total_capacity * 0.7 - additional_water) = total_capacity * 0.7) :
  let initial_water := total_capacity * 0.7 - additional_water
  let percentage_increase := (additional_water / initial_water) * 100
  percentage_increase = 30 := by
sorry

end NUMINAMATH_CALUDE_pool_water_increase_l283_28380


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_81_l283_28316

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_81_l283_28316


namespace NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l283_28398

theorem sum_of_a_values_for_single_solution (a : ℝ) : 
  let equation := fun (x : ℝ) ↦ 3 * x^2 + a * x + 12 * x + 16
  let discriminant := (a + 12)^2 - 4 * 3 * 16
  (∃! x, equation x = 0) → 
  (∃ a₁ a₂, a₁ + a₂ = -24 ∧ discriminant = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l283_28398


namespace NUMINAMATH_CALUDE_find_q_l283_28301

-- Define the polynomial Q(x)
def Q (x p q d : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- Define the properties of the polynomial
def polynomial_properties (p q d : ℝ) : Prop :=
  let zeros_sum := -p
  let zeros_product := -d
  let coefficients_sum := 1 + p + q + d
  (zeros_sum / 3 = zeros_product) ∧ (zeros_product = coefficients_sum)

-- Theorem statement
theorem find_q : 
  ∀ p q d : ℝ, polynomial_properties p q d → d = 5 → q = -26 :=
by sorry

end NUMINAMATH_CALUDE_find_q_l283_28301


namespace NUMINAMATH_CALUDE_abc_inequalities_l283_28308

theorem abc_inequalities (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l283_28308


namespace NUMINAMATH_CALUDE_supermarket_comparison_l283_28351

/-- Supermarket A discount function -/
def cost_a (x : ℝ) : ℝ := 200 + 0.8 * (x - 200)

/-- Supermarket B discount function -/
def cost_b (x : ℝ) : ℝ := 100 + 0.85 * (x - 100)

theorem supermarket_comparison :
  (∀ x > 200, cost_a x = 200 + 0.8 * (x - 200)) ∧
  (∀ x > 100, cost_b x = 100 + 0.85 * (x - 100)) →
  (cost_b 300 < cost_a 300) ∧
  (∃ x > 200, cost_a x = cost_b x) ∧
  (cost_a 500 = cost_b 500) := by
  sorry

end NUMINAMATH_CALUDE_supermarket_comparison_l283_28351


namespace NUMINAMATH_CALUDE_three_Y_five_l283_28395

-- Define the operation Y
def Y (a b : ℤ) : ℤ := 3*b + 8*a - a^2

-- Theorem to prove
theorem three_Y_five : Y 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_Y_five_l283_28395


namespace NUMINAMATH_CALUDE_guido_cost_calculation_l283_28310

def lightning_cost : ℝ := 140000

def mater_cost : ℝ := 0.1 * lightning_cost

def sally_cost_before_mod : ℝ := 3 * mater_cost

def sally_cost_after_mod : ℝ := sally_cost_before_mod * 1.2

def guido_cost : ℝ := sally_cost_after_mod * 0.85

theorem guido_cost_calculation : guido_cost = 42840 := by
  sorry

end NUMINAMATH_CALUDE_guido_cost_calculation_l283_28310


namespace NUMINAMATH_CALUDE_fraction_addition_l283_28325

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l283_28325


namespace NUMINAMATH_CALUDE_first_term_is_5_5_l283_28386

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first 30 terms is 600 -/
  sum_30 : (30 / 2 : ℝ) * (2 * a + 29 * d) = 600
  /-- The sum of the next 70 terms (31st to 100th) is 4900 -/
  sum_70 : (70 / 2 : ℝ) * (2 * (a + 30 * d) + 69 * d) = 4900

/-- The first term of the arithmetic sequence with the given properties is 5.5 -/
theorem first_term_is_5_5 (seq : ArithmeticSequence) : seq.a = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_5_5_l283_28386
