import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l277_27738

theorem cubic_root_sum_squares (p q r : ℝ) (x y z : ℝ) : 
  (x^3 - p*x^2 + q*x - r = 0) → 
  (y^3 - p*y^2 + q*y - r = 0) → 
  (z^3 - p*z^2 + q*z - r = 0) → 
  (x + y + z = p) →
  (x*y + x*z + y*z = q) →
  x^2 + y^2 + z^2 = p^2 - 2*q := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l277_27738


namespace NUMINAMATH_CALUDE_max_child_fraction_is_11_20_l277_27784

/-- Represents the babysitting scenario for Jane -/
structure BabysittingScenario where
  jane_start_age : ℕ
  jane_current_age : ℕ
  years_since_stopped : ℕ
  oldest_babysat_current_age : ℕ

/-- The maximum fraction of Jane's age that a child she babysat could be -/
def max_child_fraction (scenario : BabysittingScenario) : ℚ :=
  let jane_stop_age := scenario.jane_current_age - scenario.years_since_stopped
  let child_age_when_jane_stopped := scenario.oldest_babysat_current_age - scenario.years_since_stopped
  child_age_when_jane_stopped / jane_stop_age

/-- The theorem stating the maximum fraction of Jane's age a child could be -/
theorem max_child_fraction_is_11_20 (scenario : BabysittingScenario)
  (h1 : scenario.jane_start_age = 18)
  (h2 : scenario.jane_current_age = 32)
  (h3 : scenario.years_since_stopped = 12)
  (h4 : scenario.oldest_babysat_current_age = 23) :
  max_child_fraction scenario = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_max_child_fraction_is_11_20_l277_27784


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l277_27746

theorem min_value_sum_of_fractions (x y a b : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : a > 0) (h4 : b > 0) (h5 : x + y = 1) :
  a / x + b / y ≥ (Real.sqrt a + Real.sqrt b)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l277_27746


namespace NUMINAMATH_CALUDE_bus_problem_l277_27722

theorem bus_problem (initial : ℕ) (first_on : ℕ) (second_off : ℕ) (third_off : ℕ) (third_on : ℕ) (final : ℕ) :
  initial = 18 →
  first_on = 5 →
  second_off = 4 →
  third_off = 3 →
  third_on = 5 →
  final = 25 →
  ∃ (second_on : ℕ), 
    final = initial + first_on + second_on - second_off - third_off + third_on ∧
    second_on = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l277_27722


namespace NUMINAMATH_CALUDE_investment_value_proof_l277_27782

theorem investment_value_proof (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.23 * 1500 = 0.19 * (x + 1500) →
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_value_proof_l277_27782


namespace NUMINAMATH_CALUDE_philips_bananas_l277_27792

theorem philips_bananas (num_groups : ℕ) (bananas_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : bananas_per_group = 37) :
  num_groups * bananas_per_group = 407 := by
  sorry

end NUMINAMATH_CALUDE_philips_bananas_l277_27792


namespace NUMINAMATH_CALUDE_min_distance_circle_parabola_l277_27778

/-- The minimum distance between a point on a circle and a point on a parabola -/
theorem min_distance_circle_parabola :
  ∀ (A B : ℝ × ℝ),
  (A.1^2 + A.2^2 = 16) →
  (B.2 = B.1^2 - 4) →
  (∃ (θ : ℝ), A = (4 * Real.cos θ, 4 * Real.sin θ)) →
  (∃ (x : ℝ), B = (x, x^2 - 4)) →
  (∃ (d : ℝ), d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (∀ (d' : ℝ), d' ≥ d) →
  (∃ (x : ℝ), -2*(4*Real.cos θ - x) + 2*(4*Real.sin θ - (x^2 - 4))*(-2*x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_parabola_l277_27778


namespace NUMINAMATH_CALUDE_at_op_zero_at_op_distributive_at_op_max_for_rectangle_l277_27770

/-- Operation @ for real numbers -/
def at_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem 1: If a @ b = 0, then a = 0 or b = 0 -/
theorem at_op_zero (a b : ℝ) : at_op a b = 0 → a = 0 ∨ b = 0 := by sorry

/-- Theorem 2: a @ (b + c) = a @ b + a @ c -/
theorem at_op_distributive (a b c : ℝ) : at_op a (b + c) = at_op a b + at_op a c := by sorry

/-- Theorem 3: For a rectangle with fixed perimeter, a @ b is maximized when a = b -/
theorem at_op_max_for_rectangle (a b : ℝ) (h : a > 0 ∧ b > 0) (perimeter : ℝ) 
  (h_perimeter : 2 * (a + b) = perimeter) :
  ∀ x y, x > 0 → y > 0 → 2 * (x + y) = perimeter → at_op a b ≥ at_op x y := by sorry

end NUMINAMATH_CALUDE_at_op_zero_at_op_distributive_at_op_max_for_rectangle_l277_27770


namespace NUMINAMATH_CALUDE_solve_linear_equation_l277_27732

theorem solve_linear_equation (x y : ℝ) : 2 * x + y = 3 → y = 3 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l277_27732


namespace NUMINAMATH_CALUDE_circle_equation_l277_27763

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

/-- A circle centered at the origin -/
def circle_at_origin (x y r : ℝ) : Prop := x^2 + y^2 = r^2

/-- The circle is tangent to the line -/
def is_tangent (r : ℝ) : Prop := ∃ x y : ℝ, tangent_line x y ∧ circle_at_origin x y r

theorem circle_equation : 
  ∃ r : ℝ, is_tangent r → circle_at_origin x y 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l277_27763


namespace NUMINAMATH_CALUDE_scientific_notation_of_13000_l277_27713

theorem scientific_notation_of_13000 :
  ∃ (a : ℝ) (n : ℤ), 13000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.3 ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_13000_l277_27713


namespace NUMINAMATH_CALUDE_simplify_expression_l277_27761

theorem simplify_expression (x y : ℝ) : (5*x - y) - 3*(2*x - 3*y) + x = 8*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l277_27761


namespace NUMINAMATH_CALUDE_f_greater_than_one_factorial_inequality_l277_27724

noncomputable def f (x : ℝ) : ℝ := (1/x + 1/2) * Real.log (x + 1)

theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f x > 1 := by sorry

theorem factorial_inequality (n : ℕ) :
  5/6 < Real.log (n.factorial : ℝ) - (n + 1/2) * Real.log n + n ∧
  Real.log (n.factorial : ℝ) - (n + 1/2) * Real.log n + n ≤ 1 := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_factorial_inequality_l277_27724


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_100_l277_27740

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_100 :
  rectangle_area 625 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_100_l277_27740


namespace NUMINAMATH_CALUDE_island_population_theorem_l277_27762

/-- Represents the number of turtles and rabbits on an island -/
structure IslandPopulation where
  turtles : ℕ
  rabbits : ℕ

/-- Represents the populations of the four islands -/
structure IslandSystem where
  happy : IslandPopulation
  lonely : IslandPopulation
  serene : IslandPopulation
  tranquil : IslandPopulation

/-- Theorem stating the conditions and the result to be proven -/
theorem island_population_theorem (islands : IslandSystem) : 
  (islands.happy.turtles = 120) →
  (islands.happy.rabbits = 80) →
  (islands.lonely.turtles = islands.happy.turtles / 3) →
  (islands.lonely.rabbits = islands.lonely.turtles) →
  (islands.serene.rabbits = 2 * islands.lonely.rabbits) →
  (islands.serene.turtles = 3 * islands.lonely.rabbits / 4) →
  (islands.tranquil.turtles = islands.tranquil.rabbits) →
  (islands.tranquil.turtles = 
    (islands.happy.turtles - islands.serene.turtles) + 5) →
  (islands.happy.turtles + islands.lonely.turtles + 
   islands.serene.turtles + islands.tranquil.turtles = 285) ∧
  (islands.happy.rabbits + islands.lonely.rabbits + 
   islands.serene.rabbits + islands.tranquil.rabbits = 295) := by
  sorry

end NUMINAMATH_CALUDE_island_population_theorem_l277_27762


namespace NUMINAMATH_CALUDE_share_of_c_l277_27768

/-- 
Given a total amount to be divided among three people A, B, and C,
where A gets 2/3 of what B gets, and B gets 1/4 of what C gets,
prove that the share of C is 360 when the total amount is 510.
-/
theorem share_of_c (total : ℚ) (share_a share_b share_c : ℚ) : 
  total = 510 →
  share_a = (2/3) * share_b →
  share_b = (1/4) * share_c →
  share_a + share_b + share_c = total →
  share_c = 360 := by
sorry

end NUMINAMATH_CALUDE_share_of_c_l277_27768


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l277_27736

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees, 
    with trees at each end and 12 meters between consecutive trees, is 300 meters -/
theorem yard_length_26_trees : 
  yard_length 26 12 = 300 := by sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l277_27736


namespace NUMINAMATH_CALUDE_impossible_to_measure_one_liter_l277_27743

/-- Represents the state of water in the containers -/
structure WaterState where
  jug : ℕ  -- Amount of water in the 4-liter jug
  pot : ℕ  -- Amount of water in the 6-liter pot

/-- Possible operations on the containers -/
inductive Operation
  | FillJug
  | FillPot
  | EmptyJug
  | EmptyPot
  | PourJugToPot
  | PourPotToJug

/-- Applies an operation to a water state -/
def applyOperation (state : WaterState) (op : Operation) : WaterState :=
  match op with
  | Operation.FillJug => { jug := 4, pot := state.pot }
  | Operation.FillPot => { jug := state.jug, pot := 6 }
  | Operation.EmptyJug => { jug := 0, pot := state.pot }
  | Operation.EmptyPot => { jug := state.jug, pot := 0 }
  | Operation.PourJugToPot =>
      let amount := min state.jug (6 - state.pot)
      { jug := state.jug - amount, pot := state.pot + amount }
  | Operation.PourPotToJug =>
      let amount := min state.pot (4 - state.jug)
      { jug := state.jug + amount, pot := state.pot - amount }

/-- Theorem: It's impossible to measure exactly one liter of water -/
theorem impossible_to_measure_one_liter :
  ∀ (initial : WaterState) (ops : List Operation),
    (initial.jug = 0 ∧ initial.pot = 0) →
    let final := ops.foldl applyOperation initial
    (final.jug ≠ 1 ∧ final.pot ≠ 1) :=
  sorry


end NUMINAMATH_CALUDE_impossible_to_measure_one_liter_l277_27743


namespace NUMINAMATH_CALUDE_sinusoid_amplitude_l277_27757

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
if the function oscillates between 5 and -3, then the amplitude a is equal to 4.
-/
theorem sinusoid_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_sinusoid_amplitude_l277_27757


namespace NUMINAMATH_CALUDE_total_reduction_proof_l277_27745

-- Define the original price and reduction percentages
def original_price : ℝ := 500
def first_reduction : ℝ := 0.07
def second_reduction : ℝ := 0.05
def third_reduction : ℝ := 0.03

-- Define the function to calculate the price after reductions
def price_after_reductions (p : ℝ) (r1 r2 r3 : ℝ) : ℝ :=
  p * (1 - r1) * (1 - r2) * (1 - r3)

-- Theorem statement
theorem total_reduction_proof :
  original_price - price_after_reductions original_price first_reduction second_reduction third_reduction = 71.5025 := by
  sorry


end NUMINAMATH_CALUDE_total_reduction_proof_l277_27745


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l277_27726

theorem fraction_to_zero_power (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l277_27726


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l277_27781

/-- A quadratic function f(x) = x^2 - 2x + m with a minimum value of 1 on [3, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

/-- The domain of the function -/
def domain : Set ℝ := {x : ℝ | x ≥ 3}

theorem quadratic_minimum_value (m : ℝ) :
  (∀ x ∈ domain, f m x ≥ 1) ∧ (∃ x ∈ domain, f m x = 1) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l277_27781


namespace NUMINAMATH_CALUDE_sandals_sold_l277_27710

theorem sandals_sold (sneakers boots total : ℕ) 
  (h1 : sneakers = 2)
  (h2 : boots = 11)
  (h3 : total = 17)
  (h4 : ∃ sandals : ℕ, total = sneakers + sandals + boots) :
  ∃ sandals : ℕ, sandals = 4 ∧ total = sneakers + sandals + boots :=
by
  sorry

end NUMINAMATH_CALUDE_sandals_sold_l277_27710


namespace NUMINAMATH_CALUDE_marie_magazine_sales_l277_27796

/-- Given that Marie sold a total of 425.0 magazines and newspapers,
    and 275.0 of them were newspapers, prove that she sold 150.0 magazines. -/
theorem marie_magazine_sales :
  let total_sales : ℝ := 425.0
  let newspaper_sales : ℝ := 275.0
  let magazine_sales : ℝ := total_sales - newspaper_sales
  magazine_sales = 150.0 := by sorry

end NUMINAMATH_CALUDE_marie_magazine_sales_l277_27796


namespace NUMINAMATH_CALUDE_tom_nail_purchase_l277_27786

/-- The number of additional nails Tom needs to buy for his project -/
def additional_nails_needed (initial : ℝ) (toolshed : ℝ) (drawer : ℝ) (neighbor : ℝ) (thank_you : ℝ) (required : ℝ) : ℝ :=
  required - (initial + toolshed + drawer + neighbor + thank_you)

/-- Theorem stating the number of additional nails Tom needs to buy -/
theorem tom_nail_purchase (initial : ℝ) (toolshed : ℝ) (drawer : ℝ) (neighbor : ℝ) (thank_you : ℝ) (required : ℝ)
    (h1 : initial = 247.5)
    (h2 : toolshed = 144.25)
    (h3 : drawer = 0.75)
    (h4 : neighbor = 58.75)
    (h5 : thank_you = 37.25)
    (h6 : required = 761.58) :
    additional_nails_needed initial toolshed drawer neighbor thank_you required = 273.08 := by
  sorry

end NUMINAMATH_CALUDE_tom_nail_purchase_l277_27786


namespace NUMINAMATH_CALUDE_participants_meet_on_DA_l277_27711

/-- Represents a participant in the square walking problem -/
structure Participant where
  speed : ℝ
  startPoint : ℕ

/-- Represents the square and the walking problem -/
structure SquareWalk where
  sideLength : ℝ
  participantA : Participant
  participantB : Participant

/-- The point where the participants meet -/
def meetingPoint (sw : SquareWalk) : ℕ :=
  sorry

theorem participants_meet_on_DA (sw : SquareWalk) 
  (h1 : sw.sideLength = 90)
  (h2 : sw.participantA.speed = 65)
  (h3 : sw.participantB.speed = 72)
  (h4 : sw.participantA.startPoint = 0)
  (h5 : sw.participantB.startPoint = 1) :
  meetingPoint sw = 3 :=
sorry

end NUMINAMATH_CALUDE_participants_meet_on_DA_l277_27711


namespace NUMINAMATH_CALUDE_division_remainder_problem_l277_27714

theorem division_remainder_problem : ∃ (x : ℕ), 
  (1782 - x = 1500) ∧ 
  (∃ (r : ℕ), 1782 = 6 * x + r) ∧
  (1782 % x = 90) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l277_27714


namespace NUMINAMATH_CALUDE_unique_polynomial_with_integer_root_l277_27755

theorem unique_polynomial_with_integer_root :
  ∃! (a : ℕ+), ∃ (x : ℤ), x^2 - (a : ℤ) * x + (a : ℤ) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_with_integer_root_l277_27755


namespace NUMINAMATH_CALUDE_cube_sum_equals_94_l277_27707

theorem cube_sum_equals_94 (a b c : ℝ) 
  (h1 : a + b + c = 7) 
  (h2 : a * b + a * c + b * c = 11) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = 94 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_94_l277_27707


namespace NUMINAMATH_CALUDE_hawks_score_l277_27730

theorem hawks_score (total_points margin eagles_three_pointers : ℕ) 
  (h1 : total_points = 82)
  (h2 : margin = 18)
  (h3 : eagles_three_pointers = 12) : 
  total_points - (total_points + margin) / 2 = 32 :=
sorry

end NUMINAMATH_CALUDE_hawks_score_l277_27730


namespace NUMINAMATH_CALUDE_remaining_distance_l277_27749

theorem remaining_distance (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 8205)
  (h2 : monday = 907)
  (h3 : tuesday = 582) :
  total - (monday + tuesday) = 6716 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l277_27749


namespace NUMINAMATH_CALUDE_ball_throw_circle_l277_27789

/-- Given a circular arrangement of 15 elements, prove that starting from
    element 1 and moving with a step of 5 (modulo 15), it takes exactly 3
    steps to return to element 1. -/
theorem ball_throw_circle (n : ℕ) (h : n = 15) :
  let f : ℕ → ℕ := λ x => (x + 5) % n
  ∃ k : ℕ, k > 0 ∧ (f^[k] 1 = 1) ∧ ∀ m : ℕ, 0 < m → m < k → f^[m] 1 ≠ 1 ∧ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_ball_throw_circle_l277_27789


namespace NUMINAMATH_CALUDE_chess_team_boys_l277_27731

theorem chess_team_boys (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 26 →
  present = 18 →
  boys + girls = total →
  present = boys + girls / 3 →
  boys = 14 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_boys_l277_27731


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l277_27750

/-- Two planar vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, y)
  are_parallel a b → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l277_27750


namespace NUMINAMATH_CALUDE_august_mail_l277_27742

def mail_sequence (n : ℕ) : ℕ := 5 * 2^n

theorem august_mail :
  mail_sequence 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_august_mail_l277_27742


namespace NUMINAMATH_CALUDE_expression_factorization_l277_27777

theorem expression_factorization (x : ℝ) : 
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x) = 3 * x * (5 * x^3 - 7 * x^2 + 12) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l277_27777


namespace NUMINAMATH_CALUDE_sum_g_11_and_neg_11_l277_27708

/-- Given a function g(x) = px^8 + qx^6 - rx^4 + sx^2 + 5, 
    if g(11) = 7, then g(11) + g(-11) = 14 -/
theorem sum_g_11_and_neg_11 (p q r s : ℝ) : 
  let g : ℝ → ℝ := λ x => p * x^8 + q * x^6 - r * x^4 + s * x^2 + 5
  g 11 = 7 → g 11 + g (-11) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_11_and_neg_11_l277_27708


namespace NUMINAMATH_CALUDE_bike_ride_distance_l277_27716

/-- Calculates the total distance traveled in a 3-hour bike ride given specific conditions -/
theorem bike_ride_distance (second_hour_distance : ℝ) 
  (h1 : second_hour_distance = 12)
  (h2 : second_hour_distance = 1.2 * (second_hour_distance / 1.2))
  (h3 : 1.25 * second_hour_distance = 15) : 
  (second_hour_distance / 1.2) + second_hour_distance + (1.25 * second_hour_distance) = 37 := by
  sorry

#check bike_ride_distance

end NUMINAMATH_CALUDE_bike_ride_distance_l277_27716


namespace NUMINAMATH_CALUDE_triangle_perimeter_l277_27760

theorem triangle_perimeter (a : ℕ) (h1 : Odd a) (h2 : 3 < a) (h3 : a < 9) :
  (3 + 6 + a = 14) ∨ (3 + 6 + a = 16) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l277_27760


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l277_27774

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 15) * (Real.sqrt 8 / Real.sqrt 14) = 2 * Real.sqrt 35 / 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l277_27774


namespace NUMINAMATH_CALUDE_units_digit_of_power_l277_27751

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The base number -/
def base : ℕ := 5689

/-- The exponent -/
def exponent : ℕ := 439

theorem units_digit_of_power : units_digit (base ^ exponent) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l277_27751


namespace NUMINAMATH_CALUDE_polynomial_roots_l277_27790

theorem polynomial_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = -2 ∧ x₂ = 2 + Real.sqrt 2 ∧ x₃ = 2 - Real.sqrt 2) ∧
  (∀ x : ℝ, x^4 - 4*x^3 + 5*x^2 - 2*x - 8 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l277_27790


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l277_27773

/-- Given a real number x, this theorem states that the area of a rectangle with 
    dimensions (x+8) and (x+6), minus the area of a rectangle with dimensions (2x-1) 
    and (x-1), plus the area of a rectangle with dimensions (x-3) and (x-5), 
    equals 25x + 62. -/
theorem rectangle_area_difference (x : ℝ) : 
  (x + 8) * (x + 6) - (2*x - 1) * (x - 1) + (x - 3) * (x - 5) = 25*x + 62 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l277_27773


namespace NUMINAMATH_CALUDE_wood_length_equation_l277_27701

/-- Represents the length of a piece of wood that satisfies the measurement conditions. -/
def wood_length (x : ℝ) : Prop :=
  ∃ (rope_length : ℝ),
    rope_length - x = 4.5 ∧
    (rope_length / 2) - x = 1

/-- Proves that the wood length satisfies the equation from the problem. -/
theorem wood_length_equation (x : ℝ) :
  wood_length x → (x + 4.5) / 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_equation_l277_27701


namespace NUMINAMATH_CALUDE_remaining_bottles_l277_27737

theorem remaining_bottles (small_initial big_initial : ℕ) 
  (small_percent big_percent : ℚ) : 
  small_initial = 6000 →
  big_initial = 10000 →
  small_percent = 12 / 100 →
  big_percent = 15 / 100 →
  (small_initial - small_initial * small_percent).floor +
  (big_initial - big_initial * big_percent).floor = 13780 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bottles_l277_27737


namespace NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l277_27702

theorem tripled_base_doubled_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (3 * a) ^ (2 * b) = a ^ (2 * b) * y ^ b → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l277_27702


namespace NUMINAMATH_CALUDE_function_properties_l277_27766

/-- The function f(x) = x³ + 2ax² + bx + a -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b

theorem function_properties (a b : ℝ) :
  f a b (-1) = 1 ∧ f_derivative a b (-1) = 0 →
  a = 1 ∧ b = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 1 x ≤ 5 ∧ f 1 1 1 = 5 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l277_27766


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l277_27734

theorem reciprocal_of_negative_one_sixth : 
  ((-1 / 6 : ℚ)⁻¹ : ℚ) = -6 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l277_27734


namespace NUMINAMATH_CALUDE_summer_sun_salutations_l277_27719

/-- The number of sun salutations Summer performs each weekday -/
def sun_salutations_per_weekday : ℕ :=
  1300 / (365 / 7 * 5)

/-- Theorem stating that Summer performs 5 sun salutations each weekday -/
theorem summer_sun_salutations :
  sun_salutations_per_weekday = 5 := by
  sorry

end NUMINAMATH_CALUDE_summer_sun_salutations_l277_27719


namespace NUMINAMATH_CALUDE_market_fruit_count_l277_27776

/-- The number of apples in the market -/
def num_apples : ℕ := 164

/-- The difference between the number of apples and oranges -/
def apple_orange_diff : ℕ := 27

/-- The number of oranges in the market -/
def num_oranges : ℕ := num_apples - apple_orange_diff

/-- The total number of fruits (apples and oranges) in the market -/
def total_fruits : ℕ := num_apples + num_oranges

theorem market_fruit_count : total_fruits = 301 := by
  sorry

end NUMINAMATH_CALUDE_market_fruit_count_l277_27776


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_and_diff_l277_27779

/-- A right triangle with sides 7, 24, and 25 units -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1^2 + side2^2 = hypotenuse^2
  side1_eq : side1 = 7
  side2_eq : side2 = 24
  hypotenuse_eq : hypotenuse = 25

/-- The two longest altitudes in the right triangle -/
def longest_altitudes (t : RightTriangle) : ℝ × ℝ :=
  (t.side1, t.side2)

/-- The sum of the two longest altitudes -/
def sum_of_longest_altitudes (t : RightTriangle) : ℝ :=
  (longest_altitudes t).1 + (longest_altitudes t).2

/-- The difference between the two longest altitudes -/
def diff_of_longest_altitudes (t : RightTriangle) : ℝ :=
  (longest_altitudes t).2 - (longest_altitudes t).1

theorem longest_altitudes_sum_and_diff (t : RightTriangle) :
  sum_of_longest_altitudes t = 31 ∧ diff_of_longest_altitudes t = 17 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_and_diff_l277_27779


namespace NUMINAMATH_CALUDE_spring_equation_l277_27754

theorem spring_equation (RI G SP T M N : ℤ) (L : ℚ) : 
  RI + G + SP = 50 ∧
  RI + T + M = 63 ∧
  G + T + SP = 25 ∧
  SP + M = 13 ∧
  M + RI = 48 ∧
  N = 1 →
  L * M * T + SP * RI * N * G = 2023 →
  L = 341 / 40 := by
sorry

end NUMINAMATH_CALUDE_spring_equation_l277_27754


namespace NUMINAMATH_CALUDE_solution_to_inequalities_l277_27780

theorem solution_to_inequalities :
  let x : ℝ := 3
  (x + 3 > 2) ∧ (1 - 2*x < -3) := by sorry

end NUMINAMATH_CALUDE_solution_to_inequalities_l277_27780


namespace NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l277_27764

/-- The decimal representation of 3.127̄ as a rational number -/
def recurring_decimal : ℚ := 3 + 127 / 999

/-- The fraction 3124/999 -/
def target_fraction : ℚ := 3124 / 999

/-- Theorem stating that the recurring decimal 3.127̄ is equal to the fraction 3124/999 -/
theorem recurring_decimal_equals_fraction : recurring_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l277_27764


namespace NUMINAMATH_CALUDE_system_solution_conditions_l277_27712

/-- Given a system of equations, prove the existence of conditions for distinct positive solutions -/
theorem system_solution_conditions (a b : ℝ) :
  ∃ (x y z : ℝ), 
    (x + y + z = a) ∧ 
    (x^2 + y^2 + z^2 = b^2) ∧ 
    (x * y = z^2) ∧ 
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
    (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ a = c ∧ b = d) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l277_27712


namespace NUMINAMATH_CALUDE_min_value_expression_l277_27765

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l277_27765


namespace NUMINAMATH_CALUDE_pascal_row_15_sum_l277_27798

/-- Definition of Pascal's Triangle sum for a given row -/
def pascal_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of numbers in row 15 of Pascal's Triangle is 32768 -/
theorem pascal_row_15_sum : pascal_sum 15 = 32768 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_15_sum_l277_27798


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l277_27794

/-- Given a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has the equation y = 3x, then b = 3 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∃ x y : ℝ, x^2 - y^2/b^2 = 1 ∧ y = 3*x) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l277_27794


namespace NUMINAMATH_CALUDE_caiden_roofing_problem_l277_27725

theorem caiden_roofing_problem (cost_per_foot : ℝ) (free_feet : ℝ) (remaining_cost : ℝ) :
  cost_per_foot = 8 →
  free_feet = 250 →
  remaining_cost = 400 →
  ∃ (total_feet : ℝ), total_feet = 300 ∧ (total_feet - free_feet) * cost_per_foot = remaining_cost :=
by sorry

end NUMINAMATH_CALUDE_caiden_roofing_problem_l277_27725


namespace NUMINAMATH_CALUDE_cube_volume_l277_27706

theorem cube_volume (n : ℝ) : 
  (∃ (s : ℝ), s * Real.sqrt 2 = 4 ∧ s^3 = n * Real.sqrt 2) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l277_27706


namespace NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_neq_2_l277_27728

theorem x_gt_2_sufficient_not_necessary_for_x_neq_2 :
  (∃ x : ℝ, x ≠ 2 ∧ ¬(x > 2)) ∧
  (∀ x : ℝ, x > 2 → x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_neq_2_l277_27728


namespace NUMINAMATH_CALUDE_number_of_girls_l277_27799

theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 18 →
  happy_boys = 6 →
  sad_girls = 4 →
  happy_children + sad_children + neutral_children = total_children →
  total_children - boys = 42 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l277_27799


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l277_27735

theorem alcohol_concentration_proof :
  ∀ (vessel1_capacity vessel2_capacity total_liquid final_capacity : ℝ)
    (vessel2_concentration final_concentration : ℝ),
  vessel1_capacity = 2 →
  vessel2_capacity = 6 →
  vessel2_concentration = 0.4 →
  total_liquid = 8 →
  final_capacity = 10 →
  final_concentration = 0.29000000000000004 →
  ∃ (vessel1_concentration : ℝ),
    vessel1_concentration = 0.25 ∧
    vessel1_concentration * vessel1_capacity + vessel2_concentration * vessel2_capacity =
      final_concentration * final_capacity :=
by
  sorry

#check alcohol_concentration_proof

end NUMINAMATH_CALUDE_alcohol_concentration_proof_l277_27735


namespace NUMINAMATH_CALUDE_new_student_weight_is_62_l277_27752

/-- The weight of the new student given the conditions of the problem -/
def new_student_weight (n : ℕ) (avg_decrease : ℚ) (old_student_weight : ℚ) : ℚ :=
  old_student_weight - n * avg_decrease

/-- Theorem stating that the weight of the new student is 62 kg -/
theorem new_student_weight_is_62 :
  new_student_weight 6 3 80 = 62 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_is_62_l277_27752


namespace NUMINAMATH_CALUDE_mark_solutions_mark_coefficients_l277_27727

/-- Lauren's equation solutions -/
def lauren_solutions : Set ℝ := {x | |x - 6| = 3}

/-- Mark's equation -/
def mark_equation (b c : ℝ) (x : ℝ) : Prop := x^2 + b*x + c = 0

/-- Mark's equation has Lauren's solutions plus x = -2 -/
theorem mark_solutions (b c : ℝ) : 
  (∀ x ∈ lauren_solutions, mark_equation b c x) ∧ 
  mark_equation b c (-2) :=
sorry

/-- The values of b and c in Mark's equation -/
theorem mark_coefficients : 
  ∃ b c : ℝ, (b = -12 ∧ c = 27) ∧ 
  (∀ x ∈ lauren_solutions, mark_equation b c x) ∧ 
  mark_equation b c (-2) :=
sorry

end NUMINAMATH_CALUDE_mark_solutions_mark_coefficients_l277_27727


namespace NUMINAMATH_CALUDE_marathon_length_l277_27758

/-- A marathon runner completes a race under specific conditions. -/
theorem marathon_length (initial_distance : ℝ) (initial_time : ℝ) (total_time : ℝ) 
  (pace_ratio : ℝ) (marathon_length : ℝ) : 
  initial_distance = 10 →
  initial_time = 1 →
  total_time = 3 →
  pace_ratio = 0.8 →
  marathon_length = initial_distance + 
    (total_time - initial_time) * (initial_distance / initial_time) * pace_ratio →
  marathon_length = 26 := by
  sorry

#check marathon_length

end NUMINAMATH_CALUDE_marathon_length_l277_27758


namespace NUMINAMATH_CALUDE_point_transformation_l277_27783

-- Define the rotation function
def rotate180 (x y : ℝ) : ℝ × ℝ := (2 - x, 10 - y)

-- Define the reflection function
def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ := (y, x)

-- Theorem statement
theorem point_transformation (a b : ℝ) :
  let (x', y') := rotate180 a b
  let (x'', y'') := reflect_y_eq_x x' y'
  (x'' = 3 ∧ y'' = -6) → b - a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l277_27783


namespace NUMINAMATH_CALUDE_max_sum_of_sides_l277_27748

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def is_triangle (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given condition
def given_condition (A B C a b c : ℝ) : Prop :=
  (2 * a - c) / b = Real.cos C / Real.cos B

-- Theorem statement
theorem max_sum_of_sides 
  (h_triangle : is_triangle A B C a b c)
  (h_condition : given_condition A B C a b c)
  (h_b : b = 4) :
  ∃ (max : ℝ), max = 8 ∧ a + c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_sides_l277_27748


namespace NUMINAMATH_CALUDE_square_area_ratio_l277_27767

theorem square_area_ratio (s t : ℝ) (h : s > 0) (k : t > 0) (h_perimeter : 4 * s = 4 * (4 * t)) :
  s^2 = 16 * t^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l277_27767


namespace NUMINAMATH_CALUDE_eugene_model_house_l277_27715

/-- Eugene's model house building problem --/
theorem eugene_model_house (toothpicks_per_card : ℕ) (cards_in_deck : ℕ) 
  (boxes_used : ℕ) (toothpicks_per_box : ℕ) : 
  toothpicks_per_card = 75 →
  cards_in_deck = 52 →
  boxes_used = 6 →
  toothpicks_per_box = 450 →
  cards_in_deck - (boxes_used * toothpicks_per_box) / toothpicks_per_card = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_eugene_model_house_l277_27715


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_greater_than_10_l277_27739

theorem least_product_of_distinct_primes_greater_than_10 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 10 ∧ q > 10 ∧
    p ≠ q ∧
    p * q = 143 ∧
    ∀ a b : ℕ, a.Prime → b.Prime → a > 10 → b > 10 → a ≠ b → a * b ≥ 143 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_greater_than_10_l277_27739


namespace NUMINAMATH_CALUDE_four_digit_sum_3333_l277_27729

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Converts a FourDigitNumber to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  1000 * n.1 + 100 * n.2.1 + 10 * n.2.2.1 + n.2.2.2

/-- Rearranges a FourDigitNumber by moving the last digit to the front -/
def rearrange (n : FourDigitNumber) : FourDigitNumber :=
  (n.2.2.2, n.1, n.2.1, n.2.2.1)

/-- Checks if a FourDigitNumber contains zero -/
def containsZero (n : FourDigitNumber) : Bool :=
  n.1 = 0 || n.2.1 = 0 || n.2.2.1 = 0 || n.2.2.2 = 0

theorem four_digit_sum_3333 (n : FourDigitNumber) :
  ¬containsZero n →
  toNumber n + toNumber (rearrange n) = 3333 →
  n = (1, 2, 1, 2) ∨ n = (2, 1, 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_3333_l277_27729


namespace NUMINAMATH_CALUDE_chessboard_one_color_l277_27759

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Represents the chessboard as a function from coordinates to colors -/
def Chessboard := Fin 8 → Fin 8 → Color

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  x1 : Fin 8
  y1 : Fin 8
  x2 : Fin 8
  y2 : Fin 8

/-- Checks if a rectangle is adjacent to a corner of the board -/
def isCornerRectangle (r : Rectangle) : Prop :=
  (r.x1 = 0 ∧ r.y1 = 0) ∨
  (r.x1 = 0 ∧ r.y2 = 7) ∨
  (r.x2 = 7 ∧ r.y1 = 0) ∨
  (r.x2 = 7 ∧ r.y2 = 7)

/-- The operation of changing colors in a rectangle -/
def applyRectangle (board : Chessboard) (r : Rectangle) : Chessboard :=
  sorry

/-- Theorem stating that any chessboard can be made one color -/
theorem chessboard_one_color :
  ∀ (initial : Chessboard),
  ∃ (final : Chessboard) (steps : List Rectangle),
    (∀ r ∈ steps, isCornerRectangle r) ∧
    (final = steps.foldl applyRectangle initial) ∧
    (∃ c : Color, ∀ x y : Fin 8, final x y = c) :=
  sorry

end NUMINAMATH_CALUDE_chessboard_one_color_l277_27759


namespace NUMINAMATH_CALUDE_arithmetic_seq_first_term_arithmetic_seq_first_term_range_l277_27756

/-- Arithmetic sequence with common difference -1 -/
def ArithmeticSeq (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => ArithmeticSeq a₁ n - 1

/-- Sum of first n terms of the arithmetic sequence -/
def SumSeq (a₁ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => SumSeq a₁ n + ArithmeticSeq a₁ (n + 1)

theorem arithmetic_seq_first_term (a₁ : ℝ) :
  SumSeq a₁ 5 = -5 → a₁ = 1 := by sorry

theorem arithmetic_seq_first_term_range (a₁ : ℝ) :
  (∀ n : ℕ, n > 0 → SumSeq a₁ n ≤ ArithmeticSeq a₁ n) → a₁ ≤ 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_seq_first_term_arithmetic_seq_first_term_range_l277_27756


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l277_27797

theorem sqrt_equation_solutions (x : ℝ) :
  Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 5 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l277_27797


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l277_27720

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l277_27720


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l277_27704

theorem diophantine_equation_solution (x y z : ℤ) : x^2 + y^2 = 3*z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l277_27704


namespace NUMINAMATH_CALUDE_intersection_of_parallel_planes_l277_27775

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersection : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem intersection_of_parallel_planes
  (α β γ : Plane)
  (m n : Line)
  (h1 : m = intersection α γ)
  (h2 : n = intersection β γ)
  (h3 : parallel_planes α β) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_intersection_of_parallel_planes_l277_27775


namespace NUMINAMATH_CALUDE_profit_percentage_is_10_percent_l277_27772

def cost_price : ℚ := 340
def selling_price : ℚ := 374

theorem profit_percentage_is_10_percent :
  (selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_10_percent_l277_27772


namespace NUMINAMATH_CALUDE_tiles_needed_l277_27771

-- Define the dimensions
def tile_size : ℕ := 6
def kitchen_width : ℕ := 48
def kitchen_height : ℕ := 72

-- Define the theorem
theorem tiles_needed : 
  (kitchen_width / tile_size) * (kitchen_height / tile_size) = 96 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_l277_27771


namespace NUMINAMATH_CALUDE_correct_conclusions_l277_27769

theorem correct_conclusions :
  (∀ a b : ℝ, a + b > 0 ∧ a * b > 0 → a > 0 ∧ b > 0) ∧
  (∀ a b : ℝ, b ≠ 0 → a / b = -1 → a + b = 0) ∧
  (∀ a b c : ℝ, a < b ∧ b < c → |a - b| + |b - c| = |a - c|) :=
by sorry

end NUMINAMATH_CALUDE_correct_conclusions_l277_27769


namespace NUMINAMATH_CALUDE_new_person_weight_is_81_l277_27718

/-- The weight of a new person replacing one in a group, given the average weight increase --/
def new_person_weight (n : ℕ) (avg_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + n * avg_increase

/-- Theorem: The weight of the new person is 81 kg --/
theorem new_person_weight_is_81 :
  new_person_weight 8 2 65 = 81 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_81_l277_27718


namespace NUMINAMATH_CALUDE_remainder_polynomial_l277_27705

theorem remainder_polynomial (p : ℝ → ℝ) (h1 : p 2 = 4) (h2 : p 4 = 8) :
  ∃ (q r : ℝ → ℝ), (∀ x, p x = q x * (x - 2) * (x - 4) + r x) ∧
                    (∀ x, r x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_remainder_polynomial_l277_27705


namespace NUMINAMATH_CALUDE_bugs_meeting_time_l277_27785

/-- Two circles tangent at point O with radii 7 and 3 inches, and bugs moving at 4π and 3π inches per minute respectively -/
structure CircleSetup where
  r1 : ℝ
  r2 : ℝ
  v1 : ℝ
  v2 : ℝ
  h_r1 : r1 = 7
  h_r2 : r2 = 3
  h_v1 : v1 = 4 * Real.pi
  h_v2 : v2 = 3 * Real.pi

/-- Time taken for bugs to meet again at point O -/
def meetingTime (setup : CircleSetup) : ℝ :=
  7

/-- Theorem stating that the meeting time is 7 minutes -/
theorem bugs_meeting_time (setup : CircleSetup) :
  meetingTime setup = 7 := by
  sorry

end NUMINAMATH_CALUDE_bugs_meeting_time_l277_27785


namespace NUMINAMATH_CALUDE_f_has_inverse_when_x_geq_2_l277_27717

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem f_has_inverse_when_x_geq_2 :
  ∀ (a b : ℝ), a ≥ 2 → b ≥ 2 → a ≠ b → f a ≠ f b :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_inverse_when_x_geq_2_l277_27717


namespace NUMINAMATH_CALUDE_abc_inequalities_l277_27753

theorem abc_inequalities (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : 4*a^2 + b^2 + 16*c^2 = 1) : 
  (0 < a*b ∧ a*b < 1/4) ∧ 
  (1/a^2 + 1/b^2 + 1/(4*a*b*c^2) > 49) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l277_27753


namespace NUMINAMATH_CALUDE_speedster_fraction_l277_27744

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  convertibles : ℕ
  non_speedsters : ℕ

/-- Conditions for the inventory -/
def inventory_conditions (inv : Inventory) : Prop :=
  inv.convertibles = (4 * inv.speedsters) / 5 ∧
  inv.non_speedsters = 60 ∧
  inv.convertibles = 96 ∧
  inv.total = inv.speedsters + inv.non_speedsters

/-- Theorem: The fraction of Speedsters in the inventory is 2/3 -/
theorem speedster_fraction (inv : Inventory) 
  (h : inventory_conditions inv) : 
  (inv.speedsters : ℚ) / inv.total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speedster_fraction_l277_27744


namespace NUMINAMATH_CALUDE_courtyard_length_l277_27700

/-- The length of a rectangular courtyard given its width and paving stones --/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ)
  (h_width : width = 16.5)
  (h_num_stones : num_stones = 165)
  (h_stone_length : stone_length = 2.5)
  (h_stone_width : stone_width = 2) :
  width * (num_stones * stone_length * stone_width / width) = 50 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l277_27700


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l277_27787

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (9 + x) + Real.sqrt (16 - x) = 8) →
  ((9 + x) * (16 - x) = 380.25) :=
by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l277_27787


namespace NUMINAMATH_CALUDE_greatest_integer_solution_seven_satisfies_inequality_no_greater_integer_l277_27795

theorem greatest_integer_solution (x : ℤ) : (7 : ℤ) - 5*x + x^2 > 24 → x ≤ 7 :=
by sorry

theorem seven_satisfies_inequality : (7 : ℤ) - 5*7 + 7^2 > 24 :=
by sorry

theorem no_greater_integer :
  ∀ y : ℤ, y > 7 → ¬((7 : ℤ) - 5*y + y^2 > 24) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_seven_satisfies_inequality_no_greater_integer_l277_27795


namespace NUMINAMATH_CALUDE_chord_intersection_segments_l277_27703

theorem chord_intersection_segments (r : ℝ) (chord_length : ℝ) 
  (hr : r = 7) (hchord : chord_length = 10) : 
  ∃ (ak kb : ℝ), 
    ak = r - 2 * Real.sqrt 6 ∧ 
    kb = r + 2 * Real.sqrt 6 ∧ 
    ak + kb = 2 * r ∧
    ak * kb = (chord_length / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_segments_l277_27703


namespace NUMINAMATH_CALUDE_complex_sum_problem_l277_27793

theorem complex_sum_problem (x y u v w z : ℝ) 
  (h1 : y = 2)
  (h2 : w = -x - u)
  (h3 : Complex.mk x y + Complex.mk u v + Complex.mk w z = Complex.I * (-2)) :
  v + z = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l277_27793


namespace NUMINAMATH_CALUDE_negative_rational_power_equality_l277_27721

theorem negative_rational_power_equality : 
  Real.rpow (-3 * (3/8)) (-(2/3)) = 4/9 := by sorry

end NUMINAMATH_CALUDE_negative_rational_power_equality_l277_27721


namespace NUMINAMATH_CALUDE_school_population_theorem_l277_27747

theorem school_population_theorem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 400 →
  boys + girls = total →
  girls = (boys * 100) / total →
  boys = 320 := by
sorry

end NUMINAMATH_CALUDE_school_population_theorem_l277_27747


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_inequality_l277_27709

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x < 6} = Set.Ioo (-8/3) (4/3) := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, f a x + f a (-x) ≥ 5) ↔ 
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_inequality_l277_27709


namespace NUMINAMATH_CALUDE_right_triangle_sides_l277_27741

theorem right_triangle_sides : ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = 40) →
  (a^2 + b^2 = c^2) →
  ((a + 4)^2 + (b + 1)^2 = (c + 3)^2) →
  (a < b) →
  (a = 8 ∧ b = 15 ∧ c = 17) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l277_27741


namespace NUMINAMATH_CALUDE_inequality_proof_l277_27733

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l277_27733


namespace NUMINAMATH_CALUDE_lily_correct_answers_percentage_l277_27723

theorem lily_correct_answers_percentage
  (t : ℝ)
  (h_t_positive : t > 0)
  (h_max_alone : 0.7 * (t / 2) = 0.35 * t)
  (h_max_total : 0.82 * t = 0.82 * t)
  (h_lily_alone : 0.85 * (t / 2) = 0.425 * t)
  (h_solved_together : 0.82 * t - 0.35 * t = 0.47 * t) :
  (0.425 * t + 0.47 * t) / t = 0.895 := by
  sorry

#check lily_correct_answers_percentage

end NUMINAMATH_CALUDE_lily_correct_answers_percentage_l277_27723


namespace NUMINAMATH_CALUDE_school_governor_election_votes_l277_27788

theorem school_governor_election_votes (elvis_votes : ℕ) (elvis_percentage : ℚ) 
  (h1 : elvis_votes = 45)
  (h2 : elvis_percentage = 1/4)
  (h3 : elvis_votes = elvis_percentage * total_votes) :
  total_votes = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_school_governor_election_votes_l277_27788


namespace NUMINAMATH_CALUDE_problem_solution_l277_27791

-- Custom operation
def star (x y : ℕ) : ℕ := x * y + 1

-- Prime number function
def nth_prime (n : ℕ) : ℕ := sorry

-- Product function
def product_to_n (n : ℕ) : ℚ := sorry

-- Area of inscribed square
def inscribed_square_area (r : ℝ) : ℝ := sorry

theorem problem_solution :
  (star (star 2 4) 2 = 19) ∧
  (nth_prime 8 = 19) ∧
  (product_to_n 50 = 1 / 50) ∧
  (inscribed_square_area 10 = 200) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l277_27791
