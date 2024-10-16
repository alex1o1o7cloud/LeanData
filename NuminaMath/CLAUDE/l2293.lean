import Mathlib

namespace NUMINAMATH_CALUDE_square_of_1023_l2293_229300

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l2293_229300


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2293_229314

theorem line_intercept_sum (d : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2293_229314


namespace NUMINAMATH_CALUDE_senior_ticket_cost_l2293_229319

theorem senior_ticket_cost 
  (total_tickets : ℕ) 
  (regular_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : senior_tickets = 24) :
  ∃ (senior_ticket_cost : ℕ),
    senior_ticket_cost * senior_tickets + 
    regular_ticket_cost * (total_tickets - senior_tickets) = 
    total_sales ∧ senior_ticket_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_senior_ticket_cost_l2293_229319


namespace NUMINAMATH_CALUDE_part1_part2_l2293_229353

/-- Represents a hotel accommodation scenario for a tour group -/
structure HotelAccommodation where
  totalPeople : ℕ
  singleRooms : ℕ
  tripleRooms : ℕ
  singleRoomPrice : ℕ
  tripleRoomPrice : ℕ
  menCount : ℕ

/-- Calculates the total cost for one night -/
def totalCost (h : HotelAccommodation) : ℕ :=
  h.singleRooms * h.singleRoomPrice + h.tripleRooms * h.tripleRoomPrice

/-- Part 1: Proves that given a total cost of 1530 yuan, the number of single rooms rented is 1 -/
theorem part1 (h : HotelAccommodation) 
    (hTotal : h.totalPeople = 33)
    (hSinglePrice : h.singleRoomPrice = 100)
    (hTriplePrice : h.tripleRoomPrice = 130)
    (hCost : totalCost h = 1530)
    (hSingleAvailable : h.singleRooms ≤ 4) :
  h.singleRooms = 1 := by
  sorry

/-- Part 2: Proves that given 3 single rooms and 19 men, the minimum cost is 1600 yuan -/
theorem part2 (h : HotelAccommodation) 
    (hTotal : h.totalPeople = 33)
    (hSinglePrice : h.singleRoomPrice = 100)
    (hTriplePrice : h.tripleRoomPrice = 130)
    (hSingleRooms : h.singleRooms = 3)
    (hMenCount : h.menCount = 19) :
  ∃ (minCost : ℕ), minCost = 1600 ∧ ∀ (cost : ℕ), totalCost h ≥ minCost := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l2293_229353


namespace NUMINAMATH_CALUDE_trig_identities_and_expression_l2293_229387

/-- Given an angle α whose terminal side passes through point (4, -3),
    prove the trigonometric identities and the value of a specific expression. -/
theorem trig_identities_and_expression (α : Real) 
  (h : ∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) : 
  Real.sin α = -3/5 ∧ 
  Real.cos α = 4/5 ∧ 
  Real.tan α = -3/4 ∧
  (Real.sin (Real.pi + α) + 2 * Real.sin (Real.pi/2 - α)) / (2 * Real.cos (Real.pi - α)) = -11/8 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_and_expression_l2293_229387


namespace NUMINAMATH_CALUDE_sequence_sum_l2293_229357

theorem sequence_sum (n : ℕ) (s : ℕ → ℕ) : n = 2010 →
  (∀ i, i ∈ Finset.range (n - 1) → s (i + 1) = s i + 1) →
  (Finset.sum (Finset.range n) s = 5307) →
  (Finset.sum (Finset.range 1005) (fun i => s (2 * i))) = 2151 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2293_229357


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2293_229304

/-- Given x = 11 * 36 * 42, prove that the smallest positive integer y 
    such that xy is a perfect cube is 5929 -/
theorem smallest_y_for_perfect_cube (x : ℕ) (hx : x = 11 * 36 * 42) :
  ∃ y : ℕ, y > 0 ∧ 
    (∃ n : ℕ, x * y = n^3) ∧ 
    (∀ z : ℕ, z > 0 → z < y → ¬∃ m : ℕ, x * z = m^3) ∧
    y = 5929 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2293_229304


namespace NUMINAMATH_CALUDE_sara_kittens_count_l2293_229328

def initial_kittens : ℕ := 6
def kittens_given_to_jessica : ℕ := 3
def final_kittens : ℕ := 12

def kittens_from_sara : ℕ := final_kittens - (initial_kittens - kittens_given_to_jessica)

theorem sara_kittens_count : kittens_from_sara = 9 := by
  sorry

end NUMINAMATH_CALUDE_sara_kittens_count_l2293_229328


namespace NUMINAMATH_CALUDE_price_change_theorem_l2293_229355

theorem price_change_theorem (initial_price : ℝ) (initial_price_positive : initial_price > 0) :
  let price_after_increase := initial_price * 1.31
  let price_after_first_discount := price_after_increase * 0.9
  let final_price := price_after_first_discount * 0.85
  (final_price - initial_price) / initial_price = 0.00215 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l2293_229355


namespace NUMINAMATH_CALUDE_max_annual_average_profit_l2293_229397

/-- The annual average profit function -/
def f (n : ℕ+) : ℚ :=
  (110 * n - (n^2 + n) - 90) / n

/-- Theorem stating that f(n) reaches its maximum when n = 5 -/
theorem max_annual_average_profit :
  ∀ k : ℕ+, f 5 ≥ f k :=
sorry

end NUMINAMATH_CALUDE_max_annual_average_profit_l2293_229397


namespace NUMINAMATH_CALUDE_e_value_proof_l2293_229377

theorem e_value_proof (a b c : ℕ) (e : ℚ) 
  (h1 : a = 105)
  (h2 : b = 126)
  (h3 : c = 63)
  (h4 : a^3 - b^2 + c^2 = 21 * 25 * 45 * e) :
  e = 47.7 := by
  sorry

end NUMINAMATH_CALUDE_e_value_proof_l2293_229377


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2293_229335

theorem arithmetic_calculation : 5020 - (1004 / 20.08) = 4970 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2293_229335


namespace NUMINAMATH_CALUDE_example_theorem_l2293_229391

-- Define the necessary types and structures

-- State the theorem
theorem example_theorem (hypothesis1 : Type) (hypothesis2 : Type) : conclusion_type :=
  -- The proof would go here, but we're using sorry as requested
  sorry

-- Additional definitions or lemmas if needed

end NUMINAMATH_CALUDE_example_theorem_l2293_229391


namespace NUMINAMATH_CALUDE_puppy_feeding_last_two_weeks_l2293_229327

/-- Represents the feeding schedule and amount for a puppy over 4 weeks -/
structure PuppyFeeding where
  total_food : ℚ
  first_day_food : ℚ
  first_two_weeks_daily_feeding : ℚ
  first_two_weeks_feeding_frequency : ℕ
  last_two_weeks_feeding_frequency : ℕ
  days_in_week : ℕ
  total_weeks : ℕ

/-- Calculates the amount of food fed to the puppy twice a day for the last two weeks -/
def calculate_last_two_weeks_feeding (pf : PuppyFeeding) : ℚ :=
  let first_two_weeks_food := pf.first_two_weeks_daily_feeding * pf.first_two_weeks_feeding_frequency * (2 * pf.days_in_week)
  let total_food_minus_first_day := pf.total_food - pf.first_day_food
  let last_two_weeks_food := total_food_minus_first_day - first_two_weeks_food
  let last_two_weeks_feedings := 2 * pf.last_two_weeks_feeding_frequency * pf.days_in_week
  last_two_weeks_food / last_two_weeks_feedings

/-- Theorem stating that the amount of food fed to the puppy twice a day for the last two weeks is 1/2 cup -/
theorem puppy_feeding_last_two_weeks
  (pf : PuppyFeeding)
  (h1 : pf.total_food = 25)
  (h2 : pf.first_day_food = 1/2)
  (h3 : pf.first_two_weeks_daily_feeding = 1/4)
  (h4 : pf.first_two_weeks_feeding_frequency = 3)
  (h5 : pf.last_two_weeks_feeding_frequency = 2)
  (h6 : pf.days_in_week = 7)
  (h7 : pf.total_weeks = 4) :
  calculate_last_two_weeks_feeding pf = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_feeding_last_two_weeks_l2293_229327


namespace NUMINAMATH_CALUDE_polynomial_equality_l2293_229350

theorem polynomial_equality (p : ℝ → ℝ) : 
  (∀ x, p x + (x^4 + 4*x^3 + 8*x) = (10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5)) →
  (∀ x, p x = 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2293_229350


namespace NUMINAMATH_CALUDE_power_division_rule_l2293_229337

theorem power_division_rule (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2293_229337


namespace NUMINAMATH_CALUDE_day_150_previous_year_is_friday_l2293_229375

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeapYear : Bool

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

theorem day_150_previous_year_is_friday 
  (N : Year) 
  (h1 : N.isLeapYear = true) 
  (h2 : advanceDays DayOfWeek.Sunday 249 = DayOfWeek.Friday) : 
  advanceDays DayOfWeek.Sunday 149 = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_day_150_previous_year_is_friday_l2293_229375


namespace NUMINAMATH_CALUDE_range_of_k_value_of_k_l2293_229378

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2 - 2*k)*x + k^2 = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁ ≠ x₂

-- Define the additional condition
def root_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ 
    |x₁ + x₂| + 1 = x₁ * x₂

-- Theorem statements
theorem range_of_k (k : ℝ) : has_real_roots k → k ≤ 1/2 :=
sorry

theorem value_of_k : ∀ k : ℝ, has_real_roots k ∧ root_condition k → k = -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_value_of_k_l2293_229378


namespace NUMINAMATH_CALUDE_initial_distance_of_specific_program_l2293_229368

/-- Represents a running program with weekly increments -/
structure RunningProgram where
  initial_distance : ℕ  -- Initial daily running distance
  weeks : ℕ             -- Number of weeks in the program
  increment : ℕ         -- Weekly increment in daily distance

/-- Calculates the final daily running distance after the program -/
def final_distance (program : RunningProgram) : ℕ :=
  program.initial_distance + (program.weeks - 1) * program.increment

/-- Theorem stating the initial distance given the conditions -/
theorem initial_distance_of_specific_program :
  ∃ (program : RunningProgram),
    program.weeks = 5 ∧
    program.increment = 1 ∧
    final_distance program = 7 ∧
    program.initial_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_of_specific_program_l2293_229368


namespace NUMINAMATH_CALUDE_calculate_expression_l2293_229352

theorem calculate_expression : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2293_229352


namespace NUMINAMATH_CALUDE_fraction_equality_l2293_229374

theorem fraction_equality : (35 : ℚ) / (6 - 2/5) = 25/4 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2293_229374


namespace NUMINAMATH_CALUDE_circumcenter_property_l2293_229398

-- Define the basic geometric structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def intersection_point (c1 c2 : Circle) : Point := sorry

def tangent_point (c : Circle) (p : Point) : Point := sorry

def is_parallelogram (p1 p2 p3 p4 : Point) : Prop := sorry

def is_circumcenter (p : Point) (t : Point × Point × Point) : Prop := sorry

-- State the theorem
theorem circumcenter_property 
  (X Y A B C P : Point) 
  (c1 c2 : Circle) :
  c1.center = X →
  c2.center = Y →
  A = intersection_point c1 c2 →
  B = tangent_point c1 A →
  C = tangent_point c2 A →
  is_parallelogram P X A Y →
  is_circumcenter P (B, C, A) :=
sorry

end NUMINAMATH_CALUDE_circumcenter_property_l2293_229398


namespace NUMINAMATH_CALUDE_time_to_put_30_toys_is_14_minutes_l2293_229344

/-- The time required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_duration : ℕ) : ℚ :=
  let net_increase := toys_in_per_cycle - toys_out_per_cycle
  let cycles_needed := (total_toys - toys_in_per_cycle) / net_increase
  let total_seconds := cycles_needed * cycle_duration + cycle_duration
  total_seconds / 60

/-- Theorem: The time to put 30 toys in the box is 14 minutes -/
theorem time_to_put_30_toys_is_14_minutes :
  time_to_put_toys_in_box 30 3 2 30 = 14 := by
  sorry

end NUMINAMATH_CALUDE_time_to_put_30_toys_is_14_minutes_l2293_229344


namespace NUMINAMATH_CALUDE_unique_number_exists_l2293_229334

theorem unique_number_exists : ∃! x : ℝ, x / 2 + x + 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l2293_229334


namespace NUMINAMATH_CALUDE_triangle_theorem_l2293_229395

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle satisfying the given condition,
    angle C is π/4 and the maximum area when c = 2 is 1 + √2. -/
theorem triangle_theorem (t : Triangle) 
    (h : t.a * Real.cos t.B + t.b * Real.cos t.A - Real.sqrt 2 * t.c * Real.cos t.C = 0) :
    t.C = π / 4 ∧ 
    (t.c = 2 → ∃ (S : ℝ), S = (1 + Real.sqrt 2) ∧ ∀ (S' : ℝ), S' ≤ S) := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_triangle_theorem_l2293_229395


namespace NUMINAMATH_CALUDE_solution_equality_l2293_229339

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solution_equality :
  ∃ a : ℚ, F a 3 4 = F a 5 8 ∧ a = -2/49 := by sorry

end NUMINAMATH_CALUDE_solution_equality_l2293_229339


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2293_229317

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_of_A_and_B : A ∪ B = {-3, -2, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2293_229317


namespace NUMINAMATH_CALUDE_expression_evaluation_l2293_229393

theorem expression_evaluation :
  let x : ℚ := -1/3
  (3*x + 2) * (3*x - 2) - 5*x*(x - 1) - (2*x - 1)^2 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2293_229393


namespace NUMINAMATH_CALUDE_lowest_price_after_discounts_l2293_229347

/-- Calculates the lowest possible price of a product after applying two consecutive discounts -/
theorem lowest_price_after_discounts 
  (original_price : ℝ) 
  (max_regular_discount : ℝ) 
  (sale_discount : ℝ) : 
  original_price * (1 - max_regular_discount) * (1 - sale_discount) = 22.40 :=
by
  -- Assuming original_price = 40.00, max_regular_discount = 0.30, and sale_discount = 0.20
  sorry

#check lowest_price_after_discounts

end NUMINAMATH_CALUDE_lowest_price_after_discounts_l2293_229347


namespace NUMINAMATH_CALUDE_opposite_side_of_five_times_five_l2293_229370

/-- A standard 6-sided die with opposite sides summing to 7 -/
structure StandardDie where
  sides : Fin 6 → Nat
  valid_range : ∀ i, sides i ∈ Finset.range 7 \ {0}
  opposite_sum : ∀ i, sides i + sides (5 - i) = 7

/-- The number of eyes on the opposite side of 5 multiplied by 5 is 10 -/
theorem opposite_side_of_five_times_five (d : StandardDie) :
  5 * d.sides (5 - 5) = 10 := by
  sorry

end NUMINAMATH_CALUDE_opposite_side_of_five_times_five_l2293_229370


namespace NUMINAMATH_CALUDE_product_97_103_l2293_229379

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l2293_229379


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l2293_229399

theorem largest_prime_factor_of_12321 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 12321 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l2293_229399


namespace NUMINAMATH_CALUDE_expression_equality_l2293_229382

theorem expression_equality : (2^2 / 3) + (-3^2 + 5) + (-3)^2 * (2/3)^2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2293_229382


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l2293_229321

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {2, 4}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l2293_229321


namespace NUMINAMATH_CALUDE_triangle_reflection_area_sum_l2293_229312

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define reflection about a point
def reflect (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define area of a triangle
def area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_reflection_area_sum (t : Triangle) :
  let O := circumcenter t
  let A' := reflect t.A O
  let B' := reflect t.B O
  let C' := reflect t.C O
  area A' t.B t.C + area t.A B' t.C + area t.A t.B C' = area t.A t.B t.C := by
  sorry

end NUMINAMATH_CALUDE_triangle_reflection_area_sum_l2293_229312


namespace NUMINAMATH_CALUDE_paths_through_F_l2293_229366

/-- The number of paths on a grid from (0,0) to (a,b) -/
def gridPaths (a b : ℕ) : ℕ := Nat.choose (a + b) a

/-- The coordinates of point E -/
def E : ℕ × ℕ := (0, 0)

/-- The coordinates of point F -/
def F : ℕ × ℕ := (5, 2)

/-- The coordinates of point G -/
def G : ℕ × ℕ := (6, 5)

/-- The total number of steps from E to G -/
def totalSteps : ℕ := G.1 - E.1 + G.2 - E.2

theorem paths_through_F : 
  gridPaths (F.1 - E.1) (F.2 - E.2) * gridPaths (G.1 - F.1) (G.2 - F.2) = 84 ∧
  totalSteps = 12 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_F_l2293_229366


namespace NUMINAMATH_CALUDE_circle_equation_with_tangent_conditions_l2293_229303

/-- The standard equation of a circle with center on y = (1/2)x^2 and tangent to y = 0 and x = 0 -/
theorem circle_equation_with_tangent_conditions (t : ℝ) :
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x y : ℝ), (x - t)^2 + (y - (1/2) * t^2)^2 = r^2 ↔
      ((x = 0 ∨ y = 0) → (x - t)^2 + (y - (1/2) * t^2)^2 = r^2))) →
  (∃ (s : ℝ), s = 1 ∨ s = -1) ∧
    (∀ (x y : ℝ), (x - s)^2 + (y - (1/2))^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_tangent_conditions_l2293_229303


namespace NUMINAMATH_CALUDE_harkamal_mangoes_purchase_l2293_229330

/-- The amount of mangoes purchased by Harkamal -/
def mangoes : ℕ := sorry

theorem harkamal_mangoes_purchase :
  let grapes_kg : ℕ := 8
  let grapes_rate : ℕ := 70
  let mango_rate : ℕ := 50
  let total_paid : ℕ := 1010
  grapes_kg * grapes_rate + mangoes * mango_rate = total_paid →
  mangoes = 9 := by sorry

end NUMINAMATH_CALUDE_harkamal_mangoes_purchase_l2293_229330


namespace NUMINAMATH_CALUDE_xyz_sum_square_l2293_229362

theorem xyz_sum_square (x y z : ℕ+) 
  (h_gcd : Nat.gcd x.val (Nat.gcd y.val z.val) = 1)
  (h_x_div : x.val ∣ y.val * z.val * (x.val + y.val + z.val))
  (h_y_div : y.val ∣ x.val * z.val * (x.val + y.val + z.val))
  (h_z_div : z.val ∣ x.val * y.val * (x.val + y.val + z.val))
  (h_sum_div : (x.val + y.val + z.val) ∣ (x.val * y.val * z.val)) :
  ∃ (k : ℕ), x.val * y.val * z.val * (x.val + y.val + z.val) = k * k := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_square_l2293_229362


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_plane_perp_plane_l2293_229354

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (plane_perp_plane : Plane → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_perp_plane_implies_plane_perp_plane
  (α β : Plane) (l : Line)
  (h1 : line_subset_plane l α)
  (h2 : line_perp_plane l β) :
  plane_perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_plane_perp_plane_l2293_229354


namespace NUMINAMATH_CALUDE_water_in_sport_formulation_l2293_229385

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := 3 * standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup,
    water := 60 * standard_ratio.flavoring }

/-- The amount of water in ounces given the amount of corn syrup in the sport formulation -/
def water_amount (corn_syrup_oz : ℚ) : ℚ :=
  corn_syrup_oz * (sport_ratio.water / sport_ratio.corn_syrup)

theorem water_in_sport_formulation :
  water_amount 2 = 120 :=
sorry

end NUMINAMATH_CALUDE_water_in_sport_formulation_l2293_229385


namespace NUMINAMATH_CALUDE_day_150_of_year_n_minus_2_is_thursday_l2293_229316

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Represents a day in a year -/
structure DayInYear where
  day : ℕ
  year : Year

def is_leap_year (y : Year) : Prop :=
  sorry

def day_of_week (d : DayInYear) : DayOfWeek :=
  sorry

theorem day_150_of_year_n_minus_2_is_thursday
  (N : Year)
  (h1 : day_of_week ⟨256, N⟩ = DayOfWeek.Wednesday)
  (h2 : is_leap_year ⟨N.value + 1⟩)
  (h3 : day_of_week ⟨164, ⟨N.value + 1⟩⟩ = DayOfWeek.Wednesday) :
  day_of_week ⟨150, ⟨N.value - 2⟩⟩ = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_day_150_of_year_n_minus_2_is_thursday_l2293_229316


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2293_229308

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > |a n|

theorem condition_sufficient_not_necessary :
  (∀ a : ℕ → ℝ, satisfies_condition a → is_increasing a) ∧
  (∃ a : ℕ → ℝ, is_increasing a ∧ ¬satisfies_condition a) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2293_229308


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2293_229358

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2293_229358


namespace NUMINAMATH_CALUDE_weather_forecast_probability_l2293_229386

-- Define the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the binomial probability mass function
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem weather_forecast_probability :
  let n : ℕ := 3
  let p : ℝ := 0.8
  let k : ℕ := 2
  binomial_pmf n p k = 0.384 := by sorry

end NUMINAMATH_CALUDE_weather_forecast_probability_l2293_229386


namespace NUMINAMATH_CALUDE_exponent_problem_l2293_229311

theorem exponent_problem (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(m + n) = 6 ∧ a^(3*m - 2*n) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l2293_229311


namespace NUMINAMATH_CALUDE_calculate_expression_l2293_229342

theorem calculate_expression : 
  |-Real.sqrt 3| + (1/2)⁻¹ + (Real.pi + 1)^0 - Real.tan (60 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2293_229342


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l2293_229369

theorem no_solution_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) → m = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l2293_229369


namespace NUMINAMATH_CALUDE_marbles_found_vs_lost_l2293_229301

theorem marbles_found_vs_lost (initial : ℕ) (lost : ℕ) (found : ℕ) :
  initial = 7 → lost = 8 → found = 10 → found - lost = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_found_vs_lost_l2293_229301


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2293_229348

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2293_229348


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2293_229345

def repeating_decimal_to_fraction (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  let a := repeating_decimal_to_fraction 6
  let b := repeating_decimal_to_fraction 2
  let c := repeating_decimal_to_fraction 4
  let d := repeating_decimal_to_fraction 7
  a + b - c - d = -1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2293_229345


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2293_229388

theorem larger_integer_problem (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : y - x = 8) (h5 : x * y = 168) : y = 14 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2293_229388


namespace NUMINAMATH_CALUDE_inverse_function_solution_l2293_229383

/-- Given a function f(x) = 1 / (2ax + 3b) where a and b are nonzero constants,
    prove that the solution to f⁻¹(x) = -1 is x = 1 / (-2a + 3b) -/
theorem inverse_function_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => 1 / (2 * a * x + 3 * b)
  ∃! x, f x = -1 ∧ x = 1 / (-2 * a + 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l2293_229383


namespace NUMINAMATH_CALUDE_james_amy_balloon_difference_l2293_229343

/-- 
Given that James has 232 balloons and Amy has 101 balloons, 
prove that James has 131 more balloons than Amy.
-/
theorem james_amy_balloon_difference : 
  let james_balloons : ℕ := 232
  let amy_balloons : ℕ := 101
  james_balloons - amy_balloons = 131 := by
sorry

end NUMINAMATH_CALUDE_james_amy_balloon_difference_l2293_229343


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2293_229307

theorem no_x_squared_term (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + 2 * x * y + y^2 + m * x^2 = 2 * x * y + y^2) ↔ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2293_229307


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l2293_229302

theorem sugar_solution_percentage (x : ℝ) : 
  x > 0 ∧ x < 100 →
  (3/4 * x + 1/4 * 34) / 100 = 16 / 100 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l2293_229302


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l2293_229346

theorem sin_negative_thirty_degrees :
  Real.sin (-(30 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l2293_229346


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2293_229360

theorem diophantine_equation_solutions :
  ∀ m n : ℕ+,
    (1 : ℚ) / m + (1 : ℚ) / n - (1 : ℚ) / (m * n) = 2 / 5 ↔
    ((m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4)) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2293_229360


namespace NUMINAMATH_CALUDE_power_of_eight_mod_hundred_l2293_229394

theorem power_of_eight_mod_hundred : 8^2050 % 100 = 24 := by sorry

end NUMINAMATH_CALUDE_power_of_eight_mod_hundred_l2293_229394


namespace NUMINAMATH_CALUDE_coin_flipping_sequences_l2293_229329

/-- Represents a coin configuration as a list of booleans, where true represents heads up and false represents tails up. -/
def CoinConfiguration := List Bool

/-- Represents a move as a pair of adjacent indices in the coin configuration. -/
def Move := Nat × Nat

/-- Applies a move to a coin configuration. -/
def applyMove (config : CoinConfiguration) (move : Move) : CoinConfiguration :=
  sorry

/-- Checks if a configuration is alternating heads and tails. -/
def isAlternating (config : CoinConfiguration) : Bool :=
  sorry

/-- Generates all possible sequences of 6 moves. -/
def generateMoveSequences : List (List Move) :=
  sorry

/-- Counts the number of move sequences that result in an alternating configuration. -/
def countValidSequences (initialConfig : CoinConfiguration) : Nat :=
  sorry

theorem coin_flipping_sequences :
  let initialConfig : CoinConfiguration := List.replicate 8 true
  countValidSequences initialConfig = 7680 :=
sorry

end NUMINAMATH_CALUDE_coin_flipping_sequences_l2293_229329


namespace NUMINAMATH_CALUDE_students_count_l2293_229309

/-- The total number of students in an arrangement of rows -/
def totalStudents (rows : ℕ) (studentsPerRow : ℕ) (lastRowStudents : ℕ) : ℕ :=
  (rows - 1) * studentsPerRow + lastRowStudents

/-- Theorem: Given 8 rows of students, where 7 rows have 6 students each 
    and the last row has 5 students, the total number of students is 47. -/
theorem students_count : totalStudents 8 6 5 = 47 := by
  sorry

end NUMINAMATH_CALUDE_students_count_l2293_229309


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2293_229376

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 15 → x ≥ 8 ∧ 8 < 3*8 - 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2293_229376


namespace NUMINAMATH_CALUDE_emily_big_garden_seeds_l2293_229313

/-- The number of seeds Emily started with -/
def total_seeds : ℕ := 41

/-- The number of small gardens Emily has -/
def num_small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden -/
def seeds_per_small_garden : ℕ := 4

/-- The number of seeds Emily planted in the big garden -/
def seeds_in_big_garden : ℕ := total_seeds - (num_small_gardens * seeds_per_small_garden)

theorem emily_big_garden_seeds : seeds_in_big_garden = 29 := by
  sorry

end NUMINAMATH_CALUDE_emily_big_garden_seeds_l2293_229313


namespace NUMINAMATH_CALUDE_suv_max_distance_l2293_229367

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway : Float
  city : Float

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def maxDistance (efficiency : SUVFuelEfficiency) (fuel : Float) : Float :=
  efficiency.highway * fuel

/-- Theorem stating the maximum distance an SUV can travel with given efficiency and fuel -/
theorem suv_max_distance (efficiency : SUVFuelEfficiency) (fuel : Float) :
  efficiency.highway = 12.2 →
  efficiency.city = 7.6 →
  fuel = 24 →
  maxDistance efficiency fuel = 292.8 := by
  sorry

end NUMINAMATH_CALUDE_suv_max_distance_l2293_229367


namespace NUMINAMATH_CALUDE_radio_price_rank_l2293_229340

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 43 →
  prices.card = n →
  (∀ (p q : ℕ), p ∈ prices → q ∈ prices → p ≠ q) →
  radio_price ∈ prices →
  (prices.filter (λ p => p > radio_price)).card = 8 →
  ∃ (m : ℕ), (prices.filter (λ p => p < radio_price)).card = m - 1 →
  (prices.filter (λ p => p ≤ radio_price)).card = 35 :=
by sorry

end NUMINAMATH_CALUDE_radio_price_rank_l2293_229340


namespace NUMINAMATH_CALUDE_toys_ratio_l2293_229371

def num_friends : ℕ := 4
def total_toys : ℕ := 118

theorem toys_ratio : 
  ∃ (toys_to_B : ℕ), 
    toys_to_B * num_friends = total_toys ∧ 
    (toys_to_B : ℚ) / total_toys = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_toys_ratio_l2293_229371


namespace NUMINAMATH_CALUDE_parallel_lines_coincident_lines_perpendicular_lines_l2293_229325

-- Define the lines l₁ and l₂
def l₁ (m : ℚ) : ℚ → ℚ → Prop := λ x y => (m + 3) * x + 4 * y = 5 - 3 * m
def l₂ (m : ℚ) : ℚ → ℚ → Prop := λ x y => 2 * x + (m + 5) * y = 8

-- Define parallel lines
def parallel (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∃ k : ℚ, ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)

-- Define coincident lines
def coincident (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∀ x y, l₁ x y ↔ l₂ x y

-- Define perpendicular lines
def perpendicular (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∃ k : ℚ, ∀ x y, l₁ x y → l₂ y (-x)

-- Theorem statements
theorem parallel_lines : parallel (l₁ (-7)) (l₂ (-7)) := sorry

theorem coincident_lines : coincident (l₁ (-1)) (l₂ (-1)) := sorry

theorem perpendicular_lines : perpendicular (l₁ (-13/3)) (l₂ (-13/3)) := sorry

end NUMINAMATH_CALUDE_parallel_lines_coincident_lines_perpendicular_lines_l2293_229325


namespace NUMINAMATH_CALUDE_fathers_age_l2293_229359

theorem fathers_age (f d : ℕ) (h1 : f / d = 4) (h2 : f + d + 10 = 50) : f = 32 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l2293_229359


namespace NUMINAMATH_CALUDE_tomato_drying_l2293_229326

/-- Given an initial mass of tomatoes with a certain water content,
    calculate the final mass after water content reduction -/
theorem tomato_drying (initial_mass : ℝ) (initial_water_content : ℝ) (water_reduction : ℝ)
  (h1 : initial_mass = 1000)
  (h2 : initial_water_content = 0.99)
  (h3 : water_reduction = 0.04)
  : ∃ (final_mass : ℝ), final_mass = 200 := by
  sorry


end NUMINAMATH_CALUDE_tomato_drying_l2293_229326


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2293_229380

def proposition_p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def proposition_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, proposition_p x 1 ∧ proposition_q x → 2 < x ∧ x < 3 :=
sorry

theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(proposition_p x a) → ¬(proposition_q x)) →
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2293_229380


namespace NUMINAMATH_CALUDE_rearranged_rectangles_perimeter_l2293_229361

/-- The perimeter of a figure formed by rearranging two equal rectangles cut from a square --/
theorem rearranged_rectangles_perimeter (square_side : ℝ) : square_side = 100 → 
  let rectangle_width := square_side / 2
  let rectangle_length := square_side
  let perimeter := 3 * rectangle_length + 4 * rectangle_width
  perimeter = 500 := by
sorry


end NUMINAMATH_CALUDE_rearranged_rectangles_perimeter_l2293_229361


namespace NUMINAMATH_CALUDE_paths_equal_combinations_correct_number_of_paths_l2293_229389

/-- The number of paths from (0,0) to (8,8) on an 8x8 grid -/
def number_of_paths : ℕ := 12870

/-- The size of the grid -/
def grid_size : ℕ := 8

/-- The total number of steps required to reach from (0,0) to (8,8) -/
def total_steps : ℕ := 16

/-- The number of right steps required -/
def right_steps : ℕ := 8

/-- The number of up steps required -/
def up_steps : ℕ := 8

/-- Theorem stating that the number of paths from (0,0) to (8,8) on an 8x8 grid
    is equal to the number of ways to choose 8 up steps out of 16 total steps -/
theorem paths_equal_combinations :
  number_of_paths = Nat.choose total_steps up_steps :=
sorry

/-- Theorem stating that the number of paths is correct -/
theorem correct_number_of_paths :
  number_of_paths = 12870 :=
sorry

end NUMINAMATH_CALUDE_paths_equal_combinations_correct_number_of_paths_l2293_229389


namespace NUMINAMATH_CALUDE_range_of_a_l2293_229322

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ - 1 < 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2293_229322


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_is_1_A_intersect_B_equals_B_l2293_229315

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem intersection_A_complement_B_when_a_is_1 :
  A ∩ (Set.univ \ B 1) = {x | -2 ≤ x ∧ x ≤ 0} ∪ {x | 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem A_intersect_B_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry


end NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_is_1_A_intersect_B_equals_B_l2293_229315


namespace NUMINAMATH_CALUDE_unused_bricks_fraction_l2293_229372

def bricks_used : ℝ := 20
def bricks_remaining : ℝ := 10

theorem unused_bricks_fraction :
  bricks_remaining / (bricks_used + bricks_remaining) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unused_bricks_fraction_l2293_229372


namespace NUMINAMATH_CALUDE_complex_equation_roots_l2293_229306

theorem complex_equation_roots : 
  let z₁ : ℂ := (1 + 2 * Real.sqrt 7 - Real.sqrt 7 * I) / 2
  let z₂ : ℂ := (1 - 2 * Real.sqrt 7 + Real.sqrt 7 * I) / 2
  (z₁^2 - z₁ = 3 - 7*I) ∧ (z₂^2 - z₂ = 3 - 7*I) := by
  sorry


end NUMINAMATH_CALUDE_complex_equation_roots_l2293_229306


namespace NUMINAMATH_CALUDE_boat_speed_l2293_229336

/-- Given a boat that travels 10 km/hr downstream and 4 km/hr upstream,
    its speed in still water is 7 km/hr. -/
theorem boat_speed (downstream upstream : ℝ) 
  (h_downstream : downstream = 10)
  (h_upstream : upstream = 4) :
  (downstream + upstream) / 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_l2293_229336


namespace NUMINAMATH_CALUDE_pictures_per_album_l2293_229381

/-- Given a total of 20 pictures sorted equally into 5 albums, prove that each album contains 4 pictures. -/
theorem pictures_per_album :
  let total_pictures : ℕ := 7 + 13
  let num_albums : ℕ := 5
  let pictures_per_album : ℕ := total_pictures / num_albums
  pictures_per_album = 4 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l2293_229381


namespace NUMINAMATH_CALUDE_library_books_theorem_l2293_229333

/-- The number of books taken by the librarian -/
def books_taken : ℕ := 10

/-- The number of books that can fit on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves needed for the remaining books -/
def shelves_needed : ℕ := 9

/-- The total number of books to put away -/
def total_books : ℕ := 46

theorem library_books_theorem :
  total_books = books_per_shelf * shelves_needed + books_taken := by
  sorry

end NUMINAMATH_CALUDE_library_books_theorem_l2293_229333


namespace NUMINAMATH_CALUDE_fourth_sample_seat_number_l2293_229305

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_samples : Finset ℕ
  interval : ℕ

/-- The theorem to prove -/
theorem fourth_sample_seat_number
  (s : SystematicSampling)
  (h_total : s.total_students = 56)
  (h_size : s.sample_size = 4)
  (h_known : s.known_samples = {3, 17, 45})
  (h_interval : s.interval = s.total_students / s.sample_size) :
  ∃ (n : ℕ), n ∈ s.known_samples ∧ (n + s.interval) % s.total_students = 31 :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_seat_number_l2293_229305


namespace NUMINAMATH_CALUDE_facebook_employee_bonus_l2293_229324

/-- Represents the Facebook employee bonus problem -/
theorem facebook_employee_bonus (
  total_employees : ℕ
  ) (annual_earnings : ℕ) (bonus_percentage : ℚ) (bonus_per_mother : ℕ) :
  total_employees = 3300 →
  annual_earnings = 5000000 →
  bonus_percentage = 1/4 →
  bonus_per_mother = 1250 →
  ∃ (non_mother_employees : ℕ),
    non_mother_employees = 1200 ∧
    non_mother_employees = 
      (2/3 : ℚ) * total_employees - 
      (bonus_percentage * annual_earnings) / bonus_per_mother :=
by sorry


end NUMINAMATH_CALUDE_facebook_employee_bonus_l2293_229324


namespace NUMINAMATH_CALUDE_min_total_cards_problem_l2293_229363

def min_total_cards (carlos_cards : ℕ) (matias_diff : ℕ) (ella_multiplier : ℕ) (divisor : ℕ) : ℕ :=
  let matias_cards := carlos_cards - matias_diff
  let jorge_cards := matias_cards
  let ella_cards := ella_multiplier * (jorge_cards + matias_cards)
  let total_cards := carlos_cards + matias_cards + jorge_cards + ella_cards
  ((total_cards + divisor - 1) / divisor) * divisor

theorem min_total_cards_problem :
  min_total_cards 20 6 2 15 = 105 := by sorry

end NUMINAMATH_CALUDE_min_total_cards_problem_l2293_229363


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2293_229390

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : angle_between a b = Real.pi / 4)
  (h2 : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2)
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 3) :
  Real.sqrt (((2 * a.1 - b.1) ^ 2) + ((2 * a.2 - b.2) ^ 2)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2293_229390


namespace NUMINAMATH_CALUDE_quadratic_a_value_l2293_229351

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ
  vertex_form : ∀ x, a * (x - h)^2 + k = a * x^2 + ((-2 * a * h) * x) + (a * h^2 + k)
  passes_through : a * (x₀ - h)^2 + k = y₀

/-- The theorem stating that for a quadratic function with vertex (2, 5) 
    passing through (-1, -20), the value of 'a' is -25/9 -/
theorem quadratic_a_value (f : QuadraticFunction) 
    (vertex_h : f.h = 2) 
    (vertex_k : f.k = 5) 
    (point_x : f.x₀ = -1) 
    (point_y : f.y₀ = -20) : 
    f.a = -25/9 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_a_value_l2293_229351


namespace NUMINAMATH_CALUDE_product_of_eights_place_values_l2293_229318

/-- The place value of a digit in a decimal number -/
def place_value (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (10 : ℚ) ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 780.38

/-- The theorem stating that the product of place values of two 8's in 780.38 is 6.4 -/
theorem product_of_eights_place_values :
  (place_value 8 1) * (place_value 8 (-2)) = 6.4 := by sorry

end NUMINAMATH_CALUDE_product_of_eights_place_values_l2293_229318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l2293_229356

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The main theorem -/
theorem arithmetic_sequence_increasing_iff_a1_lt_a3 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 < a 3 ↔ MonotonicallyIncreasing a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l2293_229356


namespace NUMINAMATH_CALUDE_yarn_ball_ratio_l2293_229365

/-- Given three balls of yarn, where:
    - The third ball is three times as large as the first ball
    - 27 feet of yarn was used for the third ball
    - 18 feet of yarn was used for the second ball
    Prove that the ratio of the size of the first ball to the size of the second ball is 1:2 -/
theorem yarn_ball_ratio :
  ∀ (first_ball second_ball third_ball : ℝ),
  third_ball = 3 * first_ball →
  third_ball = 27 →
  second_ball = 18 →
  first_ball / second_ball = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_yarn_ball_ratio_l2293_229365


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2293_229341

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x * (x - 1) < 0}

-- Theorem statement
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2293_229341


namespace NUMINAMATH_CALUDE_psychologist_pricing_l2293_229332

theorem psychologist_pricing (F A : ℝ) 
  (h1 : F + 4 * A = 300)  -- 5 hours of therapy costs $300
  (h2 : F + 2 * A = 188)  -- 3 hours of therapy costs $188
  : F - A = 20 := by
  sorry

end NUMINAMATH_CALUDE_psychologist_pricing_l2293_229332


namespace NUMINAMATH_CALUDE_chord_of_ellipse_l2293_229396

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 2*y^2 - 4 = 0

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The midpoint of a line segment -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem chord_of_ellipse :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  ellipse_equation x₁ y₁ ∧ 
  ellipse_equation x₂ y₂ ∧
  (∀ x y : ℝ, line_equation x y ↔ is_midpoint 1 1 x₁ y₁ x₂ y₂) →
  line_equation x₁ y₁ ∧ line_equation x₂ y₂ := by sorry

end NUMINAMATH_CALUDE_chord_of_ellipse_l2293_229396


namespace NUMINAMATH_CALUDE_odd_indexed_sum_limit_l2293_229320

/-- For an infinite geometric sequence {a_n} where a_1 = √3 and a_2 = 1,
    the limit of the sum of odd-indexed terms as n approaches infinity is (3√3)/2 -/
theorem odd_indexed_sum_limit (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = Real.sqrt 3 →                      -- first term condition
  a 2 = 1 →                                -- second term condition
  (∑' n : ℕ, a (2 * n + 1)) = 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_odd_indexed_sum_limit_l2293_229320


namespace NUMINAMATH_CALUDE_parallel_line_with_y_intercept_l2293_229338

/-- Given a line mx + ny + 1 = 0 parallel to 4x + 3y + 5 = 0 with y-intercept 1/3, prove m = -4 and n = -3 -/
theorem parallel_line_with_y_intercept (m n : ℝ) : 
  (∀ x y, m * x + n * y + 1 = 0 ↔ 4 * x + 3 * y + 5 = 0) →  -- parallel condition
  (∃ y, m * 0 + n * y + 1 = 0 ∧ y = 1/3) →                  -- y-intercept condition
  m = -4 ∧ n = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_with_y_intercept_l2293_229338


namespace NUMINAMATH_CALUDE_orange_straws_count_l2293_229373

/-- The number of orange straws needed for each mat -/
def orange_straws : ℕ := 30

/-- The number of red straws needed for each mat -/
def red_straws : ℕ := 20

/-- The number of green straws needed for each mat -/
def green_straws : ℕ := orange_straws / 2

/-- The total number of mats -/
def total_mats : ℕ := 10

/-- The total number of straws needed for all mats -/
def total_straws : ℕ := 650

theorem orange_straws_count :
  orange_straws = 30 ∧
  red_straws = 20 ∧
  green_straws = orange_straws / 2 ∧
  total_mats * (red_straws + orange_straws + green_straws) = total_straws :=
by sorry

end NUMINAMATH_CALUDE_orange_straws_count_l2293_229373


namespace NUMINAMATH_CALUDE_businessmen_drinking_none_l2293_229349

theorem businessmen_drinking_none (total : ℕ) (coffee tea soda coffee_tea tea_soda coffee_soda all_three : ℕ) : 
  total = 30 ∧ 
  coffee = 15 ∧ 
  tea = 12 ∧ 
  soda = 8 ∧ 
  coffee_tea = 7 ∧ 
  tea_soda = 3 ∧ 
  coffee_soda = 2 ∧ 
  all_three = 1 → 
  total - (coffee + tea + soda - coffee_tea - tea_soda - coffee_soda + all_three) = 6 := by
sorry

end NUMINAMATH_CALUDE_businessmen_drinking_none_l2293_229349


namespace NUMINAMATH_CALUDE_third_number_proof_l2293_229310

theorem third_number_proof (a b c : ℝ) : 
  (a + b + c) / 3 = 48 → 
  (a + b) / 2 = 56 → 
  c = 32 := by
sorry

end NUMINAMATH_CALUDE_third_number_proof_l2293_229310


namespace NUMINAMATH_CALUDE_avery_building_time_l2293_229364

theorem avery_building_time (tom_time : ℝ) (joint_work_time : ℝ) (tom_remaining_time : ℝ) 
  (h1 : tom_time = 2)
  (h2 : joint_work_time = 1)
  (h3 : tom_remaining_time = 20.000000000000007 / 60) :
  ∃ (avery_time : ℝ), 
    1 / avery_time + 1 / tom_time + (tom_remaining_time / tom_time) = 1 ∧ 
    avery_time = 3 := by
sorry

end NUMINAMATH_CALUDE_avery_building_time_l2293_229364


namespace NUMINAMATH_CALUDE_odd_squares_sum_power_of_two_l2293_229392

theorem odd_squares_sum_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ x y : ℤ, Odd x ∧ Odd y ∧ x^2 + 7*y^2 = 2^n := by
  sorry

end NUMINAMATH_CALUDE_odd_squares_sum_power_of_two_l2293_229392


namespace NUMINAMATH_CALUDE_remainder_double_mod_seven_l2293_229323

theorem remainder_double_mod_seven (n : ℤ) (h : n % 7 = 2) : (2 * n) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_mod_seven_l2293_229323


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2293_229331

/-- Three points (x1, y1), (x2, y2), and (x3, y3) are collinear if and only if
    the slope between any two pairs of points is equal. -/
def collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

/-- The theorem states that for three collinear points (1, 2), (3, k), and (10, 5),
    the value of k must be 8/3. -/
theorem collinear_points_k_value :
  collinear 1 2 3 k 10 5 → k = 8/3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2293_229331


namespace NUMINAMATH_CALUDE_smallest_m_with_integer_price_l2293_229384

theorem smallest_m_with_integer_price : ∃ m : ℕ+, 
  (∀ k < m, ¬∃ x : ℕ, (107 : ℚ) * x = 100 * k) ∧
  (∃ x : ℕ, (107 : ℚ) * x = 100 * m) ∧
  m = 107 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_with_integer_price_l2293_229384
