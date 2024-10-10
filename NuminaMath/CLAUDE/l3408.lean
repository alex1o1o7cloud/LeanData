import Mathlib

namespace john_paid_21_dollars_l3408_340879

/-- Calculates the amount John paid for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars) * cost_per_bar

/-- Proves that John paid $21 for the candy bars -/
theorem john_paid_21_dollars :
  john_payment 20 6 (3/2) = 21 := by
  sorry

end john_paid_21_dollars_l3408_340879


namespace arithmetic_sequence_common_difference_l3408_340809

/-- An arithmetic sequence with a_5 = 3 and a_6 = -2 has common difference -5 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a5 : a 5 = 3) 
  (h_a6 : a 6 = -2) : 
  a 6 - a 5 = -5 := by
sorry

end arithmetic_sequence_common_difference_l3408_340809


namespace range_of_a_l3408_340820

theorem range_of_a (z : ℂ) (a : ℝ) : 
  z.im ≠ 0 →  -- z is imaginary
  (z + 3 / (2 * z)).im = 0 →  -- z + 3/(2z) is real
  (z + 3 / (2 * z))^2 - 2 * a * (z + 3 / (2 * z)) + 1 - 3 * a = 0 →  -- root condition
  (a ≥ (Real.sqrt 13 - 3) / 2 ∨ a ≤ -(Real.sqrt 13 + 3) / 2) := by
sorry

end range_of_a_l3408_340820


namespace max_visible_cubes_12x12x12_l3408_340866

/-- Represents a cube composed of unit cubes -/
structure Cube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def maxVisibleUnitCubes (c : Cube) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem: For a 12×12×12 cube, the maximum number of visible unit cubes is 400 -/
theorem max_visible_cubes_12x12x12 :
  let c : Cube := ⟨12⟩
  maxVisibleUnitCubes c = 400 := by
  sorry

#eval maxVisibleUnitCubes ⟨12⟩

end max_visible_cubes_12x12x12_l3408_340866


namespace phone_purchase_problem_l3408_340805

/-- Represents the purchase price of phone models -/
structure PhonePrices where
  modelA : ℕ
  modelB : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelACount : ℕ
  modelBCount : ℕ

def totalCost (prices : PhonePrices) (plan : PurchasePlan) : ℕ :=
  prices.modelA * plan.modelACount + prices.modelB * plan.modelBCount

theorem phone_purchase_problem (prices : PhonePrices) : 
  (prices.modelA * 2 + prices.modelB = 5000) →
  (prices.modelA * 3 + prices.modelB * 2 = 8000) →
  (prices.modelA = 2000 ∧ prices.modelB = 1000) ∧
  (∃ (plans : List PurchasePlan), 
    plans.length = 3 ∧
    (∀ plan ∈ plans, 
      plan.modelACount + plan.modelBCount = 20 ∧
      24000 ≤ totalCost prices plan ∧
      totalCost prices plan ≤ 26000) ∧
    (∀ plan : PurchasePlan, 
      plan.modelACount + plan.modelBCount = 20 ∧
      24000 ≤ totalCost prices plan ∧
      totalCost prices plan ≤ 26000 →
      plan ∈ plans)) :=
by sorry

end phone_purchase_problem_l3408_340805


namespace pump_x_time_is_4_hours_l3408_340881

/-- Represents the rate of a pump in terms of fraction of total water pumped per hour -/
structure PumpRate where
  rate : ℝ
  rate_positive : rate > 0

/-- Represents the scenario of two pumps working on draining a flooded basement -/
structure BasementPumpScenario where
  pump_x : PumpRate
  pump_y : PumpRate
  total_water : ℝ
  total_water_positive : total_water > 0
  y_alone_time : ℝ
  y_alone_time_eq : pump_y.rate * y_alone_time = total_water
  combined_time : ℝ
  combined_time_eq : (pump_x.rate + pump_y.rate) * combined_time = total_water / 2

/-- The main theorem stating that pump X takes 4 hours to pump out half the water -/
theorem pump_x_time_is_4_hours (scenario : BasementPumpScenario) : 
  scenario.pump_x.rate * 4 = scenario.total_water / 2 ∧ 
  scenario.y_alone_time = 20 ∧ 
  scenario.combined_time = 3 := by
  sorry

end pump_x_time_is_4_hours_l3408_340881


namespace cubic_equation_roots_l3408_340801

/-- Given a cubic equation with two known roots, find the value of k and the third root -/
theorem cubic_equation_roots (k : ℝ) : 
  (∀ x : ℝ, x^3 + 5*x^2 + k*x - 12 = 0 ↔ x = 3 ∨ x = -2 ∨ x = -6) →
  k = -12 :=
by sorry

end cubic_equation_roots_l3408_340801


namespace box_with_balls_l3408_340869

theorem box_with_balls (total : ℕ) (white : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 100 →
  blue = white + 12 →
  red = 2 * blue →
  total = white + blue + red →
  white = 16 := by
sorry

end box_with_balls_l3408_340869


namespace simplify_expression_l3408_340861

theorem simplify_expression : 4^4 * 9^4 * 4^9 * 9^9 = 36^13 := by
  sorry

end simplify_expression_l3408_340861


namespace last_three_digits_of_7_to_7500_l3408_340847

theorem last_three_digits_of_7_to_7500 (h : 7^500 ≡ 1 [ZMOD 1250]) :
  7^7500 ≡ 1 [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_to_7500_l3408_340847


namespace lines_parallel_l3408_340834

/-- Two lines in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The property of two lines being parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

theorem lines_parallel : 
  let l1 : Line := { slope := 2, intercept := 1 }
  let l2 : Line := { slope := 2, intercept := 5 }
  parallel l1 l2 := by
  sorry

end lines_parallel_l3408_340834


namespace midpoint_sum_equals_vertex_sum_l3408_340821

theorem midpoint_sum_equals_vertex_sum (a b c d : ℝ) :
  a + b + c + d = 15 →
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 := by
  sorry

end midpoint_sum_equals_vertex_sum_l3408_340821


namespace utilities_percentage_l3408_340886

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  salaries : ℝ
  research_and_development : ℝ
  equipment : ℝ
  supplies : ℝ
  transportation_degrees : ℝ
  total_budget : ℝ

/-- The theorem stating that given the specific budget allocation, the percentage spent on utilities is 5% -/
theorem utilities_percentage (budget : BudgetAllocation) : 
  budget.salaries = 60 ∧ 
  budget.research_and_development = 9 ∧ 
  budget.equipment = 4 ∧ 
  budget.supplies = 2 ∧ 
  budget.transportation_degrees = 72 ∧ 
  budget.total_budget = 100 →
  100 - (budget.salaries + budget.research_and_development + budget.equipment + budget.supplies + (budget.transportation_degrees * 100 / 360)) = 5 := by
  sorry

end utilities_percentage_l3408_340886


namespace alternative_bases_l3408_340800

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem alternative_bases
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, a + c, a] ∧
  Submodule.span ℝ {a + b, a + c, a} = ⊤ ∧
  LinearIndependent ℝ ![a - b + c, a - b, a + c] ∧
  Submodule.span ℝ {a - b + c, a - b, a + c} = ⊤ := by
sorry

end alternative_bases_l3408_340800


namespace unique_function_is_identity_l3408_340819

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that f(mn) = f(m)f(n) for all positive integers m and n -/
def IsMultiplicative (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

/-- The property that f^(n^k)(n) = n for all positive integers n -/
def SatisfiesExpProperty (f : PositiveIntFunction) (k : ℕ+) : Prop :=
  ∀ n : ℕ+, (f^[n^k.val]) n = n

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := id

theorem unique_function_is_identity (k : ℕ+) :
  ∃! f : PositiveIntFunction, IsMultiplicative f ∧ SatisfiesExpProperty f k →
  f = identityFunction :=
sorry

end unique_function_is_identity_l3408_340819


namespace seating_arrangements_l3408_340863

/-- The number of ways to arrange n elements --/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := n.choose k

/-- The number of seating arrangements for 5 people in 5 seats --/
def totalArrangements : ℕ := arrangements 5

/-- The number of arrangements where 3 people are in their numbered seats --/
def threeInPlace : ℕ := choose 5 3 * arrangements 2

/-- The number of arrangements where all 5 people are in their numbered seats --/
def allInPlace : ℕ := 1

theorem seating_arrangements :
  totalArrangements - threeInPlace - allInPlace = 109 := by
  sorry

end seating_arrangements_l3408_340863


namespace geometric_sequence_first_term_l3408_340837

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0))  -- Geometric sequence condition
  (h_4th : a 3 = 81)  -- Fourth term is 81 (index starts at 0)
  (h_5th : a 4 = 162)  -- Fifth term is 162
  : a 0 = 10.125 :=  -- First term is 10.125
by sorry

end geometric_sequence_first_term_l3408_340837


namespace two_digit_sum_to_four_digit_sum_l3408_340871

/-- Given two two-digit numbers that sum to 137, prove that the sum of the four-digit numbers
    formed by concatenating these digits in order and in reverse order is 13837. -/
theorem two_digit_sum_to_four_digit_sum
  (A B C D : ℕ)
  (h_AB_two_digit : A * 10 + B < 100)
  (h_CD_two_digit : C * 10 + D < 100)
  (h_sum : A * 10 + B + C * 10 + D = 137) :
  (A * 1000 + B * 100 + C * 10 + D) + (C * 1000 + D * 100 + A * 10 + B) = 13837 := by
  sorry


end two_digit_sum_to_four_digit_sum_l3408_340871


namespace shape_reassembly_l3408_340870

/-- Represents a geometric shape with an area -/
structure Shape :=
  (area : ℝ)

/-- Represents the original rectangle -/
def rectangle : Shape :=
  { area := 1 }

/-- Represents the square -/
def square : Shape :=
  { area := 0.5 }

/-- Represents the triangle with a hole -/
def triangleWithHole : Shape :=
  { area := 0.5 }

/-- Represents the two parts after cutting the rectangle -/
def part1 : Shape :=
  { area := 0.5 }

def part2 : Shape :=
  { area := 0.5 }

theorem shape_reassembly :
  (rectangle.area = part1.area + part2.area) ∧
  (square.area = part1.area) ∧
  (triangleWithHole.area = part2.area) := by
  sorry

#check shape_reassembly

end shape_reassembly_l3408_340870


namespace max_a_value_l3408_340859

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem max_a_value :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≤ 3/2 ∧ ∀ b : ℝ, (∀ x : ℝ, determinant (x - 1) (b - 2) (b + 1) x ≥ 1) → b ≤ a :=
by sorry

end max_a_value_l3408_340859


namespace oxygen_atom_diameter_scientific_notation_l3408_340808

theorem oxygen_atom_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000000148 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -10 := by
  sorry

end oxygen_atom_diameter_scientific_notation_l3408_340808


namespace value_of_t_l3408_340874

theorem value_of_t (u m j : ℝ) (A t : ℝ) (h : A = u^m / (2 + j)^t) :
  t = Real.log (u^m / A) / Real.log (2 + j) := by
  sorry

end value_of_t_l3408_340874


namespace mice_eaten_in_decade_l3408_340830

/-- Calculates the number of mice eaten by a snake in a decade -/
theorem mice_eaten_in_decade (weeks_per_mouse : ℕ) (years_per_decade : ℕ) (weeks_per_year : ℕ) : 
  weeks_per_mouse = 4 → years_per_decade = 10 → weeks_per_year = 52 →
  (years_per_decade * weeks_per_year) / weeks_per_mouse = 130 := by
sorry

end mice_eaten_in_decade_l3408_340830


namespace division_and_multiplication_l3408_340838

theorem division_and_multiplication (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (result : ℕ) : 
  dividend = 24 → 
  divisor = 3 → 
  dividend = divisor * quotient → 
  result = quotient * 5 → 
  quotient = 8 ∧ result = 40 := by
sorry

end division_and_multiplication_l3408_340838


namespace train_length_and_speed_l3408_340852

/-- A train passes by an observer in t₁ seconds and through a bridge of length a meters in t₂ seconds at a constant speed. This theorem proves the formulas for the train's length and speed. -/
theorem train_length_and_speed (t₁ t₂ a : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > t₁) (h₃ : a > 0) :
  ∃ (L V : ℝ),
    L = (a * t₁) / (t₂ - t₁) ∧
    V = a / (t₂ - t₁) ∧
    L / t₁ = V ∧
    (L + a) / t₂ = V :=
by sorry

end train_length_and_speed_l3408_340852


namespace power_of_three_mod_eight_l3408_340839

theorem power_of_three_mod_eight : 3^1988 ≡ 1 [MOD 8] := by sorry

end power_of_three_mod_eight_l3408_340839


namespace knitting_time_for_two_pairs_l3408_340895

/-- Given A's and B's knitting rates, prove the time needed to knit two pairs of socks together -/
theorem knitting_time_for_two_pairs 
  (rate_A : ℚ) -- A's knitting rate in pairs per day
  (rate_B : ℚ) -- B's knitting rate in pairs per day
  (h_rate_A : rate_A = 1/3) -- A can knit a pair in 3 days
  (h_rate_B : rate_B = 1/6) -- B can knit a pair in 6 days
  : (2 : ℚ) / (rate_A + rate_B) = 4 := by
  sorry

end knitting_time_for_two_pairs_l3408_340895


namespace middle_number_problem_l3408_340899

theorem middle_number_problem (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_sum1 : x + y = 24)
  (h_sum2 : x + z = 29)
  (h_sum3 : y + z = 34) :
  y = 14.5 := by
sorry

end middle_number_problem_l3408_340899


namespace product_of_difference_and_sum_of_squares_l3408_340814

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end product_of_difference_and_sum_of_squares_l3408_340814


namespace f_value_at_7_l3408_340862

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^7 + c * x^3 + d * x - 6

-- State the theorem
theorem f_value_at_7 (a b c d : ℝ) :
  f a b c d (-7) = 10 → f a b c d 7 = 11529580 * a - 22 := by
  sorry

end f_value_at_7_l3408_340862


namespace good_characterization_l3408_340829

def is_good (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → (a + 1) ∣ (n + 1)

theorem good_characterization :
  ∀ n : ℕ, n ≥ 1 → (is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1)) :=
by sorry

end good_characterization_l3408_340829


namespace cyclic_sum_inequality_l3408_340889

open Real

/-- Cyclic sum of a function over three variables -/
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hsum : a + b + c = 3) :
    cyclicSum (fun x y z => 1 / (2 * x^2 + y^2 + z^2)) a b c ≤ 3/4 := by
  sorry

end cyclic_sum_inequality_l3408_340889


namespace no_real_roots_of_composition_l3408_340833

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def f (a b c : ℝ) (ha : a ≠ 0) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_of_composition
  (a b c : ℝ) (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c ha x ≠ x) :
  ∀ x : ℝ, f a b c ha (f a b c ha x) ≠ x :=
sorry

end no_real_roots_of_composition_l3408_340833


namespace bbq_ice_packs_l3408_340897

/-- Given a BBQ scenario, calculate the number of 1-pound bags of ice in a pack -/
theorem bbq_ice_packs (people : ℕ) (ice_per_person : ℕ) (pack_price : ℚ) (total_spent : ℚ) :
  people = 15 →
  ice_per_person = 2 →
  pack_price = 3 →
  total_spent = 9 →
  (people * ice_per_person) / (total_spent / pack_price) = 10 := by
  sorry

#check bbq_ice_packs

end bbq_ice_packs_l3408_340897


namespace total_toys_cost_l3408_340812

def toy_cars_cost : ℚ := 14.88
def skateboard_cost : ℚ := 4.88
def toy_trucks_cost : ℚ := 5.86
def pants_cost : ℚ := 14.55

theorem total_toys_cost :
  toy_cars_cost + skateboard_cost + toy_trucks_cost = 25.62 := by sorry

end total_toys_cost_l3408_340812


namespace min_value_quadratic_expression_l3408_340867

/-- The minimum value of 2x^2 + 4xy + 5y^2 - 8x - 6y over all real numbers x and y is 3 -/
theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 ∧
  ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 4 * x₀ * y₀ + 5 * y₀^2 - 8 * x₀ - 6 * y₀ = 3 :=
by sorry

end min_value_quadratic_expression_l3408_340867


namespace fruit_salad_mixture_weight_l3408_340811

theorem fruit_salad_mixture_weight 
  (apple peach grape : ℝ) 
  (h1 : apple / grape = 12 / 7)
  (h2 : peach / grape = 8 / 7)
  (h3 : apple = grape + 10) :
  apple + peach + grape = 54 := by
sorry

end fruit_salad_mixture_weight_l3408_340811


namespace largest_integer_four_digits_base_seven_l3408_340813

def has_four_digits_base_seven (n : ℕ) : Prop :=
  7^3 ≤ n^2 ∧ n^2 < 7^4

theorem largest_integer_four_digits_base_seven :
  ∃ M : ℕ, has_four_digits_base_seven M ∧
    ∀ n : ℕ, has_four_digits_base_seven n → n ≤ M ∧
    M = 48 :=
sorry

end largest_integer_four_digits_base_seven_l3408_340813


namespace fuel_station_cost_fuel_station_cost_example_l3408_340872

/-- Calculates the total cost of filling up vehicles at a fuel station -/
theorem fuel_station_cost (service_cost : ℝ) (fuel_cost : ℝ) (minivan_count : ℕ) (truck_count : ℕ) 
  (minivan_tank : ℝ) (truck_tank_increase : ℝ) : ℝ :=
  let truck_tank := minivan_tank * (1 + truck_tank_increase)
  let minivan_fuel_cost := minivan_count * minivan_tank * fuel_cost
  let truck_fuel_cost := truck_count * truck_tank * fuel_cost
  let total_service_cost := (minivan_count + truck_count) * service_cost
  minivan_fuel_cost + truck_fuel_cost + total_service_cost

/-- Proves that the total cost for filling up 3 mini-vans and 2 trucks is $347.20 -/
theorem fuel_station_cost_example : 
  fuel_station_cost 2.10 0.70 3 2 65 1.20 = 347.20 := by
  sorry

end fuel_station_cost_fuel_station_cost_example_l3408_340872


namespace quadratic_two_real_roots_l3408_340858

/-- A quadratic equation kx² + (2k-1)x + k = 0 has two real roots if and only if k ≤ 1/4 and k ≠ 0 -/
theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, k * z^2 + (2*k - 1) * z + k = 0 ↔ (z = x ∨ z = y)) ↔ 
  (k ≤ 1/4 ∧ k ≠ 0) :=
sorry

end quadratic_two_real_roots_l3408_340858


namespace blank_value_l3408_340843

theorem blank_value : ∃ x : ℝ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end blank_value_l3408_340843


namespace vector_problem_l3408_340855

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (a b : ℝ × ℝ) : Prop :=
  ¬∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_problem (a b c : ℝ × ℝ) (x : ℝ) :
  NonCollinear a b →
  a = (1, 2) →
  b = (x, 6) →
  ‖a - b‖ = 2 * Real.sqrt 5 →
  c = (2 • a) + b →
  c = (1, 10) := by
  sorry

end vector_problem_l3408_340855


namespace ball_probabilities_l3408_340810

/-- Given a bag of balls with the following properties:
  - There are 10 balls in total.
  - The probability of drawing a black ball is 2/5.
  - The probability of drawing at least one white ball when drawing two balls is 19/20.

  This theorem proves:
  1. The probability of drawing two black balls is 6/45.
  2. The number of white balls is 5.
-/
theorem ball_probabilities
  (total_balls : ℕ)
  (prob_black : ℚ)
  (prob_at_least_one_white : ℚ)
  (h_total : total_balls = 10)
  (h_prob_black : prob_black = 2 / 5)
  (h_prob_white : prob_at_least_one_white = 19 / 20) :
  (∃ (black_balls white_balls : ℕ),
    black_balls + white_balls ≤ total_balls ∧
    (black_balls : ℚ) / total_balls = prob_black ∧
    1 - (total_balls - white_balls) * (total_balls - white_balls - 1) / (total_balls * (total_balls - 1)) = prob_at_least_one_white ∧
    black_balls * (black_balls - 1) / (total_balls * (total_balls - 1)) = 6 / 45 ∧
    white_balls = 5) :=
by sorry

end ball_probabilities_l3408_340810


namespace range_of_a_l3408_340844

theorem range_of_a (A B : Set ℝ) (a : ℝ) 
  (h1 : A = {x : ℝ | x ≤ 2})
  (h2 : B = {x : ℝ | x ≥ a})
  (h3 : A ⊆ B) : 
  a ≤ 2 := by
sorry

end range_of_a_l3408_340844


namespace initial_number_proof_l3408_340831

theorem initial_number_proof : ∃ n : ℕ, n ≥ 102 ∧ (n - 5) % 97 = 0 ∧ ∀ m : ℕ, m < n → (m - 5) % 97 ≠ 0 := by
  sorry

end initial_number_proof_l3408_340831


namespace x_value_l3408_340891

theorem x_value : ∃ x : ℝ, 3 * x = (26 - x) + 10 ∧ x = 9 := by
  sorry

end x_value_l3408_340891


namespace line_with_equal_intercepts_through_intersection_l3408_340854

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection point of two lines -/
def intersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- Check if a point lies on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := sorry

/-- Check if a line has equal intercepts on coordinate axes -/
def equal_intercepts (l : Line) : Prop := sorry

theorem line_with_equal_intercepts_through_intersection 
  (l1 l2 : Line) 
  (h1 : l1 = Line.mk 1 2 (-11)) 
  (h2 : l2 = Line.mk 2 1 (-10)) :
  ∃ (l : Line), 
    on_line (intersection l1 l2) l ∧ 
    equal_intercepts l ∧ 
    (l = Line.mk 4 (-3) 0 ∨ l = Line.mk 1 1 (-7)) := by
  sorry

end line_with_equal_intercepts_through_intersection_l3408_340854


namespace smallest_n_square_and_cube_l3408_340873

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 4 * x = y^2) → (∃ (z : ℕ), 5 * x = z^3) → x ≥ n) ∧
  n = 400 := by
sorry

end smallest_n_square_and_cube_l3408_340873


namespace problem_statement_l3408_340832

theorem problem_statement (x y z : ℝ) 
  (h1 : x^2 + 1/x^2 = 7)
  (h2 : x*y = 1)
  (h3 : z^2 + 1/z^2 = 9) :
  x^4 + y^4 - z^4 = 15 := by
sorry

end problem_statement_l3408_340832


namespace farmer_animals_l3408_340876

theorem farmer_animals (goats cows pigs : ℕ) : 
  pigs = 2 * cows ∧ 
  cows = goats + 4 ∧ 
  goats + cows + pigs = 56 →
  goats = 11 := by
sorry

end farmer_animals_l3408_340876


namespace inequality_system_solution_l3408_340825

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 6 < 4*x - 3 ∧ x > m) ↔ x > 3) → m ≤ 3 := by
  sorry

end inequality_system_solution_l3408_340825


namespace negation_of_universal_proposition_l3408_340868

open Set Real

theorem negation_of_universal_proposition (f : ℝ → ℝ) :
  (¬ (∀ x ∈ Ioo 0 (π / 2), f x < 0)) ↔ (∃ x ∈ Ioo 0 (π / 2), f x ≥ 0) := by
  sorry

end negation_of_universal_proposition_l3408_340868


namespace childrens_ticket_price_l3408_340848

/-- The cost of a children's ticket to the aquarium -/
def childrens_ticket_cost : ℝ := 20

/-- The cost of an adult ticket to the aquarium -/
def adult_ticket_cost : ℝ := 35

/-- The number of adults in Violet's family -/
def num_adults : ℕ := 1

/-- The number of children in Violet's family -/
def num_children : ℕ := 6

/-- The total cost of separate tickets for Violet's family -/
def total_separate_cost : ℝ := 155

theorem childrens_ticket_price : 
  childrens_ticket_cost * num_children + adult_ticket_cost * num_adults = total_separate_cost :=
by sorry

end childrens_ticket_price_l3408_340848


namespace inequality_solution_set_l3408_340802

theorem inequality_solution_set (x : ℝ) :
  x ≠ 0 →
  ((2 * x - 5) * (x - 3)) / x ≥ 0 ↔ (0 < x ∧ x ≤ 5/2) ∨ (x ≥ 3) :=
by sorry

end inequality_solution_set_l3408_340802


namespace investment_proof_l3408_340887

/-- Represents the total amount invested -/
def total_investment : ℝ := 15280

/-- Represents the amount invested at 6% rate -/
def investment_at_6_percent : ℝ := 8200

/-- Represents the total simple interest yield in one year -/
def total_interest : ℝ := 1023

/-- First investment rate -/
def rate_1 : ℝ := 0.06

/-- Second investment rate -/
def rate_2 : ℝ := 0.075

theorem investment_proof :
  total_investment * rate_1 * (investment_at_6_percent / total_investment) +
  total_investment * rate_2 * (1 - investment_at_6_percent / total_investment) = total_interest :=
by sorry

end investment_proof_l3408_340887


namespace mans_rate_in_still_water_l3408_340850

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 19) 
  (h2 : speed_against_stream = 11) : 
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

end mans_rate_in_still_water_l3408_340850


namespace ratio_c_d_equals_one_over_320_l3408_340896

theorem ratio_c_d_equals_one_over_320 (a b c d : ℝ) : 
  8 = 0.02 * a → 
  2 = 0.08 * b → 
  d = 0.05 * a → 
  c = b / a → 
  c / d = 1 / 320 := by
sorry

end ratio_c_d_equals_one_over_320_l3408_340896


namespace task_completion_time_l3408_340840

/-- The number of days needed for three people to complete the task -/
def three_people_days : ℕ := 3 * 7 + 3

/-- The number of people in the original scenario -/
def original_people : ℕ := 3

/-- The number of people in the new scenario -/
def new_people : ℕ := 4

/-- The time needed for four people to complete the task -/
def four_people_days : ℚ := 18

theorem task_completion_time :
  (three_people_days : ℚ) * original_people / new_people = four_people_days :=
sorry

end task_completion_time_l3408_340840


namespace abs_sum_inequality_iff_range_l3408_340824

theorem abs_sum_inequality_iff_range (x : ℝ) : 
  (abs (x + 1) + abs (x - 2) ≤ 5) ↔ (-2 ≤ x ∧ x ≤ 3) := by sorry

end abs_sum_inequality_iff_range_l3408_340824


namespace crazy_silly_school_movies_l3408_340893

/-- The 'crazy silly school' series problem -/
theorem crazy_silly_school_movies :
  ∀ (total_books watched_movies remaining_movies : ℕ),
    total_books = 21 →
    watched_movies = 4 →
    remaining_movies = 4 →
    watched_movies + remaining_movies = 8 :=
by sorry

end crazy_silly_school_movies_l3408_340893


namespace cube_root_of_point_on_line_l3408_340894

/-- For any point (a, b) on the graph of y = x - 1, the cube root of b - a is -1 -/
theorem cube_root_of_point_on_line (a b : ℝ) (h : b = a - 1) : 
  (b - a : ℝ) ^ (1/3 : ℝ) = -1 := by
sorry

end cube_root_of_point_on_line_l3408_340894


namespace function_decomposition_l3408_340892

-- Define the domain
def Domain : Set ℝ := {x : ℝ | x ≠ 1 ∧ x ≠ -1}

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x ∈ Domain, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x ∈ Domain, g (-x) = g x

-- State the theorem
theorem function_decomposition
  (f g : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_even : IsEven g)
  (h_sum : ∀ x ∈ Domain, f x + g x = 1 / (x - 1)) :
  (∀ x ∈ Domain, f x = x / (x^2 - 1)) ∧
  (∀ x ∈ Domain, g x = 1 / (x^2 - 1)) :=
by sorry

end function_decomposition_l3408_340892


namespace length_PR_l3408_340841

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 49}

structure PointsOnCircle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  h_P_on_circle : P ∈ Circle
  h_Q_on_circle : Q ∈ Circle
  h_PQ_distance : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 64
  h_R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the theorem
theorem length_PR (points : PointsOnCircle) : 
  ((points.P.1 - points.R.1)^2 + (points.P.2 - points.R.2)^2)^(1/2) = 4 * (2^(1/2)) := by
  sorry

end length_PR_l3408_340841


namespace cube_divisibility_l3408_340803

theorem cube_divisibility (n : ℕ) (h : ∀ k : ℕ, k > 0 → k < 42 → ¬(n ∣ k^3)) : n = 74088 := by
  sorry

end cube_divisibility_l3408_340803


namespace percentage_equation_solution_l3408_340842

theorem percentage_equation_solution :
  ∃ x : ℝ, (65 / 100) * x = (20 / 100) * 682.50 ∧ x = 210 := by
  sorry

end percentage_equation_solution_l3408_340842


namespace triangle_angle_measure_l3408_340877

/-- Given a triangle ABC where C = π/3, b = √2, and c = √3, prove that angle A = 5π/12 -/
theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  C = π/3 → b = Real.sqrt 2 → c = Real.sqrt 3 → 
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  A + B + C = π →
  A = 5*π/12 := by
  sorry

end triangle_angle_measure_l3408_340877


namespace right_triangle_sum_of_legs_l3408_340826

theorem right_triangle_sum_of_legs (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 50 →           -- Length of hypotenuse
  (a * b) / 2 = 600 →  -- Area of the triangle
  a + b = 70 := by
sorry

end right_triangle_sum_of_legs_l3408_340826


namespace parabolic_arch_height_l3408_340865

/-- Represents a parabolic arch --/
structure ParabolicArch where
  width : ℝ
  area : ℝ

/-- Calculates the height of a parabolic arch given its width and area --/
def archHeight (arch : ParabolicArch) : ℝ :=
  sorry

/-- Theorem stating that a parabolic arch with width 8 and area 160 has height 30 --/
theorem parabolic_arch_height :
  let arch : ParabolicArch := { width := 8, area := 160 }
  archHeight arch = 30 := by sorry

end parabolic_arch_height_l3408_340865


namespace simplify_expression_l3408_340823

theorem simplify_expression : (81 ^ (1/4) - Real.sqrt 12.75) ^ 2 = (87 - 12 * Real.sqrt 51) / 4 := by
  sorry

end simplify_expression_l3408_340823


namespace workshop_salary_calculation_l3408_340888

/-- Calculates the average salary of non-technician workers in a workshop --/
theorem workshop_salary_calculation 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (avg_salary_all : ℚ) 
  (avg_salary_technicians : ℚ) 
  (h1 : total_workers = 22)
  (h2 : technicians = 7)
  (h3 : avg_salary_all = 850)
  (h4 : avg_salary_technicians = 1000) :
  let non_technicians := total_workers - technicians
  let total_salary := avg_salary_all * total_workers
  let technicians_salary := avg_salary_technicians * technicians
  let non_technicians_salary := total_salary - technicians_salary
  non_technicians_salary / non_technicians = 780 := by
  sorry


end workshop_salary_calculation_l3408_340888


namespace paula_twice_karl_age_l3408_340846

/-- Represents the ages of Paula and Karl -/
structure Ages where
  paula : ℕ
  karl : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.paula + ages.karl = 50 ∧
  ages.paula - 7 = 3 * (ages.karl - 7)

/-- The theorem to prove -/
theorem paula_twice_karl_age (ages : Ages) :
  problem_conditions ages →
  ∃ x : ℕ, x = 2 ∧ ages.paula + x = 2 * (ages.karl + x) :=
sorry

end paula_twice_karl_age_l3408_340846


namespace ellipse_properties_max_radius_l3408_340807

/-- The ellipse C with foci F₁(-c, 0) and F₂(c, 0), and upper vertex M satisfying F₁M ⋅ F₂M = 0 -/
structure Ellipse (a b c : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (foci_condition : c^2 = a^2 - b^2)
  (vertex_condition : -c^2 + b^2 = 0)

/-- The point N(0, 2) is the center of a circle intersecting the ellipse C -/
def N : ℝ × ℝ := (0, 2)

/-- The theorem stating properties of the ellipse C -/
theorem ellipse_properties (a b c : ℝ) (C : Ellipse a b c) :
  -- The eccentricity of C is √2/2
  (c / a = Real.sqrt 2 / 2) ∧
  -- The equation of C is x²/18 + y²/9 = 1
  (∀ x y : ℝ, x^2 / 18 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  -- The range of k for symmetric points A and B on C w.r.t. y = kx - 1
  (∀ k : ℝ, (k < -1/2 ∨ k > 1/2) ↔
    ∃ A B : ℝ × ℝ,
      (A.1^2 / 18 + A.2^2 / 9 = 1) ∧
      (B.1^2 / 18 + B.2^2 / 9 = 1) ∧
      (A.2 = k * A.1 - 1) ∧
      (B.2 = k * B.1 - 1) ∧
      (A ≠ B)) :=
sorry

/-- The maximum radius of the circle centered at N intersecting C is √26 -/
theorem max_radius (a b c : ℝ) (C : Ellipse a b c) :
  ∀ P : ℝ × ℝ, P.1^2 / a^2 + P.2^2 / b^2 = 1 →
    (P.1 - N.1)^2 + (P.2 - N.2)^2 ≤ 26 :=
sorry

end ellipse_properties_max_radius_l3408_340807


namespace pony_discount_rate_l3408_340828

/-- Represents the discount rates for Fox and Pony jeans -/
structure DiscountRates where
  fox : ℝ
  pony : ℝ

/-- The problem setup -/
def jeans_problem (d : DiscountRates) : Prop :=
  -- Regular prices
  let fox_price : ℝ := 15
  let pony_price : ℝ := 18
  -- Total savings condition
  3 * fox_price * (d.fox / 100) + 2 * pony_price * (d.pony / 100) = 9 ∧
  -- Sum of discount rates condition
  d.fox + d.pony = 25

/-- The theorem to prove -/
theorem pony_discount_rate : 
  ∃ (d : DiscountRates), jeans_problem d ∧ d.pony = 25 := by
  sorry

end pony_discount_rate_l3408_340828


namespace pipe_length_theorem_l3408_340818

theorem pipe_length_theorem (shorter_piece longer_piece total_length : ℝ) :
  longer_piece = 2 * shorter_piece →
  longer_piece = 118 →
  total_length = shorter_piece + longer_piece →
  total_length = 177 := by
  sorry

end pipe_length_theorem_l3408_340818


namespace complex_simplification_and_multiplication_l3408_340883

theorem complex_simplification_and_multiplication :
  ((-5 + 3 * Complex.I) - (2 - 7 * Complex.I)) * (2 * Complex.I) = -20 - 14 * Complex.I :=
by sorry

end complex_simplification_and_multiplication_l3408_340883


namespace roots_sum_squares_l3408_340815

theorem roots_sum_squares (p q r : ℝ) : 
  (p^3 - 24*p^2 + 50*p - 35 = 0) →
  (q^3 - 24*q^2 + 50*q - 35 = 0) →
  (r^3 - 24*r^2 + 50*r - 35 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 1052 := by
sorry

end roots_sum_squares_l3408_340815


namespace arccos_one_half_l3408_340835

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_l3408_340835


namespace min_numbers_for_five_ones_digit_count_for_five_ones_l3408_340878

/-- Represents the sequence of digits when writing consecutive natural numbers -/
def digit_sequence (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list contains five consecutive ones -/
def has_five_consecutive_ones (l : List ℕ) : Prop :=
  sorry

/-- Counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ :=
  sorry

/-- Counts the total number of digits when writing the first n natural numbers -/
def total_digit_count (n : ℕ) : ℕ :=
  sorry

theorem min_numbers_for_five_ones :
  ∃ n : ℕ, n ≤ 112 ∧ has_five_consecutive_ones (digit_sequence n) ∧
  ∀ m : ℕ, m < n → ¬has_five_consecutive_ones (digit_sequence m) :=
sorry

theorem digit_count_for_five_ones :
  total_digit_count 112 = 228 :=
sorry

end min_numbers_for_five_ones_digit_count_for_five_ones_l3408_340878


namespace base6_greater_than_base8_l3408_340816

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem base6_greater_than_base8 : base6ToBase10 403 > base8ToBase10 217 := by
  sorry

end base6_greater_than_base8_l3408_340816


namespace geometric_sequence_third_term_l3408_340885

/-- A geometric sequence with a_1 = 1 and a_5 = 4 has a_3 = 2 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence condition
  a 1 = 1 →
  a 5 = 4 →
  a 3 = 2 := by
sorry

end geometric_sequence_third_term_l3408_340885


namespace age_problem_contradiction_l3408_340856

/-- Demonstrates the contradiction in the given age problem -/
theorem age_problem_contradiction (A B C D : ℕ) : 
  (A + B = B + C + 11) →  -- Condition 1
  (A + B + D = B + C + D + 8) →  -- Condition 2
  (A + C = 2 * D) →  -- Condition 3
  False := by sorry


end age_problem_contradiction_l3408_340856


namespace parallel_planes_intersection_theorem_l3408_340860

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes and lines
variable (intersects : Plane → Plane → Line → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_intersection_theorem 
  (α β γ : Plane) (m n : Line) :
  parallel_planes α β →
  intersects α γ m →
  intersects β γ n →
  parallel_lines m n :=
sorry

end parallel_planes_intersection_theorem_l3408_340860


namespace purple_balls_count_l3408_340804

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  white = 20 →
  green = 30 →
  yellow = 10 →
  red = 37 →
  prob_not_red_purple = 60 / 100 →
  (white + green + yellow : ℚ) / total = prob_not_red_purple →
  total - (white + green + yellow + red) = 3 := by
  sorry

end purple_balls_count_l3408_340804


namespace don_buys_from_shop_B_l3408_340884

/-- The number of bottles Don buys from Shop A -/
def bottlesFromA : ℕ := 150

/-- The number of bottles Don buys from Shop C -/
def bottlesFromC : ℕ := 220

/-- The total number of bottles Don buys -/
def totalBottles : ℕ := 550

/-- The number of bottles Don buys from Shop B -/
def bottlesFromB : ℕ := totalBottles - (bottlesFromA + bottlesFromC)

theorem don_buys_from_shop_B : bottlesFromB = 180 := by
  sorry

end don_buys_from_shop_B_l3408_340884


namespace bernoulli_expected_value_l3408_340851

/-- A random variable with a Bernoulli distribution -/
structure BernoulliRV (p : ℝ) :=
  (prob : 0 < p ∧ p < 1)

/-- The probability mass function for a Bernoulli random variable -/
def pmf (p : ℝ) (X : BernoulliRV p) (k : ℕ) : ℝ :=
  if k = 0 then (1 - p) else if k = 1 then p else 0

/-- The expected value of a Bernoulli random variable -/
def expectedValue (p : ℝ) (X : BernoulliRV p) : ℝ :=
  0 * pmf p X 0 + 1 * pmf p X 1

/-- Theorem: The expected value of a Bernoulli random variable is p -/
theorem bernoulli_expected_value (p : ℝ) (X : BernoulliRV p) :
  expectedValue p X = p := by
  sorry

end bernoulli_expected_value_l3408_340851


namespace ab_nonpositive_l3408_340890

theorem ab_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 := by
  sorry

end ab_nonpositive_l3408_340890


namespace liquid_level_rate_of_change_l3408_340853

/-- The rate of change of liquid level height in a cylindrical container -/
theorem liquid_level_rate_of_change 
  (d : ℝ) -- diameter of the base
  (drain_rate : ℝ) -- rate at which liquid is drained
  (h : ℝ → ℝ) -- height of liquid as a function of time
  (t : ℝ) -- time variable
  (hd : d = 2) -- given diameter
  (hdrain : drain_rate = 0.01) -- given drain rate
  : deriv h t = -drain_rate / (π * (d/2)^2) := by
  sorry

#check liquid_level_rate_of_change

end liquid_level_rate_of_change_l3408_340853


namespace edward_games_boxes_l3408_340827

def number_of_boxes (initial_games : ℕ) (sold_games : ℕ) (games_per_box : ℕ) : ℕ :=
  (initial_games - sold_games) / games_per_box

theorem edward_games_boxes :
  number_of_boxes 35 19 8 = 2 := by
  sorry

end edward_games_boxes_l3408_340827


namespace tourist_growth_and_max_l3408_340836

def tourists_feb : ℕ := 16000
def tourists_apr : ℕ := 25000
def tourists_may_21 : ℕ := 21250

def monthly_growth_rate : ℝ := 0.25

def max_daily_tourists_last_10_days : ℝ := 100000

theorem tourist_growth_and_max (growth_rate : ℝ) (max_daily : ℝ) :
  growth_rate = monthly_growth_rate ∧
  max_daily = max_daily_tourists_last_10_days ∧
  tourists_feb * (1 + growth_rate)^2 = tourists_apr ∧
  tourists_may_21 + 10 * max_daily ≤ tourists_apr * (1 + growth_rate) :=
by sorry

end tourist_growth_and_max_l3408_340836


namespace nick_hid_ten_chocolates_l3408_340845

/-- The number of chocolates Nick hid -/
def nick_chocolates : ℕ := sorry

/-- The number of chocolates Alix hid initially -/
def alix_initial_chocolates : ℕ := 3 * nick_chocolates

/-- The number of chocolates Alix has after mom took 5 -/
def alix_current_chocolates : ℕ := alix_initial_chocolates - 5

theorem nick_hid_ten_chocolates : 
  alix_current_chocolates = nick_chocolates + 15 → nick_chocolates = 10 := by
  sorry

end nick_hid_ten_chocolates_l3408_340845


namespace equation_solutions_l3408_340849

theorem equation_solutions : 
  ∀ x : ℝ, (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
             1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) :=
by sorry

end equation_solutions_l3408_340849


namespace number_of_common_tangents_l3408_340857

def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0

def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 6 = 0

def common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

theorem number_of_common_tangents :
  common_tangents circle_C1 circle_C2 = 3 :=
sorry

end number_of_common_tangents_l3408_340857


namespace min_value_M_min_value_expression_min_value_equality_condition_l3408_340864

open Real

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part I
theorem min_value_M : ∃ (M : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ M) ∧ ∀ (m : ℝ), (∃ (x : ℝ), f x ≤ m) → M ≤ m :=
sorry

-- Part II
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  1 / (2 * a) + 1 / (a + b) ≥ 2 :=
sorry

theorem min_value_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + b = 2) :
  1 / (2 * a) + 1 / (a + b) = 2 ↔ a = 1/2 ∧ b = 1/2 :=
sorry

end min_value_M_min_value_expression_min_value_equality_condition_l3408_340864


namespace first_discount_percentage_l3408_340898

theorem first_discount_percentage (original_price final_price : ℝ) 
  (second_discount : ℝ) (h1 : original_price = 480) 
  (h2 : final_price = 306) (h3 : second_discount = 25) : 
  ∃ (first_discount : ℝ), 
    first_discount = 15 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end first_discount_percentage_l3408_340898


namespace golf_strokes_over_par_l3408_340882

/-- Calculates the number of strokes over par in a golf game. -/
def strokes_over_par (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  let total_holes := rounds * 18
  let total_strokes := avg_strokes_per_hole * total_holes
  let total_par := par_per_hole * total_holes
  total_strokes - total_par

/-- Proves that given 9 rounds of golf, an average of 4 strokes per hole, 
    and a par value of 3 per hole, the number of strokes over par is 162. -/
theorem golf_strokes_over_par :
  strokes_over_par 9 4 3 = 162 := by
  sorry

end golf_strokes_over_par_l3408_340882


namespace remainder_sum_mod_21_l3408_340822

theorem remainder_sum_mod_21 (c d : ℤ) 
  (hc : c % 60 = 47) 
  (hd : d % 42 = 17) : 
  (c + d) % 21 = 1 := by
sorry

end remainder_sum_mod_21_l3408_340822


namespace largest_value_l3408_340875

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4) :
  w ≥ x ∧ w ≥ y ∧ w ≥ z :=
by sorry

end largest_value_l3408_340875


namespace division_problem_l3408_340880

theorem division_problem :
  ∃ x : ℝ, x > 0 ∧ 
    2 * x + (100 - 2 * x) = 100 ∧
    (300 - 6 * x) + x = 100 ∧
    x = 40 := by
  sorry

end division_problem_l3408_340880


namespace tiger_enclosure_optimizations_l3408_340806

/-- Represents a rectangular tiger enclosure -/
structure TigerEnclosure where
  length : ℝ
  width : ℝ

/-- Calculates the area of a tiger enclosure -/
def area (e : TigerEnclosure) : ℝ := e.length * e.width

/-- Calculates the wire mesh length needed for a tiger enclosure -/
def wireMeshLength (e : TigerEnclosure) : ℝ := e.length + 2 * e.width

/-- The total available wire mesh length -/
def totalWireMesh : ℝ := 36

/-- The fixed area for part 2 of the problem -/
def fixedArea : ℝ := 32

theorem tiger_enclosure_optimizations :
  (∃ (e : TigerEnclosure),
    wireMeshLength e = totalWireMesh ∧
    area e = 162 ∧
    e.length = 18 ∧
    e.width = 9 ∧
    (∀ (e' : TigerEnclosure), wireMeshLength e' ≤ totalWireMesh → area e' ≤ area e)) ∧
  (∃ (e : TigerEnclosure),
    area e = fixedArea ∧
    wireMeshLength e = 16 ∧
    e.length = 8 ∧
    e.width = 4 ∧
    (∀ (e' : TigerEnclosure), area e' = fixedArea → wireMeshLength e' ≥ wireMeshLength e)) :=
by sorry

end tiger_enclosure_optimizations_l3408_340806


namespace survey_C_most_suitable_for_census_l3408_340817

-- Define a structure for a survey
structure Survey where
  description : String
  population_size : ℕ
  resource_requirement : ℕ

-- Define the suitability for census method
def suitable_for_census (s : Survey) : Prop :=
  s.population_size ≤ 100 ∧ s.resource_requirement ≤ 50

-- Define the four survey options
def survey_A : Survey :=
  { description := "Quality and safety of local grain processing",
    population_size := 1000,
    resource_requirement := 200 }

def survey_B : Survey :=
  { description := "Viewership ratings of the 2023 CCTV Spring Festival Gala",
    population_size := 1000000,
    resource_requirement := 500000 }

def survey_C : Survey :=
  { description := "Weekly duration of physical exercise for a ninth-grade class",
    population_size := 50,
    resource_requirement := 30 }

def survey_D : Survey :=
  { description := "Household chores participation of junior high school students in the entire city",
    population_size := 100000,
    resource_requirement := 10000 }

-- Theorem stating that survey C is the most suitable for census method
theorem survey_C_most_suitable_for_census :
  suitable_for_census survey_C ∧
  (¬ suitable_for_census survey_A ∧
   ¬ suitable_for_census survey_B ∧
   ¬ suitable_for_census survey_D) :=
by sorry


end survey_C_most_suitable_for_census_l3408_340817
