import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l920_92067

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l920_92067


namespace NUMINAMATH_CALUDE_gain_percentage_is_twenty_percent_l920_92039

def selling_price : ℝ := 180
def gain : ℝ := 30

theorem gain_percentage_is_twenty_percent : 
  (gain / (selling_price - gain)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_is_twenty_percent_l920_92039


namespace NUMINAMATH_CALUDE_largest_square_from_rectangle_l920_92017

/-- Given a rectangular paper of length 54 cm and width 20 cm, 
    the largest side length of three equal squares that can be cut from this paper is 18 cm. -/
theorem largest_square_from_rectangle : ∀ (side_length : ℝ), 
  side_length > 0 ∧ 
  3 * side_length ≤ 54 ∧ 
  side_length ≤ 20 →
  side_length ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_from_rectangle_l920_92017


namespace NUMINAMATH_CALUDE_parking_lot_capacity_l920_92000

theorem parking_lot_capacity (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425)
  (h2 : num_levels = 5)
  (h3 : parked_cars = 23) :
  (total_capacity / num_levels) - parked_cars = 62 := by
  sorry

#check parking_lot_capacity

end NUMINAMATH_CALUDE_parking_lot_capacity_l920_92000


namespace NUMINAMATH_CALUDE_x_value_and_n_bound_l920_92029

theorem x_value_and_n_bound (x n : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + n < 4) : 
  x = 1 ∧ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_and_n_bound_l920_92029


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l920_92071

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem tangent_line_and_inequality :
  (∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ exp 2 * x - 4 * y = 0) ∧
  (∀ x, x > 0 → f x > 2 * (x - log x)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l920_92071


namespace NUMINAMATH_CALUDE_min_amount_for_house_l920_92013

/-- Calculates the minimum amount needed to buy a house given the original price,
    full payment discount percentage, and deed tax percentage. -/
def min_house_purchase_amount (original_price : ℕ) (discount_percent : ℚ) (deed_tax_percent : ℚ) : ℕ :=
  let discounted_price := (original_price : ℚ) * discount_percent
  let deed_tax := discounted_price * deed_tax_percent
  (discounted_price + deed_tax).ceil.toNat

/-- Proves that the minimum amount needed to buy the house is 311,808 yuan. -/
theorem min_amount_for_house :
  min_house_purchase_amount 320000 (96 / 100) (3 / 200) = 311808 := by
  sorry

#eval min_house_purchase_amount 320000 (96 / 100) (3 / 200)

end NUMINAMATH_CALUDE_min_amount_for_house_l920_92013


namespace NUMINAMATH_CALUDE_square_feet_per_acre_l920_92075

/-- Represents the area of a rectangle in square feet -/
def rectangle_area (length width : ℝ) : ℝ := length * width

/-- Represents the total number of acres rented -/
def total_acres : ℝ := 10

/-- Represents the monthly rent for the entire plot -/
def total_rent : ℝ := 300

/-- Represents the length of the rectangular plot in feet -/
def plot_length : ℝ := 360

/-- Represents the width of the rectangular plot in feet -/
def plot_width : ℝ := 1210

theorem square_feet_per_acre :
  (rectangle_area plot_length plot_width) / total_acres = 43560 := by
  sorry

#check square_feet_per_acre

end NUMINAMATH_CALUDE_square_feet_per_acre_l920_92075


namespace NUMINAMATH_CALUDE_total_bowling_balls_l920_92051

theorem total_bowling_balls (red : ℕ) (green : ℕ) (blue : ℕ) : 
  red = 30 →
  green = red + 6 →
  blue = 2 * green →
  red + green + blue = 138 := by
sorry

end NUMINAMATH_CALUDE_total_bowling_balls_l920_92051


namespace NUMINAMATH_CALUDE_tuesday_books_brought_back_l920_92019

/-- Calculates the number of books brought back on Tuesday given the initial number of books,
    the number of books taken out on Monday, and the final number of books on Tuesday. -/
def books_brought_back (initial : ℕ) (taken_out : ℕ) (final : ℕ) : ℕ :=
  final - (initial - taken_out)

/-- Theorem stating that 22 books were brought back on Tuesday given the specified conditions. -/
theorem tuesday_books_brought_back :
  books_brought_back 336 124 234 = 22 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_books_brought_back_l920_92019


namespace NUMINAMATH_CALUDE_pascal_triangle_odd_rows_l920_92022

/-- Represents a row in Pascal's triangle -/
def PascalRow := List Nat

/-- Generates the nth row of Pascal's triangle -/
def generatePascalRow (n : Nat) : PascalRow := sorry

/-- Checks if a row has all odd numbers except for the ends -/
def isAllOddExceptEnds (row : PascalRow) : Bool := sorry

/-- Counts the number of rows up to n that have all odd numbers except for the ends -/
def countAllOddExceptEndsRows (n : Nat) : Nat := sorry

/-- The main theorem to be proved -/
theorem pascal_triangle_odd_rows :
  countAllOddExceptEndsRows 30 = 3 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_odd_rows_l920_92022


namespace NUMINAMATH_CALUDE_bus_distance_problem_l920_92095

/-- Proves that given a total distance of 250 km, covered partly at 40 kmph and partly at 60 kmph,
    with a total travel time of 6 hours, the distance covered at 40 kmph is 220 km. -/
theorem bus_distance_problem (x : ℝ) 
    (h1 : x ≥ 0) 
    (h2 : x ≤ 250) 
    (h3 : x / 40 + (250 - x) / 60 = 6) : x = 220 := by
  sorry

#check bus_distance_problem

end NUMINAMATH_CALUDE_bus_distance_problem_l920_92095


namespace NUMINAMATH_CALUDE_smallest_a_value_l920_92053

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  ∀ a' : ℝ, (0 ≤ a' ∧ (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x))) → a ≤ a' → a = 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l920_92053


namespace NUMINAMATH_CALUDE_emilys_marbles_l920_92058

theorem emilys_marbles (jake_marbles : ℕ) (emily_scale : ℕ) : 
  jake_marbles = 216 → 
  emily_scale = 3 → 
  (emily_scale ^ 3) * jake_marbles = 5832 :=
by sorry

end NUMINAMATH_CALUDE_emilys_marbles_l920_92058


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l920_92050

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the angle between asymptotes
def angle_between_asymptotes (h : (x y : ℝ) → Prop) : ℝ := sorry

-- Theorem statement
theorem hyperbola_asymptote_angle :
  angle_between_asymptotes hyperbola = 60 * π / 180 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l920_92050


namespace NUMINAMATH_CALUDE_mixture_weight_l920_92063

/-- Given a mixture of substances a and b in the ratio 9:11, where 26.1 kg of a is used,
    prove that the total weight of the mixture is 58 kg. -/
theorem mixture_weight (a b : ℝ) (h1 : a / b = 9 / 11) (h2 : a = 26.1) :
  a + b = 58 := by sorry

end NUMINAMATH_CALUDE_mixture_weight_l920_92063


namespace NUMINAMATH_CALUDE_symmetry_condition_l920_92028

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

theorem symmetry_condition (m : ℝ) :
  (∀ x, f m (2 - x) = f m x) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l920_92028


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l920_92025

theorem algebraic_expression_equality (x y : ℝ) :
  2 * x - y + 1 = 3 → 4 * x - 2 * y + 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l920_92025


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l920_92046

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^2 + 33 * y - 90 - y * (y + 18)
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = -18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l920_92046


namespace NUMINAMATH_CALUDE_crups_are_arogs_and_brafs_l920_92068

-- Define the types for our sets
variable (U : Type) -- Universe set
variable (Arog Braf Crup Dramp : Set U)

-- Define the given conditions
variable (h1 : Arog ⊆ Braf)
variable (h2 : Crup ⊆ Braf)
variable (h3 : Arog ⊆ Dramp)
variable (h4 : Crup ⊆ Dramp)

-- Theorem to prove
theorem crups_are_arogs_and_brafs : Crup ⊆ Arog ∩ Braf :=
sorry

end NUMINAMATH_CALUDE_crups_are_arogs_and_brafs_l920_92068


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l920_92037

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := 180 * (n - 2)
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The exterior angle of a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_is_45 : 
  exterior_angle_regular_octagon = 45 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l920_92037


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l920_92061

/-- Proves that a replacing solution must be 40% sugar by weight given the conditions of the problem. -/
theorem sugar_solution_percentage (original_percentage : ℝ) (replaced_fraction : ℝ) (final_percentage : ℝ) :
  original_percentage = 8 →
  replaced_fraction = 1 / 4 →
  final_percentage = 16 →
  (1 - replaced_fraction) * original_percentage + replaced_fraction * (100 : ℝ) * final_percentage / 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l920_92061


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l920_92091

def is_reducible (n d : ℤ) : Prop := ∃ k : ℤ, k > 1 ∧ k ∣ n ∧ k ∣ d

theorem smallest_reducible_fraction :
  ∀ m : ℕ, m > 0 →
    (m < 30 → ¬(is_reducible (m - 17) (7 * m + 11))) ∧
    (is_reducible (30 - 17) (7 * 30 + 11)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l920_92091


namespace NUMINAMATH_CALUDE_prop_logic_evaluation_l920_92042

theorem prop_logic_evaluation (p q : Prop) (hp : p ↔ (2 < 3)) (hq : q ↔ (2 > 3)) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_prop_logic_evaluation_l920_92042


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l920_92043

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 169 ∧ x2^2 + y^2 = 169 ∧ x1 + x2 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l920_92043


namespace NUMINAMATH_CALUDE_women_work_nine_hours_l920_92056

/-- Represents the work scenario with men and women -/
structure WorkScenario where
  men_count : ℕ
  men_days : ℕ
  men_hours_per_day : ℕ
  women_count : ℕ
  women_days : ℕ
  women_efficiency : Rat

/-- Calculates the number of hours women work per day -/
def women_hours_per_day (ws : WorkScenario) : Rat :=
  (ws.men_count * ws.men_days * ws.men_hours_per_day : Rat) /
  (ws.women_count * ws.women_days * ws.women_efficiency)

/-- The given work scenario -/
def given_scenario : WorkScenario :=
  { men_count := 15
  , men_days := 21
  , men_hours_per_day := 8
  , women_count := 21
  , women_days := 20
  , women_efficiency := 2/3 }

theorem women_work_nine_hours : women_hours_per_day given_scenario = 9 := by
  sorry

end NUMINAMATH_CALUDE_women_work_nine_hours_l920_92056


namespace NUMINAMATH_CALUDE_women_in_second_group_l920_92035

/-- Represents the work rate of a man -/
def man_rate : ℝ := sorry

/-- Represents the work rate of a woman -/
def woman_rate : ℝ := sorry

/-- The number of women in the second group -/
def x : ℝ := sorry

/-- First condition: 3 men and 8 women complete a task in the same time as 6 men and x women -/
axiom condition1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + x * woman_rate

/-- Second condition: 2 men and 3 women complete half the work in the same time as the first group -/
axiom condition2 : 2 * man_rate + 3 * woman_rate = 0.5 * (3 * man_rate + 8 * woman_rate)

/-- The theorem to be proved -/
theorem women_in_second_group : x = 2 := by sorry

end NUMINAMATH_CALUDE_women_in_second_group_l920_92035


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l920_92066

theorem quadratic_integer_roots (m : ℝ) :
  (∃ x : ℤ, (m + 1) * x^2 + 2 * x - 5 * m - 13 = 0) ↔
  (m = -1 ∨ m = -11/10 ∨ m = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l920_92066


namespace NUMINAMATH_CALUDE_rectangle_area_l920_92030

/-- A rectangle with a diagonal of 17 cm and a perimeter of 46 cm has an area of 120 cm². -/
theorem rectangle_area (l w : ℝ) : 
  l > 0 → w > 0 → l^2 + w^2 = 17^2 → 2*l + 2*w = 46 → l * w = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l920_92030


namespace NUMINAMATH_CALUDE_train_journey_distance_l920_92024

/-- Represents the train's journey with an accident -/
structure TrainJourney where
  initialSpeed : ℝ
  totalDistance : ℝ
  accidentDelay : ℝ
  speedReductionFactor : ℝ
  totalDelay : ℝ
  alternateLaterAccidentDistance : ℝ
  alternateTotalDelay : ℝ

/-- The train journey satisfies the given conditions -/
def satisfiesConditions (j : TrainJourney) : Prop :=
  j.accidentDelay = 0.5 ∧
  j.speedReductionFactor = 3/4 ∧
  j.totalDelay = 3.5 ∧
  j.alternateLaterAccidentDistance = 90 ∧
  j.alternateTotalDelay = 3

/-- The theorem stating that the journey distance is 600 miles -/
theorem train_journey_distance (j : TrainJourney) 
  (h : satisfiesConditions j) : j.totalDistance = 600 :=
sorry

#check train_journey_distance

end NUMINAMATH_CALUDE_train_journey_distance_l920_92024


namespace NUMINAMATH_CALUDE_prob_two_primes_equals_216_625_l920_92072

-- Define a 10-sided die
def tenSidedDie : Finset ℕ := Finset.range 10

-- Define the set of prime numbers on a 10-sided die
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define the probability of rolling a prime number on one die
def probPrime : ℚ := (primes.card : ℚ) / (tenSidedDie.card : ℚ)

-- Define the probability of not rolling a prime number on one die
def probNotPrime : ℚ := 1 - probPrime

-- Define the number of ways to choose 2 dice out of 4
def waysToChoose : ℕ := Nat.choose 4 2

-- Define the probability of exactly two dice showing a prime number
def probTwoPrimes : ℚ := (waysToChoose : ℚ) * probPrime^2 * probNotPrime^2

-- Theorem statement
theorem prob_two_primes_equals_216_625 : probTwoPrimes = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_primes_equals_216_625_l920_92072


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l920_92007

/-- An arithmetic sequence with non-zero terms -/
def arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a b : ℕ+ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geo : geometric_sequence b)
  (h_nonzero : ∀ n : ℕ+, a n ≠ 0)
  (h_eq : 2 * (a 3) - (a 7)^2 + 2 * (a 11) = 0)
  (h_b7 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l920_92007


namespace NUMINAMATH_CALUDE_john_spent_110_l920_92078

/-- The amount of money John spent on wigs for his plays -/
def johnSpent (numPlays : ℕ) (numActs : ℕ) (wigsPerAct : ℕ) (wigCost : ℕ) (sellPrice : ℕ) : ℕ :=
  let totalWigs := numPlays * numActs * wigsPerAct
  let totalCost := totalWigs * wigCost
  let soldWigs := numActs * wigsPerAct
  let moneyBack := soldWigs * sellPrice
  totalCost - moneyBack

/-- Theorem stating that John spent $110 on wigs -/
theorem john_spent_110 :
  johnSpent 3 5 2 5 4 = 110 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_110_l920_92078


namespace NUMINAMATH_CALUDE_permutation_count_equals_fibonacci_l920_92036

/-- The number of permutations satisfying the given condition -/
def P (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else P (n - 1) + P (n - 2)

/-- The nth Fibonacci number -/
def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

/-- Theorem stating the equivalence between P(n) and the (n+1)th Fibonacci number -/
theorem permutation_count_equals_fibonacci (n : ℕ) :
  P n = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_permutation_count_equals_fibonacci_l920_92036


namespace NUMINAMATH_CALUDE_visibility_time_correct_l920_92045

/-- Represents a person walking along a straight path -/
structure Walker where
  speed : ℝ
  initial_x : ℝ
  y : ℝ

/-- Represents a circular building -/
structure Building where
  radius : ℝ

/-- Calculates the time when two walkers can see each other again after being blocked by a building -/
def time_to_see_again (jenny : Walker) (kenny : Walker) (building : Building) : ℝ :=
  sorry

theorem visibility_time_correct :
  let jenny : Walker := { speed := 2, initial_x := -75, y := 150 }
  let kenny : Walker := { speed := 4, initial_x := -75, y := -150 }
  let building : Building := { radius := 75 }
  time_to_see_again jenny kenny building = 48 := by sorry

end NUMINAMATH_CALUDE_visibility_time_correct_l920_92045


namespace NUMINAMATH_CALUDE_polygon_division_existence_l920_92023

/-- A polygon represented by a list of points in 2D space -/
def Polygon : Type := List (ℝ × ℝ)

/-- A line segment represented by its two endpoints -/
def LineSegment : Type := (ℝ × ℝ) × (ℝ × ℝ)

/-- Function to check if a line segment divides a polygon into two equal-area parts -/
def divides_equally (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a line segment bisects a side of a polygon -/
def bisects_side (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a line segment divides a side of a polygon in 1:2 ratio -/
def divides_side_in_ratio (p : Polygon) (l : LineSegment) : Prop := sorry

/-- Function to check if a polygon is convex -/
def is_convex (p : Polygon) : Prop := sorry

theorem polygon_division_existence :
  ∃ (p : Polygon) (l : LineSegment), 
    divides_equally p l ∧ 
    bisects_side p l ∧ 
    divides_side_in_ratio p l ∧
    is_convex p :=
sorry

end NUMINAMATH_CALUDE_polygon_division_existence_l920_92023


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l920_92041

theorem gcd_lcm_product (a b : ℕ) (ha : a = 180) (hb : b = 250) :
  (Nat.gcd a b) * (Nat.lcm a b) = 45000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l920_92041


namespace NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_one_l920_92049

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ : ℝ) (m₂ n₂ : ℝ) : Prop := m₁ * n₂ = m₂ * n₁

/-- The statement that a = 1 is necessary and sufficient for the lines to be parallel -/
theorem parallel_lines_iff_a_eq_one :
  ∀ a : ℝ, are_parallel a 1 3 (a + 2) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_one_l920_92049


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l920_92064

theorem smallest_x_for_equation : 
  ∃ (x : ℕ+), x = 4 ∧ 
  (∀ (y : ℕ+), (3 : ℚ) / 4 = (y : ℚ) / (200 + x)) ∧
  (∀ (x' : ℕ+), x' < x → 
    ¬∃ (y : ℕ+), (3 : ℚ) / 4 = (y : ℚ) / (200 + x')) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l920_92064


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l920_92082

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ),
    (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
      5 * x^2 / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 20 ∧ Q = -15 ∧ R = -10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l920_92082


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l920_92004

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem smallest_slope_tangent_line :
  ∃ (x₀ : ℝ), 
    (∀ (x : ℝ), f' x₀ ≤ f' x) ∧ 
    (∀ (x y : ℝ), y = f' x₀ * (x - x₀) + f x₀ ↔ y = -3 * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l920_92004


namespace NUMINAMATH_CALUDE_hexagon_coin_rotations_l920_92079

/-- Represents a configuration of coins on a table -/
structure CoinConfiguration where
  num_coins : Nat
  is_closed_chain : Bool

/-- Represents the motion of a rolling coin -/
structure RollingCoin where
  rotations : Nat

/-- Calculates the number of rotations a coin makes when rolling around a hexagon of coins -/
def calculate_rotations (config : CoinConfiguration) : RollingCoin :=
  sorry

/-- Theorem: A coin rolling around a hexagon of coins makes 4 complete rotations -/
theorem hexagon_coin_rotations :
  ∀ (config : CoinConfiguration),
    config.num_coins = 6 ∧ config.is_closed_chain →
    (calculate_rotations config).rotations = 4 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_coin_rotations_l920_92079


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l920_92074

-- Define the set of possible initial compositions
inductive InitialComposition
| NoWhite
| OneWhite
| TwoWhite

-- Define the probability of drawing a white ball given an initial composition
def probWhiteGivenComposition (ic : InitialComposition) : ℚ :=
  match ic with
  | InitialComposition.NoWhite => 1/3
  | InitialComposition.OneWhite => 2/3
  | InitialComposition.TwoWhite => 1

-- Define the theorem
theorem probability_of_white_ball :
  let initialCompositions := [InitialComposition.NoWhite, InitialComposition.OneWhite, InitialComposition.TwoWhite]
  let numCompositions := initialCompositions.length
  let probEachComposition := 1 / numCompositions
  let totalProb := (initialCompositions.map probWhiteGivenComposition).sum * probEachComposition
  totalProb = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l920_92074


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l920_92010

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (a 1 + a 2 + a 3 + a 4 = 3) →  -- First condition
  (a 5 + a 6 + a 7 + a 8 = 48) →  -- Second condition
  (a 1 / (1 - q) = -1/5) :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l920_92010


namespace NUMINAMATH_CALUDE_fifth_roots_of_unity_l920_92070

theorem fifth_roots_of_unity (p q r s t m : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m^5 = 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_roots_of_unity_l920_92070


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l920_92084

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 18*x₁ + 30 = 2 * Real.sqrt (x₁^2 + 18*x₁ + 45)) ∧ 
                (x₂^2 + 18*x₂ + 30 = 2 * Real.sqrt (x₂^2 + 18*x₂ + 45)) ∧ 
                (∀ y : ℝ, y^2 + 18*y + 30 = 2 * Real.sqrt (y^2 + 18*y + 45) → y = x₁ ∨ y = x₂)) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 18*x₁ + 30 = 2 * Real.sqrt (x₁^2 + 18*x₁ + 45)) ∧ 
                (x₂^2 + 18*x₂ + 30 = 2 * Real.sqrt (x₂^2 + 18*x₂ + 45)) ∧ 
                (x₁ + x₂ = -18)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l920_92084


namespace NUMINAMATH_CALUDE_friends_team_assignment_l920_92055

theorem friends_team_assignment (n : ℕ) (k : ℕ) :
  n = 8 → k = 4 → (k : ℕ) ^ n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l920_92055


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l920_92033

theorem pen_pencil_ratio : 
  ∀ (num_pencils num_pens : ℕ),
  num_pencils = 24 →
  num_pencils = num_pens + 4 →
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l920_92033


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_one_l920_92011

theorem at_least_one_fraction_less_than_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : y - x > 1) : 
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_one_l920_92011


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l920_92090

/-- Represents a parabola of the form y = ax^2 + bx + 1 -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a parabola passes through a given point -/
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + 1

/-- Represents the parabola after translation along x-axis -/
def translate (p : Parabola) (m : ℝ) (x : ℝ) : ℝ :=
  p.a * (x - m)^2 + p.b * (x - m) + 1

theorem parabola_translation_theorem (p : Parabola) (m : ℝ) :
  passes_through p 1 (-2) ∧ passes_through p (-2) 13 ∧ m > 0 →
  (∀ x, -1 ≤ x ∧ x ≤ 3 → translate p m x ≥ 6) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ translate p m x = 6) ↔
  m = 6 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l920_92090


namespace NUMINAMATH_CALUDE_dalmatians_with_right_ear_spot_l920_92098

theorem dalmatians_with_right_ear_spot (total : ℕ) (left_only : ℕ) (right_only : ℕ) (no_spots : ℕ) :
  total = 101 →
  left_only = 29 →
  right_only = 17 →
  no_spots = 22 →
  total - no_spots - left_only = 50 :=
by sorry

end NUMINAMATH_CALUDE_dalmatians_with_right_ear_spot_l920_92098


namespace NUMINAMATH_CALUDE_amount_distribution_l920_92047

theorem amount_distribution (amount : ℕ) : 
  (∀ (x y : ℕ), x = amount / 14 ∧ y = amount / 18 → x = y + 80) →
  amount = 5040 := by
sorry

end NUMINAMATH_CALUDE_amount_distribution_l920_92047


namespace NUMINAMATH_CALUDE_stone_game_ratio_l920_92069

/-- The stone game on a blackboard -/
def StoneGame (n : ℕ) : Prop :=
  n ≥ 3 →
  ∀ (s t : ℕ), s > 0 ∧ t > 0 →
  ∃ (q : ℚ), q ≥ 1 ∧ q < n - 1 ∧ (t : ℚ) / s = q

theorem stone_game_ratio (n : ℕ) (h : n ≥ 3) :
  StoneGame n :=
sorry

end NUMINAMATH_CALUDE_stone_game_ratio_l920_92069


namespace NUMINAMATH_CALUDE_intersection_points_l920_92092

-- Define the slopes and y-intercepts of the lines
def m₁ : ℚ := 3
def b₁ : ℚ := -4
def b₃ : ℚ := -3

-- Define Point A
def A : ℚ × ℚ := (3, 2)

-- Define the perpendicular slope
def m₂ : ℚ := -1 / m₁

-- Define the equations of the lines
def line1 (x : ℚ) : ℚ := m₁ * x + b₁
def line2 (x : ℚ) : ℚ := m₂ * (x - A.1) + A.2
def line3 (x : ℚ) : ℚ := m₁ * x + b₃

-- State the theorem
theorem intersection_points :
  ∃ (P Q : ℚ × ℚ),
    (P.1 = 21/10 ∧ P.2 = 23/10 ∧ line1 P.1 = line2 P.1) ∧
    (Q.1 = 9/5 ∧ Q.2 = 12/5 ∧ line2 Q.1 = line3 Q.1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l920_92092


namespace NUMINAMATH_CALUDE_cards_given_by_jeff_l920_92038

theorem cards_given_by_jeff (initial_cards final_cards : ℝ) 
  (h1 : initial_cards = 304.0)
  (h2 : final_cards = 580) :
  final_cards - initial_cards = 276 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_by_jeff_l920_92038


namespace NUMINAMATH_CALUDE_smallest_n_proof_l920_92040

def has_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d = 7 ∧ ∃ k m : ℕ, n = k * 10 + d + m * 100

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def smallest_n_with_properties : ℕ := 65536

theorem smallest_n_proof :
  (is_terminating_decimal smallest_n_with_properties) ∧
  (has_digit_seven smallest_n_with_properties) ∧
  (∀ m : ℕ, m < smallest_n_with_properties →
    ¬(is_terminating_decimal m ∧ has_digit_seven m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_proof_l920_92040


namespace NUMINAMATH_CALUDE_negative_square_cubed_l920_92096

theorem negative_square_cubed (m : ℝ) : (-m^2)^3 = -m^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l920_92096


namespace NUMINAMATH_CALUDE_lioness_hyena_age_ratio_l920_92054

/-- The ratio of a lioness's age to a hyena's age in a park -/
theorem lioness_hyena_age_ratio :
  ∀ (hyena_age : ℕ) (k : ℕ+),
  k * hyena_age = 12 →
  (6 + 5) + (hyena_age / 2 + 5) = 19 →
  (12 : ℚ) / hyena_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_lioness_hyena_age_ratio_l920_92054


namespace NUMINAMATH_CALUDE_square_sum_xy_l920_92052

theorem square_sum_xy (x y a c : ℝ) (h1 : x * y = a) (h2 : 1 / x^2 + 1 / y^2 = c) :
  (x + y)^2 = a * c^2 + 2 * a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l920_92052


namespace NUMINAMATH_CALUDE_f_second_derivative_at_zero_l920_92044

-- Define the function f
def f (x : ℝ) (f''_1 : ℝ) : ℝ := x^3 - 2 * x * f''_1

-- State the theorem
theorem f_second_derivative_at_zero (f''_1 : ℝ) : 
  (deriv (deriv (f · f''_1))) 0 = -2 :=
sorry

end NUMINAMATH_CALUDE_f_second_derivative_at_zero_l920_92044


namespace NUMINAMATH_CALUDE_opposite_sign_sum_l920_92059

theorem opposite_sign_sum (x y : ℝ) : (x + 3)^2 + |y - 2| = 0 → (x + y)^y = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_l920_92059


namespace NUMINAMATH_CALUDE_sum_of_number_and_predecessor_l920_92016

theorem sum_of_number_and_predecessor : ∃ n : ℤ, (6 * n - 2 = 100) ∧ (n + (n - 1) = 33) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_predecessor_l920_92016


namespace NUMINAMATH_CALUDE_expression_evaluation_l920_92031

theorem expression_evaluation : 3^(0^(2^8)) + ((3^0)^2)^8 = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l920_92031


namespace NUMINAMATH_CALUDE_distribute_basketballs_count_l920_92077

/-- The number of ways to distribute four labeled basketballs among three kids -/
def distribute_basketballs : ℕ :=
  30

/-- Each kid must get at least one basketball -/
axiom each_kid_gets_one : True

/-- Basketballs are labeled 1, 2, 3, and 4 -/
axiom basketballs_labeled : True

/-- Basketballs 1 and 2 cannot be given to the same kid -/
axiom one_and_two_separate : True

/-- The number of ways to distribute the basketballs satisfying all conditions is 30 -/
theorem distribute_basketballs_count :
  distribute_basketballs = 30 :=
by sorry

end NUMINAMATH_CALUDE_distribute_basketballs_count_l920_92077


namespace NUMINAMATH_CALUDE_triangle_external_angle_l920_92085

theorem triangle_external_angle (a b c x : ℝ) : 
  a = 50 → b = 40 → c = 90 → a + b + c = 180 → 
  x + 45 = 180 → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_triangle_external_angle_l920_92085


namespace NUMINAMATH_CALUDE_spring_center_max_height_l920_92027

/-- The maximum height reached by the center of a spring connecting two identical masses -/
theorem spring_center_max_height 
  (m : ℝ) -- mass of each object
  (g : ℝ) -- acceleration due to gravity
  (V₁ V₂ : ℝ) -- initial velocities of upper and lower masses
  (α β : ℝ) -- angles of initial velocities with respect to horizontal
  (h : ℝ) -- maximum height reached by the center of the spring
  (h_pos : 0 < h) -- height is positive
  (m_pos : 0 < m) -- mass is positive
  (g_pos : 0 < g) -- gravity is positive
  : h = (1 / (2 * g)) * ((V₁ * Real.sin β + V₂ * Real.sin α) / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_spring_center_max_height_l920_92027


namespace NUMINAMATH_CALUDE_replaced_person_weight_l920_92073

/-- Proves that the weight of the replaced person is 35 kg given the conditions -/
theorem replaced_person_weight (initial_count : Nat) (weight_increase : Real) (new_person_weight : Real) :
  initial_count = 8 ∧ 
  weight_increase = 2.5 ∧ 
  new_person_weight = 55 →
  (initial_count * weight_increase = new_person_weight - (initial_count * weight_increase - new_person_weight)) :=
by
  sorry

#check replaced_person_weight

end NUMINAMATH_CALUDE_replaced_person_weight_l920_92073


namespace NUMINAMATH_CALUDE_vidya_mother_age_l920_92094

theorem vidya_mother_age (vidya_age : ℕ) (mother_age : ℕ) : 
  vidya_age = 13 → 
  mother_age = 3 * vidya_age + 5 → 
  mother_age = 44 := by
sorry

end NUMINAMATH_CALUDE_vidya_mother_age_l920_92094


namespace NUMINAMATH_CALUDE_triangle_properties_l920_92015

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- The theorem to be proved -/
theorem triangle_properties (t : AcuteTriangle)
  (h1 : Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.a = 2) :
  t.C = π/3 ∧ t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l920_92015


namespace NUMINAMATH_CALUDE_a2_value_l920_92087

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b / a = c / b

theorem a2_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_a2_value_l920_92087


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l920_92020

theorem fraction_equation_solution (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 20 * b) / (b + 20 * a) = 3) : 
  a / b = 0.33 := by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l920_92020


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l920_92088

/-- Given two points A and B that are symmetric with respect to a line l,
    prove that the equation of line l is 3x + y + 4 = 0 --/
theorem symmetric_points_line_equation (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  A = (1, 3) →
  B = (-5, 1) →
  (∀ (P : ℝ × ℝ), P ∈ l ↔ dist P A = dist P B) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3 * x + y + 4 = 0) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_points_line_equation_l920_92088


namespace NUMINAMATH_CALUDE_corn_yield_ratio_l920_92057

/-- Represents the corn yield ratio problem --/
theorem corn_yield_ratio :
  let johnson_hectares : ℕ := 1
  let johnson_yield_per_2months : ℕ := 80
  let neighbor_hectares : ℕ := 2
  let total_months : ℕ := 6
  let total_yield : ℕ := 1200
  let neighbor_yield_ratio : ℚ := 
    (total_yield - johnson_yield_per_2months * (total_months / 2) * johnson_hectares) /
    (johnson_yield_per_2months * (total_months / 2) * neighbor_hectares)
  neighbor_yield_ratio = 2
  := by sorry

end NUMINAMATH_CALUDE_corn_yield_ratio_l920_92057


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_six_l920_92048

theorem no_solution_iff_k_eq_six (k : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 7 → (x - 1) / (x - 2) ≠ (x - k) / (x - 7)) ↔ k = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_six_l920_92048


namespace NUMINAMATH_CALUDE_divisibility_rule_37_l920_92014

-- Define a function to compute the sum of three-digit groups
def sumOfGroups (n : ℕ) : ℕ := sorry

-- State the theorem
theorem divisibility_rule_37 (n : ℕ) :
  37 ∣ n ↔ 37 ∣ sumOfGroups n := by sorry

end NUMINAMATH_CALUDE_divisibility_rule_37_l920_92014


namespace NUMINAMATH_CALUDE_power_seven_equals_product_l920_92080

theorem power_seven_equals_product (a : ℝ) : a^7 = a^3 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_equals_product_l920_92080


namespace NUMINAMATH_CALUDE_required_run_rate_calculation_l920_92065

/-- Represents a cricket game situation -/
structure CricketGame where
  totalOvers : ℕ
  firstInningOvers : ℕ
  firstInningRunRate : ℚ
  wicketsLost : ℕ
  targetScore : ℕ
  remainingRunsNeeded : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstInningOvers
  let runsScored := game.firstInningRunRate * game.firstInningOvers
  let actualRemainingRuns := game.targetScore - runsScored
  actualRemainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given game situation -/
theorem required_run_rate_calculation (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstInningOvers = 20)
  (h3 : game.firstInningRunRate = 4.2)
  (h4 : game.wicketsLost = 5)
  (h5 : game.targetScore = 250)
  (h6 : game.remainingRunsNeeded = 195) :
  requiredRunRate game = 5.53 := by
  sorry

#eval requiredRunRate {
  totalOvers := 50,
  firstInningOvers := 20,
  firstInningRunRate := 4.2,
  wicketsLost := 5,
  targetScore := 250,
  remainingRunsNeeded := 195
}

end NUMINAMATH_CALUDE_required_run_rate_calculation_l920_92065


namespace NUMINAMATH_CALUDE_circle_equation_l920_92093

/-- A circle with center on the line x + y = 0 and intersecting the x-axis at (-3, 0) and (1, 0) -/
structure Circle where
  center : ℝ × ℝ
  center_on_line : center.1 + center.2 = 0
  intersects_x_axis : ∃ (t : ℝ), t^2 = (center.1 + 3)^2 + center.2^2 ∧ t^2 = (center.1 - 1)^2 + center.2^2

/-- The equation of the circle is (x+1)² + (y-1)² = 5 -/
theorem circle_equation (c : Circle) : 
  ∀ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 5 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = ((c.center.1 + 3)^2 + c.center.2^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l920_92093


namespace NUMINAMATH_CALUDE_find_number_l920_92083

theorem find_number : ∃ x : ℝ, 13 * x - 272 = 105 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l920_92083


namespace NUMINAMATH_CALUDE_chad_age_l920_92034

theorem chad_age (diana fabian eduardo chad : ℕ) 
  (h1 : diana = fabian - 5)
  (h2 : fabian = eduardo + 2)
  (h3 : chad = eduardo + 3)
  (h4 : diana = 15) : 
  chad = 21 := by
  sorry

end NUMINAMATH_CALUDE_chad_age_l920_92034


namespace NUMINAMATH_CALUDE_bottles_from_625_l920_92005

/-- The number of new bottles that can be made from a given number of plastic bottles -/
def new_bottles (initial : ℕ) : ℕ :=
  if initial < 3 then 0
  else (initial / 5) + new_bottles (initial / 5)

/-- Theorem stating the number of new bottles that can be made from 625 plastic bottles -/
theorem bottles_from_625 : new_bottles 625 = 156 := by
  sorry

end NUMINAMATH_CALUDE_bottles_from_625_l920_92005


namespace NUMINAMATH_CALUDE_cups_sold_after_day_one_l920_92002

theorem cups_sold_after_day_one 
  (initial_sales : ℕ) 
  (total_days : ℕ) 
  (average_sales : ℚ) 
  (h1 : initial_sales = 86)
  (h2 : total_days = 12)
  (h3 : average_sales = 53) :
  ∃ (daily_sales : ℕ), 
    (initial_sales + (total_days - 1) * daily_sales) / total_days = average_sales ∧
    daily_sales = 50 := by
  sorry

end NUMINAMATH_CALUDE_cups_sold_after_day_one_l920_92002


namespace NUMINAMATH_CALUDE_compare_sqrt_l920_92086

theorem compare_sqrt : -2 * Real.sqrt 11 > -3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_l920_92086


namespace NUMINAMATH_CALUDE_stating_min_weighings_to_determine_faulty_coin_l920_92032

/-- Represents a pile of coins with one faulty coin. -/
structure CoinPile :=
  (total : ℕ)  -- Total number of coins
  (faulty : ℕ)  -- Index of the faulty coin (1-based)
  (is_lighter : Bool)  -- True if the faulty coin is lighter, False if heavier

/-- Represents a weighing on a balance scale. -/
inductive Weighing
  | Equal : Weighing  -- The scale is balanced
  | Left : Weighing   -- The left side is heavier
  | Right : Weighing  -- The right side is heavier

/-- Function to perform a weighing on a subset of coins. -/
def weigh (pile : CoinPile) (left : List ℕ) (right : List ℕ) : Weighing :=
  sorry  -- Implementation details omitted

/-- 
Theorem stating that the minimum number of weighings required to determine 
whether the faulty coin is lighter or heavier is 2.
-/
theorem min_weighings_to_determine_faulty_coin (pile : CoinPile) : 
  ∃ (strategy : List (List ℕ × List ℕ)), 
    (strategy.length = 2) ∧ 
    (∀ (outcome : List Weighing), 
      outcome.length = 2 → 
      (∃ (result : Bool), result = pile.is_lighter)) :=
sorry

end NUMINAMATH_CALUDE_stating_min_weighings_to_determine_faulty_coin_l920_92032


namespace NUMINAMATH_CALUDE_quadratic_inequality_l920_92008

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x + 5 > 0 ↔ x < 1 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l920_92008


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_perpendicular_intersection_l920_92018

-- Define the line l
def line_l (x : ℝ) : ℝ := -x + 3

-- Define the ellipse C
def ellipse_C (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define the line l'
def line_l' (b x : ℝ) : ℝ := -x + b

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_line_intersection :
  ∀ m n : ℝ, n > m → m > 0 →
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 1 ∧ line_l p.1 = p.2 ∧ ellipse_C m n p.1 p.2) →
  (∀ x y : ℝ, ellipse_C m n x y ↔ standard_ellipse x y) :=
sorry

theorem perpendicular_intersection :
  ∀ b : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    standard_ellipse A.1 A.2 ∧ standard_ellipse B.1 B.2 ∧
    line_l' b A.1 = A.2 ∧ line_l' b B.1 = B.2 ∧
    perpendicular A.1 A.2 B.1 B.2) →
  b = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_perpendicular_intersection_l920_92018


namespace NUMINAMATH_CALUDE_min_items_for_matching_pair_l920_92003

/-- Represents a tea set with a cup and a saucer -/
structure TeaSet :=
  (cup : Nat)
  (saucer : Nat)

/-- Represents a box containing either cups or saucers -/
inductive Box
| Cups : Box
| Saucers : Box

/-- The number of distinct tea sets -/
def num_sets : Nat := 6

/-- The total number of items in each box -/
def items_per_box : Nat := 6

/-- A function that selects a given number of items from a box -/
def select_items (b : Box) (n : Nat) : Finset Nat := sorry

/-- Predicate to check if a selection guarantees a matching pair -/
def guarantees_matching_pair (cups : Finset Nat) (saucers : Finset Nat) : Prop := sorry

/-- The main theorem stating the minimum number of items needed -/
theorem min_items_for_matching_pair :
  ∀ (n : Nat),
    (∀ (cups saucers : Finset Nat),
      cups.card + saucers.card = n →
      cups.card ≤ items_per_box →
      saucers.card ≤ items_per_box →
      ¬ guarantees_matching_pair cups saucers) ↔
    n < 32 :=
sorry

end NUMINAMATH_CALUDE_min_items_for_matching_pair_l920_92003


namespace NUMINAMATH_CALUDE_set_operations_l920_92060

-- Define the universal set U
def U : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- Define set A
def A : Set ℤ := {x : ℤ | x^2 - 5*x - 6 = 0}

-- Define set B
def B : Set ℤ := {x : ℤ | x^2 = 1}

-- Theorem statement
theorem set_operations :
  (A ∪ B = {-1, 1, 6}) ∧
  (A ∩ B = {-1}) ∧
  (U \ (A ∩ B) = {0, 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l920_92060


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l920_92009

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + 2*i) / (i - 1)
  Complex.im z = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l920_92009


namespace NUMINAMATH_CALUDE_yellow_not_more_than_green_l920_92097

/-- Represents the three types of parrots -/
inductive ParrotType
  | Green
  | Yellow
  | Mottled

/-- Represents whether a parrot tells the truth or lies -/
inductive ParrotResponse
  | Truth
  | Lie

/-- The total number of parrots -/
def totalParrots : Nat := 100

/-- The number of parrots that agreed with each statement -/
def agreeingParrots : Nat := 50

/-- Function that determines how a parrot responds based on its type -/
def parrotBehavior (t : ParrotType) (statement : Nat) : ParrotResponse :=
  match t with
  | ParrotType.Green => ParrotResponse.Truth
  | ParrotType.Yellow => ParrotResponse.Lie
  | ParrotType.Mottled => if statement == 1 then ParrotResponse.Truth else ParrotResponse.Lie

/-- Theorem stating that the number of yellow parrots cannot exceed the number of green parrots -/
theorem yellow_not_more_than_green 
  (G Y M : Nat) 
  (h_total : G + Y + M = totalParrots)
  (h_first_statement : G + M / 2 = agreeingParrots)
  (h_second_statement : M / 2 + Y = agreeingParrots) :
  Y ≤ G :=
sorry

end NUMINAMATH_CALUDE_yellow_not_more_than_green_l920_92097


namespace NUMINAMATH_CALUDE_game_positions_after_359_moves_l920_92001

/-- Represents the four positions of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the twelve positions of the mouse -/
inductive MousePosition
  | TopLeft | TopMiddle | TopRight
  | RightTop | RightMiddle | RightBottom
  | BottomRight | BottomMiddle | BottomLeft
  | LeftBottom | LeftMiddle | LeftTop

/-- Calculates the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : ℕ) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.TopRight
  | 2 => CatPosition.BottomRight
  | _ => CatPosition.BottomLeft

/-- Calculates the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : ℕ) : MousePosition :=
  match moves % 12 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightTop
  | 4 => MousePosition.RightMiddle
  | 5 => MousePosition.RightBottom
  | 6 => MousePosition.BottomRight
  | 7 => MousePosition.BottomMiddle
  | 8 => MousePosition.BottomLeft
  | 9 => MousePosition.LeftBottom
  | 10 => MousePosition.LeftMiddle
  | _ => MousePosition.LeftTop

theorem game_positions_after_359_moves :
  catPositionAfterMoves 359 = CatPosition.BottomRight ∧
  mousePositionAfterMoves 359 = MousePosition.LeftMiddle :=
by sorry

end NUMINAMATH_CALUDE_game_positions_after_359_moves_l920_92001


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_120_l920_92012

theorem largest_multiple_of_9_less_than_120 : 
  ∀ n : ℕ, n % 9 = 0 → n < 120 → n ≤ 117 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_120_l920_92012


namespace NUMINAMATH_CALUDE_third_group_draw_l920_92099

/-- Represents a systematic sampling sequence -/
def SystematicSampling (first second : ℕ) : ℕ → ℕ := fun n => first + (n - 1) * (second - first)

/-- Theorem: In a systematic sampling where the first group draws 2 and the second group draws 12,
    the third group will draw 22 -/
theorem third_group_draw (first second : ℕ) (h1 : first = 2) (h2 : second = 12) :
  SystematicSampling first second 3 = 22 := by
  sorry

#eval SystematicSampling 2 12 3

end NUMINAMATH_CALUDE_third_group_draw_l920_92099


namespace NUMINAMATH_CALUDE_family_income_change_l920_92081

theorem family_income_change (initial_average : ℚ) (initial_members : ℕ) 
  (deceased_income : ℚ) (new_members : ℕ) : 
  initial_average = 840 →
  initial_members = 4 →
  deceased_income = 1410 →
  new_members = 3 →
  (initial_average * initial_members - deceased_income) / new_members = 650 := by
  sorry

end NUMINAMATH_CALUDE_family_income_change_l920_92081


namespace NUMINAMATH_CALUDE_cube_edge_length_l920_92062

theorem cube_edge_length (a : ℝ) :
  (6 * a^2 = a^3 → a = 6) ∧
  (6 * a^2 = (a^3)^2 → a = Real.rpow 6 (1/4)) ∧
  ((6 * a^2)^3 = a^3 → a = 1/36) :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l920_92062


namespace NUMINAMATH_CALUDE_games_to_reach_target_win_rate_l920_92089

def initial_games : ℕ := 20
def initial_win_rate : ℚ := 95 / 100
def target_win_rate : ℚ := 96 / 100

theorem games_to_reach_target_win_rate :
  let initial_wins := (initial_games : ℚ) * initial_win_rate
  ∃ (additional_games : ℕ),
    (initial_wins + additional_games) / (initial_games + additional_games) = target_win_rate ∧
    additional_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_to_reach_target_win_rate_l920_92089


namespace NUMINAMATH_CALUDE_ratio_equality_l920_92006

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (x + y - z) / (2 * x - y + z) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l920_92006


namespace NUMINAMATH_CALUDE_bonus_allocation_l920_92076

theorem bonus_allocation (bonus : ℚ) (kitchen_fraction : ℚ) (christmas_fraction : ℚ) (leftover : ℚ) 
  (h1 : bonus = 1496)
  (h2 : kitchen_fraction = 1 / 22)
  (h3 : christmas_fraction = 1 / 8)
  (h4 : leftover = 867)
  (h5 : bonus * kitchen_fraction + bonus * christmas_fraction + bonus * (holiday_fraction : ℚ) + leftover = bonus) :
  holiday_fraction = 187 / 748 := by
  sorry

end NUMINAMATH_CALUDE_bonus_allocation_l920_92076


namespace NUMINAMATH_CALUDE_min_value_of_expression_l920_92026

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) passing through the center of the circle x² + y² + 4x - 4y - 1 = 0,
    the minimum value of 2/a + 3/b is 5 + 2√6 -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : ∃ (x y : ℝ), a * x - b * y + 2 = 0)
    (h_circle : ∃ (x y : ℝ), x^2 + y^2 + 4*x - 4*y - 1 = 0)
    (h_center : ∃ (x y : ℝ), (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (a * x - b * y + 2 = 0)) :
    (∀ (a' b' : ℝ), (a' > 0 ∧ b' > 0) → (2/a' + 3/b' ≥ 5 + 2 * Real.sqrt 6)) ∧
    (∃ (a' b' : ℝ), (a' > 0 ∧ b' > 0) ∧ (2/a' + 3/b' = 5 + 2 * Real.sqrt 6)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l920_92026


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l920_92021

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The theorem statement -/
theorem inequality_range_theorem (m : ℝ) :
  (∀ x ∈ Set.Ici (2/3), f (x/m) - 4*m^2*f x ≤ f (x-1) + 4*f m) →
  m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l920_92021
