import Mathlib

namespace NUMINAMATH_CALUDE_first_year_after_2000_sum_12_correct_l3667_366749

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a year is after 2000 -/
def is_after_2000 (year : ℕ) : Prop := year > 2000

/-- The first year after 2000 with sum of digits 12 -/
def first_year_after_2000_sum_12 : ℕ := 2019

theorem first_year_after_2000_sum_12_correct :
  (is_after_2000 first_year_after_2000_sum_12) ∧
  (sum_of_digits first_year_after_2000_sum_12 = 12) ∧
  (∀ y : ℕ, is_after_2000 y ∧ sum_of_digits y = 12 → y ≥ first_year_after_2000_sum_12) :=
by sorry

end NUMINAMATH_CALUDE_first_year_after_2000_sum_12_correct_l3667_366749


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3667_366786

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3667_366786


namespace NUMINAMATH_CALUDE_three_cubes_volume_l3667_366777

-- Define the volume of a cube
def cubeVolume (edge : ℝ) : ℝ := edge ^ 3

-- Define the total volume of three cubes
def totalVolume (edge1 edge2 edge3 : ℝ) : ℝ :=
  cubeVolume edge1 + cubeVolume edge2 + cubeVolume edge3

-- Theorem statement
theorem three_cubes_volume :
  totalVolume 3 5 6 = 368 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_volume_l3667_366777


namespace NUMINAMATH_CALUDE_nathan_tomato_harvest_l3667_366745

/-- Represents the harvest and sales data for Nathan's garden --/
structure GardenData where
  strawberry_plants : ℕ
  tomato_plants : ℕ
  strawberries_per_plant : ℕ
  fruits_per_basket : ℕ
  strawberry_basket_price : ℕ
  tomato_basket_price : ℕ
  total_revenue : ℕ

/-- Calculates the number of tomatoes harvested per plant --/
def tomatoes_per_plant (data : GardenData) : ℕ :=
  let strawberry_baskets := (data.strawberry_plants * data.strawberries_per_plant) / data.fruits_per_basket
  let strawberry_revenue := strawberry_baskets * data.strawberry_basket_price
  let tomato_revenue := data.total_revenue - strawberry_revenue
  let tomato_baskets := tomato_revenue / data.tomato_basket_price
  let total_tomatoes := tomato_baskets * data.fruits_per_basket
  total_tomatoes / data.tomato_plants

/-- Theorem stating that given Nathan's garden data, he harvested 16 tomatoes per plant --/
theorem nathan_tomato_harvest :
  let data : GardenData := {
    strawberry_plants := 5,
    tomato_plants := 7,
    strawberries_per_plant := 14,
    fruits_per_basket := 7,
    strawberry_basket_price := 9,
    tomato_basket_price := 6,
    total_revenue := 186
  }
  tomatoes_per_plant data = 16 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tomato_harvest_l3667_366745


namespace NUMINAMATH_CALUDE_sin_90_degrees_l3667_366798

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l3667_366798


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3667_366724

/-- Given a cube with surface area 24 square centimeters, prove its volume is 8 cubic centimeters. -/
theorem cube_volume_from_surface_area : 
  ∀ (s : ℝ), 
  (6 * s^2 = 24) →  -- surface area formula
  (s^3 = 8)         -- volume formula
:= by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3667_366724


namespace NUMINAMATH_CALUDE_product_of_solutions_l3667_366784

theorem product_of_solutions (x₁ x₂ : ℚ) : 
  (|6 * x₁ + 2| + 5 = 47) → 
  (|6 * x₂ + 2| + 5 = 47) → 
  x₁ ≠ x₂ → 
  x₁ * x₂ = -440 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3667_366784


namespace NUMINAMATH_CALUDE_colored_balls_probabilities_l3667_366735

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculate the probability of drawing a ball of a specific color -/
def probability (bag : ColoredBalls) (color : ℕ) : ℚ :=
  color / bag.total

/-- Calculate the number of red balls to add to achieve a target probability -/
def addRedBalls (bag : ColoredBalls) (targetProb : ℚ) : ℕ :=
  let x := (targetProb * bag.total - bag.red) / (1 - targetProb)
  x.ceil.toNat

theorem colored_balls_probabilities (bag : ColoredBalls) :
  bag.total = 10 ∧ bag.red = 4 ∧ bag.yellow = 6 →
  (probability bag bag.yellow = 3/5) ∧
  (addRedBalls bag (2/3) = 8) := by
  sorry

end NUMINAMATH_CALUDE_colored_balls_probabilities_l3667_366735


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1764_l3667_366744

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem largest_perfect_square_factor_of_1764 :
  ∀ k : ℕ, is_perfect_square k ∧ k ∣ 1764 → k ≤ 1764 :=
by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1764_l3667_366744


namespace NUMINAMATH_CALUDE_min_values_theorem_l3667_366770

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) :
  (∀ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 → 1 / x + 2 / y ≥ 9) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 1 / x + 2 / y = 9) ∧
  (∀ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 → 2^x + 4^y ≥ 2 * Real.sqrt 2) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 2^x + 4^y = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l3667_366770


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3667_366773

theorem absolute_value_equation_solution (x : ℝ) : 
  |2*x - 1| + |x - 2| = |x + 1| ↔ 1/2 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3667_366773


namespace NUMINAMATH_CALUDE_greatest_sum_of_valid_pair_l3667_366781

/-- Two integers that differ by 2 and have a product less than 500 -/
def ValidPair (n m : ℤ) : Prop :=
  m = n + 2 ∧ n * m < 500

/-- The sum of a valid pair of integers -/
def PairSum (n m : ℤ) : ℤ := n + m

/-- Theorem: The greatest possible sum of two integers that differ by 2 
    and whose product is less than 500 is 44 -/
theorem greatest_sum_of_valid_pair : 
  (∃ (n m : ℤ), ValidPair n m ∧ 
    ∀ (k l : ℤ), ValidPair k l → PairSum k l ≤ PairSum n m) ∧
  (∀ (n m : ℤ), ValidPair n m → PairSum n m ≤ 44) := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_of_valid_pair_l3667_366781


namespace NUMINAMATH_CALUDE_student_count_equality_l3667_366754

/-- Proves that the number of students in class A equals the number in class C,
    given the average ages of each class and the overall average age. -/
theorem student_count_equality (a b c : ℕ) : 
  (14 * a + 13 * b + 12 * c : ℝ) / (a + b + c : ℝ) = 13 → a = c := by
  sorry

end NUMINAMATH_CALUDE_student_count_equality_l3667_366754


namespace NUMINAMATH_CALUDE_at_least_two_primes_in_base_n_1002_l3667_366791

def base_n_1002 (n : ℕ) : ℕ := n^3 + 2

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

theorem at_least_two_primes_in_base_n_1002 : 
  ∃ n1 n2 : ℕ, n1 ≥ 2 ∧ n2 ≥ 2 ∧ n1 ≠ n2 ∧ 
  is_prime (base_n_1002 n1) ∧ is_prime (base_n_1002 n2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_primes_in_base_n_1002_l3667_366791


namespace NUMINAMATH_CALUDE_storks_on_fence_l3667_366719

/-- The number of storks that joined the birds on the fence -/
def num_storks_joined : ℕ := 4

theorem storks_on_fence :
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let total_birds : ℕ := initial_birds + additional_birds
  let bird_stork_difference : ℕ := 3
  num_storks_joined = total_birds - bird_stork_difference := by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_l3667_366719


namespace NUMINAMATH_CALUDE_min_xy_value_l3667_366795

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x * y ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ * y₀ = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3667_366795


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l3667_366778

theorem circle_equation_k_value (x y k : ℝ) : 
  (∃ h c : ℝ, ∀ x y : ℝ, x^2 + 12*x + y^2 + 14*y - k = 0 ↔ (x - h)^2 + (y - c)^2 = 8^2) ↔ 
  k = 85 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l3667_366778


namespace NUMINAMATH_CALUDE_rotation_center_l3667_366727

noncomputable def f (z : ℂ) : ℂ := ((-1 - Complex.I * Real.sqrt 3) * z + (-2 * Real.sqrt 3 + 18 * Complex.I)) / 2

theorem rotation_center :
  ∃ (c : ℂ), f c = c ∧ c = -2 * Real.sqrt 3 - 4 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_rotation_center_l3667_366727


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3667_366705

theorem perfect_square_condition (x y k : ℝ) :
  (∃ (z : ℝ), x^2 + k*x*y + 49*y^2 = z^2) → (k = 14 ∨ k = -14) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3667_366705


namespace NUMINAMATH_CALUDE_nancy_money_total_l3667_366732

/-- Given that Nancy has 9 5-dollar bills, prove that she has $45 in total. -/
theorem nancy_money_total :
  let num_bills : ℕ := 9
  let bill_value : ℕ := 5
  num_bills * bill_value = 45 := by
sorry

end NUMINAMATH_CALUDE_nancy_money_total_l3667_366732


namespace NUMINAMATH_CALUDE_prime_pair_existence_l3667_366716

theorem prime_pair_existence (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p + 2 = q ∧ Prime (2^n + p) ∧ Prime (2^n + q)) ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_pair_existence_l3667_366716


namespace NUMINAMATH_CALUDE_complex_product_real_implies_sum_modulus_l3667_366721

theorem complex_product_real_implies_sum_modulus (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := a + 3*I
  (z₁ * z₂).im = 0 → Complex.abs (z₁ + z₂) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_sum_modulus_l3667_366721


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l3667_366776

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ+) : 
  Nat.gcd A B = 23 →
  A = 391 →
  Nat.lcm A B = 23 * 16 * X →
  X = 17 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l3667_366776


namespace NUMINAMATH_CALUDE_small_circle_radius_l3667_366733

/-- Given a large circle with radius 10 meters and seven congruent smaller circles
    arranged so that four of their diameters align with the diameter of the large circle,
    the radius of each smaller circle is 2.5 meters. -/
theorem small_circle_radius (large_radius : ℝ) (small_radius : ℝ) : 
  large_radius = 10 → 
  4 * (2 * small_radius) = 2 * large_radius → 
  small_radius = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3667_366733


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3667_366706

/-- Given a triangle with two sides of lengths 4 and 6, and the third side length
    being a root of (x-6)(x-10)=0, prove that the perimeter of the triangle is 16. -/
theorem triangle_perimeter : ∀ x : ℝ, 
  (x - 6) * (x - 10) = 0 → 
  (4 < x ∧ x < 10) →  -- Triangle inequality
  4 + 6 + x = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3667_366706


namespace NUMINAMATH_CALUDE_relay_race_average_time_l3667_366731

/-- Calculates the average time for a leg of a relay race given the times of two runners. -/
def average_leg_time (y_time z_time : ℕ) : ℚ :=
  (y_time + z_time : ℚ) / 2

/-- Theorem stating that for the given runner times, the average leg time is 42 seconds. -/
theorem relay_race_average_time :
  average_leg_time 58 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_average_time_l3667_366731


namespace NUMINAMATH_CALUDE_ian_money_left_l3667_366717

/-- Calculates the amount of money Ian has left after expenses and taxes --/
def money_left (hours_worked : ℕ) (hourly_rate : ℚ) (monthly_expense : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_earnings := hours_worked * hourly_rate
  let tax := tax_rate * total_earnings
  let net_earnings := total_earnings - tax
  let amount_spent := (1/2) * net_earnings
  let remaining_after_spending := net_earnings - amount_spent
  remaining_after_spending - monthly_expense

/-- Theorem stating that Ian has $14.80 left after expenses and taxes --/
theorem ian_money_left :
  money_left 8 18 50 (1/10) = 148/10 :=
by sorry

end NUMINAMATH_CALUDE_ian_money_left_l3667_366717


namespace NUMINAMATH_CALUDE_area_probability_l3667_366780

/-- A square in a 2D plane -/
structure Square :=
  (A B C D : ℝ × ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a square -/
def isInside (s : Square) (p : Point) : Prop := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- The probability of an event occurring when a point is chosen randomly inside a square -/
def probability (s : Square) (event : Point → Prop) : ℝ := sorry

/-- The main theorem -/
theorem area_probability (s : Square) :
  probability s (fun p => 
    isInside s p ∧ 
    triangleArea s.A s.B p > triangleArea s.B s.C p ∧
    triangleArea s.A s.B p > triangleArea s.C s.D p ∧
    triangleArea s.A s.B p > triangleArea s.D s.A p) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_area_probability_l3667_366780


namespace NUMINAMATH_CALUDE_triangle_stack_impossibility_l3667_366738

theorem triangle_stack_impossibility : ¬ ∃ (n : ℕ), n > 0 ∧ (n * (1 + 2 + 3)) / 3 = 1997 := by
  sorry

end NUMINAMATH_CALUDE_triangle_stack_impossibility_l3667_366738


namespace NUMINAMATH_CALUDE_line_graph_best_for_daily_income_fluctuations_l3667_366763

-- Define the types of statistical graphs
inductive StatGraph
| LineGraph
| BarGraph
| PieChart
| Histogram

-- Define a structure for daily income data
structure DailyIncomeData :=
  (days : Fin 7 → ℝ)

-- Define a property for showing fluctuations intuitively
def shows_fluctuations_intuitively (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

-- Define the theorem
theorem line_graph_best_for_daily_income_fluctuations 
  (data : DailyIncomeData) :
  ∃ (best : StatGraph), 
    (shows_fluctuations_intuitively best ∧ 
     ∀ (g : StatGraph), shows_fluctuations_intuitively g → g = best) :=
sorry

end NUMINAMATH_CALUDE_line_graph_best_for_daily_income_fluctuations_l3667_366763


namespace NUMINAMATH_CALUDE_hidden_dots_count_l3667_366700

/-- Represents a standard six-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def SumOfDie : ℕ := Finset.sum StandardDie id

/-- The number of dice in the stack -/
def NumberOfDice : ℕ := 4

/-- The visible numbers on the dice -/
def VisibleNumbers : Finset ℕ := {1, 2, 2, 3, 4, 4, 5, 6, 6}

/-- The theorem stating the number of hidden dots -/
theorem hidden_dots_count : 
  NumberOfDice * SumOfDie - Finset.sum VisibleNumbers id = 51 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l3667_366700


namespace NUMINAMATH_CALUDE_geese_left_is_10_l3667_366793

/-- The number of geese that left the duck park -/
def geese_left : ℕ := by sorry

theorem geese_left_is_10 :
  let initial_ducks : ℕ := 25
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let final_ducks : ℕ := initial_ducks + 4
  let final_geese : ℕ := initial_geese - geese_left
  (final_geese = final_ducks + 1) →
  geese_left = 10 := by sorry

end NUMINAMATH_CALUDE_geese_left_is_10_l3667_366793


namespace NUMINAMATH_CALUDE_group_leader_selection_l3667_366746

theorem group_leader_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_group_leader_selection_l3667_366746


namespace NUMINAMATH_CALUDE_lilies_bought_l3667_366740

/-- Given the cost of roses and lilies, the total paid, and the change received,
    prove that the number of lilies bought is 6. -/
theorem lilies_bought (rose_cost : ℕ) (lily_cost : ℕ) (total_paid : ℕ) (change : ℕ) : 
  rose_cost = 3000 →
  lily_cost = 2800 →
  total_paid = 25000 →
  change = 2200 →
  (total_paid - change - 2 * rose_cost) / lily_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_lilies_bought_l3667_366740


namespace NUMINAMATH_CALUDE_not_divisible_by_two_l3667_366765

theorem not_divisible_by_two (n : ℕ) (h_pos : n > 0) 
  (h_sum : ∃ k : ℤ, (1 : ℚ) / 2 + 1 / 3 + 1 / 5 + 1 / n = k) : 
  ¬(2 ∣ n) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_two_l3667_366765


namespace NUMINAMATH_CALUDE_expected_messages_is_27_l3667_366779

/-- Calculates the expected number of greeting messages --/
def expected_messages (total_colleagues : ℕ) 
  (probabilities : List ℝ) (people_counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (· * ·) probabilities people_counts)

/-- Theorem: The expected number of greeting messages is 27 --/
theorem expected_messages_is_27 : 
  let total_colleagues : ℕ := 40
  let probabilities : List ℝ := [1, 0.8, 0.5, 0]
  let people_counts : List ℕ := [8, 15, 14, 3]
  expected_messages total_colleagues probabilities people_counts = 27 := by
  sorry

end NUMINAMATH_CALUDE_expected_messages_is_27_l3667_366779


namespace NUMINAMATH_CALUDE_petes_total_miles_l3667_366785

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total miles walked given a pedometer and steps per mile --/
def total_miles_walked (p : Pedometer) (steps_per_mile : ℕ) : ℚ :=
  ((p.resets * (p.max_steps + 1) + p.final_reading) : ℚ) / steps_per_mile

/-- Theorem stating that Pete walked 2512.5 miles given the problem conditions --/
theorem petes_total_miles :
  let p : Pedometer := ⟨99999, 50, 25000⟩
  let steps_per_mile : ℕ := 2000
  total_miles_walked p steps_per_mile = 2512.5 := by
  sorry


end NUMINAMATH_CALUDE_petes_total_miles_l3667_366785


namespace NUMINAMATH_CALUDE_sqrt_3600_equals_60_l3667_366751

theorem sqrt_3600_equals_60 : Real.sqrt 3600 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3600_equals_60_l3667_366751


namespace NUMINAMATH_CALUDE_min_degree_g_l3667_366761

variables (x : ℝ) (f g h : ℝ → ℝ)

def is_polynomial (p : ℝ → ℝ) : Prop := sorry

def degree (p : ℝ → ℝ) : ℕ := sorry

theorem min_degree_g 
  (eq : ∀ x, 5 * f x + 7 * g x = h x)
  (f_poly : is_polynomial f)
  (g_poly : is_polynomial g)
  (h_poly : is_polynomial h)
  (f_deg : degree f = 10)
  (h_deg : degree h = 13) :
  degree g ≥ 13 ∧ ∃ g', is_polynomial g' ∧ degree g' = 13 ∧ ∀ x, 5 * f x + 7 * g' x = h x :=
sorry

end NUMINAMATH_CALUDE_min_degree_g_l3667_366761


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_120_l3667_366730

theorem largest_multiple_of_8_less_than_120 :
  ∃ n : ℕ, n * 8 = 112 ∧ 
  112 < 120 ∧
  ∀ m : ℕ, m * 8 < 120 → m * 8 ≤ 112 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_120_l3667_366730


namespace NUMINAMATH_CALUDE_child_admission_price_l3667_366783

theorem child_admission_price
  (total_people : ℕ)
  (adult_price : ℚ)
  (total_receipts : ℚ)
  (num_children : ℕ)
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : total_receipts = 960)
  (h4 : num_children = 260) :
  (total_receipts - (adult_price * (total_people - num_children))) / num_children = 1 :=
by sorry

end NUMINAMATH_CALUDE_child_admission_price_l3667_366783


namespace NUMINAMATH_CALUDE_max_xy_value_l3667_366734

theorem max_xy_value (x y c : ℝ) (h : x + y = c - 195) : 
  ∃ d : ℝ, d = 4 ∧ ∀ x' y' : ℝ, x' + y' = c - 195 → x' * y' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3667_366734


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l3667_366764

/-- Represents the number of pills in a bottle -/
def pills_per_bottle : ℕ := 90

/-- Represents the fraction of a pill taken per dose -/
def fraction_per_dose : ℚ := 1/3

/-- Represents the number of days between doses -/
def days_between_doses : ℕ := 3

/-- Represents the average number of days in a month -/
def days_per_month : ℕ := 30

/-- Proves that the supply of medicine lasts 27 months -/
theorem medicine_supply_duration :
  (pills_per_bottle : ℚ) * days_between_doses / fraction_per_dose / days_per_month = 27 := by
  sorry


end NUMINAMATH_CALUDE_medicine_supply_duration_l3667_366764


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_function_l3667_366796

theorem max_value_of_quadratic_function :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x : ℝ, x - x^2 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_function_l3667_366796


namespace NUMINAMATH_CALUDE_train_speed_problem_l3667_366788

/-- Proves that the initial speed of a train is 110 km/h given specific journey conditions -/
theorem train_speed_problem (T : ℝ) : ∃ v : ℝ,
  v > 0 ∧
  v - 50 > 0 ∧
  T > 0 ∧
  T + 2/3 = 212/v + 88/(v - 50) ∧
  v = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3667_366788


namespace NUMINAMATH_CALUDE_expression_evaluation_l3667_366768

theorem expression_evaluation (a b c d : ℝ) :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3667_366768


namespace NUMINAMATH_CALUDE_ice_pop_price_is_correct_l3667_366701

/-- The selling price of an ice-pop that allows a school to buy pencils, given:
  * The cost to make each ice-pop
  * The cost of each pencil
  * The number of ice-pops that need to be sold
  * The number of pencils to be bought
-/
def ice_pop_selling_price (make_cost : ℚ) (pencil_cost : ℚ) (pops_sold : ℕ) (pencils_bought : ℕ) : ℚ :=
  make_cost + (pencil_cost * pencils_bought - make_cost * pops_sold) / pops_sold

/-- Theorem stating that the selling price of each ice-pop is $1.20 under the given conditions -/
theorem ice_pop_price_is_correct :
  ice_pop_selling_price 0.90 1.80 300 100 = 1.20 := by
  sorry

end NUMINAMATH_CALUDE_ice_pop_price_is_correct_l3667_366701


namespace NUMINAMATH_CALUDE_pencil_price_l3667_366707

theorem pencil_price (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 315)
  (eq2 : 3 * x + 6 * y = 243) :
  y = 15 := by
sorry

end NUMINAMATH_CALUDE_pencil_price_l3667_366707


namespace NUMINAMATH_CALUDE_millennium_running_time_l3667_366760

/-- The running time of Millennium in minutes -/
def millennium_time : ℕ := 120

/-- The running time of Alpha Epsilon in minutes -/
def alpha_epsilon_time : ℕ := millennium_time - 30

/-- The running time of Beast of War: Armoured Command in minutes -/
def beast_of_war_time : ℕ := alpha_epsilon_time + 10

/-- Theorem stating that Millennium's running time is 120 minutes -/
theorem millennium_running_time : 
  millennium_time = 120 ∧ 
  alpha_epsilon_time = millennium_time - 30 ∧
  beast_of_war_time = alpha_epsilon_time + 10 ∧
  beast_of_war_time = 100 :=
by sorry

end NUMINAMATH_CALUDE_millennium_running_time_l3667_366760


namespace NUMINAMATH_CALUDE_product_not_always_minimized_when_closest_l3667_366769

theorem product_not_always_minimized_when_closest (d : ℝ) (h : d > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = d ∧
  ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ x' - y' = d ∧
  abs (x' - y') < abs (x - y) ∧ x' * y' < x * y :=
by sorry

-- Other statements (A, B, D, E) are correct, but we don't need to prove them for this task

end NUMINAMATH_CALUDE_product_not_always_minimized_when_closest_l3667_366769


namespace NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l3667_366729

/-- Proves that Kenneth will be 10 yards past the finish line when Biff crosses the finish line in a 500-yard race -/
theorem kenneth_distance_past_finish_line 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_speed : ℝ) 
  (h1 : race_distance = 500) 
  (h2 : biff_speed = 50) 
  (h3 : kenneth_speed = 51) : 
  kenneth_speed * (race_distance / biff_speed) - race_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_distance_past_finish_line_l3667_366729


namespace NUMINAMATH_CALUDE_bryan_books_count_l3667_366708

/-- The number of bookshelves Bryan has -/
def num_shelves : ℕ := 2

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 17

/-- The total number of books Bryan has -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem bryan_books_count : total_books = 34 := by
  sorry

end NUMINAMATH_CALUDE_bryan_books_count_l3667_366708


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3667_366714

theorem diophantine_equation_solution (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3 * n ^ 2 + 4 * n) 
  (hb : b ≤ 3 * n ^ 2 + 4 * n) 
  (hc : c ≤ 3 * n ^ 2 + 4 * n) : 
  ∃ (x y z : ℤ), 
    (abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n) ∧ 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (a * x + b * y + c * z = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3667_366714


namespace NUMINAMATH_CALUDE_dislike_tv_and_books_l3667_366755

/-- Given a population where some dislike TV and some of those also dislike books,
    calculate the number of people who dislike both TV and books. -/
theorem dislike_tv_and_books
  (total_population : ℕ)
  (tv_dislike_percent : ℚ)
  (book_dislike_percent : ℚ)
  (h_total : total_population = 1500)
  (h_tv : tv_dislike_percent = 25 / 100)
  (h_book : book_dislike_percent = 15 / 100) :
  ⌊(tv_dislike_percent * book_dislike_percent * total_population : ℚ)⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_dislike_tv_and_books_l3667_366755


namespace NUMINAMATH_CALUDE_same_color_probability_l3667_366743

theorem same_color_probability (total : ℕ) (black white : ℕ) 
  (h1 : (black * (black - 1)) / (total * (total - 1)) = 1 / 7)
  (h2 : (white * (white - 1)) / (total * (total - 1)) = 12 / 35) :
  ((black * (black - 1)) + (white * (white - 1))) / (total * (total - 1)) = 17 / 35 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l3667_366743


namespace NUMINAMATH_CALUDE_beads_per_necklace_is_eight_l3667_366772

/-- The number of beads Emily has -/
def total_beads : ℕ := 16

/-- The number of necklaces Emily can make -/
def num_necklaces : ℕ := 2

/-- The number of beads per necklace -/
def beads_per_necklace : ℕ := total_beads / num_necklaces

theorem beads_per_necklace_is_eight : beads_per_necklace = 8 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_is_eight_l3667_366772


namespace NUMINAMATH_CALUDE_cost_of_45_daffodils_l3667_366774

/-- The cost of a bouquet is directly proportional to the number of daffodils it contains. -/
structure DaffodilBouquet where
  daffodils : ℕ
  cost : ℝ

/-- A bouquet of 15 daffodils costs $25. -/
def standard_bouquet : DaffodilBouquet := ⟨15, 25⟩

/-- The proposition that the cost of a 45-daffodil bouquet is $75. -/
theorem cost_of_45_daffodils : 
  ∀ (b : DaffodilBouquet), b.daffodils = 45 → 
  (b.cost / b.daffodils : ℝ) = (standard_bouquet.cost / standard_bouquet.daffodils) → 
  b.cost = 75 := by
sorry

end NUMINAMATH_CALUDE_cost_of_45_daffodils_l3667_366774


namespace NUMINAMATH_CALUDE_lcm_of_20_45_60_l3667_366799

theorem lcm_of_20_45_60 : Nat.lcm (Nat.lcm 20 45) 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_60_l3667_366799


namespace NUMINAMATH_CALUDE_convex_polyhedron_properties_l3667_366789

/-- A convex polyhedron with congruent isosceles triangular faces -/
structure ConvexPolyhedron where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  isConvex : Bool
  hasCongruentIsoscelesFaces : Bool
  formGeometricSequence : Bool

/-- Euler's formula for polyhedra -/
axiom euler_formula {p : ConvexPolyhedron} : p.vertices + p.faces = p.edges + 2

/-- Relation between faces and edges in a polyhedron with triangular faces -/
axiom triangular_faces_relation {p : ConvexPolyhedron} : 2 * p.edges = 3 * p.faces

/-- Geometric sequence property -/
axiom geometric_sequence {p : ConvexPolyhedron} (h : p.formGeometricSequence) :
  p.faces / p.vertices = p.edges / p.faces

/-- Main theorem: A convex polyhedron with the given properties has 8 vertices, 12 faces, and 18 edges -/
theorem convex_polyhedron_properties (p : ConvexPolyhedron)
  (h1 : p.isConvex)
  (h2 : p.hasCongruentIsoscelesFaces)
  (h3 : p.formGeometricSequence) :
  p.vertices = 8 ∧ p.faces = 12 ∧ p.edges = 18 := by
  sorry

end NUMINAMATH_CALUDE_convex_polyhedron_properties_l3667_366789


namespace NUMINAMATH_CALUDE_alphanumeric_puzzle_l3667_366723

theorem alphanumeric_puzzle :
  ∃! (A B C D E F H J K L : Nat),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
     F < 10 ∧ H < 10 ∧ J < 10 ∧ K < 10 ∧ L < 10) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ H ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ H ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ H ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ H ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧
     E ≠ F ∧ E ≠ H ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧
     F ≠ H ∧ F ≠ J ∧ F ≠ K ∧ F ≠ L ∧
     H ≠ J ∧ H ≠ K ∧ H ≠ L ∧
     J ≠ K ∧ J ≠ L ∧
     K ≠ L) ∧
    (A * B = B) ∧
    (B * C = 10 * A + C) ∧
    (C * D = 10 * B + C) ∧
    (D * E = 10 * C + H) ∧
    (E * F = 10 * D + K) ∧
    (F * H = 10 * C + J) ∧
    (H * J = 10 * K + J) ∧
    (J * K = E) ∧
    (K * L = L) ∧
    (A * L = L) ∧
    (A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0) :=
by sorry

end NUMINAMATH_CALUDE_alphanumeric_puzzle_l3667_366723


namespace NUMINAMATH_CALUDE_i_power_sum_i_power_sum_proof_l3667_366702

theorem i_power_sum : Complex → Prop :=
  fun i => i * i = -1 → i^20 + i^35 = 1 - i

-- The proof would go here, but we're skipping it as requested
theorem i_power_sum_proof : i_power_sum Complex.I :=
  sorry

end NUMINAMATH_CALUDE_i_power_sum_i_power_sum_proof_l3667_366702


namespace NUMINAMATH_CALUDE_construction_time_difference_l3667_366762

/-- Represents the work rate of one person per day -/
def work_rate : ℝ := 1

/-- Calculates the total work done given the number of workers, days, and work rate -/
def total_work (workers : ℕ) (days : ℕ) (rate : ℝ) : ℝ :=
  (workers : ℝ) * (days : ℝ) * rate

/-- Theorem: If 100 men work for 50 days and then 200 men work for another 50 days
    to complete a project in 100 days, it would take 150 days for 100 men to
    complete the same project working at the same rate. -/
theorem construction_time_difference :
  let initial_workers : ℕ := 100
  let additional_workers : ℕ := 100
  let initial_days : ℕ := 50
  let total_days : ℕ := 100
  let work_done_first_half := total_work initial_workers initial_days work_rate
  let work_done_second_half := total_work (initial_workers + additional_workers) initial_days work_rate
  let total_work_done := work_done_first_half + work_done_second_half
  total_work initial_workers 150 work_rate = total_work_done :=
by
  sorry

end NUMINAMATH_CALUDE_construction_time_difference_l3667_366762


namespace NUMINAMATH_CALUDE_sandy_money_left_l3667_366720

/-- The amount of money Sandy has left after buying a pie -/
def money_left (initial_amount pie_cost : ℕ) : ℕ :=
  initial_amount - pie_cost

/-- Theorem: Sandy has 57 dollars left after buying the pie -/
theorem sandy_money_left :
  money_left 63 6 = 57 :=
by sorry

end NUMINAMATH_CALUDE_sandy_money_left_l3667_366720


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_ten_l3667_366741

theorem sqrt_sum_equals_two_sqrt_ten : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_ten_l3667_366741


namespace NUMINAMATH_CALUDE_output_value_after_five_years_l3667_366739

/-- Calculates the final value after compound growth -/
def final_value (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ years

/-- Theorem: The output value after 5 years with 8% annual growth -/
theorem output_value_after_five_years 
  (a : ℝ) -- initial value
  (h1 : a > 0) -- initial value is positive
  (h2 : a = 1000000) -- initial value is 1 million yuan
  : final_value a 0.08 5 = a * (1 + 0.08) ^ 5 := by
  sorry

#eval final_value 1000000 0.08 5

end NUMINAMATH_CALUDE_output_value_after_five_years_l3667_366739


namespace NUMINAMATH_CALUDE_integer_linear_combination_sqrt2_sqrt3_l3667_366704

theorem integer_linear_combination_sqrt2_sqrt3 (a b c : ℤ) :
  a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0 → a = 0 ∧ b = 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_linear_combination_sqrt2_sqrt3_l3667_366704


namespace NUMINAMATH_CALUDE_num_machines_proof_l3667_366722

/-- The number of machines that complete a job in 6 hours, 
    given that 2 machines complete the same job in 24 hours -/
def num_machines : ℕ :=
  let time_many : ℕ := 6  -- time taken by multiple machines
  let time_two : ℕ := 24   -- time taken by 2 machines
  let machines_two : ℕ := 2  -- number of machines in second scenario
  8  -- to be proved

theorem num_machines_proof : 
  num_machines * time_many = machines_two * time_two :=
by sorry

end NUMINAMATH_CALUDE_num_machines_proof_l3667_366722


namespace NUMINAMATH_CALUDE_average_marks_l3667_366792

structure Marks where
  physics : ℕ
  chemistry : ℕ
  mathematics : ℕ
  biology : ℕ
  english : ℕ
  history : ℕ
  geography : ℕ

def valid_marks (m : Marks) : Prop :=
  m.chemistry = m.physics + 75 ∧
  m.mathematics = m.chemistry + 30 ∧
  m.biology = m.physics - 15 ∧
  m.english = m.biology - 10 ∧
  m.history = m.biology - 10 ∧
  m.geography = m.biology - 10 ∧
  m.physics + m.chemistry + m.mathematics + m.biology + m.english + m.history + m.geography = m.physics + 520 ∧
  m.physics ≥ 40 ∧ m.chemistry ≥ 40 ∧ m.mathematics ≥ 40 ∧ m.biology ≥ 40 ∧
  m.english ≥ 40 ∧ m.history ≥ 40 ∧ m.geography ≥ 40

theorem average_marks (m : Marks) (h : valid_marks m) :
  (m.mathematics + m.biology + m.history + m.geography) / 4 = 82 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l3667_366792


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_l3667_366710

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in October -/
structure OctoberDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a Terrific Tuesday -/
def isTerrificTuesday (date : OctoberDate) : Prop :=
  date.dayOfWeek = DayOfWeek.Tuesday ∧ 
  (∃ n : Nat, n = 5 ∧ date.day = n * 7 - 4)

/-- The company's start date -/
def startDate : OctoberDate :=
  { day := 2, dayOfWeek := DayOfWeek.Monday }

/-- The number of days in October -/
def octoberDays : Nat := 31

/-- Theorem: The first Terrific Tuesday after operations begin is October 31 -/
theorem first_terrific_tuesday : 
  ∃ (date : OctoberDate), 
    date.day = 31 ∧ 
    isTerrificTuesday date ∧ 
    ∀ (earlier : OctoberDate), 
      earlier.day > startDate.day ∧ 
      earlier.day < date.day → 
      ¬isTerrificTuesday earlier :=
by sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_l3667_366710


namespace NUMINAMATH_CALUDE_yellow_opposite_blue_l3667_366713

-- Define the colors
inductive Color
  | Red
  | Blue
  | Orange
  | Yellow
  | Green
  | White

-- Define a square with colors on both sides
structure Square where
  front : Color
  back : Color

-- Define the cube
structure Cube where
  squares : Vector Square 6

-- Define the function to get the opposite face
def oppositeFace (c : Cube) (face : Color) : Color :=
  sorry

-- Theorem statement
theorem yellow_opposite_blue (c : Cube) :
  (∃ (s : Square), s ∈ c.squares.toList ∧ s.front = Color.Yellow) →
  oppositeFace c Color.Yellow = Color.Blue :=
sorry

end NUMINAMATH_CALUDE_yellow_opposite_blue_l3667_366713


namespace NUMINAMATH_CALUDE_square_difference_greater_than_polynomial_l3667_366752

theorem square_difference_greater_than_polynomial :
  ∀ x : ℝ, (x - 3)^2 > x^2 - 6*x + 8 := by
sorry

end NUMINAMATH_CALUDE_square_difference_greater_than_polynomial_l3667_366752


namespace NUMINAMATH_CALUDE_sin_2023_closest_to_neg_sqrt2_over_2_l3667_366797

-- Define the set of options
def options : Set ℝ := {1/2, Real.sqrt 2 / 2, -1/2, -Real.sqrt 2 / 2}

-- Define the sine function with period 360°
noncomputable def periodic_sin (x : ℝ) : ℝ := Real.sin (2 * Real.pi * (x / 360))

-- State the theorem
theorem sin_2023_closest_to_neg_sqrt2_over_2 :
  ∃ (y : ℝ), y ∈ options ∧ 
  ∀ (z : ℝ), z ∈ options → |periodic_sin 2023 - y| ≤ |periodic_sin 2023 - z| :=
sorry

end NUMINAMATH_CALUDE_sin_2023_closest_to_neg_sqrt2_over_2_l3667_366797


namespace NUMINAMATH_CALUDE_centroid_satisfies_conditions_l3667_366758

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Check if a point is on a line segment between two other points -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if two line segments are parallel -/
def areParallel (p1 : Point) (p2 : Point) (q1 : Point) (q2 : Point) : Prop := sorry

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 : Point) (p2 : Point) (p3 : Point) : ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point := sorry

theorem centroid_satisfies_conditions (t : Triangle) :
  ∃ (O L M N : Point),
    isInside O t ∧
    isOnSegment L t.A t.B ∧
    isOnSegment M t.B t.C ∧
    isOnSegment N t.C t.A ∧
    areParallel O L t.B t.C ∧
    areParallel O M t.A t.C ∧
    areParallel O N t.A t.B ∧
    triangleArea O t.B L = triangleArea O t.C M ∧
    triangleArea O t.C M = triangleArea O t.A N ∧
    O = centroid t :=
  sorry

end NUMINAMATH_CALUDE_centroid_satisfies_conditions_l3667_366758


namespace NUMINAMATH_CALUDE_fred_has_nine_dimes_l3667_366766

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The amount of money Fred has in cents -/
def fred_money : ℕ := 90

/-- The number of dimes Fred has -/
def fred_dimes : ℕ := fred_money / dime_value

theorem fred_has_nine_dimes : fred_dimes = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_nine_dimes_l3667_366766


namespace NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l3667_366712

/-- Represents a chess player's probability of winning against an opponent -/
structure PlayerProb where
  prob : ℝ
  pos : prob > 0

/-- Represents the probabilities of winning against three players -/
structure ThreePlayerProbs where
  p₁ : PlayerProb
  p₂ : PlayerProb
  p₃ : PlayerProb
  p₃_gt_p₂ : p₃.prob > p₂.prob
  p₂_gt_p₁ : p₂.prob > p₁.prob

/-- Calculates the probability of winning two consecutive games given the order of opponents -/
def prob_two_consecutive_wins (probs : ThreePlayerProbs) (second_player : ℕ) : ℝ :=
  match second_player with
  | 1 => 2 * (probs.p₁.prob * (probs.p₂.prob + probs.p₃.prob) - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)
  | 2 => 2 * (probs.p₂.prob * (probs.p₁.prob + probs.p₃.prob) - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)
  | _ => 2 * (probs.p₁.prob * probs.p₃.prob + probs.p₂.prob * probs.p₃.prob - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)

theorem max_prob_with_highest_prob_second (probs : ThreePlayerProbs) :
  ∀ i, prob_two_consecutive_wins probs 3 ≥ prob_two_consecutive_wins probs i :=
sorry

end NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l3667_366712


namespace NUMINAMATH_CALUDE_S_pq_equation_l3667_366711

/-- S(n) is the sum of squares of positive integers less than and coprime to n -/
def S (n : ℕ) : ℕ := sorry

/-- p is a prime number equal to 2^7 - 1 -/
def p : ℕ := 127

/-- q is a prime number equal to 2^5 - 1 -/
def q : ℕ := 31

/-- a is a positive integer -/
def a : ℕ := 7561

theorem S_pq_equation : 
  ∃ (b c : ℕ), 
    b < c ∧ 
    Nat.Coprime b c ∧
    S (p * q) = (p^2 * q^2 / 6) * (a - b / c) := by sorry

end NUMINAMATH_CALUDE_S_pq_equation_l3667_366711


namespace NUMINAMATH_CALUDE_unique_root_implies_specific_angles_l3667_366757

/-- Given α ∈ (0, π), if the equation |2x - 1/2| + |(\sqrt{6} - \sqrt{2})x| = sin α
    has exactly one real root, then α = π/12 or α = 11π/12 -/
theorem unique_root_implies_specific_angles (α : Real) 
    (h1 : α ∈ Set.Ioo 0 Real.pi)
    (h2 : ∃! x : Real, |2*x - 1/2| + |((Real.sqrt 6) - (Real.sqrt 2))*x| = Real.sin α) :
    α = Real.pi/12 ∨ α = 11*Real.pi/12 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_implies_specific_angles_l3667_366757


namespace NUMINAMATH_CALUDE_not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq_l3667_366771

theorem not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 :=
sorry

end NUMINAMATH_CALUDE_not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq_l3667_366771


namespace NUMINAMATH_CALUDE_zhang_bin_is_journalist_l3667_366742

structure Person where
  name : String
  isJournalist : Bool
  statement : Bool

def liZhiming : Person := { name := "Li Zhiming", isJournalist := false, statement := true }
def zhangBin : Person := { name := "Zhang Bin", isJournalist := false, statement := true }
def wangDawei : Person := { name := "Wang Dawei", isJournalist := false, statement := true }

theorem zhang_bin_is_journalist :
  ∀ (li : Person) (zhang : Person) (wang : Person),
    li.name = "Li Zhiming" →
    zhang.name = "Zhang Bin" →
    wang.name = "Wang Dawei" →
    (li.isJournalist ∨ zhang.isJournalist ∨ wang.isJournalist) →
    (li.isJournalist → ¬zhang.isJournalist ∧ ¬wang.isJournalist) →
    (zhang.isJournalist → ¬li.isJournalist ∧ ¬wang.isJournalist) →
    (wang.isJournalist → ¬li.isJournalist ∧ ¬zhang.isJournalist) →
    li.statement = li.isJournalist →
    zhang.statement = ¬zhang.isJournalist →
    wang.statement = ¬li.statement →
    (li.statement ∨ zhang.statement ∨ wang.statement) →
    (li.statement → ¬zhang.statement ∧ ¬wang.statement) →
    (zhang.statement → ¬li.statement ∧ ¬wang.statement) →
    (wang.statement → ¬li.statement ∧ ¬zhang.statement) →
    zhang.isJournalist := by
  sorry

#check zhang_bin_is_journalist

end NUMINAMATH_CALUDE_zhang_bin_is_journalist_l3667_366742


namespace NUMINAMATH_CALUDE_christen_peeled_18_potatoes_l3667_366718

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  christen_join_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def potatoes_peeled_by_christen (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 18 potatoes in the given scenario -/
theorem christen_peeled_18_potatoes :
  let scenario : PotatoPeeling := {
    total_potatoes := 50,
    homer_rate := 4,
    christen_rate := 6,
    christen_join_time := 5
  }
  potatoes_peeled_by_christen scenario = 18 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_18_potatoes_l3667_366718


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3667_366750

theorem half_angle_quadrant (α : Real) (h : ∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  ∃ m : ℤ, (m * π < α / 2 ∧ α / 2 < m * π + π / 2) ∨ 
           (m * π + π < α / 2 ∧ α / 2 < m * π + 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3667_366750


namespace NUMINAMATH_CALUDE_range_of_a_l3667_366794

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(p x) ∧ q x a) → 
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3667_366794


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_third_l3667_366747

theorem tan_theta_plus_pi_third (θ : Real) (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) :
  (Real.sin (3 * Real.pi / 4) : Real) / Real.cos (3 * Real.pi / 4) = Real.tan θ →
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_third_l3667_366747


namespace NUMINAMATH_CALUDE_class_average_mark_l3667_366767

theorem class_average_mark (total_students : Nat) (excluded_students : Nat) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 25 → 
  excluded_students = 5 → 
  excluded_avg = 40 → 
  remaining_avg = 90 → 
  (total_students * (total_students * remaining_avg - excluded_students * excluded_avg)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l3667_366767


namespace NUMINAMATH_CALUDE_sine_cosine_equation_solution_l3667_366787

theorem sine_cosine_equation_solution (x : ℝ) :
  12 * Real.sin x - 5 * Real.cos x = 13 →
  ∃ k : ℤ, x = π / 2 + Real.arctan (5 / 12) + 2 * π * ↑k :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_equation_solution_l3667_366787


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3667_366790

-- Define the geometric sequence and its sum
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 2

-- Define the general formula for the sequence
def general_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 * (3 ^ (n - 1))

-- Theorem statement
theorem geometric_sequence_formula 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : geometric_sequence a S) : 
  general_formula a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3667_366790


namespace NUMINAMATH_CALUDE_train_journey_time_l3667_366782

/-- Proves that the current time taken to cover a distance is 50 minutes 
    given the conditions from the train problem. -/
theorem train_journey_time : 
  ∀ (distance : ℝ) (current_time : ℝ),
    distance > 0 →
    distance = 48 * (current_time / 60) →
    distance = 60 * (40 / 60) →
    current_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l3667_366782


namespace NUMINAMATH_CALUDE_number_divided_by_16_equals_4_l3667_366756

theorem number_divided_by_16_equals_4 (x : ℤ) : x / 16 = 4 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_16_equals_4_l3667_366756


namespace NUMINAMATH_CALUDE_min_power_of_two_greater_than_factorial_10_l3667_366775

-- Define the given logarithm values
def log10_2 : ℝ := 0.301
def log10_3 : ℝ := 0.477
def log10_7 : ℝ := 0.845

-- Define 10!
def factorial_10 : ℕ := 3628800

-- Theorem statement
theorem min_power_of_two_greater_than_factorial_10 :
  ∃ n : ℕ, factorial_10 < 2^n ∧ ∀ m : ℕ, m < n → factorial_10 ≥ 2^m :=
sorry

end NUMINAMATH_CALUDE_min_power_of_two_greater_than_factorial_10_l3667_366775


namespace NUMINAMATH_CALUDE_family_age_relations_l3667_366737

/-- Given family ages and relationships, prove age difference and Teresa's age at Michiko's birth -/
theorem family_age_relations (teresa_age morio_age : ℕ) 
  (h1 : teresa_age = 59)
  (h2 : morio_age = 71)
  (h3 : morio_age - 38 = michiko_age)
  (h4 : michiko_age - 4 = kenji_age)
  (h5 : teresa_age - 10 = emiko_age)
  (h6 : kenji_age = hideki_age)
  (h7 : morio_age = ryuji_age) :
  michiko_age - hideki_age = 4 ∧ teresa_age - michiko_age = 26 :=
by sorry


end NUMINAMATH_CALUDE_family_age_relations_l3667_366737


namespace NUMINAMATH_CALUDE_impossible_arrangement_l3667_366703

/-- A grid of integers -/
def Grid := Matrix (Fin 25) (Fin 41) ℤ

/-- Predicate to check if a grid satisfies the adjacency condition -/
def SatisfiesAdjacencyCondition (g : Grid) : Prop :=
  ∀ i j i' j', (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) →
    |g i j - g i' j'| ≤ 16

/-- Predicate to check if a grid contains distinct integers -/
def ContainsDistinctIntegers (g : Grid) : Prop :=
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement : 
  ¬∃ (g : Grid), SatisfiesAdjacencyCondition g ∧ ContainsDistinctIntegers g :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l3667_366703


namespace NUMINAMATH_CALUDE_max_area_constrained_rectangle_l3667_366715

/-- Represents a rectangular garden with given length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Checks if a rectangle satisfies the given constraints. -/
def satisfiesConstraints (r : Rectangle) : Prop :=
  perimeter r = 400 ∧ r.length ≥ 100 ∧ r.width ≥ 50

/-- States that the maximum area of a constrained rectangle is 7500. -/
theorem max_area_constrained_rectangle :
  ∀ r : Rectangle, satisfiesConstraints r → area r ≤ 7500 :=
by sorry

end NUMINAMATH_CALUDE_max_area_constrained_rectangle_l3667_366715


namespace NUMINAMATH_CALUDE_tire_price_proof_l3667_366728

/-- The regular price of a tire -/
def regular_price : ℝ := 115.71

/-- The total amount paid for four tires -/
def total_paid : ℝ := 405

/-- The promotion deal: 3 tires at regular price, 1 at half price -/
theorem tire_price_proof :
  3 * regular_price + (1/2) * regular_price = total_paid :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l3667_366728


namespace NUMINAMATH_CALUDE_percent_of_y_l3667_366753

theorem percent_of_y (y : ℝ) (h : y > 0) : ((9 * y) / 20 + (3 * y) / 10) / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3667_366753


namespace NUMINAMATH_CALUDE_student_speaking_probability_l3667_366726

/-- The probability of a student speaking truth -/
def prob_truth : ℝ := 0.30

/-- The probability of a student speaking lie -/
def prob_lie : ℝ := 0.20

/-- The probability of a student speaking both truth and lie -/
def prob_both : ℝ := 0.10

/-- The probability of a student speaking either truth or lie -/
def prob_truth_or_lie : ℝ := prob_truth + prob_lie - prob_both

theorem student_speaking_probability :
  prob_truth_or_lie = 0.40 := by sorry

end NUMINAMATH_CALUDE_student_speaking_probability_l3667_366726


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l3667_366736

theorem power_of_negative_cube (a : ℝ) : (-(a^3))^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l3667_366736


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3667_366725

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- Atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- Atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- Number of Carbon atoms in the compound -/
def num_C : ℕ := 3

/-- Number of Nitrogen atoms in the compound -/
def num_N : ℕ := 1

/-- Number of Chlorine atoms in the compound -/
def num_Cl : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ :=
  (num_H : ℝ) * atomic_weight_H +
  (num_C : ℝ) * atomic_weight_C +
  (num_N : ℝ) * atomic_weight_N +
  (num_Cl : ℝ) * atomic_weight_Cl +
  (num_O : ℝ) * atomic_weight_O

/-- Theorem stating that the molecular weight of the compound is 135.51 g/mol -/
theorem compound_molecular_weight :
  molecular_weight = 135.51 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3667_366725


namespace NUMINAMATH_CALUDE_difference_of_squares_2x_3_l3667_366709

theorem difference_of_squares_2x_3 (x : ℝ) : (2*x + 3) * (2*x - 3) = 4*x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_2x_3_l3667_366709


namespace NUMINAMATH_CALUDE_rowing_problem_l3667_366759

/-- Proves that given the conditions of the rowing problem, the downstream distance is 60 km -/
theorem rowing_problem (upstream_distance : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) (stream_speed : ℝ)
  (h1 : upstream_distance = 30)
  (h2 : upstream_time = 3)
  (h3 : downstream_time = 3)
  (h4 : stream_speed = 5) :
  let boat_speed := upstream_distance / upstream_time + stream_speed
  let downstream_speed := boat_speed + stream_speed
  downstream_speed * downstream_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_rowing_problem_l3667_366759


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l3667_366748

theorem consecutive_odd_numbers_problem (x : ℤ) : 
  Odd x ∧ 
  (8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) ∧ 
  (∃ p : ℕ, Prime p ∧ (x + (x + 2) + (x + 4)) % p = 0) → 
  x = 7 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l3667_366748
