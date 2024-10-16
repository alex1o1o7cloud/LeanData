import Mathlib

namespace NUMINAMATH_CALUDE_percent_decrease_proof_l1973_197315

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 70) :
  (original_price - sale_price) / original_price * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l1973_197315


namespace NUMINAMATH_CALUDE_cakes_left_l1973_197382

def cakes_per_day : ℕ := 20
def baking_days : ℕ := 9
def total_cakes : ℕ := cakes_per_day * baking_days
def sold_cakes : ℕ := total_cakes / 2
def remaining_cakes : ℕ := total_cakes - sold_cakes

theorem cakes_left : remaining_cakes = 90 := by
  sorry

end NUMINAMATH_CALUDE_cakes_left_l1973_197382


namespace NUMINAMATH_CALUDE_intersection_slopes_sum_l1973_197356

/-- Given a line y = 2x - 3 and a parabola y² = 4x intersecting at points A and B,
    with O as the origin and k₁, k₂ as the slopes of OA and OB respectively,
    prove that the sum of the reciprocals of the slopes 1/k₁ + 1/k₂ = 1/2 -/
theorem intersection_slopes_sum (A B : ℝ × ℝ) (k₁ k₂ : ℝ) : 
  (A.2 = 2 * A.1 - 3) →
  (B.2 = 2 * B.1 - 3) →
  (A.2^2 = 4 * A.1) →
  (B.2^2 = 4 * B.1) →
  (k₁ = A.2 / A.1) →
  (k₂ = B.2 / B.1) →
  (1 / k₁ + 1 / k₂ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_slopes_sum_l1973_197356


namespace NUMINAMATH_CALUDE_fruit_basket_total_l1973_197322

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Calculates the total number of fruits in the basket -/
def totalFruits (basket : FruitBasket) : ℕ :=
  basket.oranges + basket.apples + basket.bananas + basket.peaches

/-- Theorem stating the total number of fruits in the basket is 28 -/
theorem fruit_basket_total :
  ∃ (basket : FruitBasket),
    basket.oranges = 6 ∧
    basket.apples = basket.oranges - 2 ∧
    basket.bananas = 3 * basket.apples ∧
    basket.peaches = basket.bananas / 2 ∧
    totalFruits basket = 28 := by
  sorry


end NUMINAMATH_CALUDE_fruit_basket_total_l1973_197322


namespace NUMINAMATH_CALUDE_trains_crossing_time_l1973_197311

/-- Proves the time taken for two trains to cross each other -/
theorem trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h1 : length = 120)
  (h2 : time1 = 5)
  (h3 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l1973_197311


namespace NUMINAMATH_CALUDE_longest_piece_length_l1973_197391

/-- Given a rope of length 92.5 inches cut into three pieces in the ratio 3:5:8,
    the length of the longest piece is 46.25 inches. -/
theorem longest_piece_length (total_length : ℝ) (ratio_1 ratio_2 ratio_3 : ℕ) 
    (h1 : total_length = 92.5)
    (h2 : ratio_1 = 3)
    (h3 : ratio_2 = 5)
    (h4 : ratio_3 = 8) :
    (ratio_3 : ℝ) * total_length / ((ratio_1 : ℝ) + (ratio_2 : ℝ) + (ratio_3 : ℝ)) = 46.25 :=
by sorry

end NUMINAMATH_CALUDE_longest_piece_length_l1973_197391


namespace NUMINAMATH_CALUDE_sum_reciprocal_product_bound_sum_product_bound_l1973_197393

-- Part (a)
theorem sum_reciprocal_product_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by sorry

-- Part (b)
theorem sum_product_bound (u v : ℝ) (hu : 0 < u ∧ u < 1) (hv : 0 < v ∧ v < 1) :
  0 < u + v - u*v ∧ u + v - u*v < 1 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_product_bound_sum_product_bound_l1973_197393


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1973_197335

theorem arithmetic_calculations :
  ((-2 : ℝ) + |3| + (-6) + |7| = 2) ∧
  (3.7 + (-1.3) + (-6.7) + 2.3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1973_197335


namespace NUMINAMATH_CALUDE_complement_of_A_l1973_197362

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

theorem complement_of_A : 
  (U \ A) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1973_197362


namespace NUMINAMATH_CALUDE_apple_cost_theorem_l1973_197399

/-- The cost of groceries for Olivia -/
def grocery_problem (total_cost banana_cost bread_cost milk_cost apple_cost : ℕ) : Prop :=
  total_cost = 42 ∧
  banana_cost = 12 ∧
  bread_cost = 9 ∧
  milk_cost = 7 ∧
  apple_cost = total_cost - (banana_cost + bread_cost + milk_cost)

theorem apple_cost_theorem :
  ∃ (apple_cost : ℕ), grocery_problem 42 12 9 7 apple_cost ∧ apple_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_theorem_l1973_197399


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1973_197348

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric progression -/
def geometric_prog (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_prog (a 1) (a 2) (a 5) →
  a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1973_197348


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1973_197355

theorem inequality_system_integer_solutions :
  let S : Set ℤ := {x | (5 * x - 2 > 3 * (x + 1)) ∧ (x / 3 ≤ (5 - x) / 2)}
  S = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1973_197355


namespace NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_point_circle_radius_l1973_197359

theorem circle_radius_zero (x y : ℝ) :
  x^2 + 8*x + y^2 - 10*y + 41 = 0 → (x + 4)^2 + (y - 5)^2 = 0 :=
by sorry

theorem circle_equation_point (x y : ℝ) :
  (x + 4)^2 + (y - 5)^2 = 0 → x = -4 ∧ y = 5 :=
by sorry

theorem circle_radius (x y : ℝ) :
  x^2 + 8*x + y^2 - 10*y + 41 = 0 → ∃! (center : ℝ × ℝ), center = (-4, 5) ∧ (x - center.1)^2 + (y - center.2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_point_circle_radius_l1973_197359


namespace NUMINAMATH_CALUDE_unique_quadratic_with_prime_roots_l1973_197360

theorem unique_quadratic_with_prime_roots (a : ℝ) (ha : a > 0) :
  (∃! k : ℝ, ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    (∀ x : ℝ, x^2 + (k^2 + a*k)*x + (1999 + k^2 + a*k) = 0 ↔ x = p ∨ x = q)) ↔ 
  a = 2 * Real.sqrt 502 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_with_prime_roots_l1973_197360


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1973_197338

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x, x^2 + (m-2)*x + (5-m) = 0 → x > 2) → 
  m ∈ Set.Ioo (-5) (-4) ∪ {-4} :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1973_197338


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1973_197375

/-- The ratio of runner a's speed to runner b's speed in a race -/
def speed_ratio (head_start_percent : ℚ) (winning_distance_percent : ℚ) : ℚ :=
  (1 + head_start_percent) / (1 + winning_distance_percent)

/-- Theorem stating that the speed ratio is 37/35 given the specified conditions -/
theorem race_speed_ratio :
  speed_ratio (48/100) (40/100) = 37/35 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1973_197375


namespace NUMINAMATH_CALUDE_total_reams_is_five_l1973_197388

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := 3

/-- The total number of reams of paper bought -/
def total_reams : ℕ := reams_for_haley + reams_for_sister

theorem total_reams_is_five : total_reams = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_reams_is_five_l1973_197388


namespace NUMINAMATH_CALUDE_min_cookie_count_l1973_197347

def is_valid_cookie_count (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + 21 * b ∧ n % 13 = 0

theorem min_cookie_count : 
  (∀ m : ℕ, m > 0 ∧ m < 52 → ¬(is_valid_cookie_count m)) ∧
  is_valid_cookie_count 52 :=
sorry

end NUMINAMATH_CALUDE_min_cookie_count_l1973_197347


namespace NUMINAMATH_CALUDE_smallest_number_l1973_197307

theorem smallest_number (a b c d : ℝ) (h1 : a = 0) (h2 : b = -Real.rpow 8 (1/3)) (h3 : c = 2) (h4 : d = -1.7) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l1973_197307


namespace NUMINAMATH_CALUDE_percentage_problem_l1973_197354

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x / 100 * 20 = 8) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1973_197354


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1973_197345

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem total_interest_calculation (rate : ℕ) : 
  rate = 10 → 
  simple_interest 5000 rate 2 + simple_interest 3000 rate 4 = 2200 := by
  sorry

#check total_interest_calculation

end NUMINAMATH_CALUDE_total_interest_calculation_l1973_197345


namespace NUMINAMATH_CALUDE_surfers_ratio_l1973_197387

def surfers_problem (first_day : ℕ) (second_day_increase : ℕ) (average : ℕ) : Prop :=
  let second_day := first_day + second_day_increase
  let total := average * 3
  let third_day := total - first_day - second_day
  (third_day : ℚ) / first_day = 2 / 5

theorem surfers_ratio : 
  surfers_problem 1500 600 1400 := by sorry

end NUMINAMATH_CALUDE_surfers_ratio_l1973_197387


namespace NUMINAMATH_CALUDE_correct_fills_l1973_197324

/-- The amount of flour needed in cups -/
def flour_needed : ℚ := 15/4

/-- The amount of milk needed in cups -/
def milk_needed : ℚ := 3/2

/-- The capacity of the flour measuring cup in cups -/
def flour_measure : ℚ := 1/3

/-- The capacity of the milk measuring cup in cups -/
def milk_measure : ℚ := 1/4

/-- The number of times to fill the flour measuring cup -/
def flour_fills : ℕ := 12

/-- The number of times to fill the milk measuring cup -/
def milk_fills : ℕ := 6

/-- Theorem stating that the number of fills for flour and milk are correct -/
theorem correct_fills :
  (flour_fills : ℚ) * flour_measure ≥ flour_needed ∧
  ((flour_fills - 1 : ℕ) : ℚ) * flour_measure < flour_needed ∧
  (milk_fills : ℚ) * milk_measure = milk_needed :=
sorry

end NUMINAMATH_CALUDE_correct_fills_l1973_197324


namespace NUMINAMATH_CALUDE_solution_set_is_closed_unit_interval_l1973_197317

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the set of a that satisfy f(1) ≤ f(a)
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {a | f 1 ≤ f a}

-- State the theorem
theorem solution_set_is_closed_unit_interval
  (f : ℝ → ℝ) (h_even : is_even f) (h_incr : increasing_on_nonpositive f) :
  solution_set f = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_closed_unit_interval_l1973_197317


namespace NUMINAMATH_CALUDE_oxford_high_school_teachers_l1973_197365

theorem oxford_high_school_teachers (num_classes : ℕ) (students_per_class : ℕ) (total_people : ℕ) :
  num_classes = 15 →
  students_per_class = 20 →
  total_people = 349 →
  total_people = num_classes * students_per_class + 1 + 48 :=
by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_teachers_l1973_197365


namespace NUMINAMATH_CALUDE_equation_solution_l1973_197330

theorem equation_solution : ∃ x : ℝ, (1 / 7 + 4 / x = 12 / x + 1 / 14) ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1973_197330


namespace NUMINAMATH_CALUDE_sum_binary_digits_310_l1973_197320

def sum_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem sum_binary_digits_310 : sum_binary_digits 310 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_310_l1973_197320


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1973_197326

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + 2)^2 - (2 * Real.sqrt 3 + 3 * Real.sqrt 2) * (3 * Real.sqrt 2 - 2 * Real.sqrt 3) = 1 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1973_197326


namespace NUMINAMATH_CALUDE_loan_future_value_l1973_197314

/-- Represents the relationship between principal and future value for a loan -/
theorem loan_future_value 
  (P A : ℝ) -- Principal and future value
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  (h1 : r = 0.12) -- Interest rate is 12%
  (h2 : n = 2) -- Compounded half-yearly
  (h3 : t = 20) -- Loan period is 20 years
  : A = P * (1 + r/n)^(n*t) :=
by sorry

end NUMINAMATH_CALUDE_loan_future_value_l1973_197314


namespace NUMINAMATH_CALUDE_remainder_98_102_div_11_l1973_197397

theorem remainder_98_102_div_11 : (98 * 102) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_11_l1973_197397


namespace NUMINAMATH_CALUDE_expression_simplification_l1973_197352

/-- Proves that the given expression simplifies to the expected result. -/
theorem expression_simplification (x y : ℝ) :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1973_197352


namespace NUMINAMATH_CALUDE_john_candy_count_l1973_197380

/-- Represents the number of candies each friend has -/
structure CandyCounts where
  bob : ℕ
  mary : ℕ
  john : ℕ
  sue : ℕ
  sam : ℕ

/-- The total number of candies all friends have together -/
def totalCandies : ℕ := 50

/-- The given candy counts for Bob, Mary, Sue, and Sam -/
def givenCounts : CandyCounts where
  bob := 10
  mary := 5
  john := 0  -- We don't know John's count yet
  sue := 20
  sam := 10

/-- Theorem stating that John's candy count is equal to the total minus the sum of others -/
theorem john_candy_count (c : CandyCounts) (h : c = givenCounts) :
  c.john = totalCandies - (c.bob + c.mary + c.sue + c.sam) :=
by sorry

end NUMINAMATH_CALUDE_john_candy_count_l1973_197380


namespace NUMINAMATH_CALUDE_scooter_initial_value_l1973_197367

/-- The depreciation rate of the scooter's value each year -/
def depreciation_rate : ℚ := 3/4

/-- The number of years of depreciation -/
def years : ℕ := 4

/-- The value of the scooter after 4 years in rupees -/
def final_value : ℚ := 12656.25

/-- The initial value of the scooter in rupees -/
def initial_value : ℚ := 30000

/-- Theorem stating that given the depreciation rate, number of years, and final value,
    the initial value of the scooter can be calculated -/
theorem scooter_initial_value :
  initial_value * depreciation_rate ^ years = final_value := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l1973_197367


namespace NUMINAMATH_CALUDE_intersection_S_complement_T_l1973_197385

-- Define the universal set U as ℝ
def U := ℝ

-- Define set S
def S : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define set T
def T : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- State the theorem
theorem intersection_S_complement_T : S ∩ (Set.univ \ T) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_S_complement_T_l1973_197385


namespace NUMINAMATH_CALUDE_correct_subtraction_l1973_197333

theorem correct_subtraction (x : ℤ) : x - 64 = 122 → x - 46 = 140 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l1973_197333


namespace NUMINAMATH_CALUDE_total_distance_is_410_l1973_197316

-- Define bird types and their speeds
structure Bird where
  name : String
  speed : ℝ
  flightTime : ℝ

-- Define constants
def headwind : ℝ := 5
def totalBirds : ℕ := 6

-- Define the list of birds
def birds : List Bird := [
  { name := "eagle", speed := 15, flightTime := 2.5 },
  { name := "falcon", speed := 46, flightTime := 2.5 },
  { name := "pelican", speed := 33, flightTime := 2.5 },
  { name := "hummingbird", speed := 30, flightTime := 2.5 },
  { name := "hawk", speed := 45, flightTime := 3 },
  { name := "swallow", speed := 25, flightTime := 1.5 }
]

-- Calculate actual distance traveled by a bird
def actualDistance (bird : Bird) : ℝ :=
  (bird.speed - headwind) * bird.flightTime

-- Calculate total distance traveled by all birds
def totalDistance : ℝ :=
  (birds.map actualDistance).sum

-- Theorem to prove
theorem total_distance_is_410 : totalDistance = 410 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_410_l1973_197316


namespace NUMINAMATH_CALUDE_log_product_telescoping_l1973_197374

theorem log_product_telescoping (z : ℝ) : 
  z = (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
      (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
      (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
      (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
      (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
      (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) * 
      (Real.log 16 / Real.log 15) * (Real.log 17 / Real.log 16) * 
      (Real.log 18 / Real.log 17) * (Real.log 19 / Real.log 18) * 
      (Real.log 20 / Real.log 19) * (Real.log 21 / Real.log 20) * 
      (Real.log 22 / Real.log 21) * (Real.log 23 / Real.log 22) * 
      (Real.log 24 / Real.log 23) * (Real.log 25 / Real.log 24) * 
      (Real.log 26 / Real.log 25) * (Real.log 27 / Real.log 26) * 
      (Real.log 28 / Real.log 27) * (Real.log 29 / Real.log 28) * 
      (Real.log 30 / Real.log 29) * (Real.log 31 / Real.log 30) * 
      (Real.log 32 / Real.log 31) * (Real.log 33 / Real.log 32) * 
      (Real.log 34 / Real.log 33) * (Real.log 35 / Real.log 34) * 
      (Real.log 36 / Real.log 35) * (Real.log 37 / Real.log 36) * 
      (Real.log 38 / Real.log 37) * (Real.log 39 / Real.log 38) * 
      (Real.log 40 / Real.log 39) →
  z = (3 * Real.log 2 + Real.log 5) / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_telescoping_l1973_197374


namespace NUMINAMATH_CALUDE_derivative_of_exp_neg_x_l1973_197378

theorem derivative_of_exp_neg_x (x : ℝ) : deriv (fun x => Real.exp (-x)) x = -Real.exp (-x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_neg_x_l1973_197378


namespace NUMINAMATH_CALUDE_obtuse_angle_measure_l1973_197310

theorem obtuse_angle_measure (α β : Real) (p : Real) :
  (∃ (x y : Real), x^2 + p*(x+1) + 1 = 0 ∧ y^2 + p*(y+1) + 1 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (α > 0 ∧ β > 0 ∧ α + β < Real.pi) →
  ∃ (γ : Real), γ = Real.pi - α - β ∧ γ = 3*Real.pi/4 :=
by sorry

end NUMINAMATH_CALUDE_obtuse_angle_measure_l1973_197310


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l1973_197301

/-- A rectangular solid with volume 512 cm³, surface area 384 cm², and dimensions in geometric progression has a sum of edge lengths equal to 96 cm. -/
theorem rectangular_solid_edge_sum : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 384 →
    ∃ (r : ℝ), r > 0 ∧ b = a * r ∧ c = b * r →
    4 * (a + b + c) = 96 :=
by sorry


end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l1973_197301


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1973_197342

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a b c : ℕ → ℝ) :
  ArithmeticSequence a ∧ ArithmeticSequence b ∧ ArithmeticSequence c →
  a 1 + b 1 + c 1 = 0 →
  a 2 + b 2 + c 2 = 1 →
  a 2015 + b 2015 + c 2015 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1973_197342


namespace NUMINAMATH_CALUDE_local_max_implies_local_min_l1973_197319

-- Define the function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number not equal to 0
variable (x₀ : ℝ)
variable (h₁ : x₀ ≠ 0)

-- Define that x₀ is a local maximum point of f
def isLocalMax (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀

variable (h₂ : isLocalMax f x₀)

-- Define what it means for a point to be a local minimum
def isLocalMin (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x

-- Theorem statement
theorem local_max_implies_local_min :
  isLocalMin (fun x => -f (-x)) (-x₀) := by sorry

end NUMINAMATH_CALUDE_local_max_implies_local_min_l1973_197319


namespace NUMINAMATH_CALUDE_tan_theta_is_negative_three_l1973_197386

/-- Given vectors a and b with angle θ between them, if a • b = -1, a = (-1, 2), and |b| = √2, then tan θ = -3 -/
theorem tan_theta_is_negative_three (a b : ℝ × ℝ) (θ : ℝ) :
  a = (-1, 2) →
  a • b = -1 →
  ‖b‖ = Real.sqrt 2 →
  Real.tan θ = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_is_negative_three_l1973_197386


namespace NUMINAMATH_CALUDE_smallest_number_problem_l1973_197334

theorem smallest_number_problem (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a + b + c = 100)
  (h4 : c = 2 * a)
  (h5 : c - b = 10) : 
  a = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l1973_197334


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l1973_197357

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l1973_197357


namespace NUMINAMATH_CALUDE_D_most_stable_l1973_197371

-- Define the variances for each person
def variance_A : ℝ := 0.56
def variance_B : ℝ := 0.60
def variance_C : ℝ := 0.50
def variance_D : ℝ := 0.45

-- Define a function to determine if one variance is more stable than another
def more_stable (v1 v2 : ℝ) : Prop := v1 < v2

-- Theorem stating that D has the most stable performance
theorem D_most_stable :
  more_stable variance_D variance_C ∧
  more_stable variance_D variance_A ∧
  more_stable variance_D variance_B :=
by sorry

end NUMINAMATH_CALUDE_D_most_stable_l1973_197371


namespace NUMINAMATH_CALUDE_seaweed_distribution_l1973_197328

theorem seaweed_distribution (total_seaweed : ℝ) (fire_percentage : ℝ) (human_percentage : ℝ) :
  total_seaweed = 400 ∧ 
  fire_percentage = 0.5 ∧ 
  human_percentage = 0.25 →
  (total_seaweed * (1 - fire_percentage) * (1 - human_percentage)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_distribution_l1973_197328


namespace NUMINAMATH_CALUDE_min_sum_squares_l1973_197379

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + 3 * x₂ + 4 * x₃ = 72) (h_rel : x₁ = 3 * x₂) :
  x₁^2 + x₂^2 + x₃^2 ≥ 347.04 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1973_197379


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1973_197305

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c) ≥ 343 :=
sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c) = 343 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1973_197305


namespace NUMINAMATH_CALUDE_repair_shop_earnings_l1973_197302

/-- Calculates the total earnings for a repair shop after applying discounts and taxes. -/
def totalEarnings (
  phoneRepairCost : ℚ)
  (laptopRepairCost : ℚ)
  (computerRepairCost : ℚ)
  (tabletRepairCost : ℚ)
  (smartwatchRepairCost : ℚ)
  (phoneRepairs : ℕ)
  (laptopRepairs : ℕ)
  (computerRepairs : ℕ)
  (tabletRepairs : ℕ)
  (smartwatchRepairs : ℕ)
  (computerRepairDiscount : ℚ)
  (salesTaxRate : ℚ) : ℚ :=
  sorry

theorem repair_shop_earnings :
  totalEarnings 11 15 18 12 8 9 5 4 6 8 (1/10) (1/20) = 393.54 := by
  sorry

end NUMINAMATH_CALUDE_repair_shop_earnings_l1973_197302


namespace NUMINAMATH_CALUDE_triangles_in_circle_impossible_l1973_197339

theorem triangles_in_circle_impossible :
  ∀ (A₁ A₂ : ℝ), A₁ > 1 → A₂ > 1 → A₁ + A₂ > π :=
sorry

end NUMINAMATH_CALUDE_triangles_in_circle_impossible_l1973_197339


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1973_197381

theorem complex_modulus_problem (z : ℂ) : z = (-1 + I) / (1 + I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1973_197381


namespace NUMINAMATH_CALUDE_family_can_purchase_in_fourth_month_l1973_197350

/-- Represents the family's financial situation and purchase plan -/
structure Family where
  monthly_income : ℕ
  monthly_expenses : ℕ
  initial_savings : ℕ
  furniture_cost : ℕ

/-- Calculates the month when the family can make the purchase -/
def purchase_month (f : Family) : ℕ :=
  let monthly_savings := f.monthly_income - f.monthly_expenses
  let additional_required := f.furniture_cost - f.initial_savings
  (additional_required + monthly_savings - 1) / monthly_savings + 1

/-- Theorem stating that the family can make the purchase in the 4th month -/
theorem family_can_purchase_in_fourth_month (f : Family) 
  (h1 : f.monthly_income = 150000)
  (h2 : f.monthly_expenses = 115000)
  (h3 : f.initial_savings = 45000)
  (h4 : f.furniture_cost = 127000) :
  purchase_month f = 4 := by
  sorry

#eval purchase_month { 
  monthly_income := 150000, 
  monthly_expenses := 115000, 
  initial_savings := 45000, 
  furniture_cost := 127000 
}

end NUMINAMATH_CALUDE_family_can_purchase_in_fourth_month_l1973_197350


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l1973_197331

/-- The minimum squared distance between a point on y = x^2 + 3ln(x) and a point on y = x + 2 -/
theorem min_distance_between_curves : ∀ (a b c d : ℝ),
  b = a^2 + 3 * Real.log a →  -- P(a,b) is on y = x^2 + 3ln(x)
  d = c + 2 →                 -- Q(c,d) is on y = x + 2
  (∀ x y z w : ℝ, 
    y = x^2 + 3 * Real.log x → 
    w = z + 2 → 
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) →
  (a - c)^2 + (b - d)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l1973_197331


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1973_197366

theorem reciprocal_of_negative_fraction (n : ℕ) (h : n ≠ 0) :
  ((-1 : ℚ) / n)⁻¹ = -n := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1973_197366


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1973_197321

theorem partial_fraction_decomposition :
  ∃! (A B : ℚ), ∀ (x : ℚ), x ≠ 6 ∧ x ≠ -3 →
    (4 * x - 3) / (x^2 - 3 * x - 18) = A / (x - 6) + B / (x + 3) ∧
    A = 7/3 ∧ B = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1973_197321


namespace NUMINAMATH_CALUDE_cashback_strategies_reduce_losses_l1973_197383

/-- Represents a bank's cashback program -/
structure CashbackProgram where
  name : String
  maxCashbackPercentage : Float
  monthlyCapExists : Bool
  variableRate : Bool
  nonMonetaryRewards : Bool

/-- Represents a customer's behavior -/
structure CustomerBehavior where
  financialLiteracy : Float
  prefersHighCashbackCategories : Bool

/-- Calculates the profitability of a cashback program -/
def calculateProfitability (program : CashbackProgram) (customer : CustomerBehavior) : Float :=
  sorry

/-- Theorem: Implementing certain cashback strategies can reduce potential losses for banks -/
theorem cashback_strategies_reduce_losses 
  (program : CashbackProgram) 
  (customer : CustomerBehavior) :
  (program.monthlyCapExists ∨ program.variableRate ∨ program.nonMonetaryRewards) →
  (customer.financialLiteracy > 0.8 ∧ customer.prefersHighCashbackCategories) →
  calculateProfitability program customer > 0 := by
  sorry

#check cashback_strategies_reduce_losses

end NUMINAMATH_CALUDE_cashback_strategies_reduce_losses_l1973_197383


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_l1973_197390

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits in a 24-hour format digital watch display is 24 -/
theorem largest_sum_of_digits_24hour : 
  ∀ t : Time24, sumOfDigitsTime24 t ≤ 24 ∧ 
  ∃ t' : Time24, sumOfDigitsTime24 t' = 24 := by
  sorry

#check largest_sum_of_digits_24hour

end NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_l1973_197390


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_l1973_197376

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1

theorem tangent_line_at_pi :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y - f π = m * (x - π)) ↔ (x * Real.exp π + y - 1 - π * Real.exp π = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_pi_l1973_197376


namespace NUMINAMATH_CALUDE_both_normal_l1973_197312

-- Define a type for people
inductive Person : Type
| MrA : Person
| MrsA : Person

-- Define what it means to be normal
def normal (p : Person) : Prop := True

-- Define the statement made by each person
def statement (p : Person) : Prop :=
  match p with
  | Person.MrA => normal Person.MrsA
  | Person.MrsA => normal Person.MrA

-- Theorem: There exists a consistent interpretation where both are normal
theorem both_normal :
  ∃ (interp : Person → Prop),
    (∀ p, interp p ↔ normal p) ∧
    (∀ p, interp p → statement p) :=
sorry

end NUMINAMATH_CALUDE_both_normal_l1973_197312


namespace NUMINAMATH_CALUDE_arithmetic_relations_l1973_197318

theorem arithmetic_relations : 
  (10 * 100 = 1000) ∧ 
  (10 * 1000 = 10000) ∧ 
  (10000 / 100 = 100) ∧ 
  (1000 / 10 = 100) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_relations_l1973_197318


namespace NUMINAMATH_CALUDE_expected_games_specific_scenario_l1973_197337

/-- Represents a table tennis game between two players -/
structure TableTennisGame where
  probAWins : ℝ
  aheadBy : ℕ

/-- Calculates the expected number of games in a table tennis match -/
def expectedGames (game : TableTennisGame) : ℝ :=
  sorry

/-- Theorem stating that the expected number of games in the specific scenario is 18/5 -/
theorem expected_games_specific_scenario :
  let game : TableTennisGame := ⟨2/3, 2⟩
  expectedGames game = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_specific_scenario_l1973_197337


namespace NUMINAMATH_CALUDE_event3_mutually_exclusive_l1973_197332

-- Define the set of numbers
def NumberSet : Set Nat := {n : Nat | 1 ≤ n ∧ n ≤ 9}

-- Define the property of being even
def IsEven (n : Nat) : Prop := ∃ k : Nat, n = 2 * k

-- Define the property of being odd
def IsOdd (n : Nat) : Prop := ∃ k : Nat, n = 2 * k + 1

-- Define the events
def Event1 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ ((IsEven a ∧ IsOdd b) ∨ (IsOdd a ∧ IsEven b))

def Event2 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsOdd a ∧ IsOdd b)

def Event3 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsEven a ∧ IsEven b)

def Event4 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsEven a ∨ IsEven b)

-- Theorem statement
theorem event3_mutually_exclusive :
  ∀ a b : Nat,
    (Event3 a b → ¬Event1 a b) ∧
    (Event3 a b → ¬Event2 a b) ∧
    (Event3 a b → ¬Event4 a b) :=
sorry

end NUMINAMATH_CALUDE_event3_mutually_exclusive_l1973_197332


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1973_197313

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ+, a n * b n = 2 * n^2 - n) →
  5 * a 4 = 7 * a 3 →
  a 1 + b 1 = 2 →
  a 9 + b 10 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1973_197313


namespace NUMINAMATH_CALUDE_power_series_expansion_of_exp_l1973_197343

open Real

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the n-th term of the power series
noncomputable def power_series_term (a : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  (log a)^n / (Nat.factorial n : ℝ) * x^n

-- Theorem statement
theorem power_series_expansion_of_exp (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, f a x = ∑' n, power_series_term a n x :=
sorry

end NUMINAMATH_CALUDE_power_series_expansion_of_exp_l1973_197343


namespace NUMINAMATH_CALUDE_saline_mixture_concentration_l1973_197369

/-- Proves that mixing 3.6L of 1% saline and 1.4L of 9% saline results in 5L of 3.24% saline -/
theorem saline_mixture_concentration :
  let vol_1_percent : ℝ := 3.6
  let vol_9_percent : ℝ := 1.4
  let total_volume : ℝ := 5
  let concentration_1_percent : ℝ := 0.01
  let concentration_9_percent : ℝ := 0.09
  let resulting_concentration : ℝ := (vol_1_percent * concentration_1_percent + 
                                      vol_9_percent * concentration_9_percent) / total_volume
  resulting_concentration = 0.0324 := by
sorry

#eval (3.6 * 0.01 + 1.4 * 0.09) / 5

end NUMINAMATH_CALUDE_saline_mixture_concentration_l1973_197369


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1973_197384

theorem sphere_surface_area (r : ℝ) (d : ℝ) (h1 : r = 1) (h2 : d = Real.sqrt 3) :
  4 * Real.pi * (r^2 + d^2) = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1973_197384


namespace NUMINAMATH_CALUDE_quadratic_standard_form_l1973_197300

theorem quadratic_standard_form :
  ∀ x : ℝ, (x + 3) * (2 * x - 1) = -4 ↔ 2 * x^2 + 5 * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_standard_form_l1973_197300


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1973_197306

/-- Given two distinct real numbers k and b, prove that the x-coordinate of the 
    intersection point of the lines y = kx + b and y = bx + k is 1. -/
theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) : 
  ∃ x : ℝ, x = 1 ∧ kx + b = bx + k := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1973_197306


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1973_197336

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1973_197336


namespace NUMINAMATH_CALUDE_asymptote_slope_l1973_197370

-- Define the hyperbola parameters
def m : ℝ := 2

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / (m^2 + 12) - y^2 / (5*m - 1) = 1

-- Define the length of the real axis
def real_axis_length : ℝ := 8

-- Theorem statement
theorem asymptote_slope :
  hyperbola x y ∧ real_axis_length = 8 →
  ∃ (k : ℝ), k = 3/4 ∧ (y = k*x ∨ y = -k*x) :=
sorry

end NUMINAMATH_CALUDE_asymptote_slope_l1973_197370


namespace NUMINAMATH_CALUDE_scalene_triangle_angle_difference_l1973_197341

/-- A scalene triangle with one angle of 80 degrees can have a difference of 80 degrees between its other two angles. -/
theorem scalene_triangle_angle_difference : ∃ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- angles are positive
  a + b + c = 180 ∧  -- sum of angles in a triangle is 180°
  a = 80 ∧  -- one angle is 80°
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧  -- all angles are different (scalene)
  |b - c| = 80  -- difference between other two angles is 80°
:= by sorry

end NUMINAMATH_CALUDE_scalene_triangle_angle_difference_l1973_197341


namespace NUMINAMATH_CALUDE_arithmetic_sequence_669th_term_l1973_197389

/-- For an arithmetic sequence with first term 1 and common difference 3,
    the 669th term is 2005. -/
theorem arithmetic_sequence_669th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 1) → 
    (∀ n : ℕ, a (n + 1) - a n = 3) → 
    (a 669 = 2005) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_669th_term_l1973_197389


namespace NUMINAMATH_CALUDE_total_rectangles_is_eighteen_l1973_197303

/-- Represents a rectangle in the figure -/
structure Rectangle where
  size : Nat

/-- Represents the figure composed of rectangles -/
structure Figure where
  big_rectangle : Rectangle
  small_rectangles : Finset Rectangle
  middle_rectangles : Finset Rectangle

/-- Counts the total number of rectangles in the figure -/
def count_rectangles (f : Figure) : Nat :=
  1 + f.small_rectangles.card + f.middle_rectangles.card

/-- The theorem stating that the total number of rectangles is 18 -/
theorem total_rectangles_is_eighteen (f : Figure) 
  (h1 : f.big_rectangle.size = 1)
  (h2 : f.small_rectangles.card = 6)
  (h3 : f.middle_rectangles.card = 11) : 
  count_rectangles f = 18 := by
  sorry

#check total_rectangles_is_eighteen

end NUMINAMATH_CALUDE_total_rectangles_is_eighteen_l1973_197303


namespace NUMINAMATH_CALUDE_max_sum_exp_l1973_197398

theorem max_sum_exp (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) :
  ∃ (M : ℝ), M = 4 * Real.exp 1 ∧ ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 ≤ 4 →
    Real.exp x + Real.exp y + Real.exp z + Real.exp w ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_exp_l1973_197398


namespace NUMINAMATH_CALUDE_min_sum_squares_groups_l1973_197353

def S : Finset Int := {-9, -8, -4, -1, 1, 5, 7, 10}

theorem min_sum_squares_groups (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  ∀ (x y : Int), x = (p + q + r + s)^2 + (t + u + v + w)^2 → x ≥ 1 ∧ (x = 1 → y = 1) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_groups_l1973_197353


namespace NUMINAMATH_CALUDE_tan_double_angle_l1973_197396

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1973_197396


namespace NUMINAMATH_CALUDE_fraction_product_l1973_197368

theorem fraction_product : (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1973_197368


namespace NUMINAMATH_CALUDE_quincy_age_l1973_197364

/-- Given the ages of several people and their relationships, calculate Quincy's age -/
theorem quincy_age (kiarra bea job figaro quincy : ℝ) : 
  kiarra = 2 * bea →
  job = 3 * bea →
  figaro = job + 7 →
  kiarra = 30 →
  quincy = (job + figaro) / 2 →
  quincy = 48.5 := by
sorry

end NUMINAMATH_CALUDE_quincy_age_l1973_197364


namespace NUMINAMATH_CALUDE_f_properties_l1973_197377

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then 2 / x - 1 
  else if x < 0 then 2 / (-x) - 1
  else 1  -- f(0) is defined as 1 to make f continuous at 0

-- State the properties of f
theorem f_properties :
  -- f is an even function
  (∀ x, f (-x) = f x) ∧
  -- f(-1) = 1
  (f (-1) = 1) ∧
  -- f is decreasing on (0, +∞)
  (∀ x y, 0 < x → x < y → f y < f x) ∧
  -- For x < 0, f(x) = 2/(-x) - 1
  (∀ x, x < 0 → f x = 2 / (-x) - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1973_197377


namespace NUMINAMATH_CALUDE_complex_division_result_l1973_197363

theorem complex_division_result : (1 + 2*I : ℂ) / I = 2 - I := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l1973_197363


namespace NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l1973_197329

/-- The number of walnut trees planted in a park -/
def trees_planted (initial_trees final_trees : ℕ) : ℕ :=
  final_trees - initial_trees

/-- Theorem: The number of trees planted is the difference between the final and initial number of trees -/
theorem walnut_trees_planted (initial_trees final_trees : ℕ) 
  (h : initial_trees ≤ final_trees) :
  trees_planted initial_trees final_trees = final_trees - initial_trees :=
by sorry

/-- The specific case for the park problem -/
theorem park_walnut_trees : trees_planted 22 77 = 55 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l1973_197329


namespace NUMINAMATH_CALUDE_problem_solution_l1973_197349

theorem problem_solution (x : ℝ) 
  (h : x - Real.sqrt (x^2 - 4) + 1 / (x + Real.sqrt (x^2 - 4)) = 10) :
  x^2 - Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1973_197349


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l1973_197327

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 60) 
  (h2 : Nat.gcd a b = 12) : 
  a * b = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l1973_197327


namespace NUMINAMATH_CALUDE_sqrt2_similarity_l1973_197344

-- Define similarity for quadratic surds
def similar_quadratic_surds (a b : ℝ) : Prop :=
  ∃ (r : ℚ), r ≠ 0 ∧ a = r * b

-- Theorem statement
theorem sqrt2_similarity (r : ℚ) (h : r ≠ 0) :
  similar_quadratic_surds (r * Real.sqrt 2) (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_sqrt2_similarity_l1973_197344


namespace NUMINAMATH_CALUDE_cube_edge_length_l1973_197325

theorem cube_edge_length (V : ℝ) (h : V = 4 * Real.pi / 3) :
  ∃ (a : ℝ), a > 0 ∧ a = 2 * Real.sqrt 3 / 3 ∧
  V = 4 * Real.pi * (3 * a^2 / 4) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1973_197325


namespace NUMINAMATH_CALUDE_piggy_bank_equality_days_l1973_197395

def minjoo_initial : ℕ := 12000
def siwoo_initial : ℕ := 4000
def minjoo_daily : ℕ := 300
def siwoo_daily : ℕ := 500

theorem piggy_bank_equality_days : 
  ∃ d : ℕ, d = 40 ∧ 
  minjoo_initial + d * minjoo_daily = siwoo_initial + d * siwoo_daily :=
sorry

end NUMINAMATH_CALUDE_piggy_bank_equality_days_l1973_197395


namespace NUMINAMATH_CALUDE_subtraction_and_division_l1973_197392

theorem subtraction_and_division : ((-120) - (-60)) / (-30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_division_l1973_197392


namespace NUMINAMATH_CALUDE_triangle_inequality_l1973_197323

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1973_197323


namespace NUMINAMATH_CALUDE_work_completion_time_l1973_197308

/-- Given:
  * Mahesh can complete the entire work in 45 days
  * Mahesh works for 20 days
  * Rajesh finishes the remaining work in 30 days
  Prove that Y will take 54 days to complete the work -/
theorem work_completion_time (mahesh_full_time rajesh_completion_time mahesh_work_time : ℕ)
  (h1 : mahesh_full_time = 45)
  (h2 : mahesh_work_time = 20)
  (h3 : rajesh_completion_time = 30) :
  54 = (mahesh_full_time * rajesh_completion_time) / (rajesh_completion_time - mahesh_work_time) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1973_197308


namespace NUMINAMATH_CALUDE_cube_minimizes_edge_sum_squares_l1973_197373

/-- A parallelepiped with edges a, b, c and volume V -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  V : ℝ
  volume_eq : a * b * c = V
  positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sum of squares of edges meeting at one vertex -/
def edge_sum_squares (p : Parallelepiped) : ℝ := p.a^2 + p.b^2 + p.c^2

/-- Theorem: The cube minimizes the sum of squares of edges among parallelepipeds of equal volume -/
theorem cube_minimizes_edge_sum_squares (V : ℝ) (hV : 0 < V) :
  ∀ p : Parallelepiped, p.V = V →
  edge_sum_squares p ≥ 3 * V^(2/3) ∧
  (edge_sum_squares p = 3 * V^(2/3) ↔ p.a = p.b ∧ p.b = p.c) :=
sorry


end NUMINAMATH_CALUDE_cube_minimizes_edge_sum_squares_l1973_197373


namespace NUMINAMATH_CALUDE_cube_of_product_equality_l1973_197372

theorem cube_of_product_equality (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_equality_l1973_197372


namespace NUMINAMATH_CALUDE_no_fourfold_digit_move_l1973_197361

theorem no_fourfold_digit_move :
  ∀ (N : ℕ), ∀ (a : ℕ), ∀ (n : ℕ), ∀ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) →
    (x < 10^n) →
    (N = a * 10^n + x) →
    (10 * x + a ≠ 4 * N) :=
by sorry

end NUMINAMATH_CALUDE_no_fourfold_digit_move_l1973_197361


namespace NUMINAMATH_CALUDE_more_women_than_men_l1973_197346

/-- Proves that in a group of 15 people where the ratio of men to women is 0.5, there are 5 more women than men. -/
theorem more_women_than_men (total : ℕ) (ratio : ℚ) (men : ℕ) (women : ℕ) : 
  total = 15 → 
  ratio = 1/2 → 
  men + women = total → 
  (men : ℚ) / (women : ℚ) = ratio → 
  women - men = 5 := by
sorry

end NUMINAMATH_CALUDE_more_women_than_men_l1973_197346


namespace NUMINAMATH_CALUDE_banana_popsicles_count_l1973_197351

theorem banana_popsicles_count (grape_count cherry_count total_count : ℕ) 
  (h1 : grape_count = 2)
  (h2 : cherry_count = 13)
  (h3 : total_count = 17) :
  total_count - (grape_count + cherry_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_banana_popsicles_count_l1973_197351


namespace NUMINAMATH_CALUDE_prob_our_team_l1973_197340

/-- A sports team with boys, girls, and Alice -/
structure Team where
  total : ℕ
  boys : ℕ
  girls : ℕ
  has_alice : Bool

/-- Definition of our specific team -/
def our_team : Team :=
  { total := 12
  , boys := 7
  , girls := 5
  , has_alice := true
  }

/-- The probability of choosing two girls, one of whom is Alice -/
def prob_two_girls_with_alice (t : Team) : ℚ :=
  if t.has_alice then
    (t.girls - 1 : ℚ) / (t.total.choose 2 : ℚ)
  else
    0

/-- Theorem stating the probability for our specific team -/
theorem prob_our_team :
  prob_two_girls_with_alice our_team = 2 / 33 := by
  sorry


end NUMINAMATH_CALUDE_prob_our_team_l1973_197340


namespace NUMINAMATH_CALUDE_coins_after_five_hours_l1973_197358

/-- The number of coins in Tina's jar after five hours -/
def coins_in_jar (initial_deposit : ℕ) (second_third_deposit : ℕ) (fourth_deposit : ℕ) (withdrawal : ℕ) : ℕ :=
  initial_deposit + 2 * second_third_deposit + fourth_deposit - withdrawal

/-- Theorem stating the number of coins in the jar after five hours -/
theorem coins_after_five_hours :
  coins_in_jar 20 30 40 20 = 100 := by
  sorry

#eval coins_in_jar 20 30 40 20

end NUMINAMATH_CALUDE_coins_after_five_hours_l1973_197358


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1973_197304

theorem simplify_complex_fraction : 
  1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 7 - 2))) = 3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10) :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1973_197304


namespace NUMINAMATH_CALUDE_vasya_toy_choices_l1973_197394

/-- The number of different remote-controlled cars available -/
def num_cars : ℕ := 7

/-- The number of different construction sets available -/
def num_sets : ℕ := 5

/-- The total number of toys available -/
def total_toys : ℕ := num_cars + num_sets

/-- The number of toys Vasya can choose -/
def toys_to_choose : ℕ := 2

theorem vasya_toy_choices :
  Nat.choose total_toys toys_to_choose = 66 :=
sorry

end NUMINAMATH_CALUDE_vasya_toy_choices_l1973_197394


namespace NUMINAMATH_CALUDE_florist_roses_sold_l1973_197309

/-- Represents the number of roses sold by a florist -/
def roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) : ℕ :=
  initial + picked - final

/-- Theorem stating that the florist sold 16 roses -/
theorem florist_roses_sold :
  roses_sold 37 19 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_sold_l1973_197309
