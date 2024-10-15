import Mathlib

namespace NUMINAMATH_CALUDE_alice_walked_distance_l2150_215082

/-- The distance Alice walked in miles -/
def alice_distance (blocks_south : ℕ) (blocks_west : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_south + blocks_west : ℚ) * miles_per_block

/-- Theorem stating that Alice walked 3.25 miles -/
theorem alice_walked_distance :
  alice_distance 5 8 (1/4) = 3.25 := by sorry

end NUMINAMATH_CALUDE_alice_walked_distance_l2150_215082


namespace NUMINAMATH_CALUDE_consistency_comparison_l2150_215012

/-- Represents a player's performance in a basketball competition -/
structure PlayerPerformance where
  average_score : ℝ
  standard_deviation : ℝ

/-- Determines if a player performed more consistently than another -/
def more_consistent (p1 p2 : PlayerPerformance) : Prop :=
  p1.average_score = p2.average_score ∧ p1.standard_deviation < p2.standard_deviation

/-- Theorem: Given two players with the same average score, 
    the player with the smaller standard deviation performed more consistently -/
theorem consistency_comparison 
  (player_A player_B : PlayerPerformance) 
  (h_avg : player_A.average_score = player_B.average_score) 
  (h_std : player_B.standard_deviation < player_A.standard_deviation) : 
  more_consistent player_B player_A :=
sorry

end NUMINAMATH_CALUDE_consistency_comparison_l2150_215012


namespace NUMINAMATH_CALUDE_constant_angle_existence_l2150_215088

-- Define the circle C
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the line L
def Line (a b c : ℝ) := {P : ℝ × ℝ | a * P.1 + b * P.2 + c = 0}

-- Define the condition that L does not intersect C
def DoesNotIntersect (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) := C ∩ L = ∅

-- Define the circle with diameter MN
def CircleWithDiameter (M N : ℝ × ℝ) := 
  Circle ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Define the condition that CircleWithDiameter touches C but does not contain it
def TouchesButNotContains (C D : Set (ℝ × ℝ)) := 
  (∃ P, P ∈ C ∧ P ∈ D) ∧ (¬∃ P, P ∈ C ∧ P ∈ interior D)

-- Define the angle MPN
def Angle (M P N : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem constant_angle_existence 
  (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) (O : ℝ × ℝ) (r : ℝ) 
  (hC : C = Circle O r) (hL : ∃ a b c, L = Line a b c) 
  (hNotIntersect : DoesNotIntersect C L) :
  ∃ P : ℝ × ℝ, ∀ M N : ℝ × ℝ, 
    M ∈ L → N ∈ L → 
    TouchesButNotContains C (CircleWithDiameter M N) →
    ∃ θ : ℝ, Angle M P N = θ :=
sorry

end NUMINAMATH_CALUDE_constant_angle_existence_l2150_215088


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2150_215000

/-- A rectangle with a perimeter of 72 meters and a length-to-width ratio of 5:2 has a diagonal of 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2150_215000


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2150_215063

theorem rectangle_perimeter (length width : ℝ) 
  (h1 : length * width = 360)
  (h2 : (length + 10) * (width - 6) = 360) :
  2 * (length + width) = 76 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2150_215063


namespace NUMINAMATH_CALUDE_spheres_fit_in_box_l2150_215084

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the maximum number of spheres that can fit in a box using a specific packing method -/
noncomputable def maxSpheres (box : BoxDimensions) (sphereDiameter : ℝ) : ℕ :=
  sorry

/-- Theorem stating that 100,000 spheres of 4 cm diameter can fit in the given box -/
theorem spheres_fit_in_box :
  let box : BoxDimensions := ⟨200, 164, 146⟩
  let sphereDiameter : ℝ := 4
  maxSpheres box sphereDiameter ≥ 100000 := by
  sorry

end NUMINAMATH_CALUDE_spheres_fit_in_box_l2150_215084


namespace NUMINAMATH_CALUDE_jacket_price_before_tax_l2150_215099

def initial_amount : ℚ := 13.99
def shirt_price : ℚ := 12.14
def discount_rate : ℚ := 0.05
def additional_money : ℚ := 7.43
def tax_rate : ℚ := 0.10

def discounted_shirt_price : ℚ := shirt_price * (1 - discount_rate)
def money_left : ℚ := initial_amount + additional_money - discounted_shirt_price

theorem jacket_price_before_tax :
  ∃ (x : ℚ), x * (1 + tax_rate) = money_left ∧ x = 8.99 := by sorry

end NUMINAMATH_CALUDE_jacket_price_before_tax_l2150_215099


namespace NUMINAMATH_CALUDE_smallest_marble_collection_l2150_215080

theorem smallest_marble_collection (N : ℕ) : 
  N > 1 ∧ 
  N % 9 = 2 ∧ 
  N % 10 = 2 ∧ 
  N % 11 = 2 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 9 = 2 ∧ m % 10 = 2 ∧ m % 11 = 2 → m ≥ N) →
  N = 992 := by
sorry

end NUMINAMATH_CALUDE_smallest_marble_collection_l2150_215080


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2150_215042

theorem cos_sum_of_complex_exponentials (α β : ℝ) :
  Complex.exp (α * Complex.I) = (4 / 5 : ℂ) - (3 / 5 : ℂ) * Complex.I →
  Complex.exp (β * Complex.I) = (5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I →
  Real.cos (α + β) = -16 / 65 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2150_215042


namespace NUMINAMATH_CALUDE_milk_left_over_calculation_l2150_215072

/-- The amount of milk left over given the following conditions:
  - Total milk production is 24 cups per day
  - 80% of milk is consumed by Daisy's kids
  - 60% of remaining milk is used for cooking
  - 25% of remaining milk is given to neighbor
  - 6% of remaining milk is drunk by Daisy's husband
-/
def milk_left_over (total_milk : ℝ) (kids_consumption : ℝ) (cooking_usage : ℝ)
  (neighbor_share : ℝ) (husband_consumption : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption)
  let remaining_after_cooking := remaining_after_kids * (1 - cooking_usage)
  let remaining_after_neighbor := remaining_after_cooking * (1 - neighbor_share)
  remaining_after_neighbor * (1 - husband_consumption)

theorem milk_left_over_calculation :
  milk_left_over 24 0.8 0.6 0.25 0.06 = 1.3536 := by
  sorry

end NUMINAMATH_CALUDE_milk_left_over_calculation_l2150_215072


namespace NUMINAMATH_CALUDE_banana_distribution_l2150_215048

/-- Given three people with a total of 200 bananas, where one person has 40 more than another
    and the third person has 40 bananas, prove that the person with the least bananas has 60. -/
theorem banana_distribution (total : ℕ) (difference : ℕ) (donna_bananas : ℕ)
    (h_total : total = 200)
    (h_difference : difference = 40)
    (h_donna : donna_bananas = 40) :
    ∃ (lydia dawn : ℕ),
      lydia + dawn + donna_bananas = total ∧
      dawn = lydia + difference ∧
      lydia = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l2150_215048


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2150_215001

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2150_215001


namespace NUMINAMATH_CALUDE_abc_sum_mod_7_l2150_215068

theorem abc_sum_mod_7 (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 1 ∧ 
  (5 * c) % 7 = 2 ∧ 
  (6 * b) % 7 = (3 + b) % 7 → 
  (a + b + c) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_7_l2150_215068


namespace NUMINAMATH_CALUDE_min_value_f_inequality_abc_l2150_215026

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

-- Theorem for the inequality
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_inequality_abc_l2150_215026


namespace NUMINAMATH_CALUDE_work_completion_time_l2150_215006

/-- The time taken for three workers to complete a work together, given their individual completion times -/
theorem work_completion_time (tx ty tz : ℝ) (htx : tx = 20) (hty : ty = 40) (htz : tz = 30) :
  (1 / tx + 1 / ty + 1 / tz)⁻¹ = 120 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2150_215006


namespace NUMINAMATH_CALUDE_fraction_simplification_l2150_215043

theorem fraction_simplification :
  (240 : ℚ) / 18 * 9 / 135 * 7 / 4 = 14 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2150_215043


namespace NUMINAMATH_CALUDE_sine_cosine_roots_l2150_215062

theorem sine_cosine_roots (θ : Real) (m : Real) : 
  (∃ (x y : Real), x = Real.sin θ ∧ y = Real.cos θ ∧ 
   4 * x^2 + 2 * m * x + m = 0 ∧ 
   4 * y^2 + 2 * m * y + m = 0) →
  m = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_l2150_215062


namespace NUMINAMATH_CALUDE_twentyseven_binary_l2150_215015

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number in binary -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem twentyseven_binary :
  toBinary 27 = [true, true, false, true, true] :=
sorry

end NUMINAMATH_CALUDE_twentyseven_binary_l2150_215015


namespace NUMINAMATH_CALUDE_correlation_strength_theorem_l2150_215007

-- Define the correlation coefficient r
def correlation_coefficient (r : ℝ) : Prop := -1 < r ∧ r < 1

-- Define the strength of correlation
def correlation_strength (r : ℝ) : ℝ := |r|

-- Theorem stating the relationship between |r| and correlation strength
theorem correlation_strength_theorem (r : ℝ) (h : correlation_coefficient r) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', correlation_coefficient r' →
    correlation_strength r' < δ → correlation_strength r' < ε :=
sorry

end NUMINAMATH_CALUDE_correlation_strength_theorem_l2150_215007


namespace NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l2150_215040

theorem complex_modulus_equality_not_implies_square_equality :
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l2150_215040


namespace NUMINAMATH_CALUDE_brother_birthday_and_carlos_age_l2150_215076

def days_to_weekday (start_day : Nat) (days : Nat) : Nat :=
  (start_day + days) % 7

def years_from_days (days : Nat) : Nat :=
  days / 365

theorem brother_birthday_and_carlos_age 
  (start_day : Nat) 
  (carlos_age : Nat) 
  (days_until_brother_birthday : Nat) :
  start_day = 2 → 
  carlos_age = 7 → 
  days_until_brother_birthday = 2000 → 
  days_to_weekday start_day days_until_brother_birthday = 0 ∧ 
  years_from_days days_until_brother_birthday + carlos_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_brother_birthday_and_carlos_age_l2150_215076


namespace NUMINAMATH_CALUDE_unique_minimum_cost_plan_l2150_215097

/-- Represents a bus rental plan -/
structure BusRentalPlan where
  busA : ℕ  -- Number of Bus A
  busB : ℕ  -- Number of Bus B

/-- Checks if a bus rental plan is valid -/
def isValidPlan (p : BusRentalPlan) : Prop :=
  let totalPeople := 16 + 284
  let totalCapacity := 30 * p.busA + 42 * p.busB
  let totalCost := 300 * p.busA + 400 * p.busB
  let totalBuses := p.busA + p.busB
  totalCapacity ≥ totalPeople ∧
  totalCost ≤ 3100 ∧
  2 * totalBuses ≤ 16

/-- The set of all valid bus rental plans -/
def validPlans : Set BusRentalPlan :=
  {p : BusRentalPlan | isValidPlan p}

/-- The rental cost of a plan -/
def rentalCost (p : BusRentalPlan) : ℕ :=
  300 * p.busA + 400 * p.busB

theorem unique_minimum_cost_plan :
  ∃! p : BusRentalPlan, p ∈ validPlans ∧
    ∀ q ∈ validPlans, rentalCost p ≤ rentalCost q ∧
    rentalCost p = 2900 := by
  sorry

#check unique_minimum_cost_plan

end NUMINAMATH_CALUDE_unique_minimum_cost_plan_l2150_215097


namespace NUMINAMATH_CALUDE_hcf_problem_l2150_215089

theorem hcf_problem (a b : ℕ) (h1 : a = 588) (h2 : a ≥ b) 
  (h3 : ∃ (hcf : ℕ), Nat.lcm a b = hcf * 12 * 14) : 
  Nat.gcd a b = 7 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2150_215089


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2150_215035

theorem sum_of_solutions_is_zero :
  let f (x : ℝ) := (-12 * x) / (x^2 - 1) - (3 * x) / (x + 1) + 9 / (x - 1)
  ∃ (a b : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) ∧ a + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2150_215035


namespace NUMINAMATH_CALUDE_find_y_value_l2150_215046

theorem find_y_value (x y : ℝ) (h1 : 3 * (x^2 + x + 1) = y - 6) (h2 : x = -3) : y = 27 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l2150_215046


namespace NUMINAMATH_CALUDE_paddle_prices_and_cost_effective_solution_l2150_215059

/-- Represents the price of a pair of straight paddles in yuan -/
def straight_paddle_price : ℝ := sorry

/-- Represents the price of a pair of horizontal paddles in yuan -/
def horizontal_paddle_price : ℝ := sorry

/-- Cost of table tennis balls per pair of paddles -/
def ball_cost : ℝ := 20

/-- Total cost for 20 pairs of straight paddles and 15 pairs of horizontal paddles -/
def total_cost_35_pairs : ℝ := 9000

/-- Difference in cost between 10 pairs of horizontal paddles and 5 pairs of straight paddles -/
def cost_difference : ℝ := 1600

/-- Theorem stating the prices of paddles and the cost-effective solution -/
theorem paddle_prices_and_cost_effective_solution :
  (straight_paddle_price = 220 ∧ horizontal_paddle_price = 260) ∧
  (∀ m : ℕ, m ≤ 40 → m ≤ 3 * (40 - m) →
    m * (straight_paddle_price + ball_cost) + (40 - m) * (horizontal_paddle_price + ball_cost) ≥ 10000) ∧
  (30 * (straight_paddle_price + ball_cost) + 10 * (horizontal_paddle_price + ball_cost) = 10000) :=
by sorry

end NUMINAMATH_CALUDE_paddle_prices_and_cost_effective_solution_l2150_215059


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2150_215011

theorem complex_equation_solution :
  ∀ z : ℂ, z - 3 * I = 3 + z * I → z = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2150_215011


namespace NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l2150_215075

theorem polynomial_roots_and_factorization (m : ℤ) : 
  (∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 0 → 
    (∃ a b c d : ℤ, x = a ∨ x = b ∨ x = c ∨ x = d)) →
  (m = -10 ∧ 
   ∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 2 * (x + 1) * (x - 1) * (x + 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l2150_215075


namespace NUMINAMATH_CALUDE_smallest_two_digit_switch_add_five_l2150_215016

def digit_switch (n : ℕ) : ℕ := 
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_switch_add_five : 
  ∀ n : ℕ, 
    10 ≤ n → n < 100 → 
    (∀ m : ℕ, 10 ≤ m → m < n → digit_switch m + 5 ≠ 3 * m) → 
    digit_switch n + 5 = 3 * n → 
    n = 34 := by
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_switch_add_five_l2150_215016


namespace NUMINAMATH_CALUDE_k_range_for_inequality_l2150_215094

theorem k_range_for_inequality (k : ℝ) : 
  k ≠ 0 → 
  (k^2 * 1^2 - 6*k*1 + 8 ≥ 0) → 
  (k ≥ 4 ∨ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_inequality_l2150_215094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2150_215003

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 200) :
  4 * a 5 - 2 * a 3 = 80 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2150_215003


namespace NUMINAMATH_CALUDE_right_triangle_area_l2150_215009

theorem right_triangle_area (a c : ℝ) (h1 : a = 30) (h2 : c = 34) : 
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 240 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2150_215009


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2016_l2150_215024

theorem imaginary_unit_power_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2016_l2150_215024


namespace NUMINAMATH_CALUDE_vacant_seats_l2150_215049

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 700) (h2 : filled_percentage = 75 / 100) :
  (1 - filled_percentage) * total_seats = 175 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l2150_215049


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l2150_215093

theorem quadratic_solution_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 + (m-1)*x + 1 = 0) → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l2150_215093


namespace NUMINAMATH_CALUDE_haley_balls_count_l2150_215005

/-- Given that each bag can contain 4 balls and 9 bags will be used,
    prove that the number of balls Haley has is equal to 36. -/
theorem haley_balls_count (balls_per_bag : ℕ) (num_bags : ℕ) (h1 : balls_per_bag = 4) (h2 : num_bags = 9) :
  balls_per_bag * num_bags = 36 := by
  sorry

end NUMINAMATH_CALUDE_haley_balls_count_l2150_215005


namespace NUMINAMATH_CALUDE_del_oranges_per_day_l2150_215041

theorem del_oranges_per_day (total : ℕ) (juan : ℕ) (del_days : ℕ) 
  (h_total : total = 107)
  (h_juan : juan = 61)
  (h_del_days : del_days = 2) :
  (total - juan) / del_days = 23 := by
  sorry

end NUMINAMATH_CALUDE_del_oranges_per_day_l2150_215041


namespace NUMINAMATH_CALUDE_polynomial_equation_l2150_215023

/-- Given polynomials h and p such that h(x) + p(x) = 3x^2 - x + 4 
    and h(x) = x^4 - 5x^2 + x + 6, prove that p(x) = -x^4 + 8x^2 - 2x - 2 -/
theorem polynomial_equation (x : ℝ) (h p : ℝ → ℝ) 
    (h_p_sum : ∀ x, h x + p x = 3 * x^2 - x + 4)
    (h_def : ∀ x, h x = x^4 - 5 * x^2 + x + 6) :
  p x = -x^4 + 8 * x^2 - 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_l2150_215023


namespace NUMINAMATH_CALUDE_points_on_decreasing_line_l2150_215028

theorem points_on_decreasing_line (a₁ a₂ b₁ b₂ : ℝ) :
  a₁ ≠ a₂ →
  b₁ = -3 * a₁ + 4 →
  b₂ = -3 * a₂ + 4 →
  (a₁ - a₂) * (b₁ - b₂) < 0 :=
by sorry

end NUMINAMATH_CALUDE_points_on_decreasing_line_l2150_215028


namespace NUMINAMATH_CALUDE_alloy_mixture_l2150_215096

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 10

/-- The amount of the first alloy used (in kg) -/
def amount_1 : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def chromium_percent_new : ℝ := 10.6

/-- The amount of the second alloy used (in kg) -/
def amount_2 : ℝ := 35

theorem alloy_mixture :
  chromium_percent_1 * amount_1 / 100 + chromium_percent_2 * amount_2 / 100 =
  chromium_percent_new * (amount_1 + amount_2) / 100 :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_l2150_215096


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2150_215017

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67/144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2150_215017


namespace NUMINAMATH_CALUDE_sum_of_fractions_eq_9900_l2150_215060

/-- The sum of all fractions in lowest terms with denominator 3, 
    greater than 10 and less than 100 -/
def sum_of_fractions : ℚ :=
  (Finset.filter (fun n => n % 3 ≠ 0) (Finset.range 269)).sum (fun n => (n + 31 : ℚ) / 3)

/-- Theorem stating that the sum of fractions is equal to 9900 -/
theorem sum_of_fractions_eq_9900 : sum_of_fractions = 9900 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_eq_9900_l2150_215060


namespace NUMINAMATH_CALUDE_blue_fish_count_l2150_215070

theorem blue_fish_count (total_fish goldfish : ℕ) (h1 : total_fish = 22) (h2 : goldfish = 15) :
  total_fish - goldfish = 7 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_count_l2150_215070


namespace NUMINAMATH_CALUDE_conic_common_chords_l2150_215055

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Conic where
  equation : ℝ → ℝ → ℝ

-- Define the problem setup
def are_tangent (c1 c2 : Conic) (p1 p2 : Point) : Prop := sorry

def have_common_points (c1 c2 : Conic) (n : ℕ) : Prop := sorry

def line_through_points (p1 p2 : Point) : Line := sorry

def intersection_point (l1 l2 : Line) : Point := sorry

def common_chord (c1 c2 : Conic) : Line := sorry

def passes_through (l : Line) (p : Point) : Prop := sorry

-- State the theorem
theorem conic_common_chords 
  (Γ Γ₁ Γ₂ : Conic) 
  (A B C D : Point) :
  are_tangent Γ Γ₁ A B →
  are_tangent Γ Γ₂ C D →
  have_common_points Γ₁ Γ₂ 4 →
  ∃ (chord1 chord2 : Line),
    chord1 = common_chord Γ₁ Γ₂ ∧
    chord2 = common_chord Γ₁ Γ₂ ∧
    chord1 ≠ chord2 ∧
    passes_through chord1 (intersection_point (line_through_points A B) (line_through_points C D)) ∧
    passes_through chord2 (intersection_point (line_through_points A B) (line_through_points C D)) :=
by sorry

end NUMINAMATH_CALUDE_conic_common_chords_l2150_215055


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2150_215013

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2150_215013


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2150_215010

theorem inequality_system_solution :
  ∀ p : ℝ, (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2150_215010


namespace NUMINAMATH_CALUDE_medal_award_ways_eq_78_l2150_215037

/-- The number of ways to award medals in a race with American and non-American sprinters. -/
def medalAwardWays (totalSprinters : ℕ) (americanSprinters : ℕ) : ℕ :=
  let nonAmericanSprinters := totalSprinters - americanSprinters
  let noAmericanWins := nonAmericanSprinters * (nonAmericanSprinters - 1)
  let oneAmericanWins := 2 * americanSprinters * nonAmericanSprinters
  noAmericanWins + oneAmericanWins

/-- Theorem stating that the number of ways to award medals in the given scenario is 78. -/
theorem medal_award_ways_eq_78 :
  medalAwardWays 10 4 = 78 := by
  sorry

end NUMINAMATH_CALUDE_medal_award_ways_eq_78_l2150_215037


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2150_215091

/-- A geometric sequence with third term 5 and fifth term 45 has 5/3 as a possible second term -/
theorem geometric_sequence_second_term (a r : ℝ) : 
  a * r^2 = 5 → a * r^4 = 45 → a * r = 5/3 ∨ a * r = -5/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2150_215091


namespace NUMINAMATH_CALUDE_inequality_condition_l2150_215029

theorem inequality_condition (t : ℝ) : (t + 1) * (1 - |t|) > 0 ↔ t < 1 ∧ t ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2150_215029


namespace NUMINAMATH_CALUDE_corgi_dog_price_calculation_l2150_215051

/-- The price calculation for Corgi dogs with profit --/
theorem corgi_dog_price_calculation (original_price : ℝ) (profit_percentage : ℝ) (num_dogs : ℕ) :
  original_price = 1000 →
  profit_percentage = 30 →
  num_dogs = 2 →
  let profit_per_dog := original_price * (profit_percentage / 100)
  let selling_price_per_dog := original_price + profit_per_dog
  let total_cost := selling_price_per_dog * num_dogs
  total_cost = 2600 := by
  sorry


end NUMINAMATH_CALUDE_corgi_dog_price_calculation_l2150_215051


namespace NUMINAMATH_CALUDE_original_price_after_discount_l2150_215086

/-- Given a product with an unknown original price that becomes 50 yuan cheaper after a 20% discount, prove that its original price is 250 yuan. -/
theorem original_price_after_discount (price : ℝ) : 
  price * (1 - 0.2) = price - 50 → price = 250 := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discount_l2150_215086


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l2150_215074

/-- Represents the survey data --/
structure SurveyData where
  male_total : Nat
  male_opposing : Nat
  female_total : Nat
  female_opposing : Nat
  deriving Repr

/-- Represents different statistical methods --/
inductive StatMethod
  | Mean
  | Regression
  | IndependenceTest
  | Probability
  deriving Repr

/-- Determines the most appropriate method for analyzing the relationship
    between gender and judgment in the survey --/
def most_appropriate_method (data : SurveyData) : StatMethod :=
  StatMethod.IndependenceTest

/-- Theorem stating that the independence test is the most appropriate method
    for the given survey data --/
theorem independence_test_most_appropriate (data : SurveyData) 
    (h1 : data.male_total = 2548)
    (h2 : data.male_opposing = 1560)
    (h3 : data.female_total = 2452)
    (h4 : data.female_opposing = 1200) :
    most_appropriate_method data = StatMethod.IndependenceTest := by
  sorry


end NUMINAMATH_CALUDE_independence_test_most_appropriate_l2150_215074


namespace NUMINAMATH_CALUDE_system_solutions_l2150_215057

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (-x^7 / y)^(Real.log (-y)) = x^(2 * Real.log (x * y^2))

def equation2 (x y : ℝ) : Prop :=
  y^2 + 2*x*y - 3*x^2 + 12*x + 4*y = 0

-- Define the solution set
def solutions : Set (ℝ × ℝ) :=
  {(2, -2), (3, -9), ((Real.sqrt 17 - 1) / 2, (Real.sqrt 17 - 9) / 2)}

-- Theorem statement
theorem system_solutions :
  ∀ x y : ℝ, x ≠ 0 ∧ y < 0 →
    (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2150_215057


namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l2150_215065

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of the x-axis and y-axis -/
def XYAxes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_graph_is_axes : S = XYAxes := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l2150_215065


namespace NUMINAMATH_CALUDE_prob_not_beside_partner_is_four_fifths_l2150_215002

/-- The number of people to be seated -/
def total_people : ℕ := 5

/-- The number of couples -/
def num_couples : ℕ := 2

/-- The number of single people -/
def num_singles : ℕ := total_people - 2 * num_couples

/-- The total number of seating arrangements -/
def total_arrangements : ℕ := Nat.factorial total_people

/-- The number of arrangements where all couples are seated together -/
def couples_together_arrangements : ℕ := 
  (Nat.factorial (num_couples + num_singles)) * (2 ^ num_couples)

/-- The probability that at least one person is not beside their partner -/
def prob_not_beside_partner : ℚ := 
  1 - (couples_together_arrangements : ℚ) / (total_arrangements : ℚ)

theorem prob_not_beside_partner_is_four_fifths : 
  prob_not_beside_partner = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_not_beside_partner_is_four_fifths_l2150_215002


namespace NUMINAMATH_CALUDE_reflection_result_l2150_215021

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def C : ℝ × ℝ := (3, 3)

theorem reflection_result :
  (reflect_over_x_axis ∘ reflect_over_y_axis) C = (-3, -3) := by
sorry

end NUMINAMATH_CALUDE_reflection_result_l2150_215021


namespace NUMINAMATH_CALUDE_spherical_caps_ratio_l2150_215067

/-- 
Given a sphere of radius 1 cut by a plane into two spherical caps, 
if the combined surface area of the caps is 25% greater than the 
surface area of the original sphere, then the ratio of the surface 
areas of the larger cap to the smaller cap is (5 + 2√2) : (5 - 2√2).
-/
theorem spherical_caps_ratio (m₁ m₂ : ℝ) (ρ : ℝ) : 
  (0 < m₁) → (0 < m₂) → (0 < ρ) →
  (m₁ + m₂ = 2) →
  (2 * π * m₁ + π * ρ^2 + 2 * π * m₂ + π * ρ^2 = 5 * π) →
  (ρ^2 = 1 - (1 - m₁)^2) →
  (ρ^2 = 1 - (1 - m₂)^2) →
  ((2 * π * m₁ + π * ρ^2) / (2 * π * m₂ + π * ρ^2) = (5 + 2 * Real.sqrt 2) / (5 - 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_spherical_caps_ratio_l2150_215067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_negative_48_to_0_l2150_215036

def arithmeticSequenceSum (a l d : ℤ) : ℤ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_negative_48_to_0 :
  arithmeticSequenceSum (-48) 0 2 = -600 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_negative_48_to_0_l2150_215036


namespace NUMINAMATH_CALUDE_f_properties_l2150_215087

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let k : ℤ := ⌊(x + 1) / 2⌋
  (-1: ℝ) ^ k * Real.sqrt (1 - (x - 2 * ↑k) ^ 2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f (x + 4) = f x) ∧
  (∀ x : ℝ, f (x + 2) + f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2150_215087


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2150_215069

theorem complex_fraction_calculation : 
  27 * ((2 + 2/3) - (3 + 1/4)) / ((1 + 1/2) + (2 + 1/5)) = -(4 + 43/74) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2150_215069


namespace NUMINAMATH_CALUDE_max_radius_circle_in_quartic_region_l2150_215039

/-- The maximum radius of a circle touching the origin and lying in y ≥ x^4 -/
theorem max_radius_circle_in_quartic_region : ∃ r : ℝ,
  (∀ x y : ℝ, x^2 + (y - r)^2 = r^2 → y ≥ x^4) ∧
  (∀ s : ℝ, s > r → ∃ x y : ℝ, x^2 + (y - s)^2 = s^2 ∧ y < x^4) ∧
  r = (3 * Real.rpow 2 (1/3 : ℝ)) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_radius_circle_in_quartic_region_l2150_215039


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l2150_215058

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (((1 : ℝ) / 3) ^ 2 + ((1 : ℝ) / 4) ^ 2) / (((1 : ℝ) / 5) ^ 2 + ((1 : ℝ) / 6) ^ 2) = 25 * x / (53 * y) →
  Real.sqrt x / Real.sqrt y = 150 / 239 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l2150_215058


namespace NUMINAMATH_CALUDE_square_difference_fraction_l2150_215022

theorem square_difference_fraction (x y : ℚ) 
  (h1 : x + y = 9/17) (h2 : x - y = 1/51) : x^2 - y^2 = 1/289 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fraction_l2150_215022


namespace NUMINAMATH_CALUDE_three_digit_probability_l2150_215061

theorem three_digit_probability : 
  let S := Finset.Icc 30 800
  let three_digit := {n : ℕ | 100 ≤ n ∧ n ≤ 800}
  (S.filter (λ n => n ∈ three_digit)).card / S.card = 701 / 771 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_probability_l2150_215061


namespace NUMINAMATH_CALUDE_laptop_sale_price_l2150_215054

theorem laptop_sale_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 600 ∧ discount1 = 0.25 ∧ discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2)) / original_price = 0.675 := by
sorry

end NUMINAMATH_CALUDE_laptop_sale_price_l2150_215054


namespace NUMINAMATH_CALUDE_trader_shipment_cost_l2150_215079

/-- The amount needed for the next shipment of wares --/
def amount_needed (total_profit donation excess : ℕ) : ℕ :=
  total_profit / 2 + donation - excess

/-- Theorem stating the amount needed for the next shipment --/
theorem trader_shipment_cost (total_profit donation excess : ℕ)
  (h1 : total_profit = 960)
  (h2 : donation = 310)
  (h3 : excess = 180) :
  amount_needed total_profit donation excess = 610 := by
  sorry

#eval amount_needed 960 310 180

end NUMINAMATH_CALUDE_trader_shipment_cost_l2150_215079


namespace NUMINAMATH_CALUDE_max_a_value_l2150_215008

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, a * x^2 + 2 * a * x + 3 * a ≤ 1) →
  a ≤ 1/6 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2150_215008


namespace NUMINAMATH_CALUDE_expand_product_l2150_215052

theorem expand_product (x : ℝ) : (2 + x^2) * (3 - x^3 + x^5) = 6 + 3*x^2 - 2*x^3 + x^5 + x^7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2150_215052


namespace NUMINAMATH_CALUDE_range_of_f_l2150_215056

noncomputable def f (x : ℝ) : ℝ := (3 - 2^x) / (1 + 2^x)

theorem range_of_f :
  (∀ y ∈ Set.range f, -1 < y ∧ y < 3) ∧
  (∀ ε > 0, ∃ x₁ x₂, f x₁ < -1 + ε ∧ f x₂ > 3 - ε) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2150_215056


namespace NUMINAMATH_CALUDE_max_value_of_f_l2150_215038

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2150_215038


namespace NUMINAMATH_CALUDE_popping_corn_probability_l2150_215033

theorem popping_corn_probability (total : ℝ) (h_total : total > 0) :
  let white := (3 / 4 : ℝ) * total
  let yellow := (1 / 4 : ℝ) * total
  let white_pop_prob := (3 / 5 : ℝ)
  let yellow_pop_prob := (1 / 2 : ℝ)
  let white_popped := white * white_pop_prob
  let yellow_popped := yellow * yellow_pop_prob
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (18 / 23 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_popping_corn_probability_l2150_215033


namespace NUMINAMATH_CALUDE_concyclic_AQTP_l2150_215073

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (tangent_intersection : Circle → Point → Point → Point → Prop)
variable (concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem concyclic_AQTP 
  (Γ₁ Γ₂ : Circle) 
  (A B P Q T : Point) :
  intersect Γ₁ Γ₂ A B →
  on_circle P Γ₁ →
  on_circle Q Γ₂ →
  collinear P B Q →
  tangent_intersection Γ₂ P Q T →
  concyclic A Q T P :=
sorry

end NUMINAMATH_CALUDE_concyclic_AQTP_l2150_215073


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_exactly_two_zeros_l2150_215077

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

-- Theorem 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f 1 x_min ≤ f 1 x ∧ f 1 x_min = -1 :=
sorry

-- Theorem 2: Condition for exactly two zeros
theorem exactly_two_zeros (a : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ (z : ℝ), f a z = 0 → z = x ∨ z = y) ↔
  (1/2 ≤ a ∧ a < 1) ∨ (a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_exactly_two_zeros_l2150_215077


namespace NUMINAMATH_CALUDE_composite_polynomial_l2150_215081

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 6*n^2 + 12*n + 7 = a * b :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_composite_polynomial_l2150_215081


namespace NUMINAMATH_CALUDE_weather_forecast_inaccuracy_l2150_215032

theorem weather_forecast_inaccuracy (p_a p_b : ℝ) 
  (h_a : p_a = 0.9) 
  (h_b : p_b = 0.6) 
  (h_independent : True) -- Representing independence
  : (1 - p_a) * (1 - p_b) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_weather_forecast_inaccuracy_l2150_215032


namespace NUMINAMATH_CALUDE_square_sum_squares_l2150_215053

theorem square_sum_squares (n : ℕ) : n < 200 → (∃ k : ℕ, n^2 + (n+1)^2 = k^2) ↔ n = 3 ∨ n = 20 ∨ n = 119 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_squares_l2150_215053


namespace NUMINAMATH_CALUDE_needle_cylinder_height_gt_six_l2150_215020

/-- Represents the properties of a cylinder formed by needles piercing a skein of yarn -/
structure NeedleCylinder where
  num_needles : ℕ
  needle_radius : ℝ
  cylinder_radius : ℝ

/-- The theorem stating that the height of the cylinder must be greater than 6 -/
theorem needle_cylinder_height_gt_six (nc : NeedleCylinder)
  (h_num_needles : nc.num_needles = 72)
  (h_needle_radius : nc.needle_radius = 1)
  (h_cylinder_radius : nc.cylinder_radius = 6) :
  ∀ h : ℝ, h > 6 → 
    2 * π * nc.cylinder_radius^2 + 2 * π * nc.cylinder_radius * h > 
    2 * π * nc.num_needles * nc.needle_radius^2 + 2 * π * nc.cylinder_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_needle_cylinder_height_gt_six_l2150_215020


namespace NUMINAMATH_CALUDE_cuboid_breadth_l2150_215090

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The breadth of a cuboid with given length, height, and surface area -/
theorem cuboid_breadth (l h area : ℝ) (hl : l = 8) (hh : h = 12) (harea : area = 960) :
  ∃ w : ℝ, cuboidSurfaceArea l w h = area ∧ w = 19.2 := by sorry

end NUMINAMATH_CALUDE_cuboid_breadth_l2150_215090


namespace NUMINAMATH_CALUDE_soda_cost_l2150_215078

/-- Proves that the cost of each soda is $0.87 given the total cost and the cost of sandwiches -/
theorem soda_cost (total_cost : ℚ) (sandwich_cost : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) :
  total_cost = 8.38 →
  sandwich_cost = 2.45 →
  num_sandwiches = 2 →
  num_sodas = 4 →
  (total_cost - num_sandwiches * sandwich_cost) / num_sodas = 0.87 := by
sorry

#eval (8.38 - 2 * 2.45) / 4

end NUMINAMATH_CALUDE_soda_cost_l2150_215078


namespace NUMINAMATH_CALUDE_h_expansion_count_h_expansion_10_l2150_215047

/-- Definition of H expansion sequence -/
def h_expansion_seq (n : ℕ) : ℕ :=
  2^n + 1

/-- Theorem: The number of items after n H expansions is 2^n + 1 -/
theorem h_expansion_count (n : ℕ) :
  h_expansion_seq n = 2^n + 1 :=
by sorry

/-- Corollary: After 10 H expansions, the sequence has 1025 items -/
theorem h_expansion_10 :
  h_expansion_seq 10 = 1025 :=
by sorry

end NUMINAMATH_CALUDE_h_expansion_count_h_expansion_10_l2150_215047


namespace NUMINAMATH_CALUDE_worker_b_time_l2150_215044

/-- Given two workers A and B, where A takes 8 hours to complete a job,
    and together they take 4.8 hours, prove that B takes 12 hours alone. -/
theorem worker_b_time (time_a time_ab : ℝ) (time_a_pos : time_a > 0) (time_ab_pos : time_ab > 0)
  (h1 : time_a = 8) (h2 : time_ab = 4.8) : 
  ∃ time_b : ℝ, time_b > 0 ∧ 1 / time_a + 1 / time_b = 1 / time_ab ∧ time_b = 12 := by
  sorry

#check worker_b_time

end NUMINAMATH_CALUDE_worker_b_time_l2150_215044


namespace NUMINAMATH_CALUDE_square_pentagon_intersections_l2150_215019

/-- A square inscribed in a circle -/
structure InscribedSquare :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A regular pentagon inscribed in a circle -/
structure InscribedPentagon :=
  (vertices : Fin 5 → ℝ × ℝ)

/-- Predicate to check if two polygons share a vertex -/
def ShareVertex (s : InscribedSquare) (p : InscribedPentagon) : Prop :=
  ∃ (i : Fin 4) (j : Fin 5), s.vertices i = p.vertices j

/-- The number of intersections between two polygons -/
def NumIntersections (s : InscribedSquare) (p : InscribedPentagon) : ℕ := sorry

/-- Theorem stating that a square and a regular pentagon inscribed in the same circle,
    not sharing any vertices, intersect at exactly 8 points -/
theorem square_pentagon_intersections
  (s : InscribedSquare) (p : InscribedPentagon)
  (h : ¬ ShareVertex s p) :
  NumIntersections s p = 8 :=
sorry

end NUMINAMATH_CALUDE_square_pentagon_intersections_l2150_215019


namespace NUMINAMATH_CALUDE_max_sum_of_coeff_bound_l2150_215095

/-- A complex polynomial of degree 2 -/
def ComplexPoly (a b c : ℂ) : ℂ → ℂ := fun z ↦ a * z^2 + b * z + c

/-- The statement that |f(z)| ≤ 1 for all |z| ≤ 1 -/
def BoundedOnUnitDisk (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (f z) ≤ 1

/-- The main theorem -/
theorem max_sum_of_coeff_bound {a b c : ℂ} (h : BoundedOnUnitDisk (ComplexPoly a b c)) :
    Complex.abs a + Complex.abs b ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

#check max_sum_of_coeff_bound

end NUMINAMATH_CALUDE_max_sum_of_coeff_bound_l2150_215095


namespace NUMINAMATH_CALUDE_omega_range_l2150_215030

theorem omega_range (ω : ℝ) (h_pos : ω > 0) : 
  (∀ x ∈ Set.Ioo (2 * Real.pi / 3) (4 * Real.pi / 3), 
    Monotone (fun x => Real.cos (ω * x + Real.pi / 3))) → 
  1 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_omega_range_l2150_215030


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l2150_215014

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 5 * a + 3 * b ≤ 11) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 23 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l2150_215014


namespace NUMINAMATH_CALUDE_chord_length_l2150_215018

/-- Given a circle and a line intersecting at two points, 
    prove that the length of the chord formed by these intersection points is 9√5 / 5 -/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 4*y - 10 = 0) →  -- Circle equation
  (2*x - y + 1 = 0) →                -- Line equation
  ∃ (A B : ℝ × ℝ),                   -- Existence of intersection points A and B
    (A.1^2 + A.2^2 + 4*A.1 - 4*A.2 - 10 = 0) ∧ 
    (2*A.1 - A.2 + 1 = 0) ∧
    (B.1^2 + B.2^2 + 4*B.1 - 4*B.2 - 10 = 0) ∧ 
    (2*B.1 - B.2 + 1 = 0) ∧
    (A ≠ B) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (9*Real.sqrt 5 / 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2150_215018


namespace NUMINAMATH_CALUDE_pattern_equation_l2150_215034

theorem pattern_equation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_pattern_equation_l2150_215034


namespace NUMINAMATH_CALUDE_lemons_given_away_fraction_l2150_215045

def dozen : ℕ := 12

theorem lemons_given_away_fraction (lemons_left : ℕ) 
  (h1 : lemons_left = 9) : 
  (dozen - lemons_left : ℚ) / dozen = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lemons_given_away_fraction_l2150_215045


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2150_215027

-- Define the hyperbola C
def hyperbola_C : Set (ℝ × ℝ) := sorry

-- Define the foci of the hyperbola
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point P on both the hyperbola and the parabola
def P : ℝ × ℝ := sorry

-- Define the eccentricity of a hyperbola
def eccentricity (h : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity :
  P ∈ hyperbola_C ∧ 
  parabola P.1 P.2 ∧
  dot_product (vector_add (vector_sub P F₂) (vector_sub F₁ F₂)) 
              (vector_sub (vector_sub P F₂) (vector_sub F₁ F₂)) = 0 →
  eccentricity hyperbola_C = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2150_215027


namespace NUMINAMATH_CALUDE_jim_total_miles_l2150_215098

/-- Represents Jim's running schedule over 90 days -/
structure RunningSchedule where
  first_month : Nat  -- Miles per day for the first 30 days
  second_month : Nat -- Miles per day for the second 30 days
  third_month : Nat  -- Miles per day for the third 30 days

/-- Calculates the total miles run given a RunningSchedule -/
def total_miles (schedule : RunningSchedule) : Nat :=
  30 * schedule.first_month + 30 * schedule.second_month + 30 * schedule.third_month

/-- Theorem stating that Jim's total miles run is 1050 -/
theorem jim_total_miles :
  let jim_schedule : RunningSchedule := { first_month := 5, second_month := 10, third_month := 20 }
  total_miles jim_schedule = 1050 := by
  sorry


end NUMINAMATH_CALUDE_jim_total_miles_l2150_215098


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2150_215031

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5 : ℝ)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 20 = 0 → 
  d = -26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2150_215031


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2150_215025

/-- The perimeter of a semicircle with radius 4.8 cm is equal to π * 4.8 + 9.6 cm. -/
theorem semicircle_perimeter (π : ℝ) (h : π = Real.pi) :
  let r : ℝ := 4.8
  (π * r + 2 * r) = π * 4.8 + 9.6 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2150_215025


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_range_l2150_215066

theorem rectangular_prism_volume_range (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 9 → 
  a * b + b * c + a * c = 24 → 
  16 ≤ a * b * c ∧ a * b * c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_range_l2150_215066


namespace NUMINAMATH_CALUDE_speedster_convertibles_l2150_215083

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  (3 * speedsters = total) →  -- 1/3 of total inventory is Speedsters
  (5 * convertibles = 4 * speedsters) →  -- 4/5 of Speedsters are convertibles
  (total - speedsters = 30) →  -- 30 vehicles are not Speedsters
  convertibles = 12 := by
sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l2150_215083


namespace NUMINAMATH_CALUDE_sarahs_age_l2150_215050

/-- Given a person (Sarah) who is 18 years younger than her mother, 
    and the sum of their ages is 50 years, Sarah's age is 16 years. -/
theorem sarahs_age (s m : ℕ) : s = m - 18 ∧ s + m = 50 → s = 16 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_age_l2150_215050


namespace NUMINAMATH_CALUDE_carnation_count_l2150_215071

theorem carnation_count (vase_capacity : ℕ) (rose_count : ℕ) (vase_count : ℕ) :
  vase_capacity = 6 →
  rose_count = 47 →
  vase_count = 9 →
  vase_count * vase_capacity - rose_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_carnation_count_l2150_215071


namespace NUMINAMATH_CALUDE_five_students_three_communities_l2150_215085

/-- The number of ways to assign students to communities -/
def assign_students (n : ℕ) (k : ℕ) : ℕ :=
  -- Number of ways to assign n students to k communities
  -- with at least 1 student in each community
  sorry

/-- Theorem: 5 students assigned to 3 communities results in 150 ways -/
theorem five_students_three_communities :
  assign_students 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_communities_l2150_215085


namespace NUMINAMATH_CALUDE_logan_grocery_budget_l2150_215064

/-- Calculates the amount Logan can spend on groceries annually given his financial parameters. -/
def grocery_budget (current_income : ℕ) (income_increase : ℕ) (rent : ℕ) (gas : ℕ) (desired_savings : ℕ) : ℕ :=
  (current_income + income_increase) - (rent + gas + desired_savings)

/-- Theorem stating that Logan's grocery budget is $5,000 given his financial parameters. -/
theorem logan_grocery_budget :
  grocery_budget 65000 10000 20000 8000 42000 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_logan_grocery_budget_l2150_215064


namespace NUMINAMATH_CALUDE_correct_tense_for_ongoing_past_to_present_action_l2150_215004

/-- Represents different verb tenses -/
inductive VerbTense
  | simple_past
  | past_continuous
  | present_perfect_continuous
  | future_continuous

/-- Represents the characteristics of an action -/
structure ActionCharacteristics where
  ongoing : Bool
  started_in_past : Bool
  continues_to_present : Bool

/-- Theorem stating that for an action that is ongoing, started in the past, 
    and continues to the present, the correct tense is present perfect continuous -/
theorem correct_tense_for_ongoing_past_to_present_action 
  (action : ActionCharacteristics) 
  (h1 : action.ongoing = true) 
  (h2 : action.started_in_past = true) 
  (h3 : action.continues_to_present = true) : 
  VerbTense.present_perfect_continuous = 
    (match action with
      | ⟨true, true, true⟩ => VerbTense.present_perfect_continuous
      | _ => VerbTense.simple_past) :=
by sorry


end NUMINAMATH_CALUDE_correct_tense_for_ongoing_past_to_present_action_l2150_215004


namespace NUMINAMATH_CALUDE_final_amount_is_correct_l2150_215092

/-- The final amount owed after applying three consecutive 5% late charges to an initial bill of $200. -/
def final_amount : ℝ := 200 * (1.05)^3

/-- Theorem stating that the final amount owed is $231.525 -/
theorem final_amount_is_correct : final_amount = 231.525 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_is_correct_l2150_215092
