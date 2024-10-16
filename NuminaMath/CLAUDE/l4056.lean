import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_guessing_two_questions_correctly_l4056_405677

theorem probability_of_guessing_two_questions_correctly :
  let num_questions : ℕ := 2
  let options_per_question : ℕ := 4
  let prob_one_correct : ℚ := 1 / options_per_question
  prob_one_correct ^ num_questions = (1 : ℚ) / 16 := by sorry

end NUMINAMATH_CALUDE_probability_of_guessing_two_questions_correctly_l4056_405677


namespace NUMINAMATH_CALUDE_complement_A_B_l4056_405641

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-2) 4, y = |x + 1|}
def B : Set ℝ := Set.Ico 2 5

-- State the theorem
theorem complement_A_B : 
  (A \ B) = (Set.Ico 0 2 ∪ {5}) := by sorry

end NUMINAMATH_CALUDE_complement_A_B_l4056_405641


namespace NUMINAMATH_CALUDE_smallest_number_with_20_divisors_l4056_405687

/-- The number of divisors of a natural number n -/
def numDivisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A natural number n has exactly 20 divisors -/
def has20Divisors (n : ℕ) : Prop := numDivisors n = 20

theorem smallest_number_with_20_divisors :
  ∀ n : ℕ, has20Divisors n → n ≥ 240 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_20_divisors_l4056_405687


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l4056_405686

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l4056_405686


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4056_405601

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a_1 * a_5 = 16, then a_3 = ±4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) 
    (h_prod : a 1 * a 5 = 16) : 
  a 3 = 4 ∨ a 3 = -4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l4056_405601


namespace NUMINAMATH_CALUDE_divisibility_by_fifteen_l4056_405654

theorem divisibility_by_fifteen (n : ℕ) : n < 10 →
  (∃ k : ℕ, 80000 + 10000 * n + 945 = 15 * k) ↔ n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_fifteen_l4056_405654


namespace NUMINAMATH_CALUDE_max_circular_triples_14_players_l4056_405636

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_players : ℕ)
  (games_per_player : ℕ)
  (no_draws : Bool)

/-- Calculates the maximum number of circular triples in a tournament --/
def max_circular_triples (t : Tournament) : ℕ :=
  sorry

/-- Theorem: In a 14-player round-robin tournament where each player plays 13 games
    and there are no draws, the maximum number of circular triples is 112 --/
theorem max_circular_triples_14_players :
  let t : Tournament := ⟨14, 13, true⟩
  max_circular_triples t = 112 := by sorry

end NUMINAMATH_CALUDE_max_circular_triples_14_players_l4056_405636


namespace NUMINAMATH_CALUDE_employee_not_working_first_day_l4056_405617

def total_employees : ℕ := 6
def days : ℕ := 3
def employees_per_day : ℕ := 2

def schedule_probability (n m : ℕ) : ℚ := (n.choose m : ℚ) / (total_employees.choose employees_per_day : ℚ)

theorem employee_not_working_first_day :
  schedule_probability (total_employees - 1) employees_per_day = 2/3 :=
sorry

end NUMINAMATH_CALUDE_employee_not_working_first_day_l4056_405617


namespace NUMINAMATH_CALUDE_bret_nap_time_l4056_405625

/-- Calculates the remaining time for napping during a train ride -/
def remaining_nap_time (total_duration reading_time eating_time movie_time : ℕ) : ℕ :=
  total_duration - (reading_time + eating_time + movie_time)

/-- Theorem: Given Bret's 9-hour train ride and his activities, he has 3 hours left for napping -/
theorem bret_nap_time :
  remaining_nap_time 9 2 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bret_nap_time_l4056_405625


namespace NUMINAMATH_CALUDE_no_integer_solution_l4056_405699

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 8 * x + 3 * y^2 = 5 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l4056_405699


namespace NUMINAMATH_CALUDE_max_volume_cuboid_l4056_405628

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℕ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℕ :=
  c.length * c.width * c.height

/-- Theorem stating the maximum volume of a cuboid with given conditions -/
theorem max_volume_cuboid :
  ∃ (c : Cuboid), surfaceArea c = 150 ∧
    (∀ (c' : Cuboid), surfaceArea c' = 150 → volume c' ≤ volume c) ∧
    volume c = 125 := by
  sorry


end NUMINAMATH_CALUDE_max_volume_cuboid_l4056_405628


namespace NUMINAMATH_CALUDE_no_negative_exponents_l4056_405614

theorem no_negative_exponents (a b c d : ℤ) (h : (5 : ℝ)^a + (5 : ℝ)^b = (2 : ℝ)^c + (2 : ℝ)^d + 17) :
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d :=
by sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l4056_405614


namespace NUMINAMATH_CALUDE_expand_and_simplify_product_l4056_405678

theorem expand_and_simplify_product (x : ℝ) :
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_product_l4056_405678


namespace NUMINAMATH_CALUDE_stock_price_after_three_years_l4056_405684

theorem stock_price_after_three_years (initial_price : ℝ) :
  initial_price = 120 →
  let price_after_year1 := initial_price * 1.5
  let price_after_year2 := price_after_year1 * 0.7
  let price_after_year3 := price_after_year2 * 1.2
  price_after_year3 = 151.2 := by
sorry

end NUMINAMATH_CALUDE_stock_price_after_three_years_l4056_405684


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4056_405688

theorem sufficient_not_necessary :
  (∀ x : ℝ, x < Real.sqrt 2 → 2 * x < 3) ∧
  ¬(∀ x : ℝ, 2 * x < 3 → x < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4056_405688


namespace NUMINAMATH_CALUDE_tobacco_acreage_increase_l4056_405663

/-- Calculates the increase in tobacco acreage when changing crop ratios -/
theorem tobacco_acreage_increase (total_land : ℝ) (initial_ratio_tobacco : ℝ) 
  (initial_ratio_total : ℝ) (new_ratio_tobacco : ℝ) (new_ratio_total : ℝ) :
  total_land = 1350 ∧ 
  initial_ratio_tobacco = 2 ∧ 
  initial_ratio_total = 9 ∧ 
  new_ratio_tobacco = 5 ∧ 
  new_ratio_total = 9 →
  (new_ratio_tobacco / new_ratio_total - initial_ratio_tobacco / initial_ratio_total) * total_land = 450 :=
by sorry

end NUMINAMATH_CALUDE_tobacco_acreage_increase_l4056_405663


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l4056_405676

/-- Proves that in a 60-litre mixture, if adding 60 litres of water changes
    the milk-to-water ratio to 1:2, then the initial milk-to-water ratio was 2:1 -/
theorem initial_milk_water_ratio (m w : ℝ) : 
  m + w = 60 →  -- Total initial volume is 60 litres
  2 * m = w + 60 →  -- After adding 60 litres of water, milk:water = 1:2
  m / w = 2 / 1 :=  -- Initial ratio of milk to water is 2:1
by sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l4056_405676


namespace NUMINAMATH_CALUDE_total_distance_two_parts_l4056_405646

/-- Calculates the total distance traveled by a car with varying speeds -/
theorem total_distance_two_parts (v1 v2 t1 t2 D1 D2 : ℝ) :
  D1 = v1 * t1 →
  D2 = v2 * t2 →
  let D := D1 + D2
  D = v1 * t1 + v2 * t2 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_two_parts_l4056_405646


namespace NUMINAMATH_CALUDE_area_of_large_rectangle_l4056_405683

/-- The width of each smaller rectangle in feet -/
def small_rectangle_width : ℝ := 8

/-- The number of identical rectangles stacked vertically -/
def num_rectangles : ℕ := 3

/-- The length of each smaller rectangle in feet -/
def small_rectangle_length : ℝ := 2 * small_rectangle_width

/-- The width of the larger rectangle ABCD in feet -/
def large_rectangle_width : ℝ := small_rectangle_width

/-- The length of the larger rectangle ABCD in feet -/
def large_rectangle_length : ℝ := num_rectangles * small_rectangle_length

/-- The area of the larger rectangle ABCD in square feet -/
def large_rectangle_area : ℝ := large_rectangle_width * large_rectangle_length

theorem area_of_large_rectangle : large_rectangle_area = 384 := by
  sorry

end NUMINAMATH_CALUDE_area_of_large_rectangle_l4056_405683


namespace NUMINAMATH_CALUDE_problem_solution_l4056_405611

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^2 / y + y^2 / x + y = 95 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4056_405611


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l4056_405698

theorem binomial_expansion_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ = 1 ∧ a₁ + a₃ + a₅ = 122) :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l4056_405698


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4056_405612

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + a^n) + 1 / (1 + b^n)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4056_405612


namespace NUMINAMATH_CALUDE_minimum_coins_required_l4056_405647

/-- Represents the denominations of coins available -/
inductive Coin
  | Ten
  | Fifteen
  | Twenty

/-- Represents a passenger on the bus -/
structure Passenger where
  coins : List Coin

/-- Represents the bus system -/
structure BusSystem where
  passengers : List Passenger
  fare : Nat

/-- Function to calculate the value of a coin -/
def coinValue : Coin → Nat
  | Coin.Ten => 10
  | Coin.Fifteen => 15
  | Coin.Twenty => 20

/-- Function to calculate the total value of a list of coins -/
def totalValue (coins : List Coin) : Nat :=
  coins.foldl (fun acc coin => acc + coinValue coin) 0

/-- Predicate to check if a passenger can pay the fare and receive change -/
def canPayFare (p : Passenger) (fare : Nat) : Prop :=
  ∃ (payment : List Coin), payment.length > 0 ∧ totalValue payment ≥ fare

/-- The main theorem to prove -/
theorem minimum_coins_required (bs : BusSystem) (h1 : bs.passengers.length = 20) 
  (h2 : bs.fare = 5) (h3 : ∀ p ∈ bs.passengers, canPayFare p bs.fare) :
  (bs.passengers.map (fun p => p.coins.length)).sum ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_minimum_coins_required_l4056_405647


namespace NUMINAMATH_CALUDE_apples_in_good_condition_l4056_405618

-- Define the total number of apples
def total_apples : ℕ := 75

-- Define the percentage of rotten apples
def rotten_percentage : ℚ := 12 / 100

-- Define the number of apples in good condition
def good_apples : ℕ := 66

-- Theorem statement
theorem apples_in_good_condition :
  (total_apples : ℚ) * (1 - rotten_percentage) = good_apples := by
  sorry

end NUMINAMATH_CALUDE_apples_in_good_condition_l4056_405618


namespace NUMINAMATH_CALUDE_prime_power_composite_and_divisor_l4056_405680

theorem prime_power_composite_and_divisor (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  let q := (4^p - 1) / 3
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ q = a * b) ∧ (q ∣ 2^(q - 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_composite_and_divisor_l4056_405680


namespace NUMINAMATH_CALUDE_two_vans_needed_l4056_405652

/-- The number of vans needed for a field trip -/
def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

/-- Proof that 2 vans are needed for the field trip -/
theorem two_vans_needed : vans_needed 4 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_vans_needed_l4056_405652


namespace NUMINAMATH_CALUDE_reduced_price_calculation_l4056_405697

/-- Represents the price of oil in Rupees per kg -/
structure OilPrice where
  price : ℝ
  price_positive : price > 0

def reduction_percentage : ℝ := 0.30

def total_cost : ℝ := 700

def additional_quantity : ℝ := 3

theorem reduced_price_calculation (original_price : OilPrice) :
  let reduced_price := original_price.price * (1 - reduction_percentage)
  let original_quantity := total_cost / original_price.price
  let new_quantity := total_cost / reduced_price
  new_quantity = original_quantity + additional_quantity →
  reduced_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_reduced_price_calculation_l4056_405697


namespace NUMINAMATH_CALUDE_triangle_area_l4056_405681

theorem triangle_area (A B C : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25) (h3 : Real.cos B = Real.sin A) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b + c) / 2 * r = 525 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l4056_405681


namespace NUMINAMATH_CALUDE_power_of_two_floor_l4056_405659

theorem power_of_two_floor (n : ℕ) (h1 : n ≥ 4) 
  (h2 : ∃ k : ℕ, ⌊(2^n : ℝ) / n⌋ = 2^k) : 
  ∃ m : ℕ, n = 2^m :=
sorry

end NUMINAMATH_CALUDE_power_of_two_floor_l4056_405659


namespace NUMINAMATH_CALUDE_x_intercept_after_rotation_l4056_405658

/-- Given a line m with equation 2x - 3y + 30 = 0 in the coordinate plane,
    rotated 30° counterclockwise about the point (10, 10) to form line n,
    the x-coordinate of the x-intercept of line n is (20√3 + 20) / (2√3 + 3). -/
theorem x_intercept_after_rotation :
  let m : Set (ℝ × ℝ) := {(x, y) | 2 * x - 3 * y + 30 = 0}
  let center : ℝ × ℝ := (10, 10)
  let angle : ℝ := π / 6  -- 30° in radians
  let n : Set (ℝ × ℝ) := {(x, y) | ∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ m ∧
    x - 10 = (x₀ - 10) * Real.cos angle - (y₀ - 10) * Real.sin angle ∧
    y - 10 = (x₀ - 10) * Real.sin angle + (y₀ - 10) * Real.cos angle}
  let x_intercept : ℝ := (20 * Real.sqrt 3 + 20) / (2 * Real.sqrt 3 + 3)
  (0, x_intercept) ∈ n := by sorry

end NUMINAMATH_CALUDE_x_intercept_after_rotation_l4056_405658


namespace NUMINAMATH_CALUDE_fraction_comparison_l4056_405603

theorem fraction_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  ((1 + y) / x < 2) ∨ ((1 + x) / y < 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l4056_405603


namespace NUMINAMATH_CALUDE_task_force_combinations_l4056_405639

theorem task_force_combinations (independents greens : ℕ) 
  (h1 : independents = 10) (h2 : greens = 7) : 
  (Nat.choose independents 4) * (Nat.choose greens 3) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_task_force_combinations_l4056_405639


namespace NUMINAMATH_CALUDE_twentieth_decimal_of_35_36_l4056_405644

/-- The fraction we're considering -/
def f : ℚ := 35 / 36

/-- The nth decimal digit in the decimal expansion of a rational number -/
noncomputable def nthDecimalDigit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 20th decimal digit of 35/36 is 2 -/
theorem twentieth_decimal_of_35_36 : nthDecimalDigit f 20 = 2 := by sorry

end NUMINAMATH_CALUDE_twentieth_decimal_of_35_36_l4056_405644


namespace NUMINAMATH_CALUDE_root_in_interval_l4056_405606

-- Define the function
def f (x : ℝ) := x^3 - 2*x - 1

-- State the theorem
theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) →
  (∃ x ∈ Set.Ioo (3/2) 2, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l4056_405606


namespace NUMINAMATH_CALUDE_richmond_victoria_difference_l4056_405627

def richmond_population : ℕ := 3000
def beacon_population : ℕ := 500
def victoria_population : ℕ := 4 * beacon_population

theorem richmond_victoria_difference : 
  richmond_population - victoria_population = 1000 ∧ richmond_population > victoria_population := by
  sorry

end NUMINAMATH_CALUDE_richmond_victoria_difference_l4056_405627


namespace NUMINAMATH_CALUDE_joels_dads_age_l4056_405613

theorem joels_dads_age :
  ∀ (joel_current_age joel_future_age dads_current_age : ℕ),
    joel_current_age = 5 →
    joel_future_age = 27 →
    dads_current_age + (joel_future_age - joel_current_age) = 2 * joel_future_age →
    dads_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_joels_dads_age_l4056_405613


namespace NUMINAMATH_CALUDE_peaches_bought_is_seven_l4056_405638

/-- Represents the cost of fruits and the quantity purchased. -/
structure FruitPurchase where
  apple_cost : ℕ
  peach_cost : ℕ
  total_fruits : ℕ
  total_cost : ℕ

/-- Calculates the number of peaches bought given a FruitPurchase. -/
def peaches_bought (purchase : FruitPurchase) : ℕ :=
  let apple_count := purchase.total_fruits - (purchase.total_cost - purchase.apple_cost * purchase.total_fruits) / (purchase.peach_cost - purchase.apple_cost)
  purchase.total_fruits - apple_count

/-- Theorem stating that given the specific conditions, 7 peaches were bought. -/
theorem peaches_bought_is_seven : 
  ∀ (purchase : FruitPurchase), 
    purchase.apple_cost = 1000 → 
    purchase.peach_cost = 2000 → 
    purchase.total_fruits = 15 → 
    purchase.total_cost = 22000 → 
    peaches_bought purchase = 7 := by
  sorry


end NUMINAMATH_CALUDE_peaches_bought_is_seven_l4056_405638


namespace NUMINAMATH_CALUDE_triangle_angles_l4056_405672

theorem triangle_angles (a b c : ℝ) (h1 : a = 3) (h2 : b = Real.sqrt 8) (h3 : c = 2 + Real.sqrt 2) :
  ∃ (θ φ ψ : ℝ),
    Real.cos θ = (10 + Real.sqrt 2) / 18 ∧
    Real.cos φ = (11 - 4 * Real.sqrt 2) / (12 * Real.sqrt 2) ∧
    θ + φ + ψ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l4056_405672


namespace NUMINAMATH_CALUDE_acute_triangle_cotangent_sum_range_l4056_405673

theorem acute_triangle_cotangent_sum_range (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  b^2 - a^2 = a * c →  -- Given condition
  1 < 1 / Real.tan A + 1 / Real.tan B ∧ 
  1 / Real.tan A + 1 / Real.tan B < 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_cotangent_sum_range_l4056_405673


namespace NUMINAMATH_CALUDE_expression_value_l4056_405640

/-- The numerator of the expression -/
def numerator : ℤ := 20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1

/-- The denominator of the expression -/
def denominator : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20

/-- The main theorem stating that the expression equals -1 -/
theorem expression_value : (numerator : ℚ) / denominator = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4056_405640


namespace NUMINAMATH_CALUDE_proportionality_problem_l4056_405666

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^2,
    prove that x = 1/16 when z = 32, given that x = 4 when z = 8. -/
theorem proportionality_problem (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h₁ : x = k₁ * y^4)
    (h₂ : y * z^2 = k₂)
    (h₃ : x = 4 ∧ z = 8 → k₁ * k₂^4 = 67108864) :
    z = 32 → x = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_proportionality_problem_l4056_405666


namespace NUMINAMATH_CALUDE_twenty_numbers_arrangement_exists_l4056_405633

theorem twenty_numbers_arrangement_exists : ∃ (a b : ℝ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) := by
  sorry

end NUMINAMATH_CALUDE_twenty_numbers_arrangement_exists_l4056_405633


namespace NUMINAMATH_CALUDE_quadrilateral_area_product_is_square_quadrilateral_area_product_not_end_1988_l4056_405609

/-- Represents a convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- The area of the first triangle -/
  area1 : ℕ
  /-- The area of the second triangle -/
  area2 : ℕ
  /-- The area of the third triangle -/
  area3 : ℕ
  /-- The area of the fourth triangle -/
  area4 : ℕ

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem quadrilateral_area_product_is_square (q : ConvexQuadrilateral) :
  ∃ (n : ℕ), q.area1 * q.area2 * q.area3 * q.area4 = n * n := by
  sorry

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals cannot end in 1988 -/
theorem quadrilateral_area_product_not_end_1988 (q : ConvexQuadrilateral) :
  ¬(q.area1 * q.area2 * q.area3 * q.area4 % 10000 = 1988) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_product_is_square_quadrilateral_area_product_not_end_1988_l4056_405609


namespace NUMINAMATH_CALUDE_log_condition_l4056_405631

theorem log_condition (x : ℝ) : 
  (∀ x, Real.log (x + 1) < 0 → x < 0) ∧ 
  (∃ x, x < 0 ∧ Real.log (x + 1) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_log_condition_l4056_405631


namespace NUMINAMATH_CALUDE_range_of_a_l4056_405693

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4056_405693


namespace NUMINAMATH_CALUDE_regression_analysis_appropriate_for_height_weight_l4056_405624

/-- Represents a statistical analysis method -/
inductive AnalysisMethod
  | ResidualAnalysis
  | RegressionAnalysis
  | IsoplethBarChart
  | IndependenceTest

/-- Represents a variable in the context of statistical analysis -/
structure Variable where
  name : String

/-- Represents a relationship between two variables -/
structure Relationship where
  var1 : Variable
  var2 : Variable
  correlated : Bool

/-- Determines if a given analysis method is appropriate for analyzing a relationship between two variables -/
def is_appropriate_method (method : AnalysisMethod) (rel : Relationship) : Prop :=
  method = AnalysisMethod.RegressionAnalysis ∧ rel.correlated = true

/-- Main theorem: Regression analysis is the appropriate method for analyzing the relationship between height and weight -/
theorem regression_analysis_appropriate_for_height_weight :
  let height : Variable := ⟨"height"⟩
  let weight : Variable := ⟨"weight"⟩
  let height_weight_rel : Relationship := ⟨height, weight, true⟩
  is_appropriate_method AnalysisMethod.RegressionAnalysis height_weight_rel :=
by
  sorry


end NUMINAMATH_CALUDE_regression_analysis_appropriate_for_height_weight_l4056_405624


namespace NUMINAMATH_CALUDE_area_of_region_is_4pi_l4056_405643

-- Define the region
def region (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y = -1

-- Define the area of the region
noncomputable def area_of_region : ℝ := sorry

-- Theorem statement
theorem area_of_region_is_4pi :
  area_of_region = 4 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_region_is_4pi_l4056_405643


namespace NUMINAMATH_CALUDE_monotonic_sine_range_l4056_405690

/-- The function f(x) = 2sin(ωx) is monotonically increasing on [-π/3, π/4] iff 0 < ω ≤ 12/7 -/
theorem monotonic_sine_range (ω : ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), Monotone (fun x => 2 * Real.sin (ω * x))) ↔
  ω ≤ 12/7 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_sine_range_l4056_405690


namespace NUMINAMATH_CALUDE_abs_neg_two_plus_two_l4056_405664

theorem abs_neg_two_plus_two : |(-2 : ℤ)| + 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_plus_two_l4056_405664


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l4056_405602

theorem coefficient_x4_in_expansion : 
  let f : Polynomial ℚ := (X - 1)^2 * (X + 1)^5
  (f.coeff 4) = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l4056_405602


namespace NUMINAMATH_CALUDE_unique_m_for_direct_proportion_l4056_405682

/-- A function f(x) is a direct proportion function if it can be written as f(x) = kx for some non-zero constant k. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m+1)x + m^2 - 1 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 1) * x + m^2 - 1

/-- Theorem: The only value of m that makes f(m) a direct proportion function is 1 -/
theorem unique_m_for_direct_proportion :
  ∃! m : ℝ, IsDirectProportion (f m) ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_for_direct_proportion_l4056_405682


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l4056_405604

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (27 - 9*x - x^2 = 0) → 
  (∃ r s : ℝ, (27 - 9*r - r^2 = 0) ∧ (27 - 9*s - s^2 = 0) ∧ (r + s = 9)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l4056_405604


namespace NUMINAMATH_CALUDE_container_volume_ratio_l4056_405619

theorem container_volume_ratio :
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (3 / 4 : ℚ) * volume_container1 = (5 / 8 : ℚ) * volume_container2 →
  volume_container1 / volume_container2 = (5 / 6 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l4056_405619


namespace NUMINAMATH_CALUDE_high_school_ten_games_l4056_405661

/-- The number of teams in the conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team -/
def games_per_pair : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams.choose 2) * games_per_pair + num_teams * non_conference_games

theorem high_school_ten_games : total_games = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_games_l4056_405661


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l4056_405671

theorem sum_of_consecutive_integers (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (c = 7) → (a + b + c = 18) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l4056_405671


namespace NUMINAMATH_CALUDE_expression_value_l4056_405637

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4056_405637


namespace NUMINAMATH_CALUDE_possible_tile_counts_l4056_405662

/-- Represents the dimensions of a rectangular floor in terms of tiles -/
structure FloorDimensions where
  width : ℕ
  length : ℕ

/-- Calculates the number of red tiles on the floor -/
def redTiles (d : FloorDimensions) : ℕ := 2 * d.width + 2 * d.length - 4

/-- Calculates the number of white tiles on the floor -/
def whiteTiles (d : FloorDimensions) : ℕ := d.width * d.length - redTiles d

/-- Checks if the number of red and white tiles are equal -/
def equalRedWhite (d : FloorDimensions) : Prop := redTiles d = whiteTiles d

/-- The theorem stating the possible total number of tiles -/
theorem possible_tile_counts : 
  ∀ d : FloorDimensions, 
    equalRedWhite d → 
    d.width * d.length = 48 ∨ d.width * d.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_possible_tile_counts_l4056_405662


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4056_405630

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  -3 + 23*x - x^2 + 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4056_405630


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l4056_405610

/-- The surface area of a sphere circumscribing a right circular cone -/
theorem circumscribed_sphere_surface_area 
  (base_radius : ℝ) 
  (slant_height : ℝ) 
  (h1 : base_radius = Real.sqrt 3)
  (h2 : slant_height = 2) :
  ∃ (sphere_radius : ℝ), 
    4 * Real.pi * sphere_radius^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l4056_405610


namespace NUMINAMATH_CALUDE_counterexample_exists_l4056_405670

theorem counterexample_exists : ∃ (a b c : ℝ), 
  (a^2 + b^2) / (b^2 + c^2) = a/c ∧ a/b ≠ b/c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4056_405670


namespace NUMINAMATH_CALUDE_correct_average_calculation_l4056_405607

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 25 →
  correct_num = 35 →
  (n : ℚ) * initial_avg + (correct_num - incorrect_num) = n * 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l4056_405607


namespace NUMINAMATH_CALUDE_stationery_box_sheet_count_l4056_405675

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents the usage of a stationery box -/
structure Usage where
  sheetsPerLetter : ℕ
  usedAllEnvelopes : Bool
  usedAllSheets : Bool
  leftoverSheets : ℕ
  leftoverEnvelopes : ℕ

theorem stationery_box_sheet_count (box : StationeryBox) 
  (ann_usage : Usage) (bob_usage : Usage) :
  ann_usage.sheetsPerLetter = 2 →
  bob_usage.sheetsPerLetter = 4 →
  ann_usage.usedAllEnvelopes = true →
  ann_usage.leftoverSheets = 30 →
  bob_usage.usedAllSheets = true →
  bob_usage.leftoverEnvelopes = 20 →
  box.sheets = 40 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheet_count_l4056_405675


namespace NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l4056_405669

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Theorem for part (1)
theorem subset_condition (m : ℝ) : A m ⊆ B ↔ m < 2 ∨ m > 4 := by sorry

-- Theorem for part (2)
theorem intersection_empty_condition (m : ℝ) : A m ∩ B = ∅ ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l4056_405669


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4056_405667

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 2) :
  (1 / a + 1 / b) ≥ (5 + 2 * Real.sqrt 6) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = (5 + 2 * Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4056_405667


namespace NUMINAMATH_CALUDE_locus_of_circle_center_l4056_405650

/-
  Define the points M and N, and the circle passing through them with center P.
  Then prove that the locus of vertex P satisfies the given equation.
-/

-- Define the points M and N
def M : ℝ × ℝ := (0, -5)
def N : ℝ × ℝ := (0, 5)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  passes_through_M : (center.1 - M.1)^2 + (center.2 - M.2)^2 = (center.1 - N.1)^2 + (center.2 - N.2)^2

-- Define the locus equation
def locus_equation (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ (P.2^2 / 169 + P.1^2 / 144 = 1)

-- Theorem statement
theorem locus_of_circle_center (c : Circle) : locus_equation c.center :=
  sorry


end NUMINAMATH_CALUDE_locus_of_circle_center_l4056_405650


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_two_l4056_405622

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

-- State the theorem
theorem tangent_line_at_negative_two (b : ℝ) :
  b = -6 →
  let x₀ := -2
  let y₀ := f b x₀
  let m := (3 * x₀^2 - 12)  -- Derivative at x₀
  ∀ x, y₀ + m * (x - x₀) = 10 := by
sorry

-- Note: The actual proof is omitted as per instructions

end NUMINAMATH_CALUDE_tangent_line_at_negative_two_l4056_405622


namespace NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l4056_405689

/-- The maximum number of non-intersecting diagonals in a convex n-gon -/
def max_non_intersecting_diagonals (n : ℕ) : ℕ := n - 3

/-- Theorem: The maximum number of non-intersecting diagonals in a convex n-gon is n - 3 -/
theorem max_non_intersecting_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  max_non_intersecting_diagonals n = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l4056_405689


namespace NUMINAMATH_CALUDE_remainder_7623_div_11_l4056_405615

theorem remainder_7623_div_11 : 7623 % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_7623_div_11_l4056_405615


namespace NUMINAMATH_CALUDE_shelf_position_l4056_405674

theorem shelf_position (wall_width : ℝ) (picture_width : ℝ) 
  (hw : wall_width = 26)
  (hp : picture_width = 4) :
  let picture_center := wall_width / 2
  let shelf_left_edge := picture_center + picture_width / 2
  shelf_left_edge = 15 := by
  sorry

end NUMINAMATH_CALUDE_shelf_position_l4056_405674


namespace NUMINAMATH_CALUDE_triangle_theorem_l4056_405668

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

theorem triangle_theorem (t : Triangle) 
  (h_acute : 0 < t.A ∧ t.A < π/2 ∧ 0 < t.B ∧ t.B < π/2 ∧ 0 < t.C ∧ t.C < π/2)
  (h_sum : t.A + t.B + t.C = π)
  (μ : Vector2D)
  (v : Vector2D)
  (h_μ : μ = ⟨t.a^2 + t.c^2 - t.b^2, Real.sqrt 3 * t.a * t.c⟩)
  (h_v : v = ⟨Real.cos t.B, Real.sin t.B⟩)
  (h_parallel : ∃ (k : ℝ), μ = Vector2D.mk (k * v.x) (k * v.y)) :
  t.B = π/3 ∧ 3 * Real.sqrt 3 / 2 < Real.sin t.A + Real.sin t.C ∧ 
  Real.sin t.A + Real.sin t.C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l4056_405668


namespace NUMINAMATH_CALUDE_new_video_card_cost_l4056_405685

theorem new_video_card_cost (initial_cost : ℕ) (old_card_sale : ℕ) (total_spent : ℕ) : 
  initial_cost = 1200 →
  old_card_sale = 300 →
  total_spent = 1400 →
  total_spent - (initial_cost - old_card_sale) = 500 := by
sorry

end NUMINAMATH_CALUDE_new_video_card_cost_l4056_405685


namespace NUMINAMATH_CALUDE_log_equation_solution_l4056_405695

theorem log_equation_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx2 : x ≠ 2) :
  (Real.log x + Real.log y = Real.log (x + 2*y)) ↔ (y = x / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4056_405695


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l4056_405696

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 6) (hw : w = 8) (hh : h = 15) :
  Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 325 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l4056_405696


namespace NUMINAMATH_CALUDE_quadratic_sum_l4056_405692

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) ∧ (a + b + c = -195) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l4056_405692


namespace NUMINAMATH_CALUDE_nuts_in_boxes_l4056_405648

theorem nuts_in_boxes (x y z : ℕ) 
  (h1 : x + 6 = y + z) 
  (h2 : y + 10 = x + z) : 
  z = 8 := by
sorry

end NUMINAMATH_CALUDE_nuts_in_boxes_l4056_405648


namespace NUMINAMATH_CALUDE_residue_mod_17_l4056_405626

theorem residue_mod_17 : (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l4056_405626


namespace NUMINAMATH_CALUDE_cube_coloring_count_l4056_405657

/-- The number of distinct colorings of a cube's vertices -/
def distinctCubeColorings (m : ℕ) : ℚ :=
  (1 / 24) * m^2 * (m^6 + 17 * m^2 + 6)

/-- Theorem: The number of distinct ways to color the 8 vertices of a cube
    with m different colors, considering the symmetries of the cube,
    is equal to (1/24) * m^2 * (m^6 + 17m^2 + 6) -/
theorem cube_coloring_count (m : ℕ) :
  (distinctCubeColorings m) = (1 / 24) * m^2 * (m^6 + 17 * m^2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l4056_405657


namespace NUMINAMATH_CALUDE_gym_class_laps_l4056_405634

/-- Given a total distance to run, track length, and number of laps already run by two people,
    calculate the number of additional laps needed to reach the total distance. -/
def additional_laps_needed (total_distance : ℕ) (track_length : ℕ) (laps_run_per_person : ℕ) : ℕ :=
  let total_laps_run := 2 * laps_run_per_person
  let distance_run := total_laps_run * track_length
  let remaining_distance := total_distance - distance_run
  remaining_distance / track_length

/-- Prove that for the given conditions, the number of additional laps needed is 4. -/
theorem gym_class_laps : additional_laps_needed 2400 150 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gym_class_laps_l4056_405634


namespace NUMINAMATH_CALUDE_sin_product_10_30_50_70_l4056_405660

theorem sin_product_10_30_50_70 : 
  Real.sin (10 * π / 180) * Real.sin (30 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_sin_product_10_30_50_70_l4056_405660


namespace NUMINAMATH_CALUDE_xy_difference_l4056_405665

theorem xy_difference (x y : ℝ) (h : 10 * x^2 - 16 * x * y + 8 * y^2 + 6 * x - 4 * y + 1 = 0) :
  x - y = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_xy_difference_l4056_405665


namespace NUMINAMATH_CALUDE_fraction_equality_l4056_405629

theorem fraction_equality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h1 : (5*a + b) / (5*c + d) = (6*a + b) / (6*c + d))
  (h2 : (7*a + b) / (7*c + d) = 9) :
  (9*a + b) / (9*c + d) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4056_405629


namespace NUMINAMATH_CALUDE_mork_tax_rate_l4056_405635

/-- Proves that Mork's tax rate is 10% given the specified conditions --/
theorem mork_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) 
  (h1 : mork_income > 0)
  (h2 : mork_tax_rate > 0)
  (h3 : mork_tax_rate < 1)
  (h4 : (mork_tax_rate * mork_income + 3 * 0.2 * mork_income) / (4 * mork_income) = 0.175) :
  mork_tax_rate = 0.1 := by
sorry


end NUMINAMATH_CALUDE_mork_tax_rate_l4056_405635


namespace NUMINAMATH_CALUDE_polynomial_product_equals_difference_of_cubes_l4056_405605

theorem polynomial_product_equals_difference_of_cubes (x : ℝ) :
  (x^4 + 30*x^2 + 225) * (x^2 - 15) = x^6 - 3375 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equals_difference_of_cubes_l4056_405605


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l4056_405600

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l4056_405600


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l4056_405620

-- Define a normally distributed random variable
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
def P (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (x : normal_dist 4 σ) 
  (h : P {y : ℝ | y > 2} = 0.6) :
  P {y : ℝ | y > 6} = 0.4 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l4056_405620


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l4056_405632

/-- Represents a population for systematic sampling -/
structure Population where
  total : Nat
  omitted : Nat

/-- Checks if a given interval is valid for systematic sampling -/
def is_valid_interval (pop : Population) (interval : Nat) : Prop :=
  (pop.total - pop.omitted) % interval = 0

/-- The theorem to prove -/
theorem systematic_sampling_interval (pop : Population) 
  (h1 : pop.total = 102) 
  (h2 : pop.omitted = 2) : 
  is_valid_interval pop 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l4056_405632


namespace NUMINAMATH_CALUDE_quadratic_inequality_transformation_l4056_405651

theorem quadratic_inequality_transformation (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c < 0 ↔ x < -2 ∨ x > -1/2) →
  (∀ x : ℝ, c*x^2 - b*x + a > 0 ↔ 1/2 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_transformation_l4056_405651


namespace NUMINAMATH_CALUDE_cubic_roots_product_l4056_405656

theorem cubic_roots_product (a b c : ℂ) : 
  (3 * a^3 - 7 * a^2 + 4 * a - 9 = 0) ∧
  (3 * b^3 - 7 * b^2 + 4 * b - 9 = 0) ∧
  (3 * c^3 - 7 * c^2 + 4 * c - 9 = 0) →
  a * b * c = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l4056_405656


namespace NUMINAMATH_CALUDE_range_inequalities_l4056_405616

theorem range_inequalities 
  (a b x y : ℝ) 
  (ha : 12 < a ∧ a < 60) 
  (hb : 15 < b ∧ b < 36) 
  (hxy1 : -1/2 < x - y ∧ x - y < 1/2) 
  (hxy2 : 0 < x + y ∧ x + y < 1) : 
  (-12 < 2*a - b ∧ 2*a - b < 105) ∧ 
  (1/3 < a/b ∧ a/b < 4) ∧ 
  (-1 < 3*x - y ∧ 3*x - y < 2) := by
sorry

end NUMINAMATH_CALUDE_range_inequalities_l4056_405616


namespace NUMINAMATH_CALUDE_sum_of_non_visible_faces_l4056_405645

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of all faces on a standard die -/
def sumOfDieFaces : ℕ := (List.range 6).map (· + 1) |>.sum

/-- The total number of dice -/
def numberOfDice : ℕ := 4

/-- The list of visible face values -/
def visibleFaces : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 5, 6]

/-- The sum of visible face values -/
def sumOfVisibleFaces : ℕ := visibleFaces.sum

/-- Theorem: The sum of non-visible face values is 52 -/
theorem sum_of_non_visible_faces :
  numberOfDice * sumOfDieFaces - sumOfVisibleFaces = 52 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_non_visible_faces_l4056_405645


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l4056_405623

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 40 ∧ x - y = 10 → x * y = 375 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l4056_405623


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l4056_405653

theorem fourth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 4 * n^2) →
  a 4 = 28 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l4056_405653


namespace NUMINAMATH_CALUDE_problem_solution_l4056_405691

theorem problem_solution : (2010^2 - 2010) / 2010 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4056_405691


namespace NUMINAMATH_CALUDE_leah_bought_three_boxes_l4056_405694

/-- The number of boxes of birdseed Leah bought -/
def boxes_bought (existing_boxes weeks parrot_consumption cockatiel_consumption box_content : ℕ) : ℕ :=
  let total_consumption := weeks * (parrot_consumption + cockatiel_consumption)
  let total_boxes_needed := (total_consumption + box_content - 1) / box_content
  total_boxes_needed - existing_boxes

/-- Theorem stating that Leah bought 3 boxes of birdseed -/
theorem leah_bought_three_boxes :
  boxes_bought 5 12 100 50 225 = 3 := by sorry

end NUMINAMATH_CALUDE_leah_bought_three_boxes_l4056_405694


namespace NUMINAMATH_CALUDE_geometric_properties_l4056_405608

/-- A geometric figure -/
structure Figure where
  -- Add necessary properties here
  mk :: -- Constructor

/-- Defines when two figures can overlap perfectly -/
def can_overlap (f1 f2 : Figure) : Prop :=
  sorry

/-- Defines congruence between two figures -/
def congruent (f1 f2 : Figure) : Prop :=
  sorry

/-- The area of a figure -/
def area (f : Figure) : ℝ :=
  sorry

/-- The perimeter of a figure -/
def perimeter (f : Figure) : ℝ :=
  sorry

theorem geometric_properties :
  (∀ f1 f2 : Figure, can_overlap f1 f2 → congruent f1 f2) ∧
  (∀ f1 f2 : Figure, congruent f1 f2 → area f1 = area f2) ∧
  (∃ f1 f2 : Figure, area f1 = area f2 ∧ ¬congruent f1 f2) ∧
  (∃ f1 f2 : Figure, perimeter f1 = perimeter f2 ∧ ¬congruent f1 f2) :=
sorry

end NUMINAMATH_CALUDE_geometric_properties_l4056_405608


namespace NUMINAMATH_CALUDE_problem_solution_l4056_405679

theorem problem_solution (x y : ℝ) (h1 : x = 3) (h2 : x + y = 60 * (1 / x)) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4056_405679


namespace NUMINAMATH_CALUDE_yoongi_age_proof_l4056_405621

/-- Yoongi's age -/
def yoongi_age : ℕ := 8

/-- Hoseok's age -/
def hoseok_age : ℕ := yoongi_age + 2

/-- The sum of Yoongi's and Hoseok's ages -/
def total_age : ℕ := yoongi_age + hoseok_age

theorem yoongi_age_proof : yoongi_age = 8 :=
  by
    have h1 : hoseok_age = yoongi_age + 2 := rfl
    have h2 : total_age = 18 := rfl
    sorry

end NUMINAMATH_CALUDE_yoongi_age_proof_l4056_405621


namespace NUMINAMATH_CALUDE_distance_covered_72min_10kmph_l4056_405642

/-- The distance covered by a man walking at a given speed for a given time. -/
def distanceCovered (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A man walking for 72 minutes at a speed of 10 km/hr covers a distance of 12 km. -/
theorem distance_covered_72min_10kmph :
  let speed : ℝ := 10  -- Speed in km/hr
  let time : ℝ := 72 / 60  -- Time in hours (72 minutes converted to hours)
  distanceCovered speed time = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_covered_72min_10kmph_l4056_405642


namespace NUMINAMATH_CALUDE_inequality_solution_set_a_range_l4056_405655

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2|

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (2 * f x < 4 - |x - 1|) ↔ (-7/3 < x ∧ x < -1) :=
sorry

-- Part 2
theorem a_range (a m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_a_range_l4056_405655


namespace NUMINAMATH_CALUDE_max_xy_value_l4056_405649

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : 
  x * y ≤ 168 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l4056_405649
