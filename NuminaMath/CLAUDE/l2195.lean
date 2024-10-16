import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2195_219562

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 8) ∧ (6 * x + 3 * y = 21) ↔ x = 22/9 ∧ y = 19/9 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2195_219562


namespace NUMINAMATH_CALUDE_zero_function_satisfies_equation_zero_function_is_solution_l2195_219557

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 2*y) * f (x - 2*y) = (f x + f y)^2 - 16 * y^2 * f x

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ x => 0

/-- Theorem: The zero function satisfies the functional equation -/
theorem zero_function_satisfies_equation : SatisfiesFunctionalEquation ZeroFunction := by
  sorry

/-- Theorem: The zero function is a solution to the functional equation -/
theorem zero_function_is_solution :
  ∃ f : ℝ → ℝ, SatisfiesFunctionalEquation f ∧ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_function_satisfies_equation_zero_function_is_solution_l2195_219557


namespace NUMINAMATH_CALUDE_least_number_with_remainder_5_l2195_219528

/-- The least number that leaves a remainder of 5 when divided by 8, 12, 15, and 20 -/
def leastNumber : ℕ := 125

/-- Checks if a number leaves a remainder of 5 when divided by the given divisor -/
def hasRemainder5 (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 5

theorem least_number_with_remainder_5 :
  (∀ divisor ∈ [8, 12, 15, 20], hasRemainder5 leastNumber divisor) ∧
  (∀ m < leastNumber, ∃ divisor ∈ [8, 12, 15, 20], ¬hasRemainder5 m divisor) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_5_l2195_219528


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l2195_219572

theorem largest_perfect_square_factor_of_4410 : 
  ∃ (n : ℕ), n * n = 441 ∧ n * n ∣ 4410 ∧ ∀ (m : ℕ), m * m ∣ 4410 → m * m ≤ n * n := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l2195_219572


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2195_219548

theorem sum_of_reciprocals_of_roots (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2022*a + 1011 = 0 →
  b^3 - 2022*b + 1011 = 0 →
  c^3 - 2022*c + 1011 = 0 →
  1/a + 1/b + 1/c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2195_219548


namespace NUMINAMATH_CALUDE_fourth_to_sixth_ratio_l2195_219514

structure MathClasses where
  fourth_level : ℕ
  sixth_level : ℕ
  seventh_level : ℕ
  total_students : ℕ

def MathClasses.valid (c : MathClasses) : Prop :=
  c.fourth_level = c.sixth_level ∧
  c.seventh_level = 2 * c.fourth_level ∧
  c.sixth_level = 40 ∧
  c.total_students = 520

theorem fourth_to_sixth_ratio (c : MathClasses) (h : c.valid) :
  c.fourth_level = c.sixth_level :=
by sorry

end NUMINAMATH_CALUDE_fourth_to_sixth_ratio_l2195_219514


namespace NUMINAMATH_CALUDE_sum_congruence_l2195_219580

def large_sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruence : large_sum % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l2195_219580


namespace NUMINAMATH_CALUDE_basketball_score_proof_l2195_219516

theorem basketball_score_proof (joe tim ken : ℕ) 
  (h1 : tim = joe + 20)
  (h2 : tim * 2 = ken)
  (h3 : joe + tim + ken = 100) :
  tim = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l2195_219516


namespace NUMINAMATH_CALUDE_intersection_symmetric_implies_p_range_l2195_219535

/-- The line equation: x = ky - 1 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop := x = k * y - 1

/-- The circle equation: x² + y² + kx + my + 2p = 0 -/
def circle_equation (k m p : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + k*x + m*y + 2*p = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric about y = x -/
def symmetric_about_y_eq_x (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = y₂ ∧ y₁ = x₂

theorem intersection_symmetric_implies_p_range
  (k m p : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    circle_equation k m p x₁ y₁ ∧
    circle_equation k m p x₂ y₂ ∧
    symmetric_about_y_eq_x x₁ y₁ x₂ y₂) →
  p < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_symmetric_implies_p_range_l2195_219535


namespace NUMINAMATH_CALUDE_cardinality_of_B_l2195_219556

def A : Finset ℚ := {1, 2, 3, 4, 6}

def B : Finset ℚ := Finset.image (λ (p : ℚ × ℚ) => p.1 / p.2) (A.product A)

theorem cardinality_of_B : Finset.card B = 13 := by sorry

end NUMINAMATH_CALUDE_cardinality_of_B_l2195_219556


namespace NUMINAMATH_CALUDE_B_equals_D_l2195_219541

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

-- Define set D (real numbers not less than 1)
def D : Set ℝ := {y : ℝ | y ≥ 1}

-- Theorem statement
theorem B_equals_D : B = D := by sorry

end NUMINAMATH_CALUDE_B_equals_D_l2195_219541


namespace NUMINAMATH_CALUDE_remainder_plus_three_l2195_219574

/-- f(x) represents the remainder of x divided by 3 -/
def f (x : ℕ) : ℕ := x % 3

/-- For all natural numbers x, f(x+3) = f(x) -/
theorem remainder_plus_three (x : ℕ) : f (x + 3) = f x := by
  sorry

end NUMINAMATH_CALUDE_remainder_plus_three_l2195_219574


namespace NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l2195_219507

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square -/
def square_area (side : ℕ) : ℕ := side * side

/-- Checks if a square can contain two rectangles -/
def can_contain_rectangles (side : ℕ) (rect1 rect2 : Rectangle) : Prop :=
  (max rect1.width rect2.width ≤ side) ∧ (rect1.height + rect2.height ≤ side)

theorem smallest_square_area_for_rectangles :
  ∃ (side : ℕ),
    let rect1 : Rectangle := ⟨3, 4⟩
    let rect2 : Rectangle := ⟨4, 5⟩
    can_contain_rectangles side rect1 rect2 ∧
    square_area side = 49 ∧
    ∀ (smaller_side : ℕ), smaller_side < side →
      ¬ can_contain_rectangles smaller_side rect1 rect2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l2195_219507


namespace NUMINAMATH_CALUDE_round_trip_time_l2195_219591

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream speed, and the total distance traveled. -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : total_distance = 420) : 
  (total_distance / (boat_speed + stream_speed) + 
   total_distance / (boat_speed - stream_speed)) = 120 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l2195_219591


namespace NUMINAMATH_CALUDE_factorization_x4_minus_64_l2195_219525

theorem factorization_x4_minus_64 (x : ℝ) : x^4 - 64 = (x^2 - 8) * (x^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_64_l2195_219525


namespace NUMINAMATH_CALUDE_horror_movie_tickets_l2195_219530

theorem horror_movie_tickets (romance_tickets : ℕ) (horror_tickets : ℕ) : 
  romance_tickets = 25 →
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 := by
sorry

end NUMINAMATH_CALUDE_horror_movie_tickets_l2195_219530


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2195_219518

theorem largest_solution_of_equation : 
  ∃ (b : ℝ), (3 * b + 4) * (b - 2) = 9 * b ∧ 
  ∀ (x : ℝ), (3 * x + 4) * (x - 2) = 9 * x → x ≤ b ∧ 
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2195_219518


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l2195_219578

/-- The repeating decimal 0.76204̄ as a rational number -/
def repeating_decimal : ℚ := 761280 / 999000

theorem repeating_decimal_proof : repeating_decimal = 0.76 + (204 : ℚ) / 999000 := by sorry


end NUMINAMATH_CALUDE_repeating_decimal_proof_l2195_219578


namespace NUMINAMATH_CALUDE_ac_price_is_1500_l2195_219561

-- Define the price ratios
def car_ratio : ℚ := 5
def ac_ratio : ℚ := 3
def scooter_ratio : ℚ := 2

-- Define the price difference between scooter and air conditioner
def price_difference : ℚ := 500

-- Define the tax rate for the car
def car_tax_rate : ℚ := 0.1

-- Define the discount rate for the air conditioner
def ac_discount_rate : ℚ := 0.15

-- Define the original price of the air conditioner
def original_ac_price : ℚ := 1500

-- Theorem statement
theorem ac_price_is_1500 :
  ∃ (x : ℚ),
    scooter_ratio * x = ac_ratio * x + price_difference ∧
    original_ac_price = ac_ratio * x :=
by sorry

end NUMINAMATH_CALUDE_ac_price_is_1500_l2195_219561


namespace NUMINAMATH_CALUDE_new_scheme_fixed_salary_is_1000_l2195_219504

/-- Represents the salesman's compensation scheme -/
structure CompensationScheme where
  fixedSalary : ℕ
  commissionRate : ℚ
  commissionThreshold : ℕ

/-- Calculates the total compensation for a given sales amount and compensation scheme -/
def calculateCompensation (sales : ℕ) (scheme : CompensationScheme) : ℚ :=
  scheme.fixedSalary + scheme.commissionRate * max (sales - scheme.commissionThreshold) 0

/-- Theorem stating that the fixed salary in the new scheme is 1000 -/
theorem new_scheme_fixed_salary_is_1000 (totalSales : ℕ) (oldScheme newScheme : CompensationScheme) :
  totalSales = 12000 →
  oldScheme.fixedSalary = 0 →
  oldScheme.commissionRate = 1/20 →
  oldScheme.commissionThreshold = 0 →
  newScheme.commissionRate = 1/40 →
  newScheme.commissionThreshold = 4000 →
  calculateCompensation totalSales newScheme = calculateCompensation totalSales oldScheme + 600 →
  newScheme.fixedSalary = 1000 := by
  sorry

#eval calculateCompensation 12000 { fixedSalary := 1000, commissionRate := 1/40, commissionThreshold := 4000 }
#eval calculateCompensation 12000 { fixedSalary := 0, commissionRate := 1/20, commissionThreshold := 0 }

end NUMINAMATH_CALUDE_new_scheme_fixed_salary_is_1000_l2195_219504


namespace NUMINAMATH_CALUDE_lemonade_intermission_l2195_219565

theorem lemonade_intermission (total : ℝ) (first : ℝ) (third : ℝ) (second : ℝ)
  (h_total : total = 0.92)
  (h_first : first = 0.25)
  (h_third : third = 0.25)
  (h_sum : total = first + second + third) :
  second = 0.42 := by
sorry

end NUMINAMATH_CALUDE_lemonade_intermission_l2195_219565


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2195_219576

theorem bus_ride_difference (vince_ride zachary_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : zachary_ride = 0.5) : 
  vince_ride - zachary_ride = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2195_219576


namespace NUMINAMATH_CALUDE_valid_lineup_count_l2195_219553

def total_players : ℕ := 16
def num_starters : ℕ := 7
def num_triplets : ℕ := 3
def num_twins : ℕ := 2

def valid_lineups : ℕ := 8778

theorem valid_lineup_count :
  (Nat.choose total_players num_starters) -
  ((Nat.choose (total_players - num_triplets) (num_starters - num_triplets)) +
   (Nat.choose (total_players - num_twins) (num_starters - num_twins)) -
   (Nat.choose (total_players - num_triplets - num_twins) (num_starters - num_triplets - num_twins)))
  = valid_lineups := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l2195_219553


namespace NUMINAMATH_CALUDE_triangle_equality_l2195_219529

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l2195_219529


namespace NUMINAMATH_CALUDE_train_crossing_time_l2195_219536

/-- Given a train and a platform with specific dimensions and time to pass,
    calculate the time it takes for the train to cross a stationary point. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 600)
  (h2 : platform_length = 450)
  (h3 : time_to_pass_platform = 105) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2195_219536


namespace NUMINAMATH_CALUDE_investment_income_calculation_l2195_219593

/-- Calculates the annual income from an investment in shares given the investment amount,
    share face value, quoted price, and dividend rate. -/
def annual_income (investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  (investment / quoted_price) * (face_value * dividend_rate)

/-- Theorem stating that for the given investment scenario, the annual income is 728 -/
theorem investment_income_calculation :
  let investment : ℚ := 4940
  let face_value : ℚ := 10
  let quoted_price : ℚ := 9.5
  let dividend_rate : ℚ := 14 / 100
  annual_income investment face_value quoted_price dividend_rate = 728 := by
  sorry


end NUMINAMATH_CALUDE_investment_income_calculation_l2195_219593


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2195_219584

def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 8
def num_rabbits : ℕ := 5
def num_people : ℕ := 4

theorem pet_store_combinations : 
  (num_puppies * num_kittens * num_hamsters * num_rabbits) * Nat.factorial num_people = 115200 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2195_219584


namespace NUMINAMATH_CALUDE_multiplication_fraction_product_l2195_219586

theorem multiplication_fraction_product : 11 * (1 / 17) * 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_product_l2195_219586


namespace NUMINAMATH_CALUDE_continuity_at_one_l2195_219595

def f (x : ℝ) := -4 * x^2 - 7

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_one_l2195_219595


namespace NUMINAMATH_CALUDE_constant_for_max_n_l2195_219592

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, n ≤ 8 → c * n^2 ≤ 8100) ∧ 
  (c * 9^2 > 8100) ↔ 
  c = 126.5625 := by
sorry

end NUMINAMATH_CALUDE_constant_for_max_n_l2195_219592


namespace NUMINAMATH_CALUDE_max_tourists_per_day_may_l2195_219500

/-- Proves the maximum average number of tourists per day for the last 10 days of May --/
theorem max_tourists_per_day_may (feb_tourists : ℕ) (apr_tourists : ℕ) (may_21_tourists : ℕ)
  (h1 : feb_tourists = 16000)
  (h2 : apr_tourists = 25000)
  (h3 : may_21_tourists = 21250) :
  ∃ (max_daily_tourists : ℕ),
    max_daily_tourists = 100000 ∧
    (may_21_tourists : ℝ) + 10 * max_daily_tourists ≤ (apr_tourists : ℝ) * (1 + 0.25) :=
by sorry

end NUMINAMATH_CALUDE_max_tourists_per_day_may_l2195_219500


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2195_219589

theorem fraction_to_decimal (n : ℕ) (d : ℕ) (h : d = 5^4 * 2) :
  (n : ℚ) / d = 47 / d → (n : ℚ) / d = 0.0376 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2195_219589


namespace NUMINAMATH_CALUDE_range_of_a_l2195_219540

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 4 * x + a > 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2195_219540


namespace NUMINAMATH_CALUDE_irrational_in_set_l2195_219588

-- Define the set of numbers
def numbers : Set ℝ := {0, -2, Real.sqrt 3, 1/2}

-- Define a predicate for rational numbers
def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Theorem statement
theorem irrational_in_set :
  ∃ (x : ℝ), x ∈ numbers ∧ ¬(isRational x) ∧
  ∀ (y : ℝ), y ∈ numbers ∧ y ≠ x → isRational y :=
sorry

end NUMINAMATH_CALUDE_irrational_in_set_l2195_219588


namespace NUMINAMATH_CALUDE_bus_occupancy_problem_l2195_219531

/-- Given an initial number of people on a bus, and the number of people who get on and off,
    calculate the final number of people on the bus. -/
def final_bus_occupancy (initial : ℕ) (got_on : ℕ) (got_off : ℕ) : ℕ :=
  initial + got_on - got_off

/-- Theorem stating that with 32 people initially on the bus, 19 getting on, and 13 getting off,
    the final number of people on the bus is 38. -/
theorem bus_occupancy_problem :
  final_bus_occupancy 32 19 13 = 38 := by
  sorry

end NUMINAMATH_CALUDE_bus_occupancy_problem_l2195_219531


namespace NUMINAMATH_CALUDE_weight_measurements_l2195_219526

/-- The set of available weights in pounds -/
def weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be weighed using the given weights -/
def max_weight : ℕ := 40

/-- The number of different weights that can be measured -/
def different_weights : ℕ := 40

/-- Theorem stating the maximum weight and number of different weights -/
theorem weight_measurements :
  (weights.sum = max_weight) ∧
  (∀ w : ℕ, w > 0 ∧ w ≤ max_weight → ∃ combination : List ℕ, combination.all (· ∈ weights) ∧ combination.sum = w) ∧
  (different_weights = max_weight) :=
sorry

end NUMINAMATH_CALUDE_weight_measurements_l2195_219526


namespace NUMINAMATH_CALUDE_total_employees_is_100_l2195_219564

/-- The ratio of employees in groups A, B, and C -/
def group_ratio : Fin 3 → ℕ
  | 0 => 5  -- Group A
  | 1 => 4  -- Group B
  | 2 => 1  -- Group C

/-- The total sample size -/
def sample_size : ℕ := 20

/-- The probability of selecting both person A and person B in group C -/
def prob_select_two : ℚ := 1 / 45

theorem total_employees_is_100 :
  ∀ (total : ℕ),
  (∃ (group_C_size : ℕ),
    /- Group C size is 1/10 of the total -/
    group_C_size = total / 10 ∧
    /- The probability of selecting 2 from group C matches the given probability -/
    (group_C_size.choose 2 : ℚ) / total.choose 2 = prob_select_two ∧
    /- The sample size for group C is 2 -/
    group_C_size * sample_size / total = 2) →
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_total_employees_is_100_l2195_219564


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2195_219545

/-- Proves that given a hyperbola with specific conditions, its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧  -- hyperbola equation
  (∃ (x y : ℝ), y = Real.sqrt 3 * x) ∧    -- asymptote condition
  (∃ (x y : ℝ), y^2 = 16*x ∧ x^2/a^2 + y^2/b^2 = 1) -- focus on directrix condition
  →
  a^2 = 4 ∧ b^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2195_219545


namespace NUMINAMATH_CALUDE_number_of_divisors_3960_l2195_219585

theorem number_of_divisors_3960 : Nat.card (Nat.divisors 3960) = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3960_l2195_219585


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l2195_219524

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line (x y : ℝ) : 
  (y = 2*x + 1) → 
  (y = 3*x - 1) → 
  (∃ m : ℝ, 2*x + y - m = 0 ∧ (∀ x y : ℝ, 2*x + y - m = 0 ↔ 2*x + y - 3 = 0)) →
  (2*x + y - 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l2195_219524


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_range_of_a_when_intersection_is_empty_l2195_219566

def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a > 0}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | x > 3} := by sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ a : ℝ, A a ∩ (U \ B) = ∅ ↔ a ≤ -6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_range_of_a_when_intersection_is_empty_l2195_219566


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2195_219582

theorem inequalities_for_positive_reals (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (-a < -b) ∧ ((b/a + a/b) > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2195_219582


namespace NUMINAMATH_CALUDE_sine_transformation_l2195_219583

theorem sine_transformation (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let g (t : ℝ) := f (t - (2/3) * Real.pi)
  let h (t : ℝ) := g (t / 3)
  h x = Real.sin (3 * x - (2/3) * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_sine_transformation_l2195_219583


namespace NUMINAMATH_CALUDE_quadratic_derivative_bound_l2195_219511

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The derivative of a quadratic function f(x) = ax^2 + bx + c -/
def quadratic_derivative (a b : ℝ) : ℝ → ℝ := fun x ↦ 2 * a * x + b

theorem quadratic_derivative_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |quadratic_function a b c x| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |quadratic_derivative a b x| ≤ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_derivative_bound_l2195_219511


namespace NUMINAMATH_CALUDE_range_of_4x_plus_2y_l2195_219570

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) : 
  2 ≤ 4*x + 2*y ∧ 4*x + 2*y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_4x_plus_2y_l2195_219570


namespace NUMINAMATH_CALUDE_compute_expression_l2195_219527

theorem compute_expression : 7^2 - 2*(5) + 2^3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2195_219527


namespace NUMINAMATH_CALUDE_third_side_length_equal_to_altitude_l2195_219558

/-- Given an acute-angled triangle with two sides of lengths √13 and √10 cm,
    if the third side is equal to the altitude drawn to it,
    then the length of the third side is 3 cm. -/
theorem third_side_length_equal_to_altitude
  (a b c : ℝ) -- sides of the triangle
  (h : ℝ) -- altitude to side c
  (acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) -- acute-angled triangle
  (side1 : a = Real.sqrt 13)
  (side2 : b = Real.sqrt 10)
  (altitude_eq_side : h = c)
  (pythagorean1 : a^2 = (c - h)^2 + h^2)
  (pythagorean2 : b^2 = h^2 + h^2) :
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_equal_to_altitude_l2195_219558


namespace NUMINAMATH_CALUDE_constant_term_product_l2195_219563

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between r, p, and q
variable (h_prod : r = p * q)

-- Define the constant terms of p and r
variable (h_p_const : p.coeff 0 = 5)
variable (h_r_const : r.coeff 0 = -10)

-- Theorem statement
theorem constant_term_product :
  q.eval 0 = -2 := by sorry

end NUMINAMATH_CALUDE_constant_term_product_l2195_219563


namespace NUMINAMATH_CALUDE_min_isosceles_right_triangles_10x100_l2195_219537

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle

/-- Returns the minimum number of isosceles right triangles needed to cover a rectangle -/
def minIsoscelesRightTriangles (r : Rectangle) : ℕ := sorry

/-- The theorem statement -/
theorem min_isosceles_right_triangles_10x100 :
  minIsoscelesRightTriangles ⟨100, 10⟩ = 11 := by sorry

end NUMINAMATH_CALUDE_min_isosceles_right_triangles_10x100_l2195_219537


namespace NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_lunch_box_l2195_219547

theorem min_students_with_blue_eyes_and_lunch_box
  (total_students : ℕ)
  (blue_eyes : ℕ)
  (lunch_box : ℕ)
  (h1 : total_students = 35)
  (h2 : blue_eyes = 20)
  (h3 : lunch_box = 22)
  : ∃ (both : ℕ), both ≥ 7 ∧ both ≤ min blue_eyes lunch_box :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_lunch_box_l2195_219547


namespace NUMINAMATH_CALUDE_necklace_stand_capacity_l2195_219559

-- Define the given constants
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 30
def current_rings : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_price : ℕ := 4
def ring_price : ℕ := 10
def bracelet_price : ℕ := 5
def total_cost : ℕ := 183

-- Theorem to prove
theorem necklace_stand_capacity : ∃ (total_necklaces : ℕ), 
  total_necklaces = current_necklaces + 
    ((total_cost - (ring_price * (ring_capacity - current_rings) + 
                    bracelet_price * (bracelet_capacity - current_bracelets))) / necklace_price) :=
by sorry

end NUMINAMATH_CALUDE_necklace_stand_capacity_l2195_219559


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2195_219579

/-- Proves that in a college with 190 girls and 494 total students, the ratio of boys to girls is 152:95 -/
theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : total_students = 494) 
  (h2 : girls = 190) : 
  (total_students - girls) / girls = 152 / 95 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2195_219579


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l2195_219568

theorem cube_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l2195_219568


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2195_219577

theorem min_value_of_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x + 9 / y) ≥ 25 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 4 / x + 9 / y = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2195_219577


namespace NUMINAMATH_CALUDE_no_valid_operation_l2195_219505

def equation (op : ℝ → ℝ → ℝ) : Prop :=
  op 8 2 * 3 + 7 - (5 - 3) = 16

theorem no_valid_operation : 
  ¬ (equation (·/·) ∨ equation (·*·) ∨ equation (·+·) ∨ equation (·-·)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_operation_l2195_219505


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_7_l2195_219569

/-- The cost of one portion of ice cream in kopecks -/
def ice_cream_cost : ℕ := 7

/-- Fedya's money in kopecks -/
def fedya_money : ℕ := ice_cream_cost - 7

/-- Masha's money in kopecks -/
def masha_money : ℕ := ice_cream_cost - 1

theorem ice_cream_cost_is_7 :
  (fedya_money + masha_money < ice_cream_cost) ∧
  (fedya_money = ice_cream_cost - 7) ∧
  (masha_money = ice_cream_cost - 1) →
  ice_cream_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_is_7_l2195_219569


namespace NUMINAMATH_CALUDE_elevator_stop_time_is_three_l2195_219501

/-- Represents the race to the top of a building --/
structure BuildingRace where
  stories : ℕ
  lola_time_per_story : ℕ
  elevator_time_per_story : ℕ
  total_time : ℕ

/-- Calculates the time the elevator stops on each floor --/
def elevator_stop_time (race : BuildingRace) : ℕ :=
  let lola_total_time := race.stories * race.lola_time_per_story
  let elevator_move_time := race.stories * race.elevator_time_per_story
  let total_stop_time := race.total_time - elevator_move_time
  total_stop_time / (race.stories - 1)

/-- The theorem stating that the elevator stops for 3 seconds on each floor --/
theorem elevator_stop_time_is_three (race : BuildingRace) 
    (h1 : race.stories = 20)
    (h2 : race.lola_time_per_story = 10)
    (h3 : race.elevator_time_per_story = 8)
    (h4 : race.total_time = 220) :
  elevator_stop_time race = 3 := by
  sorry

#eval elevator_stop_time { stories := 20, lola_time_per_story := 10, elevator_time_per_story := 8, total_time := 220 }

end NUMINAMATH_CALUDE_elevator_stop_time_is_three_l2195_219501


namespace NUMINAMATH_CALUDE_min_benches_for_equal_occupancy_l2195_219543

/-- Represents the capacity of a bench for adults and children -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Finds the minimum number of benches required for equal and full occupancy -/
def minBenchesRequired (capacity : BenchCapacity) : Nat :=
  Nat.lcm capacity.adults capacity.children / capacity.adults

/-- Theorem stating the minimum number of benches required -/
theorem min_benches_for_equal_occupancy (capacity : BenchCapacity) 
  (h1 : capacity.adults = 8) 
  (h2 : capacity.children = 12) : 
  minBenchesRequired capacity = 3 := by
  sorry

#eval minBenchesRequired ⟨8, 12⟩

end NUMINAMATH_CALUDE_min_benches_for_equal_occupancy_l2195_219543


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2195_219503

theorem solution_set_quadratic_inequality :
  let S : Set ℝ := {x | x^2 + x - 2 ≥ 0}
  S = {x : ℝ | x ≤ -2 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2195_219503


namespace NUMINAMATH_CALUDE_probability_of_trio_l2195_219550

-- Define the original deck
def original_deck : ℕ := 52

-- Define the number of cards for each number
def cards_per_number : ℕ := 4

-- Define the number of different numbers in the deck
def different_numbers : ℕ := 13

-- Define the number of cards removed
def cards_removed : ℕ := 3

-- Define the remaining deck size
def remaining_deck : ℕ := original_deck - cards_removed

-- Define the number of ways to choose 3 cards from the remaining deck
def total_ways : ℕ := Nat.choose remaining_deck 3

-- Define the number of ways to choose a trio of the same number
def trio_ways : ℕ := (different_numbers - 2) * Nat.choose cards_per_number 3 + 1

-- Theorem statement
theorem probability_of_trio : 
  (trio_ways : ℚ) / total_ways = 45 / 18424 := by sorry

end NUMINAMATH_CALUDE_probability_of_trio_l2195_219550


namespace NUMINAMATH_CALUDE_zero_is_global_minimum_l2195_219502

-- Define the function f(x) = (x - 1)e^(x - 1)
noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

-- Theorem statement
theorem zero_is_global_minimum :
  ∀ x : ℝ, f 0 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_zero_is_global_minimum_l2195_219502


namespace NUMINAMATH_CALUDE_fish_estimation_l2195_219510

/-- The number of fish caught and marked on the first day -/
def marked_fish : ℕ := 30

/-- The number of fish caught on the second day -/
def second_catch : ℕ := 40

/-- The number of marked fish caught on the second day -/
def marked_recaught : ℕ := 2

/-- The estimated number of fish in the pond -/
def estimated_fish : ℕ := marked_fish * second_catch / marked_recaught

theorem fish_estimation :
  estimated_fish = 600 :=
sorry

end NUMINAMATH_CALUDE_fish_estimation_l2195_219510


namespace NUMINAMATH_CALUDE_hexagonal_quadratic_coefficient_l2195_219596

-- Define hexagonal numbers
def hexagonal (n : ℕ) : ℕ := 2 * n^2 - n

-- Define the general quadratic form for hexagonal numbers
def quadratic_form (a b c n : ℕ) : ℕ := a * n^2 + b * n + c

-- Theorem statement
theorem hexagonal_quadratic_coefficient :
  ∃ (b c : ℕ), ∀ (n : ℕ), n > 0 → hexagonal n = quadratic_form 3 b c n :=
sorry

end NUMINAMATH_CALUDE_hexagonal_quadratic_coefficient_l2195_219596


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l2195_219599

theorem sum_first_150_remainder (n : Nat) (h : n = 150) : 
  (n * (n + 1) / 2) % 11325 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l2195_219599


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_equals_three_l2195_219597

theorem quadratic_roots_imply_m_equals_three (a m : ℤ) :
  a ≠ 1 →
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
    (a - 1) * x^2 - m * x + a = 0 ∧
    (a - 1) * y^2 - m * y + a = 0) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_equals_three_l2195_219597


namespace NUMINAMATH_CALUDE_davids_biology_marks_l2195_219554

def english_marks : ℕ := 86
def math_marks : ℕ := 89
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 87
def average_marks : ℕ := 85
def num_subjects : ℕ := 5

theorem davids_biology_marks :
  let known_subjects_total := english_marks + math_marks + physics_marks + chemistry_marks
  let all_subjects_total := average_marks * num_subjects
  all_subjects_total - known_subjects_total = 81 := by sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l2195_219554


namespace NUMINAMATH_CALUDE_expression_simplification_l2195_219515

theorem expression_simplification (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = -2/3 * x - 2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2195_219515


namespace NUMINAMATH_CALUDE_modular_congruence_problem_l2195_219517

theorem modular_congruence_problem : ∃ m : ℕ, 
  (215 * 953 + 100) % 50 = m ∧ 0 ≤ m ∧ m < 50 :=
by
  use 45
  sorry

end NUMINAMATH_CALUDE_modular_congruence_problem_l2195_219517


namespace NUMINAMATH_CALUDE_toms_age_l2195_219552

theorem toms_age (j t : ℕ) 
  (h1 : j - 6 = 3 * (t - 6))  -- John was thrice as old as Tom 6 years ago
  (h2 : j + 4 = 2 * (t + 4))  -- John will be 2 times as old as Tom in 4 years
  : t = 16 := by  -- Tom's current age is 16
  sorry

end NUMINAMATH_CALUDE_toms_age_l2195_219552


namespace NUMINAMATH_CALUDE_selection_ways_eq_six_l2195_219542

/-- The number of types of pencils -/
def num_pencil_types : ℕ := 3

/-- The number of types of erasers -/
def num_eraser_types : ℕ := 2

/-- The number of ways to select one pencil and one eraser -/
def num_selection_ways : ℕ := num_pencil_types * num_eraser_types

/-- Theorem stating that the number of ways to select one pencil and one eraser is 6 -/
theorem selection_ways_eq_six : num_selection_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_eq_six_l2195_219542


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l2195_219544

/-- Calculates the number of shirts Kendra needs for two weeks -/
def shirts_needed : ℕ :=
  let school_days := 5
  let after_school_days := 3
  let saturday_shirts := 1
  let sunday_shirts := 2
  let weeks := 2
  (school_days + after_school_days + saturday_shirts + sunday_shirts) * weeks

/-- Theorem stating that Kendra needs 22 shirts for two weeks -/
theorem kendra_shirts_theorem : shirts_needed = 22 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l2195_219544


namespace NUMINAMATH_CALUDE_no_adjacent_women_correct_a_not_first_b_not_last_correct_fixed_sequence_correct_a_left_of_b_correct_l2195_219571

def num_men : Nat := 4
def num_women : Nat := 3
def total_people : Nat := num_men + num_women

-- Function to calculate the number of arrangements where no two women are adjacent
def arrangements_no_adjacent_women : Nat :=
  Nat.factorial num_men * Nat.descFactorial (num_men + 1) num_women

-- Function to calculate the number of arrangements where Man A is not first and Man B is not last
def arrangements_a_not_first_b_not_last : Nat :=
  Nat.factorial total_people - 2 * Nat.factorial (total_people - 1) + Nat.factorial (total_people - 2)

-- Function to calculate the number of arrangements where Men A, B, and C are in a fixed sequence
def arrangements_fixed_sequence : Nat :=
  Nat.factorial total_people / Nat.factorial 3

-- Function to calculate the number of arrangements where Man A is to the left of Man B
def arrangements_a_left_of_b : Nat :=
  Nat.factorial total_people / 2

-- Theorems to prove
theorem no_adjacent_women_correct :
  arrangements_no_adjacent_women = 1440 := by sorry

theorem a_not_first_b_not_last_correct :
  arrangements_a_not_first_b_not_last = 3720 := by sorry

theorem fixed_sequence_correct :
  arrangements_fixed_sequence = 840 := by sorry

theorem a_left_of_b_correct :
  arrangements_a_left_of_b = 2520 := by sorry

end NUMINAMATH_CALUDE_no_adjacent_women_correct_a_not_first_b_not_last_correct_fixed_sequence_correct_a_left_of_b_correct_l2195_219571


namespace NUMINAMATH_CALUDE_intersection_equals_zero_one_l2195_219575

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ n ∈ A, x = n^2}

def P : Set ℕ := A ∩ B

theorem intersection_equals_zero_one : P = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_zero_one_l2195_219575


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2195_219508

theorem mixed_number_calculation : 7 * (12 + 2/5) - 3 = 83.8 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2195_219508


namespace NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2195_219594

/-- The angle between clock hands at 8:30 --/
theorem angle_between_clock_hands_at_8_30 : ℝ :=
  let hours : ℝ := 8.5
  let minutes : ℝ := 30
  let angle_per_hour : ℝ := 360 / 12
  let hour_hand_angle : ℝ := hours * angle_per_hour
  let minute_hand_angle : ℝ := minutes * (360 / 60)
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  75

/-- Proof that the angle between clock hands at 8:30 is 75° --/
theorem angle_between_clock_hands_at_8_30_is_75 :
  angle_between_clock_hands_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2195_219594


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2195_219512

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 6 ∧ x₂ = -2) ∧ 
  (x₁^2 - 4*x₁ - 12 = 0) ∧ 
  (x₂^2 - 4*x₂ - 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2195_219512


namespace NUMINAMATH_CALUDE_equation_roots_l2195_219532

theorem equation_roots : 
  ∃ (x₁ x₂ x₃ x₄ : ℂ), 
    (x₁ = -1/12 ∧ x₂ = 1/2 ∧ x₃ = (5 + Complex.I * Real.sqrt 39) / 24 ∧ x₄ = (5 - Complex.I * Real.sqrt 39) / 24) ∧
    (∀ x : ℂ, (12*x - 1)*(6*x - 1)*(4*x - 1)*(3*x - 1) = 5 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2195_219532


namespace NUMINAMATH_CALUDE_sequence_growth_l2195_219567

theorem sequence_growth (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l2195_219567


namespace NUMINAMATH_CALUDE_andrew_stamping_rate_l2195_219521

/-- Andrew's work schedule and permit stamping rate -/
def andrew_schedule (appointments : ℕ) (appointment_duration : ℕ) (workday_length : ℕ) (total_permits : ℕ) : ℕ :=
  let time_in_appointments := appointments * appointment_duration
  let time_stamping := workday_length - time_in_appointments
  total_permits / time_stamping

/-- Theorem stating Andrew's permit stamping rate given his schedule -/
theorem andrew_stamping_rate :
  andrew_schedule 2 3 8 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_andrew_stamping_rate_l2195_219521


namespace NUMINAMATH_CALUDE_equilateral_is_isosceles_l2195_219539

-- Define a triangle type
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

-- Define what it means for a triangle to be equilateral
def IsEquilateral (t : Triangle) : Prop :=
  t.side1 = t.side2 ∧ t.side2 = t.side3

-- Define what it means for a triangle to be isosceles
def IsIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side3 = t.side1

-- Theorem: Every equilateral triangle is isosceles
theorem equilateral_is_isosceles (t : Triangle) :
  IsEquilateral t → IsIsosceles t := by
  sorry


end NUMINAMATH_CALUDE_equilateral_is_isosceles_l2195_219539


namespace NUMINAMATH_CALUDE_assign_roles_for_five_men_six_women_l2195_219560

/-- The number of ways to assign roles in a play --/
def assignRoles (numMen numWomen : ℕ) : ℕ :=
  let maleRoles := 2
  let femaleRoles := 2
  let eitherGenderRoles := 2
  let remainingActors := numMen + numWomen - maleRoles - femaleRoles
  (numMen.descFactorial maleRoles) *
  (numWomen.descFactorial femaleRoles) *
  (remainingActors.descFactorial eitherGenderRoles)

/-- Theorem stating the number of ways to assign roles for 5 men and 6 women --/
theorem assign_roles_for_five_men_six_women :
  assignRoles 5 6 = 25200 := by
  sorry

end NUMINAMATH_CALUDE_assign_roles_for_five_men_six_women_l2195_219560


namespace NUMINAMATH_CALUDE_problem_solution_l2195_219506

theorem problem_solution : ∃ x : ℝ, 70 + 5 * 12 / (x / 3) = 71 ∧ x = 180 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2195_219506


namespace NUMINAMATH_CALUDE_count_polynomials_l2195_219509

-- Define a function to check if an expression is a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/(5x)" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["3/4x^2", "3ab", "x+5", "y/(5x)", "-1", "y/3", "a^2-b^2", "a"]

-- Theorem: There are exactly 7 polynomials in the list of expressions
theorem count_polynomials :
  (expressions.filter is_polynomial).length = 7 :=
by sorry

end NUMINAMATH_CALUDE_count_polynomials_l2195_219509


namespace NUMINAMATH_CALUDE_calculation_proof_l2195_219513

theorem calculation_proof : (180 : ℚ) / (15 + 12 * 3 - 9) = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2195_219513


namespace NUMINAMATH_CALUDE_division_problem_l2195_219587

theorem division_problem (Ω : ℕ) : 
  Ω ≤ 9 ∧ Ω ≥ 1 →
  (∃ (n : ℕ), n ≥ 10 ∧ n < 50 ∧ 504 / Ω = n + 2 * Ω) →
  Ω = 7 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2195_219587


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_condition_l2195_219581

def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 3

theorem f_inequality_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 1/2} :=
sorry

theorem f_minimum_value_condition (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_condition_l2195_219581


namespace NUMINAMATH_CALUDE_sum_to_base3_l2195_219538

def base10_to_base3 (n : ℕ) : List ℕ :=
  sorry

theorem sum_to_base3 :
  base10_to_base3 (36 + 25 + 2) = [2, 1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_to_base3_l2195_219538


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l2195_219549

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (1/2) * diameter * r
  max_triangle_area = 64 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l2195_219549


namespace NUMINAMATH_CALUDE_double_burger_cost_l2195_219590

/-- Proves that the cost of a double burger is $1.50 given the specified conditions -/
theorem double_burger_cost (total_spent : ℚ) (total_hamburgers : ℕ) (double_burgers : ℕ) (single_burger_cost : ℚ) :
  total_spent = 70.5 ∧
  total_hamburgers = 50 ∧
  double_burgers = 41 ∧
  single_burger_cost = 1 →
  (total_spent - (total_hamburgers - double_burgers : ℚ) * single_burger_cost) / double_burgers = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_double_burger_cost_l2195_219590


namespace NUMINAMATH_CALUDE_ratio_problem_l2195_219555

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : c / b = 3)
  (h3 : c / d = 2) :
  d / a = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2195_219555


namespace NUMINAMATH_CALUDE_mean_variance_relationship_l2195_219522

-- Define the sample size
def sample_size : Nat := 50

-- Define the original mean and variance
def original_mean : Real := 70
def original_variance : Real := 75

-- Define the incorrect and correct data points
def incorrect_point1 : Real := 60
def incorrect_point2 : Real := 90
def correct_point1 : Real := 80
def correct_point2 : Real := 70

-- Define the new mean and variance after correction
def new_mean : Real := original_mean
noncomputable def new_variance : Real := original_variance - 8

-- Theorem statement
theorem mean_variance_relationship :
  new_mean = original_mean ∧ new_variance < original_variance :=
by sorry

end NUMINAMATH_CALUDE_mean_variance_relationship_l2195_219522


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2195_219519

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b - a)^2 * (c^2 + 4*a*b)^2 ≤ 2*c^6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2195_219519


namespace NUMINAMATH_CALUDE_angle_range_l2195_219551

def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b : Fin 2 → ℝ := ![1, 3]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem angle_range (x : ℝ) :
  is_acute_angle (a x) b → x ∈ {y : ℝ | y > -2/3 ∧ y ≠ -2/3} := by
  sorry

end NUMINAMATH_CALUDE_angle_range_l2195_219551


namespace NUMINAMATH_CALUDE_kitten_weight_l2195_219598

theorem kitten_weight (k r p : ℝ) 
  (total_weight : k + r + p = 38)
  (kitten_rabbit_weight : k + r = 3 * p)
  (kitten_parrot_weight : k + p = r) :
  k = 9.5 := by
sorry

end NUMINAMATH_CALUDE_kitten_weight_l2195_219598


namespace NUMINAMATH_CALUDE_prime_sum_equation_l2195_219523

theorem prime_sum_equation (p q s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s →
  p + q = s + 4 →
  1 < p →
  p < q →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equation_l2195_219523


namespace NUMINAMATH_CALUDE_acute_angles_insufficient_for_congruence_l2195_219520

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- leg
  b : ℝ  -- leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define equality of acute angles
def equal_acute_angles (t1 t2 : RightTriangle) : Prop :=
  Real.arctan (t1.a / t1.b) = Real.arctan (t2.a / t2.b) ∧
  Real.arctan (t1.b / t1.a) = Real.arctan (t2.b / t2.a)

-- Theorem statement
theorem acute_angles_insufficient_for_congruence :
  ∃ (t1 t2 : RightTriangle), equal_acute_angles t1 t2 ∧ ¬congruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_acute_angles_insufficient_for_congruence_l2195_219520


namespace NUMINAMATH_CALUDE_petes_number_l2195_219573

theorem petes_number : ∃ x : ℚ, 5 * (3 * x + 15) = 245 ∧ x = 34 / 3 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2195_219573


namespace NUMINAMATH_CALUDE_remaining_macaroons_weight_l2195_219533

theorem remaining_macaroons_weight
  (coconut_count : ℕ) (coconut_weight : ℕ) (almond_count : ℕ) (almond_weight : ℕ)
  (coconut_bags : ℕ) (almond_bags : ℕ) :
  coconut_count = 12 →
  coconut_weight = 5 →
  almond_count = 8 →
  almond_weight = 8 →
  coconut_bags = 4 →
  almond_bags = 2 →
  (coconut_count * coconut_weight - (coconut_count / coconut_bags) * coconut_weight) +
  (almond_count * almond_weight - (almond_count / almond_bags) * almond_weight / 2) = 93 :=
by sorry

end NUMINAMATH_CALUDE_remaining_macaroons_weight_l2195_219533


namespace NUMINAMATH_CALUDE_lanas_winter_clothing_l2195_219534

/-- The number of boxes Lana found -/
def num_boxes : ℕ := 5

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 7

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 8

/-- The total number of pieces of winter clothing Lana had -/
def total_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

theorem lanas_winter_clothing : total_clothing = 75 := by
  sorry

end NUMINAMATH_CALUDE_lanas_winter_clothing_l2195_219534


namespace NUMINAMATH_CALUDE_side_face_area_l2195_219546

/-- A rectangular box with specific properties -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
  front_face_half_top : width * height = (length * width) / 2
  top_face_one_half_side : length * width = (3 * length * height) / 2
  volume : length * width * height = 5184
  perimeter_ratio : 2 * (length + height) = (12 * (length + width)) / 10

/-- The area of the side face of a box with the given properties is 384 square units -/
theorem side_face_area (b : Box) : b.length * b.height = 384 := by
  sorry

end NUMINAMATH_CALUDE_side_face_area_l2195_219546
