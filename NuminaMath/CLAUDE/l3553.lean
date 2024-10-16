import Mathlib

namespace NUMINAMATH_CALUDE_lcm_of_8_12_15_l3553_355319

theorem lcm_of_8_12_15 : Nat.lcm (Nat.lcm 8 12) 15 = 120 := by sorry

end NUMINAMATH_CALUDE_lcm_of_8_12_15_l3553_355319


namespace NUMINAMATH_CALUDE_altitude_length_of_triangle_on_rectangle_diagonal_l3553_355325

/-- Given a rectangle with sides a and b, and a triangle constructed on its diagonal
    with an area equal to the rectangle's area, the length of the altitude drawn to
    the base (diagonal) of the triangle is (2ab) / √(a² + b²). -/
theorem altitude_length_of_triangle_on_rectangle_diagonal
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let rectangle_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := (1/2) * diagonal * (2 * rectangle_area / diagonal)
  triangle_area = rectangle_area →
  (2 * rectangle_area / diagonal) = (2 * a * b) / Real.sqrt (a^2 + b^2) := by
sorry


end NUMINAMATH_CALUDE_altitude_length_of_triangle_on_rectangle_diagonal_l3553_355325


namespace NUMINAMATH_CALUDE_three_digit_subtraction_convergence_l3553_355379

-- Define a three-digit number type
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n ≤ 999 }

-- Function to reverse a three-digit number
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber := sorry

-- Function to perform one step of the operation
def step (n : ThreeDigitNumber) : ThreeDigitNumber := sorry

-- Define the set of possible results
def ResultSet : Set ℕ := {0, 495}

-- Theorem statement
theorem three_digit_subtraction_convergence (start : ThreeDigitNumber) :
  ∃ (k : ℕ), (step^[k] start).val ∈ ResultSet := sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_convergence_l3553_355379


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3553_355385

theorem sufficient_not_necessary : 
  (∀ X Y : ℝ, X > 2 ∧ Y > 3 → X + Y > 5 ∧ X * Y > 6) ∧ 
  (∃ X Y : ℝ, X + Y > 5 ∧ X * Y > 6 ∧ ¬(X > 2 ∧ Y > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3553_355385


namespace NUMINAMATH_CALUDE_jenny_reads_three_books_l3553_355342

/-- Represents the number of books Jenny can read given the conditions --/
def books_jenny_can_read (days : ℕ) (reading_speed : ℕ) (reading_time : ℚ) 
  (book1_words : ℕ) (book2_words : ℕ) (book3_words : ℕ) : ℕ :=
  let total_words := book1_words + book2_words + book3_words
  let total_reading_hours := (days : ℚ) * reading_time
  let words_read := (reading_speed : ℚ) * total_reading_hours
  if words_read ≥ total_words then 3 else 
    if words_read ≥ book1_words + book2_words then 2 else
      if words_read ≥ book1_words then 1 else 0

/-- Theorem stating that Jenny can read exactly 3 books in 10 days --/
theorem jenny_reads_three_books : 
  books_jenny_can_read 10 100 (54/60) 200 400 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jenny_reads_three_books_l3553_355342


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3553_355372

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | 3 * x^2 - 4 * x + 2 = -x^2 + 2 * x + 3}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y (x : ℝ) : ℝ :=
  3 * x^2 - 4 * x + 2

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ :=
  3 * x^2 - 4 * x + 2

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ :=
  -x^2 + 2 * x + 3

theorem parabolas_intersection :
  intersection_x = {(3 - Real.sqrt 13) / 4, (3 + Real.sqrt 13) / 4} ∧
  ∀ x ∈ intersection_x, intersection_y x = (74 + 14 * Real.sqrt 13 * (if x < 0 then -1 else 1)) / 16 ∧
  ∀ x : ℝ, parabola1 x = parabola2 x ↔ x ∈ intersection_x :=
by sorry


end NUMINAMATH_CALUDE_parabolas_intersection_l3553_355372


namespace NUMINAMATH_CALUDE_taco_castle_parking_lot_l3553_355399

/-- The number of Dodge trucks in the Taco Castle parking lot -/
def dodge_trucks : ℕ := 60

/-- The number of Ford trucks in the Taco Castle parking lot -/
def ford_trucks : ℕ := dodge_trucks / 3

/-- The number of Toyota trucks in the Taco Castle parking lot -/
def toyota_trucks : ℕ := ford_trucks / 2

/-- The number of Volkswagen Bugs in the Taco Castle parking lot -/
def volkswagen_bugs : ℕ := 5

theorem taco_castle_parking_lot :
  dodge_trucks = 60 ∧
  ford_trucks = dodge_trucks / 3 ∧
  ford_trucks = toyota_trucks * 2 ∧
  volkswagen_bugs = toyota_trucks / 2 ∧
  volkswagen_bugs = 5 :=
by sorry

end NUMINAMATH_CALUDE_taco_castle_parking_lot_l3553_355399


namespace NUMINAMATH_CALUDE_employee_count_l3553_355364

theorem employee_count (avg_salary : ℕ) (salary_increase : ℕ) (manager_salary : ℕ) :
  avg_salary = 1500 →
  salary_increase = 500 →
  manager_salary = 12000 →
  ∃ n : ℕ, n * avg_salary + manager_salary = (n + 1) * (avg_salary + salary_increase) ∧ n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_count_l3553_355364


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l3553_355332

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l3553_355332


namespace NUMINAMATH_CALUDE_equation_solutions_l3553_355381

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 5 = 10) ∧
  (∃ x : ℚ, 2 * x + 4 * (2 * x - 3) = 6 - 2 * (x + 1)) :=
by
  constructor
  · use 5
    norm_num
  · use 4/3
    norm_num
    
#check equation_solutions

end NUMINAMATH_CALUDE_equation_solutions_l3553_355381


namespace NUMINAMATH_CALUDE_largest_multiple_under_1000_l3553_355351

theorem largest_multiple_under_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  n < 1000 ∧
  ∀ m : ℕ, (m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_under_1000_l3553_355351


namespace NUMINAMATH_CALUDE_max_area_rectangular_prism_volume_l3553_355305

/-- The volume of a rectangular prism with maximum base area -/
theorem max_area_rectangular_prism_volume
  (base_perimeter : ℝ)
  (height : ℝ)
  (h_base_perimeter : base_perimeter = 32)
  (h_height : height = 9)
  (h_max_area : ∀ (l w : ℝ), l + w = base_perimeter / 2 → l * w ≤ (base_perimeter / 4) ^ 2) :
  (base_perimeter / 4) ^ 2 * height = 576 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangular_prism_volume_l3553_355305


namespace NUMINAMATH_CALUDE_sunday_temp_is_98_1_l3553_355389

/-- Given a list of 6 temperatures and a weekly average for 7 days, 
    calculate the 7th temperature (Sunday) -/
def calculate_sunday_temp (temps : List Float) (weekly_avg : Float) : Float :=
  7 * weekly_avg - temps.sum

/-- Theorem stating that given the specific temperatures and weekly average,
    the Sunday temperature is 98.1 -/
theorem sunday_temp_is_98_1 : 
  let temps := [98.2, 98.7, 99.3, 99.8, 99, 98.9]
  let weekly_avg := 99
  calculate_sunday_temp temps weekly_avg = 98.1 := by
  sorry

#eval calculate_sunday_temp [98.2, 98.7, 99.3, 99.8, 99, 98.9] 99

end NUMINAMATH_CALUDE_sunday_temp_is_98_1_l3553_355389


namespace NUMINAMATH_CALUDE_problem_solution_l3553_355388

theorem problem_solution : (((3⁻¹ : ℚ) - 2 + 6^2 + 1)⁻¹ * 6 : ℚ) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3553_355388


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3553_355346

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ x y, p x y ↔ x ∣ y) →
  p 2 (3^19 + 11^13) ∧ 
  (∀ q, q < 2 → q.Prime → ¬p q (3^19 + 11^13)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3553_355346


namespace NUMINAMATH_CALUDE_min_dot_product_op_ab_l3553_355308

open Real

/-- The minimum dot product of OP and AB -/
theorem min_dot_product_op_ab :
  ∀ x : ℝ,
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, 1)
  let P : ℝ × ℝ := (x, exp x)
  let OP : ℝ × ℝ := P
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (OP.1 * AB.1 + OP.2 * AB.2) ≥ 1 :=
by sorry

#check min_dot_product_op_ab

end NUMINAMATH_CALUDE_min_dot_product_op_ab_l3553_355308


namespace NUMINAMATH_CALUDE_unique_valid_number_l3553_355326

def is_valid_number (n : ℕ) : Prop :=
  350000 ≤ n ∧ n ≤ 359992 ∧ n % 100 = 2 ∧ n % 6 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 351152 := by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3553_355326


namespace NUMINAMATH_CALUDE_difference_of_fractions_l3553_355324

theorem difference_of_fractions : (7 / 8 : ℚ) * 320 - (11 / 16 : ℚ) * 144 = 181 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l3553_355324


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3553_355384

/-- Given a tetrahedron with inradius R and face areas S₁, S₂, S₃, and S₄,
    its volume V is equal to (1/3)R(S₁ + S₂ + S₃ + S₄) -/
theorem tetrahedron_volume (R S₁ S₂ S₃ S₄ : ℝ) (hR : R > 0) (hS : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0) :
  ∃ V : ℝ, V = (1/3) * R * (S₁ + S₂ + S₃ + S₄) ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3553_355384


namespace NUMINAMATH_CALUDE_average_age_proof_l3553_355358

/-- Given the ages of John, Mary, and Tonya with specific relationships, prove their average age --/
theorem average_age_proof (tonya mary john : ℕ) 
  (h1 : john = 2 * mary)
  (h2 : john * 2 = tonya)
  (h3 : tonya = 60) : 
  (tonya + john + mary) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l3553_355358


namespace NUMINAMATH_CALUDE_baseball_price_proof_l3553_355353

/-- The price of a basketball in dollars -/
def basketball_price : ℝ := 29

/-- The number of basketballs bought by Coach A -/
def num_basketballs : ℕ := 10

/-- The number of baseballs bought by Coach B -/
def num_baseballs : ℕ := 14

/-- The price of the baseball bat in dollars -/
def bat_price : ℝ := 18

/-- The difference in spending between Coach A and Coach B in dollars -/
def spending_difference : ℝ := 237

/-- The price of a baseball in dollars -/
def baseball_price : ℝ := 2.5

theorem baseball_price_proof :
  num_basketballs * basketball_price = 
  num_baseballs * baseball_price + bat_price + spending_difference :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_price_proof_l3553_355353


namespace NUMINAMATH_CALUDE_cape_may_shark_count_l3553_355396

/-- The number of sharks in Daytona Beach -/
def daytona_sharks : ℕ := 12

/-- The number of sharks in Cape May -/
def cape_may_sharks : ℕ := 2 * daytona_sharks + 8

theorem cape_may_shark_count : cape_may_sharks = 32 := by
  sorry

end NUMINAMATH_CALUDE_cape_may_shark_count_l3553_355396


namespace NUMINAMATH_CALUDE_perfect_square_problem_l3553_355355

theorem perfect_square_problem :
  (∃ x : ℝ, 6^2024 = x^2) ∧
  (∀ y : ℝ, 7^2025 ≠ y^2) ∧
  (∃ z : ℝ, 8^2026 = z^2) ∧
  (∃ w : ℝ, 9^2027 = w^2) ∧
  (∃ v : ℝ, 10^2028 = v^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_problem_l3553_355355


namespace NUMINAMATH_CALUDE_number_of_trucks_filled_l3553_355309

/-- Prove that the number of trucks filled up is 2, given the specified conditions. -/
theorem number_of_trucks_filled (service_cost : ℚ) (fuel_cost_per_liter : ℚ) (total_cost : ℚ)
  (num_minivans : ℕ) (minivan_capacity : ℚ) (truck_capacity_factor : ℚ) :
  service_cost = 23/10 →
  fuel_cost_per_liter = 7/10 →
  total_cost = 396 →
  num_minivans = 4 →
  minivan_capacity = 65 →
  truck_capacity_factor = 22/10 →
  ∃ (num_trucks : ℕ), num_trucks = 2 ∧
    total_cost = (num_minivans * (service_cost + fuel_cost_per_liter * minivan_capacity)) +
                 (num_trucks * (service_cost + fuel_cost_per_liter * (minivan_capacity * truck_capacity_factor))) :=
by sorry


end NUMINAMATH_CALUDE_number_of_trucks_filled_l3553_355309


namespace NUMINAMATH_CALUDE_sam_has_two_nickels_l3553_355393

/-- Represents the types of coins in Sam's wallet -/
inductive Coin
  | Penny
  | Nickel
  | Dime

/-- The value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10

/-- Represents Sam's wallet -/
structure Wallet :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)

/-- The total value of coins in the wallet in cents -/
def totalValue (w : Wallet) : Nat :=
  w.pennies * coinValue Coin.Penny +
  w.nickels * coinValue Coin.Nickel +
  w.dimes * coinValue Coin.Dime

/-- The total number of coins in the wallet -/
def totalCoins (w : Wallet) : Nat :=
  w.pennies + w.nickels + w.dimes

/-- The average value of coins in the wallet in cents -/
def averageValue (w : Wallet) : Rat :=
  (totalValue w : Rat) / (totalCoins w : Rat)

theorem sam_has_two_nickels (w : Wallet) 
  (h1 : averageValue w = 15)
  (h2 : averageValue { pennies := w.pennies, nickels := w.nickels, dimes := w.dimes + 1 } = 16) :
  w.nickels = 2 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_two_nickels_l3553_355393


namespace NUMINAMATH_CALUDE_product_sum_7293_l3553_355310

theorem product_sum_7293 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a * b = 7293 ∧ 
  a + b = 114 := by
sorry

end NUMINAMATH_CALUDE_product_sum_7293_l3553_355310


namespace NUMINAMATH_CALUDE_neo_tokropolis_population_change_is_40_l3553_355390

/-- Represents the population change in Neo-Tokropolis over a month -/
def neo_tokropolis_population_change : ℚ :=
  let births_per_day : ℚ := 24 / 12
  let deaths_per_day : ℚ := 24 / 36
  let net_change_per_day : ℚ := births_per_day - deaths_per_day
  let days_in_month : ℚ := 30
  net_change_per_day * days_in_month

/-- Theorem stating that the population change in Neo-Tokropolis over a month is 40 -/
theorem neo_tokropolis_population_change_is_40 :
  neo_tokropolis_population_change = 40 := by
  sorry

#eval neo_tokropolis_population_change

end NUMINAMATH_CALUDE_neo_tokropolis_population_change_is_40_l3553_355390


namespace NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l3553_355354

theorem positive_integer_pairs_satisfying_equation :
  ∀ x y : ℕ+, 
    (x : ℤ)^2 + (y : ℤ)^2 - 5*(x : ℤ)*(y : ℤ) + 5 = 0 ↔ 
    ((x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2)) :=
by sorry

#check positive_integer_pairs_satisfying_equation

end NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l3553_355354


namespace NUMINAMATH_CALUDE_fermat_number_prime_divisors_l3553_355344

theorem fermat_number_prime_divisors (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ 2^(2^n) + 1) : 
  ∃ k : ℕ, p = k * 2^(n + 1) + 1 := by
sorry

end NUMINAMATH_CALUDE_fermat_number_prime_divisors_l3553_355344


namespace NUMINAMATH_CALUDE_chuzhou_gdp_scientific_notation_l3553_355323

/-- The GDP of Chuzhou City in 2022 in billions of yuan -/
def chuzhou_gdp : ℝ := 3600

/-- Conversion factor from billion to scientific notation -/
def billion_to_scientific : ℝ := 10^9

theorem chuzhou_gdp_scientific_notation :
  chuzhou_gdp * billion_to_scientific = 3.6 * 10^12 := by
  sorry

end NUMINAMATH_CALUDE_chuzhou_gdp_scientific_notation_l3553_355323


namespace NUMINAMATH_CALUDE_total_coins_count_l3553_355338

theorem total_coins_count (dimes nickels quarters : ℕ) : 
  dimes = 2 → nickels = 2 → quarters = 7 → dimes + nickels + quarters = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_count_l3553_355338


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3553_355386

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3553_355386


namespace NUMINAMATH_CALUDE_binomial_18_10_l3553_355350

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l3553_355350


namespace NUMINAMATH_CALUDE_fifth_root_equality_l3553_355322

theorem fifth_root_equality : ∃ (x y : ℤ), (119287 - 48682 * Real.sqrt 6) ^ (1/5 : ℝ) = x + y * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_fifth_root_equality_l3553_355322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3553_355339

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 5 + a 6 = 42 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3553_355339


namespace NUMINAMATH_CALUDE_f_has_two_roots_l3553_355303

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem f_has_two_roots : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end NUMINAMATH_CALUDE_f_has_two_roots_l3553_355303


namespace NUMINAMATH_CALUDE_binomial_n_minus_two_l3553_355391

theorem binomial_n_minus_two (n : ℕ+) : Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_minus_two_l3553_355391


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l3553_355315

variables (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

theorem polynomial_coefficients :
  (x + 2) * (x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 →
  a₂ = 8 ∧ a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l3553_355315


namespace NUMINAMATH_CALUDE_find_h_of_x_l3553_355300

theorem find_h_of_x (x : ℝ) (h : ℝ → ℝ) : 
  (9 * x^3 - 3 * x + 1 + h x = 3 * x^2 - 5 * x + 3) → 
  (h x = -9 * x^3 + 3 * x^2 - 2 * x + 2) := by
sorry

end NUMINAMATH_CALUDE_find_h_of_x_l3553_355300


namespace NUMINAMATH_CALUDE_day_305_is_thursday_l3553_355380

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the number of days after Wednesday -/
def daysAfterWednesday (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Wednesday
  | 1 => DayOfWeek.Thursday
  | 2 => DayOfWeek.Friday
  | 3 => DayOfWeek.Saturday
  | 4 => DayOfWeek.Sunday
  | 5 => DayOfWeek.Monday
  | _ => DayOfWeek.Tuesday

theorem day_305_is_thursday :
  daysAfterWednesday (305 - 17) = DayOfWeek.Thursday := by
  sorry

#check day_305_is_thursday

end NUMINAMATH_CALUDE_day_305_is_thursday_l3553_355380


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3553_355316

theorem complex_equation_solution (a : ℝ) :
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3553_355316


namespace NUMINAMATH_CALUDE_smallest_largest_8digit_multiples_of_360_l3553_355373

/-- Checks if a number has all unique digits --/
def hasUniqueDigits (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- Checks if a number is a multiple of 360 --/
def isMultipleOf360 (n : Nat) : Bool :=
  n % 360 = 0

/-- Theorem: 12378960 and 98763120 are the smallest and largest 8-digit multiples of 360 with unique digits --/
theorem smallest_largest_8digit_multiples_of_360 :
  (∀ n : Nat, n ≥ 10000000 ∧ n < 100000000 ∧ isMultipleOf360 n ∧ hasUniqueDigits n →
    n ≥ 12378960) ∧
  (∀ n : Nat, n ≥ 10000000 ∧ n < 100000000 ∧ isMultipleOf360 n ∧ hasUniqueDigits n →
    n ≤ 98763120) ∧
  isMultipleOf360 12378960 ∧
  isMultipleOf360 98763120 ∧
  hasUniqueDigits 12378960 ∧
  hasUniqueDigits 98763120 :=
by sorry


end NUMINAMATH_CALUDE_smallest_largest_8digit_multiples_of_360_l3553_355373


namespace NUMINAMATH_CALUDE_smallest_two_digit_number_divisible_by_170_l3553_355382

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem smallest_two_digit_number_divisible_by_170 :
  ∃ (N : ℕ), is_two_digit N ∧
  (sum_of_digits (10^N - N) % 170 = 0) ∧
  (∀ (M : ℕ), is_two_digit M → sum_of_digits (10^M - M) % 170 = 0 → N ≤ M) ∧
  N = 20 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_number_divisible_by_170_l3553_355382


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3553_355330

def i : ℂ := Complex.I

theorem point_in_fourth_quadrant :
  let z : ℂ := (5 - i) / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3553_355330


namespace NUMINAMATH_CALUDE_square_side_length_equal_area_l3553_355394

/-- The side length of a square with the same area as a rectangle with length 18 and width 8 is 12 -/
theorem square_side_length_equal_area (length width : ℝ) (x : ℝ) :
  length = 18 →
  width = 8 →
  x ^ 2 = length * width →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_equal_area_l3553_355394


namespace NUMINAMATH_CALUDE_certain_number_problem_l3553_355340

theorem certain_number_problem (y : ℝ) : 
  (0.25 * 900 = 0.15 * y - 15) → y = 1600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3553_355340


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3553_355314

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4)^2 → area = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3553_355314


namespace NUMINAMATH_CALUDE_fraction_proof_l3553_355302

theorem fraction_proof (N : ℝ) (F : ℝ) (h1 : N = 8) (h2 : 0.5 * N = F * N + 2) : F = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l3553_355302


namespace NUMINAMATH_CALUDE_parabola_directrix_l3553_355345

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 8 * x^2

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -2

/-- Theorem stating that the given directrix equation is correct for the given parabola -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → (∃ (d : ℝ), directrix_equation d ∧ 
    -- Additional conditions to relate the parabola and directrix
    (x^2 + (y - 2)^2 = (y + 2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3553_355345


namespace NUMINAMATH_CALUDE_tourist_groups_speed_l3553_355336

theorem tourist_groups_speed : ∀ (x y : ℝ),
  (x > 0 ∧ y > 0) →  -- Speeds are positive
  (4.5 * x + 2.5 * y = 30) →  -- First scenario equation
  (3 * x + 5 * y = 30) →  -- Second scenario equation
  (x = 5 ∧ y = 3) :=  -- Speeds of the two groups
by sorry

end NUMINAMATH_CALUDE_tourist_groups_speed_l3553_355336


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3553_355371

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 4 < 0 ↔ -1 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3553_355371


namespace NUMINAMATH_CALUDE_b_value_function_comparison_l3553_355360

/-- A quadratic function with the given properties -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry property of the function -/
axiom symmetry_property (b c : ℝ) : ∀ x : ℝ, f b c (2 + x) = f b c (2 - x)

/-- The value of b in the quadratic function -/
theorem b_value : ∃ b : ℝ, (∀ c x : ℝ, f b c (2 + x) = f b c (2 - x)) ∧ b = 4 :=
sorry

/-- Comparison of function values -/
theorem function_comparison (b c : ℝ) 
  (h : ∀ x : ℝ, f b c (2 + x) = f b c (2 - x)) : 
  ∀ a : ℝ, f b c (5/4) < f b c (-a^2 - a + 1) :=
sorry

end NUMINAMATH_CALUDE_b_value_function_comparison_l3553_355360


namespace NUMINAMATH_CALUDE_exists_min_value_subject_to_constraint_l3553_355398

/-- The constraint function for a, b, c, d -/
def constraint (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

/-- The function to be minimized -/
def objective (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

/-- Theorem stating the existence of a minimum value for the objective function
    subject to the given constraint -/
theorem exists_min_value_subject_to_constraint :
  ∃ (min : ℝ), ∀ (a b c d : ℝ), constraint a b c d →
    objective a b c d ≥ min ∧
    (∃ (a' b' c' d' : ℝ), constraint a' b' c' d' ∧ objective a' b' c' d' = min) :=
by sorry

end NUMINAMATH_CALUDE_exists_min_value_subject_to_constraint_l3553_355398


namespace NUMINAMATH_CALUDE_international_long_haul_all_services_probability_l3553_355392

/-- Represents a flight route -/
inductive FlightRoute
| Domestic
| InternationalShortHaul
| InternationalLongHaul

/-- Represents a service offered on a flight -/
inductive Service
| WirelessInternet
| FreeSnacks
| EntertainmentSystem
| ExtraLegroom

/-- Returns the probability of a service being offered on a given flight route -/
def serviceProbability (route : FlightRoute) (service : Service) : ℝ :=
  match route, service with
  | FlightRoute.InternationalLongHaul, Service.WirelessInternet => 0.65
  | FlightRoute.InternationalLongHaul, Service.FreeSnacks => 0.80
  | FlightRoute.InternationalLongHaul, Service.EntertainmentSystem => 0.75
  | FlightRoute.InternationalLongHaul, Service.ExtraLegroom => 0.70
  | _, _ => 0  -- Default case, not used in this problem

/-- The probability of experiencing all services on a given flight route -/
def allServicesProbability (route : FlightRoute) : ℝ :=
  (serviceProbability route Service.WirelessInternet) *
  (serviceProbability route Service.FreeSnacks) *
  (serviceProbability route Service.EntertainmentSystem) *
  (serviceProbability route Service.ExtraLegroom)

/-- Theorem: The probability of experiencing all services on an international long-haul flight is 0.273 -/
theorem international_long_haul_all_services_probability :
  allServicesProbability FlightRoute.InternationalLongHaul = 0.273 := by
  sorry

end NUMINAMATH_CALUDE_international_long_haul_all_services_probability_l3553_355392


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3553_355369

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  a : ℝ  -- semi-major axis
  f1 : Point  -- left focus
  f2 : Point  -- right focus

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- The main theorem -/
theorem hyperbola_parabola_intersection (h : Hyperbola) (p : Parabola) (P : Point) (a c : ℝ) :
  h.f1 = p.focus →
  h.f2 = p.vertex →
  (P.x - h.f2.x) ^ 2 + P.y ^ 2 = (h.e * h.a) ^ 2 →  -- P is on the right branch of the hyperbola
  P.y ^ 2 = 2 * h.a * (P.x - h.f2.x) →  -- P is on the parabola
  a * |P.x - h.f2.x| + c * |h.f1.x - P.x| = 8 * a ^ 2 →
  h.e = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3553_355369


namespace NUMINAMATH_CALUDE_range_of_negative_two_a_plus_three_l3553_355334

theorem range_of_negative_two_a_plus_three (a : ℝ) : 
  a < 1 → -2*a + 3 > 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_negative_two_a_plus_three_l3553_355334


namespace NUMINAMATH_CALUDE_james_bike_ride_l3553_355341

theorem james_bike_ride (first_hour : ℝ) : 
  first_hour > 0 →
  let second_hour := 1.2 * first_hour
  let third_hour := 1.25 * second_hour
  first_hour + second_hour + third_hour = 55.5 →
  second_hour = 18 := by
sorry

end NUMINAMATH_CALUDE_james_bike_ride_l3553_355341


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_four_ninety_eight_squared_minus_four_exists_n_for_equation_three_nine_nine_nine_nine_nine_one_not_prime_l3553_355363

-- Part (a)
theorem factor_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) := by sorry

-- Part (b)
theorem ninety_eight_squared_minus_four : 98^2 - 4 = 100 * 96 := by sorry

-- Part (c)
theorem exists_n_for_equation : ∃ n : ℕ+, (20 - n) * (20 + n) = 391 ∧ n = 3 := by sorry

-- Part (d)
theorem three_nine_nine_nine_nine_nine_one_not_prime : ¬ Nat.Prime 3999991 := by sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_four_ninety_eight_squared_minus_four_exists_n_for_equation_three_nine_nine_nine_nine_nine_one_not_prime_l3553_355363


namespace NUMINAMATH_CALUDE_negation_of_implication_l3553_355348

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3553_355348


namespace NUMINAMATH_CALUDE_sqrt_negative_square_defined_unique_l3553_355311

theorem sqrt_negative_square_defined_unique : 
  ∃! a : ℝ, ∃ x : ℝ, x^2 = -(1-a)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_square_defined_unique_l3553_355311


namespace NUMINAMATH_CALUDE_estevan_blankets_l3553_355397

theorem estevan_blankets (initial_blankets : ℕ) : 
  (initial_blankets / 3 : ℚ) + 2 = 10 → initial_blankets = 24 := by
  sorry

end NUMINAMATH_CALUDE_estevan_blankets_l3553_355397


namespace NUMINAMATH_CALUDE_waiter_customers_theorem_l3553_355365

/-- Calculates the final number of customers for a waiter --/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Proves that the final number of customers is correct --/
theorem waiter_customers_theorem (initial left new : ℕ) 
  (h1 : initial ≥ left) : 
  final_customers initial left new = initial - left + new :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_theorem_l3553_355365


namespace NUMINAMATH_CALUDE_cubic_range_l3553_355343

theorem cubic_range (x : ℝ) (h : x^2 - 5*x + 6 < 0) :
  41 < x^3 + 5*x^2 + 6*x + 1 ∧ x^3 + 5*x^2 + 6*x + 1 < 91 := by
  sorry

end NUMINAMATH_CALUDE_cubic_range_l3553_355343


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l3553_355361

/-- The perimeter of the shape ABFCDE formed by cutting a right triangle from a square and
    repositioning it on the left side of the square. -/
theorem perimeter_of_modified_square (
  square_perimeter : ℝ)
  (triangle_leg : ℝ)
  (h1 : square_perimeter = 48)
  (h2 : triangle_leg = 12) : ℝ :=
by
  -- The perimeter of the new shape ABFCDE is 60 inches
  sorry

#check perimeter_of_modified_square

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l3553_355361


namespace NUMINAMATH_CALUDE_beetle_speed_l3553_355395

/-- Given an ant's average speed and a beetle that walks 10% less distance in the same time,
    prove that the beetle's speed is 1.8 km/h. -/
theorem beetle_speed (ant_distance : ℝ) (time : ℝ) (beetle_percentage : ℝ) :
  ant_distance = 1000 →
  time = 30 →
  beetle_percentage = 0.9 →
  let beetle_distance := ant_distance * beetle_percentage
  let beetle_speed_mpm := beetle_distance / time
  let beetle_speed_kmh := beetle_speed_mpm * 2 * 0.001
  beetle_speed_kmh = 1.8 := by
sorry

end NUMINAMATH_CALUDE_beetle_speed_l3553_355395


namespace NUMINAMATH_CALUDE_percentage_of_210_l3553_355357

theorem percentage_of_210 : (33 + 1/3 : ℚ) / 100 * 210 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_210_l3553_355357


namespace NUMINAMATH_CALUDE_delta_zero_implies_c_sqrt_30_l3553_355366

def Δ (a b c : ℝ) : ℝ := c^2 - 3*a*b

theorem delta_zero_implies_c_sqrt_30 (a b c : ℝ) (h1 : Δ a b c = 0) (h2 : a = 2) (h3 : b = 5) :
  c = Real.sqrt 30 ∨ c = -Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_delta_zero_implies_c_sqrt_30_l3553_355366


namespace NUMINAMATH_CALUDE_acute_triangle_perpendicular_pyramid_l3553_355337

theorem acute_triangle_perpendicular_pyramid (a b c : ℝ) 
  (h_acute : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (x y z : ℝ), 
    x^2 + y^2 = c^2 ∧
    y^2 + z^2 = a^2 ∧
    x^2 + z^2 = b^2 ∧
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_perpendicular_pyramid_l3553_355337


namespace NUMINAMATH_CALUDE_mixed_fruit_cost_calculation_l3553_355301

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 34

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 22.666666666666668

/-- The cost per litre of mixed fruit juice -/
def mixed_fruit_cost : ℝ := 264.1764705882353

theorem mixed_fruit_cost_calculation :
  mixed_fruit_cost * mixed_fruit_volume + acai_cost * acai_volume = 
  cocktail_cost * (mixed_fruit_volume + acai_volume) := by sorry

end NUMINAMATH_CALUDE_mixed_fruit_cost_calculation_l3553_355301


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3553_355352

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3553_355352


namespace NUMINAMATH_CALUDE_permutations_of_red_l3553_355312

def word : String := "red"

theorem permutations_of_red (w : String) (h : w = word) : 
  Nat.factorial w.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_red_l3553_355312


namespace NUMINAMATH_CALUDE_g_squared_difference_l3553_355318

-- Define the function g
def g : ℝ → ℝ := λ x => 3

-- State the theorem
theorem g_squared_difference (x : ℝ) : g ((x - 1)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_squared_difference_l3553_355318


namespace NUMINAMATH_CALUDE_race_length_is_1000_l3553_355387

/-- The length of a race, given the distance covered by one runner and their remaining distance when another runner finishes. -/
def race_length (distance_covered : ℕ) (distance_remaining : ℕ) : ℕ :=
  distance_covered + distance_remaining

/-- Theorem stating that the race length is 1000 meters under the given conditions. -/
theorem race_length_is_1000 :
  let ava_covered : ℕ := 833
  let ava_remaining : ℕ := 167
  race_length ava_covered ava_remaining = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_1000_l3553_355387


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3553_355376

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 4 + a 7 + a 10 = 30) : 
  a 3 - 2 * a 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3553_355376


namespace NUMINAMATH_CALUDE_percentage_calculation_correct_l3553_355306

/-- The total number of students in the class -/
def total_students : ℕ := 30

/-- The number of students scoring in the 70%-79% range -/
def students_in_range : ℕ := 8

/-- The percentage of students scoring in the 70%-79% range -/
def percentage_in_range : ℚ := 26.67

/-- Theorem stating that the percentage of students scoring in the 70%-79% range is correct -/
theorem percentage_calculation_correct : 
  (students_in_range : ℚ) / (total_students : ℚ) * 100 = percentage_in_range := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_correct_l3553_355306


namespace NUMINAMATH_CALUDE_lucas_payment_l3553_355383

/-- Calculates the payment for window cleaning based on given conditions --/
def calculate_payment (windows_per_floor : ℕ) (num_floors : ℕ) (payment_per_window : ℕ) 
  (penalty_per_period : ℕ) (days_per_period : ℕ) (total_days : ℕ) : ℕ :=
  let total_windows := windows_per_floor * num_floors
  let total_earned := total_windows * payment_per_window
  let num_periods := total_days / days_per_period
  let total_penalty := num_periods * penalty_per_period
  total_earned - total_penalty

theorem lucas_payment :
  calculate_payment 5 4 3 2 4 12 = 54 :=
sorry

end NUMINAMATH_CALUDE_lucas_payment_l3553_355383


namespace NUMINAMATH_CALUDE_inequality_and_equality_proof_l3553_355367

theorem inequality_and_equality_proof :
  (∀ a b : ℝ, (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * (2*a + 1) * (3*b + 1)) ∧
  (∀ n p : ℕ+, (n^2 + 1) * (p^2 + 1) + 45 = 2 * (2*n + 1) * (3*p + 1) ↔ n = 2 ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_proof_l3553_355367


namespace NUMINAMATH_CALUDE_chicken_wing_distribution_l3553_355356

theorem chicken_wing_distribution (total_wings : ℕ) (num_people : ℕ) 
  (h1 : total_wings = 35) (h2 : num_people = 12) :
  let wings_per_person := total_wings / num_people
  let leftover_wings := total_wings % num_people
  wings_per_person = 2 ∧ leftover_wings = 11 := by
  sorry

end NUMINAMATH_CALUDE_chicken_wing_distribution_l3553_355356


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3553_355333

theorem system_solution_ratio (x y c d : ℝ) 
  (eq1 : 8 * x - 6 * y = c)
  (eq2 : 9 * y - 12 * x = d)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (d_nonzero : d ≠ 0) :
  c / d = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3553_355333


namespace NUMINAMATH_CALUDE_battery_difference_proof_l3553_355329

/-- The number of batteries Tom used in flashlights -/
def flashlight_batteries : ℕ := 2

/-- The number of batteries Tom used in toys -/
def toy_batteries : ℕ := 15

/-- The difference between the number of batteries in toys and flashlights -/
def battery_difference : ℕ := toy_batteries - flashlight_batteries

theorem battery_difference_proof : battery_difference = 13 := by
  sorry

end NUMINAMATH_CALUDE_battery_difference_proof_l3553_355329


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l3553_355370

theorem average_of_three_numbers (a : ℝ) : 
  (3 + a + 10) / 3 = 5 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l3553_355370


namespace NUMINAMATH_CALUDE_water_in_sport_is_105_l3553_355313

/-- Represents the ratios of ingredients in a flavored drink formulation -/
structure DrinkFormulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the flavored drink -/
def standard : DrinkFormulation :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation of the flavored drink -/
def sport : DrinkFormulation :=
  { flavoring := standard.flavoring,
    corn_syrup := standard.corn_syrup / 3,
    water := standard.water * 2 }

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- Calculates the amount of water in the sport formulation -/
def water_in_sport : ℚ :=
  (sport_corn_syrup * sport.water) / sport.corn_syrup

/-- Theorem stating that the amount of water in the sport formulation is 105 ounces -/
theorem water_in_sport_is_105 : water_in_sport = 105 := by
  sorry


end NUMINAMATH_CALUDE_water_in_sport_is_105_l3553_355313


namespace NUMINAMATH_CALUDE_a_equals_one_l3553_355377

def star (x y : ℝ) : ℝ := x + y - x * y

theorem a_equals_one (a : ℝ) (h : a = star 1 (star 0 1)) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_l3553_355377


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3553_355349

theorem simplify_sqrt_sum : 
  (Real.sqrt 726 / Real.sqrt 81) + (Real.sqrt 294 / Real.sqrt 49) = (33 * Real.sqrt 2 + 9 * Real.sqrt 6) / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3553_355349


namespace NUMINAMATH_CALUDE_m_range_for_fourth_quadrant_l3553_355375

/-- A point in the fourth quadrant has positive x-coordinate and negative y-coordinate -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The coordinates of point M are (m+2, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (m + 2, m)

/-- Theorem stating the range of m for point M to be in the fourth quadrant -/
theorem m_range_for_fourth_quadrant :
  ∀ m : ℝ, is_in_fourth_quadrant (point_M m).1 (point_M m).2 ↔ -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_for_fourth_quadrant_l3553_355375


namespace NUMINAMATH_CALUDE_problem_solution_l3553_355304

theorem problem_solution : (2010^2 - 2010) / 2010 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3553_355304


namespace NUMINAMATH_CALUDE_remainder_of_x_50_divided_by_x2_minus_4x_plus_3_l3553_355335

theorem remainder_of_x_50_divided_by_x2_minus_4x_plus_3 :
  ∀ (x : ℝ), ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ),
    x^50 = (x^2 - 4*x + 3) * Q x + R x ∧
    (∀ (y : ℝ), R y = (3^50 - 1)/2 * y + (3 - 3^50)/2) ∧
    (∀ (y : ℝ), ∃ (a b : ℝ), R y = a * y + b) :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_x_50_divided_by_x2_minus_4x_plus_3_l3553_355335


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_progression_l3553_355307

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fifth_term_of_specific_geometric_progression :
  let a := 2 ^ (1/4 : ℝ)
  let r := 2 ^ (1/4 : ℝ)
  geometric_progression a r 1 = 2 ^ (1/4 : ℝ) ∧
  geometric_progression a r 2 = 2 ^ (1/2 : ℝ) ∧
  geometric_progression a r 3 = 2 ^ (3/4 : ℝ) →
  geometric_progression a r 5 = 2 ^ (5/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_progression_l3553_355307


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3553_355317

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3553_355317


namespace NUMINAMATH_CALUDE_alternative_plan_more_expensive_l3553_355374

/-- Represents a phone plan with its pricing structure -/
structure PhonePlan where
  text_cost : ℚ  -- Cost per 30 texts
  call_cost : ℚ  -- Cost per 20 minutes of calls
  data_cost : ℚ  -- Cost per 2GB of data
  intl_cost : ℚ  -- Additional cost for international calls

/-- Represents a user's monthly usage -/
structure Usage where
  texts : ℕ
  call_minutes : ℕ
  data_gb : ℚ
  intl_calls : Bool

def calculate_cost (plan : PhonePlan) (usage : Usage) : ℚ :=
  let text_units := (usage.texts + 29) / 30
  let call_units := (usage.call_minutes + 19) / 20
  let data_units := ⌈usage.data_gb / 2⌉
  plan.text_cost * text_units +
  plan.call_cost * call_units +
  plan.data_cost * data_units +
  if usage.intl_calls then plan.intl_cost else 0

theorem alternative_plan_more_expensive :
  let current_plan_cost : ℚ := 12
  let alternative_plan := PhonePlan.mk 1 3 5 2
  let darnell_usage := Usage.mk 60 60 3 true
  calculate_cost alternative_plan darnell_usage - current_plan_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_alternative_plan_more_expensive_l3553_355374


namespace NUMINAMATH_CALUDE_range_of_a_l3553_355362

-- Define propositions p and q
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1

def q (x a : ℝ) : Prop := x^2 + (2*a + 1)*x + a*(a + 1) ≥ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- Theorem statement
theorem range_of_a :
  {a : ℝ | sufficient_not_necessary a} = {a | a ≤ -4 ∨ a ≥ -1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3553_355362


namespace NUMINAMATH_CALUDE_girls_exceed_boys_by_69_l3553_355328

/-- Proves that in a class of 485 students with 208 boys, the number of girls exceeds the number of boys by 69 -/
theorem girls_exceed_boys_by_69 :
  let total_students : ℕ := 485
  let num_boys : ℕ := 208
  let num_girls : ℕ := total_students - num_boys
  num_girls - num_boys = 69 := by sorry

end NUMINAMATH_CALUDE_girls_exceed_boys_by_69_l3553_355328


namespace NUMINAMATH_CALUDE_f_of_two_eq_neg_eight_l3553_355327

/-- Given a function f(x) = x^5 + ax^3 + bx + 1 where f(-2) = 10, prove that f(2) = -8 -/
theorem f_of_two_eq_neg_eight (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x + 1)
    (h2 : f (-2) = 10) : 
  f 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_eq_neg_eight_l3553_355327


namespace NUMINAMATH_CALUDE_total_revenue_proof_l3553_355331

def sneakers_price : ℝ := 80
def sandals_price : ℝ := 60
def boots_price : ℝ := 120

def sneakers_discount : ℝ := 0.25
def sandals_discount : ℝ := 0.35
def boots_discount : ℝ := 0.40

def sneakers_quantity : ℕ := 2
def sandals_quantity : ℕ := 4
def boots_quantity : ℕ := 11

def discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def revenue (price discount quantity : ℝ) : ℝ :=
  discounted_price price discount * quantity

theorem total_revenue_proof :
  revenue sneakers_price sneakers_discount (sneakers_quantity : ℝ) +
  revenue sandals_price sandals_discount (sandals_quantity : ℝ) +
  revenue boots_price boots_discount (boots_quantity : ℝ) = 1068 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_proof_l3553_355331


namespace NUMINAMATH_CALUDE_quadratic_three_times_point_range_l3553_355378

/-- A quadratic function y = -x^2 - x + c has at least one "three times point" (y = 3x) 
    in the range -3 < x < 1 if and only if -4 ≤ c < 5 -/
theorem quadratic_three_times_point_range (c : ℝ) : 
  (∃ x : ℝ, -3 < x ∧ x < 1 ∧ 3 * x = -x^2 - x + c) ↔ -4 ≤ c ∧ c < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_three_times_point_range_l3553_355378


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3553_355359

/-- Converts a number from base 11 to base 10 -/
def base11ToBase10 (n : Nat) : Nat :=
  (n / 100) * 121 + ((n / 10) % 10) * 11 + (n % 10)

/-- Converts a number from base 12 to base 10 -/
def base12ToBase10 (n : Nat) (A B : Nat) : Nat :=
  (n / 100) * 144 + ((n / 10) % 10) * 12 + (n % 10)

theorem base_conversion_sum :
  let n1 : Nat := 249
  let n2 : Nat := 3 * 100 + 10 * 10 + 11
  let A : Nat := 10
  let B : Nat := 11
  base11ToBase10 n1 + base12ToBase10 n2 A B = 858 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3553_355359


namespace NUMINAMATH_CALUDE_mother_carrots_count_l3553_355320

/-- The number of carrots Vanessa picked -/
def vanessa_carrots : ℕ := 17

/-- The number of good carrots -/
def good_carrots : ℕ := 24

/-- The number of bad carrots -/
def bad_carrots : ℕ := 7

/-- The number of carrots Vanessa's mother picked -/
def mother_carrots : ℕ := (good_carrots + bad_carrots) - vanessa_carrots

theorem mother_carrots_count : mother_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_mother_carrots_count_l3553_355320


namespace NUMINAMATH_CALUDE_second_divisor_exists_l3553_355347

theorem second_divisor_exists : ∃ (x y : ℕ), 0 < y ∧ y < 61 ∧ x % 61 = 24 ∧ x % y = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_exists_l3553_355347


namespace NUMINAMATH_CALUDE_median_condition_implies_right_triangle_l3553_355321

/-- Given a triangle with medians m₁, m₂, and m₃, if m₁² + m₂² = 5m₃², then the triangle is right. -/
theorem median_condition_implies_right_triangle 
  (m₁ m₂ m₃ : ℝ) 
  (h_medians : ∃ (a b c : ℝ), 
    m₁^2 = (2*(b^2 + c^2) - a^2) / 4 ∧ 
    m₂^2 = (2*(a^2 + c^2) - b^2) / 4 ∧ 
    m₃^2 = (2*(a^2 + b^2) - c^2) / 4)
  (h_condition : m₁^2 + m₂^2 = 5 * m₃^2) :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_median_condition_implies_right_triangle_l3553_355321


namespace NUMINAMATH_CALUDE_evaluate_expression_l3553_355368

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3553_355368
