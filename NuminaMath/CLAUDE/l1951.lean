import Mathlib

namespace NUMINAMATH_CALUDE_angle_AOD_measure_l1951_195186

-- Define the angles
variable (AOB BOC COD AOD : ℝ)

-- Define the conditions
axiom angles_equal : AOB = BOC ∧ BOC = COD
axiom AOD_smaller : AOD = AOB / 3

-- Define the distinctness of rays (we can't directly represent this in angles, so we'll skip it)

-- Define the theorem
theorem angle_AOD_measure :
  (AOB + BOC + COD + AOD = 360 ∨ AOB + BOC + COD - AOD = 360) →
  AOD = 36 ∨ AOD = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOD_measure_l1951_195186


namespace NUMINAMATH_CALUDE_smallest_a_value_l1951_195158

theorem smallest_a_value (a b : ℝ) : 
  a ≥ 0 → b ≥ 0 → 
  (∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (37 * (x : ℝ))) → 
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * (x : ℝ) + b) = Real.sin (37 * (x : ℝ))) → 
  a' ≥ 37 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1951_195158


namespace NUMINAMATH_CALUDE_remainder_theorem_l1951_195183

theorem remainder_theorem (d r : ℤ) : 
  d > 1 → 
  1059 % d = r →
  1417 % d = r →
  2312 % d = r →
  d - r = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1951_195183


namespace NUMINAMATH_CALUDE_base_b_number_not_divisible_by_four_l1951_195195

theorem base_b_number_not_divisible_by_four (b : ℕ) : b ∈ ({4, 5, 6, 7, 8} : Finset ℕ) →
  (b^3 + b^2 - b + 2) % 4 ≠ 0 ↔ b ∈ ({4, 5, 7, 8} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_base_b_number_not_divisible_by_four_l1951_195195


namespace NUMINAMATH_CALUDE_internet_service_upgrade_l1951_195188

/-- Represents the internet service with speed and price -/
structure InternetService where
  speed : ℕ  -- Speed in Mbps
  price : ℕ  -- Price in dollars
  deriving Repr

/-- Calculates the yearly price difference between two services -/
def yearlyPriceDifference (s1 s2 : InternetService) : ℕ :=
  (s2.price - s1.price) * 12

/-- The problem statement -/
theorem internet_service_upgrade (current : InternetService)
    (upgrade20 upgrade30 : InternetService)
    (h1 : current.speed = 10 ∧ current.price = 20)
    (h2 : upgrade20.speed = 20 ∧ upgrade20.price = current.price + 10)
    (h3 : upgrade30.speed = 30)
    (h4 : yearlyPriceDifference upgrade20 upgrade30 = 120) :
    upgrade30.price / current.price = 2 := by
  sorry

end NUMINAMATH_CALUDE_internet_service_upgrade_l1951_195188


namespace NUMINAMATH_CALUDE_function_is_even_l1951_195102

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_is_even (f : ℝ → ℝ) 
  (h : ∀ x y, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)) : 
  IsEven f := by
  sorry

end NUMINAMATH_CALUDE_function_is_even_l1951_195102


namespace NUMINAMATH_CALUDE_bus_trip_distance_l1951_195101

/-- The distance of a bus trip given specific speed conditions -/
theorem bus_trip_distance : ∃ (d : ℝ), 
  (d / 45 = d / 50 + 1) ∧ d = 450 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l1951_195101


namespace NUMINAMATH_CALUDE_solution_characterization_l1951_195173

def equation (x y z : ℝ) : Prop :=
  Real.sqrt (3^x * (5^y + 7^z)) + Real.sqrt (5^y * (7^z + 3^x)) + Real.sqrt (7^z * (3^x + 5^y)) = 
  Real.sqrt 2 * (3^x + 5^y + 7^z)

theorem solution_characterization (x y z : ℝ) :
  equation x y z → ∃ t : ℝ, x = t / Real.log 3 ∧ y = t / Real.log 5 ∧ z = t / Real.log 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l1951_195173


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1951_195154

def is_valid_roll (a b : Nat) : Prop :=
  a ≤ 6 ∧ b ≤ 6 ∧ a + b ≤ 10 ∧ (a > 3 ∨ b > 3)

def total_outcomes : Nat := 36

def valid_outcomes : Nat := 24

theorem dice_roll_probability : 
  (valid_outcomes : ℚ) / total_outcomes = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1951_195154


namespace NUMINAMATH_CALUDE_derivative_exp_sin_derivative_frac_derivative_ln_derivative_product_derivative_cos_l1951_195182

variable (x : ℝ)

-- Function 1
theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x := by sorry

-- Function 2
theorem derivative_frac (x : ℝ) : 
  deriv (fun x => (x + 3) / (x + 2)) x = - 1 / ((x + 2) ^ 2) := by sorry

-- Function 3
theorem derivative_ln (x : ℝ) : 
  deriv (fun x => Real.log (2 * x + 3)) x = 2 / (2 * x + 3) := by sorry

-- Function 4
theorem derivative_product (x : ℝ) : 
  deriv (fun x => (x^2 + 2) * (2*x - 1)) x = 6 * x^2 - 2 * x + 4 := by sorry

-- Function 5
theorem derivative_cos (x : ℝ) : 
  deriv (fun x => Real.cos (2*x + Real.pi/3)) x = -2 * Real.sin (2*x + Real.pi/3) := by sorry

end NUMINAMATH_CALUDE_derivative_exp_sin_derivative_frac_derivative_ln_derivative_product_derivative_cos_l1951_195182


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l1951_195148

/-- Represents the remaining oil quantity in liters -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 40

/-- The oil flow rate in liters per minute -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct (t : ℝ) :
  Q t = initial_quantity - flow_rate * t :=
by sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l1951_195148


namespace NUMINAMATH_CALUDE_factorization_identities_l1951_195151

theorem factorization_identities :
  (∀ x y : ℝ, x^4 - 16*y^4 = (x^2 + 4*y^2)*(x + 2*y)*(x - 2*y)) ∧
  (∀ a : ℝ, -2*a^3 + 12*a^2 - 16*a = -2*a*(a - 2)*(a - 4)) := by
sorry

end NUMINAMATH_CALUDE_factorization_identities_l1951_195151


namespace NUMINAMATH_CALUDE_minimize_sum_of_distances_l1951_195191

/-- The point that minimizes the sum of distances to two fixed points lies on the line connecting the first point and the reflection of the second point across the y-axis. -/
theorem minimize_sum_of_distances (A B C : ℝ × ℝ) (h1 : A = (3, 3)) (h2 : B = (-1, -1)) (h3 : C.1 = -3) :
  (∃ k : ℝ, C = (-3, k) ∧ 
    (∀ k' : ℝ, dist A C + dist B C ≤ dist A (C.1, k') + dist B (C.1, k'))) →
  C.2 = -9 :=
by sorry


end NUMINAMATH_CALUDE_minimize_sum_of_distances_l1951_195191


namespace NUMINAMATH_CALUDE_center_digit_is_two_l1951_195160

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that returns the tens digit of a three-digit number --/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- The set of available digits --/
def digit_set : Finset ℕ := {2, 3, 4, 5, 6}

/-- A proposition stating that a number uses only digits from the digit set --/
def uses_digit_set (n : ℕ) : Prop :=
  (n / 100 ∈ digit_set) ∧ (tens_digit n ∈ digit_set) ∧ (n % 10 ∈ digit_set)

theorem center_digit_is_two :
  ∀ (a b : ℕ),
    a ≠ b
    → a ≥ 100 ∧ a < 1000
    → b ≥ 100 ∧ b < 1000
    → is_perfect_square a
    → is_perfect_square b
    → uses_digit_set a
    → uses_digit_set b
    → (Finset.card {a / 100, tens_digit a, a % 10, b / 100, tens_digit b, b % 10} = 5)
    → (tens_digit a = 2 ∨ tens_digit b = 2) :=
  sorry

end NUMINAMATH_CALUDE_center_digit_is_two_l1951_195160


namespace NUMINAMATH_CALUDE_mixing_ways_count_l1951_195153

/-- Represents a container used in the mixing process -/
inductive Container
| Barrel : Container  -- 12-liter barrel
| Small : Container   -- 2-liter container
| Medium : Container  -- 8-liter container

/-- Represents a liquid type -/
inductive Liquid
| Wine : Liquid
| Water : Liquid

/-- Represents a mixing operation -/
structure MixingOperation :=
(source : Container)
(destination : Container)
(liquid : Liquid)
(amount : ℕ)

/-- The set of all valid mixing operations -/
def valid_operations : Set MixingOperation := sorry

/-- A mixing sequence is a list of mixing operations -/
def MixingSequence := List MixingOperation

/-- Checks if a mixing sequence results in the correct final mixture -/
def is_valid_mixture (seq : MixingSequence) : Prop := sorry

/-- The number of distinct valid mixing sequences -/
def num_valid_sequences : ℕ := sorry

/-- Main theorem: There are exactly 32 ways to mix the liquids -/
theorem mixing_ways_count :
  num_valid_sequences = 32 := by sorry

end NUMINAMATH_CALUDE_mixing_ways_count_l1951_195153


namespace NUMINAMATH_CALUDE_remainder_of_470521_div_5_l1951_195165

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_470521_div_5_l1951_195165


namespace NUMINAMATH_CALUDE_smaller_rectangle_area_l1951_195149

/-- The area of a rectangle with half the length and half the width of a 40m by 20m rectangle is 200 square meters. -/
theorem smaller_rectangle_area (big_length big_width : ℝ) 
  (h_big_length : big_length = 40)
  (h_big_width : big_width = 20)
  (small_length small_width : ℝ)
  (h_small_length : small_length = big_length / 2)
  (h_small_width : small_width = big_width / 2) :
  small_length * small_width = 200 := by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_area_l1951_195149


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l1951_195143

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle_side_lengths (x : ℝ) :
  (is_isosceles_triangle (x + 3) (2*x + 1) 11 ∧
   satisfies_triangle_inequality (x + 3) (2*x + 1) 11) →
  (x = 8 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l1951_195143


namespace NUMINAMATH_CALUDE_number_value_l1951_195135

theorem number_value (x : ℝ) (number : ℝ) 
  (h1 : 5 - 5 / x = number + 4 / x) 
  (h2 : x = 9) : 
  number = 4 := by
sorry

end NUMINAMATH_CALUDE_number_value_l1951_195135


namespace NUMINAMATH_CALUDE_prob_three_wins_correct_l1951_195192

-- Define the game parameters
def num_balls : ℕ := 6
def num_people : ℕ := 4
def draws_per_person : ℕ := 2

-- Define the winning condition
def is_winning_product (n : ℕ) : Prop := n % 4 = 0

-- Define the probability of winning in a single draw
def single_draw_probability : ℚ := 2 / 5

-- Define the probability of exactly three people winning
def prob_three_wins : ℚ := 96 / 625

-- State the theorem
theorem prob_three_wins_correct : 
  prob_three_wins = (num_people.choose 3) * 
    (single_draw_probability ^ 3) * 
    ((1 - single_draw_probability) ^ (num_people - 3)) :=
sorry

end NUMINAMATH_CALUDE_prob_three_wins_correct_l1951_195192


namespace NUMINAMATH_CALUDE_software_cost_proof_l1951_195150

theorem software_cost_proof (total_devices : ℕ) (package1_cost package1_coverage : ℕ) 
  (package2_coverage : ℕ) (savings : ℕ) :
  total_devices = 50 →
  package1_cost = 40 →
  package1_coverage = 5 →
  package2_coverage = 10 →
  savings = 100 →
  (total_devices / package1_coverage * package1_cost - savings) / (total_devices / package2_coverage) = 60 :=
by sorry

end NUMINAMATH_CALUDE_software_cost_proof_l1951_195150


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l1951_195193

theorem factorization_of_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l1951_195193


namespace NUMINAMATH_CALUDE_min_value_theorem_l1951_195103

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b + a * c + b * c + 2 * Real.sqrt 5 = 6 - a ^ 2) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1951_195103


namespace NUMINAMATH_CALUDE_harry_pencils_left_l1951_195140

/-- Calculates the number of pencils left with Harry given the initial conditions. -/
def pencils_left_with_harry (anna_pencils : ℕ) (harry_lost : ℕ) : ℕ :=
  2 * anna_pencils - harry_lost

/-- Proves that Harry has 81 pencils left given the initial conditions. -/
theorem harry_pencils_left :
  pencils_left_with_harry 50 19 = 81 := by
  sorry

#eval pencils_left_with_harry 50 19

end NUMINAMATH_CALUDE_harry_pencils_left_l1951_195140


namespace NUMINAMATH_CALUDE_dinner_bill_split_l1951_195131

theorem dinner_bill_split (total_bill : ℝ) (num_friends : ℕ) 
  (h_total_bill : total_bill = 150)
  (h_num_friends : num_friends = 6) :
  let silas_payment := total_bill / 2
  let remaining_amount := total_bill - silas_payment
  let tip := total_bill * 0.1
  let total_to_split := remaining_amount + tip
  let num_remaining_friends := num_friends - 1
  total_to_split / num_remaining_friends = 18 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_split_l1951_195131


namespace NUMINAMATH_CALUDE_probability_multiple_of_100_is_zero_l1951_195130

def is_single_digit_multiple_of_5 (n : ℕ) : Prop :=
  n > 0 ∧ n < 10 ∧ n % 5 = 0

def is_prime_less_than_50 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 50

def is_multiple_of_100 (n : ℕ) : Prop :=
  n % 100 = 0

theorem probability_multiple_of_100_is_zero :
  ∀ (n p : ℕ), is_single_digit_multiple_of_5 n → is_prime_less_than_50 p →
  ¬(is_multiple_of_100 (n * p)) :=
sorry

end NUMINAMATH_CALUDE_probability_multiple_of_100_is_zero_l1951_195130


namespace NUMINAMATH_CALUDE_sqrt_cube_root_problem_l1951_195171

theorem sqrt_cube_root_problem (x y : ℝ) : 
  y = Real.sqrt (x - 24) + Real.sqrt (24 - x) - 8 → 
  (x - 5 * y)^(1/3 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_root_problem_l1951_195171


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_relation_l1951_195134

/-- Given two cubic polynomials h and j, where the roots of j are one less than the roots of h,
    prove that the coefficients of j are (1, 2, 1) -/
theorem cubic_polynomial_root_relation (x : ℝ) :
  let h := fun (x : ℝ) => x^3 - 2*x^2 + 3*x - 1
  let j := fun (x : ℝ) => x^3 + b*x^2 + c*x + d
  (∀ s, h s = 0 → j (s - 1) = 0) →
  (b, c, d) = (1, 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_relation_l1951_195134


namespace NUMINAMATH_CALUDE_zhukov_birth_year_l1951_195112

theorem zhukov_birth_year (total_years : ℕ) (years_diff : ℕ) (birth_year : ℕ) :
  total_years = 78 →
  years_diff = 70 →
  birth_year = 1900 - (total_years - years_diff) / 2 →
  birth_year = 1896 :=
by sorry

end NUMINAMATH_CALUDE_zhukov_birth_year_l1951_195112


namespace NUMINAMATH_CALUDE_seniors_in_stratified_sample_l1951_195185

/-- Represents the number of seniors in a stratified sample -/
def seniors_in_sample (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) : ℕ :=
  (total_seniors * sample_size) / total_students

/-- Theorem stating that in a school with 4500 students, of which 1500 are seniors,
    a stratified sample of 300 students will contain 100 seniors -/
theorem seniors_in_stratified_sample :
  seniors_in_sample 4500 1500 300 = 100 := by
  sorry

end NUMINAMATH_CALUDE_seniors_in_stratified_sample_l1951_195185


namespace NUMINAMATH_CALUDE_range_of_abc_l1951_195123

theorem range_of_abc (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  ∀ x, (∃ a' b' c', -1 < a' ∧ a' < b' ∧ b' < 1 ∧ 2 < c' ∧ c' < 3 ∧ x = (a' - b') * c') → -6 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_abc_l1951_195123


namespace NUMINAMATH_CALUDE_smallest_three_digit_product_l1951_195174

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem smallest_three_digit_product :
  ∀ n x y : ℕ,
    n = x * y * (10 * x + y) →
    100 ≤ n →
    n < 1000 →
    is_prime x →
    is_prime y →
    is_prime (10 * x + y) →
    x < 10 →
    y < 10 →
    x % 2 = 0 →
    y % 2 = 1 →
    x ≠ y →
    x ≠ 10 * x + y →
    y ≠ 10 * x + y →
    n ≥ 138 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_product_l1951_195174


namespace NUMINAMATH_CALUDE_room_width_calculation_l1951_195114

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 9 →
  cost_per_sqm = 900 →
  total_cost = 38475 →
  (total_cost / cost_per_sqm) / length = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1951_195114


namespace NUMINAMATH_CALUDE_wednesday_occurs_five_times_l1951_195159

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Properties of December in year M -/
structure DecemberProperties :=
  (sundays : List Date)
  (hasFiveSundays : sundays.length = 5)
  (has31Days : Nat)

/-- Properties of January in year M+1 -/
structure JanuaryProperties :=
  (firstDay : DayOfWeek)
  (has31Days : Nat)

/-- Function to determine the number of occurrences of a day in January -/
def countOccurrencesInJanuary (day : DayOfWeek) (january : JanuaryProperties) : Nat :=
  sorry

/-- Main theorem -/
theorem wednesday_occurs_five_times
  (december : DecemberProperties)
  (january : JanuaryProperties)
  : countOccurrencesInJanuary DayOfWeek.Wednesday january = 5 :=
sorry

end NUMINAMATH_CALUDE_wednesday_occurs_five_times_l1951_195159


namespace NUMINAMATH_CALUDE_third_quadrant_angle_property_l1951_195116

theorem third_quadrant_angle_property (α : Real) : 
  (3 * π / 2 < α) ∧ (α < 2 * π) →
  |Real.sin (α / 2)| / Real.sin (α / 2) + |Real.cos (α / 2)| / Real.cos (α / 2) + 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_quadrant_angle_property_l1951_195116


namespace NUMINAMATH_CALUDE_power_of_product_l1951_195118

theorem power_of_product (x y : ℝ) : (-2 * x * y^3)^2 = 4 * x^2 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1951_195118


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1951_195157

/-- An odd function satisfying certain conditions -/
def OddFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, 2 * x * (deriv f (2 * x)) + f (2 * x) < 0) ∧
  f (-2) = 0

/-- The solution set of xf(2x) < 0 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f (2 * x) < 0}

/-- The main theorem -/
theorem solution_set_characterization (f : ℝ → ℝ) 
  (hf : OddFunctionWithConditions f) : 
  SolutionSet f = {x | -1 < x ∧ x < 1 ∧ x ≠ 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1951_195157


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1951_195122

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1951_195122


namespace NUMINAMATH_CALUDE_parallel_vectors_expression_l1951_195194

theorem parallel_vectors_expression (α : Real) : 
  let a : Fin 2 → Real := ![2, Real.sin α]
  let b : Fin 2 → Real := ![1, Real.cos α]
  (∃ (k : Real), a = k • b) →
  (1 + Real.sin (2 * α)) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_expression_l1951_195194


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l1951_195133

/-- The number of cards Nell initially had -/
def initial_cards : ℕ := 455

/-- The number of cards Nell has left -/
def remaining_cards : ℕ := 154

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem cards_given_to_jeff : cards_given = 301 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l1951_195133


namespace NUMINAMATH_CALUDE_square_root_product_l1951_195155

theorem square_root_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_l1951_195155


namespace NUMINAMATH_CALUDE_root_implies_product_bound_l1951_195144

theorem root_implies_product_bound (a b : ℝ) 
  (h : (a + b + a) * (a + b + b) = 9) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_product_bound_l1951_195144


namespace NUMINAMATH_CALUDE_triangle_property_l1951_195169

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, cos A = 1/2 and the area is (3√3)/4 -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Angle sum in a triangle
  c * Real.cos A + a * Real.cos C = 2 * b * Real.cos A →  -- Given condition
  a = Real.sqrt 7 →  -- Given condition
  b + c = 4 →  -- Given condition
  Real.cos A = 1 / 2 ∧  -- Conclusion 1
  (1 / 2) * b * c * Real.sqrt (1 - (Real.cos A)^2) = (3 * Real.sqrt 3) / 4  -- Conclusion 2 (area)
  := by sorry

end NUMINAMATH_CALUDE_triangle_property_l1951_195169


namespace NUMINAMATH_CALUDE_find_A_l1951_195141

theorem find_A : ∃ A : ℤ, A + 10 = 15 ∧ A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1951_195141


namespace NUMINAMATH_CALUDE_sum_of_rational_roots_l1951_195187

def h (x : ℚ) : ℚ := x^3 - 12*x^2 + 47*x - 60

theorem sum_of_rational_roots :
  ∃ (r₁ r₂ r₃ : ℚ),
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0 ∧
    (∀ r : ℚ, h r = 0 → r = r₁ ∨ r = r₂ ∨ r = r₃) ∧
    r₁ + r₂ + r₃ = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_rational_roots_l1951_195187


namespace NUMINAMATH_CALUDE_am_gm_inequality_l1951_195175

theorem am_gm_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≤ b) :
  (b - a)^3 / (8 * b) > (a + b) / 2 - Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l1951_195175


namespace NUMINAMATH_CALUDE_four_point_theorem_l1951_195178

-- Define a type for points in a plane
variable (Point : Type)

-- Define a predicate for collinearity
variable (collinear : Point → Point → Point → Point → Prop)

-- Define a predicate for concyclicity
variable (concyclic : Point → Point → Point → Point → Prop)

-- Define a predicate for circle intersection
variable (circle_intersect : Point → Point → Point → Point → Prop)

-- Define the theorem
theorem four_point_theorem 
  (A B C D : Point) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_intersect : circle_intersect A B C D) : 
  collinear A B C D ∨ concyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_four_point_theorem_l1951_195178


namespace NUMINAMATH_CALUDE_min_value_a_l1951_195104

theorem min_value_a : 
  (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 2 ∧ ∃ a : ℝ, a * 3^x ≥ x - 1) → 
  (∃ a_min : ℝ, a_min = -6 ∧ ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 2 ∧ a * 3^x ≥ x - 1) → a ≥ a_min) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1951_195104


namespace NUMINAMATH_CALUDE_square_sum_equality_l1951_195190

theorem square_sum_equality : 106 * 106 + 94 * 94 = 20072 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1951_195190


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1951_195184

theorem cyclic_fraction_inequality (x y z : ℝ) :
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1951_195184


namespace NUMINAMATH_CALUDE_cost_of_items_l1951_195128

/-- Given the costs of combinations of pencils and pens, prove the cost of one of each item -/
theorem cost_of_items (pencil pen : ℝ) 
  (h1 : 3 * pencil + 2 * pen = 4.10)
  (h2 : 2 * pencil + 3 * pen = 3.70)
  (eraser : ℝ := 0.85) : 
  pencil + pen + eraser = 2.41 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_items_l1951_195128


namespace NUMINAMATH_CALUDE_circle_mass_is_one_kg_l1951_195166

/-- Given three balanced scales and the mass of λ, prove that the circle has a mass of 1 kg. -/
theorem circle_mass_is_one_kg (x y z : ℝ) : 
  3 * y = 2 * x →  -- First scale
  x + z = 3 * y →  -- Second scale
  2 * y = x + 1 →  -- Third scale (λ has mass 1)
  z = 1 :=         -- Mass of circle is 1 kg
by sorry

end NUMINAMATH_CALUDE_circle_mass_is_one_kg_l1951_195166


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_equal_digit_sums_l1951_195163

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → sumOfDigits n = sumOfDigits (k * n)

/-- Theorem statement -/
theorem smallest_three_digit_with_equal_digit_sums :
  ∃ n : ℕ, n = 999 ∧ 
    (∀ m : ℕ, 100 ≤ m ∧ m < 999 → ¬satisfiesCondition m) ∧
    satisfiesCondition n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_equal_digit_sums_l1951_195163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1951_195181

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | n => a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference :
  ∀ a₁ : ℝ, ∃ d : ℝ,
    let a := arithmeticSequence a₁ d
    (a 4 = 6) ∧ (a 3 + a 5 = a 10) → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1951_195181


namespace NUMINAMATH_CALUDE_greatest_consecutive_odd_integers_sum_400_l1951_195145

/-- The sum of the first n odd integers -/
def sum_odd_integers (n : ℕ) : ℕ := n^2

/-- The problem statement -/
theorem greatest_consecutive_odd_integers_sum_400 :
  (∃ (n : ℕ), sum_odd_integers n = 400) ∧
  (∀ (m : ℕ), sum_odd_integers m = 400 → m ≤ 20) :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_odd_integers_sum_400_l1951_195145


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l1951_195127

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l1951_195127


namespace NUMINAMATH_CALUDE_fraction_equality_l1951_195176

def f (x : ℕ) : ℚ := (x^4 + 400 : ℚ)

def numerator : ℚ := f 15 * f 27 * f 39 * f 51 * f 63
def denominator : ℚ := f 5 * f 17 * f 29 * f 41 * f 53

theorem fraction_equality : numerator / denominator = 4115 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1951_195176


namespace NUMINAMATH_CALUDE_train_speed_fraction_l1951_195177

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 50.000000000000014 →
  delay = 10 →
  (usual_time / (usual_time + delay)) = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l1951_195177


namespace NUMINAMATH_CALUDE_line_intercepts_l1951_195199

/-- Given a line with equation x + 6y + 2 = 0, prove that its x-intercept is -2 and its y-intercept is -1/3 -/
theorem line_intercepts (x y : ℝ) :
  x + 6 * y + 2 = 0 →
  (x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_l1951_195199


namespace NUMINAMATH_CALUDE_power_of_product_l1951_195129

theorem power_of_product (x y : ℝ) : (-2 * x^2 * y)^2 = 4 * x^4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1951_195129


namespace NUMINAMATH_CALUDE_c_share_approximately_119_73_l1951_195189

-- Define the grazing capacity conversion rates
def horse_to_ox : ℝ := 2
def sheep_to_ox : ℝ := 0.5

-- Define the total rent
def total_rent : ℝ := 1200

-- Define the grazing capacities for each person
def a_capacity : ℝ := 10 * 7 + 4 * horse_to_ox * 3
def b_capacity : ℝ := 12 * 5
def c_capacity : ℝ := 15 * 3
def d_capacity : ℝ := 18 * 6 + 6 * sheep_to_ox * 8
def e_capacity : ℝ := 20 * 4
def f_capacity : ℝ := 5 * horse_to_ox * 2 + 10 * sheep_to_ox * 4

-- Define the total grazing capacity
def total_capacity : ℝ := a_capacity + b_capacity + c_capacity + d_capacity + e_capacity + f_capacity

-- Theorem to prove
theorem c_share_approximately_119_73 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((c_capacity / total_capacity * total_rent) - 119.73) < ε :=
sorry

end NUMINAMATH_CALUDE_c_share_approximately_119_73_l1951_195189


namespace NUMINAMATH_CALUDE_conference_duration_l1951_195152

theorem conference_duration (h₁ : 9 > 0) (h₂ : 11 > 0) (h₃ : 12 > 0) :
  Nat.lcm (Nat.lcm 9 11) 12 = 396 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_l1951_195152


namespace NUMINAMATH_CALUDE_additive_inverse_sum_zero_l1951_195110

theorem additive_inverse_sum_zero (x : ℝ) : x + (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_sum_zero_l1951_195110


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1951_195120

theorem arithmetic_calculations : 
  (238 + 45 * 5 = 463) ∧ 
  (65 * 4 - 128 = 132) ∧ 
  (900 - 108 * 4 = 468) ∧ 
  (369 + (512 - 215) = 666) ∧ 
  (758 - 58 * 9 = 236) ∧ 
  (105 * (81 / 9 - 3) = 630) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1951_195120


namespace NUMINAMATH_CALUDE_simplify_expression_l1951_195164

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7*x^6 + 1)) / (3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1951_195164


namespace NUMINAMATH_CALUDE_equations_solutions_l1951_195106

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6

-- State the theorem
theorem equations_solutions :
  (∃ x : ℝ, equation1 x ∧ x = 1) ∧
  (∃ x : ℝ, equation2 x ∧ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equations_solutions_l1951_195106


namespace NUMINAMATH_CALUDE_storage_unit_area_l1951_195142

theorem storage_unit_area :
  let total_units : ℕ := 42
  let total_area : ℕ := 5040
  let known_units : ℕ := 20
  let known_unit_length : ℕ := 8
  let known_unit_width : ℕ := 4
  let remaining_units := total_units - known_units
  let known_units_area := known_units * known_unit_length * known_unit_width
  let remaining_area := total_area - known_units_area
  remaining_area / remaining_units = 200 := by
sorry

end NUMINAMATH_CALUDE_storage_unit_area_l1951_195142


namespace NUMINAMATH_CALUDE_smallest_number_l1951_195109

theorem smallest_number (a b c d : ℝ) (ha : a = Real.sqrt 2) (hb : b = 0) (hc : c = -1) (hd : d = 2) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1951_195109


namespace NUMINAMATH_CALUDE_rectangle_tiling_l1951_195167

theorem rectangle_tiling (m n a b : ℕ) (hm : m > 0) (hn : n > 0) 
  (h_tiling : ∃ (h v : ℕ), a * b = h * m + v * n) :
  n ∣ a ∨ m ∣ b :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l1951_195167


namespace NUMINAMATH_CALUDE_problem_I4_1_l1951_195111

theorem problem_I4_1 (x y : ℝ) (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
by sorry

end NUMINAMATH_CALUDE_problem_I4_1_l1951_195111


namespace NUMINAMATH_CALUDE_book_cost_price_l1951_195138

/-- The cost price of a book given specific pricing conditions -/
theorem book_cost_price (marked_price selling_price cost_price : ℝ) :
  selling_price = 1.25 * cost_price →
  0.95 * marked_price = selling_price →
  selling_price = 62.5 →
  cost_price = 50 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l1951_195138


namespace NUMINAMATH_CALUDE_mode_and_median_of_data_set_l1951_195172

def data_set : List ℕ := [9, 16, 18, 23, 32, 23, 48, 23]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem mode_and_median_of_data_set :
  mode data_set = 23 ∧ median data_set = 23 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_data_set_l1951_195172


namespace NUMINAMATH_CALUDE_papi_calot_plants_l1951_195107

/-- The number of plants Papi Calot needs to buy for his potato garden. -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Theorem stating the total number of plants Papi Calot needs to buy. -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_l1951_195107


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1951_195162

/-- The ratio of the area of an inscribed circle to the area of an equilateral triangle -/
theorem inscribed_circle_area_ratio (s r : ℝ) (h1 : s > 0) (h2 : r > 0) 
  (h3 : r = (Real.sqrt 3 / 6) * s) : 
  (π * r^2) / ((Real.sqrt 3 / 4) * s^2) = π / (3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1951_195162


namespace NUMINAMATH_CALUDE_score_difference_is_five_l1951_195197

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (0.20, 60),
  (0.25, 75),
  (0.15, 85),
  (0.30, 90),
  (0.10, 95)
]

-- Define the median score
def median_score : ℝ := 85

-- Define the function to calculate the mean score
def mean_score (distribution : List (ℝ × ℝ)) : ℝ :=
  (distribution.map (λ (p, s) => p * s)).sum

-- Theorem statement
theorem score_difference_is_five :
  median_score - mean_score score_distribution = 5 := by
  sorry


end NUMINAMATH_CALUDE_score_difference_is_five_l1951_195197


namespace NUMINAMATH_CALUDE_binary_representation_properties_l1951_195179

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- A function that counts the number of 0s in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number n that is a multiple of 17 and has exactly three 1s in its binary representation:
    1) The binary representation of n has at least six 0s
    2) If the binary representation of n has exactly 7 0s, then n is even -/
theorem binary_representation_properties (n : ℕ) 
  (h1 : n % 17 = 0) 
  (h2 : count_ones n = 3) : 
  (count_zeros n ≥ 6) ∧ 
  (count_zeros n = 7 → Even n) := by sorry

end NUMINAMATH_CALUDE_binary_representation_properties_l1951_195179


namespace NUMINAMATH_CALUDE_eat_chips_in_ten_days_l1951_195126

/-- The number of days it takes to eat all chips in a bag -/
def days_to_eat_chips (total_chips : ℕ) (first_day_chips : ℕ) (daily_chips : ℕ) : ℕ :=
  1 + (total_chips - first_day_chips) / daily_chips

/-- Theorem: It takes 10 days to eat a bag of 100 chips -/
theorem eat_chips_in_ten_days :
  days_to_eat_chips 100 10 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_eat_chips_in_ten_days_l1951_195126


namespace NUMINAMATH_CALUDE_calculate_expression_l1951_195108

theorem calculate_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1951_195108


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1951_195180

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if it has an asymptote y = √5 x, then its eccentricity is √6. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1951_195180


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1951_195124

theorem sum_of_roots_quadratic (a b : ℝ) 
  (ha : a^2 - a - 6 = 0) 
  (hb : b^2 - b - 6 = 0) 
  (hab : a ≠ b) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1951_195124


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1951_195170

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 36 →
  b = 48 →
  c^2 = a^2 + b^2 →
  c = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1951_195170


namespace NUMINAMATH_CALUDE_wally_fraction_given_to_friends_l1951_195136

def wally_total_tickets : ℕ := 400
def finley_tickets : ℕ := 220
def jensen_finley_ratio : Rat := 4 / 11

theorem wally_fraction_given_to_friends :
  (finley_tickets + (finley_tickets * jensen_finley_ratio)) / wally_total_tickets = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_wally_fraction_given_to_friends_l1951_195136


namespace NUMINAMATH_CALUDE_seventh_term_is_eight_l1951_195119

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 2 ∧
  a 3 + a 4 = 9

/-- Theorem: For an arithmetic sequence satisfying the given conditions, the 7th term is 8 -/
theorem seventh_term_is_eight (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_eight_l1951_195119


namespace NUMINAMATH_CALUDE_total_balloons_l1951_195121

-- Define the number of red and green balloons
def red_balloons : ℕ := 8
def green_balloons : ℕ := 9

-- Theorem stating that the total number of balloons is 17
theorem total_balloons : red_balloons + green_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l1951_195121


namespace NUMINAMATH_CALUDE_least_integer_square_52_more_than_triple_l1951_195147

theorem least_integer_square_52_more_than_triple : 
  ∃ x : ℤ, x^2 = 3*x + 52 ∧ ∀ y : ℤ, y^2 = 3*y + 52 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_52_more_than_triple_l1951_195147


namespace NUMINAMATH_CALUDE_power_twentyseven_x_plus_one_l1951_195105

theorem power_twentyseven_x_plus_one (x : ℝ) (h : (3 : ℝ) ^ (2 * x) = 5) : 
  (27 : ℝ) ^ (x + 1) = 135 := by sorry

end NUMINAMATH_CALUDE_power_twentyseven_x_plus_one_l1951_195105


namespace NUMINAMATH_CALUDE_worker_pay_is_40_l1951_195198

/-- Represents the plant supplier's sales and expenses --/
structure PlantSupplier where
  orchids : ℕ
  orchidPrice : ℕ
  chinesePlants : ℕ
  chinesePlantPrice : ℕ
  potCost : ℕ
  leftover : ℕ
  workers : ℕ

/-- Calculates the amount paid to each worker --/
def workerPay (ps : PlantSupplier) : ℕ :=
  let totalEarnings := ps.orchids * ps.orchidPrice + ps.chinesePlants * ps.chinesePlantPrice
  let totalSpent := ps.potCost + ps.leftover
  (totalEarnings - totalSpent) / ps.workers

/-- Theorem stating that each worker is paid $40 --/
theorem worker_pay_is_40 (ps : PlantSupplier) 
  (h1 : ps.orchids = 20)
  (h2 : ps.orchidPrice = 50)
  (h3 : ps.chinesePlants = 15)
  (h4 : ps.chinesePlantPrice = 25)
  (h5 : ps.potCost = 150)
  (h6 : ps.leftover = 1145)
  (h7 : ps.workers = 2) :
  workerPay ps = 40 := by
  sorry

end NUMINAMATH_CALUDE_worker_pay_is_40_l1951_195198


namespace NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l1951_195117

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  s : ℝ  -- radius of the sphere
  a : ℝ  -- length of the box
  b : ℝ  -- width of the box
  c : ℝ  -- height of the box

/-- The sum of the lengths of the 12 edges of the box -/
def edgeSum (box : InscribedBox) : ℝ := 4 * (box.a + box.b + box.c)

/-- The surface area of the box -/
def surfaceArea (box : InscribedBox) : ℝ := 2 * (box.a * box.b + box.b * box.c + box.c * box.a)

/-- The theorem stating the relationship between the box dimensions and the sphere radius -/
theorem inscribed_box_sphere_radius (box : InscribedBox) 
    (h1 : edgeSum box = 160) 
    (h2 : surfaceArea box = 600) : 
    box.s = 5 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l1951_195117


namespace NUMINAMATH_CALUDE_exist_abcd_equation_l1951_195137

theorem exist_abcd_equation : ∃ (a b c d : ℕ), 
  (a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1) ∧
  (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c * d) : ℚ) = 37 / 48 ∧
  b = 4 := by sorry

end NUMINAMATH_CALUDE_exist_abcd_equation_l1951_195137


namespace NUMINAMATH_CALUDE_total_students_is_47_l1951_195156

/-- The number of students supposed to be in Miss Smith's English class -/
def total_students : ℕ :=
  let tables : ℕ := 6
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_multiplier : ℕ := 3
  let new_groups : ℕ := 2
  let students_per_new_group : ℕ := 4
  let german_students : ℕ := 3
  let french_students : ℕ := 3
  let norwegian_students : ℕ := 3

  let current_students := tables * students_per_table
  let missing_students := bathroom_students + (canteen_multiplier * bathroom_students)
  let new_group_students := new_groups * students_per_new_group
  let exchange_students := german_students + french_students + norwegian_students

  current_students + missing_students + new_group_students + exchange_students

theorem total_students_is_47 : total_students = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_47_l1951_195156


namespace NUMINAMATH_CALUDE_roses_distribution_l1951_195100

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) 
  (h1 : initial_roses = 40) 
  (h2 : stolen_roses = 4) 
  (h3 : people = 9) :
  (initial_roses - stolen_roses) / people = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_distribution_l1951_195100


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equality_l1951_195125

theorem cosine_sine_sum_equality : 
  Real.cos (42 * π / 180) * Real.cos (78 * π / 180) + 
  Real.sin (42 * π / 180) * Real.cos (168 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equality_l1951_195125


namespace NUMINAMATH_CALUDE_solution_set_equality_l1951_195168

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Define the set of x values satisfying both conditions
def S : Set ℝ := {x : ℝ | f x > 0 ∧ x < 3}

-- Theorem statement
theorem solution_set_equality : S = Set.Ioi (-1) ∪ Set.Ioo 1 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1951_195168


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1951_195113

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 ^ k.val : ℤ) % 3 ≠ (k.val ^ 7 : ℤ) % 3) ∧ 
  (7 ^ n.val : ℤ) % 3 = (n.val ^ 7 : ℤ) % 3 → 
  n = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1951_195113


namespace NUMINAMATH_CALUDE_min_x_coordinate_midpoint_l1951_195161

/-- Given a line segment AB of length m on the right branch of the hyperbola x²/a² - y²/b² = 1,
    where m > 2b²/a, the minimum x-coordinate of the midpoint M of AB is a(m + 2a) / (2√(a² + b²)). -/
theorem min_x_coordinate_midpoint (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 2 * b^2 / a) :
  let min_x := a * (m + 2 * a) / (2 * Real.sqrt (a^2 + b^2))
  ∀ (x y z w : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 →
    z^2 / a^2 - w^2 / b^2 = 1 →
    (z - x)^2 + (w - y)^2 = m^2 →
    x > 0 →
    z > 0 →
    (x + z) / 2 ≥ min_x :=
by sorry

end NUMINAMATH_CALUDE_min_x_coordinate_midpoint_l1951_195161


namespace NUMINAMATH_CALUDE_highway_repair_time_l1951_195196

theorem highway_repair_time (x y : ℝ) : 
  (1 / x + 1 / y = 1 / 18) →  -- Combined work rate
  (2 * x / 3 + y / 3 = 40) →  -- Actual repair time
  (x = 45 ∧ y = 30) := by
  sorry

end NUMINAMATH_CALUDE_highway_repair_time_l1951_195196


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l1951_195115

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l1951_195115


namespace NUMINAMATH_CALUDE_larger_number_proof_l1951_195146

/-- Given two positive integers with specific HCF and LCM conditions, prove the larger one is 230 -/
theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 23) 
  (h2 : Nat.lcm a b = 23 * 9 * 10) (h3 : a > b) : a = 230 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1951_195146


namespace NUMINAMATH_CALUDE_intersection_M_N_l1951_195132

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 4*x < 0}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1951_195132


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1951_195139

theorem fraction_sum_equality : 
  (2 / 20 : ℚ) + (3 / 50 : ℚ) * (5 / 100 : ℚ) + (4 / 1000 : ℚ) + (6 / 10000 : ℚ) = 1076 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1951_195139
