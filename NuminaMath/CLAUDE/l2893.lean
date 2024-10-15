import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l2893_289382

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ ∃ k, l2 a (x + k) (y + k * (a / 2))

-- Define perpendicular lines
def perpendicular (a : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l1 a x₁ y₁ ∧ l2 a x₂ y₂ → (x₂ - x₁) * (a * (x₂ - x₁) + 2 * (y₂ - y₁)) + (y₂ - y₁) * ((a - 1) * (x₂ - x₁) + (y₂ - y₁)) = 0

-- Theorem for parallel lines
theorem parallel_lines : ∀ a : ℝ, parallel a → a = -1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines : ∀ a : ℝ, perpendicular a → a = 2/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l2893_289382


namespace NUMINAMATH_CALUDE_charles_total_money_l2893_289380

-- Define the value of each coin type in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the number of coins Charles found on his way to school
def found_pennies : ℕ := 6
def found_nickels : ℕ := 4
def found_dimes : ℕ := 3

-- Define the number of coins Charles already had at home
def home_nickels : ℕ := 3
def home_dimes : ℕ := 2
def home_quarters : ℕ := 1

-- Calculate the total value in cents
def total_cents : ℕ :=
  found_pennies * penny_value +
  (found_nickels + home_nickels) * nickel_value +
  (found_dimes + home_dimes) * dime_value +
  home_quarters * quarter_value

-- Theorem to prove
theorem charles_total_money :
  total_cents = 116 := by sorry

end NUMINAMATH_CALUDE_charles_total_money_l2893_289380


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2893_289335

theorem sum_of_x_and_y (x y : ℤ) : x - y = 200 → y = 235 → x + y = 670 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2893_289335


namespace NUMINAMATH_CALUDE_cube_square_third_smallest_prime_l2893_289397

/-- The third smallest prime number -/
def third_smallest_prime : Nat := 5

/-- The cube of the square of the third smallest prime number -/
def result : Nat := (third_smallest_prime ^ 2) ^ 3

theorem cube_square_third_smallest_prime :
  result = 15625 := by sorry

end NUMINAMATH_CALUDE_cube_square_third_smallest_prime_l2893_289397


namespace NUMINAMATH_CALUDE_polynomial_value_range_l2893_289396

/-- A polynomial with integer coefficients that equals 5 for five different integer inputs -/
def IntPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a b c d e : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5 ∧ P e = 5

theorem polynomial_value_range (P : ℤ → ℤ) (h : IntPolynomial P) :
  ¬∃ x : ℤ, ((-6 : ℤ) ≤ P x ∧ P x ≤ 4) ∨ (6 ≤ P x ∧ P x ≤ 16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_range_l2893_289396


namespace NUMINAMATH_CALUDE_problem_statement_l2893_289328

theorem problem_statement : (-1 : ℤ) ^ (4 ^ 3) + 2 ^ (3 ^ 2) = 513 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2893_289328


namespace NUMINAMATH_CALUDE_divisibility_equations_solutions_l2893_289392

theorem divisibility_equations_solutions :
  (∀ x : ℤ, (x - 1 ∣ x + 3) ↔ x ∈ ({-3, -1, 0, 2, 3, 5} : Set ℤ)) ∧
  (∀ x : ℤ, (x + 2 ∣ x^2 + 2) ↔ x ∈ ({-8, -5, -4, -3, -1, 0, 1, 4} : Set ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equations_solutions_l2893_289392


namespace NUMINAMATH_CALUDE_y_value_proof_l2893_289345

/-- Given that 150% of x is equal to 75% of y and x = 24, prove that y = 48 -/
theorem y_value_proof (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2893_289345


namespace NUMINAMATH_CALUDE_cost_of_750_candies_l2893_289318

/-- The cost of buying a specific number of chocolate candies given the following conditions:
  * A box contains a fixed number of candies
  * A box costs a fixed amount
  * There is a discount percentage for buying more than a certain number of boxes
  * We need to buy a specific number of candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (discount_percentage : ℚ) 
  (discount_threshold : ℕ) (total_candies : ℕ) : ℚ :=
  let boxes_needed : ℕ := (total_candies + candies_per_box - 1) / candies_per_box
  let total_cost : ℚ := boxes_needed * cost_per_box
  if boxes_needed > discount_threshold
  then total_cost * (1 - discount_percentage)
  else total_cost

theorem cost_of_750_candies :
  cost_of_candies 30 (7.5) (1/10) 20 750 = (168.75) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_750_candies_l2893_289318


namespace NUMINAMATH_CALUDE_ball_probability_l2893_289319

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 17)
  (h5 : red = 3)
  (h6 : purple = 1)
  (h7 : total = white + green + yellow + red + purple) :
  (total - (red + purple)) / total = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2893_289319


namespace NUMINAMATH_CALUDE_luna_bus_cost_l2893_289367

/-- The distance from city X to city Y in kilometers -/
def distance_XY : ℝ := 4500

/-- The cost per kilometer for bus travel in dollars -/
def bus_cost_per_km : ℝ := 0.20

/-- The total cost for Luna to bus from city X to city Y -/
def total_bus_cost : ℝ := distance_XY * bus_cost_per_km

/-- Theorem stating that the total bus cost for Luna to travel from X to Y is $900 -/
theorem luna_bus_cost : total_bus_cost = 900 := by
  sorry

end NUMINAMATH_CALUDE_luna_bus_cost_l2893_289367


namespace NUMINAMATH_CALUDE_all_star_arrangement_l2893_289309

def number_of_arrangements (n_cubs : ℕ) (n_red_sox : ℕ) (n_yankees : ℕ) : ℕ :=
  let n_cubs_with_coach := n_cubs + 1
  let n_teams := 3
  n_teams.factorial * n_cubs_with_coach.factorial * n_red_sox.factorial * n_yankees.factorial

theorem all_star_arrangement :
  number_of_arrangements 4 3 2 = 8640 := by
  sorry

end NUMINAMATH_CALUDE_all_star_arrangement_l2893_289309


namespace NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_and_b_l2893_289358

/-- The function f(x) = x³ + ax² + b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

/-- The derivative of f(x) -/
def f_deriv (a x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem tangent_line_parallel_implies_a_and_b (a b : ℝ) : 
  f a b 1 = 0 ∧ f_deriv a 1 = -3 → a = -3 ∧ b = 2 := by
  sorry

#check tangent_line_parallel_implies_a_and_b

end NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_and_b_l2893_289358


namespace NUMINAMATH_CALUDE_x_intercept_is_one_l2893_289338

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 1 -/
theorem x_intercept_is_one :
  let l : Line := { x₁ := 2, y₁ := -2, x₂ := -1, y₂ := 4 }
  x_intercept l = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_one_l2893_289338


namespace NUMINAMATH_CALUDE_class_photo_cost_l2893_289331

/-- The total cost of class photos for a given number of students -/
def total_cost (students : ℕ) (fixed_price : ℚ) (fixed_photos : ℕ) (additional_cost : ℚ) : ℚ :=
  fixed_price + (additional_cost * (students - fixed_photos))

/-- Proof that the total cost for the class photo is 139.5 yuan -/
theorem class_photo_cost :
  let students : ℕ := 54
  let fixed_price : ℚ := 24.5
  let fixed_photos : ℕ := 4
  let additional_cost : ℚ := 2.3
  total_cost students fixed_price fixed_photos additional_cost = 139.5 := by
  sorry

end NUMINAMATH_CALUDE_class_photo_cost_l2893_289331


namespace NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l2893_289389

theorem sequence_with_positive_triples_negative_sum : 
  ∃ (seq : Fin 20 → ℝ), 
    (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
    (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l2893_289389


namespace NUMINAMATH_CALUDE_min_moves_to_exit_l2893_289308

/-- Represents the direction of car movement -/
inductive Direction
| Left
| Right
| Up
| Down

/-- Represents a car in the parking lot -/
structure Car where
  id : Nat
  position : Nat × Nat

/-- Represents the parking lot -/
structure ParkingLot where
  cars : List Car
  width : Nat
  height : Nat

/-- Represents a move in the solution -/
structure Move where
  car : Car
  direction : Direction

/-- Checks if a car can exit the parking lot -/
def canExit (pl : ParkingLot) (car : Car) : Prop :=
  sorry

/-- Checks if a sequence of moves is valid -/
def isValidMoveSequence (pl : ParkingLot) (moves : List Move) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_to_exit (pl : ParkingLot) (car : Car) :
  (∃ (moves : List Move), isValidMoveSequence pl moves ∧ canExit pl car) →
  (∃ (minMoves : List Move), isValidMoveSequence pl minMoves ∧ canExit pl car ∧ minMoves.length = 6) :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_exit_l2893_289308


namespace NUMINAMATH_CALUDE_rotate_point_D_l2893_289352

/-- Rotates a point (x, y) by 180 degrees around a center (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

theorem rotate_point_D :
  let d : ℝ × ℝ := (2, -3)
  let center : ℝ × ℝ := (3, -2)
  rotate180 d.1 d.2 center.1 center.2 = (4, -1) := by
sorry

end NUMINAMATH_CALUDE_rotate_point_D_l2893_289352


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l2893_289315

/-- Given a circle with center (4, -2) and one endpoint of a diameter at (1, 5),
    the other endpoint of the diameter is at (7, -9). -/
theorem circle_diameter_endpoint :
  let center : ℝ × ℝ := (4, -2)
  let endpoint1 : ℝ × ℝ := (1, 5)
  let endpoint2 : ℝ × ℝ := (7, -9)
  (endpoint1.1 - center.1 = center.1 - endpoint2.1) ∧
  (endpoint1.2 - center.2 = center.2 - endpoint2.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l2893_289315


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2893_289376

theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
    women = men + 6 → 
    men + women = 16 → 
    (men : ℚ) / women = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2893_289376


namespace NUMINAMATH_CALUDE_find_number_l2893_289347

theorem find_number : ∃ x : ℝ, 0.5 * x = 0.4 * 120 + 180 ∧ x = 456 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2893_289347


namespace NUMINAMATH_CALUDE_zero_exponent_equals_one_l2893_289341

theorem zero_exponent_equals_one (r : ℚ) (h : r ≠ 0) : r ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_equals_one_l2893_289341


namespace NUMINAMATH_CALUDE_range_of_a_given_solution_exact_range_of_a_l2893_289317

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop := 2 * x^2 + a * x - a^2 > 0

-- State the theorem
theorem range_of_a_given_solution : 
  ∀ a : ℝ, inequality 2 a → -2 < a ∧ a < 4 :=
by
  sorry

-- Define the range of a
def range_of_a : Set ℝ := { a : ℝ | -2 < a ∧ a < 4 }

-- State that this is the exact range
theorem exact_range_of_a : 
  ∀ a : ℝ, a ∈ range_of_a ↔ inequality 2 a :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_given_solution_exact_range_of_a_l2893_289317


namespace NUMINAMATH_CALUDE_medicine_survey_l2893_289304

theorem medicine_survey (total : ℕ) (cold : ℕ) (stomach : ℕ) 
  (h_total : total = 100)
  (h_cold : cold = 75)
  (h_stomach : stomach = 80)
  (h_cold_le_total : cold ≤ total)
  (h_stomach_le_total : stomach ≤ total) :
  ∃ (max_both min_both : ℕ),
    max_both ≤ cold ∧
    max_both ≤ stomach ∧
    cold + stomach - max_both ≤ total ∧
    max_both = 75 ∧
    min_both ≥ 0 ∧
    min_both ≤ cold ∧
    min_both ≤ stomach ∧
    cold + stomach - min_both ≥ total ∧
    min_both = 55 := by
  sorry

end NUMINAMATH_CALUDE_medicine_survey_l2893_289304


namespace NUMINAMATH_CALUDE_smallest_number_l2893_289371

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The number 85 in base 9 --/
def n1 : Nat := to_decimal [5, 8] 9

/-- The number 210 in base 6 --/
def n2 : Nat := to_decimal [0, 1, 2] 6

/-- The number 1000 in base 4 --/
def n3 : Nat := to_decimal [0, 0, 0, 1] 4

/-- The number 111111 in base 2 --/
def n4 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number : n4 = min n1 (min n2 (min n3 n4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2893_289371


namespace NUMINAMATH_CALUDE_appetizer_cost_per_person_l2893_289340

/-- Calculates the cost per person for a New Year's Eve appetizer --/
theorem appetizer_cost_per_person 
  (num_guests : ℕ) 
  (num_chip_bags : ℕ) 
  (chip_cost : ℚ) 
  (creme_fraiche_cost : ℚ) 
  (salmon_cost : ℚ) 
  (caviar_cost : ℚ) 
  (h1 : num_guests = 12) 
  (h2 : num_chip_bags = 10) 
  (h3 : chip_cost = 1) 
  (h4 : creme_fraiche_cost = 5) 
  (h5 : salmon_cost = 15) 
  (h6 : caviar_cost = 250) :
  (num_chip_bags * chip_cost + creme_fraiche_cost + salmon_cost + caviar_cost) / num_guests = 280 / 12 :=
by sorry

end NUMINAMATH_CALUDE_appetizer_cost_per_person_l2893_289340


namespace NUMINAMATH_CALUDE_multiplication_fraction_equality_l2893_289307

theorem multiplication_fraction_equality : 7 * (1 / 21) * 42 = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_equality_l2893_289307


namespace NUMINAMATH_CALUDE_function_identity_l2893_289312

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) + f (x * y) = y * f x + f y + f (f x)

-- State the theorem
theorem function_identity {f : ℝ → ℝ} (h : SatisfiesEquation f) :
  ∀ x : ℝ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l2893_289312


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2893_289302

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {1, 3, 5}

theorem intersection_complement_equality : N ∩ (U \ M) = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2893_289302


namespace NUMINAMATH_CALUDE_family_photos_l2893_289321

theorem family_photos (total : ℕ) (friends : ℕ) (family : ℕ) 
  (h1 : total = 86) 
  (h2 : friends = 63) 
  (h3 : total = friends + family) : family = 23 := by
  sorry

end NUMINAMATH_CALUDE_family_photos_l2893_289321


namespace NUMINAMATH_CALUDE_four_students_arrangement_l2893_289372

/-- The number of ways to arrange n students in a line -/
def lineArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a line with one specific student at either end -/
def arrangementsWithOneAtEnd (n : ℕ) : ℕ := 
  2 * lineArrangements (n - 1)

/-- Theorem: There are 12 ways to arrange 4 students in a line with one specific student at either end -/
theorem four_students_arrangement : arrangementsWithOneAtEnd 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_four_students_arrangement_l2893_289372


namespace NUMINAMATH_CALUDE_multiples_of_seven_between_15_and_200_l2893_289350

theorem multiples_of_seven_between_15_and_200 : 
  (Finset.filter (fun n => n % 7 = 0 ∧ n > 15 ∧ n < 200) (Finset.range 200)).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_seven_between_15_and_200_l2893_289350


namespace NUMINAMATH_CALUDE_max_k_inequality_l2893_289373

theorem max_k_inequality (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  (∃ k : ℝ, ∀ k' : ℝ, 
    (x^2 / (1 + x) + y^2 / (1 + y) + (x - 1) * (y - 1) ≥ k' * x * y) → k' ≤ k) ∧
  (x^2 / (1 + x) + y^2 / (1 + y) + (x - 1) * (y - 1) ≥ ((13 - 5 * Real.sqrt 5) / 2) * x * y) :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l2893_289373


namespace NUMINAMATH_CALUDE_sqrt_one_eighth_same_type_as_sqrt_two_l2893_289306

theorem sqrt_one_eighth_same_type_as_sqrt_two :
  ∃ (q : ℚ), Real.sqrt (1/8 : ℝ) = q * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_one_eighth_same_type_as_sqrt_two_l2893_289306


namespace NUMINAMATH_CALUDE_mod_sum_powers_seven_l2893_289313

theorem mod_sum_powers_seven : (9^5 + 4^6 + 5^7) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_seven_l2893_289313


namespace NUMINAMATH_CALUDE_equation_solution_l2893_289343

theorem equation_solution : ∃! x : ℚ, (4 * x - 12) / 3 = (3 * x + 6) / 5 ∧ x = 78 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2893_289343


namespace NUMINAMATH_CALUDE_arithmetic_comparisons_l2893_289387

theorem arithmetic_comparisons : 
  (25 + 45 = 45 + 25) ∧ 
  (56 - 28 < 65 - 28) ∧ 
  (22 * 41 = 41 * 22) ∧ 
  (50 - 32 > 50 - 23) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_comparisons_l2893_289387


namespace NUMINAMATH_CALUDE_computer_cost_l2893_289327

theorem computer_cost (total_budget fridge_cost tv_cost computer_cost : ℕ) : 
  total_budget = 1600 →
  tv_cost = 600 →
  fridge_cost = computer_cost + 500 →
  total_budget = tv_cost + fridge_cost + computer_cost →
  computer_cost = 250 := by
sorry

end NUMINAMATH_CALUDE_computer_cost_l2893_289327


namespace NUMINAMATH_CALUDE_electricity_billing_theorem_l2893_289342

/-- Represents a three-tariff meter reading --/
structure MeterReading where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_ordered : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f

/-- Represents tariff prices --/
structure TariffPrices where
  t₁ : ℝ
  t₂ : ℝ
  t₃ : ℝ

/-- Calculates the maximum additional payment --/
def maxAdditionalPayment (reading : MeterReading) (prices : TariffPrices) (actualPayment : ℝ) : ℝ :=
  sorry

/-- Calculates the expected value of the difference --/
def expectedDifference (reading : MeterReading) (prices : TariffPrices) (actualPayment : ℝ) : ℝ :=
  sorry

/-- Main theorem --/
theorem electricity_billing_theorem (reading : MeterReading) (prices : TariffPrices) :
  let actualPayment := 660.72
  prices.t₁ = 4.03 ∧ prices.t₂ = 1.01 ∧ prices.t₃ = 3.39 →
  reading.a = 1214 ∧ reading.b = 1270 ∧ reading.c = 1298 ∧
  reading.d = 1337 ∧ reading.e = 1347 ∧ reading.f = 1402 →
  maxAdditionalPayment reading prices actualPayment = 397.34 ∧
  expectedDifference reading prices actualPayment = 19.30 :=
sorry

end NUMINAMATH_CALUDE_electricity_billing_theorem_l2893_289342


namespace NUMINAMATH_CALUDE_range_of_a_l2893_289356

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := ∀ x, x ∈ A → x ∈ B a

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_condition a ↔ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2893_289356


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_square_l2893_289379

/-- A rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.length)

/-- A square formed by 5 identical rectangles -/
structure SquareFromRectangles where
  base_rectangle : Rectangle
  side_length : ℝ
  h_side_length : side_length = 5 * base_rectangle.width

/-- The perimeter of the square formed by rectangles -/
def SquareFromRectangles.perimeter (s : SquareFromRectangles) : ℝ :=
  4 * s.side_length

theorem rectangle_perimeter_from_square (s : SquareFromRectangles) 
    (h_perimeter_diff : s.perimeter = s.base_rectangle.perimeter + 10) :
    s.base_rectangle.perimeter = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_from_square_l2893_289379


namespace NUMINAMATH_CALUDE_base_b_sum_equals_21_l2893_289361

-- Define the sum of single-digit numbers in base b
def sum_single_digits (b : ℕ) : ℕ := (b * (b - 1)) / 2

-- Define the value 21 in base b
def value_21_base_b (b : ℕ) : ℕ := 2 * b + 1

-- Theorem statement
theorem base_b_sum_equals_21 :
  ∃ b : ℕ, b > 1 ∧ sum_single_digits b = value_21_base_b b ∧ b = 7 :=
sorry

end NUMINAMATH_CALUDE_base_b_sum_equals_21_l2893_289361


namespace NUMINAMATH_CALUDE_square_side_length_from_voice_range_l2893_289362

/-- The side length of a square ground, given the area of a quarter circle
    representing the range of a trainer's voice from one corner. -/
theorem square_side_length_from_voice_range (r : ℝ) (area : ℝ) 
    (h1 : r = 140)
    (h2 : area = 15393.804002589986)
    (h3 : area = (π * r^2) / 4) : 
  ∃ (s : ℝ), s^2 = r^2 ∧ s = 140 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_from_voice_range_l2893_289362


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2893_289329

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = x + f (f y)

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x + c for some real constant c -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2893_289329


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l2893_289336

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k :=
by sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l2893_289336


namespace NUMINAMATH_CALUDE_trivia_game_points_per_question_l2893_289322

theorem trivia_game_points_per_question 
  (first_half_correct : ℕ) 
  (second_half_correct : ℕ) 
  (final_score : ℕ) 
  (h1 : first_half_correct = 5)
  (h2 : second_half_correct = 5)
  (h3 : final_score = 50) :
  final_score / (first_half_correct + second_half_correct) = 5 := by
sorry

end NUMINAMATH_CALUDE_trivia_game_points_per_question_l2893_289322


namespace NUMINAMATH_CALUDE_pickle_discount_l2893_289393

/-- Calculates the discount on a jar of pickles based on grocery purchases and change received --/
theorem pickle_discount (meat_price meat_weight buns_price lettuce_price tomato_price tomato_weight pickle_price bill change : ℝ) :
  meat_price = 3.5 ∧
  meat_weight = 2 ∧
  buns_price = 1.5 ∧
  lettuce_price = 1 ∧
  tomato_price = 2 ∧
  tomato_weight = 1.5 ∧
  pickle_price = 2.5 ∧
  bill = 20 ∧
  change = 6 →
  pickle_price - ((meat_price * meat_weight + buns_price + lettuce_price + tomato_price * tomato_weight + pickle_price) - (bill - change)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pickle_discount_l2893_289393


namespace NUMINAMATH_CALUDE_four_digit_permutations_l2893_289384

-- Define the multiset
def digit_multiset : Multiset ℕ := {3, 3, 7, 7}

-- Define the function to calculate permutations of a multiset
noncomputable def multiset_permutations (m : Multiset ℕ) : ℕ := sorry

-- Theorem statement
theorem four_digit_permutations :
  multiset_permutations digit_multiset = 6 := by sorry

end NUMINAMATH_CALUDE_four_digit_permutations_l2893_289384


namespace NUMINAMATH_CALUDE_positive_integer_triplet_solution_l2893_289383

theorem positive_integer_triplet_solution (x y z : ℕ+) :
  (x + y)^2 + 3*x + y + 1 = z^2 → x = y ∧ z = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_triplet_solution_l2893_289383


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2893_289395

theorem cube_volume_problem (cube_a_volume : ℝ) (surface_area_ratio : ℝ) :
  cube_a_volume = 8 →
  surface_area_ratio = 3 →
  ∃ (cube_b_volume : ℝ),
    (6 * (cube_a_volume ^ (1/3))^2) * surface_area_ratio = 6 * (cube_b_volume ^ (1/3))^2 ∧
    cube_b_volume = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2893_289395


namespace NUMINAMATH_CALUDE_weight_estimate_for_178cm_l2893_289339

/-- Regression equation for weight based on height -/
def weight_regression (height : ℝ) : ℝ := 0.72 * height - 58.5

/-- The problem statement -/
theorem weight_estimate_for_178cm :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |weight_regression 178 - 70| < ε :=
sorry

end NUMINAMATH_CALUDE_weight_estimate_for_178cm_l2893_289339


namespace NUMINAMATH_CALUDE_simplify_expression_l2893_289388

theorem simplify_expression : ((3 + 4 + 5 + 6) / 2) + ((3 * 6 + 9) / 3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2893_289388


namespace NUMINAMATH_CALUDE_count_subset_pairs_formula_l2893_289325

/-- The number of pairs of non-empty subsets (A, B) of {1, 2, ..., n} such that
    the maximum element of A is less than the minimum element of B -/
def count_subset_pairs (n : ℕ) : ℕ :=
  (n - 2) * 2^(n - 1) + 1

/-- Theorem stating that for any integer n ≥ 3, the count of subset pairs
    satisfying the given condition is equal to (n-2) * 2^(n-1) + 1 -/
theorem count_subset_pairs_formula (n : ℕ) (h : n ≥ 3) :
  count_subset_pairs n = (n - 2) * 2^(n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_count_subset_pairs_formula_l2893_289325


namespace NUMINAMATH_CALUDE_game_winner_conditions_l2893_289305

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | AWins
  | BWins

/-- Represents the game state -/
structure GameState where
  n : ℕ
  m : ℕ
  currentPlayer : Bool  -- true for A, false for B

/-- Determines the winner of the game given the initial state -/
def determineWinner (initialState : GameState) : GameOutcome :=
  if initialState.n = initialState.m then
    GameOutcome.BWins
  else
    GameOutcome.AWins

/-- Theorem stating the winning conditions for the game -/
theorem game_winner_conditions (n m : ℕ) (hn : n > 1) (hm : m > 1) :
  let initialState := GameState.mk n m true
  determineWinner initialState =
    if n = m then
      GameOutcome.BWins
    else
      GameOutcome.AWins :=
by
  sorry


end NUMINAMATH_CALUDE_game_winner_conditions_l2893_289305


namespace NUMINAMATH_CALUDE_used_car_selections_l2893_289301

theorem used_car_selections (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ)
  (h1 : num_cars = 12)
  (h2 : num_clients = 9)
  (h3 : selections_per_client = 4) :
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selections_l2893_289301


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2893_289364

theorem mixed_number_calculation : 
  25 * ((5 + 2/7) - (3 + 3/5)) / ((3 + 1/6) + (2 + 1/4)) = 7 + 49/91 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2893_289364


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l2893_289349

/-- Calculate the overall loss percentage for three items given their cost and selling prices -/
theorem overall_loss_percentage 
  (cp_radio cp_tv cp_blender : ℝ) 
  (sp_radio sp_tv sp_blender : ℝ) : 
  let total_cp := cp_radio + cp_tv + cp_blender
  let total_sp := sp_radio + sp_tv + sp_blender
  ((total_cp - total_sp) / total_cp) * 100 = 
    ((4500 + 8000 + 1300) - (3200 + 7500 + 1000)) / (4500 + 8000 + 1300) * 100 := by
  sorry

#eval ((4500 + 8000 + 1300) - (3200 + 7500 + 1000)) / (4500 + 8000 + 1300) * 100

end NUMINAMATH_CALUDE_overall_loss_percentage_l2893_289349


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2893_289355

theorem container_volume_ratio : 
  ∀ (v1 v2 : ℝ), v1 > 0 → v2 > 0 → 
  (3/4 : ℝ) * v1 = (5/8 : ℝ) * v2 → 
  v1 / v2 = (5/6 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2893_289355


namespace NUMINAMATH_CALUDE_gold_silver_alloy_composition_l2893_289300

/-- Prove the composition of a gold-silver alloy given its properties -/
theorem gold_silver_alloy_composition
  (total_mass : ℝ)
  (total_volume : ℝ)
  (density_gold : ℝ)
  (density_silver : ℝ)
  (h_total_mass : total_mass = 13.85)
  (h_total_volume : total_volume = 0.9)
  (h_density_gold : density_gold = 19.3)
  (h_density_silver : density_silver = 10.5) :
  ∃ (mass_gold mass_silver : ℝ),
    mass_gold + mass_silver = total_mass ∧
    mass_gold / density_gold + mass_silver / density_silver = total_volume ∧
    mass_gold = 9.65 ∧
    mass_silver = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_gold_silver_alloy_composition_l2893_289300


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2893_289359

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 8) (hb : b = 3) :
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 55 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2893_289359


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2893_289386

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2893_289386


namespace NUMINAMATH_CALUDE_expression_simplification_l2893_289354

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 1) 
  (hb : b = Real.sqrt 3 - 1) : 
  ((a^2 / (a - b) - (2*a*b - b^2) / (a - b)) / ((a - b) / (a * b))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2893_289354


namespace NUMINAMATH_CALUDE_part1_correct_part2_correct_l2893_289310

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (-3*m - 4, 2 + m)

-- Define point Q
def Q : ℝ × ℝ := (5, 8)

-- Theorem for part 1
theorem part1_correct :
  ∃ m : ℝ, P m = (-10, 4) ∧ (P m).2 = 4 := by sorry

-- Theorem for part 2
theorem part2_correct :
  ∃ m : ℝ, P m = (5, -1) ∧ (P m).1 = Q.1 := by sorry

end NUMINAMATH_CALUDE_part1_correct_part2_correct_l2893_289310


namespace NUMINAMATH_CALUDE_roman_numeral_calculation_l2893_289399

/-- Roman numeral values -/
def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

/-- The theorem to prove -/
theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end NUMINAMATH_CALUDE_roman_numeral_calculation_l2893_289399


namespace NUMINAMATH_CALUDE_male_students_count_l2893_289368

theorem male_students_count (total : ℕ) (male : ℕ) (female : ℕ) :
  total = 48 →
  female = (4 * male) / 5 + 3 →
  total = male + female →
  male = 25 := by
sorry

end NUMINAMATH_CALUDE_male_students_count_l2893_289368


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2893_289353

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k < n, (7^k : ℤ) % 4 ≠ k^7 % 4) ∧ (7^n : ℤ) % 4 = n^7 % 4 ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2893_289353


namespace NUMINAMATH_CALUDE_parabola_b_value_l2893_289346

/-- Given a parabola y = ax^2 + bx + c with vertex (p, -p) and passing through (0, p),
    where p ≠ 0, the value of b is -4/p. -/
theorem parabola_b_value (a b c p : ℝ) (h_p : p ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 - p) →
  (a * 0^2 + b * 0 + c = p) →
  b = -4 / p := by sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2893_289346


namespace NUMINAMATH_CALUDE_solve_for_a_when_x_is_zero_range_of_a_when_x_is_one_l2893_289391

-- Define the equation
def equation (a : ℚ) (x : ℚ) : Prop :=
  |a| * x = |a + 1| - x

-- Theorem 1
theorem solve_for_a_when_x_is_zero :
  ∀ a : ℚ, equation a 0 → a = -1 :=
sorry

-- Theorem 2
theorem range_of_a_when_x_is_one :
  ∀ a : ℚ, equation a 1 → a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_when_x_is_zero_range_of_a_when_x_is_one_l2893_289391


namespace NUMINAMATH_CALUDE_find_x_l2893_289311

-- Define the conditions
def condition1 (x : ℕ) : Prop := 3 * x > 0
def condition2 (x : ℕ) : Prop := x ≥ 10
def condition3 (x : ℕ) : Prop := x > 5

-- Theorem statement
theorem find_x : ∃ (x : ℕ), condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2893_289311


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2893_289381

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 4) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2893_289381


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l2893_289357

/-- The area of a square containing four circles of radius 7 inches, 
    arranged so that two circles fit into the width and height of the square. -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l2893_289357


namespace NUMINAMATH_CALUDE_largest_of_three_l2893_289323

theorem largest_of_three (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p*q + p*r + q*r = -6)
  (prod_eq : p*q*r = -18) :
  max p (max q r) = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_l2893_289323


namespace NUMINAMATH_CALUDE_wall_area_calculation_l2893_289326

theorem wall_area_calculation (regular_area : ℝ) (jumbo_ratio : ℝ) (length_ratio : ℝ) :
  regular_area = 70 →
  jumbo_ratio = 1 / 3 →
  length_ratio = 3 →
  (regular_area + jumbo_ratio / (1 - jumbo_ratio) * regular_area * length_ratio) = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_wall_area_calculation_l2893_289326


namespace NUMINAMATH_CALUDE_green_hats_not_adjacent_probability_l2893_289351

def total_children : ℕ := 9
def green_hats : ℕ := 3

theorem green_hats_not_adjacent_probability :
  let total_arrangements := Nat.choose total_children green_hats
  let adjacent_arrangements := (total_children - green_hats + 1) + (total_children - 1) * (total_children - green_hats - 1)
  (total_arrangements - adjacent_arrangements : ℚ) / total_arrangements = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_green_hats_not_adjacent_probability_l2893_289351


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_ge_half_l2893_289370

/-- The polynomial p(x) = x^4 + bx^3 + x^2 + bx - 1 -/
def p (b : ℝ) (x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x - 1

/-- The polynomial p(x) has at least one real root -/
def has_real_root (b : ℝ) : Prop := ∃ x : ℝ, p b x = 0

theorem polynomial_real_root_iff_b_ge_half :
  ∀ b : ℝ, has_real_root b ↔ b ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_ge_half_l2893_289370


namespace NUMINAMATH_CALUDE_logan_snowfall_total_l2893_289348

/-- Represents the snowfall recorded over three days during a snowstorm -/
structure SnowfallRecord where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total snowfall from a three-day record -/
def totalSnowfall (record : SnowfallRecord) : ℝ :=
  record.wednesday + record.thursday + record.friday

/-- Theorem stating that Logan's recorded snowfall totals 0.88 cm -/
theorem logan_snowfall_total :
  let record : SnowfallRecord := {
    wednesday := 0.33,
    thursday := 0.33,
    friday := 0.22
  }
  totalSnowfall record = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_logan_snowfall_total_l2893_289348


namespace NUMINAMATH_CALUDE_union_implies_m_equals_two_l2893_289344

theorem union_implies_m_equals_two (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_union_implies_m_equals_two_l2893_289344


namespace NUMINAMATH_CALUDE_m_range_l2893_289332

def p (x : ℝ) : Prop := |x - 3| ≤ 2

def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- ¬p is a sufficient but not necessary condition for ¬q
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ (∃ x, ¬(q x m) ∧ p x)

theorem m_range :
  ∀ m, sufficient_not_necessary m ↔ 2 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2893_289332


namespace NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l2893_289390

theorem root_in_interval_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + x + a) 
  (h2 : a < 0) 
  (h3 : ∃ x ∈ Set.Ioo 0 1, f x = 0) : 
  -2 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l2893_289390


namespace NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l2893_289337

theorem quadratic_root_implies_v_value : ∀ v : ℝ,
  ((-25 - Real.sqrt 361) / 12 : ℝ) ∈ {x : ℝ | 6 * x^2 + 25 * x + v = 0} →
  v = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l2893_289337


namespace NUMINAMATH_CALUDE_soccer_league_teams_l2893_289330

theorem soccer_league_teams (n : ℕ) : n * (n - 1) / 2 = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l2893_289330


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2893_289316

theorem negation_of_existence (f : ℝ → Prop) : 
  (¬ ∃ x, f x) ↔ (∀ x, ¬ f x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x - 3 < 0) ↔ (∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2893_289316


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5_l2893_289365

/-- Molecular weight calculation for Dinitrogen pentoxide (N2O5) -/
theorem molecular_weight_N2O5 (atomic_weight_N atomic_weight_O : ℝ)
  (h1 : atomic_weight_N = 14.01)
  (h2 : atomic_weight_O = 16.00) :
  2 * atomic_weight_N + 5 * atomic_weight_O = 108.02 := by
  sorry

#check molecular_weight_N2O5

end NUMINAMATH_CALUDE_molecular_weight_N2O5_l2893_289365


namespace NUMINAMATH_CALUDE_min_A_over_C_l2893_289363

theorem min_A_over_C (x A C : ℝ) (hx : x > 0) (hA : A > 0) (hC : C > 0)
  (hdefA : x^2 + 1/x^2 = A) (hdefC : x + 1/x = C) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ y, y = A / C → y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_A_over_C_l2893_289363


namespace NUMINAMATH_CALUDE_dad_steps_l2893_289374

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  3 * s.masha = 5 * s.dad

/-- The ratio of steps between Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  3 * s.yasha = 5 * s.masha

/-- The total number of steps taken by Masha and Yasha -/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : masha_yasha_total s) :
  s.dad = 90 := by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l2893_289374


namespace NUMINAMATH_CALUDE_chips_in_bag_is_81_l2893_289334

/-- Represents the number of chocolate chips in a bag -/
def chips_in_bag : ℕ := sorry

/-- Represents the number of batches made from one bag of chips -/
def batches_per_bag : ℕ := 3

/-- Represents the number of cookies in each batch -/
def cookies_per_batch : ℕ := 3

/-- Represents the number of chocolate chips in each cookie -/
def chips_per_cookie : ℕ := 9

/-- Theorem stating that the number of chips in a bag is 81 -/
theorem chips_in_bag_is_81 : chips_in_bag = 81 := by sorry

end NUMINAMATH_CALUDE_chips_in_bag_is_81_l2893_289334


namespace NUMINAMATH_CALUDE_average_equals_x_l2893_289314

theorem average_equals_x (x : ℝ) : 
  (2 + 5 + x + 14 + 15) / 5 = x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_equals_x_l2893_289314


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2893_289385

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def equation (z : ℂ) : Prop := z * (1 - i) = 2 - i

-- Theorem statement
theorem z_in_first_quadrant (z : ℂ) (h : equation z) : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2893_289385


namespace NUMINAMATH_CALUDE_quadratic_equation_with_root_three_l2893_289360

theorem quadratic_equation_with_root_three :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 3*x = 0) ∧ (3 : ℝ) ∈ {x : ℝ | a * x^2 + b * x + c = 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_root_three_l2893_289360


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2893_289303

theorem smaller_number_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 45) (h4 : y = 4 * x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2893_289303


namespace NUMINAMATH_CALUDE_short_trees_calculation_l2893_289366

/-- The number of short trees initially in the park -/
def initial_short_trees : ℕ := 41

/-- The number of short trees to be planted -/
def planted_short_trees : ℕ := 57

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 98

/-- Theorem stating that the initial number of short trees plus the planted short trees equals the total short trees -/
theorem short_trees_calculation : 
  initial_short_trees + planted_short_trees = total_short_trees :=
by sorry

end NUMINAMATH_CALUDE_short_trees_calculation_l2893_289366


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l2893_289324

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  lastTwoDigits (sumFactorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l2893_289324


namespace NUMINAMATH_CALUDE_boys_at_reunion_l2893_289398

/-- The number of handshakes between n boys, where each boy shakes hands
    exactly once with each of the others. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There were 11 boys at the reunion given that the total number
    of handshakes was 55 and each boy shook hands exactly once with each
    of the others. -/
theorem boys_at_reunion : ∃ (n : ℕ), n > 0 ∧ handshakes n = 55 ∧ n = 11 := by
  sorry

#eval handshakes 11  -- This should output 55

end NUMINAMATH_CALUDE_boys_at_reunion_l2893_289398


namespace NUMINAMATH_CALUDE_persons_age_l2893_289369

theorem persons_age : ∃ (age : ℕ), 
  (5 * (age + 7) - 3 * (age - 7) = age) ∧ 
  (age > 0) := by
  sorry

end NUMINAMATH_CALUDE_persons_age_l2893_289369


namespace NUMINAMATH_CALUDE_min_sum_abc_l2893_289333

theorem min_sum_abc (a b c : ℕ+) (h : (a : ℚ) / 77 + (b : ℚ) / 91 + (c : ℚ) / 143 = 1) :
  ∃ (a' b' c' : ℕ+), (a' : ℚ) / 77 + (b' : ℚ) / 91 + (c' : ℚ) / 143 = 1 ∧
    (∀ (x y z : ℕ+), (x : ℚ) / 77 + (y : ℚ) / 91 + (z : ℚ) / 143 = 1 → 
      a' + b' + c' ≤ x + y + z) ∧
    a' + b' + c' = 79 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_l2893_289333


namespace NUMINAMATH_CALUDE_combined_box_weight_l2893_289394

def box1_weight : ℕ := 2
def box2_weight : ℕ := 11
def box3_weight : ℕ := 5

theorem combined_box_weight :
  box1_weight + box2_weight + box3_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_combined_box_weight_l2893_289394


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2893_289377

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 7) (h2 : x * y = 5) : x^2 + y^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2893_289377


namespace NUMINAMATH_CALUDE_inscribed_square_arc_length_l2893_289320

/-- Given a square inscribed in a circle with side length 4,
    the arc length intercepted by any side of the square is √2π. -/
theorem inscribed_square_arc_length (s : Real) (r : Real) (arc_length : Real) :
  s = 4 →                        -- Side length of the square is 4
  r = 2 * Real.sqrt 2 →          -- Radius of the circle
  arc_length = Real.sqrt 2 * π → -- Arc length intercepted by any side
  True :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_arc_length_l2893_289320


namespace NUMINAMATH_CALUDE_power_equality_l2893_289378

theorem power_equality (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2893_289378


namespace NUMINAMATH_CALUDE_smallest_x_value_l2893_289375

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (215 + x)) : 
  ∀ z : ℕ+, z < x → (3 : ℚ) / 4 ≠ y / (215 + z) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2893_289375
