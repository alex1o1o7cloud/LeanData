import Mathlib

namespace NUMINAMATH_CALUDE_eden_initial_bears_count_l1021_102170

/-- Represents the number of stuffed bears Eden had initially --/
def eden_initial_bears : ℕ := 10

/-- Represents the total number of stuffed bears Daragh had initially --/
def daragh_initial_bears : ℕ := 20

/-- Represents the number of favorite bears Daragh took out --/
def daragh_favorite_bears : ℕ := 8

/-- Represents the number of Daragh's sisters --/
def number_of_sisters : ℕ := 3

/-- Represents the number of stuffed bears Eden has now --/
def eden_current_bears : ℕ := 14

theorem eden_initial_bears_count :
  eden_initial_bears =
    eden_current_bears -
    ((daragh_initial_bears - daragh_favorite_bears) / number_of_sisters) :=
by
  sorry

#eval eden_initial_bears

end NUMINAMATH_CALUDE_eden_initial_bears_count_l1021_102170


namespace NUMINAMATH_CALUDE_largest_divisible_power_of_three_l1021_102138

theorem largest_divisible_power_of_three : ∃! n : ℕ, 
  (∀ k : ℕ, k ≤ n → (4^27000 - 82) % 3^k = 0) ∧
  (∀ m : ℕ, m > n → (4^27000 - 82) % 3^m ≠ 0) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_power_of_three_l1021_102138


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l1021_102156

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  (Nat.factorial 4) * 
  (Nat.factorial chickens) * 
  (Nat.factorial dogs) * 
  (Nat.factorial cats) * 
  (Nat.factorial rabbits)

/-- Theorem stating the number of arrangements for the given problem -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 3 3 5 2 = 207360 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l1021_102156


namespace NUMINAMATH_CALUDE_ship_placement_theorem_l1021_102199

/-- Represents a ship on the grid -/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Represents the grid -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Checks if a ship placement is valid -/
def isValidPlacement (grid : Grid) (ship : Ship) (x y : Fin 10) : Bool :=
  sorry

/-- Places a ship on the grid -/
def placeShip (grid : Grid) (ship : Ship) (x y : Fin 10) : Grid :=
  sorry

/-- List of ships to be placed -/
def ships : List Ship :=
  [⟨4, 1⟩, ⟨3, 1⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩]

/-- Attempts to place all ships on the grid -/
def placeAllShips (grid : Grid) (ships : List Ship) : Option Grid :=
  sorry

theorem ship_placement_theorem :
  (∃ (grid : Grid), placeAllShips grid ships = some grid) ∧
  (∃ (grid : Grid), placeAllShips grid (ships.reverse) = none) :=
by sorry

end NUMINAMATH_CALUDE_ship_placement_theorem_l1021_102199


namespace NUMINAMATH_CALUDE_palindrome_probability_l1021_102168

/-- A function that checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- A function that generates all 5-digit palindromes -/
def fiveDigitPalindromes : Finset ℕ := sorry

/-- The total number of 5-digit palindromes -/
def totalPalindromes : ℕ := Finset.card fiveDigitPalindromes

/-- A function that checks if a number is divisible by another number -/
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

/-- The set of 5-digit palindromes m where m/7 is a palindrome and divisible by 11 -/
def validPalindromes : Finset ℕ := sorry

/-- The number of valid palindromes -/
def validCount : ℕ := Finset.card validPalindromes

theorem palindrome_probability :
  (validCount : ℚ) / totalPalindromes = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_palindrome_probability_l1021_102168


namespace NUMINAMATH_CALUDE_frame_purchase_remaining_money_l1021_102187

theorem frame_purchase_remaining_money 
  (budget : ℝ) 
  (initial_frame_price_increase : ℝ) 
  (smaller_frame_price_ratio : ℝ) :
  budget = 60 →
  initial_frame_price_increase = 0.2 →
  smaller_frame_price_ratio = 3/4 →
  budget - (budget * (1 + initial_frame_price_increase) * smaller_frame_price_ratio) = 6 := by
  sorry

end NUMINAMATH_CALUDE_frame_purchase_remaining_money_l1021_102187


namespace NUMINAMATH_CALUDE_books_per_shelf_l1021_102140

theorem books_per_shelf 
  (total_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : total_shelves = 14240)
  (h2 : total_books = 113920) :
  total_books / total_shelves = 8 :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1021_102140


namespace NUMINAMATH_CALUDE_rounding_down_2A3_l1021_102110

def round_down_to_nearest_ten (n : ℕ) : ℕ :=
  (n / 10) * 10

theorem rounding_down_2A3 (A : ℕ) (h1 : A < 10) :
  (round_down_to_nearest_ten (200 + 10 * A + 3) = 280) → A = 8 := by
  sorry

end NUMINAMATH_CALUDE_rounding_down_2A3_l1021_102110


namespace NUMINAMATH_CALUDE_unique_line_intersection_l1021_102161

theorem unique_line_intersection (m b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! k : ℝ, ∃ y1 y2 : ℝ, 
    y1 = k^2 - 2*k + 3 ∧ 
    y2 = m*k + b ∧ 
    |y1 - y2| = 4)
  (h3 : m * 2 + b = 8) : 
  m = 0 ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_line_intersection_l1021_102161


namespace NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l1021_102171

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  ((x - 2) / (x + 2) + 4 * x / (x^2 - 4)) / (4 * x / (x^2 - 4)) = (x^2 + 4) / (4 * x) :=
by sorry

theorem expression_evaluation :
  let x : ℝ := 1
  ((x - 2) / (x + 2) + 4 * x / (x^2 - 4)) / (4 * x / (x^2 - 4)) = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l1021_102171


namespace NUMINAMATH_CALUDE_toy_purchase_cost_l1021_102133

theorem toy_purchase_cost (yoyo_cost whistle_cost : ℕ) 
  (h1 : yoyo_cost = 24) 
  (h2 : whistle_cost = 14) : 
  yoyo_cost + whistle_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_cost_l1021_102133


namespace NUMINAMATH_CALUDE_new_person_weight_l1021_102177

/-- Given a group of 8 people where one person weighing 40 kg is replaced,
    if the average weight increases by 2.5 kg, then the new person weighs 60 kg. -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1021_102177


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1021_102186

theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side : ℝ) (rect_width : ℝ),
    pentagon_side * 5 = 60 →
    rect_width * 6 = 40 →
    pentagon_side / rect_width = 9 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1021_102186


namespace NUMINAMATH_CALUDE_find_divisor_l1021_102181

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 163 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 163 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1021_102181


namespace NUMINAMATH_CALUDE_custom_operation_equality_l1021_102184

/-- Custom binary operation ⊗ -/
def otimes (a b : ℚ) : ℚ := a^2 / b

/-- Theorem statement -/
theorem custom_operation_equality : 
  (otimes (otimes 3 4) 6) - (otimes 3 (otimes 4 6)) - 1 = -113/32 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l1021_102184


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_4_l1021_102119

theorem greatest_three_digit_divisible_by_3_6_4 : ∃ n : ℕ, 
  n = 984 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 3 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 6 = 0 ∧ m % 4 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_4_l1021_102119


namespace NUMINAMATH_CALUDE_trapezoid_area_l1021_102172

/-- Given an outer equilateral triangle with area 16, an inner equilateral triangle
    with area 1, and three congruent trapezoids between them, the area of one
    trapezoid is 5. -/
theorem trapezoid_area (outer_area inner_area : ℝ) (num_trapezoids : ℕ) :
  outer_area = 16 →
  inner_area = 1 →
  num_trapezoids = 3 →
  (outer_area - inner_area) / num_trapezoids = 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1021_102172


namespace NUMINAMATH_CALUDE_inequality_proof_l1021_102144

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1/a) + (1/b) + (1/c) ≤ (a^8 + b^8 + c^8) / (a^3 * b^3 * c^3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1021_102144


namespace NUMINAMATH_CALUDE_prob_white_ball_l1021_102158

/-- Represents an urn with a certain number of black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of choosing each urn -/
def urn_choice_prob : ℚ := 1/2

/-- The two urns in the problem -/
def urn1 : Urn := ⟨2, 3⟩
def urn2 : Urn := ⟨2, 1⟩

/-- The probability of drawing a white ball from a given urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The theorem stating the probability of drawing a white ball -/
theorem prob_white_ball : 
  urn_choice_prob * prob_white urn1 + urn_choice_prob * prob_white urn2 = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_ball_l1021_102158


namespace NUMINAMATH_CALUDE_balloon_count_impossible_l1021_102125

theorem balloon_count_impossible : ¬∃ (b g : ℕ), 3 * (b + g) = 100 := by
  sorry

#check balloon_count_impossible

end NUMINAMATH_CALUDE_balloon_count_impossible_l1021_102125


namespace NUMINAMATH_CALUDE_dog_distance_l1021_102103

/-- The total distance run by a dog between two people walking towards each other -/
theorem dog_distance (total_distance : ℝ) (speed_A speed_B speed_dog : ℝ) : 
  total_distance = 100 ∧ 
  speed_A = 6 ∧ 
  speed_B = 4 ∧ 
  speed_dog = 10 → 
  (total_distance / (speed_A + speed_B)) * speed_dog = 100 :=
by sorry

end NUMINAMATH_CALUDE_dog_distance_l1021_102103


namespace NUMINAMATH_CALUDE_imaginary_power_2016_l1021_102190

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_2016 : i ^ 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_2016_l1021_102190


namespace NUMINAMATH_CALUDE_complex_square_root_l1021_102193

theorem complex_square_root (z : ℂ) (h : z ^ 2 = 3 + 4 * I) :
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_square_root_l1021_102193


namespace NUMINAMATH_CALUDE_eighteen_consecutive_divisible_by_sum_l1021_102148

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem eighteen_consecutive_divisible_by_sum (start : ℕ) 
  (h : ∀ k, 0 ≤ k ∧ k < 18 → isThreeDigit (start + k)) :
  ∃ n, 0 ≤ n ∧ n < 18 ∧ (start + n) % (sumOfDigits (start + n)) = 0 :=
sorry

end NUMINAMATH_CALUDE_eighteen_consecutive_divisible_by_sum_l1021_102148


namespace NUMINAMATH_CALUDE_trip_time_difference_l1021_102120

theorem trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) :
  speed = 60 → distance1 = 540 → distance2 = 570 →
  (distance2 - distance1) / speed * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l1021_102120


namespace NUMINAMATH_CALUDE_red_straws_per_mat_l1021_102188

theorem red_straws_per_mat (orange_per_mat green_per_mat total_straws mats : ℕ)
  (h1 : orange_per_mat = 30)
  (h2 : green_per_mat = orange_per_mat / 2)
  (h3 : total_straws = 650)
  (h4 : mats = 10) :
  (total_straws - (orange_per_mat + green_per_mat) * mats) / mats = 20 :=
by sorry

end NUMINAMATH_CALUDE_red_straws_per_mat_l1021_102188


namespace NUMINAMATH_CALUDE_find_divisor_l1021_102127

theorem find_divisor : ∃ (D : ℕ), 
  (23 = 5 * D + 3) ∧ 
  (∃ (N : ℕ), N = 7 * D + 5) ∧ 
  D = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1021_102127


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1021_102105

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2*x > -m*x)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1021_102105


namespace NUMINAMATH_CALUDE_intersection_M_N_l1021_102167

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1021_102167


namespace NUMINAMATH_CALUDE_optimal_solution_l1021_102182

/-- Represents a vehicle type with its capacity, quantity, and fuel efficiency. -/
structure VehicleType where
  capacity : Nat
  quantity : Nat
  fuelEfficiency : Nat

/-- Represents the problem setup for the field trip. -/
structure FieldTripProblem where
  cars : VehicleType
  minivans : VehicleType
  buses : VehicleType
  totalPeople : Nat
  tripDistance : Nat

/-- Represents a solution to the field trip problem. -/
structure FieldTripSolution where
  numCars : Nat
  numMinivans : Nat
  numBuses : Nat

def fuelUsage (problem : FieldTripProblem) (solution : FieldTripSolution) : Rat :=
  (problem.tripDistance * solution.numCars / problem.cars.fuelEfficiency : Rat) +
  (problem.tripDistance * solution.numMinivans / problem.minivans.fuelEfficiency : Rat) +
  (problem.tripDistance * solution.numBuses / problem.buses.fuelEfficiency : Rat)

def totalCapacity (problem : FieldTripProblem) (solution : FieldTripSolution) : Nat :=
  solution.numCars * problem.cars.capacity +
  solution.numMinivans * problem.minivans.capacity +
  solution.numBuses * problem.buses.capacity

def isValidSolution (problem : FieldTripProblem) (solution : FieldTripSolution) : Prop :=
  solution.numCars ≤ problem.cars.quantity ∧
  solution.numMinivans ≤ problem.minivans.quantity ∧
  solution.numBuses ≤ problem.buses.quantity ∧
  totalCapacity problem solution ≥ problem.totalPeople

theorem optimal_solution (problem : FieldTripProblem) (solution : FieldTripSolution) :
  problem.cars = { capacity := 4, quantity := 3, fuelEfficiency := 30 } ∧
  problem.minivans = { capacity := 6, quantity := 2, fuelEfficiency := 20 } ∧
  problem.buses = { capacity := 20, quantity := 1, fuelEfficiency := 10 } ∧
  problem.totalPeople = 33 ∧
  problem.tripDistance = 50 ∧
  solution = { numCars := 1, numMinivans := 1, numBuses := 1 } ∧
  isValidSolution problem solution →
  ∀ (altSolution : FieldTripSolution),
    isValidSolution problem altSolution →
    fuelUsage problem solution ≤ fuelUsage problem altSolution :=
by sorry

end NUMINAMATH_CALUDE_optimal_solution_l1021_102182


namespace NUMINAMATH_CALUDE_bumper_car_line_after_three_rounds_l1021_102150

def bumper_car_line (initial_people : ℕ) (capacity : ℕ) (leave_once : ℕ) (priority_join : ℕ) (rounds : ℕ) : ℕ :=
  let first_round := initial_people - capacity - leave_once + priority_join
  let subsequent_rounds := first_round - (rounds - 1) * capacity + (rounds - 1) * priority_join
  subsequent_rounds

theorem bumper_car_line_after_three_rounds :
  bumper_car_line 30 5 10 5 3 = 20 := by sorry

end NUMINAMATH_CALUDE_bumper_car_line_after_three_rounds_l1021_102150


namespace NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l1021_102128

theorem smallest_integer_solution_inequality :
  ∀ (x : ℤ), (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 ∧
  ∃ (y : ℤ), y < -2 ∧ (9 * y + 8) / 6 - y / 3 < -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l1021_102128


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1021_102130

/-- An arithmetic sequence with positive terms where a_1 and a_3 are roots of x^2 - 8x + 7 = 0 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (a 1)^2 - 8*(a 1) + 7 = 0 ∧
  (a 3)^2 - 8*(a 3) + 7 = 0

/-- The general formula for the arithmetic sequence -/
def GeneralFormula (n : ℕ) : ℝ := 3 * n - 2

/-- Theorem stating that the general formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∀ n, a n = GeneralFormula n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1021_102130


namespace NUMINAMATH_CALUDE_true_propositions_l1021_102100

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom p₁_true : p₁
axiom p₂_false : ¬p₂
axiom p₃_false : ¬p₃
axiom p₄_true : p₄

-- Define the compound propositions
def prop1 := p₁ ∧ p₄
def prop2 := p₁ ∧ p₂
def prop3 := ¬p₂ ∨ p₃
def prop4 := ¬p₃ ∨ ¬p₄

-- Theorem to prove
theorem true_propositions : 
  prop1 p₁ p₄ ∧ prop3 p₂ p₃ ∧ prop4 p₃ p₄ ∧ ¬(prop2 p₁ p₂) :=
sorry

end NUMINAMATH_CALUDE_true_propositions_l1021_102100


namespace NUMINAMATH_CALUDE_min_value_of_parallel_lines_l1021_102101

theorem min_value_of_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  ∀ x y : ℝ, 2 * a + 3 * b ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_parallel_lines_l1021_102101


namespace NUMINAMATH_CALUDE_remainder_double_n_l1021_102192

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l1021_102192


namespace NUMINAMATH_CALUDE_supplement_complement_difference_l1021_102132

/-- An acute angle is between 0° and 90° -/
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

/-- The supplement of an angle θ -/
def supplement (θ : ℝ) : ℝ := 180 - θ

/-- The complement of an angle θ -/
def complement (θ : ℝ) : ℝ := 90 - θ

/-- For any acute angle, the difference between its supplement and complement is 90° -/
theorem supplement_complement_difference (θ : ℝ) (h : is_acute_angle θ) :
  supplement θ - complement θ = 90 := by
  sorry

end NUMINAMATH_CALUDE_supplement_complement_difference_l1021_102132


namespace NUMINAMATH_CALUDE_simplify_expression_l1021_102115

theorem simplify_expression : ((3 + 4 + 5 + 6) / 2) + ((3 * 6 + 9) / 3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1021_102115


namespace NUMINAMATH_CALUDE_dropped_student_score_l1021_102131

theorem dropped_student_score
  (total_students : ℕ)
  (remaining_students : ℕ)
  (initial_average : ℚ)
  (final_average : ℚ)
  (h1 : total_students = 16)
  (h2 : remaining_students = 15)
  (h3 : initial_average = 61.5)
  (h4 : final_average = 64)
  : (total_students : ℚ) * initial_average - (remaining_students : ℚ) * final_average = 24 := by
  sorry

end NUMINAMATH_CALUDE_dropped_student_score_l1021_102131


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_prism_l1021_102149

theorem sphere_surface_area_of_circumscribed_prism (h : ℝ) (v : ℝ) (r : ℝ) :
  h = 4 →
  v = 16 →
  v = h * r^2 →
  let d := Real.sqrt (h^2 + 2 * r^2)
  (4 / 3) * π * (d / 2)^3 = (4 / 3) * π * r^2 * h →
  4 * π * (d / 2)^2 = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_prism_l1021_102149


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1021_102173

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x, 2 * f x + f (1 - x) = x^2) →
  (∀ x, f x = (1/3) * x^2 + (2/3) * x - 1/3) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1021_102173


namespace NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l1021_102160

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Define the fourth quadrant
def fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

-- Theorem statement
theorem reciprocal_in_fourth_quadrant :
  fourth_quadrant (z⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l1021_102160


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l1021_102185

/-- The number of routes on a grid from (0,0) to (m,n) moving only right or down -/
def numRoutes (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- The width of the grid -/
def gridWidth : ℕ := 3

/-- The height of the grid -/
def gridHeight : ℕ := 2

theorem routes_on_3x2_grid : 
  numRoutes gridWidth gridHeight = 10 := by
  sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l1021_102185


namespace NUMINAMATH_CALUDE_log_equality_l1021_102113

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equality : log10 2 + 2 * log10 5 = 1 + log10 5 := by sorry

end NUMINAMATH_CALUDE_log_equality_l1021_102113


namespace NUMINAMATH_CALUDE_base_six_conversion_and_addition_l1021_102118

def base_six_to_ten (n : Nat) : Nat :=
  (n % 10) + 6 * ((n / 10) % 10) + 36 * (n / 100)

theorem base_six_conversion_and_addition :
  let base_six_num := 214
  let base_ten_num := base_six_to_ten base_six_num
  base_ten_num = 82 ∧ base_ten_num + 15 = 97 := by
  sorry

end NUMINAMATH_CALUDE_base_six_conversion_and_addition_l1021_102118


namespace NUMINAMATH_CALUDE_expression_evaluation_l1021_102112

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := 2
  2*(a+b)*(a-b) - (a+b)^2 + a*(2*a+b) = -11 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1021_102112


namespace NUMINAMATH_CALUDE_max_common_tangents_shared_focus_l1021_102166

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  foci : Fin 2 → ℝ × ℝ
  majorAxis : ℝ

/-- Represents a tangent line to an ellipse -/
structure Tangent where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Returns the number of common tangents between two ellipses -/
def commonTangents (e1 e2 : Ellipse) : ℕ := sorry

/-- Theorem: The maximum number of common tangents for two ellipses sharing one focus is 2 -/
theorem max_common_tangents_shared_focus (e1 e2 : Ellipse) 
  (h : e1.foci 1 = e2.foci 1) : 
  commonTangents e1 e2 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_max_common_tangents_shared_focus_l1021_102166


namespace NUMINAMATH_CALUDE_centroid_count_l1021_102137

/-- A point on the perimeter of the square -/
structure PerimeterPoint where
  x : ℚ
  y : ℚ
  on_perimeter : (x = 0 ∨ x = 12) ∨ (y = 0 ∨ y = 12)
  valid_coord : (0 ≤ x ∧ x ≤ 12) ∧ (0 ≤ y ∧ y ≤ 12)

/-- The set of 48 equally spaced points on the perimeter -/
def perimeter_points : Finset PerimeterPoint :=
  sorry

/-- Predicate to check if two points are consecutive on the perimeter -/
def are_consecutive (p q : PerimeterPoint) : Prop :=
  sorry

/-- The centroid of a triangle given by three points -/
def centroid (p q r : PerimeterPoint) : ℚ × ℚ :=
  ((p.x + q.x + r.x) / 3, (p.y + q.y + r.y) / 3)

/-- The set of all possible centroids -/
def possible_centroids : Finset (ℚ × ℚ) :=
  sorry

theorem centroid_count :
  ∀ p q r : PerimeterPoint,
    p ∈ perimeter_points →
    q ∈ perimeter_points →
    r ∈ perimeter_points →
    ¬(are_consecutive p q ∨ are_consecutive q r ∨ are_consecutive r p) →
    (Finset.card possible_centroids = 1156) :=
  sorry

end NUMINAMATH_CALUDE_centroid_count_l1021_102137


namespace NUMINAMATH_CALUDE_hawks_score_l1021_102194

/-- Represents a basketball game between two teams -/
structure BasketballGame where
  total_score : ℕ
  winning_margin : ℕ

/-- Calculates the score of the losing team in a basketball game -/
def losing_team_score (game : BasketballGame) : ℕ :=
  (game.total_score - game.winning_margin) / 2

theorem hawks_score (game : BasketballGame) 
  (h1 : game.total_score = 82) 
  (h2 : game.winning_margin = 6) : 
  losing_team_score game = 38 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l1021_102194


namespace NUMINAMATH_CALUDE_rug_dimension_l1021_102174

theorem rug_dimension (floor_area : ℝ) (rug_width : ℝ) (coverage_fraction : ℝ) :
  floor_area = 64 ∧ 
  rug_width = 2 ∧ 
  coverage_fraction = 0.21875 →
  ∃ (rug_length : ℝ), 
    rug_length * rug_width = floor_area * coverage_fraction ∧
    rug_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_rug_dimension_l1021_102174


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l1021_102122

theorem magnitude_of_complex_power : 
  Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 2) ^ 6 = 1728 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l1021_102122


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1021_102107

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1021_102107


namespace NUMINAMATH_CALUDE_field_area_is_625_l1021_102102

/-- Represents a square field -/
structure SquareField where
  /-- The length of one side of the square field in kilometers -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- Calculates the perimeter of a square field -/
def perimeter (field : SquareField) : ℝ := 4 * field.side

/-- Calculates the area of a square field -/
def area (field : SquareField) : ℝ := field.side ^ 2

/-- The speed of the horse in km/h -/
def horse_speed : ℝ := 25

/-- The time taken by the horse to run around the field in hours -/
def lap_time : ℝ := 4

theorem field_area_is_625 (field : SquareField) 
  (h : perimeter field = horse_speed * lap_time) : 
  area field = 625 := by
  sorry

end NUMINAMATH_CALUDE_field_area_is_625_l1021_102102


namespace NUMINAMATH_CALUDE_number_game_l1021_102143

theorem number_game (x : ℤ) : 3 * (3 * (x + 3) - 3) = 3 * (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_number_game_l1021_102143


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1021_102163

theorem simplify_and_evaluate (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) = -2 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1021_102163


namespace NUMINAMATH_CALUDE_negation_of_implication_l1021_102129

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1021_102129


namespace NUMINAMATH_CALUDE_movie_expense_ratio_l1021_102196

/-- Proves the ratio of movie expenses to weekly allowance -/
theorem movie_expense_ratio (weekly_allowance : ℚ) (car_wash_earning : ℚ) (final_amount : ℚ) :
  weekly_allowance = 10 →
  car_wash_earning = 6 →
  final_amount = 11 →
  (weekly_allowance - (final_amount - car_wash_earning)) / weekly_allowance = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_movie_expense_ratio_l1021_102196


namespace NUMINAMATH_CALUDE_percentage_calculation_l1021_102134

theorem percentage_calculation (whole : ℝ) (part : ℝ) :
  whole = 475.25 →
  part = 129.89 →
  (part / whole) * 100 = 27.33 :=
by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1021_102134


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l1021_102123

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 → h1 > 0 → r2 > 0 → h2 > 0 →
  r2 = 1.1 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.21 * h2 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l1021_102123


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1021_102157

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  y : ℝ  -- second term
  b : ℝ  -- third term
  d : ℝ  -- common difference
  h1 : y = a + d  -- relation between y, a, and d
  h2 : b = a + 3 * d  -- relation between b, a, and d
  h3 : y / 2 = a + 3 * d  -- fourth term equals y/2

/-- The ratio of a to b in the given arithmetic sequence is 3/4 -/
theorem ratio_a_to_b (seq : ArithmeticSequence) : seq.a / seq.b = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1021_102157


namespace NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l1021_102116

theorem sequence_with_positive_triples_negative_sum : 
  ∃ (seq : Fin 20 → ℝ), 
    (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
    (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l1021_102116


namespace NUMINAMATH_CALUDE_trajectory_of_point_l1021_102164

/-- The trajectory of a point P satisfying |PF₁| + |PF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_of_point (P : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  (dist P F₁ + dist P F₂ = 8) → 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
by sorry


end NUMINAMATH_CALUDE_trajectory_of_point_l1021_102164


namespace NUMINAMATH_CALUDE_expression_evaluation_l1021_102124

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 4
  5 * x^y + 2 * y^x = 533 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1021_102124


namespace NUMINAMATH_CALUDE_total_birds_on_fence_l1021_102114

def initial_birds : ℕ := 4
def additional_birds : ℕ := 6

theorem total_birds_on_fence : 
  initial_birds + additional_birds = 10 := by sorry

end NUMINAMATH_CALUDE_total_birds_on_fence_l1021_102114


namespace NUMINAMATH_CALUDE_max_value_constraint_l1021_102145

theorem max_value_constraint (x y : ℝ) (h : 2 * x^2 + 3 * y^2 ≤ 12) :
  |x + 2*y| ≤ Real.sqrt 22 ∧ ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = 12 ∧ |x + 2*y| = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1021_102145


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l1021_102189

def initial_amount : ℚ := 204
def spend_half (x : ℚ) : ℚ := x / 2

theorem isabel_piggy_bank :
  spend_half (spend_half initial_amount) = 51 := by
  sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l1021_102189


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_recipes_l1021_102175

/-- Given a recipe that requires a certain amount of an ingredient and a total amount of that ingredient needed, calculate the number of recipes that can be made. -/
def recipes_to_make (cups_per_recipe : ℚ) (total_cups_needed : ℚ) : ℚ :=
  total_cups_needed / cups_per_recipe

/-- Prove that 23 recipes can be made given the conditions of the chocolate chip cookie problem. -/
theorem chocolate_chip_cookie_recipes : 
  recipes_to_make 2 46 = 23 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_recipes_l1021_102175


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1021_102111

def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (0, 1, -4)

theorem vector_magnitude_proof : ‖a - 2 • b‖ = Real.sqrt 122 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1021_102111


namespace NUMINAMATH_CALUDE_road_trip_duration_l1021_102135

/-- Road trip duration calculation -/
theorem road_trip_duration (jenna_distance : ℝ) (friend_distance : ℝ) 
  (jenna_speed : ℝ) (friend_speed : ℝ) (num_breaks : ℕ) (break_duration : ℝ) :
  jenna_distance = 200 →
  friend_distance = 100 →
  jenna_speed = 50 →
  friend_speed = 20 →
  num_breaks = 2 →
  break_duration = 0.5 →
  (jenna_distance / jenna_speed) + (friend_distance / friend_speed) + 
    (num_breaks : ℝ) * break_duration = 10 := by
  sorry

#check road_trip_duration

end NUMINAMATH_CALUDE_road_trip_duration_l1021_102135


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1021_102108

theorem inequality_solution_set (x : ℝ) :
  (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1021_102108


namespace NUMINAMATH_CALUDE_first_company_manager_percentage_l1021_102141

/-- Represents a company's workforce composition -/
structure Company where
  total : ℝ
  managers : ℝ

/-- The merged company resulting from two companies -/
def MergedCompany (c1 c2 : Company) : Company where
  total := c1.total + c2.total
  managers := c1.managers + c2.managers

theorem first_company_manager_percentage (c1 c2 : Company) :
  let merged := MergedCompany c1 c2
  (c1.total = 0.25 * merged.total) →
  (merged.managers = 0.25 * merged.total) →
  (c1.managers / c1.total = 0.25) := by
  sorry

end NUMINAMATH_CALUDE_first_company_manager_percentage_l1021_102141


namespace NUMINAMATH_CALUDE_sum_of_squares_values_l1021_102183

theorem sum_of_squares_values (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_values_l1021_102183


namespace NUMINAMATH_CALUDE_derivative_sin_at_pi_half_l1021_102162

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem derivative_sin_at_pi_half :
  deriv f (π / 2) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_sin_at_pi_half_l1021_102162


namespace NUMINAMATH_CALUDE_pascal_triangle_100th_row_10th_number_l1021_102179

theorem pascal_triangle_100th_row_10th_number :
  let n : ℕ := 99  -- row number (100 numbers in the row, so n + 1 = 100)
  let k : ℕ := 9   -- 10th number (0-indexed)
  (n.choose k) = (Nat.choose 99 9) := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_100th_row_10th_number_l1021_102179


namespace NUMINAMATH_CALUDE_store_sales_l1021_102169

theorem store_sales (dvd_count : ℕ) (dvd_cd_ratio : ℚ) : 
  dvd_count = 168 → dvd_cd_ratio = 1.6 → dvd_count + (dvd_count / dvd_cd_ratio).floor = 273 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_l1021_102169


namespace NUMINAMATH_CALUDE_connected_vertices_probability_is_correct_l1021_102195

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of choosing at least two connected vertices when selecting three random vertices -/
def connected_vertices_probability (d : RegularDodecahedron) : ℚ :=
  9 / 19

/-- Theorem stating the probability of choosing at least two connected vertices -/
theorem connected_vertices_probability_is_correct (d : RegularDodecahedron) :
  connected_vertices_probability d = 9 / 19 := by
  sorry


end NUMINAMATH_CALUDE_connected_vertices_probability_is_correct_l1021_102195


namespace NUMINAMATH_CALUDE_intersection_of_planes_l1021_102117

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 3*x + 4*y - 2*z = 5
def plane2 (x y z : ℝ) : Prop := 2*x + 3*y - z = 3

-- Define the line of intersection
def intersection_line (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / (-1) ∧ (y + 1) / (-1) = z / 1

-- Theorem statement
theorem intersection_of_planes :
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → intersection_line x y z :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_planes_l1021_102117


namespace NUMINAMATH_CALUDE_sum_of_digits_of_gcd_l1021_102178

-- Define the numbers given in the problem
def a : ℕ := 1305
def b : ℕ := 4665
def c : ℕ := 6905

-- Define the function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- State the theorem
theorem sum_of_digits_of_gcd : sum_of_digits (Nat.gcd (b - a) (Nat.gcd (c - b) (c - a))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_gcd_l1021_102178


namespace NUMINAMATH_CALUDE_mono_decreasing_inequality_l1021_102136

/-- A function f is monotonically decreasing on ℝ -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- Given a monotonically decreasing function f on ℝ,
    if f(2m) > f(1+m), then m < 1 -/
theorem mono_decreasing_inequality (f : ℝ → ℝ) (m : ℝ)
    (h_mono : MonotonicallyDecreasing f) (h_ineq : f (2 * m) > f (1 + m)) :
    m < 1 :=
  sorry

end NUMINAMATH_CALUDE_mono_decreasing_inequality_l1021_102136


namespace NUMINAMATH_CALUDE_farm_animal_problem_l1021_102126

/-- Prove that the number of cows is 9 given the conditions of the farm animal problem -/
theorem farm_animal_problem :
  ∃ (num_cows : ℕ),
    let num_chickens : ℕ := 8
    let num_ducks : ℕ := 3
    let total_legs : ℕ := 4 * num_cows + 2 * num_chickens + 2 * num_ducks
    let total_heads : ℕ := num_cows + num_chickens + 2 * num_ducks
    total_legs = 18 + 2 * total_heads ∧ num_cows = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animal_problem_l1021_102126


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_19_factorial_l1021_102121

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Counts the number of factors of 3 in n! -/
def count_factors_of_three (n : ℕ) : ℕ :=
  if n < 3 then 0
  else (n / 3) + count_factors_of_three (n / 3)

theorem greatest_power_of_three_in_19_factorial :
  ∀ n : ℕ, 3^n ∣ factorial 19 ↔ n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_19_factorial_l1021_102121


namespace NUMINAMATH_CALUDE_sin_3theta_l1021_102198

theorem sin_3theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 + Complex.I * Real.sqrt 2) / 2) : 
  Real.sin (3 * θ) = Real.sqrt 2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_3theta_l1021_102198


namespace NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_and_b_l1021_102159

/-- The function f(x) = x³ + ax² + b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

/-- The derivative of f(x) -/
def f_deriv (a x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem tangent_line_parallel_implies_a_and_b (a b : ℝ) : 
  f a b 1 = 0 ∧ f_deriv a 1 = -3 → a = -3 ∧ b = 2 := by
  sorry

#check tangent_line_parallel_implies_a_and_b

end NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_and_b_l1021_102159


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1021_102155

/-- A function that checks if a quadratic equation with coefficients based on A has positive integer solutions -/
def has_positive_integer_solutions (A : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - (2*A)*x + (A+1)*10 = 0 ∧ y^2 - (2*A)*y + (A+1)*10 = 0

/-- The theorem stating that there is exactly one single-digit positive integer A that satisfies the condition -/
theorem unique_quadratic_solution : 
  ∃! A : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ has_positive_integer_solutions A :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1021_102155


namespace NUMINAMATH_CALUDE_moli_payment_l1021_102165

/-- The cost of Moli's initial purchase -/
def initial_purchase (ribbon_cost clip_cost soap_cost : ℚ) : ℚ :=
  3 * ribbon_cost + 7 * clip_cost + soap_cost

/-- The cost of 4 ribbons, 10 clips, and 1 soap -/
def second_scenario (ribbon_cost clip_cost soap_cost : ℚ) : ℚ :=
  4 * ribbon_cost + 10 * clip_cost + soap_cost

/-- The cost of 1 ribbon, 1 clip, and 1 soap -/
def unit_cost (ribbon_cost clip_cost soap_cost : ℚ) : ℚ :=
  ribbon_cost + clip_cost + soap_cost

theorem moli_payment (ribbon_cost clip_cost soap_cost : ℚ) :
  second_scenario ribbon_cost clip_cost soap_cost = 164 ∧
  unit_cost ribbon_cost clip_cost soap_cost = 32 →
  initial_purchase ribbon_cost clip_cost soap_cost = 120 :=
by sorry

end NUMINAMATH_CALUDE_moli_payment_l1021_102165


namespace NUMINAMATH_CALUDE_polynomial_three_distinct_roots_l1021_102146

theorem polynomial_three_distinct_roots : 
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x : ℝ), (x - 4) * (x^2 + 4*x + 3) = 0 ↔ x = a ∨ x = b ∨ x = c :=
by sorry

end NUMINAMATH_CALUDE_polynomial_three_distinct_roots_l1021_102146


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_500_l1021_102180

theorem closest_integer_to_cube_root_500 : 
  ∀ n : ℤ, |n - ⌊(500 : ℝ)^(1/3)⌋| ≥ |8 - ⌊(500 : ℝ)^(1/3)⌋| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_500_l1021_102180


namespace NUMINAMATH_CALUDE_base_b_sum_equals_21_l1021_102147

-- Define the sum of single-digit numbers in base b
def sum_single_digits (b : ℕ) : ℕ := (b * (b - 1)) / 2

-- Define the value 21 in base b
def value_21_base_b (b : ℕ) : ℕ := 2 * b + 1

-- Theorem statement
theorem base_b_sum_equals_21 :
  ∃ b : ℕ, b > 1 ∧ sum_single_digits b = value_21_base_b b ∧ b = 7 :=
sorry

end NUMINAMATH_CALUDE_base_b_sum_equals_21_l1021_102147


namespace NUMINAMATH_CALUDE_binary_to_decimal_1010101_l1021_102152

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number -/
def binary_num : List Nat := [1, 0, 1, 0, 1, 0, 1]

theorem binary_to_decimal_1010101 :
  binary_to_decimal (binary_num.reverse) = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_1010101_l1021_102152


namespace NUMINAMATH_CALUDE_reporter_earnings_l1021_102139

/-- A reporter's earnings calculation --/
theorem reporter_earnings 
  (earnings_per_article : ℝ) 
  (articles : ℕ) 
  (total_hours : ℝ) 
  (words_per_minute : ℝ) 
  (earnings_per_hour : ℝ) 
  (h1 : earnings_per_article = 60) 
  (h2 : articles = 3) 
  (h3 : total_hours = 4) 
  (h4 : words_per_minute = 10) 
  (h5 : earnings_per_hour = 105) : 
  let total_earnings := earnings_per_hour * total_hours
  let total_words := words_per_minute * (total_hours * 60)
  let article_earnings := earnings_per_article * articles
  let word_earnings := total_earnings - article_earnings
  word_earnings / total_words = 0.1 := by
sorry

end NUMINAMATH_CALUDE_reporter_earnings_l1021_102139


namespace NUMINAMATH_CALUDE_sum_is_five_digits_l1021_102142

/-- Represents a digit (1-9) -/
def Digit := Fin 9

/-- The sum of 10765, AB4, and CB is always a 5-digit number -/
theorem sum_is_five_digits (A B C : Digit) (h : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  let AB4 := 100 * A.val + 10 * B.val + 4
  let CB := 10 * C.val + B.val
  let sum := 10765 + AB4 + CB
  9999 < sum ∧ sum < 100000 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_five_digits_l1021_102142


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_for_given_surveys_l1021_102176

/-- Represents different sampling methods -/
inductive SamplingMethod
| Stratified
| Random
| Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool

/-- Determines the optimal sampling method for a given survey -/
def optimal_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_distinct_groups then SamplingMethod.Stratified
  else if s.population_size ≤ 20 then SamplingMethod.Random
  else SamplingMethod.Systematic

/-- The first survey from the problem -/
def survey1 : Survey :=
  { population_size := 500
  , sample_size := 100
  , has_distinct_groups := true }

/-- The second survey from the problem -/
def survey2 : Survey :=
  { population_size := 12
  , sample_size := 3
  , has_distinct_groups := false }

theorem optimal_sampling_methods_for_given_surveys :
  optimal_sampling_method survey1 = SamplingMethod.Stratified ∧
  optimal_sampling_method survey2 = SamplingMethod.Random :=
sorry


end NUMINAMATH_CALUDE_optimal_sampling_methods_for_given_surveys_l1021_102176


namespace NUMINAMATH_CALUDE_product_equals_64_l1021_102191

theorem product_equals_64 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_64_l1021_102191


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1021_102197

theorem sum_of_fractions : 
  (3 : ℚ) / 100 + (2 : ℚ) / 1000 + (8 : ℚ) / 10000 + (5 : ℚ) / 100000 = 0.03285 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1021_102197


namespace NUMINAMATH_CALUDE_problem_statement_l1021_102151

theorem problem_statement (a b : ℝ) (h : 5 * a - 3 * b + 2 = 0) : 
  10 * a - 6 * b - 3 = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1021_102151


namespace NUMINAMATH_CALUDE_salt_solution_volume_l1021_102154

/-- Given a mixture of pure water and a salt solution, calculates the volume of the salt solution needed to achieve a specific concentration. -/
theorem salt_solution_volume 
  (pure_water_volume : ℝ)
  (salt_solution_concentration : ℝ)
  (final_concentration : ℝ)
  (h1 : pure_water_volume = 1)
  (h2 : salt_solution_concentration = 0.75)
  (h3 : final_concentration = 0.15) :
  ∃ x : ℝ, x = 0.25 ∧ 
    salt_solution_concentration * x = final_concentration * (pure_water_volume + x) :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l1021_102154


namespace NUMINAMATH_CALUDE_train_length_train_length_alt_l1021_102106

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 9 → speed * time * (5 / 18) = 180 := by
  sorry

/-- Alternative formulation using more basic definitions -/
theorem train_length_alt (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 72 → time_s = 9 → 
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_alt_l1021_102106


namespace NUMINAMATH_CALUDE_area_ratio_quadrupled_triangle_l1021_102104

/-- Given a triangle whose dimensions are quadrupled to form a larger triangle,
    this theorem relates the area of the larger triangle to the area of the original triangle. -/
theorem area_ratio_quadrupled_triangle (A : ℝ) :
  (4 * 4 * A = 64) → (A = 4) := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_quadrupled_triangle_l1021_102104


namespace NUMINAMATH_CALUDE_andrew_stickers_l1021_102153

theorem andrew_stickers (daniel_stickers fred_stickers andrew_kept : ℕ) 
  (h1 : daniel_stickers = 250)
  (h2 : fred_stickers = daniel_stickers + 120)
  (h3 : andrew_kept = 130) :
  andrew_kept + daniel_stickers + fred_stickers = 750 :=
by sorry

end NUMINAMATH_CALUDE_andrew_stickers_l1021_102153


namespace NUMINAMATH_CALUDE_odd_function_property_l1021_102109

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x - 3) = f (x + 2))
  (h_value : f 1 = 2) :
  f 2011 - f 2010 = 2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1021_102109
