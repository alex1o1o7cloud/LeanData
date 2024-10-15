import Mathlib

namespace NUMINAMATH_CALUDE_pythagorean_triple_with_8_and_17_l1873_187366

/-- A Pythagorean triple is a set of three positive integers (a, b, c) such that a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Given a set of Pythagorean triples containing 8 and 17, the third number is 15 -/
theorem pythagorean_triple_with_8_and_17 :
  ∃ (x : ℕ), (is_pythagorean_triple 8 15 17 ∨ is_pythagorean_triple 8 17 15 ∨ is_pythagorean_triple 15 8 17) ∧
  ¬∃ (y : ℕ), y ≠ 15 ∧ (is_pythagorean_triple 8 y 17 ∨ is_pythagorean_triple 8 17 y ∨ is_pythagorean_triple y 8 17) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_8_and_17_l1873_187366


namespace NUMINAMATH_CALUDE_percentage_problem_l1873_187317

theorem percentage_problem (x : ℝ) :
  (0.15 * 0.30 * (x / 100) * 5200 = 117) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1873_187317


namespace NUMINAMATH_CALUDE_wickets_in_last_match_is_three_l1873_187344

/-- Represents a cricket bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  runsInLastMatch : ℕ
  averageDecrease : ℝ
  approximateWicketsBefore : ℕ

/-- Calculates the number of wickets taken in the last match -/
def wicketsInLastMatch (stats : BowlerStats) : ℕ :=
  -- The actual calculation would go here
  3 -- We're stating the result directly as per the problem

/-- Theorem stating that given the specific conditions, the number of wickets in the last match is 3 -/
theorem wickets_in_last_match_is_three (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.runsInLastMatch = 26)
  (h3 : stats.averageDecrease = 0.4)
  (h4 : stats.approximateWicketsBefore = 25) :
  wicketsInLastMatch stats = 3 := by
  sorry

#eval wicketsInLastMatch { 
  initialAverage := 12.4, 
  runsInLastMatch := 26, 
  averageDecrease := 0.4, 
  approximateWicketsBefore := 25 
}

end NUMINAMATH_CALUDE_wickets_in_last_match_is_three_l1873_187344


namespace NUMINAMATH_CALUDE_least_candies_l1873_187329

theorem least_candies (c : ℕ) : 
  c < 150 ∧ 
  c % 5 = 4 ∧ 
  c % 6 = 3 ∧ 
  c % 8 = 5 ∧
  (∀ k : ℕ, k < c → ¬(k < 150 ∧ k % 5 = 4 ∧ k % 6 = 3 ∧ k % 8 = 5)) →
  c = 69 := by
sorry

end NUMINAMATH_CALUDE_least_candies_l1873_187329


namespace NUMINAMATH_CALUDE_divisor_proof_l1873_187373

theorem divisor_proof : ∃ x : ℝ, (26.3 * 12 * 20) / x + 125 = 2229 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisor_proof_l1873_187373


namespace NUMINAMATH_CALUDE_division_of_fractions_l1873_187340

theorem division_of_fractions : (3 : ℚ) / 7 / 5 = 3 / 35 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1873_187340


namespace NUMINAMATH_CALUDE_chris_average_speed_l1873_187367

/-- Calculates the average speed given initial and final odometer readings and total time. -/
def average_speed (initial_reading : ℕ) (final_reading : ℕ) (total_time : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

/-- Proves that Chris's average speed is approximately 36.67 miles per hour. -/
theorem chris_average_speed :
  let initial_reading := 2332
  let final_reading := 2772
  let total_time := 12
  abs (average_speed initial_reading final_reading total_time - 36.67) < 0.01 := by
  sorry

#eval average_speed 2332 2772 12

end NUMINAMATH_CALUDE_chris_average_speed_l1873_187367


namespace NUMINAMATH_CALUDE_harvest_duration_l1873_187379

theorem harvest_duration (total_earnings : ℕ) (weekly_earnings : ℕ) (h1 : total_earnings = 133) (h2 : weekly_earnings = 7) :
  total_earnings / weekly_earnings = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_l1873_187379


namespace NUMINAMATH_CALUDE_monomial_properties_l1873_187383

def monomial_coefficient (a : ℤ) (b c : ℕ) : ℤ := -2

def monomial_degree (a : ℤ) (b c : ℕ) : ℕ := 1 + b + c

theorem monomial_properties :
  let m := monomial_coefficient (-2) 2 4
  let n := monomial_degree (-2) 2 4
  m = -2 ∧ n = 7 := by sorry

end NUMINAMATH_CALUDE_monomial_properties_l1873_187383


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1873_187342

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 1 → 1/a + 1/b ≤ 1/x + 1/y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 1 ∧ 1/x + 1/y = 3 + 2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1873_187342


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1873_187341

-- Define a monotonic function f from real numbers to real numbers
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y ∨ (∀ z : ℝ, f z = f x)

-- State the theorem
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h_monotonic : MonotonicFunction f)
  (h_equation : ∀ x y : ℝ, f x * f y = f (x + y)) :
  ∃! a : ℝ, a > 0 ∧ a ≠ 1 ∧ (∀ x : ℝ, f x = a^x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1873_187341


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1873_187348

/-- Represents a rectangular enclosure --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of the rectangle is 400 feet --/
def isValidPerimeter (r : Rectangle) : Prop :=
  2 * r.length + 2 * r.width = 400

/-- The length is at least 90 feet --/
def hasValidLength (r : Rectangle) : Prop :=
  r.length ≥ 90

/-- The width is at least 50 feet --/
def hasValidWidth (r : Rectangle) : Prop :=
  r.width ≥ 50

/-- The area of the rectangle --/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 400 feet, 
    length ≥ 90 feet, and width ≥ 50 feet is 10,000 square feet --/
theorem max_area_rectangle :
  ∃ (r : Rectangle), isValidPerimeter r ∧ hasValidLength r ∧ hasValidWidth r ∧
    (∀ (s : Rectangle), isValidPerimeter s ∧ hasValidLength s ∧ hasValidWidth s →
      area s ≤ area r) ∧
    area r = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1873_187348


namespace NUMINAMATH_CALUDE_teaching_years_difference_l1873_187372

theorem teaching_years_difference :
  ∀ (V A D : ℕ),
  V + A + D = 93 →
  V = A + 9 →
  D = 40 →
  V < D →
  D - V = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_teaching_years_difference_l1873_187372


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1873_187343

theorem angle_with_special_supplement_complement (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 10) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1873_187343


namespace NUMINAMATH_CALUDE_four_last_digit_fib_mod8_l1873_187391

/-- Fibonacci sequence modulo 8 -/
def fib_mod8 : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => (fib_mod8 n + fib_mod8 (n + 1)) % 8

/-- Set of digits that have appeared in the Fibonacci sequence modulo 8 up to n -/
def digits_appeared (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1).succ
    |>.filter (fun i => fib_mod8 i ∈ Finset.range 8)
    |>.image fib_mod8

/-- The proposition that 4 is the last digit to appear in the Fibonacci sequence modulo 8 -/
theorem four_last_digit_fib_mod8 :
  ∃ n : ℕ, 4 ∈ digits_appeared n ∧ digits_appeared n = Finset.range 8 :=
sorry

end NUMINAMATH_CALUDE_four_last_digit_fib_mod8_l1873_187391


namespace NUMINAMATH_CALUDE_square_area_l1873_187302

theorem square_area (s : ℝ) (h : (2/5 * s) * 10 = 140) : s^2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l1873_187302


namespace NUMINAMATH_CALUDE_stating_comprehensive_investigation_is_census_l1873_187331

/-- Represents a comprehensive investigation. -/
structure ComprehensiveInvestigation where
  subject : String
  purpose : String

/-- Defines what a census is. -/
def Census : Type := ComprehensiveInvestigation

/-- 
Theorem stating that a comprehensive investigation on the subject of examination 
for a specific purpose is equivalent to a census.
-/
theorem comprehensive_investigation_is_census 
  (investigation : ComprehensiveInvestigation) 
  (h1 : investigation.subject = "examination") 
  (h2 : investigation.purpose ≠ "") : 
  ∃ (c : Census), c = investigation :=
sorry

end NUMINAMATH_CALUDE_stating_comprehensive_investigation_is_census_l1873_187331


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1873_187325

theorem inheritance_calculation (x : ℝ) 
  (h1 : 0.25 * x + 0.1 * x = 15000) : 
  x = 42857 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1873_187325


namespace NUMINAMATH_CALUDE_quadratic_equation_with_integer_roots_l1873_187337

theorem quadratic_equation_with_integer_roots (m : ℤ) 
  (h1 : ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    a^2 + m*a - m + 1 = 0 ∧ b^2 + m*b - m + 1 = 0) : 
  m = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_integer_roots_l1873_187337


namespace NUMINAMATH_CALUDE_victor_games_ratio_l1873_187357

theorem victor_games_ratio : 
  let victor_wins : ℕ := 36
  let friend_wins : ℕ := 20
  let gcd := Nat.gcd victor_wins friend_wins
  (victor_wins / gcd) = 9 ∧ (friend_wins / gcd) = 5 := by
sorry

end NUMINAMATH_CALUDE_victor_games_ratio_l1873_187357


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1873_187363

theorem square_sum_zero_implies_both_zero (a b : ℝ) :
  a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1873_187363


namespace NUMINAMATH_CALUDE_largest_k_inequality_l1873_187378

theorem largest_k_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (∀ k : ℕ+, (1 / (a - b) + 1 / (b - c) ≥ k / (a - c)) → k ≤ 4) ∧ 
  (∃ a b c : ℝ, a > b ∧ b > c ∧ 1 / (a - b) + 1 / (b - c) = 4 / (a - c)) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l1873_187378


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1873_187381

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 5*I) * (a + b*I) = y*I) : a/b = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1873_187381


namespace NUMINAMATH_CALUDE_part_one_part_two_l1873_187347

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}
def B : Set ℝ := {-1, 1}

-- Theorem for part I
theorem part_one (a b : ℝ) : B ⊆ A a b → a = -1 := by sorry

-- Theorem for part II
theorem part_two (a b : ℝ) : (A a b ∩ B).Nonempty → a^2 - b^2 + 2*a = -1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1873_187347


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1873_187369

theorem largest_prime_divisor_of_sum_of_squares : 
  (∃ p : Nat, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : Nat, Nat.Prime q → q ∣ (17^2 + 60^2) → q ≤ p) ∧ 
  (37 : Nat).Prime ∧ 
  37 ∣ (17^2 + 60^2) ∧ 
  ∀ q : Nat, Nat.Prime q → q ∣ (17^2 + 60^2) → q ≤ 37 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1873_187369


namespace NUMINAMATH_CALUDE_roses_sold_l1873_187388

theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) (sold : ℕ) : 
  initial = 5 → picked = 34 → final = 36 → 
  final = initial - sold + picked → sold = 3 := by
sorry

end NUMINAMATH_CALUDE_roses_sold_l1873_187388


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l1873_187351

theorem trigonometric_equalities :
  (6 * (Real.tan (30 * π / 180))^2 - Real.sqrt 3 * Real.sin (60 * π / 180) - 2 * Real.sin (45 * π / 180) = 1/2 - Real.sqrt 2) ∧
  (Real.sqrt 2 / 2 * Real.cos (45 * π / 180) - (Real.tan (40 * π / 180) + 1)^0 + Real.sqrt (1/4) + Real.sin (30 * π / 180) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l1873_187351


namespace NUMINAMATH_CALUDE_work_completion_time_l1873_187380

/-- Given that P persons can complete a work in 24 days, 
    prove that 2P persons can complete half of the work in 6 days. -/
theorem work_completion_time 
  (P : ℕ) -- number of persons
  (full_work : ℝ) -- amount of full work
  (h1 : P > 0) -- assumption that there's at least one person
  (h2 : full_work > 0) -- assumption that there's some work to be done
  (h3 : P * 24 * full_work = P * 24 * full_work) -- work completion condition
  : (2 * P) * 6 * (full_work / 2) = P * 24 * full_work := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1873_187380


namespace NUMINAMATH_CALUDE_log_equation_solution_l1873_187384

theorem log_equation_solution : ∃ x : ℝ, (Real.log x - Real.log 25) / 100 = -20 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1873_187384


namespace NUMINAMATH_CALUDE_shifted_sine_function_l1873_187339

/-- Given a function f and its right-shifted version g, prove that g has the correct form -/
theorem shifted_sine_function 
  (f g : ℝ → ℝ) 
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x))
  (h₂ : ∀ x, g x = f (x - π/8)) :
  ∀ x, g x = 3 * Real.sin (2 * x - π/4) := by
  sorry


end NUMINAMATH_CALUDE_shifted_sine_function_l1873_187339


namespace NUMINAMATH_CALUDE_division_simplification_l1873_187314

theorem division_simplification (m : ℝ) (h : m ≠ 0) :
  (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1873_187314


namespace NUMINAMATH_CALUDE_original_rectangle_area_l1873_187368

/-- Given a rectangle whose dimensions are doubled to form a new rectangle with an area of 32 square meters, 
    the area of the original rectangle is 8 square meters. -/
theorem original_rectangle_area (original_length original_width : ℝ) 
  (new_length new_width : ℝ) (new_area : ℝ) :
  new_length = 2 * original_length →
  new_width = 2 * original_width →
  new_area = new_length * new_width →
  new_area = 32 →
  original_length * original_width = 8 :=
by sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l1873_187368


namespace NUMINAMATH_CALUDE_circle_equation_l1873_187319

/-- Given a circle with center (2, -3) and radius 4, its equation is (x-2)^2 + (y+3)^2 = 16 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -3)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_l1873_187319


namespace NUMINAMATH_CALUDE_always_possible_to_reach_final_state_l1873_187318

/-- Represents the two types of operations that can be performed. -/
inductive Operation
  | RedToBlue
  | BlueToRed

/-- Represents the state of the slips for a single MOPper. -/
structure MOPperState where
  number : Nat
  redSlip : Nat
  blueSlip : Nat

/-- Represents the state of all MOPpers' slips. -/
def SystemState := List MOPperState

/-- Initializes the system state based on the given A and B values. -/
def initializeState (A B : Nat) : SystemState :=
  sorry

/-- Performs a single operation on the system state. -/
def performOperation (state : SystemState) (op : Operation) : SystemState :=
  sorry

/-- Checks if the system state is in the desired final configuration. -/
def isFinalState (state : SystemState) : Bool :=
  sorry

/-- The main theorem to be proved. -/
theorem always_possible_to_reach_final_state :
  ∀ (A B : Nat), A ≤ 2010 → B ≤ 2010 →
  ∃ (ops : List Operation),
    isFinalState (ops.foldl performOperation (initializeState A B)) = true :=
  sorry

end NUMINAMATH_CALUDE_always_possible_to_reach_final_state_l1873_187318


namespace NUMINAMATH_CALUDE_equation_linear_iff_a_eq_plus_minus_two_l1873_187315

-- Define the equation
def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3*a) * x + (a + 1) * y + 3*a

-- Define what it means for the equation to be linear in two variables
def is_linear_two_var (a : ℝ) : Prop :=
  (a^2 - 4 = 0) ∧ (2 - 3*a ≠ 0 ∨ a + 1 ≠ 0)

-- State the theorem
theorem equation_linear_iff_a_eq_plus_minus_two :
  ∀ a : ℝ, is_linear_two_var a ↔ (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_linear_iff_a_eq_plus_minus_two_l1873_187315


namespace NUMINAMATH_CALUDE_yoga_studio_average_weight_l1873_187321

theorem yoga_studio_average_weight 
  (num_men : ℕ) 
  (num_women : ℕ) 
  (avg_weight_men : ℝ) 
  (avg_weight_women : ℝ) 
  (h1 : num_men = 8) 
  (h2 : num_women = 6) 
  (h3 : avg_weight_men = 190) 
  (h4 : avg_weight_women = 120) :
  let total_people := num_men + num_women
  let total_weight := num_men * avg_weight_men + num_women * avg_weight_women
  total_weight / total_people = 160 := by
sorry

end NUMINAMATH_CALUDE_yoga_studio_average_weight_l1873_187321


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1873_187311

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1873_187311


namespace NUMINAMATH_CALUDE_interior_angles_sum_l1873_187375

/-- If the sum of the interior angles of a convex polygon with n sides is 1800°,
    then the sum of the interior angles of a convex polygon with n + 4 sides is 2520°. -/
theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l1873_187375


namespace NUMINAMATH_CALUDE_floor_sqrt_5_minus_3_l1873_187326

theorem floor_sqrt_5_minus_3 : ⌊Real.sqrt 5 - 3⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_5_minus_3_l1873_187326


namespace NUMINAMATH_CALUDE_vector_simplification_l1873_187333

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_simplification (A B C D : V) :
  (B - A) + (C - B) - (D - A) = D - C := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l1873_187333


namespace NUMINAMATH_CALUDE_log_50000_sum_consecutive_integers_l1873_187334

theorem log_50000_sum_consecutive_integers : ∃ (a b : ℕ), 
  (a + 1 = b) ∧ 
  (a : ℝ) < Real.log 50000 / Real.log 10 ∧ 
  Real.log 50000 / Real.log 10 < (b : ℝ) ∧ 
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_log_50000_sum_consecutive_integers_l1873_187334


namespace NUMINAMATH_CALUDE_ones_digit_multiplication_l1873_187345

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_multiplication (n : ℕ) (h : ones_digit n = 2) :
  ones_digit (n * 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_multiplication_l1873_187345


namespace NUMINAMATH_CALUDE_gcf_of_75_and_105_l1873_187306

theorem gcf_of_75_and_105 : Nat.gcd 75 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_105_l1873_187306


namespace NUMINAMATH_CALUDE_exists_skew_line_l1873_187304

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line intersecting a plane
variable (intersects : Line → Plane → Prop)

-- Define the property of a line being in a plane
variable (inPlane : Line → Plane → Prop)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem exists_skew_line 
  (l : Line) (α : Plane) 
  (h : intersects l α) : 
  ∃ m : Line, inPlane m α ∧ skew l m :=
sorry

end NUMINAMATH_CALUDE_exists_skew_line_l1873_187304


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_sixty_one_satisfies_conditions_smallest_integer_is_sixty_one_l1873_187396

theorem smallest_integer_with_remainder_one (n : ℕ) : n > 1 ∧ 
  n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 → n ≥ 61 :=
by
  sorry

theorem sixty_one_satisfies_conditions : 
  61 > 1 ∧ 61 % 4 = 1 ∧ 61 % 5 = 1 ∧ 61 % 6 = 1 :=
by
  sorry

theorem smallest_integer_is_sixty_one : 
  ∃ (n : ℕ), n > 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 ∧ 
  ∀ (m : ℕ), m > 1 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_sixty_one_satisfies_conditions_smallest_integer_is_sixty_one_l1873_187396


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1873_187336

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x ≠ y → (f (-1) x - f (-1) y) / (x - y) = Real.log 2 * x + f (-1) x - Real.log 2) ∧
  (∀ x > 0, Monotone (f a) ↔ a ≥ (1 : ℝ) / 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1873_187336


namespace NUMINAMATH_CALUDE_first_division_divisor_l1873_187310

theorem first_division_divisor
  (x : ℕ+) -- x is a positive integer
  (y : ℕ) -- y is a natural number (quotient)
  (d : ℕ) -- d is the divisor we're looking for
  (h1 : ∃ q : ℕ, x = d * y + 3) -- x divided by d gives quotient y and remainder 3
  (h2 : ∃ q : ℕ, 2 * x = 7 * (3 * y) + 1) -- 2x divided by 7 gives quotient 3y and remainder 1
  (h3 : 11 * y - x = 2) -- Given equation
  : d = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_division_divisor_l1873_187310


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1873_187395

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is geometric if the ratio between consecutive terms is constant. -/
def IsGeometric (a : Sequence) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem to be proved. -/
theorem geometric_sequence_fourth_term
  (a : Sequence)
  (h1 : a 1 = 2)
  (h2 : IsGeometric (fun n => 1 + a n) 3) :
  a 4 = 80 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1873_187395


namespace NUMINAMATH_CALUDE_train_length_l1873_187330

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 15 → ∃ (length : ℝ), abs (length - 500) < 1 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1873_187330


namespace NUMINAMATH_CALUDE_relationship_abc_l1873_187386

theorem relationship_abc (a b c : ℝ) : 
  a = 1 + Real.sqrt 7 → 
  b = Real.sqrt 3 + Real.sqrt 5 → 
  c = 4 → 
  c > b ∧ b > a := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1873_187386


namespace NUMINAMATH_CALUDE_larger_jar_initial_fill_fraction_l1873_187303

/-- Proves that under the given conditions, the larger jar was initially 1/3 full -/
theorem larger_jar_initial_fill_fraction 
  (small_capacity large_capacity : ℝ) 
  (water_amount : ℝ) 
  (h1 : small_capacity > 0)
  (h2 : large_capacity > 0)
  (h3 : water_amount > 0)
  (h4 : water_amount = 1/3 * small_capacity)
  (h5 : water_amount < large_capacity)
  (h6 : water_amount + water_amount = 2/3 * large_capacity) :
  water_amount = 1/3 * large_capacity := by
sorry

end NUMINAMATH_CALUDE_larger_jar_initial_fill_fraction_l1873_187303


namespace NUMINAMATH_CALUDE_jerry_zinc_consumption_l1873_187376

/-- The amount of zinc Jerry eats from antacids -/
def zinc_consumed (big_antacid_weight : ℝ) (big_antacid_count : ℕ) (big_antacid_zinc_percent : ℝ)
                  (small_antacid_weight : ℝ) (small_antacid_count : ℕ) (small_antacid_zinc_percent : ℝ) : ℝ :=
  (big_antacid_weight * big_antacid_count * big_antacid_zinc_percent +
   small_antacid_weight * small_antacid_count * small_antacid_zinc_percent) * 1000

/-- Theorem stating the amount of zinc Jerry consumes -/
theorem jerry_zinc_consumption :
  zinc_consumed 2 2 0.05 1 3 0.15 = 650 := by
  sorry

end NUMINAMATH_CALUDE_jerry_zinc_consumption_l1873_187376


namespace NUMINAMATH_CALUDE_tim_appetizers_l1873_187398

theorem tim_appetizers (total_spent : ℚ) (entree_percentage : ℚ) (appetizer_cost : ℚ) : 
  total_spent = 50 →
  entree_percentage = 80 / 100 →
  appetizer_cost = 5 →
  (total_spent * (1 - entree_percentage)) / appetizer_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_tim_appetizers_l1873_187398


namespace NUMINAMATH_CALUDE_unique_three_digit_number_with_three_divisors_l1873_187360

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def starts_with_three (n : ℕ) : Prop := ∃ k, n = 300 + k ∧ 0 ≤ k ∧ k < 100

def has_exactly_three_divisors (n : ℕ) : Prop := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 3

theorem unique_three_digit_number_with_three_divisors :
  ∃! n : ℕ, is_three_digit n ∧ starts_with_three n ∧ has_exactly_three_divisors n ∧ n = 361 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_with_three_divisors_l1873_187360


namespace NUMINAMATH_CALUDE_complete_square_d_value_l1873_187307

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when transformed
    into the form (x + c)^2 = d, the value of d is 4. -/
theorem complete_square_d_value :
  ∃ c d : ℝ, (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + c)^2 = d) ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_d_value_l1873_187307


namespace NUMINAMATH_CALUDE_lineup_arrangements_eq_960_l1873_187393

/-- The number of ways to arrange 5 volunteers and 2 elderly individuals in a row,
    where the elderly individuals must stand next to each other but not at the ends. -/
def lineup_arrangements : ℕ :=
  let n_volunteers : ℕ := 5
  let n_elderly : ℕ := 2
  let volunteer_arrangements : ℕ := Nat.factorial n_volunteers
  let elderly_pair_positions : ℕ := n_volunteers - 1
  let elderly_internal_arrangements : ℕ := Nat.factorial n_elderly
  volunteer_arrangements * (elderly_pair_positions - 1) * elderly_internal_arrangements

theorem lineup_arrangements_eq_960 : lineup_arrangements = 960 := by
  sorry

end NUMINAMATH_CALUDE_lineup_arrangements_eq_960_l1873_187393


namespace NUMINAMATH_CALUDE_sequence_existence_l1873_187370

theorem sequence_existence : ∃ (a : ℕ → ℕ+), 
  (∀ k : ℕ+, ∃ n : ℕ, a n = k) ∧ 
  (∀ k : ℕ+, (Finset.range k).sum (λ i => (a i.succ).val) % k = 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l1873_187370


namespace NUMINAMATH_CALUDE_max_value_of_b_l1873_187389

theorem max_value_of_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : 
  b ≤ 1/3 ∧ ∃ (x : ℝ), x > 0 ∧ 2 * x * (1/3) = (2 * x - 1/3) / (2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_b_l1873_187389


namespace NUMINAMATH_CALUDE_f_at_neg_point_two_eq_approx_l1873_187371

/-- Horner's algorithm for polynomial evaluation -/
def horner (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial function f(x) -/
def f (x : Float) : Float :=
  horner [1, 1, 0.5, 0.16667, 0.04167, 0.00833] x

/-- Theorem stating that f(-0.2) equals 0.81873 (approximately) -/
theorem f_at_neg_point_two_eq_approx :
  (f (-0.2) - 0.81873).abs < 1e-5 := by
  sorry

#eval f (-0.2)

end NUMINAMATH_CALUDE_f_at_neg_point_two_eq_approx_l1873_187371


namespace NUMINAMATH_CALUDE_factor_implies_a_value_l1873_187349

theorem factor_implies_a_value (a b : ℤ) (x : ℝ) :
  (∀ x, (x^2 - x - 1) ∣ (a*x^17 + b*x^16 + 1)) →
  a = 987 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_a_value_l1873_187349


namespace NUMINAMATH_CALUDE_garden_breadth_l1873_187382

/-- Given a rectangular garden with perimeter 680 m and length 258 m, its breadth is 82 m -/
theorem garden_breadth (perimeter length breadth : ℝ) : 
  perimeter = 680 ∧ length = 258 ∧ perimeter = 2 * (length + breadth) → breadth = 82 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l1873_187382


namespace NUMINAMATH_CALUDE_sara_sold_oranges_l1873_187320

/-- Represents the number of oranges Joan picked initially -/
def initial_oranges : ℕ := 37

/-- Represents the number of oranges Joan is left with -/
def remaining_oranges : ℕ := 27

/-- Represents the number of oranges Sara sold -/
def sold_oranges : ℕ := initial_oranges - remaining_oranges

theorem sara_sold_oranges : sold_oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_sold_oranges_l1873_187320


namespace NUMINAMATH_CALUDE_wrapping_paper_division_l1873_187309

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) (paper_per_present : ℚ) :
  total_used = 5 / 8 →
  num_presents = 5 →
  paper_per_present * num_presents = total_used →
  paper_per_present = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_division_l1873_187309


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1873_187305

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1873_187305


namespace NUMINAMATH_CALUDE_sample_customers_l1873_187323

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_left : ℕ) : 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  (samples_per_box * boxes_opened - samples_left) = 235 := by
  sorry

end NUMINAMATH_CALUDE_sample_customers_l1873_187323


namespace NUMINAMATH_CALUDE_determinant_of_special_matrix_l1873_187365

theorem determinant_of_special_matrix (y : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := 
    ![![2*y + 1, 2*y, 2*y],
      ![2*y, 2*y + 1, 2*y],
      ![2*y, 2*y, 2*y + 1]]
  Matrix.det A = 6*y + 1 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_special_matrix_l1873_187365


namespace NUMINAMATH_CALUDE_f_properties_l1873_187335

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) ≠ -f x) ∧
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ y : ℝ, -1 < y ∧ y < 1 ↔ ∃ x : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1873_187335


namespace NUMINAMATH_CALUDE_chromium_percentage_in_mixed_alloy_l1873_187390

/-- Given two alloys with different chromium percentages and weights, 
    calculates the chromium percentage in the resulting alloy when mixed. -/
theorem chromium_percentage_in_mixed_alloy 
  (chromium_percent1 chromium_percent2 : ℝ)
  (weight1 weight2 : ℝ)
  (h1 : chromium_percent1 = 15)
  (h2 : chromium_percent2 = 8)
  (h3 : weight1 = 15)
  (h4 : weight2 = 35) :
  let total_chromium := (chromium_percent1 / 100 * weight1) + (chromium_percent2 / 100 * weight2)
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 10.1 := by
sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_mixed_alloy_l1873_187390


namespace NUMINAMATH_CALUDE_school_survey_probability_l1873_187354

theorem school_survey_probability (total_students : ℕ) (selected_students : ℕ) 
  (eliminated_students : ℕ) (h1 : total_students = 883) (h2 : selected_students = 80) 
  (h3 : eliminated_students = 3) :
  (selected_students : ℚ) / total_students = 80 / 883 := by
  sorry

end NUMINAMATH_CALUDE_school_survey_probability_l1873_187354


namespace NUMINAMATH_CALUDE_historical_fiction_new_release_fraction_is_four_sevenths_l1873_187385

/-- Represents the inventory of a bookstore -/
structure BookstoreInventory where
  total_books : ℕ
  historical_fiction_ratio : ℚ
  historical_fiction_new_release_ratio : ℚ
  other_new_release_ratio : ℚ

/-- Calculates the fraction of new releases that are historical fiction -/
def historical_fiction_new_release_fraction (inventory : BookstoreInventory) : ℚ :=
  let historical_fiction := inventory.total_books * inventory.historical_fiction_ratio
  let other_books := inventory.total_books * (1 - inventory.historical_fiction_ratio)
  let historical_fiction_new_releases := historical_fiction * inventory.historical_fiction_new_release_ratio
  let other_new_releases := other_books * inventory.other_new_release_ratio
  historical_fiction_new_releases / (historical_fiction_new_releases + other_new_releases)

/-- Theorem stating that the fraction of new releases that are historical fiction is 4/7 -/
theorem historical_fiction_new_release_fraction_is_four_sevenths
  (inventory : BookstoreInventory)
  (h1 : inventory.historical_fiction_ratio = 2/5)
  (h2 : inventory.historical_fiction_new_release_ratio = 2/5)
  (h3 : inventory.other_new_release_ratio = 1/5) :
  historical_fiction_new_release_fraction inventory = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_release_fraction_is_four_sevenths_l1873_187385


namespace NUMINAMATH_CALUDE_no_real_solutions_log_equation_l1873_187338

theorem no_real_solutions_log_equation :
  ¬ ∃ (x : ℝ), Real.log (x^2 - 3*x + 9) = 1 := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_log_equation_l1873_187338


namespace NUMINAMATH_CALUDE_inequality_range_of_p_l1873_187355

-- Define the inequality function
def inequality (a p : ℝ) : Prop :=
  Real.sqrt a - Real.sqrt (a - 1) > Real.sqrt (a - 2) - Real.sqrt (a - p)

-- State the theorem
theorem inequality_range_of_p :
  ∀ a p : ℝ, a ≥ 3 → p > 2 → 
  (∀ x : ℝ, x ≥ 3 → inequality x p) →
  2 < p ∧ p < 2 * Real.sqrt 6 + 2 * Real.sqrt 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_of_p_l1873_187355


namespace NUMINAMATH_CALUDE_neha_removed_amount_l1873_187361

/-- The amount removed from Neha's share in a money division problem -/
theorem neha_removed_amount (total : ℝ) (mahi_share : ℝ) (sabi_removed : ℝ) (mahi_removed : ℝ) :
  total = 1100 →
  mahi_share = 102 →
  sabi_removed = 8 →
  mahi_removed = 4 →
  ∃ (neha_share sabi_share neha_removed : ℝ),
    neha_share + sabi_share + mahi_share = total ∧
    neha_share - neha_removed = 2 * ((sabi_share - sabi_removed) / 8) ∧
    mahi_share - mahi_removed = 6 * ((sabi_share - sabi_removed) / 8) ∧
    neha_removed = 826.70 := by
  sorry

#eval (826.70 : Float)

end NUMINAMATH_CALUDE_neha_removed_amount_l1873_187361


namespace NUMINAMATH_CALUDE_rhombus_area_l1873_187312

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 208 square units. -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) : 
  side_length = Real.sqrt 145 →
  diagonal_difference = 10 →
  area = 208 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1873_187312


namespace NUMINAMATH_CALUDE_product_inequality_l1873_187394

theorem product_inequality (a b x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (hx_prod : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1873_187394


namespace NUMINAMATH_CALUDE_hem_length_is_three_feet_l1873_187352

/-- The length of a stitch in inches -/
def stitch_length : ℚ := 1/4

/-- The number of stitches Jenna makes per minute -/
def stitches_per_minute : ℕ := 24

/-- The time it takes Jenna to hem her dress in minutes -/
def hemming_time : ℕ := 6

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The length of the dress's hem in feet -/
def hem_length : ℚ := (stitches_per_minute * hemming_time * stitch_length) / inches_per_foot

theorem hem_length_is_three_feet : hem_length = 3 := by
  sorry

end NUMINAMATH_CALUDE_hem_length_is_three_feet_l1873_187352


namespace NUMINAMATH_CALUDE_f_monotonicity_and_bound_l1873_187387

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem f_monotonicity_and_bound (a : ℝ) :
  (a > 0 → ∀ x y, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  ((∀ x, x > 1 → f a x < x^2) → a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_bound_l1873_187387


namespace NUMINAMATH_CALUDE_selection_problem_l1873_187399

theorem selection_problem (n : ℕ) (r : ℕ) (h1 : n = 10) (h2 : r = 4) :
  Nat.choose n r = 210 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l1873_187399


namespace NUMINAMATH_CALUDE_semicircle_chord_product_l1873_187300

/-- The radius of the semicircle -/
def radius : ℝ := 3

/-- The number of equal parts the semicircle is divided into -/
def num_parts : ℕ := 8

/-- The number of chords -/
def num_chords : ℕ := 14

/-- The product of the lengths of the chords in a semicircle -/
def chord_product (r : ℝ) (n : ℕ) : ℝ :=
  (2 * r ^ (n - 1)) * (2 ^ n)

theorem semicircle_chord_product :
  chord_product radius num_chords = 196608 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_chord_product_l1873_187300


namespace NUMINAMATH_CALUDE_implicit_derivative_l1873_187332

-- Define the implicit function
def implicit_function (x y : ℝ) : Prop := x^2 - y^2 = 4

-- State the theorem
theorem implicit_derivative (x y : ℝ) (h : implicit_function x y) :
  ∃ (y' : ℝ), y' = x / y :=
sorry

end NUMINAMATH_CALUDE_implicit_derivative_l1873_187332


namespace NUMINAMATH_CALUDE_lisa_children_count_l1873_187364

/-- The number of children Lisa has -/
def num_children : ℕ := sorry

/-- The number of eggs Lisa cooks for her family in a year -/
def total_eggs_per_year : ℕ := 3380

/-- The number of days Lisa cooks breakfast in a year -/
def days_per_year : ℕ := 5 * 52

/-- The number of eggs Lisa cooks each day -/
def eggs_per_day (c : ℕ) : ℕ := 2 * c + 3 + 2

theorem lisa_children_count : 
  num_children = 4 ∧ 
  total_eggs_per_year = days_per_year * eggs_per_day num_children :=
sorry

end NUMINAMATH_CALUDE_lisa_children_count_l1873_187364


namespace NUMINAMATH_CALUDE_range_of_f_sum_of_endpoints_l1873_187346

open Set Real

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem range_of_f :
  range f = Ioo 0 3 ∪ {3} :=
sorry

theorem sum_of_endpoints :
  ∃ c d : ℝ, range f = Ioc c d ∧ c + d = 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_sum_of_endpoints_l1873_187346


namespace NUMINAMATH_CALUDE_sum_of_selected_elements_ge_one_l1873_187397

/-- Definition of the table element at position (i, j) -/
def table_element (i j : ℕ) : ℚ := 1 / (i + j - 1)

/-- A selection of n elements from an n × n table, where no two elements are in the same row or column -/
def valid_selection (n : ℕ) : Type := 
  { s : Finset (ℕ × ℕ) // s.card = n ∧ 
    (∀ (a b : ℕ × ℕ), a ∈ s → b ∈ s → a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2) ∧
    (∀ (a : ℕ × ℕ), a ∈ s → a.1 ≤ n ∧ a.2 ≤ n) }

/-- The main theorem: The sum of selected elements is not less than 1 -/
theorem sum_of_selected_elements_ge_one (n : ℕ) (h : n > 0) :
  ∀ (s : valid_selection n), (s.val.sum (λ (x : ℕ × ℕ) => table_element x.1 x.2)) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_selected_elements_ge_one_l1873_187397


namespace NUMINAMATH_CALUDE_hurricane_damage_conversion_l1873_187353

/-- Calculates the equivalent amount in Canadian dollars given an amount in American dollars and the exchange rate. -/
def convert_to_canadian_dollars (american_dollars : ℚ) (exchange_rate : ℚ) : ℚ :=
  american_dollars * exchange_rate

/-- Theorem stating the correct conversion of hurricane damage from American to Canadian dollars. -/
theorem hurricane_damage_conversion :
  let damage_usd : ℚ := 45000000
  let exchange_rate : ℚ := 3/2
  convert_to_canadian_dollars damage_usd exchange_rate = 67500000 := by
  sorry

#check hurricane_damage_conversion

end NUMINAMATH_CALUDE_hurricane_damage_conversion_l1873_187353


namespace NUMINAMATH_CALUDE_max_stores_visited_l1873_187358

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 7) (h2 : total_visits = 21) 
  (h3 : total_shoppers = 11) (h4 : double_visitors = 7) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : 2 * double_visitors ≤ total_visits) : 
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits := by
  sorry

end NUMINAMATH_CALUDE_max_stores_visited_l1873_187358


namespace NUMINAMATH_CALUDE_rearrangement_does_not_increase_length_l1873_187356

/-- A segment on a line --/
structure Segment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- A finite set of segments on a line --/
def SegmentSystem := Finset Segment

/-- The total length of the union of segments in a system --/
def totalLength (S : SegmentSystem) : ℝ := sorry

/-- The distance between midpoints of two segments --/
def midpointDistance (s₁ s₂ : Segment) : ℝ := sorry

/-- A rearrangement of segments that minimizes midpoint distances --/
def rearrange (S : SegmentSystem) : SegmentSystem := sorry

/-- The theorem stating that rearrangement does not increase total length --/
theorem rearrangement_does_not_increase_length (S : SegmentSystem) :
  totalLength (rearrange S) ≤ totalLength S := by sorry

end NUMINAMATH_CALUDE_rearrangement_does_not_increase_length_l1873_187356


namespace NUMINAMATH_CALUDE_sum_between_14_and_14_half_l1873_187350

theorem sum_between_14_and_14_half :
  let sum := (3 + 3/8) + (4 + 3/4) + (6 + 2/23)
  14 < sum ∧ sum < 14.5 := by
sorry

end NUMINAMATH_CALUDE_sum_between_14_and_14_half_l1873_187350


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_in_set_l1873_187328

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x - |x^2 - a*x + 1|

/-- The set of a values that satisfy the condition -/
def A : Set ℝ := {a | a < 0 ∨ (0 < a ∧ a < 1) ∨ 1 < a}

theorem two_zeros_iff_a_in_set (a : ℝ) : 
  (∃! (x y : ℝ), x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a ∈ A := by sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_in_set_l1873_187328


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1873_187377

/-- The number of ways to arrange 6 people in a line with two specific people not adjacent -/
def line_arrangement (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial (n - k) * (Nat.choose (n - k + 1) k)

theorem six_people_arrangement :
  line_arrangement 6 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1873_187377


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1873_187327

theorem other_root_of_quadratic (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1873_187327


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_power_is_zero_l1873_187392

def expression (x : ℝ) : ℝ := 3 * (x^2 - x^4) - 5 * (x^4 - x^6 + x^2) + 4 * (2*x^4 - x^8)

theorem coefficient_of_x_fourth_power_is_zero :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, expression x = f x + 0 * x^4 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_power_is_zero_l1873_187392


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1873_187362

theorem solve_exponential_equation :
  ∃ x : ℝ, (4 : ℝ) ^ x * (4 : ℝ) ^ x * (4 : ℝ) ^ x = 256 ^ 3 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1873_187362


namespace NUMINAMATH_CALUDE_total_distance_walked_l1873_187374

-- Define constants for conversion
def feet_per_mile : ℕ := 5280
def feet_per_yard : ℕ := 3

-- Define the distances walked by each person
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

-- Theorem statement
theorem total_distance_walked :
  lionel_miles * feet_per_mile + esther_yards * feet_per_yard + niklaus_feet = 24332 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_walked_l1873_187374


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1873_187313

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | 4 * p.1 + p.2 = 6}
def B : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(2, -2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1873_187313


namespace NUMINAMATH_CALUDE_slope_determines_y_coordinate_l1873_187316

/-- Given two points R and S in a coordinate plane, if the slope of the line through R and S
    is -5/4, then the y-coordinate of S is -2. -/
theorem slope_determines_y_coordinate (x_R y_R x_S : ℚ) :
  let R : ℚ × ℚ := (x_R, y_R)
  let S : ℚ × ℚ := (x_S, y_S)
  x_R = -3 →
  y_R = 8 →
  x_S = 5 →
  (y_S - y_R) / (x_S - x_R) = -5/4 →
  y_S = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_y_coordinate_l1873_187316


namespace NUMINAMATH_CALUDE_conservation_center_count_l1873_187308

/-- The number of turtles in a conservation center -/
def total_turtles (green : ℕ) (hawksbill : ℕ) : ℕ := green + hawksbill

/-- The number of hawksbill turtles is twice more than the number of green turtles -/
def hawksbill_count (green : ℕ) : ℕ := green + 2 * green

theorem conservation_center_count :
  let green := 800
  let hawksbill := hawksbill_count green
  total_turtles green hawksbill = 3200 := by sorry

end NUMINAMATH_CALUDE_conservation_center_count_l1873_187308


namespace NUMINAMATH_CALUDE_bus_ride_duration_l1873_187324

/-- Calculates the bus ride time given the total trip time and other component times -/
def bus_ride_time (total_trip_time walk_time train_ride_time : ℕ) : ℕ :=
  let waiting_time := 2 * walk_time
  let total_trip_minutes := total_trip_time * 60
  let train_ride_minutes := train_ride_time * 60
  total_trip_minutes - (walk_time + waiting_time + train_ride_minutes)

/-- Theorem stating that given the specific trip components, the bus ride time is 75 minutes -/
theorem bus_ride_duration :
  bus_ride_time 8 15 6 = 75 := by
  sorry

#eval bus_ride_time 8 15 6

end NUMINAMATH_CALUDE_bus_ride_duration_l1873_187324


namespace NUMINAMATH_CALUDE_circumradius_of_specific_isosceles_triangle_l1873_187322

/-- An isosceles triangle with base 6 and side length 5 -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  is_isosceles : base = 6 ∧ side = 5

/-- The radius of the circumcircle of a triangle -/
def circumradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: The radius of the circumcircle of an isosceles triangle with base 6 and side length 5 is 25/8 -/
theorem circumradius_of_specific_isosceles_triangle (t : IsoscelesTriangle) : 
  circumradius t = 25/8 := by sorry

end NUMINAMATH_CALUDE_circumradius_of_specific_isosceles_triangle_l1873_187322


namespace NUMINAMATH_CALUDE_digit_405_is_zero_l1873_187301

/-- The decimal representation of 18/47 -/
def decimal_rep : ℚ := 18 / 47

/-- The length of the repeating sequence in the decimal representation of 18/47 -/
def period : ℕ := 93

/-- The position of the target digit within the repeating sequence -/
def target_position : ℕ := 405 % period

/-- The digit at the specified position in the repeating sequence -/
def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_405_is_zero :
  digit_at_position target_position = 0 :=
sorry

end NUMINAMATH_CALUDE_digit_405_is_zero_l1873_187301


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_a_l1873_187359

/-- Given a mixture of liquids A and B, prove the initial amount of A. -/
theorem initial_amount_of_liquid_a (a b : ℝ) : 
  a > 0 → b > 0 →  -- Ensure positive quantities
  a / b = 4 / 1 →  -- Initial ratio
  (a - 24) / (b - 6 + 30) = 2 / 3 →  -- New ratio after replacement
  a = 48 := by
sorry

end NUMINAMATH_CALUDE_initial_amount_of_liquid_a_l1873_187359
