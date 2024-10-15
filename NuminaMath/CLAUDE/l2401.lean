import Mathlib

namespace NUMINAMATH_CALUDE_election_votes_l2401_240102

theorem election_votes (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    loser_votes = (30 * total_votes) / 100 ∧
    winner_votes - loser_votes = 174) →
  total_votes = 435 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2401_240102


namespace NUMINAMATH_CALUDE_gcd_of_16434_24651_43002_l2401_240177

theorem gcd_of_16434_24651_43002 : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_16434_24651_43002_l2401_240177


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2401_240141

/-- The trajectory of the midpoint between a point on a parabola and a fixed point -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) : 
  y₁ = 2 * x₁^2 + 1 →  -- P is on the parabola y = 2x^2 + 1
  x = (x₁ + 0) / 2 →   -- x-coordinate of midpoint M
  y = (y₁ + (-1)) / 2 → -- y-coordinate of midpoint M
  y = 4 * x^2 :=        -- trajectory equation of M
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2401_240141


namespace NUMINAMATH_CALUDE_correct_average_after_misreading_l2401_240139

theorem correct_average_after_misreading (n : ℕ) (incorrect_avg : ℚ) 
  (misread_numbers : List (ℚ × ℚ)) :
  n = 20 ∧ 
  incorrect_avg = 85 ∧ 
  misread_numbers = [(90, 30), (120, 60), (75, 25), (150, 50), (45, 15)] →
  (n : ℚ) * incorrect_avg + (misread_numbers.map (λ p => p.1 - p.2)).sum = n * 100 := by
  sorry

#check correct_average_after_misreading

end NUMINAMATH_CALUDE_correct_average_after_misreading_l2401_240139


namespace NUMINAMATH_CALUDE_initial_balloons_l2401_240108

theorem initial_balloons (initial : ℕ) : initial + 2 = 11 → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_balloons_l2401_240108


namespace NUMINAMATH_CALUDE_initial_water_percentage_l2401_240178

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 125)
  (h2 : added_water = 8.333333333333334)
  (h3 : final_water_percentage = 25)
  (h4 : (initial_volume * x + added_water) / (initial_volume + added_water) * 100 = final_water_percentage) :
  x * 100 = 20 :=
by
  sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l2401_240178


namespace NUMINAMATH_CALUDE_projectile_max_height_l2401_240188

/-- The height function of the projectile --/
def h (t : ℝ) : ℝ := -12 * t^2 + 48 * t + 25

/-- The maximum height reached by the projectile --/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 73 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2401_240188


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2401_240130

theorem complex_fraction_simplification :
  (5 : ℚ) / ((8 : ℚ) / 15) = 75 / 8 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2401_240130


namespace NUMINAMATH_CALUDE_sum_of_herds_equals_total_l2401_240186

/-- The total number of sheep on the farm -/
def total_sheep : ℕ := 149

/-- The number of herds on the farm -/
def num_herds : ℕ := 5

/-- The number of sheep in each herd -/
def herd_sizes : Fin num_herds → ℕ
  | ⟨0, _⟩ => 23
  | ⟨1, _⟩ => 37
  | ⟨2, _⟩ => 19
  | ⟨3, _⟩ => 41
  | ⟨4, _⟩ => 29
  | ⟨n+5, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 5 n))

/-- The theorem stating that the sum of sheep in all herds equals the total number of sheep -/
theorem sum_of_herds_equals_total :
  (Finset.univ.sum fun i => herd_sizes i) = total_sheep := by
  sorry

end NUMINAMATH_CALUDE_sum_of_herds_equals_total_l2401_240186


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2401_240136

/-- Given a recipe with a ratio of butter:flour:sugar as 2:5:3 and using 9 cups of sugar,
    the total amount of ingredients used is 30 cups. -/
theorem recipe_total_cups (butter flour sugar total : ℚ) : 
  butter / sugar = 2 / 3 →
  flour / sugar = 5 / 3 →
  sugar = 9 →
  total = butter + flour + sugar →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2401_240136


namespace NUMINAMATH_CALUDE_school_travel_time_l2401_240124

theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (17 / 13 * usual_rate * (usual_time - 7) = usual_rate * usual_time) →
  usual_time = 119 / 4 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_time_l2401_240124


namespace NUMINAMATH_CALUDE_unique_point_on_line_l2401_240144

-- Define the line passing through (4, 11) and (16, 1)
def line_equation (x y : ℤ) : Prop :=
  5 * x + 6 * y = 43

-- Define the condition for positive integers
def positive_integer (n : ℤ) : Prop :=
  0 < n

theorem unique_point_on_line :
  ∃! p : ℤ × ℤ, line_equation p.1 p.2 ∧ positive_integer p.1 ∧ positive_integer p.2 ∧ p = (5, 3) :=
by
  sorry

#check unique_point_on_line

end NUMINAMATH_CALUDE_unique_point_on_line_l2401_240144


namespace NUMINAMATH_CALUDE_football_games_total_cost_l2401_240126

/-- Represents the attendance and cost data for a month of football games --/
structure MonthData where
  games : ℕ
  ticketCost : ℕ

/-- Calculates the total spent for a given month --/
def monthlyTotal (md : MonthData) : ℕ := md.games * md.ticketCost

/-- The problem statement --/
theorem football_games_total_cost 
  (thisMonth : MonthData)
  (lastMonth : MonthData)
  (nextMonth : MonthData)
  (h1 : thisMonth = { games := 11, ticketCost := 25 })
  (h2 : lastMonth = { games := 17, ticketCost := 30 })
  (h3 : nextMonth = { games := 16, ticketCost := 35 }) :
  monthlyTotal thisMonth + monthlyTotal lastMonth + monthlyTotal nextMonth = 1345 := by
  sorry

end NUMINAMATH_CALUDE_football_games_total_cost_l2401_240126


namespace NUMINAMATH_CALUDE_sine_cosine_transformation_l2401_240194

open Real

theorem sine_cosine_transformation (x : ℝ) :
  sin (2 * x) - Real.sqrt 3 * cos (2 * x) = 2 * sin (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_transformation_l2401_240194


namespace NUMINAMATH_CALUDE_boxes_with_pans_is_eight_l2401_240140

/-- Represents the arrangement of teacups and boxes. -/
structure TeacupArrangement where
  total_boxes : Nat
  cups_per_box : Nat
  cups_broken_per_box : Nat
  cups_left : Nat

/-- Calculates the number of boxes containing pans. -/
def boxes_with_pans (arrangement : TeacupArrangement) : Nat :=
  let teacup_boxes := arrangement.cups_left / (arrangement.cups_per_box - arrangement.cups_broken_per_box)
  let remaining_boxes := arrangement.total_boxes - teacup_boxes
  remaining_boxes / 2

/-- Theorem stating that the number of boxes with pans is 8. -/
theorem boxes_with_pans_is_eight : 
  boxes_with_pans { total_boxes := 26
                  , cups_per_box := 20
                  , cups_broken_per_box := 2
                  , cups_left := 180 } = 8 := by
  sorry


end NUMINAMATH_CALUDE_boxes_with_pans_is_eight_l2401_240140


namespace NUMINAMATH_CALUDE_quadratic_roots_equivalence_l2401_240133

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 -/
def QuadraticFunction (a b c : ℝ) (h : a > 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_roots_equivalence (a b c : ℝ) (h : a > 0) :
  let f := QuadraticFunction a b c h
  (f (f (-b / (2 * a))) < 0) ↔
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ f (f y₁) = 0 ∧ f (f y₂) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_equivalence_l2401_240133


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l2401_240179

/-- Given a cistern with two taps, prove the emptying time of the second tap -/
theorem cistern_emptying_time (fill_time : ℝ) (combined_time : ℝ) (empty_time : ℝ) : 
  fill_time = 4 → combined_time = 44 / 7 → empty_time = 11 → 
  1 / fill_time - 1 / empty_time = 1 / combined_time := by
sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l2401_240179


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2401_240182

theorem quadratic_roots_sum_of_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧
    x₁^2 + x₂^2 = 19) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2401_240182


namespace NUMINAMATH_CALUDE_no_solution_for_square_free_l2401_240176

/-- A positive integer is square-free if its prime factorization contains no repeated factors. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p ^ 2 ∣ n) → p = 1

/-- Two natural numbers are relatively prime if their greatest common divisor is 1. -/
def RelativelyPrime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

theorem no_solution_for_square_free (n : ℕ) (hn : IsSquareFree n) :
  ¬∃ (x y : ℕ), RelativelyPrime x y ∧ ((x + y) ^ 3 ∣ x ^ n + y ^ n) :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_square_free_l2401_240176


namespace NUMINAMATH_CALUDE_hungarian_olympiad_1959_l2401_240105

theorem hungarian_olympiad_1959 (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  ∃ k : ℤ, (x^n * (y-z) + y^n * (z-x) + z^n * (x-y)) / ((x-y)*(x-z)*(y-z)) = k :=
sorry

end NUMINAMATH_CALUDE_hungarian_olympiad_1959_l2401_240105


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l2401_240122

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 16 ways to distribute 7 indistinguishable balls
    into 3 distinguishable boxes, with each box containing at least one ball -/
theorem distribute_seven_balls_three_boxes :
  distribute_balls 7 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l2401_240122


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l2401_240164

theorem greatest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 17 ∣ n → n ≤ 9996 ∧ 17 ∣ 9996 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l2401_240164


namespace NUMINAMATH_CALUDE_f_minus_three_halves_value_l2401_240121

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def f_squared_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → f x = x^2

theorem f_minus_three_halves_value
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_period : has_period_two f)
  (h_squared : f_squared_on_unit_interval f) :
  f (-3/2) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_three_halves_value_l2401_240121


namespace NUMINAMATH_CALUDE_pentagonal_tiles_count_l2401_240113

theorem pentagonal_tiles_count (t p : ℕ) : 
  t + p = 30 →  -- Total number of tiles
  3 * t + 5 * p = 100 →  -- Total number of edges
  p = 5  -- Number of pentagonal tiles
  := by sorry

end NUMINAMATH_CALUDE_pentagonal_tiles_count_l2401_240113


namespace NUMINAMATH_CALUDE_negation_equivalence_not_always_greater_product_quadratic_roots_condition_l2401_240193

-- Statement 1
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
sorry

-- Statement 2
theorem not_always_greater_product :
  ∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d :=
sorry

-- Statement 3
theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a - 3) * x + a = 0 ∧ y^2 + (a - 3) * y + a = 0) →
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_not_always_greater_product_quadratic_roots_condition_l2401_240193


namespace NUMINAMATH_CALUDE_problem_statement_l2401_240132

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2401_240132


namespace NUMINAMATH_CALUDE_elimination_method_l2401_240149

theorem elimination_method (x y : ℝ) : 
  (5 * x - 2 * y = 4) → 
  (2 * x + 3 * y = 9) → 
  ∃ (a b : ℝ), a = 2 ∧ b = -5 ∧ 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = a * 4 + b * 9) ∧
  (a * 5 + b * 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_elimination_method_l2401_240149


namespace NUMINAMATH_CALUDE_parabola_properties_l2401_240163

/-- Represents a parabola of the form y = ax^2 + bx - 4 -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x - 4

theorem parabola_properties (p : Parabola) 
  (h1 : p.contains (-2) 0)
  (h2 : p.contains (-1) (-4))
  (h3 : p.contains 0 (-4))
  (h4 : p.contains 1 0)
  (h5 : p.contains 2 8) :
  (p.contains 0 (-4)) ∧ 
  (p.a = 2 ∧ p.b = 2) ∧ 
  (p.contains (-3) 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2401_240163


namespace NUMINAMATH_CALUDE_base6_addition_l2401_240148

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The first number in base 6 -/
def num1 : List Nat := [2, 3, 4, 3]

/-- The second number in base 6 -/
def num2 : List Nat := [1, 5, 3, 2, 5]

/-- The expected result in base 6 -/
def result : List Nat := [2, 2, 1, 1, 2]

theorem base6_addition :
  decimalToBase6 (base6ToDecimal num1 + base6ToDecimal num2) = result := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l2401_240148


namespace NUMINAMATH_CALUDE_fib_100_mod_7_l2401_240150

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_100_mod_7 : fib 100 % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_7_l2401_240150


namespace NUMINAMATH_CALUDE_quadratic_real_roots_iff_k_le_4_l2401_240181

/-- The quadratic function f(x) = (k - 3)x² + 2x + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- The discriminant of the quadratic function f -/
def discriminant (k : ℝ) : ℝ := 4 - 4 * k + 12

theorem quadratic_real_roots_iff_k_le_4 :
  ∀ k : ℝ, (∃ x : ℝ, f k x = 0) ↔ k ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_iff_k_le_4_l2401_240181


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2401_240187

-- Define the function f(x) = x³ - 2x² + 5
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5

-- Define the interval [-2, 2]
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem stating the maximum and minimum values of f(x) on the interval [-2, 2]
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 5 ∧ min = -11 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2401_240187


namespace NUMINAMATH_CALUDE_scientific_notation_of_600000_l2401_240154

theorem scientific_notation_of_600000 : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 600000 = a * (10 : ℝ) ^ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_600000_l2401_240154


namespace NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l2401_240115

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 + a = 0

-- Define lines l₁ and l₂
def line_l1 (x y : ℝ) : Prop := 4 * x + y + 3 = 0
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 5 = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define line m
def line_m (x y : ℝ) : Prop := 3 * x + y + 1 = 0

theorem fixed_point_and_bisecting_line :
  (∀ a : ℝ, line_l a (point_P.1) (point_P.2)) ∧
  (∀ x y : ℝ, line_m x y ↔ 
    ∃ t : ℝ, 
      line_l1 t (-4*t-3) ∧ 
      line_l2 (-t-2) (4*t+7) ∧
      point_P = ((t + (-t-2))/2, ((-4*t-3) + (4*t+7))/2)) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l2401_240115


namespace NUMINAMATH_CALUDE_product_of_roots_quartic_l2401_240158

theorem product_of_roots_quartic (p q r s : ℂ) : 
  (3 * p^4 - 8 * p^3 + p^2 - 10 * p - 24 = 0) →
  (3 * q^4 - 8 * q^3 + q^2 - 10 * q - 24 = 0) →
  (3 * r^4 - 8 * r^3 + r^2 - 10 * r - 24 = 0) →
  (3 * s^4 - 8 * s^3 + s^2 - 10 * s - 24 = 0) →
  p * q * r * s = -8 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_quartic_l2401_240158


namespace NUMINAMATH_CALUDE_scientific_notation_152300_l2401_240161

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_152300 :
  toScientificNotation 152300 = ScientificNotation.mk 1.523 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_152300_l2401_240161


namespace NUMINAMATH_CALUDE_ones_digit_of_prime_arithmetic_sequence_l2401_240153

theorem ones_digit_of_prime_arithmetic_sequence (a b c d : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧  -- Four prime numbers
  a > 5 ∧                                  -- a is greater than 5
  b = a + 6 ∧ c = b + 6 ∧ d = c + 6 ∧      -- Arithmetic sequence with common difference 6
  a < b ∧ b < c ∧ c < d →                  -- Increasing sequence
  a % 10 = 1 :=                            -- The ones digit of a is 1
by sorry

end NUMINAMATH_CALUDE_ones_digit_of_prime_arithmetic_sequence_l2401_240153


namespace NUMINAMATH_CALUDE_lcm_problem_l2401_240155

theorem lcm_problem (a b c : ℕ) 
  (h1 : Nat.lcm a b = 60) 
  (h2 : Nat.lcm a c = 270) : 
  Nat.lcm b c = 540 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l2401_240155


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l2401_240157

theorem gcd_lcm_problem (a b : ℕ) : 
  a > 0 → b > 0 → Nat.gcd a b = 45 → Nat.lcm a b = 1260 → a = 180 → b = 315 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l2401_240157


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2401_240173

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 10 →
  a^2 + b^2 = 100 →
  (1/2) * c * d = 250 →
  c/a = d/b →
  c + d = 30 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2401_240173


namespace NUMINAMATH_CALUDE_z_min_max_in_D_l2401_240152

-- Define the function z
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 - 2 * p.2 ≤ 0 ∧ p.1 + p.2 - 6 ≤ 0}

-- Theorem statement
theorem z_min_max_in_D :
  (∃ p ∈ D, ∀ q ∈ D, z p.1 p.2 ≤ z q.1 q.2) ∧
  (∃ p ∈ D, ∀ q ∈ D, z p.1 p.2 ≥ z q.1 q.2) ∧
  (∃ p ∈ D, z p.1 p.2 = 0) ∧
  (∃ p ∈ D, z p.1 p.2 = 32) :=
sorry

end NUMINAMATH_CALUDE_z_min_max_in_D_l2401_240152


namespace NUMINAMATH_CALUDE_josh_spending_l2401_240114

/-- Josh's spending problem -/
theorem josh_spending (x y : ℝ) : 
  (x - 1.75 - y = 6) → y = x - 7.75 := by
sorry

end NUMINAMATH_CALUDE_josh_spending_l2401_240114


namespace NUMINAMATH_CALUDE_complex_modulus_l2401_240184

theorem complex_modulus (x y : ℝ) (z : ℂ) (h : z = x + y * I) 
  (eq : (1/2 * x - y) + (x + y) * I = 3 * I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2401_240184


namespace NUMINAMATH_CALUDE_probability_consecutive_dali_prints_l2401_240120

/-- The probability of consecutive Dali prints in a random arrangement --/
theorem probability_consecutive_dali_prints
  (total_pieces : ℕ)
  (dali_prints : ℕ)
  (h1 : total_pieces = 12)
  (h2 : dali_prints = 4)
  (h3 : dali_prints ≤ total_pieces) :
  (dali_prints.factorial * (total_pieces - dali_prints + 1).factorial) /
    total_pieces.factorial = 1 / 55 :=
by sorry

end NUMINAMATH_CALUDE_probability_consecutive_dali_prints_l2401_240120


namespace NUMINAMATH_CALUDE_game_result_l2401_240199

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result :
  (total_points allie_rolls) * (total_points betty_rolls) = 702 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2401_240199


namespace NUMINAMATH_CALUDE_min_prize_cost_is_11_l2401_240118

def min_prize_cost (x y : ℕ) : ℕ := 3 * x + 2 * y

theorem min_prize_cost_is_11 :
  ∃ (x y : ℕ),
    x + y ≤ 10 ∧
    (x : ℤ) - y ≤ 2 ∧
    y - x ≤ 2 ∧
    x ≥ 3 ∧
    min_prize_cost x y = 11 ∧
    ∀ (a b : ℕ), a + b ≤ 10 → (a : ℤ) - b ≤ 2 → b - a ≤ 2 → a ≥ 3 → min_prize_cost a b ≥ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_min_prize_cost_is_11_l2401_240118


namespace NUMINAMATH_CALUDE_last_digit_2014_power_2014_l2401_240197

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo 10 -/
def powerMod10 (base exponent : ℕ) : ℕ :=
  (base ^ exponent) % 10

theorem last_digit_2014_power_2014 :
  lastDigit (powerMod10 2014 2014) = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_2014_power_2014_l2401_240197


namespace NUMINAMATH_CALUDE_share_distribution_l2401_240143

theorem share_distribution (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : total = 6600) (h2 : ratio1 = 2) (h3 : ratio2 = 4) (h4 : ratio3 = 6) :
  (total * ratio1) / (ratio1 + ratio2 + ratio3) = 1100 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l2401_240143


namespace NUMINAMATH_CALUDE_four_digit_sum_problem_l2401_240165

theorem four_digit_sum_problem :
  ∃ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    a > b ∧ b > c ∧ c > d ∧
    1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a = 10477 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_problem_l2401_240165


namespace NUMINAMATH_CALUDE_squirrel_journey_time_l2401_240104

/-- Calculates the total journey time in minutes for a squirrel gathering nuts -/
theorem squirrel_journey_time (distance_to_tree : ℝ) (speed_to_tree : ℝ) (speed_from_tree : ℝ) :
  distance_to_tree = 2 →
  speed_to_tree = 3 →
  speed_from_tree = 2 →
  (distance_to_tree / speed_to_tree + distance_to_tree / speed_from_tree) * 60 = 100 := by
  sorry

#check squirrel_journey_time

end NUMINAMATH_CALUDE_squirrel_journey_time_l2401_240104


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l2401_240134

/-- The line y = kx is tangent to the curve y = 2e^x if and only if k = 2e -/
theorem tangent_line_to_exponential_curve (k : ℝ) :
  (∃ x₀ : ℝ, k * x₀ = 2 * Real.exp x₀ ∧
             k = 2 * Real.exp x₀) ↔ k = 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l2401_240134


namespace NUMINAMATH_CALUDE_residue_problem_l2401_240174

theorem residue_problem : Int.mod (Int.mod (-1043) 36) 10 = 1 := by sorry

end NUMINAMATH_CALUDE_residue_problem_l2401_240174


namespace NUMINAMATH_CALUDE_root_product_l2401_240142

theorem root_product (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ∃ s : ℝ, ((c + 1/d)^2 - r*(c + 1/d) + s = 0) ∧ 
           ((d + 1/c)^2 - r*(d + 1/c) + s = 0) ∧ 
           (s = 16/3) := by
  sorry

end NUMINAMATH_CALUDE_root_product_l2401_240142


namespace NUMINAMATH_CALUDE_sunflower_seeds_count_l2401_240106

/-- The number of sunflower plants -/
def num_sunflowers : ℕ := 6

/-- The number of dandelion plants -/
def num_dandelions : ℕ := 8

/-- The number of seeds per dandelion plant -/
def seeds_per_dandelion : ℕ := 12

/-- The percentage of total seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64/100

/-- The number of seeds per sunflower plant -/
def seeds_per_sunflower : ℕ := 9

theorem sunflower_seeds_count :
  let total_dandelion_seeds := num_dandelions * seeds_per_dandelion
  let total_seeds := total_dandelion_seeds / dandelion_seed_percentage
  let total_sunflower_seeds := total_seeds - total_dandelion_seeds
  seeds_per_sunflower = total_sunflower_seeds / num_sunflowers := by
sorry

end NUMINAMATH_CALUDE_sunflower_seeds_count_l2401_240106


namespace NUMINAMATH_CALUDE_hyperbola_chord_of_contact_l2401_240137

/-- The equation of the chord of contact for a hyperbola -/
theorem hyperbola_chord_of_contact 
  (a b x₀ y₀ : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_not_on_hyperbola : (x₀^2 / a^2) - (y₀^2 / b^2) ≠ 1) :
  ∃ (P₁ P₂ : ℝ × ℝ),
    (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → 
      ((x₀ * x / a^2) - (y₀ * y / b^2) = 1 ↔ 
        (∃ t : ℝ, (x, y) = t • P₁ + (1 - t) • P₂))) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_chord_of_contact_l2401_240137


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l2401_240159

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ (x y u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f (x^u * y^v) ≤ f x^(1/(4*u)) * f y^(1/(4*v))

/-- The main theorem stating the form of functions satisfying the condition -/
theorem characterize_satisfying_functions :
  ∀ (f : ℝ → ℝ), (∀ x, x > 1 → f x > 1) →
    SatisfiesCondition f →
    ∃ (c : ℝ), c > 1 ∧ ∀ x, x > 1 → f x = c^(1/Real.log x) :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l2401_240159


namespace NUMINAMATH_CALUDE_backpack_price_calculation_l2401_240168

theorem backpack_price_calculation
  (num_backpacks : ℕ)
  (monogram_cost : ℚ)
  (total_cost : ℚ)
  (h1 : num_backpacks = 5)
  (h2 : monogram_cost = 12)
  (h3 : total_cost = 140) :
  (total_cost - num_backpacks * monogram_cost) / num_backpacks = 16 :=
by sorry

end NUMINAMATH_CALUDE_backpack_price_calculation_l2401_240168


namespace NUMINAMATH_CALUDE_subset_condition_intersection_condition_l2401_240156

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1: B ⊆ A iff m ∈ [-1, +∞)
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≥ -1 := by sorry

-- Theorem 2: ∃x ∈ A such that x ∈ B iff m ∈ [-4, 2]
theorem intersection_condition (m : ℝ) : (∃ x, x ∈ A ∧ x ∈ B m) ↔ -4 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_condition_l2401_240156


namespace NUMINAMATH_CALUDE_binary_10010_is_18_l2401_240128

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10010_is_18 : 
  binary_to_decimal [false, true, false, false, true] = 18 := by
  sorry

end NUMINAMATH_CALUDE_binary_10010_is_18_l2401_240128


namespace NUMINAMATH_CALUDE_sum_of_digits_888_base8_l2401_240175

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_888_base8 : sumDigits (toBase8 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_888_base8_l2401_240175


namespace NUMINAMATH_CALUDE_sum_inequality_l2401_240147

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11) ≤ 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_sum_inequality_l2401_240147


namespace NUMINAMATH_CALUDE_double_inequality_solution_l2401_240190

theorem double_inequality_solution (x : ℝ) :
  (-2 < (x^2 - 16*x + 15) / (x^2 - 2*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 2*x + 5) < 1) ↔
  (5/7 < x ∧ x < 5/3) ∨ (5 < x) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l2401_240190


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l2401_240135

theorem divisibility_by_twelve (m : Nat) : m ≤ 9 → (915 * 10 + m) % 12 = 0 ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l2401_240135


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2401_240127

/-- Proves that 15 more children got on the bus than got off during the entire ride -/
theorem bus_ride_difference (initial : ℕ) (final : ℕ) 
  (got_off_first : ℕ) (got_off_second : ℕ) (got_off_third : ℕ) 
  (h1 : initial = 20)
  (h2 : final = 35)
  (h3 : got_off_first = 54)
  (h4 : got_off_second = 30)
  (h5 : got_off_third = 15) :
  ∃ (got_on_total : ℕ),
    got_on_total = final - initial + got_off_first + got_off_second + got_off_third ∧
    got_on_total - (got_off_first + got_off_second + got_off_third) = 15 := by
  sorry


end NUMINAMATH_CALUDE_bus_ride_difference_l2401_240127


namespace NUMINAMATH_CALUDE_range_of_dot_product_line_passes_fixed_point_l2401_240110

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus and vertex
def F : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (-2, 0)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from a point to another
def vector_to (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Theorem 1: Range of PF · PA
theorem range_of_dot_product :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 →
  0 ≤ dot_product (vector_to P F) (vector_to P A) ∧
  dot_product (vector_to P F) (vector_to P A) ≤ 12 :=
sorry

-- Define the line
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Theorem 2: Line passes through fixed point
theorem line_passes_fixed_point :
  ∀ k m : ℝ, ∀ M N : ℝ × ℝ,
  M ≠ N →
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  M.2 = line k m M.1 →
  N.2 = line k m N.1 →
  (∃ H : ℝ × ℝ, 
    dot_product (vector_to A H) (vector_to M N) = 0 ∧
    dot_product (vector_to A H) (vector_to A H) = 
    dot_product (vector_to M H) (vector_to H N)) →
  line k m (-2/7) = 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_dot_product_line_passes_fixed_point_l2401_240110


namespace NUMINAMATH_CALUDE_sum_in_base3_l2401_240129

/-- Represents a number in base 3 --/
def Base3 : Type := List (Fin 3)

/-- Converts a natural number to its base 3 representation --/
def toBase3 (n : ℕ) : Base3 := sorry

/-- Adds two Base3 numbers --/
def addBase3 (a b : Base3) : Base3 := sorry

/-- Theorem: The sum of 2₃, 21₃, 110₃, and 2202₃ in base 3 is 11000₃ --/
theorem sum_in_base3 :
  addBase3 (toBase3 2)
    (addBase3 (toBase3 7)
      (addBase3 (toBase3 12)
        (toBase3 72))) = [1, 1, 0, 0, 0] := by sorry

end NUMINAMATH_CALUDE_sum_in_base3_l2401_240129


namespace NUMINAMATH_CALUDE_apple_purchase_multiple_l2401_240116

theorem apple_purchase_multiple : ∀ x : ℕ,
  (15 : ℕ) + 15 * x + 60 * x = (240 : ℕ) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_multiple_l2401_240116


namespace NUMINAMATH_CALUDE_gcd_7200_13230_l2401_240185

theorem gcd_7200_13230 : Int.gcd 7200 13230 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7200_13230_l2401_240185


namespace NUMINAMATH_CALUDE_cube_split_theorem_l2401_240146

/-- The sum of consecutive integers from 2 to n -/
def consecutiveSum (n : ℕ) : ℕ := (n + 2) * (n - 1) / 2

/-- The nth odd number starting from 3 -/
def nthOddFrom3 (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) :
  (∃ k, k ∈ Finset.range m ∧ nthOddFrom3 (consecutiveSum m - k) = 333) ↔ m = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_theorem_l2401_240146


namespace NUMINAMATH_CALUDE_slope_equals_twelve_implies_m_equals_negative_two_l2401_240198

/-- Given two points A(-m, 6) and B(1, 3m), prove that m = -2 when the slope of the line passing through these points is 12. -/
theorem slope_equals_twelve_implies_m_equals_negative_two (m : ℝ) : 
  (let A : ℝ × ℝ := (-m, 6)
   let B : ℝ × ℝ := (1, 3*m)
   (3*m - 6) / (1 - (-m)) = 12) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_equals_twelve_implies_m_equals_negative_two_l2401_240198


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l2401_240151

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

def binary_1101101 : List Bool := [true, true, false, true, true, false, true]
def binary_1101 : List Bool := [true, true, false, true]
def binary_result : List Bool := [true, false, false, true, false, true, false, true, false, false, false, true]

theorem binary_multiplication_theorem :
  nat_to_binary ((binary_to_nat binary_1101101) * (binary_to_nat binary_1101)) = binary_result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l2401_240151


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2401_240160

theorem negation_of_proposition (p : Prop) : 
  (p = ∀ x : ℝ, 2 * x^2 + 1 > 0) → 
  (¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2401_240160


namespace NUMINAMATH_CALUDE_arithmetic_operations_with_five_l2401_240107

theorem arithmetic_operations_with_five (x : ℝ) : ((x + 5) * 5 - 5) / 5 = 5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_with_five_l2401_240107


namespace NUMINAMATH_CALUDE_probability_most_expensive_chosen_l2401_240196

def num_computers : ℕ := 10
def num_display : ℕ := 3

theorem probability_most_expensive_chosen :
  (Nat.choose (num_computers - 2) (num_display - 2)) / (Nat.choose num_computers num_display) = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_probability_most_expensive_chosen_l2401_240196


namespace NUMINAMATH_CALUDE_problem_statement_l2401_240119

theorem problem_statement : 
  let p := ∀ x : ℤ, x^2 > x
  let q := ∃ x : ℝ, x > 0 ∧ x + 2/x > 4
  (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2401_240119


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2401_240170

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2401_240170


namespace NUMINAMATH_CALUDE_probability_white_given_popped_l2401_240100

-- Define the probabilities
def P_white : ℝ := 0.4
def P_yellow : ℝ := 0.4
def P_red : ℝ := 0.2
def P_pop_given_white : ℝ := 0.7
def P_pop_given_yellow : ℝ := 0.5
def P_pop_given_red : ℝ := 0

-- Define the theorem
theorem probability_white_given_popped :
  let P_popped : ℝ := P_pop_given_white * P_white + P_pop_given_yellow * P_yellow + P_pop_given_red * P_red
  (P_pop_given_white * P_white) / P_popped = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_given_popped_l2401_240100


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2401_240109

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1
    that have the same asymptotes, M equals 225/16. -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) → M = 225 / 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2401_240109


namespace NUMINAMATH_CALUDE_range_of_a_l2401_240123

-- Define the property that the inequality holds for all real x
def inequality_holds_for_all (a : ℝ) : Prop :=
  ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1

-- Define the theorem
theorem range_of_a (a : ℝ) :
  inequality_holds_for_all a ∧ a > 0 → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2401_240123


namespace NUMINAMATH_CALUDE_ball_probabilities_l2401_240125

/-- The probability of drawing a red ball on the second draw -/
def prob_red_second (total : ℕ) (red : ℕ) : ℚ :=
  red / total

/-- The probability of drawing two balls of the same color -/
def prob_same_color (total : ℕ) (red : ℕ) (green : ℕ) : ℚ :=
  (red * (red - 1) + green * (green - 1)) / (total * (total - 1))

/-- The probability of drawing two red balls -/
def prob_two_red (total : ℕ) (red : ℕ) : ℚ :=
  (red * (red - 1)) / (total * (total - 1))

theorem ball_probabilities :
  let total := 6
  let red := 2
  let green := 4
  (prob_red_second total red = 1/3) ∧
  (prob_same_color total red green = 7/15) ∧
  (∃ n : ℕ, prob_two_red (n + 2) 2 = 1/21 ∧ n = 5) :=
by sorry


end NUMINAMATH_CALUDE_ball_probabilities_l2401_240125


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2401_240169

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_selection_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that form an edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_selection_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2401_240169


namespace NUMINAMATH_CALUDE_max_classes_less_than_1968_l2401_240131

/-- Relation between two natural numbers where they belong to the same class if one can be obtained from the other by deleting two adjacent digits or identical groups of digits -/
def SameClass (m n : ℕ) : Prop := sorry

/-- The maximum number of equivalence classes under the SameClass relation -/
def MaxClasses : ℕ := sorry

theorem max_classes_less_than_1968 : MaxClasses < 1968 := by sorry

end NUMINAMATH_CALUDE_max_classes_less_than_1968_l2401_240131


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2401_240138

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2401_240138


namespace NUMINAMATH_CALUDE_rad_divides_theorem_l2401_240166

-- Define the rad function
def rad : ℕ → ℕ
| 0 => 1
| 1 => 1
| n+2 => (Finset.prod (Nat.factors (n+2)).toFinset id)

-- Define a polynomial with nonnegative integer coefficients
def NonnegIntPoly := {f : Polynomial ℕ // ∀ i, 0 ≤ f.coeff i}

theorem rad_divides_theorem (f : NonnegIntPoly) :
  (∀ n : ℕ, rad (f.val.eval n) ∣ rad (f.val.eval (n^(rad n)))) →
  ∃ a m : ℕ, f.val = Polynomial.monomial m a :=
sorry

end NUMINAMATH_CALUDE_rad_divides_theorem_l2401_240166


namespace NUMINAMATH_CALUDE_smallest_violet_balls_l2401_240103

theorem smallest_violet_balls (x : ℕ) (y : ℕ) : 
  x > 0 ∧ 
  x % 120 = 0 ∧ 
  x / 10 + x / 8 + x / 3 + (x / 10 + 9) + (x / 8 + 10) + 8 + y = x ∧
  y = x / 60 * 13 - 27 →
  y ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_smallest_violet_balls_l2401_240103


namespace NUMINAMATH_CALUDE_min_value_expression_l2401_240172

theorem min_value_expression (x : ℝ) : 
  (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 ∧ 
  ∃ y : ℝ, (8 - y) * (6 - y) * (8 + y) * (6 + y) = -196 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2401_240172


namespace NUMINAMATH_CALUDE_tablet_interval_l2401_240162

/-- Given a person who takes 5 tablets over 60 minutes with equal intervals, 
    prove that the interval between tablets is 15 minutes. -/
theorem tablet_interval (total_tablets : ℕ) (total_time : ℕ) (h1 : total_tablets = 5) (h2 : total_time = 60) :
  (total_time / (total_tablets - 1) : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tablet_interval_l2401_240162


namespace NUMINAMATH_CALUDE_max_integer_value_x_l2401_240189

theorem max_integer_value_x (x : ℤ) : 
  (3 : ℚ) * x - 1/4 ≤ 1/3 * x - 2 → x ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_value_x_l2401_240189


namespace NUMINAMATH_CALUDE_theatre_sales_calculation_l2401_240111

/-- Calculates the total sales amount for a theatre performance given ticket prices and quantities sold. -/
theorem theatre_sales_calculation 
  (price1 price2 : ℚ) 
  (total_tickets sold1 : ℕ) 
  (h1 : price1 = 4.5)
  (h2 : price2 = 6)
  (h3 : total_tickets = 380)
  (h4 : sold1 = 205) :
  price1 * sold1 + price2 * (total_tickets - sold1) = 1972.5 :=
by sorry

end NUMINAMATH_CALUDE_theatre_sales_calculation_l2401_240111


namespace NUMINAMATH_CALUDE_coefficient_x_fourth_power_l2401_240171

theorem coefficient_x_fourth_power (n : ℕ) (k : ℕ) : 
  n = 6 → k = 4 → (Nat.choose n k) * (2^k) = 240 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_fourth_power_l2401_240171


namespace NUMINAMATH_CALUDE_empire_state_height_is_443_l2401_240192

/-- The height of the Petronas Towers in meters -/
def petronas_height : ℝ := 452

/-- The height difference between the Empire State Building and the Petronas Towers in meters -/
def height_difference : ℝ := 9

/-- The height of the Empire State Building in meters -/
def empire_state_height : ℝ := petronas_height - height_difference

theorem empire_state_height_is_443 : empire_state_height = 443 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_height_is_443_l2401_240192


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2401_240195

/-- Given positive integers a, b, c with (a,b,c) = 1 and (a,b) = d, 
    if n > (ab/d) + cd - a - b - c, then there exist nonnegative integers x, y, z 
    such that ax + by + cz = n -/
theorem diophantine_equation_solution 
  (a b c d : ℕ+) (n : ℕ) 
  (h1 : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (h2 : Nat.gcd a.val b.val = d.val)
  (h3 : n > a.val * b.val / d.val + c.val * d.val - a.val - b.val - c.val) :
  ∃ x y z : ℕ, a.val * x + b.val * y + c.val * z = n :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2401_240195


namespace NUMINAMATH_CALUDE_triangle_side_length_l2401_240167

theorem triangle_side_length (a b : ℝ) (A B : ℝ) :
  b = 4 * Real.sqrt 6 →
  B = π / 3 →
  A = π / 4 →
  a = (4 * Real.sqrt 6) * (Real.sin (π / 4)) / (Real.sin (π / 3)) →
  a = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2401_240167


namespace NUMINAMATH_CALUDE_base3_sum_equality_l2401_240183

/-- Converts a base 3 number represented as a list of digits to a natural number. -/
def base3ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- The sum of 2₃, 121₃, 1212₃, and 12121₃ equals 2111₃ in base 3. -/
theorem base3_sum_equality : 
  base3ToNat [2] + base3ToNat [1, 2, 1] + base3ToNat [2, 1, 2, 1] + base3ToNat [1, 2, 1, 2, 1] = 
  base3ToNat [1, 1, 1, 2] := by
  sorry

#eval base3ToNat [2] + base3ToNat [1, 2, 1] + base3ToNat [2, 1, 2, 1] + base3ToNat [1, 2, 1, 2, 1]
#eval base3ToNat [1, 1, 1, 2]

end NUMINAMATH_CALUDE_base3_sum_equality_l2401_240183


namespace NUMINAMATH_CALUDE_smallest_k_coprime_subset_l2401_240145

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_k_coprime_subset : ∃ (k : ℕ),
  (k = 51) ∧ 
  (∀ (S : Finset ℕ), S ⊆ Finset.range 100 → S.card ≥ k → 
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ is_coprime a b) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (S : Finset ℕ), S ⊆ Finset.range 100 ∧ S.card = k' ∧
      ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬is_coprime a b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_coprime_subset_l2401_240145


namespace NUMINAMATH_CALUDE_line_xz_plane_intersection_l2401_240112

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The xz-plane -/
def xzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.x = l.p1.x + t * (l.p2.x - l.p1.x) ∧
            p.y = l.p1.y + t * (l.p2.y - l.p1.y) ∧
            p.z = l.p1.z + t * (l.p2.z - l.p1.z)

theorem line_xz_plane_intersection :
  let l : Line3D := { p1 := ⟨2, -1, 3⟩, p2 := ⟨6, 7, -2⟩ }
  let p : Point3D := ⟨2.5, 0, 2.375⟩
  pointOnLine p l ∧ p ∈ xzPlane :=
by sorry

end NUMINAMATH_CALUDE_line_xz_plane_intersection_l2401_240112


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2401_240101

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 8*x - 4*k = 0 ∧ 
   ∀ y : ℝ, y^2 - 8*y - 4*k = 0 → y = x) → 
  k = -4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2401_240101


namespace NUMINAMATH_CALUDE_exists_log_sum_eq_log_sum_skew_lines_iff_no_common_plane_l2401_240117

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Proposition p
theorem exists_log_sum_eq_log_sum : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ log (a + b) = log a + log b :=
sorry

-- Define a type for lines in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define what it means for a line to lie on a plane
def line_on_plane (l : Line3D) (p : Plane3D) : Prop :=
sorry

-- Define what it means for two lines to be skew
def skew_lines (l1 l2 : Line3D) : Prop :=
∀ (p : Plane3D), ¬(line_on_plane l1 p ∧ line_on_plane l2 p)

-- Proposition q
theorem skew_lines_iff_no_common_plane (l1 l2 : Line3D) :
  skew_lines l1 l2 ↔ ∀ (p : Plane3D), ¬(line_on_plane l1 p ∧ line_on_plane l2 p) :=
sorry

end NUMINAMATH_CALUDE_exists_log_sum_eq_log_sum_skew_lines_iff_no_common_plane_l2401_240117


namespace NUMINAMATH_CALUDE_symmetric_points_solution_l2401_240180

/-- 
Given two points P and Q that are symmetric about the x-axis,
prove that their coordinates satisfy the given conditions and
result in specific values for a and b.
-/
theorem symmetric_points_solution :
  ∀ (a b : ℝ),
  let P : ℝ × ℝ := (-a + 3*b, 3)
  let Q : ℝ × ℝ := (-5, a - 2*b)
  -- P and Q are symmetric about the x-axis
  (P.1 = Q.1 ∧ P.2 = -Q.2) →
  (a = -19 ∧ b = -8) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_solution_l2401_240180


namespace NUMINAMATH_CALUDE_range_of_m_l2401_240191

def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem range_of_m :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ (∃ x : ℝ, x ∈ A ∧ x ∉ B m)) ↔ 
  (∀ m : ℝ, m > 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2401_240191
