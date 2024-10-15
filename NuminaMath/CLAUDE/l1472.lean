import Mathlib

namespace NUMINAMATH_CALUDE_average_monthly_income_is_69_l1472_147237

/-- Proves that the average monthly income for a 10-month period is 69 given specific income and expense conditions. -/
theorem average_monthly_income_is_69 
  (X : ℝ) -- Income base for first 6 months
  (Y : ℝ) -- Income for last 4 months
  (h1 : (6 * (1.1 * X) + 4 * Y) / 10 = 69) -- Average income condition
  (h2 : 4 * (Y - 60) - 6 * (70 - 1.1 * X) = 30) -- Debt and savings condition
  : (6 * (1.1 * X) + 4 * Y) / 10 = 69 := by
  sorry

#check average_monthly_income_is_69

end NUMINAMATH_CALUDE_average_monthly_income_is_69_l1472_147237


namespace NUMINAMATH_CALUDE_eighteen_men_handshakes_l1472_147209

/-- The maximum number of handshakes without cyclic handshakes for n men -/
def maxHandshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 18 men, the maximum number of handshakes without cyclic handshakes is 153 -/
theorem eighteen_men_handshakes :
  maxHandshakes 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_men_handshakes_l1472_147209


namespace NUMINAMATH_CALUDE_inequality_solution_l1472_147273

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 ∨ (0 < a ∧ a < 1) then
    {x | a < x ∧ x < 1/a}
  else if a = 1 ∨ a = -1 then
    ∅
  else if a > 1 ∨ (-1 < a ∧ a < 0) then
    {x | 1/a < x ∧ x < a}
  else
    ∅

theorem inequality_solution (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | x^2 - (a + 1/a)*x + 1 < 0} = solution_set a :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1472_147273


namespace NUMINAMATH_CALUDE_at_least_one_passes_l1472_147251

theorem at_least_one_passes (p : ℝ) (h : p = 1/3) :
  let q := 1 - p
  1 - q^3 = 19/27 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l1472_147251


namespace NUMINAMATH_CALUDE_integer_solution_system_l1472_147274

theorem integer_solution_system :
  ∀ A B C : ℤ,
  (A^2 - B^2 - C^2 = 1 ∧ B + C - A = 3) ↔
  ((A = 9 ∧ B = 8 ∧ C = 4) ∨
   (A = 9 ∧ B = 4 ∧ C = 8) ∨
   (A = -3 ∧ B = 2 ∧ C = -2) ∨
   (A = -3 ∧ B = -2 ∧ C = 2)) :=
by sorry


end NUMINAMATH_CALUDE_integer_solution_system_l1472_147274


namespace NUMINAMATH_CALUDE_equation_solution_unique_solution_l1472_147286

theorem equation_solution : ∃ (x : ℝ), x = 2 ∧ -2 * x + 4 = 0 := by
  sorry

-- Definitions of the given equations
def eq1 (x : ℝ) : Prop := 3 * x + 6 = 0
def eq2 (x : ℝ) : Prop := -2 * x + 4 = 0
def eq3 (x : ℝ) : Prop := (1 / 2) * x = 2
def eq4 (x : ℝ) : Prop := 2 * x + 4 = 0

-- Theorem stating that eq2 is the only equation satisfied by x = 2
theorem unique_solution :
  ∃! (i : Fin 4), (match i with
    | 0 => eq1
    | 1 => eq2
    | 2 => eq3
    | 3 => eq4) 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_unique_solution_l1472_147286


namespace NUMINAMATH_CALUDE_two_digit_number_remainder_l1472_147213

theorem two_digit_number_remainder (n : ℕ) : 
  10 ≤ n ∧ n < 100 →  -- n is a two-digit number
  n % 9 = 1 →         -- remainder when divided by 9 is 1
  n % 10 = 3 →        -- remainder when divided by 10 is 3
  n % 11 = 7 :=       -- remainder when divided by 11 is 7
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_remainder_l1472_147213


namespace NUMINAMATH_CALUDE_line_equation_proof_line_parameters_l1472_147248

/-- Given a line defined by (3, -4) · ((x, y) - (-2, 8)) = 0, prove that it can be expressed as y = (3/4)x + 9.5 with m = 3/4 and b = 9.5 -/
theorem line_equation_proof (x y : ℝ) :
  (3 * (x + 2) + (-4) * (y - 8) = 0) ↔ (y = (3 / 4) * x + (19 / 2)) :=
by sorry

/-- Prove that for the given line, m = 3/4 and b = 9.5 -/
theorem line_parameters :
  ∃ (m b : ℝ), m = 3 / 4 ∧ b = 19 / 2 ∧
  ∀ (x y : ℝ), (3 * (x + 2) + (-4) * (y - 8) = 0) ↔ (y = m * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_line_parameters_l1472_147248


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1472_147249

/-- Given a geometric sequence {a_n} with common ratio q = 1/2 and sum of first n terms S_n, 
    prove that S_3 / a_3 = 7 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Geometric sequence with common ratio 1/2
  (∀ n, S n = a 1 * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Sum formula
  S 3 / a 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1472_147249


namespace NUMINAMATH_CALUDE_probability_three_green_marbles_l1472_147217

/-- The probability of choosing exactly k successes in n trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The number of purple marbles -/
def purple_marbles : ℕ := 7

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The number of desired green marbles -/
def desired_green : ℕ := 3

/-- The probability of choosing a green marble in one trial -/
def prob_green : ℚ := green_marbles / total_marbles

theorem probability_three_green_marbles :
  binomial_probability num_trials desired_green prob_green = 8604112 / 15946875 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_marbles_l1472_147217


namespace NUMINAMATH_CALUDE_product_increase_theorem_l1472_147201

theorem product_increase_theorem :
  ∃ (a b c d e : ℕ), 
    (((a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3)) : ℤ) = 
    15 * (a * b * c * d * e) :=
by sorry

end NUMINAMATH_CALUDE_product_increase_theorem_l1472_147201


namespace NUMINAMATH_CALUDE_total_trip_time_l1472_147245

/-- The total trip time given the specified conditions -/
theorem total_trip_time : ∀ (v : ℝ),
  v > 0 →
  20 / v = 40 →
  80 / (4 * v) + 40 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_total_trip_time_l1472_147245


namespace NUMINAMATH_CALUDE_smallest_transformed_sum_l1472_147271

/-- The number of faces on a standard die -/
def standardDieFaces : ℕ := 6

/-- The sum we want to compare with -/
def targetSum : ℕ := 980

/-- A function to calculate the transformed sum given the number of dice -/
def transformedSum (n : ℕ) : ℤ := 5 * n - targetSum

/-- The proposition that proves the smallest possible value of S -/
theorem smallest_transformed_sum :
  ∃ (n : ℕ), 
    (n * standardDieFaces ≥ targetSum) ∧ 
    (∀ m : ℕ, m < n → m * standardDieFaces < targetSum) ∧
    (transformedSum n = 5) ∧
    (∀ k : ℕ, k < n → transformedSum k < 5) := by
  sorry

end NUMINAMATH_CALUDE_smallest_transformed_sum_l1472_147271


namespace NUMINAMATH_CALUDE_goldfish_equality_month_l1472_147233

theorem goldfish_equality_month : ∃ n : ℕ, n > 0 ∧ 3^(n+1) = 96 * 2^n ∧ ∀ m : ℕ, m > 0 ∧ m < n → 3^(m+1) ≠ 96 * 2^m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_month_l1472_147233


namespace NUMINAMATH_CALUDE_equal_intercept_line_proof_l1472_147208

/-- A line with equal intercepts on both axes passing through point (3,2) -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y = 5

theorem equal_intercept_line_proof :
  -- The line passes through point (3,2)
  equal_intercept_line 3 2 ∧
  -- The line has equal intercepts on both axes
  ∃ a : ℝ, a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a :=
by
  sorry

#check equal_intercept_line_proof

end NUMINAMATH_CALUDE_equal_intercept_line_proof_l1472_147208


namespace NUMINAMATH_CALUDE_temperature_difference_l1472_147228

def highest_temp : ℚ := 10
def lowest_temp : ℚ := -5

theorem temperature_difference :
  highest_temp - lowest_temp = 15 := by sorry

end NUMINAMATH_CALUDE_temperature_difference_l1472_147228


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_tournament_with_23_teams_l1472_147210

/-- In a single-elimination tournament, the number of games played is one less than the number of teams. -/
theorem single_elimination_tournament_games (n : ℕ) (n_pos : n > 0) :
  let teams := n
  let games := n - 1
  games = teams - 1 := by sorry

/-- For a tournament with 23 teams, 22 games are played. -/
theorem tournament_with_23_teams :
  let teams := 23
  let games := teams - 1
  games = 22 := by sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_tournament_with_23_teams_l1472_147210


namespace NUMINAMATH_CALUDE_common_roots_product_sum_of_a_b_c_l1472_147256

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10∛4 -/
theorem common_roots_product (C : ℝ) : 
  ∃ (u v w t : ℝ),
    (u^3 - 5*u + 20 = 0) ∧ 
    (v^3 - 5*v + 20 = 0) ∧ 
    (w^3 - 5*w + 20 = 0) ∧
    (u^3 + C*u^2 + 80 = 0) ∧ 
    (v^3 + C*v^2 + 80 = 0) ∧ 
    (t^3 + C*t^2 + 80 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 10 * Real.rpow 4 (1/3) :=
by sorry

/-- The sum of a, b, and c in the form a∛b where a=10, b=3, and c=4 is 17 -/
theorem sum_of_a_b_c : 10 + 3 + 4 = 17 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_sum_of_a_b_c_l1472_147256


namespace NUMINAMATH_CALUDE_fifth_root_of_102030201_l1472_147202

theorem fifth_root_of_102030201 : (102030201 : ℝ) ^ (1/5 : ℝ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_102030201_l1472_147202


namespace NUMINAMATH_CALUDE_anna_swept_ten_rooms_l1472_147260

/-- Represents the time in minutes for various chores -/
structure ChoreTime where
  sweepingPerRoom : ℕ
  washingPerDish : ℕ
  laundryPerLoad : ℕ

/-- Represents the chores assigned to Billy -/
structure BillyChores where
  laundryLoads : ℕ
  dishesToWash : ℕ

/-- Calculates the total time Billy spends on chores -/
def billyTotalTime (ct : ChoreTime) (bc : BillyChores) : ℕ :=
  bc.laundryLoads * ct.laundryPerLoad + bc.dishesToWash * ct.washingPerDish

/-- Theorem stating that Anna swept 10 rooms -/
theorem anna_swept_ten_rooms (ct : ChoreTime) (bc : BillyChores) 
    (h1 : ct.sweepingPerRoom = 3)
    (h2 : ct.washingPerDish = 2)
    (h3 : ct.laundryPerLoad = 9)
    (h4 : bc.laundryLoads = 2)
    (h5 : bc.dishesToWash = 6) :
    ∃ (rooms : ℕ), rooms * ct.sweepingPerRoom = billyTotalTime ct bc ∧ rooms = 10 := by
  sorry

end NUMINAMATH_CALUDE_anna_swept_ten_rooms_l1472_147260


namespace NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l1472_147258

/-- Given the conditions of the problem, prove that the ratio of the speed of the bike to the speed of the tractor is 2:1 -/
theorem bike_tractor_speed_ratio :
  ∀ (car_speed bike_speed tractor_speed : ℝ),
  car_speed = (9/5) * bike_speed →
  tractor_speed = 575 / 23 →
  car_speed = 540 / 6 →
  ∃ (k : ℝ), bike_speed = k * tractor_speed →
  bike_speed / tractor_speed = 2 := by
sorry

end NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l1472_147258


namespace NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l1472_147254

/-- The number of pages that can be copied for a given amount of money -/
def pages_copied (cost_per_2_pages : ℚ) (amount : ℚ) : ℚ :=
  (amount / cost_per_2_pages) * 2

/-- Theorem: Given that it costs 4 cents to copy 2 pages, 
    the number of pages that can be copied for $30 is 1500 -/
theorem pages_copied_for_30_dollars : 
  pages_copied (4/100) 30 = 1500 := by
  sorry

#eval pages_copied (4/100) 30

end NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l1472_147254


namespace NUMINAMATH_CALUDE_RS_length_l1472_147279

-- Define the triangle RFS
structure Triangle :=
  (R F S : ℝ × ℝ)

-- Define the given lengths
def FD : ℝ := 5
def DR : ℝ := 8
def FR : ℝ := 6
def FS : ℝ := 9

-- Define the angles
def angle_RFS (t : Triangle) : ℝ := sorry
def angle_FDR : ℝ := sorry

-- State the theorem
theorem RS_length (t : Triangle) :
  angle_RFS t = angle_FDR →
  FR = 6 →
  FS = 9 →
  ∃ (RS : ℝ), abs (RS - 10.25) < 0.01 := by sorry

end NUMINAMATH_CALUDE_RS_length_l1472_147279


namespace NUMINAMATH_CALUDE_item_pricing_and_profit_l1472_147282

/-- Represents the pricing and profit calculation for an item -/
theorem item_pricing_and_profit (a : ℝ) :
  let original_price := a * (1 + 0.2)
  let current_price := original_price * 0.9
  let profit_per_unit := current_price - a
  (current_price = 1.08 * a) ∧
  (1000 * profit_per_unit = 80 * a) := by
  sorry

end NUMINAMATH_CALUDE_item_pricing_and_profit_l1472_147282


namespace NUMINAMATH_CALUDE_smallest_no_inverse_mod_77_88_l1472_147214

theorem smallest_no_inverse_mod_77_88 : 
  ∀ a : ℕ, a > 0 → (Nat.gcd a 77 > 1 ∧ Nat.gcd a 88 > 1) → a ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_no_inverse_mod_77_88_l1472_147214


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1472_147268

theorem triangle_angle_sum (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
  a = 20 →  -- Smallest angle is 20 degrees
  b = 3 * a →  -- Middle angle is 3 times the smallest
  c = 5 * a →  -- Largest angle is 5 times the smallest
  a + b + c = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1472_147268


namespace NUMINAMATH_CALUDE_three_digit_sum_proof_l1472_147205

/-- Represents a three-digit number in the form xyz -/
def ThreeDigitNumber (x y z : Nat) : Nat :=
  100 * x + 10 * y + z

theorem three_digit_sum_proof (a b : Nat) :
  (ThreeDigitNumber 3 a 7) + 416 = (ThreeDigitNumber 7 b 3) ∧
  (ThreeDigitNumber 7 b 3) % 3 = 0 →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_proof_l1472_147205


namespace NUMINAMATH_CALUDE_gcd_51_119_l1472_147247

theorem gcd_51_119 : Nat.gcd 51 119 = 17 := by sorry

end NUMINAMATH_CALUDE_gcd_51_119_l1472_147247


namespace NUMINAMATH_CALUDE_problem_statement_l1472_147230

theorem problem_statement :
  ∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 →
    (2 * x^3 - 3 * x^2 + 1 ≥ 0) ∧
    ((2 / (1 + x^3) + 2 / (1 + y^3) + 2 / (1 + z^3) = 3) →
      ((1 - x) / (1 - x + x^2) + (1 - y) / (1 - y + y^2) + (1 - z) / (1 - z + z^2) ≥ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1472_147230


namespace NUMINAMATH_CALUDE_binomial_sum_theorem_l1472_147292

theorem binomial_sum_theorem :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_theorem_l1472_147292


namespace NUMINAMATH_CALUDE_perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l1472_147235

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def perfect_squares : Set ℕ := {n : ℕ | is_perfect_square n ∧ n > 0}

theorem perfect_squares_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ perfect_squares → b ∈ perfect_squares → (a * b) ∈ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ (a + b) ∉ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_subtraction :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ a > b ∧ (a - b) ∉ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_division :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ b ≠ 0 ∧ (a / b) ∉ perfect_squares :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l1472_147235


namespace NUMINAMATH_CALUDE_apple_percentage_after_removal_l1472_147226

/-- Calculates the percentage of apples in a bowl of fruit -/
def percentage_apples (apples : ℕ) (oranges : ℕ) : ℚ :=
  (apples : ℚ) / (apples + oranges : ℚ) * 100

/-- Proves that after removing 19 oranges from a bowl with 14 apples and 25 oranges,
    the percentage of apples is 70% -/
theorem apple_percentage_after_removal :
  let initial_apples : ℕ := 14
  let initial_oranges : ℕ := 25
  let removed_oranges : ℕ := 19
  let remaining_oranges : ℕ := initial_oranges - removed_oranges
  percentage_apples initial_apples remaining_oranges = 70 := by
sorry

end NUMINAMATH_CALUDE_apple_percentage_after_removal_l1472_147226


namespace NUMINAMATH_CALUDE_paul_sandwich_consumption_l1472_147290

def sandwiches_per_cycle : ℕ := 2 + 4 + 8

def study_days : ℕ := 6

def cycles : ℕ := study_days / 3

theorem paul_sandwich_consumption :
  cycles * sandwiches_per_cycle = 28 := by
  sorry

end NUMINAMATH_CALUDE_paul_sandwich_consumption_l1472_147290


namespace NUMINAMATH_CALUDE_segment_length_l1472_147232

/-- Given points P, Q, and R on line segment AB, prove that AB has length 48 -/
theorem segment_length (A B P Q R : ℝ) : 
  (0 < A) → (A < P) → (P < Q) → (Q < R) → (R < B) →  -- Points lie on AB in order
  (P - A) / (B - P) = 3 / 5 →                        -- P divides AB in ratio 3:5
  (Q - A) / (B - Q) = 5 / 7 →                        -- Q divides AB in ratio 5:7
  R - Q = 3 →                                        -- QR = 3
  R - P = 5 →                                        -- PR = 5
  B - A = 48 := by sorry

end NUMINAMATH_CALUDE_segment_length_l1472_147232


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l1472_147227

theorem complex_root_magnitude (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a) 
  (h3 : a < (n + 1 : ℝ) / (n - 1 : ℝ)) 
  (h4 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l1472_147227


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l1472_147272

/-- The total capacity of a Ferris wheel -/
def ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  num_seats * people_per_seat

/-- Theorem: The capacity of a Ferris wheel with 14 seats and 6 people per seat is 84 -/
theorem paradise_park_ferris_wheel_capacity :
  ferris_wheel_capacity 14 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l1472_147272


namespace NUMINAMATH_CALUDE_odd_function_property_l1472_147240

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def HasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem odd_function_property (f : ℝ → ℝ) :
  IsOdd f →
  IsIncreasingOn f 1 3 →
  HasMinimumOn f 1 3 7 →
  IsIncreasingOn f (-3) (-1) ∧ HasMaximumOn f (-3) (-1) (-7) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1472_147240


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_25_l1472_147234

theorem smallest_four_digit_divisible_by_25 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 → n ≥ 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_25_l1472_147234


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1472_147236

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) (h : geometric_sequence a) (h5 : a 5 = 4) : 
  a 2 * a 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1472_147236


namespace NUMINAMATH_CALUDE_logarithmic_scales_imply_ohms_law_l1472_147211

/-- Represents a point on the logarithmic scale for resistance, current, or voltage -/
structure LogPoint where
  value : ℝ
  coordinate : ℝ

/-- Represents the scales for resistance, current, and voltage -/
structure Circuit where
  resistance : LogPoint
  current : LogPoint
  voltage : LogPoint

/-- The relationship between the coordinates of resistance, current, and voltage -/
def coordinate_relation (c : Circuit) : Prop :=
  c.current.coordinate + c.voltage.coordinate = 2 * c.resistance.coordinate

/-- The relationship between resistance, current, and voltage values -/
def ohms_law (c : Circuit) : Prop :=
  c.voltage.value = c.current.value * c.resistance.value

/-- The logarithmic scale relationship for resistance -/
def resistance_scale (r : LogPoint) : Prop :=
  r.value = 10^(-2 * r.coordinate)

/-- The logarithmic scale relationship for current -/
def current_scale (i : LogPoint) : Prop :=
  i.value = 10^(i.coordinate)

/-- The logarithmic scale relationship for voltage -/
def voltage_scale (v : LogPoint) : Prop :=
  v.value = 10^(-v.coordinate)

/-- Theorem stating that the logarithmic scales and coordinate relation imply Ohm's law -/
theorem logarithmic_scales_imply_ohms_law (c : Circuit) :
  resistance_scale c.resistance →
  current_scale c.current →
  voltage_scale c.voltage →
  coordinate_relation c →
  ohms_law c :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_scales_imply_ohms_law_l1472_147211


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l1472_147218

theorem trigonometric_system_solution (x y z : ℝ) 
  (eq1 : Real.sin x + 2 * Real.sin (x + y + z) = 0)
  (eq2 : Real.sin y + 3 * Real.sin (x + y + z) = 0)
  (eq3 : Real.sin z + 4 * Real.sin (x + y + z) = 0) :
  ∃ (k1 k2 k3 : ℤ), x = k1 * Real.pi ∧ y = k2 * Real.pi ∧ z = k3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l1472_147218


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l1472_147206

theorem right_triangle_perimeter_equals_area (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = (1/2) * a * b →
  a + b - c = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l1472_147206


namespace NUMINAMATH_CALUDE_cat_to_dog_probability_l1472_147267

-- Define the probabilities for each machine
def prob_A : ℚ := 1/3
def prob_B : ℚ := 2/5
def prob_C : ℚ := 1/4

-- Define the probability of a cat remaining a cat after all machines
def prob_cat_total : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

-- The main theorem to prove
theorem cat_to_dog_probability :
  1 - prob_cat_total = 7/10 := by sorry

end NUMINAMATH_CALUDE_cat_to_dog_probability_l1472_147267


namespace NUMINAMATH_CALUDE_school_population_l1472_147259

theorem school_population (girls boys teachers : ℕ) 
  (h1 : girls = 315) 
  (h2 : boys = 309) 
  (h3 : teachers = 772) : 
  girls + boys + teachers = 1396 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l1472_147259


namespace NUMINAMATH_CALUDE_election_votes_l1472_147225

theorem election_votes (candidate1_percentage : ℚ) (candidate2_votes : ℕ) :
  candidate1_percentage = 60 / 100 →
  candidate2_votes = 240 →
  ∃ total_votes : ℕ,
    candidate1_percentage * total_votes = total_votes - candidate2_votes ∧
    total_votes = 600 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l1472_147225


namespace NUMINAMATH_CALUDE_square_difference_81_49_l1472_147215

theorem square_difference_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_81_49_l1472_147215


namespace NUMINAMATH_CALUDE_power_equation_solution_l1472_147275

theorem power_equation_solution : ∃ x : ℝ, (1/8 : ℝ) * 2^36 = 4^x ∧ x = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1472_147275


namespace NUMINAMATH_CALUDE_function_nonnegative_m_range_l1472_147287

theorem function_nonnegative_m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≥ 0) → -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_nonnegative_m_range_l1472_147287


namespace NUMINAMATH_CALUDE_john_needs_more_money_l1472_147281

theorem john_needs_more_money (total_needed : ℝ) (amount_has : ℝ) (h1 : total_needed = 2.50) (h2 : amount_has = 0.75) :
  total_needed - amount_has = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_more_money_l1472_147281


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l1472_147257

/-- Represents the number of students in each grade -/
structure Students where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios given in the problem -/
def ratio_9_10 : ℚ := 7 / 4
def ratio_9_11 : ℚ := 5 / 3

/-- The proposition that needs to be proved -/
theorem smallest_number_of_students :
  ∃ (s : Students),
    (s.ninth : ℚ) / s.tenth = ratio_9_10 ∧
    (s.ninth : ℚ) / s.eleventh = ratio_9_11 ∧
    s.ninth + s.tenth + s.eleventh = 76 ∧
    (∀ (t : Students),
      (t.ninth : ℚ) / t.tenth = ratio_9_10 →
      (t.ninth : ℚ) / t.eleventh = ratio_9_11 →
      t.ninth + t.tenth + t.eleventh ≥ 76) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l1472_147257


namespace NUMINAMATH_CALUDE_inequality_proof_l1472_147278

theorem inequality_proof (x y : ℝ) : 2 * (x^2 + y^2) - (x + y)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1472_147278


namespace NUMINAMATH_CALUDE_derivative_implies_limit_l1472_147295

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number
variable (x₀ : ℝ)

-- State the theorem
theorem derivative_implies_limit 
  (h₁ : HasDerivAt f (-2) x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |((f (x₀ - 1/2 * h) - f x₀) / h) - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_implies_limit_l1472_147295


namespace NUMINAMATH_CALUDE_jack_jill_water_fetching_l1472_147280

/-- A problem about Jack and Jill fetching water --/
theorem jack_jill_water_fetching :
  -- Tank capacity
  ∀ (tank_capacity : ℕ),
  -- Bucket capacity
  ∀ (bucket_capacity : ℕ),
  -- Jack's bucket carrying capacity
  ∀ (jack_buckets : ℕ),
  -- Jill's bucket carrying capacity
  ∀ (jill_buckets : ℕ),
  -- Number of trips Jill made
  ∀ (jill_trips : ℕ),
  -- Conditions
  tank_capacity = 600 →
  bucket_capacity = 5 →
  jack_buckets = 2 →
  jill_buckets = 1 →
  jill_trips = 30 →
  -- Conclusion: Jack's trips in the time Jill makes two trips
  ∃ (jack_trips : ℕ), jack_trips = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_jill_water_fetching_l1472_147280


namespace NUMINAMATH_CALUDE_square_problem_l1472_147250

/-- Square with side length 1200 -/
structure Square :=
  (side : ℝ)
  (is_1200 : side = 1200)

/-- Point on the side AB of the square -/
structure PointOnAB (S : Square) :=
  (x : ℝ)
  (on_side : 0 ≤ x ∧ x ≤ S.side)

theorem square_problem (S : Square) (G H : PointOnAB S)
  (h_order : G.x < H.x)
  (h_angle : Real.cos (Real.pi / 3) = (H.x - G.x) / 600)
  (h_dist : H.x - G.x = 600) :
  S.side - H.x = 300 + 100 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_square_problem_l1472_147250


namespace NUMINAMATH_CALUDE_absolute_value_equality_implication_not_always_true_l1472_147238

theorem absolute_value_equality_implication_not_always_true :
  ¬ (∀ a b : ℝ, |a| = |b| → a = b) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implication_not_always_true_l1472_147238


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1472_147253

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

-- Define the three possible functions
def ZeroFunction : ℝ → ℝ := λ _ => 0
def IdentityFunction : ℝ → ℝ := λ x => x
def NegativeIdentityFunction : ℝ → ℝ := λ x => -x

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (f = ZeroFunction ∨ f = IdentityFunction ∨ f = NegativeIdentityFunction) :=
by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1472_147253


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l1472_147289

/-- Represents an examination with given parameters -/
structure Examination where
  totalQuestions : Nat
  correctAnswers : Nat
  marksPerCorrect : Nat
  totalScore : Int

/-- Calculates the marks lost per wrong answer -/
def marksLostPerWrongAnswer (exam : Examination) : Rat :=
  let wrongAnswers := exam.totalQuestions - exam.correctAnswers
  let totalCorrectMarks := exam.correctAnswers * exam.marksPerCorrect
  let totalLostMarks := totalCorrectMarks - exam.totalScore
  totalLostMarks / wrongAnswers

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one : 
  let exam : Examination := {
    totalQuestions := 80,
    correctAnswers := 42,
    marksPerCorrect := 4,
    totalScore := 130
  }
  marksLostPerWrongAnswer exam = 1 := by
  sorry

end NUMINAMATH_CALUDE_marks_lost_is_one_l1472_147289


namespace NUMINAMATH_CALUDE_subtracted_value_l1472_147263

theorem subtracted_value (chosen_number : ℕ) (subtracted_value : ℕ) : 
  chosen_number = 990 →
  (chosen_number / 9 : ℚ) - subtracted_value = 10 →
  subtracted_value = 100 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l1472_147263


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1472_147244

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

theorem geometric_sequence_condition (a q : ℝ) (h : a > 0) :
  (∀ n : ℕ, geometric_sequence a q n = a * q ^ (n - 1)) →
  (geometric_sequence a q 1 < geometric_sequence a q 3 → geometric_sequence a q 3 < geometric_sequence a q 6) ∧
  ¬(geometric_sequence a q 1 < geometric_sequence a q 3 → geometric_sequence a q 3 < geometric_sequence a q 6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1472_147244


namespace NUMINAMATH_CALUDE_cosine_function_parameters_l1472_147246

theorem cosine_function_parameters (a b c d : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (4 * Real.pi / b = 4 * Real.pi) →
  d = 3 →
  (∃ x, a * Real.cos (b * x + c) + d = 8) →
  (∃ x, a * Real.cos (b * x + c) + d = -2) →
  a = 5 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_cosine_function_parameters_l1472_147246


namespace NUMINAMATH_CALUDE_inequality_solution_l1472_147219

theorem inequality_solution (a : ℝ) :
  4 ≤ a / (3 * a - 6) ∧ a / (3 * a - 6) > 12 → a < 72 / 35 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1472_147219


namespace NUMINAMATH_CALUDE_sum_even_product_not_necessarily_odd_l1472_147223

theorem sum_even_product_not_necessarily_odd :
  ∃ (a b : ℤ), Even (a + b) ∧ ¬Odd (a * b) := by sorry

end NUMINAMATH_CALUDE_sum_even_product_not_necessarily_odd_l1472_147223


namespace NUMINAMATH_CALUDE_not_necessarily_no_mass_infection_l1472_147284

/-- Represents the daily increase in suspected cases over 10 days -/
def DailyIncrease := Fin 10 → ℕ

/-- The sign of no mass infection -/
def NoMassInfection (d : DailyIncrease) : Prop :=
  ∀ i, d i ≤ 7

/-- The median of a DailyIncrease is 2 -/
def MedianIsTwo (d : DailyIncrease) : Prop :=
  ∃ (sorted : Fin 10 → ℕ), (∀ i j, i ≤ j → sorted i ≤ sorted j) ∧
    (∀ i, ∃ j, d j = sorted i) ∧
    sorted 4 = 2 ∧ sorted 5 = 2

/-- The mode of a DailyIncrease is 3 -/
def ModeIsThree (d : DailyIncrease) : Prop :=
  ∃ (count : ℕ → ℕ), (∀ n, count n = (Finset.univ.filter (λ i => d i = n)).card) ∧
    ∀ n, count 3 ≥ count n

theorem not_necessarily_no_mass_infection :
  ∃ d : DailyIncrease, MedianIsTwo d ∧ ModeIsThree d ∧ ¬NoMassInfection d :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_no_mass_infection_l1472_147284


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1472_147229

theorem geometric_series_first_term
  (a r : ℝ)
  (h_sum : a / (1 - r) = 30)
  (h_sum_squares : a^2 / (1 - r^2) = 120)
  (h_convergent : |r| < 1) :
  a = 120 / 17 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1472_147229


namespace NUMINAMATH_CALUDE_toy_pile_ratio_l1472_147243

theorem toy_pile_ratio : 
  let total_toys : ℕ := 120
  let larger_pile : ℕ := 80
  let smaller_pile : ℕ := total_toys - larger_pile
  (larger_pile : ℚ) / smaller_pile = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_toy_pile_ratio_l1472_147243


namespace NUMINAMATH_CALUDE_henrys_age_l1472_147269

/-- Given that the sum of Henry and Jill's present ages is 48, and 9 years ago Henry was twice the age of Jill, 
    prove that Henry's present age is 29 years. -/
theorem henrys_age (henry_age jill_age : ℕ) 
  (sum_condition : henry_age + jill_age = 48)
  (past_condition : henry_age - 9 = 2 * (jill_age - 9)) : 
  henry_age = 29 := by
  sorry

end NUMINAMATH_CALUDE_henrys_age_l1472_147269


namespace NUMINAMATH_CALUDE_distance_difference_l1472_147241

/-- The width of the streets in Tranquility Town -/
def street_width : ℝ := 30

/-- The length of the rectangular block -/
def block_length : ℝ := 500

/-- The width of the rectangular block -/
def block_width : ℝ := 300

/-- The perimeter of Alice's path -/
def alice_perimeter : ℝ := 2 * ((block_length + street_width) + (block_width + street_width))

/-- The perimeter of Bob's path -/
def bob_perimeter : ℝ := 2 * ((block_length + 2 * street_width) + (block_width + 2 * street_width))

/-- The theorem stating the difference in distance walked -/
theorem distance_difference : bob_perimeter - alice_perimeter = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1472_147241


namespace NUMINAMATH_CALUDE_computer_price_increase_l1472_147207

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : 
  (351 - d) / d * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1472_147207


namespace NUMINAMATH_CALUDE_ten_special_divisors_l1472_147262

theorem ten_special_divisors : ∃ (n : ℕ), 
  n > 1 ∧ 
  (∀ d : ℕ, d > 1 → d ∣ n → ∃ (a r : ℕ), r > 1 ∧ d = a^r + 1) ∧
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_ten_special_divisors_l1472_147262


namespace NUMINAMATH_CALUDE_determinant_equality_l1472_147283

theorem determinant_equality (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 7 →
  Matrix.det !![p + r, q + s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1472_147283


namespace NUMINAMATH_CALUDE_product_digits_sum_base9_l1472_147293

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base 9 number --/
def sumOfDigitsBase9 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digits_sum_base9 :
  let a := 36
  let b := 21
  let product := (base9ToDecimal a) * (base9ToDecimal b)
  sumOfDigitsBase9 (decimalToBase9 product) = 19 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base9_l1472_147293


namespace NUMINAMATH_CALUDE_snowboard_final_price_l1472_147285

/-- 
Given a snowboard with an original price and two successive discounts,
calculate the final price after both discounts are applied.
-/
theorem snowboard_final_price 
  (original_price : ℝ)
  (friday_discount : ℝ)
  (monday_discount : ℝ)
  (h1 : original_price = 200)
  (h2 : friday_discount = 0.4)
  (h3 : monday_discount = 0.25) :
  original_price * (1 - friday_discount) * (1 - monday_discount) = 90 :=
by sorry

#check snowboard_final_price

end NUMINAMATH_CALUDE_snowboard_final_price_l1472_147285


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1472_147220

theorem fraction_evaluation : (15 - 3^2) / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1472_147220


namespace NUMINAMATH_CALUDE_job_completion_theorem_l1472_147294

/-- The number of days it takes the initial group of machines to finish the job -/
def initial_days : ℕ := 40

/-- The number of additional machines added -/
def additional_machines : ℕ := 4

/-- The number of days it takes after adding more machines -/
def reduced_days : ℕ := 30

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 16

theorem job_completion_theorem :
  (initial_machines : ℚ) / initial_days = (initial_machines + additional_machines : ℚ) / reduced_days :=
by sorry

#check job_completion_theorem

end NUMINAMATH_CALUDE_job_completion_theorem_l1472_147294


namespace NUMINAMATH_CALUDE_average_speed_problem_l1472_147261

/-- Given a distance of 1800 meters and a time of 30 minutes, 
    prove that the average speed is 1 meter per second. -/
theorem average_speed_problem (distance : ℝ) (time_minutes : ℝ) :
  distance = 1800 ∧ time_minutes = 30 →
  (distance / (time_minutes * 60)) = 1 := by
sorry

end NUMINAMATH_CALUDE_average_speed_problem_l1472_147261


namespace NUMINAMATH_CALUDE_intersection_point_implies_sum_of_intercepts_l1472_147255

/-- Given two lines that intersect at (2,3), prove their y-intercepts sum to 10/3 -/
theorem intersection_point_implies_sum_of_intercepts :
  ∀ (a b : ℚ),
  (2 : ℚ) = (1/3 : ℚ) * 3 + a →
  (3 : ℚ) = (1/3 : ℚ) * 2 + b →
  a + b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_implies_sum_of_intercepts_l1472_147255


namespace NUMINAMATH_CALUDE_bookshelf_length_is_24_l1472_147266

/-- The length of one span in centimeters -/
def span_length : ℝ := 12

/-- The number of spans in the shorter side of the bookshelf -/
def bookshelf_spans : ℝ := 2

/-- The length of the shorter side of the bookshelf in centimeters -/
def bookshelf_length : ℝ := span_length * bookshelf_spans

theorem bookshelf_length_is_24 : bookshelf_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_length_is_24_l1472_147266


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1472_147291

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1472_147291


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1472_147216

theorem trigonometric_identity : 
  (Real.cos (68 * π / 180) * Real.cos (8 * π / 180) - Real.cos (82 * π / 180) * Real.cos (22 * π / 180)) /
  (Real.cos (53 * π / 180) * Real.cos (23 * π / 180) - Real.cos (67 * π / 180) * Real.cos (37 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1472_147216


namespace NUMINAMATH_CALUDE_function_value_theorem_l1472_147204

/-- Given a function f(x) = √(-x² + bx + c) with domain D, 
    and for any x in D, f(-1) ≤ f(x) ≤ f(1), 
    prove that b · c + f(3) = 6 -/
theorem function_value_theorem (b c : ℝ) (D : Set ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x ∈ D, f x = Real.sqrt (-x^2 + b*x + c))
    (h2 : ∀ x ∈ D, f (-1) ≤ f x ∧ f x ≤ f 1) :
    b * c + f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_theorem_l1472_147204


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1472_147264

theorem chocolate_bars_count (large_box small_boxes chocolate_bars_per_small_box : ℕ) 
  (h1 : small_boxes = 21)
  (h2 : chocolate_bars_per_small_box = 25)
  (h3 : large_box = small_boxes * chocolate_bars_per_small_box) :
  large_box = 525 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1472_147264


namespace NUMINAMATH_CALUDE_exotic_fruit_distribution_l1472_147203

theorem exotic_fruit_distribution (eldest_fruits second_fruits third_fruits : ℕ) 
  (gold_to_eldest gold_to_second : ℕ) :
  eldest_fruits = 2 * second_fruits / 3 →
  third_fruits = 0 →
  gold_to_eldest + gold_to_second = 180 →
  eldest_fruits - gold_to_eldest / (180 / (gold_to_eldest + gold_to_second)) = 
    second_fruits - gold_to_second / (180 / (gold_to_eldest + gold_to_second)) →
  eldest_fruits - gold_to_eldest / (180 / (gold_to_eldest + gold_to_second)) = 
    (gold_to_eldest + gold_to_second) / (180 / (gold_to_eldest + gold_to_second)) →
  gold_to_second = 144 :=
by sorry

end NUMINAMATH_CALUDE_exotic_fruit_distribution_l1472_147203


namespace NUMINAMATH_CALUDE_initial_men_count_l1472_147296

/-- Given a piece of work that can be completed by some number of men in 25 hours,
    or by 12 men in 75 hours, prove that the initial number of men is 36. -/
theorem initial_men_count : ℕ :=
  let initial_time : ℕ := 25
  let new_men_count : ℕ := 12
  let new_time : ℕ := 75
  36

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l1472_147296


namespace NUMINAMATH_CALUDE_continuity_point_sum_l1472_147288

theorem continuity_point_sum (g : ℝ → ℝ) : 
  (∃ m₁ m₂ : ℝ, 
    (∀ x < m₁, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₁, g x = 3*x + 6) ∧
    (∀ x < m₂, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₂, g x = 3*x + 6) ∧
    (m₁^2 + 4 = 3*m₁ + 6) ∧
    (m₂^2 + 4 = 3*m₂ + 6) ∧
    (m₁ ≠ m₂)) →
  (∃ m₁ m₂ : ℝ, m₁ + m₂ = 3 ∧ 
    (∀ x < m₁, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₁, g x = 3*x + 6) ∧
    (∀ x < m₂, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₂, g x = 3*x + 6) ∧
    (m₁^2 + 4 = 3*m₁ + 6) ∧
    (m₂^2 + 4 = 3*m₂ + 6) ∧
    (m₁ ≠ m₂)) :=
by sorry

end NUMINAMATH_CALUDE_continuity_point_sum_l1472_147288


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1472_147222

-- Define the inequality function
def f (x : ℝ) : ℝ := |x - 5| + |x + 3|

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -4 ∨ x ≥ 6}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 10} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1472_147222


namespace NUMINAMATH_CALUDE_ellipse_properties_l1472_147239

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_max_area : a * b = 2 * Real.sqrt 3

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- The fixed point property -/
def fixed_point_property (e : Ellipse) : Prop :=
  ∃ D : ℝ × ℝ, 
    D.2 = 0 ∧ 
    D.1 = -11/8 ∧
    ∀ M N : ℝ × ℝ,
      (M.1^2/e.a^2 + M.2^2/e.b^2 = 1) →
      (N.1^2/e.a^2 + N.2^2/e.b^2 = 1) →
      (∃ t : ℝ, M.1 = t * M.2 - 1 ∧ N.1 = t * N.2 - 1) →
      ((M.1 - D.1) * (N.1 - D.1) + (M.2 - D.2) * (N.2 - D.2) = -135/64)

theorem ellipse_properties (e : Ellipse) : 
  standard_form e ∧ fixed_point_property e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1472_147239


namespace NUMINAMATH_CALUDE_certain_number_is_100_l1472_147224

theorem certain_number_is_100 : ∃! x : ℝ, ((x / 4) + 25) * 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_100_l1472_147224


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1472_147277

theorem slope_angle_of_line (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ θ = π - Real.arctan (a / b) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1472_147277


namespace NUMINAMATH_CALUDE_book_pages_l1472_147231

theorem book_pages : 
  ∀ (P : ℕ), 
  (7 : ℚ) / 13 * P + (5 : ℚ) / 9 * ((6 : ℚ) / 13 * P) + 96 = P → 
  P = 468 :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_l1472_147231


namespace NUMINAMATH_CALUDE_cubes_fill_box_completely_l1472_147212

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : Cube) : ℕ :=
  cube.sideLength ^ 3

/-- Calculates the number of cubes that can fit along each dimension of the box -/
def cubesPerDimension (box : BoxDimensions) (cube : Cube) : ℕ × ℕ × ℕ :=
  (box.length / cube.sideLength, box.width / cube.sideLength, box.height / cube.sideLength)

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) (cube : Cube) : ℕ :=
  let (l, w, h) := cubesPerDimension box cube
  l * w * h

/-- Calculates the total volume occupied by the cubes in the box -/
def totalCubeVolume (box : BoxDimensions) (cube : Cube) : ℕ :=
  totalCubes box cube * cubeVolume cube

/-- Theorem: The volume occupied by 4-inch cubes in an 8x4x12 inch box is 100% of the box's volume -/
theorem cubes_fill_box_completely (box : BoxDimensions) (cube : Cube) :
  box.length = 8 ∧ box.width = 4 ∧ box.height = 12 ∧ cube.sideLength = 4 →
  totalCubeVolume box cube = boxVolume box := by
  sorry

#check cubes_fill_box_completely

end NUMINAMATH_CALUDE_cubes_fill_box_completely_l1472_147212


namespace NUMINAMATH_CALUDE_inscribed_rectangle_length_l1472_147265

/-- Right triangle PQR with inscribed rectangle ABCD -/
structure InscribedRectangle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Length of rectangle ABCD (parallel to PR) -/
  length : ℝ
  /-- Height of rectangle ABCD (parallel to PQ) -/
  height : ℝ
  /-- PQR is a right triangle -/
  is_right_triangle : pq ^ 2 + qr ^ 2 = pr ^ 2
  /-- Height is half the length -/
  height_half_length : height = length / 2
  /-- Rectangle fits in triangle -/
  fits_in_triangle : height ≤ pq ∧ length ≤ pr ∧ (pr - length) / (qr - height) = height / pq

/-- The length of the inscribed rectangle is 7.5 -/
theorem inscribed_rectangle_length (rect : InscribedRectangle) 
  (h_pq : rect.pq = 5) (h_qr : rect.qr = 12) (h_pr : rect.pr = 13) : 
  rect.length = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_length_l1472_147265


namespace NUMINAMATH_CALUDE_multiply_fractions_equals_thirty_l1472_147299

theorem multiply_fractions_equals_thirty : 15 * (1 / 17) * 34 = 30 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_equals_thirty_l1472_147299


namespace NUMINAMATH_CALUDE_square_area_proof_l1472_147200

-- Define the length of the longer side of the smaller rectangle
def longer_side : ℝ := 6

-- Define the ratio between longer and shorter sides
def ratio : ℝ := 3

-- Define the area of the square WXYZ
def square_area : ℝ := 144

-- Theorem statement
theorem square_area_proof :
  let shorter_side := longer_side / ratio
  let square_side := 2 * longer_side
  square_side ^ 2 = square_area :=
by sorry

end NUMINAMATH_CALUDE_square_area_proof_l1472_147200


namespace NUMINAMATH_CALUDE_root_transformation_l1472_147252

-- Define the original quadratic equation
def original_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the transformed equation
def transformed_equation (a b c x : ℝ) : Prop := a * (x - 1)^2 + b * (x - 1) + c = 0

-- Theorem statement
theorem root_transformation (a b c : ℝ) :
  (original_equation a b c (-1) ∧ original_equation a b c 2) →
  (transformed_equation a b c 0 ∧ transformed_equation a b c 3) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1472_147252


namespace NUMINAMATH_CALUDE_smallest_nonnegative_congruence_l1472_147221

theorem smallest_nonnegative_congruence :
  ∃ n : ℕ, n < 7 ∧ -2222 ≡ n [ZMOD 7] ∧ ∀ m : ℕ, m < 7 → -2222 ≡ m [ZMOD 7] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_congruence_l1472_147221


namespace NUMINAMATH_CALUDE_single_digit_square_equals_5929_l1472_147242

theorem single_digit_square_equals_5929 (A : ℕ) : 
  A < 10 → (10 * A + A) * (10 * A + A) = 5929 → A = 7 := by
sorry

end NUMINAMATH_CALUDE_single_digit_square_equals_5929_l1472_147242


namespace NUMINAMATH_CALUDE_ap_eq_aq_l1472_147270

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the Euclidean plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of an acute-angled triangle -/
def isAcuteAngled (A B C : Point) : Prop :=
  sorry

/-- Definition of a circle with a given diameter -/
def circleWithDiameter (P Q : Point) : Circle :=
  sorry

/-- Definition of the intersection of a line and a circle -/
def lineCircleIntersection (l : Line) (c : Circle) : Set Point :=
  sorry

theorem ap_eq_aq 
  (A B C : Point)
  (h_acute : isAcuteAngled A B C)
  (circle_AC : Circle)
  (circle_AB : Circle)
  (h_circle_AC : circle_AC = circleWithDiameter A C)
  (h_circle_AB : circle_AB = circleWithDiameter A B)
  (F : Point)
  (h_F : F ∈ lineCircleIntersection (Line.mk 0 1 0) circle_AC)
  (E : Point)
  (h_E : E ∈ lineCircleIntersection (Line.mk 1 0 0) circle_AB)
  (BE CF : Line)
  (P : Point)
  (h_P : P ∈ lineCircleIntersection BE circle_AC)
  (Q : Point)
  (h_Q : Q ∈ lineCircleIntersection CF circle_AB) :
  (A.x - P.x)^2 + (A.y - P.y)^2 = (A.x - Q.x)^2 + (A.y - Q.y)^2 :=
sorry

end NUMINAMATH_CALUDE_ap_eq_aq_l1472_147270


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l1472_147297

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse (T : Triangle) :
  ¬ (∃ i j : Fin 3, i ≠ j ∧ is_obtuse (T.angles i) ∧ is_obtuse (T.angles j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l1472_147297


namespace NUMINAMATH_CALUDE_lisa_spoons_l1472_147276

/-- The total number of spoons Lisa has -/
def total_spoons (num_children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) 
                 (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Proof that Lisa has 39 spoons in total -/
theorem lisa_spoons : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoons_l1472_147276


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l1472_147298

/-- Represents the state of the candy game -/
structure GameState where
  pile1 : Nat
  pile2 : Nat

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (newState : GameState) : Prop :=
  (newState.pile1 = state.pile1 ∧ newState.pile2 < state.pile2 ∧ (state.pile2 - newState.pile2) % state.pile1 = 0) ∨
  (newState.pile2 = state.pile2 ∧ newState.pile1 < state.pile1 ∧ (state.pile1 - newState.pile1) % state.pile2 = 0)

/-- Defines a winning state -/
def WinningState (state : GameState) : Prop :=
  state.pile1 = 0 ∨ state.pile2 = 0

/-- Theorem stating that there exists a winning strategy -/
theorem exists_winning_strategy :
  ∃ (strategy : GameState → GameState),
    let initialState := GameState.mk 1000 2357
    ∀ (state : GameState),
      state = initialState ∨ (∃ (prevState : GameState), ValidMove prevState state) →
      WinningState state ∨ (ValidMove state (strategy state) ∧ 
        ¬∃ (nextState : GameState), ValidMove (strategy state) nextState ∧ ¬WinningState nextState) :=
sorry


end NUMINAMATH_CALUDE_exists_winning_strategy_l1472_147298
