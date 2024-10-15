import Mathlib

namespace NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l2313_231369

theorem divisible_by_seven_last_digit :
  ∀ d : ℕ, d < 10 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l2313_231369


namespace NUMINAMATH_CALUDE_line_through_point_l2313_231320

/-- Given a line with equation 5x + by + 2 = d passing through the point (40, 5),
    prove that d = 202 + 5b -/
theorem line_through_point (b : ℝ) : 
  ∃ (d : ℝ), 5 * 40 + b * 5 + 2 = d ∧ d = 202 + 5 * b := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2313_231320


namespace NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l2313_231389

def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_angle_and_perimeter 
  (a b c : ℝ) (A B C : ℝ) :
  triangle a b c →
  a > 2 →
  b - c = 1 →
  Real.sqrt 3 * a * Real.cos C = c * Real.sin A →
  (C = Real.pi / 3 ∧
   ∃ (p : ℝ), p = a + b + c ∧ p ≥ 9 + 6 * Real.sqrt 2 ∧
   ∀ (q : ℝ), q = a + b + c → q ≥ p) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l2313_231389


namespace NUMINAMATH_CALUDE_range_of_m_inequality_for_nonzero_x_l2313_231314

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m|

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) : 
  (f m 1 + f m (-2) ≥ 5) ↔ (m ≤ -2 ∨ m ≥ 3) := by sorry

-- Theorem 2: Inequality for non-zero x
theorem inequality_for_nonzero_x (m : ℝ) (x : ℝ) (h : x ≠ 0) : 
  f m (1/x) + f m (-x) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_inequality_for_nonzero_x_l2313_231314


namespace NUMINAMATH_CALUDE_solve_equation_l2313_231339

theorem solve_equation (x : ℝ) : 3 * x = (36 - x) + 16 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2313_231339


namespace NUMINAMATH_CALUDE_cube_volume_from_side_area_l2313_231351

theorem cube_volume_from_side_area (side_area : ℝ) (volume : ℝ) :
  side_area = 64 →
  volume = (side_area ^ (1/2 : ℝ)) ^ 3 →
  volume = 512 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_side_area_l2313_231351


namespace NUMINAMATH_CALUDE_investment_problem_l2313_231365

theorem investment_problem (total_interest desired_interest fixed_investment fixed_rate variable_rate : ℝ) :
  desired_interest = 980 →
  fixed_investment = 6000 →
  fixed_rate = 0.09 →
  variable_rate = 0.11 →
  total_interest = fixed_rate * fixed_investment + variable_rate * (total_interest - fixed_investment) →
  total_interest = 10000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2313_231365


namespace NUMINAMATH_CALUDE_square_vertices_not_on_arithmetic_circles_l2313_231316

theorem square_vertices_not_on_arithmetic_circles : ¬∃ (a d : ℝ), a > 0 ∧ d > 0 ∧
  ((a ^ 2 + (a + d) ^ 2 = (a + 2*d) ^ 2 + (a + 3*d) ^ 2) ∨
   (a ^ 2 + (a + 2*d) ^ 2 = (a + d) ^ 2 + (a + 3*d) ^ 2) ∨
   ((a + d) ^ 2 + (a + 2*d) ^ 2 = a ^ 2 + (a + 3*d) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_square_vertices_not_on_arithmetic_circles_l2313_231316


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2313_231390

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2313_231390


namespace NUMINAMATH_CALUDE_gift_wrap_sales_l2313_231370

/-- Proves that the total number of gift wrap rolls sold is 480 given the specified conditions -/
theorem gift_wrap_sales (solid_price print_price total_amount print_rolls : ℚ)
  (h1 : solid_price = 4)
  (h2 : print_price = 6)
  (h3 : total_amount = 2340)
  (h4 : print_rolls = 210)
  (h5 : ∃ solid_rolls : ℚ, solid_price * solid_rolls + print_price * print_rolls = total_amount) :
  ∃ total_rolls : ℚ, total_rolls = 480 ∧ 
    ∃ solid_rolls : ℚ, total_rolls = solid_rolls + print_rolls ∧
    solid_price * solid_rolls + print_price * print_rolls = total_amount := by
  sorry


end NUMINAMATH_CALUDE_gift_wrap_sales_l2313_231370


namespace NUMINAMATH_CALUDE_three_people_eight_seats_l2313_231384

/-- The number of ways 3 people can sit in 8 seats with empty seats between them -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  let available_positions := total_seats - 2 * people + 1
  (available_positions.choose people) * (Nat.factorial people)

/-- Theorem stating that there are 24 ways for 3 people to sit in 8 seats with empty seats between them -/
theorem three_people_eight_seats : seating_arrangements 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_people_eight_seats_l2313_231384


namespace NUMINAMATH_CALUDE_correct_transformation_l2313_231300

theorem correct_transformation (x : ℝ) : 2*x = 3*x + 4 → 2*x - 3*x = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l2313_231300


namespace NUMINAMATH_CALUDE_elsas_final_marbles_l2313_231361

/-- Calculates the number of marbles Elsa has at the end of the day -/
def elsas_marbles : ℕ :=
  let initial := 150
  let after_breakfast := initial - (initial * 5 / 100)
  let after_lunch := after_breakfast - (after_breakfast * 2 / 5)
  let after_mom_gift := after_lunch + 25
  let after_susie_return := after_mom_gift + (after_breakfast * 2 / 5 * 150 / 100)
  let peter_exchange := 15
  let elsa_gives := peter_exchange * 3 / 5
  let elsa_receives := peter_exchange * 2 / 5
  let after_peter := after_susie_return - elsa_gives + elsa_receives
  let final := after_peter - (after_peter / 4)
  final

theorem elsas_final_marbles :
  elsas_marbles = 145 := by
  sorry

end NUMINAMATH_CALUDE_elsas_final_marbles_l2313_231361


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2313_231394

/-- Calculates the cost of a taxi ride given the fixed cost, per-mile cost, and distance traveled. -/
def taxi_cost (fixed_cost : ℝ) (per_mile_cost : ℝ) (distance : ℝ) : ℝ :=
  fixed_cost + per_mile_cost * distance

/-- Theorem: The cost of a 10-mile taxi ride with a $2.00 fixed cost and $0.30 per-mile cost is $5.00. -/
theorem ten_mile_taxi_cost : 
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2313_231394


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2313_231332

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) :
  x + y = -5 ∨ x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2313_231332


namespace NUMINAMATH_CALUDE_keychain_arrangements_l2313_231379

/-- The number of keys on the keychain -/
def total_keys : ℕ := 6

/-- The number of keys that must be adjacent -/
def adjacent_keys : ℕ := 3

/-- The number of distinct arrangements of the adjacent keys -/
def adjacent_arrangements : ℕ := Nat.factorial adjacent_keys

/-- The number of distinct arrangements of the remaining groups (adjacent group + other keys) -/
def group_arrangements : ℕ := Nat.factorial (total_keys - adjacent_keys + 1 - 1)

/-- The total number of distinct arrangements -/
def total_arrangements : ℕ := adjacent_arrangements * group_arrangements

theorem keychain_arrangements :
  total_arrangements = 36 :=
sorry

end NUMINAMATH_CALUDE_keychain_arrangements_l2313_231379


namespace NUMINAMATH_CALUDE_common_chord_equation_l2313_231308

/-- Given two circles in polar coordinates:
    1. ρ = r
    2. ρ = -2r * sin(θ + π/4)
    where r > 0, the equation of the line on which their common chord lies
    is √2 * ρ * (sin θ + cos θ) = -r -/
theorem common_chord_equation (r : ℝ) (h : r > 0) :
  ∃ (ρ θ : ℝ), (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
    Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2313_231308


namespace NUMINAMATH_CALUDE_g_properties_l2313_231367

noncomputable def g (x : ℝ) : ℝ :=
  (4 * Real.sin x ^ 4 + 5 * Real.cos x ^ 2) / (4 * Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_properties :
  (∀ k : ℤ, g (π/4 + k*π) = 7/5 ∧ g (π/3 + 2*k*π) = 7/5 ∧ g (-π/3 + 2*k*π) = 7/5) ∧
  (∀ x : ℝ, g x ≤ 71/55) ∧
  (∀ x : ℝ, g x ≥ 5/4) ∧
  (∃ x : ℝ, g x = 71/55) ∧
  (∃ x : ℝ, g x = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_g_properties_l2313_231367


namespace NUMINAMATH_CALUDE_exists_quadratic_polynomial_with_constant_negative_two_l2313_231362

/-- A quadratic polynomial in x and y with constant term -2 -/
def quadratic_polynomial (x y : ℝ) : ℝ := 15 * x^2 - y - 2

/-- Theorem stating the existence of a quadratic polynomial in x and y with constant term -2 -/
theorem exists_quadratic_polynomial_with_constant_negative_two :
  ∃ (f : ℝ → ℝ → ℝ), (∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    f x y = a * x^2 + b * x * y + c * y^2 + d * x + e * y - 2) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_polynomial_with_constant_negative_two_l2313_231362


namespace NUMINAMATH_CALUDE_problem_statement_l2313_231340

theorem problem_statement : ∃ y : ℝ, (8000 * 6000 : ℝ) = 480 * (10 ^ y) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2313_231340


namespace NUMINAMATH_CALUDE_average_temperature_calculation_l2313_231391

/-- Given the average temperature for four consecutive days and the temperatures of the first and last days, calculate the average temperature for the last four days. -/
theorem average_temperature_calculation 
  (temp_mon : ℝ) 
  (temp_fri : ℝ) 
  (avg_mon_to_thu : ℝ) 
  (h1 : temp_mon = 41)
  (h2 : temp_fri = 33)
  (h3 : avg_mon_to_thu = 48) :
  (4 * avg_mon_to_thu - temp_mon + temp_fri) / 4 = 46 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_calculation_l2313_231391


namespace NUMINAMATH_CALUDE_bakers_remaining_cakes_l2313_231383

/-- Calculates the number of remaining cakes for a baker --/
def remaining_cakes (initial : ℕ) (additional : ℕ) (sold : ℕ) : ℕ :=
  initial + additional - sold

/-- Theorem: The baker's remaining cakes is 67 --/
theorem bakers_remaining_cakes :
  remaining_cakes 62 149 144 = 67 := by
  sorry

end NUMINAMATH_CALUDE_bakers_remaining_cakes_l2313_231383


namespace NUMINAMATH_CALUDE_congruence_problem_l2313_231321

theorem congruence_problem (x : ℤ) 
  (h1 : (2 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (4 + x) % (3^3) = 2^2 % (3^3))
  (h3 : (6 + x) % (5^3) = 7^2 % (5^3)) :
  x % 120 = 103 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l2313_231321


namespace NUMINAMATH_CALUDE_business_profit_share_l2313_231324

/-- Calculates the profit share of a partner given the total capital, partner's capital, and total profit -/
def profitShare (totalCapital : ℚ) (partnerCapital : ℚ) (totalProfit : ℚ) : ℚ :=
  (partnerCapital / totalCapital) * totalProfit

theorem business_profit_share 
  (capitalA capitalB capitalC : ℚ)
  (profitDifferenceAC : ℚ)
  (h1 : capitalA = 8000)
  (h2 : capitalB = 10000)
  (h3 : capitalC = 12000)
  (h4 : profitDifferenceAC = 760) :
  ∃ (totalProfit : ℚ), 
    profitShare (capitalA + capitalB + capitalC) capitalB totalProfit = 1900 :=
by
  sorry

#check business_profit_share

end NUMINAMATH_CALUDE_business_profit_share_l2313_231324


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l2313_231382

/-- The distance between two vehicles after a given time, given their speeds and initial positions. -/
def distanceBetweenVehicles (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

theorem distance_after_three_minutes :
  let truckSpeed : ℝ := 65
  let carSpeed : ℝ := 85
  let time : ℝ := 3 / 60
  distanceBetweenVehicles truckSpeed carSpeed time = 1 := by
  sorry

#check distance_after_three_minutes

end NUMINAMATH_CALUDE_distance_after_three_minutes_l2313_231382


namespace NUMINAMATH_CALUDE_interest_discount_sum_l2313_231331

/-- Given a sum, rate, and time, if the simple interest is 85 and the true discount is 75, then the sum is 637.5 -/
theorem interest_discount_sum (P r t : ℝ) : 
  (P * r * t / 100 = 85) → 
  (P * r * t / (100 + r * t) = 75) → 
  P = 637.5 := by
sorry

end NUMINAMATH_CALUDE_interest_discount_sum_l2313_231331


namespace NUMINAMATH_CALUDE_missile_interception_time_l2313_231368

/-- The time taken for a missile to intercept a plane -/
theorem missile_interception_time 
  (r : ℝ) -- radius of the circular path
  (v : ℝ) -- speed of both the plane and the missile
  (h : r = 10 ∧ v = 1000) -- specific values given in the problem
  : (r * π) / (2 * v) = π / 200 := by
  sorry

#check missile_interception_time

end NUMINAMATH_CALUDE_missile_interception_time_l2313_231368


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l2313_231355

/-- For any real number p > 1, the minimum value of x + y, where x and y satisfy the equation
    (x + √(1 + x²))(y + √(1 + y²)) = p, is (p - 1) / √p. -/
theorem min_sum_with_constraint (p : ℝ) (hp : p > 1) :
  (∃ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p) →
  (∀ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p → 
    x + y ≥ (p - 1) / Real.sqrt p) ∧
  (∃ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p ∧
    x + y = (p - 1) / Real.sqrt p) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l2313_231355


namespace NUMINAMATH_CALUDE_flensburgian_iff_even_l2313_231344

/-- A system of equations is Flensburgian if there exists a variable that is always greater than the others for all pairwise different solutions. -/
def isFlensburgian (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ i : Fin 3, ∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x → f x y z →
    match i with
    | 0 => x > y ∧ x > z
    | 1 => y > x ∧ y > z
    | 2 => z > x ∧ z > y

/-- The system of equations for the Flensburgian problem. -/
def flensburgSystem (n : ℕ) (a b c : ℝ) : Prop :=
  a^n + b = a ∧ c^(n+1) + b^2 = a*b

/-- The main theorem stating that the system is Flensburgian if and only if n is even. -/
theorem flensburgian_iff_even (n : ℕ) (h : n ≥ 2) :
  isFlensburgian (flensburgSystem n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_flensburgian_iff_even_l2313_231344


namespace NUMINAMATH_CALUDE_john_mean_score_l2313_231315

def john_scores : List ℝ := [86, 90, 88, 82, 91]

theorem john_mean_score : (john_scores.sum / john_scores.length : ℝ) = 87.4 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l2313_231315


namespace NUMINAMATH_CALUDE_fraction_equality_l2313_231317

theorem fraction_equality {a b c : ℝ} (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2313_231317


namespace NUMINAMATH_CALUDE_circle_radius_ratio_l2313_231337

theorem circle_radius_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (area_ratio : π * r₂^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_ratio_l2313_231337


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l2313_231346

/-- 
Given a triangle ABC where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively,
if 1 + cos A = (b + c) / c, then the triangle is a right triangle.
-/
theorem triangle_is_right_angle 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_cos : 1 + Real.cos A = (b + c) / c)
  : a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l2313_231346


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2313_231333

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + 1 + 2*m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + 1 + 2*m = 0 → y = x) → 
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2313_231333


namespace NUMINAMATH_CALUDE_john_wallet_dimes_l2313_231353

def total_amount : ℚ := 680 / 100  -- $6.80 as a rational number

theorem john_wallet_dimes :
  ∀ (d q : ℕ),  -- d: number of dimes, q: number of quarters
  d = q + 4 →  -- four more dimes than quarters
  (d : ℚ) * (10 / 100) + (q : ℚ) * (25 / 100) = total_amount →  -- total amount equation
  d = 22 :=
by sorry

end NUMINAMATH_CALUDE_john_wallet_dimes_l2313_231353


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2313_231303

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 ≥ 1 → (x ≥ 0 ∨ x ≤ -1))) ↔
  (∀ x : ℝ, (-1 < x ∧ x < 0 → x^2 < 1)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2313_231303


namespace NUMINAMATH_CALUDE_triangle_area_after_median_division_l2313_231345

-- Define a triangle type
structure Triangle where
  area : ℝ

-- Define a function that represents dividing a triangle by a median
def divideByMedian (t : Triangle) : (Triangle × Triangle) :=
  sorry

-- Theorem statement
theorem triangle_area_after_median_division (t : Triangle) :
  let (t1, t2) := divideByMedian t
  t1.area = 7 → t.area = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_after_median_division_l2313_231345


namespace NUMINAMATH_CALUDE_chessboard_sum_zero_l2313_231336

/-- Represents a chessboard with signed numbers -/
def SignedChessboard := Fin 8 → Fin 8 → Int

/-- Checks if a row has exactly four positive and four negative numbers -/
def valid_row (board : SignedChessboard) (row : Fin 8) : Prop :=
  (Finset.filter (λ col => board row col > 0) Finset.univ).card = 4 ∧
  (Finset.filter (λ col => board row col < 0) Finset.univ).card = 4

/-- Checks if a column has exactly four positive and four negative numbers -/
def valid_column (board : SignedChessboard) (col : Fin 8) : Prop :=
  (Finset.filter (λ row => board row col > 0) Finset.univ).card = 4 ∧
  (Finset.filter (λ row => board row col < 0) Finset.univ).card = 4

/-- Checks if the board contains numbers from 1 to 64 with signs -/
def valid_numbers (board : SignedChessboard) : Prop :=
  ∀ n : Fin 64, ∃ (i j : Fin 8), |board i j| = n.val + 1

/-- The main theorem: sum of all numbers on a valid chessboard is zero -/
theorem chessboard_sum_zero (board : SignedChessboard)
  (h_rows : ∀ row, valid_row board row)
  (h_cols : ∀ col, valid_column board col)
  (h_nums : valid_numbers board) :
  (Finset.univ.sum (λ (i : Fin 8) => Finset.univ.sum (λ (j : Fin 8) => board i j))) = 0 :=
sorry

end NUMINAMATH_CALUDE_chessboard_sum_zero_l2313_231336


namespace NUMINAMATH_CALUDE_x_value_l2313_231398

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem x_value (x : ℕ) (h1 : x ∉ A) (h2 : x ∈ B) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2313_231398


namespace NUMINAMATH_CALUDE_jason_picked_46_pears_l2313_231354

/-- Calculates the number of pears Jason picked given the number of pears Keith picked,
    the number of pears Mike ate, and the number of pears left. -/
def jasons_pears (keith_pears mike_ate pears_left : ℕ) : ℕ :=
  (mike_ate + pears_left) - keith_pears

/-- Proves that Jason picked 46 pears given the problem conditions. -/
theorem jason_picked_46_pears :
  jasons_pears 47 12 81 = 46 := by
  sorry

end NUMINAMATH_CALUDE_jason_picked_46_pears_l2313_231354


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2313_231350

theorem power_tower_mod_500 : 2^(2^(2^2)) % 500 = 536 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2313_231350


namespace NUMINAMATH_CALUDE_number_of_factors_of_48_l2313_231311

theorem number_of_factors_of_48 : Nat.card (Nat.divisors 48) = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_48_l2313_231311


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2313_231313

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (train_length : ℝ) 
  (h1 : jogger_speed = 9 * (1000 / 3600))  -- 9 kmph in m/s
  (h2 : train_speed = 45 * (1000 / 3600))  -- 45 kmph in m/s
  (h3 : initial_distance = 240)
  (h4 : train_length = 120) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) = 36 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_jogger_l2313_231313


namespace NUMINAMATH_CALUDE_quadratic_solution_l2313_231378

theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : 
  c = 1 ∧ d = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2313_231378


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2313_231388

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = 0 ∧ k = 0) →  -- center at (0,0)
  c = 8 →            -- focus at (0,8)
  a = 4 →            -- vertex at (0,-4)
  c^2 = a^2 + b^2 →  -- relationship between a, b, and c
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2313_231388


namespace NUMINAMATH_CALUDE_sin_fourteen_pi_fifths_l2313_231348

theorem sin_fourteen_pi_fifths : 
  Real.sin (14 * π / 5) = (Real.sqrt (10 - 2 * Real.sqrt 5)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_fourteen_pi_fifths_l2313_231348


namespace NUMINAMATH_CALUDE_mba_committee_size_l2313_231341

theorem mba_committee_size 
  (total_mbas : ℕ) 
  (num_committees : ℕ) 
  (prob_same_committee : ℚ) :
  total_mbas = 6 ∧ 
  num_committees = 2 ∧ 
  prob_same_committee = 2/5 →
  ∃ (committee_size : ℕ), 
    committee_size * num_committees = total_mbas ∧
    committee_size = 3 :=
by sorry

end NUMINAMATH_CALUDE_mba_committee_size_l2313_231341


namespace NUMINAMATH_CALUDE_complex_number_problem_l2313_231347

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = a + Complex.I * Real.sqrt 3 → z * z = 4 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2313_231347


namespace NUMINAMATH_CALUDE_disneyland_attractions_permutations_l2313_231322

theorem disneyland_attractions_permutations : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_attractions_permutations_l2313_231322


namespace NUMINAMATH_CALUDE_correct_formula_l2313_231304

def f (x : ℝ) : ℝ := 200 - 10*x - 10*x^2

theorem correct_formula : 
  f 0 = 200 ∧ f 1 = 170 ∧ f 2 = 120 ∧ f 3 = 50 ∧ f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_formula_l2313_231304


namespace NUMINAMATH_CALUDE_ratio_problem_l2313_231392

/-- Given two numbers in a 15:1 ratio where the first number is 150, prove that the second number is 10. -/
theorem ratio_problem (a b : ℝ) (h1 : a / b = 15) (h2 : a = 150) : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2313_231392


namespace NUMINAMATH_CALUDE_angle_between_quito_and_kampala_l2313_231301

/-- The angle at the center of a spherical Earth between two points on the equator -/
def angle_at_center (west_longitude east_longitude : ℝ) : ℝ :=
  west_longitude + east_longitude

/-- Theorem: The angle at the center of a spherical Earth between two points,
    one at 78° W and the other at 32° E, both on the equator, is 110°. -/
theorem angle_between_quito_and_kampala :
  angle_at_center 78 32 = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_quito_and_kampala_l2313_231301


namespace NUMINAMATH_CALUDE_frannie_jump_count_l2313_231312

/-- The number of times Meg jumped -/
def meg_jumps : ℕ := 71

/-- The difference between Meg's and Frannie's jumps -/
def jump_difference : ℕ := 18

/-- The number of times Frannie jumped -/
def frannie_jumps : ℕ := meg_jumps - jump_difference

theorem frannie_jump_count : frannie_jumps = 53 := by sorry

end NUMINAMATH_CALUDE_frannie_jump_count_l2313_231312


namespace NUMINAMATH_CALUDE_correct_factorization_l2313_231376

theorem correct_factorization (a b : ℝ) : a * (a - b) - b * (b - a) = (a - b) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2313_231376


namespace NUMINAMATH_CALUDE_n_divisible_by_40_l2313_231309

theorem n_divisible_by_40 (n : ℤ) 
  (h1 : ∃ k : ℤ, 2 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℤ, 3 * n + 1 = m ^ 2) : 
  40 ∣ n := by
sorry

end NUMINAMATH_CALUDE_n_divisible_by_40_l2313_231309


namespace NUMINAMATH_CALUDE_inequality_solution_l2313_231373

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, x > 0 → (Real.log x / Real.log a) + |a + (Real.log x / Real.log a)| * (Real.log a / Real.log (Real.sqrt x)) ≥ a * (Real.log a / Real.log x)) ↔
  -1/3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2313_231373


namespace NUMINAMATH_CALUDE_expression_simplification_l2313_231326

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2313_231326


namespace NUMINAMATH_CALUDE_dance_team_new_members_l2313_231363

/-- Calculates the number of new people who joined a dance team given the initial size, number of people who quit, and final size. -/
def new_members (initial_size quit_count final_size : ℕ) : ℕ :=
  final_size - (initial_size - quit_count)

/-- Proves that 13 new people joined the dance team given the specific conditions. -/
theorem dance_team_new_members :
  let initial_size := 25
  let quit_count := 8
  let final_size := 30
  new_members initial_size quit_count final_size = 13 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_new_members_l2313_231363


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2313_231325

theorem complex_absolute_value (z : ℂ) :
  (3 + 4*I) / z = 5*I → Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2313_231325


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l2313_231305

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 50) = last_two_digits (sum_factorials 9) := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l2313_231305


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l2313_231352

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l2313_231352


namespace NUMINAMATH_CALUDE_linear_regression_at_25_l2313_231357

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

/-- Theorem: The linear regression equation y = 0.50x - 0.81 yields y = 11.69 when x = 25 -/
theorem linear_regression_at_25 : linear_regression 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_at_25_l2313_231357


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_and_center_at_parabola_focus_l2313_231374

theorem circle_tangent_to_line_and_center_at_parabola_focus :
  ∀ (x y : ℝ),
  (∃ (h : ℝ), y^2 = 8*x → (2, 0) = (h, 0)) →
  (∃ (r : ℝ), r = Real.sqrt 2) →
  (x - 2)^2 + y^2 = 2 →
  (∃ (t : ℝ), t = x ∧ t = y) →
  (∃ (d : ℝ), d = |x - y| / Real.sqrt 2 ∧ d = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_and_center_at_parabola_focus_l2313_231374


namespace NUMINAMATH_CALUDE_factor_z4_minus_81_l2313_231387

theorem factor_z4_minus_81 (z : ℂ) : 
  z^4 - 81 = (z - 3) * (z + 3) * (z^2 + 9) := by sorry

end NUMINAMATH_CALUDE_factor_z4_minus_81_l2313_231387


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2313_231380

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, x^5 + x^4 + 1 = 3^y * 7^z ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2313_231380


namespace NUMINAMATH_CALUDE_even_z_dominoes_l2313_231359

/-- Represents a lattice polygon that can be covered by quad-dominoes -/
structure LatticePolygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents an S-quad-domino -/
inductive SQuadDomino

/-- Represents a Z-quad-domino -/
inductive ZQuadDomino

/-- Represents a covering of a lattice polygon with quad-dominoes -/
structure Covering (P : LatticePolygon) where
  s_dominoes : List SQuadDomino
  z_dominoes : List ZQuadDomino
  is_valid : Bool -- Indicates if the covering is valid (no overlap and complete)

/-- Checks if a lattice polygon can be completely covered by S-quad-dominoes -/
def can_cover_with_s (P : LatticePolygon) : Prop :=
  ∃ (c : Covering P), c.z_dominoes.length = 0 ∧ c.is_valid

/-- Main theorem: If a lattice polygon can be covered by S-quad-dominoes,
    then any valid covering with S and Z quad-dominoes uses an even number of Z-quad-dominoes -/
theorem even_z_dominoes (P : LatticePolygon) 
  (h : can_cover_with_s P) : 
  ∀ (c : Covering P), c.is_valid → Even c.z_dominoes.length :=
sorry

end NUMINAMATH_CALUDE_even_z_dominoes_l2313_231359


namespace NUMINAMATH_CALUDE_museum_visitors_l2313_231364

theorem museum_visitors (yesterday : ℕ) (today_increase : ℕ) : 
  yesterday = 247 → today_increase = 131 → 
  yesterday + (yesterday + today_increase) = 625 := by
sorry

end NUMINAMATH_CALUDE_museum_visitors_l2313_231364


namespace NUMINAMATH_CALUDE_wooden_easel_cost_l2313_231375

theorem wooden_easel_cost (paintbrush_cost paint_cost albert_has additional_needed : ℚ)
  (h1 : paintbrush_cost = 1.5)
  (h2 : paint_cost = 4.35)
  (h3 : albert_has = 6.5)
  (h4 : additional_needed = 12) :
  let total_cost := albert_has + additional_needed
  let other_items_cost := paintbrush_cost + paint_cost
  let easel_cost := total_cost - other_items_cost
  easel_cost = 12.65 := by sorry

end NUMINAMATH_CALUDE_wooden_easel_cost_l2313_231375


namespace NUMINAMATH_CALUDE_solve_equation_l2313_231371

theorem solve_equation (b : ℚ) (h : b + b/4 = 5/2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2313_231371


namespace NUMINAMATH_CALUDE_unique_number_base_conversion_l2313_231360

/-- Represents a digit in a given base -/
def IsDigit (d : ℕ) (base : ℕ) : Prop := d < base

/-- Converts a two-digit number in a given base to decimal -/
def ToDecimal (tens : ℕ) (ones : ℕ) (base : ℕ) : ℕ := base * tens + ones

theorem unique_number_base_conversion :
  ∃! n : ℕ, n > 0 ∧
    ∃ C D : ℕ,
      IsDigit C 8 ∧
      IsDigit D 8 ∧
      IsDigit C 6 ∧
      IsDigit D 6 ∧
      n = ToDecimal C D 8 ∧
      n = ToDecimal D C 6 ∧
      n = 43 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_base_conversion_l2313_231360


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2313_231323

theorem chosen_number_proof (x : ℝ) : (x / 12) - 240 = 8 ↔ x = 2976 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2313_231323


namespace NUMINAMATH_CALUDE_print_350_pages_time_l2313_231306

/-- Calculates the time needed to print a given number of pages with a printer that has a specified
printing rate and pause interval. -/
def print_time (total_pages : ℕ) (pages_per_minute : ℕ) (pause_interval : ℕ) (pause_duration : ℕ) : ℕ :=
  let num_pauses := (total_pages / pause_interval) - 1
  let pause_time := num_pauses * pause_duration
  let print_time := (total_pages + pages_per_minute - 1) / pages_per_minute
  print_time + pause_time

/-- Theorem stating that printing 350 pages with the given printer specifications
takes approximately 27 minutes. -/
theorem print_350_pages_time :
  print_time 350 23 50 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_print_350_pages_time_l2313_231306


namespace NUMINAMATH_CALUDE_tangent_chord_fixed_point_l2313_231349

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A line represented by two points -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- Determines if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Determines if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Determines if a line is perpendicular to another line -/
def isPerpendicular (l1 l2 : Line) : Prop := sorry

/-- Determines if a point is outside a circle -/
def isOutside (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem tangent_chord_fixed_point 
  (O : ℝ × ℝ) (r : ℝ) (l : Line) (H : ℝ × ℝ) :
  let c : Circle := ⟨O, r⟩
  isOutside H c →
  pointOnLine H l →
  isPerpendicular (Line.mk O H) l →
  ∃ P : ℝ × ℝ, ∀ A : ℝ × ℝ, 
    pointOnLine A l →
    ∃ B C : ℝ × ℝ,
      isTangent (Line.mk A B) c ∧
      isTangent (Line.mk A C) c ∧
      pointOnLine P (Line.mk B C) ∧
      pointOnLine P (Line.mk O H) :=
sorry

end NUMINAMATH_CALUDE_tangent_chord_fixed_point_l2313_231349


namespace NUMINAMATH_CALUDE_simplify_expression_l2313_231319

theorem simplify_expression (x : ℝ) : (3*x - 6)*(2*x + 8) - (x + 6)*(3*x + 1) = 3*x^2 - 7*x - 54 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2313_231319


namespace NUMINAMATH_CALUDE_billion_two_hundred_million_scientific_notation_l2313_231302

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_two_hundred_million_scientific_notation :
  toScientificNotation 1200000000 = ScientificNotation.mk 1.2 9 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_billion_two_hundred_million_scientific_notation_l2313_231302


namespace NUMINAMATH_CALUDE_student_count_l2313_231334

theorem student_count (n : ℕ) 
  (yellow : ℕ) (red : ℕ) (blue : ℕ) 
  (yellow_blue : ℕ) (yellow_red : ℕ) (blue_red : ℕ)
  (all_colors : ℕ) :
  yellow = 46 →
  red = 69 →
  blue = 104 →
  yellow_blue = 14 →
  yellow_red = 13 →
  blue_red = 19 →
  all_colors = 16 →
  n = yellow + red + blue - yellow_blue - yellow_red - blue_red + all_colors →
  n = 141 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l2313_231334


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l2313_231396

/-- Given two points C and D, where C is at the origin and D is on the line y = 6,
    if the slope of CD is 3/4, then the sum of D's coordinates is 14. -/
theorem point_coordinates_sum (x : ℝ) : 
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (x, 6)
  (6 - 0) / (x - 0) = 3 / 4 →
  x + 6 = 14 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l2313_231396


namespace NUMINAMATH_CALUDE_alex_upside_down_growth_rate_l2313_231366

/-- The growth rate of Alex when hanging upside down -/
def upsideDownGrowthRate (
  requiredHeight : ℚ)
  (currentHeight : ℚ)
  (normalGrowthRate : ℚ)
  (upsideDownHoursPerMonth : ℚ)
  (monthsInYear : ℕ) : ℚ :=
  let totalGrowthNeeded := requiredHeight - currentHeight
  let normalYearlyGrowth := normalGrowthRate * monthsInYear
  let additionalGrowthNeeded := totalGrowthNeeded - normalYearlyGrowth
  let totalUpsideDownHours := upsideDownHoursPerMonth * monthsInYear
  additionalGrowthNeeded / totalUpsideDownHours

/-- Theorem stating that Alex's upside down growth rate is 1/12 inch per hour -/
theorem alex_upside_down_growth_rate :
  upsideDownGrowthRate 54 48 (1/3) 2 12 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_alex_upside_down_growth_rate_l2313_231366


namespace NUMINAMATH_CALUDE_chocolate_packaging_cost_l2313_231399

theorem chocolate_packaging_cost 
  (num_bars : ℕ) 
  (cost_per_bar : ℚ) 
  (total_selling_price : ℚ) 
  (total_profit : ℚ) 
  (h1 : num_bars = 5)
  (h2 : cost_per_bar = 5)
  (h3 : total_selling_price = 90)
  (h4 : total_profit = 55) :
  (total_selling_price - total_profit - (↑num_bars * cost_per_bar)) / ↑num_bars = 2 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_packaging_cost_l2313_231399


namespace NUMINAMATH_CALUDE_tennis_tournament_l2313_231377

theorem tennis_tournament (n : ℕ) : n > 0 → (
  ∃ (women_wins men_wins : ℕ),
    women_wins + men_wins = (4 * n).choose 2 ∧
    women_wins * 11 = men_wins * 4 ∧
    ∀ m : ℕ, m > 0 ∧ m < n → ¬(
      ∃ (w_wins m_wins : ℕ),
        w_wins + m_wins = (4 * m).choose 2 ∧
        w_wins * 11 = m_wins * 4
    )
) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_l2313_231377


namespace NUMINAMATH_CALUDE_number_problem_l2313_231328

theorem number_problem : ∃ x : ℝ, x = 40 ∧ 0.8 * x > (4/5) * 25 + 12 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2313_231328


namespace NUMINAMATH_CALUDE_train_speed_train_speed_approximately_60_l2313_231318

/-- The speed of a train given its length, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed - (man_speed * 1000 / 3600)
  let train_speed_kmh := train_speed_ms * 3600 / 1000
  train_speed_kmh

/-- The speed of the train is approximately 60 km/hr given the specified conditions. -/
theorem train_speed_approximately_60 :
  ∃ ε > 0, abs (train_speed 220 6 11.999040076793857 - 60) < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_approximately_60_l2313_231318


namespace NUMINAMATH_CALUDE_amoeba_reproduction_time_verify_16_amoebae_l2313_231335

/-- Represents the number of amoebae after a certain number of divisions -/
def amoebae_count (divisions : ℕ) : ℕ := 2^divisions

/-- Represents the time taken for a given number of divisions -/
def time_for_divisions (divisions : ℕ) : ℕ := 8

/-- The number of divisions required to reach 16 amoebae from 1 -/
def divisions_to_16 : ℕ := 4

/-- Theorem stating that it takes 2 days for an amoeba to reproduce -/
theorem amoeba_reproduction_time : 
  (time_for_divisions divisions_to_16) / divisions_to_16 = 2 := by
  sorry

/-- Verifies that 16 amoebae are indeed reached after 4 divisions -/
theorem verify_16_amoebae : amoebae_count divisions_to_16 = 16 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_reproduction_time_verify_16_amoebae_l2313_231335


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l2313_231329

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the theorem
theorem min_sum_of_primes (m n p : ℕ) :
  isPrime m → isPrime n → isPrime p →
  m ≠ n → n ≠ p → m ≠ p →
  m + n = p →
  (∀ m' n' p' : ℕ,
    isPrime m' → isPrime n' → isPrime p' →
    m' ≠ n' → n' ≠ p' → m' ≠ p' →
    m' + n' = p' →
    m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l2313_231329


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2313_231358

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x : ℝ, x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2313_231358


namespace NUMINAMATH_CALUDE_phd_basics_time_l2313_231372

/-- Represents the time John spent on his PhD journey -/
structure PhDTime where
  total : ℝ
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- The conditions of John's PhD journey -/
def phd_conditions (t : PhDTime) : Prop :=
  t.total = 7 ∧
  t.acclimation = 1 ∧
  t.research = t.basics + 0.75 * t.basics ∧
  t.dissertation = 0.5 * t.acclimation ∧
  t.total = t.acclimation + t.basics + t.research + t.dissertation

/-- Theorem stating that given the PhD conditions, the time spent learning basics is 2 years -/
theorem phd_basics_time (t : PhDTime) (h : phd_conditions t) : t.basics = 2 := by
  sorry

end NUMINAMATH_CALUDE_phd_basics_time_l2313_231372


namespace NUMINAMATH_CALUDE_system_solution_l2313_231330

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, 2 * x - y = 5 * k + 6 ∧ 4 * x + 7 * y = k ∧ x + y = 2023) → k = 2022 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2313_231330


namespace NUMINAMATH_CALUDE_percentage_of_female_guests_l2313_231395

theorem percentage_of_female_guests 
  (total_guests : ℕ) 
  (jays_family_females : ℕ) 
  (h1 : total_guests = 240)
  (h2 : jays_family_females = 72)
  (h3 : jays_family_females * 2 = total_guests * (percentage_female_guests / 100)) :
  percentage_female_guests = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_female_guests_l2313_231395


namespace NUMINAMATH_CALUDE_time_per_trip_is_three_l2313_231393

/-- Represents the number of trips Melissa makes to town in a year -/
def trips_per_year : ℕ := 24

/-- Represents the total hours Melissa spends driving in a year -/
def total_driving_hours : ℕ := 72

/-- Calculates the time for one round trip to town and back -/
def time_per_trip : ℚ := total_driving_hours / trips_per_year

theorem time_per_trip_is_three : time_per_trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_trip_is_three_l2313_231393


namespace NUMINAMATH_CALUDE_prob_two_adjacent_is_one_fifth_l2313_231307

def num_knights : ℕ := 30
def num_selected : ℕ := 3

def prob_at_least_two_adjacent : ℚ :=
  1 - (num_knights * (num_knights - 3) * (num_knights - 4) - num_knights * 2 * (num_knights - 3)) / (num_knights.choose num_selected)

theorem prob_two_adjacent_is_one_fifth :
  prob_at_least_two_adjacent = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_adjacent_is_one_fifth_l2313_231307


namespace NUMINAMATH_CALUDE_number_division_problem_l2313_231310

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 24) / y = 3) : 
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2313_231310


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2313_231397

theorem rectangular_field_width (width length perimeter : ℝ) : 
  length = (7/5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 384 →
  width = 80 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2313_231397


namespace NUMINAMATH_CALUDE_zelda_success_probability_l2313_231356

theorem zelda_success_probability 
  (p_xavier : ℝ) 
  (p_yvonne : ℝ) 
  (p_xy_not_z : ℝ) 
  (h1 : p_xavier = 1/4)
  (h2 : p_yvonne = 2/3)
  (h3 : p_xy_not_z = 0.0625)
  (h4 : p_xy_not_z = p_xavier * p_yvonne * (1 - p_zelda)) :
  p_zelda = 5/8 := by
sorry

end NUMINAMATH_CALUDE_zelda_success_probability_l2313_231356


namespace NUMINAMATH_CALUDE_no_event_with_prob_1_5_l2313_231327

-- Define the probability measure
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)

-- Axiom: Probability is always between 0 and 1
axiom prob_bounds (E : Set Ω) : 0 ≤ P E ∧ P E ≤ 1

-- Theorem: There does not exist an event with probability 1.5
theorem no_event_with_prob_1_5 : ¬∃ (A : Set Ω), P A = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_no_event_with_prob_1_5_l2313_231327


namespace NUMINAMATH_CALUDE_license_plate_increase_l2313_231338

theorem license_plate_increase : 
  let old_plates := 26 * (10 ^ 3)
  let new_plates := (26 ^ 4) * (10 ^ 4)
  (new_plates / old_plates : ℚ) = 175760 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2313_231338


namespace NUMINAMATH_CALUDE_parabola_points_range_l2313_231385

-- Define the parabola
def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

-- Define the theorem
theorem parabola_points_range (a b y₁ y₂ n : ℝ) :
  a > 0 →
  y₁ < y₂ →
  parabola a b (2 * n + 3) = y₁ →
  parabola a b (n - 1) = y₂ →
  (2 * n + 3 - 1) * (n - 1 - 1) < 0 →  -- Opposite sides of axis of symmetry
  -1 < n ∧ n < 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_points_range_l2313_231385


namespace NUMINAMATH_CALUDE_mans_upward_speed_l2313_231342

/-- Proves that given a man traveling with an average speed of 28.8 km/hr
    and a downward speed of 36 km/hr, his upward speed is 24 km/hr. -/
theorem mans_upward_speed
  (v_avg : ℝ) (v_down : ℝ) (h_avg : v_avg = 28.8)
  (h_down : v_down = 36) :
  let v_up := 2 * v_avg * v_down / (2 * v_down - v_avg)
  v_up = 24 := by sorry

end NUMINAMATH_CALUDE_mans_upward_speed_l2313_231342


namespace NUMINAMATH_CALUDE_age_difference_l2313_231381

theorem age_difference (D M : ℕ) : 
  (M = 11 * D) →
  (M + 13 = 2 * (D + 13)) →
  (M - D = 40) := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2313_231381


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2313_231386

theorem pie_eating_contest (a b c : ℚ) 
  (ha : a = 4/5) (hb : b = 5/6) (hc : c = 3/4) : 
  (max a (max b c) - min a (min b c) : ℚ) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2313_231386


namespace NUMINAMATH_CALUDE_revenue_increase_percentage_l2313_231343

/-- Calculates the percentage increase in revenue given initial and new package volumes and prices. -/
theorem revenue_increase_percentage
  (initial_volume : ℝ)
  (initial_price : ℝ)
  (new_volume : ℝ)
  (new_price : ℝ)
  (h1 : initial_volume = 1)
  (h2 : initial_price = 60)
  (h3 : new_volume = 0.9)
  (h4 : new_price = 81) :
  (new_price / new_volume - initial_price / initial_volume) / (initial_price / initial_volume) * 100 = 50 := by
sorry


end NUMINAMATH_CALUDE_revenue_increase_percentage_l2313_231343
