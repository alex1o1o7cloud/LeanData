import Mathlib

namespace NUMINAMATH_CALUDE_min_value_a_l3527_352751

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : 
  (∀ b : ℝ, b > 0 ∧ (∀ x : ℝ, |x - b| + |1 - x| ≥ 1) → b ≥ 2) ∧ 
  (∃ c : ℝ, c > 0 ∧ (∀ x : ℝ, |x - c| + |1 - x| ≥ 1) ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3527_352751


namespace NUMINAMATH_CALUDE_hyperbola_distance_l3527_352747

/-- Given a hyperbola with equation x²/25 - y²/9 = 1, prove that |ON| = 4 --/
theorem hyperbola_distance (M F₁ F₂ N O : ℝ × ℝ) : 
  (∀ x y, (x^2 / 25) - (y^2 / 9) = 1 → (x, y) = M) →  -- M is on the hyperbola
  (M.1 < 0) →  -- M is on the left branch
  ‖M - F₂‖ = 18 →  -- Distance from M to F₂ is 18
  N = (M + F₂) / 2 →  -- N is the midpoint of MF₂
  O = (0, 0) →  -- O is the origin
  ‖O - N‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_distance_l3527_352747


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_8_with_digit_sum_20_l3527_352745

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → n % 8 = 0 → digit_sum n = 20 → n ≥ 1071 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_8_with_digit_sum_20_l3527_352745


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3527_352787

/-- The x-coordinate of the point on the x-axis equidistant from A(-3, 0) and B(3, 5) is 25/12 -/
theorem equidistant_point_x_coordinate :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (3, 5)
  ∃ x : ℝ, x = 25 / 12 ∧
    (x - A.1) ^ 2 + A.2 ^ 2 = (x - B.1) ^ 2 + (0 - B.2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3527_352787


namespace NUMINAMATH_CALUDE_x_range_theorem_l3527_352746

theorem x_range_theorem (x : ℝ) : 
  (∀ a ∈ Set.Ioo 0 1, (a - 3) * x^2 < (4 * a - 2) * x) ↔ 
  (x ≤ -1 ∨ x ≥ 2/3) := by
sorry

end NUMINAMATH_CALUDE_x_range_theorem_l3527_352746


namespace NUMINAMATH_CALUDE_fold_point_area_l3527_352702

/-- Definition of a triangle ABC --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Definition of a fold point --/
def FoldPoint (t : Triangle) (P : ℝ × ℝ) : Prop :=
  sorry  -- Definition of fold point

/-- Set of all fold points of a triangle --/
def FoldPointSet (t : Triangle) : Set (ℝ × ℝ) :=
  {P | FoldPoint t P}

/-- Area of a set in ℝ² --/
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of area

theorem fold_point_area (t : Triangle) : 
  t.A.1 = 0 ∧ t.A.2 = 0 ∧
  t.B.1 = 36 ∧ t.B.2 = 0 ∧
  t.C.1 = 0 ∧ t.C.2 = 72 →
  Area (FoldPointSet t) = 270 * Real.pi - 324 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_fold_point_area_l3527_352702


namespace NUMINAMATH_CALUDE_gina_payment_is_90_l3527_352703

/-- Calculates the total payment for Gina's order given her painting rates and order details. -/
def total_payment (rose_rate : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (hourly_rate : ℕ) : ℕ :=
  let rose_time := rose_order / rose_rate
  let lily_time := lily_order / lily_rate
  let total_time := rose_time + lily_time
  total_time * hourly_rate

/-- Proves that Gina's total payment for the given order is $90. -/
theorem gina_payment_is_90 : total_payment 6 7 6 14 30 = 90 := by
  sorry

#eval total_payment 6 7 6 14 30

end NUMINAMATH_CALUDE_gina_payment_is_90_l3527_352703


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l3527_352779

theorem cubic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c)^3 - (a^3 + b^3 + c^3) > (a + b) * (b + c) * (a + c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l3527_352779


namespace NUMINAMATH_CALUDE_dreamy_vacation_probability_l3527_352794

/-- The probability of drawing a dreamy vacation note -/
def p : ℝ := 0.4

/-- The total number of people drawing notes -/
def n : ℕ := 5

/-- The number of people drawing a dreamy vacation note -/
def k : ℕ := 3

/-- The target probability -/
def target_prob : ℝ := 0.2304

/-- Theorem stating that the probability of exactly k people out of n drawing a dreamy vacation note is equal to the target probability -/
theorem dreamy_vacation_probability :
  Nat.choose n k * p^k * (1 - p)^(n - k) = target_prob := by
  sorry

end NUMINAMATH_CALUDE_dreamy_vacation_probability_l3527_352794


namespace NUMINAMATH_CALUDE_january_oil_bill_l3527_352767

theorem january_oil_bill (january february : ℝ) 
  (h1 : february / january = 5 / 4)
  (h2 : (february + 30) / january = 3 / 2) : 
  january = 120 := by
  sorry

end NUMINAMATH_CALUDE_january_oil_bill_l3527_352767


namespace NUMINAMATH_CALUDE_complex_sum_equality_l3527_352714

theorem complex_sum_equality : 
  8 * Complex.exp (2 * π * I / 13) + 8 * Complex.exp (15 * π * I / 26) = 
  8 * Real.sqrt 3 * Complex.exp (19 * π * I / 52) := by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l3527_352714


namespace NUMINAMATH_CALUDE_smallest_x_value_l3527_352761

theorem smallest_x_value (x : ℝ) : 
  (5 * x^2 + 7 * x + 3 = 6) → x ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3527_352761


namespace NUMINAMATH_CALUDE_loans_equal_at_start_l3527_352727

/-- Represents the loan details for a person -/
structure Loan where
  principal : ℝ
  dailyInterestRate : ℝ

/-- Calculates the balance of a loan after t days -/
def loanBalance (loan : Loan) (t : ℝ) : ℝ :=
  loan.principal * (1 + loan.dailyInterestRate * t)

theorem loans_equal_at_start (claudia bob diana : Loan)
  (h_claudia : claudia = { principal := 200, dailyInterestRate := 0.04 })
  (h_bob : bob = { principal := 300, dailyInterestRate := 0.03 })
  (h_diana : diana = { principal := 500, dailyInterestRate := 0.02 }) :
  ∃ t : ℝ, t = 0 ∧ loanBalance claudia t + loanBalance bob t = loanBalance diana t :=
sorry

end NUMINAMATH_CALUDE_loans_equal_at_start_l3527_352727


namespace NUMINAMATH_CALUDE_vicente_spent_2475_l3527_352730

/-- Calculates the total amount spent by Vicente on rice and meat --/
def total_spent (rice_kg : ℕ) (rice_price : ℚ) (rice_discount : ℚ)
                (meat_lbs : ℕ) (meat_price : ℚ) (meat_tax : ℚ) : ℚ :=
  let rice_cost := rice_kg * rice_price * (1 - rice_discount)
  let meat_cost := meat_lbs * meat_price * (1 + meat_tax)
  rice_cost + meat_cost

/-- Theorem stating that Vicente's total spent is $24.75 --/
theorem vicente_spent_2475 :
  total_spent 5 2 (1/10) 3 5 (1/20) = 2475/100 := by
  sorry

end NUMINAMATH_CALUDE_vicente_spent_2475_l3527_352730


namespace NUMINAMATH_CALUDE_cody_final_tickets_l3527_352775

/-- Calculates the final number of tickets Cody has after various transactions at the arcade. -/
def final_tickets (initial_tickets : ℕ) (won_tickets : ℕ) (beanie_cost : ℕ) (traded_tickets : ℕ) (games_played : ℕ) (tickets_per_game : ℕ) : ℕ :=
  initial_tickets + won_tickets - beanie_cost - traded_tickets + (games_played * tickets_per_game)

/-- Theorem stating that Cody ends up with 82 tickets given the specific conditions of the problem. -/
theorem cody_final_tickets :
  final_tickets 50 49 25 10 3 6 = 82 := by sorry

end NUMINAMATH_CALUDE_cody_final_tickets_l3527_352775


namespace NUMINAMATH_CALUDE_triangle_area_l3527_352749

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  b + c = 5 →
  Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * Real.tan B * Real.tan C →
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3527_352749


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3527_352735

theorem imaginary_part_of_z (z : ℂ) (h : z * (3 - 4*I) = 5) : z.im = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3527_352735


namespace NUMINAMATH_CALUDE_integral_equality_l3527_352716

theorem integral_equality : ∫ (x : ℝ) in Set.Icc π (2*π), (1 - Real.cos x) / (x - Real.sin x)^2 = 1 / (2*π) := by
  sorry

end NUMINAMATH_CALUDE_integral_equality_l3527_352716


namespace NUMINAMATH_CALUDE_sequence_inequality_l3527_352758

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) : 
  a 2 * a 4 ≤ a 3 ^ 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3527_352758


namespace NUMINAMATH_CALUDE_small_circle_area_l3527_352700

theorem small_circle_area (large_circle_area : ℝ) (num_small_circles : ℕ) :
  large_circle_area = 120 →
  num_small_circles = 6 →
  ∃ small_circle_area : ℝ,
    small_circle_area = large_circle_area / (3 * num_small_circles) ∧
    small_circle_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_area_l3527_352700


namespace NUMINAMATH_CALUDE_corner_rectangles_area_sum_l3527_352791

/-- Given a 2019x2019 square divided into 9 rectangles, with the central rectangle
    having dimensions 1511x1115, the sum of the areas of the four corner rectangles
    is 1832128. -/
theorem corner_rectangles_area_sum (square_side : ℕ) (central_length central_width : ℕ) :
  square_side = 2019 →
  central_length = 1511 →
  central_width = 1115 →
  4 * ((square_side - central_length) * (square_side - central_width)) = 1832128 :=
by sorry

end NUMINAMATH_CALUDE_corner_rectangles_area_sum_l3527_352791


namespace NUMINAMATH_CALUDE_milk_division_l3527_352762

theorem milk_division (total_milk : ℚ) (num_kids : ℕ) (milk_per_kid : ℚ) : 
  total_milk = 3 → 
  num_kids = 5 → 
  milk_per_kid = total_milk / num_kids → 
  milk_per_kid = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_milk_division_l3527_352762


namespace NUMINAMATH_CALUDE_only_constant_one_is_divisor_respecting_l3527_352708

-- Define the number of positive divisors function
def d (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).card + 1

-- Define divisor-respecting property
def divisor_respecting (F : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, d (F (m * n)) = d (F m) * d (F n)) ∧
  (∀ n : ℕ, d (F n) ≤ d n)

-- Theorem statement
theorem only_constant_one_is_divisor_respecting :
  ∀ F : ℕ → ℕ, divisor_respecting F → ∀ x : ℕ, F x = 1 :=
by sorry

end NUMINAMATH_CALUDE_only_constant_one_is_divisor_respecting_l3527_352708


namespace NUMINAMATH_CALUDE_milford_lake_algae_count_l3527_352764

theorem milford_lake_algae_count (current : ℕ) (increase : ℕ) (original : ℕ)
  (h1 : current = 3263)
  (h2 : increase = 2454)
  (h3 : current = original + increase) :
  original = 809 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_count_l3527_352764


namespace NUMINAMATH_CALUDE_hexagon_area_lower_bound_l3527_352721

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Area of the hexagon formed by extending the sides of the triangle -/
def hexagon_area (t : Triangle) : ℝ := sorry

/-- The area of the hexagon is at least 13 times the area of the triangle -/
theorem hexagon_area_lower_bound (t : Triangle) :
  hexagon_area t ≥ 13 * area t := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_lower_bound_l3527_352721


namespace NUMINAMATH_CALUDE_systematic_sample_max_l3527_352757

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  totalProducts : Nat
  sampleSize : Nat
  interval : Nat

/-- Creates a systematic sample given total products and sample size -/
def createSystematicSample (totalProducts sampleSize : Nat) : SystematicSample :=
  { totalProducts := totalProducts
  , sampleSize := sampleSize
  , interval := totalProducts / sampleSize }

/-- Checks if a number is in the sample given the first element -/
def isInSample (sample : SystematicSample) (first last : Nat) : Prop :=
  ∃ k, k < sample.sampleSize ∧ first + k * sample.interval = last

/-- Theorem: In a systematic sample of size 5 from 80 products,
    if product 28 is in the sample, then the maximum number in the sample is 76 -/
theorem systematic_sample_max (sample : SystematicSample) 
  (h1 : sample.totalProducts = 80)
  (h2 : sample.sampleSize = 5)
  (h3 : isInSample sample 28 28) :
  isInSample sample 28 76 ∧ ∀ n, isInSample sample 28 n → n ≤ 76 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_max_l3527_352757


namespace NUMINAMATH_CALUDE_total_customers_l3527_352709

def customers_in_line (people_in_front : ℕ) : ℕ := people_in_front + 1

theorem total_customers (people_in_front : ℕ) : 
  people_in_front = 8 → customers_in_line people_in_front = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_customers_l3527_352709


namespace NUMINAMATH_CALUDE_no_spiky_two_digit_integers_l3527_352701

/-- A two-digit positive integer is spiky if it equals the sum of its tens digit and 
    the cube of its units digit subtracted by twice the tens digit. -/
def IsSpiky (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ 
  ∃ a b : ℕ, n = 10 * a + b ∧ 
             n = a + b^3 - 2*a

/-- There are no spiky two-digit positive integers. -/
theorem no_spiky_two_digit_integers : ¬∃ n : ℕ, IsSpiky n := by
  sorry

#check no_spiky_two_digit_integers

end NUMINAMATH_CALUDE_no_spiky_two_digit_integers_l3527_352701


namespace NUMINAMATH_CALUDE_nth_decimal_35_36_l3527_352734

/-- The fraction 35/36 as a real number -/
def f : ℚ := 35 / 36

/-- Predicate to check if the nth decimal digit of a rational number is 2 -/
def is_nth_decimal_2 (q : ℚ) (n : ℕ) : Prop :=
  (q * 10^n - ⌊q * 10^n⌋) * 10 ≥ 2 ∧ (q * 10^n - ⌊q * 10^n⌋) * 10 < 3

/-- Theorem stating that the nth decimal digit of 35/36 is 2 if and only if n ≥ 2 -/
theorem nth_decimal_35_36 (n : ℕ) : is_nth_decimal_2 f n ↔ n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_nth_decimal_35_36_l3527_352734


namespace NUMINAMATH_CALUDE_sum_of_a_values_l3527_352781

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := 4 * x^2 + a * x + 8 * x + 9

-- Define the condition for the equation to have only one solution
def has_one_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, quadratic_equation a x = 0

-- Define the set of 'a' values that satisfy the condition
def a_values : Set ℝ := {a | has_one_solution a}

-- State the theorem
theorem sum_of_a_values :
  ∃ a₁ a₂ : ℝ, a₁ ∈ a_values ∧ a₂ ∈ a_values ∧ a₁ ≠ a₂ ∧ a₁ + a₂ = -16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l3527_352781


namespace NUMINAMATH_CALUDE_infinite_sum_equals_three_fortieths_l3527_352770

/-- The sum of the infinite series n / (n^4 + 16) from n = 1 to infinity equals 3/40 -/
theorem infinite_sum_equals_three_fortieths :
  (∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 + 16)) = 3/40 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_three_fortieths_l3527_352770


namespace NUMINAMATH_CALUDE_power_equation_solution_l3527_352778

theorem power_equation_solution (n : ℕ) : 5^29 * 4^15 = 2 * 10^n → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3527_352778


namespace NUMINAMATH_CALUDE_exists_n_plus_Sn_1980_consecutive_n_plus_Sn_l3527_352792

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Part a: Existence of n such that n + S(n) = 1980
theorem exists_n_plus_Sn_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Part b: At least one of two consecutive naturals is of the form n + S(n)
theorem consecutive_n_plus_Sn (k : ℕ) : 
  (∃ n : ℕ, k = n + S n) ∨ (∃ n : ℕ, k + 1 = n + S n) := by sorry

end NUMINAMATH_CALUDE_exists_n_plus_Sn_1980_consecutive_n_plus_Sn_l3527_352792


namespace NUMINAMATH_CALUDE_equation_solution_l3527_352759

theorem equation_solution : 
  {x : ℝ | (x^3 - 5*x^2 + 6*x)*(x - 5) = 0} = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3527_352759


namespace NUMINAMATH_CALUDE_disco_ball_max_cost_l3527_352760

def disco_ball_cost (total_budget : ℕ) (food_boxes : ℕ) (food_cost : ℕ) (disco_balls : ℕ) : ℕ :=
  (total_budget - food_boxes * food_cost) / disco_balls

theorem disco_ball_max_cost : disco_ball_cost 330 10 25 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_disco_ball_max_cost_l3527_352760


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3527_352715

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_is_zero :
  i^23456 + i^23457 + i^23458 + i^23459 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3527_352715


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_33_l3527_352766

theorem perfect_square_power_of_two_plus_33 :
  ∀ n : ℕ, (∃ m : ℕ, 2^n + 33 = m^2) ↔ n = 4 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_33_l3527_352766


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3527_352742

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * x^2 - 8 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1) ∧
  (∀ x : ℝ, 5 * x^2 - 4 * x - 1 = 0 ↔ x = -1/5 ∨ x = 1) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3527_352742


namespace NUMINAMATH_CALUDE_triangle_tan_half_angles_inequality_l3527_352789

theorem triangle_tan_half_angles_inequality (A B C : ℝ) (h₁ : A + B + C = π) :
  Real.tan (A / 2) * Real.tan (B / 2) * Real.tan (C / 2) ≤ Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tan_half_angles_inequality_l3527_352789


namespace NUMINAMATH_CALUDE_sine_is_periodic_l3527_352756

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
def sin : ℝ → ℝ := sorry

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sin →
  IsPeriodic sin := by sorry

end NUMINAMATH_CALUDE_sine_is_periodic_l3527_352756


namespace NUMINAMATH_CALUDE_rounding_shift_l3527_352722

/-- Rounding function that rounds to the nearest integer -/
noncomputable def f (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

/-- Theorem stating that adding an integer to the input of f
    is equivalent to adding the same integer to the output of f -/
theorem rounding_shift (x : ℝ) (m : ℤ) : f (x + m) = f x + m := by
  sorry

end NUMINAMATH_CALUDE_rounding_shift_l3527_352722


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l3527_352740

theorem square_sum_geq_product (x y z : ℝ) : x + y + z ≥ x * y * z → x^2 + y^2 + z^2 ≥ x * y * z := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l3527_352740


namespace NUMINAMATH_CALUDE_rohan_sudhir_profit_difference_l3527_352780

/-- Represents an investor in the business -/
structure Investor where
  name : String
  amount : ℕ
  months : ℕ

/-- Calculates the investment-time product for an investor -/
def investmentTime (i : Investor) : ℕ := i.amount * i.months

/-- Calculates the share of profit for an investor -/
def profitShare (i : Investor) (totalInvestmentTime totalProfit : ℕ) : ℚ :=
  (investmentTime i : ℚ) / totalInvestmentTime * totalProfit

theorem rohan_sudhir_profit_difference 
  (suresh : Investor)
  (rohan : Investor)
  (sudhir : Investor)
  (priya : Investor)
  (akash : Investor)
  (totalProfit : ℕ) :
  suresh.name = "Suresh" ∧ suresh.amount = 18000 ∧ suresh.months = 12 ∧
  rohan.name = "Rohan" ∧ rohan.amount = 12000 ∧ rohan.months = 9 ∧
  sudhir.name = "Sudhir" ∧ sudhir.amount = 9000 ∧ sudhir.months = 8 ∧
  priya.name = "Priya" ∧ priya.amount = 15000 ∧ priya.months = 6 ∧
  akash.name = "Akash" ∧ akash.amount = 10000 ∧ akash.months = 6 ∧
  totalProfit = 5948 →
  let totalInvestmentTime := investmentTime suresh + investmentTime rohan + 
                             investmentTime sudhir + investmentTime priya + 
                             investmentTime akash
  (profitShare rohan totalInvestmentTime totalProfit - 
   profitShare sudhir totalInvestmentTime totalProfit).num = 393 :=
by sorry

end NUMINAMATH_CALUDE_rohan_sudhir_profit_difference_l3527_352780


namespace NUMINAMATH_CALUDE_james_pays_37_50_l3527_352750

/-- Calculates the amount James pays for singing lessons -/
def james_payment (total_lessons : ℕ) (free_lessons : ℕ) (full_price_lessons : ℕ) 
  (lesson_cost : ℚ) (uncle_payment_fraction : ℚ) : ℚ :=
  let paid_lessons := total_lessons - free_lessons
  let discounted_lessons := paid_lessons - full_price_lessons
  let half_price_lessons := (discounted_lessons + 1) / 2
  let total_paid_lessons := full_price_lessons + half_price_lessons
  let total_cost := total_paid_lessons * lesson_cost
  (1 - uncle_payment_fraction) * total_cost

/-- Theorem stating that James pays $37.50 for his singing lessons -/
theorem james_pays_37_50 : 
  james_payment 20 1 10 5 (1/2) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_james_pays_37_50_l3527_352750


namespace NUMINAMATH_CALUDE_real_roots_quadratic_range_l3527_352729

theorem real_roots_quadratic_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_range_l3527_352729


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3527_352739

theorem probability_of_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = 5)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls : ℚ) / total_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l3527_352739


namespace NUMINAMATH_CALUDE_second_quadrant_m_range_l3527_352784

theorem second_quadrant_m_range (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (4 + Complex.I) - 6 * Complex.I
  (z.re < 0 ∧ z.im > 0) → (3 < m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_m_range_l3527_352784


namespace NUMINAMATH_CALUDE_bob_pennies_l3527_352706

theorem bob_pennies (a b : ℕ) : 
  (b + 2 = 4 * (a - 2)) →
  (b - 2 = 3 * (a + 2)) →
  b = 62 :=
by sorry

end NUMINAMATH_CALUDE_bob_pennies_l3527_352706


namespace NUMINAMATH_CALUDE_xyz_equals_five_l3527_352783

theorem xyz_equals_five (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 5 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l3527_352783


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l3527_352707

theorem smaller_root_of_quadratic (x : ℝ) : 
  (x + 1) * (x - 1) = 0 → x = -1 ∨ x = 1 → -1 ≤ 1 → -1 = min x (-x) := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l3527_352707


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l3527_352743

/-- Calculates the number of male athletes in a stratified sample -/
def maleAthletesInSample (totalMale : ℕ) (totalFemale : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalMale * sampleSize) / (totalMale + totalFemale)

/-- Theorem: In a stratified sample of 14 athletes drawn from a population of 32 male and 24 female athletes, the number of male athletes in the sample is 8 -/
theorem stratified_sample_male_count :
  maleAthletesInSample 32 24 14 = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_male_count_l3527_352743


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l3527_352773

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 27 ∧ x - y = 9 → x * y = 162 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l3527_352773


namespace NUMINAMATH_CALUDE_complex_power_equivalence_l3527_352753

theorem complex_power_equivalence :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^28 =
  Complex.exp (Complex.I * Real.pi * (140 / 180)) :=
by sorry

end NUMINAMATH_CALUDE_complex_power_equivalence_l3527_352753


namespace NUMINAMATH_CALUDE_tetrahedron_circumsphere_area_l3527_352798

/-- The surface area of a circumscribed sphere of a regular tetrahedron with side length 2 -/
theorem tetrahedron_circumsphere_area : 
  let side_length : ℝ := 2
  let circumradius : ℝ := side_length * Real.sqrt 3 / 3
  let sphere_area : ℝ := 4 * Real.pi * circumradius^2
  sphere_area = 16 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumsphere_area_l3527_352798


namespace NUMINAMATH_CALUDE_union_equality_implies_m_values_l3527_352724

def A : Set ℝ := {-1, 2}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem union_equality_implies_m_values (m : ℝ) :
  A ∪ B m = A → m = 0 ∨ m = 1 ∨ m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_values_l3527_352724


namespace NUMINAMATH_CALUDE_room_length_calculation_l3527_352723

/-- Given a rectangular room with width 4 meters and a floor paving cost resulting in a total cost of 18700, prove that the length of the room is 5.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) : 
  width = 4 →
  cost_per_sqm = 850 →
  total_cost = 18700 →
  total_cost = cost_per_sqm * (length * width) →
  length = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3527_352723


namespace NUMINAMATH_CALUDE_ace_of_hearts_probability_l3527_352720

def standard_deck : ℕ := 52
def jokers : ℕ := 2
def total_cards : ℕ := standard_deck + jokers
def ace_of_hearts : ℕ := 1

theorem ace_of_hearts_probability :
  (ace_of_hearts : ℚ) / total_cards = 1 / 54 :=
sorry

end NUMINAMATH_CALUDE_ace_of_hearts_probability_l3527_352720


namespace NUMINAMATH_CALUDE_pastry_sale_revenue_l3527_352710

/-- Calculates the total money made from selling discounted pastries -/
theorem pastry_sale_revenue 
  (original_cupcake_price original_cookie_price : ℚ)
  (cupcakes_sold cookies_sold : ℕ)
  (h1 : original_cupcake_price = 3)
  (h2 : original_cookie_price = 2)
  (h3 : cupcakes_sold = 16)
  (h4 : cookies_sold = 8) :
  (cupcakes_sold : ℚ) * (original_cupcake_price / 2) + 
  (cookies_sold : ℚ) * (original_cookie_price / 2) = 32 := by
sorry


end NUMINAMATH_CALUDE_pastry_sale_revenue_l3527_352710


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l3527_352763

/-- Given two points P and Q that are symmetric about a line l, prove that the equation of l is x - y + 1 = 0 --/
theorem symmetric_points_line_equation (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (b - 1, a + 1)
  let l : Set (ℝ × ℝ) := {(x, y) | x - y + 1 = 0}
  (∀ (M : ℝ × ℝ), M ∈ l ↔ (dist M P)^2 = (dist M Q)^2) →
  l = {(x, y) | x - y + 1 = 0} :=
by sorry


end NUMINAMATH_CALUDE_symmetric_points_line_equation_l3527_352763


namespace NUMINAMATH_CALUDE_area_of_union_equals_20_5_l3527_352790

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point about the line y = x -/
def reflect (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Calculates the area of a triangle given its three vertices using the shoelace formula -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- The main theorem stating the area of the union of the original and reflected triangles -/
theorem area_of_union_equals_20_5 :
  let A : Point := { x := 3, y := 4 }
  let B : Point := { x := 5, y := -2 }
  let C : Point := { x := 7, y := 3 }
  let A' := reflect A
  let B' := reflect B
  let C' := reflect C
  triangleArea A B C + triangleArea A' B' C' = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_union_equals_20_5_l3527_352790


namespace NUMINAMATH_CALUDE_minimum_cost_for_planting_l3527_352793

/-- Represents a flower type with its survival rate and seed pack options -/
structure FlowerType where
  name : String
  survivalRate : Rat
  pack1 : Nat × Rat  -- (seeds, price)
  pack2 : Nat × Rat  -- (seeds, price)

/-- Calculates the minimum number of seeds needed for a given number of surviving flowers -/
def seedsNeeded (survivingFlowers : Nat) (survivalRate : Rat) : Nat :=
  Nat.ceil (survivingFlowers / survivalRate)

/-- Calculates the cost of buying seeds for a flower type -/
def costForFlowerType (ft : FlowerType) (survivingFlowers : Nat) : Rat :=
  let seedsNeeded := seedsNeeded survivingFlowers ft.survivalRate
  if seedsNeeded ≤ ft.pack1.1 then ft.pack1.2 else ft.pack2.2

/-- Applies the discount to the total cost -/
def applyDiscount (totalCost : Rat) : Rat :=
  totalCost * (1 - 1/5)  -- 20% discount

/-- The main theorem -/
theorem minimum_cost_for_planting (roses daisies sunflowers : FlowerType) :
  roses.name = "Roses" →
  roses.survivalRate = 2/5 →
  roses.pack1 = (15, 5) →
  roses.pack2 = (40, 10) →
  daisies.name = "Daisies" →
  daisies.survivalRate = 3/5 →
  daisies.pack1 = (20, 4) →
  daisies.pack2 = (50, 9) →
  sunflowers.name = "Sunflowers" →
  sunflowers.survivalRate = 1/2 →
  sunflowers.pack1 = (10, 3) →
  sunflowers.pack2 = (30, 7) →
  let totalFlowers := 20
  let flowersPerType := totalFlowers / 3
  let totalCost := costForFlowerType roses flowersPerType +
                   costForFlowerType daisies flowersPerType +
                   costForFlowerType sunflowers (totalFlowers - 2 * flowersPerType)
  applyDiscount totalCost = 84/5 := by
  sorry


end NUMINAMATH_CALUDE_minimum_cost_for_planting_l3527_352793


namespace NUMINAMATH_CALUDE_mrs_franklin_valentines_l3527_352738

theorem mrs_franklin_valentines (given_away : ℕ) (left : ℕ) 
  (h1 : given_away = 42) (h2 : left = 16) : 
  given_away + left = 58 := by
  sorry

end NUMINAMATH_CALUDE_mrs_franklin_valentines_l3527_352738


namespace NUMINAMATH_CALUDE_folded_rope_length_l3527_352719

/-- Represents the length of a rope folded three times -/
structure FoldedRope where
  total_length : ℝ
  distance_1_3 : ℝ

/-- The properties of a rope folded three times as described in the problem -/
def is_valid_folded_rope (rope : FoldedRope) : Prop :=
  rope.distance_1_3 = rope.total_length / 4

/-- The main theorem stating the relationship between the distance between points (1) and (3)
    and the total length of the rope -/
theorem folded_rope_length (rope : FoldedRope) 
  (h : is_valid_folded_rope rope) 
  (h_distance : rope.distance_1_3 = 30) : 
  rope.total_length = 120 := by
  sorry

#check folded_rope_length

end NUMINAMATH_CALUDE_folded_rope_length_l3527_352719


namespace NUMINAMATH_CALUDE_exponent_difference_l3527_352737

theorem exponent_difference (a m n : ℝ) (h1 : a^m = 12) (h2 : a^n = 3) : a^(m-n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_difference_l3527_352737


namespace NUMINAMATH_CALUDE_power_division_rule_l3527_352768

theorem power_division_rule (a : ℝ) : a^7 / a^5 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3527_352768


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l3527_352733

theorem smallest_integer_gcd_18_is_6 : 
  ∃ (n : ℕ), n > 100 ∧ Nat.gcd n 18 = 6 ∧ ∀ m, m > 100 ∧ m < n → Nat.gcd m 18 ≠ 6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l3527_352733


namespace NUMINAMATH_CALUDE_quadratic_with_zero_root_l3527_352772

/-- Given a quadratic equation (k-2)x^2 + x + k^2 - 4 = 0 where 0 is one of its roots,
    prove that k = -2 -/
theorem quadratic_with_zero_root (k : ℝ) : 
  (∀ x : ℝ, (k - 2) * x^2 + x + k^2 - 4 = 0 ↔ x = 0 ∨ x = (k^2 - 4) / (2 - k)) →
  ((k - 2) * 0^2 + 0 + k^2 - 4 = 0) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_zero_root_l3527_352772


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l3527_352774

def I : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {-2, 0, 1}
def B : Set ℤ := {-1, 0, 1, 2}

theorem complement_intersection_problem : (I \ A) ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l3527_352774


namespace NUMINAMATH_CALUDE_ab_inequality_relationship_l3527_352732

theorem ab_inequality_relationship (a b : ℝ) :
  (∀ a b, a < b ∧ b < 0 → a * b * (a - b) < 0) ∧
  (∃ a b, a * b * (a - b) < 0 ∧ ¬(a < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ab_inequality_relationship_l3527_352732


namespace NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l3527_352726

theorem infinite_primes_quadratic_equation :
  ∀ (S : Finset Nat), ∃ (p : Nat) (x y : Int),
    Prime p ∧ p ∉ S ∧ x^2 + x + 1 = p * y := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l3527_352726


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l3527_352731

/-- Given an ellipse with equation x^2 + 9y^2 = 144, the distance between its foci is 16√2 -/
theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), x^2 + 9*y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (16 * Real.sqrt 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_focal_distance_l3527_352731


namespace NUMINAMATH_CALUDE_housing_boom_construction_l3527_352782

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_construction :
  houses_built = 574 :=
by sorry

end NUMINAMATH_CALUDE_housing_boom_construction_l3527_352782


namespace NUMINAMATH_CALUDE_final_position_16_meters_l3527_352748

/-- Represents a back-and-forth race between two runners -/
structure Race where
  distance : ℝ  -- Total distance of the race (one way)
  meetPoint : ℝ  -- Distance from B to meeting point
  catchPoint : ℝ  -- Distance from finish when B catches A

/-- Calculates the final position of runner A when B finishes the race -/
def finalPositionA (race : Race) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the final position of A in the given race scenario -/
theorem final_position_16_meters (race : Race) 
  (h1 : race.meetPoint = 24)
  (h2 : race.catchPoint = 48) :
  finalPositionA race = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_position_16_meters_l3527_352748


namespace NUMINAMATH_CALUDE_union_M_N_l3527_352769

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < -1}
def N : Set ℝ := {x | (1/2 : ℝ)^x ≤ 4}

-- State the theorem
theorem union_M_N : M ∪ N = {x | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l3527_352769


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3527_352795

theorem pie_eating_contest (first_student second_student : ℚ) : 
  first_student = 8/9 → second_student = 5/6 → first_student - second_student = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3527_352795


namespace NUMINAMATH_CALUDE_fraction_simplification_l3527_352704

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3527_352704


namespace NUMINAMATH_CALUDE_sinusoidal_function_parameters_l3527_352752

/-- 
Given a sinusoidal function y = a * sin(b * x + φ) where a > 0 and b > 0,
if the maximum value is 3 and the period is 2π/4, then a = 3 and b = 4.
-/
theorem sinusoidal_function_parameters 
  (a b φ : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∀ x, a * Real.sin (b * x + φ) ≤ 3)
  (h4 : ∃ x, a * Real.sin (b * x + φ) = 3)
  (h5 : (2 * Real.pi) / b = Real.pi / 2) : 
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_parameters_l3527_352752


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3527_352765

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3527_352765


namespace NUMINAMATH_CALUDE_inequality_solution_l3527_352785

theorem inequality_solution (x : ℝ) :
  x ≤ 4 ∧ |2*x - 3| + |x + 1| < 7 → -5/3 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3527_352785


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3527_352788

/-- The number of different fruit baskets that can be created -/
def num_fruit_baskets (num_apples : ℕ) (num_oranges : ℕ) : ℕ :=
  num_apples * num_oranges

/-- Theorem stating that the number of different fruit baskets with 7 apples and 12 oranges is 84 -/
theorem fruit_basket_count :
  num_fruit_baskets 7 12 = 84 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3527_352788


namespace NUMINAMATH_CALUDE_survey_total_students_l3527_352786

theorem survey_total_students :
  let mac_preference : ℕ := 60
  let both_preference : ℕ := mac_preference / 3
  let no_preference : ℕ := 90
  let windows_preference : ℕ := 40
  mac_preference + both_preference + no_preference + windows_preference = 210 := by
sorry

end NUMINAMATH_CALUDE_survey_total_students_l3527_352786


namespace NUMINAMATH_CALUDE_hexagon_triangles_l3527_352728

/-- The number of triangles that can be formed from a regular hexagon and its center -/
def num_triangles_hexagon : ℕ :=
  let total_points : ℕ := 7
  let total_combinations : ℕ := Nat.choose total_points 3
  let invalid_triangles : ℕ := 3
  total_combinations - invalid_triangles

theorem hexagon_triangles :
  num_triangles_hexagon = 32 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangles_l3527_352728


namespace NUMINAMATH_CALUDE_consecutive_primes_expression_l3527_352736

theorem consecutive_primes_expression (p q : ℕ) : 
  Prime p → Prime q → p < q → p.succ = q → (p : ℚ) / q = 4 / 5 → 
  25 / 7 + ((2 * q - p) : ℚ) / (2 * q + p) = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_primes_expression_l3527_352736


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3527_352711

/-- Given a line with equation x*sin(α) + y + 2 = 0, 
    the range of the inclination angle α is [0, π/4] ∪ [3π/4, π) -/
theorem inclination_angle_range (x y : ℝ) (α : ℝ) :
  (x * Real.sin α + y + 2 = 0) →
  α ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3527_352711


namespace NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_statement_d_all_statements_correct_l3527_352796

-- Statement A
theorem statement_a (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  sorry

-- Statement B
theorem statement_b (a b : ℝ) : a + b = 0 → a^3 + b^3 = 0 := by
  sorry

-- Statement C
theorem statement_c (a b : ℝ) : a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/a > 1/b := by
  sorry

-- Statement D
theorem statement_d (a : ℝ) : -1 < a ∧ a < 0 → a^3 < a^5 := by
  sorry

-- All statements are correct
theorem all_statements_correct : 
  (∀ a b : ℝ, a^2 = b^2 → |a| = |b|) ∧
  (∀ a b : ℝ, a + b = 0 → a^3 + b^3 = 0) ∧
  (∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/a > 1/b) ∧
  (∀ a : ℝ, -1 < a ∧ a < 0 → a^3 < a^5) := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_statement_d_all_statements_correct_l3527_352796


namespace NUMINAMATH_CALUDE_students_travel_speed_l3527_352777

/-- Proves that given the conditions of the problem, student B's bicycle speed is 14.4 km/h -/
theorem students_travel_speed (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ)
  (h_distance : distance = 2.4)
  (h_speed_ratio : speed_ratio = 4)
  (h_time_difference : time_difference = 0.5) :
  let walking_speed := distance / (distance / (speed_ratio * walking_speed) + time_difference)
  speed_ratio * walking_speed = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_students_travel_speed_l3527_352777


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3527_352741

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- State the theorem
theorem quadratic_inequality_range :
  ∀ a : ℝ, (¬ ∃ x : ℝ, f a x ≤ 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3527_352741


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l3527_352799

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 3774 → 
  min a b = 51 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l3527_352799


namespace NUMINAMATH_CALUDE_darren_boxes_correct_l3527_352754

/-- The number of crackers in each box -/
def crackers_per_box : ℕ := 24

/-- The total number of crackers bought by both Darren and Calvin -/
def total_crackers : ℕ := 264

/-- The number of boxes Darren bought -/
def darren_boxes : ℕ := 4

theorem darren_boxes_correct :
  ∃ (calvin_boxes : ℕ),
    calvin_boxes = 2 * darren_boxes - 1 ∧
    crackers_per_box * (darren_boxes + calvin_boxes) = total_crackers :=
by sorry

end NUMINAMATH_CALUDE_darren_boxes_correct_l3527_352754


namespace NUMINAMATH_CALUDE_eggs_per_box_l3527_352717

theorem eggs_per_box (total_eggs : ℝ) (num_boxes : ℝ) 
  (h1 : total_eggs = 3.0) 
  (h2 : num_boxes = 2.0) 
  (h3 : num_boxes ≠ 0) : 
  total_eggs / num_boxes = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3527_352717


namespace NUMINAMATH_CALUDE_factorization_condition_l3527_352713

-- Define the polynomial
def polynomial (m : ℤ) (x y : ℤ) : ℤ := x^2 + 5*x*y + 2*x + m*y - 2*m

-- Define what it means for a polynomial to have two linear factors with integer coefficients
def has_two_linear_factors (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), 
    ∀ (x y : ℤ), polynomial m x y = (a*x + b*y + c) * (d*x + e*y + f)

-- State the theorem
theorem factorization_condition (m : ℤ) : 
  has_two_linear_factors m ↔ (m = 0 ∨ m = 10) := by sorry

end NUMINAMATH_CALUDE_factorization_condition_l3527_352713


namespace NUMINAMATH_CALUDE_distribution_schemes_l3527_352771

/-- The number of ways to distribute students among projects -/
def distribute_students (n_students : ℕ) (n_projects : ℕ) : ℕ :=
  -- Number of ways to choose 2 students from n_students
  (n_students.choose 2) * 
  -- Number of ways to permute n_projects
  (n_projects.factorial)

/-- Theorem stating the number of distribution schemes -/
theorem distribution_schemes :
  distribute_students 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_distribution_schemes_l3527_352771


namespace NUMINAMATH_CALUDE_geometric_sum_5_quarters_l3527_352776

def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

theorem geometric_sum_5_quarters : 
  geometric_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_5_quarters_l3527_352776


namespace NUMINAMATH_CALUDE_window_treatment_cost_l3527_352797

/-- The number of windows that need treatment -/
def num_windows : ℕ := 3

/-- The cost of a pair of sheers in dollars -/
def sheer_cost : ℚ := 40

/-- The cost of a pair of drapes in dollars -/
def drape_cost : ℚ := 60

/-- The total cost of window treatments for all windows -/
def total_cost : ℚ := num_windows * (sheer_cost + drape_cost)

theorem window_treatment_cost : total_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_window_treatment_cost_l3527_352797


namespace NUMINAMATH_CALUDE_trajectory_of_C_l3527_352744

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  let A := (-2, 0)
  let B := (2, 0)
  let perimeter := dist A C + dist B C + dist A B
  perimeter = 10

-- Define the equation of the trajectory
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1 ∧ y ≠ 0

-- Theorem statement
theorem trajectory_of_C :
  ∀ C : ℝ × ℝ, triangle_ABC C → 
  ∃ x y : ℝ, C = (x, y) ∧ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_C_l3527_352744


namespace NUMINAMATH_CALUDE_sqrt_23_parts_x_minus_y_value_l3527_352718

-- Part 1: Integer and decimal parts of √23
theorem sqrt_23_parts :
  ∃ (n : ℕ) (d : ℝ), n = 4 ∧ d = Real.sqrt 23 - 4 ∧
  Real.sqrt 23 = n + d ∧ 0 ≤ d ∧ d < 1 := by sorry

-- Part 2: x-y given 9+√3=x+y
theorem x_minus_y_value (x : ℤ) (y : ℝ) 
  (h1 : 9 + Real.sqrt 3 = x + y)
  (h2 : 0 < y) (h3 : y < 1) :
  x - y = 11 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_23_parts_x_minus_y_value_l3527_352718


namespace NUMINAMATH_CALUDE_correct_propositions_l3527_352725

-- Define the proposition from statement ②
def proposition_2 : Prop := 
  (∃ x : ℝ, x^2 + 1 > 3*x) ↔ ¬(∀ x : ℝ, x^2 + 1 ≤ 3*x)

-- Define the proposition from statement ③
def proposition_3 : Prop :=
  (∃ x : ℝ, x^2 - 3*x - 4 = 0 ∧ x ≠ 4) ∧ 
  (∀ x : ℝ, x = 4 → x^2 - 3*x - 4 = 0)

theorem correct_propositions : proposition_2 ∧ proposition_3 := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l3527_352725


namespace NUMINAMATH_CALUDE_school_year_work_hours_l3527_352705

/-- Amy's work schedule and earnings -/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_year_weeks : ℕ
  school_year_target_earnings : ℕ

/-- Calculate the required hours per week during school year -/
def required_school_year_hours_per_week (schedule : WorkSchedule) : ℚ :=
  let hourly_wage : ℚ := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let total_hours_needed : ℚ := schedule.school_year_target_earnings / hourly_wage
  total_hours_needed / schedule.school_year_weeks

/-- Theorem stating the required hours per week during school year -/
theorem school_year_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.summer_weeks = 8)
  (h2 : schedule.summer_hours_per_week = 40)
  (h3 : schedule.summer_earnings = 3200)
  (h4 : schedule.school_year_weeks = 32)
  (h5 : schedule.school_year_target_earnings = 4000) :
  required_school_year_hours_per_week schedule = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_school_year_work_hours_l3527_352705


namespace NUMINAMATH_CALUDE_integer_solutions_count_l3527_352712

theorem integer_solutions_count :
  let f : ℤ → ℤ → ℤ := λ x y => 6 * y^2 + 3 * x * y + x + 2 * y - 72
  ∃! s : Finset (ℤ × ℤ), (∀ (x y : ℤ), (x, y) ∈ s ↔ f x y = 0) ∧ Finset.card s = 4 :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l3527_352712


namespace NUMINAMATH_CALUDE_f_e_plus_f_prime_e_l3527_352755

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem f_e_plus_f_prime_e : f (Real.exp 1) + (deriv f) (Real.exp 1) = 2 * Real.exp (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_f_e_plus_f_prime_e_l3527_352755
